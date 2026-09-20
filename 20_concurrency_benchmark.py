"""단계 1 — 동시성 측정 (VLLM_SERVING_DESIGN.md §6.2·§6.3)

무엇을 답하는가
  §7 최종 선정은 "문의가 몰릴 때"를 근거로 들었으면서 실제로는 순차 지연(n=1)만
  쟀다. 이 스크립트가 그 공백을 닫는다.

    1. RTX 5060 8,151 MiB 에서 동시 요청을 얼마나 받아내는가
    2. **게이트 `P95 ≤ 6초` 는 동시성 몇까지 유지되는가**

두 가지 부하를 쓴다
  닫힌 루프(--sweep)   N개 워커가 응답을 받으면 곧바로 다음을 보낸다.
                       동시성이 정확히 N 으로 유지되므로 "동시성 N 에서의 P95"라는
                       문장이 성립한다. 무릎 지점을 찾는 데 쓴다.
  개방 루프(--open-loop) 도착률 λ 로 포아송 도착. 실제 문의는 독립적으로 도착하므로
                       **게이트 판정은 이쪽으로 한다.** 대기열이 발산하면 그 λ 는
                       용량 초과이고, 발산 자체가 결과다.

R5 를 코드로 강제한다
  prefix caching 상태를 밝히지 않은 처리량 수치는 폐기한다 — 그래서 `--prefix-caching`
  을 필수 인자로 둔다. 54문항이 **같은 템플릿**을 쓰므로(부록 G) 공통 접두가 길고,
  자동 prefix caching 이 켜져 있으면 처리량이 크게 부풀 수 있다.

실행:
    uv run python 20_concurrency_benchmark.py --sweep --configs A --prefix-caching off
    uv run python 20_concurrency_benchmark.py --open-loop --configs C --prefix-caching on
    uv run python 20_concurrency_benchmark.py --sweep --dry-run --prefix-caching off
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from backends import CONFIGS, build_backend, SAMPLER_PROFILES
from env_snapshot import snapshot as env_snapshot, save as env_save, \
    row_fields as env_row_fields, describe as env_describe

BASE_DIR = Path(__file__).resolve().parent
GOLD_FILE = BASE_DIR / "data" / "02_prompts_and_eval.json"
OUT_DIR = BASE_DIR / "work" / "20_concurrency"

DEFAULT_MODEL = "llama3.2:3b"
LEVELS = [1, 2, 4, 8, 16]
PASS_COUNT = 2
REQUESTS_PER_LEVEL = 108          # 54문항 × 2 — P95 를 말할 수 있는 최소 규모
WARMUP_SEC = int(os.environ.get("CONC_WARMUP_SEC", "30"))
COOLDOWN_SEC = int(os.environ.get("CONC_COOLDOWN_SEC", "30"))
OPEN_LOOP_SEC = 300
GATE_P95_SEC = 6.0                # §1.2 게이트. 결과를 보고 고치지 않는다 (R8)
POLL_SEC = 1.0

# 무릎 판정 — 직전 수준 대비 처리량이 이만큼도 안 늘면 포화로 본다
KNEE_GAIN = 0.10
# 클라이언트 병목 의심 — 처리량이 안 느는데 서버 대기열도 비어 있으면 (R6)
CLIENT_BOTTLENECK_GAIN = 0.05


# ══════════════════════════════════════════════════════════════════
# 입력 · 계측
# ══════════════════════════════════════════════════════════════════
def load_prompts() -> tuple[list[dict], str]:
    with open(GOLD_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return ([{"id": q["id"], "prompt": q["prompt"]} for q in data["questions"]],
            data.get("rubric_version"))


def pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return s[max(0, min(len(s) - 1, int(math.ceil(p * len(s))) - 1))]


class Poller(threading.Thread):
    """GPU 와 vLLM 서버 상태를 1초 간격으로 긁는다.

    vLLM 의 대기열 길이는 R6(클라이언트 병목 판별)에 반드시 필요하다.
    Ollama 에는 대응 지표가 없으므로 그 경우 GPU 만 남고, 판별은 약해진다.
    """

    def __init__(self, metrics_url: str | None, label: str):
        super().__init__(daemon=True)
        self.metrics_url = metrics_url
        self.label = label
        self.stop_flag = threading.Event()
        self.gpu_rows: list[dict] = []
        self.srv_rows: list[dict] = []
        self._sess = None
        if metrics_url:
            import requests
            self._sess = requests.Session()

    @staticmethod
    def _gpu() -> dict | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            if out.returncode != 0:
                return None
            used, total, util = [x.strip() for x in out.stdout.strip().split("\n")[0].split(",")]
            return {"vram_used_mib": int(used), "vram_total_mib": int(total), "gpu_util_pct": int(util)}
        except Exception:
            return None

    def _metrics(self) -> dict | None:
        if not self._sess:
            return None
        try:
            r = self._sess.get(self.metrics_url, timeout=3)
            if r.status_code != 200:
                return None
            want = {"vllm:num_requests_running": "running",
                    "vllm:num_requests_waiting": "waiting",
                    "vllm:gpu_cache_usage_perc": "kv_usage"}
            got: dict = {}
            for line in r.text.splitlines():
                if line.startswith("#"):
                    continue
                m = re.match(r"([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+]+)", line)
                if not m:
                    continue
                key = want.get(m.group(1))
                if key:
                    got[key] = got.get(key, 0.0) + float(m.group(3))
            return got or None
        except Exception:
            return None

    def run(self) -> None:
        while not self.stop_flag.is_set():
            t = time.time()
            g = self._gpu()
            if g:
                self.gpu_rows.append({"ts": t, "label": self.label, **g})
            s = self._metrics()
            if s:
                self.srv_rows.append({"ts": t, "label": self.label, **s})
            self.stop_flag.wait(POLL_SEC)

    def stop(self) -> None:
        self.stop_flag.set()
        self.join(timeout=5)

    def window(self, t0: float, t1: float) -> dict:
        g = [r for r in self.gpu_rows if t0 <= r["ts"] <= t1]
        s = [r for r in self.srv_rows if t0 <= r["ts"] <= t1]
        return {
            "vram_peak_mib": max((r["vram_used_mib"] for r in g), default=None),
            "gpu_util_mean": (sum(r["gpu_util_pct"] for r in g) / len(g)) if g else None,
            "waiting_max": max((r.get("waiting", 0) for r in s), default=None),
            "running_max": max((r.get("running", 0) for r in s), default=None),
            "kv_usage_max": max((r.get("kv_usage", 0) for r in s), default=None),
            "has_server_metrics": bool(s),
        }


class RowSink:
    """요청이 끝날 때마다 CSV 에 바로 쓴다.

    한 시간짜리 실행을 끝에 한 번 저장하면, 중간에 끊는 순간 전부 잃는다.
    실제로 그렇게 만들어 놨다가 고쳤다 — 측정한 것은 측정한 즉시 남긴다.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = None
        self._w = None
        self._lock = threading.Lock()
        self.n = 0

    def write(self, row: dict) -> None:
        with self._lock:
            if self._w is None:
                self._f = open(self.path, "w", encoding="utf-8-sig", newline="")
                self._w = csv.DictWriter(self._f, fieldnames=list(row.keys()),
                                         extrasaction="ignore")
                self._w.writeheader()
            self._w.writerow(row)
            self.n += 1
            if self.n % 20 == 0:
                self._f.flush()

    def close(self) -> None:
        with self._lock:
            if self._f:
                self._f.flush()
                self._f.close()
                self._f = None


class Progress:
    """블록이 도는 동안 몇 건 끝났는지, 남은 시간이 얼마인지 보여 준다."""

    def __init__(self, total: int, label: str, every: float = 10.0):
        self.total, self.label, self.every = total, label, every
        self.done = 0
        self.t0 = time.time()
        self._lock = threading.Lock()
        self._last = 0.0

    def tick(self) -> None:
        with self._lock:
            self.done += 1
            now = time.time()
            if now - self._last < self.every and self.done < self.total:
                return
            self._last = now
            el = now - self.t0
            rate = self.done / el if el > 0 else 0
            eta = (self.total - self.done) / rate if rate > 0 else 0
            print(f"\r    {self.label} {self.done}/{self.total} "
                  f"({self.done / self.total * 100:4.0f}%) "
                  f"{rate:.2f} req/s  남은 시간 ~{eta / 60:.1f}분   ",
                  end="", flush=True)


# ══════════════════════════════════════════════════════════════════
# 닫힌 루프
# ══════════════════════════════════════════════════════════════════
def closed_loop(key, model, sampler, url, prompts, n_workers, n_requests, poller,
                deadline_sec: float | None = None, sink=None, progress=None,
                extra: dict | None = None):
    """워커 n개가 응답을 받는 즉시 다음을 보낸다 → 동시성이 정확히 n으로 유지된다.

    `deadline_sec` 를 주면 그 시간이 지날 때까지 계속 돈다(워밍업용).
    요청 수로 워밍업하면 엔진이 빠를수록 덜 데워진 채로 측정에 들어간다.
    """
    # Session 은 스레드 안전이 아니다. 워커마다 자기 백엔드를 준다.
    workers = [build_backend(key, model, sampler, url) for _ in range(n_workers)]
    counter = threading.Lock()
    state = {"i": 0}
    rows: list[dict] = []
    rows_lock = threading.Lock()

    t_begin = time.time()

    def one(w_idx: int):
        be = workers[w_idx]
        while True:
            with counter:
                i = state["i"]
                if deadline_sec is not None:
                    if time.time() - t_begin >= deadline_sec:
                        return
                elif i >= n_requests:
                    return
                state["i"] = i + 1
            q = prompts[i % len(prompts)]
            t_start = time.time()
            res = be.generate(q["prompt"])
            row = {
                "config": key, "phase": "closed", "concurrency": n_workers,
                "req_index": i, "question_id": q["id"], "worker": w_idx,
                "t_start": t_start, "t_end": time.time(),
                "total_time_sec": res.total_time_sec, "ttft_sec": res.ttft_sec,
                "prompt_tokens": res.prompt_tokens,
                "generated_tokens": res.generated_tokens,
                "tps_wall": res.tps_wall,
                "is_success": res.success, "error_message": res.error_message,
                **(extra or {}),
            }
            with rows_lock:
                rows.append(row)
            if sink:
                sink.write(row)          # 측정 즉시 디스크로
            if progress:
                progress.tick()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(one, range(n_workers)))
    t1 = time.time()
    for w in workers:
        w.close()
    return rows, t0, t1


# ══════════════════════════════════════════════════════════════════
# 개방 루프
# ══════════════════════════════════════════════════════════════════
def open_loop(key, model, sampler, url, prompts, lam, duration, poller, cap=64,
              sink=None, extra=None):
    """도착률 λ(요청/초) 로 포아송 도착. 서버가 못 따라가면 동시성이 저절로 늘어난다.

    닫힌 루프와 달리 **대기열이 발산할 수 있다.** 발산하는지 보는 것이 목적이므로
    막지 않는다. 다만 클라이언트가 먼저 죽지 않도록 동시 실행 상한(cap)만 둔다.
    """
    rows: list[dict] = []
    rows_lock = threading.Lock()
    sem = threading.Semaphore(cap)
    threads: list[threading.Thread] = []
    dropped = {"n": 0}
    rng = random.Random(20260918)
    t0 = time.time()
    i = 0

    def fire(idx: int, q: dict):
        be = build_backend(key, model, sampler, url)
        t_start = time.time()
        try:
            res = be.generate(q["prompt"])
            row = {
                    "config": key, "phase": "open", "concurrency": None,
                    "arrival_rate": lam, "req_index": idx, "question_id": q["id"],
                    "worker": None, "t_start": t_start, "t_end": time.time(),
                    "total_time_sec": res.total_time_sec, "ttft_sec": res.ttft_sec,
                    "prompt_tokens": res.prompt_tokens,
                    "generated_tokens": res.generated_tokens,
                    "tps_wall": res.tps_wall,
                    "is_success": res.success, "error_message": res.error_message,
                    **(extra or {}),
            }
            with rows_lock:
                rows.append(row)
            if sink:
                sink.write(row)
        finally:
            be.close()
            sem.release()

    while time.time() - t0 < duration:
        time.sleep(rng.expovariate(lam))
        if not sem.acquire(blocking=False):
            dropped["n"] += 1          # 클라이언트 상한에 걸림 — 기록해야 한다
            continue
        q = prompts[i % len(prompts)]
        th = threading.Thread(target=fire, args=(i, q), daemon=True)
        th.start()
        threads.append(th)
        i += 1

    for th in threads:
        th.join(timeout=300)
    return rows, t0, time.time(), dropped["n"]


# ══════════════════════════════════════════════════════════════════
# 집계
# ══════════════════════════════════════════════════════════════════
def summarize(rows, t0, t1, win) -> dict:
    ok = [r for r in rows if r["is_success"]]
    lat = [r["total_time_sec"] for r in ok]
    dur = max(t1 - t0, 1e-9)
    return {
        "n": len(rows), "ok": len(ok), "fail": len(rows) - len(ok),
        "wall_sec": dur,
        "p50": pct(lat, 0.50), "p95": pct(lat, 0.95), "p99": pct(lat, 0.99),
        "mean_lat": statistics.fmean(lat) if lat else None,
        "mean_ttft": statistics.fmean([r["ttft_sec"] for r in ok]) if ok else None,
        "req_per_sec": len(ok) / dur,
        "tok_per_sec": sum(r["generated_tokens"] for r in ok) / dur,
        **win,
    }


def analyse_sweep(by_level: dict) -> dict:
    """게이트 유지 한계 N, 무릎 지점, 클라이언트 병목 의심 지점을 찾는다."""
    levels = sorted(by_level)
    gate_ok = [n for n in levels
               if by_level[n]["p95"] is not None and by_level[n]["p95"] <= GATE_P95_SEC]
    knee, bottleneck = None, []
    prev_tp = None
    for n in levels:
        tp = by_level[n]["req_per_sec"]
        if prev_tp:
            gain = (tp - prev_tp) / prev_tp
            if knee is None and gain < KNEE_GAIN:
                knee = n
            # R6 — 처리량이 안 느는데 서버 대기열이 비어 있으면 병목은 클라이언트다
            if gain < CLIENT_BOTTLENECK_GAIN:
                w = by_level[n].get("waiting_max")
                if by_level[n].get("has_server_metrics") and (w is None or w < 1):
                    bottleneck.append(n)
        prev_tp = tp
    return {
        "gate_limit": max(gate_ok) if gate_ok else None,
        "gate_fail_from": next((n for n in levels if n not in gate_ok), None),
        "knee": knee,
        "client_bottleneck_suspect": bottleneck,
    }


# ══════════════════════════════════════════════════════════════════
# 보고
# ══════════════════════════════════════════════════════════════════
def write_report(all_rows, sweep, open_res, meta, poller) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 요청별 원본은 RowSink 가 실시간으로 이미 기록했다 — 여기서 다시 쓰지 않는다.
    for name, rows in (("20_gpu_poll.csv", poller.gpu_rows),
                       ("20_vllm_metrics.csv", poller.srv_rows)):
        if rows:
            with open(OUT_DIR / name, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    L = []
    a = L.append
    a("# 단계 1 — 동시성 측정 결과")
    a("")
    if meta.get("env"):
        a(f"측정 환경 — {env_describe(meta['env'])}")
        a("")
    a(f"평가셋 `{meta['version']}` · 모델 `{meta['model']}` · 샘플러 `{meta['sampler']}`")
    a(f"· **prefix caching: {meta['prefix_caching']}** · 게이트 기준 `P95 ≤ {GATE_P95_SEC}초`")
    a("")
    a("> prefix caching 상태를 함께 적지 않은 처리량 수치는 폐기한다(설계서 §10 R5). 54문항이")
    a("> 같은 템플릿을 쓰므로 공통 접두가 길고, 캐시가 켜져 있으면 처리량이 부풀 수 있다.")
    a("")

    for key, by_level in sweep.items():
        if not by_level:
            continue
        v = analyse_sweep(by_level)
        a(f"## 닫힌 루프 — 구성 {key} ({CONFIGS[key]['label']})")
        a("")
        a("| N | 성공/전체 | P50 | **P95** | P99 | 평균 TTFT | 완료 req/s | 생성 tok/s | VRAM 최대 | 대기 최대 |")
        a("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for n in sorted(by_level):
            s = by_level[n]
            f2 = lambda x: f"{x:.2f}초" if x is not None else "—"
            a(f"| {n} | {s['ok']}/{s['n']} | {f2(s['p50'])} | **{f2(s['p95'])}** | {f2(s['p99'])} | "
              f"{f2(s['mean_ttft'])} | {s['req_per_sec']:.2f} | {s['tok_per_sec']:.0f} | "
              f"{s['vram_peak_mib'] or '—'} | {s['waiting_max'] if s['waiting_max'] is not None else '—'} |")
        a("")
        max_tested = max(by_level)
        if v["gate_limit"] is not None and v["gate_limit"] >= max_tested:
            # 측정한 마지막 수준까지 통과했다면 그것은 '한계'가 아니라
            # '아직 한계를 못 찾았다'는 뜻이다. 테스트 안 한 경계를 한계로 적지 않는다.
            a(f"- **측정한 최대 동시성 {max_tested} 까지 P95 ≤ {GATE_P95_SEC}초를 지켰다 — "
              f"한계를 찾지 못했다.** 더 높은 N 으로 `--levels` 를 늘려 다시 재야 한다.")
        elif v["gate_limit"] is not None:
            a(f"- **게이트 유지 한계: 동시성 {v['gate_limit']}** — 여기까지는 P95 ≤ {GATE_P95_SEC}초를 지킨다.")
        else:
            a(f"- **게이트 유지 한계: 없음** — 동시성 1에서도 P95 가 {GATE_P95_SEC}초를 넘는다.")
        if v["gate_fail_from"] is not None:
            a(f"- 게이트가 처음 깨지는 지점: 동시성 {v['gate_fail_from']}")
        a(f"- 무릎 지점: {v['knee'] if v['knee'] else '관측 범위 안에서는 포화하지 않음'}")
        if v["client_bottleneck_suspect"]:
            a(f"- **R6 경보 — 클라이언트 병목 의심: N={v['client_bottleneck_suspect']}.** "
              "처리량이 늘지 않는데 서버 대기열이 비어 있다. 이 지점 이상의 수치는 보고하지 않는다.")
        elif not by_level[max(by_level)].get("has_server_metrics"):
            a("- 서버 대기열 지표가 없어(Ollama) 클라이언트 병목 판별이 약하다. "
              "처리량 포화를 서버 한계로 단정하지 말 것.")
        a("")
        a("> P99 는 표 안에만 둔다. N당 108요청에서 P99 는 상위 1건이므로 결론에 쓰지 않는다(R7).")
        a("")

    for key, runs in open_res.items():
        if not runs:
            continue
        a(f"## 개방 루프 — 구성 {key} (포아송 도착, 게이트 판정용)")
        a("")
        a("| λ (req/s) | 도착 | 성공 | P50 | **P95** | 전반부 평균 | 후반부 평균 | 발산 | 클라 상한 초과 |")
        a("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
        for r in runs:
            f2 = lambda x: f"{x:.2f}초" if x is not None else "—"
            div = "**예 — 용량 초과**" if r["diverged"] else "아니오"
            a(f"| {r['lam']:.2f} | {r['n']} | {r['ok']} | {f2(r['p50'])} | **{f2(r['p95'])}** | "
              f"{f2(r['first_half'])} | {f2(r['second_half'])} | {div} | {r['dropped']} |")
        a("")
        for r in runs:
            verdict = "통과" if (r["p95"] is not None and r["p95"] <= GATE_P95_SEC and not r["diverged"]) else "탈락"
            a(f"- λ={r['lam']:.2f}: 게이트 **{verdict}**")
        a("")
        a("> λ 는 업무 실측이 없어 닫힌 루프의 무릎 처리량에서 유도한 **가정값**이다(설계서 §6.3).")
        a("> 실제 도착률이 확보되면 다시 재야 한다.")
        a("")

    a("## 하지 않을 말 (설계서 §12)")
    a("")
    a("- \"동시성 N까지 안전하다\" — 54문항 합성 부하다. 실제 문의는 길이 분포도 도착 패턴도 다르다.")
    a("- \"P95 가 6초 이내이므로 운영 가능하다\" — 이 측정에는 지식베이스 조회·네트워크·전후처리가")
    a("  전부 빠져 있다. **모델 추론 시간만이다.**")
    a("- \"8GB로 충분하다\" — 한 모델, 한 길이 설정의 결과다. 컨텍스트를 늘리면 KV 캐시가 늘고 동시성이 준다.")
    a("- 게이트 `P95 ≤ 6초` 는 이 결과를 보고 조정하지 않는다(R8).")

    (OUT_DIR / "20_concurrency_summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n기록: {OUT_DIR.relative_to(BASE_DIR)}/ (20_latency_raw.csv, 20_concurrency_summary.md 등)")


# ══════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="단계 1 — 동시성 측정")
    ap.add_argument("--sweep", action="store_true", help="닫힌 루프 스윕 (N=1..16)")
    ap.add_argument("--open-loop", action="store_true", help="포아송 도착 (게이트 판정용)")
    ap.add_argument("--configs", default="A")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sampler", default="fixed", choices=sorted(SAMPLER_PROFILES))
    ap.add_argument("--prefix-caching", required=True, choices=["on", "off"],
                    help="서버의 prefix caching 상태. 필수 — 밝히지 않은 처리량은 폐기한다(R5)")
    ap.add_argument("--levels", default=",".join(map(str, LEVELS)))
    ap.add_argument("--requests-per-level", type=int, default=REQUESTS_PER_LEVEL)
    ap.add_argument("--open-loop-sec", type=int, default=OPEN_LOOP_SEC)
    ap.add_argument("--lam", default=None,
                    help="개방 루프 도착률(쉼표 구분). 생략하면 스윕 무릎에서 유도")
    ap.add_argument("--dry-run", action="store_true")
    for k in CONFIGS:
        ap.add_argument(f"--url-{k.lower()}", default=None)
    args = ap.parse_args()

    if not (args.sweep or args.open_loop):
        print("--sweep 또는 --open-loop 중 하나는 주어야 합니다.")
        return 1
    configs = [c.strip().upper() for c in args.configs.split(",") if c.strip()]
    if any(c not in CONFIGS for c in configs):
        print(f"알 수 없는 구성 (가능: {', '.join(CONFIGS)})")
        return 1
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    urls = {k: getattr(args, f"url_{k.lower()}") for k in CONFIGS}

    prompts, version = load_prompts()
    env = env_snapshot()          # 부록 L — 환경을 레코드와 함께 남긴다
    meta = {"version": version, "model": args.model, "sampler": args.sampler,
            "prefix_caching": args.prefix_caching, "env": env}

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 단계 1 — 동시성 측정")
    print(f"  환경 {env_describe(env)}")
    print(f"  평가셋 {version} · 모델 {args.model} · prefix caching {args.prefix_caching}")
    print(f"  구성 {','.join(configs)} · 수준 {levels} · 수준당 {args.requests_per_level}요청")
    print(f"  게이트 기준 P95 ≤ {GATE_P95_SEC}초\n")
    if args.dry_run:
        for p in range(1, PASS_COUNT + 1):
            order = levels if p % 2 == 1 else list(reversed(levels))
            print(f"  패스 {p}: N = {order}")
        tot = len(configs) * len(levels) * args.requests_per_level * PASS_COUNT
        print(f"\n  닫힌 루프 총 요청 {tot}회")
        print("--dry-run 이므로 호출하지 않고 종료합니다.")
        return 0

    all_rows: list[dict] = []
    warmup_log: list[dict] = []
    sweep: dict = {}
    run_start = time.time()
    done_blocks = [0]
    total_blocks = len(configs) * len(levels) * PASS_COUNT if args.sweep else 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sink = RowSink(OUT_DIR / "20_latency_raw.csv")

    if args.sweep:
        n_req = total_blocks * args.requests_per_level
        print(f"  닫힌 루프 {total_blocks}블록 × {args.requests_per_level}요청 = {n_req}요청")
        print(f"  본실험 기준(요청당 ~3.4초)이면 대략 {n_req * 3.4 / 60:.0f}분 + "
              f"워밍업·쿨다운 {total_blocks * (WARMUP_SEC + COOLDOWN_SEC) / 60:.0f}분")
        print("  · 각 요청은 끝나는 즉시 20_latency_raw.csv 에 기록됩니다.")
        print("  · 중간에 Ctrl-C 로 끊어도 그때까지 측정한 것은 남습니다.")
        print("  · 짧게 보시려면 --levels 1,2,4 --requests-per-level 20 을 쓰십시오.\n")
    open_res: dict = {}
    poller = Poller(None, "init")

    for key in configs:
        url = urls.get(key) or CONFIGS[key]["base_url"]
        probe = build_backend(key, args.model, args.sampler, url)
        ok, msg = probe.health()
        probe.close()
        if not ok:
            print(f"구성 {key} 도달 실패: {msg}")
            return 1

        metrics_url = f"{url}/metrics" if CONFIGS[key]["engine"] == "vllm" else None
        poller = Poller(metrics_url, key)
        poller.start()

        if args.sweep:
            by_level: dict = {}
            acc: dict = {n: [] for n in levels}
            spans: dict = {n: [] for n in levels}
            for pass_idx in range(1, PASS_COUNT + 1):
                # 역순 패스 — 높은 N 이 늘 뜨거운 GPU 에서 측정되지 않게 (§6.2)
                order = levels if pass_idx % 2 == 1 else list(reversed(levels))
                print(f"\n=== 구성 {key} · 패스 {pass_idx}/{PASS_COUNT} · N 순서 {order} ===")
                for n in order:
                    print(f"  N={n:<3} 워밍업 {WARMUP_SEC}초…", end="", flush=True)
                    wt = time.time()
                    wrows, _, _ = closed_loop(key, args.model, args.sampler, url, prompts,
                                              n, 0, poller, deadline_sec=WARMUP_SEC)
                    w_wall = time.time() - wt
                    warmup_log.append({"config": key, "pass_index": pass_idx,
                                       "concurrency": n, "requests": len(wrows),
                                       "wall_sec": round(w_wall, 1)})
                    # 워밍업 처리량으로 이 블록에 걸릴 시간을 미리 알려 준다.
                    # 한 시간짜리 실행에서 "얼마나 남았나"를 모르는 것이 제일 나쁘다.
                    rate = len(wrows) / w_wall if w_wall > 0 else 0
                    eta_min = (args.requests_per_level / rate / 60) if rate > 0 else float("nan")
                    print(f" {len(wrows)}req/{w_wall:.0f}초 → 이 블록 예상 ~{eta_min:.1f}분")
                    prog = Progress(args.requests_per_level, f"N={n}")
                    rows, t0, t1 = closed_loop(
                        key, args.model, args.sampler, url, prompts, n,
                        args.requests_per_level, poller, sink=sink, progress=prog,
                        extra={"pass_index": pass_idx, "prefix_caching": args.prefix_caching,
                               **env_row_fields(env)})
                    acc[n].extend(rows)
                    spans[n].append((t0, t1))
                    all_rows.extend(rows)
                    s = summarize(rows, t0, t1, poller.window(t0, t1))
                    print(f"\r    N={n} 완료 — P95 {s['p95']:.2f}초  "
                          f"{s['req_per_sec']:.2f} req/s  ({(t1 - t0) / 60:.1f}분)          ")
                    done_blocks[0] += 1
                    left = total_blocks - done_blocks[0]
                    if left:
                        print(f"    남은 블록 {left}/{total_blocks}  "
                              f"(누적 {(time.time() - run_start) / 60:.0f}분)")
                        print(f"    cooldown {COOLDOWN_SEC}초")
                        time.sleep(COOLDOWN_SEC)
            for n in levels:
                if not acc[n]:
                    continue
                wall = sum(b - a for a, b in spans[n])
                t0, t1 = spans[n][0][0], spans[n][-1][1]
                s = summarize(acc[n], 0, wall, poller.window(t0, t1))
                by_level[n] = s
            sweep[key] = by_level

        if args.open_loop:
            # 개방 루프 행도 같은 싱크로 흘린다
            if args.lam:
                lams = [float(x) for x in args.lam.split(",")]
            else:
                base = sweep.get(key, {})
                if not base:
                    print("λ 를 유도할 스윕 결과가 없습니다. --sweep 을 함께 주거나")
                    print("--lam 으로 직접 지정하십시오.")
                    poller.stop()
                    return 1
                knee_n = analyse_sweep(base)["knee"]
                if knee_n:
                    tp = base[knee_n]["req_per_sec"]
                    src = f"무릎 N={knee_n}"
                else:
                    # 관측 범위 안에서 포화하지 않았다. 그렇다고 멈출 이유는 없고,
                    # 가장 높은 수준의 처리량을 기준으로 삼되 그 사실을 밝힌다.
                    # 진짜 용량은 이보다 클 수 있으므로 λ 는 보수적인 값이 된다.
                    knee_n = max(base)
                    tp = base[knee_n]["req_per_sec"]
                    src = f"무릎 미관측 — 최고 수준 N={knee_n}"
                    print(f"\n주의: 측정한 N 범위({sorted(base)})에서 포화가 관측되지 않았습니다.")
                    print("      λ 를 최고 수준 처리량에서 유도하므로 실제 용량보다 낮게 잡힙니다.")
                    print("      더 높은 N 까지 --levels 로 늘려 다시 재는 편이 낫습니다.")
                lams = [round(0.5 * tp, 3), round(0.8 * tp, 3)]
                print(f"λ 유도: {src} 처리량 {tp:.2f} req/s → λ {lams}")
            runs = []
            for lam in lams:
                print(f"\n=== 구성 {key} · 개방 루프 λ={lam:.2f} req/s · {args.open_loop_sec}초 ===")
                rows, t0, t1, dropped = open_loop(key, args.model, args.sampler, url,
                                                  prompts, lam, args.open_loop_sec, poller, sink=sink,
                                                  extra=env_row_fields(env))
                all_rows.extend(rows)
                ok_rows = sorted([r for r in rows if r["is_success"]], key=lambda r: r["t_start"])
                half = len(ok_rows) // 2
                fh = statistics.fmean([r["total_time_sec"] for r in ok_rows[:half]]) if half else None
                sh = statistics.fmean([r["total_time_sec"] for r in ok_rows[half:]]) if half else None
                # 후반부가 전반부보다 뚜렷이 느려지면 대기열이 발산하는 것이다
                diverged = bool(fh and sh and sh > fh * 1.5)
                runs.append({
                    "lam": lam, "n": len(rows), "ok": len(ok_rows), "dropped": dropped,
                    "p50": pct([r["total_time_sec"] for r in ok_rows], 0.50),
                    "p95": pct([r["total_time_sec"] for r in ok_rows], 0.95),
                    "first_half": fh, "second_half": sh, "diverged": diverged,
                })
                print(f"  도착 {len(rows)} 성공 {len(ok_rows)} "
                      f"P95 {runs[-1]['p95']:.2f}초 발산 {'예' if diverged else '아니오'}")
                time.sleep(COOLDOWN_SEC)
            open_res[key] = runs

        poller.stop()

    if warmup_log:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "20_warmups.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(warmup_log[0].keys()))
            w.writeheader()
            w.writerows(warmup_log)

    env_save(env, OUT_DIR / "20_env.json")
    write_report(all_rows, sweep, open_res, meta, poller)
    sink.close()

    print("\n" + "=" * 64)
    for key, by_level in sweep.items():
        v = analyse_sweep(by_level)
        mt = max(by_level)
        lim = (f"{mt}+ (한계 미관측)" if v["gate_limit"] == mt else v["gate_limit"])
        print(f"  구성 {key}: 게이트 유지 한계 N={lim}  무릎 N={v['knee'] or '미관측'}")
        if v["client_bottleneck_suspect"]:
            print(f"    R6 경보 — 클라이언트 병목 의심 N={v['client_bottleneck_suspect']}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # 한 시간짜리 실행을 중간에 끊었다고 해서 측정한 것까지 버릴 이유는 없다.
        print("\n\n중단됨 — 그때까지 완료된 요청은 "
              "work/20_concurrency/20_latency_raw.csv 에 남아 있습니다.")
        print("      요약 보고서는 만들지 않았습니다(부분 데이터로 게이트 판정을 하면 안 됩니다).")
        print("      이어서 보시려면 그 CSV 를 직접 집계하거나, 더 작은 --levels /")
        print("      --requests-per-level 로 다시 돌리십시오.")
        sys.exit(130)
