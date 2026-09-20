"""단계 0 — 서빙 구성 A·B·C 비교 (VLLM_SERVING_DESIGN.md §6.1)

  A = Ollama / Windows 네이티브 / GGUF Q4_K_M   ← 기존 185회와 같은 경로
  B = Ollama / WSL2            / GGUF Q4_K_M
  C = vLLM   / WSL2            / AWQ 4bit

  A vs B = 런타임(WSL2) 효과      (엔진·포맷이 같으므로 순수하다)
  B vs C = 엔진 + 포맷 효과        (런타임이 같으므로 WSL2 오버헤드가 소거된다)

이 스크립트가 말할 수 있는 것은 "구성끼리 이만큼 다르다"뿐이다.
"vLLM 이 더 빠르다/좋다"는 문장은 쓸 수 없다 — 설계서 §2.

실행:
    uv run python 19_serving_comparison.py --check-prompts     # 사전 점검만 (모델 생성 없음)
    uv run python 19_serving_comparison.py                     # 본실행
    uv run python 19_serving_comparison.py --configs A,C       # 일부 구성만
    uv run python 19_serving_comparison.py --dry-run           # 호출 없이 계획만 출력

사전 점검을 통과하기 전에는 본실행을 하지 않는다(설계서 §14).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from backends import CONFIGS, build_backend, prompt_sha256, SAMPLER_PROFILES
from env_snapshot import snapshot as env_snapshot, save_pair as env_save_pair, \
    row_fields as env_row_fields, describe as env_describe, \
    contention_note as env_contention

BASE_DIR = Path(__file__).resolve().parent
GOLD_FILE = BASE_DIR / "data" / "02_prompts_and_eval.json"
OUT_DIR = BASE_DIR / "work" / "19_serving"

DEFAULT_MODEL = "llama3.2:3b"
PASS_COUNT = 2
COOLDOWN_SEC = int(os.environ.get("SERVING_COOLDOWN_SEC", "30"))
WARMUP_PROMPT = "안녕하세요. 준비 확인용 문장입니다. 한 문장으로 답하세요."

# 패스 간 평균 지연이 이만큼 넘게 벌어지면 열 드리프트를 의심한다 (설계서 §6.1)
DRIFT_WARN_SEC = 0.3


# ══════════════════════════════════════════════════════════════════
# 입력
# ══════════════════════════════════════════════════════════════════
def load_questions() -> list[dict]:
    with open(GOLD_FILE, encoding="utf-8") as f:
        data = json.load(f)
    qs = []
    for q in data["questions"]:
        qs.append({
            "id": q["id"],
            "prompt": q["prompt"],
            "category": q.get("taxonomy", {}).get("category", ""),
            "eval_criteria": q.get("eval_criteria", ""),
        })
    return qs, data.get("rubric_version")


# ══════════════════════════════════════════════════════════════════
# 사전 점검 — 세 구성이 같은 것을 보고 있는지 (설계서 §8 C1)
# ══════════════════════════════════════════════════════════════════
def check_prompts(configs: list[str], model: str, sampler: str, urls: dict) -> int:
    """모델 생성을 하지 않고, 서버가 프롬프트를 어떻게 감싸는지만 대조한다.

    세 겹으로 본다.
      1겹 (클라이언트) 보내는 문자열의 SHA-256 — 우리 쪽 버그를 잡는다
      2겹 (서버)       서버가 센 입력 토큰 수 — 템플릿이 다르면 여기서 어긋난다
      3겹 (참고)       각 서버가 알려주는 템플릿 원문

    2겹이 어긋나면 **모델이 보는 텍스트가 다르다는 뜻**이고, 그 상태의 품질
    비교는 엔진 비교가 아니라 입력 비교가 된다. 그래서 여기서 막는다.
    """
    questions, version = load_questions()
    probe = questions[0]
    prompt = probe["prompt"]
    sha = prompt_sha256(prompt)
    # 두 번째 탐침 — 내용이 거의 없는 프롬프트.
    # 불일치가 났을 때 "템플릿이 붙이는 토큰이 다른 것"과 "토크나이저가 다른 것"을
    # 구분하려면 기준점이 하나 더 필요하다. §판정 참조.
    MIN_PROMPT = "x"

    print("=" * 70)
    print("사전 점검 — 세 구성이 같은 프롬프트를 보는가")
    print("=" * 70)
    print(f"평가셋      : {version}")
    print(f"탐침 문항   : ID {probe['id']}")
    print(f"모델        : {model}")
    print(f"샘플러      : {sampler}  {SAMPLER_PROFILES[sampler]}")
    print(f"클라이언트 프롬프트 SHA-256 : {sha[:16]}…  ({len(prompt)}자)")
    print()

    rows, token_counts, min_counts, unreachable = [], {}, {}, []

    for key in configs:
        cfg = CONFIGS[key]
        print(f"── 구성 {key}  {cfg['label']}  @ {urls.get(key) or cfg['base_url']}")
        try:
            be = build_backend(key, model, sampler, urls.get(key))
        except ValueError as exc:
            print(f"   구성 불가: {exc}\n")
            unreachable.append(key)
            continue

        ok, msg = be.health()
        if not ok:
            print(f"   서버 도달 실패: {msg}\n")
            unreachable.append(key)
            rows.append({"config": key, "engine": be.engine, "reachable": False,
                         "client_prompt_sha256": sha, "server_prompt_tokens": None,
                         "note": msg})
            continue

        # 생성은 최소화한다. 필요한 것은 prompt_tokens 뿐이다.
        saved = be.sampler["max_tokens"] if "max_tokens" in be.sampler else None
        if "max_tokens" in be.sampler:
            be.sampler["max_tokens"] = 1
        res = be.generate(prompt, timeout=120)
        res_min = be.generate(MIN_PROMPT, timeout=120)
        if saved is not None:
            be.sampler["max_tokens"] = saved

        ptok = res.prompt_tokens
        mtok = res_min.prompt_tokens
        token_counts[key] = ptok
        min_counts[key] = mtok
        content = (ptok - mtok) if (ptok is not None and mtok is not None) else None
        print(f"   서버가 센 입력 토큰 : {ptok}")
        print(f"   최소 탐침(\"x\")     : {mtok}   ← 템플릿이 붙이는 몫에 가깝다")
        if content is not None:
            print(f"   문항 본문 몫        : {content}")
        if not res.success:
            print(f"   호출 실패: {res.error_message}")

        tmpl = be.describe_template()
        if tmpl.get("template"):
            one = " ".join(str(tmpl["template"]).split())
            print(f"   템플릿(축약)        : {one[:110]}…")
        if tmpl.get("tokenize_count") is not None:
            print(f"   /tokenize 토큰 수   : {tmpl['tokenize_count']} (탐침 문자열 기준)")
        if tmpl.get("quantization"):
            print(f"   양자화              : {tmpl['quantization']}")
        if tmpl.get("error"):
            print(f"   템플릿 조회 실패    : {tmpl['error']}")
        print()

        rows.append({
            "config": key, "engine": be.engine, "reachable": True,
            "client_prompt_sha256": sha,
            "server_prompt_tokens": ptok,
            "min_probe_tokens": mtok,
            "content_tokens": content,
            "call_ok": res.success,
            "note": res.error_message or "",
            # 잘라 두면 정작 원인이 잘린 뒤쪽에 있을 때 못 본다 — 전문을 남긴다
            "template_full": " ".join(str(tmpl.get("template") or "").split()),
            "tokenize_count": tmpl.get("tokenize_count"),
            "quantization": tmpl.get("quantization"),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "19_prompt_hashes.csv"
    if rows:
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            cols = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"기록: {out_csv.relative_to(BASE_DIR)}")

    # ── 판정 ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    counted = {k: v for k, v in token_counts.items() if v is not None}
    if unreachable:
        print(f"도달하지 못한 구성: {', '.join(unreachable)}")

    if not counted:
        print("판정 보류 — 도달한 구성이 없습니다. 서버를 띄운 뒤 다시 실행하세요.")
        return 1

    for k, v in counted.items():
        m = min_counts.get(k)
        c = (v - m) if (m is not None) else None
        print(f"  구성 {k}: 입력 토큰 {v}   (템플릿 몫 ~{m}, 본문 몫 {c})")

    # ── 구성이 하나 ──────────────────────────────────────────────────
    # 하나만 "요청"했다면 기준선 기록이므로 성공이다.
    # 여럿을 요청했는데 하나만 닿았다면 그건 실패다 — 둘을 구분한다.
    if len(counted) == 1:
        k = next(iter(counted))
        if len(configs) > 1:
            print()
            print(f"실패 — {len(configs)}개 구성을 요청했는데 {k} 하나에만 닿았습니다.")
            print(f"       닿지 않은 구성: {', '.join(unreachable)}")
            print("       대조는 구성이 둘 이상일 때만 가능합니다. 서버를 띄운 뒤 다시 실행하거나,")
            print(f"       기준선만 먼저 잡으려면 --configs {k} 로 명시하십시오.")
            return 1
        print()
        print(f"기준선 기록 완료 — 구성 {k} 단독 실행이라 대조할 상대가 없습니다.")
        print(f"  나중에 다른 구성을 띄운 뒤, 그 구성의 입력 토큰이 {counted[k]} 와")
        print("  같은지 보면 됩니다. 값은 19_prompt_hashes.csv 에 남았습니다.")
        print("  (실패가 아닙니다 — 대조는 구성이 둘 이상일 때 이뤄집니다.)")
        return 0

    uniq = set(counted.values())
    if len(uniq) == 1:
        print()
        print("통과 — 모든 구성이 같은 수의 입력 토큰을 셌습니다.")
        print("       본실행으로 넘어가도 됩니다.")
        return 0

    # ── 불일치의 원인을 좁힌다 ──────────────────────────────────────
    # 템플릿이 붙이는 몫만 다른가, 본문을 자르는 방식(토크나이저)까지 다른가.
    contents = {k: counted[k] - min_counts[k] for k in counted
                if min_counts.get(k) is not None}
    same_content = len(set(contents.values())) == 1 if len(contents) > 1 else None

    print()
    print("실패 — 구성마다 입력 토큰 수가 다릅니다.")
    print("       서버가 프롬프트를 감싸는 방식이 다르다는 뜻이고, 이 상태에서 품질을")
    print("       비교하면 엔진 차이가 아니라 입력 차이를 재게 됩니다.")
    print()

    if same_content is True:
        print("       진단: **본문 몫은 같고 템플릿 몫만 다릅니다.**")
        print("             토크나이저는 같고 템플릿이 붙이는 고정 문구가 다릅니다.")
        print("             → 고칠 수 있는 불일치입니다.")
        print()
        print("       가장 흔한 원인 — Llama 3.x 계열에서 실제로 자주 납니다:")
        print("         Ollama 의 GGUF 내장 템플릿에는 `Cutting Knowledge Date` 만 있고")
        print("         HF 의 chat_template 에는 `Today Date: <오늘 날짜>` 가 더 붙습니다.")
        print("         이러면 토큰 수가 어긋날 뿐 아니라 **날짜가 바뀌면 입력도 바뀌어**")
        print("         날마다 다른 실험이 됩니다.")
        print()
        print("       조치 (하나를 고르고 기록할 것):")
        print("         (a) vLLM 을 Ollama 템플릿으로 맞춘다")
        print("             vllm serve ... --chat-template <ollama_template.jinja>")
        print("             (위에 찍힌 템플릿을 Jinja 로 옮겨 파일로 저장)")
        print("         (b) 차이를 안고 가되 **설계서 §2 의 교란 목록에 한 줄 추가**하고")
        print("             결과 보고서에 '입력이 N토큰 달랐다'를 명시한다")
    elif same_content is False:
        print("       진단: **본문 몫부터 다릅니다.**")
        print("             같은 한국어 문장을 서로 다른 개수로 자르고 있으므로")
        print("             토크나이저 자체가 다릅니다(체크포인트 계열이 다를 수 있음).")
        print("             → 템플릿만 맞춰서는 해소되지 않습니다.")
        print("             vLLM 쪽 체크포인트가 Ollama 가 쓰는 것과 같은 베이스인지")
        print("             먼저 확인하십시오.")
    else:
        print("       진단 불가 — 최소 탐침 토큰 수를 얻지 못했습니다.")

    print()
    print("       그 밖에 확인할 것:")
    print("         · vLLM 을 /v1/chat/completions 로 부르고 있는가")
    print("           (/v1/completions 는 템플릿을 적용하지 않아 반드시 어긋납니다)")
    print()
    print("       설계서 §11 — 이 불일치를 해소하기 전에는 단계 0을 시작하지 않습니다.")
    return 1


# ══════════════════════════════════════════════════════════════════
# 본실행
# ══════════════════════════════════════════════════════════════════
def run(configs: list[str], model: str, sampler: str, urls: dict, dry: bool) -> int:
    questions, version = load_questions()
    total = len(configs) * len(questions) * PASS_COUNT

    # 측정 환경을 레코드와 함께 남긴다 — 부록 L. 나중에 이 수치를 다른 시점의
    # 수치와 같은 표에 놓아도 되는지는 이 지문으로만 판단할 수 있다.
    env = env_snapshot()

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 단계 0 — 서빙 구성 비교")
    print(f"  환경 {env_describe(env)}")
    print(f"  평가셋 {version} · 문항 {len(questions)} · 구성 {len(configs)} · 패스 {PASS_COUNT}")
    print(f"  모델 {model} · 샘플러 {sampler}")
    print(f"  총 호출 {total}회\n")
    if dry:
        for p in range(1, PASS_COUNT + 1):
            order = configs if p % 2 == 1 else list(reversed(configs))
            print(f"  패스 {p}: {' → '.join(order)}")
        print("\n--dry-run 이므로 호출하지 않고 종료합니다.")
        return 0

    # 먼저 전부 도달 가능한지 본다. 중간에 끊기면 패스 균형이 깨진다.
    backends = {}
    for key in configs:
        try:
            be = build_backend(key, model, sampler, urls.get(key))
        except ValueError as exc:
            print(f"구성 {key} 를 만들 수 없습니다:\n{exc}")
            return 1
        ok, msg = be.health()
        if not ok:
            print(f"구성 {key} 도달 실패: {msg}")
            print("중단합니다 — 일부 구성만 돌리면 패스 균형이 깨져 비교가 성립하지 않습니다.")
            print("한 구성만 보려면 --configs 로 명시하세요.")
            return 1
        backends[key] = be

    records: list[dict] = []
    warmups: list[dict] = []
    done = 0

    for pass_idx in range(1, PASS_COUNT + 1):
        # 역순 패스 — 뒤에 도는 구성이 늘 뜨거운 GPU 에서 측정되지 않게 (설계서 §6.1)
        order = configs if pass_idx % 2 == 1 else list(reversed(configs))
        print(f"\n{'=' * 64}\n패스 {pass_idx}/{PASS_COUNT} — 순서 {' → '.join(order)}\n{'=' * 64}")

        for key in order:
            be = backends[key]
            cfg = CONFIGS[key]

            # 다른 구성이 VRAM 을 쥐고 있으면 이번 측정이 오염된다
            for other in configs:
                if other != key:
                    backends[other].unload()

            print(f"\n[{key}] {cfg['label']}")
            w = be.generate(WARMUP_PROMPT, timeout=300)
            warmups.append({"pass_index": pass_idx, "config": key,
                            "total_time_sec": w.total_time_sec, "ttft_sec": w.ttft_sec,
                            "success": w.success, "error": w.error_message})
            print(f"  워밍업 {w.total_time_sec}초 (집계 제외)")

            for q in questions:
                res = be.generate(q["prompt"])
                done += 1
                records.append({
                    # ── 05_quality_scoring.py 가 기대하는 열 ──
                    "model_name": model,
                    "question_id": q["id"],
                    "category": q["category"],
                    "run_index": pass_idx,
                    "prompt": q["prompt"],
                    "eval_criteria": q["eval_criteria"],
                    "response_text": res.response_text,
                    "total_time_sec": res.total_time_sec,
                    "ttft_sec": res.ttft_sec,
                    "generated_tokens": res.generated_tokens,
                    "tps": res.tps,
                    "tps_wall": res.tps_wall,
                    "tps_source": res.tps_source,
                    "is_success": res.success,
                    "error_message": res.error_message,
                    "manual_score": None,
                    # ── 이 실험이 덧붙이는 열 ──
                    "config": key,
                    "config_label": cfg["label"],
                    "engine": cfg["engine"],
                    "runtime": cfg["runtime"],
                    "weights_format": cfg["weights_format"],
                    "sampler_profile": sampler,
                    "pass_order": order.index(key) + 1,
                    "prompt_sha256": prompt_sha256(q["prompt"]),
                    "server_prompt_tokens": res.prompt_tokens,
                    "ttft_server_sec": res.ttft_server_sec,
                    **env_row_fields(env),
                })
                mark = "" if res.success else "  ← 실패"
                print(f"  [{done:>3}/{total}] q{q['id']:<3} {res.total_time_sec:>6.2f}초 "
                      f"TTFT {res.ttft_sec:>5.2f} {res.generated_tokens:>4}토큰{mark}")

            be.unload()
            if not (pass_idx == PASS_COUNT and key == order[-1]):
                print(f"  cooldown {COOLDOWN_SEC}초")
                time.sleep(COOLDOWN_SEC)

    env_save_pair(env, env_snapshot(), OUT_DIR / "19_env.json")
    write_outputs(records, warmups, configs, model, sampler, version, env)
    return 0


# ══════════════════════════════════════════════════════════════════
# 출력
# ══════════════════════════════════════════════════════════════════
def write_outputs(records, warmups, configs, model, sampler, version, env=None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "19_serving_results.json"
    csv_path = OUT_DIR / "19_serving_results.csv"
    md_path = OUT_DIR / "19_serving_summary.md"

    # 05_quality_scoring.py 는 평평한 리스트를 기대한다 (04 결과와 같은 모양)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    with open(OUT_DIR / "19_serving_warmups.json", "w", encoding="utf-8") as f:
        json.dump(warmups, f, ensure_ascii=False, indent=2)

    def agg(key):
        rs = [r for r in records if r["config"] == key]
        ok = [r for r in rs if r["is_success"]]
        n = len(ok)
        if not n:
            return None
        lat = sorted(r["total_time_sec"] for r in ok)
        return {
            "n": len(rs), "ok": n,
            "mean_sec": sum(lat) / n,
            "p95_sec": lat[max(0, int(round(0.95 * n)) - 1)],
            "mean_ttft": sum(r["ttft_sec"] for r in ok) / n,
            "mean_tps": sum(r["tps"] for r in ok) / n,
            "mean_tps_wall": sum(r["tps_wall"] for r in ok) / n,
            "tps_src": sorted({r["tps_source"] for r in ok}),
            "mean_tokens": sum(r["generated_tokens"] for r in ok) / n,
            "ptok": sorted({r["server_prompt_tokens"] for r in ok if r["server_prompt_tokens"]}),
        }

    L = []
    a = L.append
    a("# 단계 0 — 서빙 구성 비교 결과")
    a("")
    if env:
        a(f"측정 환경 — {env_describe(env)}")
        a("")
    a(f"평가셋 `{version}` · 모델 `{model}` · 샘플러 프로파일 `{sampler}` · 문항 {len(set(r['question_id'] for r in records))} × 패스 {PASS_COUNT}")
    a("")
    a("**이 표로 「vLLM 이 더 빠르다」고 말하지 않는다.** 구성 간에는 엔진·가중치 포맷·")
    a("런타임·샘플러가 함께 다르다(설계서 §2). A↔B 는 런타임 효과, B↔C 는 엔진+포맷 효과다.")
    a("")
    a("TPS 열은 **`tps_wall`**(= 생성 토큰 / (총시간 − TTFT))이다. 엔진 고유 TPS 는")
    a("Ollama 가 서버 `eval_duration`, vLLM 이 벽시계 기준이라 같은 자가 아니므로")
    a("엔진 간 비교에 쓰지 않는다. 원본 열 `tps` 와 `tps_source` 는 CSV 에 남아 있다.")
    a("")
    a("| 구성 | 설명 | 성공 | 평균 지연 | p95 | 평균 TTFT | 평균 TPS(공통자) | 평균 출력 토큰 | 입력 토큰 |")
    a("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    stats = {}
    for key in configs:
        s = agg(key)
        stats[key] = s
        if not s:
            a(f"| {key} | {CONFIGS[key]['label']} | 0 | — | — | — | — | — | — |")
            continue
        a(f"| {key} | {CONFIGS[key]['label']} | {s['ok']}/{s['n']} | {s['mean_sec']:.3f}초 | "
          f"{s['p95_sec']:.3f}초 | {s['mean_ttft']:.3f}초 | {s['mean_tps_wall']:.1f} | "
          f"{s['mean_tokens']:.1f} | {', '.join(map(str, s['ptok'])) or '—'} |")
    a("")

    # 입력 토큰이 구성마다 다르면 품질 비교 자체가 성립하지 않는다
    all_ptok = {k: set(s["ptok"]) for k, s in stats.items() if s}
    flat = set().union(*all_ptok.values()) if all_ptok else set()
    if len(all_ptok) > 1 and len({tuple(sorted(v)) for v in all_ptok.values()}) > 1:
        a("> **경고 — 구성별 입력 토큰 수가 다르다.** 서버가 프롬프트를 감싸는 방식이 다르다는")
        a("> 뜻이므로, 아래 품질 비교는 엔진 차이가 아니라 입력 차이를 재고 있을 수 있다.")
        a("> `--check-prompts` 를 먼저 통과시켜야 한다(설계서 §8 C1).")
        a("")

    # 열 드리프트 (설계서 §6.1)
    a("## 패스 간 드리프트")
    a("")
    a("| 구성 | 패스 1 평균 | 패스 2 평균 | 차이 |")
    a("| --- | ---: | ---: | ---: |")
    drift_flag = False
    for key in configs:
        per = []
        for p in (1, 2):
            rs = [r["total_time_sec"] for r in records
                  if r["config"] == key and r["run_index"] == p and r["is_success"]]
            per.append(sum(rs) / len(rs) if rs else None)
        if None in per:
            a(f"| {key} | — | — | — |")
            continue
        d = per[1] - per[0]
        if abs(d) > DRIFT_WARN_SEC:
            drift_flag = True
        a(f"| {key} | {per[0]:.3f}초 | {per[1]:.3f}초 | {d:+.3f}초 |")
    a("")
    if drift_flag:
        a(f"> **패스 간 차이가 {DRIFT_WARN_SEC}초를 넘는 구성이 있다.** 열 드리프트를 의심하고")
        a("> cooldown 을 늘려 재실행할 것(설계서 §6.1).")
    else:
        a(f"패스 간 차이가 모두 {DRIFT_WARN_SEC}초 이내다 — 열 드리프트는 무시할 수준.")
    a("")

    a("## 다음")
    a("")
    a("```")
    a("uv run python 05_quality_scoring.py work/19_serving/19_serving_results.json")
    a("```")
    a("")
    a("채점 결과는 `19_serving_scored.csv` 로 떨어진다. 채점기의 요약 출력은")
    a("`model_name` 으로 묶으므로 세 구성이 한 덩어리로 보인다 — 구성별 비교는")
    a("`config` 열로 직접 집계할 것.")
    a("")
    a("품질 판정이 구성 간에 갈리면 **원인을 엔진·포맷·템플릿 중 무엇인지 밝히기 전까지")
    a("단계 1로 넘어가지 않는다**(설계서 §10 R2).")

    md_path.write_text("\n".join(L), encoding="utf-8")

    print(f"\n{'=' * 64}")
    for key in configs:
        s = stats[key]
        if s:
            print(f"  {key} {CONFIGS[key]['label']:<26} {s['mean_sec']:>6.3f}초  "
                  f"TPS(공통자) {s['mean_tps_wall']:>6.1f}  입력토큰 {s['ptok']}")
    print("=" * 64)
    print(f"기록: {json_path.name}, {csv_path.name}, {md_path.name}")
    if len({tuple(sorted(v)) for v in all_ptok.values()}) > 1:
        print("\n경고: 구성별 입력 토큰 수가 다릅니다. --check-prompts 를 먼저 통과시키세요.")


# ══════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="단계 0 — 서빙 구성 A·B·C 비교")
    ap.add_argument("--check-prompts", action="store_true",
                    help="세 구성이 같은 프롬프트를 보는지만 대조하고 끝낸다 (본실행 전 필수)")
    ap.add_argument("--configs", default="A,B,C", help="쉼표 구분 (기본 A,B,C)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sampler", default="fixed", choices=sorted(SAMPLER_PROFILES),
                    help="fixed=구성 간 비교용, baseline=기존 185회 재현용(Ollama 전용)")
    ap.add_argument("--dry-run", action="store_true", help="호출 없이 실행 계획만 출력")
    for k in CONFIGS:
        ap.add_argument(f"--url-{k.lower()}", default=None,
                        help=f"구성 {k} 엔드포인트 (기본 {CONFIGS[k]['base_url']})")
    args = ap.parse_args()

    configs = [c.strip().upper() for c in args.configs.split(",") if c.strip()]
    bad = [c for c in configs if c not in CONFIGS]
    if bad:
        print(f"알 수 없는 구성: {', '.join(bad)} (가능: {', '.join(CONFIGS)})")
        return 1
    urls = {k: getattr(args, f"url_{k.lower()}") for k in CONFIGS}

    if not GOLD_FILE.exists():
        print(f"평가셋이 없습니다: {GOLD_FILE}")
        return 1

    if args.check_prompts:
        return check_prompts(configs, args.model, args.sampler, urls)
    return run(configs, args.model, args.sampler, urls, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
