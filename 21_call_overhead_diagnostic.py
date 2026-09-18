"""호출 경로에 숨어 있는 상수 오버헤드를 찾는다.

무엇을 발견했나
  단계 1(동시성) 결과를 기존 실험과 대조하다가 이상한 것이 나왔다.
  지금까지의 모든 실험 레코드에서

      총 소요시간 − TTFT − (생성 토큰 ÷ TPS)  ≈  2.05초

  가 **일정하게** 남는다. 실험·모델·양자화 수준·프롬프트가 달라도 같다.

      본실험 llama3.2:3b  n=20   2.054초  (표준편차 0.018)
      본실험 qwen2.5:7b   n=20   2.050초  (표준편차 0.014)
      양자화 전체          n=60   2.048초  (표준편차 0.016)
      프롬프트 A/B 전체     n=80   2.058초  (표준편차 0.019)

  180회에 걸쳐 표준편차가 16~19밀리초다. 이건 모델의 행동이 아니라
  **호출 경로에 박힌 고정 비용**이다.

  반면 스트리밍으로 부르는 새 백엔드(`backends.py`)에서는 같은 값이 0이다.

왜 중요한가
  FINAL_REPORT §5.1 이 공표한 `llama3.2:3b` 3.38초 / `qwen2.5:7b` 4.59초 는
  이 2.05초를 포함한 값이다. 오버헤드가 **모든 호출에 동일하게** 들어갔으므로
  모델 간 차이(1.21초)와 순위는 영향을 받지 않지만, **절대값은 부풀어 있다.**

무엇을 가리는가
  2×2 로 나눠 원인을 좁힌다.

      스트리밍 여부   : stream=False (기존 04 방식) / stream=True (새 방식)
      샘플러 프로파일 : baseline (기존 04 방식)      / fixed

  그리고 세 가지 시계를 동시에 잰다.

      client_wall        클라이언트 벽시계 (04 의 total_time_sec 과 같은 자)
      server_total       Ollama 가 보고한 total_duration
      server_parts       load + prompt_eval + eval (Ollama 보고값 합)

  gap_vs_total = client_wall − server_total   → 클라이언트·네트워크 쪽 비용
  gap_vs_parts = client_wall − server_parts   → 위 + 서버가 durations 로
                                                설명하지 않는 몫

  둘을 비교하면 2.05초가 **서버 안쪽인지 바깥쪽인지** 갈린다.

실행:
    uv run python 21_call_overhead_diagnostic.py            # 80회, 약 2분
    uv run python 21_call_overhead_diagnostic.py --runs 1   # 40회
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
GOLD_FILE = BASE_DIR / "data" / "02_prompts_and_eval.json"
OUT_DIR = BASE_DIR / "work" / "21_call_overhead"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"
N_QUESTIONS = 10          # 본실험과 같은 문항 1~10
TIMEOUT = 180

# 기존 04_local_experiment.py 가 쓰던 옵션 — temperature 만 지정
BASELINE_OPTS = {"temperature": 0.1}
# backends.py 의 fixed 프로파일과 같은 값
FIXED_OPTS = {"temperature": 0.1, "top_p": 0.9, "top_k": 40,
              "repeat_penalty": 1.1, "num_predict": 512, "seed": 20260918}

CONDITIONS = [
    ("nostream+baseline", False, BASELINE_OPTS),   # = 기존 04 방식
    ("nostream+fixed",    False, FIXED_OPTS),
    ("stream+baseline",   True,  BASELINE_OPTS),
    ("stream+fixed",      True,  FIXED_OPTS),      # = 새 backends.py 방식
]


def load_questions(n: int) -> list[dict]:
    with open(GOLD_FILE, encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    return [{"id": q["id"], "prompt": q["prompt"]} for q in qs if q["id"] <= n]


def call(sess, prompt: str, stream: bool, opts: dict) -> dict:
    payload = {"model": MODEL, "prompt": prompt, "stream": stream, "options": opts}
    t0 = time.perf_counter()
    ttft = None
    final: dict = {}
    text_len = 0
    try:
        if stream:
            with sess.post(OLLAMA_URL, json=payload, timeout=TIMEOUT, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = o.get("response", "")
                    if piece and ttft is None:
                        ttft = time.perf_counter() - t0
                    text_len += len(piece)
                    if o.get("done"):
                        final = o
        else:
            r = sess.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            final = r.json()
            text_len = len(final.get("response", ""))
        wall = time.perf_counter() - t0
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    ns = 1e9
    load_s = (final.get("load_duration") or 0) / ns
    peval_s = (final.get("prompt_eval_duration") or 0) / ns
    eval_s = (final.get("eval_duration") or 0) / ns
    total_s = (final.get("total_duration") or 0) / ns
    return {
        "ok": True, "client_wall": wall,
        "server_total": total_s,
        "server_parts": load_s + peval_s + eval_s,
        "load": load_s, "prompt_eval": peval_s, "eval": eval_s,
        "eval_count": final.get("eval_count") or 0,
        "prompt_eval_count": final.get("prompt_eval_count") or 0,
        "ttft_stream": ttft, "chars": text_len,
        "gap_vs_total": wall - total_s,
        "gap_vs_parts": wall - (load_s + peval_s + eval_s),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="호출 경로 상수 오버헤드 진단")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--questions", type=int, default=N_QUESTIONS)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    globals()["MODEL"] = args.model

    qs = load_questions(args.questions)
    total = len(CONDITIONS) * len(qs) * args.runs
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 호출 오버헤드 진단")
    print(f"  모델 {MODEL} · 문항 {len(qs)} · 조건 {len(CONDITIONS)} · 반복 {args.runs}")
    print(f"  총 {total}회\n")

    sess = requests.Session()
    # 모델 적재 비용이 첫 조건에만 실리지 않도록 먼저 한 번 깨워 둔다
    print("워밍업…", end="", flush=True)
    call(sess, "안녕하세요. 한 문장으로 답하세요.", False, BASELINE_OPTS)
    print(" 완료\n")

    rows, done = [], 0
    for run in range(1, args.runs + 1):
        # 조건 순서를 회차마다 뒤집어 드리프트가 한 조건에 쏠리지 않게 한다
        order = CONDITIONS if run % 2 == 1 else list(reversed(CONDITIONS))
        for name, stream, opts in order:
            for q in qs:
                res = call(sess, q["prompt"], stream, opts)
                done += 1
                if not res.get("ok"):
                    print(f"  [{done}/{total}] {name} q{q['id']} 실패: {res['error']}")
                    continue
                rows.append({"condition": name, "stream": stream, "run": run,
                             "question_id": q["id"], **{k: v for k, v in res.items() if k != "ok"}})
                print(f"\r  [{done}/{total}] {name:<18} q{q['id']:<3} "
                      f"벽시계 {res['client_wall']:.3f}초  "
                      f"서버total {res['server_total']:.3f}  "
                      f"gap {res['gap_vs_total']:+.3f}    ", end="", flush=True)
    print()

    if not rows:
        print("수집된 레코드가 없습니다.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "21_call_overhead_raw.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(name, field):
        v = [r[field] for r in rows if r["condition"] == name]
        return (st.fmean(v), st.pstdev(v)) if v else (float("nan"), float("nan"))

    L = []
    a = L.append
    a("# 호출 경로 상수 오버헤드 진단")
    a("")
    a(f"모델 `{MODEL}` · 문항 1~{args.questions} · 조건당 {args.runs}회 · 총 {len(rows)}건")
    a("")
    a("기존 실험 180회 전부에서 `총시간 − TTFT − 생성시간` 이 2.05초로 일정하게 남았다.")
    a("어디서 생기는지 2×2 로 가른다.")
    a("")
    a("| 조건 | 클라이언트 벽시계 | 서버 total_duration | 벽시계 − 서버total | 벽시계 − 서버부분합 |")
    a("| --- | ---: | ---: | ---: | ---: |")
    for name, _, _ in CONDITIONS:
        w_m, w_s = agg(name, "client_wall")
        t_m, _ = agg(name, "server_total")
        g1_m, g1_s = agg(name, "gap_vs_total")
        g2_m, _ = agg(name, "gap_vs_parts")
        a(f"| `{name}` | {w_m:.3f}초 (±{w_s:.3f}) | {t_m:.3f}초 | "
          f"**{g1_m:+.3f}초** (±{g1_s:.3f}) | {g2_m:+.3f}초 |")
    a("")

    ns_base = agg("nostream+baseline", "client_wall")[0]
    s_fix = agg("stream+fixed", "client_wall")[0]
    g_ns = agg("nostream+baseline", "gap_vs_total")[0]
    g_s = agg("stream+baseline", "gap_vs_total")[0]

    a("## 판정")
    a("")
    a(f"- 기존 04 방식(`nostream+baseline`) 벽시계 평균: **{ns_base:.3f}초**")
    a(f"- 새 방식(`stream+fixed`) 벽시계 평균: **{s_fix:.3f}초**")
    a(f"- 차이: **{ns_base - s_fix:+.3f}초**")
    a("")
    if abs(g_ns) > 1.0 and abs(g_s) < 0.3:
        a(f"**스트리밍 여부가 원인이다.** 비스트리밍에서는 클라이언트 벽시계가 서버가 보고한")
        a(f"`total_duration` 보다 {g_ns:.3f}초 크고, 스트리밍에서는 {g_s:.3f}초로 사라진다.")
        a("서버는 그 시간을 자기 처리 시간으로 세지 않는다 — 응답을 모아 돌려주는 구간의 비용이다.")
    elif abs(g_ns) > 1.0 and abs(g_s) > 1.0:
        a(f"**스트리밍과 무관하다.** 두 방식 모두 서버 보고값보다 1초 이상 크다")
        a(f"(비스트리밍 {g_ns:+.3f}초, 스트리밍 {g_s:+.3f}초). 클라이언트·네트워크 쪽을 봐야 한다.")
    elif abs(g_ns) < 0.3:
        a(f"**이번 실행에서는 재현되지 않았다** (비스트리밍 gap {g_ns:+.3f}초).")
        a("기존 실험 당시의 환경 요인(Ollama 버전, 동시 실행 프로세스 등)일 수 있다.")
        a("그렇다면 과거 절대값은 그때의 값으로 두되, 지금 값과 같은 표에 넣지 않는다.")
    else:
        a("판정 보류 — 위 표를 직접 읽고 해석할 것.")
    a("")
    a("## 보고서에 미치는 영향")
    a("")
    a("오버헤드가 **모든 호출에 동일하게** 들어갔다면 모델·조건 간 *차이*와 순위는 영향이 없고,")
    a("**절대 지연값만 부풀어 있다.** 이 경우 FINAL_REPORT §5.1 의 3.38초 / 4.59초 에는")
    a("측정 방식 각주를 달아야 하며, 새 측정값과 같은 표에 나란히 놓아서는 안 된다.")
    a("")
    a("게이트 `P95 ≤ 6초` 판정은 오버헤드를 **포함한** 값으로도 통과했으므로 방향이 바뀌지 않는다")
    a("— 실제 지연은 그보다 작다.")

    (OUT_DIR / "21_call_overhead_summary.md").write_text("\n".join(L), encoding="utf-8")

    print()
    print("=" * 70)
    for name, _, _ in CONDITIONS:
        w_m, _ = agg(name, "client_wall")
        g1_m, _ = agg(name, "gap_vs_total")
        print(f"  {name:<20} 벽시계 {w_m:>7.3f}초   벽시계−서버total {g1_m:>+7.3f}초")
    print("=" * 70)
    print(f"기록: {(OUT_DIR / '21_call_overhead_summary.md').relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
