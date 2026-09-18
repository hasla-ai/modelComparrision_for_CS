"""프롬프트 보강 A/B 실험 — 기권·권한·환각 규칙 추가의 효과와 부작용 측정.

설계 근거는 PROMPT_AB_DESIGN.md에 있다.

무엇을 검증하는가
  최종 보고서 §7이 '배포 전 필수 조치'로 권고한 시스템 프롬프트 보강을 실제로
  적용해, Q9(정보 부족)·Q10(범위 밖) 게이트 탈락이 해소되는지 본다. 동시에
  부록 B의 'P(프롬프트로 교정 가능)' 재판정이 이 두 문항에도 성립하는지 확인한다.

핵심 관찰 (실행 전 확인됨)
  현행 프롬프트는 category에 "정보 부족"이 허용값이라고 말하지 않는데, gold는
  Q9에서 그 값을 요구한다. 모델에게 없는 선택지를 고르라고 한 셈이다. 즉 이것은
  모델 실패가 아니라 명세 누락일 가능성이 높다.

루프는 '패스 → 모델 → 질문 → 조건' 4중 구조다(설계서 §4).
  양자화 실험과 달리 조건 교체 비용이 0이므로, 독립변수를 가장 안쪽에 두고
  같은 질문의 A/B를 몇 초 간격으로 붙여 실행한다(대응 표본 설계). 두 조건이
  같은 온도·같은 로드 상태를 공유하므로 환경 요인이 상쇄된다.
  쌍 안의 순서도 교대해 '먼저/나중' 편향을 없앤다.

주의
  조건 B는 gold가 요구하는 문자열을 프롬프트에 넣는다. 결과가 좋아져도
  '모델 품질이 올랐다'가 아니라 '명세에 없던 요구 동작을 명세에 넣었다'로
  해석해야 한다(설계서 §3 함정 1). 그리고 Q1~Q8 회귀 검사를 반드시 함께 본다.

실행:  uv run python 08_prompt_ab_experiment.py
채점:  uv run python 05_quality_scoring.py work/08_prompt_ab/08_prompt_ab_results.json
분석:  uv run python 08_prompt_ab_report.py
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ==========================================
# 1. 설정값 정의
# ==========================================
OLLAMA_BASE = "http://localhost:11434"
GENERATE_URL = f"{OLLAMA_BASE}/api/generate"
TAGS_URL = f"{OLLAMA_BASE}/api/tags"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "work" / "08_prompt_ab"
QUESTIONS_FILE = BASE_DIR / "data" / "02_prompts_and_eval.json"

MODELS = ["llama3.2:3b", "qwen2.5:7b"]
PASS_COUNT = 2
WARMUP_PROMPT = "워밍업용 짧은 요청입니다. 간단히 한 문장으로 답하세요."
REQUEST_TIMEOUT_SEC = 180

PROMPT_VARIANT_VERSION = "ab-v1-2026-09"

# 조건 B에서 현행 프롬프트 '뒤에 붙이기만' 하는 규칙 블록.
# 기존 문장은 한 글자도 고치지 않는다 — A와 B의 차이를 이 블록 하나로 한정하기 위함.
# 세 규칙은 각각 게이트 축에 대응한다: 기권 적절성 / 과잉확답 / 환각.
REINFORCEMENT = """
규칙:
- 문의에서 증상이나 요청을 식별할 수 없을 만큼 정보가 부족하면, category를 임의로 단정하지 말고 "정보 부족"으로 적고 note에 추가로 확인해야 할 항목을 적으세요.
- 보상·환불·배상처럼 상담원 권한 밖의 확정 판단이 필요한 요구는 승인하거나 약속하지 말고, 요구 내용만 그대로 옮긴 뒤 note에 권한 밖 사항임을 적으세요.
- 문의에 없는 값(주문번호, 날짜, 금액 등)은 추측해서 채우지 말고, 해당 항목을 비우거나 note에 없음을 적으세요."""

CONDITIONS = {
    "A": {"label": "현행 (baseline)", "suffix": ""},
    "B": {"label": "보강 (규칙 3개 추가)", "suffix": REINFORCEMENT},
}

_TYPE_LABELS = {
    "정상": "정상 사례",
    "경계": "경계 사례",
    "특수: 정보부족": "정보 부족 사례",
    "특수: 범위밖": "범위 밖 사례",
}


def safe_name(tag: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "_", tag)


def load_questions() -> list[dict]:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "question_id": q["id"],
            "type": _TYPE_LABELS.get(
                q["taxonomy"]["info_sufficiency"], q["taxonomy"]["info_sufficiency"]
            ),
            "base_prompt": q["prompt"],
            "eval_criteria": q.get("eval_criteria", ""),
        }
        for q in data["questions"]
    ]


def build_prompt(base_prompt: str, condition: str) -> str:
    """조건 A는 JSON 원문 그대로, 조건 B는 원문 + 규칙 블록 (불변식 1·2)."""
    return base_prompt + CONDITIONS[condition]["suffix"]


# ==========================================
# 2. Ollama 호출
# ==========================================
def _failed(msg: str) -> dict:
    return {
        "success": False, "response_text": "", "total_time_sec": 0,
        "ttft_sec": 0, "eval_count_tokens": 0, "tps": 0, "error_msg": msg,
    }


def run_ollama_query(model_name: str, prompt: str) -> dict:
    """04_local_experiment.py와 동일한 호출 경로·생성 설정."""
    payload = {
        "model": model_name, "prompt": prompt,
        "stream": False, "options": {"temperature": 0.1},
    }
    start = time.time()
    try:
        response = requests.post(GENERATE_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        end = time.time()
        if response.status_code != 200:
            return _failed(f"HTTP Error {response.status_code}")
        res = response.json()
        eval_duration = res.get("eval_duration", 0) / 1e9
        eval_count = res.get("eval_count", 0)
        return {
            "success": True,
            "response_text": res.get("response", "").strip(),
            "total_time_sec": round(end - start, 3),
            "ttft_sec": round(
                (res.get("load_duration", 0) + res.get("prompt_eval_duration", 0)) / 1e9, 3
            ),
            "eval_count_tokens": eval_count,
            "tps": round(eval_count / eval_duration, 2) if eval_duration > 0 else 0,
            "error_msg": "",
        }
    except Exception as exc:
        return _failed(str(exc))


def unload(model_name: str) -> None:
    try:
        requests.post(GENERATE_URL,
                      json={"model": model_name, "prompt": "", "keep_alive": 0},
                      timeout=30)
    except Exception:
        pass
    time.sleep(1.0)


def preflight(models: list[str]) -> bool:
    try:
        installed = {m.get("name") for m in requests.get(TAGS_URL, timeout=15).json().get("models", [])}
    except Exception as exc:
        print(f"Ollama에 연결할 수 없습니다: {exc}")
        print("Ollama 앱이 실행 중인지 확인하세요.")
        return False
    missing = [m for m in models if m not in installed]
    if missing:
        print("아래 모델을 먼저 받아야 합니다.\n")
        for m in missing:
            print(f"  ollama pull {m}")
        return False
    return True


# ==========================================
# 3. 메인 — 패스 → 모델 → 질문 → 조건 4중 루프
# ==========================================
def main() -> int:
    if not preflight(MODELS):
        return 1

    questions = load_questions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = PASS_COUNT * len(MODELS) * len(questions) * len(CONDITIONS)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 프롬프트 보강 A/B 실험 시작")
    print(f"-> 패스 {PASS_COUNT} × 모델 {len(MODELS)} × 질문 {len(questions)} × 조건 {len(CONDITIONS)} = {total}회")
    print(f"-> 같은 질문의 A/B를 붙여 실행하고, 쌍 안의 순서를 교대한다 (대응 표본)\n")

    results: list[dict] = []
    done = 0

    # ── 바깥: 반복 패스 ──────────────────────────────────────────────
    for pass_idx in range(1, PASS_COUNT + 1):
        model_order = MODELS if pass_idx % 2 == 1 else list(reversed(MODELS))
        print("#" * 64)
        print(f"# 패스 {pass_idx}/{PASS_COUNT}  모델 순서: {' → '.join(model_order)}")
        print("#" * 64)

        # ── 중간: 모델 (로드가 비싸므로 블록) ─────────────────────────
        for model in model_order:
            print(f"\n--- {model} (패스 {pass_idx}) ---")
            for other in MODELS:
                if other != model:
                    unload(other)

            print("워밍업... ", end="", flush=True)
            warm = run_ollama_query(model, WARMUP_PROMPT)
            print("완료" if warm["success"] else f"실패 ({warm['error_msg']})")
            with open(OUTPUT_DIR / f"warmup_{safe_name(model)}_pass{pass_idx}.json",
                      "w", encoding="utf-8") as f:
                json.dump({"model": model, "pass_index": pass_idx,
                           "prompt": WARMUP_PROMPT,
                           "called_at": datetime.now().isoformat(), **warm},
                          f, ensure_ascii=False, indent=2)

            # ── 안쪽: 질문 ────────────────────────────────────────────
            for q in questions:
                qid = q["question_id"]
                # 쌍 안의 순서 교대 (불변식 4)
                conds = ["A", "B"] if (pass_idx + int(qid)) % 2 == 0 else ["B", "A"]

                # ── 최내: 조건 — 교체 비용 0이므로 붙여서 실행 ──────────
                for pair_order, cond in enumerate(conds, start=1):
                    done += 1
                    prompt = build_prompt(q["base_prompt"], cond)
                    print(f"[{done}/{total}] P{pass_idx} {model} | Q{qid} | 조건 {cond}... ",
                          end="", flush=True)
                    res = run_ollama_query(model, prompt)
                    print(f"성공 ({res['total_time_sec']}s, {res['eval_count_tokens']}토큰)"
                          if res["success"] else f"실패 ({res['error_msg']})")

                    results.append({
                        "model_name": model,
                        "condition": cond,
                        "condition_label": CONDITIONS[cond]["label"],
                        "prompt_variant_version": PROMPT_VARIANT_VERSION,
                        "question_id": qid,
                        "category": q["type"],
                        "run_index": pass_idx,
                        "pair_order": pair_order,      # 이 쌍에서 몇 번째로 실행됐나
                        "prompt": prompt,              # 실제로 보낸 원문 (감사용)
                        "eval_criteria": q["eval_criteria"],
                        "response_text": res["response_text"],
                        "total_time_sec": res["total_time_sec"],
                        "ttft_sec": res["ttft_sec"],
                        "generated_tokens": res["eval_count_tokens"],
                        "tps": res["tps"],
                        "is_success": res["success"],
                        "error_message": res["error_msg"],
                        "manual_score": None,
                    })
            unload(model)

    # ==========================================
    # 4. 저장
    # ==========================================
    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "08_prompt_ab_results.csv"
    json_path = OUTPUT_DIR / "08_prompt_ab_results.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(f"실험 완료  |  {csv_path.name}, {json_path.name}")
    print("=" * 64)

    ok = df[df["is_success"]]
    print("\n[조건별 응답 길이 — 과잉 보수화 1차 점검]")
    print(ok.pivot_table(index="model_name", columns="condition",
                         values="generated_tokens", aggfunc="mean").round(1).to_string())
    print("\n[쌍 순서 균형 점검 — A/B가 먼저 실행된 횟수가 비슷해야 한다]")
    print(df.pivot_table(index="condition", columns="pair_order",
                         values="question_id", aggfunc="count").to_string())

    fails = len(df) - len(ok)
    if fails:
        print(f"\n실패 {fails}건 — 집계에서 제외됨")

    print("\n다음 단계:")
    print(f"  uv run python 05_quality_scoring.py {json_path.relative_to(BASE_DIR)}")
    print(f"  uv run python 08_prompt_ab_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
