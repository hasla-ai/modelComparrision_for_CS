import json
import time
import datetime
from getpass import getpass
from pathlib import Path

import pandas as pd
from openai import APIError, APITimeoutError, OpenAI

# ==========================================
# 설정값 — docs/model_selection_spec.md STEP 7 / §2 실험 설계 반영
# 02_luna_chat.py와 동일한 OpenAI 호출 방식(Responses API, gpt-5.6-luna)만 사용.
# Upstage Solar는 이 실험에서 완전히 제외한다(사용자 지시).
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
GOLD_FILE = BASE_DIR / "data" / "02_prompts_and_eval.json"
OUTPUT_DIR = BASE_DIR / "work" / "06_cloud_comparison"

CLOUD_MODEL = "gpt-5.6-luna"
CLOUD_BASE_URL = "https://api.openai.com/v1"


def load_cloud_questions():
    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    type_labels = {
        "정상": "정상 사례",
        "경계": "경계 사례",
        "특수: 정보부족": "정보 부족 사례",
        "특수: 범위밖": "범위 밖 사례",
    }
    questions = []
    for q in data["questions"]:
        if not q.get("cloud_comparison"):
            continue
        questions.append({
            "question_id": q["id"],
            "type": type_labels.get(q["taxonomy"]["info_sufficiency"], q["taxonomy"]["info_sufficiency"]),
            "prompt": q["prompt"],
            "eval_criteria": q.get("eval_criteria", ""),
        })
    return sorted(questions, key=lambda q: q["question_id"])


def now_iso():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()


def call_cloud(client: OpenAI, prompt: str):
    start = time.perf_counter()
    try:
        response = client.responses.create(
            model=CLOUD_MODEL,
            input=prompt,
            reasoning={"effort": "none"},
            max_output_tokens=512,
            tools=[],
            tool_choice="none",
            store=False,
        )
        elapsed = time.perf_counter() - start
        usage = getattr(response, "usage", None)
        return {
            "success": True,
            "response_text": (response.output_text or "").strip(),
            "status": response.status,
            "total_time_sec": round(elapsed, 3),
            "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
            "generated_tokens": getattr(usage, "output_tokens", None) if usage else None,
            "error_message": "" if response.status == "completed" else f"미완료 상태: {response.status}",
        }
    except APITimeoutError:
        return {
            "success": False, "response_text": "", "status": "timeout", "total_time_sec": None,
            "input_tokens": None, "generated_tokens": None,
            "error_message": "응답 대기 시간 초과",
        }
    except APIError as e:
        return {
            "success": False, "response_text": "", "status": "error", "total_time_sec": None,
            "input_tokens": None, "generated_tokens": None,
            "error_message": f"API 오류: {getattr(e, 'status_code', str(e))}",
        }


def main():
    # 키는 실행할 때만 입력받고 화면·파일·결과 레코드 어디에도 저장하지 않음
    # (docs/model_selection_spec.md STEP 7 "API 키는 코드·저장소·실험 로그·스크린샷에 포함 금지")
    api_key = getpass("OpenAI API 키를 붙여넣고 Enter (화면에 보이지 않음): ").strip()
    if not api_key:
        raise SystemExit("키를 입력하지 않아 API를 호출하지 않았습니다.")

    client = OpenAI(api_key=api_key, base_url=CLOUD_BASE_URL, timeout=60, max_retries=0)

    questions = load_cloud_questions()
    print(f"Cloud 비교 질문 {len(questions)}개 (사전 선정된 ID: {[q['question_id'] for q in questions]})")

    results = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] question_id={q['question_id']} type={q['type']} ... ", end="", flush=True)
        res = call_cloud(client, q["prompt"])

        tps = None
        if res["success"] and res["total_time_sec"] and res["generated_tokens"]:
            tps = round(res["generated_tokens"] / res["total_time_sec"], 2)

        if res["success"]:
            print(f"성공! ({res['total_time_sec']}s, {res['generated_tokens']} tokens, status={res['status']})")
        else:
            print(f"실패! ({res['error_message']})")

        results.append({
            "model_name": CLOUD_MODEL,
            "question_id": q["question_id"],
            "category": q["type"],
            "run_index": 1,
            "prompt": q["prompt"],
            "eval_criteria": q["eval_criteria"],
            "response_text": res["response_text"],
            "total_time_sec": res["total_time_sec"],
            "ttft_sec": None,  # Responses API 논스트리밍 호출 — 첫 토큰 시간은 측정하지 않음
            "input_tokens": res["input_tokens"],
            "generated_tokens": res["generated_tokens"],
            "tps": tps,
            "is_success": res["success"],
            "error_message": res["error_message"],
            "manual_score": None,
            "called_at": now_iso(),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_out = OUTPUT_DIR / "06_cloud_comparison_results.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(results)
    csv_out = OUTPUT_DIR / "06_cloud_comparison_results.csv"
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    print(f"\n저장됨: {json_out}, {csv_out}")
    print(f"성공 {df['is_success'].sum()}/{len(df)}")
    print("추정 비용은 기록하지 않았습니다. platform.openai.com/usage에서 실제 사용량을 확인하세요"
          "(README 안내 — 실험 합계 인당 최대 $5 한도).")


if __name__ == "__main__":
    main()
