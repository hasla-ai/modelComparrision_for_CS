import json
import re
import sys
import time
import datetime
from pathlib import Path

import pandas as pd
import requests

# ==========================================
# 설정값 — docs/quality_scoring_spec.md 기능 명세 구현
# ==========================================
OLLAMA_API_URL = "http://localhost:11434/api/generate"

BASE_DIR = Path(__file__).resolve().parent
GOLD_FILE = BASE_DIR / "data" / "02_prompts_and_eval.json"

# 인자 없이 실행하면 로컬 40회 본실험을 채점(기존 동작 유지).
# 다른 결과 파일(예: Cloud 비교 결과)을 채점하려면 경로를 인자로 전달:
#   uv run python 05_quality_scoring.py work/06_cloud_comparison/06_cloud_comparison_results.json
if len(sys.argv) > 1:
    RESULTS_FILE = Path(sys.argv[1])
    OUTPUT_DIR = RESULTS_FILE.parent
    OUTPUT_PREFIX = RESULTS_FILE.stem.replace("_results", "") + "_scored"
else:
    RESULTS_FILE = BASE_DIR / "work" / "04_experiment" / "04_local_experiment_results.json"
    OUTPUT_DIR = BASE_DIR / "work" / "04_experiment"
    OUTPUT_PREFIX = "05_quality_scoring_results"

# 채점 대상(qwen2.5:7b, llama3.2:3b)과 겹치지 않는 별도 judge 모델
# (docs/quality_scoring_spec.md §5.1 — judge 편향 방지)
JUDGE_MODEL = "qwen3:4b-instruct-2507-q4_K_M"

RUBRIC_VERSION = "v1-2026-09"   # 채점 규칙 자체의 버전 (아래 가중치·키워드·패턴)
EVAL_SET_VERSION = None          # 평가셋 파일에서 읽어 채운다 — load_gold() 참조
# 둘은 다른 것이다. 평가셋을 늘려도 채점 규칙은 그대로일 수 있고, 그 반대도 가능하다.
# 하나로 뭉쳐 두면 v2 골드로 채점한 기록이 v1로 남는다.

# 게이트 통과 후보의 가중합 (docs/quality_scoring_spec.md §4)
WEIGHTS = {"category_match": 0.40, "field_extraction_score": 0.35, "summary_fidelity_score": 0.25}

# 카테고리 근사 매칭용 키워드 사전 (Level 1/2 규칙 — 정확한 문자열 대신 키워드 포함 여부로 판정)
CATEGORY_KEYWORDS = {
    "로그인/계정": ["로그인", "계정", "비밀번호", "패스워드", "잠금", "잠김"],
    "결제/환불": ["결제", "환불", "결재", "페이먼트", "payment"],
    "기능 오류": ["오류", "에러", "버그", "실행", "기능"],
    "배송/주문": ["배송", "주문", "택배", "발송"],
    "컴플레인/보상요구": ["보상", "손해", "배상", "컴플레인", "불만"],
    "정보 부족": ["정보 부족", "불명확", "확인 필요", "고객 문의", "기타"],
    "기타(판단 불가)": ["기타", "불명확", "확인 필요"],
    # v2에서 추가된 gold 어휘. 없으면 category_match가 None으로 빠져
    # 해당 문항이 채점에서 조용히 제외된다(분모 축소).
    "기타": ["기타"],
}

# 과잉확답(환각) 1차 규칙 — 상담원 권한 밖 사항을 확정적으로 약속하는 표현
HALLUCINATION_PATTERNS = [
    r"보상\s*(해\s*드리겠습니다|하겠습니다|해드렸습니다|처리했습니다|완료했습니다|해드립니다)",
    r"환불\s*(해\s*드리겠습니다|하겠습니다|해드렸습니다|처리했습니다|완료했습니다|해드립니다)",
    r"전액\s*(지급|환불|보상)\s*(해\s*드리겠습니다|하겠습니다|완료했습니다)",
]


def load_gold():
    global EVAL_SET_VERSION
    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    EVAL_SET_VERSION = data.get("rubric_version")
    return {q["id"]: q for q in data["questions"]}


# ==========================================
# 1. 모델 응답 파서 — "- category: ..." 불릿 스타일과 순수 JSON 블록 스타일을 모두 처리
# ==========================================
def parse_response(text: str) -> dict:
    result = {"category": None, "summary": None, "info": None, "note": None, "format_valid": False}
    text = (text or "").strip()

    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_obj = None
    candidate = fence_match.group(1) if fence_match else text
    try:
        json_obj = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        json_obj = None

    if isinstance(json_obj, dict) and all(k in json_obj for k in ("category", "summary", "info", "note")):
        result.update({
            "category": str(json_obj.get("category") or ""),
            "summary": str(json_obj.get("summary") or ""),
            "info": json_obj.get("info"),
            "note": str(json_obj.get("note") or ""),
        })
        result["format_valid"] = True
        return result

    def extract_field(label):
        m = re.search(rf"-\s*{label}\s*:\s*(.+)", text)
        return m.group(1).strip() if m else None

    category = extract_field("category")
    summary = extract_field("summary")
    note = extract_field("note")
    info = json_obj if fence_match else extract_field("info")

    if category and summary is not None and note is not None:
        result.update({"category": category, "summary": summary, "info": info, "note": note})
        result["format_valid"] = True

    return result


# ==========================================
# 2. Level 1/2 규칙 기반 채점
# ==========================================
def rule_based_score(parsed: dict, gold: dict) -> dict:
    out = {"format_valid": parsed["format_valid"]}

    model_category = parsed["category"] or ""
    keywords = CATEGORY_KEYWORDS.get(gold["expected_category"], [])
    out["category_match"] = any(kw in model_category for kw in keywords) if keywords else None

    info_text = json.dumps(parsed["info"], ensure_ascii=False) if isinstance(parsed["info"], (dict, list)) else str(parsed["info"] or "")
    field_detail = {}
    for field in gold["expected_fields"]:
        # 필드 라벨의 핵심 키워드(괄호 앞부분)만 사용 — 예: "증상1(로그인 불가)" -> "증상1"
        key = re.split(r"[(（]", field)[0].strip()
        field_detail[field] = bool(key) and key in info_text
    out["field_extraction_detail"] = field_detail
    out["field_extraction_score"] = (
        sum(field_detail.values()) / len(field_detail) if field_detail else None
    )

    note_text = parsed["note"] or ""
    info_sufficiency = gold["taxonomy"]["info_sufficiency"]
    if info_sufficiency.startswith("특수"):
        abstention_keywords = ["추가", "정보", "확인", "권한", "불명확", "부족", "필요"]
        out["abstention_appropriate"] = any(kw in note_text for kw in abstention_keywords)
    else:
        out["abstention_appropriate"] = None

    full_text = " ".join(str(v) for v in [parsed["category"], parsed["summary"], note_text, info_text])
    out["hallucination_flag_rule"] = any(re.search(p, full_text) for p in HALLUCINATION_PATTERNS)

    return out


# ==========================================
# 3. Level 3 — LLM-as-judge (Ollama, JSON 모드)
# ==========================================
def call_judge(question_text: str, parsed: dict, gold: dict) -> dict:
    key_facts = gold["key_facts_for_summary"]
    expected_fields = gold["expected_fields"]
    info_text = json.dumps(parsed["info"], ensure_ascii=False) if isinstance(parsed["info"], (dict, list)) else str(parsed["info"] or "")
    prompt = f"""당신은 CS 문의 자동응답의 품질을 채점하는 평가자입니다. 아래 정보만 보고 JSON으로만 답하세요.
모델이 쓴 표현이 정답 라벨과 문자 그대로 같지 않아도, 같은 의미이면 "포함됨"으로 판정하세요.

[원본 문의]
{question_text}

[모델이 낸 요약]
{parsed['summary']}

[모델이 낸 note]
{parsed['note']}

[모델이 낸 info(핵심 정보 추출 결과)]
{info_text}

[필수 추출 필드 목록 (0부터 시작하는 인덱스)]
{json.dumps(list(enumerate(expected_fields)), ensure_ascii=False)}

[핵심 사실 목록 (0부터 시작하는 인덱스)]
{json.dumps(list(enumerate(key_facts)), ensure_ascii=False)}

[이 응답이 지켜야 할 note 기대 행동]
{gold['expected_note_behavior']}

다음 키를 가진 JSON 객체 하나만 출력하세요. 다른 설명은 쓰지 마세요.
- fields_covered: 필수 추출 필드 목록 중 info에 실제로(다른 표현이어도 같은 의미로) 포함된 것들의 인덱스 배열
- summary_facts_covered: 핵심 사실 목록 중 요약에 실제로 포함된 것들의 인덱스 배열 (핵심 사실이 없으면 빈 배열)
- summary_has_unsupported_claim: 요약이 원본 문의에 없는 사실을 지어냈으면 true, 아니면 false
- hallucination_confirmed: 응답이 상담원 권한 밖 사항(보상/환불 등)을 확정적으로 약속했으면 true, 아니면 false
- abstention_ok: note가 위 기대 행동을 충족하면 true, 충족하지 못하면 false"""

    payload = {
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        judged = json.loads(raw)
        return {
            "fields_covered": judged.get("fields_covered", []),
            "summary_facts_covered": judged.get("summary_facts_covered", []),
            "summary_has_unsupported_claim": judged.get("summary_has_unsupported_claim"),
            "hallucination_confirmed": judged.get("hallucination_confirmed"),
            "abstention_ok": judged.get("abstention_ok"),
            "judge_error": None,
        }
    except Exception as e:
        return {
            "fields_covered": None,
            "summary_facts_covered": None,
            "summary_has_unsupported_claim": None,
            "hallucination_confirmed": None,
            "abstention_ok": None,
            "judge_error": str(e),
        }


# ==========================================
# 4. 게이트 → 가중합 결합 (docs/quality_scoring_spec.md §4)
# ==========================================
def combine(rule: dict, judge: dict, gold: dict) -> dict:
    key_facts = gold["key_facts_for_summary"]
    facts_covered = judge["summary_facts_covered"]
    summary_fidelity_score = (
        len(facts_covered) / len(key_facts) if key_facts and isinstance(facts_covered, list)
        else (1.0 if not key_facts else None)
    )

    # 필드 추출: 규칙(문자 그대로 포함)과 judge(같은 의미면 포함) 중 하나라도 맞으면 인정
    # — 모델이 gold 라벨과 다른 표현("문제"↔"증상")을 써도 의미가 같으면 놓치지 않도록 함
    expected_fields = gold["expected_fields"]
    field_detail = dict(rule["field_extraction_detail"])
    judge_fields = judge["fields_covered"]
    if isinstance(judge_fields, list):
        for idx in judge_fields:
            if isinstance(idx, int) and 0 <= idx < len(expected_fields):
                field_detail[expected_fields[idx]] = True
    field_extraction_score = sum(field_detail.values()) / len(field_detail) if field_detail else None

    # 환각: 규칙 1차와 judge 2차가 불일치하면 자동 확정하지 않고 사람 검토로 넘김(None)
    rule_flag = rule["hallucination_flag_rule"]
    judge_flag = judge["hallucination_confirmed"]
    if judge_flag is None:
        hallucination_flag = rule_flag  # judge 실패 시 규칙 결과로 대체, 신뢰도 낮음을 needs_human_review로 표시
        needs_human_review = True
    elif rule_flag == judge_flag:
        hallucination_flag = rule_flag
        needs_human_review = False
    else:
        hallucination_flag = True  # 리스크가 큰 축이므로 불일치 시 보수적으로 위반 처리하고 사람 검토 요청
        needs_human_review = True

    abstention = rule["abstention_appropriate"]
    if abstention is not None and judge["abstention_ok"] is not None:
        abstention = abstention and judge["abstention_ok"]

    format_valid = rule.get("format_valid", False)
    overall_gate_pass = bool(format_valid) and not hallucination_flag and (abstention is not False)

    weighted_score = None
    if overall_gate_pass:
        parts = []
        if rule["category_match"] is not None:
            parts.append(("category_match", float(rule["category_match"])))
        if field_extraction_score is not None:
            parts.append(("field_extraction_score", field_extraction_score))
        if summary_fidelity_score is not None:
            parts.append(("summary_fidelity_score", summary_fidelity_score))
        total_weight = sum(WEIGHTS[k] for k, _ in parts)
        if total_weight > 0:
            weighted_score = sum(WEIGHTS[k] * v for k, v in parts) / total_weight

    return {
        "field_extraction_score": field_extraction_score,
        "field_extraction_detail": field_detail,
        "summary_fidelity_score": summary_fidelity_score,
        "summary_has_unsupported_claim": judge["summary_has_unsupported_claim"],
        "hallucination_flag": hallucination_flag,
        "abstention_appropriate": abstention,
        "overall_gate_pass": overall_gate_pass,
        "weighted_score": weighted_score,
        "needs_human_review": needs_human_review,
    }


def now_iso():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()


# ==========================================
# 5. 메인
# ==========================================
def main():
    gold_by_id = load_gold()
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    scored = []
    total = len(records)
    for i, r in enumerate(records, start=1):
        print(f"[{i}/{total}] {r['model_name']} q{r['question_id']} run{r['run_index']}... ", end="", flush=True)

        gold = gold_by_id[r["question_id"]]
        if not r["is_success"]:
            scored.append({**r, "format_valid": False, "overall_gate_pass": False,
                           "weighted_score": None, "grader": "rule", "rubric_version": RUBRIC_VERSION, "eval_set_version": EVAL_SET_VERSION,
                           "graded_at": now_iso(), "note_reason": "call_failed"})
            print("스킵(호출 실패)")
            continue

        parsed = parse_response(r["response_text"])
        rule = rule_based_score(parsed, gold)

        if not rule["format_valid"]:
            scored.append({**r, **rule, "overall_gate_pass": False, "weighted_score": None,
                           "grader": "rule", "rubric_version": RUBRIC_VERSION, "eval_set_version": EVAL_SET_VERSION, "graded_at": now_iso()})
            print("게이트 탈락(형식 오류)")
            continue

        judge = call_judge(r["prompt"], parsed, gold)
        combined = combine(rule, judge, gold)

        scored.append({
            **r,
            "parsed_category": parsed["category"],
            "parsed_summary": parsed["summary"],
            "parsed_note": parsed["note"],
            **rule,
            **combined,
            "grader": f"rule+llm_judge:{JUDGE_MODEL}",
            "rubric_version": RUBRIC_VERSION, "eval_set_version": EVAL_SET_VERSION,
            "graded_at": now_iso(),
            "judge_error": judge["judge_error"],
        })
        gate = "PASS" if combined["overall_gate_pass"] else "FAIL"
        score_str = f"{combined['weighted_score']:.2f}" if combined["weighted_score"] is not None else "-"
        print(f"게이트={gate} 점수={score_str}")
        time.sleep(0.1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_out = OUTPUT_DIR / f"{OUTPUT_PREFIX}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(scored)
    csv_out = OUTPUT_DIR / f"{OUTPUT_PREFIX}.csv"
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    print(f"\n저장됨: {json_out}, {csv_out}")

    summary = df.groupby("model_name").agg(
        n=("overall_gate_pass", "count"),
        gate_pass_rate=("overall_gate_pass", "mean"),
        avg_weighted_score=("weighted_score", "mean"),
        hallucination_count=("hallucination_flag", "sum"),
        needs_human_review=("needs_human_review", "sum"),
    ).reset_index()
    print("\n[모델별 채점 요약]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
