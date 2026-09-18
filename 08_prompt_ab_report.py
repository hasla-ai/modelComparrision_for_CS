"""프롬프트 보강 A/B 결과 분석 — 주 지표·회귀 지표 비교와 자동 판정.

08_prompt_ab_scored.csv를 읽어 설계서 §6 판정 규칙을 그대로 적용한다.
사람이 표를 보고 마음대로 해석하지 않도록, 판정 기준을 코드로 고정해 둔 것이다.

  주 지표   Q9·Q10 게이트 통과 (기권 적절성)
  회귀 지표 Q1~Q8 게이트 통과 / 분류 정확도 / 통과 건 평균 가중점수
  악화 정의 Q1~Q8에서 게이트 통과 건수 감소, 또는 분류 일치 감소,
            또는 통과 건 평균 가중점수 0.05 이상 하락

실행:  uv run python 08_prompt_ab_report.py
출력:  work/08_prompt_ab/08_prompt_ab_summary.md (표준 출력에도 같은 내용)
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "work" / "08_prompt_ab"
SCORED = WORK_DIR / "08_prompt_ab_scored.csv"
SUMMARY = WORK_DIR / "08_prompt_ab_summary.md"

SPECIAL_QIDS = [9, 10]          # 주 지표 대상 (정보 부족 / 범위 밖)
SCORE_DROP_THRESHOLD = 0.05     # 가중점수 악화 판정 임계값


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


def load() -> pd.DataFrame:
    if not SCORED.exists():
        print(f"채점 결과가 없습니다: {SCORED}")
        print("먼저 실행하세요:")
        print("  uv run python 05_quality_scoring.py work/08_prompt_ab/08_prompt_ab_results.json")
        sys.exit(1)
    df = pd.read_csv(SCORED, encoding="utf-8-sig")
    for col in ["overall_gate_pass", "category_match", "format_valid", "hallucination_flag"]:
        if col in df.columns:
            df[col] = as_bool(df[col])
    df["question_id"] = df["question_id"].astype(int)
    df["weighted_score"] = pd.to_numeric(df.get("weighted_score"), errors="coerce")
    return df


def block(df: pd.DataFrame) -> dict:
    """한 (모델, 조건) 묶음의 지표를 계산한다."""
    passed = df[df["overall_gate_pass"]]
    return {
        "n": len(df),
        "gate_pass": int(df["overall_gate_pass"].sum()),
        "category_match": int(df["category_match"].sum()) if "category_match" in df else 0,
        "format_valid": int(df["format_valid"].sum()) if "format_valid" in df else 0,
        "hallucination": int(df["hallucination_flag"].sum()) if "hallucination_flag" in df else 0,
        "mean_score": round(passed["weighted_score"].mean(), 3) if len(passed) else None,
        "mean_tokens": round(df["generated_tokens"].mean(), 1) if "generated_tokens" in df else None,
    }


def verdict(main_a: dict, main_b: dict, reg_a: dict, reg_b: dict) -> tuple[str, list[str]]:
    """설계서 §6 판정 규칙을 그대로 적용한다."""
    reasons = []

    main_delta = main_b["gate_pass"] - main_a["gate_pass"]

    # 회귀(악화) 판정 — 세 조건 중 하나라도 걸리면 악화
    regressed = []
    if reg_b["gate_pass"] < reg_a["gate_pass"]:
        regressed.append(f"게이트 통과 {reg_a['gate_pass']}→{reg_b['gate_pass']}")
    if reg_b["category_match"] < reg_a["category_match"]:
        regressed.append(f"분류 일치 {reg_a['category_match']}→{reg_b['category_match']}")
    if (reg_a["mean_score"] is not None and reg_b["mean_score"] is not None
            and reg_a["mean_score"] - reg_b["mean_score"] >= SCORE_DROP_THRESHOLD):
        regressed.append(f"평균 가중점수 {reg_a['mean_score']}→{reg_b['mean_score']}")

    if main_delta < 0:
        v = "역효과 — 규칙 문구가 오히려 해를 끼침. 원문 응답 확인 필요"
    elif main_delta == 0:
        v = ("P 주장 기각 — 프롬프트로는 개선되지 않음. "
             "이 부분은 모델 한계(M) 또는 채점 기준 문제")
    elif regressed:
        v = "조건부 성공 — 주 지표는 개선됐으나 회귀 발생. 규칙 완화 후 재설계 필요"
    else:
        v = "성공 — P 주장 입증. 배포 전 조치로 채택 가능"

    reasons.append(f"주 지표(Q9·Q10 게이트): {main_a['gate_pass']}/{main_a['n']} → "
                   f"{main_b['gate_pass']}/{main_b['n']} (Δ{main_delta:+d})")
    reasons.append("회귀: " + ("; ".join(regressed) if regressed else "없음"))
    return v, reasons


def main() -> int:
    df = load()
    models = sorted(df["model_name"].unique())
    L: list[str] = []
    add = L.append

    add("# 프롬프트 보강 A/B — 결과 요약")
    add("")
    add(f"조건 A = 현행 프롬프트 · 조건 B = 규칙 3개 추가 (기권 / 권한 / 환각)  ")
    add(f"판정 기준은 결과 열람 전에 확정됨 — PROMPT_AB_DESIGN.md §6")
    add("")

    for model in models:
        m = df[df["model_name"] == model]
        special = m[m["question_id"].isin(SPECIAL_QIDS)]
        normal = m[~m["question_id"].isin(SPECIAL_QIDS)]

        ma, mb = block(special[special["condition"] == "A"]), block(special[special["condition"] == "B"])
        ra, rb = block(normal[normal["condition"] == "A"]), block(normal[normal["condition"] == "B"])
        v, reasons = verdict(ma, mb, ra, rb)

        add(f"## {model}")
        add("")
        add(f"**판정: {v}**")
        add("")
        for r in reasons:
            add(f"- {r}")
        add("")
        add("### 주 지표 — Q9·Q10 (정보 부족 / 범위 밖)")
        add("")
        add("| 지표 | 조건 A | 조건 B | 변화 |")
        add("| --- | --- | --- | --- |")
        add(f"| 게이트 통과 | {ma['gate_pass']}/{ma['n']} | {mb['gate_pass']}/{mb['n']} | {mb['gate_pass']-ma['gate_pass']:+d} |")
        add(f"| 분류 일치 | {ma['category_match']}/{ma['n']} | {mb['category_match']}/{mb['n']} | {mb['category_match']-ma['category_match']:+d} |")
        add("")
        add("### 회귀 지표 — Q1~Q8 (정상·경계)")
        add("")
        add("| 지표 | 조건 A | 조건 B | 변화 |")
        add("| --- | --- | --- | --- |")
        add(f"| 게이트 통과 | {ra['gate_pass']}/{ra['n']} | {rb['gate_pass']}/{rb['n']} | {rb['gate_pass']-ra['gate_pass']:+d} |")
        add(f"| 분류 일치 | {ra['category_match']}/{ra['n']} | {rb['category_match']}/{rb['n']} | {rb['category_match']-ra['category_match']:+d} |")
        sa, sb = ra["mean_score"], rb["mean_score"]
        add(f"| 통과 건 평균 점수 | {sa} | {sb} | "
            f"{(round(sb-sa, 3) if sa is not None and sb is not None else '-')} |")
        add("")
        add("### 부작용 점검 (전 문항)")
        add("")
        aa, bb = block(m[m["condition"] == "A"]), block(m[m["condition"] == "B"])
        add("| 지표 | 조건 A | 조건 B |")
        add("| --- | --- | --- |")
        add(f"| 형식 준수 | {aa['format_valid']}/{aa['n']} | {bb['format_valid']}/{bb['n']} |")
        add(f"| 환각 판정 | {aa['hallucination']} | {bb['hallucination']} |")
        add(f"| 평균 생성 토큰 | {aa['mean_tokens']} | {bb['mean_tokens']} |")
        add("")

    add("## 문항별 게이트 통과 (조건 A → B)")
    add("")
    add("| 모델 | " + " | ".join(f"Q{q}" for q in range(1, 11)) + " |")
    add("| --- |" + " --- |" * 10)
    for model in models:
        cells = []
        for q in range(1, 11):
            sub = df[(df["model_name"] == model) & (df["question_id"] == q)]
            a = int(sub[sub["condition"] == "A"]["overall_gate_pass"].sum())
            b = int(sub[sub["condition"] == "B"]["overall_gate_pass"].sum())
            n = len(sub[sub["condition"] == "A"])
            mark = "" if a == b else (" ↑" if b > a else " ↓")
            cells.append(f"{a}/{n}→{b}/{n}{mark}")
        add(f"| {model} | " + " | ".join(cells) + " |")
    add("")
    add("## 해석 시 지켜야 할 것")
    add("")
    add("- 조건 B는 gold가 요구하는 문자열을 프롬프트에 넣었다. 개선은 "
        "**'모델 품질 향상'이 아니라 '명세 누락 보완'** 으로 읽어야 한다.")
    add("- Q9·Q10은 조건당 4건이다. 부분 변화(1/4→2/4)는 확정적 개선으로 말하지 않는다.")
    add("- 규칙은 기권·권한·환각만 건드리도록 작성했다. 정상 문항이 함께 오른 경우, "
        "그것을 규칙의 효과로 주장하지 않는다.")
    add("")

    out = "\n".join(L)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(out, encoding="utf-8")
    print(out)
    print(f"\n저장: {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
