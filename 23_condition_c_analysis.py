"""조건 C 결정적 분석 — LLM judge 없이 **지금 확정할 수 있는 것만** 센다.

왜 별도 스크립트인가
  `05_quality_scoring.py` 는 648건마다 judge 모델을 한 번씩 부른다. 수십 분이 걸리고
  judge 의 변덕이 섞인다. 그런데 조건 C 가 묻는 질문 — "허용 카테고리를 열거하면
  분류가 나아지는가" — 의 핵심은 **결정적으로 셀 수 있다.** 모델이 뱉은 category
  문자열과 gold 를 비교하는 일에는 judge 가 필요 없다.

  그래서 이 스크립트를 먼저 돌린다. judge 채점(요약 충실도·필드 추출)은 나중에
  붙여도 여기서 나온 분류 결론을 뒤집지 못한다. 분모가 다른 별개의 축이다.

이 스크립트가 절대 하지 않는 것
  가중 종합 점수를 내지 않는다. 가중치는 `field_extraction` 과 `summary_fidelity`
  를 포함하는데 그 둘은 judge 없이 확정되지 않는다. 여기서 종합 점수를 내면
  가중치 세 개 중 하나만 채우고 100점 만점이라 부르는 셈이다.

규칙 동기화
  파서와 관대 키워드 사전은 `05_quality_scoring.py` 에서, 엄격 사전은
  `12_rubric_sensitivity.py` 에서 **import 한다.** 베껴 오지 않는다. 베끼면
  원본이 바뀌었을 때 이 스크립트만 옛 규칙으로 남아 조용히 다른 답을 낸다.

실행:
    uv run python 23_condition_c_analysis.py
    uv run python 23_condition_c_analysis.py work/22_condition_c/22_condition_c_results.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_RESULTS = BASE / "work" / "22_condition_c" / "22_condition_c_results.json"
GOLD_FILE = BASE / "data" / "02_prompts_and_eval.json"
OUT_DIR = BASE / "work" / "22_condition_c"


# ══════════════════════════════════════════════════════════════════
# 사전 등록 — 해석 규칙
#
# 아래 규칙은 이 분석을 **돌리기 전에** 적었고, 적은 채로 커밋한 뒤 실행했다.
# 판정은 코드가 한다. 사람이 표를 보고 고르지 않는다.
#
# 다만 정직하게 적어 둘 것이 하나 있다.
#   T1(어휘 고정)은 이미 답을 알고 있는 상태에서 쓴 규칙이다.
#   `22_condition_c_experiment.py` 가 실행 직후 출력한 요약에 어휘 수
#   48→8, 50→8 이 이미 찍혀 있었다. 그러므로 T1 은 **발견이 아니라 확인**이며,
#   아래 표에서도 그렇게 표시한다. T2~T6 은 결과를 보기 전의 규칙이다.
#
#   T1. 어휘 고정 — 조건 C 의 `허용목록_내` 비율이 조건 A 보다 높지 않으면
#       열거는 실패다. (사후 확인 항목. 발견으로 세지 않는다.)
#
#   T2. 분류 개선 — 열거가 **분류**를 개선했다고 말하려면 `엄격_일치` 가
#       올라야 한다. 허용목록 준수만 오르고 엄격 일치가 오르지 않으면,
#       열거가 고친 것은 *기계 가독성*이지 *분류*가 아니다. 그렇게 적는다.
#       이것이 이 실험의 가장 그럴듯한 오독이므로 규칙으로 못 박는다.
#
#   T3. 관대-엄격 격차 — (관대 − 엄격) 격차가 A 에서 크고 C 에서 작아지면,
#       A 의 관대 점수 일부는 '어휘가 느슨해 키워드에 우연히 걸린' 몫이다.
#       격차가 줄어드는 것 자체는 좋은 신호가 아니라 **A 수치의 과대평가**
#       증거다. 부록 H 의 결론(공표 수치 불변)과 충돌하는지 확인한다.
#
#   T4. 과잉 보수화 재현 — 부록 D 는 조건 B 에서 `llama3.2:3b` 가 정상 문항
#       16건 중 8건을 「정보 부족」으로 흘렸다고 적었다. 그 수치는 **다른
#       환경에서 잰 값**이다(부록 L). 같은 실행의 B 로 재현되는지 본다.
#       재현되지 않으면 부록 D 의 B 수치를 이 실행 값으로 교체한다.
#
#   T5. 열거의 대가 — 조건 C 가 (a) 정상 문항을 「정보 부족」으로 미는 비율,
#       (b) 형식 준수율 중 하나라도 A 보다 나빠지면 "열거는 공짜가 아니다"로
#       적는다. 좋아지기만 한 실험은 대개 덜 본 실험이다.
#
#   T6. 지연은 판정에 쓰지 않는다 — 이 실행 중 사용자 PC 에서 영상 두 편이
#       재생되고 있었다(본인 고지). 지연·TPS 열은 **기록만** 하고 게이트
#       판정에 쓰지 않는다. 기록하는 이유는 나중에 누가 이 CSV 를 게이트에
#       쓰려 할 때 오염 사실이 같은 파일 안에 있어야 하기 때문이다.
#
# 판정 임계
#   "올랐다/나빠졌다"는 **퍼센트포인트 2.0 이상**의 변화로 정의한다.
#   54문항 × 2패스 = 108 시행이므로 1건이 약 0.93%p 다. 2.0%p 는 2건 이상,
#   즉 한 건의 우연으로는 넘지 못하는 선이다. 통계적 검정이 아니라
#   **선을 미리 그어 둔 것**이며, 그 이상을 주장하지 않는다.
# ══════════════════════════════════════════════════════════════════
DELTA_PP = 2.0

# `22_condition_c_experiment.py` 의 ALLOWED_CATEGORIES 와 같아야 한다 — import 한다
CONTAMINATION_NOTE = (
    "이 실행 중 사용자 PC 에서 영상 두 편(인터넷 강의·스포츠)이 재생되고 있었다. "
    "지연·TPS 열은 기록용이며 게이트 판정에 쓰지 않는다(T6)."
)


def load_module(filename: str, name: str):
    """숫자로 시작하는 모듈명은 import 문으로 못 부른다 — 파일 경로로 연다."""
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pct(num: int, den: int) -> float | None:
    return round(100.0 * num / den, 1) if den else None


def fmt(v, suffix="%"):
    return "—" if v is None else f"{v:.1f}{suffix}"


def delta(c, a):
    if c is None or a is None:
        return None
    return round(c - a, 1)


def arrow(d):
    if d is None:
        return "—"
    if d >= DELTA_PP:
        return f"+{d:.1f}%p ↑"
    if d <= -DELTA_PP:
        return f"{d:.1f}%p ↓"
    return f"{d:+.1f}%p ="


# ══════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=str(DEFAULT_RESULTS))
    args = ap.parse_args()

    scoring = load_module("05_quality_scoring.py", "qs05")
    sens = load_module("12_rubric_sensitivity.py", "rs12")
    exp = load_module("22_condition_c_experiment.py", "cc22")

    parse_response = scoring.parse_response
    LENIENT = scoring.CATEGORY_KEYWORDS
    STRICT = sens.STRICT
    ALLOWED = exp.ALLOWED_CATEGORIES
    CONDITIONS = exp.CONDITIONS

    # 규칙 동기화 확인 — 12 의 LENIENT 가 05 의 사전과 갈라졌으면 여기서 멈춘다
    if sens.LENIENT != LENIENT:
        print("[중단] 12_rubric_sensitivity.LENIENT 와 05_quality_scoring."
              "CATEGORY_KEYWORDS 가 다릅니다. 두 파일을 먼저 맞추십시오.")
        for k in sorted(set(LENIENT) | set(sens.LENIENT)):
            if LENIENT.get(k) != sens.LENIENT.get(k):
                print(f"    {k}: 05={LENIENT.get(k)}  12={sens.LENIENT.get(k)}")
        return 2

    with open(GOLD_FILE, encoding="utf-8") as f:
        gold_data = json.load(f)
    gold = {q["id"]: q for q in gold_data["questions"]}
    eval_set_version = gold_data.get("rubric_version")

    with open(args.results, encoding="utf-8") as f:
        records = json.load(f)

    env = {}
    env_path = Path(args.results).parent / "22_env.json"
    if env_path.exists():
        env = json.loads(env_path.read_text(encoding="utf-8"))

    # ── 레코드별 판정 ────────────────────────────────────────────
    rows = []
    for r in records:
        qid = str(r["question_id"])
        g = gold.get(qid)
        if g is None:
            continue
        parsed = parse_response(r.get("response_text", ""))
        cat = (parsed["category"] or "").strip()
        exp_cat = g["expected_category"]
        suff = g["taxonomy"]["info_sufficiency"]

        # 완전 일치 — 문자열이 gold 와 같은가.
        # 예외 하나: gold 「기타(판단 불가)」는 프롬프트에 「기타」로만 제시된다
        # (22_condition_c_experiment.verify_enumeration 의 주석 참조).
        # 그 흡수를 인정하지 않으면 조건 C 를 구조적으로 불리하게 만든다.
        exact = (cat == exp_cat) or (exp_cat == "기타(판단 불가)" and cat == "기타")

        lk = LENIENT.get(exp_cat, [])
        sk = STRICT.get(exp_cat, [])
        rows.append({
            "model_name": r["model_name"],
            "condition": r["condition"],
            "question_id": qid,
            "run_index": r.get("run_index"),
            "expected_category": exp_cat,
            "info_sufficiency": suff,
            "is_normal": not str(suff).startswith("특수"),
            "model_category": cat,
            "is_success": bool(r.get("is_success")),
            "format_valid": bool(parsed["format_valid"]),
            "in_allowed": cat in ALLOWED,
            "exact_match": bool(exact),
            "lenient_match": any(k in cat for k in lk) if lk else None,
            "strict_match": any(k in cat for k in sk) if sk else None,
            # 정상 문항인데 「정보 부족」으로 흘렸는가 — 과잉 보수화 지표
            "false_insufficient": (not str(suff).startswith("특수"))
                                  and ("정보 부족" in cat or "정보부족" in cat),
            "total_time_sec": r.get("total_time_sec"),
            "generated_tokens": r.get("generated_tokens"),
            "tps": r.get("tps"),
        })

    models = sorted({x["model_name"] for x in rows})
    conds = [c for c in ["A", "B", "C"] if any(x["condition"] == c for x in rows)]

    def agg(model, cond, key, only=None):
        sel = [x for x in rows if x["model_name"] == model and x["condition"] == cond
               and x["is_success"] and (only is None or only(x))]
        vals = [x[key] for x in sel if x[key] is not None]
        return pct(sum(1 for v in vals if v), len(vals)), len(vals)

    def nums(model, cond, key):
        v = [x[key] for x in rows if x["model_name"] == model and x["condition"] == cond
             and x["is_success"] and isinstance(x[key], (int, float))]
        return v

    # ── 리포트 ───────────────────────────────────────────────────
    L = ["# 부록 M — 조건 C: 프롬프트에 허용 카테고리를 열거하면 무엇이 달라지는가",
         "",
         "**결정적 분석 (LLM judge 없음).** 이 문서의 모든 수치는 저장된 응답 문자열과",
         "gold 를 비교해 다시 셀 수 있다. judge 채점(요약 충실도·필드 추출)은 별도이며",
         "여기 결론을 뒤집지 못한다 — 분모가 다른 축이다.", ""]

    if env:
        L += [f"측정 환경 — 지문 `{env.get('fingerprint', '?')}` · "
              f"ollama {env.get('ollama', {}).get('version', '?')} · "
              f"{env.get('gpu', {}).get('name', '?')}", ""]
    L += [f"평가셋 `{eval_set_version}` · 문항 {len(gold)} · 조건 {', '.join(conds)} · "
          f"레코드 {len(rows)}건", "",
          f"> {CONTAMINATION_NOTE}", ""]

    # M.1 어휘
    L += ["## M.1 어휘 — 모델이 실제로 쓴 카테고리 문자열", "",
          "`허용목록_내` 는 모델이 뱉은 category 가 프롬프트에 열거한 7개 중 하나와",
          "**문자 그대로 같은** 비율이다. 자동 라우팅에 그대로 꽂을 수 있는가를 뜻한다.", "",
          "| 모델 | 조건 | 서로 다른 어휘 | 허용목록_내 | A 대비 |",
          "| --- | --- | ---: | ---: | --- |"]
    for m in models:
        base, _ = agg(m, "A", "in_allowed")
        for c in conds:
            vocab = sorted({x["model_category"] for x in rows
                            if x["model_name"] == m and x["condition"] == c
                            and x["is_success"] and x["model_category"]})
            v, n = agg(m, c, "in_allowed")
            d = arrow(delta(v, base)) if c != "A" else "기준"
            L.append(f"| {m} | {c} ({CONDITIONS[c]['label']}) | {len(vocab)} | "
                     f"{fmt(v)} ({n}건) | {d} |")
    L.append("")

    # M.2 분류 정확도 3종
    L += ["## M.2 분류 정확도 — 세 가지 기준으로 나란히", "",
          "| 기준 | 무엇을 세는가 |",
          "| --- | --- |",
          "| 완전일치 | category 문자열이 gold 와 같다 (「기타(판단 불가)」→「기타」 흡수 인정) |",
          "| 관대 | 현행 채점 규칙. 키워드 포함이면 정답 (보고서 본문 수치의 기준) |",
          "| 엄격 | 「기타」·「고객 문의」를 「정보 부족」의 정답으로 세지 않음 (부록 H) |",
          "",
          "| 모델 | 조건 | 완전일치 | 관대 | 엄격 | 관대−엄격 |",
          "| --- | --- | ---: | ---: | ---: | ---: |"]
    for m in models:
        for c in conds:
            e, _ = agg(m, c, "exact_match")
            le, _ = agg(m, c, "lenient_match")
            st, _ = agg(m, c, "strict_match")
            gap = delta(le, st)
            L.append(f"| {m} | {c} | {fmt(e)} | {fmt(le)} | {fmt(st)} | "
                     f"{'—' if gap is None else f'{gap:.1f}%p'} |")
    L.append("")

    # M.3 사전 등록 규칙 판정
    L += ["## M.3 사전 등록 규칙 판정", "",
          f"임계: 퍼센트포인트 {DELTA_PP} 이상의 변화만 '변화'로 센다 "
          f"(1건 ≈ 0.93%p, 즉 2건 이상).", "",
          "| 모델 | 규칙 | 지표 | A | C | 판정 |",
          "| --- | --- | --- | ---: | ---: | --- |"]
    verdicts = {}
    for m in models:
        a_al, _ = agg(m, "A", "in_allowed")
        c_al, _ = agg(m, "C", "in_allowed")
        a_st, _ = agg(m, "A", "strict_match")
        c_st, _ = agg(m, "C", "strict_match")
        a_fv, _ = agg(m, "A", "format_valid")
        c_fv, _ = agg(m, "C", "format_valid")
        a_fi, _ = agg(m, "A", "false_insufficient", only=lambda x: x["is_normal"])
        c_fi, _ = agg(m, "C", "false_insufficient", only=lambda x: x["is_normal"])
        a_gap = delta(*[agg(m, "A", k)[0] for k in ("lenient_match", "strict_match")])
        c_gap = delta(*[agg(m, "C", k)[0] for k in ("lenient_match", "strict_match")])

        d_al, d_st = delta(c_al, a_al), delta(c_st, a_st)
        t1 = "어휘 고정됨" if (d_al or 0) >= DELTA_PP else "**열거 실패**"
        t2 = ("분류도 개선" if (d_st or 0) >= DELTA_PP
              else ("**분류 악화**" if (d_st or 0) <= -DELTA_PP
                    else "**분류는 그대로 — 열거가 고친 것은 기계 가독성**"))
        t3 = ("A 관대 점수에 어휘 느슨함 몫 있음"
              if (a_gap or 0) - (c_gap or 0) >= DELTA_PP else "격차 변화 없음")
        cost = []
        if (delta(c_fi, a_fi) or 0) >= DELTA_PP:
            cost.append("정상 문항을 「정보 부족」으로 더 흘림")
        if (delta(c_fv, a_fv) or 0) <= -DELTA_PP:
            cost.append("형식 준수 하락")
        t5 = "**열거는 공짜가 아니다** — " + "; ".join(cost) if cost else "대가 없음"

        L += [f"| {m} | T1 어휘 고정 *(사후 확인)* | 허용목록_내 | {fmt(a_al)} | "
              f"{fmt(c_al)} | {arrow(d_al)} · {t1} |",
              f"| {m} | T2 분류 개선 | 엄격 일치 | {fmt(a_st)} | {fmt(c_st)} | "
              f"{arrow(d_st)} · {t2} |",
              f"| {m} | T3 관대−엄격 격차 | 격차 | "
              f"{'—' if a_gap is None else f'{a_gap:.1f}%p'} | "
              f"{'—' if c_gap is None else f'{c_gap:.1f}%p'} | {t3} |",
              f"| {m} | T5 열거의 대가 | 형식 / 정상→정보부족 | "
              f"{fmt(a_fv)} / {fmt(a_fi)} | {fmt(c_fv)} / {fmt(c_fi)} | {t5} |"]
        verdicts[m] = {"T1": t1, "T2": t2, "T3": t3, "T5": t5,
                       "d_allowed": d_al, "d_strict": d_st}
    L.append("")

    # M.4 T4 — 조건 B 과잉 보수화 재현
    if "B" in conds:
        L += ["## M.4 T4 — 부록 D 의 과잉 보수화가 같은 실행에서 재현되는가", "",
              "부록 D 의 B 수치는 **다른 환경에서 잰 값**이다(부록 L). 같은 표에",
              "놓으려면 같은 실행에서 다시 재야 한다. 그래서 B 를 이번 실행에 넣었다.", "",
              "정상 문항(특수 아님)만 분모로 센다.", "",
              "| 모델 | 조건 | 정상 문항 중 「정보 부족」 오분류 | 건수 |",
              "| --- | --- | ---: | ---: |"]
        for m in models:
            for c in conds:
                v, n = agg(m, c, "false_insufficient", only=lambda x: x["is_normal"])
                cnt = sum(1 for x in rows if x["model_name"] == m and x["condition"] == c
                          and x["is_success"] and x["is_normal"] and x["false_insufficient"])
                L.append(f"| {m} | {c} ({CONDITIONS[c]['label']}) | {fmt(v)} | {cnt}/{n} |")
        L.append("")

    # M.5 패스 간 안정성
    L += ["## M.5 패스 간 안정성", "",
          "같은 문항·같은 조건을 2회 돌렸다. 두 패스의 차이가 크면 아래 수치는",
          "규칙의 차이가 아니라 샘플링 잡음을 보고 있는 것이다.", "",
          "| 모델 | 조건 | 패스1 엄격 | 패스2 엄격 | 차 |",
          "| --- | --- | ---: | ---: | ---: |"]
    for m in models:
        for c in conds:
            ps = []
            for p in (0, 1):
                sel = [x for x in rows if x["model_name"] == m and x["condition"] == c
                       and x["is_success"] and x["run_index"] == p
                       and x["strict_match"] is not None]
                ps.append(pct(sum(1 for x in sel if x["strict_match"]), len(sel)))
            d = delta(ps[1], ps[0])
            L.append(f"| {m} | {c} | {fmt(ps[0])} | {fmt(ps[1])} | "
                     f"{'—' if d is None else f'{d:+.1f}%p'} |")
    L.append("")

    # M.6 지연 (기록만)
    L += ["## M.6 지연·처리량 — 기록만, 판정에 쓰지 않음", "",
          f"> {CONTAMINATION_NOTE}", "",
          "| 모델 | 조건 | P50 지연 | P95 지연 | 평균 출력 토큰 | 평균 TPS |",
          "| --- | --- | ---: | ---: | ---: | ---: |"]
    for m in models:
        for c in conds:
            t = sorted(nums(m, c, "total_time_sec"))
            tok = nums(m, c, "generated_tokens")
            tp = nums(m, c, "tps")
            if not t:
                continue
            p50 = statistics.median(t)
            p95 = t[min(len(t) - 1, int(round(0.95 * (len(t) - 1))))]
            L.append(f"| {m} | {c} | {p50:.2f}s | {p95:.2f}s | "
                     f"{statistics.mean(tok):.1f} | {statistics.mean(tp):.1f} |")
    L.append("")

    # M.7 하지 않을 말
    L += ["## M.7 이 분석이 말할 수 없는 것", "",
          "- **요약 품질·필드 추출은 여기 없다.** judge 채점이 붙기 전까지 조건 C 의",
          "  종합 점수는 존재하지 않는다. 분류 축 하나만 본 것이다.",
          "- **지연 비교 불가.** T6 의 오염 때문이다. 조건 A/B/C 는 가장 안쪽 루프에서",
          "  교대하므로 부하를 똑같이 나눠 가졌고, 따라서 *조건 간 상대 비교*는 그나마",
          "  공평하지만 *절대값*은 다른 표에 옮길 수 없다.",
          "- **프롬프트 한 줄의 효과만 잰 것이다.** 열거 문구의 위치·표현을 바꾸면",
          "  결과가 달라질 수 있다. 이 실험은 '열거하면 달라진다'를 보였을 뿐",
          "  '이 문구가 최적이다'를 보이지 않았다.",
          "- **gold 어휘 자체가 정리되지 않았다.** 「기타」와 「기타(판단 불가)」가",
          "  둘 다 gold 에 있어 프롬프트에 그대로 옮길 수 없었다. 다음 루브릭",
          "  버전에서 정리해야 하며, 그전까지 완전일치 수치에는 이 흡수가 섞여 있다.",
          ""]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "23_condition_c_analysis.md").write_text("\n".join(L), encoding="utf-8")

    with open(OUT_DIR / "23_condition_c_rows.csv", "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── 콘솔 ─────────────────────────────────────────────────────
    print("=" * 70)
    print(f"조건 C 결정적 분석 — 레코드 {len(rows)}건 · 평가셋 {eval_set_version}")
    print("=" * 70)
    for m in models:
        print(f"\n{m}")
        print(f"  {'조건':<4}{'허용목록':>10}{'완전일치':>10}{'관대':>10}{'엄격':>10}"
              f"{'형식':>10}{'정상→정보부족':>14}")
        for c in conds:
            al, _ = agg(m, c, "in_allowed")
            e, _ = agg(m, c, "exact_match")
            le, _ = agg(m, c, "lenient_match")
            st, _ = agg(m, c, "strict_match")
            fv, _ = agg(m, c, "format_valid")
            fi, _ = agg(m, c, "false_insufficient", only=lambda x: x["is_normal"])
            print(f"  {c:<4}{fmt(al):>10}{fmt(e):>10}{fmt(le):>10}{fmt(st):>10}"
                  f"{fmt(fv):>10}{fmt(fi):>14}")
        v = verdicts[m]
        print(f"  T1 {v['T1']}")
        print(f"  T2 {v['T2']}")
        print(f"  T5 {v['T5']}")
    print(f"\n기록: work/22_condition_c/23_condition_c_analysis.md")
    print(f"      work/22_condition_c/23_condition_c_rows.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
