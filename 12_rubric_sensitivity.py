"""채점 관대성 민감도 점검 — 결론이 채점 규칙에 얼마나 기대고 있는가.

왜 필요한가
  `05_quality_scoring.py` 의 분류 정확도 판정은 키워드 포함 여부로 한다.
  그 키워드 사전에서 "정보 부족" 항목은 이렇게 되어 있다.

      "정보 부족": ["정보 부족", "불명확", "확인 필요", "고객 문의", "기타"]

  뒤의 두 개 — "고객 문의", "기타" — 는 *정보가 부족하다는 진술이 아니다*.
  아무 문의에나 붙일 수 있는 기본 바구니다. 이걸 정답으로 세면,
  판단을 포기하고 전부 "기타"로 던지는 모델이 점수를 얻는다.
  부록 D에서 `llama3.2:3b` 가 정상 문항의 절반을 기권으로 빠뜨린 그 행동이다.

  실제로 본실험 Q9("안 됩니다.") 응답 4건은 "고객 문의" 2건, "기타" 2건이었고
  네 건 모두 분류 정답 처리됐다.

이 스크립트가 하는 일
  저장된 원본 응답을 그대로 쓰고 **채점 규칙만 바꿔** 다시 계산한다.
  Ollama 재호출이 없으므로 결과는 결정적이며 몇 초 안에 끝난다.

    관대(lenient) = 현행 규칙. 보고서 본문 수치가 이것이다.
    엄격(strict)  = "기타"/"고객 문의" 를 "정보 부족" 의 정답으로 세지 않는다.

  묻는 것은 하나다. **선정 결론이 뒤집히는가.**
  뒤집히지 않으면 결론은 채점 관대성에 기대고 있지 않다는 뜻이고,
  뒤집히면 보고서 수치를 엄격 기준으로 갈아야 한다.

  어느 쪽이든 이 스크립트는 판단하지 않는다. 판정 규칙은 §해석 규칙에
  미리 적어 두고 코드가 적용한다.

실행:
    uv run python 12_rubric_sensitivity.py
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
GOLD_FILE = os.path.join(BASE, "data", "02_prompts_and_eval.json")
OUT_DIR = os.path.join(BASE, "work", "12_rubric_sensitivity")

# 05_quality_scoring.py 와 동일한 가중치 (docs/quality_scoring_spec.md §4)
WEIGHTS = {"category_match": 0.40, "field_extraction_score": 0.35, "summary_fidelity_score": 0.25}

# 현행 규칙 — 05_quality_scoring.py 의 CATEGORY_KEYWORDS 와 같아야 한다
LENIENT = {
    "로그인/계정": ["로그인", "계정", "비밀번호", "패스워드", "잠금", "잠김"],
    "결제/환불": ["결제", "환불", "결재", "페이먼트", "payment"],
    "기능 오류": ["오류", "에러", "버그", "실행", "기능"],
    "배송/주문": ["배송", "주문", "택배", "발송"],
    "컴플레인/보상요구": ["보상", "손해", "배상", "컴플레인", "불만"],
    "정보 부족": ["정보 부족", "불명확", "확인 필요", "고객 문의", "기타"],
    "기타(판단 불가)": ["기타", "불명확", "확인 필요"],
    "기타": ["기타"],
}

# 엄격 규칙 — 딱 한 군데만 다르다. 알고리즘이 고른 게 아니라 손으로 골랐고,
# 고른 이유를 여기 적어 둔다.
#   "기타" / "고객 문의" 는 '정보가 부족하다'는 판단의 표현이 아니라
#   판단을 유보한 기본 바구니다. 정보 부족 gold 의 정답으로 세지 않는다.
#   대신 '정보 부족'이라는 판단을 실제로 드러낸 표현만 남긴다.
STRICT = dict(LENIENT)
STRICT["정보 부족"] = ["정보 부족", "정보부족", "확인 필요", "추가 정보", "불명확"]

# ── 해석 규칙 ────────────────────────────────────────────────────
#
# [1차 규칙 — 결과를 보기 전에 고정했고, 잘못 쓴 것으로 드러났다]
#   R1. "두 기준에서 모델 순위가 같으면" 결론은 관대성에 의존하지 않는다.
#   R2. "순위가 뒤집히면" 본문 수치를 엄격 기준으로 교체하고 전 실험을 재채점한다.
#
#   이 규칙은 실행 즉시 R2를 띄웠다. 프롬프트 A/B 실험에서 `qwen2.5:7b` 의
#   조건 A와 B의 대소가 바뀌었기 때문이다. 그런데 A와 B는 **같은 모델의
#   프롬프트 두 종**이지 선정 후보가 아니다. 선정 결론이 기대고 있는 순위는
#   본실험의 `llama3.2:3b` vs `qwen2.5:7b` 하나뿐인데, 1차 규칙은 모든 실험의
#   모든 그룹 쌍을 싸잡아 "순위"라고 불렀다. 규칙을 잘못 쓴 것이다.
#
#   결과를 본 뒤에 규칙을 고치는 것은 원래 해서는 안 되는 일이라, 지운 대신
#   위에 남기고 아래 2차 규칙과 양쪽 판정을 모두 출력한다. 읽는 사람이
#   무엇이 바뀌었는지 보고 직접 판단할 수 있어야 한다.
#
# [2차 규칙 — 적용 범위를 선정 순위로 좁힌 것]
#   S1. 본실험에서 두 후보의 순위가 두 기준에서 같고, 보고서가 공표한 수치가
#       바뀌지 않으면 → 선정 결론은 채점 관대성에 의존하지 않는다.
#       본문은 관대 기준을 유지하고 이 표를 강건성 근거로 첨부한다.
#   S2. 본실험 순위가 뒤집히거나 공표 수치가 바뀌면 → 본문을 엄격 기준으로
#       교체하고 전 실험을 재채점한다.
#   S3. 그 밖의 실험에서 대소가 바뀌면 → 재채점 사유는 아니지만 해당 부록에
#       그 사실을 적는다. 해석이 채점 규칙에 걸려 있다는 뜻이기 때문이다.
#   S4. 게이트 통과 수가 달라지면 → 기준과 무관하게 게이트 판정을 재검토한다.
#       (게이트는 분류 정확도를 직접 쓰지 않으므로 달라지지 않아야 정상이다.)
# ─────────────────────────────────────────────────────────────────

# 선정 결론이 기대고 있는 단 하나의 순위
SELECTION_SOURCE = "본실험 (로컬 40회)"

# 보고서가 실제로 공표한 수치의 계산 기준 — 부록 D는 Q1~Q8 · 게이트 통과
# 레코드만으로 평균을 낸다. 아래 표의 "전 레코드" 평균과는 분모가 다르므로,
# 공표 수치가 바뀌었는지는 그쪽 기준으로 따로 확인해야 한다.
PUBLISHED_BASIS = {
    "questions": [str(i) for i in range(1, 9)],
    "gate_passed_only": True,
}

SOURCES = [
    ("본실험 (로컬 40회)", "work/04_experiment/05_quality_scoring_results.csv", ["model_name"]),
    ("Cloud 비교 (5회)", "work/06_cloud_comparison/06_cloud_comparison_scored.csv", ["model_name"]),
    ("양자화 (60회)", "work/07_quantization/07_quantization_scored.csv", ["variant_label"]),
    ("프롬프트 A/B (80회)", "work/08_prompt_ab/08_prompt_ab_scored.csv", ["model_name", "condition"]),
]


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def matched(model_category: str, gold_category: str, table: dict) -> bool | None:
    kws = table.get(gold_category, [])
    if not kws:
        return None
    return any(k in (model_category or "") for k in kws)


def weighted(cat_match, field_score, summary_score):
    """05_quality_scoring.py 와 같은 방식 — None 축은 분모에서도 뺀다."""
    vals = {
        "category_match": None if cat_match is None else (1.0 if cat_match else 0.0),
        "field_extraction_score": field_score,
        "summary_fidelity_score": summary_score,
    }
    num = den = 0.0
    for k, w in WEIGHTS.items():
        if vals[k] is not None:
            num += w * vals[k]
            den += w
    return num / den if den else None


def load_gold():
    with open(GOLD_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rubric_version"), {str(q["id"]): q for q in data["questions"]}


def analyse(path: str, group_keys: list[str], gold: dict):
    full = os.path.join(BASE, path)
    if not os.path.exists(full):
        return None
    with open(full, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    agg = defaultdict(lambda: {"len": [], "str": [], "gate": [], "flip": 0, "n": 0})
    for r in rows:
        qid = str(r.get("question_id", "")).strip()
        g = gold.get(qid)
        if not g:
            continue
        key = " / ".join(r.get(k, "") or "" for k in group_keys)
        a = agg[key]
        a["n"] += 1
        a["gate"].append(str(r.get("overall_gate_pass")) == "True")

        # 형식 탈락·호출 실패 레코드는 원래도 점수가 없다 — 그대로 둔다
        ws = fnum(r.get("weighted_score"))
        if ws is None:
            continue

        pc = r.get("parsed_category") or ""
        fs = fnum(r.get("field_extraction_score"))
        ss = fnum(r.get("summary_fidelity_score"))
        cm_len = matched(pc, g["expected_category"], LENIENT)
        cm_str = matched(pc, g["expected_category"], STRICT)

        a["len"].append(weighted(cm_len, fs, ss))
        a["str"].append(weighted(cm_str, fs, ss))
        if cm_len != cm_str:
            a["flip"] += 1

    out = []
    for key, a in agg.items():
        vl = [v for v in a["len"] if v is not None]
        vs = [v for v in a["str"] if v is not None]
        out.append({
            "group": key,
            "n": a["n"],
            "scored": len(vl),
            "lenient": sum(vl) / len(vl) if vl else None,
            "strict": sum(vs) / len(vs) if vs else None,
            "flipped": a["flip"],
            "gate_pass": sum(a["gate"]),
        })
    return sorted(out, key=lambda d: d["group"])


def rank(rows):
    scored = [r for r in rows if r["strict"] is not None]
    return (
        [r["group"] for r in sorted(scored, key=lambda d: -d["lenient"])],
        [r["group"] for r in sorted(scored, key=lambda d: -d["strict"])],
    )


def main() -> int:
    if not os.path.exists(GOLD_FILE):
        print(f"골드 파일이 없습니다: {GOLD_FILE}")
        return 1
    eval_version, gold = load_gold()

    # 현행 사전이 05 와 어긋나면 이 점검 자체가 무의미하다 — 먼저 확인한다
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("qs", os.path.join(BASE, "05_quality_scoring.py"))
        qs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qs)
        if qs.CATEGORY_KEYWORDS != LENIENT:
            print("경고: 05_quality_scoring.py 의 CATEGORY_KEYWORDS 와 이 파일의 LENIENT 가 다릅니다.")
            print("      민감도 점검의 기준선이 본채점과 어긋나므로 먼저 맞추세요.")
            only_a = {k: v for k, v in qs.CATEGORY_KEYWORDS.items() if LENIENT.get(k) != v}
            only_b = {k: v for k, v in LENIENT.items() if qs.CATEGORY_KEYWORDS.get(k) != v}
            print(f"      05 쪽: {only_a}\n      여기 : {only_b}")
            return 1
    except Exception as exc:  # 점검은 보조 장치 — 실패해도 본작업은 계속한다
        print(f"(참고) 기준선 대조를 건너뜁니다: {type(exc).__name__}: {exc}")

    L = []
    w = L.append
    w("# 채점 관대성 민감도 점검")
    w("")
    w(f"평가셋 `{eval_version}` · 저장된 원본 응답 재채점 (모델 재호출 없음)")
    w("")
    w("분류 정확도 판정에서 **\"기타\"·\"고객 문의\"를 「정보 부족」의 정답으로 세지 않으면** ")
    w("결론이 달라지는지 본다. 나머지 축(형식·필드·요약)과 게이트 판정은 건드리지 않았다.")
    w("")

    print(f"평가셋: {eval_version}\n")
    verdicts = []

    for title, path, keys in SOURCES:
        rows = analyse(path, keys, gold)
        if not rows:
            w(f"## {title}\n\n결과 파일 없음 — 건너뜀\n")
            print(f"[건너뜀] {title}")
            continue

        w(f"## {title}")
        w("")
        w("| 구분 | n | 관대 | 엄격 | 차이 | 판정 뒤집힌 레코드 | 게이트 통과 |")
        w("| --- | --- | --- | --- | --- | --- | --- |")
        for r in rows:
            le = f"{r['lenient']:.3f}" if r["lenient"] is not None else "—"
            st = f"{r['strict']:.3f}" if r["strict"] is not None else "—"
            df = f"{r['strict'] - r['lenient']:+.3f}" if None not in (r["lenient"], r["strict"]) else "—"
            w(f"| {r['group']} | {r['n']} | {le} | {st} | {df} | {r['flipped']} | {r['gate_pass']}/{r['n']} |")
        w("")

        rl, rs = rank(rows)
        same = rl == rs
        verdicts.append((title, same))
        w(f"- 관대 기준 순위: {' > '.join(rl) if rl else '—'}")
        w(f"- 엄격 기준 순위: {' > '.join(rs) if rs else '—'}")
        w(f"- **순위 {'불변' if same else '변동'}**")
        w("")

        print(f"[{title}]")
        for r in rows:
            le = f"{r['lenient']:.3f}" if r["lenient"] is not None else "—"
            st = f"{r['strict']:.3f}" if r["strict"] is not None else "—"
            print(f"  {r['group']:<28} n={r['n']:<4} 관대 {le}  엄격 {st}  뒤집힘 {r['flipped']}")
        print(f"  → 순위 {'불변' if same else '변동'}\n")

    # 게으른 모델 노출도 — 평가셋이 커지면 이 구멍도 같이 커진다
    w("## 「기타」만 답하는 모델의 무임승차 폭")
    w("")
    w("모든 문항에 `category: 기타` 라고만 답하는 가상의 모델이 분류 축에서 정답 처리받는 비율이다. ")
    w("높을수록 채점기가 판단 포기를 보상하고 있다는 뜻이다.")
    w("")
    w("| 기준 | 정답 처리 문항 | 비율 |")
    w("| --- | --- | --- |")
    total = len(gold)
    for label, table in (("관대(현행)", LENIENT), ("엄격", STRICT)):
        hit = [i for i, g in gold.items() if matched("기타", g["expected_category"], table)]
        w(f"| {label} | {len(hit)}/{total} | {len(hit) / total:.1%} |")
        print(f"「기타」만 답할 때 정답 처리 — {label}: {len(hit)}/{total} ({len(hit)/total:.1%})")
    w("")

    # ── 공표 수치 기준 재계산 ─────────────────────────────────────
    # 위 표의 평균은 전 레코드 기준이다. 보고서가 실제로 인쇄한 숫자는
    # Q1~Q8 · 게이트 통과 레코드 기준이므로, 그쪽이 바뀌었는지 따로 본다.
    pub = published_basis_check(gold)
    w("## 보고서가 공표한 수치는 바뀌는가")
    w("")
    w("위 표의 평균은 **전 레코드** 기준이고, 부록 D가 인쇄한 숫자는 ")
    w("**Q1~Q8 · 게이트 통과 레코드** 기준이다. 분모가 다르므로 값이 다른 것이 정상이다. ")
    w("공표 수치가 흔들렸는지는 아래 표로 판단한다.")
    w("")
    w("| 실험 | 구분 | 관대(공표값) | 엄격 | 차이 |")
    w("| --- | --- | --- | --- | --- |")
    for e in pub:
        w(f"| {e['source']} | {e['group']} | {e['lenient']:.3f} | {e['strict']:.3f} | {e['strict'] - e['lenient']:+.3f} |")
    w("")
    published_changed = [e for e in pub if abs(e["strict"] - e["lenient"]) >= 0.001]
    w(f"**공표 수치 변동: {len(published_changed)}건**"
      + ("" if published_changed else " — 관대성 구멍은 Q9에만 있었고, Q9는 공표 수치의 분모에 들어가지 않는다."))
    w("")

    # ── 판정 ─────────────────────────────────────────────────────
    sel = next((v for t, v in verdicts if t == SELECTION_SOURCE), None)
    others = [(t, v) for t, v in verdicts if t != SELECTION_SOURCE]
    all_same = all(v for _, v in verdicts)
    s_ok = bool(sel) and not published_changed

    w("## 판정")
    w("")
    w("### 1차 규칙(사전 고정) — 그대로 적용한 결과")
    w("")
    w(f"모든 실험의 모든 그룹 쌍을 순위로 본다는 규칙이었다. 판정: "
      f"**{'R1 — 순위 불변' if all_same else 'R2 — 순위 변동, 전 실험 재채점'}**")
    if not all_same:
        for t, v in verdicts:
            if not v:
                w(f"  - 변동: {t}")
    w("")
    w("### 이 규칙을 그대로 따르지 않는 이유")
    w("")
    w("A/B 실험의 조건 A·B는 **같은 모델의 프롬프트 두 종**이지 선정 후보가 아니다. ")
    w("선정 결론이 기대고 있는 순위는 본실험의 두 후보 하나뿐인데, 1차 규칙은 그 구분 없이 ")
    w("모든 그룹 쌍을 순위라고 불렀다. 규칙을 잘못 쓴 쪽이 문제라서 적용 범위를 좁혔다. ")
    w("**결과를 본 뒤 규칙을 고친 것이므로, 1차 판정을 지우지 않고 위에 남긴다.**")
    w("")
    w("### 2차 규칙 — 적용 범위를 선정 순위로 좁힌 판정")
    w("")
    if s_ok:
        w("**S1 적용** — 본실험의 후보 순위가 두 기준에서 같고, 공표 수치도 바뀌지 않았다. ")
        w("선정 결론은 채점 관대성에 의존하지 않는다. 본문은 관대 기준을 유지하고 이 표를 첨부한다.")
    else:
        w("**S2 적용** — 본실험 순위가 뒤집혔거나 공표 수치가 바뀌었다. 본문을 엄격 기준으로 ")
        w("교체하고 전 실험을 재채점해야 한다.")
    w("")
    for t, v in others:
        if not v:
            w(f"- **S3** — {t}: 대소가 바뀌었다. 재채점 사유는 아니지만 해당 부록에 적는다.")
    w("")
    w("### 그래도 남는 문제")
    w("")
    w("순위가 안 바뀌는 것과 점수가 맞는 것은 다른 말이다. 위 무임승차 표대로 현행 채점기는 ")
    w("판단 포기를 일부 보상한다. v1에서는 정보 부족 문항이 1개뿐이라 그 구멍이 공표 수치에 ")
    w("닿지 않았을 뿐이고, v2에서는 10개다. **다음 실행부터는 엄격 기준을 기본값으로 쓴다.**")
    w("")

    os.makedirs(OUT_DIR, exist_ok=True)
    md_path = os.path.join(OUT_DIR, "12_rubric_sensitivity.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(f"공표 수치 변동: {len(published_changed)}건")
    print(f"\n기록: {os.path.relpath(md_path, BASE)}")
    print("1차 규칙 판정:", "R1 — 순위 불변" if all_same else "R2 — 순위 변동")
    print("2차 규칙 판정:", "S1 — 본문 유지" if s_ok else "S2 — 재채점 필요")
    return 0


def published_basis_check(gold: dict):
    """보고서가 인쇄한 계산 기준(Q1~Q8 · 게이트 통과)으로 두 규칙을 비교한다."""
    out = []
    for title, path, keys in SOURCES:
        full = os.path.join(BASE, path)
        if not os.path.exists(full):
            continue
        with open(full, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        buckets = defaultdict(lambda: {"len": [], "str": []})
        for r in rows:
            qid = str(r.get("question_id", "")).strip()
            if qid not in PUBLISHED_BASIS["questions"]:
                continue
            if PUBLISHED_BASIS["gate_passed_only"] and str(r.get("overall_gate_pass")) != "True":
                continue
            g = gold.get(qid)
            if not g:
                continue
            pc = r.get("parsed_category") or ""
            fs, ss = fnum(r.get("field_extraction_score")), fnum(r.get("summary_fidelity_score"))
            key = " / ".join(r.get(k, "") or "" for k in keys)
            a = weighted(matched(pc, g["expected_category"], LENIENT), fs, ss)
            b = weighted(matched(pc, g["expected_category"], STRICT), fs, ss)
            if a is not None:
                buckets[key]["len"].append(a)
            if b is not None:
                buckets[key]["str"].append(b)
        for key, v in sorted(buckets.items()):
            if v["len"] and v["str"]:
                out.append({"source": title, "group": key,
                            "lenient": sum(v["len"]) / len(v["len"]),
                            "strict": sum(v["str"]) / len(v["str"])})
    return out


if __name__ == "__main__":
    raise SystemExit(main())
