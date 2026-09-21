"""조건 C judge 채점 결과 읽기 — 그리고 종합 점수의 구조적 공백을 먼저 밝힌다.

왜 이 스크립트가 따로 있는가
  `05_quality_scoring.py` 는 종합 점수(`weighted_score`)를 낸다. 그 값을 그대로
  표에 옮기기 전에 확인해야 할 것이 있다. **평가셋 v2 의 10문항은 종합 점수의
  두 축이 비어 있다.**

    key_facts_for_summary 가 빈 문항 : 9, 14, 15, 22, 29, 30, 37, 38, 46, 53 (10건)
    expected_fields 가 빈 문항       : 위 10건 + 8, 43, 45, 51 (총 14건)

  `05_quality_scoring.combine()` 은 이렇게 처리한다.

    · key_facts 가 비면  summary_fidelity_score = 1.0   (덮을 사실이 없으니 만점)
    · expected_fields 가 비면 field_extraction_score = None → 가중치에서 제외

  둘 다 빈 10문항에서 종합 점수는 다음으로 붕괴한다.

      weighted_score = (0.40 × category_match + 0.25 × 1.0) / 0.65
                     = 0.615 × category_match + 0.385

  즉 **분류를 틀려도 0.385 를 받는다.** 그리고 그 10문항이 바로 「정보 부족」
  gold 전부다 — 기권을 시험하려고 만든 칸에서 요약 축이 공짜로 만점을 준다.

  1.0 이라는 처리 자체가 틀린 것은 아니다. 덮을 사실이 없으면 놓친 것도 없다.
  문제는 그 만점이 **한 칸에만 몰려 있다**는 것이고, 그래서 칸을 가로질러
  평균을 내는 순간 그 칸이 실제보다 좋아 보인다. 부록 H 가 분류 축에서 찾은
  것과 같은 종류의 구멍이 요약 축에도 있다.

  그래서 이 스크립트는 종합 점수를 **두 벌로** 낸다.
    전체      — 54문항 전부. 05_quality_scoring.py 가 내는 값과 같다.
    공백 제외 — 두 축이 다 살아 있는 40문항만. 축 세 개가 실제로 채점된 값이다.

  어느 쪽도 '옳은 값'이라고 부르지 않는다. 둘을 나란히 두고 차이를 본다.

실행:
    uv run python 24_condition_c_scored_readout.py
    uv run python 24_condition_c_scored_readout.py work/22_condition_c/22_condition_c_scored.csv
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DEFAULT_SCORED = BASE / "work" / "22_condition_c" / "22_condition_c_scored.csv"
GOLD_FILE = BASE / "data" / "02_prompts_and_eval.json"
OUT = BASE / "work" / "22_condition_c" / "24_condition_c_scored_readout.md"

CONDITION_LABEL = {"A": "현행 (baseline)", "B": "보강 (규칙 3개)", "C": "카테고리 열거"}


def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.{nd}f}"


def pctf(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scored", nargs="?", default=str(DEFAULT_SCORED))
    args = ap.parse_args()

    path = Path(args.scored)
    if not path.exists():
        print(f"[중단] 채점 결과가 없습니다: {path}")
        print("       먼저 실행하십시오:")
        print("       uv run python 05_quality_scoring.py "
              "work/22_condition_c/22_condition_c_results.json")
        return 1

    gold = {q["id"]: q for q in json.loads(GOLD_FILE.read_text(encoding="utf-8"))["questions"]}
    no_kf = {i for i, g in gold.items() if not g.get("key_facts_for_summary")}
    no_ef = {i for i, g in gold.items() if not g.get("expected_fields")}
    hollow = sorted(no_kf | no_ef)       # 축이 하나라도 빈 문항
    solid = [i for i in gold if i not in no_kf and i not in no_ef]

    df = pd.read_csv(path, encoding="utf-8-sig")
    df["question_id"] = df["question_id"].astype(int)
    df["is_solid"] = df["question_id"].isin(solid)

    models = sorted(df["model_name"].unique())
    conds = [c for c in ["A", "B", "C"] if c in set(df["condition"])]

    def sub(m, c, solid_only=False):
        d = df[(df["model_name"] == m) & (df["condition"] == c)]
        return d[d["is_solid"]] if solid_only else d

    def mean_ws(d):
        v = pd.to_numeric(d["weighted_score"], errors="coerce").dropna()
        return float(v.mean()) if len(v) else None

    def rate(d, col):
        if col not in d.columns or not len(d):
            return None
        v = d[col].map({True: 1, False: 0, "True": 1, "False": 0})
        v = pd.to_numeric(v, errors="coerce").dropna()
        return float(v.mean()) if len(v) else None

    L = ["# 부록 M 보유 — 조건 C judge 채점 결과", "",
         "`05_quality_scoring.py` 가 낸 종합 점수를 읽는다. **읽기 전에 밝힐 것이 하나 있다.**", ""]

    # ── 구조적 공백 고지 ─────────────────────────────────────────
    L += ["## 종합 점수의 구조적 공백", "",
          f"평가셋 v2 의 **{len(no_kf)}문항**은 `key_facts_for_summary` 가 비어 있고,",
          f"**{len(no_ef)}문항**은 `expected_fields` 가 비어 있다. 채점기는 전자에",
          "요약 충실도 **1.0(만점)** 을 주고, 후자는 가중치에서 **제외**한다.", "",
          "둘 다 빈 문항에서 종합 점수는 이렇게 붕괴한다.", "",
          "```",
          "weighted_score = (0.40 × category_match + 0.25 × 1.0) / 0.65",
          "               = 0.615 × category_match + 0.385",
          "```", "",
          f"**분류를 틀려도 0.385 를 받는다.** 그리고 해당 {len(no_kf & no_ef)}문항은",
          "「정보 부족」 gold **전부**다 — 기권을 시험하려고 만든 칸에서 요약 축이",
          "공짜로 만점을 준다. 1.0 이라는 처리가 틀린 것은 아니다(덮을 사실이 없으면",
          "놓친 것도 없다). 문제는 그 만점이 **한 칸에만 몰려 있다**는 것이고, 칸을",
          "가로질러 평균을 내는 순간 그 칸이 실제보다 좋아 보인다. 부록 H 가 분류",
          "축에서 찾은 것과 같은 종류의 구멍이 요약 축에도 있다.", "",
          f"해당 문항: {', '.join(str(i) for i in hollow)}", "",
          f"그래서 아래 표는 **전체 {len(gold)}문항**과 **축이 다 살아 있는 "
          f"{len(solid)}문항**을 나란히 낸다. 어느 쪽도 '옳은 값'이라 부르지 않는다.", ""]

    # ── 종합 점수 ────────────────────────────────────────────────
    L += ["## 종합 점수 — 두 벌", "",
          "| 모델 | 조건 | 게이트 통과 | 종합(전체 54) | 종합(공백 제외 "
          f"{len(solid)}) | 차 |",
          "| --- | --- | ---: | ---: | ---: | ---: |"]
    for m in models:
        for c in conds:
            d_all, d_sol = sub(m, c), sub(m, c, True)
            g = rate(d_all, "overall_gate_pass")
            a, s = mean_ws(d_all), mean_ws(d_sol)
            diff = (a - s) if (a is not None and s is not None) else None
            L.append(f"| {m} | {c} ({CONDITION_LABEL.get(c, c)}) | {pctf(g)} | "
                     f"{fmt(a)} | {fmt(s)} | {'—' if diff is None else f'{diff:+.3f}'} |")
    L.append("")

    # ── 축별 ─────────────────────────────────────────────────────
    L += ["## 축별 — 어디서 점수가 나오는가", "",
          "| 모델 | 조건 | 분류 일치 | 필드 추출 | 요약 충실도 | 형식 | 환각 |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for m in models:
        for c in conds:
            d = sub(m, c)
            fe = pd.to_numeric(d.get("field_extraction_score"), errors="coerce").dropna()
            sf = pd.to_numeric(d.get("summary_fidelity_score"), errors="coerce").dropna()
            L.append(f"| {m} | {c} | {pctf(rate(d, 'category_match'))} | "
                     f"{fmt(float(fe.mean()) if len(fe) else None)} | "
                     f"{fmt(float(sf.mean()) if len(sf) else None)} | "
                     f"{pctf(rate(d, 'format_valid'))} | "
                     f"{pctf(rate(d, 'hallucination_flag'))} |")
    L.append("")

    # ── judge 건전성 ─────────────────────────────────────────────
    L += ["## judge 건전성", ""]
    if "judge_error" in df.columns:
        errs = df["judge_error"].notna() & (df["judge_error"].astype(str).str.strip() != "")
        n_err = int(errs.sum())
        L.append(f"- judge 호출 실패 **{n_err}건** / {len(df)}건"
                 + ("" if n_err == 0 else " — 해당 레코드는 규칙 채점으로 대체됐고"
                                          " `needs_human_review` 가 붙는다."))
        if n_err:
            top = df.loc[errs, "judge_error"].astype(str).value_counts().head(3)
            for k, v in top.items():
                L.append(f"    - {v}건: `{k[:110]}`")
    if "needs_human_review" in df.columns:
        nh = rate(df, "needs_human_review")
        L.append(f"- 사람 검토 필요로 표시된 비율 **{pctf(nh)}** — 규칙과 judge 의"
                 " 환각 판정이 엇갈렸거나 judge 가 실패한 건이다.")
    L.append("")

    # ── 분류 축 대조 ─────────────────────────────────────────────
    L += ["## 이 채점이 부록 M(결정적 분석)과 어긋나지 않는가", "",
          "`23_condition_c_analysis.py` 는 judge 없이 같은 응답의 분류 축을 셌다.",
          "두 값이 어긋나면 한쪽이 틀린 것이므로 여기서 대조한다.",
          "(관대 기준끼리 비교한다 — 05 의 `category_match` 가 관대 규칙이다.)", "",
          "| 모델 | 조건 | 05 채점 | 23 결정적 분석 | 차 |",
          "| --- | --- | ---: | ---: | ---: |"]
    det_path = BASE / "work" / "22_condition_c" / "23_condition_c_rows.csv"
    det = pd.read_csv(det_path, encoding="utf-8-sig") if det_path.exists() else None
    for m in models:
        for c in conds:
            a = rate(sub(m, c), "category_match")
            b = None
            if det is not None:
                dd = det[(det["model_name"] == m) & (det["condition"] == c)
                         & (det["is_success"].astype(str) == "True")]
                v = dd["lenient_match"].map({True: 1, False: 0, "True": 1, "False": 0})
                v = pd.to_numeric(v, errors="coerce").dropna()
                b = float(v.mean()) if len(v) else None
            d = (a - b) if (a is not None and b is not None) else None
            flag = "" if (d is None or abs(d) < 0.02) else "  ⚠ **불일치**"
            L.append(f"| {m} | {c} | {pctf(a)} | {pctf(b)} | "
                     f"{'—' if d is None else f'{100 * d:+.1f}%p'}{flag} |")
    L += ["",
          "> 차이가 2%p 를 넘으면 두 스크립트가 같은 것을 세고 있지 않다는 뜻이다.",
          "> 분모(게이트 탈락·호출 실패 처리)부터 맞춰 보아야 한다.", ""]

    OUT.write_text("\n".join(L), encoding="utf-8")

    print("=" * 70)
    print(f"조건 C judge 채점 읽기 — {len(df)}행")
    print("=" * 70)
    print(f"\n구조적 공백: key_facts 빈 문항 {len(no_kf)} · expected_fields 빈 문항 "
          f"{len(no_ef)} · 축 온전한 문항 {len(solid)}/{len(gold)}")
    print(f"  둘 다 빈 {len(no_kf & no_ef)}문항은 「정보 부족」 gold 전부 — "
          f"분류를 틀려도 0.385 를 받는다.\n")
    for m in models:
        print(f"{m}")
        print(f"  {'조건':<4}{'게이트':>9}{'종합(전체)':>13}{'종합(공백제외)':>16}")
        for c in conds:
            d_all, d_sol = sub(m, c), sub(m, c, True)
            print(f"  {c:<4}{pctf(rate(d_all, 'overall_gate_pass')):>9}"
                  f"{fmt(mean_ws(d_all)):>13}{fmt(mean_ws(d_sol)):>16}")
        print()
    print(f"기록: {OUT.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
