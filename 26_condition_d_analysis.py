"""조건 D·R 결정적 분석 — 설계서의 해석 규칙 D1~D8 을 코드가 판정한다.

설계서: `CONDITION_D_DESIGN.md` §7 측정 · §8 해석 규칙

judge 없이 셀 수 있는 축만 본다(축 1~4). 축 5(종합 점수·게이트)는 judge 채점
후 `24_condition_c_scored_readout.py` 가 낸다. 부록 M.8 에서 종합 점수와
게이트가 반대로 움직인 전례가 있으므로, 두 산출물은 **항상 함께 읽는다.**

규칙 동기화
  파서·관대 사전은 `05_quality_scoring.py`, 엄격 사전은 `12_rubric_sensitivity.py`,
  조건 정의는 `25_condition_d_experiment.py` 에서 **import 한다.**

실행:
    uv run python 26_condition_d_analysis.py
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
DEFAULT_RESULTS = BASE / "work" / "25_condition_d" / "25_condition_d_results.json"
GOLD_FILE = BASE / "data" / "02_prompts_and_eval.json"
OUT_DIR = BASE / "work" / "25_condition_d"

DELTA_PP = 2.0        # 설계서 §8 — 1건 ≈ 0.93%p, 즉 2건 이상
DELTA_CHARS = 2.0     # 설계서 D3·D8 — note 길이 변화 임계 (자)
SKEW_RATIO = 2.0      # 설계서 D4 — gold 몫의 몇 배면 쏠림인가
RECOVER_FRAC = 0.80   # 설계서 D3·D8 — 기준 조건의 몇 %면 '회복'인가

# 설계서 §3.3 · D7 — 기권 판정 분모에서 사전에 뺀 문항.
# gold 「기타(판단 불가)」는 프롬프트에 「기타」로만 제시되므로 이 문항들에서는
# 기권 판단의 정오를 물을 수 없다. **결과를 보기 전에** 선언했다.
EXCLUDED_FROM_ABSTENTION = [8, 51, 52]

# ── note 자리채움 판정 (실행 전에 목록을 고정한다) ─────────────────
#
# 부록 M.8 의 수치를 고치다 알게 된 것: 조건 C 의 짧은 note 상당수는 내용이
# 아니라 자리채움이었다. 가장 많은 값이 「정보 부족」(95회), `None`(61회),
# 「없음」(28회), 그리고 **출력 형식 설명문을 그대로 베낀 것**(16회)이다.
#
# 그래서 길이만 세면 안 된다. 자리채움을 따로 센다. 목록은 **결과를 보기 전에**
# 여기 박아 둔다 — 나중에 늘리면 원하는 답을 만들 수 있기 때문이다.
# 목록을 늘려야 할 값이 나오면 늘리되, 늘렸다는 사실과 그 값을 함께 적는다.
NOTE_PLACEHOLDERS = {
    "none", "null", "n/a", "na", "nan", "-", "없음", "없습니다", "해당 없음",
    "해당없음", "정보 부족", "정보부족", "특이사항 없음", "예외 없음",
}
# 출력 형식 설명문을 그대로 베낀 것 — 프롬프트에 있는 문구다
NOTE_ECHO = "정보 부족 또는 예외 사항이 있으면 기록"


def is_placeholder(note: str) -> bool:
    n = note.strip().rstrip(".。").lower()
    return (not n) or (n in NOTE_PLACEHOLDERS) or (NOTE_ECHO in note)


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pct(num, den):
    return round(100.0 * num / den, 1) if den else None


def fmt(v, suf="%", nd=1):
    return "—" if v is None else f"{v:.{nd}f}{suf}"


def delta(a, b):
    return None if (a is None or b is None) else round(a - b, 1)


def arrow(d, unit="%p"):
    if d is None:
        return "—"
    if d >= (DELTA_PP if unit == "%p" else DELTA_CHARS):
        return f"+{d:.1f}{unit} ↑"
    if d <= -(DELTA_PP if unit == "%p" else DELTA_CHARS):
        return f"{d:.1f}{unit} ↓"
    return f"{d:+.1f}{unit} ="


# ══════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=str(DEFAULT_RESULTS))
    args = ap.parse_args()

    if not Path(args.results).exists():
        print(f"[중단] 결과 파일이 없습니다: {args.results}")
        print("       먼저: uv run python 25_condition_d_experiment.py")
        return 1

    scoring = load_module("05_quality_scoring.py", "qs05")
    sens = load_module("12_rubric_sensitivity.py", "rs12")
    exp = load_module("25_condition_d_experiment.py", "cd25")

    parse_response = scoring.parse_response
    LENIENT, STRICT = scoring.CATEGORY_KEYWORDS, sens.STRICT
    ALLOWED = exp.ALLOWED_CATEGORIES
    CONDITIONS = exp.CONDITIONS
    DEPLOY = exp.DEPLOY_CANDIDATES

    if sens.LENIENT != LENIENT:
        print("[중단] 05_ 와 12_ 의 관대 사전이 갈라졌습니다. 먼저 맞추십시오.")
        return 2

    gold_data = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    gold = {q["id"]: q for q in gold_data["questions"]}
    eval_set_version = gold_data.get("rubric_version")
    records = json.loads(Path(args.results).read_text(encoding="utf-8"))

    env = {}
    ep = Path(args.results).parent / "25_env.json"
    if ep.exists():
        env = json.loads(ep.read_text(encoding="utf-8"))

    # ── 레코드별 판정 ────────────────────────────────────────────
    rows = []
    for r in records:
        g = gold.get(r["question_id"])
        if g is None:
            continue
        p = parse_response(r.get("response_text", ""))
        cat = (p["category"] or "").strip()
        note = (p["note"] or "").strip()
        exp_cat = g["expected_category"]
        suff = str(g["taxonomy"]["info_sufficiency"])
        lk, sk = LENIENT.get(exp_cat, []), STRICT.get(exp_cat, [])
        rows.append({
            "model_name": r["model_name"], "condition": r["condition"],
            "question_id": r["question_id"], "run_index": r.get("run_index"),
            "expected_category": exp_cat, "info_sufficiency": suff,
            "model_category": cat,
            "is_success": bool(r.get("is_success")),
            "format_valid": bool(p["format_valid"]),
            "in_allowed": cat in ALLOWED,
            "exact_match": (cat == exp_cat) or (exp_cat == "기타(판단 불가)" and cat == "기타"),
            "lenient_match": any(k in cat for k in lk) if lk else None,
            "strict_match": any(k in cat for k in sk) if sk else None,
            # 기권 축 — gold 가 「정보 부족」인가 아닌가로만 가른다
            "gold_is_insufficient": exp_cat == "정보 부족",
            "said_insufficient": ("정보 부족" in cat) or ("정보부족" in cat),
            "abstention_judgeable": r["question_id"] not in EXCLUDED_FROM_ABSTENTION,
            "note_text": note, "note_len": len(note), "note_empty": len(note) == 0,
            "note_placeholder": is_placeholder(note),
            "total_time_sec": r.get("total_time_sec"),
            "generated_tokens": r.get("generated_tokens"),
            "prompt_chars": r.get("prompt_chars"),
        })

    models = sorted({x["model_name"] for x in rows})
    conds = [c for c in ["A", "B", "C", "D", "R"] if any(x["condition"] == c for x in rows)]

    def sel(m, c, only=None):
        return [x for x in rows if x["model_name"] == m and x["condition"] == c
                and x["is_success"] and (only is None or only(x))]

    def rate(m, c, key, only=None):
        v = [x[key] for x in sel(m, c, only) if x[key] is not None]
        return pct(sum(1 for i in v if i), len(v)), len(v)

    def mean(m, c, key, only=None):
        v = [x[key] for x in sel(m, c, only) if isinstance(x[key], (int, float))]
        return round(statistics.mean(v), 1) if v else None

    # 기권 축 분모 — 설계서 D7
    over_filter = (lambda x: (not x["gold_is_insufficient"]) and x["abstention_judgeable"])
    recall_filter = (lambda x: x["gold_is_insufficient"])

    def over_abstain(m, c):
        s = sel(m, c, over_filter)
        return pct(sum(1 for x in s if x["said_insufficient"]), len(s)), len(s)

    def insuff_recall(m, c):
        s = sel(m, c, recall_filter)
        return pct(sum(1 for x in s if x["strict_match"]), len(s)), len(s)

    n_over = len([q for q in gold.values()
                  if q["expected_category"] != "정보 부족"]) - len(EXCLUDED_FROM_ABSTENTION)

    # ── 리포트 ───────────────────────────────────────────────────
    L = ["# 부록 N — 조건 D·R: 열거 + 기권 기준 + note 규칙, 그리고 순서 반전", "",
         "**결정적 분석 (LLM judge 없음).** 설계서 `CONDITION_D_DESIGN.md` §8 의",
         "해석 규칙 D1~D8 을 코드가 판정한다. 축 5(종합 점수·게이트)는 judge 채점 후",
         "`24_condition_c_scored_readout.py` 가 낸다 — **두 산출물은 함께 읽는다**",
         "(부록 M.8 에서 종합 점수와 게이트가 반대로 움직였다).", ""]
    if env:
        L += [f"측정 환경 — 지문 `{env.get('fingerprint', '?')}` · "
              f"ollama {env.get('ollama', {}).get('version', '?')} · "
              f"{env.get('gpu', {}).get('name', '?')}"]
        if env.get("contention_changed"):
            L.append("")
            L.append("> **실행 중 GPU 점유가 변했다.** 설계서 D6 에 따라 지연은 판정에 쓰지 않는다.")
        L.append("")
    L += [f"평가셋 `{eval_set_version}` · 문항 {len(gold)} · 조건 {', '.join(conds)} · "
          f"레코드 {len(rows)}건 (호출 실패 "
          f"{sum(1 for x in rows if not x['is_success'])}건 제외하고 집계)", "",
          f"> **조건 R 은 배포 후보가 아니다**(설계서 §4.1). 기전 판정(D8)에만 쓰고,",
          f"> 배포 후보 비교(D1~D4)에서는 뺀다. 표에는 참고로 함께 인쇄한다.", ""]

    # N.1 어휘
    L += ["## N.1 어휘 — 허용 목록 준수", "",
          "| 모델 | 조건 | 서로 다른 어휘 | 허용목록_내 | A 대비 |",
          "| --- | --- | ---: | ---: | --- |"]
    for m in models:
        base, _ = rate(m, "A", "in_allowed")
        for c in conds:
            v, n = rate(m, c, "in_allowed")
            vocab = {x["model_category"] for x in sel(m, c) if x["model_category"]}
            tag = "기준" if c == "A" else arrow(delta(v, base))
            star = "" if c in DEPLOY else " *(기전)*"
            L.append(f"| {m} | {c}{star} | {len(vocab)} | {fmt(v)} ({n}건) | {tag} |")
    L.append("")

    # N.2 분류
    L += ["## N.2 분류 정확도", "",
          "| 모델 | 조건 | 완전일치 | 관대 | 엄격 | 관대−엄격 |",
          "| --- | --- | ---: | ---: | ---: | ---: |"]
    for m in models:
        for c in conds:
            e = rate(m, c, "exact_match")[0]
            le = rate(m, c, "lenient_match")[0]
            st = rate(m, c, "strict_match")[0]
            gap = delta(le, st)
            star = "" if c in DEPLOY else " *(기전)*"
            L.append(f"| {m} | {c}{star} | {fmt(e)} | {fmt(le)} | {fmt(st)} | "
                     f"{'—' if gap is None else f'{gap:.1f}%p'} |")
    L.append("")

    # N.3 기권
    L += ["## N.3 기권 — 두 방향을 따로 센다", "",
          "설계서 §3.1: 「정보 부족」이 gold 인 것은 54문항 중 10문항뿐이고,",
          "나머지에서 기권은 오답이다. 그래서 두 지표가 필요하다.", "",
          f"- **재현율** — gold 「정보 부족」 10문항에서 실제로 기권했는가 (엄격 기준)",
          f"- **과잉 기권** — gold 가 「정보 부족」이 아닌 문항에서 기권해 버렸는가",
          f"  (분모 {n_over}문항 = 44 − 「기타(판단 불가)」 3문항. 설계서 D7 로 **사전 제외**)", "",
          "> 설계서 §7 축 3 은 이 분모를 「정상·경계」라고 적었으나 숫자 44 에는 "
          "「특수: 범위밖」 8문항도 들어 있다. 설계서의 **표현이 부정확**했고, 여기서는 "
          "**숫자 쪽(44−3=41)** 을 따랐다. gold 가 「정보 부족」이 아닌 모든 문항에서 "
          "기권은 오답이므로 그쪽이 맞다. 고친 사실을 지우지 않고 적어 둔다.", "",
          "| 모델 | 조건 | 재현율 (기권해야 할 때) | 과잉 기권 (하면 안 될 때) |",
          "| --- | --- | ---: | ---: |"]
    for m in models:
        for c in conds:
            rc, rn = insuff_recall(m, c)
            ov, on = over_abstain(m, c)
            star = "" if c in DEPLOY else " *(기전)*"
            L.append(f"| {m} | {c}{star} | {fmt(rc)} (n={rn}) | {fmt(ov)} (n={on}) |")
    L.append("")

    # N.4 note
    L += ["## N.4 note — 부록 M.8 이 찾아낸 축", "",
          "길이만 세면 안 된다. 짧은 note 의 상당수는 내용이 아니라 **자리채움**이다",
          "(`None`·「없음」·「정보 부족」, 또는 출력 형식 설명문을 그대로 베낀 것).",
          "자리채움 목록은 실행 전에 코드에 고정했다.", "",
          "| 모델 | 조건 | 평균 note 길이 | 빈 note | **자리채움** | 실질 note | 프롬프트 |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for m in models:
        for c in conds:
            star = "" if c in DEPLOY else " *(기전)*"
            ph = rate(m, c, "note_placeholder")[0]
            real = None if ph is None else round(100 - ph, 1)
            L.append(f"| {m} | {c}{star} | {fmt(mean(m, c, 'note_len'), '자')} | "
                     f"{fmt(rate(m, c, 'note_empty')[0])} | **{fmt(ph)}** | "
                     f"{fmt(real)} | {fmt(mean(m, c, 'prompt_chars'), '자', 0)} |")
    L += ["",
          "> C 와 R 의 프롬프트 길이는 **같다**(설계서 §10.1). 두 조건 사이의 note 차이는",
          "> 길이로 설명할 수 없다.",
          "> ",
          "> **`실질 note`가 진짜 지표다.** 부록 M.8 은 평균 길이만 봤는데, 그 수치는",
          "> 한 번 읽기 버그로 틀렸고(부록 M.8 의 정정 상자), 고친 뒤에도 자리채움과",
          "> 내용을 구분하지 못한다. 길이는 참고로 남기고 판정은 자리채움 비율로 본다.", ""]

    # N.5 사전 등록 판정
    L += ["## N.5 사전 등록 규칙 판정", "",
          f"임계: {DELTA_PP}%p (시행 2건 이상) · note 길이 {DELTA_CHARS}자 · "
          f"쏠림 gold 몫의 {SKEW_RATIO:g}배 · 회복선 기준 조건의 {RECOVER_FRAC:.0%}", "",
          "**비교 대상은 모두 이번 실행의 값이다.** 부록 M 의 A·B·C 는 오염된 실행이므로",
          "같은 표에 놓지 않는다(부록 L). 설계서 D2 가 괄호로 적은 부록 M 수치는",
          "참고용이었고, 판정은 이번 실행의 A·B 로 한다.", ""]
    verdicts = {}
    for m in models:
        a_ov = over_abstain(m, "A")[0]
        b_rc = insuff_recall(m, "B")[0]
        d_ov = over_abstain(m, "D")[0] if "D" in conds else None
        d_rc = insuff_recall(m, "D")[0] if "D" in conds else None
        c_al = rate(m, "C", "in_allowed")[0] if "C" in conds else None
        d_al = rate(m, "D", "in_allowed")[0] if "D" in conds else None
        a_nl, b_nl = mean(m, "A", "note_len"), mean(m, "B", "note_len")
        c_nl = mean(m, "C", "note_len") if "C" in conds else None
        d_nl = mean(m, "D", "note_len") if "D" in conds else None
        r_nl = mean(m, "R", "note_len") if "R" in conds else None

        # D1 — 어휘 유지
        d1 = ("판정 불가" if (c_al is None or d_al is None)
              else ("**블록 간섭** — D 가 C 보다 어휘 준수가 낮다"
                    if (delta(d_al, c_al) or 0) <= -DELTA_PP else "어휘 유지됨"))
        # D2 — 주 판정
        if None in (a_ov, b_rc, d_ov, d_rc):
            d2, d2a, d2b = "판정 불가", None, None
        else:
            d2a = d_ov <= a_ov + DELTA_PP
            d2b = d_rc >= b_rc - DELTA_PP
            d2 = ("**배포 후보 채택** — 두 방향 모두 만족" if (d2a and d2b)
                  else ("**조건부 실패** — 과잉 기권은 잡았으나 재현율이 무너짐"
                        if d2a else ("**조건부 실패** — 재현율은 지켰으나 과잉 기권 발생"
                                     if d2b else "**실패** — 두 방향 모두 불만족")))
        # D3 — note 회복
        if None in (c_nl, d_nl, b_nl):
            d3 = "판정 불가"
        elif d_nl - c_nl >= DELTA_CHARS and d_nl >= RECOVER_FRAC * b_nl:
            d3 = "note 회복"
        else:
            d3 = "**`category와 무관하게` 문구는 효과가 없었다**"
        # D8 — 기전
        if None in (c_nl, r_nl, a_nl):
            d8 = "판정 불가 (조건 R 없음)"
        elif r_nl - c_nl >= DELTA_CHARS and r_nl >= RECOVER_FRAC * a_nl:
            d8 = "**H-위치 채택** — 제약된 필드 뒤의 자유 서술 필드가 소진된다"
        elif r_nl - c_nl < DELTA_CHARS:
            d8 = "**H-지시경쟁 채택** — 열거 지시가 위치와 무관하게 note 를 누른다"
        else:
            d8 = "**부분 효과** — 둘 다 작용한다. 어느 쪽이 주된지는 말하지 않는다"

        verdicts[m] = {"D1": d1, "D2": d2, "D3": d3, "D8": d8,
                       "a_ov": a_ov, "b_rc": b_rc, "d_ov": d_ov, "d_rc": d_rc,
                       "c_nl": c_nl, "d_nl": d_nl, "r_nl": r_nl, "a_nl": a_nl, "b_nl": b_nl}

        L += [f"**{m}**", "",
              "| 규칙 | 지표 | 기준 | D (또는 R) | 판정 |",
              "| --- | --- | ---: | ---: | --- |",
              f"| D1 어휘 유지 | 허용목록_내 | C {fmt(c_al)} | D {fmt(d_al)} | {d1} |",
              f"| D2(a) 과잉 기권 | 기권 오답률 | A {fmt(a_ov)} (+{DELTA_PP}%p 까지 허용) | "
              f"D {fmt(d_ov)} | {'충족' if d2a else '**불충족**' if d2a is not None else '—'} |",
              f"| D2(b) 재현율 | 「정보 부족」 | B {fmt(b_rc)} (−{DELTA_PP}%p 까지 허용) | "
              f"D {fmt(d_rc)} | {'충족' if d2b else '**불충족**' if d2b is not None else '—'} |",
              f"| **D2 주 판정** | | | | {d2} |",
              f"| D3 note 회복 | 평균 길이 | C {fmt(c_nl, '자')} · B {fmt(b_nl, '자')} | "
              f"D {fmt(d_nl, '자')} | {d3} |",
              f"| **D8 기전** | 평균 길이 | C {fmt(c_nl, '자')} · A {fmt(a_nl, '자')} | "
              f"R {fmt(r_nl, '자')} | {d8} |", ""]

    if "R" in conds:
        L += ["> **D8 은 위치와 생성 순서를 분리하지 못한다**(설계서 C8). 자기회귀 모델에서",
              "> 둘은 같은 사건이다. 어느 판정이 나오든 이 문장이 함께 간다.", ""]
        agree = {verdicts[m]["D8"] for m in models}
        if len(agree) > 1:
            L += ["> **두 모델의 D8 판정이 갈렸다.** 모델 하나로 기전을 확정하지 않는다"
                  "(설계서 D8). 갈렸다는 사실 자체를 결과로 적는다.", ""]

    # N.6 쏠림 (D4)
    gold_dist = defaultdict(int)
    for q in gold.values():
        gold_dist[q["expected_category"]] += 1
    n_gold = sum(gold_dist.values())

    L += ["## N.6 D4 — 라벨 쏠림", "",
          f"최다 라벨 점유율이 gold 에서 그 칸이 차지하는 몫의 {SKEW_RATIO:g}배를 넘으면,",
          "그 모델의 분류 개선은 쏠림으로 설명된다고 적는다. (부록 M.3 과 같은 기준.",
          "거기서는 사후였고 **여기서는 사전 등록**이다.)", "",
          "| 모델 | 조건 | 최다 라벨 | 점유율 | gold 몫 | 배수 | 판정 |",
          "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for m in models:
        for c in conds:
            cnt = defaultdict(int)
            for x in sel(m, c):
                if x["model_category"]:
                    cnt[x["model_category"]] += 1
            if not cnt:
                continue
            top = max(cnt, key=cnt.get)
            tot = sum(cnt.values())
            share = 100 * cnt[top] / tot
            gs = 100 * gold_dist.get(top, 0) / n_gold
            ratio = share / gs if gs else float("inf")
            L.append(f"| {m} | {c} | `{top}` | {share:.1f}% | {gs:.1f}% | "
                     f"{ratio:.1f}배 | {'**쏠림**' if ratio >= SKEW_RATIO else '정상'} |")
    L.append("")

    # gold 칸별 재현율
    L += ["### gold 칸별 재현율 (엄격 기준)", ""]
    gcats = sorted(gold_dist, key=lambda k: -gold_dist[k])
    for m in models:
        L += [f"**{m}**", "",
              "| gold 카테고리 | 문항 | " + " | ".join(conds) + " |",
              "| --- | ---: | " + " | ".join("---:" for _ in conds) + " |"]
        for gc in gcats:
            cells = []
            for c in conds:
                s = [x for x in sel(m, c) if x["expected_category"] == gc
                     and x["strict_match"] is not None]
                cells.append(fmt(pct(sum(1 for x in s if x["strict_match"]), len(s))))
            L.append(f"| {gc} | {gold_dist[gc]} | " + " | ".join(cells) + " |")
        L.append("")

    # N.7 지연 (D6)
    contaminated = bool(env.get("contention_changed"))
    L += ["## N.7 D6 — 지연", ""]
    if contaminated:
        L += ["**실행 중 GPU 점유가 변했다. 지연을 판정에 쓰지 않는다.** 기록만 남긴다.", ""]
    else:
        L += ["환경 지문이 유지되고 GPU 점유 변화가 없었다. **조건 간 지연 비교가 가능하다**",
              "— 부록 M 이후 처음이다.", ""]
    L += ["| 모델 | 조건 | P50 | P95 | 평균 출력 토큰 |",
          "| --- | --- | ---: | ---: | ---: |"]
    for m in models:
        for c in conds:
            t = sorted(x["total_time_sec"] for x in sel(m, c)
                       if isinstance(x["total_time_sec"], (int, float)))
            if not t:
                continue
            p95 = t[min(len(t) - 1, int(round(0.95 * (len(t) - 1))))]
            L.append(f"| {m} | {c} | {statistics.median(t):.2f}s | {p95:.2f}s | "
                     f"{fmt(mean(m, c, 'generated_tokens'), '')} |")
    L.append("")

    # N.8 한계
    L += ["## N.8 이 분석이 말할 수 없는 것", "",
          "- **종합 점수와 게이트는 여기 없다.** judge 채점 후 `24_` 가 낸다. 부록 M.8 에서",
          "  둘이 반대로 움직였으므로, D2 가 채택이어도 게이트가 내려가면 **게이트를 따른다**"
          "(설계서 D5).",
          "- **조건 R 은 배포 제안이 아니다.** 출력 형식 변경은 다운스트림 파서 전체에",
          "  영향을 준다(설계서 C9).",
          "- **위치와 생성 순서는 분리되지 않는다**(C8).",
          "- **평가셋 과적합.** 같은 54문항으로 프롬프트를 네 번 고쳤다. D 가 이겨도 배포",
          "  근거가 아니라 후보 선정 근거다. AI Hub 세트에서 A 와 D 를 재확인해야 한다"
          "(설계서 §11).",
          "- **조건 D 의 프롬프트가 가장 길다.** D 의 note 가 길게 나왔다면 길이 효과를",
          "  배제하지 못한다. C vs R 만이 길이가 통제된 비교다(설계서 §10.1).", ""]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "26_condition_d_analysis.md").write_text("\n".join(L), encoding="utf-8")
    with open(OUT_DIR / "26_condition_d_rows.csv", "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "note_text"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ── 콘솔 ─────────────────────────────────────────────────────
    print("=" * 72)
    print(f"조건 D·R 결정적 분석 — {len(rows)}건 · 평가셋 {eval_set_version}")
    print("=" * 72)
    for m in models:
        v = verdicts[m]
        print(f"\n{m}")
        print(f"  {'조건':<4}{'허용목록':>10}{'엄격':>9}{'재현율':>9}"
              f"{'과잉기권':>10}{'note':>9}")
        for c in conds:
            print(f"  {c:<4}{fmt(rate(m, c, 'in_allowed')[0]):>10}"
                  f"{fmt(rate(m, c, 'strict_match')[0]):>9}"
                  f"{fmt(insuff_recall(m, c)[0]):>9}"
                  f"{fmt(over_abstain(m, c)[0]):>10}"
                  f"{fmt(mean(m, c, 'note_len'), '자'):>9}")
        print(f"  D1 {v['D1']}")
        print(f"  D2 {v['D2']}")
        print(f"  D3 {v['D3']}")
        print(f"  D8 {v['D8']}")
    print(f"\n기록: work/25_condition_d/26_condition_d_analysis.md")
    print(f"다음: uv run python 05_quality_scoring.py "
          f"work/25_condition_d/25_condition_d_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
