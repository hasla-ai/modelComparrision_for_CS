"""조건 C — 프롬프트에 허용 카테고리를 열거한다 (부록 D 가 지목한 진짜 해법)

왜 이 실험인가
  부록 D 에서 두 가지가 드러났다.

    · 조건 B(기권·권한·환각 규칙 3개 추가)는 `llama3.2:3b` 를 과잉 보수화시켰다.
      정상 문항 16건 중 8건을 「정보 부족」으로 오분류했다. **그대로 쓰면 안 된다.**
    · `qwen2.5:7b` 는 과잉 보수화는 없었으나 **카테고리 어휘가 흔들렸다.**
      「소프트웨어 오류 문의」, 「기술 지원」처럼 gold 에 없는 이름을 지어냈다.

  어휘가 흔들리는 이유는 단순하다. **프롬프트가 허용 카테고리를 한 번도 말해 주지
  않는다.** 모델은 무엇을 골라야 하는지 모른 채 그럴듯한 이름을 만든다.

  조건 C 는 그 목록을 알려 준다. 부록 D §D.4 와 부록 G 가 "프롬프트 변경이므로
  조건 C 로 따로 실험한다"고 두 번 적어 둔 것을 이제 실행한다.

설계
  A  현행 프롬프트 (baseline)
  B  현행 + 규칙 3개          — 부록 D 의 그 조건. 같은 실행 안에서 다시 잰다
  C  현행 + 허용 카테고리 열거

  조건을 **가장 안쪽 루프**에 둔다. 조건 전환 비용이 0이므로 같은 문항·같은 모델
  상태에서 곧바로 비교할 수 있다(짝지은 설계). 쌍 안의 순서는 교대한다.

  B 를 이번 실행에 다시 넣는 이유: 부록 D 의 B 수치는 **다른 환경에서 잰 값**이다
  (부록 L). 같은 표에 놓으려면 같은 실행에서 다시 재야 한다.

무엇을 조심하는가 — §함정 참조.

실행:
    uv run python 22_condition_c_experiment.py                      # 648회
    uv run python 22_condition_c_experiment.py --conditions A,C     # 432회
    uv run python 22_condition_c_experiment.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from env_snapshot import snapshot as env_snapshot, save as env_save, \
    row_fields as env_row_fields, describe as env_describe

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "work" / "22_condition_c"
QUESTIONS_FILE = BASE_DIR / "data" / "02_prompts_and_eval.json"

OLLAMA_BASE = "http://localhost:11434"
GENERATE_URL = f"{OLLAMA_BASE}/api/generate"
TAGS_URL = f"{OLLAMA_BASE}/api/tags"

MODELS = ["llama3.2:3b", "qwen2.5:7b"]
PASS_COUNT = 2
WARMUP_PROMPT = "워밍업용 짧은 요청입니다. 간단히 한 문장으로 답하세요."
REQUEST_TIMEOUT_SEC = 180
PROMPT_VARIANT_VERSION = "c-v1-2026-09"

# ── 부록 D 의 규칙 블록. 한 글자도 바꾸지 않는다 ──────────────────────
REINFORCEMENT = """
규칙:
- 문의에서 증상이나 요청을 식별할 수 없을 만큼 정보가 부족하면, category를 임의로 단정하지 말고 "정보 부족"으로 적고 note에 추가로 확인해야 할 항목을 적으세요.
- 보상·환불·배상처럼 상담원 권한 밖의 확정 판단이 필요한 요구는 승인하거나 약속하지 말고, 요구 내용만 그대로 옮긴 뒤 note에 권한 밖 사항임을 적으세요.
- 문의에 없는 값(주문번호, 날짜, 금액 등)은 추측해서 채우지 말고, 해당 항목을 비우거나 note에 없음을 적으세요."""

# ── 조건 C 가 덧붙이는 블록 ──────────────────────────────────────────
# 열거 목록을 **손으로 고정**한다. gold 에서 자동 추출하지 않는 이유는 아래
# verify_enumeration() 주석에 있다 — gold 어휘에 「기타」가 두 형태로 들어 있어
# 그대로 열거하면 모델을 혼란스럽게 만든다.
ALLOWED_CATEGORIES = [
    "로그인/계정",
    "결제/환불",
    "기능 오류",
    "배송/주문",
    "컴플레인/보상요구",
    "기타",
    "정보 부족",
]
ENUMERATION = "\ncategory는 반드시 다음 중 하나로만 적으세요: " + ", ".join(ALLOWED_CATEGORIES)

CONDITIONS = {
    "A": {"label": "현행 (baseline)", "suffix": ""},
    "B": {"label": "보강 (규칙 3개)", "suffix": REINFORCEMENT},
    "C": {"label": "카테고리 열거", "suffix": ENUMERATION},
}

_TYPE_LABELS = {
    "정상": "정상 사례",
    "경계": "경계 사례",
    "특수: 정보부족": "정보 부족 사례",
    "특수: 범위밖": "범위 밖 사례",
}


def safe_name(tag: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "_", tag)


def load_questions() -> tuple[list[dict], str, list[str]]:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    qs = [{
        "question_id": q["id"],
        "type": _TYPE_LABELS.get(q["taxonomy"]["info_sufficiency"],
                                 q["taxonomy"]["info_sufficiency"]),
        "base_prompt": q["prompt"],
        "eval_criteria": q.get("eval_criteria", ""),
        "expected_category": q["expected_category"],
    } for q in data["questions"]]
    gold_vocab = sorted({q["expected_category"] for q in qs})
    return qs, data.get("rubric_version"), gold_vocab


def verify_enumeration(gold_vocab: list[str]) -> tuple[bool, list[str]]:
    """열거 목록이 gold 어휘를 덮는지 확인한다.

    왜 자동 추출하지 않는가
      gold 의 expected_category 에는 「기타」와 「기타(판단 불가)」가 **둘 다** 있다.
      전자는 '분류는 되지만 여섯 칸에 안 들어감', 후자는 '무엇에 관한 문의인지
      판단이 안 됨'이라는 채점 쪽 구분이다. 이 둘을 프롬프트에 나란히 열거하면
      모델이 구분할 수 없고, 열거의 목적(어휘 고정) 자체가 무너진다.

      그래서 프롬프트에는 「기타」 하나만 넣는다. 채점기는 두 gold 모두에 대해
      「기타」를 정답으로 인정하므로(05_quality_scoring.py CATEGORY_KEYWORDS)
      점수에는 불리하지 않다.

      **이것은 이 실험의 한계이자 발견이다.** gold 어휘를 그대로 프롬프트에 옮길
      수 없다는 것은 어휘 자체가 정리되지 않았다는 뜻이고, 다음 루브릭 버전에서
      두 「기타」를 정리해야 한다.
    """
    notes = []
    for g in gold_vocab:
        if g in ALLOWED_CATEGORIES:
            continue
        # 열거에 없는 gold 값 — 어떤 열거 항목으로 흡수되는지 적어 둔다
        absorbed = [a for a in ALLOWED_CATEGORIES if a in g or g.startswith(a)]
        if absorbed:
            notes.append(f"gold 「{g}」 → 프롬프트에는 「{absorbed[0]}」 로만 제시됨")
        else:
            notes.append(f"**gold 「{g}」 가 열거 목록에 없고 흡수될 항목도 없음**")
    ok = not any(n.startswith("**") for n in notes)
    return ok, notes


def build_prompt(base_prompt: str, condition: str) -> str:
    """원문 뒤에 블록을 붙이기만 한다. 기존 문장은 한 글자도 고치지 않는다."""
    return base_prompt + CONDITIONS[condition]["suffix"]


# ══════════════════════════════════════════════════════════════════
def _failed(msg: str) -> dict:
    return {"success": False, "response_text": "", "total_time_sec": 0.0,
            "ttft_sec": 0.0, "eval_count_tokens": 0, "tps": 0.0, "error_msg": msg}


def run_ollama_query(model_name: str, prompt: str) -> dict:
    payload = {"model": model_name, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.1}}
    t0 = time.time()
    try:
        r = requests.post(GENERATE_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    except Exception as exc:
        return _failed(f"{type(exc).__name__}: {exc}")
    wall = time.time() - t0
    if r.status_code != 200:
        return _failed(f"HTTP {r.status_code}")
    j = r.json()
    ev = j.get("eval_count", 0) or 0
    ed = (j.get("eval_duration", 0) or 0) / 1e9
    return {
        "success": True,
        "response_text": (j.get("response") or "").strip(),
        "total_time_sec": round(wall, 3),
        "ttft_sec": round(((j.get("load_duration", 0) or 0)
                           + (j.get("prompt_eval_duration", 0) or 0)) / 1e9, 3),
        "eval_count_tokens": ev,
        "tps": round(ev / ed, 2) if ed > 0 else 0.0,
        "error_msg": "",
    }


def unload(model_name: str) -> None:
    try:
        requests.post(GENERATE_URL, json={"model": model_name, "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def preflight(models: list[str]) -> bool:
    try:
        r = requests.get(TAGS_URL, timeout=10)
        tags = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception as exc:
        print(f"Ollama 에 닿지 못했습니다: {exc}")
        return False
    missing = [m for m in models if m not in tags]
    if missing:
        print(f"모델 미적재: {', '.join(missing)}")
        return False
    return True


# ══════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="조건 C — 카테고리 열거 실험")
    ap.add_argument("--conditions", default="A,B,C")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conds_all = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds_all if c not in CONDITIONS]
    if bad:
        print(f"알 수 없는 조건: {', '.join(bad)} (가능: {', '.join(CONDITIONS)})")
        return 1
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    questions, version, gold_vocab = load_questions()
    ok, notes = verify_enumeration(gold_vocab)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 조건 C — 카테고리 열거 실험")
    print(f"  평가셋 {version} · 문항 {len(questions)} · 조건 {conds_all} · "
          f"모델 {len(models)} · 패스 {PASS_COUNT}")
    total = len(questions) * len(conds_all) * len(models) * PASS_COUNT
    print(f"  총 호출 {total}회\n")

    print("  열거 목록 대조 (gold 어휘 vs 프롬프트에 제시하는 목록)")
    for n in notes:
        print(f"    · {n}")
    if not notes:
        print("    · gold 어휘와 완전히 일치")
    if not ok:
        print("\n  중단 — gold 어휘 중 열거 목록이 덮지 못하는 값이 있습니다.")
        print("  그 문항은 무엇을 답해도 틀리게 되므로 실험이 성립하지 않습니다.")
        return 1
    print()

    if args.dry_run:
        for p in range(1, PASS_COUNT + 1):
            print(f"  패스 {p}: 조건 순서는 (패스+문항ID) 홀짝으로 회전")
        print("\n--dry-run 이므로 호출하지 않고 종료합니다.")
        return 0

    if not preflight(models):
        return 1

    env = env_snapshot(OLLAMA_BASE)
    print(f"  환경 {env_describe(env)}\n")

    results: list[dict] = []
    done = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for pass_idx in range(1, PASS_COUNT + 1):
        order = models if pass_idx % 2 == 1 else list(reversed(models))
        print(f"\n{'=' * 64}\n패스 {pass_idx}/{PASS_COUNT} — 모델 순서 {' → '.join(order)}\n{'=' * 64}")
        for model in order:
            for other in models:
                if other != model:
                    unload(other)
            w = run_ollama_query(model, WARMUP_PROMPT)
            print(f"\n[{model}] 워밍업 {w['total_time_sec']}초 (집계 제외)")

            for q in questions:
                qid = q["question_id"]
                # 쌍 안의 조건 순서를 회전시켜 순서 효과를 상쇄한다
                shift = (pass_idx + int(qid)) % len(conds_all)
                conds = conds_all[shift:] + conds_all[:shift]
                for pair_order, cond in enumerate(conds, start=1):
                    prompt = build_prompt(q["base_prompt"], cond)
                    res = run_ollama_query(model, prompt)
                    done += 1
                    results.append({
                        "model_name": model,
                        "condition": cond,
                        "condition_label": CONDITIONS[cond]["label"],
                        "prompt_variant_version": PROMPT_VARIANT_VERSION,
                        "question_id": qid,
                        "category": q["type"],
                        "expected_category": q["expected_category"],
                        "run_index": pass_idx,
                        "pair_order": pair_order,
                        "prompt": prompt,
                        "eval_criteria": q["eval_criteria"],
                        "response_text": res["response_text"],
                        "total_time_sec": res["total_time_sec"],
                        "ttft_sec": res["ttft_sec"],
                        "generated_tokens": res["eval_count_tokens"],
                        "tps": res["tps"],
                        "is_success": res["success"],
                        "error_message": res["error_msg"],
                        "manual_score": None,
                        **env_row_fields(env),
                    })
                    if done % 20 == 0 or done == total:
                        print(f"  [{done}/{total}] {done / total * 100:.0f}%", flush=True)
            unload(model)

    # ── 저장 ──────────────────────────────────────────────────────
    env_save(env, OUTPUT_DIR / "22_env.json")
    with open(OUTPUT_DIR / "22_condition_c_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "22_condition_c_results.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # 채점 전에도 바로 보이는 것 하나 — 모델이 실제로 뱉은 카테고리 어휘
    vocab: dict[tuple[str, str], set[str]] = {}
    for r in results:
        if not r["is_success"]:
            continue
        m = re.search(r'-?\s*"?category"?\s*[:：]\s*"?([^\n",]+)', r["response_text"])
        if m:
            vocab.setdefault((r["model_name"], r["condition"]), set()).add(m.group(1).strip())

    L = ["# 조건 C — 카테고리 열거 실험 (채점 전 원본 요약)", "",
         f"측정 환경 — {env_describe(env)}", "",
         f"평가셋 `{version}` · 문항 {len(questions)} · 조건 {', '.join(conds_all)} · "
         f"총 {len(results)}회", "",
         "## 모델이 실제로 사용한 카테고리 어휘", "",
         "열거의 목적은 어휘를 고정하는 것이다. 채점 전에도 이 표로 효과가 보인다.", "",
         "| 모델 | 조건 | 서로 다른 어휘 수 | 허용 목록 밖 어휘 |",
         "| --- | --- | ---: | --- |"]
    for (m, c), vs in sorted(vocab.items()):
        outside = sorted(v for v in vs if v not in ALLOWED_CATEGORIES)
        shown = ", ".join(f"`{v}`" for v in outside[:6]) + ("…" if len(outside) > 6 else "")
        L.append(f"| {m} | {c} ({CONDITIONS[c]['label']}) | {len(vs)} | {shown or '없음'} |")
    L += ["", "## 다음", "",
          "```",
          "uv run python 05_quality_scoring.py work/22_condition_c/22_condition_c_results.json",
          "uv run python 12_rubric_sensitivity.py",
          "```", "",
          "채점은 **관대·엄격 두 기준으로 모두** 본다. 열거의 효과가 '어휘가 맞아서'인지",
          "'분류가 나아져서'인지는 두 기준의 차이로 갈린다(§함정 2).", "",
          "## 열거 목록 대조", ""]
    for n in notes:
        L.append(f"- {n}")
    (OUTPUT_DIR / "22_condition_c_summary.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n{'=' * 64}")
    for (m, c), vs in sorted(vocab.items()):
        outside = sorted(v for v in vs if v not in ALLOWED_CATEGORIES)
        print(f"  {m:<14} {c}  어휘 {len(vs):>2}종  목록 밖 {len(outside)}종 "
              f"{', '.join(outside[:4])}")
    print("=" * 64)
    print(f"기록: work/22_condition_c/")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단됨 — 이 스크립트는 끝에 한 번 저장하므로 부분 결과가 남지 않습니다.")
        print("      더 작은 --conditions 나 --models 로 나눠 돌리십시오.")
        sys.exit(130)
