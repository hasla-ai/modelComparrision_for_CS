"""조건 D·R — 열거 + 기권 기준 + note 규칙, 그리고 필드 순서 반전.

설계서: `CONDITION_D_DESIGN.md` (실행 전 커밋됨)

무엇을 묻는가
  D — 조건 C 가 고친 것(어휘)을 유지하면서 C 가 깬 것(note)과 B 가 깬 것(과잉
      기권)을 함께 막을 수 있는가. 배포 후보를 고르는 조건이다.
  R — 조건 C 에서 note 가 무너진 **이유**가 위치인가 지시 경쟁인가.
      조건 C 의 출력 형식에서 note 를 category 앞으로만 옮긴다.
      배포 후보가 아니다(설계서 §4.1).

조건 A·B·C 를 다시 재는 이유
  부록 M 의 수치는 영상 재생 중에 측정됐다(T6). 지연이 오염됐고, 같은 표에 D 를
  놓으려면 같은 실행에서 다시 재야 한다(부록 L).

블록 동기화
  `ALLOWED_CATEGORIES`·`ENUMERATION`·`REINFORCEMENT` 는 `22_condition_c_experiment.py`
  에서 **import 한다.** 베껴 오면 한 글자 차이가 생겨도 아무도 모르고, 그러면
  부록 M 의 B·C 와 이 실행의 B·C 가 다른 조건이 된다. import 는 그 약속을
  말이 아니라 코드로 지킨다.

실행:
    uv run python 25_condition_d_experiment.py --dry-run           # 프롬프트 5종 육안 확인
    uv run python 25_condition_d_experiment.py                     # 1,080회, 약 41분
    uv run python 25_condition_d_experiment.py --conditions C,R    # 기전 검증만 (432회)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from env_snapshot import snapshot as env_snapshot, save_pair as env_save_pair, \
    row_fields as env_row_fields, describe as env_describe, \
    contention_note as env_contention

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "work" / "25_condition_d"
QUESTIONS_FILE = BASE_DIR / "data" / "02_prompts_and_eval.json"

OLLAMA_BASE = "http://localhost:11434"
GENERATE_URL = f"{OLLAMA_BASE}/api/generate"
TAGS_URL = f"{OLLAMA_BASE}/api/tags"

MODELS = ["llama3.2:3b", "qwen2.5:7b"]
PASS_COUNT = 2
WARMUP_PROMPT = "워밍업용 짧은 요청입니다. 간단히 한 문장으로 답하세요."
REQUEST_TIMEOUT_SEC = 180
PROMPT_VARIANT_VERSION = "d-v1-2026-09"


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_cc = load_module("22_condition_c_experiment.py", "cc22")
ALLOWED_CATEGORIES = _cc.ALLOWED_CATEGORIES          # 부록 M 과 같은 목록
ENUMERATION = _cc.ENUMERATION                        # 부록 M 과 같은 문장
REINFORCEMENT = _cc.REINFORCEMENT                    # 부록 D 의 규칙 3개, 그대로

# ── 조건 D 가 새로 쓰는 블록 두 개 (설계서 §4 블록 2·3) ──────────────
#
# 블록 2 — 기권 기준. 조건 B 의 첫 규칙과 **방향이 반대**다.
#   B 는 "확신 없으면 기권하라", D 는 "이 경우에만 기권하라".
#   gold 에서 「정보 부족」이 정답인 것은 54문항 중 10문항뿐이고
#   나머지 44문항에서 기권은 오답이다(설계서 §3.1).
ABSTENTION_CRITERION = (
    '\ncategory를 "정보 부족"으로 적는 것은 다음 경우뿐입니다: 문의에서 무엇에 대한 '
    '것인지(대상)와 무엇이 일어났거나 무엇을 원하는지(증상 또는 요청) 중 하나라도 '
    '파악되지 않을 때. 둘 다 파악되면 정보가 적더라도 해당 카테고리로 적으세요.'
)

# 블록 3 — note 규칙. 조건 B 규칙 3개 중 2번(권한 밖)·3번(값 추측 금지)만 살리고
#   1번(기권 지시)은 블록 2 로 교체했다. 「category와 무관하게」가 핵심이다 —
#   조건 C 에서 note 가 빈 것이 '카테고리를 정했으니 끝'이라는 판단 때문이라는
#   가설에 대응한다. 가설이 틀렸다면 이 문구는 효과가 없고, 그것도 결과다.
NOTE_RULE = (
    '\nnote에는 category와 무관하게 다음을 적으세요: 문의에 없어서 추가로 확인해야 할 '
    '항목, 그리고 상담원 권한 밖의 확정 판단이 필요한 요구가 있으면 그 사실. '
    '문의에 없는 값(주문번호, 날짜, 금액 등)은 추측해서 채우지 마세요.'
)

CONDITIONS = {
    "A": {"label": "현행 (baseline)", "suffix": "", "reorder": False},
    "B": {"label": "보강 (규칙 3개)", "suffix": REINFORCEMENT, "reorder": False},
    "C": {"label": "카테고리 열거", "suffix": ENUMERATION, "reorder": False},
    "D": {"label": "열거+기권기준+note규칙",
          "suffix": ENUMERATION + ABSTENTION_CRITERION + NOTE_RULE, "reorder": False},
    "R": {"label": "열거 + note 선행 (기전 검증)", "suffix": ENUMERATION, "reorder": True},
}
DEPLOY_CANDIDATES = ["A", "B", "C", "D"]   # R 은 배포 후보가 아니다 (설계서 §4.1)

_TYPE_LABELS = _cc._TYPE_LABELS


# ══════════════════════════════════════════════════════════════════
# 조건 R — 출력 형식에서 note 줄을 맨 앞으로 옮긴다
# ══════════════════════════════════════════════════════════════════
def reorder_note_first(prompt: str) -> str:
    """`- note:` 줄을 출력 형식 목록의 맨 앞으로 옮긴다.

    **조용히 실패하면 안 되는 함수다.** 옮기지 못했는데 그대로 반환하면 조건 R 이
    조건 C 와 똑같아지고, 그러면 "note 길이가 같다 → H-지시경쟁 채택"이라는
    **완전히 틀린 결론**이 나온다. 정규식이 안 맞았다는 사실을 아무도 모른 채로.

    그래서 실패하면 예외를 던진다. 호출부는 첫 문항에서 한 번 검증하고,
    검증을 통과하지 못하면 모델을 한 번도 부르지 않고 종료한다.
    """
    lines = prompt.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("- note:")]
    if len(idx) != 1:
        raise ValueError(f"`- note:` 줄이 {len(idx)}개입니다 (1개여야 함)")
    note_i = idx[0]

    head = [i for i, ln in enumerate(lines) if ln.strip() == "출력 형식:"]
    if len(head) != 1:
        raise ValueError(f"`출력 형식:` 줄이 {len(head)}개입니다 (1개여야 함)")
    first_field = head[0] + 1
    if note_i <= first_field:
        raise ValueError("note 가 이미 맨 앞입니다 — 옮길 것이 없습니다")

    note_line = lines.pop(note_i)
    lines.insert(first_field, note_line)
    out = "\n".join(lines)

    # 사후 검증 — 줄 집합은 그대로이고 순서만 바뀌었는가
    if sorted(out.split("\n")) != sorted(prompt.split("\n")):
        raise ValueError("줄 집합이 바뀌었습니다 — 순서 외의 변경이 일어났습니다")
    if out == prompt:
        raise ValueError("프롬프트가 그대로입니다 — 순서가 바뀌지 않았습니다")
    o = out.index("- note:")
    c = out.index("- category:")
    if not o < c:
        raise ValueError("note 가 category 앞에 오지 않았습니다")
    return out


def build_prompt(base_prompt: str, condition: str) -> str:
    spec = CONDITIONS[condition]
    body = reorder_note_first(base_prompt) if spec["reorder"] else base_prompt
    return body + spec["suffix"]


def verify_reorder(questions: list[dict]) -> list[str]:
    """모든 문항에서 순서 반전이 성립하는지 **호출 전에** 확인한다."""
    problems = []
    for q in questions:
        try:
            reorder_note_first(q["base_prompt"])
        except ValueError as exc:
            problems.append(f"문항 {q['question_id']}: {exc}")
    return problems


# ══════════════════════════════════════════════════════════════════
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
    return qs, data.get("rubric_version"), sorted({q["expected_category"] for q in qs})


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
        tags = [m.get("name", "")
                for m in requests.get(TAGS_URL, timeout=10).json().get("models", [])]
    except Exception as exc:
        print(f"Ollama 에 닿지 못했습니다: {exc}")
        return False
    missing = [m for m in models if m not in tags]
    if missing:
        print(f"모델 미적재: {', '.join(missing)}")
        return False
    return True


# ══════════════════════════════════════════════════════════════════
class RowSink:
    """한 건씩 즉시 CSV 에 쓴다.

    부록 K 때 71분짜리 실행이 진행 표시도 중간 저장도 없이 돌았다. 그건 설계
    결함이었다. 중단되면 그때까지의 결과가 남아야 한다.
    """

    def __init__(self, path: Path):
        self.path = path
        self.f = None
        self.w = None
        self.n = 0

    def write(self, row: dict) -> None:
        if self.w is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(self.path, "w", encoding="utf-8-sig", newline="")
            self.w = csv.DictWriter(self.f, fieldnames=list(row.keys()))
            self.w.writeheader()
        self.w.writerow(row)
        self.f.flush()
        self.n += 1

    def close(self) -> None:
        if self.f:
            self.f.close()


class Progress:
    def __init__(self, total: int, every_sec: float = 10.0):
        self.total, self.every = total, every_sec
        self.t0 = time.time()
        self.last = 0.0

    def tick(self, done: int, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        el = now - self.t0
        rate = done / el if el > 0 else 0
        eta = (self.total - done) / rate if rate > 0 else 0
        print(f"    [{done}/{self.total}] {100 * done / self.total:4.1f}% · "
              f"{rate:.2f}건/초 · 경과 {el / 60:.1f}분 · 남은 예상 {eta / 60:.1f}분",
              flush=True)


_INTERRUPTED = {"flag": False}


def _on_sigint(signum, frame):
    if _INTERRUPTED["flag"]:
        raise KeyboardInterrupt
    _INTERRUPTED["flag"] = True
    print("\n  중단 요청 — 현재 호출을 마치고 저장한 뒤 종료합니다. "
          "(한 번 더 누르면 즉시 중단)", flush=True)


# ══════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="조건 D·R — 열거+기권기준+note규칙 / 순서 반전")
    ap.add_argument("--conditions", default="A,B,C,D,R")
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

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 조건 D·R — 설계서 CONDITION_D_DESIGN.md")
    print(f"  평가셋 {version} · 문항 {len(questions)} · 조건 {conds_all} · "
          f"모델 {len(models)} · 패스 {PASS_COUNT}")
    total = len(questions) * len(conds_all) * len(models) * PASS_COUNT
    print(f"  총 호출 {total}회\n")

    # ── 사전 검증 1: 열거 목록이 gold 어휘를 덮는가 ──────────────
    ok, notes = _cc.verify_enumeration(gold_vocab)
    print("  열거 목록 대조 (gold 어휘 vs 프롬프트 목록)")
    for n in notes:
        print(f"    · {n}")
    if not ok:
        print("\n  중단 — gold 어휘 중 열거가 덮지 못하는 값이 있습니다.")
        return 1

    # ── 사전 검증 2: 조건 R 의 순서 반전이 전 문항에서 성립하는가 ──
    if "R" in conds_all:
        problems = verify_reorder(questions)
        print(f"\n  조건 R 순서 반전 검증 — {len(questions)}문항 중 "
              f"실패 {len(problems)}건")
        if problems:
            for pr in problems[:5]:
                print(f"    · {pr}")
            print("\n  중단 — 순서를 바꾸지 못한 문항이 있습니다. 그대로 돌리면")
            print("  조건 R 이 조건 C 와 같아지고, note 길이가 같다는 이유로")
            print("  'H-지시경쟁 채택'이라는 틀린 결론이 나옵니다.")
            return 1
        print("    · 전 문항에서 note 가 category 앞으로 이동함")

    # ── 조건 D 블록 동기화 확인 ──────────────────────────────────
    assert CONDITIONS["C"]["suffix"] == ENUMERATION
    assert CONDITIONS["D"]["suffix"].startswith(ENUMERATION)
    assert CONDITIONS["B"]["suffix"] == REINFORCEMENT
    print(f"\n  블록 동기화 — B·C 는 22_condition_c_experiment.py 에서 import 한 "
          f"문자열 그대로 (부록 M 과 동일 조건)")

    if args.dry_run:
        q = questions[0]
        print(f"\n{'=' * 70}\n문항 {q['question_id']} 의 조건별 프롬프트\n{'=' * 70}")
        for c in conds_all:
            p = build_prompt(q["base_prompt"], c)
            flag = "" if c in DEPLOY_CANDIDATES else "   ← 배포 후보 아님 (기전 검증)"
            print(f"\n──── 조건 {c} · {CONDITIONS[c]['label']} · {len(p)}자{flag}")
            print(p)
        print(f"\n{'=' * 70}")
        print("--dry-run 이므로 호출하지 않고 종료합니다.")
        return 0

    if not preflight(models):
        return 1

    env_start = env_snapshot(OLLAMA_BASE)
    print(f"\n  환경 {env_describe(env_start)}")
    note = env_contention(env_start)
    if note:
        print(f"  ** {note}")
        print("  ** 설계서 D6 — 오염된 실행의 지연은 판정에 쓰지 않습니다.")
        print("  ** 지연도 재려면 지금 중단하고 다른 프로그램을 닫으십시오. "
              "(5초 후 계속)")
        time.sleep(5)

    sink = RowSink(OUTPUT_DIR / "25_condition_d_results.csv")
    prog = Progress(total)
    results: list[dict] = []
    done = 0
    signal.signal(signal.SIGINT, _on_sigint)
    t_start = time.time()

    try:
        for pass_idx in range(1, PASS_COUNT + 1):
            order = models if pass_idx % 2 == 1 else list(reversed(models))
            print(f"\n{'=' * 70}\n패스 {pass_idx}/{PASS_COUNT} — 모델 순서 "
                  f"{' → '.join(order)}\n{'=' * 70}")
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
                        row = {
                            "model_name": model,
                            "condition": cond,
                            "condition_label": CONDITIONS[cond]["label"],
                            "is_deploy_candidate": cond in DEPLOY_CANDIDATES,
                            "field_order": "note_first" if CONDITIONS[cond]["reorder"]
                                           else "category_first",
                            "prompt_variant_version": PROMPT_VARIANT_VERSION,
                            "question_id": qid,
                            "category": q["type"],
                            "expected_category": q["expected_category"],
                            "run_index": pass_idx,
                            "pair_order": pair_order,
                            "prompt": prompt,
                            "prompt_chars": len(prompt),
                            "eval_criteria": q["eval_criteria"],
                            "response_text": res["response_text"],
                            "total_time_sec": res["total_time_sec"],
                            "ttft_sec": res["ttft_sec"],
                            "generated_tokens": res["eval_count_tokens"],
                            "tps": res["tps"],
                            "is_success": res["success"],
                            "error_message": res["error_msg"],
                            "manual_score": None,
                            **env_row_fields(env_start),
                        }
                        results.append(row)
                        sink.write(row)
                        prog.tick(done)
                    if _INTERRUPTED["flag"]:
                        break
                if _INTERRUPTED["flag"]:
                    break
                unload(model)
            if _INTERRUPTED["flag"]:
                break
    finally:
        sink.close()

    prog.tick(done, force=True)
    env_end = env_snapshot(OLLAMA_BASE)
    env_save_pair(env_start, env_end, OUTPUT_DIR / "25_env.json")

    with open(OUTPUT_DIR / "25_condition_d_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── 실행 요약 ────────────────────────────────────────────────
    L = [f"# 조건 D·R 실행 기록 (채점 전)", "",
         f"설계서 `CONDITION_D_DESIGN.md` · 프롬프트 판 `{PROMPT_VARIANT_VERSION}`", "",
         f"측정 환경 시작 — {env_describe(env_start)}",
         f"측정 환경 종료 — {env_describe(env_end)}", "",
         f"평가셋 `{version}` · 문항 {len(questions)} · 조건 {', '.join(conds_all)} · "
         f"호출 {done}/{total}회 · 소요 {(time.time() - t_start) / 60:.1f}분", ""]
    if _INTERRUPTED["flag"]:
        L += ["> **중단된 실행이다.** 부분 결과이며 조건별 건수가 고르지 않다. "
              "판정에 쓰지 말 것.", ""]
    for tag, snap in (("시작", env_start), ("종료", env_end)):
        n = env_contention(snap)
        if n:
            L.append(f"- GPU 경쟁({tag}) — {n}")
    if env_start.get("gpu", {}).get("other_vram_mib") != \
       env_end.get("gpu", {}).get("other_vram_mib"):
        L.append("- **실행 중 GPU 점유가 변했다.** 지연 비교는 설계서 D6 에 따라 재검토한다.")
    L += ["", "## 조건별 건수", "",
          "| 조건 | 라벨 | 배포 후보 | 필드 순서 | 성공 | 실패 |",
          "| --- | --- | :-: | --- | ---: | ---: |"]
    for c in conds_all:
        sel = [r for r in results if r["condition"] == c]
        L.append(f"| {c} | {CONDITIONS[c]['label']} | "
                 f"{'○' if c in DEPLOY_CANDIDATES else '— (기전 검증)'} | "
                 f"{'note 선행' if CONDITIONS[c]['reorder'] else 'category 선행'} | "
                 f"{sum(1 for r in sel if r['is_success'])} | "
                 f"{sum(1 for r in sel if not r['is_success'])} |")
    L += ["", "## 다음", "",
          "```",
          "uv run python 26_condition_d_analysis.py",
          "uv run python 05_quality_scoring.py "
          "work/25_condition_d/25_condition_d_results.json",
          "uv run python 24_condition_c_scored_readout.py "
          "work/25_condition_d/25_condition_d_scored.csv",
          "```", ""]
    (OUTPUT_DIR / "25_condition_d_summary.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"완료 {done}/{total}회 · {(time.time() - t_start) / 60:.1f}분")
    print(f"기록: work/25_condition_d/")
    print(f"다음: uv run python 26_condition_d_analysis.py")
    return 130 if _INTERRUPTED["flag"] else 0


if __name__ == "__main__":
    sys.exit(main())
