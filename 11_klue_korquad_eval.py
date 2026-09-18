"""KLUE-YNAT·KLUE-MRC·KorQuAD로 생성 모델의 능력을 외부 데이터로 측정한다.

설계 근거는 KLUE_KORQUAD_DESIGN.md에 있다.

무엇을 재는가 — 이 프로젝트 축과의 대응
  YNAT        짧은 한국어 문장을 정해진 범주로 분류      → 분류 정확도 축
  KorQuAD     주어진 지문에서 정답 조각을 뽑아냄          → 핵심 정보 추출 축
  MRC(답 있음) 같음 (KLUE 도메인)                        → 핵심 정보 추출 축
  MRC(답 없음) 지문에 답이 없을 때 지어내지 않고 물러남    → 정보 부족 대응(기권) 축  ★

★ 기권 축이 이 실험의 핵심이다. 이 프로젝트에서 두 로컬 모델과 Cloud가 공통으로 실패한
  유일한 축인데, 지금은 Q9 하나(모델당 2건)에 근거하고 있다. 부록 D에서 프롬프트로
  고치려다 과잉 기권이라는 새 문제를 만들기도 했다. 수십 건으로 재면 '이 모델이 기권이라는
  행동을 할 수 있는 모델인가'를 판정할 수 있다.

쓰지 않는 곳
  벤치마크 점수는 게이트에도 선정 가중치에도 넣지 않는다. 게이트는 '권한 밖 확답 금지'
  같은 사업 리스크 조건이고 KLUE에는 그런 항목이 없다. 이 결과는 후보 조사 단계의
  사전 정보와, 10문항 파일럿의 외부 타당도 점검에만 쓴다(설계서 §1, §4 규칙 1).

본실험과 같은 호출 경로·같은 생성 설정(temperature=0.1, stream=false)을 쓴다.

사전 준비
  uv add datasets

실행:  uv run python 11_klue_korquad_eval.py
출력:  work/11_klue_korquad/11_benchmark_results.csv | .json
       work/11_klue_korquad/11_benchmark_summary.md
"""

import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "work" / "11_klue_korquad"

OLLAMA_BASE = "http://localhost:11434"
GENERATE_URL = f"{OLLAMA_BASE}/api/generate"
TAGS_URL = f"{OLLAMA_BASE}/api/tags"

MODELS = ["llama3.2:3b", "qwen2.5:7b"]
REQUEST_TIMEOUT_SEC = 180
RANDOM_SEED = 20260918

# 표본 수 — 필요에 따라 조절
N_YNAT = 100
N_KORQUAD = 60
N_MRC_ANSWERABLE = 40
N_MRC_UNANSWERABLE = 40

WARMUP_PROMPT = "워밍업용 짧은 요청입니다. 간단히 한 문장으로 답하세요."
NO_ANSWER_TOKEN = "답할 수 없음"

YNAT_LABELS_KO = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]


# ==========================================
# 1. Ollama 호출 (본실험과 동일 경로·설정)
# ==========================================
def run_query(model: str, prompt: str) -> dict:
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.1}}
    start = time.time()
    try:
        r = requests.post(GENERATE_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        if r.status_code != 200:
            return {"success": False, "text": "", "sec": 0, "err": f"HTTP {r.status_code}"}
        j = r.json()
        return {"success": True, "text": j.get("response", "").strip(),
                "sec": round(time.time() - start, 3), "err": ""}
    except Exception as exc:
        return {"success": False, "text": "", "sec": 0, "err": str(exc)}


def unload(model: str) -> None:
    try:
        requests.post(GENERATE_URL, json={"model": model, "prompt": "", "keep_alive": 0}, timeout=30)
    except Exception:
        pass
    time.sleep(1.0)


def preflight() -> bool:
    try:
        installed = {m.get("name") for m in requests.get(TAGS_URL, timeout=15).json().get("models", [])}
    except Exception as exc:
        print(f"Ollama에 연결할 수 없습니다: {exc}")
        return False
    missing = [m for m in MODELS if m not in installed]
    if missing:
        print("아래 모델을 먼저 받아야 합니다.")
        for m in missing:
            print(f"  ollama pull {m}")
        return False
    return True


# ==========================================
# 2. 채점 — 관대하지도 각박하지도 않게
# ==========================================
def norm(s: str) -> str:
    """공백·문장부호·조사 꼬리를 정리해 비교 가능한 형태로 만든다."""
    s = re.sub(r"[\s\.,!?'\"“”‘’()\[\]]", "", str(s))
    return s.strip()


def first_line_answer(text: str) -> str:
    """모델이 설명을 덧붙여도 첫 줄(또는 '정답:' 뒤)만 답으로 본다."""
    t = text.strip()
    m = re.search(r"정답\s*[:：]\s*(.+)", t)
    if m:
        t = m.group(1)
    return t.splitlines()[0].strip() if t else ""


def says_no_answer(text: str) -> bool:
    """기권 표현을 폭넓게 인정한다 — 표현을 못 맞춰 오답 처리하는 일을 막기 위함."""
    t = str(text)
    keys = ["답할 수 없", "답변할 수 없", "알 수 없", "찾을 수 없", "정보가 없",
            "언급되지 않", "나와 있지 않", "제시되지 않", "확인할 수 없", "unanswerable"]
    return any(k in t for k in keys)


def span_match(pred: str, golds: list[str]) -> tuple[bool, bool]:
    """(완전일치, 부분일치)를 돌려준다."""
    p = norm(first_line_answer(pred))
    if not p:
        return False, False
    gs = [norm(g) for g in golds if str(g).strip()]
    exact = any(p == g for g in gs)
    partial = exact or any((g and (g in p or p in g)) for g in gs)
    return exact, partial


# ==========================================
# 3. 데이터 로딩
# ==========================================
def load_split(names: list[str], config, split: str):
    from datasets import load_dataset
    errs = []
    for n in names:
        try:
            return load_dataset(n, config, split=split) if config else load_dataset(n, split=split)
        except Exception as exc:
            errs.append(f"{n}: {exc}")
    raise RuntimeError("로딩 실패\n  " + "\n  ".join(errs))


def build_tasks(rng) -> list[dict]:
    """과제별 (프롬프트, 정답, 메타) 목록을 만든다."""
    tasks = []

    # ── YNAT: 주제 분류 ──────────────────────────────────────────────
    ynat = load_split(["klue/klue", "klue"], "ynat", "validation")
    labels = ynat.features["label"].names
    idx = rng.choice(len(ynat), min(N_YNAT, len(ynat)), replace=False).tolist()
    for i in idx:
        r = ynat[i]
        tasks.append({
            "task": "ynat", "axis": "분류", "item_id": str(r.get("guid", i)),
            "prompt": ("다음 뉴스 제목의 주제를 아래 보기 중 하나로만 답하세요. 다른 말은 쓰지 마세요.\n"
                       f"보기: {', '.join(labels)}\n제목: {r['title']}\n주제:"),
            "gold": [labels[r["label"]]], "unanswerable": False,
        })

    # ── KorQuAD 1.0: 추출형 QA ───────────────────────────────────────
    kq = load_split(["KorQuAD/squad_kor_v1", "squad_kor_v1"], None, "validation")
    idx = rng.choice(len(kq), min(N_KORQUAD, len(kq)), replace=False).tolist()
    for i in idx:
        r = kq[i]
        tasks.append({
            "task": "korquad", "axis": "추출", "item_id": str(r.get("id", i)),
            "prompt": ("아래 지문에서 질문의 답이 되는 부분만 그대로 뽑아 한 줄로 답하세요. "
                       "설명은 쓰지 마세요.\n"
                       f"지문: {r['context']}\n질문: {r['question']}\n정답:"),
            "gold": list(r["answers"]["text"]), "unanswerable": False,
        })

    # ── KLUE-MRC: 답 있음 / 답 없음 ──────────────────────────────────
    mrc = load_split(["klue/klue", "klue"], "mrc", "validation")
    imp = [i for i, r in enumerate(mrc) if r.get("is_impossible")]
    pos = [i for i, r in enumerate(mrc) if not r.get("is_impossible")]
    for group, pool, n, un in (("mrc_answerable", pos, N_MRC_ANSWERABLE, False),
                               ("mrc_unanswerable", imp, N_MRC_UNANSWERABLE, True)):
        if not pool:
            print(f"  경고: {group} 표본이 없습니다 (데이터셋 스키마 변경 가능성)")
            continue
        for i in rng.choice(pool, min(n, len(pool)), replace=False).tolist():
            r = mrc[i]
            tasks.append({
                "task": group, "axis": "기권" if un else "추출",
                "item_id": str(r.get("guid", i)),
                "prompt": ("아래 지문을 읽고 질문에 답하세요. 지문에 답이 없으면 "
                           f"반드시 「{NO_ANSWER_TOKEN}」이라고만 쓰세요. "
                           "지문에 없는 내용을 지어내지 마세요.\n"
                           f"지문: {r['context']}\n질문: {r['question']}\n정답:"),
                "gold": list(r["answers"]["text"]) if not un else [NO_ANSWER_TOKEN],
                "unanswerable": un,
            })
    return tasks


# ==========================================
# 4. 메인 — 과제 → 모델 → 문항
# ==========================================
def main() -> int:
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("datasets가 없습니다.  uv add datasets")
        return 1
    if not preflight():
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] KLUE·KorQuAD 벤치마크 시작")
    print("-> 데이터셋 로딩 중 (최초 실행 시 다운로드)")
    tasks = build_tasks(rng)
    by_task = {}
    for t in tasks:
        by_task.setdefault(t["task"], []).append(t)
    total = len(tasks) * len(MODELS)
    print(f"-> 과제 {len(by_task)}종, 문항 {len(tasks)}개 × 모델 {len(MODELS)} = {total}회\n")

    rows, done = [], 0
    # 과제를 바깥에 둔다(프롬프트 형식이 과제마다 다름). 모델은 로드 비용 때문에 블록.
    for task_name, items in by_task.items():
        print("#" * 64)
        print(f"# {task_name}  ({len(items)}문항 × {len(MODELS)}모델)")
        print("#" * 64)
        for model in MODELS:
            for other in MODELS:
                if other != model:
                    unload(other)
            print(f"  [{model}] 워밍업... ", end="", flush=True)
            w = run_query(model, WARMUP_PROMPT)
            print("완료" if w["success"] else f"실패({w['err']})")

            for it in items:
                done += 1
                res = run_query(model, it["prompt"])
                pred = res["text"]

                if it["unanswerable"]:
                    correct = says_no_answer(pred)
                    exact = partial = correct
                    mode = "기권 정답" if correct else "지어냄"
                elif it["task"] == "ynat":
                    p = norm(first_line_answer(pred))
                    exact = any(p == norm(g) for g in it["gold"])
                    partial = exact or any(norm(g) in p for g in it["gold"])
                    correct = exact or partial
                    mode = "정답" if correct else "오답"
                else:
                    exact, partial = span_match(pred, it["gold"])
                    # 답이 있는데 기권하면 과잉 기권으로 따로 표시
                    if not partial and says_no_answer(pred):
                        mode = "과잉 기권"
                    else:
                        mode = "정답" if partial else "오답"
                    correct = partial

                rows.append({
                    "task": it["task"], "axis": it["axis"], "model_name": model,
                    "item_id": it["item_id"], "unanswerable": it["unanswerable"],
                    "gold": " | ".join(map(str, it["gold"]))[:200],
                    "prediction": pred[:400], "exact": exact, "partial": partial,
                    "correct": correct, "mode": mode,
                    "total_time_sec": res["sec"], "is_success": res["success"],
                    "error_message": res["err"],
                })
                if done % 20 == 0 or done == total:
                    print(f"    진행 {done}/{total}")
            unload(model)

    # ══ 집계 ═════════════════════════════════════════════════════════
    def agg(task, model):
        rs = [r for r in rows if r["task"] == task and r["model_name"] == model]
        ok = [r for r in rs if r["is_success"]]
        n = len(rs)
        return {
            "n": n,
            "acc": round(sum(r["correct"] for r in ok) / n, 4) if n else None,
            "exact": round(sum(r["exact"] for r in ok) / n, 4) if n else None,
            "over_abstain": sum(1 for r in ok if r["mode"] == "과잉 기권"),
            "mean_sec": round(float(np.mean([r["total_time_sec"] for r in ok])), 2) if ok else None,
        }

    summary = {t: {m: agg(t, m) for m in MODELS} for t in by_task}
    results = {"generated_at": datetime.now().isoformat(), "models": MODELS,
               "seed": RANDOM_SEED, "summary": summary}

    L = []; add = L.append
    add("# KLUE·KorQuAD 벤치마크 결과")
    add("")
    add("본실험과 같은 호출 경로·같은 생성 설정(`temperature=0.1`)으로 수행했다.  ")
    add("**이 점수는 게이트에도 선정 가중치에도 넣지 않는다** — 후보 조사 단계의 사전 정보와 "
        "10문항 파일럿의 외부 타당도 점검에만 쓴다(설계서 §1, §4 규칙 1).")
    add("")
    label = {"ynat": "KLUE-YNAT (분류)", "korquad": "KorQuAD 1.0 (추출)",
             "mrc_answerable": "KLUE-MRC 답 있음 (추출)",
             "mrc_unanswerable": "KLUE-MRC 답 없음 (기권) ★"}
    add("| 과제 | 대응 축 | " + " | ".join(MODELS) + " |")
    add("| --- | --- |" + " --- |" * len(MODELS))
    for t in by_task:
        axis = by_task[t][0]["axis"]
        cells = []
        for m in MODELS:
            a = summary[t][m]
            cells.append(f"{a['acc']:.3f} ({int(a['acc']*a['n'])}/{a['n']})" if a["acc"] is not None else "-")
        add(f"| {label.get(t, t)} | {axis} | " + " | ".join(cells) + " |")
    add("")

    ua = summary.get("mrc_unanswerable")
    if ua:
        add("## 기권 축 — 이 실험의 핵심")
        add("")
        add("답이 없는 질문에 「답할 수 없음」이라고 물러났는지 본다. 지문에 없는 답을 지어내면 오답이다.")
        add("")
        add("| 모델 | 기권 정확도 | 지어낸 건수 |")
        add("| --- | --- | --- |")
        for m in MODELS:
            a = ua[m]
            add(f"| {m} | **{a['acc']:.3f}** ({int(a['acc']*a['n'])}/{a['n']}) | {a['n'] - int(a['acc']*a['n'])} |")
        add("")
        add("이 프로젝트의 10문항에서는 Q9 하나(모델당 2건)로만 판정하던 축이다. "
            "부록 D에서 프롬프트로 기권을 강제했더니 `llama3.2:3b`가 정상 문항의 절반을 "
            "기권으로 빠뜨렸다. 위 수치는 **그 모델이 기권이라는 행동 자체를 할 수 있는지**를 "
            "수십 건으로 보여준다.")
        add("")

    over = {t: {m: summary[t][m]["over_abstain"] for m in MODELS} for t in by_task
            if any(summary[t][m]["over_abstain"] for m in MODELS)}
    if over:
        add("### 과잉 기권 (답이 있는데 물러난 경우)")
        add("")
        add("| 과제 | " + " | ".join(MODELS) + " |")
        add("| --- |" + " --- |" * len(MODELS))
        for t, d in over.items():
            add(f"| {label.get(t, t)} | " + " | ".join(str(d[m]) for m in MODELS) + " |")
        add("")

    add("## 해석 시 지켜야 할 것")
    add("")
    add("- **도메인이 다르다.** KLUE는 뉴스·위키, KorQuAD는 위키다. 사내 CS 문의가 아니다. "
        "점수가 높다고 CS에서 잘한다는 보장이 없고 낮다고 실격도 아니다.")
    add("- **오염 가능성.** 두 데이터셋 모두 공개된 지 오래됐고 사전학습에 포함됐을 수 있다. "
        "절대값을 신뢰하지 말고 두 모델 간 상대 비교로만 읽는다.")
    add("- **선정을 바꾸지 않는다.** 선정은 게이트와 사전 고정 우선순위로 이미 끝났다. "
        "이 결과는 그 결정을 설명하는 보조 근거다.")
    add("- 10문항 결과와 방향이 어긋나면 어느 쪽도 결론으로 쓰지 않고 어긋남 자체를 보고한다.")
    add("")

    csv_path = OUTPUT_DIR / "11_benchmark_results.csv"
    json_path = OUTPUT_DIR / "11_benchmark_results.json"
    md_path = OUTPUT_DIR / "11_benchmark_summary.md"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({**results, "records": rows}, f, ensure_ascii=False, indent=2)
    md_path.write_text("\n".join(L), encoding="utf-8")

    print("\n" + "=" * 64)
    print("[과제별 정확도]")
    for t in by_task:
        line = "  ".join(f"{m}: {summary[t][m]['acc']:.3f}" for m in MODELS)
        print(f"  {label.get(t, t):<32} {line}")
    print("=" * 64)
    print(f"완료  |  {csv_path.name}, {json_path.name}, {md_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
