"""KLUE-STS·KLUE-NLI로 도전 실습 B의 임베딩을 검증한다 (생성 모델 호출 없음).

설계 근거는 KLUE_KORQUAD_DESIGN.md에 있다.

무엇을 검증하는가
  KLUE-STS  사람이 매긴 0~5 유사도와 우리 임베딩의 코사인 유사도가 얼마나 맞는가.
            높으면 도전 실습 B의 수치를 믿을 근거가 되고, 낮으면 전부 재검토해야 한다.
            지금까지는 임베딩 모델을 검증 없이 믿고 썼다.
  KLUE-NLI  '모순(contradiction)' 라벨이 붙은 쌍의 유사도를 본다. 모순 쌍도 유사도가
            높다면, 유사도로는 사실성·모순을 잡을 수 없다는 것이 라벨 데이터로 입증된다.
            도전 실습 B에서 손으로 고른 2쌍으로 주장한 것을 수백 건으로 정량화한다.

도전 실습 B와 동일한 임베딩 모델·동일 설정(정규화)을 쓴다. 이 조건이 깨지면 비교가 무의미하다.

사전 준비
  uv add datasets sentence-transformers

실행:  uv run python 10_klue_embedding_eval.py
출력:  work/10_klue_embedding/10_klue_sts_nli.json
       work/10_klue_embedding/10_klue_nli_pairs.csv
       work/10_klue_embedding/10_klue_embedding_report.md
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "work" / "10_klue_embedding"

# 도전 실습 B(09_embedding_experiment.py)와 반드시 같아야 한다.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NORMALIZE = True

NLI_PER_LABEL = 150     # 라벨당 표본 수 (함의/중립/모순)
STS_MAX = 0             # 0이면 검증셋 전체 사용
RANDOM_SEED = 20260918


def load_klue(config: str):
    """KLUE 검증셋을 불러온다. 허브 경로가 바뀐 경우를 대비해 후보를 순회한다."""
    from datasets import load_dataset
    errors = []
    for name in ("klue/klue", "klue"):
        try:
            return load_dataset(name, config, split="validation")
        except Exception as exc:  # 네트워크·경로·설정 변경 등
            errors.append(f"{name}: {exc}")
    raise RuntimeError("KLUE-" + config + " 로딩 실패\n  " + "\n  ".join(errors))


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    """순위 상관 — scipy 없이 순위로 바꿔 피어슨을 구한다."""
    def rank(v):
        v = np.asarray(v, float)
        order = v.argsort()
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # 동점 처리: 같은 값끼리 평균 순위
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return pearson(rank(x), rank(y))


def best_threshold(sim_pos, sim_neg):
    """유사도 하나로 두 집단을 가르는 최적 임계값과 그때의 정확도.

    sim_pos 는 '유사도가 높아야 하는' 집단(함의), sim_neg 는 낮아야 하는 집단(모순).
    """
    cands = np.unique(np.concatenate([sim_pos, sim_neg]))
    best = (0.0, 0.0)
    for t in cands:
        acc = ((np.asarray(sim_pos) >= t).sum() + (np.asarray(sim_neg) < t).sum()) \
              / (len(sim_pos) + len(sim_neg))
        if acc > best[1]:
            best = (float(t), float(acc))
    return best


def main() -> int:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers가 없습니다.  uv add sentence-transformers")
        return 1
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("datasets가 없습니다.  uv add datasets")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] KLUE 임베딩 검증 시작")
    print(f"-> 임베딩 모델: {EMBED_MODEL} (도전 실습 B와 동일)")
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()

    def embed(texts):
        return model.encode(list(texts), normalize_embeddings=NORMALIZE,
                            batch_size=64, show_progress_bar=False)

    def cos_pairs(a_list, b_list):
        va, vb = embed(a_list), embed(b_list)
        if NORMALIZE:
            return (va * vb).sum(axis=1)          # 정규화되어 있으면 내적 = 코사인
        na = np.linalg.norm(va, axis=1); nb = np.linalg.norm(vb, axis=1)
        return (va * vb).sum(axis=1) / (na * nb)

    results = {
        "generated_at": datetime.now().isoformat(),
        "embed_model": EMBED_MODEL, "embed_dim": dim, "normalize": NORMALIZE,
        "seed": RANDOM_SEED,
    }
    L = []; add = L.append
    add("# KLUE로 검증한 임베딩 — 도전 실습 B 보완")
    add("")
    add(f"임베딩 모델 `{EMBED_MODEL}` · 차원 {dim} · 정규화 {'사용' if NORMALIZE else '미사용'}  ")
    add("도전 실습 B와 **동일한 모델·동일 설정**으로 수행했다. 생성 모델은 호출하지 않는다.")
    add("")

    # ══ KLUE-STS ═════════════════════════════════════════════════════
    print("\n" + "=" * 60); print("KLUE-STS — 임베딩이 한국어 의미를 담는가"); print("=" * 60)
    sts = load_klue("sts")
    if STS_MAX and len(sts) > STS_MAX:
        idx = rng.choice(len(sts), STS_MAX, replace=False)
        sts = sts.select(idx.tolist())
    s1 = [r["sentence1"] for r in sts]
    s2 = [r["sentence2"] for r in sts]
    gold = [float(r["labels"]["label"]) for r in sts]     # 0~5 사람 점수
    sims = cos_pairs(s1, s2)
    p, sp = pearson(sims, gold), spearman(sims, gold)
    results["klue_sts"] = {"n": len(sts), "pearson": round(p, 4), "spearman": round(sp, 4),
                           "cosine_mean": round(float(sims.mean()), 4)}
    print(f"  표본 {len(sts)}쌍 | 피어슨 {p:.4f} | 스피어만 {sp:.4f}")

    verdict_sts = ("이 임베딩 모델의 한국어 의미 판단을 신뢰할 만하다" if sp >= 0.7 else
                   "상관이 중간 수준이다. 도전 실습 B의 수치를 절대값으로 읽지 말아야 한다" if sp >= 0.5 else
                   "상관이 낮다. 도전 실습 B의 유사도 해석을 재검토해야 한다")
    add("## KLUE-STS — 임베딩 타당도 검증")
    add("")
    add("사람이 매긴 0~5 유사도 점수와 우리 임베딩의 코사인 유사도가 얼마나 일치하는지 본다.")
    add("")
    add("| 표본 | 피어슨 상관 | 스피어만 상관 |")
    add("| --- | --- | --- |")
    add(f"| {len(sts)}쌍 | **{p:.4f}** | **{sp:.4f}** |")
    add("")
    add(f"**판정: {verdict_sts}.**")
    add("")
    add("도전 실습 B에서는 이 모델을 검증 없이 썼다. 이 수치가 그 자리를 메운다.")
    add("")

    # ══ KLUE-NLI ═════════════════════════════════════════════════════
    print("\n" + "=" * 60); print("KLUE-NLI — 유사도로 모순을 잡을 수 있는가"); print("=" * 60)
    nli = load_klue("nli")
    names = nli.features["label"].names          # ['entailment','neutral','contradiction']
    ko = {"entailment": "함의 (뜻이 이어짐)", "neutral": "중립", "contradiction": "모순 (뜻이 반대)"}

    picked, rows = {}, []
    for li, lname in enumerate(names):
        idx = [i for i, r in enumerate(nli) if r["label"] == li]
        take = rng.choice(idx, min(NLI_PER_LABEL, len(idx)), replace=False).tolist()
        sub = nli.select(take)
        sims_l = cos_pairs([r["premise"] for r in sub], [r["hypothesis"] for r in sub])
        picked[lname] = sims_l
        for r, s in zip(sub, sims_l):
            rows.append({"label": lname, "label_ko": ko.get(lname, lname),
                         "premise": r["premise"], "hypothesis": r["hypothesis"],
                         "cosine": round(float(s), 4)})
        print(f"  {ko.get(lname, lname):<18} n={len(sub):<4} 평균 유사도 {sims_l.mean():.4f}")

    ent, con = picked.get("entailment"), picked.get("contradiction")
    thr, acc = best_threshold(ent, con)
    overlap = float(np.mean(con >= np.median(ent)))    # 모순인데 함의 중앙값보다 높은 비율
    results["klue_nli"] = {
        "n_per_label": {k: int(len(v)) for k, v in picked.items()},
        "mean_cosine": {k: round(float(v.mean()), 4) for k, v in picked.items()},
        "best_threshold": round(thr, 4),
        "best_accuracy_entail_vs_contradict": round(acc, 4),
        "contradiction_above_entailment_median": round(overlap, 4),
    }
    print(f"\n  함의 vs 모순을 유사도 하나로 가른 최고 정확도: {acc:.3f} (임계값 {thr:.4f})")
    print(f"  모순인데 함의 중앙값보다 유사도가 높은 비율: {overlap:.1%}")

    add("## KLUE-NLI — 유사도가 모순을 잡는가")
    add("")
    add("'모순'은 두 문장의 뜻이 정반대라는 정답 라벨이다. 유사도가 사실성·정합성을 잰다면 "
        "모순 쌍은 낮게 나와야 한다.")
    add("")
    add("| 라벨 | 표본 | 평균 코사인 유사도 |")
    add("| --- | --- | --- |")
    for lname in names:
        v = picked[lname]
        add(f"| {ko.get(lname, lname)} | {len(v)} | **{v.mean():.4f}** |")
    add("")
    add(f"- 함의와 모순을 **코사인 유사도 하나로** 가른 최고 정확도: **{acc:.3f}** (임계값 {thr:.4f})")
    add(f"- 모순 쌍인데 함의 쌍의 중앙값보다 유사도가 높은 비율: **{overlap:.1%}**")
    add("")
    if acc < 0.75:
        add(f"**유사도만으로는 모순을 가려낼 수 없다.** 최적 임계값을 골라도 정확도가 {acc:.3f}에 그친다. "
            "모순 쌍이 같은 소재를 다루기 때문에 표현이 닮아서다. 도전 실습 B에서 반의어 2쌍으로 "
            f"주장했던 것이 라벨 데이터 {len(ent) + len(con)}건으로 확인됐다.")
    else:
        add(f"예상과 달리 유사도가 모순을 비교적 잘 가른다(정확도 {acc:.3f}). "
            "도전 실습 B의 결론을 이 데이터에 맞춰 수정해야 한다. 다만 임계값을 결과를 보고 "
            "고른 값이므로 실제 운용 성능은 이보다 낮다.")
    add("")
    add("**어느 쪽이든 유사도를 게이트 판정에 단독으로 쓰지 않는다.** 임계값을 데이터를 보고 "
        "고르는 것 자체가 실전에서는 불가능하고, 이 프로젝트의 게이트는 '권한 밖 확답 금지'처럼 "
        "의미 유사도와 다른 축이기 때문이다.")
    add("")
    add("## 한계")
    add("")
    add("- KLUE는 뉴스·위키 도메인이다. 사내 CS 문의가 아니다.")
    add("- 공개 데이터라 임베딩 모델의 학습에 포함됐을 수 있다. 상관계수가 실제보다 높게 나올 수 있다.")
    add("- 임계값 정확도는 정답을 보고 최적값을 고른 것이므로 상한값이다.")
    add("")

    # ══ 저장 ═════════════════════════════════════════════════════════
    json_path = OUTPUT_DIR / "10_klue_sts_nli.json"
    csv_path = OUTPUT_DIR / "10_klue_nli_pairs.csv"
    md_path = OUTPUT_DIR / "10_klue_embedding_report.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "label_ko", "premise", "hypothesis", "cosine"])
        w.writeheader(); w.writerows(rows)
    md_path.write_text("\n".join(L), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"완료  |  {json_path.name}, {csv_path.name}, {md_path.name}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
