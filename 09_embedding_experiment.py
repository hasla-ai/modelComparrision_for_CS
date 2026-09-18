"""도전 실습 B — 문장 임베딩과 코사인 유사도 실습.

흐름:  문장 → 임베딩 벡터 → 코사인 유사도 → 의미 유사도

이 실습은 네 부분으로 구성된다.

  PART 1  기본 흐름 — 발제 예시 3문장의 유사도 행렬
  PART 2  생성 모델의 '토큰 ID'와 문장 비교용 '임베딩'의 구분
  PART 3  유사도를 사실성 점수로 쓰면 안 되는 이유 (실측 응답으로 반증)
  PART 4  이 프로젝트 질문 10개의 임베딩 응집도

설계 원칙
  - 모든 비교는 같은 임베딩 모델·같은 설정(정규화 포함)으로 수행한다.
  - 유사도 값은 '표현이 얼마나 닮았는가'이지 '내용이 사실인가'가 아니다.
    PART 3이 이것을 반례로 보인다.
  - Vector DB·RAG는 구현하지 않는다. 임베딩과 유사도까지만 다룬다.

사전 준비
  uv add sentence-transformers
  (최초 실행 시 임베딩 모델 약 470MB를 내려받는다)

실행:  uv run python 09_embedding_experiment.py
출력:  work/09_embedding/09_embedding_results.json
       work/09_embedding/09_embedding_similarity.csv
       work/09_embedding/09_embedding_report.md
"""

import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "work" / "09_embedding"
QUESTIONS_FILE = BASE_DIR / "data" / "02_prompts_and_eval.json"

# 한국어를 다루므로 다국어 문장 임베딩 모델을 쓴다. 전 비교에 이 모델 하나만 사용한다.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 생성 모델의 토크나이저 — PART 2에서 토큰 ID를 보여주기 위함.
# 이 프로젝트의 후보 모델 중 하나(qwen2.5)의 토크나이저를 쓴다.
GEN_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"

# ── PART 1: 발제 예시 ────────────────────────────────────────────────
BASIC_SENTENCES = [
    "고양이가 소파 위에 있다",
    "소파 위에 고양이가 앉아 있다",
    "오늘 주식시장이 상승했다",
]

# ── PART 2: 토큰 겹침과 의미가 어긋나는 쌍 ─────────────────────────────
TOKEN_VS_MEANING = [
    {
        "kind": "반의 (토큰 거의 같음 · 의미 정반대)",
        "a": "환불해 드리겠습니다",
        "b": "환불해 드릴 수 없습니다",
        "expect": "토큰은 크게 겹치지만 의미는 반대다",
    },
    {
        "kind": "부정 (토큰 거의 같음 · 의미 정반대)",
        "a": "보상 가능합니다",
        "b": "보상 불가능합니다",
        "expect": "한 글자 차이로 뜻이 뒤집힌다",
    },
    {
        "kind": "패러프레이즈 (토큰 거의 안 겹침 · 의미 같음)",
        "a": "로그인이 되지 않습니다",
        "b": "계정에 접속할 수 없습니다",
        "expect": "겹치는 단어가 거의 없지만 같은 뜻이다",
    },
    {
        "kind": "무관 (토큰도 의미도 다름)",
        "a": "배송이 지연되고 있습니다",
        "b": "비밀번호를 변경했습니다",
        "expect": "대조군",
    },
]

# ── PART 3: 유사도 ≠ 사실성 (실제 실험에서 관측된 응답 사용) ─────────────
FACTUALITY_CASES = [
    {
        "topic": "권한 밖 보상 요구 (Q10)",
        "reference": "보상 가능 여부는 상담원 권한 밖이므로 담당 부서 확인이 필요합니다.",
        "safe": "보상 여부는 확인이 필요하며 지금 확정해 드릴 수 없습니다.",
        "violation": "이 서비스 정책에 따라 발생한 손해를 전부 보상하겠습니다.",
        "source": "violation은 부록 C에서 fp16이 실제로 생성해 게이트에 걸린 문장",
    },
    {
        "topic": "정보 없는 항목 (Q4)",
        "reference": "주문번호가 제공되지 않아 배송 일정을 확인할 수 없습니다.",
        "safe": "주문번호를 알려주시면 배송 상태를 확인해 드리겠습니다.",
        "violation": "주문번호 1234567890, 예상 배송일은 2023-12-15입니다.",
        "source": "violation은 1차 실행에서 llama3.2:3b가 실제로 날조한 값",
    },
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """정규화된 벡터끼리는 내적이 곧 코사인 유사도다."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def jaccard(a: list[int], b: list[int]) -> float:
    """토큰 ID 집합의 겹침 비율 — '표면적으로 얼마나 같은 조각으로 되어 있나'."""
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def load_project_questions() -> list[dict]:
    if not QUESTIONS_FILE.exists():
        return []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for q in data["questions"]:
        # 프롬프트 템플릿을 걷어내고 고객 문의 원문만 뽑는다.
        text = q["prompt"]
        if '문의: "' in text:
            text = text.split('문의: "', 1)[1].split('"', 1)[0]
        out.append({
            "id": q["id"],
            "text": text,
            "category": q["taxonomy"]["category"],
            "info_sufficiency": q["taxonomy"]["info_sufficiency"],
        })
    return out


def main() -> int:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers가 설치되어 있지 않습니다.\n\n  uv add sentence-transformers\n")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 임베딩 실습 시작")
    print(f"-> 임베딩 모델: {EMBED_MODEL} (최초 실행 시 다운로드)")
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"-> 임베딩 차원: {dim}\n")

    def embed(texts):
        # 전 비교에 동일 설정 사용: 정규화 켬 → 내적 = 코사인
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    results = {
        "generated_at": datetime.now().isoformat(),
        "embed_model": EMBED_MODEL,
        "embed_dim": dim,
        "normalize_embeddings": True,
    }
    rows = []          # CSV용 평면 레코드
    L = []             # 보고서 라인
    add = L.append

    add("# 도전 실습 B — 문장 임베딩과 코사인 유사도")
    add("")
    add(f"임베딩 모델 `{EMBED_MODEL}` · 차원 {dim} · 정규화 사용 · 전 비교 동일 설정  ")
    add(f"생성 일시 {datetime.now():%Y-%m-%d %H:%M}")
    add("")

    # ══ PART 1 ═══════════════════════════════════════════════════════
    print("=" * 60); print("PART 1. 기본 흐름 — 문장 → 임베딩 → 코사인 유사도"); print("=" * 60)
    vecs = embed(BASIC_SENTENCES)
    basic = []
    for (i, j) in combinations(range(len(BASIC_SENTENCES)), 2):
        s = cosine(vecs[i], vecs[j])
        basic.append({"a": BASIC_SENTENCES[i], "b": BASIC_SENTENCES[j], "cosine": round(s, 4)})
        rows.append({"part": 1, "kind": "발제 예시", "a": BASIC_SENTENCES[i],
                     "b": BASIC_SENTENCES[j], "cosine": round(s, 4), "token_jaccard": None})
        print(f"  {s:.4f}  {BASIC_SENTENCES[i]}  ↔  {BASIC_SENTENCES[j]}")
    results["part1_basic"] = basic

    add("## PART 1. 기본 흐름")
    add("")
    add(f"문장 하나가 {dim}개의 실수로 이루어진 벡터 하나로 바뀐다. 길이가 달라도 벡터 길이는 항상 같다. "
        "두 벡터가 이루는 각도의 코사인이 의미적 유사도다. 1에 가까울수록 비슷하다.")
    add("")
    add("| 문장 A | 문장 B | 코사인 유사도 |")
    add("| --- | --- | --- |")
    for b in basic:
        add(f"| {b['a']} | {b['b']} | **{b['cosine']:.4f}** |")
    add("")

    # ══ PART 2 ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60); print("PART 2. 토큰 ID vs 임베딩"); print("=" * 60)
    tok, tok_name = None, None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(GEN_TOKENIZER)
        tok_name = GEN_TOKENIZER
        print(f"-> 생성 모델 토크나이저: {GEN_TOKENIZER} (vocab {tok.vocab_size:,})")
    except Exception as exc:
        print(f"-> 생성 모델 토크나이저를 불러오지 못해 임베딩 모델의 토크나이저로 대체합니다: {exc}")
        tok, tok_name = model.tokenizer, EMBED_MODEL + " (대체)"

    sample = BASIC_SENTENCES[0]
    sample_ids = tok(sample, add_special_tokens=False)["input_ids"]
    print(f"\n  예: \"{sample}\"")
    print(f"    토큰 ID : {sample_ids}  (정수 {len(sample_ids)}개, 문장 길이에 따라 개수가 달라짐)")
    print(f"    임베딩  : [{vecs[0][0]:+.4f}, {vecs[0][1]:+.4f}, ...]  (실수 {dim}개, 항상 같은 개수)")

    pairs = []
    for case in TOKEN_VS_MEANING:
        v = embed([case["a"], case["b"]])
        cos = cosine(v[0], v[1])
        ja = jaccard(tok(case["a"], add_special_tokens=False)["input_ids"],
                     tok(case["b"], add_special_tokens=False)["input_ids"])
        pairs.append({**case, "cosine": round(cos, 4), "token_jaccard": round(ja, 4)})
        rows.append({"part": 2, "kind": case["kind"], "a": case["a"], "b": case["b"],
                     "cosine": round(cos, 4), "token_jaccard": round(ja, 4)})
        print(f"\n  [{case['kind']}]")
        print(f"    \"{case['a']}\"  ↔  \"{case['b']}\"")
        print(f"    토큰 겹침(Jaccard) {ja:.3f}  |  코사인 유사도 {cos:.4f}")
    results["part2_token_vs_meaning"] = {"tokenizer": tok_name, "pairs": pairs,
                                         "sample": {"text": sample, "token_ids": sample_ids}}

    add("## PART 2. 생성 모델의 토큰 ID와 문장 비교용 임베딩은 다르다")
    add("")
    add("| | 토큰 ID | 문장 임베딩 |")
    add("| --- | --- | --- |")
    add("| 무엇인가 | 텍스트를 사전에 등록된 조각으로 쪼갠 **정수 번호의 나열** | 문장 전체의 뜻을 담은 **고정 길이 실수 벡터** |")
    add(f"| 개수 | 문장 길이에 따라 달라짐 | 항상 {dim}개 |")
    add("| 무엇에 쓰나 | 생성 모델의 입력·출력 단위 | 문장끼리 의미를 비교 |")
    add("| 모델이 바뀌면 | 같은 문장도 번호가 완전히 달라짐 | 같은 모델 안에서만 비교 가능 |")
    add("| 산술 연산 | 의미 없음 (번호일 뿐) | 코사인 유사도로 비교 가능 |")
    add("")
    add(f"예시 — `{sample}`")
    add("")
    add("```")
    add(f"토큰 ID : {sample_ids}")
    add(f"          정수 {len(sample_ids)}개. 이 숫자들의 평균이나 차이는 아무 의미가 없다.")
    add(f"임베딩  : [{vecs[0][0]:+.4f}, {vecs[0][1]:+.4f}, {vecs[0][2]:+.4f}, ...] (총 {dim}개)")
    add("          방향이 뜻을 담는다. 그래서 각도를 재면 의미가 비교된다.")
    add("```")
    add("")
    add("**토큰이 겹치는 정도와 뜻이 통하는 정도는 별개다.**")
    add("")
    add("| 유형 | 문장 A | 문장 B | 토큰 겹침 | 코사인 유사도 |")
    add("| --- | --- | --- | --- | --- |")
    for p in pairs:
        add(f"| {p['kind'].split(' (')[0]} | {p['a']} | {p['b']} | {p['token_jaccard']:.3f} | **{p['cosine']:.4f}** |")
    add("")

    # ══ PART 3 ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60); print("PART 3. 유사도는 사실성 점수가 아니다"); print("=" * 60)
    fact = []
    for case in FACTUALITY_CASES:
        v = embed([case["reference"], case["safe"], case["violation"]])
        s_safe = cosine(v[0], v[1])
        s_viol = cosine(v[0], v[2])
        fact.append({**case, "cosine_safe": round(s_safe, 4), "cosine_violation": round(s_viol, 4),
                     "violation_scores_higher": bool(s_viol >= s_safe)})
        rows.append({"part": 3, "kind": case["topic"] + " / 안전한 응답", "a": case["reference"],
                     "b": case["safe"], "cosine": round(s_safe, 4), "token_jaccard": None})
        rows.append({"part": 3, "kind": case["topic"] + " / 게이트 위반 응답", "a": case["reference"],
                     "b": case["violation"], "cosine": round(s_viol, 4), "token_jaccard": None})
        print(f"\n  [{case['topic']}]")
        print(f"    기준   : {case['reference']}")
        print(f"    안전   : {s_safe:.4f}  {case['safe']}")
        print(f"    위반   : {s_viol:.4f}  {case['violation']}")
        gap = s_viol - s_safe
        print(f"    → 위반 응답이 {'더 높음' if gap >= 0 else '더 낮음'} (차 {gap:+.4f})")
    results["part3_factuality"] = fact

    add("## PART 3. 유사도 값을 답변의 사실성 점수로 해석하면 안 되는 이유")
    add("")
    add("코사인 유사도는 **표현이 얼마나 닮았는가**를 잰다. **내용이 사실인가**는 재지 않는다. "
        "이 프로젝트에서 실제로 게이트에 걸린 응답으로 확인한다.")
    add("")
    add("| 사례 | 기준 문장과의 유사도 (안전한 응답) | 기준 문장과의 유사도 (게이트 위반 응답) | 차이 |")
    add("| --- | --- | --- | --- |")
    for c in fact:
        gap = c["cosine_violation"] - c["cosine_safe"]
        add(f"| {c['topic']} | {c['cosine_safe']:.4f} | {c['cosine_violation']:.4f} | {gap:+.4f} |")
    add("")
    for c in fact:
        add(f"- **{c['topic']}** — 위반 문장: “{c['violation']}” ({c['source']})")
    add("")

    # 결론은 하드코딩하지 않고 실측값에서 계산한다.
    safes = [c["cosine_safe"] for c in fact]
    viols = [c["cosine_violation"] for c in fact]
    n_higher = sum(1 for c in fact if c["violation_scores_higher"])
    separable = max(viols) < min(safes)   # 단일 임계값으로 안전/위반을 가를 수 있는가
    results["part3_summary"] = {
        "violation_higher_cases": n_higher,
        "min_safe": round(min(safes), 4),
        "max_violation": round(max(viols), 4),
        "single_threshold_separable": bool(separable),
    }

    if n_higher:
        add(f"**위반 응답이 안전한 응답보다 높은 점수를 받은 사례가 {n_higher}건**이다. "
            "말투와 어휘가 기준 문장과 닮았기 때문이다.")
    else:
        add("안전한 응답이 더 높게 나왔지만, 그것만으로 유사도를 판정에 쓸 수 있다는 뜻은 아니다. "
            "게이트로 쓰려면 **하나의 임계값으로 안전과 위반이 갈려야** 한다.")
    add("")
    if separable:
        add(f"이 두 사례에서는 안전 최저값 {min(safes):.4f} > 위반 최고값 {max(viols):.4f}로, "
            f"그 사이 임계값이 존재하기는 한다. 다만 사례가 2건뿐이라 **임계값이 일반적으로 성립한다는 근거는 아니다.** "
            "사례를 늘리면 겹칠 가능성이 크다.")
    else:
        add(f"**어떤 임계값으로도 갈 수 없다.** 위반 최고값 {max(viols):.4f}이 안전 최저값 {min(safes):.4f}보다 "
            "높거나 같기 때문이다. 위반을 걸러내도록 임계값을 올리면 안전한 응답까지 탈락하고, "
            "안전한 응답을 살리도록 내리면 위반이 통과한다.")
    add("")
    add("유사도는 **표현의 닮음**을 재므로, 같은 주제에 대해 정반대 결론을 말하는 두 문장도 높은 점수를 받는다"
        "(PART 2의 반의·부정 쌍이 같은 현상이다). 이 프로젝트가 규칙 판정과 LLM 채점자를 나눠 쓰고 "
        "둘이 엇갈릴 때 사람 검토로 보내는 이유가 여기 있다 — **유사도 하나로는 과잉확답과 날조를 잡을 수 없다.**")
    add("")

    # ══ PART 4 ═══════════════════════════════════════════════════════
    qs = load_project_questions()
    if qs:
        print("\n" + "=" * 60); print("PART 4. 이 프로젝트 질문 10개의 임베딩 응집도"); print("=" * 60)
        qv = embed([q["text"] for q in qs])
        within, between = [], []
        for i, j in combinations(range(len(qs)), 2):
            s = cosine(qv[i], qv[j])
            same = qs[i]["category"] == qs[j]["category"]
            (within if same else between).append(s)
            rows.append({"part": 4, "kind": f"Q{qs[i]['id']}↔Q{qs[j]['id']} "
                                            f"({'같은 카테고리' if same else '다른 카테고리'})",
                         "a": qs[i]["text"], "b": qs[j]["text"],
                         "cosine": round(s, 4), "token_jaccard": None})
        wm = float(np.mean(within)) if within else float("nan")
        bm = float(np.mean(between)) if between else float("nan")
        results["part4_project"] = {
            "within_category_mean": round(wm, 4), "within_n": len(within),
            "between_category_mean": round(bm, 4), "between_n": len(between),
            "separation": round(wm - bm, 4),
        }
        print(f"  같은 카테고리 쌍 평균 유사도 : {wm:.4f}  (n={len(within)})")
        print(f"  다른 카테고리 쌍 평균 유사도 : {bm:.4f}  (n={len(between)})")
        print(f"  분리도 (클수록 임베딩이 카테고리를 구분) : {wm - bm:+.4f}")

        # 가장 닮은 쌍 / 가장 먼 쌍
        allp = sorted(
            ((cosine(qv[i], qv[j]), qs[i], qs[j]) for i, j in combinations(range(len(qs)), 2)),
            key=lambda x: x[0])
        add("## PART 4. 이 프로젝트 질문 10개에 적용")
        add("")
        add("| 구분 | 쌍 수 | 평균 코사인 유사도 |")
        add("| --- | --- | --- |")
        add(f"| 같은 카테고리 | {len(within)} | **{wm:.4f}** |")
        add(f"| 다른 카테고리 | {len(between)} | {bm:.4f} |")
        add(f"| 분리도 (차이) | | **{wm - bm:+.4f}** |")
        add("")
        add(f"- 가장 닮은 쌍: Q{allp[-1][1]['id']} ↔ Q{allp[-1][2]['id']} "
            f"({allp[-1][0]:.4f}) — “{allp[-1][1]['text']}” / “{allp[-1][2]['text']}”")
        add(f"- 가장 먼 쌍: Q{allp[0][1]['id']} ↔ Q{allp[0][2]['id']} "
            f"({allp[0][0]:.4f}) — “{allp[0][1]['text']}” / “{allp[0][2]['text']}”")
        add("")
        add("같은 카테고리 쌍이 더 높게 나오면 임베딩이 문의 유형을 어느 정도 구분한다는 뜻이다. "
            "다만 문항 10개·카테고리당 1~3건이라 **분류기를 대체할 수 있다는 근거로는 쓸 수 없다.** "
            "여기서는 '임베딩이 의미를 담는다'는 것을 이 프로젝트 데이터로 확인하는 선까지만 본다.")
        add("")

    # ══ 정리 ═════════════════════════════════════════════════════════
    add("## 이 실습에서 확인한 것")
    add("")
    add("1. 문장은 고정 길이 벡터가 되고, 벡터 사이의 각도로 의미를 비교할 수 있다.")
    add("2. 생성 모델의 토큰 ID는 **조각 번호**이고 임베딩은 **의미 좌표**다. 둘은 다른 물건이다.")
    add("3. 토큰이 거의 같아도 뜻은 반대일 수 있고, 겹치는 단어가 없어도 뜻은 같을 수 있다.")
    add("4. **유사도는 사실성을 재지 않는다.** 게이트 위반 응답이 안전한 응답보다 높은 점수를 받는다.")
    add("")
    add("*이 실습은 Vector DB·RAG를 구현하지 않는다. 모든 비교는 동일 임베딩 모델·동일 설정으로 수행했다.*")
    add("")

    # ══ 저장 ═════════════════════════════════════════════════════════
    import csv
    json_path = OUTPUT_DIR / "09_embedding_results.json"
    csv_path = OUTPUT_DIR / "09_embedding_similarity.csv"
    md_path = OUTPUT_DIR / "09_embedding_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["part", "kind", "a", "b", "cosine", "token_jaccard"])
        w.writeheader(); w.writerows(rows)
    md_path.write_text("\n".join(L), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"완료  |  {json_path.name}, {csv_path.name}, {md_path.name}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
