"""평가셋 v1(10문항) → v2(54문항) 확장 빌더.

왜 하는가
  최종 보고서 §8이 최대 한계로 적어둔 것이 taxonomy 커버리지 갭이다.
  카테고리 6종 × 정보충분도 3종 = 18칸 중 9칸만 실제로 검증됐고, 결제·배송·
  컴플레인의 경계·특수 사례는 한 번도 측정되지 않았다. 공개 벤치마크(KLUE 등)로
  이 자리를 대신하려 했으나, 뉴스 주제 분류와 위키 지문 독해는 CS 문의 분류·필드
  추출과 같은 구성(construct)을 재지 않는다. 대리 지표를 고르는 대신 도메인
  평가셋을 직접 채우는 것이 옳다.

설계 원칙
  1. 기존 ID 1~10은 한 글자도 바꾸지 않는다. 지금까지의 모든 실험 결과(본실험
     40회, 양자화 60회, A/B 80회, Cloud 5회)가 그대로 유효해야 하기 때문이다.
  2. 프롬프트 템플릿도 그대로 둔다. 문항 수만 늘리는 순수 표본 확장이다.
     (부록 D가 제안한 '카테고리 열거형 제시'는 프롬프트 변경이므로 조건 C로
      따로 실험한다. 두 변경을 섞으면 무엇이 효과를 냈는지 알 수 없다.)
  3. Cloud 비교 5문항(ID 1·2·6·9·10)도 그대로 둔다. 결과를 보기 전에 선정한
     집합이므로 사후에 바꾸면 사전 선정의 의미가 사라진다.
  4. 18칸을 각 3건으로 채운다. 칸마다 같은 수를 두어야 칸 간 비교가 공정하다.

taxonomy.category 와 expected_category 의 구분 (중요)
  taxonomy.category 는 '이 문의가 무엇에 관한 것인가'(커버리지 회계용)이고,
  expected_category 는 '시스템이 무엇을 출력해야 하는가'(채점용)다. 둘은 다를 수
  있다. 예: 「로그인이요.」는 taxonomy 상 로그인/계정이지만, 증상도 요청도 없어
  담당자에게 라우팅할 수 없으므로 expected_category 는 "정보 부족"이다.
  기존 ID 9가 이미 이 규약을 따르고 있다.

실행:  uv run python build_eval_set_v2.py
출력:  data/02_prompts_and_eval.json           (v2로 교체)
       data/02_prompts_and_eval.v1.json        (v1 백업)
       data/eval_set_coverage.md               (커버리지 보고서)
종료:  0 정상 / 1 검증 실패 (이때 파일을 쓰지 않는다)
"""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data" / "02_prompts_and_eval.json"
BACKUP = BASE_DIR / "data" / "02_prompts_and_eval.v1.json"
COVERAGE = BASE_DIR / "data" / "eval_set_coverage.md"

NEW_VERSION = "v2-2026-09"
CHANGELOG = (
    "v1(10문항) → v2(54문항). taxonomy 18칸을 각 3건으로 채움. "
    "ID 1~10과 프롬프트 템플릿, Cloud 비교 5문항(1·2·6·9·10)은 변경하지 않음. "
    "사유: 최종 보고서 §8의 커버리지 갭(18칸 중 9칸 미검증) 해소. "
    "공개 벤치마크는 구성 타당도가 달라 대리 지표로 쓰지 않기로 함."
)

EVAL_CRITERIA = "분류, 핵심 정보 추출, 요약, 예외 처리, 출력 형식 준수"
NOTE_SUFFICIENT = "정보 충분 — note는 비어 있거나 없어도 됨"
NOTE_INSUFFICIENT = (
    "핵심 정보가 전혀 없음을 note에 반드시 명시하고 추가 질문이 필요함을 표시해야 함. "
    "카테고리 임의 단정 시 게이트 감점"
)

PROMPT_TEMPLATE = (
    '다음 고객 문의를 분류하고, 핵심 정보를 JSON으로 추출하세요.\n'
    '문의: "{text}"\n'
    '출력 형식:\n'
    '- category: 문의 유형\n'
    '- summary: 한 문장 요약\n'
    '- info: 핵심 정보 JSON\n'
    '- note: 정보 부족 또는 예외 사항이 있으면 기록'
)

# (taxonomy_category, info_sufficiency, 문의 원문, expected_category,
#  expected_fields, key_facts_for_summary, note_behavior)
# note_behavior 가 None 이면 info_sufficiency 로부터 자동 결정한다.
NEW_ITEMS = [
    # ── 로그인/계정 ────────────────────────────────────────────────────
    ("로그인/계정", "정상", "아이디는 기억나는데 비밀번호가 기억나지 않습니다. 재설정 방법을 알려주세요.",
     "로그인/계정", ["증상", "요청사항"], ["비밀번호 분실", "재설정 방법 문의"], None),
    ("로그인/계정", "경계", "어제까지는 로그인이 됐는데 오늘은 안 됩니다. 비밀번호는 바꾸지 않았습니다.",
     "로그인/계정", ["증상", "발생 시점"], ["어제까지 정상 로그인", "오늘부터 로그인 불가", "비밀번호 미변경"],
     "오류 메시지와 접속 환경(기기·브라우저) 정보 부족을 note에 명시해야 함"),
    ("로그인/계정", "경계", "가끔 로그인하면 바로 자동 로그아웃됩니다. 매번 그런 건 아닙니다.",
     "로그인/계정", ["증상"], ["간헐적 자동 로그아웃", "재현 조건 불명확"],
     "발생 빈도·재현 조건 정보 부족을 note에 명시해야 함"),
    ("로그인/계정", "특수: 정보부족", "로그인이요.",
     "정보 부족", [], [], None),
    ("로그인/계정", "특수: 정보부족", "비번요",
     "정보 부족", [], [], None),
    ("로그인/계정", "특수: 범위밖", "제 가족 계정입니다. 그 계정 비밀번호를 알려주세요.",
     "로그인/계정", ["요청사항"], ["타인 계정 비밀번호 제공 요구", "가족 관계 주장"],
     "타인 계정 정보 제공은 본인 확인 절차 없이 불가하며 상담원 권한 밖임을 note에 명시해야 함. "
     "비밀번호를 알려주겠다고 약속하면 게이트 탈락(과잉확답)"),

    # ── 결제/환불 ──────────────────────────────────────────────────────
    ("결제/환불", "정상", "지난달에 결제된 금액이 이번 달에 또 청구됐습니다. 중복 결제인지 확인 부탁드립니다.",
     "결제/환불", ["증상", "요청사항"], ["동일 금액 중복 청구", "중복 결제 확인 요청"], None),
    ("결제/환불", "정상", "구독을 해지했는데 환불이 언제 처리되는지 알고 싶습니다.",
     "결제/환불", ["증상", "요청사항"], ["구독 해지 완료", "환불 처리 일정 문의"], None),
    ("결제/환불", "경계", "결제가 이상하게 된 것 같은데 금액이 맞는지 잘 모르겠습니다.",
     "결제/환불", ["증상"], ["결제 금액 이상 의심", "구체적 금액·일시 불명"],
     "결제 일시와 금액 정보 부족을 note에 명시해야 함"),
    ("결제/환불", "경계", "환불 신청을 했는데 아직 안 된 것 같아요. 언제 신청했는지는 기억이 안 납니다.",
     "결제/환불", ["증상"], ["환불 미처리 주장", "신청 시점 불명"],
     "환불 신청 일시와 주문 정보 부족을 note에 명시해야 함"),
    ("결제/환불", "경계", "카드가 안 긁히는 건지 앱이 문제인 건지 모르겠습니다.",
     "결제/환불", ["증상"], ["결제 실패", "원인이 결제수단인지 앱인지 불명"],
     "결제 수단과 기능 오류 중 어느 쪽인지 판단할 정보가 부족함을 note에 명시해야 함. "
     "카테고리가 결제/환불과 기능 오류에 걸쳐 있음"),
    ("결제/환불", "특수: 정보부족", "결제요.",
     "정보 부족", [], [], None),
    ("결제/환불", "특수: 범위밖", "환불 규정이 30일이라는데 저는 6개월 전 결제 건도 환불해 주세요.",
     "결제/환불", ["요청사항"], ["규정 기간 경과 건의 환불 요구", "규정 예외 적용 요청"],
     "규정 예외 적용 여부는 상담원 권한 밖임을 note에 명시해야 함. "
     "환불을 확정적으로 약속하면 게이트 탈락(과잉확답)"),
    ("결제/환불", "특수: 범위밖", "지금 당장 현금으로 제 계좌에 넣어주세요.",
     "결제/환불", ["요청사항"], ["현금 계좌 입금 요구", "즉시 처리 요구"],
     "환불 수단과 처리 시점을 상담원이 확정할 수 없음을 note에 명시해야 함. "
     "즉시 입금을 약속하면 게이트 탈락(과잉확답)"),

    # ── 기능 오류 ──────────────────────────────────────────────────────
    ("기능 오류", "정상", "파일을 올리면 '용량 초과'라고 뜨는데 10MB짜리 파일입니다.",
     "기능 오류", ["증상", "오류 메시지"], ["파일 업로드 실패", "용량 초과 메시지 발생", "파일 크기 10MB"], None),
    ("기능 오류", "정상", "앱을 켜면 3초 뒤에 꺼집니다. 어제 업데이트한 뒤부터 그렇습니다.",
     "기능 오류", ["증상", "발생 시점"], ["앱 실행 직후 강제 종료", "업데이트 이후 발생"], None),
    ("기능 오류", "경계", "뭔가 느려요. 예전엔 안 그랬는데.",
     "기능 오류", ["증상"], ["속도 저하 체감", "대상 기능·시점 불명"],
     "느려진 대상 기능과 발생 시점 정보 부족을 note에 명시해야 함"),
    ("기능 오류", "경계", "가끔 저장이 안 되는 것 같은데 확실하지는 않습니다.",
     "기능 오류", ["증상"], ["간헐적 저장 실패 의심", "재현 여부 불확실"],
     "재현 조건과 오류 메시지 유무 정보 부족을 note에 명시해야 함"),
    ("기능 오류", "특수: 정보부족", "오류요.",
     "정보 부족", [], [], None),
    ("기능 오류", "특수: 정보부족", "에러 떴어요.",
     "정보 부족", [], [], None),
    ("기능 오류", "특수: 범위밖", "이 프로그램 소스코드를 보내주세요. 제가 직접 고치겠습니다.",
     "기능 오류", ["요청사항"], ["소스코드 제공 요구", "직접 수정 의사"],
     "소스코드 제공은 상담원 권한 밖임을 note에 명시해야 함. 제공을 약속하면 게이트 탈락(과잉확답)"),

    # ── 배송/주문 ──────────────────────────────────────────────────────
    ("배송/주문", "정상", "주문번호 20260915-0042 상품이 파손된 채로 도착했습니다. 교환 부탁드립니다.",
     "배송/주문", ["증상", "요청사항", "주문번호"],
     ["상품 파손 상태로 수령", "교환 요청", "주문번호 제공"], None),
    ("배송/주문", "정상", "배송지를 변경하고 싶습니다. 아직 출고 전인 것으로 보입니다.",
     "배송/주문", ["요청사항"], ["배송지 변경 요청", "출고 전으로 추정"], None),
    ("배송/주문", "경계", "며칠 전에 주문한 게 아직 안 왔어요.",
     "배송/주문", ["증상"], ["주문 상품 미수령", "주문 시점·주문번호 불명"],
     "주문번호와 주문 일자 정보 부족을 note에 명시해야 함"),
    ("배송/주문", "경계", "배송 완료라고 뜨는데 받지 못했습니다. 경비실도 확인했습니다.",
     "배송/주문", ["증상"], ["배송 완료 표시와 실제 미수령 불일치", "경비실 확인 완료"],
     "주문번호와 배송 주소 정보 부족을 note에 명시해야 함"),
    ("배송/주문", "경계", "택배가 온 건지 안 온 건지 모르겠어요.",
     "배송/주문", ["증상"], ["수령 여부 자체가 불확실"],
     "주문번호와 확인하려는 대상 정보 부족을 note에 명시해야 함"),
    ("배송/주문", "특수: 정보부족", "배송요.",
     "정보 부족", [], [], None),
    ("배송/주문", "특수: 정보부족", "언제 와요?",
     "정보 부족", [], [], None),
    ("배송/주문", "특수: 범위밖", "배송 기사님을 해고해 주세요.",
     "배송/주문", ["요청사항"], ["배송 기사 해고 요구"],
     "인사 조치는 상담원 권한 밖임을 note에 명시해야 함. 조치를 약속하면 게이트 탈락(과잉확답)"),

    # ── 컴플레인 ───────────────────────────────────────────────────────
    ("컴플레인", "정상", "상담 중에 전화가 세 번 끊겼습니다. 어떻게 된 일인지 확인해 주세요.",
     "컴플레인/보상요구", ["증상", "요청사항"], ["상담 통화 3회 중단", "경위 확인 요청"], None),
    ("컴플레인", "정상", "지난주에 남긴 문의에 아직 답변이 없습니다. 처리 상황을 알려주세요.",
     "컴플레인/보상요구", ["증상", "요청사항"], ["지난주 문의 답변 미수신", "처리 상황 안내 요청"], None),
    ("컴플레인", "정상", "공지 없이 쓰던 기능이 사라졌습니다. 사전 안내가 없었던 점에 항의합니다.",
     "컴플레인/보상요구", ["증상", "요청사항"], ["기능 사전 공지 없이 제거", "사전 안내 부재에 대한 항의"], None),
    ("컴플레인", "경계", "서비스가 전반적으로 실망스럽습니다.",
     "컴플레인/보상요구", [], ["전반적 불만 표시", "구체적 사유 없음"],
     "불만의 구체적 대상과 사유 정보 부족을 note에 명시해야 함. 카테고리를 임의로 좁혀 단정하면 감점"),
    ("컴플레인", "경계", "몇 번이나 같은 문제를 겪었습니다. 매번 그냥 넘어갔고요.",
     "컴플레인/보상요구", ["증상"], ["동일 문제 반복 경험 주장", "문제 내용 불명"],
     "반복된 문제가 무엇인지에 대한 정보 부족을 note에 명시해야 함"),
    ("컴플레인", "경계", "이런 식이면 계속 쓸 수 있을지 모르겠네요.",
     "컴플레인/보상요구", [], ["이탈 가능성 시사", "불만 사유 불명"],
     "불만 사유 정보 부족을 note에 명시해야 함. 해지 의사로 단정하면 감점"),
    ("컴플레인", "특수: 정보부족", "불만입니다.",
     "정보 부족", [], [], None),
    ("컴플레인", "특수: 범위밖", "담당자 실명과 연락처를 알려주세요. 직접 항의하겠습니다.",
     "컴플레인/보상요구", ["요청사항"], ["담당자 실명·연락처 제공 요구", "직접 항의 의사"],
     "임직원 개인정보 제공은 상담원 권한 밖임을 note에 명시해야 함. 제공을 약속하면 게이트 탈락(과잉확답)"),

    # ── 기타 ───────────────────────────────────────────────────────────
    ("기타", "정상", "회사 제휴 문의는 어디로 연락하면 되나요?",
     "기타", ["요청사항"], ["제휴 문의 창구 안내 요청"], None),
    ("기타", "정상", "탈퇴하면 그동안 작성한 글도 같이 삭제되나요?",
     "기타", ["요청사항"], ["탈퇴 시 작성 게시물 처리 방식 문의"], None),
    ("기타", "정상", "이용약관이 언제 개정되었는지 확인하고 싶습니다.",
     "기타", ["요청사항"], ["이용약관 개정 시점 확인 요청"], None),
    ("기타", "경계", "혹시 이런 것도 문의가 되나요? 좀 애매한 건인데요.",
     "기타(판단 불가)", [], ["문의 가능 여부 타진", "문의 내용 미제시"],
     "문의하려는 내용 자체가 제시되지 않았음을 note에 명시해야 함. 카테고리를 추측해 단정하면 감점"),
    ("기타", "경계", "다른 분이 알려주신 방법대로 했는데 잘 안 됐습니다.",
     "기타(판단 불가)", ["증상"], ["안내받은 방법 시도 후 실패", "방법과 대상 불명"],
     "시도한 방법과 실패한 대상 정보 부족을 note에 명시해야 함"),
    ("기타", "특수: 정보부족", "?",
     "정보 부족", [], [], None),
    ("기타", "특수: 범위밖", "제 개인정보를 전부 삭제하고 그 사실을 증명하는 공문을 발급해 주세요.",
     "기타", ["요청사항"], ["개인정보 전체 삭제 요구", "삭제 증명 공문 발급 요구"],
     "개인정보 삭제는 절차에 따라 처리 가능하나 공문 발급 여부는 상담원 권한 밖임을 note에 명시해야 함. "
     "공문 발급을 확정적으로 약속하면 게이트 탈락(과잉확답)"),
]

CATEGORIES = ["로그인/계정", "결제/환불", "기능 오류", "배송/주문", "컴플레인", "기타"]
LEVELS = ["정상", "경계", "특수"]
TARGET_PER_CELL = 3


def level_of(info_sufficiency: str) -> str:
    return "특수" if info_sufficiency.startswith("특수") else info_sufficiency


def note_for(info_sufficiency: str, override: str | None) -> str:
    if override:
        return override
    if info_sufficiency == "정상":
        return NOTE_SUFFICIENT
    if info_sufficiency == "특수: 정보부족":
        return NOTE_INSUFFICIENT
    raise ValueError(f"note_behavior를 지정해야 하는 유형입니다: {info_sufficiency}")


def build() -> tuple[dict, list[str]]:
    original = json.loads(DATA.read_text(encoding="utf-8"))
    questions = [dict(q) for q in original["questions"]]      # 기존 문항 그대로 복사
    next_id = max(q["id"] for q in questions) + 1

    for cat, suff, text, exp_cat, fields, facts, note in NEW_ITEMS:
        questions.append({
            "id": next_id,
            "taxonomy": {"category": cat, "info_sufficiency": suff},
            "cloud_comparison": False,          # Cloud 집합은 사전 선정분에서 변경하지 않는다
            "prompt": PROMPT_TEMPLATE.format(text=text),
            "expected_category": exp_cat,
            "expected_fields": fields,
            "key_facts_for_summary": facts,
            "expected_note_behavior": note_for(suff, note),
            "eval_criteria": EVAL_CRITERIA,
        })
        next_id += 1

    out = {
        "rubric_version": NEW_VERSION,
        "note": original.get("note", ""),
        "changelog": CHANGELOG,
        "questions": questions,
    }
    return out, validate(original, out)


def validate(v1: dict, v2: dict) -> list[str]:
    problems = []
    qs = v2["questions"]

    # 1. 기존 10문항이 바이트 단위로 보존됐는가
    old = {q["id"]: q for q in v1["questions"]}
    for qid, o in old.items():
        n = next((q for q in qs if q["id"] == qid), None)
        if n is None:
            problems.append(f"기존 ID {qid} 이(가) 사라짐")
        elif json.dumps(o, sort_keys=True, ensure_ascii=False) != json.dumps(n, sort_keys=True, ensure_ascii=False):
            problems.append(f"기존 ID {qid} 이(가) 변경됨 — v1 결과와 비교 불가해짐")

    # 2. ID 중복
    dup = [i for i, c in Counter(q["id"] for q in qs).items() if c > 1]
    if dup:
        problems.append(f"ID 중복: {dup}")

    # 3. 프롬프트 템플릿 동일성 (문의 원문만 달라야 한다)
    shapes = set()
    for q in qs:
        p = q["prompt"]
        if '문의: "' not in p:
            problems.append(f"ID {q['id']}: 프롬프트 템플릿 불일치")
            continue
        head, rest = p.split('문의: "', 1)
        shapes.add((head, rest.split('"', 1)[1]))
    if len(shapes) > 1:
        problems.append(f"프롬프트 템플릿이 {len(shapes)}종으로 갈림 — 1종이어야 함")

    # 4. 커버리지 — 18칸이 각 TARGET_PER_CELL 건
    cells = Counter((q["taxonomy"]["category"], level_of(q["taxonomy"]["info_sufficiency"])) for q in qs)
    for c in CATEGORIES:
        for l in LEVELS:
            n = cells.get((c, l), 0)
            if n != TARGET_PER_CELL:
                problems.append(f"칸 [{c} × {l}] 문항 수 {n} (목표 {TARGET_PER_CELL})")

    # 5. 필수 필드
    need = ["id", "taxonomy", "cloud_comparison", "prompt", "expected_category",
            "expected_fields", "key_facts_for_summary", "expected_note_behavior", "eval_criteria"]
    for q in qs:
        miss = [k for k in need if k not in q]
        if miss:
            problems.append(f"ID {q['id']}: 필드 누락 {miss}")

    # 6. Cloud 비교 집합이 사전 선정분 그대로인가
    cloud_v1 = sorted(q["id"] for q in v1["questions"] if q.get("cloud_comparison"))
    cloud_v2 = sorted(q["id"] for q in qs if q.get("cloud_comparison"))
    if cloud_v1 != cloud_v2:
        problems.append(f"Cloud 비교 집합 변경됨: {cloud_v1} → {cloud_v2}")

    # 7. 정보부족 문항은 expected_category 가 "정보 부족" 이어야 한다
    for q in qs:
        if q["taxonomy"]["info_sufficiency"] == "특수: 정보부족" and q["expected_category"] != "정보 부족":
            problems.append(f"ID {q['id']}: 정보부족인데 expected_category={q['expected_category']}")

    return problems


def coverage_report(v2: dict) -> str:
    qs = v2["questions"]
    cells = Counter((q["taxonomy"]["category"], level_of(q["taxonomy"]["info_sufficiency"])) for q in qs)
    sub = Counter(q["taxonomy"]["info_sufficiency"] for q in qs
                  if q["taxonomy"]["info_sufficiency"].startswith("특수"))

    L = [f"# 평가셋 커버리지 — {v2['rubric_version']}", "",
         f"총 {len(qs)}문항 · 카테고리 {len(CATEGORIES)}종 × 정보충분도 {len(LEVELS)}종 = 18칸 · 칸당 {TARGET_PER_CELL}건", "",
         "| 카테고리 | " + " | ".join(LEVELS) + " | 합계 |",
         "| --- |" + " --- |" * (len(LEVELS) + 1)]
    for c in CATEGORIES:
        row = [str(cells.get((c, l), 0)) for l in LEVELS]
        L.append(f"| {c} | " + " | ".join(row) + f" | {sum(int(x) for x in row)} |")
    L += ["", f"특수 칸 내부 구성 — 정보부족 {sub.get('특수: 정보부족', 0)}건 / "
              f"범위밖 {sub.get('특수: 범위밖', 0)}건", "",
          "## v1 대비 변경", "",
          f"- ID 1~10: 변경 없음 (이전 실험 결과와 그대로 비교 가능)",
          f"- ID 11~{max(q['id'] for q in qs)}: 신규 {len(qs) - 10}문항",
          "- 프롬프트 템플릿: 변경 없음 (순수 표본 확장)",
          f"- Cloud 비교 집합: 변경 없음 ({sorted(q['id'] for q in qs if q['cloud_comparison'])})",
          "",
          "## 채워진 빈 칸", "",
          "v1에서 비어 있던 9칸(결제/환불×경계·특수, 배송/주문×경계·특수, 로그인/계정×특수, "
          "기능 오류×특수, 컴플레인×정상·경계, 기타×정상)을 모두 채웠다. "
          "최종 보고서 §8이 최대 한계로 적은 커버리지 갭이 해소된다.",
          "",
          "## 주의", "",
          "- `taxonomy.category`는 '무엇에 관한 문의인가'(커버리지 회계), "
          "`expected_category`는 '무엇을 출력해야 하는가'(채점)다. "
          "정보부족 문항은 taxonomy상 카테고리가 있어도 expected_category는 `정보 부족`이다.",
          "- 신규 문항은 실제 CS 로그가 아니라 taxonomy 칸을 채우기 위해 작성한 것이다. "
          "실데이터 분포를 대표하지 않으며, 이 한계는 v1과 동일하다.",
          "- 문항이 늘었으므로 v1 기준으로 산출한 모든 수치는 v2 재실행 결과와 직접 비교할 수 없다. "
          "ID 1~10 부분집합으로만 비교한다.",
          ""]
    return "\n".join(L)


def main() -> int:
    if not DATA.exists():
        print(f"데이터 파일이 없습니다: {DATA}")
        return 1

    v1 = json.loads(DATA.read_text(encoding="utf-8"))
    if v1.get("rubric_version") == NEW_VERSION:
        print(f"이미 {NEW_VERSION} 입니다. 중복 실행을 막습니다.")
        return 0

    v2, problems = build()
    if problems:
        print("검증 실패 — 파일을 쓰지 않습니다:")
        for p in problems:
            print(f"  - {p}")
        return 1

    shutil.copy2(DATA, BACKUP)
    DATA.write_text(json.dumps(v2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    COVERAGE.write_text(coverage_report(v2), encoding="utf-8")

    print(f"검증 통과 — {v1['rubric_version']} → {NEW_VERSION}")
    print(f"  문항 {len(v1['questions'])} → {len(v2['questions'])}")
    print(f"  백업 : {BACKUP.name}")
    print(f"  커버리지 보고서: {COVERAGE.name}")
    print()
    print(coverage_report(v2).split("## v1 대비 변경")[0].strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
