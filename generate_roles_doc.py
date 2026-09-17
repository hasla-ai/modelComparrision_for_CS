"""roles_manifest.json을 읽어 ROLES_AND_DELIVERABLES.md를 생성한다.

역할·책임·산출물의 단일 출처는 roles_manifest.json이다. 이 스크립트는
(1) 매니페스트가 주장하는 산출물이 실제 저장소에 존재하는지 검증하고,
(2) 발제문 체크리스트 18문항이 빠짐없이 한 번씩 배정됐는지 확인한 뒤,
(3) 사람이 읽을 마크다운 문서를 생성한다.

실행:  uv run python generate_roles_doc.py
출력:  ROLES_AND_DELIVERABLES.md
종료코드: 0 = 정상, 1 = 검증 실패(누락 산출물 또는 체크리스트 불일치)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MANIFEST = BASE_DIR / "roles_manifest.json"
OUTPUT = BASE_DIR / "ROLES_AND_DELIVERABLES.md"

OK, MISSING, PRIVATE = "있음", "없음", "로컬 보관"


def check(path_str: str, is_public: bool) -> str:
    """산출물의 존재 상태를 사람이 읽을 수 있는 문자열로 돌려준다."""
    exists = (BASE_DIR / path_str).exists()
    if not is_public:
        return f"{PRIVATE}{' · 확인됨' if exists else ' · 이 PC에 없음'}"
    return OK if exists else MISSING


def build(manifest: dict) -> tuple[str, list[str]]:
    problems: list[str] = []
    items = manifest["checklist"]["items"]
    roles = manifest["roles"]

    # --- 체크리스트 배정 검증 -------------------------------------------------
    assigned: list[int] = []
    for role in roles:
        assigned.extend(role["checklist_items"])
    expected = set(range(1, len(items) + 1))
    if sorted(assigned) != sorted(expected):
        missing = sorted(expected - set(assigned))
        dupes = sorted({n for n in assigned if assigned.count(n) > 1})
        if missing:
            problems.append(f"체크리스트 미배정 항목: {missing}")
        if dupes:
            problems.append(f"체크리스트 중복 배정 항목: {dupes}")

    L: list[str] = []
    add = L.append

    add(f"# 역할 분담과 산출물 — {manifest['project']}")
    add("")
    add(
        f"팀 {manifest['team']} · 수행 {manifest['owner']} · 기간 {manifest['period']} · "
        f"저장소 [{manifest['repository'].split('/')[-1]}]({manifest['repository']})"
    )
    add("")
    add(f"> {manifest['staffing_note']}")
    add("")
    add("---")
    add("")

    # --- 요약 표 -------------------------------------------------------------
    add("## 한눈에 보기")
    add("")
    add("| # | 역할 | 담당 영역 | 맡은 체크리스트 | 산출물 |")
    add("| --- | --- | --- | --- | --- |")
    for r in roles:
        nums = ", ".join(str(n) for n in r["checklist_items"])
        pub = sum(1 for d in r["deliverables"] if d["public"])
        priv = len(r["deliverables"]) - pub
        count = f"공개 {pub}건" + (f" · 비공개 {priv}건" if priv else "")
        add(f"| {r['no']} | **{r['title_en']}**<br>{r['title_ko']} | {r['mission']} | {nums} | {count} |")
    add("")
    add(
        "`비공개`는 방법론 자산(게이트 설계·가중치·루브릭 내부 정의)이라 저장소에 올리지 않고 "
        "로컬에만 보관하는 문서다. 결과와 판단 근거는 모두 공개 문서에 담겨 있다."
    )
    add("")
    add("---")
    add("")

    # --- 역할별 상세 ---------------------------------------------------------
    for r in roles:
        add(f"## {r['no']}. {r['title_en']} — {r['title_ko']}")
        add("")
        add(f"**맡은 일** · {r['mission']}")
        add("")
        add("**책임 범위**")
        add("")
        for x in r["responsibilities"]:
            add(f"- {x}")
        add("")
        add("**이 역할에서 내린 판단**")
        add("")
        for x in r["key_decisions"]:
            add(f"- {x}")
        add("")
        add("**산출물**")
        add("")
        add("| 파일 | 내용 | 상태 |")
        add("| --- | --- | --- |")
        for d in r["deliverables"]:
            status = check(d["path"], d["public"])
            if status == MISSING:
                problems.append(f"[{r['title_ko']}] 산출물 없음: {d['path']}")
            add(f"| `{d['path']}` | {d['note']} | {status} |")
        add("")
        add("**대응하는 발제문 체크리스트**")
        add("")
        for n in r["checklist_items"]:
            add(f"- **{n}.** {items[str(n)]}")
        add("")
        add("---")
        add("")

    # --- 검증 결과 -----------------------------------------------------------
    add("## 자체 검증")
    add("")
    add(
        f"이 문서는 `roles_manifest.json`에서 자동 생성된다. 생성할 때마다 "
        f"체크리스트 {len(items)}문항이 빠짐없이 한 번씩 배정됐는지, 각 역할이 주장하는 산출물이 "
        f"실제로 저장소에 있는지 확인한다."
    )
    add("")
    if problems:
        add("**확인 필요**")
        add("")
        for p in problems:
            add(f"- {p}")
    else:
        pub_total = sum(1 for r in roles for d in r["deliverables"] if d["public"])
        add(
            f"- 체크리스트 {len(items)}문항 전부 배정 완료, 중복 없음"
        )
        add(f"- 공개 산출물 {pub_total}건 전부 저장소에 존재")
    add("")
    add(f"*{manifest['generated_from']}*")
    add("")

    return "\n".join(L), problems


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    doc, problems = build(manifest)
    OUTPUT.write_text(doc, encoding="utf-8")
    print(f"생성: {OUTPUT.name} ({len(doc):,}자)")
    if problems:
        print("검증 실패:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("검증 통과: 체크리스트 배정 완전, 공개 산출물 전부 존재")
    return 0


if __name__ == "__main__":
    sys.exit(main())
