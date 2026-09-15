import json
import datetime
import subprocess
from pathlib import Path

# ---------------- 설정 ----------------
MODELS = ["qwen2.5:7b", "llama3.2:3b"]
NUM_CTX = 4096
TEMPERATURE = 0.0
# 어느 폴더에서 실행하든 항상 프로젝트 안 work/02_calls에 저장
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "work" / "02_calls"

# ---------------- 공통 출력 형식 ----------------
COMMON_FORMAT = (
    "출력 형식:\n"
    "- category: 문의 유형\n"
    "- summary: 한 문장 요약\n"
    "- info: 핵심 정보 JSON\n"
    "- note: 정보 부족 또는 예외 사항이 있으면 기록"
)

# ---------------- 평가 질문 10개 ----------------
QUESTIONS = [
    {
        "question_id": 1,
        "type": "정상 사례",
        "prompt": "다음 고객 문의를 분류하고, 핵심 정보를 JSON으로 추출하세요.\n문의: \"로그인이 갑자기 안 되고, 비밀번호 재설정 메일이 오지 않습니다. 급합니다.\"\n" + COMMON_FORMAT,
    },
    {
        "question_id": 2,
        "type": "정상 사례",
        "prompt": "다음 고객 문의를 분류하고, 핵심 정보를 JSON으로 추출하세요.\n문의: \"방금 결제를 했는데 실패했다고 뜨고, 환불은 어떻게 되는지 궁금합니다.\"\n" + COMMON_FORMAT,
    },
    {
        "question_id": 3,
        "type": "정상 사례",
        "prompt": "다음 고객 문의를 분류하고, 핵심 정보를 JSON으로 추출하세요.\n문의: \"프로그램을 실행하면 '구문 분석 오류'라는 메시지만 뜨고 실행이 안 됩니다. 어떻게 해야 하나요?\"\n" + COMMON_FORMAT,
    },
    {
        "question_id": 4,
        "type": "정상 사례",
        "prompt": "다음 고객 문의를 분류하고, 핵심 정보를 JSON으로 추출하세요.\n문의: \"주문이 아직 배송되지 않았는데, 언제쯤 배송되는지 안내받고 싶습니다.\"\n" + COMMON_FORMAT,
    },
    {
        "question_id": 5,
        "type": "정상 사례",
        "prompt": "다음 고객 문의를 분류하고, 핵심 정보를 JSON으로 추출하세요.\n문의: \"계정이 잠겨서 로그인할 수 없습니다. 해제 부탁드립니다.\"\n" + COMMON_FORMAT,
    },
    {
        "question_id": 6,
        "type": "경계 사례",
        "prompt": "다음 고객 문의를 분류하고, 핵심 정보를 JSON으로 추출하세요.\n문의: \"가끔 화면이 멈춥니다. 언제부터인지 잘 모르겠고, 어떤 상황에서 그런지도 정확히 기억나지 않습니다.\"\n" + COMMON_FORMAT,
    },
    {
        "question_id": 7,
        "type": "경계 사례",
        "prompt": "다음 고객 문의를 분류하고, 핵심 정보를 JSON으로 추출하세요.\n문의: \"로그인이 안 되고, 로그인 후에 보이는 화면도 깨져서 보입니다.\"\n" + COMMON_FORMAT,
    },
    {
        "question_id": 8,
        "type": "경계 사례",
        "prompt": "다음 고객 문의를 분류하고, 핵심 정보를 JSON으로 추출하세요.\n문의: \"지금 뭔가 큰 문제가 생긴 것 같습니다. 빨리 확인해 주세요.\"\n" + COMMON_FORMAT,
    },
    {
        "question_id": 9,
        "type": "범위 밖/정보 부족 사례",
        "prompt": "다음 고객 문의를 분류하고, 핵심 정보를 JSON으로 추출하세요.\n문의: \"안 됩니다.\"\n" + COMMON_FORMAT,
    },
    {
        "question_id": 10,
        "type": "범위 밖/정보 부족 사례",
        "prompt": "다음 고객 문의를 분류하고, 핵심 정보를 JSON으로 추출하세요.\n문의: \"이 서비스 정책으로 인해 발생한 손해를 전부 보상해 주세요.\"\n" + COMMON_FORMAT,
    },
]

# ---------------- 헬퍼 ----------------
def now_iso():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()

def decode_safe(data: bytes) -> str:
    if data is None:
        return ""
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, AttributeError):
            continue
    return data.decode("utf-8", errors="replace")

def call_ollama(model, prompt):
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    stdout = decode_safe(proc.stdout)
    stderr = decode_safe(proc.stderr)
    return stdout, stderr, proc.returncode

# ---------------- 실행 ----------------
# 다른 파일에서 QUESTIONS만 가져다 쓸 때 평가가 통째로 재실행되지 않도록 main()으로 감쌈
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        print(f"\n===== MODEL: {model} =====")

        for q in QUESTIONS:
            print(f"--- question_id={q['question_id']} type={q['type']} ---")

            stdout, stderr, code = call_ollama(model, q["prompt"])

            record = {
                "experiment_id": "step4_evaluation",
                "model": model,
                "question_id": q["question_id"],
                "type": q["type"],
                "called_at": now_iso(),
                "params": {
                    "num_ctx": NUM_CTX,
                    "temperature": TEMPERATURE,
                    "stream": False,
                },
                "prompt": q["prompt"],
                "status": "ok" if code == 0 else "error",
                "response": stdout.strip(),
                "raw": stdout.strip(),
            }

            if code != 0:
                record["error"] = {
                    "message": stderr.strip() or stdout.strip(),
                    "code": str(code),
                }

            filename = OUTPUT_DIR / f"eval_{model.replace(':','_')}_question_{q['question_id']:02d}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            print(f"[SAVE] {filename}")
            print("---- response preview ----")
            print(record["response"][:600])
            print()

    print("\n===== DONE =====")


if __name__ == "__main__":
    main()