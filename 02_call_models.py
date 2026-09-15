import json
import datetime
import subprocess
from pathlib import Path

# ---- 설정 ----
MODELS = ["qwen2.5:7b", "llama3.2:3b"]
NUM_CTX = 4096
TEMPERATURE = 0.0
# 어느 폴더에서 실행하든 항상 프로젝트 안 work/02_calls에 저장
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "work" / "02_calls"

PROMPT_QUESTION_1 = (
    "다음 고객 문의를 분류하고, 핵심 정보를 JSON으로 추출하세요.\n"
    "문의: \"로그인이 갑자기 안 되고, 비밀번호 재설정 메일이 오지 않습니다. 급합니다.\"\n"
    "출력 형식:\n"
    "- category: 문의 유형\n"
    "- summary: 한 문장 요약\n"
    "- info: 핵심 정보 JSON\n"
    "- note: 정보 부족 또는 예외 사항이 있으면 기록"
)

# ---- 헬퍼 ----
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

WARMUP_PROMPT = "워밍업용 짧은 요청입니다. 간단히 한 문장으로 답하세요."

# ---- 실행 ----
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        print(f"[CALL] model={model}")

        stdout, stderr, code = call_ollama(model, PROMPT_QUESTION_1)

        record = {
            "experiment_id": "step3_call_01",
            "model": model,
            "question_id": 1,
            "called_at": now_iso(),
            "params": {
                "num_ctx": NUM_CTX,
                "temperature": TEMPERATURE,
                "stream": False,
            },
            "prompt": PROMPT_QUESTION_1,
            "status": "ok" if code == 0 else "error",
            "response": stdout.strip(),
            "raw": stdout.strip(),
        }

        if code != 0:
            record["error"] = {
                "message": stderr.strip() or stdout.strip(),
                "code": str(code),
            }

        filename = OUTPUT_DIR / f"call_{model.replace(':','_')}_question_01.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        print(f"[SAVE] {filename}")
        print("---- response preview ----")
        print(record["response"][:500])
        print()

    for model in MODELS:
        stdout, stderr, code = call_ollama(model, WARMUP_PROMPT)
        record = {
            "experiment_id": "step3_warmup_01",
            "model": model,
            "called_at": now_iso(),
            "params": {
                "num_ctx": NUM_CTX,
                "temperature": TEMPERATURE,
                "stream": False,
            },
            "prompt": WARMUP_PROMPT,
            "status": "ok" if code == 0 else "error",
            "response": stdout.strip(),
            "raw": stdout.strip(),
        }
        if code != 0:
            record["error"] = {
                "message": stderr.strip() or stdout.strip(),
                "code": str(code),
            }
        filename = OUTPUT_DIR / f"warmup_{model.replace(':','_')}_01.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"[WARMUP SAVE] {filename}")


if __name__ == "__main__":
    main()