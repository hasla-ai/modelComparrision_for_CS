import json
import datetime
import subprocess
import sys
from pathlib import Path

# ---------------- 설정 ----------------
MODELS = ["qwen2.5:7b", "llama3.2:3b"]
NUM_CTX = 4096
TEMPERATURE = 0.0
# 어느 폴더에서 실행하든 항상 프로젝트 안 work/02_calls에 저장
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "work" / "02_calls"

# ---------------- 평가 질문 로드 (data/02_prompts_and_eval.json이 단일 출처) ----------------
QUESTIONS_FILE = BASE_DIR / "data" / "02_prompts_and_eval.json"

# JSON의 info_sufficiency 값을 기존 스크립트가 쓰는 한글 유형 라벨로 매핑
_TYPE_LABELS = {
    "정상": "정상 사례",
    "경계": "경계 사례",
    "특수: 정보부족": "정보 부족 사례",
    "특수: 범위밖": "범위 밖 사례",
}

def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for q in data["questions"]:
        info_sufficiency = q["taxonomy"]["info_sufficiency"]
        questions.append({
            "question_id": q["id"],
            "type": _TYPE_LABELS.get(info_sufficiency, info_sufficiency),
            "prompt": q["prompt"],
            "eval_criteria": q.get("eval_criteria", ""),
            "cloud_comparison": q.get("cloud_comparison", False),
        })
    return questions

QUESTIONS = load_questions()

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
            preview = record["response"][:600]
            print(preview.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))
            print()

    print("\n===== DONE =====")


if __name__ == "__main__":
    main()