import json
import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# ==========================================
# 1. 설정값 정의
# ==========================================
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# 어느 폴더에서 실행하든 항상 프로젝트 안 work/04_experiment에 저장
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "work" / "04_experiment"

# 같은 루트에 있는 run_evaluation.py에서 QUESTIONS를 가져옴
sys.path.insert(0, str(BASE_DIR))
from run_evaluation import QUESTIONS

# 비교 대상 로컬 모델 2종 지정
TARGET_MODELS = [
    "llama3.2:3b",
    "qwen2.5:7b",
]

REPEAT_COUNT = 2  # 질문당 반복 횟수

# ==========================================
# 2. Ollama 호출 및 성능 측정 함수
# ==========================================
def run_ollama_query(model_name: str, prompt: str):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    start_time = time.time()
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        end_time = time.time()

        if response.status_code == 200:
            res_json = response.json()

            total_duration_sec = (end_time - start_time)
            load_duration = res_json.get("load_duration", 0) / 1e9
            prompt_eval_duration = res_json.get("prompt_eval_duration", 0) / 1e9
            eval_duration = res_json.get("eval_duration", 0) / 1e9

            eval_count = res_json.get("eval_count", 0)

            ttft = load_duration + prompt_eval_duration
            tps = eval_count / eval_duration if eval_duration > 0 else 0

            return {
                "success": True,
                "response_text": res_json.get("response", "").strip(),
                "total_time_sec": round(total_duration_sec, 3),
                "ttft_sec": round(ttft, 3),
                "eval_count_tokens": eval_count,
                "tps": round(tps, 2),
                "error_msg": ""
            }
        else:
            return {
                "success": False,
                "response_text": "",
                "total_time_sec": 0,
                "ttft_sec": 0,
                "eval_count_tokens": 0,
                "tps": 0,
                "error_msg": f"HTTP Error {response.status_code}"
            }

    except Exception as e:
        return {
            "success": False,
            "response_text": "",
            "total_time_sec": 0,
            "ttft_sec": 0,
            "eval_count_tokens": 0,
            "tps": 0,
            "error_msg": str(e)
        }

# ==========================================
# 3. 메인 실험 진행 함수
# ==========================================
def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 로컬 모델 비교 실험 시작")

    questions = QUESTIONS
    print(f"-> 질문 목록 사용 완료 (총 {len(questions)}개 질문)")

    results = []
    total_experiments = len(TARGET_MODELS) * len(questions) * REPEAT_COUNT
    current_count = 0

    for model in TARGET_MODELS:
        print(f"\n==========================================")
        print(f" Target Model: {model}")
        print(f"==========================================")

        for q in questions:
            q_id = q["question_id"]
            category = q["type"]
            prompt = q["prompt"]
            criteria = q.get("eval_criteria", "")

            for run_idx in range(1, REPEAT_COUNT + 1):
                current_count += 1
                print(f"[{current_count}/{total_experiments}] Model: {model} | {q_id} (Run {run_idx}/{REPEAT_COUNT})... ", end="", flush=True)

                res = run_ollama_query(model, prompt)

                if res["success"]:
                    print(f"성공! (소요시간: {res['total_time_sec']}s, TPS: {res['tps']})")
                else:
                    print(f"실패! (사유: {res['error_msg']})")

                results.append({
                    "model_name": model,
                    "question_id": q_id,
                    "category": category,
                    "run_index": run_idx,
                    "prompt": prompt,
                    "eval_criteria": criteria,
                    "response_text": res["response_text"],
                    "total_time_sec": res["total_time_sec"],
                    "ttft_sec": res["ttft_sec"],
                    "generated_tokens": res["eval_count_tokens"],
                    "tps": res["tps"],
                    "is_success": res["success"],
                    "error_message": res["error_msg"],
                    "manual_score": None
                })

    # ==========================================
    # 4. 결과 저장 및 데이터프레임 변환
    # ==========================================
    df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_filename = OUTPUT_DIR / "04_local_experiment_results.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")

    json_filename = OUTPUT_DIR / "04_local_experiment_results.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n==========================================")
    print(f"실험 완료!")
    print(f"- 저장된 파일: {csv_filename}, {json_filename}")
    print(f"==========================================")

    summary = df.groupby("model_name").agg(
        total_runs=("is_success", "count"),
        success_count=("is_success", "sum"),
        avg_time_sec=("total_time_sec", "mean"),
        avg_ttft_sec=("ttft_sec", "mean"),
        avg_tps=("tps", "mean")
    ).reset_index()

    print("\n[모델별 지표 요약]")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()