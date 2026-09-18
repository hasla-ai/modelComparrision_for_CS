"""도전 실습 A — 같은 모델의 양자화 버전 비교 (llama3.2:3b).

같은 모델(llama3.2 3B Instruct)의 양자화 수준만 바꿔가며 VRAM 사용량·속도·품질
변화를 관측한다. 서로 다른 모델의 차이를 양자화 효과로 해석하지 않기 위해,
비교 대상은 계열·파라미터 수가 동일하고 양자화만 다른 세 버전으로 한정한다.

동일 조건 통제
  - 질문/정답: data/02_prompts_and_eval.json (본실험과 같은 단일 출처)
  - 호출 경로·생성 설정: 04_local_experiment.py와 동일 (temperature=0.1, stream=False)
  - 반복: 질문당 2회 (본실험과 동일)
  - 버전마다 워밍업 1회를 본실험과 분리해 실행·저장하고 집계에서 제외
  - 버전 전환 시 이전 모델을 언로드해 VRAM 측정과 콜드스타트 조건을 맞춤

VRAM 실측
  워밍업 직후 모델이 메모리에 올라가 있는 상태에서 Ollama `/api/ps`를 조회해
  size(총 점유)와 size_vram(그중 GPU 점유)을 바이트 단위로 기록한다. 둘의 비율이
  100% 미만이면 일부가 CPU로 내려간 것이므로, 그 버전의 속도 저하는 양자화가 아니라
  오프로딩 때문일 수 있다 — 이 경우 결과에 플래그를 남긴다.

실행:  uv run python 07_quantization_comparison.py
출력:  work/07_quantization/07_quantization_results.csv | .json
       work/07_quantization/07_quantization_resources.json
       work/07_quantization/warmup_<변형>.json
채점:  uv run python 05_quality_scoring.py work/07_quantization/07_quantization_results.json
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ==========================================
# 1. 설정값 정의
# ==========================================
OLLAMA_BASE = "http://localhost:11434"
GENERATE_URL = f"{OLLAMA_BASE}/api/generate"
PS_URL = f"{OLLAMA_BASE}/api/ps"
TAGS_URL = f"{OLLAMA_BASE}/api/tags"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "work" / "07_quantization"
QUESTIONS_FILE = BASE_DIR / "data" / "02_prompts_and_eval.json"

# 같은 모델(llama3.2 3B Instruct)의 양자화 버전만 비교한다.
VARIANTS = [
    {"tag": "llama3.2:3b-instruct-q4_K_M", "label": "4bit (Q4_K_M)", "quant_bits": 4},
    {"tag": "llama3.2:3b-instruct-q8_0", "label": "8bit (Q8_0)", "quant_bits": 8},
    {"tag": "llama3.2:3b-instruct-fp16", "label": "FP16 (비양자화)", "quant_bits": 16},
]

REPEAT_COUNT = 2
WARMUP_PROMPT = "워밍업용 짧은 요청입니다. 간단히 한 문장으로 답하세요."
REQUEST_TIMEOUT_SEC = 180

_TYPE_LABELS = {
    "정상": "정상 사례",
    "경계": "경계 사례",
    "특수: 정보부족": "정보 부족 사례",
    "특수: 범위밖": "범위 밖 사례",
}


def safe_name(tag: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "_", tag)


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "question_id": q["id"],
            "type": _TYPE_LABELS.get(
                q["taxonomy"]["info_sufficiency"], q["taxonomy"]["info_sufficiency"]
            ),
            "prompt": q["prompt"],
            "eval_criteria": q.get("eval_criteria", ""),
        }
        for q in data["questions"]
    ]


# ==========================================
# 2. Ollama 호출 및 자원 측정
# ==========================================
def run_ollama_query(model_name: str, prompt: str) -> dict:
    """04_local_experiment.py와 동일한 호출 경로·설정·반환 형태를 사용한다."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    start = time.time()
    try:
        response = requests.post(GENERATE_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        end = time.time()
        if response.status_code != 200:
            return _failed(f"HTTP Error {response.status_code}")

        res = response.json()
        load_duration = res.get("load_duration", 0) / 1e9
        prompt_eval_duration = res.get("prompt_eval_duration", 0) / 1e9
        eval_duration = res.get("eval_duration", 0) / 1e9
        eval_count = res.get("eval_count", 0)

        return {
            "success": True,
            "response_text": res.get("response", "").strip(),
            "total_time_sec": round(end - start, 3),
            "ttft_sec": round(load_duration + prompt_eval_duration, 3),
            "eval_count_tokens": eval_count,
            "tps": round(eval_count / eval_duration, 2) if eval_duration > 0 else 0,
            "error_msg": "",
        }
    except Exception as exc:  # 네트워크/타임아웃 등
        return _failed(str(exc))


def _failed(msg: str) -> dict:
    return {
        "success": False,
        "response_text": "",
        "total_time_sec": 0,
        "ttft_sec": 0,
        "eval_count_tokens": 0,
        "tps": 0,
        "error_msg": msg,
    }


def unload(model_name: str) -> None:
    """keep_alive=0으로 즉시 언로드한다. 실패해도 실험은 계속 진행한다."""
    try:
        requests.post(
            GENERATE_URL,
            json={"model": model_name, "prompt": "", "keep_alive": 0},
            timeout=30,
        )
    except Exception:
        pass
    time.sleep(1.0)


def read_loaded_footprint(model_name: str) -> dict:
    """모델이 올라가 있는 상태에서 실제 점유 메모리를 읽는다.

    size       : 총 점유(바이트)
    size_vram  : 그중 GPU VRAM 점유(바이트)
    두 값이 같으면 100% GPU 오프로딩이다.
    """
    try:
        models = requests.get(PS_URL, timeout=15).json().get("models", [])
    except Exception as exc:
        return {"measured": False, "error": str(exc)}

    for m in models:
        if m.get("name") == model_name or m.get("model") == model_name:
            size = m.get("size", 0)
            vram = m.get("size_vram", 0)
            ratio = (vram / size) if size else 0.0
            return {
                "measured": True,
                "loaded_size_bytes": size,
                "loaded_size_gb": round(size / 1e9, 2),
                "vram_bytes": vram,
                "vram_gb": round(vram / 1e9, 2),
                "gpu_offload_ratio": round(ratio, 4),
                "fully_on_gpu": abs(ratio - 1.0) < 1e-6,
            }
    return {"measured": False, "error": "모델이 /api/ps 목록에 없음(이미 언로드되었을 수 있음)"}


def read_disk_size(model_name: str) -> dict:
    try:
        models = requests.get(TAGS_URL, timeout=15).json().get("models", [])
    except Exception as exc:
        return {"error": str(exc)}
    for m in models:
        if m.get("name") == model_name or m.get("model") == model_name:
            details = m.get("details", {}) or {}
            return {
                "disk_size_bytes": m.get("size", 0),
                "disk_size_gb": round(m.get("size", 0) / 1e9, 2),
                "digest": (m.get("digest") or "")[:12],
                "parameter_size": details.get("parameter_size", ""),
                "quantization_level": details.get("quantization_level", ""),
            }
    return {"error": "모델이 /api/tags 목록에 없음"}


def preflight(tags: list[str]) -> bool:
    """모델이 모두 받아져 있는지 확인하고, 없으면 pull 명령을 안내한다."""
    try:
        installed = {
            m.get("name") for m in requests.get(TAGS_URL, timeout=15).json().get("models", [])
        }
    except Exception as exc:
        print(f"Ollama에 연결할 수 없습니다: {exc}")
        print("Ollama 앱이 실행 중인지 확인하세요.")
        return False

    missing = [t for t in tags if t not in installed]
    if missing:
        print("아래 모델을 먼저 받아야 합니다. 다운로드 용량이 큽니다.\n")
        for t in missing:
            print(f"  ollama pull {t}")
        print("\n받은 뒤 이 스크립트를 다시 실행하세요.")
        return False
    return True


# ==========================================
# 3. 메인
# ==========================================
def main() -> int:
    tags = [v["tag"] for v in VARIANTS]
    if not preflight(tags):
        return 1

    questions = load_questions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 양자화 비교 실험 시작")
    print(f"-> 버전 {len(VARIANTS)}개 × 질문 {len(questions)}개 × {REPEAT_COUNT}회 "
          f"= {len(VARIANTS) * len(questions) * REPEAT_COUNT}회\n")

    results: list[dict] = []
    resources: list[dict] = []
    total = len(VARIANTS) * len(questions) * REPEAT_COUNT
    done = 0

    for variant in VARIANTS:
        tag, label = variant["tag"], variant["label"]
        print("=" * 60)
        print(f" {label}  —  {tag}")
        print("=" * 60)

        # 이전 버전을 내리고 시작해 콜드스타트 조건과 VRAM 측정을 맞춘다.
        for other in tags:
            if other != tag:
                unload(other)

        # --- 워밍업 (본실험과 분리, 집계 제외) ---
        print("워밍업... ", end="", flush=True)
        warm = run_ollama_query(tag, WARMUP_PROMPT)
        print("완료" if warm["success"] else f"실패 ({warm['error_msg']})")

        with open(OUTPUT_DIR / f"warmup_{safe_name(tag)}.json", "w", encoding="utf-8") as f:
            json.dump(
                {"model": tag, "prompt": WARMUP_PROMPT, "called_at": datetime.now().isoformat(), **warm},
                f, ensure_ascii=False, indent=2,
            )

        # --- 자원 실측 (모델이 올라가 있는 상태에서) ---
        footprint = read_loaded_footprint(tag)
        disk = read_disk_size(tag)
        snapshot = {
            "tag": tag, "label": label, "quant_bits": variant["quant_bits"],
            "measured_at": datetime.now().isoformat(), **disk, **footprint,
        }
        resources.append(snapshot)

        if footprint.get("measured"):
            pct = footprint["gpu_offload_ratio"] * 100
            print(f"디스크 {disk.get('disk_size_gb', '?')}GB | "
                  f"로드 {footprint['loaded_size_gb']}GB | "
                  f"VRAM {footprint['vram_gb']}GB | GPU 오프로딩 {pct:.1f}%")
            if not footprint["fully_on_gpu"]:
                print("  주의: 일부가 CPU로 내려갔습니다. 이 버전의 속도 저하는 "
                      "양자화가 아니라 오프로딩 영향일 수 있습니다.")
        else:
            print(f"자원 측정 실패: {footprint.get('error')}")

        # --- 본실험 ---
        for q in questions:
            for run_idx in range(1, REPEAT_COUNT + 1):
                done += 1
                print(f"[{done}/{total}] {label} | Q{q['question_id']} "
                      f"(Run {run_idx}/{REPEAT_COUNT})... ", end="", flush=True)
                res = run_ollama_query(tag, q["prompt"])
                print(f"성공 ({res['total_time_sec']}s, {res['tps']} TPS)"
                      if res["success"] else f"실패 ({res['error_msg']})")

                results.append({
                    "model_name": tag,
                    "variant_label": label,
                    "quant_bits": variant["quant_bits"],
                    "question_id": q["question_id"],
                    "category": q["type"],
                    "run_index": run_idx,
                    "prompt": q["prompt"],
                    "eval_criteria": q["eval_criteria"],
                    "response_text": res["response_text"],
                    "total_time_sec": res["total_time_sec"],
                    "ttft_sec": res["ttft_sec"],
                    "generated_tokens": res["eval_count_tokens"],
                    "tps": res["tps"],
                    "is_success": res["success"],
                    "error_message": res["error_msg"],
                    "manual_score": None,
                })

        unload(tag)
        print()

    # ==========================================
    # 4. 저장
    # ==========================================
    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "07_quantization_results.csv"
    json_path = OUTPUT_DIR / "07_quantization_results.json"
    res_path = OUTPUT_DIR / "07_quantization_resources.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(resources, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("실험 완료")
    print(f"- 결과: {csv_path.name}, {json_path.name}")
    print(f"- 자원: {res_path.name}")
    print("=" * 60)

    perf = df.groupby(["quant_bits", "variant_label"]).agg(
        runs=("is_success", "count"),
        success=("is_success", "sum"),
        avg_time_sec=("total_time_sec", "mean"),
        avg_ttft_sec=("ttft_sec", "mean"),
        avg_tps=("tps", "mean"),
        avg_tokens=("generated_tokens", "mean"),
    ).round(2).reset_index()

    res_df = pd.DataFrame(resources)
    cols = [c for c in ["quant_bits", "disk_size_gb", "loaded_size_gb", "vram_gb",
                        "gpu_offload_ratio", "quantization_level"] if c in res_df.columns]

    print("\n[양자화 버전별 성능]")
    print(perf.to_string(index=False))
    print("\n[양자화 버전별 자원]")
    print(res_df[cols].to_string(index=False) if cols else "(자원 측정 없음)")
    print("\n다음 단계 — 같은 채점 파이프라인으로 품질 비교:")
    print(f"  uv run python 05_quality_scoring.py {json_path.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
