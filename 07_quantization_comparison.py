"""도전 실습 A — 같은 모델의 양자화 버전 비교 (llama3.2 3B Instruct).

설계 근거는 QUANT_EXPERIMENT_DESIGN.md에 있다. 요약하면 이렇다.

독립변수는 양자화 수준 하나뿐이다. 계열·파라미터 수·질문·정답·생성 설정·반복
횟수·채점 기준을 전부 고정하고 q4_K_M / q8_0 / fp16만 바꾼다. 서로 다른 모델의
차이를 양자화 효과로 해석하지 않기 위한 전제다.

루프는 '패스 → 버전 → 질문' 3중 구조다(설계서 §4).
  - 반복 2회를 같은 자리에서 두 번 돌리지 않고, 서로 다른 시간대에 한 번씩 배치한다.
  - 패스 2는 버전 순서를 뒤집어, GPU 발열 같은 시간 드리프트가 특정 버전에만
    쏠리지 않게 한다. 버전을 한 덩어리로 몰아 돌리면 늦게 도는 버전이 불리해지고
    그 불리함이 '양자화 효과'로 오독된다.
  - 버전을 매 호출마다 바꾸면 드리프트는 사라지지만 모델 로드가 60회로 늘어
    로드 잡음이 신호를 덮고 VRAM 측정도 불가능해진다. 그래서 버전은 블록으로 묶는다.
  결과: 총 호출 60회, 모델 로드 6회, VRAM 측정 6회(버전당 2회).

VRAM 실측
  워밍업 직후 모델이 올라가 있는 상태에서 `/api/ps`를 조회해 총 점유(size)와
  GPU 점유(size_vram)를 바이트로 기록한다. 비율이 1.0 미만이면 일부가 CPU로
  내려간 것이므로, 그 버전의 속도 저하는 양자화가 아니라 오프로딩 탓일 수 있다.
  스크립트가 이 경우를 플래그로 남긴다(설계서 §8 규칙 1).

실행:  uv run python 07_quantization_comparison.py
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

# 계열·파라미터 수가 같고 양자화만 다른 세 버전.
VARIANTS = [
    {"tag": "llama3.2:3b-instruct-q4_K_M", "label": "4bit (Q4_K_M)", "quant_bits": 4},
    {"tag": "llama3.2:3b-instruct-q8_0", "label": "8bit (Q8_0)", "quant_bits": 8},
    {"tag": "llama3.2:3b-instruct-fp16", "label": "FP16 (비양자화)", "quant_bits": 16},
]

PASS_COUNT = 2  # 반복 횟수 = 패스 수 (본실험과 동일하게 질문당 2회)
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


def load_questions() -> list[dict]:
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
def _failed(msg: str) -> dict:
    return {
        "success": False, "response_text": "", "total_time_sec": 0,
        "ttft_sec": 0, "eval_count_tokens": 0, "tps": 0, "error_msg": msg,
    }


def run_ollama_query(model_name: str, prompt: str) -> dict:
    """04_local_experiment.py와 동일한 호출 경로·생성 설정·반환 형태."""
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
            # stream=False라 첫 토큰 시점을 직접 못 잰다. load+prompt_eval의 근사치다.
            "ttft_sec": round(load_duration + prompt_eval_duration, 3),
            "eval_count_tokens": eval_count,
            "tps": round(eval_count / eval_duration, 2) if eval_duration > 0 else 0,
            "error_msg": "",
        }
    except Exception as exc:
        return _failed(str(exc))


def unload(model_name: str) -> None:
    """keep_alive=0으로 즉시 언로드. 실패해도 실험은 계속한다."""
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
    """모델이 올라가 있는 상태의 실제 점유. size와 size_vram이 같으면 100% GPU."""
    try:
        models = requests.get(PS_URL, timeout=15).json().get("models", [])
    except Exception as exc:
        return {"measured": False, "error": str(exc)}

    for m in models:
        if model_name in (m.get("name"), m.get("model")):
            size, vram = m.get("size", 0), m.get("size_vram", 0)
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
        return {"disk_error": str(exc)}
    for m in models:
        if model_name in (m.get("name"), m.get("model")):
            details = m.get("details", {}) or {}
            return {
                "disk_size_bytes": m.get("size", 0),
                "disk_size_gb": round(m.get("size", 0) / 1e9, 2),
                "digest": (m.get("digest") or "")[:12],
                "parameter_size": details.get("parameter_size", ""),
                "quantization_level": details.get("quantization_level", ""),
            }
    return {"disk_error": "모델이 /api/tags 목록에 없음"}


def preflight(tags: list[str]) -> bool:
    try:
        installed = {m.get("name") for m in requests.get(TAGS_URL, timeout=15).json().get("models", [])}
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
# 3. 메인 — 패스 → 버전 → 질문 3중 루프
# ==========================================
def main() -> int:
    tags = [v["tag"] for v in VARIANTS]
    if not preflight(tags):
        return 1

    questions = load_questions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = PASS_COUNT * len(VARIANTS) * len(questions)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 양자화 비교 실험 시작")
    print(f"-> 패스 {PASS_COUNT} × 버전 {len(VARIANTS)} × 질문 {len(questions)} = {total}회")
    print(f"-> 패스 2는 버전 순서를 뒤집어 시간 드리프트를 상쇄한다\n")

    results: list[dict] = []
    resources: list[dict] = []
    done = 0

    # ── 바깥 루프: 반복 패스 (반복을 시간축에 흩뿌린다) ──────────────────
    for pass_idx in range(1, PASS_COUNT + 1):
        order = VARIANTS if pass_idx % 2 == 1 else list(reversed(VARIANTS))
        print("#" * 64)
        print(f"# 패스 {pass_idx}/{PASS_COUNT}  순서: {' → '.join(v['label'] for v in order)}")
        print("#" * 64)

        # ── 중간 루프: 양자화 버전 (블록 단위, 로드 비용 최소화) ──────────
        for variant in order:
            tag, label = variant["tag"], variant["label"]
            print(f"\n--- {label} | {tag} (패스 {pass_idx}) ---")

            # 다른 버전을 내려 VRAM을 비운다 → 측정·콜드스타트 조건 통일
            for other in tags:
                if other != tag:
                    unload(other)

            # 워밍업: 본실험과 분리 저장, 집계 제외
            print("워밍업... ", end="", flush=True)
            warm = run_ollama_query(tag, WARMUP_PROMPT)
            print("완료" if warm["success"] else f"실패 ({warm['error_msg']})")
            warm_path = OUTPUT_DIR / f"warmup_{safe_name(tag)}_pass{pass_idx}.json"
            with open(warm_path, "w", encoding="utf-8") as f:
                json.dump({"model": tag, "pass_index": pass_idx, "prompt": WARMUP_PROMPT,
                           "called_at": datetime.now().isoformat(), **warm},
                          f, ensure_ascii=False, indent=2)

            # ★ 측정 지점 1 — 로드 직후 자원
            footprint = read_loaded_footprint(tag)
            disk = read_disk_size(tag)
            resources.append({
                "pass_index": pass_idx, "tag": tag, "label": label,
                "quant_bits": variant["quant_bits"],
                "measured_at": datetime.now().isoformat(), **disk, **footprint,
            })

            if footprint.get("measured"):
                pct = footprint["gpu_offload_ratio"] * 100
                print(f"디스크 {disk.get('disk_size_gb', '?')}GB | 로드 {footprint['loaded_size_gb']}GB "
                      f"| VRAM {footprint['vram_gb']}GB | GPU 오프로딩 {pct:.1f}%")
                if not footprint["fully_on_gpu"]:
                    print("  주의: 일부가 CPU로 내려갔습니다. 이 버전의 속도 저하는 "
                          "양자화가 아니라 오프로딩 영향일 수 있습니다(설계서 §8 규칙 1).")
            else:
                print(f"자원 측정 실패: {footprint.get('error')}")

            # ── 안쪽 루프: 질문 10개 (모델이 올라간 상태로 연속 호출) ──────
            for q in questions:
                done += 1
                print(f"[{done}/{total}] P{pass_idx} {label} | Q{q['question_id']}... ",
                      end="", flush=True)
                res = run_ollama_query(tag, q["prompt"])
                print(f"성공 ({res['total_time_sec']}s, {res['tps']} TPS)"
                      if res["success"] else f"실패 ({res['error_msg']})")

                results.append({
                    "model_name": tag,
                    "variant_label": label,
                    "quant_bits": variant["quant_bits"],
                    "question_id": q["question_id"],
                    "category": q["type"],
                    "run_index": pass_idx,          # 불변식: run_index == 패스 번호
                    "pass_order": order.index(variant) + 1,  # 이 패스에서 몇 번째로 돌았나
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

    print("\n" + "=" * 64)
    print(f"실험 완료  |  결과 {csv_path.name}, {json_path.name}  |  자원 {res_path.name}")
    print("=" * 64)

    ok = df[df["is_success"]]

    # 패스별로 먼저 본다 — 두 패스 차이가 크면 드리프트가 남아 있다는 신호다.
    print("\n[패스별 평균 응답시간(초)] — 두 패스 차이가 버전 간 차이보다 크면 드리프트 의심")
    print(ok.pivot_table(index="variant_label", columns="run_index",
                         values="total_time_sec", aggfunc="mean").round(2).to_string())

    print("\n[양자화 버전별 성능 (두 패스 합산)]")
    perf = ok.groupby(["quant_bits", "variant_label"]).agg(
        runs=("is_success", "count"),
        avg_time_sec=("total_time_sec", "mean"),
        p95_time_sec=("total_time_sec", lambda s: s.quantile(0.95)),
        avg_ttft_sec=("ttft_sec", "mean"),
        avg_tps=("tps", "mean"),
        avg_tokens=("generated_tokens", "mean"),
    ).round(2).reset_index()
    print(perf.to_string(index=False))

    fails = len(df) - len(ok)
    if fails:
        print(f"\n실패 {fails}건 — 속도 집계에서 제외됨")

    res_df = pd.DataFrame(resources)
    cols = [c for c in ["pass_index", "quant_bits", "quantization_level", "disk_size_gb",
                        "loaded_size_gb", "vram_gb", "gpu_offload_ratio"] if c in res_df.columns]
    print("\n[양자화 버전별 자원 실측]")
    print(res_df[cols].to_string(index=False) if cols else "(자원 측정 없음)")

    if "fully_on_gpu" in res_df.columns and not res_df["fully_on_gpu"].fillna(False).all():
        partial = sorted(set(res_df.loc[~res_df["fully_on_gpu"].fillna(False), "label"]))
        print(f"\n경고: {', '.join(partial)} 는 GPU에 전부 올라가지 않았습니다.")
        print("해당 버전의 속도는 양자화 효과가 아니라 오프로딩 결과로 해석해야 합니다.")

    print("\n다음 단계 — 같은 채점 파이프라인으로 품질 비교:")
    print(f"  uv run python 05_quality_scoring.py {json_path.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
