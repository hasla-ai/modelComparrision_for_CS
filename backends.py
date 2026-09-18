"""서빙 백엔드 어댑터 — Ollama / vLLM 을 같은 인터페이스로 호출한다.

왜 어댑터인가
  `04_local_experiment.py` 를 복사해 vLLM 판을 따로 만들면, 두 벌이 조용히
  어긋나고 그 차이가 결과에 섞인다. 호출부는 하나로 두고 백엔드만 바꾼다.

무엇을 맞추는가 (VLLM_SERVING_DESIGN.md §2·§8 C1)
  1. 보내는 프롬프트 문자열이 바이트 단위로 같아야 한다
     → 전송 직전 SHA-256 을 레코드에 남긴다. 실행 후 구성별로 대조한다.
  2. 서버가 프롬프트를 감싸는 방식이 같아야 한다
     → 기존 실험은 Ollama `/api/generate` 를 `raw` 없이 썼다. 즉 **Ollama 가
       모델의 chat template 을 적용**한다. 그러므로 대응되는 vLLM 엔드포인트는
       `/v1/completions`(원문 그대로)가 아니라 **`/v1/chat/completions`** 다.
       설계서 초안에는 `/v1/completions` 로 적었으나, 기존 호출 규약을 확인한 뒤
       이쪽으로 고쳤다. 잘못 고르면 모델이 보는 텍스트 자체가 달라진다.
     → 그래도 Ollama 의 GGUF 내장 템플릿과 vLLM 의 HF tokenizer 템플릿이
       같다는 보장은 없다. 그래서 서버가 센 프롬프트 토큰 수를 함께 기록하고
       `--check-prompts` 로 대조한다.
  3. 샘플러가 같아야 한다
     → §3 통제. `SAMPLER_PROFILES` 참조.

TTFT 를 어떻게 재는가
  Ollama 비스트리밍 응답에는 `load_duration` + `prompt_eval_duration` 이 있어
  서버가 직접 알려주지만, vLLM 에는 같은 필드가 없다. 두 엔진을 같은 자로 재려면
  **첫 토큰이 도착한 시각을 클라이언트에서 재는 수밖에 없다.** 그래서 항상
  스트리밍으로 호출하고 `ttft_sec` 은 벽시계 기준으로 통일한다.
  Ollama 는 서버 값도 같이 오므로 `ttft_server_sec` 에 따로 남겨 둔다
  (기존 40회 실험의 `ttft_sec` 은 이 서버 값이었다 — 같은 열에 섞지 말 것).

TPS 도 같은 문제가 있다
  Ollama 의 TPS 는 서버가 알려준 `eval_duration`(순수 생성 구간)으로 나눈 값이고,
  vLLM 에는 그런 필드가 없어 클라이언트 벽시계로 나눌 수밖에 없다. **두 값은
  같은 자가 아니므로 나란히 놓고 비교하면 안 된다.** 그래서 엔진 고유값은
  `tps` 에 그대로 두되, 두 엔진에 같은 식을 적용한 `tps_wall` 을 따로 만든다.

      tps_wall = 생성 토큰 / (총 소요 시간 - TTFT)

  엔진 간 비교는 `tps_wall` 로만 한다.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import requests

# ──────────────────────────────────────────────────────────────────────
# 샘플러 프로파일
# ──────────────────────────────────────────────────────────────────────
# "fixed"    : 세 구성에 같은 값을 명시적으로 지정한다. 구성 간 비교용.
# "baseline" : 04_local_experiment.py 와 같다(= temperature 만 지정하고 나머지는
#              Ollama 기본값). 기존 185회를 재현할 때만 쓴다.
#
# 왜 나누는가: 기존 실험은 temperature 만 지정하고 top_p·top_k·repeat_penalty·
# num_predict·seed 를 Ollama 기본값에 맡겼다. vLLM 기본값은 그것과 다르다.
# 그대로 두면 "엔진 차이"에 "샘플러 차이"가 섞인다. 그렇다고 fixed 로 맞추면
# 구성 A 가 더 이상 185회 기준선이 아니다. 둘 다 필요하므로 프로파일로 나누고
# **모든 레코드에 어느 프로파일이었는지 남긴다.**
SAMPLER_PROFILES = {
    "fixed": {
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "max_tokens": 512,
        "seed": 20260918,
    },
    "baseline": {
        "temperature": 0.1,
    },
}

DEFAULT_TIMEOUT = 180


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass
class CallResult:
    """백엔드가 돌려주는 단일 호출 결과. 모든 백엔드가 같은 모양을 채운다."""
    success: bool = False
    response_text: str = ""
    total_time_sec: float = 0.0
    ttft_sec: float = 0.0              # 벽시계 — 첫 토큰 도착까지. 엔진 간 비교는 이 값으로
    ttft_server_sec: float | None = None   # Ollama 만: load+prompt_eval (기존 40회의 그 값)
    prompt_tokens: int | None = None   # 서버가 센 입력 토큰 수 — 템플릿 대조용
    generated_tokens: int = 0
    tps: float = 0.0                   # 엔진 고유 계산 — 엔진 간 비교 금지
    tps_wall: float = 0.0              # 엔진 공통 자: 생성 토큰 / (총시간 - TTFT)
    tps_source: str = ""               # tps 가 무엇으로 계산됐는지
    error_message: str = ""
    raw_meta: dict = field(default_factory=dict)


class Backend:
    """공통 인터페이스. 구현체는 generate() 하나만 채우면 된다."""

    name = "base"
    engine = "base"

    def __init__(self, model: str, base_url: str, sampler_profile: str = "fixed"):
        if sampler_profile not in SAMPLER_PROFILES:
            raise ValueError(f"알 수 없는 샘플러 프로파일: {sampler_profile}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.sampler_profile = sampler_profile
        self.sampler = dict(SAMPLER_PROFILES[sampler_profile])
        # 부하 시험에서 매 호출마다 새 TCP 연결을 열면 연결 비용이 지연에 섞인다.
        # requests.Session 은 스레드 안전이 아니므로 **워커마다 별도 인스턴스**를
        # 만들어 쓸 것 (20_concurrency_benchmark.py 가 그렇게 한다).
        self._session = requests.Session()

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    def generate(self, prompt: str, timeout: int = DEFAULT_TIMEOUT) -> CallResult:
        raise NotImplementedError

    def health(self) -> tuple[bool, str]:
        raise NotImplementedError

    def describe_template(self) -> dict:
        """--check-prompts 용. 서버가 프롬프트를 어떻게 감싸는지 알 수 있는 만큼 모은다."""
        return {}

    def unload(self) -> None:
        """다음 구성 측정 전에 VRAM 을 비운다. 안 되면 조용히 넘어간다."""
        return None


# ──────────────────────────────────────────────────────────────────────
# Ollama
# ──────────────────────────────────────────────────────────────────────
class OllamaBackend(Backend):
    """POST /api/generate — `raw` 를 쓰지 않는다(= 서버가 chat template 적용).

    기존 04_local_experiment.py 와 같은 경로다. 여기를 바꾸면 구성 A 가
    185회 기준선이 아니게 되므로 바꾸지 않는다.
    """

    engine = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 sampler_profile: str = "fixed", name: str = "ollama"):
        super().__init__(model, base_url, sampler_profile)
        self.name = name

    def _options(self) -> dict:
        s = self.sampler
        opts = {"temperature": s["temperature"]}
        if self.sampler_profile == "fixed":
            opts.update({
                "top_p": s["top_p"],
                "top_k": s["top_k"],
                "repeat_penalty": s["repeat_penalty"],
                "num_predict": s["max_tokens"],
                "seed": s["seed"],
            })
        return opts

    def health(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            tags = [m.get("name", "") for m in r.json().get("models", [])]
            if self.model not in tags:
                return False, f"모델 미적재: {self.model} (있는 것: {', '.join(tags[:5])}…)"
            return True, "ok"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def generate(self, prompt: str, timeout: int = DEFAULT_TIMEOUT) -> CallResult:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": self._options(),
        }
        out = CallResult()
        chunks: list[str] = []
        t0 = time.perf_counter()
        ttft = None
        final: dict = {}
        try:
            with self._session.post(f"{self.base_url}/api/generate", json=payload,
                               timeout=timeout, stream=True) as resp:
                if resp.status_code != 200:
                    out.error_message = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    return out
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = obj.get("response", "")
                    if piece and ttft is None:
                        ttft = time.perf_counter() - t0
                    if piece:
                        chunks.append(piece)
                    if obj.get("done"):
                        final = obj
            total = time.perf_counter() - t0

            eval_count = final.get("eval_count", 0) or 0
            eval_dur = (final.get("eval_duration", 0) or 0) / 1e9
            load_dur = (final.get("load_duration", 0) or 0) / 1e9
            peval_dur = (final.get("prompt_eval_duration", 0) or 0) / 1e9

            out.success = True
            out.response_text = "".join(chunks).strip()
            out.total_time_sec = round(total, 3)
            out.ttft_sec = round(ttft if ttft is not None else total, 3)
            out.ttft_server_sec = round(load_dur + peval_dur, 3)
            out.prompt_tokens = final.get("prompt_eval_count")
            out.generated_tokens = eval_count
            out.tps = round(eval_count / eval_dur, 2) if eval_dur > 0 else 0.0
            out.tps_source = "server_eval_duration"
            gen_wall = max(total - out.ttft_sec, 1e-9)
            out.tps_wall = round(eval_count / gen_wall, 2) if eval_count else 0.0
            out.raw_meta = {k: final.get(k) for k in
                            ("total_duration", "load_duration", "prompt_eval_duration",
                             "prompt_eval_count", "eval_duration", "eval_count", "done_reason")}
        except Exception as exc:
            out.error_message = f"{type(exc).__name__}: {exc}"
        return out

    def describe_template(self) -> dict:
        """/api/show 로 모델에 박힌 template 을 꺼낸다."""
        info: dict = {}
        try:
            r = requests.post(f"{self.base_url}/api/show", json={"name": self.model, "model": self.model}, timeout=20)
            if r.status_code == 200:
                j = r.json()
                info["template"] = j.get("template")
                info["parameters"] = j.get("parameters")
                det = j.get("details", {})
                info["quantization"] = det.get("quantization_level")
                info["family"] = det.get("family")
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    def unload(self) -> None:
        try:
            requests.post(f"{self.base_url}/api/generate",
                          json={"model": self.model, "keep_alive": 0}, timeout=30)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────
# vLLM
# ──────────────────────────────────────────────────────────────────────
class VLLMBackend(Backend):
    """POST /v1/chat/completions — Ollama 가 템플릿을 적용하므로 이쪽을 쓴다.

    `/v1/completions`(원문 그대로 투입)를 쓰면 모델이 받는 텍스트가 Ollama 쪽과
    달라져 품질 비교가 성립하지 않는다. 자세한 이유는 모듈 docstring 2번.
    """

    engine = "vllm"

    def __init__(self, model: str, base_url: str = "http://localhost:8000",
                 sampler_profile: str = "fixed", name: str = "vllm"):
        if sampler_profile == "baseline":
            raise ValueError(
                "vLLM 에는 'baseline' 프로파일을 쓸 수 없습니다.\n"
                "  'baseline' 은 'temperature 만 지정하고 나머지는 Ollama 기본값에 맡긴다'는 뜻인데,\n"
                "  vLLM 의 기본값은 그것과 다르고 max_tokens 도 반드시 지정해야 합니다.\n"
                "  구성 간 비교에는 --sampler fixed 를 쓰고, 'baseline' 은 구성 A 단독\n"
                "  재현에만 쓰십시오."
            )
        super().__init__(model, base_url, sampler_profile)
        self.name = name

    def health(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=10)
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            ids = [m.get("id", "") for m in r.json().get("data", [])]
            if self.model not in ids:
                return False, f"모델 미제공: {self.model} (서버 제공: {', '.join(ids) or '없음'})"
            return True, "ok"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def generate(self, prompt: str, timeout: int = DEFAULT_TIMEOUT) -> CallResult:
        s = self.sampler
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": s["temperature"],
            "top_p": s["top_p"],
            "max_tokens": s["max_tokens"],
            "seed": s["seed"],
            # top_k·repetition_penalty 는 OpenAI 규격 밖이지만 vLLM 이 최상위에서
            # 받아 준다. (openai 파이썬 클라이언트의 extra_body 는 REST 필드가
            # 아니므로 여기서 쓰면 미지 필드로 거절될 수 있다 — 쓰지 않는다.)
            "top_k": s["top_k"],
            "repetition_penalty": s["repeat_penalty"],
        }
        out = CallResult()
        chunks: list[str] = []
        t0 = time.perf_counter()
        ttft = None
        usage: dict = {}
        try:
            with self._session.post(f"{self.base_url}/v1/chat/completions", json=payload,
                               timeout=timeout, stream=True) as resp:
                if resp.status_code != 200:
                    out.error_message = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    return out
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for ch in obj.get("choices", []):
                        piece = (ch.get("delta") or {}).get("content") or ""
                        if piece and ttft is None:
                            ttft = time.perf_counter() - t0
                        if piece:
                            chunks.append(piece)
            total = time.perf_counter() - t0
            gen = usage.get("completion_tokens", 0) or 0
            gen_time = max(total - (ttft or 0.0), 1e-9)

            out.success = True
            out.response_text = "".join(chunks).strip()
            out.total_time_sec = round(total, 3)
            out.ttft_sec = round(ttft if ttft is not None else total, 3)
            out.prompt_tokens = usage.get("prompt_tokens")
            out.generated_tokens = gen
            # vLLM 에는 서버가 알려주는 생성 구간이 없다. 벽시계밖에 없으므로
            # tps 와 tps_wall 이 같은 값이 되고, 그 사실을 tps_source 에 남긴다.
            out.tps = round(gen / gen_time, 2) if gen else 0.0
            out.tps_source = "client_wallclock"
            out.tps_wall = out.tps
            out.raw_meta = {"usage": usage}
        except Exception as exc:
            out.error_message = f"{type(exc).__name__}: {exc}"
        return out

    def describe_template(self) -> dict:
        """/tokenize 로 서버가 실제로 만든 프롬프트와 토큰 수를 꺼낸다.

        vLLM 버전에 따라 응답 형태가 다르므로 있는 만큼만 취한다.
        """
        info: dict = {}
        try:
            r = requests.post(
                f"{self.base_url}/tokenize",
                json={"model": self.model,
                      "messages": [{"role": "user", "content": "__PROBE__"}],
                      "add_generation_prompt": True},
                timeout=20)
            if r.status_code == 200:
                j = r.json()
                info["tokenize_count"] = j.get("count")
                info["rendered"] = j.get("prompt")      # 제공하는 버전에서만 채워진다
            else:
                info["tokenize_http"] = r.status_code
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
        return info


# ──────────────────────────────────────────────────────────────────────
# 구성 정의 — A / B / C
# ──────────────────────────────────────────────────────────────────────
# VLLM_SERVING_DESIGN.md §2.1
#   A vs B = 런타임(WSL2) 효과   (엔진·포맷 동일)
#   B vs C = 엔진 + 포맷 효과     (런타임 동일)
CONFIGS = {
    "A": {"label": "Ollama / Windows 네이티브", "engine": "ollama",
          "runtime": "windows", "weights_format": "gguf-q4_k_m",
          "base_url": "http://localhost:11434"},
    "B": {"label": "Ollama / WSL2", "engine": "ollama",
          "runtime": "wsl2", "weights_format": "gguf-q4_k_m",
          "base_url": "http://localhost:11435"},
    "C": {"label": "vLLM / WSL2", "engine": "vllm",
          "runtime": "wsl2", "weights_format": "awq-4bit",
          "base_url": "http://localhost:8000"},
}


def build_backend(config_key: str, model: str, sampler_profile: str = "fixed",
                  base_url: str | None = None) -> Backend:
    if config_key not in CONFIGS:
        raise ValueError(f"알 수 없는 구성: {config_key} (가능: {', '.join(CONFIGS)})")
    cfg = CONFIGS[config_key]
    url = base_url or cfg["base_url"]
    if cfg["engine"] == "ollama":
        return OllamaBackend(model, url, sampler_profile, name=config_key)
    return VLLMBackend(model, url, sampler_profile, name=config_key)
