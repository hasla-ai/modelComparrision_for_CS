"""측정 환경을 레코드와 함께 남긴다 — 부록 L 이 남긴 숙제.

왜 만들었나
  같은 코드·같은 모델·같은 문항의 지연 측정이 나흘 만에 3.8배 달라졌다
  (본실험 3.38초 → 재측정 0.894초). 원인을 **특정하지 못했다.** 당시 환경을
  `work/01_env/*.txt` 에 사람이 읽는 형태로 한 번 적어 둔 것이 전부였고,
  실험 레코드 자체는 자기가 어떤 환경에서 측정됐는지 말해 주지 않았기 때문이다.

  성능 수치는 환경을 고정하지 않으면 재현되지 않는다. 그러면 최소한
  **어떤 환경에서 잰 값인지 수치 옆에 붙어 있어야** 한다.

무엇을 하나
  1. 성능에 영향을 주는 환경을 한 번에 긁는다 (Ollama 버전, GPU·드라이버,
     CUDA, Python, 주요 패키지).
  2. 그중 **지연·처리량에 직접 영향을 주는 것만** 골라 짧은 지문(fingerprint)을
     만든다. 지문이 다르면 두 측정의 절대값을 같은 표에 놓아서는 안 된다.
  3. 저장소 안의 모든 실행 기록을 훑어 어느 실행끼리 비교 가능한지 알려 준다.

쓰는 법
    from env_snapshot import snapshot, fingerprint, row_fields
    snap = snapshot()
    ...
    row.update(row_fields(snap))        # 레코드 한 줄에 최소 정보
    save(snap, OUT_DIR / "20_env.json") # 실행 폴더에 전체 기록

    uv run python env_snapshot.py            # 지금 환경 출력
    uv run python env_snapshot.py --list     # 기록된 실행들의 환경 대조
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "work"
DEFAULT_OLLAMA = "http://localhost:11434"

# 버전을 함께 남길 패키지
WATCH = ["ollama", "openai", "requests", "pandas", "numpy",
         "torch", "transformers", "sentence-transformers", "datasets"]

# ── 지문에 넣을 것 ────────────────────────────────────────────────────
# 지연·처리량에 직접 영향을 주는 것만 고른다. 파이썬 패키지 버전은 넣지 않는다
# (pandas 를 올렸다고 추론이 느려지지는 않는다). 반대로 Ollama 버전과 드라이버는
# 넣는다 — 부록 L 에서 의심되는 것이 바로 그 둘이다.
FINGERPRINT_KEYS = [
    ("ollama", "version"),
    ("gpu", "name"),
    ("gpu", "driver_version"),
    ("gpu", "cuda_version"),
]


def _run(cmd: list[str], timeout: int = 10) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _ollama(url: str) -> dict:
    info: dict = {"endpoint": url, "reachable": False, "version": None}
    try:
        import requests
        r = requests.get(f"{url.rstrip('/')}/api/version", timeout=5)
        if r.status_code == 200:
            info["reachable"] = True
            info["version"] = r.json().get("version")
            return info
    except Exception:
        pass
    # HTTP 로 못 얻으면 CLI 로 한 번 더 — "ollama version is 0.34.0"
    txt = _run(["ollama", "--version"])
    if txt:
        info["version"] = txt.split()[-1]
        info["source"] = "cli"
    return info


def _gpu() -> dict:
    info: dict = {"name": None, "driver_version": None,
                  "vram_total_mib": None, "cuda_version": None}
    q = _run(["nvidia-smi",
              "--query-gpu=name,driver_version,memory.total",
              "--format=csv,noheader,nounits"])
    if q:
        parts = [p.strip() for p in q.split("\n")[0].split(",")]
        if len(parts) >= 3:
            info["name"] = parts[0]
            info["driver_version"] = parts[1]
            try:
                info["vram_total_mib"] = int(parts[2])
            except ValueError:
                pass
    # CUDA 런타임 버전은 별도 질의
    head = _run(["nvidia-smi"])
    if head:
        for line in head.splitlines()[:6]:
            for key in ("CUDA Version:", "CUDA UMD Version:"):
                if key in line:
                    info["cuda_version"] = line.split(key)[1].split()[0].strip("| ")
                    break
            if info["cuda_version"]:
                break
    return info


def _packages() -> dict:
    import importlib.metadata as md
    out = {}
    for name in WATCH:
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = None
    return out


def snapshot(ollama_url: str = DEFAULT_OLLAMA) -> dict:
    snap = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ollama": _ollama(ollama_url),
        "gpu": _gpu(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "os": {"system": platform.system(), "release": platform.release(),
               "machine": platform.machine()},
        "packages": _packages(),
    }
    snap["fingerprint"] = fingerprint(snap)
    return snap


def fingerprint(snap: dict) -> str:
    """지연·처리량에 영향을 주는 환경만 골라 만든 8자리 지문.

    지문이 다른 두 측정의 **절대값을 같은 표에 놓지 않는다.** 차이·순위는
    각 측정 안에서는 유효하므로 그대로 쓸 수 있다.
    """
    vals = []
    for a, b in FINGERPRINT_KEYS:
        vals.append(f"{a}.{b}={snap.get(a, {}).get(b)}")
    return hashlib.sha256("|".join(vals).encode()).hexdigest()[:8]


def row_fields(snap: dict) -> dict:
    """실험 레코드 한 줄에 붙일 최소 정보.

    CSV 만 열어 봐도 어떤 환경에서 잰 값인지 알 수 있어야 한다.
    """
    return {
        "env_fingerprint": snap.get("fingerprint"),
        "env_ollama_version": snap.get("ollama", {}).get("version"),
        "env_driver_version": snap.get("gpu", {}).get("driver_version"),
        "env_captured_at": snap.get("captured_at"),
    }


def save(snap: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def describe(snap: dict) -> str:
    o, g = snap.get("ollama", {}), snap.get("gpu", {})
    return (f"지문 {snap.get('fingerprint')} · ollama {o.get('version') or '?'} · "
            f"{g.get('name') or 'GPU?'} 드라이버 {g.get('driver_version') or '?'} · "
            f"CUDA {g.get('cuda_version') or '?'}")


# ══════════════════════════════════════════════════════════════════
def cmd_list() -> int:
    """기록된 실행들의 환경을 모아 어느 것끼리 비교 가능한지 보여 준다."""
    files = sorted(WORK_DIR.glob("*/*_env.json"))
    if not files:
        print("기록된 환경 스냅샷이 없습니다.")
        print("이 기능은 env_snapshot 을 붙인 뒤의 실행부터 채워집니다.")
        print()
        print("참고 — 그 이전 실행의 환경은 work/01_env/ 에 사람이 읽는 형태로만 남아 있고,")
        print("      레코드와 연결돼 있지 않습니다. 부록 L 이 원인을 특정하지 못한 이유입니다.")
        return 0

    groups: dict[str, list[tuple[str, dict]]] = {}
    for f in files:
        try:
            snap = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  읽기 실패 {f}: {exc}")
            continue
        groups.setdefault(snap.get("fingerprint", "?"), []).append((f.parent.name, snap))

    print(f"기록된 실행 {len(files)}건 · 서로 다른 환경 {len(groups)}종\n")
    for fp, items in sorted(groups.items()):
        print(f"── 지문 {fp}  ({len(items)}개 실행)")
        print(f"   {describe(items[0][1])}")
        for name, snap in items:
            print(f"     · {name:<24} {snap.get('captured_at')}")
        print()

    if len(groups) == 1:
        print("모든 실행이 같은 환경이다 — 절대값을 같은 표에 놓아도 된다.")
    else:
        print("환경이 서로 다른 실행이 있다.")
        print("**지문이 다른 측정의 절대 지연·처리량을 같은 표에 놓지 말 것**(부록 L).")
        print("차이와 순위는 각 실행 안에서는 유효하므로 그대로 쓸 수 있다.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="측정 환경 스냅샷")
    ap.add_argument("--list", action="store_true",
                    help="work/ 아래 기록된 실행들의 환경을 대조한다")
    ap.add_argument("--save", metavar="PATH", default=None, help="JSON 으로 저장")
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA)
    args = ap.parse_args()

    if args.list:
        return cmd_list()

    snap = snapshot(args.ollama_url)
    print(describe(snap))
    print()
    print(json.dumps(snap, ensure_ascii=False, indent=2))
    if args.save:
        p = save(snap, Path(args.save))
        print(f"\n기록: {p}")
    if not snap["ollama"]["reachable"] and not snap["ollama"]["version"]:
        print("\n주의: Ollama 버전을 얻지 못했습니다. 서버가 떠 있지 않으면")
        print("      이 실행의 환경 기록에 가장 중요한 항목이 비게 됩니다.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
