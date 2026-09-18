"""의존성 로딩 실패를 감추지 않고 원인을 드러내는 헬퍼.

왜 필요한가
  `except ImportError: print("패키지가 없습니다")` 는 두 가지 전혀 다른 상황을
  하나로 뭉갠다.

    (1) 정말로 설치가 없다                        → ModuleNotFoundError
    (2) 설치는 되어 있는데 불러오다 실패한다        → 그 밖의 ImportError / OSError 등
        (의존성 버전 충돌, Windows DLL 로드 실패, 설치 손상 …)

  (2)를 "없습니다"라고 출력하면 사용자는 이미 설치된 패키지를 또 설치하려 하고,
  진짜 원인은 끝내 보이지 않는다. 이 모듈은 둘을 구분하고, 실패한 경우 실제
  예외 메시지와 실행 중인 파이썬·주요 패키지 버전을 함께 찍는다.

사용:
    from deps_check import require
    st = require("sentence_transformers", "sentence-transformers")
    if st is None:
        return 1
"""

from __future__ import annotations

import importlib
import sys
import traceback

# 진단 시 버전을 함께 보여줄 패키지들
_WATCH = [
    "sentence-transformers", "torch", "transformers", "huggingface-hub",
    "tokenizers", "datasets", "numpy", "scipy", "scikit-learn", "pandas", "pyarrow",
]


def report_env() -> None:
    """실행 중인 인터프리터와 주요 패키지 버전을 출력한다."""
    import importlib.metadata as md

    print("  ── 실행 환경 ──")
    print(f"  python      : {sys.version.split()[0]}")
    print(f"  실행 파일   : {sys.executable}")
    print("  설치된 버전 :")
    for name in _WATCH:
        try:
            print(f"      {name:<22} {md.version(name)}")
        except Exception:
            print(f"      {name:<22} (설치 없음)")
    print()
    print("  참고: 다른 프로젝트의 가상환경이 활성화되어 있으면 `uv run`은 그것을 무시하고")
    print("        이 프로젝트의 .venv를 씁니다. 둘이 다르면 같은 스크립트도 결과가 달라집니다.")
    print("        `deactivate` 후 다시 실행해 환경을 하나로 맞추세요.")


def require(module_name: str, package_name: str | None = None, *, quiet: bool = False):
    """모듈을 불러오고, 실패하면 원인을 구분해 출력한 뒤 None을 돌려준다."""
    pkg = package_name or module_name.replace("_", "-")
    try:
        return importlib.import_module(module_name)

    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        top = module_name.split(".")[0]
        if missing == top:
            # (1) 정말로 설치가 없다
            print(f"[{pkg}] 가 설치되어 있지 않습니다.")
            print(f"\n    uv add {pkg}\n")
        else:
            # (2) 본체는 있는데 하위 의존성이 빠졌다 — 설치 손상 또는 충돌
            print(f"[{pkg}] 는 설치되어 있으나, 그 안에서 '{missing}' 을(를) 찾지 못했습니다.")
            print("  의존성이 빠졌거나 버전이 어긋난 상태입니다.")
            print(f"\n    uv add {missing}        # 또는  uv sync --reinstall\n")
            if not quiet:
                report_env()
        return None

    except Exception as exc:
        # (2) 설치는 있는데 불러오다 터졌다 — 버전 충돌, DLL 로드 실패 등
        print(f"[{pkg}] 를 불러오지 못했습니다. 설치는 되어 있으나 오류가 납니다.")
        print(f"  {type(exc).__name__}: {exc}\n")
        if not quiet:
            print("  ── 실패 지점 ──")
            for line in traceback.format_exc().splitlines():
                s = line.strip()
                if s.startswith("File ") or s.startswith("from ") or s.startswith("import "):
                    print(f"    {s}")
            print()
            report_env()
            print("  흔한 원인:")
            print("    · 다른 패키지를 추가하면서 공통 의존성 버전이 바뀜 → uv sync --reinstall")
            print("    · Windows에서 torch가 요구하는 런타임 DLL 부재 → Visual C++ 재배포 패키지 설치")
            print("    · 서로 다른 가상환경을 번갈아 사용 → deactivate 후 uv run 으로 통일")
        return None
