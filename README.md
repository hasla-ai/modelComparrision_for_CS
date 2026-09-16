# Python으로 모델에 첫 질문 보내기

Ollama 설치와 CLI 대화를 마친 뒤, VS Code와 uv로 실행하는 작은 예제입니다.
자세한 환경 준비와 지표 설명은 [두 번째 노션 가이드](https://app.notion.com/p/3d41698118388039be07cca84e1f743c)를 확인하세요.

## 폴더 구조

```
Python_Model_Start/          # 프로젝트 루트 (여기서 터미널을 엽니다)
├── pyproject.toml           # 의존성 정의
├── uv.lock
├── .venv/                   # uv sync로 만들어지는 가상환경 (git 제외)
├── 01_ollama_chat.py        # 로컬 모델에 질문 1번
├── 02_luna_chat.py          # OpenAI API에 질문 1번
├── 03_measure_time.py       # 응답 전체 소요 시간 측정
├── 02_call_models.py        # step3: 모델 2종 호출 + 워밍업
├── 04_local_experiment.py   # step4: data/02_prompts_and_eval.json을 QUESTIONS로 로드해 반복 실행 + 지표 집계
├── 05_quality_scoring.py    # step5: 04(또는 06) 결과를 규칙+LLM-judge로 채점 (docs/quality_scoring_spec.md 구현)
├── 06_cloud_comparison.py   # step6: Cloud 비교 5문항을 OpenAI(gpt-5.6-luna)로 호출
├── data/
│   └── 02_prompts_and_eval.json   # 질문·gold 라벨·채점 기준의 단일 출처
├── docs/
│   ├── model_selection_spec.md    # 요구사항 반영 명세, 게이트/우선순위, 실험 결과
│   ├── experiment_design.md       # 평가셋 설계 방법론, taxonomy 매핑, Cloud 5문항 선정 근거
│   ├── quality_scoring_spec.md    # 품질 채점 기능 명세 (6개 축, 게이트/가중합, 신뢰도 확보)
│   └── final_selection_report.md  # STEP 8 최종 모델 선정 리포트 (결론: llama3.2:3b)
└── work/                    # 실행 결과 저장 위치
    ├── 01_env/              # 환경 점검 기록
    ├── 02_calls/            # 호출·평가 JSON (최신)
    ├── 04_experiment/       # 실험 결과 csv/json
    └── archive/             # 지난 실행 보관
```

결과 저장 경로는 스크립트 파일 위치를 기준으로 계산합니다. 어느 폴더에서 실행하든 항상 위 `work/` 아래에 쌓이므로, `work` 폴더가 여기저기 생기지 않습니다.

## 준비

1. VS Code에서 `pyproject.toml`이 보이는 이 폴더를 엽니다.
2. **터미널 → 새 터미널**에서 PowerShell을 엽니다. Ollama 앱도 실행합니다.
3. 아래 명령을 한 줄씩 입력합니다. `.venv`를 직접 활성화하지 않아도 됩니다.

```powershell
uv sync
uv run python --version
```

Python `3.12.x`가 나오면 준비됐습니다. 최초 실행에는 Python과 패키지 다운로드가 필요할 수 있습니다.

## 실행

로컬 모델이 없다면 먼저 다운로드합니다. 이미 설치했다면 건너뜁니다.

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
uv run python 01_ollama_chat.py
```

다음 명령은 OpenAI API에 질문을 **1번** 보냅니다. 유효한 API 키와 사용 가능한 잔액·모델 접근 권한이 필요합니다.

```powershell
uv run python 02_luna_chat.py
```

키를 요청하면 터미널에 붙여넣고 Enter를 누릅니다. 입력한 키는 화면에 보이지 않습니다. 키를 코드·채팅·노션에 적지 마세요. 키 없이 Enter를 누르면 호출하지 않고 종료합니다. 키를 받지 않았다면 이 단계는 건너뜁니다.

API 사용 비용은 유료입니다. 실습과 프로젝트를 합쳐 인당 최대 5달러 범위에서 진행하고, [사용량](https://platform.openai.com/usage)을 확인하세요. 이 코드는 자동 재시도하지 않으며, 누적 지출을 자동 차단하지는 않습니다. 오류가 났을 때도 반복 실행 전에 사용량과 오류 안내를 확인합니다.

마지막 명령은 로컬 Ollama에 질문을 1번 보내고 전체 답변을 받는 데 걸린 시간을 출력합니다.

```powershell
uv run python 03_measure_time.py
```

이 값은 첫 토큰 시간(TTFT)이 아닙니다. 모델을 처음 불러오는 시간과 PC 상태, 답변 길이에 영향을 받으므로 한 번의 숫자만으로 모델 우열을 정하지 않습니다.

실행 파일 안의 `QUESTION` 한 줄을 바꾸고 **Ctrl+S**로 저장하면 다른 질문을 보낼 수 있습니다. 이 파일들은 대화 기록을 유지하지 않고, 매번 질문 1개를 새로 보냅니다.

## 평가·실험 실행

아래 세 개는 로컬 Ollama를 여러 번 호출하므로 시간이 걸립니다. 반드시 프로젝트 루트에서 실행하세요.

```powershell
uv run python 02_call_models.py       # -> work/02_calls/call_*.json, warmup_*.json
uv run python 04_local_experiment.py  # -> work/04_experiment/04_local_experiment_results.csv, .json (질문 10개 x 모델 2종 x 2회 = 40회)
uv run python 05_quality_scoring.py   # -> work/04_experiment/05_quality_scoring_results.csv, .json (04 결과를 채점)
uv run python 06_cloud_comparison.py  # -> work/06_cloud_comparison/06_cloud_comparison_results.csv, .json (사전 선정된 Cloud 비교 5문항)
uv run python 05_quality_scoring.py work/06_cloud_comparison/06_cloud_comparison_results.json  # 06 결과도 같은 방식으로 채점
```

`05_quality_scoring.py`는 `qwen3:4b-instruct-2507-q4_K_M`을 채점자(judge)로 사용합니다. `ollama pull qwen3:4b-instruct-2507-q4_K_M`으로 미리 받아두세요(채점 대상 모델과 겹치지 않아야 합니다).

`06_cloud_comparison.py`는 `02_luna_chat.py`와 동일한 OpenAI Responses API(`gpt-5.6-luna`)만 사용합니다. Upstage Solar 등 다른 Cloud API는 이 실험에서 쓰지 않습니다. 키 입력 방식과 비용 안내는 `02_luna_chat.py`와 동일합니다 — 키를 요청하면 터미널에 붙여넣고 Enter를 누르며, 화면에 보이지 않고 코드·로그·노션에 남지 않습니다. **키를 스크립트 인자나 셸 명령에 직접 넣지 마세요.** 실행 중 `getpass` 프롬프트에서만 입력해야 셸 히스토리에 남지 않습니다.

`04_local_experiment.py`는 `data/02_prompts_and_eval.json`을 직접 읽어 질문 목록(`QUESTIONS`)을 만듭니다. 문항을 바꾸려면 이 JSON 파일만 수정하면 됩니다.

`work/02_calls/eval_*.json`(질문 10개 x 모델 2종, 반복 없음)은 예전에 있던 `run_evaluation.py`로 만든 1회성 결과이며, 그 스크립트는 `04_local_experiment.py`로 기능이 흡수되어 삭제했습니다. 해당 파일들은 과거 실행 기록으로만 남아있고 재생성하는 스크립트는 없습니다.

같은 이름의 결과 파일은 덮어쓰기 됩니다. 이전 실행을 남겨두려면 `work/04_experiment`를 `work/archive/<날짜>`로 옮긴 뒤 다시 실행하세요.
