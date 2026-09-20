## **사내 CS·기술지원 문의 자동 분류·요약 시스템 — 로컬 LLM 모델 선정** ## 

최종 모델 선정 근거와 실험 결과 전체는 [FINAL_REPORT.md](FINAL_REPORT.md)에 있습니다 (문제 정의 → 후보 조사 → 실행 환경 → 평가 기준 → 로컬 40회 실험 → Local–Cloud 비교 → 최종 선정 → 한계·후속).
결과만 쉬운 말로 빠르게 보려면 [PUBLIC_REPORT.md](PUBLIC_REPORT.md)를 참고하세요.
역할별 책임과 산출물 대응은 [ROLES_AND_DELIVERABLES.md](ROLES_AND_DELIVERABLES.md)에 있습니다 (`roles_manifest.json`에서 자동 생성·자체 검증).

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
├── 05_quality_scoring.py    # step5: 04(또는 06) 결과를 규칙+LLM-judge로 채점
├── 06_cloud_comparison.py   # step6: Cloud 비교 5문항을 OpenAI(gpt-5.6-luna)로 호출
├── 07_quantization_comparison.py # 도전 실습 A: 같은 모델의 양자화 3단계 비교 + VRAM 실측
├── QUANT_EXPERIMENT_DESIGN.md # 도전 실습 A 실험 설계서 (변수 통제·루프·해석 규칙)
├── 08_prompt_ab_experiment.py # 프롬프트 보강 A/B 실험 (기권·권한·환각 규칙)
├── 08_prompt_ab_report.py   # A/B 결과 비교와 자동 판정
├── PROMPT_AB_DESIGN.md      # 프롬프트 보강 A/B 실험 설계서
├── 09_embedding_experiment.py # 도전 실습 B: 문장 임베딩과 코사인 유사도
├── 10_klue_embedding_eval.py # KLUE-STS·NLI로 임베딩 검증 (LLM 호출 없음)
├── 11_klue_korquad_eval.py  # KLUE-YNAT·MRC·KorQuAD로 생성 모델 능력 측정
├── KLUE_KORQUAD_DESIGN.md   # 공개 벤치마크 반영 설계서
├── VLLM_SERVING_DESIGN.md   # vLLM 서빙 교체 + 동시성 측정 설계서 (미실행)
├── 12_rubric_sensitivity.py # 채점 관대성 민감도 재채점 (모델 호출 없음)
├── build_eval_set_v2.py     # 평가셋 10 → 54문항 확장 (7개 검증 통과 시에만 기록)
├── deps_check.py            # 의존성 로딩 실패 원인을 구분해 출력
├── env_snapshot.py          # 측정 환경 스냅샷 + 비교 가능 여부 지문
├── data/
│   ├── 02_prompts_and_eval.json   # 질문·gold 라벨·채점 기준의 단일 출처 (v2, 54문항)
│   ├── 02_prompts_and_eval.v1.json # 10문항 판 백업 — 보고서 수치의 기준
│   ├── eval_set_coverage.md       # 18칸 커버리지 표
│   └── eval_set_v2_review.md      # 54문항 전체 검토용 대조표
├── FINAL_REPORT.md          # 최종 모델 선정 보고서 (요구사항 평가표 대응 포함)
├── ROLES_AND_DELIVERABLES.md # 역할 분담과 산출물 대응 (자동 생성)
├── roles_manifest.json      # 역할·책임·산출물의 단일 출처
├── generate_roles_doc.py    # 위 JSON → 문서 생성 + 산출물 존재 검증
├── PUBLIC_REPORT.md         # 프로젝트 결과 요약 (쉬운 말 버전)
└── work/                    # 실행 결과 저장 위치
    ├── 01_env/              # 환경 점검 기록
    ├── 02_calls/            # 호출·평가 JSON (최신)
    ├── 04_experiment/       # 실험 결과 csv/json
    ├── 07_quantization/     # 양자화 비교 결과 + 자원 실측
    ├── 08_prompt_ab/        # 프롬프트 보강 A/B 결과 + 판정 요약
    ├── 09_embedding/        # 임베딩 유사도 결과 + 실습 보고서
    ├── 10_klue_embedding/   # KLUE-STS·NLI 검증 결과
    ├── 11_klue_korquad/     # 벤치마크 결과 + 요약
    ├── 12_rubric_sensitivity/ # 관대/엄격 채점 대조
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

## 도전 실습 A — 양자화 비교 (선택)

같은 모델(llama3.2 3B Instruct)의 양자화 수준만 바꿔 VRAM·속도·품질 변화를 관측합니다. 서로 다른 모델의 차이를 양자화 효과로 해석하지 않기 위해, 계열과 파라미터 수가 같고 양자화만 다른 세 버전으로 한정합니다.

먼저 세 버전을 받습니다. `q4_K_M`은 기존 `llama3.2:3b`와 같은 실체라 즉시 끝나고, 나머지는 각각 약 3.4GB·6.4GB를 내려받습니다.

```powershell
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull llama3.2:3b-instruct-q8_0
ollama pull llama3.2:3b-instruct-fp16
```

```powershell
uv run python 07_quantization_comparison.py    # 패스 2 x 버전 3 x 질문 10 = 60회
uv run python 05_quality_scoring.py work/07_quantization/07_quantization_results.json
```

질문·정답·생성 설정(`temperature=0.1`)·반복 횟수는 `04_local_experiment.py` 본실험과 동일하며, 채점도 같은 파이프라인을 씁니다. 실험 설계의 근거(변수 통제, 루프 순서, 측정 지점, 해석 규칙)는 [QUANT_EXPERIMENT_DESIGN.md](QUANT_EXPERIMENT_DESIGN.md)에 있습니다. 루프는 `패스 → 버전 → 질문` 3중 구조이고, 패스 2는 버전 순서를 뒤집어 GPU 발열 같은 시간 드리프트가 특정 버전에만 쏠리지 않게 합니다. 버전마다 워밍업 1회를 본실험과 분리해 저장하고 집계에서 제외하며, 버전 전환 시 이전 모델을 언로드해 콜드스타트 조건을 맞춥니다.

워밍업 직후 모델이 메모리에 올라간 상태에서 Ollama `/api/ps`를 조회해 총 점유(`size`)와 GPU 점유(`size_vram`)를 바이트 단위로 기록합니다. 두 값의 비율이 100% 미만이면 일부가 CPU로 내려간 것이므로, 그 버전의 속도 저하는 양자화가 아니라 오프로딩 때문일 수 있습니다 — 스크립트가 이 경우를 경고로 표시하고 `07_quantization_resources.json`에 비율을 남깁니다. FP16(약 6.4GB)은 8GB VRAM 환경에서 KV 캐시까지 더하면 이 상황이 발생할 수 있습니다.

`work/02_calls/eval_*.json`(질문 10개 x 모델 2종, 반복 없음)은 예전에 있던 `run_evaluation.py`로 만든 1회성 결과이며, 그 스크립트는 `04_local_experiment.py`로 기능이 흡수되어 삭제했습니다. 해당 파일들은 과거 실행 기록으로만 남아있고 재생성하는 스크립트는 없습니다.

같은 이름의 결과 파일은 덮어쓰기 됩니다. 이전 실행을 남겨두려면 `work/04_experiment`를 `work/archive/<날짜>`로 옮긴 뒤 다시 실행하세요.


## 프롬프트 보강 A/B 실험

최종 보고서 §7이 '배포 전 필수 조치'로 권고한 시스템 프롬프트 보강을 실제로 적용해, Q9(정보 부족)·Q10(범위 밖) 게이트 탈락이 해소되는지 검증합니다. 동시에 부록 B의 'P(프롬프트로 교정 가능)' 재판정이 이 두 문항에도 성립하는지 확인합니다.

현행 프롬프트는 `category`에 "정보 부족"이 허용값이라고 말하지 않는데 gold는 Q9에서 그 값을 요구합니다. 모델 실패가 아니라 명세 누락일 가능성이 높다는 것이 이 실험의 가설입니다.

```powershell
uv run python 08_prompt_ab_experiment.py    # 패스 2 x 모델 2 x 질문 10 x 조건 2 = 80회
uv run python 05_quality_scoring.py work/08_prompt_ab/08_prompt_ab_results.json
uv run python 08_prompt_ab_report.py        # 비교표와 자동 판정
```

조건 교체 비용이 0이므로 독립변수를 루프 가장 안쪽에 두고, 같은 질문의 A/B를 몇 초 간격으로 붙여 실행합니다(대응 표본 설계). 쌍 안의 순서도 교대해 '먼저/나중' 편향을 없앱니다. Q9·Q10 개선과 Q1~Q8 회귀를 반드시 함께 보며, 판정 기준은 결과 열람 전에 [PROMPT_AB_DESIGN.md](PROMPT_AB_DESIGN.md) §6에 고정해 두고 `08_prompt_ab_report.py`가 코드로 적용합니다. 추가 다운로드는 없습니다.

## 도전 실습 B — 임베딩 실험 (선택)

`문장 → 임베딩 벡터 → 코사인 유사도 → 의미 유사도` 흐름을 실습합니다. Vector DB와 RAG는 구현하지 않습니다.

```powershell
uv add sentence-transformers        # torch를 함께 받으므로 용량이 큽니다
uv run python 09_embedding_experiment.py
```

최초 실행 시 임베딩 모델 `paraphrase-multilingual-MiniLM-L12-v2`(약 470MB)와 토큰 ID 비교용 Qwen2.5 토크나이저(수 MB)를 내려받습니다. 한국어 문장을 다루므로 다국어 임베딩 모델을 쓰며, 모든 비교는 이 모델 하나·같은 설정(정규화 켬)으로 수행합니다.

네 부분으로 구성됩니다.

1. **기본 흐름** — 발제 예시 3문장("고양이가 소파 위에 있다" 등)의 코사인 유사도 행렬
2. **토큰 ID와 임베딩의 구분** — 생성 모델의 토큰 ID는 조각 번호의 나열이고 임베딩은 고정 길이 의미 좌표입니다. 토큰이 거의 같은데 뜻이 반대인 쌍("환불해 드리겠습니다" / "환불해 드릴 수 없습니다")과, 겹치는 단어가 없는데 뜻이 같은 쌍("로그인이 되지 않습니다" / "계정에 접속할 수 없습니다")으로 둘이 별개임을 보입니다
3. **유사도를 사실성 점수로 쓰면 안 되는 이유** — 실제 실험에서 게이트에 걸린 응답(부록 C의 fp16 확정 보상 약속, 1차 실행의 날조된 주문번호)과 안전한 응답을 같은 기준 문장에 대고 재어, 하나의 임계값으로 갈리는지 확인합니다. 결론 문장은 하드코딩하지 않고 실측값에서 생성됩니다
4. **프로젝트 적용** — 질문 10개를 임베딩해 같은 카테고리 쌍과 다른 카테고리 쌍의 평균 유사도를 비교합니다

유사도는 **표현이 얼마나 닮았는가**를 재지 **내용이 사실인가**를 재지 않습니다. 이 구분이 실습의 핵심입니다.

## KLUE·KorQuAD 반영

공개 한국어 벤치마크를 평가에 반영하되, **벤치마크와 루브릭을 섞지 않습니다.** 벤치마크는 모델의 일반 능력을 재고, 이 프로젝트의 루브릭은 제품 요구 충족(JSON 스키마, 권한 밖 확답 금지, 기권 적절성)을 잽니다. 따라서 **벤치마크 점수는 게이트에도 선정 가중치에도 넣지 않고**, 후보 조사 단계의 사전 정보와 10문항 파일럿의 외부 타당도 점검에만 씁니다. 근거는 [KLUE_KORQUAD_DESIGN.md](KLUE_KORQUAD_DESIGN.md)에 있습니다.

처음에는 아래 5개 과제를 프로젝트의 채점 축에 하나씩 대응시켰습니다. 그 대응이 **축 이름만 보고 맺은 것**이었다는 지적을 받고 다시 따진 결과가 「판정」 열입니다. 우리는 사내 CS 전용 도메인이고 후보는 3B·7B인데, 범용 LLM 벤치마크는 다른 모집단·다른 과제를 위해 만들어졌습니다.

| 대응시켰던 축 | 벤치마크 | 실제로 재는 것 | 판정 |
| --- | --- | --- | --- |
| 분류 | KLUE-YNAT | 뉴스 제목의 주제 | **기각** — 라벨 체계·문체·길이가 CS 라우팅과 다름 |
| 추출 | KorQuAD 1.0 / KLUE-MRC(답 있음) | 1,000자 위키 지문의 정답 구간 | **기각** — 한 문장 불만글에서 주문번호 뽑기와 다름 |
| **기권** | **KLUE-MRC(답 없음)** | **지문에 답이 없을 때 물러나는가** | **채택 (대리 지표)** — 재는 행동이 같음 |
| 의미 유사도 | KLUE-STS | 문장 쌍 유사도 | **채택 (계측기 검증)** — 부록 E가 쓴 도구의 타당성 점검 |
| 유사도 ≠ 사실성 | KLUE-NLI(모순 라벨) | 문장 쌍 모순 | **채택 (계측기 검증)** — 같은 사유 |

채택한 기권 축의 결과는 이렇습니다 (모델당 40건).

| 모델 | 기권 정확도 | 지어낸 건수 |
| --- | --- | --- |
| `llama3.2:3b` | 0.550 (22/40) | 18 |
| `qwen2.5:7b` | 0.825 (33/40) | 7 |

기각한 과제 점수는 참고 기록으로만 남깁니다. **YNAT 점수가 낮다는 사실 자체가 기각 판단을 뒷받침합니다** — 축 이름은 양쪽 다 「분류」인데 본실험과 값이 크게 벌어지니, 같은 능력을 재고 있지 않다는 뜻입니다. 자세한 내용은 FINAL_REPORT.md 부록 F에 있습니다.

```powershell
uv add datasets
uv run python 10_klue_embedding_eval.py    # STS 상관 + NLI 모순 분리, LLM 호출 없음
uv run python 11_klue_korquad_eval.py      # YNAT 100 + KorQuAD 60 + MRC 80, 모델 2종
```

해석 규칙 네 가지를 실행 전에 고정했습니다 — 선정을 바꾸지 않을 것, 도메인이 다르다는 사실을 매번 붙일 것, 공개 데이터라 사전학습 오염 가능성이 있으니 절대값 대신 상대 비교만 할 것, 10문항 결과와 어긋나면 어긋남 자체를 보고할 것.

## 도메인 평가셋 확장 — 10문항 → 54문항

보고서가 최대 한계로 지목한 것은 커버리지 갭이었습니다. 카테고리 6종 × 정보충분도 3종 = 18칸 중 10문항이 덮은 것은 9칸뿐이었습니다. 위 벤치마크로는 이 자리를 메울 수 없어(재는 것이 다르므로) **우리 도메인 문항을 직접 늘렸습니다.**

```powershell
uv run python build_eval_set_v2.py     # 모델 호출 없음, 수초 내 완료
```

| 지킨 것 | 이유 |
| --- | --- |
| 기존 ID 1~10을 한 글자도 바꾸지 않음 | 지금까지의 **185회 실험 결과가 그대로 유효**해야 함 |
| 프롬프트 템플릿 그대로 (54문항 전부 1종) | 순수 표본 확장. 「카테고리 열거형 제시」는 프롬프트 변경이므로 조건 C로 분리 |
| Cloud 비교 5문항(ID 1·2·6·9·10) 고정 | 비교 대상이 흔들리지 않게 |

빌더는 7개 검증을 **모두 통과해야만** 파일을 씁니다 (ID 1~10 동일성, ID 중복 없음, 템플릿 1종, 18칸 각 3건, 필수 필드 존재, Cloud 셋 불변, 정보부족 라벨 일관성). 하나라도 실패하면 기록을 거부합니다. v1 백업(`data/02_prompts_and_eval.v1.json`)과 커버리지 표를 함께 만듭니다.

**주의**: 54문항은 아직 실행하지 않았습니다. 보고서의 모든 수치는 여전히 10문항 기준입니다.

## 채점 관대성 민감도 점검

분류 정확도 판정 키워드에서 「정보 부족」 항목에 `"고객 문의"`, `"기타"` 가 들어 있었습니다. 둘 다 정보가 부족하다는 **판단의 표현이 아니라** 아무 데나 붙일 수 있는 기본 바구니입니다. 판단을 포기하는 모델이 점수를 얻습니다.

```powershell
uv run python 12_rubric_sensitivity.py   # 저장된 응답 재채점, 모델 호출 없음
```

저장된 원본 응답을 그대로 두고 채점 규칙만 바꿔 185회를 다시 계산합니다. 결과는 **본실험 순위 불변, 공표 수치 변동 0건** — 관대성 구멍은 Q9에만 있었고 Q9는 공표 수치의 분모에 들어가지 않았기 때문입니다. 다만 v2에서는 정보 부족 문항이 1개에서 10개가 되므로, **다음 실행부터는 엄격 기준을 기본값으로 씁니다.** 자세한 내용은 FINAL_REPORT.md 부록 H에 있습니다.

## vLLM 서빙 교체와 동시성 측정 (설계 완료, 미실행)

§7 최종 선정에서 "문의가 몰릴 때"를 근거로 들었으면서 실제로는 순차 지연(n=1)만 측정했습니다. 그 공백을 메우는 작업입니다. 설계는 [VLLM_SERVING_DESIGN.md](VLLM_SERVING_DESIGN.md)에 있고, **아직 실행하지 않았습니다.**

- **단계 0** — 문항·프롬프트·채점기를 그대로 두고 서빙만 갈아 품질 차이를 격리합니다.
- **단계 1** — 8GB에서 동시 요청을 얼마나 받아내는지, 게이트 `P95 ≤ 6초`가 동시성 몇까지 유지되는지 잽니다.

Ollama → vLLM 은 **네 가지가 동시에 바뀝니다** (엔진·가중치 포맷·런타임(WSL2)·샘플러 기본값). 그래서 "vLLM이 더 빠르다"는 문장은 이 실험으로 쓸 수 없습니다. 대신 WSL2 안에서 Ollama도 한 번 돌려(구성 B) 런타임 효과 하나는 떼어냅니다.

| 구성 | 엔진 | 포맷 | 런타임 |
| --- | --- | --- | --- |
| A | Ollama | GGUF Q4_K_M | Windows 네이티브 |
| B | Ollama | GGUF Q4_K_M | WSL2 |
| C | vLLM | AWQ 4bit | WSL2 |

A↔B 가 런타임 효과, B↔C 가 엔진+포맷 효과입니다.

에이전트는 이 문서의 범위가 아닙니다. Ollama도 도구 호출을 지원하므로 **에이전트를 만들려고 vLLM이 필요한 것이 아닙니다.** vLLM을 쓰는 이유는 동시성과 제약 디코딩 두 가지뿐입니다.

## 측정 환경 기록 — 성능 수치가 스스로 환경을 증언하게

같은 코드·같은 모델·같은 문항의 지연 측정이 나흘 만에 3.8배 달라졌는데 원인을 특정하지 못했습니다(FINAL_REPORT 부록 L). 당시 환경이 레코드와 연결돼 있지 않았기 때문입니다.

이제 성능 스크립트(19·20·21)는 실행할 때마다 환경을 함께 남깁니다.

```powershell
uv run python env_snapshot.py          # 지금 환경과 지문
uv run python env_snapshot.py --list   # 기록된 실행들이 서로 비교 가능한지
```

레코드 각 줄에 `env_fingerprint`·`env_ollama_version`·`env_driver_version`·`env_captured_at` 이 붙고, 실행 폴더에 `<n>_env.json` 전체 기록이 남습니다. 지문은 지연·처리량에 직접 영향을 주는 것만(Ollama 버전·GPU·드라이버·CUDA) 골라 만든 8자리 해시입니다.

> **지문이 다른 두 측정의 절대값을 같은 표에 놓지 않습니다.** 차이와 순위는 각 측정 안에서 유효하므로 그대로 씁니다.
