# On-device/Cloud LLM Routing — 사전 Router 기반 접근 초안 실험 보고서

## 목적

- On-device/Cloud LLM routing 관점에서, **accuracy를 최대한 유지하면서 weak(on-device) model 사용 비율을 얼마나 높일 수 있는지** 확인
- 핵심 질문: full strong(cloud) model 대비 성능 하락을 작게 유지하는 조건에서, 어느 정도의 요청을 weak model로 처리할 수 있는가
- 본 실험은 특정 routing 방법론의 **최종 성능 검증이 아님**
  - 사전 router(prompt 입력만으로 실행 전 모델을 선택하는 router) 기반 접근이 strong/weak routing 문제에서 어느 정도 가능성이 있는지 **초안 수준으로 확인**하는 것이 목적
  - 결과는 최종 성능이 아니라 추가 실험 필요성을 판단하기 위한 기준으로 활용

---

## 1. 관련 방법론 소개

### 1.1 RouteLLM (본 실험에서 사용한 MF router)

- Strong/weak **두 모델 사이에서 query별로 어떤 모델을 쓸지** 정하는 binary routing framework
- query에 대해 **strong이 weak보다 나은 응답을 낼 확률** `P(strong wins | query)`을 예측하고, threshold 이상이면 strong, 미만이면 weak로 routing → **threshold로 strong 호출 비율 ↔ 품질 trade-off 제어**
- 논문은 여러 router 구조(Similarity-weighted ranking, **Matrix Factorization**, BERT classifier, Causal LLM classifier)를 비교하며, 그중 **MF router를 성능·routing overhead 측면에서 가장 안정적인 대표 방법**으로 제시
- **본 실험은 이 MF router를 사용**
  - query embedding과 model embedding의 **latent interaction**을 학습해 모델별 예상 품질 score를 예측하고, **strong score − weak score**로 routing
  - query embedding만으로 학습·추론하여 **가볍고**, 연속 score를 출력해 **threshold로 weak/strong 비율을 유연하게 조절** 가능

### 1.2 UniRoute (Universal Model Routing)

- **여러 LLM 후보 pool**에서 query별 cost 대비 적절한 모델을 고르는 routing framework (RouteLLM의 strong/weak 2-모델과 달리 다중 LLM 대상)
- 핵심 차별점: 각 LLM을 내부 weight가 아니라 **validation set에서의 정답/오답(error) 패턴 벡터**로 표현 → 학습 때 못 본 **새 LLM이 추가돼도 재학습 없이** routing 가능 (dynamic routing)
- prompt feature와 LLM error feature의 interaction으로 예상 error를 추정하고, `예상 error + λ·cost`가 가장 작은 LLM을 선택
- 본 실험에서는 그 **cluster 기반 변형**(prompt를 K-means clustering 후 LLM의 cluster별 평균 error 사용)을 **strong/weak 2-모델에 맞춰** 비교 대상으로 사용

---

## 2. 실험 세팅

### 2.1 데이터 (BFCL)

- 평가 데이터로 **BFCL(Berkeley Function Calling Leaderboard) v4**의 single-turn 카테고리(non-live + live)를 사용
- BFCL은 query와 함께 **호출 가능한 function 정의**가 주어지고, 모델이 적절한 function call을 생성하는지를 채점하는 **function calling / tool calling 중심** 데이터
- 각 sample은 query(question)와 function 목록으로 구성되며, 모델의 출력이 정답 function call과 일치하는지로 pass/fail 판정

**BFCL 데이터 예시** (후보 function이 여러 개인 `multiple` 카테고리)

```json
{
  "id": "multiple_0",
  "question": [[{"role": "user",
    "content": "Can I find the dimensions and properties of a triangle, if I know its three sides are 5 units, 4 units and 3 units long?"}]],
  "function": [
    {
      "name": "triangle_properties.get",
      "description": "Retrieve the dimensions, such as area and perimeter, of a triangle if lengths of three sides are given.",
      "parameters": {
        "type": "dict",
        "properties": {
          "side1": {"type": "integer", "description": "The length of first side."},
          "side2": {"type": "integer", "description": "The length of second side."},
          "side3": {"type": "integer", "description": "The length of third side."},
          "get_area": {"type": "boolean", "description": "Calculate area (default: true)."},
          "get_perimeter": {"type": "boolean", "description": "Calculate perimeter (default: true)."},
          "get_angles": {"type": "boolean", "description": "Calculate internal angles (default: true)."}
        },
        "required": ["side1", "side2", "side3"]
      }
    },
    {
      "name": "circle_properties.get",
      "description": "Retrieve the dimensions, such as area and circumference, of a circle if radius is given.",
      "parameters": {
        "type": "dict",
        "properties": {
          "radius": {"type": "float", "description": "The length of radius."},
          "get_area": {"type": "boolean", "description": "Calculate area (default: true)."},
          "get_circumference": {"type": "boolean", "description": "Calculate circumference (default: true)."}
        },
        "required": ["radius"]
      }
    }
  ]
}
```

- 위 예시처럼 모델은 **여러 후보 function 중 올바른 것(`triangle_properties.get`, `circle_properties.get` 아님)을 선택**하고, **argument(`side1=5`, `side2=4`, `side3=3`)까지 정확히 추출**해야 pass
- 즉 단순 호출이 아니라 **올바른 function 선택 + argument 추출**이 모두 맞아야 하며, 후보가 많을수록 weak model이 틀릴 여지가 커짐

### 2.2 모델 / 임베딩 / Router 설정

- **모델 쌍 (strong vs weak)** — 두 조합을 평가
  - Pair A: `Qwen3.5-0.8B` (weak) vs `Qwen3.5-9B` (strong)
  - Pair B: `Qwen3.5-2B` (weak) vs `Qwen3.5-9B` (strong)
- **Embedding model**: `multilingual-e5-small`
  - 기존에 사용하던 embedding(STE)과 **동일한 크기**이면서 **MTEB 성능이 유사**
  - RouteLLM MF router가 사용하는 query embedding을 **OpenAI embedding 대신 multilingual-e5-small로 대체**하여 실험 (로컬에서 동작, 별도 API 불필요)
- **Router**: MF router(주), Random / UniRoute(비교)
- **실험 방식**: **threshold를 조절**하면서 weak model 사용 비율과 전체 성능 변화를 함께 확인

---

## 3. 평가 방식

- 각 sample에 대해 router가 **strong 또는 weak 중 하나의 모델을 선택**하고, 선택된 모델의 pass/fail 결과로 전체 성능(pass rate)을 계산
- **full strong model(모든 요청을 strong에 보냄)의 성능을 기준선**으로 삼고, router 적용 후 성능이 얼마나 유지되는지 확인
  - 본 실험 기준선: full strong = **81.62%**
- **핵심 평가 포인트**: full strong 대비 성능 하락이 **1% 이내(pass rate ≥ 80.62%)**인 조건에서, weak model을 **몇 %까지 사용**할 수 있었는지

---

## 4. 실험 결과

### 4.1 Threshold 및 graph 해석 가이드

- Graph는 **x축 = strong model 호출 비율(%)**, **y축 = 전체 pass rate(%)**로, threshold를 바꿔가며 측정한 곡선
- threshold는 **weak 사용 비율과 전체 성능 사이의 trade-off를 확인하기 위한 기준**으로 사용
  - threshold가 높을수록 weak로 많이 보내고(=strong 호출 비율 낮음), 낮을수록 strong으로 많이 보냄
  - 곡선을 따라 오른쪽(strong 비율 ↑)으로 갈수록 pass rate가 full strong(81.62%)에 가까워짐
- 곡선이 **왼쪽(weak 비율이 높은 구간)에서도 높은 pass rate를 유지**할수록, 성능 손실 없이 weak를 더 많이 쓸 수 있다는 의미
- 좋은 router일수록 동일한 strong 호출 비율에서 **Random 곡선보다 위**에 위치

```
[Graph 삽입 위치]
routing_curves.png — Pair A / Pair B 각각에 대해
x축: Strong Model Calls (%), y축: Pass Rate (%)
곡선: MF / UniRoute (+ Random), 기준선: weak-only / strong-only(81.62%)
```

### 4.2 주요 결과

- 곡선 전반에서 **MF와 UniRoute는 Random 대비 위쪽**에 위치 → 사전 router가 무작위보다 의미 있는 routing 신호를 학습함을 확인
- 다만 **full strong 대비 1% 이내(≥ 80.62%)**라는 엄격한 조건에서 weak로 보낼 수 있는 비율은 아래와 같이 제한적

| 모델 쌍 | Random | MF router | UniRoute |
|---|---|---|---|
| Pair A (0.8B vs 9B) | ~0% | **약 20%** | 약 19% |
| Pair B (2B vs 9B) | ~10% | 약 10% | **약 39%** |

- **Pair A**: MF(약 20%, weak 20.03% 지점 pass 80.66%)와 UniRoute(약 19%, weak 19.34% 지점 pass 81.48%)가 비슷하게 높았고, Random은 사실상 0%
- **Pair B**: UniRoute가 **약 39%**까지 사용 가능(weak 38.82% 지점 pass 80.8%)으로 가장 좋았고, MF는 약 10%(weak 10.01% 지점 pass 80.8%)로 Random(약 10%)과 큰 차이가 없었음
- 종합하면, 1% 이내 조건에서 weak 사용 비율은 방법·모델 쌍에 따라 **약 10–39%로 편차가 큼**
  - Pair A는 MF·UniRoute 모두 ~20%로 Random(0%)을 분명히 상회
  - Pair B는 **UniRoute가 ~39%로 가장 우수**, 반면 **MF는 ~10%로 Random 수준**에 머물러 이 조합에서는 효과가 제한적
  - 즉 **한 방법이 일관되게 우월하지 않고**, router 구조·모델 쌍에 민감

### 4.3 결과 분석

- weak 사용 비율이 (특히 일부 조합에서) 더 높지 못한 데에는 다음 요인이 작용했을 가능성이 있음 (단정이 아닌 추정)

- **데이터 특성: function calling 중심의 난이도**
  - BFCL은 단순 질의응답이 아니라 **function calling / tool calling** 중심이라, weak model이 안정적으로 처리하기 어려운 경향
  - **tool selection + argument extraction + value grounding**이 모두 정확해야 pass로 인정 → weak model의 작은 오류 하나도 전체 성능 하락으로 직결
  - 즉 "weak가 안전하게 처리 가능한 쉬운 sample"의 비중 자체가 일반 QA보다 작을 수 있음

- **데이터 수 부족: safe subset 학습의 한계**
  - BFCL 데이터 수가 충분하지 않아, router가 **weak model이 성공 가능한 safe subset을 충분히 학습**하지 못했을 가능성
  - 특히 소형 카테고리의 sample 수가 적어, weak가 잘 처리하는 영역과 그렇지 않은 영역의 경계를 세밀하게 구분하기 어려웠을 수 있음

- **방법론 간 차이 / 일관성 부족**
  - Pair A는 MF·UniRoute가 비슷했으나, Pair B는 **UniRoute(~39%)가 크게 우수**하고 **MF(~10%)는 Random 수준**에 머묾 → **한 방법이 일관되게 우월하다고 보기 어려움**
  - 동일 데이터에서도 **router 구조와 모델 쌍에 따라 weak safe subset을 구분하는 정도가 달라짐**을 시사

- **해석 범위**
  - 본 실험은 사전 router 기반 접근의 **초안 test 성격**
  - 위 수치를 최종 성능으로 해석하기보다는, **추가 실험(데이터 보강, router 구조/세팅 비교)의 필요성을 확인한 결과**로 정리하는 것이 적절

---

## 5. 추가 진행 계획

### 5.1 NVIDIA 공개 proxy-based router 확인

- `NVIDIA-AI-Blueprints/llm-router` 구조 확인
- 해당 router가 **prompt classification 기반으로 LLM을 선택하는 proxy 구조**인지 확인
- 현재 on-device/cloud routing task에 **바로 적용 가능한지**, 또는 **custom policy 학습이 필요한지** 검토

### 5.2 BFCL 데이터 augmentation 후 router 성능 재확인

- 현재 BFCL 데이터 수가 충분하지 않아 router 학습에 한계가 있을 수 있음
- BFCL 데이터를 **augmentation하여 학습 데이터를 늘린 뒤** MF router 성능을 다시 확인
- 특히 **weak model이 안정적으로 처리 가능한 safe subset**을 더 잘 구분할 수 있는지 확인
