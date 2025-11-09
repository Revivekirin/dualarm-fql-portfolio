# Scalable RL Portfolio — Data-Driven Viewer (+ Video)

Flow Q-Learning(FQL), QC-FQL, 그리고 **QC-FQL(+EMA)** 실험 산출물을 **즉시 시각화**하기 위한 React 포트폴리오 뷰어입니다.
아래 3가지 결과물을 드래그&드롭 또는 URL로 불러오면 대시보드에서 한 번에 확인할 수 있습니다.

1. `learning_curves.csv` → **학습 곡선** (reward / success_rate / distil_loss / bc_flow_loss / q_loss / critic_loss)
2. `vector_field_bcflow.json` → **BC-flow (teacher) 벡터필드** 시각화 (여러 t-slice 애니메이션)
3. `embedding_student_teacher.json` → **Teacher vs Student 행동 임베딩** 2D 산점도
4. (선택) **평가/데모 영상**(mp4/webm) → 성능 질적 비교

>
> * - 📄 [QC-FQL Research Report (DOCX)](assets/QC-FQL_Research_Report.pdf)
> * - 🖼️ [Architecture Diagram (PNG)](assets/architecture.png)  

> **Google Drive Shared Dataset**  
> The trained robot datasets (including exported learning curves, BC-flow vector fields, and teacher–student embeddings)  
> can be accessed directly from the shared Drive folder below.  
> You can download or use the “URL Load” option in the portfolio viewer to visualize these files in the app.  
>
> 🔗 [Robot Training Dataset – Google Drive Folder](https://drive.google.com/drive/folders/1TvNr2LxKEUckLGo9Tnn9-xB8bMza-Jdm?usp=drive_link)

---

## Visualization Tool for Flow matching

* **학습 곡선(learning_curves)**: optimization 안정성(critic/actor loss), 성능(reward/success) 변화를 시간축으로 추적
* **BC-flow 벡터필드(vector_field_bcflow)**: teacher flow가 목표 행동으로 **연속 경로를 유도하는 패턴**을 직관적으로 확인
* **Teacher-Student 임베딩(embedding_student_teacher)**: 동일 관측에서 **student가 teacher 분포로 수렴**하는지 한눈에 비교
* **영상(rollouts)**: 수치만으로는 부족한 **질적 동작의 연속성/안정성**을 시각적으로 검증

특히 **QC-FQL(+EMA)** 평가에서, **chunk size가 커질수록** 타깃 분산이 증가하는 문제를 **EMA 타깃 액터**가 얼마나 완화하는지,
(1) `critic_loss` 변동폭, (2) `success_rate` 상승 추세, (3) 임베딩 정렬 정도로 확인할 수 있습니다.

---

## 데모 스크린샷(예시)

* ![Architecture](assets/FQL_archi.png)
  *QC-FQL(+EMA) 개념도: Flow teacher + Student actor, Chunked critic, EMA target actor/critic 구성*

---

## 빠른 시작 (Vite + React + Tailwind)

> Node.js 18+ 권장

1. 의존성 설치

```bash
npm i
npm i -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

2. 개발 서버 실행

```bash
npm run dev
```

브라우저에서 `http://localhost:5173` 로 접속하세요.

---

## 사용 방법

### 1) 드래그 & 드롭

* 페이지 상단의 **드롭 존**에 다음 파일들을 끌어다 놓습니다.

  * `learning_curves.csv`
  * `vector_field_bcflow.json`
  * `embedding_student_teacher.json`
  * (선택) `*.mp4`, `*.webm` 동영상 파일(복수 개 가능)

### 2) URL 로딩

* GitHub raw 같은 **직접 URL**을 입력해 불러올 수 있습니다.
* 또는 **Base URL**을 입력한 뒤, 준비된 **Quick-pick** 버튼으로 기본 경로를 호출:

  * `portfolio_logs/learning_curves.csv`
  * `portfolio_logs/vector_field_bcflow.json`
  * `portfolio_logs/embedding_student_teacher.json`

---

## 파일 포맷(스키마)

### A) `learning_curves.csv` 

```csv
step,reward,success_rate,distil_loss,bc_flow_loss,q_loss,critic_loss,mse
10000,0.23,0.02,1.24,0.31,0.75,0.42,0.59
15000,0.28,0.05,1.10,0.29,0.68,0.39,0.55
...
```

* **필수**: `step`(정수)
* **권장**: `reward`, `success_rate`, `distil_loss`, `bc_flow_loss`, `q_loss`, `critic_loss`, `mse`
  (열 이름은 컴포넌트에서 그대로 참조하므로 가급적 동일 키 사용)

### B) `vector_field_bcflow.json`

```json
{
  "vector_field": [
    {
      "t": 0.25,
      "points": [[-2.0,-2.0],[-2.0,-1.8], ...],
      "vectors": [[0.2,0.1],[0.18,0.11], ...]
    },
    {
      "t": 0.5,
      "points": [...],
      "vectors": [...]
    }
  ]
}
```

* `points[i]`와 `vectors[i]`는 동일 길이여야 합니다.
* 각 화살표는 해당 `(x,y)`에서 teacher flow의 **예측 속도(∂x/∂t)** 를 의미합니다.

### C) `embedding_student_teacher.json` 

```json
{
  "teacher": [[-0.42, 0.18], [0.11, -0.07], ...],
  "student": [[-0.39, 0.20], [0.09, -0.05], ...]
}
```

* 동일한 관측 배치에서 샘플한 **teacher/student 행동 임베딩**(2D) 비교
* student가 teacher 분포로 **정렬/수렴**하는지 시각적으로 점검

---

## TroubleShooting

* **그래프가 비어있어요**: CSV 헤더 키 이름과 스펠링을 확인하세요. `step`이 정수로 파싱되는지 확인
* **벡터필드가 안 나와요**: `points.length === vectors.length` 확인, 값이 숫자인지 확인
* **임베딩이 안 나와요**: `teacher`/`student` 둘 다 2D 배열인지 확인
* **CORS 오류**: 외부 URL 로딩 시 서버가 CORS를 허용해야 합니다. GitHub는 `raw.githubusercontent.com` 사용
* **패키지 아이콘/차트 미표시**: `lucide-react`, `recharts`, `framer-motion`, `papaparse` 설치 여부 확인

---

## 라이선스 & 인용

* 코드: MIT
* 인용:

  * Park et al., 2024. *Flow Q-Learning: Wasserstein-Regularized Offline-to-Online RL*.
  * Li et al., 2025. *Reinforcement Learning with Action Chunking*. arXiv:2507.07969.

---

## 실행 명령 요약

```bash
npm i
npm i -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

npm run dev
```

