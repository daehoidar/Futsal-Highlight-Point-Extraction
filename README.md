
# 풋살 하이라이트 포인트 추출 (Futsal-Highlight-Point-Extraction)
<br/>

이 저장소는 풋살 영상에서 개인별 하이라이트 포인트를 학습하고 추론하기 위한 파이프라인을 제공합니다. 전체 과정은 크게 두 단계로 이루어집니다.

1. **이벤트 탐지 모델**  
   - 구조: ResNet-18(이미지 특징 추출) + LSTM(시계열 맥락 이해)  
   - 입력: 골·슛 장면 전후 5초를 5 FPS로 샘플링한 클립, `goal`, `shoot`, `none` 레이블  
   - 학습: 직접 라벨링한 풋살 데이터 사용 (train.py)  
   - 데모: SoccerNet 축구 데이터로 사전 학습된 모델 제공(`pretrained/`) <br/>
   <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbzPAWb%2FbtsNuC4y4Yk%2F56Arx0HdDQTNUapqFzFIL0%2Fimg.png" alt="train_img" width="700"/><br/>
   Acc: 학습 데이터에 대한 정확도<br/>
   ⚠️**SoccerNet 데이터의 Non Disclosure Agreement에 따라 pretrained 모델의 상업적인 사용을 금지합니다.**

2. **이벤트 주체 검출**  
   - 구성 요소:
     - 풋살 공 탐지 (YOLOv8)  
     - 사람 객체 탐지 (YOLOv8)  
     - 등번호 추출 (EasyOCR)  
     - 발 키포인트 추출 (YOLOv8-pose)  
   - 핵심 아이디어: 이벤트 발생 직전 3초(5 FPS) 동안 공과 발의 평균 거리가 가장 짧은 선수를 주체로 판단  
   - 결과: 각 이벤트 시점에 `time`, `event`, `player`(등번호) 정보를 포함한 타임라인 출력  

---

## 📁 저장소 구조
```
highlight_extraction/
├── detectors/              # 검출 모듈
│   ├── __init__.py         # 패키지 초기화
│   ├── ball_detection.py   # YOLOv8 공 탐지
│   ├── person_detection.py # YOLOv8 사람 탐지
│   ├── number_extraction.py# EasyOCR 등번호 추출
│   └── foot_keypoint_extraction.py # 포즈 기반 발 키포인트 추출
├── event_extraction/       # 이벤트 탐지 및 추론 스크립트
│   ├── __init__.py
│   ├── train.py            # ResNet18+LSTM 모델 학습
│   └── predict.py          # 전체 파이프라인 추론
├── pretrained/             # 사전 학습된 모델
│   └── event_model_soccernet.pt
├── utils.py                # 영상 샘플링 유틸리티
└── requirements.txt        # 의존성 목록
```

---

## ⚙️ 설치

1. 저장소 클론:
   ```bash
   git clone https://github.com/daehoidar/Futsal-Highlight-Point-Extraction.git
   cd Futsal-Highlight-Point-Extraction
   ```

2. 가상환경 생성 및 의존성 설치:
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # Linux/macOS
   venv\\Scripts\\activate.bat  # Windows
   pip install -r requirements.txt
   ```

---

## 🚀 모델 학습 (`train.py`)

직접 라벨링한 풋살 클립을 사용해 이벤트 탐지 모델을 학습합니다:

### 데이터 구조

학습 데이터는 다음과 같은 구조로 준비되어야 합니다:

```
/path/to/your/labeled/data/
├── clips/
│   ├── clip_0000.npy
│   ├── clip_0001.npy
│   ├── clip_0002.npy
│   └── ...
└── labels.csv
```

**clips/ 디렉토리:**
- 각 `.npy` 파일은 5초 클립을 5 FPS로 샘플링한 프레임들을 포함
- 배열 형태: `[T, H, W, C]` (T=프레임 수, H=높이, W=너비, C=채널)
- 예: 5초 × 5 FPS = 25 프레임, 224×224 RGB 이미지 → `[25, 224, 224, 3]`

**labels.csv 파일:**
- 필수 컬럼: `clip_name`, `label`
- `clip_name`: 클립 파일 이름 (확장자 포함 또는 제외 가능, 예: `clip_0000.npy` 또는 `clip_0000`)
- `label`: 이벤트 레이블 정수 값
  - `0`: none (이벤트 없음)
  - `1`: goal (골)
  - `2`: shoot (슛)

예시 `labels.csv`:
```csv
clip_name,label
clip_0000,0
clip_0001,1
clip_0002,2
clip_0003,0
```

### 학습 실행
```bash
python event_extraction/train.py \
  --train-new \
  --data-dir /path/to/your/labeled/data \
  --save-path models/my_event_model.pt \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-4 \
  --device cuda
```

- `--train-new`: 신규 학습 플래그
- `--data-dir`: 위 구조를 따르는 데이터 디렉토리 경로
- `--save-path`: 학습된 가중치 저장 경로
- `--epochs`, `--batch-size`, `--lr`, `--device`: 학습 설정

---

## 🎯 추론 (`predict.py`)

전체 파이프라인을 실행하여 하이라이트를 추출합니다:

```bash
python event_extraction/predict.py \
  --video-path /path/to/futsal.mp4 \
  --video-length 600           # 전체 영상 길이(초)
  --model-path models/my_event_model.pt  # 선택: 사용자 모델 경로
  --ball-model yolov8_ball.pt  # 선택: 공 탐지 모델
  --device cuda
```

- `--model-path` 미제공 시 `pretrained/event_model_soccernet.pt` 사용
- 결과 예시:
  ```json
  [
    {"time": 30, "event": 1, "player": "10"},
    {"time": 150, "event": 2, "player": "7"},
    ...
  ]
  ```
  `event` 코드: `1=goal`, `2=shoot`  

---



