## 프로젝트 설명서
이 프로젝트는 모델 개발하기 전 기본적인 보일러 코드들을 작성 해놓은 프로젝트 입니다.

```text
AI_Project/
│
├── configs/
│   └── config.py
│
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── datasets/
│   ├── dataset.py
│   ├── dataloader.py
│   ├── preprocessing.py
│   └── transforms.py
│
│── docs/
│   ├── gpt_result.md
│   └── claude_result.md
│
├── models/
│   ├── model.py
│   ├── unet.py
│   └── resnet.py
│
├── losses/
│   └── losses.py
│
├── utils/
│   ├── metrics.py
│   ├── logger.py
│   ├── checkpoint.py
│   └── visualize.py
│
├── notebooks/
│   ├── 01_train.ipynb
│   ├── 02_test.ipynb
│   ├── 03_inference.ipynb
│   └── 04_visualization.ipynb
│
├── checkpoints/
├── outputs/
│   ├── logs/
│   ├── figures/
│   └── predictions/
│
├── requirements.txt
├── AGENT.md
└── README.md
```
기본적으로 위와 같은 형태의 구조를 띄며 내용을 수정해서 사용하면 됩니다.

각 폴더별 설명은 다음과 같습니다.
- data : 원본 데이터 저장
- dataset : dataset, dataLoader, Optimizer와 같은 학습 전 전처리, 학습 데이터 준비를 담당합니다.
- docs : GPT나 Claude의 결과물을 저장합니다.
- losses : loss 함수를 저장합니다.
- utils : 로거나 시각화 함수, 체크포인트 함수 등 모델 학습의 유틸리티와 관련된 코드입니다.
- notebooks : 모델 학습을 위한 주피터 노트북들을 저장합니다.
- checkpoints : 체크포인트 활용 시 모델을 저장합니다
- outputs : 모델의 결과, 학습 경과 이미지, 예측 이미지 등을 저장합니다.
- AGENT.md : CLI 모델 활용 시 룰을 정하는 파일입니다.