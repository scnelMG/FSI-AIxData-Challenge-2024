# Local competition data

이 폴더는 실행 시에만 사용하는 대회 입력 파일의 위치입니다. 원본 데이터와 제출 산출물은 공개 저장소에 포함하지 않습니다.

## 필요한 파일

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

- `train.csv`: `Fraud_Type` 타깃을 포함한 학습 데이터
- `test.csv`: 최종 분류 예측용 데이터
- `sample_submission.csv`: `clf_submission.csv`의 행·컬럼 형식 기준

## 당시 대회 입력 프로필

원본 실행 기록 기준으로 `train.csv`는 120,000행·64열, `test.csv`는 120,000행·63열이었습니다. `Fraud_Type` 분포는 `m` 118,800행과 나머지 12개 클래스 각 100행으로 매우 불균형했습니다. `notebooks/final_modeling.ipynb`는 실행 시 이 프로필을 함께 출력해, 다른 버전의 데이터로 실행했을 때 결과를 혼동하지 않게 합니다.

## 출처와 공개 범위

- 출처: [FSI AIxData Challenge 2024](https://dacon.io/competitions/official/236297/overview/description)
- 대회 설명상 데이터는 합성 금융거래 데이터입니다.
- 데이터 재배포 조건을 보수적으로 적용하기 위해 CSV, 생성 데이터, 제출 ZIP은 `.gitignore`로 제외합니다.
- 노트북 실행 후에는 `outputs/`에 결과가 생성되며, 이 폴더 역시 Git에 포함되지 않습니다.
