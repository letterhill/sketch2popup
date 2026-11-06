# sketch2popup

종이에 그린 캐릭터 사진에서 배경을 제거하고(분할), 두께를 줘서(2.5D) 움직이는 애니메이션으로 만드는 컴퓨터비전 팀 프로젝트.

**연구 질문**: 조명 보정과 분할 후처리가 캐릭터 추출 정확도를 얼마나 높이는가?

## 빠른 실행 (Colab)
`notebooks/00_colab_setup.ipynb`를 Colab에서 열고 위에서부터 실행.

## 로컬 실행
```bash
pip install -e .
python scripts/run_experiment.py --config configs/exp_D_full.yaml --split dev
python scripts/make_demo.py --image photo.jpg --out demo.gif
```

## 실험 설정
| 실험 | 조명 보정 | 후처리 |
| --- | --- | --- |
| A | ✗ | ✗ |
| B | ✓ | ✗ |
| C | ✗ | ✓ |
| D | ✓ | ✓ |

## 결과
(4주차에 표와 데모 GIF 추가)

## 팀 규칙
- `main`에 직접 push 금지 → `feat/<이름>-<기능>` 브랜치에서 PR, 1명 리뷰 후 squash merge
- 로직은 `src/sketch2popup/`, 노트북은 호출·시각화만. 노트북은 담당자 1명
- 커밋 전 `nbstripout --install` (한 번만) 으로 노트북 출력 자동 제거
- 데이터·가중치·큰 영상은 커밋 금지 → Google Drive `MyDrive/sketch2popup/`

## 데이터
Meta Amateur Drawings Dataset — `data/README.md` 참고.
