"""Render quantitative A4 tables without selecting models from evaluation results."""
from pathlib import Path
import json
import sys
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
STABLE='--stabilized' in sys.argv
if STABLE: OUT=HERE/'stabilized'
LABELS={'sequential_ols':'EQ5 순차 OLS','pca_1':'PCA 1요인','pca_2':'PCA 2요인',
 'conditional_linear':'조건부 선형 민감도','ae_relu_1':'ReLU AE 1요인','ae_relu_2':'ReLU AE 2요인',
 'ae_tanh_1':'tanh AE 1요인','ae_tanh_2':'tanh AE 2요인','conditional_neural':'조건부 신경망 민감도',
 'pca_selected':'과거 선택 PCA','ae_selected':'과거 USDT 오차로 선택한 AE',
 'ae_reconstruction_selected':'과거 참조 재구성으로 선택한 AE'}
CLOCKS={'clock_UTC_quote':'UTC/UTC','clock_NY_quote':'NY/NY','clock_FXUTC_USNY_quote':'UTC/NY',
 'clock_FXKST_USNY_quote':'KST/NY','clock_NY_macro_end1h_quote':'NY/NY+1h'}


def main():
 manifest=json.loads((OUT/'RUN_MANIFEST.json').read_text())
 scores=pd.read_csv(OUT/'scores.csv');primary=scores.loc[scores.period=='Jan_Mar']
 intervals=pd.read_csv(OUT/'paired_block_intervals.csv');intervals=intervals.loc[intervals.block_days==7]
 lines=['# A4 전체 수치표','',
  '학습 이후 동시점 자료를 이용한 조건부 적합이다. 미래 가격 예측이 아니다. 2026년 1~3월(원자료 종료일까지)의 같은 관측치를 비교한다. 음수는 RMSE 감소, 양수는 증가다. 시각 가정 5개 중 UTC/UTC와 UTC/NY의 분해 입력은 같으므로 독립 재현 횟수로 해석하지 않는다.','',
  '## 1. 같은 요인 수·입력의 대조군과 비교','',
  '| 방법 | 적절한 선형 대조군 | RMSE 변화율 범위 | 감소 조건 수 | 7일 구간 전체가 음수인 조건 수 |',
  '|---|---|---:|---:|---:|']
 comparisons=[('ae_relu_1','pca_1'),('ae_relu_2','pca_2'),('ae_tanh_1','pca_1'),('ae_tanh_2','pca_2'),
  ('conditional_neural','conditional_linear'),('ae_selected','pca_selected'),('ae_reconstruction_selected','pca_selected')]
 for method,base in comparisons:
  d=intervals.loc[(intervals.model==method)&(intervals.baseline==base)]
  lines.append('| %s | %s | %+.2f~%+.2f%% | %d/5 | %d/5 |'%(LABELS[method],LABELS[base],
   d.relative_rmse_pct.min(),d.relative_rmse_pct.max(),(d.delta_rmse_bp<0).sum(),(d.rmse_ci_high<0).sum()))
 lines += ['', '구간은 고정된 예측의 짝지은 7일 달력 블록 2,000회 재표집 구간이다. 모형 재학습·선택·반복 탐색·다중 비교 전체를 반영하지 않는다. 3일 구간도 CSV에 보존했다.','',
  '## 2. 기존 EQ5 OLS 대비','', '| 방법 | RMSE 변화율 범위 |','|---|---:|']
 for method in LABELS:
  if method=='sequential_ols':continue
  d=intervals.loc[(intervals.model==method)&(intervals.baseline=='sequential_ols')]
  lines.append('| %s | %+.2f~%+.2f%% |'%(LABELS[method],d.relative_rmse_pct.min(),d.relative_rmse_pct.max()))
 lines += ['', '## 3. 주 평가 오차의 절대 크기 (bp)','',
  '| 시각 가정 | n | EQ5 OLS | PCA1 | PCA2 | 선택 PCA | 선택 AE | 조건부 선형 | 조건부 NN |',
  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for scenario in manifest['scenarios']:
  d=primary.loc[primary.scenario==scenario].set_index('model')
  vals=[d.loc[m,'rmse_bp'] for m in ['sequential_ols','pca_1','pca_2','pca_selected','ae_selected','conditional_linear','conditional_neural']]
  lines.append('| %s | %d | %s |'%(CLOCKS[scenario],d.loc['pca_1','n'],' | '.join('%.4f'%v for v in vals)))
 lines += ['', '## 4. 선택 절차의 월별 오차','',
  '| 시각 가정 | 월 | AE 선택 / 선택 PCA 변화율 | 조건부 NN / 조건부 선형 변화율 |',
  '|---|---|---:|---:|']
 for scenario in manifest['scenarios']:
  for month in ['2026-01','2026-02','2026-03']:
   d=scores.loc[(scores.scenario==scenario)&(scores.period==month)].set_index('model')
   lines.append('| %s | %s | %+.2f%% | %+.2f%% |'%(CLOCKS[scenario],month,
    100*(d.loc['ae_selected','rmse_bp']/d.loc['pca_selected','rmse_bp']-1),
    100*(d.loc['conditional_neural','rmse_bp']/d.loc['conditional_linear','rmse_bp']-1)))
 lines += ['', '## 5. 기존 질문에 대한 동일 후속 분석','',
  '2025년 8월~2026년 3월, 기본 통제 M2·6시간·q10, 설명변수 1표준편차당 bp의 시각 가정별 범위. 아래는 **점추정**이다. 새 잠재요인 모델의 생성 잔차 재적합 신뢰구간을 계산한 것이 아니므로 부호만으로 유의성·인과성을 주장하지 않는다.','',
  '| 방법 | BTC 하방 지표 | 롱/숏 비율 |','|---|---:|---:|']
 coefs=pd.read_csv(OUT/'coefficients.csv')
 d=coefs.loc[(coefs.scope=='core')&(coefs['sample']=='all')&(coefs.model=='M2')&(coefs.h==6)&(coefs.q==.1)]
 for method in LABELS:
  parts=[]
  for term in ['downside','account_ls']:
   a=d.loc[(d.method==method)&(d.term==term),'effect_bp_per_sd']
   parts.append('%+.3f~%+.3f'%(a.min(),a.max()))
  lines.append('| %s | %s |'%(LABELS[method],' | '.join(parts)))
 lines += ['', '기본/거시 통제·q10/q50/q90·1/6/12시간·공통 원점 비교는 [coefficients.csv](results/coefficients.csv), 전후반·상위 스트레스 날짜 제외는 [subperiod_and_stress.csv](results/subperiod_and_stress.csv)에 있다. 미래 잔차 예측모형의 새 평가를 실행한 것은 아니다.','',
  '## 6. 학습 진단','', '| 모형군 | 후보×월×시각 가정 적합 수 | 경고가 있는 적합 수 |','|---|---:|---:|']
 fits=[x for p in OUT.glob('*/*_fits.json') for x in json.loads(p.read_text())]
 for family in sorted({f['candidate']['family'] for f in fits}):
  rows=[f for f in fits if f['candidate']['family']==family]
  lines.append('| %s | %d | %d |'%(LABELS[family],len(rows),sum(bool(f['warnings']) for f in rows)))
 ae_members=[m for f in fits if f['candidate']['family'].startswith('ae_') for m in f['members']]
 dead=sum(any(v<1e-8 for v in m['latent_sd']) for m in ae_members)
 lines += ['', 'AE seed 단위 적합 %d개 중 적어도 하나의 잠재 차원 표준편차가 1e-8 미만인 경우: %d개. 실패하거나 불리한 seed를 결과 확인 후 버리지 않았다. 월별 선택과 학습 경고는 각 시각 가정 폴더에 저장했다.'%(len(ae_members),dead),'',
  '공통 요인을 참조 코인만으로 추출했으며 USDT 자체는 인코더에 입력하지 않았다. 과거 검증으로 USDT 설명력을 높이는 요인을 선택하는 절차와 참조 코인 재구성만으로 선택하는 절차를 구분했다.','',
  '시간대 미확정, USDC/USD=1 가정, 이미 여러 번 분석한 짧은 표본이라는 제약이 유지된다. 오차 감소는 경제적으로 참된 공통 성분·USDT 고유 성분의 식별을 뜻하지 않는다.','',
  '[실행 규칙 및 문헌 적용 범위](PROTOCOL_KO.md) · [실행 해시](results/RUN_MANIFEST.json) · [독립 출력 검증](VERIFICATION.json)']
 content='\n'.join(lines)+'\n'
 if STABLE:
  content=content.replace('# A4 전체 수치표','# A4 안정화 수치표\n\n인코더는 고정하고 PCA/AE의 요인→USDT 회귀에 동일한 Ridge alpha=1을 적용했다. 아래 PCA도 Ridge 회귀를 쓴다. [안정화 규칙](STABILITY_KO.md).')
  content=content.replace('results/','stabilized/').replace('(VERIFICATION.json)','(STABILITY_VERIFICATION.json)')
  content=content.replace('AE seed 단위 적합','원 실행에서 재사용한 AE seed 단위 적합')
 else:
  content=content.replace('# A4 전체 수치표','# A4 초기 수치표 (보존)\n\n일부 AE의 거의 같은 잠재요인으로 인해 요인 회귀가 불안정했다. 최종 해석에는 [동일 Ridge로 안정화한 비교](STABILIZED_TABLES_KO.md)를 사용한다. 초기 실패 후보를 포함한 이 표는 보존 기록이다.')
 (HERE/('STABILIZED_TABLES_KO.md' if STABLE else 'TABLES_KO.md')).write_text(content)
 print('Generated stabilized tables' if STABLE else 'Generated original tables')

if __name__=='__main__':main()
