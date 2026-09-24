"""Generate numerical comparison tables without choosing a post-result winner."""
from common import *
from run_inference import BOOT


def table(df, digits=3):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return ('%.*f'%(digits,x)) if np.isfinite(x) else '—'
        return str(x)
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |',
        '| '+' | '.join(['---']*len(df.columns))+' |']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])


def ranges(df, keys, columns):
    rows=[]
    for key,g in df.groupby(keys[0] if len(keys)==1 else keys,sort=False):
        key=key if isinstance(key,tuple) else (key,)
        row=dict(zip(keys,key))
        for col,scale in columns:
            row[col]='—' if g[col].isna().all() else '%.3f ~ %.3f'%(scale*g[col].min(),scale*g[col].max())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    text=['# A5 수치 비교표','범위는 다섯 시간 정렬 시나리오의 최솟값~최댓값이며 신뢰구간이 아니다. UTC 두 시나리오는 core 자료가 같아 독립 반복이 아니다.']
    scores=pd.read_csv(SOURCE/'scores.csv');scores=scores.loc[scores.model.isin(PRIMARY)]
    for period in ['Jan_Mar','Aug_Mar']:
        x=scores.loc[scores.period==period]
        text+=['## 동시점 USDT 설명오차/잔차 RMS: '+period,
               table(ranges(x,['model'],[('rmse_bp',1),('reference_reconstruction_mse',1)]),6),
               'USDT 잔차 RMS 단위는 bp. 기준 5종 재구성 MSE는 표준화한 코인 프리미엄의 무차원 오차다. 서로 다른 지표다. Aug–Mar는 과거 검증 구간을 포함한 안정성 진단이다.']
    probes=pd.read_csv(OUT/'probe_zero_baseline_audit.csv');probes=probes.loc[(probes.period=='all')&probes.method.isin(PRIMARY)]
    text+=['## 잔차 설명 probe: MSE 개선율(%)',table(ranges(probes,['method','scope','algorithm'],[('skill_vs_zero',100),('skill_vs_past_mean',100)])),
           '양수는 해당 기준보다 오차 감소, 음수는 악화다. 과거 평균 이동 때문에 두 기준이 달라지며 AE의 큰 과거 평균 대비 수치를 공통 요인 잔여 설명력으로 해석하지 않는다.']
    forecast=pd.read_csv(OUT/'forecast_scores.csv');forecast=forecast.loc[(forecast.period=='all')&forecast.method.isin(['baseline']+PRIMARY)]
    for task in ['observed_change','own_residual']:
        for q in [.1,.5]:
            f=forecast.loc[(forecast.task==task)&(forecast.q==q)]
            text+=['## '+task+' / q='+str(q),table(ranges(f,['method','info','algorithm'],[('skill',100),('below_rate',100)])),
                   'skill(%)는 해당 과거 분위수 기준 대비 pinball loss 감소율. below_rate(%)는 실제값이 예측 분위수보다 낮은 비율이다. own_residual은 목표가 달라 raw loss로 분해 모형의 순위를 매기지 않는다.']
    intervals=pd.read_csv(OUT/'forecast_intervals.csv')
    f=intervals.loc[(intervals.task=='observed_change')&(intervals.q==.1)&(intervals.block_days==7)&
                    intervals.method.isin(PRIMARY)&intervals.contrast.isin(['add_residual','AE_minus_pca_2'])]
    text+=['## 동일 관측 목표 q10의 paired 손실 차이',table(f[['scenario','method','info','algorithm','contrast','relative_pct','ci_low','ci_high']],5),
           'relative_pct는 손실 증감률, CI는 bp 단위 pinball loss 차이다. 음수는 앞의 방법이 개선. 고정 예측 2,000회 7일 블록 구간이며 재학습·선택 불확실성 전체를 포함하지 않는다.']
    ci=pd.read_csv(BOOT/'coefficient_intervals.csv')
    x=ci.loc[(ci.scenario=='clock_UTC_quote')&(ci['sample']=='all')&(ci.h==6)&ci.refit_first_stage&
             ci.method.isin(PRIMARY)&ci.term.isin(['downside','account_ls','e_origin'])]
    text+=['## 원래 연구 M2: Jan–Mar, h=6, UTC 기준',table(x[['method','q','term','n','effect','ci_low','ci_high']]),
           '효과 단위: 원래 평가표본에서 설명변수 1표준편차당 bp. 구간: 7일 달력 블록, 199회, PCA/AE 인코더부터 재학습. AE 월별 구조/규제 선택은 고정했다.']
    x=ci.loc[(ci['sample']=='all')&(ci.h==6)&(ci.q==.1)&ci.refit_first_stage&
             ci.method.isin(PRIMARY)&ci.term.isin(['downside','account_ls'])]
    text+=['## 시각 가정별 q10 계수',table(x[['scenario','method','term','effect','ci_low','ci_high']])]
    contrast=pd.read_csv(BOOT/'contrast_intervals.csv')
    x=contrast.loc[contrast.refit_first_stage&contrast.method.isin(PRIMARY)&contrast.term.isin(['downside','account_ls'])]
    text+=['## 하방 분위수 특이성·시차별 차이',table(x[['scenario','method','term','contrast','effect','ci_low','ci_high']]),
           'q10−q50의 downside가 안정적으로 음수여야 중앙부보다 하방에서 더 큰 음의 관련성이라는 해석이 가능하다. h6−h1/h12−h1은 같은 origin 표본 비교다. 유의하지 않다는 결과는 효과가 정확히 0임을 입증하지 않는다.']
    (HERE/'TABLES_KO.md').write_text('\n\n'.join(text)+'\n')


if __name__=='__main__':main()
