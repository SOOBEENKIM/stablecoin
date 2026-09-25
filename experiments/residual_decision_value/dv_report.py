"""Post-score presentation only; does not alter forecasts, cohorts or tests."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dv_core import HERE,OUT,DEFS


def table(d,cols):
    rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,row in d.iterrows():
        values=[]
        for c in cols:
            v=row[c]
            values.append(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v))
        rows.append('| '+' | '.join(values)+' |')
    return '\n'.join(rows)


def save_figure(fig,name):
    for ext in ['png','svg']:
        path=OUT/(name+'.'+ext)
        fig.savefig(path,dpi=200)
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def main():
    primary=pd.read_csv(OUT/'primary_tests.csv')
    metrics=pd.read_csv(OUT/'decision_metrics.csv')
    select=(metrics.variant=='original')&(metrics.information=='F')&(metrics.calibration=='calibrated')&(metrics['sample']=='full')&(metrics.threshold_bp==-10)
    st=pd.read_csv(OUT/'state_contributions.csv')
    paths=pd.read_csv(OUT/'path_summary.csv')
    disagreements=pd.read_csv(OUT/'disagreement_contrasts.csv')
    monthly=pd.read_csv(OUT/'monthly_metrics.csv')
    lines=['# 12시간 잔차 위험의 상태·경보·가격 경로 결과','',
        '[해석과 기여 판단](CONCLUSION_KO.md) · [실행 전 설계](PROTOCOL_KO.md) · [선행연구](REFERENCES_KO.md)','',
        '기존 자료를 재사용한 탐색적 분석이다. 아래 개선율은 미탐 비용을 오경보의9배로 둔 단위 없는 경보 비용의 감소다. 실제 투자손실이나 수익 개선율이 아니다. 주 문턱은 잔차의 실제12시간 **추가** 하락10bp다. 네 seed의 시점별 비용 평균을 사용하며509개 시점을2,036개로 세지 않는다.','',
        '## 주12개 비교','',table(primary,['definition','reference','n','reference_cost','ml_cost','improvement_pct','improvement_lo','improvement_hi','p_holm_12']),
        '','## 경보 부담과 미탐','',table(metrics[select],['definition','strategy','n','events','alarms','false_positives','false_negatives','recall','precision','cost']),
        '','ML 빈도는 seed 평균이므로 소수가 될 수 있다. 실제 한 실행의 빈도·민감도는 [seed 표](results/seed_sensitivity.csv)에 있다.',
        '','## 정보 추가의 경보 가치','',table(pd.read_csv(OUT/'information_value.csv'),['definition','comparison','improvement_pct','improvement_lo','improvement_hi']),
        '','## 주 EQ 상태별 손실 개선의 기여','',
        table(st[(st.definition=='EQ')&(st.variant=='original')&(st.information=='F')&st.reference.isin(['linear','threshold'])],['reference','state','n','share','difference_bp','difference_lo','difference_hi','contribution_bp','too_high_contribution_bp','too_low_contribution_bp']),
        '','상태0/1은 학습 잔차 하위1/3,2/3은 중간,4/5는 상위1/3이며 짝수/홀수는 학습 RMS 중앙값 미만/이상이다. contribution은 전체 표본 비중을 반영해 가산되며, 특정 상태만 선택한 새 성과가 아니다.',
        '','## 실제 가격 경로: 주 EQ, 선형과 다른 경보','',
        table(paths[(paths.definition=='EQ')&(paths.reference=='linear')&(paths.cohort=='full12')&paths.outcome.isin(['delta_e','local_logreturn_bp','delta_local_bp','delta_fx_bp','delta_market_bp','fall10'])],['group','outcome','n','days','status','mean','ci_low','ci_high']),
        '','경로 집단은 첫 seed의 고정 경보다. 아래 집단 간 차이는 인과효과가 아니다.',
        '',table(disagreements[disagreements.cohort=='full12'],['definition','reference','outcome','ml_only_n','reference_only_n','difference','ci_low','ci_high','bootstrap_valid_fraction']),
        '','## 동일 시작점 자료 가용성','',table(pd.read_csv(OUT/'path_availability.csv'),['definition','origin_n','h1_n','h6_n','h12_n','common_n']),
        '','## 전체 결과 파일','',
        '- [월별 비용·경보](results/monthly_metrics.csv), [모든 문턱/보정/중첩 제거 비교](results/comparisons.csv), [state 지표](results/state_metrics.csv)',
        '- [Murphy 문턱별 비용](results/murphy_curve.csv), [seed 민감도](results/seed_sensitivity.csv), [경로 seed 민감도](results/path_seed_sensitivity.csv)',
        '- [가격 구성요소 전체 표](results/path_summary.csv), [구현 검증](results/VERIFICATION.json), [현재 잔차 연결 검증](results/ALIGNMENT_VERIFIED.json)',
        '','![문턱별 경보 비용](results/decision_thresholds.png)','',
        '![상태별 손실 기여](results/state_error_contributions.png)','']
    (HERE/'RESULTS_KO.md').write_text('\n'.join(lines))
    curve=pd.read_csv(OUT/'murphy_curve.csv')
    labels={'ml_selected':'ML selection','linear':'Linear','threshold':'Threshold','state_hist':'State empirical','historical':'Unconditional'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
    for ax,definition in zip(axes,DEFS):
        for name,label in labels.items():
            d=curve[(curve.definition==definition)&(curve.strategy==name)].sort_values('threshold_bp',ascending=False)
            ax.plot(-d.threshold_bp,d.cost,label=label,lw=1.6 if name=='ml_selected' else 1.1)
        ax.axvline(10,color='gray',ls=':',lw=.8);ax.set_title(definition);ax.set_xlabel('Additional residual fall threshold (bp)')
    axes[0].set_ylabel('Alarm cost (miss : false alarm = 9 : 1)');axes[-1].legend(fontsize=7)
    save_figure(fig,'decision_thresholds')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,ref in zip(axes,['linear','threshold']):
        d=st[(st.definition=='EQ')&(st.variant=='original')&(st.information=='F')&(st.reference==ref)].sort_values('state')
        ax.bar(d.state-.18,d.too_high_contribution_bp,width=.36,label='Boundary too high')
        ax.bar(d.state+.18,d.too_low_contribution_bp,width=.36,label='Boundary too low')
        ax.axhline(0,color='black',lw=.8);ax.set_title('ML advantage vs '+ref)
        ax.set_xticks(range(6));ax.set_xticklabels(['Low e\nlow vol','Low e\nhigh vol','Mid e\nlow vol','Mid e\nhigh vol','High e\nlow vol','High e\nhigh vol'],fontsize=8)
    axes[0].set_ylabel('Contribution to total pinball improvement (bp)');axes[1].legend(fontsize=8)
    save_figure(fig,'state_error_contributions')
    plt.close(fig)
    print('Rendered fixed result tables and scientific figures.')


if __name__=='__main__':main()
