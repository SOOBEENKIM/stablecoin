"""Render already-sealed results without model/threshold selection."""
from ac_core import *
import ac_overshoot as x
import ac_symmetric_null as sn


def table(d,cols):
    def cell(v):
        if pd.isna(v):return ''
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']+
        ['| '+' | '.join(cell(v) for v in row)+' |' for row in d[cols].itertuples(index=False,name=None)])+'\n'


def subset(d,definition='EQ',sample='full',threshold=5,block=5,learner='forest40',seed=20260926):
    a=(d.definition==definition)&(d['sample']==sample)&(d.minimum_gap_bp==threshold)&(d.block_days==block)&(d.learner==learner)&(d.seed==seed)
    if 'h' in d:a &= d.h==12
    return d[a]


def main():
    sn.check()
    for path in [OUT/'EVALUATION_COMPLETE.json',x.DEST/'COMPLETE.json',sn.DEST/'COMPLETE.json']:
        for name,digest in json.loads(path.read_text())['file_sha256'].items():assert sha(ROOT/name)==digest,name
    primary=read(OUT/'primary_tests.csv');groups=read(OUT/'group_channels.csv');orth=read(OUT/'orthogonal_contrasts.csv')
    extra=read(x.DEST/'primary_tests.csv');contrasts=read(x.DEST/'contrasts.csv');quality=read(OUT/'nuisance_quality.csv')
    eqg=subset(groups);eqo=subset(orth);eqa=subset(contrasts)
    raw=read(OUT/'absolute_gap_and_prices.csv');mapping=read(OUT/'residual_vs_price.csv')
    chunks=['# 부호별 가격 조정과 과도 반전: 전체 결과\n',
        '설계 `9288dfe`, 후속 해석 검증 `b11d137`. 같은 과거 자료의 탐색이며, 이번 결과를 확인한 뒤 추가한 검정은 아래에서 분리한다. 기존 예측성능 비교와 원고 재현은 변경하지 않았다.\n',
        '## 1. 원래 고정한 주18개 검정\n',
        '단위 bp. 조정 대비는 할증−할인의 부분선형 투영 계수다. 평균 행의 n은 전체 분석표본이고 group_n이 실제 해당 부호의 표본 수다. 개별95% 구간과 정의별 Holm6/전체 Holm18을 구분한다.\n',
        table(primary,['definition','test','n','group_n','estimate','lo','hi','p_holm6','p_holm18']),
        '## 2. 주 EQ 가격 구성요소\n',
        '방향 이동의 양수는 처음 괴리를 닫는 방향이다. 이것만으로 절대 괴리 감소나 특정 가격이 조정을 주도했다고 해석하지 않는다. 다섯 회계적 가격 기여의 합이 total이다.\n',
        table(eqg,['outcome','group','group_n','estimate','lo','hi']),
        '조건 통제 후 구성요소 차이(부호 대비):\n',table(eqo,['outcome','n','estimate','lo','hi']),
        '## 3. 절대 괴리와 부호 반전: 사전에 고정한 기술 통계\n',
        'absolute_gap_reduction=처음 절대 괴리−12h 후 절대 괴리(bp); crossed_zero=반대편으로 넘어간 비율; overshot_farther=반대편으로 넘어가 처음보다 절대 괴리가 더 커진 비율. 비율의 단위는0~1이다.\n',
        table(raw[raw.minimum_gap_bp==5],['definition','group','n','quantity','estimate','lo','hi']),
        '## 4. 주 결과를 본 뒤 추가한 조건 통제 검증\n',
        '`directional movement = absolute gap reduction + reversal excess`. excess는 0을 통과한 후 이동을 두 번 센 차이이며, 모든 excess가 유해한 과도 반전인 것은 아니다. 기존 total과 m을 고정하고 새 absolute gap nuisance를 학습했다. theta_gap+theta_excess=기존 theta_total이 정확히 성립한다. cross/overshot 계수는0~1 단위라100배하면 %p다.\n',
        '새4개(정의별)/12개(세 정의)/기존18개와 합친30개 Holm을 공개한다. 사후 탐색 전체를 보정하거나 독립 확증으로 바꾸는 절차는 아니다.\n',
        table(extra,['definition','quantity','n','estimate','lo','hi','p_holm4','p_holm12','p_holm30']),
        '## 5. 월·정의·설정 민감도\n',
        'EQ 12h, 고정 RF40/첫 seed: 2월 제외,3월,10bp 조건까지 모두 공개한다. 아래 보조95% 구간은 다중비교 보정 구간이 아니다.\n',
        table(orth[(orth.definition=='EQ')&(orth.h==12)&(orth.learner=='forest40')&(orth.seed==SEEDS[0])&(orth.outcome=='total')],
            ['sample','minimum_gap_bp','block_days','n','estimate','lo','hi','p']),
        '추가 검증의 같은 민감도:\n',
        table(contrasts[(contrasts.definition=='EQ')&(contrasts.learner=='forest40')&(contrasts.seed==SEEDS[0])&(contrasts.block_days==5)],
            ['quantity','sample','minimum_gap_bp','n','estimate','lo','hi']),
        '같은 주 표본에서 RF20/RF40/seed/선형 비교:\n',
        table(orth[(orth.definition=='EQ')&(orth.h==12)&(orth['sample']=='full')&(orth.minimum_gap_bp==5)&(orth.block_days==5)&(orth.outcome=='total')],
            ['learner','seed','estimate','lo','hi']),
        table(contrasts[(contrasts.definition=='EQ')&(contrasts['sample']=='full')&(contrasts.minimum_gap_bp==5)&(contrasts.block_days==5)],
            ['learner','seed','quantity','estimate','lo','hi']),
        '동일 출발점의1/6/12h(최소5bp) 경로:\n',
        table(groups[(groups.definition=='EQ')&(groups['sample']=='common_path')&(groups.learner=='forest40')&(groups.seed==SEEDS[0])&groups.outcome.isin(['total','dominance'])],
            ['h','outcome','group','group_n','estimate','lo','hi']),
        '## 6. 잔차 하락과 실제 국내 가격 하락은 다른 사건\n',
        '509개 전체 평가 시점에서, 각 기준보다 잔차가 더 하락한 사건의 국내 USDT 가격 비하락 비중이다. 10bp가 고정 주 기준이며5/20bp도 공개한다. 비율의95% 구간은5일 블록이다. 국내 평균수익률의 양수만으로 통계적인 가격 상승을 주장하지 않는다.\n',
        table(mapping,['definition','residual_fall_bp','n','local_up_or_flat_n','local_up_or_flat_fraction','fraction_lo','fraction_hi']),
        '## 7. ML nuisance의 실제 품질\n',
        '학습 이전 자료 평균/빈도와 비교한 미래 월 MSE/Brier다. ML이 선형보다 우월해야만 경제적 발견을 인정하는 것은 아니지만, ML이 더 정확하게 조건을 통제했다고 단정하지 않는다.\n',
        table(quality[quality.h==12],['definition','learner','seed','overlap_fraction','brier','constant_brier','mse_total','constant_mse_total','mse_dominance','constant_mse_dominance']),
        table(read(x.DEST/'nuisance_quality.csv'),['definition','learner','seed','quantity','mse','constant_mse']),
        '## 8. 부호별 구조 차이 없이도 반전 비대칭이 생기는가\n',
        '앞선 결과를 본 뒤 단순 대칭 평균회귀 대안 두 개를 고정해 반증 검사했다. affine은 절편과 공통 rho, local_center72는 과거72h 중심과 공통 rho다. 오차를 +/-로 복제하므로 충격 분포와 rho에 부호별 비대칭을 넣지 않는다. 아래 모멘트는 (사건-귀무모형 예측확률)의 오차가 현재 부호와 연결되는지를 검사한다. 새로운 DML 인과효과가 아니며 앞선29.6%p와 직접 빼서 설명 비중을 만들지 않는다. 비기각은 귀무모형의 참이나 두 모형의 동등성을 증명하지 않는다.\n',
        table(read(sn.DEST/'primary_tests.csv'),['definition','null','quantity','n','estimate','lo','hi','p_holm12']),
        table(read(sn.DEST/'observed_expected.csv'),['definition','null','quantity','group','n','observed','expected','past_center_mean_bp']),
        table(read(sn.DEST/'quality.csv'),['definition','null','quantity','brier','constant_brier']),
        '## 9. 해석과 검증 기록\n',
        '핵심 해석과 선행연구 대비 범위는 [CONCLUSION_KO.md](CONCLUSION_KO.md)를 참조한다. 모든 회귀 대비는 관측 조건부 연관성이다. 학습된 모형을 고정한 시계열 블록 구간이며, 전체 재탐색·잔차 추정·nuisance 재학습 불확실성을 모두 포함한 인과 신뢰구간이 아니다. CAP 가중치는 고정 공급량 대용이다. EQ/PCA가 매우 유사하므로 서로 독립된 시장의 반복 증거로 세지 않는다.\n',
        '주 검증:6개 사전 테스트,3588개 원시 가격 경로 재계산,9건 재학습. 후속:3개 사전 테스트,모든 관측의 piecewise 항등식,기존 m/g_total 보존,9건 재학습. 후속 lock 생성에서 상대 소스 경로를 절대 경로로 바꾸는 기록 오류를 수정했다(`3704309`→`b11d137`); 후속 학습/결과 개봉 이전 수정이며 주 계산은 바뀌지 않았다.\n',
        '추가로 대칭 모형의 반사 성질과 미래 표적 비사용 테스트2개, 모든 평가 시점72h 중심의 독립 시간 슬라이스 대조를 통과했다. 단순 대안 검사 설계는 `f0a35c2`로 고정했다.\n',
        '전체 CSV, 예측, 원자료 연결, 실행 시각과 SHA256을 보관했다. [주 설계](PROTOCOL_KO.md) · [사후 해석 검증 설계](OVERSHOOT_PROTOCOL_KO.md) · [대칭 모형 검사](SYMMETRIC_NULL_PROTOCOL_KO.md) · [참고문헌과 읽은 범위](REFERENCES.md)\n',
        '![EQ 주 결과](adjustment_summary.png)\n']
    (HERE/'RESULTS_KO.md').write_text('\n'.join(chunks))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.7));colors=['#2b6f9e','#d17935']
    for j,(group,color) in enumerate(zip(['discount','premium'],colors)):
        rr=raw[(raw.definition=='EQ')&(raw.minimum_gap_bp==5)&(raw.group==group)&(raw.quantity=='absolute_gap_reduction')].iloc[0]
        ccg=eqg[(eqg.group==group)&(eqg.outcome=='total')].iloc[0]
        vals=[ccg.estimate,rr.estimate];lo=[ccg.lo,rr.lo];hi=[ccg.hi,rr.hi]
        axes[0].bar(np.arange(2)+(j-.5)*.32,vals,.30,color=color,label=group.title(),
            yerr=np.array([np.array(vals)-lo,np.array(hi)-vals]),capsize=3)
    axes[0].set_xticks([0,1]);axes[0].set_xticklabels(['Directed movement','Absolute gap reduction'])
    axes[0].set_ylabel('Mean basis points');axes[0].legend(frameon=False)
    axes[0].set_title('Observed 12-hour changes')
    vals=[eqo[eqo.outcome=='total'].iloc[0]]+[eqa[eqa.quantity==q].iloc[0] for q in ['gap','excess']]
    axes[1].errorbar([v.estimate for v in vals],[2,1,0],xerr=np.array([[v.estimate-v.lo for v in vals],[v.hi-v.estimate for v in vals]]),fmt='o',color='#335c67',capsize=4)
    axes[1].set_yticks([2,1,0]);axes[1].set_yticklabels(['Directed movement','Absolute reduction','Reversal excess'])
    axes[1].axvline(0,color='gray',lw=.8);axes[1].set_xlabel('Premium minus discount (bp)')
    axes[1].set_title('RF-adjusted contrast; exploratory')
    rr=raw[(raw.definition=='EQ')&(raw.minimum_gap_bp==5)&(raw.quantity=='overshot_farther')].set_index('group')
    for j,group in enumerate(['discount','premium']):
        r=rr.loc[group];axes[2].bar(j,r.estimate*100,.6,color=colors[j],yerr=np.array([[r.estimate-r.lo],[r.hi-r.estimate]])*100,capsize=3)
    axes[2].set_xticks([0,1]);axes[2].set_xticklabels(['Discount','Premium']);axes[2].set_ylabel('Observed fraction (%)')
    axes[2].set_title('Crossed zero and ended farther away')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Korean USDT relative gaps: larger reversal is not automatically safer adjustment',fontsize=13)
    fig.text(.5,.01,'EQ | 318 origins (204 discount, 114 premium) | Dec 2025-Mar 2026 | 95% 5-day block intervals | association, not causation',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(HERE/'adjustment_summary.png',dpi=180);fig.savefig(HERE/'adjustment_summary.pdf');plt.close(fig)
    print('Rendered sealed result tables and figure.')


if __name__=='__main__':main()
