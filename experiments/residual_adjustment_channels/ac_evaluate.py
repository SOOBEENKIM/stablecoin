"""Predefined conditional contrasts and full price-route diagnostics."""
from ac_core import *

OUTCOMES=PARTS+['total','dominance']


def inference(point,draws):
    finite=np.isfinite(draws);draws=np.asarray(draws)[finite]
    if not len(draws):return dict(estimate=point,lo=np.nan,hi=np.nan,p=np.nan,bootstrap_valid=0.)
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(estimate=float(point),lo=float(lo),hi=float(hi),bootstrap_valid=float(finite.mean()),
        p=float((1+np.count_nonzero(np.abs(draws-point)>=abs(point)))/(len(draws)+1)))


def group_result(d,w,name,group):
    mask=(d.premium.to_numpy()==group).astype(float)
    y=d['close_'+name].to_numpy();den=w@mask
    draws=np.divide(w@(mask*y),den,out=np.full(len(w),np.nan),where=den>0)
    point=float(y[mask>0].mean()) if mask.sum() else np.nan
    return inference(point,draws)


def evaluate_cell(d,block):
    d=d.sort_values('origin').reset_index(drop=True)
    w=dv.day_weights(d[['origin','fold']],block)
    orth=[];groups=[]
    for outcome in OUTCOMES:
        result=inference(orthogonal(d,outcome),orthogonal(d,outcome,w))
        orth.append(dict(outcome=outcome,**result))
        for group in [0,1]:
            g=d[d.premium==group]
            groups.append(dict(outcome=outcome,group='premium' if group else 'discount',n=len(g),
                days=g.origin.dt.normalize().nunique(),**group_result(d,w,outcome,group)))
    # Partialling-out preserves additive price-route identity exactly.
    values={r['outcome']:r['estimate'] for r in orth}
    assert abs(sum(values[c] for c in PARTS)-values['total'])<1e-7
    assert abs(values['local']-values['basket_price']-values['dominance'])<1e-7
    return orth,groups


def main():
    stamp=verify_seal();v=json.loads((OUT/'VERIFICATION.json').read_text())
    assert v['predictions_sha256']==sha(OUT/'predictions.csv.gz')
    write_new(OUT/'SCORES_OPENED.json',dict(**stamp,opened_utc=now(),independent_confirmation=False))
    pred=read(OUT/'predictions.csv.gz');context=read(PREP/'context.csv.gz')
    orth_rows=[];group_rows=[];quality=[];membership=[]
    for (definition,h,learner,seed),g in pred.groupby(['definition','h','learner','seed']):
        g=g.sort_values('origin').reset_index(drop=True)
        keys=dict(definition=definition,h=int(h),learner=learner,seed=int(seed))
        # Common origins for a true 1/6/12-hour path, independent of outcomes.
        available=[set(context[(context.definition==definition)&(context.h==hh)].origin) for hh in HORIZONS]
        common=set.intersection(*available)
        samples=[]
        if h==12:
            samples += [('full',5,g[g.abs_e>5],b) for b in [1,5,10]]
            for threshold in [0,10]:samples.append(('full',threshold,g[g.abs_e>threshold],5))
            base=g[g.abs_e>5].copy();samples.append(('nonoverlap',5,base[dv.nonoverlap(base)],5))
            samples.append(('overlap',5,base[(base.m>=.1)&(base.m<=.9)],5))
            for fold in sorted(g.fold.unique()):
                samples.append((fold,5,base[base.fold==fold],5))
                samples.append(('without_'+fold,5,base[base.fold!=fold],5))
        samples.append(('common_path',5,g[(g.abs_e>5)&g.origin.isin(common)],5))
        for sample,threshold,d,block in samples:
            if len(d)<10 or d.premium.nunique()!=2:
                membership.append(dict(**keys,sample=sample,minimum_gap_bp=threshold,block_days=block,n=len(d),status='insufficient_groups'))
                continue
            pos=int(d.premium.sum());neg=len(d)-pos
            days=d.origin.dt.normalize().nunique()
            key=dict(**keys,sample=sample,minimum_gap_bp=threshold,block_days=block,n=len(d),premium_n=pos,discount_n=neg,days=days)
            membership.append(dict(**key,status='adequate' if min(pos,neg)>=20 and days>=10 else 'limited'))
            orth,groups=evaluate_cell(d,block)
            orth_rows.extend(dict(**key,**r) for r in orth)
            for r in groups:
                r['group_n']=r.pop('n');r['group_days']=r.pop('days');group_rows.append(dict(**key,**r))
        base=g[g.abs_e>5] if h==12 else g[(g.abs_e>5)&g.origin.isin(common)]
        v=base.premium-base.m
        row=dict(**keys,n=len(base),premium_fraction=float(base.premium.mean()),probability_min=base.m.min(),
            probability_max=base.m.max(),overlap_fraction=float(base.m.between(.1,.9).mean()),
            mean_residualized_sign_square=float(np.mean(v*v)),brier=float(np.mean(v*v)),
            constant_brier=float(np.mean((base.premium-base.m_constant)**2)))
        for outcome in OUTCOMES:
            row['mse_'+outcome]=float(np.mean((base['close_'+outcome]-base['g_'+outcome])**2))
            row['constant_mse_'+outcome]=float(np.mean((base['close_'+outcome]-base['g_constant_'+outcome])**2)) if 'g_constant_'+outcome in base else np.nan
        quality.append(row)
    orth=pd.DataFrame(orth_rows);groups=pd.DataFrame(group_rows)
    primary=[]
    for definition in DEFS:
        mask=(orth.definition==definition)&(orth.h==12)&(orth.learner=='forest40')&(orth.seed==SEEDS[0])&\
            (orth['sample']=='full')&(orth.minimum_gap_bp==5)&(orth.block_days==5)
        for row in orth[mask&orth.outcome.isin(['total','dominance'])].to_dict('records'):
            primary.append(dict(test='adjusted_premium_minus_discount_'+row['outcome'],**row))
        mask=(groups.definition==definition)&(groups.h==12)&(groups.learner=='forest40')&(groups.seed==SEEDS[0])&\
            (groups['sample']=='full')&(groups.minimum_gap_bp==5)&(groups.block_days==5)
        for row in groups[mask&groups.outcome.isin(['total','dominance'])].to_dict('records'):
            primary.append(dict(test='mean_'+row['group']+'_'+row['outcome'],**row))
    primary=pd.DataFrame(primary);assert len(primary)==18
    primary['p_holm18']=dv.holm(primary.p)
    primary['p_holm6']=np.nan
    for definition,g in primary.groupby('definition'):primary.loc[g.index,'p_holm6']=dv.holm(g.p)
    # Observed residual loss is not equivalent to a fall in the local USDT price.
    mapping=[];overshoot=[]
    for definition,d in context[context.h==12].groupby('definition'):
        d=d.sort_values('origin').reset_index(drop=True)
        w=dv.day_weights(d[['origin','fold']],5)
        for threshold in [5,10,20]:
            event=d.delta_e<-threshold
            n=int(event.sum())
            den=w@event.astype(float).to_numpy()
            num=w@(event&(d.local_return_bp>=0)).astype(float).to_numpy()
            draws=np.divide(num,den,out=np.full(len(w),np.nan),where=den>0)
            interval=inference(float((d.loc[event,'local_return_bp']>=0).mean()),draws)
            mapping.append(dict(definition=definition,residual_fall_bp=threshold,n=n,
                local_up_or_flat_n=int((event&(d.local_return_bp>=0)).sum()),
                local_up_or_flat_fraction=float((d.loc[event,'local_return_bp']>=0).mean()),
                fraction_lo=interval['lo'],fraction_hi=interval['hi'],
                local_mean_bp=float(d.loc[event,'local_return_bp'].mean()),
                basket_implied_mean_bp=float(d.loc[event,'basket_implied_return_bp'].mean())))
        for threshold in [0,5,10]:
            for sign in [0,1]:
                mask=((d.abs_e>threshold)&(d.premium==sign)).astype(float).to_numpy()
                den=w@mask
                for name in ['absolute_gap_reduction','crossed_zero','overshot_farther','local_return_bp','basket_implied_return_bp']:
                    y=d[name].to_numpy()
                    draws=np.divide(w@(mask*y),den,out=np.full(len(w),np.nan),where=den>0)
                    point=float(y[mask>0].mean()) if mask.sum() else np.nan
                    overshoot.append(dict(definition=definition,minimum_gap_bp=threshold,
                        group='premium' if sign else 'discount',n=int(mask.sum()),quantity=name,**inference(point,draws)))
    # Empirical dominance and bootstrap formulas verified separately from fit/score production.
    max_orth_error=0.
    for row in primary[primary.test.str.startswith('adjusted')].itertuples():
        d=pred[(pred.definition==row.definition)&(pred.h==12)&(pred.learner=='forest40')&(pred.seed==SEEDS[0])&(pred.abs_e>5)]
        vv=np.asarray(d.premium-d.m);yy=np.asarray(d['close_'+row.outcome]-d['g_'+row.outcome])
        estimate=float(np.linalg.lstsq(vv[:,None],yy,rcond=None)[0][0])
        max_orth_error=max(max_orth_error,abs(estimate-row.estimate))
    assert max_orth_error<1e-8
    for name,frame in [('orthogonal_contrasts',orth),('group_channels',groups),('primary_tests',primary),
        ('nuisance_quality',pd.DataFrame(quality)),('sample_audit',pd.DataFrame(membership)),('residual_vs_price',pd.DataFrame(mapping)),
        ('absolute_gap_and_prices',pd.DataFrame(overshoot))]:save(frame,OUT/(name+'.csv'))
    files=list(OUT.glob('*.csv'))+[OUT/'SCORES_OPENED.json']
    write_new(OUT/'EVALUATION_COMPLETE.json',dict(**stamp,completed_utc=now(),independent_orthogonal_error=max_orth_error,
        independent_confirmation=False,causal_identification=False,file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print(primary[['definition','test','n','estimate','lo','hi','p_holm6','p_holm18']].to_string(index=False))


if __name__=='__main__':main()
