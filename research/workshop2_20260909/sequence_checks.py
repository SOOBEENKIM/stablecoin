"""Independent chronological-window, label, and causal-convolution checks."""
from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
import torch
from train_sequences import CausalBlock
HERE=Path(__file__).resolve().parent


class SequenceChecks(unittest.TestCase):
    def test_tcn_does_not_use_future_positions(self):
        torch.manual_seed(44)
        block=CausalBlock(3,4,2).eval()
        x=torch.randn(2,3,24);changed=x.clone();changed[:,:,12:]+=100
        with torch.no_grad():
            before=block(x);after=block(changed)
        torch.testing.assert_close(before[:,:,:12],after[:,:,:12],rtol=0,atol=0)

    def test_histories_and_labels_against_price_source(self):
        data=np.load(HERE/'sequence_data.npz');X=data['X'];y=data['y']
        ix=pd.to_datetime(data['origin_ns'],utc=True)
        src=Path('/reference.csv') if Path('/reference.csv').exists() else HERE.parent/'extension_20260909/basis_definitions.csv'
        z=pd.read_csv(src,index_col=0,parse_dates=True);z.index=pd.to_datetime(z.index,utc=True)
        z=z.reindex(pd.date_range(z.index.min(),z.index.max(),freq='h'))
        cut=z.loc[z.index<pd.Timestamp('2025-11-01',tz='UTC'),['BTC','ETH']].quantile(.1)
        c=z[['BTC','ETH']].lt(cut).all(axis=1).astype(float).where(z[['BTC','ETH']].notna().all(axis=1))
        col=list(data['feature_names']).index('basis_BTC')
        for i,t in enumerate(ix):
            expected=z.BTC.reindex(pd.date_range(t-pd.Timedelta(hours=23),t,freq='h')).to_numpy()
            np.testing.assert_allclose(X[i,:,col],expected,rtol=1e-6,atol=1e-5)
            future=c.reindex(pd.date_range(t+pd.Timedelta(hours=1),t+pd.Timedelta(hours=6),freq='h'))
            self.assertTrue(future.notna().all())
            self.assertEqual(float(y[i]),float(future.max()))
            self.assertEqual(float(c.loc[t]),0.)


if __name__=='__main__':unittest.main()
