"""Independent checks for the new forecast target and delayed calibration."""
import unittest
import numpy as np
import pandas as pd
from run_pilot import future_any,intercept_calibration


class TimingChecks(unittest.TestCase):
    def test_current_breach_is_not_a_future_breach(self):
        st=pd.Series([1.,0,0,0,0,0,0],index=pd.date_range('2025-01-01',periods=7,freq='h',tz='UTC'))
        self.assertEqual(future_any(st).iloc[0],0)
        self.assertTrue(future_any(st).iloc[1:].isna().all())

    def test_missing_in_horizon_is_not_silently_zero(self):
        st=pd.Series([0.,1,0,np.nan,0,0,0,0],index=pd.date_range('2025-01-01',periods=8,freq='h',tz='UTC'))
        self.assertTrue(np.isnan(future_any(st).iloc[0]))

    def test_six_hour_endpoint_is_included(self):
        st=pd.Series([0.,0,0,0,0,0,1],index=pd.date_range('2025-01-01',periods=7,freq='h',tz='UTC'))
        self.assertEqual(future_any(st).iloc[0],1)

    def test_unavailable_labels_cannot_change_current_calibration(self):
        ix=pd.date_range('2025-11-01',periods=24*65,freq='h',tz='UTC')
        f=pd.DataFrame({'target':(np.arange(len(ix))%7==0).astype(float),
                        'target_time':ix+pd.Timedelta(hours=6)},index=ix)
        raw=np.full(len(f),.3)
        p,records=intercept_calibration(f,raw)
        altered=f.copy();cut=pd.Timestamp('2025-12-01',tz='UTC')
        mask=altered.target_time>=cut
        altered.loc[mask,'target']=1-altered.loc[mask,'target']
        other,_=intercept_calibration(altered,raw)
        dec=f.index.strftime('%Y-%m')=='2025-12'
        np.testing.assert_allclose(p[dec],other[dec],rtol=0,atol=0)
        dec_record=[r for r in records if r['month']=='2025-12'][0]
        self.assertLess(pd.Timestamp(dec_record['last_available_target']),cut)


if __name__=='__main__':
    unittest.main()
