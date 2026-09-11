"""Independent safety properties for delayed-label forecast calibration."""
from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_extension import calibrate


class InformationTiming(unittest.TestCase):
    def fixture(self):
        idx=pd.date_range('2025-11-01',periods=400,freq='h',tz='UTC')
        frame=pd.DataFrame({'target':np.arange(400,dtype=float),
                            'target_time':idx+pd.Timedelta(hours=6)},index=idx)
        return frame,np.zeros((400,3))

    def test_six_hour_label_delay_and_strict_cutoff(self):
        frame,pred=self.fixture()
        adjusted,n=calibrate(frame,pred)
        # At origin 200, labels from origins 0..193 have arrived strictly earlier.
        self.assertEqual(n[200],194)
        np.testing.assert_allclose(adjusted[200],[19.3,96.5,173.7])

    def test_unobserved_targets_cannot_change_current_prediction(self):
        frame,pred=self.fixture()
        original,_=calibrate(frame,pred)
        frame.loc[frame.target_time>=frame.index[200],'target']=1e8
        changed,_=calibrate(frame,pred)
        np.testing.assert_allclose(changed[:201],original[:201])

    def test_calendar_gap_does_not_keep_old_errors(self):
        frame,pred=self.fixture()
        idx=frame.index.to_list()
        idx[-1]=idx[-1]+pd.Timedelta(days=60)
        frame.index=pd.DatetimeIndex(idx)
        frame['target_time']=frame.index+pd.Timedelta(hours=6)
        adjusted,n=calibrate(frame,pred)
        self.assertEqual(n[-1],0)
        np.testing.assert_allclose(adjusted[-1],pred[-1])


if __name__=='__main__':unittest.main()
