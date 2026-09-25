import dv_verify_v2 as original
from dv_alignment import aligned_read,verify_alignment
from dv_core import OUT,read,sha,now,write_new


if __name__=='__main__':
    stamp=verify_alignment()
    missing=int(read(OUT/'archived_forecasts.csv.gz').e_now.isna().sum())
    original.read=aligned_read
    original.main()
    write_new(OUT/'ALIGNMENT_VERIFIED.json',dict(**stamp,verified_utc=now(),recovered_current_state_rows=missing,
        raw_archived_forecasts_sha256=sha(OUT/'archived_forecasts.csv.gz'),predictions_unchanged=True))
