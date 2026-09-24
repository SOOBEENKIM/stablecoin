"""Verification-only launcher; preserve all locked model/analysis sources.

Inherited modules prepend legacy directories, which also contain run.py.
Restore this directory after importing dependencies so the verification script
imports the intended, hash-locked nested run module. No forecasts are selected
or scored here; the original verification entry point is called unchanged.
"""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import nested
sys.path.insert(0,str(HERE))
import run
import verify_results

assert Path(run.__file__).resolve()==HERE/'run.py'
assert Path(verify_results.__file__).resolve()==HERE/'verify_results.py'

if __name__=='__main__':
    verify_results.main()
