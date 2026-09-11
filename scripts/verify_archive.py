"""Verify all bundled research bytes, including actual Git LFS objects."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify():
    manifest=json.loads((ROOT/'provenance/archive_manifest.json').read_text())
    archive=ROOT/'research';failures=[]
    for name,entry in manifest['files'].items():
        path=archive/name
        if not path.is_file():failures.append(name+' (missing)')
        elif path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
            failures.append(name+' (size/hash mismatch; run git lfs pull if this is a pointer)')
    if failures:raise SystemExit('\n'.join(failures))
    latest=archive/'option1_factorial_20260910'
    preserved=json.loads((latest/'PRESERVED_OPTIONS_SHA256.json').read_text())
    for name,digest in preserved.items():assert sha(archive/name)==digest,name
    current=json.loads((latest/'ARTIFACT_MANIFEST.json').read_text())['files']
    for name,entry in current.items():assert sha(latest/name)==entry['sha256'],name
    context=json.loads((ROOT/'provenance/context_manifest.json').read_text())
    for name,entry in context.items():assert sha(ROOT/name)==entry['sha256'],name
    return dict(status='passed',archived_files=manifest['count'],bytes=manifest['bytes'],
        previous_preserved_files=len(preserved),latest_manifest_files=len(current),context_files=len(context))


if __name__=='__main__':print(json.dumps(verify(),indent=2))
