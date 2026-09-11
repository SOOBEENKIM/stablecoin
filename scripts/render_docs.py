"""Make navigable GitHub copies of immutable historical Markdown reports."""
from pathlib import Path
import json
import os
import re
from urllib.parse import quote,unquote

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/ssd-990/soobeenkim')
SOURCES=list((ROOT/'research').rglob('*.md'))+list((ROOT/'context').rglob('*.md'))
TARGETS={p:ROOT/'docs'/p.relative_to(ROOT) for p in SOURCES}
PATTERN=re.compile(r'(!?\[[^\]\n]*\])\(([^)\n]+)\)')


def map_path(path,source):
    if path.startswith(str(OLD/'stablecoin_workshop')+'/'):
        return ROOT/'research'/Path(path).relative_to(OLD/'stablecoin_workshop')
    if path.startswith(str(OLD/'stablecoin_v4')+'/'):
        return ROOT/'research/reference/original_v4'/Path(path).relative_to(OLD/'stablecoin_v4')
    if path.startswith(str(OLD/'stablecoin_audit_20260908')+'/'):
        rel=Path(path).relative_to(OLD/'stablecoin_audit_20260908')
        context=ROOT/'context/audit_20260908'/rel
        return context if context.exists() else ROOT/'research/reference/audit_20260908'/rel
    if path.startswith('/'):
        return Path(path)
    return (source.parent/path).resolve()


def main():
    missing=[];links=0
    for source,target in TARGETS.items():
        target.parent.mkdir(parents=True,exist_ok=True)
        def replace(match):
            nonlocal links
            label,url=match.groups()
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',url) or url.startswith('#'):return match.group()
            url=unquote(url.strip('<>'));path,sep,anchor=url.partition('#')
            line=re.search(r':(\d+)$',path)
            if line:path=path[:line.start()];anchor='L'+line.group(1)
            mapped=map_path(path,source)
            if mapped in TARGETS:mapped=TARGETS[mapped]
            if not mapped.exists() and mapped not in TARGETS.values():
                missing.append(dict(source=str(source.relative_to(ROOT)),target=url))
                return label.strip('![]')+' (historical source path; see archive notes)'
            links+=1
            rel=os.path.relpath(mapped,target.parent)
            return label+'('+quote(rel,safe='/._-')+('#'+anchor if anchor else '')+')'
        body=PATTERN.sub(replace,source.read_text())
        original=quote(os.path.relpath(source,target.parent),safe='/._-')
        root_guide=quote(os.path.relpath(ROOT/'docs/REPRODUCING.md',target.parent),safe='/._-')
        note='> GitHub 탐색용 사본입니다. [체크섬이 보존된 원본]('+original+')의 내용과 과거 시점 표기를 유지하고 문서 링크만 변환했습니다. 실행은 [현재 재현 안내]('+root_guide+')를 따릅니다.\n\n'
        target.write_text(note+body)
    (ROOT/'provenance/document_links.json').write_text(json.dumps(dict(rendered_documents=len(SOURCES),
        converted_links=links,historical_unbundled_references=missing),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(documents=len(SOURCES),links=links,unbundled_references=len(missing)),indent=2))


if __name__=='__main__':main()
