"""Replay the saved search map from immutable public refs, without scientists."""
import argparse
import json
from pathlib import Path

from trottertracks.saved_rpe_cost_catalog import Reader
from trottertracks.h4_search_run_evidence import discover_runs
from trottertracks.h4_search_map import build_index, report
from trottertracks.h4_search_map_html import index_text, render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True,help='Saved map SOURCES.json, containing fixed heads')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    sources=json.loads(a.sources.read_text());runs,issues=discover_runs(Reader(root),sources['heads'],legacy_heads=set(sources.get('legacy_heads',sources['heads'])))
    index=build_index(json.loads(a.catalog.read_text()),runs,sources=sources.get('sources',[]),issues=issues)
    index['included_public_heads']=sources['heads']
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'INDEX.json').write_text(index_text(index)+'\n')
    (a.output/'index.html').write_text(render(index))
    (a.output/'REPORT.md').write_text(report(index))
    print(json.dumps(dict(runs=len(index['runs']),points=len(index['points']),targets=len(index['targets']),issues=len(index['issues']))))


if __name__=='__main__':main()
