#!/usr/bin/env python3
"""Metadata inspection by default. Evidence evaluation needs user permission.

The CLI does not generate H4 operators or build circuits. The documented pilot
uses c1_stage3_reference/connection and existing saved-data/native/RTE routines
to produce evidence, then this fixed assessment entry to calculate resources.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--evaluate-evidence',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if bool(args.evaluate_evidence) != bool(args.output):
        p.error('Evidence evaluation requires a fresh output; metadata inspection writes no result')
    contract=json.loads(args.contract.read_text())
    source_path=ROOT/contract['source_manifest']
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=contract['source_manifest_sha256']:
        raise ValueError('Scientific SOURCE manifest mismatch')
    source=json.loads(source_path.read_text())
    for name,want in source['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=want:
            raise ValueError('Scientific SOURCE changed: '+name)
    for name,want in contract['saved_inputs'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=want:
            raise ValueError('Saved input changed: '+name)
    if args.evaluate_evidence is None:
        print(json.dumps(dict(status='STAGE3_METADATA_PASS_SCIENCE_UNAPPROVED',
            new_scientific_calls=0,CA_target=contract['target_energy_ha'],proposal_adopted=False)))
        return
    if args.output.exists():raise ValueError('Fresh output required; never overwrite science')
    packet=json.loads(args.evaluate_evidence.read_text())
    if packet['target']!=contract['proposal']['accuracy_target']:
        raise ValueError('Accuracy conditions differ from the registered proposal')
    if packet['target']['energy_ha']!=float(Fraction(contract['target_energy_ha'])):
        raise ValueError('Energy target differs from the fixed common CA contract')
    if packet['cost_scope']!=contract['proposal']['cost_scope']:
        raise ValueError('Cost scope differs from the registered proposal')
    from trottertracks.c1_stage3_accuracy import evaluate_evidence_packet
    result=evaluate_evidence_packet(packet)
    result['provenance']=dict(contract_sha256=hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        evidence_sha256=hashlib.sha256(args.evaluate_evidence.read_bytes()).hexdigest(),
        source_manifest_sha256=contract['source_manifest_sha256'])
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['accuracy']['status'],mandatory_stop=True)))


if __name__=='__main__':main()
