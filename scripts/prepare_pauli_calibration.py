"""B2-selected A candidate + supplied D2 price applicability -> calibration request.

Metadata only. Does not execute path/input_preflight, compile, fit, quote or Q3.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import pauli_calibration_request as C
from trottertracks import pauli_signal_entry as E


def read_input(value):
    if ':' in value and not value.startswith('/') and len(value.split(':', 1)[0]) >= 7:
        commit, path = value.split(':', 1)
        ref = dict(commit=commit, path=path)
        return E.read_ref(ROOT, ref), ref
    data = json.loads(Path(value).read_text())
    if isinstance(data,dict) and set(data) <= {'commit', 'path', 'selector'} and 'commit' in data:
        return E.read_ref(ROOT, data), data
    return data, dict(file=str(Path(value).resolve()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--preparation')
    inputs.add_argument('--batch',help='JSON list of {preparation/configuration, quote}; deduplicate physical calibration cells')
    p.add_argument('--price-result')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--configuration-id')
    p.add_argument('--inspect-saved-cell', action='store_true')
    p.add_argument('--execution', type=Path, help='Optional future output/cohort/branch JSON; no execution')
    args = p.parse_args(argv)
    execution = json.loads(args.execution.read_text()) if args.execution else None
    if args.batch:
        batch,batch_ref=read_input(args.batch)
        result=C.prepare_many(ROOT,batch,execution=execution)
        result['input_sources']=dict(batch=batch_ref)
    else:
        if not args.price_result:p.error('--price-result required with --preparation')
        prep, prep_ref = read_input(args.preparation); quote, quote_ref = read_input(args.price_result)
        if prep.get('schema')=='PR_RPE_RMSE_V2':
            prep=C.canonical_preparation(prep,prep_ref if 'commit'in prep_ref else None)
    if not args.batch and args.configuration_id and 'items' in prep:
        matches = [x for x in prep['items'] if x.get('configuration_id') == args.configuration_id]
        if len(matches) != 1:
            p.error('one matching prepared configuration required')
        prep = matches[0]
    if not args.batch:
        result = C.prepare(ROOT, prep, quote, execution=execution, inspect_saved_cell=args.inspect_saved_cell)
        result['input_sources'] = dict(A_preparation=prep_ref, D2_applicability=quote_ref)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if result.get('driver_request') is not None:
        req = args.output.with_name(args.output.stem+'.CALIBRATION_REQUEST.json')
        req.write_text(json.dumps(result['driver_request'], indent=2, allow_nan=False)+'\n')
        result['driver_request_file'] = str(req.resolve())
        result['driver_validation_command'] = [sys.executable, '-B', str(ROOT/'scripts/run_pauli_split_cal.py'),
                                                '--request', str(req.resolve()), '--metadata']
    for i,group in enumerate(result.get('calibration_requests',[])):
        req=args.output.with_name(args.output.stem+'.CALIBRATION_%02d.json'%i)
        req.write_text(json.dumps(group['driver_request'],indent=2,allow_nan=False)+'\n')
        group['driver_request_file']=str(req.resolve())
        group['driver_validation_command']=[sys.executable,'-B',str(ROOT/'scripts/run_pauli_split_cal.py'),
            '--request',str(req.resolve()),'--metadata']
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result.get('status','BATCH_PREPARED'), configuration_id=result.get('configuration_id'),
        reused_axes=len(result.get('reused_prices',[])), missing_axes=len(result.get('missing_prices',[])), needs=result.get('needs',[]))))
    return result


if __name__ == '__main__':
    main()
