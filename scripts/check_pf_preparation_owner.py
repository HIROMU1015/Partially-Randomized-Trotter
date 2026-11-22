"""Read the selected original D producer before making a task-owned clone.

This is metadata/preflight only. It never calls signals/acquire/freeze/finish.
Unregistered partial results and consumed STOPs remain at their original owner.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from run_pf_preparation_stage import (PRICE_PLAN_HOLDS, bind_preflight_snapshot, cache_mean_pool,
                                      saved_price_helpers, use_exploratory_plan)


def inspect(root,request,observations,signal_index=None,saved_price_index=None):
    root=Path(root).resolve();sys.path[:0]=[str(root/'scripts'),str(root/'src')]
    spec=importlib.util.spec_from_file_location('original_pf_provider',root/'scripts/prepare_pf_candidate.py')
    provider=importlib.util.module_from_spec(spec);spec.loader.exec_module(provider)
    source=provider.source_gate();rows=[]
    bind_preflight_snapshot()
    exploratory = request.get('exploratory_cost') is True
    if exploratory:
        use_exploratory_plan()
    helpers, extra = None, None
    for item in request['selected']:
        configuration=provider.selected_configuration(item);cid=configuration['id']
        registry=Path(request['registry_file']);entry=None
        if registry.is_file():entry=json.loads(registry.read_text()).get('configurations',{}).get(cid)
        if entry and all(entry.get(k)for k in ['signal_packet','prices','q3_inputs']):
            rows.append(dict(configuration_id=cid,status='REUSE_COMPLETE_ORIGINAL_OWNER',entry=entry));continue
        owners={Path(request['output_root'])/cid,root/provider.STUDY/'selected'/cid,
                root/'.pf_evidence_private'/cid}
        owners.update(Path(p)for p in request.get('prior_owner_directories',[]))
        found=[]
        for directory in sorted(owners):
            if not directory.is_dir():continue
            # Explicit selected directories only, bounded depth; no project or
            # private-circuit archive scan and no process signalling.
            for pattern in ['*.json','*/*.json','*/*/*.json','*/*/*/*.json']:
                for path in directory.glob(pattern):
                    if any(word in path.name.upper()for word in ['STARTED','STARTING','LAUNCH','STOP','RESULT','STATUS']):
                        found.append(str(path))
            completed=directory/'COMPLETION.json'
            if completed.is_file():
                saved=json.loads(completed.read_text());candidate=saved.get('registry_entry')
                if saved.get('configuration_id')==cid and candidate and all(candidate.get(k)for k in ['signal_packet','prices','q3_inputs']):
                    entry=candidate;break
        if entry:
            rows.append(dict(configuration_id=cid,status='REUSE_COMPLETE_ORIGINAL_OWNER',entry=entry));continue
        if found:
            rows.append(dict(configuration_id=cid,status='PRIOR_OWNER_HOLD',
                reason='Selected original owner has running, UNKNOWN, STOP or unregistered partial scientific artifacts; recover/admit those before another execution',
                original_paths=sorted(set(found)), original_owner_directories=sorted(map(str, owners))));continue
        # Preserve D's exact selected-cell duplicate check at the ORIGINAL
        # root, where its live/private owner markers can still be observed.
        from trottertracks import pf_evidence_preparation as E
        if signal_index is not None:
            cache_mean_pool(root,signal_index)
            signal_index=None
        if saved_price_index is not None:
            helpers = saved_price_helpers()
            extra = helpers['use_saved_prices'](saved_price_index)
            saved_price_index = None
        try:
            if request.get('selected_axis_compile') is True and not getattr(provider, 'SELECTED_AXIS_COMPILE', False):
                rows.append(dict(configuration_id=cid, status='PRICE_EVIDENCE_REQUIRED',
                                 reason='Selected axis compile requires a compatible fixed PF provider'))
                continue
            options = {'allow_single_axis': True} if request.get('selected_axis_compile') is True else {}
            plan=E.make_plan(root,configuration,source,**options)
        except ValueError as error:
            if str(error) not in PRICE_PLAN_HOLDS:
                raise
            rows.append(dict(configuration_id=cid,status='PRICE_EVIDENCE_REQUIRED',
                             reason=str(error),signal_deferred_for_price=True))
            continue
        consumed = {tuple(axis) for axis in request.get('previously_audited_price_axes', ())}
        admitted = {(p['physical_id'], p['axis']) for p in plan['direct_prices']} if consumed else set()
        if consumed - admitted:
            rows.append(dict(configuration_id=cid, status='PRICE_EVIDENCE_REQUIRED',
                             reason='Previously completed PF axes must pass saved-price admission before further work',
                             missing_audited_axes=sorted(consumed - admitted)))
            continue
        destination=Path(observations)/cid;destination.mkdir(parents=True,exist_ok=True)
        try:
            discovered = []
            deferred = {}
            if helpers:
                report = destination/'DUPLICATE_ROUNDS.json'
                deferred = helpers['unresolved_rounds'](root, plan, report, extra, discover=True)
                if report.exists():
                    discovered = json.loads(report.read_bytes())['discovered_admitted_receipts']
                if discovered:
                    plan = E.make_plan(root, configuration, source, **options)
                    deferred = helpers['unresolved_rounds'](
                        root, plan, destination/'REMAINING_DUPLICATES.json', extra, discover=True)
            cleared = helpers['independent_plan'](plan, deferred) if deferred else plan
            available = [cell for cell in cleared.get('new_compile_cells', ())
                         if not deferred or cell['role'] != 'HOLDOUT']
            raw_calibration = helpers['original_calibration'](
                root, plan, deferred, Path('docs/research/distributed_selected_pf')/cid) if deferred else None
            needs_signal = plan.get('new_signal_means', 0) and cid not in request.get('independent_saved_signals', {})
            if deferred and not available and not needs_signal and not raw_calibration and not exploratory:
                rows.append(dict(configuration_id=cid, status='PRICE_EVIDENCE_REQUIRED',
                    reason=('No independently clear wrapper axis' if plan.get('selected_axis_compile')
                            else 'No independently clear pair of X/Y wrappers'), deferred_price_rounds=deferred,
                    discovered_price_receipts=discovered))
                continue
            provider.preflight(cleared, destination)
            row = dict(configuration_id=cid,status='ORIGINAL_OWNER_PREFLIGHT_PASS')
            if helpers:
                row.update(reused_guarded_axes=sum(price['price_evidence']['admission']['path'].endswith('/GUARD_CHECK.json')
                                                   for price in plan['direct_prices']),
                           clear_compile_rounds=[cell['m'] for cell in available])
            if deferred:
                row['deferred_price_rounds'] = deferred
            if raw_calibration:
                row['original_calibration'] = raw_calibration
            if discovered:
                row['discovered_price_receipts'] = discovered
            rows.append(row)
        except (OSError, KeyError, TypeError, ValueError) as error:
            rows.append(dict(configuration_id=cid,status='PRIOR_OWNER_HOLD',reason=type(error).__name__+': '+str(error)))
    return dict(schema='PF_ORIGINAL_OWNER_METADATA_V1',source=source,rows=rows,scientific_calls=0,compile_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--provider-root',type=Path,required=True)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--signal-index',type=Path)
    parser.add_argument('--saved-price-index',type=Path)
    args=parser.parse_args();request=json.loads(args.request.read_text())
    result=inspect(args.provider_root,request,args.output.parent/'original_preflight',args.signal_index,args.saved_price_index)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
