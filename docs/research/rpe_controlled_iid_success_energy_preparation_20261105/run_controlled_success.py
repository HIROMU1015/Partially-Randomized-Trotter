"""Default: byte/schema/literal checks only. --run is separate H4 science.

No Hamiltonian/state decoder, mean evaluator, circuit builder, RNG or decoder
execution is reachable here. Permission is the user's instruction, not a file.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction as Q
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'docs/research/rpe_controlled_connection_runner_20261104')]
import run_connection as base
from integration_audit import require
from connection_pipeline import STATUS, write_json

CONVENTION = dict(cosine='Re', sine='+Im', bit0=1, bit1=-1, evolution='exp(-itH)')
AXES = ('cosine', 'sine')


def validate_join(rows, categorical, weighted, old_axes, mean_axes, a_axes):
    """Identity comparisons only; no Uctrl, t, tail, sum or scientific decision."""
    require(len(rows)==len(weighted)==18, 'all18 rounds required')
    for table in (old_axes, mean_axes, a_axes):
        keys = [(int(x['m']), x['axis']) for x in table]
        require(len(keys)==36 and set(keys)=={(m,a) for m in range(18) for a in AXES},
                'missing/duplicate axis')
    joined = []
    for m,(row,w) in enumerate(zip(rows,weighted,strict=True)):
        require(row['m']==m and row['q']==1<<m and row['K']==2
                and all(row[k]==w[k] for k in ('m','q','r','K')), 'round identity')
        for axis,coordinate in (('cosine','real'),('sine','imag')):
            cat = categorical[m][axis]
            c = next(x for x in old_axes if (int(x['m']),x['axis'])==(m,axis))
            tsv = next(x for x in mean_axes if (int(x['m']),x['axis'])==(m,axis))
            old_a = next(x for x in a_axes if (int(x['m']),x['axis'])==(m,axis))
            new = w['axes'][axis]
            require(all(c[k]==str(row[k])==tsv[k] for k in ('m','q','r','K')),
                    'axis tuple identity')
            require(all(c[k]==row['axes'][axis][k]==old_a[k] for k in ('N','epsilon'))
                    and c['alpha']==row['axes'][axis]['alpha']==old_a['saved_alpha_unchanged']
                    and c['saved_mu0']==row['mu0'][coordinate], 'G center/budget identity')
            require(c['U_cat_upper']==cat['U_cat_upper']==old_a['U_cat_upper']
                    ==new['U_cat_upper']==tsv['U_cat_upper'], 'Ucat identity')
            require(new['extra_upper']==w['mean_extra_upper']==tsv['mean_extra_upper']
                    and new['epsilon']==tsv['epsilon']==str(Q(c['epsilon']))
                    and new['decision']==tsv['decision']=='SUFFICIENT_CONDITION_CONFIRMED'
                    and new['pi64_added_again'] is False, 'mean evidence identity')
            joined.append(dict(m=m,q=row['q'],r=row['r'],K=2,axis=axis,
                N=c['N'],epsilon=c['epsilon'],mu0=c['saved_mu0'],alpha=c['alpha'],
                U_cat_upper=c['U_cat_upper'],mean_extra_upper=w['mean_extra_upper']))
    return joined


def proof_connection(d, c):
    """Reuse archived conclusions after input identities, never recompute them."""
    require(len(d['rounds'])==18 and all(x['m']==m and x['q']==1<<m
            and x['decision']=='SUFFICIENT' for m,x in enumerate(d['rounds']))
            and d['saved_phase_budget_strict']==[True]*18, 'D geometry inventory')
    require(d['branch']['ideal_decision']=='SUFFICIENT'
            and len(d['branch']['ideal_margins'])==17, 'D 17 strict branch evidence')
    require(all(d['energy'][k]=='SUFFICIENT' for k in
                ('reference_alias_decision','representative_lift_decision','literal_target_decision')),
            'D alias/lift/energy evidence')
    require(c['decision']=='REPRESENTED_DF_SECTOR_GROUND_BRACKET'
            and c['positive_shift']['decision']=='STRICT_POSITIVE_DEFINITE'
            and c['positive_shift']['positive_pivots']==70
            and c['ideal_saved_DF_target_decision']=='SUFFICIENT'
            and c['reference_sector_eigenvalue_alias_decision']=='SUFFICIENT'
            and c['eta_ref_upper'] is not None, 'C reference/target evidence')
    return dict(G_to_mathematical_branch='ARCHIVED_PROOF_REUSABLE',
        mathematical_decoder_to_saved_DF_sector_energy='ARCHIVED_PROOF_REUSABLE',
        energy_upper=c['ideal_decoder_saved_DF_energy_upper'],
        energy_bound_origin='61f569c32a94c2b698df10a62a51e515b12160ed:C.json',
        external_alias_prior='ASSUMED', delta_mean_added_to_energy=False,
        Float64_decoder='UNKNOWN/STOP', real_molecular_energy='UNKNOWN/STOP')


def static_inputs():
    spec = json.loads((HERE/'FIXED_BINDINGS.json').read_text())
    require(spec['precision_bits']==160 and spec['passes']==1 and spec['target']=='1/20'
            and spec['signal_convention']==CONVENTION
            and spec['support_policy']=='support_for_source_basis_run_le_1'
            and spec['bound_scope']=='IID_CATEGORICAL_MEAN_ONLY_NOT_POINTWISE', 'fixed scope')
    raw = {k:base.verify_bound(v,current=v['current']) for k,v in spec['files'].items()}
    original = base.static_inputs()  # Existing byte/schema/literal path, not execute().
    ac,ai,di,d,w,a,c,b = [json.loads(raw[k]) for k in
        ('AC_RESULT','AC_INPUTS','D_INPUTS','D_RESULT','MEAN_RESULT','A_RESULT','C_RESULT','B_RESULT')]
    require(ac['SOURCE']['commit']=='d29a86963f91044a95217d7d2e2ee9703d2f1dc1'
            and ac['status']=='COMPLETED_MANDATORY_STOP' and ac['selected_tasks']==['A','C']
            and ac['A']==a and ac['C']==c, 'A/C provenance')
    require(d['SOURCE']['commit']=='6942e5176d2dec115ac3c271e48f84377abda915'
            and ai['constants']==di['constants'] and ai['files']==ac['inputs'], 'D/A/C identity')
    for document in (ac,d,w):
        require(all(document[k]==v for k,v in STATUS.items()), 'formal status preserved')
    for entry in ai['files'].values():
        base.verify_bound(entry)  # Source/evidence bytes only, including H/state NPZ.
    require(ai['files']['D_RESULT']['sha256']==spec['files']['D_RESULT']['sha256']
            and di['files']==d['inputs'], 'D result inputs')
    for key,dkey in (('ROUNDS','FIXED_ROUNDS'),('CAT_AXES','AXES')):
        require(ai['files'][key if key=='ROUNDS' else 'AXES']['sha256']
                ==d['inputs'][dkey]['sha256']==spec['files'][key]['sha256'], 'same G input files')
    for key in ('H','STATE'):
        require(ai['files'][key]['sha256']==original['manifest']['files'][key]['sha256'],
                'same saved Hamiltonian/state')
    require(str(b['physical_state']['energy'])==ai['constants']['energy_reference_binary64_decimal']
            and b['physical_state']['fingerprint']==ai['state_fingerprint'], 'common Eref/state identity')
    require(w['source']['commit']=='7a39a78eccdbe457786978e14d36b2f0d7abc290'
            and w['status']=='COMPLETED_MANDATORY_STOP'
            and w['decision']=='CONDITIONAL_IDEAL_CONTROLLED_IID_MEAN_LOCAL_CONDITIONS_CONFIRMED'
            and w['bound_scope']==spec['bound_scope'] and w['pi64_added_again'] is False,
            'weighted mean scope')
    parse = lambda key:list(csv.DictReader(io.StringIO(raw[key].decode()),delimiter='\t'))
    axes = validate_join(original['rows'],original['saved_axes'],w['rounds'],
                         parse('CAT_AXES'),parse('MEAN_AXES'),a['axes'])
    return dict(mode='FIXED_H4_CONTROLLED_IID',axes=axes,proof=proof_connection(d,c),
                inputs=spec['files'])


def alpha_module():
    sys.path.insert(0,str(ROOT/'docs/research/rpe_three_gaps_preparation_20261104'))
    import alpha_adapter
    return alpha_adapter


def controlled_axis(row, alpha):
    # Never use the already widened interval or total/shot upper as Ucat.
    u,delta = Q(row['U_cat_upper']),Q(row['mean_extra_upper'])
    require(u>=0 and delta>=0, 'nonnegative certified mean bounds')
    u_ctrl = u+delta
    value = alpha.failure_upper(int(row['N']),row['epsilon'],u_ctrl)
    require(value['decision']=='BOUND_AVAILABLE', 't not proven strictly positive')
    return dict(row,U_controlled_upper=str(u_ctrl),pi64_added_again=False,**value)


def evaluate(output,bundle,source=None):
    """Shared synthetic/future authorized science path; only 36 scalar tails."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    record = dict(STATUS,status='STARTED',decision='UNDETERMINED',mode=bundle['mode'],
        SOURCE=source,inputs=bundle.get('inputs'),precision_bits=160,scientific_passes=1,
        axes=[],counts={},aggregate=None,proof=bundle['proof'],pi64_added_again=False)
    started = time.monotonic()
    save = lambda:write_json(output/'RESULT.json',record)
    save()
    try:
        rows = bundle['axes']
        require(len(rows)==36 and {(x['m'],x['axis']) for x in rows}
                =={(m,a) for m in range(18) for a in AXES}, 'all36 tails required')
        from flint import ctx
        alpha = alpha_module()
        with ctx.workprec(160):
            for row in rows:
                record['counts']['tail_cells_started']=record['counts'].get('tail_cells_started',0)+1
                record['axes'].append(controlled_axis(row,alpha))
                save()
            record['counts']['exact_aggregations_started']=1
            record['aggregate']=alpha.aggregate(record['axes'],target='1/20')
        if record['aggregate']['decision']=='DIRECT_HOEFFDING_95_SUFFICIENT':
            record['decision']='CONDITIONAL_IDEAL_CONTROLLED_IID_MATHEMATICAL_ENERGY_95_SUFFICIENT'
            record['status']='COMPLETED_MANDATORY_STOP'
        else:
            record['status']='UNKNOWN_MANDATORY_STOP'
    except BaseException as exc:
        record['status']='UNKNOWN_MANDATORY_STOP'
        record['error']=type(exc).__name__+': '+str(exc)
    finally:
        record['elapsed_seconds']=time.monotonic()-started
        save()
    return record


def describe():
    bundle = static_inputs()
    return dict(STATUS,status='TECHNICAL_PREPARATION_ONLY',fixed_axes=len(bundle['axes']),
        precision_bits=160,new_H4_scientific_calls=0,controlled_success_probability=None,
        controlled_probability_decision='NOT_EXECUTED_UNDETERMINED',
        default_mode='bytes_schema_literals_only',proof=bundle['proof'])


def run_science(output):
    bundle = static_inputs()
    source = base.frozen_source()  # Existing runtime and base SOURCE checks.
    for path in HERE.iterdir():
        if path.suffix in ('.py','.json','.md'):
            require(path.read_bytes()==base.git('show',source['commit']+':'+path.relative_to(ROOT).as_posix()),
                    'publish/freeze new SOURCE before --run')
    output = Path(output).resolve()
    namespace = ROOT/'artifacts/rpe_controlled_iid_success_energy'
    require(output.is_relative_to(namespace) and output!=namespace, 'fresh result namespace required')
    bundle['resources'] = dict(affinity_cpus=len(os.sched_getaffinity(0)),
        available_memory=[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')],
        backend='36 scalar Arb160 tails and one exact-rational sum; no worker/GPU workload')
    source['resources']=bundle['resources']
    return evaluate(output,bundle,source)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    require(args.run==bool(args.output), '--run and fresh --output required together')
    result=run_science(args.output) if args.run else describe()
    print(json.dumps(result if not args.run else {k:result[k] for k in
        (*STATUS,'status','decision','counts')},ensure_ascii=False,indent=2))
    return 0 if not args.run or result['decision'].startswith('CONDITIONAL_') else 2


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (OSError,KeyError,ValueError,subprocess.CalledProcessError) as exc:
        print('UNKNOWN/STOP: '+str(exc),file=sys.stderr)
        raise SystemExit(2)
