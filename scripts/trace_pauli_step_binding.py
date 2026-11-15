"""Replay fixed A/B step metadata; no signal, circuit, fit or Q3 evaluation."""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from trottertracks import rpe_optimization_pipeline as I
from trottertracks.r_price_phase1_science import functions

ROOT = Path(__file__).resolve().parents[1]
PLAN = dict(commit='08812cd30cf46c73346695cc20b6552c8359314a',
    path='docs/research/pauli_signal_entry_20261116/PLANS.json')
PACKETS = {
    3: dict(commit='cdbce383f9944b7f9da1cbbd7851406dbbaae88a',
        path='docs/research/df_head_pauli_tail_ld3_signals_20261116/MEAN_HANDOFF.json'),
    5: dict(commit='ac46d100e74f3f56b4a3ff4d95c15fdc7fe8ca98',
        path='docs/research/df_head_pauli_tail_ld5_signals_20261116/MEAN_HANDOFF.json'),
}
ENTRY = dict(commit='93a5ef3b9fed337cc4d26ee684defd304fa0eac0',
    path='src/trottertracks/pauli_signal_entry.py')


def read(ref):
    return json.loads(I.C.blob(ROOT, ref['commit'], ref['path']))


def trace():
    B=I.providers(str(ROOT)).B
    helper=I._pauli_helpers(str(ROOT))
    bapi=SimpleNamespace(validate=B._validate,exact_scalar=B.a.exact_scalar)
    scope=dict(hashlib=hashlib, BINDING_NOT_PHYSICAL=('compiler','profile','profile_is_reference_only'),
        canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False))
    functions(ROOT,ENTRY['commit'],ENTRY['path'],('physical_key',),scope)
    key=scope['physical_key']
    plans=read(PLAN); rows=[]
    for prefix,ref in PACKETS.items():
        packet=read(ref)['packets'][0]; original=packet['configuration']
        configs={}
        for name,width in (('exact_1_5',Fraction(1,5)),('binary64_0_2',Fraction(.2))):
            spec=deepcopy(original['spec']);spec['delta_pf']=B.a.exact_value(width)
            configs[name]=B.generate(spec)  # metadata only; never re-ceil using float
        for name,config in configs.items():
            plan_name='LD%d delta_pf_%s' % (prefix,'1/5' if name=='exact_1_5' else 'binary64_0.2')
            saved_plan=plans[plan_name]
            if config['id']!=saved_plan['configuration_id']:
                raise ValueError('B-generated metadata differs from the fixed A plan')
            other=configs['binary64_0_2' if name=='exact_1_5' else 'exact_1_5']
            for row,pr in zip(config['rounds'],saved_plan['rounds']):
                m=row['m']; source=packet['rounds'][m]['mean_record']['source']
                job=helper.job(config,m,source,bapi);ctx=job['physical_context']; c=ctx['physical_candidate']
                old_ctx=packet['rounds'][m]['mean_record']['physical_context']
                other_ctx=helper.job(other,m,source,bapi)['physical_context']
                facts=I.pauli_round_binding(B,config,row,ctx)
                if any(c[k]!=pr['job'][field] for k,field in
                    (('q','q_PF'),('T','T'),('delta','delta'),('formula','formula'),('K','K'),('R_outer','R_outer'))):
                    raise ValueError('A saved job metadata differs from replay')
                same_saved=key(ctx)[0]==key(old_ctx)[0]
                if (pr['decision']=='REUSE')!=same_saved:
                    raise ValueError('Saved physical-key reuse differs from replay')
                time=c['T']/c['q']; count=row['structure']['allocation'][0]
                rows.append(dict(prefix=prefix,case=name,configuration_id=config['id'],**facts,
                    physical_candidate=deepcopy(c),structure=deepcopy(row['structure']),
                    pieces=[dict(index=0,weight_hex=float(1).hex(),event_positions=count,
                        A_signed_time_hex=(c['T']*1./c['q']).hex(),D_signed_time_hex=time.hex(),
                        event_time_hex=(time/count).hex(),head_half_time_hex=(time/2).hex())],
                    A_saved_plan=dict(PLAN,selector=[plan_name,'rounds',m]),
                    A_support=dict(decision=pr['decision'],reasons=pr['provider_reasons']),
                    saved_packet=dict(ref,selector=['packets',0,'rounds',m]),
                    saved_mean_source=deepcopy(packet['rounds'][m]['mean_source']) if same_saved else None,
                    same_physical_key_as_saved_delta_0_1=same_saved,
                    same_physical_key_as_other_width_case=key(ctx)[0]==key(other_ctx)[0],
                    required_evidence=[] if facts['matched'] else [
                        'SOURCE-bound signal provider supporting this exact canonical T/q without changing q/law',
                        'same-q/R/step/piece/scalar/wrapper/lowering/axis price evidence; not rounded-width substitution'],
                    price_record_accepted=False))
    return dict(schema='PAULI_EXACT_STEP_BINDING_TRACE_V1',rows=rows,
        sources=dict(B_example=dict(commit='ff9d25d62ada75fc56e89540459c15d49157b90c',
            path='docs/research/pauli_common_entry_intake_20261116/HANDOFF.md'),
            A_plan=PLAN,A_entry=ENTRY,A_job=dict(commit=I.A_PAULI_SOURCE,path=I.A_PAULI_JOB),packets=PACKETS),
        interpretation='Parameter metadata replay only; no new signal or accepted price is produced',
        new_science_calls=dict(signal=0,sampling=0,compile=0,fit=0,Q3=0),mandatory_stop=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(trace(),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(out)


if __name__=='__main__':main()
