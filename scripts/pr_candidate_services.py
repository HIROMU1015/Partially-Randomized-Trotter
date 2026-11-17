"""SOURCE-pinned metadata services for B2-selected PR candidates.

Run each service in a fresh process so DF/Pauli provider module graphs do not mix.
Numerical signal and compile work is delegated to the separate fixed workers.
"""
import argparse
import ast
from copy import deepcopy
from fractions import Fraction
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
DF_VIEW=ROOT.parent/'d2-pr-df-ld1-m3-acquisition-20261115'
SIGNAL_VIEW=ROOT.parent/'d2-pr-df-ld7-signal-source-fefd-20261116'
DF_SOURCE='326feadd093920bbea29b498108aa4eae6f67a75'
INDEX=dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')
DF_TEMPLATE=dict(commit='0defa286c53a1b9acf513be453907bcad83102b3',path='docs/research/d_pr_df_ld7_v0_results_20261116/generation/CONFIGURATION.json')


def read(loc):
    if isinstance(loc,dict)and 'commit'in loc:
        value=json.loads(subprocess.check_output(['git','show',loc['commit']+':'+loc['path']],cwd=ROOT))
        for k in loc.get('selector',[]):value=value[k]
        return value
    return deepcopy(loc)


def dec(x):return Fraction(int(x['numerator']),int(x['denominator']))if isinstance(x,dict)else Fraction(x)
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def module(view,name,path):
    sys.path[:0]=[str(view/'scripts'),str(view/'src')]
    s=importlib.util.spec_from_file_location(name,view/path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def df_extension():
    spec=importlib.util.spec_from_file_location('_df_supplied_descriptor',ROOT/'src/trottertracks/pr_df_descriptor_extension.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);return helper


def scope_df_saved_index(entry,configuration):
    """Validate with the fixed entry; unrelated rounds cannot conflict here.

    Every packet is indexed by the original implementation in isolation. Only
    the requested physical keys are merged; a conflict on a requested key
    retains the original strict record-equality STOP. No raw value is replaced.
    """
    wanted={entry.physical_key(job)for job in entry.runtime.jobs(configuration)}
    inventory=list(entry.saved);selected={}
    try:
        for source in inventory:
            entry.saved=[source];entry._index=None
            for key,item in entry.saved_index().items():
                if key not in wanted:continue
                if key in selected and entry.reader.read(selected[key]['source'])!=entry.reader.read(item['source']):
                    raise ValueError('Conflicting saved means: select explicit saved inventory')
                selected.setdefault(key,item)
    finally:
        entry.saved=inventory;entry._index=None
    entry._index=selected
    return entry


def df_entry(tail_descriptor=None,saved_bundles=(),configuration=None):
    # Same fixed entry as326fe; no source edits to its canonical/provider contracts.
    sys.path.insert(0,str(DF_VIEW/'src'))
    from trottertracks.pr_df_signal_entry import DFSignalEntry
    entry=DFSignalEntry(ROOT)if tail_descriptor is None else df_extension().entry(ROOT,tail_descriptor,DFSignalEntry)
    entry.saved.extend(x for x in saved_bundles if x not in entry.saved)
    if configuration is not None and not entry._support(configuration):scope_df_saved_index(entry,configuration)
    return entry


def df_generate(request):
    if request.get('configuration'):
        config=read(request['configuration']);df_entry(request.get('tail_descriptor')).runtime.b.validate(config)
        return dict(status='CONFIGURATION_READY',configuration=config)
    S=module(DF_VIEW,'_df_auto_generate_fixed','scripts/run_pr_df_ld7_v0.py');S.ROOT=ROOT
    _,_,_,b=S.signal_modules();base=read(request.get('base')or DF_TEMPLATE)
    design=request['design'];ld=design['L_D'];inputs=deepcopy(base['spec']['A_input'])
    inputs['design']['prefix']=ld
    if 'PF'in inputs['design']:inputs['design']['PF']['prefix']=ld
    if design.get('formula'):inputs['design']['formula']=design['formula']
    h=Fraction(inputs['design']['base_time'])
    delta=dec(design['delta_pf'])if design.get('delta_pf')is not None else h*dec(design['delta_over_h'])
    if Fraction(float(delta))!=delta:
        return dict(status='UNSUPPORTED',reason='REQUESTED_DF_DELTA_NOT_EXACT_BINARY64',requested=str(delta),
                    requested_delta_pf=deepcopy(design.get('delta_pf')),requested_delta_over_h=deepcopy(design.get('delta_over_h')))
    inputs['design']['delta']=float(delta)
    inputs['kappa_paper']=str(dec(design['kappa']))  # absolute, never a multiplier
    inputs['design']['K']=design.get('K',base['spec']['finite_K'])
    index=read(INDEX);desc=index['T20_reference_files']['H4']
    raw=Path(request.get('controller_paths',{}).get('DF_DESCRIPTOR',desc['path'])).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=desc['sha256']:raise ValueError('Saved DF descriptor bytes differ')
    context=json.loads(raw)
    if type(ld)is not int or not 1<=ld<context['binding']['rank']:
        return dict(status='UNSUPPORTED',reason='NONEMPTY_DF_HEAD_TAIL_REQUIRED',L_D=ld)
    if request.get('tail_descriptor'):
        source=request['tail_descriptor'];document=read(source)
        provisional=deepcopy(base);provisional['spec']['L_D']=ld
        tail=df_extension().validate(document,source,provisional)
    elif str(ld)not in context['prefixes']:
        return dict(status='NEEDS_DF_TAIL_DESCRIPTOR',reason='FIXED_SIGNAL_PROVIDER_HAS_NO_SAVED_PREFIX_DESCRIPTOR',L_D=ld,
                    available_prefixes=sorted(map(int,context['prefixes'])),descriptor_index=INDEX,
                    required=['DERIVE_PREFIX_SPECIFIC_DF_LAMBDA_AND_sR_FROM_SAVED_INPUT',
                              'BIND_NEW_DESCRIPTOR_SOURCE_TO_CONFIGURATION_AND_PROVIDER_CONTEXT'],configuration=None)
    else:
        tail=context['prefixes'][str(ld)]
        source=dict(index=INDEX,context_sha256=desc['sha256'],selector=['prefixes',str(ld)])
    inputs['tail_norm'].update(prefix=ld,lambda_R=tail['lambda_R'],source=source)
    try:config=b.generate(b.a_spec(inputs,'PR'))
    except ValueError as exc:
        if 'Nonintegral PF grid; no implicit delta change'not in str(exc):raise
        return dict(status='UNSUPPORTED',reason='DF_MAPPER_REJECTED_REQUESTED_STEP_WIDTH',mapper_reason=str(exc),
                    requested_delta_pf=deepcopy(design.get('delta_pf')),requested_delta_over_h=deepcopy(design.get('delta_over_h')),
                    requested_exact=dict(numerator=delta.numerator,denominator=delta.denominator))
    b.validate(config)
    if dec(config['spec']['kappa']['value'])!=dec(design['kappa']):raise ValueError('DF absolute kappa changed')
    if config['spec']['L_D']!=ld:raise ValueError('DF prefix changed')
    return dict(status='CONFIGURATION_READY',configuration=config,tail_descriptor=tail,descriptor_source=source)


def main():
    global DF_VIEW,SIGNAL_VIEW
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('df-generate','df-plan','df-stage','df-bundle','df-price-design','pauli-price-plan','quote','intake'))
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.loads(a.input.read_text())
    # Deployment paths only. Existing provider SOURCE/hash/runtime guards still
    # validate each relocated checkout and descriptor before scientific work.
    paths=r.get('controller_paths',{})
    DF_VIEW=Path(paths.get('DF_VIEW',DF_VIEW));SIGNAL_VIEW=Path(paths.get('SIGNAL_VIEW',SIGNAL_VIEW))
    if a.mode=='df-generate':value=df_generate(r)
    elif a.mode=='df-plan':
        config=read(r['configuration']);value=df_entry(r.get('tail_descriptor'),r.get('saved_bundles',[]),config).plan(config)
    elif a.mode=='df-stage':
        config=read(r['configuration']);value=df_entry(r.get('tail_descriptor'),r.get('saved_bundles',[]),config).stage(config,ROOT/r['directory'],obtained={int(m):v for m,v in r.get('obtained',{}).items()})
    elif a.mode=='df-bundle':
        config=read(r['configuration_source']);entry=df_entry(r.get('tail_descriptor'),r.get('saved_bundles',[]),config);value=entry.assemble(r['configuration_source'],{int(m):v for m,v in r['raw_sources'].items()})
    elif a.mode=='df-price-design':
        spec=importlib.util.spec_from_file_location('_df_candidate_price_design',ROOT/'scripts/run_pr_df_candidate_prices.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        value=helper.design_groups(read(r['configuration']),r.get('price_quote'))
        for group in value.get('groups',[]):group.get('design',{}).pop('matrix',None)
    elif a.mode=='pauli-price-plan':
        sys.path.insert(0,str(ROOT/'src'))
        from trottertracks import pauli_calibration_request as P
        config=read(r['configuration'])
        preparation=r.get('prepared_item')or P.canonical_preparation(config,r.get('configuration_source'))
        value=P.prepare(Path(r.get('workspace',ROOT)),preparation,r['price_quote'])
    elif a.mode=='quote':
        view=Path(r['view']);sys.path.insert(0,str(view/'src'))
        from trottertracks.pr_price_rules import PriceRules,GitReader
        registry=json.loads(Path(r['registry_file']).read_text())
        rules=PriceRules(GitReader(ROOT),registry)
        config=read(r['configuration']);method='PR_PAULI'if config['spec']['split'].get('tail_dictionary')else'PR_DF'
        implementation=r.get('implementation')
        if implementation is None:
            if method=='PR_DF':implementation=next(e['implementation']for e in registry['entries']if e['name']=='DF_LD7')
            else:
                raw=subprocess.check_output(['git','show','02e7cc6a1:src/trottertracks/pauli_calibration_request.py'],cwd=ROOT)
                tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='implementation')
                ns=dict(LOWERING='PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF',LEGACY='13ab1acfaebe5f33958e1db67862f5c45725ab91')
                exec(compile(ast.Module(body=[node],type_ignores=[]),'fixed:A_candidate_implementation','exec'),ns);implementation=ns['implementation']()
        value=rules.quote(dict(configuration=config,implementation=implementation,policy=r.get('policy','MODEL')))
        value['implementation']=implementation
    else:
        view=Path(r['view']);sys.path[:0]=[str(view/'scripts'),str(view/'src')]
        if r['method']=='PR_PAULI':
            from trottertracks import rpe_pauli_signal_intake as A,rpe_total_binding as T
            got=A.accept_saved(view,integration=T.C.interfaces(str(view)).pipeline,handoff_source=r['packet'])
            item=got['results'][0];binding=T.create(view,item['configuration'],original_source=r['configuration']);T.bind_signal(view,binding,item['signal_packet'])
        else:
            adapter=module(ROOT,'_portable_prdf_intake','scripts/run_rpe_optimizer_q3_job.py')
            W,_=adapter.load_prdf_route(view,r.get('controller_paths',{}).get('route_views',{}))
            config=read(r['configuration']);binding,_=W.U.T.create(view,config,original_source=r['configuration']),None
            case=dict(configuration=r['configuration'],original_configuration_id=config['id'],signal_packet=r['packet'],
                signal_packet_kind='A_BUNDLE_REQUIRES_EXISTING_K_ACCEPT_SAVED',binding_id=binding['id'],total_configuration_id=binding['total_configuration']['id'])
            W.U.metadata(case)
        value=dict(status='INTAKE_AND_TOTAL_VIEW_READY',configuration=r['configuration'],signal_packet=r['packet'],
            binding_id=binding['id'],total_configuration_id=binding['total_configuration']['id'],statistics_executed=False)
    put(a.output,value)


if __name__=='__main__':main()
