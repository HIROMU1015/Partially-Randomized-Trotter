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
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(globals().get('FIXED_RUN_CODE_ROOT', Path(__file__).resolve().parents[1])).resolve()
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


def expose_intake_objects():
    """Share imported objects with this intake process and its fixed adapters."""
    objects = subprocess.check_output(
        ['git', 'rev-parse', '--path-format=absolute', '--git-path', 'objects'],
        cwd=ROOT, text=True,
    ).strip()
    alternates = os.environ.get('GIT_ALTERNATE_OBJECT_DIRECTORIES')
    direct = json.dumps(objects, ensure_ascii=False)
    os.environ['GIT_ALTERNATE_OBJECT_DIRECTORIES'] = os.pathsep.join(
        value for value in (direct, alternates) if value
    )


def check_intake_inputs(view, locators):
    env = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_TERMINAL_PROMPT='0')
    for locator in locators:
        ref = f"{locator['commit']}:{locator['path']}"
        subprocess.run(['git', 'cat-file', '-e', ref], cwd=view, env=env, check=True)


def df_extension():
    path = Path(__file__).resolve().parents[1]/'src/trottertracks/pr_df_descriptor_extension.py'
    spec=importlib.util.spec_from_file_location('_df_supplied_descriptor',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);return helper


# The fixed stage() rebinds these canonical labels after matching the physical
# key and checking each record against its original job. The lineage fields
# describe that rebind. Numerical values, recipe and input/reference evidence
# remain in the comparison.
DF_DECLARATION_FIELDS = frozenset({
    'B_configuration_id', 'B_candidate_id', 'B_identity', 'B_law_binding', 'reuse', 'horizon_origin',
})


def df_record_differences(left, right):
    fields = (left.keys() | right.keys()) - DF_DECLARATION_FIELDS

    def exact(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

    return {key for key in fields if key not in left or key not in right
            or exact(left[key]) != exact(right[key])}


def fixed_locator(ref):
    if not isinstance(ref, dict) or not ref.get('path'):
        return False
    commit = ref.get('commit')
    return isinstance(commit, str) and re.fullmatch(r'[0-9a-f]{40}', commit) is not None


def index_reissued_df_packet(entry, source, packet):
    """Verify the original carriers before passing the normal packet to fixed intake."""
    packet = deepcopy(packet)
    if [mean['m'] for mean in packet['means']] != list(range(14)):
        raise ValueError('Reissued DF horizon must have all fourteen rounds')
    reader = entry.reader.read
    for mean in packet['means'][:13]:
        origin = mean.pop('horizon_origin', None)
        record = reader(mean['source'])
        if (not fixed_locator(origin) or record.get('horizon_origin') != origin
                or df_record_differences(record, reader(origin))):
            raise ValueError('Reissued DF carrier differs from its source at m=' + str(mean['m']))
    entry.reader.read = lambda loc: deepcopy(packet) if loc == source else reader(loc)
    try:
        return entry.saved_index()
    finally:
        entry.reader.read = reader


def validated_packet_index(entry, source, cache_dir, scope):
    entry.saved = [source]
    entry._index = None
    packet = entry.reader.read(source)
    # DF's fixed validator expects the assembled top-level means bundle.
    # Pauli packets use a different envelope and must not reach that validator
    # (or a cached index) through a mixed-method continuation inventory.
    labels = ' '.join(str(packet.get(name, '')) for name in ('method', 'kind', 'schema')) if isinstance(packet, dict) else ''
    if (not isinstance(packet, dict) or 'PAULI' in labels.upper()
            or packet.get('method') in ('FULL_PF', 'TRUNCATED_PF')
            or not isinstance(packet.get('configuration_source'), dict)
            or not isinstance(packet.get('means'), list)
            or any(not isinstance(mean, dict) or 'source' not in mean for mean in packet['means'])):
        raise ValueError('DF_SAVED_BUNDLE_EXPECTED: saved_df_bundles requires a DF bundle with configuration_source and top-level means: '+str(source))
    if any('horizon_origin' in mean for mean in packet['means']):
        return index_reissued_df_packet(entry, source, packet)
    if cache_dir is None or not fixed_locator(source):
        return entry.saved_index()
    cache_dir = Path(cache_dir)
    key = hashlib.sha256(json.dumps([scope, source], sort_keys=True).encode()).hexdigest()
    path = cache_dir / (key + '.json')
    try:
        saved = json.loads(path.read_bytes())
        if saved['key'] == key and isinstance(saved['index'], dict):
            return saved['index']
    except (OSError, ValueError, TypeError, KeyError):
        pass
    index = entry.saved_index()
    if not (fixed_locator(packet['configuration_source'])
            and all(fixed_locator(mean['source']) for mean in packet['means'])):
        return index
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', dir=cache_dir, delete=False) as out:
            try:
                json.dump(dict(key=key, index=index), out, allow_nan=False)
                out.flush()
                Path(out.name).replace(path)
            finally:
                Path(out.name).unlink(missing_ok=True)
    except OSError:
        pass  # The original fixed validation remains authoritative.
    return index


def scope_df_saved_index(entry, configuration, *, cache_dir=None, scope=None):
    """Use the fixed validator for every packet before comparing duplicates."""
    wanted = {entry.physical_key(job): job['m'] for job in entry.runtime.jobs(configuration)}
    inventory = list(entry.saved)
    if cache_dir is not None:
        scope = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    selected, alternatives, differences, first_records = {}, {}, {}, {}
    try:
        for source in inventory:
            for key, item in validated_packet_index(entry, source, cache_dir, scope).items():
                if key not in wanted:
                    continue
                first = selected.setdefault(key, item)
                evidence = alternatives.setdefault(key, [])
                if item in evidence:
                    continue
                evidence.append(item)
                if first['source'] != item['source']:
                    if key not in first_records:
                        first_records[key] = entry.reader.read(first['source'])
                    fields = df_record_differences(first_records[key], entry.reader.read(item['source']))
                    if fields:
                        differences.setdefault(key, set()).update(fields)
    finally:
        entry.saved = inventory
        entry._index = None
    entry._index = {key: item for key, item in selected.items() if key not in differences}
    return {wanted[key]: dict(m=wanted[key], physical_key=key, differing_fields=sorted(fields),
                              sources=alternatives[key])
            for key, fields in differences.items()}


def bind_df_conflicts(entry, conflicts):
    # The fixed stage and bundle paths also consult self.plan(). Pass the
    # affected rounds through their existing UNSUPPORTED guard.
    original_plan = entry.plan

    def plan(configuration):
        result = original_plan(configuration)
        for row in result['rounds']:
            if row['m'] not in conflicts:
                continue
            row.pop('request', None)
            row.pop('saved', None)
            row['decision'] = 'UNSUPPORTED'
            row['reasons'] = [*row.get('reasons', []), 'DF_SAVED_SIGNAL_CONFLICT']
        result['counts'] = {name: sum(row['decision'] == name for row in result['rounds'])
                            for name in result['counts']}
        result['unsupported_rounds'] = [row['m'] for row in result['rounds']
                                        if row['decision'] == 'UNSUPPORTED']
        result['saved_signal_conflicts'] = [conflicts[m] for m in sorted(conflicts)]
        return result

    entry.plan = plan


def df_entry(tail_descriptor=None,saved_bundles=(),configuration=None,cache_dir=None,terminal_reference=None):
    # Same fixed entry as326fe; no source edits to its canonical/provider contracts.
    sys.path.insert(0,str(DF_VIEW/'src'))
    from trottertracks.pr_df_signal_entry import DFSignalEntry,FixedRuntime
    runtime=FixedRuntime(ROOT,SIGNAL_VIEW)
    path_spec=importlib.util.spec_from_file_location('_candidate_preparation_paths',ROOT/'src/trottertracks/rpe_preparation_paths.py')
    path_module=importlib.util.module_from_spec(path_spec)
    path_spec.loader.exec_module(path_module)
    from trottertracks.pr_df_signal_entry import SCALAR
    descriptor=read(INDEX)['T20_reference_files']['H4']
    data=Path(path_module.resolve(descriptor['path'])).read_bytes()
    if hashlib.sha256(data).hexdigest()!=descriptor['sha256']:
        raise ValueError('Saved tail descriptor differs')
    metadata=json.loads(data)
    metadata['descriptor_sha256']=descriptor['sha256']
    metadata['hamiltonian_constant']=read(SCALAR)['hamiltonian_constant']
    class RelocatedEntry(DFSignalEntry):
        def __init__(self,root,**kwargs):
            super().__init__(root,metadata=metadata,**kwargs)
    DFSignalEntry=RelocatedEntry
    entry=(DFSignalEntry(ROOT,runtime=runtime)if tail_descriptor is None else
           df_extension().entry(ROOT,tail_descriptor,DFSignalEntry,runtime=runtime))
    entry.saved.extend(x for x in saved_bundles if x not in entry.saved)
    if terminal_reference is not None:
        if configuration is None or configuration['spec']['M'] != 13:
            raise ValueError('Terminal reference requires a fourteen-round DF configuration')
        provider = runtime.provider
        reference = df_extension().terminal_reference
        runtime.provider = SimpleNamespace(numerical_recipe=provider.numerical_recipe,
            saved_h4_reference=lambda root, job: reference(root, terminal_reference, job, provider))
        original_plan = entry.plan

        def terminal_plan(config):
            return dict(original_plan(config), terminal_reference=terminal_reference)

        entry.plan = terminal_plan
    if configuration is not None and not entry._support(configuration):
        scope = None
        if cache_dir is not None:
            views = (DF_VIEW, SIGNAL_VIEW)
            # Cache only immutable Git locators validated by unmodified views.
            if ((tail_descriptor is None or fixed_locator(tail_descriptor))
                    and all(subprocess.run(['git','diff','--quiet','HEAD','--','src'],cwd=view).returncode == 0
                            for view in views)):
                scope = dict(views=[subprocess.check_output(['git','rev-parse','HEAD'],cwd=view,text=True).strip()
                                    for view in views], descriptor=tail_descriptor,
                             caller=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             extension=hashlib.sha256((ROOT/'src/trottertracks/pr_df_descriptor_extension.py').read_bytes()).hexdigest(),
                             validated_metadata=entry.info)
            else:
                cache_dir = None
        conflicts = scope_df_saved_index(entry, configuration, cache_dir=cache_dir, scope=scope)
        if conflicts:
            bind_df_conflicts(entry, conflicts)
    return entry


def df_generate(request):
    if request.get('configuration'):
        config = read(request['configuration'])
        b = df_entry(request.get('tail_descriptor')).runtime.b
        b.validate(config)
        horizon = request.get('M', config['spec']['M'])
        if horizon != config['spec']['M']:
            if (config['spec']['M'], horizon) != (12, 13):
                raise ValueError('Expected an existing twelve-to-thirteen DF horizon extension')
            inputs = deepcopy(config['spec']['A_input'])
            inputs['M'] = horizon
            extended = b.generate(b.a_spec(inputs, 'PR'))
            b.validate(extended)
            fields = ('physical_candidate', 'structure', 'T', 'actual_delta_pf', 'q_PF', 'R_outer')
            if any(before[key] != after[key] for before, after in zip(config['rounds'], extended['rounds'])
                   for key in fields):
                raise ValueError('DF prefix physical schedule changed during horizon extension')
            config = extended
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


def df_price_disposition(config, quote, runner):
    """Keep valid prices and choose the existing minimum acquisition scope."""
    plan = runner.design_groups(config, quote)
    rows = {(r['m'], r['axis']): r for r in quote['rows']}
    scoped = {axis for axis, row in rows.items()
              if row.get('unit_price_RZ') is None and (row.get('needs') or {}).get('entry_ids')}
    actions = {(r['m'], r['axis']): r for r in plan['round_actions']}
    for index, action in actions.items():
        row = rows[index]
        needs = row.get('needs') or {}
        action['physical_id'] = row['physical_id']
        if row.get('source'):
            action['source'] = row['source']
        if needs:
            action['quote_needs'] = needs
    groups = []
    for group in plan['groups']:
        pending = {(r['m'], r['axis']) for r in group['axis_keys']}
        if (group['status'] == 'CALIBRATION_DESIGN_READY' and not pending & scoped
                and not any(rows[k].get('direct_alternatives') for k in pending)):
            groups.append(group)
            continue
        for m in sorted({m for m, _ in pending}):
            axes = [axis for axis in ('X', 'Y') if (m, axis) in pending]
            companion = [axis for axis in ('X', 'Y') if axis not in axes]
            design = runner.design(config, [m])
            direct_conflicts = [axis for axis in axes if
                                (rows[m, axis].get('needs') or {}).get('source_kind') == 'DIRECT_RECEIPTS']
            reasons = design['reasons']
            if direct_conflicts:
                # This provider finalizes X/Y from the same finite cohort.
                # An ambiguous direct axis cannot be used as a frozen companion.
                for axis in axes:
                    reason = ('MULTIPLE_ADMITTED_DIRECT_COHORTS_REQUIRE_SOURCE_SELECTION'
                              if axis in direct_conflicts else 'DIRECT_PAIR_REQUIRES_RESOLVED_COMPANION')
                    actions[m, axis].update(action='NO_SAFE_PRICE_ACTION', group_id=None, reasons=[reason])
                continue
            if reasons:
                for axis in axes:
                    actions[m, axis].update(action='NO_SAFE_PRICE_ACTION', group_id=None, reasons=reasons)
                continue
            round_row = config['rounds'][m]
            identity = dict(family={k: config['spec'][k] for k in ('binding', 'split', 'formula', 'finite_K', 'law', 'profile')},
                            physical=round_row['physical_candidate'], structure=round_row['structure'])
            ident = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            groups.append(dict(group_id=ident, actual_step=round_row['actual_delta_pf'], covered_rounds=[m],
                               axis_keys=[dict(m=m, axis=axis) for axis in axes],
                               required_saved_companions=companion, direct_resolution=True,
                               design={k: v for k, v in design.items() if k != 'matrix'},
                               status='CALIBRATION_DESIGN_READY'))
            for axis in axes:
                actions[m, axis].update(action='VERIFY_SAVED_PATHS_THEN_ACQUIRE_MISSING', group_id=ident, reasons=[])
    return dict(plan, groups=groups, round_actions=list(actions.values()))


def main():
    global DF_VIEW,SIGNAL_VIEW
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('df-generate','df-plan','df-stage','df-bundle','df-price-design','pauli-price-plan','quote','intake'))
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.loads(a.input.read_text())
    # Deployment paths only. Existing provider SOURCE/hash/runtime guards still
    # validate each relocated checkout and descriptor before scientific work.
    paths=r.get('controller_paths',{})
    mapping = json.loads(os.environ.get('RPE_PREPARATION_PATH_MAP', '{}'))
    mapping.update(r.get('preparation_path_map') or {})
    os.environ['RPE_PREPARATION_PATH_MAP'] = json.dumps(mapping)
    DF_VIEW=Path(paths.get('DF_VIEW',DF_VIEW));SIGNAL_VIEW=Path(paths.get('SIGNAL_VIEW',SIGNAL_VIEW))
    def signal_entry(config):
        return df_entry(r.get('tail_descriptor'), r.get('saved_bundles', []), config,
                        r.get('saved_index_cache'), r.get('terminal_reference'))

    if a.mode=='df-generate':value=df_generate(r)
    elif a.mode=='df-plan':
        config=read(r['configuration']);value=signal_entry(config).plan(config)
    elif a.mode=='df-stage':
        config=read(r['configuration']);value=signal_entry(config).stage(config,ROOT/r['directory'],obtained={int(m):v for m,v in r.get('obtained',{}).items()})
    elif a.mode=='df-bundle':
        config=read(r['configuration_source']);value=signal_entry(config).assemble(r['configuration_source'],{int(m):v for m,v in r['raw_sources'].items()})
    elif a.mode=='df-price-design':
        spec=importlib.util.spec_from_file_location('_df_candidate_price_design',ROOT/'scripts/run_pr_df_candidate_prices.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        config=read(r['configuration'])
        value=(df_price_disposition(config,r['price_quote'],helper) if r.get('resolve_axes')
               else helper.design_groups(config,r.get('price_quote')))
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
        inputs = (r['configuration'], r['packet'])
        expose_intake_objects()
        check_intake_inputs(view, inputs)
        if r['method']=='PR_PAULI':
            from trottertracks import rpe_pauli_signal_intake as A,rpe_total_binding as T
            got=A.accept_saved(view,integration=T.C.interfaces(str(view)).pipeline,handoff_source=r['packet'])
            item=got['results'][0];binding=T.create(view,item['configuration'],original_source=r['configuration']);T.bind_signal(view,binding,item['signal_packet'])
        else:
            adapter=module(ROOT,'_portable_prdf_intake','scripts/run_rpe_optimizer_q3_job.py')
            W,_=adapter.load_prdf_route(view,r.get('controller_paths',{}).get('route_views',{}))
            check_intake_inputs(W.U.ROOT, inputs)
            config=read(r['configuration']);binding,_=W.U.T.create(view,config,original_source=r['configuration']),None
            case=dict(configuration=r['configuration'],original_configuration_id=config['id'],signal_packet=r['packet'],
                signal_packet_kind='A_BUNDLE_REQUIRES_EXISTING_K_ACCEPT_SAVED',binding_id=binding['id'],total_configuration_id=binding['total_configuration']['id'])
            W.U.metadata(case)
        value=dict(status='INTAKE_AND_TOTAL_VIEW_READY',configuration=r['configuration'],signal_packet=r['packet'],
            binding_id=binding['id'],total_configuration_id=binding['total_configuration']['id'],statistics_executed=False)
    put(a.output,value)


if __name__=='__main__':main()
