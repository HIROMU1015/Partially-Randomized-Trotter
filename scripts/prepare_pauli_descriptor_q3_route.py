"""Prepare an existing Pauli packet for the already published descriptor intake.

This copies one saved REQUEST, preserving its science key, shots and evaluation
contract. It changes only the fixed passive intake route and host placement.
No signal provider, numerical evaluator, queue or worker is started.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import rpe_q3_coordinator as Q
from trottertracks import rpe_method_optimizer as O

INTAKE = '33e324b43690caeaa353ecdb9e3d42f3139198b6'
ENTRY = '6e4b6bd6a491815c2a3cb8f809cbfc0240b60fdb'
ENTRY_FILES = ('src/trottertracks/pauli_signal_entry.py', 'scripts/run_pauli_signal_entry.py',
               'src/trottertracks/pauli_split_descriptor.py')


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def prepare(request, profile, source_root, route_view, problem):
    if request['method'] != 'PR_PAULI' or request['route']['kind'] != 'pauli_b_common_entry':
        raise ValueError('Existing common-entry Pauli REQUEST required')
    if git(route_view, 'rev-parse', 'HEAD').decode().strip() != INTAKE:
        raise ValueError('Descriptor intake view HEAD differs')
    cid = request['configuration_id']; selected = request['route']['configurations'][cid]
    packet = selected['signal_packet']; entry_hashes = {}
    for path in ENTRY_FILES:
        expected = git(source_root, 'show', ENTRY+':'+path)
        actual = git(source_root, 'show', packet['commit']+':'+path)
        if actual != expected:
            raise ValueError('Saved descriptor assembly SOURCE differs: '+path)
        entry_hashes[path] = hashlib.sha256(actual).hexdigest()
    new = deepcopy(request)
    new['route'].update(view=str(Path(route_view).resolve()), view_commit=INTAKE,
        configuration_ids=[cid], configurations={cid:deepcopy(selected)},
        signal_entry_source=ENTRY, signal_intake_source=INTAKE,
        signal_packets_commit=packet['commit'],
        existing_caller='Existing fixed28d common adapter + published B33e descriptor intake; numerical workers unchanged')
    new['route'].pop('configurations_file', None)
    for key in request:
        if key != 'route' and new[key] != request[key]:
            raise ValueError('Scientific REQUEST field changed: '+key)
    new_profile = deepcopy(profile)
    new_profile.setdefault('route_views', {})[INTAKE] = str(Path(route_view).resolve())
    manifest = Q.source_manifest(source_root, request['optimizer_source'])
    assignment = dict(job_id=request['request_id'], method=request['method'],
        attempt_id='READINESS_ONLY_NO_EXECUTION_DESCRIPTOR_INTAKE', request=new,
        request_sha256=Q.digest(new), source_manifest=manifest,
        input_hashes=Q.input_hashes(source_root, new), prefix_key=Q.prefix_key(new))
    # Reuse the optimizer's real key builder, without preparing candidates or
    # creating an execution directory. Route view SHA is not a new science ID.
    executor = O.Executor(source_root, Path('READINESS_KEY_ONLY_NO_OUTPUT'),
        {request['method']:[request['route']]}, {}, problem, launch_disabled=True).for_method(request['method'])
    proposal = dict(configuration_id=cid, vector=O.interleave(request['N_X'],request['N_Y']))
    old_key = executor.job_dir(proposal).name
    executor.routes = {request['method']:[new['route']]}
    new_key = executor.job_dir(proposal).name
    if old_key != request['request_id'] or new_key != old_key:
        raise ValueError('Scientific job key changed')
    proof = dict(original_request_sha256=Q.digest(request), new_request_sha256=Q.digest(new),
        scientific_job_id_preserved=request['request_id'], configuration_id=cid,
        old_scientific_job_key=old_key, new_scientific_job_key=new_key,
        evaluation_contract_digest=executor.contract,
        shots_expected_contract_cost_preserved=True, computation_source=request['optimizer_source'],
        old_route_commit=request['route']['view_commit'], new_route_commit=INTAKE,
        assembly_implementation_source=ENTRY, unchanged_entry_file_sha256=entry_hashes,
        numerical_route_workers='UNCHANGED_EXISTING_CBC_AND_710_SOURCE_GUARDS',
        scope='PASSIVE_ROUTE_AND_EXISTING_SAVED_PACKET_ONLY_NO_SCIENTIFIC_EXECUTION',
        signal_packet_source=packet, input_locators=assignment['input_hashes'])
    return new, assignment, new_profile, proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True, type=Path)
    p.add_argument('--profile', required=True, type=Path)
    p.add_argument('--source-root', required=True, type=Path)
    p.add_argument('--route-view', required=True, type=Path)
    p.add_argument('--problem', type=Path, help='Same accuracy/target/rpe contract; defaults to the fixed source problem')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    problem = a.problem or a.source_root/'configs/rpe_method_optimizer/h4_df10_problem.json'
    values = prepare(json.loads(a.request.read_text()), json.loads(a.profile.read_text()), a.source_root, a.route_view,
                     json.loads(problem.read_text()))
    a.output.mkdir(parents=True, exist_ok=False)
    for name, value in zip(('REQUEST.json','ASSIGNMENT.json','PROFILE.json','ROUTE_CHANGE.json'), values):
        (a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output), request_sha256=values[1]['request_sha256'], scientific_calls=0)))


if __name__ == '__main__':
    main()
