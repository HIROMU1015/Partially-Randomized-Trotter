"""Saved JSON/hash audit only: never load molecular arrays or invoke science code."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); root=args.root.resolve()
    def read(p): return json.loads(p.read_text())
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    def write(name,value):
        with (args.output/name).open('x') as f:
            json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    execution='artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/'
    raw='artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/'
    old_commit='a87cf25548a3b93262027ce780317872d7c4e883'
    old=read(root/'artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/saved_execution_audit_v2.json')
    baseline=read(args.baseline)
    source_manifest=read(root/execution/'sealed_s4_mp_v1.json')
    for path,h in source_manifest['source_hashes'].items():
        assert sha(root/path)==h,path
        blob=subprocess.check_output(['git','show',source_manifest['source_commit']+':'+path],cwd=root)
        assert hashlib.sha256(blob).hexdigest()==h,path
    assert all(sha(root/p)==h for p,h in old['file_hashes'].items())
    units={}; file_hashes={}; cells={}; events={}; mp_records={}; missing=[]
    for unit,stem,expected in [('EVENT_CONTROL','event_control',(0,0,4)),('S4_MP','s4_mp',(2,4,0))]:
        directory=root/raw/(stem+'_launch_v1');manifest=read(root/execution/('sealed_'+stem+'_v1.json'))
        grant=read(root/execution/('authorization_'+stem+'_v1.json'))
        files=sorted(p for p in directory.rglob('*') if p.is_file())
        assert not any(p.name.startswith('.pending_') for p in files)
        size=sum(p.stat().st_size for p in files);assert size<=manifest['plan']['caps_proposed']['output_bytes']
        invalid={}
        for p in files:
            file_hashes[str(p.relative_to(root))]=sha(p)
            if p.suffix=='.json':
                try: read(p)
                except (ValueError,OSError) as error: invalid[p.name]=str(error)
        assert not invalid,invalid
        t=read(directory/'terminal_status.json');w=read(directory/'worker_terminal.json') if (directory/'worker_terminal.json').is_file() else None
        assert read(directory/'frozen_preparation.json')==manifest
        assert read(directory/'authorization.json')==grant
        assert grant['manifest_digest']==digest(manifest)
        binding={'manifest_digest':digest(manifest),'authorization_digest':digest(grant)}
        assert read(directory/'launch_binding.json')==binding
        claim=read(directory/'worker_claim.json')
        assert claim==dict(binding,assigned_cpu=1,retry=False,resume=False)
        assert manifest['assigned_resources']=={'assigned_cpu':1,'science_workers':1,'blas_threads':1}
        assert grant['retry'] is False and grant['resume'] is False
        assert grant['unit']==unit and manifest['unit']==unit
        assert grant['exclusive_output']==str(directory)
        if w:
            assert t['worker_terminal']==w
            assert w['compiled_wrappers']==0
            for key,value in [('N',None),('G',None),('numerical_allowance_certified',False),('accuracy_eligibility','UNDETERMINED'),('mandatory_stop',True),('next_stage_authorized',False),('H6_status','H6_NOT_AUTHORIZED'),('contract_status','DRAFT_NOT_AUTHORIZATION')]:
                assert w[key]==value and t[key]==value,(unit,key)
        correctness=sorted(directory.glob('*_correctness.json'))
        # Exact names below govern completeness.
        mp=sorted([*directory.glob('*_mp80.json'),*directory.glob('*_mp120.json')])
        explicit=sorted(directory.glob('*_explicit_order*.json'))
        if t['status']=='H4_SUPPLEMENT_COMPLETE':
            assert t['reason'] is None and t['worker_exit_code']==0
            assert (len(correctness),len(mp),len(explicit))==expected
            assert (w['completed_correctness_cells'],w['completed_mp_records'],w['completed_event_groups'])==expected
            assert w['primitive_completed']==537
            assert w['calls_reserved_before_work']=={'compile':0,'control_probe':100 if unit=='EVENT_CONTROL' else 0,'occurrence':0,'primitive':537,'reference_matvec':36,'trajectory':0}
        coverage=read(directory/'coverage_comparison_v3.json');assert coverage['equal'] is True and coverage['differences']==[]
        assert coverage['actual_sha256']==coverage['expected_sha256']
        assert coverage['actual_sha256']=='eacf9dec340e9b356090527597cfc1a277775350f050d5b50e2bb26e3c5e9609'
        reference=read(directory/'input_reference.json');primitive=read(directory/'primitive_validation.json')
        assert reference['reference_matvecs']==36
        assert reference['independent_occupation_all_columns_error']==old['input_reference_summary']['independent_occupation_all_columns_error']
        assert reference['reference']==old['input_reference_summary']['reference']
        assert primitive['actual_time_count']==179 and primitive['probe_count']==3 and primitive['certified'] is False
        progress=sorted(directory.glob('progress_*.json'));assert len(progress)<=1024
        for index,p in enumerate(progress):assert read(p)['progress_sequence']==index
        latest=read(progress[-1]);assert t['latest_progress']==latest
        for p in correctness:
            c=read(p);id_=p.name.removesuffix('_correctness.json')
            assert id_ in ('H4_B1_S4_q1','H4_B1_S4_q4')
            fields=('cell','signals','action_counts','log_B','reference_difference_mp80','reference_difference_mp120','precision_comparison','classical_cell_wall_seconds','worker_peak_rss_bytes')
            cells[id_]={k:c[k] for k in fields}
            cells[id_]['stage_summary']={}
            for dps in (80,120):
                v=c['stage_comparison_mp'+str(dps)];assert v['certified'] is False
                differences=[Decimal(x['state_difference']) for x in v['stages']]
                cells[id_]['stage_summary'][str(dps)]={'count':len(differences),'max_saved_state_difference':str(max(differences)), 'certified':False}
        for p in mp:
            v=read(p);assert v['dps'] in (80,120) and v['certified'] is False
            mp_records[p.name]={'path':str(p.relative_to(root)),'sha256':sha(p),'dps':v['dps'],'oracle_work':v['oracle_work'], 'trace_paths':sorted({x['path'] for x in v['trace']}),'trace_records':len(v['trace'])}
        for p in explicit:
            v=read(p);assert v['sampling_performed'] is False and v['full_molecular_event_mean_enumerated'] is False
            events[p.name]={'path':str(p.relative_to(root)),'sha256':sha(p),'error':v['error'],'algebraic_event_action_error':v['algebraic_event_action_error'], 'taylor_order':v['event']['taylor_order'],'phase':v['event']['phase'], 'selected_component_ids':v['event']['selected_component_ids'],'evidence_kind':'EMPIRICAL_REPRESENTATIVE_TECHNICAL_AGREEMENT'}
        units[unit]={'status':t['status'],'reason':t['reason'],'worker_exit_code':t['worker_exit_code'],'parent_wall_seconds':t['wall_seconds'], 'worker_log_bytes':t['worker_log_bytes'],'raw_files':len(files),'raw_bytes':size,'progress_records':len(progress), 'worker_peak_RSS_at_last_snapshot':latest['worker_peak_rss_bytes'],'peak_scope':'observed boundary snapshot; not an independently observed whole-process lifetime peak', 'launch_count':1,'grant_consumed':True,'retry':False,'resume':False,'counts':{'correctness':len(correctness),'mp':len(mp),'explicit':len(explicit)},'calls_reserved_before_work':w['calls_reserved_before_work'] if w else None,'coverage_comparison':coverage,'reference':reference,'primitive':primitive,'latest_progress':latest,'unit_manifest':str((root/execution/('sealed_'+stem+'_v1.json')).relative_to(root)), 'unit_grant':str((root/execution/('authorization_'+stem+'_v1.json')).relative_to(root))}
    for id_ in ('H4_B1_S4_q1','H4_B1_S4_q4'):
        for suffix in ('correctness','mp80','mp120'):
            p=root/raw/'s4_mp_launch_v1'/(id_+'_'+suffix+'.json')
            if not p.is_file():missing.append(str(p.relative_to(root)))
    for id_ in ('H4_B2_K2','H4_B3_K6'):
        for order in (0,2):
            p=root/raw/'event_control_launch_v1'/(id_+f'_explicit_order{order}.json')
            if not p.is_file():missing.append(str(p.relative_to(root)))
    # Preservation covers every original file, not only tracked files. Current
    # index introductions, if present, are stripped using exact recorded bytes.
    insertion_path=Path('/tmp/track_a_h4_supplement_execution_index_insertions_v1.json')
    insertions=read(insertion_path) if insertion_path.is_file() else {}
    changed=[]
    for p,h in baseline['files'].items():
        data=(root/p).read_bytes()
        if p in insertions:
            own=bytes.fromhex(insertions[p]['inserted_hex']);assert data.count(own)==1
            data=data.replace(own,b'',1)
        if hashlib.sha256(data).hexdigest()!=h:changed.append(p)
    assert not changed,changed
    root_repo=root.parents[1]
    for p,h in baseline['root_review_files'].items():assert sha(root_repo/p)==h,p
    audit={'schema':'track_a_ax2b_h4_supplement_saved_execution_audit_v1','created_utc':datetime.now(timezone.utc).isoformat(),
       'status':'SAVED_EXECUTION_CONSISTENCY_PASS','source_commit':source_manifest['source_commit'], 'seal_grant_commit':'5bbebb4d562ba9812eaf0838ba6419518c21d529', 'source_hashes_checked':len(source_manifest['source_hashes']), 'units':units,'new_correctness_summary':cells,'new_MP_records':mp_records,'new_explicit_summary':events,'new_file_hashes':file_hashes,'missing_records':missing,'old_result_commit':old_commit,'old_run_status_unchanged':old['science_run_status'],'old_raw_files_unchanged':len(old['file_hashes']),'protected_baseline_files_unchanged_except_exact_own_index_introductions':len(baseline['files']), 'root_review_files_unchanged':len(baseline['root_review_files']), 'scientific_acceptance_or_u_certification':False,'N':None,'G':None,'accuracy_eligibility':'UNDETERMINED','numerical_allowance_certified':False,'mandatory_stop':True,'next_stage_authorized':False,'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION'}
    union=[]
    old_raw='artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/'
    for id_ in old['correctness_summary']:
        union.append({'cell_id':id_,'run':'old_limited_v2','result_commit':old_commit, 'files':{suffix:{'path':old_raw+id_+'_'+suffix+'.json','sha256':old['file_hashes'][old_raw+id_+'_'+suffix+'.json']} for suffix in ('correctness','mp80','mp120')}})
    for id_ in cells:
        union.append({'cell_id':id_,'run':'supplement_s4_mp_launch_v1','result_commit_locator':'commit adding this union and the referenced new raw paths; fixed SHA in subsequent remote receipt', 'files':{suffix:{'path':raw+'s4_mp_launch_v1/'+id_+'_'+suffix+'.json','sha256':file_hashes[raw+'s4_mp_launch_v1/'+id_+'_'+suffix+'.json']} for suffix in ('correctness','mp80','mp120')}})
    expected_ids={c['id'] for c in source_manifest['plan']['cells']}
    assert {x['cell_id'] for x in union}==expected_ids
    write('saved_execution_audit_v1.json',audit)
    write('coverage_union_v1.json',{'schema':'track_a_ax2b_h4_supplement_coverage_union_v1','provenance':'old six + new two across independent runs; never a single 8/8 run','correctness_cells':union,'correctness_count':len(union),'MP_record_count':sum(len(x['files'])-1 for x in union),'explicit_representative_groups':events,'explicit_count':len(events),'old_raw_status_unchanged':'H4_LIMITED_STOP','new_unit_statuses':{k:v['status'] for k,v in units.items()},'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION','mandatory_stop':True,'next_stage_authorized':False,'N':None,'G':None,'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED'})
    print(json.dumps({'status':audit['status'],'units':{k:{x:v[x] for x in ('status','parent_wall_seconds','raw_files','raw_bytes','progress_records','counts')} for k,v in units.items()},'new_correctness_summary':cells,'events':events,'missing':missing},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
