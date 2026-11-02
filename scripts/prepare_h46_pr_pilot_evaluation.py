"""Bind completed acquisitions and authorize saved-only pilot summaries."""
import argparse
from pathlib import Path
import subprocess
import sys
import h46_pr_pilot_evaluation as e

def acquisition_indices(h4path,h6path,files):
    def add(p,role):
        row=e.seal(p,role);files[row['path']]=row;return row
    h4=e.read(h4path);h6=e.read(h6path);a=e.read(h6['acceptance_path']);old=e.read(a['old_config_path'])
    configs={'H4':h4,'H6':h6};owner={h4['SOURCE']:h4,h6['SOURCE']:h6}
    h4audit=e.read(Path(h4['run_root'])/'end_audit.json');h6audit=e.read(Path(h6['run_root'])/'end_audit.json')
    for cfg,audit in ((h4,h4audit),(h6,h6audit)):
        e.require(audit['status']=='ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP'
          and not audit['errors'],'COMPLETED_NATIVE_BYTE_ACQUISITION_REQUIRED')
        c=e.read(Path(cfg['run_root'])/'completion_witness.json');x=e.read(cfg['supervisor']['exit'])
        e.require(c['status']=='UNIT_COST_STAGE_COMPLETE_MANDATORY_STOP' and not c['errors']
          and x['native_exit']==0 and x['reaped']
          and all(v['exit']==0 and v['reaped'] for v in c['workers'].values()),'COMPLETE_EXIT_REAP')
        e.require(e.sha(Path(cfg['run_root'])/'completion_witness.json')==audit['completion_sha256'],'COMPLETION_AUDIT_SHA')
        for p,role in [(Path(cfg['run_root'])/'end_audit.json','acquisition_end_audit'),
          (Path(cfg['run_root'])/'completion_witness.json','acquisition_completion'),
          (cfg['supervisor']['exit'],'acquisition_native_exit'),(cfg['marker'],'acquisition_consumed')]:add(p,role)
    add(h4path,'H4_acquisition_config');add(h6path,'H6_continuation_config')
    add(h6['acceptance_path'],'H6_separate_saved74_acceptance');add(a['old_config_path'],'H6_original_STOP_config')
    for row in a['old_state_files']:
        e.require(e.sha(row['path'])==row['sha256'],'OLD_STOP_STATE_UNCHANGED');add(row['path'],'original_STOP_state')
    sr=Path(h6['run_root'])/'continuation_summary_PRIVATE.json';summary=e.read(sr)
    e.require(summary['status']=='ORIGINAL396_COVERAGE_COMPLETE_SEPARATE74_ACCEPTANCE_PLUS322_CONTINUATION'
      and summary['new_end_audit_sha256']==e.sha(Path(h6['run_root'])/'end_audit.json')
      and summary['old_saved_byte_PASS'] and summary['old_whole_run_STOP_retained'],'H6_FULL396_SEPARATE_ACCEPTANCE')
    add(sr,'H6_fixed396_completion_summary')
    jpath=Path(h6['run_root'])/'combined_result_references_PRIVATE.json';join=e.read(jpath);add(jpath,'H6_fixed396_references')
    refs={'H4':[dict(id=r['id'],path=str(Path(h4['output'])/r['id']/'result.json'),sha256=r['result_sha256'],SOURCE=h4['SOURCE']) for r in h4audit['results']],
          'H6':join['results']}
    result={};prep={v['candidate']['id']:v['preparation_hash'] for v in e.read(e.ROOT/'references/h46_m5_cost_readiness_v1.json')['candidate_table']}
    for system in ('H4','H6'):
        full=h4['tasks'] if system=='H4' else old['tasks'];byid={r['id']:r for r in refs[system]}
        e.require(len(refs[system])==len(byid)==len(full)==396 and set(byid)=={t['id'] for t in full},'FIXED396_NO_PARTIAL_SELECTION')
        groups=[]
        for candidate in [r for r in e.design.groups() if r['system']==system]:
            indexed=[]
            for task in full:
                if task['candidate']!=candidate:continue
                ref=byid[task['id']]
                origin=h4 if system=='H4' else (old if ref['SOURCE']==old['SOURCE'] else h6)
                e.require(ref['SOURCE']==origin['SOURCE'],'ACQUISITION_SOURCE_NOT_OTHER_STUDY')
                p=Path(ref['path']);e.require(p==Path(origin['output'])/task['id']/'result.json'
                  and e.sha(p)==ref['sha256'],'ACQUISITION_RESULT_BINDING')
                add(p,'paired_pilot_unit_cost_result')
                pieces=candidate['q']*{'2nd':1,'4th':3,'4th(m5_best)':11}[candidate['formula']]
                paths=[p.parent/f'piece_law_{i}.json' for i in range(pieces)]
                for law in paths:add(law,'saved_generation_law')
                indexed.append(dict(task=task,result_path=str(p),result_sha256=ref['sha256'],
                  SOURCE=origin['SOURCE'],plan_sha256=origin['plan_sha256'],
                  law_paths=list(map(str,paths)),input_binding=origin['systems'][system]['binding'],
                  preparation_hash=prep[candidate['id']]))
            e.require([r['task']['sample'] for r in indexed]==list(range(33)),'ORIGINAL33_PER_GROUP')
            groups.append(dict(candidate=candidate,tasks=indexed))
        result[system]=dict(groups=groups,campaign_id=configs[system]['campaign_id'])
    result['H4']['proof_paths']=[str(Path(h4path).resolve()),str(Path(h4['run_root'])/'end_audit.json'),
      str(Path(h4['run_root'])/'completion_witness.json'),h4['supervisor']['exit'],h4['marker']]
    result['H6']['proof_paths']=[str(Path(h6path).resolve()),str(Path(h6['run_root'])/'end_audit.json'),
      str(Path(h6['run_root'])/'completion_witness.json'),h6['supervisor']['exit'],h6['marker'],
      h6['acceptance_path'],a['old_config_path'],str(sr),str(jpath),*[r['path'] for r in a['old_state_files']]]
    return result

def prepare(source,h4config,h6config,h2root,private):
    private=Path(private).resolve();private.mkdir(parents=True,exist_ok=False);files={}
    for rel in e.SOURCE_PATHS:
        p=e.ROOT/rel;e.require(p.read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=e.ROOT),'COMMITTED_STATISTICS_SOURCE')
        row=e.seal(p,'evaluation_SOURCE:'+rel);files[row['path']]=row
    stages=acquisition_indices(h4config,h6config,files)
    e.require(sys.executable==e.read(h4config)['runtime']['python']==e.read(h6config)['runtime']['python'],
      'FIXED_ACQUISITION_PYTHON_FOR_STATISTICS')
    import concurrent.futures.process,concurrent.futures.thread
    import multiprocessing.spawn,multiprocessing.synchronize
    binary=e.seal(Path(sys.executable).resolve(),'evaluation_interpreter_binary');files[binary['path']]=binary
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).resolve().is_relative_to('/usr/lib/python3.12'):
            row=e.seal(path,'evaluation_stdlib_runtime');files[row['path']]=row
    h2root=Path(h2root).resolve()
    h2paths=['docs/research/track_a_h2_m5_existing_pf_comparison_results_20261020.md',
      'artifacts/resource_applicability/track_a_h2_m5_existing_pf_comparison/2026-10-20/comparison.json',
      'artifacts/resource_applicability/track_a_h2_m5_existing_pf_comparison/2026-10-20/provenance.json']
    h2refs=[]
    for rel in h2paths:
        p=h2root/rel;e.require(p.read_bytes()==subprocess.check_output(['git','show',
          '2d415da2b09a73fab5580eded80e822fc44a25e1:'+rel],cwd=h2root),'EXISTING_H2_COMMITTED_EVALUATION')
        row=e.seal(p,'H2_EXISTING_ANALYSIS_REUSE_ONLY');h2refs.append(row)
    e.save(private/'H2_reuse_references_PRIVATE.json',dict(commit='2d415da2b09a73fab5580eded80e822fc44a25e1',files=h2refs,
      new_H2_numeric_analysis=False,new_H2_ranking=False))
    configs=[]
    # All input bytes/paths and targets are fixed before the first evaluation.
    for system in ('H4','H6'):
        stage=system+'_PILOT_EVALUATION';dest=private/stage;dest.mkdir();root=dest/'run01'
        needed=set(stages[system]['proof_paths'])
        for group in stages[system]['groups']:
            for row in group['tasks']:needed.add(row['result_path']);needed.update(row['law_paths'])
        closure=[r for r in files.values() if r['role'].startswith('evaluation_') or r['path'] in needed]
        plan=dict(SOURCE=source,stage=stage,groups=stages[system]['groups'],campaign_id=stages[system]['campaign_id'],
          sealed_files=closure,relative_MCSE_targets=list(e.TARGETS),variance_inflations=list(e.INFLATIONS),
          n_E_adopted=False,shot_accounting=False,ranking=False,complete_acquisition_accepted=True,
          old_STOP_rewritten=False,run_root=str(root),
          python_executable=sys.executable,python_version=sys.version,
          user_authority='USER_NEXT_STAGE_APPROVAL_EVALUATE_H2_H4_H6_REUSE_EXISTING_ANALYSES',
          rare_law_model='IDEAL_INDEPENDENT_STEPS_SAVED_PROBABILITIES_NOT_A_COST_BOUND')
        pp=dest/'plan_APPROVED_PRIVATE.json';e.save(pp,plan)
        cfg=dict(SOURCE=source,stage=stage,plan=plan,plan_path=str(pp),plan_sha256=e.sha(pp),
          run_root=str(root),authorized=True,grant=dict(valid=True,SOURCE=source,stage=stage,
            plan_sha256=e.sha(pp),run_root=str(root),grant_id=stage+'_'+e.sha(pp),retry=False,next_phase_authorized=False,
            scope=dict(groups=12,trajectories=396,nominal_n_E_scenarios=108,new_sampling=0,
              family_n_E_aggregations=27,new_compile=0,new_signal=0,new_shot_calculation=0,ranking=False)))
        e.authorize(cfg);cp=dest/'config_APPROVED_PRIVATE.json';e.save(cp,cfg);configs.append(str(cp))
    e.save(private/'preflight_PRIVATE.json',dict(status='PASS',SOURCE=source,stages=list(e.STAGES),
      new_H2_numeric_analysis=0,source_and_input_files=len(files),new_numerical_evaluations_started=0))
    return configs

def main():
    p=argparse.ArgumentParser()
    for key in ('source','h4-config','h6-config','h2-root','private'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(__import__('json').dumps(dict(configs=prepare(a.source,a.h4_config,a.h6_config,a.h2_root,a.private))))
if __name__=='__main__':main()
