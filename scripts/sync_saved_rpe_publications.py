"""Poll public Git refs and reconcile saved H4 evidence, never run producers.

No signal/Q3/price solver is imported or invoked. Unknown schemas are reported
without blocking independent saved evidence. Only this dedicated checkout and
its own publication branch may be changed.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time

from trottertracks.saved_rpe_cost_catalog import Catalog, Reader, report, ref

ROOT=Path(__file__).resolve().parents[1]
NAMES={'C_RESULT_JOIN.json','C_COMPARISON_JOIN.json','HANDOFF_BC.json','C_HANDOFF.json',
       'HANDOFF_C.json','SUMMARY.json','PRICE_AND_COST.json','ROUND_RESULT.json','ROUND_RESULT_INTERRUPTED.json',
       'KAPPA_ABS_RESULTS.json','REQUEST.json'}


def git(root,*args):
    return subprocess.check_output(['git','-c','gc.auto=0','-c','maintenance.auto=false',*args],
        cwd=root,stderr=subprocess.PIPE,text=True).strip()


def put(path,value):
    """Byte-stable writes; same evidence creates no timestamp/HEAD churn."""
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists() and path.read_text()==text:return False
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);return True


def stage_summary(result,publications):
    """Scientific dependencies only: a cost-only change cannot alter roadmap."""
    stages={}
    for method in result['method_minima']:
        rows=[r for r in result['candidates']if r['method']==method]
        stages[method]=dict(priced_feasible=any(r['status']=='FEASIBLE'for r in rows),
            Q3_waiting=any(r['status']=='Q3_WAITING'for r in rows),
            price_waiting=any(r['status']=='PRICE_WAITING'for r in rows),
            waiting_configurations=sorted({r['configuration_id']+':'+r['status']for r in rows
                if r['status']in ('Q3_WAITING','PRICE_WAITING','MISMATCH','PENDING')}),
            search_states=sorted({r['search_status']for r in rows if r.get('search_status')}),
            qualification='CONDITIONAL_MODEL_COMPARISON_ONLY')
    return dict(methods=stages,unresolved_publication_binding=any(p['status']=='PENDING'for p in publications),
        formal_E_C_G_winner='NOT_QUALIFIED',science_execution='NOT_STARTED_BY_SYNC')


def roadmap_block(stage):
    lines=['<!-- C_AUTO_PROGRESS_START -->',
        '現在の研究段階は、保存raw・Q3判定と価格規則を束縛した、保存候補集合内の条件付き比較である。',
        '方式別の最新費用・配分・SOURCEは[C比較表](h4_saved_rpe_cost_catalog_20261116/run01/REPORT.md)と',
        '[機械可読catalog](h4_saved_rpe_cost_catalog_20261116/run01/CATALOG.json)を正本とし、本書に数値を重複記載しない。','',
        '|方式|価格付き適合点|Q3待ち|価格待ち|公開探索の状態|',
        '|---|---|---|---|---|']
    for method,s in stage['methods'].items():
        lines.append('|'+method+'|'+'|'.join('あり'if s[k]else'なし'for k in ('priced_feasible','Q3_waiting','price_waiting'))+
            '|'+(', '.join(s['search_states'])or'保存評価のみ')+'|')
    lines+=['','公開結果の結合に未解決項目：'+('あり'if stage['unresolved_publication_binding']else'なし')+'。',
        '理由と固定SOURCEは[同期状況](h4_saved_rpe_cost_catalog_20261116/SYNC_STATUS.md)を参照。',
        'Full PFの完了とPR-Pauliの方針変更による途中終了は区別する。進行中jobのprivate rawを同期処理から読まない。',
        '正式E[C]、十分N、物理的global RMSE、G、winnerの資格はこの集計で変更しない。',
        '<!-- C_AUTO_PROGRESS_END -->']
    return '\n'.join(lines)


def update_roadmap(path,stage):
    text=path.read_text();start='<!-- C_AUTO_PROGRESS_START -->';end='<!-- C_AUTO_PROGRESS_END -->'
    if text.count(start)!=1 or text.count(end)!=1:raise ValueError('Unique roadmap current-state block required')
    left,tail=text.split(start);_,right=tail.split(end)
    return put(path,left+roadmap_block(stage)+right)


def schema_supported(d,path=''):
    return isinstance(d,dict) and (d.get('schema')in {
        'D_PF_EVIDENCE_REGISTRY_V1',
        'RPE_OPTIMIZER_FIXED_RESULTS_FOR_C_V1','B2_INTERRUPTED_SAVED_TOTAL_ADMISSION_V1','RPE_KAPPA_ABS_RESULTS_V1',
        'PR_DF_LD7_PF_BUDGET_MODEL_COST_Q3_JOIN_FOR_C_V1','PR_DF_LD9_CONFIGURATION_V0_Q3_JOIN_V1',
        'PR_DF_LD7_MODEL_COST_Q3_JOIN_FOR_C_V1','PR_DF_LD1_LD2_LD7_NINE_MODEL_MENU_V1',
        'A_PAULI_LD3_PRICE_HANDOFF_BC_V1','PR_DF_LD_SPLIT_SIX_ENTRY_Q3_COMPARISON_V1',
        'PR_DF_THREE_SPLITS_SAME_V0_Q3_V1','FULL_H_ZERO_ZZ_ADOPTED_EMPIRICAL_MODEL_SCENARIO_V1'} or
        d.get('kind')in {'LD5_FIXED_ALLOCATION_MODEL_COST_AND_Q3_NOT_QUALIFIED_G',
        'CONDITIONAL_MODEL_REFERENCE_COST_AND_Q3_NOT_QUALIFIED_G',
        'FIXED_ALLOCATION_Q3_RESULT_JOIN_NOT_COST_OR_N_QUALIFICATION',
        'CONDITIONAL_FIXED_ALLOCATION_COST_WITH_MODEL_REFERENCE_PRICES'} or
        d.get('status')=='CONDITIONAL_WIDTH_SPECIFIC_A_SHAPE_COST' or
        d.get('classification')=='DIRECT_FINITE_COHORT_DESCRIPTIVE_PRICE_ONLY' or
        ('/best_points/'in path and path.endswith('/REQUEST.json')and 'request_id'in d and 'route'in d) or
        ('conditional_minima'in d and 'adopted_evaluation'in d) or
        ('achieved_conditional_cost_RZ'in d and 'rows'in d) or isinstance(d.get('summary'),str) or
        (path.endswith('C_RESULT_JOIN.json')and ('candidates'in d or 'price_terms'in d or
            (d.get('price_evaluations')==0 and 'rows'in d))))


def discover(reader,heads):
    """One immutable locator per unique public path/blob; no filesystem crawl."""
    found={}
    for branch,head in sorted(heads.items()):
        listing=git(reader.root,'ls-tree','-r',head,'--','docs/research')
        for line in listing.splitlines():
            meta,path=line.split('\t',1);sha=meta.split()[2]
            parts=PurePosixPath(path).parts
            if PurePosixPath(path).name=='REQUEST.json' and 'best_points'not in parts:continue
            if PurePosixPath(path).name not in NAMES or any(s in parts for s in ('jobs','load_check','checks','local_plan')) or any(
                    s.startswith(('artificial','executor_acceptance_','preparation_connection_','generation_acceptance_',
                                  'priority_acceptance_','a_acceptance_','pf_power_stability_artificial'))for s in parts):continue
            key=path+'@'+sha
            if key not in found:
                # Last commit touching the bytes, not the moving branch tip.
                fixed=git(reader.root,'log','-1','--format=%H',head,'--',path)
                found[key]=dict(source=ref(fixed,path),blob=sha,branches=[])
            found[key]['branches'].append(branch)
    return found


def reconcile(reader,inputs,candidates,previous):
    catalog=Catalog(reader,inputs['comparison'])
    for source in inputs['sources']:catalog.add(source)
    records={};accepted=[]
    known={(s['commit'],s['path'])for s in inputs['sources']}
    blobs={s['path']+'@'+git(reader.root,'rev-parse',s['commit']+':'+s['path'])for s in catalog.inputs}
    joined_parents={str(PurePosixPath(s['path']).parent)for s in catalog.inputs
        if PurePosixPath(s['path']).name in ('C_RESULT_JOIN.json','C_COMPARISON_JOIN.json','HANDOFF_BC.json','PRICE_AND_COST.json')}
    for key,item in sorted(candidates.items()):
        source=item['source'];record=dict(item)
        try:
            d=reader.read(source)
            if key in blobs or (source['commit'],source['path'])in known or (source['commit'],source['path'])in catalog.visited:
                record.update(status='INTEGRATED',reason=None)
            elif PurePosixPath(source['path']).name=='SUMMARY.json' and str(PurePosixPath(source['path']).parent)in joined_parents:
                record.update(status='REFERENCE_ONLY',reason='Sibling fixed consumer join already loaded; no additional point inferred')
            elif d.get('schema')=='RPE_OPTIMIZER_ROUND_RESULT_V1' and d.get('methods') and all(
                    r.get('best') is None and r.get('new_calls')==0 for r in d['methods'].values()):
                record.update(status='REFERENCE_ONLY',reason='Metadata-only optimizer return; no evaluated Q3 point published')
            elif PurePosixPath(source['path']).name.startswith('ROUND_RESULT'):
                # A moving optimizer snapshot can lag the final public selected
                # points. Configuration identity alone says nothing about N/cost.
                bests=[m['best']for m in d.get('methods',{}).values()if m.get('best')]
                matched=bool(bests)and all(any(r['configuration_id']==b['configuration_id']and
                    r['N_X']==b['N_X']and r['N_Y']==b['N_Y']and r['q3_status']==b['status']and
                    r['conditional_cost'] is not None and r['conditional_cost']=={
                        k:b['cost'][k]for k in ('numerator','denominator')}for r in catalog.rows)for b in bests)
                record.update(status='REFERENCE_ONLY'if matched else'PENDING',reason=
                    'Exact saved allocation/status/cost already admitted; snapshot is not another point'if matched else
                    'ROUND_RESULT selected allocation lacks matching public OUTCOME/hash/price binding; configuration-only match rejected. '
                    'A later KAPPA_ABS_RESULTS/best_points selection is admitted separately; no job started')
            elif d.get('schema')in {'PR_DF_LD1_F0_DIAGNOSTIC_HANDOFF_V1','LD1_F0_RZ_RANGE_HANDOFF_V1',
                    'PR_DF_LD1_M3_SC_PATHS_V1','PR_DF_LD1_M4_DESCRIPTIVE_HANDOFF_V1'} or d.get('status')in {
                    'PHYSICAL_HEFF_INPUT_NOT_COMPILED_PRICE_OR_GLOBAL_RISK','FINITE_SPECTRAL_SCREEN_SHORTLIST_NOT_RPE_QUALIFIED',
                    'MENU_FINITE_MEANS_COMPLETE_NOT_RMSE_QUALIFIED','PUBLISHED_A0_DIAGNOSTICS_ACCEPTED',
                    'S0A_RUN01_SETUP_FAILED_BEFORE_S_MANDATORY_STOP'}:
                record.update(status='REFERENCE_ONLY',reason='Diagnostic/carrier evidence, not a price/Q3 comparison point')
            elif not schema_supported(d,source['path']):
                record.update(status='PENDING',reason='Unsupported saved schema; needs consumer adapter, not scientific rerun')
            else:
                # The existing adapter may load a fixed Q3 dependency first.
                # Roll back only this item on a bad/missing binding.
                before=(deepcopy(catalog.rows),set(catalog.visited),deepcopy(catalog.inputs))
                try:
                    catalog.add(source)
                    if any(r['status']=='MISMATCH'for r in catalog.rows[len(before[0]):]):
                        raise ValueError('Snapshot/precision/physical/price mismatch; see fixed source')
                except Exception:
                    catalog.rows,catalog.visited,catalog.inputs=before;raise
                accepted.append(source);record.update(status='INTEGRATED',reason=None)
        except (ValueError,KeyError,TypeError,AttributeError,subprocess.CalledProcessError,StopIteration)as exc:
            record.update(status='PENDING',reason=type(exc).__name__+': '+str(exc)[:400])
        records[key]=record
    return catalog.result(),records,accepted


def synchronize(root,config,*,fetch=True,publish=False):
    root=Path(root);local=root/'artifacts/local_h4_catalog_sync';local.mkdir(parents=True,exist_ok=True)
    with (local/'sync.lock').open('a')as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='ALREADY_RUNNING')
        branch=git(root,'branch','--show-current')
        if branch!=config['branch']:raise ValueError('Only the dedicated C publication branch may be written')
        if publish and git(root,'status','--porcelain','--untracked-files=normal'):
            raise ValueError('Dedicated checkout has changes; skip this cycle without touching them')
        print('SYNC FETCH_START',flush=True)
        if fetch:
            branches=set(config['branches'])
            patterns=['refs/heads/'+s for s in config.get('branch_globs',[])]
            if patterns:
                public=git(root,'ls-remote','--heads',config['remote'],*patterns)
                branches.update(line.split()[1].removeprefix('refs/heads/')for line in public.splitlines())
            git(root,'fetch','--no-tags',config['remote'],
                *['+refs/heads/'+b+':refs/remotes/'+config['remote']+'/'+b for b in sorted(branches)])
        else:branches=set(config['branches'])
        heads={b:git(root,'rev-parse','refs/remotes/'+config['remote']+'/'+b)for b in sorted(branches)}
        publication_heads={}
        if config.get('search_map'):
            # Every environment publishes the same small controller manifest.
            # No branch-name transfer to C is required for new independent runs.
            if fetch:
                public=git(root,'ls-remote','--heads',config['remote'])
                publication_heads={line.split()[1].removeprefix('refs/heads/'):line.split()[0]
                    for line in public.splitlines()if line.split()[1]!='refs/heads/'+config['branch']}
                missing=[b for b,h in publication_heads.items() if b not in heads]
                if missing:
                    git(root,'fetch','--no-tags',config['remote'],
                        *['+refs/heads/'+b+':refs/remotes/'+config['remote']+'/'+b for b in missing])
            else:
                # Offline replay uses the exact heads from the last manifest.
                source_path=root/config['search_map']/'SOURCES.json'
                if source_path.exists():publication_heads=json.loads(source_path.read_text()).get('heads',{})
            publication_heads.update(heads)
        directory=root/config['directory']
        def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
        map_files=['src/trottertracks/'+name for name in
            ('h4_search_map.py','h4_search_run_evidence.py','h4_search_map_html.py')]if config.get('search_map')else[]
        implementation=hashlib.sha256((sha(root/'scripts/sync_saved_rpe_publications.py')+
            sha(root/'src/trottertracks/saved_rpe_cost_catalog.py')+
            sha(root/'src/trottertracks/pf_optimizer_connection.py')+''.join(sha(root/p)for p in map_files)+
            json.dumps(config,sort_keys=True)).encode()).hexdigest()
        prior=json.loads((directory/'PUBLICATIONS.json').read_text())if (directory/'PUBLICATIONS.json').exists()else{}
        if (prior.get('heads')==heads and prior.get('search_publication_heads',{})==publication_heads and prior.get('implementation')==implementation and
                prior.get('input_manifest_sha256')==sha(root/config['inputs']) and
                (not config.get('search_map') or (prior.get('search_map')or{}).get('status')=='CURRENT')):
            if publish:
                remote=git(root,'ls-remote',config['remote'],'refs/heads/'+config['branch']).split()[0]
                if remote!=git(root,'rev-parse','HEAD'):git(root,'push',config['remote'],config['branch'])
            return dict(status='UNCHANGED',changed=[],head=git(root,'rev-parse','HEAD'))
        print('SYNC DISCOVERY_START',flush=True)
        reader=Reader(root);inputs=json.loads((root/config['inputs']).read_text())
        candidates=discover(reader,heads)
        result,records,accepted=reconcile(reader,inputs,candidates,{})
        print('SYNC RECONCILED',len(result['candidates']),flush=True)
        # Persist the fixed accepted locators so replay needs neither network
        # nor the producer's local directory. Failed items are separately listed.
        all_sources=inputs['sources']+[s for s in accepted if s not in inputs['sources']]
        combined=dict(inputs,sources=all_sources)
        # result() uses source order for display; accepted sources are deterministic.
        changed=[]
        def write(relative,value):
            if put(root/relative,value):changed.append(relative)
        output=config['directory']+'/run01'
        result['input_manifest']=config['inputs']
        write(config['inputs'],combined)
        write(output+'/CATALOG.json',result)
        write(output+'/REPORT.md',report(result))
        write(output+'/INPUTS.json',combined)
        write(output+'/CHECKS.json',dict(schema='H4_SAVED_CATALOG_RECONCILIATION_CHECKS_V1',
            normalized_candidates=len(result['candidates']),
            status_counts={s:sum(r['status']==s for r in result['candidates'])for s in
                sorted({r['status']for r in result['candidates']})},
            method_minima=result['method_minima'],duplicates_removed=result['duplicates_removed'],
            fixed_inputs=len(all_sources),new_scientific_calls=0,
            scope='SAVED_RESULT_METADATA_AND_CONDITIONAL_TOTAL_COMPARISON_ONLY'))
        map_state=None
        if config.get('search_map'):
            from trottertracks.h4_search_run_evidence import discover_runs
            from trottertracks.h4_search_map import build_index, report as map_report
            from trottertracks.h4_search_map_html import index_text, render
            print('SYNC SEARCH_MAP_START',flush=True)
            try:
                # The comparison has already normalized its saved values. Do
                # not retain every large producer document through both phases.
                reader.document.cache_clear()
                runs,map_issues=discover_runs(reader,publication_heads,legacy_heads=set(heads))
                reader.document.cache_clear()
                fixed_sources=[r['source']for r in runs]
                map_sources=dict(schema='H4_SEARCH_MAP_SOURCES_V1',heads=publication_heads,
                    legacy_heads=sorted(heads),sources=fixed_sources,
                    catalog_inputs=config['inputs'],controller_publication='SEARCH_PUBLICATION.json')
                index=build_index(result,runs,sources=fixed_sources,issues=map_issues)
                index['included_public_heads']=publication_heads
                write(config['search_map']+'/SOURCES.json',map_sources)
                write(config['search_map']+'/INDEX.json',index_text(index)+'\n')
                write(config['search_map']+'/index.html',render(index))
                write(config['search_map']+'/REPORT.md',map_report(index))
                map_state=dict(status='CURRENT',runs=len(index['runs']),unique_points=len(index['points']),
                    issues=len(index['issues']),path=config['search_map']+'/index.html')
            except (ValueError,KeyError,TypeError,AttributeError,subprocess.CalledProcessError)as error:
                # Preserve the last good map and update independent C evidence.
                map_state=dict(status='PENDING',reason=type(error).__name__+': '+str(error)[:400])
            write(config['search_map']+'/STATUS.json',map_state)
            print('SYNC SEARCH_MAP_END',json.dumps(map_state),flush=True)
        status=dict(schema='H4_PUBLICATION_SYNC_V1',heads=heads,publications=records,implementation=implementation,
            search_publication_heads=publication_heads,search_map=map_state,
            input_manifest_sha256=hashlib.sha256((json.dumps(combined,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest(),
            stage=stage_summary(result,list(records.values())),new_science_calls=0)
        write(config['directory']+'/PUBLICATIONS.json',status)
        pending=[r for r in records.values()if r['status']=='PENDING']
        lines=['# 公開成果同期の状況','',
            '公開Git成果のみを読取。個別結果の取得・fit・価格予測・Q3は起動しない。',
            '方式別の費用は[比較表](run01/REPORT.md)を参照。','',
            '|判断・接続待ちの固定成果|理由|','|---|---|']
        for r in pending:lines.append('|`'+r['source']['commit'][:9]+':'+r['source']['path']+'`|'+r['reason'].replace('|','/')+'|')
        lines+=['','未対応schema/結合不足は保存し、他の取込みを止めない。新科学計算の承認・採択・旧STOPは変更しない。','']
        write(config['directory']+'/SYNC_STATUS.md','\n'.join(lines))
        if update_roadmap(root/config['roadmap'],status['stage']):changed.append(config['roadmap'])
        print('SYNC WRITTEN',len(changed),flush=True)
        if publish:
            if changed:
                git(root,'add','--sparse','--',*changed)
                git(root,'-c','user.name=HIROMU1015','-c','user.email=u411612g@ecs.osaka-u.ac.jp',
                    'commit','-m','Sync fixed public H4 results and dependency progress [skip ci]')
            actual=git(root,'ls-remote',config['remote'],'refs/heads/'+config['branch']).split()[0]
            if actual!=git(root,'rev-parse','HEAD'):
                git(root,'push',config['remote'],config['branch'])
            actual=git(root,'ls-remote',config['remote'],'refs/heads/'+config['branch']).split()[0]
            if actual!=git(root,'rev-parse','HEAD'):raise ValueError('Remote SHA differs after push')
            # Fetch the published ref back; verify its tree, rather than only
            # trusting the local remote-tracking ref updated by push.
            git(root,'fetch','--no-tags',config['remote'],'refs/heads/'+config['branch'])
            if git(root,'rev-parse','FETCH_HEAD^{tree}')!=git(root,'rev-parse','HEAD^{tree}'):
                raise ValueError('Remote tree differs after publication')
        return dict(status='UPDATED'if changed else'UNCHANGED',changed=changed,
            accepted=len(accepted),pending=len(pending),head=git(root,'rev-parse','HEAD'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--publish',action='store_true');p.add_argument('--no-fetch',action='store_true')
    args=p.parse_args();start=time.monotonic()
    result=synchronize(ROOT,json.loads((ROOT/args.config).read_text()),fetch=not args.no_fetch,publish=args.publish)
    print(json.dumps(dict(result,wall_seconds=time.monotonic()-start),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
