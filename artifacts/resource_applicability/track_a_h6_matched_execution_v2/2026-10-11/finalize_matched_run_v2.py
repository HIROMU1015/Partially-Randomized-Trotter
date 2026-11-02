"""Publication-only helper: never launches, retries, or changes scientific source."""
import argparse, ast, datetime, hashlib, json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path

ROOT=Path('/home/abe/Project/Partially Randomized Trotter/.worktrees/track-a-ax2b-h4-post-review-20261010')
META=ROOT/'artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11'
OUT=ROOT/'artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1'
PREP=ROOT/'artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11'
BRANCH='track-a-ax2b-h4-post-review-20261010'
REMOTE='git@github.com:HIROMU1015/Partially-Randomized-Trotter.git'
GRANT_COMMIT='2157da1d208a04102166fa1ae3c154490546c90f'
BASELINE=META/'baseline_before_execution_v2.json'
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h6_matched_contract_v2 import verify_sources, verify_parent, environment, file_hash, digest

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def git(*args, data=None):return subprocess.check_output(['git',*args],cwd=ROOT,input=data)
def load(path):return json.loads(Path(path).read_text())
def save(path,obj):
    with Path(path).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
def alive(pid,start):
    try:
        text=Path(f'/proc/{pid}/stat').read_text(); fields=text[text.rfind(')')+2:].split()
        return fields[0]!='Z' and int(fields[19])==start
    except (OSError,ValueError):return False
def workers_live():
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if len(args)>1 and b'run_track_a_h6_matched_v2.py' in args[1] and (p/'cwd').resolve()==ROOT:
                result.append(int(p.name))
        except OSError:pass
    return result
def project_paths(path):return sorted(str(p.relative_to(ROOT)) for p in path.rglob('*') if p.is_file() and '.runtime_cache' not in p.parts)
def add_prefix(path,block):
    before=(ROOT/path).read_text(); clean=git('show','HEAD:'+path).decode()
    insertion='\n'+block.rstrip()+'\n\n'
    def apply(text):
        pos=text.index('\n')+1
        return text[:pos]+insertion+text[pos:]
    (ROOT/path).write_text(apply(before))
    return insertion,apply(clean)
def stage_clean(path,text):
    blob=git('hash-object','-w','--stdin',data=text.encode()).decode().strip()
    mode=git('ls-files','--stage','--',path).decode().split()[0]
    git('update-index','--cacheinfo',mode,blob,path)
def fetch(bare):
    subprocess.run(['git','-c','maintenance.auto=false','-c','gc.auto=0','--git-dir='+bare,
        'fetch','--no-tags',REMOTE,f'refs/heads/{BRANCH}:refs/heads/verify'],check=True)
def remote_bytes(bare,commit,path):return subprocess.check_output(['git','--git-dir='+bare,'show',commit+':'+path])
def verify_remote(bare,commit,expected):
    assert subprocess.check_output(['git','--git-dir='+bare,'rev-parse','refs/heads/verify']).decode().strip()==commit
    for path,sha in expected.items():
        if hashlib.sha256(remote_bytes(bare,commit,path)).hexdigest()!=sha:raise ValueError('REMOTE_BYTES:'+path)

def finalize(expected_head,exit_code):
    assert not workers_live(),'Scientific processes still live: publication must wait.'
    assert git('rev-parse','HEAD').decode().strip()==expected_head,'HEAD changed; preserve and stop publication.'
    assert git('branch','--show-current').decode().strip()==BRANCH
    assert git('remote','get-url','--push','origin').decode().strip()==REMOTE
    assert not git('diff','--cached','--name-only'),'Existing staged files: preserve and stop publication.'
    self_path=str(Path(__file__).resolve().relative_to(ROOT))
    assert hashlib.sha256(git('show',expected_head+':'+self_path)).hexdigest()==file_hash(Path(__file__)), 'Publication helper changed.'
    baseline=load(BASELINE)
    for path,sha in baseline['file_hashes'].items():
        assert file_hash(ROOT/path)==sha,'Existing file changed: '+path
    m=load(PREP/'sealed_manifest_v2.json'); g=load(META/'authorization_v2.json')
    verify_sources(ROOT,m['source_commit'],m['source_hashes'])
    assert verify_parent(ROOT)==m['input_identity']
    env_equal=environment()==m['environment']
    assert file_hash(META/'authorization_v2.json')=='a006c3b933db79d372a09366aeb0e7ae508842e96bd3d786e7d6fb9729fd2e39'
    terminal=load(OUT/'execution_terminal.json') if (OUT/'execution_terminal.json').exists() else None
    status=terminal['status'] if terminal else 'EXTERNAL_PROCESS_EXIT_TERMINAL_MISSING'
    if terminal:
        assert terminal['mandatory_stop'] is True and terminal['next_stage_authorized'] is False
        assert terminal['retry'] is False and terminal['resume'] is False
    command=[sys.executable,str(ROOT/'scripts/resource_applicability/audit_track_a_h6_matched_v2.py'),'--output',str(OUT)]
    checked=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},text=True,capture_output=True)
    audit={'recorded_at_utc':now(),'command':command,'exit_code':checked.returncode,
        'stdout':checked.stdout,'stderr':checked.stderr,'scientific_GO_issued':False,
        'numerical_certificate_verified':False,
        'scope':'STOP: source/input/grant/limits and saved byte closure only; complete: also frozen task/seed/paired-wrapper coverage. No new numerical computation.'}
    if checked.returncode==0:audit['return_value']=ast.literal_eval(checked.stdout.strip())
    save(META/'static_identity_audit_v2.json',audit)
    inv=load(OUT/'execution_inventory.json') if (OUT/'execution_inventory.json').exists() else None
    completed_signals=[]; signal_missing=[]
    for cell in m['plan']['cells']:
        path=OUT/'signal'/f'{cell["id"]}_signal.json'
        if path.exists():
            record=load(path);assert record['cell']==cell
            completed_signals.append(cell['id'])
        else:signal_missing.append(cell['id'])
    b3=load(OUT/'signal/b3_normalization_diagnostic.json') if (OUT/'signal/b3_normalization_diagnostic.json').exists() else None
    phase_records={}; cost_rows=[]; incomplete=[]
    for phase in ('exploration','confirmation'):
        path=OUT/(phase+'_tasks.json');registration=load(path) if path.exists() else None
        complete_count=0;missing_tasks=[]
        if registration:
            for i,task in enumerate(registration['tasks']):
                d=OUT/f'cost_{phase}_{i:04d}';r=d/'cost_result.json';t=d/'worker_terminal.json'
                if r.exists() and t.exists() and load(t)['status']=='TASK_COMPLETE':
                    result=load(r);assert result['task']==task;cost_rows+=result['rows'];complete_count+=1
                else:
                    missing_tasks.append({'directory':str(d.relative_to(OUT)),'task':task,
                        'status':'PARTIAL_TASK_PRESERVED' if d.exists() else 'NOT_DISPATCHED_AFTER_STOP'})
        phase_records[phase]={'registration_created':registration is not None,
            'registered_groups':len(registration['tasks']) if registration else None,
            'completed_groups':complete_count,'missing_tasks':missing_tasks,
            'selected_ids':registration.get('selected') if registration and phase=='confirmation' else None}
        incomplete+=missing_tasks
    wrappers=list(OUT.glob('cost_*_????/wrapper_*.json'))
    required=['signal/worker_claim.json','signal/worker_terminal.json','signal/prepared_representation.json',
        'signal/input_reference.json','signal/primitive_validation.json','signal/signal_summary.json',
        'signal/b3_normalization_diagnostic.json','exploration_tasks.json','exploration_resource_map.json',
        'confirmation_tasks.json','confirmation_resource_map.json','cost_coverage.json','aggregate_calls.json',
        'cost_order_diagnostics.json']
    missing={'status':status,'signal_candidates_registered':92,'signal_candidates_saved':len(completed_signals),
        'completed_signal_ids':completed_signals,'missing_signal_ids':signal_missing,
        'b3_normalization_registered':168,'b3_normalization_saved':len(b3['rows']) if b3 else 0,
        'epsilons':m['plan']['epsilons'],'phases':phase_records,
        'missing_records':[p for p in required if not (OUT/p).exists()],
        'saved_wrapper_records':len(wrappers),'completed_cost_rows':len(cost_rows),
        'maximum_wrappers':1708,'unregistered_cost_counts_not_inferred':True,
        'aborted_task_calls_not_inferred':True,'retry':False,'resume':False,'mandatory_stop':True,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
        'next_stage_authorized':False,'H8_authorized':False,'additional_search_authorized':False}
    save(META/'missing_coverage_v2.json',missing)
    raw_hashes={p:file_hash(ROOT/p) for p in project_paths(OUT)}
    raw_bytes=sum((ROOT/p).stat().st_size for p in raw_hashes)
    observation={'status':status,'recorded_at_utc':now(),'runner_invocations':1,
        'one_shot_authorization_consumed':True,'runner_exit_code':exit_code,'supervisor_terminal':terminal,
        'grant_publication_commit':GRANT_COMMIT,'source_commit':m['source_commit'],
        'authorization_sha256':file_hash(META/'authorization_v2.json'),
        'source216_exact':True,'parent33_exact':True,'environment_equal_after':env_equal,
        'frozen_inventory_source_input_after_verified':inv.get('source_input_after_verified') if inv else None,
        'coordinator_limits':load(OUT/'coordinator_limits.json') if (OUT/'coordinator_limits.json').exists() else None,
        'signal_worker_limits':load(OUT/'signal/worker_claim.json').get('resource_limits') if (OUT/'signal/worker_claim.json').exists() else None,
        'live_aggregate_peak_rss_frozen_value':inv.get('live_aggregate_peak_rss_bytes') if inv else None,
        'rss_peak_scope':'Returned complete batches only; on failed batch the frozen value can remain initialized zero or exclude the failed batch. Not a whole-run peak certificate.',
        'calls_record':load(OUT/'aggregate_calls.json') if (OUT/'aggregate_calls.json').exists() else None,
        'calls_scope':'Stored aggregate or completed records only. Counts in aborted tasks are unknown; no zero inferred.',
        'science_backend':'CPU; sector/cache/compact; signal four threads and at most two disposable cost processes',
        'scientific_wall_caps':{'phase':None,'total':None},'new_input_generated':False,
        'measurement_shots_sampled':False,'formal_winner_certified':False,'ground_state_certified':False,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
        'mandatory_stop':True,'next_stage_authorized':False,'retry':False,'resume':False,
        'H8_authorized':False,'additional_search_authorized':False,
        'raw_file_count':len(raw_hashes),'raw_total_bytes':raw_bytes,
        'excluded_cache_file_count':sum(p.is_file() for p in (OUT/'.runtime_cache').rglob('*'))}
    save(META/'execution_observation_v2.json',observation)
    result_path='docs/research/track_a_h6_matched_accuracy_result_v2.md'
    reason=terminal.get('reason') if terminal else 'Original supervisor terminal not saved; external exit observation only.'
    result=f'''# Track A H6精度一致・測定込み資源比較 v2：一次結果・実行監査

2026-10-11。固定source `{m['source_commit']}` と公開v2契約に従い、ユーザーが認可した科学実行を一回だけ行った。原statusは **`{status}`**、reasonは `{reason}`。retry/resumeなし。grantは消費済み、mandatory STOP。H8・候補追加・他の科学計算へ進まない。

## source・grant・入力・公開来歴

- [固定契約](track_a_h6_matched_accuracy_execution_contract_v2.md)・[sealed manifest](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/sealed_manifest_v2.json)。manifest SHA-256 `99cd2055283d2bb35ec5b373d934b3107fac6997a013e5f523914df3216014b9`、digest `7693137a582dc80d2f221dc2ed8a51161ebd79f6f183f6a4bb9dbe4e0f20b19e`。公開準備 `{g['authorized_preparation_commit']}`。科学source216 pathは実行後も固定bytesと一致した。
- [authorization-only grant](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/authorization_v2.json) 公開commit `{GRANT_COMMIT}`、SHA-256 `{file_hash(META/'authorization_v2.json')}`。[起動前remote照合](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/remote_prelaunch_verification_v2.json)で253 pathを再取得照合後だけ起動した。source・入力・環境・資源条件を変更しなかった。
- [runner](../../scripts/resource_applicability/run_track_a_h6_matched_v2.py)・[supervisor/worker](../../src/trottertracks/resource_applicability/h6_matched_execution_v2.py)・[静的auditor](../../scripts/resource_applicability/audit_track_a_h6_matched_v2.py) はscience source commitに固定されている。
- 保存入力source `f37005f01b2be38c5993d6e82df91abe9c643d21`、結果 `554fc52add39c2c1b45b765a3135df76fda6f15a`、[入力監査](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/execution_evidence_inventory_v2.json) `f52787a22542b31bd39fd004a8d3d71325bc56b0`。親33 pathと[snapshot](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/h6_input_snapshot.npz) bytesを保持。新SCF・DF分解・state生成・solverは行っていない。

## 一次artifact・保存範囲・欠測

- [原出力ディレクトリ](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/)：raw {len(raw_hashes)} file、{raw_bytes} bytes。derived runtime cacheを除く全原記録を公開対象とする。
- [原terminal](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/execution_terminal.json)、[原inventory](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/execution_inventory.json)、[launch binding](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/launch_binding.json)、[coordinator limit receipt](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/coordinator_limits.json)。未保存の場合は下記欠測一覧で明示し、代用値や新terminalを作らない。
- [signal出力](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/)：92候補中{len(completed_signals)}件のsignal record、B3 normalization-only 168組中{missing['b3_normalization_saved']}組が保存されている。個別bias/u/normalization/shot情報は原recordを参照し、未保存値を推測しない。
- 保存wrapper record {len(wrappers)}件、TASK_COMPLETEのcost row {len(cost_rows)}件。探索登録group {phase_records['exploration']['registered_groups']}／完了{phase_records['exploration']['completed_groups']}、確認登録group {phase_records['confirmation']['registered_groups']}／完了{phase_records['confirmation']['completed_groups']}。[欠測・coverage](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/missing_coverage_v2.json)で候補・未完了task・未作成recordを区別する。最大1708 wrapperは上限であり、未登録の残task数を意味しない。途中でcompileしたがrecordが未保存の回数は推測しない。
- [外部観測](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/execution_observation_v2.json)、[静的audit](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/static_identity_audit_v2.json)、[bytes inventory](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/execution_evidence_inventory_v2.json)、[保全audit](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/preservation_v2.json)。原出力・過去のfreeze・監査を後から書き換えない。

## 条件と監査の解釈

linear H6／等間隔1.00 Å／STO-3G／12 modes／α3β3／sector400。tol-only1e-8・final_rankなし・cutoff0、signed generation-order DF rank19、採用済みweighted Hermitian projectionと同じ保存state。T=0.8、B0/B2 prefix5/10/15、B1 full19 S2/S4、q1/2/4/8、B2 r1/2/4・R=qr・K2/4、delta0.8/0.4/0.2/0.1、ε_sig=0.05/0.01/0.005/0.001の固定92候補を維持した。B3は168 normalization scalarだけでsignal/sampling/compileを追加しない。

signal先行screen、適格候補のexploration n8、精度別B2上位2のunion最大8を独立seed confirmation n32、paired X/Yと登録ordinary anchor、最大1708 measured wrapperとcompile guard1712は固定契約のまま。旧pilot標本をfresh平均にpoolしない。quantum shotは解析的Hoeffding会計で、measurement shot自体のsamplingは行わない。

coordinator AS soft2 GiB／hard8 GiB、signal・cost AS soft/hard8 GiB。signal CPU [0,2,5,6]・Numba/OMP4／BLAS1、costは最大2 disposable process・各thread1。RSS18 GiB／output2 GiB／log1 MiB／call・instruction guardを維持し、phase/total wall capはnull。元inventoryのRSS値は失敗batchのpeakを含まない場合があり、特に初期値0を実測peakと解釈しない。

静的audit exit codeは{checked.returncode}。STOP時のPASSは保存bytesとidentity/limitの整合性だけを意味し、全signalや費用coverageの合格ではない。COMPLETE時は登録task・seed・paired wrapper coverageも検査する。環境事後一致は{env_equal}。いずれもempirical numerical allowanceを厳密な誤差保証とせず、sample mean/SEをpopulation費用やformal winnerの保証としない。ground-state、化学的energy精度、全候補の資源優位、H8 GOを認定しない。

旧[v1 STOP](track_a_h6_matched_accuracy_result_v1.md)／source／消費済みgrant、旧[pilot 7 cell・32/36 wrapper](track_a_h6_technical_pilot_result_v2.md)、保存入力と既存dirty/untrackedを保全。**mandatory STOP。`H6_NOT_AUTHORIZED`／`DRAFT_NOT_AUTHORIZATION`／`next_stage_authorized=false`。** 今回の一回grantを再使用しない。科学的判断と追加runの要否はGPT・利用者へ戻す。

公開結果commitとGitHub再取得hash照合は[remote結果照合](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/remote_result_verification_v2.json)で特定する。この文書はCodexによる事実と監査scopeの整理で、GPT独立科学レビューや承認済み次段階契約ではない。
'''
    if not terminal:
        result=result.replace('[原terminal](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/execution_terminal.json)','原terminal欠測')
    if not inv:
        result=result.replace('[原inventory](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/execution_inventory.json)','原inventory欠測')
    with (ROOT/result_path).open('x') as f:f.write(result)
    note_path='docs/research/研究ノート/2026-10-11_track_a_h6_matched_execution_v2.md'
    with (ROOT/note_path).open('x') as f:f.write(f'# 2026-10-11 H6精度一致比較v2の一回認可・実行・STOP\n\nユーザーが固定source54986a9の一回科学runを明示認可。grantだけを2157da1で公開し、253 remote pathと環境・資源を照合後に一回起動した。原status {status}、reason {reason}。sourceや科学planの修正・retry/resume・追加候補・H8なし。\n\n[原結果・欠測・実行監査](../track_a_h6_matched_accuracy_result_v2.md)。signal {len(completed_signals)}/92、保存wrapper record {len(wrappers)}。旧source/freezes/入力/結果とdirty/untrackedを保全し、原記録を公開・remote照合してmandatory STOP。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION / next_stage_authorized=false。科学的GO/STOPをCodexで代行しない。\n')
    prefix_paths={
        'PROJECT_MAP.md':'docs/research/track_a_h6_matched_accuracy_result_v2.md',
        'VALIDATION_STATUS.md':'docs/research/track_a_h6_matched_accuracy_result_v2.md',
        'docs/README.md':'research/track_a_h6_matched_accuracy_result_v2.md',
        'docs/research/README.md':'track_a_h6_matched_accuracy_result_v2.md',
        'docs/research/研究概要・現状.md':'track_a_h6_matched_accuracy_result_v2.md',
        'docs/research/数値実験・評価計画.md':'track_a_h6_matched_accuracy_result_v2.md',
        'docs/research/研究ノート/README.md':'2026-10-11_track_a_h6_matched_execution_v2.md'}
    insertions={};clean_texts={}
    for path,link in prefix_paths.items():
        block=f'''## 2026-10-11 Track A H6精度一致比較 v2：一回実行・原結果公開・mandatory STOP

[結果・欠測・実行監査]({link})。固定source54986a9・公開v2 manifestに対する一回grant2157da1をremote照合後起動。
linear H6/1.00Å/STO-3G/tol-only rank19/sector400/保存state/T0.8、prefix5/10/15・full19 S2/S4、92候補・4精度・最大1708 wrapperを維持。
原status `{status}`、signal record {len(completed_signals)}/92、保存wrapper record {len(wrappers)}。未取得値・未登録費用task数を推測しない。
CPU [0,2,5,6]・signal4 thread・cost最大2 process、coordinator AS soft2/hard8 GiB・worker8/8、phase/total wall cap=null。source216/入力33・旧v1 STOP/旧pilot/dirty/untrackedを保全。
原結果・外部静的audit・hash inventory・欠測を公開、remote再取得照合。empirical u/有限費用標本でformal winnerやground-stateを認定しない。
一回grant消費済み、retry/resumeなし、mandatory STOP。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、next_stage_authorized=false、H8・追加探索未認可。
以下は各段階当時の履歴。'''
        insertions[path],clean_texts[path]=add_prefix(path,block)
    manifest_path='artifacts/validation_manifest.json'
    entry={'status':status,'source_commit':m['source_commit'],'grant_publication_commit':GRANT_COMMIT,
        'validation_document':result_path,'artifact_directory':str(META.relative_to(ROOT)),
        'raw_output_directory':str(OUT.relative_to(ROOT)),'runner_invocations':1,'one_shot_authorization_consumed':True,
        'signal_candidates_registered':92,'signal_records_saved':len(completed_signals),
        'saved_wrapper_records':len(wrappers),'completed_cost_rows':len(cost_rows),'stop_reason':reason,
        'static_audit_exit_code':checked.returncode,'external_source_input_verified':True,
        'environment_equal_after':env_equal,'source216_exact':True,'parent33_exact':True,
        'phase_wall_seconds':None,'total_wall_seconds':None,'formal_winner_certified':False,'ground_state_certified':False,
        'retry':False,'resume':False,'mandatory_stop':True,'next_stage_authorized':False,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
        'H8_authorized':False,'additional_search_authorized':False}
    old=(ROOT/manifest_path).read_text(); clean=git('show','HEAD:'+manifest_path).decode()
    key='track_a_h6_matched_execution_result_v2'
    assert key not in json.loads(old) and key not in json.loads(clean)
    insertion=json.dumps({key:entry},ensure_ascii=False,indent=2)[2:-2]+',\n'
    assert old.startswith('{\n') and clean.startswith('{\n')
    changed=old[:2]+insertion+old[2:];json.loads(changed);(ROOT/manifest_path).write_text(changed)
    insertions[manifest_path]=insertion;clean_texts[manifest_path]=clean[:2]+insertion+clean[2:]
    for path,sha in baseline['file_hashes'].items():
        data=(ROOT/path).read_bytes()
        if path in insertions:
            ins=insertions[path].encode();assert data.count(ins)==1;data=data.replace(ins,b'',1)
        assert hashlib.sha256(data).hexdigest()==sha,path
    old_sources=load(ROOT/'artifacts/resource_applicability/track_a_h6_matched_preparation_v1/2026-10-11/sealed_manifest_v1.json')['source_hashes']
    for path,sha in old_sources.items():assert file_hash(ROOT/path)==sha,path
    legacy=ROOT/'artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1'
    v1raw=ROOT/'artifacts/resource_applicability/track_a_h6_matched_accuracy_v1/2026-10-11/launch_v1'
    assert len(project_paths(legacy))==142 and len(project_paths(v1raw))==10
    preservation={'status':'PRESERVATION_PASS','baseline_head':baseline['head'],
        'baseline_file_hashes':baseline['file_hashes'],'baseline_status_porcelain_z':baseline['status'],
        'baseline_after_removing_declared_prefix_matches':True,'additive_insertions':insertions,
        'source216_exact':True,'old_source211_exact':True,'parent33_exact':True,
        'legacy142_hashes':{p:file_hash(ROOT/p) for p in project_paths(legacy)},
        'old_v1_raw10_hashes':{p:file_hash(ROOT/p) for p in project_paths(v1raw)},
        'existing_dirty_and_untracked_preserved':True,'old_freezes_and_grants_preserved':True,
        'runtime_source_changed':False,'retry':False,'resume':False,'mandatory_stop':True,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION','next_stage_authorized':False}
    save(META/'preservation_v2.json',preservation)
    evidence={'status':status+'_EVIDENCE_READY','source_commit':m['source_commit'],
        'grant_publication_commit':GRANT_COMMIT,'source_hashes':m['source_hashes'],
        'input_identity':m['input_identity'],'raw_file_hashes':raw_hashes,'raw_file_count':len(raw_hashes),
        'raw_total_bytes':raw_bytes,'metadata_file_hashes':{p:file_hash(ROOT/p) for p in project_paths(META)},
        'audit_exit_code':checked.returncode,'scientific_GO_issued':False,'mandatory_stop':True,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION','next_stage_authorized':False}
    save(META/'execution_evidence_inventory_v2.json',evidence)
    new_paths=project_paths(OUT)+project_paths(META)+[result_path,note_path]
    new_paths=[p for p in new_paths if not git('ls-files','--',p).strip()]
    for path in new_paths:
        assert (ROOT/path).stat().st_size<100*2**20,'GitHub file size limit: '+path
        git('add','-f','--',path)
    for path,text in clean_texts.items():stage_clean(path,text)
    staged=set(git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0'))
    assert staged==set(new_paths)|set(clean_texts),'Unexpected staged files.'
    git('diff','--cached','--check')
    verify_sources(ROOT,m['source_commit'],m['source_hashes']);assert verify_parent(ROOT)==m['input_identity']
    git('-c','maintenance.auto=false','-c','gc.auto=0','commit','-m','Publish one H6 matched-resource v2 run, missing coverage and audit')
    result_commit=git('rev-parse','HEAD').decode().strip()
    git('-c','maintenance.auto=false','-c','gc.auto=0','push','origin','HEAD:refs/heads/'+BRANCH)
    bare=tempfile.mkdtemp(prefix='h6_matched_v2_result_remote_')
    subprocess.run(['git','init','--bare','--quiet',bare],check=True);fetch(bare)
    expected=dict(m['source_hashes']);expected.update(m['input_identity']['parent_raw_hashes'])
    expected.update(raw_hashes)
    expected.update({p:file_hash(ROOT/p) for p in project_paths(META)})
    expected.update(preservation['legacy142_hashes']);expected.update(preservation['old_v1_raw10_hashes'])
    for path in staged:expected[path]=hashlib.sha256(git('show','HEAD:'+path)).hexdigest()
    expected[str((PREP/'sealed_manifest_v2.json').relative_to(ROOT))]=file_hash(PREP/'sealed_manifest_v2.json')
    expected[str((META/'authorization_v2.json').relative_to(ROOT))]=file_hash(META/'authorization_v2.json')
    verify_remote(bare,result_commit,expected)
    for path,sha in m['source_hashes'].items():
        assert hashlib.sha256(remote_bytes(bare,m['source_commit'],path)).hexdigest()==sha,path
    proof={'status':'REMOTE_RESULT_SOURCE_INPUT_GRANT_PASS','verified_at_utc':now(),
        'repository':'HIROMU1015/Partially-Randomized-Trotter','branch':BRANCH,'result_commit':result_commit,
        'grant_commit':GRANT_COMMIT,'source_commit':m['source_commit'],'remote_path_hashes':expected,
        'remote_paths_verified':len(expected),'raw_files_verified':len(raw_hashes),
        'source216_exact':True,'parent33_exact':True,'legacy142_exact':True,'old_v1_raw10_exact':True,
        'scientific_status':status,'mandatory_stop':True,'next_stage_authorized':False,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
        'H8_authorized':False,'additional_search_authorized':False,'retry':False,'resume':False}
    proof_path=META/'remote_result_verification_v2.json';save(proof_path,proof)
    git('add','--',str(proof_path.relative_to(ROOT)));git('diff','--cached','--check')
    assert git('diff','--cached','--name-only').decode().strip()==str(proof_path.relative_to(ROOT))
    git('-c','maintenance.auto=false','-c','gc.auto=0','commit','-m','Record remote byte verification for H6 matched-resource v2')
    final_commit=git('rev-parse','HEAD').decode().strip()
    git('-c','maintenance.auto=false','-c','gc.auto=0','push','origin','HEAD:refs/heads/'+BRANCH)
    fetch(bare);expected[str(proof_path.relative_to(ROOT))]=file_hash(proof_path);verify_remote(bare,final_commit,expected)
    for path,sha in baseline['file_hashes'].items():
        data=(ROOT/path).read_bytes()
        if path in insertions:data=data.replace(insertions[path].encode(),b'',1)
        assert hashlib.sha256(data).hexdigest()==sha,path
    assert not git('diff','--cached','--name-only')
    final={'status':'PUBLISHED_VERIFIED_MANDATORY_STOP','recorded_at_utc':now(),'scientific_status':status,
        'grant_commit':GRANT_COMMIT,'result_commit':result_commit,'final_commit':final_commit,
        'remote_verified_paths':len(expected),'raw_files':len(raw_hashes),'signal_records':len(completed_signals),
        'wrapper_records':len(wrappers),'audit_exit_code':checked.returncode,'baseline_preserved':True,
        'retry':False,'resume':False,'mandatory_stop':True,'next_stage_authorized':False,'H8_authorized':False}
    save('/tmp/h6_matched_v2_publication_complete.json',final)
    print(json.dumps(final,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-head',required=True)
    p.add_argument('--runner-pid',type=int);p.add_argument('--runner-start-ticks',type=int);p.add_argument('--runner-exit-code',type=int)
    a=p.parse_args()
    with Path('/tmp/h6_matched_v2_publication_claim.json').open('x') as f:
        json.dump({'publisher_pid':os.getpid(),'started_at_utc':now(),'expected_head':a.expected_head,
            'runner_pid':a.runner_pid,'runner_start_ticks':a.runner_start_ticks,
            'scientific_runner_invoked_by_publisher':False,'retry':False,'resume':False},f,indent=2)
    if a.runner_pid:
        assert a.runner_start_ticks is not None
        while alive(a.runner_pid,a.runner_start_ticks):time.sleep(15)
    while workers_live():time.sleep(1)
    finalize(a.expected_head,a.runner_exit_code)

if __name__=='__main__':
    try:main()
    except Exception:
        detail=traceback.format_exc()
        path=Path('/tmp/h6_matched_v2_publication_failure.json')
        if not path.exists():save(path,{'status':'PUBLICATION_BLOCKED_NO_SCIENTIFIC_RETRY','recorded_at_utc':now(),
            'traceback':detail,'raw_outputs_preserved':True,'retry':False,'resume':False,'next_stage_authorized':False})
        print(detail,file=sys.stderr,flush=True);raise
