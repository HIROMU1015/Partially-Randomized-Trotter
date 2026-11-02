"""Add execution documentation and adapt new run05 preflight, no science."""
import ast,hashlib,json,os
from pathlib import Path

BASE=Path('/home/AbeHiromu/projects/partially-randomized-trotter')
ROOT=Path('/tmp/track-a-h4-lazy-identity-run05-20261006')
BUNDLE=ROOT/'artifacts/resource_applicability/track_a_h4_lazy_identity_run05/2026-10-06'
CONTROL=BASE/'.server-preparation/executions/h4-signal-compile-run05-lazy-identity'
OLDCTL=BASE/'.server-preparation/executions/h4-signal-compile-run04-streaming-monitor'
checks=json.loads((BUNDLE/'binding_checks_v1.json').read_bytes())
reuse=json.loads((BUNDLE/'input_reuse_and_prior_budget_v1.json').read_bytes())

def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,text):
    with path.open('x') as f:f.write(text)
def save(path,v):write(path,json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n')

body=f'''# H4 run05：全identity変換をlazy化・12 worker再実行

利用者の増員再実行指示と「続きを行って」に基づく修正。旧run04はfresh検査PASS後12 owned workersを起動し、5件の新compileを投入してmonitor interval/freshness STOP。完成compile/signalは0、全own processの終了を確認して全証跡を保持した。失敗stackはJSON encodingの前に巨大exact container treeを作る箇所だった。5秒deadlineを緩和せず、normalize/encodeを一緒にlazy走査し、64KiBごとにSHAへ渡してprocess内sleep(0)を使う。同じndarrayのparameter metadataは1回のserialize内で共有し、変更可能なglobal cacheやcircuitsの置換は使わない。

75 local tests（67関連tests＋正しいcounter初期化による人工serialization8件）がPASS、failures/errors/skips0。追加transpile0。最初のserialization8 pytestには既存runner COUNTS初期化不足による6fail/2passがあり、その履歴を保持した。controlled relative phase、params、Unicode、signed zero、ordering、nonfinite STOP、serial/12worker集計とlazy hash byte同値性を検査した。科学データを使わない256×256 complex-hex metadata128繰返しでは旧/新hashは8bc4b0421dd538a1c9078229e66ea184da5de5cc57b6c45a6485f9ffcf9cf216で一致。全監視PASS、39.49秒で終了。旧全JSON29.42秒/encodingだけstreaming59.60秒は診断比較であり、production速度や完了を保証しない。

- branch：track-a-h4-lazy-identity-run05-20261006
- SOURCE_COMMIT：{checks['source_commit']}
- actual checkout：{ROOT}。/home容量条件を圧迫しない/tmpの独立worktree。
- plan SHA：{checks['plan_sha256']}
- plan fingerprint：{checks['plan_fingerprint']}
- auth digest：{checks['authorization_digest']}
- review digest：{checks['review_digest']}
- CPU [3,5,6,7,8,9,10,11,12,13,14,15]、12 owned workers、own新driverだけmask0xffe8、内部thread1。
- fresh memory>=120GiB、nonroot available disk>=3.5GiB、inode>=560000、quota確認、PSI0/OOM増加0。
- cumulative carry：165184714 bytes、12 consumed invocations、{reuse['prior_wall_seconds']}s。
- unchanged caps：10GiB累積charge/72h累積wall/74784 actual invocations。新attempt残invocation74772。
- driver/worker AS/RSS各8GiB、headroom16GiB、monitor freshness5秒。上限は増やさない。

H4 linear neutral singlet/STO-3G、4spatial/8system+ancilla1、DF12。凍結6距離0.70/0.80/0.90/1.10/1.40/1.60Åを元generation source049e69919af16ad29a67a217dc7a407d6b1754a6と元freeze/array identities付きでread-only再利用。分子アクセス・SCF/DF/state/input再生成をしない。T0.8、二次DF-prefix PF/canonical finite-RTE、218templates/距離、L_D0/3/4/5/6/9/12、q1/2/4/8、delta0.8/0.4/0.2/0.1、固定r/K、32paired trajectories/master20261006。科学式/回路/parameter数値とcompiler optionsは不変。SOURCE_COMMITを含む従来seed identityへ新sourceを結合するため旧random結果を混ぜない。

fresh gateはsource19/45dependencies/compiler・実checkout/plan/auth/review、6NPZ stream SHA/元freeze、run02/03/04 journal不変、全旧own processes停止、CPU online/cpuset/NUMA/physical cores/SMT低負荷/全祖先CPU quota、memory/pressure/OOM、fixed outputのmount/quota/available bytes/inodes、新output不在を照合。不合格や欠測ならSTOP。利用者の実認可を別reviewへ転記し外部reviewerを捏造しない。旧false草案は保持。

candidate間bounded compileはmax12 outstanding、候補/trajectory/axis集計順とcache scopeを保持。一度だけ実行し完成時signal1308/logicalwrappers74784のMAP_COMPLETE_STOP。失敗時もrunnerは自動retry/resumeせず全証跡を保持。旧source/入力/runtime/cache/checkpointを削除/移設/上書きしない。GPU/共有設定/他job変更なし、H6/TrackB/追加trajectory/長RPE/最終総costへ進まない。

実起動前source/plan/auth/reviewは本bundleに固定し、runnerはこのcheckoutのscripts/resource_applicability/run_h4_geometry_signal_compile.py。fresh検査後だけabsolute pathで一度execする。control入口：{CONTROL}/README.md。sourceはlocal実装検査済みでproduction成功・全campaign完了は未確定。科学NPZ/runtime/cache/checkpointはcommitしない。

[bundle入口](../../artifacts/resource_applicability/track_a_h4_lazy_identity_run05/2026-10-06/README.md)。
'''

doc='docs/research/track_a_h4_lazy_identity_run05.md';write(ROOT/doc,body)
write(BUNDLE/'README.md','# H4 streaming monitor run05\n\n[資料入口](../../../../docs/research/track_a_h4_lazy_identity_run05.md)。\n\n75 local tests PASS、旧一括encoder monitor STOPと新streaming encoder monitor PASSは同じ人工JSON digest。旧runの科学的停止原因は未確定。CPU12件/12 worker、旧入力再利用、全budget累積、fresh gate後one shot MAP_COMPLETE_STOP。production成功は未確認。source_freeze / input_reuse_and_prior_budget / binding_checks / plan / auth / review / stage_storage_estimate / test_resultを一組で参照。\n')
links={'PROJECT_MAP.md':doc,'docs/README.md':'research/track_a_h4_lazy_identity_run05.md','docs/research/README.md':'track_a_h4_lazy_identity_run05.md','scripts/README.md':'../'+doc,'src/trotterlib/README.md':'../../'+doc,'docs/research/研究概要・現状.md':'track_a_h4_lazy_identity_run05.md','docs/research/研究ノート/README.md':'../track_a_h4_lazy_identity_run05.md','docs/research/研究ノート/2026-10-06.md':'../track_a_h4_lazy_identity_run05.md'}
for path,link in links.items():
    with (ROOT/path).open('a') as f:f.write('\n\n## H4 run05：identity hash分割・5秒監視維持\n\n[run05固定sourceと再実行binding]('+link+')を追加した。run04はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。75 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/12 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。\n')
manifest=ROOT/'artifacts/validation_manifest.json';raw=manifest.read_text();obj=json.loads(raw)
entry={'path':'artifacts/resource_applicability/track_a_h4_lazy_identity_run05/2026-10-06','kind':'directory','supports':['h4_server_execution_implementation'],'reason':'Exact streaming identity hash, artificial JSON monitor diagnostic and75 local tests PASS; user-authorized run05 cumulative-budget binding. Production completion pending; no scientific NPZ/runtime/cache committed.'}
start=raw.index('[',raw.index('"present":',raw.index('"artifact_inventory":')));quoted=False;escape=False;depth=0
for i in range(start,len(raw)):
    ch=raw[i]
    if quoted:
        if escape:escape=False
        elif ch=='\\':escape=True
        elif ch=='"':quoted=False
    elif ch=='"':quoted=True
    elif ch=='[':depth+=1
    elif ch==']':
        depth-=1
        if depth==0:end=i;break
before=raw[:end].rstrip();raw=before+',\n'+'\n'.join('      '+l for l in json.dumps(entry,ensure_ascii=False,indent=2).splitlines())+raw[len(before):]
status={'status':'implementation_locally_validated_awaiting_fresh_one_shot_launch','source_commit':checks['source_commit'],'tests':75,'new_test_transpiles':0,'workers':12,'original_input_generation_preserved':True,'prior_consumed_invocations':7,'old_run04_status':'monitor_STOP','old_exact_monitor_cause_unknown':True,'metadata_GIL_starvation_mechanism_reproduced':True,'mandatory_completion_stop':'MAP_COMPLETE_STOP','entry':doc}
end=raw.rfind('}');snippet='"h4_streaming_monitor_run05": '+json.dumps(status,ensure_ascii=False,indent=2);raw=raw[:end].rstrip()+',\n'+'\n'.join('  '+s for s in snippet.splitlines())+'\n'+raw[end:];json.loads(raw);manifest.write_text(raw)

wrapper=(OLDCTL/'preflight_then_exec_fixed_runner.py').read_text().replace('h4-run04-twelve-worker','h4-run05-twelve-worker')
wrapper=wrapper.replace("reused['prior_actual_invocations']==7", "reused['prior_actual_invocations']==12").replace("'prior_actual_invocations':7", "'prior_actual_invocations':12")
wrapper=wrapper.replace('h4-signal-compile-run03-revision/process_record_v1.json','h4-signal-compile-run04-streaming-monitor/process_record_v1.json').replace('run03 own','run04 own')

ast.parse(wrapper);write(CONTROL/'preflight_then_exec_fixed_runner.py',wrapper)
write(CONTROL/'old_owned_run_stopped_v1.json',(OLDCTL/'old_owned_run_stopped_v1.json').read_text())
cfg=json.loads((OLDCTL/'launch_configuration_v1.json').read_bytes())
paths={'plan':'signal_compile_plan_v1.json','authorization':'authorization_user_approved_v1.json','review':'stage_review_user_approved_v1.json'}
cfg.update(control_root=str(CONTROL),source_root=str(ROOT),output_root=json.loads((BUNDLE/paths['plan']).read_bytes())['output_root'],wrapper_sha256=sha(wrapper.encode()))
for key,name in paths.items():cfg[key]=str(BUNDLE/name);cfg[key+'_sha256']=sha((BUNDLE/name).read_bytes())
save(CONTROL/'launch_configuration_v1.json',cfg)
write(CONTROL/'README.md','# H4 run05: streaming identity hash; twelve owned workers\n\nUser continued authorized fix/increase/reexecute. SOURCE '+checks['source_commit']+'. Source checkout '+str(ROOT)+'.\n\nCPU '+str(checks['CPU_list'])+', own-run mask0xffe8,12 workers, internal threads1. Reuses original six frozen NPZ; no input generation, shared settings/other jobs/GPU changes. Carries old175168060bytes/12invocations/'+str(reuse['prior_wall_seconds'])+'s into unchanged10GiB/74784/72h caps. Fresh source/hash/CPU/quota/memory120GiB/disk3.5GiB/inodes560000/PSI/OOM gates required. One shot; no auto retry/resume; final MAP_COMPLETE_STOP.\n\nSee fresh_preflight_result,process_record,startup_observation and runner_stdout_stderr.log after submission.\n')
with (BUNDLE/'materials_method_v1.py').open('xb') as f:f.write(Path(__file__).read_bytes())
save(BUNDLE/'artifact_manifest_v1.json',{'schema_version':'h4-run05-lightweight-artifact-manifest-v1','source_commit':checks['source_commit'],'scientific_NPZ_runtime_cache_excluded':True,'files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(BUNDLE.iterdir()) if p.is_file()]})
print(json.dumps({'documents':'written','preflight_AST':'PASS','CPU_list':checks['CPU_list'],'workers':12,'source':checks['source_commit'],'science_started':False}))
