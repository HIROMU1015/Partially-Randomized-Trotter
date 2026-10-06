"""Add execution documentation and adapt new run04 preflight, no science."""
import ast,hashlib,json,os
from pathlib import Path

BASE=Path('/home/AbeHiromu/projects/partially-randomized-trotter')
ROOT=Path('/tmp/track-a-h4-streaming-monitor-run04-20261006')
BUNDLE=ROOT/'artifacts/resource_applicability/track_a_h4_streaming_monitor_run04/2026-10-06'
CONTROL=BASE/'.server-preparation/executions/h4-signal-compile-run04-streaming-monitor'
OLDCTL=BASE/'.server-preparation/executions/h4-signal-compile-run03-revision'
checks=json.loads((BUNDLE/'binding_checks_v1.json').read_bytes())
reuse=json.loads((BUNDLE/'input_reuse_and_prior_budget_v1.json').read_bytes())

def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,text):
    with path.open('x') as f:f.write(text)
def save(path,v):write(path,json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n')

body=f'''# H4 run04：exact fingerprint分割・監視維持・12 worker再実行

利用者の「その修正を入れて、ワーカー数を増やして再実行して」と、停止後の「続きを行って」に基づく。
local実装検査はPASS。科学的成功・全campaign完了は未確認。起動直前fresh gatesがPASSの場合だけ、
新run04を一度起動しMAP_COMPLETE_STOPで停止する。自動retry/resumeはしない。

## 停止の証拠と修正

run03はfresh preflight PASS後12 owned workersを起動し、候補間に3 invocationsを投入した後
monitor STOPで終了した。compile完成0、signal0。driver/全worker終了をread-only確認した。
旧source・run01/run02/run03・全input/partial records/journals/logsを削除・移設・上書きしない。
旧監視の元の例外は保存されなかったため、run03そのものの正確な停止原因は未確定である。

人工JSONメタデータ（256×256のcomplex-hex parameter metadataを128回繰返し、科学array/回路0）で、
旧fingerprintの一括json.dumpsが監視threadを妨げ、monitor interval/freshnessを再現した。
旧観測のsince_previous_checkは6.6886 s、5秒上限を超えた。JSON処理のGIL占有が
run03の不規則な監視間隔と整合する有力な原因であり、実runの唯一の原因を証明したとはしない。

fingerprintは同じexact変換・sort keys・UTF-8・separators・nonfinite禁止を維持し、
JSONEncoder.iterencodeでencodingを分割して64 KiB単位でSHA256へ渡し、own process内でsleep(0)する。
同じ大きいpayloadのdigestは旧/新とも8bc4b0421dd538a1c9078229e66ea184da5de5cc57b6c45a6485f9ffcf9cf216。
新方式の診断は全監視PASSでthreadも終了した。この診断のelapsedは旧29.42s/新59.60sで、
新方式を高速化結果とは主張しない。巨大な単一encoded stringの追加allocationも避ける。
65直接関連testsはPASS（failures/errors/skips0）。追加transpile・分子/科学array・production workersは0。
signed zero、complex、極端finite float、Unicode、ordering、unsupported/nonfinite STOP、
旧canonical SHAとの一致、候補間queue、seed/axis/metrics、台帳/認可、監視原因ログを検査した。

監視5秒、driver/worker AS/RSS8 GiB、headroom16 GiB、PSI0/OOM増加0を緩和しない。
watcher停止時は最初の例外内容をbounded driver logへ出し、cleanup中の例外に隠さない。
候補間compile queue、最大12 outstanding invocations、logical集計順、cache scopeはrun03修正を維持する。
compiler options/45 dependencies・科学input/math/circuit条件・seed算法/masterは変更しない。
SOURCE_COMMITがidentity/seedへ入る従来仕様に従い、新sourceへ再結合した結果だけを新runへ保存する。
旧random/partial metricsを新sourceの結果へ混合しない。

## 固定条件・入力・累積予算

- SOURCE_COMMIT：{checks['source_commit']}
- branch：track-a-h4-streaming-monitor-run04-20261006
- actual checkout：{ROOT}。/homeのoutput空きを圧迫しない別filesystemの独立worktree。共有設定変更なし。
- plan SHA：{checks['plan_sha256']}
- plan fingerprint：{checks['plan_fingerprint']}
- auth digest：{checks['authorization_digest']}
- review digest：{checks['review_digest']}
- CPU [3,5,6,7,8,9,10,11,12,13,14,15]、12 owned workers、own新規driverだけmask0xffe8。
- fresh memory>=120 GiB、quota確認済みnonroot disk>=3.5 GiB、available inode>=560000。
- run02の凍結6入力をread-only再利用、生成source049e69919af16ad29a67a217dc7a407d6b1754a6・freeze・32 arrays identitiesを保持。
- old charge165168060 bytes（run02を含むrun03 journal）をcarryし、新journal128-byte carry rowも課金する。
- old invocations7件を含めactual cap74784、new attempt残上限74777。
- prior wall{reuse['prior_wall_seconds']}sをcarryし72h累積上限。run03最終failure logまでの経過+60sを保守的に含む。
- 全campaign10 GiB charge cap。入力再生成なし。stage容量planningは既存12 worker小計3576795136 bytesを維持する。

H4 linear neutral singlet/STO-3G、4 spatial/8 system+ancilla1、DF12。
6距離0.70/0.80/0.90/1.10/1.40/1.60 Å、T=0.8、二次DF-prefix PF/canonical finite-RTE。
218 templates/距離、L_D=0/3/4/5/6/9/12、q=1/2/4/8、delta=0.8/0.4/0.2/0.1。
固定r/K、random32 paired trajectories/master20261006。signal1308・logical wrappers74784、
H6/Track B/追加trajectory/anchor/長RPE/最終総costへ進まない。

## launch直前と記録

source19/45 dependencies/compiler・actual checkout/plan/auth/review・old freeze・6 NPZ streaming SHAをfresh照合する。
run02/03 journalsの不変性と累積charge・invocation/wall、旧own PID/worker全終了、新output不在を再確認する。
CPU online/cpuset/12 distinct cores/NUMA0・3秒受動SMT load・全祖先CPU quota、memory/pressure/OOM、
fixed output mount/user-group-project quota/available bytes/inodesを確認する。不合格・欠測なら科学runnerを呼ばずSTOP。
承認reviewは実際の利用者指示の転記であり、外部reviewerの承認を捏造しない。旧false草案を保持する。

source/plan/auth/reviewは本bundleに固定する。runnerはこのcheckoutの
scripts/resource_applicability/run_h4_geometry_signal_compile.py。
control入口：{CONTROL}/README.md。
実起動argv/時刻/PID・fresh preflight・progress・最初のfailure reasonはcontrolの別file/logへ記録する。
科学NPZ・production runtime/checkpoint/cacheはcommitしない。

[bundle入口](../../artifacts/resource_applicability/track_a_h4_streaming_monitor_run04/2026-10-06/README.md)。
'''
doc='docs/research/track_a_h4_streaming_monitor_run04.md';write(ROOT/doc,body)
write(BUNDLE/'README.md','# H4 streaming monitor run04\n\n[資料入口](../../../../docs/research/track_a_h4_streaming_monitor_run04.md)。\n\n65 local tests PASS、旧一括encoder monitor STOPと新streaming encoder monitor PASSは同じ人工JSON digest。旧runの科学的停止原因は未確定。CPU12件/12 worker、旧入力再利用、全budget累積、fresh gate後one shot MAP_COMPLETE_STOP。production成功は未確認。source_freeze / input_reuse_and_prior_budget / binding_checks / plan / auth / review / stage_storage_estimate / test_resultを一組で参照。\n')
links={'PROJECT_MAP.md':doc,'docs/README.md':'research/track_a_h4_streaming_monitor_run04.md','docs/research/README.md':'track_a_h4_streaming_monitor_run04.md','scripts/README.md':'../'+doc,'src/trotterlib/README.md':'../../'+doc,'docs/research/研究概要・現状.md':'track_a_h4_streaming_monitor_run04.md','docs/research/研究ノート/README.md':'../track_a_h4_streaming_monitor_run04.md','docs/research/研究ノート/2026-10-06.md':'../track_a_h4_streaming_monitor_run04.md'}
for path,link in links.items():
    with (ROOT/path).open('a') as f:f.write('\n\n## H4 run04：identity hash分割・5秒監視維持\n\n[run04固定sourceと再実行binding]('+link+')を追加した。run03はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。65 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/7 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。\n')
manifest=ROOT/'artifacts/validation_manifest.json';raw=manifest.read_text();obj=json.loads(raw)
entry={'path':'artifacts/resource_applicability/track_a_h4_streaming_monitor_run04/2026-10-06','kind':'directory','supports':['h4_server_execution_implementation'],'reason':'Exact streaming identity hash, artificial JSON monitor diagnostic and65 local tests PASS; user-authorized run04 cumulative-budget binding. Production completion pending; no scientific NPZ/runtime/cache committed.'}
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
status={'status':'implementation_locally_validated_awaiting_fresh_one_shot_launch','source_commit':checks['source_commit'],'tests':65,'new_test_transpiles':0,'workers':12,'original_input_generation_preserved':True,'prior_consumed_invocations':7,'old_run03_status':'monitor_STOP','old_exact_monitor_cause_unknown':True,'metadata_GIL_starvation_mechanism_reproduced':True,'mandatory_completion_stop':'MAP_COMPLETE_STOP','entry':doc}
end=raw.rfind('}');snippet='"h4_streaming_monitor_run04": '+json.dumps(status,ensure_ascii=False,indent=2);raw=raw[:end].rstrip()+',\n'+'\n'.join('  '+s for s in snippet.splitlines())+'\n'+raw[end:];json.loads(raw);manifest.write_text(raw)

wrapper=(OLDCTL/'preflight_then_exec_fixed_runner.py').read_text().replace('h4-run03-twelve-worker','h4-run04-twelve-worker')
old='''    require(charged==reused['prior_cumulative_charge_bytes'] and charged+128+CONFIG['stage_charge_bound']<=10*2**30, 'cumulative bytes not reset')
    require(reused['prior_actual_invocations']==4 and 0<=reused['prior_wall_seconds']<72*3600, 'cumulative invocation/wall')'''
new='''    predecessor=reused['predecessor_budget_journal']
    prior_data=(Path(predecessor['root'])/'byte-budget.journal').read_bytes()
    require(hashlib.sha256(prior_data).hexdigest()==predecessor['sha256'] and len(prior_data)%128==0, 'run03 stopped journal bytes')
    charged=sum(int(prior_data[i:i+128].strip()) for i in range(0,len(prior_data),128))
    require(charged==reused['prior_cumulative_charge_bytes'] and charged+128+CONFIG['stage_charge_bound']<=10*2**30, 'cumulative bytes not reset')
    require(reused['prior_actual_invocations']==7 and 0<=reused['prior_wall_seconds']<72*3600, 'cumulative invocation/wall')
    old_driver=json.loads((Path(CONFIG['base_root'])/'.server-preparation/executions/h4-signal-compile-run03-revision/process_record_v1.json').read_bytes())
    p=Path('/proc/%d/stat'%old_driver['pid'])
    if p.exists():
        fields=p.read_text().rsplit(')',1)[1].split()
        require(fields[19]!=old_driver['proc_starttime_ticks'] or fields[0]=='Z', 'run03 own driver remains')
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            if d.stat().st_uid!=os.getuid():continue
            cmd=(d/'cmdline').read_bytes()
            require(b'--owned-worker '+str(old_driver['pid']).encode()+b'\\0' not in cmd,'run03 own worker remains')
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass'''
assert old in wrapper;wrapper=wrapper.replace(old,new).replace("'prior_actual_invocations':4","'prior_actual_invocations':7")
ast.parse(wrapper);write(CONTROL/'preflight_then_exec_fixed_runner.py',wrapper)
write(CONTROL/'old_owned_run_stopped_v1.json',(OLDCTL/'old_owned_run_stopped_v1.json').read_text())
cfg=json.loads((OLDCTL/'launch_configuration_v1.json').read_bytes())
paths={'plan':'signal_compile_plan_v1.json','authorization':'authorization_user_approved_v1.json','review':'stage_review_user_approved_v1.json'}
cfg.update(control_root=str(CONTROL),source_root=str(ROOT),output_root=json.loads((BUNDLE/paths['plan']).read_bytes())['output_root'],wrapper_sha256=sha(wrapper.encode()))
for key,name in paths.items():cfg[key]=str(BUNDLE/name);cfg[key+'_sha256']=sha((BUNDLE/name).read_bytes())
save(CONTROL/'launch_configuration_v1.json',cfg)
write(CONTROL/'README.md','# H4 run04: streaming identity hash; twelve owned workers\n\nUser continued authorized fix/increase/reexecute. SOURCE '+checks['source_commit']+'. Source checkout '+str(ROOT)+'.\n\nCPU '+str(checks['CPU_list'])+', own-run mask0xffe8,12 workers, internal threads1. Reuses original six frozen NPZ; no input generation, shared settings/other jobs/GPU changes. Carries old165168060bytes/7invocations/'+str(reuse['prior_wall_seconds'])+'s into unchanged10GiB/74784/72h caps. Fresh source/hash/CPU/quota/memory120GiB/disk3.5GiB/inodes560000/PSI/OOM gates required. One shot; no auto retry/resume; final MAP_COMPLETE_STOP.\n\nSee fresh_preflight_result,process_record,startup_observation and runner_stdout_stderr.log after submission.\n')
with (BUNDLE/'materials_method_v1.py').open('xb') as f:f.write(Path(__file__).read_bytes())
save(BUNDLE/'artifact_manifest_v1.json',{'schema_version':'h4-run04-lightweight-artifact-manifest-v1','source_commit':checks['source_commit'],'scientific_NPZ_runtime_cache_excluded':True,'files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(BUNDLE.iterdir()) if p.is_file()]})
print(json.dumps({'documents':'written','preflight_AST':'PASS','CPU_list':checks['CPU_list'],'workers':12,'source':checks['source_commit'],'science_started':False}))
