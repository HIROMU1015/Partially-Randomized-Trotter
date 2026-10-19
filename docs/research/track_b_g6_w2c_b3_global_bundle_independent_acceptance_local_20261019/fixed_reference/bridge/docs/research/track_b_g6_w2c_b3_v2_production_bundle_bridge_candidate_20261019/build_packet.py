"""Build the requested technical handoff from saved artificial observations."""
from pathlib import Path
from hashlib import sha256
import collections,json,subprocess,shutil,sys,os
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_contract_v1 as K
MEASURED='e099dcc461247e81eff981867dc28c3f3b16df5e';P='596781daeba253418ba35f4b7964312a54caf233';I='77126767c00585d0967558c0756907df2d69d1ff'
def read(path):return json.loads((HERE/path).read_text())
def save(path,value):(HERE/path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def md(path,text):(HERE/path).write_text(text.strip()+'\n')
def main():
    c=read('technical_contract_candidate.json');matrix=read('runs_sealed/matrix.json');new=read('tests_evidence/results.json');old=read('legacy_development2/results.json');fixed=read('fixed_regressions_final/results.json');counts=dict(collections.Counter(x['verification_status']for x in matrix['cases']))
    assert new['successful']and new['run']==107 and len(new['mutation_observations'])==35
    assert old['original']['pass_count']==622 and old['original']['fail']==1 and old['corrected']['pass_count']==623
    ids=old['corrected']['suites'];allids=[x for row in ids for x in row['collected']]+new['collected']+[x for row in fixed for x in row['collected']];assert len(allids)==len(set(allids))==828
    assert all(x['measured_git_SOURCE_commit']==MEASURED for x in matrix['cases'])
    assert c==K.candidate(ROOT)
    save('bundle_manifest_candidate.json',K.manifest(c,'ARTIFICIAL_ONLY/STATIC_MANIFEST_CANDIDATE_NO_AUTHORITY',profile_n=256))
    save('artificial_acceptance_matrix.json',matrix)
    save('new_regression_results.json',new);save('legacy_regression_results.json',old);save('fixed_v2_regression_results.json',fixed)
    save('unique_regression_ids.json',dict(count=828,ids=allids,original_legacy_not_double_added=True,historical_108_680_820_not_added=True,matrix_and_mutations_are_in_107=True))
    normal=read('runs_sealed/normal_bundle_n256_verification.json');save('n256_artificial_cost_and_budget.json',dict(calls=normal['calls'],output_budget=normal['output_budget'],condition_byte_attribution=normal['condition_byte_attribution'],cost=normal['cost'],scientific_CPU_extrapolation=False,not_algorithm_intrinsic_cost=True))
    save('negative_mutation_results.json',dict(count=35,are_in_107_tests=True,observations=new['mutation_observations']))
    # Publish the current source/test evidence, while old artifacts stay fixed refs.
    delivery=HERE/'case_verifications';delivery.mkdir(exist_ok=True)
    for row in matrix['cases']:
        for suffix in ('_spec.json','_record.json','_result.json','_verification.json','.log'):
            source=HERE/'runs_sealed'/(row['case']+suffix)
            if source.exists():shutil.copyfile(source,delivery/source.name)
    logs=HERE/'regression_logs';logs.mkdir(exist_ok=True)
    for folder in ('legacy_development2','fixed_regressions_final','tests_evidence'):
        for p in (HERE/folder).glob('*.log'):shutil.copyfile(p,logs/(folder+'_'+p.name))
    diagnostic=HERE/'development_diagnostics';diagnostic.mkdir(exist_ok=True)
    for name in ('runs_development1','runs_development2','runs_development3','runs_final'):
        p=HERE/name/'matrix.json'
        if p.exists():shutil.copyfile(p,diagnostic/(name+'_matrix.json'))
    for name in ('tests_development1','legacy_development1'):
        p=HERE/name/'results.json'
        if p.exists():shutil.copyfile(p,diagnostic/(name+'_results.json'))
    delta=next(x['delta']for x in old['corrected']['suites']if x['delta']);save('legacy_test_delta.json',delta);md('legacy_test_minimal.diff',delta['diff'])
    closed={k:c[k]for k in K.CLOSED}
    verification=dict(schema='b3_bundle_candidate_verification_v1',status='ARTIFICIAL_TECHNICAL_CANDIDATE_COMPLETE_EXECUTION_CLOSED',measured_runtime_SOURCE_commit=MEASURED,producer_public_HEAD=P,independent_public_HEAD=I,**closed,
        new_tests=dict(run=107,pass_count=107,fail=0,error=0,skip=0,mutation_tests=35),original_legacy=dict(run=623,pass_count=622,fail=1,error=0,skip=0),versioned_legacy=dict(run=623,pass_count=623,fail=0,error=0,skip=0),existing_v2_storage_tests=41,existing_v2_independent_tests=57,unique_versioned_regression_ids=828,
        final_artificial_case_counts=counts,expected_non_success_fault_cases=39,mutation_rejections=dict(INTEGRITY_MISMATCH=32,INCOMPLETE_EVIDENCE=3),unresolved_unexpected_regression_failures=0,
        historical_public_cases=dict(total=108,normal=2,technical_failure=40,incomplete_evidence=66,unchanged=True),
        actual_scientific_G7_F3_imports=0,actual_scientific_sampling=0,new_scientific_calculations=0,scientific_SOURCE_adoption=0,real_authorizations_issued_or_consumed=0,real_grants_issued_or_consumed=0,real_claims_issued_or_consumed=0,native_quantum_build_compile_synthesis=0,subagents_used=False,third_party_independent_review=False,
        full_N_calls31710848='NOT_EXECUTED',pilot_scientific_calls16896='UNADOPTED_NOT_EXECUTED',n256_artificial_calls16896='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE',next_owner=['USER','GPT'])
    save('verification.json',verification)
    new_files=[p for p in HERE.glob('*.py')]+[ROOT/'tests/tracks/algorithm_codesign/test_g6_w2c_b3_bundle_v1.py']+list((ROOT/K.SOURCE).glob('g6_w2c_b3_bundle_*_v1.py'))+[HERE/'bundle.schema.json',HERE/'fixed_scientific_reference.json']
    inventory=[]
    for p in sorted(new_files):
        rel=str(p.relative_to(ROOT));b=p.read_bytes();old_bytes=subprocess.run(['git','show',MEASURED+':'+rel],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        inventory.append(dict(path=rel,bytes=len(b),sha256=sha256(b).hexdigest(),at_measured_SOURCE=old_bytes.returncode==0 and old_bytes.stdout==b))
    save('source_inventory.json',dict(schema='b3_bundle_source_inventory_v1',base_commit=P,measured_runtime_SOURCE_commit=MEASURED,previous_independent_commit=I,files=inventory,measured_runtime_pins=c['source_pins'],runtime_python=c['python'],test_SOURCE_sha256=new['source_sha256'],test_and_evidence_runner_revisions_after_runtime_measurement=True,old_SOURCE_modified=False))
    before=read('preservation_before.json');observations=[]
    for row in before:
        p=row['observation_only_worktree'];b=subprocess.check_output(['git','-C',p,'status','--porcelain=v1','-z']);head=subprocess.check_output(['git','-C',p,'rev-parse','HEAD'],text=True).strip();observations.append(dict(path=p,before=row,after_HEAD=head,after_status_sha256=sha256(b).hexdigest(),same_status_digest=sha256(b).hexdigest()==row['status_sha256'],observation_only=True,unrelated_changes_are_not_execution_gates=True))
    checked=[]
    for path,h in c['source_pins'].items():
        b=(ROOT/path).read_bytes();assert sha256(b).hexdigest()==h
        oldblob=subprocess.run(['git','show',P+':'+path],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        if oldblob.returncode==0:assert b==oldblob.stdout;checked.append(dict(path=path,sha256=h,base_bytes_unchanged=True))
    save('preservation_verification.json',dict(status='PASS_FOR_DECLARED_WRITE_SCOPE_AND_FIXED_SOURCE_BYTES',fixed_source_checks=checked,worktrees_observations=observations,write_scope=['new candidate worktree','fresh /tmp/b3-bundle-artificial-*','fresh owned mutation/restore directories'],old_raw_grants_ledgers_markers_STOP_opened_for_write=False,all_dirty_untracked_content_snapshot_claim=False,shared_OS_cgroup_packages_services_or_other_process_changes=False,old_independent_57_tests_readonly_sources_and_new_temp_only=True))
    save('resource_observations.json',dict(cpu_count=os.cpu_count(),affinity_available=len(os.sched_getaffinity(0)),memory=Path('/proc/meminfo').read_text(),pressure={p:Path('/proc/pressure/'+p).read_text()for p in ('cpu','memory','io')},artificial_matrix_parallel_processes=4,restore_verification_threads=8,no_global_system_changes=True,no_pressure_based_forced_worker_stop=True))
    budget=normal['output_budget'];art=read('artificial_raw_manifest.json')
    md('README.md',f'''# Storage V2 S/P1 Production Bundle Bridge 技術候補

**人工技術候補を実装・検証した。科学入口を閉鎖し、公開照合後 mandatory STOP。正式採択・科学的 GO はしていない。**

公開測定 SOURCE は `{MEASURED}`。Producer `{P}` を直接基点とし、独立受入 `{I}` を固定参照した。既存 branch の merge/rebase、旧 SOURCE/raw/claim の変更はない。

|最終集合|結果|
|---|---:|
|新規 bundle tests|107 PASS／FAIL・ERROR・SKIP 0|
|上記に含まれる mutation|35 拒否：integrity mismatch 32、証拠不完全 3|
|最終人工 matrix|43：bundle 正常 2、単独条件正常 2、技術失敗 13、証拠不完全 26|
|旧10suite 原版|622 PASS／1 FAIL|
|旧10suite versioned 最小補正版|623 PASS|
|既存 V2 storage／独立 consumer 回帰|41 PASS／57 PASS|
|重複しない補正版 regression ID|828 PASS（623＋41＋57＋107）|

43 matrix と35 mutation は107の内数。原版623件と補正版623件は重ねて加算しない。歴史108／680／820も加算しない。旧108の正常2／技術失敗40／証拠不完全66は固定参照のまま。

[Q1–Q10と次担当](handoff.md)、[architecture](bridge_architecture.md)、[技術契約](v2_technical_contract_candidate.md)、[差分](source_and_contract_delta.md)、[legacy保守](legacy_test_maintenance.md)、[global identity/予算](global_bundle_identity_and_budget.md)、[seed/F01](seed_prefix_and_condition_binding.md)、[Admission](admission_and_one_shot.md)、[STOP/storage](stop_unknown_and_storage_semantics.md)、[費用](cpu_wall_and_output_accounting.md)、[gate](production_gate_and_authority_boundaries.md)、[人工受入](artificial_acceptance_results.md)、[残件](pending_and_required_decisions.md)。

機械結果：[verification](verification.json)、[contract](technical_contract_candidate.json)、[static manifest](bundle_manifest_candidate.json)、[JSON schema](bundle.schema.json)、[matrix](artificial_acceptance_matrix.json)、[SOURCE inventory](source_inventory.json)、[固定公開照合](fixed_public_reference_verification.json)、[保全](preservation_verification.json)、[公開照合](publication_receipt.json)。

人工 raw は {len(art['parts'])} part、archive {art['archive_bytes']:,} bytes／SHA256 `{art['archive_sha256']}`。全 {len(art['members'])} member を記録した。正常raw＋35差分overlayを復元して再読できる。実行済み保存先への上書き・同じfake scopeの再使用はしない。

```bash
python3 {K.DOC}restore_evidence.py --new-directory /tmp/b3-bundle-new-restore
python3 {K.DOC}verify_restored.py
python3 {K.DOC}run_matrix.py reproduce --jobs 4
B3_BUNDLE_MATRIX={K.DOC}runs_reproduce python3 {K.DOC}run_tests.py reproduce
```

既存回帰は `prepare_legacy.py` → `run_legacy.py reproduce`。固定の旧独立57件は77126767の指定packetを復元した専用参照treeが必要であり、`run_fixed_regressions.py` は使用した同一サーバー上の参照pathを明示している。新環境で公開分割rawを復元するだけの手順と、同じruntime bytesを要するテスト再実行は別である。Python 3.12／標準ライブラリのみ。SOURCE/input/runtime はhashで固定する。

作者はprimary Codex一名。offline判定をProducerから分離したが、別reviewerによる独立レビューではない。次の担当は **ユーザーとGPT**。
''')
    md('bridge_architecture.md', '''# Architecture

bridge は Storage V2 `Coordinator` の versioned subclass を一つだけ生成する。S/P1 は共通の48-task plan、uint8 statefile、RAM shadow、global counter/cursor/ticket、byte budget、ordered trace、control journalを共有する。Sの3 phaseをjoin後、同じcoordinatorのsettingsをP1へ切り替える。条件ごとに新しいclaim・budget・statefileを作る構造ではない。

worker/F01/Stream/fused codec/CachedObserver/coalesced clientは固定SOURCEを再利用し、既存moduleは変更しない。新bridgeはcondition begin/end、global trial identity、global offset、PHASE_ABORT、post-close保存とcaller観測を追加する。S/P1各24pairはfresh spawn child。Sは逐次、P1は既存8-pair構造。固定順S→P1で科学反復1候補、順序反転は含めない。

```mermaid
flowchart TD
  A[単一 fake atomic claim: SPENT] --> B[単一 global coordinator / budget]
  B --> C[S: correctness → baseline → instrumented]
  C --> D{S完了}
  D -->|yes| E[P1: correctness → baseline → instrumented]
  D -->|no| X[STOP / drain / 部分証拠保持]
  E --> F[全worker join / summary / diagnostic]
  X --> F
  F --> G[fake Admission STOP / snapshot]
  G --> H[trace・state・journal close]
  H --> I{全条件と保存証拠が完全}
  I -->|yes| J[global receipt close → marker close]
  I -->|no| K[失敗・不完全の final evidence]
  J --> K
  K --> L[final slot → delivery slot → caller observation]
  L --> M[Producer-free offline join]
```

runtime SOURCE と科学参照/input/runtime bytesを起動時・phase境界・worker setupで照合する。call entryのpermitは親RPCが唯一のgate。consumerはbridge/Producer/scheduler/Admissionをimportせず、旧global/F01/cost primitivesと新trace判定を使う。
''')
    md('v2_technical_contract_candidate.md','''# V2統合技術契約候補

正式項目名は `b3_storage_v2_global_bundle_candidate_v1`。storage protocol は `GLOBAL_MONOTONE_V2_POST_CLOSE_JOIN_V1`、manifest/result/receipt/diagnosticは別schema。candidateのcontent_adopted/run_ready/contract_grants_executionはfalse、science_release/scientific_adoptionはnull、mandatory_STOPはtrue。技術的PASSから正式科学採択へ移転しない。

予約はcounter・cursor・current ticket・RAM stateを先に整合させ、後でfallible state/trace I/Oへ進む。保存失敗が予約を消したりcall creditを返却したりしない。RESERVEとENTERを区別し、ENTERのtrace/state保存後だけworkerへpermitを返す。未証明COMPLETEはUNKNOWN、STOP後の新規ENTERは禁止する。

RAM shadowはmonotone intentでありdurability証明ではない。physical state、journal、ordered trace、trusted prefix、未保存RAM、partial tail、flush/close不確実性、byte unknownを別々に記録する。十分な物理witnessがなければINCOMPLETE_EVIDENCE。flushをfsyncまたはpower-failure durabilityと呼ばない。

成功joinは全48pairと必要trial、各F01、A/B/pass/S/P1 prefix、global counters、summary/receipt、trusted trace、fake STOP/spent、全storage close、post-close receipt/marker、最終slot/delivery/caller観測、global call/byte capを要求する。診断schemaはcomplete=false/final_acceptance=false。一条件だけのreceipt、marker単独、RAM申告単独はbundle成功ではない。

固定サイズfinal 4MiB、delivery 4096B、caller observation 4096Bのcreditを最初にglobal resultへ予約する。final report内のその3つのACKは明示的なprojectionであり、物理slotと外側の保存後観測が揃うまで成功根拠へ使わない。delivery自身のclose失敗はcallerが報告し、consumerは不完全とする。caller観測自身のclose/power-failureを自己証明する契約ではない。
''')
    md('source_and_contract_delta.md',f'''# 固定SOURCEとの差分

|役割|固定点／新版|
|---|---|
|直接基点Producer|{P}|
|Producer人工測定SOURCE|45bf46b4a833e71f138903fb511f2741e54043b7|
|主要固定受入|{I}|
|旧独立core/test|49a0dcc9c8ec6dfc96eba5bb862cfdcb3d0fc866|
|今回の人工runtime測定SOURCE|{MEASURED}|

101公開ファイルのbytesをGit blobへ照合し、測定45bfは公開履歴bundleからfresh bare Gitへ復元した。bundle 49,475B、前提76ca3c40、15測定SOURCEの全bytes一致。公開HEAD後追加helper2件とは区別する。

新規6moduleはbundle contract/bridge/fixture/store/trace/offline v1。旧Producer・独立checker・G7/F3・R1/F01/Admission/Hostの既存bytesを編集していない。新schemaとrunner/testsを追加し、branch間の無条件merge/rebaseは0。

実質技術差分は、単一global台帳への二条件結合、modeを含むglobal trial ID、6phase barrier、dispatch中断PHASE_ABORT、post-close global receipt、receipt close後のmarker、callerによるdelivery close観測、最後の証拠slotの事前global creditである。旧V2のmarkerがcloseより前に残り得る履歴は修正しない。新順序と外部観測joinの正式採択はユーザー/GPT判断。

新人工endpointは人工専用surrogateであり、law/weight/evidenceの人工形状は旧fixtureとの科学的同値を主張しない。人工SHA256 counter namespace・160bit消費・S/P1/A/B/pass prefix対応を維持し、科学law/event/precision/angle参照は固定bytesとして保持した。実科学APIの同値実行証明は今回取得していない。

測定後のtest修正は、出力前Admission拒否の期待分類とmutation証拠overlay保存だけ。新tests/evidence helperの現行SHAと `at_measured_SOURCE` をinventoryで区別する。runtime six modulesとrun_case/source/schema pinsは測定後変更していない。
''')
    md('legacy_test_maintenance.md',f'''# 旧co_qualname testのversioned保守

原版は49a0dcc9のtest bytesを専用legacy_runtimeへ復元し、そのまま実行した。結果は **622 PASS／1 FAIL**。対象は `CandidateAcceptance.test_cached_observer_call_count_label_exclusive_partition`。実labelは `...<locals>.leaf`、旧期待値は `endswith(':leaf')`。

`versioned_legacy_test_v1.py` は原fileを変更せず別moduleとしてcompileする。文字列期待値1箇所だけを `endswith('.<locals>.leaf')` へ置換し、元filenameをobserver filterに使う。Scoped/Cached全測定行の等値比較、call数、exclusive partitionとresidualのassertionは原文のまま。補正版は **623 PASS**。

原版 SHA256 `{delta['original_sha256']}`、補正後compile対象SHA256 `{delta['corrected_compiled_source_sha256']}`。[機械差分](legacy_test_delta.json)、[1行diff](legacy_test_minimal.diff)、[全旧回帰](legacy_regression_results.json)、regression_logsに失敗traceと補正後記録を保存した。原版を623 PASSへ記録し直していない。CPU測定定義や科学契約は変更0。

最初の限定復元では136依存不足ERRORがあった。固定artifact/fixtureのGit bytesを補完した後の最終値が上記。初期記録はdevelopment_diagnosticsに保持し、最終件数へ加算しない。
''')
    md('global_bundle_identity_and_budget.md',f'''# Global identity・call/byte予算

candidate digest、source release candidate（全source pinのdigest）、scientific input、global scope/claim、run、condition、task、arm/layer/axis/phase/pass/local trial、global trial ID、global offset、seed namespace、receipt/STOPを区別する。global trial IDはrun/mode/local trial。modeを含むtask keyとglobal offsetもS/P1の衝突を防ぐ。trace.run_idはglobal run ID、manifest_bindingは別の契約digest。

n=256、16 cells=4方式×2layer×2axis。各条件correctness16×16=256、baseline16×256=4096、instrumented16×256=4096、計8448。二条件16896。candidate、独立導出plan、scheduler、static manifest、global receipt、consumerが同じ計算に到達し、人工full形状でreserved=started=completed=16896を確認した。

Global resultは64GiB=68,719,476,736B、logは256MiB=268,435,456B。一つのBudget objectを使い、S/P1へ別々のcapを付与しない。statefile retainedはplanned1B/slot、RAM shadowは別のRAMでありlog二重課金しない。累積pwrite trafficとretained bytesは別。

n256人工実測：result **{budget['result']['physical']:,}B**、log **{budget['log']['physical']:,}B**。双方reserved=observed=physical、unknown0。trace自己課金、journal、state、全pair、条件receipt、summary、diagnostic、snapshot、post-close receipt/marker、final/delivery/caller slotを含む。条件別byte内訳はtraceのconditionとshared最終化を別表にし、globalへ一度だけ足す。

FAKE_ONLY.sqlite3とcandidate_input等は試験harnessのauthority/input snapshotで、runtime result/logとは別の保存領域。公開rawには含める。実authority storageの会計/耐久性は正式契約未実装。早期crash/初期化失敗でglobal保存証拠が欠ける場合は完全なbudget監査へ格上げしない。
''')
    md('seed_prefix_and_condition_binding.md','''# Seed-prefix・canonical/F01

固定科学参照はP5_general_order、p=(1/5,3/10,1/2)、x=5/7、m5、sigma+1。ordinary、partial_return_tail、closed_P3_tail、full_return。A=G7、B=既存F3 finite-bit/angle監査、X/Y別。A6はmath-only tau1e-6/K_phi32/J32。root_bits256、probability_bits160、eta/rho等の元precision policy、元SOURCE/input hashesをfixed_scientific_reference.jsonに保持する。

科学seed proposalはPROPOSED_B3_LIMITED_G7_PILOT_20261019_V1、namespace/arm/axisのSHA256 counter。mode/layer/phase/passをbitstream domainへ追加しない。modeはtrial identityにだけ加える。科学bit列は生成していない。今回のbitsはARTIFICIAL_R1_PARALLEL/arm/axisだけを生成し、256bit counterのdiscard規則で1call160bitsを供給する。IID主張false。

両条件ともfused codec/CachedObserver/同人工endpoint・precision参照・保存/trace粒度。Consumerはpair/F01/typed failureの固定coreで個々のrawを検査し、正常bundleではA/B、correctness prefix、baseline/instrumented、S/P1全対応trialのevent traceを照合する。新fused canonicalは元exact canonicalと照合した。

新fixtureの有理数/event/angleは人工surrogateであり科学lawや真のangle enclosureではない。保持を示したのは固定科学参照のbytesとbridgeの比較結合・人工prefix/F01で、実G7/F3での意味論やpath coverageを今回実行証明したとはしない。coverage不足を埋める科学seed/law変更や追加callsは0。
''')
    md('admission_and_one_shot.md','''# Admission・one-shot

一つのprivate FAKE_ONLY.sqlite3を新設し、一つのglobal scope/claimをBEGIN IMMEDIATE＋unique constraintで原子的にSPENTへcommitしてからACKする。S/P1の切替でclaimし直さない。32競合attemptではwinner1・拒否31を一つのtestで確認した。snapshotはread-only/immutable URIで照合する。

commit ACK不明でもspentを返却しない。bridgeでのACK不明はoutput作成・worker entry前に停止し、fake STOPが可能ならSTOPPEDを残す。同scope再使用、二重claim、binding mismatch、retry1、resume true、redirect/symlinkされたstore pathを拒否した。STOP書込不能でもSPENTは消さない。

同一fake scopeでP1だけ再開するAPIはない。S失敗でP1は未開始、P1失敗でS receipt/rawは保持する。claim/STOPの保存先と失敗をresultで区別する。fake atomic履歴は人工testとSQLite記録の証拠であり、実authority・元private approval・power-failure durabilityの証明ではない。

production APIは無条件例外。人工APIは組込みFakeStoreと組込み人工fixtureだけを受け、外部endpoint/bits/authorityを渡す引数を持たない。正式grant発行・消費や実storeの使用は0。
''')
    md('stop_unknown_and_storage_semantics.md','''# STOP・UNKNOWN・保存の境界

親arrival順にfailureを保持し、first failureでSTOPをlatchする。新RESERVE/ENTERを拒否し、STOP前permit済みin-flightはcooperative drainする。CANCELはentry前、FAILはentry済みの証明可能な失敗、失われたworker/未証明COMPLETEはUNKNOWN。reserved call/未ACK byteは返却しない。current ticketとactive workerをdrain後に最終化する。

S/P1のRESERVE/ENTER/COMPLETE保存故障、RAMとphysical stateの不一致、trace partial/flush、WRITE_ACK欠落、summary/diagnostic/receipt/marker保存失敗、trace/state/journal/final/delivery close失敗、STOP trace/journal/fake-store書込失敗を分けて人工注入した。trace失敗後はtrusted prefixへappendしない。tailに完全なJSONが見えてもprefix外のoperationを確定履歴へ使用しない。

dispatch中断はPHASE_ABORTを記録する。WORKER_START保存前に生成済みだったworkerは、存在と最終reportだけで完全なstart履歴へ補完せず、証拠不完全に残す。parent crashではfinal report/close/RAM/CPUが欠け、snapshotだけからUNKNOWNの実entry時刻を推定しない。

旧V2 marker-before-closeの履歴は保持する。新candidateのglobal receiptはstorage close後、markerはreceipt close後、final slot→delivery slot→caller observationの順。一つのmarkerや条件receiptだけでは成功にしない。delivery closeの失敗はそのdelivery自身で自己証明できないため、呼出元の観測もjoinする。

26故障ケースのINCOMPLETE_EVIDENCEは、障害が期待どおり拒否されたというtest PASSと両立する。保存の完全性そのものがPASSになったという意味ではない。
''')
    md('cpu_wall_and_output_accounting.md',f'''# CPU・wall・output会計

Total CPUはhost parent＋全worker、global wallはcaller outer（bridge前からdelivery close後まで）。pair CPU、IPC、observer、coarse/fineをtotalへ再加算せず、pair wallの和をglobal wallにしない。最後のcaller観測文書自身の保存とoffline consumerは後処理windowであり、保存byte creditは初めからglobal resultへ含める。

条件別parent/worker CPUとwallはS/P1の非重複window。共有setup、claim、receipt/diagnostic/STOP/finalization等の未配賦分はsharedに残す。条件別byteはparent traceのconditionとshared creditから導出する。S/P1のcapを二つ足す方式ではない。

pair記録はconstructor/source checks、generator+audit、angle/requery、evidence、serialization/hash/write/flush/finalize、coarse/fine代替partition、RPC、RSS/HWM/IO、retained bytesを保持する。親保存の複合診断は保存したが、親trace/state/全個別I/Oのexclusive費用やisolated observer overheadはnull。RSSのshared-page sumをunique厳密peakと呼ばない。失敗prefix・drain・unknown費用は正常比較へ混ぜない。

今回n256人工観測はparent {normal['cost']['parent_cpu_ns']/1e9:.6f}s、全worker {normal['cost']['all_worker_cpu_ns']/1e9:.6f}s、total {normal['cost']['total_cpu_ns']/1e9:.6f}s、global wall {normal['cost']['global_wall_ns']/1e9:.6f}s。同一共有サーバーの人工形状の記述値で、実G7のCPU/保存量予測やspeedup主張ではない。

S→P1の順序、cold/warm cache、負荷、runtimeの交絡が残る。両者fusedなのでfused単独改善率は識別不能。固定portfolio運用費用の候補であり、machine-independent algorithm costやCPU母平均CIではない。量子signal confidenceをCPU CIへ転用しない。

旧人工trace41,899,756Bは特定形状の参考値に留める。今回のglobal trace/outputにも実科学上界の証明はない。Full-N31,710,848 callsのper-trial保存を実行可能と仮定しない。必要ならcheckpoint/transition集約と保持規則を別契約で設計する。
''')
    md('production_gate_and_authority_boundaries.md','''# Production gate・権限境界

production_entry/ProductionFactory/ProductionBitsはすべて無条件例外を投げる。run_ready/adopted/authority引数を渡しても解放されない。科学factory・real bits・real authorityへdispatchする実装やunseal flagは存在しない。人工APIは組込みfixture/storeをsourceとruntimeにbindし、science module import blockerをworkerへ設置する。parentでも禁止moduleの既loadを拒否する。

今回の正式science SOURCE採択、real G7/F3 import/sampling、新科学計算、実authorization/grant/claim発行・消費、native quantum build/compile/synthesisはすべて0。Full-Nと限定pilotの科学実行も0。新しいfake scope/claimだけを人工試験へ使用し、旧one-shot/STOPを再使用・解除していない。

将来の正式productionには別の採択済SOURCEとcontract、実API binding、authority bootstrap・grant/monitoring/保存範囲の明示認可が必要。このcandidateのbooleanを反転して実行へ移る手順は提供しない。
''')
    md('artificial_acceptance_results.md','''# 最終人工受入と開発履歴

最終runtime SOURCE e099dcc461247e81eff981867dc28c3f3b16df5e、43fresh fake scopes。bundle正常2（n16/n256）、単独条件正常2、期待された技術失敗13、期待された証拠不完全26。未解決の予期しないregression failureは0。INCOMPLETE_EVIDENCEを正常保存へ再分類していない。

新規107 PASSには43matrix assertion、35mutation拒否とそのnormal control、global算術/seed/canonical、atomic race/spent/ACK、gate、費用/byte、SOURCE/runtime mutation等のtestsを含む。35mutationは32 integrity mismatch／3 incomplete、raw差分overlayを保存した。旧指定10suiteは原版622/1、補正版623/0。既存V2 storage41と旧独立57もPASSし、unique IDは828。

開発初期にdelivery自身のclose失敗がconsumerから漏れる欠陥とdispatch abort表現の不足を検出した。callerのpost-delivery観測、PHASE_ABORTと欠落start証拠の不完全扱いを追加した。修正前2 rawをarchiveのdevelopment_before_repairへ保存し、全初期matrix/失敗testsをdevelopment_diagnosticsへ残した。依存不足の旧回帰と、新testの一条件分だけだった期待値・欠落証拠分類も修正した。初期結果を最終件数へ加算しない。

offline consumerは別processでbridge/Producer/scheduler/Admission/scienceをimportしていない。作者自身による検査であり、別人・別AI・別物理環境の独立レビューではない。原証拠・個別結果・ログはcase_verifications、negative_mutation_results、regression_logs、raw_partsを参照する。
''')
    md('pending_and_required_decisions.md','''# 残件・正式採択前の判断

**ユーザーとGPT**：V2のRAM/physical/trusted prefix/UNKNOWN/非返却/STOPと、新global一契約・post-close receipt/marker・caller観測joinを正式技術契約へ採るか判断する。n256/二条件16896、64GiB/256MiB、seed namespace、固定S→P1、CPU比較estimand、観測精度・coverage条件を正式に決める。今回の人工PASSから採択しない。

科学実行を選ぶ場合は、別のproduction SOURCE/contractと実endpoint/bits/authority/監視/保存先のbinding、その独立した技術確認と明示実行認可が必要。今回のgateは閉鎖されたまま。実G7/F3のevent・有理数bit長・requery・path coverage・精度・bit消費・実CPU/保存総量は未測定。現surrogateをその証明へ転記しない。親個別I/Oとobserver単独費用、power-failure耐久性、実authorityのatomic/ACK/STOP保証も今回の証明範囲外。

歴史108の証拠不完全66、旧47/18等pending、旧科学結果とspent/STOPは不変。今回26故障ケースも証拠不完全を保持する。Full-Nの保存集約・checkpoint設計が必要なら別契約候補で検討し、現在のper-trial traceをそのまま適用しない。

**Codex**：このpacketのcommit/publication/readbackまで実施しmandatory STOP。追加科学計算、反復、順序反転、実claim、新stageは未実行で、自動進行しない。
''')
    md('handoff.md',f'''# Q1–Q10と引継ぎ

**人工技術候補は完成。新規107 PASS、補正版既存を含むunique828 PASS。科学入口は閉鎖、正式採択せずmandatory STOP。**

1. **Q1 固定照合**：Producer {P}、独立受入 {I}を固定参照し101公開ファイルをGit bytesへ照合。45bf測定SOURCEは公開bundleをfresh bare Gitへ復元し15測定SOURCE一致、後追加2helperと区別。旧108=正常2/技術失敗40/不完全66を維持。
2. **Q2 技術契約**：`b3_storage_v2_global_bundle_candidate_v1`／`GLOBAL_MONOTONE_V2_POST_CLOSE_JOIN_V1`へRAM/physical、trusted prefix、UNKNOWN非返却、cooperative STOP、全証拠joinを統合。新global receiptはstorage close後、markerはreceipt close後、final/delivery/caller観測も必要。旧V2履歴は変更0。
3. **Q3 legacy保守**：原版SHA・test・失敗traceを保全。co_qualname suffix期待値1個だけの別versionをcompile。原版622 PASS/1 FAIL、補正版623 PASS。全observer rowsとexclusive partition比較を維持。
4. **Q4 global結合**：単一fake claim、coordinator、RAM/statefile、call counter、trace、result/log budget、STOPでS→P1を結合。二つの独立実行ではない。S失敗でP1未開始、P1失敗でS証拠保持、spent/retry/resume拒否を人工確認。
5. **Q5 算術**：16cell・3phase・二条件、各256+4096+4096=8448、global16896をcontract/plan/manifest/receipt/consumerで一致。人工n256のreserved=started=completed=16896。
6. **Q6 prefix/F01**：固定科学SOURCE/input/precision/seed参照bytesを維持。modeはtrial IDへ追加しseed domainへ追加しない。人工S/P1/A/B/pass/correctness prefixとcanonical/F01を確認。実科学law/bit/angleの実行同値証明は未取得。
7. **Q7 障害**：atomic32競合、commit ACK不明/spent、STOP、in-flight drain、worker/parent crash、UNKNOWN、予約/entry/complete/state/partial trace/ACK、summary/diagnostic/receipt/STOP/close、marker偽成功、caller post-close観測を人工検証。
8. **Q8 件数**：最終43人工=global正常2/条件正常2/技術失敗13/証拠不完全26。新規107 PASSに43matrixと35mutationを含む。旧補正623＋既存storage41＋既存独立57＋新107=unique828 PASS。原版622/1、初期開発・歴史108/680/820は加算しない。予期しない最終test FAIL/ERROR/SKIP=0、未解決新規整合不一致0（実施範囲内）。
9. **Q9 残件**：[正式採択前の判断](pending_and_required_decisions.md)。V2/global/順序・費用・予算候補の正式採択、別production SOURCE・実API/authority/監視bindingと追加技術確認、科学実行の別明示認可が必要。実G7費用/coverage/保存量、独立環境再現、実store耐久性は未取得。
10. **Q10 禁止処理0**：実G7/F3 sampling/import、新科学計算、科学SOURCE採択、実authorization/grant/claim発行消費、native quantum build/compile/synthesisはすべて0。旧STOP解除/旧claim再使用0。サブエージェント0。作者とchecker実装者は同じprimary Codex。

今回の公開測定SOURCE：[e099dcc4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/{MEASURED})。最終contentとremote HEAD/各公開SHA256は[publication_receipt.json](publication_receipt.json)で照合する。runtime/source inventory・raw復元・保全記録を同梱した。

**次の担当：ユーザーとGPT**。本候補のSOURCE/技術契約と未採択条件をレビューし、正式採択の可否と次に必要な範囲を判断する。**Codexは公開照合後mandatory STOP**し、pilot/Full-N/科学実行へ進まない。
''')
    # Discovery indexes: append a narrow technical entry, preserving history.
    links={
      'PROJECT_MAP.md':f'[S/P1 global bundle bridge候補・人工受入]({K.DOC}README.md)',
      'scripts/README.md':f'[再現runner](../{K.DOC}README.md)',
      'src/trotterlib/README.md':f'[Track B専用bundle SOURCEと検証](../../{K.DOC}README.md)',
      'docs/README.md':f'[Evidence Packet](research/{Path(K.DOC).name}/README.md)',
      'docs/research/研究概要・現状.md':f'[技術候補とQ1–Q10]({Path(K.DOC).name}/README.md)'}
    for path,link in links.items():
        p=ROOT/path
        if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(subprocess.check_output(['git','show',P+':'+path],cwd=ROOT))
        text=p.read_text();entry=f'\n\n## 2026-10-19：Storage V2 S/P1 global bundle bridge 技術候補\n\n{link}。単一global claim/call/byte/STOP、人工43ケースと新107 testsを検証。科学入口閉鎖、正式採択なし、実G7/F3 import/sampling 0、mandatory STOP。旧科学結果とSTOPを保持。\n'
        if '## 2026-10-19：Storage V2 S/P1 global bundle bridge 技術候補'not in text:p.write_text(text+entry)
    note=ROOT/'docs/research/研究ノート/2026-10-19_track_b_storage_v2_global_bundle_candidate.md';note.write_text('# Storage V2 global bundle技術候補\n\n利用者の人工実装・検証指示に従い、S/P1を一つの台帳へ結合する候補を作成。新107、補正版623、既存V2 41/57がPASS。43人工rawを保存し、正常4と期待した非成功39を区別した。delivery自己closeの観測漏れを修正し、初期記録を保持。科学条件や既存SOURCE/結果/claim/STOPは変更0。正式採択はユーザー/GPT、公開照合後mandatory STOP。\n\n[packet](../'+Path(K.DOC).name+'/README.md)\n')
    # Exclude generated local working copies; only the curated delivery is staged.
    ignore=HERE/'.gitignore';s=ignore.read_text()
    for line in ('legacy_development*/','tests_development*/','tests_sealed/','tests_evidence/','fixed_regressions/','fixed_regressions_final/'):
        if line not in s:s+=line+'\n'
    ignore.write_text(s)
    print('built packet: 43 final cases; 107 new / 828 unique regressions; 35 mutation overlays')
if __name__=='__main__':main()
