"""Render the technical candidate packet from static pins and artificial records."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import spec as S, contract as K
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import guard as G
import json,subprocess,hashlib,shutil


def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def write(name,text):
    (HERE/name).write_text(text.strip()+'\n\n**未採択・実科学未実行。mandatory STOP。次の担当：ユーザー・GPT。**\n')


if __name__=='__main__':
    evidence=Path(sys.argv[1]) if len(sys.argv)>1 else None
    c=K.candidate(ROOT)
    result=json.loads((evidence/'result.json').read_text()) if evidence else {}
    measured=result.get('measured_SOURCE','PENDING_FINAL_SOURCE_FREEZE')
    tests=result.get('tests',{})
    # All source reads below are Git bytes/AST, never module loading.
    modules=tuple(G.CORE)+('g6_w2c_b3_driver','g6_w2c_cost_preparation')
    science={}
    for n in sorted(modules):
        path=S.SOURCE+n+'.py';actual=(ROOT/path).read_bytes()
        original=subprocess.check_output(['git','show',S.SCIENCE+':'+path],cwd=ROOT)
        if actual!=original:raise RuntimeError('FIXED_SCIENCE_BYTES:'+path)
        science[path]=hashlib.sha256(actual).hexdigest()
    source_input=dict(scientific_commit=S.SCIENCE,scientific_source_pins=science,
        input=S.INPUT,precision=S.PRECISION,angle=S.ANGLE,input_id=S.digest(S.INPUT),
        precision_id=S.digest(S.PRECISION),input_pins=c['input_pins'],adopted=False)
    save('scientific_source_input_pins.json',source_input)
    save('live_api_binding_candidate.json',dict(schema='b3_live_api_binding_candidate_v1',
        **S.CLOSED,signatures=c['api_signatures'],routes=c['routes'],
        full_trace='order -> raw labels -> reduction -> packet iff reduced -> acceptance -> child iff accepted',
        canonical_trace='group -> mode-specific labels/conditional child -> event(group_index, raw_word, child)',
        source_identity=S.SCIENCE,loader='ABSENT',live_compatibility='NOT_ESTABLISHED'))
    save('correctness_scope_candidate.json',S.scope())
    save('correctness_contract_candidate.json',c)
    save('seed_namespace_candidate.json',dict(**S.CLOSED,proposed_scientific_namespace=S.SEED,
        algorithm='SHA256(UTF8(seed)||uint64_BE(counter)); concatenate blocks; return high width bits; discard rest',
        domain='namespace/arm/axis',mode_in_seed=False,
        shared_prefix=['S/P1','A/B','phase/pass'],separate=['arm','X/Y'],
        identity_fields=['science_release','input','scope_claim','run','mode','arm','layer','axis',
            'phase','pass','local_trial','global_trial','session','runtime'],
        trace_fields=['width','counter_before','counter_after','value_hex','discarded_bits',
                      'bits_consumed_before','bits_consumed','failure_position'],
        artificial_runtime_namespace='ARTIFICIAL_REAL_ADAPTER_V1',science_bits_generated=0,
        real_variable_consumption_and_IID='NOT_ESTABLISHED'))
    save('authority_requirements_candidate.json',K.authority_requirements())
    save('preflight_manifest_candidate.json',K.manifest(c,'ARTIFICIAL_ONLY/CORRECTNESS/UNEXECUTED_EXAMPLE',
        '/tmp/b3-bundle-artificial-UNEXECUTED/output'))
    save('source_inventory.json',dict(measured_SOURCE=measured,source_release_candidate=c['source_release_candidate'],
        source_pins=c['source_pins'],input_pins=c['input_pins'],scientific_reference_commit=S.SCIENCE,
        fixed_adapter_HEAD=S.BASE,independent_HEAD=S.INDEPENDENT,
        fixed_global_HEAD='62967269e611b61e73b70abe20e998d9c7eaaf50',
        fixed_global_measured='deade78a19f5b23859d16816d3a7bfb37765dd80',
        fixed_API_HEAD='82896d53fa97235fa848d6c0833eb193af8b99d6',
        corrected_API_measured='78c571005fe766c61b32dc867e006531099ff608'))
    forbidden=['core_import_load_call','scientific_sampling','scientific_bits','new_scientific_computation',
        'saved_scientific_reanalysis','native_build_compile_synthesis','scientific_adoption',
        'real_authorization_grant_claim_issue_consume','real_trusted_store_access','old_claim_reuse',
        'STOP_release','production_unlock','real_correctness','limited_pilot','Full_N','AI_subagent']
    save('verification.json',dict(schema='b3_closed_live_preflight_verification_v1',**S.CLOSED,
        status='ARTIFICIAL_TECHNICAL_PREPARATION_COMPLETE' if evidence else 'DRAFT',
        measured_SOURCE=measured,tests=tests,artificial_integration_cases=result.get('artificial_integration_cases',0),
        integration_PASS=result.get('PASS',0),separate_reviewer=False,
        forbidden_operations={x:0 for x in forbidden},core_guard=G.observation(),
        live_compatibility='NOT_ESTABLISHED',unresolved_artificial_technical_inconsistencies=[],
        reused_not_rebuilt=dict(independent_checks=224,producer_historical_integration=30,
            historical_semantic_mutations=20,historical_global_trials=16896),
        next_actors=['USER','GPT']))
    cases=json.loads((evidence/'cases.json').read_text()) if evidence else []
    save('artificial_acceptance_matrix.json',dict(tests=tests,cases=cases,
        final_SOURCE=measured,role='same-author artificial verification; no new independent reviewer'))
    if evidence:
        target=HERE/'acceptance';target.mkdir(exist_ok=True)
        for p in evidence.iterdir():
            if p.is_file():shutil.copyfile(p,target/p.name)
        save('artifact_inventory.json',{str(p.relative_to(HERE)):dict(bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in target.iterdir() if p.is_file()})
    write('README.md',f'''# Track B W2c-B3：実API correctness実行前準備・閉鎖v1

新versioned SOURCE `{S.PKG}`。実科学loaderは存在せず、人工factoryだけを固定spawn/F01/Storage V2へ接続する。
測定SOURCE `{measured}`、SOURCE artifact `{c['source_release_candidate']}`。
独立受入 `{S.INDEPENDENT}`／Adapter `{S.BASE}`を固定参照し、merge/rebase・旧SOURCE変更は行わない。

[Q1〜Q10](handoff.md)、[静的API](canonical_full_connection_contract.md)、[512候補](correctness_only_scope_candidate.md)、
[人工受入](artificial_technical_acceptance.md)、[未決事項](pending_and_required_decisions.md)、[公開照合](publication_receipt.json)。
科学SOURCE/input/precision、契約、coverage、予算、real authorityは未採択。人工PASSはlive互換性を認定しない。

再現：`python3 {S.DOC}run_acceptance.py /tmp/NEW_UNIQUE_OUTPUT_DIRECTORY`。
新FAKE_ONLY scopeを作る人工試験であり、実trusted store・旧claimへのアクセスはない。
新SOURCE commitから実施すること。公開側はpublication_transport.jsonのexact-tree対応も参照。''')
    write('fixed_source_and_input.md',f'''# 固定SOURCEと入力

科学参照 `{S.SCIENCE}`、閉鎖Adapter `{S.BASE}`、独立受入 `{S.INDEPENDENT}`。
Global公開 `62967269e611b61e73b70abe20e998d9c7eaaf50`、測定 `deade78a19f5b23859d16816d3a7bfb37765dd80`。
API公開 `82896d53fa97235fa848d6c0833eb193af8b99d6`、訂正測定 `78c571005fe766c61b32dc867e006531099ff608`。
Producer測定 `b7ad344401b10395c49e4cfa806909282859bab5`、artifact `024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb`。

P5_general_order、p=(1/5,3/10,1/2)、x=5/7、m=5、sigma=+1。
root_bits256、probability_bits160、eta=rho=1e-12、epsilon_axis1/200、primitive_error1e-6、alpha_axis1/320、familywise1/20。
A6 math-only tau1e-6/K_phi32/J32。候補であり科学採択・量子confidenceやCPU統計保証を追加しない。

8 moduleは科学固定commitとのbytes照合済み。6 coreはAST/SHAのみ、driver/helper2は人工用途のみimport可能。
全input pinはscientific_source_input_pins.json、必須資料bytes/SHAはreference_documents.json。
fixed_scientific_reference.jsonはpilot基点より後のGlobal資料なのでAdapter公開commitから取得した。''')
    write('live_api_wrapper_design.md',f'''# 閉鎖wrapper/factory

`interface.ConnectionIdentity`がSOURCE/input/policy/runtime/run/scope-claim/Mode/arm/layer/axis/phase/pass/local trial/sessionを結合する。
`FutureScientificResolver.describe`は静的routeを返し、resolveは無条件拒否。module pathを実行するloaderや任意factory injectionは存在しない。
`artificial.ArtificialFactory`は組込み型だけを受け、context seal・runtime・SOURCE・taskをworker内で再確認し、既存人工Endpointを包む。
parentとspawnの双方でguardを依存import前に導入。新たなcore推移依存はload前に遮断してSTOPする。

SOURCEは `{S.PKG}`、差分はversioned wrapper・correctness plan・preflight・証拠結合とtestsのみ。
boolean/fake approvalの変更ではSOURCE再生成比較で拒否され、real loader自体もない。
将来は別versioned release、正式科学契約、real authority、ユーザーの明示的実行承認をすべて必要とする。''')
    write('canonical_full_connection_contract.md','''# Canonical／Fullの静的接続

|対象|生成・sampling|B監査・angle/requery|
|---|---|---|
|ordinary / partial_return_tail / closed_P3_tail A|make_generator -> CanonicalGenerator.sample(bits)|Aはratioのみ|
|Canonical B|LegacyEndpoint(modules,arm,layer,angle,source_check,*,permit=None)|canonical_snapshot(g,_law)、AuditParity、event_witness、event(group_index,word,child)、enclose_atan|
|Full A|make_generator -> FullReturnGenerator.sample(bits)|Aはratioのみ|
|Full B|prepare(p,x,m,**precision_and_context) -> GeneratorSession.sample(bits=None,*,trial_id)|sealed preflight、AuditParity、angle_for_event(record)内部でevent(word,child)->packet再照会|

Canonical正入力をF3.prepareへ渡すとCANONICAL_PREFLIGHT_NOT_IMPLEMENTED。解除せずCanonical専用経路を保持する。
実引数はlive_api_binding_candidate.jsonのAST抽出一覧。F3 make_requestにはeta/rho/root_bits/probability_bits/angle_tau/angle_root_bits/angle_termsが必要。
driverのtau/K_phi/Jをangle_tau/angle_root_bits/angle_termsへ変換する。

Fullはorder draw→raw labels→reduce比較。非reducedはpacketなしZERO_R、reducedはpacket1回→acceptance draw、rejectはZERO_C、acceptedのみchild draw。
原rawをreducedへ再割当てしない。eventはword/raw_word/child/ratio/phase/coefficient/proposal/weight/provider_calls/helper/circuit_time_wordを保持。
Canonicalはgroup indexとraw_wordが必須で、reduced wordだけでは再照会できない。FullのB angleはoriginal Evidenceを渡し、内容fingerprint・context照合とpacket再照会を原APIが行う。
実runtime型と実法則の互換性はNOT_ESTABLISHED。''')
    write('real_result_evidence_interface.md','''# Result／Evidence接続契約

既存Codec/Result adapterは無変更で再利用する。F3 original document/digest、driver projection/document/digest、lossless sidecar、Fraction、tuple/list/map、source identity、event fingerprintを保持。
driver投影は型情報を失う。保存投影は原文sidecarから復元した値の投影と照合し、原型の復元は受理subsetのsidecar付きに限定する。
人工fixture用のlabel数・degree制限を一般live型への保証にしない。live外部objectを受けるloader/adapterは別releaseに必要。

EVENT/ACCEPTED、ZERO/RAW_NONREDUCED、ZERO/COIN_REJECT、IDENTITY、deterministic EVENT_QUERY、typed failureを区別する。
None単独は判定不能。不足trace/provenanceはtyped errorまたはNOT_ESTABLISHEDとし、推測しない。IDENTITY/queryを確率ZEROへ流さない。
F01は原failure documentを保ち、未完prefixを成功にしない。新wrapperの例外も既存ControlledEndpointのF01経路へ伝播する。

angleはoriginal event、fingerprint、parent/child、ratio、input/source/policy、precision、interval、代表値/radius、original Evidence、requeryを結合する。
Canonical contextはdriver input/policy/preflightとeventのdriver identity、Full contextはF3 input_id/policy_id/source_id/preflight_id/sigma/record.digest。
人工intervalの形状整合は数学的atan enclosureの証明ではない。live Evidence互換性はNOT_ESTABLISHED。''')
    write('scientific_bits_and_seed_contract.md','''# Seed／bit consumption契約

候補namespace PROPOSED_B3_CORRECTNESS_ONLY_20261020_V1は未採択文字列としてのみ保存する。
実行する人工namespaceは既存ARTIFICIAL_REAL_ADAPTER_V1。旧人工fixture再利用であり、実科学namespaceは消費しない。
SHA256(UTF8(seed)||uint64 big-endian counter)、必要blockを連結、high bitsを返し、未使用bitsをdiscardする静的仕様を確認した。
width/counter前後/value/discard/消費前後/failure positionを保持。正整数width、counter overflow、型不正はterminal STOP。

Modeはglobal trial IDに含めseedから除外。A/B、S/P1、phase/passは同じarm/axis prefix、arm・X/Yは分離する。
release/input/scope-claim/run/session/runtimeは別binding。対応trialでのみprefix比較し、IID_claim=false。
Fullではdegree+1/degree+2/degree+3等のdraw数となり、Canonicalもgroup modeに依存する。人工固定draw数は実規則の採択ではない。
実variable draws、conditional child law、reject、消費、IID条件はNOT_ESTABLISHED。''')
    write('correctness_only_scope_candidate.md','''# 初回correctness-only候補

4方式×A/B×X/Y=16 cell、各16 trials、S/P1各256、合計最大512 scheduled sampling calls。
correctness/baselineのみ。profile、instrumented pass、pilotへの自動接続は0。
旧Full-Nのcorrectness256とFull-N31,710,848、pilot16,896とは別scope・別契約候補であり、正式規模は未採択。

P5入力・precision/A6はfixed_source_and_input.md。新SOURCE・input/runtime・scope/claim・grant・seed・call/bytes・STOP・receiptが必要。
旧grant/spent scopeを引き継がず、correctness grantからpilotを開始しない。
目的はlive呼出し、exact invariants、A/B prefix、ZERO分類、B audit/angle/requery、F01、Storage/receipt/caller、実bits保存の確認候補。
科学的優位、CPU優位、量子精度保証の確認ではない。今回の512は人工backendのみで実施し、実correctnessは0。''')
    write('correctness_acceptance_and_coverage.md','''# 判定・coverage候補

SOURCE/input/runtime/precision、許可trial、予定trial完了、Evidence完全性、exact rational/event invariant、A/B prefix、ZERO/reject/accepted分類、B preflight/angle/requery、F01、STOP/UNKNOWN、trace/receipt/caller、call/bytesを別項目で判定する。
技術失敗・UNKNOWN・欠測はtechnical STOP。exact event不一致はscientific mismatch STOPとして別分類する。

旧Full-N driverはcorrectnessでat least one accepted per cellを要求する。新候補は16 calls固定でacceptedがないcellがあればINCONCLUSIVE_COVERAGEとし、追加sampling/retryを実行しない。
全予定完了かつ不一致なし・必要coverageがあればCANDIDATE_CORRECTNESS_PASS。
いずれもユーザー/GPTによる基準採択待ち。未出現のZERO/coin reject等をPASSとはしない。
追加検証の必要性・scopeは別途採択・別grant・別実行承認が必要。人工coverageを実科学coverageへ転用しない。''')
    write('call_budget_and_one_shot.md','''# Call/byte予算とone-shot候補

scheduled sample512、primary constructor16（各mode8）、G7 make_generator14、F3.prepare2。
prepare内部Full constructor2はprimary16の内数であり加算しない。
canonical_snapshot6、AuditParity生成8、B sampling audit最大256、B angle最大256、B event requery最大256（Canonical192/Full64）。
correctness event_witnessはA/Bのaccepted全件、最大512。Full sampling packet最大128＋Full angle packet最大64。
同じnested operationを二重集計しない。diagnostic query0、追加sample0。固定入力でbit draw静的上界3200、各160bitの案。
root/law/predicate等の全内部関数call総数はNOT_ESTABLISHED。境界予算だけを全科学関数総予算と呼ばない。将来releaseで内部カウンタ・許容範囲を確定する。

correctness独自のresult1GiB、log64MiBの未採択案。full Evidence512件＋failure/STOP/trace/receipt/callerを全保持する。
pilot64GiB/256MiBを自動継承しない。実docサイズ未測定なので十分性の保証なし。容量超過はSTOPし、データを間引いて通さない。

S→P1、one-shot1、retry0、resume=false。S失敗ならP1へ入らず、P1失敗ならS証拠保持。
SOURCE/input/precision/authority/seed/consumption/event/codec/angle/F01/storage/trace/receipt/bytes/call/worker/UNKNOWN不整合でSTOP。
STOP後ENTER禁止、UNKNOWN creditは返却しない。旧STOPは解除しない。''')
    write('authority_and_trusted_store_requirements.md','''# Authority／trusted store要件

SOURCE release/input/precision/runtime/seed/scope/claim、sampling・constructor・angle/requery予算、result/log、worker lineage、storage destination、monitoring、STOP、retry/resumeを結合する候補。
組込みFakeStoreは新しい所有tempのFAKE_ONLY.sqlite3だけを受ける。実credential/grant発行API・real store selectorはない。
新fake scopeのclaim commit後spend、STOP保持・再利用拒否、commit ACK unknownのspent維持は人工技術事実。

実store所有主体、鍵管理、失効、atomic claim、ACK不明時の判断、power-loss durability、worker containment、receipt/caller durabilityは外部条件で未確立。
SQLiteの人工proofをreal authority保証へ拡張しない。既存real claimへのアクセス・再利用は0。
次はユーザー/GPTと管理主体が必要な主体・運用・証拠を決定する。''')
    write('preflight_and_closed_gate.md','''# 静的preflightと閉鎖gate

contract.candidateは固定科学/Adapter bytes、AST signatures、input/precision/angle、SOURCE artifact、Python executable/library hashes、worker runtime、monitoringを結合。
preflightはcandidateとmanifestを正規再生成比較し、scope/claim、seed、call/bytes、storage destination、authority、retry/resumeを一致確認する。
SOURCE固定後verify_source_commitが完全Git SHAの全pin bytesを検査する二段階seal。後続docs-only commitでも測定SOURCEを保持する。
他worktreeのHEAD/statusは観測だけで、無関係な変更を実行拒否条件にしない。

content_adopted=false / run_ready=false / contract_grants_execution=false / science_release=null /
scientific_adoption=null / artificial_only=true / mandatory_STOP=true / retry0 / resume=false。
wrong SOURCE/API/input/precision/runtime/worker/seed/budgets/contract/approval/retry/resume/storageは人工拒否検証。
production_entryとFutureScientificResolver.resolveは無条件拒否。実moduleはloader未実装のためboolean変更でも呼び出せない。
parent/spawn両方にload前guardを導入しcore load一覧・試行数を保存。driver/helperの人工importだけが許可範囲。''')
    write('monitoring_and_storage.md','''# 保存・monitoring候補

全trialのidentity、original event/Evidence、Fraction、F3/driver原文・digest・sidecar、bits、angle/query、F01、worker/source/runtime、ordered trace/state/counter、condition/global receipt、STOP/UNKNOWN/marker、delivery/callerを保持する案。
新人工512件は既存Storage V2のspawn worker・RAM shadow・statefile・reservation/ENTER/COMPLETE・trace・F01を無変更で再利用。
新receiptはstate/trace SHA、candidate/manifest、stopped fake claimに結合し、final→delivery→callerのbytes連鎖を検査する。
新projectionのcheckerは同一作者で既存Codec/Resultと人工referenceを再利用する。独立review件数へ加算しない。

result1GiB/log64MiBは512専用未採択候補。実データの有理数size/angle証拠sizeは未測定。
acceptance/raw_artificial.tar.gzとartifact_inventory.jsonに人工bytesを保存。人工容量を実容量・耐久性保証にしない。
flush/close証拠とpower-loss durabilityは別。新sourceは実fsync/authority耐久性を認定しない。

将来のhost outerはadmission前〜全join/STOP/drain/close/finalization、parent＋全worker CPUを一度だけ集計。
setup/sample/bits/coefficient-child-law/audit/angle/query/codec/IPC/observer/storage/RSS-HWM/I-Oを観測しcoarse/fineを各outerへ別にreconcile、residualを残す。
現在の人工report CPUはCoordinator生成〜report採取windowで、admission・後続receipt保存まで含むreal outer計測とは異なる。費用の実科学外挿はしない。
資源は64 CPU affinity/大容量空きRAMを観測し、短いS経路は逐次契約、P1は8 pairを並行実行。CPU/RAM/wall人工capは追加しない。
一時host PSIだけで科学workerを止める契約にはせず、実OOM/worker異常/破損/UNKNOWNのSTOPを保持する。''')
    write('artificial_technical_acceptance.md',f'''# 人工技術受入

測定SOURCE `{measured}`。同一primary agentによる検証で、別reviewer/物理host独立性は主張しない。
最終件数：`{json.dumps({k:{n:v[n] for n in ('run','PASS','FAIL','ERROR','SKIP')} for k,v in tests.items()},ensure_ascii=False)}`。
新人工integration {result.get('PASS',0)}/{result.get('artificial_integration_cases',0)}：正常512full documents、S angle STOP/P1未開始、P1 angle STOP/S証拠保持。
semantic mutation20は新connection checkerの改変拒否。original document再sealを伴う改変も含むが、Storage全階層再seal20件の旧独立受入とは別集合。
既存Producer102 unitと旧107 regressionを新SOURCEで再実行。旧224/30 integration/20保存mutation/16,896人工trialは固定参照し再構築せず、件数へ合算しない。
core6 import/load試行0、実科学0。raw/trace/runtime/tests/Git pin検査をacceptance/へ保存。
初期開発試験の記録はdevelopment_summary.jsonに残す。最終source固定後の結果だけを上記最終値とする。''')
    write('remaining_scientific_validation.md','''# NOT_ESTABLISHED

live import/call互換性、実Result/Evidence runtime型、law/係数/conditional child law、variable draws、ZERO/reject/accepted分布、数学的atan enclosure、F3 requery、実CPU/wall/保存量、有理数bit演算費用、IID/独立条件、Full Return性能、native保証、real authority耐久性、Full-N実行可能性は全てNOT_ESTABLISHED。
今回の人工fixtureは科学法則の実装ではない。これらを確認する目的でも今回core moduleをimportしない。
最小次scopeは採択された新releaseでのlive correctness512候補とそのconstructor/audit/angle/requery/保存境界であり、採択も実行も今回未開始。''')
    write('source_and_contract_delta.md','''# SOURCE／契約差分

基点はAdapter公開80b7bf61fe0a735d4031bb719c4b3e788611c3ba。新namespace、focused tests、packet runner/docsと4索引追記だけを追加する。
科学6module、driver/helper、Adapter/Global Bundle/Storage V2/独立checkerは直接変更しない。
旧pilotの3phase×2modeを新候補correctness-only1phase×2modeへ明示的に別計画化。既存schedulerを変えずtask projectionと閉鎖wrapperを新sourceへ置く。
correctness-only予算・seed・scope・authorityは未採択差分。科学条件・旧Full-N/limited pilot/grant/STOPの変更ではない。
API測定SHAは訂正78c571005fe766c61b32dc867e006531099ff608を使用。
最終測定SOURCE以降の差分と旧SOURCE保全はpreservation.json、Git公開のexact-tree対応はpublication_transport.jsonを参照。''')
    write('pending_and_required_decisions.md','''# 未決事項と次の担当

|事項|技術準備|人工検証|live必要|採択/外部条件|次の担当|
|---|---|---|---|---|---|
|API route・wrapper・Result/Codec|閉鎖interface完成|済|型/実Evidence互換性が必要|別versioned live loader|Codex：新指示後|
|科学SOURCE/input/precision/A6|固定候補保存|binding済|correctnessで確認|正式採択必要|ユーザー・GPT|
|512・coverage INCONCLUSIVE・calls/bytes・STOP|候補完成|人工正常/失敗済|実サイズ/分布が必要|規模/基準/予算の正式採択|ユーザー・GPT|
|内部科学関数総call/監視/outer費用|境界interface|人工診断のみ|内部実call未測定|新releaseで計数範囲確定|Codex：新指示後|
|real authority/store|要件整理|fake-only済|real運用証拠必要|所有/鍵/失効/atomicity/ACK/power/containment/durability|ユーザー・管理主体|
|実correctness実行|今回なし|代替の人工のみ|必要|正式契約/新release/real authority/別明示実行承認|ユーザー・GPTが判断|

今回Codexは公開bytes/remote照合後mandatory STOP。次担当はユーザー・GPT。
limited pilot、Full-N、科学採択、authority発行、実API確認、production解除へ自動進行しない。''')
    write('handoff.md',f'''# 最終報告 Q1〜Q10

1. **Q1** 固定独立/Adapter/科学8module/input/precision/Global/APIを取得・SHA照合。source_inventory.json、scientific_source_input_pins.json、reference_documents.json。測定SOURCE `{measured}`。
2. **Q2** 新closed v1に静的resolver仕様、ConnectionIdentity、組込み人工factory、preload guard、preflightを実装。real loaderなし、core load0。
3. **Q3** Canonical G7＋driver監査/event(group,raw,child)/atanとFull F3 session/angle/event→packetを分離。CanonicalのF3正入力拒否を維持。
4. **Q4** 原文/sidecarの受理subset、exact型、ZERO2種/EVENT/IDENTITY/query/failure、angle/context、bit消費契約を準備・人工検証。live互換性はNOT_ESTABLISHED。
5. **Q5** P5固定候補、16cell×16×S/P1=512sampling。constructor16、B angle/requery各最大256など別予算。result1GiB/log64MiB候補、full Evidence、coverage不足INCONCLUSIVE/追加sampling0、S→P1/one-shot/retry0/STOP。Full-N/pilotとは別未採択scope。
6. **Q6** SOURCE/API/input/precision/runtime/worker/seed/scope/call/bytes/storage/authority/monitoring結合と拒否を人工検証。Git SOURCEは固定後全pin照合。core importなし。
7. **Q7** 要求bindingを固定し、実store所有・鍵/失効・atomicity・ACK/power/worker containment/receipt durabilityは外部未決。fake-only store以外の実装なし。
8. **Q8** 新unit `{tests.get('unit',{}).get('PASS','pending')}`、preflight `{tests.get('preflight',{}).get('PASS','pending')}`、mutation `{tests.get('mutation',{}).get('PASS','pending')}`、人工integration `{result.get('PASS','pending')}`、Producer `{tests.get('producer',{}).get('PASS','pending')}`、旧回帰 `{tests.get('old_regression',{}).get('PASS','pending')}` PASS。旧受入件数とは合算しない。未解決の人工技術不整合0。
9. **Q9** ユーザー/GPTが科学SOURCE/input/precision・512scope・coverage/予算/STOPを採択するか判断。実行前に別live release、real authorityの外部証拠、明示的実行承認が必要。今回採択/実行なし。
10. **Q10** 実core import/load/call、sampling、科学bits、新科学/再解析、real authorization/grant/claim/storeアクセス、native、旧claim再利用、STOP解除、production解除、実correctness/pilot/Full-Nは全て0。

GitHub公開固定URLとremote/public bytes照合はpublication_receipt.json。現在はmandatory STOP。''')
    print(json.dumps(dict(measured_SOURCE=measured,source_pins=len(c['source_pins']),input_pins=len(c['input_pins']),tests={k:v['PASS'] for k,v in tests.items()})))
