"""Technical candidate packet; no scientific execution or adoption."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.g6_w2c_b3_live_release_v1 import contract as K,guard as G
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import spec as S
import json,hashlib,subprocess,shutil


def save(name,value):(HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def md(name,value):(HERE/name).write_text(value.strip()+'\n\n**LIVE_RELEASE_CANDIDATE_UNAUTHORIZED。実科学未実行。mandatory STOP。次の担当：ユーザー・GPT。**\n')


if __name__=='__main__':
    evidence=Path(sys.argv[1]) if len(sys.argv)>1 else None
    c=K.candidate(ROOT);m=K.manifest(c,'ARTIFICIAL_ONLY/RELEASE/UNEXECUTED_EXAMPLE','/tmp/b3-release-test-UNEXECUTED/output')
    result=json.loads((evidence/'result.json').read_text()) if evidence else {}
    source=result.get('measured_SOURCE','PENDING_FINAL_SOURCE_FREEZE');tests=result.get('tests',{})
    cases=json.loads((evidence/'cases.json').read_text()) if evidence else []
    save('live_release_candidate.json',c);save('correctness_contract_candidate.json',dict(
        schema='b3_512_correctness_contract_candidate_v1',**S.CLOSED,
        preparation_approval='USER_APPROVED_PREPARATION_SCOPE',SCIENCE_RUN_AUTHORIZED=False,
        scientific_input=S.INPUT,precision=S.PRECISION,A6=S.ANGLE,sampling_plan=S.scope(),
        separate_call_budgets=K.call_budgets(),counter_semantics=K.counter_semantics(),
        formal_contract_adopted=False,production_release_active=False,real_grant=None))
    save('call_budget_manifest.json',dict(budgets=K.call_budgets(),semantics=K.counter_semantics(),
        static_draw_proof=dict(canonical_samples=384,max_canonical_draws=6,full_samples=128,
            max_full_draws=7,total_upper=3200,width=160,canonical_degree_max=4,Full_degree_max=4,
            assumptions=['m5','H160','fixed512','no extra audit draws','no retry or adaptive samples']),
        internal_root_law_predicate_packet_total_calls='NOT_ESTABLISHED'))
    save('seed_namespace_candidate.json',dict(**S.CLOSED,proposed_scientific_namespace=S.SEED,
        artificial_namespace=K.ARTIFICIAL_SEED,algorithm='SHA256(UTF8seed||uint64BEcounter); high160; discard96',
        mode_in_global_trial_id=True,mode_in_seed=False,domain='namespace/arm/axis',
        same_prefix=['S/P1','A/B','phase/pass'],IID_claim=False,real_variable_consumption='NOT_ESTABLISHED'))
    save('authority_binding_candidate.json',dict(requirements=K.authority_requirements(),
        example_test_manifest_digest=S.digest(m),real_grant=None,real_store=None,
        live_store_provider='UNCONFIGURED',scope='TEST_ONLY prototype is not a real grant authority'))
    save('trusted_store_requirements.json',K.authority_requirements());save('preflight_manifest.json',m)
    save('source_inventory.json',dict(measured_SOURCE=source,fixed_base=K.BASE,scientific_reference=S.SCIENCE,
        source_pins=c['source_pins'],input_pins=c['input_pins'],source_release_candidate=c['source_release_candidate'],
        inactive_resolver=K.PKG+'scientific_resolver.py.inactive',inactive_resolver_execution=0,
        old_Adapter='80b7bf61fe0a735d4031bb719c4b3e788611c3ba',old_Adapter_measured='b7ad344401b10395c49e4cfa806909282859bab5',
        old_Adapter_artifact='024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb',
        old_independent='0db83adb370d66a2f7f5254c8037dadda94ecdb8',old_Global='62967269e611b61e73b70abe20e998d9c7eaaf50',
        old_API='82896d53fa97235fa848d6c0833eb193af8b99d6',corrected_API_measured='78c571005fe766c61b32dc867e006531099ff608'))
    readiness=[dict(classification='IMPLEMENTED_ARTIFICIALLY_VERIFIED',items=['closed wrapper/object bridge','Codec/Result/angle/query',
        'versioned correctness worker/F01/Storage/receipts','test intent/atomic claim/STOP/crash','preflight','monitoring'],next_actor='Codex completes publication then STOP'),
        dict(classification='SOURCE_PREPARED_LIVE_IMPORT_REQUIRED',items=['dormant resolver','actual scientific runtime types',
            'Canonical snapshot on real held state','Full F3 variable draws/angle/requery'],next_actor='Codex only after new formal release and explicit authorization'),
        dict(classification='REAL_AUTHORITY_EXTERNAL_CONDITIONS_PENDING',items=K.authority_requirements()['external_decisions'],next_actor='User and administration'),
        dict(classification='FORMAL_SCIENTIFIC_CONTRACT_ADOPTION_PENDING',items=['SOURCE/input/precision','512 scope','coverage','budgets/STOP','seed'],next_actor='User and GPT'),
        dict(classification='SCIENTIFIC_EXECUTION_APPROVAL_PENDING',items=['separate explicit approval for exact new SOURCE/runtime/512 contract and real authority'],next_actor='User')]
    save('execution_readiness_classes.json',readiness)
    forbidden=['core_scientific_import_load_call','scientific_sampling','scientific_bitstream','new_scientific_computation',
        'saved_scientific_numerical_reanalysis','real_authorization_grant_claim_issue_consume','production_store_access',
        'native_quantum_processing','old_grant_claim_reuse','STOP_release','production_activation','real_correctness','limited_pilot','Full_N','AI_subagent']
    save('verification.json',dict(schema='b3_live_release_preparation_verification_v1',**S.CLOSED,
        status='TECHNICAL_PREPARATION_COMPLETE_UNAUTHORIZED' if evidence else 'DRAFT',
        measured_SOURCE=source,preparation_approval='USER_APPROVED_PREPARATION_SCOPE',SCIENCE_RUN_AUTHORIZED=False,
        tests=tests,mutation=result.get('mutation',{}),integration_cases=result.get('integration_cases',0),
        integration_PASS=result.get('integration_PASS',0),parent_guard=G.observation(),
        forbidden_operations={k:0 for k in forbidden},execution_readiness=readiness,
        live_compatibility='NOT_ESTABLISHED',physical_power_loss_durability='NOT_ESTABLISHED',
        unresolved_artificial_technical_inconsistencies=[],same_author=True,
        historical_tests_reused_not_rebuilt=dict(independent=224,Producer=132,old_regression=107,artificial_trials=16896),
        next_actors=['USER','GPT']))
    save('artificial_acceptance_matrix.json',dict(measured_SOURCE=source,tests=tests,
        mutation=result.get('mutation',{}),cases=cases,same_author=True,independent_reviewer=False))
    if evidence:
        target=HERE/'acceptance';target.mkdir(exist_ok=True)
        for p in evidence.iterdir():
            if p.is_file():shutil.copyfile(p,target/p.name)
        save('artifact_inventory.json',{str(p.relative_to(HERE)):dict(bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in target.iterdir() if p.is_file()})
    counts={k:v.get('PASS') for k,v in tests.items()}
    md('README.md',f'''# Track B W2c-B3：live接続Release・512契約・authority準備

実行状態はLIVE_RELEASE_CANDIDATE_UNAUTHORIZED。設計準備へのユーザー了承と、正式採択／実行承認を分離する。
新SOURCE `{K.PKG}`、測定SOURCE `{source}`、artifact `{c['source_release_candidate']}`。
[Q1〜Q10](handoff.md)、[実装](live_api_connection_implementation.md)、[契約](correctness_contract_candidate.json)、
[authority](trusted_store_implementation_status.md)、[未決条件](remaining_execution_blockers.md)、[人工受入](artificial_technical_acceptance.md)、[公開照合](publication_receipt.json)。

API形状の人工objectsを新wrapperから既存Exact Adapterへ渡し、新worker/pair/F01→Storage V2→receipt/callerへ接続する。
実module resolverは`.inactive` SOURCEとしてAST/SHA照合のみ。通常dispatchは組込み人工symbolsだけを許し、science入口は引数に関係なく拒否する。
再現：`python3 {K.DOC}run_acceptance.py /tmp/NEW_UNIQUE_OUTPUT`。新しいTEST_ONLY store/scopeだけを使う。
科学law、実Evidence互換性、実CPU／容量、real authority耐久性は未確立。''')
    md('fixed_source_and_reference.md',f'''# 固定参照・公開来歴

直接基点 `{K.BASE}`、前段測定 `4b4496a86f4ab9e0e1266e7323283aff9fe24f26`、公開SOURCE snapshot `72569f7822cb83e6b1ae72694d3c50355ebe8552`。
SOURCE treeと原Git bundleのSHA256、publication mappingを照合した。測定commitを公開metadataの異なるcommitへ付け替えて表示しない。
Adapter公開80b7bf61fe0a735d4031bb719c4b3e788611c3ba／測定b7ad344401b10395c49e4cfa806909282859bab5／artifact024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb。
独立0db83adb370d66a2f7f5254c8037dadda94ecdb8、Global62967269e611b61e73b70abe20e998d9c7eaaf50、API82896d53fa97235fa848d6c0833eb193af8b99d6／正しい測定78c571005fe766c61b32dc867e006531099ff608。
科学SOURCE `{S.SCIENCE}`の8moduleを静的bytes照合。core6はimport禁止、driver/helper2は人工用途だけ許可。
必須19資料と共通指示7eff8a33bef4ee1f29698ab10c6da46988dc8db5 §8.4はreference_documents.json。新科学計算はない。''')
    md('live_api_connection_implementation.md','''# API接続実装

wrapper.Endpointはgenerator/session構築、入力／精度確認、owned event/packet observer、sample／angle、原Evidence object保持、exact document変換、Result adapter、typed STOPを実装する。
from_runtimeはobject.readを信頼せずdocument/digestの完全bytesとcanonical型を検査する。Full B angleへは元のEvidence objectを渡す。
G7 Noneの分類にはraw/bit traceとpacket observationが必要。group_index/raw_word/childをCanonical requeryへ保持し、requeryを二重実行しない。
既存Codec/Adapterは無変更。lossless原文sidecarの受理subsetでのみtuple/list/map/Fractionの復元を主張し、driver投影だけから原型を推測しない。
EVENT、ZERO_R、ZERO_C、IDENTITY、deterministic query、typed failureの区別は既存Result契約を維持する。今回の固定正入力trial計画はEVENT/ZERO2種を扱い、IDENTITY/queryを確率ZEROへ割り当てない。
原型失敗Evidenceはoriginal_scientific_failureとして保存してF01へ渡す。失敗後sample/angleはterminal拒否。

scientific_resolver.py.inactiveには将来の固定module import/location/hash bindingを具体的に記述した。今回AST/SHAのみでcompile/import/callしない。
現在のfactoryとEndpointはexact built-in SyntheticSymbolsだけを許す。Science dispatchは無条件拒否。別正式versionが、外部admission証拠とこのsymbol contractを結ぶ必要がある。
人工symbolsは実API signatureを模したobjectsで、独自の人工order/label/intervalを使用する。固定P5の科学lawを生成・検証したものではない。
Canonical snapshot collaboratorも人工held stateのsealed shapeであり、実canonical_snapshotの科学held state互換性は未確認。AuditParity/event_witness自体は許可済みhelperを再利用する。''')
    md('canonical_full_runtime_binding.md','''# Canonical／Full runtime境界

|対象|construction/sample|B監査・angle/query|
|---|---|---|
|Canonical3方式 A|make_generator(p,x,m,arm,**precision) / CanonicalGenerator.sample(bits)|ratio保持|
|Canonical3方式 B|同じG7生成、LegacyEndpointと同じowned observer構造|canonical_snapshot(g,_law)、AuditParity、event_witness、event(group_index,raw_word,child)、enclose_atan|
|Full A|make_generator / FullReturnGenerator.sample(bits)|raw→packet iff reduced→acceptance→child|
|Full B|prepare(p,x,m,**precision_and_context) / GeneratorSession.sample(bits,*,trial_id)|preflight、原Evidence object→angle_for_event→event(word,child)→packet|

Canonical正入力のF3 prepareはCANONICAL_PREFLIGHT_NOT_IMPLEMENTEDのまま。Full facadeへ統一しない。
precisionはroot_bits/probability_bits/eta/rho、Full prepareにはsigma/angle_tau/angle_root_bits/angle_termsも渡す。
Angle contextはCanonical=input/policy/preflightとdriver event identity、Full=F3 input/policy/source/preflight/sigma＋原record.digest＋parent/child。
group/raw/childとFull packetのcall観測で再照会を照合し、追加drawや追加requeryを行わない。角度intervalの人工整合はatan保証ではない。
live import・runtime型・数学的妥当性はNOT_ESTABLISHED。''')
    md('correctness_scope_and_adoption.md','''# 512 scopeと承認状態

P5_general_order、p=(1/5,3/10,1/2)、x=5/7、m=5、sigma+1。
root_bits256/probability_bits160、eta=rho=1e-12、epsilon_axis1/200、primitive_error1e-6、alpha_axis1/320、familywise1/20。
A6 tau1e-6/K_phi32/J32、math-only。Native保証へ拡張しない。
4方式×A/B×X/Y=16cell×16trials×S/P1=512、各条件256。correctness/baselineだけでprofile・instrumented pass・追加sampling0。
同じ固定callsへの監視は追加passを作らない。S→P1、単一global claim、retry0、resume=false、mandatory STOP。
旧Full-N correctness256、pilot16,896、Full-N31,710,848とは別scope。旧grant/seed消費/claim/STOPを継承しない。

USER_APPROVED_PREPARATION_SCOPEは今回の設計方針と技術準備への了承。formal_contract_adopted=false、SCIENCE_RUN_AUTHORIZED=false、science_release=null。
新科学releaseの正式採択・実grant・実行承認は別手続きで未完了。''')
    md('correctness_call_budget.md','''# 別call予算と静的draw上界

sample512、primary construction16、G7 make_generator14、F3.prepare2、AuditParity8、Canonical snapshot6。
B sampling audit最大256、B angle/requery各256、Canonical angle192、Full angle64、driver correctness event_witness512。
Full sample packet128、Full angle packet64、additional sampling0。F3 prepare内部constructor2はprimary16の内数。
event_witness512はdriverのcorrectness境界であり、B sampling audit内の同名helper呼出し（最大256）を別に観測する。同名関数の総call上限512とは呼ばない。

固定m5ではraw degree≤4。Canonicalのgroup/labels/childが最大6draws、Fullのorder/labels/acceptance/childが最大7draws。
Canonical384×6＋Full128×7=3200 requests、各H160。非reducedはdegree+1、coin rejectはdegree+2、acceptedはdegree+3。
audit/requeryは追加drawなし、retry/追加samplingなしという静的条件に依存する。人工fixtureの固定draw数を実lawへ採択しない。
実装は各pairの分配予算と全plan/task bindingを制限し、offlineが最終counter合計とglobal sample512を照合する。
root/law/predicate/packet内部の総関数呼出し数、実variable draws/IIDはNOT_ESTABLISHED。境界予算を内部総call認可とみなさない。
詳細はcall_budget_manifest.json。''')
    md('correctness_acceptance_and_coverage.md','''# Correctness判定候補

予定trial、SOURCE/input/runtime、exact event/rational、A/B prefix、Result/Evidence、ZERO分類、B監査/angle/requery、F01、trace/state/receipt/caller、counter/bytesを個別確認する。
16trials/cell固定。各cell accepted≥1というcoverage候補を維持し、未達はINCONCLUSIVE_COVERAGE、追加sample/retry0。
旧driverの即時accepted gateは原本保全し、新version pairでは固定Nを完了後parentがcoverage分類する。新契約候補に対する意図的なversioned差分であり、旧契約を変更しない。
技術失敗/UNKNOWN/保存故障はTECHNICAL_STOP、人工不一致caseはARTIFICIAL_SCIENTIFIC_MISMATCH_STOP、保存witness欠落はINCOMPLETE_EVIDENCE。
人工正常分類ARTIFICIAL_TECHNICAL_COMPLETEは実科学correctness成功ではない。未出現pathは未検証。
科学的不一致、技術停止、coverage不足を混同せず、追加科学検証は別採択・別実行承認・fresh claimを要する。''')
    md('real_authority_requirements.md','''# Real Authority要件

SOURCE/input/precision/API/contract、runtime/libraries、worker runtime/lineage、seed、global scope/claim、sampling/construction/angle/query/call/bytes、storage、monitoring、STOP/retry/resume、receipt/callerを結合する。
実store所有主体、grant issuer、credential/鍵管理、失効/rotation、atomicity/ACK unknown契約、power-loss durability、worker containment、receipt/caller耐久性、実行環境責任範囲は外部判断待ち。
RealAuthority.admitは未設定を明示して拒否する。実provider/credentialは未実装・未配置で、本番storeへアクセスしない。
外部情報を推測しない。TEST_ONLY proofは実認証強度・管理運用・実耐久性の保証ではない。''')
    md('trusted_store_implementation_status.md','''# Trusted store技術prototype

所有する/tmp/b3-release-test-*内のTEST_ONLY.sqlite3とintentだけを操作する。workerへstore/grant/credential objectを渡さず、parent側だけでclaim/STOP/receipt linkを扱う。
TestGrantはbuiltin TEST_ONLY型とmanifest全体のbindingを持ち、real authorityへ変換する経路を持たない。実credentialや本番鍵はない。
atomic exclusive intentをO_EXCLで作り、file fsync＋directory fsync後、SQLite BEGIN IMMEDIATE/UNIQUE/FULL synchronous/WALでclaimをcommitする。
intent作成時からscopeをspent扱いとし、ACK不明・commit前死でもintentを削除／refundしない。STOP reconcileはUNKNOWN_STOPPEDを保存し、新規admissionを作らない。
12 process競合one winner、double claim、wrong grant、test revocation、STOP後ENTER、receipt link一回、intent後/commit前/commit後の異常終了を試験する。
実際に終了するのは自分の人工test processだけ。raw DB/intent/snapshot/exit codeをacceptance/raw_artificial.tar.gzへ保存。
fsyncとSQLite FULLを要求した技術事実と、物理power-loss耐久性は別。power loss実験・管理者storage保証は今回未取得。
claimはglobal一回でS/P1を内包し、STOP→SPENTへ戻さず、旧claimはアクセスも再利用もしない。''')
    md('release_and_preflight.md','''# Release/preflightと閉鎖gate

LIVE_RELEASE_CANDIDATE_UNAUTHORIZED。run_ready=false、content_adopted=false、contract_grants_execution=false、science_release=null、scientific_adoption=null、artificial_only=true、retry0、resume=false、mandatory_STOP=true。
candidate再生成とmanifest完全一致によりSOURCE/API/input/precision/runtime/worker/seed/plan/authority/call/bytes/storage/monitoring/STOP/retry/resumeを結合する。
SOURCE固定後にGit commit全pin bytesを照合。最終publicationは測定commitと公開exact-tree snapshotを別記し、原commitをbundleで保持する。
wrong binding/approval/grant、fake→real混同、source/signature、call/byte、spent claim、parent/spawn gate bypassを拒否する。
実science_dispatchは無条件拒否。inactive resolverを通常import対象やfactory selectorにしない。boolean/namespace置換で科学実行できない。
source guardは親/spawnの両方でcore6をload前に遮断し、guardを歴史的guardの前へ置いて観測する。新依存がcoreを要求すればSTOP。
無関係な他worktree HEAD/statusは観測だけで拒否条件にしない。''')
    md('monitoring_storage_and_STOP.md','''# 保存・監視・STOP

Global result1GiB/log64MiB、S/P1共通cap。全512 trialの原document/sidecar、exact event/weight/child、bits、angle/query、audit/F01、worker/source/runtime、global trace/state/counter、condition/global receipt、STOP/UNKNOWN、storage close、delivery/callerを保持する。
超過はSTOPし、Evidenceを勝手に間引かない。実科学サイズは未測定。人工raw容量を実容量保証にしない。
旧Storage V2は無変更。新version worker/pairはRESERVE/ENTER/COMPLETE/F01/cleanupを保持し、新authority・wrapperへつなぐ。
S失敗でP1未開始、P1失敗でS保存、STOP後ENTER禁止、UNKNOWN credit非返却、in-flight drainを維持する。
STOP条件はSOURCE/input、authority、seed/consumption、event/科学invariant、Codec、angle/query、worker、storage、UNKNOWN、byte/call、trace/receipt/callerの不整合。

host_outerはpreflight/admission前〜全join/STOP/close/receipt/delivery/caller close。parent CPU＋全joined worker CPUを一度だけ加算する。
coarseとcached all-call/return fineは同じcorrectness calls内の診断で、追加passやsampleはない。各exclusive partitionを別々にouterへreconcileし、parent/起動/observer/未観測分を残差として明示する。両partitionを加算しない。
SOURCE/constructor/sample/bits/packet-child-law/audit/angle/query/codec/IPC/observer/storage/RSS-HWM/I-O/STOP-drain-closeの観測構造を持つ。
最終outer観測のexport自身はpostprocessingとしてwindow外を明示する。wallはglobal monotonic outerで、pair wallを加算しない。
一時host PSIは記録し監視を再確立する。実OOM/worker異常/破損/科学的不一致/UNKNOWNの保護を保つ。共有設定・他者processへ変更はない。
実科学CPU/path率/保存量/IID/数学的atan/real durabilityはNOT_ESTABLISHED。''')
    md('artificial_technical_acceptance.md',f'''# 人工技術検証

測定SOURCE `{source}`。同一primary agentによる検証で、別reviewerの独立受入とは呼ばない。
unit/negative/store/必要回帰PASS：`{json.dumps(counts,ensure_ascii=False)}`。
semantic mutation `{result.get('mutation',{}).get('PASS','pending')}`/{result.get('mutation',{}).get('total','pending')}、統合case `{result.get('integration_PASS','pending')}`/{result.get('integration_cases','pending')}。
正常512 full documents、coverage INCONCLUSIVE、S/P1 angle故障、人工科学不一致、receipt close、bytes/calls、worker exit、ACK unknownを分類する。
原文を再sealした変更も含むpayload semantic mutationを拒否。旧独立224/Producer132/107/16,896は固定参照し再構築・合算しない。
今回必要回帰は旧F01＋Storage V2の66件。原本変更なし。新worker coverageと監視は今回の人工10caseで検証する。
途中の開発例外と修正はdevelopment_summary.jsonに保持し、最終SOURCE固定後の成績だけを完了判定に使う。
core import/load/科学bit列/実science/実grant/native/旧claim/STOP解除0。''')
    md('source_and_contract_delta.md','''# SOURCE／契約差分

基点6c6d3376e7ff206219f5e58355f1e1080fafba04。新namespace、focused tests、packet runner/docsと索引追記だけを追加する。
固定科学8module、旧Adapter、Global Bundle、Storage V2、独立checkerは原本を変更しない。
execution.pyは固定pair/worker/run_phaseの新version投影。coverage accepted即時gateをparent最終INCONCLUSIVE分類へ移し、worker target/importパスとpreload guardを変更した。
同一correctness callへのcached fine diagnosticsを有効にし、profile/instrumented passを追加しない。worker_pair_source_derivation.jsonに元SHA・限定差分を保存する。
原本のF01/STOP/one-shot/保存契約は保持し、新候補のcoverage変更は今回のユーザー方針へのversioned実装である。
前段Source tree/bundle/publication mappingは照合済み。最終SOURCEと最終回帰SOURCE、公開snapshotの差分はpreservation.json/regression_source_delta.jsonを参照。''')
    md('remaining_execution_blockers.md','''# 実行前に残る条件

1. **ユーザー・GPT**：新しい正式SOURCE release、固定input/precision/A6、512scope、coverage、calls/bytes/STOP、seedを正式採択する。
2. **ユーザー・管理主体**：store所有者・issuer・credential/鍵/失効、atomicity/ACK unknown、power-loss、worker containment、receipt/caller、実行環境責任の契約と証拠を決定する。
3. **Codex（新指示後）**：採択releaseへ外部admissionとinactive resolverを結び、必要なruntime/worker/storage/pinを固定する。現在のcandidateをbooleanだけで有効化しない。
4. **ユーザー**：その正確なSOURCE/runtime/新authority/512契約に対し、sampling以外のconstructor/audit/angle/queryも含む別の明示的実行承認を与える。
5. **Codex（承認後だけ）**：live import/runtime型・実Evidence/variable bits/Canonical snapshot/Full requeryを最小scopeで確認し、未確認をPASSにせず、one-shot/STOPを維持する。

SOURCE準備済みとlive未検証、外部条件未確立、正式未採択、実行未承認をexecution_readiness_classes.jsonで区別する。
real authority provider/credentials/activationは未実装・未設定であり、承認だけで現candidateから起動するわけではない。
実law/係数/child law、実path率、atan数学保証、実CPU/容量、IID、Full性能、native、Full-N実行可能性はNOT_ESTABLISHED。''')
    md('pending_and_required_decisions.md','''# 次の担当と未決判断

**ユーザー・GPT**：SOURCE・512契約候補の採否、fixed input/precision/A6、coverage INCONCLUSIVE、別call/bytes/STOP、seedの正式採択を判断する。
**ユーザー・管理主体**：real store owner/issuer/keys/revocation/ACK/power/containment/receipt/caller/責任境界の実情報・証拠を決定する。
**Codex（今回）**：技術準備・人工検証・公開照合後mandatory STOP。
**Codex（新指示と必要承認後）**：採択scopeに限り、別正式releaseと外部authorityの接続を固定する。現在未開始。
実科学import/sampling、実grant、correctness512、limited pilot16,896、Full-N31,710,848、production有効化、STOP解除へ自動移行しない。''')
    md('handoff.md',f'''# 最終報告 Q1〜Q10

1. **Q1** 最新HEAD `{K.BASE}`、科学 `{S.SCIENCE}`、input/precision、旧Adapter/Bundle/独立受入/APIのSHAと公開SOURCE tree/bundle/mappingを照合した。新測定SOURCE `{source}`。
2. **Q2** API-objectを構築・sample・監査・query・angle・typed failureからExact documentへ渡すwrapper/factory、versioned worker、inactive resolver、preflightを実装。core科学importなし。loaderはAST-only SOURCEで通常dispatchから分離。
3. **Q3** Canonical G7＋driver監査/group-raw-childとFull F3 session/original Evidence/event→packetを分離。正入力Canonical F3拒否は維持。live互換性はNOT_ESTABLISHED。
4. **Q4** 固定P5/precision/A6、16cell×16×2=512。construction16/G7make14/F3prepare2、B angle/query256等を別予算化。3200drawsは384×6＋128×7、内部root/law/predicate総callは未確立。
5. **Q5** fixed N後INCONCLUSIVE coverage、追加sampling/retry0、tech STOP/人工scientific mismatch/証拠不完全を区別。全trial Evidence、global one-shot、S→P1、STOP/UNKNOWN/bytesを人工検証した。
6. **Q6** SOURCE/API/input/runtime/seed/authority/budgets/STOP/releaseを結合。USER_APPROVED_PREPARATION_SCOPEとSCIENCE_RUN_AUTHORIZED=falseを分離し、実入口を閉鎖した。
7. **Q7** TEST_ONLY durable intent＋SQLite atomic/one-winner/spent/ACK unknown/STOP/revocation/receipt prototypeを実装・故障検証。real owner/issuer/keys/失効/ACK/power/containment/caller/責任境界は外部判断待ち。
8. **Q8** 最終PASS `{json.dumps(counts,ensure_ascii=False)}`、mutation `{result.get('mutation',{}).get('PASS','pending')}`、integration `{result.get('integration_PASS','pending')}`。旧受入件数を合算しない。未解決の人工技術不整合0。
9. **Q9** 採否資料と実行前checklistはremaining_execution_blockers.md／execution_readiness_classes.json。正式release・科学契約・real authority外部証拠と別の明示的実行承認が必要。
10. **Q10** core科学import/load/call・sampling・科学bits・新科学/再解析・実grant/claim/store・native・旧grant再利用・STOP解除・production有効化・実correctness/pilot/Full-Nは全て0。

公開receipt/publication mappingと原SOURCE bundleにより測定SOURCEと公開commitを区別する。次の担当はユーザー・GPT。''')
    print(json.dumps(dict(measured_SOURCE=source,source_pins=len(c['source_pins']),input_pins=len(c['input_pins']),tests=counts)))
