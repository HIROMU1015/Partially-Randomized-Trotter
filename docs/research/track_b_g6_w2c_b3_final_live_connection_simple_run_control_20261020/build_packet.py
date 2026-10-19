"""Render evidence packet from frozen acceptance; no science import or execution."""
from pathlib import Path
import sys,json,shutil,hashlib,subprocess
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import contract as K,guard as G
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import spec as S


def save(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')


def build(src,public_source):
    src=Path(src);r=json.loads((src/'result.json').read_text());c=json.loads((src/'candidate.json').read_text())
    source=r['measured_SOURCE'];K.verify_source_commit(c,source)
    assert all(x['FAIL']==x['ERROR']==x['SKIP']==0 for x in r['tests'].values())
    assert r['mutation']['PASS']==20 and r['integration_PASS']==r['integration_cases']==10
    acceptance=HERE/'acceptance';acceptance.mkdir(exist_ok=False)
    for p in src.iterdir():
        if p.is_file() and (p.suffix in ('.json','.log') or p.name=='raw_artificial.tar.gz'):shutil.copy2(p,acceptance/p.name)
    cases=json.loads((src/'cases.json').read_text());static=json.loads((src/'static_preparation_audit.json').read_text())
    normal=cases[0]['verification'];raw=acceptance/'raw_artificial.tar.gz'
    common=dict(S.CLOSED,SCIENCE_RUN_AUTHORIZED=False,preparation_approval='USER_APPROVED_PREPARATION_SCOPE')
    forbidden={n:0 for n in ['core_scientific_import_load_call','scientific_loader_call','scientific_sampling',
        'scientific_bitstream','real_permission_issue_consume','real_store_access','real_correctness_512',
        'limited_pilot','Full_N','old_grant_claim_reuse','STOP_release','production_activation','native',
        'saved_scientific_reanalysis','AI_subagent']}
    verification=dict(r,**common,status='FINAL_TECHNICAL_PREPARATION_COMPLETE_UNAUTHORIZED',
        measured_SOURCE=source,public_SOURCE_snapshot=public_source,
        forbidden_operations=forbidden,source_git_pins=json.loads((src/'git_source_check.json').read_text()),
        normal_artificial_full_documents=normal['full_documents'],one_shot='ARTIFICIALLY_VERIFIED',
        activation_anchor=None,live_compatibility='NOT_ESTABLISHED',
        unresolved_artificial_technical_inconsistencies=[],
        historical_reused_not_rerun=dict(artificial_trials=16896,independent_checks=224),
        raw=dict(path='acceptance/raw_artificial.tar.gz',bytes=raw.stat().st_size,sha256=hashlib.sha256(raw.read_bytes()).hexdigest()),
        next_actors=['USER','GPT'])
    save('verification.json',verification);save('final_live_connection_candidate.json',c)
    save('correctness_run_manifest_candidate.json',K.run_manifest_candidate(c,source_commit=source))
    save('simple_run_control_contract.json',K.authority_requirements())
    save('call_budget_manifest.json',dict(budgets=K.call_budgets(),semantics=K.counter_semantics(),
        observed_artificial=normal['boundary_counts'],scientific_cost='NOT_ESTABLISHED'))
    save('source_inventory.json',dict(measured_SOURCE=source,public_SOURCE_snapshot=public_source,
        fixed_references=static['references'],source_release_candidate=c['source_release_candidate'],
        source_pins=c['source_pins'],input_pins=c['input_pins'],runtime=c['python'],worker_runtime=c['worker_runtime'],
        resolver_binding=c['final_runtime_binding'],old_regression_SOURCE=static['old_regression_SOURCE'],
        new_regression_SOURCE=source,old_fixed_pin_comparison='acceptance/static_preparation_audit.json',
        source_git_check=verification['source_git_pins']))
    save('artificial_acceptance_matrix.json',dict(tests=r['tests'],cases=cases,mutation=r['mutation'],same_author=True))
    save('development_history.json',dict(authoritative_SOURCE=source,
        development_runs=[dict(path='/tmp/b3-final-development-01',status='normal artificial Bundle PASS before setup additions'),
            dict(path='/tmp/b3-final-development-02',status='normal + shared pair validation PASS'),
            dict(path='/tmp/b3-final-development-03',status='raw-export teardown rejected expected symlink state; fixed recorder to preserve denial and symlink'),
            dict(path='/tmp/b3-final-development-04',status='wrong-location negative rejected earlier by inherited guard; broadened expected reason to LOADED_LOCATION'),
            dict(path='/tmp/b3-final-development-05',status='raw copy raced delayed SQLite connection GC; explicitly close every owned store connection before export')],
        note='Dirty development HEAD labels are not measured SOURCE. All final counts below come from the frozen final acceptance.'))
    count={k:v['PASS'] for k,v in r['tests'].items()};footer='\n\n**実科学未実行、実行未承認。mandatory STOP。次の担当：ユーザー・GPT。**\n'
    pages={
    'README.md':f'''# Track B W2c-B3：最終接続・簡易実行管理・512件準備

実G7/F3 resolver、runtime factory、G7 Bits委譲、Exact Evidence、保存監査と簡易one-shotを新namespaceに実装した。
測定SOURCE `{source}`、公開SOURCE snapshot `{public_source}`、artifact `{c['source_release_candidate']}`。

[Q1〜Q10](handoff.md)・[接続](final_live_connection.md)・[512契約](correctness_512_contract.md)・[簡易管理](simple_execution_management.md)・[未確立事項](remaining_live_validation.md)・[verification](verification.json)・[公開bytes照合](publication_receipt.json)。
SOURCE `{K.PKG}`、test `tests/tracks/algorithm_codesign/test_g6_w2c_b3_final_live_v1.py`。
再現は新しい人工出力先を使い、`python3 {K.DOC}run_acceptance.py /tmp/NEW_UNIQUE_ACCEPTANCE`。
このrunnerは人工専用で科学loaderを呼ばない。親・spawnでCore6 load/attempt 0。

新規/焦点tests {count}、semantic mutation20、integration10分類PASS。旧66回帰を同じ測定SOURCEで実施。
旧16,896 trials・独立224 checksは固定参照だけで再実施していない。実API互換性・実科学費用・数学的精度の検証結果にはしない。
''',
    'fixed_source_and_input.md':f'''# 固定SOURCE・入力・runtime

直接基点 `{K.BASE}`、旧測定 `2405f5e2e5fb0dc5add39b102e3ccae3d56df0a4`、新測定 `{source}`。
科学参照 `{S.SCIENCE}`。Core6はGit bytes/SHA256/ASTだけで照合し、import/loadしなかった。
旧114 SOURCE pinsと10 input pinsは旧測定・基点公開HEAD・新worktreeで等しい。
詳細は[source inventory](source_inventory.json)と[静的監査](acceptance/static_preparation_audit.json)。

P5_general_order：p=(1/5,3/10,1/2)、x=5/7、m=5、sigma=+1。
root256/H160、eta=rho=1e-12、epsilon_axis=1/200、primitive=1e-6、alpha_axis=1/320、familywise=1/20。
A6 tau=1e-6/K_phi=32/J=32、math-only。Nativeへ拡張しない。
Python executable、実際にloadedしたstdlib/shared library、worker runtime/schema/import pathsをSHA256で固定する。
正式共通指示7eff8a33の§8.4を実際に読み適用。空き資源観測を静的監査JSONへ保存。
保護4 worktreeは既知pathのHEAD/status観測だけ。無関係な変更を実行拒否条件にしない。
''',
    'final_live_connection.md':'''# 最終接続SOURCE

resolver.static_bindingはCore6と許可済みhelperのAST signature/argument shape/location/SHAを取得する。
resolver.resolveは承認gate、固定preflight、parentのSPENT状態を確認した後だけimportlib resolverへ進む。
loaded symbolsは__file__・code filename・qualname・signatureを再照合する。classmethodは元の関数を照合。
同じworker内のsymbols再利用でもsource/location/permitを確認する。今回resolveは一度も呼ばなかった。

wrapper.ScientificFactoryとScientificBitsFactoryはspawn可能な固定c/m/permitを受け取る。
Endpointは既存parallel_driver.LegacyEndpointのversioned subclassで、admissionとobservationを新しい接続へ置き換える。
旧LegacyEndpoint constructorを無断で有効化しない。Canonical/Fullの元のAPI経路と監査・angle contextを保つ。
人工Factoryはexact builtin Symbolsだけを受け取り、任意objectをreal symbolsへ昇格させない。

G7側では実sampleのraw/bit stage、owned event/packet observerから枝を捕捉し、None単独からZEROを推測しない。
Full Bは元のF3 Evidence objectをangle_for_eventへ渡し、document/digestの完全bytesを検査・sidecarに保存する。
Canonicalのgroup_index/raw_word/child、Fullのraw/reduction/packet/coin/child/fingerprintを保持する。
原型typed failure、取得済みResult、観測済みbits、Audit failure sealをF01へ渡す。失敗後はterminal拒否。

bits.ScientificBitsは固定G7.DeterministicBitsを委譲先にする。追加hash/drawをせず位置・幅・消費を記録する。
今回の動的Bits試験はARTIFICIAL_FINAL_LIVE_V1だけ。science namespaceは文字列候補のままでbit列0。
setup証拠は各pairの最初の行へoriginal documentとして保存し、全行からdigestで参照する。
保存監査はSOURCEを呼ばず、既存Exact Codec/Resultと同じ契約で元文書・audit/angle/query/prefix/budgetを照合する。
''',
    'canonical_full_runtime_mapping.md':'''# Canonical/Fullの経路

|方式|Layer A|Layer B sample/audit|B angle/requery|
|---|---|---|---|
|ordinary|G7.make_generator → CanonicalGenerator.sample|versioned LegacyEndpoint / canonical_snapshot / AuditParity / event_witness|event(group_index,raw_word,child) → enclose_atan、Canonical context|
|partial_return_tail|同上|同上|同上|
|closed_P3_tail|同上|同上|同上|
|full_return|G7.make_generator → FullReturnGenerator.sample|F3.prepare → GeneratorSession.sample / F3 preflight / AuditParity|angle_for_event(original Evidence) → event(word,child) → packet|

Canonical正入力をF3 facadeへ流さず、CANONICAL_PREFLIGHT_NOT_IMPLEMENTEDは保持する。
Canonical angleはAuditParity内で元のgroup/raw/childによるrequeryを1回だけ行い、元のcontext/EvidenceとSealを保存する。
Full Bのangle requeryはF3内部の1回をobserverで捕捉する。別の診断queryやsamplingを追加しない。
EVENT/ZERO_R/ZERO_C/IDENTITY/query/typed failureは混同しない。今回の固定正入力sampling計画はEVENT/ZERO2種だけ。
full angle enclosureの数学的正しさ・実Canonical held state・実F3 typed objectsの互換性は今回未検証。
''',
    'correctness_512_contract.md':f'''# 512件correctness契約候補

[manifest候補](correctness_run_manifest_candidate.json)は未承認テンプレート。予定Run IDと新しいoutput/state pathを含み、directory・real permission・claimは作成していない。
4方式×A/B×X/Y=16cell、各16trials、S/P1それぞれ256、合計512。
correctness/baselineだけ、profile/instrumented pass/追加sampling0。S→P1、単一global one-shot、retry0/resume=false。

Primary16、G7make14、F3.prepare2、AuditParity8、canonical snapshot6。
B sampling audit/angle/requery各最大256、Canonical angle192、Full angle64、correctness event_witness512。
Full sample packet128、Full angle packet64。F3.prepare内部constructor2はprimary16の内数。
Nested B witnessとdriver correctness witnessを分け、同名関数の総call上限512とはしない。
3200draws案はCanonical384×6＋Full128×7、固定m5・raw degree≤4・H160・audit/query追加draw0・retry0に依存する。
内部root/law/predicate等の総call数は未確立。[別予算](call_budget_manifest.json)。

Science seed候補 `{S.SEED}`、SHA256 counter方式、domain=namespace/arm/axis。
modeはglobal trial identityに含め、seed domainに含めない。A/B・S/P1の対応prefixを比較する。
各cell accepted≥1を維持し、16trials完了で0ならINCONCLUSIVE_COVERAGE。延長・retryなし。
科学不一致・技術STOP・証拠不完全を別分類にする。
保存はS/P1共通result1GiB/log64MiB、全512 full Evidence。超過時はSTOPし間引かない。
実science Evidenceの容量十分性、accepted/reject分布、IIDはNOT_ESTABLISHED。
''',
    'simple_execution_management.md':'''# 簡易実行管理

承認者はユーザー本人。準備了承と実行承認を分離し、今回有効な実行許可は0。
従来の外部issuer/keys/revocation/authorization server等は新scopeの必須工程から外した。
旧SOURCE・claim・STOPの契約や記録は変更していない。変更は今回§5の方針に基づく候補であり、正式な512実行契約採択はユーザー・GPTへ戻す。

run_control.SimpleStoreは所有する専用rootだけを使う。現在作成できるのは人工専用prefixのstoreとMockApproval。
実用rootではpermit gateが先に拒否する。実storeへの接続・許可生成は今回0。
同じRun IDとscopeの両方をO_EXCL intentで予約し、SQLite UNIQUE/BEGIN IMMEDIATE/FULL/WALで記録する。
一方のintentだけ、空/partial intent、commit/ACK不明、process死でもspent扱いで、削除・refundしない。
NOT_STARTED→START_RESERVED/SPENT→COMPLETED/STOPPED/UNKNOWN。STOP/UNKNOWN/COMPLETED後ENTER禁止。
COMPLETEDは512完成、storage close、receipt/delivery/caller証拠確認後だけ。coverage不足でも固定Nを完了した実行状態と科学的判定は分ける。
STOP markerのclaimは常にspentで、COMPLETEDへの最終記録も再実行権限にならない。

12process競合one winner、同Run別scope/同scope別Run拒否、3crash点、partial intent、STOP/UNKNOWN、receipt一回、symlink拒否を人工試験した。
file/directory fsyncとSQLite FULLを要求した事実を記録し、物理power-loss完全耐久性や悪意あるownerの改ざん防止は主張しない。
専用storeとinput/provenanceの固定管理費はresult/log output capの外。科学Evidenceをstoreへ逃がす経路はなく、全trial保存はglobal cap内。
''',
    'preflight_and_execution_gate.md':'''# Preflightと実行入口

candidate再生成、固定Git SOURCE/input、API AST/signature、Python/lib/worker、new namespace loaded location、seed/plan/call/bytes/STOP/outputを照合する。
現在SCIENCE_RUN_AUTHORIZED=false/run_ready=false/contract_grants_execution=false/science_release=null/retry0/resume=false/mandatory_STOP=true。
activation.APPROVED_ACTIVATION_ID=None。fake approval、booleans、forged LivePermit、runtime属性だけの書換えでdispatchできない。
親・spawnのgate試験を保存し、Core6 load/attempt 0を確認した。実resolverは呼ばない。

後日の承認手順（今回は実施せず）：
1. ユーザー・GPTが固定SOURCE/manifest・512scope/seed/coverage/予算/STOPと簡易管理差分を採択。
2. activation-only releaseで承認IDを設定し、実科学ロジックを変えず、最終SOURCEとmanifestを再固定する。
3. その正確なSOURCE/manifestとconstructor/audit/angle/queryを含むscopeへのユーザーの明示実行承認本文を記録する。
4. approval target digestを結合した新しい実行manifestを完成させ、新しい専用state rootで一度だけ開始する。
5. parentがSPENTを保存後、worker resolver/ScientificBitsを許す。異常/UNKNOWN/STOPは非返却、retry/resumeなし。

activation.admit/validate、factory/resolver/Bits、worker/Bundle/Store接続の処理本体は実装済み。残る有効化は承認ID・承認記録とSOURCE固定という正式手続きであり、新しい認可サービス開発ではない。
この未承認候補のbooleanだけを変更して有効化しない。activation-only変更もSOURCEの来歴へ明示する。
''',
    'storage_monitoring_and_STOP.md':'''# Storage・monitoring・STOP

Global scheduler/spawn/pair/F01/Storage V2/trusted prefix/global receipt/post-close callerを再利用。
新execution.run_pairのASTは旧最終準備版と同じで、旧ファイルは無変更。
単一global claimにS→P1を内包し、S失敗でP1未開始、P1失敗でS証拠を保存。
SOURCE/input/runtime/承認/seed/消費/Codec/Result/angle/query/科学invariant/worker/storage/call/bytes/UNKNOWN不整合でSTOP。
保存完成前の未確認結果を成功にせず、in-flight drain・UNKNOWN credit非返却・STOP後ENTER禁止を維持。

全512の原文/digest/sidecar、exact rational/event/child、bits、setup/audit/angle/query、F01、source/runtime/worker/global IDsとtrace/state/receipt/caller/STOPを保持する。
成功receiptの前に保存済みpairのA/B・S/P1 prefix、全512件、fixed N、counter予算を照合する。追加samplingなし。
結果/logのglobal capはS/P1で共有し、超過時はSTOP。人工rawの容量を実科学の容量保証にしない。

Parent CPU＋全joined worker CPUを一度だけ加算し、wallはglobal outer。coarse/fineは別々にreconcileして加算しない。
constructor/setup、online sampling、bits、law/child、audit、angle/query、Codec、IPC、observer、storage、RSS/HWM、I/O、STOP/drain/closeを観測。
outer windowはpreflight/admission前から全join/STOP/storage close/receipt/delivery/caller closeまで。
最終outerのexportとsimple run stateの最終チェックポイントはpostprocessingとしてwindow外。byte cap内の予約slotへ保存する。
独立caseは別parent processで並列化し、RUSAGE_CHILDRENをcase間で混合しない。S内は固定逐次、P1は独立8pairをspawn。
一時host PSIだけで正常workerを停止しない。実OOM・worker異常・破損・不一致・UNKNOWNの保護を維持。
''',
    'artificial_acceptance.md':f'''# 人工受入・差分回帰

測定SOURCE `{source}`、test PASS `{count}`、mutation20/20拒否、integration10/10期待分類。
[受入matrix](artificial_acceptance_matrix.json)、[raw](acceptance/raw_artificial.tar.gz)、[tests](acceptance/tests.json)、[静的監査](acceptance/static_preparation_audit.json)。
正常・coverage不足・S/P1 angle故障・人工invariant不一致・receipt close・bytes/call cap・worker死・ACK unknownを区別する。
正常は512全EvidenceとCOMPLETED、coverage不足はINCONCLUSIVE_COVERAGE、故障はTECHNICAL_STOPまたはINCOMPLETE_EVIDENCE。
20semantic mutationはglobal ID/source/runtime/session/pid/trial/event/digest/sidecar/seed/width/counter/angle/query/auditとresealed originalを検査。
同一作者による検査であり別reviewerの独立受入ではない。Exact Codec/Result validatorを共有する。

旧回帰SOURCE `2405f5e2e5fb0dc5add39b102e3ccae3d56df0a4` と新測定SOURCEで旧124pin bytesが等しく、F01 pair ASTも同じ。
旧F01/Storageの必要66 checksを新SOURCEで実行した。旧16,896/独立224/Producer等の件数を新結果へ合算していない。
人工symbolsはAPI形状fixtureでありP5科学lawやatanを計算しない。held-state文書の人工試験も数学検証ではない。
開発時の記録処理・connection close・期待reason修正はdevelopment_history.jsonへ保存。最終判定はfrozen SOURCEだけに基づく。
''',
    'source_and_contract_delta.md':'''# SOURCE・契約差分

変更は新g6_w2c_b3_final_live_v1、専用test/runner/packetと4 indexのみ。
旧科学Core6、driver/helper、Exact Adapter、Global Bundle、Storage V2、独立checker、旧raw/claim/ledger/marker/STOPは無変更。
新wrapperはAPI-shaped専用からgated ScientificSymbols/factoryへ拡張し、実resolver/signature/G7 Bits委譲とsetup保存/成功前pair監査を実装。
旧Canonical/Full分岐とF3正入力Canonical拒否を維持。F01 pair ASTは旧最終準備版から無変更。
新Storeは旧TEST_ONLY intent/SQLiteを基に、Run IDとscope両方のspend、COMPLETED/UNKNOWN、簡易ユーザー承認へ変更。

512 sampling・別constructor/audit/angle/query予算・seed/prefix・fixed N coverage・1GiB/64MiB・S→P1・retry0・STOP・全Evidenceは維持。
旧Real Authorityの外部owner/issuer/keys/revocation/server/power-loss証拠等を新実行の必須工程から外す差分は、今回ユーザー§5の明示方針による。
旧契約を遡及変更せず、新scopeの候補として採択を待つ。未取得power-loss保証を成立済みにしない。
現在のactivation anchor未設定も固定SOURCEの一部。後日のactivation-only来歴を隠さない。
''',
    'remaining_live_validation.md':'''# 実行時に残る確認

実resolverを呼んでいないため、固定科学moduleの実load・runtime callable/type/constructorと実Evidence互換性はNOT_ESTABLISHED。
実Canonical canonical_snapshot/held state、Full F3 preflight/variable consumption/original Evidence/angle/query/typed failureは人工試験から証明しない。
SOURCE location/hash/signature、exact original bytes、bit positions、source state、invariant、budget、prefix、Storage/STOPの検査処理は実装済み。
承認後にどれかが失敗したら技術STOPし、そのscopeでの追加sampling・retryはしない。

実G7の法則/係数/child law・atan数学保証の新しい独立証明、IID、accepted/reject率、実CPU/bit費用/保存量、性能優位性、native、Full-N可能性は未取得。
初回512 correctnessは性能優位性を主張する工程ではない。人工rawサイズから実容量十分性を保証しない。
外部認証サービス開発・鍵管理・power-loss完全保証は次工程の必須研究課題にしない。
''',
    'pending_and_required_decisions.md':'''# 必要な採択・次の担当

ユーザー・GPT：固定SOURCE、P5/precision/A6、512 scope、別call/byte予算、seed、coverage、簡易管理差分、STOPを確認・採択する。
ユーザー：正式に固定したSOURCE/実行manifestに対し、samplingとconstructor/audit/angle/requeryを含む別の明示的実行承認を判断する。
Codex（その指示後だけ）：activation-only記録と最終manifestを固定し、実科学ロジックを変えず承認対象を照合する。承認された新scopeでだけone-shotを開始する。
今回のSOURCEは実処理本体の実装済み候補であり、実科学を自動で開始しない。
現在run_ready=false、承認IDなし、real permission/claim/storeなし。旧STOPは維持。
''',
    'handoff.md':f'''# 最終報告 Q1〜Q10

1. **Q1** 基点841ac892、旧測定2405f5e2、科学76ca3c40と旧Adapter/Bundle/API/独立参照の完全SHAを静的監査JSONで照合。旧124 pins、新SOURCE Git pins、input/precision/runtimeを確認。新測定 `{source}`、公開SOURCE snapshot `{public_source}`。公開最終HEADと全公開bytesはpublication receipt/transportと最終チャットに記録。
2. **Q2** 実resolver/location/hash/signature、ScientificFactory、G7 Bits委譲、元Evidence/Result/angle/F01、worker/Store/Bundleと保存監査を実装。実loaderは0call、live互換性はNOT_ESTABLISHED。
3. **Q3** Canonical3のG7＋versioned LegacyEndpoint監査/Canonical angleと、Full A G7／B F3 sessionを分離。元Evidence・group/raw/child・Full packet/query・typed failureを保持。人工routeはPASS。
4. **Q4** 512 calls、別construction/audit/angle/query予算、seed/domain/prefix、fixed16 coverage、追加0、1GiB/64MiB、retry0/STOPは整合。3200draw案の静的仮定と内部call未確立を明記。
5. **Q5** manifestテンプレート、Run/scope両方のone-shot、spent、STOP、UNKNOWNと二重開始拒否を人工検証。外部認証/鍵/失効serverの開発は不要。実permissionは0。
6. **Q6** Source/input/runtime/worker/seed/plan/bytes/output/STOPをpreflightで結合。親/spawnでfake approval/forged permit拒否、activation IDはNone、science gate閉鎖。
7. **Q7** Global scheduler/spawn/F01/Storage V2/receipt/caller/monitoringと新Storeを人工通し検証。正常512原EvidenceとCOMPLETED、故障時STOP/UNKNOWN非返却を確認。
8. **Q8** 最終tests `{count}`、mutation20/20、integration10/10 PASS。未解決の人工技術不整合0。旧大規模人工/独立checksは再実施していない。
9. **Q9** 固定候補/manifest/予算/出力先/approval手順と実装を準備済み。残るのは正式採択、activation-only SOURCE/記録の固定とその正確な対象へのユーザー明示実行承認、および承認後にしか確認できない実API互換性。
10. **Q10** Core科学import/load/call・science bits・実512・実permission/claim/store・pilot/Full-N・native・再解析・旧grant再利用・STOP解除・production有効化・AI subagentは全て0。既存証拠/保護worktreeを保全しmandatory STOP。次の担当はユーザー・GPT。
'''
    }
    for name,text in pages.items():(HERE/name).write_text(text+footer)
    print(json.dumps(dict(measured_SOURCE=source,source_pins=len(c['source_pins']),input_pins=len(c['input_pins']),tests=count,guard=G.observation())))


if __name__=='__main__':build(sys.argv[1],sys.argv[2])
