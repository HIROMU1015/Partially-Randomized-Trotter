# Q1–Q10：限定Pilot 16,896件の実行・保全

**LIVE_TECHNICAL_COMPLETE_AWAITING_REVIEW／16,896件完了／保存値検査PASS／mandatory STOP。** 実F01失敗・UNKNOWN・追加samplingは0。科学的GO/NO-GOや方式の一般優位は認定しない。同じprimary Codexの技術検証で、別reviewerによる独立再現ではない。

## Q1 採択・最終SOURCE・承認binding

最終測定SOURCE `c7e8806cf308131b5629ac54b627d46c360799a3`、tree `fc022df06a27a3e68c51dea6f3e7e128e5da0a68`、manifest digest `92ddf84e0d6d90ead7a98dac7eadf69ef9c2bf0622694c8870b69cf841a23bb6`、candidate digest `c33746e2f3d255ea24f1f001e939b289ac308281efb24531e1516ee124047b7e`。科学参照は76ca3c4089b201d55da0fede08e1f77733ac38a5、直接基点f6088e2cc453816b30e985e52b10696830844fa9。

原承認は[user_approval.json](user_approval.json)のCRLFを含む原文。本文SHA256 `74509e124e5b956e70dac2e669e0dd2336b3263731eb0de0d849eb7fa1d456c6`、local content activation `e48ad571606972895e7465e6e8969b3af9a342fcd534080246a5efa62c8f6707`、approval record digest `2c0533d27e4714f313e4c683ffcdc1e884f10ddf20ac3a46cb2b7a00dbcdc665`。外部署名・外部認可serviceではない。[adopted_contract.md](adopted_contract.md)はCodexの具体化として分離した。最終実値は[preflight.json](preflight.json)、[approved_manifest.json](approved_manifest.json)、[source_inventory.json](source_inventory.json)。SOURCE155＋入力10pins、executable/runtime/API signaturesを照合。承認条件との不一致なし。

P5_general_order、p=(1/5,3/10,1/2)、x5/7、m5、sigma+1、root256/H160、eta/rho1e-12、epsilon_axis1/200、primitive1e-6、alpha_axis1/320、familywise1/20。A6 tau1e-6、K_phi32、J32、math-only。新Hamiltonian/DF/native/RZ合成なし。n256は十分性保証ではない。

直前候補は未採択・scope未発行の準備版。今回の明示指示が設計採択と一回実行を承認し、固定実値との一致を実行前にbindingした。使用済み512の個別承認・SOURCE・claimは移転していない。科学開始前preflightの記録用field不足1件は、空scope/科学0のまま保全・修正して最終SOURCEを再固定した（preflight_development_9a0b63a6/）。実行開始後のSOURCE変更・retryはない。

## Q2 実装・再利用・人工受入

新namespace g6_w2c_b3_limited_pilot_live_v1で、6phase/48pair/tasks、16/256 index、採択seed、phase barrier、coverage STOP、global budgets、baseline observer-off/instrumented-on、compact retention、phase-aware checker、planned16896終端、pre-admission outer窓をversioned拡張。既存resolver/Factory/Bits recorder/監査・angle経路、簡易intent+SQLite SPENT、worker F01とschedulerを基礎にした。実science callableを固定resolverで解決した。科学Core6、Exact Adapter、Global Bundle、F01 primitive、Storage V2、旧checkerのfilesは不変。

人工正常16896＋coverage/S angle/ACK UNKNOWNの4integration、最終focused unit/control9件、manifest改変20件、再封印semantic mutation20件を検証。receipt-close専用STOP slotは人工controlで検証。親/spawnの未承認Core loadなし。初回人工のcompact型復元漏れと診断JSON失敗を修正し、rawを保持。最終checkerを人工正常全保存値に再適用した。148/80旧試験の全面再実行・科学確認samplingなし。[人工受入](artificial_acceptance.json)、[mutation](semantic_mutations_final.json)。

## Q3 実call・constructor・draw

|境界|実数|上限|
|---|---:|---:|
|AuditParity_construction|24|24|
|B_angle|8140|8448|
|B_event_requery|8140|8448|
|B_sample_audit|8448|8448|
|F3_prepare|6|6|
|Full_angle|1804|2112|
|Full_angle_packet|1804|2112|
|Full_sample_packet|3900|4224|
|G7_make_generator|42|42|
|bit_draw_requests|41768|105600|
|canonical_angle|6336|6336|
|canonical_snapshot|18|18|
|nested_B_event_witness|8140|8448|
|primary_constructions|48|48|
|scheduled_sampling|16896|16896|
|driver correctness event_witness|480|512|
|F3.prepare内Full constructor（固定経路から導出、primaryへの再加算なし）|6|6|
|additional sampling|0|0|

Boundary、nested B witness、科学内部callは別階層。primary48=make42+prepare6。F3内constructor6を追加primaryとして数えない。Full sample packet3900とangle packet1804を区別。draw41768≤105600、固定m5/raw degree≤4/H160をSOURCE静的照合済み。未知の内部総callを無制限認可と解釈しない。instrumented内部観測は[supplemental_saved_checks.json](supplemental_saved_checks.json)に別列で保存し、全passの総数と混同しない。

## Q4 Result/Event/seed/prefix

新採択namespace PROPOSED_B3_LIMITED_G7_PILOT_20261019_V1、SHA256 counterのnamespace/arm/axis domain。mode/layer/passはseedに含めず、Global IDには含む。全16,896 COMPLETEと48pairのoffset0..16895が一対一で一致。A/B・S/P1・baseline/instrumented・correctnessの対応prefixでkind/classification/event/bit consumptionが一致し、X/Yは別domain。baseline fine=None、instrumented fineあり。全phase barrierとsource/manifest/runtime/worker/session/routeを検査した。[固定checker](saved_evidence_check.json)、[追加保存値join](supplemental_saved_checks.json)。IIDや一般的API互換性の証明ではない。

## Q5 cell・coverage・未観測path

合計accepted16,280、ZERO_R324、coin reject292。correctness512はaccepted480、ZERO_R20、coin reject12。全32 correctness cellでaccepted≥11、coverage成立。S/P1は同じprefixであるため、これらの重複を独立サンプルとは扱わない。

Canonical3方式は全件acceptedで、ZERO/coin分岐を未観測成功とはしない。FullはX/Yともraw-nonreduced/coin-reject/acceptedを観測。correctnessの各mode/layer共通でX=(13,2,1)、Y=(11,3,2)（accepted,ZERO_R,coin）。全96 cellの件数と未観測pathを[cost_tables.md](cost_tables.md)、[cost_and_outcomes.json](cost_and_outcomes.json)へ保存。coverage不足を補うsamplingなし。

## Q6 Evidence・Codec・保存・metadata

Full original Evidence1,536件、compact15,360件。correctness全件とprofile各axis先頭16をfull保存。残りも固定workerでfull Evidence生成・Exact Codec/不変条件/監査/angle検査後、original digest・exact event・Result・bits・分類・traceを保持した。compactは原文全体を保持しないため、offlineでoriginal全文を復元・独立再検証できるとは主張しない。保存digestの存在・形式・chain/file SHAと保持されたjoinを照合する。

Receipt/delivery/caller、全COMPLETED state、claim spent、STOPのhash join PASS。Storage V2のstorage/state/trace errorsなし。実科学F01失敗は発生しておらず、一般failure保証は未確立。人工F01/UNKNOWN/保存障害だけを別証拠とする。

Actual metadataはartificial_only=false、execution_type=USER_APPROVED_LIVE、approval_mode=USER_SIMPLE_ONE_SHOT、external_authority_used=false。legacy real_authority=falseは外部認可不使用の意味。parent_guard.artificial_only=trueは親のimport guard局所観測で、実行modeではない。親Core loadなし、workerは承認/SPENT後に固定Coreをload。live_API_compatibility=NOT_ESTABLISHED、science_correctness_certified=falseを保持。

全scope単一cap Result64GiB/Log256MiB。final予算snapshotは終端slot書込前のviewで、予約未観測12,664,832bytesをUNKNOWN scientific trialと解釈しない。後続witness実file12,660,736bytesを照合し、post-witness result209,513,726bytes、log41,737,308bytes、未使用emergency予約4,096bytesを無返却で保持。[storage_reconciliation.json](storage_reconciliation.json)。物理原本76files/251,288,923bytes。容量保証への一般化なし。

## Q7 CPU/wall・内訳

Global outer（pre-admission→全join/finalization/STOP、outer export自身除外）はCPU **1,258.033384573秒**、wall **792.156549420秒**。parent own159.135530573秒＋reaped child hierarchy1,098.897854000秒。後者はGit/helperを含む。worker END own CPU合計1,020.444851161秒は別観測で、再加算しない。resource tracker等の未回収CPUはnull、archive/offline集計は窓外。

|条件/phase/pass|calls|worker pair CPU s|phase execution wall s|coarse setup CPU s|axis online CPU s|
|---|---:|---:|---:|---:|---:|
|S correctness/baseline|256|13.750403|25.391922|4.738200|8.485730|
|S profile/baseline|4096|145.049685|161.806727|4.947334|138.585008|
|S profile/instrumented|4096|330.674752|348.154103|8.159938|315.581605|
|P1 correctness/baseline|256|14.065385|4.310520|4.778403|8.746711|
|P1 profile/baseline|4096|149.919880|30.115602|4.878957|143.444909|
|P1 profile/instrumented|4096|361.691999|71.686146|7.809742|346.225063|

Phase wallは保存PHASE_BEGIN timestamp→join後resource sample、後続phase offline検査を含まない。worker pair CPUはworker計測窓内、parent/Git helperを含むglobal CPUとは別。P1 pair wall和をglobal wallとしない。共有parent/helperを条件・arm・axisへ恣意配賦しない。

方式/layer/axis/pass/condition別online CPU/wallとblockは[cost_tables.md](cost_tables.md)。pair setup/online/stage/RPC/serialization/hash/write/flush、memory/RSS/HWM/I/Oは[cost_and_outcomes.json](cost_and_outcomes.json)。law/child/rational/audit/angle/Codecのinstrumented関数診断は[fine_cost_buckets.json](fine_cost_buckets.json)と元fine_records。selected callbackのexclusive partitionで、coarseへ加算しない。Observer callback固有CPU、純粋IPC、STOP/close単独CPU等の未分離費用はnull/residual。正常時stop_to_join等がnullであることを0へ補完しない。outerはこれらを含むが、完全な因果的分解ではない。

## Q8 Block・clock・比較限界

16/64/256の累積prefixと非重複blockのmin/median/max/sumを全profile cellに保持。16,64,256は同じprefixから得た記述値で、独立反復・母平均CIではない。[clock_calibration.json](clock_calibration.json)ではreported1ns、人工loopのpositive minはCPU320ns/wall79ns、medianCPU340ns/wall90ns。これはloop/call負担を含み時計単体誤差の厳密上界ではない。

S→P1順序固定、condition一回、各worker fresh process・warm filesystem/cache、共有host負荷/I/O/GC/observer/guard/bit長差による交絡を保持。資源観測は各phase・outerに保存。samplingしたRSSの共有page重複を含むpeak357,851,136bytesで、連続・unique peakではない。恣意的CPU/RAM/wall上限や共有設定変更なし。S serial/P1既存8workerは比較条件。量子confidenceをCPUへ転用せず、一般方式順位・Full-N費用へ外挿しない。

## Q9 最終状態・保全・公開

Run `B3_LIMITED_PILOT_16896/20261020_V1_db32d387de8fec1e`、claim `6d400313de1c01910cdcd98899aeff032c49775fd12a74a8d85273bd402274f8`。RunCOMPLETED、claimSTOPPED/spent1、COMPLETED16896、FAILED/UNKNOWN/IN_FLIGHT/NOT_STARTED0、retry0/resume false、mandatory STOP。Receipt/STOP snapshotはCOMPLETED state exportより前のSTOPPED履歴を保持し、後続simple_run_stateはCOMPLETED。新規ENTER不可。

[原本inventory](raw_inventory.json)、archive `raw_scientific.tar.gz`（25,051,778bytes、SHA256 `42bcdd40f5cffe69435b6a4778c3f44760bfd8a95c7663dc9660369fa13d2d8b`）、archive roundtrip76files全SHA一致。原本state rootはmanifestに明記し、専用サーバー領域へ保持。SOURCE/input archiveとGit identityは[measured_source_retention.json](measured_source_retention.json)。旧512原本35files/旧archive/旧SOURCE入力pinsは[preservation_after.json](preservation_after.json)で不変。旧claim/grant/STOP、他worktree変更なし。無関係worktreeのHEAD/statusを実行中断条件にしない。

GitHub公開状態・完全remote HEAD・公開bytes/SHAはpublication_receipt.jsonに記録する。公開失敗なら未公開のまま報告し、迂回・公開済み表現をしない。

## Q10 言えること・残ること・次担当

この固定P5/m5/precision/A6・seed・有限prefixで実API接続、意味論/bit/監査/保存の対応、観測path、固定実行pipelineのCPU/wall・費用支配項を記述できる。compact原文再構成、未観測失敗branch、IID、一般正しさ、n256の統計十分性、普遍的性能優位、Full-N総費用・完了、物理RZ/native合成は保証しない。

Full-N/追加pilot/別codecや反復・順序条件の必要性は、これらのpath頻度、block費用変動、setup/observer/guard/保存負担、未分離残差を材料として**ユーザー・GPTが判断**する。Codexは科学的GO/NO-GOを認定せず、追加計算・詳細科学レビューを開始しない。**mandatory STOP。次の担当：ユーザー・GPT。**
