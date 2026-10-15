# Track A：Native DFモデルA/BとH6/H8保存metadataの接続監査

**判定：保存されたRTE eventの記述は接続可能。ただし全H6/H8回路の理論費用・期待費用を計算できるmetadata一式は未完成。**
Gaussian操作記述の順序/hashを確認し、schema adapterと20人工testsを追加した。実H6/H8の理論ゲート数、期待費用、総費用、PR/PF比、候補順位は計算していない。
実nativeは「PF部分は再利用なし、RTE部分はoccurrence内の隣接basis再利用」という混合政策である。モデルBのPF相殺を実装済みcompiler最適化と認定しない。

[軽量証跡入口](../../artifacts/resource_applicability/track_a_native_df_metadata_binding/2026-10-16/README.md)／[adapter](../../src/trotterlib/native_df_metadata_binding.py)／[人工tests](../../tests/test_native_df_metadata_binding.py)。
主担当による技術確認であり、別担当の独立レビュー・GPT/Claude科学レビューではない。進行中のClaude Level 1レビューの採否を代行しない。

## 1. 固定来歴と作業範囲

Repository `HIROMU1015/Partially-Randomized-Trotter`。
起点 `track-a-native-df-theoretical-qiskit-validation-20261016` のorigin SHAは
`d2431b8c2a385af20a3d996cf8074b7a826a6e6b` と一致した。
新branch `track-a-native-df-h6-h8-metadata-binding-20261016` を同SHAから独立worktreeへ作成した。
科学SOURCEや結果branchのmerge/cherry-pickは0。

|役割|固定commitと主要path|
|---|---|
|理論モデル・旧人工42検証|[d2431b8c…:接続報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d2431b8c2a385af20a3d996cf8074b7a826a6e6b/docs/research/track_a_native_df_qiskit_connection_validation_20261016.md)、同commit `src/trotterlib/native_df_gate_model.py`|
|H6 native科学provider|[87b6b9f4…:native v5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87b6b9f46b7e8ba9abe132ee21478c114b6f46f7/src/trottertracks/resource_applicability/ax2b_native_df_v5.py)、同commit `h6_matched_port_v1.py`|
|H6確認32の公開状態|[321cd8c3…:独立確認結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/research/track_a_h6_independent_confirmation_result_v1.md)、後続D0とB0I資料のprovenanceも参照|
|H6 D0|[8dbdb682…:保存値解析](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dbdb68293c988a999645e21a82b53966f9ef6cd/docs/research/track_a_d0_saved_value_reanalysis_result_20261014.md)|
|H6 B0I結果|[34efc914…:結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34efc914b2f485ec694c7d6016025d5c76829be6/docs/research/track_a_h6_native_b0i_full_family_results_20261015.md)。execution `ac6d28e35cf01398f169ce59577182a38da9b959`、旧pilot `2e733d8a683d415792ce92a56c7d149b78f42716` を別記|
|H8元PR結果|[681ec23c…:結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/docs/research/track_a_h8_resource_map_result_v1.md)|
|H8 D0|[3d910ee1…:保存値解析](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_saved_value_result_20261015.md)|
|H8 native科学provider|[c31ef424…:native v5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c31ef424175c02379eafebe8fc6c7548c1d9aabe/src/trottertracks/resource_applicability/ax2b_native_df_v5.py)、同commit `h8_resource_map_port_v1.py`, `h8_fast_cost_port_v1.py`|
|H8 B0I結果|[5f79257f…:結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_h8_native_b0i_full_family_results_20261016.md)。execution `54ed98c5fa5b7190f77255271f4e3bec8db15630` とproviderを区別|

各参照のpath/bytes/SHAは[固定参照receipt](../../artifacts/resource_applicability/track_a_native_df_metadata_binding/2026-10-16/fixed_reference_receipts_v1.json)に保存した。
`df_rte_{qiskit,circuit,tail}.py`, `rte.py`, `df_partial_s2.py`, `df_trotter/ops.py` の6共通ファイルは、d243・H6 provider・H8 providerの間で実bytesが一致した。[比較記録](../../artifacts/resource_applicability/track_a_native_df_metadata_binding/2026-10-16/common_library_byte_comparison_v1.json)。provider全closure・全venvを再hashしたという意味ではない。

共通指示は[server原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b78e5f39917d92d6c60694d3e0b0d06394d7eb5/docs/research/provenance/research_project_instructions_server_original_20261012.md)を適用。原本・AGENTSのサブエージェント禁止を維持した。

## 2. 系と取得範囲

|項目|H6|H8|
|---|---|---|
|固定分子|linear、1.00 Å、STO-3G、12 spin modes、Na=Nb=3、sector400|linear、1.00 Å、STO-3G、16 spin modes、Na=Nb=4、sector4900|
|DF / order|保存rank19、signed generation order|保存rank28、signed generation order|
|時間・prefix|T=.8、prefix5/10/15、B0I q1/2/4/8|T=.8、prefix7/14/21、B0I q1/2/4/8|
|独立確認|union8 ×32 trajectory|union6 ×32 trajectory|
|探索|探索8。確認32とは別集合|探索8。確認32とは別集合|
|今回読取したPR trajectory|探索K2/K4各1、確認K2 1|探索K2/K4各1、確認K2 1|
|今回読取したB0I metadata|p10/q1 screen・軸cost・contract receipt|p14/q1 screen・軸cost・contract receipt|

保存trajectory計6件について、event digest＝wrapper event digest、candidate/seed/phase、q個のoccurrence・各r events・R=qr、application順序、Gaussian registry記述hashを照合した。
**全group・全basis・全candidateの走査ではない。** 保存数値costを読み直して集計する処理も行っていない。
原本bytes/SHA・JSON field一覧・代表candidate/SOURCE・照合結果を[saved metadata audit](../../artifacts/resource_applicability/track_a_native_df_metadata_binding/2026-10-16/saved_metadata_audit_v1.json)に保存した。private原本の絶対path対応はhome内のprivate locatorに保持し、raw trajectoryやruntimeをcommitしない。

H6探索の確認した2件はexecution `54986a9f54515c8c2289e38fa16fd81e2b6008c2`、H6確認の1件は87b6b9f4…だった。H8の今回の3件はc31ef424…だった。これは各保存 `worker_claim.json` のSOURCEであり、公開報告commitやseed/data SOURCEと同義ではない。H8の既存完了報告にも複数SOURCE由来groupが含まれる。未走査groupをc31ef424…一色に置換しない。
H6確認の固定報告は、元159 groupのexecution `73018cbcc39803890974c7ee51401063e6c8daf6` と新97 groupのexecution87b6b9f4…、scientific SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12` を分けている。この既存provenanceも保持し、今回の代表1件だけで全256 groupのSOURCEを置換しない。

## 3. A：Gaussian操作の接続

|必要metadata|H6|H8|判定・限界|
|---|---|---|---|
|gate種類・順序・qubit順|代表eventの`application_sequence[].basis_change_operations[]`に存在|同じschemaで存在|観測gateは`rz`, `xx_plus_yy`。二qubitの順序は保存どおり保持し、sortしない|
|numeric parameter|decimal string配列、zero RZも保存|同左|float化可能なfinite値を検査。文字列のままoriginal registry digestへ入れる|
|basis ID/hash|applicationごとに保存|同左|IDとhashの両方を照合。同hashからID aliasの同一性を推測しない|
|局所行列hash|operationごとに`matrix_sha256`|同左|hash記述の整合のみ。局所行列の再生成・再hashはしていない|
|forward/inverse|SOURCEで逆順＋各gate.inverse、forwardは原順|同左|実行済み全inverse列は保存されていない。符号・順序規約はSOURCEに追跡可能|
|PF blockの全basis列|今回のblueprint/summaryはhash・block orderまで|recipe/observed preparation hashまで|完全なPF registryの直接入力は未完成。RTEで出現したbasisの存在から全PF basisを推測しない|
|opaque/custom|代表eventには現れない|同左|全runで存在しないとは未認定。未対応gateはadapterが拒否|

原SOURCEは `df_trotter/ops.py:U_to_qiskit_ops_jw` の `BogoliubovTransform(U.T)` → `_normalize_u_ops` でgate列を抽出する。import失敗時にはdense fallbackもあるため、SOURCE名だけでstructured経路を断定しない。今回の実event記述はstructured gateを示す。
`_normalize_u_ops` はcircuitの独立global_phaseを取り込まずgate列を使う。従ってadapterの `native_gate_list_only` は**実nativeが使ったgate列のSOURCE規約**で、元single-particle Uのvacuum phaseを今回新しく認定したものではない。Gaussianのcontrol化やFock liftの新保証は範囲外。

元registry hashは `df_rte_tail.py:canonical_basis_hash` の
`{num_system_qubits, operations, hash_policy: ordered_local_basis_operations_v1}` をsorted compact JSONでhashする。operationにはgate名・ordered qubits・parameter文字列・局所行列hashが含まれる。
一方、理論モデル `Basis.operation_digest` はfloat.hexを使う別形式である。**両digestを同一視しない。** `Basis.source_hash`へ元registry hashを渡す前の照合責任を新adapterで明示した。

`basis_from_runtime` は `(gate.name/gate.params, ordered qubits)` を受ける。保存JSONはQiskit Gate objectではないので、schema確認済みの記述objectへ変換する小adapterが必要となる。新adapterは記述objectだけを作り、Qiskit Gate・回路・局所行列を生成しない。実eventはschema/hash検査まで、元model adapterへの接続は人工fixtureだけで行った。

## 4. B：DF block・controlled evolution

以下のSOURCE規約は6共通ファイルのbyte一致と固定native v5から確認できる。実H6/H8のprimitive数値列を新たに生成したという意味ではない。

|構成|SOURCE対応|保存状態・接続条件|
|---|---|---|
|one-body|`one_body_diagonal_primitives` → `DiagonalEvolutionPrimitives(rz, rzz=(), global_phase)`|PFの実eigenvalues/RZ列そのものは確認したsummaryに未収録。full primitive receiptが必要|
|DF squared|`df_squared_diagonal_primitives`、lambda/etaからRZ/RZZとphaseを生成|実lambda/etaから再評価することは今回は禁止。primitive metadata未取得として残す|
|explicit zero|producerはzero RZをprune。appenderは渡されたRZ entryをzeroでも保持。RZZ zeroはappend時skip|d243のzero RZ修正を維持。Gaussianの保存zero RZとdiagonal producerによるpruneを混同しない|
|普通制御|RZ→CRZ、RZZ→CRZZ、global phase→ancilla P|ancilla relative phaseを落とせない。SOURCE対応は既存modelと一致|
|symmetric-directional|forwardはuncontrolled、reverseはdirectional。RZZ parity/CX、負angle RZ、RZ(2 global_phase)|普通制御とはraw列が異なる。policyを同一候補のbindingへ必須化|
|Gaussian sandwich|inverse G、diagonal、forward G|各PF blockで閉じ、native v5に隣接block cancellationはない|
|scalar|native終了時P(-scalar*T)を一度|PRはconstant＋extracted identity。B0はconstantのみ、B0Iはconstant＋discarded full-Fock identity。一stepごとへ重複しない|
|wrapper|H/U/H（cosine）、H/U/Sdg/H（sine）、測定あり|state preparationはcost範囲外。theory X/Yと保存cosine/sineの写像を明示する|

prefixは元DF fragment index `[0, prefix)` で、one-body correctionは別blockである。`_prepare` は明示順序partitionを使い、`_prepare_discard` はprefix以外のfragmentを除く。元signed generation orderをweight順へ並べ直さない。

|系列|構成差と保存上の扱い|
|---|---|
|B0|prefix deterministic S2、tail discard。保存trajectoryのevents/lawはnull|
|B0+I|同prefix回路にdiscarded tailのfull-Fock scalarを最終ancilla phaseへ一度加える。保存screenにscalar/signal/allowance、軸costにfingerprintがある|
|full PF|全rankのS2またはglobal Yoshida S4。S4の負時間piece/順序を維持する|
|PR/B2|prefix half → r-event closed tail occurrence → reverse prefix half、q回。尾部は普通制御、prefixは指定control policy|
|B3等|一般builderで表現可能でも今回H6/H8の登録比較への存在は推測しない。未登録系列へ拡張しない|

H6のcurrent-profile blueprintとlegacy preparation hashは異なり、元報告が明示する `legacy_bit_identity_claimed=false` を保持する。H8のsemantic recipe SHAはGaussian操作列hashやnumeric preparation hashの代用にならない。元B0とB0Iのmatched再compileは未取得で、identityだけの純粋なcompiled差を認定しない。

## 5. C：有限RTE law・tau・event

SOURCE追跡先：providerの `ax2b_molecular_ports_v3.py:sampled_steps`、`h6_matched_port_v1.py:cost_task`、H8の `h8_fast_cost_port_v1.py:cost_task`、共通 `rte.py`, `df_rte_tail.py`, `df_partial_s2.py`。

|項目|保存・SOURCEで確認できたこと|残る条件|
|---|---|---|
|tau|outer δ=T/q、occurrence内r events。native tau=`exact_rte_lambda_r*δ/r`、保存distributionは`dimensionless_step_time`|model入力tauは同じ無次元event刻み。T/qそのもの、ranking proxy lambda、Rではない別rとの取り違えを拒否する。今回lambdaから数値再評価はしない|
|q/r/K/R|task cellに保存、代表6件でR=qrとevent dimensions一致|Kは最大even Taylor order。実ordersはK2→[0,2]、K4→[0,2,4]。今回K0-only候補は取得していない|
|order law|`finite_distribution`にorders/probabilities/unnormalized weights/B_K/tau/K|観測次数頻度からlawを推定しない。未観測k2/k4もlawから削除しない|
|component law|SOURCEはabs(coefficient)/exact tail L1。samplerはcomponent確率を正規化してiid draw|保存eventは選ばれたcomponentのID/sign/abs/support/basis記述のみ。全未抽出componentを含むcatalog・完全な確率分布は未取得|
|application順|product k個の後にrotation1個。indexとID列を照合|Hamiltonian積の数式上の表示順と実gate application順を混同しない|
|signとphase|event phase=(-1)^(k/2)、product sign、rotation coefficient signとatan角を別保存|phase JSON `{real,imag}`の解釈adapterが必要。event phaseやscalarを消さない|
|identity|empty support/no Gaussian。SOURCEのextract_identity_phaseとidentity application経路を区別|一般identity rotationを含む期待phase費用は既存の**0〜1 P/occurrence**幅を維持。今回新保証なし|
|normalization|B_Kはevent単位、r回でoccurrence、q回でtrajectory。signal会計のlog_B・normalized signal・allowanceに接続|quantum wrapper gate費用へB_Kを直接gateとして足さない。shot会計と分離|

モデルの`canonical_order_probabilities`はlog-weightのbinary64評価をFractionへ再正規化する。native保存lawは元producerのbinary64 weights/B_Kで、samplerにもcomponent sumの正規化がある。数式の対応は確認できるが、異なる評価順によるbit差を同一lawの厳密証明としない。将来は保存binary64 law・正規化規則・component catalogを固定し、supplied Fraction lawの意味と小確率underflowを検査する必要がある。今回その確率評価・期待値DPは呼んでいない。

`event_probability`と`event_coefficient`は別で、realized eventだけを集めたファイルは全有限分布ではない。seedだけからmissing eventを再samplingしない。探索8と確認32、独立確認union8(H6)/union6(H8)を別scopeとし、両軸は同じevent_digest/seed/replica/SOURCEへ結合する。

## 6. D：A/Bと実basis reuseの判定

分類は「SOURCE上の構成対応」と「保存metadataによる実適用可否」を分ける。`DIRECTLY_COMPATIBLE`を全実入力へ無条件に付けられる系列は今回ない。

|実系列|H6：A / B|H8：A / B|適用上の意味|
|---|---|---|---|
|B0 prefix S2|A：`METADATA_MISSING`（構成対応）／B：`COMPATIBLE_WITH_DECLARED_LIMITS`＋欠測|同左|Aはnative PFのno reuse。Bは仮想の隣接相殺構成で、実施済み最適化ではない|
|B0I prefix S2|A：`METADATA_MISSING`／B：`COMPATIBLE_WITH_DECLARED_LIMITS`＋欠測|同左|scalar receiptはあるがfull block primitive/registry一式なし。B0と別identity|
|full S2/S4|A：`METADATA_MISSING`／B：`COMPATIBLE_WITH_DECLARED_LIMITS`＋欠測|同左|Aはliteral schedule対応。Bはpiece/step境界も含む数学的相殺を許す別構成|
|保存registered-full-basis RTE occurrence|A：`COMPATIBLE_WITH_DECLARED_LIMITS`／B：`ADAPTER_REQUIRED`|同左|native default reuse=trueのtailはBに構成対応。Aはreuseを無効化した比較用構成。代表eventのschema/hashは確認済み|
|PR全wrapper|A/Bとも`ADAPTER_REQUIRED`＋`METADATA_MISSING`|同左|literal実nativeはA-prefix＋B-tailの混合。全体を単にA又はBと同一視しない|
|PRの全law期待値|`METADATA_MISSING`|同左|sampled basisだけでは全component遷移を評価できない。現DPを実行しない|
|support-restricted/history依存plan|`MODEL_EXTENSION_REQUIRED`（今回の経路では使用を確認せず）|同左|一般libraryには能力がある。今回SOURCEのrequestはplanを供給せずregistered full basis default。仮に発見したら履歴状態/certificateの別対応が必要|

native `build_sequence` はactive `(basis_id,basis_hash)` が同じ場合にのみreuseする。identity applicationはphase0でもactive basisを閉じる。sequence終了も閉じるため、q個のclosed tail occurrenceをflattenして跨いだ相殺はしない。
PF v5はblockごとに閉じる。モデルBはPF schedule内の隣接sandwichで相殺し得るが、`partial_prefix_circuit` はtail/step境界でprefix halfを閉じる。Bのこの宣言範囲を拡大しない。
同じ行列、近いparameter、異なるbasis ID、別plan certificateを「同じ操作列」と扱わない。SOURCEのhash文字列一致と局所行列意味論の認証も別段階である。

## 7. E：compiled費用へ接続するためのbinding

将来の `Σ_axis N_axis E[C_theory,axis]` と `Σ_axis N_axis mean(C_compiled,axis)` は、以下のidentityが揃ったときだけ比較対象となる。今回両式を評価していない。

|field|既存所在|今回の判定|
|---|---|---|
|H/DF/state/input bytes|frozen input receipts、B0I contract binding、元PR manifest|H6/H8別hashを[既存binding転記](../../artifacts/resource_applicability/track_a_native_df_metadata_binding/2026-10-16/binding_inventory_v1.json)に保存。NPZを開かずreceiptを使用|
|execution / dataset / model SOURCE|worker_claim、task/checkpoint、seed binding、model fixed commit|別fieldで保持。単一の「SOURCE」へ潰さない。確認したgroupだけのSOURCEで全結果を再ラベルしない|
|candidate / T/q/r/K/prefix|task cell、plan、D0 candidate accounting|名前だけでjoinせず元fragment order・method・formulaも保持|
|axis/control/ancilla|wrapper row、native v5、compiler-independent fingerprint|cosine↔real/X、sine↔imag/Y、ancilla=n。state prep除外・measurement含有scopeを揃える|
|basis/decomposition/compiler/profile|registry hash、preparation hash、numeric wrapper fingerprint、compiler hash、profile receipt|theory固定分解`explicit_rz_sx_x_cx_v1`とoptimization1 transpileは別操作。旧compiler output完全同一性は未検証|
|epsilon/normalization/u/shots|H6 D0、H8 D0、B0I screenと既存accounting|signal εでありenergy精度ではない。factor1/10・経験的u・shot十分条件を保持。新shot算出0|
|sample aggregation|exploration8 / confirmation32、union IDs、paired axes/replica/seed/event digest|theoryはlaw期待値、compiledは有限標本平均。compiled母平均認定ではない。rare event未網羅を残す|
|欠測・reuse origin|H6旧pilot6軸＋新12軸、H8必要16軸、未測定不適格軸のstatus|測定なしを0へ補完しない。旧SOURCEの数値を新SOURCEの再compileと表示しない|

両B0I契約に記録されたcompiler hashは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、profile digestは`622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`。ここでは既存receiptの記録値を転記しただけで、全profileを再監査したものではない。
H8 raw comparisonの歴史的field名`confirmed_PR_union8`をH6 union8の証拠にしない。固定H8 D0/公開projectionはunion6/32のscopeで、旧rawを変更していない。

## 8. Adapter・人工検査

`native_df_metadata_binding.py`はstdlibのみ。`runtime_basis_metadata`は元registryのordered-description hash・期待ID・期待hash・gate集合・qubit/parameter・phase/control規約を検査し、既存`basis_from_runtime`へ渡せる記述objectを返す。
`validate_connection_identity`はSOURCE/input/profile/candidate/formula/fragment order・primitive・basis registryのreceipt/control state/axis/law receipt/tau semantics/normalization/shot/u/sample/reuse/occurrence scopeを、**独立に固定したexpected envelope**と型を含めて比較する。expectedへcandidateをそのままコピーするだけでは実証拠の認証にならない。

新schemaは将来の完全bindingに必要なfieldを定義したもので、既存runが既にそのschemaを満たすという主張ではない。law receiptの内容、local matrix hashとgate実体、tauの数値導出、全componentの完全性は別検査として残す。結果は `SCHEMA_ONLY_VALIDATED` で、常に`cost_evaluated=false`、`receipt_contents_certified=false`、`runtime_authorization=false`。

20人工tests PASS、failure/error/skip0。SOURCE/input/profile不一致、ID alias、basis操作列/hash不一致、gate/parameter、ordered qubits、missing phase/axis、tau/K/component law、R=qr、sample混合、reuse、support/history/flattening、モデル範囲外を拒否した。人工2qubit metadataから既存model adapterへ受け渡す確認も費用関数を呼ばず実施した。
旧人工42件・旧benchmark・科学回帰campaignは繰り返していない。

既存system Python3.12.3、単一test process/thread1、Python -P -B。AS256MiB・CPU60秒・wall60秒・出力2MiBの限定試験計画を実行前保存した。実wall約0.0843秒、子process peak RSS約20.5MiB。[試験結果](../../artifacts/resource_applicability/track_a_native_df_metadata_binding/2026-10-16/test_result_v1.json)。Qiskit/NumPy/SciPy/CuPy等の科学import0。既存科学venvは無変更で、今回のstdlib試験環境をproduction compiler profileへ置換していない。

## 9. 欠測分類と次に必要なもの

分類：1=公開済み、2=private保存あり、3=SOURCE規約、4=保存値から追加数値処理で復元、5=新しいprepared metadata/科学・回路生成が必要、6=現モデルでは未表現。

|不足/所在|分類|今回の処置・将来の条件|
|---|---|---|
|公開result/compiled row/D0 accounting/provenance|1|固定commit/pathへリンク。再会計・重複公開しない|
|今回読んだ6trajectoryのevent/basis記述、元task/worker/wrapper|2|bytes/SHAとfield索引のみ公開。private rawは保全|
|PF schedule/control/scalar/fragment order/phase/normalization規約|3|固定SOURCEへ接続。数値gate列の存在証明に置き換えない|
|既存event JSONの記述形式→model metadata、phase JSON→実数sign、異なるSOURCEを保ったrecord join|2/3＋adapter|schema部分を実装。全候補のmetadata joinは未実施。欠測を自動補完しない|
|binary64 probabilities→明示Fraction lawの正規化/underflow検査|4|元lawが完全に入手された後の追加数値処理。今回未実行、期待値計算の別承認へ含める|
|全PF basis/diagonal primitiveリスト、全tail component law|確認範囲で未取得。元配列からなら4/5|既存の完全preparation exportが見つかればまず受領する。見つからない場合、保存配列のdiagonalization/係数展開/Gaussian再生成等が必要となる。今回は行わず、別scope/承認で固定receiptを作る|
|gate semanticsとsaved local matrix hashの数値照合、全basis網羅|未検証。既存記述があれば4、欠ければ5|今回のhash整合から全行列同一性を認定しない|
|opaque/custom/controlled Gaussian、support/history依存basis plan|6|model拡張・certificate・科学的意味の確認が必要。今回のregistered-full経路とは分離|

「未取得」は全private storageを網羅して不存在を証明した意味ではない。確認したfixed public preparation exportはH4用で、H6/H8 full registryを代用しなかった。今後既存exportが提供されれば、新生成より先にSOURCE/bytes/SHAを照合する。

**モデル変更なしで接続できる範囲**は、supported Gaussianの保存記述、既存producerから渡された完全diagonal primitive、明示registered-full RTE event、ordinary/symmetric-directional、閉じたoccurrence、明示scalar/axisの宣言範囲である。今回実値で確認したのは代表eventのschema/hashまでで、全実分子費用の実行準備完了ではない。

次に必要なのは、(a)欠測full preparation/law exportの受領又は別途承認された最小限の再生成、(b)全candidateのSOURCE/input/axis/law/shot binding、(c)A/Bという比較構成とliteral nativeの混合政策を区別した費用評価scope、(d)実H6/H8理論費用・期待値計算への明示実行承認である。科学的採択・中心解釈はClaude/GPTレビューと利用者判断へ残す。ここで新しい数学的保証や研究GOは発行しない。

## 10. 保全・公開・STOP

新sourceはschema adapterだけで、元`native_df_gate_model.py`と科学builderの意味・bytesを変更していない。既存rootの958 dirty/untracked entries、実行worktree、raw/result/grant/ledger/venvを保全した。新作業・log・tmpはhome配下だけ。
公開は新adapter/tests/runner・本書・索引/日付付きnote・軽量hash/metadata監査資料に限定する。NPZ・raw event/実runtime・profile全体・private locator・認証/内部接続情報は含めない。
新SCF/DF/state/signal/reference/trajectory/native build/compile/transpile、実H6/H8理論費用・新再会計・新比/順位、新grant、原稿改訂、サブエージェントはすべて0。
commit後にnon-force pushを試し、成否を区別して報告する。公開または認証によるpush待ち報告の後、**mandatory STOP**。次の科学計算・詳細レビューは自動開始しない。


## 後続追補：H6 archiveの既存保存を発見

[追加受領報告](track_a_native_df_h6_h8_metadata_completion_20261016.md)で、H6の7 preparation・完全component law・Gaussian列を取得した。本書のH6「未取得」はこの追補で更新する。H8 full metadataとprimitive数値列生成・理論費用評価は未実行。旧監査時点の記録は保全する。
