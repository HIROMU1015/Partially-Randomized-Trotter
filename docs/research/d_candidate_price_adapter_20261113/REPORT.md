# PR候補依存context・compiled価格取得adapter

2026-11-13 JST。**候補ごとにprepare/head/tail/signed scheduleを構築し、既存compiled_price_fastの価格取得ループへ渡すadapterを追加した。** C1のsystemだけの固定contextを他候補へ使い回さない。候補生成・signal/N評価・最適化・平均価格/信頼区間の認定は追加していない。

[接続HANDOFF・使用法](HANDOFF.md)／[初回の有限取得計画](FIRST_ACQUISITION_PLAN.md)／[SOURCEとテスト](SOURCES.md)／[人工・回帰](TESTS.txt)／[別process残差回帰](RESIDUAL_REGRESSION.txt)。

## 実装と再利用

追加は[src/trottertracks/candidate_price_adapter.py](../../../src/trottertracks/candidate_price_adapter.py)。基点24b0a6eb1の既存D branchを再利用した。A821c・C37cbc・Bcdf6の指定資料/APIと過去の人工harnessを確認した。新しい化学input reader、PF/RTE emitter、fingerprint、compiler、候補探索器、N solverを作っていない。

| API・変更 | 役割 |
|---|---|
| `from_b` | 外部V2 configurationの検証をBへ委譲。physical row/structure/law/profile/axisを対応付け、request/context keyを作る。物理入力・回路は読まない |
| `from_a_job` | A821cのpr_rpe_signal_plan.planが出すjobを、B identity/lawと照合して同じrequestにする。signal numerical_recipeは価格モデルに流用しない |
| `NativeRuntime.context` | callerが復元・照合したHamiltonianからpinned prepare(ham,L_D)を呼ぶ。候補ごとのhead、component/basis/tail、λ_R、scalarを保持 |
| `NativeRuntime.sample` | 明示呼出しだけで元carrier sample_requestsへ委譲。normal finite-RTE sampler、candidate/cohort/index/outer/piece seed規則を変えない |
| `Adapter.acquire` | 明示された1 pathのrequestsを検査し、専有するfresh Backendへcontextを渡す。compose/wrapper/compile/cacheはcompiled_price_fastへ委譲 |
| `pending_price_records` | path結果の存在と平均価格の資格を分け、既存Bのquantity/identity/law仕様でmean=nullの未取得応答を返す |
| compiled_price_fastの小拡張 | `make_backend_class(..., axes=...)`を追加。既定X/Y動作を保持し、明示された単独軸/軸順も取得できる。P1の同価格を根拠とした片軸省略はしない |

## contextと価格の同一性

requestはBのidentity/law_bindingをそのまま保持し、physical candidate、signed weights/allocation、H/state/snapshot/decomposition、split/λの出典、input_source、execution_source、provider/carrier/C1の固定SOURCE、compiler/control/wrapper/loweringをcontext keyへ含める。qRの積だけでkeyを作らない。

preparation cacheは同じinput/binding・split・実装SOURCEのときだけ共有する。full contextはさらにformula・T/q/R/K・law等を区別する。prepare後はH hash、8 system wires、rank、元順のdeterministic/randomized fragment、λ_R、指定済みならpreparation hashを照合する。resolved_contextには実preparation/tail hash、scalar、λ_R、fragment一覧を保存する。tail hashはcomponentの内容・順序・basisとのbindingを保持する。callerの復元済みHamiltonian・contextは実行中read-onlyで扱う。

新loweringは既存`efficient_controlled_zz`を一時有効化し、例外時も復元する。制御ZZ以外のbuilderを変更しない。原scalar `ham.constant + prep.extracted_identity_coefficient`を元composeでtotal Tに一度だけ付け、測定付きfull Hadamard X/Yを個別取得する。q outer→signed piece→元event順を維持する。

Sは同じevent pathを元Accountingで計数する。headのhalf-timeはnativeと同じ `(T * signed_weight / q) / 2` から作る。固定C1のdyadic gridでは元`delta * weight / 2`との一致も人工確認した。m5等の非dyadic算術で不要な割り直しを入れず、PF/RTE間の時間を揃える。basis operationsの表現、pending-Zの加算順、途中finish/resetを変えない。

cacheは従来どおり**実wrapper全体のcanonical fingerprint＋compiler hash**を照合する。keyに角度・control・definition・qubit/global phaseを含む既存writerを再利用し、contentの再確認を省略しない。旧cacheへの書込みなし。別candidateでも実回路bytesが同一なら計算cacheを共有できるが、過去path/平均recordのidentityやlawを新候補へ付け替えない。hit元pathもCANDIDATE_PATHに記録する。価格meanの標本数・資格はcache hitで増えない。

## 対応範囲

| 条件 | 今回の対応 |
|---|---|
| system/input | H4と同じ8 system wires、元保存DF-Hとgeneration-order prefix。復元・state/snapshot証拠はcallerの既存loaderに委譲 |
| split | deterministic one-bodyと非空のordered prefix/tail、1≤L_D<rank。L_Dを1に固定しない。人工L_D1/2で独立性を検証 |
| formula | 既存S2 (`2nd`) と既存signed m5 (`4th(m5_best)`)。weights/allocationはpinned carrierのまま |
| finite-K | 元libraryの非負even K。人工ではK=0/2/4/6と各最高order eventを使用。Cartesian gridの科学評価はしていない |
| schedule | 外部から与えたlegal q/R、ABS_TIME_PER_OUTER・pieceごと最低1の元配分。全roundを新規生成せず、必要なm/axisを選んで取得 |
| axis | X→cosine、Y→sine。両軸・単軸・逆順を明示指定。Sは共通prefixの後に両suffixを評価し、要求軸を選ぶ |
| profile | Qiskit1.3.0、NumPy1.26.4、rz/sx/x/cx、opt1、seed17、backend/target/coupling_mapなし、symmetric_directional、full measured wrapper/P0、新ZZ |
| 未対応 | 他wire数、空head/tail、別fragment ordering、Full-H RTE/PF、現Bが拒否するPR標準Y4や未実装formula、別law/sampler/profile。OUTSIDE_DOMAIN又は不一致として停止 |

Bのexact timeとphysical binary64値が一致しない場合、今回は丸め直して採用せず拒否する。source-bound A fixed-gridは元値を保持する。これは現adapterの到達点であり、最終探索空間から未対応候補を除外する科学判断ではない。必要なら別SOURCEの最小拡張として扱い、Y4をm5等へ偽装しない。

## 出力と資格

`CANDIDATE_PATH.json`はpath単位のS/C/D、axis別6指標/fingerprint/cache、seed/order law/event digest、resolved contextとinput/execution SOURCEを持つ。元Backendのevents_PRIVATE/STRUCTURAL/軸別/result/TECHNICAL_TIMINGも保存する。mockの値を実価格として保存する経路はtest harness内だけである。

compiled quantityは`FULL_WRAPPER_PATH_COMPILED_RZ_COUNT`、Sは`FULL_WRAPPER_PATH_STRUCTURAL_RZ_COUNT`。`mean_compiled_price`と`confidence_interval`はnull、`qualified_expectation`はfalse。既存paired_residuals/cohort_summaryを後で使う場合も、同じcontext/law/profile/roleの両軸pathだけを集める。片軸結果を2軸標本へ補完しない。今回のadapterは新しいmean/CI評価入口を作らない。

十分な根拠を持つcompiled mean/区間のBへの供給は、将来の取得scopeでtarget law・normal/forced・rare・統計/数値/model誤差を確定した後に行う。S、event count、outer-windowをcompiler上下界へ昇格しない。outer-windowの長q tailとm12平均価格は引き続きUNKNOWN。normalizationや古典価格path数をC×Nへ追加乗算しない。

## 技術検証・限界

新20項目＋既存fast10/P2 metadata6＝36 PASS、別processの既存残差13 PASS、合計49項目。新testは任意の8-wire行列による人工Hamiltonian、明示構築event、実prepare/native compose/wrapper/fingerprintと**mock compiler**を使う。実H4 inputの読込・samplingを禁止し、元B保存configurationもmetadata互換だけを確認した。

既存fast回帰では小さい人工回路を固定Qiskitでcompileし、元C1とcompiled circuit/6指標が一致することを再確認した。新L_D2/K6等の実H4 compiled価格や科学的平均を確認したものではない。初回の人工mapping契約違反・旧C1属性不足を修正した経緯は[開発記録](DEVELOPMENT_CHECKS.md)に残す。Qiskitの大量の既存deprecation warningsは抑止表示のみで、compiler条件は変えていない。

技術高速化は既存prefix共有・RZ projection・fast writer・preparation/context cacheを再利用する。今回、新しいwall/RSS benchmark・実H4倍率は測定しない。TECHNICAL_TIMINGの価格call時間にはcontext準備/normal samplingが入らないので、将来runではrunnerが全pipeline wall/peak RSSを別記する。

新実H4のsampling/build/compile、保存科学値の再解析、残差fit、新E[C]/Gは0。過去SOURCE・raw・input・KEEP・grant/ledger/marker・STOPは保全し、旧P2の実行契約は変更していない。

**次：A/探索担当**がphysical候補・schedule、**B**がconfigurationと価格要求、**C**が必要なE[S]/feature/law証拠を渡す。**ユーザー**が対象/SOURCE/有限path・compile範囲を承認した後に限り、**D**が価格を取得する。現在はcommit/push・remote照合後mandatory STOP。
