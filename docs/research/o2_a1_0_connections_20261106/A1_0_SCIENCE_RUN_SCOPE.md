# A1-0：後日の科学実行scope

2026-11-06 JST。**未承認の実行仕様。今回の技術接続指示は、以下の数値処理の実行許可ではない。**
前回草案は[固定a2fcbe2の契約](../o2_a1_server1_handoff_20261106/A1_SCIENCE_EXECUTION_CONTRACT_DRAFT.md)、今回の接続根拠・不足は[CONNECTION_STATUS](A1_0_CONNECTION_STATUS.md)を参照する。
同じ調査をさらに繰り返す段階ではなく、次は下記の限定数値処理を許可するか判断する段階である。

## 1. 全処理に共通する固定対象・SOURCE

保存linear H4、隣接1.00 Å、STO-3G、legacy DF rank12、4電子sector、8 system qubits、L_D3。
最終opt2のm0/q1、δ=t=.02、r1/K2、X/Y、canonical unweighted PR、threshold0、元selected/boundary policy。
H_D ground/PF有効固有状態controlへ置換せず、保存DF Hamiltonian全体のphysical-sector groundを用いる。
状態準備費を除外する仮定を維持するが、simulationの古典実費は別に記録する。

| 固定物 | 指定 |
|---|---|
| H4 NPZ | carrier `4591b8d75a2a6a55c74a51225f0c61648023c8ee` の `artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz` |
| snapshot SHA256 | `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` |
| H hash / partition | `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5` / `9034397f5bde1f20cb712578d14245f4dac1236413d34642a75d83d867b52086` |
| opt2 preparation / tail hash | `50c77f5455cc1a3c0710ed3c7588efc957b9cead7436bd0a6d851af61738a4bc` / `c51e35da56f3de1646896339f38da3cb40dcb8405a24d781cef134a654bab642` |
| 原core SOURCE | `6e8d099a5252d3b424045fb22aeb112d5ebad5c9` の既存rte、DF preparation/tail、repeated、selected policy、wrapper/compiler。閲覧carrierと元dirty executionのSOURCE記録は前回HANDOFFに保持 |
| A1推定・費用接口 | 同6e8d099の `src/trottertracks/algorithm_codesign/a1_control_variates.py`、変更なし |
| 今回の薄い接続 | 本3文書とともに公開するcommitの `a1_connections.py`。公開SHAは最終チャットで固定。元SOURCEを置換しない |
| policy固定 | `wp06b_sequence_policy_proxy_bridge_v1.json`、`support_run_le_1`、training fingerprint `9dcdc4a28866dc223afb0ea212df645a3ed4a28e5c6dad6ddfc0ec1fcb28ba9e` |
| price/compiler | Qiskit1.3.0、rz/sx/x/cx、opt2、seed17、backend/coupling/layout/routing指定なし。NumPy choiceは1.26.4 |

新しい科学SOURCEを固定する際は、実際に呼ぶcore/adapter bytes・入力・versionsを記録する。過去のSOURCE commitの名前だけを新実行SOURCEと偽らない。
未近似物理Hへのoperator/state bindingは本scopeに含めず、保証対象を保存DF modelに限定する。

## 2. Q1：threshold0 catalog・実CDF・support・control moment

**目的**：元opt2の実qと既知νを得る。測定・RNG・state対角化は不要。

- 入力：上表NPZ、同じLD3 split、identity_policy=`extract_identity_phase`、明示的coefficient_atol=0.0、t=.02/r1/K2。
- 既存関数：`load_connected_cluster_hamiltonian_snapshot`、`split_df_hamiltonian_by_ld`、`prepare_df_partial_s2`、`_finite_inputs`（または同じ`make_rte_config/finite_rte_distribution`呼出し）、`rte._validate_components`。
- 接続：`connect_sampler_inputs`→`law_from_choice_cdf`→`OccurrenceLaw`→`proposed_control_banks`/`ControlBank.means/joint_mean/pattern_mass`。
- 出力：完全な順序付きcomponent ID/係数/sign/probability/basis/support、λ、identity coefficient、config/distribution/B、choice入力配列・実CDFの可逆binary64表現、53-bit質量、zero/positive support、ν/joint moments、2層/5層の正質量。SOURCE/hash/versionをbindする。
- 検査：既存hash一致、order(0,2)、component順不変、質量非負/和1、CDF境界と格子対応、正層の全被覆、前回3controlのjoint恒等式。a=`df-fragment-3:Z0Z1`、b=`df-fragment-6:Z0`を維持する候補。
- STOP：hash/profile不一致、CDF不正、必要order/a/bの格子質量0、未宣言相関、targetの不明瞭化。別labelへ変更しない。

完全catalog再構成・実科学CDF/normalizer/momentの数値取得が科学callである。今回実行していない。
この処理にHadamard回路・全event列挙・seed replay・2^53格子の総当たりは不要。
既存preparationが基底操作を再構成する範囲はQ1に明示して含めるが、full interrogation build/compileへ入らない。

## 3. Q2：finite-law/numerical biasと、必要な場合だけthreshold橋渡し

**Q2-L（初期finite-target診断に必要）**：Q1の実qと元algebraic finite target `μ_fin^0` の対応を閉じる。
reference側の正規化・order weight・sign/rotationをSOURCEへ結び、categorical/product lawの標準的なL1伝播とbounded response等で `b_law` を取得する。
binary64入力の和が厳密1とは限らないこと、angle/phaseの丸め、selected basis policyの等価性に必要なnumerical項を分離する。
可能ならstateを使わないoperator/range評価を優先し、未知path期待値や真varianceを計算して推定器へ渡す必要はない。

入力はQ1 payloadと同じSOURCE。出力はreference lawの定義、実qとの差の上界、適用条件、numerical allowance、使用する座標誤差予算。
νをexact Fractionでcenterする場合も、実装binding・IIDモデルを含めてcenteringの条件を記録する。λを凍結した後の `Σ|λ|e` はQ5で会計する。
丸めを含む正当な上界・正の残余幅を得られなければ測定前STOP。実qを新targetに読み替えてbiasを0にしない。

**Q2-T（WP04のsignal/PF/RTE/radius保証を移送する場合だけ）**：同じsnapshot、cutoff0と1e-12のpreparationを比較する。
元の `threshold_operator_error_bound` と保存dropped L1を再利用し、random tail、抽出identity phase、λ、B、m0のnormalized P3 operatorの差を伝播する。
必要なら既存 `finite_rte_corrected_operator/finite_rte_attenuation`、sector operator定義、triangle/telescoping評価を限定再利用する。新しいPF係数fit・全grid・PF control固有stateの生成は不要。
出力は追加のthreshold/finite-target誤差と、旧数値保証の移送が可能/条件付き/不可の区別。数値同一性を強要しない。

**最小化の選択肢**：初期評価をthreshold0のfinite meanの座標推定に限り、旧εを事前指定する比較許容幅として採るなら、WP04の物理signal/半径/PF/RTE保証を移送しない設計にできる。
この場合Q2-Tと新しいradius評価を初期scopeから外せる。threshold差を0と認定した意味ではなく、旧保証を使用しないという明示的な範囲指定である。
RPEのenergy/phase保証を要求するならQ2-Tだけでも足りず、正当なρ_lb・branch/alias・全round誤差合成が別に必要であり、今回の初期scopeへ追加しない。

## 4. Q3：simulation用physical ground payloadと限定signal

**目的**：fresh Hadamard outcomeを生成するengineに必要な同じDF sector groundを用意する。
対応する保存state vectorを実際に回収できれば再利用し、fingerprint欄だけでpayloadを取得済みとはしない。
回収できない場合、上表NPZから同じ4電子sectorのDF行列を作り、ground vector一つだけを再構成する範囲を許可する必要がある。

既存の `dense_df_operator_in_sector`、`PhysicalSector`、元validatorと同じ`np.linalg.eigh`/lowest-vector規約を再利用する。
元 `finite_rte_signal_validation` runnerはPF control固有state・複数grid・shot再評価も行うため起動しない。
出力はengine内のstate payload、sector/hamiltonian binding、fingerprintと必要な数値残差。E0/gap/vector/path条件付き期待値をfit・control・配分へ公開しない。
既存state fingerprintとの違いがphase規約等によるかも分け、無断state置換で一致させない。

新しい平均signal/radiusの評価はfinite-mean推定の実行に必須とはしない。必要なbenchmark truthを要求する場合だけ、threshold0/m0/同じstateの複素signalを一つ取得し、その事後診断用途を別欄へ隔離する。
RPE用の厳密半径1をこの数値から発行しない。
STOP：sector/H/state不一致、非有限値、許された数値精度を満たさない状態、measurement engineからのoracle情報漏洩。
状態再構成の許容差と、Q2-Lで採るnumerical allowanceの数値は未固定であり、科学実行前に選ぶ。

## 5. Q4：selected full価格の不足取得

**目的**：量子trialと同じpath・axis・selected policyのfull no-prep価格を取得する。
入力はQ1 preparation/基底定義、同じrequest、元WP06b policyとcompiler。必要なruntime support basisの再構成もこのscopeへ含める。

既存 `register_support_restricted_bases`、`apply_run_threshold_policy`、repeated builder、Hadamard wrapper、`transpile_and_measure_cost` を新 `selected_full_prices` で接続する。
独立片軸trialには `axes=(axis,)` を渡し、不要な反対軸やfull-basis対照を生成・compileしない。
新規価格取得の範囲は、許可されたQ5全stage/全方式の実行trialに対応するselected full path/axisの不足分だけ。全catalog/pathの価格列挙・別q/r/K/δ・native synthesisはしない。

同一actual numeric circuit/profileのcacheがあれば使える。既存 `get_or_transpile` はmiss時にcompileするので、cache-onlyの許可と不足compileの許可を混同しない。
新規cacheは自分専用のfresh出力内。旧shared cacheやrawは変更しない。重複pathはcompileを省けても、量子shot費は全件課金する。
出力はtrial/path/wrapper/axis/profileへのbinding、元6 metrics、hit/miss、古典取得実費。T/native未取得はUNKNOWN。1Qは取得できたgate counts等だけに基づく。

**初期推奨案は実現campaign費用の報告**。期待費用のconfidenceには、有効な価格range・独立/正当な価格取得保証を別に固定する必要がある。
旧8 trajectoryのmean/SEを全stratumの正確な期待価格にせず、旧5% proxy gateをconfidenceへ格上げしない。
STOP：wrong compiler/policy/axis/scope、未知価格の0扱い、予定外build/compile、価格call範囲超過。未知期待費用のまま優位を認定しない。

## 6. Q5：有限train→独立calibration→fresh production

Q1/Q2-L/Q3/Q4の必要条件が閉じ、ユーザーがこの測定処理も明示許可した場合だけ開始する。
以下は前回草案を維持した**未承認の最終候補**であり、今回採択・数値評価していない。

| 条件 | 提案 |
|---|---|
| 比較 | B0 raw、B1-O order2層、B1-OL order/3featureの共通細分5層、A1 3control、X/Y両軸 |
| A1 train | 各axis1024、既存intercept付きOLSのslopesを一回凍結。rank不足はminimum-normのまま記録、追加fit/feature探索なし |
| B0/B1 pilot | B0は不要train=0。B1は各正層/各axis64、有料・独立配分学習 |
| cal | η=1/8、`n_cal=max(2,1+ceil(2 log(1/δ_cal)/η²))`。各変数/層で固定数のfresh独立標本 |
| 誤差 | 保存ε=`.266146579872968`をmodel有限targetの参照許容幅とする候補。未包含law/centering/numerical/native項を控除し正残余を要求 |
| 失敗配分 | 各axis α*=`2.8897922699779505e-08`、4方式へA=α*/4、cal=A/4、prod=A/2、price=A/4。B1は該当cal/prodを全正層へ等分 |
| range | 原weight W=1を固定する候補。CVはD=2W+Σ|λ|、guard D≤4W。sample rangeで狭めず、超過時clipしない |
| prod | 独立calのvariance上界と既存草案Bernstein/Hoeffding十分条件でN/各層配分を決め、fresh trialで一回。各方式/axis総数≤8192。十分Nではなくscope上限 |
| conditional cap | `ceil(2M_stage,h/p_h)`を各stage前に凍結。正層が揃わなければ不完了STOP |

全式・5層・費用会計の詳細は前回草案 §§4–9のまま。旧510、V2 binary tail、真variance/最適配分は使わない。
priceの期待confidenceを付けない場合、未使用price αを結果後に他用途へ移さず、実現費用と期待費用を明確に区別する。

既存 `EventMetadata`/bank.values、`Batch/require_disjoint`、`fit_and_freeze/FrozenCV`、`evaluation_inputs`、raw/stratified_mean、TrialCost/account_costsを再利用する。
測定engineは既存fresh-trajectory/conditional-bit規約を流用する小接続に限り、旧一括runnerを実行しない。
streamはmethod/axis/stage/stratum/trajectory/measurementを分離し、fresh PCG64の固定規約へbindする。seed/domainの具体値はoutcome前に一度決める未固定事項で、有利なseed探索はしない。
推定器へ渡すのはbit、元weight、古典path metadataと価格だけ。engine内部の未知条件付きmeanをデータとして出さない。

出力は全stageのrecord、凍結λ/range、cal variance上界、十分Nと丸め、production推定/interval、全正層のquotaと棄却費、方式別単独運用費・実験取得費の分離、古典時間/memory、未完了費用。
記述的なvariance低下と、厳密なvariance改善検定を区別する。後者を行ったとは主張しない。
STOP：reuse/stream/path不整合、正残余なし、N/attempt/range/price scope超過、入力破損、実際の共有資源不足。結果を見た延長・再実行・別label/roundへの救済なし。

## 7. 最小の次の許可と停止境界

推奨する最小の次段階は**Q1＋Q2-Lの非sampling数値取得**。これで実q/support/νと元targetへの誤差予算を閉じられるか判定してSTOPする。
WP04保証を使用しないfinite-mean診断を選べば、Q2-T・新PF/radius評価を初回から要求しない。
その後のQ3–Q5を同時に許可する場合は、条件が閉じたときだけ有限scope内で継続し、段階ごとの新しい承認発行手続きは不要。
Q3–Q5までの許可がない場合、Q1/Q2の成功を測定実行許可と解釈しない。

ユーザーが次に選ぶ事項は、有限target座標診断か旧RPE保証の移送も必要か、3control/比較4方式/各提案予算・ε/α・ideal gate仮定を採るか、Q3の状態再構成、Q4の不足compileと費用報告scopeをどこまで許可するかである。
新しい重要科学レビュー、C1解凍、S2/R1再実行、全18round/別分子・rank/parameter探索は含めない。

新規実行は自分の独立SOURCE/output/cacheで行い、既存SOURCE/raw/grant/ledger/marker/STOP/one-shotを保全する。
人工的な低いCPU/RAM/worker制限は設けず、実際の空き資源と非干渉に応じて効率化する。共有設定・他ユーザーjob/files/venv/cacheは操作しない。
**今回のmandatory STOPは技術接続・本実行仕様・公開本文照合まで。Q1–Q5は一つも開始していない。**
