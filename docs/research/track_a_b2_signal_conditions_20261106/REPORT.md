# Track A B2：RPE科学条件案と最初の2round信号検証仕様

2026-11-07 JST / SERVER 2（GPUA100-gleap02）/ Codex主エージェント。
専用branch `codex/track-a-b2-signal-conditions-20261106`、起点は減衰結果 `e69a51de70651ced673c22d2b215683a9f30059b`。

**結論：既存手法を使う条件付き数値pilotへ接続できる。以下のP1案の採択、private入力の受渡し承認・照合、薄いadapterの実装後に、B2のm=0・17だけを検証する。現時点の科学実行はSTOP。** 元H/stateの科学的対応はLOCALの固定結果で確認済みだが、SERVER 2でのprivate実体の受領・照合は未実施。18roundのenergy精度・95%成功保証は未認定である。

今回は固定SOURCE・公開結果・metadataの読取と本仕様の作成だけを行った。新しい数値解析、位相・horizon・shot・費用算出、state読込計算、simulation、回路build/compile、sampling、詳細科学レビューは行っていない。以下の数値結果は固定成果の引用、誤差配分と数値許容値は**採択前の設計案**である。

## 1. 固定入力と証拠

| 対象 | 固定結果・今回使う意味 |
|---|---|
| 元H/state | [state結果 `89165cf…`][STATE]。cutoff前の`H_phys`、linear H4 / 1.00 Å / STO-3G、Nα=Nβ=2、sector36。元Hの厳密基底状態・無料state preparationを研究上仮定する。保存stateの21/21 correctness PASS、raw H_phys full残差約2.56e-15 Ha。全Fock最低性・厳密interval証明ではない |
| 元H–DF12 | [対応結果 `275813d…`][CORR]、構築SOURCE `126b9d4480c7937b39c8be3aef7fbecda48cb556`。65/65 operator checksと80/120桁の安定性検査PASS。H_phys→cutoffを含むH_src→全signed DF12を区別。h_effの一体補正、Hermitian化、核scalar、spin/JW対応を確認済み |
| B2接続 | [SOURCE `66a41ad…`][CONN]。generation-prefix L_D3、二次partial-S2、r4/K2。決定論部分はone-bodyとfragments0..2、randomは3..11。coefficient_atol=0、tail identity抽出。219人工テストPASSは実H4信号のPASSを意味しない |
| horizon・減衰 | [結果 `e69a51d…`][ATT]。指定ε_E/β/δを既存`required_rpe_round_horizon`へ適用してM=17、Q_max=131072、nominal分解能.000152587890625 Ha。今回再計算していない |
| 旧仕様の継承 | [Agent A仕様 `2796d26…`][SPEC]の比較・入力区別を継承。そこで未取得だった元H/stateは上記LOCAL成果で解決した。旧DF基底state、M1の別SCF/DF10/stateへの置換はしない |

`H_phys`が共通energy target、`H_src`は生成経路の比較診断、`H_DF12`はB2と将来の強PFの共通simulation modelである。核scalarは`2.29310124732 Ha`（hex `0x1.25845777d3feap+1`）を一度だけ保持する。旧DF12を生成し直さず、rank・prefix・r/Kを変更しない。

## 2. 採択を提案するP1条件

| 項目 | P1案と根拠・限界 |
|---|---|
| 精度・成功目標 | ε_E=`0.00015936001019904 Ha`、全体成功目標95%。まず条件付きpilotの**目標**として採択する。達成の認定はしない |
| schedule | β_RPE=.4 rad、δ=.02 Ha^-1、m=0..17、Q_m=2^m、全round r4/K2。既存horizon結果と整合。β/(Q_max δ)というnominal条件だけでは元Hのenergy保証にならない |
| 事前energy区間 | **I_E=[−128,128] Ha、E_shift=0**を提案。下記の係数和による入力依存・state非依存の根拠を用いる。M1の[−10,0]や計算済み基底energyからの逆算は使わない |
| 位相・energy変換 | `U(t)=exp(−itH)`、`φ=−δE`、`E_hat=−φ_hat/δ`。全round同じH_phys・同じ絶対energy。核scalarと抽出tail identityはcontrolled相対位相へ一度だけ含め、phase alignmentや既知Eによる事後補正をしない |
| branch | 下記の初期区間条件と、全roundの誤差≤β_RPE、strict `β_RPE<π/3`を併用。m0 principal phaseを初期estimateとし、その後は既存nearest-branch decoderを使う |
| 失敗確率 | 全18round・X/Y均等：`α_m,X=α_m,Y=.05/(2·18)`。各armでunion bound≤.05。2round検証のために.05/4へ振り替えず、未使用分を再配分しない |

### 2.1 energy区間と初期aliasの根拠

対応SOURCEの[`arithmetic_rule`][CORRSRC]と保存[result.json][CORRJSON]には、

\[
S=|c|+\sum|h_{\rm phys}|+\sum|W_{\rm phys}|
+\sum|h_{\rm eff}|+\sum_l|\lambda_l|(\sum|G_l|)^2
=103.50446476572802
\]

という**既計算のoperand_scale**がある。Wはspin-lift後の1/2を含むtwo-body tensorである。フェルミオン単項式のoperator norm≤1と三角不等式より、最初の3項が元Hのnormを抑え、Sはさらに非負項を含む。したがって、保存係数を対象とする条件付き数値pilotでは、Sより広い[−128,128]を事前区間とする案に根拠がある。これはstate solveの結果に依存しない。**新たに積分和やnormを算出した結果ではない。**

このSは元来roundoff診断用で、SOURCEは`rigorous_roundoff_bound=False`と明記する。Sの理論的な係数和の意味と、保存binary64値の厳密な上向き包含は区別する。保証付きclaimには、元係数和の保守的包含が128未満であることの確認が別途必要。保存`max_entry_tolerance=4.2802809114947716e-10 Ha`をmodel誤差やenergy幅へ代入しない。

初期条件は`−π+β_RPE < −δ(E−E_shift) < π−β_RPE`をI_E全体で要求する。この案は`δ·128+β_RPE<π`を満たす設計であり、shift追加は不要。以後のnearest-branchは**円周上の前roundとの距離**を使う。既存[`reconstruct_rpe_phase`][DECODER]はQ=1から始まる連続した2の冪だけを受け付ける。m=0・17だけをdecoderへ渡すことや、保存Eを使ってbranchを選ぶことはしない。保存Eは信号・誤差の比較参照にだけ使う。

ユーザーがこの係数和に基づく数値priorを採択しない場合は、独立した根拠を伴う別の事前energy区間を指定するまでSTOP。区間幅・δ・shiftを探索して代替しない。厳密energy保証を先に要求する場合も、係数和の包含と後述の全round誤差資格を満たすまでSTOPである。

### 2.2 全round共通の位相配分案

次は今回の入力用に明示する**新しい配分の提案**であり、既存結果から実現可能性を確認した値ではない。phase差は各roundの`Q_m φ`に対するradで、base phaseの誤差と混ぜない。

| 成分 | 上限案（rad） | 次の2roundで調べるもの |
|---|---:|---|
| model | .02 | 同じ元H stateのDF信号対元H信号の位相差＋数値allowance |
| deterministic PF | .06 | ideal-tail PF信号対DF信号の位相差＋allowance |
| finite-RTE | .04 | corrected対ideal-tail PFの複素誤差diskを、PFの実半径下限で既存arcsin変換 |
| statistical | .24 | 将来の量子測定のために確保。今回・最初の信号pilotでは測定もshot算出もしない |
| implementation・numerical | .02 | controlled/nativeの意味、平均operatorの独立照合、長roundへの数値誤差伝播を含む。未検証を0としない |
| reserve | .02 | 未使用の余裕。自動再配分しない |
| 合計 | **.40** | 全roundで各capを個別に満たすことも要求 |

既存`RPEErrorAllocation`へはPF=.06、RTE=.04、stat=.24と上記αを渡す方針。その`beta_total`は**この3項だけ**で、model/implementation/reserveを含まない。薄い資格判定でmodel≤.02、implementation≤.02、reserve=.02を別に保持し、6項の総枠を照合する。数値allowanceを加えたPF/model/diskの値を使い、同じallowanceを重ねて加算しない。baselineにも同じcap・配分規則を適用し、RTE=0の余りを事後流用しない。

保存された元H–DF12 full spectral差約9.27343e-15 Haは、高精度差行列をbinary64 SVDした**数値診断**であり、厳密上界ではない。時間・位相換算も本作業では実行していない。保証を付けるなら、適切なHermitian表現と数値差を区別して`||H_phys−H_DF12||≤η`を正当化し、既存のDuhamel型`|z_DF−z_phys|≤T_m η`等と非零半径から位相へ変換する必要がある。今回この数値を自動的にηと認定しない。

P1の2round pilotは**観測した同一stateの信号差＋採択した数値allowanceによる条件付き判定**とする。`RPEConnectionContext.model_error_bound`は厳密上界が未資格なら`None`のまま保持できる。これはsignal-only入口を妨げないが、`require_evaluated_conditions`・shot会計のGOにはしない。条件付きpilotを回すために架空のmodel上界を入力しない。

量子測定の95%目標には、将来、正のraw半径下限と採択stat予算、X/Yのbounded outcomes、canonical lawのfresh IID trajectory per shotが必要。価格MCの標本数・標準誤差・CIはこの.05とは別である。[既存実半径会計][ACCOUNT]

## 3. private入力の結合と受渡し

既存[`agent_b_input_spec.json`][INPUTSPEC]を入力の正本として使い、書き換えない。

| 照合対象 | 必須identity |
|---|---|
| raw H_phys full operator / S0 array hash | `bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2` |
| 元H full state / S0 array hash | `1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db` |
| private state.npz / file SHA256 | `deb561367e216fd4c62d013c10a4adbbef5da8c3e8139c39170d71f80530a80c` |
| 旧DF12 NPZ / file SHA256 | `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、commit `bf5d3b2405b2bec4f88dd0589b75dd737b13e052`、path `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz` |
| 保存DF Hamiltonian hash | `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。DF file hash、full matrix hashとは別の規約 |
| sector / OF basis S0 hash | `Nalpha=2,Nbeta=2` / `665f72f65a4300043a32f3aabc76b7deafbd2563ad3947228dd937bc6d521dd7` |
| 元Hの既存HDF5 / file SHA256 | `a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958` |
| state生成SOURCE | `b168cbf93d7c319045057e8b67ea7ff014c8d7dc`。将来のsignal実行SOURCEとは別に保持 |

受渡し対象は、LOCALの`artifacts/b2_hphys_ground_state/2026-11-06/run01/private/state.npz`と`provenance.json`、および**同じ既存HDF5**のprivateコピーを提案する。state NPZには`state_vector(256)`、`sector_state_vector(36)`、両basis index、energy、metadataが入るが、H_phys operatorやh/gは入っていない。元H hashをSERVER 2で再構築・照合するには、このHDF5の保存積分が必要。state NPZだけでoperator実体を受領済みとは扱わない。新SCF・積分再計算・DF生成・state solveは不要で、行わない。

**既存の公開入力仕様と環境同期policyには、private state用の実動するLOCAL→SERVER 2転送endpoint・経路の指定がない。今回もprivate実体や接続性は確認していない。** [同期policy][SYNC]はprivate実体と公開metadataを分ける。したがって、ユーザーが既存の認証済みSSH/SFTP/scp等の経路と受領先を指定し、対象ファイルの転送を明示承認するか、自身で受け渡す。GitHub・PR添付・公開artifactは経路に使わず、秘密endpoint・鍵・絶対path対応表を公開しない。鍵発行・共有設定変更は不要。

次の承認範囲での受領・照合手順は次に固定する。

1. ユーザー所有のGit外private directory（700）へ既存ファイルをコピーし、ファイルは600、原本不変更。転送前後のfile SHA256を上表と照合。private配列・HDF5・内部pathを公開logに出さない。
2. `np.load(...,allow_pickle=False)`で指定field・shape・dtypeを確認し、既存S0 `array_hash`を使用。`dtype.str` UTF8＋compact JSON(shape) UTF8＋C-order bytesであり、M1の別hash関数に置換しない。stateは`<c16`、indexは`<i8`。保存stateの再正規化・phase変更をしない。
3. spin順序は`2p=alpha,2p+1=beta`、Qiskit full indexは`sum(n_q*2**q)`。OF sector indicesを既存bit reverseでQiskitへ写し、保存full/sector stateとの一致とsectorを確認。旧DF stateを取り出してfallbackしない。
4. 固定構築SOURCEの`exact_jw`/既存spin liftを使い、HDF5の保存h/g・MO・AO overlap hash（[対応JSON][CORRJSON]）とraw H_phys operator hashを照合。H_src cutoff・DF一体補正・核scalar・identityを別々に保持。異なるhash、同名別入力、DF rank違いはSTOP。
5. `RPEConnectionContext`には上表の元H/DF/state、sector/order、将来のdriver SOURCEを渡す。input specの`source`をdriver SOURCEと誤認しない。`state_role=original_H_ground_state`、prep除外を維持。compiler/backend/cost_metricも型上必要なので、既存仕様のQiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend指定なし・`no_backend`、`rz_count`を**識別用**に提案する。compiler実行や価格取得を意味しない。他の費用指標の比較は後段で同じcontext条件から行う。

受渡し後のoperator/state計算は次の科学実行承認に含める必要がある。今回はファイルを転送・受領・読込していない。

## 4. 次の限定科学検証：B2のm=0・17だけ

P1と受渡しを採択・承認した後、adapterの人工確認とSOURCE固定を済ませ、**同一の元H state**で下記を実行する案とする。既存の219テストや減衰スクリーニングは再実行しない。baseline、他round、探索、shot・価格・総費用は範囲外。

| round | 固定時間（既存減衰結果から引用） | normalizationの既存参照 |
|---|---|---|
| m0 / Q=1 | T=.02 Ha^-1、r4/K2 | B=1.000034020844717、A=.9999659803126615 |
| m17 / Q=131072 | T=2621.44 Ha^-1、r4/K2 | B=86.41007760254907、A=.0115727242440354 |

各roundで`z_phys=<ψ|exp(−iT H_phys)|ψ>`、`z_DF`、`z_PF`（ideal tail）、`z_F`（finite corrected）、`z_raw`（正規化を含むcanonical event平均）を別fieldに保存する。元H厳密固有状態の仮定による`exp(−iT E_phys)`も比較参照にし、operatorからの元H信号と区別する。これは既知Eをbranch選択へ使う許可ではない。

raw H_physは微小なanti-Hermitian成分を持つ数値表現で、state生成は明示した`H_solve=(H_phys+H_phys†)/2`を使用している。[state結果][STATE]のraw残差・symmetrization差を保持し、元H参照を旧`_eigendecomposition`へ入れて黙ってHermitian化しない。raw operatorからのexpm診断と、Hermitianな物理H・厳密基底状態という仮定を区別し、差を数値/実装側へ記録する。元targetをH_solveやH_srcに変更しない。

検証・保存するものは以下に限定する。

- 5信号の実部・虚部・半径・位相、隣接するmodel/PF/RTE段階の複素差と円周位相差。`z_DF`にも元H stateを使い、DF ground stateを使わない。
- prefixのforward half sweep→tail→reverse half sweep、全signed factor、核scalarと抽出tail identityの一回計上。finite operatorは既存canonical K2（paired order0/2、degree3 polynomial）の平均を使う。
- `λ_R=.5832704246339983`、抽出tail identity `.09261516589219072`、`τ=.0029163521231699912`は既存結果の照合参照。BとAは既存finite関数を使い、`z_raw=A z_F`、`BA=1`を確認。raw平均を別の既存moment経路でも照合し、単に`A z_F`を代入した値を独立検査と呼ばない。減衰AをPF半径やcorrected半径へ二重に掛けない。
- `B2SignalRecord.bounds`からPF半径下限・RTE複素disk・model/PF/実装phaseを取り、既存M1のdisk/radius条件で各capを判定。`ρ_PF,lb>0`、`e_RTE<ρ_PF,lb`、`asin(e_RTE/ρ_PF,lb)≤.04`、raw半径下限が正であることを要求。減衰Aだけではこれらは判定できない。
- 数値許容値のP1案は`u_z=u_radius=u_normalization=1e-8`（dimensionless）、`u_phase=1e-8 rad`。データを見る前に固定するpilot用余裕で、厳密誤差上界ではない。既存B2作用経路とM1の独立paired-polynomial/moment・matrix-power経路を接続し、対応する平均・normalization等の不一致が各u/10以下か調べる。成立後にuをboundsへ渡す。失敗時に許容値を緩和せずSTOP。独立実装の一致も厳密包含や95%保証とはしない。
- controlled位相について、将来の承認に**実H4の代表primitive・一つのouter stepの未compile correctness build**を含める案とする。既存AX emitterとM1 `controlled_reference`を用い、control-0 identity、control-1相対位相、signed event位相、核scalar/identity、X/Y wrapper（X=Re、Y=Im、YはSdg→H）を確認。RNG sampling・価格compileはしない。m17のQ回を展開した巨大回路は作らない。反復の意味は既存request順序と平均operatorのQ乗で接続し、m0のprimitive PASSだけで長roundの実装phase boundを0としない。長roundへ伝播する数値誤差を.02枠内に資格化できなければ**implementation未適格でSTOP**とする。

m0・17のPASSは「この2点でP1の信号・数値・実装条件を満たした」という条件付き結果に限る。中間16roundの半径・位相を補間したり、全18roundのnearest-branch成立・energy精度・95%を認定したりしない。残るroundへの拡張は2点の結果を見てユーザーが別途判断・明示承認する。

## 5. 既存コードの再利用と最小差分

| 接続 | 再利用する具体的箇所 | 次段階の差分だけ |
|---|---|---|
| 入力 | state結果の`agent_b_input_spec.json`、構築SOURCEの`array_hash`/`exact_jw`、既存snapshot loader/bit reverse | private pathを明示引数で受けるloader。DF loaderが返す旧DF stateは破棄。元stateの生成・代用はしない。`b2_hphys_ground_state.run`を呼ばない |
| B2分割・作用 | [旧B2 kernel][KERNEL]の`_prepare`、`_dense_block_operators`、`_preparation_eigensystems`、`_apply_exponential`、`_apply_spectral_values`、`_apply_outer_step` | LD3固定、t/Qを指定。`_target_signal`はDF固有state＋旧TOTAL_TIME=.8専用なので呼ばない |
| finite平均 | 同kernelの`_random_signal_record`内の平均作用、[rte.py][RTE]の`make_rte_config`/`finite_rte_operator_moments`/normalization/attenuation、[pauli_signal_adapter.py][PAULI]の`paired_polynomial_action` | 旧`_random_signal_record`は`_base_record`→`_axis_record`でshot/Wまで計算するため丸ごと呼ばない。**既存の平均作用部分だけをmean-only helperへ抽出し、旧入口もそこを呼ぶ**。PF/RTE式の再実装はしない |
| 独立照合・Q乗 | [M1 validator][M1SIGNAL]のpaired-polynomial/moment照合と`np.linalg.matrix_power`、独立CAR/occupation参照 | M1のLD0・faithful-identity専用runnerを直接使わず、既存B2 half sweepsを前後に接続。H4-A/DF10 loaderやM1信号・価格を移植しない |
| 信号→資格判定 | [b2_rpe_connection.py][CONN]の`B2RPERound`、`SignalValue`、`B2SignalRecord.bounds`、[M1 `account_radius_round`][ACCOUNT]の半径・arcsin・予算判定 | M1の判定部分だけを小helperへ分離して共有し、shot呼出の前で止まるsignal-only入口を追加。既存式を複写しない。model/impl cap・reserveを薄いadapterで照合 |
| native・wrapper | [AX native][AX]の`partial_native_from_step_requests`、`build_native_hadamard_wrapper`、M1 `controlled_reference`、[既存wrapper][WRAPPER] | prefixのあるB2 requestを使用。代表event/primitiveと1 outer stepだけの実H4 probe。全m17回路・compile・価格を呼ばない |

`account_signal_round(...,cost=None)`でも既存`account_radius_round`は適格ならshotを算出するため、**cost=Noneだけではsignal-onlyにならない**。`evaluate_rpe_round_candidate`のunit-radius/effective-PF-eigenstate保証経路も、この元H stateへ適用しない。

新規差分は上表の薄い入力・mean-only・資格判定adapterと2点driver、これらの変更に対応する小さな人工テストだけ。既存219件の接続を作り直さない。テストではprivate fixtureを使わず、shot/price入口が呼ばれないこととprefix/scalar/identityの接続を確認する。今回は実装・テストを行っていない。

将来の費用会計は既存`account_signal_round`→`aggregate_b2_rounds`、AX full no-prep価格へ接続する。その際の比較対象は同DF12の強Full S2・標準Yoshida4、基本δあたりsubsteps1/2/4/8の有限8arm、各arm全round固定。spin-aware実装・relative Gaussian fusion・同compiler・X/Y whole wrapperとprep除外を共通にする。**最初の2round pilotではbaseline信号・shot・価格は取得しない。** 終点は後段の`G_j=Σ_m Σ_b N_j,m,b E[C_j,m,b^(full,no-prep)]`であり、今回の減衰値や単回価格を総費用と呼ばない。

## 6. GO/STOPと必要な承認

| 区分 | 状態・次の扱い |
|---|---|
| 既存証拠で確認済み | 元H/state/DF12対応、sector/order、固定B2、horizon18round、normalization減衰。これらの範囲の再調査・再実行は不要 |
| ユーザーの採択が必要 | **P1一式**：条件付きpilotという証拠水準、I_E=[−128,128]/shift0、表のschedule・位相配分・均等α、数値許容値、入力・compiler識別条件。95%達成の認定は含めない |
| 別途の転送承認・情報が必要 | private state/provenance/既存HDF5のLOCAL→SERVER 2経路・受領先を指定し、転送を明示承認するかユーザー自身で受け渡す。現時点で受領済みとしない |
| 接続実装・人工テストで解決 | mean-only抽出、入力照合、shotを呼ばない資格判定、prefixを保持した独立平均経路。上表の差分を作りSOURCE固定 |
| 新しい科学実行の承認が必要 | **固定入力の受領後照合・再構築と、m0/17の5信号・半径・誤差診断、代表primitive/1 outer stepの未compile controlled correctness**。state solve、新SCF/DF、他round、sampling、shot/費用、価格compile、探索は含めない |
| GO | P1採択＋private受渡し・科学実行の明示承認＋入力照合PASS＋adapter人工確認後、指定2点だけへ進む |
| STOP | hash/sector/order不一致、必要private入力なし、予算・半径・normalization・数値照合・controlled意味が未適格、証拠を資格化できない場合。DF stateや別入力に置換せず、r/K/δ/prefixを変更しない |
| 2点完了後 | 結果を公開してSTOP。2点PASSでも全roundへ自動拡張しない。失敗でも再最適化しない |

## 7. 詳細科学レビューと終了

**P1の条件付き2round pilotにGPT詳細科学レビューは不要。開始していない。** 区間設定、既存nearest-branch条件、複素誤差disk/実半径、均等union bound、既存canonical平均の技術的接続で進められる。private入力不足やadapter修正はレビューで代替しない。

保証付きenergy/成功claimには、数値prior・model/実装の厳密上界、全18roundの正半径と位相条件、IID測定条件の資格化が残る。これらを既存の十分条件で示す限り、新しい科学理論のレビューは自動的には必要ない。**元H stateをPF固有stateと見なす、観測差を無条件に厳密上界とする、中間roundを省いてbranch成功を保証する、既知Eで補正して推定成功とするなど、既存条件を超える主張を採る場合のみ**、その仮定・定理の妥当性に限定した別レビューの明示承認を要する。P1はそれらを採用しない。

**次の担当：ユーザー**がP1・受渡し方法・上表の限定科学実行範囲を採択・承認する。**Codex**は別途の明示承認後にだけ、最小adapter・人工確認・SOURCE固定・受渡し照合と指定2roundの検証を行う。

本作業はREPORT一つを専用branchへGitHub連携でcommit公開し、remote SHA・tree・本文一致を確認して**mandatory STOP**。旧SOURCE/input/raw/result/grant/marker/STOP、M1終了提案・M2保留は不変更。private転送、実H4計算、実装、全round拡張、詳細レビューへ自動進行しない。

文書確認：17件の固定commit/pathの存在、公開input specと上表のidentity、変更対象がREPORTだけであること、staged diffの空白検査を確認した。科学ライブラリのimportや数値配列のdecodeは行っていない。

[STATE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89165cf70958e7e4676d0580e3acb7faf588772c/docs/research/b2_hphys_ground_state_20261106/REPORT.md
[INPUTSPEC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89165cf70958e7e4676d0580e3acb7faf588772c/artifacts/b2_hphys_ground_state/2026-11-06/run01/agent_b_input_spec.json
[CORR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/275813d0a12a4bca62a8c972312c80386829de22/docs/research/b2_original_df12_correspondence_20261106/REPORT.md
[CORRJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/275813d0a12a4bca62a8c972312c80386829de22/artifacts/b2_original_df12_correspondence/2026-11-06/run03/result.json
[CORRSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/126b9d4480c7937b39c8be3aef7fbecda48cb556/src/trotterlib/b2_original_df12_correspondence.py
[SPEC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2796d26270cf39e46b40accc4f898fe29d459819/docs/research/track_a_b2_original_state_spec_20261106/REPORT.md
[ATT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e69a51de70651ced673c22d2b215683a9f30059b/docs/research/track_a_b2_rpe_attenuation_20261106/REPORT.md
[CONN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trotterlib/b2_rpe_connection.py
[DECODER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trotterlib/rpe_four_round_phase_validation.py
[ACCOUNT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trotterlib/o1_m1_resource_bridge.py
[SYNC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e69a51de70651ced673c22d2b215683a9f30059b/docs/research/track_a_environment_sync_policy.md
[KERNEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trotterlib/pr2_matched_accuracy_m1_execution.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trotterlib/rte.py
[PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trotterlib/pauli_signal_adapter.py
[M1SIGNAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trotterlib/o1_m1_circuit_signal_validation.py
[AX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trottertracks/resource_applicability/ax2a_native_df.py
[WRAPPER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trotterlib/rpe_hadamard_interrogation.py
