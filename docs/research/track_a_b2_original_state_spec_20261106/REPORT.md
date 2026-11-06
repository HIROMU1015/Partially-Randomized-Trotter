# Track A B2：元H/state対応と限定総RPE pilot仕様

2026-11-06 JST / SERVER 2 / Codex主エージェント。起点は[前回REPORT][PRIOR]。今回はその成果一覧を再作成せず、入力の結合と次の実行仕様を確認した。**科学実行への判定はSTOP：旧DF12に対応する元Hの積分・軌道と元H stateを、取得した保存資産だけでは結合できなかった。** DF12自体の固定入力は回収・同定できた。科学計算、数値再解析、adapter実装、回路build/compile、詳細レビューは行っていない。

## 1. 元H・DF12・state：今回確定できた来歴

### 1.1 DF12の実体と生成SOURCE

DF12 snapshotは起点branchのtreeにはないが、固定commit **`bf5d3b2405b2bec4f88dd0589b75dd737b13e052`** の [`artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz`][DFNPZ] に存在する。Git blobを読み、file SHA256、NPYのmember/header、Unicode metadataだけを確認した。数値配列はdecodeせず、数値比較もしていない。

| 固定対象 | 同定値・根拠 |
|---|---|
| NPZ file SHA256 | `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。保存metadataと今回のfile bytesが一致 |
| DF Hamiltonian hash | `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`（metadata記録。今回operatorから再計算していない） |
| corrected one_body / lambdas hash | `0e1b42516947b620a3c60b012b62fc87ef22b2567988cfe1aae01392db660012` / `1b80d4122cabadf1e9b08d4380d7679158009b8f19dd128f06e45720e20780d0`。12個のG hashとgeneration順0..11もmetadata内に保存 |
| 核scalar | `constant_hex=0x1.25845777d3feap+1`。SOURCEは核反発のconstantを保持し、DF分解第4戻り値をscalarへ加えない |
| 保存state | `state_hash=31e63b0104126c85136ee173f1dce7642aee2d272924e70e8b120ac340ab45bd`、`state_vector_hash=c9aca811b5c023772d148d0331c82958bac6824b593a367cb65a5f89937f4f63`。**DF ground**であり元H groundではない |
| 生成SOURCE | `c644925b50587072784846df09bf02c39e8453e1`。[`pr2_s0_s1_validation.py`][GEN] SHA256 `1ccf6505d2933699504db187e3849251d1fef7bc58d8dca0b5f19871896ca6fe`、[`df_hamiltonian.py`][DFGEN] SHA256 `7d711e1c3c093079ae890ffc1907813d55d211a4fb622e4f36edee05fdea0c52`。両SOURCE bytesがsnapshot内source_hashesと一致 |

SOURCEの実際の順序は、`run_pyscf(...,run_scf=1,run_fci=0)` → `get_molecular_hamiltonian()` → `df_hamiltonian_from_integrals(...,df_rank=12,df_tol=None)` → `solve_df_ground_state(hamiltonian,sector,tol=1e-12)` → phase canonicalization → snapshot保存である。生成時の旧S0 `STOP_INPUT_REPRODUCTION_MISMATCH` は別の旧pilotとの入力不一致であり、後続B2が使ったこのsnapshotを新たに失効させるものではない。旧STOPは変更しない。[生成SOURCE][GEN]、[保存S0 metadata][S0]

### 1.2 元分子Hの定義経路と保存されなかったもの

| 項目 | SOURCE・metadataから確定した内容 | 残る不足 |
|---|---|---|
| 原子・基底 | linear H4、1 Å、STO-3G、charge0、multiplicity1。`geo/call_geometry`はz方向の中心化した直線配置。8 spin orbitals、Nα=Nβ=2/sector36 | 当時のMO係数・AO overlap・SCF中間入力の保存結合 |
| h/g積分 | PySCF SCFのMO係数Cを使うh=Cᵀh_core C。OpenFermion-PySCF v0.5の二電子配列はchemist ERIを`transpose(0,2,3,1)`した規約 | **元h/gの配列・hashはB2 NPZにない**。保存one_bodyはbare hではなく一体補正後 |
| spin/JW | spatial pの2p=α、2p+1=β。`get_molecular_hamiltonian`のInteractionOperator two_bodyには1/2を含む。OpenFermion stateからQiskitへは既存`_to_qiskit_state`のbit reverse。配列寸法一致だけで変換を省かない | 元H stateをこの軌道順序・Qiskit順序へ結合した保存記録 |
| DFとHermitian化 | `low_rank_two_body_decomposition`のλ/Gとone_body_correctionを使用し、h_eff=h+correction。第4戻り値はtruncation値。one_bodyと各Gを`(A+A†)/2`へHermitian化。全12 factorsと符号・順序を保持 | 変換前後の実誤差は未評価。保存`df_truncation_value=0.0`だけで元H一致を認定しない |
| B2 identity | `generation_partition(ham,3)`、`explicit_ordered_partition`、`extract_identity_phase`、`coefficient_atol=0`。one-bodyとfirst3 fragmentsを決定論、残9をfinite-RTE。抽出tail identityと核scalarは一度だけ相対位相へ | 元h/gからDFまでのmodel誤差は、identity抽出の同値性とは別 |

[geometry/JW SOURCE][GEO]、[DF生成][DFGEN]、[B2準備][B2]、[OpenFermion v1.6.1][OF]、[OpenFermion-PySCF v0.5][OFPY]

追加で判明した境界として、OpenFermion v1.6.1の`spinorb_from_spatial`は`EQ_TOLERANCE`未満の成分を0にする。したがって **元MO積分によるH_phys → thresholdを含むInteractionOperator H_src → DF分解・Hermitian化後H_DF** を分ける必要がある。今回採択したDF12を変更する話ではない。元HをH_physとするなら、このcutoffもmodel誤差へ含める。H_srcを元Hと定義するなら、その定義を明示する。packageの固定版SOURCEは規約の確認に用い、当時の実行dependency全bytes一致を新たに認定したものではない。

### 1.3 保存された元H ground stateの候補を結合できたか

| 資産 | 読取確認 | 今回の採否 |
|---|---|---|
| 旧DF12 NPZ | constant、h_eff、λ/G、sector、DF state、metadataだけ。bare h/g、MO係数、original-H stateのmemberなし | DF12本体として固定。元H stateとしては**不採用** |
| AX H4 v5 | input_referenceは上記NPZ SHAと同じ。保存stateのnative/independent occupation参照を検査する資産 | DF12/native correctnessへ再利用。original-H stateの新供給にはならない。[AX入力][AXINPUT] |
| 旧RPE physical_df_ground_state | 別H hash `56e4df83…` のDF groundを使用 | 同じ分子名でB2へ結合しない。既存RPE証拠は方法の再利用対象 |
| O1 M1 H4-A | 下記の元H stateが実在。ただし**新しいSCF・tol-only DF10**からの保存物 | B2元入力とのMO/h/g bindingなし。直接結合不可 |
| 歴史的MolecularData HDF5 | [旧V0監査][V0]にS0時生成 `H4_sto-3g_singlet_df_d100.hdf5`、SHA256 `a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958` の観測あり | 元h/g/MO回収の**具体的候補**。今回読んだ固定treeにHDF5実体なし。旧LOCAL作業pathとSERVER2既知package dataのexact pathも不在。内容は未読・元H ground保存の有無も未確認 |

M1入力は `4591b8d75a2a6a55c74a51225f0c61648023c8ee` の [`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`][M1NPZ]、file SHA256 **`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`**。今回はhashとheaderだけを確認した。`mo_coeff/one_mo/eri_chemist/spin_one/spin_two/state/DF_state/non_DF_matrix/DF_matrix`が別memberで保存されている。

その生成SOURCEは **`0c329b30a81accb35c212be8defa66fee0dd0f12`**、[`h4_molecular_inputs.py`][M1GEN]の`build_h4_inputs`。SOURCE SHA256 `fb53bf44459dc6ef18223d55594511c03766c53537f6c445dad7a0b7a42225ea` は保存SOURCE記録と一致する。元積分から構築した`full`へ`canonical_ground(full,indices)`を呼び`state`を、DF行列へ別に同関数を呼び`DF_state`を生成している。よってM1の`state`はDF stateの別名ではない。receiptの`state_hash=8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`、`df_state_hash=bfdc4948b4c08230d3ab7a3f11d4a3990875f6cf3423eea2d15a72a2d0ff8667`、Pauli H hash `e0dbbe9a2b6e86f81004307cd337ba83ee1e25642f04a30b02365be4d2dae795` は[保存receipt][M1RECEIPT]の引用である。hash関数の異なる旧B2 state hashと直接比較しない。

ただしM1は新SCF・unthresholded spin lift・DF10で、旧B2の元h/g・MOを再利用した実行ではない。global translationを含むgeometryの表記差や近いenergyだけでは、同じ軌道表現の同一入力と認定できない。M1のoperator/overlap検査もM1内の元H対DF10であり、旧DF12との比較ではない。**元H厳密基底状態・無料state preparationという研究仮定は維持するが、仮定を理由に異なるstateを代入しない。**

## 2. 第一B2・RPE・baselineの比較仕様

以下は次回承認対象を具体化する仕様案であり、右欄の未確定条件が埋まるまで実行しない。

| 対象 | 条件・意味 | 状態 |
|---|---|---|
| B2 | ユーザー採択：H4/1 Å/STO-3G、保存DF12、generation L_D3、partial-S2、r4/K2。旧ID `B2-rank3-q1-r4-K2`。新rank/prefix/r/K探索なし | 採択済み。旧T=.8/q1は新RPE条件にしない |
| RPE候補 | ε_E=`.00015936001019904 Ha`、成功目標95%、β_RPE=.4、δ=.02、Q_m=2^m、r_m4/K_m2 | **候補のまま**。旧18round・旧N・信号・価格は不使用 |
| horizon | `required_rpe_round_horizon(ε_E,β_RPE,δ)`を承認後に一度適用。全m=0..M、t_m=Q_mδ。SOURCEのnominal条件はβ_RPE/(Q_Mδ)≤ε_E | 今回M/Q_max/分解能を計算していない。nominal分解能だけで成功保証にしない |
| baseline集合 | 強Full S2と標準Yoshida4、s=1/2/4/8の**8本**。同じ全DF12 factors、one-body、核scalarを使用 | sは基本δ区間あたりのsubstepsとする案。round全体はQ_m s steps、各substep δ/s。同じt_mへ比較 |
| baseline選定規則 | 各formula/sを全round固定したarmとして評価し、全round適格armの総費用を比較。roundごとの勝者を事後混合しない | 有限集合内の比較。全PF最適性・baselineが必ず適格との主張なし |
| native/control | exact spin lift、relative Gaussian fusion、同basis隣接合算、controlled相対位相を保持する共通lowering。ordinary-controlledを共通参照とする | `spin_adapted_eigh`はG自体のexact spin構造を要求。旧DF12への適合はmetadataでは未確認。失敗時にspin projectionせずSTOP |
| compiler/価格 | Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/target/coupling/layout/routing指定なし。X/Yを個別whole compile、ancilla準備・readout・測定を含むfull no-prep | 連続RZ/CX等のlogical価格。状態準備・FT synthesis・noiseを含めない。M1保存価格は不使用 |

[horizon SOURCE][HORIZON]、[強baseline SOURCE][STRONG]、[Yoshida schedule SOURCE][Y4]

**未確定で、ユーザーの採択が必要な項目は次の3群に限る。**

1. **元H/state入力**：H_physとH_srcのどちらをtargetとするか、旧DF12と同じ元h/g・MOへのbinding、そのHの保存ground state。保存stateがなければ、同定済み元積分からのstate solveを次回の明示承認範囲へ含めるか。積分再計算・新DF生成で不足を埋めることは本仕様に含めない。
2. **energy/alias条件**：上記RPE候補の正式採否と、事前の絶対energy区間I_E。既存decoderはm0のprincipal phaseから始めるため、I_Eを単一の初期aliasへ結合し、`phi=-δE`、`E_hat=-phi_hat/δ`を同じ分枝で使う。必要なら事前既知shift E_refとancilla相対位相補正を全armへ同じく適用するが、その追加はユーザー採択前に実装しない。M1の[-10,0]区間を黙って移さない。
3. **誤差・失敗配分と証拠の強さ**：各roundのmodel/PF/RTE/stat/implementation/reserve予算、numerical allowanceの定義、条件付きpilotか保証付きclaimかを採択する。旧(.02,.02,.36)はβ=.4を使い切り、元H/modelとimplementationへ新予算を割けないため、そのまま採用しない。新しい配分数値は本報告で作らない。

既存方法の受入条件は、全m・全armで

\[
\beta_{\mathrm{model},m}+\beta_{\mathrm{PF},m}+\beta_{\mathrm{RTE},m}
+\beta_{\mathrm{stat},m}+\beta_{\mathrm{impl},m}+\beta_{\mathrm{reserve},m}
\leq\beta_{\mathrm{RPE}},\qquad
\sum_{m,b}\alpha_{m,b}\leq .05
\]

である。reserveはAPIの自動fieldではなく未使用余裕として確保する。uniform失敗配分なら既存方針`α_m,b=.05/[2(M+1)]`を事前採択する。これは各armの量子測定失敗予算で、価格MCのCI予算ではない。B2・baselineで同じ配分規則を使い、deterministic armのRTE誤差は0として記録する。unused予算の再配分を行うなら、その規則も結果前に固定する。[既存実半径会計][ACCOUNT]

strict nearest-branch条件`β_RPE<π/3`、元H stateでの全round共通energy参照、正の実半径、aliasの外部条件を併用する。元H groundを仮定してもPF固有状態とは限らず、基準半径1は仮定しない。既存V2の条件付き成功やM1 `not_certified`を、このB2の95%達成証拠へ移さない。[既存branch SOURCE][DECODER]、[既存V2][V2]

## 3. 既存コードを直接使う最小差分

| 接続 | 直接再利用する版・関数 | 必要差分だけ |
|---|---|---|
| 入力 | [旧B2 SOURCE `2978e2f…`][B2]のsnapshot/bit-order規約、[M1 SOURCE `8a13539…`][MATCHED]の明示original-H/state設計 | 新しい薄い入力adapterに、旧DF12と別指定元h/g/MO/stateを渡す。M1 rank10専用loaderのgateを外さず、B2 DF-state用`_target_signal`も変更しない。入力生成・state fallbackなし |
| 分割・finite law | B2 `_prepare`、`_preparation_eigensystems`、`_apply_outer_step`、`_random_signal_record`、既存`make_rte_config`/finite moments | r4/K2、generation first3を固定。固定Tに束縛したrunnerから必要kernelだけを呼び、t_m/Q_mを引数で渡す。元H stateをDF固有状態として検査しない |
| 信号分解 | B2のideal-tail/finite polynomial、[M1独立signal validator][SIGNAL]のCAR/paired-polynomial参照、AX independent occupation参照 | M1 validator自体はLD0・faithful identity専用なので無変更での直接適用不可。既存prefix sweepで挟み、同じ元H stateに対するoriginal exact、DF exact、PF exact-tail、finite corrected/raw、implementationを別recordへ接続。original/DF差をPF誤差へ混ぜない |
| B2 native | [`df_partial_s2_repeated`][REPEATED]の明示step requestsとevent sampler、[AX native][AXNATIVE]の`partial_native_from_step_requests`/wrapper、[StrongBuilder][STRONG] | law/event列は既存requestを保持。共通spin/fusion経路にはStrongBuilderの**明示prefix half-sweep schedule**を使う薄い組立adapterを追加する案。forward `[one_body,F0,F1,F2]`、reverse逆順、中央に既存controlled RTE sequence。halfの`scalar_time=0`、核＋抽出tail scalarは全体で一回。新Gaussian/event/control合成法は書かない |
| Full S2/Y4 | `prepare_xdf_basis`、`StrongBuilder.sequence`、`c1_stage2_native.ordered_schedule`、`ax2a_control_plan.deterministic_control_plan` | S2専用`build_strong_df_baseline`をY4として流用しない。Y4は既存`formula='4th'`の正負3段scheduleをStrongBuilderへ渡す薄いadapter。負時間・chronological順・scalarを保持。samplingなし |
| RPE/N/価格/総和 | [M1 `account_radius_round`/`aggregate_registered_rounds`][ACCOUNT]、`wrap_controlled_no_prep`、`transpile_and_measure_cost`/既存phase保持compiler、metric accumulators | B2 context/lawと全roundを結合するprovider callback。`observed_signal_bounds`はM1 schema専用のためfield翻訳だけ追加。`rho_obs=A*(rho_PF_lb-eps_corrected)`を一度適用し、X/Y Nと価格を接続。両軸の同trajectory費用を先に加重してpaired SEを保持 |

spin-aware loweringをB2の決定論prefixとbaselineに共通利用し、同じfactorsを投影・切捨て・再順序化しない。random DF eventの既存basis規約も保存する。共通half-sweep adapterとAX旧builderの作用一致は次回の科学検証対象であり、今回静的読取だけで同値性PASSとはしていない。

synthetic検査は既存 `test_pr2_matched_accuracy_m1_execution.py`、`test_df_partial_s2_repeated.py`、`test_o1_m1_resource_bridge.py`、`test_o1_m1_circuit_signals.py`、`tests/management/test_research_c_c1_h4_strong_baseline.py`、`test_rpe_four_round_phase_validation.py`を再利用する。追加は非零prefix、正負Y4順序、scalar一回、同一lawのX/Y、半径の二重減衰防止、欠測round拒否に絞る。今回testは起動していない。旧研究runner・grant・SOURCE lockを書き換える必要はない。

## 4. 次の限定科学実行の範囲とGO/STOP

| 段階 | 実施する範囲 | GO/STOPと現在の状態 |
|---|---|---|
| 入力gate | 旧DF12は上記固定blob、元h/g/MO/stateは同じ入力来歴へ結合。数値評価前にstate roleとbasis/JWを固定 | **現在STOP**。原本未取得をM1 stateやDF stateで補わない。必要な元H stateが保存されていなければ、明示承認後にその元Hだけのsolveを行う範囲を別途指定 |
| 接続実装・synthetic | 上表adapterを最小実装、既存testを再利用し、新SOURCE commitを固定 | 技術作業で解決可能。分子inputを使うspin構造検査・native比較はsyntheticへ紛れ込ませず次段へ |
| signal-only資格 | 承認後にhorizonを確定。B2一つとbaseline8 armについて全required roundの元H/DF/PF/finite信号・normalization・半径・phase/branch・必要Nを計算。元H refを全armで共有 | 各armは全roundの事前予算・alias条件・正半径・実装整合を通過した場合だけ資格あり。B2固定r4/K2不適合なら**B2 pilotをSTOP**し、r/Kやδを再探索しない。baseline不足も隠さず報告 |
| native correctness | exact spin構造、独立occupation/CAR参照、B2半sweepとAX規約、S2/Y4 signed stage、controlled相対位相、X/Y wrapperを確認 | 既存閾値を別inputへ無条件移送しない。新SOURCEのprogram-scale allowanceと既存参照の適用範囲を事前指定。失敗時にglobal phase alignment、spin projection、dense UnitaryGate fallbackでPASSにしない |
| full価格 | 資格を通ったarm/全roundだけ直接whole compile。B2は初期案**32 fresh IID trajectory/round**、X/Yは同じevent列。各baselineは決定論2軸/round。6指標を保存 | 旧標本pool、旧単価、中央RTE価格proxyによる欠測補完なし。追加MC・自動retry・resampleなし。32標本は条件付き平均/SEのpilotで、希少eventや母平均順位を認証しない |
| 会計 | 各指標で`Σ_m Σ_b N_j,m,b E[C_full,no-prep]`。paired X/Y sample SE、全round欠測・context・Σαを確認 | 既存`aggregate_registered_rounds`を使用。欠測を0にしない。価格精度未資格なら点推定だけと明示し、総CI・95%実達成・一般優位へ昇格しない |

実行順はB2のsignal資格を先に確認し、最初の不適合roundでSTOPする。B2が全round通過した場合だけbaseline資格、次に価格へ進む。model/PF比較の参照信号が0で位相を定義できない場合や、normalization/実半径/Nが非有限・underflowになる場合もSTOPし、残roundの欠測を成功で埋めない。これは追加の最適化や再試行を始める許可ではない。

価格MCの実行指定案は、既存`iter_sample_rte_events`のPCG64、round stream seed `20261106030000+m`、trajectory順序固定、重複draw保持。これは将来の実行仕様案で、seedの消費・samplingは今回0。X/Yは同一列、round間は別stream。量子shot列と価格MC列を共有せず、古典標本数をNへ掛けない。指標別主結果はRZとCX、depth/sizeも同じ記録から表示する。全round価格SEは独立round streamという仮定を明記し、sample SEをCIと呼ばない。

**今回完了したのは、固定入力の所在、SOURCE/hash、変換規約、reuse境界、有限armと実行手順の確定である。** 全readyの科学契約とはしていない。次に必要なものは上記3群の入力・科学条件であり、別の一般的な調査や準備書類一式ではない。これらが埋まり明示承認されたら、最小実装・synthetic確認・SOURCE固定・同じ限定科学検証をまとめて進められる。SOURCE固定は再現性のためで、承認証明書や承認台帳を新設する意味ではない。

## 5. ユーザー判断・レビュー・終了

**ユーザー／元入力の保有担当**：第一に既知HDF5 SHA `a3c88f6a…` または同一元h/g/MOの保存物と、対応する元H ground stateの来歴を提示・回収する。保存stateがなければ、元積分を同定した後、その元Hのみのstate solveを許可するか判断する。M1への入力変更やDF再生成は今回の採択に含まれない。原本が得られない場合、旧DF12に対する今回pilotはSTOPのままにする。

**ユーザー**：§2のRPE候補、energy区間／shift、誤差配分・numerical allowance・条件付きclaimを採択し、必要なら上記の限定実装・科学実行範囲を明示承認する。一つの固定範囲への明示承認後に、通常の接続・debugごとに承認を取り直す運用はしない。入力や科学条件の実質変更は別途判断する。

**GPT詳細科学レビューは現時点では不要、開始していない。** 原本不足はレビューで補えない。条件付き数値pilotは既存の信号分解・誤差disk・実半径Hoeffding・branch条件・会計を適用できる。元Hへの厳密なmodel/数値誤差上界を新たに正当化する、非PF固有状態・多位相信号から既存十分条件を超える成功定理を採択する、または新しいalias/状態同定仮説が必要になった場合だけ、その論点に限定してレビュー開始の別承認を求める。単なるinput hash不一致やadapter変更のたびにレビューを要求しない。

**Codex**：このREPORT一つを専用branch `codex/track-a-b2-original-state-spec-20261106`へGitHub連携でcommit公開し、remote SHA・tree・REPORT本文とローカルの一致を確認して**mandatory STOP**する。通常pushの既知の認証不足を理由に共有Git設定・認証設定を変更しない。元SOURCE/input/raw/result/grant/marker/STOP、O1 M1終了提案、O1 M2保留は不変更。

検査範囲：NPZのfile hash/member/header/文字列metadata、SOURCEのstatic readと保存source hash照合、固定GitHub SOURCE/receiptの取得、文書の参照と変更範囲。科学ライブラリをimportせず、配列decode・h/g再構成・operator比較・state solve・PF/RTE simulation・horizon/N/energy/phase/総費用算出・compile/sampling・科学レビューは0件。探索は既知の入力・manifest・SOURCEと当該fixed treeに限定し、全projectsや他ユーザー領域を横断していない。

[PRIOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e17f4d39ec3733c5e1f58eda6871d69b06a651c/docs/research/track_a_b2_total_rpe_connection_20261106/REPORT.md
[DFNPZ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf5d3b2405b2bec4f88dd0589b75dd737b13e052/artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz
[GEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c644925b50587072784846df09bf02c39e8453e1/src/trotterlib/pr2_s0_s1_validation.py
[DFGEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c644925b50587072784846df09bf02c39e8453e1/src/trotterlib/df_hamiltonian.py
[GEO]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c644925b50587072784846df09bf02c39e8453e1/src/trotterlib/chemistry_hamiltonian.py
[S0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf5d3b2405b2bec4f88dd0589b75dd737b13e052/artifacts/pr2_s0_s1_validation/2026-09-28/pr2_s0_validation_v1.json
[V0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e17f4d39ec3733c5e1f58eda6871d69b06a651c/docs/research/pr2_v0_input_recovery_audit_bf5d3b2_20260928.md
[OF]: https://github.com/quantumlib/OpenFermion/blob/v1.6.1/src/openfermion/chem/molecular_data.py
[OFPY]: https://github.com/quantumlib/OpenFermion-PySCF/blob/v0.5/openfermionpyscf/_run_pyscf.py
[B2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/pr2_matched_accuracy_m1_execution.py
[AXINPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e17f4d39ec3733c5e1f58eda6871d69b06a651c/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/input_reference.json
[M1NPZ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz
[M1GEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h4_molecular_inputs.py
[M1RECEIPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e17f4d39ec3733c5e1f58eda6871d69b06a651c/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json
[MATCHED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a13539e51bb113e82cb86ddb608e2e6e27786ca/src/trotterlib/o1_m1_matched_split.py
[SIGNAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a13539e51bb113e82cb86ddb608e2e6e27786ca/src/trotterlib/o1_m1_circuit_signal_validation.py
[ACCOUNT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a13539e51bb113e82cb86ddb608e2e6e27786ca/src/trotterlib/o1_m1_resource_bridge.py
[HORIZON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a13539e51bb113e82cb86ddb608e2e6e27786ca/src/trotterlib/rpe_target_round_horizon_validation.py
[STRONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a13539e51bb113e82cb86ddb608e2e6e27786ca/src/trottertracks/c1_h4_strong_baseline.py
[Y4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a13539e51bb113e82cb86ddb608e2e6e27786ca/src/trottertracks/resource_applicability/ax2a_control_plan.py
[REPEATED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/df_partial_s2_repeated.py
[AXNATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_native_df_v5.py
[DECODER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a13539e51bb113e82cb86ddb608e2e6e27786ca/src/trotterlib/rpe_four_round_phase_validation.py
[V2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e17f4d39ec3733c5e1f58eda6871d69b06a651c/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
