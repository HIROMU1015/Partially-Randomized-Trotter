# M1：同じ物理分割でのtail辞書比較の準備

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。本書は固定SOURCE・保存metadata・既存結果を用いた静的調査であり、新しいH/state/residual、simulation、sampling、shot、回路、資源集計は生成していない。M2とは独立し、M1では一つの比較内で元H・HD・HRを固定する。候補や数値条件の正式採択はしない。

## 1. 固定資料と今回の追加確認

基点は[O1 preflight `aeb9d1847f9d500f2389db12b384bac6087117a9`][O1]。同directoryの4文書、[3環境統合台帳・再利用map `3571477093441ac4852166fe829f8ccc5db5ec4f`][I]、[文献調査 `e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8`][S]を継承する。共通AGENTS、正式共通指示2026-10-25、repository AGENTS/PROJECT_MAP、研究概要を確認した。旧資料や既存台帳は更新しない。

前回から進めた点は、①F1/F2の入力payloadと公開metadataの区別、②F2価格にstate preparationが含まれること、③公開H4 snapshotのmember/headerとstate生成由来、④P0専用APIと分子用APIの接続差である。科学値を再解析したものではない。

## 2. 保存入力候補と得られる情報

| 候補 | 固定入力・SOURCE | 同一物理分割の状態 | 情報価値／現在の不足 |
|---|---|---|---|
| P0/F1・F2合成fixtureのSS対PP | F1 SOURCE `8192530b17d6bc4af195c734ff43a6e78c506942`、F2修正版SOURCE `9468c7089727f5648d9f59cddbaebe0a42f5f0a0`。保存Bのphysical/arms/bundleを共用 | 元契約でH、D、R、scalar、occupied(0,1)を固定。SS/PPは同じ2残差fragmentの異辞書 | full operator/control/meanの既存検査を再利用できる。分子ではなく、ground-state energy/RPE全体の実証でもない。raw payloadのSERVER 2上の取得先は未結合。旧価格はprep込み |
| **H4-A：T_eff / V_squareのP対S** | 入力生成SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`、保存anchor `4591b8d75a2a6a55c74a51225f0c61648023c8ee` | 既存matched-partition案A。共通T_effを保持し、同じV_squareをglobal collected Pauliまたはrotated Z/ZZで実装する候補 | 分子・元H stateとの接続を問える。既存DFのone-body保持／fragment LD0に近く、追加接続が比較的少ない。T/V別数値一致と元Hへの誤差転送は未検証 |
| **H4-B：T_bare / V_normalのP対S** | H4-Aと同じsnapshot。案Bの入力素材も保存済み | 共通T_bareを保持。S側はV_squareに一体収縮D̂を含めて同じV_normalを表す必要がある | bare一体／二体分割を問える。D̂のI/Z辞書接続が追加で必要。DFのone-bodyをbareへ置換するだけではHが欠落する |

H4-A/Bは同じ保存物から考えられる**別の未採択候補**である。AとBを同じ分割としてpoolしない。各候補内の最小比較はP/S二端点であり、混在内点、new selector、rank/PF探索は追加しない。既存H2四方式やH4のPauli prefix27対DF prefix1の価格を、このP/S比較の結果に流用しない。

## 3. P0の保存物・固定条件・公開範囲

[B入力要約][BINPUT]と[F2修正版結果][F2REPORT]に従い、元の合成fixtureは4 mode、2粒子、occupied(0,1)、LD1、density-square残差2fragment。geometry/AO basis/分子DF rankは非該当。F1はT=1/5、q2、δ1/10、r2、h1/20、K2、F2はT=1/5、q1、δ1/5、r2、h1/10、K0。F1の保存K2 lawをF2のK0 lawとして使わない。

`reference.original_operators`はDを`Σ d_p n_p`、Rを2つの回転density-squareの和、Hをscalar I＋D＋Rとして構成する。stateは`reference.signal`でoccupied計算基底の対角要素を読むもので、元H groundの資格を与えていない。[F2 reference SOURCE][REF]

| 保存B内の相対path | 公開された科学入力SHA256 | SERVER 2で今回取得できたもの |
|---|---|---|
| `inputs/physical.json` | `d796385d5bc5f52fa2fbffe9d4f327b738055dae0420f4d255b40d862ea14658` | 公開hash・fixture記述・SOURCEのschema |
| `arms/SS.json` | `e08985a9b560d49f7e6694d3d36c23e462bb1f996a9dd8f542e589d346e10358` | 公開hash・arm metadata |
| `arms/PP.json` | `4c036eb36262930669650e356accdc129aab0ec5f14bbbdac144fd55e1009a71` | 公開hash・arm metadata |
| `handoff/f1_bundle.json` | `9d87488f673551ace8e13456bc64cb73a5384ba56c42cac28f345fb270c6e4e0` | 公開hash・loader schema |

これらのraw payloadはB報告ではprivate原本として保全されており、選定固定Git treeの公開artifactにはない。取得資料にSERVER 2で許可された私的原本の具体pathもない。**原本の不存在とは断定せず、今回未取得**とする。元Bを再生成して穴埋めせず、他サーバーSSH・広域private走査もしない。再利用するなら利用可能な保管場所の確認が先である。F2のlaw/price/mean要約は公開されているが、それだけで全physical/arm入力が復元可能とは認定しない。[B handoff][BHAND]・[F2 result][F2RESULT]

### F2価格に関する新しい適合性の注意

F2修正版の[EXECUTION_REPORT「Actual 1-shot資源比較」][F2REPORT]は、価格が**state preparation X0X1、ancilla操作、controlled evolution、readout、terminal measureを含む**と明記する。固定SOURCEの`circuits.hadamard_wrapper`も`for p in spec.occupied: qc.x(p)`を実行し、metadataを`state_preparation_included=True`にする。[circuits.py][CIRCUITS]

したがってF2保存値は、今回の`C_full,no-prep`へ直接代入できない。例えば保存RZ表示はSS 70.825335、PP 35.445140だが、これは元のprep込み・固定K0条件における引用である。prep単独価格を差し引いて新しいno-prep価格・順位を算出していない。full compileは非加法的なので、その単純減算を将来の方法にも採らない。

F2の全登録operator/signal/finite corrected mean PASSは維持する一方、`N=null`、`weighted_costs=null`、統計資格未成立を維持する。旧qualified entryのSTOPは、別のresource-only実行成功によって解除されていない。前回O1文書も修正していない。

## 4. 公開H4 anchorの実物とstate由来

保存commitは`4591b8d75a2a6a55c74a51225f0c61648023c8ee`。以下は実際にSERVER 2の固定Git objectから取得可能だった。snapshotはbytesのSHA256とZIP/NPY headerだけを確認し、数値配列payloadのdecodeやoperator再構成はしていない。

| 固定path | 役割・取得確認 |
|---|---|
| [`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`][SNAP] | 2,157,448 bytes。SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`が保存receiptと一致 |
| [`…/data/receipt.json`][RECEIPT] | geometry、rank policy、state/hash、旧prefix、full-H/ground診断の保存結果 |
| [`…/data/pauli.json`][PAULIJSON] | 保存full Pauli係数と規約。これを旧prefixのまま新HRと呼ばない |
| [`artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json`][PMODEL] | Pauli側full-H/reference/state descriptor。blob存在とschemaを確認 |
| [`…/df_native_model.json`][DMODEL] | DF側descriptor。同一raw state/hashを持つという既存matched設計の照合を継承。大きい行列fieldを新計算に使わない |

NPY headerにあった主memberは以下。shape/dtypeはmetadataであり、今回求めた科学的数値ではない。

| member | shape | dtype | 用途／取り違え防止 |
|---|---|---|---|
| `one_mo`, `eri_chemist`, `nuclear_repulsion` | (4,4)、(4,4,4,4)、scalar | f8 | 同じ元MO積分・chemist ERI・核scalar |
| `spin_one`, `spin_two` | (8,8)、(8,8,8,8) | f8 | 入力SOURCEのindex/1/2規約を併せて使う |
| `df_one_body`, `df_lambdas`, `df_g_matrices` | (8,8)、(10,)、(10,8,8) | c16、f8、c16 | 保存DFの一体項と10 fragment。再DFせず利用する素材 |
| `state`, `DF_state` | 各(256,) | c16 | **別々のHから作った別field**。今回の元H state候補は`state` |
| `non_DF_matrix`, `DF_matrix` | 各(256,256) | c16 | 別の保存full operator。厳密同一と仮定しない |
| `physical_qiskit_indices` | (36,) | i8 | Nα=Nβ2の保存sector。個別eventを無断射影する許可ではない |

`mo_coeff`、FCI/energy関連の保存fieldもあるが、E₀/gap/古典全stateを選択algorithmへ無料提供しない。診断に保存vectorを利用する仮想数値検証と、実装algorithmの情報アクセスは区別する。

[入力生成SOURCE `h4_molecular_inputs.py`][H4INPUT]は、同じMO積分からPauli fullとDF fullを作り、`canonical_ground(full, indices)`を`state`、`canonical_ground(df_full, indices)`を`DF_state`として保存する。receiptの共通`state_hash`は`8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`、独立DF state hashは`bfdc4948b4c08230d3ab7a3f11d4a3990875f6cf3423eea2d15a72a2d0ff8667`。近似DF groundへ置換しない。

保存条件はlinear H4、隣接1Å、STO-3G、中性singlet、real orthonormal MO4、interleaved spin8、Nα=Nβ2/sector36。**DF rankはtol-only1e-8で返った10、requested rank=None**。legacy DF12や研究C numberCDF8とは別入力である。receiptにはfull-H一致誤差`8.462237051574415e-15`（許容差1e-9）と元H state residual`1.813556937761976e-15`（許容差1e-10）の過去PASSがある。これは既存数値診断の引用で、数学的厳密groundや新しいT/V別一致の証明ではない。

## 5. 同一分割を保つための対応

記号は既存[matched-partition設計][MATCHED]を継承する。`c_N I+T_bare+V_normal`と`c_N I+T_eff+V_square`の関係には一体収縮D̂があり、`T_eff=T_bare+D̂`、`V_normal=D̂+V_square`。保存丸め・tolを含む演算子では無条件の厳密等号へ格上げしない。

| 項目 | H4-A内で共有するもの | H4-B内で共有するもの |
|---|---|---|
| 保持HD | 同じT_eff operator、順序、half時間、保持入力program候補 | 同じT_bare operator、順序、half時間、保持入力program候補 |
| random HR | 同じV_square。Pはglobal Pauli、Sは10 fragment由来のrotated Z/ZZ | 同じV_normal。S側にはD̂由来I/Zを含める。D̂の材料は`df_one_body−spin_one`で、生成SOURCE規約との対応も検査対象 |
| scalar | c_N、tail identity、一体Gaussianのphaseを共通方針で一度ずつ | 同左。D̂のidentityも対応させる |
| state | 保存`state`に対応する元H ground仮定。`DF_state`やcandidate groundに変更しない | 同左 |
| sector／register | interleaved spin、Qiskit little-endian、system8＋ancilla-lastを双方で共有 | 同左 |
| 同じmean | 同じT/V、r/K/q、paired cutoff、保持operator/近似なら同じ補正済み名目meanを比較 | 同左。D̂が抜ける場合はこの条件を満たさない |

共通Gaussian一体block、共通Pauli-PF保持block、exact-Tの信号診断は既存設計の未採択選択肢。保持operatorが同じでも入力programが異なれば価格差に交絡が残る。できる場合は同じ保持programを双方へ共有し、full回路の結合差は残したまま報告する。

P0のfaithful identity tailとDF library既定のidentity抽出は異なる。H4で使う共通identity/scalar方針は別途選ぶ。Pauli full constantは電子identityを含み、DF constantは核scalarを表すので、数値fieldをそのまま相互コピーしない。物理分割を変えて解決せず、同じ方針を両辞書へ対応付ける。

また、保存Pauli-HとDF-Hの小さい既存差を、今回の元H条件から消さない。次の実行前に、共通targetの定義と数値DF差のallowance/転送を決める。**近似DF energyを新targetにすることや、未許可の残差補償方式をM1へ追加することはしない。** 同じ元H/HD/HRを説明できない場合、その候補のM1比較はSTOPする。

## 6. 最小paired比較の意味と未決定事項

最小の科学scope候補は、**H4-AまたはH4-Bのどちらか一つについて、P/S二端点のfull H・T・V・辞書和・scalar対応だけを確認する処理**である。時間・q/r/K・compiler・shotを使わないoperator確認に留められる。その先にmean/価格/全RPEを加えるかは別の判断であり、今回実行していない。

「paired」は同じ保存入力と共通物理splitを対にする意味である。辞書間でlaw/support/角度が変わるため、同じseedだけで価格MCのpaired標本にはならない。価格標本間のcouplingや共分散を採る場合は、その条件を将来明示する。新sampling法は今回設計しない。

正式選択待ちは、P0のraw回収を含む技術確認を求めるか、H4-A/Bのどちらか、共通保持block、identity方針、target/allowance、初段scopeである。signal以上へ進むなら既存T=.8、q8/r4/K4等は**保存候補**として参照できるが、新条件へ自動採用しない。必要な関数・最小接続・全RPEへの条件は[M1_EXECUTION_READINESS.md](M1_EXECUTION_READINESS.md)に記す。

[O1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/aeb9d1847f9d500f2389db12b384bac6087117a9/docs/research/o1_hamiltonian_partition_preflight_20261106
[I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105
[S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
[BINPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc7378ab34015287215bcf4ce52147e46d94ecd9/docs/research/pr_algorithm_improvement_phase2d_binding_execution_20261023/INPUT_AND_LAW_SUMMARY.md
[BHAND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc7378ab34015287215bcf4ce52147e46d94ecd9/docs/research/pr_algorithm_improvement_phase2d_binding_execution_20261023/F1_HANDOFF.md
[F2REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc7378ab34015287215bcf4ce52147e46d94ecd9/docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md
[F2RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc7378ab34015287215bcf4ce52147e46d94ecd9/artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/RESULT_SUMMARY.json
[REF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/reference.py#L53
[CIRCUITS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/circuits.py#L198
[SNAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz
[RECEIPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json
[PAULIJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/pauli.json
[PMODEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json
[DMODEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/df_native_model.json
[H4INPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h4_molecular_inputs.py#L57
[MATCHED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/docs/research/h4_df_pauli_matched_partition_static_design_20261102.md
