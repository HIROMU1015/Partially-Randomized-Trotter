# M2：保存圧縮coreと元Hamiltonianの対応

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。**保存coreは存在し、新fitを行わずに接続準備を進められる。core＋完全残差の科学性能・実行資格は未検証。** 本書は静的SOURCE・保存metadata・file bytes/headerの技術照合であり、配列の数値decode、Hamiltonian/state/residual生成、再解析・再ランキングは行っていない。

## 1. 固定資料と確認範囲

調査基点は[O1 preflight `aeb9d1847f9d500f2389db12b384bac6087117a9`][O1]。その4文書、[3環境統合 `3571477093441ac4852166fe829f8ccc5db5ec4f`][I]の台帳・再利用地図・Gap・残検証、[文献調査 `e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8`][S]を引き継ぐ。共通/repository AGENTS.md、PROJECT_MAP.md、研究概要全文、正式共通運用指示2026-10-25の実行・再利用・STOP条件を確認した。今回の複数agent分担は最新ユーザー指示による明示許可の範囲で、M1とは別worktree・branch・文書とする。

以下では**文書収録commit、fit/科学実行SOURCE、結果commit**を分ける。O1基点のtreeには関連JSONやSOURCEがあっても一部NPZは収録されていない。元の結果commitを参照すると下記coreと入力を取得できた。全branch・全rawの棚卸しはしていない。

確認方法は `git show <固定commit>:<path>`、ZIP central directory、NPY headerのshape/dtype、file bytesの比較のみ。科学moduleをimportせず、NumPyによる配列load・積分再構成・norm計算は0。4 core＋共通入力の計5 fileをGitHub接続からbase64で取得し、固定Git blob bytesと一致を確認した。Headerのshapeは内容の科学的適合性を証明するものではない。

## 2. 共通の元物理H

既存研究Cの対象は保存直鎖H4、隣接1.00 Å、STO-3G、実直交空間MO、interleaved spin、Nα=Nβ=2（sector36、full Fock256）。これは**保存条件の引用**であり、今回の正式検証条件として採択していない。

| 保存入力 | 固定所在・今回確認 | 内容と役割 |
|---|---|---|
| 元のh/g/核scalar | [結果 `aced1647b81e8d28091058cdc7b6e3e1ee54896a` の input.npz][input]。SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` | `h:<f8 (4,4)`、`g:<f8 (4,4,4,4)`、`constant:<f8 ()`。GitHub取得bytes一致。chemist `(pq|rs)`、核scalarは一回 |
| 元P-H4 packet | [CDF7結果 `794875af24cb954985ac5e43fc799421f5d8736a` の P-H4.npz][ph4]。既存契約記載の原snapshot SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` | `one_mo`/`eri_chemist`/`nuclear_repulsion`等のheaderを確認。元来歴は[既存CDF7契約][cdf7contract]に `86b8b002704aced8ca239414080f53793de81658` のsnapshotをbyte同一保全したと記録。既存抽出h/gとの数値再照合は今回していない |
| 物理Hの保存ground | [aced…/C/ground_physical.npz][gphysical] | headerは`energy:<f8 ()`、`sector_state:<c16 (36,)`。原物理Hに対する既存数値reference。今回residual・eigenpairを再評価せず、厳密groundの証明にはしない |
| candidate Hの保存ground | [aced…/C/ground_candidate.npz][gcandidate] | 同じheader形でも別Hのground。今回仮定する元Hの厳密groundへ差し替え不可 |

演算子規約は原SOURCE `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4` の[c1_stage1.py][stage1]に従う。`corrected_kappa(h,g)=h−0.5 einsum("prrq->pq",g)`。`physical_sector`はspin liftとtwo-body係数1/2を使う。元hをcorrected κとして渡したり、fit後gからκを無断再補正すると、保存coreとは違う演算子になる。

## 3. 実取得できたcore候補

順位は付けない。いずれもJは一般実対称signed、Uは実直交という保存仕様で、旧実行の数値gateを今回再実行していない。

| 候補 | fit/科学実行SOURCE → 保存結果pin・export | headerとbytes識別 | 今回利用できること／留保 |
|---|---|---|---|
| **通常CDF7** | SOURCE `c13d6f82a22d497b3334eaaa40d8430a5010f70f` → [結果 `794875af24cb954985ac5e43fc799421f5d8736a`: `artifacts/research_c_cdf7_onefit/2026-10-30/run01/fit/final.npz`][cdf7] | J/U=`<f8 (7,4,4)`、x/jac=`<f8 (112,)`。SHA256 `4e4939d1380386ebb98c3b39a13954cbfc3148e35fb8e2cfd3cf536bd0b57ff4` | 元Hの近似を与える7 frame core。新fit不要。既存DF budget PASSはengineering guard付き数値pilotで、M2の残差λ・元H stateのPF/RTE・全RPE費用ではない。旧own-DF PR/native結果は完全残差接続へ移送しない |
| **PR-aware CDF8 update005** | SOURCE `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4` → [結果 `aced1647b81e8d28091058cdc7b6e3e1ee54896a`: `artifacts/research_c_pr_integrated/2026-11-02/run01/update005_candidate.npz`][u005] | J/U=`<f8 (8,4,4)`のみ。SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e` | 同じ元h/g/constantに対応し、cleanなJ/U入力がある。既存同candidate強Dの部品を参照可能。既存性能は元順LD2＋candidate tail、own-DF ground。M2とは別task |
| **PR-aware CDF8 update064** | SOURCE `025d495166311a58ec9c24c43f07ed6e40f65f42` → [結果 `ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6`: `artifacts/research_c_u_step_control/2026-11-03/run01/terminal.npz`][u064] | J/U=`<f8 (8,4,4)`のみ。SHA256 `53d8f5027ce9e61bc72f6943c0268666bfcf87515154850a9a99326687b26ed3` | `terminal.npz`と同commitの`update064_candidate.npz`がbyte同一。`physical/update064.json`のcandidate hashも一致し、候補名を推定で付けていない。64更新後の保存端点であって同時一次収束・最適DFの認定ではない |
| **RC2比較端点（CDF8）** | 元fit SOURCE `e2f7dfd44357510d54ab9c6c9c5032e0072ab376`、出力修復は別SOURCE。 [修復結果carrier `f6bb336b41cddface5647c6b1acf23c330f32637`: `artifacts/research_c_common_convergence/2026-11-01/run01/rc2/final.npz`][rc2] | J/U=`<f8 (8,4,4)`、x/jac=`<f8 (128,)`。SHA256 `d6cb5d56dfc6a83090a6ad1bc1207fdaee62a14cf120542fcb5a7d116644a798` | RC-DF型の既存正則化対照として参照可能。原25-callのserialization失敗と後続JSON復旧を分離。fitのsolver/一次条件は未資格で、最適RC-DFを代表させない |

各候補の旧結果の解釈は[通常CDF7報告][cdf7report]、[統合update005報告][integratedreport]、[update064生成報告][u064report]、[RC2出力復旧報告][rc2report]へ戻る。候補間の保存性能を今回再集計・再ランキングせず、新しい最適候補も選んでいない。RC2をRC-DF一般の代表的勝者、PR-aware端点を最適解とは扱わない。

### core再構成に必要な情報

J/Uだけでは核scalarと一体項を定義できない。原SOURCEの`SectorCAR.generators`は、**元gからのκ＋核scalarをbase**、各frameをdensity-productの`0.5 Σ_ij J_ij N_i(U)N_j(U)`として構成する。[c1_molecule_extension.py][molecule] M2で全保存coreをDに置くなら、このbaseと**全frame**がH_coreである。従来LD2の保持部だけをH_coreと呼ぶ場合とは別の物理分割になる。

`physical/update005.npz`も同原commitに存在し、J/U、`generators:(9,36,36)`、`H:(36,36)`、`tensor_residual:(4,4,4,4)`を持つことをheaderで確認した。これらは診断referenceであり、J/Uから回路を構成する際の無料dense-H oracleとしては使わない。`tensor_residual`はSOURCE上で**g_fit−g_original**。M2のH−H_coreへ正符号で流用すると向きが逆になる。

## 4. 候補間の情報価値と必要作業

| 比較候補 | 得られる情報 | 必要な接続・区別 |
|---|---|---|
| CDF7を1 coreとして検討 | 少ないframeのcoreで完全残差負担を支払えるか | rank7/general signed J、base＋7 frameのD順序、完全残差adapter。旧rank8固定loopを引数だけ変えて再起動しない |
| update005または064を1 coreとして検討 | 保存済みの同candidate強D・native部品を活かし、元Hへの残差補償を隔離できるか | rank8の既存general-K部品は適合しやすい。ただしLD2のtailを完全残差へ置換し、元H stateと新PF/RTEを評価する必要がある |
| RC2を別対照候補として検討 | 同じ元入力で正則化の異なるcoreと残差の交換を見る | 元fit/出力復旧の来歴・未収束を保持。同じcore候補を強D/既知法へ開放。新RC-DF fitや収束探索を加えない |

最小科学比較ではユーザーが**1つの保存core**を選べる。4候補の全走査や再ランキングは必要条件にしていない。既存SOURCE・入力から接続可能性を具体化できたため、現状は「保存coreがないので新fit不可欠」というSTOPには該当しない。一方、どの候補でも元H state上のM2性能は未検証で、科学実行は未承認・未準備である。

[O1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/aeb9d1847f9d500f2389db12b384bac6087117a9/docs/research/o1_hamiltonian_partition_preflight_20261106
[I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105
[S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
[input]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz
[ph4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/794875af24cb954985ac5e43fc799421f5d8736a/artifacts/research_c_cdf7_onefit/2026-10-30/input/P-H4.npz
[cdf7contract]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/794875af24cb954985ac5e43fc799421f5d8736a/docs/research/research_c_cdf7_onefit_contract_20261030.json
[gphysical]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/C/ground_physical.npz
[gcandidate]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/C/ground_candidate.npz
[cdf7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/794875af24cb954985ac5e43fc799421f5d8736a/artifacts/research_c_cdf7_onefit/2026-10-30/run01/fit/final.npz
[u005]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/update005_candidate.npz
[u064]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6/artifacts/research_c_u_step_control/2026-11-03/run01/terminal.npz
[rc2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f6bb336b41cddface5647c6b1acf23c330f32637/artifacts/research_c_common_convergence/2026-11-01/run01/rc2/final.npz
[stage1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c1_stage1.py
[molecule]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c1_molecule_extension.py
[cdf7report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/794875af24cb954985ac5e43fc799421f5d8736a/docs/research/research_c_cdf7_onefit_execution_20261030.md
[integratedreport]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/docs/research/research_c_pr_integrated_execution_20261102.md
[u064report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6/docs/research/research_c_u_step_control_execution_20261103.md
[rc2report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f6bb336b41cddface5647c6b1acf23c330f32637/docs/research/research_c_common_convergence_export_repair_20261101.md
