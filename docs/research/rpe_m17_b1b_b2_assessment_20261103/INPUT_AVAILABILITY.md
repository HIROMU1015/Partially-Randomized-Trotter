# B-1bの固定入力可用性

2026-11-03 JST。静的調査のみ。新しい実H4数値計算・再解析・成分生成は0。
対象はlinear H4/隣接1.00 Å/STO-3G/保存legacy DF12、Bのphysical state、4電子sector70、
Partial L_D3、単一δ=.02、m17/q131072/r32/K2、extract_identity_phase、coefficient_atol=1e-12。

**結論：元の218成分の完全な保存入力は、確認した関連保存先では揃わない。**
固定Hamiltonian snapshotとSOURCEからの再生成経路はあるが、保存原本を直接読むB-1bとは異なる。
aggregate Aから係数や確率を逆算せず、不足分を推測で補っていない。

## SOURCEと結果の役割

| 役割 | Commit |
|---|---|
| 元library carrier | `4591b8d75a2a6a55c74a51225f0c61648023c8ee` |
| 元m17実行SOURCE / 結果 | `128e9a5bf2630cd0333ec5f69b81b25def78eb79` / `cd749066904d3f29b28b03c19ab69370372b157f` |
| Task A実行SOURCE / 結果 | `4479a1c9cf150250254e2c2650f70fb43841fed1` / `a5304864ec5e1412a36c31574bcb95fb13f80202` |
| B-1a実行SOURCE / 結果 | `9d27f931d6a572210eb206569806f96a46c2d094` / `f766a1f7fc0cd9131a21d0594da242865fe1cb0d` |
| B physical state | `e22d7718c47358ff650dbc04e8e72f6ab1c8bf41` |
| V2保存round・予算 | `78c49d0b5f1d71e63b5959e78813237923bbc648` |
| 候補Aの修正実行SOURCE / run02結果 | `d0b1da1b04c266a1f7abf8e43966ca6e289a8695` / `80616e6450a63c77d5e76276739a2322111adda4` |

[B-1a報告](../rpe_m17_b1_construction_precision_execution_20261103/REPORT.md)、
[入力境界](../rpe_m17_b1_construction_precision_preparation_20261103/B1_INPUT_BOUNDARIES.md)、
[元m17 runner](../rpe_m17_mean_validation_preparation_20261103/run_m17_mean_validation.py)を参照。
上記carrierのdf_rte_tail.py、候補A修正runner、B-1a報告はGitHub固定commitのblobとLOCALを照合。
Hamiltonian snapshotの公開blob/sizeとLOCALも照合した。数値payloadはロードしていない。

## 入力ごとの分類

`SAVED_FIXED_INPUT`は原保存値の直接取得、`RECONSTRUCTABLE_FROM_FIXED_SOURCE`は新しいSOURCE replay、
`PARTIALLY_AVAILABLE`は一部だけ、`NOT_AVAILABLE`は必要な原本を未確認という意味で使う。

| 必要な入力 | 分類 | 根拠・不足 |
|---|---|---|
| Hamiltonian snapshot | SAVED_FIXED_INPUT | carrierのNPZ。constant、one_body(8×8)、lambdas(12)、g_matrices(12×8×8)、metadata/hash。header確認のみ |
| ψ・sector indices | SAVED_FIXED_INPUT | B physical_state.npz。ψを再solve・再正規化しない |
| A_prob/A_coeff、F/R、元event/tail | SAVED_FIXED_INPUT | 元m17 NPZの7行列。全成分やbasis Uは保存されていない |
| 全218成分のID・並び | RECONSTRUCTABLE_FROM_FIXED_SOURCE | extract_df_diagonal_tailのfragment/support順sortとextraction_to_symbolic_rte_tailの順序保存。完全な原リストなし |
| 一部のID・sign・support・basis ID | PARTIALLY_AVAILABLE | B RESULTのconditional_events.componentsは選択2成分。A shotsは選択eventのID/順序。全定義ではない |
| 成分別coefficient/abs/sign | RECONSTRUCTABLE_FROM_FIXED_SOURCE | DF eta/lamからexact_df_diagonal_coefficients（実際はbinary64演算）を経て生成。原vectorは未保存 |
| 成分別sampling probability | RECONSTRUCTABLE_FROM_FIXED_SOURCE | coefficient_abs / extraction.rte_lambda_r。原vectorは未保存。λだけから復元不可 |
| 保存λ_R、component_probability_sum、component_count | SAVED_FIXED_INPUT | m17 RESULT contextに集約値あり。個々の確率やその順序の代用にはならない |
| 全basis ID、rotation descriptor、local gate matrix | RECONSTRUCTABLE_FROM_FIXED_SOURCE | DFBasisRegistry/describe_basis_change_operations。完全な原descriptor/local matrix bodyは未確認 |
| 既存basis hash/preparation hash/tail hash | PARTIALLY_AVAILABLE | 集約hashと選択basis IDがある。hashは欠けた入力本体を復元しない |
| 成分作用演算子P_i、sector basis U | RECONSTRUCTABLE_FROM_FIXED_SOURCE | SOURCEからbasis/diagonalを組み立てる新しい数値処理が必要。元全行列はNOT_AVAILABLE |
| cutoff・identity規則 | SAVED_FIXED_INPUT | SOURCEとm17 metadataで1e-12/extract_identity_phase。abs(c)>cutoffを保持、threshold後にidentityを抽出 |
| 確率再正規化規則 | SAVED_FIXED_INPUT | _validate_componentsのmath.fsum後、p_i/sumをchoiceへ。さらにchoice内のCDF正規化あり |
| prefix/tail分割の記録 | PARTIALLY_AVAILABLE | 元V2参照ABLGRIDにdeterministic [0,1,2]、randomized [3,4,5,6,7,8,9,10,11]。元m17の完全なpartition本文は未保存 |
| r32/K2のπ、τ、B | SAVED_FIXED_INPUT | 元m17 RESULT context.distribution/config。新生成せず取得可能 |
| θ64/cos64/sin64/logB64/normalizer64 | SAVED_FIXED_INPUT（B-1aで新生成された値として） | B-1a RESULT scalar_trace。元m17に保存されていた値とは呼ばない |
| samplerが構成するCDF・各bucketの質量 | NOT_AVAILABLE | 原実行時のCDFは未保存。確率vector取得後の新評価が必要 |

## 確認した保存資料とhash

以下はSHA256。新しい数値評価ではなく、byte照合とJSON/headerの構造確認である。

| 資料・固定commit | 相対path | SHA256 |
|---|---|---|
| H snapshot / carrier | `artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz` | `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` |
| ABLGRID / carrier | `artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json` | `56bca166b3fcbf6d5b235b77b8819b6bb0f51a99245fe87357479d3d55fa38d3` |
| r32 calibration dataset / carrier | `artifacts/research_direction_prevalidation/2026-09-21/wp01s_calibrations/ld3_dt0p02_r32_k2_q1_q2_mc8.dataset.json` | `dc56276dbbb4d46bc720fa2bdc1a8d30ca5de25191600238e2c887b9ec2346c2` |
| finite signal / carrier | `artifacts/finite_rte_signal_validation/h4_sto3g_d100_rank12_ld3_dt0p1_v1.json` | `79f13194b5183a7b1e8fc4fd891b452c02aabdd624214289b08e540fbf386e55` |
| 元m17 RESULT / cd749066… | `artifacts/rpe_m17_mean_validation/2026-11-03/run01/RESULT.json` | `79fdf7e84f90c48230857b13f39f9becb30f19892e2e9c09d80805d346432a4a` |
| 元m17 operators / 同上 | 同folderの`operators.npz` | `e1a65238e9d9b61a6d0aa542ca7f509827b0a761fc751615f1f2018a0e7020df` |
| B state / e22d7718… | `artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz` | `793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4` |
| Task A RESULT / a5304864… | `artifacts/rpe_m17_precision_isolation/2026-11-03/run01/RESULT.json` | `e5c3cb1cd455d102c8a1e74d7556ec95b26ab79e72f0727e172f4483162f362e` |
| B-1a RESULT / f766a1f7… | `artifacts/rpe_m17_b1_construction_precision/2026-11-03/run01/RESULT.json` | `d392c2430882ca239e1e4ed5f8095dc912b5ef3597313b98cf0bf51e5ea7415e` |
| B-1a replay / 同上 | 同folderの`float_source_replay.npz` | `f3831949a05207a8c677a257577d29d24fc34264297f5252fcdf8c96b3d93d8c` |
| B-1a HP演算子 / 同上 | 同folderの`high_precision_operators.zip` | `70c368097986b295914b776133789c2d254b8ab1ae72d7fedb75a40a650cb1a5` |

V2の固定入力参照を辿り、上記ablation、finite signal、r32 calibration、関連RPE horizon/four-round/failure資料、
候補A/Bの原出力を調べた。関連JSONで完全なcomponents/pre_threshold_components/basis_definitions本文は確認できなかった。
Bの2成分記録を除き、選択event・集約値・fingerprint・費用/信号の記録だった。
LOCALの既知の元m17/A run02/Bと最新B-1aの保存先にも、追加のpayload exportはなかった。
無関係な研究・全サーバーを探索したという主張ではない。

特に古いfinite signal資料はrandomized index末尾が[11,10]、ABLGRIDは[10,11]で、保存λも文字列上で一致しない。
同じH4/rank/L_Dというだけで、別artifactの成分・順序・集約値を元m17入力へ流用しない。

## 再生成の可否とidentityの限界

既存load_connected_cluster_hamiltonian_snapshot → split_df_hamiltonian_by_ld → prepare_df_partial_s2 →
extract_df_tail_from_hamiltonian → extraction_to_symbolic_rte_tailの経路を再利用できる。
snapshotはH/DF因子の保存物で、eta/eigenvectorsやGaussian gate列の完成snapshotではない。
prepare_df_partial_s2は対角化・Gaussian basis定義の生成を伴うため、今回は呼んでいない。
SCFやDF分解の再実行をする必要はないが、成分・basisのSOURCE replay自体が新しい科学計算となる。

既存preparation hash `89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919`、
tail hash `65b31bcf432b703c523753ee4607236a2e9530d93401dcec9255e686cc3f04af`を将来の照合に使える。
tail hashはpre-threshold coefficient/supportとbasis descriptor/local-matrix hashを含み、
preparation hashはdeterministic block・partition hash・tail hash・policyを含む。
一方、全sampling probability vectorや元sector basis matrixの本文を直接保存したhashではない。
再生成後のhash照合は有力なprovenance検査だが、新生成値を元の保存値と改称する理由にはしない。
SOURCE/環境・配列順序・係数/λ・basis descriptorを固定し、照合できない橋渡し誤差を0と置かないことが必要。
Gaussian gate生成はqiskit-natureのBogoliubovTransform等の可用性にも依存する。
元のpackage_versionsはnumpy/scipy/qiskitが中心で、分解dependencyの全versionを記録したものではない。
将来は実際に選ばれた生成経路とgate descriptorを記録し、version名の一致だけで元列の一致を推定しない。
