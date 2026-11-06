# A1 SERVER1 引き継ぎ確認

2026-11-06 JST。文書・SOURCE・保存metadataの照合と人工テストだけを実施した。
科学実行契約の草案は [A1_SCIENCE_EXECUTION_CONTRACT_DRAFT.md](A1_SCIENCE_EXECUTION_CONTRACT_DRAFT.md)、実績とSTOPは [REPORT.md](REPORT.md) に記す。

## 1. 環境と作業領域

ユーザーは今回の環境をSERVER1と指定した。実測hostnameは `GPUA100-gleap-01`、Linux 6.8.0-49-generic/x86_64、Python 3.12.3。
このhostnameは既存の[環境記録][HOST]と一致する。主repository、WP11 worktree、HEADは、ユーザー提供の環境contextとSERVER1の既存inventoryに対応した。
SERVER1という役割名の対応はこの指定と保存記録による。directory名だけから物理SERVER1/2を判別したものではなく、SERVER2へのSSHや作業領域の操作はしていない。

| 項目 | 確認結果 |
|---|---|
| repository | `HIROMU1015/Partially-Randomized-Trotter` |
| origin fetch/push | `https://github.com/HIROMU1015/Partially-Randomized-Trotter` |
| 元main HEAD | `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`。開始時origin/mainに19commit遅れ、dirty/untrackedあり |
| 専用branch | `codex/o2-a1-server1-handoff-20261106` |
| 専用worktree | 主repository内 `.server-preparation/worktrees/o2-a1-server1-handoff-20261106` |
| 基点 | 指定固定commit `6e8d099a5252d3b424045fb22aeb112d5ebad5c9` から直接作成。merge/cherry-pickによるSOURCE改変なし |
| 通常実行の障害 | `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`。実行環境の承認機構を通した限定コマンドで作業。OS/network設定は変更なし |

元mainのHEAD、porcelain状態、既存dirty/untrackedファイルのSHA256を私有一時領域へ記録した。
既存のstaged/dirty/untrackedを移動・stash・reset・cleanせず、他sessionのworktree/branch/lockを操作しない。
新worktree登録と今回の専用branch以外の既存refを意図して変更していない。

`/usr/bin/python3`には科学依存が入っていなかったが、既存WP11 `.venv-server-opt2` はPython 3.12.3、NumPy 1.26.4、SciPy 1.14.1、Qiskit 1.3.0、PySCF 2.7.0、OpenFermion 1.6.1であることをpackage metadataで確認した。
この環境は読み取り利用のみ。install/upgrade・共有venv/cache変更はしていない。科学worker/GPU jobは起動していない。
文書作業時の空きRAM/storageと負荷を読み取り確認し、人工的なCPU/RAM/worker制限は追加していない。

## 2. 固定5ファイルの取得と照合

指定commitは既存local Git objectに存在した。GitHub接続で5本文を固定SHAから実取得し、local本文との全文一致とGit blob OID一致を確認した。
下表のSHA256はファイルbytesのhashである。A1実装・科学条件を修正していない。

| path（基点6e8d099） | Git blob OID | SHA256 |
|---|---|---|
| `docs/research/o2_a1_preparation_20261106/A1_PREFLIGHT.md` | `39e0ba532888611a18174cf44ec43c1e556e54a1` | `020fc2b05313d0814020581dc9c2e51765a51b3a6a7e652cbd80d49c9f444209` |
| `docs/research/o2_a1_preparation_20261106/A1_EXECUTION_PLAN.md` | `8c23846417f7d7db0d9c25d08c0d37e2eb955d7e` | `911e56370a9391586ae258fd5d500389b77cc41a59d52cbfb3325724ca398807` |
| `docs/research/o2_a1_preparation_20261106/REPORT.md` | `ccf05edb12f877f9c85bc48f3f758aadc7bfcf65` | `34471792856117df8af7cda9cfcef3e4594e95c913cdadaacc7f8ec32656dc9a` |
| `src/trottertracks/algorithm_codesign/a1_control_variates.py` | `72a42cd6b964a4c408b48bf801d3cacfc67ae10d` | `22c8bea657f52b74ffb40e3e74b4f953fb67fee312cb801bccc3f7f369ec1960` |
| `tests/tracks/algorithm_codesign/test_a1_control_variates.py` | `a522542424d118cfa0fea9ac7a054d475c0de0ef` | `45ecbbc04abd5405733542521e9d20dd81789fd1a52eb4b1cfee825d245639e6` |

5ファイルの公開入口は[引き継ぎcommit][A1]。
SERVER2の保存15件PASSを、SERVER1の既存opt2環境で同じ人工テストに限って再確認し、**15 tests / OK**だった。
手書きbit・架空label/CDF・mock metadata・readout SOURCEのASTだけであり、実科学CDF、RNG、NPZ、量子回路生成、測定simulationは使用していない。
科学的variance reductionや総費用改善、CI・独立科学レビューを意味しない。

## 3. 適用指示・証拠の読み方

主checkoutの `AGENTS.md`、`PROJECT_MAP.md`、`docs/research/研究概要・現状.md` 全文、固定基点のAGENTS/PROJECT_MAP、研究目的・状態/半径の規範節、関連 `VALIDATION_STATUS.md` とmanifest、V1/V2、opt2固定protocolを確認した。
正式共通指示 `research_project_instructions_unified_2026-10-25.md` を全文確認した。§8.1/8.2/8.4/10.1/10.3を適用する。
主checkoutの同指示と、基点に収録された[公開保存コピー][COMMON]はbyte一致した。未取得の後継版を読んだとは扱わない。

新しい研究判断・validation結果を採択していない。既存概要・規範文書・manifest・研究ノート・科学SOURCEは変更せず、この独立directoryの3文書だけを追加する。
`VALIDATION_STATUS.md`/manifestの古いopt1・δ=.1の検証を今回opt2 m0の新しい保証へ転用しない。

## 4. SERVER1に存在する入力・SOURCE

以下のcanonical cohortはlinear H4、1.00 Å、STO-3G、legacy DF rank12、8qubit、4電子sector、`L_D=3`、δ=.02、m0/q1/r1/K2に限る。

| 資産 | 所在・今回確認できた範囲 |
|---|---|
| 保存H4 NPZ | [carrier内snapshot][NPZ]。3589 bytes、Git blob `77a69d5021b33a7fa1fde723fd3d1806f3c77950`、SHA256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。今回byte hash一致、配列decodeなし |
| H/model identity | WP04/V1/V2にH hash `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`、sector=4、physical state fingerprint `a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90`。vector bytesが保存されているとの認定ではない |
| 最終schedule/予算 | [opt2 JSON][OPT2] `/coherent_reoptimization/best_by_ld/3/best/rounds/0`。m0と保存ε/α/Nを読み取り引用した |
| 状態・信号・分枝の保存記録 | [WP04 grid][GRID]、[V1][V1]、[V2][V2]および各CSV。再解析runnerは実行していない |
| 元価格policy | [WP06b][POLICY] `support_run_le_1`、training fingerprint `9dcdc4a28866dc223afb0ea212df645a3ed4a28e5c6dad6ddfc0ec1fcb28ba9e`。再選択しない |
| label metadata | [WP06a][CATALOG] `/representative_events/0,1` に2つの既存IDとpositive coefficient_abs。完全な順序付きcatalog/CDFではない |
| 元opt2 raw | SERVER1の既存WP11 worktree内 `artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_initial_20260924_023012/checkpoints/2998a4077c000ae2093159ee830e8e84823c6be40179195b712e531d142f87d0.json`。m0/8 trajectory/completed、31172 bytes。10 SOURCE-fileの記録hashと本基点bytesが一致。再集計・seed replayなし |
| rawの限界 | 上記rawにaxis別policy価格とtrajectory fingerprintはあるが、全event label列・実CDF・fresh測定bitはない。旧8 trajectoryをtrain/cal/prodへ移さない。非公開rawを今回新たに公開しない |

必要なNPZや元価格rawがSERVER1に全くない状況ではなかった。ただし完全catalog、正規化済み実CDF、学習用fresh bit、保存state vector bytesの今回利用可能な固定payloadは未取得である。
catalogは上記canonical研究の保存JSONを対象に探索した範囲の記述で、repository全域・他環境に存在しないとの断定ではない。
既存SOURCEはNPZからpreparationとcatalogを再構成する経路を持つが、これは今回禁止された科学処理に入るため実行しない。
stateをmeasurement engine内で再構成する必要がある場合も、将来の承認範囲に明記する。H_D/PF control stateや別packetで無断代替しない。

## 5. 同一Hだけでは閉じないtail/finite targetの接続

静的調査で、同じH hash・partitionでも次の差を確認した。既存値の引用であり、差の大きさ・影響は新評価していない。

| 項目 | WP04保存signal grid | opt2回路/価格側のSOURCEとWP05/WP06 |
|---|---|---|
| `coefficient_atol` | `/request/coefficient_atol = 1e-12` | `_worker_preparation` が省略し、`prepare_df_partial_s2` 既定値 `0.0` |
| preparation hash | `89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919` | `50c77f5455cc1a3c0710ed3c7588efc957b9cead7436bd0a6d851af61738a4bc` |
| 保存exact_rte_lambda_r | `0.5832704246339503` | WP06a `0.5832704246339986` |
| 支持・打切りmetadata | randomized component count=218、threshold dropped=108、dropped L1=`4.8294649659693557e-14` | WP05a tail hash `c51e35da56f3de1646896339f38da3cb40dcb8405a24d781cef134a654bab642`。完全catalog/actual-q supportは未取得 |

**218-labelのsignal側lawを、opt2の実sampling lawとみなしてνを作ってはいけない。** WP04/V1/V2の保存ε・半径・biasは参照として使えるが、opt2側の厳密な同一finite target/保証へ自動移送しない。
草案はopt2 SOURCEのcanonical policy/lawを優先する提案とし、signal側との差を正当に予算化するか、対応入力を固定してから測定する条件を残す。
この確認はSOURCE/metadataの版照合であり、新たな科学的破綻・有意差の認定ではない。固定SOURCEを修正して差を消していない。

## 6. 再利用範囲と不足接続

| 既存部品 | 継承・不足 |
|---|---|
| `rte.py` | `finite_rte_distribution`、`make_rte_config`、`_make_event`、`iter_sample_rte_events`を静読。変更・科学呼出しなし |
| A1 adapter | `law_from_choice_cdf`、`EventMetadata`、`ControlBank`、`fit_and_freeze`、`FrozenCV`、`Batch`、`raw_mean`、`stratified_mean`、`TrialCost`、`account_costs`をそのまま継承 |
| full interrogation | repeated builder、ordinary control、Hadamard X/Y wrapperと既存selected basis policy。標準providerは選択policyの価格を自動保証しない |
| 価格 | `research_direction_full_opt2.py` / `research_direction_full_scope.py` の `_compile_paired_requests` 系。元関数はfullとselected両方をbuild/compileするため将来のcall範囲で注意。今回は未呼出し |
| 測定 | `rpe_hadamard_failure_validation.py` のfresh trajectory/conditional bit規約を参照。旧一括runnerはstate生成・別条件・解析を含むのでそのまま起動しない。最小のpath/axis/bit bindが必要 |
| decoder | V2の決定論的符号・nearest-branch部分のみ将来再利用可。CVへbinary PMF/tailを移送しない |
| 未実装接続 | actual CDF入力、exhaustive条件付きstrata取得、独立stream、bounded confidence/十分N、selected full価格とのbinding。今回コード追加なし |

`rte`、wrapper、full価格provider、accounting、partial_s2、repeated、full_scopeの7 SOURCE-fileはcarrier `4591b8d` と本基点でGit blob一致。
これは必要な版確認であり、全推移的依存・clean再生成・未知rawの完全監査ではない。

## 7. 保全と公開後の停止

C1一時凍結、S2のT主指標追加不改善、R1の `NO_SUPPORTED_R1_CANDIDATE`、既存one-shot/retry/STOPを維持する。
SOURCE、raw、result、grant、ledger、markerを変更しない。承認証明書・新grant・汎用pipelineは作らない。
今回の3文書だけを専用branchへcommit/pushし、remote SHAと本文を再取得照合する。確定SHAと照合実績は最終チャットで報告し、自己SHA追記のための追加commitは不要とする。
**公開・照合後mandatory STOP。新しい科学計算・測定・実装・別Trackへ進まない。**

[A1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/6e8d099a5252d3b424045fb22aeb112d5ebad5c9
[HOST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0c942fbb5dd8741dda69af0933fce88e8eaea91/docs/research/environment_evidence_gpua100_20261023/README.md
[COMMON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/archive/server1_main_collection_20261103/b05c68a693d31e3a15652faccfc4e6f0a05f8c34/research_project_instructions_unified_2026-10-25.md.original
[NPZ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz
[OPT2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json
[GRID]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json
[V1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md
[V2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
[POLICY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json
[CATALOG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_structure_pilot/2026-09-22/wp06a_circuit_structure_pilot_v1.json
