# DF10長時間価格設計：固定SOURCE・再利用箇所

[REPORT](REPORT.md)。2026-11-10 JST。以下は本文・保存metadataの読取出典で、今回の公開commitを元科学実行SOURCEへ付け替えない。pathはrepository相対。既存横断[価格索引][SURVEYINDEX]を再利用し、再調査はDF10への適用差分に限った。

## 必須資料とDF10の正本

| 固定commit | 読んだ資料・用途 |
|---|---|
| `305439c01b6e9d8ec264045585ad77b36630275c` | [費用接続REPORT][CONNECTION]と同directoryのCOVERAGE.json/PREPARED_INPUTS.json。m3会計との分離、48候補coverage、compiler hash、direct/model欠測。`h4_df10_rpe_preparation_20261110/{SOURCE_INDEX.md,SAVED_POINTS.json}`の元H/DF/state/snapshot binding。GitHub connectorでもREPORT冒頭を取得 |
| `155529e20a8fac39e00786c40b57830b71203c4f` | [元PF/PR REPORT][H46]、`artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/`のscientific_provenance、6価格表、per_circuit_metrics.csvのschema。保存candidate/cohort/原本参照だけを選択し、平均・残差・価格比を新計算しない |
| `caa592a807c533c36e06a94d2fd4e62af4a38734` | [価格再利用REPORT][SURVEY]と[SOURCE_INDEX][SURVEYINDEX]全文。D/C/DF12/新Aの再利用入口と限界を継承 |
| `ffe8a71da98bec330ce1af109a9606f14e1fc598` | D1結果は305439c01の固定参照を継承。科学SOURCEは`311255aaac39dc1c3321108c296bde7dc93dd543`。D1 runner・Nは再実行しない |

155529e2のscientific_provenanceの元SOURCE：signal `e417ffc9f`、native `27ebabdd2`、extension_signal `206936049`、extension_native `f1e95576a`、continuation `ff323f1b0`。短縮表記は元欄の引用。実providerは完全SHA `c31ef424175c02379eafebe8fc6c7548c1d9aabe`。詳細source file digestと中断/継続の由来は元provenanceを正本とし、今回新しいrun記録を作らない。

## DF10価格・candidate selector

以下の表は全て155529e2の`artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/`配下。JSON selectorは`/<candidate ID>`。候補IDだけで科学binding全体が確定するとは扱わず、上記SAVED_POINTS/scientific_provenanceと結ぶ。

| 役割・固定構成 | T / qPF / 保存δ | candidate ID | 価格表・cohort |
|---|---|---|---|
| 主PF B0/S2/p7 | .8 / 8 / .1 | `49709717a6c3d405c60d` | `reused_compiled_prices.json.gz`。exploration表にも同価格alias。全metric補足は`saved_secondary_metrics.json` |
| 主PF B0/S2/p7 | 1.6 / 16 / .1 | `8e50c5560bb8f47f026f` | `exploration_costs.json.gz`、決定論n1 |
| 主PR S2/p1/R4/K2 | .8 / 8 / .1 | `14bd81a375400c9b0305` | `exploration_costs.json.gz` n9、`confirmation_costs.json.gz` n33。別cohort、poolしない |
| 同PR、別δの補助 | 1.2 / 16 / .075 | `19e9df7e3f3cbed512ca` | `exploration_costs.json.gz` n9 |
| 同PR、別δの補助 | 1.6 / 32 / .05 | `d75c7dfdff029e0c298b` | `extension_parallel_costs.json.gz` n9、`confirmation_costs.json.gz` n33 |
| PF p7、δ=.2の補助族 | .8 / 4 / .2 | `3e0827f4ee15284b17c6` | `reused_compiled_prices.json.gz`、exploration alias |
| 同上 | 1.6 / 8 / .2 | `65bab126eb4f811edd1d` | `exploration_costs.json.gz` n1 |
| 同上 | 2 / 10 / .2 | `21bc6f56884489026a5d` | `exploration_costs.json.gz` n1 |
| 丸めを保持する別metadata点 | 1.2 / 6 / .19999999999999998 | `d76bceed98ac3bdea7a9` | `exploration_costs.json.gz` n1。exact .2との科学的頑健性とcache key同一性は別 |
| B0I/S2/p7、補助族 | .8 / 8 / .1 → 1.6 / 16 / .1 | `398f55738920acb1d4a5` → `b4156f8b791f26be5ae6` | 前者reused＋exploration alias、後者exploration |
| FULL/S2/p10、補助族 | .8 / 8 / .1 → 1.6 / 16 / .1 | `83d703b801b8545ce61f` → `dc96042e5f8605a29c8f` | 同上 |

6表目までの調査対象は`exploration_costs.json.gz`、`reused_compiled_prices.json.gz`、`confirmation_costs.json.gz`、`reused_confirmation_costs.json.gz`、`extension_parallel_costs.json.gz`、`fixed_delta_exploration_costs.json.gz`。reused_confirmationは既存COVERAGE記録を継承し、追加の5表は固定Git objectからmetadataだけを直接対応付けた。全候補の価格統計を再集計していない。

PR q8の元`trial_files`を入口に、指定9/33本のresultと同directoryの`events_PRIVATE.json`の存在を確認した。eventのschemaにはtaylor_order、selected/product component IDs、basis ID/hash/reuse interval、application_sequence、rotation_angle、phase等がある。今回はそれらの特徴を数え直していない。元compiled cache directoryも存在したが、全entryのcompiler/QPY検証・decodeは行っていない。内部path・seed・event payload・QPY・NPZをこの公開成果へ追加しない。

## 関数・テストと適合性

| 固定版・path | 読んだ箇所／再利用境界 |
|---|---|
| 155529e2 `scripts/h46_long_time_native.py` | `compose`のpiece順・scalar一回、`sample_requests`のτ/K/law、`cdf_law/rarity`、`compile_cached`のmiss compile、`cost_trial`のfull wrapper/paired X-Y。SOURCE本文だけを読む |
| 同 `scripts/h46_long_time_signal.py` | `cid`のcandidate tuple、`install`の外部provider、`saved_assets`のalias・再利用由来。signal/contextは呼ばない |
| 同 `src/trotterlib/h46_m5_plan.py`、`scripts/h46_m5_contract.py` | `weights/allocation/schedule`と実core/providerへの参照。旧authorization/grant関数を新作業へ呼ばない |
| c31ef424 [ax2b_native_df_v5.py][PROVIDER] | block instruction boundはnative gate数ではない。`append_directional_diagonal/_append_block/build_deterministic_native/build_partial_native/build_native_hadamard_wrapper`のzero/phase・basis・制御とreadout |
| 同 `src/trottertracks/resource_applicability/h6_hash_reuse_v1.py` | canonical v2の実数parameter/順序/wire/global phase/測定を含むfingerprint再利用。mutable object IDを科学identityとして使わない |
| 同 `src/trotterlib/rte_compiled_cost.py`、`df_rte_qiskit.py` | `transpile_and_measure_cost`の明示kwargsと出力RZ数、`build_sequence`のbasis plan/phase処理。D専用P0と同一視しない |
| 305439c01 `tests/test_h46_long_time_native.py` | finite-bit categorical/希少確率とcohort seed分離の人工テスト。`study.install`を含むため今回実行/importしない。保存の旧PASSを新規PASSとしない |
| 305439c01 `h4_df10_rpe_connection.py`、`h4_df10_rpe_cost_connection.py`と対応`check_*` | 元会計報告の機能と14件保存記録を再利用。新しい会計器・重複テスト・実H4会計は追加しない |

## 他系列の方法を読む固定入口

| carrier・原資料 | 今回の用途 |
|---|---|
| `64072ba81bb70db1df1b34e94adb08e78926f24b`：[C7-4モデル報告][D74]。科学SOURCE `dd31e1e2878479eabe2e7cc27c21b26c8c7dc794` | absolute/paired応答、Affine/sqrt/run、共線性、探索CVと独立holdout、共有bootstrap共分散。`track_d_c7_4_models.py`のevent_features/design/fit/residual_summaryも静的確認 |
| `fe361b206a6572142775ac01972545d312e5fb14`：[C8-1完了報告][D64] | q64 raw8/post32/pairedχ完了、有限絶対価格とpaired差の違い、母平均/長q未資格。旧FAIL・修正・限定完了SOURCEを保持 |
| `c002153a98a2f2d209a4a60f035c6bd7e36293c9`：[bridge再利用報告][BRIDGE]。engine SOURCE `a01dc6620037d9feccd00560e0a6b8d203c06dc0` | q4/q8有限P0 bridge、全走査・O(L)履歴が残る限界。305439c01の`track_d_native_resources.py`でcounterのinput domain/zero/P0_UNQUALIFIEDを確認 |
| `cc60beb51930d4311d2ded84440682e583ccafc7`：[DF12 V3報告][V3] | legacy DF12/opt2のq1/2校正、r別holdout、直接域と外挿域。DF10へ係数を流用しない |
| `141fa7624d74aa5deacbd752d09df516cad9206f`：[DF12 R1の次アクション][R1] | 保存R1の情報価値判断。低寄与補完・中間q一点だけで順位を保証しない。R1は完了済みで新計画として再実行しない |
| `c5a3c102199fe97f4d5c21837c059ae2684f5ef2`：[Track C共同選択報告][C]、`src/trottertracks/track_c_closed_p3_resources.py`等（既存索引） | 既存training/holdoutを選択に使えばtrainingへ役割変更、policy凍結後に新holdout。modular価格・shared cell/weighted lawは別domain |
| `224440f740051ed5794e8db87a57908fc02dd02f`：[新A V3回転分類報告][A] | 構造RZ内訳・angle key、zero/数値pi/未確定分類、finite cohort照合。V5/max-plus/strict-Tは既存索引を入口にし、別資源をDF10 RZへ移植しない |

C6a〜C8の残りのSOURCE・価格・共分散、D B/C、Track Cの関数path、新A V5等の入口は既存[SOURCE_INDEX][SURVEYINDEX]を正本とする。今回全raw・全テスト・全数値証明を読み直したとはしない。第三者レビュー・外部再現ではない。

## 作業基盤と文書検査

- branch：`codex/resource-eval-h4-df10-longtime-price-design`。親305439c01。m3作業中の既存枝と分離し、固定資料のみを使用。
- 適用：projects/AGENTS、基点のAGENTS/PROJECT_MAP/研究概要、正式共通指示2026-11-09。今回の静的設計は詳細科学レビューの開始ではない。
- 変更は本directoryの2文書とPROJECT_MAPの入口のみ。科学計算・新コード・scientific module import・QPY decode・新fit・残差再解析・sampling・build/compile・N×C・全Gは0。
- 文書検査：14件の固定Gitリンクと2件の相対リンク、13候補selector、主対象tuple/cohort・引用PR価格の保存欄一致を確認。Git差分の形式も検査。文書のみのため新しい人工テストを作成/実行しない。公開SHA/成果物はpush後に照合する。
- commit/push・remote公開照合後mandatory STOP。V1の実行は別の明示科学許可後のみ。旧SOURCE/結果/STOP、他worktreeは変更しない。

[SURVEYINDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caa592a807c533c36e06a94d2fd4e62af4a38734/docs/research/rpe_price_reuse_survey_20261110/SOURCE_INDEX.md
[CONNECTION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/305439c01b6e9d8ec264045585ad77b36630275c/docs/research/h4_df10_rpe_cost_connection_20261110/REPORT.md
[H46]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caa592a807c533c36e06a94d2fd4e62af4a38734/docs/research/rpe_price_reuse_survey_20261110/REPORT.md
[PROVIDER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c31ef424175c02379eafebe8fc6c7548c1d9aabe/src/trottertracks/resource_applicability/ax2b_native_df_v5.py
[D74]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64072ba81bb70db1df1b34e94adb08e78926f24b/docs/research/track_d_c7_4_model_stability_20261107/MODEL_REPORT.md
[D64]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe361b206a6572142775ac01972545d312e5fb14/docs/research/track_d_c8_1_q64_20261108/COMPLETION_RESULT.md
[BRIDGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c002153a98a2f2d209a4a60f035c6bd7e36293c9/docs/research/track_d_bridge_run_reuse_20261110/REPORT.md
[V3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc60beb51930d4311d2ded84440682e583ccafc7/docs/research/pr_rpe_v3_cost_transfer_preflight_20261103/REPORT.md
[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/141fa7624d74aa5deacbd752d09df516cad9206f/docs/research/pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/MINIMAL_NEXT_ACTION.md
[C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md
[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/224440f740051ed5794e8db87a57908fc02dd02f/docs/research/pr_reference_rotation_v3_20261107/REPORT.md
