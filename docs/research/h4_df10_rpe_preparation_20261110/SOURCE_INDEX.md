# 再利用SOURCE・保存成果・実装範囲

[REPORT](REPORT.md)。この索引は既存成果を再取得する入口。今回新しい科学runはなく、公開carrierと元科学SOURCEを分ける。

## 必須の3成果

| 固定版 | 参照・用途 |
|---|---|
| `155529e20a8fac39e00786c40b57830b71203c4f` | 主研究`REPORT.md`、`artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/`。今回のPF/PRの正本。B3/pure RTEを対象に追加しない |
| `fd76a3edaf5b6fee9cadb556ba57efc7c46ed778` | `docs/research/rpe_shot_error_connection_20261110/REPORT.md`。既存shot・phase/radius・reference・q二種の条件と未解決事項を引き継いだ |
| `caa592a807c533c36e06a94d2fd4e62af4a38734` | `docs/research/rpe_price_reuse_survey_20261110/{REPORT,SOURCE_INDEX}.md`。Dのq64完了、coverage・共分散、DF12 V3/R1。既存fit/残差を再計算せず使用 |

## 主研究の保存入力・選択した欄

以下は155529e2の`artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/`配下。

| ファイル | 今回の利用 |
|---|---|
| scientific_provenance.json、signal_provenance.json、completion_provenance.json | H4 H/state/snapshot/DF政策、compiler/control/runtime、元科学SOURCE・中断/継続と旧再利用の由来。私有実行path本文は新成果へコピーしない |
| best_configurations.json | system=H4、T=.8/2、epsilon=.0005、formula_group=ALLのPF/PR行だけを参照。新しいbestを計算していない |
| signal_records.json.gz | 対象candidate IDのcandidate/reference/raw/corrected/log_B/total biasと、存在する詳細分解。既存値の選択・転記のみ |
| signal_endpoints.json.gz | 同ID・同signal精度のu/log_B_margin/old_u/old_marginと旧軸別N。新RPE Nへ転用しない |
| exploration_costs.json.gz、reused_compiled_prices.json.gz | candidate全tupleと、保存RZ・mean_metrics/統計を利用。重複する決定論carrierを二標本としない |
| saved_secondary_metrics.json、per_circuit_metrics.csv | 再利用価格のCX/depth/sizeも保存済みと確認。欠測の幅はnull。新compile・再計数なし |
| confirmation_costs.json.gz、confirmation_selection.json | 同じPR candidateの独立n33 cohort。探索n9と別に転記し、pool/新G再会計をしない |

科学signal SOURCEは`e417ffc9f52b13c18f5493193ce510d858664259`。native/extension/continuation等の完全な役割・file digestはscientific_provenanceの元欄を維持する。`155529e2`へ付け替えない。H4の元receiptは同provenanceの`inputs.H4.files.H4_tol_receipt.json`に結合し、ground_state_certified=false等のmetadataだけを確認した。NPZ/vector/eigenpairを再解析していない。

## 再利用するコード（今回の基点fd76a3ed／元接続1f5ea971f）

| SOURCE path | 再利用箇所・注意 |
|---|---|
| src/trotterlib/rpe_resource_accounting.py | `_hoeffding_shots`を人工scalarでそのまま使用。全evaluate_rpe_round_candidateはPF有効固有状態等の別仮定・providerを伴うため実H4へ呼ばない |
| src/trottertracks/c1_stage3_accuracy.py | `_assess_raw_signal`のraw disk幾何、`METRICS`、等Nの`aggregate_totals`を人工比較で使用。own-DF assess_energyのenergy shiftと元HにDFを含めたlaneを混同しない |
| src/trottertracks/track_d_original_h_accuracy.py | 元H operator誤差一回計上・state/alias/law条件の既存実装を確認。DF10の座標biasをoperator_errorへ入れない。今回変更なし |
| src/trotterlib/rpe_target_round_horizon_validation.py | required_rpe_round_horizonをD2の時間/energy条件で使用する予定。今回は実H4のhorizon計算なし |
| src/trotterlib/rpe_four_round_phase_validation.py | reconstruct_rpe_phaseをtoy位相だけで試験。実H4のphase復元/RPE simulationなし |
| src/trottertracks/track_d_resource_interface.py | 既存の科学inputs/control/価格scopeとread-only readerを再利用設計として確認。CDF8専用SavedCDF8StudyをDF10に偽装しない。aggregate_suppliedの等N制約を保持 |
| src/trottertracks/track_d_b6b_bc.py、track_d_bc_total.py | 保存N/単価/総資源の接続方法を再利用。固定B4 N・A/F/S係数をDF10へ移植しない |
| scripts/h46_long_time_signal.py（155529e2） | cid/context/evolve_batch/qualifyのSOURCEを本文で確認。保存H/state/sector、finite mean、raw/corrected、log_B、signed formula、元u/marginを利用する将来入口。context/evolve/qualifyは今回未実行・非import |
| scripts/d0_frozen_shot_rule_v1.py、run_pf_common_run.py、h46_long_time_native.py | 元の有限時間headroom/u/B/compiled価格のpolicyと再利用由来。RPE energy成功のsolverとして直接使わない。SOURCEは主研究provenanceへ結合 |

D C6〜C8/B4/B-C、DF12 V1/V2/V3/R1、Track C/newAの固定資料とコードは上の価格索引・shot調査から辿れる。新たな価格fit・q64再検証・R1再解析・算法再探索を実施していない。pure RTEのT20候補IDを主研究のcandidateへ混ぜない。

## 新規部分・テスト・境界

- `src/trottertracks/h4_df10_rpe_connection.py`：保存point照合、明示dyadic時刻対応、supplied state-specific diskから既存geometry/Hoeffding呼出し、軸別failure union、軸別Nの価格総和。旧固定readerと科学SOURCEは変更なし。
- `tests/check_h4_df10_rpe_connection.py`：全15 test methodは人工scalar・mock価格のみ。実データfixtureを数値入口へ渡さず、passive import、q/時間、B/phase/radius、N/α、欠測、direct/model、旧等N会計の一致を確認。
- `SAVED_POINTS.json`：公開済み保存欄の限定projection。実データにはinspect_saved_pointだけを通し、bind_saved_round/assess_supplied_round/aggregate_supplied_axesは起動していない。B/A/phase/radius/N/Gの新数値はnull。
- `ARTIFICIAL_TEST_RESULTS.txt`：15件PASSの標準出力。既存project Python3.12のnumpy/scipy/pytest等を変更せず使用。library importによる依存初期化を、実H4の科学callとは数えない。主要numerical入口に対するsentinel試験も人工範囲で確認。

既存scientific runnerの再実行・原runへの書込みはない。新grant、approval record、全依存hash frameworkは作らない。固定SOURCE/input/raw/結果/ledger/marker/KEEP/STOPと共有環境を保全し、公開照合後mandatory STOP。
