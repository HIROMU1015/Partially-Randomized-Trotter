# H6 DF19価格設計：固定SOURCE・selector・再利用範囲

[REPORT](REPORT.md)、[COVERAGE](COVERAGE.json)。公開carrierと科学SOURCEを区別する。今回の文書commitを元runのSOURCEへ付け替えない。

## 必須成果・H6入力

| 固定commit | path / 用途 |
|---|---|
| 95d01b6ab1d820921fcd2521324e41e82389713a | docs/research/h6_df19_rpe_d2_results_20261110/{REPORT.md,COVERAGE.json,run01/result.json}。m0/1/2/5のcandidate/N、保存weights/allocation、m3/4の出典。科学SOURCE e0503b769358f239ef63245dda78ef1ee6ffce9b |
| 45607a5086721ea1d21b1ed2d7edfe870549f0c8 | docs/research/h6_df19_rpe_d1_results_20261110/{REPORT.md,result.json}。H/state/snapshot bindingと4候補/価格cohort。科学SOURCE4c928b905c65d73de23caf82a113a5fb4826e48d |
| 47b3157ecd8c651c326b1bcfa3e8037534d289e1 | docs/research/h6_df19_rpe_d2_preparation_20261110/{REPORT.md,PLAN.json}。固定candidate、元H6 context/入力参照、期待compiler、旧binding.Tの意味 |
| 10132af4a67ce112f79c539a9552f5ca0e83e516 | docs/research/h4_df10_longtime_price_design_20261110/{REPORT.md,SOURCE_INDEX.md}。構造会計、開発/留保、補正・反証pilot設計の方法だけ |
| caa592a807c533c36e06a94d2fd4e62af4a38734 | docs/research/rpe_price_reuse_survey_20261110/{REPORT.md,SOURCE_INDEX.md}。D/C/旧DF12/新Aの既存fit/holdout/共分散とコードの正本索引 |
| 155529e20a8fac39e00786c40b57830b71203c4f | REPORT.md、artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/。元PF/PR価格とscientific/signal/completion provenance |

H6 Hamiltonian hash `223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9`、state `664c63350c2510518b0ec561f804641447d39f75e68b960192c31d37540265f3`、snapshot `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`。completion/sector/fragment/scalarは元receipt/D1/D2を保持する。実発展Tはcandidate.T、歴史的binding.T=.8を流用しない。

## H6主価格selector

以下は155529e2のartifacts/resource_applicability/track_a_h46_long_time/2026-10-25配下。selectorは `/<candidate_id>`、全候補tupleはCOVERAGE.direct_price_metadataに転記した。

| 役割 | ID | 直接価格源 |
|---|---|---|
| Full19/m5 T.8/q1 | 76601611270ed9f0677d | reused_compiled_prices.json.gz。exploration表は同価格alias。旧PF_endpoint_costs receipt由来 |
| Full19/m5 T1.6/q1 | a4f5ffc0f320644dadfa | exploration_costs.json.gz、n1 |
| Full19/m5 T1.2/q1 / T2/q1 | f772e9ccab13f925452c / c47cbfb52f2212129c40 | exploration_costs.json.gz。schedule外の補助点 |
| Full19/m5 T.8/q2 / q4 / q8 | 4ecf37965885c656e554 / cc3c13cd35c59ad085eb / bc9c21796827c69cc0ad | reused_compiled_prices.json.gz、旧直接値。δは.4/.2/.1で異なる |
| PR5/R32/K2/m5 T.8/q1 | 2606efdab20e3b8522e5 | exploration_costs.json.gz n9、confirmation_costs.json.gz n33 |
| 同PR T1.2/q1 | 2b124e8f8b631f1f646f | 同2表、n9/n33。別δ |
| PR7/R16/K2/m5 T1.6/q1 | 2d2af6ca830dff14a695 | 同2表、n9/n33 |
| 同PR T2/q1 | 5bff2b8e798d984b0dde | exploration_costs.json.gz n9 |

上記に加えreused_confirmation_costs.json.gz、extension_parallel_costs.json.gz、fixed_delta_exploration_costs.json.gzを同族で確認した。対応無しを全保存先の不在証明にしない。保存平均/SE/variance/covarianceは元unit_statisticsを正本とし、今回再集計していない。PR2族の9/33 resultとevents_PRIVATEの存在は固定trial_filesから限定確認した。個別event列・seed・内部path・QPYを新たに公開しない。

PF m3のproof/receiptには旧科学SOURCE635ba563766875d688ad6912499b7f90397d0cc5、fingerprint/compiled_SHA/preparation_hash等がある。X fingerprint291fd7447e487b795146c526da96d234ebbd09a444e6022abeb952375313bd45、Y fb4fdc1bb8c1557659977f7b70b524ba23ffcd7dc9a76a26c5bee14c4dd94eccのJSONは長時間cache内では未確認（該当pathは存在しなかった）。元旧cacheの所在や全QPYを横断検索していない。PR trial result/eventsにはwrapper fingerprintがなく、元cache全entryとのjoinも行っていない。これらを埋めるために旧compile_cachedを起動しない。

## 実builder・モデル・会計コード

| SOURCE | 読んだ関数・再利用境界 |
|---|---|
| 155529e2 scripts/h46_long_time_native.py | compose/sample_requests/compile_cached/cost_trial。11piece順、ABS_TIME配分、scalar一回、finite-bit sampler、paired full-wrapper、miss compile。本文のみで非import |
| 同 src/trotterlib/h46_m5_plan.py、scripts/h46_new2_core.py | weights/schedule、allocationの元定義。今回はD2保存weights/allocationを転記し関数を起動しない |
| 同 scripts/h46_long_time_signal.py、scientific_provenance.json | candidate ID tuple、実provider install、元private保存先の解決。context/evolve/旧runnerは起動しない |
| **実provider c31ef424175c02379eafebe8fc6c7548c1d9aabe**：src/trottertracks/resource_applicability/ax2b_native_df_v5.py | append_directional_diagonal/_append_block/build_deterministic_native/build_partial_native/build_native_hadamard_wrapper。instruction_upper_boundは実native countではない |
| 同 src/trotterlib/df_rte_qiskit.py | build_sequence/_buildのapplication順、basis plan、phase。K2を固定event価格に置換しない |
| 同 src/trotterlib/rte_compiled_cost.py、h6_hash_reuse_v1.py | transpile_and_measure_cost、canonical fingerprint。angle/wire/order/global phase/測定とcompilerまで一致するcacheだけ再利用 |
| caa592a8と10132af4の既存索引 | D native counter/transfer/max-plus、C7-4 event_features/design/residual_summary/paired統計、Track C shared-cell/holdout、新A角度分類、DF12 V3/R1。H6固有価格予測に実行/importしていない |
| H6 D1/D2の既存adapter・accounting | 価格には元科学bindingとN_origin/C_origin/cohort/domainを渡す。今回は会計器追加実装やN×Cを行わない |

主研究の元signal SOURCEe417ffc9f、native27ebabdd2、extension_signal206936049、extension_nativef1e95576a、continuationff323f1b0はprovenanceの短縮欄を保持する。公開carrierだけで実import SOURCEを判断しない。

## 技術確認と作業基盤

固定D2を親に独立branch codex/resource-eval-h6-df19-price-design、専用worktreeを作成した。projects/AGENTS、worktree AGENTS/PROJECT_MAP、研究概要を適用。研究概要はH6の追加prefaceを読み、既読の共通本文と一致を確認した。D1/D2/H4設計/価格横断調査のSOURCEを再利用し、同じ科学検証は重複実行していない。

文書/selector/metadata一致、Git差分、公開成果物を検査する。新しい科学module import・feature計数・科学/人工test・予測/残差/fit/bootstrap/sampling/build/compile/signal/N/Gは0。独立科学レビューではない。元SOURCE/raw/結果/KEEP/STOPは変更せず、他担当worktreeへ書き込まない。公開照合後mandatory STOP。
