# 4方式の再利用SOURCEと不足経路

本表はSOURCEの技術調査であり、保存値の再解析・科学的詳細レビューではない。`B0`の意味は表現ごとの原SOURCEを保持する。

| 対象 | 固定SOURCE・既存関数 | 再利用できる部分 | 新RMSE比較で不足する部分 |
|---|---|---|---|
| DF Full PF、H4/H6 | B基点`f09af82482aad11460c4bb73d3e13d7ada40be6f`の`scripts/h46_long_time_signal.py:candidates/evolve_batch`、`scripts/pf_common_run_native.py:new_signal/cost_one`、`scripts/h46_long_time_native.py:cost_trial` | `FULL`はrank全prefix。元一体項・scalar、S2/m5の既存作用、full Hadamard wrapperとcompiler。強PF対照の有限探索も既存 | 固定時間signal精度の結果を全roundエネルギーRMSE Nへ転用不可。H4の現在の13round PFはこのFull PFではない。正式formula/幅menuと各候補のN/C照合が必要 |
| DF truncated PF、H4/H6 | 同基点の`h46_long_time_signal.py`の`B0/B0I`、`pf_common_run_native.py:overlay`、A`510fbacd01a2d855202de4c045d8ba1972e5ff3d`の`pr_rpe_rmse.py:existing_finite_mean` | 一体項＋保持DF prefixの決定論作用。B0Iのtail identity scalar補正は元SOURCE通り、B0と勝手に同一視しない | 元Hに対する切捨て誤差を含むRMSE契約。**H4 B0/p7はtruncated**。H6の採用対象truncation範囲は未決定 |
| DF PR、H4 DF10/H6 DF19 | A510fbacdの`schedule/paper_event_mapper`、B基点の`pr_rpe_rmse_integration.py`、`pr_r_dependent_n.py`と有限mean作用 | exact固定grid、H6 m0–3/p5/R32/q1、以後p7・δ1.6、signed11piece、共通κ、integer allocation。H4 δ.1。既存HP型N、A0診断とmean保存元 | finite-K/signedκ/実効law・state/prior/decoderを含む全schedule RMSE証拠、十分N、同identityの適格C |
| DF Full-H RTE、H4/H6 | [`9b83b96b83f62142c9d87858dc24a94db7ac5d61:scripts/h46_full_h_pure_rte.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9b83b96b83f62142c9d87858dc24a94db7ac5d61/scripts/h46_full_h_pure_rte.py)の`prepare_full_h/full_oracle/run_signal/cost_job`、`7a8bc7f132e166390c943e6b32c92cc6abe1baf4:artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json` | 元DF tailに一体項の固有basis Z成分を加え、deterministic blocksを空にし、scalarを一回適用する既存実装。固定時間のfinite mean・native wrapper・有限cohort価格の来歴 | Aの現RMSE APIには未接続。全Hのλ/law/preparation、共通κ→event_count、normalization、全round Nと平均compiled Cが必要。**prepare(ham,0)で一体項を残すB3は代用不可** |
| Pauli Full/truncated PF、H4/H6 | B基点の`scripts/pauli_h2_h6_study.py:initial/price_job`、`scripts/run_h4_pauli_b0.py`、`src/trotterlib/h4_pauli_b0_execution.py`、`pauli_signal_adapter.py`。結果案内`docs/research/pauli_jw_resource_evaluation_summary.md` | 全非identity項ならFull PF、短いprefixはB0。constantを保持する既存S2・full-wrapper制御と保存reference。Pauli B0/B0Iの同等範囲はDFへ移さない | 元のε_signal/条件付き費用は全RPE RMSEではない。現在のA/B DF scheduleへの無条件移植不可。表現固有の候補・policy・N/Cを接続する必要 |
| Pauli PR/Full-H RTE、H4/H6 | B基点の`src/trotterlib/h4_pauli_pr_search.py:law_descriptor/central_trajectory/account`、`h4_pauli_pr_mean.py`、`scripts/run_h4_pauli_pr_search.py`。固定結果`10cb3e5cca7834ebf75301bd91bd093ca51d4a6b`、科学SOURCE`e755ecf581600192fd2745d7066e410cebd933d3`。既存説明`docs/research/pauli_pure_rte_cost_mechanism_inventory_20261101.md` | 0<L_D<MはHybrid、L_D=0は全非identity項RTE。constantは残す。保存effective-grid mean・referenceとfull-wrapper有限cohort価格が存在 | 旧candidate名の`PR`を現在の方式識別へ無条件aliasしない。旧Nは固定signal精度用。共通κ・全round RMSE・同law期待価格への追加bridgeが必要 |

## Bが再利用した共通処理

- `pr_rpe_rmse_controller.generate/assess/cost/select`：候補生成、外部証拠の受入判定、exact積和、whole-schedule選択。新しいN理論、物理event生成器、最適化frameworkは追加していない。
- `pf_pr_rpe_pipeline.identity/structure/measurement/SavedArtifacts`：同一候補・元SOURCE・signed配分・測定/価格単位、固定Git JSON読取。
- `h4_df10_rpe_connection.exact_scalar/exact_value`：巨大整数Nと価格有理数。旧HP型会計のPASSをRMSEとして偽装しない。
- `rpe_r_selection`、`rpe_provider_integration`、`rpe_r_price_intake`：旧候補/law/domain管理と保存成果への入口。round別R最小を共通κの代替にしない。
- PFの保存価格：`pf_pr_rpe_pipeline.saved_deterministic_points`、`rpe_provider_integration.FixedSavedPrices`。旧identityのまま利用する経路は既存。新RMSE identityへの受入には物理recipeとpolicy使用条件の明示照合が必要で、自動relabelは追加しない。
- PR価格：既存H4/H6固定R provider、Cの`mean_compiled_price.MeanPriceProvider`、Bの`MeanPriceLookup`。C providerはPF/Full-H RTE・H6早期recipeを補完しない。

## A/C最新準備との差分確認

Aの`e2a472422f0206a2f09fcc8e28d6938ebceb6d0f`では`pr_rpe_rmse.py`を変更せず、`pr_rpe_rmse_preflight.py`と保存metadata inventoryを追加している。Bの既存exact-grid接続で早期recipeは既に扱える。新preflightはB旧pinのASTを使うため、今回のidentity変更へA側で追従が必要。Bへ別preflight/数値評価器を重複実装していない。

Cの`fcf5b64fac24f3897ab3408b84fd8256f46e8513`では`mean_compiled_price.py`にstream consumerを追加したが、既存MeanPriceProviderとmean algebra本体は不変。新consumerのquantityは`SINGLE_PATH_STRUCTURAL_PROXY`でmean Cはnull。今回Bはstream計数を必要とせず、C09489eedの既存passive lookupを保全した。A/Cのbranch/worktree・プロセス・科学原本へ書込みは行っていない。
