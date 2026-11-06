# O1 M1：最小総費用比較の条件案とMC接続

2026-11-06、SERVER 2（GPUA100-gleap02）。起点 `3c003c661562dc2698cf9e94922af1ec66d32e7a`。**実装・人工テストのみ。以下の数値は採択前の実行条件案であり、H4の計算結果ではない。公開後mandatory STOP。** 共通AGENTSに従い主agentで実施し、候補別worktree/branchを使用した。

## 実装・再利用

[既存bridge](../../../src/trotterlib/o1_m1_resource_bridge.py)に次の最小差分を追加した。

- `MatchedNoPrepCostProvider.monte_carlo`：既存`rte.iter_sample_rte_events`（PCG64）をstreamで再利用する。1 trajectoryごとにfreshな`q*r` eventを消費し、同じtrajectoryのX/Yを各full no-prep wrapperでcompileする。重複drawを残し、確率を再度掛けず、既存unweighted `CompiledMetricAccumulator`で平均・sample SEを返す。sample_count/seed/law/compiler/backendを記録する。古典価格MCであり、量子測定samplingではない。
- `observed_signal_bounds`：既存`validate_matched_circuit_signals`の同一roundの出力を、PF半径・finite-RTE複素誤差disk・PF/model/実装位相誤差へ接続する。数値allowanceは明示入力とし、q/δ/r/K、辞書λ、normalizationを照合する。短roundの記録を別qへ渡すと拒否する。記録が同じ固定入力・元stateから生成されたことは呼出側の科学条件であり、このfield照合だけでは証明しない。
- 既存`account_radius_round`は価格MC数・軸別SE・価格不確かさ未資格を出力する。総費用は引き続き`Σ N_axis E[C_axis]`でありMC数を乗算しない。全round集計も平均価格の条件付き算術で、価格CIや科学保証には昇格しない。

共通Gaussian T、faithful tail identity、signed factor、核scalar、controlled phase、強DF-PFの`StrongBuilder`とbasis fusionは既存実装を維持した。旧F2 prep込み価格、旧proxy係数、単位PF半径、旧4条件の結果は転用していない。UWC接続は追加していない（既存simple shift/BLISS等とOO未実装を区別する）。

## ユーザー採択用の具体的な最小比較案 C0

未固定項目を単に列挙せず、実行候補を1組に絞った。**全項目は未採択、科学実行も未承認。** 精度は小規模接続pilot向けで、chemical accuracyの優位性を主張する設定ではない。

| 項目 | C0の提案条件 |
|---|---|
| Hamiltonian / input | linear H4、1 Å、STO-3G、保存DF rank10、H4-Aの共通`T_eff/V_square`、fragment LD0、全signed factor |
| state / 価格境界 | 元Hの保存`state`を厳密基底状態と仮定。`DF_state`へ変更せず、状態準備費用を除外。PF基準信号の半径1は仮定しない |
| energy精度 / 成功確率 | `epsilon_E=0.1 Ha`、条件付き`1-alpha_tot=0.95`、`beta_RPE=0.30 rad` |
| 共通δ / horizon | `δ=0.05 Ha^-1`、登録round `m=0..6`（`q=1,2,4,8,16,32,64`）。採択条件の算術として`beta_RPE/(64δ)<=epsilon_E`。保存値・H4 shot/費用から算出していない |
| alias / branch | 無shiftの元H energy区間`[-10,0] Ha`を候補とする。実H4で区間包含・分枝を確認できなければSTOPし、別区間を黙って採用しない |
| Pauli / DF PR | 各roundとも`r=2,K=2`、fresh IID law。各qの独立mean/PF/元H信号を新規確認し、不適格なら固定scheduleのFAILとしてSTOP。r/K/δ探索は含めない |
| 主baseline | 全rank10の強決定論DF-S2、δあたり4 substeps、`relative_fusion=True`、`exact_spin_lift_spatial_eigh_v1`。厳密spin構造がない場合、legacyへ黙ってfallbackしない |
| 位相配分 | PF `.06`、RTE `.03`、stat `.15`、model `.03`、実装 `.01` rad、計`.28`（残り`.02`は自動再配分しない）。baselineのRTE枠は未使用のまま |
| 測定失敗配分 | 全arm同じ`alpha_axis,m=alpha_tot/(2*7)`。同じ達成精度・成功条件で比較。価格MC SEはこの測定失敗予算に混ぜない |
| compiler / metric | Qiskit 1.3.0、`rz/sx/x/cx`、optimization 1、seed_transpiler 7、backend/target/coupling_mapなし（all-to-all）、routing指定なし。主指標CX count、副指標RZ/count/depth/size。連続RZ価格でありfault-tolerant T価格ではない |
| 価格MC | PR各arm/各round 32 trajectory、seed=`2026110600+100*arm_index+m`（P=0、DF=1）。baselineは各軸1回の決定論価格。価格relative sample SE 10%以下をpilot精度診断とし、超過時は追加sampleを自動実行しない |

horizonは位相分枝と誤差仮定が成立した場合の算術条件に過ぎない。厳密元H基底状態からDF/PF固有状態・PF半径1は従わない。価格32標本やsample SE 10%はCI・最適性・総費用改善の証明ではない。より高精度のenergy目標や価格信頼区間には別のdomain/計画が必要になる。

数値allowanceはH4結果を見る前に、既存signal validatorのbinary64/program-scale式をそのまま使用する案とする。信号換算allowanceを`eta=d*max(operator,basis,implementation tolerance)`、半径allowanceを`eta`、複素差allowanceを`2 eta`、位相allowanceを`asin(2 eta/(rho_PF-eta))`と定義する。非正半径・asinのdomain外はSTOPする。この式自体はengineering envelopeで、厳密保証へ転用しない。baselineにも同じ精度・normの独立参照とallowanceを適用する。実行時に差を見て式や予算を緩めない。

## 科学実行のGO/STOPと残作業

1. **ユーザー**がC0の全条件または具体的な代案を採択し、対象SOURCE・固定入力・新科学実行範囲を明示承認する。
2. **Codex（承認後のみ）**が同一入力から全登録qの独立参照、元stateのPF/RTE/model/実装誤差・半径とnormalizationを検証する。baselineも独立operator/controlled phaseと元state信号を確認する。条件を通過したroundだけshotとX/Y full no-prep compile/価格へ接続する。
3. いずれかのroundが不適格・価格未取得・MC精度不足なら全horizonの改善判定はSTOP。旧短時間結果で穴埋めしない。費用改善を主張するには、全roundが揃うことに加えて、価格不確かさを扱う比較方法を別途固定する必要がある。

実入力は固定計画どおりcommit `4591b8d75a2a6a55c74a51225f0c61648023c8ee`、`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。今回はNPZ decode・保存数値再解析・新H4 operator/signal・shot・実回路build/compile・価格・全round総費用を実行していない。未検証proxyを呼ぶ経路は追加していない。

## 人工テストと公開

**56 passed in 39.03 s**（新9件＋既存47件）。対象は新`tests/test_o1_m1_price_mc.py`と既存resource bridge、matched split、circuit signals、Hadamard wrapperの計5ファイル。発明した2-mode入力、mock価格でMC平均/SE、fresh stream、paired axes、重複draw、seed再現性、MC非乗算、invalid law、同一qの信号接続と外挿拒否を確認した。既存wrapper回帰には人工gateのみのtranspileも含む。既存runtime（Python 3.12.3、NumPy 1.26.4、SciPy 1.14.1、OpenFermion 1.6.1、Qiskit 1.3.0）をread-onlyで再利用し、cacheは専用`/tmp/o1_m1_cost_validation_prep_cache`、bytecode/pytest cacheは無効にした。pytestはpassive `trotterlib` package aliasでeager研究importを避け、上記5ファイルを`-q -p no:cacheprovider`で指定した。新科学結果やCI再現性の証拠ではない。

この専用branchをcommit/pushしremote SHA・全変更内容を照合して**mandatory STOP**する。旧SOURCE・raw・結果・grant・marker・STOPは変更しない。
