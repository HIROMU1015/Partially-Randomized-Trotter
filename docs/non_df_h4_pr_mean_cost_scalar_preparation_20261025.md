# H4 Pauli／DF PR母平均価格：限定scalar判定の実装準備

2026-10-25。**限定scalar batchの実装・人工検証・入力binding確認まで完了した。H4のU・e・L・分離可否は未評価である。** 今回の指示§9は実行承認ではないため、科学batchは起動していない。別途明示承認後の対象は、固定95%配分による両軸の十分条件 `U_P,a<L_D,a` の一batchだけである。

基点branch `codex/non-df-h4-pr-mean-cost-stage1-20261025`、commit `c5f17b117b9d34c27b4e1d680c02a444fff4e084`。GitHubの固定refと指定4資料、再利用するSOURCE・DF保存要約を取得して照合した。専用branchは `codex/non-df-h4-pr-mean-cost-scalar-preparation-20261025`、専用worktreeも同名suffixとし、Pauli B0/B0I側の作業・出力を変更していない。

## 固定資料と再利用

- [Stage 1報告](non_df_h4_pr_mean_cost_stage1_20261025.md)、[母平均費用設計](non_df_h4_pr_mean_cost_design_20261025.md)、[旧非実行仕様](../artifacts/non_df_h4_pr_mean_cost_stage1/2026-10-25/scalar_batch_SPEC_DISABLED.json)、[データカタログ](../artifacts/non_df_h4_data_catalog/2026-10-25/README.md)を保全する。
- `h4_d2_pauli_control.py`の固定control familyと、Stage 1のcount式を変更せず参照する。builderを呼んだり、新しい回路を構築したりしない。
- `h4_conditional_cng.py:trans_bounds('ln', value)`をそのまま再利用する。古いshot／scale／価格統計の処理は呼ばない。
- DFのn_samplesと両軸minimumは、[保存cost_summary](../artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/cost_summary.json)の `/df_native_PR/n`、`/df_native_PR/observed_min` から引用する。元256本の再集計はしない。
- Stage 1で再利用した[人工4検査](../artifacts/non_df_h4_pr_mean_cost_design/2026-10-25/artificial_checks.py)を今回も同じコードで実行した。旧SOURCE・結果・grant・ledger・marker・STOPは変更しない。

条件は固定linear H4／隣接1.00 Å／STO-3G、Pauli184非identity項・PR prefix27・central制御、DF tol-only1e-8／rank10・one-body＋fragment0保持、S2／T=.8／q8／r4／K4／system8。compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、Backend/Target/coupling/layout/routing指定なし。価格はstate準備を除くfull Hadamard wrapper compiled RZ数。

## 最小実装と包含

[scalar module](../src/trotterlib/h4_pr_mean_cost_scalar.py)は標準libraryと既存ln helperだけを使う。import時にH4入力を読み込まず、評価もしない。新規部分はPython整数の固定count式、sqrt区間、DF lowerのexact有理数縮約、厳密な3分類である。

sqrtは96桁固定で、入力有理数のlowerをROUND_FLOOR、upperをROUND_CEILINGでDecimalへ変換する。それぞれsqrtを一回計算し、隣接Decimalで外側へ広げる。ゼロ端点はexactゼロを使う。さらに結果をexact有理数へ戻し、`0≤lo≤hi`、`lo²≤input_lower`、`hi²≥input_upper`を検査する。これが成立しなければ例外でSTOPし、精度を上げない。[Python Decimalの公式仕様](https://docs.python.org/3.12/library/decimal.html#decimal.Decimal.sqrt)に沿う処理に加え、包含を有理数の二乗で直接確認する。

両軸のδは十進文字列からexactに解釈し、共通lnとeを一度だけ評価する。下界区間は `L_lo=c_min max(0,1−e_hi)`、`L_hi=c_min max(0,1−e_lo)`。`U<L_lo`だけを分離、`U≥L_hi`をこの十分条件で未分離、それ以外を数値包含幅による未確定とする。等号を厳密分離へ入れない。

[runner](../scripts/run_h4_pr_mean_cost_scalar.py)は既定ではSOURCE/inputの照合だけを行い、数値kernelもimportしない。`--output`を指定した場合だけ科学関数を一回呼ぶ通常の実行入口である。**この引数はユーザーの実行許可を証明・代替するものではない。今回は指定していない。** 新しい承認flag・grant・認証・承認ledgerは作らない。旧無効仕様も有効化しない。

新しいresult fileは排他的に作成し、既存成功結果を上書きしない。数値処理中の例外では結果を捏造せず、exceptionとSTOPを保存して終了する。自動retry・昇精度・law-weighted・追加データ取得へ進む処理はない。

## 検証結果と保全

[新規人工テスト](../tests/test_h4_pr_mean_cost_scalar.py)8件と既存4件がPASSした。新規検査は、非平方・exact平方・ゼロ・大小の有理数・平方近傍、周囲のDecimal contextからの独立性、故意に壊したsqrt包含の拒否、厳密不等号と等号、人工batchでのln/sqrt共用、異なるδの停止、人工SOURCE不一致で計算前に停止することを扱う。人工結果の保存・例外STOP・既存出力の上書き拒否も確認した。全て人工値であり、実H4の値を渡していない。

実H4では新しい[入力binding](../artifacts/non_df_h4_pr_mean_cost_scalar_preparation/2026-10-25/input_bindings.json)を照合しただけである。既定runnerの結果は `INPUTS_VERIFIED_NO_H4_EVALUATION`、scalar_batch_calls=0。必要なsource/input6ファイルのSHAを固定し、無関係なworktreeのHEADや共有環境の変更をgateにしていない。

[verification.json](../artifacts/non_df_h4_pr_mean_cost_scalar_preparation/2026-10-25/verification.json)は主agent自身の技術確認で、独立科学レビューや正式certificate受理ではない。新しい科学SOURCEの実行・grant発行・消費、H4 scalar、build/compile、sampling/RNG replay、Hamiltonian/state/Arb作用、N/Gはすべて0。旧数値結果・証拠statusは昇格させない。

## 別途承認後の実行範囲

固定値はn=8/L=27/q=8/r=4/K=4、n_samples=256、DF minimum=21754（両軸）、δ_cost=.05、δ_axis=.025（各軸）。今回これらをU・e・Lへ代入していない。

許される最大範囲は、両軸U各1回、共通ln helper1回、共通sqrt正端点2評価、両軸lower縮約各1回、両軸比較各1回からなる**一つの限定scalar batch**。精度96桁固定。予定する出力の主要fieldは次の四つである。

| field | 意味 |
|---|---|
| `numerical_separation_status` | 各軸と両軸の十分条件の数値判定 |
| `count_premise_status` | 正常returnするSOURCEモデルでのcount補題。全law正常終了と実wheel対応は未証明 |
| `statistical_model_status` | 固定effective lawのIID仮定、両軸union bound、軸独立性の仮定なし |
| `population_claim_scope` | 分離した場合も上記前提付きのnamed母平均回路価格比較だけ |

IIDは仮定であり、PCG64/SeedSequence等の独立性証明ではない。価格δ_axisと量子shot α_axisは別。未知eventの費用0埋め、成功trajectoryだけへの再正規化は行わない。数値的に両軸分離しても、全allowed入力のcompiler正常終了・Rust SOURCEとwheel binaryの完全な対応を解決したことにはしない。

今は次の**入力照合のみ**のコマンドを実行済みである。

```bash
python3 scripts/run_h4_pr_mean_cost_scalar.py
```

別途明示承認された後だけ、準備したSOURCEを使うworktreeのrootで、未使用の出力pathを指定する。次は手順の提示だけで、今回は実行していない。

```bash
python3 scripts/run_h4_pr_mean_cost_scalar.py \
  --output artifacts/non_df_h4_pr_mean_cost_scalar_preparation/2026-10-25/scalar_result_after_approval.json
```

両軸分離・いずれか未分離・数値未確定・SOURCE不一致・例外のいずれでもSTOPする。未分離を逆順位へ読み替えない。G、ρ、B0/B0I、最良PF、他分子、正式certificateは対象外のまま。

## 次の担当・公開・STOP

**ユーザー**：この固定SOURCE・入力・95%配分・compiler未証明前提を明記した限定scalar batchの実行を、別の指示で承認するか判断する。**Codex**：明示承認後だけ上記一batchを実行し、数値分離と前提statusを分けて保存してSTOPする。現在は実行未承認である。

最新版共通指示2026-10-25の§8.4／§10／§10.3を適用し、共有環境・他ユーザー・別のB0/B0I作業に変更しない。専用branchでcommit・GitHub公開・local/remote SHA照合を行い、[公開記録](../artifacts/non_df_h4_pr_mean_cost_scalar_preparation/2026-10-25/publication_status.json)を保存する。**準備完了後mandatory STOP。実H4の科学batchへ自動移行しない。**

主成果物commit **`cc67e46a8bee8682d0ab1a1222ac657284973b9c`** をGitHub接続のGit data APIで公開した。local準備tree `a0cdb43f57f4d50106e63a39f9913cec1909fa29` と公開treeが一致。同commitをgit fetchし、local HEAD／origin ref／GitHub refの同SHA・clean状態を確認した。この公開記録だけを追記し、最終HEADもremote照合してチャットへ記載する。**準備完了・mandatory STOP。実H4 scalar batchは未承認・0件のまま。**
