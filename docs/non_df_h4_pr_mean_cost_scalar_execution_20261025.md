# H4 Pauli／DF PR母平均回路価格：限定scalar判定結果

2026-10-25。**承認されたscalar batch 1件で、cosine/sineの両軸とも `U_P,a < L_D,a_lower` が成立した。** 固定SOURCE-modelのcount前提とIID統計モデルの下では、Pauli central PRの母平均compiled RZ価格がDF-native PRより小さいという十分条件が得られた。実compilerの全allowed入力での正常終了・Rust SOURCEとwheelの完全対応は未証明のままである。

## SOURCE・固定条件

実行SOURCEは準備branch `codex/non-df-h4-pr-mean-cost-scalar-preparation-20261025` のcommit [`fa3337b57e7d1e6ba5296317770a4f49a0cbaaf1`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/fa3337b57e7d1e6ba5296317770a4f49a0cbaaf1)。専用branch/worktreeは `codex/non-df-h4-pr-mean-cost-scalar-execution-20261025`／同名suffix。B0/B0I側には変更を加えていない。

[準備報告](non_df_h4_pr_mean_cost_scalar_preparation_20261025.md)、[Stage 1](non_df_h4_pr_mean_cost_stage1_20261025.md)、[設計](non_df_h4_pr_mean_cost_design_20261025.md)を継承した。GitHubの固定refと準備報告・binding・kernel・runnerの4 blobを実取得してlocalと照合し、runnerがSOURCE/input6件のSHAと保存fieldの一致を確認してから数値評価へ進んだ。[input_bindings.json](../artifacts/non_df_h4_pr_mean_cost_scalar_preparation/2026-10-25/input_bindings.json)と旧 `scalar_batch_SPEC_DISABLED.json` は書き換えていない。

| 条件 | 今回の固定値 |
|---|---|
| 分子・基底 | Linear H4、隣接1.00 Å、STO-3G、system8 |
| Pauli | JW184非identity項、retained prefix27、PR central-rotation制御 |
| DF | tol-only1e-8、rank10、one-body＋fragment0保持 |
| 科学条件 | S2、T=0.8、q=8、r=4、K=4、元のeffective RTE law |
| Compilerモデル | Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17 |
| 制約 | Backend/Target/coupling/layout/routing指定なし |
| 費用 | state準備を除くfull Hadamard wrapper compiled RZ数 |
| 統計入力 | DF保存256 trajectory、保存minimum各軸21754、再集計なし |
| 信頼配分 | δ_cost=0.05、各δ_axis=0.025、明示的IID仮定 |

## 一batchの処理と数値包含

[固定kernel](../src/trotterlib/h4_pr_mean_cost_scalar.py)と[runner](../scripts/run_h4_pr_mean_cost_scalar.py)を変更せず再利用した。Pauli上界はStage 1の式をPython整数で評価した。

`U_P,a = 12*q*L + q*r*((10*n+2)+7*n*K) + 7 + I(a=sine)`

DF下界は `e=sqrt(ln(2/δ_axis)/(2*256))`、`L_D,a=c_min,a*max(0,1-e)`。δは十進文字列からexact有理数で解釈し、既存 `h4_conditional_cng.trans_bounds` のln包含を1回だけ使用。sqrtは96桁Decimalで共通2端点を外向きに評価し、exact有理数の二乗による包含検査を通過した。lower縮約と厳密比較は各軸1回。精度昇格・再試行は0。

| 軸 | Pauli uniform upper U（RZ） | DF統計的下界Lの数値包含（RZ） | 十分条件 |
|---|---:|---|---|
| cosine | 12,391 | [19741.473302966152, 19741.473302966153] | U < L_lower：分離 |
| sine | 12,392 | [19741.473302966152, 19741.473302966153] | U < L_lower：分離 |

共通lnの表示包含は `[4.382026634673,4.382026634674]`、共通eは `[0.092512949206,0.092512949207]`。上表とこの表示は保存exact端点を小数12桁へ外向きに整形したもの。判定は小数表示やbinary64値ではなく、[scalar_result.json](../artifacts/non_df_h4_pr_mean_cost_scalar_execution/2026-10-25/scalar_result.json) のexact端点で行った。**Lの数値包含はDF母平均そのものの両側信頼区間ではない。**

DKW–Massartは非負価格分布全体を扱う。経験CDFのminimum直下では経験累積質量が0であるため、DKW事象上でその下の真の質量は高々eとなり、`E[C_D,a]≥c_min,a max(0,1-e)` を得る。保存minimumが標本依存でも、一様CDF制御から導出するため事前固定stratumとは扱わない。未観測order4やcomponentの費用を0と仮定していない。cosine/sineは同じtrajectoryに対応し、独立とは仮定せず、δ_cos＋δ_sinによるunion boundで少なくとも95%の同時coverageを使う。IIDは統計仮定であり、PCG64／SeedSequenceの独立性証明ではない。価格δ_axisと量子shot α_axisも別である。

## 四つの判定と適用範囲

| field | 保存status・解釈 |
|---|---|
| `numerical_separation_status` | `BOTH_AXES_SEPARATED` |
| `count_premise_status` | `SOURCE_COUNT_BOUND_SUPPORTED_ON_NORMAL_RETURN; TOTALITY_AND_WHEEL_CORRESPONDENCE_UNPROVED` |
| `statistical_model_status` | `EXPLICIT_IID_ASSUMPTION; UNION_BOUND; AXES_NOT_ASSUMED_INDEPENDENT` |
| `population_claim_scope` | `CONDITIONAL_NAMED_MEAN_PRICE_ORDER_ONLY` |

Pauli上界は全allowed order0/2/4を被覆する固定builder/native lowering/count非増大のSOURCEモデルに基づく。nativeの**総gate count**による上界であり、RZ自体の単調非増加や保存最大値の全law上界化ではない。数値分離によって、全trajectoryでの実compiler正常終了やRust tagと実wheel binaryの完全な対応を証明したことにはしない。これらとcount前提が満たされ、固定effective lawのIIDモデルが適用できる場合に、両軸同時で `E[C_P,a] ≤ U_P,a < L_D,a ≤ E[C_D,a]` という95%信頼の条件付き母平均価格比較になる。失敗trajectoryを価格0にしたり、成功trajectoryだけへlawを再正規化したりしていない。

この結果はnamed Pauli-central PR対DF-native PRの**回路価格**だけを対象とする。shot数込みG、実装残差ρ、PR全lawの精度保証、最良Pauli PF／B0／B0I、他分子への一般化は判定していない。旧N/G=null、6件の局所Arbと全lawρの区別、元のnative終了UNKNOWN、正式certificate未受理を維持する。母平均価格tailの影響は今回の十分条件により上記仮定の範囲で覆えるが、分布そのものや条件付き母平均Gまで確定したわけではない。

## 検証・終了・保全

準備済み人工テストは新8＋既存4の12PASS。SOURCEとテストが変わっていないため、その保存検証を再利用し、成功済み科学batchや人工検査を重複実行しなかった。今回のSOURCE/input照合は実行前後に一致。新コード・compile・sampling/RNG replay・Hamiltonian/state/Arb作用・元256価格の再集計・N/G・grant・承認証明機構は0。

[execution.json](../artifacts/non_df_h4_pr_mean_cost_scalar_execution/2026-10-25/execution.json) に2026-10-25T08:10:19Zの実行、CPython3.12.3、native returncode0、空のstdout/stderr、約0.060秒を保存した。科学runnerの起動は1回、U2／ln helper1／sqrt端点2／lower2／比較2。固定96桁、libmpdec2.5.1。科学計算後に子processを待機回収して終了を記録した。[verification.json](../artifacts/non_df_h4_pr_mean_cost_scalar_execution/2026-10-25/verification.json) は主agentの技術確認であり、独立レビューや正式certificateではない。

追加物は専用の新実行artifactと文書・索引のみ。旧SOURCE・raw・結果・grant・ledger・marker・STOPは変更せず、他worktree／共有環境への変更もない。準備時点の未実行記録は当時の履歴として保持し、今回の結果を別namespaceに保存した。

## 公開・次の担当・mandatory STOP

[公開記録](../artifacts/non_df_h4_pr_mean_cost_scalar_execution/2026-10-25/publication_status.json) にcommitとremote照合を記す。専用branchでlocal treeとGitHub tree、local/origin/GitHub SHAを確認する。最終HEADはチャットで報告する。

**ユーザー**：この条件付き母平均価格結果を比較研究・原稿でどの範囲の主張に用いるか判断する。未証明compiler前提の解消やG／ρへの拡張を希望する場合は、その具体的範囲に別の明示承認が必要。**Codex**：今回の保存・公開を完了してSTOPする。上記十分条件は両軸で分離しており、今回の価格判定のための追加compile／sampling／law-weighted評価は不要。新しい科学実行・研究方針採択には進まない。

**mandatory STOP：成功した限定scalar batchを再実行せず、追加科学計算・正式certificate受理へ自動移行しない。**

主成果物commit **`d719a017719d2a621db1878bee0155d24703f840`** をGitHub接続のGit data APIで公開し、local index／GitHub tree `da905492e9a030eeff50ec713054beecfac29474` の一致、local HEAD／origin／GitHub refの同SHA・clean状態を確認した。この公開記録のみを追記し、最終metadata HEADもremote照合してチャットへ報告する。科学result・実行記録は主成果物commitから変更しない。**mandatory STOP継続。**
