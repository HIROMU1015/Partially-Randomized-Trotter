# Track A Mechanism：説明側one-shot結果

2026-10-19 JST。**63 method sideでEXPLANATION_COMPLETE、1 sideは既知欠測を保持。31比較が両側完了、1比較が片側欠測。mandatory STOP。**

SOURCE `b6716396acc5760fde7cf7cc6f2481b183a08f10`、承認対象commit `b174dd7144b3fef08a205c3486520895ed976ef8`の契約に従い、新しいgrantで説明側の保存値検査を1回実行した。launcher／workerはnative exit0・wait/reap済み。数値不一致・新たな欠測は0、retry0。元core32件は値をそのまま継承し、再計算・変更していない。

## 検証範囲と成立件数

|単位|完了|既知欠測|数値未解決／追加欠測|
|---|---:|---:|---:|
|method side|63（PR31／PF32）|1|0|
|比較|31（PR/PF両側）|1（PRのみ）|0|

[side別status](method_side_status.csv)、[比較別status](comparison_status.csv)、[保存出力から転記した因子62軸行](explanation_factors.csv)。同じ候補が複数精度に現れる件数を独立標本数と数えない。

63 sideのnormalization段階、empirical log margin、signed bias・allowance・headroom、alpha、保存log shot bound、保存整数のceil整合性を確認した。headroom／log boundの許容abs差1e-12を維持し、全126軸がBINARY64_DIAGNOSTIC経路で受入となった。保存shotの置換・再最適化は0。PFで未保存のRTE専用b/Rはnullを保持した。

**今回の説明成立は、既存SOURCEと保存値の条件付き整合性を意味する。** EMPIRICAL allowanceをrigorous error boundや母平均保証へ変更していない。費用coreのexact恒等式と、説明因子のbinary64／log表現は別の精度契約として扱う。

## 対象条件と旧coreの継承

固定32比較は、H4/H6/H8の5精度×旧PR／新Aの30比較とH8独立n48の2比較。線形・隣接1 Å、STO-3G、T=.8、P=0、通常signed、allowance倍率1。H4 legacy DF12・実測PF subset16・H4_PRE_D_P_VALIDATION、H6 tol-only DF19・強化PF136、H8 tol-only DF28・強化PF184を維持した。

分割prefix・outer PF・q/r/Kは[固定候補metadata](comparison_metadata.csv)の保存スケジュールに従う。新しいdelta窓やcandidate探索はない。元compile条件はQiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling=None。H8の元profile bytes／旧compiler出力の完全同一性、private seed/controlを再監査していないという旧制約も継承する。

新A探索n8、H4旧fixed-anchor n32、H6/H8旧confirmation union n32、H8独立固定n48をpoolしていない。H4 D/P候補・独立標本の接続は0。

元core正本は結果commit `2bf5d83293b3a47f672805fae956ff2e50df2878`とSOURCE `5c256067db29a326bcaa0a0e244b38f6b87bd138`に由来する。元results.jsonの固定SHAを前後照合し、joined_core_reference.jsonの32件が元のJSON値と同一であることを確認した。G・回路単価平均・R_G／R_N／R_Ceff・削減率・SEは新たに算出していない。[旧core表とexact比率](../../track_a_mechanism_results/2026-10-19/integrated_results_report.md)をそのまま参照する。

旧results.json内の説明statusも履歴のまま保持した。今回の説明結果はexplanations.jsonへ別に保存し、旧recordへ上書きしていない。

## normalization／headroomの説明式

`ell_upper = ell_nominal + factor*du_log`、`h = epsilon/sqrt(2) - abs(saved signed bias) - saved allowance`。

`Nstar = 2*exp(ell_upper)**2/h**2*log(2/alpha_axis)`、`kappa = N_saved/Nstar`。

PR/PF間では、連続shot比をnominal normalization²、empirical log margin²、headroom²、alpha-logの因子に分け、整数丸め因子比を別欄に保持した。step bを累積Bと同一視せず、factor1の元binary64演算順序とguardを使った。以下は保存された因子を表示丸めした値であり、追加の数値再解析ではない。

具体的な因子は`exp[2*(ell_nominal,PR-ell_nominal,PF)]`、`exp[2*(du_log,PR-du_log,PF)]`、`(h_PF/h_PR)^2`、`log(2/alpha_PR)/log(2/alpha_PF)`である。表のempirical margin²はexpしたnormalization補正の比を指す。整数丸め因子比は`kappa_PR/kappa_PF`。

## H8独立n48の2比較

ε=.05はB2_p8_q2_2nd_r4_K4対B0I_p11_q1_2nd、ε=.0005はB2_p7_q16_2nd_r4_K4対B0I_p15_q4_4th。各構成・軸の因子は次のとおり。

|ε|軸|nominal normalization²|empirical margin²|headroom²|alpha-log|連続shot比|丸め因子比|
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.05 | cosine | 1.029163 | 1.000000000067 | 0.773297 | 1.000000 | 0.795849 | 1.0000716209 |
| 0.05 | sine | 1.029163 | 1.000000000067 | 0.421497 | 1.000000 | 0.433789 | 1.0000056529 |
| 0.0005 | cosine | 1.036023 | 1.000000000448 | 1.097061 | 1.000000 | 1.136581 | 0.9999999903 |
| 0.0005 | sine | 1.036023 | 1.000000000448 | 1.366147 | 1.000000 | 1.415360 | 0.9999999922 |

ε=.05では、normalization因子は1を超え、headroom因子は両軸とも1未満となり、連続shot比は両軸とも1未満になる。ε=.0005ではnormalization因子とheadroom因子が両軸とも1を超え、連続shot比も1を超える。比較ごとに候補と対照が異なるため、この差を単独パラメータの因果効果と解釈しない。

旧coreが報告した固定n48の点費用削減率2.935924%／3.302597%はそのまま保持する。今回の説明検証により新しい母平均優位性や統計的有意性を認定していない。Ceffの軸別shot構成依存やmodel条件も保持する。

## 残る欠測

`A82_OLD_PR.PR`：H8／ε=.05／B2_p7_q2_2nd_r4_K2、旧confirmation union n32。normalizationとsigned_biasが未回収で、EXPLANATORY_INPUT_MISSINGのまま。既存会計由来のallowance/headroom等があっても、今回そのsideの説明guardを通過したとは扱わない。B=1、margin=0、推定bias等による補完は0。同じ比較のPF側は完了している。

元signal_summary.jsonは旧D0 receiptに存在とhashが記録されているが、調査済み系譜からq2原recordを取得できていない。これを解消する場合は、既存原本の所在確認、SOURCE/candidate/DF/T/cohortの再結合と、新しいbinding／別承認が必要である。

## 監査と停止

実行前後でSOURCE3 file、入力24 role、Pythonと68 runtime file、元core SHA、承認receipt・新grant・ledger・marker、出力SHAとschemaを確認した。旧結果・旧認可記録・旧STOPも不変を確認し、他worktreeのHEADや無関係資料を実行停止条件にしていない。

原入力・raw shot・bias/headroom値・完全なnorm記録・grant／ledger／marker・native receipt・hash一覧・canonical出力3件はprivate専用領域に保存した。このbranchはstatus・説明因子・公開済み科学metadataの軽量projectionで、公開pushは未実施。現在の実行証拠はローカルで、immutable CIや別担当による独立再現ではない。

新統計・sampling・signal・compile・再ランキング・212補足・H4 D/P統合・論文最終freeze・次stageは0。高次event未観測、有限標本、探索・選択効果、empirical allowance、model推定を含む条件の制約は保持する。rare tail保証・FT資源・scaling lawへ拡張しない。

**mandatory STOP。次担当はユーザー。** A82原signalの回収を進めるか、63 side完了・1 side欠測を保持して次の研究判断へ進むかを判断する。Codexの追加計算・公開push・次stageへの自動移行はない。
