# O1 M1：保存q64信号のshot・必要CX単価評価

2026-11-06、SERVER 2（GPUA100-gleap02）。**承認された4点の再解析を完了し、mandatory STOP。** 減衰軽減後もr4の必要単価は極小で、r8でもfull no-prep期待単価に約4.34／6.29 CX未満を要求する。大規模な価格MCへ進む根拠は今回得られていない。ただし新law全体の回路価格下界は未取得であり、不可能性・全RPEの勝敗は未判定である。

## 固定対象・再利用・判定

- 準備SOURCE：`c64007e7a3310bddcc5c65b04463f5f98986f1e8`。[既存helper][PREP]の`account_saved_c0_signal`と`optimistic_unit_price_ceiling`を変更せず使用した。新しい[薄いrunner](../../../scripts/run_o1_m1_saved_cost_feasibility.py)は固定JSONの復元・照合とこの2関数の呼出だけを行う。
- 保存信号：`185104f64a4594355a3813d04d9e50bd4abdb6b5`の[限定結果][SIG]、`result.json`と`signal_r4.json`／`signal_r8.json`。新signalや元H行列は生成しない。
- baseline：`50ee55bc30f30838e1ac0704bc94f720914640de`の[C0結果][C0]。主比較は強DF-S2・4 substeps、spin policy／relative fusionあり。全7roundの保存総CX費用 **B=9,827,008,128** をそのまま用い、再集計していない。r2の旧shot・価格を新rへ転用しない。
- H4-A、直鎖H4・1 Å・STO-3G、保存DF rank10、共通`T_eff/V_square`、faithful tail identity。元Hの保存`state`を厳密基底状態と仮定し、状態準備費用を除外する。snapshot commit `4591b8d75a2a6a55c74a51225f0c61648023c8ee`、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`のbindingを両結果で照合した。NPZはloadしていない。
- q=64、delta=.05、K=2、r=4/8のP/DFだけ。C0の0.1 Ha／成功目標95%、全7round m0..6、energy区間[-10,0]、beta_RPE=.30を維持。PF/RTE/stat/model/実装/reserveは`.06/.03/.15/.03/.01/.02`、**各軸alpha=.05/14**。q64だけの評価でも失敗配分を付け替えない。
- 比較compilerはQiskit1.3.0、rz/sx/x/cx、optimization1、seed7、all-to-allのC0 profile。実compileは行わず、physical／backend／compiler hash・full no-prep scope・固定strong baseline armを照合した。

4つの入力JSONは固定Git blobと全文一致を確認した。各rの既存257 checksは保存されたPASSを再利用し、再実行していない。元state・sector無射影・identity規約、同roundのq/r/K/delta・lambda、保存normalization／attenuation／半径を確認した。保存configを`RTEConfig`へ復元し、既存accountingの正半径・PF/RTE/model/実装／総位相予算判定を全4点が通過した。

保存configのPauli tail hashはvalidator用の数値照合ラベルであり、canonicalな価格lawの復元ではない。今回必要なのはnormalizationの算術だけで、価格providerへこのconfigを渡してはいない。保存A／半径との`rel_tol=1e-12`は演算順差を含むdata-law照合用で、科学的operator／phase許容値の変更ではない。

## 結果

既存Hoeffding shot式は、事前allowanceを含む観測半径下界rhoに対し概念的に

`N_b = ceil[4 log(2/alpha_b) / (rho^2 sin(beta_stat)^2)]`

である。実装の演算順とceilは固定SOURCEのまま使用した。単位半径は仮定せず、量子shot samplingを実行した結果ではない。

| 辞書 | r | 観測半径下界rho（accounting） | N_X | N_Y | 必要なC_weightedの厳密上限の表示値 [CX/shot] |
|---|---:|---:|---:|---:|---:|
| Pauli | 4 | 1.3803483095535074e-6 | 594,869,749,589,604 | 594,869,749,589,604 | **8.259798161513152e-6未満** |
| DF | 4 | 1.970095404246325e-6 | 292,028,168,233,855 | 292,028,168,233,855 | **1.682544561956532e-5未満** |
| Pauli | 8 | 0.0010000423544962395 | 1,133,345,886 | 1,133,345,886 | **4.335396744008651未満** |
| DF | 8 | 0.0012047918115553648 | 780,863,721 | 780,863,721 | **6.292396396272071未満** |

`C_weighted=(N_X E[C_X]+N_Y E[C_Y])/(N_X+N_Y)`は両軸full no-prep interrogationのshot重み付き期待CX単価。必要条件は厳密な不等号

`C_weighted < B/(N_X+N_Y)`

である。表示値は丸めた値であり、判定用の正確なFraction（分子・分母）は[新result][RAW]へ保存した。Bは保存binary64値だが、この整数B自体はbinary64で正確に表せる。全4点のaccountは`feasible=true`、価格なし・総候補費用なしを保持している。

この条件は、他roundの費用が非負であることだけを使う**楽観的な必要条件**である。他roundを無料と認定した全RPE集計ではない。条件を満たしても、残る6roundを含む総費用がBを下回るとは限らない。`full_candidate_total_cost`、`full_rpe_improvement`、`price_lower_bound`は未評価のまま。`not_certified`、価格不確かさ未適格というC0の留保も維持し、0.1 Ha／95%達成の包括的保証へ昇格しない。

## 価格測定へ進む価値とGO/STOP

**r4：** 全7roundのbaseline予算をq64だけへ許す楽観条件でも、期待単価に約1e-5 CX以下を要求する。追加の大規模価格評価を正当化する材料は乏しい。ただし期待CXは整数とは限らず、「1未満」だけで実現不能とは判定しない。ほぼ全回路がzero-CXとなるような法を排除する下界も今回得ていない。

**r8：** 減衰は軽減したが、約4.34／6.29 CXという単価条件は、全controlled evolution・両軸wrapperを含むscopeに課される。新lawの価格は未測定であり、達成可能とも不可能とも断定しない。大量MCを自動開始する価値は現時点で確認できず、続けるなら先に**同じ新law全体に通用するCX下界の検討、または事前固定した小規模full no-prep価格pilot**に限定するのが次の候補である。これは未実施・未採択の案であり、価格下界の数学的レビューや回路生成を今回行ったものではない。

旧r2平均価格や代表controlled検査1本の価格は、新law全体の期待価格下界として使用しない。下界が必要閾値以上なら、その点はq64だけで費用目標を満たせず終了できるが、その前提は今回未成立。pilot平均が閾値を超えた場合も、標本誤差・CIなしで数学的不可能性へ読み替えない。

次の**ユーザー判断**は、この厳しい条件を踏まえ、有限4点をここで終了するか、下界確認／小規模価格pilotの範囲を定めるかである。後者を選ぶ場合は正しいpricing law、同一compiler／scope、標本数・seed・停止規則・不確かさ条件と必要なbuild/compileの実行承認を先に固定する。q16/32のr4/8信号、全roundのlaw、候補選択や全費用は今回の結果から補完しない。**Codexは追加指示・必要な科学実行承認がある場合だけ次へ進む。**

## 来歴・終了

新runner SHA256：`dda6c355a7d37651014878c631ff3881cb114e7140647cd5034a1f02af96f6e0`。準備helperと関係6 SOURCEのbyte一致を確認し、全hash・保存JSONのcommit/path/blob/SHA256・runtimeを[新result][RAW]へ記録した。result SHA256：`4713a5b92b1bf582bc14d5785ca2c85ab59eda7ce6d4961e9e4b69572eaf890e`。

単回実行・exit0、runner内elapsed約1.92秒。Python3.12.3／NumPy1.26.4／SciPy1.14.1／OpenFermion1.6.1／Qiskit1.3.0／Nature0.7.2の既存専用runtimeをread-only使用した。単純な4点算術で科学worker分割は不要。恣意的CPU／RAM capや共有設定・他ユーザーへの変更はない。[実行ログ](../../../artifacts/track_a_o1_m1_cost_feasibility_science_20261106/run01/execution.log)を保存した。中断前は入力copy取得とschema確認だけで、科学入口は再開後の1回のみ起動した。準備段階の人工32 PASSは[PREP]の証拠であり、今回の科学判定へ読み替えていない。

**mandatory STOP。** 新H4 simulation・signal生成・回路build/compile・価格MC・全RPE再最適化・r/K格子拡張は未実施。M2は保留のまま追加作業なし。C0負結果・既存SOURCE／raw／result／grant／marker／STOPを変更していない。成果公開とremote照合後、追加科学計算へ自動進行しない。

[PREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c64007e7a3310bddcc5c65b04463f5f98986f1e8/docs/research/track_a_o1_m1_cost_feasibility_preparation_20261106/REPORT.md
[SIG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185104f64a4594355a3813d04d9e50bd4abdb6b5/docs/research/track_a_o1_m1_attenuation_science_20261106/REPORT.md
[C0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/research/track_a_o1_m1_c0_parallel_science_20261106/REPORT.md
[RAW]: ../../../artifacts/track_a_o1_m1_cost_feasibility_science_20261106/run01/result.json
