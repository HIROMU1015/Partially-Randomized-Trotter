# PR-DF LD1：保存確率F0診断の結果

2026-11-15 JST。採択計画の数値診断を一回実行し、**14/14表、条件付きevent上界13件・path上界13件を保存した。** 冒頭の3点への回答は次のとおり。

1. **段階差とsupport**：componentの正規化TVはexact0、CDF丸めTV≈3.9132533×10⁻¹⁵、格子TV≈8.3802388×10⁻¹⁶、P→G全TV≈4.5412949×10⁻¹⁵。CDF段階で25、格子段階で追加7、合計32 componentが正→0、逆変化0。order2のsupport消失は全13表で0、order表の格子TVは全てexact0。
2. **条件付きpath上界の有用範囲**：m0〜m12の全13roundで1未満。最大はm12の約1.5248229×10⁻⁷で、1に飽和するroundはなかった。これは独立一様53-bit格子等の仮定下でのlaw差の情報であり、実PCG64又は価格の採用精度を認定するものではない。
3. **次に必要な証拠**：実運用の確率モデル/seed ensembleとconditional-gridモデルの接続を明示し、ideal E[C−S]へ移すための全support残差幅又はobservableに即した移送根拠を得る必要がある。有限残差/rare/長roundの価格資格も残り、今回の差だけからE[C]補正は出さない。

予定14／再利用0／新規試行14／完了14／保留0／異常0／無断retry0。新path・H4 build・compileは全て0。E[S]再計算、E[C]・CI・価格予測・fit・Q3・N/Gは0。[C HANDOFF](HANDOFF_C.md)／[機械入力](C_HANDOFF.json)／[元RESULT](run01/RESULT.json)。

## 1. 固定SOURCEと入力、重複差分

| 役割 | 固定参照 |
|---|---|
| 今回の実行SOURCE | **`fad0c1e4fe41c0e4c6d0690158056572295ed15a`**。reader、INPUTS、差分照合、人工試験を実行前commitで固定 |
| 採択計画 | `950577f416c17a20d309a8053a4bb016027b724d:docs/research/d_pr_df_ld1_ec_plan_20261115/NEXT_F0_PLAN.json` |
| 固定F0 helper・数式・規則 | `d72dc4e6228bb380f4d1d000e5e030794fce7164` の `src/trottertracks/finite_k_sampler_law.py`、`docs/research/finite_k_sampler_law_alignment_20261113/MATH.md` / `SCIENCE_PLAN.json` |
| 元CDF helper | `155529e20a8fac39e00786c40b57830b71203c4f:scripts/h46_long_time_native.py#cdf_law` |
| 元SOURCE正規化 | `c31ef424175c02379eafebe8fc6c7548c1d9aabe:src/trotterlib/rte.py` の `_validate_components/iter_sample_rte_events/finite_rte_distribution` |
| 保存component・m0/m6 | `a637ed1b813dc3952339cb05f36beb79168ace23:docs/research/h4_pr_physical_closure_20261113/run02/RESULT.json` |
| 保存残11law | `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f:docs/research/h4_structural_schedule_results_20261113/run01/LAWS.json` |
| 元raw公開SHA | **`ec78fde57e117e9bbf79a26bfaa2e1575c1f5784`**。`run01/` の14診断と上界、X/Y対応、実行metadata、progressを保全 |
| 最終公開SHA | 本REPORT/HANDOFFを収録するcommit。remote照合後の完全SHAを最終回答で報告する |

保存linear H4/1.00 Å/STO-3G・tol-only DF10、PR-DF LD1/S2/Δ=.1/K2・元exactκ、1 preparation `b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7`、元component順を維持。現在のQ3の `ideal_finite_RTE_law` を変えていない。Bの元/total IDとの26軸対応は950577f4の既存 [AXIS_EVIDENCE](../d_pr_df_ld1_ec_plan_20261115/AXIS_EVIDENCE.json) を参照する。

[DELTA_CHECK](DELTA_CHECK.json)：計画時の§10.5調査を再利用し、その後の関連C/D refs、同じPython job名、出力先、SOURCE/入力だけを確認した。関連refsに新F0結果の公開差分はなく、同等診断0、matching local job0、出力先未使用。出所未解決候補0。確認範囲は既知の関連ref/旧索引/local script名に限定し、アクセス不能な外部jobまで不存在とは主張しない。他担当のworktree/processは変更していない。

## 2. 実行経路・演算・資格

[reader](../../../scripts/run_pr_df_ld1_f0.py) は固定helper全定義をそのcommitから読み、`categorical_audit/k2_event_bound/trajectory_bound` を呼ぶ。CDFは同じ固定SOURCEの純粋関数だけをASTで取り出し、旧launcherや物理loaderを起動しない。元helper/旧providerは書き換えていない。

[INPUTS](INPUTS.json) に固定commit/path/selector、元file SHA256、324 componentの順序・id/probability/probability_hex、各lawのm/physical ID/q/R/event_count、orders、order_weights、order_probabilities、tau/tau_hex/event_normalizationを保持した。実行時に固定SOURCE/INPUTSと保存値を再照合した。component choice入力は元SOURCEどおり `p/math.fsum(raw_probabilities)`、order入力は保存値を使い、元重みの `math.fsum` と保存normalization、その除算順序が一致することを確認。τや重みを解析式から再生成していない。

- P：保存binary64 raw weightを厳密な有理数とみなして正規化した分布。解析実数の重みから保存値までの誤差は含まない。
- Q：元SOURCEがchoiceへ渡すbinary64入力を、helperの定義に従い有理数として正規化したcategorical分布。入力自体のhexも別に保存した。
- F：NumPy1.26.4のbinary64 `cumsum` と最終値での除算によるCDFへ連続一様乱数を入れる分布。
- G：同じCDFへ一様53-bit格子を入れる分布。right境界を元helperのceilingで評価し、全格子列挙は不要。

**ここでのQ/Gは実PCG64の軌道全体のjoint law／総量子資源Gではない。** exactは全てFractionの分子・分母で保存し、表示小数は丸めた参考値。tableのexact TVは[CSV](TABLE_TV.csv)、分布本体は `run01/components.json` / `order_m0..m12.json` が正本である。

## 3. 段階別TVとsupport

| 表 | TV(P,Q)：正規化 | TV(Q,F)：CDF丸め | TV(F,G)：格子 | TV(P,G)：全差 |
|---|---|---|---|---|
| components | 0.000000000000e+00 | 3.913253283878e-15 | 8.380238783673e-16 | 4.541294854531e-15 |
| order_m0 | 3.113233832473e-19 | 1.717286481512e-17 | 0.000000000000e+00 | 1.686154143187e-17 |
| order_m1 | 1.389835906533e-20 | 5.205861477036e-17 | 0.000000000000e+00 | 5.207251312943e-17 |
| order_m2 | 4.058825830861e-20 | 9.513206724325e-17 | 0.000000000000e+00 | 9.517265550156e-17 |
| order_m3 | 1.054722555589e-20 | 4.917846759097e-17 | 0.000000000000e+00 | 4.916792036541e-17 |
| order_m4 | 2.531333649047e-21 | 4.962059233295e-17 | 0.000000000000e+00 | 4.961806099930e-17 |
| order_m5 | 5.237704177653e-22 | 3.928271774745e-17 | 0.000000000000e+00 | 3.928324151787e-17 |
| order_m6 | 8.409873188560e-23 | 4.417011661445e-17 | 0.000000000000e+00 | 4.417003251572e-17 |
| order_m7 | 5.706029510068e-23 | 7.548985802174e-17 | 0.000000000000e+00 | 7.548991508204e-17 |
| order_m8 | 9.193630267370e-24 | 3.789630102859e-17 | 0.000000000000e+00 | 3.789629183496e-17 |
| order_m9 | 1.199804163822e-24 | 3.735607056143e-18 | 0.000000000000e+00 | 3.735605856339e-18 |
| order_m10 | 1.092852476803e-24 | 5.088240462561e-17 | 0.000000000000e+00 | 5.088240571846e-17 |
| order_m11 | 1.515416065782e-25 | 7.482298401579e-17 | 0.000000000000e+00 | 7.482298386425e-17 |
| order_m12 | 4.098390375762e-26 | 2.085242344937e-17 | 0.000000000000e+00 | 2.085242349036e-17 |

component正規化TVはexact0で、保存正規化入力とsource weight比の差はこの段階ではなかった。一方、CDF丸めと格子化で非零の差を確認した。order表の正規化・CDF差は非零、格子差は全13表でexact0。段階TVの和を全TVと同一視せず、helperのexact `total=TV(P,G)` を区別している。

| 段階 | 正→0 | 0→正 | 該当component（元index） |
|---|---|---|---|
| P→Q | 0 | 0 | なし |
| Q→F | 25 | 0 | 72–79 (fragment3 Z0–Z7), 144–151 (fragment5 Z0–Z7), 252–259 (fragment8 Z0–Z7), 323 (fragment9 Z6Z7) |
| F→G | 7 | 0 | 1–7 (fragment1 Z1–Z7) |
| P→G：最終差 | 32 | 0 | 上記の異なる32 component。段階数との重複集計ではない |

componentごとのbefore/after exact値とcomponent IDは [SUPPORT_CHANGES.csv](SUPPORT_CHANGES.csv) / `components.json#support_changes`。P→G行は段階結果の最終照合であり、新たに32件が追加消失したという意味ではない。Hamiltonianからこれらの項を除いたり、確率を修正したりしていない。

Pで正、Gで0のsupportが実際にあるため、**grid側の標本へ尤度比を付けるだけでidealの全supportを回復する方法は、そのままでは使えない**。欠ける部分の寄与を扱う別証拠が要る。ただし「32成分」の個数自体は価格誤差の大きさではなく、TVの小ささからE[C]への影響を0とはしていない。

## 4. order2 mass（各段階）

| m | Pのorder2 mass | Qのorder2 mass | Fのorder2 mass | Gのorder2 mass |
|---|---|---|---|---|
| 0 | 1.005174294723e-02 | 1.005174294723e-02 | 1.005174294723e-02 | 1.005174294723e-02 |
| 1 | 2.549090351224e-03 | 2.549090351224e-03 | 2.549090351224e-03 | 2.549090351224e-03 |
| 2 | 6.395799320973e-04 | 6.395799320973e-04 | 6.395799320972e-04 | 6.395799320972e-04 |
| 3 | 1.600399542864e-04 | 1.600399542864e-04 | 1.600399542864e-04 | 1.600399542864e-04 |
| 4 | 4.001906125445e-05 | 4.001906125445e-05 | 4.001906125450e-05 | 4.001906125450e-05 |
| 5 | 1.000533254373e-05 | 1.000533254373e-05 | 1.000533254369e-05 | 1.000533254369e-05 |
| 6 | 2.501368590746e-06 | 2.501368590746e-06 | 2.501368590702e-06 | 2.501368590702e-06 |
| 7 | 6.302586364670e-07 | 6.302586364670e-07 | 6.302586363915e-07 | 6.302586363915e-07 |
| 8 | 1.575647998010e-07 | 1.575647998010e-07 | 1.575647997631e-07 | 1.575647997631e-07 |
| 9 | 3.939120874304e-08 | 3.939120874304e-08 | 3.939120873930e-08 | 3.939120873930e-08 |
| 10 | 9.847802735309e-09 | 9.847802735309e-09 | 9.847802684426e-09 | 9.847802684426e-09 |
| 11 | 2.461950718174e-09 | 2.461950718174e-09 | 2.461950643351e-09 | 2.461950643351e-09 |
| 12 | 6.154876816902e-10 | 6.154876816902e-10 | 6.154876608377e-10 | 6.154876608377e-10 |

全roundでorder2はP/Q/F/Gの全段階で正のまま。exact値は [ORDER2_MASS.csv](ORDER2_MASS.csv) / 各order JSON `order2_mass`。order0/order2のsupport変化は0。これは分布の診断であり、新しいeventを観測した結果ではない。m4の未観測rare eventや価格寄与を補完したことにはならない。

## 5. 条件付きevent・path上界と使える範囲

固定helperのK2 coupling（orderのmaximal couplingと1個/3個のcomponent選択）を使い、保存event_countで `min(1, event_count*event_TV_upper)` を求めた。各モデル内でorderに条件付けたcomponent drawが独立であること、全event/outer再開始を含め一様53-bit drawが独立であること、又は一致履歴に一様な条件付き距離の根拠があることを仮定する。同じevent labelが同じ保存word/angle/phaseを表すという既存SOURCE対応も維持する。

| m | q / R | 保存event_count | 条件付きevent TV上界 | 条件付きpath TV上界 |
|---|---|---|---|---|
| 0 | 1 / 2 | 2 | 4.649452253013e-15 | 9.298904506027e-15 |
| 1 | 2 / 4 | 8 | 4.616519709452e-15 | 3.693215767561e-14 |
| 2 | 4 / 8 | 32 | 4.642276552142e-15 | 1.485528496685e-13 |
| 3 | 8 / 16 | 128 | 4.591916352138e-15 | 5.877652930737e-13 |
| 4 | 16 / 32 | 512 | 4.591276392244e-15 | 2.350733512829e-12 |
| 5 | 32 / 64 | 2048 | 4.580668970379e-15 | 9.381210051337e-12 |
| 6 | 64 / 128 | 8192 | 4.585487605951e-15 | 3.756431446795e-11 |
| 7 | 128 / 255 | 32640 | 4.616790493994e-15 | 1.506920417239e-10 |
| 8 | 256 / 510 | 130560 | 4.579192577462e-15 | 5.978593829135e-10 |
| 9 | 512 / 1020 | 522240 | 4.545030818161e-15 | 2.373596894477e-09 |
| 10 | 1024 / 2040 | 2088960 | 4.592177349693e-15 | 9.592874796414e-09 |
| 11 | 2048 / 4080 | 8355840 | 4.616117860756e-15 | 3.857154226562e-08 |
| 12 | 4096 / 8160 | 33423360 | 4.562147283611e-15 | 1.524822910332e-07 |

[ROUND_BOUNDS.json](run01/ROUND_BOUNDS.json)／[exact CSV](ROUND_BOUNDS.csv)。**全m0〜m12で1未満、最大m12でも約1.524822910332×10⁻⁷**。この条件付きTV尺度では長roundでvacuousになる箇所はなく、全13roundで数量的な情報を得た。価格許容差が未採択であり、この値を価格精度のPASSに読み替えることはできない。実PCG64のjoint-lawに対する上界でもない。

[AXIS_MAP](run01/AXIS_MAP.json) と [C_HANDOFF.json](C_HANDOFF.json) `rows` は各roundの同じ診断をX/Y26軸から参照する。軸別に同じ確率診断を重複実行しておらず、26個の独立標本を得たものでもない。

## 6. ideal E[C]へ進むための不足

F0で、保存source比→正規化→CDF→一様格子の数値差が未評価という課題は解消した。一方、次が残る。

1. **実運用の確率モデルの接続**：固定seedのPCG64列は決定論的で、seed非重複だけではIIDを示さない。実運用をどのseed ensemble/確率モデルで評価するかを明示し、conditional-gridからの移送証拠又は採用する近似の資格を定める。CDF距離を実PCG64のjoint距離へ自動転記しない。
2. **残差observableへの接続**：現在のI-targetではE_I[C−S]が必要。law距離とfull-supportの残差oscillation、又はobservableに即したcoupling/直接評価法が必要で、観測min/maxやm4の−0.544502%を範囲上界へ代用しない。今回のsupport消失部分の価格/残差寄与も別に扱う。これらがないため価格補正はnull。
3. **有限残差と長roundの資格**：元m0/m3/m6較正・m4独立holdoutについて有限path/rare・共有係数の不確かさ、m7〜m12への移送、必要精度・同時信頼性が残る。今回F0はこれらの価格検証ではない。価格予測・CI・再fitは行っていない。

現時点で無条件に追加価格compileを投入するのではなく、次の指示では採用する確率モデル/保証水準と、残差移送に必要な証拠の有限scopeを決める必要がある。この報告は次の科学実行許可やn/roundの採択を代行しない。E[C]・price shift・CI・Q3・N/Gは全てnullのまま、保存E_I[S]も不変。

## 7. 資源・人工試験・保全

GNU time wall **0.50秒**、user1.57秒/system5.54秒、CPU1407%、最大RSS **41552 KiB**、swap0、exit0。14表ループ内部は **0.075383830秒**。小さいsupport表の処理は単一Python processで順次行い、NumPy等runtimeの既定動作は変更していない。CPU%を表のworker数や科学的進捗の根拠にはしない。巨大pathの生成・格子全列挙・プロセスpool・追加benchmarkはない。

[実行ログ](EXECUTION.txt)／[資源](EXECUTION_RESOURCES.txt)／[14組の協調的START/END](run01/PROGRESS.jsonl)／[元実行metadata](run01/EXECUTION.json)。Python3.11.0rc1の既知不具合がある定期faulthandler dumpは使用せず、各段階の終了と終了codeを保存した。[性能手法集](../../PERFORMANCE_METHODS.md)に再利用条件を記載した。

新readerの人工14件PASS、0.30秒：[log](tests/TECHNICAL_01.txt)。順序/hex/physical/正規化/τ/order/event_countの不一致拒否、support変化のラベル、同じ入口からのbound/XY参照、部分結果保存・無断retryしない経路、progressを検査した。既存helperの人工12＋toy5件の記録も再利用し、再実行していない。保存H4 P0演算子を再検査するsuiteは起動していない。

[保存後CHECKS](run01/CHECKS.json) で14表の56分布、格子質量、exact値の整合、13上界/26軸、progress、旧m3/m4/計画保全を確認した。元raw・固定SOURCE・凍結価格モデル・m4独立holdout・保存E_I[S]・旧STOPは変更していない。SOURCEとraw公開SHAを分け、C入力sidecarの全locatorは原commitへ結ぶ。C自身のconsumer受入は今回のD2診断とは別である。公開前に [FINAL_CHECK](FINAL_CHECK.json) でC入力の204 locator参照、exact CSV転記、raw/reader不変と旧manifest項目の保全を静的確認した。これは診断の再実行ではない。

## 8. ロードマップとmandatory STOP

PR-DF LD1のF0未実施は解消し、全13roundについて条件付きgrid-law診断が揃った。長roundでTV上界が1に飽和するという懸念は今回の値では生じなかった。実PCG64/解析入力・残差range・有限標本/rare・モデル移送・正式E[C]/Gは未解消。

Cへ14表＋13 round上界＋26軸の固定参照を引渡す。ロードマップ本体は統合担当が更新する。commit/push、remote SHA・公開成果物一致確認で **mandatory STOP**。追加sampling・価格取得・E[S]/E[C]/CI/予測/fit・Q3/N/Gへ進まない。
