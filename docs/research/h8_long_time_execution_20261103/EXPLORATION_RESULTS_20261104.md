# H8長時間・高次PR：Stage A/B実行結果

2026-11-04。Stage AとStage Bの探索を完了。Stage Cは未実行。高速化率のbenchmarkは行っていない。

Linear H8、隣接距離1.00 Å、STO-3G、保存tol-only1e-8 DF rank28、元state・fragment順序0..27。
T=1.6、ε=0.0005、状態準備P=0。費用はG=N_cos C_cos+N_sin C_sin。

## SOURCEと検査

Signal SOURCE：`d1272016592ff06c0567862f770ed796fd61b366`。Native価格SOURCE：`df93fbb88761b61c2691bf01f761ed20a7987d7f`。
Compiler位相補助SOURCE：`18ce41942051065fa097644a8a937bd11adba708`。
補正がある代表では、実際の非basisゲートを小operatorで検査し、余分な位相をfull wrapperのglobal phaseだけへ会計した。状態vectorからの位相整列・許容誤差緩和・RZ数変更は行わない。元HOLDと補正結果は別の来歴で保持する。
価格経路では既存APIが要求する整数instruction boundへ修正し、元Stage AのSOURCEと入力・値を保全した。
新T full-H sparse reference、signed primitive、finite mean/normalization、bias/headroom/shotを検査した。
Stage Aは初期57＋境界補完12＝69候補。適格40、不適格29、判定保留0。
Stage Bは選んだPR4候補（各n=9）とPF15候補（各1回）、計51 trajectory・102軸の価格を取得した。
起動セッション消失からの復旧がある場合、そのscript hash・再利用件数・未完了件数を保存結果のexecution_recoveryに記す。元seed・event列を一致させ、新しい標本を追加しない。
各群の代表例でlow/high control、algebraic event vector、cos/sin期待値、compiled vectorを検査した。
CompilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/topology未指定。measurement付きfull Hadamard wrapperのRZを用いる。

## 探索価格の比較

測定したPF15候補内の最小Gは B0I / 4th(m5_best) / prefix15 / q1、G=8.929122e+13。

| PR公式 | prefix | q | 平均G | Gのsample SE | PR/PF比 |
|---|---:|---:|---:|---:|---:|
| 4th(m5_best) | 8 | 1 | 6.732336e+13 | 1.2166e+11 | 0.753975 |
| 4th(m5_best) | 7 | 1 | 1.086763e+14 | 2.4840e+11 | 1.217099 |
| 2nd | 8 | 32 | 5.172612e+14 | 1.2871e+12 | 5.792968 |
| 2nd | 7 | 32 | 3.984594e+14 | 1.5087e+12 | 4.462471 |

観測平均の比較であり、母集団での優位や全domainの最適性の保証ではない。
同じtrajectoryのcos/sin価格を固定Nで重み付けし、軸間共分散を含むpaired Gの分散・SEを用いた。
未測定候補の費用はunknown。未観測次数の条件付き費用もunknownとし、希少orderとpopulation tail bound未確立の限界を保持する。

PR m5/prefix8の観測平均は、測定済みPF最小値より24.60%低い。sample SEは1.2166e11（平均比約0.181%）だが、9標本・288 eventではorder2/4をともに観測していない。保存sampling lawによる9標本での未観測確率は、order2以上が約85.31%、order4以上が約99.9984%。小さなsample SEだけで母集団の費用差を保証しない。
最後のS2代表は11月4日23:06 JSTにPASS。19代表の計133検査が全て許容内（native/期待値1e-8、compiled vector1e-7、位相整列なし）で、全51標本の一意性と19候補の正確な固定N平均Gを保存価格から照合した。過去のHOLDは上書きしていない。

## 保存結果と次の段階

[69候補のsignal/shot](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-03/stage_A_combined.json.gz)、
[価格・会計・比較](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-04/exploration.json.gz)、
[価格CSV](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-04/compiled_costs.csv)、
[集計コード](../../../scripts/report_h8_long_time_execution.py)。

seed実値、event列、入力raw、QPY、内部pathはprivateに保全。元SOURCE・結果・grant/ledger/marker・既存STOPは変更していない。
主担当Codexによる検査であり、別担当者の独立科学レビューではない。

次の担当：ユーザーが確認精度・n・候補・方向を判断しStage Cを許可する。Codexがweighted G分散と希少orderの限界を示し、許可後に別cohortで確認する。
**Stage Cは未実行。自動top-up・候補差替え・旧rare-RTE run再開は行わない。**
