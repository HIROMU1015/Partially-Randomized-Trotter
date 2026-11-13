# 未承認案：代表1失敗セル、旧spectralと直接expmの二経路診断

2026-11-13 JST。**今回は実行していない。ユーザーの明示的な科学実行承認後のみ実施する。** 人工調査からの推奨案で、科学的採択・閾値変更・全20セルの再実行ではない。

## 対象とSOURCE

候補は既存F2-COARSE/m12、physical ID=`4f0aca817edccc7fbe17`。Linear H4/1Å/STO-3G、保存tol-only DF10/sector(2,2)/36、Full PF/prefix10/S2、T409.6、qPF1024、Δ.4。H/state/snapshot、one-body＋全fragment順、scalar・control/profile、prior/decoder/予算は元PLANのまま。q_RPE=4096とqPF1024を区別する。

recipe正本は`256caad3414a80de02fb68eca89c13c957ed6354:docs/research/h4_four_method_signal_screen_preparation_20261113/PLAN.json`。失敗来歴は`699178305cec241b83220f6c307c65e0539a7856`のrun02。新diagnostic helperは本技術成果の公開commitへ固定し、旧SOURCEを上書きしない。新しい科学SOURCEは承認時に公開SHAを指定できる。

## 有限call範囲

| 処理 | 承認する場合の範囲 |
|---|---|
| 入力 | 保存H/state/contextを既存loaderで1回復元。`src/`＋`scripts/`のimport条件を維持 |
| 旧経路 | 元spectrumの11block対角化、元`deterministic_step`、q1024 power。1回だけ |
| 独立対照 | 同じ11blockのhalf-step expmをcacheし、同じforward/reverse22適用のS2 macro-step、q1024 power。1回だけ |
| 観測 | 各blockの再構成/固有方程式/Gram/identity defect、macro-step/powerのunitarity、state norm、二経路差、scalar一回適用後のcomplex mean |
| complex mean | 同じphysical candidateを**2数値経路・各1回、最大2個**。追加候補0 |
| reference | m12の保存独立pairを1組再利用。新reference0 action |
| normalizer | PFのみで新有限RTE normalizer0、B=1 |
| 対象外 | 実EVD/EV追加経路、旧4セルの再計算、他15未完セル、PR/B0I/Full-H RTE、sampling、decoder、N/C/G、全13round |

block expmの数値対照は新しいPF公式や量子回路をbuildするものではないが、実H4に対する新科学計算なので別途承認が必要。新SCF・DF fit・state生成・ground-state eigensolveは不要。

## 取得と判定

`power_diagnostics`の残差とgate boolを**判定前に保存**する。元の1e−9 norm/unitarity gateを両経路へ同じ条件で適用し、失敗した経路も数値失敗として保持する。scalarは保存constantをtotal Tで一回だけ会計し、raw/correctedはPFのB1で一致する。旧run02の欠測値へ新診断値を事後代入しない。

元経路のgate失敗自体が今回の診断対象。承認を得る場合は、**事前に固定した二経路を各一回観測する有限実行**として承認し、片方のgate失敗をきっかけに自動で別backendへ切替・retryする契約にはしない。非有限・SOURCE/binding不一致・入力破損等では部分保存しSTOP。第三経路追加、閾値緩和、再正規化は禁止。

両経路のrecipeが同一で、必要な診断値が有限で保存されれば、数値機構を調べる証拠になる。直接expmがgateを通過しても、その一点の局所数値検証であり、全20セルの安定性・energy RMSE・十分N・winnerを保証しない。二経路差に新しい科学的許容値を勝手に設定せず、元allowanceの適用条件と合わせて結果を報告する。

残16セルへ直接expmを用いるか、別の数値改善を行うかは、この限定診断を見た後に**ユーザー/B/A**が判断する。今回の技術準備・人工テストの許可を、この実診断や全screenの科学実行承認として扱わない。成果公開後mandatory STOP。
