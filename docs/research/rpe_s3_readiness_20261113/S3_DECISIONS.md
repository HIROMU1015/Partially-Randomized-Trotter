# S3を開始する前の判断と証拠

2026-11-13。科学実行計画の正本は既存B170047の[rpe_decoder_statistics/SCIENTIFIC_PLAN](../rpe_decoder_statistics_20261113/SCIENTIFIC_PLAN.md)。本書は最新A/C/D受入で変わったreadinessだけを整理する。**今回S3実行は未許可・未実行。**

## 入力が揃った範囲

固定H4 DF10/S2/L_D1/ΔPF=h=.1/K2/M12/元exactκ・元H/state/snapshot、canonical `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。

A169d3316と補助物理packet bba150834から、13roundのraw/corrected mean、finite logB/normalization、局所bias/reference/allowance、元SOURCEが揃った。nominal各軸は64,60,56,52,47,43,38,34,30,25,21,17,12。取得の目的は入力の受入であり、値の数値再解析・biasをenergyへ換算する評価はしていない。m3/m4の観測norm欠測・SOURCE gate条件付きという水準は変わらない。

## ユーザー判断が必要な項目

| 項目 | 利用可能な事実 | 決めること・不足 |
|---|---|---|
| 検証対象 | 1固定PR configuration・13round・26軸 | 保存nominalを1固定配分として試すか。N探索/最適化は別scope |
| count caller | Bのexact整数和→ratio、元decoder固定 | 科学検証callerを明示。旧count-gridの2k/N−1とbit差があり、無断で同一視しない |
| shot law | Aはfresh ideal-law trajectoryとBorn、軸/round独立を仮定する記号契約を提供 | ideal条件付きmodelを採るか。実sampler法則・跨shot/軸/round依存は未資格 |
| zero/tie | 元decoderはenergyを返さず、priorもbranch選択に使わない | loss未定義のまま部分診断にするか、SOURCE付きloss区間でriskを評価するか。未解決energyを新定義するなら別decoder契約が必要 |
| target | 保存DF-H同sectorのsolver energy/residual、ground_state_certified=false | 条件付きtruth proxyを採用するか、厳密ground証拠を追加するか。今回はenergy_ha=null・truth_proxy_adopted=false |
| 系統誤差 | 全13roundのsignal空間bias・経験的allowance | finite-K/PF・saved-state/target・normalization・law/実装差のenergy L2への移送。unknownを0にしない |
| Minkowski | A bba150834の誤差契約が公開済み | 同一target上の推定量、比較model/coupling、全trial lossと各L2量を固定。局所phase/Tの単純和で予算達成にしない |
| 数値/統計精度 | 既存Bにbinomial/decoder/CI入口 | squared-loss support U、MSE半幅η、confidence、joint報告のα配分、FP精度扱い |
| 有限規模 | 既存trial_budgetはn≥ceil(U² log(2/α)/(2η²))の提案 | U/η/αを決めてから反復数を固定。今回実H4のnは未評価。恣意的なtrial cap/成功率・CIが通るまでの自動追加はしない |
| SOURCE・実行境界 | 入力は固定Git refs、B generator/decoder不変 | B実行SOURCE・seed/runtime・出力path・finite repetitions/STOPを公開して別途明示許可 |

## 最小の後続科学scope案

上記判断後、**1固定configuration × 全13round/26nominal軸 × 1固定shot配分**に対する独立反復のdecoder検証をBが担当する。実際の実行は別指示・許可が必要。新mean/reference/normalizer・RTE trajectory sampling・compile・価格fit・N探索・G比較は含めない。

入力rawからBernoulli確率を初めて生成する時点が科学統計実行となる。rawは既にattenuatedであり、Bで更に割らず、corrected meanをBernoulli平均にしない。反復ごと全26countと元decoderの出力/未解決を会計し、targetとの差の実直線energy lossを使う。alias大誤差を円周phase誤差で消さない。

未解決lossが未定義なら、未解決率、条件付きRMSE、解決分の全trial MSE寄与まで。全population RMSE/CIはnull。未解決0件の観測でも真の未解決確率0としない。明示loss区間・有界loss・IID反復が成立すれば、既存Bのfinite-MC CIを条件付き診断として利用できるが、十分量子Nの証明/数学的global boundとは区別する。

成功/STOP条件は既存計画を保持：全identity/入力/26軸countの一致、全trial会計、固定loss/support/CI前提。欠測・非有限/確率範囲外、SOURCE/law不一致、loss support超過、FP未解決補完、scope外の試行追加はSTOP。単なるdecoder未解決を試行除外の理由にしない。

## 価格側の独立した不足

C83d219ccのK0/K1はm0/m6の4個のideal-law E[S]。Daf6f27dの実E[C−S]は未取得、E[C]はnull。ideal E[S]＋sampler-law残差を無条件に結合しない。Cの他11round kernel準備やDの機序検証/paired compileは別担当・別scopeで、Bが重複実装しない。Full PFの73 EXACT価格は保存物理一致の分類であり、RMSE適格menuではない。

**担当**：ユーザー＝科学条件/有限scope判断、B＝shot/decoder/全loss統計、A＝物理mean・系統誤差/state/target、C/D＝全wrapper compiled平均・残差/法則/不確かさ。今回技術成果の公開照合後mandatory STOP。
