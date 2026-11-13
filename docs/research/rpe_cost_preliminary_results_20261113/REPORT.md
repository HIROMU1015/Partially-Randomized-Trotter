# B：RPE総コスト最適化の有限予備分析P0〜P4

2026-11-13 JST。**ユーザーが許可した固定保存結果のP0〜P4を一回集計し、完了した。** 固定nominal×ideal E[S]の構造費用ではm12が66.2031%、m10〜m12が96.9198%、m9〜m12が99.0845%を占める。これは正式Gの寄与率ではなく、長時間domainの未取得証拠を優先検討するための条件付き指標である。

[数値DECISION_PACKET_V0](DECISION_PACKET_V0.md)／[次の最小計画](NEXT_OPTIMIZATION_PLAN.md)／[A/C/D HANDOFF](HANDOFF.md)／[exact結果](run01/RESULT.json)／[寄与図](run01/ROUND_CONTRIBUTIONS.png)／[技術4件](TESTS.txt)／[保存・結果確認](CHECKS.txt)。[80dade8eの準備資料](../rpe_cost_preliminary_analysis_preparation_20261113/REPORT.md)は未実行時点のまま保全した。

## 得られた判断材料

| 結果 | 水準・意味 | 次の判断 |
|---|---|---|
| H=Σ nominal N×E[S]は約2.5094226614×10^11、m10〜m12が96.9198% | **条件付き導出**。nominalは十分Nでなく、E[S]はE[C]でない | この固定proxyを代表例とする研究では、長時間価格domain・全round Nの競合を放置できない |
| m12/m11の構造量比は約3.99745 | **固定scheduleの記述**。T²モデルのfit/compiled外挿ではない | 共通κでevent数を増やす設計とN側の利得を結合して評価する必要がある |
| Full PF C / PR Sの損益分岐係数を16軸で算出 | **条件付き導出**。N比rとcompiled補正ρは未知、局所PR費用優位の条件はrρ<θ | 例えばS2/.1 m4のθ≈1.01127とY4/.4 m4のθ≈0.75985は異なる。N差や価格補正を無視した方式比較はできない |
| fresh pilot内bar C/bar Sはm0≈1.11095、m6≈1.04210 | **有限development標本の比**。各4 path、全law比/CIではない | 構造量からcompiled価格への差を別に扱う必要がある。m12へ補正率を移植しない |
| PR13＋PF短round12＋PF m6の4＝29保存signalを一覧化 | **保存観測の転記**。新phase/radius/bias/N/RMSEを計算していない | 減衰・局所bias・数値allowanceの出典をrisk担当へ渡せるが、必要N比は得られない |
| 価格欠測8軸、screen未完status、十分N/global boundを保全 | **UNKNOWN/未資格**。欠測を0にしない | 早期の局所診断や数値gate失敗からPRの最適性・baselineの敗北を作らない |

短roundの各公式/幅は表の順序で表示し、候補rankingは行っていない。新しいL_D/K/κ/幅の候補生成・正式採択もない。現L_D1固定構成の費用構造を、PR全設計空間の最適性へ拡張しない。

## 実行SOURCEと範囲

- 数値caller/算術SOURCE：**`6ec7d1958b0691aa322b4adfa6595ac6329f7508`**。実行前にコードと人工4件PASSをcommitした。
- 固定計画/locator：**`80dade8e9b21bf62f54dabf5b554e42f157dc7f3`**の`SOURCE_SCOPE.json`。A169d/C9d95/D39ddd/e7dca/A699178/B2084791の指定保存物だけ。
- PR canonical：`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。linear H4/1Å/STO-3G、保存DF10/sector(2,2)/36、元H/state/snapshot、S2/L_D1/ΔPF=.1/K2/h=.1/M12/exactκ/P0。
- 実行入口：[`scripts/analyze_rpe_cost_preliminary.py`](../../../scripts/analyze_rpe_cost_preliminary.py)。保存物の読取と[`rpe_cost_preliminary.py`](../../../src/trottertracks/rpe_cost_preliminary.py)のexact算術だけ。既存`SavedArtifacts`、`same`、`exact_scalar/exact_value`を再利用。

新signal/reference/normalizer、RTE sampling、Monte Carlo、trial再解析、回路compile、残差/affine fit、新N配分、実G、方式winnerは**0**。S3 trial archive、private event/circuit cache、C kernel arraysは開いていない。科学的なloss/評価水準/confidenceを新たに採択していない。

## 不確かさと比較の限界

構造proxyの端点/寄与率boxは元Cの数値契約に条件付き、confidenceやcompiler差の囲いではない。exact有理数で保存binary64値を会計し、表示丸めとは区別した。巨大整数の精度、欠測/分母0のUNKNOWN、pilotの有限cohort性は人工テストで確認した。科学的な独立レビューではない。

PRのpilot価格はm0/m6、選択したFull PF直接価格はm3/m4で、同roundのdirect/direct比較は揃わない。短roundのθはFull PF compiled C対PR構造Sの条件式。旧Full PF nativeとDの新controlled ZZ loweringを同一化せず、各価格の元receipt/cohort/compiler/SOURCEを保持した。

全roundの誤branch/alias/未解決loss、state/target、raw数値誤差、ideal対sampler law、モデル外挿/rare/共有校正の資格は依然不足する。S3の経験的点RMSEや局所phase/Tから必要Nを作らず、formal N/G/winnerはnullを維持する。早いroundは構造費用が小さくてもbranchを支えるため、Nを独立削減できない。

## 次の担当とSTOP

**ユーザー＋A/B**：全事象risk/loss・target/誤差移送・N評価水準を選ぶ。**A**：baseline数値gateの技術調査を既存計画で進め、必要なら代表1セルの科学診断を別提案。**D/C**：長時間使用domainのcompiled価格/law/区間資格を設計し、必要な独立実測だけを有限提案。**A**：初期PR対照として共通κ1変更等の1本を検討し、**B**は明示許可後だけ全N/価格付き比較を行う。

これらは提案であり実行済みではない。今回の再解析許可は新MC・compile・候補探索を含まない。成果をcommit/push、remote SHA/tree/主要成果物照合後**mandatory STOP**。
