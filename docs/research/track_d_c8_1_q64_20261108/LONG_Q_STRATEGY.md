# 長q価格の次段階準備（q64限定完了後）

2026-11-09 JST。[q64結果](COMPLETION_RESULT.md)。旧preflight FAILと旧戦略は固定commit `b1e7cc5b6270e6e62ab57b7c889f37484ce7d7b7` 以前の履歴で保全する。これは保存された有限結果に基づく技術計画の更新で、詳細科学レビュー・新しい科学実行の採択ではない。

## q64で取得できたこと

既存nofusion double経路で全8 raw/32 post/8 pairedχが同じ閾値を通過した。旧sample00 4 P0/4修正postを再実行せず、残7を完了した。Referenceの有限平均削減はCX10.962%、total depth54.588%、RZ0.987%、幅9→17。CX候補208−224qの整数残差は8本で0。主A/F/Sを凍結維持し、CX/depthの今回の点残差はSで小さかった。確認的モデル選択・母平均精度・全law/全q定理は得ていない。

q64の最大compiled component6.754e−11、CAR/raw observable Frobenius5.996e−11、paired X/Y Frobenius5.465e−11、reference leakage component8.324e−13。固定許容値はcomponent1e−10、observable/signal2e−10。観測余裕をq≥128の数値保証へ外挿しない。旧q32とはpath・長さ・数値fusion等の条件も異なり、増加をqだけへ帰属しない。

## 科学計算前に決める最小課題

| 方法 | 次に明確にする条件 | 再利用資産・限界 |
|---|---|---|
| CX構造bridge | whole-path/context/cut/phase/basis共有/ゼロ角・合成・P0を含む閉包。整数式の適用域または反例分類 | C7-2の非加法性、C7-4、q64保存構造descriptor/compile価格。有限整数一致だけでは全path証明にならない |
| Depthのlane/DAG/max-plus | 事前event特徴とcompile後DAGの橋渡し、分岐lane/maxの揺らぎ、外挿bias | 保存basis-run/lane features、既存DAG指標。単純な局所価格和やshell+maxだけのモデルへ既存反例があり、E[max]の扱いが必要 |
| 少数の独立長q直接検証 | 数値参照・誤差会計・P0全wrapperが長qでも可能か。sample00限定gateと独立holdoutの順序、call数を事前固定 | 現whole-path/Strong/gadget/P0/nofusion checker。q64のPASSをそのまま長qのPASS根拠にしない |
| 追加calibration/特徴モデル | 目標とする価格精度、モデルbiasと標本変動の区別、未使用holdoutとの分離 | 既存A/F/Sを比較対照として凍結維持。新fit・標本数設計には別scopeが必要 |

既存B/C条件付き結果では総資源の約99.5%がq≥64 proxyで、高寄与q=1024〜4096の資格が課題である。この既存記述を引用するだけで、q64を用いた全13-roundの再会計はしていない。次qを順番だけでq128に固定せず、中間qで切り分けたい数値/構造要因と、高寄与地点を直接確かめる情報量・負担を比較する。

## 次の実行案を採択するときの順序とSTOP

1. ユーザー/GPTが必要な精度・主指標と保証範囲、構造bridge先行か直接長q pilotかを判断する。新しい数値目標/成功確率/α familyを本書では固定しない。
2. 既存のSOURCE/input/law・比較baseline・step融合・cut・scalar/χ・actual inverseの再利用範囲を定め、数値checkerと参照の誤差を見積もる。phase fit/再正規化/列削減/閾値緩和でPASSにしない。
3. 新sampling・compile・再postcheck・モデルfitが必要なら対象/本数/seed/cut/許容値/STOPを別途固定し、その科学実行を承認する。q1024/4096やq128などの候補は未採択・未実行。
4. 科学条件不整合・公平な対照不能・数値不一致は依存実行をSTOPし、失敗原本を保全する。結果依存の対象変更や無断retryをしない。

今回は全て提案に留めた。B段階のactual統計保証・N、全13-round費用、追加q、詳細科学レビューへ自動進行せず、成果公開照合後mandatory STOP。次の担当はユーザー/GPT。
