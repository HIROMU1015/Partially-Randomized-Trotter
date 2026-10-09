# Track B：GPT G1研究レビューの受領と次作業範囲

2026-10-09 JST。基点はG1結果`e62407b3c51e9318f700673b6cf403d54e344c76`。
利用者の「こんな感じで進めていく」と指定された
[GPT G1研究レビューv1.0](../../research/track_b_G1_scientific_review_20261009.md)を今後の方針として記録した。
今回の処理は文書受領・scope整理・provenance確認・公開だけで、次の診断を実施した記録ではない。

## 研究判断の更新

RA-RTEを有力候補として維持し、resource-aware representation設計を暫定主線とする。
有限平均を保つ構成に、実装費用・合成誤差・測定負担を戻し、同条件の強い対照後に残る独立した価値を早く判断する。

- 方針B：構造を利用したresource-aware representation設計を暫定主線。
- 方針A：既存canonical B2/B3は次数の連動解除を調べる機構検査。
- 方針C：identity returnは強い対照・第二候補。新主線として自動採択しない。
- 方針D：multiblock／高次数／別方式への拡張は第一選択にしない。具体的な理論目的と実験拡張を区別する。

G2は、従来の単なるsource完成確認から、追加自由度の研究価値・必要な最小実装・次実験範囲の判断点へ更新する。
この更新は旧契約・marker・結果を変更せず、実行認可を自動付与しない。

## Codexの次の一括診断（レビュー§13）

| 作業 | 固定する範囲と報告 |
|---|---|
| 数式の独立監査 | identity-return閾値、線形価格の条件、ISと線形分数目的、pure-precision profile補題を仮定別に検査する。一般証明を有限点のテストだけで代用しない。 |
| 保存表の診断 | development x={1/8,1/4}、7 prototypes、既存3 precision。同じprecision選択を対照にも与え、提案上限252 profile/x、計504以内。event確率・phase・費用・誤差のidentityを確認する。 |
| 強い対照の整理 | 同じsampling自由度を与えた既知IS、既存identity-return式を考慮する。returnの未取得angle費用・誤差はMISSINGとする。 |
| GPTへの引継ぎ | 係数再配分・precision・sampling・known returnの効果を分け、未判定幅と次の必要最小取得／実装を科学的な問いへ対応付ける。 |

この252/xはGPT側の提案範囲であり、完全性の独立証明を今回済ませたという意味ではない。
保存表の根拠は既存RA-D0 preparationの21 columns/x、固定commit
`0ddf67756516e08f85fed1b987459a5e862676b7`に対応する。
今回、表の数値再評価・profile列挙・新しいdiagnostic winnerの取得は行っていない。

数式検算・保存値算術・限定実装・単体testsは、承認された診断scopeの下でCodexがまとめて扱う。
testごとのGPT承認は追加しない。数学target、主要metric、baselineの意味、独立性、研究仮説の変更はGPTへ返す。

## 診断とclaimの境界

GPTレビュー内のreturn交点、局所IS比、線形価格条件、pure-profile補題は、GPT側の導出・自己検算として扱う。
今回の記録は、その数式のCodex独立監査、一次文献全文の再確認、優先性の認証ではない。

T/CX/1Qは別々に評価し、任意の重みで混ぜない。平均sqrt(cost)とsqrt(平均cost)を区別する。
二次モーメント最適ISと有限confidenceのproposal最適性を分け、range、整数shot、law、numeric certificate、
zero-cost、bias exhaustion、shot capと達成可能性を記録する。T=0を勝手な正costで置換せず、測定・状態準備無料とも扱わない。

診断はpost-hoc/developmentであり、新held-outや旧RA-D0の`U3<L2` witnessへ転記しない。
旧B2へJ1–J3・returnを追加せず、旧B0_saved/B0_ideal/B1_num/B2_num/B3_numと旧分類を保持する。
CTSの情報access・取得費用を別記し、現在のtoyでPauli情報が取得可能な点も保持する。

## G2で返すGO/STOP判断材料

同等sampling対照後に追加自由度の候補が残れば、その候補・対照だけの有限confidence/native検査をGPTが検討する。
差がなく未判定幅が大きい場合はnegativeとせず、不足するbounds・precision混合・range/capsを明示する。
適切な上下界が改善余地を小さく抑える場合は主線の縮小、既知IS/低次数吸収だけで説明される場合は独立claimの縮小をGPTが判断する。

strict科学witnessは同一class/queryで有効な`U3<L2`。
headroomは有効な`U2-L3`等を用い、B3 inner minimumをL3へ使わない。
過去の5%/10%を自動転用せず、次の正式実行で必要なmaterialityは結果前に固定する。

## 維持する禁止事項

全面v4 production、新registered B2/B3 grid、旧run retry、新angle/synthesis/dictionary、
分子・DF・NPZ・GPU・quantum matrix/circuit/trajectory、追加backend pilotへ自動移行しない。
新アルゴリズムの採択・性能・新規性・論文着地点はGPT側の判断。
今回の文書記録では新solver・tests・数式検算・科学実行は0。旧STOPとconsumed markerは保持する。

## 入力identityと保存

参照元はrootの未commit文書
`/home/abe/Project/Partially Randomized Trotter/track_b_G1_scientific_review_20261009.md`。
root HEADは`e098c54c78f589055082f9cfc2b13de50c90ca94`、branchは`all-r-coherent-opt2-reoptimization`。
rootは最新G1証拠のbranchではなく、方針文書の入力元としてのみ使用した。
本文57,003 bytes、SHA256は`889729bc0d81c0984b391b6011a5a4a9b683fb730d15354424f5a0afd4570480`。
指定した一文書だけをbyte一致で保存し、root/Track Aの作業や他の未commit資料を含めない。

新しい現在地は研究概要へ追記し、旧本文を保持する。既存758 pathを照合し、概要追記以外の757 pathは不変。
[受領manifest](../../../artifacts/track_b_g1_scientific_review_intake/2026-10-09/evidence_manifest_v1.json)に公開path・入力・境界を記録する。
