# Track A 原稿v0.2：claim/evidenceと編集監査

2026-10-05。[本文](track_a_resource_study_v0_2.md)と[Supplement](track_a_resource_study_supplement_v0_2.md)を対象とする。
利用者提示の外部GPT review `CLOSE_AS_TECHNICAL_RESOURCE_NOTE`を採用した
[scope判断](../research/track_a_technical_resource_note_scope_decision_20261005.md)に従う。
元原稿commitは`4c23453c541700c6a41ba71fc5ec9323b53858d6`、
evidence commitは`5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc`。
v0.1、図、science artifactを上書きしない。

## 1 Reviewへの対応

| 指摘・依頼 | v0.2での対応 |
|---|---|
| technical resource/application noteへ収束 | Abstract末尾・§1.1・§6.2で位置付けを限定。新手法・一般的最適性を主張しない |
| Abstractを何をした／分かった／scopeへ | 三段落に圧縮し、保存値・登録集合・点推定・元精度transferを区別 |
| Introduction末尾へ三寄与 | §1.2に有限候補resource map、近接discard stress test、適用条件分析を明示。「初めて」は付けない |
| Güntherとの差分を明示 | §1.1の一段落で既知のpartial・DF部分保持・truncation・化学資源評価と今回の限定mapを区別 |
| Discussionを整理 | §6.1 scientific limitations、§6.2 scope choices |
| stage・commit・内部statusを補足へ | 本文の結果リンク・旧selector履歴・terminal文字列・執筆workflowを補足へ集約 |
| 科学値・図・claim不変 | 4主図・図S1と全assetを再利用。式・候補・閾値・元科学statusを維持。新科学実行0 |

## 2 C1〜C3の章間整合性

本文Abstract、Introduction、Results、Conclusionを通読して限定の強さを揃えた。
下表は意味の監査であり、キーワード検索による科学的正しさの証明ではない。

| claim | Abstract / Introduction | Results | Conclusionと限定 |
|---|---|---|---|
| C1 nearby discard stress test | ε=.05、P=0の登録集合。低いB2 primary点推定が粗いdiscard gridだけに依存しない | §3：B0 L4/5×q1/2/4/8を追加。最良登録B0 L5 q1との43.07%点差 | 同じ構成・精度・point差。未登録prefix、全deterministic法、厳密winnerへ一般化しない |
| C2 precision dependence | 一つの保存値精度感度でB2内部q4→2→1。method逆転とはしない | §4.1：development218、ε=.005〜.1、302表示点、隣接点間の切替、区間重なり | 同じ保存値の一つの感度解析。302独立条件、連続crossover根、最適設定の保証ではない |
| C3 frozen transfer vs applicability | 固定5構成を1.30 Åへ移し元ε=.05で支持。厳しい側では現行規則不適格 | §5：結果前固定transferと使用済みgeometryのPOSTHOC感度を別に記載 | 元精度の支持を維持。held-out再最適化・method optimum・新blind evidenceへ拡張しない |

共通scopeはlinear H4、STO-3G、保存DF rank12・8 system qubits・状態、T=.8、
二次DF-prefix PF、canonical finite-RTE、δ=T/q、対称軸十分shot規則、指定Qiskit compilerである。
state preparationはprimaryから除外し、P感度を仮想secondaryとする。
C4のsame-domain/selector監査はSupplement S3、C5のsame-R競合は§4.2・図4、
C6の候補集合を揃えたP感度は§6.2・Supplement S4を維持する。

## 3 過剰claimと科学値の保存

本文・caption・Supplementで次を確認した。

- B2の一般最適性、rank3/q1の普遍最適、r4/r8の厳密winnerは主張しない。
- M2固定5構成をheld-out再探索とせず、未移送のB0 L5結果を1.30 Åへ補間しない。
- ε_minは現行対称軸shot規則のstrict境界。原理的精度限界とはしない。
- 302点を独立実験とせず、工程名ではなく一つの保存証拠によるprecision sweepとして述べる。
- paired-axis covarianceとengineering点±2SEを維持し、formal CI・同時winner保証とはしない。
- finite-time signal・analytic shots・compiled RZをenergy/QPE/RPE最終総costへ拡張しない。
- 未分解B0 error、実準備費用、det/random/basis別費用の欠測を埋めない。

[機械的editorial audit](../../artifacts/resource_applicability/track_a_manuscript_v0_2/2026-10-05/editorial_audit.json)は、
Methodsの不変性、Results数値列、図リンクと科学的caption、全46 frozen filesのhash、
local Markdownリンクとcommit provenanceを検査する。
図4末尾の編集上の「固定図番号」説明だけをSupplementへ移し、科学的captionは維持した。
この検査は数値の再計算・元scienceの再実行・OS sandbox・外部再現ではない。
旧21 synthetic/local testsの結果はv0.1 auditにあるが、今回はその再実行結果と称さない。

## 4 書誌とprovenance

Related WorkのGüntherら、Kanasugiら、Simon–Loveは一次公開資料のversion・Abstract等を再確認した。
GüntherらとのDF・truncation重複は既存の本文Sec. VII・Appendix D.2照合を維持する。
review文中の2503.05647リンクはGünther論文を指すため、Kanasugiには別の2603.22778v2を使う。
その他の書誌とReferencesはv0.1から不変。全先行研究の定理・実装を網羅監査したとはしない。

v0.1 SupplementのPM-2 launch hash
`bea4cf00f08e28b2ddf90550cd485db7fce89e39`は存在しない転記値だった。
global validation manifestのlaunch_head
`bea4cf00f08e1c2d5fba3649c6f58cc31490fd1e`と実在commitを確認し、
v0.2 Supplementだけを訂正した。元manifest・v0.1・scientific resultは不変である。
この管理情報の訂正は新しい科学値・科学結果ではない。

## 5 証拠階層とSTOP

利用者提示の外部GPT reviewを採用した投稿形態判断と、執筆者のclaim audit、科学local evidenceを区別する。
technical noteへの収束は採択保証ではなく、独立再現・immutable CIでもない。
新signal/sampling/circuit/compile/分子計算/quantum shot/GPU、NPZ/runtime/cache accessは0。
scientific_next_stage_authorized=false。PM-3、追加96、strong synthesis、高次PF、
別geometry/分子、energy/RPE、Track B統合へ進まない。
次は投稿先と形式の選択であり、scienceや投稿の自動実行ではない。
