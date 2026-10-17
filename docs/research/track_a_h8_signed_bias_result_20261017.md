# H8 signed bias成分復元・数値整合性検証：run01完了

**固定SOURCEで一回実行し、全184候補368軸・3976検査がPASSした。不合格・欠測によるSTOPは0。**

SOURCE `64ec6cbba7e48ac7c04a6ad0b38c3e76f0f14526`、準備REVIEW `f9bf3da5f6cffca95f269dd65bdeb806d875234d`。
[summary・368軸CSV・binding・native exit・資源・保全証拠](../../artifacts/resource_applicability/track_a_h8_signed_bias_run01/2026-10-17/README.md)。
[実行前の固定契約](track_a_h8_signed_bias_readiness_20261017.md)。

## 対象と認可

H8直線鎖1.0 Å/STO-3G、保存tol-only DF28、16 modes/sector4900、T=.8。
旧32＋Phase A追加152の固定184決定論候補、各real/imag。
旧S2はprefix7/14/21、追加S2はprefix8〜20（14を除く）、S4追加はprefix15〜21、q1/2/4/8。full PFはprefix28のS2/S4。
利用者の今回の明示承認を別one-shot grantへ結合した。公開済みplanのbyte列・SOURCE・科学条件・許容差は変更していない。
plan内の旧未承認flagを上書きせず、今回の実行権限は別の承認済みgrantで与えた。

|対象|候補|軸|結果|
|---|---:|---:|---|
|旧B0+Iのsigned成分復元|12|24|保存端点からdiscard/PFを復元し、親B0・scalar位相・totalと照合PASS|
|既存signed成分の整合性|172|344|保存成分・端点・closure・保存forwardの範囲でPASS|
|全固定集合|184|368|全3976検査PASS、不合格0軸、候補除外0|

## 数値整合性の範囲

相対・絶対許容差はともに `1e-12`。real/imag別に比較し、軸平均や許容差緩和による救済は0。
`discard = truncated - full_H reference`、`PF = corrected - truncated`、`total = corrected - full_H reference` を照合した。
full PFはPF成分だけで、discard欄は空欄とする。

|検査|今回の最大絶対差|
|---|---:|
|保存成分と端点差の一致|0|
|保存totalと端点差の一致|0|
|component closure|0|
|scalar位相による保存overlay端点の一致|0|
|保存correctedとforward経路の一致|2.204902926905561e-13|
|非回転full-H referenceの一致|2.1094237467877974e-15|

この表は既に生成された検査出力を集約したもの。新signalや参照の生成ではない。
保存成分と復元成分には同じ端点を使うため、差0やclosure0を独立した物理的証明とは扱わない。
scalarは保存full-Fock規約・stored-rational算術receiptへ照合した。DF行列からの再抽出は行っていない。
full-H referenceは回転せず、親B0・保存参照との一致を確認した。

旧12 overlayのforward証拠は保存親B0の経路までであり、別overlay経路が保存されていたとは報告しない。
追加B0+Iのforwardもreferenceとscalarを共有する。成分別の完全に独立した証明や厳密bias保証は未認定（null）。
旧native correctness projectionは8構成のみで、残4構成へnative PASSを補完しない。
Phase A prefix19の元runtime SHA欠測16候補は、今回の数値整合性PASSでも解消していない。公開時の別hashと区別して保全した。

## 実行と監査

実行ID `h8-signed-bias-20261017-run01`。
Python native wall 0.303020秒、runner body 0.147584秒、peak RSS 47.71 MiB。
native exit0を親processがwait4でその場で観測し回収済み。起動した数値runnerは1、runner内childは0、実行試行1・修正0・再実行0。

SOURCE closure6件と固定26入力のbytes/SHA、gzip展開元byte列、旧SOURCE/profile/reference/worker来歴を確認した。
完了前にも入力とSOURCEを再照合した。one-shot lockと消費済みgrantはruntime領域に保持する。
既存Python/venv・共有環境・他job・CPU affinityを変更していない。
保護した既存3 worktreeのHEAD/dirty状態と、B2主要7証拠は不変。B2の元STOPも変更していない。
新worker pool・observer・GPU query/useは0。小さい保存JSONを一度parse/index化した既存の軽量実装を用いた。

新規signal/compile/sampling/SCF/DF/state/fitは0。
非相殺条件のshot再計算・PR費用比較・再会計・最良候補ランキングは0。
source/testsの追加変更0、追加科学レビュー・subagents0。

公開用bundleは結果CSV・監査projectionのみ。raw入力、実runtimeの全出力、承認済みgrant・lockはhome配下の実行領域に保全し、Gitへ追加しない。
全checkの詳細を含む元axis resultは1,376,537 bytesで、そのSHAはexecution bindingへ記録した。

## 次の担当と停止

**ユーザー**が結果を確認する。後続解析は別途指示がある場合に限り、**Codex**がその範囲で対応する。
今回のgrantはこの完了済み一回だけの認可であり、追加stageの起動権限を与えない。mandatory STOP。
