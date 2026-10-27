# Pauli-JW代表開発ラインの決定

**選択肢A `codex/pauli-jw-research-summary-20261026` を第一推奨とする。** 計画base/着手時remote tip `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`。
成果の固定版はこのcommitと後続H6固定refのまま保全し、開発tipは新codeの版として別に扱う。
今回branchの役割決定・統合計画を確定したが、branchへコードを書き込んでいない。

## 候補の正確な参照

|候補|既存branch|着手時remote tip|
|---|---|---|
|A：summary開発代表|codex/pauli-jw-research-summary-20261026|`b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`|
|B：後続H6実装branch|codex/h6-hybrid-fairness-extension-execution-20261027|`00958ae1945c780718b2deef07d5dab1096a5cd8`|
|C：P2開発branchの恒久代表化|codex/p2-pauli-jw-pipeline-integration-20261027|`04f4cd9cdc8e348a9f0a182ead9cd07f1ee69696`|

関連17 refのmetadataだけを照会し、全branch再監査はしなかった。他の既存Pauli名のrefは個別実行/結果/準備の入口で、
新しい明確な非凍結汎用開発代表は確認できていない。Cは既に存在する技術開発branchとして比較した。
「未確認」は他環境での不存在/利用終了を意味しない。現行worktree/job/runtimeは調査していない。

|選定基準|A summary|B H6 execution|C P2 development|
|---|---|---|---|
|必要API|57 path中56あり、1 file追加|既存57あり|既存57＋P2あり|
|H4/H6依存closure|56は同blob、H6 adapter追加で選定closure成立|P2直接base|P2実装の現状|
|科学SOURCEとの分離|保存資料入口から非凍結code入口を明示できる|固定4候補の完了run/実行handoffと密接|技術開発だが後続H6履歴を継承|
|現行契約への影響|旧commit/結果/runnerを変えず新pathのみ|今後の共通開発で専用実行refを動かす理由が弱い|新roleの正式採用が必要|
|将来再利用|分子横断成果入口から共通composerを辿れる|H6固有名/用途が一般化を誤解させやすい|開発用途には適合、Aとの二重代表化は避けたい|
|技術test|同じinitializer・同じ依存不足、mockは可能|同じ制約|過去109 PASS、実stack未確認|
|必要差分|P2の13 path＋H6 adapter1、拒否修正|P2の13 path、拒否修正|移植0だが拒否修正/代表役割変更が必要|
|旧SOURCE/STOP保全|既存bytes不変、固定後続refを参照|物理資産は触れずとも実行refの役割混同が残る|履歴は保持、原SOURCEと新codeは要分離|
|重複開発|既存H6 fileを同pathで再利用、再実装なし|追加移植不要|再実装不要、A/C両方を今後更新しない|
|運用上の妥当性|既存成果入口・将来開発入口を版で区別し一元化|固定実行/結果branchは保全が適切|代替案として可能、P2試作のまま自動昇格はしない|

Aは1追加moduleでBの専用実行refを動かさずに済む。新しいbranchは作成せず、Aを共通開発tipとする。
同じfileが固定過去branchにも存在することは歴史保全であり、新しい独立H6実装を保守する意味ではない。
今後の変更はAへ選択差分として集め、科学runはその使用code SHAを固定する。旧SOURCEへ新codeを暗黙適用しない。

## 成果とコードの役割

- 固定summary `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`：保存された比較と索引の時点を保持。後続H6の結果で旧表を上書きしない。
- 固定H6 `00958ae1945c780718b2deef07d5dab1096a5cd8` と科学SOURCE `887b69e449cb3c378286f09d0d7626599850fb67`：完了handoff/結果/契約の別版。
- Aの将来tip：新非凍結CLI/composer/adapterコードと技術testの版。旧結果を生成したSOURCEとは表示しない。
- CのP2固定commit：採用元・過去109 PASS記録。採用後に全Pauli開発の別主線として並行保守しない。

代替案はCを明示的に恒久開発代表へ昇格しAを成果専用に固定すること。この場合も現版の拒否不足を修正する必要がある。
ただし今回の第一推奨・次回実装計画はAに一本化する。Bへの書込み、新規統合branch作成は提案しない。
