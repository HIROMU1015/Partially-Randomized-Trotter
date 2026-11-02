# Track B G10：RSS問題の限定修正と実行前source準備

## 1. 目的

G10 one-shot実行は、RSS上限512 MiBを超過して `G10_TECHNICAL_INCONCLUSIVE` で停止した。

その後の技術調査では、全resultの再帰複製と一括JSON化が最も有力なメモリ増幅要因と判明した。

保存JSONのみの独立検算では、同一bytes・SHA256を生成する条件で以下の結果が得られている。

- 従来の一括方式：peak RSS 508.91 MiB
- 逐次方式：peak RSS 253.25 MiB

ただし、元G10の正確な例外発生行は未確定であり、逐次方式によって本番G10が512 MiB以内で完了することも未確認である。

**今回の目的は、G10の科学的条件を変更せず、メモリ問題と失敗時の記録方法を修正した新しいsourceを準備し、GPTの実行前レビューへ提出することである。**

G10の本番再実行は今回の作業範囲に含めない。

## 2. 対象repositoryと固定証拠

Repository：
`HIROMU1015/Partially-Randomized-Trotter`

G10 frozen science source S：
`05c5ef23fce775a822ab5686f5da2f0d77675864`

G10 authorization-only A：
`f5cd0755424d1b11e2249cc115518d84fb8bb8d3`

G10 technical failure result R：
`e429c99d77b3222c5cca62750b2d111f87e4cb50`

最新RSS原因調査：
- Branch：`track-b-g10-rss-cause-audit-20261010`
- Commit：`21f66137ab205fb0d71ee6ee001844f4c2d725e3`

調査報告：
`docs/tracks/algorithm_codesign/g10_rss_failure_technical_investigation_20261010.md`

まず、この調査報告・technical findings・provenance・元G10の実行契約・sourceを確認する。

新しい修正作業は専用branchで行い、既存のS/A/Rと調査commitを保持する。

既存branchの履歴を書き換えず、旧result、marker、STOPを変更しない。

## 3. 採択する修正範囲

今回の修正対象は以下の三点に限定する。

### A. JSON serializationのメモリ使用削減

既存実装では、`serial(result)`による全containerの再帰複製と、`json.dumps`による一括文字列生成がメモリ増幅につながっている。

以下を満たす逐次的な出力方式を実装する。

- 全resultの再帰複製を回避する
- 一括chunk list/joinを回避する
- 必要な型変換を逐次的に行う
- UTF-8出力をbounded chunkで処理する
- 出力bytesとSHA256を逐次計算できるようにする
- 既存の出力byte capを維持する
- 既存のJSON schemaと意味論を維持する

特に、Fraction、tuple/list、文字列key、Unicode、JSON formatting、非finite値・非対応型の拒否条件について、既存方式と同等であることを確認する。

調査では非文字列dict keyに対する一般的な置換の非同値性が確認されているため、実際のsourceで使う型・schemaを調べ、適用可能な範囲を明示すること。

同じ論理的payloadに対しては、可能な限り旧出力とのbyte単位の一致を検証する。

保存済みG10 JSONを使ったI/O-only検算を行ってよいが、その結果を新しい科学的検証として扱わない。

### B. 不要な参照保持の削減

調査報告で指摘された以下の保持構造を確認する。

- `old_raw` / `old`
- m5 anchorのdeepcopy
- `pending`
- row resultとevent参照
- loop localによる参照保持
- その他の不要な中間データ

不要になった参照は、安全に寿命を短縮する。

ただし、必要なevent、native IR、budget、synthesis identity、m5再利用情報を削除したり、省略したりしてはいけない。

また、`del`や`gc.collect()`によってRSSが必ず低下するとは仮定しない。

科学的処理順序、sampling分布、operator mean、誤差会計、結果の保存項目を変更しないこと。

### C. Guard・失敗時の保存処理の改善

元G10では、例外発生後の大きなJSON serializationと保存処理がRSS guardの範囲外で実行される問題が確認された。

以下を改善する。

- 正常時のserialization、write、flush、close、最終確認を適切な監視範囲に含める
- 出力途中のfileを科学的な完成結果として扱わない
- 正常完了とtechnical failureを明確に分離する
- 失敗時に巨大なresultを再び一括serializeしない
- 失敗時の記録をboundedな小型receiptとして設計する
- 失敗時にも可能な範囲でprovenance、failure reason、STOP、partial output identityを記録する
- 一度消費したone-shot markerを再利用しない

可能ならstage別のcurrent RSS、peak RSS、処理時間を記録し、再度停止した場合に原因を切り分けられるようにする。

ただし、OSによる強制終了やメモリ枯渇時にも必ず記録できるといった、保証できない完了性を主張しないこと。

成功時の科学的な出力内容と、失敗時の新しい記録契約を区別する。

## 4. 固定する科学的条件

G10の研究目的・比較条件・実行意味論は変更しない。

特に以下を維持する。

- `p=(1/5,3/10,1/2)`
- `x=5/7`
- Taylor次数 `m=3,5,7`
- G9から引き継いだ3-qubit provider
- 各次数内で同一の有限演算子を比較する契約
- 登録済みの比較手法と評価指標
- native T/CX/1Qの計数規則
- confidence・error budget
- sampling・precision・seed
- operator meanとcontrolled phaseの意味論
- 既存のsource/provenance保護条件

RSS 512 MiB、AS 1536 MiB、その他の実行・出力上限も変更しない。

今回の修正を理由に、新しい科学的条件や実験対象を追加しない。

## 5. 実装・検証方針

Codexは、上記の科学的固定条件と修正scopeの範囲内で、具体的なコード構成、streaming encoder、memory cleanup、failure handler、focused testsを自律的に設計・実装してよい。

細かな技術修正ごとにGPTへ戻る必要はない。

修正後は、以下の検証をまとめて実施する。

1. JSON serializationの型・schema・bytes同値性
2. 保存G10 JSONによるread-only I/O検算
3. 新旧方式のメモリ使用量比較
4. 不要な参照保持の解消と必要な参照の維持
5. 出力byte cap・guard failure・I/O failureの処理
6. partial resultとfinal resultの区別
7. 旧sourceとの科学的条件・主要意味論の不変性
8. 旧result・marker・STOP・protected pathの不変性
9. 修正sourceのprovenanceと再現性

必要なsynthetic test、保存値監査、非科学的memory stress testは実施してよい。

ただし、本番G10 runner、native synthesis、量子回路評価、sampling、LPなどの新しい科学実行は禁止する。

同じ科学条件での修正であることを、説明だけでなく、source差分とfocused validationから検証すること。

## 6. 今回は採択しない変更

以下は行わない。

- RSS capの引き上げ
- G10の本番再実行
- 旧one-shot markerの解除・再利用
- 既存17行の科学的結果への昇格
- native event・IR・予算fieldの省略
- Fraction等の精度を下げる変更
- 科学的比較手法・error budget・samplingの変更
- 新規合成・分子計算・DF・PR/QPE検証
- 新しいp、x、m、provider、seedの探索
- G11以降への進行

row単位の大規模spoolingなど、科学的出力構造や実装を広範囲に変更する方法は、今回の限定修正で不十分と判明した場合に限り、追加候補として報告する。

自動的にscopeを拡張しないこと。

## 7. 新しいsource・contractの扱い

修正は新しい準備source S2として整理する。

旧S/A/Rを上書きせず、変更したsourceと変更理由を明確に記録する。

科学的条件は旧G10と一致させる一方、serializationやfailure reportingの契約変更は隠さず、新旧差分を明示する。

新しい実行契約・authorization方式が必要であれば、実行前レビューに提出できる設計として準備する。

ただし、**今回のS2準備完了を、G10本番実行のauthorizationとして扱わない。**

新しいauthorization-only child、fresh one-shot markerを用いた本番実行は、GPTの実行前レビューとユーザーの別途明示承認を経て行う。

## 8. 成果物と報告

一つの作業単位として、以下を完成させる。

- 修正済みsource
- 修正前後の差分・修正理由
- 技術的な実装契約
- focused tests・memory validation結果
- JSON同値性・provenance監査
- 保護対象の不変性確認
- 残る技術的リスクと未確認事項
- GPT実行前レビュー向けの引継ぎ報告

報告では特に次を明確にする。

1. どのメモリ増幅経路を除去できたか
2. メモリ使用量はどこまで低下したか
3. 成功時のJSON出力内容は維持されているか
4. 科学的意味論は維持されているか
5. 失敗時のbounded記録は成立したか
6. RSS 512 MiB以内で本番完了できることを保証しているか、それとも未確認か
7. 本番前に残る検証・承認は何か

資料・結果はGitHubから取得可能な形でcommit・pushする。

push後はremote SHA、worktree clean、主要成果物・監査資料のremote取得可能性を確認する。

既存の研究結果、実行記録、marker、STOPは保護する。

## 9. 最終停止条件

修正sourceの準備とfocused validationが完了したら、mandatory STOPする。

技術的な不具合が見つかった場合は、承認済みscopeの範囲内で修正を続けてよい。

ただし、科学的意味論・固定実験条件・resource capを変更する必要が生じた場合は、変更せずGPTへ戻す。

**今回の完了条件は、G10の科学的結果を得ることではなく、メモリ問題への修正sourceを作成し、その実行前レビューに必要な証拠を揃えることである。**

最終的にbranch、source commit、主な修正、検証結果、監査結果、残るリスクを報告し、GPTへ判断を戻す。

本番実行・再実行は行わない。