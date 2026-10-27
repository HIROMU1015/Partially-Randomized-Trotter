# P1採用評価・選択統合判定

LOCAL / 2026-10-27T22:34:51.199664+09:00。評価対象はP1 `af54aeb7ebac042a3ca02379d61516ac635feef9`、固定base `f29eafbf4412193e8e8660d9f051889c2cba684b`。
**DF pipelineを条件付きで選択採用し、`trotterlib/__init__.py`のlazy化は今回の統合から除外する（B-3選択肢4）。**
P1の研究配線をDF開発主線へ接続する方針は採用推奨。実装移植・科学的同等性の承認とは分ける。

## 21変更ファイルの結論

|判定|件数|意味|
|---|---:|---|
|ADOPT_AS_IS|2|副作用のないpackage入口2|
|ADOPT_WITH_CHANGES|12|composer、H4 adapter、core binding、人工fixture、CLI、回帰test、関連案内6|
|DEFER|6|global lazy facade1、固定P1の履歴文書5|
|DO_NOT_ADOPT|1|旧TEST_MANIFESTを新統合の実施済み証拠として移植しない|
|ALREADY_PRESENT|0|固定差分が既に取り込まれたファイルはない|

正確な21 path、機能範囲、理由、変更・test条件は[判定CSV](ADOPTION_DECISION_MATRIX.csv)。
保留/不採用は原本削除を意味しない。[P1の公開6資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/af54aeb7ebac042a3ca02379d61516ac635feef9/docs/integration/p1_df_pipeline_h4_20261027)は固定履歴として保全・再利用する。

## 実装評価と必要な修正

composer→adapter→既存coreの責務分離、SOURCE/cohort/input/compiler/metricのbinding、STOP/欠損の保持、実科学mode拒否は有用。
`ScientificContext`はDFのrank/partition必須型、stage列もDF用であり、全研究共通schemaとして採用しない。
H4のrank12/生成順prefix/S2/q1/sector/geometry条件はH4 adapterに保つ。
現在`df_core_stages.bind_existing_core`はH4 shots/wrapperを直接選び、fixed-signalに不要なphase-RPE moduleもimportする。
このbindingとprepare policyを明示的な研究側指定へ変え、数値kernel自体を変更しない。

lazy化を除外してもplan/no-output mock配線は成立した。一方`write_new_result_directory`は既存executorのatomic writerをimportし、旧eager初期化を通る。
出力だけを目的にglobal lazy化を採用せず、既存writerを維持し必要な依存不足を新CLIで明示拒否する。新directoryは依存解決後にのみ作る。
科学計算をするreal provider、q>1、real input loader/identity、mean/bias/qualified N/Gは引き続き未実装・未検証。

## 互換性と検証の強さ

413 exportの名称・順序・module/attribute対応は静的一致。ただし完全互換判定は不可。
今回追加**19 PASS / 0 FAIL / 0 SKIP**。固定P1の**過去62 PASSは再実行していない**。
全413の動的取得比較は人工provider、実moduleでのidentityは`rte`のtype/定数/関数だけ。
旧初期化はoptional stack不足で失敗し、P1 root importは成功する。star importは依存不足で失敗し、lazy化でも全依存を回避できない。
reload/属性cacheとimport順序、副作用タイミングの差があり、CUDA準備等は静的確認のみ。
詳しくは[import評価](LAZY_IMPORT_COMPATIBILITY.md)と[新規技術test](TECHNICAL_TEST_RESULTS.md)。

## 正確な統合先と次作業

代表branch `all-r-coherent-opt2-reoptimization` の着手時remote tipは `f29eafbf4412193e8e8660d9f051889c2cba684b`、P1固定baseと同一。後続差分0。
将来の選択統合baseをこのSHAに固定する。実施直前に再確認し、進んでいれば関係pathだけを再比較する。
ローカル元checkoutは別HEADかつdirtyなので切替・commit・上書きの対象にしない。
[選択統合計画](SELECTIVE_INTEGRATION_PLAN.md)は採用14ファイルの必要差分だけを同checkoutへ取り込む順序・受入条件・rollbackを確定した。adapterへのpolicy移設とfixture同期も修正採用に数えた。
全branch merge/cherry-pickや新しい統合branchは使わず、今回は実装取り込みcommitを作成していない。

P2は[引継ぎ](P2_HANDOFF.md)に既存Pauli API/runner/frozen namespace、条件差、mock sliceを具体化した。
summary `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`は入口として保持する。
より新しいH6 Hybrid branchの正式handoffは追加段階の完了を記録しており、旧計画の「未実装/未取得」を現況へ転用しない。
これは公開資料の版対応の確認であり、稼働環境・科学結果の再監査ではない。

## 保護と終了

研究branch/main、P1 code、元dirty checkout、既存runner/SOURCE/input/raw/result/cache/CEM/manifest/STOPに書込みなし。
新科学計算/sampling/実build/compile/数値再解析0。B3本体・5保護worktree・9保護ref内容へのアクセス0。
他研究の現行checkout/非公開実行環境、SERVER_A/B SSH/転送、他job/共有環境変更0。
作業中に非対象のlocal ref/登録HEADの並行更新をmetadataで観測した。元checkoutのHEAD/index/dirty状態は不変で、非対象の更新を巻き戻していない。内部対応は非公開詳細だけに保持する。
N1/N3旧SOURCE未解決は保持し、H6/H8展開・P2実装は開始していない。
正式指示2026-10-25実ファイルを確認し、§8.1/8.4/10を適用。新しい承認証明/grant制度は作成していない。

**次のユーザー許可候補**：本計画の14ファイル相当の選択差分をDF代表ラインへ取り込む技術実装・限定mock回帰・文書化・通常commit/push。
Codexはその明示指示後にだけ実施する。global lazy化、P2実装、実科学検証は含めず、各々別範囲として判断する。
公開・remote照合後mandatory STOP。
