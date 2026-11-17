# PR探索：候補単位の保留とR08継続

**照合未解決の旧jobを候補単位で保留し、他候補・他方式を継続するよう修正した。** 旧jobを無条件に同等と認めたり、再計算して埋めたりはしない。数値UNKNOWN・保存/起動異常の既存停止は維持する。後着READYも、別候補の準備中に同じqueueへ投入する。

R07は2026-11-17 07:19 JSTにPauliの`STALE_JOB_NEEDS_REVIEW`で停止した。停止要求は[PRIOR_STOP_REQUEST](PRIOR_STOP_REQUEST.json)。旧jobは完了していたが実行route/SOURCEの同一性が未確認だったため再利用を拒否したこと自体は適切だった。この保留を探索全体へ広げた経路を修正した。旧STOP・入力・raw・結果は変更していない。

## 停止範囲

|状態|新しい扱い|
|---|---|
|旧jobのSOURCE/routeが未解決、同じjobが進行中、実行先が未登録|canonical configurationを保留。新規retry・無検証の再利用を禁止し、保存適合点は保持。他候補・方式を継続|
|保留jobが完了し、既存のjob key/route/identity/終了gateを通る|同じguardを使って保留を解除。追加Q3なし|
|独立した候補がREADYになる|producerのREADY snapshotを既存installerへ渡し、producer stateを変更せず共通queueへ投入|
|内部batchの区切りに達する|採択済み全体予算の範囲で次のbatchへ。有限予算は増やさない|
|数値UNKNOWN、起動/保存の重大異常、稼働SOURCE・共有入力の破損/無断変更|既存の停止を維持。準備は次の独立callの前にyieldし、進行中workerは結果を保全して終了させる|
|実行可能な候補が残らず保留のみ|入力/照合待ちとして終了。探索完了・大域最適・方式不適合とは扱わない|

保留は既存`ROUND_RESULT.json`の`execution_holds`と`held`、段階logの`CANDIDATE_HELD`へ残す。新しい承認台帳や別queueは追加していない。`SAVED_JOB_REVIEW_REQUIRED`は科学的な不足取得要求と区別し、Completionへ再試行として渡さない。再開でも保留候補の二重起動を避ける。

通常CLI、V2、legacy選択、shared Executor、Completion、価格・信号・Q3 workerを再利用した。SOURCE/identity・compiler/lowering・scalar・seed/order・law・decoder・loss・exact τは変更していない。既存の選択順位と価格式は、保留候補を新規投入から外す変更以外をAST/回帰で照合した。

## 検証と再開

[REGRESSION_CORE](REGRESSION_CORE.json)48件、[REGRESSION_QUEUE](REGRESSION_QUEUE.json)34件の人工回帰82件PASS。本番Python 3.11.0rc1でも[追加8件](NATIVE_TESTS.json)PASS。実装担当自身の検査であり、独立レビューや実H4の性能保証ではない。

通常CLIで、未解決候補を保留して他方式が進むこと、保存最安点を失わないこと、同run再開で新規起動0、準備が遅い候補を待つ間にも後着READYのQ3が返ること、有限予算、重大UNKNOWNとSTOPの保全を人工検証した。

[PRELAUNCH_ACCEPTANCE](PRELAUNCH_ACCEPTANCE.json)はR07の実際の停止要求を使い、SOURCE不一致を無視せず候補1件に限定でき、他候補のrouteと保存最安を保持することを確認した。科学callは0件。

R08は[RESUME_INPUT_CHECK](RESUME_INPUT_CHECK.json)のとおり、R07の121件の完了Q3、READY16構成、保留5構成、255点・36 familyとDP cacheを引き継ぐ。保留5件の旧callは再試行しない。開始時の条件付き最安はPauli 410,199,237 RZ、DF 1911681919/3 RZ。実行可能な範囲の継続であり、未評価幅や保留領域の最適性は認定しない。

状態は[STATUS](STATUS.json)、通常入口・再開・回収は[HANDOFF](HANDOFF.md)。実行SOURCEと入力は科学開始前にcommitで固定する。実H4の速度倍率は未測定。
