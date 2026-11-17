# サーバー1：固定PR-DF実入力のruntime readiness

2026-11-17 JST。**指定REQUESTは `REAL_HOST_DRY_CHECK_PASS`。共有環境は変更していない。**
実Q3・compile・本番queue操作はすべて0件。

|照合対象|固定値|
|---|---|
|計画|`33928bf80cca18ce3f1fa3ddde88d27d7e2b7e5b`|
|REQUEST/job ID|`48494ae31276a64c46ce`|
|REQUEST canonical SHA256|`2906eeafee0f39909ef2db9c169924bcb74d48354995ab443e9231f94c0d2e2a`|
|計算SOURCE|`28d976b99cdf8329adcc1b79327e329e947ef774`|
|route|`prdf_d2_split` / `23160cc876807c0300be3de811f767b5810640a7`|
|内側SOURCE|`27e70efe533862f38f4080aabb9cdd3ad1749fe9`。既存pathの実体を確認、path変更なし|
|実際のsys.version_info|`(3, 11, 0, 'candidate', 1)`|
|到達範囲|既存adapterのruntime・AST guard、入力/構成/信号metadata、count scope|

原本は [assignment](inputs/assignments/PR_DF.json) / [REQUEST](inputs/requests/PR_DF.json)。
[公式READINESS](dry/PR_DF/READINESS.json)、[ADAPTER.log](dry/PR_DF/ADAPTER.log)、
[DRY_CHECK](dry/PR_DF/adapter/DRY_CHECK.json)、[JOB_STAGES](dry/PR_DF/adapter/JOB_STAGES.jsonl) を保存した。
SOURCE manifestの8ファイル・入力Git blob 2件のhashと、SOURCE/外側route/内側routeのHEADを再照合した。
詳細は [SOURCE_INPUT_VERIFICATION](checks/SOURCE_INPUT_VERIFICATION.json)。

## 利用した既存環境

専用profileのinterpreterは、既存のshell launcher：

`/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-q3/runtime-acceptance/bin/python3`

その`sys.executable`は `.../runtime/python311rc1/bin/python3`、実バイナリは
`/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-q3/runtime/python311rc1/bin/python3.11`。
launcherのファイル実パスと、exec後の実バイナリ実パスは別に記録した。
既存launcherは同領域のaddon/site-packagesを子processのPYTHONPATHへ設定して実行する。
今回launcher・環境本体・packageは編集していない。shellのactivateや永続環境変数設定もしていない。

|既存候補|実測version_info|今回の扱い|
|---|---|---|
|`/usr/bin/python3`|3.12.3 final 0|PR-DF exact要件に不一致|
|`/home/AbeHiromu/venvs/trotter-common/bin/python`|3.12.3 final 0|実バイナリは/usr/bin/python3.12。Full PF/B0I用に保持|
|`.../runtime/python311rc1/bin/python3`|3.11.0 candidate 1|version適合。今回のadapterは既存addon launcher経由で実行|
|`.../runtime-acceptance/bin/python3`|3.11.0 candidate 1|選択。実依存import・指定実入力dry-check PASS|
|`.../runtime/Python-3.11.0rc1/python`|3.11.0 candidate 1|ビルドtree内の別実バイナリ。今回未採用|

全候補を実行して`sys.version_info`、`sys.executable`、realpath、prefixを確認した。
[INTERPRETERS](checks/INTERPRETERS.json)。選択環境ではnumpy1.26.4、scipy1.14.1、mpmath1.3.0、
ctypes/_ctypesの実importと読込実パス・hashを確認した。[DEPENDENCIES](checks/DEPENDENCIES.json)。
dry-checkが到達した範囲で次の依存不足は発生していない。
実Q3のprocess起動・DP評価・compile内で初めて使う依存や性能は今回未検証。

## 過去の成功との対応

- 同サーバー1の先行dry-checkは同じroute/view/input templateでPython3.11.0rc1を使用し、元guardを通過した。
  [保存記録](checks/PRIOR_SERVER1_ROUTE_DRY_CHECK.json)。このときの小さい配分と今回の実REQUESTは別である。
- 固定計画が保存している過去実Q3 `99ecbaa68b07b0a9ab47` も同じ`prdf_d2_split`/view23160cc87。
  [当時のSOURCE/runtime](checks/PRIOR_REAL_Q3_SOURCE.json) と [exit receipt](checks/PRIOR_REAL_Q3_PROCESS_RETURN.json) は
  Python3.11.0rc1、原AST guard PASS、exit=0。これはサーバー2の別バイナリでの過去記録で、今回再実行していない。
- 同じREQUESTのサーバー2乾式PASSも固定計画から保存した。
  [サーバー2の記録](inputs/dry-check/server2/PR_DF/READINESS.json)。サーバー2の絶対venv pathをサーバー1へ流用しない。
- 前回のサーバー1 Python3.12による停止は [元READINESS](checks/PRIOR_SERVER1_PY312_READINESS.json) に保持した。

## 共有環境と既存方式の保全

[前](checks/PROTECTED_BEFORE.json) / [後](checks/PROTECTED_AFTER.json) のsnapshotは一致。
[PROTECTED_UNCHANGED](checks/PROTECTED_UNCHANGED.json) が比較結果である。
共有trotter-common、既存3.11.0rc1、既存runtime-acceptanceの全directory entryの
名前/mode/size/mtime_ns/symlink先の集約hashを照合した。重要なバイナリ・launcher・既存profile・
Full PF/B0I成功receiptは内容SHA256も照合。固定SOURCE/routeのscripts/srcも内容manifest hashで照合した。
今回はpackage install/build、共有Python更新、既存profile置換、symlink変更、guard緩和を行っていない。
Full PF/B0Iは前回PASSの環境・profile・receiptを保持しており、今回のPR-DF profileへ一括置換しない。

今回新設したのは専用worktree内のprofile・入力コピー・乾式出力・報告だけ。
profileのprefix cacheは今回専用の未使用pathであり、dry-checkはcacheを作成していない。
追加環境の構築は不要だった。

## 中央への引渡し

[HANDOFF](HANDOFF.md) のコマンドと [HOST_METHOD_READINESS](HOST_METHOD_READINESS.json) を使い、
host=server1 / method=PR_DF / job ID / SOURCE / route / interpreterを照合して乾式準備状態を取り込める。
これは環境readinessのみ。MSE判定、Q3完走、科学結果の採用、実行許可やqueue投入の記録ではない。
