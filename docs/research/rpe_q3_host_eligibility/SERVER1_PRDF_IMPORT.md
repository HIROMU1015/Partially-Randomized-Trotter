# サーバー1のPR-DF専用profileの取込み

固定 `9d022a0f7cc953ba3c1dac0b5e514e295c429fbe` の
`docs/research/rpe_distributed_h4_plan_20261117/server1_prdf_runtime/` を
[evidence/server1_prdf_runtime_9d022a0f7](evidence/server1_prdf_runtime_9d022a0f7/HANDOFF.md)
へ元byteのまま保存した。26ファイルを取り込み、元 `EVIDENCE_FILES.json` の25件の
hash/サイズを照合した。[固定locatorと検査](checks/IMPORT_SERVER1_PRDF_9D.json)。

公式REQUEST `48494ae31276a64c46ce`、SOURCE28d、route23160cc87について、
新しい科学計算・実dry-check・遠隔接続を行わず、保存公式READINESSと
HOST_METHOD_READINESSからサーバー1のPASSを採用した。

`EVIDENCE.json` のprofile overrideと証拠選択はroute kind/view commitで行う。
新しい配分はrequest hashも照合されるため、同じrouteというだけでPASSにしない。
通常のFull PF/B0I profile、旧3.12でのPR-DF FAIL、その入力・結果は保持した。
旧集計は [BEFORE](checks/BEFORE_SERVER1_PRDF_HOST_ELIGIBILITY.json) に保存している。

READINESS最上位のPython3.12.3はcheckerの版である。
adapterの `command[0]` は専用launcherで、実runtimeは3.11.0rc1。
profileのhash、公式READINESS/解決後REQUESTのhash、SOURCE/input/route、
HOST_METHOD_READINESSの呼出path・実executable・version_info・元AST guardを照合する。
poll時もlauncherと実binaryのhash・実versionを再取得するため、共通Pythonへの
fallbackや同pathでのbinary差替えは新しいclaimを許可しない。

9d取込直後、後続route準備前のcheckpointは次のとおり。
[取込時の固定記録](checks/SERVER1_PRDF_IMPORT_RESULT.json) と
[現在の集計](HOST_ELIGIBILITY.json) を区別する。

| 方式 | 保存証拠でPASSのhost |
|---|---|
| Full PF | server2・server1・local |
| B0I | server2・server1・local |
| PR-DF | server2・server1 |
| PR-Pauli | なし |

変化はサーバー1 PR-DFの1セルだけ（FAIL→PASS）。全12セルで8 PASS・4 FAIL。
RomuTail PR-DFと全host PR-Pauliの不足は残る。
これは今回の固定REQUESTの実行経路の準備状況であり、未評価MSEの適合や科学的最安を
示さない。本番queueは起動していない。

[取込結果](checks/SERVER1_PRDF_IMPORT_RESULT.json)。focused testsは13件PASS。
追加3件は専用launcher/runtime結合、共通profile fallbackとhash不一致の拒否、
poll時のlauncher/実binary hash取得を確認した。著者自身による人工検査である。


## 割当から実行引数まで

[人工profile受入](checks/PROFILE_ACCEPTANCE.json)で、実controllerの割当から
mailbox submit・保存WORKER_PROFILE・supervisor・adapterのexecute引数までを通した。
PR-DFは専用launcher、Full PF/B0Iは従来interpreterを維持した。
割当後のpolicy変更や再接続でも、同attemptの保存profileとassignmentは不変。
専用ルールのない共通profileへのfallbackは拒否した。Popenはmockし実Q3は0件。
[関連24検査](checks/SERVER1_PRDF_PROFILE_TESTS.json)がPASS。

通常prepareは `EVIDENCE.json` のprofile_overridesを自動読込みし、
`worker_profiles/server1.json` に共通profileとPR-DF route専用ルールをまとめる。
実workerと新しい実入力checkerへ、この生成profileを渡す。
[生成済みprofile](checks/server1_prdf_profile/worker_profiles/server1.json)と
[保存計画を通常結合処理へ通した結果](checks/server1_prdf_profile/READY.json)を保存した。
後者は新規候補選定やlive queue照合を再実施したものではなく、保存4REQUESTのprofile更新である。
同じREQUESTは公式PASSを照合して再利用できる。新しいREQUESTは同じroute規則で専用profileを選び、
新しいcheckerでdry-checkする。呼出し側のPython3.12を科学adapter用Pythonとして選ばない。

このcheckpoint後も12セルすべてをPASSにする整備を継続する。
RomuTail PR-DFは別担当の成果を取り込み、PR-Pauliは既存descriptor対応の固定routeを確認中。
既にPASSの8セルを、他セルの不足を理由に一律停止しない。
