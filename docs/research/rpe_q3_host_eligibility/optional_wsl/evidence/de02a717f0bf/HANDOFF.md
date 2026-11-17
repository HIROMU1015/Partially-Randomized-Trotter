# local0-wsl PR-DF専用runtime：実入力fresh PASS

REQUEST `48494ae31276a64c46ce`は、通常の`check_rpe_q3_host_readiness.py`から
新しい出力先へ1件だけfresh dry-checkし、`REAL_HOST_DRY_CHECK_PASS`になった。
checker自身は既存Python3.12.3、実adapterは専用Python3.11.0rc1。
WSLの新規受付は停止のまま、運用spoolは空。科学Q3・信号生成・circuit compile・本番queue操作は0件。

## runtimeとprofile

WSLの既存候補（標準配置、root/user領域、home、pyenv/conda等）にはexact版が見つからなかった。
サーバー1の固定成果`9d022a0f7cc953ba3c1dac0b5e514e295c429fbe`に記録された実バイナリと同じhashの
runtimeを読み取りでexportし、Windows経由でWSL専用領域へ複製した。
archive・各ファイルのhashを確認し、検証済みローカルwheelからnumpy1.26.4、scipy1.14.1、mpmath1.3.0を
専用site-packagesへ導入した。OS機能追加・再起動・共有Pythonの更新・ソースbuildは行っていない。

- 専用領域：`/root/projects/Partially-Randomized-Trotter-local0-wsl/.local0-wsl/runtime-prdf-311rc1/e80a6533ce787e22`
- profileのinterpreter（shell launcher）：上記の`bin/python3`
- 実Python：上記の`runtime/bin/python3`、realpathは`runtime/bin/python3.11`
- 実version_info：`[3,11,0,"candidate",1]`
- 実バイナリSHA256：`5374e2d1d711fd9cc01fb15c48e3915718765b8200fb315f719d9d0017fa6331`
- 更新した運用profile：`.local0-wsl/PROFILE.current.json`。PR-DF routeのinterpreterだけを変更した。

Full PF・B0I・PR-Pauliの選択profileと現在fingerprintは前回と一致。
3方式の既存3.12.3、SOURCE、route、spool、DP cache設定は保持した。
今回3方式のdry-checkを再実行してはいない。前回のWSL実機PASSをhashと現在fingerprintで照合して再利用した。

## 照合した固定入力とguard

- 計画：`33928bf80cca18ce3f1fa3ddde88d27d7e2b7e5b`
- 計算SOURCE：`28d976b99cdf8329adcc1b79327e329e947ef774`。manifestの8実ファイルhashが一致。
- canonical request SHA256：`2906eeafee0f39909ef2db9c169924bcb74d48354995ab443e9231f94c0d2e2a`
- route：`23160cc876807c0300be3de811f767b5810640a7`、内側SOURCE：`27e70efe533862f38f4080aabb9cdd3ad1749fe9`
- portable adapter：`e37f6544dea3e021d3694264f7dae87add7ad30e`。前回と同じ固定root/manifestを使用。
- 元AST guard：`PASS_UNCHANGED_GUARD`、guard_modified=false。
- expected AST SHA256：`3d043e716246e57a8b05784b0d5c0f6777aaf43410fed8c4dd8b68df6dd590b8`
- Q3_INPUTS SHA256：`bad7f828ab46757c9ec724e189b038c47edec751b2ac6878041cbacfd742c063`
- Q3_CALL_INPUTS SHA256：`671d7f0d2e3286111355cdc0d33f7fae7e46da32107be0337a53b0991e5c8fc1`

実測は[HOST_METHOD_READINESS](evidence/HOST_METHOD_READINESS.json)、
[公式READINESS](evidence/dry/PR_DF/READINESS.json)、[ADAPTER.log](evidence/dry/PR_DF/ADAPTER.log)、
[SOURCE・入力・guard照合](evidence/CHECKS.json)、[runtime実測](evidence/preparation/RUNTIME.json)を参照。
公式READINESSの最上位PythonをadapterのPythonとして扱わない。
shell launcherとsys.executableの違いは、サーバー1で使った既存HOST_METHOD_READINESS形式で結合した。
共通コード・AST/SOURCE guard・割当APIの照合条件は変更していない。

## サーバー2への取り込み

従来のZIP返送・member hash照合・固定Git bundle方式を使う。
同梱`SERVER2_IMPORT.json`のfixed_git evidence descriptorには、`host_method_readiness`と
PR-DF専用`profile`を含める。これで実launcher/バイナリ/guardを照合できる。
Full PF・B0I・PR-Pauliは同じdescriptor内の前回実機PASSとfingerprintを使う。
新しい証拠で4方式PASSとなるが、今回freshに検査した方式はPR-DFだけである。

Windowsの参加操作・relay経路・optional host登録方法は従来と同じ。
local0-wslをrequired_hostsへ追加しない。今回の返送は本番登録や科学実行の許可ではない。
検証は作成者自身による実機確認で、独立再現や科学Q3完走の主張ではない。
