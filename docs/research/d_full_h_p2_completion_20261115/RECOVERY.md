# 保存結果の残検査再開：監視起因の異常終了修正

利用者の「原因を調査して修正して再実行してよい」により、0dcのSTOPを保全して残検査を再開する。
対象は固定D7 CALIBRATIONのG36223 path0。18実compileは取得済みで、新sampling・compileは0。
保存native path0と完了path1はFROZEN_COMPLETIONと同じSOURCE・physical/cohort/index/seed・event・価格を使う。
新しい出力/private namespaceと固定入力は[FROZEN_RECOVERY.json](FROZEN_RECOVERY.json)。
0dcのSOURCE・STOP・STACKS、旧d323/887のraw/compile記録は変更しない。

## 原因調査と変更

0dcのworkerは2回目の定期stack dumpが `File` の途中で切れ、child receiptなしで終了した。
旧runnerは子return codeを保存していないため、**旧実H4 workerのsignalは不明**。
既存Pythonは3.11.0rc1。CPythonの公式報告
[116008](https://github.com/python/cpython/issues/116008)と
[140815](https://github.com/python/cpython/issues/140815)には、GILを取らずに変化中のframeを読む
faulthandler watchdogの競合・segmentation faultが報告されている。

同じ専用Python環境と既存FastWriterで、3-qubitの人工回路のみを反復fingerprintした。
2msのdump_traceback_laterを有効にしたcaseはSIGSEGV (-11)、無効caseは8秒・308反復完了。
dynamic-frameの有効/無効caseはいずれも完了。結果と実行script・raw traceは
[diagnostics/RESULTS.json](diagnostics/RESULTS.json)、[REPRODUCER.py](diagnostics/REPRODUCER.py)。
科学的sampling、分子input、実H4構築・compileは使用していない。
**観測機能の同環境でのクラッシュは再現済み。旧workerの原因もこれが有力だがsignal欠落により断定しない。**

worker内の非同期stack dumpを除去し、協調的な段階PROGRESSのみを残す。
親processのPopen/wait(timeout)でPID・段階・経過・/proc資源を記録し、実終了code/signalを保存する。
監視の遅延・読取エラーをworker停止条件にしない。計算workerへPython threadは追加しない。
元runnerにも子終了code/signalを記録する最小変更を行った。
native/provider/compiler/profile/数値gateは変更しない。sampler/compilerは再開workerで明示的に禁止する。

## 検証と実行前照合

既存11人工testを再利用し、親observerが実際に複数回発火して正常終了とSIGTERMを記録する検査、
再開入力差分・科学的FAILを拒否する検査を追加する。通常・checkpoint・完了path再利用を含む。
結果はRECOVERY_TESTS.txtへ保存する。これはD1本人の技術検査であり独立reviewやC科学受入ではない。

§10.5はdf0の調査と後続の全取得記録を引き継ぐ。変更影響は監視と子終了記録だけ。
18個の既存compileの価格・SOURCE・hash・axisの対応は既公表の保存記録を再利用する。
新出力/関連jobは0dcのSTOPとその部分結果公開までを確認し、未検査path0だけを再開対象とする。
完了path1はhashと全checksを照合してそのまま使う。P2両親event receiptとpaired依存は既公表のまま。
科学実行前にSOURCE/inputをcommitで固定し、専用checkoutで実行する。

全5 cell引渡し、または重大gate失敗でmandatory STOP。fit/凍結はC、HOLDOUT/外挿/E[C]/N/Gへ進まない。

実行前検証：13人工test PASS（10.235秒）。保存input/hash・STOP来歴の読取検査もPASS。
関連する科学jobの重複稼働0を確認。[RECOVERY_PREFLIGHT.json](RECOVERY_PREFLIGHT.json)。
追加testの初回では人工fixtureの既存fileへの排他writeで1件ERRORとなり、fixtureのみ修正して13件を再検査した。
実行状況：このSOURCEと入力を固定してから残検査を再開する。実H4残検査はまだ再開していない。

## 固定後の実行

SOURCE/input `8cce540645b99e515d72888b817782a1b9f97334` をcommit/pushし、remote SHA一致を確認した。
旧887/0dcのHEADを保持して専用detached checkout `d-full-h-p2-recovery-20261115` を作成した。
session21666で `--recover-approved-job FROZEN_RECOVERY.json` を実行中。追加sampling/compileは禁止。
進捗はrecovery01のconsole/time、private/recovery01/G36223のPROGRESS/OBSERVATIONS/CHILD_EXITへ保存する。
実行前の記載は履歴であり、最終gateのPASSはまだ未確認。

共通AGENTS.mdにも、親からの観測を優先し、このPython環境での非同期dumpを使わない指示を追記した。
Claudeは既存CLAUDE.mdの共通AGENTS importから同じ指示を読む。別AIによる検査は行っていない。
