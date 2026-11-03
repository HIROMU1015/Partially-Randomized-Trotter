# 2026-10-09：H4停止後のメモリ処理修正・v4準備

利用者の継続指示を、前回具体化した実装修正・合成検証・v4準備として進めた。
旧worktreeはsource集合全体をhashで固定しているため、別worktreeへ既存状態をbyte一致でコピーした。
旧sourceと24-wrapper partial pilotの正本を保持し、新v4 filesだけを追加した。

回路を1 cell-replicaずつ処理し、両control/axis完了後に解放する。
canonical JSON bytesを変えずstream fingerprintへ接続し、stage/RSS/VmSize/tracebackを追加した。
合成検証155件が通過した。弱参照による回路解放、nested numeric bytesの一致、
最後のq4で失敗する24-record shape、stdlib processで実際にaddress-space exhaustionを起こす検査を含む。
64 KiBはhash bufferでありtotal memoryではない。Qiskitのdefinition生成メモリが残ることも確認した。
toy memory比較を分子peakへ外挿しない。

science planのcell/seed/compiler/gate/capを維持し、S4 q4を外さなかった。
数値処理15 function/methodのASTとnative lowering bodyは旧実装と一致する。
153 science source hashesを新manifestへ固定し、旧v3 authorizationを新v4へ流用しない。

[v4実装・検証・実行前固定](../track_a_ax2b_h4_memory_preparation_v4.md)を最新入口とする。
未commitのlocal準備証拠であり、分子MemoryError解消・accuracy/shot/total cost/winnerの結果ではない。
今回の分子signal/sampling/compile、H4再実行、H6/H8/GPUは0。
`AX2B_H4_MEMORY_V4_PREPARED_SCIENCE_NOT_AUTHORIZED`、CPU未割当、launch=falseでSTOP。
次は同じH4 scopeの新しい一回実行を判断する段階で、旧partialのresumeや自動拡張をしない。
