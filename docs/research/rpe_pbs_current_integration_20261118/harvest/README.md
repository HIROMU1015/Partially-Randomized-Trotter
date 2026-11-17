# 保存exact OUTCOMEの通常転送・終了後harvest

新engine SOURCE `95a74e4486131a97bd5235221bf2d8ab9f99f591` のgatewayを、
既存専用Python `/home/AbeHiromu/projects/rpe-pbs-20261118/runtime/venv311rc1/bin/python` で実行した。
[EXACT_TRANSFER.json](EXACT_TRANSFER.json) は通常 `/v1/complete` メッセージのexchangeと、
接続を閉じて開き直した後の終了後harvestでPASS。
OUTCOME全35,529 byte、decoded dict、長整数4項目と、転送対象6ファイルのhashが元receiptまで一致する。
float化・丸め・整数切詰めは行っていない。

| 保存field | 整数桁数 |
|---|---:|
| MSE_lower.numerator | 7168 |
| MSE_lower.denominator | 7174 |
| MSE_upper.numerator | 7741 |
| MSE_upper.denominator | 7747 |

元の科学実行は、SOURCE `925985b4ea76a15f8a07c507179d4fa6c492eabd` の
Full PF/M5・1件である。[元の固定記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/840525ed6a2dc3e018dd568898b9bc9e06349578/docs/research/rpe_pbs_h4_q3_portability_20261118/HANDOFF.md)
と [36ファイルの元回収receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/840525ed6a2dc3e018dd568898b9bc9e06349578/docs/research/rpe_pbs_h4_q3_portability_20261118/results/AFTER_EXIT_HARVEST.json)
を保全し、その実H4検収を新SOURCEへ付け替えない。

元jobは `98693b829295fc884330`、attemptは `18c2c7bc2d7b4a5581ef2543b1359e1b`。
検査は元runを起動せず、完了済みreceiptとOUTCOMEをコピーした専用metadata spoolで行った。
旧controllerの起動・移植検収callerの再実行・実科学workerの実行・中央queueへの採用はない。

専用コピー先は
`/home/AbeHiromu/projects/rpe-pbs-20261118/metadata-checks/95a74e448-exact-transfer`。
通常の人工PBS検収runとも分離している。
既存の `validate_completed()` は保存済みcollect receiptを確認するため、adapterのcollect/execute subprocessを起動しない。
既存の `result_bundle()` が作る元bytesのcompletion messageを、同じgatewayのexchangeへ通した。
その後、同じ専用Pythonで新しいSSH clientを開き、harvestのbytes・exact整数・元attemptを照合した。
このmetadata spoolにはQSUB_STARTEDがなく、PBS計算資源の新規予約は0件。

最初の固定95 checkerは「最大桁数==7168」と仮定して、コピー後・転送前に停止した。
7168は旧system Pythonが最初に拒否したfieldの長さで、最大は7747だった。
engine/gateway/worker/SOURCE95は変更せず、[metadata verifier](check_saved_exact_transfer.py) で
全4項目の実際の長さ・digestと元OUTCOME全bytesの一致を確認した。同じコピーを使い、再コピーもしていない。
これは検査用metadataの仮定の訂正であり、科学guardや科学条件の緩和ではない。
初回checkerの停止ログは [CHECKER_ASSUMPTION.log](CHECKER_ASSUMPTION.log)。
corrected verifierのpath/hashはEXACT_TRANSFERに固定し、engine SOURCE95とは別に記録した。
このverifierは最初の未応答completion messageを照合した一回の技術検査callerであり、通常のStart/Resume入口ではない。
再確認では保存receiptと既存metadata spoolのharvestを参照し、元科学runを再起動しない。

[終了後の保全確認](ORIGINAL_PRESERVATION.json) は、元9ファイルとコピー9ファイルの一致、
元840の6ファイルreceiptとの一致、metadata spoolのqsub marker0、専用Python実行先・full versionを記録する。

担当Codex sub-agent `pbs_exact_harvest` による技術検査。
新規実H4信号・実compile・実Q3・fit・本番探索・queue採用・PBS投入は各0件。
科学性能・他方式の実H4適格性・新SOURCEの実科学検収を意味しない。mandatory STOPを維持する。
