# C向け：固定3候補のQ3結果とSOURCE join

[REPORT](REPORT.md)／[SUMMARY](SUMMARY.json)／[C_RESULT_JOIN](C_RESULT_JOIN.json)。科学実行SOURCE **27e70efe533862f38f4080aabb9cdd3ad1749fe9**、計画/元入力索引 **0629a8cf359b3aa99400f79de8f61a5ef50d8162**。数値判定はPR-DF LD1/LD2がモデル適合、Full-H κ2がモデル不適合。運用は全3件WORKER_ARTIFACTS_SAVED、errors/monitor_errors0。数値判定と保存/monitorの運用statusは別に読む。このHANDOFF/結果の公開commitを結果carrierとして固定し、科学入力/callerのcommitとは区別する。

## 原記録

各`run01/<label>/API_RETURN.json`は既存`T.evaluate`が返した配列そのものを、返却直後にworker内で保存したもの。`RESULT.json`のrecordと一致する。coarse/lemma/DP・全round進捗、worker自身の`RESOURCE_FINAL.json`、保存完了`PERSISTED.json`、親のreceipt/資源観測を別に保存する。復元recordではない。UNKNOWNでAPI自体が返らなかった場合は、その不在とpartial/理由を記録する。

原APIと数値kernel710aa69b・accelerator9e294778・C c628e9e8は無変更。PR-DFのK.accept_saved→fixed-grid total bridgeと、Full-H κ2の保存baseline bridgeを使った。旧partial DP・Pauli/PFのMSEを今回の結果へ移植していない。

## ID・law・allocation

| label | 元configuration先頭 | total先頭 | signal正本 |
|---|---|---|---|
| PR_DF_LD1 | a4dda8f7 | 9c3d392e | A714247619 B_MEANS_LD1、raw16210932、数値fefd548e |
| PR_DF_LD2 | 1fa33556 | e6fe7e68 | A714247619 B_MEANS_LD2、raw16210932、数値fefd548e |
| FULL_H_KAPPA2 | 54ac3eb3 | 265b90dc | ce6aacfa FULL_H_K2_QDYADIC_KAPPA2、producer5ccc1dfc→5cc7c07e |

完全なID・packet/configuration locator・binding IDはINPUTSとC_RESULT_JOINを正本とする。各rowはm/axis、**元physical ID**、元canonical candidate、新total candidate、N、元/新law bindingのselectorを分ける。selectorはその候補のRESULT内のcanonical metadataを参照する。

配分は3候補両軸共通`[64,60,56,52,47,43,38,34,30,25,21,17,12]`。allocation IDはINPUTSの計画値を保持。旧controlの配分形状を再利用したUSER_SUPPLIED vectorであり、十分Nや方式固有のnominal Nではない。PR-DFの元grid・κ/R/signed allocation、Full-Hのone-body込みκ2/grouping、候補別normalization/scalarを維持した。R448は含めない。

## consumerの境界

評価は保存binary64 raw・solver proxy・ideal IID・Z1＋T1・全事象平方lossと既存engineering allowanceに条件付く。閾値は十進εEのexact二乗。直接riskへδEを再加算しない。モデル適合/不適合/INCONCLUSIVEと物理的ground/global RMSE保証を区別する。

`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`。新しいN資格producerを発行せず、原論文型nominal/G_paperとも分離する。今回の3件の結果を十分N packetとして受け付けない。

将来の価格joinは既存C ALIGNMENTから元configuration/m/axisを選び、元receiptのphysical/law/cohort/compiler/lowering/full wrapper/P0をguardで照合した後、既存source-bound decoder bridgeを使う。新canonicalを理由にreceiptを付け替えない。D1価格較正は本riskの前提ではないが、正式Gには適格な価格とN/誤差資格が別途必要。

次の候補/配分・scope変更・科学retry・価格/Gは別の明示許可が必要。ロードマップ影響はREPORT、統合担当が本体を更新する。公開・remote照合後mandatory STOP。
