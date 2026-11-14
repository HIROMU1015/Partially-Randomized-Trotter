# C/A向け：PF6 Q3保存componentの受入

[REPORT](REPORT.md)／[SUMMARY](SUMMARY.json)／[C_RESULT_JOIN](C_RESULT_JOIN.json)。全6 workerの数値評価は各1回完了・保存DP端点は閾値適合。ただしdriverの監視TypeErrorで原API RESULT保存・CPU/RAM記録が欠測。**原結果の代わりに保存componentから復元した索引であり、完全なAPI returnや十分N packetではない。**

## 正本とSOURCE

- 実際の科学SOURCE：**313ed1950d3e02cbb0c392bcd147bc40aafcf176**、`INPUTS.json`と`run01/EXECUTION.json`。
- 数値原記録：各`run01/<label>/PARTIAL_TOTAL_DP.json`、`PARTIAL_LEMMA.json`、`PARTIAL_COARSE.json`、`STAGES.jsonl`、`ACCEL_PROGRESS_job.jsonl`。STAGESの科学call開始/完了は各1回。
- 復元view：`run01/<label>/RECOVERED_RESULT.json`。`original_api_return_available=false`と運用欠測を保持。`model_bound`は保存DPの端点・momentsをそのまま転記し、DPが全保存有効上界のminであることをexact比較したもの。既存Q3 helperの閾値判定は元完了stageと一致。新しいrisk/DP/binomial/MC/G callは0。
- A signal：carrier06a15ee45beb177c018f6260ea4a5ac8cf66cfe6、統合producerfb9a1878b677b7ee116b52e804e5a01cdaa96004、REPORT1adb7e65a9e7c3adc32491d853f31d124393ead7。元producerの由来はA packetのprovider_bindingに保持。古い段階packetを使わない。
- T.create/bind_signal/evaluate、kernel710aa69b・accelerator9e294778・C c628e9e8は元SOURCEのまま。現callerの監視修正を科学実行SOURCEとして遡及させない。

## Cのjoin手順

1. C_RESULT_JOINの`original_configuration_id`で既存C0b4b2d94 ALIGNMENTの候補を選ぶ。Full/p7や公式名だけで選ばない。
2. 候補の`result_path`はこのdirectory内の復元結果を指す。結果公開commitをcarrierとして固定し、科学SOURCE313ed1950（入力/caller）と区別する。
3. 各rowのm/axis/physical_id/original_candidate_id/total_candidate_idを区別。`original_law_selector`・`total_law_selector`は、その`result_path`内のcanonical metadataのSOURCE付きmappingを指す。
4. 配分は全候補両軸`[59,55,51,47,43,39,35,31,27,23,19,15,11]`、allocation IDはlabel＋`_DET_PAPER_SHAPED_CONTROL_01`。raw直接MSEにδE/δE²を二重加算しない。
5. 将来の価格は元physical identity・compiler/lowering/full measured wrapper/P0/cohort/law/SOURCEを元receiptで照合し、既存source-bound decoder bridgeとguardを使う。価格やreceiptを新canonicalへ単純に付け替えない。今回価格値の受入・算術は0。

## 判定と未資格

モデル内の評価済みcontrolが6件閾値適合であることと、運用UNKNOWN（driver保存欠測）を併記する。十分N・最適N・全RPEの物理保証・数値certificate・価格/G資格を発行していない。原論文型nominal/G_paperとも別の結果。`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`。

次のshot探索が必要かは、今回のbias/Varと適格な価格domainを材料に別途有限案を作る。現在の6件だけでは最小Nも資源winnerも決まらない。監視記録の欠測を埋めるだけの科学retryは許可されておらず、今回の固定6件を再実行しない。追加vector・他方式risk・E[C]/Gは別の明示許可が必要。

ロードマップ本体は編集せず、更新点はREPORT末尾に記載した。公開・remote照合後**mandatory STOP**。
