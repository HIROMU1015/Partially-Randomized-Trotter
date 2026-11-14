# A/B/Dへのconsumer側HANDOFF

[REPORT](REPORT.md)／[ALIGNMENT.json](ALIGNMENT.json)／[CSV](ALIGNMENT.csv)。これは価格・N producerではなく固定証拠の索引。科学実行・比較候補・policyを採択していない。

## 読取手順

1. candidatesの**configuration_id**でscheduleを選ぶ。同じroute名・prefixだけで選ばない。
2. rowsの`configuration_id,m,axis`で取得し、physical_candidate、元canonical ID、SOURCEを確認する。価格はprice配列でcohortとroleを識別する。価格値は各sourceの元recordから既存providerで読む。本表に新しい価格値はない。
3. decoderの元→totalのIDを保持する。B `rpe_total_binding.create/bind_signal/bind_prices`を使い、元record/sidecarを書き換えない。接続は既存guardとcost側のpolicy検査を引き続き通す。
4. MSE.sourceとN.vector_source/usedを別々に読む。十分Nは全行null。signalのS1/S2/S2L/S3と価格tierを独立に判断する。
5. price_planは**将来の役割**。NEW_COMPILEを保存価格と解釈しない。E_C/G=null、unknown CI/covarianceは元SOURCEのまま。価格算術は本builderにはない。

再生成（metadataのみ）:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts python -B scripts/build_h4_candidate_alignment.py
```

固定Git objectを読むため、選択的取込み時もALIGNMENT.sourcesのcommitを参照可能にする。branch全体のmergeは不要。持ち込む対象はbuilder・test・この成果物directory。既存M5/p7 reader、C guard、baseline/optimization metadata APIを依存として使う。新scientific runnerを呼ぶ経路はない。

## A：信号

- 最新公開`c212cee85`の364行とSOURCE監査をconsumerへ取り込んだ。Aの監査・再解析を再実行していない。
- Full PF/p7 shortlistの現行mean（S2）は、保存reference・normalization B1・元formula/scalar/actual幅を束縛してpacket化する技術作業が先。S2Lは旧run/診断条件を保全し受入方針をユーザーへ戻す。
- S3の最小mean範囲はAのREPORT §7を参照する。新signal実行は今回未承認。PR-DF/Pauli/κ2の既存13 rawを再計算しない。
- p7 S2/.1のm0はp7 S2 shortlist m0と同物理点だが、configuration/decoder/Nの全体を転用しない。

## B：decoder・MSE・N

- 元/totalの旧対応はB `999304918`のまま保持し、最新公開 **`9c0e12d87`** のPR-DF fixed-grid版付きdecoder-only bridgeを反映した。LD1 total `9c3d392e…`、LD2 `e6fe7e68…`。旧grid/law/physical ID不変をCANONICAL_MAPPINGで照合済み。Pauli・p7 S2/.1・Full-H κ2の既存metadata接続も保持し、R448の独立受入を追加した。
- PR-DFの保存partial DPとPauli total Q3の条件付きmodel上界は別SOURCE。使用済みUSER_SUPPLIED whole-vectorは同形でも、十分N・新decoderの評価済みvectorへ自動移送しない。
- 全round Nが将来適格になっても、本対応表の有限cohort、構造E[S]、未検証モデルを正式E[C]として採用しない。
- PR-DFの`rpe_total_binding.create/bind_signal/bind_prices`は新版BのAPIを使う。Cの古いcontrollerを修正して代用しない。本成果は新版の固定metadataを読むだけでB全branchを取り込んでいない。A監査の旧S1B/NO_INTAKEという文字列より、各行decoder/current_B_readinessを優先する。価格のscalar/source受入は引き続き未完了。
- Bの新公開結果は、元configuration/bridge/whole allocation/消費signal/sourceを一致させた上で、MSE/Nのlocatorを差し込める。private checkpointを読まない。

## D：価格

- D `acea1118a`のADMISSION_UPDATEを現行として照合済み。CELL_PLAN V2は36軸＝保存4＋新cal20＋holdout12。旧「p7受入待ち」は解消済み。新規32compile、fit/凍結/holdout、2%観測値基準policyは未承認・未実行。
- 受入済みM5/m4 X/YはFull/p7各2record。原EXPLORATION・trial・receipt hash/fingerprint・sourceを保持して再利用し、再compileしない。
- p7 S2/.1 m3/m4のlegacy locatorを発見済みだが、現行full measured wrapperの適格receiptと認定していない。原表の`kind/origin/reuse_provenance/result_sha256`のみを索引した。正式に使う場合は元receipt・軸・scalar・control・compiler・wrapperを既存guardへ束縛する。今回その追加受入はしていない。
- PR-DF P1/P2は別cohort。canonical/law/total scalarのconsumer受入が先。旧tail-only requestをtotal scalarへ単に付け替えない。
- Pauli B0・Full-H P-1はsingle-path wiring。P-2/B1の較正・holdoutへ自動流用しない。Aの非configuration校正セルは既存price_model_validation record、実round holdoutは元B価格recordとsidecarへ渡す設計を維持する。

価格モデル、CI/rare/law差/共有誤差/長q外挿の採択は未決。新計算の最小案は[NEXT_VALIDATION](NEXT_VALIDATION.md)。**mandatory STOP**。
