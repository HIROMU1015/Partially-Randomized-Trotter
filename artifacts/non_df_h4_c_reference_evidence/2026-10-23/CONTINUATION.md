# 引継ぎ：修正版SOURCEの固定40軸実行

SOURCE `e80d7510880f8db993b777c627bb8e95b05a070d`、科学runtimeとコードは隣接 `non-df-h4-c-reference-fix-20261023` に封印されたまま。本worktreeは実行結果用で、科学コードを編集していない。

今回のgrant `6862c863f376be24c4ab56bc2d21abd2f8c2560b80250f95d4a1e6fd8b0cb4b4` は既に一回消費済み。**launch・issue-grant・retry/resumeを再度行わない。** DFは計算を継続しているので、chatの切替・待機・一時的競合を理由に停止しない。

現在の読取観測（本worktree rootから）：

```bash
python3 -B artifacts/non_df_h4_c_reference_evidence/2026-10-23/launch01/observe_run.py
```

`launch01/supervisor_native/STOP.json` が保存された後に、Codexは `saved_result_audit.py` を修正用private Pythonで一回実行し、保存済み40軸・native wait/reap・保全SHAを整理する。既存の監査outputがあれば再実行せず読取照合する。監査で科学値・shot/Gは再計算しない。

結果・最終監査は未完了。完了後に結果表・報告書・index・manifest・local commit・mandatory STOPを記録する。正式certificate受理・詳細科学レビュー・最終freeze・N/G会計・pushへ自動移行しない。旧SOURCEのexecution hold、過去STOP/消費済みgrant、元H4 native未証明/監視原因UNKNOWNを保持する。


## 計算待ちによるchat終了時点

利用者「計算の待ち時間ならチャットを一旦終了してよい」。`launch01/CHAT_HANDOFF.json` と `observation15_chat_handoff.json` を確認する。科学計算は停止していない。保存済み不変fileだけをcommitし、active ledger/log/future結果は完了としてsealしていない。次回の結果回収・終了監査・報告・local commitは既存承認の範囲で継続し、中間承認を取り直さない。新しい科学stageへは進まず、終了監査完了後mandatory STOPする。
