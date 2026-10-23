# 引継ぎ：reference修正後は技術STOP

修正SOURCE `e80d7510880f8db993b777c627bb8e95b05a070d`。原実行SOURCE49c18ccc3・失敗C run01・消費済みgrantと `FAILED_NO_RETRY` / execution holdは保持する。本directoryに有効な科学grantはない。

現在許可されている作業は修正・人工検証・技術seal・保存・local commitまで完了。**次の科学実行は別の明示承認が必要**。新契約は全実行flag false、grant/approval null。ここで自動有効化しない。

読み取り専用の確認例（このworktreeのrootから）：

```bash
.c-reference-fix-venv/bin/python -B scripts/run_h4_numerical_evidence.py \
  --inspect artifacts/non_df_h4_c_reference_fix/2026-10-23/C_execution_contract.json
```

人工テストの再現は必ず未使用の技術output directoryを指定する。既存ログへ上書きしない。実H4入力を人工testへ渡さない。

将来の担当は次のとおり。

1. **ユーザー**：報告書にある固定40軸の数値証拠候補生成・残る受理制約を確認し、新科学実行の範囲を明示承認する。push/N/G再会計は別判断。
2. **Codex**：承認後、source/runtime/input SHAと契約を検査し、新しいapprovalとactive契約を別fileに作る。旧inactive契約・SOURCEを編集しない。
3. **Codex**：新しい未使用namespaceに新grantを一回だけ発行・消費し、契約内で実行・native回収・証拠保存・STOPを行う。元runのretry/resumeや消費済みgrantの再利用はしない。

N/G、DF event範囲、compiled wrapper転送、正式certificate受理、元H4監視の欠落を別の証拠問題として扱う。本技術修正はそれらの解決証明ではない。mandatory STOP。
