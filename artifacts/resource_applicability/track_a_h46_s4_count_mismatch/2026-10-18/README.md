# H4/H6 S4：8 metric行の不一致原因調査

**TECHNICAL_INVESTIGATION_COMPLETED / mandatory STOP**。

[報告書](../../../../docs/research/track_a_h46_s4_count_mismatch_investigation_20261018.md)に、8行の一覧・発生段階・1 ULPとの関係・分類・不足証拠・最小追加検証案・科学的主張への影響をまとめた。

S4負時間のbinary64差が共有primitive producerの厳密zero判定を変える機構を人工で再現。保存raw/RZ/CX差から、保持RZリストの正味差はH4 +1組、H6 −8組と特定した。実block/qubit別の発生箇所と係数統一後の実系一致は未検証。元8行・元UNATTRIBUTED監査は保持。

- [8行の保存値](eight_metric_rows.json)
- [静的解析・人工fixture・全84行provenance照合](static_and_artificial_audit.json)
- [分類・確認済み／未確認](classification.json)
- [固定source closure](source_closure.json)
- [16テストの結果](test_results.txt)
- [未承認の最大66 primitive評価案](minimal_validation_plan_UNAPPROVED.json)
- [ファイルSHA](manifest.json)

調査sourceは、この資料を含むcommit内の `scripts/audit_h46_s4_count_mismatch.py` と `tests/test_h46_s4_count_mismatch.py`。科学SOURCEは `536a3671c0ec2939ff3886e40467db26a27ec65a` のまま。

## 再現方法（人工・metadataのみ）

元base/provider/SOURCEのGit objectとNumPyがある既存Pythonで、repository直下から実行する。install不要。出力はhome配下の未使用fileを指定する。

```bash
python -P -B tests/test_h46_s4_count_mismatch.py
python -P -B scripts/audit_h46_s4_count_mismatch.py --output "$HOME/h46-s4-static-audit-NEW.json"
```

Qiskit、科学array、実H4/H6 primitive生成、compile、理論費用再計算、samplingは使わない。元grant/ledger/marker/STOPを変更しない。人工のidentity basisはmockであり、実native回路検証とは扱わない。

公開branch：`track-a-h46-s4-count-mismatch-investigation-20261018`。最終commit/remote一致はチャット報告の40文字SHAを参照。

次の担当：ユーザーが実block別の最大66 primitive評価を行うか判断する。Codexは明示承認まで追加科学計算を行わない。
