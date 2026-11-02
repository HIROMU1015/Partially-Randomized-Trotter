# GPT・ユーザーへのC1-1 handoff

**C1-1は`REANALYSIS_INCONCLUSIVE`。保存入力の同一性のみ確認でき、basis成果物・G6 exact modelの取得は未完了。1回実行、retry 0、SOURCE修正0。公開確認後mandatory STOP。**

- 固定参照：v1.1 `243eea8e840f3c472753f7935e510391626fd325`、C1-0 `a48b3c8244fb5d932e0d6bf4a5de6bb573ce35ed`。
- 実行SOURCE `d5f962b8b35f3912d45801f4d18998aa309feb0c`、新authorization `cb5dc85ca0386579dd79649ea0e3c72a751c7e6c`。
- branch `track-b-c1-1-input-basis-exact-model-20261020`。最終HEAD/公開照合記録はpublication receiptおよびチャット報告を参照。
- 指定H4 1.00 Å / STO-3G / DF12 / 8 qubit / prefix3 snapshot、全保存digest、state支持・norm、Hermiticity、fragment index列は`VERIFIED`。厳密ground stateやstate物理準備は認定していない。
- 既存basis preparationのreturn後、partition hashが保存B2-G値と異なるため停止。実値とpayloadを新runnerが保存しておらず、原因は未確定。basis操作列/registryも未保存。byte-exact再現や数値operator一致の認定なし。
- exact係数/λ/p、component比較、モデル差、short-stepは未到達。`MODEL_BINDING_UNCLOSED`を維持。保存float λや旧残差で今回の未検証項目を埋めていない。
- 二次不備：最終stdoutで`stage`引数が重複。構造化failure/terminalと原ログは保存済み。修正・再実行なし。
- Gate 1は入力側だけ閉じ、全体は`CONDITIONAL`。C1-0の誤差/confidence条件とadapter `IMPLEMENTATION_REQUIRED`は変わらない。

## 次の担当と必要な承認（提案、未実行）

| 担当 | 次の作業 | 必要な判断・承認 |
|---|---|---|
| ユーザー・GPT | 今回の失敗記録を読み、partition identityの限定診断を継続する価値があるか決める | 診断の必要性・対象・合格基準。研究方向や科学GOはCodexが採択しない |
| Codex（別指示がある場合） | 新namespace/SOURCEで、比較前にexpected/actual hashと全payloadを保存する診断・終端logger修正を準備する | 旧SOURCE/契約/失敗記録を保持する新しい準備指示。現packetの上書きはしない |
| ユーザー | 固定入力のbasis再構成等を再度実行するか承認する | 新SOURCE/input/environment/検査範囲を固定した別one-shot実行承認。今回のreceiptは再利用不可 |
| Codex（その承認後のみ） | 限定診断と、承認された残りのbasis/exact入力/モデル差検査を実施して保全する | 新しい実行契約・authorization・STOP。今は未承認・未実行 |

adapter実装・新証明・native費用・sampling・追加科学計算はそれぞれ別の明示承認が必要。失敗はG6の原著価値の判定材料を完成させておらず、研究価値や論文GOの結論を代行しない。

[README](README.md)、[failure assessment](failure_assessment.json)、[failure原記録](result/failure.json)、[stderr](result/raw_stderr.log)、[入力同一性](result/input_identity.json)、[保全検証](preservation_verification.json)がレビューの最小セット。非公開レビュー原本は未公開。**mandatory STOP。**
