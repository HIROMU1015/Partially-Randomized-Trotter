# C production引継ぎ：科学実行は未承認

worktree `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-c-production-20261023`、branch `codex/non-df-h4-c-production-20261023`。基点b0f94185、旧科学SOURCE0c329b30。実装SOURCEはSOURCE.jsonのSOURCE欄、後続監査commitはこのbranchのgit logで確認する。

## 完了したものと残るもの

full-space Pauli/DF assembler、旧runtime/hash/preparation対応、Arb/Acb reference・S2・degree5有限法則・bias/normalization・CDF/53bit mapping、one-shot/予算/独立witness/native exit/reap/STOPを実装し、人工・回帰を検証した。実H4についてはschema/hash確認だけで、数値証拠・certificate生成0。

数値設計目標の達成と、科学的certificateの正式受理を分ける。特にDF encoded basisのexact event rangeとliteral compiled wrapperへの転送、IID統計仮定は自動で解決済みにしない。全出力はformal_certificate_accepted=false、旧N/G=null/UNDETERMINEDを保持する。元H4 native欠落/消失原因UNKNOWNも未解決。

## 次の担当・判断

**ユーザー**：SOURCE・実行契約・技術報告の保証範囲を確認し、実H4の数値証拠候補生成を承認する対象と予算を明示する。今回の実装承認は実行承認ではない。N/G再会計、公開push、詳細科学レビューは別承認。

**Codex（承認後）**：保存SOURCE/runtime/inputのfresh SHAを確認し、承認原文を保存、新しいactive contractを別ファイルへ作り、SOURCE seal SHA・approval SHA・新しい未使用namespaceを結合する。旧C契約・SOURCE、元H4/Bのgrant/claim/STOPは書き換えない。

## 承認後だけのgrant・起動手順

現在のC_execution_contract.jsonは全実行flag false、approvalとgrantは未発行である。inspectは読み取り専用で使用できる。

```bash
.c-production-venv/bin/python -B scripts/run_h4_numerical_evidence.py \
  --inspect artifacts/non_df_h4_c_production/2026-10-23/C_execution_contract.json
```

実行承認後のactive contractでは、execution_authorized/grant_valid/dispatch_enabledをtrueにした新versionと承認原文のSHAを結合する。SOURCEが封印した入力・旧root/runtime・method map・結果の参照先、40軸、precision schedule、予算は変更しない。新しいnamespaceを契約へ記載する。

```text
.c-production-venv/bin/python -B scripts/run_h4_numerical_evidence.py \
  --issue-grant <承認後のactive契約JSON> \
  --seal artifacts/non_df_h4_c_production/2026-10-23/SOURCE.json \
  --namespace <その契約に記載した未使用namespace>

.c-production-venv/bin/python -B scripts/run_h4_numerical_evidence.py \
  --execute <同じnamespace>
```

これらは**未実行の将来手順**であり、現在の無効契約では拒否される。issue-grantは既存namespaceを拒否し、executeは消費前に全hashを検査してCONSUMEDを排他的に保存する。科学callは予約先行・非返却。例外後のretry/resume/予算拡張は不可。precisionは幅不足だけで64→96→128、最後でも不足ならUNDETERMINEDとしてSTOPする。

次回最大予算：operator復元2、probability mapping2、reference6、nominal mean12、実装PR mean6、PR normalization6、dense exp39。SCF/FCI/新DF/状態最適化/sampling/compile/shot/Gは0。2表現を独立processにし、実際に空いている資源を使う。共有設定や他jobは変更しない。

native証拠は各representation_nativeとbridge_nativeへ終了直後に保存する。主親が消失した場合もwitness側のexit/reap/STOPを読み取りで回収する。witnessのwait記録まで欠けていればUNKNOWNを保持し、新しいSTOPで代替しない。

## 現在の停止

有効C科学grantなし、実H4科学call0、pushなし。SOURCE・結果・技術終了監査のlocal commit後mandatory STOP。今回ここから新grant発行・実H4数値評価・certificate・N/G再会計へ進まない。
