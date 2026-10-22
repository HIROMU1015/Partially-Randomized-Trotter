# 独立DIAGNOSTICS後のCOMPARE_REUSE_PR

**比較分岐はBLOCKED。今回のinterfaceはmetadata/hashの受入前検査であり、比較科学実行のloaderではない。** PR Cの完全identityと、政策・科学範囲の別途承認が必要。

既存 `execute` は全てのCOMPARE_REUSE_PRでsetup1/prepare2に続きdiagnose4/path_signal8/reference_rebind4を実行する。別run_rootの完了DIAGNOSTICSを引数で受け取る分岐はなかった。古いgrantを渡す、budgetのdiagnoseだけ0へ書き換える、古いcompletionを未検証で読む方法は成立しない。今回も既存countsは変更していない。

## 実装した技術interface

`h2_diagnostics_binding.accepted_diagnostics_receipt(root, expected_contract, completion_sha256=...)` は科学importもNPZ decodeもせず、以下を検査する。

- 外部で封印されたcompletionの全byte SHA、DIAGNOSTICS_ONLYの正常完了、STOPなし。
- 元契約digest、SOURCE、runtime、H/state/snapshot、物理条件、3政策、exact used予算。
- 消費済みmarkerのSHAと元grant/root/budget/contract対応。元ledgerの連番、nonrefundable、正整数amount、kind、contract、合計予算。
- prefix1/3 prepと全4候補のdiagnostic/vectorが揃い、全結果hash・candidate/native policy・strict1e-9・scalar一回・phase fitなしが一致すること。

返り値はACCEPTED_METADATA_ONLY、prior_grant_reusable=false、science_GO=false。厳密数値correctnessの再評価を行わず、過去のgateを自動でrigorous証明へ格上げしない。future consumerの新契約が元completion/marker/ledger/prep/diagnostics/NPZ全体を使用入力として封印する必要がある。

診断出力には今後の実行からvector archiveとresult_files indexを保存する。以前の出力形式だけでは独立vectorが不足し得るため、古いJSONを新形式と偽って読み込まない。

## 二つの後続案

| 案 | 追加科学call | SOURCE/grant/root |
|---|---|---|
| 現行COMPARE_REUSE_PRをそのまま使う | bootstrap1/setup1/prepare2 **に加え診断4/path8/ref4を再実行**。Full native2/compose2/lowering10/wrapper4/compile4/transpile4/simulate12、allowance4/accounting4/ranking1 | 現行SOURCEを固定した別契約、新grant、新root。診断重複を承認対象として明記 |
| 将来COMPARE_FROM_ACCEPTED_DIAGNOSTICSを別に実装 | bootstrap1/setup1/prepare2で同じprepを再構成・hash照合。診断4/path8/ref4は0、受入bundle load1を新しい明示単位として設計。Full native2/compose2/lowering10/wrapper4/compile4/transpile4/simulate12、allowance4/accounting4/ranking1 | **未実装・未承認**。新SOURCE/new branch contract/new grant/new root。元SOURCE/結果は不変の入力として封印 |

両案ともsampling/request_factory/rte_event/occupation_trajectory0、新reference0、quantum shots0、q探索0。compile/transpile等は内包関係であり単純合計を独立実験数としない。新interfaceを使うだけで現行branchのbudgetが減るわけではない。将来load分岐ではNPZ schema/basis/qindices/shape、stored AdaptiveInputsの復元、diag vectorと新prepのidentity、許可外の計算をしないことを人工テストし、数値再評価が必要ならそのcallを別途列挙する。

最小案としては、受入済みDIAGNOSTICSが得られた後に後者の専用分岐を完成させ、重複path8を回避することを推奨する。現在は設計とmetadata interfaceまでで停止する。元DIAGNOSTICS grantを継ぎ足したり再使用したりしない。

## PR Cが不足する場合

[PR_C_IDENTITY.md](PR_C_IDENTITY.md)のevent/operation/runtime/correctness witnessがない限りCOMPAREは開始しない。保存67 paired taskを再sampling/recompileして補う自動処理はない。

原eventのmetadataが回収できた後、新しいoperation streamがどうしても必要なら、prefix1 preparation1・Gaussian operation記録を含む代表準備検査の単位を別契約に固定する。実event request生成を行う場合はsamplingとして、候補S2は1 trajectory当たり4 requests/128 events、Y4は6 requests/64 eventsを明示する。compileが必要な影響軸だけを事前に確定し、axisごとwrapper1/compile1/transpile1と必要correctness simulationを列挙する。影響軸が未確定の今はlive budgetを作らない。

任意PR_TWO_NATIVE_CONTROLSは別分岐で10 requests/192 events・4 compile・12 simulation等を含む。代表2 trajectoryの検査で67 task全cohortの単価identityを証明しない。今回は実行せず、DIAGNOSTICSの条件にも加えない。

次の担当：ユーザーがDIAGNOSTICSの政策と限定範囲を判断。承認後のCodexが診断だけを実行してSTOP。正常な受入bundleとPR C witnessが揃ってから、Codexが後続SOURCE/契約の完成、ユーザーが別の比較実行承認を行う。科学レビューの開始は別指示を要する。
