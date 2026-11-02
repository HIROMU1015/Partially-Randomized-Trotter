# H2共通native：DIAGNOSTICS実行binding準備

DIAGNOSTICSの実SOURCE/input/runtimeを封印し、科学importなしのpreflightがPASSした。実科学実行・政策正式採択・詳細科学レビューは未実施。専用branch `codex/h2-diagnostics-binding-20261023`、runner SOURCE `35444654f9159a5d85debdbc577feab0590e2610`。科学結果・順位・費用評価の追加ではない。

linear H2/隣接1 Å/STO-3G/tol-only DF1e-8/rank3、T=.8、ε_sig=.0005、α_axis=.025、P=0、指定Full S2/q4・Full Y4/q2・PR S2/q4・PR Y4/q2だけ。bootstrap1/setup1/prepare2/diagnose4/path_signal8/reference_rebind4、他kind0を変更していない。実行には政策採択・契約digestへの明示承認・新grantを要する。

- [完了報告](../../artifacts/resource_applicability/track_a_h2_diagnostics_binding/2026-10-23/REPORT.md)
- [3政策の採択資料](../../artifacts/resource_applicability/track_a_h2_diagnostics_binding/2026-10-23/POLICY_DECISIONS.md)
- [SOURCE/runtime/input/import封印](../../artifacts/resource_applicability/track_a_h2_diagnostics_binding/2026-10-23/RUNTIME_INPUT_SEAL.md)
- [科学実行契約・受入証拠](../../artifacts/resource_applicability/track_a_h2_diagnostics_binding/2026-10-23/DIAGNOSTICS_EXECUTION_CONTRACT.md)
- [PR C identity監査](../../artifacts/resource_applicability/track_a_h2_diagnostics_binding/2026-10-23/PR_C_IDENTITY.md)
- [独立診断後のCOMPARE計画](../../artifacts/resource_applicability/track_a_h2_diagnostics_binding/2026-10-23/COMPARE_CONTINUATION.md)

旧72件を含む人工98件PASS。静的許可集合は科学importの動的attestationとは異なり、集合外loader/native mappingを拒否する。PR67 tasksのfull event/operation/runtime witnessは不足し、COMPAREはBLOCKED。独立診断を再利用するmetadata interfaceだけでは既存COMPAREの診断callを省略できず、新SOURCE/契約が必要。

今回の変更は技術準備であり、研究上の共通政策をCodexが正式採択したものではない。旧結果・STOP・grantを保持し、local commit後mandatory STOP。push/PR/mergeなし。次はユーザーによる政策・限定DIAGNOSTICSの判断、その明示承認後にCodexが別作業として科学実行する。
