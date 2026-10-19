# 人工受入・差分回帰

測定SOURCE `a1c4f41b6c29baf691006a8ef39e87b8ab9aa8d8`、test PASS `{'connection': 17, 'preflight': 23, 'regression': 66, 'store': 16, 'unit': 10}`、mutation20/20拒否、integration10/10期待分類。
[受入matrix](artificial_acceptance_matrix.json)、[raw](acceptance/raw_artificial.tar.gz)、[tests](acceptance/tests.json)、[静的監査](acceptance/static_preparation_audit.json)。
正常・coverage不足・S/P1 angle故障・人工invariant不一致・receipt close・bytes/call cap・worker死・ACK unknownを区別する。
正常は512全EvidenceとCOMPLETED、coverage不足はINCONCLUSIVE_COVERAGE、故障はTECHNICAL_STOPまたはINCOMPLETE_EVIDENCE。
20semantic mutationはglobal ID/source/runtime/session/pid/trial/event/digest/sidecar/seed/width/counter/angle/query/auditとresealed originalを検査。
同一作者による検査であり別reviewerの独立受入ではない。Exact Codec/Result validatorを共有する。

旧回帰SOURCE `2405f5e2e5fb0dc5add39b102e3ccae3d56df0a4` と新測定SOURCEで旧124pin bytesが等しく、F01 pair ASTも同じ。
旧F01/Storageの必要66 checksを新SOURCEで実行した。旧16,896/独立224/Producer等の件数を新結果へ合算していない。
人工symbolsはAPI形状fixtureでありP5科学lawやatanを計算しない。held-state文書の人工試験も数学検証ではない。
開発時の記録処理・connection close・期待reason修正はdevelopment_history.jsonへ保存。最終判定はfrozen SOURCEだけに基づく。


**実科学未実行、実行未承認。mandatory STOP。次の担当：ユーザー・GPT。**
