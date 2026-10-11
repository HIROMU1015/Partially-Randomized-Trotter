# N1/N3再設計への提供入力

[科学レビュー全文](n123_feasibility_scientific_review_2026-10-11.md)を利用者のroot fileからbyte同一で保存した。
SHA-256: `4b2d08be93b22aa75fafc68f2e6a1642cd2f38c8ee00772a968f01fa0be23a2a`。
レビューは完了済みであり、同じ結果の再レビューや付録の再実行をしていない。
付録scriptとJSONは原文に含まれる既存GPT作業で、新しいCodex計算の成果ではない。

現在の採用scopeは[限定設計v1](../n13_redesign_limited_scope_v1_20261011.md)、
[machine contract](../n13_redesign_limited_scope_v1_20261011.json)。科学実行は未認可。
旧result/IR/sourceは基点ae743445で取得確認済みという完了レビューの状態を保持する。
今回の受領で旧科学結果を再分類・再compile・再対角化しない。

[N1候補だけの入力](../../../artifacts/n13_redesign/2026-10-11/preparation/n1_candidate_input.json)は
旧44候補から係数・frame・spectrumだけをcopyした。many-body真値/誤差diagnosticsは含まない。
source/inputのSHAはmachine contractに記録し、将来selectorが旧full resultを開かない境界を作った。
