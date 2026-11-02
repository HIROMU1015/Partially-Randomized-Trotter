# 2026-10-18 H4/H6 S4 count mismatchの技術調査

[報告](../track_a_h46_s4_count_mismatch_investigation_20261018.md)。既存7e454b91…から独立worktree/branchで実施。科学SOURCEは536a3671…、元実装・結果は変更していない。

負Yoshida係数とstage時間のbinary64差が、共有DF producerのexact-zero RZ選別へ作用する機構を人工3-modeで再現した。保存raw/RZ/CX差により正味RZ保持数差へ切り分けた。実block/qubit別の観測はなく、係数統一後の実総数一致をPASSとはしない。epsilon pruningで数を揃える変更は行わない。

16 tests PASS、Qiskit import0、新実科学0。実block別の最大66 primitive評価だけを未承認案にした。H4順位逆転の統計解析は並行担当に委ね、GPT/Claudeの科学レビューは開始しない。

次の担当：ユーザーが追加数値検証の要否を判断。公開後mandatory STOP。
