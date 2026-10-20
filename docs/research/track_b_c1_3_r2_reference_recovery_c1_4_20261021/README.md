# Track B C1-3-R2〜C1-4：限定correctness PASS・契約草案完了

**Phase A/B PASS、Phase C `C1_3_CORRECTNESS_PASS`、Phase D（C1-4）契約草案完了。公開・保全確認後mandatory STOP。** C1-5の実行GO、native取得、資源優位、新規性、投稿GOは認定していない。

- [Reference修正・原因](REFERENCE_REPAIR.md)：旧`real+1j*imag`でzero符号bitが変わる。固定R1全412局所行列の新復元はbytes/hash一致、旧式59件の差も保存。
- [H4 correctness報告](CORRECTNESS_REPORT.md)：新one-shot 1回・retry 0、固定6 eventすべて既存数値Gate PASS。最大controlled Frobenius残差2.868147051928566e-14、X/Y最大差1.3322676295501878e-15（診断）。
- [誤差・confidence草案](C1_4_ERROR_CONFIDENCE.md)：10項目ledger、共通target、既存保証と不足、未採択epsilon/alpha候補。新Nは未計算。
- [Native費用・baseline草案](C1_4_NATIVE_BASELINES.md)：D1全列挙箇所、G8/P5の利用条件、5費用層、強い対照とUNKNOWN。
- [GPT・ユーザーhandoff](handoff.md)：残件、担当、次の明示承認。新旧SOURCE/結果/STOPを保全。

対象はlinear H4 1.00 Å/STO-3G/DF12、R1新partition（generation prefix L_D=3）、8 qubit/13 basis/324 label、m=5、T=4/5、q=1/r=4、sigma=+1、outer symmetric S2。これらは限定correctness用。旧PR-2の同一科学入力や性能最適条件とは主張しない。

branch: `track-b-c1-3-r2-reference-recovery-c1-4-20261021`

SOURCE: `2be82f7bff78b7d11ffb11013b03a49e296acd1d`

新authorization: `324634a37538eea2bf6153467fa0216ffa8a5c56`

Source/input/runtimeは固定manifestと照合し、[execution_contract.json](execution_contract.json)で旧科学条件の同一性を保持した。[verification.json](verification.json)に保全、[publication_receipt.json](publication_receipt.json)に結果commit・公開bytes照合を記録する。

原データは`phase_b/`と`result/`、人工資料は`phase_a_checks.json`・`artificial_checks.json`。既存R1/C1-2/C1-3/C1-3-R1の成果・authorization・STOP、科学SOURCE/raw/claim/ledger/marker、非公開レビュー原本は変更せず、新namespaceだけ追加した。
