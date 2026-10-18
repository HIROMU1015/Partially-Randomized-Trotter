# Decision Handoff：B3/C1後に強い対照と何を比較できるか

[入口](README.md) / [条件対応表](comparison_conditions.md) / [既存結果](existing_strong_baselines.md) / [必要時だけの追加案](evidence_gaps_and_minimal_plan.md)。状態は`STATIC_COMPARABILITY_AUDIT_COMPLETE_WITH_CONTRACT_GAPS`。

**既存G9 v2内のmatched P5 native比較は、追加計算なしで閉じている。** 一方、G7または新B3/C1の結果とG9のN/native総量を横断して単一順位にするには、confidence/bias・provider・direct/helper・費用scopeの一致が必要である。現在Codex Aの完了・結果を確認済みとはしていない。

## 1. 現在引用できる内容

- **General Full**：G7の登録3対照に対する限定conditional期待T優位を保持する。G9の同一provider/direct/policyでもordinary/partial/P3より小さいT切片という既存結果があり、保存affine判定はmatched CTSとのT-primaryでもFull側非劣を記録している。
- **Closed P5**：G9登録direct6方式のT切片とKが最小という既存結論。General Fullに対してもT-primaryで強い同family fast pathとなる。一般Fullの構成価値を否定する普遍命題ではない。
- **Matched CTS**：G9ではnative CXに利点があり、T-primaryとのトレードオフを残す。T=0 real correction10 eventを消さず、任意sampling/辞書/精度最適化に対する最適性を主張しない。
- G9 directのworkspaceは1、helper診断は2（system3以外）。1Q/期待量/capは保存表を参照できる。depth・全N古典費用・実state/backend費用は未確定。

上記は原報告・保存判定の引用で、新しい再会計・再ランキング・科学的GOではない。[G7R][G7R] [G9R][G9R] [G9AFF][G9AFF]

## 2. 何が不足するか

1. **共通budget**：G7は3ρ/16 axes/α=1/320、G9は8ρ/22 axes/α=49/22000＋resource allocation。total0.05、ε_axis、H/Kが同じでもNを混合しない。G9内の比較を引用するだけなら新budgetは不要。
2. **Provider/実装identity**：G7の抽象helperとG9の明示3-qubit/directは同じ価格モデルではない。G9の既存provider・direct/helper rows・19 RZ keysを再利用できるか、将来公開されたC1のidentityを先に照合する。同じcontextなら再compile必須ではない。
3. **強い対照の古典費用**：4方式B3結果だけではClosed P5/CTSの同じ入口/監査/保存/Pauli取得・全N費用が埋まらない。G9のsmall reference/native会計CPUをproduction μとして使わない。
4. **主張scope**：native T-primaryかCX/depthか、固定workloadの実費か母期待費用か、query-onlyかI1かを区別する。prep/readout/backendは未知のままで、0として総資源優位を完成させない。

## 3. 最小の次段階

最初の担当は**ユーザー**：既存の「m=5ではClosed P5が強く、FullとCTSにはT/CXの交換関係がある」という限定証拠で判断できるかを決める。それで十分なら追加検証は不要である。

追加が必要なら、まず**Codex**がCodex Aの**公開固定資料だけ**を使うP1 identity照合を行う。別途指示が必要で、今はAのSOURCE/schema/契約・作業領域へ介入していない。次いで、判断を変える不足がある場合だけP2（共通policyの数値再解析）又はP3（強い対照までの古典計測）を検討する。必要な実装・科学実行・新native取得にはそれぞれ対象限定の承認が要る。

「Fullがordinary/partial/P3より小さい費用を得た」から「Closed P5/CTSを含む全既知法より有利」へ拡張する根拠はない。同じm=5の強い対照はすでに評価済みなので、これを未取得として取り直す計画に戻さない。新m・新入力・新provider探索、G10等の後続系列の再開へも進まない。

## 4. 保全・STOP

専用branchに新規文書・管理metadataだけを追加。G9 v1の技術失敗、v2完了結果・別承認/消費済みmarker、G7/G8/W2b/F2/F3、既存研究状態を保持した。[旧失敗][G9V1] [v2境界修正][G9BOUND]は同じ科学再現の二重証拠として数えない。G7のprimitiveをG9へ再利用した18 keysも独立再現ではない。

[管理検査](verification.json)はGit/blob/hash/link/静的対応に限定し、旧25-check verifier、数式・native比較・testsは実行していない。他worktree保全の観測範囲は親tracked/root原本と既往W2c-A checkout、固定base blobs、既知consumed ledgerに限る。Codex Aのlive私有状態は調べず、他worktree全体を監査したとはしない。

新科学計算・数値再解析・科学SOURCE変更/import/実行・実装・test・compile/synthesis・新非自明証明・重要レビュー・科学条件採択・authorization/grant/marker/claimはすべて0件。[公開照合](publication_receipt.json)完了後、**mandatory STOP**。追加実行・科学判断へ自動移行しない。

<!-- FIXED_REFERENCES -->

[G7R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[G9AFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_affine_T_comparison_v2.json
[G9BOUND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/docs/tracks/algorithm_codesign/g9_v2_api_boundary_source_review_20261010.md
[G9R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[G9V1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/docs/tracks/algorithm_codesign/g9_results_and_gpt_handoff_20261010.md
