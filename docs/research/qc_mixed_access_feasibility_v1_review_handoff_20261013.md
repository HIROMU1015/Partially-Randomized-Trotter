# mixed-access feasibility v1：新結果のGPT判断用引継ぎ

2026-10-13 JST。[一回結果・scope・限界](qc_mixed_access_feasibility_v1_results_20261013.md)。
**重要科学レビューの開始承認待ち。これはCodexの結果引継ぎで、新しいGPTレビューではない。Mandatory STOP。**

## 取得済みの新情報

H4直線1.00Å/STO-3G/real RHF/8mode・ν4/frozen coreなし、一fresh input、16DF全保持（numerical rank10はreportingのみ）、L_Dなし/PRなし。
T1/4、r倍増1…1024、εU=.001、7方式154 PF evaluations完了。operator errorは256rows×70ν-inputでphase/leakageを含む。ground-energy1mHa・QPE全taskの達成を意味しない。
fraction.6ではR4のrはT1/4で1/4、E4/D4は2/8。R2はT4でE2より半分のrとなった。
一方、固定template/κ3,6,12の費用hypothesisでは273 eligible R frontiersの全てが負、`NO_POSITIVE_BUDGET_IN_REGISTERED_MODEL`。
入力SHA `f345af12281836731c3122bb2a4c0c2b389d7c90ded7ba16976d3cfeeb7bd951`。source payload `102e6b31401356d6eea775169041ebdf2cfa6ba0`、承認・preflight `214f9ab2d7d3e0b64b287f14f49795c99a1a8244`。
norm/CAR saved audit154件、resource arithmetic1386 rows/273 frontiers、source16locks、raw163files PASS。物理量子回路のcompile・function backendは0。

## GPTが判断する論点

1. このdevelopment一分子・有限grid・保守的templateの負frontierを、詳細backend投資の判断にどう用いるか。upper envelopeとhypothetical synthesis価格から普遍的な不能は証明できない。
2. ideal PFのstep低下がある一方、登録費用モデルではknown overheadが大きい。既存方法と同機能・同精度を維持して、費用差を変え得る具体的な構造/実装の根拠が残るか。
3. 追加情報の価値があるなら、必要な情報と最小scopeを結果前に具体化するか、現候補への投資を止めて他方向を比較するか。Codexはどちらも採択していない。
4. ground-energy全taskに接続する根拠は未取得である。今回のoperator比較だけを研究の総資源改善claimへ読み替えない。

現constructorの独立新規算法としての非採択は前の完了レビューの判断で保持する。同じ新規性監査を繰り返すことは今回の論点ではない。
N1STOP/N2HOLD/N3CLOSED、他Track、旧結果/STOP/原契約、global manifest既存description欠落FAILを維持する。

## 資料索引

- [一次result](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/result.json)、[全probes](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/probes.json)、[resource/symbolic frontiers](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/resource_evaluation.json)。
- [raw seal](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/raw_seal.json)、[原Hamiltonian pack](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/input.npz)、[structure](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/structure.json)。
- [独立saved audit](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/independent_audit.json)、[disk replay](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/independent_checker_replay.json)、[保存整数/Fraction監査](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/saved_execution_integrity.json)。
- [集計](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/descriptive_summary.json)、[実行性能](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/execution_metrics.json)、[reference guard](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/reference_checks.json)。
- [原科学計画](qc_mixed_access_feasibility_plan_v1_20261013.md)、[固定実装契約](qc_mixed_access_source_preparation_v1_20261013.md)、[source manifest](../../artifacts/qc_mixed_access_source_v1/2026-10-13/source_manifest.json)、[承認](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/user_authorization.json)。

公開・byte照合はCIや外部科学再現とは別。今回の実行承認は消費済みで、科学レビュー開始や追加計算を認可しない。

## 公開・remote取得確認

固定証拠payload `ae3975de24bff6ab6aec3494b522851db2a8f3dd`。初回data commit `c23b3f1422fdf3956a823af6fc7197a9942194f8`へ、ignoreされていた保存済み実行logを追加して証拠索引を完成した。source・科学結果・raw seal・生成索引は変更せず、科学再実行0。[補完記録](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/publication/retrieval_precheck_failure.json)。
[GitHub固定結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae3975de24bff6ab6aec3494b522851db2a8f3dd/docs/research/qc_mixed_access_feasibility_v1_results_20261013.md)、[固定引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae3975de24bff6ab6aec3494b522851db2a8f3dd/docs/research/qc_mixed_access_feasibility_v1_review_handoff_20261013.md)、[一次resource JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae3975de24bff6ab6aec3494b522851db2a8f3dd/artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/resource_evaluation.json)。
[remote receipt](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/publication/receipt.json)：GitHub196変更blobs/65,061,432bytes、生成索引182files、raw163files、source16locks、旧優先12blobs、追加relative links48件一致。local object alternatesなし。新しい科学計算・tests・GPTレビューを公開確認のために実行していない。
これは資料の取得性・byte保全の確認。CI/外部科学再現とは異なる。次はGPT科学判断の開始承認待ち、Mandatory STOP。
