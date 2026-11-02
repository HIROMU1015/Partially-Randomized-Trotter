# N3保存証拠の事後保証幅 v1：GitHub公開・再取得

2026-10-12 JST。認可済み一回評価を完了し、専用HIROMU1015 branch `n3-saved-width-v1-evaluation-20261012`へ保存した。
有限保存集合でOCC strict2/8（S1の1base）、BASE strict6/8。元NO_ADDITIONAL_ACCEPTANCE_OBSERVED/2-base未達は保持、Mandatory STOP。

- frozen source11 `3f295df0263059ce16d866f84c1132f8bace0670`、fingerprint `b190ea508afb9a10fad3d3dbfeb92d50f46ad5813baadd70965aa6fc5ad044d1`
- execution authorization commit `8370128c94641f70f21113ea9a82c85cc33a08af`
- immutable evaluation payload `652667187ce83c7c36041820a0a752669134865f`
- plan SHA `bd3f5f4a21deafb227a6794d3277ca3eab1d9a30c610236d42294103e91e53dd`
- ledger seal `8b1e75e89da78c13b82f24896db387adec6e76597ec12669a7549279592d0d69`、input SHA `c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4`
- [結果・比較・費用・限界](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/docs/research/n3_saved_width_v1_results_20261012.md)
- [一次result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/artifacts/n3_saved_width_v1/2026-10-12/evaluation1/result.json)
- [小さいcase summaries入口](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/case_summaries/index.json)
- [aggregate summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/aggregate_summary.json)
- [実行STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/artifacts/n3_saved_width_v1/2026-10-12/evaluation1/MANDATORY_STOP.json)
- [source/input/result分割inventory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/publication_inventory/index.json)

GitHubから別bare cacheへfetchし、82 blobs / 7,782,198 bytesをexport、local byte/SHA/sizeと照合PASS。
新payloadはfetch前に取得cacheへ存在せず、GitHub由来の過去object cacheだけを再利用した。local object copy/alternatesなし。
source11/旧source8、共通原pack/ledger/secondary/plan/旧result/STOP、2832保護tree entries、旧manifest96 entriesと他field、root1347 files/HEAD/既存dirty entriesを保全した。
元巨大ledgerを別の重複blobへ書き換えず、既存immutable objectを使用した。

取得したsourceと保存結果で独立CLI checkerを実行し、8件すべてPASS、元checker conditions 5,284,307。
finite集合の完全性/PSD/κ/τ/Z/原pointer/区間多項式endpoint/比較labelを確認し、local保存auditと同じ結果だった。
source/input/result byte一致と保存証拠再検証であり、same local environment。immutable CIや外部環境での科学batch再現とは区別する。
remoteでproducer/新input/eig/optimizer/新証拠/E_forced/旧batch再実行は0。24 syntheticは準備時archived logとsource identityを再利用し、再実行なし。
既存global manifestのdescription欠落FAILは保持し、今回entry schema/prefix保全は別にPASSとして記録した。

実対称one-body＋非負unordered density、一外部mode、full7/8/9/10、active6/7/8/9、ν2/3/3/4、元δgoal.004/.006/.003/.005。
化学geometry/basis/DF rank/L_D/PF窓は適用なし。同じ保存U,c_starで有限保存集合を比較したposthoc development。
S1_edge_pairの幅縮小約20.424087%、S1_distributed_three約0.343340%は保証幅の事後差であり、真誤差/条件族最適性/旧族不可能性/独立移送/新規性の採択判断ではない。
新結果の研究上の意味・費用対効果・継続判断はGPT担当。詳細レビュー開始は別認可。中心テーマnull、次段false、Mandatory STOP。

[公開receipt](../../artifacts/n3_saved_width_v1/2026-10-12/evaluation_publication/publication_receipt.json)、[remote saved audit](../../artifacts/n3_saved_width_v1/2026-10-12/evaluation_publication/remote_saved_verifier.json)。
評価はattempt1/retry0。N1追加探索停止、N2保留、実分子/多外部/一般二体拡張は保留を維持する。

[実行command記録](../../artifacts/n3_saved_width_v1/2026-10-12/evaluation_publication/commands_record.json)。一回producer実行と保存checker再検証を区別する。
