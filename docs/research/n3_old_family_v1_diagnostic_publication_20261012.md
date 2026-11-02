# N3旧条件族診断 v1：結果公開・GitHub再取得と独立保存監査

2026-10-12 JST、専用branch `n3-old-family-v1-diagnostic-20261012`。
認可済み一回診断でS1二例ともCERTIFIED_NORM_AND_OLD_FAMILY_SEPARATION。原primary/2-base未達/旧STOPを保持、Mandatory STOP。

- frozen source13 `1aa12ab0b777b28f1428d9bb8417af93c4376aae`、fingerprint `46baa537269db923dd8b8c95e4788555471c60bad1572d3a01ec1e70245c6488`
- immutable result payload `86c69983544064d1ed7ee7910634c541c7b982b7`
- plan SHA `ef3c66d2e01e47027d9d382afc324aab9b7263aab3d550c3576c31e0f5bd6ff5`、input fingerprint `81e90cb760d6e6e9774aad808ed40953bae0b461c18532bff78d7ead6e4e584d`
- [結果・証拠・費用・scope](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86c69983544064d1ed7ee7910634c541c7b982b7/docs/research/n3_old_family_v1_results_20261012.md)
- [一次result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86c69983544064d1ed7ee7910634c541c7b982b7/artifacts/n3_old_family_v1/2026-10-12/diagnostic1/result.json)
- [小さい証拠summary索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86c69983544064d1ed7ee7910634c541c7b982b7/artifacts/n3_old_family_v1/2026-10-12/execution_audit/case_summaries_index.json)
- [edge_pair complete certificate](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86c69983544064d1ed7ee7910634c541c7b982b7/artifacts/n3_old_family_v1/2026-10-12/diagnostic1/S1_edge_pair.json)
- [distributed_three complete certificate](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86c69983544064d1ed7ee7910634c541c7b982b7/artifacts/n3_old_family_v1/2026-10-12/diagnostic1/S1_distributed_three.json)
- [STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86c69983544064d1ed7ee7910634c541c7b982b7/artifacts/n3_old_family_v1/2026-10-12/diagnostic1/MANDATORY_STOP.json)
- [source/input/result分割inventory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86c69983544064d1ed7ee7910634c541c7b982b7/artifacts/n3_old_family_v1/2026-10-12/execution_audit/publication_inventory/index.json)

GitHubへpush後、別bareの過去GitHub object cacheへfetchし、69 blobs / 7,560,899 bytesをexportしてlocal byte/SHA/size一致PASS。
新payloadはfetch前にcacheへ存在しなかった。local object copy/alternatesなし、元巨大ledgerは既存immutable objectで、重複blobを追加しない。
取得source/input/plan/resultで独立saved checkerを実行し、WのPSD・両端negative trace・NORMと元OCC/U,c、identity、scope/STOPを全2件PASS確認。
新dual checker236 conditions/PSD2、原OCC checker2,873,507 conditions。local archiveのsaved auditと同じ結果だった。
producer再実行0、新W/κ/μ提案0（取得監査時）、eigh0、新input/U,c/OCC0。
same local environmentでの保存再検証であり、immutable CIや外部環境での科学batch再現ではない。23 syntheticは既存archive/source identityを再利用し、再実行なし。

source13/plan/元pack/ledger/保存OCC勝者、2920保護tree entries、旧manifest100 entries/他field、root1354 files/HEAD/dirty entriesを保持。
root/BentoSlide/別Trackをcommit/pushせず、元NO_ADDITIONAL_ACCEPTANCE_OBSERVED/2-base未達と旧STOPを変更しない。
既存global manifestのdescription欠落FAILは保持、新entry schema/prefix保全は別にPASS。

モデルはS1 star、実対称one-body＋非負density、full10→active9、ν4、一外部mode8、元δgoal=.005。
化学geometry/basis/DF rank/L_D/PF窓なし。今回δは保存OCCの正の有理上端、同じU/c/vを維持したposthoc development。
認証範囲は指定δでのNORM不成立と全旧κ/μ条件族の不可行性（FIVE包含）。同じS1 baseの2方向であり、新規独立移送ではない。
旧族の最小保証幅・最適旧法との差率・全既知法への優位・新規性・量子資源削減・中心テーマを採択していない。

[compact receipt](../../artifacts/n3_old_family_v1/2026-10-12/execution_publication/publication_receipt.json)、[remote independent saved audit](../../artifacts/n3_old_family_v1/2026-10-12/execution_publication/remote_saved_verifier.json)。
attempt1/retry0、次はGPTによる新分離結果の科学判断（レビュー開始認可別）、中心テーマnull、次段false、Mandatory STOP。
