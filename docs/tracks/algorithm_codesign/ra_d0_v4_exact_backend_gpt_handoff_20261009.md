# RA-D0 v4 Exact Backend Pilot：GPT handoff

2026-10-09 JST。**`V4_EXACT_BACKEND_RESOURCE_INCONCLUSIVE`**。
branch：`track-b-ra-d0-v4-exact-backend-pilot-20261009`。
基点：`beb82427d202f479cc2ba954480d73a51941e322`。
本書を含むGitHub固定commitの[全文report](ra_d0_v4_exact_backend_pilot_20261009.md)を確認してください。

## 何が起きたか

SoPlex 7.0.0（公式`release-700`、source `6657fb3b27044bad7bf2bb58b16de2461de82109`）と
GMP 6.2.1 / Boost 1.74.0をprivate領域へ展開し、静的library buildは完了した。
system/Python/旧runtimeの変更は0。既存g++ 11.4.0とmakeを使用した。
しかし初回harness compileをRSS guardが6.260 sでkillした。

**Codexが作成したguardは、supervisorのtreeに含まれるcompiler treeをもう一度加算していた。**
1,537.664 MiBというindicatorは二重計上を含み、true peak RSSではない。
SoPlexが実際に1,536 MiBを超えたとは結論できない。
grandchild CPU回収と全build output cap監視にも未決点がある。
cap発火後のretry禁止を適用し、guard修正も再buildもせず停止した。

## 完了／未実証

| 項目 | 判定 |
|---|---|
| private dependency展開、Ubuntu package checksum照合 | 完了 |
| SoPlex静的library compile | 完了、7.071 s |
| harness executable compile | guard停止、binaryなし |
| exact rational I/O | NOT_RUN |
| exact primal / dual / Farkas | 全てNOT_RUN |
| independent Fraction verification | NOT_RUN |
| demonstrated solvable LP dimensions | 未確認 |
| synthetic / registered LP calls | 0 / 0 |
| build / solver retries | 0 / 0 |

23件の人工fixture定義を保存したが、未実行である。
B2-shaped 8×3（28 variables）、B3-shaped 7×3（22 variables）を含む。
LP timings、証明取得費用、variation、production件数への外挿はできない。
primal/dual/Farkasの失敗が観測されたという意味でもない。

## 次の判断として依頼すること

**現証拠によるbackend採用・production実装は保留。**
技術的には、RSSをsupervisor rootから一回だけ集計し、CPU accountingとoutput capをreviewした
別pilotを認可するか検討する余地がある。ただし今回はその修正・再実行を認可されていない。
backendが利用不能という結果ではなく、監視実装の問題で機能実証に到達していない。
研究方向をCodex側で変更せず、追加検証の情報価値・範囲と実装方針をGPT側へ戻す。

新production source、registered optimization、budget freeze、新authorization、旧one-shot再実行は0。
旧135 paths＋数学監査18 paths、計153 protected hashesは不変。
旧v3/T0/T0.1/T0.2/R1/R1.5/Track Aのsource、結果、分類、STOPを保持した。
新science/synthesis、angle/precision、IS/CTS、circuit/matrix/trajectory、DF/molecule/NPZ/GPUは全て0。
**公開後mandatory STOP。PASSへの読み替え、cap緩和、solver/compile retry、自動次stage移行はしない。**

## GitHubで確認する証拠

- [全証拠manifest](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/evidence_manifest_v1.json)
- [入力・153 protected hashes](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/input_identity_v1.json)
- [backend/dependency/license inventory](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/backend_inventory_v1.json)
- [build command・identity](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/build_runtime_identity_v1.json)
- [resource benchmarkの限界](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/resource_benchmark_v1.json)
- [guard停止の分類・retry=0](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/failure_semantics_v1.json)
- [検証状態](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/verification_v1.json)
- [23件の未実行人工fixture](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/synthetic_fixture_manifest_v1.json)
- [raw harness compile resource record](../../../artifacts/track_b_ra_d0_v4_exact_backend_pilot/2026-10-09/build_logs/harness_compile.resource.json)
- [実行時supervisor：既知の二重計上不具合を保持](../../../scripts/tracks/algorithm_codesign/exact_backend_pilot/supervise.py)
