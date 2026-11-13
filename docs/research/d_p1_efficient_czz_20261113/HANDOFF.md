# HANDOFF：効率的controlled ZZとP1（run02完了）

参照：[REPORT](REPORT.md)、[PLAN](PLAN.json)、[run02/RESULT](run02/RESULT.json)、[run02/PHASE_A](run02/PHASE_A.json)、[連続性の照合](run02_precheck/POST_RUN_CHECK.json)、[run01/STOP](run01/STOP.json)。

## ユーザー

- P1 run02は完了した（`P1_COMPLETE`）。SOURCE `20fe1194`、full-wrapper compile 8/8件、4 path、旧回路のcompile 0件。
- run01のeventとのSHA-256一致と、phase A parityの全件PASSを確認してからcompileした。
- 結果（X＝Y）：

  | path | C_optimized | S | D |
  |---|---:|---:|---:|
  | m0/0 | 1,906 | 1,845 | +61 |
  | m0/1 | 1,674 | 1,529 | +145 |
  | m6/0 | 1,826,433 | 1,747,585 | +78,848（C比4.3%） |
  | m6/1 | 1,815,119 | 1,750,316 | +64,803（C比3.6%） |

- 新loweringでも、Dは正のまま残り、N_ZZだけでは説明できない。成分の識別には、P2の設計が必要。
- **P2/P3、fit、全lawの平均価格、N/RMSE/Gには進んでいない（mandatory STOP）。** 次の段階に進むかどうかは、ユーザーが判断する。

## Codex C

- 新しいlowering（`CONTROLLED_ZZ_CX_CRZ_CX_V1`）は、構造S（`Accounting`の`'crzz'`＝CX·CRZ·CX）と同じ分解である。
- 既存のK0/K1・E[S]は、そのまま利用できる。
- P1のDは、PCG64/CDF samplerから取ったpath標本である。ideal lawのE[S]と組み合わせるときは、law距離が未資格であることを保持してほしい。今回、C側の作業は不要。

## Codex B

- 作業は不要。
- 将来の価格recordでは、旧lowering（`CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424`）と新lowering（`CONTROLLED_ZZ_CX_CRZ_CX_V1`）を別のSOURCE/identityとして扱い、混同しない。
- 旧価格（C1、長時間研究、Full-H）は旧loweringのままで、上書きしない。
- P1の4 pathは、finite development cohortの観測である。価格modelの採用根拠にはしない。
