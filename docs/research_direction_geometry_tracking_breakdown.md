# P-C geometry tracking・breakdown validation

最終更新：2026-09-25 JST

## 結論

事前登録したH4 geometry tracking検証を16/16 task完了した。固定decision ruleによるstatusは
`stop_pc_current_h4_family_as_primary`である。

0.80--1.20 Å内の局所的なsigned PF error予測は再現したが、外挿した1.40、1.60 Åでは
coefficient予測誤差が33.653%、123.245%へ増えた。orbital/DF-fragment追跡は8 geometry全てで
独立分解の先頭3 fragmentと同じprefixを選び、独立policyとの差を作らなかった。さらに事前登録した
continuity診断はstretch側の2 breakdownをflagできなかった。

従って、近傍geometryでの誤差相殺は限定結果として保持する一方、このH4・DF rank 12・
`L_D=3`・二次partial-`S_2` familyからP-Cを主研究へ拡張しない。P-A interval、P-B signal-weight、
P-C current H4 familyはいずれも固定停止条件に達したため、現時点でA/B/Cに確認済みの主題はない。

## 固定条件

- molecule：H4 linear chain、0.70、0.80、0.90、1.00、1.10、1.20、1.40、1.60 Å
- basis／sector：STO-3G、8 qubit、4 electron number sector
- DF／prefix：rank 12、`L_D=3`
- PF：二次partial-`S_2`、exact dense randomized-tail reference
- coefficient training：0.80、1.00、1.20 Å
- blind：0.70、0.90、1.10、1.40、1.60 Å
- fit delta：0.025、0.05、0.10
- delta holdout：0.20
- policy：geometryごとの独立先頭3 fragmentと、1.00 Åをanchorにした連続追跡prefix
- 固定pair：0.90→1.10、0.70→0.90、1.40→1.60、0.90→1.40 Å

条件、7 gate、decision ruleは
[事前登録](research/pc_geometry_tracking_breakdown_preregistration.md)で本計算前に固定した。
expected-task artifactは16 taskと実装・runnerのSHA-256を保持する。

## 結果

### representationとdelta

- full operatorのfragment並べ替え差：0
- training 3点のground-energy再現差：最大 `2.220e-15 Ha`
- 先行P-C係数との相対差：最大0.195%
- 全geometry・両policyのdelta holdout最大相対誤差：0.305%

representation integrityとdelta holdout gateは通過した。

### blind coefficient

| geometry (Å) | actual `C(R)` (Ha) | predicted `C(R)` (Ha) | 相対誤差 | 分類 |
|---:|---:|---:|---:|---|
| 0.70 | 0.03335524 | 0.02841653 | 14.806% | inconclusive |
| 0.90 | 0.01730680 | 0.01836862 | 6.135% | smooth |
| 1.10 | 0.01060359 | 0.01097054 | 3.461% | smooth |
| 1.40 | 0.00580008 | 0.00384815 | 33.653% | breakdown |
| 1.60 | 0.00387217 | -0.00090010 | 123.245% | breakdown |

15%以内は3/5点で、固定条件4/5点に届かなかった。中央値は14.806%である。

### pair prediction

| pair (Å) | exact energy difference (Ha) | cancellation ratio | endpoint-bias正規化予測誤差 |
|---|---:|---:|---:|
| 0.90→1.10 | 0.04234609 | 0.2405 | 3.910% |
| 0.70→0.90 | -0.07331970 | 0.3175 | 18.163% |
| 1.40→1.60 | 0.06151018 | 0.1994 | 48.589% |
| 0.90→1.40 | 0.15124612 | 0.4984 | 17.253% |

全4 pairでexact difference 0.02 Ha以上かつcancellation ratio 0.50以下となり、
nontrivial cancellation gateは通過した。一方、予測誤差15%以内は1/4 pairだけで、
固定条件3/4 pairに届かなかった。

### trackingとcontinuity診断

tracked prefixは全8点で`[0,1,2]`となり、independent prefixから変化しなかった。そのため
blind coefficient誤差、pair誤差とも両policyで同一であり、trackingによる改善は0%だった。

分類可能なblind点は4点で、そのうち0.90、1.10 Åのsmooth 2点だけを正しく判定した。
1.40、1.60 Åのbreakdownでは、orbital singular value、ground-state overlap、gap、
fragment similarityのいずれも固定thresholdを越えず、事前flagはfalseだった。
診断正解率は50%で、固定条件80%に届かなかった。

## gateと判断

| gate | 結果 |
|---|---|
| representation integrity | pass |
| delta holdout | pass |
| blind coefficient prediction | fail（3/5） |
| pair prediction | fail（1/4） |
| nontrivial cancellation | pass（4/4） |
| diagnostic transfer | fail（50%） |
| mechanism discrimination | fail（prefix変更0、breakdown未検出） |

結果後にthreshold、geometry、pair、gateは変更していない。先行P-C pilotの局所補間結果を
取り消すものではないが、同じH4 pathへ点やthresholdを足してP-Cを復活させない。
P-Cを再検討するなら、別分子、別PF family、状態変化を含む別条件など、独立した問いとblind領域を
改めて事前登録する必要がある。当初案の比較を続ける場合、未実施なのは解析寄りのP-Dである。

## 証拠

- expected artifact：
  `artifacts/research_direction_geometry_tracking_breakdown/2026-09-25/pc_tracking_breakdown_expected_tasks_v1.json`
- expected fingerprint：
  `cbe260750d081316070d3684a2a24194d91d029ad700a54cf4f61152f2ed4a4e`
- expected file SHA-256：
  `a3dd1621dc8777e9f6429fd2d35c6e1270e18d27e4996c7168ec9bd216b2d5b2`
- final artifact：
  `artifacts/research_direction_geometry_tracking_breakdown/2026-09-25/pc_tracking_breakdown_validation_v1.json`
- final fingerprint：
  `26845effe8efda56390aabdf9e40d61fa3a033e3ac7e6ff6911e2e124156f07a`
- final file SHA-256：
  `58e31d3877d45e91e9c6c9f4238d2c1f876f02bb875643813a126f283f2dd2b7`
- implementation：
  `src/trotterlib/research_direction_geometry_tracking_breakdown.py`
- runner：
  `scripts/run_research_direction_geometry_tracking_breakdown.py`
- test：
  `tests/test_research_direction_geometry_tracking_breakdown.py`
- 実行時間：29.711秒
- 専用test：`4 passed`、訂正済み先行P-Cと合わせて`7 passed`
- 関連test：`14 passed`、全suite：`598 passed, 2 skipped, 4 warnings`、失敗0

artifactはlocal dirty-worktree evidenceであり、immutable CIまたは外部再現ではない。
potential-energy surface、force、反応障壁、別分子、別basis/rank/PF、RPE/RTE sampling、
回路resource、noise、H12、最終総costまたは科学的優位性は評価していない。
