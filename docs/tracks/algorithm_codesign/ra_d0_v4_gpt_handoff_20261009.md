# RA-D0 v4 独立数学監査：GPT handoff

2026-10-09 JST。**最終分類 `V4_EXACT_SOLVER_FEASIBILITY_UNVERIFIED`**。
数学は `V4_MATH_AUDIT_PASS`、独立人工検証207/207 PASS。production実装・登録実行は未認可。

branch=`track-b-ra-d0-v4-mathematical-audit-20261009`。
基点T0.2=`d3a7cbb239487ddedf44699378f6c182c1fe5993`。
設計入力は`06575a3bc9d2354b829e0e9a6c21ad1512f77909`のGPT設計案。
[監査全文](ra_d0_v4_mathematical_audit_20261009.md)と
[監査専用script](../../../scripts/tracks/algorithm_codesign/check_ra_d0_v4_mathematical_audit.py)を先に確認する。

## 成立した十分条件

exact normalizer/構造等式、precision間D identity、非負cost/d、正値norm、workspace preflight、
元tau用preflightの下で、XiとGamma_xi/d/Q/hは安全な上界として成立する。
exact continuous inner feasible pointから階層LRMを通して、sampler、membership、mean、confidence、
非目的resource caps、workspaceの元certificateを同時に満たせる。
固定nでは全て有理係数の線形制約である。

固定tableのx={1/8,1/4}両方でsum v=t、三precisionのD/event/conditional phase identity、
positive norm、Ymax rad<=2/Nを静的に確認した。
saved 2-qubit distinct-basis controlled finite P3、p=(3/4,1/4)が対象であり、
分子geometry/basis/DF rank/split L_D/PF delta窓は適用外。新science評価ではない。

## 必要な区別と反例

- K1 subset K2 subset K3は元certificate classの関係。B2 aliasesをB3へ整数countsのまま合算できる。
  B2/B3を別々にLRMしたlawの一致や、新generatorのdecode像のnestingは仮定しない。
- Primaryは元B2 outer lowerに対するB3 certified upperのstrict改善のまま。
  保守的innerのminimumを元B2 lowerには使わない。
- 元lawがconfidence境界で認証可能でも、新margin付きinnerが空になるexact人工反例がある。
  inner用Farkasを元class infeasible/正常skipとして扱わない。
- 同event内のprecision移動はconfidenceを改善してもresource capを破り得る。
- unequal precision D intervals、negative cost、membership preflight不成立では、提案式を使えない。
  固定tableはこれらの前提を満たす。
- y<=Ymaxだけでは丸め後上端を保証しない。固定tableでは元mean certificateとcolumn L1<2から元Ymaxが従う。

主要前提を全て満たした反例や、数学的blocking issueは見つからなかった。
任意nominal lawの修復保証、登録innerのfeasibility、最適性、RA-RTE優位・新規性は主張しない。
修正候補のproduction採用は0。

## Synthetic・backend・計算費用

人工39 fixtures＋補助小分母LRM網羅168件=207件。期待した受理・拒否・反例検出が全て一致した。
GPT自己検算240例と別であり、T0.2回帰fixtureを新しい成功件数として加えていない。
丸め後lawのdirect Fraction certificateはXi/Gamma acceptanceから独立に再計算した。
wall約0.153 s / CPU約0.152 s / peak RSS 44,116 KiB。

SoPlex等のexact backendはread-only inventoryで利用を確認できなかった。
新install/runtime変更0、synthetic solver calls0、登録solver calls0。
公式能力と、現環境でのparser/primal/dual/Farkas/timing検証を区別する。
後者はUNVERIFIED_BACKEND。旧2秒/LPや総capは継承しない。

**実装への提案：数理上のGOは条件付き。現時点ではproductionへ進めず、
承認済みisolated exact backendの最小off-domain検証を次に設計するかGPTで判断する。**
backend・計算範囲が閉じるまではsource freeze/authorization/registered optimizationを開始しない。

## GitHubで確認する資料

- [入力identity](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/input_identity_v1.json)
- [定理別判定](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/theorem_audit_v1.json)
- [丸めbound・table preflight・反例](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/rounding_bound_audit_v1.json)
- [class nesting](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/baseline_nesting_audit_v1.json)
- [infeasibility規約](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/infeasibility_semantics_v1.json)
- [全207人工検証](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/synthetic_verification_v1.json)
- [backend調査](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/exact_solver_feasibility_v1.json)
- [静的resource reserve](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/resource_cost_audit_v1.json)
- [実行記録](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/audit_execution_v1.json)
- [provenance・全出力hash](../../../artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/evidence_manifest_v1.json)

旧v3/T0/T0.1/T0.2の135 protected hashesは不変。旧source/contract/authorization/consumed marker、
R1/R1.5、Track Aを保持し、旧分類も不変。
registered LP、synthesis/science、production implementation、circuit/matrix/trajectory、DF/NPZ/GPU、
追加authorization/marker変更は全て0。
**資料公開後mandatory STOP。研究方針・次検証の必要性/範囲をGPTへ戻す。**
