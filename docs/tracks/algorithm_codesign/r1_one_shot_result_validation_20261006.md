# R1一回結果・保存値監査 — mandatory STOP

2026-10-06 JST。固定source Sの直接子authorization-only Aから、R1を一回だけ実行した。
terminalは **`R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW`**。
126/126 synthesis keys、264/264 resource rows、72/72 comparison groupsが完了した。
controlled 132 tasksは全件適格。保存値監査は`PASS_SAVED_FIELDS_ONLY`。
**scienceは終了しmandatory STOP。retry=0、次stageは未認可、研究判断はGPTへ戻す。**

## 固定source・authorization・結果

- source S：`d43d64a821a0249a0dfab12a2472bd3a72fdee74`。
- [source review対象](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d43d64a821a0249a0dfab12a2472bd3a72fdee74/docs/tracks/algorithm_codesign/rte_reallocation_r1_source_review_request_20261006.md)。
- 準備記録P：`09fe527274e8de5c0d808ada5890fa06a59f78ea`。実行HEADには使わなかった。
- authorization A：`411f08f768244fe87b600d82308c3851847fe9e4`、parents=[S]。
- [固定Aのauthorization JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/411f08f768244fe87b600d82308c3851847fe9e4/artifacts/track_b_rte_reallocation_r1_source/2026-10-06/authorization.json)。
- [固定Aの明示指示receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/411f08f768244fe87b600d82308c3851847fe9e4/docs/tracks/algorithm_codesign/r1_execution_authorization_receipt.md)。
- branch/worktree：`track-b-r1-one-shot-execution-20261006`、
  `/home/abe/Project/prt-worktrees/track-b-r1-one-shot-execution-20261006`。

Aの差分はauthorization JSONとreceiptだけ。clean HEAD=AをGitHubへ公開・remote照合してから、
固定source/runtime/契約のlaunch確認を通し、一回だけrun modeを呼んだ。
markerはrunnerが登録合成前にexclusive createした。原result/markerは以後変更していない。
authorizationのscience=trueはこの消費済み一回の許可履歴であり、二回目の認可ではない。

| Identity | SHA256 |
|---|---|
| contract | `e8a4201203dc8d2fbb35c7f3025079d7f0f1dce270be55afd1b17f2505717025` |
| authorization | `9d30dd6cbe5b4d47d4b9080db656564fec6503d9e77a4f08442a39c8f30a582f` |
| tool identity | `803d902a18d925a9565749fbc64422dca4979bfa6c0bd8b438b28de18d3aa984` |
| key inventory | `fd5e8e35dcda12cb9a070bf63cefca31052642a43a7f648802942440035373e4` |
| original result | `f726ad70cb2643533f0d037b518cde1b702724adb4e6571fea25e26e4bfdd61e` |
| one-shot marker | `f25000ee5e3b94eb499b4a89bb28814e3de1aea641b249bcbb9d953d427c2ced` |

## 入力と比較scope

[preregistration v2](rte_reallocation_r1_preregistration_v2.md)の2-qubit synthetic involution taskだけ。
targetはP₃(-iσxR̂)、R̂=3Q₀/4+Q₁/4、m=3 / K=2、x={1/8,1/4}、σ=±1。
分子、geometry、basis、DF rank、Hamiltonian splitは該当しない。

| Context | Q₀ / Q₁ | Arms・役割 |
|---|---|---|
| Pauli commuting | Z₀ / Z₁ | ordinary / PTSC-K0 / A / collected CTS、collection control |
| Pauli noncommuting | Z₀ / X₀X₁ | 同上、phase/collection control |
| Distinct basis | Z₀ / V†Z₁V、V=exp(-iπX₀X₁/16) | ordinary / PTSC-K0 / A。controlledがprimary |

native ε={10⁻³,10⁻⁴,10⁻⁶}、canonical sampling、IS/PAIなし。
ordinary native rowsはdiagnosticで、controlled one-block coherent taskへ混ぜない。
distinct-basis toyもPauli展開可能であり、登録I0-style native実装でPauli reductionを使わない比較に限る。
実問題でI0しか使えないこと、I1取得不可能、DF規模の古典取得優位は立証しない。

metricは **native IR after shared exact cancellation, additive synthesized-primitive Clifford+T counts**。
basis/conjugator/odd extra Q/controlled relative phaseを含む固定loweringであり、
post-synthesis whole-circuit最適化、hardware cost、分子DF wrapper改善ではない。
primaryはB²、implemented second moment、E[T/CX/1Q]、workspace、C_classical structured record。
secondaryは共通finite-confidence taskのG_T/G_CX/G_1Q。single scalar research GOはない。

## 完全性・guard・資源

| 保存項目 | 結果 |
|---|---|
| run / retry | 1 / 0 |
| synthesis attempts / pygridsynth calls / saved sequences | 126 / 126 / 126 |
| resource rows / comparison groups / comparison pairs | 264 / 72 / 192 |
| 保存event records | 2,904 |
| controlled tasks / axes | 132 / 264、全132 tasks ELIGIBLE_COMMON_FINITE_CONFIDENCE_TASK |
| primary distinct-basis controlled | 36 arm rows / 12 groups / 24 A-vs-baseline pairs |
| strict synthesis guards | 全126 saved guards PASS、up_to_phase=false、W phase identity保持 |
| technical failure / bias-ineligible / shot-cap-ineligible | 0 / 0 / 0 |
| recorded BudgetGuard wall / CPU | 5.277212 s / 5.277026 s |
| recorded peak RSS | 295,188 KiB（約288.27 MiB）/ cap512 MiB |
| per-key max wall / CPU | 0.302715 s / 0.302649 s |
| max sequence length / original result bytes | 221 / 7,824,519、各cap20,000 / 16 MiB内 |
| sufficient shots/axis range | 762,708–1,106,746 / cap10⁹ |
| actual shots / exact-signal evaluation / trajectory | 0 / 0 / 0 |

上記wall/CPUはrunnerが保存したBudgetGuard区間のusageであり、公開・監査まで含めたend-to-end時間ではない。
runtimeはPython3.10.12、pygridsynth2.0.0、mpmath1.3.0、固定source .py hash一致。
guardは100 dpsのphase-preserving interval Frobenius upper。saved error/εの最大は約0.17445。
coherent measurement biasは係数変位＋2Σαδを含み、controlled保存値の範囲は約2.754×10⁻⁷–7.876×10⁻⁴。
ε_axis=0.005、ε_complex=0.01、α_axis=1/5280、familywise α=1/20を変更していない。
shot数はBernstein sufficient forecastであり、測定による統計的精度の検証結果ではない。

## Primaryの全登録比較

以下は保存されたratio/labelの記述的集計。小数は表示用で、厳密fractionは原result/summaryを参照。
σの二符号を独立replicationとは扱わない。precisionも事前登録された比較座標であり、新入力ではない。

| 比較（各12 groups） | B² LOWER | E[T] LOWER / HIGHER | E[CX] LOWER / HIGHER | E[1Q] LOWER / HIGHER | G_T LOWER / HIGHER | G_CX LOWER / HIGHER |
|---|---:|---:|---:|---:|---:|---:|
| A / ordinary | 12 | 8 / 4 | 12 / 0 | 8 / 4 | 8 / 4 | 10 / 2 |
| A / PTSC-K0 | 12 | 4 / 8 | 0 / 12 | 4 / 8 | 4 / 8 | 4 / 8 |

A/ordinary B² ratioは約0.99925120–0.99994776、A/PTSC-K0は約0.99457154–0.99932040。
normalization改善だけをmethod採択根拠にしない。A/PTSC-K0ではnative CXが全12 groupsで増える一方、
shot込みT/CXが低い条件もある。全resource座標のstrict dominanceを要求する契約でもない。

| x | σ | ε_native | G_T A/ordinary | G_T A/PTSC-K0 | G_CX A/ordinary | G_CX A/PTSC-K0 |
|---|---:|---|---:|---:|---:|---:|
| 1/8 | -1 | 1e-3 | 1.06294841 | 1.07243346 | 0.97750498 | 0.98513582 |
| 1/8 | -1 | 1e-4 | 0.90836269 | 0.91419698 | 1.00028937 | 1.00696544 |
| 1/8 | -1 | 1e-6 | 1.02299307 | 1.03015610 | 0.99914894 | 1.00569058 |
| 1/8 | +1 | 1e-3 | 1.06294841 | 1.07243346 | 0.97750498 | 0.98513582 |
| 1/8 | +1 | 1e-4 | 0.90836269 | 0.91419698 | 1.00028937 | 1.00696544 |
| 1/8 | +1 | 1e-6 | 1.02299307 | 1.03015610 | 0.99914894 | 1.00569058 |
| 1/4 | -1 | 1e-3 | 0.79628191 | 0.82348951 | 0.89746437 | 0.92741074 |
| 1/4 | -1 | 1e-4 | 0.99018633 | 1.01526759 | 0.98814763 | 1.01122247 |
| 1/4 | -1 | 1e-6 | 0.99721320 | 1.02275977 | 0.99604766 | 1.01858190 |
| 1/4 | +1 | 1e-3 | 0.79628191 | 0.82348951 | 0.89746437 | 0.92741074 |
| 1/4 | +1 | 1e-4 | 0.99018633 | 1.01526759 | 0.98814763 | 1.01122247 |
| 1/4 | +1 | 1e-6 | 0.99721320 | 1.02275977 | 0.99604766 | 1.01858190 |

これらのratioはstored sufficient-shot countとstored additive primitive costからの既存会計。
精度による合成列costとcoherent-bias budgetの違いを含む。最小ratioの条件だけをpositive証拠へ選ばない。
新しいmateriality閾値、Pareto採択、algorithm昇格、科学分類を追加していない。

## Controlsと全vectorへの入口

Pauli-only collected CTSはI1 controlとして保存し、distinct-basis primaryへ追加しなかった。
controlled A/CTSのB² ratioは、commutingで約1.00004052–1.00063904、noncommutingで約1.00580544–1.02264840。
保存G_Tではcommuting12 groups中LOWER4/HIGHER8、noncommutingでLOWER2/HIGHER10。
両controlでA/CTS G_CXはHIGHER12。CTSがすべてのT条件で勝ったという解釈はしない。
ordinary diagnostic rowsのzero-CX baselineは`ZERO_OR_UNRESOLVED_BASELINE`として保持し、division/winへ使わない。

- [全264 resource rowsの表示CSV](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/resource_rows_display_v1.csv)：
  E[T/CX/1Q]、workspace、classical subvector、bias、shots、G_1Qを含む。12桁表示だけでinterval labelを変更しない。
- [記述summaryと全24 primary比較](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/descriptive_summary_v1.json)。
- [原result](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/result.json)：
  全sequence、phase-preserving guard、IR identity、係数/probability、exact fraction、全event/row/comparison。
  原resultは約7.8 MBなので、GitHub表示制限時はCSV/summaryから入り、正確な値は固定commitのraw fileを読む。
- [marker](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/one_shot_consumed.json)、
  [evidence manifest](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json)。

## 監査の範囲と限界

[saved-field audit](../../../scripts/tracks/algorithm_codesign/audit_r1_saved_result.py)はstdlibだけで、科学sourceをimportしない。
[audit記録](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/saved_field_audit_v1.json)は以下を確認した。

- S/Aのparent・二path制限・明示指示、contract/tool/authorization/key/result/marker identity。
- critical26 pathと既存証拠66 pathのhash、既存27 focused local testsのsource保持。
- 全126 sequence hash、T/T†/1Q/W count、保存strict error guardとcaps。
- 全264 row / 2,904 eventのexact rational係数・probability・B²・expected native cost・bias係数2・workspace。
- 132 controlled tasksの保存budget inputとG_T/G_CX/G_1Q会計、72 groups / 192 pairsの既存ratio/label算術。

strict matrix guard、native circuit/IR hashの生成、Bernstein logarithm/shot ceiling、finite first-moment signalは再計算しなかった。
原result/markerへの書き込みは0、新target・合成・circuit build・resource science・測定・trajectoryは0。
これはsource-bound local evidenceと保存値照合であり、immutable CIやindependent replicationではない。
共有APIとTrack Aは変更していない。BF/BM/SP等の過去STOPも解除していない。

## STOPとhandoff

run terminalと保存比率の記述だけを報告し、Aの主method昇格や次stage実行は判断しない。
[GPT review依頼](r1_post_run_gpt_review_request_20261006.md)へ戻す。
研究方針・RQ・新規性・論文着地点・追加検証の必要性と範囲はGPTが判断する。
**mandatory STOP、run=1、retry=0、追加science未認可。**
