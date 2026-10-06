# SP-1 one-shot結果・保存値監査

2026-10-06 JST。**SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW / mandatory STOP / retry0。**
明示指示に基づく一回の科学実行を完了した。12 wrappers / 48 mask rows / 96 axesを保存し、
保存値監査はPASS。42 rowsが適格、6 rows（12 axes）がモデル上のshot cap超過。
technical runtime/memory/output/numeric failureは記録されていない。
研究Bの方針・RQ・新規性・追加検証の必要性は[GPT reviewへ戻す](sp1_post_run_gpt_review_request_20261006.md)。

## 実行・証拠identity

| 項目 | 固定記録 |
|---|---|
| science source S | `0d01ed9a332ebc5b66ed08acf56214a9b9c0236d` |
| source最終review R | `d0502f22039dc0b5de227ce4b7b5a229c55f1e74`、PASS_FOR_SEPARATE_SP1_AUTHORIZATION |
| 承認待ち準備P | `c2a49f72eaffba4514743689eb8ccee02ca060a8`、保存したまま実行HEADへ使わない |
| 実行HEAD A | `9630e06122172af238ac5219bec6dfc8b01ca837`、Sの直接子、JSON/receiptだけの差分 |
| contract SHA256 | `f95931dbde2cfbba23c7585e851a26e3f3217d5820343f88b155b9a3a45093d1` |
| authorization SHA256 | `cd86fb271d43239f3b072a4822bc5f05f049e2de3fa5adf04e04518dbe85fd8d` |
| raw result SHA256 | `9a874f027b2927dcfde44cfb3b3e08a577cdf4b71aa16760a9d58c46a9464c48` |
| consumed marker SHA256 | `fd1064efdacde90953c73eb0092e6f3bd22ab283457bfe5cf63e862986f101f8` |
| SP-0.5固定primitive入力 | result commit `e57c1fdd28589422e9c973e34e53f6c725b921e9`、元JSON SHA256 `d92c3a3d618219f3f46c3cc8143a49b085772e3d1943036511bf39c792dfb759` |
| tool identity SHA256 | `803d902a18d925a9565749fbc64422dca4979bfa6c0bd8b438b28de18d3aa984` |

branch `track-b-sp1-one-shot-execution-20261006`、
worktree `/home/abe/Project/prt-worktrees/track-b-sp1-one-shot-execution-20261006`。
Sから別worktreeを作り、必要なPのJSON/receipt二pathだけをgit blobのSHAとともに参照した。
明示指示原文・原文hashは[Aのreceipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9630e06122172af238ac5219bec6dfc8b01ca837/docs/tracks/algorithm_codesign/sp1_execution_authorization_receipt.md)。
Aをcommit/pushし、clean HEAD・親[S]・二path差分・23 critical identities・既存B runtime・fresh outputを照合してrun1。
実行後は同じrunを再呼出しせず、source/contract/tool/test・原result・markerを保持した。

[raw result / marker / audit / evidence manifest](../../../artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/)を参照する。
元のsource/preparation manifestやstatic planの未認可表示は、その時点の履歴。
実行の効力はA authorization・actual HEAD・markerの一致で判定する。
これは固定sourceに紐づく**local execution evidence**であり、immutable CIや外部再現ではない。

## modelとclaim scope

modelは2-qubit synthetic coherent-signal wrapper、ancilla |+⟩、system |0⟩。
分子Hamiltonian、geometry、basis、DF rank/split L_D、PF delta窓は適用外。
A/B/Cの列・符号・role・n・maskは[固定Sの契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)通り。
mask間で同じ理想νを使い、異なるtemplate/nのν同士を競わせない。
Cのσ=±1 coinはwrapper単位toy distributionで、actual finite RTEではない。
D/R populationはgate数・angle・sign・outer randomnessがconfoundする。
SP-0.5で開封済みのprimitiveを使ったdevelopment/mechanism pilotで、新held-outではない。

primaryは**fusion-normalized additive synthesized-primitive T cost**の期待総費用G_T_add。
exact fusion後の保存primitive T/T†数を加算し、Bernstein sufficient shotsでRe/Imを戻す。
primitive境界を越えたsimplification、actual compiled wrapper費用、最小必要shot数、実測quantum費用とは異なる。
complex ε=.05、familywise α=.05、各軸α=1/1920、native operator error10^-6、
一つのexact kπ/4 catalogue、5% NONE比、shot cap1e9/axisを維持する。
oracle signalはshot削減・順位・primaryに使わず、保存point diagnosticsだけ。

## 保存結果

表は同wrapper NONE比G_T_add。表示値は保存70桁intervalの丸めで、分類は元intervalに基づく。
capはshot cap超過で、G/ratioをnullとして不適格にした。
適格rowのreferenceはすべて正costのNONE。zero-cost referenceは今回のdomainにない。

| n | A：D（DRと同じ） | B：D（DRと同じ） |
|---|---:|---:|
| 8 | 0.05952139：gain | 0.01748891：gain |
| 16 | 0.63926228：gain | 0.08961841：gain |
| 32 | 74.321369：loss | 2.3716502：loss |
| 64 | shot cap | shot cap |

A/BのR=NONE、DR=Dは全保存値で一致した。duplicateを独立positiveにしない。
A/Bはn=16から32の登録点間でgain→lossへ反転した。
未登録nでの精密crossover位置を推定・追加計算していない。

| Cのn | D-only | R-only | DR |
|---|---:|---:|---:|
| 8 | 0.99858037：no material separation | 1.0258429：no material separation | 0.01856084：gain |
| 16 | 1.3194975：loss | 4.2817191：loss | 0.10260741：gain |
| 32 | 2.3065429：loss | 75.369137：loss | 3.1606213：loss |
| 64 | 7.0805204：loss | shot cap | shot cap |

**CでD-only/R-onlyのmaterial gainは0。DRだけがn=8/16でgain、n=32でloss。**
事前reviewで挙げた「単独placementがgain、DRが不利」というpatternのwitnessは今回の登録setにない。
CのDRもgain→loss反転を示したが、crossoverそのものは既知の機構。
この観測だけで新規性、DF優位、一般D/R placement則を採択しない。

| 保存classification | rows |
|---|---:|
| BASELINE | 12 |
| MATERIAL_GAIN | 10 |
| NO_MATERIAL_SEPARATION | 10 |
| MATERIAL_LOSS | 10 |
| SHOT_CAP_EXCEEDED | 6 |

raw gain10のうち4はA/Bのduplicate DR。独立gainは6（A-D2、B-D2、C-DR2）。
適格42 rows／84 axes、shot cap6 rows／12 axes。
bias-budget exhaustion、numeric-inconclusive分類は0。model shot capとtechnical resource capを混同しない。

### 保存会計の機構例：C, n=16

同じν・同じconfidence規則で、保存値は以下の通り。

| mask | E[C_T_add]（表示丸め） | V₂（表示丸め） | sufficient shots/axis | NONE比 |
|---|---:|---:|---:|---:|
| NONE | 2160 | 1 | 13,523 | 1 |
| D | 1633.071 | 1.755142 | 23,601 | 1.319498 |
| R | 534.2374 | 17.62027 | 234,105 | 4.281719 |
| DR | 7.308022 | 30.92607 | 410,115 | 0.1026074 |

この表は保存値の記述。通常合成を残すsingle-maskと、全primitiveをcheap notch化するDRで、
per-shot費用とmoment/shot burdenの交換が異なる。新しいobjectiveやsamplerを設計した証拠ではない。
各pathのp/b/Γ/conditional E[C]/E[W²C]、各軸bias/u/range/marginはraw resultに保存されている。

## 資源・diagnostics・監査

観測resource snapshotはwall6.971782 s / CPU6.93838 s / peak RSS286,564 KiB（約279.85 MiB）/1 process。
scopeはモデルと最初のserialization後、最後のserialization/persistence前。
source内の最後のcap checkも通過した。raw result＋markerは1,385,486 bytes、出力上限8MiB内。
wall1200/CPU900/RSS1GiBのtechnical cap超過、timeout、memory/output/numeric failureの記録なし。
quantum shotsは実際には生成しておらず、sufficient countはresource model上の値。

保存point residual最大は約3.03253×10^-6、trace residual最大約1.57627×10^-77。
全rowが保存synthesis＋coefficient bias guardとtrace guardに整合した。
pointはmpmath 80 dps診断であり、独立のfinite-time certificateとは呼ばない。
operator/channel errorとtrue coefficient区間は元SP-0.5の固定guardを前提とする。

[audit_sp1_saved_result.py](../../../scripts/tracks/algorithm_codesign/audit_sp1_saved_result.py)はstdlibだけで
source/A/contract/tool/input/receipt/markerのidentity、全48/96/64 path-mask inventory、
13保存sequenceと32 specのhash/count/canonical probability、lineage/placement、
保存intervalの算術整合、分類predicate、duplicate除外、cap/STOPを照合した。
science module/runnerをimportせず、matrix/guard/coefficients/shot rule/resourceを再評価しない。
raw result/markerは前後のhashが同一。[監査report](../../../artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/saved_field_audit_v1.json)はPASS。
事前59 focused testsはSのlocal passとして保持し、今回はfull suiteや追加science testsを実行していない。

新synthesis、trajectory、分子NPZ/DF/Hamiltonian、GPU、whole-wrapper compileは0。
SP-0.5一回結果/marker、B-F/BM closures、旧BM-1/16-cell未実行、A証拠/共有APIを保持する。
結果保存後の作業は保存値監査・文書・公開だけ。**mandatory STOPを維持し、GPT側の研究再評価へ戻す。**
