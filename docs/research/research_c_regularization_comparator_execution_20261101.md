# 研究C：RC-DF型uniform L2対照・Stage A限定1-fit結果

2026-11-01 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。

**新規1 fit、17/17 logical units COMPLETE、correctness FAIL0、retry/resume0で完了した。** Optimizer success=trueとDF予算PASSを独立に確認し、結果はSTAGE_A_ELIGIBLE_FOR_RESEARCH_DECISION。RC2のexact tail λは2.209761871954849 Haで、保存PR-tail γ=.1の1.4424298855756243 Haより大きかった。これは固定初期寄与・共通optimizer下のproxy比較で、PR資源の追加価値や原論文RC-DF全体への優位を示す結果ではない。

## 固定SOURCE・契約・条件

| 項目 | 固定条件・来歴 |
|---|---|
| 科学実行SOURCE | [150f904f6975868abe09966dda8ab41722e2a7cc](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/150f904f6975868abe09966dda8ab41722e2a7cc)、無変更で一度だけ実行 |
| 数値実装SOURCE | 25e4c3c8e9a2a949fd4f626f502089ca2d8c6332 |
| [固定契約](research_c_regularization_comparator_contract_unapproved_20261101.json) SHA256 | 9b57ae71e79a6c6d7eb45036b53f28307d6b2c2957bc6d5fa60599935ab58cc3 |
| [SOURCE36](../../artifacts/research_c_regularization_comparator/2026-11-01/technical/SOURCE.json) SHA256 | 4074644f94b8b4af64601e86a87bb5114bcd0735d3d584bbd0e8e7b9b3e901ec |
| 分子・sector | 保存直鎖H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、sector36 |
| DF・規約 | 通常number CDF rank8、一般対称J、元g補正κ、nuclear scalar一回、元frame順 |
| 初期値 | 保存Stage Aと同じqualified J/U/x0。再初期化・再logmなし。元fit科学SOURCEはe1085c95f5c01b68566e17d543ace5832cb94061 |
| 対照目的 | 全frame・ordered空間Z=Jへのuniform L2、f/f*＋αP/P0、α=.1 |
| PR proxy | 元順LD2、非identity Z/ZZ exact tail mass。今回PF・δ evolution・RTEは非適用/未実行 |
| Solver | ffsim0.0.83、既存real_schur_column_gauge_v1、L-BFGS-B、options=None。tol1e-8はffsim初期分解設定でSciPy tol上書きではない |
| 物理予算 | sector norm＋sector_operation_scale_guard8_proposal_v1 ≤ CA/8=.00019875 Ha |
| Fresh output | [run01原本22件](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/) |

[技術設計](research_c_regularization_comparator_design_20261101.md)を変更せず、今回許可されたStage Aを実行した。契約の歴史的なUNAPPROVED_PROPOSAL metadataはそのまま保全した。承認証明・承認原文保存・新grantを作成していない。

[Preflight](../../artifacts/research_c_regularization_comparator/2026-11-01/execution_preflight.json)はSOURCE36/input15、契約hash、両依存環境、未使用run01を照合してPASS。実行worktreeで人工22件も再確認PASSした。[実行時人工log](../../artifacts/research_c_regularization_comparator/2026-11-01/execution_synthetic_tests.log)。準備時の171 PASS＋29 subtestsを、実行時にも全件再実行したとは扱わない。

## 全gate・科学call

| Call/gate | Units | 結果 |
|---|---:|---|
| Saved snapshots load | 1 | PASS：保存h/g/c、qualified J/U/x0、scale |
| Initial tensor | 1 | COMPLETE：独立tensor再構成 |
| Sector context / physical sector | 1＋1 | COMPLETE：sector36、元gのHamiltonian一回 |
| Initial/final sector | 2 | 両DF予算PASS、有限性/Hermiticityのcorrectness FAILなし |
| Initial/final general-K対raw CAR | 2 | PASS、最大entry誤差2.761679773755077e-15／2.886579864025407e-15、allowance1e-10 |
| Initial gate | 1 | PASS：optimizer前の予算/表現 |
| Fit input / objective setup | 1＋1 | PASS：同x0/既存f*、P0を一回固定 |
| Existing fitter | 1 | 一回のみ、corrected prepared-x0経路の検査を通過 |
| Initial/final objective-gradient | 2 | COMPLETE：export一致と有限性PASS |
| Final tensor | 1 | COMPLETE |
| Final snapshot/scalar load / comparison | 1＋1 | COMPLETE、STAGE_A_ELIGIBLE_FOR_RESEARCH_DECISION |
| **合計** | **17** | **17 attempts＝17 COMPLETE、FAIL/inflight0** |

親11・fit process6の[call journal](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/science_calls.jsonl)／[fit journal](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/fit/science_calls.jsonl)と[STOP](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/STOP.json)を保存した。新fit1、retry/resume0。新積分・新初期分解・旧γ0/.1の再fit/DF norm再評価、ground/alias、PF/RTE、sampling、native build/compile、shots、総費用は0。

## 強度・optimizer・初期/終了値

P0=Σ_tij J0²=4.410884790420646 Ha²、f*=2.1839632053132894e-9 Ha²、α=.1から、raw ρ=4.951304123962429e-11を登録objective_setupで得た。[Objective原本](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/fit/objective.json)。ρ=.1を直接使ったものではなく、初期penalty寄与を保存PR目的と揃えた比較である。結果を見たρの変更は行っていない。

| 指標 | 初期 | 終了RC2 |
|---|---:|---:|
| Combined objective | 1.1 | 0.10000199116572639 |
| ffsim raw tensor objective | 2.1839632053132894e-9 | 1.2213060585293254e-17 |
| 独立tensor Frobenius | 6.609028983603955e-5 | 4.942278943464885e-9 |
| 独立tensor max-entry | 1.0550030930422949e-5 | 9.513901700196463e-10 |
| Physical sector operator norm [Ha] | 7.666027761495033e-5 | 5.8389056735133376e-9 |
| Numerical guard [Ha] | 3.278760734406151e-9 | 3.2787811814051313e-9 |
| DF予算 | PASS | PASS |
| Exact tail λ [Ha] | 2.2093719370656357 | 2.209761871954849 |
| Full ordered J L2 [Ha²] | 4.410884790420646 | 4.410972371783312 |
| Gradient L2 | 32834.03913852919 | 1.5563974379521772 |
| Gradient infinity | 9754.561788218592 | 0.4222446600957083 |

[初期diagnostic](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/fit/initial_diagnostic.json)／[終了diagnostic](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/fit/final_diagnostic.json)／[独立tensor](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/fit/tensor.json)／[初期sector](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/initial_sector.json)／[終了sector](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/final_sector.json)。

Optimizerはsuccess=true、status0、nit97、nfev=njev106、messageはCONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH。[生optimizer metadata](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/fit/optimizer.json)。生funは0.10000199116572644で、上表のdiagnosticとの違いは固定objective export allowance内。

**相対目的減少による停止であり、gradient停止・stationarity・大域最適を示さない。** Combined objectiveは減ったが、Pそのものは初期よりわずかに大きい。したがって「L2を十分に最小化した最強RC-DF対照」とは扱わない。停止政策、正規化、初期値、強度を結果後に変更せずに記録する。

[Final export](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/fit/final.npz)はJ/U各(8,4,4)、raw x/jac各(128,)のfloat64。SHA256：4972174836a3fe33897a1bcdb922645596f45c8e28141ddafcdf1ba5a4a96ee6。初期/終了xとgradientの別exportも保存済み。公開確認ではheader/bytesを照合し、追加のoperator/gradient評価を行わない。

## 保存γ0／PR-tail γ.1との比較

同じ保存直鎖H4/1Å/STO-3G、rank8・sector36・元順LD2・qualified x0、元g補正κ/scalarに対する比較。旧2行は[元Stage A](research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md)と今回登録のsaved scalar loadから引用し、旧J/Uを数値再評価していない。

| 候補 | Exact tail λ [Ha] | Sector norm [Ha] | Guard [Ha] | DF予算 | Optimizer |
|---|---:|---:|---:|---|---|
| 保存γ0 | 2.2097618367788057 | 7.018855038843621e-9 | 3.278781198621627e-9 | PASS | success、nit97 |
| 保存PR-tail γ.1 | 1.4424298855756243 | 9.039161828383144e-9 | 3.2564531608037766e-9 | PASS | success、nit1129 |
| 新RC2、α=.1初期寄与一致 | 2.209761871954849 | 5.8389056735133376e-9 | 3.2787811814051313e-9 | PASS | success、nit97 |

[固定comparison出力](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/result.json)のtail_vs_saved_proposedはHIGHER_NUMERICAL_POINT、差は+0.7673319863792247 Ha。これはtail mass差であってenergy誤差ではない。登録1e-10 Haのmass comparison allowance内の同等判定には該当しない。

この固定pilotでは、PR-tail保存候補のtail massがfull L2対照より小さい。一方で最終tensor/sector誤差は一致せず、初期penalty寄与一致はgradient/曲率/最適ρの一致でもない。旧/新combined objectiveは異なる関数なので、その値だけを性能順位に使わない。

**この段階で得たのはproxy差までである。** RC-DFの原論文solver全体、最適にtuneしたρ、別初期値・分子、PF/RTE・A・shots・native価格、PR内部総RZ、強決定論への競争力、新規性・論文化GOは未評価または未判断。既存PRの価格やshotsをRC2へ移送していない。Result内のPF/RTE/A/shots/native_cost/total_cost/CA_achieved/independent_noveltyはすべてnullを維持した。

## 資源・保全・公開

Preflightでaffinity32 CPU、利用可能RAM56,285,118,464 bytes、CPU quotaなしを確認。既存cached JAX value/gradient・CPU routeを利用し、one-fitとprocess-local hookのためfit専用processを一つ用いた。Tiny BLASのみ自process内1 threadでJAX poolと協調し、人工的なCPU/RAM/worker/wall上限は追加していない。

固定jaxlibはCUDA対応buildではないためCPUを使用した。[実行log](../../artifacts/research_c_regularization_comparator/2026-11-01/execution_run01.log)にbackend通知を保存しており、依存環境変更やGPU版での再実行は行っていない。

| 記録 | Wall [s] | Process CPU [s] | Peak RSS [KiB] |
|---|---:|---:|---:|
| Parent | 2.5204127049073577 | 1.519465786 | 276784 |
| Fit process | 0.8778180982917547 | 1.852048637 | 374152 |

Parent wallには子process待ちを含み、CPU/RSSは各processの記録である。RSSをaggregate peakとは呼ばない。[Parent資源](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/resources.json)／[Fit資源](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/fit/resources.json)。

[Postflight](../../artifacts/research_c_regularization_comparator/2026-11-01/execution_postflight.json)でSOURCE36/input15、元契約とrun contract、実行SOURCE、17 attempts/COMPLETE、retry/resume0、raw22件のhash、x/jac header、B/C nullを照合した。旧SOURCE・fit exports・QPY・raw・科学契約・grant・ledger・marker・STOP・共有/既存venv・他jobは変更していない。成果物公開時にもSOURCE/inputと旧原本の不変、remote SHA/tree・公開本文を照合する。

実行済みcommandは次の履歴であり、消費済みrun01の再起動許可ではない。

~~~bash
PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_regularization_comparator.py \
 --contract docs/research/research_c_regularization_comparator_contract_unapproved_20261101.json \
 --output artifacts/research_c_regularization_comparator/2026-11-01/run01 \
 --fit-python /tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python --execute
~~~

**Stage Aを完了し、公開照合後mandatory STOPする。** 次はユーザー/研究Cが、固定pilotでのλ差からStage Bへ進むか、強度/停止政策の比較条件を先に検討するかを判断する。Codexは追加fit・ρ調整・B/C・重要科学レビューを自動開始しない。
