# 研究C：PR局所方向・3段階対称診断の限定科学実行

2026-11-02 JST。固定科学SOURCE `e06913a234999d9f42df8a5aabd917f9510a2ac7` を変更せず、fresh `artifacts/research_c_pr_local_step/2026-11-02/run01` で一度実行した。**49/49 COMPLETE、correctness FAIL 0、retry/resume 0。全7点が登録数値資格を満たし、3段階の正方向probeすべてでFの経験的減少を観測した。** U更新受理・外側最適化は0で、公開照合後mandatory STOP。

## 1. 固定条件と実行範囲

保存直鎖H4・隣接1.00Å・STO-3G、number CDF8・元frame順・LD=2、Nα=Nβ=2/sector36の入力規約。元qualified J/Uのeta=0から開始し、γ=.1、ε=1e-8Ha、f*=2.1839632053132894e-9、M*=2.2093719370722074を維持した。元g補正κ・scalar・spin順を変更せず、今回はそれらのoperatorを評価しない。

[固定49-call契約](research_c_pr_local_direction_contract_unapproved_20261102.json)（scope `C_PR_LOCAL_DIRECTION_SYMMETRIC_DIAGNOSIS_V1`）と[技術準備](research_c_pr_local_step_accuracy_preparation_20261102.md)に従う。契約の歴史的なunapproved metadataは変更していない。契約SHA256は `02f6c7aa275ffe23f94e259a06c770190ab05be6299c18fe01f4c57a3fd6a51c`、SOURCE manifest SHA256は `8b22d88f22620879c45f1b3d78e377e2dfe471ce13377291f6935e249c577925`。

既存の同一decoded J評価器、Newton-CG/Hessian、独立J/gradient検査、2回refinement、rank/nullspace・latest-pair stability、guarded Oracle、Calls・raw/STOP保存を再利用した。新しいfitter・solver、旧60-callの第9trial、RC2継続を開始していない。

## 2. 局所Fと方向微分の観測

[全差分データ](../../artifacts/research_c_pr_local_step/2026-11-02/run01/directional_diagnostics.json)。初期F0=`0.012267644429815034`、報告候補方向微分s=`-8.731877899141525`、d=-g0、αF=F0/(-s)=`0.001404926245134638`。登録順にq=1/8,1/32,1/128、各h=αF qのplus、minusを評価し、各点で元Jからsolve＋2 refinementを実施した。

Fは正規化tensor lossとPR penaltyの自己目的関数で、エネルギー誤差やDF sector誤差ではない。

| q | h | F(+h) | F(-h) | F(+h)-F0 | 正方向の観測 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1/8 | 1.7561578064182975e-4 | .010792467936921136 | .013856205454742273 | -1.4751764928938975e-3 | EMPIRICAL_DECREASE |
| 1/32 | 4.390394516045744e-5 | .011887576525259309 | .012654315697712786 | -3.80067904555725e-4 | EMPIRICAL_DECREASE |
| 1/128 | 1.097598629011436e-5 | .012172009722815336 | .012363691845003593 | -9.563470699969732e-5 | EMPIRICAL_DECREASE |

各forward比較尺度は`2.2267644429815037e-10`で、上記の減少はその尺度を超える。すべて未受理の診断点であり、最小Fの点を自動採用していない。Exportは元Uにおけるrefined initial J/Uだけで、新しいDF予算適格候補を作ったとは扱わない。

| q | Forward FD | Backward FD | Central FD | Central FD − s | FD sensitivity scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1/8 | -8.400022409731706 | -9.045662178657661 | -8.722842294194685 | +9.035604946840081e-3 | 6.792158816125203e-7 |
| 1/32 | -8.656805286328511 | -8.807210069267583 | -8.732007677798046 | -1.297786565217507e-4 | 2.5799863332232658e-6 |
| 1/128 | -8.71308550064715 | -8.75068651234242 | -8.731886006494785 | -8.10735325984524e-6 | 1.0187554564069424e-5 |

Central FDの隣接level差は`0.009165383603361832`、`0.00012167130326190545`。各pairで差分信号は登録比較尺度より大きい。最小hではFDとsの差がFD sensitivity scale内に入った。**この尺度は工程的な感度尺度であり、真のreduced gradientの許容誤差や厳密enclosureではない。** 元gradient許容差を緩めてFDをPASS認定していない。

旧60-callの最小α=.002670284759424921ではF=.020543518021918193で初期Fより高かったが、今回のより小さい3尺度ではすべて正方向のF低下を観測した。この初期点・方向・固定inner手順の範囲で、旧探索のstep幅だけでは捉えなかった減少が存在したという数値結果である。非線形曲率、inner bias、差分truncationの寄与を一意に分離した結果ではない。

## 3. Correctness・J資格・stabilityとraw status

[実行集計](../../artifacts/research_c_pr_local_step/2026-11-02/execution_summary.json)／[全trace](../../artifacts/research_c_pr_local_step/2026-11-02/run01/diagnostic_trace.jsonl)／[結果](../../artifacts/research_c_pr_local_step/2026-11-02/run01/result.json)。点0=initial、1/2=level0±、3/4=level1±、5/6=level2±。

| 点 | 最終独立J∞ | partial U₂ | latest-pair U shift | Raw status（solve/refine1/refine2） | 数値資格 |
| --- | ---: | ---: | ---: | --- | --- |
| 0 | 7.600391912793852e-8 | 2.9549751184186754 | 4.2369251510155467e-10 | 0/2/0 | PASS |
| 1 | 1.222707730341721e-7 | 2.669694288051786 | 2.7856999425731587e-10 | 0/0/0 | PASS |
| 2 | 6.939303045030987e-8 | 3.2717736954096517 | 7.157705509223168e-10 | 0/0/0 | PASS |
| 3 | 8.294264836704637e-8 | 2.904390768117925 | 1.0380244387962835e-9 | 0/0/0 | PASS |
| 4 | 9.343209633402849e-8 | 3.006196704105615 | 5.675746330626997e-10 | 0/0/0 | PASS |
| 5 | 1.495025196514072e-7 | 2.9422703127096828 | 1.6786822477783292e-10 | 2/0/0 | PASS |
| 6 | 7.501386493566353e-8 | 2.967719731239359 | 1.676051613548671e-10 | 0/2/0 | PASS |

全7点で最終inner/独立J、rank/nullspace、full numeric rank、latest-pair stability、block/refinement nonincreaseがPASS。PR head quotient rank19・universal nullity1・tail injectivityを元のguardで判定した。独立Jは元1e-5＋margin、stabilityは元1e-6で判定し、閾値変更0。全21 innerのJ基準もMETだった。

Raw Newton-CGはstatus0が18件、status2 precision lossが3件。すべて正常収束したとは記載しない。Chart/physical-chain gradient比較は全21件PASS、最大絶対差`2.8294832009973637e-19`。Raw jacは元のpre-step OR endpoint規約を維持し、endpoint単独不一致の観測を削除していない。Subcheck全件PASSという主張ではない。

Partial U一次条件は全7点NOT_METで、J/U同時一次収束は未達。状態`LOCAL_DIRECTIONAL_DIAGNOSTICS_AVAILABLE`は3組のqualified pairが揃ったという意味だけで、真のreduced微分精度の認定ではない。

## 4. Call・資源・証拠

[Science call journal](../../artifacts/research_c_pr_local_step/2026-11-02/run01/science_calls.jsonl)／[STOP](../../artifacts/research_c_pr_local_step/2026-11-02/run01/STOP.json)。

| Call種別 | 実績 |
| --- | ---: |
| saved_load / initial_identity / objective_setup | 各1＝3 |
| local_probe_driver | 1 |
| inner_J_solve | 21 |
| precision_state | 14 |
| reduced_admission | 7 |
| directional_diagnostics | 1 |
| classification / export | 各1＝2 |
| **合計** | **49/49 COMPLETE、FAIL 0** |

7 distinct U点・6 signed probes・3 pairs、inner21（refinement14）。Raw nit合計4142、nfev/njev7353、nhev4145、実VG7416/Hessian4145/independent adjoint42/binary64 same-array21。外側optimizer・BFGS・U受理は0。`directional_diagnostics.optimizer_iterations=0`は外側診断driverにoptimizerがない意味であり、inner Newton-CGの反復0を意味しない。新ffsim fit・RC2継続・retry/resumeも0。

[Preflight](../../artifacts/research_c_pr_local_step/2026-11-02/execution_preflight.json)でSOURCE165/input4・依存lock/ffsim SOURCE/version・fresh outputを確認。準備時の269 unique人工/回帰PASSを対応する不変SOURCEの技術証拠として参照し、今回再実行やCI/外部再現を行ったとは主張しない。実行時affinity32、load約1.25、MemAvailable約54.8GB、CPU/memory pressure avg10=0。逐次依存するprobe/refinement順を維持し、既存vectorized NumPy/SciPy/JAX/longdouble・小行列BLAS調整・task-local cacheを使用した。任意のCPU/RAM/worker/wall上限や共有環境変更を追加しない。

[資源実績](../../artifacts/research_c_pr_local_step/2026-11-02/run01/resources.json)：wall18.437584秒、process CPU84.135707秒、peak RSS453716KiB。これらはrunner開始以降の実績で、環境確認・publication時間を含まず、speedupの主張でもない。

[保全postflight](../../artifacts/research_c_pr_local_step/2026-11-02/execution_postflight.json)で旧raw2495と旧SOURCE17組＋今回165件、旧STOP/契約、準備技術証拠を照合する。新raw59件を保存した。保存配列の追加decode・objective/gradient問い合わせはrunner終了後0で、集計は計算済みJSON/JSONLのみを使用した。新結果1件をmanifestへ追加し旧135件を保持、当日noteは追記のみ。

## 5. 取得できたこと・未資格事項・次の担当

今回取得したのは、元初期点・候補方向・登録6側点におけるF低下と差分の数値観測、およびinner/J/stabilityの工程的数値資格である。未知inner解誤差、FD truncation上界、真のreduced gradient誤差、厳密な降下保証はnullを維持する。FD一致やrefinement一致をその代わりに使わない。

新しいDF物理予算、PF/RTE/ground/native/shot/CA/総資源は未評価。DF予算CA/8=.00019875Haは変更せず、既存の初期DF PASSをprobeへ転用しない。RC2・Joint方式・paired最適化の継続や方式の有効性に関する新しい結論も出さない。

次は**ユーザー／研究C**が、今回観測した小stepを用いる限定的なU更新試験の技術準備へ進むか判断する。採択されれば**Codex**が、既存inner数値資格と値比較の不確実性を保ったstep初期化・受理規則・停止条件を人工検証し、新しい固定範囲を準備する候補である。今回の診断点を自動受理せず、追加実H4・refinement・optimizer・重要科学レビューは実施しない。

Branch `codex/research-c-pr-local-direction-execution-20261102`。結果と不変SOURCEのprovenanceをcommit/pushし、remote SHA/tree・公開成果物bytesを照合する。**公開照合後mandatory STOP。**
