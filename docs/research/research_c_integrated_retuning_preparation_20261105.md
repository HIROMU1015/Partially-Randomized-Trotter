# 研究C：公平retuningと下流資源評価の統合準備（担当B）

2026-11-05 JST。技術準備・人工/mock検証のみ。新しい実H4配列解析、PF/RTE、shots、native build/compile、実H4統計は実行していない。本書は未承認統合科学契約の下流設計であり、研究方向の採択・新科学実行許可ではない。公開照合後mandatory STOP。

## 固定された出発点と再利用

主要保存結果は [beb8de5 の共通r32報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/docs/research/research_c_near_practical_r32_downstream_execution_20261103.md)。科学SOURCE `4df8eca29c59ffd55e2dd898f05da16bb246ddbe`、実装SOURCE `703938aba8cc38080e29e75acde8e0edc8dce868` と、結果収録commit beb8de5 を区別する。

H4直鎖・隣接1.00 Å・STO-3G、number-CDF8、Nα=Nβ=2/sector36、元順LD2、partial ordered S2/s1/r32/K0、δ=h=.1、m0..12、own-DF ground oracle・状態準備除外、P0/serial会計で、有限16受理U更新endpointのPR/ENTRY総RZ比は保存値 `0.494181551983`。物理sector誤差と元engineering guardはそれぞれ候補固有の保存資格で、U一次収束は未達である。この一点を最適化条件・実行政策・移送先に無条件に移さない。

別系列 update005（`938340…f6e56e`）の [r再調整](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f200e45afecc0a80b9f8a895b68c75ba4dda5dd5/docs/research/research_c_rte_r_retuning_execution_20261103.md) は r16/32/64 の新96nativeと保存r128で、r32が条件付き総RZの最小点。これはPR/ENTRY両endpointをretuneした結果ではない。update064（`53d8f5…6ed3`）の独立256校正による約4.48%の同J/U強D差とも別候補である。旧cohortを新候補へpoolせず、旧STOP/残callを継承しない。

| 処理 | 再利用する実API | 適合性・必要な接続差分 |
|---|---|---|
| 独立物理sectorと候補J/U | `c_near_practical_r32.prepare_arm` の既存binding、`c_pr_aware_df_stage_b_runner` のload/ground、`SectorCAR` | 今回の候補固有h/g/constant・J/U・sector export・DF PASSを上流から受け取る。旧endpoint hardcodeのrunner自体を新候補に流用しない |
| Partial PF/RTE | `c_pr_aware_df_stage_b.kernels`、`reference.reference_round`、`c_pr_aware_df_m4.corrected_kernel/rte_round` | r32で候補固有PFとexact-H参照を一度取得。r16/64は同じPFとPF誤差を共有しRTEだけ変更 |
| Signed law/identity | `stage_b.configure/tail_gate`、`c_rc2_pr_stage_b.distribution_gate`、`c_rte_r_retuning.normalization` | 各rのτ・normalization・event数を同じ候補に束縛。identity位相、元g補正κ、nuclear scalar一回を維持 |
| RPE・shots | `c_rte_r_retuning.assess_lanes` → Stage C `qualification/assess_round/energy` | 同じCA/RPE/failure政策、各候補・各r・各roundのPF/RTE/Aを使う。新shot式を作らない |
| Native/P0/融合 | `c_pr_integrated_performance.native`、`StrongBuilder`、`compile_checked` | native32をselectionで1batch、finalで2batch。全512 raw/compiled/actual-adjoint/X-Y等の既存検査を維持 |
| 会計・不確かさ | Stage C `cost_calibration`、M3 `serial_cost/calibration_uncertainty`、`aggregate_totals` | `Ctotal=W mean_kernel+Sshell` の既存実装。N32 hardcodeをdefaultのまま保持し、explicit expected_n=64へ最小拡張。rも校正NもWに重複乗算しない |
| 同J/U強D | `c_update064_strong_d.evaluate` | 名称は歴史的だが実値は渡されたA-packetのJ/U・own ground・input identityから取得。全frame S2/s1、D独自PF/nativeを評価 |
| 価格統計 | `c_near_r32_statistics.primary/arm_summary`、`c_r32_price_statistics.price_statistics` | primary/arm_summaryへ任意固定Nとpolicyのkwargsのみ追加。既存32/95% defaultを保つ |

実装の取込み元は原則beb8de5で、元SOURCEとのbyte同一性と今回の拡張を統合担当が固定する。収集版 `546bca7…` は一部の報告・artifactを持つが、`c_near_practical_r32.py` と `c_near_r32_statistics.py` が未収録で、shared統計codeも古かった。ファイル名だけで同一SOURCEと判断せず、beb8de5の実Git objectを確認した。原固定branchは変更していない。

## 固定比較と公平retuning

retuneするのは **r={16,32,64}のみ**。両手法へ同じ候補集合、各r selection native32、同じcompilerと数値資格を与える。集合は既存s1/h=.1/K0の実施済み隣接power-of-twoから選び、r32の両側に対する減衰/単価の交換を調べる。r128の過去高費用は参考資料に残し、新候補へその価格・資格を転用しない。r8/128追加、K/s/LD/PF/law/価格surrogate/compiler変更は本設計に含めない。

CA=.00159 Ha、DF reserve=CA/8=.00019875 Ha、βtotal=π/6、βstat=π/12、failure=.01、αaxis=.01/26、m0..12。P0はQiskit1.3.0/opt1/seed17、rz/sx/x/cx、approximation1、qubits_initially_zero=false、MPFR160 declared phase、phase fitなし。component/adjoint/operator allowance1e-10、complex signal2e-10、rtol0を保つ。移送先でも同じ政策を採用する提案で、物理的適格性を新たに確認することが必要である。

Head内部relative Gaussian融合とtailの厳密basis_key共有を両armへ同様に適用。cross-head/tail・近似U共有・未実行の全長再compile削減は仮定しない。Kernel単価とshots/W/shellを別列で保存する。RZ depth、CX count/depth、total depth、sizeは同じfenced serial会計で保存し、RZの勝敗を全指標支配へ置き換えない。

Developmentは3start×3strength×2budget×2arm=36endpointを、同じ3rでselection評価する。各start/budget/armについて、固定r32は3strength中のselection総RZ最小、retunedは3strength×3r中の最小を別々に選ぶ。raw objective・λの大小で総資源候補を選ばない。物理/round資格がない候補はnullのまま保持し、missing/correctness FAILを都合よく除外して選択を完了させない。数値完全同値のtieはstrengthの事前順→r32,r16,r64の順とする。

Transferはdevelopmentで凍結されたmethod別strength政策を使用するため12endpoint。各armに同じ3r機会を与え、r32とtunedを独立に残す。移送先の結果でstrengthやrecipeを変更しない。上流の初期値・有限予算政策と結合した正本は統合契約を優先する。

## Selection/finalの独立性と統計

Selectionは各candidate/rで固定32件。候補・r・strengthの選択を完了・保存してから、選ばれた固定r32およびtunedの(candidate SHA,r)ペアに **独立final64件=2×native32** を生成する。選択データ、旧32/256/640件をfinalにpoolしない。両estimandのペアが完全一致する場合だけfinal一cohortを共有する。同じrでもcandidateが違えば別cohortである。共有時に二つの独立replicationと数えない。

新master2026110501、cohortは geometry/start/strength/budget/candidate identity、selection/final、r、batch を識別し、既存seed APIがSOURCE digestとcandidate/axis/sampleも束縛する。最終統合preflightで全予定seedの重複をhashで確認する。これはPRNG独立性の数学的証明ではない。Final結果を見て選択変更・N追加・外れ値除去・途中CI打切りをしない。

N64は広い頑健性gridに対し、既存PR/ENTRY約50%差を追跡する費用と精度の折衷。既存4.48%対強DのN256/1%目標を全endpointへ機械的に適用しない。N32より標本数は増すが、変動未知なので精度・符号の成功保証ではない。微小差は未解決を許容する。

主推論familyは事前neutral start・budget64における、共通r32/公平retuned×geometry2の計4差。family α=.05をBonferroni等分し各 **98.75%近似Welch差区間**。これはモデル内の多重性調整で、validated coverageではない。主RPE failure .01とは別のαである。残りstart/budgetの頑健性は点比・候補ごとの資格・全列の記述を主とし、全cell同時優位を主張しない。

既存5%半幅/ENTRY精度目標、abs moment skew>2、最大delete-one平均変化が対応する近似半幅の25%超で保留を継承する。ゼロ観測分散はゼロ幅保証を出さない。範囲内でも正規/IID・希少tail・被覆率を認証しない。PR−ENTRY区間のみを主とし、random ENTRY分母の比区間を単なる割算で作らない。同J/U Dは副比較として個別近似区間、全資源列は記述的に保存する。

Evidence laneは全law synthesis/implemented-law信号誤差がnullならshots/totalもnull。Idealはexact native/ideal finite-RTE IIDとfinite-bit価格proxyという条件付き比較。Assumedの各誤差≤.001Aは未検証感度仮定。いずれもactual CA・全law保証・一般的優位・独立新規性へ昇格させない。

## 強baselineと保存結果の再利用条件

最終採用されたcandidateごとに同J/Uの全frame強Dを一回評価し、fixed/tunedに共有する。PRのrが不適格でもDは独立に適格性を調べる。別candidateのD分母は流用しない。

保存CDF8/XDF8の新しい共通ideal lane比較が科学条件一致なら、その固定済み総量を参考基準に引用できる。保存数値を新候補と再ランキング・統計比較する処理は今回行わない。古い強baseline比較 `1aeedd9…` は歴史的CDF8/XDF8、D1/R13/P0で、DF誤差は同じではない。A8や今回新fitと同一候補でも同等最適化予算でもない。公平な独立baseline主張には、h/g/constant/input identity、geometry、sector、rank/codec、元κ/scalar、目標/round/shot政策、P0/融合/serial scope、DF資格が一致することを固定JSON/SOURCEで確認する。不一致なら「参考値」と明記する。H4/1.00 ÅのCDF/XDF価格を1.15 Åへ移送しない。

## 科学logical-call積算と実行時間の見通し

以下は新科学実行の未承認上限を設計するための静的積算。量子測定回数・内部BLAS/optimizer iterationとは別単位である。

| 単位 | 最大calls | 根拠 |
|---|---:|---|
| candidate準備（r32 law込み） | 17 | 既存near-r32 A。新DF資格取得は上流別枠 |
| candidate r32 PF/RTE/RPE | 90 | kernel1+exact13+reference13+classification13+summary1+qualification3+accuracy39+energy3+shots3+admission1 |
| 同candidate追加r | 80/r | law3+corrected1+reference13+classification13+summary1+qualification3+accuracy39+energy3+shots3+admission1。新PF/exact0 |
| native共通 | 35/candidate | CAR1+raw9+sector1+context1+law1+Gaussian9+relative4+shell load6+assessment3 |
| native32 batch | 289 | sample/build/reference/raw/compile/compiled/adjoint/XY/metricsの9×32+calibration summary1 |
| 一policyの会計 | 33 | cost calibration1+serial26+aggregation3+uncertainty3 |
| final N64追加総校正 | 1/pair | 2batchの64件を既存calibrationへ一度渡す |
| 同J/U強D | 141/candidate | 既存near-r32 Dの全141。候補groundは共有、D PF/native/合成伝播を独立に評価 |
| Final統計wrapper | 6/pair | native rows検査1+affine binding1+個別価格統計2+secondary summary1+arm summary1 |

adapterではlaw3をreference_rが呼ぶため、上流candidate準備は14、最初referenceは93と配分でき、合計17+90は変わらない。初回lawを上流とadapterで二重計上・二重実行しない。

全3rのselection基本費用はcandidate当たり `17+90+2×80+35+3×(289+33)=1268`。final64は採用pair当たり `2×289+1+33=612`。D141は採用candidate当たり一回。Development最大36candidate、24採用pair/24採用candidate、transfer最大12candidate、24採用pair/12採用candidateより、下流本体上限は `48×1268+48×612+36×141=95,316`。共通physical/input/target/seed、selection、主統計、結果summaryの明示単位と、fit/物理評価は統合担当が別途加算する。重複pair/同candidateを使う場合は消費数が減るが未使用枠を新条件へ流用しない。

Final統計wrapper6×48=288、各geometryの共通/selection/comparison/summary32×2=64を加えた下流全体は **95,668 logical units**。これは統合側の明示集計規則で、fit/物理評価とinput初期化を加えた全実験上限は統合契約を参照する。

最大PR native compileはselection `48×3×32=4608`、final `48×64=3072`、計7680、強D最大36。PR QPY最大30720（raw/compiled/actual-adjoint pair）、D QPY最大144。PF/RTE等の回数とclassic optimizer callは別々に保存する。

既存共通r32両arm/64native+D2/1231callsはwall930.38秒・CPU14449.14秒・peakRSS約2.53e6 KiB。旧r再調整3r/96native/1265callsはwall1531.30秒・CPU24431.41秒・peakRSS約2.74e6 KiB。同様の32CPU環境なら下流全gridは概ね数十時間級が計画尺度で、r64、compiler排他、入力/候補cache、共有負荷に依存し、実行時間やspeedupの保証ではない。最適化時間はこの見積りに含めない。実開始前の実資源/予定call計画で再評価し、時間capは置かない。

候補間はprocess分離し、candidate内13round/native32の既存resource_plan並列とQiskit一時global状態のmutexを維持する。Immutable CAR/Gaussian/shell cacheは同じarmで共有。BLAS/OpenMP/process数を協調させ、実CPU affinity/quota/空RAM/GPU/disk/pressureと他jobへの影響から同時候補数を選ぶ。人工CPU/RAM/worker上限、共有環境・他user jobの変更は行わない。

## 実装・人工検証・残る実行境界

新 [選択helper](../../src/trottertracks/c_integrated_retuning_policy.py) は既存会計済みselection総RZを選ぶだけで、3r機会、同candidate、selection-only、null、correctness、final exact-pairの扱いを固定する。新 [callable adapter](../../src/trottertracks/c_integrated_retuning_adapter.py) は候補を束縛し、既存reference/configure/RPE/native/accounting/強Dの関数を直接呼ぶ。optimizer、Hamiltonian、compiler、shot式、科学会計は再実装していない。新 [mock検証](../../tests/management/test_research_c_integrated_retuning_adapter.py) でr32→r16/64のPF共有、39round dispatch、全3lane、selection1/final2batch、同J/U D、fresh output、統計N64/98.75%と旧defaultを確認する。

`prepare_candidate(candidate_path, reference_path, physical_record, common, out, call)` が新候補のbytes/JU/DF/sector/nine-factor/groundを既存関数で束縛し、downstream A-packetを構成する。共通準備では元h/g/constantを既存loaderで読み、`SectorCAR(4).physical`、`c_near_practical_r32.ground`、`AccuracyTarget(**m4.TARGET)` を用い、alias資格・canonical sector basisを確認して `common={p,input_identity,ctx,basis,target,shells}` を作る。物理DF資格段階から保存する独立referenceは `J,U,generators,H` を含む。新移送inputに旧H4物理入力を代入しない。

これらは統合実行runner全体ではない。上流の固定input生成/初期化/fit/物理DF資格、全予定候補とSOURCE/seedの固定、strength×rの事前選択、family4主比較、fresh output/STOPとcall dispatchは統合契約・統合adapterに接続する。科学許可後に旧hardcoded run()や旧停止jobを再開する経路は使わない。

新helper8＋adapter/stat11＝**19人工/mock PASS**。既存 `test_research_c_rte_r_retuning.py` の18人工/mock回帰もPASS（計37）。N64の実会計APIにも人工64行を通し、default32での拒否、explicit64での受理、shared-SEのWのみの倍率を検証した。最終統合時は同じSOURCEで必要な回帰だけ確認する。実H4数値性能・価格CI被覆率を検証した意味ではない。実行環境は既存専用venvを読み取り使用し、共有packageの更新はしていない。

Correctness/input/source/候補binding FAILは依存仕事を停止しpartialを保全する。物理/alias/round未資格はそのnullを保持して、比較不成立や符号不定を報告する。単なる小差・精度未達で同じ科学条件の未登録標本を追加しない。r32先行はPF共有の依存であり中間承認要求ではない。最終報告・公開照合後mandatory STOP、retry/resume0。科学実行・重要reviewはそれぞれ別途ユーザー明示許可が必要で、承認証明・grantの追加は不要である。
