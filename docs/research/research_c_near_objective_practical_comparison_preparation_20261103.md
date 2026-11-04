# 研究C：近接weighted L1とPRの有限手順比較・技術準備

2026-11-03 JST。Repository `HIROMU1015/Partially-Randomized-Trotter`。
**技術GO：Stage Oの実装・人工検証・実行可能な未承認契約まで。実H4科学実行は0、mandatory STOP。**

[判断資料](research_c_comparison_readiness_integration_20261103.md)／
[未承認科学契約](research_c_near_objective_practical_comparison_contract_unapproved_20261103.json)／
[SOURCE](../../artifacts/research_c_near_practical/2026-11-03/technical/SOURCE.json)／
[passive preflight・人工検証記録](../../artifacts/research_c_near_practical/2026-11-03/technical/preflight.json)。
SOURCE実装commitは`9e9fc8391590aeb7f9d36b76d90b5a25bbe2fe5e`。契約hashは上記機械可読資料に固定する。公開commitは最終チャットから参照できる。

## 1. 比較IIと初回の範囲

ユーザーが採用した主比較は「比較II：実用的な有限計算手順」である。
同じ元初期値・表現自由度・物理予算・有限アルゴリズム政策に対して得られるendpointを比較する。
各armの自己Fの大小は勝敗に用いない。有限未収束を消去せず、目的そのものの最適能力を問う比較Iへ拡張しない。

初回の科学案を**Stage O：paired最適化＋独立物理DF判定＋exact mass診断**に限定した。
Bの未固定事項のうち、stable接続・初期値・強度・有限予算・entry・call上限を閉じる。
まだ存在しない新候補にN256/640や旧価格を割り当てることは避け、下流PF/RTE/RPE/nativeは設計だけを記す。
この分離により、solverの実H4での資格と候補のDF適格性を先に判断できる。

## 2. 共通初期値・物理条件

| 項目 | 固定する条件 |
|---|---|
| 系 | 保存直鎖H4、隣接1.00Å、STO-3G、電荷0/singlet、Nα=Nβ=2、sector36 |
| 表現 | 通常number CDF8、一般signed対称J、frame別実直交U、元frame順、LD2。PSD化、frame並替え、rank圧縮なし |
| 元Hamiltonian | 保存`prepare/input.npz`のh/g/nuclear scalar。元P-H4の一電子/chemist ERI/核反発に対応する固定byte identity |
| 中立な共通起点 | PR正則化前のtensor-only CDF8 export（fit SOURCE `da2edb465d5ae6432439fbd1df8e57512ee59c11`）を忠実な修正codecで保存した`prepare/initial.npz`。update005/064は使わない |
| パラメータ | 保存J/U/x0を両armに共通束縛。既存ffsim unpackでx0とJ/Uを確認し、新logm/再fitを行わない。新innerのx0は同じisometric pack(J0)、U局所chartはη=0。保存full-ffsim x0は来歴/一致確認用であり、full fitterへ再投入する方式ではない |
| κ/scalar | 元gからκ=h−0.5 einsum(prrq→pq,g)を固定、核scalar一回、fragment identityを保持 |
| 物理予算 | CA=.00159Ha、DF reserve=.00019875Ha。各候補のsector operator 2-norm＋既存guardを独立判定 |

入力3件のSHA256は契約に記録した。元fit exportと、新しい評価/最適化SOURCEを別に保持する。
元CDF8はtensor-only fitであり、全ての初期化に中立・最良であるという意味ではない。
両armの共通起点として採った一つの既存初期化である。人工準備中はNPZを数値decodeしていない。

## 3. 目的・強度・smoothing

PRは既存`c_pr_aware_df.Objective`、L1はBの`c_near_objective.Objective`をそのまま使う。

`F_PR=f/f*+γ(Mε/M*)²`、`F_ENTRY=f/f*+β(Bη/B*)²`、`f=0.5||g−gDF||F²`。

| 量 | 固定政策 |
|---|---|
| f* | 保存初期tensor loss `2.1839632053132894e-9 Ha²` |
| PR | γ=.1、M*=保存`2.2093719370722074 Ha`、ε=1e-8Ha |
| ENTRY | ordered重みdiag=5/4、offdiag=3/2、η=ε(2n+1)/(1.5n−1/4)。B*=Bη(J0)を将来の許可済みsetupで一度だけ評価 |
| 強度 | **各arm一設定**。自己初期massで正規化するので、β=.1がPRの初期penalty寄与.1と一致する。数値γ/βを物理的に同じと仮定して選んだものではない |
| 探索 | 両armとも追加強度探索0、multistart0、再調整0。代替は同じ有限relative-strength gridだが初回には含めない |
| 限界 | 初期寄与一致は最終DF誤差・曲率・solver負荷・最適強度を一致させない。結果はこの一つの有限recipeの比較 |

Zero floorだけを合わせ、ε/ηによる滑らかさの完全一致を主張しない。
Endpointでexact λ、exact B0、row-cancellation gap、両smooth mass、構造的smoothing上限を保存する。
差がsmoothing感度と同程度なら、相殺機構の利益と即断しない。未知の科学誤差を0と扱わない。

## 4. 接続と再利用

| 既存資産 | 今回の最小差分 |
|---|---|
| B `c_near_objective.py` | **無変更**。mass/重み/η/scale/ffsim AD oracleを再利用。二重実装なし |
| `c_pr_coefficient_point.CoefficientChart` | 明示的ENTRY用の線形smooth-mass map/ε vectorを受けられるよう拡張。元PR map・計算順・閾値は既定のまま |
| `c_pr_inner_point_solver` | 明示backendで係数問題・same-array stateを選ぶ。既存Newton-CG、Hessian、raw pre-step jac意味、独立adjoint、二精度検査をそのまま使用 |
| `c_j_u_precision/state` | オプションのparts evaluatorのみ追加。UのJAX微分、物理接空間、forward/reverse検査、extended accumulationは共有 |
| reduced/guarded oracle | 既存3 solve（初回＋2 refinement）、rank/head gauge、tail injectivity、J一次、latest-pair stability、非増加/admissionを再利用。ENTRYをRC2へ誤dispatchしない |
| 固定`c_u_step_control.py` | `025d4951…`から取り込み、oracle/独立終端のopt-in hookとtotal-point budgetを追加。既定値のPR経路は同じbounded recentered Armijo |
| `c_near_stable.py` | 上記への短いENTRY backend。独立のoptimizerを新設せず、analytic derivativeと既存BのADを人工検証で照合 |
| `c_near_practical.py`/runner | 既存Calls・CAR evaluation child・byte preflight・save/exportを利用するpaired orchestration。新しい承認機構なし |
| `Evaluation`/`SectorCAR` | 元Hamiltonian contextを一回作成し共有。候補ごとのrank_assessment、General-K/raw CAR、tensor誤差、exact λ、元κ/scalarを評価 |

ENTRYのunique-entry線形mapは`c=(5/4 Jii,3 Jij;i<j)`、vector εは対応する重み×η。
`Σhypot(c,ε_vector)=Bη`なので既存smooth-mass Hessian templateを再利用できる。
M自体の値はBのmass関数を用いる。Jをbinary64で一度decodeした同じ配列から両勾配経路の残差を評価する。
PR既定経路は人工点で固定版を別namespaceに読み込み、値/gradient/Hessian/adjointのbitwise一致も検査した。
これは実分子上のbitwise再現保証ではない。

## 5. 共通の有限アルゴリズム政策

各arm最大16受理更新、合計128 trial、各方向最大32 halving、初期点込み最大129新U点。
16は最初のpaired手順検証として複数更新の経過を得る有限pilot設定であり、64更新の旧PR結果を比較対照に流用しない。
128 trialにより棄却が多いarmにも同じ機会を与え、受理更新数だけの予算にしない。
両armとも同じ上限を使い、早期停止後の余りを他方や追加強度へ移さない。

- 各新U点は元cold J0から初回Newton-CG、次にちょうど2 refinement。最大387 inner solve/arm、うち258 refinement。
- 各solveは元のmaxiter1000/xtol1e-12。Gaugeを落とさない既存full-direction metric。契約内の継承Policyのmax_cycles/u_max_nfev等は互換用で、このdriverでは使用しない。実際の上限はstep_policyとmaximum_total_trials/points。
- J∞/物理U接空間L2の一次閾値各1e-5、元の工程margin、stability1e-6、直交性1e-10。
- c1=1e-4、α0=min(1,1.01/||g||)、受理後は直前α×2とのmin、半減.5。
- 元の値比較尺度を含むArmijoの後、独立終端stateと物理DF budgetを通過した点だけを受理。
- Raw status/success、J一次、U一次、stability、rank/nullspace、独立終端、DF予算を別々に記録。Precision lossだけでは資格を与えず、また自動retryもしない。

Nfev/njev/nhev/nit、実VG/Hessian回数、棄却を含むtrial、accepted数、CPU/wallを保存する。
同じアルゴリズム上の機会であり、実際のFLOPsやCPU消費の完全一致ではない。
並列中のraw process CPUは重なるためarm間で単純加算せず、thread CPU/JAX pool除外の限界、全process CPU/wallも併記する。
未収束endpointは残す。厳密なargmin、真のreduced-gradient誤差、同時一次収束の保証とは扱わない。

物理予算は既存`sector_operation_scale_guard8_proposal_v1`の
`8 eps64 (n^4+R n^3)(d+1) max(1,||H||F+Σ||G_t||F)`を維持する。
General-K/raw CARのsector max-entry allowanceは1e-10。
いずれも数値pilot/engineering gateであり、厳密丸めenclosureではない。
Stage Oは新candidateのfull256/native/ground/aliasを検証したとは言わない。

## 6. 科学call：上限1,629

既存の一回のsolve/state/physical assessmentというlogical unit定義を維持する。
内部のAD/LAPACK/係数操作数を一callずつ数える定義ではない。VG等は別に実績を保存する。

| 種別 | 共通 | 各arm上限 | 合計上限 |
|---|---:|---:|---:|
| 保存load・initial identity・physical context・共通初期物理・目的setup・比較・export | 7 | 0 | 7 |
| optimization driver | 0 | 1 | 2 |
| inner J solve | 0 | 3×129=387 | 774 |
| precision state | 0 | 2×129=258 | 516 |
| reduced admission | 0 | 129 | 258 |
| independent terminal（各受理候補＋最終） | 0 | 16+1=17 | 34 |
| physical assessment（refined初期＋各受理候補） | 0 | 16+1=17 | 34 |
| endpoint mass/cancellation診断 | 0 | 1 | 2 |
| arm export | 0 | 1 | 2 |
| **総計** | **7** | **811** | **1,629** |

`7+2[1+6×129+17+17+1+1]=1629`。旧実績1,042や旧上限12,430の二倍ではない。
共通初期physicalに加え、各armのJ再solve後のphysicalを独立に評価するため上の行は重複ではない。
Raw optimizer flag不成功でも既存の診断を継続できる場合は規約通りに行い、追加solve/refinementはしない。
新ffsim full-fitter呼出し0、outer BFGS0。新科学処理は上表のNewton J/U trajectoryであり、単なる保存値閲覧とは扱わない。

## 7. 下流資源への接続設計（今回の実行契約には含めない）

1. Stage Oの候補identity・物理予算を確認し、新候補ごとにground/aliasを判定。
2. **共通政策Stage 1**は両armともr32/s1/K0/δ=h=.1/m0..12/LD2を推奨・設計固定。既存H4 retuningがr32を支持するため共通anchorに採用するが、どちらの新候補にも最適との証拠はない。
3. 候補固有のPF/RTE26 roundセルとkernel/reference、attenuation、phase room、RPEを同じ規約で検証する。CA=.00159、DF reserve=CA/8、βtotal=π/6、βstat=π/12、候補failure .01/axis .01/26、own-DF ground oracle/状態準備除外。
4. **Stage 2 retuning**が必要なら、両armに同じr集合{16,32,64,128}と同じ選定予算・基準を登録する別契約。PRだけr32/L1だけr128にはしない。
5. Nativeは共通P0（Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation1、initial-zero false）、合法的basis共有・相対融合、serial head/readout/kernel会計。同じ各J/Uで全frame ordered S2/s1の強Dも測る。
6. 価格は既存独立cohort/t-SE実装を再利用するが、r/候補凍結後に新cohortを使用する。N256/640や旧selection cohortを新候補のholdoutとして使わない。

下流のcall式は「2候補×13round＋candidate-specific setup/ground/kernel/RPE等」、nativeは既存「9N＋候補固有reference/shell/集計等」。
これらは次契約を閉じるための設計式で、1,629-callへの上乗せ許可ではない。
候補未生成の現段階で価格Nや校正precisionを採択しない。次の候補凍結時にestimand、精度、N、multiplicityを一括固定する。
W×kernel mean＋shell、SE=W×kernel SEを保ち、rをcoherent反復へ二重計上しない。
064のground/PF/RTE/shots/W/nativeを新L1へ転用しない。
全law synthesis、implemented-law signal、actual CAが未資格ならnull。現段階で総資源の勝者は出ない。

## 8. 人工検証・環境・passive preflight

Bの38件を再利用し、新backend/runnerと触れた既存inner/stable/Armijoを回帰した。
**165 unique PASS（fit環境162＋evaluation環境3）**。Fit側の2 deselectはCAR環境で実行した2件で、統合coverageに未実行skipはない。結果・環境・実行commandはtechnical記録に保存する。人工J/U、人工Hamiltonian、mockのみを使用した。

対象は値/J/U微分/HVP、AD/同一配列/拡張精度、det±/π、zero/small係数、tail/smoothing/scale/相殺、
rank/head gauge、precision loss、J/stability未資格、物理NOT_MET、correctness FAIL、U受理、total point/call上限、
unapproved実行拒否、誤candidate/SOURCE、fresh/STOP、PR既定値の非回帰である。

Fit環境にはOpenFermionを加えず、CAR pipe試験は既存evaluation環境へ分離した。
安定性のfault-injection試験は、元人工問題のrefinement差が厳密0だったため、実際に非零の差を注入する形へ修正した。
閾値を緩和してPASSさせたものではない。最終の全対象PASSを記録する。

SOURCE closureは既存ローカルimport（関数内部を含む）と必要な既存lock、ffsim外部2 source、入力3 hashを固定する。
既存環境は読み取り利用。Python3.11環境を明示し、古いsystem Pythonへ依存しない。
起動前にaffinity/quota/空きRAMを確認し、独立2armをThreadPoolExecutorで処理する。
JAX形状cacheと元CAR contextを共有し、CARの短いJSON pipeはlockで逐次化する。
Arm内のtrial/refinementは依存順を維持。小さな固定U演算のBLASは自process内で協調し、多重threadの過剰生成を避ける。
CPU/RAM/worker/wallの人工capは追加しない。他ユーザーjob・共有venv/cache/OS設定を変更しない。

Passive command（科学計算なし）：

```bash
/tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python -B scripts/run_research_c_near_practical.py --contract docs/research/research_c_near_objective_practical_comparison_contract_unapproved_20261103.json
```

`--execute`単独は実行を拒否する。科学許可後だけ同じentryへ`--execute --science-permitted`を渡す。
後者は意図的な実行switchで、許可の認証/証明ではない。契約のunapprovedという歴史的metadataを書換える必要はない。
今回そのswitchによる実H4実行は行っていない。

## 9. 結論・STOP

実装上の未解決blockerはない。Stage Oの契約は技術的に実行可能で未承認。
科学的な制約は、単一初期化/単一strength/有限endpoint、J/U数値資格の工程的性格、候補固有下流・価格・厳密CAの未資格にある。
これらはStage Oの実行を妨げる未記入値ではなく、結論の範囲である。

共有Evidence Map、研究概要、科学validation manifestや科学レビュー完了記録は更新しない。
今回は科学結果や資格を変えず、関連する最小索引だけ接続した。
A/B原本・旧SOURCE/raw/契約/grant/ledger/marker/STOPを保全し、公開remote照合後mandatory STOP。
次はユーザーが固定1,629-call Stage Oを採択・明示許可するか判断し、許可された場合にCodexがfresh run01を一度実行する。
下流計算や正式科学レビューへ自動的には進まない。
