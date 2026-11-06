# O1 M2：保存core＋完全残差の改善余地（静的調査）

2026-11-06、SERVER 2（GPUA100-gleap02）。起点は`68ef4feee3835ed7135d254ac879e8d88c69c895`。**SOURCEの静的確認・既存結果の引用・技術成立条件の整理のみ。新科学計算、保存値の再集計／再ランキング、実装、詳細科学レビュー、正式方針採択は行っていない。** 既存B operator PASSと物理model `UNASSESSED`、研究Cの技術NO-GO、旧FAIL・許容値・SOURCE・raw・STOPを保持する。

結論は、**完全残差を扱う代数的接続は成立しているが、総費用改善の証拠はまだない**。小さい残差のnormalization負担を軽減できる可能性と、全coreの決定論PF・Gaussian変換の固定費との交換を調べる候補である。core-only近似と同J/U強決定論PFを省くと、不要な残差random化を利益と誤認し得る。

## 引用できる既存証拠と限界

対象は保存直鎖H4・1 Å・STO-3G、update005のnumber-CDF rank8／全8 frame、元g由来κ、核scalar一回、`Nα=Nβ=2`。新M2の目的は元Hの厳密基底状態・状態準備費用除外での同精度／同成功確率の全RPE費用であり、旧研究Cのown-DF ground oracleとは区別する。

| 固定証拠の引用 | 意味と留保 |
|---|---|
| [B検証][B]：R_H mapper−CAR max-entry full256 `8.271806125530277e-25`、sector36 `4.1359030627651384e-25`、残差専用τ `2.342260361136004e-18` | 完全残差自身のoperator一致。PF・信号・回路・総費用の証拠ではない |
| [B検証][B]：`H_H−H0` max-entryは両space `5.551115123125783e-17`、`R_H−P(raw R)`は両space `2.2551405187698492e-17` | 観測entry差。spectral norm、元H energy誤差の厳密上界ではなく、物理modelはUNASSESSED |
| [B検証][B]：`C_H−C0` max-entryはfull `1.1102230246251565e-16`／sector `5.551115123125783e-17`、`R_H−raw R0`は両space `1.0104764247564901e-16` | 後者はraw残差からのprojection＋差形成を含む変更量。上段の`R_H−P(raw R)`という形成経路差と区別する |
| [B検証][B]：R_H参照max-entry `2.2651961936020587e-09`、電子I_R `4.6249802598818945e-11`、nonidentity360項 | 残差非零と微小係数保持。Pauli λ_Rやnative価格を今回算出・推定していない |
| [同J/U強D][D]：同じupdate005で旧LD2 PRの主RZ条件付き総費用は強Dの`1.9701854206178373`倍 | 旧δ=.1、s1、PR r128/K0、13 round、CA=.00159 Ha、failure=.01、own-DF state、seed17の有限cohort／serial会計。新M2完全残差、元H state、full no-prep全長価格へ転用不可。副指標CX depthではPR側が低く、全指標支配ではない |
| [研究C技術判断][C]：`TECHNICAL_NO_GO_CURRENT_EVIDENCE` | λ以外の構造指標からcomplete native価格を下げるcost-aware fitの根拠が未成立。PR一般の不可能性や研究全体終了ではない。B PASSはこのNO-GOを解除しない |

上記は固定報告に書かれた値の引用であり、NPZ/QPYのdecode、norm再算定、旧13 round／32標本の再集計は行っていない。保存core/inputのhashは[B]と[統合報告][I]に結び付く。旧LD2 tailはcore内部の一部で、`R_H=H_H−C_H`という完全残差ではない。

## 規約Bから元H energy精度へ接続する条件（概念式のみ）

1. **比較対象を先に定義する。** 保存raw `H0`にはanti-Hermitian丸め成分があるため、その固有値をHermitianな元物理Hの基底energyと同一視しない。元のHermitian target `H_phys`、保存bit値・積分生成誤差の扱い、`H_H=P(H0)`との対応をユーザーが決める。旧FAILや許容値の置換で済ませない。
2. **適切なnormと参照誤差が必要。** sector射影Πで`epsilon_H >= ||Π(H_H−H_phys)Π||₂`を正当化できれば、両者Hermitianの同sector最低固有値についてWeyl型のenergy誤差評価が可能である。sector boundにはtarget・実装のsector保存、元stateのsector、漏れの扱いが必要。entry比較からは一般に`||Δ||₂ <= d ||Δ||max`という変換はあるが、観測entry差と参照丸めのenclosureを先に区別する。今回その積や新しい数値boundは計算していない。
3. **元H stateをH_Hの固有状態へ置き換えない。** 元Hの保存stateの基底・spin／bit順を元h/gにbindする。厳密H_phys基底状態という仮定から、H_H／C_H／有限PFの固有状態・半径1は従わない。epsilon_Hが保存sector内のnorm boundなら、式もそのsector内の`||Π[exp(−itH_H)−exp(−itH_phys)]Π||₂ <= |t| epsilon_H`とする。このような同normの信号disk評価で元H理想位相へ接続する案なら、H_H固有stateを仮定せずに済む。別途固有branch／overlapを使うならgap等の追加条件が必要になる。
4. **errorの単位と重複計上を整理する。** model差[Ha]、core内部PF／outer split／RTEの複素信号誤差、実装／law誤差、統計位相予算を分ける。時間依存の信号誤差と半径下界を通じて位相へ変換し、全必要roundのbranch・誤差配分を満たすことを確認する。核scalar・電子identityの相対制御位相は一度ずつ保持する。数値model boundをenergyとphaseへ無根拠に二重加算しない。

## 改善機構、反証材料、既知法との重複

**仮説：** 微小R_Hの辞書λ_Rが小さければ、finite-RTE normalization／打切りによるshot増分が小さくなり得る。ただし残差max-entryの小ささからλ_Rやshot数は決まらず、core全体の非可換PF誤差・Gaussian往復・controlled scalar・X/Y測定wrapperの費用は残る。小角度でもexact-zero policy下でgate価格が小さくなる保証はない。

比較の目標は`G = Σ_m,b N_m,b E[C_full,no-prep,m,b]`であり、core-only、M2 PR、同J/U強D、元Hに対応した保存rank10強DF-PFを同じenergy精度・成功確率・元H state・compiler／metric・初終端境界で比べる。classical価格MC数はquantum shotの乗数ではない。旧serial kernel価格、prep込みwrapper、未検証proxyを新full no-prep価格へ埋めない。

- **core-onlyを反証対照にする。** R_Hの省略誤差とBのmodel差を予算内に正当化できるなら、同じ元Hを目標とする決定論core-only近似で足りる可能性がある。これは別Hamiltonianを目標へ変更する提案ではない。追加random化の費用に見合う精度／shot改善が必要で、core-onlyが十分安ければM2の費用仮説は支持されない。compilerの非加算性があるので、構文上の追加gateだけで費用支配を証明しない。
- **core内部PFを無視しない。** `R→0`でも一般J/Uの全8 frameと一体項の強い決定論積公式が主仕事になる。λだけを減らす旧研究Cの改善が主RZで強Dに届かなかった既存反証と整合して評価する。
- **新規性は未認定。** 近似core＋完全補正は一般的な分割／残差simulationの構造で、既存partial-PR・general-K/native・相対basis融合の組合せだけを新算法と呼ばない。M2で区別できる設計点は旧LD2 tailではなく完全残差を修正すること、Bのmodel差と形成／mapper誤差を分けること、全core費を含めることである。これが既知residual-PRを超える機構・資源利益かは未検証。UWC／RC-DF一般への新規性・優位性は判定しない。[旧調査][SURVEY]も同じ留保を置く。

## 再利用資産と必要な最小接続

| 既存SOURCE／資産（指定の固定commit内） | 再利用可能な部分／未接続 |
|---|---|
| [B adapter・runner][SRC]、`o1_m2_cutoff_free_pauli.py`、独立CAR／density | 全core＋完全残差のbit値・scalar・κ・sector・符号・微小係数。B operator PASSを再実行して作り直す必要はない。model→energy資格は別 |
| [同J/U強D][D]と`c1_stage2_native.prepare_cdf/ordered_schedule`、`c1_h4_strong_baseline.StrongBuilder` | 全frame general-K、ordered S2、exact spin／relative fusion、controlled phase、actual adjoint、X/Yの既存資産。既存native coreが**C_H**を実装するかはB固有の新確認が必要 |
| `WholeCoreResidual.native_core/residual_rte` | 現状はraw general_core／raw Rへつながる。B candidateのstrict R_Hへ自動切替しないため、そのまま新B kernelとは扱えない。core PF半step＋R_H lawの最小接続は未実装 |
| [M1 bridgeのprimitive][M1]と`rpe_hadamard_interrogation`／`rte_compiled_cost` | 実半径accounting、両軸full no-prep wrapper／compile／MC統計を再利用候補とする。M1固有`T_eff/V_square` preparationをM2へ偽装しない。native旧wrapperはoccupied-Xのprepを含むため、その価格はno-prep対照に使わない |

## 最小の次検証案とGO/STOP（未採択・別途実行承認が必要）

**ユーザー**がtarget H_phys／B model差の扱い・normと参照precision、元H stateのbinding、energy／failure／branch、coreとRのPF／RTE／実装誤差配分、同一compiler・metricを先に固定する。これが未決定なら科学STOP。fit・rank探索・新しい目的関数の採択は含めない。

承認後の最小案は、**Codex**が①既存B結果を再利用して元Hへのmodel norm／signal接続を確認、②保存J/Uを変えずB対応coreと完全残差の短いcontrolled kernelを独立参照・相対位相・scalar込みで確認、③core-only／同J/U強D／rank10強DF-PFと、登録した短round・目標horizon付近のcore PF／残差信号を比較、という順である。少数roundから全RPEへ外挿しない。各候補が精度・半径・実装資格を通過した場合に限り、同じ条件のfull no-prep価格校正と全必要round会計を別途限定承認する。価格MCのcohort・seed・SE／CIの扱いも実行前に固定する。

model関係不明、B native不一致、core PFが予算超過、残差省略で同精度を満たしPRの費用利益が見えない、または価格domain／信頼性が未資格ならSTOPし、パラメータ・許容値を後から変えて救済しない。現在は「技術候補の整理」までであり科学GO・費用改善・新規性の認定ではない。公開SHA・本文一致を確認後**mandatory STOP**し、新計算・実装・詳細レビューへ自動進行しない。

[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68ef4feee3835ed7135d254ac879e8d88c69c895/docs/research/track_a_o1_m2_hermitian_b_science_20261106/REPORT.md
[SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68ef4feee3835ed7135d254ac879e8d88c69c895/src/trottertracks/o1_m2_hermitian_b.py
[D]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38b8d0df39b8943178ecef611194819cff1c2169/docs/research/research_c_same_ju_strong_d_execution_20261102.md
[I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/docs/research/research_c_pr_integrated_execution_20261102.md
[C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68ef4feee3835ed7135d254ac879e8d88c69c895/docs/research/research_c_pr_native_cost_improvement_feasibility_20261102.md
[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68ef4feee3835ed7135d254ac879e8d88c69c895/docs/research/research_c_hamiltonian_circuit_survey_20261024.md
[M1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a13539e51bb113e82cb86ddb608e2e6e27786ca/src/trotterlib/o1_m1_resource_bridge.py
