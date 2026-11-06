# O1 M1：RPE総コスト評価への接続計画

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。**技術整理・静的SOURCE確認のみ完了。科学実行は未承認、mandatory STOP。** 起点は`424ea9d1f345f15ec863ee9102a997b0b3fbd858`。以下は実装・検証案であり、採択済みの精度条件や新しい性能結果ではない。既存4条件の再解析、shot/費用計算、回路生成、simulation、テスト実行はしていない。

## 1. 固定条件と再利用資産

H4-A（linear H4、1Å、STO-3G）、保存DF rank10、元Hの保存`state`、HD=T_eff／HR=V_square、fragment LD0を維持する。input commitは`4591b8d75a2a6a55c74a51225f0c61648023c8ee`、pathは`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。[既存報告][M1]の範囲はδ=0.1/0.05、q=2、r=2、K=0/2、faithful tail identityである。mean/operator PASSを全RPEの精度・費用PASSへ読み替えない。

| 確認したSOURCE・資産（この起点の版） | 再利用と制約 |
|---|---|
| [o1_m1_matched_split](../../../src/trotterlib/o1_m1_matched_split.py)、[回路・信号検証](../../../src/trotterlib/o1_m1_circuit_signal_validation.py) | 同じT半step、核scalar一度、Pauli/DF tail、独立CAR/mean、controlled phase。204代表eventは確率標本・価格期待値ではない |
| [rpe_resource_accounting](../../../src/trotterlib/rpe_resource_accounting.py)：`evaluate_rpe_round_candidate`、`build_rpe_resource_summary` | 誤差配分・Hoeffding・全round集約を再利用。ただしDF型、二次PF、未減衰半径1／effective PF eigenstateという仮定がある |
| [finite_rte_signal_validation](../../../src/trotterlib/finite_rte_signal_validation.py)：`_state_result`、`_hoeffding_shots` | 実PF信号半径・打切りdisk・観測半径を分ける既存処理。数値pilotと厳密下界の資格は別 |
| [rpe_target_round_horizon_validation](../../../src/trotterlib/rpe_target_round_horizon_validation.py)：`required_rpe_round_horizon` | 精度・位相予算からhorizonを決める算術。旧rank12のM、C、scheduleは移送しない |
| [df_rpe_resource](../../../src/trotterlib/df_rpe_resource.py)：`DFLevel5RCompiledCostProvider` | controlled evolution部分のみ。full no-prep価格として使わない |
| [df_rpe_hadamard_compiled_cost](../../../src/trotterlib/df_rpe_hadamard_compiled_cost.py)：`DFRPEHadamardCompiledCostProvider`、[wrapper](../../../src/trotterlib/rpe_hadamard_interrogation.py) | 既存にfull no-prep短round経路はある。X/Y、測定込み一体compile、同一trajectoryを両軸へ使用。DFのpreparation/result型への接続が必要 |
| [pr_cost_native.compile_task](../../../scripts/pr_cost_native.py)、[StrongBuilder](../../../src/trottertracks/c1_h4_strong_baseline.py)、[stable Gaussian](../../../src/trotterlib/stable_gaussian_basis.py) | native wrapper一体価格、scalar/control照合、同基底・境界融合の部品。旧入力・grant・runを再利用せず、新M1の同値性検査とcompilerを固定する |
| [validated proxy provider](../../../src/trotterlib/rpe_hadamard_validated_proxy_provider.py)、[UWC](../../../src/trotterlib/uwc.py)／[grouped比較](../../../src/trotterlib/grouped_uwc_comparison.py) | proxyは旧較正domain限定。UWCはsimple shift/BLISS等で、OOは`NotImplementedError`。grouped比較はPF+QPEの係数・layer価格で、今回のRPE価格ではない |

回帰資産は`tests/test_o1_m1_circuit_signals.py`、`test_rpe_resource_accounting{,_guards}.py`、`test_df_rpe_hadamard_compiled_cost.py`、`test_rpe_hadamard_interrogation.py`、`test_uwc_preprocessor.py`。今回はSOURCEを読んだだけで、既存testの過去PASSを新条件の証拠にしていない。

## 2. normalization・半径・誤差からshot／価格へ

辞書d、round mについてq_m=2^m、t_m=q_m δ。B_{d,m}を**1 short-step**のfinite distribution normalizationとすれば、A_{d,m}=B_{d,m}^{−q_m r_m}、z_obs=A z_corr。既存M1 rawの`normalization`は既に全反復分なので、再度q r乗して二重計上しない。同一trajectoryをq回再利用するlawは、独立eventを合成した現在のmeanと同一ではない。

採択された基準信号の半径下界をρ_star,lb、補正mean誤差diskをε_Zとすると、接続案は

\[
\rho_{obs,lb}=A(\rho_{star,lb}-\epsilon_Z)>0,\qquad
\beta_{RTE}\leq\arcsin(\epsilon_Z/\rho_{star,lb}),\quad \epsilon_Z<\rho_{star,lb}.
\]

元H ground `state`からPF半径1は従わない。数値で観測したraw半径、モデル値Aρ、正当化した下界を別fieldで渡す。補正半径を1へclampせず、ε_Zをnormalizationへ二重に混ぜない。実装差・PF bias・RTE打切り・元H/DF差も別の誤差源である。

独立な±1 Hadamard結果を仮定した既存式は、η_{m}=ρ_obs,lb sin(β_stat,m)/√2、N_{m,b}=ceil[2 log(2/α_{m,b})/η_m²]。**式を確認しただけでNは算出していない。** RTEは各quantum shotの内部eventとshot間で承認済みfresh IID lawを使う。古典価格MC標本数はNの乗数ではなく、価格SE/CIも測定失敗確率と分ける。

目標はG_d=Σ_m Σ_b N_{d,m,b} E[C_full,no-prep,d,m,b]。CはT保持、RTE、scalar/control、ancilla H/X/Y、readout/measurementを含む各軸の一体compile価格。state preparationを含めず、prep込み旧価格からの単純減算もしない。event単価の和やevolution部分価格をfull価格へ代入しない。価格用trajectoryは元の有限lawと重みを維持し、選定eventを等重み平均しない。

## 3. 必要な最小接続とfair baseline

| 不足 | 最小差分・検証案（未実装） |
|---|---|
| Pauli側がgeneric prefix分割で共通Tを失い得る | `PauliSignalAdapter(full H,ld)`へ丸投げせず、既存M1 splitと同じGaussian T保持program＋各辞書tailを薄く接続する。新selector不要 |
| 数値検査用Pの`tail_hash="matched_P_numerical_check"` | 価格cacheには使わず、既存`PauliHamiltonian.hamiltonian_hash`等から係数・順序・identity/law/入力に結び付く辞書hashを作る |
| accountingがDF型・単位半径specialization | 既存実半径helperを小さく切り出し、半径・誤差diskを明示入力するM1経路を追加。既存certified意味を黙って広げず、数値下界/経験PFなら条件付き・empiricalのまま |
| full no-prep providerがDF専用 | 既存wrapper/metric/compiler/cacheを使い、matched P/DFの同じcontrolled repeated programを受ける最小adapter。native lowering採用後も元programとoperator・相対phase・複素信号を検査 |
| 長round・q=1を含む信号／価格が未検証 | 精度から必要horizonを決めて新scheduleを固定。直接可能な短roundで接続し、長roundはそのM1入力・辞書・compilerに対するholdout済proxyだけを使う。providerの旧q guardは検証domainでありCPU/RAM上限ではない |

共通比較条件は、同じ元H/物理状態・sector・energy branch、同じε_Eと1−α_tot、同じ費用metric・compiler/runtime・gate set・routing・制御方式・no-prep境界である。model/DF・合成・state資格に必要な予算を明示し、二重計上せず、残りε_phaseに対してβ_RPE/(2^M δ)≤ε_phase、各roundのPF/RTE/統計等の位相配分和≤β_RPE、Σα_{m,b}≤α_totを課す。alias-free energy区間とbranch成立も必要。数値の精度・成功確率・Mは今回未指定／未計算。

- **主baseline候補：強決定論DF-PF。** 同じ保存rank10の全fragmentを保持し、scalar/one-bodyを含め、適用可能なspin構造・同基底融合・relative Gaussian境界を検査する。同じJ/U/core8や旧rank12価格を流用しない。単純未最適化PFだけとの勝敗で優位を主張しない。
- **補助baseline候補：決定論Pauli-PFとUWC-style Pauli。** 前者は同じ元Hで全項保持。後者は使用可能な既存変換・target sector・補償scalar・元stateとの対応を先に固定し、新入力のoperator/signalと同じRPE/no-prep/compiler経路へ接続できた場合だけ比較する。OO予約機能や旧QPE/STAR/layer価格をUWC全体の強baselineとして扱わない。新OO・fit・辞書探索は本計画の最小範囲に含めない。
- 比較は同じ達成精度・成功確率で行い、必要なら方法別δ/r/Kを事前指定した同等の調整範囲で選ぶ。M1辞書差の因果比較では共通保持program・scheduleを固定した対照も残す。別runtime・別target・別精度の保存価格をpoolしない。

## 4. 検証順序・GO/STOP・必要な承認

1. **ユーザーの判断**：ε_E、α_tot、energy区間、誤差／失敗確率配分、baseline範囲、価格metric/compiler、直接評価とproxyのdomainを固定する。4条件から新しいCやround scheduleをfit・外挿しない。
2. **Codexの次の実装候補**：上記のmatched trajectory／実半径accounting／full no-prep providerを最小接続し、人工fixtureでBの反復指数、核/電子identity、controlled phase、両軸、半径不足・未知価格・欠損roundの拒否を検査する。今回はその実装も未実施。
3. **別途科学承認が必要**：保存H4 decodeと新roundのPF/RTE/元state信号・下界評価、精度／horizon／shot算術、native回路build/compile、価格期待値のexact列挙または明示許可した古典sampling、長round proxy較正/holdout、全round集計とbaseline再評価。必要な範囲を一括固定して承認でき、段階ごとの承認儀式は不要。

**GO**は全必要roundの誤差・半径・branchと価格domainが資格を持ち、同じ物理精度・成功確率・費用scopeで比較できる場合のみ。**STOP**は未検証外挿、半径非正、単位半径の無条件使用、empirical→certified昇格、未知価格を0扱い、旧price/契約/STOPの無断転用、非対応UWC機能・価格domainの利用。全roundが揃う前は限定pilotと明記し、総費用改善を主張しない。

今回の成果物は本Markdownだけ。主agentが静的確認を行い、詳細科学レビュー・新科学計算・停止済み実験の再開はしていない。旧SOURCE/raw/grant/marker/STOPを保全する。独立branch公開・remote SHAと全文一致確認後に**mandatory STOP**。次は**ユーザー**が比較条件と実施範囲を決め、**Codex**が許可された実装・検証だけを行う。

[M1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/424ea9d1f345f15ec863ee9102a997b0b3fbd858/docs/research/track_a_o1_m1_circuit_signal_validation_20261106/REPORT.md
