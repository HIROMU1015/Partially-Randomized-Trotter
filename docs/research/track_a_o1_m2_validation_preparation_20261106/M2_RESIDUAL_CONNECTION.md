# M2：完全残差・元H状態・PF/RTE/費用の接続

2026-11-06 JST。静的SOURCE調査。保存core・入力pinは[core map](M2_EXISTING_CORE_MAP.md)。本書は必要な最小接続の特定であり、新しい算法・runner・実行契約を設計／実装したものではない。式は既存SOURCEの記帳を説明する。残差・係数・state・回路・shot・資源は新計算していない。

## 1. 保存coreから何を保持するか

原実行SOURCE `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4` の[c1_stage1.corrected_kappa][stage1]、[SectorCAR.generators/raw_frame][molecule]、[prepare_arm][stageb]は、元h/gからのκ、核scalar、一般signed JとUを使う。模式的に

\[
H_{\rm core}=c_{\rm nuc}I+\sum_{pq}\kappa^{\rm orig}_{pq}E_{pq}
+\tfrac12\sum_{t,ij}J^{(t)}_{ij}N_i(U_t)N_j(U_t),
\quad R=H_{\rm physical}-H_{\rm core}
\]

である。ここでのH_coreは選んだ保存CDFの**全frameとbase**。核scalarとκをJ/Uから再fitしない。この定義を用いる最小比較候補ではD=H_core、random部分=Rとなる。候補coreの正式選択や時間発展公式・step幅・rank policyは未決定。

### 旧PRのtailは完全残差ではない

| 対象 | 旧研究Cの固定LD2 PR | M2で検討する全core＋完全残差 |
|---|---|---|
| D | 元κのbase＋frame0,1 | 選んだcoreのbase＋全frame |
| random operator | **candidate H内**のframe2..末尾、identityを既知phaseへ移す | **元物理Hと全candidate coreとの差** |
| target | 旧評価ではcandidate Hのenergy/own-ground | 元物理HのE₀と元H ground |
| 再利用できる価格・誤差 | 元候補の元split/時間/law条件の値 | 別split・別lawなので旧値を転記できない |

[split_native_model][connection]は`model.blocks[ld+1:]`のI/Z/ZZをtailへ出すprefix分割である。ldを全rankにするとcandidate内tailが空になるだけで、元Hとの差を自動生成しない。ld=2のまま旧tailへ差分を加える構成を選ぶならDも別になるので、全coreをDにするM2比較と混同しない。今回その追加構成は設計しない。

## 2. 完全残差の表現で落とせないもの

1. **残差の符号。** 保存`physical/update005.npz`の`tensor_residual`は[Evaluation.assess][evaluation]で`reconstruct(J,U)−g_original`として保存される。R=H−H_coreは逆向きの差。field名だけを見てそのままrandom tailへ投入しない。
2. **一体補正。** coreは元gからのκを保持する。残差tensorをnormal-ordered quartic用のPauli mapperへ渡すなら、density-product表現から生じる一体収縮を対応させる必要がある。`from_spatial_integrals(one_body=0, eri=差分)`と機械的に呼ぶだけでは保存coreとの差を実装したとはいえない。元κを保つ経路か、normal-orderへ変換して一体差を含める経路かを明示する。
3. **核scalarと電子identityを分ける。** 元Hとcoreで核scalarを共有するため、その差を重ねて加えない。一方、Pauli/JW展開や回転基底対角項から出る電子identity差がゼロとは限らない。Rのidentityを抽出する場合は既知相対phaseとして一度だけ保持し、未知E₀や経験的biasを補正scalarにしない。
4. **operator単位のcollect。** rotated Z/ZZはbasisを含むoperator。同じsupport labelだけで異basisの項を相殺しない。global Pauliへ書き戻すならspin/JW順、係数prefactor、exact-zero方針を揃える。
5. **sectorと実装path。** 完全な和operatorが粒子数を保存しても、各Pauli factorやgate途中がsector36に閉じるとは限らない。sector行列の差をそのまま全FockのPauli residualと見なさず、無断の途中射影もしない。診断referenceと実装を分離する。

上記はSOURCEの規約から必要になる接続条件であり、今回その残差を生成して数値同一性を確認した結果ではない。`||g−g_fit||`、既存sector error、DF予算PASSから、RTE辞書のλ_Rや安いevent生成を推測しない。

## 3. 既存コードの再利用と足りない境界

Iの[再利用地図][reuse]にあるDF-CONTROL、RTE-BASE、RPE-COST、研究Cのgeneral-K/native/codecを使う。以下は静的に読んだ関数・型の具体的な対応であり、import・test実行・新条件PASSではない。

| 接続 | 再利用できる既存処理 | 適合しないこと／最小の不足 |
|---|---|---|
| 保存J/U→core | 原SOURCE980…の`c1_general_k_adapter.from_ffsim_number_arrays`、`GeneralKModel`、`diagonal_primitives`、`c_pr_aware_df_stage_b.prepare_arm` | input一体は元hではなくcorrected κ、`z_representation=False`、`enable_general_k=True`が明示条件。既存roleやrank固定runnerの再起動はしない |
| 一般signed-JのD | `GeneralKFrame`のI/Z/ZZ、[StrongBuilder.sequence][strong]、既存full-frame ordered S2とactual-adjoint、analytic vacuum-phase補償 | 一般Jを単一の正平方DF fragmentへ型だけ変換できない。全core側のD half sweepと残差centralの境界・scalar ownershipを既存emitterへ渡す最小接続が必要 |
| 原物理Hとcore reference | `SectorCAR.physical/generators/raw_frame`、`Evaluation.assess`、既存general-K/raw CAR比較 | 候補rank/shape固定に注意。保存sector/full行列は検査referenceで、実装oracleではない。`assess`を呼ぶだけでも新科学再評価なので今回未実行 |
| 完全残差→Pauli tail | [PauliHamiltonian.from_terms/from_spatial_integrals][pauli]、既存unthresholded spin/JW規約、collect・identity抽出 | **Rをglobal fermionic/Pauli係数として組み立てる接続が未実装／未資格。** 元H/core双方を同規約にする小さいadapterが必要。新mapper一式やdense-Pauli分解を前提にしない |
| 完全残差→RTE | [rte.py][rte]の`RTEComponent`、`finite_rte_distribution`、`make_rte_config`、`finite_rte_operator_moments`、ordered event・phase処理 | まずRの実装可能involution辞書が必要。旧CDF tail確率・λ・normalizationはRへ流用不可。`normalize_involutory_tail`のdense検査は大規模用無料oracleではない |
| 標準DF library | `df_partial_s2.py`、`df_rte_tail.py`、`df_rte_qiskit.py`のGaussian registry・control・tail metadata | 標準`DFHamiltonian`のsquare/fragment prefixと、今回のgeneral signed-J全core＋任意Rは異なる型。`DFPartialS2Preparation`へ見かけだけ詰め替えない。任意RとDを分離して渡せるtyped境界が必要 |
| 状態・mean reference | 原SOURCE980…の`c1_stage3_reference.reference_kernels/reference_round`、Sの`paired_polynomial_action`、保存physical groundとindex規約 | referenceのtargetとstateを**元H側**へ渡す必要がある。旧candidate専用runner・ground選択を置換する最小入口が必要。新decoder全体を作らない |
| RPE統計・価格 | [rpe_resource_accounting.py][accounting]、研究Cの`c1_stage3_accuracy`/accounting、後続18round/energyの既存SOURCE | 旧library APIはDFPartialS2型・二次PF・reference radius1に特化。Rの新signal・保証された半径・誤差と価格providerを渡す適合照合が必要。旧CA/shot/long-round価格を移送しない |

`c_pr_aware_df.py`はPR tailの係数map/既存fit hook、`c_rcdf_comparator.py`は全frame L2型正則化対照のfit hookを持つ。保存coreは既に取得できるため、これらのfit呼出しを最小比較の前提にしない。新penalty・η・rank sweep・optimizerは不要であり、今回追加しない。

既存の`tests/management/test_research_c_c1_general_k_adapter.py`、`test_research_c_pr_integrated_pipeline.py`、`test_research_c_pr_integrated_native.py`、`test_research_c_same_ju_strong_d.py`と、`scripts/run_research_c_same_ju_strong_d.py`の所在をO1基点で確認した。将来の最小接続検査でまず参照できるが、旧固定runnerを再起動せず、元のtest成績をM2の完全残差correctnessへ移送しない。`run_research_c_cdf7_onefit.py`も既存だが、新fitは今回不要である。

### SOURCE版を混ぜない

`c_pr_aware_df_stage_b.py`は980…、025d…、O1基点で同一Git blobだった。一方`c_pr_integrated_performance.py`はO1基点の収録版と原実行版で異なる。原実行SOURCE980…のstate選択を直接読んで過去条件を確認した。Pauli adapterはO1基点に存在する別系列資産であり、980…の実行closureに元から含まれていたとは扱わない。

## 4. 元H groundと保存stateの扱い

[原performance SOURCE][performance]はphysical groundを評価した後、`ground('candidate',H)`から`psi=eig[1][:,0]`を選び、exact evolutionもcandidate Hで計算する。したがって旧13roundの信号・必要N・同candidate強D費用は、M2の元H厳密ground結果ではない。

元physical groundの保存数値vectorは[aced…/C/ground_physical.npz][ground]に存在する。将来の小規模診断で再利用する場合は、同じ入力bytes・sector basis/index・normalization・H対応を既存報告から結合し、足りない残差/固有状態検査だけを明示許可の範囲で行う。新しいground生成を自動前提にしない。保存数値vectorの利用は**検証用reference**であり、算法が古典全vector・E₀・gapを無料取得するoracle条件ではない。厳密状態を受け取れるという今回の仮定から、近似PFの半径1や単一位相は導かれない。

旧physical/candidate ground reportのalias検査にE₀数値を使ったことも、将来algorithmへの無料E₀ priorを意味しない。共通energyのbranch/alias priorは、取得可能な情報として別途明示する必要がある。

## 5. 最小比較の構造と費用交換

ユーザーが保存coreを1つ選んだ後に必要な比較は、まず同じ元H state/targetで**全core＋完全残差**を扱う既知residual PRと適用可能な強対照である。core-only強Dを対照に入れる場合も、元Hへのmodel誤差・状態信号・successの転送を支払う。近似H自身のground energyを比較して済ませない。UWC、collected Pauli、DF/RC-DF、既存PF・split再最適化には同じinput・error/price情報と合法fusionを与える。[既存先行研究比較][prior]

| 変化 | 減り得る費用 | 増え得る費用・必要な確認 |
|---|---|---|
| 全coreをDへ保持 | 安い少数frameでbackboneを構成できる可能性 | 原PRのLD2よりDが重くなる場合もある。内部PFとcore/residual境界の非可換biasを元H state上で確認 |
| 完全残差をRTEへ | 元Hを保ち、近似core groundを別oracleとして要求しない | 残差λ_R、nonlocal Pauli、basis移動、finite-bit law・高次数event・古典collect負担 |
| r/KやPF step | finite-RTE誤差・attenuationを制御できる | event数、D sweep反復、normalization/shot、native合成精度。旧r32/r128/K0やδをM2へ固定しない |
| 全round接続 | 同じ元E₀への統計会計ができればN×full価格で比較可能 | 元H state上のPF/mean位相・半径、全round branch成功、価格不確かさ・rare-law・長q移送が残る |

最終指標は `Σ_(m,b) N_(m,b) E[C_full,no-prep^(a)]`。RZ/CX/T/depthは別々に扱い、古典前処理・追加qubitも別欄とする。core fit誤差をRへ完全に戻すなら同じ誤差をmodel allowanceとして二重加算せず、反対にfinite-bit residual・native誤差まで消えたとはしない。量子shotと古典価格sample数を二重に乗じない。

## 6. 次に必要な範囲とSTOP

静的入力準備はできたが、**残差生成・最小adapter実装・元H stateでのmean/PF/半径検査は未実施**。最初の科学検証へ進む前に、ユーザーは保存core、元H/state reference、必要な時間・精度・失敗確率と主資源指標を選ぶ。既存の保存数値条件は候補であり、自動採択しない。

将来許可された場合の処理順は、(i)選択済みbytesと規約の結合、(ii)完全残差とD/R/phaseの対応、(iii)元H state上のPF/finite mean/半径、(iv)必要なfull-wrapper価格と全round N/Gの順に不足を閉じる。これは新しい実行契約やcall数の固定ではなく、今回見つかった依存関係である。

STOP条件は、共通target/stateを保てない、補正mean/phase不一致、一体/scalarの二重計上、非正半径・誤差予算超過、oracle依存、残差負担で利益消失、強baselineに差なし、価格/成功条件未資格による判定不能。新fitが不可欠と判明した場合も、研究Cの既存cost-aware fit NO-GOをM2名で解除せず、その事実を報告して停止する。

今回の3成果物を公開・取得照合したらmandatory STOP。科学計算・実装・compile・旧retry/resumeへ進まない。

[stage1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c1_stage1.py
[molecule]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c1_molecule_extension.py
[stageb]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c_pr_aware_df_stage_b.py
[connection]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c1_stage3_connection.py
[evaluation]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c_pr_integrated_evaluation.py
[strong]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c1_h4_strong_baseline.py
[pauli]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/src/trotterlib/pauli_signal_adapter.py
[rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/src/trotterlib/rte.py
[accounting]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trotterlib/rpe_resource_accounting.py
[performance]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/980b514c81eb415fe053f558d7d3c0b8cfb5b9b4/src/trottertracks/c_pr_integrated_performance.py
[ground]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/C/ground_physical.npz
[reuse]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/DUPLICATION_AND_REUSE_MAP.md
[prior]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/PRIOR_ART_COMPARISON.md
