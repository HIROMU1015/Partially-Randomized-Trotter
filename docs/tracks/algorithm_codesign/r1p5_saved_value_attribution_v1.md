# R1.5 saved-value attribution — POSTHOC / mandatory STOP

2026-10-06 JST。利用者の[固定指示snapshot](inputs/r1p5_saved_attribution_user_instruction_20261006.txt)に従い、
**保存済みR1だけ**を再集計した。新science・synthesis・compile・matrix評価・候補追加は0。
目的は原因帰属とGPTの次設計への入力であり、R1の科学分類を変更しない。
R1は元契約どおり`R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW`のまま。

| Provenance | Identity |
|---|---|
| input R1 commit | `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b` |
| source S / authorization A | `d43d64a821a0249a0dfab12a2472bd3a72fdee74` / `411f08f768244fe87b600d82308c3851847fe9e4` |
| original result SHA256 | `f726ad70cb2643533f0d037b518cde1b702724adb4e6571fea25e26e4bfdd61e` |
| original marker SHA256 | `f25000ee5e3b94eb499b4a89bb28814e3de1aea641b249bcbb9d953d427c2ced` |
| analysis script SHA256 | `b26c3bdea5a42988c0534d45bd092d0cc28e50535e08dc75b4692ed601799911` |
| classification | POSTHOC_ATTRIBUTION_DESIGN_INPUT、no science/synthesis rerun、no new candidates |

[stdlib解析script](../../../scripts/tracks/algorithm_codesign/analyze_r1p5_saved_attribution.py)、
[summary](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json)、
[provenance manifest](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/evidence_manifest_v1.json)。
独立branch/worktreeは`track-b-r1p5-saved-attribution-20261006`、
`/home/abe/Project/prt-worktrees/track-b-r1p5-saved-attribution-20261006`。
元result・marker・authorization・source、R0/R0.5、BF/BM/SP/BS、Track Aを保持した。

## 1. Scopeと会計

R1の2-qubit synthetic P₃(-iσxR̂)、R̂=3Q₀/4+Q₁/4、m=3/K=2、x={1/8,1/4}、σ=±1だけ。
primaryはdistinct_basis、Q₀=Z₀、Q₁=V†Z₁V、V=exp(-iπX₀X₁/16)、controlled=true。
armはordinary/PTSC_K0/A、登録native ε={10⁻³,10⁻⁴,10⁻⁶}。
分子・geometry・basis・DF rank・Hamiltonian splitは該当しない。
native metricはshared exact cancellation後のadditive synthesized-primitive Clifford+T counts。
pygridsynth2.0.0の固定設定・保存列だけであり、whole-circuit最適化済みcostではない。

task resourceは保存された共通confidence forecast：ε_axis=.005、ε_complex=.01、familywise α=.05。
G_T/G_CX/G_1Qは測定結果ではなく、保存sufficient shots×保存per-shot costである。
G_1Qは準備・読出しを含むため、N_total=2n_axisとして

\[
G_T=N_{\rm total}E[T],\qquad G_{\rm CX}=N_{\rm total}E[CX],\qquad
G_{1Q}=N_{\rm total}\bigl(E[1Q]+5/2\bigr).
\]

1Qのfactorizationではtask per-shot費用E[1Q]+2.5を使い、native E[1Q]も別保存した。
この補正なしにG_1Qの差をnative費用の比だけで説明しない。

σの二符号は各々解析した。primaryの全9点について、resource/bias/shot座標が符号間でexactに一致した。
以下は+1の表示を用いるが、二符号を独立replicationとして数えない。
全36 primary arm rowsと全126 angle rowsを保持し、他の登録controlsもfactorization/angle表へ残した。

## 2. Precision-envelope Pareto

各x・各σの登録9点を(G_T,G_CX,G_1Q)で比較した。
全座標non-worse、少なくとも一つstrict betterだけをdominanceとし、scalar score、tolerance、materialityを追加しない。
表示小数で判定せず、保存exact fractionsを使用した。
interval座標にはupper(left)≤lower(right)の保守的robust ruleを用いる。
主taskのG座標とimplemented B²は保存されたexact model値である。
optional mechanism比較ではideal B² enclosureも保持し、overlapをpoint表示で消さない。
同じ非零幅のideal interval間の共有相関は利用しないため、そのfrontは保守的である。

| x | 全task Pareto点（arm:precision） | Aの残留点 |
|---|---|---|
| 1/8 | PTSC_K0:1e-4、A:1e-4、PTSC_K0:1e-6 | A:1e-4 |
| 1/4 | A:1e-3、PTSC_K0:1e-4、A:1e-4、PTSC_K0:1e-6 | A:1e-3、A:1e-4 |

| A登録点 | ordinary/PTSC-K0からのdominator | A内のdominator |
|---|---|---|
| x=1/8, 1e-3 | PTSC_K0:1e-4 | A:1e-4 |
| x=1/8, 1e-4 | なし（全6 baseline点を比較） | なし |
| x=1/8, 1e-6 | PTSC_K0:1e-6 | なし |
| x=1/4, 1e-3 | なし（全6 baseline点を比較） | なし |
| x=1/4, 1e-4 | なし（全6 baseline点を比較） | なし |
| x=1/4, 1e-6 | PTSC_K0:1e-6 | なし |

登録ordinary点は両xのtask frontに残らなかった。
これはAが全baselineに勝つという意味ではない。PTSC-K0の1e-4/1e-6とAの間にtrade-offが残る。
特にx=1/4のA:1e-4はPTSC_K0:1e-4よりG_T/G_CXが高いが、G_1Qが
約539.976 million対542.149 millionで低いため三座標frontに残る。
Pareto残留と実用的materialityを同一視しない。閾値は追加していない。

[全座標・front・全baseline dominance checks JSON](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/precision_envelope_pareto_v1.json)と
[36点CSV](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/precision_envelope_pareto_v1.csv)に、
implemented B²、n_axis、E[T/CX/1Q]、G_T/G_CX/G_1Q、coherent bias、workspaceを保存した。
mechanism-only frontsはimplemented B²版とideal interval版を分離している。

## 3. G資源のexact factorization

全36 controlled comparison conditions、各2 baseline、3資源の**216 factorization**を照合した。
N_A/N_Bとtask per-shot費用比の積が保存G比にexactに一致する。
shot側は非線形なので、B/B²、coefficient/synthesis bias、remaining axis budget、保存nを併記し、
N比を一意なcausal percentageへ分解していない。

### x=1/4、ε_native=1e-3

| A / baseline | B²比 | N_total比 | E[T]比 | G_T比 |
|---|---:|---:|---:|---:|
| ordinary | 0.9992511965 | 0.9002996171 | 0.8844632383 | 0.7962819147 |
| PTSC_K0 | 0.9945715353 | 0.9055130347 | 0.9094176195 | 0.8234895084 |

| Arm | coherent bias upper | remaining axis budget | n_axis |
|---|---:|---:|---:|
| A | 0.0004148079809 | 0.004585192019 | 996,403 |
| ordinary | 0.0006480711792 | 0.004351928821 | 1,106,746 |
| PTSC_K0 | 0.0006252151365 | 0.004374784864 | 1,100,374 |

normalizationは低い方向にあり、保存biasも低く、remaining budgetが大きく、shot数は少ない。
per-shot T費用も低い。この二つのratioが掛け合わさった結果であり、normalizationだけのgainではない。
N差をB²差とbias差へ因果的な百分率で割り振ることはしない。

native expected Tの差A−baselineを、保存angle寄与のexactな加算として整理すると：

| baseline | rotation keysの差 | basis keysの差 | 合計E[T]差 |
|---|---:|---:|---:|
| ordinary | -11.76380910 | -0.19425272 | -11.95806182 |
| PTSC_K0 | -9.32888565 | +0.21086114 | -9.11802452 |

主なnative T差は共通ρ rotation側に現れ、PTSC-K0比ではbasis寄与が逆方向である。
これはそれぞれの実際の保存probabilityで重み付けしたobservationalな差であり、
確率やsequenceを入れ替えたcounterfactualではない。

### x=1/8、ε_native=1e-4

| A / baseline | N_total比 | E[T]比 | G_T比 | E[CX]比 | G_CX比 |
|---|---:|---:|---:|---:|---:|
| ordinary | 1.001215783 | 0.9072596638 | 0.9083626948 | 0.9990747070 | 1.000289365 |
| PTSC_K0 | 1.000717196 | 0.9135417884 | 0.9141969766 | 1.006243767 | 1.006965441 |

Aのbias=7.278148226×10⁻⁵はordinary=6.965331975×10⁻⁵、PTSC-K0=6.933352251×10⁻⁵より高い。
B²はAの方が低いが、n_axisは789,751対788,792/789,185で少し多い。
T側はper-shot T削減がこのshot増を上回る。ordinary比ではper-shot CXは低いがshot増でG_CXが高くなり、
PTSC-K0比ではper-shot CXとshot数がともに高い。normalization最小化だけではこのtrade-offを説明できない。

[全216 factorization JSON](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/factorization_table_v1.json)／
[CSV](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/factorization_table_v1.csv)は1Q overheadと全shot入力を含む。

## 4. 保存angle/precisionの離散効果

全126 synthesis rowsのangle identity、T/T†/1Q count、guard、sequence length/hashを保持した。
angle keyはRZ primitiveのθであり、controlledでは±atan(q) pair、ordinaryでは±2atan(q)を使う。
Aのρは全even groupとodd complementで共通。保存a/b/complement/signから既存keyを参照し、atanを評価していない。

以下は正符号の保存T count。負符号も全表に残す。T_countはTとT†の合計である。

| x | q role | controlled half-angle scale=1：1e-3 / 1e-4 / 1e-6 | ordinary scale=2：1e-3 / 1e-4 / 1e-6 |
|---|---|---|---|
| 1/8 | x=1/8 | 36 / 52 / 68 | 38 / 52 / 72 |
| 1/8 | x/3=1/24 | 40 / 46 / 68 | 38 / 48 / 70 |
| 1/8 | ρ=385/3096 | 40 / 46 / 70 | 38 / 48 / 68 |
| 1/4 | x=1/4 | 42 / 50 / 70 | 42 / 58 / 84 |
| 1/4 | x/3=1/12 | 38 / 48 / 70 | 36 / 44 / 76 |
| 1/4 | ρ=97/396 | 36 / 50 / 70 | 38 / 46 / 68 |

x=1/4,1e-3ではρ controlled pairが36+36=72 T、ordinary order0 pairが42+42=84 T。
Aの全eventが共通ρを使うため、event probabilityで重み付けしたrotation部分は72 Tとなる。
ordinaryのrotation平均との差は上記-11.76380910 T。basis差は-0.19425272 Tである。
大きいG_T gainは、少数の共通ρ keyの安い保存列と、より小さい保存biasによるshot差で増幅されて見える。
別sequenceや別angleに変えても残ることを示したものではない。

x=1/8ではρ half-angleは1e-3でxより高く（40対36）、1e-4で低く（46対52）、1e-6で高い（70対68）。
同じrepresentationのnative T優劣が登録precisionで反転する。ordinary scale=2とcontrolled half-angleの傾向も一致しない。
共通basis key±π/8のT countは38/50/68で、これもexpected T/biasへ含めた。

全42 angle keysの84隣接precision差は+6–+32 T。最大+32はatan(1/12)、scale=±2の1e-4→1e-6。
raw差・relative差を全件保存しただけで、resonanceのthresholdや新しいpositive分類は設けない。

[angle table JSON](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/synthesis_angle_table_v1.json)／
[CSV](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/synthesis_angle_table_v1.csv)、
[precision jumps](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/angle_precision_jumps_v1.json)、
[ρ対x/x/3 contrasts](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/rho_angle_contrasts_v1.json)。

### 回路を再構築しないusage帰属

保存eventのrotationはcontrolled pairまたはordinary一primitiveとして既存keyへ対応する。
distinct-basisの追加費用は、保存T−rotation Tの残差を保存±π/8 pair費用で割った非負整数として取得した。
固定loweringではconjugatorが両符号を一組ずつ使い、exact cancellationも一組ずつ減らすという既存規約を参照した。
全2,904 eventで、これら保存primitiveのT和と保存strict-error和が元event recordにexactに一致した。
1Qの残差は非負整数のexact Clifford費用として保存した。

これはsaved-cost residualによるcomponent attributionであり、新IR生成、word reduction、compileではない。
全264 rowの確率加重angle寄与は元E[T]/E[1Q]にexactに一致する。
異なるcontext/arm/precisionのprobabilityを合算した新しいensembleは作らない。
[angle usage寄与](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/angle_usage_contributions_v1.json)、
[row component totals](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/angle_row_component_totals_v1.json)。

## 5. Precisionとbias/shot/Tの交換

[precision curves](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/precision_bias_shot_v1.json)／
[CSV](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/precision_bias_shot_v1.csv)に全132 controlled rowsを保存した。
各arm/x/σについて(B²、coefficient/synthesis bias、remaining budget、n、E[T]、G_T)を並べた。
primaryの全6 arm/x combinationsで、最も厳しい1e-6のG_Tは両coarser登録precisionより高い。
精度を厳しくしたbias/shot側の利益だけでは、増えるT費用を相殺できていない。
未登録precisionの補間、precision選択実行、新しいscalar optimumは作らなかった。

## 6. Q1–Q5への限定回答

**Q1：複数要因の相互作用。** x=1/4,1e-3はnative T費用とbias/remaining-budgetを含むshot側の両方が有利。
x=1/8,1e-4はshot側が悪化してもnative T削減でG_Tが低い。normalization低下中心という説明は不十分。
shot内部の一意なcausal percentageは判断できない。

**Q2：Aは登録precision envelopeに残る。** x=1/8は1e-4、x=1/4は1e-3/1e-4。
1e-6のAは両xでPTSC-K0:1e-6にdominateされる。x=1/4のA:1e-4残留はG_1Q座標に依存する。

**Q3：単一登録precisionだけには限定されないが、離散合成列への依存が見える。**
大きいgainの条件ではρ half-angleが安い保存列を持つ。より厳しいprecisionではこの優位が消える条件がある。
一般的resonance、別synthesizerでの再現、連続curve、counterfactualは判断できない。

**Q4：`SUPPORTS_RA_RTE_DESIGN`。** representationとprecisionを一緒に考える設計入力としての情報価値に限る。
RA-RTEの採択、新規性成立、性能保証、eta最適化や次scienceの承認ではない。
このlabelは利用者が求めた限定的な質的診断であり、新しい数値GO閾値ではない。

**Q5：GPT側の新しい数学設計が必要。** 未登録representation parameterでのcost/bias、feasible classとfair objective、
離散precisionを扱うphase-preserving native resource/bias model、強いsame-target/access対照、
multi-block confidence/correlation、実際の情報取得費用とDF scaleは端点の保存値だけでは決まらない。
R0のrestricted normalization optimumからresource optimumは導けない。新しい手法の新規性も本解析では確定しない。

## 7. 検証・証拠境界・STOP

interval overlap、strict/equal dominance、trade-off、1Q overheadの5つのsynthetic bookkeeping checksを通した。
216 exact G-factorizations、36 primary points、126 sequences、全2,904 eventの保存cost/error寄与を照合した。
[verification](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/verification_v1.json)、
[analysis policy](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/analysis_policy_v1.json)、
[input identity](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/input_identity_v1.json)。
matrix guardやBernstein shot ceilingは再評価せず、保存値・既存semantic contractの範囲に留めた。

R0 `672d6bc667eaa7b9ca4979b012f1530499d701b8`、R0.5 `61dd534567fda5c7348fdc688814089eb26a3561`の
class theorem・先行研究境界は保持する。Euler/common-angleは既知、global LCU最適性・優先性は未確定。
distinct-basis toyはPauli展開可能であり、実問題のI0/I1取得cost分離やDF利益の証拠ではない。
本解析はPOSTHOC design inputであり、独立validation、science rerun、R1の再分類ではない。

資料をcommit/pushして[GPT handoff](r1p5_gpt_handoff_20261006.md)へ戻す。
**mandatory STOP。eta探索、R2、DF接続、追加scienceへ進まない。**
