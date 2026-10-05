# BM-0.5後のTrack B：合成cost・測定負担を含むplacement設計仕様案

2026-10-06 JST。**設計・一次文献対応・保存情報の所在確認まで。実装／pilot実行は未認可。**
利用者が返却したGPT reviewと計画を具体化した一つのreview packetである。
方針・新規性・次の検証scopeの採否はGPT側、承認後の技術作業はCodex側で扱う。

## 1. 受領判断と固定資料

- B-Fの今回の限定仮説をnegativeとして閉じる。原BF-1のINCONCLUSIVEと、R0の事後BF-Aを保持する。
- B-M現adapterのnew-method路線を閉じる。compact BCHとの三次同値性を維持し、旧BM-1の72列案を実行・改名しない。
- B-Mのapplicationとしての価値は保留。研究B全体、multirate、確率的合成全体のno-goとはしない。
- 次の候補は **Synthesis-aware randomization for coherent-signal estimation with factorized Hamiltonians**。
  未実証の候補であり、新algorithm・性能改善・独立論文の成立を採択した状態ではない。

| 読む対象 | 固定identity／公開入口 |
|---|---|
| BF-R0・B-F限定結果 | [6d2645a… result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d2645a09440f50e5b869ef42a1b73a1b625a1af/docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md) |
| BM-0.5同値性監査 | [d55de044… audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d55de044b8e956ba6292209a94bb081014dfdae2/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md) |
| Aの別系列の参照 | [4c23453… claim/evidence map](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c23453c541700c6a41ba71fc5ec9323b53858d6/docs/research/track_a_post_pm2_claim_evidence_map.md) |
| 利用者の計画、原文bytesを保存 | [track_b_post_bm05_research_redesign_20261005.md](inputs/track_b_post_bm05_research_redesign_20261005.md) |
| GPT review、原文bytesを保存 | [post_bm05_gpt_review_20261006.txt](inputs/post_bm05_gpt_review_20261006.txt) |
| 小さい機械可読仕様・provenance | [design_contract_v1.json](../../../artifacts/track_b_synthesis_placement_design/2026-10-06/design_contract_v1.json) |

独立branchは `track-b-synthesis-placement-design-20261006`、基点は
`d55de044b8e956ba6292209a94bb081014dfdae2`。worktreeは
`/home/abe/Project/prt-worktrees/track-b-synthesis-placement-design-20261006`。
必要なtext文書だけをsparse checkoutし、原文二件だけを明示copyした。A/root worktreeは編集しない。
入力元rootは別branch `all-r-coherent-opt2-reoptimization`、受領時HEADは
`e098c54c78f589055082f9cfc2b13de50c90ca94`。計画はそのcommitに含まれる資料とは主張せず、raw SHA256で識別する。

## 2. 狭いRQ、非claim、過去STOPとの差

固定DF/PF/finite-RTEのfull Hadamard wrapperについて、ランダム補間の適用箇所を変えると、
**有限合成bias・全weight・同じ有限confidenceを戻した総非Clifford費用**の有効域／不利域が変わるか。
その選択をgate角度・tail分布・basis負担・workspaceから説明できるか。

RTE+PAIという組合せ、塔則、二次モーメント、cost×variance最小化は新規性としない。
新一般定理を必須にせず、実行可能なprotocolと再現可能な設計知見も候補貢献とする。
**正しいscoreが既知対照と同じ、またはsplitが同じという理由だけでは停止しない。**
BMの現adapterを閉じた理由は同値性に加え、同backend／同情報／同reuseの強い対照に対する
構成・評価cost・情報の差も確認できなかったことである。その旧gateを新RQへ一般化しない。

P-D/B-Fのformula係数・finite objective探索、R3の汎用selector、B-Mのcompact BCH adapterは再開しない。
今回はsimulation列を固定し、compilerと測定負担の相互作用を扱う。既知法の再現だけなら
new-method claimを縮小するが、application/design studyの価値はGPTが別途判断する。
最良PF、新しいheld-out、chemical accuracy、full QPE/RPE、physical runtime、最終分子総costは主張しない。
AのH4 1.00 Åはknown development、1.30 Åは既に評価済み。Bの独立validationは未定。

## 3. 一次文献と比較対照

2026-10-06に以下の**本文の該当節**を確認した。体系的引用網監査ではない。
PRのversionless取得を避け、今回はPDF **2503.05647v2**（44 pages）へ固定した。
Appendix E §3は印刷pp.42–43、roundingの詳細はp.43、式(E24)–(E25)。式(E3)とは別物である。

| 一次資料・確認箇所 | 既知の内容 | この候補が追加し得る内容／未解決 | 比較時点 |
|---|---|---|---|
| [PR v2](https://arxiv.org/pdf/2503.05647v2), App.E §1b/§2/§3a,c | DF cost、同角度の合成、係数roundingと残差のH_R移送、bit grouping／Hamming weight phasing | 固定wrapperのplacement別の有限合成・weight会計。PRの構成を超えるか未確認 | chemistry/task claim前にrounding-to-residual対照が必須 |
| [PAI v2](https://arxiv.org/html/2305.19881v2), §II.2–II.4、App.A | 3-notch channel QPD、符号補正、積normalizationによる期待値推定 | ancilla込みDF wrapperへの正しい実装とplacement選択。PAI自体の発明ではない | primitive pilotから直接適用対照を含める |
| [TE-PAI v2](https://arxiv.org/html/2410.16850v2), §I/II/IV、App.A/B | channel平均による観測量・相関推定、gate/shot trade-off、合成／RUS／catalyst | 同一coherent observable・workspaceへ揃えた比較は未実装 | 広いrandom-algorithm優位を主張する前 |
| [Resource-Optimal IS v1](https://arxiv.org/html/2603.13495v1), §II、Thm.1/§II.1 | 平均cost×weight二次モーメント、cost依存sampling最適化 | catalogue／phase／有限biasが具体的実装でどう効くか | 一般最適化・IS原理は既知。今回はsamplerを追加変更しない |
| [Structure-Aware Variance v1](https://arxiv.org/html/2606.23544v1), §II/IV.4、shot-noise注意 | trajectory varianceの構造分解・stratification | 新compilerの測定shotを含むweight分布の評価 | 新stratificationは採用しない。trajectory-only利得と混ぜない |

PR §3aではbit分解・reordering後のTrotter errorを詳細検証せず、通常値と同じと仮定している。
この仮定の存在は未解決比較条件であり、新手法成立の証明ではない。
初期compiler ablationではsimulation列を固定する。後にPR方式や強いfull deterministic DF/PFと競争する際は、
対照にも同じ合成器・誤差配分・workspace・必要なq/r/K再調整機会を与える。
SPRINTは[既存BM-0の一次claim監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d55de044b8e956ba6292209a94bb081014dfdae2/docs/tracks/algorithm_codesign/bm0_df_information_and_prior_art_v1.md)を参照し、
広いpractical claim前に適用部分を具体比較する。全frameworkを比較済みとはしない。

## 4. 四placementと最初のcompiler手順

DF/state/T/split/PF係数/q/r/K/finite-RTE samplerを全maskで固定する。

| mask | ランダム補間の対象 | 共通で費用に含めるもの |
|---|---|---|
| NONE | なし。全て通常の有限精度合成 | basis、prep、control／identity phase、measurement、workspace |
| D | deterministic diagonal rotation由来のnative Pauli rotationsだけ | 同上。RTE側は通常合成 |
| R | RTE event rotation由来のnative Pauli rotationsだけ | 同上。D側は通常合成 |
| DR | DとRの双方 | 同上 |

basis transformation／inverseは同じdeterministic policyを使い、費用・biasに含める。
fusion前後のroleを追跡する。D/Rをまたぐfusionは禁止する初期案とし、maskを見て都合よく順序を変えない。
各maskは同じnative loweringを使う。NONEだけ未lowering、PAIだけloweringとしない。

既知PAIの最初の実装候補は次の通り。

1. 固定された、phaseを含むfull ancilla wrapperをnative Pauli rotationsとCliffordへlowerする。
2. rotationの元role D/R/basis/phaseを付け、対象maskにだけ3-notch PAIを適用する。
3. conditionalに独立なbranchを選び、符号・normalizationをmeasurement outcomeへ戻す。
4. 対象外rotationとnotch rotationを同じ有限精度Clifford+T合成器へ渡す。
5. 全wrapperのbias、二次モーメント、range、shot、実合成costを集計する。

単にsystem channelをPAI化し、そのsampled unitaryをcontrolする方法は採用しない。
具体的にはPがsystem Pauli、R_P(θ)=exp(-iθP/2)なら、

\[
C(R_P(\theta))=
 e^{-i\theta(I_a\otimes P)/4}e^{+i\theta(Z_a\otimes P)/4}.
\]

この**joint-space**の二つのPauli rotationへchannel QPDを適用すればよい、という設計案である。
両方へ元roleを継承する。Uと−Uの区別、signed time、identityの相対ancilla phaseをlowering前に落とさない。
P=Iの場合はjoint-spaceのglobal identity factorと相対Z_a phaseを分離し、後者を共通phase合成policyへ渡す。
identity channelに不要なPAI overheadを付けず、controlled scalar phaseの費用／biasを落とさない。
sampled primitiveが途中でcontrol sectorを混ぜても、全channelの条件付き平均を検証する。
従来のlogical-block個数だけの誤差boundをnative gatesへ流用しない。

PAIの単一gate係数は上記一次資料の式を使用する。0≤t<Δ<π、下notch Θ、
三つのchannel angle Θ, Θ+Δ, Θ+πに対して、等価な三角式は

\[
g_2=\sin t/\sin\Delta,\quad
g_1=[1+\cos t-g_2(1+\cos\Delta)]/2,\quad
g_3=[1-\cos t-g_2(1-\cos\Delta)]/2.
\]

γ=Σ|g_j|、p_j=|g_j|/γ、sample weight=γ sign(g_j)。ゼロ係数はsampleしない。
これは既知式の適用であり、将来の有限数値実装・semantic testsは未実施。
channel角度の周期化はfull-wrapper lowering後にだけ行う。systemのglobal phaseを先に同一視しない。
basisとinverseへの乱数共有は独立積の式を破るため、初期protocolでは禁止する。
TE-PAIを比較するならancilla込みHamiltonian |1><1|⊗Hの観測量推定等へ揃え、
system channelの一致だけでamplitude比較を成立させない。

## 5. 平均・weight・二次モーメント・range

対象は z(T)=<ψ|exp(-iHT)|ψ>、Pr(|ẑ−z|>ε)≤α。
軸aはRe/Im。outer finite-RTE trajectoryをω、補正normalizationを𝔅>0、
そのideal wrapperの±1 outcomeをY_a、条件付きmeanをμ_a(ω)とする。
𝔅は今回の固定構成ではtrajectoryに依存しない。ν_a=𝔅 E_ω μ_a(ω)はfinite targetで、exact z_aとは別。

PAI branch ξの符号s、Γ(ω)=Π_i γ_iについて必要な条件は

\[
 E_{\xi,Y}[\Gamma(\omega)s(\omega,\xi)Y_a\mid\omega]=\mu_a(\omega),\qquad
 Z_a=\mathcal B\Gamma sY_a,\quad E Z_a=\nu_a.
\]

既知のchannel compositionと塔則であり、新定理ではない。conditional independence、
normalizationとsampling係数の一致、全wrapper phaseを前提とする。
canonical samplingでY_a²=1なら、合成後も二次モーメントは

\[
V_{2,a}=E Z_a^2=\mathcal B^2 E_\omega\Gamma(\omega)^2,\quad
\operatorname{Var}Z_a=V_{2,a}-(E Z_a)^2,\quad |Z_a|\le M_a=\mathcal B\Gamma_{\max,a}.
\]

finite synthesis後のmeanはν̃_aであり、そのvarianceはV₂,a−ν̃_a²。理想時だけν_aを使う。
general samplingでは実際のW=𝔅 g/pを用い、E[(WY_a)^2]とrangeを計算し直す。Wに𝔅を二重に掛けない。
quantum shot noiseを含む式であり、Var_ω μ_aだけをshot規則へ代入しない。
EΓ²を(EΓ)²やΓ(E length)²へ置き換えない。
例えば同じγを持つn個の独立microstep、length分布p_lなら
EΓ²=Γ_D²(Σ_l p_l γ^{2l})^n。角度・basis fusion・相関がある場合はjoint分布を使う。

既存B-Sの有限32標本に対するIS plug-in診断は新compiler／samplerのpopulation上界ではない。
この設計でその値を再集計・転用せず、headroomなし／ありの結論も出さない。

## 6. 有限合成bias、共通finite-confidence、総費用

ideal PAI notchと、実際に有限Clifford+Tで合成したnotchを分ける。
dyadic angleを無料・exact Clifford+Tとは扱わない。実装証明のあるexact Clifford/T angleだけ例外を記録する。
full-wrapper observable normは1。sample ξのobservable error上限e_a(ω,ξ)なら、

\[
 b_{\rm synth,a}\le E_{\omega,\xi}[|W|e_a(\omega,\xi)].
\]

全primitiveのdiamond distanceをδ_jと定義すればe_a≤Σ_j δ_jという保守的案を使える。
unitary operator distance η_jを用いる合成器ではδ_j≤2η_jへの変換を明記する。
prep／basis／relative phaseも含める。coefficient・probability・normalizationの有限数値誤差はu_aへ別計上する。
certificateのない「実測biasが小さかった」をそのままa priori boundにしない。

common shot規則案は**全mask同じBernstein十分条件**とし、結果後にHoeffding等へ切り替えない。
軸許容誤差ε_a=ε/√2、α_a=α/2、
s_a=ε_a−b_PF/RTE,a−b_synth,a−u_a>0。
V̄₂,aはpopulation二次モーメントの結果前上界、M_aは結果前range上界である。
各shotでω,ξをfreshに独立生成する。固定trajectoryの多shot再利用はこのIID規則のscope外。
|Z−EZ|≤2M、Var Z≤V̄₂から、二側Bernsteinで

\[
n_a=\max\!\left(1,\left\lceil
 \frac{2\overline V_{2,a}+\frac43 M_a s_a}{s_a^2}
 \log\frac{2}{\alpha_a}\right\rceil\right).
\]

両軸のunion boundとbiasの三角不等式で上のcomplex taskへ接続する、という保守的仕様案。
真のνやoracle varianceを新方式だけに与えない。sampled momentだけで上界扱いしない。
bound未取得またはs_a≤0なら「評価不能／不適格」。数値guardによる未確定と科学的negativeを分ける。

primaryは**実合成Clifford+T sequenceのT/T† count**：

\[
G_T=C_{\rm init,T}+\sum_a n_a E_{\omega,\xi} C_{T,a}(\omega,\xi).
\]

全wrapperのprep/basis/phase/measurement関連費用を含める。初期化のshot毎／batch毎／reuse回数を明示する。
T/CCZ/Toffoliを無条件換算しない。depth、workspace、classical生成費用、RZはsecondary。
catalogue間でworkspaceが違えば制約を揃えるか別frontierとして報告する。
設計量J=V₂ E C_T、cost比g・moment倍率hのgh<1は既知の粗い損益条件。
finite bias・range項・shot切上げ・initを戻したG_Tの改善と同一視しない。

## 7. Primitive catalogueと小さいrecord schema

最初の候補catalogueは、ancilla-freeのdeterministic Clifford+T RZ合成を使い、
通常角度とPAI notchを**同じ合成器**で処理する。
[Ross–Selinger](https://arxiv.org/abs/1403.2975v3)は選定候補の一つ（今回はabstract/metadata確認だけ）。
repoの限定したtracked source/dependency検索ではその合成器bindingを確認していない。
採用実装・版・seed・保証を未選定のままT-countを捏造しない。
同じ実装による二つ以下のnotch catalogueを候補とし、bit resolutionはまだ固定しない。
RUS/catalyst/phase-gradientはknown strong optionsだが、初期pilotへ自動追加せず別review事項とする。

一つのrecordは少なくとも次を保存する（小さいJSONにfield listを収録）。

| 群 | 必須情報 |
|---|---|
| identity／angle | template/axis/mask、元D/R/basis/phase role、native位置、signed angleの厳密表現、support、control／relative phase、fusion lineage |
| sampling／moment | outer finite分布identityと𝔅、notch/code/Δ、g/p/sign/γ、Γ、W、V₂上界とrangeの導出identity |
| synthesis／error | 合成器版/source、precisionとnorm、sequence identity、δ、coefficient数値guard、全wrapperのbias集約 |
| resource | actual T/T†・Clifford・depth・workspace、state/init cost、reuse範囲、classical生成cost、shot数とG_T |

angle histogramだけでは順序／fusion／phase／joint momentを復元できない。
旧A compile fingerprintは角度列そのものではない。schema field listは結果schema完成やscience source freezeを意味しない。

## 8. 保存情報の所在確認（静的、再生成なし）

固定A参照 `4c23453c541700c6a41ba71fc5ec9323b53858d6` のgit text/JSONだけを確認した。
以下は**確認したpathに限る**。全repo、untracked runtime、外部保存の不存在を証明していない。

| path／該当箇所 | 確認したもの | angle-level inventory判断 |
|---|---|---|
| `src/trotterlib/df_trotter/ops.py:18–24,233–316` | global_phase/rz/rzzのprimitive型と生成・controlled化 | angleを作る能力。保存済み全wrapper列の存在証拠ではない |
| `src/trotterlib/rte.py:623–643` / `df_rte_qiskit.py:463–501` | eventのto_dict、signed rotation、RZ/RZZへの変換 | event-level serializable情報の能力。basis native angles／fusion後inventoryは別 |
| `src/trotterlib/rte_compiled_cost.py:524–542` | SQLite payloadの静的定義：gate counts/global_phase/fingerprints | gate別angle列はこのpayloadにない。SQLite自体は開いていない |
| `src/trotterlib/pr2_matched_accuracy_m1_b1_execution.py:334–407`、M1-B1 result schema v2 | 集計axis record・seed・metricsの保存方式 | schemaでordered gate-angle recordを確認できない |
| `src/trotterlib/pr2_matched_accuracy_m2_transfer_execution.py:431–438`、M2 result v2 | wrapper metrics、seeds、semantics/circuit fingerprints。JSON keyだけの確認 | 解析対象JSONにgate別角度・primitive catalogueはない |
| `artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/summary.json` | summary keysにcontrolled_diagonal_primitives等の監査項目 | 個々のgate angle一覧は確認できない |

M1 schema full pathは `artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/pr2_matched_accuracy_m1_b1_result_schema_v2.json`、
M2 JSONは `artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json`。
確認範囲とgit blob identityは小さいJSONに記録する。数値の再採点・再分類はしていない。
したがって現在の設計入力は**ANGLE_INVENTORY_NOT_ESTABLISHED_IN_INSPECTED_RECORDS**。
aggregate RZを角度別TやΓ²へ換算せず、将来の角度取得が必要なら別認可とする。

## 9. 分子なしの最小pilot設計案と予算の閉じ方

**未認可の案**：最大4 control template ×4 mask=16基本比較、catalogueは最大2。
empty D/Rのmaskもnegative/duplicate controlとして残し、独立な改善例に数えない。

| template | 調べるもの | 判定上の注意 |
|---|---|---|
| T1 single controlled Pauli rotation | notch mean、signed weight、実合成cost | D/Rのrole割当を結果前固定する |
| T2 two noncommuting controlled rotations | joint wrapper mean、積weight、順序 | basis/prepと参照stateを全maskで共通にする |
| T3 finite event-length distribution pair | 同じ平均長でもEΓ²が変わること | length-only moment control。異なる分布のtargetが同じとは仮定せず、G比較は各分布内のmask間だけ |
| T4 phase/basis inverse control | +U/−U、正しい独立sampling、誤った共有の反例 | 共有sampleをproduction protocolへ入れない。phaseを落とすwrong controlも記録 |

angle/state/finite probabilities/notch bits/precisionを公開primitive条件から結果前固定する。
最初は同じtemplateで全mask共通の保守的合成precisionを使い、原始placement効果を切り分ける案。
後の実用競争では各対照へ同じprecision allocation最適化機会を与える。
PAI直接適用の正しいwrapper対照は必須。mask違いを新PAI原理と呼ばない。

予算は旧BFのwall上限を流用せず、合成器選定後に次の有限ledgerから決める。
初期案は各templateの補間対象native位置を最大4、全rotation位置を最大6とする。
これなら1 cellのbranch enumerationは最大3^4=81。
outer pathをtemplate当たり最大4とする案では、16×2 catalogue×2 axis×4 pathで
最大20,736 branch-axis評価。複数分布で同じpathを使う場合は条件付き値だけreuseする。
phase/correlation反例の追加probe・参照stateの数は別枠で結果前固定する。
全mask・両axisでtemplateごとのprecisionを共通にし、角度×precision×合成器をkeyとしてreuseする案では、
最大4 template×4 path×6 position×(元角度＋3 notch)=384 synthesis key/catalogue、合計768。
maskごとにprecisionを変える設計へ改訂する場合は、このkey上限も結果前に改訂する。
これは上限計算の案で、合成を行った回数／実行時間の測定ではない。
内蔵探索iteration、compiler seed、timeout時のSTOP、cache reuse、全CPU/RSS/output上限は別に固定する。
source-bound contractに数値wall/CPU/RSS/出力上限・materiality・guardが入るまでRUN_READY=false。
第二catalogueは必要性をreviewしてから固定し、失敗した結果を見て追加しない。

## 10. 次のreviewと停止位置

GPT reviewで閉じる項目は、(1)初期PAI/native lowering仕様、(2)実際の合成器と最大二catalogue、
(3)四templateの具体inputと完全budget、(4)materiality／未確定／STOP基準、(5)独立研究としての必要性。
同値性確認だけ、理想angleを任意costで採点した利得だけでは次段への根拠にしない。
同じtaskでweight・有限bias・実costを戻した利益／制約が残るかを小さく判別する。
不利な結果からmask／precision／geometryを増やしてpositiveを探さない。

正式実装を採用する場合は `src/trottertracks/algorithm_codesign/synthesis_placement/` の新namespace、
新runner/test/artifactへ置く。共有APIと旧B-F/B-M/A evidenceは変更しない。
今回の小さいJSONは設計状態で、authorizationではない。
実装・限定semantic tests・source freeze・pilot実行は、それぞれ採用scopeの明示reviewを経る。
現在はNPZ操作、science signal、trajectory sampling、circuit build/compile/synthesis、GPU、testsを全て0で終了する。
必要資料をこの独立B branchへcommit/pushした後に**STOPして利用者/GPT reviewへ戻す**。
