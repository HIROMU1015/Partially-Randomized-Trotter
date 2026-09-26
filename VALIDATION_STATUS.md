# Validation status

## 2026-09-26 finite-RTE phase/radius FR-1

事前登録した2×2 fixed gridを変更せず実行した。33条件、99状態評価、495個の適用可能method recordで
位相上界・信号半径下界の違反は0だった。paired-eventのordinary平均最大residualは
`4.4431e-16`、controlled relative-phase最大residualは`3.3379e-16`である。負時間、K=4、
非対称配置を含め、G0/G1/G3/G4は通過した。

主判定の`analytic_mixture_state`へ利用可能な$\underline\rho=0.8$だけを与えると、
`PROPOSED-AVAILABLE`対`STRONG-NORM`の位相上界比は最小0.631998で、固定50%基準を通らなかった。
提案法だけが`1e-3 rad`を認証する点もなくG2は不通過。一方、$\rho=1$またはdense真値を使う
`PROPOSED-REF`では改善したため、事前規則どおり`GO_FR2_MECHANISM_ONLY`とした。
これは利用可能情報による実用的GOでなく、FR-2は開始しない。

artifact fingerprintは`d96b200163f3a432652656ae97c65c837e01fe81323cd69528373dff16ce6152`、
file SHA-256は`6a81a0ba6e39ba0f5d79ba026a2a45c3c65709e071e9c6fa5606b3e99a89b0f7`。
完全な実行契約は凍結FR-1事前登録SHA-256
`bc8066d7a31a3f46f476d2c591c7f0dad5e6f49023fc261dc518b61e2ec600a3`である。凍結親契約は後の監査で
末尾切断が判明したため不変の監査履歴として残し、現行親契約で修復したが、FR-1のgrid・gate・判定には影響しない。
専用testは`4 passed`、全suiteは`619 passed, 2 skipped, 4 warnings`で失敗0。artifactはdirty worktreeで生成したlocal evidenceであり、immutable CIではない。
H4/H12、circuit compilation、Monte Carlo sampling、RPE総cost、最終costは0件。詳細は
[FR-1検証](docs/finite_rte_phase_amplitude_validation.md)。

事前検証カタログの実施ID・work packageと、以下のstatus、専用文書、artifact、runner、testの対応は
[事前検証カタログ実施証拠索引](docs/research/prevalidation_catalog_evidence_map.md)を参照する。

## 2026-09-26 P-D S1 固定artifact事後再解析

S1 v2のfingerprintとfile SHA-256を固定し、保存済み308候補だけを再集計した。一次の
`Case B + undetermined_boundary`と停止statusは変更していない。主baselineをB1b/B2/B4、B0/B1aを
診断用ablationとし、各scopeの選択、regret、false acceptance、B4最良から5%以内、B1aの`m_D`列、
nested/native work内訳を別schemaへ保存した。

B1b/B2/B4はnested/native/combinedの各scopeで同一候補を選び、false acceptanceなし、B4 regret 0。
5%近傍はnested 5候補、native/combined各1候補で、B2 objective相対誤差は最大0.03132%だった。
従ってこの固定候補集合とB4参照に限り、主baselineをCase A相当と事後解釈する。ただし全候補では
B2 proxy受理/B4不適格が45件あり、全域のfinite feasibility判定能力は支持されない。

B1a選択の固定tailはunit signal radiusでもfinite位相上界`1.8178593e-6 rad`が予算`8e-7 rad`を超え、
`m_D`だけの追加では採用中のB4上界を満たせない。nested/native B4 objective比13.2297は主に
deterministic action差だが、解析的component-action proxyなのでcompiled circuit優位性とはしない。

artifact fingerprintは`976212ee45a472bf0091064e8baf3eb7a861f2c240cdcfda445e64b4c72e0245`、
file SHA-256は`d8657c52e609e524c4e43a3004948ffb3fd54e6f60219442638b847ade3e699b`。専用testは
`4 passed`、全suiteは`615 passed, 2 skipped, 4 warnings`で失敗0。新しいHamiltonian、対角化、RTE sampling、compile、H12、長RPE、最終総costは0件。
P-D S2には進まず、R3は先行研究差分と別契約を固定する前の候補段階である。詳細は
[P-D S1事後再解析](docs/research_direction_pd_s1_posthoc.md)。

## 2026-09-26 P-D S1 公平PF再最適化

S0で主RQ、既知baseline、共通比較契約、停止規則を固定した。S1はH4 linear chain、1.0 Å、
STO-3G、8 qubit、4-electron sector、DF rank 12、`L_D=3`、固定5公式を対象とし、共通物理時間
`T=0.8`、総位相誤差予算`8e-7 rad`でnested/native構成を比較した。primary gridは
`delta={0.1,0.2,0.4}`、nested `m_D={8,16,32,64}`、`R={16,32,64,128}`、K2で、
事前登録規則による一段境界延長と限定K4感度だけを追加した。

one-shot workを含むB1b、leading absolute-tail-time modelのB2、finite modelのB4は全scopeで同じ
new fourthを選んだ。nestedは`delta=0.2,m_D=16,R=16`、native/combinedは
`delta=0.2,R=16`で、B2のB4 regretは0だった。限定K4でもformula・delta・m_D・Rは不変で、
decision-relevantなfinite補正は確認されなかった。

outer-stageだけを見るB1aはnested/combinedでfinite-infeasibleな`delta=0.4,m_D=128,R=16`を
選び、一段延長後も`m_D`上限依存が残った。従って一次分類はCase Bだが、正式statusは
`stop_s1_undetermined_boundary_no_go_decision`である。Case Bを生じさせたのはB1aだけで、B1bは
B2/B4と一致する。Case C/Dの証拠はなく、S2へ進まず計算を停止した。

v1本実行は高段PFでtail occurrence数が`R=16`を超える配分不能点をinfeasibleとして保存せず、
result生成前に停止した。grid・閾値・分類規則を変えず修正し、v1 expectedを保持したままv2へ
非上書きで再固定した。v2 expected fingerprintは
`e3eacbb9d8928f781df7709208f048c59aaab7052304e4eae4adc346bbf6d0d5`、result fingerprintは
`0ba7764da7b7d8b7e195a5c315d3dc0a65c2c79ce01c51cf021a2685977487d2`、result file SHA-256は
`6b8de6e255eb0796d93398c767017c2899837a63c7230e9956fb2d9beedbeaec`。専用testは`5 passed`、
全suiteは`611 passed, 2 skipped, 4 warnings`で失敗0、artifact validatorも通過した。これはdirty worktreeのlocal evidenceで、immutable CIまたは外部
独立再現ではない。H12、長RPE、compiled total cost、sampled H4 finite operatorは未評価である。
詳細は[P-D S1公平再最適化](docs/research_direction_pd_fair_comparison.md)。


## 2026-09-25 P-D現実化 Go/No-Go gate

H4 linear chain、1.0 Å、STO-3G、8 qubit、4-electron sector、DF rank 12、固定5公式、
診断delta 0.2、主判断delta 0.4を用いた。D1負時間14 task、D2 `L_D=3,4`、D3 fresh
`L_D=5`を結果前に固定し、energy tolerance `1e-6 Ha`、minimum weight 0.9995、tail burden
20%削減、内部`H_D` 32 substep等の閾値を結果後に変更していない。

D1は14/14通過した。ordinary oracle residual最大`3.3314e-16`、signed adjoint 0、controlled
residual最大`6.6613e-16`、identity relative-phase residual最大`3.4694e-18 rad`、sampled mean
最大絶対誤差`5.1531e-4`、最大standardized residual 2.4874だった。

fragment内部`H_D`誤差を戻した主判断deltaでは、`L_D=3,4`のenergy-only選択はMorales 8次、
tail-aware選択は新4次のまま残った。fresh `L_D=5`ではenergy-onlyがMorales 8次、tail-awareが
二次となり、`log B_K`を96.580%、stage proxyを6400から768へ減らした。全splitのfragment再構成
residualは0、最大unitary defectは`1.4382e-11`、最小target weightは`0.9999959245`である。
D1--D3は全て通過し、statusを
`advance_pd_to_formal_primary_candidate_then_stop_for_research_redesign`とした。

P-Dを正式主研究候補としてRQ・新規性・最小着地点・必要な本検証の再設計へ進めるが、計算は
ここで停止する。D1はdense small-matrix oracleでcompiled Qiskit controlではない。D2/D3の
各`H_R` occurrenceはexactで、sampled H4 operator、compiled depth、long RPE、最終総cost、H12、
backend/noise、global PF optimality、科学的優位性は未評価である。

v1本実行は旧P-D負担表にないdelta 0.2を参照する技術的`KeyError`でresult生成前に停止した。
task、seed、thresholdを変えず修正し、v1 expectedを保持したままv2へ非上書きで再固定した。
v2 expected fingerprintは`8924d637e52b03900f32e2f167e63593cf9729b4b78d4dee3af87fca661183f4`、
result fingerprintは`805a17f95497a4d61286748a126c01b1235fbe0d987528be86ea3938700b9ede`、
file SHA-256は`59e8019805e429280a5f6338e38a1cbfd803ae8b51c30e87a965a2ef3dcb694d`。
専用testは`4 passed`、全suiteは`606 passed, 2 skipped, 4 warnings`で失敗0。これはclean
worktreeから生成したlocal evidenceだが、immutable CIまたは外部独立再現ではない。詳細は[P-D現実化Go/No-Go](docs/research_direction_pd_realization.md)。

## 2026-09-25 P-D energy係数・random-tail負担Pareto監査

H4 linear chain、1.0 Å、STO-3G、8 qubit、4-electron sector、DF rank 12の固定snapshotで、
二次、標準/新四次、Yoshida/Morales八次の5公式を比較した。`L_D=3`は探索を開示したdevelopment、
expected-task固定前に未確認の`L_D=4`をblind holdoutとした。非可換3次元toyの局所/global
operator次数と、`H_D`内部だけの高次化が外側Strangの二次を変えないことも確認した。

`delta=0.4`、energy tolerance `1e-6 Ha`では、両splitともenergy-only選択は
`8th(Morales)`、tail-aware選択は`4th(new_2)`となった。新4次はMorales 8次より
`Gamma_R`が56.702%小さく、full exponential stage数も35から11へ減る。係数・次数、
nested/global区別、finite-RTE配分、development/blind逆転、unitarity、target weightの7 gateは
全て通過し、statusを
`advance_pd_as_conditional_candidate_pending_signed_time_and_inner_hd_validation`とした。

これはexact dense `H_D/H_R`二blockと解析的finite-RTE normalizationのlocal dirty-worktree
evidenceである。負のtail係数を含むsampled operator、identity/control位相、実際のfragment列での
`H_D`内部誤差、compiled回路、RPE総cost、H12、全PF family最適性、科学的優位性は未評価である。
従ってP-Dは正式主題ではなく条件付き候補で、次はsigned-time RTE oracleと内部`H_D`誤差だけを
検証する。

expected fingerprintは`f302dafce37fb90f3acfe32aa83edf563dd1609015a1b3d50972880e13407c7d`、
result fingerprintは`846362ab808e9b26e5648f7f9d12d541dd01a6f9c2954e45d9194b4dfef8d835`。
専用testは`4 passed`、P-D追加後のlocal全suiteは`602 passed, 2 skipped, 4 warnings`、失敗0。
これはimmutable CIまたは外部独立再現ではない。詳細は
[P-D energy・tail Pareto監査](docs/research_direction_energy_tail_pareto.md)。

## 2026-09-25 P-C geometry tracking・breakdown validation

H4 linear chain、STO-3G、8 qubit、4-electron sector、DF rank 12、`L_D=3`、
二次partial-`S_2` exact-tail参照を固定し、0.70--1.60 Åの8 geometryでindependent/tracked
prefixを比較した。trainingは0.80/1.00/1.20 Å、blindは0.70/0.90/1.10/1.40/1.60 Å、
fit deltaは0.025/0.05/0.10、holdout deltaは0.20である。compile-before expected taskは16件、
本計算は16/16完了した。

tracked prefixは全点で独立先頭3 fragmentと同じだった。blind coefficient予測の15%基準は3/5点、
pair予測の15%基準は1/4 pairだけが通過した。0.90/1.10 Åの係数誤差は6.135%/3.461%だったが、
1.40/1.60 Åでは33.653%/123.245%へ増えた。分類可能blind 4点のcontinuity診断正解率は50%で、
stretch側2 breakdownをflagできなかった。

固定7 gate中、representation integrity、delta holdout、nontrivial cancellationの3 gateだけが通過した。
thresholdを変更せずstatusを`stop_pc_current_h4_family_as_primary`とする。局所P-C pilotの
0.80--1.20 Å結果は保持するが、同じH4 pathへの点追加で主張を復活させない。P-A interval、
P-B current grid、P-C current H4 familyはいずれも停止点に達し、現時点でA/B/Cに確認済み主題はない。

expected fingerprintは`cbe260750d081316070d3684a2a24194d91d029ad700a54cf4f61152f2ed4a4e`、
final fingerprintは`26845effe8efda56390aabdf9e40d61fa3a033e3ac7e6ff6911e2e124156f07a`、
final file SHA-256は`58e31d3877d45e91e9c6c9f4238d2c1f876f02bb875643813a126f283f2dd2b7`。
専用testは`4 passed`、訂正済み先行P-Cと合わせて`7 passed`、関連testは`14 passed`。
全suiteは`598 passed, 2 skipped, 4 warnings`で失敗0だった。
結果はlocal dirty-worktree evidenceであり、immutable CIまたは外部再現ではない。
詳細は[P-C geometry tracking・breakdown validation](docs/research_direction_geometry_tracking_breakdown.md)。


## 2026-09-25 P-A nondegenerate mechanism validation

形式化監査後に事前登録した明示的`one_segment_per_source_run` baselineとの比較を完了した。
固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、`L_D=3`、$\delta=0.02$、
Qiskit 1.3.0 opt1で、training fragment 3/5/7とblind fragment 4/6/8を分離し、5種類の
forced-support profileを全てTaylor order 2で評価した。

training 15、blind 15の全30 taskで`interval_union_dp`は一区間baselineと同じplanを選び、
run内分割、plan変更、RZ改善rowはいずれも0件だった。blind pooled RZは4,378対4,378で、
RZ depth、CX、depth、circuit sizeも完全に一致した。5 operator probeの最大差は
$8.327\times10^{-16}$でrelative ancilla phaseも一致した。

事前固定した7 gate中、order-2 coverage、最大個別RZ悪化、operator同値性の3 gateだけが通過し、
分割のbasis/profile移送、25% plan変更、2% pooled RZ改善の4 gateは不通過だった。thresholdを変更せず、
statusを`stop_pa_interval_dp_as_primary_and_return_to_pc`とする。P-Aの既存run-level
full/support-union改善は保持するが、interval subdivisionを独立研究寄与として主張しない。
この時点ではP-Cへ戻った。後続tracking・breakdown validationによりcurrent H4 familyのP-Cも
固定停止条件に達した。

expected-task fingerprintは`e8b064e9821fae5c2e7a44d0c98d3f9a0ed973a4cd87945fb051151e446a96fc`、
final artifact fingerprintは`fdc89974e89a4a6809cecd2c5608a36d684d40d76d9b3055fbbe6ec9276abbaf`。
詳細は[P-A非退化mechanism validation](docs/research_direction_joint_synthesis_mechanism_validation.md)を参照する。

## 2026-09-25 P-A v1 formalization / mechanism audit

凍結済みP-A v1について、有限候補問題、4成分の辞書式目的、DP漸化式、計算量、operator同値性条件を
形式化し、blind artifactの48 holdoutと6 operator probeを再解析した。DP遷移数は全recordで
$\sum_r n_r(n_r+1)$と一致し、保存された選択目的もsegment metadataから完全に再構成できた。

ただし、全54 recordでselected segment数はsource-basis run数と一致し、run内部を複数区間へ分けた
recordは0だった。H4/H5 holdoutの256 eventは全てTaylor order 0で、6 probeもapplication数から
非零orderを含まない。従ってblind検証が直接支持するのはrunごとのfull/support-union選択であり、
明示的なone-segment-per-source-run baselineに対するinterval分割の増分利益は未識別である。

本監査完了時点のstatusは
`pa_v1_formalized_but_interval_mechanism_not_empirically_distinguished`、
P-Aは`conditional_candidate_pending_nondegenerate_mechanism_validation`とした。既存blind gate通過と
H5/H4のRZ改善値は有効だが、interval分割の効果または非零Taylor-order移送を主張しない。次は
one-segment baseline、support変化を持つforced run、Taylor order 2を含む小さいmechanism判別を
事前登録する、とした。後続P-A検証は完了してinterval DPを停止し、その後のP-C tracking検証でも
current H4 familyが停止条件に達した。現行statusは本書先頭のP-C節を優先する。
H12、長RPE総cost、full wrapper、backend/noiseは不要である。

artifact fingerprintは`aaa5fdba8ddc6ec25fe1f286d886aba14a7a440c435f1ca5dd78ce33404b3676`。
詳細は[P-A v1 DP形式化・mechanism監査](docs/research/pa_joint_synthesis_v1_formalization.md)を参照する。


## 2026-09-25 P-A v1 blind transfer validation

事前登録した固定v1、4 policy、6 gateを変更せず、未使用H5 physical snapshotと元H4 event streamの
Qiskit optimization level 2へのpaired compiler transferを実行した。holdoutは48/48、operator probeは
6/6完了し、両stratumで全6 gateが通過した。H5では現行policy比pooled RZが-17.076%、最大個別悪化
0%、4-policy oracle regret/full RZが0%、basis列変更率95.83%、operator最大残差が
$2.998\times10^{-15}$だった。H4 opt2ではそれぞれ-6.598%、+0.265%、0.0116%、87.5%、
$3.126\times10^{-15}$だった。

blind gateだけに基づくこの時点のstatusを`advance_pa_v1_to_formal_primary_theme_candidate`とした。
これは文献上の新規性証明、全Gaussian circuitに対するglobal optimum、
coupling/noise/backend、full partial-$S_2$ wrapper、RPE総cost、H12または科学的優位性の検証ではない。
後続形式化でmechanism範囲を狭めたため、現行statusは直前節を優先する。

final artifact fingerprintは`78af3474898dbf989780ea5f2881cb5b61595609dd2698164b9846c1ce1c5919`、
file SHA-256は`ff6a8f846795b3f56e3688c62eab3ad26c3ace6632063ba72ff97f4a12ac4ea3`である。
詳細は[P-A v1 blind transfer validation](docs/research_direction_joint_synthesis_blind_validation.md)を参照する。
結果はlocal dirty-worktree evidenceであり、immutable CIまたは外部再現ではない。
## 2026-09-25 P-A scoped prior-art audit / blind preregistration


P-A v1について、DF/low-rank回路、partial basis rotation、fermionic Gaussian/Givens合成、
隣接network融合、completion自由度、DP/block synthesisを対象にscoped prior-art auditを行った。
各構成要素は既知だが、明示した検索範囲では、同一source-basis runを区間分割し、full basisまたは
support-union completionを選ぶ現行v1と同じ組合せは確認できなかった。これは網羅的な新規性証明、
特許調査または査読上の新規性判定ではない。

statusを`provisional_pending_blind_validation_after_scoped_prior_art_audit`へ更新した。次の計算は
事前登録済みの2 stratum、すなわち未使用H5 physical snapshotへの移送と、元H4 event streamの
Qiskit optimization level 2へのpaired compiler移送に限定する。両stratumは同じ4 policyと6 gateで
別々に判定し、どちらかが不通過ならP-Aを正式主題化せずP-Cへ戻る。P-A v2、H12、長RPE総costは
このblind検証へ含めない。この事前登録の固定時点では両stratumとも未実行だったが、後続のblind
transfer validationで両方が完了・通過した。

## 2026-09-25 P-B/P-C/P-A theme selection

3件のfingerprint済みpilotを4問で比較した。P-Bは現H4 gridで実用的なenergy/signal選択差がなく停止、
P-Cは未使用geometry/deltaの差分bias予測gateを通過、P-Aは強いproject baselineに対して未使用列長で
compiled-cost改善とoperator同値性を示した。

選定はP-Aを暫定主題、P-Cを副候補、P-Bを現範囲で停止とする。その後のscoped prior-art auditでは、
各構成要素は既知だが現行v1と同じ組合せを検索範囲内で確認できなかった。これは新規性の証明ではなく、
この時点のstatusを`provisional_pending_blind_validation_after_scoped_prior_art_audit`とした。その後、
事前登録済みH5 physical transferとH4 optimization-level-2 compiler transferは両方とも全gateを通過した。
H12、長RPE総cost、追加$q>32$は次の必須作業ではない。selection artifact fingerprintは
`7daa49c3c30b453d69d52830d0889ce04104cc5a0f02a2c3083517486d575fbb`。

## 2026-09-25 P-A interval-aware joint synthesis pilot

固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、`L_D=3`、$\delta=0.02$、K=2、
topology-free Qiskit 1.3.0 opt1で、full共有、event-support、現行`support_run_le_1`、interval-union DPを
比較した。training列長2/4/6とは独立なholdout列長3/5/8を各8 trajectory評価した。

DPのpooled RZは現行比-7.194%、各長で改善、最大個別悪化+0.231%、4候補内oracle regret/full RZは
0.00482%、basis列変更率87.5%、operator最大残差$3.126\times10^{-15}$だった。事前固定した6 gateは
全て通過した。full wrapper、coupling/backend/noise、RPE総cost、科学的優位性または外部文献上の新規性は
評価していない。artifact fingerprintは
`1a9840a4ee46daa6e3749272593acf3e8ae29be9fcecc1a05b0bd9ea817fbc37`。専用testは`3 passed`、全suiteは`582 passed, 2 skipped, 4 warnings`で失敗0。

## 2026-09-25 P-C geometry signed-error pilot

H4 linear chainの0.80、0.85、1.00、1.15、1.20 Åについて、STO-3G、8 qubit、DF rank 12、
`L_D=3`、二次partial-$S_2$を固定し、geometry間エネルギー差の符号付きPF biasを評価した。
0.80/1.00/1.20 Åをcoefficient training、0.85/1.15 Åをgeometry holdout、$\delta=0.4$をdelta holdoutとした。

geometry holdout coefficient誤差は最大4.409%、delta holdout誤差は最大0.840%。0.85→1.15 Å、
$\delta=0.4$の二重holdoutではactual/predicted difference biasが-0.00169849/-0.00178045 Haで、
endpoint-bias正規化誤差は2.539%だった。coefficient spanは63.303%、$\delta=0.1$隣接pairの
最小cancellation ratioは9.786%で、固定した5 gateを全て通過した。

従って案Cを次段へ残した。後続tracking・breakdown validationではcurrent H4 familyを固定条件で停止した。
これは0.80--1.20 Åの単一H4/DF/PF条件に限るlocal dirty-worktree結果で、potential-energy surface、
別系移送、RPE/RTE cost、回路compile、最終総costまたは科学的優位性の評価ではない。

初版統合artifactのexact-energy欄はsurrogate値を誤って転記していた。訂正版ではfull
DF-rank-12 Hamiltonian値へ直したが、PF bias、係数、holdout予測、5 gateは変わらない。
訂正版fingerprintは`3c199dbc20d0892cf1bcfad4b27646ea8c90d1c1d32fc5e2c75be610bb1fe611`。
専用testは`3 passed`、当時のPF/P-B/P-C関連は`10 passed`、全suiteは
`576 passed, 2 skipped, 4 warnings`で失敗0だった。
詳細は[P-C geometry signed-error pilot](docs/research_direction_geometry_energy_difference_pilot.md)。

## 2026-09-25 P-B energy-bias / target-weight pilot

研究テーマ選定の最初のpilotとして、固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12の
`pf_delta_validation_v5`を再解析した。二次partial-$S_2$、$L_D=0,...,11$、6 delta、$q=1,2,4$の
12 artifact・72候補を使用し、新しいHamiltonian/PF計算または回路compileは行っていない。

screeningの最小target weightは0.9999803239、最小q別signal半径は0.9999744904で、全候補が通過した。
各deltaのenergy-only最良とsignal-screened後のenergy最良は全て`L_D=0`で一致し、選択不一致は0/6。
ordering inversionは118組あるが、最大leakage比1.00349で、意味のあるinversionは0だった。

従って、このH4・二次partial-$S_2$・prefix gridではenergy-only基準の実質的なsignal failureを
確認できず、案Bを主題へ進めない。これは他PF family、小gap、別系に対する一般的な棄却ではなく、
weight問題を区別するstate-action診断も未検証である。後続のP-CとP-Aを含む比較では現範囲の停止を維持した。

artifact fingerprintは`e7b6e98d1a4049f5e061c4ec47de891edf40689ee9ccfc03f62c9022ea350197`、専用
testは`3 passed`、全suiteは`573 passed, 2 skipped, 4 warnings`で失敗0だった。詳細は
[P-B signal-weight pilot](docs/research_direction_signal_weight_pilot.md)。
新しい結果はlocal dirty-worktree解析であり、immutable CIまたは外部再現ではない。

## 2026-09-25 A0 fresh-proxy / legacy-holdout再照合

新規compileなしで、固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、
`L_D=3, delta=0.02, r=32, K=2`、Qiskit 1.3.0 optimization level 2の同一条件を再照合した。
最新fresh-32の`q=1,2`だけでaffine compiled-RZ proxyを固定し、旧M08/opt2の`q=16,32`
各8 trajectoryをfitへ混ぜずholdoutとして適用した。

selected `support_run_le_1`の絶対相対誤差は`q=16`で3.345%、`q=32`で4.911%となり、
既存5%基準を両方通過した。旧holdoutのdirect RZ relative SE最大は、full basisの`q=16`に
おける1.3865%で2%基準内だった。ただしselected `q=32`の標準化残差は2.012で1.96を僅かに
超える。非gatingのfull-basis診断は`q=32`で5.598%となり5%を超えた。

従って、最新fresh較正の適用domainを`q=32`まで接続できるのは、この単一cellのselected-policy
compiled RZに限る。全metric、他の`delta/r`、`q>32`へは拡張しない。schedule・cost再最適化、
最終総cost、科学的優位性の評価も行っておらず、外部instance pilotを次のdiscriminator候補とする
既存方針は変更しない。artifact fingerprintは
`b786b1d48995d320c2301718cc958018462cf1cbb05fcfa19ca6cc9460dd2295`、専用testは`3 passed`。
M06-F関連全体は`13 passed, 2 skipped`、全suiteは`570 passed, 2 skipped, 4 warnings`で
失敗0、manifest検査もpassした。
詳細は[M06-F all-r coherent opt2](docs/research_direction_full_opt2.md)を参照する。

## 2026-09-25 M06-F fresh-32 and coherent opt2 result

事前登録済みfresh-32拡張は固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、
`L_D=3`、`delta=0.01,0.02`、`r=1,2,4,8,16,32`、`q=1,2,8`のうち初期精度gateで
指定された5 groupだけをQiskit 1.3.0、optimization level 2、basis `rz,sx,x,cx`、seed 17で
実行した。15/15 task、1,920 direct transpile、failed 0、wrapper exit code 0、13,833秒で完了した。
初期36 taskと統合して51/51/0 expected/completed/failedで、missing、partial、duplicate、seed重複、
破損JSON、aggregate不一致は0。CPU-only、各thread=1、compute source hash不変も確認した。

12個のrandomized group全てが事前gateを通過した。direct RZ relative SE最大1.9844%、
selected-policy RZ holdout最大4.4898%、selected全metric holdout最大4.7488%である。
`q=1,2`は較正、`q=8`は固定holdoutとして分離した。同一opt2 contextでbeta、alpha、integer
shot、scheduleを再最適化すると、両候補とも`delta=0.02`を選び、状態準備なしのcompiled-RZ
点推定は`L_D=3/12`で`1.263314e12/1.327822e12`、shotは13,588/11,162だった。
点推定は`L_D=3`が4.858%低いが、local 5%、per-r実測幅、25%移送の全区間は重なる。

旧mixed compiler-context focused推定に対し、coherentな`L_D=3`点推定は3.8918%増え、
点利得は8.422%から4.858%へ縮小した。共通状態準備costの点break-evenは26,590,335
compiled-RZ相当/shotだが、P=0ですでに区間が重なるため頑健な非負P範囲はない。
all-rのcompiler-context欠落は現H4測定範囲で解消した一方、q>32、状態準備、外部instance、
coupling/backend、H12、immutable再現は未解決であり、最終総costまたは科学的優位性は主張しない。

監査artifact fingerprintは
`b39960a630746e2c05009f8d7e13bd982ff565b3a3c70a7abfc2dc65dc7009ca`、
coherent解析fingerprintは
`5ce368a94daa39680b4edc0cfb59b30168d8bc159ad2538b29cb67b928e3cdba`。
T4/T7主軸、T1範囲変更、T2/T5/T6限定、T3保留は維持し、局所compiler精密化を止めて
外部instance pilotを次のdiscriminator候補として再開可能にする。
commit済みartifactを使うM06-F専用testは`10 passed`、Git管理外raw checkpointを再集計する
optional integration testは`2 skipped`。manifest検査はpass。clean checkout相当の全suiteは
`567 passed, 2 skipped, 4 warnings`で、失敗は0だった。raw再集計はserver evidenceが利用できる
環境でのみ実行し、clean checkoutの通常testを失敗させない。
詳細は[M06-F all-r coherent opt2](docs/research_direction_full_opt2.md)。

## 2026-09-24 M06-F all-r coherent opt2 initial result

終了後の完全性監査で、初期36 taskのtask spec、worker result、checkpoint、aggregateを再照合した。
failed、missing、partial、duplicate、破損JSON、fingerprint不一致はいずれも0で、tmuxと関連processも
終了していた。runner契約から終了code 0と判定したが、元shellの`$?`自体は独立保存されていない。
初期batchは36/36完了、所要17,475.324秒、事前登録workflow全体はfresh-32 15 task未実行のため
36/51である。監査artifact fingerprintは
`7d184c4ebde0665fbc85452c69b9de14997f101fb51f9e2969fd13bf1d5ebf35`。
この監査は下記の研究判断を変更せず、coherent再最適化を完了扱いにしない。
監査専用testは`3 passed`、関連testは`33 passed`。全suiteは`562 passed, 2 failed, 5 warnings`で、
失敗2件は今回の変更外にある保存referenceのPython版不一致と既存four-round artifactの
DF preparation hash不一致である。この監査では無関係な保存artifactを変更しない。


WP11が選択した`all_r_coherent_opt2_reoptimization`について、固定H4 linear chain、1.0 Å、
STO-3G、8 qubit、DF rank 12、`L_D=3,12`、`delta=0.01,0.02`、Qiskit 1.3.0、
optimization level 2、basis `rz,sx,x,cx`、seed 17の同一compiler contextで初期計算を実行した。
36 cell task、1,062 direct transpileは全て完了し、失敗・中断は0だった。

12個のrandomized `(delta,r)` groupのうち7個は全基準を通過した。selected-policy RZおよび
全compiled metricのholdout誤差は全groupで5%以内だった。一方、direct RZ relative SEの2%基準は
`(0.01,16)`, `(0.01,32)`, `(0.02,8)`, `(0.02,16)`, `(0.02,32)`の5 groupで不通過となった。
最大値は順に3.911%、2.416%、2.735%、2.543%、2.141%である。これは初期8 trajectoryの
精度不足として扱い、事前規則どおり各groupの`q=1,2,8`をfresh 32 trajectoryで再計算する
15 task、1,920 direct transpileのextension manifestを生成した。追加計算は未実行である。

初期aggregate fingerprintは`ae0e0d9b616da5c31cbdd09d27d2b6e103cc07de05f0b400ab04f51da8f0c63a`、
解析fingerprintは`b256b47a83fb54657716d4d7f910a8772aa0f583c54ec9a63259f37784d0bcaf`、
extension manifest fingerprintは`c150a17c92ade7bd5257a8b99a92bdfdae688fb467000500a9e92754ac23e997`である。
statusは`requires_fresh_32_trajectory_extension`であり、coherent再最適化、`L_D=3/12`比較、
状態準備、q>32、backend/noise、最終総cost、科学的優位性は未評価である。専用testは`4 passed`、
関連testは`32 passed`。これはlocal dirty-worktree evidenceであり、immutable CIではない。
詳細は[M06-F all-r coherent opt2](docs/research_direction_full_opt2.md)。

## 2026-09-23 WP11 scoped direction synthesis note

Gate S1、WP06-a/b、WP05-a/b/R、WP01-D/C07、G08、M08、M06/L08、N07/P03の11個の
fingerprint済みartifactを新規compileなしで統合した。固定範囲はH4 linear chain、1.0 Å、
STO-3G、8 qubit、DF rank 12、`CA/10`、`L_D=3,12`、compiled RZである。

T4（長いランダム回路cost予測）とT7（信頼できる資源評価・否定的結果）を主軸として継続し、
T1は一般的優位性からH4条件付き成立限界へ範囲変更、T2/T5/T6は限定継続、T3は保留とした。
状態準備なしの点推定候補は`L_D=3`だが、頑健判定は
`undetermined_under_compiler_transfer_and_preparation_sensitivity`のままである。

次の一件は`all_r_coherent_opt2_reoptimization`とした。既存opt2証拠は`L_D=3,r=32`だけなので、
未測定`r=1,2,4,8,16`をoptimization level 2で較正・holdoutし、同一compiler contextで
beta、alpha、shot、scheduleを再最適化する。外部instance pilotは棄却せず、この比較を完了または
停止した後へ延期する。選択次段はCPU transpile中心でGPUを必要としない。

WP11自体は新規物理計算、full opt2実行、q>32直接検証、状態準備計測、backend/noise、最終総cost、
科学的優位性を含まない。artifact fingerprintは
`45def7f696eddba574878cc7530837dfdfc5c6e9c2767ee3e115cf4a5f1ac092`。専用testは`3 passed`、
変更後のlocal全suiteは`557 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。
詳細は[WP11限定判断統合](docs/research_direction_wp11_synthesis.md)。

## 2026-09-23 N07/P03 uncertainty ledger and preparation break-even note

WP01-D/C07、M08再集計、M06/L08 compiler-transferのfingerprint済みartifactを使い、新しい
回路compileなしで不確かさ台帳と状態準備costのbreak-evenを再集計した。固定H4 linear chain、
1.0 Å、STO-3G、8 qubit、DF rank 12、$L_D=3,12$、$\delta=0.02$、compiled RZ比較である。
shot数は13,538と11,162で、$L_D=3$が2,376多い。

共通の1 shot当たり状態準備costをPとすると、点推定break-evenはopt1で99,045,126、opt2
focusedで47,067,344 compiled-RZ相当/shotだった。opt1 local 5%区間で$L_D=3$が確実に低い
P範囲は0--640,843に限られる。opt2 focused selected実測幅はP=0ですでに区間が重なるため、
compiler-robustに$L_D=3$区間が低い非負P範囲はない。候補別準備では共通Pを使わず、
`13538*P3-11162*P12`の二次元境界を使用する。

sampling、model discrepancy、compiler、q>32移送、opt2の$r<32$、状態準備、外部snapshot/backend/noiseを
別classとして記録した。頑健判定は
`undetermined_under_compiler_transfer_and_preparation_sensitivity`である。後続WP11限定判断統合へ入力済みである。
状態準備costは測定しておらず、q>32、full opt2、backend/noise、最終総cost、科学的優位性を含まない。
artifact fingerprintは`ff70308a64798c6ba8c9d20533c9e9e8c614e58c0d433dd861b7de45ac70c32d`。
専用testは`2 passed`。変更後のlocal全suiteは`540 passed, 4 warnings`で、warningは既存
grouped-UWC test由来である。詳細は
[N07/P03不確かさ・break-even](docs/research_direction_uncertainty_break_even.md)。

## 2026-09-23 M06/L08 compiler-transfer analysis and reaggregation note

固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、$\delta=0.02$で、
同一trajectoryをQiskit optimization level 1から2へ変更して再compileした。$L_D=3,r=32$の
$q=1,2,16,32$とtail-free $L_D=12$を評価し、その他のcompiler条件は固定した。
optimization level 2の$q=1,2$ affine proxyは$q=16,32$でselected RZ最大2.340%、selected
全metric最大2.470%、full-basis RZ最大3.420%、direct RZ relative SE最大1.387%となり、
5%点誤差・2%精度基準を通過した。

WP01-D/C07のschedule、shot、$\alpha$、$\beta$を固定し、$L_D=3$は直接証拠のある$r=32$の
最後3 roundだけ、$L_D=12$は決定論provider全体をoptimization level 2へ置換した。点推定は
$1.2159903\times10^{12}$と$1.3278223\times10^{12}$で$L_D=3$が8.42%低いが、対称discrepancyの
分離上限1.881%に対しselected実測値は2.340%で、区間は重なった。一様比率を未測定$r<32$へ
移す反実仮想もdiscrepancyの選択で分離判定が変わる。従ってcompiler-invariantなlocal分離は
未確立で、頑健判定は`undetermined_under_compiler_and_transfer_sensitivity`とする。

raw artifact fingerprintは`87f66c8944dedfaa9fe5f0edf864d2a2a07cb82e23245af4f5944f90bda83ad6`、
解析・再集計artifact fingerprintは`7ebe8815a633a182b4afa4608b969df5c53d1b6d1efe6eeaf2ce4e653fcbbd95`。
専用testは`3 passed`、変更後のlocal全suiteは`538 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。これはlocal dirty-worktree evidenceであり、$L_D=3,r<32$のopt2直接検証、
$q>32$、full opt2再最適化、最終総cost、科学的優位性を含まない。詳細は
[M06/L08 compiler-transfer解析](docs/research_direction_compiler_transfer.md)。

## 2026-09-22 M08 late-round holdout and WP01-D/C07 reaggregation note

固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、$L_D=3$、$\delta=0.02$、
$r=32,K=2$について、G08で固定した未使用$q=16,32$を各8 fresh trajectory、cosine/sine両軸、
selected `support_run_le_1`／full basisのcomplete controlled Hadamard wrapperで直接transpileした。
selected policyのRZ誤差は最大2.466%、全metric最大2.569%、full-basis RZ誤差最大3.286%、
direct RZ relative SE最大1.353%で、5%基準、5.0484%分離限界、2%精度診断を全て通過した。
事前規則による$q=64$ follow-upは発火しなかった。

WP01-D/C07の点推定・schedule・shot数・較正half-widthを保持し、M08 selected RZ 2.466%と
観測RZ最大3.286%を共通の対称model-discrepancy scenarioとして再集計した。両scenarioとも
$L_D=3$と$L_D=12$の区間は分離し、従来5%区間も僅かに分離したままだった。一方、25%移送区間は
重なったため、頑健な方向判断は`undetermined_under_transfer_sensitivity`を維持する。

M08測定は$q\leq32$だけの直接証拠で、実scheduleの$q_{\max}=131072$、$L_D=12$、別snapshot、
別compilerへの直接証拠ではない。再集計は同じ許容幅を両候補へ置く反実仮想であり、点推定の
再最適化でも最終総cost評価でもない。M08 artifact fingerprintは
`e010a63bfa5aecd7de01ba074f56300615460de9e6f51a40dca352954992aaf8`、再集計artifactは
`6aa69bc756a97aa025e26994f596f9ee3df999be2e64be870ce938dca5180b3e`。詳細は
[G08/M08後半round proxy精度](docs/research_direction_late_round_proxy.md)。変更後のlocal全suiteは
`535 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-22 G08 round-dominance note

WP01-D/C07のfingerprint済みschema-v2 computeと判断統合artifactを、新しい回路compileなしで
round別に再集計した。最後3 roundは$L_D=3$のcompiled RZ cost 90.94%、較正不確かさ87.70%、
$L_D=12$のcost 83.03%を占めた。両候補の最大cost/PF-riskはround 17だが、$L_D=3$の最大
finite-RTE riskはround 7であり、最大cost roundとは一致しない。

この結果からM08を$\delta=0.02,r=32,K=2,q=16,32$の各8 fresh trajectoryへ限定した。
判定はselected RZ・全metricとfull-basis RZの5%、local区間分離限界5.0484%、direct RZ
relative SE 2%である。G08 artifact fingerprintは
`e696ced27b06e871368f3afa164f507c240d4aa6693223f9f6abe7990b30d064`。これはlocal
dirty-worktreeの検証資源配分判断で、$q>8$精度または最終総costそのものの検証ではない。詳細は
[G08/M08後半round proxy精度](docs/research_direction_late_round_proxy.md)。

## 2026-09-22 WP01-D/C07 candidate-specific optimization and interval synthesis note

固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、CA/10について、$L_D=3$の
`support_run_le_1`とtail-free $L_D=12$、$\delta=0.01,0.02$を比較した。WP05-bRまでの
complete controlled partial-$S_2$／Hadamard wrapperの$q=1,2$ affine proxyと$q=4,8$ holdoutを
使い、compiled RZ、$\beta_{\mathrm{RPE}}=0.40$ rad、$\alpha_{\mathrm{tot}}=0.05$、
cost-weighted $\alpha$の下でPF/RTE/statistical budgetと整数shot数を候補ごとに再最適化した。
coarse/fine gridの後、上位20解のPF/RTE制約境界を反復的に締めている。

両候補とも$\delta=0.02$を選んだ。$L_D=3$は13,538 shot、点推定
$1.4557921\times10^{12}$、$L_D=12$は11,162 shot、$1.6911234\times10^{12}$で、前者が
13.916%低い。5% local model区間はそれぞれ$[1.30654,1.60504]\times10^{12}$と
$[1.60657,1.77568]\times10^{12}$で僅かに分離した。ただし分離幅は$L_D=12$点推定の約0.090%、
対称model discrepancy 5.0484%が分離限界で、採用した5%との差は0.0484 percentage pointに過ぎない。
25%移送区間$[1.01538,1.89620]\times10^{12}$と$[1.26834,2.11390]\times10^{12}$は重なる。

したがってlocal model条件付きの候補は$L_D=3$だが、頑健な方向判断は
`undetermined_under_transfer_sensitivity`であり、部分ランダム化の科学的優位性または最終総costを
示さない。次はM08/G08として、総costの83--91%を占める後半3 round付近の$q>8$ proxyを直接
較正・holdoutする。状態準備、実backend、noise、別snapshot/系サイズ、immutable CI、外部再現は
含まない。詳細は[WP01-D/C07再最適化](docs/research_direction_decision_cost.md)。
compute artifact fingerprintは
`709142f76a78232804cae9971a24bc5757b3ef2e5258adb7314fc42e91937474`、判断統合artifactは
`7d85b472851a6c4046b847a7e9898a718e96b2bad7a50fd491d939b34244250f`である。schema-v1 computeは
grid下端依存を検出した予備診断で、現行判断には使わない。変更後のlocal全suiteは
`530 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-22 WP05-b/R full-scope extension and focused replication note

WP05-aと同じ固定H4 snapshot・compilerで、$L_D=3$の`support_run_le_1`とfull basisを
$q=8$および比較対照$\delta=0.01$へ拡張した。$\delta=0.01$は選択policyの$q=8$ RZ誤差
最大2.650%、全metric最大2.804%、full basis RZ誤差最大4.321%、全metric最大4.500%で
事前5%基準を通過した。初回$\delta=0.02,r=32,q=8$は選択policy RZ 5.084%、
全metric 5.392%、full basis RZ 8.995%で不通過だった。

この一点を新しいseedの独立32 trajectoryで再検証したWP05-bRでは、選択policyの$q=8$ RZ誤差
0.516%、全metric最大0.537%、full basis RZ誤差0.829%、選択policy$q=4$全metric最大1.094%で、
全checkが5%基準を通過した。初回超過は高統計再検証で再現しなかった。これは1 snapshot・
1 compiler・最大$q=8$のlocal dirty-worktree evidenceであり、$q>8$や別条件への精度移送、
最終総costを保証しない。詳細は
[WP05-b/R拡張・再検証](docs/research_direction_full_scope_extension.md)。初回artifact fingerprintは
`363ac90ace643556ff068ff7b22ed8e43c51c6dbd0085a2256dac29a0303bfda`、再検証artifactは
`7bfeddaccfe10f28b67bdf857ebd76cd43b0eb74b5af5d209a435ee9d8abb472`である。

## 2026-09-22 WP05-a full controlled-interrogation connection note

固定H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、$L_D=3,12$、$\delta=0.02$、
Qiskit 1.3.0の`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなしで実施した。
WP06-bの`support_run_le_1`を明示basis planとしてcomplete controlled partial-$S_2$、反復回路、
cosine/sine Hadamard wrapper、ancilla Z測定まで伝播した。production既定値はfull basisのままである。

$L_D=3$では$r=1,2,4,8,16,32$、$q=1,2,4$、各8 trajectoryについてfull/policyを対応付け、
576本のmeasurement-bearing wrapperを直接transpileした。$q=1,2$から固定したaffine式は、独立seedの
$q=4$で選択policyのRZを最大2.288%、全6 metricを最大2.431%で予測した。full basisのRZ最大誤差は
4.234%、tail-free $L_D=12$は0%だった。WP06-b中央RTE additive bridgeと今回の直接wrapper差の
RZ残差はfull-wrapper平均比で最大2.625%となり、事前5%基準を通過した。H4の固定ランダム状態作用
比較はcontrolled evolutionと両wrapperで最大$1.08\times10^{-16}$、relative ancilla phaseも一致した。
専用小系testでは完全operatorを比較している。

WP04のround、shot、$\alpha$を固定した非decision-grade長$q$感度では、選択policyの$L_D=3$が
$1.5933\times10^{12}$、tail-free $L_D=12$が$1.6963\times10^{12}$で、点推定は$L_D=3$が6.07%
低い。ただしlocal区間$[1.3489,1.8377]\times10^{12}$と
$[1.6115,1.7811]\times10^{12}$は重なる。$q>4$を直接transpileせず、$\alpha$・shot・scheduleも
再最適化していないため、最終総costまたは科学的優位性ではない。

後続のWP05-b/Rで$q=8$と$\delta=0.01$の5%基準を通過し、WP01-D/C07の再最適化まで完了した。
状態準備、backend実行、noise、量子shot、immutable CI、外部再現は含まない。詳細は
[WP05-a full-scope接続](docs/research_direction_full_scope.md)。専用testは`4 passed`、成果物fingerprintは
`d8196ef1d8a576b7b7ab443c8613d2bd70b7a4fa57f43ac495be62bf2748f512`である。変更後のlocal全suiteは
`524 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-22 WP06-b sequence-policy and proxy-bridge note

WP06-aと同じH4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定snapshot、$L_D=3$、
$\delta=0.02,K=2$、Qiskit 1.3.0 compilerでsequence-aware basis policyを比較した。元DF basisの
run長1,2,3以下または全runをsupport限定へ置換する候補を列長1,2,4の独立trainingで比較し、各trajectory
のRZ悪化5% guardを通る`support_run_le_1`を固定した。production builderの既定値は変更せず、明示
basis planとして渡す。

未使用列長3,6の各12本では、full basis共有比でRZ -10.67%、CX -8.13%、total depth -2.25%、
circuit size -10.63%だった。trajectory別oracleに対するpooled RZ regretはfull RZ基準0.390%。sampled
列長1,3と強制$K=2$非零phase eventのcontrolled operator残差は最大$1.34\times10^{-15}$で、強制
eventのrelative ancilla phaseも一致した。

$r=1,2,4,8,16,32$の独立各8本から中央RTE差を既存Hadamard proxyの$q$ slopeへ加えると、RZ slopeは
最大8.00%変化した。WP04のround、shot、$\alpha$、wrapper interceptを固定したbridgeでは$L_D=3$の
点推定が$1.7848\times10^{12}$から$1.6503\times10^{12}$へ下がり、$L_D=3/12$の点順位が反転した。
ただし両local区間は重なるため未決定を維持する。

後続WP05-aで選択policyをcomplete controlled partial-$S_2$／Hadamard wrapperへ直接接続し、
未使用$q=4$とadditive bridgeの5%基準を通過した。さらにWP05-b/Rの$q=8$・$\delta=0.01$、
WP01-D/C07の候補別再最適化まで完了した。
WP06-b自体は中央RTEだけのadditive bridgeであり、full wrapper、$\alpha$・shot再最適化、
状態準備、実backend、noise、最終総cost、immutable CIまたは外部再現ではない。詳細は
[WP06-b sequence policy](docs/research_direction_sequence_policy.md)。専用testは`5 passed`だった。
変更後のlocal全suiteは`520 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-22 WP06-a circuit-structure pilot note

Gate S1と同じH4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定snapshot、$L_D=3$、
$\delta=0.02$で、係数最大のZ/ZZ event、同一fragmentの異なるsupportからなる長さ1--3列、
full Gaussian basis、support限定completion、basis融合、control、scalar/relative phaseを比較した。
compilerはQiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなしである。

support限定completionは必要なunitary列を保持し、単一controlled ZのRZを204から78へ61.76%、
ZZを311から189へ39.23%減らした。全比較の最大operator残差は$4.73\times10^{-15}$で、
$10^{-10}$基準を通過した。Gate S1で事前固定した5% triggerは発火した。一方、異なるZZ supportの
列では長さ2がRZ 19.58%減・depth 45.26%増、長さ3がRZ 15.01%増・depth 86.64%増となり、
full basis共有とsupport限定の優劣が列長で反転した。support限定への一律置換は採用しない。

whole-event controlと現行diagonal-only controlはrelative phaseを含め一致し、現行方針はRZを
2,567から311へ87.88%、CXを1,834から98へ94.66%減らした。controlled scalar補償を省くと
operator差0.0477098が生じ、現行補償を入れると$4.73\times10^{-15}$以内で一致した。controlと
phase方針は維持する。

次はfocused WP06-bとしてsequence-awareなfull/support basis policyをproduction builderへ統合し、
物理event分布と未使用短列holdoutでproxyを再較正する。その後にWP05へ進む。これはlocal
dirty-worktreeの構造pilotであり、$L_D$候補順位、RPE $q$傾き、full controlled interrogation、
最終総cost、immutable CIまたは外部再現ではない。詳細は
[WP06-a回路構造pilot](docs/research_direction_structure_pilot.md)。専用testは`4 passed`だった。
変更後のlocal全suiteは`515 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-22 Gate-S1 research-direction synthesis note

WP00、WP02、WP01-S、WP04、WP03のfingerprint済み成果物を統合した。新しい物理計算または
回路compileは行っていない。WP04の$L_D=12$点推定は$L_D=3$より4.958%低いが、local 5%と
transfer 25%の両scenarioが重なるため、`undetermined_not_tied`を維持する。主な共通利得は
$\beta$、次いで$\alpha$再配分で、PF係数policyは全て$L_D=12,\delta=0.02$を選択した。
partial randomization固有の優位性は示されていない。

最大の残存不確かさをfull controlled interrogationの回路scope・構造と判定し、T4/T7を主軸、
T1/T2/T5/T6を限定継続、T3を保留とした。次はWP06-aだけを行い、その後WP05へ進む。
WP06-a専用のresearch-routing triggerはRZ相対変化5%、候補順位反転、$q$傾き・適用domainの変更、
またはcontrolled relative phase補償の欠落である。5%は現点推定差4.958%に合わせたtask固有値で、
普遍的な回路精度保証ではない。

これはlocal dirty-worktreeの研究方向判断で、科学的優位性のdecision-grade評価、最終総cost、
immutable CIまたは外部再現ではない。詳細は
[Gate S1判断](docs/research_direction_gate_s1.md)。専用testは`4 passed`だった。
変更後のlocal全suiteは`511 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-22 research-direction WP03 PF-coefficient sensitivity note

WP00/WP04と同じH4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定snapshot
`56e4df83...31e5`、CA/10 taskで、costed候補$L_D=3,12$、
$\delta=0.01,0.0125,0.02$について、$C_D$、論文D6、支配固有位相係数だけを差し替えた。
compiled-cost provider、round別compiled-RZ schedule、cost感度重み$\alpha$、WP04の候補別
$\beta$配分は固定した。$L_D=0$はWP01-Sでcompiled-cost評価前にscreen out済みなので、
係数反例の監査だけに含めた。

$C_D$もD6・支配固有位相と同じconditioned窓$\delta=0.05,0.1,0.2,0.4$で再fitした。
$L_D=3$の係数は$C_D=0.0117236$、D6 0.0133991、支配固有位相0.0133569で、$C_D$はD6比
12.50%低かった。$L_D=12$では0.0134411、0.0134257、0.0133833だった。costed候補のD6と
支配固有位相係数差は最大0.317%である。$L_D=0$では$C_D=0$でもD6 0.0115338、支配固有位相
0.0114931が非零なので、$C_D$は引き続き広いscreening専用とし、shortlist後は候補ごとのD6を使う。

3係数×2候補×3$\delta$の18条件は全てPF予算内で、全係数が$L_D=12,\delta=0.02$を
RZ点推定最良とした。D6で$L_D=3,\delta=0.02$を選ぶ点regretは5.216%だが、相対差の
local 5%＋較正scenarioは$[-15.91\%,28.57\%]$、25%移送scenarioは
$[-46.20\%,90.91\%]$で0を跨ぐ。従って係数選択はWP04の`undetermined`判定を解消しない。
$\delta=0.02$がPF予算外になるまでのD6係数増加余裕は$L_D=3$で6.76%、$L_D=12$で6.55%で、
現行D6対支配固有位相差より大きいが、別instanceへ無条件に移送できる余裕ではない。

保存されたD6のsigned biasは支配固有位相と逆符号で、定義の符号規約が直接揃っていない。
また個々のmixed/tail交換子項は分解していないため、符号差を物理的相殺と解釈せず、$C_D$から
full partial係数への差も省略効果の合計として扱う。

これは$q=1,2$直接較正から長$q$へaffine外挿したlocal dirty-worktree screeningである。
係数familyは統計的信頼分布でなく、状態準備、実backend、noise、full-scope holdout、最終総cost、
優位性、immutable CIまたは外部再現を含まない。Gate S1に必要なWP01-S、WP02、WP04、WP03が
揃ったため、次は研究方向判断を統合する。詳細は
[WP03係数感度](docs/research_direction_pf_sensitivity.md)。専用testは`4 passed`だった。
変更後のlocal全suiteは`507 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-21 research-direction WP04 ablation note

WP00/WP01-Sと同じH4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定snapshot
`56e4df83...31e5`、CA/10、$\delta=0.02$、$M=17$、$q_{\max}=131072$で、$L_D=3,12$の
round schedule、$\beta$、$\alpha$、schedule-selection cost providerを分離した。状態準備なし
Hadamard scopeの$q=1,2$直接較正を再利用し、$q>2$は軸別affine外挿とした。

3 schedule policy、2 alpha policy、候補別beta profileから42 factorial cellを評価し、固定順の
逐次差分、完全設定からのleave-one-out、beta--alpha interactionを保存した。完全設定から
beta再配分を戻すとRZ点推定は$L_D=3,12$で143.2%、149.6%、alpha再配分を戻すと35.4%、
26.3%増えた。$L_D=3$のcompiled-RZに整合したround scheduleは同じ完全設定の固定schedule比
1.93%減に留まる一方、成分作用数を目的に選ぶscheduleはRZを16.87%増やした。したがって、
WP04での主要な共通利得はbeta、次いでalpha再配分であり、cost provider変更をfinite-RTE固有の
schedule利得と同一視しない。

完全設定のRZ点推定は$L_D=3$で$1.7848\times10^{12}$、$L_D=12$で
$1.6963\times10^{12}$となり、決定論endpointは4.96%低い。ただしlocal 5%＋較正区間と
25%移送＋較正区間はともに重なるため、方向判定は引き続き`undetermined`である。
$L_D=3$の選択schedule全18点を含むsector行列gridは既存validatorを通過し、最小観測半径
0.5727013937は最小保守下界0.5727013920以上だった。$L_D=12$のtailなし信号の最小半径は
0.9999999998だった。最後の3 roundは両候補のRZの91.9%、83.0%を占める。

これはlocal dirty-worktreeの`model_conditional_screening`である。長$q$ costは未使用holdoutの
ない$q=1,2$ affine外挿、beta gridは連続最適化でなく、厳密二項値は既知の小系信号に対する
counterfactualである。状態準備、実backend、noise、full-scope holdout、最終総cost、部分
ランダム化の優位性、immutable CIまたは外部再現を主張しない。次はWP03でPF係数だけを
差し替え、候補順位とregretの変化を評価する。詳細は
[WP04寄与分解](docs/research_direction_ablation.md)。専用testは`4 passed`だった。
変更後のlocal全suiteは`503 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-21 research-direction WP00/WP02/WP01-S note

H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12の固定Hamiltonian snapshot
`56e4df83...31e5`について、研究方向screeningの最初の3段階を実行した。WP00では
$L_D=0,3,12$のPF入力を同じsnapshotから再生成し、CA/10、$\beta=(0.02,0.02,0.36)$、
$\alpha_{\rm total}=0.05$一様配分、状態準備なしHadamard scope、同一compilerの比較契約を固定した。
既存PF artifactはcost snapshotとHamiltonian hashが一致しなかったため、この比較には使用しない。

WP02ではCA、CA/10、CA/100と$\delta=0.01,0.0125,0.02$の9条件を監査した。
CA/10の$q_{\max}$は131,072--262,144で、既存3 schedule・56点sector行列検査を再利用した。
CA/100の$q_{\max}$は2,097,152--4,194,304で、3条件すべてが経験的
$qC\delta^3\leq0.02$を満たさない。従ってCA/100は長回路costを先に外挿せず、
小さい$\delta$と新scheduleを作る必要がある。

WP01-Sでは$L_D=0$だけ$r=1,\ldots,65536$、$K=0,2,\ldots,16$へ探索域を拡張した。
成分作用数proxyの最良条件は$\delta=0.02$、半径下界0.5473、shot合計25,400、
proxy値$4.4839\times10^{12}$で、探索境界には当たらなかった。同じ目的関数の$L_D=3$最良値の
560.6倍だったため、compiled RZ cost評価前にscreen outした。$L_D=3$の7種類の$(r,K)$は
$\delta=0.02,q=1,2$のfull Hadamard wrapperを各8 classical trajectoryで直接compileし、
$L_D=12$はtailなしの$q=1,2$を厳密評価した。CA/10の全roundへaffine外挿すると、
両候補とも$\delta=0.02$が最小で、no-prep総RZ点推定は$L_D=3$が
$3.0814\times10^{12}$、$L_D=12$が$2.4329\times10^{12}$だった。決定論endpointは
点推定で21.0%低く、5% scenarioでは区間が分離するが、25%移送scenarioでは重なる。

従って現状は、$L_D=0$をこの解析的screen内で強く不利とし、$L_D=3$対12は
`undetermined`とする。$L_D=0$対3の560.6倍はcomponent-application proxyの比較であり、
compiled RZ cost比または一般的な理論上の棄却ではない。
PF係数は経験値、schedule別$q=1,2$ fitには未使用holdoutがなく、$q>2$は直接compileしていない。
scenario幅も統計的信頼区間ではない。この結果を最終総cost、部分ランダム化の優位性、
immutable CIまたは外部再現とは扱わない。次はWP04、WP03で寄与と係数感度を分離し、
WP05後のWP01-Dでdecision-gradeに再評価する。詳細は
[研究方向screening検証](docs/research_direction_prevalidation.md)。関連testの部分実行は`8 passed`で、
変更後のlocal全suiteは`499 passed, 4 warnings`だった。warningは既存grouped-UWC test由来である。

## 2026-09-20 delta-schedule central-RTE compiled-cost note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなしで、
前項の$\delta=0.01,0.0125,0.02$ scheduleへ境界補正型compiled-cost modelを接続した。

scheduleで現れる短時間幅0.02--0.000390625の12点について、同じ1--4イベント列を
角度だけ変えて個別にコンパイルした。1--3イベントは280 trajectory・3,080 metric比較、
4イベントは160 trajectory・1,760比較で、RZ/CX数・深さ、全体深さ、回路サイズの差は
すべて0だった。したがって、この固定compiler範囲では短時間幅0.02の係数を再利用した。

固定K1--K3を未使用イベント列へ適用すると、$L=8$は全6指標最大2.319%、$L=16$は
3.556%以下で点基準を通過したが、$L=32$は全次数0 RZ誤差7.966%、$z=3.151$で不通過だった。
各4イベントTaylorパターン100標本でK4を較正すると、$L=32$の最大点誤差は4.175%へ下がった。
ただしK4の点ごとの95%診断は5%を超え、厳密な5%保証ではない。次数2が2か所以上のK4窓は
今回のround分布では最大期待数$7.64\times10^{-13}$以下だった。

$r\leq16$にK1--K3、$r=32$にK1--K4を使い、各roundの解析Taylor確率、$q_m$、暫定shot数で
中央RTEブロックを集計した。RZ代理値は$\delta=0.02$で$7.9877\times10^{11}$、0.01で
$9.2021\times10^{11}$、0.0125で$1.2626\times10^{12}$となり、全6指標で0.02が最小だった。
0.01はRZで15.203%増、0.0125は58.073%増である。

これは中央$\widetilde U_{\rm RTE}$ブロックだけのlocal dirty-worktree proxyである。
決定論DF half sweep、決定論/RTE外側境界、制御化、Hadamard wrapper、状態準備、$q>8$一体回路、
最終総cost、immutable CIは未評価。従って$\delta=0.02$は次の優先候補であって最終採用ではない。
次は0.02と比較対照0.01の制御付きpartial-$S_2$反復proxyを検証する。詳細は
[delta schedule中央RTE cost検証](docs/rpe_delta_compiled_cost_validation.md)。
変更後のlocal全テストは`494 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-20 delta and round-specific finite-RTE schedule note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$で、
既存PF検証で実行済みの10個の$\delta$候補を、暫定`CA/10`、
$\beta_{\rm RPE}=0.4$、経験的$C=0.01342567$でscreeningした。PF位相予算0.02 radを
通過したのは$\delta=0.01,0.0125,0.02$の3候補だった。$\delta=0.0125$は較正と
独立なPF検証gridにおける唯一の通過点である。

各候補の18--19 roundで$r_m\in\{1,2,4,8,16,32,64,128\}$、
$K_m\in\{0,2,4,6,8\}$を走査し、shot数で重み付けしたランダム成分作用数を
暫定proxyとしてround別scheduleを構成した。選択した全56 round点のH4 sector行列検査で、
演算子・信号・適用可能な位相上界、PF/RTE予算、半径下界がすべて通過した。
最小観測半径は0.572701、最大実PF位相誤差は0.0139884 rad、最大有限RTE位相上界は
0.0176405 radだった。

暫定proxyでは$\delta=0.02$が最小だが、$\delta=0.01$との差は約0.87%である。
このproxyはcompiled costではないため、両者をshortlistとし、回路cost modelの
再較正またはholdout後に比較する。$q>8$一体compile、fresh-IID実験、最終総cost、
immutable CIは未評価。詳細は
[delta/round schedule検証](docs/rpe_delta_round_schedule_validation.md)。
変更後のlocal全テストは`490 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-20 target-precision round-horizon note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、
$L_D=3,\delta=0.1$、$\beta_{\rm RPE}=0.4$について、
$\beta_{\rm RPE}/(2^M\delta)\leq\epsilon_E$を満たす最小round範囲を計算した。
正本文書では$\epsilon_E$は外部入力なので、既存設定`TARGET_ERROR=CA/10`を暫定主条件、
化学精度$CA$を感度比較とした。

補足資料の架空例$\epsilon_E=0.50$は$M=3,q_{\max}=8$、化学精度は
$M=12,q_{\max}=4096$、暫定`CA/10`は$M=15,q_{\max}=32768$となった。
従って、既存4段検証は実目標候補のround範囲を覆わない。

固定$r=4,K=2$をsector行列上で$q=8,4096,32768$へ延ばすと、$q=32768$の
PF位相誤差は0.437285 radで0.02 rad予算を超え、attenuationは
$7.90584\times10^{-13}$だった。finite-RTE演算子・信号上界は通過したが、この固定設定を
長roundへ単純外挿する候補は棄却する。次は$\delta$候補とround別$(r_m,K_m)$を再探索する。

これはlocal dirty-worktreeの小規模行列診断である。$q>8$回路コンパイル、cost proxy、
fresh-IID shot、最終総cost、immutable CIは未評価。詳細は
[round範囲診断](docs/rpe_target_round_horizon_validation.md)。
変更後のlocal全テストは`487 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-20 physical q=8 and four-round branch-reconstruction note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、
$L_D=3,\delta=0.1,r=4,K=2$、$q=1,2,4,8$で、限定4段集計と同じ
$\beta=(0.02,0.02,0.36)$および重み付き$\alpha$を使い、$q=8$物理信号と4段の
逐次分枝復元を検証した。

$q=8$のfinite-RTE信号半径は0.993219824261、exact信号からの系統位相差は
$1.06819\times10^{-4}$ rad、厳密二項の統計位相失敗率は$2.07129\times10^{-6}$だった。
4段合成の座標失敗率は$1.33625\times10^{-3}$、統計位相失敗率は
$2.14873\times10^{-6}$で、割当予算0.05以内だった。

4段8軸の全1,572 shotへ異なるRTE trajectory seedを割り当てた明示的監査を行い、seed重複なし、
trajectory平均の解析信号からの差は最大1.925標準誤差だった。解析的周辺分布から生成した
10万回の4段測定では分枝失敗・最終位相失敗とも0件で、最終失敗率の片側95%上限は
$2.99569\times10^{-5}$だった。

固定H4の4段end-to-end接続はlocalに通過したが、目標エネルギー精度からのround数決定、$q>8$、
別条件、実backend、noise、状態準備、最終総cost、immutable CIは未評価である。PF係数が経験値の
ため保証statusは引き続き`empirical_screening`である。詳細は
[4段分枝復元検証](docs/rpe_four_round_phase_validation.md)。
変更後のlocal全テストは`484 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-18 limited four-round RPE accounting note

H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、
$L_D=3,\delta=0.1,r=4,K=2$、$q=1,2,4,8$で、前回選んだ暫定配分
$(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.02,0.02,0.36)$と重み付き
$\alpha$を既存の厳格な資源集計APIへ接続した。$q=1,2,4$の固定Hadamard直接costと、
未使用$q=8$ holdout通過済みproxyを出典付き複合providerとして使用した。

4段・8軸の合計は1,572 shot、RZ数32,673,960.607143で、shot×1 shot costの直接再計算と
前回診断値に一致した。保守的$\alpha$ union boundは0.05の予算内だった。
前回の$q=1,2,4$物理信号を固定し、新しいshot数と$\beta_{\rm stat}=0.36$で厳密二項失敗率を
再計算すると、合成座標失敗率$2.2246\times10^{-4}$、統計位相失敗率
$7.7444\times10^{-8}$で、各軸・各段の割当額を満たした。

これはlocal dirty-worktreeの限定4段診断で、PF入力が経験値のため`empirical_screening`である。
$q=8$物理信号・厳密失敗率、4段branch復元、最終全round総コスト、実backend、noise、
immutable CIは未評価。詳細は[限定4段検証](docs/rpe_four_round_accounting_validation.md)。
変更後のlocal全テストは`481 passed, 4 warnings`で、warningは既存grouped-UWC由来である。

## 2026-09-01 beta/alpha allocation-sensitivity note

H4 chain、距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
$\delta=0.1$、$r=4$、$K=2$、$q=1,2,4,8$で、RPE位相誤差・失敗確率配分の感度を評価した。
$q=1,2,4$の状態準備なしHadamard直接compiled costと、未使用holdoutを通過した$q=8$ proxyを
固定入力としたため、sweep中の回路再compileはない。

5種類の$\beta$配分と、一様／cost感度重み$\alpha$配分の10 scenarioは、$\beta$和、$\alpha$和、
PF・RTE実寄与、正半径、shot式、round cost恒等式を全て通過した。単一条件への過適合を避ける
暫定100倍headroom規則では$(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.02,0.02,0.36)$と
cost感度重み$\alpha$を選んだ。各軸shotは$q=1,2,4,8$で229、207、186、164、RZ comparison costは
$3.2674\times10^7$である。現行$(0.08,0.08,0.24)$・一様$\alpha$比では56.35%小さいが、同じ
$\beta$での$\alpha$変更単独は4.35%だった。

これは1 snapshot・1 compiler・固定$(L_D,\delta,r,K)$・RZ指標のlocal dirty-worktree sensitivity
diagnosticである。100倍guardは理論値でなく、比較costは最終総costでもその削減率でもない。
選択後の非一様$\alpha$に対する厳密二項失敗率・仮想測定も再実行していない。今回の失敗確率条件は
Hoeffding shot式とunion boundである。$q=8$物理信号・位相、branch reconstruction、$q>8$、noise、
実backend、immutable CIは未評価である。
変更後のlocal全test suiteは`479 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-01 q=8 Hadamard cost-proxy/resource connection note

H4 chain、距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
$\delta=0.1$、$r=4$、$K=2$、Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、
seed 17、coupling mapなしで、状態準備を除くHadamard interrogation全体のcompiled costを
評価した。各$q$ 8 trajectoryの$q=1,2,4$だけでaxis・metric別affine proxyを較正し、
係数固定後に未使用$q=8$の一体compileを予測した。

両軸・全6指標の最大相対点誤差はcircuit sizeの0.726%、RZ countは0.675%で、
事前の5%基準を通過した。較正点とholdoutを含むRZ平均の最大相対標準誤差は
0.732%で、事前2%条件を通過した。holdoutはfitに使用していない。

通過したvalidation fingerprint、Hamiltonian・DF split、$L_D,\delta,r,K$、compilerを要求し、
実際にholdoutした$q$だけを返すproviderをresource accountingへ接続した。
$(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.08,0.08,0.24)$ rad、
$\alpha_{m,b}=0.05/8$の$q=8$ candidateは各軸414 shot、1 shot RZ count 48135.0804となり、
round RZ cost $3.9855847\times10^7$の再計算が一致した。未検証$q=16$は拒否した。

これは1 snapshot・1 compiler・$q=8$のlocal dirty-worktree evidenceである。$q>8$、別分割、
proxy係数共分散、$q=8$物理信号・位相、複数round合計、状態準備、noise、実backend、
最終総costまたはimmutable CI evidenceではない。
変更後のlocal全test suiteは`478 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。

## 2026-09-01 virtual-Hadamard statistical failure note

H4 chain、距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
$\delta=0.1$、$r=4$、$K=2$、$q=1,2,4$で、finite-RTEの物理基底状態信号を用いた
仮想Hadamard測定を検証した。$(\beta_{\mathrm{PF}},\beta_{\mathrm{RTE}},
\beta_{\mathrm{stat}})=(0.08,0.08,0.24)$ rad、$\alpha_{m,b}=0.05/6$から得た各軸shot数は
389、390、391である。

厳密二項計算では、6軸のいずれかの座標誤差が許容量以上となる確率は0.0010363、
3 roundのいずれかの統計位相誤差が0.24 radを超える確率は$1.4775\times10^{-6}$で、
いずれも$\alpha_{\mathrm{tot}}=0.05$以内だった。10万回の周辺Bernoulli反復では座標失敗111回、
位相失敗0回で、片側95%上限はそれぞれ0.001299、$2.996\times10^{-5}$だった。

別に各shotへfresh IIDなRTE trajectoryを割り当て、計2340 trajectoryをsector内で直接作用した。
seed重複はなく、trajectory平均信号と解析信号の差は最大2.063標準誤差、全軸の条件付き測定数は
周辺二項分布の99.9%中央区間内だった。これはshort-roundの測定統計とfresh-IID実装のlocal検証であり、
全roundのRPE branch selection、最終位相復元、実backend、noise、状態準備、最終総costまたは
immutable CI evidenceではない。変更後のlocal全test suiteは`475 passed, 4 warnings`だった。
最初の全suite実行では既存の並列SQLite cache testが一時的な`database is locked`で1件失敗したが、
単独再実行と続く全suite再実行では通過した。4 warningは既存grouped-UWC test由来である。

## 2026-09-01 short-round signal・shot・compiled-cost connection note

H4 chain、距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian snapshot、$L_D=3$、
$\delta=0.1$、$r=4$、$K=2$、$q=1,2,4$で、finite-RTE信号検証、RPE shot式、
controlled time-evolution direct provider、状態準備なしHadamard interrogation providerを接続した。
物理full-$H$基底状態のPF信号半径は1から最大$1.11\times10^{-8}$のずれで、単位半径仮定と
実半径から得る各軸shot数は$q=1,2,4$で389、390、391と一致した。

同一compiler条件ではHadamard interrogationのRZ countはtime-evolution部分より各軸+2、
circuit sizeは+5で、CX count/depth、RZ depth、total depthは同じだった。
全roundで`round_cost=N_c g_c+N_s g_s`の再計算、scope識別、古典Monte Carlo標本数8を
量子shot数へ追加乗算していないことを確認し、専用payload validatorを通過した。

これは1 snapshot・1 compiler・$q\leq4$のlocal接続検証である。8 trajectoryのcompiled-cost
点推定を精密な候補順位または最終総costには使わない。仮想Hadamard測定は上記の別検証で
追加した。$q=8$ proxyの1 round接続は上記の別検証で追加したが、$q>8$、
全round集計、状態準備、noise、実backend、immutable CIは未評価である。
変更後のlocal全test suiteは`473 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。
保存した接続結果JSONは専用validatorを再通過した。

## 2026-08-26 H4 follow-up・H5 system-size circuit-cost completion note

2026-08-25のH4 follow-upは全jobがreturn code 0で完走し、専用validatorを通過した。同一H4 chain、
距離1.0 Å、STO-3G、DF rank 12、固定snapshot、Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、
seed 17、coupling mapなしの範囲で、$L_D=0$の全Taylor patternを同一trajectoryで比較した
K1--K3 paired residualは全metric最大1.679%、主RZ 95%上側診断3.201%だった。$L_D=6$、
short-step 0.025、$K=2$の固定K1--K4を未使用$L=8$へ適用した点誤差は最大3.750%だが、
主RZ 95%上側診断は7.045%だった。同一trajectoryのpaired K1--K4 $L=8$構造検証は最大4.008%、
主RZ 95%上側診断4.902%だった。controlled $q=1,2$ affine modelの未使用$q=8$ holdoutは
全metric最大0.0529%だった。

H5 chain、距離1.0 Å、STO-3G、10 qubit、project設定DF rank 9、$L_D=4$、short-step 0.025、
$K=2$、同じcompiler条件では、$L=4,6,8$のpaired K1--K3構造残差が全metric最大1.665%、
K1--K4が0.551%だった。5%を満たす最小cluster長としてK1--K3を選んだ。独立calibrationは
最大RZ相対standard error 0.745%で停止し、all-order-0各長さ500、single-order-2各位置125の
独立full holdoutは全metric最大点誤差3.776%、主RZ最大z 2.009、予測側95%半幅1.459%だった。
事前の5%点誤差と2%予測精度を通過した。点wise正規近似95%上側診断7.461%は硬い受理条件ではなく、
rigorousな5%保証とは扱わない。

したがって、exact Hamiltonian snapshot・compiler条件ごとにK1--K3を較正して代表holdoutを行い、
不通過条件だけK4へ進む運用規則を採用する。回路costの広いpilot検証は一旦区切るが、$L>8$、
compiler/coupling/backend変更、新snapshot、controlled scope拡張または5%未満の候補差では再検証する。
これらはlocal dirty-worktree evidenceであり、full RPE、quantum shot、noise、実backend、最終総cost、
immutable CIまたは外部再現結果ではない。

整理後のlocal全test suiteは`472 passed, 4 warnings`だった。4 warningは既存grouped-UWC testの
complex-to-real castであり、今回の回路cost検証由来ではない。H4 paired K4、H5 paired、H5独立
calibration、H5独立holdoutの保存payloadは各専用validatorを再通過した。

## 2026-08-25 circuit-cost follow-up実装note

最初の追加batchはA--Cすべてreturn code 0で完走し、専用validatorを通過した。Aの独立K1--K3は
主誤差12.376%で不通過だが最大z 1.787で原因を分離できず、Bの独立paired K4は最大点誤差
1.281%だがRZ 95%上側診断5.714%、Cのcontrolled $q=8$は全metric最大0.0529%で通過した。

follow-up用に、複数order-2のpaired full/local-window残差、K4 500標本へのincremental resume、
固定K1--K3のL8 holdout、固定K1--K4のL8評価を実装した。変更後のlocal全suiteは
`470 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。follow-upのlive stateは
`artifacts/rte_cost_followup_batch/2026-08-25/status.json`へ保存し、成果物validator通過前は
追加科学結果として扱わない。

## 2026-08-25 circuit-cost追加batch実装note

同一H4 fixed snapshot上で、$K=2$の複数order-2 pattern、$L_D=6$の対応あり独立K4、
controlled $q=8$ holdoutを採取する実装とdetached batch runnerを追加した。
変更後のlocal全suiteは`469 passed, 4 warnings`で、warningは既存grouped-UWC test由来である。
数値jobのlive stateは`artifacts/rte_cost_data_batch/2026-08-25/status.json`に保存する。
完走済み成果物を専用validatorで検証するまでは、追加の科学結果や最終cost評価として数えない。

## 2026-08-24 connected-cluster short-step 0.030 transfer completion note

中断していたH4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、short-step 0.030、
finite Taylor cutoff `K=2`、同一Hamiltonian snapshot・compiler条件のtransfer holdoutを再開した。
固定calibrationに対し、未使用`L=4,6` full回路の全order-0を各1500標本、order-2が1回の条件を
各位置500標本とした。

全6 metricの最大絶対相対点誤差は3.541%、主RZ countは3.488%、RZ最大absolute z-scoreは
3.037、予測側95%相対半幅は1.784%だった。事前の点誤差5%と予測精度2%は通過した。
500/150標本の予備runでは全metric 7.138%、RZ 5.754%、予測半幅2.265%で不通過だったため、
当該超過は再開runで維持されなかった。再開時はcalibration相対標準誤差目標も1.0%から0.8%へ
締めており、改善をholdout標本数だけには帰属しない。一方、RZの点wise正規近似95%上側診断は5.738%であり、
残差ゼロまたはrigorousな5%保証とは扱わない。

再開には旧pattern checkpoint、v3固定sample chunk、既存SQLite metric cacheを併用した。8000回路中
1250件がpersistent cache hit、6750件がmissで、3 workerの実経過時間は393.2秒だった。生成artifactは
内部fingerprint・chunk意味検査を通過し、一時checkpointは残っていない。この結果はdirty local
worktree evidenceであり、別compiler、coupling map、系サイズ、full RPEまたは最終総コストへは
一般化しない。

## 2026-08-24 connected-cluster lightweight operation implementation note

既存の主1500/375 holdout結果を変更せず、compiled-cost処理をoffline calibration、transpileを
呼ばないprediction、固定calibrationのtransfer holdoutへ分離した。v2 calibration/transfer schema、
pattern単位task/checkpoint、same/different基底の直接条件付きsampling、cache状態に依存しない
deterministic-work Neyman配分、完全な数値回路・compiler・backendをkeyとするSQLite metric cacheを
実装した。さらにv3では各patternの標本列を固定indexのsample chunkへ分割し、chunkごとの十分統計量を
atomic checkpointへ保存して合成する。標本数を増やす際は完了済みfull chunkを再利用し、末尾partial
chunkの置換と新規chunkだけを計算する。same/different pairはclass別統計量を合成後に解析確率で
再重み付けする。旧v2 pattern checkpointは標本数まで完全一致する場合だけ読み取り再利用する。
production後の実現相対標準誤差が目標未達なら、分散に基づき不足層へ再配分するadaptive roundも追加した。

同じevent identityを複数short-step時間でcompileする角度不変性validatorも追加した。既存H4固定
snapshot、`L_D=3`、short-step 0.020/0.025/0.030、cluster長1--3、各pattern 2標本のsmokeでは
全6 metric差0だった。ただし低標本のimplementation smokeであり、manifestへ科学的artifactとして
登録せず、角度を除外したstructural cache reuseも有効化しない。別`L_D`、境界coverage、
compiler/coupling条件で検証するまで、short-stepごとに数値回路keyを分離する。

H4のpilot 2・production cap 4 smokeはcold約39.8秒、別checkpointから同じSQLite cacheを使う
warm run約37.3秒だった。warm transpile missは0だが、回路構築とcanonical fingerprint計算が残る。
同一checkpoint再開は完了taskを読み飛ばす。checkpoint fingerprintにはtask実装versionも含め、
実装変更後に旧summaryを黙って再利用しない。固定chunk範囲、十分統計量の合成、標本数増加時の
full chunk再利用、旧checkpoint互換、worker数による科学的出力の不変性、chunk seed改変の拒否を
回帰testで確認した。変更後のlocal全suiteは`468 passed, 4 warnings`で、warningは既存grouped-UWC
test由来である。本節は実装能力の記録であり、
新しいcost精度、full RPE、
最終総コストまたは大規模性能の検証ではない。

## 2026-08-24 operational connected-cluster compiled-cost note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`delta=0.1`、`r=4`、`K=2`の
1 compiler条件で、Taylor次数条件付きのK1--K3 connected-cluster運用推定器を検証した。
order-0単eventは厳密列挙し、pairはsame/different基底で層別、pilotからRZ countのNeyman配分を
決めた。productionとholdoutのseedは分離した。

正確なDF Hamiltonian配列をNPZ snapshotへ固定し、未使用full回路を`L=4,6,8`で評価した。
全order-0は各長さ1500標本、order-2がちょうど1回は各位置375標本である。RZ countと全6 metricの
最大点誤差は2.936%、RZ最大absolute z-scoreは2.074、予測側95%相対半幅は1.537%だった。
事前の点誤差5%と予測精度2%は通過した。一方、点wise正規近似95%上側診断は5.724%で未達、
order-2 K1は要求1535に対して標本cap 1500へ到達した。したがって代表1条件の
「実用点誤差5%内の暫定候補」であり、rigorousな5%保証とは扱わない。

別processで同じ分子条件から再構築したholdoutは、元holdoutと最大11.68%、z 5.215ずれた。
これは主結果へ結合していない。compiled-cost検証の再現単位は分子条件だけでなく正確なDF snapshotとする。
generatorはpilot K1--K3、production K1--K3、holdout L4/L6/L8の9 taskをfingerprinted checkpointへ
保存し、中断後は未完了taskだけを再開する。

全order-0 RZの差は`L=4,6,8`で+2.59%、-0.35%、+2.94%と単調増加せず、order-2が1回の
RZ点誤差は1.02%以下だった。この条件ではK4を追加せず、別`L_D`、short-step、compiler/coupling
条件への移送holdoutで悪化した場合に再検討する。full RPE、量子shot、noise、実backend、
resource accounting接続、最終総コストは未評価である。

主artifact、snapshot、9 checkpoint、versioned source、2 generator、専用test 4件および
[`docs/rte_connected_cluster_cost_validation.md`](docs/rte_connected_cluster_cost_validation.md)を
追加した。fingerprintとsnapshot SHA-256は再検査済みである。artifactはdirty local worktree evidenceで
ありimmutable CI evidenceではない。repository全体の
`overall_status = not_reproducible_from_repository`は変更しない。

当該主artifact生成後に軽量運用APIを追加した最新local全test suiteは`468 passed, 4 warnings`だった。warningは既存grouped-UWC testの
complex-to-real castであり、今回のconnected-cluster検証由来ではない。

## 2026-08-24 hierarchical compiled-cost holdout note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`delta=0.1`、`r=4`の
同一Hamiltonian表現を3 workerへ渡し、`K=0`の未使用`L=8`、`K=2`の`L=4,6`、
controlled partial-S2の未使用`q=4`を独立seedで評価した。三artifactの`preparation_hash`は一致した。

`K=0`では`C2,C3,C8`を各2000標本とし、pair-onlyはcount/sizeで最大8.851%残った一方、
triple補正は全metric最大1.744%、最大absolute z-score 0.521だった。したがって、この条件では
count/sizeにtriple、depthにpairを候補とし、4-event以上の係数は追加しない。

`K=2` runを監査すると、1 eventのorder-2確率は0.0001063で、旧`C1,C2,C3,C4,C6`
全8000 event位置中order-2は1回だけだった。したがって旧4.113%値は`K=2`内部の根拠から外す。
Taylor次数を強制した独立較正/holdoutでは最大点誤差9.119%、最大z 1.651となり、500標本の
独立係数差引きは精度不足と確認した。一方、同一trajectory上で1--3 event局所窓と全回路の
差を直接取る対応あり検証では、`L=4,6`のall-order-0/order-2が1回の全条件・全metricで
最大点誤差1.373%、RZ countの点ごとの正規近似95%診断1.796%だった。最大z 7.535なので
小さい4-event以上の残差は非ゼロだが、代表条件の暫定5%を通過した。運用時は解析的order重みと
対応ありconnected-cluster係数を使い、独立500標本係数推定は使わない。

controlled `q=1,2`各300標本から別seedの`q=4`を予測すると、全metric最大点誤差0.307%、
最大z 0.925、点ごとの正規近似95%上側診断0.958%だった。この代表条件ではaffine `q` model候補を支持するが、
`q>4`、別`L_D`、compiler/coupling条件または最終resource accountingへ一般化しない。

versioned source、並列generator、専用test、五つのfingerprinted local artifactおよび
[`docs/hierarchical_cost_validation.md`](docs/hierarchical_cost_validation.md)を追加した。
fingerprint、source hash、seed分離は再検査済みである。artifactはdirty local worktree evidenceで
ありimmutable CI evidenceではない。repository全体の
`overall_status = not_reproducible_from_repository`は変更しない。

最新local全test suiteは`457 passed, 4 warnings`だった。warningは既存grouped-UWC testの
complex-to-real castであり、今回のcost検証由来ではない。

## 2026-08-23 high-statistics and stratified RTE boundary-cost note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`K=0`、short-step時間0.025の
同一Hamiltonian表現を3 workerへ渡した。独立2 seedについて`C2,C3,C4,C6`を各1000標本、
別のcalibration/holdout seedでfragment-pair補正を1500/1500標本評価した。三artifactの
`preparation_hash`は一致した。

1000標本runでpair補正の最大絶対相対誤差は8.07%、8.18%、最大absolute z-scoreは
2.96、3.02だった。系統差はcount/sizeで確認し、depthではpair-onlyの最大zは1未満だった。
triple補正は最大2.33%、3.73%、最大z 0.79、0.97だったが、`mu3`自体のabsolute z-scoreは
最大1.40なので、非ゼロを確定したとは扱わない。

same-fragment確率は0.7310604だった。different境界をゼロとするsame-only modelは別seedの
pair holdoutに対して最大誤差3.65%、最大z 2.59で外れた。same/different双方の条件付き補正を
解析確率で重み付けすると最大0.849%、最大z 0.587だった。したがってpair係数には少なくとも
二分類が必要であり、長いevent列ではdepthをpair候補、count/sizeをtriple候補とする。

`rte_boundary_pair_validation_v1` source、並列generator、専用test、三つのfingerprinted local
artifactおよび[`docs/rte_boundary_pair_validation.md`](docs/rte_boundary_pair_validation.md)
を追加した。受理閾値、他の`L_D,K`、compiler、controlled回路、resource accounting接続、
最終総コストは未評価である。artifactはdirty local worktree evidenceでありimmutable CI
evidenceではない。repository全体の`overall_status = not_reproducible_from_repository`は
変更しない。

最新local全test suiteは`451 passed, 4 warnings`だった。

## 2026-08-23 RTE boundary-corrected compiled-cost pilot note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`K=0`でshort-step時間を0.025に
固定し、RTE event列のcompiled-cost cluster modelをcalibration/holdout分離して検証した。
`C1`は218 eventを厳密列挙し、`C2,C3`は各300標本で較正した。別seedの未使用`C4,C6`を
各300標本で評価した結果、六指標を通じた最大絶対相対誤差はevent単純和157.51%、pair補正
8.73%、triple補正4.06%だった。最大absolute z-scoreは48.21、1.50、0.52だった。
同一DF fragmentが隣接する確率は0.73106だった。

triple残差は全metricで自身の標準誤差より小さく、depthではpair補正を一様に改善しなかった。
したがってpair補正を次の最小model候補、triple項を高次境界効果の診断量として記録する。
受理閾値、他parameter・compiler条件への一般化、resource accounting接続および最終総コストは
未評価である。

source、generator、専用test 2件、versioned/fingerprinted JSON artifact 1件および
[`docs/rte_boundary_cost_validation.md`](docs/rte_boundary_cost_validation.md)を追加した。
最新local全test suiteは`449 passed, 4 warnings`だった。artifactはdirty local worktree
evidenceでありimmutable CI evidenceではない。したがってrepository全体の
`overall_status = not_reproducible_from_repository`は変更しない。

## 2026-08-23 random-circuit compiled-cost pilot note

H4 chain、距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、`delta=0.1`、`K=0`で、
complete circuitのcompiled costと、部分回路を別々にtranspileしたコスト和をpaired比較した。
`r=1`のpartial-S2は218 trajectoryを完全列挙し、3部分加法モデルは最大0.987%過大評価した。
同じDF表現を共有した`r=2`の100標本では、event別加算がRTE occurrence一体compileを
48.30--57.58%過大評価し、paired differenceのabsolute z-scoreは15.86--17.79だった。
別DF表現の300標本replicateでも52.99--61.15%の過大評価を確認した。
したがって、個々のeventを独立加算するモデルは採用せず、RTE occurrence以上をcost proxyの
最小較正単位とするpilot判断を記録した。

source、generator、専用test 3件、versioned/fingerprinted JSON artifact 3件および
[`docs/random_circuit_cost_validation.md`](docs/random_circuit_cost_validation.md)を追加した。
関連する既存cost testを含む最新local実行は`53 passed`だった。artifactはdirty local worktree
evidenceであり、1条件だけのpilotである。controlled回路、実backend、量子shot、ノイズ、
全round RPE、最終compiled総コスト、または他の`L_D,delta,r,K,q`への一般化を検証したものではない。
また、同じ分子条件でもprocess間でDF `preparation_hash`が変わるため、主$r=1,2$比較は
Hamiltonian共有batchへ置き換えた。異なるhash間の絶対compiled costは直接比較しない。

したがって、repository全体の`overall_status = not_reproducible_from_repository`は変更しない。

## 2026-08-19 current local approximation-validation note

研究内容と現在地の短い統合要約は
[`docs/research/研究概要・現状.md`](docs/research/研究概要・現状.md)を参照する。

2026-08-19のdirty local worktreeでは、最終コスト評価の入力を検証するため、次の三つの
result setを追加した。

- finite-RTE signal、attenuation、radius、phase-boundのH4 grid検証
- PF誤差surrogate、論文Appendix D Eq. (D6)のCPU Qiskit摂動係数、理想QPE分枝の
  H4全`L_D`検証
- H2--H6の実行可能delta窓と、PF演算子を構築しないEq. (D6) state-action係数検証

対応する文書、source、test、dirty-worktree artifactはmanifestへ登録され、構造検査は
成功している。H4全12分割では、well-conditionedな4点でfitしたEq. (D6)係数と
支配固有位相係数の差が最大0.288%だった。H2--H5の実行可能窓では両者の上包絡差が
最大1.144%で、事前の2%条件を全系が通過した。H6（DF rank 11、$L_D=5$）ではEq. (D6)による
`C_use=0.02086663`を得た。local testは`444 passed, 4 warnings`だった。これらは近似手法と実装経路の
local evidenceであり、最終compiled cost、H12の係数、量子shot、ノイズ、または外部から
再現された科学的結論ではない。artifactはimmutable CI evidenceでもない。

したがって、下記auditの`overall_status = not_reproducible_from_repository`は変更しない。
旧DF screeningとprose-only UWCを使用禁止とする判断も引き続き有効である。

## 2026-08-02 implementation hardening note

The current worktree replaces the DF legacy overlap proxy with a
shift-invariant, state-specific survival-phase-bias estimator (cache schema 8,
definition v3). It records explicit estimator status and is marked
`is_rigorous_bound=false`. Legacy/unmarked Cgs tables are now rejected by the
analytic PR-bound screening entry point, so the stale rankings described below
remain invalid and cannot be silently regenerated from the new surrogate.

Finite RTE distribution validation/serialization, actual-circuit/backend cache
identity, exact-zero circuit pruning, pre/post build workload guards, bounded
metric-only LRU caching, online compiled-cost statistics, and rolling
provenance digests were also hardened. A non-scientific Level-5-R regression
fixture now freezes all 32 combinations of `q=1..4`, raw/boundary-optimized,
controlled/uncontrolled, and exact/Monte Carlo compiled-cost evaluation.

The follow-up memory hardening makes event, partial-S2 request, exact
trajectory, and Monte Carlo trajectory generation single-pass. Level-5-R
provenance retains an explicit bounded prefix (1024 records by default) while
rolling digests cover the full stream. Cache-independent total build,
transpile-request, and instruction-application plans are rejected before any
Qiskit builder is called and checked against actual post-build work. The
Level-5-R fixture was rerun without changing gate/depth/mean/standard-error
values; schema 2 adds runtime/PRNG metadata and a digest over all 32 result
streams. The local regression suite reports `224 passed` in the documented
Python 3.11 environment. This remains implementation evidence only; no H3--H14
scientific baseline was generated by this change, and no long RPE circuit,
quantum shot, GPU statevector, noise simulation, or backend job was run.

## 結論

**監査基準 commit [`cf285c0`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/cf285c0ac1e3d587df4a8eb6bee2279a12ced462) の内容だけでは、現在の DF screening / UWC 検証結果を外部から再現・追跡できません。**

ここで「外部から再現可能」とは、clean checkout から、commit 済みの入力と手順を使って結果を再生成し、その結果が公開済みの数値と一致することを確認できる状態を指します。本書は既存 artifact の棚卸しであり、新たな科学計算を実行した結果ではありません。

> **DO NOT USE:** 現在 commit されている DF screening JSON を、修正済みの結果または最終結果として引用しないでください。ファイル内の算術は整合していますが、その Cgs 入力は後の commit で基底状態の不整合を理由に削除され、screening は再生成されていません。

## ステータス一覧

| 対象 | commit `cf285c0` にある証拠 | 判定 | 読み方 |
|---|---|---|---|
| 旧来の高次 Trotter 評価 | 出力付き `abe_trotter_project.ipynb` と、`artifacts/trotter_expo_coeff_gr{,_original}/` 内の係数 pickle 計 540 個 | **historical** | README が説明する旧来の高次 Trotter 解析の成果。現在の DF screening / UWC の検証証拠ではない |
| DF reduced screening | `epsilon_total=1e-4` の JSON 1件。635候補、12分子の best を収録 | **DO NOT USE / stale** | 保存値の加算と best 選択は内部整合するが、元の Cgs 表が削除済みで再生成不能。protocol 上も shortlist 前の近似 screening |
| DF 最終評価 | protocol と実装 | **incomplete** | shortlist の explicit-`L_D` Cgs 再 fit、H14 `8th(Morales)`、`4th(new_2)` が未完了 |
| UWC | 実装説明と H2--H6 等の数値表を含む Markdown | **reported only** | 表が参照する machine-readable JSON は commit されておらず、表から元 run を追跡できない |
| テスト | 4ファイルに `test_*` 関数定義が28件。UWC note に過去の `26 passed` の記録 | **current result unknown** | `cf285c0` に対するテスト実行結果ではない。この変更で追加する manifest 構造検査も科学計算・全 test suite は実行しない |

## 証拠と監査結果

### 1. DF screening

対象 artifact:

- [`artifacts/partial_randomized_pf/screening_results/df_screening_cost_minimization_eps_1.000e-04.json`](artifacts/partial_randomized_pf/screening_results/df_screening_cost_minimization_eps_1.000e-04.json)
- [`Partial Randomized Study Protocol.md`](Partial%20Randomized%20Study%20Protocol.md)
- [`artifacts/partial_randomized_pf/README.md`](artifacts/partial_randomized_pf/README.md)

JSON 自体について確認できる範囲は次のとおりです。

- `candidates` は635件で、1件は `(molecule, PF, L_D)` の組です。
- `best_by_molecule` は H3--H14 の12件です。
- 全635候補で、保存値の `g_total` は `g_det + g_rand` と一致します（最大絶対差 0）。
- 12件の `best_by_molecule` は、それぞれ同じ molecule の候補中で最小の `g_total` と一致します。

これは **JSON 内部の算術と選択処理だけ** の確認です。入力データ、Cgs fit、物理モデル、または結果の科学的妥当性を検証したことにはなりません。

再現性を失っている直接の理由は次のとおりです。

1. JSON の `cgs_table` は `/home/AbeHiromu/Project/.../df_cgs_cost_table.json` という生成環境の絶対パスを指します。
2. commit [`98f960c` (`基底状態ずれてたので削除`)](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/98f960c2dd09fc1ae6b8b5c802dc5ce84fc61604) は、集約 Cgs 表、split 表、index の計37ファイルを削除しています。
3. その後も上記 screening JSON は残っていますが、削除理由を反映した正しい Cgs 入力から再生成された artifact はありません。

また protocol は、この計算を候補を絞るための近似と定義しています。screening では anchor の `C_gs,D(p,L_anchor)` を各 `L_D` に使い回し、**最終評価では shortlist の各 `(p, L_D)` で Cgs を再 fit して `G_total` を再計算する必要があります**。同じ protocol には、次も未完了と記録されています。

- H14 `8th(Morales)` の anchor Cgs
- H3--H14 `4th(new_2)` の anchor Cgs 計算、cost table への merge、再 screening
- shortlist に対する explicit-`L_D` Cgs の再 fit

したがって、入力問題がなかったとしても現在の JSON は最終結果ではありません。

### 2. UWC

[`notes/uwc_current_implementation_and_results.md`](notes/uwc_current_implementation_and_results.md) には、H2--H6 grouped UWC、H3 time-grid 診断、theta sweep、simple shift の条件と数値表があります。一方、同文書が参照する次の出力を含む `artifacts/grouped_uwc_pf_qpe/` は commit `cf285c0` に存在せず、`.gitignore` でディレクトリ全体が除外されています。

- `H2_H6_2nd_grouped_uwc_alpha_bliss_quadratic_theta_0p01_gpu.json`
- `H3_bliss_sector_scaling_diagnostics.json`
- `H2_H6_2nd_grouped_uwc_alpha_simple_shift_gpu.json`
- theta sweep 表の元になった run 出力

したがって Markdown の表は「報告された数値」として読めますが、repository 内の canonical raw/summary artifact と照合することはできません。なお文書自身の結論も、現在の simple BLISS quadratic shift では grouped PF+QPE cost がほぼ低下していない、という限定的なものです。

### 3. テストと CI

commit `cf285c0` の `tests/` には、静的に数えた `test_*` 関数定義が28件あります。

- `tests/test_df_hamiltonian.py`: 5件
- `tests/test_df_partial_randomized_pf.py`: 9件
- `tests/test_grouped_uwc_comparison.py`: 7件
- `tests/test_uwc_preprocessor.py`: 7件

UWC note が保存している実行記録は `.venv/bin/python -m pytest -q` の `26 passed` です。これは後から追加されたテストを含む現在の suite に対する結果ではなく、実行 commit、依存環境、完全なログも記録されていません。監査基準 commit `cf285c0` には `.github/workflows/` もありませんでした。この変更では manifest と記載パスの構造検査だけを追加しており、科学計算または全 test suite の CI ではありません。このため、`cf285c0` の28定義が pass するとは本監査から主張できません。

## 再現を妨げているもの

- 修正済みの DF Cgs 集約表・split 表・index がない。
- stale screening JSON に入力 hash、生成元 commit、実行環境、実行 command がない。
- DF screening の修正後再実行と shortlist の explicit-`L_D` 再 fit がない。
- protocol に記載された H14 `8th(Morales)` と `4th(new_2)` が未完了。
- UWC の Markdown 表に対応する machine-readable run artifact がない。
- UWC artifact の保存先が `.gitignore` され、レビュー可能な canonical summary の例外設定がない。
- 現在の HEAD を対象とする自動テスト結果がない。

## 「検証完了」とするための条件

以下をすべて満たした時点で、DF / UWC の結果を repository から外部検証可能と扱います。

1. **DF 入力を修正して固定する。** ground-state のずれを修正した Cgs を再計算し、集約表、全 split 表、index を同時生成する。各表に molecule、PF、`L_D`、入力 Hamiltonian hash、生成元 commit、生成 command を記録し、相互の件数と hash を検査する。
2. **未完了の DF ケースを埋める。** H3--H14 `4th(new_2)` と H14 `8th(Morales)` の必要な anchor Cgs を生成し、同じ canonical table に merge する。失敗または除外する場合は、対象、理由、結果への影響を明記する。
3. **screening を再生成する。** 修正後の canonical table だけを入力として `epsilon_total=1e-4` screening を実行する。結果には相対的な入力 path、全入力の content hash、生成元 commit、command、依存環境、candidate 件数を保存する。`g_total = g_det + g_rand` と molecule ごとの best 選択を自動検査し、旧 JSON を stale として置換または明確に隔離する。
4. **最終 DF 評価を実行する。** screening の shortlist と選定規則を保存し、各 `(PF, L_D)` で anchor ではない explicit-`L_D` Cgs を再 fit して `G_total` を再計算する。最終表から各 fit の machine-readable artifact と入力 hash へ追跡できるようにする。
5. **UWC の根拠データを公開する。** Markdown に載せる全表について canonical JSON/CSV を commit し、条件、baseline、seed（使用時）、backend、入力 hash、生成元 commit、command を保存する。Markdown の値が artifact から自動生成または自動照合されるようにし、必要な summary だけを `.gitignore` の例外にする。
6. **clean checkout で検証する。** 固定した依存環境と文書化した command で、小規模な end-to-end 再生成および全 test suite を CI から実行する。結果 artifact を作った commit に対する成功 check を GitHub 上に残し、比較 tolerance と期待値を test または検証 script に固定する。

上記が完了するまでは、旧来の高次 Trotter artifact、DF screening、UWC 表を互いに独立した進捗資料として扱い、現在の partial-randomized DF/UWC の完成済み検証結果として一括して引用しないでください。
