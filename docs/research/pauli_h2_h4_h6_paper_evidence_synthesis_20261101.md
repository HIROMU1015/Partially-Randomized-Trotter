# Pauli-JW H2・H4・H6時間発展：論文用証拠統合

2026-11-01。Results・Methods草稿と一次証拠の対応を整理した文書専用成果物である。**固定された有限候補・仮想compiled残差ρ=0・保存価格標本の比較では、純RTEはHybrid PRより15精度条件中14条件で低いGを示し、H4のε_sig=.05のみHybrid PRが低かった。** H6は10/27のHybrid大event追加検証を含む。これは主要claimの正式採択、独立科学レビュー、新規性・投稿可否の判断ではない。

新規科学計算、保存値の数値再解析、候補選択・再ランキング、sampling、simulation、build/compile、Hamiltonian/state/reference生成は0件である。既存の式・値・選抜結果を引用し、新しいG・倍率・SE・図の数値は生成していない。相互参照：[主張・根拠・図表対応表](pauli_h2_h4_h6_claim_figure_map_20261101.md)。

## 1. 研究目的・比較対象

本研究系列は、決定論的なPauli項の保持と有限RTEへのランダム化の配分が、単回のcompiled回路価格および同一信号精度に必要なshot負担へ与える影響を調べたものである。比較の目的はHybrid化の優位を前提とせず、切り捨て・全保持・部分保持・全非identity項のランダム化を同じ費用会計で扱うことにある。

| 方式 | 定義 | 本資料での区別 |
|---|---|---|
| Full PF | 全M非identity Pauli項をS2で実行しconstantを保持 | qを比較した有限Full PF familyの保存選択 |
| B0/B0I | 係数順prefixだけを保持し、残りの非identity項を切り捨てる | constantを常に保持するPauli実装では同じprefix/qのB0・B0Iは同等。二つの独立方式と数えない |
| Hybrid PR | 0<L_D<MをS2で保持し、tailをfinite RTEで処理 | 保持half→r event→保持reverse halfをq step |
| 純RTE | L_D=0、全非identity項をfinite RTEへ置く | constant phaseは保持。非自明な保持S2誤差をこの境界に帰属しない |

旧資料のPR familyは純RTE境界も含む。「PR familyの低G」は「決定論的prefixを残すこと自体の優位」を意味しない。Full PFとの比較と、B0/B0Iを含む決定論的familyの最良保存値との比較も異なる。[既存統合資料][E8]、[H4 PR探索][H4report]。

## 2. 固定科学条件と費用の定義

| 条件 | H2 | H4 | H6 |
|---|---|---|---|
| 配置・電子状態 | linear、隣接1.00 Å、STO-3G、中性singlet | 同左 | 同左 |
| Pauli非identity項M / system qubit数 | 14 / 4 | 184 / 8 | 918 / 12 |
| state / reference | 旧non-DF pilotの保存raw state / 保存full-H reference | 保存共通raw ray / 保存full-H reference区間 | 元研究でPauli full-Hのsector最低固有vectorを保存 / 保存Duhamel reference |
| 分子内の共有条件 | 保存H、raw ray、共通exact norm | 同左 | 同左 |

共通設定はS2、T=.8、central-rotation制御、cosineと+Im sineのfull Hadamard測定回路、Qiskit1.3.0、basis `rz,sx,x,cx`、optimization_level=1、seed=17、追加topology/coupling/layout/routing指定なしである。状態準備費用はP=0とする。prefixは保存abs係数降順・label tie-breakで定義し、constantを別枠で保持する。分子間でstate/reference・標本を混合しない。H6のstate生成は過去の元研究の処理であり、今回実行したものではない。[入力と元研究][D26report]、[既存統合§2][E8]。

軸aについて、補正信号へB_savedを一回適用したideal-law biasをb_a、数値包含allowanceをu_a、実compiled残差をρ_aとする。既存のε_a=元binary64 ε_sig/√2、α_a=.025、h_a=ε_a−|b_a|−u_a−ρ_a、h_a>0、(u_a+ρ_a)/h_a≤.01の条件下で、保存shot十分式は

$$
N_a=\left\lceil A_{a,\mathrm{upper}}/h_a^2\right\rceil,
\qquad A_{a,\mathrm{upper}}\ge 2B_{\mathrm{up}}^2\ln(2/\alpha_a),
\qquad \widehat G=N_{\cos}\widehat C_{\cos}+N_{\sin}\widehat C_{\sin}
$$

である。A_upperは保存会計の外向き上端を指し、この式を今回評価していない。B_upの旧log marginと後続の共通会計を無断で同一化しない。決定論的unitary候補では数学的B=1を用いる。主比較は仮想ρ=0であり、実ρ=0または実ρのguard所属を証明した結果ではない。

C_aは状態準備を除くfull-wrapper compiled RZ数、Gの単位はRZ×shotである。Nは量子測定shotの十分条件による会計値で、実量子shotをN回実施した値ではない。nは古典的なtrajectory価格標本数であり、探索と独立確認をpoolしない。同一trajectoryの両軸価格からGのSEを評価した保存paired会計を用いる。決定論的n=1にはMC標本誤差のSEを付けない。[原因調査§2][E7]、[H2会計][H2report]。

finite RTEのKは最大偶数event次数で、K2/K4はorder0,2 / order0,2,4を許す（多項式次数K+1）。S2の次数とは別である。s=q*rは総event数であるが、同じsのq/r分解がlaw・seed・浮動小数点処理・compiled価格まで同一とは仮定しない。CDF/53bit uniform-wordのeffective lawとIID仮定を維持し、PRNG streamの数学的独立性を証明したとはしない。[既存Methodsの構成][oldmethods]、[今回のPauli-only正本][E8]。

## 3. 版・SOURCE・選抜規則

| 証拠ID | 結果の固定commit・入口 | 科学SOURCE・用途 |
|---|---|---|
| E1 | `8fd1ceb283309a7e515ff4a533a253260bdd0f78`、[H2/H6保存family表][E1] | 元native SOURCE `d17780fcb57bcff344325a8632f718c0cc56f1dc`、数値 `0880c38e3e8a6ea57882ec66e6366a8b86e408e2`、比較 `a8c7c7dca5a03aae1b008a09048d89c216bbf89c`。保存Full PF/B0と旧Hybrid。来歴は[元報告][D26report]・[SOURCE bundle][nativebundle]・[numeric SOURCE][numeric_source] |
| E2 | `fbfbfd21a252a4a32f52e3b40e50acd91b3c626a`、[H4決定論的表][E2]・[全候補行][E2all] | `47384f5698094e466e83027a5a54751e0cf6000e` / `8ab472db8f90d6f64f6cc626584655c5a7487a56`。106構成の保存最良値 |
| E3 | `d3fdfad880902b7ff35b710d22e1c9759f7087c2`、[H4 PR探索主表][E3] | worker `52865921930b3afd11b1447dff0a6cbcb6085af8`、選択 `4a7f91fa7ba8aa3ae1ce4ccb30bde9b973869410`、最終比較 `a17ad381b6ff2ac615e3cd930c641947c551bcbc` |
| E4 | `4aea054b3369709bc0d35ff081071e62b5f77cbe`、[H2追加確認][E4] | [SOURCE][H2source] `bf5f2339f723f4edb7bf91a8ca88f33251bbef57`。Phase Aは旧target/Nの再利用と新確認価格。今回の草稿作成での再会計ではない |
| E5 / E5b | `10cb3e5cca7834ebf75301bd91bd093ca51d4a6b`、[選択行][E5]。旧直接比較は `90c1c60820d52ca7ddcd4fb5315e9d21bb4a86c3` の[費用表][E5b] | event [SOURCE][extsource] `e755ecf581600192fd2745d7066e410cebd933d3`。H4旧値・Hybridを再計算したSOURCEではない。新確認価格の[直接比較SOURCE][directsource]は `b19ed83c6b4b64821965fa0b756cd2691e34eda8` |
| E6 | `00958ae1945c780718b2deef07d5dab1096a5cd8`、[全候補][E6]・[選択比較][E6compare] | [SOURCE][H6source] `887b69e449cb3c378286f09d0d7626599850fb67`、解析 `af891fd45805454c6592bbee5441266f14235ffd`。固定4Hybrid候補の追加 |
| E7 / E8 | `5432857b94c3714ea9f4d6031e493e8dfc308a91` の[原因調査][E7]、`b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b` の[統合資料][E8]・[latest15][E8csv] | 文書・既存値の引用。新しい科学SOURCEではない |

E8のlatest15は10/26までの値で、H6 Hybridは追加前である。本資料では**H6の5行だけE6の保存expanded_Hybrid結果を引用**する。H2/H4の保存選抜、Full PF/B0対照、旧latest15自体は変更しない。[10/25統合][oldintegrated]の主表はPR対決定論的比較であり、現在のHybrid対純RTE表と同じ評価時点ではない。

| 分子・段階 | 保存候補集合・価格標本 | 選抜の意味 |
|---|---|---|
| H2 | 元120構成（決定論的56、PR64）。旧Hybrid n96。未確認の既存純RTE8候補を追加しn96→必要時192/384 | 最新純RTEの確認集合min。追加後minはs4、選択後・逐次延長の効果を含む。Phase Aのtarget/Nは旧値再利用 |
| H4 | 決定論的106、PR141。元確認はHybrid p36のK2/K4各n48、純RTE p0のK2/K4各n384。後続p1確認n48 | E3主表は探索nomineeを確認前に凍結。現在のHybrid/RTE直接比較は確認集合minで、.05のHybrid K4と.01のRTE K2が旧主表と異なる |
| H6 | 元215構成、純RTE fairness→event拡張。s512の純RTE確認n48。追加Hybridは固定4候補を探索n128・確認各n64 | E6は探索nomineeと確認後minが全5精度でp78/q8/r32/K2。旧Hybrid anchor、純RTE成功cohortの再取得はなし |

H2/H4/H6の5要求精度はそれぞれ同じ価格標本を共有する。15行を15個の独立実験として数えない。確認標本を使用していても、確認後の最小選択は全候補の母平均winner保証ではない。[H2 coverage][h2coverage]、[H4 nomination][h4nomination]、[H6設計][h6sampling]。

## 4. 最新の四方式比較と分子別Results

### Table 1：保存family選択値の併記

全GはRZ×shot、分数は保存exact文字列である。PF/B0列は各有限familyの既存選択値、Hybrid/RTE列は上記最新確認集合の選択値であり、この4列を今回再ランキングしていない。H/Rは保存Hybrid/RTE比で、新しく比を計算していない。

| 分子 | ε_sig | Full PF G | B0/B0I G | Hybrid PR G | 純RTE G | 保存H/R exact | 保存比較で低G |
|---|---|---|---|---|---|---|---|
| H2 | 0.05 | 1353240 | 2229334 | 1251625/2 | 27036763/48 | 30039000/27036763 | 純RTE |
| H2 | 0.01 | 45976320 | — | 250331975/16 | 28598617/2 | 250331975/228788936 | 純RTE |
| H2 | 0.005 | 293617350 | — | 500687975/8 | 922336283/16 | 1001375950/922336283 | 純RTE |
| H2 | 0.001 | 15183677784 | — | 25043990925/16 | 23061208939/16 | 25043990925/23061208939 | 純RTE |
| H2 | 0.0005 | 77733393012 | — | 200448054525/32 | 138388318709/24 | 601344163575/553553274836 | 純RTE |
| H4 | 0.05 | 32680930 | 23237760 | 47945769/4 | 1613053221/128 | 511421536/537684407 | Hybrid PR |
| H4 | 0.01 | 1723689280 | 1440845160 | 15733989557/48 | 40329828201/128 | 9682455112/9306883431 | 純RTE |
| H4 | 0.005 | 10167897280 | 9734669448 | 21002399873/16 | 242000558639/192 | 36004114068/34571508377 | 純RTE |
| H4 | 0.001 | 615615107120 | 579283719324 | 794800405295/24 | 2016670655853/64 | 6358403242360/6050011967559 | 純RTE |
| H4 | 0.0005 | 2982596955988 | 2663993522484 | 134004611546 | 8066682748313/64 | 8576295138944/8066682748313 | 純RTE |
| H6 | 0.05 | 223024650 | 182956812 | 3435079271/32 | 570714023/6 | 10305237813/9131424368 | 純RTE |
| H6 | 0.01 | 12156251072 | 10708083010 | 5480745435/2 | 114140561815/48 | 26307578088/22828112363 | 純RTE |
| H6 | 0.005 | 75555076284 | 65486795360 | 720445857737/64 | 19023451889/2 | 720445857737/608750460448 | 純RTE |
| H6 | 0.001 | 4001626381500 | 2989511806608 | 22654881652461/64 | 2853557405885/12 | 67964644957383/45656918494160 | 純RTE |
| H6 | 0.0005 | 22193309922300 | 12486761806824 | 63691899506611/32 | 45657735316457/48 | 191075698519833/91315470632914 | 純RTE |

H2のB0の「—」は元表の `NO_ELIGIBLE_REGISTERED_CANDIDATE` であり、0費用、無限費用、全B0の不可能性を表さない。決定論的SEの空欄も0推定で補わない。表の候補ID・q/r/K・n・G SEと元CSVの行選択は本節末の詳細表に収録する。

### H2

保存Hybrid n96との比較は、有望な既存純RTE候補を追加独立確認した後、全5精度で純RTE低Gへ更新された。最新確認集合minは.05/.01でq1/r4/K2、他3精度でq2/r2/K4（いずれもs4、n384）である。保存H/Rは約1.086–1.111。s64/128への追加探索では改善せず、追加独立確認が主要な更新要因である。

粗精度の純RTEはHybridよりB・Nが大きいにもかかわらずCが低く、Gが低い（§6）。したがって、H2の低費用をnormalizationの低下だけで説明することはできない。Full PFは元の決定論的familyの選択、B0は.05のみ適格保存候補がある。[H2報告][H2report]、[E1][E1]。

### H4

現在の確認集合min比較では、.05はHybrid p36/q2/r8/K4が低Gで、残る4精度は純RTEが低い。純RTEの低費用候補はs64に留まり、s128/256ではB/Nが下がっても単回Cの増加が上回った。厳精度では同じq8/r8/K4のL0/L1対照があるが、prefix変更はtail・law・B・biasも変える。

決定論的106構成では、B0/B0Iの保存prefix/qは132/1、144/2、176/2、180/8、180/8である。141 PR構成の探索後、事前固定nomineeの独立確認主表では全5精度でこの最良決定論的値を下回った。.05の旧固定p27 PRに対するB0優位は、この後続PR探索の結論へ持ち越さない。[H4 PR探索][H4report]、[B0比較][D4report]。

### H6

純RTEのq8/K2系列ではs512の確認Gが低く、s1024の探索Gで増加した。Hybrid側もprefix78/q8/K2のrを拡張するとr32が低Gとなり、r64でGが増加した。この固定追加集合でも全5精度で純RTE低Gが維持され、保存H/Rは約1.13–2.09である。Hybrid探索機会の大event方向を補った結果であり、全prefix/q/r/Kでの対称探索や大域最適化を完了した意味ではない。

最厳精度では純RTEの小さいbiasと広いheadroomが、BがHybridより大きくても両軸Nが小さいことを会計上説明する。粗精度では純RTEのNの方が大きく、低いCがG差を支える。同じ分子でも精度によりN/Cの競合が異なる。[event拡張][extreport]、[Hybrid追加検証][H6report]、[原因調査][E7]。

### Table S1：候補・標本・SEの追跡表

行の対応は分子（列または出典directory）、epsilon_sig、candidate IDで示す。cohort列を持つ表ではdeterministic / confirmationを指定する。H4の決定論的詳細はE2all、E3の探索nomineeは次節で別掲する。q/r/Kの「—」は決定論的方式で未使用の欄。SEはGの保存SEでありCのSEではない。出典IDは§3の結果commitと科学SOURCEを対応させる。

### H2：固定選択行の詳細

| ε_sig | 方式 | 保存candidate ID | L_D | q/r/K | cohort n | G exact | G SE | 出典 |
|---|---|---|---|---|---|---|---|---|
| 0.05 | Full PF | `PF_p0014_q1` | 14 | 1/—/— | det / 1 | 1353240 | — | [E1][E1] |
| 0.05 | B0/B0I | `B0_p0013_q1` | 13 | 1/—/— | det / 1 | 2229334 | — | [E1][E1] |
| 0.05 | Hybrid PR | `PR_p0001_q1_r4_K4` | 1 | 1/4/4 | conf / 96 | 1251625/2 | 20808.97393554041 | [E1][E1] |
| 0.05 | 純RTE | `PR_p0000_q1_r4_K2` | 0 | 1/4/2 | conf / 384 | 27036763/48 | 11489.641344756476 | [E4][E4] |
| 0.01 | Full PF | `PF_p0014_q1` | 14 | 1/—/— | det / 1 | 45976320 | — | [E1][E1] |
| 0.01 | B0/B0I | 適格登録候補なし | — | —/—/— | det / — | — | — | [E1][E1] |
| 0.01 | Hybrid PR | `PR_p0001_q1_r4_K4` | 1 | 1/4/4 | conf / 96 | 250331975/16 | 520238.84380379034 | [E1][E1] |
| 0.01 | 純RTE | `PR_p0000_q1_r4_K2` | 0 | 1/4/2 | conf / 384 | 28598617/2 | 291680.9403132072 | [E4][E4] |
| 0.005 | Full PF | `PF_p0014_q1` | 14 | 1/—/— | det / 1 | 293617350 | — | [E1][E1] |
| 0.005 | B0/B0I | 適格登録候補なし | — | —/—/— | det / — | — | — | [E1][E1] |
| 0.005 | Hybrid PR | `PR_p0001_q1_r4_K4` | 1 | 1/4/4 | conf / 96 | 500687975/8 | 2081055.2325204255 | [E1][E1] |
| 0.005 | 純RTE | `PR_p0000_q2_r2_K4` | 0 | 2/2/4 | conf / 384 | 922336283/16 | 1191602.831955389 | [E4][E4] |
| 0.001 | Full PF | `PF_p0014_q4` | 14 | 4/—/— | det / 1 | 15183677784 | — | [E1][E1] |
| 0.001 | B0/B0I | 適格登録候補なし | — | —/—/— | det / — | — | — | [E1][E1] |
| 0.001 | Hybrid PR | `PR_p0001_q1_r4_K4` | 1 | 1/4/4 | conf / 96 | 25043990925/16 | 52046315.23022426 | [E1][E1] |
| 0.001 | 純RTE | `PR_p0000_q2_r2_K4` | 0 | 2/2/4 | conf / 384 | 23061208939/16 | 29793690.638132833 | [E4][E4] |
| 0.0005 | Full PF | `PF_p0014_q4` | 14 | 4/—/— | det / 1 | 77733393012 | — | [E1][E1] |
| 0.0005 | B0/B0I | 適格登録候補なし | — | —/—/— | det / — | — | — | [E1][E1] |
| 0.0005 | Hybrid PR | `PR_p0001_q1_r4_K4` | 1 | 1/4/4 | conf / 96 | 200448054525/32 | 208285146.4116902 | [E1][E1] |
| 0.0005 | 純RTE | `PR_p0000_q2_r2_K4` | 0 | 2/2/4 | conf / 384 | 138388318709/24 | 119192905.75654344 | [E4][E4] |

### H4：固定選択行の詳細

| ε_sig | 方式 | 保存candidate ID | L_D | q/r/K | cohort n | G exact | G SE | 出典 |
|---|---|---|---|---|---|---|---|---|
| 0.05 | Full PF | `p184_q1` | 184 | 1/—/— | det / 1 | 32680930 | — | [E2][E2] |
| 0.05 | B0/B0I | `p132_q1` | 132 | 1/—/— | det / 1 | 23237760 | — | [E2][E2] |
| 0.05 | Hybrid PR | `p036_q2_r8_K4` | 36 | 2/8/4 | conf / 48 | 47945769/4 | 37995.654220550095 | [E5b][E5b] |
| 0.05 | 純RTE | `p000_q8_r8_K2` | 0 | 8/8/2 | conf / 384 | 1613053221/128 | 44580.37483807345 | [E5][E5] |
| 0.01 | Full PF | `p184_q2` | 184 | 2/—/— | det / 1 | 1723689280 | — | [E2][E2] |
| 0.01 | B0/B0I | `p144_q2` | 144 | 2/—/— | det / 1 | 1440845160 | — | [E2][E2] |
| 0.01 | Hybrid PR | `p001_q8_r8_K4` | 1 | 8/8/4 | conf / 48 | 15733989557/48 | 2844998.0498678633 | [E5b][E5b] |
| 0.01 | 純RTE | `p000_q8_r8_K2` | 0 | 8/8/2 | conf / 384 | 40329828201/128 | 1114606.0371405967 | [E5][E5] |
| 0.005 | Full PF | `p184_q2` | 184 | 2/—/— | det / 1 | 10167897280 | — | [E2][E2] |
| 0.005 | B0/B0I | `p176_q2` | 176 | 2/—/— | det / 1 | 9734669448 | — | [E2][E2] |
| 0.005 | Hybrid PR | `p001_q8_r8_K4` | 1 | 8/8/4 | conf / 48 | 21002399873/16 | 11392873.968442421 | [E5b][E5b] |
| 0.005 | 純RTE | `p000_q8_r8_K4` | 0 | 8/8/4 | conf / 384 | 242000558639/192 | 4866794.412346326 | [E5][E5] |
| 0.001 | Full PF | `p184_q8` | 184 | 8/—/— | det / 1 | 615615107120 | — | [E2][E2] |
| 0.001 | B0/B0I | `p180_q8` | 180 | 8/—/— | det / 1 | 579283719324 | — | [E2][E2] |
| 0.001 | Hybrid PR | `p001_q8_r8_K4` | 1 | 8/8/4 | conf / 48 | 794800405295/24 | 287429401.7937058 | [E5b][E5b] |
| 0.001 | 純RTE | `p000_q8_r8_K4` | 0 | 8/8/4 | conf / 384 | 2016670655853/64 | 121669820.11916487 | [E5][E5] |
| 0.0005 | Full PF | `p184_q8` | 184 | 8/—/— | det / 1 | 2982596955988 | — | [E2][E2] |
| 0.0005 | B0/B0I | `p180_q8` | 180 | 8/—/— | det / 1 | 2663993522484 | — | [E2][E2] |
| 0.0005 | Hybrid PR | `p001_q8_r8_K4` | 1 | 8/8/4 | conf / 48 | 134004611546 | 1163065295.2161603 | [E5b][E5b] |
| 0.0005 | 純RTE | `p000_q8_r8_K4` | 0 | 8/8/4 | conf / 384 | 8066682748313/64 | 486679288.01218957 | [E5][E5] |

### H6：固定選択行の詳細

| ε_sig | 方式 | 保存candidate ID | L_D | q/r/K | cohort n | G exact | G SE | 出典 |
|---|---|---|---|---|---|---|---|---|
| 0.05 | Full PF | `PF_p0918_q1` | 918 | 1/—/— | det / 1 | 223024650 | — | [E1][E1] |
| 0.05 | B0/B0I | `B0_p0724_q1` | 724 | 1/—/— | det / 1 | 182956812 | — | [E1][E1] |
| 0.05 | Hybrid PR | `PR_p0078_q8_r32_K2` | 78 | 8/32/2 | conf / 64 | 3435079271/32 | 138534.1239994281 | [E6][E6] |
| 0.05 | 純RTE | `H6_p000_q8_r64_K2` | 0 | 8/64/2 | conf / 48 | 570714023/6 | 355807.7258949337 | [E5][E5] |
| 0.01 | Full PF | `PF_p0918_q2` | 918 | 2/—/— | det / 1 | 12156251072 | — | [E1][E1] |
| 0.01 | B0/B0I | `B0_p0830_q2` | 830 | 2/—/— | det / 1 | 10708083010 | — | [E1][E1] |
| 0.01 | Hybrid PR | `PR_p0078_q8_r32_K2` | 78 | 8/32/2 | conf / 64 | 5480745435/2 | 3536548.453418626 | [E6][E6] |
| 0.01 | 純RTE | `H6_p000_q8_r64_K2` | 0 | 8/64/2 | conf / 48 | 114140561815/48 | 8895018.366266174 | [E5][E5] |
| 0.005 | Full PF | `PF_p0918_q4` | 918 | 4/—/— | det / 1 | 75555076284 | — | [E1][E1] |
| 0.005 | B0/B0I | `B0_p0830_q4` | 830 | 4/—/— | det / 1 | 65486795360 | — | [E1][E1] |
| 0.005 | Hybrid PR | `PR_p0078_q8_r32_K2` | 78 | 8/32/2 | conf / 64 | 720445857737/64 | 14527515.657820158 | [E6][E6] |
| 0.005 | 純RTE | `H6_p000_q8_r64_K2` | 0 | 8/64/2 | conf / 48 | 19023451889/2 | 35580120.073359944 | [E5][E5] |
| 0.001 | Full PF | `PF_p0918_q8` | 918 | 8/—/— | det / 1 | 4001626381500 | — | [E1][E1] |
| 0.001 | B0/B0I | `B0_p0864_q8` | 864 | 8/—/— | det / 1 | 2989511806608 | — | [E1][E1] |
| 0.001 | Hybrid PR | `PR_p0078_q8_r32_K2` | 78 | 8/32/2 | conf / 64 | 22654881652461/64 | 456827038.9755441 | [E6][E6] |
| 0.001 | 純RTE | `H6_p000_q8_r64_K2` | 0 | 8/64/2 | conf / 48 | 2853557405885/12 | 889515353.0322384 | [E5][E5] |
| 0.0005 | Full PF | `PF_p0918_q8` | 918 | 8/—/— | det / 1 | 22193309922300 | — | [E1][E1] |
| 0.0005 | B0/B0I | `B0_p0864_q8` | 864 | 8/—/— | det / 1 | 12486761806824 | — | [E1][E1] |
| 0.0005 | Hybrid PR | `PR_p0078_q8_r32_K2` | 78 | 8/32/2 | conf / 64 | 63691899506611/32 | 2568645672.459057 | [E6][E6] |
| 0.0005 | 純RTE | `H6_p000_q8_r64_K2` | 0 | 8/64/2 | conf / 48 | 45657735316457/48 | 3558125067.408183 | [E5][E5] |

## 5. 決定論的familyへの低費用性：保存された比較量の範囲

Full PFだけをbaselineとすると、H4/H6の良好なB0を落とすことになる。Table 1ではFull PFとB0を別列に保持した。元比較で選択された決定論的family代表はH2がFull PF、H4/H6がB0/B0Iである。以下は[既存統合CSV][oldratio]の保存PR/D比をそのまま示す。Dはその時点の最良保存決定論的family値である。

| 分子 | ε_sig | 10/25保存PR候補 | 方式分類 | 保存決定論的family候補 | 保存PR/D exact | 選抜規則 |
|---|---|---|---|---|---|---|
| H2 | 0.05 | `PR_p0001_q1_r4_K4` | Hybrid PR | `PF_p0014_q1` | 250325/541296 | 確認集合min |
| H2 | 0.01 | `PR_p0001_q1_r4_K4` | Hybrid PR | `PF_p0014_q1` | 50066395/147124224 | 確認集合min |
| H2 | 0.005 | `PR_p0001_q1_r4_K4` | Hybrid PR | `PF_p0014_q1` | 20027519/93957552 | 確認集合min |
| H2 | 0.001 | `PR_p0001_q1_r4_K4` | Hybrid PR | `PF_p0014_q4` | 8347996975/80979614848 | 確認集合min |
| H2 | 0.0005 | `PR_p0001_q1_r4_K4` | Hybrid PR | `PF_p0014_q4` | 66816018175/829156192128 | 確認集合min |
| H4 | 0.05 | `p036_q2_r8_K2` | Hybrid PR | `p132_q1` | 20548399/39836160 | 探索nominee確認 |
| H4 | 0.01 | `p000_q8_r8_K4` | 純RTE | `p144_q2` | 12100034177/55328454144 | 探索nominee確認 |
| H4 | 0.005 | `p000_q8_r8_K4` | 純RTE | `p176_q2` | 34571508377/267008076288 | 探索nominee確認 |
| H4 | 0.001 | `p000_q8_r8_K4` | 純RTE | `p180_q8` | 672223551951/12358052678912 | 探索nominee確認 |
| H4 | 0.0005 | `p000_q8_r8_K4` | 純RTE | `p180_q8` | 1152383249759/24356512205568 | 探索nominee確認 |
| H6 | 0.05 | `PR_p0684_q1_r8_K4` | Hybrid PR | `B0_p0724_q1` | 1380373939/1463654496 | 確認集合min |
| H6 | 0.01 | `PR_p0078_q8_r8_K2` | Hybrid PR | `B0_p0830_q2` | 2333007872/3212424903 | 確認集合min |
| H6 | 0.005 | `PR_p0078_q8_r8_K2` | Hybrid PR | `B0_p0830_q4` | 153337395055/314336617728 | 確認集合min |
| H6 | 0.001 | `PR_p0078_q8_r8_K2` | Hybrid PR | `B0_p0864_q8` | 2678777796305/7972031484288 | 確認集合min |
| H6 | 0.0005 | `PR_p0078_q8_r8_K2` | Hybrid PR | `B0_p0864_q8` | 22593289103905/49947047227296 | 確認集合min |

H2のこの表のPRは現行と同じHybridであり、最新純RTEの追加確認前である。H4は.05がHybrid K2、.01が純RTE K4というE3の固定nominee主表であり、現在の確認後minへ差し替えていない。H6はHybrid追加前のp684/r8またはp78/r8であり、p78/r32の新倍率ではない。

H4の元主表は、最良決定論的に対する保存削減率48.42%、78.13%、87.05%、94.56%、95.27%を報告した。これは上表と同じnominee範囲である。H2/H6も上表のPR/Dが有限確認標本下で1未満という元報告の比較を引用できる。一方、**最新純RTE対D、最新H6 r32 Hybrid対Dの倍率は、ここで使用した保存正本に対応列がないため新たに算出しない**。低費用の大きさはTable 1の既存Gを併記して追跡可能にし、旧倍率を新候補へ流用しない。追加の数値再解析を本資料の完成条件にはしない。

## 6. Event数とB/N/C/Gの費用内訳

### 6.1 純RTEのevent数

次表は原因調査E7が原報告から転記した既存表示を再利用する。q8/K2、ε_sig=.0005。H4厳精度の確認集合minはs64/K4であり、このK2系列と区別する。

| 分子 | s | cohort / n | B | 保存N_total | C_cos=C_sin | G | G SE |
|---|---|---|---|---|---|---|---|
| H4 | 64 | confirmation / 384 | 1.66209 | 388296014 | 325.3047 | 1.263145e+11 | 4.468464e+08 |
| H4 | 128 | confirmation / 48 | 1.290387 | 233557853 | 631.4167 | 1.474723e+11 | 1.073676e+09 |
| H4 | 256 | exploration / 8 | 1.136081 | 180992303 | 1268.75 | 2.29634e+11 | 3.17433e+09 |
| H6 | 64 | confirmation / 48 | 20.6408 | 60852202283 | 424.1042 | 2.580767e+13 | 3.140839e+11 |
| H6 | 128 | exploration / 8 | 4.692424 | 3094651181 | 793.125 | 2.454445e+12 | 5.557905e+10 |
| H6 | 256 | exploration / 8 | 2.175295 | 663720708 | 1580.625 | 1.049094e+12 | 1.58689e+10 |
| H6 | 512 | confirmation / 48 | 1.475668 | 305364103 | 3114.979 | 9.512028e+11 | 3.558125e+09 |
| H6 | 1024 | exploration / 8 | 1.214851 | 206953734 | 6239.5 | 1.291288e+12 | 4.538681e+09 |

H6ではs64→512のshot節約がC増加を上回り、s1024ではC増加が勝る。H4ではs128からC増加が勝る。H6 s128/256/1024とH4 s256は探索価格であり、隣接全点の独立確認を意味しない。law・angle・order確率・biasもevent数とともに変わるので、Bだけの独立因果効果ではない。[元CSV][eventcsv]、[原因調査§3][E7]。

H2の大event追加は次の保存探索表である（ε=.05、各n8）。Phase Aで確認したs4の改善とは別段階である。

| candidate | s | B | N_cos+N_sin at ε=.05 | C_cos | G at ε=.05 | G SE | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| PR_p0000_q8_r8_K2 | 64 | 1.0251115 | 14736 | 219 | 3227184 | 135287.17 | 8 |
| PR_p0000_q8_r8_K4 | 64 | 1.0251119 | 14736 | 220.625 | 3251130 | 153720.71 | 8 |
| PR_p0000_q8_r16_K2 | 128 | 1.0124801 | 14376 | 420.75 | 6048702 | 201845.35 | 8 |

### 6.2 H6 Hybridのevent拡張

prefix78/q8/K2、ε=.0005。r8は旧確認n48、r16/32/64は追加確認n64である。以下はE7の保存表をそのまま再利用する。

| r (s) | 確認n | B exact | N_cos / N_sin | C_cos = C_sin | G exact | G SE |
|---|---|---|---|---|---|---|
| 8 (64) | 48 | 2961839000617845/1125899906842624 | 755225928 / 1053563646 | 74945/24 | 22593289103905/4 | 6378678743.383964 |
| 16 (128) | 64 | 3664256682156955/2251799813685248 | 288969543 / 403143584 | 233501/64 | 161609107267627/64 | 2700894285.578722 |
| 32 (256) | 64 | 5747355790514877/4503599627370496 | 177727546 / 247950732 | 299249/64 | 63691899506611/32 | 2568645672.459057 |
| 64 (512) | 64 | 1271969024947877/1125899906842624 | 139280852 / 194313258 | 432931/64 | 72211615818205/32 | 2769615401.911481 |

追加後のr32選択は全5精度で探索nomineeと確認後minが一致した。rを増やしてB/Nが下がっても、C増加でGが再上昇する。保持S2・split由来のbiasが残るという説明には整合するが、biasの成分別帰属は未分離である。p684/q1/r16/K4は小Bでも.05以外でbias不適格となった。[E6比較][E6compare]、[全候補][E6]。

### 6.3 代表的なN/C競合とH6のheadroom

次の略号・値はE7§2を再利用する。粗=.05、厳=.0005、R/Hは純RTE/Hybrid。H2は粗精度の比較。Cは当該保存cohortで両軸同値だった値である。candidate ID・nはTable S1とE7へ対応する。

| 略号 | B（保存exact又は保存decimal） | N_cos / N_sin | C_cos = C_sin | G exact | G SE |
|---|---|---|---|---|---|
| H2-R | 1638430308470357/1125899906842624 | 14906 / 14903 | 907/48 | 27036763/48 | 11489.641344756476 |
| H2-H | 6113487734987227/4503599627370496 | 12920 / 12920 | 775/32 | 1251625/2 | 20808.97393554041 |
| H4粗-R | 1.6620900065204434 | 19369 / 19370 | 41639/128 | 1613053221/128 | 44580.37483807345 |
| H4粗-H | 1.3837493166001436 | 14384 / 13570 | 10291/24 | 47945769/4 | 37995.654220550095 |
| H4厳-R | 1.662369839186228 | 193753833 / 193753845 | 124901/384 | 8066682748313/64 | 486679288.01218957 |
| H4厳-H | 1.5871499573638734 | 183683129 / 177738535 | 17797/48 | 134004611546 | 1163065295.2161603 |
| H6粗-R | 1661454695514047/1125899906842624 | 15268 / 15268 | 149519/48 | 570714023/6 | 355807.7258949337 |
| H6粗-H | 5747355790514877/4503599627370496 | 11465 / 11493 | 299249/64 | 3435079271/32 | 138534.1239994281 |
| H6厳-R | 1661454695514047/1125899906842624 | 152683318 / 152680785 | 149519/48 | 45657735316457/48 | 3558125067.408183 |
| H6厳-H | 5747355790514877/4503599627370496 | 177727546 / 247950732 | 299249/64 | 63691899506611/32 | 2568645672.459057 |

H6の最厳精度では、保存bias_cos/sinは純RTEで `1117751031/144115188075855872` / `-1390296821/288230376151711744`、Hybrid r32で `1234339285/17592186044416` / `1998945861/17592186044416` である。対応するuおよび保存headroomはE7にあり、次のh表はその原文である。

| 略号 | h_cos exact | h_sin exact | u_cos/h_cos exact | u_sin/h_sin exact |
|---|---|---|---|---|
| H6厳-R | 6521765120365847/18446744073709551616 | 6521819213501719/18446744073709551616 | 720168576/6521765120365847 | 720168128/6521819213501719 |
| H6厳-H | 5227609803667223/18446744073709551616 | 4425861698631447/18446744073709551616 | 558891008/5227609803667223 | 558891008/4425861698631447 |

純RTEの大きいhと小さいbiasは、Bが大きいにもかかわらずNが小さいという会計結果を説明する。これを「prefixをなくしたことだけが原因」という主張に置き換えることはできない。H2/H4厳では純RTEのNが大きくC節約が勝つこと、H4粗では逆にHybridのN節約が勝つことも同じ会計内で観測されている。

## 7. 説明できること、因果未分離のこと、未保証事項

| 証拠段階 | 本資料で使用できる内容 | 使用できない解釈 |
|---|---|---|
| 観測・技術的確認 | 保存NとCの競合、event数増加時のB/N低下とC増加、H6高精度のbias/headroomとNの対応 | 小Bまたは短い回路だけから小Gを断定しない |
| 整合的だが因果未分離 | prefix保持費用、tail構造、sampling law、finite approximation bias、compiler応答の関与 | 同じq/r/Kでもtail/law/B/bias/circuitが変わる。保持部分だけの効果や一意な寄与率を主張しない |
| 未保証 | 母平均価格・希少event tail、実compiledρ、全パラメータ最適性、別分子/geometry/compiler、実機・FT-QPE/RPE全体 | 有限確認平均・名目SE・局所包含から一般的方式優位やend-to-end保証へ格上げしない |

K2のorder4はsupport外で、K4におけるorder4未観測とは意味が異なる。後者や未観測componentの費用寄与を0としない。名目SE・Student-t区間や2 combined-SEの記述的比較は、選抜後・逐次観察後の同時coverageや母平均tail上界ではない。[H6価格統計][h6prices]、[H2検証][H2verify]、[H4検証][H4verify]、[event検証][extverify]、[H6検証][H6verify]。

既存の回路構造対照は原因調査E7を参照し、本資料へDF-nativeの数値を混ぜない。PauliのD2同一body対照は制御構成＋compiler応答の差を示す補助証拠であり、最新L0対Hybridの因果効果ではない。本資料はPauli-onlyの文書統合である。

## 8. 論文Methods草稿

### 8.1 計算対象と比較方式

線形水素分子H2、H4およびH6について、隣接原子間距離1.00 Å、STO-3G基底、中性singletの保存Pauli–Jordan–Wigner Hamiltonianを用いた。非identity項数はそれぞれ14、184、918であった。各分子内では同じ保存raw state、full-H referenceおよびexact normを使用し、方式別の再正規化は行わなかった。T=0.8の固定時間信号を対象とし、全項を保持するFull PF、prefix外を切り捨てるB0/B0I、部分保持のHybrid PRおよびL_D=0の純RTEを比較した。Pauli実装はconstantを常に保持するため、同じprefix/qのB0とB0Iを独立方式として重複計上しなかった。

### 8.2 回路実装と条件付き資源会計

決定論的部分にはS2を使用し、ランダム化部分には最大偶数event次数Kのfinite paired-Taylor RTEを使用した。各stepのr eventをq step実行し、総event数をs=qrとした。符号付きPauli product、rotation、constant/global/relative phaseおよび演算順序を保持した。回路はcentral-rotation制御とfull Hadamard測定wrapperを用い、cosineおよび+Im sineを評価対象とした。Qiskit1.3.0においてbasis gatesをrz、sx、x、cx、optimization levelを1、seedを17とし、追加のtopology/routing条件は設定しなかった。

費用は状態準備を除くcompiled RZ数を単回価格とし、保存bias・normalization・数値allowanceを用いたshot十分条件と組み合わせてG=N_cos C_cos+N_sin C_sinと定義した。要求信号精度は.05、.01、.005、.001、.0005とし、各軸へ元binary64規則によりε_sig/√2、α_axis=.025を配分した。主結果は仮想compiled残差ρ=0での条件付き会計であり、実compiled精度の保証ではない。数値包含とcompiled残差を区別し、候補固有のB・bias・u・Nを別候補へ転用しなかった。

### 8.3 候補選抜・不確かさ・再現性

候補集合および確認標本数は分子と検証段階ごとに異なる。探索価格と別seedの確認価格を区別し、両者をpoolしなかった。H4のPR対決定論的主表では探索nomineeを確認前に固定した一方、最新Hybrid対純RTE表は確認集合内の観測最小を含む。H2の追加確認には逐次延長を含み、H6 Hybrid追加検証では確認開始前にnを固定した。同一trajectoryから得たcosine/sine価格の共分散を保持した保存G SEを引用し、決定論的単一価格にMC SEを付与しなかった。

本稿の統合表は固定commitの既存選択結果と保存exact値の転記であり、値の再解析や新しい候補探索は実施しなかった。各分子の5精度は同じ価格標本を共有するため、3分子×5精度を15の独立実験とは解釈しなかった。有限標本平均、名目SE、ideal-law数値包含を、希少eventを含む母平均価格、全law compiled残差、選抜後同時信頼保証へ転用しなかった。詳細なSOURCEとcohortはTable S1および固定出典表に記載した。

## 9. 論文Results草稿

### 9.1 Hybrid PRと純RTEの条件付き費用比較

保存された有限候補の確認集合を比較すると、純RTEはH2およびH6の全5精度、H4の4精度でHybrid PRより低いGを示した。H4のε_sig=.05ではHybrid PRが低く、純RTE低Gは計14/15精度条件であった。H6ではHybridの大event候補を追加してもこの比較は維持され、Hybrid/RTEの保存比は約1.13–2.09であった。これらは固定候補、固定compiler、仮想ρ=0および有限価格標本に条件付けた観測であり、母平均winnerや方式一般の優越性を示すものではない。

### 9.2 決定論的対照と比較集合の影響

Full PFとB0/B0Iを含めた既存の決定論的探索では、H2の最良保存対照はFull PF、H4およびH6ではB0/B0Iであった。H4の141 PR構成探索では、事前固定nomineeの独立確認に基づく主表Gが、全5精度で106決定論的構成の最良保存値を下回った。対最良決定論的の保存削減率は約48.42–95.27%であった。この主表のPRにはL_D=0の境界候補が含まれるため、低GをHybrid化自体の効果とは解釈しなかった。

H2の追加確認後の順位変化は、既存s4候補の確認範囲が充実したことによる更新であり、大s探索の改善によるものではなかった。H4では.05/.01の確認集合minが元nomineeと異なったため、両者を区別して報告した。H6では純RTEとHybrid双方のevent数を拡張した履歴を保持し、旧範囲の比較値を最新familyの倍率として流用しなかった。

### 9.3 Event数と費用差の説明

H6純RTEではs64から512へevent数を増やすとBおよび必要shot数が低下し、単回価格の増加を上回ってGが低下した。s1024ではC増加が勝りGが増加した。H4ではs128/256への増加によるshot節約をC増加が上回り、保存純RTEの低費用候補はs64に留まった。H6 Hybridもr32で低いGを示し、r64で再上昇した。これらの最小点は有限grid上の観測であり、連続域の単峰性や大域最適性の証明ではない。

高精度のH6比較では、純RTEはHybridより大きいBを持ちながら、小さいsignal biasと広いshot headroomにより必要Nが小さく、Cも低かった。一方、H2の粗精度およびH4の厳精度では、純RTEの大きいNを低いCが補うという異なる競合が観測された。したがって費用差はNとCを同時に扱うことで説明できるが、決定論的prefixの除去だけを共通原因とすることはできない。tail、law、finite approximation、compilerの個別寄与は未分離であり、分子サイズに関する一般法則は導かなかった。

## 10. 図表・実施範囲・終了

主要図は既存の[H4 G対s][H4g]、[H6 G対s拡大図][H6g]、[H6 B/N/C][H6bnc]を候補とする。H2の[追加確認・event図][H2g]およびB/N/C、全精度の候補・SE表はSupplementaryへ配置可能である。H6の既存G図のHybrid参照線は10/26時点の旧n48値であり、最新r32/n64の値を描いたものではない。最新比較はTable 1とE6保存表で示す。図の配置・caption・採否は[対応表](pauli_h2_h4_h6_claim_figure_map_20261101.md)にまとめた。

独立branch `codex/pauli-h2-h4-h6-paper-evidence-synthesis-20261101` で、新規2文書と最小索引のみを作成した。既存論文本体・過去版・SOURCE/input/raw/result/grant/ledger/marker/STOP・固定branchは変更しない。既存文書専用の `one`、`source_rows`、`check_links` を再利用し、科学runner・会計関数・解析pipelineは呼び出していない。過去の人工テストPASSは過去の証拠であり、今回実施したテスト件数として数えない。今回行うのは転記文字列・固定参照・変更範囲と公開本文/SHAの技術照合で、独立科学レビューではない。

**公開とremote SHA・本文照合後mandatory STOP。次の担当はユーザー/GPT：論文の中心claim、図表構成、詳細科学レビューや追加検証の要否を判断する。Codexの次の実行作業なし。新科学計算、原因追加検証、DF横断統合、既存原稿の無断変更へ自動移行しない。**

[E1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fd1ceb283309a7e515ff4a533a253260bdd0f78/artifacts/pauli_h2_h6_comparison/2026-10-25/H2_H6_best_comparison.csv
[E2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fbfbfd21a252a4a32f52e3b40e50acd91b3c626a/artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/best_by_precision.csv
[E2all]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fbfbfd21a252a4a32f52e3b40e50acd91b3c626a/artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/candidate_precision_table.csv
[E3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/artifacts/non_df_h4_pauli_pr_search/2026-10-25/best_by_precision.csv
[E4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/candidate_precision_table.csv
[E5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/selected_cost_breakdown.csv
[E5b]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/90c1c60820d52ca7ddcd4fb5315e9d21bb4a86c3/artifacts/pauli_hybrid_rte_comparison/2026-10-25/cost_breakdown.csv
[E6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/CANDIDATE_RESULTS.csv
[E6compare]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/COMPARISON_RESULTS.csv
[E7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5432857b94c3714ea9f4d6031e493e8dfc308a91/docs/research/pauli_pure_rte_cost_mechanism_inventory_20261101.md
[E8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/pauli_jw_resource_evaluation_summary.md
[E8csv]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/data/pauli_jw_summary/latest15.csv
[oldintegrated]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/pauli_h2_h4_h6_integrated_comparison_20261025.md
[oldratio]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/artifacts/pauli_h2_h4_h6_integrated_comparison/2026-10-25/H2_H4_H6_optimized_PR_comparison.csv
[oldmethods]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/non_df_h4_paper_methods_DRAFT_20261025.md
[H2report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/docs/h2_pauli_rte_validation_20261026.md
[H4report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/docs/non_df_h4_pauli_pr_search_20261025.md
[D4report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fbfbfd21a252a4a32f52e3b40e50acd91b3c626a/docs/non_df_h4_pauli_b0_comparison_20261025.md
[extreport]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/docs/pauli_rte_event_extension_20261026.md
[H6report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/REPORT.md
[D26report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fd1ceb283309a7e515ff4a533a253260bdd0f78/docs/pauli_h2_h6_resource_comparison_20261025.md
[H2source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/SOURCE.json
[extsource]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/SOURCE_e755ecf581600192fd2745d7066e410cebd933d3.json
[H6source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/SOURCE.json
[directsource]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/90c1c60820d52ca7ddcd4fb5315e9d21bb4a86c3/artifacts/pauli_hybrid_rte_comparison/2026-10-25/SOURCE.json
[nativebundle]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fd1ceb283309a7e515ff4a533a253260bdd0f78/artifacts/pauli_h2_h6_comparison/2026-10-25/SOURCE_history.bundle
[numeric_source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fd1ceb283309a7e515ff4a533a253260bdd0f78/artifacts/pauli_h2_h6_comparison/2026-10-25/numeric_SOURCE.json
[H2verify]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/verification.json
[H4verify]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/artifacts/non_df_h4_pauli_pr_search/2026-10-25/verification.json
[extverify]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/verification.json
[H6verify]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/verification.json
[h6sampling]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/sampling_design.json
[h6prices]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/price_statistics.json
[H2g]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/figures/H2_G_vs_s.png
[H4g]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H4_G_vs_s.png
[H6g]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H6_G_vs_s_extension_zoom.png
[H6bnc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H6_B_N_C_vs_s.png
[eventcsv]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/event_cost_trends.csv
[h2coverage]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/old_candidate_confirmation_map.csv
[h4nomination]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/artifacts/non_df_h4_pauli_pr_search/2026-10-25/confirmation_nominations.json
