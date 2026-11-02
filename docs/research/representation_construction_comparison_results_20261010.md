# Hamiltonian表現の限定構成・同一信号精度比較と次のGPTレビュー

2026-10-10 JST。独立系列の第2 batch。sourceは
`ce99b57166a9f422f151f56fa17ed799cd9909dd`、基点は
`39345830ddfe7c3e2a488c284a0623f489764087`、branchは
`representation-construction-comparison-20261010`。

利用者提供の[独立レビュー](representation_construction_inputs/representation_exploration_independent_review_2026-10-10.md)
第9節に従い、Aの入力からの構成と実finite-RTE、Cの対称PF対照、Bのisometry接続を一つの小さいbatchで閉じた。
**Aに資源の利害分岐、Cに強い既知対照での利益消失、Bに入力接続と直接実装への費用劣位が得られたため、科学計算を停止する。**
新規性・中心仮説・次段の採択を代行しない。
`LIMITED_CONSTRUCTION_COMPARISON_COMPLETE_AWAITING_GPT_REVIEW`、mandatory STOP、next-stage=false、central-hypothesis=null。

## 1. 読む資料と証拠の位置

| 用途 | 正本 |
|---|---|
| 認可された作業と比較上の訂正 | [利用者提供review](representation_construction_inputs/representation_exploration_independent_review_2026-10-10.md) |
| 入力、全候補、資源上限、STOP | [実行前固定scope](representation_construction_comparison_scope.md) |
| 固定科学source | [construction_comparison.py](../../src/trottertracks/representation_exploration/construction_comparison.py)をsource commitで読む |
| 起動とsource/input fingerprint | [runner](../../scripts/run_representation_construction_comparison.py)、[run audit](../../artifacts/representation_construction_comparison/2026-10-10/run1/run_audit.json) |
| 全候補、mean、bias、sampled events、全費用 | [一次result](../../artifacts/representation_construction_comparison/2026-10-10/run1/result.json) |
| 実際のnative gate列とabsolute phase | [全374 native IR](../../artifacts/representation_construction_comparison/2026-10-10/run1/native_ir.json) |
| 同一精度会計の全行表示 | [136行CSV](../../artifacts/representation_construction_comparison/2026-10-10/precision_resource_rows.csv)（保存JSON由来、別実験ではない） |
| 別実装による保存値監査 | [verifier](../../scripts/verify_representation_construction_comparison.py)、[PASS receipt](../../artifacts/representation_construction_comparison/2026-10-10/saved_evidence_audit.json) |
| テスト、失敗、依存版、入力取得 | [provenance](../../artifacts/representation_construction_comparison/2026-10-10/provenance.json)、[tests v3](../../artifacts/representation_construction_comparison/2026-10-10/tests_v3.log)、[artifact README](../../artifacts/representation_construction_comparison/2026-10-10/README.md) |
| 旧結果・root保護 | [protected state audit](../../artifacts/representation_construction_comparison/2026-10-10/protected_state_audit.json) |
| 取得後の公開blob照合 | [remote retrieval receipt](../../artifacts/representation_construction_comparison/2026-10-10/remote_retrieval_receipt.json)（公開後の別取得を子commitに保存） |

これは固定小系のlocal development evidence。commit済みblobは追跡可能だが、immutable CI、外部の科学再現、分子・held-out検証ではない。
前回の[初期結果](representation_exploration_initial_validation_20261010.md)、failure、source、raw artifactsは保持した。
既存Track A/Bのsource・契約・結果・STOP、元rootのdirty差分は変更していない。
共有READMEへの追加は索引であり、共有science変更ではない。

## 2. 共通task・会計と保証の範囲

固定時間の複素Hadamard信号を、X/Y各軸で推定する。system preparation、Hadamard ancilla、
controlled evolution、basis変換、identityの相対位相、読み出しbasisを含む。
Z測定1件/axisはmetadataに記録し、unitary IR/count外とする。
RZ/CX/sx/x/size/depthとqubit数を別指標にし、任意の重み付き合計や論理T countに置き換えない。
compilerはQiskit 1.3.0、basis rz/sx/x/cx、optimization level1、固定seed20261010、process1。
反復boundaryの隣接同一D half-blockを結合し、Gaussian basisを無制御、中心の対角作用を制御する扱いを新旧へ揃えた。

paired finite-RTEの内部sampling law・return semanticsは既存実装のまま。
K=2のcorrected短時間平均はtail Rについて
\(I-i\delta R-\delta^2R^2/2+i\delta^3R^3/6\)、identity phaseは厳密に別適用。
\(\tau=\lambda_R\delta\)の短時間normalizationは
\(B(\tau)=\sqrt{1+\tau^2}+\tau^2\sqrt{1+(\tau/3)^2}/2\)、q反復のrange補正は\(\Gamma=B^q\)。
量子sample平均はcorrected mean/Γであり、meanの小ささとrange補正の負担を混同しない。

複素信号epsilon=.05/.02、各軸epsilon/sqrt(2)、X/Y合計failure=.05。
\(b=\|A_{\mathrm{corr}}-e^{-iTH}\|_2\)を両軸共通のbias診断として用い、
適格なら\(N_{\mathrm{axis}}=\lceil 2\Gamma^2(\epsilon/\sqrt2-b)^{-2}\log(4/.05)\rceil\)。
費用は\(N_{\mathrm{axis}}(\overline C_X+\overline C_Y)\)。
ここでbは小系binary64密行列によるoperator診断で、解析的上界・interval算術証明ではない。
従って以下のshotsはその診断に条件付けた会計であり、保証付き大系アルゴリズムではない。
量子shots実行0、energy/QPE/RPEの最終予算評価0。

乱択costは各row8 classical trajectories、X/Yは同trajectoryを共有する。
SEは各pairの費用和から計算し、独立軸の二乗和で代用しない。
8標本で稀な高Taylor-order eventの平均費用を十分認証できるとは限らず、point±SEはformal CIやwinner認定ではない。
deterministic-only回路は1件/axisでSE=0。

## 3. A：入力から構成するcoreと正確残差

### 3.1 入力と算法

3 fermionic modes、Jordan–Wigner full Fock（8次元）、固定exact synthetic DF rank2、
\(H=d\Gamma(g_0)^2+d\Gamma(g_1)^2\)、独立one-body/constant correction0。
geometry・化学basis・分子rank policyは該当なし。

```
g0 = [[ .8, .09, 0 ], [ .09, -.35, 0 ], [ 0, 0, .15 ]]
g1 = [[-.2, 0, 0 ], [ 0, .6, .07 ], [ 0, .07, -.45 ]]
```

0–1、1–2 hoppingを持つconnected入力で、one-particle sectorは3次元。
前回の共通occupation条件付き二準位簡約だけで閉じる入力から一段進めた。
検証stateはmode0占有（X0準備を含む）、T=.4、q=1/2/4、delta=.4/.2/.1。

入力からframe VをI、g0固有frame、g1固有frameの3件に生成する。
各frameで\(h_l=V^\dagger g_lV\)、\(d_l=\operatorname{diag}(h_l)\)とし、
\(D=\sum_l d\Gamma(d_l)^2\)、\(R=\sum_l d\Gamma(h_l)^2-D\)を返す。
Rはanalytic JW Pauli係数の積で正確に取得し、角度gridやdense Hの固有解を候補生成に使わない。
全factorのdiagonal coreを1 blockとし、outer Γ(V)は反復全体の外に一度置く。
native-prefixのL_Dとは異なるため、frame候補にnative L_D値を付けない。

比較対象は元factorのweight-ranked native DF prefix L_D=0/1/2。
deterministic blockのGaussian変換と中心作用を構造的に実装し、全乱択・全決定論も同じ精度会計に含めた。
tailはr=1/step、K=2。frame候補の評価にdense corrected mean/biasを使うので、
この有限候補生成器と「大系で安く最良候補を選べるselector」は区別する。
小系batch全体の時間は記録したが、constructor単体のscaling benchmarkは行っていない。

固定frameの線形diagonal projection Pに対し、実直交factor-label混合Oは
\(\sum_a d\Gamma(P\sum_lO_{al}g_l)^2=\sum_l d\Gamma(Pg_l)^2\)。
Oの直交性で交差項が消えるので、factor同士の可換性を仮定しない。
この全factor coreにはlabel-angle探索を加えても自由度がないため除いた。
subset、別frame、別supportの構成まで不変という主張ではない。

### 3.2 全候補のtailと同一精度費用

| constructor | native L_D | tail lambda | involutions |
|---|---:|---:|---:|
| native_df_ld0 | 0 | 0.900412171 | 12 |
| native_df_ld1 | 1 | 0.274861527 | 6 |
| native_df_ld2 | 2 | 0.000000000 | 0 |
| frame_identity | shared core | 0.091500000 | 10 |
| frame_factor_0 | shared core | 0.054474183 | 15 |
| frame_factor_1 | shared core | 0.123210565 | 15 |

epsilon_complex=0.05：

| constructor | q | N/axis | G_RZ ± paired SE | G_CX ± paired SE | bias |
|---|---:|---:|---:|---:|---:|
| native_df_ld0 | 1 | 9,285 | 603,525 ± 88,364.437 | 194,985 ± 30,392.297 | 0.0006921342 |
| native_df_ld1 | 1 | 7,236 | 1,506,897 ± 6,939.18 | 864,702 ± 3,618 | 0.0001315107 |
| native_df_ld2 | 1 | 7,063 | 1,617,427 ± 0 | 1,045,324 ± 0 | 0.0001285939 |
| frame_identity | 1 | 7,151 | 514,872 ± 5,405.648 | 582,806.5 ± 5,233.996 | 0.00029863 |
| frame_factor_0 | 1 | 7,075 | 891,450 ± 5,348.197 | 689,812.5 ± 5,178.37 | 0.0001426603 |
| frame_factor_1 | 1 | 7,175 | 1,015,262.5 ± 5,251.562 | 703,150 ± 5,423.79 | 0.0003195306 |

epsilon_complex=0.02：

| constructor | q | N/axis | G_RZ ± paired SE | G_CX ± paired SE | bias |
|---|---:|---:|---:|---:|---:|
| native_df_ld0 | 1 | 61,665 | 4,008,225 ± 586,859.778 | 1,294,965 ± 201,846.093 | 0.0006921342 |
| native_df_ld1 | 1 | 45,732 | 9,523,689 ± 43,856.079 | 5,464,974 ± 22,866 | 0.0001315107 |
| native_df_ld2 | 1 | 44,629 | 10,220,041 ± 0 | 6,605,092 ± 0 | 0.0001285939 |
| frame_identity | 1 | 45,854 | 3,301,488 ± 34,662.366 | 3,737,101 ± 33,561.691 | 0.00029863 |
| frame_factor_0 | 1 | 44,761 | 5,639,886 ± 33,836.136 | 4,364,197.5 ± 32,761.697 | 0.0001426603 |
| frame_factor_1 | 1 | 46,093 | 6,522,159.5 ± 33,736.622 | 4,517,114 ± 34,843.033 | 0.0003195306 |

表は上記3-mode rank2入力、native L_D=0/1/2または1 shared core、
delta=.4/.2/.1の全q比較から、候補ごとの最小RZ pointを表示する。全18行と除外理由は一次JSONに残る。
全候補とも今回のRZ最小はq1、delta=.4。表の±はpaired SEでありformal CIではない。

identity-frame coreはnative L_D1/2より低いRZ pointで、元factorの構造化実装を入れても差は残った。
しかしL_D0の全乱択と比べると、RZ差は小さくそのSEが大きい一方、CXが増える。
epsilon=.05でframe_identity RZ514,872、CX582,806.5、native L_D0 RZ603,525、CX194,985。
lowest-lambdaのfactor0 frameがlowest-costではなく、frame取得とbasis・dictionary費用が効く。
この一入力から全乱択に対する優位、一般的PR資源改善、新規factorization算法の成立を認定しない。
「known native prefixに対する局所RZ差」と「全endpointを含む多資源判断」を分けてGPTへ戻す。

## 4. C：generated mixed accessと強い既知対照

### 4.1 入力・oracle・適用条件

diagonal Ising Aと\(\alpha\sum_iX_i\)の2入力。
old_twoはfields=(1,.7)、edge01=.3、T=.2。
chain_threeはfields=(.9,.6,-.4)、edges01=.2/12=-.15、T=.4。
alpha=.1/.2、q=1/2/4/8、deltaはそれぞれ(.2,.1,.05,.025)、(.4,.2,.1,.05)。
geometry・化学basis・DF rank・L_Dは該当なし、basisはcomputational Pauli、stateは全plus（system H準備を含む）。

mixed \(e^{-it(A+\alpha X_i)}\)はgraphからneighborsを取得し、各neighbor bitで
\(h_i+\sum_jJ_{ij}z_j\)を計算したconditional SU(2)回転と、可換なrest Aを返す。
degree≤2、枝数≤4のconstructor。Hadamard ancillaには中心rotationだけを制御し、条件付きbasis変換はそのancillaに無制御。
通常一次PF、境界結合した対称S2、既知THRIFT積を同じcompiler・task・精度で比較する。
これはTHRIFT compositionの改良ではなく、graphに制限したaccessの実装確認である。
[THRIFT一次資料](https://arxiv.org/html/2403.08729v3)ではmixed propagator accessの費用が適用判断に関わる。
対称性を使う可解Hamiltonian拡張は既知であり、このconditional回転だけを新規性にしない。
[Patel–Yen–Izmaylov](https://arxiv.org/html/2305.18251v2)。

### 4.2 同一精度の全比較

| input | alpha | epsilon | ordinary: q / G_RZ / G_CX | S2: q / G_RZ / G_CX | THRIFT: q / G_RZ / G_CX |
|---|---:|---:|---:|---:|---:|
| old_two | 0.1 | 0.05 | 1 / 342,333 / 265,032 | 1 / 363,426 / 285,040 | 1 / 1,024,628 / 729,872 |
| old_two | 0.1 | 0.02 | 2 / 4,638,639 / 3,773,808 | 1 / 2,327,946 / 1,825,840 | 1 / 6,412,320 / 4,567,680 |
| old_two | 0.2 | 0.05 | 1 / 616,559 / 477,336 | 1 / 370,617 / 290,680 | 1 / 1,027,402 / 731,848 |
| old_two | 0.2 | 0.02 | 4 / 9,036,700 / 7,543,680 | 1 / 2,446,980 / 1,919,200 | 1 / 6,455,974 / 4,598,776 |
| chain_three | 0.1 | 0.05 | 4 / 2,071,254 / 1,872,320 | 1 / 608,839 / 537,676 | 1 / 5,090,400 / 4,072,320 |
| chain_three | 0.1 | 0.02 | 8 / 30,033,240 / 27,225,600 | 1 / 4,623,619 / 4,083,196 | 1 / 32,214,960 / 25,771,968 |
| chain_three | 0.2 | 0.05 | 4 / 4,123,392 / 3,727,360 | 1 / 718,179 / 634,236 | 1 / 5,220,000 / 4,176,000 |
| chain_three | 0.2 | 0.02 | 8 / 81,390,151 / 73,781,440 | 2 / 6,291,516 / 5,615,568 | 1 / 34,335,360 / 27,468,288 |

表は上記各入力・alpha・epsilonについてq1/2/4/8内のRZ point最小、G_RZ/G_CXを表示する。
全48行はJSONとCSVに残し、biasがaxis budget以上のprecision rowはshots/work=nullとして保存した。
deterministic回路のcountにMC SEは付かない。
chain_three/alpha=.2/epsilon=.02のordinary q8はgrid境界なので一般最適値とは扱わない。

今回の8 contextで対称S2はTHRIFTよりRZ/CXの両方が安い。
old_two/alpha=.1/epsilon=.05ではordinary q1の方がS2 q1よりRZ/CXが少ないため、S2を全対照の一律winnerにはしない。
THRIFTの小さいbiasはconditional oracleの費用を取り戻せなかった。
前回toyの低時間次数の消失を一般的優位にしないよう対称S2を加えると、現入力の資源優位を支持する証拠は残らなかった。

n=2..8のchain/starはdegree、neighbors、\(2^d\)枝数のmetadataだけを保存した。
chainはdegree≤2、starはn≥4で中心degree>2となり今回のexplicit multiplexingから除外される。
大系の回路・行列・時間scalingは測定していない。これはconstructorの上限で、別の算術oracleの不可能性ではない。
bounded-degree classへの生成法は明示できたが、既知法との差分や資源利益は未確定。

## 5. B：実isometryへの構造接続と小系費用

physical3 fermionic modes＋aux1 mode、JW full Fock、geometry・化学basis・DF rank該当なし、L_D=0。
4×4 unitary completion Vをorbital Givens (01,π/8)、(23,π/6)、(12,π/10)で構成し、u=Vの上3rows、uu†=I。
enlarged diagonal density interactionsは01=.7、12=.4、23=.25、03=-.2。
\(\widetilde H=\Gamma(V)D\Gamma(V)^\dagger\)、aux vacuum Pに対して\(H_{\mathrm{phys}}=P\widetilde HP\)。
uから独立にnormal-ordered quarticを作って比較し、encoding残差は5.55×10^-16。
[Luo–Cirac isometric THCの式(4)–(7)](https://arxiv.org/html/2407.04432v2)に対応する構造fixtureである。
対象分子から圧縮factorをfittingした結果でも、実圧縮利益の実証でもない。

S=Z_auxとし、\(\overline H=(\widetilde H+S\widetilde HS)/2\)を反射したnative diagonal involution辞書でRTE実装した。
[P,Hbar]=0は全有限多項式の物理block保存に十分。
一般のtime-window/積全体で第一momentが保存されるための必要条件と主張しない、というreviewの訂正を採用する。
個々のsample trajectory、平均channel、coherent stateの保護へ読み替えない。

baselineはHphysを直接Pauli展開するfinite-RTE。同じ第一moment複素信号task、
physical modes0/1占有＋aux vacuum（X0/X1準備）、T=.2、r=1/step、K=2。
q1/2/4、delta=.2/.1/.05のmean/biasを全保存し、epsilon=.05に最初に適格なqでだけ8 cost draws/axisを行う規則を事前固定した。
実costは両構成ともq1。Bには全qのcost最適化を行っていない。
direct Pauli dictionaryは小系dense referenceからの取得であり、一般大系の効率的oracleではない。

| candidate | epsilon | N/axis | G_RZ ± paired SE | G_CX ± paired SE |
|---|---:|---:|---:|---:|
| enlarged_generator_reflection | 0.05 | 7,544 | 711,022 ± 18,792.522 | 226,320 ± 8,064.875 |
| enlarged_generator_reflection | 0.02 | 47,197 | 4,448,317.25 ± 117,570.343 | 1,415,910 ± 50,455.715 |
| direct_projected_pauli_rte | 0.05 | 7,643 | 145,217 ± 26,944.769 | 57,322.5 ± 9,020.22 |
| direct_projected_pauli_rte | 0.02 | 47,830 | 908,770 ± 168,620.739 | 358,725 ± 56,448.664 |

表は上記isometry fixture、L_D0、q1、delta=.2に限定。±はpaired SE。
reflected側はlambda=.9625、Gamma=1.036925817、bias=1.20269×10^-5。
direct側はlambda=1.045353063、Gamma=1.043529784、bias=1.64126×10^-5。
反射側のshots減少はbasis/reflection費用を補わず、直接小系実装よりRZ/CXが高い。
反射側はsystem4＋Hadamard ancillaで5 qubits、直接側はsystem3＋ancillaで4 qubits。

本batchで入力encodingとprimitive接続は成立した。一方、分子compression fitting、
圧縮の規模利益、first-moment taskに揃えたreset/echo費用、効率的direct oracle、asymptotic利益は未取得。
既知reset法のchannel/state taskのcostを第一momentの比較へそのまま移さない。
これらが未接続のまま同型toyを再列挙しても、新規圧縮アルゴリズムの支持にはならない。

## 6. テスト・実資源・別実装監査

source固定前に新31、旧探索23、既存finite-RTE16の計70 local tests passed、fail/skip0、18 warnings。
16件はQiskit multi-control synthesisのPendingDeprecationWarning、2件は旧探索testのqiskit-nature ComplexWarning。
新Gaussian入力はcomplex dtypeを使い、全実回路はabsolute operator検査を通した。
pre-freeze v1の2 failure（normalize引数名の不一致）はlogを保存し、修正後v2/v3も保存した。
科学run1は固定sourceで初回成功、retry0。旧系列の別run1 failure/run2成功を今回のattempt数へ混ぜない。

science batchはA18/C48/B2 candidate（B mean診断6）を完了し、
A246＋C96＋B32＝374 actual wrappersをcompileした。
wall16.9223秒、CPU16.9367秒、peak RSS561,500,160 bytes（535.5 MiB）。
CPU1、BLAS/OpenMP/MKL/Numba各thread1。
上限はwall900秒、CPU600秒、address space4 GiB、file32 MiB、compile512件、
native10000 gates/circuit、total5 qubits。最大5856 gates/circuit、保存gate総数119,883。
このtimeは各構成の独立取得費用やlarge-n scalingではなく、diagnostic・sampling・compileを含むbatch全体。
分子load・GPU・ground-state solve・量子shots0。

保存値verifierはQiskit・project moduleをimportせず、
Gaussian Fock作用をexterior-power minor、native rz/sx/x/cxを行更新から構成した。
176 source/input blobをcommitからhash照合し、全374 IRのhash/count/depth、
両ancilla枝を含むabsolute operator、保存イベント順序、全finite mean/bias/Γとshots/paired費用を照合した。
最大absolute residual6.08023×10^-12、mean residual5.61403×10^-16、bias差1.06130×10^-16。
未署名位相改変、再hashした位相改変、shot改変の3ケースを拒否した。
保存監査wall1.8846秒、新sampling・compile・科学条件0。
これは独立実装によるlocalチェックで、外部科学再現や厳密数値証明ではない。

利用者ZIPのSymPy algebra scriptは別tmpで実行し、同梱rational JSONと意味論一致を確認した。
ZIP原本と4展開fileのhash、provided manifestを照合し、元JSONを上書きしていない。
有理数toyチェックであり、旧binary64科学runの同bytes再実行ではない。

## 7. GPTに戻す判断事項と停止位置

1. **A**：common-frame全factor coreの局所RZ差を、既知構造化DF・all-R/all-D対照と多資源で評価すると、
   深める価値のある新しい構成自由度が残るか。lambda最小と費用最小の不一致をどの設計情報で解決するか。
   今回はexact-data aided評価であり、oracle-free selector・近接factorizationとの差分・classical scalingは未確定。
2. **C**：この2入力では対称PF対照でTHRIFT利益が消えた。新しいmixed-access構成を定義しない限り、
   既知THRIFT適用として優先度を下げる判断が妥当か。graph degreeのmetadataだけを効率性証明にしない。
3. **B**：isometry接続だけで圧縮利益を表せない現状態を保留とするか。
   次に必要なのは真のcompressed input/oracleと同taskの保護baselineであり、toy数の増加ではない。
4. **研究全体**：A/Cを再構成する、特定class/taskに絞る、別構成を探す、のどれを採るか。
   本batchの結果だけで新規算法成立や方向全体の不可能性を結論しない。

重要な構成・資源結果が得られたため、ここからは保存値監査・文書・GitHub保存だけを行った。
次の科学条件、分子benchmark、候補再最適化、Track A/Bへの統合を自動実行しない。
次段はこの固定証拠を読むGPT科学レビューと、その後の利用者の指示に戻す。
