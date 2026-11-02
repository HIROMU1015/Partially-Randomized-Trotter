# 研究C：既存成果のEvidence Mapと次の研究判断

2026-10-31 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。
調査起点：a776aa84aefc128cc317e33075415140916013a0。

本書は、ユーザーが依頼した**Codexによる技術的な証拠整理**である。正式共通運用指示2026-10-25版の§8・9・10を実際に確認した。重要なGPT科学レビュー、独立新規性の認定、論文化GO、現在の改善系列の終了・継続の採択は行っていない。数値は既存報告からの引用で、新しい保存数値の再集計・ランキング・統計解析を含まない。

## 1. Executive Summary

研究Cの中心的な肯定的成果は、**保存H4の同じ初期値から、DF物理予算を保持しながらPR tail massを減らし、その保存候補をPF/RTE、条件付きRPE shots、native価格、PR内部資源比較まで接続したこと**である。

| 分けて扱う主張 | 現在の証拠と到達範囲 |
|---|---|
| **Claim A：目的変更によるtail mass低下とDF予算保持** | Stage Aのrank8・γ=0/.1各1 fit。同じ修正codec後x0で、exact tail λは2.2097618367788057→1.4424298855756243 Ha、登録削減率34.72464491113348%。両armのsector norm＋guardはCA/8内。[E01] |
| **Claim B：特定PF/RTE・RPE条件での改善** | s4/r16/K0で全13 roundのattenuationと補正RTE誤差が改善。m12のAは.20959759→.51386660。後続のideal仮定で十分shotsは53,136→19,574。[E02][E04] |
| **Claim C：仮定付き費用モデル内のPR内部改善** | 同じs4/r16比較のideal条件付き総RZは3,194,333,274,946→540,290,160,109、元報告の削減率83.08598%。価格は各32標本の校正平均をproxyとして使用。[E03][E04] |
| **Claim D：強決定論S2への総資源優位** | **既存の主RZ比較では支持されない。** 同じProposed J/UのM2決定論s1は101,825,929,455、M4 PR s1/r128/K0は234,117,788,135。Round別割当も両仮定laneで全13 roundがDを選択した。[E06][E09][E20] |

A〜Cは一つのH4・保存候補対での数値的・条件付き成果である。tail λだけの因果効果、新分子への汎化、母平均価格、厳密CA、独立した方法上・論文上の新規性は確立していない。Dの不成立はA〜Cの観測を消さず、A〜CもDを含意しない。

構造疎化、低rank化、surrogate、round割当には有用な限定的負結果がある。一方、shared-basisの低rank CDFへの帰着は数学的同値、sampling lawのNO_DISTINCT_FREEDOMは調査集合内の方法差未特定であり、同じ種類の負結果ではない。

この資料は、**A：既存成果による論文化検討、B：判断を変える限定検証、C：現在の改善系列を区切る**、の選択材料を提供する。いずれも本書では採択しない。公開照合後はmandatory STOPする。

## 2. 研究目的・検証の流れ・比較条件

### 2.1 問いの変化

出発点は、tensor近似だけを最小化する通常CDFから、PRのtailに置かれた係数massも考慮するCDFへ進めば、減衰に伴うshots負担を下げられるか、という問いだった。検証は次の接続をたどった。

1. **Stage A→B1→B2→C1→M1**：DF精度とtail mass、PF/RTE・A、有限native価格、条件付き費用、その会計上の寄与を分離。
2. **M2→M3→M4**：同じJ/Uの強決定論対照を置き、PR macrostep・r変更で残る差を確認。
3. **Tail-P4→surrogate→第三DF→shared-basis→CDF7**：係数support、basis切替、rankがnative価格・shotsへどう接続するかを確認。
4. **Native構造→round別D/PR→sampling law**：未使用の実装・割当・分布の自由度が残るかを調査。

H2/H4/H6の以前の通常CDF/XDF検証は、この系列を支えるadapter・codec・native・精度評価資産である。異なるrank政策・codec・分解生成法を使っており、今回のH4 γ=.1 PR-aware候補の分子間汎化試験ではない。[旧分子間整理](research_c_c1_h2_h4_h6_integration_update_20261026.md)

### 2.2 条件辞書

以下の条件コードはEvidence MapとSOURCE表で共通に使う。条件差を省略して直接比較するための記号ではない。

| コード | 固定条件 |
|---|---|
| **H4** | 直鎖H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、sector36。8 system qubit/full256、最後のancillaを含むfull512。通常number CDF、一般対称J、元g補正κ、scalar一回、元frame順を保持。 |
| **A8** | H4・rank8の共通保存初期J/Uをreal-Schur codecでパラメータ化し、同じqualified x0からγ0/.1各1 fit。ffsim0.0.83、L-BFGS-B、options=None。tol1e-8はffsim初期分解条件で、SciPy停止tolの上書きではない。 |
| **P4/16** | A8最終2 arm、元順prefix L_D=2、partial ordered symmetric S2、δ=.1、s4、h=.025、r16、K0。 |
| **P1/64・P1/128** | 指定保存J/U、同じL_D=2、δ=h=.1、s1、r64または128、K0。DFを変えた場合は固有のground・law・shots・価格を使用。 |
| **D1** | 指定保存J/Uの全frame ordered symmetric S2、δ=h=.1、s1。PRのL_D分割は使わず、合法な全sweep相対Gaussian融合を適用。 |
| **R13** | m0..12、t_m=.1×2^m、own-DF ground-state oracle、状態準備費用除外。CA=.00159 Ha、DF reserve CA/8=.00019875 Ha。既存RPEの総位相π/6・統計位相π/12、候補別failure .01、軸別.01/26を保持。 |
| **P0** | Qiskit1.3.0、opt1、seed17、basis rz/sx/x/cx、approximation1、qubits_initially_zero=false、MPFR160 declared phase、位相fitなし。PR head内部相対融合、tail厳密basis_key共有、既存のfenced serial会計。 |
| **N32** | 指定lawの各32独立trajectoryを有限bit/PCG64で校正。異なるcohortをpairedと扱わず、同じ32件を13 roundの費用proxyに用いる。新しいDF32個ではない。 |

RPEのfailure配分は各候補の推定問題に対するもので、系列全候補の事後選択を含むfamilywise保証ではない。費用表のshotsは式による条件付き十分数であり、実量子shotsを実行した数ではない。

### 2.3 保存候補のidentity

同じ「CDF8」という名前でも、以下を同一視しない。SHA256は既存報告・manifestの転記であり、配列を新たにdecodeしていない。

| 保存物 | SHA256・由来 |
|---|---|
| 元P-H4 | 8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50 |
| Stage A h/g/c snapshot | 1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db |
| 歴史的CDF8初期export | c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08。後のγ0最終候補とは異なる。 |
| Stage A γ0最終J/U | f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02。第三DF検証もこの候補を使用。 |
| Stage A γ.1最終J/U | f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227。M2 Proposed、M3、M4、round別比較で再利用。 |
| Tail-P4最終J/U | ec0dfe3d53e8209b343b294a103ab89405ac757d3e26164ae591344d4423b985 |
| 通常CDF7最終J/U | 4e4939d1380386ebb98c3b39a13954cbfc3148e35fb8e2cfd3cf536bd0b57ff4。rank7独自Cholesky初期値からの1 fit。 |

## 3. 肯定的な主要成果とその意味

### 3.1 Claim A：PR tailを意識したDF目的と予算保持

Stage Aの既存実装は、ffsimのtensor loss・解析gradient・optimizerを再利用し、目的を次のように変更した。[E01]

F_γ = f/f* + γ(M_ε/M*)²、f = ½‖g−g_DF‖²_F。

M_εは元順L_D=2以後の一般KをI/Z/ZZへ展開した**非identityのZ/ZZ係数**についてΣ√(c²+ε²)、ε=1e-8 Ha。f*とM*は同じ初期値から固定したscaleである。判定対象のexact λ=Σ|c|とsmooth目的を区別し、tail identityは別の相対位相として保持した。λは全Hamiltonianの任意のLCU normと同じ定義ではない。

| 終了値：H4/A8 | γ0 | γ.1 |
|---|---:|---:|
| Exact tail λ [Ha] | 2.2097618367788057 | 1.4424298855756243 |
| Tensor Frobenius | 5.607958698344648e-9 | 6.285301032493658e-9 |
| Sector operator norm [Ha] | 7.018855038843621e-9 | 9.039161828383144e-9 |
| Norm＋engineering guard [Ha] | 1.0297636237465247e-8 | 1.229561498918692e-8 |
| DF予算CA/8 | PASS | PASS |

物理予算を保ってtail massを下げられたことが支持される。一方、γ.1のtensor/sector残差はγ0よりわずかに大きく、物理近似をより正確にしたという結果ではない。両optimizer successは相対目的減少による停止であり、停留点・大域最適の保証ではない。1初期値・各1 fitからγ全体の最適性も主張しない。

### 3.2 Claim B/C：減衰・shots・PR内部資源への接続

P4/16・R13で、m12のPF誤差はγ0/.1で約.0044769543/.0044762458、補正RTE誤差は.1643051229/.1256120553、Aは.20959759/.51386660。PFがほぼ同じ一方、A・補正RTEが改善したという観測である。Aをraw complex signalの絶対値と同一視しない。[E02]

P0/N32 kernel RZ平均は5,198.3125→4,765.875、元報告の8.3188%低下。C1のideal仮定ではshots53,136→19,574、条件付き総RZの削減は83.08598%。各roundで合成・law誤差をそれぞれ.001A以内と**仮定**する別laneでも同方向だったが、その仮定を実測保証へ変更していない。evidence laneはUNRESOLVED_SIGNAL_ERRORでshots/総費用nullのままである。[E03][E04]

M1の対称的会計分解では、総RZ差2,654,043,114,837に対し、価格側157,377,117,755、shots側2,496,665,997,082と報告された。これは保存価格と反復重みの二因子分解であり、λ、U、head、event lawを一つずつ固定した因果ablationではない。[E05]

### 3.3 技術的資産として成立したこと

修正codec、一般K adapterと独立CAR、full512の位相を含むraw/compiled/actual-adjoint/X/Y、fixed seedのnative校正、RPEとserial費用の接続が存在する。人工PASS、有限cohortの全operator数値検証、Arbによる固定回路enclosureは異なる証拠階層である。後者が一部で取得済みだからといって全lawを保証せず、全law未資格だからといって固定回路の保証を消さない。[E03][E09][E23]

## 4. 負結果・NO-GO・同値性の分類

| 区分 | 該当結果 | 結論の射程 |
|---|---|---|
| **肯定的な数値結果** | Stage A/B/C、M3/M4のPR内部改善、CDF7のDF予算PASS | 固定分子・候補・目的・費用仮定内の観測。別分子・全探索への一般化ではない。 |
| **限定的な負結果** | 強Dへの主RZ不利、tail-P4、CDF7 native、surrogate/第三DF、round割当 | 登録主指標・有限cohort・固定候補で不成立。全PR、全疎化、全低rank、全指標の否定ではない。 |
| **数学的同値・既知性** | Shared-basis一般Jの通常低rankへの帰着、Euler/Parent/IS対応 | 表現集合・平均operator等の明示した水準の同値/包含。数値的費用不利とは別の結論。 |
| **技術的な成立確認** | Codec/CAR/native/full512、round switchingの十分条件、人工PoC | 正しさと適用範囲を示す。性能改善・論文GOは別。 |
| **未資格・未確立** | 全law合成/implemented-law誤差、母平均価格、独立DF汎化、厳密CA | UNKNOWN/UNQUALIFIED/nullを保持。有限probe最大値や観測min/maxで埋めない。 |
| **未採択の研究仮説** | Basis-switch目的のη、head/tail境界融合、別DFへの転移等 | 技術設計・代数的可能性。新しい科学実行対象として採択していない。 |

「NO-GO」は表中の範囲で読む。特にnative構造のNATIVE_STRUCTURE_NO_GOは追加実験の優先順位に関する技術提案、sampling lawのNO_DISTINCT_FREEDOMは方法差未特定であり、研究C全体の正式終了判断ではない。Track Bの既存D＋E採択とは権限・対象とも異なる。

## 5. Evidence Map

各行は「問い・比較・固定条件／結果／資格／支持・非支持／次の判断への意味」を対応させる。IDは§11の元報告・科学SOURCE・rawへ接続する。条件辞書§2と資格辞書§6は各行の一部であり、表を条件なしの順位表として使用しない。

### 5.1 初期成果と強決定論対照

| ID・区分 | 問い・方法・固定条件 | 元報告の結果 | 誤差・価格・統計の資格 | 支持／非支持・判断への意味 |
|---|---|---|---|---|
| **[E01] 肯定** | PR tail目的で精度を保てるか。H4/A8、γ0/.1、各1 fit。 | λ34.72464491113348%低下、両DF予算PASS。30 calls完了。 | Sector norm＋guardの数値pilot。Tensor残差と独立判定。相対目的減少停止。 | Aを支持。γ最適・λ単独因果・新規性は非支持。成果の核として保持。 |
| **[E02] 肯定・技術** | 同じ最終2 armのPF/RTEは改善するか。P4/16・R13。 | 因子/総和CAR・ground/alias PASS、26/26 round潜在適格。ProposedのA/RTEが全13 roundで改善。141 calls。 | 数値referenceと登録十分条件。合成/law/RPE実行は別。 | BのPF/RTE部分を支持。CA達成ではない。誤差と減衰を分けて保持。 |
| **[E03] 肯定・技術** | 同P4/16のnative単価は下がるか。P0/N32、各arm32。 | RZ mean5,198.3125→4,765.875。64 full512検証PASS、652 calls。 | 有限bit標本平均。母平均CI・全law supportなし。旧run01の技術STOPは保全。 | この校正での単価低下を支持。平均DF差の機構は未分離。 |
| **[E04] 条件付き肯定** | A/B/価格がPR内部総資源へつながるか。P4/16・R13・P0/N32再利用。 | Ideal shots53,136→19,574、総RZ83.08598%低下。164 calls。 | Ideal/assumedの反実条件付き。Evidenceは全round null。有限価格校正。 | Bのshots、Cを支持。D・保証shots・実CAは非支持。 |
| **[E05] 会計的知見** | 価格とshotsのどちらが保存差へ寄与するか。E04の2×2置換。 | 対称分解のshots側2,496,665,997,082 RZ、価格側157,377,117,755 RZ。20 calls。 | 保存scalarの代数的会計、因果推定なし。 | 減衰負担を考慮する必要を支持。λだけの寄与率と呼ばない。 |
| **[E06] 限定負・技術** | PRに同じJ/Uの強D対照を置く。A8両arm/D1/R13/P0。 | 両kernel RZ11,026。Proposed Dのideal総RZ101,825,929,455、253 calls。 | 全matrix・adjoint/X/YとRPE合成伝播は数値pilot。PRとはs・融合機会・誤差資格も異なる。 | 強Dを省けない。旧PR内部改善からD優位は導けない。PF単独因果ablationでもない。 |
| **[E07] 条件付き肯定・対D負** | Head反復を減らせるか。Proposed P1/64と旧P4/16。 | PF m12は約.004476→.071615へ増加、A同じ。Ideal総RZ約2.95179019e11、旧PR比.546334249、D比2.89885907。471 calls。 | 非paired native32、shots再評価、全law未資格。旧技術STOP保全。 | PR内部macrostep改善を支持。Headだけの因果効果やD勝利は非支持。 |
| **[E08] 肯定・資格限定** | r128で減衰を軽減できるか。Proposed P1/128/R13。 | A_m12=.716844438、補正RTE=.0609453085、ideal shots16,056。13/13潜在適格、98 calls。 | 同sのPFをSOURCE照合して再利用。価格未取得だった段階。 | rによるshots交換を支持。価格込み結論はE09へ。 |
| **[E09] 条件付き肯定・対D負** | E08の実native単価を含める。P1/128/R13/P0/N32。 | RZ mean14,076.4375、ideal総234,117,788,135。M3比.793138、D比2.299196。332 calls、32 full512 PASS。 | 有限校正、ideal/assumedのみ。Evidence null。 | PR内部改善、主RZではD不利を支持。全指標支配ではない。 |
| **[E22] 強baselineの限定比較** | 通常CDF対XDFも強い実装で比較する。歴史的CDF8/XDF prefix8、D1/R13/P0、B2/B3経路。 | 総RZ102,056,444,569対100,267,421,030、元報告ではXDFが1.7529745883%小さい。86 calls。 | 数値CA pilot、DF誤差は別。A8最終J/Uではない。 | 弱い旧XDF10対照からのCDF優位は一般化不可。A8とXDFの新順位は作らない。 |

### 5.2 DF構造・rank・basis

| ID・区分 | 問い・方法・固定条件 | 元報告の結果 | 誤差・価格・統計の資格 | 支持／非支持・判断への意味 |
|---|---|---|---|---|
| **[E10] 肯定・後続未判定** | Tail JをP4 supportへ制限できるか。H4/rank8/LD2、head-full、tail対角＋辺01/12/23、γ.1。 | 1 fit、DF予算PASS、λ1.516165932377233 Ha。係数位置216→144。21 calls。 | Projected初期値が変わるため同x0 ablationではない。Tensor/sector別測定。 | 構造制約下の近似成立を支持。疎化だけで安くなるとはしない。 |
| **[E11] 限定負** | E10をP1/128/R13/P0/N32へ接続。 | A_m12=.692254786、ideal shots16,568、RZ mean15,047.46875、総275,422,425,481.5。M4比1.176427。449 calls。 | 全13 round・固定32 correctness PASS、点比較のみ。 | このP4候補の主RZ改善なし。λ、補正RTE、A、価格を単一指標に置換しない。全疎DF否定ではない。 |
| **[E12] 限定負** | 保存M4/P4各32で価格proxyを検証。固定4-fold OLS。A=switches、B=overlap、C=3特徴、cohort平均対照。 | RZ MAEはM4 A221.4073<C227.7980、P4 B261.5380<C273.5792。148 calls、SURROGATE_NOT_QUALIFIED。 | 静的DFは2個。Head等はcohort interceptへ吸収。RMSEの別結果で主MAE判定を変更しない。 | Cの登録条件不成立。DF間機構・η・新DF汎化の証拠ではない。 |
| **[E13] 未採択仮説** | Switch期待値をDF目的へ組み込めるか。厳密basis_key群のmass比q_b。 | IIDで(r−1)(1−Σq_b²)。固定partitionではJのmass配分を扱い、自由Uの幾何へ直接勾配を与えない。 | 技術設計・人工検証のみ。非凸性、零tail、smooth biasを整理。 | ηや新目的を採択していない。E12のCを復活させる根拠でもない。 |
| **[E14] 限定負・技術** | 別DFへswitch proxyが移送できるか。第三DF=A8 γ0、P1/128/R13/P0/N32。旧2DFで学習後固定。 | Third RZ MAE A369.78698、B303.32856、平均2892.23438。SWITCH_TRANSFER_NOT_SUPPORTED。493 calls。 | 静的3DF中、独立holdoutは1DF。32trajectory≠32DF。再fitなし。 | Aは平均より良いがBを上回らず、登録移送判定不成立。新η採用不可。 |
| **[E15] 数学的同値** | Shared-basisは独立した分解classか。一般対称J、厳密同一UをG群で共有。 | 群内ΣJへ集約でき、表現集合は通常CDF outer rank≤G。CDF8のtail2 frame共有はCDF≤7に帰着。 | 数学条件付き同値と人工61 PASS。rank-one J、別support制約、近似U共有へ拡張しない。 | 重複fitterの必要性は支持されない。低rankの物理/資源性能は別に検証する必要があった。 |
| **[E16] 肯定** | 通常CDF7がDF予算を満たすか。H4/rank7独自Cholesky初期値、γ0、1 fit。 | 初期norm＋guard9.0647035e-4→終了3.022641747295844e-7 Ha、予算PASS。25 calls。 | Tensor Frobenius4.6164582354636235e-7とsector残差を別記。相対目的減少停止、最適rank証明なし。 | 一つのrank7近似の成立を支持。初期予算未達は許容済み診断。総資源改善は含意しない。 |
| **[E17] 技術・潜在適格** | CDF7をP1/128/R13へ接続。LD2、tail F2..6。 | 13/13潜在適格、A_m12=.460928623、ideal shots21,768。119 calls。 | 独立CAR/ground/alias PASS、evidence未資格。後の価格はE18で取得。 | PR接続は成立。rank削減によるshots減少は示していない。 |
| **[E18] 限定負・技術** | CDF7のnative単価/総RZは下がるか。E17＋P0/N32、保存shots再利用。 | Mean RZ17,754.9375、ideal総638,880,821,593.5。γ0 CDF8比1.0188545、M4比2.7288863。332 calls、32 full512 PASS。 | 小差の母平均有意性なし。独自fit/law/初期値の差があり、rank単独効果ではない。 | 当該候補の主RZ点改善なし。一般的な低rankの無効性とはしない。 |

### 5.3 実装構造・round・lawと保証資産

| ID・区分 | 問い・方法・固定条件 | 元報告の結果 | 誤差・価格・統計の資格 | 支持／非支持・判断への意味 |
|---|---|---|---|---|
| **[E19] 技術・優先順位提案** | Nativeの未利用構造は何か。Head相対融合、tail厳密key共有、境界/中央Z/ZZ融合。人工小系のみ。 | Head/tail基本共有は実装済み。Tail相対融合は人工complete P0でRZ増、中央融合は局所削減例があってもcomplete例で不変。68人工PASS。 | 実H4新build/compile/再解析0。Head/tail境界案のcomplete性能未取得。 | NATIVE_STRUCTURE_NO_GOは追加実験優先度の提案。H4で全融合が無効との証明ではない。 |
| **[E20] 数学成立＋限定負** | 同一DF energyへD/PRをround別選択できるか。Proposed D1対P1/128、R13、固定位相/failure配分。 | 共通raw-signal/aliasとβ<π/3のnearest-branch条件。保存189-callでideal/assumedとも13/13全D、MIXED_ROUND_DOMINATED。 | 設計117人工PASS、科学run189完了。主RZ固定点モデル、evidenceはUNQUALIFIED。 | Switchingは合法だが、この点モデルで混合利益なし。全指標/母平均/全候補の支配ではない。 |
| **[E21] 既知性・方法差未特定** | Euler/Parent/IS/return/相関に未利用自由度があるか。Cの固定K0平均operatorとTrack B一次資料対応。 | Common-parent内L1最小性等を整理しNO_DISTINCT_FREEDOM。新9＋既存4＝13人工PASS。 | 限定技術調査、H4性能評価0。Affine辞書依存、全path重み等の適用条件を維持。 | 独立した実装可能改善機構は未特定。全sampling lawの不存在・新規性否定・研究終了の正式採択ではない。 |
| **[E23] 固定対象でのvalidated保証** | Baseline/Proposed各1保存trajectoryを独立CARへ接続。P4/16の一h-step、full512全基底、Arb192、κH/right-polar Q target。 | F2-K1/F3-K1各7 calls。位相込みraw/target、compiled/targetのvalidated operator2上界が固定allowance内。 | Literal κ/Uからのmodel bridgeも別計上。各1固定trajectory・特定targetの保証。 | 厳密enclosure資産は存在する。r128全law・全13round・元physical input全誤差・CAの保証へ転用不可。 |

## 6. 強決定論との比較と誤差資格

### 6.1 同じProposed J/Uでの比較

以下は元報告が実施した比較の再掲であり、新しい統一条件への再計算や順位付けではない。全行H4/A8 γ.1最終J/U・R13・P0・状態準備除外。PRはL_D=2、K0、N32価格proxyを用いる。決定論は全frameで、適用可能な全sweep融合を利用する。

| 元検証 | PF・RTE設定 | Kernel RZ | Ideal十分shots | Ideal条件付き総RZ |
|---|---|---:|---:|---:|
| M2 [E06] | D1 | 11,026 | 13,610 | 101,825,929,455 |
| C1 [E04] | PR s4/r16、4 kernel/δ | 4,765.875/一h-step | 19,574 | 540,290,160,109 |
| M3 [E07] | PR s1/r64 | 9,105.59375 | 20,734 | 約2.95179019e11 |
| M4 [E08][E09] | PR s1/r128 | 14,076.4375 | 16,056 | 234,117,788,135 |

M3ではDよりkernel単価が低くても、shotsと高mの反復重みを含む総RZが高い。M4ではr増加でshotsは減るがkernel単価が上がる。これが「小さいλ」「少ないrank」「安い一kernel」のどれか一つから総資源を決められない理由を示す既存の具体例である。新たな因果分解は行っていない。

主RZ以外は別に読む。M4のideal条件付きCX depthは56,962,137,414、M2は57,959,299,320と報告され、PRが小さい指標もある。[E09] したがって全資源でDが支配するとは主張しない。逆にCX depth一点から主RZや総実行時間の優位にも昇格させない。

Round別選択は同じ候補・固定位相/失敗配分の下、RZ点モデルで各roundの選択を行った。全Dと選択後のideal総RZは同じ101,825,929,455。Assumed laneでも全Dだった。[E20] 新しいDF、可変精度配分、別native政策まで最適化した結果ではなく、同じ価格標本で選択しているため独立な選択後統計保証もない。

### 6.2 CDF/XDF baselineと分子間比較

歴史的CDF8/XDF10の約27%削減、同rank8の旧routeでの約12%削減、さらにspin対応/相対融合を揃えた強baselineのXDF側1.7529745883%低下は、**異なるrank・実装・精度政策の結果**である。[旧統合](research_c_c1_h2_h4_h6_integration_update_20261026.md)[E22]

最後の比較では歴史的CDF8/XDF8のDF norm＋guardはそれぞれ7.666215916301914e-5／9.533901455091894e-5 Haで、ともに予算内でも誤差は等しくない。これは共通目標を満たす比較であり、誤差完全一致比較ではない。Stage A後のγ.1最終J/UとXDFを新しく比較した証拠としては使わない。

H2/H6の結果も保存し、通常CDFとXDFの資産として参照する。ただしH6のcodec修復、rank14政策、sector400、state-action/全sector基底のcoverageはH4のfull512/N32と同一でない。分子数に対する単調傾向やPR-aware目的の一般化をここから作らない。

### 6.3 資格辞書：どこまで分かり、何が未資格か

| 量・証拠 | 確認済み範囲 | そのまま置き換えられないもの |
|---|---|---|
| Tensor Frobenius・optimizer objective | g再構成の残差、目的/gradient/停止情報 | Sector operator norm、energy誤差、DF予算。目的改善とsector改善は同義でない。 |
| DF sector norm＋engineering guard | 指定sectorでの登録数値予算PASS | 全Fock保証、入力積分の厳密性、丸めenclosure、最終CA全体。 |
| PF operator誤差 | 指定DF・時間・S2に対する数値reference残差 | 物理HamiltonianのDF誤差や統計誤差。 |
| 補正RTE operator誤差・A | 指定ideal finite-RTE IID lawの平均operatorと補正・減衰 | 各sampleの回路誤差、implemented finite-bit law誤差、raw signal誤差そのもの。 |
| Raw complex signal・位相予算 | 合成/law/PF/RTEをraw信号へ戻す既存十分条件とalias/branch条件 | 次元・単位の違う誤差の無根拠な単純加算。Aや補正誤差だけでCA判定。 |
| RPE条件付き十分shots | Idealまたはassumed budgetが成立すれば使えるHoeffding算定値 | 実施済みshots、普遍的必要最小shots、情報理論的下界、全候補選択を含む保証。 |
| Full512 native数値検証 | 校正に用いた各固定回路の位相込み全matrix、adjoint、X/Y等 | 全sampling lawの一様上界、厳密丸め保証。人工gate試験だけの結果とも区別する。 |
| F2/F3-K1 validated CAR | 各固定P4/16 trajectory・κH/polar targetで全基底enclosureとmodel bridge | 全r128 law、全roundのCA、元literal targetへ橋なしで戻した保証。 |
| N32 native平均・SE | 特定有限bit cohortの記述的価格校正 | Ideal-law母平均の保証値、分布support、独立DF数、事後候補選択補正。 |
| Conditional total resources | 登録serial/fenced会計、n=s·2^m、毎occurrence独立trajectoryを前提とした費用proxy | 同一sample反復による費用縮約、未観測fusion、hardware実時間、状態準備込み費用。 |
| 厳密CA・全law保証 | 上記からの接続設計と一部固定回路保証は存在 | 現PR系列のevidence laneに全要件が揃ったわけではない。UNKNOWN/UNQUALIFIED/nullを維持。 |

総費用の記述的校正SEは、同じ価格cohortを全roundへ使うため、既存報告どおりW×SEで伝播する。13roundで独立に価格を再観測したように√13で縮めない。N32の標本最大/最小も厳密supportにしない。今回これらのSE・shots・総費用を再計算していない。

## 7. 既存研究との対応と方法差の未判断部分

### 7.1 RCDF・PR-aware DF

RCDFはtensor残差に係数のnorm型正則化を加え、λやmeasurement/qubitization資源との交換を扱っている。L1/L2型も既に検討されており、**「CDFへnorm罰則を足した」だけを独立新規性にはできない**。本系列との確認可能な実装差は、元順L_D=2の非identity tail Z/ZZ mass、初期値で固定した正規化、sector予算、PR/RPE/nativeへの接続である。目的全体の数学的完全同値・非同値や論文上の新規性は未認定。[RCDF v3 §B・式17・付録](https://arxiv.org/html/2212.07957v3)

PR論文はorbital basisによるλ最小化と、重みを少数項へ集中させる最適化の可能性を論じ、DFと回転basisのZ/ZZを使うPRも扱う。したがってλ最小化、tail重みを小さくする発想、DFとPRの接続そのものを新発見とはしない。本系列の実装・固定H4検証と、論文の大系/Toffoli評価とは対象・費用・精度条件が違い、H4でDが安いことは原論文全体の反証ではない。[PR v2 §VII A/B・付録C3](https://arxiv.org/pdf/2503.05647v2)

既存の[初期設計](research_c_pr_aware_df_optimization_design_20261026.md)にはSCDF等との対応も記録済みである。本書ではその記録を再利用し、SCDFの新しい完全同値性監査や網羅的な先行研究レビューは実施していない。

### 7.2 Sampling lawとTrack B

以下は[E21]と、固定Track B四資料の対応の再掲である。原本取得commit・SHA256・GitHub readbackは[既存取得索引](../../artifacts/research_c_sampling_law_freedom/2026-10-30/technical/reference_index.json)を参照する。

| 自由度に見える変更 | 既存対応と必要条件 | 今回の結論の限界 |
|---|---|---|
| Euler pairing | Identityとinvolutionを回転へ束ねる既知の平均operator恒等式。controlledでは位相も保持。 | 回転一個の恒等式だけでsampling・実装費用全体が一致するとはしない。 |
| Parent allocation | 同じparentをchildへ配分し、正規化Bと角度/価格を交換。現在K0のcommon配分はこのfamilyのL1最小。 | Family外の辞書や相殺によるB削減まで不可能とはしない。 |
| Cost-aware IS | 固定分解でqを変え、P/qなどのweight・moment/range・bias・十分shotsを戻す既知の枠組み。 | 安いeventを多く選ぶだけでは推定量を保存しない。q∝係数を離れる自由度が全くないともいわない。 |
| 有限Taylor return・辞書変更 | 同じfinite targetを別group/dictionaryで実現し、normalizer、回転、情報取得費用を交換。 | K0→別finite targetはlawだけの変更でない。自由なnormalizer/取得oracleを仮定しない。 |
| Normalization/native価格 | B低下と一eventの安さは別。ParentでL1最小でも総費用最小とは限らない。 | 理想係数だけの改善をweighted shot総資源へ移送しない。 |
| Interstep correlation | 各marginalの一致だけではoperator積の平均を保存しない。全pathの演算子条件または適切なpath重みが必要。 | 既知full-path ISで記述できる場合と、同じweight/rangeのまま独立性を外す案を区別。 |
| 固定辞書の確率変更 | 線形/affine依存があれば平均を保つq変更の例外がある。 | 常に「qは一意」としない。実DF辞書で新機構を同定したわけではない。 |

[Track B終了記録][TB1]は、**D：限定的成果を保持、E：現在のParent allocation数値探索を区切る**という2026-10-15の正式採択を記録する。[資源構造][TB2]は価格、normalization、情報アクセス、guaranteeのscopeを分離し、[次期判断資料][TB3]は未採択の候補群を整理した。[先行研究対応][TB4]はEuler/SCU/PTSC/IS/SPS等の同値・包含・条件差を明示する。

これらの終了判断・旧STOPを変更していない。研究CのNO_DISTINCT_FREEDOMは、今回調べたEuler/IS/Parent/return/相関等の範囲で、既存構成と区別される実装可能な改善機構を特定できなかったという技術判断である。全samplerの不存在証明、新規性全般の否定、研究C全体の終了採択ではない。

## 8. 論文化へ利用できる主張候補と限界

以下は主張を組み立てる際の候補であり、論文テーマ・投稿先・採択可能性の判断ではない。**数値改善の存在、既知手法との数学的方法差、独立した論文上の新規性**は三つの別問題である。

| ストーリー候補 | 中心となる限定主張・根拠 | 限界・重複可能性 | 主張前に必要な確認／追加計算の必要性 |
|---|---|---|---|
| **PR-aware DFの限定改善** | 同x0からtail目的を加え、DF予算保持→A/十分shots→PR内部条件付き費用まで接続。[E01][E02][E03][E04][E05] | RCDF/PRのnorm・表現最適化と重なる。H4一条件・非因果・ideal/assumed価格。 | まず目的/λ/既知実装の対応と主張範囲を整理。既存事実の技術報告には新計算不要。汎化や独立method claimを要求するなら不足を別判断。 |
| **PF/PR資源の比較研究** | 同J/Uの強DとPRのerror/shots/native交換を分離し、PR内部改善と強Dへの不利を併記。[E06][E07][E08][E09][E20][E22] | s/r/融合/誤差資格の差がある。全PFの優劣・漸近主張は不可。主RZ以外で逆方向もある。 | 比較条件と会計を明示すれば既存範囲の記述は可能。一般的競争力を主張するための範囲拡張は未許可。 |
| **負結果と設計上の知見** | Tail λ・support/rank・switch proxyは単独で総費用を決めず、共有basisは通常低rankに帰着する。[E10]〜[E21] | 探索結果は固定候補/単一heldout DF、数学的同値は条件付き。既知IS等の再説明自体は新規性でない。 | 負結果の適用範囲と技術的同値を精査。事実整理には新計算不要。反復可能性・外的妥当性を論文の中心に置くかで追加検証の必要性が変わる。 |

現時点で書けるのは、これらの**条件を明示した研究記録・主張候補**である。独立新規性・重要性・論文化GOをここで認定していない。重要科学レビューを実施する場合は、ユーザーによるそのレビュー自体の明示許可が必要であり、計算許可とは別である。

## 9. 不足証拠と「追加検証が変える判断」

現在のA〜Cという限定主張と負結果の保存・説明には、新しい科学計算を足す必要はない。追加検証が必要かは、次に要求する主張によって変わる。γ/rank/r/Kやlawの再探索を既定の次工程にしない。

| 要求する判断 | 現在不足する証拠 | 最小限の確認の性質と、変わる判断 | 現在の状態 |
|---|---|---|---|
| 既知正則化との差を論文上どう扱うか | RCDF等との目的・変数・λ定義・利用taskの十分な対応評価 | まず一次資料と式/実装の対応確認。方法差を主張するか、既知構成のtask特化評価として扱うかが変わる。 | 新計算より先に資料上の判断が可能。重要レビューの実施は未許可。 |
| H4特定fixture以上へ一般化できるか | 独立した物理入力または初期値での、同じ事前指定pipelineの検証 | 主張対象を先に一つ選び、独立条件で予算・A・価格を同じ対照と接続する必要性を判断。Fixture依存か再現する傾向かが変わる。 | 条件・回数・新契約は未採択。追加fit/計算は未許可。 |
| 価格差が母平均でも残るか | Independent prospective native cohortと事前指定の統計判断。特にCDF7の小差や候補選択の扱い。 | 固定候補の点比較を、誤差付き価格判断へ進められるかが変わる。標本数を今ここで選ばない。 | N32のSEは保証付き差に未昇格。追加sampling/compileは未許可。 |
| λ改善がどの機構で資源改善に効くか | 同時に変わったJ/U、head、event分布のmatchedな対照 | 因果機構を中心主張にする場合だけ、分離可能な対照を定義する必要がある。M1会計説明から因果説明へ進めるかが変わる。 | 新アルゴリズム/実験の設計・採択は行わない。既存相関を因果にしない。 |
| 厳密CA/全law資源を主張できるか | 全law synthesis・implemented-law・物理referenceからRPEまでの適用可能な保証 | 固定回路enclosureをlaw全体へ適用できる根拠が必要。有限32追加だけで解消しない。条件付きpilotから保証付き結論へ進めるかが変わる。 | 固定F2/F3資産を保持。全lawは未資格。厳密保証の再開は未採択。 |
| 改善系列を区切るか | 未実行候補の性能を網羅的に否定する証拠はない | 未否定の可能性を残したまま、現在の主張の範囲・研究目的・投入価値をユーザー/研究Cが判断する。終了判断に全候補の反証は要求しない。 | 技術的に資料は揃った。正式終了は未採択。 |

以上は必要性の対応表であり、実行契約、科学call枠、追加実験の採択ではない。

## 10. 選択肢A／B／CとQ1〜Q10への回答

### 10.1 次の研究判断

| 選択肢 | 判断を支える証拠 | 残る条件・未否定の可能性 | 担当と次の行動 |
|---|---|---|---|
| **A：既存成果で論文化を検討** | A〜Cの接続、強D対照、再現可能なSOURCE/raw、限定負結果のまとまり。 | 既知法との重複、重要性、主張範囲、一般化、保証水準の採否。現状で論文GOとはしない。 | **ユーザー/研究C**が主張候補と、必要なら重要レビューの開始を選ぶ。Codexは選ばれた範囲の資料補完を担当。 |
| **B：判断を変える限定検証** | §9で、何を調べればどの判断が変わるかを分離。 | 追加探索だけでは現在の新規性・汎化・全law問題を同時に解消できない。 | **ユーザー/研究C**が判断対象を一つ選ぶ。Codexがその後に最小条件を準備し、科学実行は別途明示許可後。 |
| **C：現在の改善系列を区切る** | 強Dへの主RZ不利、P4/CDF7/proxy/roundの限定負結果、lawの方法差未特定。 | 別分子・目的・辞書・hardware価格・未検証境界融合まで否定していない。Track B D＋Eを研究Cへ自動移植もしない。 | **ユーザー/研究C**が現在の系列を区切るかを決める。Codexは指示後に成果と未解決事項を引き継ぎ、新探索を自動開始しない。 |

### 10.2 必須質問への回答

| 質問 | 回答 |
|---|---|
| **Q1 最重要の肯定的成果** | 同じ初期値のPR-aware CDFでDF予算を保ってtail λを34.7246%下げ、特定条件のA・shots・PR内部条件付き資源改善まで接続したA〜C。[E01][E02][E03][E04][E05] |
| **Q2 固定条件と証拠** | H4/1Å/STO-3G、rank8、sector36、元順LD2、γ0/.1各1 fit、P4/16・R13・P0・N32、own-DF oracle/準備除外。§2のidentity、§11の科学SOURCE・rawで特定。 |
| **Q3 既存研究との差** | Tail限定係数定義、共通x0/scale、sector予算、PR/RPE/nativeの一体評価という実装・検証条件を特定できる。Norm正則化・λ最適化・DF-PR接続は既知。数学的方法差全体と独立新規性は未判断。 |
| **Q4 主要な負結果と範囲** | 当該強DへのRZ不利、当該P4/CDF7の点改善なし、C surrogate/単一第三DFへの移送判定不成立、固定RZの全D選択。Shared-basisは条件付き数学的帰着、lawは方法差未特定。一般的な不可能性ではない。[E06]〜[E22] |
| **Q5 強Dへの競争力/不足** | Macrostep/r変更でPR内部RZを減らしたが、同Proposed J/UのM4はDの2.299196倍。M3のkernel単価やM4のCX depthなど局所/副指標の利点はあり、全指標支配はない。主RZ競争力は不足。[E06][E07][E08][E09][E20] |
| **Q6 未資格** | 全law synthesis/implemented-law誤差、母平均価格/support、選択後統計、分子/DF汎化、実CA、状態準備込み/hardware費用。固定回路validated保証と全law未資格を両立して記録する。§6。 |
| **Q7 論文ストーリー** | 限定PR-aware改善、PF/PR費用交換、負結果/同値性の三候補を証拠付きで整理可能。既知性・重要性・主張範囲の判断が不足。論文GOや新規性は未認定。§8。 |
| **Q8 追加実験の必要性** | 既存限定成果の説明には不要。汎化、母平均、因果、厳密CAを主張対象に選ぶ場合に、それぞれ§9の不足証拠を埋める検証が判断を変える。無目的な探索は提案しない。 |
| **Q9 区切る場合の保全** | 修正codec/ffsim接続、一般K/独立CAR、native/P0/位相、PF/RTE/RPE/serial会計、人工・回帰資産、fit生結果とx/jac、全SOURCE/input/J/U、QPY/cohort、正負の報告・旧STOP/契約/grant/ledger/marker・Git履歴。§11へのリンクを残す。 |
| **Q10 次の決定** | ユーザー/研究CがA/B/Cと、必要な主張・保証水準を選ぶ。重要レビューと科学実行は別の許可。Codexは今回の公開照合後STOPし、採択・計算・レビューへ自動移行しない。 |

## 11. SOURCE・原本リンクと今回の照合範囲

### 11.1 科学SOURCE登録

以下のSHAは、結果が後で追記された報告commitではなく、各実行の固定科学SOURCEを表す。元報告とraw directoryのSOURCE/STOPを併読する。技術設計だけの行は「技術公開commit」と明記した。全報告・rawは今回の起点a776aa84aのtreeにも保存されている。

| ID・元報告 | 固定科学SOURCE／技術公開commit | 原本 |
|---|---|---|
| [E01] Stage A | [e1085c95f5c01b68566e17d543ace5832cb94061][S01] | [run01](../../artifacts/research_c_pr_aware_df/2026-10-27/run01/) |
| [E02] B1 | [7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9][S02] | [B1-run01](../../artifacts/research_c_pr_aware_df_stage_b/2026-10-27/B1-run01/) |
| [E03] B2修正実行 | [a1cf0046967c206f414dbad568a75884ee076ebf][S03] | [run02](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/run02/) |
| [E04] C1 | [b1c735da394d4b0a444cbc618ead9781ef4d9134][S04] | [C1-run01](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1-run01/) |
| [E05] M1 | [47e0d406226be365951bd8666d5acae61106f500][S05] | [M1-run01](../../artifacts/research_c_pr_aware_df_mechanism/2026-10-28/M1-run01/) |
| [E06] M2 | [a6ffc23d94f6488a4769596f089a91c895d49214][S06] | [run01](../../artifacts/research_c_pr_aware_df_M2/2026-10-28/run01/) |
| [E07] M3修正実行 | [5354edecc6aa995082440dd14be0310bdd9902e1][S07] | [run02](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/run02/) |
| [E08] M4 PF/RTE | [7dbabc7866cc62a5460bf1579ce84c59493d8e1e][S08] | [run01](../../artifacts/research_c_pr_aware_df_M4/2026-10-28/run01/) |
| [E09] M4 native | [dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad][S09] | [run01](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/run01/) |
| [E10] P4 fit | [59e186ad5132ffde66684a781a47a2bd11a77ed4][S10] | [run01](../../artifacts/research_c_df_structural_support/2026-10-28/run01/) |
| [E11] P4 PR | [4ac0e5f96a5ad5009f6273d0fc6ff5657ff80aff][S11] | [run01](../../artifacts/research_c_df_structural_pr/2026-10-28/run01/) |
| [E12] Surrogate修正実行 | [0301286493ebf2b95a1786c2bd78756015c1015d][S12] | [run02](../../artifacts/research_c_df_basis_surrogate/2026-10-29/run02/) |
| [E13] Switch目的設計 | 技術公開 [14270239ad2f448feeedb4751e4174a3b571a430][S13] | [technical](../../artifacts/research_c_df_basis_switch/2026-10-29/technical/) |
| [E14] 第三DF修正実行 | [30436b142d7fa79c841b5977d1b4c0fbfbd1bb4c][S14] | [run02](../../artifacts/research_c_df_basis_third/2026-10-29/run02/) |
| [E15] Shared-basis | 技術公開 [0876449d7ce75e25277865834efbc3bafaed7b3e][S15] | [technical](../../artifacts/research_c_shared_basis_df/2026-10-29/technical/) |
| [E16] CDF7 fit | [c13d6f82a22d497b3334eaaa40d8430a5010f70f][S16] | [run01](../../artifacts/research_c_cdf7_onefit/2026-10-30/run01/) |
| [E17] CDF7 PR | [3e75b16901f0f4a9966d2bd5b9ff6bde8e75f472][S17] | [run01](../../artifacts/research_c_cdf7_PR/2026-10-30/run01/) |
| [E18] CDF7 native | [173ba2fd83172f41f08900fcba7e2a977565f57f][S18] | [run01](../../artifacts/research_c_cdf7_native32/2026-10-30/run01/) |
| [E19] Native構造設計 | 技術公開 [0cb2e5ea986320283465ea51e5b273d0b3313b96][S19] | [technical](../../artifacts/research_c_pr_native_structure/2026-10-30/technical/) |
| [E20] Round設計・実行 | [79fa0107658c92c3b3169d62348498c7dc37fab0][S20] | [設計](research_c_roundwise_det_pr_allocation_design_20261030.md)／[run01](../../artifacts/research_c_roundwise_det_pr/2026-10-30/run01/) |
| [E21] Law調査 | 技術公開 [a776aa84aefc128cc317e33075415140916013a0][S21] | [technical・固定参考資料](../../artifacts/research_c_sampling_law_freedom/2026-10-30/technical/) |
| [E22] 歴史的強baseline | [1aeedd96178a3f50a15bc2701c3c332f4f391b64][S22] | [run01](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/run01/) |
| [E23] F2/F3-K1 | [3742ec10ddc0df5faaaa2a6856a9cc60ea013eb9][S23a]／[9b0f4d2b40159b77f28caf3f215ad22d0d075879][S23b] | [F2原本](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-K1-run01/)／[F3報告](research_c_pr_aware_df_F3_K1_execution_20261028.md)／[F3原本](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-28/F3-K1-run01/) |

F2/F3のrun SOURCE manifestのbase_commitは実装基点であり、上表の科学実行commitとは別である。上表は元実行報告・postflightの実行SOURCEを使用し、この来歴の区別を保った。

### 11.2 保全と今回の作業

今回再利用したのは既存報告、SOURCE/STOP metadata、保存summaryへの参照、Track B固定参考資料・取得索引、既存コードの静的仕様である。Stage A目的のc_pr_aware_df、一般K/CAR、native/P0、RPE・serial会計の定義は元報告と照合し、新しい分析pipelineは作成していない。

| 保持する実装資産 | 参照先・用途 |
|---|---|
| Fit目的・初期化 | [PR-aware目的](../../src/trottertracks/c_pr_aware_df.py)、[修正実直交codec](../../src/trottertracks/c1_real_orthogonal_codec.py)、[ffsim接続](../../src/trottertracks/c1_ffsim_corrected_codec.py) |
| 表現・独立reference | [一般K](../../src/trottertracks/c1_general_k_adapter.py)、[SectorCAR等の分子拡張](../../src/trottertracks/c1_molecule_extension.py)、[PF/RTE接続](../../src/trottertracks/c_pr_aware_df_stage_b.py) |
| Native・精度・費用 | [NativeBuilder](../../src/trottertracks/c1_stage2_native.py)、[RPE精度](../../src/trottertracks/c1_stage3_accuracy.py)、[serial会計](../../src/trottertracks/c1_h6_ca_resources.py)、[Stage C](../../src/trottertracks/c_pr_aware_df_stage_c.py) |
| Native比較・保証・割当 | [M2](../../src/trottertracks/c_pr_aware_df_m2.py)、[M4 native](../../src/trottertracks/c_pr_aware_df_m4_native.py)、[K1 reference](../../src/trottertracks/c_pr_aware_df_fixed_k1.py)、[round別接続](../../src/trottertracks/c_roundwise_det_pr.py) |

これらは現在のtreeへの案内であり、過去の再現には§11.1の各科学SOURCEの版とmanifestを使う。コードの再利用は過去の実行許可・結果資格の移転ではない。

Stage A、C1、M2、M4 native、law調査の5報告について、起点commitのGitHub公開本文とローカル本文の一致を確認した。20実行directoryのSOURCE/STOP metadataを読み取り、科学実行と技術設計のcommitを区別した。さらにStage A/C1/M2/M4 native/round/F2-K1/F3-K1の7 SOURCE manifest計539 entryについて、各固定科学commitのGit blobと保存SHA256の一致を確認した。主要転記値も元報告の文字列と照合した。リンクとGit blobの照合は技術確認であり、旧科学runnerや数値計算の再実行ではない。

新科学call・保存値再解析・配列decode・fit・sampling・build/compile・shots再算定はすべて0。新規実験契約・承認証明・grant・分析scriptは作成していない。変更は本書と最小限の索引/状態説明/研究ノート/既存manifestに限定する。コード・既存実行契約・SOURCE manifest・raw・旧報告・grant・ledger・marker・STOPは変更しない。テストを再実行して新しい科学的再現証拠と称してもいない。

Git差分・リンク・SOURCE/既存原本の不変を確認し、commit/push後にremote SHA/treeと公開文書を照合する。照合後は**mandatory STOP**。次の正式判断はユーザー/研究Cが行う。

[E01]: research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md
[E02]: research_c_pr_aware_df_stage_b1_execution_20261027.md
[E03]: research_c_pr_aware_df_stage_b2_execution_20261027.md
[E04]: research_c_pr_aware_df_stage_c1_execution_20261027.md
[E05]: research_c_pr_aware_df_mechanism_execution_20261028.md
[E06]: research_c_pr_aware_df_M2_same_JU_deterministic_execution_20261028.md
[E07]: research_c_pr_aware_df_M3_s1_r64_repair_execution_20261028.md
[E08]: research_c_pr_aware_df_M4_s1_r128_feasibility_breakeven_20261028.md
[E09]: research_c_pr_aware_df_M4_native128_execution_20261028.md
[E10]: research_c_df_structural_support_execution_20261028.md
[E11]: research_c_df_structural_pr_performance_20261028.md
[E12]: research_c_df_basis_surrogate_repair_execution_20261029.md
[E13]: research_c_df_basis_switch_optimization_design_20261029.md
[E14]: research_c_df_basis_third_repair_execution_20261029.md
[E15]: research_c_shared_basis_df_design_20261029.md
[E16]: research_c_cdf7_onefit_execution_20261030.md
[E17]: research_c_cdf7_PR_execution_20261030.md
[E18]: research_c_cdf7_native32_execution_20261030.md
[E19]: research_c_pr_native_structure_optimization_design_20261030.md
[E20]: research_c_roundwise_det_pr_execution_20261030.md
[E21]: research_c_sampling_law_freedom_survey_20261030.md
[E22]: research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md
[E23]: research_c_pr_aware_df_F2_K1_execution_20261027.md
[S01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e1085c95f5c01b68566e17d543ace5832cb94061
[S02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9
[S03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a1cf0046967c206f414dbad568a75884ee076ebf
[S04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/b1c735da394d4b0a444cbc618ead9781ef4d9134
[S05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/47e0d406226be365951bd8666d5acae61106f500
[S06]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a6ffc23d94f6488a4769596f089a91c895d49214
[S07]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/5354edecc6aa995082440dd14be0310bdd9902e1
[S08]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/7dbabc7866cc62a5460bf1579ce84c59493d8e1e
[S09]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad
[S10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/59e186ad5132ffde66684a781a47a2bd11a77ed4
[S11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/4ac0e5f96a5ad5009f6273d0fc6ff5657ff80aff
[S12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/0301286493ebf2b95a1786c2bd78756015c1015d
[S13]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/14270239ad2f448feeedb4751e4174a3b571a430
[S14]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/30436b142d7fa79c841b5977d1b4c0fbfbd1bb4c
[S15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/0876449d7ce75e25277865834efbc3bafaed7b3e
[S16]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/c13d6f82a22d497b3334eaaa40d8430a5010f70f
[S17]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3e75b16901f0f4a9966d2bd5b9ff6bde8e75f472
[S18]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/173ba2fd83172f41f08900fcba7e2a977565f57f
[S19]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/0cb2e5ea986320283465ea51e5b273d0b3313b96
[S20]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/79fa0107658c92c3b3169d62348498c7dc37fab0
[S21]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a776aa84aefc128cc317e33075415140916013a0
[S22]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/1aeedd96178a3f50a15bc2701c3c332f4f391b64
[S23a]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3742ec10ddc0df5faaaa2a6856a9cc60ea013eb9
[S23b]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/9b0f4d2b40159b77f28caf3f215ad22d0d075879
[TB1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[TB2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_resource_structure_and_bottlenecks_20261015.md
[TB3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_next_algorithm_research_decision_packet_20261015.md
[TB4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/parent_allocation_prior_art_correspondence_20261014.md
