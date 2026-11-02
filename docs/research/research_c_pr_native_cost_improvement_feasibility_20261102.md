# 研究C：PR-aware DFのnative価格改善機構・統合技術判断

2026-11-02 JST。**`TECHNICAL_NO_GO_CURRENT_EVIDENCE`。現在確認できる証拠から、新しいcost-aware DF fitを準備する候補は選ばない。** λ以外の自由度と安価な構造指標は存在するが、それを操作するとcomplete P0 kernel価格が下がる、という接続が未確立である。登録surrogateの不適格や過去のnative負結果を解消する新しい機構は特定できなかった。

これは今回の技術的な実装・実験準備の判断であり、PR一般の不可能性、独立新規性、論文化、研究C全体の継続／終了の決定ではない。新目的・η・scale・solver・compilerを採択せず、新runner・実行契約を追加しない。新しい実分子計算、NPZ/QPYの数値decode・再解析、fit、PF/RTE、sampling、shots再算定、build/compileは0。許可された保存scalarの代数的照合だけを§3に示す。公開照合後mandatory STOP。

## 1. 固定refと統合の範囲

| 資料 | 固定commit・役割 | 今回の扱い |
|---|---|---|
| [A：同じ新J/Uの強決定論比較][A] | `38b8d0df39b8943178ecef611194819cff1c2169`、科学SOURCEは`5f27539ae0ab88181d9cfcd45badb54daf855c28` | 本worktreeの起点。156-callの既報値・資格を引用 |
| [B：PR資源寄与分析][B] | `2639221a59713d431bc7c157854628bbc006a1bd` | 別branchのGit objectと公開本文を照合。A未収録という歴史的記述を保全し、本書で接続 |
| [統合実行][I] | `aced1647b81e8d28091058cdc7b6e3e1ee54896a`、科学SOURCEは`980b514c81eb415fe053f558d7d3c0b8cfb5b9b4` | 新selected、DF予算、PRの保存PF/RTE・価格・資源の出所 |

既存の[成果統合・証拠資格の整理][E]は歴史的な入口として参照し、その後のA/Bで更新された範囲を区別する。BをAのcommitに収録された文書として引用しない。B公開本文と固定Git blobのSHA256は`7c277a7402f7e4101caf92ae5b82815704ab38ebb09a7ad4060e789ebb5e7ad6`で一致した。以下の相対リンクの過去資料は本起点Aのtree内で確認したもの。過去報告の科学SOURCEと、報告を収録するcommitは区別する。

対象は直鎖H4・隣接1.00 Å・STO-3G、通常number CDF8、Nα=Nβ=2/sector36、一般signed対称J・実直交U。PRは元順LD2・partial ordered S2/s1/r128/K0、Dは全9因子ordered S2/s1。共通δ=h=.1、m0..12、CA=.00159 Ha、DF reserve=CA/8=.00019875 Ha、βtotal=π/6、βstat=π/12、failure=.01/axis=.01/26。Own-DF ground-state oracle、状態準備費用除外、元g補正κ・核scalar一回・spin/frame順0..7を維持する。

P0はQiskit1.3.0/opt1/seed17、rz/sx/x/cx、approximation degree1、qubits_initially_zero=false、MPFR160 declared phase、位相fitなし。PRのhead-relative/tail exact-key共有とDの全sweep relative融合は既存範囲のまま。費用はfenced serial kernel反復＋interrogationごとのancilla head/readout shellであり、全長回路の追加最適化を仮定しない。

共通selectedは統合`update005`、export SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`。保存h/g/scalar snapshotは`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、元P-H4来歴は`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。Aの配列同一性PASSを引用し、今回再decodeしない。旧PRはStage A γ=.1最終export `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227`。γ0第三DFやRC2終点とは別候補である。

## 2. AとBを接続して分かる課題（Q1・Q2）

以下はA/B/統合報告および対応JSONの既存scalarの引用。13roundや32trajectoryの再集計、Bの寄与分解の再実行は行わない。

| 項目 | 旧PR | 新PR | 説明できること・限界 |
|---|---:|---:|---|
| Exact tail λ [Ha] | 1.4424298855756243 | .5683498546862645 | 約60.60%減。Smooth目的のmassとは別 |
| m12 attenuation | .716844437765 | .949629430088 | 固定K0のnormalizationと整合 |
| m12 PF operator誤差 | .071614997914 | .094546804594 | PF側は悪化。λだけでPFは制御できない |
| m12補正RTE operator誤差 | .060945308518 | .012063601599 | RTE側は改善。Aとは別の量 |
| Ideal十分shots/X+Y | 16056 | 14048 | 約12.5%減。実量子shotsではない |
| Kernel使用重みW | 16631888 | 10487392 | 約36.94%減。shotsとは異なる反復加重 |
| Kernel標本平均RZ | 14076.4375 | 19825.625 | 約40.84%増。各32の非paired cohort |
| Shell込み条件付き総RZ | 234117788135 | 207919150188 | 約11.19%減。母平均や実CA保証ではない |

新DFの保存sector誤差5.23337351802e-9 Ha＋guard3.22739270291e-9 Haは予算PASS。一方、終端U勾配₂=.6589358143は一次条件NOT_METで、最適解・同時収束を認定しない。γ=.1、ε=1e-8 Ha、元のf*/M*を変更していない。[I]・[B]

K0では`A_m=(1+(λh/r)^2)^(-r·2^m/2)`なので、h/r/m固定でλ低下がA増加へつながる。PF非可換積、raw信号誤差、shots、native価格の単調改善はこの式からは導けない。Bは全13roundで新PF増加・補正RTE減少・A増加を報告している。特にm12 raw signal errorは増えた一方、半径下界ρが増えてshotsが減った。すべての誤差が改善したとの説明はしない。[Bの§3][B]

**Aによって、旧J/Uの強Dとの比較に残っていたcandidate不一致は除かれた。** 新Dはshots13728、kernel RZ11026、総RZ105532782860。新PR idealとの主RZ比は1.9701854206178373。両者をideal nativeにそろえたAの反実laneも同じ整数費用である。D numerical lane/PR ideal laneの資格差だけでこの点差を説明することはできない。ただしDの丸めenclosureとPR全law保証は得られていない。[A]

副指標の総CX depthはD60069240312、PR43549878473、PR/D=.724994660275。全指標でDが支配するとは扱わない。Aでは旧/新Dのkernel6指標は一致し、D総RZは約3.64%増だった。従って、今回のJ/U更新が全決定論にも共通するnative削減を生んだとは観測されていない。[Aの§4–5][A]

現在の直接課題は、**W改善を保ちながらcomplete kernel単価を下げられる操作の同定**である。Jのatom確率、UのGaussian recipe、head、中央Z/ZZ、相対境界、P0の相互作用はSOURCE上の価格要因だが、新旧二つのJ/Uと非paired価格cohortだけでは、どれが40.84%増の主因か未特定。新selectedのq_basis、switch頻度、head単独・basis-pair単独価格を今回推定していない。旧cohort相関から「新Uが高価になった」「switchが増えた」と断定しない。

## 3. 固定W・shellでの損益分岐（許可されたscalar代数）

既存会計は

\[
C_{total}=W\bar C_{kernel}+S_{shell},\qquad
W=\sum_{m,b}N_{m,b}2^m\quad(s=1).
\]

B §6の保存scalarを使う。新PR idealのW=10487392、μ=19825.625、総RZ=207919150188。BにあるS=49168は、総RZ−Wμの**会計残差として確認した値**であり、独立に測定された新fieldではない。今回も同じ差が49168になることをDecimal演算で照合した。Ancilla shellはinterrogationごとに一回、決定論half-headはkernel内。r128や校正標本32をWへ再乗算しない。

Aの同一新J/U D総RZを固定すると、

\[
\mu_{break}=\frac{105532782860-49168}{10487392}
=10062.8195925164\ldots\ \mathrm{RZ/kernel}.
\]

厳密に下回るにはμ<μ_break。現在の標本平均19825.625からの必要な点価格削減尺度は**約49.2434%**。新しい平均価格・shots・Wを測った結果ではなく、既存値を固定した仮想条件である。新J/Uを最適化すればW、PF/RTE、N、shellも変わり得る。目的関数の必須閾値、達成可能性、全law下界、統計的有意差として採択しない。旧J/Uの6122.33 RZ等の損益分岐を新selectedへ転用しない。

## 4. 過去の負結果と再探索を避ける範囲（Q3・Q4）

### 4.1 Surrogate：cohort内とDF間を分ける

[修正後148-call][S]はM4/P4各32、固定4-fold（index mod4）のmean/A-switch/B-overlap/C-compositeを比較した。Cはswitches/ZZ_count/same_basis_support_pairsの3特徴＋切片。主指標はheld-out RZ MAEで、両cohortで全対照を改善する必要があった。

| Cohort | Mean MAE | A MAE | B MAE | C MAE | 不適格の直接理由 |
|---|---:|---:|---:|---:|---|
| M4 | 1248.5547 | 221.4073 | 305.6085 | 227.7980 | CがAを下回らない |
| tail-P4 | 1082.2266 | 287.4242 | 261.5380 | 273.5792 | CがBを下回らない |

全rankはfull、負予測0で、rank欠損による未判定ではない。CのRMSEが最小でも主MAEを事後変更しない。SwitchとRZのPearson .9844/.9733は同一DF内の記述相関であり、DF間の価格差・因果効果ではない。静的DFは2個で、64個ではない。Headや静的basis差はcohort切片に吸収され得る。保存raw opcode censusも単独head/basisのcompiled価格ではない。[S]

[第三γ0・493-call][T]では、旧64行で共通interceptを固定し、第三DF32行に再fitせず適用した。RZ MAEはmean2892.23438、A369.78698、B303.32856。Aはmeanを改善したがBより小さい登録条件を満たさず`SWITCH_TRANSFER_NOT_SUPPORTED`。これは予測力が皆無という結果ではないが、switch-onlyを新目的へ採択する根拠にもならない。観測平均17454.1875を説明した近似精度を、そのまま介入時のgradientやηへ移送できない。[T]

従って旧Cの改名・特徴追加・RMSEへの変更、旧A係数の新selectedへの直送は見送る。B-overlapがこの一つのholdoutで良かったことも、独立DF全体への一般化やUを近付ける介入の成功を認定しない。

### 4.2 Support・rank・係数再分解

| 既存案 | 取得済みの結果 | 再探索しないこと／未検証の区別 |
|---|---|---|
| [Tail-P4構造設計][SD]→[性能449-call][P4] | tail係数位置216→144、λ1.44243→1.51617、ideal shots16056→16568、W16631888→18303568、kernel平均RZ14076.44→15047.47、総RZ約17.64%増 | atomの種類数が減ってもK0/r128のevent数は128。別U・law・projected startの変化もあり、疎化だけの因果ではない。別maskを盲目的に第一候補にしない |
| [Shared basis][SB] | 一般対称JでG共有basisの和は通常CDF outer rank≤Gの表現集合 | 同basis Jを足す専用fitterは不要。PF fragment順・split・atom化は別問題で、Hが同じだけではPR law/価格を保存しない |
| [通常CDF7 native32][R7] | 既報kernel平均17754.9375、条件付き総RZ638880821593.5 | rank削減だけでは改善しなかった有限候補。CDF8全般や他rankの不可能性ではない |
| [固定U・tail保存LP][LP] | exact λ .954543284658→.954432028127 Ha、約.0116555%減、tensor/sector/DF/1h-step保持PASS | λ最小化の小さい改善。Native/総資源改善は未測定。最新selectedの余地を示す結果ではなく、weighted-price LPと同じでもない |
| [固定U・凍結event価格weighted mass][W] | 設計済み。価格を定数にした係数penaltyは実装可能 | 現在のcomplete P0の加算性も、使える固定価格も未資格。未実施を負結果と呼ばず、再命名して新案にしない |

P4ではZZ event期待数が減る一方、basis切替期待数は増えたという保存descriptorがある。しかしこの2点からGaussian増の因果割合は分離できない。Support数、outer/inner rank、raw中央命令数、compiled complete価格を別々に扱う。[S]・[P4]

### 4.3 Native構造・Gaussian融合

[既存native技術調査][N]と現SOURCEを照合した。StrongBuilderのtransitionは時系列Bi→Bj†に対して`Vj†Vi`を合成し、解析的vacuum phaseを補正する。実装済みのhead半sweep内部4境界、tail exact-key run共有、D全sweep相対融合を新削減として数えない。

| 構造 | 実装・証拠 | 今回の扱い |
|---|---|---|
| 異tail run間relative Gaussian | 人工PoCあり。2軌道complete RZ748→780、3軌道1903→2039。正しさはPASS | 2個を1個へ置換してもP0後RZは減らない反例。H4で常に悪化の証明ではないが、再度第一候補にする新根拠なし |
| Head/tail外側2境界 | 境界積の合法性は人工確認。Complete emitter/価格は未検証 | Head内部4境界とは別。未実装を削減実績と数えず、改善不能とも断定しない |
| 同一exact-key run内中央角度結合 | 局所人工ではRZ8→2。既存2/3軌道complete fixtureは価格不変 | Compilerと重複し得る。局所削減をwhole-P0へ移送しない |
| Singleton support completion | 辞書用BasisProviderには既存関数あり。現PR Providerはsingleton引数でもfull-frameを返す | 接続差は実在するがH4削減は未測定。[旧singleton/PS48調査][H]の限定結果・停止を保全し、選択法を新名で再開しない |
| 近いUのmerge・小角pruning | Exact key共有ではない。誤差とlawの扱いを変更する | 今回の固定条件に含めない |

P0 opt1は既存単qubit簡約・inverse cancellationを含む。未集約raw命令数がそのままcompiled削減量になるとは限らない。一般相対Gaussianがfull-frameよりdenseなrecipeになる可能性もある。過去人工結果を新selectedや全lawへ外挿しない。

### 4.4 Round・sampling・既知手法

[旧同J/U round割当189-call][RW]はideal/assumedの全13roundでDを選び`MIXED_ROUND_DOMINATED`。同じ保存条件の再計算は不要。ただしこれは旧J/Uの結果で、新selectedのround最適割当を今回測ったわけではない。今回も主RZをCX depthへ事後変更しない。

[Sampling law調査][LAW]はEuler/IS/Parent allocation/全path再重み付け等との対応から`NO_DISTINCT_FREEDOM`。安いeventを多く選ぶだけでは平均operatorを維持しない。各marginalが同じでも相関した積はIIDの積平均と同じとは限らない。現在のK0 common配分familyのnormalizer最小性は、全辞書や全費用の最適性ではない。新sampler名だけで既知構成との差が生まれるとは扱わない。

[RC-DF比較設計][RC]では一般signed Jの線形写像後smooth tail L1二乗と、全frame uniform entrywise L2を区別済み。符号制約・非平滑の場合にweighted L1二乗へ帰着する領域もあり、norm正則化の追加自体を独立新規性にしない。今回の限定一次資料確認でも、RC-DFは係数正則化、SCDFはsymmetry shiftと分解の同時最適化を扱う。[RC-DF式17・Appendix A](https://arxiv.org/html/2212.07957v3)／[SCDF §2.3–2.4](https://arxiv.org/html/2403.03502v1)。Qubitizationのλ/Toffoliを現PRのλR/whole-P0へ同一視しない。SCDFのshiftはsectorの補償位相・full-space targetを明示する別設計で、現κ/scalar固定へ無断追加できない。RC2の未収束も維持し、最新PRと最適RC-DF一般の勝敗へ広げない。

## 5. λ-onlyが持たない情報と、安価な評価の限界（Q5–Q8）

### 5.1 J、law、basis group

既存`c_pr_aware_df.coefficients`の一般K写像は、interleaved spin a=2i+σについて`K_ab=J_ij/2`、`c_Za=−Σ_jJ_ij/2`、`c_ZaZb=J_ij/4 (a<b)`。Identityは別位相に保存する。行和の符号相殺を含み、entrywise Jのsupportや絶対和だけではλを再現しない。

`c_a=L_a(J)`、`m_b=Σ_{a:key(a)=b}|c_a|`、`λ=Σ_bm_b>0`、`q_b=m_b/λ`とすると、K0の独立r eventで

\[
E[T]=(r-1)(1-\sum_b q_b^2),\qquad
E[R_b]=q_b+(r-1)q_b(1-q_b).
\]

同λでもqの集中度は異なる。既存`c_df_basis_switch.exact_law`と人工`test_mass_switch_tradeoff_and_nonconvexity`はその例を持つ。一般signed Jへ適用でき、計算は係数slot数に線形で安価。固定partition・非zero係数でのgradientも既実装で、零係数では非微分、λ=0は別分岐となる。[SW]

Smooth版は全slotに`√(c²+ε²)`を使うため、厳密zeroにもghost massがありexact sampling lawではない。固定group内のswitch項はU幾何を直接持たずU gradientは0。近いUがexact keyを共有するわけではなく、一致groupが微小変化で分裂すると指標が跳ぶ。Signed-zeroを含むkey生成は保存生産側`_matrix/_spin_lift`にそろえる必要があり、過去run01の近似照合で救済していない。[S]・[SW]

**既存λ目的と独立な情報はあるが、既に設計・人工検証済みの情報である。** 切替指標はスケール不変なのでλ増を抑制せず、固定Uでも一般に非凸。現PRの固定U J solveはtensor quadratic＋smooth mass二乗の構造を使う。単にηを足して既存Hessian・rank/nullspace・reduced-gradient admissionを維持できたことにはならない。新項を使うなら微分・内側資格の設計も必要で、今回は変更しない。

### 5.2 Uの幾何と回路価格

既存overlap `d(U,V)=n−Σ|UᵀV|⁴`は滑らかな幾何記述量だが、列符号／置換への不変性とnative宣言keyの同一性は違う。近接Uでもexact-zeroを維持するGivens/P0のgate数が連続的に下がる保証はない。一般Jと物理tensorを保持するgauge操作でもrecipe、中央support、相対境界を対応させる必要がある。[SW]・[N]

U変更はhead/relative/tail basis構造を変え得る。しかしDの旧/新kernel価格一致と、PRの旧/新価格上昇を並べてもU単独の寄与は同定できない。固定JでUだけ変えると通常はHamiltonianも変わり、固定UでJを変えてもlaw・λ・PF/RTEが変わる。比較用の「ablation」を作る場合も物理DF予算を独立判定しなければならない。

### 5.3 評価コストと情報量

| 指標・方法 | 評価コストと得られる情報 | Complete価格／最終資源との不足接続 |
|---|---|---|
| λ、smooth mass | O(atom slots)、既存解析/AD。固定K0のAを制御 | U/head・PF・P0価格は制御しない |
| Exact-key q/T、smooth dispersion | O(atom slots＋groups)、固定groupならJ微分可能 | 数学的にrun数を表すが、DF間のcompiled価格と介入効果は未資格 |
| Overlap | basis pairごとの小行列積。U微分可能 | 全frame距離・run予測からGaussian/P0 gate数へ直接写せない |
| 凍結event／basis価格の加重和 | 価格が既知なら安価な和 | 費用がrun履歴・head境界・中央角度・compilerへ依存。必要な単独価格は現資料でnull、whole-P0との誤差も未資格 |
| Complete kernelの実価格 | trajectory生成、build/P0、必要なcorrectness校正 | 対象の価格は測れるが有限cohort。全J/U optimizer評価ごとの実行コスト・ノイズ・微分可能性は未検証 |
| 全round PF/RTE/RPE＋Wμ+S | 候補固有reference、13round、整数ceil shots、価格校正 | 情報は直接的だが高価で非滑らか。安価な目的関数として既成事実にしない |

`Σ_b E[R_b]G_b＋中央＋head`は構造的な会計候補にはなるが、G_bやhead価格を取得済みとして埋めない。Whole-P0は部分回路価格の単純和とは限らず、depthも非加算。旧回帰係数・raw opcode差・標本最大値をG_b、厳密bound、ηへ置き換えない。

将来J/Uを変えるなら、まず正しさと独立DF budget、次に候補固有PF/RTE/A、条件付きN/W、complete kernel μ、最後にshell込み資源を判定する。DF誤差[Ha]、operator/signal誤差、RPE統計誤差、合成/law誤差を無根拠に加算しない。純粋な等価回路変換ならJ/U・event・理想PF/RTEを保持して価格差を調べられるが、変換後の合成誤差は別に必要。[A]・[B]・[N]

## 6. 候補の技術選別（Q9）

| 候補群 | λとの独立性・実装／最適化 | 物理・PRリスク／価格証拠 | 判断 |
|---|---|---|---|
| Jのbasis mass集中 | 独立情報あり。既存PoC/ADを再利用可能、安価だが非凸 | λ・DF・PFを悪化させ得る。登録cohort/第三DFの価格基準が不成立 | 新η-fitはNO-GO。旧surrogateの再命名にしない |
| Uのoverlap/Givens構造罰則 | λにない情報あり。Overlap微分は可能、exact recipe数は離散的 | 近いUが安い保証なし。Head・全P0との未分離性と旧relative反例 | 新U目的はNO-GO。任意のUを安価にできると仮定しない |
| 固定Uの凍結価格weighted mass | 既提案。非負定数重みなら現loss資産を利用しやすい | 加算価格自体が未資格。λ-only LPの小改善は価格改善証拠でない | 価格モデルの不足を残してNO-GO |
| Tail relative／中央結合／head-tail | 理想operatorを保存でき、既存合成資産あり | 既存complete人工でRZ増／不変。外側2境界は未測定 | 新H4実験を優先する根拠なし。改善不存在の主張ではない |
| Support/共有basis/rank | 既知制約CDF・低rankとの重複が強い | 固定P4・CDF7で資源改善なし。別条件へ一般化しない | 同条件の再探索は不要 |
| Sampling/round割当 | λ以外の価格・normalizer自由度はあるが既知枠組み | 固定law維持条件や誤差予算が変わる。旧round主RZはD全選択 | 今回のnative単価問題を解く新機構として採択しない |
| 毎点complete総資源最小化 | 最終taskと直接対応する問い | 高価、整数shots/compile/標本ノイズ。実用的勾配・評価負担未確認 | 大規模な新optimizerを作らない |

上表は、数学的独立性、既知法との重複、価格との予測根拠、微分と評価コスト、DF精度、PR誤差、約49%という現在の点価格差、反証可能性をまとめたもの。数学的に定義できることだけをGO基準にしていない。**今回の不足は新しい式の個数ではなく、操作可能な構造量とcomplete価格との実証的な接続**である。小さい改善にも研究価値はあり得るが、強Dとの差を縮めるという今回の目的への寄与は別に判定する。

## 7. 新実装・科学契約を作らない理由と再検討条件（Q10）

現時点では選定候補なし。従って新目的・runner・未承認科学契約を形式的に増やさず、既存SOURCEを保全する。新fit/追加trajectory/full native32/全13roundの一括実行案や、一点診断を連続させる案も作成しない。

再検討に必要なのは、例えば**現在のprecision/P0でcomplete kernelへ作用し、既存負結果と異なる具体的な変換**、または**価格を操作する方向について別DFでも反証可能な根拠を持つ指標**である。旧相関をもう一度計算するだけ、近いUを無料共有するだけ、未取得の価格を0に置くものはこの条件を満たさない。研究Cが不足部分の取得自体に情報価値を認めれば、それを新しい依頼として設計できるが、今回その採択を代行しない。

具体機構が得られた場合の評価原則だけを残す。回路等価変換なら同じ保存event集合を使うpaired complete-kernel比較にし、正しさ・phase/adjoint/X/Y・P0後6metricsを先に確認する。J/Uを変える機構なら同じ物理予算と候補固有law/誤差を先に確認する。どちらも選別済みの安いtrajectoryだけを使わず、結果後の指標変更・未知誤差の補完を避け、最初の一群で仮説を区別できる範囲を実行前に固定する。これは実行可能契約でも、科学callの許可でもない。

**次の担当：ユーザー／研究C**が、この限定NO-GOを受けて現系列の成果を整理するか、不足する価格機構の証拠取得へ研究範囲を定めるか判断する。**Codex**は、その具体的な範囲の依頼がある場合に限り後続技術設計を行い、新しい科学計算は別途明示許可後に実施する。今回の技術整理に追加GPT詳細レビューを挟んでいない。

## 8. SOURCE再利用・検証・保全

| 確認した既存SOURCE | 再利用した情報／今回変更0 |
|---|---|
| `c_pr_aware_df.py`、`c1_general_k_adapter.py`、`c1_real_orthogonal_codec.py` | 一般signed J写像、元目的、spin/phase、既存codec規約 |
| `c_df_basis_switch.py`、`c_df_basis_surrogate.py`／`_v2.py` | Exact key/q/run、smooth gradient、修正key生産、旧モデル評価条件 |
| `pr_dictionary_validation/circuits.py`、`c_pr_aware_df_stage_b.py` | Support completionと現Providerの差、event順・exact共有、lawとの接続 |
| `c1_h4_strong_baseline.py`、`c1_stage3_native_bridge.py`、`c_pr_native_structure.py` | Relative方向・位相、現head/tail境界、過去人工PoC |
| `c_pr_integrated_performance.py`、`c1_stage3_accuracy.py`、`c1_stage3_connection.py` | 保存3lane、N/W/serial metrics、共有校正、unknown資格 |
| `c1_molecule_extension.SectorCAR`、`c1_stage1.Calls`、`c_pr_aware_df_pilot.finish` | 独立物理評価とfresh/call/STOPの既存入口。今回は起動しない |
| `c_pr_coefficient_point.py`、`c_separated_j_u_consistent.py`、`c_j_u_stable.py` | 現innerの同一点評価・Hessian・独立gradient/stabilityと新目的導入時の非互換範囲 |

新たな有望候補を選定しなかったため、人工試験の追加・既存科学runner起動は行わない。既存[131人工/回帰PASS記録][VT]にはsigned J、同λ/別switch、非凸性、zero/epsilon、near-U・gauge、ffsim形式とADの試験があり、既存[68人工/回帰PASS記録][VN]にはphase/相対方向/full controlled/P0/adjoint/X/Yと価格反例がある。これらは**過去のlocal検証の引用**で、今回新しくPASSしたテスト数やCI／外部再現ではない。技術PoCを実H4性能へ資格昇格しない。

今回の検証は、固定refの資料取得、引用scalar/key・SOURCE定義の照合、§3のDecimal代数、Markdownリンク・Git差分・公開内容の照合。実H4 scientific calls=0、NPZ/QPY decode=0、optimizer/fit/build/compile/sampling=0。新しく算出した数値は§3の許可された代数的損益分岐のみで、既存PF/RTE/shotsや模型の再評価ではない。

独立worktree `/tmp/prt-c-native-price-feasibility-20261102`、branch `codex/research-c-native-price-feasibility-20261102`。正式共通運用指示2026-10-25版§8/8.1/8.4/10/10.1/10.3に従った。共有workspaceのdirty変更・他担当のB文書は触らず、旧SOURCE/raw/契約/grant/ledger/marker/STOPは保全する。新しい数値証拠集合がないためvalidation manifestや過去statusを改定せず、本報告への最小限の案内と追記だけを加える。承認証明・新grant・管理frameworkは追加しない。重い計算はなく、CPU/RAM/worker/wallの上限や共有環境変更は加えていない。

Commit/push後にremote SHA/tree、報告本文、起点からの変更pathと既存SOURCE・結果の不変を確認して完了する。**Mandatory STOP。今回のNO-GOから、新しいfit・再解析・科学計算や研究Cの正式終了へ自動進行しない。**

## 出典

[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38b8d0df39b8943178ecef611194819cff1c2169/docs/research/research_c_same_ju_strong_d_execution_20261102.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2639221a59713d431bc7c157854628bbc006a1bd/docs/research/research_c_pr_resource_contribution_analysis_20261102.md
[I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/docs/research/research_c_pr_integrated_execution_20261102.md
[S]: research_c_df_basis_surrogate_repair_execution_20261029.md
[SW]: research_c_df_basis_switch_optimization_design_20261029.md
[T]: research_c_df_basis_third_repair_execution_20261029.md
[SD]: research_c_df_structural_design_20261028.md
[P4]: research_c_df_structural_pr_performance_20261028.md
[SB]: research_c_shared_basis_df_design_20261029.md
[R7]: research_c_cdf7_native32_execution_20261030.md
[LP]: research_c_fixed_u_tail_lp_numerics_execution_20261101.md
[W]: research_c_pr_aware_df_optimization_design_20261026.md
[N]: research_c_pr_native_structure_optimization_design_20261030.md
[H]: research_c_hamiltonian_circuit_survey_20261024.md
[RW]: research_c_roundwise_det_pr_execution_20261030.md
[LAW]: research_c_sampling_law_freedom_survey_20261030.md
[RC]: research_c_regularization_comparator_design_20261101.md
[VT]: ../../artifacts/research_c_df_basis_switch/2026-10-29/technical/verification.json
[VN]: ../../artifacts/research_c_pr_native_structure/2026-10-30/technical/verification.json
[E]: research_c_evidence_synthesis_and_decision_readiness_20261031.md
