# 研究C：包括的科学レビュー

**実施日：2026-10-31**  
**対象：PR-aware DF、PF/RTE/RPE接続、native費用、構造変更、rank、round割当、sampling law、研究戦略**  
**レビュー起点：`354a090da388c6d9718624bb4525ff3c72a22022`**  
**Repository：`HIROMU1015/Partially-Randomized-Trotter`**  
**状態：承認された包括的レビューを実施。追加科学計算・保存値再解析・実験・repository変更は実施していない。**

## 0. 総合結論

研究Cの成果を「特に成果なし」とするのは不適切である。同じ初期値からのDF目的変更、物理sector予算、有限RTEの平均operator、条件付きRPE十分shots、実コンパイルした回路価格を接続した数値事例は成立している。一方、これを「独立した新しいDF分解法」「強い決定論PFを上回るPR」「保証付きの実用的量子優位」と主張する証拠はない。[R00–R04]

本レビューの推奨は、次のとおりである。

1. **A〜Cの限定的成果を保持する。** Tail λの34.7246%低下、固定s4/r16/K0での条件付き総RZ83.08598%低下は、記録された条件での有効な肯定的結果である。後のs1/r128でも同じ保存γ0候補の総RZがγ.1候補の2.678387倍と報告されており、改善の観測は最初のPR政策だけに限られない。ただし同じDF候補対の再利用であり、独立初期値・分子への再現ではない。[R01,R03,R07]
2. **新方式の独立新規性は未確立である。** CDFへの係数正則化、用途に合わせたλ低減、DFとPRの接続は既知である。研究Cの差分は、固定tailの実Z/ZZ係数写像、独立sector予算、PR/RPE/nativeまでの評価にある。これは研究上の差分だが、それだけで新しい一般原理を構成しない。[R02,R15,P1,P2]
3. **主RZについて、保存されたDとPR-r128の順位は維持する。** 同じJ/U・同じ対象energy・宣言された費用モデルでDが安い。ただし、この比較を全PR政策・最良実装・全資源指標に拡張しない。[R04,R06,R14]
4. **過去の一部の進行判断は射程を狭める必要がある。** 特にround別判定はDとr128の二方式だけで、保存済みr64を含まない。Nativeの見送りは人工2/3軌道例などに基づく優先順位判断であり、H4の全改善可能性の否定ではない。方法上の新規性がないことと、既知手法の適用に資源価値がないことも別である。[R05,R14,R16]
5. **現在の無差別な改善探索を止めることには賛成するが、研究C全体が科学的に行き詰まったとの認定には賛成しない。** 続けるなら中心課題を一つに絞る。ユーザーが重視してきたDF近辺の研究としては、「tail特化目的は、通常の既知正則化を同等に適用するより有効か」が最優先の未解決比較である。新しいsupport、rank、samplerを次々追加するより重要である。
6. **現状の推奨着地点は、task特化DF最適化のproof of conceptと条件付きPF/PR資源比較である。** 新一般アルゴリズム論文としてのGOは出さない。限定研究記録・ベンチマークとしての執筆は可能だが、独立した査読論文の重要性・十分性を確定するには中心claimと対照の不足を解消する必要がある。

正式な研究方針の採択、旧契約の変更、新しい科学計算の実行は、このレビューによって自動的には行われない。

---

## 1. レビューの実施範囲と証拠の区別

### 1.1 実際に行ったこと

統合資料の23項目Evidence Mapを通覧し、主要な一次実行報告、固定科学SOURCEの目的関数・一般K adapter・PF/RTE reference・RPE精度処理・位相復元器・native providerを取得して読んだ。M4の保存result JSONも取得した。肯定的成果、強baseline、r64/r128、第三γ0候補、native構造、round割当の報告を重点的に照合した。RCDF、PR、RPE訂正、SCDF、importance samplingの関連一次文献を確認した。

### 1.2 行っていないこと

元の分子NPZ/QPYの復元、fit、Hamiltonian対角化、PF/RTE数値評価、コンパイル、sampling、保存値の新しい比率・統計・ランキング・schedule計算、既存テストの再実行は行っていない。Codexが報告した全SOURCE hashや全2681/2830等の保護ファイルをGPTが再監査したわけではない。既存結果を再現実行したという認定もしない。

したがって、本書の数値は既存報告・保存JSONの引用である。数式についての独立した推論と、保存データについての新規数値再解析を区別する。

### 1.3 記述の読み分け

- **保存事実**：固定commitの報告または保存結果に記載された内容。
- **SOURCE確認**：取得したソースコードが実際に定義している処理。
- **本レビューの推論**：これらから導いた数理的・科学的解釈。
- **未確認**：必要な性能、独立再現、保証、比較がない事項。

23項目を一律に全原本まで再検算したわけではない。主要結論を左右する経路を重点的にレビューし、残りは統合Evidence Mapとその限定を確認した。この限界をもって、未検証事項を不存在や不可能性と扱わない。

---

## 2. 研究課題を三つに分離する

研究Cでは、途中から別の問題が一つの「改善」にまとめられがちだった。次の三つを分けるべきである。

| 問い | 現在の回答 | 残る問題 |
|---|---|---|
| Q-A：PRに向いたDFを生成できるか | 固定した候補対では、物理予算を保持してtail massと条件付きPR費用を改善した | 既知の正則化対照、初期値依存、accuracy–resource trade-off |
| Q-B：そのPRは強決定論PFより有利か | 保存された主RZ比較では不利 | PR側候補集合・既知の実装機会・policy最適化をどこまで揃えるか |
| Q-C：独立した新しいアルゴリズム原理か | 現在の資料では未確立 | 既存RCDF/PRの用途特化との違い、一般性または新しい数理機構 |

Q-Bが不成立でもQ-Aの観測は消えない。Q-Aが成立してもQ-Cが確立したことにはならない。この区別を主要claimの見出しから維持することが重要である。[R00]

---

## 3. DF目的・表現・物理精度のレビュー

### 3.1 実装している目的関数

固定SOURCEは

\[
F_\gamma(J,U)=\frac{f(J,U)}{f_*}
+\gamma\left(\frac{M_\epsilon(J)}{M_*}\right)^2,
\qquad
f=\frac12\|g-g_{\mathrm{DF}}(J,U)\|_F^2
\]

を用いる。一般対称J、実直交U、rank8、元順prefix LD2、共通のqualified x0を使い、γ=0と0.1を比較している。`f_*`と`M_*`は共通初期値から固定される。[R01,R02]

一般number fragmentのspin-orbital行列は

\[
K_{ab}=\frac12J_{\lfloor a/2\rfloor,\lfloor b/2\rfloor},
\]

であり、非identityの係数は

\[
c_{Z_a}=-\frac12\sum_b K_{ab},\qquad
c_{Z_aZ_b}=\frac12K_{ab}\quad(a<b).
\]

この線形写像をtail frameに適用し、\(M_\epsilon=\sum\sqrt{c^2+\epsilon^2}\)、判定にはexact \(\lambda_R=\sum|c|\)を使う。Identityは決定論位相へ抽出される。[R02,R15,R18]

**本レビューの判断**：実際にsampleする係数に正則化を接続している点は妥当である。単なるJの要素ノルムと、実tailのZ/ZZ massを取り違えてはいない。Uはtensor reconstructionを通して間接的に最適化されるが、このpenalty自体はUの近接性やGaussian価格を直接測っていない。

### 3.2 精度と資源の目的は同じでない

Tensor Frobenius residualからsector誤差を推測せず、物理sectorのoperator normを別に評価している点は支持する。これはRCDF等の誤差指標と比較する際にも必要な区別である。[R01,R18]

κは元gから補正して固定され、gの新近似ごとに勝手に変更されていない。これは宣言された近似Hamiltonianの定義であり、独立CARおよび物理Hamiltonianとの差の評価と組み合わせて読む必要がある。別のnormal-ordering規約を用いた文献のλと、値だけを直接比較してはいけない。[R15,R18]

### 3.3 最適化の成功と収束保証

保存値は次のとおりである。[R01]

| 指標 | γ0 | γ0.1 |
|---|---:|---:|
| Exact tail λ [Ha] | 2.2097618367788057 | 1.4424298855756243 |
| Sector norm＋guard [Ha] | 1.0297636237465247e−8 | 1.229561498918692e−8 |
| Iterations | 97 | 1129 |
| 正規化された目的の終了gradient infinity norm | 0.7423484410886623 | 1.4825504139166679 |
| 停止理由 | 相対目的減少 | 相対目的減少 |

`success=True`は停留点や大域最適の証明ではない。もっとも、正規化されたgradient値をraw objectiveのgradientと直接比較し、値の大きさだけで不具合と断定するのも誤りである。

**結論**：今回の終点が有効なDF候補であることと、最適化法の到達限界は別。前者は支持される。後者は未判定である。対比較の科学的価値を取り消す必要はないが、「最適なPR-aware DFを得た」とは言えない。

### 3.4 未使用のDF精度予算

許容予算は約\(2\times10^{-4}\) Haなのに、両終点のsector誤差＋guardは約\(10^{-8}\) Haである。これは保存値の桁の比較であり、新しい数値実験ではない。[R01]

この余裕は「さらに必ず安くできる」証拠ではない。しかし、現在の小さい初期tensor lossで正規化された目的が、物理予算の境界に沿って資源を最小化したとは言えない。**現実装は一つのpenalized fitであり、同じ物理精度の下での最適なaccuracy–resource Pareto解ではない。**

追加γ探索を無条件に始めるべきではないが、研究全体の最適性や資源下限を主張する際にはこの未探索性を残す必要がある。

### 3.5 固定tailの意味

正則化は元順LD2の後ろだけを罰する。Headの係数massを罰していないため、再表現の自由度によって重みをheadへ移せる可能性がある。これは禁止された操作ではなく、task-aware設計の一部になり得る。ただしheadの非可換性、PF誤差、実価格を含めない限り、tailの減少を総資源改善と同一視できない。

同じx0と実装で目的だけを変更した比較は、**この固定手順を変更した効果の対比較**として意味がある。「すべて単なる相関」と切り捨てる必要はない。一方、λだけを変えた介入ではなく、J/U・head・lawが同時に変わるので、83.09%をλだけの因果効果とすることはできない。

---

## 4. Finite RTEとRPEの数学的整合性

### 4.1 K0の平均operator

Tail identityを分離し、\(H_R^0=\lambda_R\sum_\ell p_\ell P_\ell\)、\(P_\ell^2=I\)とする。K0では

\[
\tau=\frac{\lambda_R h}{r},\quad
\phi=\arctan\tau,\quad
B_0=\sqrt{1+\tau^2},\quad
\mathbb E V_\ell=\frac{I-i\tau H_R^0/\lambda_R}{B_0}.
\]

これは有限一次polynomialのnormalized meanであり、exact exponentialそのものではない。補正operatorの非unitarityはこの構成では不具合ではない。有限cutoffのbiasはPFとは別に評価されている。[R08,R19]

IID microstepおよびfresh independent coherent occurrenceを使う場合に限り、ordered productの平均を積へ分離できる。\(n_m=s2^m\)に対して

\[
A_m=B_0^{-rn_m}=(1+(\lambda_Rh/r)^2)^{-rn_m/2}
\]

となる。同じsample circuitを\(n_m\)回反復する場合の平均\(\mathbb E[W^{n_m}]\)とは違う。研究Cがこの区別を明示していることは正しい。[R03,R08,R19]

### 4.2 λの二乗penaltyに意味はあるか

上式より、固定h/r/nではλ低下はA増加へ単調に作用する。また小さいτでは

\[
-\log A_m\simeq\frac{n_m\lambda_R^2h^2}{2r}.
\]

したがって、λの二乗型penaltyを局所的な代理目的とする根拠はある。これは本レビューの数式からの推論である。ただしPF/RTE誤差とnative価格は同時に変わり得る。正しい代理目的であることと、全taskの最適目的であることは別である。

### 4.3 Operator誤差からraw signalへ

取得したreferenceコードは、各round全体について

\[
\epsilon_{\rm PF,m}=\|U_{\rm PF}(h)^{n_m}-e^{-iH_{\rm DF}t_m}\|_2,
\]
\[
\epsilon_{\rm RTE,m}=\|F_h^{n_m}-U_{\rm PF}(h)^{n_m}\|_2
\]

を計算する。既にround全体の誤差なので、accuracy側で反復回数を再度掛けてはいない。[R08]

Own-DF ground oracleを仮定すると

\[
|z_m-A_me^{-iE_{\rm DF}t_m}|\le d_m,
\quad d_m=A_m(\epsilon_{\rm PF,m}+\epsilon_{\rm RTE,m})+
u_{\rm syn,m}+
u_{\rm law,m}.
\]

\(d_m<A_m\)なら、原点から円板への接線を考えることで

\[
|z_m|\ge\rho_m=A_m-d_m,
\quad |\arg z_m+E_{\rm DF}t_m|_{\rm circ}\le\arcsin(d_m/A_m)
\]

となる。A自体をraw signalの絶対値と呼ばない点も妥当である。[R09]

### 4.4 統計位相誤差とbranch復元

X/Yの各標本平均の許容誤差を

\[
\eta_m=\rho_m\sin\beta_{\rm stat}/\sqrt2
\]

とすれば、複素平面での統計誤差半径を\(\rho_m\sin\beta_{\rm stat}\)以内に抑えられる。±1 outcomeのHoeffdingから十分shotsを与えるという既存構成は整合的である。Weighted importance sampling outcomeには同じ±1の式を無条件移植できない。[R09,R19,P5]

実際のdecoderは前roundの推定位相に最も近いbranchを選ぶ。前段のbase-phase誤差が\(\beta/2^{m-1}\)、当roundの正しい枝が\(\beta/2^m\)以内なら、距離は最大\(3\beta/2^m\)。枝間隔の半分\(\pi/2^m\)より小さい\(3\beta<\pi\)が十分である。設定\(\beta=\pi/6\)はこれを満たす。RPE論文の訂正もπ/2型の誤った枝条件とπ/3条件を区別しており、この点は重要である。[R10,P3]

この議論は共通target energyへの各roundの偏りが制限されれば成立し、同一の近似回路を全roundで使うこと自体は要求しない。従ってround別D/PR切替の数学的な成立性は支持する。

最終energyへの戻しは

\[
|\widehat E-E_{\rm physical}|\le\epsilon_{\rm DF}+
\frac{\beta_{\rm total}}{\delta2^M}
\]

であり、DF誤差は一度だけ加える。各軸のfailureをunion boundで合算する際、round間独立は不要だが、各軸の集中不等式に必要なper-shot条件は維持しなければならない。[R09,R14]

### 4.5 数学の前提と保存数値の資格

理想theoremはHermitian Hamiltonian・unitary basis・適切なIID lawを前提とする。一方、元のbinary64 κ/Uには厳密Hermiticityや直交性との微差があり、研究Cはnumerical pilotとK1 targetのvalidated保証を別に扱っている。固定F2/F3-K1の証明を、元literal targetやr128全lawへ移すことはできない。[R00]

本レビューで読んだ主要経路では、誤差の単位混合、DF誤差の13重計上、Aの重複適用といった致命的な論理不整合は見つからなかった。ただしこれは全SOURCE・全数値の完全監査や、数値guardを厳密証明へ格上げしたという意味ではない。

---

## 5. 資源会計・統計のレビュー

### 5.1 宣言された会計式は整合している

\[
C_k=\sum_{m,a}N_{m,a}\bigl(n_m\mu_k+S_{a,k}\bigr)=W\mu_k+S_k
\]

は、fresh trajectoryのcomplete kernelをfenced serialに合成し、outer ancilla shellをinterrogationごとに一度だけ入れるという政策の下で妥当である。Kernel内のdeterministic headと、Hadamard readoutのheadを区別している。[R03,R06,R14]

同じ32本で価格平均を校正し全roundへ使うため、価格由来の記述的SEが\(W\,\mathrm{SE}(\bar C)\)になる点も正しい。13個の独立校正として\(\sqrt{13}\)で小さくしていない。[R03,R06]

### 5.2 何が実測され、何がモデルか

実行されたのは古典計算上のfinite native回路生成・コンパイル・全operator検査である。報告された数兆RZは、それだけの量子gateを実機実行した数ではない。Shotsも式による十分数であり、実quantum shot数ではない。

32 trajectoryのfull512検査は強い有限回路検証だが、32個の独立DF、全law合成上界、ideal-law母平均費用、再現性の外的検証ではない。観測min/maxからsupportを作らず、nullを残している点は妥当である。[R03,R06]

### 5.3 上界を小さくしたことと必要費用を小さくしたこと

Operator normと三角不等式、Hoeffdingに基づく十分shotsは安全側の構成であり、一般的な必要最小shotsではない。実際のcomplex signalにおける位相・振幅の誤差が小さい場合にも、uniform operator boundは大きくなり得る。

したがって主張は「同じ登録評価規則で算定した条件付き十分資源の削減」であるべきで、「その手法に本質的に必要な量子資源を83%削減した」という下界的な読み方を避ける。これは報告値を無効にする指摘ではなく、比較対象の意味を定める指摘である。

### 5.4 指標・実装モデルの範囲

RZ count、CX count、各depth、sizeは別指標である。RZはfault-tolerant T/Toffoli数でもhardware実行時間でもない。角度合成誤差の予算、routing、measurement/reset、state preparationを入れれば順位が変わる可能性がある。現在の主RZを結果後にCX depthへ変更することはしない。

Own-DF oracle・状態準備除外は両方法に宣言された条件であり、限定比較としては利用できる。ただし各DFで異なるground stateを無料で供給していることを、実際の状態準備アルゴリズムを含む優位と呼ばない。[R00,R14]

---

## 6. 主要肯定的成果の独立評価

### 6.1 保持すべき結果

| 比較 | 保存された結果 | 本レビューの評価 |
|---|---|---|
| γ0→γ.1、同x0・CDF8 | λ 2.20976184→1.44242989、両DF予算PASS | 特定最適化手順に対する有効なpaired設計の結果 |
| 同候補対、s4/r16 | Ideal総RZ 3,194,333,274,946→540,290,160,109 | 固定policy内の条件付き資源改善を支持 |
| 同候補対、s1/r128 | γ0総RZ627,057,959,562.5、γ.1総RZ234,117,788,135、保存比2.678387 | 別の保存PR政策でも同方向。独立DF汎化ではない |
| Proposed s4/r16→s1/r64 | 総RZ約5.403e11→約2.952e11 | Head反復・PF誤差・価格の交換条件の実例 |
| Proposed r64→r128 | 総RZ約2.952e11→約2.341e11 | Kernel単価増加を高mのshots重み削減が上回る例 |

値は既報の転記であり、新しい比率計算はしていない。[R01,R03,R05,R06,R07]

83.09%は見せかけだけと退けるべきではない。同時に、これを各方式の最良policyを探索した後のbest-to-best改善率と扱ってもいけない。特にStage Aの通常γ0とProposedの両方に、同じ程度のPR政策最適化機会を与えた最良値同士の比較は、現在の代表値とは別問題である。

### 6.2 M1の機構解釈

保存M1では価格側よりshots側の対称配分が大きい。これは会計のfactorizationとして意味があるが、J/U、λ、PF誤差、sampling distributionを別々に固定した因果実験ではない。今後「減衰を改善すれば同じ割合で総費用が下がる」と一般化しない。[R00]

---

## 7. 強baselineと実装比較の重要な未決着点

### 7.1 強決定論S2を置いたことは正しい

同じProposed J/Uを使ったDの総RZ101,825,929,455と、PR-r128の234,117,788,135の差は、現在のモデルでは保持すべき結果である。弱いXDFや古い不利な実装だけを比較対象にしなかったことは重要な改善である。[R04,R06,R14]

ただし「同じJ/U」と「PF以外が完全同一」は違う。Dには全frame sweepのrelative融合、PRにはhead内部relativeとtail exact-key共有という異なる適用scopeがある。両方に同じcompilerを使っているだけで、最良の実装機会が完全に揃ったことにはならない。

### 7.2 強い正則化対照が不足している

γ0との比較は、通常tensor-fitからtail-aware正則化を加えた効果を測る。しかし**用途に適切な既知RCDF正則化より、このtail特化設計が優れているか**は示していない。

独立したDF方法論を主張するなら、tail特化の係数写像・固定splitに利点があるのか、一般的な係数regularizationでも同じ改善が得られるのかを分ける必要がある。比較では同じ物理誤差目標、初期化、停止診断、古典最適化への機会、下流PR評価を明示すべきである。単純に同じ数値のγを使うだけでは、scaleと目的が異なるため公平性を保証しない。

RCDFは既にweighted L1/L2や係数・回転の最適化を扱う。研究Cのtailの線形写像と目的全体がRCDFの特定実装に完全一致するとまでは断定しないが、「norm罰則の追加」だけでは新規性を主張できない。[P1,R02,R15]

### 7.3 Full-frame providerと既知のsupport活用

M4の固定SOURCEでは、`Provider.basis(atom, singleton=False)`がsingleton引数を利用せず、常に精密なfull-frame Gaussianを返す。一方、generic `BasisProvider`には、1または2列のsupportを保持して他の列を補完する経路が存在する。[R11,R12]

PR原論文でも、1/2-qubit Pauliを回転した項に必要なbasis変換は全基底変換とは異なる費用構造を持つことが論じられている。[P2]

ただし、この研究系列の過去資料ではGaussian singleton／support-union／run分割を別条件で検討済みである。PS48等の小さいsynthetic、rank12/LD3/δ=.02かつorder2固定のH4などの条件であり、一律置換の利益や独立した算法としての価値が確立したわけではない。これらの終了判断を無断で覆さない。[R13]

**レビュー結論**：support活用を新発見として再提案しない。同時に、独立算法としてのNO-GOを、現CDF8/K0/r128の比較で既知実装を考慮しなくてよい理由にも使わない。現在の費用は「このfull-frame政策」の費用として報告できるが、最良既知PR実装全般の費用とは呼べない。

### 7.4 Small-H4と大系の原論文は異なる問い

H4の一条件の費用比較は有用な機構検証である。しかし原PR論文が扱う多数項・大系・Toffoli評価や別の分割・誤差評価と、そのまま同じ主張ではない。研究CのD優位は原論文全体への反証ではない。[P2,R00]

この指摘は、分子や結合距離を自動的に変える提案ではない。現在のH4を保ったまま、主張を「限定benchmark」に絞る選択も正当である。

---

## 8. 過去のNO-GO判定の再評価

| Evidence | 維持すること | 修正・限定すること |
|---|---|---|
| E01–E04 PR-aware肯定例 | 物理予算・λ・固定policy条件付き費用の改善 | 新規性、一般最適性、強D優位には拡張しない |
| E05 M1分解 | Shots側の大きな会計寄与 | 因果寄与率やλだけの効果ではない |
| E06 D対照 | 同J/U強Dが登録主RZで低い | 全PR/全実装・PFだけの純粋ablationではない |
| E07–E09 M3/M4 | PR政策変更による改善 | 低m/highmに同じrを使う最適性は未証明 |
| E10–E11 P4 | 当該support制約fitは適格、主費用は悪化 | 疎DF一般が無効とは言えない |
| E12 複合surrogate C | 登録MAE基準の不成立 | 全basis特徴量の予測力ゼロではない |
| E13 switch目的 | 固定basis groupでは主にJ mass配分を扱う | U幾何の滑らかな最適化を実現したとは言えない |
| E14 第三DF移送 | AがBを上回る登録基準は不成立 | A/Bは定数平均より良い保存結果。移送情報が皆無とはしない |
| E15 Shared-basis | 一般対称J、同一U群はΣJで統合できる | 制約付きJ、近似基底、別atom化まで同値とはしない |
| E16–E18 CDF7 | 一つの精度適格候補、主RZの点改善なし | rank7γ0対rank8γ.1は正則化等も交絡。小差を母平均のrank効果にしない |
| E19 native構造 | 現候補の追加実験見送りは資源判断として可能 | 人工2/3軌道・短event列はH4/全native改善の反証ではない |
| E20 round割当 | 固定D/r128では全13D | r64を含む候補集合・全schedule政策へ拡張しない |
| E21 sampling law | 調査範囲で独立機構未特定、Parent内L1最小 | 既知手法での性能改善不可能性、全辞書最適性とは別 |
| E22 CDF/XDF | 強いnative実装で旧優位が反転した限定結果 | CDF/XDF一般の優劣ではない |
| E23 validated固定回路 | 特定K1 targetの厳密上界資産 | 全law、r128、実CA、元target全誤差への自動移送不可 |

出典はEvidence Mapと重点取得原報告である。全行を同じ強さの新しい定理として再認証していない。[R00,R05,R07,R14,R16,R19]

### 8.1 最も具体的な見落とし：r64を含めないround割当

189-callで比較した集合はDとPR-r128だけである。一方、既存M3のr64はkernel平均9,105.59375 RZであり、Dの11,026より低い。r64は低roundで減衰も小さく、全13roundで登録十分条件を通過したという保存証拠がある。[R05,R14]

これだけで混合scheduleの総費用改善を認定してはならない。必要shots、誤差lane、shell、共通energy bindingまで揃えた再会計は未実施である。本レビューはその新しいランキングを計算していない。

それでも、「固定二方式で全Dだからround別PRは研究価値がない」という広い停止理由には不十分である。r128が単一policy全体ではr64より低費用だったことは、すべてのroundでr128がr64より有利であることを含意しない。**最初の候補選定で全体最良のPR一つだけを残す操作は、round最適化の問題を縮めてしまう。**

### 8.2 Surrogateの不合格を情報の不存在にしない

第三DFでは、保存MAEはconstant mean 2892.23438、switch A369.78698、overlap B303.32856である。AがBを上回る事前登録条件は不成立だが、A/Bとも予測に有用な情報を持つという記述的観測は残る。[R07]

したがって、この実験は「switch-onlyを最良のモデルとして採択しない」判断を支持するが、「basis-aware最適化全体を棄却する」根拠にはならない。実DFは3種類、holdoutは1種類なので一般化の証拠は弱いが、負結果だけに要約するのも不正確である。

### 8.3 新規性と有用性を分ける

K0 common-parentのnormalization最小性は、そのfamilyでは三角不等式から従う。一方、別の辞書・affine nullspace・同じnormalized meanを保つq・既知importance samplingによる費用交換まで否定しない。周辺分布を保持するだけではoperator積の平均が保存されないという反例も妥当である。[R19,P5]

`NO_DISTINCT_FREEDOM`は独立した新しい機構を特定できなかったという結論である。「既知手法を組み合わせても今回の量子資源が下がらない」とは違う。新理論を狙う研究を区切る理由にはなるが、既知手法を含む比較実装の完成度を認定する理由にはならない。

---

## 9. 先行研究と独立新規性

### 9.1 現時点で既知と判断する部分

RCDFにはtensor近似に係数正則化を加える枠組みがあり、weighted L1/L2も検討されている。PR原論文には表現・軌道基底・λ、重みを少数項へ集中させる方向とDFとの接続がある。SCDFは対称性とDFを使った別の用途特化資源最適化である。[P1,P2,P4]

よって、「DFに罰則を足す」「λを小さくする」「DFをPRに用いる」だけの独立新規性は支持しない。

### 9.2 研究Cの差分

実tailのZ/ZZ係数への線形写像、固定head/tail政策、共通x0でのfit、物理sector誤差予算、finite-RTE/RPE/P0までの統一的評価は具体的な差分である。この差分は、用途特化した設計・評価の貢献候補になる。

ただし、目的が既存の一般regularization familyに含まれ得ることと、実装・task・評価が完全同じであることは同義ではない。「数学的に既知枠組みだから研究価値ゼロ」とも、「実装差があるから新算法」とも判断しない。

**現状の最も妥当な分類は、既知CDF正則化をPR taskに特化した方法と、その数値的資源評価である。** 独立した強い方法論的貢献を主張するには、通常の適切な正則化に対する追加価値、または一般的に使える設計原理の証拠が必要である。

### 9.3 文献確認の限界

本レビューは関連一次資料の重点確認であり、全論文・全公開実装に対するpriority調査ではない。PRの指定v2本文取得はツールのcache問題で成功しなかった。無版PDFの本文抽出と実画像に版の不整合があり、実画像で確認できた2025年v1の該当節を採用した。固定v2の全本文を読了したとは扱わない。主な実装正否の判断は、repositoryの固定SOURCEと保存原報告に立脚する。

この取得上の限界によって、既知の正則化やsupport付きbasis変換が存在するという確認を取り消す必要はないが、2026年の最新本文との差まで完全に監査したという主張はしない。

---

## 10. 研究運用についての反省と修正

過去のGPTの進行提案には、技術的NO-GOを受けるたびに別の候補へ移り、研究の中心仮説と強い対照を再確認する機会が遅れた問題があった。今回、その点は修正する。

### 修正すべき三点

**第一：局所的な停止を積み重ねても、未探索の研究領域全体の否定にはならない。** 固定候補、特定surrogate、人工native例、方法差未特定は別種の結果である。

**第二：新規性確認と実用上の比較対照整備を混同した。** 既知の方法であるから新アルゴリズムとして追わない判断と、既知の方法を強baselineとして適用すべきかは別である。

**第三：追加実験の情報価値の優先順位が適切でない箇所があった。** 既知正則化との対照、保存r64のround適合性、current providerのcoverageといった中心claimに直結する論点より先に、多数の構造候補・surrogateの系列へ広がった。

これはCodexが登録条件を守らなかったという意味ではない。報告は多くの場合、scopeと未資格事項を適切に限定している。広い方向判断を担ったGPT側の課題である。テスト数、call数、保全ファイル数は再現性資産だが、科学的独立性や研究の重要性を代替しない。

---

## 11. 論文化の評価

### 11.1 現在支持できるストーリー

> 固定された分子・PR政策で、通常CDFのtensor fitをtail-awareに変更すると、物理精度予算を維持しながら条件付きPR資源が改善する。しかし、実コンパイル費用とshotsを合わせて同J/Uの強決定論対照と比較すると、PR内部改善だけでは全体の競争力を保証しない。

この主張はA〜Cと強D比較を矛盾なくまとめる。負結果を主張の補助として使える。一方、「DF新算法が一般的にPRを最良化する」「PRを超える/下回る方式を一般に証明した」という主張は支持しない。

### 11.2 何を今すぐ書けるか

技術報告・研究ノート・限定benchmarkとしては、既存の条件と未資格事項を明示して記述できる。これまでのコード・原本・正負の結果はその基盤になる。

独立した方法論的論文としては、既知RCDF等との差を性能・設計原理のどちらかで示す必要が残る。比較研究としては、強PR実装のcoverageとcandidate集合を明記し、「現実装の比較」と「手法の最良性能比較」を分ける必要がある。

H4一条件だけで一般性や漸近優位を主張することはできない。しかし、それを理由に無条件でH6や別geometryへ進む必要もない。現在のH4でしか主張しないcase studyにするか、より広いclaimを選ぶかによって追加データの必要性が変わる。

### 11.3 厳密保証は自動的な投稿条件ではない

実装CAや全law保証を定理として主張するなら、その証拠は必要である。しかし、明示仮定付きの資源benchmarkとして書く場合、全round・全lawのArb等の厳密囲い込みを完成させることを一律の先行条件にすべきではない。

一部のfixed circuit validated結果を保持しつつ、主要数値をnumerical pilotと表示するという立て方は可能である。保証研究を再開しても、既知法との比較不足や主張の一般性が自動的に解決するわけではない。

---

## 12. 次の研究判断：優先順位を絞る

### 12.1 今回の推奨

**同種の新アルゴリズム候補を増やす探索は一区切りとする。ただし、中心成果を主張するための対照確認は別問題として残す。**

ユーザーがこれまで重視したDF近辺の研究に戻すなら、次の問いに一本化することを推奨する。

> 固定H4・同じ物理予算・比較可能な最適化条件で、tail特化目的は、通常の既知の係数正則化よりも下流PR資源に有利か。

その差が残れば、task-specific DFの研究価値を強める。差がなければ、独立した新方式の主張を取り下げ、既知構成の統合評価としてまとめる。どちらの結果も判断を変える。

この比較のために新しいfitterやsimulation engineを作る必要はない。既存ffsim、係数写像、CAR、PF/RTE、RPE、nativeを再利用し、目的とscale・停止状態・許容物理誤差を揃える必要がある。ただし本レビューでは対照の数値parameter、fit回数、科学callを採択・実行していない。

### 12.2 別の中心claimを選ぶ場合

PF/PRの資源比較を主論点にする場合は、まず保存D/r64/r128を同じround評価へ接続する最小再解析の価値がある。新fitやcompileをしなくても、既存r64を除いたことの影響を判定できる可能性がある。ただし、それは新しい保存値再解析であり、別途許可が必要である。

その後、より一般的なPRの劣位を論じる必要がある場合だけ、既存support活用・compiler政策の適用範囲を確認する。ここで既知のsingleton研究を新算法として再開したり、最初から別の32回路実験を積み増したりしない。

この二つの経路を同時に全部進めることは推奨しない。**方法論としてのDFの追加価値を調べるのか、PF/PR比較の範囲を完成させるのかを先に決める。**

### 12.3 終了を選ぶことも合理的

現在の成果を限定benchmarkとして保存し、追加の計算を行わず別研究へ移ることも合理的な研究資源判断である。終了のためにすべての可能性を反証する必要はない。

ただし終了理由は「この研究目的に対して追加投資を優先しない」であり、「PR-aware DFに成果がない」「あらゆる既知・新方式で改善不可能」ではない。

---

## 13. 最終判定表

| 論点 | 本レビューの判定 |
|---|---|
| 主要保存値を撤回すべきか | 重点確認した経路では、その根拠を見いだしていない |
| DF→finite RTE→条件付きRPEの論理 | 宣言された理想条件・数値pilotの範囲で整合 |
| PR-aware目的に動機があるか | ある。λ²と減衰の関係から局所surrogateとして合理的 |
| 83.09%は研究成果か | 固定policy内の条件付き資源改善として保持 |
| 強決定論に勝ったか | 保存主RZ比較ではいいえ |
| 独立した新一般DF方式か | 現在の証拠では未確立 |
| Surrogate/rank/native/lawの全般的NO-GOか | いいえ。各固定条件・family・調査範囲に限定 |
| Round混合一般を終える根拠があるか | D/r128の結果だけでは不足。r64が既存候補として残る |
| strict保証作業を今再開すべきか | 主要claimに厳密保証を選ばない限り、優先しない |
| 次に無差別な探索を増やすべきか | いいえ |
| 次のDF研究の核 | 既知正則化対照に対するtail特化の追加価値 |
| 正式終了・実験採択 | ユーザー／研究Cの判断。レビューだけで自動変更しない |

---

## 14. 参考資料・取得したSOURCE

以下のURLは固定commitを優先する。R00のEvidence Map経由で参照した結果は、そのことを本文で区別した。

### Repository資料

- **[R00] 統合Evidence Map**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/354a090da388c6d9718624bb4525ff3c72a22022/docs/research/research_c_evidence_synthesis_and_decision_readiness_20261031.md
- **[R01] Stage A実行報告**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/354a090da388c6d9718624bb4525ff3c72a22022/docs/research/research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md
- **[R02] Stage A固定目的SOURCE**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e1085c95f5c01b68566e17d543ace5832cb94061/src/trottertracks/c_pr_aware_df.py
- **[R03] C1条件付き資源報告**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/354a090da388c6d9718624bb4525ff3c72a22022/docs/research/research_c_pr_aware_df_stage_c1_execution_20261027.md
- **[R04] M2強決定論報告**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/354a090da388c6d9718624bb4525ff3c72a22022/docs/research/research_c_pr_aware_df_M2_same_JU_deterministic_execution_20261028.md （主要値はR00/R14とも照合）
- **[R05] M3 r64報告**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/354a090da388c6d9718624bb4525ff3c72a22022/docs/research/research_c_pr_aware_df_M3_s1_r64_repair_execution_20261028.md
- **[R06] M4保存結果JSON**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/354a090da388c6d9718624bb4525ff3c72a22022/artifacts/research_c_pr_aware_df_M4_native/2026-10-28/run01/result.json
- **[R07] 第三γ0 DF報告**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/354a090da388c6d9718624bb4525ff3c72a22022/docs/research/research_c_df_basis_third_repair_execution_20261029.md
- **[R08] B1固定reference SOURCE**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9/src/trottertracks/c1_stage3_reference.py
- **[R09] Accuracy SOURCE**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/79fa0107658c92c3b3169d62348498c7dc37fab0/src/trottertracks/c1_stage3_accuracy.py
- **[R10] RPE decoder SOURCE**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/79fa0107658c92c3b3169d62348498c7dc37fab0/src/trotterlib/rpe_four_round_phase_validation.py
- **[R11] M4のfull-frame Provider**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad/src/trottertracks/c_pr_aware_df_stage_b.py
- **[R12] Generic BasisProvider/support_completion**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad/src/trottertracks/pr_dictionary_validation/circuits.py
- **[R13] 以前のHamiltonian/circuit調査**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/354a090da388c6d9718624bb4525ff3c72a22022/docs/research/research_c_hamiltonian_circuit_survey_20261024.md
- **[R14] Round別189-call報告**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/354a090da388c6d9718624bb4525ff3c72a22022/docs/research/research_c_roundwise_det_pr_execution_20261030.md
- **[R15] PR-aware目的の設計資料**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/354a090da388c6d9718624bb4525ff3c72a22022/docs/research/research_c_pr_aware_df_optimization_design_20261026.md
- **[R16] Native構造調査原報告**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/354a090da388c6d9718624bb4525ff3c72a22022/docs/research/research_c_pr_native_structure_optimization_design_20261030.md
- **[R18] General-K adapter固定SOURCE**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e1085c95f5c01b68566e17d543ace5832cb94061/src/trottertracks/c1_general_k_adapter.py
- **[R19] Sampling law調査**：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/354a090da388c6d9718624bb4525ff3c72a22022/docs/research/research_c_sampling_law_freedom_survey_20261030.md （結論はR00および会話中の取得済み本文と対応）

### 外部一次資料

- **[P1] Oumarou et al., Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization**, arXiv:2212.07957v3。Regularization式17、weighted L1/L2、最適化appendix、用途依存λの関連箇所を確認。https://arxiv.org/html/2212.07957v3
- **[P2] Günther et al., Phase estimation with partially randomized time evolution**, arXiv:2503.05647。取得版の制約は§9.3。実画像の§VII.A/Bにおけるλ/表現最適化、DF・support basis変換、大系費用評価を確認。https://arxiv.org/pdf/2503.05647
- **[P3] Kimmel, Low, Yoder, Robust Calibration of a Universal Single-Qubit Gate-Set via Robust Phase Estimation**, arXiv:1502.02677v3、冒頭2021訂正を確認。https://arxiv.org/pdf/1502.02677
- **[P4] Symmetry-Compressed Double Factorization**, arXiv:2403.03502v1。対称性とDFの資源目的に関する関連節。https://arxiv.org/html/2403.03502v1
- **[P5] Cugini, Atif, Subaşı, Resource-Optimal Importance Sampling for Randomized Quantum Algorithms**, arXiv:2603.13495v1。固定protocolのweight、測定second moment、正価格のcost-aware ISを確認。https://arxiv.org/html/2603.13495v1

### 今回の実行・保全

新規科学call：0。保存値の新数値再解析：0。Repository write：0。本レビューMarkdownのみを新しく作成した。過去のSOURCE、raw、契約、grant、ledger、marker、STOPの状態を変更していない。
