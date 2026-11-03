# 研究C・C1 科学レビュー：H2・H4・H6、比較の公平性と次の最小検証

- 日付：2026-10-25
- 状態：重要科学レビュー完了。下記の原因仮説・一般的優位・独立新規性は、それぞれ明示した範囲で保留。
- 対象：通常Compressed Double Factorization（CDF）対Explicit Double Factorization（XDF）、決定論S2／RPEの条件付きinterrogation資源比較、およびH6のDF予算未達。
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 運用：`research_project_instructions_unified_2026-10-25.md`を実際に確認して適用。
- 本レビューで実施した作業：固定報告・契約・保存scalar表／optimizer metadata・実装SOURCEの読解、一次文献との比較、数学的検討。
- 実施していない作業：NPZ配列のdecode／再解析、再fit、分子Hamiltonian・operator・shots・総費用の新規数値計算、回路build／compile、GitHubへの変更。以下の結果数値は保存報告・CSV・metadataから引用したもので、独立再実行の結果ではない。

## 0. 結論

研究Cは継続する価値がある。ただし現時点の成果は、**固定された数値pilot・oracle・compiler・serial実行規則の下での、通常CDFの有用性を示す条件付きbenchmark**である。「新しいCDFアルゴリズム」「最適化されたXDF一般に対する優位」「状態準備込みの総量子資源優位」「厳密CA保証」は未成立である。

H2の約36.4%、H4の約27.1%というs=1の総RZ削減は、保存された比較条件のもとで保持できる。H6のSTOPも保持するが、通常CDFの表現能力の限界やH6での化学的精度達成不能とは解釈しない。[E1–E3]

次の優先順位は、別分子やrank探索を追加することではなく、次の二点である。

1. **ffsimのXDF初期値→実反対称生成子パラメータ→CDF初期値の往復が、元の分解を保持しているか確認する。** 固定SOURCEから、一般の直交行列で保持されない可能性を数学的に指摘できる。H6に実際に関与したかは未確認である。
2. **H4のXDF対照を共通DF予算で評価し、既知の隣接Gaussian相対回転融合を含むbaselineの適合性を確認する。** 同じP0を使ったことと、既知の強い合成法を使用したことは別である。

厳密な浮動小数点enclosureを全経路に整備することは、現在の最優先事項とはしない。通常の数値研究として必要な再現性・独立検証と、厳密certificateを主張するための保証は区別する。

## 1. 固定証拠と到達状態

### 1.1 主要commit

| 対象 | 固定結果commit | 科学SOURCE |
|---|---|---|
| H4決定論の精度・固定shots接続 | `6202f036457940532aeaa2873cd6d9e6f123c0ca` | `002d0dc2e1dafd8cb71735c43253fa054e675f6d` |
| H4 native cost-only／compile競合修正後 | `5d7faf7121cc281b7e2f38c2e04c224e594ae53d` | `9ce6703c89874c142c64ad934354bdff0db703d1` |
| H2／H6報告を含む固定結果 | `c7131bbbf6be10be3c1842344e4f2ad3a225a774` | `33c380d8aa792e5119bb3adec5dd07d1e1377214` |
| 固定ffsim 0.0.83 | `558d3517385778c733f1aa226bb61b8f30f4b0f3` | 同commitのPython SOURCE |

報告・CSV・契約・主要関数を実際に取得した。Codexの全raw byte照合は報告された来歴として扱い、本レビューで全QPY／NPZを独立に検算・照合したとは主張しない。

### 1.2 保存結果

共通の分子条件は直鎖、隣接距離1.00 Å、STO-3G、中性singlet。決定論ordered symmetric S2はs=1を主対象、s=4を対照、δ=0.1、m=0,…,12。共通精度目標は0.00159 Ha、DF予算はその1/8である。[E1–E3,S1]

| 項目 | H2 | H4 | H6 |
|---|---:|---:|---:|
| Nα=Nβ | 1 | 2 | 3 |
| sector次元 | 4 | 36 | 400 |
| 採択CDF rank | 2 | 8 | 未採択 |
| 採択XDF rank | 3 | 10 | 14 |
| CDFの物理sector誤差（guard前） | 1.3176231587e-6 Ha | 7.6660277614e-5 Ha | 最小観測4.6795466224e-4 Ha |
| XDFの物理sector誤差（guard前） | 4.8652059505e-16 Ha | 3.3751558355e-15 Ha | 1.0247385485e-4 Ha |
| 決定論4候補のCA数値pilot | PASS | PASS | 未評価 |
| s=1 CDF総RZ | 3,990,125,812 | 111,798,946,393 | 未評価 |
| s=1 XDF総RZ | 6,274,985,632 | 153,454,725,962 | 未評価 |
| s=1 CDF削減率 | 報告値36.412192% | 約27.1% | 未評価 |

H2とH4の数値energy評価値は、それぞれCDF/XDFで、H2が0.0012796349488257434／0.0012783173262851165 Ha、H4が0.0013549794824010532／0.0012783188045012185 Haである。**実際に推定されたenergyの真値からの誤差ではなく、DF誤差と登録RPE位相幅を接続した数値条件付き評価値**である。[E1,E2]

H6ではXDF prefix14がnorm+guard=0.00010361230763094324 Haで合格した。CDFの最小観測値はrequested18／actual18の0.00046940339495824489 Haで、CA/8=0.00019875 Haに届かなかった。CDF fitは21回すべてoptimizer successだが、PF・ground/alias・selected adapter・費用の検証前に停止した。[E3,E4]

## 2. 数理的な精度接続のレビュー

### 2.1 同じ物理targetと異なる近似Hamiltonian

固定STO-3G・指定sectorの元HamiltonianをH、arm aの近似HamiltonianをH_a、sector射影をPとする。

\[
d_a=\|P(H-H_a)P\|_2.
\]

両Hamiltonianが対象sectorを保存するなら、同sectorの最低固有値について

\[
|E_{0,a}-E_0|\le d_a
\]

が成り立つ。したがって、異なるDF精度のCDFとXDFを、**同じ最終許容誤差を満たすか**で比較すること自体は正当である。実DF誤差、rank、fit残差を完全一致させる必要はない。

ただしHは有限基底STO-3Gのモデルである。この比較はモデル内ground energyへのアルゴリズム精度であり、基底不完全性等を含む現実の分子energyの化学的精度を保証するものではない。

### 2.2 Own-DF ground-state oracle

各armにそのH_aの正確なsector ground state \(|\psi_a\rangle\) が与えられるなら、

\[
\langle\psi_a|e^{-itH_a}|\psi_a\rangle=e^{-itE_{0,a}}.
\]

この仮定により、PF・compiled回路の信号を単一固有位相へ結び付けられる。実装はHF survival signalをground-energy保証へ読み替えていない。[S4]

一方、\(|\psi_C\rangle\) と \(|\psi_X\rangle\) は別の状態であり、準備費用・成功確率・physical groundとの重なりが等しいとは限らない。小さいHamiltonian norm差だけからground-state overlapが大きいとはいえず、gap等の追加条件が必要である。よって結果は、**各armのoracleを外部入力として認めたinterrogation費用**の比較に限定する。状態準備費用が候補間で相殺されるとは主張しない。

### 2.3 PF・合成・統計・DFの単位分離

決定論ではattenuation A=1、RTE補正誤差0。実装はdimensionlessなPF operator誤差とcompiled信号誤差をまず

\[
e_m=\epsilon_{\rm PF,m}+\epsilon_{\rm syn,m}
\]

へ接続し、\(\rho_m\ge1-e_m\)、\(\beta_{{\rm det},m}\le\arcsin e_m\) を用いる。各X/Y軸の統計誤差ε_Nについて、

\[
\epsilon_N=\sqrt{2\log(2/\alpha_{\rm axis})/N},\qquad
\sqrt2\epsilon_N\le\rho_m\sin\beta_{\rm stat},
\]

\[
\arcsin e_m+\beta_{\rm stat}\le\beta_{\rm total}<\pi/3
\]

を全roundで要求する。分岐・alias前提とRPEの反復条件を満たせば、最終位相幅をenergyへ変換し、

\[
\epsilon_{{\rm energy},a}
\le d_a+\frac{\beta_{\rm total}}{\delta 2^{12}}
\]

と評価する。DFをdimensionlessな信号誤差へそのまま加えておらず、最後にHa単位で一度だけ計上している。[S4,S5]

この構成には数理的な筋が通っている。候補単位で26軸のfailure配分を合計する構成も整合する。ただし、4候補全部の同時成功や実機実験による成功確率の検証を行ったことにはならない。

### 2.4 合成誤差の意味

固定SOURCEは、kernelの差をFrobenius残差等で数値評価し、unitaryのtelescopingと期待値差の不等式から全roundへ伝播している。

\[
\|C^n-U^n\|_2\le n\|C-U\|_2,
\quad
\|A\|_2\le\|A\|_F,
\]

\[
d_X=\min(2,n\eta+\eta_{\rm head}+\eta_X),\quad
d_Y=\min(2,n\eta+\eta_{\rm head}+\eta_Y),
\quad e_{\rm syn}\le2\sqrt{d_X^2+d_Y^2}.
\]

max-entry toleranceをoperator normへ無断流用していない点は適切である。しかし代入した残差、CAR／expm／Operator再構成の数値誤差は厳密に囲われていない。したがって、解析的不等式の正しさと、代入値がrigorous enclosureであることは別問題である。[S5,E2]

現在のP0は連続角RZを基本gateに含む。Clifford+Tへの離散rotation synthesisやfault-tolerant物理実装費用は評価していない。総RZの削減を、そのままT-count・Toffoli-count・実機wall timeの削減と呼ばない。

### 2.5 数値pilotとしての利用

H2/H4の報告は、指定された数値referenceを用いた十分条件の成立を支持する。厳密enclosure未取得だけを理由に、この成果を無価値・全面未成立とは判断しない。

ただし主張は「数値的に検証した十分条件」「数値pilot条件付き」とする。厳密CA保証を研究の主題にする場合にはenclosureが必要だが、通常の数値benchmarkを整えるために全経路の区間演算を最初から作る必要はない。先にtight margin・DF選定境界・代表kernelを対象とする独立経路／高精度整合性確認を検討する。これは厳密証明とは別であり、実施には新たな科学実行許可が必要である。

## 3. 比較の公平性

### 3.1 共通許容誤差は適切な比較条件

CDFとXDFの実DF誤差が異なること自体は、不公平の証拠ではない。resource minimizationでは、同じ最終誤差目標を満たす表現・分割・時間刻みを比較するのが自然である。

一方、現在示されているのは固定設定 \(\widehat\theta_C,\widehat\theta_X\) の比較であり、

\[
G_C(\widehat\theta_C)<G_X(\widehat\theta_X)
\]

から、各方式の最適化された費用

\[
\min_{\theta_C:\epsilon_C\le\epsilon}G_C(\theta_C)
<\min_{\theta_X:\epsilon_X\le\epsilon}G_X(\theta_X)
\]

が従うわけではない。現在の数字を最適化XDF一般への優位と表現してはならない。

### 3.2 H4のXDFが適切な強さか

H4はCDF rank8に対し、ほぼ数値精度まで再構成する保存XDF rank10を用いる。共通DF予算内に入る、より少ない保存XDF prefixが存在するかは、今回取得したH4報告では確認されていない。[E2]

したがって「rank10が過剰」と断定はしないが、**XDFにも同じDF予算を使わせたとき、rank10が必要か**は優先して確認すべきである。適合する保存分解・既存prefix評価を再利用し、新しい広い分解探索より先に解決できる。

H2では、同じmodified-Cholesky系列のprefix2が予算未達、prefix3が合格と保存されている。その範囲ではXDF rank3の必要性が示されており、H4よりrank選定の対照が明確である。ただし単一ordered prefixが全XDFの最適解とは限らない。[E1]

### 3.3 分子間で異なる実験設計

H4のXDFはOpenFermion-eigh、H2/H6のXDFはffsim modified-Cholesky prefixである。H4はCDF rank8固定、H2/H6はfirst-passing ladderである。H4のPF誤差はfull-space、H2は保存sectorで評価し、H4は保存Nの再判定、H2はその分子のPF・合成からNを新たに算定している。[E1–E3,S1,S2]

それぞれの分子内の結果を直ちに無効にはしない。しかしH2→H4の削減率の違いを分子サイズ依存性と解釈したり、三分子を同一benchmark設計の系列として描いたりする前には、policyの統一または差の明示が必要である。

sector評価については、完全な一体項／密度fragmentがそのsectorを保存する場合に限る。この完全fragmentの制限は数理的に正当であり、非保存な基本gateの途中で射影する近似とは違う。固定SectorCARのSOURCEはこの区別を意図している。[S2]

### 3.4 既知Gaussian baselineの不足

固定`NativeBuilder`の`runs()`は、隣接するrotation配列が**完全一致**した場合にだけbasis往復を共有する。それ以外は、各runについてbasis inverse→中央diagonal→basisという独立往復を作る。[S6]

しかし、異なる隣接basisについても、number-conserving Gaussianの表現Γを用いれば

\[
\Gamma(V)^\dagger\Gamma(U)=\Gamma(V^\dagger U)
\]

のように相対回転として合成できる（左右の向きは採用した回路規約に従う）。これは同一basisの相殺より広い既知の最適化であり、CDF/Trotter文献で扱われる。[L1,L3]

したがって「同じP0」「literal/sharedが同費用」は、一般の相対Gaussian融合の効果を測り終えたことを意味しない。compilerがこの構造最適化を同等に実現するとの証拠も取得していない。改善量やCDF/XDF比がどう変わるかは未測定である。

既存ffsimのbasis持越しやプロジェクト内の相対rotation実装を先に調べ、適合すれば両armへ同じ規則で適用する最小比較が望ましい。終了済みのGaussian singleton選択やRectangular Givens探索を再開する提案ではない。

固定fenceでkernel間融合を禁じた費用も、そのscheduleの費用として正しい。ただし全長回路をglobal最適化した最小費用とは区別する。

## 4. H6停止の解釈と、新しく確認すべき初期値経路

### 4.1 未達の直接的意味

H6のSTOPは、登録されたCDF fit経路がCA/8を満たす出力を返さなかったことを示す。PF精度や総費用は未評価であり、H6のCDFが物理的・数学的に不可能という証拠ではない。[E3,E4]

XDFの各frameは一般CDFの特殊形（Jがrank-one）である。同じκ・scalar規約なら、合格したXDF rank14自体は一般CDFの表現族に含まれる。よって「CDFの表現自由度では予算達成不能」という説明は成り立たない。通常CDFの最適化がその解を見つける、または保持する保証は別問題である。[S3,L1]

また、requested18〜21がすべてactual18・同一exportであるため、requested値だけを増やした4点は異なるframe数を探索していない。`max_vecs`は上限であり、初期Choleskyがtolで停止すれば実frame数はそこで止まる。[E3,S3]

### 4.2 optimizer successとtolの意味

固定ffsimのCDF目的関数は

\[
F(J,U)=\tfrac12\|g-g_{\rm fit}(J,U)\|_F^2
\]

であり、物理sector operator normではない。H2/H6の呼出しにsolver optionsはなく、`tol=1e-8`は初期分解へ渡される。これをL-BFGS-Bの最終物理誤差許容値と解釈してはいけない。[S2,S3]

保存rank18の停止は目的関数の相対減少条件で、objectiveは4.2855875026842845e-8、nit159、nfev168である。SciPyのこの停止条件は関数値の改善が小さくなったかを判定し、sector誤差が閾値を満たしたかは判定しない。[E5,L4]

さらに、tensor Frobenius誤差が小さくなることとsector normが小さくなることは同じ順序関係ではない。fit前後の両指標を区別する必要がある。

### 4.3 SOURCE上の初期値round-trip問題

固定ffsimの流れは次のとおりである。[S3,S7,S8]

```text
modified-Cholesky XDF → 各因子のeighで実直交Uを取得
→ df_tensors_to_params(..., real=True)
→ logm(U)の上三角の実部だけをパラメータ化
→ 実反対称行列Aへ復元 → expm(A)
→ J/Uの同時L-BFGS-B最適化
```

プロジェクトのcompat bridgeはbatch logmを各2次元logmへ振り分けるだけで、この実部抽出は変更していない。

実反対称Aでは

\[
\det(e^A)=e^{\operatorname{tr}A}=1.
\]

一方、eighの固有ベクトル行列はdet=-1を取り得る。したがって、一般のO(n)行列をそのまま実反対称生成子の指数へ変換して、必ず同じUを戻せるわけではない。

**解析的な例（数値実行していない）：**

\[
U=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\quad U^2=I,\quad\det U=-1.
\]

固有値-1に対して+iπを選ぶ複素対数では

\[
\log U=i\pi\frac{I-U}{2},\quad\Re\log U=0.
\]

その実部だけから生成子を作れば、復元回転はIになる。一般のJ（例えば一つの回転densityのみを残すJ）について、元のUで作るdensity積とIで作るものは同じではない。このため、**初期XDFがparameter conversion前後で保存される保証は、現在の実装からは得られない**。

ただしこれはH6で実際に発生したと確認したものではない。保存Uのdet、変換前後のtensor、初期objectiveを本レビューでは数値評価していない。H6の根本原因と断定しない。

CDFでは列の符号変更U→UD（Dの対角が±1）はU_pi U_qiを変えず、density表現を保持する。これを利用したdet符号の整理や、実反対称logの適切な構成は候補だが、detだけで全branch問題が解決したとみなしてはならない。修正時には必ずtensor／Hamiltonianのround-tripを検証する。

### 4.4 成功したH2/H4への影響

この問題は初期化の忠実性とfit性能の問題である。最終出力J/UのHamiltonian再構成と回路同値性が独立に検証されているH2/H4の保存結果を、それだけで否定しない。一方、現在のfit性能を通常CDFの最良性能や標準実装一般の性能として扱うことは避ける。

### 4.5 次の原因切り分け

まず人工行列でcodecのround-tripを確認する。次に別の数値再解析許可を得た場合だけ、保存H6入力・初期XDFを使い、少なくとも次を同じ入力で比較する。

- parameter conversion前のtensor／sector誤差。
- conversion直後、optimizer開始前のtensor／sector誤差。
- 保存optimizer終了後のtensor／sector誤差。
- 実frame数、初期objective、最終objective、停止条件、取得可能なgradient情報。

conversionで悪化するなら最小互換修正、conversionで保持されるならsolver停止条件・目的関数とsector条件の不整合を調べる。この順序なら不要な21-rank再探索を避けられる。新fitは原因の絞り込み後に、限定条件を別途決定する。

## 5. 先行研究と新規性

### 5.1 既知である部分

Cohn–Motta–Parrish（2021）はCDF、Trotter時間発展、回路削減をQFDへ接続している。CDFを使ってtime-evolution回路を短くすること、そのために一般対称Jとorbital rotationを最適化すること自体は新規ではない。[L1]

OumarouらのRC-DFは、tensor残差の低減だけでは係数や測定負担が悪化し得る点を扱う。今回のPRにおける回路単価と測定負担の逆転は有用な局所観測だが、一般的な「圧縮率だけで資源は決まらない」という主張だけでは新規性にならない。RC-DFの測定variance／qubitization λを、本研究のPR λ_R・attenuationと同一視もしない。[L2]

Casaresらの2026年preprintは、SPRINT/GRADEを通じてHamiltonian factorizationとPF誤差・回路資源を共同検討している。したがって「表現とPFを一体に最適化する」という広い着想だけでも、独立新規性は確立しない。[L3]

ffsimの現在使用しているfitはJ/U同時最適化であり、CDF原論文やRC-DFで議論されるnested two-step solverそのものではない。この実装差はH6の解釈でも重要である。[S3,L1,L2]

### 5.2 現成果の研究上の価値

価値があるのは、表現誤差・controlled phase・PF・RPE shot処方・実compiler費用を同一の有限条件で接続し、**回路単価の改善が総費用改善へ移る場合／移らない場合を切り分けたこと**である。特にPRと決定論の結果を両方残した点は、rankやRZ単価だけで選ぶ危険を具体的に示す。

ただし、H2/H4二例とH6 STOPだけで、一般的な選択規則・漸近優位・新アルゴリズムを認定することはできない。現状は、再現可能なimplementation/benchmarkとしての貢献が先行している。

### 5.3 論文化へ進む中心課題の提案

中心課題の候補は、

> 固定されたenergy精度とfailure予算の下で、Hamiltonian表現の候補を、rankやtensor normではなく、PF誤差・信号減衰・実compiler費用を含むinterrogation資源で選択するための条件と実用的な判断規則は何か。

である。

これは方向性の提案であって、完成済みの新手法ではない。論文化を主張するには、強いbaselineに対して何を新しく判断・予測・最適化できるのかを具体化し、その規則の検証を事前に定める必要がある。単に既知CDFを別package経由で接続したことや、既知の相対Gaussian融合を追加したことを新アルゴリズムとは呼ばない。

## 6. 最小の追加検証と判断条件

| 優先 | 次の作業 | どの判断が変わるか | 担当・実行境界 |
|---|---|---|---|
| P0 | 固定ffsim／compatの初期値round-tripを静的・人工例で監査 | H6 STOPがfit初期化の問題を含むか、既存fitter性能の解釈 | Codex技術調査。保存分子配列の評価・新fitは別許可 |
| P1 | 保存H6のconversion前／直後／終了後を最小限照合 | codec修正、solver設定、目的関数・物理誤差不整合のどれを優先するか | 新規数値再解析として別許可。広いrank sweepは不要 |
| P1 | H4保存XDFを共通DF予算でprefix評価する設計を固定 | rank10対rank8の差が予算に対して妥当か | 既存prefix処理再利用。実評価は別許可 |
| P1 | 全frameの一般相対Gaussian融合を既存実装で適用可能か監査 | P0固定baselineから強い構造aware baselineへ進む必要性 | まずSOURCE・人工検証。終了済みsingleton研究の再開ではない |
| P2 | 必要な一組だけ、同じ精度・compiler規則で費用比較 | CDFの削減が強いbaselineでも残るか | 対象・条件を固定後、別科学許可 |
| P2 | tight margin・代表kernel・DF閾値近傍を独立経路／高精度で確認 | 数値pilotの頑健性、厳密enclosureの必要度 | 大規模保証基盤より限定再現を先行。別許可 |

baseline監査で削減が保たれるなら、数値benchmarkとしての主張を強化できる。削減が消える・大きく縮小するなら、旧結果を削除せず、compiler／対照選択依存の結果として整理し、研究中心課題を見直す。H6の初期化が正常なら、codec原因説を棄却し、必要なsolver/物理誤差の調査だけへ進む。

CA/8は設計上の予算であり自然法則ではないが、H6を通すために事後的に緩めない。変更する場合は残りのPF／統計／合成予算を含めて新しい比較契約として固定する。

PRの全law合成保証・費用supportの大規模資格付け、別結合距離、さらに大きな分子、H6のnative full-unitaryは、現時点の優先対象にしない。固定PR gridで不利だった点推定は保全し、一般的NO-GOにはしない。

## 7. 主張できる文章と避ける文章

### 現時点で使用可能な主張

> 直鎖H2およびH4、1.00 Å、STO-3Gに対し、各近似DF Hamiltonianのsector ground-state oracleと理想測定を仮定した。固定ordered S2、RPE誤差配分、native compilerおよびserial実行規則の下で、CDFと各登録XDF対照は共通energy許容値0.00159 Haの数値的十分条件を満たした。状態準備を除くinterrogation総RZ数は、s=1でそれぞれ約36.4%、27.1%少なかった。数値referenceの厳密enclosureは未取得であり、baselineの最適性、実機精度、状態準備込みの優位を主張しない。H6は登録CDF fit経路のDF予算未達により、PF・費用評価前に停止した。

### 現時点では使用しない主張

「CDFはXDFより一般に27%以上有利」「H6でCDFは不可能」「化学的精度を実機で保証した」「状態準備込みで量子計算全体を削減」「RZ数から同率のT費用削減を証明」「既知CDF適用自体が新アルゴリズム」。

## 8. 過去の進行判断からの更新

前の通常応答では、H6の予算配分・optimizer停止条件を主に挙げていた。今回、固定SOURCEを詳細に読んだことで、初期parameter conversionの忠実性を先に確認すべき具体的理由が加わった。

また、同じP0を使った公平性だけでは、一般相対Gaussian融合を含む強いbaselineになっているとはいえないことを明確にした。H4の約27%削減を否定したのではなく、その数字がどの対照と実行規則に依存するかを限定した。

rankや実DF誤差を完全一致させる必要はない。必要なのは共通の最終目標と適切な対照であり、既存数値pilotを厳密保証未取得だけで破棄することもしない。

## 9. 再開・保全

このレビューの提案は、新しい科学計算の実行許可でも、GitHubの科学契約の書換えでもない。次の技術作業・必要な数値再解析はユーザーの指示に従う。既存SOURCE・raw・結果・grant・ledger・marker・STOPを保持し、修正は別SOURCEへ限定する。

今後のCodex作業も既存コード・テスト・pipelineの再利用を優先する。資源に恣意的な低い上限を設けず、共有環境や他ユーザーへの干渉は禁止する。承認証明書・承認記録・新grant発行を要求しない。

## 参考資料・固定リンク

### 実行結果

- [E1] H2結果報告、固定commit `c7131bbbf6be10be3c1842344e4f2ad3a225a774`：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7131bbbf6be10be3c1842344e4f2ad3a225a774/docs/research/research_c_c1_h2_execution_20261025.md
- [E2] H4決定論精度報告、固定commit `6202f036457940532aeaa2873cd6d9e6f123c0ca`：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6202f036457940532aeaa2873cd6d9e6f123c0ca/docs/research/research_c_c1_stage3_deterministic_accuracy_execution_20261025.md
- [E3] H6停止報告、固定commit `c7131bbbf6be10be3c1842344e4f2ad3a225a774`：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7131bbbf6be10be3c1842344e4f2ad3a225a774/docs/research/research_c_c1_h6_execution_20261025.md
- [E4] H6保存rank評価CSV：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7131bbbf6be10be3c1842344e4f2ad3a225a774/artifacts/research_c_c1_h2_h6_science/2026-10-25/H6_rank_selection.csv
- [E5] H6 CDF18 optimizer metadata：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7131bbbf6be10be3c1842344e4f2ad3a225a774/artifacts/research_c_c1_h2_h6_science/2026-10-25/H6-run01/CDF_18.optimizer.json
- [E6] H4の52round保存CSV：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6202f036457940532aeaa2873cd6d9e6f123c0ca/artifacts/research_c_c1_stage3_deterministic_accuracy_science/2026-10-25/round_accuracy.csv
- [E7] H4の8候補cost-only報告：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5d7faf7121cc281b7e2f38c2e04c224e594ae53d/docs/research/research_c_c1_stage3_cost_compat_execution_20261025.md

### 契約・実装SOURCE

- [S1] H2/H6固定契約：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7131bbbf6be10be3c1842344e4f2ad3a225a774/docs/research/research_c_c1_h2_h6_contract_unapproved_20261025.json
- [S2] `c1_molecule_extension.py`、科学SOURCE `33c380d8aa792e5119bb3adec5dd07d1e1377214`。`factor`、`SectorCAR`、`rank_assessment`、`PF_rounds`、`native_H2`、`run`：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33c380d8aa792e5119bb3adec5dd07d1e1377214/src/trottertracks/c1_molecule_extension.py
- [S3] ffsim固定`double_factorized_decomposition.py`、`558d3517385778c733f1aa226bb61b8f30f4b0f3`。`modified_cholesky`、`_double_factorized_compressed`、`_make_compressed_value_and_grad`：https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/linalg/double_factorized_decomposition.py
- [S4] `c1_stage3_accuracy.py`、`assess_round`／`assess_energy`：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/002d0dc2e1dafd8cb71735c43253fa054e675f6d/src/trottertracks/c1_stage3_accuracy.py
- [S5] `c1_stage3_deterministic_accuracy.py`、`numerical_segment`／`round_synthesis`／`assess_fixed_round`：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/002d0dc2e1dafd8cb71735c43253fa054e675f6d/src/trottertracks/c1_stage3_deterministic_accuracy.py
- [S6] `c1_stage2_native.py`、`runs`／`NativeBuilder.sequence`。H2/H6 SOURCEでも同内容（Git blob `81d622a30384d07e2d9bf2bfb0cabb303473b8c3`）を確認：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33c380d8aa792e5119bb3adec5dd07d1e1377214/src/trottertracks/c1_stage2_native.py
- [S7] ffsim固定`linalg/util.py`、`unitaries_to_parameters`／`antihermitians_to_parameters`／`df_tensors_to_params`：https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/linalg/util.py
- [S8] `c1_ffsim_compat.py`、per-matrix logm bridge：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33c380d8aa792e5119bb3adec5dd07d1e1377214/src/trottertracks/c1_ffsim_compat.py

### 外部一次資料

- [L1] J. Cohn, M. Motta, R. M. Parrish, *Quantum Filter Diagonalization with Compressed Double-Factorized Hamiltonians*, PRX Quantum **2**, 040352 (2021), DOI 10.1103/PRXQuantum.2.040352。CDF定義、SO生成子、nested two-step、時間発展・回路削減を確認。https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040352 ; https://arxiv.org/abs/2104.08957
- [L2] O. Oumarou et al., *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*, Quantum **8**, 1371 (2024), DOI 10.22331/q-2024-06-13-1371。tensor残差と測定・λの違い、同時／two-step最適化を確認。https://arxiv.org/html/2212.07957v3
- [L3] P. A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741 (2026-06-29), preprint。SPRINT/GRADEと§II.1の一般隣接basis回転融合を確認。査読済みとの認定はしていない。https://arxiv.org/abs/2606.30741
- [L4] SciPy 1.14.1 official documentation, `minimize(method='L-BFGS-B')`、ftol／gtolの意味。https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.minimize-lbfgsb.html

以上の文献確認は中心的な近接先行研究との比較であり、網羅的な不存在証明・「世界初」の証明ではない。
