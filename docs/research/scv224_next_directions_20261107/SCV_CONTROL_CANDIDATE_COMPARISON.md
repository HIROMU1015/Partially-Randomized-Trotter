# 既存の層内control候補：何が異なり、何が未知か

2026-11-07 JST。**既存候補の静的比較。新特徴・実装・数値順位付け・正式採択なし。**
対象は[証拠整理の固定H4有限signal](SCV224_EVIDENCE_AND_LIMITATIONS.md)と同じ。
候補の定義・条件付きlaw・式は[84e021b6の既存候補調査 §2–7][candidates]を再掲し、224結果との違いを整理した。

## 共通の比較軸

層内で非定数、known conditional meanをexactに取得可能、path metadataから追加量子測定なしで得られることは入口条件。
有効性には実際のbitとの層内共分散、有限train係数、range、train/cal費、全targetへの重みが必要である。
構造上の期待理由を、実測済み相関・分散削減に置き換えない。

適用順はproducts→rotation。n=0ではL0がrotation、n=2ではL0,L1がproducts、L2がrotation。
ρ(l)はactual Qのlabel質量、a≠bは元5層を定義する固定label。理想係数比へ戻さない。

| 観点 | A：rotation fragment群indicator | B：signed support | C：product label一致 |
|---|---|---|---|
| 特徴 | 1{F(L_n)∈S} | s_(L_n)×support size/2 | 1{n=2,L0=L1} |
| 層内変化 | h0は定数。適格な群はh1で変化、自由rotationのh2–4でも群両側が正なら変化 | h0は定数。h1のa以外、h2–4の自由rotationに異なる特徴値が正質量なら変化 | h2・h3で変化。h0・h1・h4では0に固定 |
| 信号との構造的関係 | fragmentはDF回転基底・生成子族を区別し得る | 符号はsigned生成子、support sizeは演算子構造に関わる | 同一signed involutionのproductsがP_l²=Iになる |
| exact平均・二次moment | 一位置の有限和、indicatorなので二次moment=平均 | 一位置のfとf²の有限和 | label平方和または条件付き比、二次moment=平均 |
| feature取得の追加量子測定 | 不要 | 不要 | 不要 |
| 独立学習費 | 現Sの実行では有料trainを取得。別群は新しい検証対象 | 活動層のscalar trainが必要。現在の群係数を移植不可 | h2/h3へ係数を置くなら各活動層のconditional trainが必要 |
| 有界range | u∈[0,1]、residual box幅2+abs(λ) | u∈[−1,1]、保守的幅2+2abs(λ)。実support幅を用いるなら根拠が必要 | u∈[0,1]、幅2+abs(λ) |
| 全targetへの接続 | h1を対象にできるが相関・学習費回収は別 | h1を対象にし得る点はCと異なる。実変化・相関は未知 | 効く層は希少なh2/h3。p_h²/n_hの重みと有料trainを併せて見る |
| 確認済み範囲 | 固定S={fragment-3}は理論・人工・実H4 224取得。族全体は未評価 | 既存理論・SOURCE上の表現可能性のみ。新特徴の人工/実H4検証なし | 恒等関係・既知平均式の整理のみ。新特徴の人工/実H4検証なし |
| 主な未知 | 現群の相関・係数精度。他群が良いかも未知 | 実h1で非定数か、X/Yの相関、rangeと学習費 | 層内削減が全targetの十分N/総費用に現れるか |

## A：具体的な一群と特徴族を分ける

既報224のS={df-fragment-3}と、任意のSを持つ特徴族は同じ評価対象ではない。
元の候補調査にあるfragment-6は非定数性を示す例で、性能が良いと検証された代替群ではない。
今回のQ質量balance規則はbit相関を選んでおらず、他群の有利性も示していない。
この一群で小さい点差だったことから、全fragment群を否定も推奨もしない。

平均は元の式で、R_S=Σ_(F(l)∈S)ρ(l)としてν0=1{F(a)∈S}、
ν1=(R_S−ρ(a)1{F(a)∈S})/(1−ρ(a))、ν2=ν3=ν4=R_S。
既存`FragmentControl`はこの族専用であり、今回は群選択関数も平均関数も呼んでいない。

## B：符号とsupportを用いる別の説明変数

既存例は \(f(l)=s_l|S_l|/2\)、\(u_{ss}=f(L_n)\)。S_lは保存diagonal supportで、群Sとは別物。
固定SOURCEのI/Z/ZZ support（size≤2）とsign±1から[−1,1]で有界。h1で異なるf値が正質量なら非定数だが、labelが多いだけでは保証できない。

既存一位置lawの式を使えば、k=1,2に対し

\[
E[u_{ss}^k\mid0]=f(a)^k,\qquad
E[u_{ss}^k\mid1]=\frac{\sum_{l\ne a}\rho(l)f(l)^k}{1-\rho(a)},\qquad
E[u_{ss}^k\mid h]=\sum_l\rho(l)f(l)^k\quad(h=2,3,4).
\]

これらは既知式の記号的再掲で、実H4の平均・二次moment・非定数性を今回算出したものではない。
実数化した角度や未知state期待値を平均に混ぜない。

fragment IDによる群分けと異なり、符号とsupport sizeという生成子情報を使える。
ただしsupportは各DF基底内の構造で、基底変換・元H state・決定論ブロックに依存するHadamard信号をそれだけで予測するものではない。
X/Yとの相関の符号・大きさは未知。回路価格との相関があってもbit相関の証拠にはならない。
係数の大きさを用いる別案でも、Qやbit weightを二重に変更しない。
特徴のscaleで係数のscaleも変わるため、幅の式の係数だけでA/Bの優劣を決めず、actual frozen残差のrangeを比較する。

## C：productの完全label一致と希少層

既存式は

\[
\nu_{0,eq}=\nu_{1,eq}=\nu_{4,eq}=0,\quad
\nu_{2,eq}=\frac{\sum_{l\ne a}\rho(l)^2}{1-\rho(a)},\quad
\nu_{3,eq}=\frac{\rho(a)}{1-\rho(b)}.
\]

h2ではL0≠a、h3ではL0=aかつL1≠b、h4ではa,b固定で不一致。二次momentはindicator平均と同じ。
同じlabelなら同じbasis/signを含むP_lを使うのでP_l²=Iだが、rotation・決定論PF・event/scalar位相・Hadamard wrapperまで恒等になるとはいえない。
異なるDF基底の同じZ文字列だけを同じ演算子として扱わない。

既存lawではh2/h3は希少である。直接条件付き取得は可能でもtrain一件はfull価格であり、quantum棄却待ちがないことは学習無料を意味しない。
全target分散への寄与はΣp_h²τ_h²/n_hで評価される。強い構造的相関が仮にあっても全体RZ改善が大きいとは限らない。
一方で小さいp_hだけから一般的無用性を結論せず、quota・精度・rangeとの接続が必要。本書でその数値を計算しない。
h1専用の現driverではu_eqは定数0なので、この候補をそのまま差し替えて学習できない。

## その他の既存候補

| 既存候補 | 位置付けと障害 |
|---|---|
| productsのsign積 | h2/h3で変化し得る。条件付きsign平均の積でknown meanを表せる。h4は固定。sign/phaseをbitへ二重適用しない |
| basis群・局所操作数の加法proxy | 自由labelのlookup値が違えば非定数。一位置和/有限隣接項の二位置和でmeanを扱えるが、価格予測力とbit説明力は別 |
| 特定label・label組合せ | 既存LocalControlを再利用しやすいが、希少出現・rank・多数列の学習費が障害。全label one-hotへ拡張しない |
| Taylor次数・event phase・基本rotation角 | 元5層内で固定。新しい層内削減を生まない |
| full compiled RZそのもの | 安いexact条件付き平均が既知とはいえず、compile/平均取得費が必要。無料known-mean controlと扱わない |

Born確率、真のpath信号、state vector、真variance、測定bit/wordはcontrolへ無料で与えない。

## 再利用と最小差分（設計だけ）

版は[準備560c6092のSOURCE][prep]。既存`a1_n3_connection`のsampler/PathChoice/独立stream、bit-only facade、SavedPrices、journal/STOPは保つ。

| 候補 | 再利用できる核 | 将来必要な最小差分と人工確認 |
|---|---|---|
| Aの同じS | `FragmentControl/ConditionalBank/fit_cells/FrozenCV/paired_evaluate`、既存collector | 新規実行の目的・独立性・有限scopeだけを先に明確化。guard・旧STOPを変更しない |
| B | sign/supportの既存`RTEComponent`・`DFDiagonalPauliComponent`、一位置law/OLS/加重合成 | sign/supportをbindするconditional bank、f/f²のexact表、一般rangeを扱う薄いadapter。toy mean/moment・負値・range包含・中心化・rank・fresh/STOPを確認 |
| C | products末尾規約、既存joint mass/包含排除、indicator OLSと全5層合成 | 任意label一致の有限和、h2/h3活動層とcell別modelへの限定接続。toy一致/不一致・h0/1/4定数・位相・全費用・独立性を確認 |

`scv_scalar.Metadata`は現状label→fragmentを保持し、Bのsign/support bankではない。
`FrozenCV.output_range`は[0,1] indicator box専用。Bへそのまま流用すると不正なrangeになる。OLS数値coreを再利用できることと、既存`fit_cells`を変更なしで使えることは別。
Cにも現h1-only `fit_cells/Driver`を無条件流用しない。新しい一般framework・sampler/engine/価格器の再実装は不要。
有限confidence用のcal dispatch/明示family/plan adapterは候補差分とは別に未接続の部分がある。[既存再利用表][reuse]

**改善の数値順位は付けない。** Bはh1の別の演算子情報を問える候補、Cは恒等関係という別の構造を問える候補という違いまでが既存証拠から言える。
両者とも実H4相関・分散/RZ改善は未検証で、今回実装・人工実行もしない。

[candidates]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/84e021b6a8decde7fa3ed3fa44d7148cc5428061/docs/research/intrastratum_cv_design_20261107/INTRASTRATUM_CONTROL_CANDIDATES.md
[prep]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/docs/research/scv224_preparation_20261107/SCV_IMPLEMENTATION_REPORT.md
[reuse]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af47cf9161241c8710383ff8539a4a7f8f67492c/docs/research/scv_finite_confidence_20261107/SCV_REUSE_AND_VALIDATION_MAP.md
