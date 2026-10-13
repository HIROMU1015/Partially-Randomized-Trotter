# Track A：H4・6距離×302精度点の科学的レビュー
## geometry／精度依存、比較対照、標本不確かさ、H6・H8との統合方針

- 作成日：2026年10月13日（日本時間）。
- 開始承認：利用者の「その方針でレビューを開始して」。
- 主対象：`HIROMU1015/Partially-Randomized-Trotter` のH4 geometry計算。
- 結果branch：`track-a-h4-geometry-results-20261013`。
- 結果commit：`7c4483378047b57f514c47dad98905e3f2a9a756`。レビュー時にremote HEADとの一致を確認。
- 実行SOURCE：`d0726e946f89f8c30a08d383fb8ff48b363629ec`。
- Dataset／seed identity SOURCE：`5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`。
- 固定REVIEW artifact：`81c78b366d2cf2c699bb31977f72ae9deab4f1d4`。
- レビュー状態：**現登録実験の科学的評価は完了。未実施の代替baselineとの勝敗は保留。**
- 実行認可：本レビューは新しいtrajectory、signal、compile、H8本格計算のgrantではない。既存STOP・結果・source・消費を保全する。

---

## 0. 結論と研究判断

**H4 geometry結果を、Track Aの主要な実証的成果の一つとして原稿へ統合することを推奨する。** 単一geometryで得た低費用点を、異なる6距離でも同じ登録候補集合・実装系列で評価したことに意味がある。特に、完全決定論B1だけでなく、より安い残差discard B0を含む登録決定論集合に対してもB2の低費用点が残った。[R1–R3]

ただし、今回支持されるのは、**固定DF12・二次PF・structured Gaussian Givens・ordinary controlled wrapper・状態準備P=0・保存biasに基づくshot会計という条件下での、geometry／要求精度依存の資源地図**である。「全ての決定論的PFよりPRが優れる」「1,812回独立に勝利した」「分子サイズに対する漸近優位を証明した」という主張にはならない。[R7–R10]

次の優先順位を採択する。

1. **新しいH4 geometry大量計算や一律trajectory増量は、現在の必須工程にしない。** 既存データの比較構造、費用要因、選抜・不確かさを原稿に接続する。
2. **比較対照の定義を明確にする。** 今回のgeometry系列にS4はない。さらにsourceではB0が未採用DFブロックのidentity項もdiscardしているため、既知scalarを残す安価な対照の影響を、まず保存値で診断する価値がある。これは新しいレビュー上の確認事項であり、現結果の誤りを認定したものではない。
3. **H8の採択済みtechnical preparationは継続する。** 今回のH4結果を理由に、H8のprefix、q/r/K、主評価、sourceを無断変更しない。本格実行は従来どおり別承認。
4. **論文はgeometry方向とサイズ／実装方向の二つの証拠群として構成する。** incompatibleなH4/H6/H8のcostを一本のsize-scaling curveに混ぜない。

主課題を厳密certificate開発に変更しない一方、baselineの不足を多数のデータ点で覆い隠さない。この二つを両立させる。

## 1. このレビューで確認したこと／していないこと

### 1.1 資料と確認範囲

結果報告、summary、公開結果索引・manifest、保存値projectionの参照、公開照合コード、runtimeのsignal・回路・入力・保存処理・authorization gateをGitHubの固定commitから取得した。H4固定anchor、H6独立確認、H8限定feasibilityの固定結果も確認した。[R1–R15]

本レビューの数値検算は、公開summaryの代表18点から転記したB・軸別bias・shot・G・SE・対照Gについて実施した。**36軸のshot数を独自の標準ライブラリ算術で再計算し、すべて一致した。** 費用比、削減率、G/N分解、報告SEによる記述的な幅を再計算した。RTE lawからの確率診断は、保存normalizationとsource式に基づくスカラー算術である。

全1,308候補×302精度点の独立再集計、74,784 raw record全件照合、全32 paired標本からのSE再計算、NPZからのHamiltonian再生成、実compile再現、独立CIは実施していない。公開大容量projectionはconnector経由で参照できたが、作業containerへの全bytesの取得はネットワーク／ツール経路の制約で完了していない。そのため、ここで「全395,016点を自ら再検証した」「公開fileのSHAを自ら全件検算した」とは主張しない。

全件coverage・raw照合・cache identityについては、Codexの保存監査を根拠として区別している。公開verifierは全395,016候補精度点と1,812点最小を再現するコードであることを静的に読んだが、このレビュー環境での実行PASSではない。[R2,R5,R6]

### 1.2 根拠の区分

本書の「報告値」は保存結果の値、「本レビューの算術」は報告されたスカラーから再計算した量、「解釈」はその費用式からの読み取り、「提案」は今後の比較設計である。外部文献はP1–P3として独立に区別する。未測定対照の勝敗、真の母平均、全geometry・全epsilonへの一般化は推測で埋めない。

## 2. 実験の実体と研究全体での位置付け

### 2.1 完了した範囲

| 項目 | 対象と範囲 |
|---|---|
| 系 | 直鎖H4、STO-3G、8 system qubits＋ancilla1 |
| 距離 | 0.70、0.80、0.90、1.10、1.40、1.60 Å |
| Hamiltonian | 各geometryの保存されたDF12 Hamiltonian。固定12fragment選択 |
| State | 保存DF Hamiltonianの物理sector36で数値的に得た基底state。厳密certificateではない |
| 時間 | T=0.8、q=1/2/4/8、delta=T/q |
| 候補 | 各距離218登録候補、全1,308候補 |
| 方法 | B0：残差discard、B1：full DF二次PF、B2：prefix＋finite RTE、B3：random-dominant endpoint |
| 費用標本 | B2/B3各32 paired trajectory、B0/B1各1 paired circuit |
| Wrapper | structured Gaussian Givens系列、ordinary controlled rotation、実部・虚部のfull measured wrapper |
| 指標 | 軸別解析shot×paired compiled RZ費用、state preparation P=0 |
| 精度 | complex signal ε=.005〜.1の302点。Haではない |
| 完了 | 74,784 logical wrappers、MAP_COMPLETE_STOP |

6距離は事前scopeで固定され、各距離に同じtemplate集合が適用される。科学的な比較対象を結果後に選んだ6点という記録ではない。[R15] ただし、乱数標本から抽出した分子ensembleではなく、設計された6つの入力である。geometry一般化の確率的な成功率として扱わない。

218候補とlogical wrapper数・標本数の関係から、1距離あたり24 deterministic候補＋194 stochastic候補に対応する。すなわち2×24＋2×32×194=12,464 wrappers、6距離で74,784となる。74,784を「一つのPR構成の7万標本」「量子backendの7万shots」と数えない。

### 2.2 以前の研究から増えた情報

旧H4の単一geometryや限定transferに対し、今回はgeometryとprecisionの二軸で、同じ登録候補内の競合を調べられる。H6の独立確認が「選抜した構成の費用標本が再現するか」を扱ったのに対し、H4 geometryは「入力と誤差予算が変わると、どの構成群が低費用となるか」を扱う。両者の価値は異なる。[R12,R13]

長時間・多数compileを実施したこと自体は新規性ではないが、以前は一条件しかなかった主張を、条件依存の地図へ進めた点は評価できる。

## 3. 主な結果と、何を主張できるか

### 3.1 代表3精度の距離別比較

削減率の分母は、各geometry／precisionで適格な**登録B0/B1を合わせた最小値**である。full PFだけを分母にした値ではない。表は公開18行の算術再計算である。[R2,R3]

| 距離 Å | ε=.1 削減率 | ε=.05 削減率 | ε=.005 削減率 | ε=.05の点最良B2 | ε=.005の点最良B2 |
|---:|---:|---:|---:|---|---|
| 0.70 | 16.20% | 27.86% | 59.39% | p3/q2/r4/K2 | p3/q4/r2/K2 |
| 0.80 | 25.68% | 34.12% | 59.48% | p3/q1/r4/K4 | p3/q4/r2/K4 |
| 0.90 | 29.63% | 42.21% | 60.36% | p3/q1/r4/K4 | p3/q4/r2/K4 |
| 1.10 | 30.46% | 42.69% | 58.39% | p3/q1/r4/K2 | p3/q2/r4/K2 |
| 1.40 | 23.17% | 35.62% | 58.07% | p3/q1/r8/K4 | p3/q2/r8/K2 |
| 1.60 | 17.43% | 32.08% | 58.40% | p3/q1/r8/K4 | p3/q2/r8/K2 |


ε=.05では決定論最小が全距離でPM1追加のB0 p5/q1だった。ε=.005ではp9/q2またはq4へ移る。ε=.1の長距離ではp4/q1が最小対照となる。従って、PRは「高価なfull PFだけに勝った」のではないが、決定論側もepsilonに応じて選び直した有限候補集合である。

302精度点×6距離の全1,812点で点最小familyがB2であることは保存監査に記録されている。[R2] 代表18点はすべてp3だが、このレビューでは全302点のprefix占有率を独立再集計していないため、p3が全1,812点で最良という主張には拡張しない。

### 3.2 強い結論と弱い結論を分ける

**支持される結論**：登録したS2/finite-RTEと同じ実装・shot規則の下で、部分的なランダム補完が、全決定論化・残差discard・random-dominant endpointを含む比較集合の低費用側に位置する経験的証拠が得られた。要求精度とgeometryに応じて費用比・q/r/Kの点順位が変化する。

**支持されない結論**：B2の特定r/Kが真の最適、全てのPF次数・全compilerでの優位、全geometry・連続epsilon区間での優位、分子サイズのべき則、化学精度の最終QPE費用、状態準備・誤り訂正込み費用。

## 4. 302精度点の意味：独立性と費用曲線

sourceのepsilon gridは、.005〜.1の301 log-spaced点に.05を加えた302点である。同じ候補について、保存したbias、normalization、paired回路費用標本を全点で使っている。[R5,R7]

候補xとaxis aについて、sourceの定義は

\[
 e=\epsilon/\sqrt2,\qquad
 h_{x,a}=e-b_{x,a},\qquad
 N_{x,a}=\left\lceil\frac{2 B_x^2\log80}{h_{x,a}^2}\right\rceil,
\]

\[
 \widehat G_x(\epsilon,0)
 =N_{x,c}\overline C_{x,c}+N_{x,s}\overline C_{x,s}.
\]

hが正でない候補は不適格であり、費用ゼロではない。εを変える操作は新しい回路実験ではなく、同じbiasと費用を用いた解析shot再会計である。

したがって302点の主な役割は、費用曲線の下側envelope、適格境界、構成選択が切り替わる領域を描くことにある。1,812点を独立Bernoulli試行としてp値や成功確率を計算するのは不適切である。近接epsilonのSEも同じ32本に依存する。

また、全gridでB2が勝つことは、各epsilonで選べるB2構成が存在するという結果であり、**一つの固定構成が全点で勝ったことではない**。候補選抜に用いたexact/model biasの取得負担も、ここでの量子Gには含まれない。未知入力で安価に最良設定を予測できる実用selectorを検証した結果とは別である。

連続epsilon区間での優位も、この離散gridの検査だけでは証明されない。原稿ではgrid内結果と明記する。必要に応じた境界表の作成は保存値解析でできるが、今回それを必須の新しい科学計算にしない。

## 5. なぜgeometry／precisionで費用が変わるか

### 5.1 回路費用だけ、normalizationだけでは説明できない

\[
 N_x=N_{x,c}+N_{x,s},\qquad
 C_x^{\rm eff}=\widehat G_x/N_x
\]

と置けばG=N C_effは恒等的に成立する。C_effはshot-weightedな有効1-shot費用で、両軸の単純平均と常に同じとは限らない。今回のgeometry sourceでもcosine/sineのRZが1異なる例があり、H6/H8の一部で軸指標が一致したことを流用しない。[R4,R8]

| 距離 Å | ε=.05の点最良 q/r/K | B | 総shot数 | 有効1-shot RZ費用 G/N | G（百万RZ） |
|---:|---|---:|---:|---:|---:|
| 0.70 | 2/4/2 | 1.026526 | 17,187 | 12,015.50 | 206.510 |
| 0.80 | 1/4/4 | 1.053371 | 28,102 | 6,096.92 | 171.336 |
| 0.90 | 1/4/4 | 1.054149 | 22,921 | 5,848.92 | 134.063 |
| 1.10 | 1/4/2 | 1.057285 | 19,252 | 5,864.70 | 112.907 |
| 1.40 | 1/8/4 | 1.032194 | 16,474 | 6,248.29 | 102.934 |
| 1.60 | 1/8/4 | 1.035498 | 15,981 | 6,087.06 | 97.277 |


0.70 Åではq2なので1-shot費用が約12,015、他の代表q1では約5,849〜6,248である。一方、0.70 Åの総shot数は0.80 Åより少ない。従って「短距離だからshot数が必ず多い」「rを増やせば費用は必ず増減する」という単一要因の説明はできない。

### 5.2 同じ構成で比較できる例：0.80→0.90 Å

ε=.05、p3/q1/r4/K4を固定した公開値では、

- G比：0.78245837（約21.75%低下）。
- 総shot数比：22,921/28,102=0.81563590（約18.44%低下）。
- 有効1-shot費用比：0.95932311（約4.07%低下）。
- B²比：1.00147761（約0.148%増加）。

Bはむしろ僅かに増えているのにshot数は減っている。cosine biasが0.01312847から0.00947307へ下がり、統計誤差へ使えるheadroomが広がったことが、費用低下の主要因として読み取れる。この分解は同じq/r/Kの保存値比較であり、geometry変化の分子物理的な因果機構を証明したものではない。[R2]

相関の強さ、DF tail spectrum、軌道frameの変化のどれがbiasを支配したかを、この表だけで断定しない。そこまで説明するには保存Hamiltonian・fragment・state診断との対応が必要であり、資料にない物理的説明を補わない。

### 5.3 絶対費用と相対利益は別

ε=.05のB2 Gは0.70→1.60 Åで減っているが、削減率は1.10 Å付近の42.69%から1.60 Åの32.08%へ小さくなる。B2が高価になったからではなく、対照B0もより安くなっているためである。原稿では絶対費用と比を両方表示すべきである。[R2,R3]

### 5.4 厳しい精度では、discardとランダム補完の競合が変わる

ε=.1／.05では安価なp4/p5 discardが強い対照である。ε=.005ではより多くのfragmentを保持するp9が対照最小となり、B2はp3を保ちながらq/rを調整した点が最小になる。保存値は「精度を厳しくしたとき、discardで残すHamiltonian部分と、random tailを補完する際の回路／測定負担の釣り合いが変わる」という解釈を支持する。[R2,R3,R7]

ただし、B0のerrorはdiscardとPF誤差の合成で、両者が打ち消す場合もある。q増加が総biasを単調に改善するとは限らない。sourceを読んだだけで全geometryのcommutatorや相関機構を解明したとは主張しない。

## 6. Monte Carlo不確かさ、選抜、希少次数

### 6.1 Paired axesは正しく扱われている

sourceは同一trajectoryのcosine/sineを組として保存し、

\[
 \widehat{\mathrm{SE}}^2(\widehat G)
 =\frac{1}{n}
 \begin{bmatrix}N_c&N_s\end{bmatrix}
 \widehat\Sigma_C
 \begin{bmatrix}N_c\\N_s\end{bmatrix}
\]

で評価する。n=32であり64ではない。candidate内の同じ数値回路が再出現した場合に実compileをreuseすること自体は、元32drawの重複度・重みを残せば標本を増減したことにならない。保存監査はこのcache-ownerとmetricsの対応を確認したと記録する。[R2,R5,R7]

### 6.2 B2同士の細かな順位と、登録決定論との差は別

代表18点では、報告されたG+2SEがいずれも登録決定論最小Gより小さいことを算術確認した。これは現在の標本分散の記述の範囲では大きな差があるという意味であり、未知tailや選抜後の同時信頼保証を除いた結果である。一方、近いB2候補は±2SE帯が重なっており、r/Kの厳密順位は主張できない。[R2]

本レビューの18点では最小の差でも報告SEの約23倍に相当するが、これを正規分布のz値、p値、母平均の有意差認証に変換しない。SE自体が未観測の高費用部分を反映しない可能性があるためである。

### 6.3 Geometryごとの再選抜は固定設定transferではない

各geometryで同じ候補集合を評価する設計には価値があるが、各入力で点最良を取り直している。従って「1 Åで決めた一つの設定が全6geometryに転移した」とは言えない。また、このgeometry系列の32本は費用地図の作成と最小値選抜に使われており、H6の探索8→別32という二段階と同じではない。[R7,R10,R13]

現在の点最良結果を捨てる必要はないが、最小値選抜による楽観性を留保する。過去に事前固定した共通設定があるなら、その設定の保存値を併記することは有用である。今から選んだ設定を事前固定と呼ばず、32本の事後分割を独立した新実験と称さない。

### 6.4 大量compileでも、一候補のrare event問題は残る

74,784 logical wrappersは多数のgeometry・候補に分配された数であり、各stochastic候補は32本である。多数のepsilon表示もrare-event観測を増やさない。

元sourceの有限次数lawは

\[
 w_k(\tau)=\frac{|\tau|^k}{k!}
 \sqrt{1+(\tau/(k+1))^2},\quad k=0,2,\ldots,K,
\]
\[
 B=\left(\sum_k w_k\right)^{qr},\qquad
 p_0=w_0/\sum_k w_k
\]

である。[R7] 保存Bとq/r/Kからこの単調関係を数値的に反転し、有限law上の確率だけを診断した。これは元のorder列を再生成したものでも、実際に何本がzeroだったかを数えたものでもない。転記されたBのbinary64丸めに依存する近似診断である。

| 距離 Å | 点最良 q/r/K（ε=.05） | 1 trajectoryが非ゼロ次数を含む推定確率 | 32本すべてzero次数となる推定確率 |
|---:|---|---:|---:|
| 0.70 | 2/4/2 | 1.300% | 65.79% |
| 0.80 | 1/4/4 | 2.564% | 43.55% |
| 0.90 | 1/4/4 | 2.600% | 43.04% |
| 1.10 | 1/4/2 | 2.743% | 41.06% |
| 1.40 | 1/8/4 | 1.571% | 60.24% |
| 1.60 | 1/8/4 | 1.729% | 57.24% |


この範囲では、32本すべてがzero次数であっても必ずしも異常ではない。実際のH4各候補のorder countは、今回取得した主要summaryでは確認できていない。H6 pilotの+0.00734%をH4の補正値や上界として使わない。

**次は新しいsamplingより、既存の保存event／count metadataを確認することが先である。** それでpopulation boundが得られるとは約束しない。未知費用への強い保証を論文の中心claimにしない限り、全rare supportを埋めるまで計算を続ける方針は採らない。

## 7. このレビューで新しく特定したbaseline上の論点

### 7.1 S4がgeometry地図に含まれていない

今回のB1は二次full PFである。新H4固定anchorやH6にはS4があるが、geometry・DF・回路構成が異なるため、その数値を今回の6geometryに追加したことにはならない。[R1,R8,R12,R13]

したがって、現結果は「登録S2主体の決定論集合に対する比較」である。先行研究に対して「最良の決定論PFを上回る」と主張するなら、同じinput/control/compilerでの強い対照が必要になる。

今すぐ6geometry全体・全PR標本を再計算することは要求しない。S4を補う必要がある場合も、既存PR費用を保全した独立の小さなdeterministic supplementを設計できる。例えば4q×6geometry×2axesのS4費用は48 primary wrappersという範囲になるが、signal評価・数値検査・source同一性を含む別契約と実行承認が必要であり、本レビューでその実行を認可していない。

### 7.2 B0はtailのidentity成分もdiscardしている

runtime sourceの`_prepare`を読むと、全方法でconstantは核反発から始まる。B2/B3では、tail各DFブロックのI係数をconstantに足し、非identity成分だけをrandom componentへ入れる。一方、B0ではこのtail定数を加えず、tail block全体を除く。[R7]

\[
 H=H_{\rm kept}+c_{\rm tail}I+V_{\rm tail}
\]

と書けば、現在のB0はH_keptのS2である。自然な別対照として、

\[
 U_{\rm B0+I}(T)=e^{-iTc_{\rm tail}}\,
 U_{\rm B0}(T)
\]

を考えられる。c_tailはHamiltonianの係数だけで決まり、真の基底energyに合わせる事後fitではない。これはrandom tailの非identity演算を導入せず、既知scalarを残す構成である。

**これは現B0の実装bugという判断ではない。** 現B0は「残差ブロックをdiscardする」という定義を実装している。ただし、安い別の決定論的対照が未評価なので、「最良の決定論手法」と一般化する前に比較範囲を点検する必要がある。

### 7.3 global phaseとして無視しない／費用ゼロと断定しない

Hadamard testではcontrolled枝のglobal phaseがancillaの相対位相となるため、既知scalarを無視してよいとは言えない。sourceもancillaのphase gateでconstantを保持する。[R8]

一方、scalar保持でbiasが必ず減るわけでもない。現在のdiscardとPF誤差の位相キャンセルを壊す可能性がある。また、既知phaseを入れたwrapperの実compiled費用を、未compileのまま旧値と全く同じ・ゼロ追加と認定しない。

保存された複素signalをz_Dとすると、候補となるsignalはz_D^I=e^{-iTc_tail}z_D。これから同じtargetに対するaxis bias・shot変化をまず診断できる。ただし公開projectionは主にbias絶対値とcostを保持し、signed complex signalやc_tailを十分提供しているとは限らない。絶対biasだけから回転後biasを復元しない。

### 7.4 推奨する最小の対応

Codexには既存rawの複素signal、targetのenergy/phase、保存済みtail scalar係数を取得・公開できるか確認させる。保存値だけで可能な範囲のphase保持対照のsignal/shot感度を作り、旧B0と区別する。新しい対角化・signal生成・実compileが必要なら、その取得案を別途提示する。

この結果が小さければ留保を短く記して原稿化を進められる。大きければ、B0+IやS4を含む限定的な対照強化を検討する。どちらになるかは現在未測定であり、PRの利益が消えるとも残るとも結論しない。

## 8. 数値誤差・reference・状態準備の範囲

### 8.1 H4 geometryとH6のallowanceを同じにしない

取得したgeometry sourceのshot式はh=ε/√2−saved_biasで、H6の別sourceで使う明示的uをさらに引くfieldはない。[R5,R7] saved_biasも、corrected signalとe^{-iET}の軸差の絶対値として保存される。[R10]

そのため、「H6と同じ経験的uを含めたaccuracy matching」とは記述できない。H4 geometryは、source中の数値gateを通った保存biasを条件とするshot会計である。厳密な数値certificateがないことは両者共通でも、具体的な余裕の使い方は異なる。

これは直ちに今回の大きな費用差が数値誤差で説明されるという意味ではない。しかしheadroomの小さな候補では、biasの僅かな変化がNを大きく変え得る。必要なのは、採用した数値政策と対象scopeを正しく記し、元の数値gateや保存診断との対応を示すことである。

### 8.2 Referenceの意味

入力sourceは固定12fragmentのDF Hamiltonianを構築し、sector36の数値eighからstateを選び、元Hに対する残差やgap、normを検査する。signal側のtargetは保存energy Eからe^{-iET}を計算している。[R9,R10]

Hermitian Hと正規化stateについて、r=||(H−E)ψ||なら

\[
 |\langle\psi|e^{-iHT}|\psi\rangle-e^{-iET}|\le |T|r
\]

がDuhamel積分から従う。ただし、保存residualの演算誤差・Hの数値Hermiticity・DFモデルと元分子Hamiltonianの差を一括保証するものではない。sourceのgateをinterval certificateへ格上げしない。

また、DF12の固定rankとgeometryに伴う係数変化を、tol-only方式のH6/H8と同じpolicyとはみなさない。ここでの標的は保存DF Hamiltonianで、DF近似誤差込みのchemical accuracy全体ではない。

### 8.3 P=0と古典計算費用

共通の準備費用Pを含めるとG_x(P)=G_x(0)+N_x Pであり、同じ状態準備を使うから常に相殺されるわけではない。総shot数が違うためである。P>0の交点は保存値から分析できるが、実際のstate preparation circuitの費用は今回未測定である。

74.53時間の累積wallやhash/cache高速化は古典的な取得費用であり、量子RZ-workの改善と別量である。73,018固有compile＋1,766合法reuse=74,784 logical wrappersという会計も、量子shot数そのものではない。[R1,R2,R11]

## 9. H4固定anchor、H6、H8への位置付け

| 証拠系列 | 新しく答える問い | 混ぜてはいけない解釈 |
|---|---|---|
| H4 6geometry | 同じS2主体の登録実装で、距離とprecisionを変えた資源地図 | S4込み全PF最適性、サイズscaling、独立1,812試験 |
| H4 1Å固定16 anchor | H6に近いcontrol／S2/S4対照下の固定構成比較 | 6geometryの全grid最良の代替、DF政策まで同一 |
| H6探索＋独立32＋s2 pilot | 新しいサイズの登録map・選抜後費用の再現・希少層補足 | H4 structured/ordinary系列との同一cost curve、厳密母平均保証 |
| H8限定20compile＋本格準備 | 16mode回路の実装可能性と今後のsize方向resource map | 既にH8のshot込み優位を実証、全rare費用を評価済み |

H4 geometryは、もはや単なる脚注ではなく、**geometry／precision方向の主要な証拠群**として使える。一方、H6/H8は、より強いPF対照を残した別native系列でサイズ方向の問いを扱う。原稿ではこれらを別panel・別表にし、共通指標と異なる実装条件を明示する。[R12–R14]

H4のp3/12、H6のp5/19、H8のp7/28を見て「約1/4が普遍最適」と推定しない。入力・探索範囲・選抜方法・DF順序が違い、候補gridの粗さも影響する。

H8の採択済みcore（rank28、prefix7/14/21、q1/2/4/8、r1/2/4、K2/4、B1 S2/S4、B0/B2、探索と独立確認）は維持する。H4長距離でr8が選ばれたことだけでH8 r8を自動追加しない。H8準備を止める根拠は現時点ではない。H4のphase保持対照の結果により追加baselineが有用と判明した場合も、別scopeとして検討し、既存coreへ無断挿入しない。

## 10. 先行研究・新規性・論文化の着地点

### 10.1 既に知られていること

GüntherらのPR原論文は、部分ランダム化を提案しただけでなく、単一ancilla位相推定の量子化学資源見積もりと水素鎖のサイズ依存を扱っている。従って、本研究を「最初の具体的PR資源評価」と位置付けない。[P1]

Simon–Loveは、対称Trotter制御化の任意回転費用を削減する既知構成を示している。geometry系列のordinary controlled実装だけで現代的な最良control全般を代表させない。[P2] Casaresらも、電子構造向けに最適化したPF枠組みを扱う。[P3] これらの数値を今回のtaskへ換算・移植したわけではなく、比較classがS2一実装に限られることの外部contextである。

外部確認は上記一次文献の書誌・要旨・対象scopeに限る。geometry×precisionの同一実験が先行研究の全付録・コードに存在しないことまで、今回の限定文献確認から証明しない。

### 10.2 本研究で狙う寄与

推奨する中心的な問いは、**既知PR法が、具体的なDF-native回路と有限時間signal taskに対して、どのようなgeometry／precision条件で資源を節約し、その利益を何が左右するか**である。

貢献は、入力・基底回路・control・近似bias・normalization・shot規則・費用標本・候補選抜を接続した実装資源研究として整理する。302点や7万回路の数ではなく、以下を明確にできることが重要である。

- B0、B1、B2、B3という異なる誤差／測定負担の比較を、同じtaskで行う。
- 要求精度によりdiscardの保持prefixとPRのq/rが変わることを示す。
- 同じPR設定で、回路短縮とshot低下の寄与を分ける。
- sample点最良と母平均最良、geometryごとの再最適化と固定transferを区別する。
- 実装を変えたときに主張の範囲も変わることを、H4/H6/H8で明示する。

### 10.3 原稿の推奨構成

1. 研究質問とcost／accuracy accountingの定義。
2. H4 geometry系列の条件・比較classと結果地図。
3. N、B、bias、C_effによるmechanism、B0対照の解釈。
4. finite-sampleとrare-order、数値政策、known scalarの感度。
5. 別native系列のH4固定anchor・H6・将来H8によるcase comparison。
6. 支持される主張、支持されない一般化、実装依存の限界。

H4 geometryだけを別論文に分けるより、**同じ資源研究の主要sectionとして統合する**方が現時点では自然である。単独論文化の可能性を否定はしないが、「6geometryすべてでB2」という結果だけを独立論文の十分条件とは判断しない。新規性の最終採択には、より強いbaselineを含む比較の範囲と、先行研究との差分を原稿で具体化する必要がある。

## 11. 次作業とGO／STOP

### 11.1 Codexへまとめて任せる作業

**作業単位：H4 geometry結果の機構・比較対照・不確かさの保存値整理と原稿統合。**

- 既存1,308候補から、全302点の構成切替と適格集合、B0/B1/B2/B3最小、N・C_eff分解を生成する。新しい科学的selectorとして採択せず、posthoc記述とする。
- B2内の近接候補と、登録決定論との差を分ける。探索／選抜に同じ32本を使ったことを明記する。既存の事前固定設定があれば、その保存値を独立に併記する。
- 元保存eventから次数countと未知・未保存の区分を抽出できるか確認する。H6の補正を使わない。追加samplingはしない。
- B0のtail scalar保持という限定的な別対照について、既存complex signal・target・係数metadataを受領する。保存値だけで可能なphase/shot感度を作成し、compiled costは未取得なら未取得として残す。
- 各系列のPF次数、control、Gaussian lowering、DF政策、数値allowance、precision・state-preparationを一枚の表で区別し、H6原稿へ統合する。現行H8準備のsourceを変更しない。
- source・raw・停止記録を保全し、必要な軽量資料だけを別branchで公開、remote参照・bytesを照合してSTOPする。

新しいSCF／DF／state、行列指数・回路作用を再計算するsignal評価、trajectory生成、transpile、retry、H8本格開始はこの保存値整理に含めない。保存complex signalへの既知位相の乗算は、元の結果を変更しない別表のスカラー感度として扱う。新たな取得が必要な数値は未取得と記録し、目的・対象・call予算とともに別提案へ出す。技術的な保存解析／テストのたびにGPTへ戻す必要はない。

### 11.2 H8への対応

**H8の本格resource-map技術準備は継続GO。** H4が終了したことで利用可能資源が増える可能性はあるが、CPU・RAMを自動的に再割当済みと扱わない。実行直前のfresh resource確認と別grantは必要。H4の新しい結果を入力にしてH8の最適候補を事後選別し直すこともしない。

### 11.3 次にGPTの科学的判断が必要になる条件

- phase保持／強いPF対照の診断で、現在の科学的結論を変え得る差が出た。
- 原稿の主張を、登録S2実装の経験的優位から、全PFや母平均保証へ広げる必要が生じた。
- H8の新しい結果により、geometry方向とsize方向の解釈・研究の着地点を統合する段階になった。

単に保存値整形・source変更・テストPASSが出たことを、新しい詳細レビューの条件にしない。

### 11.4 決定表

| 判断対象 | 結論 |
|---|---|
| H4 geometry成果の研究的価値 | 採用。geometry／precisionの主要実証sectionにする |
| 現登録S2/B0/B1/B2/B3集合の点比較 | 保存値による限定benchmarkとして支持 |
| 特定r/Kの母平均最適 | 未認定 |
| 全決定論PFより優位 | 未支持。S4／control／scalar保持対照を区別 |
| H4の一律追加sampling／追加geometry | 現時点で必須にしない |
| 保存値でのphase保持対照診断 | 優先。実compileや新signal生成は別承認 |
| H8準備 | 継続。既定coreの変更は不要 |
| H8本格実行／追加S4やpilot | 従来どおり別途実行承認 |

## 12. 前回から維持する判断と変更点

維持：PR資源研究を継続する、全supportの厳密上界を必須化しない、H6 samplingを自動増量しない、H8はresource-map主評価、lineageを混ぜない、Codex technical workをまとめる。

新しい証拠：H4の6geometryが完了し、geometry／precision全体での登録family点順位と、有力なdiscard対照に対する差が得られた。

今回の追加判断：geometry系列を原稿の主要証拠とする一方、source上のB0 scalar discard、ordinary control、S2-only比較を明示する。特にscalar保持は未評価の安い対照として保存値診断を優先する。これは過去の結果を撤回する判断でも、既に存在した文書を単に再レビューしたものでもなく、今回のsource／新結果から比較範囲を具体化した判断である。

---

## 付録A. 本レビューで実施した算術

同梱スクリプト`analyze_selected_points.py`は、固定summaryから手動転記した18点だけを使う。36軸のshot数一致、G比と削減率、NとC_effの分解、記述的2SE比較、Bからのfinite-law確率診断を行う。乱数、化学library、Qiskit、eigensolver、transpileは使わない。

全1,308候補の公開検算には、リポジトリの`verify_saved_result_projection.py`と、manifestに対応する全公開JSON/CSVが必要である。これは本書の18点検算とは別である。manifestに記載されたprojection SHAは`9c78ef7a2a7ddb6c79551491044444c2d2dd42b55b0cbb961bb4fa4d548c68832`だが、本レビューcontainerではその全bytesを取得していないため、自らの検算済みSHAとは表現しない。

### 結果の要約

- 代表点数：18。
- 軸別shot再計算：36／36一致。
- 代表18点の報告G+2SE<登録決定論最小G：18／18（記述的診断のみ）。
- 0.80→0.90 Å、同じp3/q1/r4/K4：G比0.78245837=N比0.81563590×C_eff比0.95932311。
- 全395,016候補精度点：本レビューでの独立再実行なし。
- 新科学signal／trajectory／compile：0。

## 付録B. 原稿で使用できる主張の例

> 直鎖H4の事前固定6距離について、DF12の二次prefix PFとcanonical finite RTEを用いた登録候補を比較した。保存された有限時間complex-signal biasに基づくshot会計とstructured Gaussian controlled-wrapperの費用標本を組み合わせると、ε=.005〜.1の302点すべてで、部分ランダム化familyに最小費用の点推定が存在した。代表ε=.05では、登録discard/full-PF決定論対照の最小値より約27.9〜42.7%低かった。要求精度を厳しくすると、対照の保持prefixとPRのq/rが変わり、費用差は回路長と統計的headroomの競合として説明できる。これらは状態準備を除外した有限標本・固定compilerの結果であり、強いPF／scalar保持対照、母平均の同時保証、最終energy-estimation資源まで確立するものではない。

この主張は、302点を独立実験と数えず、S4がないこと、P=0と保存bias会計、未知tailを明示する。表現上の強さを狙ってこれらを省かない。

## 付録C. 証拠索引

- **[R1]** [H4 geometry結果報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_geometry_results_20261013.md)
- **[R2]** [H4 summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/summary.json)
- **[R3]** [18点比較CSV](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/precision_comparison.csv)
- **[R4]** [全候補の保存値projection](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/saved_candidates_v1.json)
- **[R5]** [公開保存値の照合コード](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/verify_saved_result_projection.py)
- **[R6]** [公開artifact manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/artifact_manifest_v1.json)
- **[R7]** [signal・RTE law・費用会計source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0726e946f89f8c30a08d383fb8ff48b363629ec/src/trottertracks/resource_applicability/h4_geometry/signal.py)
- **[R8]** [structured Gaussian・controlled wrapper source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0726e946f89f8c30a08d383fb8ff48b363629ec/src/trottertracks/resource_applicability/h4_geometry/circuits.py)
- **[R9]** [H4入力・DF・state source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0726e946f89f8c30a08d383fb8ff48b363629ec/src/trottertracks/resource_applicability/h4_geometry/inputs.py)
- **[R10]** [signal保存・target・bias source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0726e946f89f8c30a08d383fb8ff48b363629ec/src/trottertracks/resource_applicability/h4_geometry/execution.py)
- **[R11]** [H4 resume02の来歴・STOP契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_production_resume02_20261011.md)
- **[R12]** [H4固定16 anchor補完](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/README.md)
- **[R13]** [H6独立32 trajectory結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/research/track_a_h6_independent_confirmation_result_v1.md)
- **[R14]** [H8限定20 compile完了・技術計画](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a55da5c2b012912c485184845aa73ca053ac00b9/docs/research/track_a_h8_fast_feasibility_result_v1.md)
- **[R15]** [H4事前固定contract/source gate](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0726e946f89f8c30a08d383fb8ff48b363629ec/src/trottertracks/resource_applicability/h4_geometry/gates.py)
- **[P1]** [Günther et al., Phase Estimation with Partially Randomized Time Evolution, PRX Quantum 7, 020332 (2026-05-19)](https://journals.aps.org/prxquantum/abstract/10.1103/ynxb-p2xq)
- **[P2]** [Simon and Love, Halving the arbitrary rotation cost of controlled, Trotterized time evolution, Phys. Rev. A 114, 032425 (2026-09-09)](https://journals.aps.org/pra/abstract/10.1103/h9gv-ml7w)
- **[P3]** [Casares et al., Theory and practice of Trotter product formulas for quantum chemistry, arXiv:2606.30741 (2026-06-29)](https://arxiv.org/abs/2606.30741)

外部文献の書誌・要旨は2026-10-13にオンライン確認。詳細内容が取得できないHTML版について、全本文を読んだとは主張していない。リポジトリ・先行研究の外部送信、push、投稿は実施していない。
