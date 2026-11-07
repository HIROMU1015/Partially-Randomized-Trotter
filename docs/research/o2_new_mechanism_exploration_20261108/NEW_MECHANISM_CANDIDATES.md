# O2再探索：coherent label消去の失敗を使う早期終了

2026-11-08 JST／SERVER1。**次の限定技術確認へ提案する操作案を1つ残した。** 案Eは、正確率labelの小さいpoolをcoherentに実装し、label消去の失敗を観測したattemptではX/Y信号に寄与しないsuffixを実行しない方法である。独立の学術的新規性や総資源優位は未認定。LCUを使う既知法の応用候補として扱い、案M/Dの追加検証は保留する。

## 1. 既存成果から引き継ぐ制約

[O2統合][SYNTH]の39 stageを再調査せず、次の障害を再利用した。S2は単価の低下が十分shot数で相殺され、SCV224/448は当該controlの学習費込み改善を支持せず、A2は具体的差分LCUのscale/成功率補正後にnoise床を残した。J1は安い全path価格normalizerを得ていない。Closed/CTS・ParentやD2は強い既存対照であり、再命名しない。

直近の[案M結果][MRESULT]は、保存linear H4／1.00 Å／STO-3G／legacy DF12、8 system qubits、L_D=3、m0/q1、δ=.02、r1/K2、coefficient_atol=0、N1 actual finite-grid Q・selected full policyのn=0唯一rotation slotに限定される。正232 labelは232単独群、非自明群0、A=B†はliteral判定でUNKNOWN。今回、この既報値を引用するだけで、raw/headerの新しい分割・再集計はしない。M一般のNO-GOへ広げないが、ユーザー指定どおり研究主線の追加検証を保留する。Dも保留を維持する。

以下は新しい設計提案の**記号的導出**である。実科学入力、量子simulation、回路build/compile、RNG、既存bit再解析、人工代数testは実行していない。

## 2. 案Eの入力と新しい操作

### 保存するtarget

\[
 Z=\mathbb E_Q\langle\psi|U_\omega|\psi\rangle
\]

という有限coherent平均を維持する。channelや|Z|²に変更しない。pathを外側context η、事前に固定したdisjoint pool g、pool内label jへ分解し、

\[
 U_{\eta j}=A_\eta W_{\eta j}B_\eta,\quad
 Q(\eta,g,j)=\pi_{\eta g}p_{j|\eta g},\quad \sum_{j\in g}p_j=1
\]

とする。scalar/event phaseはWまたはA/Bのcoherent phaseへ含める。全正supportを保持し、pool外のlabelもsingleton等で元lawに残す。πとpは**actual finite-grid Q**の正確な質量から得る。理想確率へ変更しない。

Mのような共通V・同時対角化を要求しない代わりに、**異なるW_jをlabel registerで制御する新しい有料SELECT操作**を要求する。W_jはそれぞれ元selected演算のままで、basis条件を緩めたり元SOURCEを変更したりしない。最初の技術対象は人工の2-label pool一つだけとし、実H4でのpool選択や別slot探索を今回行わない。

### 利用可能な情報

| 使える入力 | 使えない無料情報・新しい費用 |
|---|---|
| actual label/order質量、既知のphase/sign、元のgate記述 | 未知state振幅・Born・path信号・真variance・真overlapを使わない |
| 外側Qの取得、共通prefix/suffixの記述 | whole-path価格表や最適pool oracleを仮定しない |
| 追加registerのPREP/SELECT/逆PREPを明示的に作る能力 | coherent sampler/qRAMを無料の既存接口としない |
| 有料の消去flagとX/Y測定bit | p_successを事前に知る必要はない。学習・校正に使うならその測定費を払う |

### 一attemptの操作

signal ancilla a、pool register kを0にし、systemにfreshな|ψ⟩を供給する。PREPはP_g|0⟩=Σ_j√p_j|j⟩。

```text
外側 (η,g) をπから取得する。j自体は古典取得しない。
aへH、a=1で共通prefix Bを実行。
kへPREP、a=1かつk=jでW_jを実行、kへPREP†。
kを測定する。
  k≠0: suffix Aとancilla readoutを省き、出力0。retryしない。
  k=0: a=1でsuffix Aを実行し、aをXまたはYで読む。
        bit0→+1、bit1→−1を出力する。
全attemptを分母にして平均する。
```

2-labelならkは一つのlogical qubit、PREPは既知pによるRY、SELECTは明示W_0/W_1の二重制御で構成できる。これらを定数費用のblack boxとはしない。X/Yは別attempt、YはSdg→H。未知stateの逆準備・reflectionは必要ない。失敗した入力状態は消費済みで、prefix費を払い、次のattemptではfreshな準備が必要。

## 3. 平均・二次moment・失敗suffixの導出

\[
 K_r=\langle r|P_g^\dagger\operatorname{SELECT}(W_j)P_g|0\rangle,
 \quad S_g=K_0=\sum_jp_jW_j,\quad \sum_rK_r^\dagger K_r=I.
\]

pool消去後、suffix前の全状態は

\[
 \frac{|0\rangle_a|0\rangle_k|\psi\rangle
       +|1\rangle_a\sum_r|r\rangle_kK_rB|\psi\rangle}{\sqrt2}.
\]

k≠0のbranchではaは|1⟩だけなので、systemへどんな共通unitary suffix Aを適用しても、X/Yの非対角期待値は0である。従ってそのbranchのAとreadoutを省略して0を返してよい。これは未知のpath平均を推測して捨てる操作ではなく、**instrumentの構造上、寄与0と確認できる結果に基づく省略**である。

成功branchにAを適用した非正規化状態は(|0⟩|ψ⟩+|1⟩AS_gB|ψ⟩)/√2である。したがって、

\[
 \mu_X+i\mu_Y=\langle\psi|AS_gB|\psi\rangle,
 \qquad p_s=\frac{1+\|S_gB|\psi\rangle\|^2}{2},\quad \tfrac12\le p_s\le1.
\]

出力Y_aは成功時±1・失敗時0なので

\[
 \mathbb E[Y_a|\eta,g]=\mu_a,
 \quad\mathbb E[Y_a^2|\eta,g]=p_s,
 \quad\operatorname{Var}(Y_a|\eta,g)=p_s-\mu_a^2.
\]

成功数で割らず、1/p_sも掛けない。成功branchだけの平均ならμ_a/p_sとなりtargetが変わる。**ここでの分母は初めから全attempt数**であり、確率を0と仮定したscale会計ではない。

2-labelでp_0=p、p_1=1−pなら

\[
 1-p_s=p(1-p)\bigl[1-\operatorname{Re}
 \langle B\psi|W_0^\dagger W_1|B\psi\rangle\bigr]
 =\frac{p(1-p)}2\|(W_0-W_1)B\psi\|^2.
\]

このoverlapは解析にだけ現れ、algorithmの入力やpool最適化へ渡さない。W_0=W_1なら消去は常に成功し、追加費だけになる。正の失敗確率なら、同じ(η,g)を普通に混合して測る±1 Hadamardに比べM2を1−p_sだけ下げる。M2の絶対減少は高々1/2であり、noiseをゼロ化する構成ではない。

さらにCauchy–Schwarzからμ_a²≤||S_gBψ||²=2p_s−1。通常の混合測定のvarianceをV_B=1−μ_a²、案EをV_E=p_s−μ_a²とすると

\[
 2V_E-V_B=\|S_gB\psi\|^2-\mu_a^2\ge0,
 \qquad \tfrac12 V_B\le V_E\le V_B.
\]

従ってこの構成の一attempt当たりvariance利益も高々factor 2である。外側を同じlawで混合してもJensenにより同じ不等式を得る。これは通常±1 mixtureとの関係であり、最適層別/直接測定へのfactor保証ではない。

### 小角度の制約

同じφでW_j=exp(−iφP_j)、P_j²=Iなら、

\[
 1-p_s=p(1-p)\sin^2\phi\,
 [1-\operatorname{Re}\langle B\psi|P_0P_1|B\psi\rangle].
\]

短いRTE rotationだけをpoolにすると統計的利益もsuffix省略の頻度も小角度の二次に小さくなる。この式を根拠に実H4の値を計算したり、別orderで救済したりしていない。

## 4. 全費用へ利益が残る条件

固定(η,g)で、C_preを共通prefix、C_EをPREP/SELECT/逆PREP/flagの新費用、C_sufをA＋ancilla readoutとする。primitiveの加法が正しいと証明した値ではなく、実装scopeを説明する記号である。

\[
 \overline C_{\mathrm{early}}=C_{\mathrm{pre}}+C_E+p_sC_{\mathrm{suf}}.
\]

同じcoherent poolを最後まで実行してからflagを処理する方式に対しては、M2を変えず(1−p_s)C_sufを省ける。**この省略は一般の既知LCU実装にも認める強いbaseline**であり、それを案E独自の優位と数えない。

古典的label取得の通常PRに対しては、元W_j一つを実行する費用を新SELECT等の費用が上回り得る。もし同じ費用座標で新しい期待attempt費が元の2倍以上なら、上のvariance boundだけでcost×varianceの改善はなく、追加校正費を払う前から不利になり得る。外側η,gが変わる場合、E[p_s C_suf]をE[p_s]E[C_suf]へ置き換えず、費用と成功確率の相関を保持する。価格oracleなしで最適pairを選べるとはしない。

同じtotal attempt数でのvariance低下だけでは総費用改善を示さない。Neyman/cost-awareなlabel層別も強い対照である。例えば2 labelの有料単独bitの平均μ_j、σ_j²=1−μ_j²とすると、理想的な固定shot配分のvariance係数は[pσ_0+(1−p)σ_1]²。案Eのp_s−μ²より小さい場合もある。費用予算で配分する理想対照は[Σ_jp_jσ_j√C_j]²を持ち、案Eが常に勝つとは言えない。実装可能な有限pilotの費用も両側に戻す。

主資源を一つ固定した場合の**診断用条件**は、同targetのN_E \overline C_Eが、既知直接測定/grouping・強い層別・既知coherent方式のN_b \overline C_bより小さいこと。cost×varianceだけは漸近的診断であり、有限confidenceの十分Nや失敗率の証明ではない。T、CX、RZ、depth、古典費を根拠なく足さない。

## 5. 新しい情報取得と残る負担

出力の二乗Y_a²は成功flagそのもの。必要なら、独立のflag校正を**suffixとancilla readoutなしのprefixまで**で取得してM2上界に接続できる。ただし新しい有料量子データであり、校正数・confidence・独立streamの採択と費用が必要。学習係数や無料ν/真varianceは要らないが、既存±1 facade/固定N式を無変更で流用しない。rangeは[-1,1]のままなので、Hoeffdingだけではshot削減が現れない場合がある。

相殺要因はcoherent PREP精度、二重制御、basisを含む全W_jの実行、flag測定/feedback、動的suffix、古典pool生成、全attemptのfresh状態準備、校正・compile取得費、whole-wrapper最適化損失。全pathをcoherentに準備する巨大register/qRAMを作る案ではないが、小poolでも無料にはならない。

actual Qのpが有理数でも√pは一般に有限gateでexactに作れない。理想式はexact PREPでのtarget保存を示すもので、実装ではPREP/逆PREP・phase・測定grid・weight/decoder・gateの新しい一様誤差が要る。labelを消去するため、測定後に古典importance weightで個々のp誤差を自由補正できない。PREPの実効pと元pの差を拘束しなければtarget biasが残る。失敗branchへinactive側振幅が漏れる数値/物理誤差も省略biasへ寄与する。旧A1 exact engine/Mの誤差資格は継承しない。

## 6. 先行研究・過去O2との差

正係数unitaryのcoherentな加算は既知であり、Childs–WiebeのLemma 2はその回路と失敗確率を扱う。[一次資料](https://arxiv.org/html/1202.5822)。randomized/composite LCUのinstrument・linear estimatorも既知である。[Sun–Zeng](https://arxiv.org/html/2506.15658v1)。従ってPREP/SELECT/flagを新発明としない。Hadamardの追加測定情報/anti-control利用も既に研究されている。[Faehrmann–Eisert–Kueng](https://arxiv.org/html/2505.15913v1)

今回のrepositoryに対する新操作は、**正確率の小poolをcoherent化し、signalに寄与0のflagを共通suffix前に測り、動的な有料suffix省略とprefix-only M2校正へつなぐこと**。統計と費用の取得を同じflagに結ぶ設計である。既知LCUが同じ早期終了を使えば同値となる可能性があるため、独立新規性/既知法への優越はUNKNOWNと明記する。

旧A2のfine−coarse augmented-stateを再使用せず、対象は正のp_jの平均。Σp_j=1なので旧A2のβ² scaleを掛ける必要はない。この点は数式で区別できるが、p_s≥1/2のnoise床と追加量子費が残る。Mの共通V grouping条件を救済したのでもなく、新しいcoherent labelアクセスを有料で導入する別instrumentである。SCVの特徴量追加・Parentのq最適化・Closed/D2の古典事前zeroとは異なる。

**技術分類：KNOWN_METHOD_APPLICATION + ACTIONABLE_LIMITED_TECHNICAL_CANDIDATE。** 確認可能なinstrument/全attempt会計を持つが、科学的GO/総資源改善の支持はまだない。

## 7. 別方向の検討：Gaussian product-wordの圧縮

number-conserving Gaussian basis Sで共役されたsigned Z/ZZ product P_lは、scalarを分ければsingle-particleのreflectionをFock空間へ持ち上げたものになる。ここではJWTのmode/wire対応を含めてGaussianであることを仮定し、number保存やgate名だけからこの資格を推測しない。第二量子化Γの群準同型から

\[
 P_{l_k}\cdots P_{l_1}=\Bigl(\prod_j\sigma_j\Bigr)\Gamma(T_{l_k}\cdots T_{l_1}).
\]

これをreturn出力に持たせれば、k個のbasis/product/basisを一つのGaussian変換として出力する操作が考えられる。入力は既知basis記述、出力は位相付きsingle-particle積。state/Bornは不要で、単なる隣接inverse相殺より広い。ただし一般のZZ rotationやsquared-DF evolutionはGaussianではなく、越えるなら新しい正当化が必要。

Gaussian分解自体は既知である。[Jiang et al.](https://arxiv.org/html/1711.05395v3)。既存`df_trotter.ops`もbasis/Givens構築を持つ。強いcompilerに同じGaussian fusionを与えれば量子回路が同じになる可能性があり、sampling law・normalizerや価格取得が新しく安くなることはまだ説明できない。連続matrix summaryだけではJ1が欠く価格部分和も得られない。

従って**KNOWN_METHOD_APPLICATION / REDUCES_TO_COMPILATION_OPTIMIZATION**として独立O2候補から外す。数値積・再合成・O3実験を開始せず、案Eの後の自動移行先にもしない。

[SYNTH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d28164b5d51e0ca5a0ade74ded8ff176b443dab8/docs/research/o2_research_results_integration_20261108/O2_RESEARCH_SUMMARY.md
[MRESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e98589ae129d3b81ad428c947d2f4c88623caa9b/docs/research/o2_m_n0_metadata_result_20261108/REPORT.md
