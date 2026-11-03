# PR-2：M2後の研究方針・RQ・新規性・着地点・次検証の再設計

作成日：2026-10-04  
対象リポジトリ：`HIROMU1015/Partially-Randomized-Trotter`  
証拠を固定するcommit：`b6e65c6123475add5e620ec1064f361378bead95`  
branch：`pr2-v4-s2-parallelization-20260928`  
位置付け：**研究再設計の提案。既存契約の上書き、M2の再分類、新規科学計算のauthorizationではない。**

## 0. 結論

次の主題として推奨するのは、**DF-prefix部分ランダム化を、残差を破棄する方法・全て決定論的に保持する方法と比較した、精度条件付きの実装資源・適用条件研究**である。

仮題：

> **DF-prefix Hamiltonian simulationにおける残差処理の資源境界**  
> ――同一信号精度・実装コスト・状態準備負担に基づく比較

英語仮題：

> **Accuracy- and implementation-dependent resource trade-offs in DF-prefix Hamiltonian simulation**

ただし、「fixed-q/proxyでは分からなかった部分ランダム化の優位性を、actual compileによって初めて発見した」という現在までの説明は、そのまま中心claimにしない。保存値を共通候補集合で読み直すと、この説明には候補rankの変更、qの変更、proxy選抜とcost modelの変更が混在している。

今回の再設計は以下の三点を中心にする。

1. **残す事実**：M1の登録候補内でintermediate partialが低コストであり、M2の固定5構成でも精度と競争力を維持した。
2. **修正する帰属**：q最適化・候補集合・コンパイル・統計費用を分け、何が方式選択を変えたかを必要以上に主張しない。
3. **次に確かめること**：既知の回路改善を認めたbaseline、近接するdiscard候補、必要精度の変更に対して、この利益の範囲と原因を説明できるか。

新しいPF、selector、factorization、importance samplerの開発は本研究の必須要件にしない。反対に、既存実装を用いた小系の成功率・改善率だけで独立した論文の新規性が確立したとも扱わない。

**直近の実行単位は、保存済みJSONとsourceだけを使う証拠帰属・資源機構の再解析である。H4/H6等の新しいsignal、trajectory、compileを直ちに開始する提案ではない。**

---

## 1. 情報の区分と今回の確認範囲

本書では次の区分を使う。

- **既存結果**：commit済みの報告・artifactに記録されている値と判定。
- **今回の再計算・導出**：既存報告の数値の比、共通5構成の直線交点、実装式からの代数的整理。新しいHamiltonian simulationではない。
- **提案**：今後の研究scope、baseline、検証順序、終了条件。未実行・未認可。
- **未確定**：実際の回路改善率、未評価rank・精度・系での順位、独立論文としての十分な貢献。

今回新しく実行していないもの：分子生成、ground-state solve、NPZ読込、PF/RTE signal計算、trajectory sampling、Qiskit compile、量子測定、全test再実行、リポジトリ書換え。

研究証拠はM1/M2のcommit済み報告・JSONの可視部分と関連sourceを照合した。raw checkpoint/cacheからの全件再集計や、128 source hash全件の独立再検算を今回行ったとは主張しない。数値表の再計算は、以下の報告・集計に明記された値を用いる。[R1–R6]

---

## 2. 既に支持されていること

### 2.1 M1：developmentでの有限候補集合内の資源比較

H4 1.00 Å、STO-3G、DF rank 12、T=0.8、Qiskit 1.3.0 opt1という条件で、206 accuracy-eligible候補を比較した。primary RZ-workの点最小は `B2-rank3-q1-r4-K2` で、約1.308×10^8。6指標のpoint Paretoには同候補と `B2-rank3-q1-r8-K2` が残った。[R1,R2]

これは登録された有限候補内の結果であり、全split、全q/r/K、全PF、全合成法に対する最適性ではない。

### 2.2 M2：無調整の固定構成transfer

H4 1.30 Åへdevelopmentで決めた5構成を移し、全構成がaccuracy適格となった。B2二構成が6指標point Paretoに残り、最小B2／最小endpointのprimary RZ-work比は0.5860898、engineering upper 2SEは0.5935841だった。[R3]

したがって、今回の `TRANSFER_SUPPORTED` は境界ぎりぎりの判定ではない。ただしこれは**固定された構成のtransfer**であり、held-outで各方式を再最適化した比較ではない。

### 2.3 成功したことと、未達成のこと

達成したのは、定義済みの複素信号taskに対する実装資源比較と固定構成の移送である。

未達成なのは、次のような広い主張である。

- 分子基底エネルギーを化学精度で求める全RPE/QPEの総資源優位。
- 古典的に解けない系に適用できる、truth-freeな候補選択・精度認証。
- rank 3、q=1、r=4等の一般的最適性。
- Clifford+T、物理量子ビット数、実機時間への優位性。
- 新しいpartial-randomization algorithmの発明。

---

## 3. 研究ストーリーの重要な修正

### 3.1 qを自由にしたことと、rank 3を主比較に入れたことを分ける

M1の同じ集計から、全splitを含むq=8部分集合と、qを自由にした集合を比べると次になる。[R2]

| 比較する集合 | B0最小 | B1 | B2最小 | B3最小 |
|---|---:|---:|---:|---:|
| M1、q=8のみ | 1.726785246×10^9（rank 9） | 2.238254358×10^9 | 6.840614794×10^8（rank 3） | 1.152790918×10^9 |
| M1、q∈{1,2,4,8} | 2.533793500×10^8（rank 6） | 3.725231280×10^8 | 1.307748967×10^8（rank 3） | 1.152790918×10^9 |

**rank 3のB2はq=8でも既に最小である。**

旧S2のprimaryはrank 6のB2とB3で、rank 3はcontrolだった。したがって、旧S2の「rank 6/B3」からM1の「rank 3/q1」への変化全体を、q変更だけの因果効果とはしない。

同じM1候補集合内でq自由度を戻した効果としては、B2最小costが約80.88%低下する。一方、B0最小は約85.33%、B1は約83.36%低下する。**q最適化は比較相手にも大きく効く。**

B2/B1比はq=8の0.3056からmatched-qの0.3511へ、B2/B0比は0.3961から0.5161へ変わる。したがって、q最適化が常にpartialの相対的優位を拡大したわけではない。

今後の記述：

> 登録候補内でpartial領域は固定qでも残った。qを方式ごとに選ぶと絶対資源とendpointに対する相対利得が変わるため、固定qの改善率をそのまま手法の利益と解釈してはいけない。

これが既存データで支持される、より正確な説明である。

### 3.2 「proxyでは重要な候補を選べない」は強すぎる

M1の旧16-cell selectorは、actual Paretoの2件中1件を落とした。しかし、primary RZの点最小は保持しており、RZ regretは0。落とした候補によるsecondary metricの最小値regretは約0.68〜2.26%だった。[R1,R2]

また、64候補のproxy frontier自体はactual Paretoの2件とも含んでいた。**proxyのPareto集合と、16件に圧縮する選抜手順は別である。**

したがって、現在言えるのは「旧selectorは6指標frontierの完全保持を保証しなかった」であり、「actual compileを使わないとprimary方式選択を大きく誤る」ではない。

10% materialityを超える主判定の誤選択がこのselector比較で示されたとは扱わない。direct compileは資源評価の信頼性を支えるが、その実施自体を新規性にしない。

### 3.3 状態準備費用のM1/M2比較も、候補集合をそろえる

M1の全候補集合ではP≥0のlower envelopeがB2のみだった。一方、M2の固定5構成では大きいPでB1へ移った。この二つを直接比べて「geometryにより構造が変化した」とは結論しない。比較候補数が違うためである。[R1,R3]

M1もM2と同じ5構成に限定し、G(P)=G(0)+NPを再計算すると、point lower envelopeは両方とも次の順になる。

| データと集合 | B2 r4→B2 r8の交点P | B2 r8→B1の交点P |
|---|---:|---:|
| M1、M2と共通の5構成 | 約1,796.42 | 約235,578.46 |
| M2、固定5構成 | 約385.08 | 約208,735.13 |

これは**今回の事後的な算術再計算**である。M1の全候補結果とM2の正式結果は書き換えない。

r4/r8差は小さく、上の交点も点推定に基づく。精密な閾値を実用保証として示さない。PはRZ相当/shotの感度パラメータであり、実際の状態準備costを測った値ではない。

### 3.4 現在のtaskは化学精度のエネルギー推定ではない

M1/M2で固定したのは、T=0.8における複素信号の絶対精度ε_complex=0.05と、各軸の誤差・失敗確率配分である。[R3,R5]

固有状態のunit-radius信号z=exp(-iET)で、正しい位相branchを既に選べる場合に限れば、|z_hat-z|≤ε<1から幾何学的に

\[
|\Delta\theta|\le\arcsin\epsilon,\qquad
|\Delta E|\le\frac{\arcsin\epsilon}{T}
\]

となる。ε=0.05、T=0.8では右辺は約0.06253 Haである。

これはactual energy errorが0.06253 Haだったという意味ではない。**現在の許容信号誤差だけでは化学精度の保証にならない**という区別である。alias、入力overlap、複数round、最終推定器は別途必要となる。

本研究を信号推定のresource studyとして閉じるなら、そのtaskを表題とabstractで明示する。エネルギー推定全体へ戻すなら、別の達成要件を追加する。

### 3.5 B0の誤差欄を純粋なPF誤差と解釈しない

M1の `_deterministic_signal_record` は、B0にもB1にも同じfull-H targetを用い、`outer_pf_bias_abs=abs(mean-exact_target)`を保存している。[R5]

B0ではmeanがtruncated HamiltonianのPF信号なので、この値にはdiscardとPFの両方が入る。**フィールド名だけから純粋なouter PF biasと解釈してはいけない。**

B0の「破棄誤差」と「破棄後HamiltonianのPF誤差」を分離するには、同じ入力状態に対するexact truncated evolutionの信号が別に必要である。保存値に存在しなければ、分解は未評価と記載する。full-targetに対する総誤差でaccuracyを判定したこと自体は、この命名問題によって無効にならない。

---

## 4. 先行研究との関係を再評価する

新規性は「構成要素を何個組み合わせたか」ではなく、既知研究が残した具体的な資源判断・適用条件に答えたかで評価する。

| 先行研究 | 既知として除外する内容 | 今回の候補との差分・注意 |
|---|---|---|
| Günther et al., PRX Quantum 7, 020332 (2026) | partial randomization、RTE、single-ancilla phase estimation、factorized Hamiltonian、truncationとの比較、資源最適化 | 最接近文献。Sec. VIIとApp. D.2、Eを具体的に比較する。「DFに適用」「discardと比較」だけでは差分にならない。 |
| Hagan–Wiebe, Composite Quantum Simulations | deterministic/random分割の誤差・費用の競合 | hybrid原理そのものは既知。今回の有限task・回路実装に関する検査と区別する。 |
| Casares et al., SPRINT/GRADE | factorization、norm階層、near-integrable PF、random remainder、error/costに基づく設計 | 既知の強い決定論・残差処理を無視して、単純PFだけとの比較を一般化しない。 |
| Oumarou et al., RC-DF | 圧縮・正則化DFによる資源削減 | prefix切断は新しいRC-DFではない。今回の第一研究では表現そのものを再最適化しない。 |
| Kanasugi et al. (2026) | single-ancilla QPE、partial randomization、UWC、化学系のphysical-resource評価 | 主たるPauli-LCU/architecture依存taskと今回のDF-native compiled-RZ taskは異なるが、「end-to-endに考えた」は新規性にできない。 |
| Cugini–Atif–Subaşı (2026) | 回路costとvarianceを共同最適化するimportance sampling | 本研究はcanonical samplingに限定する。分布まで含めてresource-optimalとは主張しない。 |
| Simon–Love (2025) | 対称Trotterの制御付き実装における任意角回転の削減 | baselineを強化する具体的な候補。任意角回転半減を総RZ半減と同一視せず、同じ実装指標で確認する。 |

特に元のPR論文は、Sec. VIIでfactorizationの残差をrandomizeする考えを述べ、App. D.2/Figs. 19–20でsmall-term truncationを検討している。[W1] 「discardを初めて対照に置く」ことも今回の新規性にはできない。

### 残る新規性候補

以下を**一つの実証的な問い**としてまとめることは可能である。

> 既知のDF-prefix残差処理について、同一信号精度、登録した有限candidate domain、明示した回路合成、shot規則の下で、どこまでpartialの利益が残るかを、費用成分と誤差成分に分けて説明する。固定設計の移送と、各条件での再最適化を混同せず、未使用条件でもその説明を検査する。

ただし、これが独立論文として十分かは、強いbaseline後に残る知見で決まる。現在の資料と今回照合した文献だけから、網羅的な先行研究不存在や採択可能性は保証しない。

**新しい定理がないから数値資源研究は成立しない、とは考えない。逆に、実装条件を細かくしただけで新規性が成立する、とも考えない。**

---

## 5. 再設計する主RQと副RQ

### 主RQ

> **固定DF Hamiltonianに対して、同じcoherent-signal精度を要求し、各方式の実装費用と測定費用を公平に数えたとき、残差の破棄・決定論保持・ランダム補完のどれが有利になるか。その境界を、誤差・normalization・deterministic work・random workから説明できるか。**

### RQ1：資源利得の成立範囲

登録したcandidate family内で、intermediate partialが最良endpointに対し低コストまたはmaterially competitiveとなるε_signal、P、splitの範囲はどこか。既知の回路合成改善を認めてもその範囲は残るか。

### RQ2：何が順位・利得を決めるか

qによるdeterministic work、R=qrによるRTE負担、discard/PF/finite-RTEのbias、回路合成のどれが支配するか。単なる相関ではなく、共通条件の比較や同じ演算の実装比較で寄与を分離できるか。

### RQ3：結論の移送と限界

developmentで固定した構成の精度・資源競争力は、未使用条件でどこまで維持されるか。固定構成のtransfer、局所的な再最適化、利用可能情報による予測を区別して報告できるか。

M2はRQ3の限定された一例を既に満たす。RQ1/RQ2の一般的な答えや、truth-free predictionまで満たしたわけではない。

---

## 6. 数理モデル：新しい発明ではなく、検証を整理する座標

### 6.1 対象と候補

\[
H=H_D(L_D)+H_R(L_D),\qquad
x=(\text{方式},L_D,q,r,K,\mathcal I).
\]

\(\mathcal I\) は明示した実装policyである。保存済みM1/M2のpolicyと新しく比較するpolicyを別identityにする。

参照信号は

\[
z_H(T)=\langle\psi|e^{-iHT}|\psi\rangle.
\]

現在のbenchmarkは、保存済みのHの固有状態を用いている。任意の近似入力状態を同じコードがそのまま扱うとは仮定しない。

finite-RTEの補正後信号を\(\nu_x=B_x\mathbb E[Y_x]\)とし、各軸aの系統biasを

\[
b_{x,a}=|\nu_{x,a}-z_{H,a}|,
\quad \epsilon_a=\epsilon/\sqrt2,
\quad s_{x,a}=\epsilon_a-b_{x,a}
\]

とする。s≤0ならその候補は不適格であり、Nを増やしてもこの受理規則では救済しない。

M1/M2のHoeffding型shot規則は

\[
N_{x,a}=\left\lceil\frac{2B_x^2}{s_{x,a}^2}
\log\frac{2}{\alpha_a}\right\rceil,
\qquad \alpha_a=0.025.
\]

この式が与えるのは当該規則で十分なshot数であって、全ての推定器に対する最小shot数ではない。また、biasをexact referenceから得る現在のbenchmarkと、未知の量子問題で同じbiasを得る実用法は区別する。

primary資源は

\[
G_x(\epsilon,P)=\sum_a N_{x,a}(\epsilon)
\left(\bar C_{x,a}^{\mathrm{RZ}}+P\right).
\]

RZは登録gate setにおけるcountであり、Toffoli/T countでもphysical runtimeでもない。\(N\times\mathrm{depth}\)は累積depth-workであり、単一回路の最大深さと区別する。

### 6.2 qとrを別々に眺めるより、qとR=qrに整理する

現在のsecond-order partial構成では各outer stepに一つのtail occurrenceがあり、各occurrenceをr個に等分する。固定Tなら

\[
\delta=T/q,\qquad
\tau=\lambda_R\delta/r=\lambda_RT/(qr).
\]

R=qrと書けば、同一rank、K、Tに対して

\[
B_{\mathrm{total}}=
\left[B_K(\lambda_RT/R)\right]^R.
\]

したがって、**qとrが異なってもRが同じならnormalizationは同じ**である。canonical distributionのTaylor-order平均も同じなので、random component作用数の期待値も同じになる。[R5]

一方で、deterministic fragmentの出現数は基本的に2qL_Dであり、PFのinterleavingとboundary fusionもqに依存する。よって同一Rは同一signal・同一compiled costを意味しない。

例：rank、K、Tを固定して

\[
(q,r)=(1,8),(2,4),(4,2),(8,1)
\]

を比較すれば、RTE normalizationを固定しながら、deterministic分割頻度と列構造の効果を調べられる。

この恒等式を新しい理論成果としては主張しない。既知RTE式と現実装から得られる、**因果を取り違えにくい再集計軸**である。

### 6.3 資源競合を説明する最小分解

\[
N_x=N_{x,\mathrm{Re}}+N_{x,\mathrm{Im}},\qquad
\bar C_x^{\mathrm{eff}}=
\frac{\sum_aN_{x,a}\bar C_{x,a}}{N_x}.
\]

これなら、軸別costが異なっても

\[
G_x(0)=N_x\bar C_x^{\mathrm{eff}},\qquad
\frac{G_x}{G_y}=
\frac{N_x}{N_y}\frac{\bar C_x^{\mathrm{eff}}}{\bar C_y^{\mathrm{eff}}}
\]

が正確に成立する。

さらにshot式の中で、B²と、残余統計予算\(\epsilon_a-b_{x,a}\)を分ける。biasとnormalizationを同時変更した比較を、一方だけの効果と呼ばない。ceilingを外した近似は説明用に限り、最終集計は整数Nを使う。

### 6.4 誤差分解

partialでは保存済みのexact-tail PF信号を\(z_{\mathrm{PF,exact-tail}}\)として、複素数のまま

\[
\nu_x-z_H=
(z_{\mathrm{PF,exact-tail}}-z_H)+
(\nu_x-z_{\mathrm{PF,exact-tail}})
\]

と分けられる。大きさだけでは、項間の相殺を判別できない。

discardでは

\[
z_{\mathrm{PF},D}-z_H=
(z_D-z_H)+(z_{\mathrm{PF},D}-z_D),
\quad z_D=\langle\psi|e^{-iH_DT}|\psi\rangle
\]

が必要である。z_Dが保存されていない場合、この分解は新規参照計算なしには行わない。

相殺は正当なtask誤差の一部であり、一律に禁止しない。ただし、相殺に依存する精度許容と、各誤差を独立に保証する受理規則は別の主張である。

---

## 7. M2から既に分解できる機構

r4のB2を基準とした、保存数値の算術分解は次のとおり。[R3]

| 比較相手 | shot比 N_B2/N_ref | 有効1-shot RZ比 | 総RZ比 |
|---|---:|---:|---:|
| B0 rank6 | 0.9831 | 0.5962 | 0.5861 |
| B1 rank12 | 1.1275 | 0.3115 | 0.3512 |
| B3 rank0 | 0.5889 | 0.1775 | 0.1045 |

B1に対しては、B2は約12.8%多いshotを必要とするが、1-shot RZが約31.2%なので総資源が下がる。**shot数の少なさで勝ったのではない。**

B0に対してはshot数がほぼ同程度で、利得は主に1-shot circuit costから来る。

B3に対しては、1-shot costとshot数の両方がB2側で小さい。ただしB3はone-bodyを決定論的に保持するrandom-dominant endpointであり、H全体を完全random化したものではない。

この分解は既存値で確認できる。しかし、その1-shot cost差のうち、不可避なfragment数の差と、合成policyで改善できる差がどの程度かはまだ別問題である。

---

## 8. 研究の着地点

### 8.1 最小の完結点：限定されたresource/application note

次を満たせば、H4の限定studyとして閉じる。

- task、候補集合、実装、shot規則、入力状態の範囲が一貫している。
- fixed-q、rank集合、proxy選抜、actual costの効果を分ける。
- M1のdevelopment mapとM2の固定transferを別の証拠として示す。
- 主要baselineに明白な不公平・未適用の大きい既知改善がないかを監査する。
- B2が安い理由を、shotと1-shot cost、その内部要因で説明する。
- 成立しない範囲、未評価範囲、oracle情報を使う部分を明記する。

この着地点は「作業ログをまとめたもの」ではない。ただしH4二geometryのみなら、広い一般性やscalingを表題に入れない。

### 8.2 推奨する目標の着地点：反証可能な資源適用条件研究

上に加えて、次を狙う。

> **単純なprefix処理の比較において、どの費用・誤差がpartialの利益を作るか、既知の強い実装・精度条件の変更でその利益が残るか、または消えるかを示す。候補集合を明示した資源境界を与え、一つの独立した構造条件で説明の移送を検査する。**

必要なのは、partialの勝利を多数並べることではなく、利益の原因と限界を説明すること。独立条件を追加する場合も、単一のsize違いから漸近scalingを主張しない。

### 8.3 元のエネルギー推定への発展

chemical-accuracy ground-state energyを中心claimに残す場合には、ε_E、失敗確率、状態overlap、alias条件、各roundの信号・誤差・shot、最終推定器、全roundの費用が必要となる。

これは本書の信号taskと別の拡張である。長い実回路を全てcompileする必要があるとは限らないが、短時間q=1の結果だけから最終energy資源を外挿してはならない。

最初の研究をここまでの全完成に依存させることは推奨しない。

---

## 9. 次の検証：順序と停止位置

段階名を**PM-0〜PM-3**とする。旧P-D、FR、PR-2 S3とは別であり、既存authorizationは再利用しない。

| 段階 | 目的 | 新規科学計算 | 完了時の判断 |
|---|---|---|---|
| PM-0：証拠帰属・同一集合再解析 | 今ある主張のどこが支持され、何が混同されているか | なし | 主claimと反証対象を確定 |
| PM-1：強いbaselineへの最小反証試験 | 実装や近接discard候補で利益が消えないか | 必要部分のみ、別認可 | resource study継続／限定note／別方針 |
| PM-2：固定taskの資源境界 | ε_signalとPに対する有効領域・破れを説明 | 保存値で可能な部分はなし | 予測・説明として残す内容を確定 |
| PM-3：独立構造条件またはenergy接続 | 目標claimに足りない一軸だけ検査 | 条件付き | 原稿完成またはscope縮小 |

**直近はPM-0まで。その結果を使い、PM-1の対象を固定する。文書freezeだけを何巡も繰り返すことは目的にしない。**

---

## 10. PM-0：科学計算ゼロの再解析

### 入力

M1-A、M1-B1、M1-B1 validation、M2 result/report、旧S2結果、current source。NPZ、runtime、cacheの追加pushは要求しない。

### 必須出力

**A. 比較集合を一致させた表。**

- M1のq=8部分集合と可変q集合。
- M1全候補と旧selector16件。
- M1のM2共通5構成とM2の5構成。
- 同じq/r/Kでrankだけを変える比較。比較できない組はmissingとする。

**B. primaryとsecondaryを分離したregret。**

RZのregret、secondary metricのregret、frontier membershipを別々に記録する。6指標の一つにわずかに優れることを、primaryの大きい利益と混同しない。

**C. N×1-shot、normalization、axis allowanceの分解。**

既存値から上の7節を全必要候補へ拡張する。B0の純粋なdiscard/PF分解は保存値がなければmissingとする。

**D. 同一R=qrの再集計。**

rank/K/T固定でnormalization・expected random applicationsが一致する組を確認し、qによるdeterministic workとsignalの変化を分ける。たとえばR=8の四組は登録grid内にある。

**E. candidate-domain/boundary監査。**

M1ではr64が全q/Kへ網羅的に追加されたわけではなく、B3 rank0 q8 r64 K2とB2 rank3 q1 r64 K2の二件である。[R4] q、r、Rを混同せず、B3 endpointの結論は登録grid内と記載する。未評価のq1/r256等が必要かは、既存の誤差・費用分解で判断し、自動追加しない。

**F. baseline source監査。**

現行sourceではGaussian basis changeを無制御で前後に置き、diagonal primitiveを制御している。この点は既に確認でき、全basisを無駄にcontrolしていると決めつけない。[R6]

一方で、block間の相対basis融合、対称PFのcontrol-aware合成、diagonal primitiveの実装などが同じscopeで十分反映されているかを、既知手法と照合する。`opt1`を全方式に与えただけでalgorithm-level baselineが強いと結論しない。

### 成果物案

- `docs/research/pr2_post_m2_evidence_attribution.md`
- `artifacts/pr2_post_m2_evidence_attribution/2026-10-04/summary.json`
- common-domain tables、N×cost分解、同一R比較、missing-evidence一覧。

これらのpathは提案であり、現在リポジトリに存在するとの主張ではない。

### 終了条件

PM-0の目的は新たなGOラベルの作成ではなく、**次に一件だけ反証するなら何かを決めること**である。結果から支持できない文はclaim表で落とす。M1/M2のraw結果・正式statusは保持する。

---

## 11. PM-1：最優先は一般化追加ではなく、強いbaselineへの反証

### 11.1 同じ演算の既知合成改善

対象はまずdevelopment H4 1.00 Å。化学条件、T、signal、candidate、sampling分布を変えず、実装policyだけを比較する。

比較する既知要素は、source監査で未適用と分かったものに限る。

- 隣接Gaussian basisの相対変換への融合。
- 対角演算の制御化とphase規約。
- 対称Trotterに適用可能なcontrol-aware構成。[W7]
- random eventで利用できるsupport-restricted構成。ただし停止したinterval-DPを新手法として再開しない。

全てを再実装する必要はない。既に適用済みなら、それを明記して当該項目を閉じる。

**必要な意味論test**：uncontrolled operatorだけでなく、ancilla=0/1を含むcontrolled operatorと相対位相を確認する。global phaseを無条件に同値として落とさない。

**費用比較**：同一candidateの同一trajectoryを旧/new policyへ渡すpaired比較にする。再samplingで差を大きく見せない。変更policyから得たactual circuit costを、旧cacheやfingerprintへ上書きしない。

実施規模の提案：新policy一つにつき、現行M2共通5構成をdevelopment上で評価する場合の上限は196 wrappers相当。まずB0/B1の4 wrappersとsynthetic correctnessを見て、random候補へ進める必要を判断する。これは新たな予算案であり、M2の実行枠を再利用するものではない。

**何が研究判断を変えるか**：B2が安かった理由が、既知の強いbaselineを用いると消えるのか、それでも同じ誤差・sampling負担のもとで残るのかが分かる。

たとえばM2のB2/B0点比0.5861に対し、B2側の新cost倍率をa、B0側をdとすると、新比は0.5861a/dとなる。この式は感度の説明であり、特定の既知手法が総RZを半減すると予測したものではない。

### 11.2 近接するdiscard候補の穴を確認する

M1のdiscard候補rankは3/6/9であり、rank3が不適格、rank6が適格でもrank4/5の結果は自動的に決まらない。B0に対する大きい利得を中心にするなら、この間の候補が最も直接的な反証候補になる。

最小案：developmentだけでB0 rank4/5、q∈{1,2,4,8}、既存T・精度・state・DF表現を使う。最大8 signal record、16 deterministic wrappersであり、random trajectory追加はない。

ただしPM-0で既存の対応値が見つかれば再計算しない。B0との比較を登録rank集合に限定したnoteとして閉じるなら、この追加を必須にはしない。

ここでB0が有利になっても「失敗したから別rankを試す」とはしない。**どの程度圧縮すればrandom補完が不要になるか**というRQへの結果として扱う。

### 11.3 高次PF・他の決定論法の扱い

「二次DF-PFを使う固定実装class内」の主張なら、全高次PFの網羅比較は不要。一方、「有力な決定論法よりpartialが有利」という表題・abstractを採用するなら、二次だけでは不足する。

その場合は、既存実装と文献から事前選択した有力な決定論PFを少数追加し、qと精度をそれぞれ選ばせる。新しい高次partial-RTEやPF係数探索へは戻らない。SPRINT全体を再実装することも本段階の必須要件にはしないが、比較していない範囲は明記する。

### PM-1後の停止

- 強いbaseline後も利得と機構が残る：PM-2と原稿化へ。
- 利得が消え、既知の実装差で説明できる：implementation-conditioned resultへscopeを縮小。
- 定義・意味論・受理規則が不整合：修正と影響範囲の特定。科学的NOT_SUPPORTEDと実装failureを混同しない。
- 小差のr/K winnerだけが不明：その差を解像するためだけに追加trajectoryを使わない。

---

## 12. PM-2：保存済み信号から作れる精度・準備費用の資源境界

### 12.1 εを変えるために全Hamiltonianを再計算する必要はない

固定H、state、T、candidate、implementationなら、保存したb_x,a、B_x、compiled meanを用いて、εだけ変更したNとGを再評価できる。

候補xが適格になる厳密境界は

\[
\epsilon_{\min,x}=\sqrt2\max(b_{x,\mathrm{Re}},b_{x,\mathrm{Im}}),
\qquad \epsilon>\epsilon_{\min,x}.
\]

そこからεに対するcandidateのresource curveとP方向のlower envelopeを作る。

これは新しいsignalの実測でもfresh validationでもなく、**固定候補に対する事後感度解析**である。元のM1/M2のε=0.05の正式結果を置き換えない。

数点の見栄えのよいεを選ぶのではなく、各候補の適格性境界と方式間crossingを先に列挙する。ceilingによる細かい段差とmaterialなmethod-regionの変化を分離する。

### 12.2 有効な図

1. 固定candidateの1-shot cost対N。B0/B1/B2/B3を同じ軸で示す。
2. ε_signal対最小G。候補が存在しない領域も明示する。
3. (ε_signal,P)におけるmethod-region。fine r/K winnerは不確かならまとめる。
4. 同一R=qrでqを変えたGとbias。normalization不変の対照として示す。
5. legacy/known-optimized implementationのmatched comparison。

### 12.3 避けること

- 全候補不適格な厳しい精度で「partialは不可能」と一般化する。
- M2結果を見て設計した規則を、同じM2でblindに通ったと表現する。
- 1.30 Åの5構成だけでheld-outのglobal frontierを描いたと呼ぶ。
- 予算方向の事後sweepを「事前登録されたtransferが拡張された」とする。

M2は凍結した5構成と元の精度について独立評価の価値を保つ。M2を新規設計へ使った時点から、以後の新しい仮説にはdevelopment/exploratoryとして扱う。

---

## 13. PM-3：最後に足す検証は、足りない主張に合わせて一軸だけ

### 13.1 一般性を広げる場合

同じH4のgeometryを多数追加するより、違う系サイズまたはfragment構造を持つ小さいinstanceを一件、結果前に固定する方が目的が明確である。

具体例としてH6/STO-3Gは候補になるが、これは今回採択済みの対象ではない。入力snapshotの利用可能性、state-action/sector計算、compile上限を確認した上で選ぶ。大きいfull-space dense行列を現在のH4実装から自動的に生成しない。

ここでは二つの実験を混ぜない。

- **fixed-configuration transfer**：developmentで定めたmapping・構成を移す。新条件で再最適化しない。
- **fixed-protocol new-instance benchmark**：結果前に候補生成規則と探索範囲を固定し、全方式に同じ最適化機会を与える。これはM2と別の評価設計である。

今回の目標が「資源境界の説明」なら後者が自然な場合があるが、最適化後の結果を無調整transferと呼ばない。

### 13.2 元のエネルギーtaskへの接続を優先する場合

size追加と同時に行わず、一つの小系で最終energy目標を定めた費用集計へ進む。必要な複数時刻でaccuracy・phase branch・overlap条件を確認し、選択したRPE等の推定器の保証と整合させる。

T=0.8のq=1を長い時刻へそのまま外挿しない。finite-RTEのBは累積し、task最適なq/r/Kも変わる。

### 13.3 PM-3は最小noteの必須条件ではない

PM-0〜PM-2でH4限定の有用な比較と機構が閉じるなら、そのscopeで原稿を完成できる。より広いresource claimを目指す場合だけ、必要な一軸を追加する。

---

## 14. 主張と証拠の対応表

| claim | 現在の証拠 | 残る不足 | 着地点での扱い |
|---|---|---|---|
| 登録M1候補内でpartialが低コスト | あり | 強い既知実装・候補穴との関係 | 維持し、範囲を限定 |
| 固定5構成は未使用geometryへtransferした | M2で支持 | 他size・他taskの一般性 | 既存の主要実証結果 |
| q最適化で初めてpartialが有利になった | 同一集合では不支持 | 帰属の訂正 | 中心claimから除外 |
| 旧16-cell selectorはprimary RZを大きく誤った | 不支持。RZ regret 0 | なし | 主張しない |
| 6指標frontierの完全保持は失敗した | 支持 | その科学的重要度 | secondaryな監査結果 |
| M2だけ大きいPでdeterministicへ移る | 共通5集合では不支持 | candidate-domain差の訂正 | geometry固有現象としない |
| DF残差のdiscard/random比較は初めて | 先行研究あり | なし | 新規性にしない |
| 強いbaselineでもpartial利益が残る | 未確定 | PM-1 | 目標claim |
| 精度・費用条件による成立境界を説明できる | 部分的 | PM-0/PM-2 | 研究の中心 |
| 化学精度のenergy推定総資源が改善 | 未検証 | 別task設計と全round集計 | 現段階では主張しない |

---

## 15. 過去の停止方向との整合

| 方向 | 今回の扱い |
|---|---|
| P-A interval/joint synthesis | interval-DPを復活させない。既知basis/control改善のbaseline監査は別目的。 |
| P-B energy/signal weight | 同じH4 gridで反例探索を再開しない。状態情報のscopeを明記するのみ。 |
| P-C geometry差分cancel | 現familyの停止を保持。今回のgeometryは固定構成の移送であり、差分energy設計ではない。 |
| P-D finite-tail-aware selection | finite-K補正が重要だという結論を先に置かない。十分小さければ補助要因として扱う。 |
| R3 selector/multi-fidelity | 新selector開発や汎用最適化の再包装に戻らない。保存集合の再集計は解析手順。 |
| FR phase/amplitude | 条件付き理論noteと本resource studyを区別する。FRのpractical GOをM2から遡及的に与えない。 |

過去pilotのSTOPはその設定・仮説への判断であり、関連分野全体に新規性がないという証明ではない。一方、次の結果が出るまで条件を増やすための再開理由にも使わない。

---

## 16. 想定する論文構成と図

### 本文構成

1. **問題設定**：既知partial randomizationに対して何を追加検査するか。signal-taskとenergy-taskの区別。
2. **方法と比較契約**：DF snapshot、residual処理、canonical finite-RTE、shot規則、implementation scope。
3. **development資源地図**：M1、q/rの整理、同一candidate-domain比較。
4. **利得の機構とbaseline依存**：N×cost、同一R比較、discard/PF/RTE誤差、強い実装比較。
5. **適用条件と移送**：ε/P境界、M2の固定5構成、条件付きの独立追加。
6. **限界と結論**：oracle情報、finite domain、Monte Carlo cost、energy接続未達成を明記。

### 主図候補

| 図 | 内容 | 根拠・実施段階 |
|---|---|---|
| 図1 | residual処理の共通taskと資源分解 | 定義・既存source |
| 図2 | M1のq=8と可変q、同一split集合 | 保存JSON、PM-0 |
| 図3 | M1/M2共通5構成のshotと1-shot cost | 既存結果＋今回分解 |
| 図4 | 同一R=qrでqを変える対照 | PM-0 |
| 図5 | legacy/strong baseline下のresource比 | 必要時PM-1 |
| 図6 | ε/Pの成立範囲とM2 transferの位置 | PM-2、M2 |

全図を無条件に必須としない。図5に検査する独立要素が残らなければ監査表で置き換える。新しい合成法の図を作るためだけに新手法開発を始めない。

---

## 17. 不確かさ・統計・再現可能性

32 trajectoryはexpected compiled costを見積もるための古典Monte Carloであり、解析shot Nは実際に実行した量子測定回数ではない。

M1の多数候補からのpoint minimumにはselection効果がある。M2の固定候補評価はその問題を抑えるが、二候補の最小値や6指標Paretoをformal多重比較保証へ変えるわけではない。

以下を区別する。

- source/data identityが一致したこと。
- correctnessの数値検査を通ったこと。
- cost期待値のMonte Carlo誤差。
- 固定candidate集合を超える探索不足。
- 別instanceへの一般化誤差。
- 実装costからphysical resourceへ変換するmodel誤差。

既存のpoint±2SEはengineering intervalとして維持する。新しい対照でformal confidenceを主張するなら、その標本設計と多重性を結果前に別途定める。少差のwinnerを決めない方針で閉じられる場合は、そのためだけの高統計化をしない。

希少Taylor eventが平均costへ与える寄与について、保存分布・最大event長・既存記録で確認できる範囲は記載する。32件で観測されなかったeventを確率ゼロとみなさず、sample平均から厳密な期待値保証を作ったとは言わない。

---

## 18. 最終的な採否・完了判断

### 本研究として進める根拠

強い比較対照と同一候補domainのもとで、partialの有効・不利領域を誤差・費用の競合から説明できる。移送結果または独立検査がその説明を支える。一般に使える説明・評価上の教訓が、既知trade-offの単なる再図示を超えて残る。

### technical/resource noteへ縮小する根拠

結果は正しく再現可能だが、残るのが特定H4・特定compilerでの既知手法の実装比較に限られる。これは失敗を隠すための名称変更ではなく、主張のscopeを証拠に合わせる判断である。

### 主題変更を検討する根拠

強いbaseline後に利益が消え、その消失が既知の実装改善で説明され、新しい適用条件・失敗機構・評価手法も残らない。この場合、同じ仮説で勝つ条件を追加探索しない。既存結果を閉じてから次RQを考える。

### 不確定な場合

研究主張に必要な比較かどうかを先に判断する。r4/r8の細かい差だけなら未確定でよい。partial対best endpointの結論を本当に左右する不確かさであれば、費用と得られる情報を明記した一回の追加設計を検討する。

**合格status、test数、hash数は研究貢献の代わりではない。** 同時に、独立した新アルゴリズムがないことだけを理由に実証研究を棄却しない。

---

## 19. Codexへ渡す直近の作業指示案

以下はPM-0の範囲を表す。利用者が採用した場合の作業案であり、本書だけで新しい科学実行を認可しない。

```text
基準commit b6e65c6123475add5e620ec1064f361378bead95 の保存済みJSON・報告・sourceだけを使い、
M1/M2の証拠帰属を再解析してください。

1. M1のq=8部分集合と可変q集合を同じsplit/方式domainで比較する。
2. 旧16-cell selector、64-cell proxy frontier、actual frontierを区別し、primary RZとsecondaryのregretを別表示する。
3. M1をM2と同じ5構成に限定してP>=0 envelopeを計算し、M2の5構成と比較する。
4. 各代表候補をtotal analytic shots × shot-weighted effective one-shot costへ分解する。
5. rank/K/T固定、R=q*r固定の既存候補を比較し、normalization・random actionとq依存部分を分離する。
6. B0のouter_pf_bias_absがdiscard+PFの総biasであることをsourceから確認し、純粋な成分分解に必要な保存値の有無を示す。
7. 既知のbasis/control最適化が現行sourceへ適用済みかを静的に監査し、未確定点だけを一覧にする。
8. 読めない値、無い値はmissingとし、推定値で埋めない。

新しいNPZ読込、分子計算、signal評価、trajectory、circuit build/compile、量子shot、
M2再実行、held-out再探索、H5/H6/H12、長RPE、全repo testsを行わない。
旧result、status、fingerprint、契約は書き換えない。
新しい集計はPOSTHOCとして別artifactへ保存し、完了後停止する。

最終出力は「支持されるclaim／支持されないclaim／未検証claim」の対応表と、
PM-1で検査する必要がある最小のbaseline項目だけにする。
```

### ここで完成させる研究契約

PM-0の結果を反映して、主RQを本書5節の形に固定する。主題候補を再び八方向へ増やすのではなく、どの主張まで完成させるかを選ぶ。

- 最小note：同一条件比較、機構分解、M2固定transfer。
- 目標paper：強いbaselineとε/P境界、必要なら一つの独立構造条件。
- energy発展：別の達成要件として切り分ける。

これにより、「もう一つ検証してから方針を考える」を繰り返すのではなく、**固定した問いを閉じるために必要な検証だけを行う**進行へ移す。

---

## 20. 参照資料

### リポジトリ資料（全て基準commitで固定）

[R1] `docs/pr2_matched_accuracy_m1_b1_result_validation.md`。M1の条件、primary/Pareto、selector監査、P感度。

[R2] `artifacts/pr2_matched_accuracy_m1_b1_result_validation/2026-10-03/pr2_matched_accuracy_m1_b1_result_validation_v1.json`。`analysis.best_by_method_rank`、`fixed_q8_best_by_method_rank`、`old_selector_actual_comparison`。

[R3] `docs/pr2_matched_accuracy_m2_transfer_result_validation.md`。M2の5構成、signal、N、cost、SE、transfer判定、P交点。

[R4] `docs/pr2_matched_accuracy_m1_a_validation.md`。候補集合、boundary追加、accuracy集計。

[R5] `src/trotterlib/pr2_matched_accuracy_m1_execution.py`。`_target_signal`、`_axis_record`、`_base_record`、`_deterministic_signal_record`、`_random_signal_record`。

[R6] `src/trotterlib/df_partial_s2.py` の `_append_basis` / `_append_block`、`src/trotterlib/df_partial_s2_repeated.py` の `_build_boundary_optimized`。

[R7] `docs/research/pr2_matched_accuracy_prior_art_gate_v1.md`、`docs/research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md`。

[R8] `docs/research/研究概要・現状.md`、`VALIDATION_STATUS.md`、過去チャット引継ぎ。過去のP-A/P-B/P-C/P-D/R3/FR停止理由の文脈。古い文書の「現在」は、その文書の作成時点として読む。

### 一次文献

[W1] J. Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum **7**, 020332 (2026), DOI `10.1103/ynxb-p2xq`, arXiv `2503.05647`。重点：Sec. VII、App. A.3、D.2、E。journal PDFのFigs. 10, 19, 20と周辺本文を確認。

[W2] M. Hagan and N. Wiebe, *Composite Quantum Simulations*, Quantum **7**, 1181 (2023), DOI `10.22331/q-2023-11-14-1181`, arXiv `2206.06409`。

[W3] P. A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv `2606.30741v1` (2026)。重点：Secs. II, III, IV, V。factorization、remainder、near-integrabilityと実装の範囲を確認。

[W4] O. Oumarou et al., *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*, Quantum **8**, 1371 (2024), DOI `10.22331/q-2024-06-13-1371`, arXiv `2212.07957`。

[W5] S. Kanasugi et al., *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*, arXiv `2603.22778v2` (2026)。重点：Pauli-LCUのscope、single-ancilla QPE、UWC、resource model。

[W6] D. Cugini, T. A. Atif, Y. Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv `2603.13495v1` (2026)。重点：Sec. IIのnet-costとsampling、適用範囲。本研究ではcanonical samplerを保持し、optimal IS一般を再提案しない。

[W7] W. A. Simon and P. J. Love, *Halving the Cost of Controlled Time-Evolution*, arXiv `2511.13855v1` (2025)。対称Trotterにおける任意角回転の制御化。総compiled-RZやDF実装への改善率は別途確認が必要。

[W8] M. Motta et al., *Low rank representations for quantum simulation of electronic structure*, npj Quantum Information **7**, 83 (2021), DOI `10.1038/s41534-021-00416-z`, arXiv `1808.02625`。部分基底変換・低ランク実装の先行研究。

本文の新規性評価は上記の重点照合に基づく。網羅的systematic reviewや同一研究の不存在証明ではない。
