# Track A：PM-2後の研究方針・RQ・新規性・完成条件

作成日：2026-10-05  
対象：`HIROMU1015/Partially-Randomized-Trotter`  
branch：`pr2-v4-s2-parallelization-20260928`  
証拠を固定するcommit：`5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc`  
位置付け：研究方針の提案・文献照合・保存値の解釈。既存statusの変更、新しい科学計算の認可、Track Bとの優劣判定ではない。

## 0. 推奨判断

**Track Aの新規計算キャンペーンをここで区切り、固定DF表現・二次PF・canonical finite-RTE・固定shot規則における有限時間信号の資源比較として原稿化へ移る。PM-3は自動開始しない。**

PM-2は、研究の中心を再び探し始めるためではなく、既に定義した限定RQへ答えて研究を閉じるための解析だった。その役割は果たしている。[R1,R2]

推奨する中心は、次の三点である。

1. 登録した候補集合内では、近接discardを追加してもpartialの低いRZ-work点推定が維持される。
2. 要求精度を変えると、同じpartial familyの中でも望ましい離散化設定が変わる。
3. 固定構成の移送成功は、別の要求精度、別の候補集合、別のshot規則での最適性を意味しない。

ただし、これは新しいpartial-randomizationアルゴリズム、普遍的な最適化原理、全化学系の設計則ではない。新規性の現在の評価は**限定された実証的寄与**であり、独立した方法論論文に十分な新規性が確立したとはしない。

原稿化へ進むことと、特定誌で採択されることは別である。一方、採択可能性が未確定だからといって、勝つ条件を追加探索し続けることも勧めない。

## 1. 今回の根拠と確認範囲

リポジトリから、PM-2の契約、結果報告、summary、representative decompositionの一部、PM-0の帰属解析、PM-1報告、M2報告を確認した。主要一次文献はPR論文、Composite Quantum Simulations、SPRINT/GRADE、RC-DF、Kanasugiら、resource-optimal importance sampling、controlled-Trotter compilationを照合した。[R1–R7,W1–W7]

PR論文はjournal本文に加え、Sec. VII付近のFig. 10とAppendix D.2のFigs. 19–20をページ画像でも確認した。

本レビューで行っていないものは、分子データの読込、新しいsignal評価、trajectory sampling、回路build/compile、tests再実行、全ledgerの独立再集計、全source hashの再検証、リポジトリの変更である。数値は保存報告・可視のCSVから引用し、一部の比と不等式だけを算術的に整理した。

以降では、[既存結果]、[本レビューの解釈・導出]、[提案]、[未確認]を区別する。

最新Track B資料は今回の根拠に含めていない。この結論はTrack Aの完成方針であり、「Track Bより有望／不利」といった比較ではない。

## 2. PM-2までで得られた証拠

### 2.1 共通のtaskと費用scope

H4 linear、STO-3G、固定DF rank 12、8 system qubits、保存参照状態、T=0.8。developmentは1.00 Å、M2は1.30 Å。primaryは状態準備を含まないfull measured Hadamard wrapperのcompiled RZ平均と解析shot数の積である。compilerは保存されたQiskit 1.3.0、opt1、basis rz/sx/x/cx、seed17、topology指定なし。[R1,R5,R6]

DF rank 12はHamiltonian表現のrank、L_Dは決定論側へ残すprefix長である。B2 rank3というcandidate名のrankはL_D=3であり、元のHamiltonianをDF rank3へ置換したことを意味しない。

| 方法 | 残差処理 |
|---|---|
| B0 | DF-prefixを保持し、残りをdiscardする |
| B1 | rank12の全DF fragmentを決定論的に扱う |
| B2 | 中間prefixを決定論的に残し、残差をcanonical finite-RTEで補完する |
| B3 | 二体DFの決定論prefixが0のrandom-dominant構成。one-bodyは決定論的に残る |

### 2.2 PM-1の近接discard反証

追加したB0 rank4/5 × q=1/2/4/8の8構成は全てaccuracy適格だった。最小はrank5/q1でG_RZ=229,718,060、保存B2 rank3/q1/r4/K2は130,774,896.65625。[R5]

保存値から、B2/B0比は0.56928435、B2のRZ-work点推定は約43.07%低い。総shot比0.7936318と有効1-shot費用比0.7173154の積がこの比になる。

これはrank4/5という欠測を埋めても順位が覆らなかった証拠であり、全prefix、全高次PF、全合成法の最適性ではない。PM-1は新しいgeometryの独立検証でもない。

### 2.3 developmentの精度依存

M1の210候補とPM-1の8候補を合わせた218候補を、ε=0.005〜0.1の301対数点と正確な0.05の計302点で再評価した。[R1,R2]

| 表示されたεの範囲 | primary RZの点最小構成 | 表示点数 |
|---|---|---:|
| 0.005〜0.0054158189504 | B2 rank3 q4 r4 K4 | 9 |
| 0.00547017101788〜0.0235054643684 | B2 rank3 q2 r4 K4 | 147 |
| 0.0237413604716〜0.1 | B2 rank3 q1 r4 K2 | 146 |

全表示点で点最小のmethodはB2であり、精度を厳しくすると点最小設定のqが大きくなる。しかしq/r/Kの厳密なwinnerは未確定であり、表は連続ε上の厳密境界ではない。

### 2.4 M2元5構成の精度依存

M2元5構成では、表示ε=0.005〜0.00667938131366の30点でB3、それ以降の0.00674641423837〜0.1の272点でB2 rank3/q1/r4/K2がprimary点最小だった。[R1]

元のε=0.05でのTRANSFER_SUPPORTEDは保持される。別精度の結果は保存値の事後解析であり、新しいheld-out試験ではない。

M2にはB2 q2/q4やPM-1の新B0 rank5が含まれない。したがって、ここから「1.30 Åでは厳しい精度にB3がmethod最適」と結論できない。

### 2.5 P感度

元ε=0.05のdevelopment全218候補のpoint lower envelopeはB2だけだった。M2元5構成では、P≈385.08でB2 r4からr8へ、P≈208,735.13でB1へ移る。[R1,R6]

ただし候補集合が異なる。PM-0でM1も共通5構成へ制限すると、大きいPでB1が入ることを確認済みである。したがってこの差をgeometry効果だけに帰属しない。[R4]

Pは共通の仮想RZ-equivalent費用/shotであり、実際の状態準備回路や物理実行時間の実測ではない。

## 3. 今回さらに限定すべき解釈

### 3.1 302精度点は302件の独立な成功ではない

同じ保存bias、normalization、同じ32 cost trajectoryを再利用しているため、PM-2の302精度点は強く関連した感度曲線である。成功率302/302という形で一般化の証拠量に換算しない。[R1,R2]

候補xの二軸平均費用をC_hat_x、精度εのshot重みをw_x(ε)とすれば、

G_hat_x(ε)=w_x(ε)^T C_hat_x

であり、同じ標本を使う二精度間には

Cov(G_hat_x(ε),G_hat_x(ε'))
= w_x(ε)^T Cov(C_hat_x) w_x(ε')

がある。これは保存値再利用からの代数的帰結であり、新しい統計解析を実行した主張ではない。

### 3.2 developmentではmethod切替ではなく、同一family内の設定変更が主結果

固定q=8でもM1の最小methodはB2だったことはPM-0で確認されている。PM-2のdevelopmentでも最小methodはB2のままである。[R4,R1]

したがって「精度変更でdeterministicからpartialへ転換した」ことを中心結果にしない。正確には、**partial familyの点推定競争力が保持されつつ、望ましい離散化設定が精度とともに変化した**という結果である。

### 3.3 重要：ε_minは現行受理規則の境界であり、原理的精度限界ではない

現行規則は実部・虚部へそれぞれε/sqrt(2)を割り当てる。そのため候補xの適格条件は

ε > ε_min,x := sqrt(2) max(b_x,Re,b_x,Im)

である。[R2]

一方、M2 B2 r4の総複素biasは約0.003725912、現行規則でのε_minは約0.00525637654710である。[R6,R1]

要求ε=0.005では現行規則は不適格だが、

|ν_x-z_H| ≈ 0.003725912 < 0.005

でもある。

したがって「この回路・推定器の平均が元の信号から0.005以上ずれている」ことは意味しない。三角不等式

|z_hat-z_H| ≤ |z_hat-ν_x|+|ν_x-z_H|

から、理想的な測定モデルで平均ν_xを十分精密に推定すれば、ε=0.005へ到達する余地はある。例えば統計誤差ノルムをε−|ν_x−z_H|より小さくする余地が正である。

これは別のaxis配分やshot規則を実装・費用評価したという意味ではなく、その方がB3より安いことも示さない。現在のルールを結果後に変更する提案でもない。

**完成原稿では「現行の対称軸配分・Hoeffding十分shot規則における適格境界」と呼ぶ。** これによりM2の精度依存transitionを、方法一般の不可能性や根本的な精度限界として過大評価しない。

### 3.4 近接構成間の不確かさと、method間の不確かさを分ける

全302点で点最小と他候補のengineering intervalが重なることは、全点でB2とendpointが区別不能という意味ではない。重なる候補が同じB2の別r/Kである可能性がある。[R1,R7]

逆に、点最小が全点B2だから全域で統計的に認証されたmethod優位とも言えない。原稿ではmethod別の点最小と、そのfamily内の設定順位を別に示す。

点±2SEはformal CIではない。α_a=0.025は一つの信号推定のshot規則であり、候補選択や302点全体の同時winner保証ではない。[R2]

### 3.5 PM-1で未評価の機構はPM-2でも未評価

B0ではfull-Hに対するdiscard＋PFの総biasを使っている。qの増加に伴う総biasの非単調性は観測済みだが、exact truncated-H signalがないのでpure discard/PFや干渉項は未分解である。[R4,R5]

「誤差相殺を発見・実証した」とせず、「総biasはqに単調ではなかった。成分機構は未確認」とする。新しい科学計算をしない完成形では、その範囲で十分である。

## 4. 固定する研究方針とRQ

### 4.1 仮題

**DF-prefix時間発展における精度条件付き資源比較：残差切断・ランダム補完・固定構成移送の事例研究**

英語案：

*Accuracy-conditioned resource trade-offs in DF-prefix coherent-signal estimation: a case study of truncation, randomized completion, and frozen-configuration transfer*

「case study」を外すだけで一般性を得たことにしない。表題ではfinite-time coherent-signal taskを明示し、chemical-accuracy energy estimation、global optimality、新algorithmを示唆しない。

### 4.2 主RQ

> 固定DF表現・二次PF・canonical finite-RTE・所定のshot規則のもとで、同じ有限時間coherent signalを所定精度で推定するとき、登録された残差処理構成の資源競争力は、要求精度、総bias、normalization、full-wrapper回路費用によってどう変わるか。

### 4.3 副RQと既存の回答

**RQ1：同一taskでの残差処理比較。** 近接discardを含む登録候補内でpartialの低コスト点は残るか。M1/PM-1が答える。残るのは一般優位性でなく固定実装class内の観測。

**RQ2：適格性と資源効率の分離。** 精度を満たすだけで低資源とは限らない理由を、統計誤差の余裕・normalization・1-shot費用から説明できるか。PM-0/PM-2が答える。未取得のHamiltonian誤差成分を原因として断定しない。

**RQ3：固定構成移送の精度依存性。** 元精度で移送された構成は、別の要求精度でも適格・競争的か。M2は元精度の独立移送、PM-2は同じ固定5構成の事後感度として答える。二つの証拠階層を分ける。

「最適splitの安価な予測器」「新しいsampling分布」「新しいPF係数」「一般的な誤差certificate」は今回のRQに含めない。過去のP-D/R3/FR等を名前だけ変えて再開しない。

## 5. 結果を説明する数理的骨格

以下は既存評価式の整理であり、新しい最適化理論の主張ではない。[R2]

候補xの補正後信号をν_x、targetをz_H、normalizationをB_xとする。aはReまたはIm。

\[
e=\epsilon/\sqrt2,\qquad b_{x,a}=|\nu_{x,a}-z_{H,a}|,\qquad s_{x,a}=e-b_{x,a}.
\]

\[
N_{x,a}(\epsilon)=\left\lceil \frac{2B_x^2}{s_{x,a}^2}\log\frac{2}{\alpha_a}\right\rceil.
\]

この十分shot数を使う平均資源は

\[
G_x(\epsilon,P)=\sum_a N_{x,a}(\epsilon)(\overline C_{x,a}+P).
\]

PはRZに関する仮想共通準備費用である。primary以外へ同じPを無条件転用しない。

### 5.1 正規化した統計誤差の余裕

\[
h_{x,a}=s_{x,a}/e=1-b_{x,a}/e.
\]

αが両軸共通の場合、ceilを除いた説明用連続式は

\[
G_x^*(\epsilon,P)=\frac{4\log(2/\alpha)}{\epsilon^2}
B_x^2\sum_a\frac{\overline C_{x,a}+P}{h_{x,a}^2}.
\]

この形なら、費用増加の原因をB_x^2、h^{-2}、Cの三つへ分けられる。ただし全軸の和なので、一つの最大biasだけを用いた掛け算へ無条件に簡約しない。

hが半分なら、その軸の連続shot式のheadroom因子は4倍になる。これが「適格だが高資源」という現象の評価式上の理由である。新しい定理でも、未知系のtruth-free設計則でもない。

ceilを除いた式は説明補助であり、正式ledgerは整数shotを使う。有限で非負の費用のもとで

\[
0\le G_x-G_x^*\le\sum_a(\overline C_{x,a}+P)
\]

と評価できる。巨大Pでの少数shot差・丸め依存の切替を、実用上の精密な境界として強調しない。

### 5.2 同じR=qrで見るqの役割

同じL_D,K,T,R=qrなら、現行のRTE構成では

\[
\tau=\lambda_R T/R,\qquad B_{\rm total}=[B_K(\lambda_R T/R)]^R.
\]

normalizationとrandom action期待値を固定して、qによるdeterministic workとsignal biasの違いを比較できる。PM-0に既存56 groupの照合がある。[R4]

この整理も実装式の帰結であり、新しいsampling法ではない。同じRでも演算順序とdeterministic分割が違うため、同じsignalや同じcompiled costにはならない。

### 5.3 誤差分解は条件付き補足に留める

\[
z_{L,q}-z_H=(z_L-z_H)+(z_{L,q}-z_L).
\]

この恒等式は既知の誤差の足し算であり、交差項が総biasへ影響し得る。ただしz_Lを取得していない現研究では、実際の寄与は未知である。

今回の完成形は総biasによる資源説明で閉じるため、新たにz_Lを求めることを必須にしない。相殺機構をabstractの主claimへ上げるときだけ、別scopeとして直接検査する。

## 6. 先行研究との照合

### 6.1 最接近：GüntherらのPR論文

W1はpartial randomization、RTE、single-ancilla位相推定、量子化学資源評価を既に扱う。Sec. VIIではDF fragmentを決定論側へ残し残差をランダム化する構成と、水素鎖のfactorized resource estimateを示す。

Appendix D.2ではPauli表現におけるsmall-term truncationとground-state energy errorを調べ、Figs. 19–20ではその切断比較を示す。その図の分析ではTrotterization errorを除外すること、実用上はTrotter・位相推定にも予算が必要であることを明記している。

従って「DFに適用した」「discardを比較した」「測定込みで評価した」だけを新規性とはしない。

今回の追加対象は、保存したDF-native構成と有限coherent-signal taskに対して、総bias・finite normalization・所定の整数shot・実wrapper費用の精度依存を同じ台帳で比較した数値結果である。元論文が誤差と費用の接続を扱っていない、という批判はしない。

### 6.2 その他の最接近研究

| 文献 | 既知として扱うもの | 今回の位置付け |
|---|---|---|
| W2 Composite Quantum Simulations | deterministic/random分割とerror/costの競合、partitionの理論 | hybrid原理の新提案ではない |
| W3 SPRINT/GRADE | factorization・near-integrability・randomization・remainder戦略・回路資源の共同設計 | 統合設計一般の優位性は主張しない |
| W4 RC-DF | 圧縮・正則化DFによる資源削減 | 固定prefixを新しい圧縮法とはしない |
| W5 Kanasugiら | single-ancilla Trotter QPE、partial randomization、UWC、STAR系の化学資源評価 | Pauli-LCU中心・物理資源taskと今回のDF-native有限信号taskを区別する |
| W6 Cugini–Atif–Subasi | 回路費用と推定varianceのnet cost最適化、biasを保つimportance sampling | cost×shotの原理や資源最適samplingを新規claimにしない |
| W7 Simon–Love | 対称Trotterのcontrolled実装で任意角rotation増加を抑える合成 | 既知の強化baseline候補。総RZ一律半減やDFでの実測利益は未確認 |

### 6.3 新規性の判定

| 主張候補 | 評価 |
|---|---|
| 新しいpartial-randomization algorithm | 支持しない |
| cost×shot、bias headroom、P crossoverという一般原理 | 既知または直接的帰結。新しい理論成果にしない |
| 最適qが精度に依存する一般命題 | それ自体は新規性にしない |
| 本task・candidate domain・実装での定量的資源map | 本研究の実証的寄与として残せる |
| 近接discard反証、固定transfer、その事後精度感度を分離した比較 | 結果の解釈と再利用価値を補強する |
| 未知系にも使える安価な選択法・精度certificate | 未達成 |
| 独立誌論文に十分な新規性 | 現時点では確定できない。方法論的寄与は弱く、限定実証研究として評価すべき |

重点照合した文献では、今回と全て同じ候補domain・signal task・有限cutoff・compiled cost・固定transfer精度解析の評価をそのまま確認したわけではない。ただしそれは不存在証明ではない。初めて／世界初という表現は使わない。

**推奨する寄与の一文：**

> 既知のDF-prefix残差処理を、同一の有限時間信号と明示した十分shot規則・回路実装で比較し、近接discard対照を含む資源順位と、精度適格性・費用競争力・固定構成移送の異なる成立範囲を定量化した。

これは「新しい原理を提案」ではなく、「条件を明示した実装資源の数値検証と限界整理」という寄与である。

## 7. 研究の完成形

### 7.1 採用する完成形

**限定された実証的resource/application studyとして、現在のデータで原稿を完成させる。**

最小完成は、問題設定、方法、M1+PM-1、M2、PM-2、寄与分解、先行研究差分、限界、再現用の軽量入力・表・図を揃えること。

H4二geometry、固定T、固定実装であることを表題またはabstractの早い位置に書く。参照signalに依存したbiasを使うbenchmarkであり、大系にそのまま使える運用policyではないと明記する。

現在の証拠はこの限定された問いに答えるには足りる。ただし「広く一般化する設計論文として完成した」わけではない。

### 7.2 原稿の強さと投稿判断

最小完成形は技術報告・再現可能なcase studyとして成立する。単独の査読誌論文としての十分性は、明示した差分と読者への情報価値による。投稿先、採択確率、締切は新たに仮定しない。

寄与が限定的なら、それを認めて閉じる。新しい主張を作るためだけに精度域、geometry、baseline、seedを追加し続けない。

新アルゴリズムとして強い独立貢献を狙う場合は、このTrack Aを新規性未確定のまま延長するのではなく、別のresearch questionとして設計する。Track B側資料なしにその優劣は決めない。

### 7.3 今回必須にしないもの

高次PF、RC-DF再最適化、strong synthesis全網羅、未知分子への追加移送、H12、長RPE、状態準備の実回路、hardware noise、Clifford+T総資源、物理量子優位を完成条件へ追加しない。

代わりに、それらを評価していないことを明示し、それらへの優位性を主張しない。scopeを限定することは不公平な対照を正当化することではない。等価な演算・精度・費用scopeが揃っていることは原稿内で説明する。

## 8. 必須図表と原稿構成

### 図1：元精度の費用分解

ε=0.05、developmentのB0 rank5、B1、B2代表、B3代表について、N、軸別C、Gを別々に示す。PM-1 rank4も「適格だが高資源」の補助例として示せる。

B2対B1はshot数でなく回路費用が優位を作ること、B2対B0 rank5は双方が小さいことを説明する。[R1,R5]

### 図2：development全登録集合の精度曲線

302表示点と全218候補を維持し、method別の最小点推定曲線と、B2内のq変化を示す。細かなr/Kの厳密winnerを色分けで断言しない。切替を連続精度の正確な根として表示しない。

method間の資源差と同family内のMC不確かさを区別する。既存ledgerからの図表化であり、新しいsamplingはしない。

### 図3：M2固定5構成の適格性と競争力

B3→B2の点順位切替と、B2の現行規則による適格境界ε_minを別に表示する。元の移送精度0.05へ印を付ける。

「fixed five / posthoc / symmetric-axis Hoeffding rule」を図中またはcaptionへ明記する。0.005256を根本的な精度限界と呼ばない。

### 図4：同じRでのq比較

PM-0の同じrank,K,T,Rによる組から、bias、N、C、Gを比較し、q増加が統計負担とdeterministic費用へ逆向きに効く様子を説明する。[R4]

### P envelopeは補足を基本とする

共通5構成のM1/M2比較、development全218集合の感度を別に示す。巨大Pや丸め依存の微差を主成果にしない。M1全218とM2五件の差をgeometry効果として主張しない。

### 原稿の構成

1. 問題設定と既知研究との差分。
2. 固定task、候補domain、finite-RTE、accuracy/shot/cost規則。
3. M1とPM-1のdevelopment資源比較。
4. PM-0/PM-2の精度依存と費用分解。
5. M2固定transferと、その事後精度感度。
6. 不確かさ、ルール依存性、実装・状態・system-sizeの限界。
7. 結論と再現可能性。

stage名とhashの履歴を本論のストーリーにしない。M1/PM-1等はprovenanceとして残すが、本論はRQへの回答順で書く。

## 9. 追加検証が必要になる条件

今回の推奨は追加科学計算0での原稿化である。以下は将来、主張を変える場合だけの条件付き項目。

| 追加したい主張 | そのとき必要になる検査 | 現在の扱い |
|---|---|---|
| discard/PF相殺が非単調性の原因 | 同一状態でrank4/5のexact truncated-H signalを各1件取得して成分を分離 | 原稿の必須claimにせず保留 |
| 強い決定論実装にも利益が頑健 | 適用可能な具体的合成法を決め、両方式へ公平に適用 | 固定実装scopeで閉じるため保留 |
| 高次PF一般への優位 | 有力な高次deterministic baselineに同一taskで最適化を与える | 主張しない |
| 別分子・サイズへの設計則 | 独立問題族・同じ比較domain・別事前契約 | 主張しない |
| chemical-accuracy energy/RPE総資源 | 時刻列、alias、overlap、最終推定器、全誤差と費用 | 別taskとして扱う |
| 軸配分や推定器に頑健な切替 | 別shot規則を同一候補へ公平に適用 | 現規則依存を明示し、今回新規実行しない |

値や実装の不整合が具体的に見つかった場合は、完成を急ぐために隠さず停止する。ただし「もっと強い主張ができるかもしれない」というだけでは、新しい検証キャンペーンを開始しない。

## 10. 直近の作業と終了条件

### 直近の作業

Codexには新しいscience runnerを作らせず、現在の保存結果を使った原稿・図表・根拠対応表を作らせる。

- 主張ごとにsource path、固定commit、既存/事後/導出の区分、対照集合、限界を対応付ける。
- 4枚の主図と補足表を、保存した範囲から作る。未評価領域を補間で埋めない。
- 定理や方法の新規性でなく、実証的差分をintroductionとrelated workへ記す。
- abstract、方法、結果、限界を含む通し原稿を作る。

これらは既存データの原稿化であって、追加scientific authorizationを当然に伴わない。旧result/status/manifestは変更しない。

### 終了条件

以下を満たした時点で、Track Aの限定研究は完成として区切る。

1. 同じestimand・accuracy規則・費用scopeで比較したことを説明できる。
2. 近接discardの反証、精度曲線、固定移送の各結論が元artifactへ遡れる。
3. PM-0の帰属訂正と、今回の受理境界／精度限界の区別が本文・abstractで一貫する。
4. 点推定、engineering interval、解析的shot十分条件、未登録条件が分離される。
5. 新規性と既知事項がclaim単位で整理され、過大な一般化がない。
6. 原稿と図表を共同研究者が読み、論証の不足がデータ不足なのか表現不足なのか判断できる。

これらを確認する最終レビューは原稿レビューであり、さらに次のpilotを自動的に提案する会議ではない。

## 11. Abstract草案

量子化学Hamiltonianの部分ランダム化時間発展では、決定論的に保持する項を減らすことによる回路費用の削減と、信号biasおよびnormalizationに由来する測定負担が競合する。本研究では、固定double-factorized Hamiltonianに対する有限時間coherent-signal推定を対象とし、残差切断、全項の決定論実装、中間prefixとcanonical finite-RTEの組合せ、およびrandom-dominant構成を比較する。H4のdevelopment条件で取得した候補集合に近接discard候補を加え、所定の対称軸配分とHoeffding型十分shot規則のもとで、実コンパイルした測定付きwrapperの期待RZ費用を評価した。元の信号精度0.05では、最小partial費用の点推定は新しい最良近接discardより約43.1%低かった。保存結果を0.005〜0.1の302表示精度へ再評価すると、developmentの点最小は全点でpartialに属したが、その離散化設定は変化した。一方、別geometryへ事前固定して移送した五構成の事後解析では、現行受理規則による適格境界と、random-dominantからpartialへの費用順位切替は一致しなかった。本結果は、精度適格性、測定負担、回路費用、固定構成移送を区別した条件付き資源比較を与える。新アルゴリズム、推定器一般の最適性、化学精度のエネルギー推定全体、未知分子への一般化は主張しない。

この草案は保存値と現在の限定claimに基づく。細かな設定の点順位は統計的winnerとして記述しない。[R1,R5,R6]

## 12. Codexへ渡す原稿化作業案

```text
Track AをPM-2完了時点で限定resource/application studyとして完成させるため、
新しい科学計算ではなく、原稿・図表・claim-to-evidence表を作成してください。

証拠commit: 5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc

最初に読むもの:
- docs/pr2_pm2_precision_resource_result_validation.md
- docs/research/pr2_pm2_precision_resource_contract_v1.md
- PM-2 summary/precision_ledger/eligibility_boundaries/representative_decomposition/P_envelope
- docs/research/pr2_post_m2_evidence_attribution.md
- docs/pr2_pm1_discard_result_validation.md
- docs/pr2_matched_accuracy_m2_transfer_result_validation.md
- 今回のpost-PM2研究方針文書

主RQは固定DF・二次PF・canonical finite-RTE・所定shot規則における
有限coherent-signal推定の精度条件付き資源比較に限定する。

守る解釈:
- 302精度点は同じ保存値・cost標本の再利用であり独立試験数ではない。
- developmentのmethod点最小はB2のまま。q変更とmethod変更を混同しない。
- M2精度曲線は元5構成のposthoc結果で、未登録q/rankのmethod最適性ではない。
- epsilon_minは対称軸配分・Hoeffding規則の適格境界。原理的精度限界ではない。
  M2 B2 r4の総bias≈0.003725912<0.005にもかかわらず、現規則では0.005が不適格。
  元の規則や正式statusを変更せず、本文でこの区別を明示する。
- B0 pure discard/PFはmissing。相殺機構を実証したと書かない。
- primary/secondary、method間/family内不確かさを分ける。
- RZ-workをClifford+T総資源、物理時間、energy/RPE最終資源と呼ばない。

作るもの:
1. 主張・根拠path・commit・証拠階層・限界・対応図の表。
2. 保存された範囲での主図4枚と補足表。既存parser/集計を必要な範囲だけ再利用する。
3. Abstract、Introduction、Methods、Results、Discussion/Limitationsの通し原稿。
4. 先行研究との差分表。文献の内容を確認できなければ未確認とする。
5. 完成判断に必要な未解決事項だけの短い一覧。

新しいNPZ/NPY/pickle、runtime/cache/registryへアクセスしない。
新signal、ground-state、trajectory、build/compile、追加精度点、候補追加、全repo testsは行わない。
元M1/M2/PM-0/PM-1/PM-2のresult/status/source/manifestを変更しない。
新しい研究status、execution authorization、PM-3を自動作成しない。
Track Bの未commit変更へ触れない。
保存値の欠損・矛盾を見つけたら推測で埋めず報告する。
```

## 13. 参照資料

### リポジトリ（全て証拠commitで固定）

[R1] `docs/pr2_pm2_precision_resource_result_validation.md`。

[R2] `docs/research/pr2_pm2_precision_resource_contract_v1.md`。

[R3] `artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/summary.json`。

[R4] `docs/research/pr2_post_m2_evidence_attribution.md`。

[R5] `docs/pr2_pm1_discard_result_validation.md`。

[R6] `docs/pr2_matched_accuracy_m2_transfer_result_validation.md`。

[R7] `artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/representative_decomposition.csv`。

図表化時には、同directoryの`precision_ledger.csv`、`eligibility_boundaries.csv`、`P_envelope.csv`、`claim_audit.json`と既存報告を併用する。本レビューで全行を独立再検証済みとはしない。

### 一次文献（2026-10-05に重点照合）

[W1] J. Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum 7, 020332 (2026), DOI `10.1103/ynxb-p2xq`, arXiv `2503.05647`。重点：Sec. VII、Appendix D.2、Figs. 10, 19, 20。Fig. 19/20のtruncation比較はPauli表現のground-state energy errorであり、今回と同じfinite-signal図ではない。

[W2] M. Hagan and N. Wiebe, *Composite Quantum Simulations*, Quantum 7, 1181 (2023), DOI `10.22331/q-2023-11-14-1181`, arXiv `2206.06409`。

[W3] P. A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv `2606.30741v1` (2026)。重点：factorization、SPRINT、remainder strategy、implementation cost。

[W4] O. Oumarou et al., *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*, Quantum 8, 1371 (2024), DOI `10.22331/q-2024-06-13-1371`, arXiv `2212.07957`。

[W5] S. Kanasugi et al., *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*, arXiv `2603.22778v2` (2026)。重点：Sec. II–V。本文ではPauli-LCUを中心に扱う。

[W6] D. Cugini, T. A. Atif, Y. Subasi, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv `2603.13495v1` (2026)。重点：Sec. IIのnet cost、Sec. IIIのbias、Conclusions。

[W7] W. A. Simon and P. J. Love, *Halving the Cost of Controlled Time Evolution*, arXiv `2511.13855v1`。重点：Sec. III。任意角rotation数の改善を総compiled-RZ一律改善とは扱わない。

以上は重点的な新規性照合であり、網羅的systematic review、同一研究の不存在証明、採択の保証ではない。
