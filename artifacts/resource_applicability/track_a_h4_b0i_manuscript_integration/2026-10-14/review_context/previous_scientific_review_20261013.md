# Track A：H4 B0+Iの対照採用・実測範囲・主要claimに関する科学的レビュー

- 作成日：2026年10月13日（日本時間）。
- 開始承認：利用者の「レビューを開始して」。
- レビュー対象：H4 geometryのB0+I保存値診断、技術統合、検証用adapter、6 full-wrapper compile準備。
- Repository：`HIROMU1015/Partially-Randomized-Trotter`。
- 準備branch：`track-a-h4-b0i-compile-preparation-20261013`。
- 固定SOURCE：`8f6c1bc8ceeeffddbecf170ef6ddb0fe6d18b63b`。
- 固定REVIEW：`34ad305d55a797b57d00aebed99fc03a49a3ed2f`。今回GitHubからbranch HEADとの一致を確認した。
- 先行技術統合：`f6b0cb5e2ddabd2d7919d0822c58324c7338bac0`。
- B0+I保存値診断：`c81f7326dde3f40686b3caefe05e2f544d8d1826`。
- 元H4 geometry結果：`7c4483378047b57f514c47dad98905e3f2a9a756`。
- 元科学SOURCE：`d0726e946f89f8c30a08d383fb8ff48b363629ec`。
- レビュー状態：**完了。B0+Iを正式な追加評価対象として採用し、直近の6回路実測を科学的に推奨する。実測費用と勝敗は未確定。**
- 実行認可：**本書はユーザーの科学計算・CPU/RAM使用・launch承認を代行しない。新grantは発行せず、既存mandatory STOPを維持する。**
- H8：利用者から計算中と報告されている。今回H8のライブ監視、停止、source・plan・grant変更は行っていない。

---

## 0. 結論と今回確定する判断

### 0.1 B0+Iを正式な追加対照として採用する

**B0+Iを、単なる参考感度に留めず、H4の追加資源比較において費用を評価する決定論的対照として採用する。** ただし「比較に入れる方法の採用」と「計算済みの正式な費用値・勝敗の認定」は別である。既存B0、元の登録集合、元結果、source、図表の歴史的versionは変更しない。

理由は、B0+Iが真の基底energy・成功したgeometry・PRの出力へフィットした補正ではなく、既知のDF Hamiltonianに元から含まれるidentity係数を保持するだけの対照だからである。現在のB2/B3はこのidentity係数を既に定数項として保持する。PRの非identity tail補完による利益を論じる際、安価に保持できる既知位相まで捨てるB0との比較だけでは、その利益の帰属が不十分になる。[R2,R5,R6]

B0が登録仕様どおりであることと、より強い比較対照が必要であることは両立する。B0を実装バグと認定しない。

### 0.2 直近の実測は準備済み6回路とする

0.70 Å/p5/q1、0.80 Å/p5/q1、1.40 Å/p3/q1のcosine/sine、計6 full-wrapper compileを、既存の固定SOURCE・入力・compiler・one-shot契約で測ることを推奨する。[R1,R8,R9]

目的は、(i) 新adapterの実H4 lowering・compileが完走するか、(ii) 代表3構成の軸別費用がいくつか、(iii) 追加費用ゼロの仮定が代表点の判断を変えるか、(iv) 次の取得の実資源をどう見積もるか、である。6回路だけで全397点や全補助集合の最小費用を確定しない。

### 0.3 全面地図の標準的な完成案は240回路

原稿で6距離×302精度の**B0+Iを含む追加比較地図**を示すなら、元のB0 domainに対応する120構成（6距離×20構成）×2軸の240回路を評価する案を標準的な完成方針とする。科学的な対象domainの選び方はここで明確にし、B0+Iが勝った場合だけ拡張する方針にはしない。

ただし、240回路の実装・資源契約・実行認可は未完了である。まず6回路で実装と資源を確認し、追加取得が可能な予算をCodexが具体化する。6→240へ自動進行しない。完全な再利用bindingが成立すれば残り234回路、成立しなければ新240回路を予約する。旧grantや未完了予約は再使用・返却しない。[R8,R10]

10回路案は「397点でΔ=0最小だった5構成を測る限定版」であり、全面地図の完成案の代わりではない。6→10→240という段階を必須にはしない。資源制約から固定5構成の報告に研究scopeを絞るときの代替案として残す。

### 0.4 H4の追加乱数標本は、今は優先しない

H4全候補の独立32 trajectory再取得、希少次数の一律pilot、172件のB=1丸めの完全復元は、今回の次作業にはしない。いま結論に直接効く不明量は新しい決定論対照のcompiled費用であり、PRのsamplingを増やしてもそれは分からない。

### 0.5 H8は継続する

H8の既存承認範囲の計算を中断・変更しない。今回確認したB0の実装はH4 geometry系列である。H8に同じ処理が存在するか／既にscalar保持しているかを本レビューで断定しない。H8完了後、比較対照の定数項処理を条件対応として記載するが、計算中の候補やsourceへ後付け変更を行わない。

---

## 1. 資料取得とレビュー範囲

### 1.1 今回の直接確認

GitHub接続から、以下を固定refで取得した。

- 準備branchのremote HEADと準備報告。
- B0+I技術統合報告、B0+I診断の代表値、6回路registry。
- 新adapter、実compileへ渡すruntime経路、位相・wrapper人工テストのsource。
- 元H4の`signal.py`と`circuits.py`。
- 6/10/240の予算・対象scope、実行・STOP契約。
- Taylor次数保存状況と統計的限界に関する技術統合の該当部分。

前回のH4科学レビューMarkdownも読み、既存方針を引き継いだ。外部情報はPR原論文の出版情報・abstract、Qiskit 1.3 PhaseGateの定義に限定して補足した。先行研究の網羅的再調査は行っていない。

### 1.2 実行した算術と、していない検証

公開スカラーを手動転記した標準ライブラリscriptで、ε=.05の代表3点について、旧軸別RZを使った参考費用、Δの交点、B2比、報告SEを使う記述的感度を再計算した。0.70 Åの保存complex signalについて位相回転・2軸bias・整数shot数も再計算した。結果は保存値と一致した。

**新しい分子計算・signal simulation・sampling・eigensolver・compile・transpileは実行していない。44人工テストをこの環境で再実行していない。全76ファイルのbytes/SHA、全1,812点・4,803近接行・全rawを独立に再監査したとも主張しない。** それらの完了はCodex保存報告の証拠scopeとして扱う。

B0+I実compile値がないことは、今回の「実測を採択すべきか」を判断するための資料欠損ではなく、まさに次の取得対象である。巨大raw・private NPZ・サーバー全環境の複製は、この採否判断に必須とはしない。

### 1.3 根拠の区分

本書では、保存結果の値を「報告値」、それからの再算術を「本レビューの算術」、sourceからの数学的帰結を「導出」、採否・scopeの選択を「今回の研究判断」と区別する。未測定のB0+I費用は最後まで未測定として扱う。

---

## 2. 研究全体の中での位置付け

Track Aの主目的は、固定Hamiltonian表現・回路実装・有限時間signal taskにおけるPRの資源量と適用範囲を明らかにすることである。PRが必ず勝つこと、全supportの母平均費用上界を証明すること、あるいは新しいPR法を必ず作ることを必須の成功条件にしない。

H4 geometryは、固定DF12・ordinary controlled・structured Gaussian・S2主体の6距離／302精度gridで、登録B0/B1/B2/B3に対しB2/prefix3が点最小となった。H6は別のnative系列でS2/S4 full PF対照、探索後の独立確認、局所的な希少次数費用pilotを備える。H8はサイズ方向の資源比較として別の承認scopeで進んでいる。[R2,R11]

今回のB0+Iは、前回レビューで特定した「より安価な対照を省略していないか」への診断から生じた。これはPRの数学的構成が間違っていたという話ではなく、**資源的な利益をどの対照に対して、どの要因に帰属させるか**という論点である。

大量の費用標本や1,812点という件数は、比較対照が不十分である可能性を解消しない。逆に、対照を強化して一部の領域でPRが敗れることは、元データを無価値にしない。利益の残る領域・縮小する領域を分けて説明する研究へ進める。

---

## 3. B0+Iの数理的妥当性

### 3.1 DFブロックのidentity係数

元実装が扱うDFブロックを

\[
A_\ell=\lambda_\ell\left(\sum_i\eta_{\ell i}n_i\right)^2,
\qquad n_i=(I-Z_i)/2
\]

とする。\(n_i^2=n_i\)を用いると、identity係数は

\[
c_\ell=\frac{\lambda_\ell}{2}\sum_i\eta_{\ell i}^2
+\frac{\lambda_\ell}{2}\sum_{i<j}\eta_{\ell i}\eta_{\ell j}
=\frac{\lambda_\ell}{4}\left[\left(\sum_i\eta_{\ell i}\right)^2+
\sum_i\eta_{\ell i}^2\right].
\]

元`diagonal_coefficients`と一致する。下三角から定義されるHermitian行列\(G_{\ell,L}\)を使えば、

\[
c_\ell=\frac{\lambda_\ell}{4}
\left[(\operatorname{Tr}G_{\ell,L})^2+\operatorname{Tr}(G_{\ell,L}^2)\right].
\]

これは元`eigh`の下三角規約と整合する数学的identityであり、upper/lower平均化や微小片pruneを新たに導入しない。実際の浮動小数点eigensolverとのbit単位の一致をこの恒等式から主張しない。新adapterの計算はこのlower規約に従う。[R5,R6]

保持prefixをpとし、\(c_{\rm tail}=\sum_{\ell\ge p}c_\ell\)。保持済みone-body/DFブロックのidentity項は二重に足さない。

### 3.2 単に既知の成分を保持する対照

\[
H=H_{\rm kept}+c_{\rm tail}I+V_{\rm tail}
\]

に対して、元B0は非保持ブロック全体を捨てる。B0+Iは\(H_{\rm kept}+c_{\rm tail}I\)を使い、\(V_{\rm tail}\)は捨てたままとする。

\(c_{\rm tail}\)はDF係数から定まり、参照energy、PRの費用、選抜結果を最適化対象として決めない。したがって、既知情報を利用する対照として妥当である。今回の「対照を着想した時点」は事後的だが、係数をtruthへフィットしているわけではない。この二点は別々に記録する。

### 3.3 なぜ自然な対照なのか：identity射影としての説明

以下はsourceの定義からの本レビューの導出であり、既存報告の数値結果を追加したものではない。full Fock空間の次元をd、捨てる演算子をAとすると、そのPauli identity係数は\(c=\operatorname{Tr}(A)/d\)である。\(A=cI+V\)、\(\operatorname{Tr}V=0\)だから、

\[
\|A\|_F^2=d|c|^2+\|V\|_F^2.
\]

scalar approximationの範囲では、\(cI\)を保持することは、full空間Hilbert–Schmidt誤差のidentity部分を取り除く自然な選択である。

ただし、これをoperator-norm改善、指定stateのsignal改善、あるいは粒子数sector内での最適scalarの証明へ拡張しない。full空間のtrace係数と固定粒子sectorの平均は一般に異なる。今回sector最適scalarやstate期待値shiftへ変更する提案は行わない。

### 3.4 回路・signalの関係

identityはすべてのsystem演算と可換であり、元B0のPF近似を\(\widetilde U_0(T)\)とすると、

\[
\widetilde U_I(T)=e^{-iTc_{\rm tail}}\widetilde U_0(T),
\qquad z_I=e^{-iTc_{\rm tail}}z_0.
\]

有限qの元PF構成に対してもこの関係は成立する。PF分割誤差の追加近似を導入していない。

ただしHadamard wrapperでは制御枝間のrelative phaseである。controlled evolutionを

\[
W_0=|0\rangle\!\langle0|\otimes I+|1\rangle\!\langle1|\otimes\widetilde U_0
\]

とすれば、

\[
W_I=\bigl(P(-Tc_{\rm tail})\otimes I\bigr)W_0.
\]

全wrapperのglobal phaseとして捨ててはいけない。Qiskit PhaseGateは\(\operatorname{diag}(1,e^{i\theta})\)であり、上の符号と一致する。[P2]

元builderは各q stepの冒頭に\(P(-\delta\,\text{constant})\)を置く。新adapterはその角度にtail scalarを加える。systemの保持ブロック、basis、control、PF順序を変更せず、pre-transpile命令数を増やさずに実装できる構成になっている。[R5,R6]

### 3.5 改善は必ずしも単調ではない

位相保持で\(|z_I|=|z_0|\)は保たれるが、targetとの距離は一般に増減し得る。任意のtarget tに対し、

\[
|e^{-i\theta}z-t|^2-|z-t|^2
=-2\operatorname{Re}\{(e^{-i\theta}-1)zt^*\}.
\]

右辺の符号は固定されない。既存B0で起きていた誤差の打消しを壊す場合もある。したがって、B0をB0+Iへ一律置換するのではなく、元B0とB0+Iを併記して比較する。

また、位相回転は\(|z|\)を変えないので、非identity tailを捨てたことによる振幅損失・状態混合・長時間誤差まで修復する方法ではない。B0+Iをfull Hamiltonianの正確なsimulationや最終energy推定の解と呼ばない。

---

## 4. Source・人工検証から判断できること

新`adapter.py`はB0のpreparationをコピーし、保持DF係数に触らずconstantだけを更新する。二重適用を入力schemaで拒否し、独立method/template/namespaceとvalidation fingerprintを持つ。numeric circuit fingerprintが同じになるzero-scalar特殊例と、科学的task identityの分離も区別している。[R5]

元`signal.py::_prepare`ではB2/B3だけがtail identityをconstantへ移すこと、元`circuits.py`ではcontrolled phase、cosine H-E-H、sine H-E-Sdg-Hが実装されていることを直接確認した。[R6,R7]

人工テストsourceでは、q1/2/4/8、scalarの正負・zero、low/high branchのunitary関係、cosine/sine期待値、元B0不変、double count拒否を検査する構造を確認した。[R12] Codex報告の44件PASSを、実H4の新compile成功や全process raceの証明とはしない。

runtimeの実費用取得経路は、承認後に必要な小さい凍結memberだけを読み、保持ブロックの既存loweringを使い、各axisの新full wrapperをcompilerへ渡してmetricsを得る。旧B0費用cacheを新しい実測として代用しない。新signal solver、tail eigensystem、SCF/DF/state再生成を呼ばない設計である。[R13]

**今回読んだ科学的に重要な経路に、B0+Iの数学的定義と矛盾するblocking issueは確認されなかった。** これは全source・環境・native processを独立再現した認証ではない。起動時のsource/profile/input/fresh gateは別に必要である。

現固定compiler pipelineの出力RZ差は、位相角の融合・削除・再合成によって変わり得るため未測定である。\(\Delta=0\)、\(\Delta=1\)、\(\Delta\ge0\)のいずれも実測事実にしない。genericなPhaseGateとRZの関係や、別の回路構成による上界を、そのまま固定adapterの実compile値として代用しない。

---

## 5. 保存値診断の影響と代表点の検算

### 5.1 397点の意味

全1,812 grid点のうち397点、約21.91%で、旧B0の軸別RZをそのまま使う\(\Delta=0\)参考値がB2点最小を下回る。これは397回の独立実験でも、実B0+Iの勝利数でもない。同じ保存signal・費用を302精度へ再会計した感度である。[R2,R3]

| geometry Å | Δ=0の参考逆転点数 | 該当するgrid精度の範囲 |
|---|---:|---|
| 0.70 | 178 | 0.017247616–0.100000000 |
| 0.80 | 103 | 0.023505464–0.064443967 |
| 0.90 | 29 | 0.026763740–0.035397800 |
| 1.10 | 0 | なし |
| 1.40 | 87 | 0.042793207–0.100000000 |
| 1.60 | 0 | なし |

これらはgrid上の領域であって、連続epsilon全域の厳密境界ではない。代表ε=.005は上の逆転領域に含まれない。しかし、それだけで高精度域のB0+I実費用に対する優位が確定したわけでもない。

### 5.2 ε=.05の代表3構成

報告された旧B0の軸別RZをC_old、新対照の未知差を\(\Delta_c,\Delta_s\)とする。

\[
G_I=N_{I,c}(C_{\rm old,c}+\Delta_c)+N_{I,s}(C_{\rm old,s}+\Delta_s).
\]

以下の費用は百万RZ-work。B0+Iの列は未測定の\(\Delta=0\)参考値である。[R3,R4,R8]

| geometry／構成 | B2 G | B0+I G(0) | G(0)/B2 | B0+I shot数 | 共通Δの交点 |
|---|---:|---:|---:|---:|---:|
| .70／p5 q1 | 206.510352 | 152.874144 | 0.740274 | 18,084 | 2,965.948 RZ/shot |
| .80／p5 q1 | 171.335623 | 152.912878 | 0.892476 | 18,080 | 1,018.957 RZ/shot |
| 1.40／p3 q1 | 102.934381 | 98.907746 | 0.960882 | 18,957 | 212.409 RZ/shot |

これらのG(0)と交点を本レビューで再計算し、保存値との一致を確認した。0.70 Åと0.80 Åには大きめの追加費用余地があり、1.40 Åはより小さい。この異なる感度を含む3構成を測る意義がある。

一般の軸別費用の判定条件は、

\[
N_{I,c}\Delta_c+N_{I,s}\Delta_s<G_{B2}-G_I(0).
\]

実測後はこの式に各axisの費用を入れ、共通Δを強制しない。B2はMonte Carlo費用点推定であり、上の不等式が成立しても母平均の勝敗認証ではない。

参考として、B2の報告点推定から2SEを引いた値と比較する共通Δの交点は、それぞれ約2827.00、867.88、91.20 RZ/shotである。これは記述的な感度にすぎず、同時信頼区間・選択バイアス補正・未知tail上界ではない。

### 5.3 位相保持が効く理由の具体例

0.70 Å/p5/q1の保存tail scalarは0.01363699750131806、T=.8なので位相は-0.010909598001054448 rad。

\[
z_0=-0.13124135477680748+0.9898046302132318i,
\]
\[
z_I=-0.12043538831129537+0.9911774898763147i.
\]

保存targetと比較すると、B0+Iのcosine/sine biasは0.005886183820368124／0.0022400858442358507。元H4の固定shot式でN_cos=10,092、N_sin=7,992となる。ここは本レビューでも位相・bias・整数shotを算術検算した。[R4,R7]

影響の中心は「回路が大きく変わる」ことではなく、既知位相がaxis biasと残りの統計的headroomを変えることである。0.80 ÅではB0+I参考shot18,080に対しB2は28,102であり、1-shot回路がB2より高くても総参考費用が低くなり得る。[R3]

1.40 ÅではB0+I参考点最良がp3/q1で、旧登録B0最良のp5/q1とは異なる。異なる候補間の比較を、同じ回路に対するphase修正効果だけと説明しない。

---

## 6. 保存signalの回転と、測定回路の変更を混同しない

今回の保存値診断は、B0+Iの数学的signal平均を、既存complex signalから解析的に求めている。実compileで取得するのは、その新しい平均を測定する回路のRZ/CX/depth等である。これは新しいSCFやsignal行列指数を全面再実行する必要がないことを意味する。

一方、旧B0の量子測定結果を古典的に回転するという**別の測定方式**を採るなら、誤差・shot会計も対応させる必要がある。\(\theta=Tc_{\rm tail}\)として、

\[
\widehat x_I=\cos\theta\,\widehat x+\sin\theta\,\widehat y,
\qquad
\widehat y_I=-\sin\theta\,\widehat x+\cos\theta\,\widehat y.
\]

旧axisの推定誤差上限が\(e_x,e_y\)なら、例えば新xの誤差は\(|\cos\theta|e_x+|\sin\theta|e_y\)に依存する。新しいaxisを直接測る場合のHoeffding配分を、旧回路のsampleへ無条件に流用できない。

**今回採択するのは準備済みadapterによる回路内位相保持と両wrapperの測定であり、古典後処理だけで「回路費用ゼロ・新shot数もそのまま」とする方式ではない。** 後処理方式を別に比較することは可能だが、今回それを追加の必須研究課題にはしない。

---

## 7. 6／10／240回の情報価値と選択

| 案 | 対象 | 得られる情報 | 本レビューの採否 |
|---|---|---|---|
| 6 | .70 p5/q1、.80 p5/q1、1.40 p3/q1の両軸 | 実装・実資源・代表3構成の実CとΔ | 直近の実行推奨。準備済み契約を使う |
| 10 | 上記＋.70 p4/q1、.90 p5/q1 | Δ=0逆転領域で最小だった5構成の費用 | 限定版の代替案。必須中間段階にしない |
| 240 | 6距離×元B0対応20構成×2軸 | 同じ登録domainの全B0+I費用と追加比較地図 | 全面地図を完成する標準案。新scope・予算・実行承認が必要 |

### 7.1 なぜ6回を先に測るか

すでにSOURCE・registry・input/profile・人工テスト・実行契約が固定されており、ここでさらに実装準備を増やすより、未知の実compiler費用を取得する方が判断に直結する。3構成は参考余裕の大きい点と小さい点を含む。6件はB0+Iが勝つことを実行成功条件にしない。

### 7.2 なぜ10回だけで全面結論を確定しないか

5構成はΔ=0の仮定から選ばれている。実Δが候補・axisごとに異なれば、別候補が最小へ移る可能性がある。10回で測定した候補の費用を全precisionへ再会計することはできるが、未測定候補を含むfamily最小は決まらない。

### 7.3 全面地図に240回を推奨する理由

費用はgeometry／prefix／q／axisごとに測ればよく、302精度ごとにcompileする必要はない。120構成を測った後、保存phase-rotated signalとshot規則で各precisionへ投影する。

240回は元74,784 logical wrappersの約0.321%という件数規模だが、wall timeも0.321%で済むという予測ではない。全q・prefixの実資源は6件から厳密には予測できず、資源・監視・容量の再見積もりは必要である。[R8]

B0+Iが勝つ点だけを実測して「PRが負けた地図」を作ることも、PRが勝つ点だけを残して「PRの全面優位」を作ることも避ける。完成版を主張する場合は、同一domainで全対照を評価する。

### 7.4 段階間の承認

6件後は既存契約どおりmandatory STOPし、native結果・費用・資源を公開する。成功時の240向け実装・予算整理は、このレビューで採択した対象domainの範囲内ならCodexに任せられる。単なるsource改訂・テストPASSのたびに科学的採否を再審議しない。

ただし、6件のgrantを拡張したり、成功を理由に残234／新240件へ自動進行したりしない。新scopeに対するユーザーの実行承認は別途必要である。旧6件の結果を再利用する場合もcomplete/source/input/compiler/axis/phase/namespaceの明示mappingを必要とする。

資源制約によって240件が不適切と判明した場合は、5構成の限定結果へscopeを縮小し、全面family最小や全1,812点の新しい勝敗率を主張しない。その変更が研究の主要な着地点を変える場合だけGPTへ戻す。

---

## 8. 統計・未保存次数・追加samplingの判断

### 8.1 H4の現在の不確かさ

H4は同じ32 paired trajectoryで候補選抜と費用推定を行う。302precisionは同じ保存値の再会計である。±2SEは記述的で、多重選択・希少event未観測を保証するものではない。新B0+Iのcompiled費用を測っても、B2の母平均費用が保証されるわけではない。[R11]

### 8.2 未保存と未観測の区別

技術統合は、H4のevent次数・component列をNOT_SAVEDと報告している。これは「すべて0だった」という観測ではない。保存Bが1に丸められた172候補も、tauや次数確率が一意に復元できないので未解決のままである。B=1を元のtau=0や非zero確率0と同一視しない。[R11]

seedがあれば、元のsource・乱数engine・component確率・入力がすべて再現できる条件下で履歴再構成を検討する余地はある。しかし現在のscopeでは再samplingは禁止されており、今回は実施していない。将来replayを行っても、新しい独立費用標本や母平均上界が得られるわけではない。必要性の薄い再構成を次の必須工程にはしない。

### 8.3 今追加しないもの

- 全1,164 stochastic候補の新しい32 trajectory確認。
- 全geometryのK4 order4条件付き費用pilot。
- 172候補の正確な確率を復元することだけを目的とした新計算。
- H6で観測した0.00734%をH4へ適用する補正。

B0+I実費用を得た後、主張の中心となる境界でB2の標本変動が判断を左右する場合、特定候補をfreezeした独立費用確認を検討する。厳密な母平均優位を主張するなら、独立標本だけでなく希少support・costの扱いを含む別の統計設計が必要になる。その強い保証を本研究の完成条件にはしない。

---

## 9. 原稿v0.3のclaimと研究の着地点

### 9.1 維持できる事実

- 元の登録B0/B1/B2/B3集合において、保存grid全1,812点の点最小はB2/prefix3。
- geometryとprecisionによってq/r/Kの最小点が変わり、費用をshot数と有効1-shot費用へ分解できる。
- H6の別系列で記録されたPR対full S2/S4 PFの費用比は、今回のH4対照追加によって数値的に書き換わるものではない。

### 9.2 主要な表現は限定し直す

「全条件でPRが決定論的方法より優れる」という無限定の中心表現は採用しない。元の結果を歴史的な登録集合内比較として示し、B0+Iを追加する動機と、追加対照の実測scopeを明記する。

397点の感度だけを小さな脚注へ移し、元の1,812点を実用的な広い優位性として前面に出す構成は推奨しない。潜在的に重要なbaselineを把握した後は、その情報を主な解釈へ反映する必要がある。ただし実cost未取得の今「397点でB0+Iが実際に勝つ」と書き換えるのも誤りである。

### 9.3 提案する研究の中心

**既知scalarの保持を含む対照と比べたとき、PRの非identity tail補完が、どのgeometry・precision領域で測定込み資源を改善するか。**

これは比較対照を改善した後の問いであり、まだ答えが得られたという意味ではない。高精度域で利益が残る、特定geometryでのみ残る、ほぼ消える、いずれの結果でも、その費用要因を説明して研究のscopeを判断する。

### 9.4 新規性

PR原論文は既に量子化学・単一ancilla位相推定の具体的資源評価と水素鎖のサイズ依存を扱っている。[P1] したがってPRを初めて具体的に評価した、H4/H8を増やしたこと自体が新規、とは主張しない。

B0+Iそのものも、既知identity項の保持であり、新しい量子アルゴリズムとして売り出すことを推奨しない。追加の価値は、具体的実装・bias・normalization・shot・compiler・統計設計を接続した公平な比較から、何が本当にPRの利益なのかを切り分ける点にある。

B0+Iを入れると主要な利益が広く消え、別の設計知見も乏しいなら、H4を単独の大きなPR優位性論文へ拡張せず、baseline-sensitive benchmark／否定的診断としてscopeを縮小する。逆に条件依存の機構が明確になるなら、H4 geometryとH6/H8を異なる証拠群として統合する価値が残る。独立論文成立を現時点で保証しない。

S4以上のPF、state preparation、ノイズ、fault-tolerant合成、最終energy/RPEは今回の比較へ追加されていない。B0+Iを取得しただけでその不足まで解消したことにしない。

---

## 10. H8との関係

H8は利用者報告上、別Codexで計算中である。このレビューではH8の新結果を取り込まず、ライブprocessを確認・操作していない。

- 承認済みのH8実行をそのまま継続する。
- H4 B0+Iを理由にH8のgrid・phase・compiler・source・grantを途中変更しない。
- H4のΔや397点をH8へ外挿しない。
- H8完成後の解釈では、B0のscalar保持がどう実装されているかを同系列のsourceで確認する。
- H6/H8の「PR対full PF」の既存比と、B0+Iを含む「最良deterministic全体」の比を区別する。

H4 geometryのordinary/structured系列と、H6/H8のsymmetric-directional/native系列を、compiler labelだけで同一費用としない。geometry方向とサイズ／実装方向の二つのcase-study群を分けて提示する方針は維持する。[R2,R11]

---

## 11. 直近6回路の実行準備に関する判断

### 11.1 科学的には既存固定SOURCEで実測へ進める

SOURCE `8f6c1bc8...`、REVIEW `34ad305d...` の数学的定義・対象3構成・固定phase・full-wrapper費用というscopeを、直近の実測対象として採用する。これ以上、同じ保存値解析や同じphase人工検証を繰り返すことをレビューから要求しない。

### 11.2 資源値は上限とfresh gate

保存proposalはdriver/worker各8 GiB、observer AS256 MiB/RSS64 MiB、1 worker、内部thread1、stage1h/task10min、出力256 MiBである。admission120.25 GiBは旧保守floorを引き継いだ起動条件で、pilotが120.25 GiBを必要とする実測結果ではない。[R8]

CPU候補0/1/2は予約・使用権・現在の空き保証ではない。H8と共存するため、ユーザーのCPU/RAM使用承認と、起動直前のCPU/physical core・memory/PSI/OOM・disk/inode/quota検査を満たす必要がある。満たさない場合はH4 pilotを起動せず、H8を停止・移動して空きを作らない。

### 11.3 既存STOP条件を維持

新6件のユーザー実行承認を得た後、plan/auth/reviewのdigestと固定SOURCE・input・compilerを結ぶ。科学レビュー採択だけでdraft flagsをtrueへ変えない。実行に必要な最終binding確認はCodexの技術作業であり、同じ科学レビューの再実施ではない。[R9]

成功時も失敗時もmandatory STOPし、6件の結果・partial・first cause・native exit/reap・observer終端・call/charge/wall carryを保存する。契約上observerは全compute後に意図的SIGTERMで終了するため、正常observer codeはworker/coordinatorと同じexit0ではない。実際のnative終了を記録し、コードから終了値を推測しない。[R9]

旧actual/reserved73023に6を加える最大73029は旧上限74805以内という計画であるが、余裕があること自体は旧grant再利用や240への自動拡張の許可ではない。[R8]

---

## 12. 次作業、GO／STOP、レビューを再び行う条件

### 12.1 次の担当

**次の担当はCodex 1。ただし、直近6回路の起動はユーザーによる別途実行承認後。** 起動前は完成済みsourceを保全し、承認されたCPU/resource/commandとの最終binding以外に不要な準備を追加しない。

### 12.2 6回路実測後の作業

1. 各axisの実C、旧Cとの差、source/input/phase/compiler/FPを保存する。
2. 保存B0+I signalと元shot式を使い、代表ε=.05と、その測定済み固定3構成の全302精度への費用を算術再評価する。
3. B2との点差・記述的SE・未確定の母平均を区別する。非測定B0+I候補を含むfamily最小とは呼ばない。
4. Wall/RSS/output/guardを記録し、120構成の完成案に必要なrunner・資源予算・complete-only reuse案を整理する。
5. 結果・監査を公開しmandatory STOP。追加234／240件の実行は新scopeの別承認を待つ。

### 12.3 GPTへ戻す条件

- 位相関係・source/input対応・元targetとの数値整合が崩れた。
- 実compiled費用が想定から大きく外れ、別compiler policyや別測定方式の採用が必要になる。
- 資源制約で120構成の完成を断念し、原稿のscopeを実質的に縮小する必要がある。
- B0+Iを含む結果から最終的なgeometry／precisionの優劣領域や論文の主要claimを確定する。
- H8の重要結果を統合し、研究継続・縮小・着地点を判断する。

単なる6件成功・新commit・人工テストPASSだけを、新しい同範囲の科学レビューの理由にしない。6件で一部の順位が変わることは既に本レビューが想定する結果であり、その事実だけで既定の技術的整理を即中断させない。ただし旧one-shot後のmandatory STOPと新実行承認は維持する。

### 12.4 最終決定表

| 判断対象 | 今回の結論 |
|---|---|
| B0+Iの数学的妥当性 | 固定定義で支持 |
| 正式な追加対照への採用 | 採用。旧B0・旧集合を保全し、追加比較を区別 |
| 実compiled費用・B2との実測勝敗 | 未確定 |
| 直近6回路の科学的必要性 | 実測を推奨 |
| 10回路 | 限定scopeの代替。必須中間段階ではない |
| 240回路 | 全domain地図の標準完成案。技術予算・実行承認は別 |
| 新しいH4 sampling／rare pilot | 現時点では要求しない |
| 原稿の「全点でPR優位」 | 元登録集合内の歴史的結果に限定。追加対照の結果と分離 |
| H8 | 承認済み範囲で継続、途中変更なし |
| 現在のgrant／STOP | 未承認を維持。本書だけではlaunch不可 |

---

## 13. 前回レビューからの変更点

前回は、B0がtail identityも捨てる点を特定し、保存値のphase感度を最優先にした。その後、397点の参考逆転、全grid分類、source上の角度吸収、独立adapter、44人工検査、固定6件の実行契約が揃った。

その新証拠を受け、今回は**「B0+Iの採否は保留」から「追加対照として採用し、実費用を取得する」へ判断を進める**。元のB0が仕様どおりであること、元集合の点比較が有効であることは撤回しない。

240件の完成domainを標準案として示すことで、6→10→240という細分化と同じ採否レビューの反復を避ける。ただし技術的なfeasibility確認、資源予算、各実行のユーザー承認を省略するものではない。

研究を「PRは必ず有利」とする方向へ維持するのではなく、scalar保持を含む合理的対照の上で利益が残る領域を明らかにする方向へ主張を正す。利益が消える場合の縮小・否定的結果の提示も許容する。

---

## 付録A. 算術再計算のscope

添付補助directory `h4_b0i_review_arithmetic_20261013` のscriptは、公開診断から手動転記した代表3点だけを扱う。自動ダウンロードやsource実行はしない。

- 3点のGI(0)一致。
- 3点の共通Δ交点一致。
- 3点の比・削減率・記述的2SE感度の再計算。
- 0.70 Å/p5/q1の位相回転、axis bias、2軸の整数shot一致。
- 397/1812の割合の算術。
- 新signal simulation・sampling・eigensolver・compile：0。

`old_registered_best_B0_G`は元登録集合の最良B0費用である。特に1.40 Åでは旧最良p5と補助p3が異なるため、この値との差を同一構成のphase効果と解釈しない。

---

## 付録B. 証拠索引

以下のR番号は本レビューで実際に取得・参照した資料。large JSONの範囲取得や準備reportを使った箇所について、全bytesの独立検算を主張しない。

- **[R1]** [B0+I実装・6 compile準備](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34ad305d55a797b57d00aebed99fc03a49a3ed2f/docs/research/track_a_h4_b0i_compile_preparation_20261013.md)
- **[R2]** [H4 geometry技術統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f6b0cb5e2ddabd2d7919d0822c58324c7338bac0/docs/research/track_a_h4_geometry_technical_integration_20261013.md)
- **[R3]** [B0+I診断・代表18点スカラー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c81f7326dde3f40686b3caefe05e2f544d8d1826/artifacts/resource_applicability/track_a_h4_geometry_scalar_diagnostic/2026-10-13/diagnostic_summary_v1.json)
- **[R4]** [6 wrapper固定registry](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34ad305d55a797b57d00aebed99fc03a49a3ed2f/artifacts/resource_applicability/track_a_h4_b0i_compile_preparation/2026-10-13/registry.json)
- **[R5]** [独立adapter](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8f6c1bc8ceeeffddbecf170ef6ddb0fe6d18b63b/src/trottertracks/resource_applicability/h4_b0i/adapter.py)
- **[R6]** [元H4回路・DF係数・wrapper](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0726e946f89f8c30a08d383fb8ff48b363629ec/src/trottertracks/resource_applicability/h4_geometry/circuits.py)
- **[R7]** [元H4 signal・RTE・shot会計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0726e946f89f8c30a08d383fb8ff48b363629ec/src/trottertracks/resource_applicability/h4_geometry/signal.py)
- **[R8]** [6/10/240の予算とscope](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34ad305d55a797b57d00aebed99fc03a49a3ed2f/artifacts/resource_applicability/track_a_h4_b0i_compile_preparation/2026-10-13/BUDGET_AND_SCOPE.md)
- **[R9]** [実行・fresh gate・STOP契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34ad305d55a797b57d00aebed99fc03a49a3ed2f/artifacts/resource_applicability/track_a_h4_b0i_compile_preparation/2026-10-13/EXECUTION_CONTRACT.md)
- **[R10]** [Codexから返された科学判断事項](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f6b0cb5e2ddabd2d7919d0822c58324c7338bac0/artifacts/resource_applicability/track_a_h4_geometry_technical_integration/2026-10-13/SCIENTIFIC_DECISIONS.md)
- **[R11]** [H4の未保存event・法則診断・統計（同資料D/E節）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f6b0cb5e2ddabd2d7919d0822c58324c7338bac0/docs/research/track_a_h4_geometry_technical_integration_20261013.md)
- **[R12]** [B0+I人工/mock test source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8f6c1bc8ceeeffddbecf170ef6ddb0fe6d18b63b/tests/tracks/resource_applicability/test_h4_b0i_preparation.py)
- **[R13]** [実費用取得runtime（230–335行を重点確認）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8f6c1bc8ceeeffddbecf170ef6ddb0fe6d18b63b/src/trottertracks/resource_applicability/h4_b0i/runtime.py)

### 外部一次資料（補足）

- **[P1]** Güntherほか, *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum **7**, 020332, 2026年5月19日。出版情報・abstractを確認。今回のH4 B0+Iの数値の根拠ではない。https://journals.aps.org/prxquantum/abstract/10.1103/ynxb-p2xq
- **[P2]** IBM Quantum Documentation, Qiskit 1.3 `PhaseGate`。行列定義とRZとの位相関係を確認。特定のH4 full-wrapper compileのRZ差を保証する資料ではない。https://docs.quantum.ibm.com/api/qiskit/1.3/qiskit.circuit.library.PhaseGate

### 引継いだ研究記録

- `track_a_h4_geometry_scientific_review_20261013.md`：会話添付の既存GPTレビュー。冒頭の研究判断と次作業・GO/STOP節を再確認。元結果を保持しscalar対照を保存値診断する方針を継承した。
- `research_project_instructions_gpt_codex_review_approval_2026-10-12.md`：会話添付の共通運用指示。科学レビュー承認と実行承認を区別する。
