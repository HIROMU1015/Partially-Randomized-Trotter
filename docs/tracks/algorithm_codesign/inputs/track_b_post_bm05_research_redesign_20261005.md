# 研究B：BM-0.5後の方針再設計
## 回路合成と測定負担を含むランダム化の共同設計

作成日：2026-10-05  
対象：`HIROMU1015/Partially-Randomized-Trotter`  
最新の判断根拠：`d55de044b8e956ba6292209a94bb081014dfdae2`  
Bの履歴入口：`0da4d18acf3f5d32d1bc32c9661b667885bcf5f2`  
BF1-R0：`6d2645a09440f50e5b869ef42a1b73a1b625a1af`  
Track A参照：`4c23453c541700c6a41ba71fc5ec9323b53858d6`

**位置付け：研究方針の提案。新手法の新規性・性能の認定、科学実行の認可、既存契約の変更ではない。**

今回行ったこと：指定repository文書の読解、一次文献の追加確認、設計上の代数的整理、計画書の作成。  
今回行っていないこと：新しい分子・toy Hamiltonian生成、state/signal評価、sampling、回路生成・合成、test実行、旧結果再分類、repository書換え。

---

## 0. 推奨する判断

**B-F現行仮説と、B-M現adapterのnew-method主張は閉じる。一方、研究B全体は止めず、何を成果にするかを変更する。**

次期Bの第一候補は、

> **部分ランダム化simulationと確率的な回路合成を組み合わせるとき、どこをランダム化し、どこを決定論的に実装すれば、同じcoherent-signal精度をより少ない合成後資源で達成できるか。**

とする。

仮題：

> **部分ランダム化Hamiltonian simulationにおける回路合成・測定負担の共同設計**

英語仮題：

> **Synthesis-aware randomization for coherent-signal estimation with factorized Hamiltonians**

これは「PAIとRTEを組み合わせれば新しい」という提案ではない。PAI、TE-PAI、元のPR論文のrounding-to-residual、SPRINTの実装最適化を既知として比較する。**成功の中心は、正しく実装できる具体的なprotocolと、その有利・不利条件を定量化すること**である。

新しい一般定理を唯一の完成条件にはしない。しかし、組合せ・目的関数・恒等式だけを新規性とも呼ばない。現在、次期Bの優位性や独立した新規性は未実証である。

BM-1の既存72列pilotを名称変更して実行する案ではない。BM applicationは保留資料として残す。新しい科学実行の前に、下記の有限な仕様と比較条件を利用者が採用する必要がある。

---

## 1. 保存証拠の解釈を固定する

### 1.1 B-F

[R1]の復元結果では、F/Lのfinite最良は同じSuzuki5、q1/R10/K2、action-work 20,709,936.722970817。共通参照の最良native S2 q2/R5/K2は12,924,223.160897588。F/L比は1、F/全参照比は約1.6024である。

閉じるのは、固定H4 1.00 Å・DF rank12・L_D=3・T=0.8・epsilon=.01・5段4次family・32評価/armにおける、finite-task objective固有の追加価値という仮説である。

原BF-1は保存失敗のINCONCLUSIVE。BF1-R0はfailure後のread-only replayによるBF-A。両者を統合して原runを成功扱いしない。bridge欠落を埋める必要はない。

### 1.2 BM-0.5

[R2]は、固定native nested列に対し、

\[
K_m^{\mathrm{BM}}=K_m^{\mathrm{compact}}
=K_{\mathrm{floor}}+K_A/(4m^2)
\]

という理想三次BCH係数の同値性を一般サイズの再帰から示し、9つの形式fixtureでも照合した。

同じDF backend・集約・cost・tieならscoreは同じ。floor/internalの再利用も強いcompact対照が利用できる。この式の整理やcache機会を独立方法上の改善とする根拠はなくなった。

ただし、以下はBM-0.5では未判定である。

- 非一様な実行頻度が物理系で有用か。
- finite-T信号やcompiled costの順位。
- 別の計算法、近似、実装による実用上の改善。

**同じ演算の正しいBCH係数が等しいことと、その演算を用いる全ての研究が無価値であることは別である。** 現adapterに独立の計算量・情報保持の改善もなかったことまで含めて、今回のnew-method路線を閉じる。

### 1.3 B-S

[R3]の0.0307–0.1241%は、32標本を持つ三つの保存経験分布に、特定のcost-aware importance-sampling目的を適用したplug-in headroomである。

これは、量子測定noise込みのあらゆる分散削減法の上限でも、新しい合成器を使った後のweight変動の上限でもない。B-Sは保留を維持するが、「ランダム側の最適化余地はない」と一般化しない。

### 1.4 Track A

[R4]は、固定DF、二次PF、canonical finite-RTE、所定shot規則、QiskitのRZ等による有限信号resource case studyである。原稿化までの証拠は独立に保持する。

RZ数は、角度精度・合成方法・workspace・catalogueを固定したT-state費用そのものではない。したがってAのRZ順位から、次期Bの合成後順位が既知だとは扱わない。

---

## 2. 研究の採否基準を修正する

これまでの進行で、次の条件を厳しくしすぎていた。

1. **既知法では原理的に得られないscoreを要求する。** 正しい同一量を評価する方法なら、値が等しいことは自然である。比較すべきは、具体的な計算手順、取得情報、費用、保証、実装と適用範囲である。ただしBM現案には、その差も残らなかった。
2. **最適splitが変わることを必須にする。** 同じsplitのまま資源が改善しても価値はある。逆にsplitが動いただけでは、優れた方法を作ったことにならない。
3. **application研究はAと重なるから弱いと決める。** 同じ種類の論文でも、RQ・比較対象・未解決事項が違えば独立し得る。今回のBM applicationの優先順位が低いのは、具体的に追加される知見と必要costがまだはっきりしないためであり、applicationという分類のためではない。
4. **試験前に新定理の完成を要求する。** 探索的な小型検証は、新規性・有用性を具体化するためにも使える。一方、同じ式と判明した二つのselectorの差を数値で探す必要はない。
5. **STOPを研究領域全体の禁止にする。** STOPは、その仮説・入力・比較に対する運用判断である。独立した機構とRQがあれば別研究を設計できる。ただし結果後の条件変更を元実験の継続や成功へ読み替えない。

今後の採否は、**構成の正しさ、最接近対照との実質差、費用を含む有用性、再利用できる範囲**で判断する。

---

## 3. 今回確認した文献と含意

### 3.1 既知として扱うこと

| 文献 | 今回確認した箇所・範囲 | 計画への含意 |
|---|---|---|
| Maxwell et al., 2606.30738v1 [W1] | §III.2.1–III.2.3、compact BCH、recursive PF、cache | BMのfloor/internal分解だけを新しいerror estimatorとしない |
| Poulin et al., 1406.4920 [W2] | abstractおよび既存BM監査の§VI対応 | 異なる項のTrotter刻みを変えることは既知。今回その全文の再監査をしたとはしない |
| SPRINT, 2606.30741v1 [W3] | Fig.1を画像確認、関連本文、実装・random remainder・QROM | algorithm群の共同設計・FT実装を初めて考えたとしない |
| Günther et al., 2503.05647 [W4] | Appendix E、特にE3の係数丸め・random residualへの移送 | **角度合成とrandom tailのtrade-off自体は既知。最重要対照へ追加** |
| Koczor–Morton–Benjamin, PAI [W5] | PRL原著ページのabstract、およびTE-PAI Appendix Bの式 | 確率的角度補間とその測定overheadは既知 |
| Kiumi–Koczor, TE-PAI 2410.16850v2 [W6] | §I、II、IV、Appendix A/B | fixed-angle random circuit、合成costとshot overhead、Loschmidt/SPEへの利用は既知 |
| Hayata–Kikuchi, 2604.02854 [W7] | arXiv abstract・metadata | continuous TE-PAIを分子energy推定へ使う方向にも先行例。詳細性能は今回確認していない |
| Dai–Hasselgren–Kiumi, 2606.23544v1 [W8] | §II、IV、測定shot noiseの留保 | counting/ordering分散の分解・stratificationも既知。trajectory分散改善をそのまま量子shot削減へ使わない |
| Cugini–Atif–Subaşı, 2603.13495v1 [W9] | 一般importance-sampling枠組みとnet-cost | weight二乗と費用を合わせた最適化原理だけでは新規性にならない |

### 3.2 最接近文献を見落とさないための修正

元PR論文Appendix E3は、決定論側の係数を丸め、その残りをH_Rへ入れて合成costを減らす案を明示する。bitwise groupingとHamming-weight phasingも扱う。

従って、次期Bを「決定論とrandomのcostを初めて一緒に最適化した」と書くことはできない。PAIをDFへ適用した、Hadamard信号へ使った、というだけでも独立新規性を認定しない。TE-PAI本文もLoschmidt amplitude・statistical phase estimationへの応用を述べている。

今回の検索で特定の同一実装が見つからなかったとしても、それは不存在証明ではない。**現在確認できたのは、一般原理が既知であることと、具体的なDF implementation/weight/costの比較が今後の検証対象になることまで**である。

---

## 4. 次候補の比較と第一候補

| 候補 | 既存資産との接続 | 成果として残すべき差 | 主なリスク | 判断 |
|---|---|---|---|---|
| BM application | native列、BCH、DF backendを再利用可能 | finite-T・実回路での有効/失敗条件 | known coalescingの図示だけに終わる | 保留。価値を否定はしないが旧72列は自動実行しない |
| **回路合成・測定込みのrandomization placement** | DF/RTE/wrapper/normalizationを直接使える | 実装可能なprotocol、同一精度下の合成後利益・不利条件 | PR E3・PAI等との重複、weight増幅 | **第一候補** |
| counting/ordering stratification | trajectory記録・有限mean基盤を利用 | 実測shot noiseと学習cost込みの追加価値 | 最新文献と近い、未知conditional meansの取得cost | 第二候補にはせず、対照・知識として保持 |
| 全RPE/energy資源比較 | 元の研究目的に接続 | 最終推定器での総費用と安全性 | scope拡大、別プロジェクトとの重複 | 信号taskの結果が必要性を示した後の発展 |

第一候補は最も新しいと証明された案ではない。**これまで固定していた「合成器」と「回転gateの重さ」を明示的な設計変数へ移すため、BF/BMの不成立と独立に調べる理由がある**。

---

## 5. 主RQ、副RQ、対象task

### 主RQ

> 同じDF Hamiltonian・入力状態・有限時間coherent signalを同じ精度と失敗確率で推定するとき、Hamiltonian-level randomizationとgate-level stochastic compilationの配置・精度をどう選べば、合成後の総非Clifford資源を削減できるか。

### 副RQ

- 固定PF/RTE回路に対して、決定論側、random tail側、双方のどこへ確率的角度補間を入れる価値があるか。
- random circuitごとの補間weightと回路長の変動を入れると、平均gate数だけの設計と結論が変わるか。
- 元PRのrounding-to-residual、最適化済みdeterministic synthesis、PAI単独と比較して何が残るか。
- その設計を、exact signalを使わない入力情報から作れるか。使えなければoracle-assistedな範囲を明示する。

### Task

\[
z(T)=\langle\psi|e^{-iHT}|\psi\rangle,
\qquad
\Pr(|\widehat z-z|>\epsilon)\le\alpha.
\]

最初は有限時間信号taskを使い、既存基盤で機構を切り分けるscopeを提案する。energy精度の保証や全RPE費用は別の達成要件とする。

主claimは、同じsimulation候補へのcompiler改善と、同じH/epsilonでのsimulation方式間改善を分ける。前者の成功だけからpartial randomization一般の優位を結論しない。

---

## 6. 何を固定し、何を変更するか

初期比較ではDF表現、split、PF係数、q/r/K、state、Tを固定したsimulation回路を用いる。新PF係数、新factorization、新split、新sampling分布を一度に探索しない。

変更対象は**コンパイル方針**である。

1. 全て通常の決定論合成。
2. 決定論側のdiagonal rotationだけ確率的補間。
3. RTE event中の対象rotationだけ確率的補間。
4. 上記二つを同時適用。

basis変換は初期比較で同じ合成方針にする。これは恣意的にbasis overheadを隠すためではなく、diagonalとtailの差を切り分けるためである。basisの費用は必ず数える。basis自体をrandomizeする案は、この四つの結果を踏まえて必要性が生じた場合の別変更にする。

最初のablationでconfigurationを固定した後、方式全体の競争力を主張する段階では、対照にも同じ精度配分・q/r/K・compile自由度を与える。新方式だけjoint tuningし、対照を古いwinnerに固定しない。 元PRのrounding-to-residualはHを保ったままD/Rの配分を変える既知対照であり、初期ablationのsplit固定を理由に最終方式比較から除外しない。

---

## 7. 意味論：first momentとchannelを混同しない

RTEの元trajectoryをomega、source-defined normalizationをBとする。Hadamardのaxis測定Y_a∈{−1,+1}について、

\[
\mathbb E_{\omega,Y}[B Y_a]=\nu_a
\]

が、fixed finite-RTE corrected signalのmeanである。nuと物理的なexact signal zの差は残る。

各omegaについて、完成した**ancillaを含むwrapper全体**を対象に確率的コンパイルをする。その抽出をxi、補間normalizationをGamma(omega)、符号をs(omega,xi)∈{−1,+1}とし、

\[
\mathbb E_{\xi,Y}[\Gamma(\omega)s(\omega,\xi)Y_a\mid\omega]
=\mu_a(\omega)
\]

を要求する。mu_aは元wrapperのaxis期待値である。

これなら

\[
Z_a=B\Gamma(\omega)s(\omega,\xi)Y_a,
\qquad \mathbb E Z_a=\nu_a
\]

となる。**この塔則による合成は既知LCU/PAIの直接的利用であり、新定理として主張しない。**

system channelだけを再現するsamplerを、同じsample unitaryへcontrolを付けるだけでHadamardへ流用しない。±Uは同じsystem channelだが平均operatorでは相殺し得る。

TE-PAIを全方式のbaselineにする場合、ancillaを含む

\[
\widetilde H=|1\rangle\langle1|\otimes H
\]

のchannelを扱う構成、または原論文のcorrelator/SPE構成を明示して用いる。これも新規性ではない。単一runのcoherent QPEと、平均信号を使う統計的phase estimationを区別する。

### Correlationの注意

独立なgate replacementで証明した式を、同じ乱数でbasisとinverseを選ぶ方式へ無条件に移さない。正確なinverseを再利用する工夫は可能だが、必要な条件付き平均を新たに確認する。

既存のglobal phase、controlled relative phase、identity抽出、negative timeは保持する。中間gateがsectorを破る場合に、完全なlogical block用sector boundを全gateへ適用しない。

---

## 8. 測定負担：平均costだけで設計しない

canonicalなPAI抽出で、固定omegaに対する重みの絶対値がGamma(omega)なら、Y_a²=s²=1より、

\[
V_2:=\mathbb E Z_a^2=B^2\mathbb E_\omega\Gamma(\omega)^2.
\]

実際の分散は

\[
\operatorname{Var}(Z_a)=V_2-\nu_a^2.
\]

従って一般に

\[
B^2\mathbb E\Gamma^2
\ne B^2(\mathbb E\Gamma)^2.
\]

random circuitの長さに応じて補間gate数が変わる場合、平均長さを代入するだけではsampling負担を過小評価し得る。

### 単純化した生成関数の診断

独立なmicrostepをn回持ち、一microstepの対象gate数Lがp_kに従い、対象gate一つあたり同じ補間norm gammaなら、

\[
\mathbb E\Gamma^2
=\left(\sum_k p_k\gamma^{2L_k}\right)^n
\]

（固定deterministic部分の係数は別途掛ける）。これは確率生成関数の標準恒等式で、新規性にしない。

実際のDFでは、basis遷移、fusion、angleとevent typeへの依存を戻す必要がある。独立性がない箇所へこの積を使わない。Gammaの二次モーメントを元の32 sampleから厳密なpopulation値として推定しない。

### なぜB-Sの小さいheadroomと矛盾しないか

B-Sは固定費用分布を別の確率で引く特定の問題を見た。次期Bでは回路そのもの、合成cost、補間weightが変わる。費用の小さい変動は、Gammaの指数的蓄積が小さいことの証明ではない。

### Quantum shot noise

trajectoryごとの期待値mu(omega)を古典的に計算したときの分散と、Y∈{−1,+1}を実測する分散は別である。[W8]も主要なtrajectory-sampling改善値にmeasurement shot noiseを含めないと明記する。

本計画では実測outcomeに対応するZを定義し、全ての費用に同じmeasurement modelを使う。

---

## 9. 有限精度・失敗確率・資源

### 9.1 Bias budget

総biasの受理は、例えばaxisごとに

\[
b_{a,\mathrm{PF/RTE}}+b_{a,\mathrm{synth}}+u_a
\]

を扱い、統計予算s_a>0を残す。

PAIが理想notch gateを用いれば正確でも、そのgateを有限精度で合成すれば追加誤差がある。dyadic angleだから任意の精度で無料・exactなClifford+T gateになるとは仮定しない。

conditional expectationの合成誤差をe(omega,xi)でboundすれば、補正後biasには

\[
|\mathbb E[W(Y'-Y)]|\le\mathbb E[|W|e]
\]

を戻す。eをoperator norm誤差から作る場合はobservable期待値への変換係数も含める。weightを掛ける前の小さいgate誤差だけで最終accuracyを判定しない。

### 9.2 Shot rule

V2は有用な設計量だが、それだけで特定のfinite-confidence式を完成したとしない。初期実装では全方式に同じ事前固定のconcentration ruleを使う。

有限cutoffでrange上限MとV2上界が得られる場合、Bernstein型の共通十分shot式を使える。必要ならmedian-of-means等を使うが、新推定器を同時探索しない。真のnuやoracle varianceを新方式にだけ与えてshotを減らさない。

weight rangeが小さい場合のHoeffding、分散boundを使う場合のBernstein等を結果後に有利な側だけ切り替えない。設計用Jと正式なN_alphaを分ける。

### 9.3 Primary resource

最終目標は、明示したlogical primitive catalogueにおける

\[
G_T=\sum_a n_a\,\mathbb E C_{T,a}
\]

またはそのcatalogueで定義した非Clifford state費用。Clifford+Tを使うならT countとancilla、depth、classical generation costを示す。

CCZ/T/Toffoli換算やRUS/catalystの初期化を無条件に同じ単位へ変換しない。異なるworkspaceを使う方式は同じ条件で比較し、初期化・再利用可能回数を記録する。physical qubits・実機時間はhardware modelなしに主張しない。

raw RZは比較可能性のためsecondaryに残す。全方式を一律「RZ一つ=一定T」で換算すると、調べたい合成差を消してしまう。

### 9.4 設計用の損益条件

同じfinite target、同じ統計予算、同じouter Bの下で、二次モーメントを使う粗い設計量は

\[
J=V_2\,\mathbb E C_T.
\]

合成費用比をg、補間による二次モーメント倍率をhとすれば、J上はgh<1が改善条件になる。これは標準的な費用×分散の整理であって新理論ではない。

正式判断はsynthesis bias、有限confidence、tail、state preparation、初期化costを戻す。J改善をそのまま最終G_T改善と呼ばない。

---

## 10. 作るべき具体的な設計手順

候補methodは、大規模な黒箱optimizerではなく、まず次の限定手順とする。

1. 固定simulation構成から対象rotationの型・回数・分布を取得する。
2. 事前固定した二つ以下のgate catalogue、四つのplacement maskを用意する。
3. 各maskについてPAI等の既知係数からconditional normalizationを計算する。
4. finite-RTEの既知分布からV2上界・range・classical生成costを評価する。
5. catalogueの実costとsynthesis誤差を入れ、同一shot ruleでGを計算する。
6. exact signalを使わない設計では、共通の保守的bias情報または既存calibrationを使い、その取得費用を明示する。
7. 判別できない場合は新しいmaskを探索せず、基準法を選ぶか未確定として報告する。

この手順自体が既知のresource-optimal ISや回路最適化の直接適用にとどまる可能性はある。その場合、new-methodではなく、特定DF taskのconstructive resource/design studyとして評価する。

**「係数が違った」「新しい組合せが出た」だけではGOにしない。一方、同じsplit・同じ正しいmeanを持つから差分がない、ともしない。**

---

## 11. 比較対照と寄与分離

| 対照 | 必須にする時点 | 注意 |
|---|---|---|
| 同じsimulation回路＋通常deterministic synthesis | 原始回路pilotから | 最も直接的なcompiler ablation。precision allocationを公平にする |
| 既知PAIを対象wrapperへそのまま適用 | 原始回路pilotから | 提案mask/解析の価値を、PAI一般の価値と混ぜない |
| RTE側のみ／D側のみ／双方／どちらもなし | 原始回路pilotから | best fixed placementに対してjoint designが必要か |
| 元PR Appendix E3 rounding-to-residual | chemistry/taskレベル比較の前 | 最重要の近接構成。新方式だけgate synthesis自由度を増やさない |
| TE-PAIの同一coherent observable実装 | 広いrandom-algorithm比較を主張する前 | system channelとamplitudeの対象差、ancilla、bit precision、classical costを合わせる |
| 強いfull deterministic DF/PF | partial方式全体の優位を主張する前 | 同じgate catalogue・ancilla budget・誤差配分。new compilerがdeterministicにも効く可能性を残す |
| SPRINT関連の適用可能な合成技法 | broad practical advantageを主張する前 | 全framework未実装なら、比較した部分と比較外を明示 |

最初のpilotで全frameworkを再実装しない。ただし、そのpilotをもって広い優位性が確立したとも言わない。

---

## 12. 段階計画と具体的な成果物

### 段階1：既知構成を明記した設計仕様

科学計算なしで閉じる内容：

- 四placement mask。
- full wrapperを対象にした条件付き平均の等式。
- 理想gate interpolationと有限合成の区別。
- B、Gamma、V2、range、bias、G_Tの定義。
- primitive catalogue、workspace、初期化／再利用costの候補。
- PR E3、PAI単独との違いがどこにあり得るか。
- 保存資料にangle-level情報があるかの確認。不明なら未確認とする。

RZ総数・平均costだけからangle別合成costやGamma²を復元しない。今回読んだsummaryではこれらの情報は確認できていないが、repositoryの全fileについて不存在を証明したわけではない。

この仕様は一つの文書と一つの小さいschemaで十分。文書を何巡も作ることを進捗の代わりにしない。

### 段階2：原始回路・意味論の最小pilot

新しく採用する場合の提案scope：分子入力なし、少数qubitと明示的有限RTE分布だけ。

Controls：

1. 単一Pauli rotationの補間meanとweight。
2. 二つの非可換rotationを含むcontrolled wrapper。
3. 同じ平均回路長でも長さ分布が異なる有限分布：E Gamma²と平均長近似の違い。
4. +U/−Uのphase control、およびbasis/inverseに乱数を共有する場合の反例／正しい条件。

原始templateは最大4種、mask4種で最大16基本比較とし、catalogueは最大2種を提案上限とする。angleとprecisionは実行前に公開primitiveの実装条件から固定し、結果に応じて増やさない。

ここで得たいものは同値性だけでなく、**weight増加を戻したG_Tに改善可能性があるか**である。合成前countだけの改善は次段への十分な根拠にしない。

対象primitive・allowance・数値guard・wall/CPU/RSS/output上限は、使用する合成器を決めた後に利用者reviewで固定する。本書では未確認の実行時間を断定せず、旧BFの4h等を流用しない。

### 段階3：一つのDF development問題

候補が残った場合、既知H4 1.00 Å等をdevelopmentとして利用する。最初は登録した少数simulation構成だけ。

元state、DF、PF、q/r/Kを保持したcompiler ablationを先に行う。次に必要なら、method全体の比較で各対照へ同じ再最適化機会を与える。

測定costはactual circuit側、bias参照は保存値・小系exact評価側と区別する。compiled後のsynthesis biasは別に必要。noisy hardwareを使っていなければnoisy performanceは主張しない。

### 段階4：設計を凍結した独立評価と原稿化

最終的に作ったmask選択・budget配分手順を凍結し、未使用の一つ以上の条件へ移す。H4 1.30 Åを新held-outとは呼ばない。

新しい系の選択は、tail長、basis負担、angle distribution等の構造差を検査できることを優先する。大量geometryを増やすことを一般性の代わりにしない。

一つの追加系から漸近scalingを主張しない。独立した広い主張には、それに対応する規模・scopeの検証が必要である。

---

## 13. 新規性として成立し得るもの／しないもの

### それだけでは成立しない

- PAI、TE-PAI、RTE、known QPDを使うこと。
- 期待値の塔則やE Gamma²の式を導くこと。
- norm×cost、variance×costを最小化すること。
- 同じH4で新しいwinnerを見つけること。
- PRが既に述べるrounding-to-residualをDF表記にし直すこと。
- 一般optimizerにgate類別のcostを入力すること。

### 目標とする追加貢献

- 現実に生成・実行可能なDF wrapperのcompiler/samplerを構成し、全weightと相対位相を正しく扱う。
- 従来の平均回路長ベースの設計がどの条件で過小評価し、どのplacementがその問題を避けるかを、実行可能な情報から定量化する。
- PR E3／PAI単独／強いdeterministic synthesisに対して、同じtask・情報・workspaceで利益が残る。
- 新しい規則がなくても、従来予測と異なる実装上の有効・不利領域を、複数control・独立条件・再現可能なcost modelで示す。

上の最後はresource/design studyとしての貢献であり、新algorithm一般とは呼ばない。現在は全て未実証の目標である。

**新規性確認の現時点の結論：一般原理は既知。具体的なDF implementationとjoint overheadに関する研究余地は候補として残るが、独立paperとしての十分性は未確定。**

---

## 14. 論文としての着地点

### 最小着地点

実装可能な少数protocolを共通taskで比較し、合成costとshot overheadの二重計上／過小計上を避けた、再現可能な小型resource/design study。

ただし単一toyと既知式だけなら独立論文に十分とは判断しない。その場合は内部technical noteまたはAとは別の実装付録として閉じる。

### 目標着地点

> **DFのcoherent signal推定に対する合成・randomization placementの具体的手順を与え、同じ要求精度でのG_T・depth・qubit/classical-costのtrade-offを示し、既知構成より有利な領域と不利な領域を説明・独立確認する。**

新しい一般theoremがなくても、構成と定量的知見が独立していればmethod/implementation paperを目指せる。一方、良い数値が出たことだけでは採択可能性を保証しない。

### 発展的着地点

複数時刻のsignalを使うstatistical phase estimationへ接続し、epsilon_E、失敗確率、入力overlap、phase wrapping、準備・合成・測定を含む総費用へ進む。

これは初期paperの必須条件にしない。有限signalのepsilonをchemical accuracyと同一視しない。

### 論文構成案

1. 同じsimulationでも合成器を変えると資源会計が変わる問題設定。
2. full-wrapper protocolとconditional estimator。
3. weight moment・range・合成bias・costの会計。
4. 原始controlとknown-baseline比較。
5. DF development、固定手順の独立評価。
6. 有利/不利領域、oracle/catalyst/hardware条件、限界。

主図候補は、(a)二層protocol、(b)平均長近似と実moment、(c)placement別G_T/depth frontier、(d)独立条件における固定規則の成績。勝ち例だけの図にしない。

---

## 15. 停止条件と再設計の位置

### 早期に閉じる場合

- elementaryな試験から、回路費用の削減をweight増加が一貫して上回る。
- 比較可能な現実的catalogueがなく、理想gate costを任意に設定しないと利益が出ない。
- 最接近法を同じ条件へ実装すると提案と完全に同じ構成・費用になる。
- joint designの利益が結果後のstate/threshold調整にしか現れない。
- 開発手順がexact signalを常に必要とするのに、oracle-free methodを主張したいままになっている。

### 続ける場合

- 同じtarget/precisionで、重み・合成誤差込みの利益または新しい制約の具体例が残る。
- 対照を公平に調整しても消えず、どの構造が効くか説明できる。
- 新方式が全域で勝たなくても、利用可能な情報から選べる有効域がある。

old 5%/10%は流用しない。practical materialityは実際のresource modelと不確かさから実行前に定める。数値guardを過度に広げて何も分からなくした場合と、科学的negativeは分ける。

一つのcompilerが不利という結果を、確率的合成全体のno-goへ一般化しない。一方、条件を増やして勝つ例を探す継続にも自動で進まない。

---

## 16. Track Aとの分離

| Track A | 次期Track B |
|---|---|
| 固定DF・PF・finite-RTE classの資源競争を説明 | compilerとrandomization placementを変更し、実行可能なprotocolを設計 |
| primaryは指定Qiskit文脈のcompiled RZ | 明示した合成器による非Clifford資源、補間weight、synthesis bias |
| 保存信号と32cost標本によるcase study | 新しいprimitive/weight/synthesis記録と独立した検証 |
| 原稿v0.1を現在のscopeで閉じる | Aの完成を待たず、まず小型設計を進める候補 |

同じcodeを使うことと、同じ独立evidenceを二重計上することは別である。Aのsource/status/resultを書き換えない。

B固有実装は既存方針どおり `src/trottertracks/algorithm_codesign/` の新submodule、docs/artifactsも新namespaceへ置く。B-F/B-Mをrenameして履歴を消さない。

---

## 17. Codexへ渡す直近の指示案

この節は、利用者が計画を採用した場合に渡すための案であり、本書から実行を開始しない。

```text
BM-0.5後の研究B計画を、まず設計仕様へ落としてください。

固定根拠はd55de044、BF-R0 6d2645a、A参照4c23453です。
B-F限定closure、BMの同値性、未実行BM-1をそのまま保持します。

新しい検討対象は、固定DF/PF/RTE wrapperに対する
「決定論合成／D-diagonalだけ確率的補間／RTE-event rotationだけ／双方」
の四placementです。新PF係数・split・Hamiltonianはまだ探索しません。

1. 既知のPR Appendix E3 rounding-to-residual、PAI、TE-PAIを比較表に残す。
   組合せや塔則を新規性として記載しない。
2. ancilla込みwrapperに対する条件付き平均、weight、二次モーメント、range、
   finite synthesis bias、同一の有限confidence shot規則を定義する。
3. primitive catalogue、angle/error/cost、workspace、初期化costを扱う仕様案を作る。
   理想dyadic rotationを無料・exact Clifford+Tとしない。
4. 既存記録にangle-level inventoryがあるかだけ確認する。
   aggregate RZから推定せず、未確認・欠測を明記する。
5. 分子入力を使わない最大4control template、4placement、最大2catalogueの
   pilot設計案と、使用する合成器に対応する予算案を作る。
6. 正式sourceは新B namespaceへ置く案を示す。旧source/resultを変更しない。

今回の作業は設計・文献対応・保存情報の所在確認まで。
NPZ、science signal、sampling、circuit build/synthesis、GPU、全repo testsは行わない。
設計文書、comparison matrix、pilot proposalを一つのreview packetにまとめてSTOP。
科学実行は、別のsource-bound仕様・限定test・利用者の明示承認が必要。
```

研究判断の中心は「次のgateの名前を作る」ことではない。小さい実装試験で、現実的なcostとweightのもとに改善可能性があるかを確かめるところまで進めるための仕様にする。

---

## 18. 限界・未確認事項

- 新compilerのT-count、weightの実分布、合成後bias、最良placementは未計算。
- 次期Bが元PR E3を超えること、独立論文として十分であることは未確認。
- repositoryの全angle/circuit記録を網羅棚卸ししたわけではない。
- 旧数値を新しい合成器へ変換した結果はない。
- 文献比較は今回の候補に近い一次資料に限定し、systematic review・全引用網監査ではない。
- 今回の数式は研究仕様を整理する既知原理の適用・代数導出で、新定理の完成報告ではない。

---

## 19. 参照資料

### Repository

[R1] BF1-R0 result validation, commit `6d2645a09440f50e5b869ef42a1b73a1b625a1af`,
`docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md`.

[R2] BM-0.5 equivalence audit, commit `d55de044b8e956ba6292209a94bb081014dfdae2`,
`docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md`.

[R3] BF-0 prior-art matrix §6, commit `6d2645a09440f50e5b869ef42a1b73a1b625a1af`,
`docs/tracks/algorithm_codesign/bf0_prior_art_claim_matrix.md`.

[R4] Track A claim/evidence map, commit `4c23453c541700c6a41ba71fc5ec9323b53858d6`,
`docs/research/track_a_post_pm2_claim_evidence_map.md`.

### 一次文献

[W1] William Maxwell et al., *Practical Estimation of Trotter Error for Hamiltonian Simulation*, arXiv:2606.30738v1 (2026).
https://arxiv.org/html/2606.30738v1

[W2] David Poulin et al., *The Trotter Step Size Required for Accurate Quantum Simulation of Quantum Chemistry*, arXiv:1406.4920 (2014).
https://arxiv.org/abs/1406.4920

[W3] Pablo A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741v1 (2026).
https://arxiv.org/pdf/2606.30741v1

[W4] Jakob Günther et al., *Phase estimation with partially randomized time evolution*, PRX Quantum 7, 020332 (2026), arXiv:2503.05647. 今回読解のPDFはversionless取得で、arXivページはv2を表示。引用はAppendix Eの章名・内容で特定し、別版の式番号を混ぜない。
https://arxiv.org/abs/2503.05647
https://arxiv.org/pdf/2503.05647

[W5] Bálint Koczor, John J. L. Morton, Simon C. Benjamin, *Probabilistic Interpolation of Quantum Rotation Angles*, Physical Review Letters 132, 130602 (2024).
https://doi.org/10.1103/PhysRevLett.132.130602
https://arxiv.org/abs/2305.19881

[W6] Chusei Kiumi, Bálint Koczor, *TE-PAI: Exact Time Evolution by Sampling Random Circuits*, arXiv:2410.16850v2; Quantum Science and Technology 10, 045071 (2025).
https://arxiv.org/html/2410.16850v2
https://doi.org/10.1088/2058-9565/ae1160

[W7] Tomoya Hayata, Yuta Kikuchi, *Continuous-time evolution via probabilistic angle interpolation and its applications*, arXiv:2604.02854 (2026). 今回はabstract・metadata確認の範囲。
https://arxiv.org/abs/2604.02854

[W8] Joshua W. Dai, Fredrik Hasselgren, Chusei Kiumi, *Structure-Aware Variance Reduction for Unbiased Randomized Hamiltonian Simulation*, arXiv:2606.23544v1 (2026).
https://arxiv.org/html/2606.23544v1

[W9] Davide Cugini, Touheed Anwar Atif, Yiğit Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1 (2026). 著者名は版指定HTMLのmetadataと照合した。
https://arxiv.org/html/2603.13495v1

