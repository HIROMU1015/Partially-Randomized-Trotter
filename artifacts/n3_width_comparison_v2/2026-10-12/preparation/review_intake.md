# N3理論・文献・幅指向縮約taskの統合：科学レビュー

- レビュー日：2026-10-12 JST
- 対象：N3 theory/task integration v1
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 統合payload：`2de3b26fc36df9e99b7504631c97635f1473cb51`
- 公開commit：`837abb5b200c61720c8b5097901a437c89d8aed3`
- 保存帰結checker source：`370c42a6d76ab034714c065182143d0aa112c0ff`
- 状態：ユーザーの明示的開始承認を受けた科学レビュー完了
- 新しい科学計算の実行認可：本レビューによっては付与しない
- 新しい数値実験・固有値計算・最適化・certificate探索・科学runner実行：本レビューでは0

## 0. 結論

N3は限定継続する。幅指向generatorと一mode縮約taskの**比較実装・限定検証準備へ進む科学的方針**を採択する。ただし、現在の仕様案v2をそのまま「新しい強力な保証算法」として採択するのではない。

今回の判断は次の四点である。

1. 同一の固定情報におけるS1の旧対照族との厳密分離、および登録旧ORの幅下端に対する達成改善は維持する。これをさらに同じS1で再探索する必要はない。
2. 旧ORとの分離だけでは独立した算法的新規性の根拠として不十分である。OCCはFantopeのSDPで表現でき、active側の証拠は粒子・正孔の正値二次形式と非負densityによるSOS証拠に含まれる。したがって「一般SDPより強い保証」を目指すのではなく、構造を利用して有用な証拠を少ない取得費用で返せるかを評価する。
3. 一mode縮約taskは次の限定検証として採用する。ただし同じmodeの比較とHamiltonian全体でいずれかのmodeを削除できるかの比較を分ける。mode候補の追加受理数を、追加qubit削減数に読み替えない。
4. 次の担当はCodex。文献差分と以下の比較契約を反映し、generator・比較対照・独立checker・synthetic検査・未使用入力とplanの固定を一つの作業として進める。通常のsource修正ごとにGPT詳細レビューへ戻さない。科学実行の直前には別途ユーザー認可を得る。

中心研究テーマ、論文の主要claim、未発表性、量子総資源優位は未確定である。元の追加保証0、旧2-base基準未達、N1停止、N2保留、既存のMandatory STOPは保持する。

## 1. レビュー範囲と証拠の区別

### 1.1 今回実際に行ったこと

固定commitから統合報告、claim map、文献対応表、幅指向v2仕様、保存解析値、14項目照合、公開監査を取得して読んだ。関連する過去の固定結果と前回レビューの結論も踏まえた。

外部調査では、既知のFantope表示、Feshbach–Schur、検証付きRDM下界、2026年のweighted SOS/v2RDM研究、関連するactive-space/sparsificationの位置付けを確認した。本書の第5・6・7節には、取得済み数式から本レビューで導出した解析的対応を明示する。

### 1.2 今回行っていないこと

30 synthetic tests、14項目照合、既存の数百万条件のPSD監査を自分で再実行してはいない。PASS件数は公開記録として扱う。新しいS1の数値、OCC/SDP/RDMの性能、未使用入力の採否、実量子資源は取得していない。

v2 generatorは仕様案であり、実装済みsourceのsoundnessを認定したわけではない。今後の独立checker・source-bound tests・実行認可は必要である。

### 1.3 資料取得の限界

歴史的原典Overton–Womersley、Colemanの全文未確認は文献表のまま残す。原典が未取得であることと、Fantope等の数学的表示を証明できないことは別である。

SQuISHについてはCodexの全文取得・対応箇所の記録を参照した。本レビュー側のweb PDF取得は失敗し、全文を独立に再読したとは扱わない。Fantopeおよび2026年SOS論文のPDF screenshotには取得エラーがあった。Fantopeは取得できたPDFテキスト中の式、SOSは取得成功したarXiv HTMLの数式を利用した。数値図表は評価に使っていない。Sionの原論文Theorem 3.4はPDFテキストとp.174の画像を確認した。

2020年の検証付きRDM論文は出版社・著者所属機関の公開書誌とabstractで確認した。具体的実装の性能や全定理までは照合していない。2026年SOS論文はarXiv v1の第I・II・IV節を中心に参照し、出版版と全式が一致すると仮定していない。

## 2. 研究全体の振り返り

### 2.1 当初の目的を維持する

本研究の目的は、PR内部の乱択則改良そのものとは区別して、Hamiltonianの表現・構成・前処理による、新しい算法または本質的改良を得ることである。単に監査が充実した実装を作ることや、既知定理を数値的に再確認することへ目的を下げない。

N3は量子計算前の古典的な誤差保証付き縮約として、その目的と接続している。しかし、現時点で量子Hamiltonian simulation全体の費用改善を実証したわけではない。

### 2.2 経緯と今回への教訓

| 系統・段階 | 支持されたこと | 残った限界・今回への意味 |
|---|---|---|
| 初期表現探索A/B/C | core、補助空間、near-integrable表現などを構成・比較できた | 係数や局所回路の改善だけでは精度をそろえた最終目的の改善を保証しない |
| N1 | 近似・frame構成と費用込みselectorが成立 | 固定3 contextのRZ/CX selectorはexactを選び、改善0%。proxyと実際の目的の不一致を再発させない |
| N2 | signed charge構成と全wrapperを実装 | 限定小系では算術・uncompute込みの費用優位を支持しなかった。部分回路の安さを全費用と混同しない |
| N3初期・直接生成 | 係数由来の保証、方向情報の利用 | 強化した下界と既存対照を入れると固有gainが消える場合があった |
| OCC固定8入力 | NORM/強い旧OR/raw OCCが8/8 | 追加受理0、旧2-base未達を維持。飽和したendpointでの比較には限界があった |
| 保存幅評価 | S1二方向で狭いOCC幅 | 当初は保存旧候補の不足という交絡が残った |
| 旧族診断・全幅補題 | S1で全旧LMIを排除し、同情報の登録旧OR下端を確定 | この旧族は既知のFantope/SOS全体ではない。今回そこを明確にする |
| 今回の統合準備 | 理論、文献、取得費用、mode選択の仕様を接続 | 新generatorの取得能力・同等/強い対照に対する効率・新規性は未評価 |

N1/N2を一般的な不可能性として終了したとは扱わない。ただし、新しい構成機構なしに同型探索を再開する根拠もない。親リポジトリのTrack A/Bは費用会計・同精度比較の参考にとどめ、別taskの成績をN3へ移植しない。

### 2.3 今回の新情報は何か

今回の30 testsと14項目一致は前回の解析を独立に支えた。新しいHamiltonianでの改善例が増えたわけではない。本レビューが判断する対象は、統合された仕様と先行研究対応によって、次の独立した算法検証へ進む根拠があるかである。

## 3. 保持する数学的成果と適用範囲

Hamiltonianは

\[
H=\sum_{ij}h_{ij}a_i^\dagger a_j+\sum_{i<j}J_{ij}n_i n_j,
\quad h=h^T,\quad J_{ij}\ge0
\]

であり、固定ν-sectorを一つの外部mode dが空/占有の部分へ分ける。

\[
H_\nu=\begin{pmatrix}A&B\\B^\dagger&C\end{pmatrix},\qquad
A=d\Gamma_\nu(h_A)+V_A,\quad V_A\succeq0,
\]
\[
BB^\dagger=d\Gamma_\nu(vv^T),\qquad v_i=h_{id}.
\]

係数と証拠から取得した

\[
a=\lambda_{\min}(A)\le U,\qquad C\succeq cI,\qquad g=c-U>0
\]

を使う。目標は基底エネルギーの投影差

\[
0\le a-E_0(H_\nu)\le\delta
\]

の保証であり、時間発展誤差、真の相互作用ground state、測定統計、量子回路費用の保証ではない。

\[
K_\kappa=\kappa h_A-vv^T,\quad D=g+\delta,
\]
\[
0\le\kappa\le D,\quad Z\succeq0,\quad K_\kappa-\tau I+Z\succeq0,
\quad s=\nu\tau-\operatorname{tr}Z\ge\kappa U-\delta D
\]

というOCC証拠からSchur補元で保証が得られる。現在の証明の前提に破綻を見つけたわけではない。反対に、非負density、固定粒子数、一外部mode、c>Uを外して一般化済みとは扱わない。

S1の登録旧OR下端は、保存Wに対するr≤0、b>β²/νからδ_Nに等しい。これは今回の独立照合で維持された。達成幅改善約20.424087%と0.343340%は、同じ固定U,cにおける当該登録旧ORへの比較である。OCC最適幅や全既知法への最適性ではない。

## 4. 新規性の評価を一段厳しくする必要がある

### 4.1 旧族との分離は実質的だが、相手の範囲が狭い

旧族の最適scalar lowerはνλ_min(K)であり、OCCはσν(K)を用いる。

\[
\nu\lambda_{\min}(K)
=\min_{\gamma\succeq0,\operatorname{tr}\gamma=\nu}\operatorname{tr}(K\gamma),
\]
\[
\sigma_\nu(K)
=\min_{0\preceq\gamma\preceq I,\operatorname{tr}\gamma=\nu}\operatorname{tr}(K\gamma).
\]

占有上限を入れることによる強化である。既知のFantope/1-RDM表示は後者を表すため、旧族への分離を一般SDPへの分離として述べることはできない。

### 4.2 「有理certificate」「粒子数付きSDP」だけでは新規性にならない

Codex文献表に加え、本レビューで以下を確認した。

- Chaykinほか（2020）：必要N-representability条件を使う分子ground-energy下界について、浮動小数solverの誤りを考慮した検証付き数値下界を扱う。[L8]
- Rubin–Low–DePrince（2026）：weighted SOSとv2RDM双対の対応、粒子数等の対称性を含む下界certificateを扱う。[L9]

両論文がN3の一mode空射影・同じU,cによる幅generatorを既にそのまま提案していると結論したわけではない。そこまでの同一性は未確認である。一方、N3の新規性を「粒子数制約」「SDP化」「検証付き下界」一般へ置くことは支持できない。

### 4.3 残す中心的な研究課題

候補となる貢献は、限定Hamiltonian構造を利用し、所定の誤差予算で使える縮約certificateを、同等またはより強い一般手法に比べて有利な取得費用で構成する算法である。

単に既知SDPを呼ぶ実装と監査を追加しただけなら、当初目標の「本質的算法改良」を満たしたことにはしない。証拠取得効率、情報取得方法、構造を使う非自明な計算削減、または保証とtaskを結ぶ新しい理論差が必要である。現時点ではその差は未実証である。

## 5. 本レビューの解析的対応A：OCC証拠は制限されたSOS証拠に含まれる

以下は文献のN3についての主張ではなく、本レビューが現在の数式から導出した対応である。

\[
R=K_\kappa-\tau I+Z\succeq0,\qquad s=\nu\tau-\operatorname{tr}Z
\]

とする。固定ν-sectorでは

\[
\boxed{
\kappa A-BB^\dagger-sI
=d\Gamma_\nu(R)
+\bigl(\operatorname{tr}Z\,I-d\Gamma_\nu(Z)\bigr)
+\kappa V_A.
}
\]

右辺はすべて非負である。

- dΓ(R)はR≽0から粒子の正値二次形式。
- trZ I−dΓ(Z)はZ≽0から正孔の正値二次形式。実対称ZならCARによりΣ Z_ij a_i a_j†と等しい。
- κV_AはΣ_{i<j}κJ_ij(a_j a_i)†(a_j a_i)で、非負densityによるSOS。

全Fock空間では右辺にτ(N−νI)を加えた恒等式として表され、固定sectorでこの項が消える。よって粒子数等式を含むweighted SOSの範囲内にある。これは[L9]が扱う一般的な粒子数付きSOSの種類との明示的な包含であって、その論文が現在の縮約taskの全体を解決したという断定ではない。

この対応から、適切な粒子・正孔・pairの生成子と粒子数等式を持つ一般SOS参照法は、理想的にはOCC証拠を再現できる。より自由な相互作用の分配やmoment制約を加えれば、さらに強い下界を返す可能性がある。有限solverが予算内にその証拠を発見できるか、厳密検証できるかは別である。

したがって、一般SDP/SOSと比べる際のN3の可能性は「保証の包含関係で上に立つこと」ではなく、特化構造による計算・証拠取得・記憶量の利益にある。

## 6. 本レビューの解析的対応B：理想OCC全体を表す同値SDP

### 6.1 変分表示

\[
\mathcal F_\nu=\{\gamma=\gamma^T:0\preceq\gamma\preceq I,\operatorname{tr}\gamma=\nu\}
\]

をFantopeとする。D=g+δ>0を固定し、

\[
\Phi(\kappa)=\sigma_\nu(\kappa h_A-vv^T)-\kappa U
\]

と定義する。理想OCCはmax_{0≤κ≤D}Φ(κ)+δD≥0と等価。

Fantopeと[0,D]はともにコンパクト凸集合で、

\[
f(\kappa,\gamma)=\kappa(\operatorname{tr}(h_A\gamma)-U)-v^T\gamma v
\]

は連続bilinear/affineである。Sionのminimaxを適用できる。[L10]

\[
\max_{0\le\kappa\le D}\min_{\gamma\in\mathcal F_\nu} f
=\min_{\gamma\in\mathcal F_\nu}\max_{0\le\kappa\le D} f
=\min_{\gamma\in\mathcal F_\nu}
\{D\max(\operatorname{tr}(h_A\gamma)-U,0)-v^T\gamma v\}.
\]

従って、

\[
\boxed{
\rho_D=\max_{\gamma,t}\{v^T\gamma v-Dt:
\gamma\in\mathcal F_\nu,\ t\ge0,\ t\ge\operatorname{tr}(h_A\gamma)-U\}
}
\]

と置けば

\[
\boxed{\text{理想OCCが幅δを保証する}\iff\rho_D\le\delta D.}
\]

これはκ最適化まで含む理想条件族の同値表示であり、固定κの固有値和primitiveだけの対応ではない。ν=mの場合もFantopeが単一点となるだけで議論は保たれる。

### 6.2 証拠の向きを誤らない

現在の有理数OCC証拠はρ_D≤q=κU−ντ+trZを与える。q≤δDなら受理できる。

逆に、厳密に検証された有理γ,tがFantopeおよびt制約を満たし、v^Tγv−Dt>δDならρ_D>δDなので、この**OCC十分条件族**には可行解がないと証明できる。これは物理的な縮約不可能性や他の保証法の不可能性ではない。

浮動小数SDPのprimal値・dual値をそのまま厳密上下界として使わない。必要なPSD、trace等式、スカラー不等式を元係数で検証する。丸めで可行性を失った場合は未判定とする。

この同値表示を同等SDP参照実装の仕様へ組み込むことを提案する。新しい単独の理論研究を始めるための追加gateではない。Codexが数学・syntheticの対応を確認し、比較実装とまとめて進める。

### 6.3 幅最適化への意味

有限generatorがδで証拠を見つけられないことは非可行証明ではないため、二分探索の下端は上げられない。一方、上記の厳密な反証点が得られれば、その条件族について非可行なδを区別できる。

これを直ちに理想最適幅の高精度探索の認可とはしない。幅出力は基本的に「最良発見済みの証拠の安全上端」であり、最適性を述べるときだけ適切な証明済み下界を付ける。

## 7. 強い比較対照の採択

今後、旧ORだけを主対照として独立した算法の利益を判断しない。以下の役割を分ける。

| 対照 | 採択する役割 | 主な評価 |
|---|---|---|
| NORM/FIVE/OLDと安全なOR | 従来の安価な対照・占有情報のablation | 同情報での保証範囲、費用、fallback |
| 第6節の同値Fantope/一般SDP | 必須の同保証・参照実装 | 同じcertificate目標までの取得費用、有限budgetの未判定率、メモリ |
| 非負densityを保持する低次数RDM/SOS | 強い対照として限定比較へ含める | 同じ係数からのより強い証拠と、その取得費用の競合 |
| 全Hの係数由来lowerを使う区間挟み | 安価な別経路の対照として含める | Schur/OCCを使わず縮約認証が成立するか |
| 元Hを返すfallback | 安全性・mode削減の対照 | 削減0、射影誤差0。成功扱いにしない |

### 7.1 相互作用を保持する参照法の意味

\[
F_\kappa=\kappa A-BB^\dagger
\]

について、物理的な状態を含むmoment/2-RDM緩和と厳密dualからℓκ≤λ_min(Fκ)を得る。そのℓκがκU−δD以上なら、同じSchur議論で縮約を保証する。

比較の限定案として、固定粒子数の2-positivity（P/D、Q、G）を用いる一つのレベル、またはその明示的に対応するweighted SOSを採用する。全hierarchyを走らせることや、結果後に都合のよいレベルを選ぶことは求めない。使用するmoment/生成子、等式制約、PSD検証方法をsource固定前に明示する。

非負pair期待値とFantopeの条件を含む緩和なら、active densityを捨てるOCCとの理想的包含関係を検討できる。ただし名前が「RDM」なら自動的に包含するとはしない。実際のconstraint/生成子を対応させる。

RDMはここでは最適化変数であり、量子状態を測定して1-RDM/2-RDMを取得する方式と同一ではない。Hamiltonian係数から計算する参照法を「追加の物理oracleが必要」として不当に除外せず、必要な古典計算を会計する。

この参照実装が固定資源内で証拠を得られない場合は未判定として残す。N3が当該有限予算で実用的である可能性は残るが、理想RDM/SOSより強いという結論にはしない。

### 7.2 全Hのlowerによる対照

任意の厳密lower L_H≤E0(Hν)と共通上界a_d≤U_dがあれば、

\[
0\le a_d-E_0(H_\nu)\le U_d-L_H.
\]

例えばこのclassではV≥0を使い、full one-body matrix hのσν(h)の認証下界がL_H候補になる。この対照は追加のground truthを必要としない。計算・検証費用を含める。

これは本レビューの基本不等式からの提案で、S1で勝つことを計算したものではない。一つのHのL_Hを複数mode候補へ共有するときは、一度の取得と各modeのbindingを区別する。

### 7.3 同等SDPに対する「追加受理」を主張しない

完全に解いた同等SDPとは数学的に同じ条件族なので、理想保証範囲でOCCがそれを上回ることを成功要件にしない。研究上の問いは、構造化されたspectral法が同じ保証をどれだけ安価・安定に取得できるかである。

有限予算内で相手が未判定、こちらが有効certificateを取得したなら、それは有限取得能力の差として報告できる。理想族の厳密分離とは別ラベルにする。

## 8. 一mode縮約taskの採択と限界

### 8.1 採用する最小task

固定されたh,J,ν、複数候補を含む場合は結果前に定めたD_set、正の射影誤差予算ε_projを入力とする。各候補で一modeのみを空へ射影し、元係数への対応を持つ縮約A_dと証拠を返す。

縮約Hamiltonianはone-body部分だけではなく、残存mode上のdensityを含むA_dである。mode順序と固定粒子数を出力へbindする。可行候補なしなら元Hを返す。

### 8.2 二つのendpointを分ける

1. 同一mode dの比較：同じH,U_d,c_d,δで新しい証拠が採否や幅を変えるか。
2. H全体のtask比較：各方法が同じD_setの中で一つでもmodeを安全に削除できるか。

あるmodeではOCCだけが通っても、旧法が別modeを削除できるなら、両算法はともに一mode削除である。「追加候補の受理」と「新たに縮約できるHamiltonian」を別々に数える。

mode選択は、発見済みの安全上端が最小のものを返す方針を採用できる。ただし最良発見済みのselectorであり、真の投影誤差最小、全mode/全証拠最適、最小量子費用を保証するselectorではない。

区間が重なる二候補の上端で選ぶこと自体は安全だが、strictな幅順位が証明されたとは表示しない。

### 8.3 量子費用の主張をしない

全候補が一mode削除なら、認証候補が増えただけで多数modeの削減を達成したことにはならない。回路compile、状態準備、PF、測定会計を含まない段階で、gate/shot削減率を作らない。

将来その接続を評価する場合でも、有限時間信号の別Trackのshot会計やCAの数字を無条件に持ち込まない。

## 9. 誤差区間の表現に関する仕様上の補正

仕様案v2の

\[
E_0(H)\in[E_{est}-\epsilon_{solver}-\epsilon_{proj},
E_{est}+\epsilon_{solver}]
\]

は正しい。ただし「総energy予算は少なくともε_proj+ε_solver」という表現は、どの点推定を報告するかに依存する。

- E_est自体を元Hの推定値とする場合、worst-caseの絶対誤差はε_solver+δ_cert。
- 区間の中点E_est−δ_cert/2を報告する場合、絶対誤差半径はε_solver+δ_cert/2。

後者は未知の真の投影誤差を正確に引いたという意味ではない。既知の片側certificateの区間を中心化しただけである。本レビューでは後段の量子solver契約を新しく採択せず、次の一mode pilotは投影certificateをendpointにする。将来のend-to-end報告では区間・代表値・精度定義を固定する。

## 10. U,cの取得費用を含めた評価

### 10.1 二層の比較

**同情報kernel層**では、同じ証拠付きU,cを与え、保証生成部分を比較する。S1の既存分離が属するのはこの層である。

**係数からのcold task層**では、trial上界、外部lower、証拠生成、独立検査、候補選択、出力までを測定する。比較法が利用しない処理を強制的に実行したことにせず、依存する処理を正しく課金する。

両者を一つの費用比へ混ぜない。複数mode間の共有と別Hamiltonianのwarm再利用も区別する。

### 10.2 列挙方式の扱い

全determinant列挙や外部occupation列挙は有限小系では合法で有用だが、一般fillingでの多項式性の根拠にはならない。

次の一次pilotは、結果前に固定した有界trial poolまたは指定Slater trialと、外部一体固有値和の認証下界を基本とする方針を推奨する。全列挙・より強いdensity lowerは小系補助層として明確に分け、同じ値を無償で利用したことにしない。

これは既存S1のU,cを置き換える指示ではなく、新しいcold taskの設計方針である。方法間では同じ取得policyを使う比較を残し、強い対照独自のlowerを使うend-to-end比較は別に表示する。

### 10.3 記録するもの

U/c生成、numerical proposal、有理化・修復、exact checker、selector、出力に分けたwall/CPU/メモリ、数値呼出し、PSD、算術量、bit長、証拠bytes、失敗理由を記録する。solver反復数と固有値計算回数を同じ単位として足し合わせない。

cold/warm、独立実測とevent再構成、factorial検査と単独経路を分ける。同じハードウェア・thread・precision・計測規則で比較し、短時間の単発測定だけから安定したspeedup倍率を結論しない。

## 11. 独立限定検証の科学的範囲

### 11.1 実装・準備に進む価値はある

同じS1の証拠を再検証しても、新しいcold generatorの取得効率やmode選択の価値は分からない。ここまでの理論を使って、強い対照を含む限定pilotを実装する情報価値はある。

ただし、無条件に任意の新入力を大量探索するGOではない。以下の科学的範囲に閉じる。

### 11.2 固定する事項

- 実対称one-body＋非負density、固定ν、一候補につき一外部modeの空射影。
- 新しいorbital rotation、多mode同時/逐次削除、一般二体への拡張、実分子・PF/PR回路費用は入れない。
- 上記の安価な旧対照、同等SDP、固定低次数density-aware対照、full-H lower経路を比較の役割に応じて分ける。
- 新しいcold generatorはε_projに通ったというだけで、幅比較に必要な候補探索を恣意的に省略しない。決めた有限budget内で最良発見上端を返す。
- 実験の係数recipe、seed、D_set、ε_proj、method budgets、solver version、集計規則を結果前に固定する。
- 原12入力、原8入力、S1二方向は全てdevelopment。新たな無作為標本や独立成功として数えない。

### 11.3 独立性と判別性

未使用係数を使っても、結果前固定がなければ確認的検証にはならない。S1に合わせてεを選んだ例は機構説明用developmentとして分ける。

新しい確認用集合のεを結果後に変える、NORMが落ちた入力だけ残す、成功baseだけ増やす、都合のよいRDMレベルを後から選ぶことはしない。構造に基づくstress testを設計する場合も、選択規則を事前に開示し、一般的な分布性能と混同しない。

全法が同じ採否になった場合も、その結果を残す。幅・cold費用の比較は継続できるが、追加縮約能力の実証にはならない。未判定を分母から除かない。

### 11.4 結果の評価

主張は次を区別して判定する。

- 厳密な証拠のsoundness。
- 旧安価対照に対する同modeの幅差・採否差。
- H単位の一mode削減の成立数。
- 同等SDPに対する同保証達成までの取得費用・未判定率。
- density-aware対照との保証幅・費用・メモリの競合。

同等SDPに対する理想的な保証優位を成功要件にしない。旧2-base gateを新taskへ流用して成功に変更しない。新しい大きな改善率閾値を結果後に導入しない。

各input/familyと全体を報告し、非支配な取得費用・幅の点が残るかを見る。計測誤差や未判定が大きければ結論保留とし、有限pilotの結果を全classでの最適性にはしない。

件数、具体seed、CPU/メモリ、primitive上限などの実行詳細はCodexが結果を見ずに設計し、実行planとして固定する。本レビューはその未作成planを実行認可したものではない。

## 12. 代替研究方針の比較

| 方針 | 採否と理由 |
|---|---|
| S1の旧族最適幅をさらに探索 | 不採用。登録旧ORの下端は解析的に決着済み |
| OCCの旧対照分離だけで主要claimを確定 | 不採用。既知Fantope/SOSとの関係を踏まえると不十分 |
| generic SDP/SOSを含む証拠取得・一mode task pilot | 条件付き採用。現時点で未解決な算法価値を直接判別する |
| すぐ実分子や多modeへ拡張 | 不採用。classと情報取得の課題が未解決で、現保証をそのまま適用できない |
| 監査・実装比較だけを成果の中心にする | それだけでは当初の算法改善の目標を満たさない |
| N1/N2再開 | 新しい構成原理がないため停止・保留を維持 |
| N3全体を今終了 | 現段階では早い。ただし次の比較で実質差が残らなければ中心候補から外す |

## 13. 論文の着地点

現時点で支持できるのは、限定classのOCC十分条件、旧scalar条件族に対する厳密分離と機構、固定S1の達成幅改善である。これらはmethod sectionや数学的例として価値を持つ。

候補となる独立論文の中心は、例えば「固定粒子数を利用した、有限費用・検証可能なHamiltonian縮約certificateの生成と選択」である。ただし仮題にすぎず、次の要件は未達である。

- 既知のFantope/SOS利用を超える具体的な算法差、または非自明な理論構成。
- 同等・強い対照を含めた取得費用/保証の関係。
- 未使用入力におけるcold取得とtaskの判別。
- U/c取得を含む限界の明示。

量子アルゴリズム論文として総費用の改善を主張するには、後段の状態準備・simulation・readout等の契約が別に必要。実分子を必須条件にするわけではないが、現在のmode数の減少を量子総資源優位と呼ばない。

現時点では「投稿準備へGO」や「中心テーマ確定」とは判断しない。比較対象を強くした結果、既知手法の安価な特殊化に過ぎず算法差も利益も残らないと分かれば、保証primitive・否定的結果・再利用可能なコードとして整理し、主研究の探索を別方向へ戻す。

## 14. 次の担当と作業のまとめ方

**次の担当：Codex。** 次を一つのまとまりとして進める方針を採択する。

1. 本レビューの同値SDP、SOS包含、比較対照の役割、誤差区間の注意を仕様v2への追補として反映する。旧文書・結果は保持する。
2. 2020年検証付きRDMと2026年SOSの対応を文献表へ追加する。未取得全文や未解決の包含は明示し、調査結果を都合よく新規性認定へ変えない。
3. 幅generator、同等SDP、固定density-aware参照、安価なfull-H lower、selector、独立checkerを比較可能にする。数学的に解決済みのS1旧族scanは繰り返さない。
4. 合成fixture、敵対的証拠、境界・縮退、cache、cold/warm、失敗分類、resource accountingを技術検証し、通常の問題はCodex内で修正する。
5. 結果未取得の小さな比較集合、誤差予算、資源と計測規則を固定し、source/input/planとGitHub取得資料を揃える。
6. 新しい科学実行の明示承認が必要な位置で停止する。

同じ科学的範囲を満たす実装・資料整備・テスト完了のたびに、GPTの詳細レビューを要求しない。技術的preflightとユーザー実行承認を経て固定pilotを実行し、その新結果が揃った時点で次の重要レビューを提案する。

ただし、保証の前提に破綻がある、比較対照を実質的に変えなければ実装できない、別の物理oracleが必要、主要task/精度の変更が不可避である場合は、変更前にGPTへ科学的問題を返す。

## 15. STOP・認可・成果物の保全

本レビューは進行方針と比較設計の科学的判断であり、新しいrunner実行の認可ではない。消費済みone-shot認可の流用、source/入力の無断差し替え、旧結果への追記上書き、失敗後の自動再試行は行わない。

元primaryの`NO_ADDITIONAL_ACCEPTANCE_OBSERVED`、旧2-base gate、saved-widthのposthoc位置付け、旧族分離、今回の独立照合はそれぞれ別の結果として保持する。

global manifestのdescription欠落は既存の技術的問題として残し、今回の科学証拠全体をglobal PASSと呼ばない。他Trackやrootのdirty差分、外部編集を巻き込まない。

## 16. 今回の最終判定

**限定継続・比較実装へGO。**

前回の「理論・前処理算法として優先的に継続」を、今回得た統合仕様と追加文献を踏まえて具体化した。方向転換の中心は、弱い旧族を超えたこと自体から、同等/強い既知手法を相手に有用なcertificateを取得する算法へ研究の問いを移すことである。

N3を続ける前提を無条件に置いていない。次の限定比較で非自明な差が残らなければ、中心研究候補から外す。反対に同等保証の取得効率や実際の縮約判断に再現可能な利益が残れば、そこで初めて論文の主要claimを強める。

---

## 付録A. このレビューの根拠資料

### 固定GitHub資料

共通root：`https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/`

| ID | commit | path |
|---|---|---|
| R1 | `2de3b26fc36df9e99b7504631c97635f1473cb51` | `docs/research/n3_theory_task_integration_v1_20261012.md` |
| R2 | 同上 | `docs/research/n3_theory_claim_map_v1_20261012.md` |
| R3 | 同上 | `docs/research/n3_literature_correspondence_v1_20261012.md` |
| R4 | 同上 | `docs/research/n3_width_reduction_v2_spec_20261012.md` |
| R5 | 同上 | `artifacts/n3_theory_task_integration_v1/2026-10-12/saved_analytic_checks.json` |
| R6 | 同上 | `artifacts/n3_theory_task_integration_v1/2026-10-12/review_correspondence.json` |
| R7 | `837abb5b200c61720c8b5097901a437c89d8aed3` | `docs/research/n3_theory_task_integration_v1_publication_20261012.md` |
| R8 | `86c69983544064d1ed7ee7910634c541c7b982b7` | `docs/research/n3_old_family_v1_results_20261012.md` |
| R9 | `652667187ce83c7c36041820a0a752669134865f` | `docs/research/n3_saved_width_v1_results_20261012.md` |
| R10 | `e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c` | `docs/research/n3_occupation_transfer_v1_results_20261012.md` |
| R11 | `4c719a72eb8caae2f77fc746bd4672d610d97610` | `docs/research/n1_decode_recovery_v1_results_20261011.md` |
| R12 | `ae7434452450e80ec007d65417335e18a063fa1e` | `docs/research/hamiltonian_algorithm_design_results_20261011.md` |
| R13 | `c24dcede10ed9726e1e0b1030eca749ba7bc5cba` | `docs/research/研究概要・現状.md` |

R1–R7は今回のlive GitHub取得で本文を確認した。R8–R13は会話中の固定取得資料・以前のレビューを引き継いだ歴史資料であり、各branchの現在HEAD全体を横断して再監査したとは扱わない。

### 外部一次資料

[L1] X. Liangほか, “On Generalizing Trace Minimization,” arXiv:2104.00257v1. `https://arxiv.org/pdf/2104.00257v1`。固有値和のtrace変分原理の位置付け。

[L2] V. Q. Vu, J. Cho, J. Lei, K. Rohe, “Fantope Projection and Selection: A near-optimal convex relaxation of sparse PCA,” NeurIPS 2013. `https://papers.nips.cc/paper_files/paper/2013/file/81e5f81db77c596492e6f1a5a792ed53-Paper.pdf`。第2節式(2)とFantope定義を参照。sparse-PCA固有の統計的結論をN3へ適用しない。

[L3] G. Dusson, I. M. Sigal, B. Stamm, “The Feshbach–Schur map and perturbation theory,” arXiv:2105.02058v1. `https://arxiv.org/pdf/2105.02058v1`。N3独自の射影前提と論文の孤立固有空間仮定を分ける。

[L4] D. B. Chamakiほか, “Self-consistent Quantum Iteratively Sparsified Hamiltonian method (SQuISH): A new algorithm for efficient Hamiltonian simulation and compression,” arXiv:2211.16522. `https://arxiv.org/abs/2211.16522`。本レビューの独立確認は公開abstractまで。全文の対応箇所はR3のCodex記録に依存。

[L5] J. Greiner, J. Gauss, J. J. Eriksen, “Error Control and Automatic Detection of Reference Active Spaces in Many-Body Expanded Full Configuration Interaction,” arXiv:2406.11343v1. `https://arxiv.org/html/2406.11343v1`。経験的誤差推定の位置付けを参照。

[L6] M. L. Overton, R. S. Womersley, SIAM J. Matrix Anal. Appl. 13 (1992), 41–45. DOI: `10.1137/0613006`。原典全文未確認、R3に記録された限界を保持。

[L7] A. J. Coleman, Rev. Mod. Phys. 35 (1963), 668–686. DOI: `10.1103/RevModPhys.35.668`。原典全文未確認。詳細定理を検証済みとして引用しない。

[L8] D. Chaykin, C. Jansson, F. Keil, M. Lange, K. T. Ohlhus, S. M. Rump, “Rigorous Lower Bounds for the Ground State Energy of Molecules by Employing Necessary N-Representability Conditions,” J. Chem. Theory Comput. 16 (2020), 7342–7356. DOI: `10.1021/acs.jctc.0c00497`。出版社/著者所属機関のabstractを確認。`https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00497`、`https://tore.tuhh.de/entities/publication/fce317e4-612d-4332-bfff-02471b71c072`。

[L9] N. C. Rubin, G. H. Low, A. E. DePrince III, “Near-frustration-free electronic structure Hamiltonian representations and lower bound certificates,” arXiv:2602.05069v1 (2026-02-04). `https://arxiv.org/html/2602.05069v1`。第II節のweighted SOS、粒子数等式とPSD Gram形式、第IV節のv2RDM双対との関係を参照。出版情報はDOI `10.1021/acs.jctc.6c00318`でも確認したが、数式の議論は取得したarXiv v1に固定。論文の数値図表やspeedupをN3の性能として転用しない。

[L10] M. Sion, “On general minimax theorems,” Pacific Journal of Mathematics 8 (1958), 171–176. DOI: `10.2140/pjm.1958.8.171`。Theorem 3.4、p.174。`https://msp.org/pjm/1958/8-1/pjm-v8-n1-p14-s.pdf`。第6節の両集合のcompact/convex性と連続affine性に適用。

## 付録B. 読み手が取り違えてはいけないこと

- 新しい文献を見つけたことは、N3の一mode構成全体が既報と確定したことではない。
- 旧OR最適下端の確定は、全既知保証法の最適下端の確定ではない。
- generic SDPとの数学的同値性は、同じ有限budget/精度で同じ証拠が発見される保証ではない。
- RDM/SOSの数値solver出力は、検証済みの下界certificateとは別である。
- 今回のSOS包含と同値SDPは解析的導出であり、新しい実験成績ではない。
- 幅が狭いことは、真誤差が小さくなったことや量子費用が同率で減ったことではない。
- あるmodeで新たに認証できたことは、H全体で旧法より多くmodeを削除できたこととは限らない。
- 元H fallbackは安全だが、縮約成功ではない。
- 同じS1で既存証拠を再照合することは、独立入力への移送ではない。
- 実装へ進む科学的GOと、新しい計算を実行するユーザー認可は別である。
