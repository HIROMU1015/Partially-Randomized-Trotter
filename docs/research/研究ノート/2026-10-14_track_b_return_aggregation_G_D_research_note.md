# Track B研究ノート：有限Taylor RTEのreturn集約、G系列からD1〜D3まで

2026-10-14 JST。将来のCodexが研究を引き継ぐための日本語記録。

**現在の判断：return集約の構成・生成法と、実装資源の成立条件・限界を研究の中心として保持する。D1〜D3の現在の数値探索は一区切りとし、mandatory STOPを維持する。** 最新GPT科学レビューは完了している。今回は利用者の指示により研究ノートだけを作成し、論文原稿、新実験、追加レビューは開始しない。

このノートの数値は固定資料の代表値を要約した表示値である。正確な有理数、証明、実行契約、監査、分類は末尾の固定commitを正本とする。外部再現・実機測定・分子PR/QPE全体の総費用評価を済ませた記録ではない。

## 1. 何を研究してきたか

Track B全体の目的は、Partially Randomized Trotter（PR）のランダム部分に適用できる新しい時間発展構成や、資源効率を改善する機構を見いだすことである。Taylor法が最善であることを前提にはしていない。

初期の有限taskを戻すPF係数設計（B-F）、compact BCHに対応するDF adapter（B-M）、probabilistic synthesisの配置（SP）、generic whole-commutator補正、Gaussian積圧縮は、それぞれの限定レビューで一区切りとなった。理由には限定的negative、既知構成との同値性、独立した方法差や実装利益の不足があり、同じ種類の性能negativeとして一括しない。これらの終了判断は本ノートで解除しない。Track AのPR資源評価、N3の保証幅・縮約とも別系列である。

現在につながる問いは、**同じ有限Taylor演算子を、どのランダムunitary ensembleとして生成するとよいか**である。G1〜G5ではdegree-local再配分と既存representationの混合を調べ、限定classの分離と強いCTS対照を区別した。その後G6〜G10で、隣接同labelの消去によるreturnを集約し、係数表だけでなく実際に生成できる構成を調べた。D1〜D3はその既存構成を、同じ理想時間発展・精度task、境界相殺、誤差保証、合成価格へ接続した検証である。

研究として残す問いは「P5が常に勝つか」ではない。係数質量の削減が、生成情報、有限bit、controlled位相、十分測定予算、回路価格を払った後にどのような資源差になるか、その成立条件と限界を説明することにある。

## 2. return集約と生成法の核心

正の係数 \(\sum_i p_i=1\)、Hermitian involution \(Q_i^2=I\) に対し、

\[
R=\sum_{i=1}^{L}p_iQ_i,\qquad
P_m(-i\sigma xR)=\sum_{n=0}^{m}\frac{(-i\sigma xR)^n}{n!}
\]

を扱う。構成の正値性の範囲は奇数 \(m\)、\(0<x\le1\)、\(\sigma=\pm1\)。raw語から隣接する同じlabelを消去したreduced語を \(u\)、長さを \(|u|\) とする。IID labelによる到達確率を \(P_n(u)\) とすれば、

\[
a_u=\sum_{\substack{|u|\le n\le m\\n-|u|\;\mathrm{even}}}
(-1)^{(n-|u|)/2}\frac{x^n}{n!}P_n(u),\qquad
P_m(-i\sigma xR)=\sum_u(-i\sigma)^{|u|}a_uQ(u).
\]

集約は \(Q_i^2=I\) だけを使う。Pauliの交換・反交換や異なる語の物理的一致まで無料で収集するものではない。\(\chi=\sum_i p_i^2\) に対する \(P_{n+2}(u)\le(n+1)\chi P_n(u)\) とalternating sumから、このshort-step域で \(a_u>0\) を示す。

偶数語parentの係数 \(a_u\) とchild係数の和 \(s_u=\sum_i a_{iu}\) を、\(d_u=\sqrt{a_u^2+s_u^2}\)、\(\phi_u=\arctan(s_u/a_u)\) によりpairする。childは空parentなら全label、非空なら先頭labelと異なるlabelである。unitaryは

\[
V_{u,i}=(-i\sigma)^{|u|}e^{-i\sigma\phi_uQ_i}Q(u),
\qquad q(i\mid u)=a_{iu}/s_u
\]

となり、その平均がparentとchildの項を再構成する。回転generatorは単一の \(Q_i\) であり、一般の奇数語全体がinvolutionだとは仮定しない。operatorの積順序と回路の時間順序も区別する。

一般局所法は、既知のfree-product Green再帰を有限次数へ切り、必要なparentとchildの係数だけをqueryする。ordinaryの係数質量 \(B_{\rm ord}\) をenvelopeとして、raw語がreducedでない試行や局所rejectを量子実行前のzeroにする。全parent表と真のglobal normalizer \(B_{\rm full}\) を入力に要求しない。理想的には

\[
Z=B_{\rm full}/B_{\rm ord},\quad E[X]=\text{対象axis信号},\quad
E[X^2]=B_{\rm ord}B_{\rm full},\quad |X|\le B_{\rm ord}.
\]

zeroを含む全試行数を分母とする。真の \(Z\) が不明なら、無料でそれを十分shot数削減へ使わず、取得可能な上界を使う。normalizationの改善だけからnative総費用の改善は導かれない。

P5特殊化では、root、長さ2のordered pair、長さ4のfirst-label群を用い、normalizerを高々 \(L^2+1\) 群から構成する。一般法と違い、少数群の初期化でclosed normalizerを利用できる。初期化は \(O(L^2)\) 有理演算、条件付き語生成は \(O(L)\) だが、線形の群選択は \(O(L^2)\) であり、sampler全体を無条件に \(O(L)\) としない。bit複雑性、実時間、回転角の取得・合成費用は別に残る。\(L\) はinvolution項数でありqubit数やDF rankではない。

証明と一般生成手順は[既存技術ノートv1.0][theory]を参照する。

### 六方式の違い

| 方式（保存名） | 構成と情報の違い |
| --- | --- |
| ordinary | 隣接Taylor次数をpairし、labelをIID生成。形式returnを収集しない。 |
| partial-return（partial_return_tail） | identity/root returnの一部を吸収し、残りはordinary tailとして扱う。 |
| P3+tail（closed_P3_tail） | P3のclosed部分と次数4・5のordinary tail。同じ有限P5 targetを実現し、Taylor次数を3に変えた比較ではない。 |
| full-return／local full | 全形式returnを集約し、局所queryとreject/zero-fillで生成。ordinary envelopeと取得可能な上界で予算を組む。 |
| closed P5（closed_P5_full） | full-returnと同じ理想ensembleのP5専用少数群生成。normalizer取得とfinite-bit予算が異なる。 |
| matched CTS | 同じ有限P5のPauli係数を収集した対照。I1情報を使うfinite specializationで、CTS全体の最適実装ではない。 |

### 有限bitとnative実装

実装では確率をfull-support dyadic値 \(q_e\) へ丸め、実際のproposalで \(w_e=\widetilde\alpha_e/q_e\) を計算する。\(q_ew_e=\widetilde\alpha_e\) はexact rationalで保持し、係数近似のmean biasは誤差予算へ戻す。理想式とdigital平均を無条件に同一としない。

保存lawからacceptance、二次moment、range、十分Bernstein shot数を算出し、finite-bitとnative合成誤差を分ける。unitaryのglobal phaseはcontrolled枝の相対位相になるためstrict operator guard（`up_to_phase=false`）を使い、逆回転は実sequenceのadjointとして扱う。RZ数を実合成T数と呼ばない。

## 3. G系列で何が分かったか

| 段階 | 目的・確認された結果 | 残る限界／研究上の扱い |
| --- | --- | --- |
| G1〜G3 | 固定P3のdegree-local係数classを解析し、有限law・既知return対照を実装会計へ接続。G1のbackend検査は人工LPの技術証拠。 | 旧RA-D0全体の成功や無制限B2/B3最適性とはしない。 |
| G4・G5 | 限定B2 classに対するJ1 lawの分離を認証。一方、同targetのCTS lawはその選択lawを上回り、固定degree-local classの実用routeを区切った。 | 別辞書・別confidenceを含む全アルゴリズムの不可能性ではない。 |
| G6 | 形式returnの係数恒等式・short-step正値性・局所query・global normalizerなしの生成を整理。 | 実装資源の利益や構成全体の独立新規性は未確立。 |
| G7・G8 | controlled primitiveを条件付き費用で会計し、on-demand生成からlive synthesis cache・native IRへ接続する経路を確認。 | provider単価未取得の条件付き結果、有限controlのcache coverageを一般費用へ拡張しない。 |
| G9 v2 | P5少数群構成と明示3-qubit providerを接続。登録六方式ではP5のT利益とCTSのCX利益が併存。 | 同じ有限P5の比較。Taylor remainderを含む同一理想時間発展の精度taskはD1で追加。 |
| G10 v3 | 同一次数m=3/5/7の17行。一般fullの登録T費用は各closed対照より高く、m7では固定辞書・予算policyの下界も限定準備費用域で対照を上回った。 | 異なる次数の同精度総費用比較ではない。一般fullの性能探索は一区切り、P5構成と条件付き利益は保持。 |

G1〜G5は2-qubitのdegree-local toy、G9〜D3は別の3-qubit providerを使う。段階間の差を同一入力の改善率として読まない。G10の途中technical failureと、別source・authorization・markerによるv3完了はともに保存されている。最新レビューは過去のtechnical分類を書き換えるものではない。

## 4. D1〜D3の共通taskと費用モデル

以下の代表値は**既知の3-system-qubit synthetic development入力**に限定する。

- \(p=(1/5,3/10,1/2)\)、\(Q_0=Z_0\)、\(Q_1=V_1^\dagger Z_1V_1\)、\(Q_2=V_2^\dagger Z_2V_2\)。\(V_1=R_{XX,01}(\pi/4)\)、\(V_2=R_{XX,12}(\pi/4)R_{ZZ,01}(\pi/4)\)、\(R_P(\theta)=e^{-i\theta P/2}\)。
- 入力 \(|000\rangle\)、総時間 \(\tau=5/7\)、\(\sigma=+1\)、\(r=1,2\)、step \(x=\tau/r\)。理想 \(e^{-i\tau R}\) に対し \(M_r=P_5(-i\tau R/r)^r\) を用いる。
- 同じrの六方式は同じ有限targetを表現する。異なるrでは有限多項式が変わるが、同じ理想時間発展と精度taskへ誤差を戻す。
- 全方式に同じI1アクセス。分子・geometry・軌道basis・DF rank・split \(L_D\)・PF delta windowは適用外。Pauli情報を取得不能とする実問題の優位性を示すfixtureではない。
- axis許容誤差は各 \(1/200\)。各12行ledgerのfamilywise失敗予算は0.05。D3の四ledgerを同時に0.05保証する意味ではない。保存Nは一axis当たりの十分予算で、実測shot数・必要最小shot数ではない。

保存されたpositive-step条件付き単価 \(d\)、step acceptance \(Z\)、literal境界節約 \(S\)、一axis十分予算 \(N\) を用い、各指標の費用は

\[
C_{i,r,k}=rZ_i^r d_{i,k}-S_{i,r,k},\qquad
G_{i,r,k}(h_k)=2N_{i,r}\bigl(C_{i,r,k}+Z_i^r h_k\bigr)
\]

と会計する。\(h_k\) は受理された全回路の準備・読出し単価。\(h=0\) は比較モデルの断面で、無料だった実測ではない。T・CX・native1Qは別座標であり、Tを含むnative1QへTを再加算しない。未知のclassical online生成費用は未取得で、0に置かない。

native費用は合成primitiveの加法モデルとliteral逆gate簡約まで。一般commutation、sequenceを展開した全体最適化、whole-circuit compiler、fault-tolerant実時間の測定ではない。

## 5. D1、D2：同精度比較と境界相殺

**D1**はTaylor remainder、finite-bit、native誤差を同じ理想taskへ戻し、六方式×r1/r2の12行を取得した。1,890 eventと36 synthesis keyの費用・誤差・identityは保存監査済み。固定rでclosed P5のT費用はpartialよりr1で約1.16%、r2で約5.84%低かった。r1は主に十分shot予算、r2は主に単価の差が効く。これは各方式にr選択を許した主比較の改善率とは別である。

**D2**はr2の共通規則で219,303 step pairのliteral境界相殺を一回評価し、D1のlaw・N・合成価格を維持した。P5対partialの固定r2 T差は約5.86%へ移り、D1からの追加利益は約0.02 percentage pointである。CXのpartial/P5交点は準備費用約3.515から約6.637へ移った。全六方式ではCX最小領域はCTS→partial→P5となり、二方式の交点だけで最良方式を決められない。

保存された相殺深さ分布は構造分析に使えるが、depth別費用への完全帰属traceは未取得。D3では保存IRの静的確認により、このdomainの境界相殺はR tokenを含まないliteral-onlyと確認し、節約額を再利用した。pair再評価や新sequence内部の相殺は行っていない。

## 6. D3：保証と合成価格を分離した四ケース

Hermitian contraction \(\|R\|\le1\)、実数xなら、Taylor積分剰余とunitarityから

\[
\|e^{-ixR}-P_5(-ixR)\|\le |x|^6/6!
\]

を用いる。従来G0の \(3|x|^6/6!\) に対し改善G1を比較し、反復では \((1+\delta)^r-1\) などの積誤差上界へ接続する。G0も有効な旧保証であり、誤った結果へ再分類しない。G1が変えるのは認証予算で、物理回路や実際のTaylor誤差ではない。

価格P0は旧 \(10^{-6}\) sequence、P1は同じtool・seed・strict phase条件で新取得した \(10^{-7}\) の36 key。**全四ケースのnative誤差会計は旧 \(10^{-6}\) に固定**した。P1の実際の誤差改善を追加の統計幅として使わないため、価格変更だけでNは変わらない。D3 v2は36合成・48行を完了し、2,579項目の保存監査を通過した。

次表は \(h=0\) のT費用であり、各方式へ同じr1/r2選択を許した結果である。保証のG0/G1は過去のG系列番号とは別の記号。

| ケース | 全六方式×二反復のT最小 | primary partial/P5のT最小 | 最良partialに対するP5削減 |
| --- | --- | --- | ---: |
| G0_P0 | r2 closed P5 | r2 closed P5 | 約1.360% |
| G1_P0 | r1 closed P5 | r1 closed P5 | 約1.161% |
| G0_P1 | r2 closed P3+tail | r1 closed P5 | 約0.713% |
| G1_P1 | r1 closed P3+tail | r1 closed P5 | 約1.195% |

primaryのP5 T利益は維持されたが、全方式の最良候補は同じではない。G1_P1ではCX最小はCTS、native1Q最小はpartialであり、T最小を全指標優位としない。

### P3+tail／P5の小さい逆転を説明する保存値

G1_P1/r1、\(h_T=0\) について：

| 保存量 | closed P3+tail | closed P5 |
| --- | ---: | ---: |
| 十分N／axis | 1,103,489 | 1,083,653 |
| P0の全trial T単価 | 143.277319 | 139.489994 |
| P1の全trial T単価 | 159.700057 | 162.899692 |
| P1の二axis総T forecast | 352,454,511.977 | 353,053,479.316 |

両方式のacceptanceは1、biasと残余幅は同じ。P5は二次moment・range・Nで有利だが、新価格では単価が高く、総費用はP3+tailが **598,967.339 T（P5を分母に約0.170%）** 低い。丸めや量子sampling noiseによる差ではなく、固定有理数会計内の差である。一般的なP3優位の証拠とはしない。

主要な価格差は先頭三root eventのkeyである。P5のangle ratio `2189372/2921709` はT価格68→80、P3+tailの `310/413` は70→78となった。保存incidenceで重み付けすると、二keyが約+1,500.933万Tの「P5−P3+tail」margin移動を作り、他keyのnet約−51.797万Tを合わせ約+1,449.136万Tとなる。旧margin約−1,389.239万Tを上回り、上の約59.897万Tの逆転になった。統計予算変更の寄与は混ぜていない。

この差は準備費用にも依存する。同じr1二方式の保存affine差は約 \(598967.339-39672h_T\) で、tieは約15.098。全候補のh領域は既存機構分析を参照する。記号的なtie価格は、新しい合成sequenceで実現可能な価格を証明するものではない。

### 保証変更でr選択が変わる理由

旧保証はr1の統計残余幅をより大きく消費する。G1ではr1の十分Nが約29.5%減る一方、r2は約0.93%の減少である。反復回路の費用増とN減少の交換が変わり、P0の最良rは2→1へ移る。law、回路、合成価格を変えた利益ではない。

保証と価格のinteractionは \(2\Delta N\,\Delta C\) という費用積の会計項であり、新しい量子機構とはしない。同じ実装ならG1はG0以上に強い保証であるため、G0は保証の保守性を見せる対照として残す。P1も価格感度の対照であり、自動的な最適precisionではない。旧価格のG1_P0/r1 P5約3.023億Tは依然利用可能で、P1内でP3+tailが勝つことをP1採用の根拠にはしない。

## 7. 確立事項、既知原理、未解決事項

### 今回の証拠から支持されること

- 数学：上記仮定下の係数正値性、parent/child pairing、局所query、未知global normalizerを要求しない生成、P5少数群構成。証明の範囲を小規模テストの範囲と混同しない。
- 固定実装：digital係数・proposal・weight、controlled位相、native価格、十分confidence予算を結ぶ会計が検証済み。全方式は同じrで同じ有限targetを比較する。
- 固定結果：return集約の統計上の利益が実装資源で逆転する場合がある。primaryの限定T利益、CTSのCX利益、保証によるr選択、root価格によるP3/P5逆転が併存する。
- 来歴：G/D各stageの固定source、実行認可、one-shot marker、terminal、STOPを保存。技術失敗と完了runの記録を保持し、今回変更しない。

### 既存研究で知られている部分

Euler pairingとordinary RTE、低次数のidentity吸収、free-product Green再帰、zero-fillとdiscardのnormalization区別、importance samplingの費用×moment最適化、CTSのPauli収集、probabilistic synthesisのsamplingと回路費用の交換は既知の構成・原理である。これらを独立した初発見として扱わない。

一次本文・取得版・該当節・claim対応は[先行研究照合][literature]にまとまっている。Wan–Berta–Campbell、Zhao–Yuan、Aomoto–Kato、Cugini–Atif–Subaşı、Peetz–Smart–Narang、Koczor、Ross–Selinger、PR元論文との対応を継承し、本作業では再調査しない。channel平均とcontrolled first operator moment、情報アクセス、辞書、confidence policyが異なる場合を同一taskとしない。

### 未確立のこと

構成の結合全体の独立新規性、最適な表現選択則、未取得価格への予測、他providerや実問題への一般化、合成器・seed・精度への頑健性は未確立。今回のroot機構説明は、独立入力に対する事前予測検証ではない。

最適ISを含む強いproposal対照、whole-circuit最適化、setupとは別のclassical online生成費用、bit計算量・cache/角度取得費用も未取得部分がある。既存CTS対照を最強の全CTSだとはしない。分子DFのアクセス費用、PRのdeterministic部分、長時間反復、状態準備、QPE/RPEまで含む優位性は示していない。約0.170%の差や本fixtureの順位から一般的優位・論文化十分性を認定しない。

## 8. 現在の方針と再開時の入口

最新GPT判断は、構成・生成法と有限精度資源の条件付き評価を一体として残し、現在の数値系列を一区切りにするもの。本作業はその判断を研究ノートへ記録しただけで、科学的採否の新判断をしていない。原レビューの原稿化提案に対し、今回の利用者指示は研究ノート作成のみである。

必要性が明確になるまで新precision、seed、provider、rを探索しない。旧run・markerを再利用しない。新科学計算には別途ユーザーの明示承認、GPTの重要科学レビューにも明示的な開始承認が必要である。**mandatory STOP。追加の計算・研究・レビューを自動開始しない。**

再開時は、まず本ノートと最新GPTレビューを読み、構成の疑問はv1.0、数値の疑問はD3原rowと最終機構分析、既知法との対応はclaim表へ進む。新しい主張が他provider、事前選択器、最適sampling、実回路全体、分子PR/QPEのどれを必要とするかをGPT側で判断する。その判断前に具体的な新実験を採択しない。文書に残る過去の「承認待ち」は当時の状態で、現在のrun許可ではない。

## 固定参照と原本identity

| 確認したい内容 | 固定資料 |
| --- | --- |
| 現在の研究判断 | [最新GPT科学レビュー][latest]（原本copy commit `79a1ff0a45e84f3e45c09cc7ede9255310449302`） |
| 構成・証明・一般生成・P5特殊化 | [return集約技術ノートv1.0][theory]／[G6先行研究対応][g6art] |
| G系列各stageと終了判断 | [G資料索引][gindex]／[2026-10-11 endpoint記録][endpoint] |
| D1・D2の科学レビュー | [D1/D2統合GPTレビュー][d12review] |
| 同一理想taskへの接続 | [D1結果][d1]（`fc96571881f301435a655f057709df4d6c1678e0`） |
| 境界相殺 | [D2結果][d2]（`d1c59b28f4370cc7be65cff60d1ec877fd1491f5`） |
| 保証×価格の事前条件 | [D3事前計画][plan]／[数学確認][math] |
| 48行、exact資源表、terminal、実行・保護対象 | [D3 v2結果と監査入口][d3]（`f0df9a3b36bc3540f0c83941871fa0d3110ddbf7`） |
| root/key寄与、十分N、h領域 | [D3最終機構分析][mechanisms]（`534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f`） |
| 一次文献との対応と主張の境界 | [先行研究照合・claim-to-evidence][literature]（`d3a232a446062e6baadef2b69eab87faa0285363`） |

D3のscience source S2は `dceccaa774229527a46a1386131583a4f618f8ad`、authorization-only A2は `daa2569208a945f8e41b3c4d0de715c1e11b104b`。原結果分類 `D3_GUARANTEE_PRICE_MAP_COMPLETE_AWAITING_GPT_REVIEW` は当時のterminalとして保持する。現在のレビュー完了を理由に上書きしない。

最新GPTレビュー原本は `docs/research/track_b_d1_d3_post_literature_scientific_review_20261014.md` にbyte一致で保存した。40,122 bytes、SHA256 `57c11366a617de8adcf2451615141e7c14b889ef2b9687ba1bda1989d5f9f670`。固定commitのGitHubから再取得し一致を確認した。本ノートはこの原本、既存資料、今回の利用者指示を要約した新しい記録であり、過去ノート・原データの代替ではない。

[latest]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/79a1ff0a45e84f3e45c09cc7ede9255310449302/docs/research/track_b_d1_d3_post_literature_scientific_review_20261014.md
[theory]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md
[g6art]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md
[gindex]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/README.md
[endpoint]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/research/研究ノート/2026-10-11_track_b_return_aggregation_endpoint.md
[d12review]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/research/track_b_d1_d2_integrated_scientific_review_20261013.md
[d1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md
[d2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1c59b28f4370cc7be65cff60d1ec877fd1491f5/docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md
[plan]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e1011e3cb7e306a1cc992644aa70af4bf6ff514/docs/tracks/algorithm_codesign/d3_guarantee_price_preregistration_v1_20261013.md
[math]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dceccaa774229527a46a1386131583a4f618f8ad/docs/tracks/algorithm_codesign/d3_mathematical_confirmation_20261013.md
[d3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md
[mechanisms]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md
[literature]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md
