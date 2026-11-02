# Track B全体：新RQ探索・既存成果の発展・一時休止の科学的レビュー

- 作成日：2026-10-13（JST）
- レビュー開始承認：利用者の「レビューを開始して」
- 主対象：`HIROMU1015/Partially-Randomized-Trotter` のTrack B
- 今回の状態確認基準：Gaussian積生成closure `64bb39fdcc4928d0f026b63f874c8bf4be2e69ab`
- 主要な履歴：G5、G6–G10、BF/BM/SP、DF三次補正E/T、Gaussian積生成監査、Codex横断調査
- 共通指示：`research_project_instructions_gpt_codex_review_approval_2026-10-12.md`
- レビュー状態：**完了。現在の候補集合から新しい実装・計算系列を連続して立ち上げる運用を一時休止することを推奨する。**
- 新研究候補・新科学契約：本レビューでは採択しない。
- 範囲：PR改善一般の不可能性、Track Bの永久終了、既存成果の無価値を認定するものではない。Track A・N3の独立した方針・実行へ介入しない。
- 実施内容：文書・source・一次文献の読取り、数学的な関係の検討、保存表示スカラーの比の算術、本文の作成のみ。
- 未実施：新しい行列・回路・Hamiltonian・trajectoryの生成、compile、T合成、既存runner/tests再実行、GitHubへの書込み、authorization・marker作成。

## 0. 結論と判断の水準

今回の問いは、直前のGaussian候補をもう一度採点することではない。BF/BM/SP、return集約、DF三次補正、Gaussian積生成までの経緯を踏まえ、Track Bとして新しいRQを具体化するか、既存成果の発展へ進むか、新規実験を伴う探索を一時休止するか、である。

**現時点では一時休止を選ぶ。** 具体的には、既に非採択となった候補の条件を少しずつ変えたり、新しい名前を付けたりして、設計→source準備→実行監査→closureを自動反復する運用を止める。これは理論を考えることを禁じる指示でも、今後のPR改善可能性を否定する判断でもない。

判断は次の三層に分ける。

1. **証拠として確立したこと**：有限meanの構成、限定classの分離、P5の少数群構成、限定native費用の利益、局所whole/Pauli費用分離、Gaussian構成の既知手順への対応は残る。
2. **未確定なこと**：同じ最終物理taskでのTrack B候補の総資源優位、一般DF accessでの利益、独立した新規性、現在のGaussian候補の期待費用は未確定である。
3. **研究運営上の判断**：未確定な欄を全て埋めることを当然の義務にせず、現時点で新しい実装系列への追加投資を優先しない。

今回比較した中で、既存成果の発展として最も根拠が具体的なのは、**closed P5の構成を、同じ物理的時間発展精度・同じ利用可能情報・実際の回路費用へ接続する問い**である。しかし、これを今回の新しい実験計画として採択はしない。既存の固定P5結果の価値と、その先への投資判断は別である。

| 項目 | 今回の判断 |
|---|---|
| generic whole三次補正のclosure | 維持 |
| Gaussian積生成候補のclosure | 維持 |
| G5固定辞書・一般full-returnの旧終了判断 | 維持 |
| closed P5・一般構成・局所分離の成果 | 保持。全てをnegativeへ統合しない |
| closed P5の物理時間発展への拡張 | 将来再検討し得る具体的RQ。今回は未採択・未実行 |
| 新しい相関sampler／補正／compiler系列 | 今回採択しない |
| Track Bの新規実験主線 | 現候補集合に基づく自動的な継続を一時休止 |
| Track A／N3 | 別系列。変更・統合・終了指示なし |
| 新しい科学計算 | 認可しない。既存mandatory STOPを維持 |
| 次の必須Codex作業 | なし。記録反映が必要なら本文への参照追加だけでよい |

## 1. 目的を変えずにレビューする

Track Bの中心目的は、PR内部の構成・時間発展・乱択実装を改良し、同じ物理的精度・成功条件の下で必要な資源を減らすことである。既存PRを評価するTrack Aや、別taskのHamiltonian縮約・保証幅研究を、Track Bへ無言で置き換えない。

### 1.1 新しい数学primitiveを発明しなければ研究にならない、とはしない

既知部品の組合せでも、新しい適用条件、具体的な計算手順、保証、取得費用の節約、再現可能な資源上の効果があれば研究上の寄与になり得る。定数因子改善も、それが科学的に意味のある差なら対象になり得る。漸近指数の改善、全baselineへの勝利、pilot前の新規性完全証明を必須にする根拠はない。

従って、今回の非採択を「既知のGaussian／LCU／BCHを使ったから不可」という一般則で説明してはいけない。問題は、**その組合せが最も近い既知手順と比べて何を変え、どの利益を生むのかが具体化されているか**である。

### 1.2 研究の目的変更と、対象候補の終了は別

応用・数値研究も正当な研究である。しかし、方法開発の候補で独立差が見つからなかったからといって、自動的に分子benchmarkを始めてよいわけではない。その場合は「この適用で何を解明するのか」を主RQとして選び直す必要がある。

Gaussian候補を区切った前回判断は、そのような目的変更を無言で行わないという意味を含んでいる。[R1]

### 1.3 今回は締切や人物属性を判断根拠にしない

利用者がTrack Bの優先順位について指定していない締切、労力配分率、出版期限を仮定しない。本判断は、現在の候補と取得済み証拠、その先に必要な仕事の内容に基づく。例えば「改善が5%以上必要」といった新しい数値閾値も導入しない。

## 2. 証拠と確認範囲

### 2.1 主な資料

- Gaussian積生成closure・監査・監査後レビュー。[R1–R3]
- DF三次補正のclosure、最新科学レビュー、4条件の保存E/T会計。[R4]
- BF closure、BM同値性監査、SP-1保存結果。[R5–R7]
- G5の固定辞書終了認証と、G6–G10の完成ノートv1.0。[R8–R9]
- Codex横断調査。133 branch確認・主要16資料のbyte照合という調査範囲と、その限界はCodex報告として扱う。[R10]
- Track A H6の公開済み科学レビュー。参考にするが、再実行・再審査を主題にしない。[R11]
- 近接する一次文献。PR、CTS、importance sampling、Markov random compiler、LCU補償・HNCC、corrected PF、controlled Gaussian圧縮、SPRINT。[P1–P8]

新たに取得したBF/BM/SP/G5/完成ノートの内容は、単なるファイル名やPASS件数ではなく、target・比較条件・保存判断まで読んだ。Gaussian closureも固定commitから取得した。sourceの詳細については前段で取得された該当処理と監査の対応を継承し、今回全moduleを新たに精読・実行したとはしない。

### 2.2 本レビューでしていないこと

全133 branchの再調査、G10の全10,936 event再認証、全E/T synthesis列の行列検証、全wrapperの再compile、全metadata検査の再実行は行っていない。現在のローカルworktreeのclean状態を直接観測したとも主張しない。

個別の数理・費用結果については、既存の保存監査・科学レビューの限定を継承する。今回の主題は、それらの結果を別taskへ混ぜずに、次の研究方針を判断することである。

文献の「全ての主張を独立再現した」「関連論文が存在しないことを網羅的に証明した」とはしない。本文取得の成否と確認範囲は第11節に残す。

## 3. Track Bの到達点を、失敗の種類ごとに整理する

### 3.1 BF：固定familyでの追加decision valueが得られなかった

対象は同じ5-stage symmetric fourth-order family、同情報、各arm32 coefficient evaluations、H4固定development条件である。FとLは同じSuzuki5・q1/R10/K2を選び、保存action proxyは20,709,936.722970817。共通参照集合のnative S2は12,924,223.160897588である。[R5]

これはF固有の追加価値を支持しない限定negativeであり、全高次PF、全係数最適化、全precisionの不可能性ではない。旧serialization failureと保存値復元の分類は別に保つ。

### 3.2 BM：具体的な手順・scoreが同情報対照と一致した

固定nested列の三次係数は

\[
K_m=K_{\mathrm{floor}}+\frac{K_A}{4m^2}
\]

と書け、compact BCH再帰から得る同じ係数と一致する。同じDF backend・集約・cache機会を与えれば同じscoreが得られる。[R6]

これは性能実験の敗北ではなく、現adapterを独立手法として区別する理由がなかったという結果である。cacheを禁止した弱い対照へ戻して改善を作るべきではない。

### 3.3 SP：大きな局所・短wrapper利益は残る

SP-1は2-qubit synthetic coherent wrapper、fusion-normalized additive synthesized-primitive T会計である。C,n=16ではDRのNONE比が約0.10260741で、n=32では約3.1606213へ反転し、n=64ではshot capとなる。[R7]

従って「SPは全て改善しなかった」は誤り。一方、actual finite-RTE、actual compiled full-wrapper、一般D/R placement則が確立したとも言えない。ゲート数・角度・符号・outer randomnessの交絡を戻す必要がある。

### 3.4 R0–G5：単なるoptimizer未完了ではなく、強い対照まで到達した

G5は固定toy、同一dictionary、非負degree matching、指定digital誤差課金、precision自由度、任意full-support event IS、共通confidence policyのclassを扱う。Tのclass下界171,032,500に対し、同一の保存CTS lawのT費用は約164,669,575.769880である。[R8]

旧RA-D0のtechnical failureを性能negativeへ書き換えてはいけない。しかし、旧source準備資料の未到達表示だけを根拠に、RA-D0/R1を未消化の新候補として再開するのも不適切である。後続G系列の到達点を読む必要がある。

G5は任意LCU、任意辞書、別のconfidence policy、真の最小必要shot数、全アルゴリズムを排除したものではない。

### 3.5 G6–G10：一般構成と低次数の実装成果が残る

一般の形式return集約、局所係数query、全normalizerを入力に要求しない生成、finite-bit補正、closed P5の高々L²+1群による構成は、完成ノートに整理されている。[R9]

一般fullの登録T費用は対応closed対照より大きいが、これは一般構成の恒等式を否定しない。P5の特定native比較の利益も消えない。一方、固定有限多項式の比較を、同じexponential誤差・同じQPE精度での比較へ変換することはできない。

### 3.6 DF whole補正：固定実装計画の局所分離

3条件でA側completion下界がB側完全event上界を超える。A全Tは未確定、r6 controlledのphaseも未完了である。A/Bは有限stepで同じ平均を実装せず、EとCDも別fixtureである。[R4]

現generic wholeのclosureは維持するが、全whole方式の非優位を示す定理ではない。残りTを埋めれば同じ3条件の既知下界が下がるわけではないため、空欄補完を優先する理由にはならない。

### 3.7 Gaussian積生成：正しい既知構成への対応、性能未判定

理想構成は支持される。しかし、現在のactive-span・反射積・partial Gaussian合成の手順に、既知法から区別できる追加処理が定義されなかった。現sourceの専用hookがないことと、文献上の新規性は別である。[R1–R3]

条件付きwrapper費用は未取得であり、改善も非改善も未判定である。候補を性能で負けた例へ数えない。

### 3.8 全体の読み方

以上は独立な同一実験を何回も失敗した一覧ではない。異なるtask・比較class・費用指標を持ち、研究課題を変更しながら進んだ履歴である。成功率や候補数の統計へまとめない。

共通する問題は、**数理・normalization・局所単価の利益を、同じ利用情報を持つ強い対照との総資源比較へ接続する段階で、追加価値が縮小したか、未確定のまま残ったこと**である。この解釈は履歴の統合であり、新しい一般不可能性定理ではない。

## 4. 一番残っている成果を過小評価しない：closed P5

### 4.1 保存された同一P5 native表

G10の固定3-system-qubit provider、p=(1/5,3/10,1/2)、x=5/7、m=5、共通confidence会計の表示値を使う。以下はT費用の切片、すなわちG_T(h_T)=C_{0,T}+K h_TにおけるC_{0,T}である。量子実機の時間やPR全体のT数ではない。[R9]

| 方式 | T切片 / 10^6 | CX切片 / 10^6 | K / 10^6 |
|---|---:|---:|---:|
| ordinary | 371.085420 | 21.904736 | 2.651626 |
| partial_return_tail | 275.437399 | 16.998103 | 2.019804 |
| closed_P5_full | 272.239831 | 16.649261 | 1.951680 |
| matched_CTS | 387.633441 | 10.391732 | 3.396808 |

保存表の表示値から比を計算すると、closed P5のT切片低減はordinary比約26.6369%、matched CTS比約29.7687%、partial-return比約1.16091%である。CXはmatched CTS比で約60.2164%大きい。

これらはexact原表を再生成した値ではなく、小数6桁・百万単位の表示からの算術である。小さい丸め差を除いて関係を説明するためのもので、新しい閾値判定や再採点ではない。

### 4.2 ここから言えること

closed P5の限定T利益は実在の保存結果である。「Track Bでは一度も資源改善がなかった」とする根拠はない。同時に、matched CTSとの約30%だけを代表値にして、最も近い既存partial-return対照との差を無視するのは不適切である。

TとCXの順位も異なる。どの資源が主評価かを結果後に有利なものへ切り替えない。今回の比較はm=5の同じP5についてであり、m間のexponential精度、実分子、QPE全体、あらゆるCTS実装の最良値の比較ではない。

### 4.3 約1.16%だから研究価値がない、とも判定しない

この1点の切片差は、長時間・多step・別入力の総費用改善率を上から抑える値ではない。例えば、理想化した固定weightの独立R-step samplerでnormalizationがB_i^Rになるなら、同じ統計幅に対する十分shot予算の主要因子はB_i^{2R}となり、局所normalization差が累積する場合がある。

逆に、bias、provider、angle生成、basis変換、共通prep/readoutが支配すれば、局所利益が薄まる場合もある。この式は条件付きの説明で、G10のrejection付き全方式へそのまま代入するものではない。

**従って、1.16%は打切り閾値ではなく、強い対照を選ぶと未解決の主張がどこに残るかを示す観察である。**

## 5. 改善すべき量を固定する

### 5.1 最終taskと資源

PRの代表的なcoherent taskを

\[
f(t)=\operatorname{Tr}(\rho e^{-itH})
\]

とし、複素signalの誤差と失敗確率を固定する。これは説明用に現在の研究targetを記号化したもので、全Track Bの過去実験が同じtaskだったという意味ではない。

axis aに対して実装・アルゴリズムbias上界b_{i,a}、許容幅epsilon_a、統計幅s_{i,a}=epsilon_a-b_{i,a}>0を考える。十分shot数n_{i,a}は、その範囲・二次moment・confidence規則で決まる。期待量子資源は

\[
G_i=\sum_a n_{i,a}\,\mathbb E[C_{i,a}]
\]

である。古典的な係数取得・sampler生成・compileの費用は、これと別の単位で記録する。時間をゲート数へ恣意的に加算しない。

normalizationだけ、local eventのT数だけ、classical実行時間だけを下げても、G_iが下がるとは限らない。[R7–R9, P2]

### 5.2 固定ensembleの確率変更は、既知の重要度サンプリング問題である

説明用の有限ensembleを

\[
A=\sum_j a_jU_j,\qquad a_j\ge0
\]

とする。符号・複素位相をU_jへ含める際はcoherent controlでその位相を実装する必要がある。q_j>0でsampleし、bounded outcome Y_jにweight a_j/q_jを付けると、二次momentの上界は

\[
V(q)=\sum_j\frac{a_j^2}{q_j},
\qquad \bar c(q)=\sum_jq_jc_j.
\]

c_j>0を仮定すると、Cauchy–Schwarzより

\[
\bar c(q)V(q)
\ge\left(\sum_ja_j\sqrt{c_j}\right)^2,
\quad q_j^*\propto\frac{a_j}{\sqrt{c_j}}.
\]

これは既知のresource-optimal ISと対応する。[P2] 固定した実装のmeanを変えずにreweightする限り、同じbiasも保持される。c_j=0の枝、zero-fill、有限range、integer shots、有限bit、bias配分は別に扱う必要があり、この簡約式をそのまま全confidence最適解とは呼ばない。

従って「高価なeventを低確率にする」という一文だけでは次の独立RQにならない。効率的な分布生成、利用できる費用情報、有限confidenceでの具体差が必要である。

### 5.3 相関を導入するだけではmeanは保存されない

回路間で同じbasisを続ければ費用が下がる可能性はある。しかし、各時刻の周辺分布を保つことは、ordered productの平均を保つことではない。

単純な代数例として、U_1,U_2の各周辺分布をI/Z半々とする。独立なら

\[
\mathbb E[U_2U_1]=((I+Z)/2)^2=(I+Z)/2.
\]

完全相関U_2=U_1なら、同じ周辺分布でも

\[
\mathbb E[U_2U_1]=I.
\]

これは新しい回路実験ではなく、Z²=Iからの例である。経路全体へのimportance weightで元meanを戻せる場合はあるが、そのsecond moment・range・正しいsupportを評価しなければならない。Markov random compilerなどの近接研究も存在する。[P3]

### 5.4 weight 1は、誤差が無料という意味ではない

正確率p_jのunitary ensembleの平均をA=Σp_jU_jとすると、直接展開により

\[
I-A^\dagger A
=\sum_jp_j(U_j-A)^\dagger(U_j-A)\succeq0.
\]

特にAが厳密にunitaryなら、p_j>0の全U_jはAと一致する。異なるunitaryの正確率混合で、厳密なunitary first momentをweightなしに実装できるとは限らない。

この恒等式は初等的な分散分解であり、本研究の新しい不可能性定理として主張しない。近似、高次誤差相殺、状態限定task、符号付きweightを排除するものでもない。whole三次補正の追加weight1を、全体biasやshot費用の不在へ読み替えないための判断補助である。

## 6. 新しいRQを具体的に比較する

### 6.1 既存closed P5を物理時間発展へ接続する

候補RQ：

> 同じH・初期状態・coherent signal精度・confidence・native accessの下で、closed P5表現への置換が、ordinary/partial-return RTEと適用可能な強い対照を上回る総資源の利益を持つ条件は何か。

これは既に構成と限定T結果があるため、現時点の発展候補の中では最も証拠に近い。一般fullを再開することとも、Gaussian候補を再開することとも違う。

ただし、Taylor remainder、多step伝播、provider/角度・古典取得費用、同じ最終taskのnとC、独立入力への適用は未完了である。I0アクセスだけを許す対照と、全Pauli表を既に持つI1対照を混ぜない。既知CTSには、Taylor項をPauli収集する構成とclassical costの議論がある。[R9, P4]

この問いの発展は正当であるが、「全体接続が未完了だから必ず次に計算する」という義務ではない。現在の証拠から独立した新しい主要claimや有利な適用条件が既に得られたとは言えず、本レビューでは新規実行計画へ昇格させない。

### 6.2 軌道・basis reuseを考慮する経路sampler

候補RQ：

> 元のordered first momentを保持し、full-wrapper費用とsampling負担を合わせた経路分布を、経路全列挙なしに生成できるか。

これは単なる周辺確率の変更ではなく、joint law、経路weight、有限range、compile費用の非加法性を含む問いである。全trajectory ISそのものは一般ISに含まれ、Markov的な状態変数が小さく十分であることも自明ではない。[P2–P3]

新しい寄与があり得る箇所は、例えばnative費用を保持する小さな十分状態、誤差を制御した分布生成、情報取得費用の具体的節約である。しかし、それらを実現するrecurrenceやoracleが今回提示されているわけではない。費用が既知・加法的だと都合よく仮定して既存DPを呼ぶだけなら、その適用として整理する必要がある。

したがって、具体的な新規実装を立ち上げるほどの構成は本レビューでは採択しない。これは方向の不可能性を示す判断ではない。

### 6.3 near-integrable・multirate・補正の共同設計

候補RQ：

> D/R内部のnativeアクセスを使い、外側PF biasと内側sampling費用の両方を下げる実行列を構成できるか。

この方向はBF/BMとgeneric wholeに関係する。BCH整理、既知PF係数の最適化、補償追加だけでは独立差にならない。LCU補償、corrected PF、HNCC、SPRINTなどが近接する。[R5–R6, P5, P6, P7a, P7b]

一方、既知構成全てが現DFアクセスで最適という証明はない。具体的な実行列、必要oracle、同情報対照、誤差の支配項が変わる理由が示されれば検討価値はある。しかし、本レビューでその追加構成まで定義できていないため、旧列のparameter gridを拡大する新計算を提案しない。

### 6.4 Gaussian圧縮の応用・engineering研究

候補RQ：

> 既知controlled Gaussian圧縮を実DF-RTE builderへ接続したとき、元分布におけるfull-wrapper費用はどの程度変わるか。

正当な応用研究の問いである。しかし、現候補の期待費用は未確定であり、orbital sidecar・runtime対応・finite precision・強いthin対照の実装が必要である。性能を測れば新しい生成手順が自動的に生まれるわけではない。[R1–R3, P8]

前回のclosureと同じ候補を、目的の変更を明示せず再開しない。今回応用研究を主目的として選び直す積極的な根拠は不足すると判断する。応用価値を否定した結論ではない。

### 6.5 一般下界・保証の研究へ転換する

G5や第5節の式には一般化し得る数学的な問題がある。しかし、「全てのPRを排除する」「小さなnormalizationなら安い」といった主張は支持されない。既存のpolicy下界、初等的な分散分解、Cauchy–Schwarzを並べるだけでは、新しい理論的主要貢献とは言えない。

新しい下界研究へ進むなら、対象class、許容するoracle、何の最適性を証明するかを別途定義する必要がある。それは現在のアルゴリズム改善研究とは目的が変わり得るため、今回自動採択しない。

### 6.6 比較のまとめ

| 方向 | 残る可能性 | 現在不足している具体事項 | 今回 |
|---|---|---|---|
| closed P5→同じ物理task | 既存構成と限定positiveがある | 支配的baselineとの差が残る条件、全誤差・実装・取得費用の接続 | 保留する発展RQ。新計画を採択しない |
| 経路相関・native費用sampler | 再利用とweightの両立 | 平均を保つ効率的joint law、非加法費用の情報モデル | 未具体化、実装を開始しない |
| near-integrable／補正共同設計 | 外側biasと内側費用を同時変更 | 既知構成を超える実行列・access・保証 | 旧系列を復活させない |
| Gaussianの実装適用 | 現builderのengineering gap | 実際の条件付き費用、有限精度・orbital接続 | 応用研究へ自動転換しない |
| 一般限界・下界の理論 | G5等からの一般化可能性 | 新しい比較class・非自明な定理・利用目的 | 別目的として未採択 |
| **新規実験主線の一時休止** | 根拠の弱い系列立上げを止める | 再開の具体的理由を待つ | **推奨** |

## 7. なぜ、今回は新候補を一つ選ばないのか

### 7.1 候補が二つ閉じたからではない

異なる候補の終了回数から、分野全体の成功可能性を推定することはできない。二つ連続で非採択だった、metadata監査が多かった、技術停止が繰り返された、という件数を科学的理由にはしない。

### 7.2 未知の性能は、敗北の証拠でも継続義務でもない

Gaussian期待費用、P5の同物理task総費用などは未確定である。情報を取れば結論が変わる可能性はある。しかし、それだけで全ての未知量の取得に優先順位を与えることはできない。

次の実験が何を識別し、そのどちらの結果が研究方針をどう変えるかを具体化する必要がある。現時点では、closed P5以外はその前段の方法・accessに大きな未定義事項があり、closed P5にも応用と主要claimの選択が残る。

### 7.3 新しい研究課題の名称を決めることを成果にしない

前回のGaussian案も、最初から既知構成への対応が疑われており、実際の監査で独立差が残らなかった。候補監査自体は必要な情報を得たが、同程度に未具体化な案をGPTが一つずつ出してCodexへ送り続けることを、標準的な前進としない。

特に、私が新しい候補を提案する場合は、既知の部品名を並べるだけでなく、何を省く・何の平均を変える・どの保証を保つ、という手順上の差を先に示すべきである。この段階を十分に詰めずに新しい実装系列を増やす提案はしない。

### 7.4 今回の判断の確かさ

現在の個別closureを維持すること、性能negativeの範囲を越えないこと、同じtaskで総費用を見る必要性については根拠が明確である。一方、「次の新規投資を今行わない」という優先順位は科学的不可能性の証明ではなく、不確実性を伴う研究判断である。新しい具体的構成・費用証拠が出れば変更できる。

## 8. 休止の意味と、再開条件

### 8.1 止めるもの

- 現候補集合を小変更して新しい実装・実行系列を自動的に作ること。
- 旧STOP解除、消費済みmarkerの再利用、残りのcompile/T/LPを埋めるためだけの計算。
- 具体的な方法差が定義される前の大規模adapter・compiler開発。
- closure文書の公開が終わるたび、同じ全体レビューを再要求すること。

### 8.2 止めないもの

- 保存済みの正しい結果・一般構成・技術ノートの参照。
- 現在の記録に直接関係する重要な誤りの訂正。
- 新しい数式・実行列・アクセス方法についての理論的な着想。
- 利用者が明示的に目的を選び直した上での、将来の研究方針レビュー。
- Track AやN3の、それぞれ独立に承認された作業。

休止期限や研究資源配分率は設定しない。「何日後に再開」と約束するものでもない。

### 8.3 再開するに足る具体的な材料

以下を、完成した勝利証明ではなく、次の実験の意味を説明する最小限の研究構想として整理できればよい。

1. **taskとアクセス**：何の量を、どの精度・confidenceで推定するか。どの入力・controlled primitiveが実際に使えるか。
2. **変更する手順**：式、擬似コード、実行列などで、最も近い既知構成と異なる処理を特定できるか。
3. **改善する量の理由**：bias、second moment/range、回路単価、古典取得費用のどこに変化があるか。対照にも同じ情報・最適化機会を与えるか。
4. **最小判別検証**：どちらの結果なら進め、どちらなら方針を変えるかを、実測前に説明できるか。

これらはpilot前に全性能・全新規性を証明する要求ではない。既知手法の組合せ、定数因子、適用域の拡張も排除しない。また、2番の独立した新手順を狙わず、応用研究の新しい問いを明示的に選ぶ場合は、その目的に適した基準で判断し直す。

5%などの数値閾値、全手法への普遍優位、厳密certificateだけを認めるという新条件は置かない。

## 9. 既存成果の整理・論文化上の位置付け

G6–G10の一般構成・P5特殊化・有限bit・native限界は完成ノートがある。BF/BM/G5/whole/Gaussianの結果やclosureも既に整理されている。今回、新しい巨大な統合報告書や全資料ZIPの作り直しを必須にしない。

論文の主要貢献として現時点で安全に言えるのは、「新しい一般PR算法の優位」ではなく、各資料で明示された構成・限定class・資源trade-off・適用限界である。研究として無価値ではなく、ただし独立論文の成立・採録を今回保証するものでもない。

closed P5を主題に再び論文化を進めるなら、最も近い既存表現との差、入力取得・古典費用、同targetの強い対照、scopeを中心に組み立てるべきである。「30%改善」という一値や「全normalizerを列挙しない」だけを新規性の代用にしない。[R9, P4]

Gaussian応用や一般評価研究へ変更する場合も、既知法の適用を正直に明示し、その適用で得る知見自体を主張する。現在のclosureを遡及的に成功へ書き換えない。

## 10. 直近の実務とGPT/Codexの分担

**現時点で、Codexへ必ず依頼する新しい計算・技術監査はない。** 今回の全体判断はこのMarkdownが記録である。

repositoryに研究状態を反映する必要がある場合、Codexへ一度だけ、原本の内容を保持した保存と既存Track B索引への短い参照追加を依頼すればよい。既存source、契約、raw、marker、STOPを変えず、必要な文書だけを対象とする。

その反映完了を理由に、同じ全体科学レビューをもう一度行う必要はない。追加の大きなmanifest、全1,960ファイルの再監査、旧結果の再取得campaign、再現ZIPを作ることを求めない。

将来、具体的な新着想が出た場合には、技術的な実装可能性・文献対応・source調査はCodexが可能な限りまとめて完結し、重要な新方針・比較基準・主要claimはGPTで判断する。新しい研究仮説に対する開始承認と、科学計算の実行承認は引き続き別である。

## 11. 外部一次資料の確認範囲

| 一次資料 | 今回確認した範囲 | 用途・限界 |
|---|---|---|
| PR original, arXiv:2503.05647v2 | 公式abstract・更新履歴。v2は2026-07-10、PRX Quantum 7,020332の書誌 | 部分乱択と化学資源/QPEの既知範囲。HTML本文は取得不可で、全定理を再確認したとはしない |
| Resource-optimal IS, arXiv:2603.13495v1 | HTML §II, Theorem 1, §II.1、bias保持の記述 | 固定ensembleのcost×moment最適化。有限Bernstein/full-wrapperの万能最適解とはしない |
| Markov random compiler, arXiv:2411.06485v2 | 公式abstractとHTMLの定式化・対象 | 相関乱択が未開拓という主張を避ける。現DF-RTEとの性能比較を実施したものではない |
| CTS/SCU, arXiv:2407.21095v2 | Theorem 1、§IV.2–IV.3、Supplementary Note 5の関連式 | finite/Pauli収集とnormalizationの近接性。全CTS最適化の下界とはしない |
| LCU Trotter compensation, arXiv:2212.04566v2 | 公式abstract、HNCC本文におけるNCC/PSTCの関連整理 | 補償という上位機構の既知性。全本文再認証ではない |
| HNCC, arXiv:2607.11856v2 | HTML §§I–II, main theoremの対象・LCQC sampling | observable/channel・k-local等の条件。coherent signal/DFへ無条件移植しない |
| Corrected PF, arXiv:2409.08265v4 | HTMLの構成・scope概要 | corrector自体の既知性。各定理の独立証明はしない |
| SPRINT/GRADE, arXiv:2606.30741 | 公式abstract。指定HTML v1は404 | near-integrability/randomization等の広い既知設計。現benchmarkとの勝敗には使わない |
| Controlled Gaussian compression, arXiv:2303.09538v2 | HTML §IV.2、Appendix Cと関連説明 | controlled/Hadamard利用の既知性。任意DF compilerの最適性を意味しない |

Pathwise Random Hamiltonian Simulation, arXiv:2608.29756は検索で一次abstractの記録を取得したが、直接HTML/abstractページの本文取得は成功しなかった。従って、その具体的な定理・性能・同値性を今回の採否の根拠に使わない。これ一件の未取得を理由に、本レビュー全体を開始不能とも扱わない。

今回の新規性判断は、関連論文の不存在証明ではない。先行研究の存在は「研究ができない」理由ではなく、候補が何を追加するかを明確にするために参照する。

## 12. 最終判断

**現時点のTrack Bでは、新しい実装・計算候補の連続立上げを一時休止する。** generic whole、Gaussian積生成、固定G5辞書、一般full-returnに関する終了判断は維持する。一方、closed P5・一般生成法・局所分離・限定positiveを失敗へ書き換えない。

本レビューで候補RQを比較したが、今すぐ実験系列を立ち上げる優先候補として、独立した方法差と検証価値を十分に具体化した新手法を採択してはいない。これはPR改善一般の可能性を否定する主張ではない。

既存成果の発展を将来選ぶなら、closed P5を同じ物理時間発展・同じアクセス・同じ最終精度とnative資源へ接続する問いが最も直接的である。しかし、その未完了部分を埋める実行を今回自動的には選ばない。

**今回の終了点は、新しい候補名を付けて次のsource準備を開始することではなく、取得済みの成果と未確定事項を確定し、再開する具体的理由が示されたときに判断し直せる状態を残すことである。**

---

## 付録A：有限meanを同じexponential taskへ戻す際の最小会計

本付録は方針比較のための初等的な解析であり、新しい研究仮説・保証の採択や数値計算ではない。

### A.1 P5のtail

A=P5(-i delta R)、U=exp(-i delta R)、x=delta||R||>=0とする。

\[
\|A-U\|\le r_5(x):=\sum_{k=6}^{\infty}\frac{x^k}{k!}
\le e^x\frac{x^6}{6!}.
\]

後者は(6+j)!>=6!j!から従う。実際のDF解析で何を||R||の有効な上界として使えるか、native errorをどのノルムで課金するかは別に定める必要がある。

### A.2 多stepの伝播

||A-U||<=r、||U||=1なら||A||<=1+rなので

\[
\|A^n-U^n\|
\le r\sum_{j=0}^{n-1}(1+r)^j
=(1+r)^n-1.
\]

これはtelescopingによる保守上界であり、最良の相殺評価ではない。D/Rの外側PFを挟む場合は、そのPF biasと各stepの誤差も別に課金する。

同じP5を異なるsamplerで表すなら理想P5 tailは共通でも、finite-bit・angle・native誤差やsampling rangeは同じとは限らない。「同じ有限mean」の局所比較だけで同じ物理精度の総費用比較を完了したとは言えない。

### A.3 channelの結果をcoherent signalへ持ち込むとき

f(t)=Tr(rho exp(-itH))は、制御付き時間発展とancilla X/Yの期待値として表現できる。例えばHを|1><1|_c⊗Hへ持ち上げれば、原理的にはobservable-estimationの形式に移せる。

しかし、その場合はPauli support、局所性、係数、controlled access、初期状態、ancilla費用が変わる。channelアルゴリズムだから対照から排除するのも、その元論文の費用をそのまま適用するのも不適切である。HNCC等の比較に必要なのは、この共通taskへの変換と費用の明示である。[P6]

## 付録B：判断用スカラー算術と非claim

使用した保存表示値は[R9]のm=5表である。

- P5/ordinary = 272.239831 / 371.085420 ≈ 0.7336311704。
- P5/partial return = 272.239831 / 275.437399 ≈ 0.9883909447。
- P5/matched CTS = 272.239831 / 387.633441 ≈ 0.7023125515。
- CXのP5/CTS = 16.649261 / 10.391732 ≈ 1.6021642013。
- KのP5/partial return = 1.951680 / 2.019804 ≈ 0.9662719749。

これらは表示値からの比であり、新しい実行・最適化・再classificationではない。共通prep/readout単価を任意に選んで全費用の勝者を新たに固定していない。約1.16%を今後の全入力の上界、約29.77%をPR一般の改善率、約60.22%を全hardwareでの不利益とはしない。

## 参考資料

### Repository・利用者資料

- [R1] [Gaussian積生成候補closure](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64bb39fdcc4928d0f026b63f874c8bf4be2e69ab/docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md)。今回直接取得。レビュー原本・監査・欠測への索引。
- [R2] [Gaussian積生成の限定監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6530c055c505712d5f1d1af604f1accf85cebba/docs/tracks/algorithm_codesign/df_rte_product_generation_audit_v1_20261013.md)。前回レビューで取得された本文と今回のclosureの対応を継承。
- [R3] [監査後科学レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64bb39fdcc4928d0f026b63f874c8bf4be2e69ab/docs/research/track_b_gaussian_product_generation_scope_review_20261013.md)。提供済み原本・既存判断を継承。
- [R4] [DF三次補正E/T closure](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4192077723770f4846cc391ea0da96c1e115f2fa/docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md)。対応する提供済み科学レビューと会計表の限定を継承。
- [R5] [BF現仮説closure](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d55de044b8e956ba6292209a94bb081014dfdae2/docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md)。今回直接取得。
- [R6] [BM-0.5同値性監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d55de044b8e956ba6292209a94bb081014dfdae2/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md)。今回直接取得。
- [R7] [SP-1 one-shot結果・保存値監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9d2bb1fa439748b02084bd9fbc9b10a705328f8a/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)。今回直接取得。
- [R8] [G5固定辞書終了認証](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23adf61f9342abff20f059be69089e89f66f7797/docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md)。今回直接取得。
- [R9] [Return集約・P5・native資源の完成ノートv1.0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md)。今回§3–7の関連本文・表を直接取得。全raw再監査ではない。
- [R10] 利用者添付 `貼り付けたテキスト（1）.txt`：Codex横断調査。本文213行相当の内容を提供済み。調査範囲133 branch、16資料のGit blob一致はCodexの報告として扱う。
- [R11] [Track A H6科学レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/docs/research/track_a_h6_independent_confirmation_scientific_review_20261013.md)。前段で取得済みの独立した方針。今回Track Aの新実行判断をしない。

### 外部一次文献

- [P1] Günther et al., *Phase estimation with partially randomized time evolution*, [arXiv:2503.05647v2](https://arxiv.org/abs/2503.05647v2), PRX Quantum 7, 020332 (2026)。今回公式書誌・abstractを確認。
- [P2] Cugini, Atif, Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, [arXiv:2603.13495v1](https://arxiv.org/html/2603.13495v1)。特に§II, Theorem 1。
- [P3] Dubus, Cunningham, Roland, *New random compiler for Hamiltonians via Markov Chains*, [arXiv:2411.06485v2](https://arxiv.org/html/2411.06485v2)。
- [P4] Peetz, Smart, Narang, *Quantum Simulation via Stochastic Combination of Unitaries*, [arXiv:2407.21095v2](https://arxiv.org/html/2407.21095v2)。Theorem 1、§IV.2–IV.3、Supplementary Note 5。
- [P5] Zeng et al., *Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations*, [arXiv:2212.04566v2](https://arxiv.org/abs/2212.04566v2), PRX Quantum 6,010359 (2025)。今回公式abstractを確認。
- [P6] Wang et al., *Trotter error compensation with polylogarithmic precision and nested-commutator scaling without ancillas*, [arXiv:2607.11856v2](https://arxiv.org/html/2607.11856v2)。2026-07-23版。
- [P7a] Bagherimehrab et al., *Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas*, [arXiv:2409.08265v4](https://arxiv.org/html/2409.08265v4)。
- [P7b] Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, [arXiv:2606.30741](https://arxiv.org/abs/2606.30741)。今回abstractの範囲。
- [P8] Kökcü et al., *Algebraic Compression of Free Fermionic Quantum Circuits: Particle Creation, Arbitrary Lattices and Controlled Evolution*, [arXiv:2303.09538v2](https://arxiv.org/html/2303.09538v2)。§IV.2, Appendix C。

---

**記録上の最終状態：Track Bの現候補集合に基づく新規実験主線を一時休止。既存成果と不確実性は保持。次の自動実行なし。新しい具体的構成・研究目的・証拠が出た場合のみ、その内容を根拠に再検討する。**
