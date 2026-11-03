# Track B全体の研究方針レビュー
## 既存系列の到達点、未完了表示の訂正、次の研究課題と継続条件

- 作成日：2026-10-11 JST
- 開始承認：利用者の「Track B全体の研究方針レビューをできるだけ丁寧に行って」
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 主たる固定参照：`22a0d032717b4139cdfa806850643e93ba25624f`
- Branch：`track-b-return-aggregation-technical-note-20261011`
- レビュー区分：研究系列横断の科学的・戦略的レビュー。G10の全event再認証、研究データの新規取得、投稿実行ではない。
- 結論：**既存候補をそのまま再開することは推奨しない。Track Bの目標は保持し、物理的な時間発展・複素信号に対するbiasとsampling負担を同時に変えるPR内部構成へ、限定的な理論再設計を優先する。ただし具体的な新手法の新規性・優位は未確立であり、本番実験へのGOは出さない。差のある構成を特定できなければ、現成果を保持してTrack Bの新規実験を休止する。**

---

## 0. 最終判断を先に示す

本レビューは、次の三つを区別する。

1. **これまでの研究を残す価値**：ある。一般構成、P5実装、限定クラスの最適性・分離、native資源の条件付き改善、改善が消える条件が残っている。
2. **過去の未完了表示を理由に実験を再開する価値**：認めない。R1／RA-D0はG1–G5へ、SP／BSは後続の表現研究へつながっている。準備文書の古い状態を現在の未消化課題と誤認してはいけない。
3. **今後もPR内部アルゴリズム改善を目指す価値**：否定しない。ただし新しい研究課題と、強い既知法との差が必要である。「まだ全条件を試していない」こと自体は継続理由にならない。

推奨する次の問いは以下である。

> **与えられたD/R分割と利用可能なnative操作を使い、PRの一連の時間発展が残す誤差を、実装可能でsampling負担の小さい補正・再構成へ変換できるか。その結果、同じ複素信号精度・失敗確率の下で、既存PRと既知誤差補償法を上回る資源上の利益が生じるか。**

これは、補正法の新規性が確認されたという意味ではない。NCC/PSTC/HNCC、THRIFT、確率的合成など、既知法の直接適用へ帰着する可能性は高い。したがって、現時点で採るのは**研究課題としての優先順位**であり、特定の新アルゴリズムの採択や大規模pilotではない。

既存P5のPR全体への適用評価も候補には残るが、これはまず応用・資源評価研究として位置付ける。独立新規アルゴリズム研究と無言で入れ替えない。

## 1. 今回の資料と確認範囲

### 1.1 正本と確認したこと

固定branchのHEADが`22a0d032717b4139cdfa806850643e93ba25624f`であることをGitHub接続から確認した。研究概要・現状、2026-10-09の総合再評価、各系列の結果・終了判断、G5 claim/evidence表、G10技術ノートv1.0、代表的な保存監査を参照した。[R1–R15]

重要な研究方針判断には、原結果の全数値を毎回再計算することと、既存の科学的結論・比較クラス・実行状態を正しく読むことを区別する必要がある。今回は後者を中心とし、全研究を同じ実験として再集計していない。

| 資料群 | 今回の利用 | 独立に実施していないこと |
|---|---|---|
| B-F | 固定仮説、recovery audit、closureの照合 | 原1269 cellの全replay、分子計算 |
| B-M | 同値性の再帰式、比較条件、既知compact法の確認 | 全symbolic fixtureの再実行、DF backend評価 |
| SP／BS | 登録wrapper結果、target・scope・既存法同一性の確認 | primitive合成、全channel行列、全shot計算 |
| R0–G5 | 履歴、RA-D0実行停止、6頂点・固定クラス分離の確認 | 全LP・全digital lawの再認証 |
| G6–G10 | 完成ノート、固定native結果、以前の採用判断 | 全10,936 event、全136分割、57,021監査項目の再実行 |
| P-D／R3／FR | Track Bに継承された前段の停止・限定成果の確認 | 過去の全理論証明・全fixtureの再監査 |
| 外部文献 | 一次本文・定理・必要access・比較対象の照合 | 全論文・全版を網羅した優先性証明 |

科学実行、新しい物理入力、synthesis、回路compile、Hamiltonian行列、量子shots、LP再求解、remoteの書込みは行っていない。既存の結果分類・source・contract・authorization・marker・STOPは変更しない。

### 1.2 資料の欠落と取得制約

本レビューの研究方針判断に不可欠な主要報告・履歴・比較クラスはGitHubから取得できた。全原結果を独立再認証できたという意味ではない。既存BFのbridge欠測や、古い大きなJSONの取得形式上の問題を、新たな未pushとして扱わない。

外部文献のうちpathwise correlated simulationの2026年preprintは検索で抄録を確認できたが、本文をこの環境で取得できなかった。性能や定理の妥当性の根拠には使用せず、類似の研究方向が既に提示されているという注意にとどめる。[P11]

G10の以前のレビューに添付された検算ZIPのCodex側未受領は、今回の全体方針判断を止める理由ではない。今回そのZIPを再実行したとも、Codexが独立対数certificateを再現したとも表示しない。

## 2. 先に訂正すべき、直前の説明の問題

### 2.1 RA-D0を「source準備済み・本番未実行」としたのは誤り

RA-D0 v3は一回実行され、最初のB2 T minimumで実装certificateが得られず、`D0_TECHNICAL_INCONCLUSIVE`で停止した。比較可能なB2/B3 queryは0件である。これは数学的な非改善結果でも、実行待ちの状態でもない。[R4]

その後、技術調査と研究再評価を経て、G1の構造還元、G2の保存値診断、G3の有限law、G4の強い対照、G5の固定辞書終了認証へ進んでいる。古いv3準備文書を現在の状態として引用した前の回答は訂正する。[R2,R3,R5,R6]

ただし、G5が旧RA-D0の全数値許容幅classを証明したという意味ではない。旧D0はtechnical incompleteのまま、G5は別に明示されたclassの結果である。

### 2.2 R1、SP、BSを独立した未消化候補の一覧にしたのは不適切

R1の結果は、後続のdegree-local表現研究へ進む際の入力になっている。SP-1後にも、block合成・BS-0.5・RTE再配分へ研究上の検討が進んでいる。各時点の「GPTへ戻す」は、その後の履歴で扱われた場合がある。[R2,R3,R8,R9]

従って、

`R1未判断 + SP未判断 + RA-D0未実行 + G10完了`

という横並びの理解ではなく、概略として

`B-F → B-M → SP → BS → R0/R1 → RA-D0/技術調査 → G1–G5 → G6–G10`

という研究上の移行を読むべきである。全てが同じtarget・同じ独立実験という意味ではなく、研究課題を変更しながら移った履歴である。

### 2.3 この訂正が今後の判断に与える影響

「未実行の計画が多数あるからTrack Bには継続候補が多い」という根拠は弱くなる。今後の候補は、古いタスクリストから選ぶのではなく、現在の強い対照と停止理由を踏まえて、別の科学的利益を説明できるものに限って検討する。

これは過去の研究成果を減らす訂正ではなく、再開理由を誤らないための訂正である。

## 3. Track Bの目的と、今回勝手に変更しない範囲

大きな目的は、**PR内部の時間発展・乱択実装を改良し、同じ物理的精度・成功条件で量子資源を削減すること**である。2026-10-09の全体再評価も、Taylor、single block、canonical samplingを普遍的制約ではなく、その時点の研究範囲としていた。[R2]

一方、次を混ぜない。

- Hamiltonianの軌道回転、DF再分解、対称性による外部前処理を主題とする別研究。
- Track Aの実装・並列化・転移検証の成果を、Track Bの新規アルゴリズム成果として数えること。
- 既に別研究で扱っているPF誤差予測・エネルギー誤差校正そのもの。
- 固定finite polynomialの比較と、同じ指数関数精度・同じエネルギー精度の比較。

既存DF表現やnative操作を**入力として使う**ことは、DF前処理研究をTrack Bへ移すこととは違う。新しいnative samplerや時間発展構成をその入力上で設計するなら、Track Bの範囲にある。

「新規アルゴリズムを確立する研究」と「既存構成を実問題で比較する応用研究」はどちらも研究になり得る。ただし、今回の優先順位では前者の狙いを重く置き、後者へ無言で切り替えない。

## 4. 系列別の科学的評価と今回の扱い

### 4.1 P-D／R3：詳細化した評価が、同じ情報の対照を超える判断を生まなかった

P-Dの公平な比較では、内部workを数える対照、leading-tail model、finite-RTE modelが固定候補集合で同じ構成を選んだ。R3でも、一般multi-fidelity／best-arm手法を超えるquantum-specificな認証則を固定できず、停止している。[R1]

**残るもの**は、弱いcost対照や情報層の不公平さが見かけの改善を作ること、評価精度の向上と資源選択の向上を区別した記録である。

**今回の判断**は現路線を再開しないこと。単に目的関数を精緻化したり、汎用optimizerへ既知特徴を増やしたりすることを、次の主題にしない。ただし、別の量子固有の構成や利用可能情報を用いて新しい判断を作れるなら、それは別課題である。

### 4.2 B-F：固定familyにおける詳細finite目的の追加価値は不支持

固定対象はH4 linear 1.00 Å、STO-3G、DF rank12、prefix L_D=3、T=0.8、5-stage symmetric fourth-order familyである。F/L各32評価の共通finite採点最良は同じSuzuki5で、action proxyは20,709,936.72。共通参照集合のS2は12,924,223.16で、F／共通参照は約1.6024だった。[R7]

元runの保存失敗によるINCONCLUSIVEと、その後の保存値recoveryによるBF-Aは別層のまま保持する。[R7a]

**残るもの**は、固定family内で詳細モデルを最適化することが、それ自体で有用な新構成を作るわけではないという限定結果である。

**今回の判断**はclosure維持。bridge欠測を埋める、同じ方法でstage数やgeometryを順番に増やす、といった救済的継続はしない。全高次PFや全係数設計の不可能性へ一般化もしない。

### 4.3 B-M：違う書き方を、違うアルゴリズムに数えない

固定nested列について、三次BCH係数はcompact再帰と

\[
K_m=K_{\rm floor}+\frac{K_A}{4m^2}
\]

の分解で一致する。同じbackend、aggregation、cache機会を与えればscoreと候補順位も同じである。[R10]

Maxwellらの一次本文にも、再帰的PFとcache、compactな三次・五次commutator表現がある。[P2]

**残るもの**は正しい式とreuse構造であり、無価値ではない。しかし現adapter専有の科学的差分ではない。

**今回の判断**はBM-1の旧new-method routeを再開しないこと。一般nested／multirateを全て閉じる意味ではないが、新しい実行列、取得法、保証などを示さず名前だけ変えることは認めない。

### 4.4 SP／BS：大きな局所改善はあるが、未開拓の新手法が残っているとは限らない

SP-1はsynthetic coherent wrapperで、短いDR配置に大きな利益が残り、長い列でloss／shot capに変わった。例えばC,n=16ではDRの一回当たりT費用は約7.308、NONEは2160だが、二次momentは約30.926対1、十分shotsは410,115対13,523となる。両方を掛けたDR比は約0.1026である。[R8]

これは実際の保存結果として重要なpositiveであり、「何も改善しなかった」とは言えない。しかしactual finite-RTEの外側分布ではなく、additive primitive価格のtoyである。D/Rのgate数・角度・符号・coinも交絡している。期待費用の低下を一般的なselective-placement則へ広げられない。

BS-0.5では、候補が同じ入力・sparse処理・phase-aware辞書・residual制約を持つgeneric operator LCUと独立差を持たないと整理された。system channelとcontrolled first momentを混同しないtarget整理は成果だが、現candidateを新アルゴリズムとして実験する理由にはならない。[R9]

**今回の判断**は旧SP／BSを新規手法の主線として再開しないこと。特定装置・native仕様の応用研究として選び直す可能性は残るが、別の必要性・比較が必要である。四maskの続きを埋めること自体を目的にしない。

### 4.5 R0／R1／RA-D0／G1–G5：局所差から強い対照まで進んだ、一つの到達済み系列

この系列では少なくとも三種類の成果が残る。[R5,R6]

- R0の、明示class内の有限mean保存とnormalization最適構成。
- G1の固定P3・7prototype理想classを6頂点へ還元した構造。
- G4-Aの、指定J1有限lawと限定digital B2下界との条件付き分離。

その一方、G5では既知development x=1/4、固定dictionary／precision／IID group shares／L1-charge／Bernstein policyのclassについて、同じ保存CTS lawがT・CX・1Qの保守的下界を下回った。Tの例はclass下界171,032,500に対しCTS約164,669,575.77である。[R5]

**残るもの**は限定理論、有限law構成、強い対照まで戻した分離結果である。G5のCTS優越はR0／G1の恒等式やG4-Aの比較を反証しない。

**今回の判断**は固定toy・同辞書での実用優位探索の終了を維持すること。汎用exact LP基盤を先に完成させる方向へ戻らない。旧D0の未到達query数が多いことを継続理由にしない。

### 4.6 G6–G10：構成は残るが、一般fullを資源改善の主役へ戻さない

一般の全形式return集約、局所query、finite-bit補正、P5の少数群構成は技術ノートv1.0として完成した。

固定m=7でfull returnはP5＋tailよりT切片約2.91155%、K約0.61484%大きい。固定full辞書・合成列・同じpolicyでの任意proposal下界も、共通準備T単価0≤h≤970でP5＋tailの登録費用を上回る。[R11]

この結果は、nominal normalizationやsupportの減少をnative資源改善と取り違えないための具体例である。P5系のliteral CTSへのT利益は維持するが、一般fullの追加利益ではない。

**今回の判断**は現系列を完成済み成果として保存すること。v1.0をもう一度「完成させる」作業、同じ論文化review、m9や新seedによる救済探索は追加しない。

### 4.7 FR：理論の未完了と、有望な主法であることを分ける

前段のFR-R1bでは同情報でFR境界がnorm境界より厳しいwitnessが8件あった一方、固定予算の片側認証差は0で、`MECHANISM_ONLY_NO_PRACTICAL_GO`だった。理論主張の監査は未完だが、数値拡張を止める判断は残っている。[R1]

**残るもの**はphase/radiusの分離、正scalar、情報層、signal floorの必要性などの理論材料である。

**今回の判断**は休止維持。理論を扱うなら、既存の強い同情報境界を超える命題を具体的に切り出す必要がある。真のoverlapや真のsignalを無料の情報として使う研究へ戻らない。未完了の証明義務を埋めることだけを理由に、この系列を優先しない。

## 5. Track B全体で本当に分かったこと

### 5.1 「数学的に良い」から「研究上有用」までに、別々の壁があった

各系列は同じ失敗を繰り返しただけではない。

| 層 | 代表的な問題 | 教訓 |
|---|---|---|
| 評価・選択 | 詳細なFと強いLが同じ選択 | よい誤差評価と、新しい資源判断は別 |
| 数式・構成 | BMとcompactが同じ | 再表記・cache整理を新手法としない |
| 表現・sampling | B²改善がnative価格で相殺 | 平均保存・norm・sampling・compileを分ける |
| 比較対象 | J1の差が強いCTSで消える | 弱い対照への改善だけでmain claimにしない |
| 認証 | boundが厳しくても予算が変わらない | 保証の強さとdecision relevanceは別 |
| 研究進行 | 古いpendingを新候補と誤認 | 履歴・実施状態・後続判断を照合する |

これらの教訓は有効だが、その一般論だけで独立新規論文が成立するとまでは判断しない。既知の研究原則と、今回の具体的な構成・反例・class分離を区別する。

### 5.2 Positiveを残し、失敗を過大に一般化しない

P5系の固定条件でのT改善、SPの短いwrapperでの利益、R0／G1の構造、G4-Aの限定分離は残る。それらを総じて「Track B失敗」と呼ぶのは不正確である。

同時に、その集合から「PR内部の新アルゴリズムを確立した」と結論することもできない。異なるtarget、native model、confidence policyのpositiveを一つの仮想手法へ合成しない。

## 6. 資源評価をつなぐ式：何を変えれば、本当の改善になるか

以下は既存会計を整理するための本レビューの説明であり、新しい独立定理や新実験ではない。[R8,R11,P3]

一つのcoherent-axis estimatorをXとし、物理的targetの誤差許容量をε、bias上界をb、残余幅をs=ε−b>0、二次moment上界をν、range上界をWとする。既存の型の十分予算は

\[
N=\left\lceil\log(2/\alpha)\left(\frac{2\nu}{s^2}+\frac{4W}{3s}\right)\right\rceil.
\]

複数のinterrogation／axis jについて、各資源aの期待量は

\[
G_a=\sum_jN_j\,\mathbb E[C_{j,a}],
\]

となる。zero-fillの事前棄却は量子価格0として入れ、準備・readoutは実際に量子実行するeventへ課金する。古典前処理・query・compile・cache・postprocessingは別軸で記録する。CPU秒とT数を無断で加算しない。

改善の経路は少なくとも、biasを減らす、二次moment／rangeを減らす、一回価格を減らす、情報取得・古典計算を減らす、の四つである。いずれか一つだけを改善しても、他の増加で総費用が悪化し得る。

### 6.1 分布を変えるだけでは、固定された実装のbiasは変わらない

固定された実装V̂_eと元係数a_eについて、q_eで抽出してa_e/q_eを掛ければ、平均は常にΣ_e a_eV̂_eである。従ってproposalだけを変えても、その固定平均と理想targetの差は変わらない。これは既知ISのbias不変性とも整合する。[P3]

したがって、次にbiasを下げたいなら、表現・実装・補正・近似対象のいずれかを変える必要がある。単なる確率最適化に、それ以上の効果を期待しない。

### 6.2 Local改善を、全時間の改善と同一視しない

独立なr個の同じstepの補正重みがBなら、積重みはB^r、二次moment側の代表的負担はB^{2r}になる。局所の差と全時間の差は、step size・必要r・biasを戻して評価する必要がある。

R0の非負adjacent-degree構成に限った小step展開

\[
B_{\rm pair}(x)-B_A(x)=x^4/9+O(x^6)
\]

を使い、固定総時間でx=Λt/rとすると、log重みの差のleading sizeはrO(x^4)=O((Λt)^4/r^3)となる。これはそのlocal normalization効果が細分化で弱まり得るという説明であり、全return集約、全native利益、全PR手法の上界ではない。[R2]

### 6.3 より小さいoperator normと、より小さい実装LCU normは別

補正演算子Eが小さくても、利用可能なunitary辞書でE=Σ_e β_eV_eと書いたとき、Σ|β_e|が同じ程度に小さいとは限らない。operator normでの相殺を、実装できる係数集合へ移す方法が必要である。

これは今後の候補を見極める最重要点である。小さいcommutatorを評価するだけならBM型の誤差評価研究に戻り、実装係数の負担まで小さくできなければ、randomized補正の量子資源利益はまだない。

## 7. 今後の研究を囲む一次文献

以下は関連箇所の確認であり、論文全体・全版の優先性調査ではない。文献の結果と、本研究へ適用できるという判断を混ぜない。

| 文献・参照版 | 確認した重要点 | Track Bへの含意 |
|---|---|---|
| PR原論文、arXiv:2503.05647v2 | deterministic側biasとrandom側sampling負担を組み合わせる | 物理targetへ戻した比較が元の目的に合う |
| Maxwellら、2606.30738v1 | compact BCH・再帰・cache | BMの既知対照を弱くしない |
| Cuginiら、2603.13495v1 | cost–moment最適ISと固定実装bias不変性 | probability変更だけを新しい補償法にしない |
| Peetzら、2407.21095v2 | CTSとlayer sampling | 全展開しかできないCTSを仮想対照にしない |
| Zengら、PRX Quantum 6,010359 | Trotter remainderをLCUで補償する既存法 | 「PFの後ろへ乱択補正」は既知の出発点 |
| Wangら、2607.11856v2 | channel-level高次nested-commutator補償 | commutator保持・高次補償も未開拓の標語ではない |
| TE-PAI、2410.16850v2 | channel平均の時間発展、離散角、classical／measurement trade-off | 合成とsamplingの交換そのものは既知 |
| Wadaら、2512.06260v2 | coherent／randomized LCUのgroupedな交換 | block/group化だけを新規性にしない |
| THRIFT、Nature Communications 2025 | dominant部と部分perturbationの実装条件を使う | Dが大きいから無料interaction picture、とはしない |
| Randomized MPF、Quantum 6,806 | 複数PFの乱択結合 | step size混合・extrapolationを白紙の方向としない |

HNCCの本文はk-local Pauli／local-strength条件でobservableのchannel推定を扱う。これを分子DFやPRのcontrolled信号に、そのままの価格・qubit数・scalingで転用しない。[P6]

THRIFTも、dominantなH0と必要な部分Hamiltonianの時間発展を効率よく実装できる条件が重要である。Dを多数の非可換DF fragmentとして持つというだけでは、その条件は閉じていない。[P9]

Pathwise correlated simulationの2026年preprintは抄録のみ確認した。相関した乱択を「まだ誰も試していない」とは扱わないが、本文未取得のためその性能や優先性の確定には用いない。[P11]

### 7.1 Channel法を、単純に比較対象外にしない

system上のchannel平均の一致は、同じsystem unitaryのfirst operator meanの一致ではない。これは従来の区別として維持する。

ただし、以下は別の正しい経路である。補助qubit aを用い

\[
H_c=|1\rangle\!\langle1|_a\otimes H
\]

を時間発展させると、e^{-itH_c}=|0><0|⊗I+|1><1|⊗e^{-itH}である。初期|+><+|⊗ρに対し、標準のPauli規約では⟨X_a⟩=Re Tr(ρe^{-itH})、⟨Y_a⟩=Im Tr(ρe^{-itH})となる。

従って、適切に拡大系へ適用したchannel法は、複素信号taskの比較に使える可能性がある。これは本レビューの基本的なoperator上の説明であって、新しいアルゴリズムの発明ではない。

その場合H_c=(I−Z_a)⊗H/2を扱い、追加control、Pauli support、補助qubit、sampling負担を課金する。system上の「ancilla-free」を、phase-estimation全体でも補助qubit0と宣伝してはいけない。TE-PAIも相関関数・Loschmidt overlapへの接続を論じている。[P7]

## 8. 何を終え、何を未解決として残すか

| 内容 | 状態 |
|---|---|
| 既存の固定B-F仮説、BM adapter、同toy degree-local優位探索 | 再開しない |
| G6–G10の現full-return性能探索と文書化 | 完了として保持 |
| R0／G1／G4-A、P5、SPの限定成果 | 消さない。範囲付きで再利用できる |
| FRの理論材料 | 休止。未完了を理由に優先しない |
| 任意input／任意representationでのPR改善可能性 | 否定されていない |
| Track Bの次の具体的新アルゴリズム | 本レビュー時点で未確立 |
| 次の研究課題の優先順位 | 以下の限定再設計へ置く |
| 新しい本番実験 | 認可しない |

「否定されていない」は「有望と実証された」と違う。継続するなら、その未解決性から具体的な問いを選び、どの証拠で答えを得るかを説明する必要がある。

## 9. 代替方向の比較

ここからは資料の単純要約ではなく、本レビューの研究上の評価・提案である。

| 方向 | 期待できる内容 | 問題 | 今回の優先順位 |
|---|---|---|---|
| 同辞書のLP／IS／precisionをさらに精緻化 | 固定policy内の改善 | G5/G10下界、既知IS、残余scopeだけを追う危険 | 採らない |
| general fullのm9・新p・新seed探索 | 勝つ点が見つかる可能性 | 既存停止に答える機構が先にない | 採らない |
| FRのphase-specific認証 | boundからdecisionへの差 | signal floor・情報取得・同情報強対照の未解決 | 休止 |
| SP／BSの応用評価 | 特定nativeモデルの資源設計 | 既知構成の適用であり、現在は新method差なし | 主線にしない |
| P5等を実PR全体へ接続 | 実用的な適用域・資源比較 | 新規性は別、全時間biasと価格を戻す必要 | 次点の応用研究候補 |
| PRのD/R構造を使う補正・再構成 | finite-P_m最適化では触れないbias／weightの共同改善 | NCC等との差と実装accessが未確立 | **限定的な理論再設計の優先課題** |
| 全時間の相関sampling／MPF等 | step間構造の利用 | 既知法が多く、mean・全trajectory費用の保証が必要 | 主候補が具体化しない時の自動退避先にしない |
| 現成果を残してTrack B新規実験を休止 | 研究資源を他の確立した課題へ振り向ける | 新method目標は未達のまま | 差のある構成を得られない場合の正式な帰結 |

「補正・再構成」を優先する理由は、最も成功しそうだという実証があるからではない。PRの本来のbiasとsamplingの結び付きに直接関わり、旧固定mean最適化が触れなかった自由度を、既知法との差として明確に問えるからである。探索価値の判断であり、成功確率や資源改善量の予測ではない。

## 10. 推奨する研究課題：誤差を評価するだけでなく、低負担で実装する

### 10.1 固定finite polynomialを、全研究の目的にしない

旧taskは

\[
\mathbb E[wV]=P_m(-ixR)
\]

を保った上で表現を変えるものだった。これは公平で意味のある比較だが、m、step数、補正構造を含む物理的な設計全体ではない。

次の候補を考えるときは、初めから

\[
z(t)=\operatorname{Tr}[\rho e^{-itH}],\qquad H=D+R
\]

の複素信号精度を主たる中間taskとして定めることを推奨する。各方法が別の有限多項式・補正を使ってよい一方、truncation・deterministic bias・digital/native bias・statistical errorを共通targetへ戻す。

これは過去G10の比較条件変更や再採点ではなく、新しい研究課題である。また単一時刻のε_zを、直ちにエネルギー精度ε_EやRPE全体の成功率へ読み替えない。エネルギー主張へ進む場合は状態重なり、信号、時刻列、phase reconstructionまで戻す。

### 10.2 構成の最初の分岐

利用可能なunitary kernel S_δに対しては、形式的なmultiplicative residual

\[
F_\delta=e^{-i\delta H}S_\delta^\dagger
\]

を考えられる。しかしこの恒等式自体は既知のTrotter compensationの出発点であり、新規性ではない。[P5]

実際のPRの平均M_δが非unitaryなら、M_δ^†をM_δ^{-1}と扱ってはいけない。その場合はまず

\[
E_\delta=e^{-i\delta H}-M_\delta
\]

というadditiveな差か、別の正当に実装できる分解を定義する必要がある。

これらは**研究対象を定義する式**であって、未知のexact Uを実装できたという算法ではない。実際に必要なのは

\[
E_\delta\approx\sum_e\beta_e V_e
\]

の係数取得、native V_e、signed weight、有限精度のbias、全時間のsampling上界を具体化することである。

### 10.3 狙う差は、D/Rというラベルではなく実装上の構造

研究候補として意味があるのは、例えばD/Rの関係から得られる相殺や小さい混合誤差を、次のいずれかへ変換できる場合である。

- 既知の強い補正実装より小さい係数質量・二次moment。
- 同じ保証をより安価なnative eventで実現。
- 同じ入力から補正係数・eventを取得する古典処理を具体的に削減。
- 既存法では実装条件が閉じないinput classに、費用付きの実装を与える。

全てを同時に達成する必要はない。漸近orderの改善だけを新規性の必要条件にもしない。定数因子や適用範囲の明確な改善も研究内容になり得る。

ただし「D/Rへ分けた」「commutator normを使った」「Pauli展開を使わなかった」だけでは差を示したことにならない。generic NCC側にも同じgrouping、native oracle、情報、cache機会を与える。

逆に、一般LCUやNCCの枠組みに含まれることだけを理由に、新規性を否定してはいけない。比較対照へ同じ入力情報と既知の最適化機会を与えることと、候補で初めて開発した生成・取得手順を対照へ無償で組み込むことは違う。無限に最適化された仮想compilerや未発明の最適baselineを要求しない。BS/BMの停止理由は一般classへの包含だけではなく、当時の具体的な処理が同じだった点にある。

### 10.4 事前に分かる注意点

二成分の対称splittingを理想的に考えた場合、leading mixed defectにはδ³[D,[D,R]]、δ³[R,[D,R]]型が現れる。一方、P3によるR側finite-tail errorはδ⁴R⁴から始まる。したがって、**二つの誤差を同じ表へ入れただけでleading termが相殺するわけではない**。異なる次数の項を都合よく打ち消したと説明しない。

またRが小さくても、D内部の非可換splittingによるpure-D誤差は残り得る。D全体をexactに実装するという仮定を無断で足さない。理想二成分の説明と、既存DF/PFの実装を分けて会計する。

THRIFTやcorrected PFが同じ小さいperturbationを利用する場合もあるため、弱い一律norm対照だけに勝つ構成を新methodとはしない。[P9,P12]

## 11. 最初の一作業で、何を明らかにするか

推奨するのは、巨大な原稿や新しい汎用solver基盤ではなく、**一つの候補に対する、構成・同値性・access・資源成立条件をまとめた理論／技術packet**である。今回のレビューをまた一度やり直すことではない。

### 11.1 必要な内容

| 問い | 完成物 |
|---|---|
| 何を出力する方法か | operator/channel/複素signalのいずれかを明示した平均と誤差式 |
| 既知法と何が違うか | 同じD/R、同じinput、同じprimitiveでの写像と差分 |
| どう生成するか | β_eとproposal、signed weight、termination、local取得手順 |
| 安価だと言う根拠は何か | bias、二次moment、range、native費用、classical取得の式 |
| どの条件を判別すればよいか | 最小のcontrolと、仮説を反証できる比較 |

具体的なβ_e／event取得が書けず、未知U、真のground state、未取得の全Pauli表、無料exact D、または無制限事後compileを使う場合は、実装可能性未確立とする。

### 11.2 研究判断の分岐

**継続できる場合**：既知法の直接適用とは異なる構成・取得法・保証の差があり、その差が資源へ作用する機構と、比較可能な最小検証が定義できる。一般定理だけでなく、強い対照で判別できる具体的な構造差でもよい。

**応用研究へ位置付け直す場合**：手法そのものは既知だが、指定されたPR/native inputで未整理の有用な適用域を調べる明確な理由がある。利用者の研究目的を無言で置換せず、method noveltyとは分ける。

**休止する場合**：generic NCC/LCUと同じ処理へ帰着する、作用素の小ささをsampling負担へ移せない、利用可能な情報で予算を改善できない、または追加検証の目的が勝つ点の探索だけになる。

後者なら、名前を変えて別のpilotへ自動移行しない。Track Bの新規実験を休止し、既存技術ノート・構造結果・負例を保存して区切る。

ここに結果後の5%／10%の新しい基準は置かない。量的なmaterialityが必要な次実験では、研究上の意味と測定精度から結果前に決める。旧experimentの基準は変更しない。

## 12. 次の担当と、今回の承認範囲

**研究方針の全体レビューは今回完了した。** 次の課題は、同じレビューの反復ではなく、上記の限定課題に対する具体的構成である。

GPTが担当するのは、研究上の差を持つ候補の式・比較・狙いの設定である。Codexには、具体的な案が渡された後、その平均式、既知sourceとの同値性、native access、算術・実装量、保存条件を一つの作業単位として技術確認してもらう。未定義の新methodや新規性の採否をCodexに丸投げしない。

現時点では、旧RA-D0、G10、SP、BM runnerを呼び出す次作業はない。新しい本番計算を始める段階にもない。原source、contract、authorization、STOP、markerはそのまま保持する。

利用者がこの新しい研究課題を採用した場合でも、それ自体で旧one-shot authorityは再利用できない。後の具体的pilotが必要になった時点でのみ、新しい範囲・資料・実行承認を分離する。一方、既に合意された案の型修正や記号検算を、過剰な個別レビューで分割しない。

## 13. 既存成果の扱いと論文への着地点

G10技術ノートv1.0は既に完成している。この全体reviewのためにv1.1を作り続ける必要はない。

R0の限定最適性、G1の6頂点、G4-A/G5の条件付き分離、SPの機構例は、後の研究で利用できる。ただし、異なる成果を詰め合わせただけで一つの新規アルゴリズム論文になるとは判断しない。

今後の着地点は三つに分かれる。

1. 差のある新構成が成立するなら、そこを主要貢献とし、必要な旧成果だけをbaseline／補助理論へ使う。
2. 新構成は既知でも、PR全体で有用な適用域を示す研究を選ぶなら、応用・資源比較として評価する。
3. どちらの問いも明確にならなければ、現技術ノート等を成果として、新規実験を休止する。

「独立論文にならなければ全成果が無価値」でも、「監査と文書が多いので投稿十分」でもない。

## 14. 今回の判断の確かさと限界

**比較的確かな部分**は、履歴の訂正、旧系列を未消化候補として再開する不適切さ、限定positiveの保持、同辞書救済のscope、G10文書化が完了済みであること、強い既知法を比較から外せないことである。

**不確かな部分**は、新しいPR-specific補正・再構成が本当に既知法を超えるか、その最良の構成が何か、どの実問題で利益が残るか、独立新規性・投稿可能性である。本レビューはこれらを証明・実証したとはしない。

それでも現在の行動は決められる。既存候補の実験を増やすのではなく、旧固定mean最適化とは別の自由度について、具体的で反証可能な構成があるかを先に確かめる。ない場合は休止する。

## 15. 最終状態表

| 項目 | 本レビューの判断 |
|---|---|
| Track B全体の研究方針レビュー | 完了 |
| 旧候補一覧の古い未完了表示 | 現状態として使わない。RA-D0の未実行説明を訂正 |
| 既存固定系列の新たな実験拡張 | 推奨しない |
| 既存成果 | 範囲を保ち保存、必要に応じ再利用 |
| 次に優先する問い | 同じ物理的複素signalで、D/R構造を低負担の補正・再構成へ移す具体的利益があるか |
| 次の新手法・優位性 | 未確立。generic補正との差を要する |
| 次の本番実験 | 未認可 |
| 次の実務 | まず具体的構成の理論設計。成立した案の技術確認はCodexへまとめる |
| 成立する差がない場合 | Track B新規実験を休止。別名のpilotへ自動移行しない |
| Track A／外部Hamiltonian前処理研究 | 本レビューの採否対象へ混入しない |
| G10技術ノート | 完成済みのまま、再完成・同一review不要 |

**総括：Track Bは、限られたアルゴリズム候補の比較を通じて、どこに改善があり、どこで強い対照に吸収されるかを明らかにした。次の一歩を「残りのタスクを実行すること」には置かない。PRの物理的な目的へ戻り、実装可能な誤差構造を使う具体的な差があるかに研究資源を限定する。それが得られなければ、現在の成果を持って新規実験を休止する。**

---

## 参考資料・取得範囲

### Repository：固定commit 22a0d032717b4139cdfa806850643e93ba25624f

以下は本レビューの事実確認に用いた正本。リンク先の結果を全て今回再実行したという意味ではない。

[R1] 研究概要・現状。前段P-D/R3/FR、Track B移行、最新追記。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/research/研究概要・現状.md

[R2] 2026-10-09 Track B研究再評価・GPTチェックポイント。RA-D0実行履歴と当時の研究RQ。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/research/track_b_reassessment_and_gpt_checkpoints_20261009.md

[R3] Track B README。最新と書かれた古い記事は履歴として読む。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/tracks/algorithm_codesign/README.md

[R4] RA-D0 v3 one-shot結果。比較0、technical incompleteの根拠。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/tracks/algorithm_codesign/ra_d0_one_shot_result_validation_20261007.md

[R5] G5固定辞書終了認証・結果。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md

[R6] G5 claim/evidence対応表。R0〜G5の成果と非claim。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/tracks/algorithm_codesign/g5_claim_evidence_map_20261010.md

[R7] B-F現仮説closure。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md

[R7a] BF保存値recovery audit。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/artifacts/track_b_bf1_read_only_recovery/2026-10-05/v1/result_validation_audit.json

[R8] SP-1保存結果・監査の役割。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md

[R9] BS-0.5 method/target設計監査。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md

[R10] BM-0.5 compact BCH同値性。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md

[R11] 技術ノートv1.0。既存G6/G9の一般構成・P5、G10の固定native結果とclass分離。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md

[R12] G10論文化・着地点レビュー。既に採用した個別系列の判断。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/research/track_b_G10_v3_publication_and_endpoint_review_20261011.md

[R13] G10科学レビュー。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/research/track_b_G10_v3_scientific_review_20261010.md

[R14] 技術ノート完成記録。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/tracks/algorithm_codesign/g10_publication_endpoint_completion_20261011.md

[R15] Live branchの照合先（回答作成時点HEADは本レビュー固定commitと一致）。
https://api.github.com/repos/HIROMU1015/Partially-Randomized-Trotter/branches/track-b-return-aggregation-technical-note-20261011

### 一次文献

[P1] *Phase estimation with partially randomized time evolution*, arXiv:2503.05647v2, 2026-07-10。本文とFig.6のあるPDF p.9を確認。PRのbias／samplingの区別を使用。表・他論文の数値を本研究の費用として移植しない。
https://arxiv.org/abs/2503.05647
https://arxiv.org/pdf/2503.05647

[P2] Maxwell et al., *Practical Estimation of Trotter Error for Hamiltonian Simulation*, arXiv:2606.30738v1。§III.2の再帰・cache・compact BCHを確認。
https://arxiv.org/html/2606.30738v1

[P3] D. Cugini, T. A. Atif, Y. Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1, 2026-03-13。Theorems 1–2、composite条件、ZeroFill/Discardの関連本文。
https://arxiv.org/html/2603.13495v1

[P4] J. Peetz, S. E. Smart, P. Narang, *Quantum Simulation via Stochastic Combination of Unitaries*, arXiv:2407.21095v2。Theorem 1とsampling関連本文。
https://arxiv.org/html/2407.21095v2

[P5] P. Zeng, J. Sun, L. Jiang, Q. Zhao, *Simple and High-Precision Hamiltonian Simulation by Compensating Trotter Error with Linear Combination of Unitary Operations*, PRX Quantum 6,010359 (2025)。出版社の本文入口・抄録、arXivメタデータ、および[P6]でのNCC/PSTC位置付けを確認。全付録を独立再証明していない。
https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.6.010359
https://arxiv.org/abs/2212.04566

[P6] X. Wang et al., *Trotter error compensation with polylogarithmic precision and nested-commutator scaling without ancillas*, arXiv:2607.11856v2, 2026-07-23。target・k-local条件・LCQC・sampling負担・比較を確認。v1のscalingやsystemだけのancilla数を本研究へ混ぜない。
https://arxiv.org/html/2607.11856v2

[P7] *TE-PAI: Exact Time Evolution by Sampling Random Circuits*, arXiv:2410.16850v2。exactnessの意味、classical／measurement負担、coherent相関関数への接続を確認。
https://arxiv.org/html/2410.16850v2

[P8] Wada et al., *Trade-offs between Quantum and Classical Resources in the Linear Combination of Unitaries*, arXiv:2512.06260v2。取得PDFはv2。grouped構成とquadratic targetの関連箇所、PDF p.1/p.4を視認。全T価格のpartition単調性と解釈しない。
https://arxiv.org/pdf/2512.06260

[P9] *Efficient and practical Hamiltonian simulation from time-dependent product formulas*, Nature Communications (2025), DOI 10.1038/s41467-025-57580-5。THRIFTのResultsと実装仮定。
https://www.nature.com/articles/s41467-025-57580-5

[P10] P. K. Faehrmann et al., *Randomizing multi-product formulas for Hamiltonian simulation*, Quantum 6,806 (2022)。出版社の論文情報・抄録を参照。今回MPF全定理を独立に査読したものではない。
https://quantum-journal.org/papers/q-2022-09-19-806/

[P11] D. Cugini, pathwise-correlated Hamiltonian simulationに関するarXiv:2608.29756 (2026)。検索で抄録を確認したが本文取得は失敗。詳細比較や優位性判断の根拠には使用しない。
https://arxiv.org/abs/2608.29756

[P12] *Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas*, PRX Quantum 7,033018 (2026)。出版社でcorrector・perturbed systemとTHRIFTへの言及を確認。全文の最適性証明の再監査ではない。
https://journals.aps.org/prxquantum/abstract/10.1103/phk5-slkl

---

このMarkdownは本レビューの記録であり、新しい数値実験の再現データではない。科学計算を実施していないため、新しい「科学計算再現ZIP」を作成したかのような添付は行わない。
