# N3幅比較v2・固定6入力pilot：科学レビューと研究全体の再評価

- レビュー日：2026-10-13 JST
- 状態：ユーザーの明示的な開始承認を受けて、科学的解釈・研究方針のレビューを実施
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 実行branch：`n3-width-comparison-v2-execution-20261012`
- 結果payload：`6d8f769b857e50c8d262e7a4e94f8277a8967597`
- 公開commit：`f07591a965f1c228e7af497b71c42e337a5da300`
- 数値source：`3eba0814e690e95993db60809c4c3c96784b07c0`
- 実行認可commit：`fcc75c9ff442df5368c316fd1cc16c475278f3c8`
- input SHA-256：`4c59ba790c7a1faefa6e60dbfbde3d13b718e168bb6491fa71594103b3ac8395`
- plan SHA-256：`dc43a8d781e76ba4c3fc92f528c973f225861bf47160c5866bff2bebd9eb9d53`
- 新しい科学runner、固有値計算、最適化、証拠探索、登録入力での再実行：本レビューでは0
- レビューと計算実行の認可は別。既存Mandatory STOPを維持

## 0. 総合判断

**N3は限定継続とするが、今回のpilotだけを根拠に主研究として採択・大系拡張・実分子展開へ進めない。直近の次作業は、Codexによる保存結果の取得費用・数値品質の帰属監査と研究主張の整理である。**

これは同じ6入力を再実行する指示ではない。過去の数学的成果を取り消す判断でもない。今回の科学的結論は、次の四点に分ける。

1. **固定task**：投影誤差予算0.01では、NORMが全6H／12候補modeを認証した。OCCは追加のH縮約を得ていない。このtaskだけなら、測定されたNORM経路の方が短時間だった。[R1]
2. **有限取得法の性能**：保証幅とcold取得時間を二指標で見ると、OCCは固定した参照実装に対して有用な中間点を持つ。幅はNORMより5/6Hで狭く、同値SDPより6/6Hで狭く、同値SDPのcold中央値より短かった。ただし数学的に同値な条件族への新しい保証能力ではない。[R1,R2,R4]
3. **数学的な限界**：保存されたSOS証拠と理想OCC反証を同じqueryで組み合わせると、`V2_two_external_stress_5`、external 4、δ=1/400で、SOSは成立し理想OCC条件は不成立である。公開済み保存監査を前提とする論理的帰結であり、新しいsolver実験ではない。[R5,R6,R7]
4. **研究方針**：差の原因を同じ情報・品質・機能の下で切り分ける前に、入力数や精度を変えて成功例を増やさない。新規性・中心テーマ・量子総費用優位は未採択。N1停止、N2保留、旧primaryと旧2-base gateは維持する。

今回のレビューが新たに支持するのは、**OCCが万能に強い方法であることではなく、限定された保証族を構造利用して取得する軽量な実装候補であること**である。この候補を独立した算法研究として押し出せるかは、取得差の帰属と、何のために幅を小さくするかを具体化できるかに依存する。

## 1. レビュー範囲、資料、実際に行ったこと

### 1.1 利用した固定資料

結果報告、公開・再取得報告、数学契約、事前登録、文献追補、独立保存監査、厳密反証query索引をGitHub接続で取得した。元sourceのgenerator、pipeline、runner、独立checkerを読み、全6Hのkernel層・反復0についてCSVの全7法×2modeの行を確認した。[R1–R10]

前回レビューのMarkdownと研究プロジェクトの最新版共通指示も読んだ。N1/N2/N3や初期表現探索については、既にこのチャットで取得・検討した固定結果と前回レビューの判断を引き継ぐ。別Trackの誤送信は今回の研究判断へ使わない。他のPR研究の現在の進捗を再確認したとは主張しない。

### 1.2 今回の独立作業

- source上の候補生成、SDP呼出し、不可行証拠の生成、有理化・PSD修復、SOS残差会計、checker構成、timing境界を静的に確認。
- 公開CSVの表示幅から、NORM→OCCおよび同値SDP→OCCの表示縮小率を計算。
- 公開結果の丸め済みcold中央値から、説明用の時間比を計算。これはraw timingの再集計ではない。
- 公開CSVのexactなSOS上端と1/400との厳密大小をFractionで確認。他の候補modeのOCC上端が1/400より小さいことも確認。
- certificate件数・refutation件数の公開監査集計を加算照合。
- SCS/CVXPY公式資料、検証付きRDM下界およびweighted SOSの一次文献を外部確認。外部情報は研究結果そのものと区別する。

### 1.3 実施していないことと限界

1224 certificateの全PSD再検査、36反証の全matrix再認証、全504 recordのraw再生成、数値solver、固有値計算、性能benchmarkは行っていない。全件の有効性はCodexの公開済み独立保存監査を根拠とする。[R6]

約30MBの全raw case結果をこのレビュー環境で再取得して一括検算したとは扱わない。コンテナの直接HTTP取得はDNS障害で失敗したため、GitHub connectorで取得できたsource、結果報告、CSVの関連行、監査記録を使用した。これはrepository側の未pushやデータ不存在を意味しない。

本レビューは科学的解釈・研究方針を判断したもので、producer全体の形式的soundness証明や第三者環境での再現を認定したものではない。後述のSOS/OCC分離の接続は、取得したquery索引・CSVおよび公開証拠監査を前提にする。

## 2. 研究全体の経緯と今回の位置

### 2.1 目的を置き換えない

この研究は、PR内部の乱択則そのものの改良とは区別して、Hamiltonianの表現・構成・前処理から新しい算法や本質的な改善を得る目的で進んできた。監査件数を増やすことや、汎用solverより短いPythonプログラムを作ることだけへ目的を置き換えない。

理論的な限定classで独立した寄与が成立する道もある。実分子の量子総費用削減を、あらゆる理論研究の成立に必要な絶対条件として追加しない。一方、そのような寄与が確立していない状態で、実装成功を研究完成と呼ばない。

### 2.2 経緯

| 系統／段階 | 支持された内容 | 未解決または否定的な内容 | 今回への意味 |
|---|---|---|---|
| 初期の独立表現探索A/B/C | core・補助空間・構造化表現の構成・比較 | 局所proxyの改善が精度込み費用へ直結しなかった | 評価目的と途中の数値を分ける |
| N1 | 縮退/frame構成、保証付き費用selector | 固定3contextの最終selectorはexact、改善0% | 誤差幅や回路だけの改善を総taskの勝利としない |
| N2 | signed chargeとcompute/phase/uncompute | 固定小系では全費用が高い | 部品の低rank化だけで有利とはしない |
| N3直接generator | 係数保証、方向情報 | 強い下界・既存対照を入れると追加利益が限定的 | 弱い対照だけへの勝利に依存しない |
| OCC固定8入力 | 有理certificateを取得 | 既存対照も8/8、追加保証0、旧2-base未達 | 飽和した採否だけでは幅差を識別できない |
| 保存幅・S1旧族分離 | 幅改善と全旧族排除、登録旧OR下端 | 同じS1の事後development、全既知法比較ではない | 数学的な成果として保存するが適用能力と区別 |
| 統合理論レビュー | Fantope同値、SOS包含、縮約taskと取得費用 | 新規性・有限cold取得能力は未評価 | 今回pilotを必要とした理由 |
| 今回v2固定6入力 | 新しい有限取得法の幅・費用比較、SOSと同値SDPを含む | 目標0.01の追加H縮約0、取得差の原因は未帰属 | 規模拡張ではなく主張と原因を整理する |

N1/N2は一般的不可能性として切り捨てたのではない。ただし、新しい構成機構の根拠なしに再開する優先度は低い。N3は現在最も具体的な数学的根拠を持つが、それだけで主研究採択を自動化しない。[H1,H2,H3]

### 2.3 前回レビューからの判断変更

前回は、同等SDP・強いSOSを含む比較を新しく実装し、有限取得費用の違いが残るかを確かめる必要があった。今回はその結果が得られた。よって、同じ小pilotをさらに拡張するより、現比較に含まれる機能差・数値品質差・証拠修復の影響を分ける方が情報価値が高い。

これは新しいテストPASSのたびにレビューを増やす判断ではない。研究の主要な未確定点が、可用性から『差の帰属とtask上の価値』へ移ったためである。

## 3. 固定6入力pilotの条件をそのまま評価する

対象は3 stress family×full modes5/6の6H、ν2/3、active4/5、候補modeは各Hで{0,n−1}、一候補につき一mode空射影、非負unordered density、ε_proj=1/100。[R2,R3]

geometry、化学basis、DF rank、L_D、PF窓、時間発展、QPE/RPE shotsの評価は適用外。単位は係数recipeの共通energy単位で、Hartreeへ読み替えない。

Uはlexicographic determinantの先頭最大8件の期待値最小、cは外部占有sectorのone-body固有値和の有理下界である。全occupationを列挙した最良U/cを共通に無料提供したのではない。この点は以前の懸念に対する改善である。[R2]

kernel/cold各3反復で、6×2×7×2×3=504 mode-method records。504は独立Hamiltonian数ではない。1224 certificateも独立試験数や統計的標本数とは違う。反証36recordは6種類の(case,external,δ)の反復である。[R1,R5,R6]

幅とcold取得費用は事前登録された評価項目である。したがって『目標0.01で追加縮約0だからpilot全体が失敗』とは判断しない。また、今回の幅評価を以前の保存幅解析のように初めて事後導入した指標とも扱わない。[R3]

## 4. 目標0.01での縮約task

NORM/OCC/SDP_EQUIVALENT/DENSITY_SOSはすべて6/6H、12/12候補を受理。FIVEは5/6H・7/12候補、OLDは5/6H・8/12候補。FULL_H_LOWERは有効な下界証拠を得たが、今回の目標には全件で達しなかった。[R1]

**固定された採否だけならNORMで足りた。** 測定cold中央値もNORMが全6HでOCCより短い。これ以上の精度を必要としないtaskにOCCを追加する利益は、このpilotでは示されていない。

ただし、全法の目的を採否だけに強制的に変更しない。幅を小さくする価値が独立に存在するtaskなら、次節の二目的評価が必要になる。今回の段階で、未計測の後段量子費用を仮定して正当化してはいけない。

再び主対照が全件受理する状況になったことは、確認した応用taskとテスト集合の関係を見直す理由になる。ただし、結果前にNORMを実行して失敗例だけを採用すべきだったという意味ではない。今後の新しいtaskは実際の用途や理論から設定し、結果後の入力篩選で『独立成功』を作らない。

## 5. 保証幅と取得費用：何が有望なのか

### 5.1 幅比較の大きさ

次表は公開CSVの表示幅および報告書の丸めcold中央値からの説明用算術。採否は公開されたexactな証拠と区間比較を根拠とし、この表の小数を新しいgateに使わない。[R1,R4]

| H | NORM→OCC 幅縮小 | 同値SDP→OCC 幅縮小 | 同値SDP/OCC cold中央値比 |
|---|---:|---:|---:|
| banded_5 | 0.0000% | 0.117435% | 2.085 |
| banded_6 | 44.6805% | 0.002531% | 1.942 |
| dense_density_5 | 41.4270% | 0.005173% | 1.679 |
| dense_density_6 | 5.0237% | 0.020624% | 2.368 |
| two_external_stress_5 | 32.3765% | 0.002833% | 1.605 |
| two_external_stress_6 | 1.0311% | 0.005103% | 1.752 |

同値SDPとの差はH最小幅では約0.00253–0.11744%。NORMとの最大約44.68%・41.43%などとは桁が異なる。『同値SDPより6/6で狭い』は正しい有限結果だが、それだけを強調して新しい保証能力を印象付けるべきではない。

原表の範囲ではOCCはcold約33–71ms、同値SDP約68–169ms、NORM約7–12ms。上の1.605–2.368という値は丸め表示中央値の比であり、安定した一般speedupや計算量次数を示すものではない。

### 5.2 記述的な非支配性

保存幅上端と報告cold中央値の二指標に限れば、OCCは5/6Hで非支配点に残る。

- NORMは安いが、この5Hでは幅が広い。
- FIVE/OLD/FULL_H_LOWERを含むcheap ORの幅も、この5HではOCCより広い。
- 同値SDPは今回の保存幅が広く、cold中央値も長い。
- SOSが狭いHでもcold中央値は長いため、OCCを両指標で支配しない。
- `V2_banded_5`ではNORMとOCCの保存幅が同じでNORMが安いため、OCCはNORMに支配される。[R1,R4]

これは固定実装・固定表示中央値に対する記述的評価である。理想的なPareto最適性、統計的な信頼付きfront、未計測のOR専用運用費用までの最適性を意味しない。

### 5.3 coldの意味

今回のcoldは、各法がfresh Engineで係数bound取得・候補生成・有理化/修復・checker・width・selector・メモリ内serializationを行う費用である。依存import、共有case-file disk IO、プロセス起動等を含む利用者視点の全cold費用ではない。除外は事前に明示され、方法間で一貫している。[R3,R8]

除外があるから比較が無効という意味ではない。どの費用を測ったかを限定して主張する。各methodのRSSも境界値とプロセスlifetime highwaterで、孤立method peakではない。

## 6. sourceから確認した、時間差・幅差の解釈上の制限

### 6.1 同値SDPは取得している出力が多い

OCCは32-stepのconcave spectral searchから端点とdyadic候補を作る。一方、SDP_EQUIVALENTはδ={0.01,0.005,0.0025}の各tierで1回ずつ解き、正の幅証拠に加えてdualからγを取り出し、trace修正・identity混合・有理PSD検査を行い、理想OCCの不可行証拠も取得する。[R2,R8]

この不可行証拠の生成は、現在のOCC経路にはない。少なくとも一部の参照法費用は追加の情報を返すためのものである。したがって、今回のcold差を全て『同じ出力を返す核心算法の速度差』に帰属できない。

異なる算法に同じprimitive呼出し回数を強制する必要はない。3solver callsと32spectral stepsの数を単純に同じにするのも不適切である。必要なのは、比較したい証拠品質・達成task・出力義務に対して、何が含まれた時間かを分けることである。

保存時間から負証拠処理を勝手に控除したり、3で割って『1回なら同等』としたりしてはいけない。未計測の反実仮想である。

### 6.2 探索・solver・有理化の精度は同じ物差しではない

数値OCCの32更新・40bit proposalとSCSのeps_abs=eps_rel=1e−6は、同じ最終width品質を指定するものではない。SCSの許容誤差はprimal residual、dual residual、duality gap等の規約に関係し、δ幅誤差を直接1e−6以下と保証する設定ではない。[R2,R8,W1]

`optimal_inaccurate`18callは、要求数値精度へ到達していないsolver出力として扱う。ただし、その後のexact certificate検査を通っているなら保証の健全性はそのexact証拠が支える。『solverがinaccurateだからcertificateも無効』と扱わない。一方、保証が有効でも最良幅へ十分近いとは限らない。[R1,R6,W2]

今回確認した記録だけでは、18callの全method/mode/tierへの帰属や、それが幅差・費用差をどの程度説明するかを完結していない。sourceに保存されたsolver_eventsの取得整理を次作業に含める。

### 6.3 有理PSD修復は、保証の幅を変える

sourceのOCC/SDP証拠では、ZとR=K−τI+Zを同じrIで修復する場合、τ,κを固定すると

\[
s'=\nu\tau-\operatorname{tr}(Z+rI)=s-mr,\qquad q'=q+mr.
\]

box条件が非活性でq>0なら、widthはδ(g+δ)=qの根であり、

\[
\frac{d\delta}{dq}=\frac1{g+2\delta}.
\]

従ってproposalやPSD修復の違いだけで、厳密には異なる保存幅が生じ得る。これは今回の差の原因として成立し得る解析であって、『全差が丸め由来と証明した』という結果ではない。[R8]

SOSもGramを修復した後、係数一致の残差Eに対するeを差し引き、s=ℓ−eとする。修復前の理想的な優位が、有限solverと残差評価によって失われ得る。反対に、有効なSOS証拠が実際にOCCより狭い場合は、その優位を無視してはいけない。[R2,R9]

### 6.4 数値精度の追加最適化を直ちに認可しない

上の指摘から『SCSの許容誤差をもっと小さくする』『OCCの精度を下げて同じ回数にする』という追加実験へ自動的に進まない。まず保存されたcandidate、solver status/iterations、phase time、repair前後の保存範囲を確認する。

raw float proposal・repair前objective・repair量などが未保存なら、保存されていないと記録する。後から再実行して復元した値を原記録として補ってはいけない。

### 6.5 全法で同じ最適化精度を達成したと扱わない

現OCCはq(κ)をκ∈[0,g+ε_proj]で探索し、有限κ証拠のwidth最小を返す。SDP/SOSは3つのDに対する候補を返す。証拠のwidthは正しく保証されても、この有限方式がすべてのκ,δに対して最適な幅を得るという完全性・近似最適性の証明は別である。

とくにwidth=max(0,κ−g,root(q))にはbox項があるため、一般にqだけの最小化とwidth全体の最小化を無条件で同一視しない。今回これが誤った採否を生んだ証拠は確認していない。報告が『有限保存幅』と限定していることは適切である。[R2,R8]

## 7. 保存記録から読み取れる重要なSOS対OCCの分離

### 7.1 同じqueryを結び付ける

次の入力・mode・targetを固定する。

- `V2_two_external_stress_5`
- external mode=4（full5、active4、ν2）
- δ0=1/400=0.0025

反証索引[R5]には、このqueryで理想OCC条件族の厳密不可行が登録されている。CSV[R4]のkernel反復0のSOS上端は

\[
\overline\delta_{\mathrm{SOS}}=
\frac{1219549601967953115443785837705603355}
{510423550381407695195061911147652317184}
\simeq0.002389289446101497<\frac1{400}.
\]

固定証拠のδ受理は上方単調である。したがって公開監査が検証したSOS証拠はδ0でも有効であり、同じU,cにおいて

\[
\boxed{\text{SOS certificate exists at }\delta_0,\quad
\text{ideal OCC condition is infeasible at }\delta_0}
\]

を得る。[R2,R4,R5,R6]

本レビューではexactな上端<1/400を独立に照合した。SOS GramとOCC dualの全行列を自分で再認証したわけではない。分離のmatrix-levelの根拠は公開された独立保存監査であり、今回の新規solver runではない。

### 7.2 何を意味しないか

- 物理的にそのmodeを削除できない、という結果ではない。SOSは実際に認証する。
- SOSが今回OCCより多くHamiltonianを一mode縮約できた、という主評価結果ではない。
- 全SOS familyが実用的に安いという結果ではない。
- 36反証recordを36独立系の結果とはしない。6queryの反復である。

実際、この同じHのexternal0にはOCC上端約0.001988397586<0.0025が存在する。よってδ0でもOCCは別modeを一つ削除できる。**同modeにおける厳密分離と、H単位で一つでもmodeを削除できるかは異なる。**[R4]

### 7.3 研究全体への意味

以前は『OCCが旧LMIより強い』という片方向の結果が主であった。今回、より強いdensity-aware証拠が理想OCCでは認証できない同modeのtargetを認証する例が得られている。

これは、OCCを万能な最上位保証として扱う根拠を弱める一方、**保証の強さと取得費用を異なる層で扱う**研究の動機を具体化する。しかし、安価な方法から高価な方法へfallbackする原理だけを新規アルゴリズムとして採択することはしない。

## 8. SOSの大きい取得費用をどう解釈するか

今回の固定SOSは粒子・正孔・D・Q・Gの5 Gram blocksとnumber等式を持つ。[R2]

active m=4/5では、それぞれの行列サイズは

- particle/hole：m
- D/Q：m(m−1)/2
- G：m²

である。OCCが使うm×m PSD2個より一般的に大きい表現を解き、修復・残差評価・checkerへ進む。さらに現在の独立SOS checkerは固定ν-sectorのladder action行列を再構成し、そのdimensionはbinom(m,ν)である。今回では6／10だが、一般fillingで多項式規模とは限らない。[R9]

したがって、SOSの約1.2–6.5秒対OCCの約0.033–0.071秒を、そのまま一般SOSの本質的な計算量差や新しい算法の速度倍率と主張しない。

一方、これは実在する固定実装の取得費用であり、強い証拠を得る代償として記録する価値がある。『参照実装に改善余地がありそう』という理由だけで現在の結果を無効にもしない。

現sourceと保存情報から必要なのは、SOS assembly、solver、rational repair、sector checker、width/serializationの寄与を分けることである。保存情報で分離できない時間を推定で埋めない。新しいsolver切替やchecker大型最適化は、この帰属確認より先に主作業へしない。

## 9. taskの意味と研究上の価値

### 9.1 狭い保証幅が有用になり得る理由

縮約先のHamiltonianについて|E_est−a_d|≤ε_solverが得られるなら、元Hのground energyは

\[
[E_{\rm est}-\epsilon_{\rm solver}-\delta_{\rm cert},
 E_{\rm est}+\epsilon_{\rm solver}]
\]

に含まれる。中心を用いると半幅はε_solver+δ_cert/2であり、小さいδ_certは後段に割り当てられる誤差予算を増やし得る。[R2]

ただし、今回後段solver・PF・QPE/RPE・状態準備の費用は測定していない。選ばれたmodeが変われば縮約Hamiltonianも変わり、後段費用が同じとも限らない。δの相対改善率を量子総費用の相対改善へ変換しない。

### 9.2 現時点で支持するclaim

- この固定6H・同情報・固定backendで、厳密証拠を返すOCC実装を実現し、NORMより狭い有限幅を5Hで取得した。
- 同値SDPの固定取得手順に対して、H最小保存幅が全6Hで小さく、測定cold中央値も全6Hで短かった。
- 幅とcold費用を二指標とする記述的比較では、5HにOCCの非支配点が残った。
- 特定queryでdensity-aware SOSが理想OCC条件の限界を超える保存証拠が存在する。

### 9.3 現時点で支持しないclaim

- OCCは一般SDP/SOSより数学的に強い。
- 所定のε_proj=.01でOCCが追加のH縮約を実現した。
- 一般的な冷生成速度、漸近計算量、method memoryの優位を示した。
- 全1224 certificateを独立した科学サンプルと数えられる。
- SCSの18inaccurate callだけが幅差の原因だと特定した。
- 5/6の幅優位が論文の新規性や実用利益を自動的に確立する。
- 既存6Hのεを後から下げて得た成功を新しい確認的実験と呼べる。

## 10. 文献・新規性の再評価

### 10.1 既知の部分

Fantope、固有値和、Schur補元、粒子数制約、weighted SOS/v2RDM、数値SDPの検証付き下界は既知の枠組みである。前回の文献対応と今回の追補を保持する。[R10,W4,W5]

Chaykinら2020年論文の出版社abstractは、数値誤差で不正な下界が得られ得ること、VSDPで厳密な下界を評価することを述べている。これは厳密checkerの必要性を裏付けるが、N3の有理checkerを世界初とする根拠ではない。全文の実装比較は今回行っていない。[W4]

Rubin–Low–DePrinceのweighted SOS/v2RDM研究は、2026-08-21オンライン公開、2026-09-08号のJCTC22(17),8751–8766として出版社情報を確認した。過去のv1読解と出版版全式の一致を仮定しないが、固定粒子数等の制約とSOS/v2RDMの関係が既存研究であることは確認できる。[W5,W6]

### 10.2 今回のpilotが未発表性を確定しない理由

専用の最小固有値和評価と一変数探索が、変数を持ち上げた汎用SDPより小系で安いことは、構造利用として合理的である。しかし、その事実だけで『新しい量子アルゴリズム』や非自明な数理貢献になるとは限らない。

独立した寄与として候補になるのは、例えば、同じ認証品質を低い取得費用で保証する具体的な算法と解析、再利用可能なコンパクト証拠、あるいは目的を固定した縮約taskでの実質的な改善である。これらのどれを主要claimにするかは、まだ決着していない。

構造化SDPやRDM/SOSの一般実装を網羅的に比較したわけではない。SCSだけを相手にした小系prototypeの成績から、文献全体への優位や世界初を認定しない。

### 10.3 形式的な新しい条件を増やさない

今回のレビューを理由に、全solverの実装、大規模分子、全SOS hierarchy、量子回路の実機測定を必須化しない。研究目的に対する重要な未確定点は、取得差の帰属とtaskの必要性である。そこに答えない追加作業を増やさない。

## 11. 代替方針の比較と採否

| 方針 | 評価 | 今回の扱い |
|---|---|---|
| 同型の小入力や精度を増やす | 保存幅の差は増えても、現取得差の機能・品質の不一致を解決しない | 見送り |
| SCSの設定だけ調整して再競争 | 結果依存tuningになりやすく、目的がbenchmark勝利へずれる | 見送り |
| すぐ大系・実分子へ拡張 | checker、U/c、Hamiltonian class、taskの未解決を同時に増やす | 見送り |
| N3を即時全面終了 | 幅・費用の非支配性と数学的成果まで捨てるのは根拠不足 | 採らない |
| **保存結果の取得差帰属・証拠接続・研究主張整理** | 現時点の情報で科学的未確定を最小費用で絞れる | **直近の推奨** |
| 新しい理論寄与または応用taskへ限定再設計 | 帰属後に具体的な問いが残る場合に価値がある | 現時点で新batchは採択しない |
| N1/N2再開・別主研究へ戻る | N3を続ける根拠が乏しくなれば選択肢。ただし自動再開しない | 現状の停止・保留を維持 |

## 12. 次のCodex作業：関連作業を一括する

### 12.1 目的

新しい勝ち例を作るのではなく、**現pilotで支持できる研究主張と、保存情報で判断できない点を閉じた形で示す。** そのための以下の作業を一つの単位にする。

### 12.2 保存された比較経路と出力義務の対応

OCC、同値SDP、SOSの各pathについて、候補生成回数、固定tier、正の証拠、負の証拠、検証、修復、serializationを対応させる。各処理の全記録を参照し、同値SDPの追加のrefutation情報も成果として数える。

phase_times、solver_events、reported solve/setup time、iterations、exact checker費用など、既に保存された値をcase/mode/tierへ結び付ける。18件のoptimal_inaccurateの所属を特定する。反復を独立科学入力として増やさない。

全時間差を『refutationのせい』等と推測で配分しない。タイマのexclusive/inclusive、CVXPYのproblem構築・canonicalization・solve・postprocessのどこが分離可能かを明記する。

### 12.3 保存幅の差の帰属

raw数値proposal、修復前objective、Z/Gram修復量、SOS残差、候補κ、box活性を、保存された範囲だけで整理する。未保存なら未保存とする。保存情報だけで未判定な原因を、後から再生成した数値で埋めない。

『同じ理想族』『固定sourceで発見された証拠』『最良保証幅の最適性』『最終taskの成功』の四層を表へ分ける。必要な次の比較がある場合も、異なる演算回数を一律同じにするのではなく、同じ証拠目標を定義する。

### 12.4 SOS対理想OCCの証拠接続

`V2_two_external_stress_5`, external4, δ=1/400について、既存のSOS勝者と既存のOCC反証のraw pointer、同じ入力/U/c/ν、元のchecker結果を一つの小さい索引へ結ぶ。同Hのexternal0のOCC証拠も添え、同mode分離とH縮約を混同しない。

これは新証拠の探索ではない。原certificateを変えず、元の監査を参照または許可されたread-only照合で利用する。資料を小さくするために証拠の数値を丸めない。

### 12.5 研究の収束判断資料

N3の既存理論を保存し、論文の候補claimを『確認済み』『有限prototypeのみ』『外部文献未確認』『追加の数値判断が必要』へ分類する。現データから実際に支持される用途がない場合は、それも明記する。

外部文献の情報取得と技術的対応はCodexでまとめる。独立した非自明な数理的差分の採択・研究方針の転換はGPTの科学判断へ戻す。

### 12.6 実行と承認の境界

この提案は、既存の消費済みrun1認可を再利用する指示ではない。元source/result/STOP、別Track、root dirty、既存manifestの履歴を保持する。

保存結果の新たな数値的事後評価、証拠の再認証等を実行する場合は、個別の運用指示に応じた明示的な作業・実行承認を得る。結果を見た新入力・新ε・新solver・新certificate・新最適化・元batch再実行は含めない。

通常の資料抽出・source確認・synthetic・公開のたびにGPT詳細レビューへ戻さず、上の資料が一式揃った段階にまとめる。重大なsoundness・比較意味論の問題が見つかった場合だけ途中で科学判断へ戻す。

## 13. 今後のGO／縮小／終了の判断

### 継続に意味がある場合

- 同じ認証目標・情報・出力義務を整理しても、構造利用による取得費用の差が非自明に残る見通しがある。
- 狭い幅が必要な具体的な縮約task、または独立した理論問題が説明できる。
- 次の限定計算が何を否定・支持するか明確で、単に成功例を増やす試験ではない。
- 既存RDM/SOS/構造化最適化との差分が具体的な主張として整理できる。

### 縮小・整理へ進む場合

- 主張が『小さい行列の専用計算が持上げSDPより速い』だけに残り、新しい理論・task上の利益がない。
- 差が出力機能の違い、証拠修復、設定依存の取得品質に還元され、研究目的に対する利益を示せない。
- 目標を動かす以外に追加縮約の必要性を説明できない。
- より妥当な対照との情報・費用比較で独立の寄与が残らない。

その場合も、既存のN3理論、S1分離、検証付き実装、SOSに対する限界例を技術・研究記録として保存する。全Hamiltonianにおける数学的無価値や不可能性と結論しない。主研究を別方向へ戻すことと、成果を捨てることは別である。

## 14. 最終的な状態

- 本科学レビュー：実施・完了。研究全体を含む方針判断を報告。
- 技術的な時間差・幅差の完全帰属：未完了。次のCodex保存資料整理の対象。
- N3：限定継続。主研究の最終採択・規模拡張は保留。
- 次の担当：Codex。保存結果の帰属・証拠接続・主張整理をまとめる。
- 新しい科学batch、再試行、新入力、精度変更：未認可。
- 元run1 Mandatory STOP、旧primary、旧2-base gate：保持。
- N1停止、N2保留：維持。
- 全体manifestのdescription欠落FAIL：未解消のまま区別し、個別証拠PASSと混同しない。これを直すだけで科学的結論が変わるとはしない。

## 15. 参照資料

以下の番号は本文中の根拠への参照。GitHubのbranch HEADではなく固定commitを正本にする。

### 今回のrepository資料

[R1] 一次結果。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/docs/research/n3_width_comparison_v2_results_20261013.md

[R2] 数学・対照・証拠の実装契約。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/docs/research/n3_width_comparison_v2_contract_20261012.md

[R3] 固定6入力・事前登録。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/docs/research/n3_width_comparison_v2_preregistration_20261012.md

[R4] 保存CSV。今回確認したのはheaderとkernel反復0の全6H：raw file lines 1–15、86–99、170–183、254–267、338–351、422–435。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/saved_metrics/mode_method_records.csv

[R5] 6種類の厳密反証query索引。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/strict_refutation_queries.json

[R6] 独立保存監査。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/independent_saved_audit.json

[R7] 公開・remote取得。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f07591a965f1c228e7af497b71c42e337a5da300/docs/research/n3_width_comparison_v2_results_publication_20261013.md

[R8] generator source。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eba0814e690e95993db60809c4c3c96784b07c0/src/trottertracks/representation_exploration/n3_width_generator_v2.py

[R9] 独立checker source。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eba0814e690e95993db60809c4c3c96784b07c0/src/trottertracks/representation_exploration/n3_width_check_v2.py

[R10] 文献追補。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/docs/research/n3_width_comparison_v2_literature_amendment_20261012.md

[R11] methodとselectorのpipeline。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eba0814e690e95993db60809c4c3c96784b07c0/src/trottertracks/representation_exploration/n3_width_pipeline_v2.py

[R12] runner／依存import／process。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eba0814e690e95993db60809c4c3c96784b07c0/scripts/run_n3_width_comparison_v2.py

### 引き継いだ研究履歴

[H1] N1復旧後固定結果。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/docs/research/n1_decode_recovery_v1_results_20261011.md

[H2] N1/N2構成・N3係数下界の初期結果。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/docs/research/hamiltonian_algorithm_design_results_20261011.md

[H3] 理論・task統合の固定claim map。https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2de3b26fc36df9e99b7504631c97635f1473cb51/docs/research/n3_theory_claim_map_v1_20261012.md

前回GPTレビュー：`n3_theory_task_integration_v1_scientific_review_2026-10-12.md`（このチャット添付、ローカル本文確認）。過去資料を今回再実行・全件再監査したとは扱わない。

### 外部一次資料（2026-10-13確認）

[W1] SCS official Algorithm / Termination criteria。閲覧時の現行docsは3.2.11、pilotの固定版3.2.8と同一のruntimeを検証したものではない。一般的な停止条件の意味を参照。https://www.cvxgrp.org/scs/algorithm/index.html

[W2] SCS official Exit flags。SOLVED_INACCURATEはdesired accuracyに未達のbest guessと区別される。https://www.cvxgrp.org/scs/api/exit_flags.html

[W3] CVXPY official Solver Features、solver_stats、warm startの説明。今回の固定runtimeの再検査とは区別。https://www.cvxpy.org/tutorial/solvers/index.html

[W4] D. Chaykin et al., Rigorous Lower Bounds for the Ground State Energy of Molecules by Employing Necessary N-Representability Conditions, J. Chem. Theory Comput. 16(12), 7342–7356 (2020). DOI:10.1021/acs.jctc.0c00497。出版社書誌・abstractを確認、全文・全性能実装未比較。https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00497

[W5] N. C. Rubin, G. H. Low, A. E. DePrince, Near-Frustration-Free Electronic Structure Hamiltonian Representations and Lower-Bound Certificates, J. Chem. Theory Comput. 22(17),8751–8766 (2026), DOI:10.1021/acs.jctc.6c00318。出版社の書誌、公開abstract、HTML本文を確認。https://pubs.acs.org/jctcce/article/22/17/8751/5287195/Near-Frustration-Free-Electronic-Structure

[W6] 同論文arXiv v1（過去レビューとの版対応、最新出版版と全式同一とは仮定しない）。https://arxiv.org/html/2602.05069v1

[W7] 著者所属機関Google Researchの書誌。https://research.google/pubs/near-frustration-free-electronic-structure-hamiltonian-representations-and-lower-bound-certificates/

## 付録A. 本レビューで用いた算術コード

次のコードは公開された表示スカラーとexactな幅上端を手入力で転記したものであり、repository原fileのbyte exportではない。result commitを明記し、数値sourceやsolverをimportしない。表示比率は説明用で、新しい科学的gateや登録結果の再分類に使わない。

```python
"""Read-only review arithmetic from published display scalars and one exact width.
No repository modules, solvers, eigensolvers, new Hamiltonians, or runners.
Timing ratios use displayed rounded medians, not raw timing observations.
Source: fixed 6d8f769b857e50c8d262e7a4e94f8277a8967597 result report / saved CSV.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path

rows = [
 # id, norm width, occ width, sdp width, sos width, NORM ms, OCC ms, SDP ms, SOS ms
 ('V2_banded_5','0.00019301297180414632','0.00019301297180414632','0.0001932399028546859','0.00020328493528962616','7.100','33.299','69.440','1195.192'),
 ('V2_banded_6','0.0005830445420328433','0.0003225371181145687','0.0003225452825647359','0.0013097773196476792','11.733','57.320','111.341','5776.770'),
 ('V2_dense_density_5','0.00210374815378043','0.001232228196676455','0.001232291946923245','0.0012759554141395333','7.116','41.790','70.162','1603.426'),
 ('V2_dense_density_6','0.001598413690027815','0.0015181142326897515','0.0015184273937374076','0.0015855330605459704','11.548','71.176','168.523','6500.116'),
 ('V2_two_external_stress_5','0.0029403943553579404','0.0019883975862574417','0.0019884539212336742','0.0017456173389045358','7.104','42.275','67.851','1211.030'),
 ('V2_two_external_stress_6','0.004820368414458942','0.004770665141669009','0.004770908585629643','0.004950524213575165','11.449','71.104','124.597','4097.345'),
]
with localcontext() as ctx:
    ctx.prec = 40
    calculated = []
    for row in rows:
        ident = row[0]
        n,o,d,s,tn,to,td,ts = map(Decimal,row[1:])
        assert o <= n and o < d and tn < to < td
        calculated.append({
            'input_id':ident,
            'norm_to_occ_width_reduction_percent_from_displays':str((1-o/n)*100),
            'sdp_to_occ_width_reduction_percent_from_displays':str((1-o/d)*100),
            'sdp_over_occ_cold_median_ratio_from_rounded_ms':str(td/to),
            'occ_over_norm_cold_median_ratio_from_rounded_ms':str(to/tn),
            'sos_over_occ_cold_median_ratio_from_rounded_ms':str(ts/to),
            'occ_raw_width_strictly_below_sos_from_display':o<s,
        })
    # Exact published SOS upper endpoint, kernel repetition 0, external mode 4.
    sos_high = Fraction(1219549601967953115443785837705603355,
                        510423550381407695195061911147652317184)
    target = Fraction(1,400)
    occ_low = Fraction(5981041463670632830819417428869033213,
                       2041694201525630780780247644590609268736)
    other_mode_occ_high = Fraction(4059699822189378686961683520918622805,
                                   2041694201525630780780247644590609268736)
    assert sos_high < target < occ_low
    assert other_mode_occ_high < target
    cross = {
        'input_id':'V2_two_external_stress_5','external':4,
        'target':[target.numerator,target.denominator],
        'saved_sos_width_high':[sos_high.numerator,sos_high.denominator],
        'sos_width_below_target':True,
        'strict_margin_target_minus_sos_high':[ (target-sos_high).numerator,(target-sos_high).denominator],
        'saved_occ_width_low_above_target':True,
        'ideal_occ_infeasibility':'Source-reported exact refutation at this query; not re-certified by this scalar script.',
        'other_mode_0_OCC_already_certifies_target':True,
        'meaning':'Same-mode certificate-family separation conditional on published proof audit; not extra Hamiltonian reduction.'
    }
    report = {
        'kind':'REVIEW_DISPLAY_ARITHMETIC_AND_EXACT_SCALAR_COMPARISON',
        'source_transcription':'display scalars from published CSV/report; not a raw-file export',
        'result_commit':'6d8f769b857e50c8d262e7a4e94f8277a8967597',
        'new_science_runner_calls':0,'new_eigensolver_calls':0,'new_solver_calls':0,
        'new_certificate_searches':0,'full_PSD_audit_reexecuted':False,
        'rows':calculated,'cross_source_inference':cross,
        'published_certificate_count_total':sum([180,180,210,216,216,222]),
        'published_refutation_record_total':sum([0,0,6,6,6,18]),
        'repetitions_are_not_new_inputs':True,
    }
    assert report['published_certificate_count_total']==1224
    assert report['published_refutation_record_total']==36
    out = Path(__file__).with_name('review_arithmetic.json')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for x in calculated:
        print(x['input_id'],
              'width NORM:',Decimal(x['norm_to_occ_width_reduction_percent_from_displays']).quantize(Decimal('.000001')),
              'width SDP:',Decimal(x['sdp_to_occ_width_reduction_percent_from_displays']).quantize(Decimal('.000001')),
              'time SDP/OCC:',Decimal(x['sdp_over_occ_cold_median_ratio_from_rounded_ms']).quantize(Decimal('.001')))
    print('exact SOS < 1/400, other mode OCC < 1/400: PASS')
```

## 付録B. 算術出力

```json
{
  "kind": "REVIEW_DISPLAY_ARITHMETIC_AND_EXACT_SCALAR_COMPARISON",
  "source_transcription": "display scalars from published CSV/report; not a raw-file export",
  "result_commit": "6d8f769b857e50c8d262e7a4e94f8277a8967597",
  "new_science_runner_calls": 0,
  "new_eigensolver_calls": 0,
  "new_solver_calls": 0,
  "new_certificate_searches": 0,
  "full_PSD_audit_reexecuted": false,
  "rows": [
    {
      "input_id": "V2_banded_5",
      "norm_to_occ_width_reduction_percent_from_displays": "0",
      "sdp_to_occ_width_reduction_percent_from_displays": "0.1174348812989361937558090367756550258200",
      "sdp_over_occ_cold_median_ratio_from_rounded_ms": "2.085347908345595963842758040782005465630",
      "occ_over_norm_cold_median_ratio_from_rounded_ms": "4.69",
      "sos_over_occ_cold_median_ratio_from_rounded_ms": "35.89272951139673864080002402474548785249",
      "occ_raw_width_strictly_below_sos_from_display": true
    },
    {
      "input_id": "V2_banded_6",
      "norm_to_occ_width_reduction_percent_from_displays": "44.68053555736742282466768125914913708960",
      "sdp_to_occ_width_reduction_percent_from_displays": "0.0025312570384784244906225362636252742400",
      "sdp_over_occ_cold_median_ratio_from_rounded_ms": "1.942445917655268667131891137473831123517",
      "occ_over_norm_cold_median_ratio_from_rounded_ms": "4.885366061535839086337680047728628654223",
      "sos_over_occ_cold_median_ratio_from_rounded_ms": "100.7810537334263782274947662247034193999",
      "occ_raw_width_strictly_below_sos_from_display": true
    },
    {
      "input_id": "V2_dense_density_5",
      "norm_to_occ_width_reduction_percent_from_displays": "41.42700995543862451128532697167273152753",
      "sdp_to_occ_width_reduction_percent_from_displays": "0.0051733071005754752258432721876414245200",
      "sdp_over_occ_cold_median_ratio_from_rounded_ms": "1.678918401531466858100023929169657812874",
      "occ_over_norm_cold_median_ratio_from_rounded_ms": "5.872681281618887015177065767284991568297",
      "sos_over_occ_cold_median_ratio_from_rounded_ms": "38.36865278774826513519980856664273749701",
      "occ_raw_width_strictly_below_sos_from_display": true
    },
    {
      "input_id": "V2_dense_density_6",
      "norm_to_occ_width_reduction_percent_from_displays": "5.023696796332253556718485292327467072430",
      "sdp_to_occ_width_reduction_percent_from_displays": "0.0206240383272653968741877203001419582700",
      "sdp_over_occ_cold_median_ratio_from_rounded_ms": "2.367694166573002135551309430144992694167",
      "occ_over_norm_cold_median_ratio_from_rounded_ms": "6.163491513682022861101489435400069276065",
      "sos_over_occ_cold_median_ratio_from_rounded_ms": "91.32454760031471282454760031471282454760",
      "occ_raw_width_strictly_below_sos_from_display": true
    },
    {
      "input_id": "V2_two_external_stress_5",
      "norm_to_occ_width_reduction_percent_from_displays": "32.37649968160852789875702603860690332366",
      "sdp_to_occ_width_reduction_percent_from_displays": "0.0028331044351054774015104148106552855000",
      "sdp_over_occ_cold_median_ratio_from_rounded_ms": "1.604991129509166173861620342992312241277",
      "occ_over_norm_cold_median_ratio_from_rounded_ms": "5.950872747747747747747747747747747747748",
      "sos_over_occ_cold_median_ratio_from_rounded_ms": "28.64648137196924896510940272028385570668",
      "occ_raw_width_strictly_below_sos_from_display": false
    },
    {
      "input_id": "V2_two_external_stress_6",
      "norm_to_occ_width_reduction_percent_from_displays": "1.031109419787198983306594897998040181400",
      "sdp_to_occ_width_reduction_percent_from_displays": "0.0051026750201685401591646761491161700400",
      "sdp_over_occ_cold_median_ratio_from_rounded_ms": "1.752320544554455445544554455445544554455",
      "occ_over_norm_cold_median_ratio_from_rounded_ms": "6.210498733513844003843130404402131190497",
      "sos_over_occ_cold_median_ratio_from_rounded_ms": "57.62467653015301530153015301530153015302",
      "occ_raw_width_strictly_below_sos_from_display": true
    }
  ],
  "cross_source_inference": {
    "input_id": "V2_two_external_stress_5",
    "external": 4,
    "target": [
      1,
      400
    ],
    "saved_sos_width_high": [
      1219549601967953115443785837705603355,
      510423550381407695195061911147652317184
    ],
    "sos_width_below_target": true,
    "strict_margin_target_minus_sos_high": [
      1412731849639153063596723504088185949,
      12760588759535192379876547778691307929600
    ],
    "saved_occ_width_low_above_target": true,
    "ideal_occ_infeasibility": "Source-reported exact refutation at this query; not re-certified by this scalar script.",
    "other_mode_0_OCC_already_certifies_target": true,
    "meaning": "Same-mode certificate-family separation conditional on published proof audit; not extra Hamiltonian reduction."
  },
  "published_certificate_count_total": 1224,
  "published_refutation_record_total": 36,
  "repetitions_are_not_new_inputs": true
}
```
