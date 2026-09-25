# Partially-Randomized-Trotter：事前検証結果のレビューと研究方針の判断

**対象commit：`898da8487aca30a176ce5931d34a923c28c34ac9`**<br>
指定branch：`all-r-coherent-opt2-reoptimization`<br>
対象repository：`HIROMU1015/Partially-Randomized-Trotter`<br>
レビュー日：2026-09-25<br>
対象：カタログ144項目のうち、evidence mapで実施内容が対応付けられた17の検証・判断単位。これは「144項目中17項目だけを実施」という意味ではない。1つの実施単位が複数IDの限定範囲を扱う。

## 要旨

**研究の優先順位と、次に具体化すべき研究上の問いを判断する材料は、十分得られた。一方、部分ランダム化が最適化された決定論PFより有利か、という最終結論は未確定である。**

今回の結果から最も具体的に支持される方向は、**DF回路の基底共有とsupport限定構成を、ランダムなevent列・shot数・PF誤差の制約と合わせて設計すること**である。「精密な資源評価系を完成すること」を単独の研究目的とするより、どの構造が候補選択を変え、その効果を未使用条件で予測できるかを主題とする。

ただし、repositoryの「T4/T7優先、T3保留」という分類を、そのまま最終的な科学判断として採用することは勧めない。T4とT6の組合せには具体的な方法・機構がある。T7の監査性は重要な基盤だが、それだけで独立した研究貢献が確定するわけではない。また、PF係数感度が小さかった結果は限定設計の結果であり、高次PFや最適分割の価値を否定していない。[MAP][WP11][PF][STRUCT][POLICY]

**次の優先事項は、同じH4・同じ候補の誤差棒をさらに縮めることではなく、まず現在の探索境界を確認すること。** 最新の比較は `L_D∈{3,12}`、`δ∈{0.01,0.02}`、`r∈{1,2,4,8,16,32}`、`K=2` に限られ、両候補のδとランダム側の最後3 roundのrが探索上端にある。ここを確認せずに、優位性の有無や「PFが重要でない」を一般化することはできない。[OPT2][OPTCODE]

---

## 1. 今回確認した範囲と、確認していないこと

### 1.1 読み方と照合

指定どおりPROJECT_MAPを入口とし、evidence map、対応する結果文書、カタログの運用規則を読んだ。最新M06-Fについては、集計JSONの主要箇所、audit JSON、探索範囲・grid・区間伝播・fresh-32統合の実装、committed artifactとraw再構成を分けるtestも確認した。下記のリンクはすべて指定commitへ固定してある。[PROJECT][MAP][OPT2JSON][AUDITJSON][OPTCODE][GRIDCODE][TESTFINAL]

このレビューは**文書・保存済み結果・実装の照合と研究上の解釈**であり、全回路の再compile、量子シミュレーションの再実行、全pytestの再実行ではない。後述のδ=0.04の数値は保存された係数による算術診断であり、新しい物理検証結果ではない。

### 1.2 証拠の限界

全ての巨大artifactを末尾まで読み、全fingerprintを独立に再計算したわけではない。各実施単位の条件・status・主結果は専用報告を中心に整理し、最新M06-Fの主要入力・集計・実装を重点照合した。WP04の大型JSONはGitHub内容取得で本文が返らなかったため、その詳細結果は専用報告から確認した。リンクの掲載を、rawデータ全件の独立監査と同義には扱わない。

M06-Fの`checkpoints/`、`tasks/`、`worker_results/`はGit管理外である。committed artifactを検証するtestと、server raw evidenceから再構成するtestが分かれ、後者はrawがないclean checkoutではskipされる。報告値の`567 passed, 2 skipped, 4 warnings`は、そのcheckout相当での記録であり、今回こちらが実行した成績ではない。[MAP][OPT2][TESTFINAL]

この制限は「研究内容が無価値」を意味しない。ただし、外部独立再現や最終論文の再現パッケージが完成したとも意味しない。

---

## 2. 最新結果：何が変わり、何が未確定か

### 2.1 固定範囲

H4直鎖、原子間距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定DF/Hamiltonian snapshot。二次partial-S2、状態準備を除いたcontrolled Hadamard interrogation、主指標はcompiled RZ count。Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 2、seed 17、coupling map・backendなし。[OPT2]

最新比較の目標はCA/10、β_RPE=0.4、α_total=0.05である。これらは当該検証の条件であり、このレビューで追加した必須条件ではない。[DEC][OPT2JSON][OPTCODE]

### 2.2 M06-Fの最適化結果

| L_D | δ | 状態準備なしRZ点推定 | shot総数 |
|---:|---:|---:|---:|
| 3 | 0.01 | 1.568700×10^12 | 12,868 |
| 3 | 0.02 | 1.263314×10^12 | 13,588 |
| 12 | 0.01 | 2.492378×10^12 | 11,548 |
| 12 | 0.02 | 1.327822×10^12 | 11,162 |

両候補ともδ=0.02。L_D=3/12の点比は0.951418、部分ランダム化側が4.858%低い。L_D=3の18 roundのrは`[1×8, 2×2, 4×1, 8×2, 16×2, 32×3]`である。[OPT2][OPT2JSON]

| 不確かさの扱い | L_D=3 | L_D=12 | 判断 |
|---|---:|---:|---|
| local 5%＋較正幅 | [1.1416, 1.3850]×10^12 | [1.2614, 1.3942]×10^12 | 重なる |
| r別の実測discrepancy＋較正幅 | 対応artifactに保存 | 対応artifactに保存 | 重なる |
| 25%移送scenario＋較正幅 | 対応artifactに保存 | 対応artifactに保存 | 重なる |

これらのscenario区間は、統計的な95%信頼区間だけからできているわけではない。較正sampling幅に、実測discrepancyや仮定した5%・25%のmodel感度幅を加えている。**区間重なりは、この扱いでは勝敗を確定できないことを表す。真のcostが同じだという証明ではない。**[OPT2][OPTCODE]

### 2.3 過去の数値と混ぜない

| 段階 | 候補比較の点推定 | 比較上の意味 |
|---|---|---|
| WP01-S | 決定論側が21.0%低い | 初期配分・粗いproxy |
| WP04 | 決定論側が4.96%低い | 両候補へ公平な配分改善 |
| WP06-b / WP05 | 構造変更で部分側へ点順位反転 | initially bridge、その後full wrapper |
| WP01-D/C07 | 部分側が13.916%低い | opt1のfull-scope再最適化 |
| M06/L08 focused | 部分側が8.42%低い | 一部rにopt1を残す感度分析 |
| M06-F coherent opt2 | 部分側が4.858%低い | 今回の最新・単一compiler context内 |

これは同じ固定アルゴリズムのcostが時間とともに揺れたという意味ではない。配分・回路構成・provider・compilerを更新しているため、変更因子を区別して読む必要がある。[PRE][ABL][POLICY][DEC][COMP][OPT2]

---

## 3. 今回の検証から研究上判断できること

### 3.1 「安いランダムevent」を作る問題には、具体的な構造がある

単一Z/ZZでは必要な軌道列だけを保存するGaussian completionが安い一方、同じ元basis内の異なるsupportを連続実行する場合、full basisを共有した方が安くなる例が得られた。単発RZ -61.76% / -39.23%と、長さ3列の+15.01%が同じ検証に存在する。したがって、一律に小さいbasis変換へ置換するだけでは不十分である。[STRUCT]

この現象は、研究の問いを次のように具体化する。

> ランダムに選ばれたDF-conjugated Z/ZZ列について、support限定変換とbasis共有をどの区間で使い分けると、controlled回路の期待costを下げられるか。

固定した`support_run_le_1`が未使用短列でRZ -10.67%、CX -8.13%を示したことは、この問いの初期証拠になる。次は別instanceやrun分布に移したときの予測が必要であり、H4固有の閾値1が普遍的に最適とは主張しない。[POLICY]

### 3.2 統計と回路を別々に最適化すると設計判断を誤ることがある

WP04では、成分作用数を減らすscheduleがcompiled RZを増やす例があり、β・α改善の大きな部分は決定論にも適用できた。[ABL]

したがって研究上の比較は、各手法について同じ目標・scope・最適化機会を与え、

\[
G=\sum_{m,b}N_{m,b}\,\mathbb E[C_{m,b}]
\]

を用いる必要がある。両手法のshot数を同じに固定することが「公平」なのではなく、同じ目標と保証水準の下で、それぞれのcostを最適化することが公平である。

ただし、この原則自体を新規性としない。部分ランダム化をsingle-ancilla位相推定の資源評価へ接続する研究、回路costと推定量varianceを同時に扱うimportance sampling研究は既にある。新規性の候補は、今回のDF列構造・非加法性・control・有限scheduleが、どんな具体的な選択変更を生むかに置く。[Lit1][Lit2]

### 3.3 現H4でのD6係数を、さらに小数点以下まで精密化する優先度は低い

現shortlistのD6対支配固有位相差は最大0.317%であり、初期の候補差・compiler効果より小さい。この限定範囲では、D6を運用上の経験的入力として使い、より重要な不確かさへ資源を配る判断は妥当である。[PF]

しかし、これは次の主張ではない。

- PF errorそのものが総costへ効かない。
- 全L_Dの誤差構造を調べても意味がない。
- 高次化しても有利にならない。
- C_Dを最終係数として使ってよい。

WP03はβとproviderを固定し、係数が変わっても全候補が同じ実行可能領域に残る比較だった。costを変えない構造の中で選択が変わらなかったため、結論はその範囲に留める。[PF][GRIDCODE]

### 3.4 信頼性は貢献を支えるが、「まだ勝敗不明」だけでは弱い

失敗した初回runを保持し、fresh seed追試、scope、compiler、sampling、model discrepancyを分離したことは、比較の信頼性を高めている。[EXT][UNC][OPT2]

ただし、「H4で4.86%の点差が誤差範囲内だった」という事実だけでは、部分ランダム化の一般的限界を示す否定的結果ではない。意味のある限界研究にするなら、どの量が利得を消し、どんな条件なら同じ傾向を事前に予測できるか、また探索範囲を広げても結論が変わらないかを示す必要がある。

---

## 4. 現行の方向判断へ追加すべき、三つの注意

### 4.1 最適化する候補の範囲が、結果を制限している可能性

M06-Fで直接確認したのは、列挙済みのr=1,2,4,8,16,32についてcompiler contextを揃えることだった。「all-r」は全整数rを探索したという意味ではない。δも0.01と0.02だけ、L_Dも3と12だけ、Kも2だけである。[OPT2][OPTCODE]

両者δ=0.02、部分側の最後3 roundがr=32という結果は、**最適化上端が活性になっている可能性を調べる理由**である。境界で選ばれたことだけで外側に良い解があるとは証明できないが、外側を調べずに大域的結論を出すこともできない。

#### δ=0.04は旧固定PF予算だけで除外できない

最新artifactでL_D=12、δ=0.02、q_max=131072の最大経験的PF位相proxyは

\[
\beta_{\rm PF}^{\rm proxy}=0.0140778302555.
\]

同じ最終発展時間を保ち、δを0.04、q_maxを65536へ変える算術診断を行うと、同じ経験モデル \(qC\delta^3\) では

\[
\beta_{\rm PF}^{\rm proxy,new}
=\frac{1}{2}\,2^3\,\beta_{\rm PF}^{\rm proxy}
=0.0563113210220.
\]

tail-freeならRTE予算は0なので、β_RPE=0.4からは

\[
\beta_{\rm stat}^{\rm model}=0.343688678978
\]

が残る。これは正であり、旧固定予算0.02 radを超えたことだけを理由に現在の再最適化から除外するのは適切でない。[OPT2JSON][GRIDCODE]

**この計算はδ=0.04の採用推奨、cost改善の実測、真の実行可能性の証明ではない。** 当該δのPF分枝・信号半径・alias-free条件・係数の妥当性と、回路costを確認する必要がある。ここから言えるのは、δ探索を広げる検証には具体的な根拠がある、ということまでである。

#### δとrは同時に考える必要がある

同じL_D、同じKで、あるroundの \((\delta,q,r)\) を \((2\delta,q/2,2r)\) へ変えれば、

\[
\tau=\lambda_R\delta/r,\qquad R_{\rm short}=qr
\]

は保たれる。従って、既存のfinite-RTE式における当該roundのnormalizationと打切り上界は同じ引数を持つ一方、deterministic sweepの反復数とPF biasは変わる。

これは二つのpartial-S2回路が同じunitaryという意味ではない。PF誤差・RPE前半round・β/α・整数shot・basis境界は再評価が必要である。この構造が、δ/rの共同設計を調べる動機になる。

### 4.2 「PF推定量の選択不変」と「PF設計が不要」は別問題

WP03では、全18条件が同じ固定予算内に入ったのでcost値が同じになった。現在はβを実際の使用量まで締め直しており、係数・δ・β・qの関係は初期診断と異なる。[PF][GRIDCODE]

またcosted prefixは3と12だけである。従って、PF係数の役割を本当に判断するには、少なくとも探索境界近くで係数を変えてδ/βを再最適化することと、他の中間分割を少数試すことが必要になる。

### 4.3 高次の部分ランダム化を開発することと、強いbaselineを置くことは別

高次partial-RTEそのものを今すぐ全面実装する必要はない。しかし、決定論比較が二次endpointだけなら、主張は「決定論二次との比較」に留まる。最適化された決定論PF一般への優位性を主張する前には、少なくとも有力な決定論4次を同じscopeでscreeningする意味がある。[MAP][CAT]

「T3保留」は研究作業の優先順位としては理解できるが、強いdeterministic baselineまで無期限に省略する根拠にはならない。


### 4.4 過去のq=16,32検証を、更新後proxyの検証と混同しない

final fresh-32統合の実装では、`_load_combined_points()`がfresh対象cellの点集合を辞書単位で置換する。このため、`L_D=3,δ=0.02,r=32`について、以前のq=1,2,16,32の集合は、fresh q=1,2,8の集合へ置き換わる。`_diagnostics()`も全rを`HOLDOUT_Q=(8,)`で評価する。testは最終auditの当該cellのq集合が`{1,2,8}`であることを明示的に確認する。[FINALCODE][TESTFINAL]

これは直ちに実装の誤りを意味しない。fresh evidenceを分離して使う意図と整合しており、過去のq=16,32の生の測定値もrepositoryの別artifactに残る。しかし、**「campaignでq=32を測った」と「最新fresh-32較正のproxyがq=32を予測できた」は別の命題**である。

最新モデルへ旧q=16,32の固定測定値をholdoutとして再適用すれば、この間を新規compileなしで照合できる。sampling分布、snapshot、compiler、policy、axisの一致を検査したうえで、更新後の傾き・切片による予測残差を記録する。この確認が済むまでは、最新ランダム側proxyの直接holdout範囲を過去のq=32へ自動的に広げない。

---

## 5. 研究方向の評価と、より具体的な問い

以下はこのレビューによる提案であり、repositoryの方針を自動変更するものではない。

| 方向 | 現段階の扱い | より意味を持つ具体的な問い |
|---|---|---|
| T1：部分ランダム化の優位領域 | 継続。ただし候補限定・経験モデル限定を明記 | 分割・δ/rの探索を広げたとき、どんなtail重み・basis集中度・精度で利得が残るか |
| T2：PF誤差・分割 | 係数の再精密化は低優先。境界感度は維持 | 同じ精度の係数でも、分割によるCとδの選択変化が総costを左右するか |
| T3：高次PF | 新規高次partial実装は保留可。deterministic4次baselineは独立に実施 | 低いPF誤差による大きいδの利益がstage増を上回るか |
| T4：ランダム回路cost予測 | 有力。T6の構造と結び付ける | 短い列の構造から、長いcontrolled回路のcostと候補順位を予測できるか |
| T5：RTE/RPE配分 | 共通最適化基盤として継続 | r・δの境界とround別配分を解消しても、改善がランダム化固有か |
| T6：表現・sampling・回路構造 | WP06の具体的な発見を重点化 | support限定変換とbasis共有の切替規則を未使用分布へ移せるか |
| T7：信頼性・否定的結果 | 全方向の基盤。単独主題には条件付き | どの省略が手法選択を誤らせるかを、再現可能な反例・予測・regretとして示せるか |

### 5.1 最も具体的な主題候補

> DFに基づくランダム化時間発展における、event列構造を利用した回路合成と資源最適化。

この主題なら、部分ランダム化が最終的に決定論に勝つかに依存せず、同じ物理演算を安く実装する方法と、その適用範囲を成果にできる。

ただし、「singleton runだけsupport化する」というH4で選んだ一規則で終わらず、basis run長、supportの和集合、基底間の距離、component確率集中度などから、full/shared/supportのどれを使うべきかを予測する必要がある。[STRUCT][POLICY]

### 5.2 cost予測を独立した貢献へする条件

例えばiidなbasis labelを持つ説明用モデルなら、長さLの列のrun数は

\[
\mathbb E[N_{\rm runs}]
=1+(L-1)\left(1-\sum_b p_b^2\right)
\]

と計算できる。この式はiid labelモデルからの導出であり、現実の全RTE回路がそのままこのモデルだという主張ではない。event内の複数component、identity、support差、compiler境界を追加して、どこで近似が破れるかを調べる。

このように、単なるinstance別の回帰係数表ではなく、**確率分布と回路構造から予測するモデル**へ進むと、T4とT6が一つの説明可能な研究になる。countとdepthの非加法性は異なるため、RZ countで成立した式をdepthへ自動一般化しない。

### 5.3 新規性に関する限定

先行PR研究は部分ランダム化とsingle-ancilla phase estimationの資源評価を既に扱う。resource-optimal importance sampling研究も回路costとvarianceを共同で扱う。したがって「総costを最適化する」「cost-awareにする」という抽象的な説明だけでは新規性の特定にならない。[Lit1][Lit2]

現時点で新規性があり得るのは、DF-conjugated componentのsequence-aware synthesis、basis共有とsupport限定の競合、controlled wrapperを含む予測誤差と選択regretの関係である。これらが既存文献に対して未実施かどうかは、個別の方式・定理・実験条件を対比して確認する必要がある。本レビューは網羅的な新規性証明ではない。

---

## 6. 次に実施する検証：全面再開ではなく、三つの小さな判別

以下のgridは提案であり、新しい研究条件を確定するものではない。全直積を一括実行せず、解析・小系行列で候補を絞り、必要な短回路だけをcompileする。

### 追加A：現在のH4で、探索境界を外す【最優先】

**問い**：4.858%という小さい点差は、物理的な競合を見ているのか、δ/r/prefixの狭い探索範囲の結果なのか。


#### A0：最新proxyと既存q=16,32データの再照合

新規compileに入る前に、4.4節のlineage差を解消する。最新fresh q=1,2のaffine係数で、旧opt2同条件のq=16,32を予測し直す。旧点をfitへ混ぜず、holdoutとして保持する。通過・不通過どちらでも、最新モデルが実際に確認されたdomainを更新する。

**成果物**：old/new model fingerprint、fixed holdout artifact、q別予測値・直接平均・SE・残差、適用可能domain。これはモデル更新後の再照合であり、旧測定を再実施したことにはしない。

#### A1：δ境界

現行δ=0.01,0.02を残し、その上側の少数候補を追加する。例は0.025、0.03、0.04である。ただし一律の等間隔gridより、目標精度からMが切り替わる点の前後を含める方が目的に合う。

最初にsame-snapshot sector行列でPF分枝・実信号・係数の有効性・alias-free条件を確認し、各δでM、β、α、shotを再計算する。旧固定β_PF=0.02の採否を引き継がない。compiler角度不変性も新範囲へ自動移送せず、少数の対応回路で確認する。

**最小成果物**：δ、M、q_max、実PF位相、使用した係数/不確かさ、最良β、shot数、cost点推定、providerの直接/移送domain、探索上端に当たるか。

#### A2：r境界

まずcostを支配する後半roundを対象に、r=64を一段追加する。さらに上端が有力ならr=128へ進む。最初から全r×全q×全δの高統計compileはしない。

finite-RTEの式からattenuation・打切り・shot数を先に評価し、costが改善し得る場合だけ、同一compiler・固定policyのq=1,2較正と未使用qの短回路holdoutを追加する。r>32に現在のcost係数を根拠なく外挿しない。構造比較は同じtrajectoryに対するfull/selectedの対応付き比較とする。

**最小成果物**：roundごとのr候補、attenuation、RTE phase bound、shot、期待1-shot cost、round cost、上端hit flag、追加cost較正のSEとholdout残差。

#### A3：他のprefix

L_D=3と12の間で最適性を判断しない。まず全prefixまたはその粗い部分集合のλ_R、D6/小系固有位相、安価な決定論costを整理する。例としてL_D=2,4,6,9などを候補にするが、どの分割を直接compileするかは、この安価なscreeningから選ぶ。

新L_DではDF tail、sampling確率、basis-run分布も変わるので、LD3のproviderをそのまま使わない。最初は数個の代表分割だけを直接評価し、4.858%差に影響しない枝の高精度化は行わない。

**終了条件**：次のいずれかを明確に報告する。
1. 外側の候補で点選択や利得が大きく変わった。
2. 追加した候補が不適格または明確に不利で、旧shortlistが安定した。
3. なお上端に最良点があり、どの変数だけを追加すべきか特定できた。
4. cost不確かさより候補差が小さく、同H4での優位性確定を保留する。

「最適点が絶対に内部へ入るまで無限に拡張する」という終了条件にはしない。使った有限domainと未探索方向を明記して区切る。

### 追加B：構造policyを未使用条件へ移す【Aと並行可能】

**問い**：WP06の改善はH4の一つのbasis/support列に特有か、再利用可能な合成原理か。

まず`support_run_le_1`の規則を固定し、調整に使っていない条件へblind適用する。最小候補は、同じnで異なるgeometryまたは分布条件を一つ、異なるnの小系を一つである。既存snapshotを再利用できるなら使うが、policy調整に使った入力を未使用holdoutと呼ばない。

比べるものは、full-basis共有、固定policy、そしてtrainingとtestを分けた新しい簡単なrun/support規則である。最初から各instanceでpolicyを再学習すると、汎化性の検証にならない。

測定項目はRZ/CX/depth差、trajectory別の悪化率、run長とsupport数、同値性残差、事前予測の誤差、compile負荷である。全RPE総costを最初から求めなくても、同じ回路意味論のcost差の再現性は調べられる。

**分岐**：
- blindで利益と同値性が保たれる → T4＋T6を主題として具体化する。
- 特定runで失敗するが機構で予測できる → run/support適応選択の研究へ進む。
- full compilerの既存最適化で常に差が消える → 基底回路改良の主張を縮め、別の機構へ移る。

### 追加C：強い決定論baselineを一つ置く【T1の結論前】

**問い**：部分ランダム化の利得は、二次endpointに対してだけか。

同じHamiltonian・入力状態の扱い・RPE目標・control scope・compilerで、有力な決定論4次を一つ追加する。まず小系で誤差と1-step/短反復costを測り、δと配分をbaselineにも最適化する。必要がある場合にだけ、他の高次公式へ広げる。

この検証は「高次partial-RTEを新規開発する」こととは別である。決定論4次が大きく低ければ、T1は部分ランダム化が効く条件の再選定または限界研究へ変える。一方、構造policyが同じrandomized回路を安くするというT4/T6の価値は、baseline勝敗に依存して消えない。

### 今は優先しないもの

同じH4・同じδ/r/LDの下で、1%未満の差を確定するためだけにMC標本を増やすこと。qを32→64→128と延ばすだけで長q一般の正当化が得られると期待すること。H12・広い分子群・状態準備法・noiseを一度に追加すること。144項目すべてのチェックを完了目標にすること。

長qの追加は、探索を広げた後にも有力候補が残り、長qモデルだけが実際に結論を左右する場合に絞る。その場合も、構造的なcost上界や局所境界モデルなど、外挿を説明する根拠と組み合わせる。

### 次回の共有までの最小区切り

まず**A0の既存holdout再照合、A1のδ境界、A2のr境界**までで、次の判断が可能になる。Bのblind transferを並行して一条件進めれば、アルゴリズム優位性と回路合成の価値を切り離して評価できる。A3とCは、T1のより広い優位性を論じる前の確認として位置付ける。

---

## 7. 実施済み検証の対応台帳

以下はevidence mapに掲載された実施単位ごとの対応である。`completed`は当該scopeの終了条件を満たしたという意味であり、各カタログIDの一般的な問いを全条件で解決したという意味ではない。[MAP]

共通条件は、特記しない限りH4直鎖・1.0 Å・STO-3G・8 qubit・DF rank 12・固定snapshot・状態準備なしである。旧報告中の「次の検証」は後続で完了している場合があるため、各節の順番と最新M06-Fを優先して読む。

### 7.01 WP00：比較契約

**カタログID**：A01–A03、B01、B05、B08、N03<br>
**報告status**：`completed_for_representative_H4_contract`

**実行条件**：共通H4条件。候補 L_D=0,3,12。主精度CA/10、CAとCA/100は感度診断。opt1、ordinary controlled、状態準備なしHadamard。

**主要結果**：PF入力と回路cost入力のsnapshot不一致を検出し、同一snapshotからPF入力を再生成した。L_D=3のD6係数変化は相対4.15×10^-9と小さいが、hashが一致しない別入力を混合しなかった。

**結果文書**：[docs/research_direction_prevalidation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_prevalidation.md)

**Artifact**：
- [artifacts/research_direction_prevalidation/2026-09-21/wp00_comparison_contract_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_prevalidation/2026-09-21/wp00_comparison_contract_v1.json)

**Runner**：
- [scripts/run_research_direction_prevalidation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_prevalidation.py)

**Test**：
- [tests/test_research_direction_prevalidation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_prevalidation.py)

**残る未検証範囲・注意**：比較契約の固定は全候補・全精度の最適化や物理検証の完了ではない。L_D=0のPF artifactにはC_Dのnoise-floor由来のoverall_pass=falseが残るが、採用したD6検証とは区別されている。

**このレビューでの意味付け**：後続比較を解釈する基盤として有効。これ自体は新しいアルゴリズム上の優位性を示さない。


### 7.02 WP02：round horizonとcoverage

**カタログID**：G01–G03、G05、E02、F04、M01<br>
**報告status**：`completed_as_coverage_audit`

**実行条件**：β_RPE=0.4、同一snapshotの経験的D6係数。δ=0.01,0.0125,0.02とCA、CA/10、CA/100。

**主要結果**：CA/10ではδ=0.02でM=17、q_max=131072、δ=0.01でM=18、q_max=262144。既存3 schedule・56 sector行列点を再利用できる。CA/100では既存3δが当時の固定PF予算0.02 radを超えた。

**結果文書**：[docs/research_direction_prevalidation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_prevalidation.md)

**Artifact**：
- [artifacts/research_direction_prevalidation/2026-09-21/wp02_round_horizon_coverage_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_prevalidation/2026-09-21/wp02_round_horizon_coverage_v1.json)

**Runner**：
- [scripts/run_research_direction_prevalidation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_prevalidation.py)

**Test**：
- [tests/test_research_direction_prevalidation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_prevalidation.py)

**残る未検証範囲・注意**：CAとCA/100の最適総costは未評価。固定予算での不適格を、後続のβ再最適化後も不適格と自動判定してはいけない。

**このレビューでの意味付け**：短roundの実装検証と、目標精度に必要な長roundを分離できた。必要qを過小評価しないための重要な診断。


### 7.03 WP01-S：条件付きscreening

**カタログID**：C01–C03、E01、F02、H01、Q05<br>
**報告status**：`completed_model_conditional_undetermined`

**実行条件**：L_D=0,3,12、δ=0.01,0.0125,0.02。L_D=3/12のq=1,2 Hadamard costからaffine外挿。opt1。

**主要結果**：L_D=0は拡大探索でも成分作用数proxyがL_D=3の560.6倍。これはcompiled RZ比ではない。L_D=3対12の最良δはともに0.02で、RZ点推定3.0814×10^12対2.4329×10^12。決定論側が21.0%低いが25%移送scenarioでは区間が重なる。

**結果文書**：[docs/research_direction_prevalidation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_prevalidation.md)

**Artifact**：
- [artifacts/research_direction_prevalidation/2026-09-21/wp01s_model_conditional_screening_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_prevalidation/2026-09-21/wp01s_model_conditional_screening_v1.json)

**Runner**：
- [scripts/run_research_direction_prevalidation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_prevalidation.py)

**Test**：
- [tests/test_research_direction_prevalidation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_prevalidation.py)

**残る未検証範囲・注意**：L_D=0のcompiled-cost評価、他prefix、強い高次PF baseline、長q直接回路を含まない。L_D=0もone-bodyは決定論側であり、Hamiltonian全体の完全ランダム化と同義ではない。

**このレビューでの意味付け**：大きく不利そうな候補の安価なscreeningとして有効。最終的な『部分ランダム化が不利』という結論ではない。


### 7.04 WP04：配分・schedule・cost目的関数のablation

**カタログID**：F03–F05、F08、G04、H02–H06<br>
**報告status**：`completed_model_conditional_ablation`

**実行条件**：L_D=3,12、δ=0.02、CA/10。固定/round別schedule、複数β、uniform/weighted α、成分作用数/compiled RZの42 factorial条件。opt1のq=1,2 proxy。

**主要結果**：β再配分が両候補のcostを約59%低下させ、その後のα再配分も両候補に効いた。完全設定でcompiled RZに合わせたround scheduleの追加利得は固定schedule比1.93%。成分作用数で選ぶと同条件のRZが16.87%悪化する場合があった。公平な配分後の決定論点優位は4.96%まで縮小し、区間は重なる。

**結果文書**：[docs/research_direction_ablation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_ablation.md)

**Artifact**：
- [artifacts/research_direction_ablation/2026-09-21/wp04_finite_rte_statistical_ablation_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_ablation/2026-09-21/wp04_finite_rte_statistical_ablation_v1.json)
- [artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json)

**Runner**：
- [scripts/run_research_direction_ablation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_ablation.py)

**Test**：
- [tests/test_research_direction_ablation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_ablation.py)

**残る未検証範囲・注意**：上記の寄与率は当時のproviderと有限候補集合に依存し、最新opt2での不変な寄与率ではない。逐次削減率は交互作用があるため加算しない。厳密二項counterfactualは既知の真の信号を使うため、そのまま実用shot数にしない。

**このレビューでの意味付け**：大きな総改善をランダム化固有の効果に帰属させないこと、目的関数に整合した設計が必要なことを示す。


### 7.05 WP03：PF係数選択感度

**カタログID**：D01–D05、E01–E03、I01、Q02<br>
**報告status**：`completed_selection_invariant_intervals_overlap`

**実行条件**：係数監査はL_D=0,3,12、cost比較は3,12だけ。3係数×2候補×3δの18条件。WP04のβとcost providerを固定。

**主要結果**：L_D=3のC_DはD6を12.50%過小評価し、L_D=0ではC_D=0でもfull partial係数は非零。D6対支配固有位相の差はcost対象候補で最大0.317%。全18条件が固定予算内に残り、全係数policyでL_D=12,δ=0.02が選ばれた。

**結果文書**：[docs/research_direction_pf_sensitivity.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_pf_sensitivity.md)

**Artifact**：
- [artifacts/research_direction_pf_sensitivity/2026-09-22/wp03_pf_coefficient_selection_sensitivity_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_pf_sensitivity/2026-09-22/wp03_pf_coefficient_selection_sensitivity_v1.json)

**Runner**：
- [scripts/run_research_direction_pf_sensitivity.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_pf_sensitivity.py)

**Test**：
- [tests/test_research_direction_pf_sensitivity.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_pf_sensitivity.py)

**残る未検証範囲・注意**：全prefixでの候補選択は未評価。固定予算内ではCがcostを直接動かさない設計なので、選択不変からPF誤差の研究一般を否定できない。D6と他推定量のsigned biasは規約が揃わず、相殺の物理的証拠にしない。

**このレビューでの意味付け**：現shortlistのD6精度は実用的だった。一方、δやβを再最適化する条件・PF制約境界・別分割については再判定が必要。


### 7.06 Gate S1：最初の研究方向統合

**カタログID**：WP00、WP02、WP01-S、WP04、WP03の統合<br>
**報告status**：`completed_direction_synthesis`

**実行条件**：既存fingerprint済みartifactの統合。新規の物理計算・回路compileなし。

**主要結果**：区間重なりを『同点』ではなくundeterminedとし、最大の残存要因をfull controlled interrogationのscope・構造と判断。T4/T7を優先し、WP06-aへ進んだ。

**結果文書**：[docs/research_direction_gate_s1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_gate_s1.md)

**Artifact**：
- [artifacts/research_direction_gate_s1/2026-09-22/gate_s1_research_direction_decision_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_gate_s1/2026-09-22/gate_s1_research_direction_decision_v1.json)

**Runner**：
- [scripts/run_research_direction_gate_s1.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_gate_s1.py)

**Test**：
- [tests/test_research_direction_gate_s1.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_gate_s1.py)

**残る未検証範囲・注意**：自動生成された方向分類は、独立した新規性確認や科学的優位性の証拠ではない。

**このレビューでの意味付け**：方針を小さな検証に戻した点は有効。ただしT3保留等は当時の作業順であり、高次PFの価値を検証した結論ではない。


### 7.07 WP06-a：回路構造pilot

**カタログID**：L02–L04、J04、J05<br>
**報告status**：`completed_triggered_focused_followup`

**実行条件**：共通H4、L_D=3、δ=0.02、opt1。最大係数の代表Z/ZZ単発、同一basisでsupportの異なる短列。synthetic n=4,6,8も補助診断。

**主要結果**：support限定Gaussian completionでcontrolled ZのRZが204→78、ZZが311→189。一方、異なるZZ supportの長さ3列ではfull basis共有353に対しsupport別406で15.01%悪化。controlled relative phaseを含む最大operator残差4.73×10^-15。

**結果文書**：[docs/research_direction_structure_pilot.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_structure_pilot.md)

**Artifact**：
- [artifacts/research_direction_structure_pilot/2026-09-22/wp06a_circuit_structure_pilot_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_structure_pilot/2026-09-22/wp06a_circuit_structure_pilot_v1.json)

**Runner**：
- [scripts/run_research_direction_structure_pilot.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_structure_pilot.py)

**Test**：
- [tests/test_research_direction_structure_pilot.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_structure_pilot.py)

**残る未検証範囲・注意**：単発削減率を全trajectoryや総costへ流用しない。syntheticサイズ診断はchemistry scalingの証拠ではない。基底変換を非制御にする改善は既存実装であり、新規改善へ二重計上しない。

**このレビューでの意味付け**：『単発を安くすること』と『同じ基底を共有すること』の競合という、機構を伴う研究課題を示した。


### 7.08 WP06-b：sequence-aware basis policy

**カタログID**：L02–L04、J04、J05のfocused follow-up<br>
**報告status**：`completed_sequence_policy_holdout`

**実行条件**：opt1。training長1,2,4各8列、独立holdout長3,6各12列。物理finite-RTE分布。policyはtrainingだけで固定。

**主要結果**：元basisのsingleton runだけsupport限定にするsupport_run_le_1を採用。holdoutのRZ -10.67%、CX -8.13%、total depth -2.25%。controlled同値性残差1.34×10^-15。中央RTE差だけを使うbridgeでL_D=3/12の点順位が反転したが区間は重なる。

**結果文書**：[docs/research_direction_sequence_policy.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_sequence_policy.md)

**Artifact**：
- [artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json)

**Runner**：
- [scripts/run_research_direction_sequence_policy.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_sequence_policy.py)

**Test**：
- [tests/test_research_direction_sequence_policy.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_sequence_policy.py)

**残る未検証範囲・注意**：H4以外へのpolicy汎化、別compilerでの再学習との比較、より良いrun分割規則は未評価。bridgeの順位反転はfull algorithmの勝利ではない。

**このレビューでの意味付け**：現時点で最も具体的な新しい方法候補。固定規則のblind transferと、どんなrun分布で効くかの説明へ進める。


### 7.09 WP05-a：full controlled interrogation接続

**カタログID**：L01、L06、L07、M01、M02、M05、N04<br>
**報告status**：`completed_full_scope_q4_holdout`

**実行条件**：opt1、δ=0.02、r=1,2,4,8,16,32、K=2。q=1,2較正、q=4独立holdout。full/selectedと両軸、状態準備なし。

**主要結果**：576本のrandomized wrapperと6本のdeterministic wrapperをcompile。selected RZ holdout最大2.288%、全metric2.431%、中央additive bridgeのfull-wrapper残差2.625%。同じWP04 planでselected policyはfull basisより9.94%低い点推定。

**結果文書**：[docs/research_direction_full_scope.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_full_scope.md)

**Artifact**：
- [artifacts/research_direction_full_scope/2026-09-22/wp05a_full_controlled_interrogation_connection_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_full_scope/2026-09-22/wp05a_full_controlled_interrogation_connection_v1.json)

**Runner**：
- [scripts/run_research_direction_full_scope.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_full_scope.py)

**Test**：
- [tests/test_research_direction_full_scope.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_full_scope.py)

**残る未検証範囲・注意**：H4のstate-action同値性診断は単一固定複素状態に対するもの。q>4の一体compile、shot再最適化、実backendは当該検証に含まれない。

**このレビューでの意味付け**：局所構造改善がwrapper接続で消えないことを確認。アルゴリズム全体の優位性とは分離する。


### 7.10 WP05-b / WP05-bR：q=8・δ拡張と独立追試

**カタログID**：WP05-aと同じIDの拡張<br>
**報告status**：`completed_after_focused_replication`

**実行条件**：opt1。δ=0.02のfresh q=8とδ=0.01のq=1,2,4,8、各8 trajectory。δ=0.02,r=32はfresh32 trajectoryを追加。

**主要結果**：初回δ=0.02,r=32,q=8のselected RZ誤差5.084%でtrigger。その後のfresh32追試ではselected RZ0.516%、全metric0.537%、full basis RZ0.829%。最初の逸脱を履歴に保持して追試した。

**結果文書**：[docs/research_direction_full_scope_extension.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_full_scope_extension.md)

**Artifact**：
- [artifacts/research_direction_full_scope_extension/2026-09-22/wp05b_q8_delta_0p01_full_scope_extension_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_full_scope_extension/2026-09-22/wp05b_q8_delta_0p01_full_scope_extension_v1.json)
- [artifacts/research_direction_full_scope_replication/2026-09-22/wp05br_r32_32trajectory_replication_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_full_scope_replication/2026-09-22/wp05br_r32_32trajectory_replication_v1.json)

**Runner**：
- [scripts/run_research_direction_full_scope_extension.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_full_scope_extension.py)
- [scripts/run_research_direction_full_scope_replication.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_full_scope_replication.py)

**Test**：
- [tests/test_research_direction_full_scope_extension.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_full_scope_extension.py)
- [tests/test_research_direction_full_scope_replication.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_full_scope_replication.py)

**残る未検証範囲・注意**：追試での通過は全qでの線形性の証明ではない。同じ形式をfresh平均で再較正したことと、元の数値係数を固定したことを混同しない。

**このレビューでの意味付け**：小標本の初回逸脱を、構造的非線形性と即断しなかった点を支持する。


### 7.11 WP01-D / C07：full-scope候補再最適化

**カタログID**：C01–C03、C07、E01、F02、H01、Q05<br>
**報告status**：`completed_local_conditional_robust_undetermined`

**実行条件**：opt1、L_D=3/12、δ=0.01/0.02、r<=32、K=2。β粗細grid＋上位解の境界refinement、α、shot、round scheduleを候補別に更新。

**主要結果**：両候補δ=0.02。L_D=3は1.4557921×10^12 RZ・13538 shot、12は1.6911234×10^12 RZ・11162 shot。点差13.916%。5% local区間は僅かに分離したが、分離を維持できる対称discrepancy上限5.0484%で余裕が小さい。25%scenarioは重なる。

**結果文書**：[docs/research_direction_decision_cost.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_decision_cost.md)

**Artifact**：
- [artifacts/research_direction_decision_cost/2026-09-22/wp01d_c07_full_scope_optimization_compute_v2.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_decision_cost/2026-09-22/wp01d_c07_full_scope_optimization_compute_v2.json)
- [artifacts/research_direction_decision_cost/2026-09-22/wp01d_c07_conditional_interval_synthesis_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_decision_cost/2026-09-22/wp01d_c07_conditional_interval_synthesis_v1.json)

**Runner**：
- [scripts/run_research_direction_decision_cost.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_decision_cost.py)
- [scripts/run_research_direction_decision_synthesis.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_decision_synthesis.py)

**Test**：
- [tests/test_research_direction_decision_cost.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_decision_cost.py)
- [tests/test_research_direction_decision_synthesis.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_decision_synthesis.py)

**残る未検証範囲・注意**：比較対象・δ・r・Kは有限集合で、global optimumではない。v1 computeは旧grid下端依存の予備診断であり、現行判断に使わない。

**このレビューでの意味付け**：配分と構造を同時に反映すると点順位が変わる。13.916%を最新opt2の結果として引用しない。


### 7.12 G08：round別支配度

**カタログID**：G08<br>
**報告status**：`completed_round_dominance_audit`

**実行条件**：WP01-D/C07 v2結果の再集計のみ。新規compileなし。

**主要結果**：L_D=3の最後3 roundがcost90.94%、較正不確かさ87.70%を占め、全てr=32。最大cost/PF-riskはround17、finite-RTE risk最大はround7。deterministicの最後3 round costは83.03%。

**結果文書**：[docs/research_direction_late_round_proxy.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_late_round_proxy.md)

**Artifact**：
- [artifacts/research_direction_round_dominance/2026-09-22/g08_round_cost_risk_proxy_dominance_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_round_dominance/2026-09-22/g08_round_cost_risk_proxy_dominance_v1.json)

**Runner**：
- [scripts/run_research_direction_round_dominance.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_round_dominance.py)

**Test**：
- [tests/test_research_direction_round_dominance.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_round_dominance.py)

**残る未検証範囲・注意**：この比率は当時のopt1 planの値。最新planでも支配度を再計算して使う。cost支配点とerror支配点を同一視しない。

**このレビューでの意味付け**：精密化の対象をcostの大きいroundへ絞る根拠を与える。一方、r上限に張り付く影響を別途確認する必要がある。


### 7.13 M08：q=16,32 holdout

**カタログID**：M08<br>
**報告status**：`completed_q16_q32_holdout_and_reaggregation`

**実行条件**：opt1、δ=0.02,r=32,K=2、q=16,32各8 fresh trajectory、両policy・両軸。

**主要結果**：64本のwrapperを直接compile。selected RZ最大誤差2.466%、全metric2.569%、full basis RZ3.286%、direct RZ relative SE1.353%。実測幅scenarioでは再集計区間が分離したが25%移送幅では重なる。

**結果文書**：[docs/research_direction_late_round_proxy.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_late_round_proxy.md)

**Artifact**：
- [artifacts/research_direction_proxy_precision/2026-09-22/m08_late_round_q16_q32_proxy_precision_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_proxy_precision/2026-09-22/m08_late_round_q16_q32_proxy_precision_v1.json)
- [artifacts/research_direction_proxy_precision/2026-09-22/wp01d_c07_m08_measured_discrepancy_reaggregation_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_proxy_precision/2026-09-22/wp01d_c07_m08_measured_discrepancy_reaggregation_v1.json)

**Runner**：
- [scripts/run_research_direction_proxy_precision.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_proxy_precision.py)
- [scripts/run_research_direction_m08_reaggregation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_m08_reaggregation.py)

**Test**：
- [tests/test_research_direction_proxy_precision.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_proxy_precision.py)
- [tests/test_research_direction_m08_reaggregation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_m08_reaggregation.py)

**残る未検証範囲・注意**：q_max=131072の直接costではない。L_D=3で測ったdiscrepancyを両候補へ共通適用した再集計はscenarioであり、deterministicの追加測定ではない。

**このレビューでの意味付け**：短qのproxy adequacyは支持された。ただし4096倍先のqまで同じ相対精度を保証するものではない。


### 7.14 M06 / L08：compiler感度

**カタログID**：M06、L08<br>
**報告status**：`completed_focused_compiler_transfer_robust_undetermined`

**実行条件**：同じphysical trajectoryをopt1→opt2へ変更。δ=0.02、L_D=3,r=32とL_D=12。q=1,2,16,32。

**主要結果**：opt2 selected RZ holdout誤差2.340%、全metric2.470%。selected RZはopt1比約18.2%減、deterministic再集計は21.48%減。r<32をopt1のまま残したfocused比較では点利得8.42%、実測幅区間は重なった。

**結果文書**：[docs/research_direction_compiler_transfer.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_compiler_transfer.md)

**Artifact**：
- [artifacts/research_direction_compiler_transfer/2026-09-23/m06_l08_opt2_same_trajectory_compute_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_compiler_transfer/2026-09-23/m06_l08_opt2_same_trajectory_compute_v1.json)
- [artifacts/research_direction_compiler_transfer/2026-09-23/m06_l08_opt2_focused_analysis_reaggregation_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_compiler_transfer/2026-09-23/m06_l08_opt2_focused_analysis_reaggregation_v1.json)

**Runner**：
- [scripts/run_research_direction_compiler_transfer_compute.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_compiler_transfer_compute.py)
- [scripts/run_research_direction_compiler_transfer_analysis.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_compiler_transfer_analysis.py)

**Test**：
- [tests/test_research_direction_compiler_transfer_compute.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_compiler_transfer_compute.py)
- [tests/test_research_direction_compiler_transfer_analysis.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_compiler_transfer_analysis.py)

**残る未検証範囲・注意**：focused結果は混在contextであり、opt2完全比較ではない。この欠落は後続M06-Fが埋めた。平均削減比を未測定rへ移した結果はcounterfactual。

**このレビューでの意味付け**：compiler改善が両候補に同じ割合で効くわけではない。固定compiler下の限定結果まで無効にする必要はないが、一般的な優位性には頑健性確認が要る。


### 7.15 N07 / P03：不確かさ台帳・準備cost

**カタログID**：N07、P03<br>
**報告status**：`completed_uncertainty_ledger_break_even_robust_undetermined`

**実行条件**：既存opt1、opt2 focused等の再集計。共通準備cost Pまたは候補別P3,P12をRZ相当/shotでパラメータ化。

**主要結果**：sampling、model discrepancy、compiler、長q、state preparation、external transferを分離。歴史的固定planではL_D=3のshotが2376多く、共通Pの点break-evenはopt1約9905万、opt2 focused約4707万RZ相当/shot。後続M06-Fでshot差と交点は更新された。

**結果文書**：[docs/research_direction_uncertainty_break_even.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_uncertainty_break_even.md)

**Artifact**：
- [artifacts/research_direction_uncertainty_break_even/2026-09-23/n07_p03_uncertainty_break_even_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_uncertainty_break_even/2026-09-23/n07_p03_uncertainty_break_even_v1.json)

**Runner**：
- [scripts/run_research_direction_uncertainty_break_even.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_uncertainty_break_even.py)

**Test**：
- [tests/test_research_direction_uncertainty_break_even.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_uncertainty_break_even.py)

**残る未検証範囲・注意**：準備回路は測定していない。固定planの直線交点であり、Pごとにscheduleと配分を再最適化した最終境界ではない。

**このレビューでの意味付け**：状態準備を無条件に相殺しない扱いは妥当。現段階で準備法全体を研究へ追加する必要はない。


### 7.16 WP11：限定的な方向判断統合

**カタログID**：A04–A08、N08、P03、P08、Q08<br>
**報告status**：`completed_scoped_direction_synthesis`

**実行条件**：11個のfingerprint済みartifactを統合。新規compileやstatevector計算なし。

**主要結果**：T4/T7を主軸、T1をH4条件付き限界へ範囲変更、T2/T5/T6限定、T3保留。次にall-r coherent opt2を一件選び、それでも区間が重なれば局所compiler精密化を止める方針を設定した。

**結果文書**：[docs/research_direction_wp11_synthesis.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_wp11_synthesis.md)

**Artifact**：
- [artifacts/research_direction_wp11_synthesis/2026-09-23/wp11_scoped_direction_synthesis_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_wp11_synthesis/2026-09-23/wp11_scoped_direction_synthesis_v1.json)

**Runner**：
- [scripts/run_research_direction_wp11_synthesis.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_wp11_synthesis.py)

**Test**：
- [tests/test_research_direction_wp11_synthesis.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_wp11_synthesis.py)

**残る未検証範囲・注意**：新規性の文献レビューや他研究方向の棄却ではない。未実施の高次PF・外部instanceを失敗と扱わない。

**このレビューでの意味付け**：終了条件を置いたことを支持する。ただし次の研究課題はT番号だけでなく、機構・予測・比較の具体的な問いへ落とす必要がある。


### 7.17 M06-F：all-r coherent opt2再最適化

**カタログID**：主にM06/L08。G08/M08/N07/P03の判断入力も更新<br>
**報告status**：`completed_all_r_coherent_opt2_reoptimization / coherent_opt2_reoptimization_complete`

**実行条件**：共通H4、opt2、L_D=3,12、δ=0.01,0.02、r=1,2,4,8,16,32、K=2。q=1,2較正・主holdout q=8。既存r=32のq=16,32測定は過去の別artifactに残るが、final fresh-32 cell集合はq=1,2,8へ置換される。support_run_le_1固定。

**主要結果**：初期36 task・1062 transpileとfresh32の15 task・1920 transpileを完了。最終51/51、失敗0。direct RZ relative SE最大1.9844%、selected RZ holdout最大4.4898%、全metric4.7488%。最良点はLD3=1.263314×10^12、LD12=1.327822×10^12でLD3が4.858%低いが、全scenario区間は重なる。shot13588対11162。共通Pの点交点26590335 RZ相当/shot。

**結果文書**：[docs/research_direction_full_opt2.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_full_opt2.md)

**Artifact**：
- [artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_fresh32_audit_20260925_065827.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_fresh32_audit_20260925_065827.json)
- [artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json)

**Runner**：
- [scripts/run_research_direction_full_opt2_compute.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_full_opt2_compute.py)
- [scripts/run_research_direction_full_opt2_analysis.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_full_opt2_analysis.py)
- [scripts/run_research_direction_full_opt2_completion.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_full_opt2_completion.py)
- [scripts/run_research_direction_full_opt2_extension_analysis.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/scripts/run_research_direction_full_opt2_extension_analysis.py)

**Test**：
- [tests/test_research_direction_full_opt2.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_full_opt2.py)
- [tests/test_research_direction_full_opt2_completion.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_full_opt2_completion.py)
- [tests/test_research_direction_full_opt2_extension_analysis.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_full_opt2_extension_analysis.py)

**残る未検証範囲・注意**：all-rは列挙済みr<=32を指す。全整数r、全δ、全prefix、高次PFを最適化していない。旧q=16,32の結果は最新fresh較正で再照合してからmodel domainへ接続する。q>32直接回路、他instance、状態準備、backend/noiseは未実施。checkpoints/tasks/worker_resultsはGit管理外で、raw再構成testはclean checkoutではskip。

**このレビューでの意味付け**：compiler混在を解消した点は重要。小さな点利得の有無を同一H4で追い続けるより、探索境界と構造policyの一般性を確認すべき段階。

---

## 8. 未実施の枝と、してはいけない一般化

| 対象 | 指定commitでの位置付け |
|---|---|
| WP07：K01–K03、K06 | work packageとして未実施。高次PFの不利益を実証したわけではない |
| WP08：R01、F07、N01 | work packageとして未実施 |
| WP09：I02–I06、J01–J03 | work packageとして未実施。全分割・表現の最適性は未確定 |
| WP10：N03、N05、O01、O05等 | work packageとして未実施。外部instanceへの現policyの適用は未確定 |
| H12 | 延期。H4/H6からの無条件外挿はしない |
| q>32のopt2直接holdout | 未実施。主要target q=131072とは別domain |
| 状態準備回路 | 未測定。Pによる感度だけ |
| backend、coupling、noise、FT synthesis | 現campaignの対象外 |
| 新規性 | 関連文献の存在を確認した段階。独自性の網羅的判定は未完了 |
| 最終科学的優位性 | 未確定。ただし方法論・構造に関する限定結果は得られている |

以上はevidence mapのscopeに基づく。coverage外は「失敗」ではなく「未評価」である。[MAP]

---

## 9. 次回の結果共有フォーマット

全ログを最初から読む必要がないよう、結果文書に次の表を置く。raw artifact、生成commit、snapshotとsource hashは別途保持する。

| instance / snapshot | p | L_D | δ | M / q_max | r,K schedule | β / α方式 | shot | 1-shot provider / domain | 総cost点 | sampling幅 | model感度幅 | 境界hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

加えて、今回動かした要因を一つずつ記載する。例えば「r上限だけを32→64」「δの上側だけを追加」「policyは固定して未使用instanceへ移送」である。同時に多要因を変えた最終bestだけでは、研究上の機構が分からなくなる。

各候補のstatusは、少なくとも次を分ける。
- 数学・物理的に不適格。
- 実装domainがないため未評価。
- 評価済みだがcost不確かさで未判定。
- このmodel/scopeでは他候補より不利。
- このmodel/scopeでは有力。

同じq方向のfitから複数roundのcostを求めた場合、その共有較正誤差を独立roundとして平均しない。compilerごとの結果は別に報告する。point estimateを選ぶために使ったdataと、最終的な選択を評価するholdoutを区別する。[UNC][OPTCODE]

---

## 10. 最終判断

**現在の研究を続ける意味はある。ただし、『H4で部分ランダム化が数%有利と確定するまで同じ条件を精密化する』ことを中心にすべきではない。**

今回の検証は、研究方向を選ぶための役割を果たしている。とくに、単発のsupport限定構成と長いbasis共有の競合、目的関数の不一致によるschedule選択の悪化、compilerで候補差が変わることは、実装・モデルの選び方が科学的判断へ影響する具体例である。[ABL][STRUCT][POLICY][COMP][OPT2]

今後の主題は、**何が利得を作るのかを説明し、未使用条件の回路costや選択を予測すること**へ進めるとよい。同時に、アルゴリズム優位性を判断するためにはδ/r/prefixの探索境界と、強いdeterministic baselineを確認する。

ここまでを踏まえた優先順は、**探索境界の安価な確認 → 構造policyのblind transfer → 必要なbaselineと候補の直接cost評価 → 結論を左右する部分だけ高精度化**である。一つの研究方向へ固定する必要はない。

---

## 出典

repository内リンクはすべて指定commitに固定している。本文の方針評価・追加A/B/C・説明用モデルは、このレビューの提案または導出であり、repositoryで既に実施された結果ではない。

- [MAP] `docs/research/prevalidation_catalog_evidence_map.md`
- [PROJECT] `PROJECT_MAP.md`
- [CAT] `partial_randomized_trotter_prevalidation_catalog.md`
- [S1] `docs/research_direction_gate_s1.md`
- [PRE] `docs/research_direction_prevalidation.md`
- [ABL] `docs/research_direction_ablation.md`
- [PF] `docs/research_direction_pf_sensitivity.md`
- [STRUCT] `docs/research_direction_structure_pilot.md`
- [POLICY] `docs/research_direction_sequence_policy.md`
- [FULL] `docs/research_direction_full_scope.md`
- [EXT] `docs/research_direction_full_scope_extension.md`
- [DEC] `docs/research_direction_decision_cost.md`
- [LATE] `docs/research_direction_late_round_proxy.md`
- [COMP] `docs/research_direction_compiler_transfer.md`
- [UNC] `docs/research_direction_uncertainty_break_even.md`
- [WP11] `docs/research_direction_wp11_synthesis.md`
- [OPT2] `docs/research_direction_full_opt2.md`
- [OPT2JSON] `artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json`
- [AUDITJSON] `artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_fresh32_audit_20260925_065827.json`
- [OPTCODE] `src/trotterlib/research_direction_full_opt2.py`
- [GRIDCODE] `src/trotterlib/research_direction_decision_cost.py`
- [TESTFINAL] `tests/test_research_direction_full_opt2_extension_analysis.py`
- [Lit1] Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647v2。部分ランダム化とsingle-ancilla資源評価の既存研究。
- [Lit2] Cugini, Atif, Subasi, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1。回路costと推定量varianceを共同最適化する既存研究。

- [FINALCODE] `src/trotterlib/research_direction_full_opt2_extension_analysis.py` — fresh-32のcell置換と最新q=8 diagnostics。

[MAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research/prevalidation_catalog_evidence_map.md
[PROJECT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/PROJECT_MAP.md
[CAT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/partial_randomized_trotter_prevalidation_catalog.md
[S1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_gate_s1.md
[PRE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_prevalidation.md
[ABL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_ablation.md
[PF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_pf_sensitivity.md
[STRUCT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_structure_pilot.md
[POLICY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_sequence_policy.md
[FULL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_full_scope.md
[EXT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_full_scope_extension.md
[DEC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_decision_cost.md
[LATE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_late_round_proxy.md
[COMP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_compiler_transfer.md
[UNC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_uncertainty_break_even.md
[WP11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_wp11_synthesis.md
[OPT2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/docs/research_direction_full_opt2.md
[OPT2JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json
[AUDITJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_fresh32_audit_20260925_065827.json
[OPTCODE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/src/trotterlib/research_direction_full_opt2.py
[GRIDCODE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/src/trotterlib/research_direction_decision_cost.py
[TESTFINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/tests/test_research_direction_full_opt2_extension_analysis.py
[Lit1]: https://arxiv.org/abs/2503.05647v2
[Lit2]: https://arxiv.org/abs/2603.13495v1

[FINALCODE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/898da8487aca30a176ce5931d34a923c28c34ac9/src/trotterlib/research_direction_full_opt2_extension_analysis.py
