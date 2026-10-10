# Track B G9 v2 科学的研究レビュー
## 明示native比較の意味、CTS samplingに対する条件付き下界、低次数特殊化と一般生成法の研究上の役割

- 作成日：2026-10-10 JST
- レビュー開始承認：利用者の「レビューを開始して」
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 対象branch：`track-b-g9-v2-one-shot-execution-20261010`
- 対象結果commit：`c95736fd2990f5ef6dd1cb5866421fd78bb28687`
- 固定source S：`0ef2b92738750a3c0d187743dd4b7c9b927db802`
- authorization-only / 実行HEAD A：`5ed597d8163af002d315a9e2b9eee0e9f14edce2`
- 原結果：`G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`。本レビューはこのstatus、旧v1失敗、各markerを変更しない。
- レビュー判断：**return集約familyを優先候補として限定継続。closed P5を低次数の標準実装として位置付け、一般生成器の役割を同一構造の次数比較で判別する。一般優位・独立新規性・投稿十分性は未確定。**
- 本書の追加下界はGPT側のsource由来の数学・算術検討。G9で事前登録／独立認証された結果ではない。

## 1. 結論と変更点

G9は、単なる環境・回路接続テストではない。明示された3-system-qubit providerで、同じ有限P5 first operator momentを実現する複数の方法について、direct native回路、相対位相、合成誤差、十分shot予算まで接続した。その登録比較ではclosed P5の期待T切片と準備呼出し数が最小であり、literal matched CTSを含めても有効な点が残る。[R1–R5]

したがって、G8時点の限定継続を維持しつつ、根拠は強まった。抽象的なprovider価格の符号だけでなく、具体的なClifford+T provider上で結果が得られたためである。ただし、実量子測定・誤り訂正装置・実分子の実験ではなく、小さいsynthetic providerに対する論理資源予測である。

今回の追加検討は次の三点である。

1. 約29.77%というT差はcanonical law同士の値である。CTSのT=0 eventを残した任意proposalに対しても、固定CTS辞書・保存合成列・同じBernstein policyなら、closed P5より高い保守的下界が得られる。したがって、今回のCTSとの差をsampling未最適化だけに帰することはできない。ただし29.77%を任意proposalに対する削減率とはしない。
2. closed P5とgeneral local fullは同じ理想ensembleで、一回路当たり平均Tも一致している。約0.639%の差は、normalizerの取得、zero-fill、十分予算の違いで説明される。同familyの実装上の役割分担として整理する。
3. CTSのnative CXは小さいが、CTSは準備呼出しが多い。共通準備CXまで含めた順位は、その一回当たり価格に依存する。T/CX・期待量/上限・古典/量子の指標を混同しない。

現在の主張は、全既知手法に対する最適性ではない。新規性を確定するには、既知のreturn吸収・RTE・Green関数・CTS等と、本方法全体の入力・出力・複雑性・保証の差を明確にする必要がある。

## 2. レビュー資料と確認の深さ

### 2.1 使用した正本

主要handoff、契約、source manifest、nativeおよびcomparison source、保存監査、原結果blob、旧v1失敗資料、前回G8レビューを参照した。原結果は約11.34 MBであり、通常のContents取得では本文が空になる一方、Git blobから取得可能である。これは未pushや欠損ではない。

原結果SHA256：
`3eb8430014a7881b5189ca8779a58ccd7ab2f50c37a58795bd0458b174abda20`

原結果blob SHA：
`73a27fa783f9bf7f841abf5d555cdf79e33d4c99`

marker SHA256：
`5f895ad90418d9a662093724fc1348921dba22d5da95b5413a94ae8f8bde9dc4`

レビューでは11.34 MBの全event証明書を一件ずつ別実装で再認証していない。個別sequenceのstrict matrix guardも再実行していない。原結果・保存監査・sourceの整合を確認し、中心判断に必要な式と固定入力のscalar代数を独立に検討した。

### 2.2 旧v1をどう扱うか

G9 v1は`Fraction`をmpmathへ渡したAPI境界のTypeErrorで、registered比較0 rowの`G9_TECHNICAL_INCONCLUSIVE`だった。v2は独立のsource修正・authorization・markerを持ち、v1を削除・再分類していない。[R6]

従って「v2はrun1/retry0」と「G9の開発過程で技術失敗がなかった」は異なる。前者を採用し、後者は述べない。失敗は優位性の反例でも、不利な科学結果の選別でもないことを、0 rowという到達範囲から区別する。

### 2.3 本レビューで新しく実行したこと

実行したのは、明示QのPauli係数からの有限P5代数、CTS coefficient midpointのsource-compatible再構成、平方根の有理区間、保守的対数不等式、保存表示値の費用因数分解である。

実行していないもの：旧runner、native合成、新入力・provider、回路またはHamiltonian行列実行、量子測定、LP、分子/DF、repository変更。再現用Pythonはrepository moduleをimportしない。数値表示にmpmathを使うが、重要な粗い下界の符号はFraction/isqrtで検査した。

## 3. 固定された科学task

\[
R=\sum_{i=0}^2p_iQ_i,\quad p=(1/5,3/10,1/2),\quad x=5/7,
\qquad M=P_5(-ixR).
\]

全first operator momentがtargetであり、特定状態の期待値だけに置換しない。Hermitian involutionの関係と、明示providerのPauli情報を用いる。

\[
Q_0=Z_0,\quad
Q_1=V_1^\dagger Z_1V_1,\quad
Q_2=V_2^\dagger Z_2V_2,
\]
\[
V_1=R_{X_0X_1}(\pi/4),\quad
V_2=R_{X_1X_2}(\pi/4)R_{Z_0Z_1}(\pi/4).
\]

右側が先に作用する。sourceのPauli係数は

\[
Q_0=ZII,
\quad Q_1=\frac{IZI+XYI}{\sqrt2},
\quad Q_2=\frac{IIZ}{\sqrt2}+\frac{IXY-ZYY}{2}.
\]

このproviderはexact Clifford+T modelに含まれる。G8の仮想provider誤差$10^{-6}$を実機で達成した結果ではない。[R3]

Re/Im各精度$1/200$、22 axesの推定failure $22\times49/22000=0.049$、11 rowsのresource failure $11\times1/11000=0.001$。Rz strict精度$10^{-6}$、root K256/probability H160、$\rho=\eta=10^{-12}$。共通biasと十分shot規則を維持する。[R4]

有限P5の比較なのでTaylor打切り誤差をmethod間の同一target比較へ追加していない。これをexact exponentialやPR/QPE全体の精度保証へ延長する場合には、別途tail・step・signalの誤差会計が必要である。

## 4. 登録結果の評価

### 4.1 Direct primary

期待費用は二axes合計。Tはnative-event切片、Kは共通の準備/readout単価を掛ける呼出し数である。[R1]

| 方式 | N/axis | native T | native CX | K | system外workspace |
|---|---:|---:|---:|---:|---:|
| ordinary | 1,246,046 | 348,759,216.423 | 20,586,846.721 | 2,492,092.000 | 1 |
| partial return + tail | 949,142 | 258,865,913.425 | 15,975,424.661 | 1,898,284.000 | 1 |
| closed P3 + tail | 933,915 | 267,617,674.368 | 16,067,266.079 | 1,867,830.000 | 1 |
| general local full | 1,076,085 | 257,506,695.137 | 15,748,232.374 | 1,846,058.549 | 1 |
| closed P5 full | 917,129 | 255,860,636.626 | 15,647,565.043 | 1,834,258.000 | 1 |
| matched literal CTS | 1,596,220 | 364,311,583.986 | 9,766,516.587 | 3,192,440.000 | 1 |

closed P5は登録canonical lawのT切片とKの両方で最小である。そのため、固定lawの比較なら任意の共通非負T準備単価を戻しても最良である。

一方、最も近いpartial return対照に対するT切片の差は約1.16094%。CTSに対する29.7687%だけで、既知低次数吸収後の追加利益の大きさを説明してはいけない。

### 4.2 比較の整合性

全方式にdirect実装、共通CZ lowering、同じadjacent inverse cancellation、相対位相、strict精度を与えた。最初の5方式のgeneric helperは5行の補助診断であり、CTSと対照側へ不均衡にhelper費用を課していない。[R1,R3]

CTSのreal correction10 eventはT=0のまま保持されている。別法と同じfull P5 meanへspecializeされており、channel平均の一致だけをoperator平均へ流用していない。[R3,R5]

これは公平な比較を行う重要な条件を満たすが、全compiler、全precision配分、全representationの最適性を証明するものではない。

### 4.3 期待資源と実測・上限

Nは数学的な十分予算であり、百万回の量子測定を実施した数ではない。行列参照は小系の意味論診断、native costは保存sequenceを加算した論理資源である。

一般local fullのaccepted tail capは1,864,070、hard attemptsは2,152,170。各max-event Tを掛けた上限も保存されている。これらを期待Tと入れ替えない。wall約3.09秒はguard区間で、全command時間や大Lの生成速度ではない。[R1,R5]

## 5. closed P5と一般生成法は、同じfamily内の実装選択

closed P5とgeneral local fullの理想ensembleは同じである。finite-bit coefficientの丸めは異なるため、全digital係数のbyte一致を要求しない。

両者のaccepted一回当たり平均Tは約139.489993570で一致する。Kはlocal1,846,058.549、closed1,834,258であり、T差約1,646,058.511（約0.639229%）は主にnormalizer・zero-fill・保守的予算に由来する。[R1]

したがって「closed P5が一般法に勝ったから一般研究は失敗」という扱いは適切でない。ただし「一般Green生成器をP5でも使うべき」という結論も出ない。

推奨architectureは、低い固定次数では閉形式のgroup法、閉形式表が大きくなる一般次数では局所query法を選べるfamilyである。どちらを使うかは、normalizer取得量、群数、bit費用、cache、十分予算に基づいて説明する。

G9が認証したO(L²)はP5 group/root構築の算術operationである。現在のselectorは群lawをlinear scanするため、一試行全体O(L)と宣伝しない。lookupやalias化は通常の技術改善であり、それ自体を新規性には数えない。

## 6. CTSとの差の機構

### 6.1 費用と回数の因数分解

保存表示値から、

\[
\frac{K_{P5}}{K_{CTS}}\simeq0.57456303,
\quad
\frac{\overline T_{P5}}{\overline T_{CTS}}\simeq1.22234223,
\]
\[
\frac{G_{P5}}{G_{CTS}}\simeq0.702312657.
\]

従ってP5は一回路が約22.23%高価である一方、必要呼出しが約42.54%少なく、結果として約29.77%のT低下となる。これは積の記述的分解であり、反実仮想による因果寄与率ではない。

### 6.2 情報を多く使うCTSが常に最良とは限らない

G9 providerではPauli情報が明示されている。CTSをI0制約で排除した比較ではない。しかし、Pauliへ展開した後のliteral CTS表現が最良のnative費用を持つとは限らない。

source由来の代数から、今回のRのPauli係数L1は約1.47781746であり、involution係数の和1とは異なる。literal CTSのodd Pauli係数L1は約1.01763069、rotation係数質量は約1.42673481、real correction係数質量は約0.27359015、全normalizationは約1.70032496となる。closed P5のnormalizationは約1.28844456である。

これらは表現の違いを説明する診断量であって、Pauli L1差だけからnative勝敗を導いたものではない。G5の旧2-qubit P3でCTSが勝った事実と、今回の別provider・別次数での結果は矛盾しない。G5の閉鎖を取り消す理由にはならない。

## 7. 今回の追加導出：CTSの任意samplingに対する固定policy下界

### 7.1 動機とscope

G9の登録CTSはcanonical samplingであり、10個のreal correction eventのTは0である。従ってcanonical勝敗だけからsampling最適化後も勝つとは言えない。

以下は固定されたCTS event、有限rational係数、保存合成列、共通bias、同じBernstein予算規則だけに対する検討である。別dictionary、identity吸収、precisionの再配分、stratification、既知寄与除去、state依存分散、別confidence方式には適用しない。

### 7.2 source-compatible係数の再構成

[R3]の正確な$\mathbb Q(\sqrt2)$ providerを用いて、同じP5を形式Pauli積で再計算した。Q_i²=Iもこの代数で確認した。行列やnative回路を実行していない。

literal CTSは10 real correction eventsと14 rotation eventsを持つ。odd絶対係数和を$\widetilde L$、rotation係数和を$R_*$とすると、sourceの有限化では

\[
R_* = \operatorname{mid}\left(\operatorname{sqrt\_interval}(1+\widetilde L^2)\right).
\]

256bitの同じ意味の外向き区間から再構成した$\widetilde L$は保存監査のnew CTS tangent

`5069898366684595608883629288685085891702607078252273946648339800355852044758849588043/4982061168157627638998931315305005228262166098625077657116578852917749682837515141120`

とexact Fraction一致した。[R5]

14 rotation eventは共通Rz列とそのactual adjointを持ち、一eventのT費用は136。real correctionのTは0。$R_*>7/5$が厳密に確認できる。

### 7.3 下界

非負係数$a_e$、full-support proposal$q_e$について、

\[
m_2=\sum_e\frac{a_e^2}{q_e},\qquad
m_2\sum_e q_eC_e\ge\left(\sum_e a_e\sqrt{C_e}\right)^2.
\]

同じ予算規則なら

\[
n(q)\ge\frac{2\ell m_2}{s^2},\quad
\ell=\ln(44000/49),\quad s=0.004967999983999968.
\]

共通一回当たりT準備単価を$h\ge0$とする。rotation eventsだけを和に残しても下界の向きは正しいため、

\[
G_{CTS}(q;h)\ge\frac{4\ell}{s^2}
\left(\sum_e a_e\sqrt{T_e+h}\right)^2
\ge\frac{4\ell R_*^2}{s^2}(136+h).
\]

$\ell>34/5$、$s<1/200$、$R_*>7/5$を用いると、

\[
\boxed{G_{CTS}(q;h)>290,017,280+2,132,480h.}
\]

一方、保存されたclosed P5値の保守的表示上端は

\[
G_{P5}(h)<255,860,637+1,834,258h.
\]

従って、固定CTS classとpolicyについて、任意のfull-support proposalおよび任意の共通非負準備単価に対し、保存closed P5の費用が低い。CTSのT=0イベントを微小な正費用へ置換していない。ISのinfimumが達成されることも要求していない。

より鋭い値ではh=0のleading下界は約305,097,829だが、主たる符号確認には上の粗い290,017,280を使った。約29.77%のcanonical削減率そのものを、最適sampling後の保証削減率とは呼ばない。

### 7.4 検算の正確な位置付け

対数の粗い不等式はexp(34/5)の有理Taylor部分和＋幾何tail上界で確認した。係数・平方根はFraction/isqrt、重要な比較は整数・有理数で行った。

ただし、全11行・全1,866 event・全strict guardの独立再実行ではない。closed P5のT/KはG9保存報告・監査から使用した。これはsource由来のレビュアー検討であり、正式なclass証明書としてrepositoryへ追加したものではない。次のCodex作業で独立確認する価値はあるが、これだけの確認を別の何段もの研究承認へ分割しない。

## 8. この下界でも閉じない比較

### 8.1 literal CTSという範囲

論文CTSのoperator分解はreal correctionとcommon-angle rotationを分ける。[P1] sourceもidentity correctionを別eventに保つ。

identity係数をrotation側へ吸収する変形、他のPauli grouping、precision再配分は辞書・実装変更であり、上記下界の外である。前者は新たな角度を必要とし、取得済み136 Tをそのまま使って資源優位／非優位を結論してはいけない。

本レビューは「literal matched CTSの固定実装・任意samplingに対する分離」を支持するのであって、「CTSに関連する全構成を打ち破った」とはしない。新規性表ではidentity吸収・modified Taylorとの差を明示する。[P4]

### 8.2 最も近いpartial対照

登録canonical partialとの差は約1.16%であり、約29.77%より小さい。G8のRz-only下界を、G9のprovider込みnative価格へそのまま転用してはいけない。native価格を使ったcost-aware対照、残余精度配分の影響は未完了である。

CTSとの差がsamplingだけで消えないと確認できたことは有用だが、全既存3対照についてのnative最適proposal分離まで同時に証明したわけではない。

### 8.3 全体の時間発展

P5単体を同じmeanで実現する比較は意味がある。しかしPR/QPE全体では、時間分割、Taylor remainder、deterministic部、複数ステップの二次モーメント、状態準備、推定器の費用を再構成する必要がある。今回の削減率を全体へ乗算してはいけない。元PR研究はQPEまでの費用を扱っており、それが最終応用目標なら同じ段階の会計が必要となる。[P3]

## 9. CX、準備費用、期待値とtail

登録native CXではCTSが小さい。しかし共通一回当たりCX準備/readout単価を$h_{CX}$とすると、保存canonical二点の差は

\[
G_{CTS}^{CX}-G_{P5}^{CX}
=-5,881,048.456+1,358,182\,h_{CX}.
\]

よってcrossoverは約4.33009 CX/call。これを超える共通付帯費用なら、当該二点の総CX順位は反転する。

これは現実の準備費用を4.33へ設定したという意味でも、再最適化後のPareto境界でもない。native-event座標と最終taskの総費用を区別するための固定law感度である。

T/CXを任意の重みで足して後付けwinnerを作らない。T-primaryを維持し、CX/1Q/workspace、期待費用、accepted tail、hard attempts、classical取得費用を別々に報告する。

## 10. 新規性・先行研究の再評価

### 10.1 確認した一次文献

- CTS：出版本文Theorem 1/Eqs.(5)–(6)、Methodsのfinite specialization、Markov/layeringの記述を確認した。[P1]
- Resource-optimal IS：Theorem 1のnet-cost下界・最適proposalを確認した。これは固定protocolを前提とする既知原理である。[P2]
- 元PR：arXiv上の現行表示はv2（2026-07-10）、出版はPRX Quantum 7, 020332（2026-05-19）。G9の有限mean比較とPR/QPE全体の目的の違いを確認した。[P3]
- Aomoto–Kato：一次資料の書誌と§1の解析テキストでGreen multiplier/自由積の基盤を確認した。PDFページ画像取得はcache errorで失敗したため、今回の全ページ視覚精読を主張しない。G6の詳細対応監査も区別して参照する。[P5,R7]
- Wan–Berta–Campbell：ordinary RTEの基盤として出版情報を再確認した。今回全Algorithm 2を再証明したわけではない。[P6]
- Zhao–Yuan：modified Taylor/anticommutationを用いるLCUの近接研究として出版・概要を確認した。今回のHTML/PDF取得は失敗しており、詳細な同値性監査にはG6記録を用いる。[P4,R7]

検索で本familyと全く同じ接続を見つけられなかったことをpriorityの証明には使わない。網羅的な引用ネットワーク監査は完了していない。

### 10.2 主張してよい候補と避ける主張

主張候補は、有限Taylorの全形式returnを保持したphase-sensitive operator ensemble、正の短時間係数、局所生成、有限bit/bias/予算、低次数特殊化、必要なnative取得までを接続する構成である。

Green関数そのもの、Euler identity、return吸収という一般原理、importance sampling、有限辞書の凸最適化を新規性には数えない。P5 O(L²)の特殊化だけについても、既知modified Taylorや記号代数からの容易な系ではないかを比較する必要がある。

G9は実装利益の具体例を追加したが、priorityや独立論文十分性を自動的に与えない。一方、理論method論文に実機での量子優位を必須条件として追加することもしない。入力契約・構成・計算量・保証が明確なら、条件付きアルゴリズム研究として成立する可能性を検討できる。

## 11. 研究の着地点

### 11.1 推奨する中心課題

> 同じ有限Taylor平均を実現する際、involution構造とreturn集約を保つ表現を、取得可能な情報・有限bit・実装費用から選択し、低次数特殊化と一般局所生成をどの範囲で使い分けるべきか。

これは「closed P5を常に勝たせる」「CTSよりいつでも優れる」「Pauli情報が取れないと仮定して勝つ」という問いではない。

### 11.2 原稿をまとめる形

第一の成立形は条件付き構成・アルゴリズム研究である。

1. 入力、operator target、control accessとclassical取得費用の明示。
2. 短時間・有限奇数次数の恒等式と非負性。
3. 非列挙query・zero-fill・finite-bit誤差/予算。
4. P3/P5特殊化と一般実装の関係。
5. native固定例、強い対照、固定policy分離、適用限界。

第二の成立形は実用資源研究だが、その場合は複数の構造、取得費用、実際のPR全体への接続がさらに必要である。現在の証拠をもって分子scaleの論文にしたとは言えない。

原稿のclaim/evidence表を作る段階には進める。本文執筆を新しい実験の自動承認と同一視しない。R0–G5の全開発履歴を全て主論文へ詰め込む必要はなく、古い限定結果は補足・研究記録へ分ける。

## 12. 次の検証の比較と採否

| 選択肢 | 判断 | 理由 |
|---|---|---|
| 全面v4や元G5辞書の再開 | 不採用 | 既存閉鎖を変更する根拠ではない |
| 同じP5を再実行してPASSを増やす | 不採用 | 重要な未解決事項に答えない |
| 大規模分子/DFへ直行 | 保留 | 有限meanから全taskへの接続と一般次数の役割が未整理 |
| p,x,providerを同時に変えた多数探索 | 不採用 | 要因帰属が曖昧で、勝つ条件の選別になりやすい |
| 固定P5でprecision/seedを細かく探索 | 保留 | 現結果の救済目的にしない。必要なら将来のrobustness契約で行う |
| **同じp,x,providerで次数だけを変え、低次数吸収＋tailと一般法を比較** | **次の優先検証として提案** | 一般生成器の追加価値を、低次数特殊化後に直接判別できる |

七次は最小の未評価高奇数次数として選ぶ提案であり、七次で勝つことが一般的な研究成立の必要条件だと主張するものではない。ここでの目的はG9のP5実装から先へ広げる情報価値を測ることである。

## 13. 次のCodex作業：G10の一括した研究判断用bundle

本節は推奨scopeである。レビュー自体がrepositoryの`next_science_authorized=false`を変更したり、実行を開始したりしたわけではない。利用者が本方針を採用してCodexへ渡す場合、以下を一つのbundleとして準備・実施する。

### A. 保存証拠の独立補強とclaim設計

本レビュー§7のCTS固定policy下界を、原resultのrational coefficients・native costから独立に再導出する。実際にT=0 real eventsを保持し、rotation全event136 T、共通prep費用の係数を確認する。理想Pauli係数、rational angle、digital係数の違いも保つ。新しいCTS law探索や合成は、この部分には不要。

同じ作業内で、partial/P3対照へnative-cost-aware proposalを戻したときの不足を整理し、登録値・下界・候補値を混同しない。分離できなければ未分離を正確に記録する。新規性比較表も、Green、RTE、modified Taylor、literal CTSとの入力/ensemble/cost/保証の差まで具体化する。

### B. 結果前に固定する次数比較

p=(1/5,3/10,1/2)、x=5/7、同じG9 3-qubit provider、同じdirect loweringを維持し、m=3,5,7の比較を設計する。m=5は既存anchorとして再利用する。各m内部では必ず同一P_mを比較し、異なるm間のTを同一accuracyのexponential性能としてランキングしない。

対照はstreaming ordinary、partial P3+ordinary tail、closed P3+ordinary tail、closed P5+ordinary tail（m>=5）、general full、同じP_mのliteral matched CTS。P3/P5に閉形式がある場合はそれを使い、一般器を無理に強制しない。

m=7では特に「closed P5+次数6/7 ordinary pair」とfull returnを比較する。これが、P5までの安価な集約で説明できない追加価値に答える対照である。

cost-awareな有限proposalを加えるなら、全方式へ同じ情報accessと有限構成規則を認め、T=0を正しく扱う。leading ISのinfimumを実行可能な予算と呼ばない。共通prep Tを結果後に選んでwinnerを作らず、切片とK、必要なら成立領域を保存する。

native angle/precision/source/backend/compiler、bias、failure配分、全row数、CPU/RSS/key/output上限は取得前に固定する。keys上限は静的なgroup/angle見積りからCodexが決めてよい。未登録のp/x/provider/seed探索で有利域を探さない。

### C. 非列挙性と費用を同じbundleで記録

productionは参照event全表を読まない。小support列挙は別referenceに留める。一般mのlocal query、取得angle/cache、finite-bit情報量、termination/abort条件と、低次数のgroup構築・selectorを分ける。

reference全eventから正確な期待native費用を出すことと、productionが非列挙で動くことは別証拠である。compileや期待値の表だけから古典scaling優位を宣言しない。

### D. 過剰な承認分割をしない

科学的target・provider・主要比較を変えない範囲の型修正、テスト、数値区間、source整理はCodexがまとめて処理してよい。各テストでGPTへ戻す必要はない。

結果の勝敗によらずbundle後はSTOP。m=7で差がない場合も、同じ応答でm=9や新pへ自動拡張しない。旧v1/v2のresult・marker・authorization・sourceとTrack Aを保護する。

## 14. 次のGPT判断：何をもって継続／縮小するか

- 低次数対照・native sampling自由度を戻した後にも、一般次数の利益または有効条件を説明できる場合：方法研究としてのclaimを具体化し、構造移送は一つの科学的対比から選ぶ。
- closed P5+tailで費用が説明し切れる場合：一般Green-generatorを性能上の主役としては主張せず、低次数実装＋一般構成の限界を整理する。これだけで全理論を否定しない。
- literal CTSを超えても、簡単な既知再構成に吸収される場合：method deltaを縮小する。名称だけ変えて同じ探索を続けない。
- 認証が弱く未分離の場合：真の非改善と下界不足を区別する。科学的な改善余地が小さいのにcertificateを精密化し続けることを目的にしない。

判定は既存の結果から5%/10%を逆算するものではない。取得費用・実装範囲・比較強度・理論上の差と、追加検証が何を明らかにするかで判断する。

重要なのは、G10を次の番号を増やすための一段階にしないこと。結果が戻ったら、投稿へ向けて現claimをまとめる、構成ノートへ区切る、一般化検証を一つだけ進める、のいずれかを選び、無期限の小修正検証へ流さない。

## 15. 最終的な推奨

G9 v2は、return集約familyの優先継続を支持する具体的なnative evidenceである。closed P5をfamily内の低次数標準実装とし、一般器はそれで代替できない範囲を検証する。

canonical CTSへの約29.77%の差を全世界の優位と呼ばない一方、固定CTSのsampling未最適化だけを理由に成果を過小評価もしない。本レビューの保守的下界は、固定CTS classとpolicyの下で、その懸念を越えた差が残ることを示す。

次はCodexで、保存分離の独立補強・claim設計・同一構造の次数比較を一括して進めることを推奨する。DF/分子/QPE全体と広い一般化は、その結果で必要性を判断する。

---

## 参考資料

### Repository：固定commit c95736fd…

[R1] [G9 v2 handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md)

[R2] [Evidence manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/evidence_manifest_v2.json)

[R3] [g9_native.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/src/trottertracks/algorithm_codesign/g9_native.py)

[R4] [g9_comparison.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/src/trottertracks/algorithm_codesign/g9_comparison.py)

[R5] [保存値監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_output_audit_v2.json)、[原result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/result_v1.json)

[R6] [G9 v1 technical failure](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/docs/tracks/algorithm_codesign/g9_results_and_gpt_handoff_20261010.md)

[R7] [G6 prior-art audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md)、[G6 mathematical audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)

[R8] [G9 contract v2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/contract_v2.json)、[v2 source manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/source_manifest_v2.json)

### 一次文献

[P1] Peetz, Smart, Narang, *Quantum simulation via stochastic combination of unitaries*, npj Quantum Information 12, 52 (2026). [出版本文](https://www.nature.com/articles/s41534-025-01168-w), DOI:10.1038/s41534-025-01168-w. [arXiv v2](https://arxiv.org/abs/2407.21095v2) は2026-07-06。

[P2] Cugini, Atif, Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, [arXiv:2603.13495v1](https://arxiv.org/html/2603.13495v1), 2026-03-13. Theorem 1/Eqs.(9)–(13)。

[P3] Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum 7, 020332 (2026), [DOI:10.1103/ynxb-p2xq](https://doi.org/10.1103/ynxb-p2xq), [arXiv:2503.05647v2](https://arxiv.org/abs/2503.05647v2)。

[P4] Zhao, Yuan, *Exploiting anticommutation in Hamiltonian simulation*, Quantum 5, 534 (2021), [出版ページ](https://quantum-journal.org/papers/q-2021-08-31-534/), DOI:10.22331/q-2021-08-31-534。

[P5] Aomoto, Kato, *Green functions and spectra on free products of cyclic groups*, Annales de l'Institut Fourier 38(1), 59–85 (1988), [一次資料](https://www.numdam.org/articles/10.5802/aif.1123/), DOI:10.5802/aif.1123。

[P6] Wan, Berta, Campbell, *Randomized Quantum Algorithm for Statistical Phase Estimation*, Physical Review Letters 129, 030503 (2022), [出版ページ](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.030503), [arXiv:2110.12071](https://arxiv.org/abs/2110.12071)。

## 再現添付の読み方

`cts_fixed_algebra.py`は固定providerの形式Pauli代数とCTS midpoint、粗いpolicy下界を検査する。`cts_policy_scalar_check.json`のexact checksが主たる符号検査であり、`cts_fixed_algebra.json`のmpmath表示値は診断用である。原result全体の代替ではない。

`summary_arithmetic.json`はhandoffの表示値から行った因数分解・差の計算である。登録certificateや新しい科学runではない。

再現scriptを実行してもrepositoryを変更せず、外部ネットワーク、合成backend、量子/行列計算、sampler、乱数を使用しない。
