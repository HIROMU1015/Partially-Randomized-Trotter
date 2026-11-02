# Track A H6精度一致資源比較 v2：MemoryError後の独立科学レビュー
## 保存証拠の再利用、q=8側の欠測、公平な補完と独立確認への接続

- 作成日：2026年10月11日（日本時間）
- レビュー開始承認：利用者の「レビューを開始して」
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- 対象branch：`track-a-ax2b-h4-post-review-20261010`
- 固定結果commit：`8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb`
- 固定science source：`54986a9f54515c8c2289e38fa16fd81e2b6008c2`
- 原run：`H6_MATCHED_RESOURCE_STOP` / `RuntimeError:MemoryError:`
- レビュー状態：**完了**
- 判断：**科学方針を維持し、保存済み部分を再利用する新しい補完契約を準備する。原runを再開・成功へ改変せず、欠測143探索groupと既定の独立確認を完成させる方針を推奨する。**
- 次の担当：**Codex（限定的なメモリ対処・保存証拠再利用・補完実行の準備）**
- 実行認可：本書は新grantではない。旧grantは消費済み。H8、追加候補探索、実行時の無断上限変更は認可しない。

---

## 0. 結論

今回の実行は、全92候補のsignalを取得し、費用探索592 groupのうち449 groupを完了した。全体はMemoryErrorでSTOPしたが、数値処理前に止まった旧v1とは違い、研究判断に使える部分証拠が大量に残っている。[R01–R08]

**研究方針を変える根拠よりも、現在の固定比較を完了させる根拠の方が強い。** 特に、qを増やしたB2は厳しい信号精度でも適格になり、取得済みの少数費用標本には、四次PFとの比較を継続する価値がある。一方、費用の欠測はq=8側に偏り、強い決定論対照B1 S4 q8と、PR側の多くのq8構成が未完了である。観測済み部分だけで全候補のwinnerを決めてはいけない。

推奨は次の組合せである。

1. 原STOP・消費済みgrant・保存raw・欠測を保持する。
2. 保存92 signalと完了449 cost groupを、新契約の入力証拠として明示的に再利用する。全面再計算を第一案にしない。
3. B1 S4 q8を重いという理由で削除せず、同じ回路・compiler条件の費用を取得できるメモリ対処を準備する。
4. 原登録592 groupの未完了143 groupを補完する方針を維持する。別のgrid探索へ広げない。
5. 全探索が揃ってから、元の「各精度B2上位2→union最大8」を固定し、探索と別seedの32 trajectory確認へ進む。
6. 独立確認、未観測order mass、数値allowanceの限界を併記してGPTへ戻す。H8はまだ開かない。

これは「PRの一般的優位が証明された」という結論ではない。また、旧runが成功したという結論でもない。**部分証拠を失わず、元の科学的問いに答えるための補完方針を決めたレビュー**である。

## 1. 証拠区分と確認範囲

### 1.1 原記録・再計算・判断の区別

本書の数値は、(A) repositoryに保存された観測値、(B) そのスカラーから本レビューで計算した量、(C) 未検証の見通し、を区別する。特にNは実測量子shot数ではなく、宣言された経験的allowanceの下での十分shot会計である。Gの本書内計算は、原runが保存できなかった最終resource mapの復元物とは扱わない。

### 1.2 実際に確認した内容

GitHubから結果報告、停止・coverage記録、登録と会計のsource、signal/cost処理、既存fingerprint、compile helperを参照した。数値的な重点照合は、四次PFのq2/q4/q8、B2のp5/p10/p15・q4、およびp5/q8という**7 signal record**である。費用は四次PF q2/q4の2 group、B2 p5/q4/r1/K2の8 group、B2 p5/q8/r1/K2の2 group、計**12 group・24 primary wrapper**の保存値を直接確認した。失敗q8対照のcontrol検査も確認した。

全92 signalと449 groupの完了件数・配置・来歴は保存coverageと監査に基づく。**全902 wrapperを一件ずつ別実装で再認証したわけでも、全92候補×4精度の全順位を独立再集計したわけでもない。** したがって本書では全候補の最良構成を宣言しない。選抜・母平均認定より前の、研究方針と補完範囲の判断を目的とする。

全signal summaryはContents経路で本文が空になる場合があったが、Git blob経路で取得できることを確認した（blob `9bc570495d256d55868ddd658fff57e8a7c55669`）。取得経路の制約を未pushと扱っていない。数値検算には、上記7候補の個別recordから転記したfieldを使用した。

### 1.3 本レビューで新たに実行したこと・していないこと

実行したのは保存スカラーの四則演算・指数・shot式・標本平均/SD/SE・閾値の計算である。7候補のepsilon=0.001の軸別Nを独立のstdlibコードで再計算し、保存整数と一致を確認した。

SCF、DF decomposition、state solver、Hamiltonian作用、signal再評価、回路build、transpile、trajectory sampling、repository test、本番runner、remoteへのwriteは実行していない。全hashの独立再計算もしていない。添付コードはrepositoryのmoduleをimportせず、保存値の転記入力だけを使う。

## 2. 現在の研究対象と、変えていないもの

対象はlinear H6、1.00 Å、STO-3G、12 modes、alpha3/beta3、400次元sector、T=0.8。同じ保存stateと、tol-only 1e-8・rank19・cutoff0・signed generation orderのweighted Hermitian projection受理DFを使っている。[R01,C01,C02]

| 比較family | 登録範囲 | 候補数 |
|---|---|---:|
| B0 | prefix5/10/15、q1/2/4/8、S2 | 12 |
| B1 | full19、q1/2/4/8、S2とglobal S4 | 8 |
| B2 | prefix5/10/15、q1/2/4/8、r1/2/4、K2/4 | 72 |
| 合計 | epsilon_sig=0.05/0.01/0.005/0.001を共通評価 | 92 |

B3はq4値×r7値×K3値の84 normalization組のみ。signal・cost候補を追加したものではない。以前の168という表記は誤記であり、84/84完了を採用する。原文・rawを歴史的に保全したことと、誤った件数を今も使い続けることは別である。[R04,R14]

主RQは、同じ有限時間coherent signal taskで、PRの回路短縮がbias・normalization・測定回数を払っても残るか、というRQ-Rである。RQ-P1は補助。信号精度を化学的なエネルギー精度へ換算しない。新たなHermitization政策、DF再分解、state再生成、Track Bのevent law導入は今回必要ない。

## 3. STOPと保存証拠の科学的な扱い

### 3.1 完了と欠測

| 項目 | 保存状態 | 本レビューでの扱い |
|---|---|---|
| signal | 92/92 | 条件付き数値・精度会計の証拠として保持 |
| cost exploration | 449/592 group | 完了group単位で再利用候補 |
| wrapper | 902件 | 898 primary＋登録ordinary感度4件という会計と整合 |
| confirmation | 未登録・未開始 | 独立確認の成功とは扱わない |
| B3 normalization | 84/84 | 件数訂正を反映。新計算不要 |
| 原run全体 | STOP | COMPLETEへ書き換えない |

592−449=143 groupが未完了である。ただし、すべてが未着手ではない。B1 S4 q8の失敗group、並行していたB2の中断group、その後の未dispatch groupを分ける。rawに途中の計算回数が残らない場合、未保存を0回で埋めない。[R03,R09,R12]

### 3.2 欠測はランダムな標本欠落ではない

費用taskはqの順序に沿って進み、q8のB1 S4で停止している。q8側にはPRの残り候補も多数ある。このため、保存449 groupは、全候補から無作為に得た部分集合ではない。

ここで二つの不適切な結論を避ける必要がある。

- 高価な決定論対照が未取得だから、取得済みPRが全対照に勝ったとする。
- q8のPR候補がほとんど未取得だから、低qでのPRだけで非優位を判断する。

どちらにも偏り得る。**失敗は手法の物理的優劣の証拠ではなく、現実のcompile pipelineで得られた技術的制約として別に報告する。**

### 3.3 静的監査PASSの意味

STOP時のstatic auditor PASSは、source/input/grantと保存byte closureの整合を表す。全taskの完了、各cost母平均の精度、全候補のwinnerを表すものではない。新しい再利用manifestでは、完了済みgroupのterminal・task・両軸・seed・event・compiler・hashを再度結び付ける必要がある。[R06,S02]

## 4. 代表signalから分かること

次の表は今回の重点照合7候補であり、全92のwinner表ではない。epsilon=0.001のNは、保存empirical allowanceとalpha_axis=0.025に基づく両軸の合計である。[D20–D26]

| 構成 | total signal差 | log B | N_total（0.001） |
|---|---:|---:|---:|
| B1 S4 q2 | 2.36742718454e-4 | 0 | 60,083,125 |
| B1 S4 q4 | 1.51515882700e-5 | 0 | 36,114,598 |
| B1 S4 q8 | 9.52567526289e-7 | 0 | 35,121,323 |
| B2 p5/q4/r1/K2 | 6.66322898290e-4 | 0.132256236583 | 660,963,710 |
| B2 p10/q4/r1/K2 | 6.67252093816e-4 | 0.000205719741 | 517,456,576 |
| B2 p15/q4/r1/K2 | 6.67252076864e-4 | 1.6165e-13 | 517,244,857 |
| B2 p5/q8/r1/K2 | 1.66111355770e-4 | 0.067112728055 | 57,403,863 |

### 4.1 最初に適格になるqは、総資源最小とは限らない

B2 p5/r1/K2のq4は、0.001の宣言された精度会計で既に適格である。しかしimag biasが約5.656e-4なので、統計誤差へ残る余裕は約1.415e-4しかない。一方、q8ではimag bias約1.412e-4、余裕約5.659e-4となる。

この違いによりNは約6.61億から約5,740万へ下がる。q8の回路が長くなっても、必要な測定回数が大きく減る可能性がある。**適格になる最小qを自動採用する規則は、今回の目的には不十分**である。

### 4.2 四次PFにはbiasを下げすぎる領域もある

S4 q4→q8でsignal差は約16倍小さくなっているが、0.001のN削減は約2.7503%に留まる。既にbiasが小さいと、さらにbiasを下げても総shot数は統計誤差予算の限界に近づく。

これはq8を削除してよい証明ではない。だが、なぜ最小biasだけを目的関数にしてはいけないかを示す、今回の保存値に基づく例である。

### 4.3 q4のprefix比較：主にどの差を動かしているか

B2 q4/r1/K2で、p5のfinite-RTE成分は約1.593e-6、outer-PF成分は約6.662e-4。p10ではfinite約3.424e-10、outer約6.673e-4、p15ではfinite差が約4.6e-14となる。[D23–D25]

したがって、少なくともこれらの登録点では、p10からさらにfinite近似だけを改善しても支配的なouter-PF差はほぼ変わらない。一方、p5はnormalizationが大きく、prefixを小さくする回路利益と測定負担が競合する。p15の10^-14差は数値allowanceより小さく、それを精密な真のfinite誤差として解釈しない。

今回もKやrの単独増加を主題にするより、q・prefix・normalizationを一緒に比較する現在の計画が適切である。表の3点から他の全r/Kの単調性まで主張しない。

## 5. 取得済み費用による局所的な比較

### 5.1 読み取った費用

primaryはsymmetric_directional、state preparationを除くmeasured Hadamard wrapperである。以下の直接確認した24 wrapperではcosine/sineのRZ数が同じだった。この観測を、未実行の軸compileを省略する一般則にはしない。

| 構成 | 取得済みRZ/軸 | 用いた標本数 | 読み方 |
|---|---:|---:|---|
| B1 S4 q2 | 221,341 | 決定論1 group | 固定compilerの取得値 |
| B1 S4 q4 | 441,493 | 決定論1 group | 固定compilerの取得値 |
| B1 S4 q8 | 未取得 | 0 | 比例外挿で埋めない |
| B2 p5/q4/r1/K2 | 平均47,143.875 | 探索8/8 | 宣言された探索標本の平均 |
| B2 p5/q8/r1/K2 | 平均93,301 | 探索2/8 | 不完全な探索subsetの記述値 |

q4 B2のRZ列は、46809, 47139, 46963, 46975, 47016, 46939, 48497, 46813。q8 B2の2標本は93441, 93161である。[W300–W307,W448,W449]

### 5.2 epsilon=0.001の条件付き点比較

保存されたNと上記標本平均を掛けた、本レビュー内の事後計算を示す。**これはoriginal exploration_resource_map.jsonでもconfirmation結果でもない。**

| 構成 | N_total | RZ平均 | G_RZ点診断 |
|---|---:|---:|---:|
| B1 S4 q2 | 60,083,125 | 221,341 | 1.3298858970625e13 |
| B1 S4 q4 | 36,114,598 | 441,493 | 1.5944342214814e13 |
| B2 p5/q4/r1/K2 | 660,963,710 | 47,143.875 | 3.116039052377625e13 |
| B2 p5/q8/r1/K2 | 57,403,863 | 93,301（2標本） | 5.355837821763e12 |

四次PFではq4の方がbiasは小さいが、q2より点Gが約19.89%大きい。B2ではq8の2標本点Gは、q4の8標本点Gの約0.17188倍となる。

さらにq8 B2の点GはS4 q2の約0.40273倍である。この比は、N比約0.95541と1-wrapper費用比約0.42153の積として理解できる。

**この差は、q8側を補完する情報価値を支持する。だが「PRが59.7%改善した」という確定成果にはしない。** q8 B2は2/8に過ぎず、候補集合全体・独立確認・未観測高費用eventが残っている。未完了探索からconfirmationの候補を今ここで確定しない。

### 5.3 準備費用について

同じ比較点で共通1-shot準備費用Pを加えるなら、G(P)=G(0)+N_total Pである。q8 B2の現在のNはS4 q2より小さいため、この二点の点比較は非負Pを加えることだけでは逆転しない。ただしこれは現在の費用標本平均に基づく二点の関係であり、全候補の最適性でも物理装置のruntime評価でもない。

## 6. 欠測しているS4 q8の情報価値を定量化する

「最も正確だから必ず必要」「高価そうだから不要」の二択ではなく、どの費用なら選択が変わるかを示す。

q8 S4のshot-weighted per-wrapper費用を

\[
C_{8,N}=\frac{N_{8,R}C_{8,R}+N_{8,I}C_{8,I}}{N_{8,R}+N_{8,I}}
\]

と定義すると、両軸の費用が同じと仮定しなくても、G_8=N_8 C_{8,N}である。epsilon=0.001の保存Nから、次の閾値を得る。

| 比較対象 | S4 q8が下回るためのC_8,N |
|---|---:|
| S4 q2の取得済み点G | 378,654.84未満 |
| S4 q4の取得済み点G | 453,978.97未満 |
| q8 B2の2標本点G | 152,495.33未満 |

既にq4 S4の費用は441,493であり、q8に大きな回路削減がなければ第一・第三条件は厳しいと見込まれる。しかし、**qを倍にしたcompiled costは必ず倍、あるいは必ず単調増加、という証明は本レビューでは得ていない。** この閾値だけでq8を除外してはいけない。

残り1個のdeterministic groupを取得できるなら、元の固定比較を閉じる方法として最も明快である。将来、既存証拠から有効な費用下界を安価に証明できる場合は、取得の代わりにその下界で選択可能性を閉じる余地はある。ただし、その新しい証明研究を今回の必須作業として追加しない。

## 7. Rare eventと独立確認の限界

### 7.1 実際の分布での例

q8 B2 p5/r1/K2の保存event lawでは、1 eventの次数0確率は0.9958176334985577、次数2確率は0.004182366501442349である。R=8なので、n trajectoryの全eventが次数0となる確率は

\[
P_{0,\mathrm{all}}(n)=p_0^{8n}.
\]

| n | 次数2が一度も現れない確率 |
|---:|---:|
| 2 | 0.9351407812 |
| 8 | 0.7647297529 |
| 32 | 0.3420046016 |

これらは保存した確率からのレビュー内計算であり、追加samplingの結果ではない。replica0の保存observed_ordersは全8個が0だった。[E01]

### 7.2 32標本でも何が未解決か

小さいSD/SEは、未観測次数やcomponent列の影響を自動的に含むわけではない。n=32は現在のengineering確認計画として維持できるが、全supportの平均費用を厳密に認定する標本数ではない。

同時に、34%の確率で次数2を見ないことは、「平均費用が大きく誤る」と直ちに証明するものでもない。必要なのは、その確率と高費用eventの寄与を区別することである。現在は有効なpopulation compiled-cost上界がないため、確率だけからtail費用を認定しない。

既定のorder診断を完成させ、観測order、未観測mass、全次数0の確率、標本数とSD/SEを併記する。instruction capは、そこで停止する実装の制限であり、そのcap以下に全distributionのcompiled costが収まるという定理ではない。[S02]

### 7.3 追加統計を無制限に要求しない

今回の目的は、宣言された有限比較のpoint mapと独立確認を取得することだった。したがって、32でformal certificateが出ないという既知の理由だけで、突然全event列挙や大量再samplingを必須にしない。

確認後に母平均の不確かさが科学的claimを左右する場合には、population winnerをUNDETERMINEDのままにし、必要ならorder層化・有効なtail bound・追加標本を別のamendmentで検討する。都合のよい結果が出るまで標本を増やさない。

## 8. 数値allowanceとshot会計の評価

保存Nはempirical numerical allowanceを仮定する会計で、u-aware関数自体はcertificateを検証しない。sourceはlog-domainで安全性を管理し、整数化可能範囲では既存演算順でceilを取る。[S02,S03]

\[
h_a=\epsilon/\sqrt2-|\widehat{\Delta z}_a|-u_a,
\qquad
N_a=\left\lceil\frac{2e^{2(\log B+\Delta\log B)}\log(2/0.025)}{h_a^2}\right\rceil.
\]

cost取得にはh>0、整数shotが表現可能、u/h≤0.01を要求する。この規則は全候補に共通だが、同時信頼度つきwinner選択の保証ではない。

重点照合した7候補では、epsilon=0.001のNを再計算し保存整数に一致した。uとlog_B marginを10倍にしたときのNの最大相対変化は、この7点では約1.1581e-5、すなわち約0.00116%だった。これは代表点の点比較がこの感度だけで反転しそうにないという限定的診断であり、92候補全部の境界安定性や、実際の数値誤差を証明するものではない。

raw/correctedの一致、別occupation/forwardとの一致、Hermitian target受理は、それぞれのscopeで扱う。共通規約誤り・finite precision・ground-state認証まで一括PASSとしない。一方で、総uの厳密証明を今から一律に新しい前提にして、費用補完を止める必要もない。

## 9. MemoryErrorについて確認できたことと、未確認のこと

### 9.1 旧RLIMIT継承不具合とは別である

現在の失敗taskはworkerのAS soft/hard8 GiBで起動している。以前の2 GiB hardを継承して数値import前に止まった問題とは異なる。今回はsignal全92と多くのcost取得へ進んだ後、B1 S4 q8のcost taskがMemoryErrorで止まった。[R01,R09,R11]

### 9.2 失敗位置を限定できる範囲

B1 S4 q8の`validation_symmetric_directional.json`は保存され、最大差は8.922212901461267e-14である。sourceの順序では、primary evolutionの構築、保存stateのcontrol branch/X/Y検査、validation保存を終えた後にmeasured wrapperを構築し、fingerprint、transpile、count/depth計測、wrapper保存へ進む。[R10,S01]

したがって、最初のwrapper recordより前の費用処理区間で失敗したことが分かる。しかしMemoryErrorの正確なallocation、traceback、失敗時VmSize/RSSが保存されていないため、「Qiskitのtranspile本体が原因」「物理RAMを使い切った」とは断定しない。

### 9.3 既に行われている対策を見落とさない

現在のcost取得はtaskごとのdisposable processで、ordinaryとsymmetricを同時に保持しない方針になっている。fingerprintもbounded cacheとstreaming hashを使い、トップレベルの巨大JSON文字列を一括生成していない。[S01,S05]

従って「processを終了すれば直る」「まずstreaming hashへ変更する」という一般論だけでは、今回の追加対処にならない。一方、処理中はevolution、measured wrapper、transpiled circuit、compiler内部表現等が重なり得る。`transpile_and_measure_cost`はtranspile後も回路を保持し、複数のdepth/countを計測する。[S06]

### 9.4 メモリ計測の注意

q4 S4の取得済みwrapperでは保存peak RSSが約5.96–5.97 GBだった。これを単純に2倍してq8の必要量を予測しない。また、失敗batchを含まないcoordinator inventoryのpeakをrun全体の上限として使わない。[W299,R05]

RLIMIT_ASは仮想アドレス空間の上限であり、RSSと同じ量ではない。二つのcost workerを一つに減らしても、各workerのAS8 GiBという制限自体は変わらない。[X01] 並列度低下だけで個別taskのMemoryErrorが解消すると保証しない。

## 10. 推奨する技術対処：scopeを限定し、比較の意味を保つ

第一候補は、失敗区間の保持量を記録し、不要な参照を短くする等、**同じfull-wrapperとcompiler条件のままpeakを下げる**修正である。記録すべき区間は、wrapper構築前後、fingerprint前後、transpile前後、cost metric計測前後。RSSとVmSize、AS soft/hard、instruction数を小型receiptで保存する。失敗時tracebackはboundedに保存し、OOM後に巨大objectをserializeしない。

ただし、8 GiB内に収める最適化を際限なく続ける必要はない。現行の量子回路・compiler・精度・比較lawを変えず、十分な利用可能メモリを確認できるなら、問題task用の単独workerと別途認可されたAS予算を採ることも科学的には許容できる。具体的な数値はCodexがホストの現状と計測を基に提案・固定する。worker上限を変更する場合は、coordinatorから継承するhard ceilingとの整合も確認し、以前の継承不具合を再導入しない。**本書では自動的に16 GiB等へ変更する認可を与えない。**

次は避ける。

| 対応 | 今回の判断 |
|---|---|
| B1 S4 q8だけを比較から削除 | 第一案にしない |
| qを変更して成功扱い | 禁止。別候補である |
| optimization level/basis等をそのtaskだけ弱くする | 元費用と混ぜない。比較条件変更になる |
| block単位費用を足し、actual full-wrapper費用に置換 | 境界簡約等が異なる。検証なしに代用しない |
| 旧8 GiBを実行中に引き上げる | 禁止。新契約・新grantが必要 |
| 新しいphase/total wall capを導入 | 利用者方針に反するため採らない |
| 同じ失敗runを最初から全部繰り返す | 第一案にしない |

軽量化でuntranspiled wrapperのcanonical内容が変わらず、compiler/hash/seed/環境条件も維持されることを確認する。単なるunitary一致だけでは、compiled costの比較条件が同じとは言えない。元回路を変える最適化へ進む場合は、その範囲のbridgeと科学的な比較設計を別に扱う。

## 11. 保存結果の再利用は可能か

**可能であり、推奨する。ただし新しい証拠統合契約が必要である。** 原runのretry/resume禁止を無視して続きを呼ぶのではなく、完了済み記録をimmutable入力とする別の補完実行にする。

### 11.1 再利用するもの

92 signal、B3全84組、原登録task一覧、完了449 groupと902 wrapper、各groupのイベント・seed・compiler・両軸・terminalを固定する。旧pilotの標本は今回のfresh探索へ混ぜない。保存値を新たに測り直した値と表現しない。

### 11.2 完了groupの受理単位

worker terminalがTASK_COMPLETE、task identityが登録表と一致、両primary軸と必要なordinary感度が存在、event digestが同じ、source/compiler/input/seedが追跡可能、保存hash整合、必要なvalidationが存在するgroupだけを完了再利用集合に入れる。保存一fileがあることだけでは完了としない。

signal側も、全92 ID・precision field・input/sourceの照合を経て再利用する。数値計算を再実行して同一性を作るのではなく、旧sourceで取得した事実をそのまま入力来歴へ結ぶ。

### 11.3 未完了group

B1 S4 q8と中断したB2 groupは、保存event等があってもcost完了ではない。補完では、元task・seed・lawのまま処理し、新出力に新実行identityを付ける。部分recordを推定で完成させず、新seedで都合のよいtrajectoryに差し替えない。

未完了event列を再利用するか、元seedから決定論的に再生成するかは、新契約に明示する。いずれも旧runの未保存値を復元したという主張にはしない。64-bit seed等は型・値を保ち、丸めた表示値を実行入力に使わない。

## 12. 補完する計算量と順序

元592 groupのうち449を受理できれば、探索の未完了は143 groupである。登録ordinary感度4 wrapperは完了側に含まれているため、未完了探索は通常primary二軸の286 wrapperとなる。[R01,R03,R07]

| 内容 | 新たに必要となる範囲 |
|---|---:|
| signal再計算 | 原則0 |
| B3 normalization再計算 | 0 |
| 未完了探索 | 143 group / 286 primary wrapper |
| 独立確認 | 最大8構成×32 trajectory×2軸 = 512 wrapper |
| 補完＋確認の新wrapper | 最大798（新しい技術検査は別計上） |
| 旧902と統合した保存wrapper | 最大1700（確認unionが8のとき） |

これは固定taskに基づく予定取得数であり、旧失敗taskで未保存のcompile attemptを0回と数えたものではない。探索最大596ではなく、今回登録済み592を使う点が重要である。最大1708から902を引いて残件を推測するのではない。

推奨する実行順序は、メモリ修正後のB1 S4 q8を補完taskとして先に処理し、同じ条件で保存まで進めることを確認した後、残る元登録taskを処理する方法である。これはそのtaskを余分に一回計算する提案ではなく、補完順序を変える提案である。失敗を早く見つけ、他のworkerが途中で巻き添え停止する機会を減らせる。

失敗taskの成功を待たず他の不足taskを取得する回復方式をCodexが設計する場合も、失敗を明示し、全探索が揃う前の選抜に進まないことを維持する。実行順でseedや候補集合を変えない。

## 13. 独立確認を今の部分結果から始めてはいけない理由

既存`resource_row()`は、両軸のrecordがあることを検査するが、必ずB2のn=8が揃っていることをその関数だけで保証する設計ではない。元runnerはbatch完了後に呼ぶことで登録条件を守っている。新しい補完経路でpartial cost_rowsを直接渡すと、q8 B2の2標本が完全探索の8標本と並んで選抜へ入るおそれがある。[S02,S04]

したがって新経路では、元登録592 groupと受理済みgroupの集合一致、候補別replica0–7、両軸、重複なしを確認してからresource mapと`confirmation_ids`を呼ぶ。

確認候補は元規則のまま、各精度B2上位2・ID辞書順tie・union最大8とする。選択集合と探索入力hashを記録した後、別seed namespaceの32 fresh trajectoryを取得する。確認値を見て候補を追加せず、探索8と確認32をpoolして40標本の確認と呼ばない。

今回示したq8 p5の有望さも、選抜規則を迂回してこの候補だけを確認する理由にはしない。全探索が揃った結果、別のr/K/prefixが選ばれれば、その結果を受け入れる。

## 14. 代替方針の比較

| 進路 | 利点 | 問題 | 判断 |
|---|---|---|---|
| 全92＋全costを最初からやり直す | 単一runに見せやすい | 不要な再計算、履歴の誤解 | 採らない |
| 取得済み部分で最終順位を決める | 追加計算不要 | q8・強い対照の非ランダム欠測 | 採らない |
| q8やS4を除外し軽い比較に変更 | 完走しやすい | 科学的な比較範囲の弱化 | 第一案にしない |
| 欠測だけ補完し既定確認へ進む | 計画と証拠を最も保つ | 再利用検証・メモリ対処が必要 | **採用推奨** |
| 保存値から有効な費用下界で一部除外 | 取得削減の余地 | 下界の正当性が未証明 | 任意の補助案 |
| B3/Track B/H8へ主線を移す | 別の可能性を見る | 現在の問いが未解決のまま拡張 | 今は採らない |

もしメモリ対処を施しても同じ比較を現実的に完成できない場合は、技術的実行可能性を明示した限定mapとして閉じる選択肢がある。その場合でも、未取得対照に対するPRの優位を結論しない。これは将来の条件分岐であり、現時点で研究を縮小すべきという判定ではない。

## 15. 研究方針・論文化の着地点

今回の保存値は、「qを大きくするとbiasが下がる」という単独の事実だけでなく、**最小bias、最小回路、最小十分shot、最小測定込み費用が一致しない**という研究の中心を具体化している。

B2 p5/r1/K2のq4→q8は、回路が長くてもstatistical headroomが改善して資源量を下げる可能性を示した。S4 q2→q4は、bias低下が追加回路費用を補わない点比較を示した。このような機序と、その成立する有限条件を説明する方向に価値がある。

しかし現時点で独立論文の十分性、新規性、PRの一般優位を確定しない。本レビューでは新しい網羅的先行研究調査を行っていない。過去レビューで区別した既存PR資源見積もりやcontrol改善の既知性を覆すものではない。

当面の着地点は、H6 developmentにおいて同一task・同一実装条件で費用mapと独立確認を完成させ、何が資源差を決めたかを説明すること。続いてH8に持ち込む仮説・candidate policy・予測を固定してから、独立のサイズ確認へ進む。

Track Bは参照に留める。今回のMemoryErrorを機に新しいreturn集約event lawを組み込むと、元の比較task・normalization・sampling・wrapperの対応が変わる。統合を補完実行の前提にしない。

## 16. 次の担当・委任範囲・戻る条件

### 16.1 Codexにまとめて任せる作業

- sourceを保全した別versionで、失敗区間の軽量化・bounded telemetry・必要な資源政策を準備する。
- 実行時の数値・signal・compiler・候補・seed・shot規則を維持していることを確認する。
- 保存92 signal、84 B3組、449 cost groupの受理manifestと、143未完了taskを固定する。
- 科学入力なしのsynthetic/process検証、型・hash・重複・片軸欠測・誤った成功record拒否を検査する。
- 新source、plan、入力証拠hash、出力namespace、資源とSTOP規則をcommit/push/remote確認する。
- 新しい明示実行認可後にのみ補完と既定確認を実施し、完了・STOPにかかわらず公開して停止する。

細かなmodule構成、memory profiling手順、cache/reference寿命、process配置はCodexの技術裁量でよい。科学的な条件を保つ通常修正やtest追加のたびに同じ研究レビューを繰り返す必要はない。

### 16.2 GPTへ早期に戻す条件

数値的矛盾、入力/target/estimatorの実質変更、保存記録のidentity不一致、同じcompiler費用で補完できず比較条件を変える必要、確認規則を変える必要、候補削除による主張範囲の変更が生じた場合はGPTへ戻す。

通常のメモリ予算再設計は、その値と計算範囲を新しい実行認可に明示すればよい。本レビューの科学判断を理由に、上限を実行中に変更したり、旧grantを再使用したりしない。

### 16.3 次の主要科学レビュー

全探索と独立確認の結果、未観測massと標本精度、数値allowanceの感度、メモリ対処後の実装整合性が揃った時点で、H6で支持される資源的主張と、H8へ持ち込む独立検証を評価する。現在はH8を開かない。

## 17. 最終判定

**研究を練り直すより、既存証拠を再利用し、固定比較を完成させる。**

92 signalと完了449 cost groupは保持する。残り143探索groupは、新しい補完契約の下で取得する方針を推奨する。S4 q8を無条件に削除せず、q8 B2の有望な少数標本も勝者認定へ昇格させない。元の選抜・独立32 trajectory・未観測order診断を維持する。

本レビューは完了。次はCodexによる限定修正・再利用manifest・補完実行準備である。**科学実行そのものは、固定した新しい対象と資源条件への明示的な認可を待つ。**

---

## 付録A. 本レビューの算術検算

添付`checks.py`は7 signalの保存fieldと12 cost groupのRZを転記したデータから、N、G、比率、費用閾値、次数0だけの確率を計算する。原repositoryのmoduleをimportせず、ネットワーク接続・科学runner呼出しは含まない。

7件のepsilon=0.001軸別Nが保存整数に一致した。これをrepository test件数に足したり、全92候補の独立再現と呼んだりしない。7件以外の境界安定性は未評価である。

費用欄が未抽出のp10/p15 q4について、添付JSONのnullは本レビューの転記subsetに費用を入れていないことを示す。**元runの費用が欠測であるという意味ではない。** S4 q8のnullだけは今回の実行で費用が未保存だったことに対応する。

### A.1 四精度の選択例（局所診断・全候補の最良表ではない）

| epsilon_sig | S4 q2 | S4 q4 | B2 p5/q4/r1/K2（n8） | B2 p5/q8/r1/K2（n2） |
|---:|---:|---:|---:|---:|
| 0.05 | 3.13263917e+09 | 6.19502978e+09 | 8.84089088e+08 | 1.50606474e+09 |
| 0.01 | 8.13038615e+10 | 1.55228939e+11 | 2.46429521e+10 | 3.86477933e+10 |
| 0.005 | 3.41213531e+11 | 6.22751042e+11 | 1.14145504e+11 | 1.59823120e+11 |
| 0.001 | 1.32988590e+13 | 1.59443422e+13 | 3.11603905e+13 | 5.35583782e+12 |

B2 q8は2標本、B2 q4は8標本、S4は取得済みdeterministic費用。表の相対関係から未検査のq1/q2のPR、他prefix/r/K、S2/B0を排除しない。

### A.2 主な比率と閾値

| 計算 | 値 |
|---|---:|
| q8 B2点G / S4 q2点G（0.001） | 0.402729123874 |
| q8 B2 N / S4 q2 N（0.001） | 0.955407412647 |
| q8 B2 RZ標本平均 / S4 q2 RZ | 0.421526061597 |
| q8 B2点G / q4 B2点G（0.001） | 0.171879675824 |
| S4 q4点G / S4 q2点G（0.001） | 1.19892558076 |
| S4 q4→q8のN減少率（0.001） | 0.0275034211927 |
| S4 q8 shot-weighted費用の対S4 q2損益分岐 | 378654.840839 |
| S4 q8 shot-weighted費用の対S4 q4損益分岐 | 453978.974961 |
| S4 q8 shot-weighted費用の対q8 B2点値損益分岐 | 152495.332302 |
| 7候補のN相対変化最大（u等10倍、0.001） | 1.15805888041e-05 |

## 付録B. 出典・固定identity

以下は本書の[R/C/S/D/W/E番号]の参照先である。取得・観測のscopeは§1を参照する。source行番号はrefで変わり得るため、固定commitとpathを主identityにした。

| ID | 資料 | 固定ref |
|---|---|---|
| R01 | [結果・監査索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/docs/research/track_a_h6_matched_accuracy_result_v2.md) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| R02 | [原terminal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/execution_terminal.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| R03 | [欠測・coverage](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/missing_coverage_v2.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| R04 | [公開後のB3件数訂正・終了記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/post_publication_observation_v2.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| R05 | [実行観測・RSS記録のscope](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/execution_observation_v2.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| R06 | [静的保存監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/static_identity_audit_v2.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| R07 | [探索task登録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/exploration_tasks.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| R08 | [全signal summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/signal_summary.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| R09 | [B1 S4 q8 worker terminal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0447/worker_terminal.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| R10 | [B1 S4 q8 control・測定検査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0447/validation_symmetric_directional.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| R11 | [B1 S4 q8 worker claim](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0447/worker_claim.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| R12 | [並行していた未完了B2 task](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0450/task_binding.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| R13 | [原inventory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/execution_inventory.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| R14 | [B3有限normalization診断](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/b3_normalization_diagnostic.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| C01 | [v1科学契約（v2で維持）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/docs/research/track_a_h6_matched_accuracy_execution_contract_v1.md) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| C02 | [v2実行準備契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/docs/research/track_a_h6_matched_accuracy_execution_contract_v2.md) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| S01 | [数値・cost port](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54986a9f54515c8c2289e38fa16fd81e2b6008c2/src/trottertracks/resource_applicability/h6_matched_port_v1.py) | `54986a9f54515c8c2289e38fa16fd81e2b6008c2` |
| S02 | [会計・選抜・seed・order診断](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54986a9f54515c8c2289e38fa16fd81e2b6008c2/src/trottertracks/resource_applicability/h6_matched_accounting_v1.py) | `54986a9f54515c8c2289e38fa16fd81e2b6008c2` |
| S03 | [数値allowanceを伴うshot会計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54986a9f54515c8c2289e38fa16fd81e2b6008c2/src/trottertracks/resource_applicability/ax2b_numerical_accounting.py) | `54986a9f54515c8c2289e38fa16fd81e2b6008c2` |
| S04 | [v2 supervisor/worker](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54986a9f54515c8c2289e38fa16fd81e2b6008c2/src/trottertracks/resource_applicability/h6_matched_execution_v2.py) | `54986a9f54515c8c2289e38fa16fd81e2b6008c2` |
| S05 | [既存streaming fingerprint](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54986a9f54515c8c2289e38fa16fd81e2b6008c2/src/trottertracks/resource_applicability/ax2b_stream_fingerprint_v5.py) | `54986a9f54515c8c2289e38fa16fd81e2b6008c2` |
| S06 | [transpile・gate count・depth計測](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54986a9f54515c8c2289e38fa16fd81e2b6008c2/src/trotterlib/rte_compiled_cost.py) | `54986a9f54515c8c2289e38fa16fd81e2b6008c2` |
| D20 | [B1_p19_q2_4th signal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/B1_p19_q2_4th_signal.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| D21 | [B1_p19_q4_4th signal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/B1_p19_q4_4th_signal.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| D22 | [B1_p19_q8_4th signal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/B1_p19_q8_4th_signal.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| D23 | [B2_p5_q4_2nd_r1_K2 signal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/B2_p5_q4_2nd_r1_K2_signal.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| D24 | [B2_p10_q4_2nd_r1_K2 signal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/B2_p10_q4_2nd_r1_K2_signal.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| D25 | [B2_p15_q4_2nd_r1_K2 signal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/B2_p15_q4_2nd_r1_K2_signal.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| D26 | [B2_p5_q8_2nd_r1_K2 signal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/B2_p5_q8_2nd_r1_K2_signal.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| W151 | [exploration group 151 cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0151/cost_result.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| W299 | [exploration group 299 cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0299/cost_result.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| W300 | [exploration group 300 cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0300/cost_result.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| W301 | [exploration group 301 cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0301/cost_result.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| W302 | [exploration group 302 cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0302/cost_result.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| W303 | [exploration group 303 cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0303/cost_result.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| W304 | [exploration group 304 cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0304/cost_result.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| W305 | [exploration group 305 cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0305/cost_result.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| W306 | [exploration group 306 cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0306/cost_result.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| W307 | [exploration group 307 cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0307/cost_result.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| W448 | [exploration group 448 cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0448/cost_result.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| W449 | [exploration group 449 cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0449/cost_result.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |
| E01 | [B2 p5/q8/r1/K2 replica0 event law](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/cost_exploration_0448/trajectory.json) | `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb` |

[X01] Python 3.11 `resource`公式documentation（RLIMIT_ASの意味のみ参照）。特定ホストのOOM原因を証明する資料ではない。
https://docs.python.org/3.11/library/resource.html

## 付録C. 判断の変更履歴

前回レビューは、H6のprecision-matched準備・実行を進める方針だった。今回、その方針自体は維持する。新しく決めたのは、実行途中STOPを受け、完了済み証拠を再利用する補完契約を採用する方向と、q8側の非ランダム欠測を残したまま選抜へ進まないことである。

既存のno-resume規則を後から削除したのではなく、旧runを停止したまま、新たな一回認可の範囲を定める。phase/total wall上限なしの利用者方針も維持する。H6の新rawを追加取得したとは報告しない。
