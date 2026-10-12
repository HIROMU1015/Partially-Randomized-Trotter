# Track B：DF三次補正 E/T会計結果の科学的レビュー

- 作成日：2026-10-12 JST
- 開始承認：利用者の「レビューを開始して」
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- レビュー対象公開commit：`901cafcd594b0399b8638ee1d4ea612bb56fc69e`
- Branch：`track-b-df-correction-et-saved-recovery-v2-20261012`
- 元科学source S7：`74682c40d0cdcca92d244a9edf4bb2852361eef5`
- 元科学実行authorization A7：`84562f8c5af9249a902662324200456498a83d15`
- 元科学result R7：`ad4b4a334171cc39053c0804876a63c93093bfa3`
- 保存値復元source S9：`3afef9e418a8d2173a2780f5534404390bfbe6fb`
- 保存値復元authorization A9：`eea9c497421c5dcfc541dcaa508f607112dbd0d5`
- 状態：**科学的レビュー完了。現行generic whole実装を新規PRアルゴリズムの主線として拡張することは推奨しない。取得済みの限定結果を保持して、この検証系列を区切る。**
- 重要な限定：**PR全体でBがAを支配すること、構造を利用したwhole実装が不可能であること、Track B全体が終了したことを証明・判定したものではない。**
- 実行：本レビューでは登録行列の生成、simulation、compile、T合成、runner再実行、追加テストは行っていない。保存スカラーの表示用丸めと保存整数の比のみを算術処理した。

## 0. 判断の要約

今回の結果は、単なる技術的な正常終了報告より先へ進んでいる。旧v7は保存段階で停止したままだが、native22件とT key39件の保存結果を用い、元のpost-T会計を別の保存値復元として完了している。元runを成功に書き換えず、復元の来歴を分けている点は適切である。[R1, R5, R6]

科学的に認める結果は、**同じ局所入力を使う二つの補正実装について、3条件で、取得済み合成sequenceを保持する固定加法計画のA側completion下界が、B側完全計画のworst-event T費用を上回った**ことである。Aの全T費用を埋めなくても、この限定不等式は成立する。[R2–R4]

ただし、Aの受理時平均は指数回転、Bはweight付きの線形補正であり、有限stepで同じ平均を実装していない。Bのbranch回転を高精度に合成できることは、Bのweighted平均がAの指数回転に同じ精度で一致するという意味ではない。したがって、今回のT分離をそのままPR総費用の勝敗へ読み替えてはならない。[R7–R10]

推奨は次のとおり。

1. Aの全T合成を埋める作業、4行目のphase対応だけを完成させる作業、generic compilerの追加探索を、現系列の必須継続作業にしない。
2. 現generic whole方式を新規PRアルゴリズムとして採択しない。これは一般的な不可能性・全PR非優位の結論ではなく、得られた証拠と残る課題を比較した研究資源配分の判断である。
3. 数理構成、最大6軌道への縮約という既存の条件付き結果、Bへの同情報アクセス、固定計画の局所費用分離、未解決の総費用を一緒に記録する。
4. Bを「勝った新手法」として自動採択しない。既知pairingとの関係、同入力・同信号精度での全費用、非補正PRや他手法との比較は別途未解決である。
5. 次にCodexへ任せるのは、既存成果とこの判断の文書・証拠索引への反映程度とし、追加科学計算を開始しない。同じ結論を確認するための詳細レビューを繰り返さない。

## 1. 研究全体における位置付け

### 1.1 本来の目的と、この候補の狙い

Track Bは、PR内部アルゴリズムの変更により、物理的な時間発展・複素信号推定で必要となる量子資源を削減し、研究として区別できる貢献を作ることを目指してきた。この系列では、ordered DF partial-S2の三次誤差を補正し、決定論sweepの回数を減らせるかを狙った。

候補Aの構成上の特徴は、三重交換子をwholeのまま局所指数回転として実装し、低確率で挿入することにある。既存設計では、指定されたDF primitiveクラスについて、一つの交換子が最大6軌道の部分空間へ閉じる構成を導き、理想sampling・正時間・P5以上等の条件の下で平均演算子の精度を評価した。これらは既存監査から継承する結果であり、本レビューで一般証明をやり直したものではない。[R12]

一方、同じ局所行列は、Bの局所Pauli分解にも利用できる。局所性があることはAだけの利点ではない。そこで、whole指数回転の実装費用と、Pauli pairingのweight・biasを比較することが今回の問いになった。[R7, R10, R12]

### 1.2 これまでの技術問題と、今回の研究判断を分ける

これまでfingerprint、自動生成gate名、strict phase会計、結果JSONの取得性などで停止が生じた。しかし、それらの問題が解消したことは、アルゴリズムの資源優位を示すものではない。

逆に、技術的な停止があったことだけで、保存済みの妥当な数値を無価値と扱う必要もない。今回は「正しく取得された範囲の結果をどう使うか」と「次へ投資する理由があるか」を判断する。

G10の一般return集約系列や、それ以前の他候補の終了判断を、この結果だけで覆さない。親リポジトリの他研究全体についての新しい横断調査は今回実施しておらず、それらの最新優劣を本レビューの根拠にしていない。

## 2. 使用した証拠と取得範囲

GitHub接続を通じ、固定commitのhandoff、4条件のexact表、復元会計snapshot、E-r3／r6の原snapshot、保存値監査の関連範囲、S7のT completion実装、v7のnative・phase受理実装、復元source、旧CD保存summaryを確認した。[R1–R11]

| 証拠 | 本レビューでの扱い |
|---|---|
| 4条件のexact会計表 | 全4行を読み、completion sourceと対応を確認 |
| 復元T accountingのtyped snapshot | 4行、inventory binding、未知辞書への参照、result identityの構造を確認 |
| E-r3／r6 snapshot | 条件付きweight・moment、linearization上界、frame/Pauli proxyの情報、native参照を確認 |
| 保存値監査815 PASS | 記録と主要な照合対象を確認。815項目を独立再実行したわけではない |
| S7のt_completion/local_t_separation | 元科学sourceの固定commitから読み、最新公開側とblob一致を確認 |
| recovery source | 保存値だけを使うAST抽出・入力来歴・結果の別保存を確認 |
| 全731 remote再取得 | Codexの保存receiptに基づく。GPTによる731ファイルの独立byte照合ではない |
| 大きいnative IR／未知key辞書の全chunk | 全量を再構成・再計数していない。保存監査に依拠する |

取得できるdescriptorやindexがあることと、全rawを再検証したことは区別した。原raw全量の再合成・行列検算を、科学レビューのために無断実行していない。

公開branchのHEADが指定commitと一致することは確認した。現在の利用者ローカルworktreeを直接観測したわけではなく、cleanの記録はCodexのreceiptに依拠する。

## 3. process statusと研究証拠のstatusを分離する

### 3.1 元v7は成功へ再分類しない

元v7の停止理由は、native／T keyの取得後に発生した `ValueError: Review file exceeds 512 KiB` である。原terminalは `TECHNICAL_INCONCLUSIVE` のまま残されている。保存値復元側のstatusは `SAVED_E_T_ACCOUNTING_COMPLETE` であり、別の実行履歴である。[R1, R6]

復元sourceは、原terminalの完了stage、元source・authorization、inventoryのbinding、native保存値、各T jobのidentityを確認している。その上で元runnerの `sequences={}` から `T_accounting` 保存までのASTを使い、出力先だけをcaptureする処理である。[R5]

復元前に会計JSONが存在しなかったため、**失われた元会計JSONのbytesと一致した**とは言えない。妥当性の根拠は、元AST、保存入力の同一性、別のcount照合、復元したlogical値のlossless保存である。復元logical accountingのidentityは次の通り。[R1, R3]

- bytes：1,135,179
- SHA256：`b9a6d6aa684f4df4808c1b59d33782cd83f36afad1d510dec5c89a758b0f6ad4`

### 3.2 この復元会計を科学レビューに使うことは支持する

これは、途中までの都合のよい候補だけを抜き出した比較ではなく、固定したnative22件・39 keyの保存結果を元の会計規則で扱ったものとして報告されている。sourceもその状態を検査している。[R1, R5]

したがって、本レビューでは復元会計を、限定scopeの研究証拠として使用する。ただし、独立再現実験、追加sample、元one-shotの成功という意味は与えない。

39 key評価は、backend合成33件とClifford fast path6件を含む。「39回全てgridsynthを実行した」とは表現しない。さらに最新technical continuationでもv6/v7の複数epochがあり、native合計32件が記録されている。v7単独の22件とcampaign全体を混同しない。[R1]

## 4. 局所T費用の結果

### 4.1 原表

| E modes | controlled | A固定計画completion下界 | A actual total | B完全計画worst-event T | B条件付き期待T | A未知RZ出現数 |
|---:|:---:|---:|---|---:|---:|---:|
| 3 | false | 576 | 未確定 | 112 | 112 | 57 |
| 3 | true | 692 | 未確定 | 236 | 236 | 179 |
| 6 | false | 576 | 未確定 | 116 | 116 | 5,519 |
| 6 | true | 未確定 | 未確定 | 232 | 232 | 11,622 |

出典：[R2, R3]。T/T†は保存sequenceの文字countであり、RZ数からの平均係数換算ではない。

`FIXED_PLAN_LOCAL_T_SEPARATION` は最初の3行でtrue、4行目でfalse。この4行は2個の固定synthetic入力とcontrol条件の組合せであり、独立な4分子の検証や成功率75%ではない。

### 4.2 Aの全Tがなくても、限定不等式は成立する

取得済みkey集合を K、各keyの出現回数を m_k、保存sequenceのT数を t_k とする。取得済みsequenceを保持し、gate間の最適化・cancellationを行わない固定加法計画なら、

\[
T_A^{\mathrm{completion}}
=\sum_{k\in K}m_kt_k
 +\sum_{k\notin K}m_kt_k^{\mathrm{future}}
\ge L_A:=\sum_{k\in K}m_kt_k.
\]

未知費用を0と推定したのではなく、非負性だけを使う。sourceはsequence/hash/key/errorの対応を確認し、元IR上の出現回数で加算している。[R4]

Bの固定計画について全対象eventのT費用が取得されていれば、worst-event上界 U_B と比較できる。今回の3行では L_A > U_B である。保存整数の比は次の通り。

- r3 uncontrolled：576/112 = 36/7 ≈ 5.142857
- r3 controlled：692/236 = 173/59 ≈ 2.932203
- r6 uncontrolled：576/116 = 144/29 ≈ 4.965517

これらは**固定された受理補正の局所実装費用に関するbound比**であり、PR高速化率、shot込みの改善率、量子実機の実行時間比ではない。

追加のT合成で未知分を埋めても、同じ取得済みsequenceを保持する計画では既知下界は減らない。従って、この3行の大小関係を確定する目的で全T合成を追加する情報価値は低い。

### 4.3 最適費用の下界ではない

以下を許すと、別の計画になる。

- 取得済みsequenceを別の短いsequenceへ置き換える。
- gate間のcancellationや全回路最適化を行う。
- 別のwhole分解、粒子数構造を使う分解、別precision allocationを採用する。
- coherent controlやphaseの実装契約を変更する。

従って「whole方式は最低576 Tを要する」「全てのwhole算法がBに負ける」とは言えない。固定実装の費用screeningとしては有効だが、最適compilerや算法クラスに対する不可能性結果ではない。[R4]

## 5. r6 controlledのphase問題

v6では、raw phaseを保持した回路に対し、実際には行っていないnearest-W置換の距離を加算して停止した。v7はその未実施置換距離をnative誤差に入れない一方、raw phaseそのものとstrict Operator検査を保持している。[R1, R9]

この修正は、phaseを黙って捨てたり、許容誤差を緩めたりする処理ではない。sourceではunsupported phaseについて、

- raw phaseをIRに保持する。
- nearest-W距離は診断値として保持する。
- 未実施置換の誤差だけをnative ledgerへ加算しない。
- scalar phaseの実装費用はunknownのままとする。

という扱いをしている。[R9]

従って、native数値guardの通過と、Clifford+Tの全実装完了は分ける。r6 controlledは後者が未解決で、sourceはAのcompletion下界もNoneを返す。保存されたfalseをB非優位やA優位へ翻訳しない。[R2, R4]

この未解決性を埋めるだけのために新しいphase方式や追加合成を開始することは推奨しない。それが必要になるのは、その情報が継続判断を変える具体的な研究課題がある場合である。

厳密phase契約の未対応は、任意の実装で物理的に不可能という証明ではない。一方、system global phaseとcontrolled relative phaseを混同して無料扱いしてもならない。今回、新しいphase conventionやclassical phase補正は採択していない。

## 6. 最重要の限定：A/Bは有限stepで同じ平均を実装しない

### 6.1 固定した一つの補正項

以下の T はT gateではなく、局所Hermitian三重交換子を表す。理想化した説明では、

\[
T=\sum_j\beta_jP_j,\qquad \ell=\sum_j|\beta_j|,\qquad u=1/128.
\]

Aは、受理時に

\[
V_A=e^{iuT},\qquad W_A=1
\]

を使う。Bは

\[
p_j=|\beta_j|/\ell,\quad
V_{B,j}=e^{i\,\mathrm{sgn}(\beta_j)\arctan(u\ell)P_j},\quad
w=\sqrt{1+(u\ell)^2}
\]

を使うため、

\[
\sum_j p_jwV_{B,j}=I+iuT.
\]

これはPauliの P_j²=I による恒等式であり、NCCのidentity/Pauli pairingと対応する。[R7, R8, P1]

実装Bでは計算用Pauli proxy、48-bit確率、weight丸め、angle/synthesis誤差を別に計上している。上式を、実装された有限精度平均の誤差ゼロという主張にしない。[R7, R8]

### 6.2 gate合成精度1e-6と、補正の近似誤差は別

BのPauli回転を精度1e-6で実装できても、weighted平均がAの指数回転へ精度1e-6で一致するとは限らない。

保存されたlinearization上界は、両E入力について

\[
\|e^{iuT}-(I+iuT)\|
\le e^{u\|T\|}-1-u\|T\|
\le e^{1/32}-1-1/32
\approx 4.934075\times10^{-4}.
\]

最終値は保存された有理数上界の表示用丸めであり、今回新たに行列誤差を測定した値ではない。[R7, R8]

この上界が1e-6を超えることは、Bの実誤差がその値である証明でも、BがPR最終契約を満たせない証明でもない。Bは外側coinと三次補正全体に組み込んで評価する方法であり、Eの一回の受理eventは、その全体ではない。

**同じ精度の任意角回転を合成したことと、同じ最終物理問題を同じ精度で解いたことを区別する。**

### 6.3 外側coinを戻すと、別のbias–weight交換になる

固定したtuple分布 q_tau と外側確率 eta に対して、理想化した半補正は

\[
\bar C_A=(1-\eta)I+\eta\sum_\tau q_\tau e^{iuT_\tau},
\qquad
\bar C_B=I+i\eta u\sum_\tau q_\tau T_\tau.
\]

小さいuでは、

\[
\bar C_A-\bar C_B
=-\frac{\eta u^2}{2}\sum_\tau q_\tau T_\tau^2+O(\eta u^3).
\]

これは登録式の展開による解釈で、数値比較の新しい判定式として採択したものではない。Aが個々の指数回転を使うことだけで、補正全体・PR全体がBより正確とは言えない。Aは正確率のunitary平均による近似biasを持ち、Bはweighted線形平均による別の残差とsampling負担を持つ。

## 7. sampling負担と総費用

### 7.1 E保存値から読める条件付き量

次の表は各E入力で補正が受理された一回についての値であり、PR trajectory全体のnormalizationではない。保存有理数を丸めて表示した。[R7, R8]

| E modes | 計算用ell | B weight | B second moment | A weight / second moment |
|---:|---:|---:|---:|---:|
| 3 | 7.824838972 | 1.001866791 | 1.003737067 | 1 / 1 |
| 6 | 29.927108694 | 1.026968855 | 1.054665029 | 1 / 1 |

Bの追加weightがこの受理eventで比較的小さいことは、Aのweight1だけを強い資源優位の根拠にしにくくする。しかし、この表からBの全shot負担が0.37%や5.47%増えると結論することはできない。

理想的な独立slotがm個あり、外側coinを含めると、一例として補正weightの二次moment因子は

\[
[1+\eta u^2\mathbb E_q(\ell_\tau^2)]^m
\]

となる。実際には、元RTEのnormalization、全range、有限bit分布、残余統計幅、confidenceが必要になる。Eには登録されたPR outer rateもPR shot数もない。[R7, R8, R10]

### 7.2 CDとEを混ぜない

CDは別の固定3-mode/3-block入力であり、Eのr3と同じdimensionでも別fixtureである。CDの保存結果は、条件付き精度・十分shot予算・記号費用である。新しいE単価をCDの補正頻度へ掛けて実測PR費用にする根拠はない。[R10, R11]

共通費用を C_x=h_x+C_base,x、補正費用を c_i,x と書けば、既存会計は

\[
G_{i,x}=2N_i(C_x+c_{i,x}).
\]

従って

\[
\frac{G_{A,x}}{G_{B,x}}
=\frac{N_A}{N_B}\frac{C_x+c_{A,x}}{C_x+c_{B,x}}.
\]

局所補正の単価だけでは、Nや共通費用を決められない。多数の共通gate、状態準備、embeddingが支配する場合には、補正回転の削減率は全費用の削減率ではない。

Nも契約上の十分shot予算であり、最小shot数や実測shot数ではない。現データからT-depth・wall時間・fault-tolerant全費用まで判定しない。

### 7.3 今回支持できない主張

- BはAよりPR全体で約3～5倍安い。
- Bは同じ最終信号精度で常に有利である。
- Aの全T費用が不明だから、3条件の局所分離も分からない。
- 4行目のfalseはAの勝ちである。
- 3/4条件の分離は分子一般の75%成功率を示す。
- 保存結果がtechnical recoveryなので、全数値を使えない。

いずれも、記録されたscope・上界・欠測の意味を越えた解釈である。

## 8. 新規性と先行研究

### 8.1 今回の結果が新規性について追加したこと

今回のE/T結果は、実装費用の比較を具体化した。しかし、費用表が得られたこと自体で独立新規性が確定したわけではない。

既存の監査では、三次補正、positiveな係数比例draw、Bernoulli疎化、Gaussian変換、Pauli pairingそれぞれの既知部分が整理されている。[R12]

### 8.2 今回再確認した一次文献

- **NCC／Trotter-LCU**：Zeng et al., arXiv:2212.04566v2。PDFのEq.(111)にidentityとPauli一次項を平方根weightとatan回転へまとめるpairingがある。今回のBの主要な代数操作はこの既知機構と対応する。全NCC protocolと現在のBが同じという主張ではない。[P1]
- **Corrected PF**：Bagherimehrab et al., arXiv:2409.08265v4（2026-07-28改訂）。correctorを加える精度改善自体は既知。同論文のexactly simulatableな二分割の結果を、現在の多block DF・finite RTEへ無条件に移さない。[P2]
- **qDRIFT**：Campbell, arXiv:1811.08017。係数強度比例で指数回転を選ぶ基本機構は既知。channel保証をcurrentのfirst operator momentへそのまま転用しない。[P3]
- **HNCC**：Wang et al., arXiv:2607.11856v2（2026-07-23改訂）。Pauli-rotation channelによるnested-commutator補償を扱う。channel observableの費用を、今回のcoherent signalやDF accessに条件を戻さず比較しない。[P4]
- **PR原論文**：Günther et al., arXiv:2503.05647v2（2026-07-10改訂）。Hamiltonian simulationとphase estimation全体の資源が本来の対象であることを確認した。掲載された他系の改善率は今回の比較へ流用しない。[P5]

NCCはPDF本文の該当式を取得できた。ページ画像の取得は接続エラーで成功していないため、図表の見た目や図からの数値読み取りは根拠にしていない。CPF/HNCCはHTML・一次abstractも確認した。

旧監査が挙げるRandomized THRIFT arXiv:2608.07720は今回のweb再取得ではcache missとなった。対応の詳細は旧監査の記録として保持するに留め、今回独立に本文確認したとは扱わない。この未取得文献一つに本レビューの非採用判断を依存させていない。

### 8.3 残る可能性と、現在不足しているもの

指定DF入力から局所三重交換子を取得する具体的な生成法には、方法論的な価値が残り得る。しかし、同じ局所表現をBも使えるため、局所性だけではwhole指数回転を選ぶ優位を示さない。

現在、production DFへの認証付き接続、basis rotationを含む同入力費用、同最終信号の資源優位、強い既知法とのend-to-end比較は未実証である。今回の費用表は、それらを補ったことにはならない。

同一方法の不存在を網羅的文献調査で証明したものでもない。独立新規性の判断は「確立していない」のままであり、「新規性が絶対にない」とは断定しない。

## 9. 今後の候補を比較する

### A. 残りの全Tを合成する

3条件の固定計画分離を確認するためには不要。全量が分かれば比較の量的な幅は増えるが、generic実装の全量であって最適wholeの下界や全PR費用ではない。**現在は採用しない。**

### B. r6 controlledのphase実装を完成する

実装上は未解決であり、完成すれば4行目の情報を得られる可能性はある。しかし、これだけではCD/Eの別入力問題や新規性は解決しない。**表を4行とも埋めること自体を目的にしない。**

### C. generic compilerを最適化・交換する

今回の費用が最適でないことは認める。とはいえ、別compilerの探索を始めれば、現在の固定計画とは異なる研究になる。構造を利用して何を減らせるかという具体的な機構を先に示すべきで、敗勢の固定実装を救済するために無期限の探索へ移らない。**自動的な継続作業にはしない。**

### D. 同じCD入力へnative全費用を接続する

最終PR費用を比較するには科学的に正当な次の問いである。単なる無意味な追加検証ではない。しかし、これまで数学的な差分の大部分は既知機構と対応し、generic wholeの局所実装は既知対照より高く、同入力への接続とbaseline比較には更なる工数がかかる。

したがって、**「全PR比較が未実施だから、とりあえず必ず続ける」という扱いにはしない。** 現generic wholeを優先して研究資源を投入する根拠は不足していると判断する。総費用での非優位を証明したから止める、という論理ではない。

### E. Bを新しい主手法として採択する

Bは有力な対照として機能したが、NCC型pairingとの既知関係があり、全PRの優位も未実証である。**Aの固定実装より局所的に安いことだけで、新しいB研究を自動開始しない。**

### F. 現成果を保持して、この検証系列を区切る

数理構成、同情報の対照、固定計画の3条件分離、phase/coverageの限界、全費用未確定まで記録する。現在のgeneric wholeを新規アルゴリズムの主線には採択しない。**これを推奨する。**

## 10. 終了・休止判断の意味

今回の判断には二つの水準がある。

**証拠に関する結論**：3条件の局所固定計画T分離を認める。4行目、A全T、PR総費用、一般whole優劣は未確定。

**研究運営に関する結論**：未確定な可能性を全て否定し終えていなくても、現generic wholeの追加計算を優先しない。数学的に成立すること、局所性があること、実装できることだけでは、現在の研究目標に対する継続理由として十分でない。

従って「現系列を区切る」は、whole補正クラスの不可能性証明でも、Bの科学的総合勝利でもない。固定実装系列をさらに拡張する投資判断を行わないという意味である。

また、Track B全体や他のPR研究を全て終了すると判断したものではない。別の具体的構成、入力取得法、保証、native primitiveに基づく差が提示されれば、新しい仮説として検討できる。新しい仮説を今回勝手に採択することはしない。

## 11. 研究成果として残すべき内容

### 支持できるclaim

> 指定されたDF primitive由来の局所補正に対し、同じ局所情報を与えたwhole指数回転実装とcoin-first Pauli pairing実装を比較した。固定generic native分解と指定されたClifford+T加法計画において、3条件でwhole側の取得済みsequenceから得るcompletion下界がPauli側の完全event費用を上回った。平均演算子・weight・線形化biasを区別して会計したが、PR全体の資源優位は未確定である。

### 支持できないclaim

> 少数軌道correctorが新規かつ一般に高効率である。
>
> Pauli方式が同じ最終精度でwhole方式より約3～5倍低コストである。
>
> 6軌道whole回転は原理的に高価である。
>
> 今回の2固定入力から分子一般・PR全体・QPE全体の性能が分かる。

現在の証拠だけで独立論文の主貢献が確立したとは判断しない。数理構成と実装上の限界を残す技術ノートや、他の主張を支える補助資料として整理するのが妥当である。publicationを保証する評価ではない。

## 12. 次の担当・作業・STOP

直近の次担当はCodexでよい。ただし、新しい比較runnerやT合成ではなく、今回のレビューを既存記録へ反映する文書作業とする。

- 既存の設計、独立監査、最新E/T結果、今回の科学判断を結ぶ証拠索引または技術ノートを整理する。
- claim/evidence/limitationを分け、3行の分離と4行目の欠測を保持する。
- 元v7 `TECHNICAL_INCONCLUSIVE` と復元 `SAVED_E_T_ACCOUNTING_COMPLETE` を別々に維持する。
- 旧source、raw結果、marker、authorizationを書き換えない。
- 新しいmatrix、compile、合成、CD計算、分子計算、fixture/grid探索は行わない。
- 記録整理完了だけを理由に、同じ科学レビューをもう一度要求しない。

この文書自体が研究判断の記録であり、記録のためだけに新しい再現ZIPや検証系列を作り直す必要はない。GitHubへの反映は必要な文書だけを対象にする。

新たな実行が必要になる場合には、その科学的目的を別途決め、対応する実行承認を得る。今回の承認はレビューの範囲であり、既存mandatory STOPを解除するものではない。

## 13. 確認の限界と再検討条件

- 815項目、797項目、731 remote取得の全件独立再検証はしていない。
- 全39 synthesis sequenceの行列を再評価していない。
- 全量native IR・Pauli表を別プログラムで再生成していない。
- 同入力PR総費用、production DF、分子、QPE/RPE end-to-endの結果はない。
- 先行研究全体の完全同値性や優先性は決定していない。
- 保存値の一部は条件付きIEEE/BLAS/libmの数値モデルに基づく。無条件の厳密算術certificateへ読み替えない。

これらを明示した上で、現固定計画の局所費用分離と、現在は拡張しないという研究判断を行った。

再検討の根拠となるのは、一般的な「もっと最適化できるかもしれない」ではなく、既存対照へ同じ情報を与えても残る具体的な新機構、native accessの改善、保証または取得計算量の差である。今回その新機構の探索を自動開始してはいない。

## 14. 参照資料

以下のRepository参照は、特記がない限り公開commit `901cafcd594b0399b8638ee1d4ea612bb56fc69e` に固定する。


### リポジトリ一次資料

**[R1] 最新E/T完了handoff**  
[docs/tracks/algorithm_codesign/df_correction_et_completed_saved_accounting_and_gpt_handoff_20261012.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/docs/tracks/algorithm_codesign/df_correction_et_completed_saved_accounting_and_gpt_handoff_20261012.md)  
元v7と復元の状態、実行履歴、4条件会計、未確定事項を確認。

**[R2] 4条件のexact会計表**  
[artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/comparison_rows.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/comparison_rows.json)  
全4行を取得。

**[R3] 復元T accountingのtyped snapshot**  
[artifacts/track_b_df_correction_et_saved_recovery_result/2026-10-12/v2/_snapshots/5661589cb905a7fdd8959ca8efc80f26294042c930179c1ec8b94fd3d9418267.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/artifacts/track_b_df_correction_et_saved_recovery_result/2026-10-12/v2/_snapshots/5661589cb905a7fdd8959ca8efc80f26294042c930179c1ec8b94fd3d9418267.json)  
4行、logical identity、native/T bindingを確認。全unknown blobの独立再構成は未実施。

**[R4] T completion/local separation source**  
[src/trottertracks/algorithm_codesign/df_correction_comparison/native.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/src/trottertracks/algorithm_codesign/df_correction_comparison/native.py)  
t_completion、local_t_separation、固定key選択を確認。元科学S7の同pathでもGit blob一致を確認した。

**[R5] 保存値復元source**  
[src/trottertracks/algorithm_codesign/df_correction_et_saved_recovery/accounting.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/src/trottertracks/algorithm_codesign/df_correction_et_saved_recovery/accounting.py)  
元AST抽出、保存入力・元terminalの検査、別出力へのcaptureを確認。

**[R6] 復元terminal snapshot**  
[artifacts/track_b_df_correction_et_saved_recovery_result/2026-10-12/v2/_snapshots/c7d7e367eb7d49d2a1152693efa08c5080cc0fff2761f6a5e2189b508aa5e7b2.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/artifacts/track_b_df_correction_et_saved_recovery_result/2026-10-12/v2/_snapshots/c7d7e367eb7d49d2a1152693efa08c5080cc0fff2761f6a5e2189b508aa5e7b2.json)  
SAVED_E_T_ACCOUNTING_COMPLETE、旧terminal再分類なし、追加science0。

**[R7] E-r3原result snapshot**  
[artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v4/_snapshots/dc4ba84bfc243069ac1d12f81641131ea5035e8ada7e16515e3611146d97f942.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v4/_snapshots/dc4ba84bfc243069ac1d12f81641131ea5035e8ada7e16515e3611146d97f942.json)  
ell、weight、moment、linearization bound、native参照を確認。

**[R8] E-r6原result snapshot**  
[artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v4/_snapshots/9359174a417d130ad68caee155e854db809c2a53c929da9ccb8a127d19544089.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v4/_snapshots/9359174a417d130ad68caee155e854db809c2a53c929da9ccb8a127d19544089.json)  
同上。logical全native IRの再計数は未実施。

**[R9] v7 raw phase・layered acceptance source**  
[src/trottertracks/algorithm_codesign/df_correction_et_phase_v7/acceptance.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/src/trottertracks/algorithm_codesign/df_correction_et_phase_v7/acceptance.py)  
raw scalar phase保持、未実施置換距離、native IRと費用の結合を確認。

**[R10] v7契約と局所評価source**  
[artifacts/track_b_df_correction_et_phase_preparation/2026-10-12/v7/contract_v7.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/artifacts/track_b_df_correction_et_phase_preparation/2026-10-12/v7/contract_v7.json)  
A/Bの定義、固定入力・条件を確認。下記のnative_evaluationも参照。

**[R11] 旧CD保存summary**  
[artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/CD_saved_summary_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/CD_saved_summary_v1.json)  
conditional envelope、記号費用、signal診断の位置付けを確認。全CD計算を再実行していない。

**[R12] 既存の先行研究・構成比較**  
[docs/tracks/algorithm_codesign/df_sparse_correction_prior_art_comparison_20261011.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/docs/tracks/algorithm_codesign/df_sparse_correction_prior_art_comparison_20261011.md)  
既存設計の位置付けと、既知法への部分的還元を継承。各比較の未確定条件を保持。

**[R13] 保存値監査**  
[artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/saved_output_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/saved_output_audit.json)  
815 PASSとの記録、native IR・T sequence・下界・B期待値など主要検査項目を確認。全件再実行なし。

**[R14] GitHub再取得receipt**  
[artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/publication_receipt.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/publication_receipt.json)  
731ファイル、42,345,218 bytes、同一性確認のreceipt。GPTが全件を独立照合した結果ではない。

**[R15] v7局所native評価source**  
[src/trottertracks/algorithm_codesign/df_correction_et_phase_v7/native_evaluation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/src/trottertracks/algorithm_codesign/df_correction_et_phase_v7/native_evaluation.py)  
A/B branch target、Pauli proxy、weight・linearization上界、strict numeric guard、局所scopeを確認。

**[R16] v7保存sequence検査**  
[src/trottertracks/algorithm_codesign/df_correction_et_phase_v7/accounting.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/901cafcd594b0399b8638ee1d4ea612bb56fc69e/src/trottertracks/algorithm_codesign/df_correction_et_phase_v7/accounting.py)  
sequence/key/hash/T-count/strict-error対応を確認。

元科学source S7のT completion参照：

[native.py @ 74682c40d0cdcca92d244a9edf4bb2852361eef5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/74682c40d0cdcca92d244a9edf4bb2852361eef5/src/trottertracks/algorithm_codesign/df_correction_comparison/native.py)。取得したGit blobは `1e6bb3a6bd4141bac3719905729938e4dfdaf437`。

### 外部一次文献

**[P1] [Pei Zeng et al., Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations](https://arxiv.org/abs/2212.04566v2)**  
v2、2025-03-27改訂。PRX Quantum 6, 010359 (2025)。NCCのidentity/Pauli pairingは本文Eq.(111)付近の解析textで確認。PDF画像の取得は接続側のcacheエラーで失敗したため、図表の画像からの主張は使用していない。

**[P2] [Mohsen Bagherimehrab et al., Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas](https://arxiv.org/abs/2409.08265v4)**  
v4、2026-07-28改訂。PRX Quantum 7, 033018 (2026)。HTML本文・abstractと版情報を確認。

**[P3] [Earl Campbell, A random compiler for fast Hamiltonian simulation](https://arxiv.org/abs/1811.08017v2)**  
v2、2019-06-25改訂。Phys. Rev. Lett. 123, 070503 (2019)。係数比例samplingの既知原理を参照し、論文の他分子のspeedupを本研究へ移植していない。

**[P4] [Xinzhao Wang et al., Trotter error compensation with polylogarithmic precision and nested-commutator scaling without ancillas](https://arxiv.org/abs/2607.11856v2)**  
v2、2026-07-23改訂。channel/observable targetのHNCC。HTML本文と版を確認。同一DF入力・coherent signalの数値比較は行っていない。

**[P5] [Partially Randomized Trotter関連原論文（arXiv:2503.05647）](https://arxiv.org/abs/2503.05647v2)**  
v2、2026-07-10改訂。PRX Quantum 7, 020332 (2026)。PR本来の全体資源評価という背景の参照。本文の全再評価は本レビューの目的ではない。

### 付記：表示用算術と未実施作業

本レビューに付随する `track_b_et_review_20261012/saved_scalar_display.json` は、R7/R8から転記した既存の正確な有理数を表示用に十進数へ変換し、R2の既存整数比を表示しただけの記録である。新しい行列、norm、bias bound、sampling、回路、合成sequence、物理的信号を生成したものではない。表の丸め値を新しいcertificateにしていない。

今回GitHubへcommit・push・source変更はしていない。このMarkdownはチャット側で作成した研究判断記録であり、リポジトリへの反映が完了したと主張しない。
