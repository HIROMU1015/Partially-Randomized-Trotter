# Track B：DF三次補正A/B比較 source v2・契約整合レビュー

- 作成日：2026-10-11 JST
- 開始承認：利用者の「レビューを開始して」
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 対象source S：`b4aba2b698cdc81710dbdbb5769a1686f716ffe2`
- 公開確認HEAD H：`f41cb7c33ee1cded5cef4e1942491e0447b6be69`
- Branch：`track-b-df-correction-comparison-v2-source-preparation-20261011`
- 範囲：既に採用した比較契約v2と、その実装・数値会計・実行／保存経路の整合。研究方針全体、新規性、未実行比較の勝敗を再判定するレビューではない。
- 今回の判断：**比較設計と主要な数式・有限bit会計の実装は支持する。ただし、保存資料の512 KiB取得性とT inventoryのfreeze SHA256結合を補うまで、このSをそのまま本番実行することは推奨しない。**
- 提案ラベル：`SOURCE_REVIEW_COMPLETE / LIMITED_INTEGRITY_REPAIRS_REQUIRED / EXECUTION_NOT_AUTHORIZED`。これは今回のレビュー判断で、repositoryの保存statusを書き換えたものではない。

## 0. 結論と次の担当

次の担当はCodexである。新しい研究案、fixture、精度、gate数上限を増やす作業ではない。今回の二つの保存・来歴管理の修正と、それに直接対応するoff-domain testsをまとめて行い、新しい固定sourceを公開する。

1. 大きな結果ファイルを後続GPTレビューから取得できるよう、512 KiB以下の閲覧用分割資料・索引・原本照合を用意する。内部のraw JSONを保持する方法でもよい。
2. 合成前に作ったT inventoryの**元のSHA256**を、job・worker結果・最終集計へ結び付け、同じinventoryを使ったことを検査する。

これは補正法の数理が反証されたためのSTOPではない。固定Sの公開資料とsourceに明示的にある保存・凍結契約を閉じるための指摘である。本番sourceを一回実行してから修復する必要はない。

修正後はこの差分と回帰testsの確認に限定し、A/Bの研究意義や既存の理論を初めからレビューし直さない。比較本番は、修正後sourceに対する別途の明示的実行承認と、直接子authorizationの固定後に限る。

## 1. 証拠の取得とレビュー方法

GitHub接続から、契約MD、実行計画、機械可読契約、主な実行module、coordinator、test source、focused-test receipt、publication receiptを取得した。[S01–S18]
Remote branchのHEADはHで、親としてSが記録されている。[S19]

今回直接読んだ主な実装は、`comparison.py`、`certification.py`、`digital.py`、`intervals.py`、`angles.py`、`native.py`、`native_evaluation.py`、`guard.py`、`launch.py`、`worker.py`、one-shot coordinator、および再利用される`numerics.py`である。

確認を三層に分けた。

- **source由来の事実**：固定codeに書かれた式、分布、保存field、実行順、guard。
- **今回の静的検討**：その式や制御フローが契約と一致するか、どこに保証の条件が残るか。
- **限定した補助検査**：登録入力を使わないJSON schemaのサイズ確認、単純な確率予算の有理数算術、`claim_job`関数のmockでの検査。

補助検査のコード・JSONは同時提供する。Qiskit/pygridsynthをこの環境で新規installしていない。登録CD/E入力を生成せず、ZIP内の局所NPZも開いていない。登録mean、signal、compile、T合成、分子、量子shot、旧runnerを実行していない。

GitHubのreadは成功したが、runtimeからraw URLを直接取得する試みは接続エラーだった。これはrepository資料の欠落や未pushを意味しない。必要な内容はGitHub connectorから読めた。全208保護fileや全45公開資料を独立にbyte再照合した、という主張はしない。

## 2. 維持する研究契約

CDは3軌道・3ブロックのsynthetic入力による条件付き信号精度、moment、十分shot数、記号的な加法資源会計である。Eは別の3／6軌道入力による局所中心回路の実装screeningである。[S01–S03]

CDの固定値は、係数9/20、7/20、1/4、seed6121107、rank/sign=(1,+),(2,-),(1,-)、t=1、delta=1/8,1/16,1/32、gamma=5/4、kappa=4、s=1、paired cutoff4/P5である。初期状態、axis epsilon=1/20000、各axis/task alpha=1/240も維持する。

Eはseed7101107について最初の4×4 drawを消費し、r3の3 frame、r6の3 frameの順を守る。u=1/128、system/controlの二条件を保つ。`cd_recipe`と`e_recipe`はこの生成順序を実装し、登録recipeへの入口でauthorizationを再確認する。[S04]

重要な限定：

- Eの回路単価をCDの補正頻度に掛けない。
- CDの未取得費用を0にしない。
- 原RTEのnormalizationを落とさない。
- Eの汎用whole分解をwhole方式一般の最小費用や下界としない。
- `T_INCONCLUSIVE`や数値guard失敗を科学的な非優位の証拠にしない。
- 今回の全体statusがCOMPLETEでも、新規アルゴリズム採択やPR総資源の勝者を意味しない。

これらの区別は契約だけでなく、`whole_PR_resource_winner=None`、`PR_measured_total_cost=None`、`CD_PR_total_winner=None`等の出力fieldにも反映されている。[S04,S08,S12,S13]

## 3. 数学的意味論と有限bit実装の照合

### 3.1 AとBの違いを保持している

Aは外側coinとtuple選択の後にwhole回転`exp(i u T_tau)`をweight1で使用する。Bは同じ外側coinとtuple構造を使い、局所Pauli proxyの係数に比例してPauliを選び、atan角の回転と平方根weightを使用する。[S01,S04,S06]

理想係数について

\[
T_\tau=\sum_j\beta_{\tau j}P_j,\quad
\ell_\tau=\sum_j|\beta_{\tau j}|,\quad
\alpha_\tau=\arctan(u\ell_\tau)
\]

とすれば、

\[
\sqrt{1+(u\ell_\tau)^2}
\sum_j\frac{|\beta_{\tau j}|}{\ell_\tau}
 e^{i\operatorname{sgn}(\beta_{\tau j})\alpha_\tau P_j}
=I+iuT_\tau.
\]

v2は有限stepでA/Bのmeanが異なることを維持している。BをweightなしのPauli-qDRIFTへすり替えていない。数値proxyと有限bit化による追加差は別台帳に入れる。

### 3.2 理想frameとPauli proxyのamendment

`frame_guard`はbinary64 recipeの列を厳密数とみなし、それをordered Gram–Schmidtした理想frameを定義する。pivot、rank drop、新seedは使わない。区間で正の列normを確認できない場合は停止する。[S05]

`pauli_table`はfloat行列のHermitian projectionをexact rationalに変換し、trace／parityによる構造的zeroを区別した計算用tableを作る。理想Tとの差をradiusへ戻し、その他の計算上のzeroを理想zeroと断定しない。

この設計を、理想targetを別の簡単なHamiltonianに変更したものとは扱わない。**同じtargetに対する実装近似と、その距離の課金**として明示されているため、既存方針内の数値化として支持する。ただし、これは元の分子DF分解の誤差を保証する処理ではない。

### 3.3 完全lawのL1距離

外側coin etaはexact dyadicで保持される。tupleとconditional Pauliの丸めは正規化した有理数に対する48-bit LRMで行われる。ideal／computational／executedを混同していない。[S06]

例えばtuple分布qと丸め後q_tilde、conditional Pauli分布p_tauと丸め後p_tilde_tauに対して、

\[
\|\eta q p-\eta\widetilde q\widetilde p\|_1
\le \eta\|q-\widetilde q\|_1
+\eta\sum_\tau\widetilde q_\tau
 \|p_\tau-\widetilde p_\tau\|_1.
\]

sourceの`error=eta*qerr`と`eta*qt*conditional_distance`はこの分解に対応する。外側coin不成立や証明済みzero tupleでも、PR baseを捨てずidentity補正を返す。

RTE order確率は、区間midpointを正規化した計算用lawを経由してLRMし、midpoint化とLRMの両方の距離を課金する。ideal remainderの順位が不確定なのに恣意的にtie処理する経路ではない。[S04,S06]

### 3.4 実装weight・moment・range

`slot_mean`はBの実装weightを80-bit dyadic点へ固定し、そのpoint誤差を保存する。`account`は同じexecuted complete lawから、

\[
\mathbb E_{p_{exec}}C,\qquad
\mathbb E_{p_{exec}}W_{exec}^2,\qquad
\max_{p_{exec}>0}|W_{exec}|
\]

を評価する。費用をE[W²C]へ置き換えていない。zero massの枝をrangeへ入れない。[S04,S06]

CDの費用が未取得なので、価格そのものではなく、実装確率とcost variableを保存し、`2N(h+C_base+2n sum p_exec C_event)`という記号式を返す。これは今回の目的に一致する。

理想3-qubit Pauli係数質量の上界32に対し、proxy距離r_Tを使った`32+8*r_T`とweightの点丸めを戻すため、理想32を実装値へ無断で同一視していない。独立drawの条件でslot moment／rangeの積を作り、元RTEのGamma_execを保持している。

### 3.5 finite-step boundとconfidence

`scalar_error_bounds`は前回確認済みの異なるA/B伝播式を実装する。Bについてcontractionを仮定せず、`2e_B+e_B²+e_R(1+e_B)²`を使用する。[S07]

\[
s_{stat}=\epsilon_{axis}-\beta_{algorithmic}
-5\times10^{-6}-5\times10^{-6}-10^{-6}.
\]

最大algorithmic額2.5×10^-5を使ってもs_stat=1.4×10^-5が残る。alphaの和は12/240=0.05である。この算術だけを補助検査で再確認した。登録deltaに対する新しいbiasやshot数は算出していない。

primaryのshot数は実装weightを覆う上向きmoment／rangeと事前に固定した統計幅から計算し、diagnostic signalを使って減らさない。[S04,S06]

## 4. 数値認証の評価と限界

区間scalarとBallの役割分担は理解できる。`intervals.py`はmp/iv両方のdpsを100に設定し、endpointを有理数へ変換し、半径の保存は固定256-bit dyadicで上向き丸めする。[S07]

`Ball`の積・和・scale、frame距離、proxy再構成距離、spectral actionには誤差が明示される。[S05,S17]
一方で全操作が形式検証済みというわけではなく、IEEE binary64、標準のcomplex BLAS forward-error、native Operatorについてはlibm4ulp等のモデルを前提にする。これを無条件のmachine-independent certificateに格上げしない。

今回の読解では、主要な式の不一致や、Bにcontractionを仮定する等の明確な意味論上の反例は確認していない。しかし本番inputの全radius・finite-time margin・runtimeを独立認証したのではない。その成否は、固定条件で実行した結果の一部である。

mpmathの公式文書も、interval arithmeticの包含の考え方を説明する一方、interval機能には実験的・未対応部分があると明記している。[P02] この一般注意だけでsourceを否定するのではなく、今回使う演算と既存のoff-domain controls、および計算モデルを限定して扱う。

## 5. Native／T経路で確認できた点

### 5.1 raw radiansとstrict phase

`Angle.at`はpi、atan、raw-radを明示分岐し、未定義kindを拒否する。前回懸念したraw-radをatanへ誤解釈する流用は修正されている。[S09]

strict guardは位相を最小化せず、目標RZとのFrobenius距離を覆う。`RZ(k*pi/2)=W^(-k)S^k`というstrict Clifford fast pathと、4pi周期を保持する。RZ(pi/4)をphase-freeの1Tへ置き換えない。

合成器は固定version、dps100、seed/key、epsilon/4の要求とstrictな事後guardを使う。pygridsynth2.0.0の公開仕様のradians、dps、seed、up-to-phaseの意味と整合している。[P01] 本環境ではそのruntimeを実行していない。

### 5.2 Bの回路単価

Pauli basisのX/Y/Z内訳をIRに保ち、support weightだけで1Q数を代用していない。controlled回転はancilla相対位相を保持し、固定parity loweringを使う。[S10,S08]

全ラベルの解析IR単価を記録し、代表compileでloweringを確認する設計は、4,096回のcompileを回避しつつ契約した比較を行う方法として支持する。ただし、全系embeddingや状態準備の費用はここから得られない。

### 5.3 部分T下界とcoverage

`select_synthesis_inventory`はB必要角度、phase、A固定位置の順で集合を作り、dedupの空き枠を補充しない。coordinatorもB batchを先行させる。[S10,S12]

`t_completion`は取得済みsequenceのTを**出現回数ごと**に加算する。未取得keyを0で補ってactual_total_Tを出すことはせず、phase、numeric allowance、occurrence cap等が揃わなければ全Aを不明とする。

Aの部分下界とBの最悪event費用の比較は、同じ固定加法completion計画に限った局所分離である。最短合成・別compiler・cross-gate cancellationに対する下界でも、CDの総資源比較でもない。この限定は維持されている。

全T coverageが未確定なことは、それ自体では実行不可の理由ではない。未取得を認識する経路があることが重要である。

## 6. 必須修正R1：512 KiBのレビュー用出力経路

### 6.1 source上の事実

- `pauli_table`は6軌道で4^6=4,096 coefficient recordを作る。[S05]
- `native_fixture`はそれを`Pauli_coefficients`として一つのresult辞書へ入れる。[S08]
- `worker.dump`はその辞書を`json.dump(..., default=str, indent=2)`で単一ファイルへ書く。[S13]
- `guard.py`はaggregate64MiBとlog4MiB等を監視するが、review file512KiBに対応する分割・索引生成・個別チェックは、この経路にはない。[S11]
- 実行計画と機械可読契約はreview files<=512KiBを記載している。[S02,S03]

### 6.2 登録データなしで確定できるサイズ不整合

係数を計算せず、以下の最短fieldだけを4,096件置いたJSONを同じindent=2でserializeした。

```json
{
  "label": "XXXXXX",
  "computational_beta": "1",
  "zero_status": "COMPUTATIONAL_NONZERO",
  "ideal_beta_radius_upper": "0"
}
```

実際には全6文字labelを使う。数値は架空の短い文字列であり、operatorや研究結果を表すものではない。sourceで生成する他のzero_status文字列はこれより長く、実際の有理数文字列も通常もっと長い。

`Pauli_coefficients`のみを持つ最小のtop-level JSONでも、**630,817 bytes**となった。512KiB=524,288 bytesを**106,529 bytes**超える。frame certificate、native inventory、all_B_event_costs等は含めていない。

したがって、実際のE結果の正確なサイズは未計測だが、現在の単一JSON経路が512KiB以下に収まらないことは登録入力を開かずに判断できる。これは量子回路の性能結果ではなく、保存schemaのチェックである。

### 6.3 なぜ実行前に閉じるか

内部計算用のraw JSONが512KiBを超えること自体を禁止する必要はない。しかし、後続の科学的レビューに必要な係数・確率・gate inventoryを、どの小さい資料から完全に取得するかがsource計画にない。既存のreview capを守る経路を後付けでなく用意すべきである。

修正は、raw保持＋閲覧用chunk/indexのどちらでもよい。巨大なtableをsummaryから削除して取得不能にすることや、数値を丸めて原証拠を置き換えることは避ける。分割資料は順序、件数、元bytes・SHA256、chunkのbytes・SHA256、再構成一致を持つ。

本番前の受入確認は数値を作らないmetadata fixtureで足りる。4,096件以上、長い有理数、途中停止prefix、端数chunkを検査する。64MiBのaggregate上限、既存の科学入力、compile/T呼出し上限は変更しない。

## 7. 必須修正R2：T inventoryの元SHA256への結合

### 7.1 現在ある保護

source／contract／authorizationはhashで結合され、one-shot markerと各job-start receiptは排他的に作られる。synthesis jobはその時点のinventoryに含まれるangleかどうかを確認する。これらは有効な保護である。[S12–S14]

### 7.2 足りない部分

coordinatorは合成前に`synthesis_inventory.json`を排他的に作るが、その**作成時bytesのSHA256**をjob payloadやterminal provenanceへ渡していない。[S12]

`claim_job`は現時点のinventoryを読み、angleが一つ含まれるかを確認する。その後のjob hashはjob自身の内容を記録するもので、最初にfreezeしたinventoryのhashではない。[S13]

`verify_child`が照合するmarker SHAは、source、authorization、contractを結ぶmarkerのhashである。後から生成したinventoryを指すhashではない。[S14]

契約の「exclusive artifact + SHA256 before ANY synthesis」について、exclusive artifactの部分は実装されているが、元digestへの結合が不足している。source SのGit hashが正しいことも、実行中に作られたこのfileの固定を代用しない。

### 7.3 mock確認の意味と限界

公開された`claim_job`関数を小さい一時ディレクトリで使った。dummy inventoryを1/7 radから2/7 radへ変更し、jobも現inventoryに一致させると、関数はstart receiptを作る。

これは、同関数が現在のfileへのmembershipを検査し、元digestを検査していないことを示すだけである。実行branchのauthorizationを破った、通常のrunnerが勝手に角度を変更する、あるいは既存結果が改変されたという主張ではない。`verify_child`全体の攻撃実験もしていない。

今回の指摘は安全脆弱性の網羅監査ではなく、sourceと宣言されたfreeze契約の対応である。

### 7.4 最小の修正

freeze時にdigestを計算し、変更不能な受領記録として保存する。jobへ期待digestを渡し、workerが開始前に実際のfile hashを照合し、出力と最終集計にも同じdigestを記録する。親も元digestとの一致を確認する。

元fileの変更、key削除、順序変更、同じpathへの置換をmockで検知するtestsを足す。filenameやschemaの細部はCodexの裁量とし、合成対象集合、順番、seed、precision、上限を変えない。

## 8. 非blockingの注意事項

### 8.1 coordinatorのruntime

coordinatorはnumeric/synthesis両方のruntimeをread-only subprocessで検査するが、そのcoordinator自身の`sys.executable`を同じ関数で照合する呼出しは見当たらない。[S12,S13]

実行指示では契約のnumeric Pythonを明示する必要がある。上の二修正と同時に親の自己runtime照合を加えるのは小さい堅牢化として適切だが、未知のPythonで実行された証拠があるという意味ではない。固定した正しい起動コマンドを使えば、この点だけで研究計画を差し戻す必要はない。

### 8.2 resource guardの主張

50ms sampling、process group停止、RUSAGE_SELF＋reaped children＋live childrenというaccountingは、宣言したモデルに対応している。[S11]
Python公式のRUSAGE_CHILDRENは終了してwaitしたchildを対象とするため、live childrenのCPUを別に見る設計には理由がある。[P03]

一方でゼロovershootのRSS上限ではない。また監視loopは子processが動くphase内で回り、coordinator自身のmetadata処理はboundaryで検査する。親が実行する全処理に別processのwatchdogが常時付いている、という意味の無条件hard capではない。現在の限定されたparent処理と、この限界を明示したモデルで扱う。

identity preflightの各30秒timeoutはmarked scienceとは別である。実行計画にその区別があるため、直ちにscience total capの違反とは扱わない。

### 8.3 generic phase、T coverage、full CD native

generic Aのphaseが現在のW表現に収まるか、四つの過去fingerprintが再現するか、各認証が予算内に収まるかは未実行である。T coverage不足も起こり得る。失敗時は技術的に停止し、違うangle/seed/precisionで再実行しない。

CDのglobal native・prep/readout誤差は未取得なので、sourceが返す条件付き精度のまま扱う。今回の修正でこれらを実測済みにする必要はない。

## 9. 62件PASSの評価

保存receiptには62 passed／0 failure／0 error、登録CD/E・Tは0、off-domain compileは最終suite1回（準備全体5回）とある。[S16]

source testsを読むと、有限bit期待費用／moment、weightとerrorの相関、angle意味論、Clifford相対位相、局所IR、部分T multiplicity、STOP／parent／path／source保護、wall/CPU/RSS等を検査している。[S15]

一方、今回指摘した4,096係数resultのreview-size上限と、作成時inventory digestからchildまでの結合は、そのtestsでは閉じていない。したがって、62件PASSと今回の修正要求は矛盾しない。

今回62件をこの環境で再実行していない。補助metadata checksを62へ足して「総合認証件数」を増やすこともしない。

## 10. Codexへまとめて渡す修正範囲

> 固定source Sは履歴として保持し、別の修正commitで以下を実装する。比較条件・研究方式は変更しない。
>
> 1. 大きなraw結果／prefixに対し、512KiB以下のremote-review用chunk/indexと完全再構成SHA256確認を追加する。内部raw JSONを保持してよい。reader側への影響をテストする。
> 2. 合成前freeze inventoryのSHA256をjob・worker・結果・最終集計に結合し、実行前／集計前に照合する。
> 3. 元のfocused testsと、この二項目に直接対応するmetadata/mock testsを実施する。coordinator起動runtimeの自己照合も小さい防御的追加として検討してよい。
> 4. 登録CD/E、compile本番、T合成、分子、最適化、旧runner再開は行わない。新しい式、fixture、格子、budget、synthesizerを導入しない。
> 5. source/contract/test/hash/差分を保存して新source S'を固定し、commit・push・remote再取得後STOP。旧Sや旧結果をamend・上書きしない。
>
> 報告では二つの指摘と修正file/testを対応付ける。sourceと公開receiptのcommitを区別し、source review後の明示承認まではauthorizationをfalseのまま保持する。

実装の分割方法やfile名はCodexが決めてよい。大規模な新serialization frameworkや新compilerを開発する必要はない。

## 11. 再開条件と、その後の研究判断

次のsource確認はS→S'の差分、元契約の不変性、上記受入testsだけに限定する。数理構成や既存研究の比較をもう一度最初から評価することは求めない。

この差分が閉じた後、利用者が実行を明示承認すれば、S'の直接子authorization-only Aを作って既定のone-shotを実行する。公開receiptのみのHEADをscience sourceと取り違えない。

結果が出たら、CDの条件付きbias／shot envelope、Eの局所価格、Tのcoverage／固定plan分離を区別して報告する。本番失敗や取得不足を新手法の科学的negativeに置き換えない。同時に、今回のsource修正だけを理由に研究候補へscientific GOを出さない。

## 12. 今回実施しなかったこと

- 登録CD/Eのframe生成・開封、mean・signal・shot-budgetの新規評価。
- 22 native compile、64 T key合成、分子・DF production、量子shot。
- 62-test suiteの同runtime再実行、全保護fileの独立照合、GitHubへのwrite。
- ライブラリのinstall/update、旧G10/RA-D0再実行。
- 新規性や論文化可能性の再レビュー、A/Bの資源勝者認定。

今回の結果は、source読解と、保存schema／最小mock／単純な有理数算術の補助確認に基づく。登録実験のpreviewではない。

## 13. 参考資料

以下のrepository資料はSに固定する。公開receiptとbranch endpointだけはH／live参照である。

- [S01] [`docs/tracks/algorithm_codesign/df_correction_comparison_preregistration_v2_20261011.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/docs/tracks/algorithm_codesign/df_correction_comparison_preregistration_v2_20261011.md)
- [S02] [`docs/tracks/algorithm_codesign/df_correction_comparison_execution_plan_v2_20261011.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/docs/tracks/algorithm_codesign/df_correction_comparison_execution_plan_v2_20261011.md)
- [S03] [`artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/contract_v2.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/contract_v2.json)
- [S04] [`src/trottertracks/algorithm_codesign/df_correction_comparison/comparison.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/src/trottertracks/algorithm_codesign/df_correction_comparison/comparison.py)
- [S05] [`src/trottertracks/algorithm_codesign/df_correction_comparison/certification.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/src/trottertracks/algorithm_codesign/df_correction_comparison/certification.py)
- [S06] [`src/trottertracks/algorithm_codesign/df_correction_comparison/digital.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/src/trottertracks/algorithm_codesign/df_correction_comparison/digital.py)
- [S07] [`src/trottertracks/algorithm_codesign/df_correction_comparison/intervals.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/src/trottertracks/algorithm_codesign/df_correction_comparison/intervals.py)
- [S08] [`src/trottertracks/algorithm_codesign/df_correction_comparison/native_evaluation.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/src/trottertracks/algorithm_codesign/df_correction_comparison/native_evaluation.py)
- [S09] [`src/trottertracks/algorithm_codesign/df_correction_comparison/angles.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/src/trottertracks/algorithm_codesign/df_correction_comparison/angles.py)
- [S10] [`src/trottertracks/algorithm_codesign/df_correction_comparison/native.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/src/trottertracks/algorithm_codesign/df_correction_comparison/native.py)
- [S11] [`src/trottertracks/algorithm_codesign/df_correction_comparison/guard.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/src/trottertracks/algorithm_codesign/df_correction_comparison/guard.py)
- [S12] [`scripts/tracks/algorithm_codesign/run_df_correction_comparison_v2.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/scripts/tracks/algorithm_codesign/run_df_correction_comparison_v2.py)
- [S13] [`src/trottertracks/algorithm_codesign/df_correction_comparison/worker.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/src/trottertracks/algorithm_codesign/df_correction_comparison/worker.py)
- [S14] [`src/trottertracks/algorithm_codesign/df_correction_comparison/launch.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/src/trottertracks/algorithm_codesign/df_correction_comparison/launch.py)
- [S15] [`tests/tracks/algorithm_codesign/test_df_correction_comparison_v2.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/tests/tracks/algorithm_codesign/test_df_correction_comparison_v2.py)
- [S16] [`artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/focused_tests_final_v2.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/focused_tests_final_v2.json)
- [S17] [`src/trottertracks/algorithm_codesign/numerics.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4aba2b698cdc81710dbdbb5769a1686f716ffe2/src/trottertracks/algorithm_codesign/numerics.py)
- [S18] [`artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/publication_github_verification_v2.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f41cb7c33ee1cded5cef4e1942491e0447b6be69/artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/publication_github_verification_v2.json)
- [S19] GitHub branch endpoint：`track-b-df-correction-comparison-v2-source-preparation-20261011`。レビュー時HEAD H、親S。
- [P01] [pygridsynth 2.0.0の公開仕様](https://pypi.org/project/pygridsynth/2.0.0/)。radians、dps、seed、phase、wheel hashの限定照合。
- [P02] [mpmath 1.3.0 contexts](https://mpmath.org/doc/current/contexts.html)。interval arithmeticの包含と実験的機能の注意。
- [P03] [Python 3.11 resource](https://docs.python.org/3.11/library/resource.html)。RUSAGE_CHILDREN／RLIMIT_CPUの意味。

外部仕様はAPI／数値モデル確認に限って参照した。新しいアルゴリズムの先行研究調査は行っていない。

## 14. 同時提供する補助記録

`df_correction_source_v2_review_20261011/metadata_checks.py`と`metadata_checks.json`を提供する。登録データを使わない保存schema／mockの検査であり、science resultではない。
