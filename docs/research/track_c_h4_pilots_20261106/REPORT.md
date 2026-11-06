# Track C H4：3候補の並列pilot

基準SOURCE `7db4792d4cd0ed9a7abc255565f207dd31a437bd`、専用branch `codex/track-c-h4-parallel-pilots-20261106`。
ユーザーの実行許可により[実行範囲追補](../track_c_o3_pilot_plans_20261106/PILOT_EXECUTION_SCOPE.md)を実施。
SERVER 2の専用worktree・venvで取得したローカル数値pilotであり、CI・外部再現の証拠ではない。

## 固定条件と評価の意味

Linear H4、隣接1.00 Å、STO-3G、neutral singlet、8 spin orbitals、4電子number sector70。
legacy DF12の生成順prefix LD3とtail-free Full LD12。元Hの厳密groundを入力とする仮定で、状態準備費を除く。
H4入力照合PASSを再利用し、SCF・DF fit・元H ground solve・入力照合は再実行していない。

保存meanを既存decoderで復元した20設定の最大エネルギー誤差は約 `5.35e-6 Ha`。これはSOURCE meanの診断でありnative/shotを含む性能証拠ではない。

全armで `epsilon_E=1.5936e-4 Ha`、失敗確率上限0.05、全位相予算0.4 rad、delta0=0.02、18 round、q=2^m。
PF、元RTE、残差RTE、合成、入力・数値、proposal丸めを差し引いた残りを統計へ配分する。
既存decoderにE0を渡さず、外部alias windowと端点margin、fresh IID有界shot、条件付き数値誤差モデルを仮定する。
平均信号の復元誤差と、十分shot数を実行した場合の成功条件を区別する。仮想量子shotの大量samplingはしていない。

Q profileはQiskit1.3.0、rz/sx/x/cx、opt2、seed17、topology指定なし。RZ・CX・depth・sizeを分記する。
C-Sは同じfull-wrapper Q最適化後にstrict Clifford+T/CXへlowerし、T/Tdagger・Clifford・CX・depthを分記する。
Tの主指標はTとTdaggerの合計。Q profileのRZをTへ換算しない。

## 結果とSTOP理由

**指定native compilerの未計上誤差を確認し、比較をSTOPした。全RPEのG・勝者・native成功確率は未資格。**
価格はQ 63/306セル、T 24/108セルを完了した時点で停止し、取得済みrawを保持した。未取得セルを0で補完していない。

| pilot | 取得した結果 | 判断 |
|---|---|---|
| C-N | 12構成×18 roundのideal/finite mean。強baselineのD/D融合補正と部分価格 | native lowering誤差を含むfairな全G比較はUNDETERMINED |
| C-S | 6構成×18 roundのfinite mean、range、残差rと合成精度、部分T価格・strict key | 登録した保守range/最小r policyには非常に大きいshot負担。native全G比較は未資格 |
| C-P | P0/P1×18 round、whole-kernel共役・両軸測定等価のSOURCE検査 | SOURCE信号・uniform shotの改善は分解できず。native費用便益は未資格 |

同じQiskit1.3.0/opt2/seed17の人工2-qubit `CRZ(1e-5)` が、`CommutativeCancellation` 後に空回路となる。
joint global phaseを合わせても作用素差は `4.999999999994792e-6`。これは合成後のT guardで検出・補正する対象より前の誤差である。
H4の負時間Full D halfでもSOURCEはHamiltonian参照と約4e-15で一致する一方、nativeのprojective状態差は `3.7748602395e-5`、complex信号差は `7.0225739790e-5` だった。
さらにFull Y4のq1 full wrapper（測定をcompiler入力に保持）でX期待値差 `1.4077178828e-5`、projective状態差 `2.1933549047e-4` を観測した。
これは当初の数値diskに含めていない。全18roundにこの誤差を0としてN/Gを確定することはできない。
Partial S2の同じq1診断はX差約3.7e-14であり、すべてのarmが同程度に誤っているとは結論しない。

測定をcompiler入力に保持した価格profileの短q検査と、その実native QASMも保存する。
価格workerの停止理由は科学的資格の不一致であり、CPU競合や他チャットの更新ではない。必要なfull-wrapper反例を回収して終了した。C-Pの追加native probeは上流のcompiler資格不一致により未完了で打ち切り、PASS扱いしていない。
compiler設定・位相予算・精度・alias・標本数は変更していない。新profileや追加batchによる救済実験は実行していない。


### C-N：PF native境界価格

S2/Yoshida4/公開Morales Y8m10b × h=.02/.01 × LD3/12の12構成を評価。
初期のideal/finite meanモデルでは全12構成が18 roundの位相・正半径条件を満たした。上記native誤差を含む適格判定とは区別する。
強加法baselineはR occurrence間のmaximal D runをまとめて最適化し、Fullはtail-free全体の実full価格を与える。
rawに保存したhalf-run加法値は弱い分解診断であり、最終baselineではD/D接続とwrapperの補正を加える。
full/強加法を同じshot配分でも比較し、配分まで最適化した比較と区別する。

### C-S：合成精度・残差配置

S0通常 / S1 D+R同位置残差 / S2 Dのみ同位置残差、合成cap .01/.02、残差cap .01。
登録した最小r_res規則は全roundで1を選んだ。D half当たり116 eligible回転すべてでcoarse indexは0、最大角は0.023138268390304983 rad。
同一generator位置の残差・coarseを厳密に融合する共通の機会を与えた。

終端log(range)はS0=0.5573908260、S1=386.8029033、S2=386.2455126。
有限rangeをfloat overflowで不適格とせず、log domainで会計した。初期float cutoffの出力はdebug記録として保持する。
全18 roundのliteral gate数とrangeに対する合成精度はS0=1e-13、S1/S2=1e-181。
2つのcapは同じ割当epsilonへ落ちるため価格keyを共有し、不使用の合成予算を統計へ戻す。
この大きなnormalization負担は、登録したpi/4 coarse・最小r_res・canonical lawに限る負材料であり、別のr/lawの最適性は主張しない。

元tailの条件付きstratum内で、実現xに依存する残差stratum確率を使う。周辺確率同士を掛けない。
正規化は実現ごとのWと解析的なglobal rangeに一度ずつ反映し、回路価格へabs(W)を再度掛けない。

### C-P：Processing

固定D-context raw YP8m8 kernel、h=.02、nD=1、P=Ppub adjoint。P1は最初の物理Pを実行・課金し、ancilla X/Y測定に不要な終端P adjointを省く。
P0/P1のuniform shot数は36軸すべて同じだった。
PとIの作用素距離は1.2697412914e-12、P0/P1のcorrected signal最大差は7.6635583279e-15。
固有値matching差はidealで1.8317967302e-13、finiteで2.4493935600e-13であり、数値丸めを越えるspectral改善は観測していない。

512次元full joint-spaceで測定等価loweringをordinary controlled参照と照合し、両軸のcomplex差は2.8575730769e-14。
control0の作用残差は2.9246549302e-14。中間gateを70次元へ射影する検査ではない。

## 価格・不確かさ・解釈の限界

各distinct r/precision recipeでq1/2の各非空元stratum32本を較正に使用し、独立q8の128本をholdoutとする。
Fullの同一決定論標本は1回のexact expectationとして計算する。
C-Sは元実現ごとに非空残差stratum各1本を生成し、その条件付き確率で戻してから元実現平均を取る。
C-N full/加法、C-P P0/P1は同じ元trajectoryをpairにする。

会計コードにはq1/2較正のaffine価格モデルを接続したが、今回のnative資格不一致により全18 roundのGは確定していない。
q8点誤差・MC半幅は局所診断であり、q131072までのモデル誤差上界は未取得である。
実装上は同じ較正を複数roundで使う共分散とpaired差を扱う。今回は未完了標本から最終MC区間や選択結果を作らない。
長qの未知幅が残るため、無条件の全RPE総コスト優越は確立しない。
exact E0/meanに依存する有限集合の選択はoracle diagnosticであり、実行可能selectorの利得ではない。

数値位相・半径はfloat64の条件付き診断であり、元H・全native回路の包含証明ではない。
b-bit整数CDF・正support保持・weight積の区間上界と丸めdiskは実装・人工検査を行ったが、native資格不一致により全Gの採用会計には到達していない。量子shotは実行していない。
価格samplingは記録したNumPy float64の条件付き実装であり、その系統誤差をMC標準誤差で覆ったとは主張しない。

量子shot数はSOURCE meanモデルに対するHoeffdingの十分数であり、最小必要shot数の証明ではない。
合成cap .01をそのまま留保した初期uniform診断の終端log10(N/軸)は、S0=2.8274、S1=338.3226、S2=337.8385。
これをnative回路が実際に目標精度・成功率を達成する証拠には使わない。

## 実装と再利用・技術修正

DF/RTE生成・有限law・native basis registry・既存RPE decoderを再利用し、新規Track C adapterとrunnerに接続した。
旧科学SOURCE・結果・grant・ledger・marker・STOPは変更しない。
公開Y8関数は指定blob、YP8m8は固定SOURCEの純粋関数を使用し、旧runnerは起動していない。

長い合成wordの素朴な逐次区間積では幅が過大になり、一部高精度keyのguardが不合格となった。
pygridsynthの厳密cyclotomic行列表現を区間評価するguardへ修正し、Wを保持したstrict Frobenius上界で検査した。
同じseed/epsilon/標本の未完了セルだけを再試行し、技術失敗の経緯と成功keyを区別して記す。
既定標本の角度keyを先に並列取得し、重複keyの待ちを減らした。科学条件や標本数の追加ではない。

SOURCEは実行中にdebugされたローカル版であり、公開commitの固定CI実行ではない。
保存INPUTSの初期adapter hashと、公開時点の最終SOURCEを区別する。
初回Full signalの共有r列ラベルはkernelのtail-free r=0 recipeには使われておらず、最終会計ではregistered_r_mと実r=0を分記する。

人工testは8件PASS。これはSOURCE/量子回路の研究仮説の性能PASSとは別。
strict key記録は3276件（epsilon -13:2087、-181:1188、人工debug -12:1）、完了Tセルで参照されたkeyは1927件。
記録済みguard上界・epsilon適合・sequence hashの検査はPASSであり、backend全再取得ではない。
Q価格driverの実時間は停止まで約6542秒、T初回は約5681秒。並列取得keyの延べwall時間とCPU総時間・全実行時間を混同せず、`ACQUISITION_SUMMARY.json`で範囲を示す。

## 成果物・停止・次の担当

[実行artifact](../../../artifacts/track_c_h4_pilots/2026-11-06/run01/)に、入力identity、18round信号、correctness、価格raw、合成key、条件付き会計、環境とbundle索引を保存する。
価格・keyの多数のrawは圧縮bundleへ収録し、元の個別ファイルはローカルにも保全する。runtime pickleとvenvは公開しない。

比較資格の不一致により未完了の価格呼出しを打ち切り、取得済みの科学結果・負材料・技術失敗を公開してmandatory STOPする。
予定された価格gridを完走したとは報告しない。旧実験のSTOPは不変更。新しいSTOPは本run01に限る。

**次の担当：ユーザー**。compilerの省略誤差を予算化するか、精度を保つcompiler profileへ変更するかを判断し、必要なら新しい有限実行範囲を明示許可する。
**Codex**はその判断・許可があるまで、新profileへの変更、残り価格の再実行、追加科学callを開始しない。
