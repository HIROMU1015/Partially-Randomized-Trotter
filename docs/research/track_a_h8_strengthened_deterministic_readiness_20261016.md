# Track A H8：prefix細格子・打切りS4の準備と次の実行計画

**H8の旧粗格子とB0+I取得は完了している。同種の細格子・打切りS4の実施済み結果は、確認した固定branch・成果物にはない。今回、新規152候補の台帳、既存証拠の再利用関係、Phase別の資源上限案とSTOP条件を準備した。新signal・数値再解析・compileは0。実行用adapterの接続と人工検証は残っており、現時点では起動できない。**

branch：`track-a-h8-strengthened-deterministic-readiness-20261016`。
起点：`68a010b35d35a1500dcc901c480fcce433fac4a8`。
[機械可読の未承認計画・来歴・検査](../../artifacts/resource_applicability/track_a_h8_strengthened_readiness/2026-10-16/README.md)。
[V2現行fit設計の終了と代替案](track_a_v2_residual_design_closure_20261016.md)は低優先の別作業として整理した。

## 準備済みの範囲と重複しない範囲

|既存成果|固定commitと取得済み範囲|今回の扱い|
|---|---|---|
|H8 resource map|`681ec23c93fdd1090d325908f4b155a79115afdf`。B0 p7/14/21 S2、full p28 S2/S4、q=1/2/4/8、探索PR72候補/n8と確認union6/n32|signal・保存費用・確認結果を再利用。未測定をゼロにせず、旧候補を再実行しない|
|H8 D0|`3d910ee1de4597b8895e8506500553dbad5a5085`。保存5精度、factor1/10、6指標・P感度の会計|元D0を再実行しない。新格子のRZ/P=0会計は将来の別Phase|
|H8 B0+I全family|[結果commit 5f79257f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_h8_native_b0i_full_family_results_20261016.md)。p7/14/21 S2の12構成を評価し、適格8構成・16軸を実compile。p7の4構成・8軸は全条件不適格で未測定|保存scalar/signal/shot/軸費用/正しさreceiptを再利用。p7の欠測費用を埋めるcompileは登録しない|
|H6 V1|[結果commit 25bbdd33](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/25bbdd33fe758a0e730138cb91ddac8c41b0a298/docs/research/track_a_v1_phase_b_readiness_20261016.md)。追加104候補、固定model、S4最大4軸|候補分離、重複排除、per-axis STOPとPhase分離の実装を参考にする。H6の数値・費用係数・dense参照は移送しない|
|H6/H8理論費用|`7e41670565d9284da9aded94063be591fdcaa3dd`。832行と保存compiled接続、H8の7 rootのpreparation記述・private export|既存full p28の操作記述を新prefix viewの接続元候補として再利用。理論費用を新格子のcompiled費用の認証に使わない|

H8 B0+Iの実行SOURCEは`54ed98c5fa5b7190f77255271f4e3bec8db15630`、native providerは`c31ef424175c02379eafebe8fc6c7548c1d9aabe`。旧provider 265ファイルをGit原本と現在diskのbytes/SHAで照合した。8保存入力・exportのhash、25固定文書/code/成果物も照合した。NPZ配列を開かず、preparationを復元せず、832行の費用payloadと接続gzipを解析していない。これはprimary agent自身の読取専用検査であり、独立レビューや外部再現ではない。

確認対象は登録worktree、関連する固定成果物と既存の保存exportである。全storageに未発見の計算が存在しないことを証明したものではない。起動前にも候補IDと新しい成果物の有無を照合し、重複が見つかれば自動実行せずscopeを再固定する。

## 固定条件と追加候補案

linear H8、1.00 Å、STO-3G、16 spin modes、Nα=Nβ=4、sector dimension4900。保存tol-only DF rank28、tol=1e-8、cutoff=0、signed generation order、weighted Hermitian projection。T=.8、q=1/2/4/8、δ=.8/.4/.2/.1。complex signal ε=.05/.01/.005/.001/.0005、経験的allowance factor1/10。εはenergy精度ではない。

|種類|prefix|手法・q|新規候補数|
|---|---|---|---:|
|追加S2|8〜20、既存14を除外|B0・B0+I、各q=1/2/4/8|96|
|追加打切りS4|15〜21|B0・B0+I、各q=1/2/4/8|56|
|再利用する要求内S2|14|B0・B0+I、各q=1/2/4/8|新規0、再利用8|
|旧決定論集合|p7/14/21 S2の両手法、full p28 S2/S4|各q=1/2/4/8|新規0、再利用32|

合計は旧32＋新152＝184決定論候補。152は将来のsignal評価数であり、compile数ではない。B0の新しい独立signalは76で、B0+Iの76は対応するB0への解析的位相補正とする。新signalが必要なprefixは13種類。prefix21の保存tail scalarを再利用し、新規tail scalarは最大12種類とする。

この格子はH6 V1の「旧1/4〜3/4域の整数S2」と「旧1/2より上の打切りS4」をH8の旧p7/14/21へ対応させた**提案**である。新PR prefix、prefix1〜6・22〜27、q16、新T、新ε、全prefix最適化は含めない。旧full p28は保持する。登録内の対照を強くする検証であり、決定論の大域最良やPRとの全パラメータ最適化比較ではない。

比較規則は既存のsymmetric-directional control、original scalar＋tail scalarを一度だけ加えるpolicy、full-H reference、経験的uを維持する。B0+Iでは
`C_tail = Σ_tail signed_lambda/4 * ((Tr G)^2 + Tr(G^2))`
を用い、B0 signalだけを`exp(-i T C_tail)`で回転する。full-H referenceは回転しない。独立経路と各実数/虚数軸のsigned discard/PF/total分解を保存する。分解が閉じない場合や旧分解が欠ける場合は未解決とし、非相殺診断を厳密bias保証と呼ばない。

数値policyは既存H8 contractから、binary64、safety factor10、floor1e-12、roundoff multiplier64、最大u/headroom=.01、感度factor1/10を保持する。`empirical_allowance`と`precision_rows`を同じ固定SOURCEから使い、各軸alpha=.025・epsilon/sqrt(2)のHoeffding割当を変更しない。uの丸めproxy・reference/path差・closureを省略しない。新候補のuやshotは今回評価していない。compilerはQiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling mapなし。native control probe toleranceは旧H8 B0I契約の1e-8を保持する。

## 次の技術作業とPhase A

H6のsignal adapterはsector400のdense行列・expm/eigh対照を使う。H8へはそのまま移さず、既存`H8SignalPort`、`OccupationCSR`、`SpinPrimitive`、`exponential_action`と独立forward recurrenceを使う。16 modes全Fock行列や4900×4900のdense参照を新設しない。full-H referenceは保存state/T/DFと同一であることをhashと誤差receiptで結合して再利用する。結合できなければSTOPし、新referenceを自動生成しない。

起動前に必要な実装・人工検証は以下である。

1. 既存H8 exportのfull p28から、保存操作列・qubit順序・scalar・basis identityを保つprefix viewを作るadapter。新規diagonalizationやGaussian再生成を行わず、旧discard:p21との対応を確認する。復元後の実分子数値確認は将来の承認scopeに入れる。
2. 旧p7/14/21・S2専用whitelistを変更せず、新SOURCEに152候補だけを登録するH8 signal adapter。既存full-wrapper bridgeはS4・新prefix対応済みではない。
3. S4の符号付きYoshida時間を含むprimitive検証のdedup台帳と、3 probeによるnative/spin/CSR経路の比較。新prefix exact-truncated参照は疎な指数作用で最大13 pairs、q・S2/S4間で共有する。既存証拠で時間・profile・basisまで同一と確認できる検証は再利用する。
4. H6のowned worker/監視/cleanupをH8へ接続し、primitive検証waveとprefix waveを分ける。snapshotと保存full-H referenceを一度だけ共有し、prefix task内でq・次数とB0I overlayを共有する。fork前single thread、各role thread1、nested parallelismなし。退出後native exit0とwait4回収が揃うまで結果を受理しない。
5. 独立経路差・norm・非finite、state/sector/DF/basis/scalar結合、実数/虚数の誤差成分、旧shot規則・allowance・適格性を人工fixtureで検査し、新SOURCE/runtime/input closureをsealする。

今回追加したcheckerは**提案台帳の検査専用**である。上記science adapterの代替ではなく、科学起動・grant発行の入口を持たない。誤って旧候補を再登録する、B0I-S4を落とす、compileをPhase Aへ混ぜる、Phaseを自動継続する等を拒否する10人工testsがPASSした。local技術検査でありCI・独立レビューではない。

次の科学実行承認の対象は、上記実装・人工検証・最終bindingが完了した後の**Phase Aだけ**となる。Phase Aは追加152候補のsignal・bias・経験的適格性まで。new SCF/DF/state、PR sampling/law/normalization、native full-wrapper build、compile、費用fit/再会計は0。Phase終了時に必ずSTOPする。

## Phase別の承認境界

|Phase|承認後の限定処理|終了時|
|---|---|---|
|A|新76 B0 signal＋76 B0I overlay、最大13 prefix参照、bias・適格性|STOP。費用modelへ自動移行しない|
|B1|保存H8 anchorだけからmethod・軸・compiler/profile/control/scalar/native構成ごとの費用modelを固定。係数と具体的検証軸を事前seal|STOP。数値fit自体も別承認、新compile0|
|B2|適格な打切りS4の最大2構成・4軸をnative full-wrapper compile|各軸判定後、全PASSでもSTOP|
|C|承認された固定model・実測費用と保存shotから5精度RZ/P=0を再会計|STOP。追加実測・PR探索へ進まない|

B1はS2の`a+b*q+d*prefix*q`を候補とし、H6係数を使わない。各stratumに少なくとも3つの非共線anchorを必要とする。B0の旧p14/21とB0Iの新p14/21は費用の所在があるが、旧B0のbinary/profile同一性は未認定である。current profileを遡及補完せず、無条件に混ぜない。整合するanchorがなければSTOPする。H6で使ったような限定transfer仮説が必要なら、未知を明示した独立receiptと利用者による条件の受容が先に必要であり、自動採択しない。今回係数fit、rank計算、anchor費用の再抽出は行っていない。

S4は非結合の符号付き3段schedule・wrapper/scalar一回という構造が確認できる場合だけ、`a+3*b*q+3*d*prefix*q`を検証仮説とする。単にS2費用の3倍とはしない。S4のalias/負時間/隣接境界mergeが異なればmodelを適用せずSTOPする。

B2の事前選択規則は、Phase Aで少なくとも1つのε/factor条件に適格な新S4集合から、B0は(prefix,q)辞書順最小、B0Iは最大を選び、各cosine/sineを検証する。選択には新compiled費用を使わない。familyに適格S4がなければそのfamilyの検証は0で、そのS4 modelを利用しない。代替候補の追加はしない。実際の候補ID・軸・予測値・model SHAをB1後に固定してからcompile承認を受ける。

軸順はB0 cosine→sine→B0I cosine→sine、**compile並列数1**。`abs(predicted−measured_RZ)/measured_RZ >= 1/50`ならその軸で即STOPし、次軸を起動しない。測定分母・2%境界を使い、両軸平均での救済はしない。最大2 native evolution、4 wrapper、8 control probes。left/right branch・X/Y・保存独立signalの一致、scalar once、compiler hash、前後fingerprintを確認する。自動再fit・追加compile・retry/resumeは禁止。

4軸PASSでも未測定全格子の2%保証やS2外挿の保証ではない。将来選ばれた最良がモデル推定で、強い優劣主張に実測が必要なら、その候補だけを別scope・別承認で検討する。今回そのcompileは登録しない。

Cの主比較は保存確認PR union6/n32と登録内決定論最良、P=0、RZ、経験的allowance factor1/10。PR探索72/n8は分離する。実測、固定model推定、保存PR標本平均を区別し、未測定を0としない。旧6指標・P感度の結論を新格子へ移送しない。母平均優位、厳密bias、未観測希少event、state preparation/energy/QPE/RPEへの一般化は未確立のまま保持する。

## 資源上限案と起動gate

すべて未承認のSTOP上限であり、予測時間・実測speedupではない。H8旧B0Iの16軸はwall約2時間20分、観測driver/worker peak RSSは約10.95/11.13 GiBだったが、今回のsignalやS4の所要時間には換算しない。H6用RSS9 GiBをH8へコピーしない。

|項目|A：並列signal案|B1：保存model固定案|B2：S4 compile案|C：再会計案|
|---|---|---|---|---|
|role・thread|driver1＋worker最大16＋observer1、各thread1|1 process/thread1|driver1＋observer1、compile1/thread1|1 process/thread1|
|AS/RSS|driver64/32 GiB、各worker64/16 GiB、observer512/512 MiB|512/256 MiB|driver64/32 GiB、observer512/512 MiB|2/1 GiB|
|累積CPU / wall|192 CPU h / 48 h、worker CPU48 h|240 s / 600 s|48 CPU h / 48 h|600 s / 900 s|
|output / stage storage|2 / 8 GiB|64 MiB / 同じ新作業領域|2 / 8 GiB|512 MiB / 同じ新作業領域|

Aのaggregate RSS上限は288.5 GiB、admission reserve32 GiB。最大worker数を減らす場合も、実行前にCPU割当と並列数を新契約へ固定し、実行中に自動拡大しない。仮想ASの合計を実RAM消費と呼ばない。ホスト/cgroup実効空き容量と全roleのRSS上限＋reserveが合わなければ起動しない。空きCPUが足りない場合は他jobの停止・affinity変更をせずgateで拒否する。

各Phaseのfresh gate（30秒以内）でSOURCE/provider/input/runtime/profile、専用physical core、cgroup/quota、空きRAM・disk・inode、host/cgroup PSI、OOM counter、旧grant/ledgerと保護job、exclusive output・one-shot lockを照合する。A/B2のstageは空きdisk16 GiB・inode40,000以上、output最大20,000 files。native instruction最大1,000,000、transpiled最大5,000,000。既存H5/H7/V1/V2実行域・cache・venvを共有書込先にしない。

observerは1秒間隔、観測gap5秒超でSTOP。host full PSI≥1%、cgroup full PSI正、OOM増加、role/aggregate資源超過、thread/affinity逸脱、未知のquota・profile、receipt改変、科学予算超過、native exit不明/非0ならSTOP。停止時はPID/start/UIDが一致するown processだけをterminate/reapし、部分結果・予約・unknownを保持する。失敗をrefund/resetして再試行しない。

## 今回の成果とSTOP

公開物は提案候補台帳、無効なgrant template、既存SOURCE/inputのhash receipt、metadata checkerと人工tests、本報告・V2終了記録・索引である。private絶対path、分子配列、runtime、credential、有効grant・ledgerを含めない。既存111 worktreeのHEAD/statusと旧667保護ファイル、旧V2未追跡scriptを前後照合する。

**科学runnerは未seal、CPU[]・SOURCE=null・command=null・approved=false。新科学計算・数値再解析・compile・fit・新有効grant0。公開後remote SHA/treeを確認してmandatory STOPし、次の技術作業・Phase Aへ自動移行しない。**
