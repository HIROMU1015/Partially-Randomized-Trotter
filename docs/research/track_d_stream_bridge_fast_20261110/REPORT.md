# 長q資源評価器：同一ruleの高速化・省メモリ化結果

2026-11-10 JST。**保存q4/q8のStrong／Reference X/Y、計8 wrapperで、旧新の最終命令列・位相・各反復・全7資源が一致した。** 同条件のnative列以降の比較では、q8は約2.09〜2.16倍高速、peak RSSは約53.2〜54.2%減。q4は約1.82〜1.87倍高速、約35.1〜37.5%減だった。最適化ruleは追加していない。一方、反復数10／20と全列走査は残り、長qの実用性は未認定である。

[実装・対応条件](IMPLEMENTATION.md)／[正確性と56資源](CORRECTNESS.md)／[性能](PERFORMANCE.md)／[次段階案](NEXT_EXECUTION_PROPOSAL.md)／[固定PLAN](PLAN.md)／[STOP](STOP.md)。

## 固定科学条件・入力

Linear H4／隣接1.00 Å／STO-3G／Nα=Nβ=2、元H／number-CDF8／元frame順LD2／ordered partial S2/s1/r32/K0／h=δ=.1。元Hの厳密基底状態仮定・準備費除外。対象は保存q4_sample00/cut2とq8_sample00/cut4の各Strong symmetric-directional／whole-path一gadget Reference X/Y、計8 wrapperのみ。C1 head-tail追加融合OFF、既存Gaussian共有・basis-run共有・step境界融合・scalar/identity・actual adjoint・χ・CSWAP／readoutを保持した。

保存P0はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approx1／qubits initially zero=False／topology-free／既存宣言global phase規約。新P0は人工分も0。旧q4/q8のraw hash、native列／phase、保存価格と生成SOURCEの対応を再利用し、別path／別cutや生成し直したwhole-pathを使わない。

基点は817a1cd54da11ac20dc5a84477bf0f742a567214、既存リソース評価branchを継続した。最適化器／入力準備SOURCEは **53f5d69e5c52159b55e6785b7e26256354fac874**、native入力を **b5b1ad594c9261ebf00e0fcb5700d885a98d11ae** で固定した。準備したq4 native列は旧保存列と完全hash一致し、q8も既存lowererによる同じraw／phaseから取得した。canonical入力を固定してから旧新を測り、出力を既存保存結果と比較した。

## 改善内容と実測

新[stream bridge](../../../src/trottertracks/track_d_stream_bridge_fast.py)は、旧oneq_runsでcache missを解き、旧CX feedで相殺する。wire／opcode／float64全bitで一致するrunだけを再利用し、1Q flush順とphase加算順を保つ。1Q出力をCXへ直接streamし、元input／folded全wordのlist化を省く。初回T0 validationを維持し、使わない中間Counterだけを省いた。固定幅frontierへの切捨てやchunk価格の独立加算はない。

主比較は **run02_paired**、実行SOURCE **97bf92583d6b03ace5c9173e8e2044a2469aa381**。同じvenv・入力・phase・観測器で旧新16 fresh processを同じpoolへ投入した。全体wallは40.823秒。q8各workerは旧約38.5〜39.3秒→新約18.2〜18.4秒、旧RSS約228.8〜230.0 MiB→新約105.3〜107.4 MiB。数値は各一回の記述的測定であり、全環境の速度保証ではない。

**この倍率はnative入力decodeからbridge／計数までで、raw読込・loweringやfull P0とのend-to-end倍率ではない。** 初期化／入力検査・出力検査のtimer範囲、RSSの定義、raw準備の別性能を[PERFORMANCE](PERFORMANCE.md)に記録した。native以降でのメモリ低下をraw pipeline全体の同じ低下率として使わない。

q4／q8の各round数は旧新で同じ10／20、1Q／CXの走査命令数も同じ。再合成の繰返しと不要な中間処理は大幅に減ったが、計算量はO(RL)時間・O(L)記憶、最悪O(L²)のままである。

## 正確性・位相・接続

旧新16実行すべてで、最終native wordのcanonical hash、binary64 phaseのbit、MPFR160 phase ledger全体、各round、全7資源を固定保存結果とexact比較してPASSした。8 wrapper×7資源＝56値は旧＝新＝保存P0で、残差は全て0。保存P0価格の再利用であり、新compile／全入力simulationはしていない。

最終の人工・技術回帰は新fast7＋旧53＋計測adapter1＋stream接続2＝63件PASS。固定54人工raw QPYでも旧新のword／phase／round／保存P0の全7資源が一致した。数学的に同値な別wordで価格だけを合わせたのではない。相対χ・actual adjointは固定native列の値と順をそのまま保ち、global phaseも既存MPFR160の項を同じ順に集約した。

[phase-producing stream接続adapter](../../../src/trottertracks/track_d_stream_resources.py)は、producer終了後に確定する入力phaseを扱い、全logical roundのphase_additionsから位相を再構成する。cache missだけのphaseを集めてはならない。この接続は小人工2件で確認し、H4 raw→最終価格の新end-to-end測定とは区別した。

## 失敗・修正・追加計測の保全

初回benchmarkは旧関数のphase引数へglobal_phase keywordを渡す計測呼出し不備により、旧optimizer開始前に停止した。改善側8件は正常PASSして保存し、旧runnerを変更せず、位置引数へ渡す互換adapterだけを **4c78a6c0ae16dbc2679859bff296f050d690f47d** に追加して未実行の旧8件を完了した。

この別wave比較をrun01に残し、同時条件の比較を確保するため、同じ8 wrapperの旧新をrun02_pairedへ一度だけ追加測定した。初回失敗・全結果・SOURCEは上書きせず、主表はrun02のみを使用する。新fixture／新科学条件／価格合わせのrule変更は0。小人工integration testにも入力phase項数と最終ledger項数を混同した期待値の失敗があり、旧phase関数をoracleにした期待値へ修正した。adapter本体の変更や閾値緩和は不要で、失敗logも保全した。

## 研究目的への寄与と限界

full-wrapper再構築／P0／CAR target／全入力simulationを省く経路に近づいた。有限価格検査や将来の校正点を安く取得する技術資産として有用である。一方、q8でも約216万の1Q走査と約183万のCX走査が残る（Reference X）。raw QCとgate.definition等のcacheも保持し、native入力準備のpeakはq8約369〜372 MiBだった。一般P0／全law期待価格／q1024〜4096の速度・RAM・精度・全RPE総費用は未確認である。

推奨は、改善版を軽量な有限価格検査器として保全し、追加開発は再走査削減の小さな技術案へ限定すること。全P0再実装を研究目的にしない。既存A/F/Sを条件付きproxyとして維持し、PR＋RPE全体の共同最適化へ重点を移す案も合理的である。正式採択・科学的GO／研究終了はユーザー／GPTへ戻す。

新sampling／whole-path生成／P0／H・law変更／全入力simulation／q16以上／fit／N／全13-round更新／詳細科学レビューは0。旧SOURCE／input／raw／native／価格／grant／ledger／marker／STOPを保持した。公開phase_additionsはpassごとの集約scalarであり、個々のgate/run phase原本・native角列はprivate localに保全する。内部保存先／原本QPY/NPZは公開しない。公開ref／tree／bytes照合後mandatory STOP。
