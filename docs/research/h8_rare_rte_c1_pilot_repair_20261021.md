# H8希少RTE C-1 pilot：技術修正・固定S2_SINGLE取得結果

2026-10-21。基準commit `d31ac1080e139eaad717301ffaa8d80160bbd378`を引き継いだ。ユーザーはprocess内thread制御等の技術修正と、既存C-1固定予算での科学完了を明示承認した。**両条件43/43本・計86 trajectory/172 full-wrapper compileを完了、数値照合・終了監査を保存しmandatory STOP**。新科学SOURCE `383dd9068504b8f578944984fc45a60dd4ed18b2`、実行前契約封印commit `17fd9052`。追加層・全層期待費用・PR/PF再順位・productionは実行していない。

## 64-thread問題の原因と修正

無設定の新processを段階的に観測し、NumPy import直後に同梱`libopenblas64_p-r0-0cf96a72.3.23.dev.so`が64 threadsを生成することを確認した。さらに`scipy.linalg`は別の`libscipy_openblas-c128ec02.so`をloadし、計127 threadsとなる。前回の64 threadsを単なるidle poolだから安全と読み替えず、元のsingle-thread fork gateを維持した。

新しく起動する自分の子processだけに、NumPy/SciPyのimportより前から`OPENBLAS_NUM_THREADS=1`を設定した。BLASの直列化はfork親を単一threadに保ち、trajectory/両軸process並列の二重並列を避けるためである。共有環境・他process・既存venv/package・OS/cgroup/affinity設定は変更していない。worker総数やCPU/RAM/wallに固定の低い上限を設けていない。

同じ元runtime/providerの主要20 moduleをimportしても1 threadを維持し、fresh process間と親子でSessionEquivalenceLibraryのsignatureが一致した。人工回路では修正前後のcompiled命令列/global phase・gate counts・depthがexact一致し、人工PCG64 raw word100個もexact一致した。これは人工確認であり、全possible circuitのbit同一性の数学的保証ではない。実H8のpreparation/compiler hash、元law hash、control/oracleを別途実行前・代表受入gateとして維持した。

同時に、loaderのWriterが使う`restored` directoryの事前作成漏れを修正した。元Pでも使用した、同一canonical bytesを返す限定fingerprint再利用を使い、重複したgraph/hash処理を削減した。これはcompile結果cacheではなく、各axisの使い捨てprocess内でだけ有効、終了時にaliasを復帰する。compileは各条件86回のまま。元cost cacheは無効である。

[技術修正証拠](../../artifacts/resource_applicability/h8_rare_rte_c1_pilot_repair/2026-10-21/technical_repair_evidence.json)、[runtime結果](../../artifacts/resource_applicability/h8_rare_rte_c1_pilot_repair/2026-10-21/runtime_repair.json)。新12＋既存55＝67人工test PASS。q2/q16のsymbolic43本・代表先行barrier、予算/registry改変拒否、代表失敗時に残42本未投入、非返還・子/孫回収、thread/library拒否、fingerprint復帰、独立momentを確認した。別AIへの委任や独立科学reviewは行っていない。

## 固定科学条件・確率規約

Linear H8/1 Å/STO-3G/tol-only 1e-8 DF rank28、T=.8/P=0、二次partial PF、通常signed bias・allowance1。元Hamiltonian/state/snapshot・純粋prefix復元・native DF、analytic shots・scalar/control/full Hadamard-wrapperを維持。Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed_transpiler17。

|条件|候補|cosine/sine shots|取得層|固定本数|
|---|---|---|---|---|
|ε=.05|B2_p8_q2_2nd_r4_K4|7659 / 8832|S2_SINGLE|43|
|ε=.0005|B2_p7_q16_2nd_r4_K4|79863816 / 99933856|S2_SINGLE|43|

S2_SINGLEは全qr event中、N2=1/N4=0。他のeventは次数0で、次数2の位置は固定しない。C-0.5の`EXACT_INTEGER_CONDITIONING_OF_STORED_BINARY64_V1_PROPOSED`を、今回の整合gate付き承認に従い使用した。保存binary64の厳密整数比・全支持count・整数CDF・元のcomponent正規化・PCG64 raw-bit消費・元_make_eventを変更していない。旧NumPy choiceのbit列再現は主張しない。

S0は既存P n48を別anchorのまま保存し、再解析・cohort混合0。S2_MULTI/S4_PRESENTの平均/分散は未知null。pilotのn43は正規iidモデル下の分散CI設計（η=1/2・各条件計画90%）であって、実分布の正規性、母平均の厳密保証、2条件同時90%を意味しない。実pilotの最大値を全support上界にしない。

## 結果と数値照合

|条件|n|条件付きG平均|不偏sample variance|標本SE|
|---|---:|---:|---:|---:|
|H8_0.05|43|1.266122603791E+9|6.124605768256E+14|3.774025599874E+6|
|H8_0.0005|43|9.128324880501E+13|9.459815822091E+23|1.483225258660E+11|

paired費用は同じevent列の`G_i=Ncos*RZcos+Nsin*RZsin`。mean・不偏sample varianceは整数/Fraction、SEはDecimal100で保存した。表は保存値の表示丸めのみ。正確なFractionと100桁SE、全trajectoryの両軸RZ/CX/depth/size・event digest・次数配置は[成果物packet](../../artifacts/resource_applicability/h8_rare_rte_c1_pilot_repair/2026-10-21/README.md)を参照。全event/seed/native内部pathはprivateに保全し、SHA inventoryを保存した。

独立数値経路はcentered Fraction計算とは別にraw整数sum/sum-squareからmean/varianceを照合した。同じ著者による別算術であり、別reviewerの独立再現ではない。数値PASSを母平均certificateへ格上げしない。

|条件|τ（元binary64）|normalization B|canonical p差の1e-12 gate|代表control最大誤差|
|---|---|---|---|---|
|H8_0.05|0.042407605139478735|1.001798225656013|PASS|1.3322676295501878e-14|
|H8_0.0005|0.016628983439974276|1.0002765188435128|PASS|2.635753752209741e-14|

実load後、元preparation hash、元law dict/hash、q/r/K/T/τ・normalizationと正のK4/component支持を確認し、保存binary64の有理数正規化p対B canonical文字列の差を固定`1e-12`で照合した。これらは実装の整合gateで、未知費用を含む期待費用biasの上界・PRNG bit-law保証ではない。希少層の無条件確率質量を再計算していない。

## 科学call・資源・終了

|科学call|coarse 予定 / 予約 / 実使用|tight 予定 / 予約 / 実使用|
|---|---|---|
|保存input/preparation decode|1 / 1 / 1|1 / 1 / 1|
|純粋prefix復元|1 / 1 / 1|1 / 1 / 1|
|元law形成|1 / 1 / 1|1 / 1 / 1|
|数値law照合|1 / 1 / 1|1 / 1 / 1|
|count表|1 / 1 / 1|1 / 1 / 1|
|component表|1 / 1 / 1|1 / 1 / 1|
|trajectory|43 / 43 / 43|43 / 43 / 43|
|occurrence|86 / 86 / 86|688 / 688 / 688|
|RTE event|344 / 344 / 344|2752 / 2752 / 2752|
|native evolution|43 / 43 / 43|43 / 43 / 43|
|測定full wrapper build|86 / 86 / 86|86 / 86 / 86|
|full wrapper compile|86 / 86 / 86|86 / 86 / 86|
|control probe（simulation内包）|4 / 4 / 4|4 / 4 / 4|
|event oracle|1 / 1 / 1|1 / 1 / 1|
|非測定validation wrapper|2 / 2 / 2|2 / 2 / 2|
|paired pilot会計|1 / 1 / 1|1 / 1 / 1|
|別経路moment数値検査|1 / 1 / 1|1 / 1 / 1|

科学callは実行前にlock/O_EXCLで非返還予約。新SOURCEに結合した新receipt・各条件独立grant・ledger・one-shot markerを使用し、科学load前にmarkerを消費した。旧消費済みgrantや旧STOPは変更・再利用していない。粗精度の全43本・数値/終了検査成功後にだけ細精度の独立grantを発行した。retry/resume・追加科学予算0。

各条件replica000のcontrol probe4（statevector simulationを含む）、event oracle1（元のvector作用・block-action準備を含む）、非測定validation wrapper2だけを実施。その受入後に残42本をtrajectory並列・同eventの両軸compile並列で実行した。入力/preparation/count/component表は親で1回だけ準備してfork/COW共有し、evolutionは各trajectory1回、測定wrapperは各axis1回。

|条件|実時間秒|native CPU秒（子孫・監視を含む）|最大active trajectory|観測axis RSS最大bytes|native exit/reap|
|---|---:|---:|---:|---:|---|
|H8_0.05|815.629|14057.458|32|1242869760|0 / reaped|
|H8_0.0005|5835.538|87888.548|16|5409017856|0 / reaped|

memory admissionには、同じ固定候補の保存済みnative RSS実測最大値（q2:1,203,666,944 bytes、q16:4,996,349,952 bytes）を使用した。leader＋2axisへ各最大値を割り当て、未観測S2 payloadを見込んだ係数2の予測を置き、合計6倍をprivate memory需要として予約した。これは経験的なworkload予測であって、実使用量のhard capや理論上界ではない。実空きCPU/physical topology/quota/memory/IOをread-only観測し投入数を動的調整した。元のresource制御・共有環境を変更しない。

native wait4、axisとtrajectoryの全owned worker回収、post-loader親子library/fork、最終SOURCE/input/runtime SHAを確認した。資源観測はRSS/PSS/private/thread/並列度の標本で、未観測の瞬間peak保証ではない。新signal/SCF/DF/state solver/Gaussian生成は0。追加のcontrol/sampling/compile、全層期待費用、順位、production本数計算は0。

## 保全とmandatory STOP

[終了監査](../../artifacts/resource_applicability/h8_rare_rte_c1_pilot_repair/2026-10-21/completion_audit.json)、[最終STOP](../../artifacts/resource_applicability/h8_rare_rte_c1_pilot_repair/2026-10-21/STOP.json)。旧SOURCE・raw/results・grant/ledger/marker/STOP、旧contractとindex本文を保全した。技術修正は新branch/新SOURCEで封印し、実行中の封印SOURCEは変更していない。限定private-pattern/構造/リンク/SHA検査を行ったが、包括的security certificateとはしない。

専用branch `codex/h8-rare-rte-c1-pilot-repair-20261021`へ結果と監査をローカルcommitしてmandatory STOP。GitHub push、S2_MULTI/S4追加取得、全層期待費用/PR-PF再順位、production、詳細GPT/Claude科学review、final evidence freezeは開始していない。

**次担当ユーザー**：pilotを踏まえた次の科学範囲・必要保証と公開範囲を判断する。**Codex**：現時点で停止。次段階へ自動進行しない。
