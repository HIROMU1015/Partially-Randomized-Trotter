# Track D C7-2：実行前固定PLAN

2026-11-07 JST。今回の利用者指示に従う技術検証。実H4の新sampling、保存eventからの新回路、実分子compile、価格再fit・数値再解析、N算定、全RPE集計、詳細科学レビューは実施しない。旧SOURCE・結果・STOPを保全する。

## SOURCEと境界

基点はC7-1最終 `86c5c9d48dbddc085f388b6521be8cd92ee49d0d`。
元SOURCEは `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。指示中の38桁prefixはこのcommitへ一意に解決される。
専用branch `codex/track-d-c7-2-cx-structure-20261107`。本PLAN、CASES.json、人工adapter、runnerを実行前commitで固定し、そのSHAをSOURCE.jsonへ保存する。旧guardは変更しない。

P0は既存CompilerProfileそのまま：Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation1、initially_zero=false、topology-free。既存MPFR160宣言位相保持compilerを用い、phase fittingは行わない。
許容値は既存のcomponent max-entry 1e-10、observable Frobenius 2e-10、rtol0。失敗後に緩和しない。

## 固定case

数値・順序・cutの正本は[CASES.json](CASES.json)。全54 compileで、結果を見て追加・除外・変更しない。

|人工family|構成|compile数|期待する検査|
|---|---|---:|---|
|Diagonal|非零正RZ、明示零、signed RZ/RZZ＋宣言位相、それぞれuncontrolled/ordinary/directional|9|occupation対角から独立に作るU、diag(I,U)、diag(U†,U)と位相付き一致。control premiumの条件を調べる|
|Tail center|同Z正連続、同Z正負、同ZZ正連続、support変更、それぞれcontrolled/control-one|8|exp(-iθP)の積と一致。対消滅・非零・supportの境界とP0加法性|
|CX interface|隣接CX対、間にcontrol-wire RZ、間にtarget-wire RZ|3|I、RZ、RZZの位相付き一致。可換/非可換の書換えを区別|
|CSWAP|単体Fredkin|1|独立のbit置換と一致。単体P0価格だけを特定|
|whole-path|q2/q4×same/mixedの4 fixture。各fixtureにstrong X/Y、Reference X/Y、A†/B単体、妥当なscalar-only shell X/Y|32|AB=V、actual inverse、真空χ、strong/direct-V coherence、X/Y observable/complex coherence、raw/P0/reference leakage。shell＋max(lane)を診断|
|actual inverse|q2_sameの実AにA†を連結|1|Iに一致。単体価格加法性を仮定しない|

whole-pathは2 mode、人工三block、各step2 events、h=.1、cut=q/2。実H4のr32・law・cohortを名乗らない。scalar=.17、extracted identity=.04、τ=.2とframe角・signed atomを事前固定。正負はatom係数へ結び付ける。normalizationはunitaryへ掛けない。真空ゼロエネルギーの人工diagonalと、既存exterior-minor liftによる独立ordered referenceを用いる。

実際のStrongBuilder相対Gaussian transition、analytic vacuum phase、C3 prepare_steps、full-frame tail共有、directional/ordinary emitter、Reference gadget/actual inverseを再利用する。C3の定数Nだけを人工実行process内で2へ一時束縛し、返却されたH4専用descriptionは人工identityへ置換する。保存H4 request/price APIのguardを緩めない。既存C5簡易fixtureの置換emitterは構造検証には用いない。

## 判定と保存

全raw correctnessを先に確認し、PASS時のみ54本を各一回P0 compileする。全回路のraw/P0 unitaryをglobal phase込みで比較し、whole-pathのP0後observable・complex coherence・reference leakageも確認する。raw/P0の全資源、命令列、wire終端時刻、DAG critical pathを保存する。raw明示CXは未展開CRZ/RZZ/CSWAP等のnative価格ではない。

局所価格候補の反例は正常な技術結果として残す。零角をpruneする通常H4 domainと明示零interface、人工の正負相殺と合法H4 atom列の違いを明記する。人工PASSは実H4全path定理ではない。local sumとfull P0の間のbridgeを別の未解決条件にする。

既存のC5/C3人工回帰を再利用する。旧実分子runは呼び出さない。最大5qubitの小規模Operatorを使い、新simulatorを実装しない。CPU affinity/空きRAM/負荷を実際に観測し、独立54 compileをavailable CPU内でprocess並列化する。小行列BLAS/nested transpileの重複threadを防ぎ、共有設定を変えない。

回路QPY原本は新namespaceのignored localへ保存し、公開は人工SOURCE・fixture・数値検査・scalar価格・命令/依存列・hash索引に限る。過去の非公開rawを追加公開しない。内部絶対path・CPU/RAM観測を公開しない。

correctness/SOURCE/compiler不整合なら該当実行を停止してpartialとSTOP_FAILUREを保存し、無断retryしない。価格反例や全域bridge不足だけなら独立の構造調査とq32未採択案を完了する。最後に報告・q32案・STOPを公開し、remote SHA/tree/bytesを照合してmandatory STOP。次の担当はユーザー/GPTであり、q32以降の実行は別途指示を要する。
