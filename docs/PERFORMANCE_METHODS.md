# 並列化・高速化の再利用

PR-DF信号入口の保存index：[REPORT](research/d_pr_df_signal_entry_20261116/REPORT.md)、SOURCE2a5e085e5。既存GitReaderのcommit/path cacheとentry単位の物理key indexを再利用し、同じJSON・保存52roundを繰り返し読まない。keyにhead/tail・λ/s_R/c SOURCE・law・実step/piece配分を含め、κ/IDだけでcacheを分けない。新inventoryではfresh entryを作る。固定providerのmacro cacheは変更なし。確認は保存再現・人工/metadata検査のみで、新しい実信号の速度benchmarkは0。

LD9 m6への既存prefix context/fast/path worker適用：[REPORT](research/d_pr_df_ld9_m6_price_20261116/REPORT.md)、SOURCE8f0a82842。親でprefix9 contextを一回prepare、独立4 pathをforkしX/Y個別compile。実測wall248.674s/8compile、workerCPU合計968.346s、単一worker最大RSS1,082,800KiB。倍率benchmarkなし。seed/order/profile/cache条件を維持、協調stage/exit保存。

LD9への同じsignal cache/元Q3 acceleratorの適用：[REPORT](research/d_pr_df_ld9_v0_results_20261116/REPORT.md)。SOURCE303ed2ef7のcallerはhelper1cbc24cfc/fefd providerを再利用、13roundのmacro初回計算後12cache hit、signal wall5.066s、Q3 worker232.882s。新性能アルゴリズム/倍率benchmarkは0。3.11元AST guard・協調進捗/exit保存を維持し、b>1 gateを緩めない。

## PR-DF prefix7：既存context/fast/path workerの再利用

[LD7 caller](../scripts/run_pr_df_ld7_prices.py)はe827のfrom_b/NativeRuntime/Adapterとmatching fastを使い、既存ea9e6609 path workerの出力とstateを専用scopeへ渡す。SOURCE8d4503e00。prefix1/2用request生成を流用せず、7本head・3本tail・corrected total scalarを人工fixtureと実preparationで確認。較正8独立pathをfork並列（X/Y個別compile）、holdout2 pathはモデル公開後。実測較正202.816s/16compile、holdout29.992s/4compile、倍率benchmarkなし。[REPORT](research/d_pr_df_ld7_price_results_20261116/REPORT.md)。Q3は既存3.11.0rc1 supervisor/accelerator、協調stage/round logを再利用し2 callとも完了。共有環境・sampler/seed/order・cache identity・compiler条件を変更しない。

## PR-DF P2/m3/m4：context共有、path並列、同一eventからS/C

保存H4 DF10・PR-DFの新ZZ経路では、[compiled_price_fast](../src/trottertracks/compiled_price_fast.py) の構造数え上げ、composeの回転再利用、fingerprintを使う。元実装SOURCEは `39ddd59edf54a3a901e5302207b2765ddbd03fbb`。contextは親で一回準備し、[m3 worker](../scripts/run_pr_df_ld1_m3.py) を4 pathにforkする。独立した4 pathが今回の並列単位であり、各compiler内部は元P2同様1 processとして重複poolを避ける。X/Yはそれぞれのwrapperをcompileする。

m3で実対象確認済み：[REPORT](research/d_pr_df_ld1_m3_results_20261115/REPORT.md)。m4は[専用caller](../scripts/run_pr_df_ld1_m4.py) が同じworkerを使い、保存seed表を元samplerの実行globalへ注入する。sampler本体・PCG64/CDF・event順序を変更しない。人工試験で実sampler bodyからのseed参照、fake compilerの同一event S/Cを確認した。m4の実測範囲は[専用結果](research/d_pr_df_ld1_m4_holdout_results_20261115/REPORT.md) に追記する。正常pathごとのstage進捗・所要時間・RSS、60秒周期stackを保存する。SOURCE固定後の書換えやscience retryはしない。

調査時には共通catalog `0dc1eeaee3d9d6465f18edc33d4e17ed312bd6f4:docs/PERFORMANCE_METHODS.md`（`codex/d-full-h-p2-caller-prep-20261114`）を参照した。Full-H固有のaxis workerやresume方式をPR-DFへ持ち込まず、既存P2の適合する経路を再利用した。未使用helperの存在を高速化の実行証拠としない。元SOURCE、cache identity、seed/order、lowering、wrapper/compilerを確認し、cache出所不明時のfallback compileは行わない。

2026-11-15 m4実対象確認：SOURCE `7bdfc6f629aa74a5c1347abf84ad6486c669c9d8`、固定4 path/8 compile、GNU wall61.27秒・CPU397%・max RSS593212 KiB、内部wall60.382917秒。全stage完了・receipt8/8一致。単回の資源計測であり速度倍率の比較試験ではない。科学再実行なし。詳細とpath別時間は上記m4 REPORT。

## PR-DF F0：保存support表でlaw診断

[F0 reader](../scripts/run_pr_df_ld1_f0.py) は固定 `d72dc4e6228bb380f4d1d000e5e030794fce7164:src/trottertracks/finite_k_sampler_law.py` と元CDF helperを再利用する。324-component表1個と2-order表13個だけを評価し、path位置数は条件付き上界の整数入力に使う。巨大なevent列生成、preparation再構築、sampling、compileは不要。Git blobはcommit/pathごとにcacheし、同じ保存JSONの再読を避ける。表ごとの処理が小さいためprocess poolを作らず、既存のsupport長に依存する計算を順次使う。人工reader試験14件・0.30秒で実経路、partial STOP、進捗logを確認した。[実行結果](research/d_pr_df_ld1_f0_law_diagnostic_20261115/REPORT.md)に実対象の時間・範囲を記録する。NumPy1.26.4のCDF、Fraction、元正規化・順序・hexを維持し、任意runtimeへのlaw移送はしない。Python3.11.0rc1の定期faulthandler dumpは使用せず、協調的START/END/所要時間と終了codeを保存する。

F0実対象確認：SOURCE `fad0c1e4fe41c0e4c6d0690158056572295ed15a`、14表・13条件付きevent/path bound、内部0.075383830秒、GNU wall0.50秒・max RSS41552 KiB、全表完了・異常0。追加benchmarkなし。起動を含むGNU timeのCPU1407%はruntime既定動作を含む計測で、14個のscientific workerを作ったものではない。再利用元のhelperを修正せず、巨大event列なしで固定scopeを完了した。


## PR-DF prefix2：matching adapter/fast、path並列、固定Q3のruntime検査

[caller](../scripts/run_pr_df_ld_split.py)はe8273676のNativeRuntime/Adapterと同commitのaxes/observer対応fastを専用moduleで読み、prefix2 contextを親で準備して12独立pathをforkする。元normal sampler/seed/event順、new-ZZ、scalar once、X/Y別wrapper/compilerを維持。現LD1 fastの旧signatureやprefix1固定C1 contextは代用しない。[実測](research/d_pr_df_ld_split_results_20261115/REPORT.md)：12 path/24 compile、wall262.288秒・path CPU合計1053.660秒・単一worker最大RSS1,223,216 KiB。倍率benchmarkではない。巨大event rawは原bytesをprivate保全＋可逆gzipで公開する。

有理数配分はe094b2d6 `generate`、solveはd83737b6 `_solve`を無変更再利用。saved Q3 rationalは16,000桁超の整数を含み得るため、読取時は既存B2同様integer digit limitに対応する。fit済み出力から配分だけ継続でき、fit/予測をやり直さない。

失敗例：動的worker moduleはsys.modules登録前にProcessPoolへ渡すとPicklingErrorになる。fake compiler/worker本体mockだけではactual accelerator constructorのAST/runtime適合は分からない。Python3.12はFunctionDefに空type_params fieldを追加し、旧AST hash gateがSOURCE bytes一致でも停止する。今回、実Q3は0 callのまま停止し、constructorをpreflightで確認する最小回帰を追加した。guardを緩めて通す修正は未採用。科学同等性を保つbootstrapとruntimeの照合を次回の実SOURCEで先に行う。旧STOPと未開始callを保全し、同じ科学評価を自動retryしない。


## 固定total-Q3 AST gate：元Python3.11環境で再利用（2026-11-16）

[再開REPORT](research/d_pr_df_ld_split_q3_resume_20261116/REPORT.md)／[caller](../scripts/run_pr_df_ld_split_q3_resume.py)。成功B2 package738df226のTESTSにPython3.11.0rc1と `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python` の参照がある。今回この既存runtimeで固定constructorを本番workerの実入口から人工検証し、元AST hash3d043e71…を変更せず4 Q3を実完了した。Python3.12のtype_paramsを除去する修正や期待hash置換は不要。元SOURCE/入力・decoder/target/loss/thresholdは維持する。

4独立case並列、各8 DP worker/batch8を既存benchmarkから再利用、全体wall232.920秒。価格/配分は保存参照のまま、性能倍率の新benchmarkではない。人工fixtureのmetadata intakeだけを置換し、実constructor/evaluator/child/persistenceを通す2件PASS。outer worker exitcodeも保存した。Python3.11.0rc1のperiodic faulthandlerは使わない。旧3.12 STOPは保存し、新規科学callの再開は今回の明示指示に限定した。
