# S4 counter/native互換性：Stage 2完了

COMPLETE_MANDATORY_STOP。SOURCE `f01fa5fbd9c508e468dac3cff8405f57370dbb51`。Stage 1のPASS/STOPと新grantへ結合し、固定H4 full p12/q1/S4・H6 full p19/q8/S4、T=.8、symmetric-directionalを2 workerで一回評価した。

H4はlinear1Å/STO-3G/legacy DF12/8 modes、H6はlinear1Å/STO-3G/tol-only DF19/12 modes。保存Hamiltonian/DF/state・scalar・full Hadamard wrapper・P=0・compiler設定は維持。

|系|軸|修正後RZ＝保存native|修正後CX＝保存native|
|---|---|---:|---:|
|H4|cosine|48,274|21,446|
|H4|sine|48,275|21,446|
|H6|cosine|1,521,173|667,248|
|H6|sine|1,521,174|667,248|

8整数行全て完全一致。A/B/mixed各両軸を取得した。Bのcosine RZ/CXはH4 45,218/20,454、H6 1,452,773/644,824。A/B双方に修正の影響があり、旧mixed8行だけの変更とは仮定しない。

修正後のcounter評価に必要な元producer値は正時間33・修正された負時間33の計66個を一度だけ生成し、全model・反復q・両軸でcacheした。旧counter負時間との二者比較66評価は再実行せず、今回の負時間33件のRZ全hex・RZZ support/hash・phase全hexを既存native側の保存証拠へ照合し全件一致。A＝PF mixedを再利用し、同じcircuitを重複構築しない。全word bit同一性は未認定。

[受入・全モデル値・33block照合・source/input・消費済みgrant projection・native退出](../../artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/stage2/)。元grant原本はprivate保持。

本体9.924秒、coordinatorと2 workerはnative exit0・wait/reap。coordinator RSSは45,613,056 bytesでworker peakではない。入力2codecのbytes/SHAとsource/数値環境を前後照合PASS。新native build/compile/sampling/signal/reference0、retry0、旧SOURCE/結果/STOP/ledgerは不変。

Stage 2 STOPを記録済み。利用者の3-stage事前承認に従い、Codexが別SOURCE/grantでStage 3の固定7候補・84/84/360行の保存再会計へ進む。S4の今回取得値は実評価SOURCEを保持して再利用する。
