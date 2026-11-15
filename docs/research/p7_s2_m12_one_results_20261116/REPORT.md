# p7/S2・16h/7：m12各軸1shotの固定Q3結果と4点比較

2026-11-16 JST。**1shotはQ3モデル不適合（有効MSE下界>τ）。新規Q3は承認された1件だけで終了し、確認済みの条件付き達成費用395,962,678 RZを維持する。** 数値gateは通過、UNKNOWN/INCONCLUSIVE/retryは0件。4/3/2shotは元APIを保存再利用し、再計算していない。

## m12が各軸4・3・2・1shotの比較

保存H4 linear / 1 Å / STO-3G / tol-only1e−8 DF10・sector(2,2)、B0I p7/S2・要求幅16h/7、保存H/state/snapshot・元13round schedule。全点でX/Y同数、m0〜m11は `[59,55,51,47,43,39,35,31,27,23,19,13]` に固定し、m12だけを変えた。

| m12のX/Y各軸N | 条件付き総compiled RZ数 | モデルMSE下界 Ha²（表示） | 上界 Ha²（表示） | Q3判定 |
|---:|---:|---:|---:|---|
| 4 | 435,481,090 | 1.506660616251764083585358E−6 | 1.506660616251764083585362E−6 | 適合・保存再利用 |
| 3 | 415,721,884 | 1.951734976728839865403189E−6 | 1.951734976728839865403194E−6 | 適合・保存再利用 |
| **2** | **395,962,678** | **2.415680996446553638790769E−6** | **2.415680996446553638790773E−6** | **適合・保存再利用** |
| 1 | 376,203,472 | **4.725530991854865818396863E−6** | **4.725530991854865818396867E−6** | **モデル不適合・新規1call** |

exact τ=(0.00159360010199040 Ha)²=**0.0000025395612850638132820416921600 Ha²**。判定は原APIのexact有理数で行い、表の端点は表示用の丸め値。1shotは下界>τで、単に保守的上界が大きいだけのINCONCLUSIVEではない。費用376,203,472 RZは当該配分の会計値として保存するが、適合達成費用へは採用しない。

**他round固定・X/Y同数のこの4配分に限り、確認された最小適合費用は2shot点の395,962,678 RZ。** 他round/軸別配分/他幅/他公式への単調性、必要Nの大域最適性、物理的に十分なNは認定しない。

正本は[EXACT_COMPARISON](EXACT_COMPARISON.json)、表示[SUMMARY](SUMMARY.json)、新[原API](run01/P7_S2_M12_ONE_FIXED/API_RETURN.json)。元4shotはb4296b110b129d226dbef2743c14f725a27c3bf6、3shotは0b33b4afdd285a99f610e59d8f5b77960a59a0d0、2shotは93ab9dbbe610d66eba39ea9299b070a13a867a3eの各原APIを参照し、locatorをINPUTSと比較JSONへ保持した。

## 固定SOURCE・再利用経路

- 今回の具体列は `N_X=N_Y=[59,55,51,47,43,39,35,31,27,23,19,13,1]`、allocation ID `P7_S2_16H_OVER7_M12_ONE_01`。2shot点からm12の両軸だけ2→1とし、元raw/normalization/scalarを変更・再適用していない。
- caller・[INPUTS](INPUTS.json)・人工検査を科学実行前に **cdf3b1f91ef5eb72f2de28ce10be853bb80f6217** へcommit/pushして固定した。callerは `scripts/run_p7_s2_m12_one_fixed.py`。結果公開SHA、元signal/価格/過去評価のSOURCEとは区別する。
- 元configuration `a6e557770a2e4d96585ad5ca5903c4626ae47dda9775e9a9babc756149c38d84`、total view `e496358b012be53878305a83a3bfed42ae4d444d3c0b6fe6dee061d2f44a7abc`、binding `d54a41b4520e36c02b306ad35715f4a731fef0fd17754a4d5c828badb756faad` を保持。
- 13round packet/PLANはcarrier06a15ee45beb177c018f6260ea4a5ac8cf66cfe6のp7/S2・16h/7、26軸価格は1002f1dfea34de439ca1b704a0192bca473c5387の同物理候補行。**16直接軸＋10条件付き閉包軸**、元receipt/physical ID・生成ID・compiler/loweringを維持する。[C_RESULT_JOIN](C_RESULT_JOIN.json)にallocation・軸別N/C/積・元価格行・結果SOURCEを対応付けた。
- 同じZ1＋T1 total decoder、保存full DF10 solver energy proxy −2.1663874486347634 Ha、保存deterministic rawのideal IID二項・EXACT_AFFINE_OF_BINARY64_RAW、全事象平方loss・exact τを使用。元Minkowskiは別参照、AδEは再加算しない。

§10.5の限定差分照合で同等1shot結果/進行中jobを未同定だったため、新規1件のみ取得した。[SCOPE_AND_PREFLIGHT](SCOPE_AND_PREFLIGHT.md)。既存worker/supervisor・結果検査・高速DPを再利用し、旧caller/kernel/結果を変更していない。固定1件の実行入口と保存先を人工supervisorで通す検査、metadata/固定列/費用の検査という新2件PASS。[TESTS](TESTS.txt)。実H4 riskは人工テストでは0件。

## 数値gate・保存・負荷

新格子のfinite binary64閉包は**2,555,118隣接組**で検証。実遷移86,427,423組、非終端frontier最大8,192、終端702状態。ρ=1、outcomes4,225、verification55,043,173、transitions141,559,036の固定scopeを維持した。

各round/first-unsafe分割の質量、残余loss、MSE上下界・moments、原API/RESULT整合を検査。残余mass約1.191987942062620E−31、残余loss上界約4.222733485506272E−30 Ha²を会計に残した。any-zero=0、追跡zero HOLD/tie argmin=0。元total decoderの規則は不変で、zero/tieの頻度をMSE寄与や誤branch率と同定しない。bias²約3.323480062309001E−7 Ha²、variance約4.393182985623966E−6 Ha²は診断として保存した。

数値判定は `MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD`、保存状態は別に `WORKER_ARTIFACTS_SAVED`。原API・coarse/lemma/total-DP partials・RESULT・RESOURCE_FINAL・PERSISTEDをworker自身で保存し、monitor errorなし、callerと記録済み子process97件は終了code0。[SUPERVISION](run01/SUPERVISION.json)。

worker wall142.702秒、supervisor wall142.748秒。worker CPU134.865秒、child CPU158.125秒。worker peak RSS603,248 KiB、完了child単体peak最大440,780 KiB、親monitorのsampled subtree peak3,882,148 KiB。原[RESOURCE_FINAL](run01/P7_S2_M12_ONE_FIXED/RESOURCE_FINAL.json)の測定範囲を保持し、単体peakの合計をRAM peakとしない。既存8 DP worker/batch8を利用した。

## 主張範囲とSTOP

上の4配分の条件付き価格/Q3比較である。閉包価格のcompiler/反復/終端等の前提、価格誤差上界/CI=null、solver proxyのground未認証、実sampler law移送未認証、engineering allowanceは厳密数値certificateでないことを保持する。十分N・global RMSE・G_qualified・方式winnerは未認定。

達成費用は**395,962,678 RZを維持**。旧適合点・早期終了・固定SOURCE・原結果・旧STOPを保全した。新signal/compile/価格fit/他配分Q3/retryは各0件。今回の1件終了後、追加探索へ進まず、公開・remote照合後 **mandatory STOP**。
