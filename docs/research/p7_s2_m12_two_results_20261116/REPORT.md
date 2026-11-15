# p7/S2・16h/7：m12各軸2shotの固定Q3結果

2026-11-16 JST。**2shotもQ3モデル適合。新規Q3は承認された1件だけで終了し、今回更新対象の条件付き達成費用を415,721,884 RZから395,962,678 RZへ更新する。** 4/3shotは保存結果を再利用。retry・UNKNOWN・他配分の新Q3は0件。旧局所探索の早期終了・原runは変更していない。

## 4・3・2shotの比較

保存H4 linear / 1 Å / STO-3G / tol-only1e−8 DF10・sector(2,2)、B0I p7/S2・要求幅16h/7。元H/state/snapshot、13roundのphysical schedule・scalar・raw・normalization・targetを保全。全点でm0〜m11はX/Y各軸 `[59,55,51,47,43,39,35,31,27,23,19,13]`、m12だけが異なる。

| m12のX/Y各軸shot | 条件付き総compiled RZ数 | Q3モデルMSE下界 Ha²（表示） | 上界 Ha²（表示） | 判定・取得 |
|---:|---:|---:|---:|---|
| 4 | 435,481,090 | 1.506660616251764083585358E−6 | 1.506660616251764083585362E−6 | 適合・b4296b110の保存API |
| 3 | 415,721,884 | 1.951734976728839865403189E−6 | 1.951734976728839865403194E−6 | 適合・0b33b4afdの保存API |
| **2** | **395,962,678** | **2.415680996446553638790769E−6** | **2.415680996446553638790773E−6** | **適合・新規1call** |

判定はexact τ=(0.00159360010199040 Ha)²=**0.0000025395612850638132820416921600 Ha²**と原APIのexact有理数による。表は表示用小数。2shotの有効上界≤τを確認した。3shot点から19,759,206 RZ、元4shot点から39,518,412 RZ低い。この3点では費用低下とともにMSEは増えているが、未評価Nの単調性・適否や最小Nを推定しない。

今回の固定列は `N_X=N_Y=[59,55,51,47,43,39,35,31,27,23,19,13,2]`、allocation IDは `P7_S2_16H_OVER7_LOCAL_R12_D2_01`。元B_PRICEからm12だけ4→2で、3shotのrunを再開したものではない。

正本は[原API](run01/P7_S2_LOCAL_R12_D2/API_RETURN.json)、exactの[EXACT_COMPARISON](EXACT_COMPARISON.json)、表示[SUMMARY](SUMMARY.json)。旧4shot原APIはb4296b110b129d226dbef2743c14f725a27c3bf6の `docs/research/pf6_cost_budget_q3_results_20261115/run01/B0I_P7_S2_16h_over_7__B_PRICE/API_RETURN.json`、旧3shotは0b33b4afdd285a99f610e59d8f5b77960a59a0d0の `docs/research/p7_s2_local_shot_search_results_20261116/run01/step01/P7_S2_LOCAL_R12_D1/API_RETURN.json`。いずれも再計算していない。

## SOURCE・条件付き会計

- 入力候補の正本は **e7d24c2192d4640137bdaf993a4e954a49f7932c** の `single_cases[1]`。今回のcaller・[INPUTS](INPUTS.json)・人工検査は科学実行前に **f60fafeca26f1cc65048c9a4f33e79434ca8fdc8** へcommit/pushして固定した。callerは `scripts/run_p7_s2_m12_two_fixed.py`。両SOURCEと結果公開SHAを区別する。
- 元configuration `a6e557770a2e4d96585ad5ca5903c4626ae47dda9775e9a9babc756149c38d84`、total view `e496358b012be53878305a83a3bfed42ae4d444d3c0b6fe6dee061d2f44a7abc`、binding `d54a41b4520e36c02b306ad35715f4a731fef0fd17754a4d5c828badb756faad` は不変。
- signal/PLANはcarrier06a15ee45beb177c018f6260ea4a5ac8cf66cfe6のp7/S2・16h/7保存13round。価格は1002f1dfea34de439ca1b704a0192bca473c5387の同候補26軸で、**16直接軸＋10条件付き閉包軸**を維持する。元receipt・physical ID・価格生成IDを付け替えない。[C_RESULT_JOIN](C_RESULT_JOIN.json)にallocation/axis/価格行/結果SOURCEを対応付けた。
- 費用quantityはP=0の測定付きfull controlled Hadamard wrapperのcompiled RZ数。閉包の固定compiler・反復/終端等の前提と価格誤差上界/CI=nullを保持し、無条件の価格保証や正式G_qualifiedへ昇格させない。
- 同じZ1＋T1 total decoder、保存full DF10 solver energy proxy −2.1663874486347634 Ha、全事象平方loss、deterministic raw・ideal IID二項・EXACT_AFFINE_OF_BINARY64_RAWを使用。AδEを再加算せず、元Minkowskiは別参照として保全した。

§10.5照合では今回2shotの同等保存/進行中結果を確認範囲で未同定で、不足分1件だけ取得した。[SCOPE_AND_PREFLIGHT](SCOPE_AND_PREFLIGHT.md)。既存のlocal候補・PF worker・kernel・高速DP・保存supervisorはbyte不変で再利用し、探索関数を起動していない。新2人工/metadata test PASS：[TESTS](TESTS.txt)。

## 数値gate・保存・負荷

新count格子の有限binary64閉包を**2,555,169隣接組**で確認。実遷移86,468,383組、非終端frontier最大8,192、終端9,632状態（終端は非終端cap対象外）。ρ=1、outcomes4,225、verification55,043,173、transitions141,559,036の固定scope内である。

各roundとfirst-unsafeの質量、残余loss、MSE上下界・moments、原API/RESULT整合を検査した。残余mass約1.191987942062620E−31、残余loss上界約4.222733485506272E−30 Ha²を捨てずに残した。any-zero約0.03683318242078534、m12の追跡tie-argmin約0.0001195015516590647で、zero HOLD/tie argminを元total規則の全事象lossへ含めた。未解決として除外していない。3shotとの偶奇差もあり、zero/tieの変化を単純なshot倍率則へ外挿しない。

bias²約3.359019922691624E−7 Ha²、variance約2.079779004177391E−6 Ha²は診断。engineering allowanceは既存1E−12 rad/transitionであり、厳密数値certificateではない。

数値判定は `MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`、保存状態は別に `WORKER_ARTIFACTS_SAVED`。原API・coarse/lemma/total-DP partials・RESULT・RESOURCE_FINAL・PERSISTEDをworker自身で保存。callerと記録済み子process99件は終了code0、monitor errorなし。[SUPERVISION](run01/SUPERVISION.json)。

worker wall144.464秒、supervisor wall144.506秒。worker CPU136.093秒、child CPU158.134秒。worker peak RSS606,196 KiB、完了child単体peak最大444,548 KiB、親monitorのsampled subtree peak4,002,092 KiB。原[RESOURCE_FINAL](run01/P7_S2_LOCAL_R12_D2/RESOURCE_FINAL.json)に測定範囲を保持し、単体peakの和を同時RAM peakと扱わない。既存8 DP worker/batch8を使用した。

## 達成費用の更新とSTOP

今回の更新対象であるp7/S2・16h/7のQ3適合固定配分の達成費用は**395,962,678 RZ**。これは3点の保存比較に条件付く更新であり、配分空間の最小費用や十分Nではない。ground/state/実sampler law移送・数値certificateは未認証、global RMSE/G_qualified/winnerは未認定のまま。

前回の早期終了と旧SOURCE・原結果・STOPを保全した。m12=1やm11変更、振替、他の幅・公式・PRへ進んでいない。新signal・compile・価格fit・他配分のQ3・retryは各0件。公開・remote照合後 **mandatory STOP**。
