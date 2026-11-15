# Full PF/S2・16h/7：m12各軸1shotの固定Q3結果

2026-11-16 JST、B2（Claude Code、主agentのみ。独立レビューではない）。

**1shotはQ3モデル不適合（下界>τ）。新規Q3は許可された1件だけ。Full PF/S2・16h/7の確認済み達成費用は559,457,386 RZ（m12各2shot）のまま維持する。** 他roundを固定し、m12をX/Y同数・各軸1shot以上とする範囲では、**2shotが最安の適合点**である。retry・UNKNOWN・他配分の新規Q3は0件。4shot・2shot点は保存結果を引用し、再評価していない。

## 4・2・1shotの比較

保存H4 linear／1.00 Å／STO-3G／tol-only 1e−8 DF10・sector(2,2)、Full PF/S2・要求幅16h/7（元configuration `11cf8ca6…`）。元H・state・snapshot、13roundのphysical schedule・scalar・raw・normalization・targetは不変。3点ともm0〜m11はX/Y各軸 `[59,55,51,47,43,39,35,31,27,23,19,13]` で、m12だけが異なる。

| m12のX/Y各軸shot | 条件付き総compiled RZ数 | Q3モデルMSE下界 Ha²（表示） | 上界 Ha²（表示） | 上界/τ | 判定・取得 |
|---:|---:|---:|---:|---:|---|
| 4 | 615,304,502 | 1.495783385949356167254620E−6 | 1.495783385949356167254624E−6 | 0.589 | 適合・`b4296b110`の保存API |
| 2 | 559,457,386 | 2.402471387794261100742341E−6 | 2.402471387794261100742346E−6 | 0.946 | 適合・`a006b8e15`の保存API |
| **1** | **531,533,828** | **4.706720749798782654111633E−6** | **4.706720749798782654111638E−6** | **1.853** | **モデル不適合・新規1call** |

- 判定は exact τ=(0.00159360010199040 Ha)²=0.0000025395612850638132820416921600 Ha² と原APIのexact有理数による。表の小数は表示用。1shotは**下界がτを約2.167×10⁻⁶ Ha²上回る**（下界/τ ≈ 1.853）。閾値をまたいだ判定保留ではなく、保守的な上界だけが大きいわけでもない。
- 費用：m0〜m11は503,610,270 RZで共通。m12は各軸 N×13,961,779。1shotは2shotより27,923,558 RZ低いが、適合しない。
- **他round固定・m12のX/Y同数かつ各軸1shot以上の範囲で、2shotが最安の適合点**：この範囲で2shotより安いのはm12=1だけで、それが不適合だから。m12=3は2shotより高く、最安かどうかには影響しない（Full PFでは未評価）。
- 不適合の結論はこの1列に限る。m11以前の変更、X/Yで異なるshot数、振替を含む配分は評価していない。
- 今回の列は `N_X = N_Y = [59,55,51,47,43,39,35,31,27,23,19,13,1]`、allocation ID `FULL_PF_S2_16H_OVER7_M12_ONE_01`。

正本は[原API](run01/FULL_PF_S2_M12_ONE_FIXED/API_RETURN.json)、exact値の[EXACT_COMPARISON](EXACT_COMPARISON.json)、表示用の[SUMMARY](SUMMARY.json)、C向けの[C_RESULT_JOIN](C_RESULT_JOIN.json)（固定commitとパスは[HANDOFF](HANDOFF.md)）。2shotの原APIは `a006b8e15f86ee42ed66f89fbd1def721d027e87` の `docs/research/full_pf_s2_m12_two_results_20261116/run01/FULL_PF_S2_M12_TWO_FIXED/API_RETURN.json`、4shotは `b4296b110b129d226dbef2743c14f725a27c3bf6` の `docs/research/pf6_cost_budget_q3_results_20261115/run01/FULL_PF_S2_16h_over_7__B_PRICE/API_RETURN.json`。

### 参考：p7の同じ列

別の物理候補なので直接の比較ではない。

| 方式 | m12=4 | m12=2 | m12=1 |
|---|---|---|---|
| Full PF/S2・16h/7 | 615,304,502・適合 | 559,457,386・適合 | 531,533,828・**不適合**（上界4.707×10⁻⁶） |
| B0I p7/S2・16h/7（`93ab9dbbe`、`cbc471c91`） | 435,481,090・適合 | 395,962,678・適合 | 376,203,472・**不適合**（上界4.726×10⁻⁶） |

両方式とも、m12=2までは適合し、m12=1で不適合になった。

## 診断（判定には使わない）

| 項目 | 4shot | 2shot | 1shot |
|---|---:|---:|---:|
| bias² Ha² | 3.641×10⁻⁷ | 3.259×10⁻⁷ | 3.225×10⁻⁷ |
| variance Ha² | 1.132×10⁻⁶ | 2.077×10⁻⁶ | 4.384×10⁻⁶ |
| any-zero確率 | 0.00298 | 0.0364 | 0 |
| m12の追跡tie-argmin確率 | 1.72×10⁻⁵ | 1.18×10⁻⁴ | 0 |

MSEの増加はおもに分散による（bias²はほぼ変わらない）。1shotでは、zero HOLD・tieの追跡確率はどちらも0だった。MSE上界は2shotの約1.96倍。zero・tieの変化があるため、shot数の単純な倍率則で外挿しない。

## 数値gate

- 新しいcount格子に対するρ=1の有限binary64閉包 `VERIFIED_FINITE_BINARY64_SUPPORT`（隣接組2,555,118）。2shot点の閉包PASSは流用していない。
- total状態DP：遷移組86,427,423、非終端frontier最大8,192、終端702状態。固定範囲（outcomes 4,225・verification 55,043,173・transitions 141,559,036／call）の内側。
- 残余質量 約1.210×10⁻³¹、残余loss上界 約4.287×10⁻³⁰ Ha²（捨てずに上界へ会計）。下界とτの差（約2.17×10⁻⁶ Ha²）に比べて無視できる大きさ。
- 保存結果の検査（既存の `validate_result`）：first-unsafe分割＋all-good=1、各round good＋unsafe=1、保持MSE寄与＋残余loss上界=DP上界、moments区間とMSE上下界の整合、exact閾値判定、原APIとRESULT.recordの一致。
- transition allowanceは既存の1×10⁻¹² rad/transition（engineering allowance）で、厳密な数値certificateではない。

## SOURCE・条件付き会計

- 科学実行SOURCE：**`d6366b03d278836139107e311d45aa82b380140e`**（caller `scripts/run_full_pf_s2_m12_one_fixed.py`、[INPUTS](INPUTS.json)、人工テスト）。Q3の前にcommit・pushした（[SCOPE_AND_PREFLIGHT](SCOPE_AND_PREFLIGHT.md)）。結果公開commitとは別。
- 候補：元configuration `11cf8ca6677a7ce651a00d827d0948e758510f95876acd24bf4d826c2bb6ca5d`、total view `af18d37da2ffa763739fe6f01631c85516c210b8ce8e30fd949fbfaabf4755f5`、binding `69cc82b4dfc0d3f35020e5678f8202da2e5fcbb628a52646a0271c4fbe00dca0`。信号はcarrier `06a15ee4` のFull PF S2保存13round（再計算なし）。
- 価格：B2 `1002f1dfea` のFull PF自身の26軸（16直接軸＋10条件付き閉包軸、m12は閉包軸）。2shot点のprice_termsからNだけを差し替えた。閉包の固定compiler等の前提と、価格誤差上界／CI=nullを引き継ぐ。
- 評価条件：a006b8e15の2shot INPUTSをそのまま使い、caseだけ差し替えた（＝b4296b110と同一の statistical_contract・options・adopted_evaluation）。Z1＋T1 total decoder、保存full DF10 solver energy proxy −2.1663874486347634 Ha、全事象平方loss、保存deterministic raw・ideal IID二項・EXACT_AFFINE_OF_BINARY64_RAW。AのδEは加算せず、元Minkowskiは別参照のまま。
- 経路：2shot caller（a006b8e15と同一bytes）経由で、Bの既存PF worker・supervisor・cache照合・結果検査と高速DP（8 DP worker／batch 8）を再利用した。

## 実行・保存・負荷

- status `MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD`、運用状態 `WORKER_ARTIFACTS_SAVED`。workerが原API・coarse／lemma／total-DP partials・RESULT・RESOURCE_FINAL・PERSISTEDを自分で保存した。SCIENTIFIC_CALL_STARTは1回、最終stageはEVALUATION_COMPLETE。
- caller終了code 0（[PROCESS_RETURN](PROCESS_RETURN.json)）、記録した子process 97件も全て終了code 0、supervisor・monitorのエラーなし（[SUPERVISION](run01/SUPERVISION.json)）。
- worker wall 142.75秒、supervisor wall 142.80秒。worker CPU 134.90秒、child CPU 156.94秒。worker peak RSS 605,892 KiB、完了child単体peakの最大441,260 KiB、親monitorが定期観測したprocess-tree RSSの最大3,983,020 KiB（厳密なpeakではない）。実行中、別チャット（D）のp7 Yoshida4 Q3が並行して動いていたが、互いに触れていない。
- 実行前テスト：新4件＋再利用経路の既存12件＝16件PASS（[TESTS](TESTS.txt)）。

## 資格の範囲とSTOP

Full PF/S2・16h/7のQ3モデル適合した固定配分の達成費用は **559,457,386 RZ（m12各2shot）のまま**。これは保存ideal raw・IID二項・solver proxyと固定価格の前提に条件付く。配分空間全体の最小費用、十分N、物理的global RMSE、正式G_qualified、方式全体のwinnerではない。ground/state/target包含・実sampler移送・数値certificateは未取得。

固定SOURCE・4shot／2shotの原結果・旧STOPを保全した。m11の変更、振替、他の幅・公式・PRへは進んでいない。新規信号・compile・fit・他配分のQ3・retryは各0件。公開とremote照合の後 **mandatory STOP**。
