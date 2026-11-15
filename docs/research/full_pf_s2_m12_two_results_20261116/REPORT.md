# Full PF/S2・16h/7：m12各軸2shotの固定Q3結果

2026-11-16 JST、B2（Claude Code、主agentのみ。独立レビューではない）。

**2shotもQ3モデル適合。新規Q3は許可された1件だけ。Full PF/S2・16h/7の確認済み達成費用を615,304,502 RZから559,457,386 RZへ更新する。** retry・UNKNOWN・他配分の新規Q3は0件。4shot点は保存結果を引用し、再評価していない。

p7で費用削減につながった変更（m12を各軸4→2）は、Full PF自身の信号でも採択閾値を満たした。ただし余裕は小さく、MSE上界はτの約94.6%である。

## 4shotと2shotの比較

保存H4 linear／1.00 Å／STO-3G／tol-only 1e−8 DF10・sector(2,2)、Full PF/S2・要求幅16h/7（元configuration `11cf8ca6…`）。元H・state・snapshot、13roundのphysical schedule・scalar・raw・normalization・targetは不変。両点ともm0〜m11はX/Y各軸 `[59,55,51,47,43,39,35,31,27,23,19,13]` で、m12だけが異なる。

| m12のX/Y各軸shot | 条件付き総compiled RZ数 | Q3モデルMSE下界 Ha²（表示） | 上界 Ha²（表示） | 上界/τ | 判定・取得 |
|---:|---:|---:|---:|---:|---|
| 4 | 615,304,502 | 1.495783385949356167254620E−6 | 1.495783385949356167254624E−6 | 0.589 | 適合・`b4296b110`の保存API |
| **2** | **559,457,386** | **2.402471387794261100742341E−6** | **2.402471387794261100742346E−6** | **0.946** | **適合・新規1call** |

- 判定は exact τ=(0.00159360010199040 Ha)²=0.0000025395612850638132820416921600 Ha² と原APIのexact有理数による。表の小数は表示用。2shotの有効上界≤τを確認し、余裕は τ−上界 ≈ 1.371×10⁻⁷ Ha²。
- **差額：2shotは4shotより55,847,116 RZ低い**（m12のX/Y各2shot×13,961,779 RZ、4shot点の約9.08%）。m0〜m11の費用503,610,270 RZは同じ。
- 2点の比較であり、MSEの単調性・m12=1や他配分の適否・最小Nは推定しない。
- 今回の列は `N_X = N_Y = [59,55,51,47,43,39,35,31,27,23,19,13,2]`、allocation ID `FULL_PF_S2_16H_OVER7_M12_TWO_01`。

正本は[原API](run01/FULL_PF_S2_M12_TWO_FIXED/API_RETURN.json)、exact値の[EXACT_COMPARISON](EXACT_COMPARISON.json)、表示用の[SUMMARY](SUMMARY.json)、C向けの[C_RESULT_JOIN](C_RESULT_JOIN.json)（固定commitとパスは[HANDOFF](HANDOFF.md)）。4shotの原APIは `b4296b110b129d226dbef2743c14f725a27c3bf6` の `docs/research/pf6_cost_budget_q3_results_20261115/run01/FULL_PF_S2_16h_over_7__B_PRICE/API_RETURN.json`。

### 参考：p7の同じ変更との並び

別の物理候補なので直接の比較ではない。

| 方式 | m12=4 | m12=2 | 削減額 | MSE上界 2shot/4shot |
|---|---:|---:|---:|---:|
| Full PF/S2・16h/7（今回） | 615,304,502 | 559,457,386 | 55,847,116 | 1.606倍 |
| B0I p7/S2・16h/7（`93ab9dbbe`） | 435,481,090 | 395,962,678 | 39,518,412 | 1.603倍 |

m12=2同士では、Full PFはp7より163,494,708 RZ高い（4shot同士では179,823,412 RZ）。p7はm12=1で不適合だった（`cbc471c91`）。Full PFのm12=1は評価していない。

## 診断（判定には使わない）

| 項目 | 4shot | 2shot |
|---|---:|---:|
| bias² Ha² | 3.641×10⁻⁷ | 3.259×10⁻⁷ |
| variance Ha² | 1.132×10⁻⁶ | 2.077×10⁻⁶ |
| any-zero確率 | 0.00298 | 0.0364 |
| m12の追跡tie-argmin確率 | 1.72×10⁻⁵ | 1.18×10⁻⁴ |

MSEの増加はおもに分散の増加による。zero HOLD・tie argminは元のtotal規則で全事象lossに含めた（除外していない）。zero・tieの変化をshot数の単純な倍率則で外挿しない。

## 数値gate

- 新しいcount格子に対するρ=1の有限binary64閉包 `VERIFIED_FINITE_BINARY64_SUPPORT`（隣接組2,555,169）。4shot点の閉包PASSは流用していない。
- total状態DP：遷移組86,468,383、非終端frontier最大8,192、終端9,632状態（終端は非終端capの対象外）。固定範囲（outcomes 4,225・verification 55,043,173・transitions 141,559,036／call）の内側。
- 残余質量 約1.210×10⁻³¹、残余loss上界 約4.287×10⁻³⁰ Ha²（捨てずに上界へ会計）。
- 保存結果の検査（既存の `validate_result`）：first-unsafe分割＋all-good=1、各round good＋unsafe=1、保持MSE寄与＋残余loss上界=DP上界、moments区間とMSE上下界の整合、exact閾値判定、原APIとRESULT.recordの一致。
- transition allowanceは既存の1×10⁻¹² rad/transition（engineering allowance）で、厳密な数値certificateではない。

## SOURCE・条件付き会計

- 科学実行SOURCE：**`b7f147ad81c88568ff2793546689da959f0751c4`**（caller `scripts/run_full_pf_s2_m12_two_fixed.py`、[INPUTS](INPUTS.json)、人工テスト）。Q3の前にcommit・pushした（[SCOPE_AND_PREFLIGHT](SCOPE_AND_PREFLIGHT.md)）。結果公開commitとは別。
- 候補：元configuration `11cf8ca6677a7ce651a00d827d0948e758510f95876acd24bf4d826c2bb6ca5d`、total view `af18d37da2ffa763739fe6f01631c85516c210b8ce8e30fd949fbfaabf4755f5`、binding `69cc82b4dfc0d3f35020e5678f8202da2e5fcbb628a52646a0271c4fbe00dca0`。信号はcarrier `06a15ee4` のFull PF S2保存13round（再計算なし）。
- 価格：B2 `1002f1dfea` のFull PF自身の26軸（`AXIS_TERMS.json` rows 0〜25）、**16直接軸＋10条件付き閉包軸**。m12は条件付き閉包軸。元receipt・physical ID・価格生成IDは変更していない。費用quantityはP=0の測定付きfull controlled Hadamard wrapperのcompiled RZ数。閉包の固定compiler等の前提と、価格誤差上界／CI=nullを引き継ぐ。
- 評価条件：b4296b110のINPUTSと同一の statistical_contract・options・adopted_evaluation（テストで一致を確認）。Z1＋T1 total decoder、保存full DF10 solver energy proxy −2.1663874486347634 Ha、全事象平方loss、保存deterministic raw・ideal IID二項・EXACT_AFFINE_OF_BINARY64_RAW。AのδEは加算せず、元Minkowskiは別参照のまま。
- 経路：Bの既存PF worker（`run_pf6_cost_budget_q3_fixed.py`、b4296b110と同一bytes）、supervisor、cache照合・結果検査（`run_p7_s2_local_shot_search.py`、e7d24c219と同一bytes）を再利用し、高速DP（8 DP worker／batch 8）もそのまま使った。新規実装は固定1件のcaseを作る薄いcallerだけ。

## 実行・保存・負荷

- status `MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`、運用状態 `WORKER_ARTIFACTS_SAVED`。workerが原API・coarse／lemma／total-DP partials・RESULT・RESOURCE_FINAL・PERSISTEDを自分で保存した。SCIENTIFIC_CALL_STARTは1回、最終stageはEVALUATION_COMPLETE。
- caller終了code 0（[PROCESS_RETURN](PROCESS_RETURN.json)）、記録した子process 99件も全て終了code 0、supervisor・monitorのエラーなし（[SUPERVISION](run01/SUPERVISION.json)）。子process数は科学call数ではない。
- worker wall 144.66秒、supervisor wall 144.71秒。worker CPU 136.26秒、child CPU 156.77秒。worker peak RSS 605,392 KiB、完了child単体peakの最大434,196 KiB、親monitorが定期観測したprocess-tree RSSの最大3,894,988 KiB（厳密なpeakではない）。単体peakの和を同時RAMとして扱わない。
- 実行前テスト：新4件＋再利用経路の既存10件＝14件PASS（[TESTS](TESTS.txt)）。

## 資格の範囲とSTOP

Full PF/S2・16h/7のQ3モデル適合した固定配分の達成費用は **559,457,386 RZ**。これは保存ideal raw・IID二項・solver proxyと固定価格の前提に条件付く、2点比較での更新である。配分空間の最小費用、十分N、物理的global RMSE、正式G_qualified、方式全体のwinnerではない。ground/state/target包含・実sampler移送・数値certificateは未取得。

固定SOURCE・4shotの原結果・旧STOPを保全した。m12=1、m11の変更、振替、他の幅・公式・PRへは進んでいない。新規信号・compile・価格fit・他配分のQ3・retryは各0件。公開とremote照合の後 **mandatory STOP**。
