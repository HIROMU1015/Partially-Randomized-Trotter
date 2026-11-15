# p7/S2・16h/7：保存適合点からの局所shot探索結果

2026-11-16 JST。**改善点あり。実際の新規Q3は1件、最初の候補が「出発点より厳密に安い＋Q3モデル適合」を満たしたため早期終了した。** 停止理由は `FIRST_STRICTLY_CHEAPER_MODEL_FEASIBLE_POINT`。retry・UNKNOWN・振替評価は0件。後続候補の科学投入は行っていない。

## 得られた固定配分

H4 linear / 1 Å / STO-3G / 保存tol-only1e−8 DF10、sector(2,2)、B0I p7/S2・要求幅16h/7、元H/state/snapshot・全13round schedule・scalarを保持した。**m12のX/Yを各軸4→3とし、それ以外は元B_PRICE点のまま**。

```text
m = 0..12
N_X = N_Y = [59,55,51,47,43,39,35,31,27,23,19,13,3]
allocation_id = P7_S2_16H_OVER7_LOCAL_R12_D1_01
```

| 指標 | 元B_PRICE（保存引用） | 今回の最初の候補 |
|---|---:|---:|
| m12の各軸N | 4 | 3 |
| 条件付き総compiled RZ数 | 435,481,090 | **415,721,884** |
| 元点からの削減額 RZ | — | **19,759,206** |
| Q3モデルMSE下界 Ha²（表示） | 1.506660616251764083585358E−6 | **1.951734976728839865403189E−6** |
| Q3モデルMSE上界 Ha²（表示） | 1.506660616251764083585362E−6 | **1.951734976728839865403194E−6** |
| モデル判定 | 適合 | **適合** |

exact τ=(0.00159360010199040 Ha)²=0.0000025395612850638132820416921600 Ha²。有効上界≤τを原APIのexact有理数で判定した。表は表示用の丸め値。今回のMSEは元点より大きいが、共通モデル閾値の範囲内で費用を削減できた。

元点は **b4296b110b129d226dbef2743c14f725a27c3bf6** の `docs/research/pf6_cost_budget_q3_results_20261115/run01/B0I_P7_S2_16h_over_7__B_PRICE/API_RETURN.json`。再評価していない。今回の正本は[原API](run01/step01/P7_S2_LOCAL_R12_D1/API_RETURN.json)、exactの[SEARCH_EXACT](run01/SEARCH_EXACT.json)、表示[SUMMARY](SUMMARY.json)、短表[SEARCH_HISTORY.csv](SEARCH_HISTORY.csv)。

## 探索順・実装・再利用

採択計画 **9992c594d4c72d1e8b65b42029d2b231641c0d17** を実装した。単roundは元seedからm12−1、m12−2、m11−1、m11−2の順、全て未達の場合だけ計画のfamily優先度で振替へ進む。元N0 cap・1以上、最大2round編集、1/2shot操作、直接全13round Q3、最大8件、最初の厳密費用改善かつ適合で停止する。MSEの単調性・round独立性・1/N則を仮定していない。

今回は第1案だけを実行。m12：4→2、m11：13→12/11と振替poolは**未評価**であり、適否・最小N・局所最適性を推定しない。seed・A_SHAPEなど既評価配分を再利用catalogへ登録し、同等SOURCE/配分/decoder/target/law/loss・数値契約を検査する経路を用意した。§10.5の限定履歴・job照合で今回第1案の保存/進行中同等結果を未同定だったため、不足の1件だけを取得した。別幅hや他方式のQ3を転用していない。

実行前にcaller、全有限候補の具体列・価格参照・費用/削減額、選択規則、count-only、テストを **e7d24c2192d4640137bdaf993a4e954a49f7932c** にcommit/pushして固定した。callerは `scripts/run_p7_s2_local_shot_search.py`。結果公開SHAとは区別する。各stepのQ3前に[SELECTION](run01/step01/SELECTION.json)へ親seed、全X/Y列・費用、選択理由・順序を保存した。適応分もcommit済み有限poolから選ぶ設計で、科学投入は常に1件ずつである。

既存PF metadata/total binding・Q3 kernel・高速DP・worker自身の原API/partials/資源保存・supervisorを再利用し、追加部分は候補順序と履歴処理。新8件＋既存worker/monitor回帰6件＝**14 test method PASS**。[TESTS](TESTS.txt)。人工入力で1shot減不適合→2shot減適合、早期終了、8件上限、振替優先度、INCONCLUSIVE/UNKNOWN、保存再利用とSOURCE拒否を確認した。旧decoder・DP kernel・元科学結果を変更していない。

## 価格と物理SOURCE

- 元configuration：`a6e557770a2e4d96585ad5ca5903c4626ae47dda9775e9a9babc756149c38d84`。total view：`e496358b012be53878305a83a3bfed42ae4d444d3c0b6fe6dee061d2f44a7abc`。binding：`d54a41b4520e36c02b306ad35715f4a731fef0fd17754a4d5c828badb756faad`。
- 保存signal/PLAN：`06a15ee45beb177c018f6260ea4a5ac8cf66cfe6:docs/research/pf_shortlist_mean_completion_20261115/run01/B0I_P7_S2_16h_over_7/`。元signal SOURCE・raw/normalization・targetは不変。要求幅からscheduleを再生成していない。
- 価格：`1002f1dfea34de439ca1b704a0192bca473c5387:docs/research/pf6_allround_cost_20261115/AXIS_TERMS.json` の同候補・m/axis。p7/S2の**16直接軸＋10条件付き閉包軸**を保持。今回費用の内訳は直接部分81,621,086 RZ、条件付き閉包部分334,100,798 RZ。元価格receipt/生成IDをtotal IDへ付け替えていない。
- 費用はP=0、測定付きfull controlled Hadamard wrapperのcompiled RZ数。閉包のcompiler/障壁・反復・終端構造等の前提、価格誤差上界/CI=nullを引き継ぐ。正式G_qualifiedや無条件の価格保証へ昇格しない。[C_RESULT_JOIN](C_RESULT_JOIN.json)に元/total ID・physical ID・allocation・全26軸積・価格/結果SOURCEを保存した。

## 数値gate・運用状態

新格子の有限binary64閉包は2,555,220隣接組で検証。実DP遷移86,492,959組、非終端frontier最大8,192、終端2,091状態。ρ=1、outcomes4,225、verification55,043,173、transitions141,559,036の固定scope内。各roundの質量、first-unsafe分割、残余loss、上下界・moments、原API/RESULT整合を確認した。

残余mass約1.191987942062620E−31、残余loss上界約4.222733485506272E−30 Ha²を会計に残した。any-zero=0、追跡zero HOLD/tie argmin=0で、元total規則を変更していない。偶奇の変化も含む今回の1点の診断であり、他Nへの単調性を推論しない。bias²約3.553519051803125E−7 Ha²、variance約1.596383071548527E−6 Ha²は診断として保存した。

科学判定 `MODEL_BOUND_MEETS_ADOPTED_THRESHOLD` と保存状態 `WORKER_ARTIFACTS_SAVED` を分離して記録。原API・coarse/lemma/total-DP partials・RESULT・RESOURCE_FINAL・PERSISTEDをworker自身で保存した。[SUPERVISION](run01/step01/SUPERVISION.json)ではmonitor errorなし、記録済み子process99件と親callerは終了code0。科学call数は1件で、子process数とは異なる。

worker wall142.465秒、supervisor wall142.507秒。worker CPU134.548秒、child CPU156.641秒。worker peak RSS606,224 KiB、完了child単体peak最大443,636 KiB、親monitorのsampled subtree peak3,991,372 KiB。各値は原[RESOURCE_FINAL](run01/step01/P7_S2_LOCAL_R12_D1/RESOURCE_FINAL.json)記載範囲の観測であり、個別peakの合計をRAM peakと扱わない。候補は逐次、Q3内の既存8 DP worker/batch8を再利用した。

## 主張の範囲と停止

保存deterministic raw・ideal IID二項・保存DF10 solver energy proxy・Z1＋T1・全事象平方loss・exact τという同一条件の下で、費用改善の適合点を1件確認した。十分N・最適N・物理的ground RMSE・方式winnerではない。ground/state/実sampler law移送と数値certificateは未認証、engineering allowanceは不変。AδEを再加算せず、元Minkowskiは別参照として保全する。

初期目標達成で終了し、残りの科学枠7件を消費していない。別幅・公式・PR、追加compile/fit/MC、retryなし。旧適合点・SOURCE・原結果・旧STOPを保全し、公開・remote照合後 **mandatory STOP**。
