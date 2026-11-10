# H6 DF19 RPE m6：run02同条件再実行

2026-11-10 JST。**保存結果の再利用を報告した後、利用者から同じm6範囲の実行指示を改めて受け、前回run01を保全した別run02として一回実行した。PF・PRともideal条件付きPASS、各軸NはPF705、PR772。** 新しい物理条件や候補を追加していない。

科学SOURCEは **`49523b11db9f7b6fc065e506424c22732cd86e96`**。数値kernel・runner・固定PLANの14 file hashはrun01と同一で、SOURCE commitの差は文書変更だけ。今回の正本は[result.json](result.json)、[実行出力](../execution_run02.txt)、[照合記録](CHECKS.txt)。[前回REPORT](../REPORT.md)、run01原本、COVERAGE、価格接続メモ、過去STOPを変更しない。

## 固定条件とcall範囲

Linear H6/1.00Å/STO-3G、保存tol-only1e−8 DF19、weighted-Hermitian completion、12 system mode・Nα=Nβ=3のsector400、保存DF-H同sector ground target、保存数値state条件付き、P0。CA=.00159360010199040 Ha、h=.1、βtotal/stat=.4/.2、各方式αtotal=.05、αaxis=.05/26、m0..12の13round、固定prior・無shiftは前回と同じ。H/state/snapshot binding、signed weights、fragment順、scalar、tail identity、normalizationを維持した。

| arm | candidate ID | 構成 |
|---|---|---|
| PF | `201a8e365ab6f4af6ebc` | Full19、m5_best、q_PF4、δ1.6 |
| PR | `ef41b65942d56977d6cd` | prefix7、m5_best、R_outer16/K2、q_PF4、δ1.6 |

共通m6/T6.4/q_RPE64。q_PF4とは別変数。既存11 signed piece・ABS_TIME_PER_OUTERを使い、PR配分`[1,2,1,2,1,2,1,2,1,2,1]`は合計16/outer、全event数64。R16を各pieceへ重複適用しない。

今回の科学callはreference1時点/2経路、finite mean2候補、PR exact-tail内部対照1、scalar2。旧m0..5、prior/horizon/α/βは保存値の読取のみ。新eigensolve、SCF/DF/state生成、sampling、build/compile、価格照会/fit/予測、N×C、全RPE成功率simulationは全て0。失敗に対する自動retry、閾値変更、候補差替えはない。

## Reference・signal・phase・radius・N

| reference経路 | complex signal |
|---|---|
| expm_multiply | -0.28640442981123+0.95810881562822i |
| expm | -0.28640442981123+0.95810881562822i |
| 経路差 | 2.278663452567752e-15 |

PR corrected/rawは独立列で伝播した。以下は表示を丸めた値、全binary64値とbias分解は正本に保持する。

| arm | corrected | raw | total signed bias |
|---|---|---|---|
| PF | -0.28650023419895+0.95807917202979i | -0.28650023419895+0.95807917202979i | -9.5804387722331e-05-2.9643598425722e-05i |
| PR | -0.28650023316152+0.9580791694574i | -0.27374626029205+0.91542888746904i | -9.5803350293022e-05-2.9646170818465e-05i |

| arm | log B | B | raw半径下界 | βdet rad | corrected位相差 rad | N_X / N_Y |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0 | 1 | 0.999899713014 | 0.000100286986214 | 0.000100281182544 | **705 / 705** |
| PR | 0.045537729978 | 1.04659049171 | 0.955387729187 | 0.000100286741029 | 0.000100280925595 | **772 / 772** |

位相差の基準はexp(−i E_state T)であり、E_stateをpriorやdecoder branchの決定へ使っていない。既存のempirical u/roundoff・reference経路差・T×保存residual・log_B marginを含むdiskから旧geometry/Hoeffdingへ接続した。両方式で正半径とβdet+βstat≤.4が成立し、独立式でもshot整数が一致した。PR vector closureは約5.08e−15。量子shotを生成した回数ではなく、条件付き十分Nである。

**actual N_X/N_Yは両方式null、C/Gもnull。** 厳密ground同定、actual implementation/law距離、rigorousな誤差上界、全13round成功・総費用は未資格。今回の再実行でそれらが新たに保証されたとは扱わない。

## run01との一致・coverage・価格接続

reference全欄とcandidate row全欄（signal/bias/normalization/disk/phase/radius/Nを含む）はrun01と一致した。14 file hash、固定設定、input bindingも一致する。run01 wall約11.961秒・59 worker、run02 wall**11.831秒**・63 worker。今回は観測affinity64/load約.625/available RAM約526.6 GBを用い、120共通半step指数を並列cacheした。BLASはprocess内1 threadで入れ子を避け、共有設定を変更していない。

これは同じ数値手法の決定論的再実行で、独立な量子測定・trajectory標本や別担当による科学レビューではない。run01/run02のNを合算せず、統計的な標本数や保証水準を増やしたことにしない。既存人工3件はSOURCEが同一のため記録を再利用した。

m0..5は再計算せず、coverageは引き続き**7個の異なるround m0..6**。[COVERAGE.json](../COVERAGE.json)の保存Nは各軸PF `705/705/705/705/705/705/705`、PR `705/708/719/765/721/737/772`で、m6を二重に追記しない。scope中の`D1_D2_point_re_evaluations=0`は旧baseline95d01b6aのm0..5を指す。m6については今回run02で再実行したことを本書に明記する。

価格担当向け`m6_price_handoff`も同じcandidate/binding・signed配分・Nをそのまま利用できる。価格設計4c965ff6との静的対応は[接続メモ](../REUSE_AND_PRICE_CONNECTION.md)を参照。Cの存在・model資格は本作業で判定せず、Bや古典価格標本数を価格へ追加乗算しない。

## 保全とSTOP

今回の1 campaignと前回run01の1 campaignを分け、m6の実行履歴は計2 campaignとして残す。前回原本を破棄・置換せず、SOURCE/input/raw/結果/grant/ledger/marker/KEEPと過去STOPを保持した。run02/progress.jsonは途中checkpoint、最終状態はresult.jsonを参照する。科学SOURCE・input・previous-resultsの実行中保全判定と、終了後の原本bytes・数値出力・N整数・scopeの照合をPASSと確認した。

残りはm7..12/T12.8..409.6の6roundであり、今回の再実行はその評価を含まない。次の最小案は既存REPORT記載のm7/T12.8/q_PF8/δ1.6のFull19対PR7/R16/K2の2候補だが、未実行・未承認である。

公開remote SHA・主要成果物の全文一致を確認後**mandatory STOP**。**ユーザー**が次の科学範囲を判断し、**本担当Codex**は追加許可までm7以降・価格評価・詳細科学レビューへ進まない。
