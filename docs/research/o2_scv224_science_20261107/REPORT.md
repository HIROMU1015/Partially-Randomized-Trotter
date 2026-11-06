# O2-SCV224：224-shot予備科学検証の完了結果

2026-11-07 JST。**224 bit/224価格を完備、正常完了、mandatory STOP。**

科学SOURCE [`996e434e63ce9b1d14c6ae191649c9abe213cf47`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/996e434e63ce9b1d14c6ae191649c9abe213cf47) を公開照合後、一campaignだけ実行した。
新しい実装は固定準備560c6092への薄いactivationで、旧科学SOURCE・exact kernelは変更していない。

## 固定条件・preflight

保存linear H4、1.00 Å、STO-3G、legacy DF12、8qubits/全4電子sector、L_D3、m0/q1/δ=.02/r1/K2/coefficient_atol0、X/Y、理想gate。
N1 `744e1a0` actual finite-grid Q/selected full、N2 `5e9bd08` 保存保証state/witness、engine `fdf00ce` のexact Fractionとstrict座標誤差証明を再利用した。
理論基準は元Hの厳密ground。数値実行stateとの差を既存engine保証（保存strict有理数上界、表示上界7.8079008e−7）内に区別する。H_D ground/PF有効stateへ変更していない。
対象は有限時間signalの探索で、RPE energy精度・全round総費用・実機noiseの検証ではない。

[実行仕様](EXECUTION.md)、[SOURCE](SOURCE.json)、[環境lock](PROFILE_LOCK.json)、[入力binding](INPUT_BINDING.json)。
activation12件＋準備済106件＝**人工118件PASS**。初回mock testはcollectorのtimer属性不足だけを補って再検証し、初回logも保存。科学入力を使う前の修正で、科学retryではない。
公開SOURCE19ファイルの本文/blob/tree・remote SHAを照合し、依存/native profileとN1/N2/engineの固定byte identityを検査してから起動した。

## Metadata群選択と凍結

一回のmetadata-only規則から **S={df-fragment-3}** を選択した。正質量fragmentの文字列辞書順、非空proper subset、ν1∈[1/8,7/8]、1/2最近接、同点は整数mask昇順を変更していない。
群選択にbit・Born・state・旧pilot情報を渡していない。群・全ν/moment/Σ・input/control identityをtrain前に保存した。

| 層 | ν_h（表示） | E[u²|h] | Σ_h（表示） |
|---|---:|---:|---:|
| 0 | 1 | 1 | 0 |
| 1 | 0.842715247177 | 0.842715247177 | 0.132546259352 |
| 2 | 0.850553287849 | 0.850553287849 | 0.127112392378 |
| 3 | 0.850553287849 | 0.850553287849 | 0.127112392378 |
| 4 | 0.850553287849 | 0.850553287849 | 0.127112392378 |

ν1のexact値は `7212243195074438/8558339509381393`。h0はpath固定でcontrol定数。control identity `1a94746a015655b69e36b5dca13e8016b0d04247a59e57b80d702de69034818c`。
[exact表・全5質量](../../../artifacts/o2_scv224/2026-11-07/campaign_run01/METADATA_CONTROL.json)。

## 取得件数・学習

X h1 train32→Y h1 train32→両軸fit/freeze→X評価80→Y評価80を実行。各評価quotaは(4,64,4,4,4)。
全5正層を保持し、学習64件＋共通fresh評価160件＝**224 physical shots**。評価160件をB1-OLとSCVが共有するpaired設計で、二つの独立campaignではない。

| 軸 | train u=0 / u=1 | rank | 凍結λ1（表示） | D1=2+|λ1|（表示） |
|---|---:|---:|---:|---:|
| X | 4 / 28 | 2 | 1.01641576466e-16 | 2 |
| Y | 2 / 30 | 2 | 0.0666666666667 | 2.06666666667 |

両軸rank2・finite slope・D1≤4を通過。h0/2/3/4のλ=0は取得前固定。clip/refit/追加shotなし。
Xのλは約1.0164e−16で、OLSが出したbinary64値をそのままFractionへ凍結した。事後に0へ丸めていない。
Y trainのu=0は2件、評価では4件だけであり、係数と相関の学習精度には大きな不確かさが残る。balance条件だけでは十分なpowerを保証しない。

## Paired評価：点指標と限界

以下のVhatは各層の不偏sample varianceをΣp_h²s_h²/n_hで合成した**推定量分散の点推定**。真variance・有限confidence上界ではない。

| 軸 | B1-OL平均 | SCV平均 | Vhat_OL | Vhat_SCV |
|---|---:|---:|---:|---:|
| X | 0.99993027462 | 0.99993027462 | 1.39572692233e-09 | 1.39572692233e-09 |
| Y | 0.0593507420493 | 0.0533470637 | 0.0151001791303 | 0.0150964477577 |

| 軸 | paired Covhat | Vhat(推定差) |
|---|---:|---:|
| X | 1.39572692233e-09 | 1.25765614978e-35 |
| Y | 0.0150964477577 | 3.7313725623e-06 |

Xのh1 raw bitsは全同値でsample variance0、残差の微小分散は非零の凍結OLS丸め係数に由来する。h1 raw0を母分散0としない。表示桁では全targetのVhatは同じで、追加削減は観測されない。
Yのh1 raw sample varianceは1.0119047619、residualは1.01164021164。全targetのVhat低下は**約0.024711%**という小さい点差だった。
この有限標本では相関・分散差の精度保証を取得していない。両軸に共通する明確な利益、一般的CV優位性、厳密な期待費用改善へは結び付けない。
B1-OL参照は今回の同じ固定quota版であり、最適配分B1-OLを上回ったという比較ではない。
[層別raw/residual variance・feature count・all-equal・paired covariance](RESULT_SUMMARY.json)と[全記録archive](../../../artifacts/o2_scv224/2026-11-07/FULL_RUN.tar.gz)を保全した。

## 全段階の実現compiled RZ

Qiskit1.3.0、basis=(rz,sx,x,cx)、opt2、seed17、selected full no-prep。状態準備だけを除外し、全224件を一回ずつ課金した。131 cache hitも量子RZ全額を計上。

| 軸 | train RZ | 共通評価/B1-OL参照 RZ | SCV参照/physical RZ |
|---|---:|---:|---:|
| X | 153,545 | 386,274 | 539,819 |
| Y | 153,445 | 385,977 | 539,422 |
| X/Y合計 | 306,990 | 772,251 | 1,079,241 |

B1-OL参照772,251とSCV参照1,079,241は重複する評価を含むため、合算して物理支出にしない。**実physical支出は1,079,241 RZ**。学習306,990 RZをSCVから除外しない。
学習込みRZは評価のみ参照より約39.7526%多い。今回のVhat×各手法実現RZという探索的point scoreは以下。

| 軸 | B1-OL Vhat×C_eval | SCV Vhat×(C_train+C_eval) |
|---|---:|---:|
| X | 0.000539133021195 | 0.000753439911483 |
| Y | 5828.32184018 | 8143.35604237 |

**このpoint scoreでは両軸とも学習費込みSCVが大きく、今回の学習費を回収する傾向は確認できなかった。** 事後のGO/NO-GO閾値を設定した判定ではなく、保存された指標の方向の記述である。
実現価格と期待価格、Vhatと真variance、有限時間signalと最終RPE総資源を区別する。今回で手法の一般的無効性・有効性を確定しない。

## 古典費・負荷

親wallは **1455.440秒（約24.26分）**。metadata群/平均表は0.159秒、saved bundle decodeは3.648秒。
実入力前の二窓idle観測で127 workerを選択。契約の逐次stageによりtrainは32件、評価は80件を各並列処理し、他workerは待機。任意のCPU/RAM/worker上限を新設せず、exact kernel・共有設定・他ユーザーprocessを変更していない。
engine worker wall区間の和は68334.761秒、compile区間和は97.642秒、price/cache/build/待機区間和（compile除外）は1142.827秒。CPU時間やcampaign wallとは区別する。
path生成親区間は0.554秒。fit X/Yは0.214/0.165秒、集計は0.390/0.411秒。
parent peak RSS 510,152,704 bytes、単一worker最大peak RSS 450,035,712 bytes。共有pageを含むprocess別RSSを加算して物理peakと呼ばない。
legacy raw_pipelineのfit/correction=0は未加算placeholder。新driverの独立timerを上記の観測値とし、未分離の親/manager overheadはUNKNOWNまたは各wallに含まれる。
読取専用監視では資源の枯渇・OOM・強いpressureを観測していない。微小で一時的なpressureは監視記録に残した。全時間・他ユーザーへの影響を証明したとはしない。

## Call・終了・原本保全

記録照合：metadata群/ν/Σ一回、saved bundle decode一回、stream book一回、224 unique trial、exact engine/Born各224、selected full build448、compile93、uniform53 word474。全call cap内。
N1 preparation/CDF再計算、state solve/eigh/spectral、engine再認証、追加shot/control/seed、float/GPU置換、science retry/追加campaignは全0。
bit→price順序とrecord identity、12domain/24spawn、word cap、全5正層、凍結係数、全tail分類、source/input byte一致を[metadata照合](FINAL_METADATA_AUDIT.json)で確認。
親・manager exit0、127 workers全exit0、signalなし、finalizer errorsなし、全224 attemptはBIT_AND_PRICE_DURABLE。一時cron撤去済み、自動再起動なし。
[全530原本のarchive・manifest](../../../artifacts/o2_scv224/2026-11-07/RAW_MANIFEST.json)とarchive全memberのbyte照合を保存。詳細worker/word/cache/manager記録を含み、live原本は削除・上書きしない。
[保全照合](PRESERVATION_CHECK.json)：旧SOURCE2918原本（README索引2件だけ追記）、元main HEAD/dirty/untrackedと3,087原本、旧STOP実行1,086原本、前pilot29原本、完了752-shot raw698原本、固定契約/input201原本を保全。旧grant/ledger/marker/STOP・C1凍結維持。

## 到達点と次の最小判断

層内で変化するmetadata control、exact条件付き中心化、rank/range、fresh paired評価、正しい学習費込み会計を実H4で接続できた。
今回の小さいY点差とほぼ零のX効果は、学習費込みの明確な利益を示さなかった。低いfeature出現数と少数評価・希少層の4件という不確かさを維持する。
次に必要なのは、この小さな点差と学習費を踏まえて、別の有限検証を設計する価値があるかという研究判断。今回の結果だけで追加shot・別control探索・保証付きcalibration/productionへ進まない。
精度/成功確率・期待RZ費用の保証は未取得。詳細科学レビューや正式GO/NO-GOは実施していない。
結果を専用branchへcommit/pushし、remote SHA・本文・blob/treeを照合後 **mandatory STOP**。旧実験や本campaignを再開しない。
