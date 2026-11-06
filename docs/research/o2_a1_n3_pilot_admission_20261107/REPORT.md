# A1対B1-OL：752-shot予備検証 attempt02 結果

2026-11-07 JST、SERVER1 GPUA100-gleap-01。
**752/752 shotsを完了し、全bit・価格を保存。契約の探索的判定はX/YともINCONCLUSIVE（判定保留）。科学計算は終了し、公開照合後mandatory STOPする。**

## 対象・SOURCE・実行範囲

保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits/全4電子sector、L_D3/m0/q1/δ=.02/r1/K2/coefficient_atol0、X/Y、理想gate・保存DFモデル限定。N1 actual finite-grid Q/selected full policy、N2保証ground、認証済みexact Fraction engine、3-control/全5正層を維持した。実機測定ではなく、固定exact engineのBorn確率と53-bit wordによる測定bit生成である。

科学SOURCEは [cae22763c86cc54b1fd5336cd9e9652ac1b66516](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/cae22763c86cc54b1fd5336cd9e9652ac1b66516)。[実行仕様](EXECUTION.md)、PROFILE_LOCK.json、INPUT_BINDING.jsonを実入力前に公開し、19ファイルの本文/blob、tree `4e4c59f0617bea0481feb4f12649b249ba882610` とremote SHAを照合した。終了後もSOURCE/依存/native profile・保存入力bytesを照合した。

前attemptの結果8d9241c・0件STOPを保全し、ユーザーの技術修正・継続許可に基づいて一回再投入。前attemptはword/bitを消費していないため、同じcampaign `A1-N3-PILOT-H4-m0-q1-20261106-v1`、seed20261108、32domain/64子streamを使った。今回output/cacheはrun02として独立。旧397件、新B1339件、前attemptのcache/係数/乱数を学習・配分・評価へ取り込んでいない。

変更はCPU admission、実入力前の有限資源待機、attempt metadata。旧数値kernel・統計式・乱数/測定・価格profileを変更していない。人工52件PASS。SOURCEを固定してからconsume-once managerで一回起動した。

## 成立確認と探索的結果

A1学習は各軸5層×16、計160。X/Yともrank4。補正後range幅はX約3.125、Y約2.5でD_CV≤4を満たし、係数を評価前に凍結した。
B1-OL probeは各軸5層×8、計80。保存された質量・probe分散・価格proxyから、各軸の評価quotaを **9,107,4,4,4** に凍結。全5正層とfloor4を保持した。
A1はunconditional Q、B1-OLはdirect conditional Qで、各方式・各軸128の独立評価を実施した。

| 方式 | 軸 | 推定平均 | 診断Vhat(mean) | 評価RZ | 診断score=Vhat×評価RZ | zero-variance cell |
|---|---|---:|---:|---:|---:|---|
| A1 | X | 0.985917681 | 0.00024328269 | 613,714 | 149.306 | なし |
| B1-OL | X | 0.999897955 | 1.0475128e-9 | 615,321 | 0.0006445566 | 0,1,4 |
| A1 | Y | -0.077939674 | 0.0078391053 | 613,613 | 4,810.1769 | なし |
| B1-OL | Y | 0.072048329 | 0.0087119666 | 615,412 | 5,361.4488 | 4 |

数値は保存済み診断の表示値。[厳密な保存値・凍結結果](RESULT_SUMMARY.json)、[評価表](EVALUATION_TABLE.md)参照。
Xのpoint scoreはB1-OLが小さく、YはA1が小さい。しかしB1-OLでは両軸に全bit同値の層があり、sample varianceの0は真分散0の保証にならない。契約に従い**X/YともINCONCLUSIVE、全体もINCONCLUSIVE**とした。

Yの点推定の差や、Xの極小plug-in値を、統計的有意差・期待費用改善・一般的なCV優位性の証明にしない。係数の母集団での最適性も未保証。正式ε/α・精度保証付きcalibration/productionは今回の範囲外であり、未実施である。

## 全段階の実現RZと古典費用

| 方式 | 学習/probe shots | 学習/probe RZ | 評価shots | 評価RZ | 全shots | 全段階RZ |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 160 | 782,346 | 256 | 1,227,327 | 416 | **2,009,673** |
| B1-OL | 80 | 391,709 | 256 | 1,230,733 | 336 | **1,622,442** |
| 合計 | 240 | 1,174,055 | 512 | 2,458,060 | 752 | **3,632,115** |

今回のA1全段階費用はB1-OLより大きい。これは指定された学習/probe/evaluationを含む一campaignの実現値で、同精度の期待総費用比較ではない。全価格はQiskit1.3.0、basis(rz,sx,x,cx)、opt2、seed17、selected full interrogationの状態準備のみ除外。同一profileのcache hit496、miss/compile256。cache hitにも全RZを課金した。未価格bit・未完了trialは0。

実入力前の二窓CPU観測で127 workersを選択。実行時間は約70分、engine worker区間時間の和は250,491.399秒、compile区間の和は261.224秒。これらworker時間はperf_counterによるwall区間和で、CPU使用時間の測定ではない。親peak RSSは509,755,392 bytes（約486 MiB）、単一worker最大peakは471,846,912 bytes（約450 MiB）。共有pageを重複計上するRSS和を物理メモリpeakとは扱わない。

setupは3.572秒、親のpath生成区間は1.952秒、価格/cache/build/待機区間の和は5,347.421 worker秒（compile区間を除く）。fit2回・配分2回・平均集計4回の測時はDIAGNOSTIC_TIMING.jsonlに保存した。legacy raw_pipelineのfit/correction欄の0は未加算placeholderであり、実費0ではない。分散計算等の未分離の親overheadは親wallに含まれる。

128件のgroupに対して127 workersとなり、最後の1件が次の空きworkerを待つ構成になった。この境界は今回の起動設計の改善点で、四つの評価groupに待ち時間が生じた。科学SOURCEを実行中に変更せず完遂した。今後は実入力前の技術設計で、この丸めとgroup schedulingの境界を改善できる。CPU/RAM/workerへ別の固定上限は設けず、他process・共有設定への操作はしていない。

研究全体の既知支出は今回の比較と分ける。

| 実験 | 保存bit | 価格済 | 既知RZ | campaign総費用 |
|---|---:|---:|---:|---|
| 旧中断d97e916 | 397 | 397 | 1,935,700 | 未完了77 trialがありUNKNOWN |
| 新B停止36ee9cの保全資料 | 1,339 | 1,318 | 6,435,609 | 価格欠測21・bit未保存35がありUNKNOWN |
| pilot attempt01 8d9241c | 0 | 0 | 0 | 古典setupのみ、比較なし |
| 今回attempt02 | 752 | 752 | 3,632,115 | 保存752件のRZ完備 |

既知RZの研究累計は12,003,424。旧未完了分があるため研究全体の総費用はUNKNOWN。この支出履歴を新pilotの学習や価格取得に流用していない。

## 監査・証拠・保全

[metadata監査](FINAL_METADATA_AUDIT.json)：752 trialのunique identityとbit→price順序・record一致、全stage件数、32domain/64spawn、path64word上限を確認。消費wordは **1,876**。exact engine/Born各752、selected full build1,504、compile256。N1 preparation/CDF、N2 solve/spectral、engine再認証は全0。追加shots、refit、control変更、float/GPU置換、自動retryは0。

qualified cron cgroupでouter/manager/親を一回起動し、triggerを除去。親・managerはexit0、127 workers全exit0、signalなし、finalization_errorsなし。STOPはCOMPLETED/MANDATORY_STOP。追加のstage/campaignを起動していない。

[全698原本のarchiveとmanifest](../../../artifacts/o2_a1_n3_pilot_attempt02/2026-11-07/)に、全shot/bit/価格/path-word/journal/cache/attempt/call/manager・cgroup/終了記録を保存。全archive memberの展開byte hashを照合した。主要JSONは[科学run](../../../artifacts/o2_a1_new_target/2026-11-07/n3_pilot_752_run02/)にも公開し、詳細ログ/cache原本はarchiveに含める。live worktreeの原本を削除・変更していない。

[保全確認](PRESERVATION_CHECK.json)：元main HEAD/dirty/untrackedと3,087原本、旧停止実行1,086原本、前pilot29原本、旧SOURCE/STOP/grant/ledger/marker、C1凍結を維持。今回の固定SOURCE/依存/native profile・保存入力も終了後照合した。公開結果は固定commitに対応するが、CI実行や独立外部再現を主張しない。

## 到達点

技術修正から752-shot予備検証・結果保全まで完了した。3-controlのrank/range成立と全5層取得は確認できたが、A1が強いB1-OLに有望という判定には達していない。zero/all-equalセルと少数標本による不確かさを保持する。

公開remote SHA・本文/blob/tree照合後mandatory STOP。保証付き本検証、追加測定、別seed/campaignへ自動進行しない。
