# PR境界拡張：固定3候補の取得・検証結果

linear H4/H6、隣接1 Å、STO-3G、tol-only1e-8、保存rank10/19。H6保存weighted-Hermitian completionを維持。T=2、δ=2、m5_best、11 signed piece、q1、H4 prefix3/K2・H6 prefix7/K6、ABS_TIME_PER_OUTER。元H/state/T2 reference/scalar/generation順、finite-RTE/PCG64、symmetric-directional/actual adjoint、cosine/+Im sine測定込みfull wrapper、P=0、Qiskit1.3.0/NumPy1.26.4、rz/sx/x/cx/opt1/seed17。有限時間signalの経験的allowance条件付き費用であり、全QPE/RPE評価ではない。

## H8：原記録不足で停止

q16 ε=.001の探索n8保存G=40802612537883/2、SE=50209504581.336464と、.0005の独立n48価格は存在する。しかし正確な.001 A_signal原記録（N/bias/allowance/headroom等）を回収できなかった。原privateパスは現環境に存在せず、local索引・exporter archiveにも該当原記録がなかった。SOURCE/index/保存結果の範囲は公開済みPhase A監査とローカル回収記録を参照。Nの逆算・.0005からのスケール、新signal/sampling/compileは行っていない。n48の.001再会計および新しいq8対q16比較は成立せず、H8旧STOPを維持する。

## H4/H6主評価 ε=.0005

|候補|cohort|n|N_cos / N_sin|logB|C_cos / C_sin|G ± paired SE|固定対照との差の±2SE診断|
|---|---|---:|---|---:|---|---|---|
|H6 R11 K6|BOUNDARY_20261027_EXPLORATION|229|82807345 / 113255472|0.0261743913|176733.9475982533 / 176733.9475982533|34650955625643.92 ± 7117429335.464949|INCONCLUSIVE|
|H4 R128 K2|BOUNDARY_20261027_EXPLORATION|233|96764648 / 80505864|0.0257196736|46134.7339055794 / 46134.7339055794|8178327900425.819 ± 23051915627.406155|HIGHER|
|H4 R16 K2|BOUNDARY_20261027_EXPLORATION|187|145036571 / 114372825|0.220221011|34254.005347593586 / 34254.005347593586|8885810837800.021 ± 6535815870.543033|HIGHER|
|H6 R11 K6|BOUNDARY_20261027_CONFIRMATION|229|82807345 / 113255472|0.0261743913|176731.54148471614 / 176731.54148471614|34650483876245.812 ± 8617501252.260885|INCONCLUSIVE|
|H4 R128 K2|BOUNDARY_20261027_CONFIRMATION|1185|96764648 / 80505864|0.0257196736|46492.60675105485 / 46492.60675105485|8241768202974.15 ± 10362735296.841389|HIGHER|
|H4 R16 K2|BOUNDARY_20261027_CONFIRMATION|745|145036571 / 114372825|0.220221011|34281.51946308725 / 34281.51946308725|8892948257881.707 ± 3422683197.6427016|HIGHER|

全5精度の適格性、N、両軸単価、G/SE、元PF/PR点参照、固定R対照と差はRESULTS.csvに保存。R16は元signal/Nをそのまま使用し、新signal/shot評価0。R128/R11は各1候補を新評価し、少なくとも1精度で適格な場合だけ費用を取得した。各物理候補の価格は複数精度へ会計し、独立実験数を精度数倍に数えない。

H4はR32 E9、R64 E33/C800を再取得せず参照し、R16/R128のEとCを分離した。R32に独立確認はないためC順位の保証には使わない。H6はR16 E9/C33を分離し、R11と比較した。原PF・PR最良は各system/T/epsilonのALL公式取得済み域に固定し、今回結果との点比較を保存。既存33/40を変更していない。

## 標本数・不確かさ

Eは保存paired分散・共分散からf1/f4を比較し、f4、相対MCSE .5%を選択。元finite-bit lawからorder≥2の見逃し確率を名目5%以下とする標本数も取り、最大値を事前固定した。C規則はE開始前に固定し、max(EのCV²,保存analogue CV²)、f4、相対MCSE .25%、同rare guardを適用。CはE勝者だけを選ばず、適格かつ技術PASSの固定候補すべてを確認した。E_PLAN.json/C_PLAN.jsonにnとseed registry digestを保存し、C結果によるtop-upは0。

±2 combined-SEは新旧両側の不確かさを含む正規近似感度診断。LOWER/HIGHERはその診断方向であり、厳密CI・多重比較保証・母平均順位ではない。INCONCLUSIVEを許容する。旧参照のSEは新候補の追加nでは消えない。rare guardは少なくとも1回のorder≥2観測を計画するだけで、高次2/4/6の費用tail上界ではない。実際の未観測次数と残余確率はpaired_statistics_and_rare.jsonに保存し、未観測費用を0補完しない。

## 並列化・実測・保全

取得paired trajectory=2808、論理軸=5616、実compile=5616、compiled価格cache hit=0、RTE event=201454。取得wall=3860.751339457929秒。
2系signalをprocess並列、native contextは系/prefixごと共有、trajectory poolをE/Cで再利用し、空きCPU/RAMと代表RSSにより投入数を動的調整した。固定の低いworker上限は置かない。scheduler/signed schedule/compose memo/fingerprint/metrics cacheは追加7候補SOURCEを再利用。代表のcache hitでは保存QPYでcompiled gateを確認し、QPYがない場合だけ同条件のvalidation compileを取得する。cohort別・候補別performance JSONとresource観測に実際のinflight/cacheを保存。未最適化の実分子対照を大量再実行していないため、全体のspeedup倍率は未測定。

元結果・Phase A・7候補の保護対象96 fileを終了時にSHA照合し、変更0。原private raw/grant/ledger/marker/STOPを実行先として使用せず、専用rawに保存した。B3関連操作0、共有設定/他user process/他worktree変更0。ローカル検証であり、immutable CIや外部再現済みではない。

## 公開資料

- RESULTS.csv：新候補・保存対照の精度別会計。
- EXECUTION_SUMMARY.json：科学条件・取得数・技術検査・STOP。
- results/paired_costs.csv.gz：seed情報・保存先・資源情報を除いたpaired科学費用原値。
- results/signal_endpoints.json：新2候補のsignal・allowance・5精度endpoint。
- results/paired_statistics_and_rare.json、limited_set_rankings.csv、limited_cells.csv：有限標本統計と限定集合比較。

実行SOURCEの完全版、seed/raw/cache、詳細実行記録はローカル保全。公開版は科学値を保持し、内部metadataを除いた別コピーである。

## 固定集合の比較結果と解釈

|独立確認候補|Gの固定対照比|差の±2 combined-SE診断|
|---|---:|---|
|H6 R11|-0.17209%|INCONCLUSIVE|
|H4 R128|+11.19767%|HIGHER|
|H4 R16|+19.98337%|HIGHER|

H4のR32はE9のみ、R64はE33/C800で、各保存価格を再測定していない。R16/R128の独立Cは同じR64 C800を対照とする。R16のB/N増大と単価低下、R128のB/N低下と単価増加を、同じprefix3/q1/K2に固定して区別できる。H6も同prefix7/q1/K6でR11/R16の単価とNを分けて評価する。上記のG差は両要因を実測・計算で合わせた結果であり、単価の外挿ではない。

results/limited_set_rankings.csv は、各cohortの新固定候補、保存R対照、元ALL公式のPF/PR最良を含む限定集合の点順位。Cでは保存された独立確認があるPRだけを載せ、R32 E9をCへ代入していない。これは異なるcohortの標本をpoolする処理ではなく、それぞれの有限標本推定値を並べた比較である。確認群を見て候補や標本数を差し替えていない。

T2の対象10セルの限定E集合ではPR点費用が固定PF未満なのは8/10、限定C集合では8/10。元の33/40は保存のままで、40セルの元会計表を更新していない。

実測最大同時投入=128、pool容量最大=128、動的資源観測=79回。controller exit0とfinalizer exit0をwaitで回収した。

**上記は固定集合・経験的allowance・有限標本に条件付く。未測定候補、未観測rare費用、母平均の順位は未確定。mandatory STOP。**

公開版には内部保存先・seed情報を含めない。実行SOURCEとrawの完全版はローカル保全。

保存CSVはcommit用に改行だけCRLFからLFへ統一した。CSVの全field一致を検査し、科学値・原trial・旧監査記録は変更していない。
