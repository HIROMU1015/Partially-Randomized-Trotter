# H4/H6未取得7候補：実測費用取得結果

2026-10-27 JST。固定7候補の探索1,856本と独立確認54,833本、計56,689 paired trajectoryを取得した。両cohortを分けて会計し、所定本数の取得・保存値監査後に**mandatory STOP**。追加候補・top-upは0。

最優先のH4・T=0.8・epsilon=0.05では、2候補の確認Gは約7.583e7と8.021e7で、固定PFの7.502e7を下回らず、PF→PRの点順位逆転はなかった。差の事前固定近似区間も両候補で正だった。今回の探索価格追加でPR点最良は7セルで改善したが、40セルのPF/PRの点方向は全て不変で、PR低費用数は33/40のままである。

H6・T=1.6・epsilon=0.05では、R4/K2とR8/K2の旧PRに対する改善方向を探索・独立確認の両方で観測した。確認Gはそれぞれ旧PR比3.40%・6.74%低く、新旧双方のSEを含む正規近似診断でも差は負だった。H4の小幅改善は旧PR側の不確かさを含むと判断不能で、特にm5 prefix2候補は探索と確認で旧PRに対する点方向が異なる。新たに取得した7候補以外の費用最小性は確定していない。

対象はlinear H4/H6、隣接1.00 Å、STO-3G、tol-only DF threshold 1e-8、保存rank H4=10/H6=19。H6保存weighted-Hermitian completion、元Hamiltonian/state/reference/scalar/fragment順を維持した。保存prefix、q、R_outer、Kのみを使い、今回のdelta=T/q窓は0.2〜2.0。signed S2/m5_best、ABS_TIME_PER_OUTER、finite-RTE/PCG64/NumPy、symmetric-directional control/actual adjoint、cosine/+Im sineの測定付きfull Hadamard wrapper、状態準備P=0を使用した。Qiskit1.3.0/NumPy1.26.4、basis rz/sx/x/cx、optimization_level=1、seed_transpiler=17。元のsignal適格性・allowance・N_cos/N_sinを再利用し、Hamiltonian/state/reference/signal/shotの新規計算は0。

## 固定7候補の結果

GはRZ単価×保存shot数、SEはcosine/sineのpaired covarianceを含む標本平均のSE。同じ物理候補を複数epsilonに会計しても新しい候補や独立試行とは数えない。

今回価格を取得した物理候補は7つ。保存signalが適格な候補×epsilonは19 endpointで、探索・確認の2cohortを区別した会計表は38行である。

|Candidate ID|系/T/epsilon|公式/prefix/q/R/K|探索n / 確認n|探索G|確認G ± SE|確認点Gの旧PR比|
|---|---|---|---:|---:|---:|---:|
|d748ee85d02193720b34|H4/0.8/0.05|2nd/3/1/4/6|427 / 13734|75914565.7|75827614.2 ± 51466|-0.4046%|
|205f25b8cb63949b4b18|H4/0.8/0.05|2nd/3/1/8/2|1127 / 36295|80182933.8|80209377.8 ± 49350|+5.3506%|
|53acfe65c49018ff2506|H6/1.6/0.05|2nd/5/4/4/2|49 / 390|1.21985147e+09|1.22213947e+09 ± 1.7092e+06|-3.3982%|
|6eb81b93d048deeeb63f|H4/0.8/0.0005|4th(m5_best)/2/1/32/4|38 / 1216|5.19481975e+12|5.13819047e+12 ± 3.9023e+09|-0.2196%|
|54510520e99d85af4690|H4/1.6/0.01|2nd/3/8/4/6|39 / 1247|1.19600514e+10|1.19128556e+10 ± 1.0094e+07|-2.0418%|
|cf094355c78d50ae45f7|H6/1.6/0.05|2nd/5/4/8/2|143 / 1151|1.18048591e+09|1.17989428e+09 ± 1.497e+06|-6.7374%|
|45f9fc0a53c1a37c7345|H4/2.0/0.0005|4th(m5_best)/3/1/64/2|33 / 800|7.42761452e+12|7.41181733e+12 ± 8.6605e+09|-0.6540%|

## 最優先：H4・T=0.8・epsilon=0.05のPF/PR順位

固定PF参照G=75,021,842（B0+I/prefix5/S2/q1）と、旧PR参照G=76,135,649.8889を使用する。事前固定した正規/Bonferroni係数 z=2.83759691294（11比較）による差の診断を示す。以下の近似区間は母平均・rare tailの厳密保証ではない。

|Candidate ID|確認G − 固定PF|差の固定点近似区間|確認点がPF未満|
|---|---:|---|---|
|d748ee85d02193720b34|+805772.19|[+659733.18, +951811.21]|いいえ|
|205f25b8cb63949b4b18|+5187535.8|[+5047499.4, +5327572.3]|いいえ|

## 旧PRとの比較と残る不確かさ

旧PRの探索参照値自体にSEと選択の不確かさがある。候補側だけの対固定点区間と、新旧双方のSEを含む差の正規近似診断を分離する。旧PRの追加取得は今回の範囲外で行っていない。

|Candidate ID|確認G − 旧PR|双方SEを含む差の近似区間|確認点が旧PR未満|
|---|---:|---|---|
|d748ee85d02193720b34|-308035.69|[-6272285.9, +5656214.5]|はい|
|205f25b8cb63949b4b18|+4073727.9|[-1890378.3, +10037834]|いいえ|
|53acfe65c49018ff2506|-42992051|[-56948286, -29035817]|はい|
|6eb81b93d048deeeb63f|-1.1308268e+10|[-2.0635402e+11, +1.8373748e+11]|はい|
|54510520e99d85af4690|-2.483124e+08|[-6.0721242e+08, +1.1058761e+08]|はい|
|cf094355c78d50ae45f7|-85237238|[-98995739, -71478738]|はい|
|45f9fc0a53c1a37c7345|-4.8793512e+10|[-1.8233127e+11, +8.4744242e+10]|はい|

保存済み独立確認の旧PR価格が利用できる場合、そのG/SEと比較はprimary_comparisons.csvの`saved_independent_reference_*`欄に別途保持した。参照値の差し替えやcohortのpoolingは行っていない。

## 40セルの拡張集合の点順位

旧取得集合のPR低費用セル数 **33/40** を変更せず保存する。今回の探索価格を追加した集合では **33/40**。独立確認は点順位へpool/代入せず、各候補の別cohortの検証として扱う。全40セルはextended_cells.csvに保存した。

|系/T/epsilon|探索追加後のPR最小値が変化|PF/PRの点方向が変化|採用Candidate ID|
|---|---|---|---|
|H4/0.8/0.05|はい|いいえ|d748ee85d02193720b34|
|H4/1.6/0.01|はい|いいえ|54510520e99d85af4690|
|H4/1.6/0.005|はい|いいえ|54510520e99d85af4690|
|H4/2.0/0.005|はい|いいえ|45f9fc0a53c1a37c7345|
|H4/2.0/0.001|はい|いいえ|45f9fc0a53c1a37c7345|
|H4/2.0/0.0005|はい|いいえ|45f9fc0a53c1a37c7345|
|H6/1.6/0.05|はい|いいえ|cf094355c78d50ae45f7|

## R_outer対照

単価はcosine/sineの平均RZで、総Gとは異なる。H4の対照はRとKの同時変更なのでR単独の効果を同定しない。H6の対照は同prefix/q/KでRのみが異なるが、Nも変化するため単価差とG差を分ける。

|cohort|左 → 右|差（右単価−左単価）|近似区間|
|---|---|---:|---|
|探索|d748ee85d02193720b34 → 205f25b8cb63949b4b18|+422.8212|[+364.7808, +480.8617]|
|探索|53acfe65c49018ff2506 → cf094355c78d50ae45f7|+5675.499|[+4764.531, +6586.467]|
|確認|d748ee85d02193720b34 → 205f25b8cb63949b4b18|+428.4062|[+418.3038, +438.5087]|
|確認|53acfe65c49018ff2506 → cf094355c78d50ae45f7|+5545.573|[+5244.769, +5846.378]|

## 希少event・保証の限界

|Candidate ID|cohort|n|観測order2|観測order4|観測order6|
|---|---|---:|---:|---:|---:|
|d748ee85d02193720b34|探索|427|9|0|0|
|205f25b8cb63949b4b18|探索|1127|10|0|0|
|53acfe65c49018ff2506|探索|49|3|0|0|
|6eb81b93d048deeeb63f|探索|38|3|0|0|
|54510520e99d85af4690|探索|39|1|0|0|
|cf094355c78d50ae45f7|探索|143|8|0|0|
|45f9fc0a53c1a37c7345|探索|33|0|0|0|
|d748ee85d02193720b34|確認|13734|348|1|0|
|205f25b8cb63949b4b18|確認|36295|494|0|0|
|53acfe65c49018ff2506|確認|390|25|0|0|
|6eb81b93d048deeeb63f|確認|1216|60|0|0|
|54510520e99d85af4690|確認|1247|18|0|0|
|cf094355c78d50ae45f7|確認|1151|35|0|0|
|45f9fc0a53c1a37c7345|確認|800|20|0|0|

K6のd748ee85d02193720b34では確認cohortでorder4を1 event観測し、その費用を含めて集計した。order6は全て未観測。Kを超えるorderは構成に含まれないため、表のその0を未観測リスクとは扱わない。

未観測orderの条件付き費用や母平均tail boundはnullとし、0を補入しない。事前nは既存近傍の最大CV²に分散倍率4を置いた経験的計画で、未知候補の分散上界ではない。有限PCG64列の独立性、無条件のPR母平均優位、全探索域の最適性、全trajectory/全状態でのcompiled誤差rhoを保証しない。本結果は固定T・保存endpointによる条件付きGであり、全RPE/energy評価の最終総費用ではない。

## SOURCE・入力・取得と監査

実行SOURCE `cab73373593ba7c113c1af018b1dc3077c8e78fa`、PLAN SHA256 `b4be2aa419e73c3eab5f726b27e3ed7bcc61d42a98cda50a6987fab81f9c04a0`。元研究SOURCE 155529e20a8fac39e00786c40b57830b71203c4f、選定SOURCE 2e518b86e7c052dafb20dece0de7ffa6c7c85f8f、provider c31ef424175c02379eafebe8fc6c7548c1d9aabeを参照する。H6は原snapshot/receiptのbyte一致、H4は保存済み2026-10-22再構築入力の7科学配列一致でありNPZ容器SHAは異なる。今回は入力の再生成をしていない。詳細はPLAN.jsonのsystemsと元準備commitのREPORTを参照する。

全56,689 trial・113,378軸・539,044 RTE eventを照合した。実新規compile 113,234軸、compile cache再利用 144軸。native検査70件、compiled-vector検査28件。最大native誤差1.4777e-13、最大compiled誤差1.4614e-10。14本の代表trialは固定nに含まれる。元のnative1e-9/compiled1e-7閾値とphase-alignment禁止を維持した。元cache仕様どおり、representativeのcompiled価格がcache hitした場合はそのtrialでのcompiled-vector検査を再実行しない。検査実施件数は上記の実記録で示し、全113,378軸に対するvector検査とは扱わない。

保存seed 91,760識別子の列digestは `1143a94bfba4f8ea4102a3752020c4b8cfa2c4094326d9ed3210a42c17b3f9b5` で事前計画と一致。known除外資料との照合も再実行し衝突0。全歴史的seedの回収保証はない。全trialのID/index/cohort、event数とorder、scalar/preparation/log_B、整数price、paired momentsと共分散、保存N/exact Gを監査した。21件の事前人工/mock検査とは区別し、今回の終了監査はローカル保存記録の照合であって外部再現ではない。

元controllerのtool sessionは終了後に参照できず、OSの終了codeは取得不可（null）。COMPLETIONの全worker join記録、STOP不在、該当実行process不在を直接照合した。終了code=0を推測で補っていない。完成試行の再取得やcontrollerの再起動はしない。

最大128 workerのpoolと資源に応じた投入数調整を使った。cohort別のcache件数・同時投入数は*_performance.json、候補別の実trial時間・RSSはcohort_diagnostics.csvに保存した。比較対象となる未最適化の実分子runは実施しておらず、pipeline全体の速度向上率は未測定。Phase Aのschedule単体4.06倍を全体の速度比として扱わない。

provenanceの開始時刻01:58:56 JSTからCOMPLETIONの保存mtime 03:38:52 JSTまで約99分56秒（mtimeに基づくローカル記録）。schedule cache hitは両cohort合計112,475件、compose再利用hitは0件で、このrunでcompose cacheによる高速化が得られたとはいえない。compile cacheのJSONは113,234件で、記録上の新規compile軸数と一致した。RSSは再利用workerのprocess生涯最大値なので、候補単独の必要RAMと同一視しない。

## 成果物・終了

- results/candidate_costs.csv：適格epsilon別のN、C、G、paired SE。
- results/unit_cost_statistics.csv：両軸分散・共分散、RZ/CX/depth等の平均。
- results/primary_comparisons.csv、decision_diagnostics.csv、R_unit_price_contrasts.csv：固定参照との差と診断。
- results/cohort_diagnostics.csv、trajectory_costs.csv.gz：全56,689 trialのpaired費用・order数と取得範囲。
- results/extended_cells.csv、extended_point_summary.json：旧33/40と拡張探索集合を分離した40セル。
- results/execution_provenance.json、COMPLETION.json、END_AUDIT.json：実行SOURCE、固定範囲の完了、結果hash。

終了監査と機械的な表の出力は `python3 -B analysis/h46_seven_cost_readiness_20261027/finish.py` で実行した。科学runtimeのimportとsampling/build/compile関数を禁止した保存値解析である。本文冒頭の結果解釈・endpoint数・実行時間の説明は出力表の確認後に追記した。

元研究・元SOURCE/raw/grant/ledger/marker/STOPは変更していない。B3関連の操作・別研究worktreeの切替・共有環境の変更はしていない。private raw/event/cacheは専用run01に保全し、公開表はその派生コピーである。

**固定取得と終了監査後のmandatory STOP。追加候補、標本追加、別研究、詳細科学レビューへ自動進行しない。**
