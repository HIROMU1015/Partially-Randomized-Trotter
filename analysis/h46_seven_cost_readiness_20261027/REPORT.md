# H4/H6未取得7候補：Phase A実行準備

2026-10-27 JST。**Phase Aのみ完了。新しい分子計算・sampling・回路build/compileは0。Phase Bはユーザーの別途実行許可待ちでmandatory STOP。**

準備SOURCE `1255f6e60656494a45aebceeefe5224313e4355f` を保全し、別branch `codex/h46-seven-cost-speedup-20261027` で並列化・高速化を追加した。固定標本計画は不変。[高速化の内容・検証](PERFORMANCE.md)を参照する。

基準はスクリーニングcommit `2e518b86e7c052dafb20dece0de7ffa6c7c85f8f` と元研究commit `155529e20a8fac39e00786c40b57830b71203c4f`。7 ID・全科学条件・primary εの保存Nは両方の固定資料と一致した。実行SOURCEは本ディレクトリのPLAN.jsonとexecute.pyを含む固定commit（最終報告のSHA）。既存科学SOURCE・結果・33/40・grant/ledger/marker/STOP、B3と他研究worktreeは変更しない。

## 固定条件・利用可能な入力と環境

Linear H4/H6、隣接1.00 Å、STO-3G、tol-only DF threshold1e-8、保存rank10/19、固定fragment順、元state・reference・scalar。H6の保存weighted-Hermitian completionを含む。signed S2/m5_best、ABS_TIME_PER_OUTER・各piece最低1 step、finite-RTE・PCG64/NumPy choice、symmetric-directional control、actual adjoint、cosine/+Im sine、測定付きfull Hadamard wrapper、状態準備P=0を維持する。δ=T/qで、今回の7候補の窓は.2〜2。元の適格性・allowance・N_cos/N_sinを保存endpointから参照し、再適格化しない。

ユーザーの「こちらの環境に合わせてよい、ある程度の環境の違いは許容」に従う。既存の `.venv-server-opt2` は読み取り利用可能で、Python3.12.3、Qiskit1.3.0、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1、PySCF2.7.0、psutil7.2.2、Numba0.60.0をmetadataで確認した。新しい環境・packageのinstallや既存venvの変更は行っていない。compilerはrz/sx/x/cx、opt1、seed17、hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。compiler主要条件は実際に元と同じ版で準備できた。

元runnerの固定privateパスは存在しないが、固定commit内の `track_a_h246_reconstructed_confirmation/2026-10-22/published_inputs/` に利用可能な保存入力を確認した。

- H6：原snapshot・2 receiptが元研究のSHAと完全一致。completion前の別Hamiltonianへ置換しない。
- H4：2026-10-22に取得済みの保存再構築版を使用する。NPZ容器SHAは `ed9b5a13…` で、元 `12219b7d…` と異なる。既存の同一性証拠は7科学配列・Hamiltonian/state hashの一致を記録している。今回もNPY header/payloadのbyte hashを保存receiptに照合し、Hamiltonian/state/rankの一致を確認した。入力の再生成・数値配列化・SCF/DF/state solveは0。scalarのreceipt metadataは既存 `ascontiguousarray` のshape=[1]規約に合わせて照合し、保存scalar自体は変更していない。

現worktreeのライブラリから元providerの一部モジュールが失われているため、元研究provider commit `c31ef424175c02379eafebe8fc6c7548c1d9aabe` の `src/` を**新しい専用private領域**へGitから展開し、元の実装をそのまま使用する。既存の別研究providerを改変・更新していない。全provider fileをGit blobと照合するpreflightを用意した。元のreferenceは保存endpointの値を使い、reference再計算をしない。

## 再利用と最小差分

再利用する実体は `h46_long_time_native.compose/sample_requests/compile_cached/cost_trial/native_context/summarize_trials`、`h46_long_time_report.cost_for`、`h246_saved_cost_evaluation.paired`、元のsigned schedule・seed hash recipe・保存入力loader・occupation oracle・native correctness検査である。既存のC/N/Gとpaired covarianceの意味を変更しない。

元の `run_cost` は旧private config/旧取得集合・固定9本・旧出力先と結び付いているため直接起動しない。今回のexecute.pyは固定7候補・可変n・新cohort/出力のためのcontrollerで、科学kernelは元の関数を呼ぶ。旧SOURCE本体を編集していない。

各cohortのindex0がnative control・occupation・Hadamard検査を通ってから残りを投入する。元のnative検査1e-9、compiled-vector検査1e-7を維持し、phase alignmentや許容値の緩和はしない。これは全trajectoryや全入力状態に対するcompiled誤差ρの保証ではない。

## 事前標本計画

全7候補をEXPLORATIONと独立CONFIRMATIONの2cohortで取得する。**全探索の成功後、全7候補の確認を無条件に開始する。** 探索結果による候補選抜・n変更・top-upはない。各表の本数は両軸を共有するpaired trajectory数で、各trajectoryにcosine/sineの2 wrapperがある。

|ID|system/T/primary ε|formula|prefix/q/δ|R/K|探索n|確認n|確認の名目相対半幅|
|---|---|---|---|---|---:|---:|---:|
|d748ee85d02193720b34|H4/.8/.05|S2|3/1/.8|4/6|427|13,734|0.5%|
|205f25b8cb63949b4b18|H4/.8/.05|S2|3/1/.8|8/2|1,127|36,295|0.5%|
|53acfe65c49018ff2506|H6/1.6/.05|S2|5/4/.4|4/2|49|390|1.0%|
|6eb81b93d048deeeb63f|H4/.8/.0005|m5_best|2/1/.8|32/4|38|1,216|0.5%|
|54510520e99d85af4690|H4/1.6/.01|S2|3/8/.2|4/6|39|1,247|0.5%|
|cf094355c78d50ae45f7|H6/1.6/.05|S2|5/4/.4|8/2|143|1,151|1.0%|
|45f9fc0a53c1a37c7345|H4/2/.0005|m5_best|3/1/2|64/2|33|800|0.5%|
|計|||||**1,856**|**54,833**||

合計 **56,689 trajectory・113,378 logical wrapper軸**、RTE event総数539,044。新規transpile最大113,378軸で、同一数値回路の検証済みcache hitにより実呼出数は減り得る。追加compiler anchorは0。14本の代表検査（7×2cohort）は上記nに含まれ、別加算しない。Hamiltonian/state/reference/signal/shot/allowanceの新規評価は0。analytic Nは費用会計の重みであり、この数だけ量子測定を実施する計画ではない。

### nの根拠と分解能

保存近傍群の各trajectoryに対し `Y=N_cos*C_cos+N_sin*C_sin` とし、`v_Y=N_cos² v_cos+N_sin² v_sin+2 N_cos N_sin cov` をexact paired momentsから計算した。近傍群ごとに `CV²=v_Y/mean(Y)²` を求め、その最大値に**分散倍率4（標準偏差倍率2）**を置いた。近傍のmean/min/max/n/SE/variance/covarianceはanalogue_moments.csvに保持し、分布を正規と実証したとは扱わない。

近傍は同system/formula/prefix/q/Rを優先し、なければ同prefix/qの別Rを使う。m5 prefix2/q1には同q価格がないので、同prefixのq移送に独立加法stepを仮定したCVのsqrt(q比)補正を置く。これは標本数設計の参考仮定であり、未知候補の真の分散上界ではない。別Rからの移送にも保証はない。

探索は名目相対SE1%として `n_E=max(33,ceil(4 CV_max²/.01²))`。確認は正規近似の両側Bonferroni感度を11比較、α=.05に固定し、`z=2.83759691294`、半幅hに対し `n_C=max(33,ceil(4 CV_max²*(z/h)²))`。11比較は候補1/2のPF・PR比較4件、他5候補対PR5件、構成の平均単価対照2件。候補1/2はRとKの同時変更、候補3/6はRだけの変更であり、前者からR単独の効果を同定しない。別εへの会計は副次的な点比較で、独立実験数やこの11比較へ追加した正式推論とはしない。

最優先2候補は既存PF/PRの約1.5%差と、近傍単価の変動を踏まえてh=.5%を選んだ。R8のproxy上の約.4%差まで必ず分離できる設計ではない。H6の2候補は約4〜7%の費用改善とR対照が主目的なのでh=1%。他の3候補は約.8〜1.1%の接近が想定されるためh=.5%。大きい確認nはこの精度目標と観測分散から得たもので、旧9本の無条件流用や計算資源の恣意的上限から決めていない。結果が接近・分散過大なら**固定nで判断不能として終了**する。

### 旧基準価格の不確かさを残す

主たる点比較の参照値は元研究で取得済みの最良PF/PRの保存値に固定する。H4/.8/.05のPF G=75,021,842は決定論価格で、母平均推定のSEを持たない。一方、旧最良PR G=76,135,649.89のSEは2,101,236.41で、新候補側SEが0でもz倍半幅は約5.962e6（旧PR Gの約7.83%）残る。新候補だけを大量取得しても、この旧PR母平均の不確かさは解消しない。今回、旧基準候補の追加測定は7候補の範囲外で実施しない。

他のPR比較でも旧探索価格のz倍半幅floorは、H6/1.6/.05で約1.03%、H4/.8/.0005で約3.78%、H4/1.6/.01で約2.94%、H4/2/.0005で約1.76%。保存済みの独立確認価格がある場合は、その参照とSEも別診断に示すが、旧点参照や33/40を書き換えず、cohortをpoolしない。baseline_precision.csvに全9比較の数値を保存した。

したがって、保存された最良「点費用」を改善したかと、PR同士の母平均順位を判別できたかは別に報告する。新cohortの対固定点区間、両側SEを含む差の診断、保存独立確認との副次診断を分ける。旧最良の選択バイアス・rare tail・有限PRNG独立性未証明は残り、正規/Bonferroni計画を厳密信頼保証とは呼ばない。

K6等の未観測rare orderの費用を0にせず、未取得条件付き費用・母平均tail boundはnullのまま。多数のtrajectoryも全rare eventの観測を保証しない。保存近傍のmoment/min/maxしか利用できない箇所では、分布の形やtailを新たに推定したとは主張しない。

## Cohort・保全・実行範囲

cohortは `SEVEN_20261027_EXPLORATION` と `SEVEN_20261027_CONFIRMATION`。元の `trial_seed(candidate,cohort,index,outer,piece)` を使用する。今回はhash由来のseed識別子を列挙するだけでrandom samplingは0。既知識別子74,851個（元長時間研究4,022試行のcohort/index由来を含む）と新しい91,760 piece-seedとの衝突0、新cohort内/間重複0。seed列のdigestをPLAN.jsonに固定した。全歴史的seedを回収した保証や、PCG64の独立性証明ではない。旧grant・markerを起動に流用せず、新しい承認証明・grant基盤も作っていない。

private結果：`.server-preparation/h46-seven-cost-speedup-20261027/run01/`。公開結果：本ディレクトリの `results/`（Phase Aでは未作成）。旧cacheパスは不在なので持ち込まず、新しい数値回路fingerprint cacheを専用private領域で使用する。cache再利用はコンパイル結果の再利用であって、同じrandom drawの重複標本計上ではない。

完成PASSはcandidate/cohort/index・event総数・保存log_B・scalar/preparation binding・検査結果で照合し、再sampling/再compileしない。partial/STOPは保全して自動再実行しない。controller lockと元のatomic trial-directory作成で同時二重取得を防ぐ。科学/入力不一致、seed衝突、実際の資源不足・他利用への干渉は安全停止。固定n取得後もmandatory STOPで、追加候補やtop-upへ進まない。

並列度は実idle CPU/RAMと代表RSSで決め、30秒ごとに自分のworkerのCPU使用も分けて読み取り、処理中の投入数を増減する。pool容量の上限は実CPU affinity/cgroup quota・利用可能RAMから定め、固定CPU affinity・RSS/AS上限・少数worker上限は追加しない。共通contextは各系1回構築してfork共有し、bulk poolも探索・独立確認で再利用する。重い構造workを先に投入する。内部compiler/BLASは1 thread/processとして多重並列を避ける。RAMが代表footprint未満なら自己投入を止める。共有venv/cache・system設定・他ユーザーjobを変更しない。実分子の性能・所要時間は今回測定していない。

## 検証・次の承認

prepare.pyは保存値解析と入力byte照合だけ。execute.pyの `check` もmetadataのみ。元の人工/mock 10件と高速化11件の計21件で、paired共分散、7候補/N/n/seed不変、partial/STOP拒否、代表FAIL後のsampling抑止、exact会計に加え、投入数の増減、全trialの欠落・重複なし、context/pool再利用、回路cache入力の分離を確認した。科学runtime importと新しいcompose/sample/compileを禁止した検査結果をverification.jsonに保存する。元の保存endpoint・価格等11ファイルのhashも固定した。これは実分子native correctness・実compiler性能の新証拠ではない。元のcorrectness検査はPhase Bの計上済み代表trialで実施する。

```bash
python3 -B analysis/h46_seven_cost_readiness_20261027/verify_phase_a.py
```

Phase Bで許可が必要な具体的範囲は、**表の7候補、探索1,856＋独立確認54,833＝56,689 trajectory、両軸最大113,378 compile、元入力復元・既存builder/代表correctness・保存値会計・終了検査・公開まで**。この範囲の明示的な実行許可後は、途中の細かな再承認なしで進める。今回その許可は受けていない。

承認後の入口はPLAN.jsonに記載した既存Pythonによる `execute.py run`。ここには承認証明書の検証機構はなく、Codexが会話での別途許可を確認してから呼び出す。E/Cの費用・G/SE・既存最良との比較・R対照・新拡張集合の点countをresultsへ保存し、母平均・全域最適性と分離する。必要な結果REPORTの仕上げ・技術検査・commit/pushも許可範囲の作業とする。元33/40は歴史的結果として維持する。

**添付指示§1・§4 A-5に従い、ここでmandatory STOP。Phase Bは未実行。**
