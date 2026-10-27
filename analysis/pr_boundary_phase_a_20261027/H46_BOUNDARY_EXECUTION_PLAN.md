# H4/H6境界拡張・価格補完の実行準備（Phase A）

2026-10-27。**今回は保存証拠・入力bytes・SOURCE・整数scheduleの確認と実行設計まで。新signal、normalization、bias、shot、sampling、実回路build/compile、保存統計の数値再解析・再順位付けは0。Phase Bは未承認、mandatory STOP。**

正式共通指示 `research_project_instructions_unified_2026-10-25.md` を適用（読取原本SHA256 `e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720`）。元worktreeを切り替えず、境界調査commitから専用branch `codex/pr-boundary-phase-a-20261027` を作成した。変更は本ディレクトリの3資料と最小reference adapterのみ。旧SOURCE/raw/grant/ledger/marker/STOP、B3、別研究・整理作業は変更しない。共通索引の科学的結論も更新しない。

## 対象と準備状態

|対象|candidate ID|今回の準備結果|将来必要な科学処理|
|---|---|---|---|
|H4 / p3 q1 R16 K2|`f85eb7b56ba5131d1eb1`|保存signal・5精度の適格shot・入力・元価格pipelineの対応確認済み。互換保存価格は未発見。|価格取得のみ。新signal・shot計算は計画しない。標本数の数値設計と新候補専用の小さな実行adapterは未完であり、即起動可能とはしない。|
|H4 / p3 q1 R128 K2|`3a55c1dac3947ae23c76`|元allocation/nativeが受け付ける構成として静的確認。IDは元cid関数でmetadataから生成。科学的適格性は未確認。|保存H/state/T2 referenceを用いる新signal・B/bias/u/N。その後、適格endpointがある場合だけpaired価格取得。|
|H6 / p7 q1 R11 K6|`0e0e9fb6da33cb6ab2ab`|元11-piece allocationの下限として成立。weighted-Hermitian completion付き保存入力のbytes照合済み。科学的適格性は未確認。|同上。K8/R8や別prefixへ進まない。|

全対象T=2.0、m5_best（実装label `4th(m5_best)`）、`ABS_TIME_PER_OUTER`。H4はlinear隣接1 Å/STO-3G/tol-only1e-8保存rank10、H6は同条件rank19・元weighted-Hermitian completionを含む。prefixはDF generation順の先頭3/7（one-body共通）、δ=T/q=2。元のfinite paired RTE、scalar一回、reference非回転、symmetric_directional control、actual adjoint、測定込みcosine/sine full wrapper、P=0を維持する。finite-time complex signal精度であり、energy/QPE/RPE全体費用ではない。

主評価はε=.0005、他の元5精度集合の.05/.01/.005/.001を副評価とする**計画案**。同じ物理候補は精度ごとに再samplingしない。結果や40セルの勝敗は今回更新しない。

## 固定資料と価格重複の確認

|役割|固定commit|
|---|---|
|境界調査・本作業の親|`db1b5bdf96c05aa9ff23e9f843c2d3101496c8ed`|
|H4/H6長時間signal/endpoint/元価格/独立確認|`155529e20a8fac39e00786c40b57830b71203c4f`|
|既存未取得候補の調査|`2e518b86e7c052dafb20dece0de7ffa6c7c85f8f`|
|追加7候補の実行SOURCE|`cab73373593ba7c113c1af018b1dc3077c8e78fa`|
|追加7候補の結果|`bb97a93d65992fdb53e642dde1584dd7d5b811d7`|
|固定native provider|`c31ef424175c02379eafebe8fc6c7548c1d9aabe`|

元結果の根は `artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/`、追加7候補結果の根は `analysis/h46_seven_cost_readiness_20261027/results/`。

価格は元 `h46_long_time_report.costs()` の642候補結合を再利用し、`h46_long_time_signal.saved_assets()[2]` の旧候補ID変換・保存430価格も別途確認した。元のreused203/exploration481/extension_parallel22/fixed_delta1は重複するので合計してdistinct数としない。元confirmation39、reused_confirmation0、追加7候補のE7/C7も別に確認。候補IDと物理条件（T/formula/prefix/q/R/K）を照合し、R16/p3/q1/K2/T2の候補結合済み価格はなかった。旧T=.8や別Kの価格を転用しない。未索引のfingerprint cacheだけでは当該候補の両軸価格と認定できない。この確認は、参照した保存系列の範囲における価格不足である。

## R16の保存signal・shot（再評価しない）

出所：元結果 `signal_records.json.gz#f85eb7b56ba5131d1eb1`、`signal_endpoints.json.gz` と `candidate_precision_accounting.csv` のid/epsilon selector。5行すべて保存status=ELIGIBLE、u=1e-9、logB=0.22022101070674038、logB_margin=1.5432941318195487e-12。

|ε|N_cos|N_sin|
|---|---:|---:|
|.05|10921|10897|
|.01|275951|272940|
|.005|1118781|1094405|
|.001|31260205|27898108|
|.0005|145036571|114372825|

保存corrected=(real -0.37051487788553317, imag -0.9288162531492418)、signed bias=(4.717655218644534e-05, -8.542258071231323e-06)。raw/corrected closure=2.412989437205993e-15、maximum intermediate norm=1.000000000000309も保存済み。これらを再計算して改善・保証したとは扱わない。

R16では `evolve_batch`、`qualify`、shot ruleを呼ばない。将来のsampler内部で元law/preparationの分布係数を構築・照合する処理は価格取得に必要だが、新しいsignal/shot会計とは分ける。

## 入力・referenceの準備

本worktree内の保存入力根：`artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/published_inputs/`。`prepare.input_info()` とNPY header/payload検査を再利用して、配列を数値ロードせずbytes/receiptを照合した。

|入力|SHA256|
|---|---|
|H4/tol_only_snapshot.npz|`ed9b5a136e54a47047b6458ebb7ccb6c9b9ef699cdd534ae1f9f18da5310a520`|
|H4/tol_only_receipt.json|`623ac3c34fbb22ab364f8c637377b70cdd475abb73d2969f3a9acd86293c817b`|
|H6/h6_input_snapshot.npz|`99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`|
|H6/snapshot_receipt.json|`3dc5b1c695f1a042ee0c2b5e3762ae0032af90f03017667541ea470577f35e3f`|
|H6/df_receipt.json|`89cc00ab1c1b6ff65804781b5904b127e5d2e81acd15d66b0c87a86adb4c5f55`|

H4は2026-10-22に保存された再構築入力を読む。元NPZ容器SHA `12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96` とは異なるが、保存7科学配列のhashは元入力に一致することを既存reconstruction資料と今回bytes検査で確認した。今回再構築したのではない。H6は元snapshot/receiptとbyte一致、completionは変更しない。

|系|Hamiltonian hash|state hash|保存T2 reference（real, imag）|
|---|---|---|---|
|H4|`c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0`|`9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b`|(-0.3705620544377196, -0.9288077108911705)|
|H6|`223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9`|`664c63350c2510518b0ec561f804641447d39f75e68b960192c31d37540265f3`|(0.9822025116177282, 0.18782498814541881)|

referenceの出所は元 `signal_records.json.gz` のH4 R16/R32/R64とH6 R16/K6各record内の `signal.reference`。将来のsignalにも同じ保存複素値を使う。**既存 `h46_long_time_signal.context()` をそのまま呼ばない**：full-Hのスペクトルと新時間reference、T2の独立expm_multiplyを計算するため、本範囲に合わない。この違いのためだけに [saved_reference_context.py](saved_reference_context.py) を追加した。既存restore/occupation/spectral関数で必要なdeterministic blocksと対象tailのみを準備し、`ctx.refs[2.0]` を上記保存値で与える。full-H referenceの生成・再検算は0とする。保存H/stateの復元や既存operator表現の準備を、新規Hamiltonian/DF/state生成に置き換えない。

旧入力binding中のT=.8は共通入力由来の履歴値。本実行rowのT=2.0とは区別し、実signal/nativeのtimeにはrowの2.0を渡す。`h46_m5_plan.SYSTEMS/pilot/screen` の歴史的H4 rank12や別prefixを候補生成に使用せず、上記固定3行だけを使う。

## schedule・整数allocationの静的確認

元 `h46_m5_plan.weights` は11係数、中央が負、順序は `reverse(TAIL), 1-2 sum(TAIL), TAIL`。signed timeは `2*coefficient`、Rは絶対時間に沿う元largest-fraction allocation（各piece最低1）。丸めや同順位処理を変更・対称化しない。

|R_outer|piece 0..10へのstep allocation|
|---|---|
|11|1,1,1,1,1,1,1,1,1,1,1|
|16|1,2,1,2,1,2,1,2,1,2,1|
|32|3,3,3,3,2,5,2,3,2,3,3|
|64|5,6,5,6,5,10,5,6,5,6,5|
|128|10,11,10,12,10,22,10,12,10,11,10|

この表は許可された整数metadata検査であり、実分子signal/B計算ではない。R8は元関数が `INSUFFICIENT_OCCURRENCE_STEPS` で拒否した。H6 R11はAPI・allocation上成立するが、bias/normalization/shotやcompiled correctnessのPASSは未確認。

samplerは各pieceのsigned timeと割当stepで `canonical_tau`、元finite cutoff、K2/K6を使う。lawのsupportは元の偶数Taylor次数、finite meanは既存polynomial実装を使う。native `compose` は11 pieceを順番どおりS2へlowerし、各piece scalar=0、最後に元Hamiltonian constant＋tail identityのphaseをTで一度だけ付ける。R11でもscalarを11回付けない。新公式、K8、別Rの自動追加はない。

## 再利用する処理と必要な最小adapter

|処理|既存実装|適用・必要差分|
|---|---|---|
|ID/保存値読取|`scripts/h46_long_time_signal.py:cid,saved_assets`、`h46_long_time_report.py:costs,endpoints`|今回metadata照合に使用。旧ID変換を再実装しない。|
|signal|同signalの `evolve_batch`、`qualify`、元 `run_pf_common_run.qualify_record`|将来R128/R11各1行だけ。新referenceを作らないcontext adapterを追加・人工検査済み。5精度を同じsignalでqualifyし、u/logB margin/closure政策を維持する。|
|schedule|`src/trotterlib/h46_m5_plan.py:allocation,schedule` → 元 `h46_new2_core.allocation`|純関数を直接loadして今回確認。productionも同じ処理を使う。|
|保存入力とnative|`h46_m5_tol_native.restore`、`h46_long_time_native.native_context,sample_requests,compose,cost_trial`|H4 p3/H6 p7の2contextを再利用。旧privateパスを新専用output/保存入力へ局所bindingする。既存SOURCEの編集はしない。|
|compiled価格/cache|`h46_long_time_native.compile_cached`、追加7候補の `performance.py` と `execute.py` のworker/context/pool処理|両軸full wrapper、compiler/law/fingerprint一致時のみcache再利用。代表correctness用のcompiled circuitがないcacheをPASS根拠にしない。|
|統計/会計|`h246_saved_cost_evaluation.paired`、`h46_long_time_report.cost_for`、追加7候補のfinish/audit|別承認後に同一trajectory両軸の共分散・exact整数moment・G/SEを計算。今回は呼んでいない。|
|実行制御|追加7候補 `execute.py:process_jobs,ensure_contexts,resource_workers`|固定7候補・旧cohort・n・outputを含むmainをそのまま起動しない。今回3候補の薄いadapterが必要。新しい承認/監査基盤は作らない。|

現時点の成果物は実行設計とreference adapterであり、新しいproduction launcherは作成していない。標本数が未確定のまま旧runnerを流用起動することを防ぐ。固定3行の取得controllerは、別承認後のn設計を反映して科学call前に固定する。reference adapterはimport時に科学依存をloadせず、引数を固定2候補に検査してから既存数値kernelを遅延importする。今回実行したのは人工operationsを注入した経路のみ。

利用可能な既存専用venvは `/home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python`。今回はpackage metadataだけを読み、Python3.12.3、NumPy1.26.4、Qiskit1.3.0、SciPy1.14.1、OpenFermion1.6.1、PySCF2.7.0、psutil7.2.2を確認した。インストール・既存venv更新はしていない。compilerはrz/sx/x/cx、opt1、seed17、backend/target/couplingなし、hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

## 保存統計を踏まえた標本計画案

以下の数値は保存統計からの転記。新しいCV、必要n、差、比率、CIは算出していない。Gはε=.0005、u=1e-9、元empirical allowance・保存Nに条件付く。

|固定対照|cohort / n|G|G_SE|各軸平均C|各軸分散=cos/sin共分散|
|---|---|---:|---:|---:|---:|
|H4 R32 K2 `d9cb4e923530349e940a`|元E / 9|7498501185789.556|55880659384.189064|36180.11111111111|654269.1111111111|
|H4 R64 K2 `45f9fc0a53c1a37c7345`|7候補E / 33|7427614516704.546|44085966117.88573|39790.454545454544|1840660.7556818181|
|同H4 R64 K2|7候補C / 800|7411817334226.875|8660467438.08822|39705.8275|1721994.4257759699|
|H6 R16 K6 `efbb07c848b303a6abab`|元E / 9|34630535249194.445|58544949454.47112|179618.88888888888|829864.1111111111|
|同H6 R16 K6|元C / 33|34710218156959.09|31946367216.00101|180032.18181818182|906029.8409090909|

出所：元 `exploration_costs.json.gz`、`confirmation_costs.json.gz` のID selector、`candidate_precision_accounting.csv`、`independent_comparisons.csv`。追加7候補は `candidate_costs.csv` と `unit_cost_statistics.csv` のID/cohort/epsilon selector。

H4 R32とR64の保存Gは近く、R16/R128の価格は不明。H6もR11のB/Nと単価のtrade-offは不明。必要nを既存n9/n33/n800から直接指定する根拠はない。次の**未採択の設計案**とし、数値評価は別承認事項とする。

1. **探索E**：固定3候補を対象とする。ただしR128/R11はsignal適格性確認後のみ。新旧Nを固定した上で `V=N_cos² s_cc + N_sin² s_ss + 2 N_cos N_sin s_cs` を使う。H4はR32/R64の両統計、H6はR16のE/C統計をanaloguesとし、E/Cをpoolせず感度案として比較する。名目相対MCSE目標ρ_E=0.5%、分散倍率f=1と4を評価し、保守案f=4を第一候補とする。これはR変更による分散増大の保証ではない。
2. **独立確認C**：固定3候補のうち適格なものをすべて対象とし、Eの勝者だけに差し替えない。名目ρ_C=0.25%を第一案とする。E完了後の分散を用いる数値設計とn_C固定を終えてからC seedを確定・取得する。確認の結果でtop-upしない。各候補のn_E/n_Cは現時点では **PENDING_NUMERICAL_DESIGN**。
3. 必要nの既存設計関数は `ceil(f*V/(ρ*G_plan)^2)` 型。G_planと分散proxyの選び方を明記し、roundingや最小nを含めて固定する。過去runnerのmax(33,...)を無検討に移植しない。小標本/ゼロ観測分散のときは正のanalogue分散とrare限界を残し、n=1/SE=0を精度保証としない。この算術も今回は未実施。
4. **比較の不確かさ**：H4新C対R64 C800、H6新C対R16 C33を固定一次比較とする。独立cohortなら差の分散は両方のSE²を加える。保存対照のSEは増やせない精度floorであり、新候補nだけを増やしても解消しない。H4 R32 E9との比較は探索の補助に限定し、新しいR32確認を対象へ追加しない。
5. 名目判定案は差と±2 combined-SEの感度表示。差の符号が感度幅に収まる場合はINCONCLUSIVEとし、母平均勝敗や厳密CIとはしない。H4 R16/R128対同じR64の2比較や精度間の相関を明記する。多重比較の厳密保証を新たに主張しない。
6. **rare RTE**：保存観測はH4 R32 EでTaylor0=287/2=1、R64 Eで0=2112/2=0、R64 Cで0=51180/2=20。H6 R16 E/Cはそれぞれ0=144/528、高次2/4/6=0。R11/R128の希少確率・分散は未計算。承認後の元lawに基づく未観測確率診断は費用取得/統計の範囲へ明示して含める。条件付きrare強制samplingや0補完は行わない。SE目標の達成は未観測tailの上界ではない。

n_Eの最終化には保存momentからの新しい統計設計算術、n_Cには新E momentの評価が必要。R128/R11ではさらに新Nが必要で、今回整数nを提示することはできない。**この未確定部分を別承認後に解く設計でPhase Aを完了し、価格取得まで完了したとは報告しない。**

## 将来の最小実行範囲（未承認）

1. 人工検査済みreference adapterを使用し、固定3行controllerと入力binding・標本数の接続検査を完了して新SOURCEへ固定する。元SOURCEを変更しない。実H/state/reference生成は0。
2. R128/R11の新finite-mean signalを各1候補、元5精度を各1回qualifyする。新referenceは0。既存u/logB margin、raw/B closure、signed bias/headroomを保存。INELIGIBLE/UNDETERMINEDのendpointは価格会計に進めず、全endpoint不適格なら当該候補の価格取得0。R16はこのstageを通らない。
3. 別承認に含まれる数値標本設計でn_Eを固定し、適格候補だけE取得。Eに基づく固定n_C設計までを許可範囲に含め、確認開始前に候補・比較・n_C/seedを固定する。ユーザーがsignalだけを承認した場合、価格取得へは進まない。
4. 各trajectoryは11piece request、RTE event数はH4 R16=16/H4 R128=128/H6 R11=11、native evolution1、full wrapper2軸。compileは最大 `2×sum(n_E+n_C)` 軸で、完全一致cache hit時は実呼出を減らし別記する。代表correctnessは各候補・各cohortの最初のtrajectoryに含め、別のpilotを自動追加しない。量子測定shot実行0。
5. 原lawの非条件付きPCG64 samplingを維持する。新cohort名 `BOUNDARY_20261027_EXPLORATION/CONFIRMATION` を元seed hash APIへ渡す。元2系の既知seed資料、長時間study、追加7候補の**実際に取得したseed**を除外照合し、cohort間・候補間重複を検査する。未回収の歴史seedがあれば完全除外を主張しない。今回はseed/RNGを新生成していない。
6. 元native gateを維持：low branch不変、high leakage、独立occupation oracle、Hadamard cos/sinは1e-9未満、compiled vectorは1e-7未満、phase alignmentなし。scalar once、actual adjoint、fingerprint、compiler設定、両軸event不変を検査。今回の静的PASSを新Rの実numerical PASSへ格上げしない。
7. 別承認後の会計ではE/Cを分離し、Nとpaired価格から元 `cost_for` でG/SEを保存する。元PF対照も固定参照とし、既存40セル表/33勝を上書きせず拡張結果として出す。H4のR16/32/64/128 trade-off、H6のR11/16下限trade-offを主に調べる。価格比例proxyだけで改善を判定しない。

### 並列化とSTOP

新専用output/cache/tempだけを用いる。H4の2候補は同じp3 preparation/context、H6はp7 contextを共有し、独立候補・trajectoryをprocess poolへ投入する。元のwarm context、fingerprint再利用、on-disk metrics cache、pool再利用を採用する。compile内部は1 processにして外側trajectory並列との過剰競合を避ける。CPU/RAM/worker数に固定の低い上限を置かず、実idle CPU・available RAM・代表RSS・pressure/I/Oに従って投入数を決め直す。共有設定・他job・他userのprocessを変更しない。旧研究の固定資源契約を上書きしない。

input/参照selectorの不一致、schedule/law逸脱、correctness失敗、両軸欠落、資源圧迫、未承認callへの到達時に当該新jobをSTOPし、証拠を保全する。自動retry、n_Cの結果依存増量、別R/K/prefixへの代替、旧STOP解除はしない。signal不適格は科学結果として保存し、別構成に置き換えない。承認された範囲が終わればmandatory STOP。新しいapproval certificate/grant発行システムは要求しない。

## 今回実施した技術確認と残る判断

- PASS：元cidの純関数でR16 ID一致、新R128/R11 IDが既存対象と重複しないこと、R11/16/32/64/128の11piece・整数総数・各1step以上、R8拒否。
- PASS：既存input_info/NPY payload照合、H4の保存再構築7配列一致、H6元snapshot/receipt一致。科学配列の数値ロード・新H/state生成なし。
- PASS：既存価格・旧ID変換・独立確認・新7候補両cohortの読取照合。R16は未取得のまま。
- PASS：再利用予定source7ファイルのAST parseとhash取得、runtime package metadataの読取。実sampler・builder・compiler・statistical/accounting driverの起動なし。
- PASS：新reference adapterの人工22項目。2系について保存reference値の引渡し、restore1回、必要なprefix＋tailだけのspectrum、対象外R/K/T/prefix/IDと誤reference時刻・誤rankのrestore前拒否、R16の新signal拒否、NumPy/SciPy/Qiskit等をimportしないことを確認。演算対象は記号的mockであり、実H配列を数値評価していない。adapter SHA256=`371d9efe6ca74d7192aa35a43f129b5c87514e0354936e62039769a794c19ea2`。
- 未実施：固定3行launcherと標本数の接続、数値n設計、実R128/R11 signalと全新価格。新しい大規模なframework/test群は作らない。

H4ではR16が不足価格の補完、R128が上限延長の検証となる。H6ではR11が現lawを維持した実装下限で、単価削減とB/N増大の比較を直接調べられる。いずれも改善・PF/PR逆転は未確定。H8については隣の監査文書を参照し、新規実測ではなく既存record回収を先にする。

**必要な次の明示許可は、固定2候補のsignal/shot、保存統計によるn設計、適格時だけの固定3候補E/C価格取得と会計のうち、実際に進める範囲である。今回の添付指示はこれらの実行許可を含まない。Phase A完了、mandatory STOP。**
