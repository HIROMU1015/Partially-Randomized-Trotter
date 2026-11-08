# 通常CDF＋Z再配置＋PR：環境修復後run02の限定科学結果

2026-11-08 JST。**登録した科学範囲は完了したが、主比較の正式判定はCASE C・`UNRESOLVED_PRECISION_TARGET`。** Selection32で固定した通常CDFのr64、独立final64の条件付き総RZは **231,951,425,282.5**、保存ENTRY E2/r16は **75,621,395,471**。点値ではENTRYが67.3977%低く、CDF0−ENTRYの名目95%差区間は正だった。一方、区間半幅/ENTRYは6.09945%で登録5%目標を満たさず、ENTRY低費用の正式採択へ格上げしない。追加標本・条件変更は0。

[結果原本](../../artifacts/research_c_cdf0_relocated/2026-11-08/science/run02/CDF0.json)／[公開要約・全scope監査](../../artifacts/research_c_cdf0_relocated/2026-11-08/publication_run02/summary.json)／[final64原本](../../artifacts/research_c_cdf0_relocated/2026-11-08/science/run02/final_summary.json)／[選択固定](../../artifacts/research_c_cdf0_relocated/2026-11-08/science/run02/frozen_selection.json)／[STOP](../../artifacts/research_c_cdf0_relocated/2026-11-08/science/run02/STOP.json)。

## 1. 承認・修復・固定SOURCE

前回run01はAerのCUDA共有ライブラリのシンボル解決で停止した。ユーザーの「修正して再実行してよい」という明示許可を受け、専用環境を修復してfresh run02を一回実行した。旧run01をresumeしたものではない。後の「続きを行って」は完了後の監査・公開を継続する指示として扱い、新科学計算を開始していない。

既存venvには`qiskit-aer`と`qiskit-aer-gpu`の両distribution metadataがあり、同じmodule領域を使用していた。既存領域を変更せず、**CPU版qiskit-aer==0.15.1**を専用`/tmp/research-c-cdf0-aer-cpu-20261108`へ配置してPYTHONPATH先頭で利用した。固定compiler・CPU full-unitary方式・数値allowanceは維持する。修復後controllerのlddにcusparse/nvJitLink依存がなく、Aerの利用可能deviceはCPUのみだった。

実行前にimport/linkと人工9 qubit回路の全512行列・位相・実inverseを検査。独立Operatorとの最大成分差`5.551115123125783e-17`、実inverse復帰差`2.9893669801409083e-16`でPASS。実H4を使う環境probeではない。[修復記録](../../artifacts/research_c_cdf0_relocated/2026-11-08/execution_run02/environment_repair.json)。共有venv/package、CUDA全体設定、他processは変更しない。

| 来歴 | 固定内容 |
|---|---|
| 科学code SOURCE commit | `bde646c2b0d275c66d9890eee835d4692f55cafb` |
| 実行checkout基点 | 前回失敗原本を含む`4ef705940b83d7c60de5966611f2d265435336f8` |
| SOURCE manifest | [112-file SOURCE](../../artifacts/research_c_cdf0_relocated/2026-11-08/science/run02/SOURCE.json)、SHA256 `640052dfeed3b886bb95e11abb410e3275cdda1ffe82661e544037c120f94c32` |
| run02契約 | [別契約](research_c_cdf0_relocated_contract_rerun02_20261108.json)、SHA256 `045b15ea10ecfdc7ee86cd781e9601b923092951e10df8bfa2431c790be7e02c` |
| 旧proposalとの差 | **fresh_outputだけ**をscience/run01→science/run02へ変更。旧proposal、科学SOURCE、入力、乱数namespaceは不変更 |
| 候補ID | `CDF0_STAGE_A_X0_V2_GAMMA0_FINAL_R8` |
| J/U export SHA256 | `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02` |
| 元h/g/c input identity | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| 固定前段fit | Stage A V2 γ=0最終CDF8、fit SOURCE `e1085c95f5c01b68566e17d543ace5832cb94061` |

準備時の`scientific_execution_permission=false`は歴史的固定フィールドとして保持し、会話の明示許可で実行した。承認原文の保存・grant・証明書は要求も作成もしていない。今回新しいsolver、科学runner、Hamiltonian生成器、PF/RTE/shot式、compilerは作っていない。

## 2. 科学条件とDF精度

**以下の全数値は、Linear H4・隣接1.00 Å・STO-3G、number CDF8、Nα=Nβ=2/sector36/full256/control512、元順F0〜F7・LD2に限定する。** 保存κ/scalarを受け取り、F2〜F7の線形Zだけを既存headへ移す。`Δκ=Σ_tail U diag(Jの行和) Uᵀ`、`Δc=−Σ_tail 行和`、frame I/ZZを保持する。

PFはordered partial S2/s1、δ=h=.1のみ、K0、r={8,16,32,64,128}、m0..12。各候補のown-DF ground oracleを使用し、状態準備は費用から除外する。CA=.00159 Ha、DF reserve=CA/8=.00019875 Ha、βtotal=π/6、βstat=π/12、candidate failure=.01、axis=.01/26。P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx、MPFR160 declared phase、位相fitなし、fenced serial。

保存CDF0のsector norm=`7.018855038843621e-9 Ha`、engineering guard=`3.278781198621627e-9 Ha`、保存和=`1.0297636237465247e-8 Ha`でDF予算PASSを再利用した。Tensor Frobenius=`5.607958698344648e-9`とは区別する。新fit・DF norm評価は0。元fitは相対目的減少停止であり、十分な一次/大域収束を認定していない。

ENTRY E2は同じ元物理input、CDF8/LD2、C0/ENTRY strength .03/64更新という**別の保存候補**。候補ごとに近似Hamiltonian・own-groundが異なり得る。今回は同じ元物理task・共通誤差予算に対する保存候補比較で、penalty単独の因果ablationや通常CDF全体の最適能力比較ではない。新しいDF最適化を比較のために追加していない。

## 3. 再配置・独立CAR・精度資格

固定SOURCEの再配置、元9因子と保存B1 referenceの独立照合、scalar込み全H保存、ground/alias、sector/full CAR、tail signed law＋identity、normalizationを通過した。J/Uを新候補へ置換せず、κをJ/Uから再導出せず、核scalarを二重計上していない。

再配置後tail mass **λR=`1.1246233153558292 Ha`**、抽出identity=`.4793928496324132`、atom数168。[保存law原本](../../artifacts/research_c_cdf0_relocated/2026-11-08/science/run02/r64/reference/law.json)のCAR tail対signed law＋identity最大成分差は`6.661338147750939e-16`、許容差1e-10。旧中立C0や初期強Dのλを代用した値ではない。

元因子9回・Hamiltonian gate1回・ground gate1回の成功は保存`call_counts`と固定SOURCEで確認できる。一部gateの戻り値は既存runnerが個別JSONへ保存していないため、未保存のfull-H残差・ground残差・scalar shiftの数値を今回新たに計算して埋めていない。成功callbackと保存数値原本の区別を維持する。

## 4. 五つのrとselection

計65 reference roundを保存した。[65行CSV](../../artifacts/research_c_cdf0_relocated/2026-11-08/publication_run02/reference_rounds65.csv)は保存JSONの転記。下表は同じδ=.1/m12の値。PF kernelと13 exact evolutionsは同一候補内で一回だけ生成し、他rで再利用した。

| r | m12 PF誤差 | m12補正RTE誤差 | m12 attenuation | 全13round政策のnative適格性 | selection標本 | selection総RZ |
|---:|---:|---:|---:|---|---:|---:|
| 8 | .112259680552 | .753652202401 | .039261504503 | 不適格 | 0 | null |
| 16 | .112259680552 | .324260480907 | .198121393118 | 不適格 | 0 | null |
| 32 | .112259680552 | .150765655858 | .445101618806 | 不適格 | 0 | null |
| 64 | .112259680552 | .072737510603 | .667158114732 | 適格 | 32 | 226,213,686,059 |
| 128 | .112259680552 | .035730429700 | .816797283307 | 適格 | 32 | 263,520,979,229 |

r8/16/32は登録PF/RTE十分条件がphase roomを超えた。価格0や手法の一般的敗北として扱わない。r64/128はideal条件付き精度を通過し、各selection32を取得した。両側に以前のENTRYと同じr/K機会を与え、登録範囲を追加拡張しなかった。

selectionだけで**r64を固定**し、選択後に別cohortのfinal64を取得。Finalに基づく再選択、selection/finalのpool、旧run01の結果・残枠の転用、r128のfinal取得は0。[r64 selection原本](../../artifacts/research_c_cdf0_relocated/2026-11-08/science/run02/r64/selection/result.json)／[r128 selection原本](../../artifacts/research_c_cdf0_relocated/2026-11-08/science/run02/r128/selection/result.json)。

## 5. r64独立final64・RPE・native費用

Ideal laneでplanned shots17,250、W=20,535,472、kernel平均RZ=11,295.15625、shell RZ=60,375。`総RZ=W×kernel平均＋shell`を維持し、rや校正標本数をWへ重複乗算しない。m12のshots/axis=1,729、radius lower=.543735737450、raw signal error=.123422377282、deterministic phase=.186069004151 rad。

| 指標 | final64のkernel平均 | shots加重ideal条件付き総費用 |
|---|---:|---:|
| RZ count | 11,295.15625 | 231,951,425,282.5 |
| RZ depth | 3,505.703125 | 71,991,328,738.75 |
| CX count | 4,618.53125 | 94,843,719,165.5 |
| CX depth | 2,897.15625 | 59,494,471,051.5 |
| Total depth | 7,719.15625 | 158,516,629,160.5 |
| Circuit size | 22,430.84375 | 460,628,075,889.5 |

Ideal/assumed両laneは`CONDITIONAL_CA_PASS`、条件付きenergy error bound pilot=.0012783276208742716 Ha、statistical failure upper=.01。assumed laneはshots17,322、総RZ233,038,606,913.875。これは宣言した誤差仮定を適用したシナリオで、未知の全law合成誤差を測定0としたものではない。

取得証拠だけのevidence laneは`UNQUALIFIED_ROUNDS`、shots/総費用=nullを維持。actual CA、whole-law synthesis、implemented-law誤差、ideal-law価格移送、厳密rounding enclosure、hardware/FT・状態準備込みの資源保証は未資格。有限価格標本はclassical calibrationで、実量子shot抽選は0。

## 6. 保存ENTRYとの登録主比較と精度未達

保存ENTRY E2/r16の独立final64をそのまま再利用した。新ENTRY計算・新ENTRY標本・旧区間の付替えは0。歴史的helperの`PR`キーは今回CDF0を指し、外側`comparison_label`に明記した。新たな同時取得prospective two-arm trialとは呼ばない。

| 登録比較 | 保存値・今回取得値 |
|---|---:|
| CDF0 r64 ideal総RZ | 231,951,425,282.5 |
| 保存ENTRY E2/r16 ideal総RZ | 75,621,395,471 |
| CDF0−ENTRY点差 | +156,330,029,811.5 |
| CDF0/ENTRY点比 | 3.067272480729 |
| ENTRYのCDF0に対する点削減率 | 67.3977448602% |
| 差の名目95% Welch区間 | [151,717,539,065.4622, 160,942,520,557.5378] |
| 区間半幅 | 4,612,490,746.037811 |
| 半幅/ENTRY | **6.0994520364% > 登録5%** |
| 正式判定 | **UNRESOLVED_PRECISION_TARGET** |

各側64標本、固定近似IID Welch（df=76.4010491285）。CDF0/ENTRYのmoment skewnessは−.238285742/−.351786239、delete-one shift/halfwidthは.138169370/.165797515で、登録のskew2・influence .25を通過した。外れ値削除0、標本追加0、confidence変更0、ratio interval=null。

名目差区間が正であることと、正式判定が未解決であることを両方報告する。既存判定の優先順はprecision未達が費用の符号判定に先行するため、点差が大きいことを理由に登録精度基準を外さない。点削減率はこの一組の保存点値の表示算術であり、新しい推定法・CI・研究横断ランキングではない。

**この保存条件でENTRYの低い費用点値は観測されたが、登録した統計・精度条件を全て満たす追加利益の最終判定は未解決。** 通常CDF一般、正則化だけの因果効果、最適能力、新規性、RC-DF等に対する普遍優位へ拡張しない。

## 7. Correctness・実行call・資源

新native128標本すべてで全512入力/出力の位相込みraw/compiled/reference、実raw/compiled inverse、return、X/Yを通過。Correctness FAIL0、worker128件PASS。原本128行の[価格・診断CSV](../../artifacts/research_c_cdf0_relocated/2026-11-08/publication_run02/native_samples128.csv)を公開する。登録allowance component/adjoint1e-10、complex signal2e-10、rtol0、phase fitなしを維持した。

| 保存診断の全128件最大 | 残差 |
|---|---:|
| raw/reference max-entry | 2.922924368344142e-14 |
| compiled/reference max-entry | 4.465833995601025e-12 |
| compiled実inverse/reference | 4.5937736134633994e-12 |
| compiled実inverse return | 1.395552077419527e-12 |
| compiled X/Y complex signal | 1.256143472281133e-12 |

これは有限binary64数値pilot。全law合成やactual CAへ格上げしない。旧SOURCEのspin/JW・leakage・scalar/control相対位相の検査を使用し、少数stateやsector-only回路検査へ置換していない。

| 科学範囲 | 実績 | 登録上限 |
|---|---:|---:|
| CDF候補／Z再配置／PF kernel | 各1 | 各1 |
| exact evolution／reference round | 13／65 | 13／65 |
| selection native／final native | 64／64 | 160／64 |
| native標本／kernel build／compile | 各128 | 各224 |
| 実inverse build | 256 | 448 |
| native kernel full512 decode | 512 | 896 |
| shell full512 decode | 6 | 6 |
| 登録主比較 | 1 | 1 |
| 新fit／DF norm／ENTRY／強D／量子shot | 全て0 | 全て0 |

保存`call_counts`の合計**1,710成功callback**。そのうち子workerの9種×128=1,152件は、既存backendがworker完了記録を親へ転記したもの。転記時に同じ科学処理を二重実行したわけではない。worker原本との件数照合、登録scope上限との照合は[監査JSON](../../artifacts/research_c_cdf0_relocated/2026-11-08/execution_run02/result_audit.json)に保存。Run02内retry/resume0、初回環境失敗後の許可済みtechnical relaunchは1。

開始preflightは13:54:41 JST。GNU timeのrunner wallは**156.88秒（2分36.88秒）**、user4103.34秒/system153.67秒、exit0、swap0。各13roundの並列とnative32 workerを使用し、実際のaffinity128 CPU・MemAvailable約1TB・空きdisk約591GBを確認した。最大worker RSSは1,060,708 KiBで、同時全worker合算peakを測った値ではない。CPU/GPU共有設定や他processを変更せず、人工的なCPU/RAM capも追加しなかった。[time](../../artifacts/research_c_cdf0_relocated/2026-11-08/execution_run02/process_time.txt)。

## 8. 保存物・公開範囲・再現手順

rawは**1,524ファイル、811,462,302 bytes**をfresh run02に保持。うち科学証拠1,518ファイル・607,493,710 bytes、worker用pickle/mmap cache6ファイルは別区分。NPZ129、QPY512、JSON869等が含まれる。[全raw file hash一覧](../../artifacts/research_c_cdf0_relocated/2026-11-08/publication_run02/raw_files.json.gz)はcacheを含む全ファイルのSHA256/sizeを記録する。

直接公開するのはrootの結果/選択/final/SOURCE/契約/STOP、適格2 policyのresult、代表law、65round転記、128価格/診断CSV、修復/preflight/time/log/監査と本報告。**全NPZ/QPY・全raw binaryをGitHubへ公開したとは主張しない。** 原本所在は `/tmp/research-c-cdf0-z-rerun-20261108/artifacts/research_c_cdf0_relocated/2026-11-08/science/run02`。公開ファイル一覧も別に固定する。

監査は[保存物監査script](../../artifacts/research_c_cdf0_relocated/2026-11-08/execution_run02/audit_saved.py)によるJSON/JSONL scalar転記とbytes/hash確認で、NPZ/QPY decode・科学関数・新統計推定を呼んでいない。SOURCE112、保存input32、旧科学SOURCE68、旧run01原本4ファイルのpostflight hashは全一致。旧proposalとrun02契約との差はfresh_outputだけ、科学SOURCE bytes/乱数namespaceは同一。

実行履歴は以下。これは再起動の許可ではない。

```bash
cd /tmp/research-c-cdf0-z-rerun-20261108
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/research-c-cdf0-aer-cpu-20261108:/tmp/research-c-cdf0-preparation-deps-20261108:src \
/home/AbeHiromu/projects/partially-randomized-trotter/.venv/bin/python -B -u -c \
'from pathlib import Path; from trottertracks.c_cdf0_relocated import execute; print(execute(Path.cwd(), "docs/research/research_c_cdf0_relocated_contract_rerun02_20261108.json"))'
```

## 9. 未解決事項とMANDATORY STOP

科学runnerは正常完了し、STOP.failed=false、completed=[CDF0]、mandatory_stop=true。旧run01のSTOP.failed=trueは保持し、今回の正常STOPで旧失敗を書き換えない。

次に残る研究判断は、登録精度目標未達を踏まえ、この限定した点値・名目区間をどの範囲の知見として用いるかである。追加標本、精度目標変更、新fit、ZZ-only正則化、r/K/LD/δ拡張、強D再計算、H6、詳細科学レビューは自動開始しない。重要科学レビュー・一般的なENTRY優位・研究継続/論文化判断は今回実施していない。

本結果・監査・必要原本をcommit/公開し、remote SHA/tree・公開本文・SOURCE/result・worktree cleanを照合した後、**MANDATORY STOP**する。
