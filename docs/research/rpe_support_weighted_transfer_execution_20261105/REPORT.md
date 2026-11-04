# 全18round：ideal controlled IID平均の全36軸十分条件が成立

2026-11-05 JST / LOCAL。
固定SOURCEを96 bit・1 passだけ実行し、**全18round・36軸で
U_cat＋Δ_mean＋保存shot半幅上端 < 保存εのstrict十分条件を確認した。**
判定はCONDITIONAL_IDEAL_CONTROLLED_IID_MEAN_LOCAL_CONDITIONS_CONFIRMED。
COMPLETED_MANDATORY_STOP、native exit0、欠測・エラー・retryなし。

これは独立uniform53モデルでのideal logical controlled circuitの**平均**に限定する。
任意trajectoryの旧一様上界や固定PCG64への新保証ではない。
RPE_tolerance_certified=false、scientific_status=UNDETERMINED、
U_virtual_IID_upper=null、U_exec_upper=nullを維持する。

## SOURCE・固定条件・再利用

科学SOURCEは7a39a78eccdbe457786978e14d36b2f0d7abc290。
[固定実行計画](../rpe_support_bound_sharpness_20261105/SCIENTIFIC_EXECUTION_PLAN.md)、
[数学的導出](../rpe_support_bound_sharpness_20261105/SUPPORT_BOUND_ANALYSIS.md)、
[bindings](../rpe_support_bound_sharpness_20261105/FIXED_BINDINGS.json)のまま実行した。
実行branchはcodex/rpe-support-weighted-transfer-execution-20261105。

Linear H4／隣接1.00 Å／STO-3G／保存legacy DF12、
Partial L_D3・単一δ=.02・K2、8 qubit・4電子sector70、
B保存physical stateのnormalized ray、extract_identity_phase、coefficient_atol=1e-12。
保存m0..17/q2^m/r/N/α/β/ε、component/basis/sign、
support_for_source_basis_run_le_1とtraining fingerprint、
signal側τ・保存θ・uniform53 lawを固定した。

旧b0c354ce071af71a6ce857b11f8acc0c129a05b8の218 component certificate、
source-basis color、18 boundary/lowering/一様追加上界を再利用した。
229 basis ballsはbyte保全だけで、decode・column/projector差の新評価は0。
全18categorical結果d2e7a7db6（SOURCEf4905fff5）、m17包含34840ce9b、
C run02結果82b2df6af（SOURCE7279f700a）の入力provenanceを同じvalidatorで照合した。
component/order lawや既存包含値・shot幅を作り直していない。

## 方法と得られた上界

保存component grid mass p_l、旧operator上端 d_l、source color b_lについて、
h_b=Σ_(l:b_l=b)p_l、
D_j=Σ_l p_l d_l(1−h_(b_l))^j（j=0,1,2）をexact Fractionで取得した。
保存上端の上向き9桁表示はD0=5.31263822e-12、D1=5.10549575e-12、
D2=4.92942173e-12。これは旧certificateの確率重み付き量で、実biasの観測値ではない。

rotationでは|sin θ_n|、productでは1を係数とする。
singleton runだけsupportを使うSOURCEに対し、tail端/内部の隣接color確率を
それぞれD1/D2へ取り込む。r=1,n=0の隣接なしはD0で別扱いする。
12個の|sin θ|上端を96-bit Arbで評価し、6rのη_rを構成した。
残りはすべてexact rational計算。
誤差の符号相殺や実装側event平均の独立積分解は用いない。

| r | η_r上端（上向き9桁表示） |
|---:|---:|
| 1 | 6.26497102e-14 |
| 2 | 5.98943855e-14 |
| 4 | 5.86987387e-14 |
| 8 | 5.81011467e-14 |
| 16 | 5.78024115e-14 |
| 32 | 5.76530596e-14 |

各mで
Δ_mean=min(2,旧Δ_uniform,Δ_boundary＋q min(2,η_r)＋Δ_lowering)
を計算し、保存U_cat＋Δ_mean＋保存shot半幅上端 < 保存εをFractionでstrict比較した。
等号をPASSにしない。旧I_cat/U_catに含まれるπ64項を二重加算していない。
N/α/β/εの再計算・配分変更、追加B除算も行っていない。

| m（q=2^m） | r | 新Δ_mean上端 | 旧Δ_uniform上端 | cosine | sine |
|---:|---:|---:|---:|---|---|
| 0 | 1 | 6.26546975E-14 | 2.12665359E-7 | CONFIRMED | CONFIRMED |
| 1 | 1 | 1.25309395E-13 | 4.25330717E-7 | CONFIRMED | CONFIRMED |
| 2 | 1 | 2.50618790E-13 | 8.50661433E-7 | CONFIRMED | CONFIRMED |
| 3 | 1 | 5.01237580E-13 | 0.00000170132287 | CONFIRMED | CONFIRMED |
| 4 | 1 | 1.00247516E-12 | 0.00000340264573 | CONFIRMED | CONFIRMED |
| 5 | 1 | 2.00495032E-12 | 0.00000680529146 | CONFIRMED | CONFIRMED |
| 6 | 1 | 4.00990064E-12 | 0.0000136105830 | CONFIRMED | CONFIRMED |
| 7 | 1 | 8.01980128E-12 | 0.0000272211659 | CONFIRMED | CONFIRMED |
| 8 | 2 | 1.53342395E-11 | 0.000108884664 | CONFIRMED | CONFIRMED |
| 9 | 2 | 3.06684789E-11 | 0.000217769327 | CONFIRMED | CONFIRMED |
| 10 | 4 | 6.01126155E-11 | 0.000871077307 | CONFIRMED | CONFIRMED |
| 11 | 8 | 1.19001363E-10 | 0.00348430923 | CONFIRMED | CONFIRMED |
| 12 | 8 | 2.38002725E-10 | 0.00696861846 | CONFIRMED | CONFIRMED |
| 13 | 16 | 4.73558211E-10 | 0.0278744739 | CONFIRMED | CONFIRMED |
| 14 | 16 | 9.47116422E-10 | 0.0557489477 | CONFIRMED | CONFIRMED |
| 15 | 32 | 1.88933888E-9 | 0.222995791 | CONFIRMED | CONFIRMED |
| 16 | 32 | 3.77867776E-9 | 0.445991582 | CONFIRMED | CONFIRMED |
| 17 | 32 | 7.55735552E-9 | 0.891983163 | CONFIRMED | CONFIRMED |

上表の数値は原RESULTに保存された正のrational上端を9有効桁へ上向き表示したもの。
正式なmean追加上端、U_cat、shot幅、ε、拡張interval、total_upper、判定は
[AXIS_RESULTS.tsv](AXIS_RESULTS.tsv)の36行と原RESULTに保存した。
表への転記・表示で新しい上界やstrict判定を再計算していない。

m17の新しい平均専用追加上端は上向き表示7.55735552e-9。
全36軸のstrict条件が成立したので、旧一様転送では未判定だったm6とm10..17についても、
**IID平均という対象では**局所十分条件を確認できた。
旧一様結果の18/36成立・18/36未判定を変更・置換した結果ではない。
新旧の上端は保証対象が異なり、新値を任意wordへの上界として使ってはいけない。

## 実行・科学call・資源

~~~sh
/usr/bin/time -v -o artifacts/rpe_support_weighted_transfer/2026-11-05/run01.resources.txt \
 /tmp/prt-rpe-m17-enclosure-venv-20261103/bin/python \
 docs/research/rpe_support_bound_sharpness_20261105/run_weighted_transfer.py \
 --run --output artifacts/rpe_support_weighted_transfer/2026-11-05/run01 \
 > artifacts/rpe_support_weighted_transfer/2026-11-05/run01.stdout.log \
 2> artifacts/rpe_support_weighted_transfer/2026-11-05/run01.stderr.log
~~~

default byte/schema/literal・固定SOURCE/runtime・fresh directoryを確認後、1回だけ起動した。
科学callは218成分のmoment集約1回、sin上端12、η_r 6、転送18、strict比較36。
Arb96 bit・1 pass、追加precision・retry/resumeは0。
H/state load、preparation/support再登録、basis/column/projector/operator/mean matrix生成、
CDF/atan再計算、sampling/shot、trajectory、長q build/transpile、D/費用/noise/状態準備は全て0。

共通のexact集約を1回だけ計算して6rで共有し、小さいscalar処理を逐次評価した。
spawn/GPUは使わず、CPU/RAM/workerの人工的上限やwall-time capは設けていない。
起動時affinity32、MemAvailable=54,431,072 kB、parent peak RSS=110264 KiBを記録。
preflightのmemory pressure avg10は0だった。共有環境・他jobは変更していない。
NumPy1.26.4／Qiskit1.3.0／python-flint0.9.0／FLINT3.6.0・Python3.11・RNEをSOURCEへ照合した。

runner elapsed=0.03355226665735245秒、timeのwall=.60秒、maximum RSS=111884 KiB。
起動・import・入力照合はrunner elapsedの外側も含むため、両時間の対象は異なる。
比較測定を行っていないのでspeedup倍率は主張しない。stderrはempty、native exit0。

## 原本・技術照合・保全

[RESULT.json](../../../artifacts/rpe_support_weighted_transfer/2026-11-05/run01/RESULT.json)のSHA256は
df9b0082b6b6f51ff01058639f92e0d873d5b1f28c51cc2cc666078b3746f84e。
RESULTと隣接stdout/stderr/resourcesの4原本を編集せず保存した。
[VALIDATION.json](VALIDATION.json)にbytes/SHA256、call数・36判定のliteral照合を記録した。
SOURCE無変更なので準備時の人工173 PASSを再利用し、同じ科学判定を再実行していない。
これは実行者自身のbyte/schema/literal照合であり、独立科学レビューやimmutable CIではない。

旧C run01/run02、旧96-bit 229 balls/218 certificates、
旧18round categorical包含・旧一様18/36判定、grant/ledger/marker/STOPは保全した。
研究概要・状態・manifest・当日ノートには新しい**平均専用**結果として追記した。

## 保証範囲と次段階

今回接続できたのは、固定exact-parameter logical controlled gates、
保存normalized ray、同一の独立uniform53 law、固定run_le_1 policyの下での
categorical reference平均とcontrolled実装平均の局所座標条件である。
trajectoryごとの一様誤差上界が同じ値まで縮んだことや、
実際の測定biasを直接観測したことは意味しない。

固定PCG64列の数学的IID性、Float64 state-action/guard/clip/Bernoulli、
compiled/native再帰lowering・noise・状態準備の保証は対象外。
共通E0への位相・branch/decoder・αの全体条件などRPE/energyの保証も別であり、
Dの科学処理を実行していない。
全体RPE_tolerance_certified=false、scientific_status=UNDETERMINED、
U_virtual_IID_upper=null、U_exec_upper=null、旧INCONSISTENT/u_num=null等を維持する。

このideal IID平均scopeについて、追加のsupport数値改善は今回の完了条件に不要。
次はユーザーが次の保証対象（例：既存Float64仮想測定またはPRNG接続）を別指示で選び、
Codexがその対象に必要なSOURCE・残存gapを技術整理する段階を提案する。
新科学計算・重要な別レビューは、それぞれ固定範囲への別途明示許可後に限る。

4原本と公開SHAの照合後mandatory STOP。別候補・追加精度・新科学実行へ進まない。
