# O1 M1：固定C0総費用pilot（並列継続）

SERVER 2（GPUA100-gleap02）。**固定C0の21/21価格pointが資格を通過し、全7roundの条件付き平均費用を集計した。科学実行はmandatory STOP。** このpilotの平均推定では、Pauli/DF PRは強決定論DF-PFより総CX費用が大きい。価格CIと認証された95%達成保証はなく、一般的な優位性・UWC新規性を判定しない。

## SOURCE・入力・再利用

- 固定library：`8a13539e51bb113e82cb86ddb608e2e6e27786ca`。matched adapter、Gaussian/native回路、独立finite-RTE信号validator、resource bridge、実半径shot accounting、Hadamard full no-prep価格provider、強DF-S2を再利用し、固定libraryのbyte一致を実行前に確認した。
- 並列SOURCE：`93d6b49b77c619e0e12e669b5a5884ffea343ff0`。既存[継続driver](../../../scripts/run_o1_m1_c0_parallel_resume.py)と[並列provider](../../../src/trottertracks/o1_m1_parallel_price.py)を変更せず使用した。科学SOURCE hashは[result](../../../artifacts/track_a_o1_m1_c0_parallel_science_20261106/run01/result.json)の`parallel_continuation.source_files_sha256`へ保存した。本branchの追加実装は、既存[報告用6指標集計](../../../scripts/summarize_o1_m1_c0_pilot.py)とその人工テストの再利用だけである。
- input：commit `4591b8d75a2a6a55c74a51225f0c61648023c8ee`、`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`。SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。旧inputを書き換えず専用`/tmp`コピーを保持した。
- 旧逐次runの16完了pointと全7roundのPR/baseline信号を再計算せず引き継いだ。旧result SHA256 `1242e030c7a28d6476aed9c699b2abe3acd41c51d2281329b9ccc8c888786cbe`、旧保全公開commit `dc6432a69dee24aec8bad19968a643aaee8ac5ee`。新しい科学計測は未完了5pointだけであり、旧signal_m*.jsonは[旧branchのraw](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/dc6432a69dee24aec8bad19968a643aaee8ac5ee/artifacts/track_a_o1_m1_c0_science_20261106/run01/)を参照する。
- 新result SHA256：`5b8442abad948a11e005c6ce755a87a9ff4bdf87f2a77b6d117e141cc8c9e64f`。元Hの保存`state`を厳密基底状態と仮定し、状態準備コストを除外する。`DF_state`への変更、state solve、新fit、rank探索はない。

## 固定C0条件

H4-A、linear H4、1 Å、STO-3G、保存DF rank10、共通`T_eff/V_square`（LD0）、全signed factor、核scalar、faithful tail identity。`epsilon_E=.1 Ha`、条件付き成功目標95%、`delta=.05 Ha^-1`、m=0..6（q=1,2,4,8,16,32,64）、r=2、K=2、`beta_RPE=.30 rad`、energy区間`[-10,0] Ha`を維持した。PF/RTE/stat/model/実装/reserveの位相配分は`.06/.03/.15/.03/.01/.02`、各軸の失敗配分は`.05/(2*7)`。

Qiskit 1.3.0、`rz/sx/x/cx`、optimization 1、seed 7、backend/target/coupling mapなし（all-to-all）、routing指定なし。連続RZの価格でありfault-tolerant T価格ではない。主baselineは強DF-S2の4 substeps、`relative_fusion=True`、`exact_spin_lift_spatial_eigh_v1`。PR価格MCは各arm/round32 trajectory、PCG64 seed `2026110600+100*arm_index+m`（P=0、DF=1）。baselineは各軸の決定論価格。旧F2価格、旧4条件、単位PF半径、未検証proxyで補完していない。

数値allowanceは[事前固定C0報告](../track_a_o1_m1_cost_validation_preparation_20261106/REPORT.md)のbinary64/program-scale式のまま適用した。結果を見て配分・閾値・標本数・許容値を変更していない。元stateのRayleigh energyは`-2.1663874486347643 Ha`、元H eigen residual L2は`1.916853751084478e-15`。ground-state性自体は本pilotの仮定であり、新たな基底状態証明を行っていない。

## 信号・normalization・shot

元H stateに対する全7roundのPR operator/controlled-signal検査、PF/RTE/model/実装phaseと実半径のC0条件を通過した。強baselineも全roundで元stateの独立S2参照とのcontrolled state-actionを通過した。q=1のfull controlled operator最大差は`1.8962726262663195e-14`で、phase alignmentはしていない。q>1についてfull operator全成分検査とは主張しない。PF/RTE誤差は原理的近似誤差で、実装差と別fieldへ記録している。

| q | P attenuation | DF attenuation | P shot/axis | DF shot/axis | PF phase bound | RTE phase bound |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.679025234 | 0.6854604478 | 2459 | 2413 | 3.50256345e-06 | 1.057745294e-05 |
| 2 | 0.4610752684 | 0.4698560255 | 5332 | 5135 | 6.77541595e-06 | 2.092512506e-05 |
| 4 | 0.2125904031 | 0.2207646847 | 25082 | 23259 | 1.332111884e-05 | 4.162025617e-05 |
| 8 | 0.04519467951 | 0.048737046 | 555014 | 477265 | 2.641247006e-05 | 8.30096128e-05 |
| 16 | 0.002042559056 | 0.002375299653 | 271773370 | 200964493 | 5.259465912e-05 | 0.0001657843905 |
| 32 | 4.172047496e-06 | 5.642048443e-06 | 65164976569615 | 35631910593242 | 0.0001049563693 | 0.0003313178659 |
| 64 | 1.740598031e-11 | 3.183271063e-11 | 3746527790271109800132608 | 1120156469303095630757888 | 0.0002096802271 | 0.0006623369369 |

cosine/sineのshot数は各arm/roundで同じ。q64のnormalization productはP `57451518508.36441`、DF `31414227069.492317`、PF半径下界は両arm `0.99994107284067169`、観測半径下界はP `1.7393426682643843e-11`、DF `3.1809752083705854e-11`。全normalization・複素信号・誤差は新resultと旧rawに保存する。PF/RTE位相誤差が配分内でも、PR減衰により長roundの必要shot数が増大した。

強baselineは全roundでA=B=1、各qで実際の元state PF半径を使い、shot/axisは全round1134。q64のPF phase boundは`0.00012453277131208758`、controlled state-action L2差は`9.3467259195412301e-13`。量子測定samplingを実行したのではなく、固定stat/失敗配分の式から必要shot数を条件付き算出した。

## full no-prep価格と不確かさ

全21pointで価格取得済み、全PR pointでCX relative sample SEが固定10%以下だった（最大約4.7018%）。各軸のfull Hadamard interrogationを独立にcompileし、state preparationを含めない。

| q | arm | CX mean cosine | sample SE cosine | CX mean sine | sample SE sine |
|---|---|---:|---:|---:|---:|
| 1 | P | 197.59375 | 9.290436829 | 197.59375 | 9.290436829 |
| 1 | DF | 256.4375 | 10.56011378 | 256.4375 | 10.56011378 |
| 1 | strong_DF_PF | 34208 | N/A（決定論） | 34208 | N/A（決定論） |
| 2 | P | 309.125 | 11.67535918 | 309.125 | 11.67535918 |
| 2 | DF | 471.28125 | 16.34563881 | 471.28125 | 16.34563881 |
| 2 | strong_DF_PF | 68320 | N/A（決定論） | 68320 | N/A（決定論） |
| 4 | P | 520.28125 | 16.2644622 | 520.28125 | 16.2644622 |
| 4 | DF | 847.34375 | 22.16605265 | 847.34375 | 22.16605265 |
| 4 | strong_DF_PF | 136544 | N/A（決定論） | 136544 | N/A（決定論） |
| 8 | P | 1053.71875 | 19.50678306 | 1053.71875 | 19.50678306 |
| 8 | DF | 1649.625 | 35.86752479 | 1649.625 | 35.86752479 |
| 8 | strong_DF_PF | 272992 | N/A（決定論） | 272992 | N/A（決定論） |
| 16 | P | 1998.59375 | 29.87746681 | 1998.59375 | 29.87746681 |
| 16 | DF | 3228.09375 | 49.81593862 | 3228.09375 | 49.81593862 |
| 16 | strong_DF_PF | 545888 | N/A（決定論） | 545888 | N/A（決定論） |
| 32 | P | 3858.6875 | 42.93769956 | 3858.6875 | 42.93769956 |
| 32 | DF | 6620.125 | 55.14001149 | 6620.125 | 55.14001149 |
| 32 | strong_DF_PF | 1091680 | N/A（決定論） | 1091680 | N/A（決定論） |
| 64 | P | 7598.875 | 60.7101082 | 7598.875 | 60.7101082 |
| 64 | DF | 12805.375 | 120.9271621 | 12805.375 | 120.9271621 |
| 64 | strong_DF_PF | 2183264 | N/A（決定論） | 2183264 | N/A（決定論） |

全副指標と軸別sample SEはresultの`priced_rounds[].cost`へ保存した。価格MC32は古典標本数で、量子shot数へ重ねて乗算しない。X/Yが同じtrajectoryを共有することを無視した独立軸CIも作成していない。sample SEはCIではなく、全round価格CI・総費用CIは未算出（null）。

## 全7roundの条件付き平均総費用

同じ記録済みNに対し`Σ_round Σ_axis N_axis E[C_axis]`を集計した。[6指標summary](../../../artifacts/track_a_o1_m1_c0_parallel_science_20261106/run01/mean_cost_summary.json)と主CXの既存`total_cost`が根拠である。depthの総和はshot数を掛けたper-shot depthであり、並列量子hardwareのwall-clock timeではない。

| arm | CX | RZ | CX depth | RZ depth | total depth | circuit size |
|---|---:|---:|---:|---:|---:|---:|
| P | 5.69387927e+28 | 1.33083458e+29 | 3.52068241e+28 | 3.91432540e+28 | 8.69168690e+28 | 2.50706166e+29 |
| DF | 2.86880473e+28 | 8.45564113e+28 | 1.39714316e+28 | 1.99816311e+28 | 4.33438245e+28 | 1.58443122e+29 |
| strong_DF_PF | 9.82700813e+09 | 1.52846394e+10 | 8.89749101e+09 | 7.85167765e+09 | 1.70825828e+10 | 2.89451141e+10 |

この条件付きCX平均のbaseline比はP約`5.7941e18`、DF約`2.9193e18`、DF/Pは約`0.50384`である。辞書間の点推定差があっても、両PR armともこのC0では強DF-PFより総費用が大きい。

これは固定C0・元H厳密基底状態・無料state preparation・固定compilerと記録済み実半径/誤差仮定に条件づけた平均価格の算術である。既存accountingの`guarantee_status=not_certified`と`price_uncertainty_qualified=false`を保持する。C0の数値feasibilityとMC SE診断が通過しても、厳密なenergy/alias/状態誤差・IID・価格信頼区間の包括的保証へ昇格しない。このpilotではPRの総CX費用改善を示さず、一般的な手法優位性やUWCに対する新規性は判定しない。

## 並列切替・テスト・STOP

ユーザーが旧正常runの計画停止と未完了pointの同一seed再計測を明示許可した。旧PID1848166のowner/args/startを照合してSIGINTし、exit130と消失を確認した。旧resultのRUNNING checkpoint・raw・driverは不変更で、[旧transition](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc6432a69dee24aec8bad19968a643aaee8ac5ee/artifacts/track_a_o1_m1_c0_science_20261106/run01/transition.json)と[partial報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc6432a69dee24aec8bad19968a643aaee8ac5ee/docs/research/track_a_o1_m1_c0_science_20261106/REPORT.md)へ経緯を保存した。旧停止は科学FAILではない。停止時stackは`canonical_qiskit_circuit_fingerprint`内のQiskit gate definition展開だったが、全runtimeの定量profileではない。

新並列runは1回だけ起動し、旧16行と全信号をcarry、SOURCE/input/compiler/law/contextを照合して未完了5点を計測した。PRは親が同じseedでevent列を生成してtrajectory/axisへ分配、最初の1taskの価格を再利用してpeak RSSを測り、残63workerで集計した（全64axis task、worker内thread1、submission order）。先頭taskも32 trajectory内に含み、追加標本ではない。baselineは2軸並列。実行metadataはresultへ保存した。共有環境・他ユーザーprocessや設定は変更していない。

並列SOURCEの人工テストは[準備報告](../track_a_o1_m1_parallel_price_preparation_20261106/REPORT.md)で28 PASS。本branchで再利用した報告集計helperの人工テストは6 passed in 0.08s（発明したスカラーのみ）。人工PASSをH4性能の証明とはしない。新SCIの[実行ログ](../../../artifacts/track_a_o1_m1_c0_parallel_science_20261106/run01/execution.log)と結果を保全した。

**科学statusは`C0_CONDITIONAL_PILOT_COMPLETE`、`mandatory_stop=true`。** 不足価格はなく、条件変更・追加標本・新CI・新fit・rank探索・追加実験は実施せず終了する。旧SOURCE・raw・結果・grant・marker・STOPは保全した。

次の作業：**ユーザー**がこの限定pilotの扱いと後続検証の要否・対象条件を判断する。**Codex**は明示された新指示・必要な科学実行承認がある場合だけ次段階を実施する。現在の許可範囲では追加科学作業なし。
