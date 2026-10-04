# PR-2 M2 held-out transfer：結果と照合

検証日：2026-10-04 JST。terminal status：`TRANSFER_SUPPORTED`。

developmentで固定した5構成をH4 linear 1.30 Åへ一度だけ移した結果、全件がaccuracy適格で、
重大primary cost underestimateのないB2二件が6指標point Paretoに残った。primary RZの最小B2／最小endpoint比は
0.586090、engineering upper 2SEは0.593584で、事前条件1.10を満たす。
固定構成のtransferを支持するlocal evidenceであり、held-outでのmethod最適性や一般的なrank 3・q=1の最適性は示さない。

`mandatory_stop_reached=true`、`next_stage_authorized=false`、`automatic_next_stage=null`。
追加trajectory、retuning、別geometry・分子、S3、長RPE、最終総cost評価へ進まず、研究方針の全面reviewへ戻る。
結果と本照合はsource-boundのlocal execution evidenceで、軽量artifactと本報告をresult commitへ収録する。
commit固定後も、immutable CIや外部再現ではない。

## 1. 固定条件と実行認可

対象はH4 linear 1.30 Å、STO-3G、DF rank 12 Hamiltonian、8 system qubits、`T=0.8`。
DF-prefixの決定論側splitはB2が`L_D=3`、B0 discardが6、B1 full deterministicが12、B3 random-dominantが0。
B2/B0/B1は`q=1, delta=0.8`、B3は`q=8, delta=0.1`で、delta sweepは行わない。
候補のmethod/rank/q/r/K/T、threshold、32 trajectoryを変更せず、ineligible候補の置換も行わない。

正本は[契約v1](research/pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md)、
[usable B2 amendment v2](research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md)、
[実行authorization](research/pr2_matched_accuracy_m2_transfer_execution_authorization_v1.md)。
利用者が提示した外部最終review `APPROVE_M2_EXECUTION`と明示指示「M2で実行して」を受け、
固定root/output/commandで一度だけlaunchした。reviewは利用者提供の回答であり、runnerが独立承認を機械検査したとは主張しない。

compilerはQiskit 1.3.0、basis `rz,sx,x,cx`、optimization level 1、transpiler seed 17。
backend、coupling map、layout/routing指定なし。Python 3.11.0rc1、NumPy 1.26.4、SciPy 1.14.1。
最大5 spawned CPU workers、各BLAS thread 1、環境変更なし。状態準備を除くfull measured Hadamard wrapperを測った。

## 2. Accuracyとprimary resource

corrected finite-RTE signalとanalytic shot accountingはM1と同じ式を使う。
complex-signal accuracyは0.05、各軸allowanceは`0.05/sqrt(2) - axis_bias`。
primaryは`G_RZ = N_real*E[C_cosine,RZ] + N_imag*E[C_sine,RZ]`。
shot数は解析的な必要数であり、量子shotやhardware測定を実行した数ではない。

| 固定candidate | corrected bias abs | normalization | N_real / N_imag | primary RZ work（point） | primary SE | 6指標point Pareto |
|---|---:|---:|---:|---:|---:|---|
| B2-rank3-q1-r4-K2 | 0.003725912 | 1.062030845 | 9,876 / 8,026 | 111,753,794.4375 | 714,495.1212 | yes |
| B2-rank3-q1-r8-K2 | 0.003725822 | 1.030758331 | 9,303 / 7,561 | 112,153,505.0000 | 1,124,290.2641 | yes |
| B0-rank6-q1-r0-K0 | 0.007082059 | 1.000000000 | 10,943 / 7,267 | 190,676,910.0000 | 0 | no |
| B1-rank12-q1-r0-K0 | 0.003732646 | 1.000000000 | 8,760 / 7,117 | 318,175,080.0000 | 0 | no |
| B3-rank0-q8-r32-K4 | 0.000044323 | 1.471409676 | 15,218 / 15,183 | 1,069,452,078.1875 | 13,718,013.0240 | no |

accuracy適格は5/5、accuracy feasibilityの誤分類は0。各candidateの6指標matched work、
軸別mean、paired trajectory、signal/cost fingerprintは[result JSON](../artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json)に保存した。
2つのB2のprimary point差は約0.358%で、各SEより小さいため、r4/r8の厳密winnerを確定しない。

## 3. 事前固定されたtransfer判定

usable B2は`method=B2 AND accuracy_eligible=true AND primary major_cost_underestimate=false`。
重大underestimateは、held-out shots×developmentの軸別1-shot costによる予測に対し、
actual primaryが**厳密に10%超**上回る場合である。

| candidate | 事前予測primary RZ | primary underestimate | major | usable B2 |
|---|---:|---:|---|---|
| B2-rank3-q1-r4-K2 | 113,851,685.0625 | 0% | no | yes |
| B2-rank3-q1-r8-K2 | 114,830,138.0000 | 0% | no | yes |
| B0-rank6-q1-r0-K0 | 192,115,500.0000 | 0% | no | no（endpoint） |
| B1-rank12-q1-r0-K0 | 320,207,336.0000 | 0% | no | no（endpoint） |
| B3-rank0-q8-r32-K4 | 972,635,343.5313 | 9.954063% | no | no（endpoint） |

underestimate欄は`max(actual/prediction - 1, 0)`。0%はpredictionと完全一致という意味ではない。
B3は10%境界に近いが、事前規則のpoint判定を変更しない。

usable B2二件とaccuracy適格endpoint三件が存在する。B2二件が6指標point Paretoの証人となり、
ratio経路も最小B2 r4対最小endpoint B0で次を満たす。

- point：0.5860898125、SE：0.0037471507。
- point ± 2SE：`[0.5785955111, 0.5935841140]`、upperが1.10以下。
- terminal：`TRANSFER_SUPPORTED`。

Re/Imは同じrandom evolutionを共有し、SEはtrajectoryごとのshot-weighted sumからpaired covarianceを保持して計算した。
候補間ratioはindependent-candidate delta methodで、point ± 2SEはengineering intervalであってformal CIではない。
これは固定5構成内の比較であり、held-out上で各methodを再最適化した比較ではない。

状態準備のRZ相当cost `P>=0` はsecondaryである。保存済みpoint lower envelopeは
`P<385.078`でB2 r4、以後`P<208,735.132`でB2 r8、それ以上でB1となる。
これは共通準備costを加えた感度モデルであり、状態準備回路をcompileした結果ではない。
大きいPまでB2が常に最良というclaimは行わず、terminal判定にも使わない。

## 4. 実行資源・完全性

science runは2026-10-04 05:54:28.620222–06:10:40.444279 UTC（14:54:28–15:10:40 JST）。
wall timeは971.824秒、約16分12秒。exit code 0。

| candidate | actual wrapper computed / reused | trajectory | cell wall秒 |
|---|---:|---:|---:|
| B2-rank3-q1-r4-K2 | 64 / 0 | 32 | 118.155 |
| B2-rank3-q1-r8-K2 | 64 / 0 | 32 | 127.708 |
| B0-rank6-q1-r0-K0 | 2 / 0 | 0 | 6.251 |
| B1-rank12-q1-r0-K0 | 2 / 0 | 0 | 12.351 |
| B3-rank0-q8-r32-K4 | 64 / 0 | 32 | 914.203 |

unique wrapper keys、actual compile、completed checkpointは各196。reuse 0、未解決予約0、壊れたrecord 0。
random trajectoryは96、step/occurrence sampleは8,576。snapshot hash/readとloadは各1、signal評価5。
追加snapshot access、candidate search、再最適化、追加trajectory、ground-state solve、分子再計算、量子shot実行は0。
GPU query/allocation/kernelは全て0。

parent CPU timeは55.692秒、parent peak RSSは1,291,228 KiB、最大worker peak RSSは1,754,368 KiB（約1.67 GiB）。
これらは各processのpeakであり、同時aggregate memoryの実測値ではない。
数値gateはblock reconstruction relative error `1.828e-15`、reference Rayleigh residual `6.635e-16`、
tail reconstruction absolute error最大`2.240e-14`。追加のexact ground stateは求めていない。

## 5. 保存済み結果の照合とtests

held-out NPZを再度resolve/stat/hash/loadせず、saved JSONとcheckpointだけを検査した。
128 source blobs、plan/authorization/environment、候補・axis・seed・196 keys、signal/cost fingerprints、
paired evolution、平均・SE、usable B2、strict 10% boundary、decision priority、result schema/fingerprint、
marker、runner manifestのbytes/SHA-256が全件一致した。runner生成manifest/result/markerは書き換えていない。

| local test scope | pre-run | post-run | fail / skip |
|---|---:|---:|---:|
| M2 execution/contract＋M1-B1 contract/execution/result-validation | 84 passed（6.49秒） | 84 passed（6.54秒） | 0 / 0 |
| DF partial repeated/repeated cost＋RPE Hadamard helper回帰 | 134 passed（18.43秒） | 134 passed（18.37秒） | 0 / 0 |

正確なcommand argv、Python identity、environment、開始・終了UTC時刻、stdout/stderrは
[launch audit](../artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/launch_audit_v1.json)と
[post-execution audit](../artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/post_execution_audit_v1.json)へ収録した。
full repository testsは実行せず、許可されたsynthetic/保存JSON/mock testsだけを使った。
testsのsynthetic compileをM2のH4 science wrapper数へ混ぜない。
launch audit内のpreflightの旧pending-review labelは検査scriptの履歴labelであり、上位に記録した実行承認・launchと区別する。

## 6. Provenance

branch：`pr2-v4-s2-parallelization-20260928`。run時HEAD/review bundle：`40a11d02a67954175686b2e532b83c7953ec8316`。
actual source：`2978e2fea672b7a1ff20cac74269ec9a610159dc`、authorization：`90a9f24707ec439cd3618cc4ec2616a8caaf1148`。
M1-B1 evidence：`8e0814e70c14ecf526444fac8a2142799610dc96`。
実行完了後の利用者指示「コミットプッシュして」を受け、本報告と軽量result/auditを同じresult commitへ収録する。
正確なresult commit hashとpush結果はcommit後のhandoffで示す。run時HEADとactual source commitは変更しない。

| identity | SHA-256 |
|---|---|
| execution plan v1 | `2aa09a927e5ac58ebe417397802ace0e70c0097e8d0c53c05457075a41e85527` |
| authorization JSON v1 | `dbc8b66fad5316004ef404fe8253d3f6cb0f29bf7065502ea995240e5fbd7ff1` |
| result schema v2 | `8612a9ba3ebb1a281214ad687b0ca766a51783172bda6cf5a40fca6847c06a36` |
| source module | `3b0d1e76ca488a28cf81165df1b36bdd671d07b17c4c155476ce38162c30c35b` |
| source runner | `1783da8cbdb22b60e491d073c1bc3e77dbb41b470998fb9f19cf8d1cf045224c` |
| source test | `e28a58f80d61b5b0f621aa8fb25b8f416fc272ac5abd5b660780b12574cd1234` |
| result JSON v2 | `f41a92beb57e59cddc8c063b061c40acd4da50cb76ac0698efc2bce004937931` |
| runner manifest | `fa6dad1b5c49c08ee4c6d96447847cc0ad357b17f67937a405f0550a4df12546` |
| complete marker | `619c33cb7d3285216c51a8332627a15a8995abd061c54564e1cbc03ee446900a` |
| launch audit | `abdd63942d8be1f6927712ae84b623972453910b47335f980009cd8bca36b2d7` |
| post-execution audit | `6f737aefb3e22a09c3e8eca9f821d607059c08494cfbaa7a1f43e8cc0a612035` |

result fingerprint：`d9003ac6e32b2888d69aa1fed226dbef48cf13e1a6f10829e137c136824bb320`。
input snapshotのfile SHA-256は`ad7e3e7165c55dbaa395eef7a1dd74db89e1f7ab29a69ac64333f4aebf8b3e37`、
Hamiltonian hashは`a70d9619e794a0238aae57096a33759bf72d7b294a251b235508cd2ccc6b16c0`。
これらはrunで照合したidentityをsaved resultから引用したもので、検証時の再読み込みではない。
source、旧contract/plan、M1-B1 result/validationは変更しない。

軽量result/audit/reportだけを成果物候補とし、`.runtime`、one-shot registry、NPZ、matrix/state/vectorはcommitしない。
既存Markdown整理の未commit変更もこの実行とは別に保持した。
次に必要なのは本結果を使った研究方針reviewであり、runnerのSUPPORTEDは追加計算の認可ではない。
