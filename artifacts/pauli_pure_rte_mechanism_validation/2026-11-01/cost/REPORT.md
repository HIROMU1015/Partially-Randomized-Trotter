# H6 Pauli-JW P2：固定保存trajectoryの部品価格と結合応答

**新規compile 12/12成功、保存Full 6軸再利用、Full再build・再compile 0。P2科学処理は完了して停止。**

固定programではHybridの保持回路に大きな単独価格があり、結合の非加法性は主に正の値だった。純RTEのscalar-only保持部はEmptyと意味が異なるが、Full内の追加価格はcosineで0 RZ、sineで1 RZにとどまった。母平均・一般的方式優位・独立因果寄与を示す結果ではない。

## 1. 条件・対象・SOURCE

Linear H6、隣接1.00Å、STO-3G、中性singlet、保存Pauli-JW918非identity項＋constant、12 system＋1 ancilla、T=.8、S2、q8/K2、P=0。DF rankは適用外。central-rotation制御、cosine/+Im sineの測定付きfull Hadamard wrapper、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、追加backend/coupling/layout/routingなし。これはδ窓の掃引ではなく、固定step_time=.1の3本の保存program診断である。

| ID | candidate | L_D | r | 保存cohort/index | event数 |
|---|---|---:|---:|---|---:|
| H32 | PR_p0078_q8_r32_K2 | 78 | 32 | confirmation/0000 | 256 |
| H64 | PR_p0078_q8_r64_K2 | 78 | 64 | confirmation/0000 | 512 |
| R64 | H6_p000_q8_r64_K2 | 0 | 64 | confirmation/0000 | 512 |

事前固定indexを維持し、他のindexやcohort平均を使用していない。H32/R64はrも違うfamily比較、H64/R64はq/r/K一致だがprefix・tail・sampling law・event内容が違う。

科学SOURCE：[`11e50e262f95630b14fa4070ad30894ba3f64b74`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/11e50e262f95630b14fa4070ad30894ba3f64b74)。compile前に公開し、local/remote tree一致後に実行した。親はP1結果`bfb4ca75437a9b3605a8f57b5f82100615dd034f`。

Hybrid原本：結果`00958ae1945c780718b2deef07d5dab1096a5cd8`／SOURCE`887b69e449cb3c378286f09d0d7626599850fb67`。純RTE原本：結果`10cb3e5cca7834ebf75301bd91bd093ca51d4a6b`／SOURCE`e755ecf581600192fd2745d7066e410cebd933d3`。各data_archives.jsonから選択18 memberを復元し、part/archive/member、original gzipとraw QPY、result内QPY fingerprintを照合した。Full input6＋compiled6を読み戻し、個別保存RZ/CX/depth/size/phaseと一致した。新旧wrapper/compiler関数も同一。

[source_and_inputs.json](source_and_inputs.json)に全参照・hash・runtime・実行条件、[saved_full.json](saved_full.json)にFull価格と元resultへの対応を保存した。古いspecのNOT_EXECUTEDという準備時statusを、新しい実行statusと混同していない。

## 2. 命令・phase・部品対応

Fullは新規QuantumCircuitへ再buildせず、保存input QPYの命令を復元source tokenのdescriptorと比較した。命令名・順序・qubit/clbit・angle・phase・wrapper境界が全6軸で一致し、最大parameter差は0、phase分解誤差も0。H32命令数8,974/8,975、H64は14,516/14,517、R64は6,932/6,933（cosine/sine）。これはbyte SHA確認と区別した入力意味論の確認であり、全unitary/compiled誤差certificateではない。

| 部品 | 持つphase | aggregate body phase（rad、2π同値） |
|---|---|---|
| Hybrid共通R、R64のR | Hamiltonian constantのみ | 0.2598732288501919 |
| H32 E | 負productとorder phase | π |
| H64 E / R64 E | 負productとorder phase | 0 |
| H32 F | scalar＋event | 3.401465882439985 |
| H64 F / R64 F | scalar＋event | 0.2598732288501919 |
| Empty | なし | 0 |

負rotation符号は保存係数と角度の組として残し、aggregate body phaseをancilla P(phi)へ一度移す。Hybrid compact recordにはcontrol_sidecarを仮定せず、full recordと同じ保存application viewへ復元した。確率・angleの新生成、RNG replayは0。H32 Eは256 event中1つのorder2を含み、H64/R64は保存512 eventがorder0。

H32/H64のRは1248 source evolution tokenとscalar phaseが完全一致するため共有した。R64 Rはevolution token 0でもscalarを保持し、Emptyとは別program。Emptyは元wrapperと測定だけで、cosineのH-Hがcompileで消えてmeasure1、sineはrz2＋sx1＋measure1となった。

[semantic_checks.json](semantic_checks.json)、[parts.json](parts.json)にphase所有・event対応hash・Full照合・共有を記録した。新規input/compiled QPYは[programs/](programs/)にあり、raw/gzip SHAも個別JSONへ保存した。

## 3. 各構成のcompiled価格

C/Sはcosineとsineで表示metricsが同じという意味であり、同一programとして共有したという意味ではない。Fの行は保存個別価格、R/E/Emptyは今回の新規価格。深さは記述量であり、加法的寄与率を定義しない。

| 構成 | 軸 | RZ | CX | RZ depth | CX depth | total depth | size |
|---|---|---:|---:|---:|---:|---:|---:|
| H32 F | C/S | 4732 | 7762 | 2399 | 5615 | 8064 | 13592 |
| H64 F | C/S | 6735 | 10934 | 3265 | 8310 | 11659 | 19716 |
| R64 F | C/S | 3081 | 4344 | 1354 | 3593 | 5084 | 8742 |
| Hybrid共通 R | C/S | 2500 | 4576 | 1443 | 2816 | 4262 | 7079 |
| R64 R | C/S | 3 | 0 | 3 | 0 | 6 | 6 |
| H32 E | C/S | 2143 | 3154 | 946 | 2791 | 3776 | 6354 |
| H64 E | C | 4163 | 6326 | 1828 | 5501 | 7410 | 12486 |
| H64 E | S | 4162 | 6326 | 1828 | 5501 | 7410 | 12485 |
| R64 E | C | 3081 | 4344 | 1354 | 3593 | 5084 | 8742 |
| R64 E | S | 3080 | 4344 | 1354 | 3593 | 5084 | 8741 |
| Empty | C | 0 | 0 | 0 | 0 | 1 | 1 |
| Empty | S | 2 | 0 | 2 | 0 | 4 | 4 |

全軸・全構成のgate histogramとQPY参照は[metrics_and_interactions.json](metrics_and_interactions.json)、新規12件の個別記録は[new_metrics.json](new_metrics.json)。

## 4. 非加法性とcontextual increment

Δinteraction=F−R−E+Empty、contextual retained=F−E、contextual event=F−R。すべて整数gate count。

| 対象・軸 | Δ RZ | Δ CX | retained RZ | retained CX | event RZ | event CX |
|---|---:|---:|---:|---:|---:|---:|
| H32 cosine | 89 | 32 | 2589 | 4608 | 2232 | 3186 |
| H32 sine | 91 | 32 | 2589 | 4608 | 2232 | 3186 |
| H64 cosine | 72 | 32 | 2572 | 4608 | 4235 | 6358 |
| H64 sine | 75 | 32 | 2573 | 4608 | 4235 | 6358 |
| R64 cosine | -3 | 0 | 0 | 0 | 3078 | 4344 |
| R64 sine | 0 | 0 | 1 | 0 | 3078 | 4344 |

Hybridでは単独R=2500 RZ/4576 CXに対し、FullからEを引いた増分は2572–2589 RZ/4608 CX。今回のphase所有・削除操作では、RとEを結合したFull価格は孤立部品の和よりRZ72–91/CX32だけ大きい。したがって、Hybrid低費用の主要説明を「保持部を結合すると大きく相殺される」とする観測ではない。正の非加法性も存在する。

R64ではscalar-only Rが3 RZ/0 CXであり、Emptyのcosine0 RZ/sine2 RZとは異なる。FullとEはcosineでRZ/CX同数、sineではFullが1 RZ多い。cosineのΔRZ=-3は孤立scalar-wrapper費用が結合時に吸収される会計、sineのΔRZ=0はEmptyを引くと加法的になる会計である。これはphaseを削除できるという意味ではない。

相殺・共有・増加は全体最適化後の整数差としてのみ確認した。特定compiler pass、Pauli support、rotation angle、Gaussian操作、表現、sampling lawの独立した因果配賦を得たわけではなく、pass traceを新たに要求・実行していない。

## 5. 候補間の構造と未分離要因

H32/H64は同じRを持つ一方、Eのevent数が256/512であり、E価格はH32=2143 RZ/3154 CX、H64=4163（sine4162）RZ/6326 CX。rに加えevent列・角度・order内容も異なるため、単一eventの一定価格や母平均の比例則とはしない。H32のFull4732 RZはR64の3081 RZより大きいが、EだけならH32の方が小さい。この固定family比較では、大きな保持部価格がFull差の説明材料になる。

H64/R64は同じ512 eventだが、Hybrid Eが4163/4162 RZ・6326 CX、純RTE Eが3081/3080 RZ・4344 CXである。q/r/Kを揃えてもevent部単独価格が一致しない。Fullの差を保持78項だけに断定できず、prefix除去によるtail/law/抽選内容・Pauli support・angle分布等の差は未分離。Hamiltonian表現間のmatched比較も行っていない。

可換I/Zの保持部に理想内部PF誤差がないことと、元の項別central回路がopt1で多くのgateを要することは両立する。今回、保持回路の別実装や再最適化を比較していない。固定index0000以外の価格分布、rare-event tail、母平均価格、実compiled残差ρ、全law精度保証、一般的方式順位は未評価。元Nとの積、新N/G/SE、cohort集計は0。

## 6. P1と合わせた判断材料

[P1保存報告](../bias/REPORT.md)（結果commit bfb4ca75、科学SOURCE a0a3aff9）では、Hybridに共通するΔouter=7.016398502702e-5＋1.136269663863e-4i、|B−C|=7.731011545855e-15を報告した。I/Z保持78項の内部理想PF誤差ではなく、外側分割が残存biasのスケールに対応するというP1の限定観測である。全grid差とH32/H64 finite項のUNRESOLVEDは変更していない。ここでは数値を引用しただけで、P1再計算は0。

P2は、同じ研究系の固定programにおいて保持回路の大きな価格と、正負両方の結合非加法性を分けた。P1の信号分解とP2の回路価格分解は異なる対象であり、これだけでshot数や総費用の原因分離は完成しない。

- ユーザー/GPTが必要shotの量的影響・bias headroomの感度を知りたい場合、P1Nが対応する。今回の価格表へ元Nを掛けるだけで新しい方式順位は作らない。
- DFとPauliの差をHamiltonian表現・Gaussian操作・共通分割で説明したい場合、P3のmatched条件設計・比較が対応する。今回のPauli内P2では答えられない。
- 固定programで保持部が重いことと非加法性の存在を述べる目的なら、今回のP2範囲で材料が得られた。母平均や全law保証を述べるには別scopeが必要。

これらは次段階を選ぶ判断材料であり、P1N/P3の実施や新中心claim・正式実験方針を採択する判断ではない。

## 7. 再利用・検証・保全・STOP

D1のevent復元方針/interaction会計をcompact/fullの最小viewへ拡張し、D2 central_control_bodyのtoken emitterをdescriptor用にも共有した。元append helpers、両軸Hadamard wrapper、compiler/measurement、archive gzip復元を再利用した。D1 generic controlや新sampler/law/angle生成は使用していない。[固定SOURCEと実行条件](../../../../docs/research/h6_pauli_p2_cost_20261101.md)。

新人工14件PASS、既存D2関連回帰7件PASS。人工compileは3-qubit1件であり、新規H6 science12件とは別。作成者によるcorrectness確認で、別reviewerの独立科学レビューではない。科学実行はR4/E6/Empty2=12件submit・12件成功、失敗0、retry0、12 workerを全reap、親exit0、全体12.07秒。64 CPU affinityと十分な空きRAMを観測し、12独立仕事を並列化した。共有環境/他ユーザーへの変更なし。

実行後のSOURCE11 file hashは一致。P1 code/protocol6 fileとbias証拠tree `f9bfbbacc2b222b68416a678b2a09e204cf9eb42` はbaseと同一。旧SOURCE/input/raw/result/grant/ledger/marker/STOP・main/元研究branchは書き換えていない。新grant/approval証明機構なし。

[execution_status.json](execution_status.json)、[finalization_checks.json](finalization_checks.json)、[STOP.json](STOP.json)を保存した。結果commitのGitHub公開後、remote SHAと変更全fileの全文/bytesを照合し、その完了を最終チャットに記載する。

**STOP_P2_COMPLETE_NO_AUTOMATIC_FOLLOWUP。公開・remote照合後mandatory STOP。P1のSTOP_UNRESOLVED_COMPONENTSを保持。追加科学計算、詳細科学レビュー、主要claim採択、論文改訂へ進まない。次の担当はユーザー/GPT：この限定結果の扱いと、必要ならP1N/P3等の具体的scopeを判断。Codexの追加科学実行作業なし。**
