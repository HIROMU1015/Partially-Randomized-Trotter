# H6 DF19 RPE D1：係数prior・保存4候補・部分費用の結果

2026-11-10 JST。**承認された限定campaignを一回完了した。係数priorからh=.1を採用し、保存4候補すべてでideal条件付き十分Nを算定した。** T=.8/1.6の保存2roundの部分費用までを評価した。actual実装・真ground同定・全RPE成功・全RPE費用は未資格。[result.json](result.json)が数値の正本、[実行出力](execution.txt)は正常終了の記録。

科学実行SOURCEは **`4c928b905c65d73de23caf82a113a5fb4826e48d`**。[固定実行計画][PLAN]と同じSOURCE・条件・4候補で実行し、SOURCE変更や科学的retryは0。H4 D1の係数prior・scalar幾何を再利用し、H4のデータや進行中の価格調査には介入していない。

## 科学条件とprior

Linear H6、隣接1.00Å、STO-3G、保存tol-only 1e−8／DF19、12 system qubit、Nα=Nβ=3のsector400。targetは保存weighted-Hermitian completion後のDF-Hの同sector ground energy。signed weights、fragment順、scalar、H/state/snapshot bindingを保持した。精度は既存CA=0.00159360010199040 Ha（1 kcal/mol）、各方式αtotal=.05、保存数値state、P=0。元分子Hとの差をenergy reserveに加算しない。

保存表現 `H=c+dΓ(h)+Σλ_l dΓ(G_l)^2` を使った。Hermitian係数行列のGershgorin区間を6粒子へ拡張し、signed λとsquare区間を正しく加算した。上界には保存sectorの先頭occupation `[0,1,2,3,4,5]` の固定Slater determinantに対する変分式を使った。prior関数の入力に保存E0はなく、determinantをE0に合わせて選んでいない。

係数binary64を正確な有理数として演算し、採用ground区間のfloat端点を外向き丸めで保存した。元の有理数端点との包含を結果から照合した。この包含根拠は**保存Hermitian DFモデル**に限られ、元分子H、completion前のH、保存stateの厳密ground同定へ拡張しない。full Fock Hamiltonianやtrial state vectorは作っていない。

| prior・RPE設定 | 結果 |
|---|---|
| ground energy prior [Ha] | **[−9.695399206821989, −3.1355322139814668]** |
| h=.8、無shift alias余裕 | `π−h max(abs(Emin),abs(Emax))−.4 = −5.014726711867799 rad`：不成立 |
| h=.1、同余裕 | `1.7720527329075941 rad`：成立、**h=.1を採用** |
| βtotal / βstat | .4 / .2 rad。最適化・事後変更なし |
| 既存horizon関数 | **M=12、13round m0..12**、qmax4096、Tmax409.6 a.u. |
| nominal energy resolution | .0009765625 Ha、直前powerは.001953125 Ha。解像度条件の判定であり全round成功保証ではない |
| 各round・各軸α | .05/26 = .0019230769230769232、全26軸の和=.05 |
| 今回の保存時間 | T=.8→m3/q_RPE8、T1.6→m4/q_RPE16。q_PFは各候補1で別変数 |

h=.8の不成立はこの保守的prior・無shift principal decoderに対する判定である。priorを保存E0で狭めたり、hやβの候補を自動拡張したりしていない。13roundは旧13/18roundの流用ではなく、今回のh/β/CAから求めた。全round分のαを予約しても、未評価roundの成功資格は与えない。

## phase・normalization・誤差disk・十分N

保存solver energyは−3.2360662799087647 Ha、residualは6.1060388500553216e−15 Ha。referenceと`exp(−i E_state T)`の整合性を診断し、T×residualを数値guardとして加えた。これは保存数値stateをgroundと同定する条件付きの評価であり、厳密groundの証明ではない。E_stateはpriorやdecoderのbranch選択には使っていない。

H4 D1と同じ式で、corrected diskを

`hypot(|bias_X|+u_X, |bias_Y|+u_Y) + |zref−exp(−i E_state T)| + T×residual`

とした。元endpointの軸uは4候補とも1e−9。A=exp(−log B)、raw≈A×correctedを照合し、closureとlog_B_margin由来のraw余裕を加えた。βdet=asin(δraw/A)、ρlower=A−δraw、εaxis=ρlower sin(.2)/√2を既存geometry/Hoeffdingへ接続した。経験的u・reference差・数値residualをrigorousな対真ground誤差上界とは扱わない。

| arm / T | 構成（全てm5_best/q_PF1） | log B / A | corrected disk | raw半径下界 | βdet rad | 十分N_X / N_Y |
|---|---|---|---:|---:|---:|---:|
| PF / .8 | Full19、`76601611270ed9f0677d` | 0 / 1 | 9.0593671e−7 | .999999094057 | 9.0594298e−7 | **705 / 705** |
| PR / .8 | prefix5/R32/K2、`2606efdab20e3b8522e5` | .04138848360 / .959456324494 | 9.0318804e−7 | .959455457923 | 9.0319001e−7 | **765 / 765** |
| PF / 1.6 | Full19、`a4f5ffc0f320644dadfa` | 0 / 1 | 2.5494465e−5 | .999974505529 | 2.5494471e−5 | **705 / 705** |
| PR / 1.6 | prefix7/R16/K2、`2d2af6ca830dff14a695` | .01138443249 / .988680124942 | 2.5494429e−5 | .988654919104 | 2.5494432e−5 | **721 / 721** |

4候補とも正半径、βdet+βstat≤.4で`CONDITIONAL_ROUND_PASS`。十分Nはこの配分・数値pilotでの計画値であり最小Nではない。量子shotやRTE trajectoryは生成していない。PRはT=.8と1.6でprefix/Rが異なり、同一固定回路の単純時間倍率ではない。

referenceの符号付き位相差はT=.8で1.8614510e−14 rad、T1.6で−1.4684314e−14 rad。raw/corrected closureはPF0、PRは約3.69e−15／2.84e−15だった。観測reference半径1.0000000000000004等は丸めを含む保存値で、物理的な半径超過を主張しない。全詳細は各rowのreference_diagnostic／normalization／error_diskに保存した。

**actual laneは4候補すべてUNRESOLVED_REFERENCE_OR_ERROR、N_X/N_Yはnull。** 未取得のimplementation/law距離、真ground同定、経験的allowanceの厳密包含を0で補完していない。ideal条件の誤差0は、fresh IID ±1測定・exact nativeを仮定したモデル内の値であり、actualの未知誤差ではない。

## 保存直接価格と2round部分費用

同じcandidate・H/state・T・formula/prefix/R/K・compiler・測定付きfull Hadamard wrapperの保存価格を使った。Qiskit1.3.0／rz,sx,x,cx／opt1／seed17、元symmetric-directional control、P=0。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`は各rowのideal_numerical_pilot.bindingに保持し、原価格出典はpartial_resourcesのprice_originsへ辿れる。新compile、model価格、fitは使っていない。

| arm / cohort | T=.8 C_RZ,X=C_RZ,Y | T1.6 C_RZ,X=C_RZ,Y | nと資格 |
|---|---:|---:|---|
| PF deterministic | 427,996 | 428,918 | 直接決定論価格。二重に独立標本として数えない |
| PR EXPLORATION | 141,775.11111111112 | 179,583.33333333334 | 各候補n9の保存paired平均 |
| PR CONFIRMATION | 141,173.21212121213 | 179,579.36363636365 | 各候補独立n33の保存paired平均。探索とpoolしない |

`G_partial=Σ_(m∈{3,4},a) N_m,a C_m,a`。表示は小数3桁、原値はresult.jsonに保持する。nやnormalization Bを再乗算していない。

| 資源 | PF deterministic | PR EXPLORATION | PR CONFIRMATION |
|---|---:|---:|---:|
| **RZ count** | **1,208,248,740.000** | **475,875,086.667** | **474,948,456.909** |
| CX count | 530,520,960.000 | 202,087,246.222 | 201,822,017.455 |
| RZ depth | 230,492,700.000 | 90,583,070.444 | 90,438,831.818 |
| CX depth | 240,950,670.000 | 90,646,333.333 | 90,574,002.848 |
| Total depth | 594,800,040.000 | 229,280,036.222 | 228,971,329.758 |
| Circuit size | 2,528,654,520.000 | 980,268,844.667 | 978,614,465.212 |

これは**保存2roundの条件付き部分費用**。PRの有限cohort平均は全law期待価格の保証ではなく、未観測event tail・母平均に対するbias・全体covariance/CIは未資格。total_confidence_interval、total_covariance、peak_qubitsは保存coverage不足によりnull、actual_totalsもnull。未知を0補完していない。depth和は実機wall timeではない。全RPE費用・PF/PRの最終優劣は判定しない。

## H4との共通点、未評価roundと次段階

[H4 D1][H4]と共通なのは、係数prior、数値reference/disk・normalizationの扱い、既存geometry/Hoeffding/horizon、β=.4/.2、今回導いたh=.1／M12／α=.05/26である。H6 priorは独自係数から求めた。H4のstate/sector/candidate/N/価格をH6へ移植せず、H4の完了結果も再計算していない。H6では異なるPR構成のT=.8/1.6まで評価し、部分会計を行った点が異なる。

未評価roundは**m0,1,2,5,6,7,8,9,10,11,12**、時間は`.1/.2/.4/3.2/6.4/12.8/25.6/51.2/102.4/204.8/409.6`。保存T1.2/2.0をこのdyadic列へ挿入したり、信号を補間・冪乗して埋めたりしない。

最小D2/D3案（未実行・未承認）：

1. **ユーザー／研究方針担当**が、今回のh/β/α・数値state仮定を引き継ぐD2の有限candidate menuと追加signal範囲を決める。今回の2点を再計算せず、未評価roundだけを対象にする。長時間構成の成立を安価に切り分けるなら、まずT3.2のPF/PR少数構成を同じD2 scope内の判別点とする。全11roundの構成がそのまま適格とは仮定しない。
2. **Codex**は別途許可されたD2範囲だけで、既存保存input/stateとfinite-mean/action pipelineを再利用し、必要なreference・bias・B・diskを評価する。新state/H生成や広いq/R/K最適化を既定にしない。全roundのN・phase/branch/failure条件が閉じるまで全energy成功とはしない。
3. **価格担当／Codex**はH4側の価格会計・model方法の成果を参照し、同一条件のH6直接価格/cacheを優先する。今回の4点Cは再compile不要。未取得T/δ/q_PF/R/K/prefixのCはH6 domainの追加資格が必要で、CDF8/DF12/H4の係数を使わない。native計数／小さい校正・holdoutで足りる範囲を先に検討し、長回路の全compileを既定にしない。
4. **ユーザー**がD3の価格資格と総会計scopeを許可した後、全roundに揃ったN/Cで初めて全Gを比較する。現在はD2/D3、追加signal、価格fit、長q評価、詳細科学レビューへ進まない。

## 実行・技術確認・保全・STOP

計算は保存scalarと12×12の係数からの有理数演算が中心なので、一つの既存project Python processで実施した。wall約5.934秒（importを含む）。CPU/RAM/workerへの独自cap、依存install、共有設定変更、他process操作は0。prior1回、horizon1回、保存point4件、部分費用3表。state vectorロード、eigensolve、full-H/trial vector構築、新signal、sampling、build/compile、fit、全RPE simulationは0。

科学SOURCEの8ファイルhashと保存input binding、prior外向き端点、β条件、全6部分資源のΣNC恒等式、included/missing round、actual未知量の保持を照合した。結果を再生成せず、SOURCE/input/旧projection/過去結果を保持した。既存[人工12件][TESTS]を再利用し、SOURCE変更がないため無関係なtest再実行を増やしていない。本担当自身の技術確認であり、独立科学レビューではない。

結果公開commitは科学実行SOURCEと区別する。公開remote SHA・主要成果物の全文bytes照合後**mandatory STOP**。次の担当は**ユーザー**：上記D2の有限範囲を判断する。**Codex**：追加の実行許可までD2/D3等の科学計算を開始しない。

[PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c928b905c65d73de23caf82a113a5fb4826e48d/docs/research/h6_df19_rpe_prior_execution_preparation_20261110/REPORT.md
[H4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ffe8a71da98bec330ce1af109a9606f14e1fc598/docs/research/h4_df10_rpe_d1_20261110/REPORT.md
[TESTS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c928b905c65d73de23caf82a113a5fb4826e48d/docs/research/h6_df19_rpe_prior_execution_preparation_20261110/REPORT.md
