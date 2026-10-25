# 研究B：Outer Splitting固定22行の科学実行結果

2026-10-25 JST。**実行状態：COMPLETE。入力・全correctness・22行比較・登録scopeを完了し、mandatory STOP。εへの保証付き適格性は全行で未確定。**

## 1. SOURCE・実行・原本保全

科学SOURCE **`26e8734aff3969dc8077cb058826d12394be28e1`** を変更せず、[固定22行契約・command](research_b_outer_rte_22row_execution_ready_20261025.md)に従って一回実行した。実行直前にHEAD・clean SOURCE・fixture参照・依存availability／version・scope・fresh outputの非存在を再確認した。科学計算は正常完了し、CLI exit0。中断・修正・retry／resumeは0件。

| 対象 | 保存先 |
|---|---|
| 実行原本 | `/tmp/prt-research-b-outer-rte22-science-20261025/artifacts/research_b_outer_rte22/20261025/analytic-once` |
| 公開原本 | [analytic-once](../../artifacts/research_b_outer_rte22/20261025/analytic-once/)：既存9ファイルを変更せずコピー |
| 閲覧用全22行 | [COMPARISON.csv](../../artifacts/research_b_outer_rte22/20261025/COMPARISON.csv)：保存JSONの62列を直接抽出 |
| process／環境実績 | [execution_logs/attempt-1](../../artifacts/research_b_outer_rte22/20261025/execution_logs/attempt-1/)：stdout・stderr・time・実行直前metadata |

公開branchは `codex/research-b-outer-rte22-results-20261025`。科学SOURCEと結果公開commitを区別する。Raw9ファイル・process記録4ファイルは原本とのbyte一致を確認した。CSVは値の列抽出だけで、operator／signalの再評価、追加grid、fit、bound再計算を行っていない。以下の表も今回の同一実行内の保存値による記述である。

旧16行・36行のSOURCE／結果／契約／grant／ledger／marker／STOPは変更していない。新しい承認証明・grant／packet／receiptは作成していない。

## 2. 固定した科学範囲

Synthetic 2-mode、one-body＋3 DF因子、N=1／mode0 occupation、L_D=2 weight-ranked prefix、T=±0.18、D内部n_D=1。分子geometry／AO basis／分子DF rank探索は非該当。
内部A2は既存Yoshida4、A3はpublished Morales Y8m10b。Complex128、correctness atol=1e-12、phase診断無効、独立RTE occurrencesのanalytic moments、samplingなし。Scalar／identity分離・event phase・signed durationを維持した。

各TでB1=S2/A2/q2、B2=refined-S2/A2/q2/f2、B3=Yoshida4/A2/q2、B4=SABA2/A2/q2、D1=Yoshida4/A3/q2、R1/R2=Yoshida4/A2/A3/q2/r4K2、R3/R4=Yoshida4/A2/A3/q2/r2K4、S1=SABA2/A2/q4、F1=refined-S2/A2/q2/f8。指定のないr/Kは2/2。計11条件×両T=22行で、F1は16実step。q2のmacro幅は±0.09、q4は±0.045。他の条件は実行していない。

## 3. 実績とcorrectness

| 登録境界 | 実績 |
|---|---:|
| 比較行 | 22／22 |
| DAG dispatch | 1,239／1,239 |
| Unique exponential key／backend.expm | 274／274 |
| RTE config／moments／truncation record | 各96／96 |
| Bound composition | 14／14 |
| Signal expectation | 110／110 |
| Input package高位境界 | 10／10 |
| D数学／outer数学順／signed RTE gate | 36／12／96 unique node、全PASS |
| Raw／corrected rescaling gate | 22、全PASS |
| Input／weak診断／tail norm | 各1、完了 |
| Event数 | 3,876（登録枠3,902以内） |
| 失敗・未完了・保存失敗 | 0 |

Attempted／completedは全登録境界に一致し、failedは空。Eventsから**全169 unique correctness nodeの完了が最初のoperator／signal比較に先行した**ことを確認した。全22行がresultとpartial_resultsに同じ値で保存されている。親子・入れ子境界を加算した値を独立科学call総数とは呼ばない。

実入力はDにDF因子0,1、Rに因子2を選択し、sector basis `[1,2]`、mode0を使用した。H identity誤差 `6.16645e-17`、D-pair commutator normは `0.023895 / 0.00392 / 0.01607445`、D/R commutator normは `0.00428043`。必要な非可換性を確認した。Tail sector leakageは0。

入力array identityは `54a9233d59f670c7b8d38c186c9543925b6ca70bb09357c0176f1daf98fa819a`。全H／sector／state／preparation／tail identity・fixture literal・入力arrayは公開inputsに保持する。Hのidentity分離とstate norm／Hermiticity gateはPASSした。

D列の独立数学参照残差の最大は `5.55382e-17`、outer数学順は `1.22456e-15`、full rescalingは `7.80888e-16`、すべて固定atol内。RTE configは各stageのtail・signed time・r・K・tolerance・seed・lawと一致し、N_j／attenuationも整合した。140個の物理R occurrence参照を96 momentへ共有しても、normalizationとboundの重複因子は保持した。

弱摂動診断は `||D||=1.08120585`、`||R||=0.0830399525`、比 `0.0768030923`、`||[D,[D,R]]||=0.00770743512`、`||[R,[R,D]]||=0.000356722068`。既定の記述診断であり、弱摂動領域の認定閾値やSABA2一般四次の保証へ格上げしない。

## 4. Operator／coherent signal比較

以下は**T=+0.18**の保存値。bはphysical複素coherent signalへの直接絶対誤差。両符号22行の全精度値・raw X/Y／半径・5対象差はCSVと原本を参照する。T=−0.18でもbとNはこの表の値と一致し、operator差等の末尾に浮動小数点の差がある。

| ID | 条件の変更点 | Corrected physical operator誤差 | b | Finite-RTE signal差 | 全体N |
|---|---|---:|---:|---:|---:|
| B1 | S2/A2/q2 | 5.09754e-7 | 2.26738e-7 | 3.04108e-11 | 1.000055855727 |
| B2 | Refined-S2/A2/f2 | 1.27309e-7 | 5.66296e-8 | 3.80079e-12 | 1.000027927648 |
| B3 | Yoshida4/A2/r2K2 | 1.75192e-9 | 6.75687e-10 | 4.58175e-10 | 1.000365893033 |
| B4 | SABA2/A2/q2 | 5.96822e-9 | 2.90506e-9 | 3.80092e-12 | 1.000027927648 |
| D1 | Yoshida4/A3/r2K2 | 5.28691e-10 | 4.52693e-10 | 4.58176e-10 | 1.000365893033 |
| R1 | Yoshida4/A2/r4K2 | 1.51896e-9 | 5.84380e-10 | 5.72714e-11 | 1.000182932417 |
| R2 | Yoshida4/A3/r4K2 | 1.77540e-10 | 8.33775e-11 | 5.72715e-11 | 1.000182932417 |
| R3 | Yoshida4/A2/r2K4 | 1.48821e-9 | 5.92676e-10 | 2.37144e-16 | 1.000365893522 |
| R4 | Yoshida4/A3/r2K4 | 1.43716e-10 | 7.00360e-11 | 2.28878e-16 | 1.000365893522 |
| S1 | SABA2/A2/q4 | 1.45669e-9 | 7.13218e-10 | 4.77313e-13 | 1.000013963748 |
| F1 | Refined-S2/A2/f8 | 7.95421e-9 | 3.53826e-9 | 6.22221e-14 | 1.000006981853 |

**A3対照ではRTEの変更がfull corrected signalへ残った。** D1→R2→R4のbは4.53e-10→8.34e-11→7.00e-11。A3のideal-tail physical signal差は約7.0036e-11、D近似のsignal差は約1.57e-14で、K4は保存値上ideal-tailへ近づいた。ただしK4のfinite-RTE差自体は丸め誤差水準で、真の残差をcomplex128で精密識別したとはしない。

A2対照のideal-tail signal差は約5.92676e-10、D近似signal差は約5.22697e-10であり、RTEだけを高精度化してもこの寄与が残る。R1のbがR3より小さいことからも、成分差が小さくなればfull複素誤差が単調に下がるとはいえない。複素誤差の相殺を含み得る結果であり、位相診断は実施していない。別条件へ一般化しない。

SABA2のq4とrefined-S2のf8も、それぞれ固定対照B4/B2より観測bが小さかった。これは同一T・同一fixtureでstep数も変わる比較であり、費用を揃えた優位性や収束次数の認定ではない。

## 5. Normalization・rawと構造費用

Yoshida4のr2K2→r4K2ではNが約1.000365893→1.000182932へ下がる一方、RTE short-step数が12→24へ増える。K2→K4ではNが約1.000365893033→1.000365893522へ僅かに増え、有限Taylor精度の改善とnormalizationの変化を別に扱う。

Raw physical signal誤差はA2のB3で約3.64737e-4、R1で約1.82388e-4、R3で約3.64737e-4。A3のD1/R2/R4も同じ桁であり、correctedの約1e-10～1e-11という値をraw精度やshot精度と混同しない。S1は約1.39246e-5、F1は約6.96283e-6。全stageのN_j、全体N、attenuation、raw X/Y・半径は原本に保存した。

| ID | D runs | D kernel反復 | Literal D block指数数 | R occurrence | RTE short-step数 |
|---|---:|---:|---:|---:|---:|
| B1 | 3 | 4 | 60 | 2 | 4 |
| B2 | 5 | 8 | 120 | 4 | 8 |
| B3 | 11 | 12 | 180 | 6 | 12 |
| B4 | 5 | 6 | 90 | 4 | 8 |
| D1 | 11 | 12 | 1260 | 6 | 12 |
| R1 | 11 | 12 | 180 | 6 | 24 |
| R2 | 11 | 12 | 1260 | 6 | 24 |
| R3 | 11 | 12 | 180 | 6 | 12 |
| R4 | 11 | 12 | 1260 | 6 | 12 |
| S1 | 9 | 12 | 180 | 8 | 16 |
| F1 | 17 | 32 | 480 | 16 | 32 |

各Tで同じ構造。これはDAG／planの構造値で、実native gate数・期待wrapper費用ではない。K4はevent support・次数分布も変わるため、short-step数が同じでも同費用とは限らない。Native wrapper、γ、shots、N加重費用を評価しておらず、総資源の最小候補は確定しない。

## 6. Taylor boundと数値的な限界

Boundは既存関数を固定scope内で評価し、全occurrenceを合成した。理論上の `||H̄_R||≤1` とunitary D因子等を前提とする**nominal analytic bound**で、浮動小数点誤差を含まない。Tail normの数値値は `1.0000000000000002` で、固定atol内でPASSしたが、厳密な包含証明ではない。

| 例（T=+0.18） | 保存finite-RTE operator差 | Nominal Taylor operator bound |
|---|---:|---:|
| B3/A2/r2K2 | 4.89611e-10 | 4.90186e-10 |
| R1/A2/r4K2 | 6.12010e-11 | 6.12378e-11 |
| R3/A2/r2K4 | 3.70749e-16 | 5.52810e-16 |
| R4/A3/r2K4 | 3.58911e-16 | 5.52810e-16 |
| S1/SABA2 q4 | 5.09846e-13 | 5.07865e-13 |
| F1/refined f8 | 6.62042e-14 | 6.34772e-14 |

**S1/F1では浮動評価した差がnominal boundを僅かに上回る。** この差を隠したり、atolを加えて厳密なbound PASSとしたりしない。今回のcomplex128評価から、数値的不確かさを含むbound包含は確定できない。K4の約1e-16残差の順位・正確な大きさもINCONCLUSIVE。追加高精度計算・再実行は行わない。

## 7. ε予算と未確定の適格性

契約どおりε=1e-8／1e-9、deterministic側は `b+u_num≤ε/2`、実装γはε/10、統計sは2ε/5を使用する。Energy／RPE位相精度ではない。

保存された**bだけの観測値**では、両Tともε/2=5e-9以下なのはB3/B4/D1/R1/R2/R3/R4/S1/F1、5e-10以下なのはD1/R2/R4。B1/B2は両方を上回った。このbooleanは予算との記述比較であり、候補の適格認定ではない。

全22行で `u_num=null`、両εの `deterministic_qualified=null`、statusは `UNRESOLVED_NUMERICAL_UNCERTAINTY`。Atol=1e-12をu_numとみなさず、bが小さい行も保証付き適格としない。**実行COMPLETEと、同精度保証の未確定を区別する。**

95% confidence、α=.05、X/Yへ各.025という将来の共通配分はmetadataのみ。Familywise budgetは未配分、γ・s・必要shot数も未取得。End-to-end statusは `NOT_CERTIFIED_NO_TOTAL_RESOURCE_CLAIM` のまま。実shot simulation・finite-bit certificate・総資源比較は実施していない。

## 8. 環境・資源・技術実績

既存 `/tmp/pr-phase2df1pre-private-20261023/venv/bin/python`、CPython3.11.9。NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Qiskit Nature0.7.2、OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0、Numba0.60.0、Matplotlib3.9.2。依存・実行環境・全versionはbindingに保存した。共有環境を変更していない。

実行直前32 CPU、空きRAM約52.7 GiBを確認し、既存2×2 sectorのCPU vectorized演算、274 exponential cacheと96 moment共有を使用した。小行列ごとのprocess分割は追加せず、CPU／RAM／worker／thread／wall-timeへ人工的な低い上限を設定していない。GPU確認はdriverと通信できなかったが、固定CPU backendに影響しない。

Whole process wall **10.33秒**、user CPU15.74秒、system CPU3.07秒、最大RSS371,844 KiB、exit0。科学速度倍率は比較測定していない。StderrにはMatplotlibが書込不可の既定cacheから一時cacheへ移したwarningだけがあり、計算・保存の中断はなかった。環境修正・SOURCE修正・再実行は不要だった。

Input package内の既存Gaussian基底準備・tail／sector変換は登録入力境界内で実行した。Native full PF wrapperのbuild／compile、追加PF係数探索、別の物理条件、高精度化、shot評価、Stage 2は行っていない。準備時の人工119件PASSは技術的証拠として区別し、SOURCE不変のため同じsuiteの不要な再実行はしていない。

## 9. 次の研究判断とSTOP

確認できたのは、この固定fixture・両TにおけるRTE高精度化のbias／normalization／構造への影響と、A2/A3対照でのD近似の寄与である。R4の観測bが小さいことだけでは、A3の長いD列やK4のevent費用を含む総資源優位は判断できない。

次の担当は研究Bのユーザー。まず、u_numとnominal boundの数値的不確かさをどの水準で扱うか、またr4K2とr2K4等のnative費用・γ・shotsへ進む価値があるかを判断する。これらは**提案であり未承認**。Codexは本結果を公開・remote照合した後に停止し、追加解析・再実行・別grid・費用／shot段階・Stage 2を開始しない。
