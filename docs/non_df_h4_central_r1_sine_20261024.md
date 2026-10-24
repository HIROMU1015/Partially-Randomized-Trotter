# H4 Pauli central R1 sine：PF・PR0・PR1経験的診断

PF・PR0・PR1のsine **3論理診断を完了、状態作用は成功6／総試行6**。
新C-target作用・cosine再計算・compile・sampling・Arbは0。
全対象でdは保存sine予算より小さい経験値だった。
**EMPIRICAL_ONLY_NO_RHO_UPPER**、`rho_upper=null`、`numerical_error_upper=null`、`LOCAL_BUDGET_MET=false`を維持する。
今回のnative exit0・reapとmandatory STOPを保存した。

## 1. 承認・固定条件・SOURCE

ユーザーの「次の作業に進んで大丈夫」と、続く「sine経験的診断（推奨）：PF・PR0・PR1、3診断・最大6状態作用」の選択回答を
[approval.txt](../artifacts/non_df_h4_central_r1_sine/2026-10-24/approval.txt)へ引用保存し、grantへ結合した。
Linear H4／隣接1.00 Å／STO-3G、system8＋ancilla wire8、Pauli-JW非identity184項／PR prefix27。
S2、T=.8、q8、r4、K4、元PR event law・保存trajectory・raw ray・νを維持した。
Qiskit1.3.0、basis rz/sx/x/cx、optimization1、seed17、backend/coupling/layout/routing指定なしの保存QPYを使用した。
DFやH2/S2/Yoshida4解析には統合していない。

| 来歴 | commit |
|---|---|
| 基点 | `c7585f2d6e4a0fa3f0cc873fb6e468be1d42bce9` |
| **今回の科学SOURCE（作用前commit）** | **`198d4916f7f62b775d9f60020c42ffb07d4c1c89`** |
| 元H4 | `0c329b30a81accb35c212be8defa66fee0dd0f12` |
| C | `e80d7510880f8db993b777c627bb8e95b05a070d` |
| D2 | `74515d8eb4f073c40fc783681a3775dbca89e9b1` |
| A_GATE | `ee7d281b5daead1d0d5e81e00c9a0ed4727792b0` |
| A_AXES | `a37232f85812688c3b9ab957c5073c5ce143b17e` |
| R0 | `84a326598b006384d13c7a354a3eec6881ee841c` |
| R1 order2 | `ca9a1c0e2426565ba63cc8bfd0247f73339d95b8` |

専用branchは`codex/non-df-h4-central-r1-sine-20261024`。
基点・A_AXES・A_GATE・R0・R1の5commitと、sine metadata／cacheの4blobをGitHubから実取得・照合した。
[GitHub記録](../artifacts/non_df_h4_central_r1_sine/2026-10-24/github_verification.json)。過去の公開失敗履歴は変更しない。

| 対象 | 固定trajectory ID | sine保存元 |
|---|---|---|
| PF | `d761627d56a4454e644358fef8907b8a7db28c4feeb44130224c20e4e50aae4d` | A_AXES PF |
| PR0 | `93ef5e4e28b499472c8ba9329f6cb139f63c5de071279783b3cca3319793fc97` | A_AXES PR0000 |
| PR1 | `596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa` | A_GATE sine |

## 2. 実QPY・測定軸・C-targetの対応

source body／controlled body／保存cosine inputの構造／sine input／sine compiledの計15 QPYを本番前と本番内で実decodeした。
cosine QPYはbody/SOURCEの照合だけであり、cosine状態作用を実行していない。
SOURCE recipe・event ID/hash・Pauli符号と順序・保存aggregate phase・control0/1のtoken対応・fingerprint・compiler profileを照合した。
sine inputは保存cosine wrapperの**最終H直前へSdgを1個追加した列と厳密に対応**する。
wire8→clbit0、最終measurement、未解決parameter・中間measure・reset・condition・layout・未対応gateの不在を確認した。
規約は`+Im〈ψ|Uψ〉/ν`。新wrapper生成・新control生成は0。

PF/PR0の保存body aggregate phaseは`.2651822507334482`、PR1は`.26518225073344787`を維持した。
PR1のposition30／step7／inner2のorder2はproduct43(+1)→4(−1)→rotation72(−1)、angle `.029672025283184623`、order phase−1。
PR1の位相をPR0の近い値へ置換せず、負product・order phaseを重複加算していない。
[実QPY照合](../artifacts/non_df_h4_central_r1_sine/2026-10-24/run01/preflight.json)の本番前後記録は同一。
**今回の限定構造検査と信号評価で、位相・scale・control・sine符号の不一致は検出していない。** 一般compiler certificateではない。

元state hashは`8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`。
R0保存exact νを再利用し、binary64表示は`1.0000000000000002`。
R0/R1の同一state hash／ν、同一Python hashとruntime manifest SHA、C数値SOURCE不変、個別trajectory IDを照合した。
PR0/PR1の`complex_signal.imag`をそのまま個別C-targetとして引用し、law meanを代用していない。
PFは保存`implemented_law.PF.corrected_encoded_mean.imag`区間を同一exact νでscalar変換し、区間内exact midpointを代表値とした。
保存C norm区間のν包含も確認した。
これらは保存値の再利用・scalar変換であり、**新しいC-target状態作用は0**。
[cache対応](../artifacts/non_df_h4_central_r1_sine/2026-10-24/cache_binding.json)、
[PF再利用記録](../artifacts/non_df_h4_central_r1_sine/2026-10-24/run01/PF_C_target_reused.json)、
[PR0再利用](../artifacts/non_df_h4_central_r1_sine/2026-10-24/run01/PR0_C_target_reused.json)、
[PR1再利用](../artifacts/non_df_h4_central_r1_sine/2026-10-24/run01/PR1_C_target_reused.json)。

## 3. 三信号と経験的残差

最終measurementだけを除いたQPYをR0と同じ`Statevector.evolve`で作用させ、
振幅からunclipped ancilla sectorの`m=(p0−p1)/ν`を縮約した。
raw rayも出力も再正規化せず、Probabilities APIのclippingや出力normによる除算を使わない。
δ=`B_saved abs(m_out−m_in)`、η=`B_saved abs(m_in−mC)`、d=`B_saved abs(m_out−mC)`。
B_savedはPF=1、PR=`1.2873709223899619`を一度だけ使用し、B_upを代用しない。

| sine保存値 | PF | PR0 | PR1 |
|---|---:|---:|---:|
| mC | .9868700902957119 | .9003076520475742 | −.02372933466577869 |
| m_in | .9868700902930314 | .900307652047544 | −.023729334665777917 |
| m_out | .9868700902955961 | .9003076520475544 | −.023729334665777972 |
| δ：compiler-only | 2.564726209186574e-12 | 1.3292200205213428e-14 | 7.146344196351306e-17 |
| η：C→input | 2.6804985924207587e-12 | 3.88761124281511e-14 | 9.960217223664632e-16 |
| **d：C→compiled** | **1.1577238323418473e-13** | **2.5583912222937673e-14** | **9.245582804029502e-16** |
| 保存sine ρ予算（ε_sig=.0005） | 3.3613942524486874e-6 | 3.3523233722408497e-6 | 3.3523233722408497e-6 |
| d／保存ρ予算 | 3.4441774614759214e-8 | 7.631695806790915e-9 | 2.7579626955407117e-10 |
| 保存ρ予算−d | 3.361394136676304e-6 | 3.3523233466569374e-6 | 3.3523233713162913e-6 |
| 経験的大小 | BELOW_SAVED_BUDGET | BELOW_SAVED_BUDGET | BELOW_SAVED_BUDGET |

比較は表示の丸め値ではなく、保存sine guardのexact値PF `25651956654414547227/7631344236317346665332736`、
PR `102330934580088554157/30525376945269386661330944`を使用した。
[引用元guard](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/guard_inputs_quoted.json)の予算は変更していない。
`d≤δ+η`は保存点値のexact rational恒等式で成立する。
PF mCの表示floatは表示用で、差の会計は保存区間内exact midpointを使用する。
[全exact結果](../artifacts/non_df_h4_central_r1_sine/2026-10-24/run01/results.json)、
[両軸転記CSV](../artifacts/non_df_h4_central_r1_sine/2026-10-24/results.csv)。

### Cosineとの限定比較

| 局所対象 | cosineのd（保存引用） | sineのd（今回） |
|---|---:|---:|
| PF | 1.8722085654184194e-14 | 1.1577238323418473e-13 |
| PR0 | 7.860978615986437e-16 | 2.5583912222937673e-14 |
| PR1 | 4.3771358202651746e-16 | 9.245582804029502e-16 |

選択した3対象では両軸とも引用した各軸予算より小さい経験値。
sineの表示dは各cosineより大きいが、数値縮約・信号値・compiler programが異なり、軸固有誤差の一般則とは扱わない。
δ/ηに打消しがあり、δの小ささだけでC-target→compiledの誤差を判断しない。
cosineは再作用せず、[axes_comparison.json](../artifacts/non_df_h4_central_r1_sine/2026-10-24/run01/axes_comparison.json)へ保存値を転記しただけである。
PR0/PR1差をorder2単独の因果効果とせず、PR全law・未観測order4へ外挿しない。

## 4. 数値保証の限界・未解決状態

全結果は**EMPIRICAL_ONLY_NO_RHO_UPPER**、numerical_error_upper/rho_upper/PR_law_rhoはnull、LOCAL_BUDGET_MET=false。
PF保存C区間はcompiled/input伝播誤差の包含証拠ではなく、PR複素cacheもbinary64点値である。
三角関数・matvec・Statevector伝播・sector縮約・ν変換の丸めは保証していない。
とくにPR1の小さい残差はbinary64丸めの影響を受け得る桁であり、真の非零残差や確定した有効桁とはしない。
exact scalar照合は保存点値の整合性を検証するだけで、真値包含へ格上げしない。

作用後normはPF input/output `.9999999999972838`／`.9999999999998842`、
PR0 `.9999999999999707`／`.9999999999999827`、PR1 `.999999999999972`／`.9999999999999821`。
これらで再正規化していない。
元hash FAIL/原因UNKNOWN、元H4 native UNKNOWN、旧N/G null・UNDETERMINED、C/L COMBINATION_BLOCKED、
旧compiler MISSING_EVIDENCE、PR全lawρ UNKNOWN、母平均費用tail UNKNOWN、正式certificate未受理を保持する。
今回の両軸6局所結果をPR256本の残差平均・全event上界・母平均費用保証・条件付きN/Gの正式保証へ読み替えない。

## 5. 再利用・人工検証・実行監査・保全

[adapter](../src/trotterlib/h4_r1_sine.py)はR0のpropagate／saved_PF_target／compare_signals、R1のevent意味論検査を再利用した。
新規部分はsine wrapperの読取照合、保存C虚部cache接続、SOURCE別metadata結合、3件6作用dispatchのみ。
[runner](../scripts/run_h4_r1_sine.py)はR0のverify/prepare/check/worker/launchを再利用し、sine・3論理・6作用・禁止call・namespaceを明示設定した。
新simulator、Hamiltonian builder、compiler、shot会計は作っていない。
新[人工9件](../tests/test_h4_r1_sine.py)と既存R0 10件、合計**19 PASS**。
Sdg/+Im、誤ったS・wire・phaseの拒否、raw ray/clipping、PF保存imag区間、個別PR虚部cache、metadata schema差、非返還6作用上限を検査した。
主担当Codex自身の検査であり、独立別reviewerの査読ではない。

SOURCE前の初回事前検査は、A_AXES metricsのtrajectoryが`target`内にあるのに直下と仮定して停止した。
実schemaを確認し、A_AXES nested identity/caseとA_GATE flat identityをSOURCE別に厳密照合するよう修正した。
[初回log](../artifacts/non_df_h4_central_r1_sine/2026-10-24/structural_inspect01.log)、
[回復記録](../artifacts/non_df_h4_central_r1_sine/2026-10-24/pre_SOURCE_recovery.json)、
[修正後log](../artifacts/non_df_h4_central_r1_sine/2026-10-24/structural_inspect02.log)、
[最終9人工log](../artifacts/non_df_h4_central_r1_sine/2026-10-24/artificial9_final_PASS.log)を保持する。
当時は科学grant0・状態作用0。科学SOURCE固定後の失敗・code修正・retry・回復実行は0。

科学SOURCE commit後にcode23／runtime6999／入力109／旧証拠309／承認・schemaをsealした。
新grant IDは`76870f554265ace43dd1543c6efa0114fb2bd7bf88a8fc2cba803a38314b954e`。
[seal](../artifacts/non_df_h4_central_r1_sine/2026-10-24/seal.json)、
[active contract](../artifacts/non_df_h4_central_r1_sine/2026-10-24/active_contract.json)、
[grant](../artifacts/non_df_h4_central_r1_sine/2026-10-24/run01/grant.json)、
[ledger](../artifacts/non_df_h4_central_r1_sine/2026-10-24/run01/call_ledger.json)に結合した。
preflight1＋論理診断3＋input/output6の合計10予約を呼出前に記録した。
科学作用成功6／総試行6、新C-target0、cosine作用0。
worker PID631463は2026-10-24 13:56:59 UTCにexit0・reap、13:57:00 UTCに[STOP](../artifacts/non_df_h4_central_r1_sine/2026-10-24/run01/STOP.json)を保存した。
stderr空、all_owned_reaped=true、automatic_next_stage=false。元H4 native UNKNOWNの代替receiptにはしない。

[保存検証](../artifacts/non_df_h4_central_r1_sine/2026-10-24/verification.json)はPASS。SOURCE/runtime/input/旧証拠SHA、旧manifest184行不変、
grant消費、6receipt、C cache・scalar恒等式・sine予算・native終了を照合した。
後検証の新状態作用・QPY decode・保存vector再縮約はすべて0。
[出力inventory](../artifacts/non_df_h4_central_r1_sine/2026-10-24/output_inventory.json)、
[完了監査](../artifacts/non_df_h4_central_r1_sine/2026-10-24/completion_audit.json)へ保存した。
元SOURCE/raw/結果/grant/ledger/marker/STOPは変更していない。
最新版運用規約§8.4/10.1/10.2と共通AGENTSを適用し、利用可能版に§10.3がない版不一致を記録した。
6短作用を同一workerで入力共有し、CPU/RAM/thread/affinity上限や共有環境・他process変更は加えていない。

## 6. 次段階（提案だけ・別途承認）

選択したPF/PR0/PR1の両軸について、経験的な局所diagnostic coverageは揃った。
保証上界が目的なら、次は**CodexがPauli compiled programのvalidated enclosure接続を設計・人工検証する契約案**を作り、
**ユーザーが対象・precision schedule・call上限を明示承認**してから実H4のArb評価へ進む。
既存DF local RZ/XXPlusYY kernelだけではPauliの全wrapper/native rz/sx/x/cx伝播をそのまま保証できず、SOURCE・relative phase・sector縮約までの接続が必要である。
PRの保存C点値を真値として固定せず、同じ個別C-targetの包含も別途必要。
最小pilotは選んだ1対象・1軸のC-target／compiled包含で、δ/η分離が必要ならinputを追加する。
PFなら既存C区間の適合を確認して理想targetの新作用を省略できる候補がある。
現在のbinary64結果から新しいroundoff allowanceを仮定したり、N/Gを再計算したりしない。

PR全law保証には、これとは別に未観測order4まで含むuniformまたは正当なlaw-weighted転送設計R2が必要。
局所包含の成功でも母平均費用tailは解決しない。
**今回Arb・R2・追加trajectory・追加compile/sampling・N/G・正式受理を開始していない。mandatory STOP。**

## 7. 成果物・公開状態

[artifact入口](../artifacts/non_df_h4_central_r1_sine/2026-10-24/README.md)、
[実行集計](../artifacts/non_df_h4_central_r1_sine/2026-10-24/run01/result_summary.json)、
[公開状態](../artifacts/non_df_h4_central_r1_sine/2026-10-24/publication_status.json)を参照。
科学SOURCEと結果commitを区別し、push試行と確認可能なremote SHAを記録する。
認証失敗なら結果をlocal保全し、再実行用pushコマンドを最終チャットに提示する。

## 付記：SOURCE準備時の記録

ユーザーの「次の作業に進んで大丈夫」と、sine3診断・最大6状態作用を選択した回答に基づく。
基点c7585f2d6e4a0fa3f0cc873fb6e468be1d42bce9、専用branch codex/non-df-h4-central-r1-sine-20261024。
Linear H4/1Å/STO-3G/n8+ancilla8、Pauli184/prefix27、S2/T=.8/q8/r4/K4、元state/rayと同一νを維持する。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なしの保存sine input/outputを使う。

PF・PR0・PR1についてinput/output各1、3論理・初回最大6新状態作用だけを承認範囲とする。
PF C imag保存区間とPR0/PR1選択trajectoryの保存複素C-target.imagを再利用する。
新C-target作用・cosine再評価・compile・sampling/RNG replay・新Hamiltonian/state/trajectory/回路・Arb/precision escalation・full C/L・N/G・R2・certificate受理は0。
R0のpropagate／saved_PF_target／compare_signalsとone-shot controllerを再利用する。
Sdg-before-final-Hの+Im規約・raw ray・unclipped sector縮約を人工2〜3 qubitで検証した。

新9＋既存R0 10人工テストPASS。実保存QPY15個をdecodeし、source/control/cosine構造/sine input-output、phase、fingerprint、compiler、cacheの同一ray/runtimeを照合PASS。
cosine QPYの構造照合は状態作用ではなく、cosine信号を再計算していない。
初回事前検査はA_AXES metricsのnested targetとA_GATE flat schemaの差で停止。
SOURCE前・科学grant0・状態作用0でsource別の正規項目へ接続修正し、失敗log・回復記録・追加regressionを保存した。

入力109／旧保全309／旧manifest184行を保全。GitHub5commitと4blobを実取得・照合した。
新SOURCEをcommitしてからruntime・入力・schema・承認をsealし、新one-shot grantへ固定する。
出力はEMPIRICAL_ONLY_NO_RHO_UPPER、numerical_error_upper/rho_upper=null、LOCAL_BUDGET_MET=false。
旧hash FAIL/原因UNKNOWN・元native UNKNOWN・旧N/G null・C/L COMBINATION_BLOCKED・旧compiler MISSING_EVIDENCE・PR全lawρ/tail UNKNOWNを保持する。

最新版運用規約§8.4/10.1/10.2、共通AGENTSを適用。利用可能版に§10.3は存在せず版不一致を記録する。
6短作用は入力共有・parse重複抑制のため同一workerで実施し、CPU/RAM/thread/affinity上限や共有環境変更を加えない。
完了後mandatory STOP。Arb/R2/N_G等へ自動移行しない。
