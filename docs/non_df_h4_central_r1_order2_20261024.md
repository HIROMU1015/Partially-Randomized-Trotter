# H4 Pauli central-control R1：PR index1 order2 cosine経験的残差

PR index1のorder2 cosine **1論理診断を完了、新状態作用は成功3／総試行3**。
直接差d=`4.3771358202651746e-16`は保存PR cosine予算より小さい経験値であった。
判定は**EMPIRICAL_ONLY_NO_RHO_UPPER**、`rho_upper=null`、`numerical_error_upper=null`、`LOCAL_BUDGET_MET=false`。
今回のnative exit0・reapと保存検証を完了し、**mandatory STOP**した。sine・Arb・R2へ進んでいない。

## 1. 固定条件・SOURCE・入力

Linear H4／隣接1.00 Å／STO-3G、system8＋ancilla wire8、Pauli-JW非identity184項／PR prefix27。
S2、T=.8、q8、r4、K4、元event law・保存trajectory・raw ray・cosine Hadamard wrapperを維持した。
Qiskit1.3.0、basis `rz,sx,x,cx`、optimization1、seed17、backend/coupling/layout/routing指定なしのA_GATE保存QPYを使用した。
本件はPauli central実装の局所診断であり、DFやH2/S2/Yoshida4費用順位解析を混合しない。

| 来歴 | commit |
|---|---|
| 基点 | `f01c4018201de9335f20a8901ec5f40f9cf9e89b` |
| **今回の科学SOURCE（状態作用前にcommit）** | **`ca9a1c0e2426565ba63cc8bfd0247f73339d95b8`** |
| 元H4 | `0c329b30a81accb35c212be8defa66fee0dd0f12` |
| C | `e80d7510880f8db993b777c627bb8e95b05a070d` |
| D2 | `74515d8eb4f073c40fc783681a3775dbca89e9b1` |
| A_GATE | `ee7d281b5daead1d0d5e81e00c9a0ed4727792b0` |
| 条件付きC/N/G | `d96dfe5ca0a6c950a5f1791ff61659e6dc8f1c34` |
| R0 | `84a326598b006384d13c7a354a3eec6881ee841c` |

専用branch：`codex/non-df-h4-central-r1-order2-20261024`。
基点・主要SOURCE7commitをGitHubから実取得し、targets・R0 source・A_GATE input metadataの3blobをGit blob SHAで照合した。
[GitHub記録](../artifacts/non_df_h4_central_r1_order2/2026-10-24/github_verification.json)。基点は今回GitHubから取得できた事実として記録し、R0に残る過去のpush失敗履歴は改変しない。

対象trajectory ID：`596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa`。
trajectory hash：`0d16d201c87fdbb5f3e219dffb7dca7d48a69799290453e9ebb110a267db0e4b`。
event hash：`9cea6ce6d3659094b6b0d0a6da8b935cf32860c97a2e01a75a0761eded9f1779`。
[inputs.json](../artifacts/non_df_h4_central_r1_order2/2026-10-24/inputs.json)に元price_join・state model・4 QPY・fingerprintの参照を保存した。

## 2. Order2のproduct・sign・phase対応

32event中、position30／outer step7／inner2のみorder2であり、他31eventはorder0。

| application順 | component | Pauli label | sign |
|---|---|---|---:|
| 0：product | 43 | IYXXYIII | +1 |
| 1：product | 4 | IIIZIIII | −1 |
| 2：rotation | 72 | YYIIIIXX | −1 |

θ=`0.029672025283184623`、order phase=−1。
unsigned PauliをP43/P4/P72と書くと、元`event_unitary`の式は
`U30 = (−1) [cos θ I − i sin θ (−P72)] (−P4) (+P43)`。
P43の後にsigned P4、最後にsigned rotation72を適用し、非可換productを逆順にしていない。
負productの符号はC-targetではsigned operand、元appendではπ加算として保持し、order phaseとは別のSOURCE上の所有者を維持した。
constant Hamiltonian phaseは各outer stepのreverse保持部の後に適用する。

保存source bodyのaggregate phase **`0.26518225073344787`**を変更せず使用し、
PR0やscalar metadataの近い値`0.2651822507334482`へ置き換えなかった。
元のπ・order phase・constantの更新順序を保った保存phase historyとSOURCE recipeを照合した。
controlled bodyの全体phaseは0、保存aggregate phaseをancilla P gateへ一度だけ転送した。
compiled全体phase `3.2741837789565587`も保持した。

実QPY4個（source body、controlled body、cosine input、compiled output）を本番前・本番内にdecodeした。
469 source tokens、1588 controlled instructions、1591 input instructions、3043 compiled instructionsをSOURCE・保存mapping・histogramに照合した。
input/output fingerprintはそれぞれ`156bcc95a46db4ed6466ac5a969fb86f71e8b9dfc5ff214b1641a583b07852a9`／
`e7411e76d88c7f02f9b7d3313f81adf58b412cd5fe5a284da6071dc65a704aea`。
wire8→clbit0、cosine H/body/H/measure、control0/1のSOURCE対応を検査し、
未対応gate・未解決parameter・中間measurement・reset・condition・layoutは検出しなかった。
[preflight.json](../artifacts/non_df_h4_central_r1_order2/2026-10-24/run01/preflight.json)にphase更新履歴も保存した。
**この限定照合と信号診断では位相・scale・control不一致を検出していない。** 全programに対するcompiler certificateではない。

## 3. 三信号・δ／η／d・保存ρ予算

元raw rayのhashは`8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`。
R0保存のexact νと同一state hashを照合して再利用し、binary64表示ν=`1.0000000000000002`を全observableに共通使用した。
raw stateや出力を再正規化せず、最終measurementだけを除いたQPYを作用させ、
伝播振幅のancilla sectorから`m=(p0−p1)/ν`を直接縮約した。Probabilities APIのclippingは使わない。
C-targetはPR index1の**個別32event**のvector作用であり、law meanではない。

| PR1 cosine保存値 | 結果 |
|---|---:|
| mC：個別C-target | 0.01919550205004739 |
| m_in：保存未compiled input | 0.019195502050046772 |
| m_out：保存compiled output | 0.01919550205004705 |
| B_saved | 1.2873709223899619 |
| δ：B_saved × abs(m_out−m_in) | 3.573172098175653e-16 |
| η：B_saved × abs(m_in−mC) | 7.950307918440827e-16 |
| **d：B_saved × abs(m_out−mC)** | **4.3771358202651746e-16** |
| ε_sig=.0005の保存PR cosine ρ予算 | 2.593611368819805e-6 |
| d／保存ρ予算 | 1.687660639094493e-10 |
| 保存ρ予算−d | 2.5936113683820912e-6 |
| 経験的な大小 | BELOW_SAVED_BUDGET |

B_saved exact=`1449450801590761/1125899906842624`、保存予算exact=`19792741170690161541/7631344236317346665332736`。
[引用元guard](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/guard_inputs_quoted.json)を読み、B_savedを一度だけ掛けた。B_upは使用していない。
`d≤δ+η`は保存点値のexact rational恒等式として成立し、表示値とexact値を保存した。
新しいρ予算・shot数N・総費用Gを計算したのではない。
[PR1_result.json](../artifacts/non_df_h4_central_r1_order2/2026-10-24/run01/PR1_result.json)、
[R0/R1転記CSV](../artifacts/non_df_h4_central_r1_order2/2026-10-24/results.csv)。

### R0との限定比較

| cosine対象 | δ | η | d | 来歴 |
|---|---:|---:|---:|---|
| PF | 4.197753256107717e-13 | 4.384974112649559e-13 | 1.8722085654184194e-14 | R0保存値、再計算なし |
| PR0 | 4.502196843701322e-15 | 5.2882947052999665e-15 | 7.860978615986437e-16 | R0保存値、再計算なし |
| PR1 | 3.573172098175653e-16 | 7.950307918440827e-16 | 4.3771358202651746e-16 | 今回の新3作用 |

3対象とも引用予算より小さい経験値。PR1の表示dはPR0より小さいが、
trajectory全体のevent列・信号・programが異なるため、order2が残差を小さくしたという因果効果にはしない。
PR1ではmC > m_out > m_inとなり、δとηに打消しがある。
PR0/PR1のdはbinary64丸めに近い桁であり、真の非零残差や確定した有効桁として扱わない。
保存comparisonは[JSON](../artifacts/non_df_h4_central_r1_order2/2026-10-24/run01/R0_R1_comparison.json)で、平均・全law外挿・N/G再解析は行っていない。

### 未保証の数値誤差

input/outputの観測normは`0.999999999999972`／`0.9999999999999805`。これらで信号を割り直していない。
三角関数、matvec、Qiskit状態伝播、sector縮約、ν変換には数値誤差上界がない。
C-targetもbinary64点値であり、真値包含区間ではない。exact scalar会計はその点値間の算術関係を検証するだけである。
**`EMPIRICAL_ONLY_NO_RHO_UPPER`、`rho_upper=null`、`numerical_error_upper=null`、`LOCAL_BUDGET_MET=false`を保持**する。
PR全lawのρ、未観測order4、256本の残差平均、母平均費用tail、条件付きN/Gの正式保証は未確定。

## 4. 実装再利用・人工検証・実行終了

[adapter](../src/trotterlib/h4_r1_order2.py)はR0 `propagate`／`selected_C_action`／`compare_signals`を変更せず呼び出した。
R0のselected C作用は元からgenericなorder0/2/4の`event_unitary`を扱うため、新simulatorやorder2数値kernelは追加不要だった。
今回追加したのはPR1の固定保存descriptor、事前event意味論検査、QPY/SOURCE接続、1件3作用のdispatchである。
[runner](../scripts/run_h4_r1_order2.py)はR0のverify/prepare/check/worker/launchをAST抽出し、scopeの定数とSOURCE来歴を変更した。
namespace種別、基点、2→1論理／5→3作用、禁止call、schema、実行script bindingを明示設定し、元SOURCEを変更していない。

新[人工7テスト](../tests/test_h4_r1_order2.py)＋再利用R0 10テストの**17 PASS**。
非可換productの逆順・負product符号・rotation符号・order phase・position・angle不一致の拒否、
元appendとselected vector schedule、control両branch、raw ray observable、3作用の非返還予算を人工2 qubitで確認した。
別AI agentによる独立reviewではなく、主担当Codex自身の検証である。
SOURCE前にcontrollerのjson namespaceを静的修正し最終7テストを再実行した。科学的失敗・retryには当たらない。
[新7最終log](../artifacts/non_df_h4_central_r1_order2/2026-10-24/artificial7_final_PASS.log)、
[再利用R0 10log](../artifacts/non_df_h4_central_r1_order2/2026-10-24/reused_R0_10_PASS.log)。

SOURCE commit後にcode21、runtime6999、入力97、保全253、承認・schemaをsealした。
新grant ID：`4cbf315e6afa2fd28a165b8869bec04624f1122c8b4a201ef076bacadacdc647`。
[seal](../artifacts/non_df_h4_central_r1_order2/2026-10-24/seal.json)、
[active contract](../artifacts/non_df_h4_central_r1_order2/2026-10-24/active_contract.json)、
[grant](../artifacts/non_df_h4_central_r1_order2/2026-10-24/run01/grant.json)、
[ledger](../artifacts/non_df_h4_central_r1_order2/2026-10-24/run01/call_ledger.json)。
preflight1、PR1論理診断1、input/output/C-target各1を呼出前に予約した。科学作用は**3成功／3試行**、科学的中断・回復実行・追加grantは0。
worker PID619848は2026-10-24 11:08:21 UTCにexit0・reap、11:08:22 UTCに[STOP](../artifacts/non_df_h4_central_r1_order2/2026-10-24/run01/STOP.json)を保存。
stderr空、`all_owned_reaped=true`、`automatic_next_stage=false`。元H4 native UNKNOWNの代替証拠にはしない。

[保存検証](../artifacts/non_df_h4_central_r1_order2/2026-10-24/verification.json)はPASS。R0由来のhash verifierを再利用し、
SOURCE/code/runtime/input/保全、旧manifest183行不変、grant消費とcall数、receipt・scalar恒等式・quoted budget・native終了を照合した。
後検証は状態作用0、QPY decode0、保存vectorからの信号再縮約0。
[出力SHA inventory](../artifacts/non_df_h4_central_r1_order2/2026-10-24/output_inventory.json)と
[完了監査](../artifacts/non_df_h4_central_r1_order2/2026-10-24/completion_audit.json)へ結果を保存した。

## 5. 保全・運用版・未解決状態

R0のSOURCE・結果・grant・ledger・STOPと旧H4原本を保全し、今回の成果物だけを専用namespaceへ追加した。
Pauli-PR historical fingerprint FAIL／原因UNKNOWN、元H4 native UNKNOWN、旧N/G null、
C/L COMBINATION_BLOCKED、旧compiler MISSING_EVIDENCE、PR全lawρ UNKNOWN、母平均費用tail UNKNOWN、正式certificate未受理を保持する。
PF/PR0再計算・sine・Arb・precision escalation・R2・新compile・sampling/RNG replay・新Hamiltonian/state/trajectory/回路・full C/L・N/G・正式受理はすべて0。

最新版運用規約§8.4/10/10.1/10.2と共通AGENTSを適用。
利用可能版に§10.3はなく、版不一致として記録し直接ユーザー指示を適用した。
3つの短い256/512 amplitude作用は同一workerで検査済みQPY・stateを共有し、重複parseとprocess起動を避けた。
[資源観測](../artifacts/non_df_h4_central_r1_order2/2026-10-24/resource_before.json)を保存し、CPU/RAM/thread/affinityの独自上限や共有設定・他process変更を加えていない。

## 6. 次段階の候補（提案だけ・別途明示承認）

| 候補 | 科学的意義と最小範囲 | 担当・優先度 |
|---|---|---|
| A：R1 sine | PF/PR0/PR1の実保存sine QPY input/outputを各1、計3論理・6新状態作用。PF保存C imag区間とPR0/PR1保存複素C-targetの適合を確認して再利用できれば理想target再計算0 | ユーザーが対象・上限を承認、Codexが新SOURCE/grantで実施。経験的coverage拡張の第1候補 |
| B：Arb区間評価 | 選んだ局所対象でC-targetとcompiled測定信号のvalidated enclosureを得る。まずPR1 1件ならselected C-target／compiled各1を採用precisionごとに固定。δ/η分離も必要ならinputを追加 | ユーザーが対象・precision schedule・call上限を承認、Codexがbackendを人工検証後に実施。局所ρ上界が目的ならAより優先する。PR点値を真値として流用しない |
| C：R2全law転送 | 未観測order4を含むevent lawへのuniform／正当なlaw-weighted上界を設計。初段階は静的設計、科学call0の契約案から開始できる | ユーザー承認後Codex。局所3件や保存256本から全lawへ外挿せず、tail保証とも区別する |

今回大きなSOURCE・phase不一致が検出されなかったため、再compileや入力の選び直しは不要と判断した。
経験的な測定軸coverageならA、厳密な局所保証ならBを次に選ぶ。**今回はA/B/Cの実装・科学計算を開始していない。**

## 7. 保存入口・commit・公開状態

[artifact入口](../artifacts/non_df_h4_central_r1_order2/2026-10-24/README.md)、
[実行集計](../artifacts/non_df_h4_central_r1_order2/2026-10-24/run01/result_summary.json)、
[公開状態](../artifacts/non_df_h4_central_r1_order2/2026-10-24/publication_status.json)を参照。
科学SOURCEと結果commitを区別し、push試行と確認可能なremote SHAを公開状態へ保存する。
認証失敗時は成果物をlocal保全し、再実行コマンドを最終チャットへ提示する。
**mandatory STOP。sine・Arb・R2・N/G再解析・追加compile/sampling・certificate受理へ自動移行しない。**

成果物commit：`dc48d402e98bece3e14361e1981a5f7370581fb1`。
push試行1はexit128、GitHub HTTPS Username取得不能で失敗した。
[失敗log](../artifacts/non_df_h4_central_r1_order2/2026-10-24/push_attempt01.log)、
[remote再確認](../artifacts/non_df_h4_central_r1_order2/2026-10-24/remote_branch_check.json)を保存。
GitHub connectorの専用branch検索は0件、remote SHA=null。
後続metadata commitはこの公開状態だけを記録し、科学SOURCE・grant・run・STOPを変更しない。

## 付記：SOURCE準備時の記録

基点`f01c4018201de9335f20a8901ec5f40f9cf9e89b`、専用branch `codex/non-df-h4-central-r1-order2-20261024`。
ユーザーはPR index1 cosineの1論理診断・初回最大3状態作用（input/output/selected C-targetを各1）を明示承認した。
固定Linear H4/1Å/STO-3G/n8+ancilla8、Pauli184/prefix27、S2/T=.8/q8/r4/K4。
A_GATEのQiskit1.3／rz,sx,x,cx／opt1／seed17／backend・coupling・layout・routing指定なしの保存QPYを使用する。
新compile・sampling・Hamiltonian/state/trajectory/回路生成、PF/PR0再計算、sine、Arb、full C/L、N/G、R2、正式certificate受理は0。

対象ID `596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa`。
trajectory hash `0d16d201c87fdbb5f3e219dffb7dca7d48a69799290453e9ebb110a267db0e4b`、
event hash `9cea6ce6d3659094b6b0d0a6da8b935cf32860c97a2e01a75a0761eded9f1779`。
32eventのposition30／step7／inner2のみorder2。product43(+1)→4(−1)、rotation72(−1)、angle .029672025283184623、order phase−1。
保存aggregate phase `.26518225073344787`をそのまま使い、PR0の近い位相へ丸め直さない。
新状態作用前の実保存QPY4個のdecode・SOURCE recipe・D2 control mapping・phase history・wrapper・fingerprint照合はPASS。

[adapter](../src/trotterlib/h4_r1_order2.py)はR0のraw observable、selected C vector作用、残差会計を呼ぶ最小接続。
[runner](../scripts/run_h4_r1_order2.py)はR0のverify/prepare/check/worker/launchを再利用し、namespace種別、基点、1論理／3作用、schema、禁止call、SOURCE来歴を明示設定する。
R0の元ファイル・grant・ledger・STOPを変更しない。
元append処理を使った非可換product順序・負product符号・order phase・control両branch・raw observableと、有限scopeを人工2 qubitで検査した。
新7テスト＋再利用R0 10テストを実行。最終ログをartifactへ保存する。
SOURCE前の静的点検で再利用controllerのjson namespaceを明示接続した。科学grant・状態作用はまだ0。

入力97file、旧証拠保全253file、旧manifest183行を固定。
GitHubの基点・主要SOURCE7commitと保存blob3個を実取得・照合した。
新SOURCE commit後にruntime/input/schema/承認をsealし、専用未使用one-shot grantに接続してから実行する。
出力はbinary64経験値のみ。`rho_upper=null`、`numerical_error_upper=null`、`LOCAL_BUDGET_MET=false`、`EMPIRICAL_ONLY_NO_RHO_UPPER`。
PR0比較は保存値の引用だけ。旧hash FAIL/原因UNKNOWN・元native UNKNOWN・旧N/G null・C/L/旧compiler/PR全law/tailを保持する。

最新版運用規約§8.4/10/10.1/10.2を適用。利用可能版に§10.3はなく、版不一致を記録し直接ユーザー指示を適用する。
3つの短い256/512 amplitude作用は同一workerで入力・検査済みQPYを共有する。
CPU/RAM/thread/affinity上限や共有環境・他process変更を加えない。実際の利用可能資源を記録する。
完了後mandatory STOP。sine/Arb/R2等は提案だけとし自動移行しない。
