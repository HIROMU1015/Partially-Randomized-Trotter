# H4 Pauli central R1 sine：PF・PR0・PR1経験的診断

## SOURCE準備

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
