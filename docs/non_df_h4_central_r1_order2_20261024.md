# H4 Pauli central-control R1：PR index1 order2 cosine経験的残差

## SOURCE準備

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
