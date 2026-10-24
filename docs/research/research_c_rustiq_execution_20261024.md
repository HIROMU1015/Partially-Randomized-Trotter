# 研究C・C4：Q13限定科学実行（2026-10-24）

## 実行前固定と新しい承認

公開準備SOURCE `5038427e034c5e2cca6674998d28715cd93ec551` の
[限定契約](research_c_rustiq_limited_contract_20261024.json)を引き継ぐ。
そのSHA256は `e948a2e7fdf47e20b8af8cd7debae0631dabbd5e6ad77f419a9a9e2c668319a1`。
ユーザーが「次の作業に進んで大丈夫」「意図しないところで中断された場合は修正を行って再実行してよい」と明示承認した。
旧contractの承認待ち／retry0の記録は書き換えず、今回の承認と技術的中断時の再実行許可を別にbindする。

科学条件は4case `SS_motif_2,SS_motif_3,SS_motif_4,PP_motif_0` とB0/B1/R1のまま。
保存F1の4 mode・2粒子・occupied(0,1)、density-square合成fixture、L_D=1、残差2fragment、
T=1/5、q2、δ1/10の一点、r2、h1/20、K2。分子geometry／AO basis／分子DF rankは非該当。
Hamiltonian・DF・law・finite target・native basis/profile・許容差・gate順序を変えない。
各fresh attemptの上限は旧契約の28 reference、局所20／full12 build、adjoint12、wrapper24、
Operator68、Statevector48、cold24／warm24 request・実transpile24、Q13合成20のまま。
技術的中断時は失敗原本・STOP・SOURCEを保存し、必要な技術修正を新commitへ固定してfresh attemptを実施できる。
科学的FAILを隠すための条件変更・許容差緩和・case置換・追加探索は許可しない。

追加SOURCEは
[`rustiq_execution.py`](../../src/trottertracks/pr_dictionary_validation/rustiq_execution.py)、
入口は[`run_research_c_rustiq_limited.py`](../../scripts/run_research_c_rustiq_limited.py)、
管理testは[`test_research_c_rustiq_execution.py`](../../tests/management/test_research_c_rustiq_execution.py)。
既存adapter／builder／reference／compilerは変更しない。
`controlled_gate`のreport-only APIで同じOperatorを再評価せず保存するため、
実行入口ではOperatorを一度だけ取り、同じ既存`matrix_gate`で絶対誤差とunitarityを検査する。
比較するdiag(I,U)、tol1e-10／rtol0、位相fit禁止は同一である。

preflightで37 SOURCE hashと7保存入力hash、Python3.11.9／Qiskit1.3.0とAPIの一致を確認した。
24件のadapter＋実行管理unit/mock testがPASS。まだ実回路のPASSを意味しない。
共有環境への導入・変更なし。保存F1専用venvを読み、cache/tempは本実行専用領域へ置く。
起動時観測はCPU affinity32、load約1、MemAvailable約56 GB、CPU/memory pressureほぼ0。
独立wrapper task24に対応する最大24 process、BLAS/OpenMP/Numba/Qiskit各1 thread、
Rayonはavailable CPUs÷worker capacityとしてoversubscriptionを避ける。
gate間の依存はbarrierで保ち、各stage内の独立taskを並列化する。人工的CPU/RAM/wall上限なし。
GPUは既存driverに接続できず、今回のQ13／compiler／32次元検査はCPU経路を使う。

approval packetは固定実行commit、追加実行SOURCE hash、元contract hash、fresh outputへbindし、
最初の科学callより前に保存する。旧F1/F2 grantを再利用しない。
結果の保存先は`artifacts/research_c_rustiq_limited/2026-10-24/<fresh-run-id>/`。
実行後に結果・科学的限界・全attempt履歴を以下へ追記し、GitHub公開・照合後mandatory STOPする。
