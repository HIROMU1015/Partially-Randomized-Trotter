# 12局所probe：runner実装・人工検証・実行SOURCE固定

2026-11-04 JST / LOCAL / 主エージェント自身の技術作業。
**新規人工26＋前段人工34＝60件PASS。新しい実H4の科学実行は0。**
前段の方針採択を受け、実行可能なrunnerとsignal側τ adapterを準備した。
科学実行はこの公開SOURCEへのユーザーの別途明示許可後。独立詳細科学レビューは行っていない。

専用branch `codex/rpe-controlled-hadamard-runner-preparation-20261104`、別worktree、
親commit `bdffe73a6a8af0988e62ff90c9dd344880e8b8b5`。
前段のSOURCE・4成果物・人工結果を編集せず、新しいディレクトリへ追加した。
新科学結果やevidence statusの変更ではないため、旧synthesis/manifest/rawも更新しない。

## 実装と再利用

| SOURCE | 役割 |
|---|---|
| [run_local_probes.py](run_local_probes.py) | defaultはbyte/schema照合。明示--runでのみH4 contextをload。6r群の並列化、call集計、raw保存、STOP |
| [probe_adapter.py](probe_adapter.py) | signal τのhex完全一致、12強制profile、既存request/builder/policy/wrapper、copyした入力へのstate-action |
| [reference_adapter.py](reference_adapter.py) | 96-bit analytic occupation-energy参照、signed component、MatrixBall伝播、normalized ray、誤差込み比較 |
| [source_inputs.json](source_inputs.json) | 追加の固定SOURCE。前段20 SOURCEと12固定入力のbyte bindingを再利用 |
| [EXECUTION_SCOPE.md](EXECUTION_SCOPE.md) | 次の科学実行の具体的な対象・call・精度・STOP。今回の科学実行許可ではない |
| [test_local_probes.py](test_local_probes.py) | 実H4 loaderを呼ばない2-mode人工検証・並列統合mock |

production moduleは変更していない。
既存_make_event、_request_for_events、RepeatedRequest、raw/boundary builder、
support_run_le_1、Hadamard benchmark wrapper、mean_adapter/ball_adapterを再利用した。
元runner全体にはsampling・他の科学実行が含まれるため呼ばず、必要な関数だけを利用する。
新規コードはτの結合、固定profile、独立参照、局所batchの入出力に限定した。
承認証明・保存承認文・grant発行・認証・管理DBは作成していない。

固定科学条件はlinear H4/1.00Å/STO-3G/legacy DF12、Partial L_D3/δ=.02/K2、
保存physical-state ray/sector70、extract_identity_phase/cutoff1e-12。
対象はr1,2,4,8,16,32の各q_probe1/all0とq_probe2/stressのみ。
親m0,8,10,11,13,17のq/N/α/β/εをmetadataに保ち、親の大きいqで回路を構築しない。

## 対応差を黙って消さない

signal_configは保存signal側τと固定preparation λ_R×(δ/r)をhex完全一致で検査する。
model側のcloseなliteralへの置換、validatorの緩和、cutoff既定0への復帰をしない。
既存finite distributionとRTEConfigを再利用し、CDF・samplingは呼ばない。
θのrecipeは既存_make_eventのatan(τ/(n+1))。
将来承認された6群parameter replayでθ hexを保存してから、参照行列・回路を構成する。
今回、実H4 τからθやsignalを新評価していない。

参照はcontrolled出力とproduction diagonal primitive generatorを使わず、
保存exportのanalytic basisとpreparationのoccupation energyから作る。
保存step位相の評価順を保ち、boundary側の再結合差も比較へ現れる構成。
その差を長qへ転送する新しい上界を与えたとはしない。

## 人工検証

既存環境を変更せず使用：Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、
Qiskit1.3.0、python-flint0.9.0/FLINT3.6.0、pytest9.0.3。
人工入力は手書き2-mode DFと未正規化vector [1,i]。
実H4のNPZ deserialize、preparation、build、state-action、signal、compile、samplingは0。
default CLIだけ実保存資料のbyte/schema/literalを照合した。

```bash
PYTHONPATH=src <research-python> -m pytest -q \
  docs/research/rpe_controlled_hadamard_runner_preparation_20261104/test_local_probes.py \
  docs/research/rpe_controlled_hadamard_preparation_20261104/test_circuit_semantics.py \
  tests/test_df_rte_qiskit_builder.py::test_controlled_event_is_exact_block_diagonal \
  tests/test_df_rte_qiskit_builder.py::test_every_small_event_matches_dense_reference_with_all_signs_and_phases \
  -p no:cacheprovider
```

**60 passed in 15.84s**（新26、前段30、再利用4）。local検証でありimmutable CIではない。

| 新規検証 | 件数 | 確認 |
|---|---:|---|
| 6rのprofile・signal literal | 6 | order/products/rotation、r1とr≥2のstress配置、model literalと分離 |
| 近接τの拒否 | 1 | 既存close許容内でも別hexを拒否 |
| 6rの完全pipeline | 6 | 96-bit参照、uncontrolled/raw/optimized、control両branch、測定両軸、正確なcall数、入力不変 |
| primitive generator不使用 | 1 | 独立参照がproduction generatorを呼ばない |
| 参照phase誤り／sine符号誤り | 2 | 不合格とfailed probe/rawの保全、後続action停止 |
| 参照不確かさ／nonfinite | 1 | 点が一致しても不確かさ超過をPASSにしない |
| runtime gateの差替え | 1 | basis class/parameter/descriptor不整合を拒否 |
| normalized ray | 1 | rawのscale変更に対する正規化とraw不変 |
| sibling STOP | 1 | 通知後の新action0 |
| spawn経路 | 1 | serialと同じ結果・call数 |
| default CLI | 1 | --runなしで科学入口を呼ばない |
| 長q／output未指定 | 1 | q_probe制限とCLI境界 |
| runner全体の人工統合 | 1 | H4 loaderをmockし、6群/12probe・counts・8出力・全statusを実コードで照合 |
| θ保存の順序 | 1 | 参照行列/actionより前にparameter hexを保存 |
| 既存outputの保全 | 1 | 既存dirへの上書きをinput処理前に拒否 |

人工統合の一部setup countはmock metadata。実H4科学callを実施したという記録ではない。
人工rawはpytestの一時領域だけに保存し、実H4結果としてartifactsへ登録していない。

最初のtestでは未導入threadpoolctlへの不要な依存が判明し、その依存を削除した。
spawn前の自process環境設定と終了後の復元でBLAS threadを指定し、共有venvを変更していない。
続いて、Qiskit1.3 Statevectorのglobal phase処理で共有input bufferが変わり、
後続probeが異なる入力を受ける人工例を検出した。
全actionにcopyを渡し、6r全てでinput不変と参照一致を確認した。
旧BのSOURCE・実結果を修正／再実行したのではなく、新runnerの技術不具合を人工例で解消した。
比較基準1e-10や科学条件を変えてPASSにしていない。

## 到達点と次の担当

SOURCE固定・defaultの静的検査・公開照合までが今回の作業。
実H4のpreparation identity・signal τ結合・12probeの数値合否は、まだこのrunnerで確認していない。
全218成分、任意trajectory、保存全18q、長qのphase転送、実IID/guard/PCG64、
compiled/noise/state preparation、RPE全体のtolerance保証は残る。

**次の担当はユーザー**：公開SOURCE SHAとEXECUTION_SCOPEの12probe・call範囲に対する
科学実行を別途明示許可するか判断する。
共通運用指示2026-10-25 §8.1は「研究方針の採択、科学計算の実行許可は別」とする。
**Codex**はその許可後だけ同SOURCEの固定batchを1回実行し、原本・条件付き判定を保全し報告後STOP。
今回の公開照合後もmandatory STOP。review・sampling・大きいq・費用再解析へ自動移行しない。

`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、U_exec=nullを維持。
