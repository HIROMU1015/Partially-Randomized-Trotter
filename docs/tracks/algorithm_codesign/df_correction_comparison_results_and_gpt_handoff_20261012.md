# DF whole / local Pauli 比較 one-shot：technical STOPとGPT引き継ぎ

2026-10-12 JST。最終分類は **`TECHNICAL_INCONCLUSIVE`**。
凍結契約で一回だけ実行したが、E-r3の最初のwhole回路compile後に既存fingerprintとの不一致を検出し停止した。
CD 6 taskは保存済み、E_NATIVE/T_KEYSは未完了。部分結果からA/Bの科学的勝敗・全PR資源優位を判定しない。
**run=1、retry=0、marker消費済み、mandatory STOP。** 修正・再実行・次stageは今回行わない。

## 固定sourceと実行承認

| 項目 | identity |
| --- | --- |
| 固定source S' | `41349b73157c7781e0b534cf0fe61e06bf439954` |
| 実行HEAD / authorization-only直接子 A | `a1f537b68f0f22419dff0202c553b7511653caa3` |
| 実行branch | `track-b-df-correction-comparison-one-shot-execution-20261012` |
| authorization SHA256 | `30f21541c8ae90615fcc4a8a779815749b2f13eb0d8faf6686e014c9fe2b75f0` |
| contract v2 SHA256 | `0079988b0e63ea7c62fec8e435bcf17313a3563d9480caf8aaf569e2e5972305` |
| marker SHA256 | `dd6b9d5ec96e9d34d7bea0277eadb03f031992e1010cfdb2c940a393b2242701` |

Aの唯一のparentはS'。変更はauthorization JSONと許可済みreceiptのみ。
利用者の最新指示「作業を進めて」を、直前に確認された限定one-shot workflowへの実行指示として記録した。
[authorization](../../../artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/authorization.json)と
[receipt](df_correction_comparison_execution_authorization_v2.md)には、原文・認可の文脈・GPT review原本のbyte identityが入っている。
receipt内の原本は17,708 bytes、SHA256 `0058e87918e856ac8934b12938e5cf6e56db93951c78bb60dbf905a300374f07`。
公開確認用HEAD `a1ec59e6b2d1c25f10daabfc752755bda317f718`は実行HEADに使っていない。

実行前にremote SHA=A、clean worktree、source manifestの216 files、numeric/synthesis両runtime、
contract/auth/review identityとmarker未存在を確認した。
[prelaunch receipt](../../../artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/prelaunch_gate_receipt.json)を参照。
runnerは固定numeric Pythonで `-B scripts/tracks/algorithm_codesign/run_df_correction_comparison_v2.py` を一度のみ呼び出した。

## 登録条件と実際に完了した範囲

科学契約v2・source・seed・fixture・precision・error/confidence・capsは変更していない。
CDはsynthetic 3-mode / 3-block、coefficients 9/20, 7/20, 1/4、ranks 1/2/1、signs +/−/−、
`L_D=2`、seed6121107、t=1、δ={1/8,1/16,1/32}、γ=5/4、s=1、paired cutoff4 / P5。
ε_complex=1e−4、全α=0.05、各axis family0.025、6 task×2 axisごとα=1/240を維持。
登録状態・同じ理想signal・条件付き誤差会計を維持した。shot数は計算済みsufficient-shot値で、実量子shotsは0。
Eは別fixture（seed7101107、r3/r6、元生成順序）であり、E単価をCD頻度へ移植していない。

| stage / label | 保存済み状態 | process status |
| --- | --- | --- |
| CD | A/B×δの6 task、conditional envelope / symbolic cost | exit0 |
| E-inputs | r3/r6の登録frameを元生成順序で保存 | exit0 |
| E-r3 | first compile直前のprefix、`compile_calls=1`、accepted native record0 | exit1 |
| E-r6 | 並列起動済み、prefix/result未保存 | sibling failure後SIGKILL、exit−9 |
| T inventory / T_KEYS / T accounting | 未到達、inventoryなし、T synthesis0 | 未起動 |

**native compile総回数は未確定**。E-r3の保存prefixは1 attemptを示すが、E-r6は未保存のためin-flight回数を復元できない。
E-r6を未起動／compile0と表現しない。登録上限22を拡張するjobは起動していない。
T全coverage・限定下界・actual T数は取得できず、未実行として残す。RZ→T換算推計も行っていない。
分子/NPZ/GPU/trajectory/LP/実量子shots、G10/RA-D0再開は0。

## 停止理由：確認できることと未確定事項

[terminal_status.json](../../../artifacts/track_b_df_correction_comparison_result/2026-10-11/v2/terminal_status.json)の理由は
`CapFailure: Child failed or output missing: E-r3`。
これはsupervisorがchildのexit1を処理した例外名であり、資源cap超過そのものを意味しない。
[E-r3.log](../../../artifacts/track_b_df_correction_comparison_result/2026-10-11/v2/E-r3.log)には、
`ArithmeticError: Frozen E whole fingerprint mismatch; no synthesis` と
`native_evaluation.py:87` のtracebackが保存されている。

静的sourceでは、wholeをuncontrolled→controlledの順にcompileし、最初のuncontrolled回路で
`actual_circuit_fingerprint`と旧Phase E r3/uncontrolled fingerprintを比較している。
旧期待値は `171f321e2f3bcc51b177e7b62735b5fbf9d80e36a5f56639b7f03809efac41b5`。
失敗時のactual fingerprint、native IR、operator errorは保存されていないため、どのgate/angle/phase/数値構成が
異なったかはこの記録から確定できない。fingerprint不一致だけで数学的・物理的誤りとも断定しない。
停止を起こした記録はfingerprint guardであり、RSS/CPU/wall breachを示す記録はない。
source修正、guard緩和、旧fingerprint更新、追加compileによる原因再現は行っていない。

## 実行資源と外側process

CPU affinity32・host RAM約62.49 GiBをmetadataで確認。GPU照会なし。
CD1 / native2 / synthesis2 workers、BLAS/OpenMP/MKL/NumExpr/Rayon各thread1、Qiskit内部parallel=false。
上限は全wall2400s / CPU1800s / aggregateRSS3072MiB、native22compile、T64keys、
output64MiB / log4MiB。各phase/worker/per-key上限も契約どおりで、速度向上率は主張しない。

| 会計範囲 | wall s | CPU s | RSS MiB |
| --- | ---: | ---: | ---: |
| 外側Popen（runtime preflight込み） | 5.285329016856849 | 5.147606 | 356.3515625（max child、aggregateではない） |
| marker後のsupervisor | 2.7439750181511045 | 4.598258（parent＋reaped children） | 728.81640625（sampled aggregate peak） |
| CD worker保存receipt | 0.6899423371069133 | 0.671576162 | 96.40625 |

外側exit2、stdoutの分類・理由、保存terminal分類は一致。
[outer receipt](../../../artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/outer_process_receipt.json)と
[stdout](../../../artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/runner.stdout.log)をbyte identity付きで保存した。
RSSは50ms samplingであり、瞬間的な超過を完全に排除する主張ではない。

## 保存値・provenance・取得性

[read-only監査](../../../artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/saved_output_audit_v1.json)は
source216 files、authorization/contract/marker、job canonical hash、outer status、
CD6行の保存確率和・moment/rangeの厳密有理数factorization、全chunkのUTF-8/order/bytes/SHA256を照合した。
監査はstdlibだけで行い、scientific runner・matrix・compile・synthesis・solverを追加していない。
初回の監査では`job_sha256`をraw file hashと仮定したため検査が停止した。
凍結workerの定義である `SHA256(json.dumps(job,sort_keys=True).encode())`へ監査側のみを合わせた。
[訂正記録](../../../artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/audit_field_assumption_correction.json)を保持し、
science retryとは区別した。原結果・source・STOPは不変。

CD原JSON logical identityは1,118,571 bytes、SHA256
`98f62b87107681a931d0502f37b81eea9806c890ef278c64532b7d5d60dbd075`。
[CD.review index](../../../artifacts/track_b_df_correction_comparison_result/2026-10-11/v2/CD.json.review.json)に従い
chunkを順序どおり連結すれば原bytesへ完全復元できる。各chunkは256KiB以下、各保存fileは512KiB以下。
logical `CD.json`は大きいためraw fileがなく、indexとchunkが原結果である。
[CD保存field summary](../../../artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/CD_saved_summary_v1.json)は
正確な有理数文字列を保持した閲覧用抽出。event/Pauli表は原結果のJSON pointerから参照する。
全28原結果filesは2,280,394 bytes。原prefixも残し、旧証拠・markerの削除や再生成はない。

[evidence manifest](../../../artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/evidence_manifest_v1.json)に
新資料のbytes/hash、logical identity、scope、STOPを記録する。
GitHub公開後の固定commitからの再取得照合は同audit directoryの`publication_receipt_v1.json`へ保存する。
ローカル実行・保存値監査であり、immutable CIや外部再現ではない。

## GPTへ戻す事項

比較全体はtechnical failureであり、whole/local Pauliの新規性・総資源優位・研究継続を決める証拠は完成していない。
今回の作業は原結果・失敗証拠・正確な来歴の保存で終了する。
次に技術原因調査またはsource変更が必要か、そのscopeと追加実行の価値はGPT/利用者判断へ戻す。
consumed markerの解除、retry、登録fixture変更、precision/cap変更、G10/RA-D0再開、次stage移行はしない。
**mandatory STOPを維持する。**
