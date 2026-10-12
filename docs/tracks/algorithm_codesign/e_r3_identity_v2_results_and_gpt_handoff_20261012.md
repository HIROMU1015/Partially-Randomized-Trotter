# E-r3 identity-only v2：一回実行・保存値監査・GPT引き継ぎ

2026-10-12 JST。**E_R3_IDENTITY_DIAGNOSTIC_COMPLETE / mandatory STOP**。
登録runner一回、retry0。外側process exit0、child exit0、terminal COMPLETEが一致した。
今回のcurrent／legacy再構成では全matrix-stage bytesが一致し、recursive payload差は自動生成operation名4箇所だけだった。
両actualは旧expectedと不一致だが、旧実行のactual/payload/入力行列bytesが未保存なので、旧停止原因を一意に確定しない。
A/B科学比較の完了・優劣やfingerprint policy修正の採択ではない。

## 固定source・認可・来歴

| 役割 | identity |
| --- | --- |
| Repository | HIROMU1015/Partially-Randomized-Trotter |
| 固定source S2 | `b9519c29b5f1528ad5c8a6e398b2fa12062db8c6` |
| source公開確認commit | `e3b18413aa56748bb926c0992b2bf83acfc24198`、実行HEADには使用せず |
| authorization-only直接子 A2 / 実行HEAD | `5e8baf475dd086e85e4b2611cc20b1f77161e7d5` |
| execution branch | `track-b-e-r3-identity-only-v2-one-shot-execution-20261012` |
| contract SHA256 | `12345d68a1d3d50869f2928d9a441073fce8e9e82f66c73856c6956686a5f488` |
| source manifest SHA256 | `359c606b13a04dbba5991baa3415c3a9c762956a77891c170d3d1e019ac99e8c` |
| authorization JSON SHA256 | `bd99c562799ad04155fd5afa27b5f62423370ae2ca9bfe15514591fcbd22cbd6` |
| 消費済みone-shot marker SHA256 | `f6432a982a414d970972883fbf40f4901b682834582b325d79feef2e4336dca5` |

[authorization receipt](e_r3_identity_diagnostic_execution_authorization_v2.md)は最新指示の原文
「次の作業に進んで」と、その直前のS2 one-shot承認待ち提案との文脈を記録した。
以前の同文指示が認可したobserver修正とは別turnであり、以前の指示を実行認可へ流用していない。
A2の変更は新authorization JSON＋receiptだけ、parentはS2一つだけ。
diagnostic_execution_authorized=true、science_execution_authorized=false、runs1/retry0/STOPtrue。

[実行前監査](../../../artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v2/launch_audit_v2.json)で
HEAD=A2、remote SHA=A2、clean、319保護files、contract/manifest/input identities、固定runtime、fresh namespaceを確認。
markerより前には登録Q/T/U/payloadを取得していない。
旧比較source S' `41349b73157c7781e0b534cf0fe61e06bf439954`、旧A `a1f537b68f0f22419dff0202c553b7511653caa3`、
旧result公開 `ce0e286430a2a23869cb9417c82113c98064c72c` と旧marker
`dd6b9d5ec96e9d34d7bea0277eadb03f031992e1010cfdb2c940a393b2242701`は不変。
旧science classification `TECHNICAL_INCONCLUSIVE`を変更しない。

## 実行範囲と記録

保存済みE-r3、3 mode、rank2×3、uncontrolledだけ。
signs=(+1,-1,+1)、`comm(Q0,comm(Q1,Q2))`、`expm(1j*.03125*T/4)`。
current→legacyを各一回、同じ保存frame・固定runtime・precisionで再構成した。
legacyは旧helperの三関数だけの再構成で、historical runtime/layoutの完全再現ではない。
name counterのreset、名前を除いたhash、新expected、別入力・compileを採用していない。

各pathの保存stageはframe0/1/2、Q0/1/2、T、U、input target、actual gate parameterの10項目。
全C-order raw bytes・dtype/shape/layout・hashと、全recursive payload・definition index・exceptions・counter contextを保存した。
payload attempts合計2、UnitaryGate `_define`合計2、QSD Python entries合計10。
constructor/expmは固定sourceの一path一回に従う2経路であり、別のmatrix probeを行っていない。
observer設置・復元のverificationは両経路PASS、fingerprint/observer例外0、native blocked entry0。

## 保存されたhashと差分

旧expected：`171f321e2f3bcc51b177e7b62735b5fbf9d80e36a5f56639b7f03809efac41b5`。

| path | observation | actual fingerprint |
| --- | --- | --- |
| current | MISMATCH、非null | `406640bb455fae569df5f41c0079ba2d90fbc34993b1765ceca8b49714cdb1d4` |
| legacy | MISMATCH、非null | `73965b732709af19321489dd1a4dbdfb899ad39743b54ba5c60e94da2b0bfb7e` |

[matrix-stage identity表](../../../artifacts/track_b_e_r3_identity_diagnostic_execution_audit/2026-10-12/v2/matrix_stage_identity_v2.csv)：
全10 stageでdtype/shape/raw bytes一致。first_different_stage=null。
各U、input target、actual parameterのSHA256は同じ
`744d3c221bd0064dded5032aec24cd13d8f0f718ceb87d63acbda410b8fa0e77`。

[全payload差分](../../../artifacts/track_b_e_r3_identity_diagnostic_execution_audit/2026-10-12/v2/payload_auto_name_differences_v2.json)：
いずれも`/instructions/0/operation/definition/instructions/<index>/operation/name`。

| instruction index | current | legacy |
| ---: | --- | --- |
| 0 | circuit-230 | circuit-311 |
| 8 | circuit-233 | circuit-314 |
| 18 | circuit-236 | circuit-317 |
| 28 | circuit-239 | circuit-320 |

この4 field以外のrecursive payload値には差がなかった。
root circuit名はcurrent `circuit-162`、legacy `circuit-243`。
QuantumCircuit.instancesはpayload前/後current163/243、legacy244/323で、固定順序でcounterが進んだ。
これらはfresh childの現在の構成contextであって、当時のcontextへ値を戻したものではない。
名前差を消した受理hashやsemantic acceptanceは計算・採択していない。

確認できた技術的事実は、**この保存frameと現在の固定runtimeでは、二経路のmatrix bytes差が現れず、
recursive自動名だけで二経路のfingerprint差が生じる**こと。
旧expectedとの差がその4 fieldだけによるとは検証できない。旧expectedの完全payloadがないためである。
旧float DAG／当時のlayout／例外などを今回の観測だけで一般的に否定しない。

## CPU・RSS・出力・終了status

1 worker/threads1、QISKIT_PARALLEL=false、GPU query/use0。
CPU32・MemAvailable約51.8 GiBを実行前metadataで確認。
内部QSDが主要work候補で、counter/contextを守るため2経路を並列化しなかった。
backend、精度、資源上限を変更せず、speedup claimをしない。

| 会計 | 実測 |
| --- | ---: |
| outer runner wall | 3.854700522 s |
| outer reaped runner-tree CPU | 2.023852 s |
| outer sampled runner descendants RSS peak | 382.16015625 MiB |
| source Supervisor wall | 1.734450231 s |
| source Supervisor CPU(parent/reaped) | 1.944898 s |
| source Supervisor aggregate RSS peak | 382.16015625 MiB |
| worker evaluation wall / CPU | 0.986477107 / 0.982905308 s |
| worker peak RSS | 352.63671875 MiB |
| outer stdout / child log | 98 bytes / 0 bytes |

outer wallはrunner全体、source Supervisorはその生成以降、workerはruntime/input gate後の評価期間なので範囲が異なる。
RSSは50 ms sampled、瞬間的無overshootを主張しない。
CPU total120/worker90 s、RSS aggregate768/worker512 MiB、output8 MiB/log1 MiB、wall capなしを維持した。
cap、numeric/observer/storage failure、partial/incomplete、retryは0。
one-shot markerとexclusive identity.started receiptは一組だけ。

## 保存値監査と資料取得

[read-only auditor](../../../scripts/tracks/algorithm_codesign/audit_e_r3_identity_v2_saved_outputs.py)はstdlibだけを使用。
NumPy/Qiskit/worker/runnerをimportせず、新Q/T/U/payloadを生成しない。
[保存値監査421項目](../../../artifacts/track_b_e_r3_identity_diagnostic_execution_audit/2026-10-12/v2/saved_output_audit_v2.json)は全てPASS。
marker/child/result/terminal/outer source binding、stored payload hash、raw array bytes hash、
保存hexからのframe byte表現、全差分、caps、319保護files、旧result28filesを照合した。
原runner8ファイルのinventoryを外側receipt受領より先に取得し、後にもbytes不変を確認した。
元のJSON・marker・terminalを上書きしない。

一次データは[結果directory](../../../artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v2/)の
`current.json`157,610 bytesと`legacy.json`154,570 bytes、`identity_result.json`、marker/terminal。
全JSONは512 KiB以下で、GitHubから全payloadを直接取得できる。
[外側receipt](../../../artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v2/outer_process_receipt_v2.json)、
[独立監査directory](../../../artifacts/track_b_e_r3_identity_diagnostic_execution_audit/2026-10-12/v2/)とevidence manifestを併せて読む。
資料はcommit/push後にremoteから再取得し、bytes/SHA256、remote SHA、cleanを確認する。
これはsource-bound local diagnostic evidenceでありimmutable CIや外部再現ではない。

## STOPと次の判断

**mandatory STOP。登録run1、retry0、追加Q/T/U/payload取得0。**
native transpile body、CD再評価、E-r6、controlled、Pauli費用、T合成、sampling/trajectory/LP/DF/分子/NPZ/GPUは0。
旧one-shotの再試行、expected更新、hash guard緩和、旧float実装採択、全A/B再実行は行っていない。

次の判断へ渡す証拠は、matrix-stage全一致／payload自動名差4／両actual非null MISMATCH／正常observer／旧expected payload欠測。
これを使って旧guardや実装方式を変更してよいかは今回採択しない。
fingerprint policy・科学的な受理条件・全A/B再実行に関わる新判断が必要ならGPT側へ戻す。
科学的なA/B結果評価や研究方針の全面再設計はこの診断だけで開始しない。
GPTレビューの必要性と開始承認は共通指示に従い区別する。

凍結manifestに含まれる既存indexes/synthesisは変更せず、新報告・日付note・監査manifestを入口として公開する。
結果公開commit／publication確認commitは実行HEAD A2と区別する。
新source修正や新診断を自動で行わない。
