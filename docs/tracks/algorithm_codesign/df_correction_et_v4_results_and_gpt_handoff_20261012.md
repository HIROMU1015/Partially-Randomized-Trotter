# E/T v4 one-shot：output cap technical STOP・保存値監査

2026-10-12 JST。**TECHNICAL_INCONCLUSIVE / mandatory STOP**。
固定sourceから一回だけ実行し、output/log guardで停止した。E-r3 child exit0、E-r6 childはguardによるSIGKILL（exit−9）。外側runner exit2とterminal/stdoutの分類が一致する。retry0。
E/T比較全体は未完了で、A/B科学的勝敗、新規性、PR総資源優位は判定しない。

## 固定source・認可・実行来歴

| 役割 | identity |
| --- | --- |
| source S | `016e02b486cd4b182b654af18743ef1ef50fe7e6` |
| authorization-only直接子A / 実行HEAD | `111199c45a9c37ac81821512ea8a44e6093f50cc` |
| branch | `track-b-df-correction-et-v4-one-shot-execution-20261012` |
| contract SHA256 | `c2d2c91004e6fbe78b08a38620bc96f1d099d8cdfb514abfcdedb267cab3a77b` |
| source manifest SHA256 | `94d7013e7e6a34e2cd2a2efc409b832b03176660e31a95aaee1a4b36e993a6b8` |
| authorization SHA256 | `9e20d2e41202a69d793335874c6c6c3221933a137985dca2fb4d48c4bf9b63d0` |
| 消費済みmarker SHA256 | `446d97e0f1744285ed21fbfff6e349b3f06c65ee0959af0f605dff9bb0c19cbb` |

[authorization receipt](df_correction_et_execution_authorization_v4.md)とJSONは利用者の今回の明示実行指示を記録した。AのparentはS一つ、変更は新authorization JSON＋receiptだけ。source公開確認HEADは実行に使わない。
[実行前監査](../../../artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v1/launch_audit.json)でremote=A、clean、manifest369files、契約・保存入力・固定numeric/synthesis runtime、新namespace不存在を確認した。CPU32/affinity32、memoryをread-only記録、GPU query0。

新result namespace：`artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v1/`。
raw原結果とchunk/indexを全284files保存し、消費済みmarker・job-start・terminalを上書きしない。
旧A/B比較TECHNICAL_INCONCLUSIVE、旧E-r3 identity診断COMPLETE、旧source/authorization/marker/STOPは保持。

## 実行scope・到達点

保存済みE-inputsだけ、seed7101107由来のr3/r6、rank2×3、signs(+,-,+)、`comm(Q0,comm(Q1,Q2))`、u=1/128。
各rankでuncontrolled→MSB controlled、Bは固定support-weight代表。basis rz/sx/x/cx、Qiskit1.3.0、opt0/seed20261011、既定precision/strict phaseを維持。
2 native workers、各numeric threads1、QISKIT_PARALLEL=FALSE。r3/r6の並列構成と既存guard/storage helperを使用。compilerやfixtureの再選択は0。

| 記録 | 状態 |
| --- | --- |
| completed stages | `CD_SAVED_REFERENCE_ONLY` |
| CD | 保存済み6行を来歴付き参照。再計算0、旧run再分類0 |
| E-r3 | child完了。native予約8、保存native rows8（A2/B6） |
| E-r6 | 未完了prefix。native予約2、保存native rows1（A uncontrolled） |
| native cost-return観測 | 合計9 |
| E-r6 controlled | raw recursive payload/hash取得。committed stateはFINGERPRINT_COMPLETE、transpile_attempts0 |
| native予約の保存合計 | 10（上限22）；完了native bodyの観測9と区別 |
| T inventory / keys / synthesis | すべて0（上限64未使用） |

r3完了substageも全体未完了run内の診断証拠であり、prefixを最終positive/negative研究判断へ使わない。r6 controlledの内部UnitaryGate definition/QSDはpayload取得に必要な仕事として行われたが、native transpile bodyとは区別する。CPU/RSS/outputは全worker workを監視した。

[layered観測表](../../../artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v1/stage_observation_summary.json)には原native IRと費用のidentity、保存numeric guard、無加工actual/expected hashを索引化した。取得したAの旧hash不一致を理由に停止した記録はない。取得済み9native rowの数値allowanceは保存値上適格。r6 controlledのnative数値確認、E全体、T coverageは未取得。

## 停止理由・保存構造

固定output64MiB＝67,108,864 bytes、terminal予約256KiBを差し引くactive guardは66,846,720 bytes。log cap4MiB。
保存後の原result inventoryは**67,637,141 bytes**、logは0 bytes。output超過であり、log超過ではない。
この値は終了後のdisk snapshotで、最初のguard tripの厳密な瞬間のbytesを逆算した値ではない。

[storage accounting](../../../artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v1/storage_accounting.json)でactive review indexesを完全復元・SHA照合した。

| 大きなcommitted logical artifact | bytes |
| --- | ---: |
| E-r6 prefix | 11,915,408 |
| E-r6 A uncontrolled raw observation | 12,525,042 |
| E-r6 A controlled raw observation | 32,718,729 |
| r6 controlled observationのunreferenced replacement chunks | 7,869,608（31 files） |

未参照chunksは同じcontrolled observationの別generationだけにある。既存のcommitted indexはFINGERPRINT_COMPLETEを指し、追加generationは未公開。全bytesを失敗証拠として保持し、未公開generationをcomplete resultへ復元・救済・解釈しない。

静的sourceでは、`dump_json`は旧generationを保持したまま新generationを書き、完成・復元確認後にindexをpublishする。raw adapterは同じpayloadを含む観測をFINGERPRINT_COMPLETE→BEFORE_TRANSPILEへ保存更新し、保存が戻ってからnative transpileを呼ぶ。
**大きなcontrolled raw観測の重複保存中の世代併存がoutput pressureの直接的な候補**であり、保存snapshotとsourceの順序が対応する。exact allocation/guard瞬間のtraceはないため、33MBの内容すべてが不要だったとは断定しない。保存圧縮・差分state・payload分離等の実装修正は今回採択・実施せず、capも増やさない。

## 実行資源・外側process

| scope | 実測 |
| --- | ---: |
| outer full runner wall | 36.097764267 s |
| outer reaped runner-tree CPU | 42.243528 s |
| outer sampled aggregate RSS peak | 734.015625 MiB |
| source Supervisor wall | 33.526425685 s |
| source Supervisor parent/reaped CPU | 41.716667 s |
| source sampled aggregate RSS peak | 734.015625 MiB |
| outer stdout / child logs | 130 / 0 bytes |

[外側receipt](../../../artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v1/outer_process_receipt.json)と[raw stdout](../../../artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v1/outer_runner_stdout.log)を別audit namespaceへ原bytesのまま保存。sourceの出力会計を後からこれらで書換えない。
外側wallにはpre-marker launch/runtime gateが含まれ、Supervisorは生成後のscope。CPUはOSのparent/children/reaped会計、RSSは50ms sampledで瞬間無overshoot保証ではない。
wall/CPU/RSS超過理由や数値・fingerprint例外は記録されていない。固定cap：native wall600/CPU450/worker RSS1024/aggregate2560、T wall1200/CPU900/worker512/aggregate1536、total wall2400/CPU1800/aggregate3072を維持。

## read-only監査と取得性

[stdlib専用auditor](../../../scripts/tracks/algorithm_codesign/audit_df_correction_et_v4_saved_outputs.py)で、保護369files、原284files、source→A、marker/child/outer/terminal、CD保存値、E入力、全committed chunksの順序/UTF-8/bytes/SHA、raw payload/array hashes、native IRのwire/angle/phase/count結合、保存numeric guardを照合した。
[保存監査1533項目](../../../artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v1/saved_output_audit.json)は全PASS。final audit exportの512KiB確認は別1項目（auditor stdoutは合計1534と表示）。
最初のauditorはraw JSON globがreview indexも選んだためKeyErrorで停止した。[初回記録](../../../artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v1/audit_attempt1_failure.json)を保持し、post-STOP auditorの選択filterだけ修正した。原run/source/resultの修正・科学retryではない。

原result全内容はraw JSONまたは512KiB以下のindex＋256KiB以下のUTF-8 chunkで取得できる。partial generationも[original inventory](../../../artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v1/original_run_inventory.json)にbytes/SHAを記録して保存する。
新資料のmanifestと公開receiptにresult commitとremote取得照合を記録する。result/publication commitは実行HEAD Aと区別する。
これはsource-bound local technical failure evidenceでありimmutable CI/外部再現・有効な完了A/B比較ではない。

## mandatory STOP・次の判断

登録runner1、retry0。marker消費済み。source/contract/manifest/authorizationと旧result/marker不変。
T合成、追加compile、CD再計算、identity再診断、source修正、output cap変更、fixture追加、G10/RA-D0/DF/分子/NPZ/GPU/LP/trajectory/samplingへの進行はない。
原結果と未参照chunksを移動・削除・再生成せず、後続科学実行を開始しない。

次へ渡すのはoutput保存の技術的制約と未完了範囲の証拠。A/Bの利点・不利、whole方式やTrack Bの継続可否は今回確定しない。追加技術修正のscope／新source／新one-shotが必要かは利用者/GPT判断へ戻す。追加GPTレビューと計算承認は別。
凍結sourceの既存indexes/synthesisは変更せず、この新報告・日付note・audit manifestを公開入口にする。**mandatory STOP。**
