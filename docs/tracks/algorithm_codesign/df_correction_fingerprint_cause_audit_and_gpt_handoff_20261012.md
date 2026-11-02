# DF補正比較：E-r3 input fingerprint停止の限定技術監査

2026-10-12 JST。保存済みsource・記録だけを独立branchで監査した。
停止点は確定したが、**fingerprint差を生んだ具体的な数値・実装原因は未確定**。
比較する指標はtranspile後のnative gate列ではなく、**transpile前の入力回路のraw数値と再帰的gate definitionのhash**である。
source上の浮動小数演算順序・frame受け渡しに差を確認した。これらだけで実際のbyte差や物理的誤りを断定しない。
旧科学分類 **`TECHNICAL_INCONCLUSIVE`**、one-shot消費済み、retry0、mandatory STOPは保持する。

今回の利用者指示「次の作業に進んで」は、直前に提示した保存資料による技術原因監査への認可として扱った。
production source変更・compile・synthesis・runner再実行は認可範囲に含めず、実施していない。

## 正本とscope

| 項目 | 固定identity |
| --- | --- |
| 公開済み結果・監査の基点P | `ce0e286430a2a23869cb9417c82113c98064c72c` |
| 元結果commit R | `8f1b81d0589e5f496608de4cc09939412c573275` |
| science source S' | `41349b73157c7781e0b534cf0fe61e06bf439954` |
| authorization-only A | `a1f537b68f0f22419dff0202c553b7511653caa3` |
| 旧Phase E source commit | `b8c4c446d3b913fa5c85b568dcac2051f4054c1e` |
| 原実行marker SHA256 | `dd6b9d5ec96e9d34d7bea0277eadb03f031992e1010cfdb2c940a393b2242701` |
| 監査branch | `track-b-df-correction-fingerprint-cause-audit-20261012` |

正本は[既存handoff](df_correction_comparison_results_and_gpt_handoff_20261012.md)、
[terminal](../../../artifacts/track_b_df_correction_comparison_result/2026-10-11/v2/terminal_status.json)、
[E-r3 traceback](../../../artifacts/track_b_df_correction_comparison_result/2026-10-11/v2/E-r3.log)、
[旧Phase E](../../../artifacts/track_b_df_sparse_correction_audit/2026-10-11/phase_E_checks.json)。
input frameとjobは保存済みファイルを照合し、登録frameの再生成・array構築・行列演算を行っていない。
sourceのASTはparseのみ。science moduleをimportせず、stdlibでmetadataと保存値のidentityを検査した。
原実行worktreeとTrack Aは編集していない。

## 1. 停止点で確定した事実

E-r3のprefixは`compile_calls=1`、accepted native recordは0。
`native_fixture`はwhole uncontrolled→controlledの順に進み、旧r3/uncontrolled行を取得する。
旧期待fingerprintは `171f321e2f3bcc51b177e7b62735b5fbf9d80e36a5f56639b7f03809efac41b5`。
返却された`cost.actual_circuit_fingerprint`がこの値と一致せず、`native_evaluation.py:87`で停止した。
外側exit2、E-r3 exit1、保存classificationとstdoutが一致する。
並列E-r6はsibling failure後SIGKILL。E-r6のin-flight compile数は未保存、T合成は0。
`CapFailure`はsupervisorのchild failure処理であり、RSS/wall超過を示す理由ではない。

`transpile_and_measure_cost`では、入力`circuit`をfingerprintに渡してから`transpile(circuit,...)`を呼ぶ。
fingerprintはndarrayのdtype/shape/全raw bytesのSHA256、operation型・パラメータ・global phase・qubit順序、
再帰的`operation.definition`を含む。native T/CX/RZ数やoperator-norm同値性のcertificateとは異なる。
`operation.definition`の参照はUnitaryGateの分解を生成し得るため、単なる行列hashでもない。
本監査はこのpropertyを評価せず、sourceを読むだけに留めた。

**もう一つの未保存分岐**：fingerprint生成で`TypeError`/`ValueError`が出ると共通helperは`None`へ落とす。
その場合も今回と同じ比較guardで停止する。actual値も当該例外も未保存のため、
「非null hashの違い」だったか「fingerprintが作れずNone」だったかは区別できない。

## 2. sourceを対応させた結果

[source比較・行番号付きAST/excerpt](../../../artifacts/track_b_df_correction_fingerprint_cause_audit/2026-10-12/source_comparison_v1.json)が検算の正本。

| 項目 | 確認結果 | 限界 |
| --- | --- | --- |
| RNG / draw順序 | 両sourceともseed7101107、r4 draw1回→r3 draw3回→r6 draw3回。QR phase調整のASTもlocal import以外一致 | 旧frame値自体は保存されていないため、実frame bytes同一は未証明 |
| current frame受け渡し | E-inputs→E-r3/r6 jobのhex文字列は完全一致し、scalar hex round tripも一致 | 旧sliceとcurrent deserialize後のarrayのownership/stridesは異なり得る。実stride記録なし |
| primitive符号・代数 | `(1,-1,1)`、同じCAR/JW構成 | 浮動小数演算の括弧と係数2の適用位置は一致しない |
| triple commutator | `comm(Q0,comm(Q1,Q2))`、`AB-BA`の符号・順序一致 | Ball.value/旧raw generatorのbyte比較はできない |
| whole回転 | `expm(1j*.03125*T/4)`、t.valueへ置換したASTは同一 | Tのbytesが同一という保証にはならない |
| controlled assembly | False→True、Trueは`block_diag(I,raw_u)`、UnitaryGate/qubit順序一致 | 失敗したのは最初のuncontrolledで、controlled結果は未取得 |
| compiler | constructor AST一致、basis rz/sx/x/cx、opt0、all-to-all、seed20261011、Qiskit1.3.0 | input fingerprint計算はtranspileより前。seedを変える解決根拠はない |
| versions / source | Python/NumPy/SciPy/Qiskitの記録version一致、旧helper・fingerprint policy sourceも旧commitから不変 | 旧package/binary treeは保存されておらず、完全なhistorical runtime一致は未証明 |
| prior row選択 | r3・controlled=falseを照合するsourceとprefixが整合 | actual fingerprintと入力matrixが未保存 |

sourceから復元したcompiler settings hashは旧全4行と一致する
`59f18e1e5165d7a9ac8edb657b3e8e50812974c7e5936dc4e1a0634e199938b6`。
これはmetadataのhash検算であり、失敗compileが保存していないruntime費用を補完したものではない。

旧primitiveのPython ASTは、括弧内部を **`(2 * a.conj().T) @ a`** と解釈する。
current Ballは`b.adjoint().mul(b).scale(2)`なので、valueの経路は **`float(2) * (a.conj().T @ a)`**。
厳密代数では同じであり、通常のbinary64範囲では2倍自体はexactである。
このsource差だけで「roundoffが必ず違った」と言わない。BLASのoperand状態・layout・signed zero等を含めて
registered inputの同一bytesを保証する証拠が不足している、という限定した結論である。

保存済みfocused testは、2-modeの特定rank2 frameと単純なcommutatorで旧/newの`tobytes()`一致を示している。
有効な限定controlだが、r3/r6のregistered frames・whole expm・recursive fingerprintまでの一般保証ではない。
このtestも今回再実行していない。

## 3. 原因候補と確信度

| 候補 | 根拠・確信度 | 今回確定できない点 |
| --- | --- | --- |
| raw generator / buffer経路の差 | source構成差の存在は確定。実hash差の原因としては未確定の検査点 | 旧/current Q・T・raw_u bytes、operand layoutの観測値がない |
| recursive UnitaryGate definitionの差 | hash policyがdefinitionも含むことは確定 | 入力matrix hashとdefinition hashを分離した記録がない |
| fingerprint生成不能→None | 共通helperのfallbackは確定 | actual値・例外が未保存で、今回この分岐だったか不明 |
| compiler設定・tuple/sign/時間の取り違え | source上は一致を確認。原因として支持する差分なし | historical全runtime一致、registered全中間値を証明したわけではない |
| RSS/wall上限超過 | 原因として支持する記録なし。観測停止はfingerprint guard | sampled RSSから瞬間的非超過を証明することはしない |

**最も有用なtechnical findingは、原因を一つに断定できたことではなく、
input identityの検査点と未保存分岐を特定し、確証のないcompiler/cap変更を避けられることである。**
回路が物理的に誤っていた、誤差予算内だった、A/Bが有利だった、という結論は今回出さない。

## 4. 修正候補（提案のみ）

| 候補 | 変更の種類・期待効果 | 必要なvalidation / 残るリスク / 承認 |
| --- | --- | --- |
| failure前の診断保存を追加 | science意味論を変えない観測追加。actual値（None含む）/expected、入力のdtype/shape/bytes hash、frame layout、compiler hash、guard locationを保存 | 分割/size/cap内保存とfailure-path testsが必要。原因自体は直さない。別source準備scopeと新manifestが必要 |
| raw whole構成を旧float DAGへ合わせる | 同じ理想generatorを狙い、旧byte identity保持を意図する数値実装修正 | 原因未確定なので成功保証なし。frame materializationも確認し、変更したDAGのassembly enclosure・strict controlled phaseを再検証。Bのproxyと同じidealへ距離を戻す必要がある。別source準備・差分review必須 |
| fingerprintをsemantic guardへ置き換える | **契約変更**。近いoperatorを許容できる可能性 | 旧fingerprintのnative dictionaryとの関係が変わり、異なるnative単価を認め得る。GPT/利用者の明示判断、source/contract reviewが必要。単なるguard削除として採用しない |
| runtime/BLAS/definition identityを追加固定 | 再現性metadataの強化 | 旧runtimeの復元可能性は未確定。version一致だけで旧実行を復元したと扱わない。変更scopeと必要性の判断が必要 |
| cap拡大 / compiler seed・精度変更 / expected値更新 | 資源契約または比較条件の変更 | 今回の証拠から解決策として支持されない。今回実施せず、単に実行を通すために採用しない |

現時点の技術的推奨は、**診断不足を埋めるsource準備とoff-domainのold/new全構成経路検証を、必要な範囲として次に審査すること**。
登録fixtureを今ここで再compileして不一致を再現しない。科学的条件を変更しない修正候補でも、
変更箇所の数値certificateが成立することと、旧identity guardを満たすことは別に検証する。
どの候補を実装するかは未採択で、この報告に実装認可はない。

## 5. 修正・再実行を検討する場合の管理

必要であれば新science source S''、新source manifest、必要な新versionのlaunch/result namespaceを準備する。
旧S'/A/marker/terminal/prefix・過去fingerprintを不変で保持し、旧result pathを上書きしない。
source review通過後も、新S''の直接子authorization-only commitと別の明示one-shot指示が必要。
旧Aの認可や消費済みmarkerを新実行へ流用しない。

fixture/seed/time/delta/rate/precision/error/confidence/baseline/call capsは結果後に調整しない。
変更が避けられない場合は比較条件の変更としてGPTへ戻す。
failure保存はguard判定前に行い、実matrix/compile/keyごとのprogressを残す案をreviewする。
追加記録は既存memory/output/log制限内で、bounded streaming/chunkingを用いる。
再失敗した場合もpartial prefixから科学的勝敗を作らず、no retryとSTOPを維持する。
今回はこれらを提案として文書化しただけで、source修正・新authorization・新marker・再実行は0。

## 6. 監査証拠と停止

[stdlib inspector](../../../artifacts/track_b_df_correction_fingerprint_cause_audit/2026-10-12/inspect_saved_sources.py)、
[source comparison](../../../artifacts/track_b_df_correction_fingerprint_cause_audit/2026-10-12/source_comparison_v1.json)、
[保存frame・失敗metadata](../../../artifacts/track_b_df_correction_fingerprint_cause_audit/2026-10-12/saved_input_and_failure_metadata_v1.json)、
[input provenance/checks](../../../artifacts/track_b_df_correction_fingerprint_cause_audit/2026-10-12/input_provenance_and_checks_v1.json)、
[manifest](../../../artifacts/track_b_df_correction_fingerprint_cause_audit/2026-10-12/evidence_manifest_v1.json)を公開する。
保護source216 files、原結果28 files、authorization/contract/markerのidentityを確認した。
静的ASTと保存値の監査のみ。新science/matrix/circuit/compile/synthesis/sampling/LP/分子/NPZ/GPU照会は0。
原科学分類は不変であり、独立外部再現・immutable CI・科学的GOではない。

監査完了後 **mandatory STOP**。
次に限定source準備へ進めるか、どのvalidationを認可するかをGPT/利用者へ返す。
原因の未確定部分を残したまま、guardを外す・expected値を書き替える・one-shotを再試行することはしない。
