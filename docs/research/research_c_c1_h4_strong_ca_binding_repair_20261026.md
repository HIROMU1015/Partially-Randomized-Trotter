# H4強baseline CA：実行前の契約参照パス修正

2026-10-26 JST。固定86-call契約への科学実行許可後、実行前のmetadata確認で参照パス不一致を検出した。**科学runnerは開始していない。科学call0、fit0、retry0、resume0、run01未作成・未消費。** 旧契約／SOURCE／raw／STOPを保全し、byte同一の保存契約への参照だけを修正した契約v2と人工回帰を公開して停止する。

## 原因と変更

許可された公開起点は `2ca4365b747c9edc65f0209ef9b53a06c3163bc7`。[旧86-call契約](research_c_c1_h4_strong_baseline_ca_resources_contract_draft_20261026.json)のSHA256は `813857ab62dba2cc785814045e4bad8fc6cb617fd2bbbb883beb262ed64c6b7d`。

旧契約の `records.strong_contract` は `artifacts/research_c_c1_h4_strong_baseline/2026-10-26/run01/contract.json` を参照する。一方、保存B3契約の `previous_contract` は `docs/research/research_c_c1_h4_strong_baseline_science_contract_draft_20261026.json` を参照する。両ファイルのbytesは完全一致し、SHA256はともに `e5ca93221b78a846a3c8c7fb56f2caebf6e30a748fd5f7c366b7efc04eb21f7c` だが、固定 `load_bundle` は文字列の一致も要求するため、旧契約のままでは `SAVED_REFERENCE_SHELL_GROUND_BINDING` で停止する。

前回の人工fixtureはB3のprevious pathを新records pathと同じにしていたため、この実資料の差を再現できていなかった。「追加の技術的blockerなし」とした前回報告には、この検証漏れがあった。

[修正契約v2](research_c_c1_h4_strong_baseline_ca_resources_contract_binding_v2_20261026.json)は `records.strong_contract` をB3のcanonical docs pathへ合わせ、その同一bytesのhashをsaved_inputsへ追加する。旧artifact copyのhashも残す。加えてbinding revision／旧契約の来歴を記載した。変更後契約SHA256は **`1bf82b39693f07503eec2879c93cb17a734ccdfdcdca03b50544a15444a56cd2`**。

**科学的な数値SOURCE／runner／既存testsを一切変更していない。** 実装SOURCE `969c17d623d56eb645417b281deadb1be4757699`、SOURCE manifest31件とそのSHA256 `881694bde50cfa1ad50a67fef86c0bffc42c165559a6f8be56e148573e7cad8c` は不変。物理Hamiltonian・DF・PF・κ・scalar・frame順・precision/P0・RPE／shots式・CA・費用政策・allowance・候補順・科学callの定義と上限も不変である。

## 人工検証・passive preflight

[新規回帰test](../../tests/management/test_research_c_c1_h4_strong_ca_binding.py)は、同一bytesだが異なるcanonical／artifact pathを持つ人工証拠を作り、旧bindingのFAILを再現した上で、canonical参照に直した人工2候補×13roundの86-call経路が完了することを確認する。実資料はpath／hash／metadataと既存passive preflightだけで照合し、実B2/B3の数値join・合成伝播・CA・shots・費用を評価していない。

新規2件＋前回76件の**78 tests PASS（2.27秒）**。[final log](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/binding-repair/tests_final.txt)。最初のbatchは新testへの既存fixture import不足で77 PASS／1 setup errorとなり、import追加後に全78件がPASSした。旧fixture／旧testsは変更していない。

修正契約を既存CLIへ渡した結果は `BYTE_PREFLIGHT_PASS, science_calls=0, maximum_calls=86`。旧契約・SOURCE・rawを再実行／上書きしていない。[技術確認・STOP](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/binding-repair/)。共有環境・venv・他ユーザーのjobへの変更は0。試験cache／tempは独立worktree内のtask専用領域を使用した。

## 固定範囲と次の担当

将来の対象は直鎖H4・1.00Å/STO-3G、Nα=Nβ2/sector36、保存B2 CDF8対B3 XDF8、全frame ordered S2 s1のみ、δ=.1／m0..12、PRなし、own-DF ground oracle／ideal IID Born／状態準備除外、共通CA=.00159 Ha・DF reserveCA/8、既存β/failure/P0／serial費用政策である。最大86 logical units、retry/resume0。回路／Operator再生成、fit、DF/PF再計算、s4や候補拡張は0。

今回、合成誤差／CA／shots／総資源の新しい科学結果は取得していない。保存証拠はlevel B数値pilotであり、厳密rounding enclosure、厳密CA、実機・状態準備込みの総資源優位は未資格のままである。

旧固定契約のSTOPは「First correctness/SOURCE/input mismatch stops dependent calls」と定め、AGENTS.mdも「既存の固定契約を無断変更しない」ことを要求する。したがって、byte同一の参照先への技術的修正でも、旧契約を黙って置換して科学実行を開始しない。**ユーザー／研究C**に公開した修正契約v2で同じ86-call範囲を実行してよいか確認する。**Codex**はその許可後に限り、未消費のfresh run01を一度実行する。承認証明書・承認原文保存・新grantは不要。

Branch：`codex/research-c-c1-h4-strong-ca-binding-repair-20261026`。独立worktree：`/tmp/prt-c1-h4-strong-ca-binding-repair-20261026`。commit・push・remote照合後mandatory STOP。旧契約による消費済み科学run／retryは発生していない。
