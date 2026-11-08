# Track D（O5）C1：head-tail融合の最小人工検証結果

2026-11-08。**ARTIFICIAL_INCREMENTAL_GAIN_OBSERVED。固定q2_mixedで数学的な局所書換えを数値確認し、P0後もStrong・Reference双方に同じ追加削減を観測した。** X/YともRZ−5、CX−2、total depth−7、size−11、qubits不変。**Reference対Strongの相対差は全7資源で変わらない。** 独立新規性や実H4/全RPEでの利益は未確認。

| 構成（X/Y共通） | RZ | CX | RZ depth | CX depth | Total depth | Size | Qubits |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strong S0 | 182 | 94 | 114 | 94 | 265 | 377 | 3 |
| Strong S1 | 177 | 92 | 111 | 92 | 258 | 366 | 3 |
| Reference R0 | 217 | 90 | 77 | 56 | 170 | 428 | 5 |
| Reference R1 | 212 | 88 | 74 | 54 | 163 | 417 | 5 |

## 固定入力・SOURCEと成果物

2-mode/3-block、frame角.23/−.37/.41、scalar .17/extracted identity .04、h=.1/τ=.2/K0、q2_mixed `(plus,zz)→(minus,plus)`、cut1。分子geometry/AO basis/分子DF rank/L_Dは非該当。P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=False/topology-free/MPFR160。

既存StrongBuilder/C3/full-frame Provider/C7-2独立operator/checker/phase compilerを無変更で再利用。head末尾とtail先頭のactual basis native列・wire・gaugeを照合し、異frameはG_T†G_Hへ、同frameはactual inverse pairをidentityへ置換した。中心mode・event順・scalar/identity/χ・normalization・step/cut・CSWAP/readoutは保存。A†は融合後Aのactual inverse。

初回SOURCE **da975efa9c11fcc41602d6acb56e8fd85f2ff514** はlocal PASS後、event保存時の型取り違えで停止。旧partial/STOPを保全して保存処理だけを修正し、継続SOURCE **af6fc07f35dce88d1833163f01781478e5919c7e** で未完了処理を完了した。科学条件・precision・fixtureは不変。局所7 positive/7 negativeは再利用、whole-path raw41比較と全8 postcheckはPASS。新P0は4、既存baseline4をhash/IR/phase/SOURCE/profile/wire/旧post照合後に適格再利用。freshな独立whole pathを追加していない。

- [固定PLAN](PLAN.md)／[機械可読条件](PLAN.json)／[技術継続PLAN](CONTINUATION_PLAN.md)
- [実装・correctness・negative・実行履歴](IMPLEMENTATION_AND_CORRECTNESS.md)
- [全8行・全7資源と差の差](ARTIFICIAL_RESOURCE_COMPARISON.md)
- [次の最小H4移送案（未実行）](NEXT_EXECUTION_PROPOSAL.md)
- [SOURCE](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/SOURCE.json)／[fixture identity](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/FIXTURE_IDENTITY.json)／[呼出数](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/CALL_COUNTS.json)／[summary](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/SUMMARY.json)／[payload hash索引](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/PAYLOAD_INDEX.json)／[保全](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/PRESERVATION.json)／[STOP](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/STOP.json)。

## 成立範囲・未解決事項

全4入力・全reference出力、AB=V/actual A†/vacuum/χ、Strong枝/coherence、X/Y observable、全unitary位相を固定精度で検証した有限binary64証拠。P0後に追加利益が残ったので、このfixtureの全利益が既存compilerで回収済みという仮説は支持されない。一方、異frameと同frameの寄与を個別に切り分ける追加ablationはしておらず、どのpassが何を消したか・任意接口への一般性は未確定。

既知relative Gaussianの接口への応用で、新しいGaussian合成器やSE-QPE干渉機構を発明したものではない。Referenceの既存利点を今回の増分として二重計上しない。RZ→T、topology-free depth→実機時間、人工利益→全law期待価格/N/全RPEへ換算しない。今回H4入力/新sampling/DF fit/ground solve/qモデル/N/13-round更新/詳細科学レビューは0。

## 保全・次の担当・mandatory STOP

旧C7-2原本/source/public索引/STOPと今回SOURCE/inputの前後照合PASS。元main・他worktree・C6a/C7/C8-1/O2のraw/result/grant/ledger/marker/STOPを変更せず、q64のprocess/進捗/未完了出力にアクセス・介入しない。共有OS/venv/cache/scheduler設定を変えない。QPY原本と運用情報は既存C7-2方針どおりlocal専用、公開は科学scalar/compact構造/hash/SOURCE/STOPのみ。

次の担当は**ユーザー／GPT**。保存H4一pathへの最小移送を許可するか判断する。候補は価格結果で選ばないC6a q4_sample00、Strong/Reference両方を同条件で検証する未実行案。新しい科学実行と重要科学レビューの許可は別であり、今回自動開始しない。

専用branch `codex/track-d-o5-c1-head-tail-artificial-20261108` へ公開し、remote SHA・tree・公開bytesを照合後**mandatory STOP**。q64ジョブは今回のSTOP対象に含めない。
