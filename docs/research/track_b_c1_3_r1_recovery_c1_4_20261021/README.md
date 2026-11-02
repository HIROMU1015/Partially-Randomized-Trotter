# C1-3-R1 recovery — state修正PASS、reference生成で技術STOP

**Phase A PASS / Phase B `C1_3_TECHNICAL_INCONCLUSIVE_STOP` / Phase C（C1-4）未着手。** 新規科学実行は1回、retry 0。6 eventのoperator・X/Y比較は未実施。SOURCE修正・再実行は行わずmandatory STOPする。

| 確認対象 | 結果・証拠 | 範囲 |
|---|---|---|
| State hash修正 | [state_identity.json](result/state_identity.json)でR1期待値とactualが一致 | dtype.str・canonical shape・C-order bytes。期待hashは変更していない |
| Phase A | [phase_a_checks.json](phase_a_checks.json)、[人工operator確認](artificial_checks.json) | 8人工hash fixture、failure/terminal/logger、既存人工controlled比較10件と63-event first moment。H4 inputを開かず確認 |
| H4入力復元 | [raw stdout](result/raw_stdout.log)のR1_RESTORE通過 | R1新exact model、13保存basis、324label、正有理pの処理。全label/eventの動的正しさの保証ではない |
| State norm・schedule | [schedule.json](result/schedule.json) | norm誤差0、T=4/5、q=1、r=4、各tail microstep 1/5、sigma=+1、short-step確認 |
| 有限bit preflight | [preflight.json](result/preflight.json)でP0〜P5全PASS、ready=true | 固定4096/2048bit・eta/rho=10^-12で既存F1/F3条件と実生成lawの結合を通過。全parent列挙なし。物理provider・fresh IIDは未認証 |
| Reference basis | [failure.json](result/failure.json)、[terminal.json](result/terminal.json) | `saved_basis`の`REFERENCE_LOCAL_BYTES_MISMATCH`。固定event queryを開始する前に停止 |
| C1-4 | NOT_STARTED | Phase B必須Gateを完了していないため進まない |

対象はlinear H4 1.00 Å/STO-3G/DF12、R1新partition、8 system qubit、13 basis/324 label、m=5、outer symmetric second-order PF。科学条件・6 event・角精度・数値基準・reference定義は旧契約と同一で、[contract_comparison.json](contract_comparison.json)が同一性を記録する。旧PR-2と135係数のbinary64が異なり、同じ科学入力とは主張しない。

State修正の成立と、H4全体のcorrectness PASSを区別する。今回得られたのはstate/input/scheduleと条件付きF1/F3 preflightの技術証拠まで。selected operator、relative phase、root/angleのeventごとの結合、X/Y期待値、全path/round、native、分子energy、資源優位、新規性、投稿GOは未認定。

新SOURCE [`9a3503375c07037ab3e0b9428a7c0254b66c31f6`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/9a3503375c07037ab3e0b9428a7c0254b66c31f6/docs/research/track_b_c1_3_r1_recovery_c1_4_20261021)、新authorization [`8c892129f0d16211ee2fd2f3eafafeba4f181bd7`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8c892129f0d16211ee2fd2f3eafafeba4f181bd7/docs/research/track_b_c1_3_r1_recovery_c1_4_20261021/authorization_receipt.json)。旧失敗SOURCE/authorization/result/STOPとR1原成果は保持した。

読む順序：[SOURCE_REPAIR.md](SOURCE_REPAIR.md) → [STOP_ANALYSIS.md](STOP_ANALYSIS.md) → [handoff.md](handoff.md)。公開HEAD/bytes照合は`publication_receipt.json`、既存保全は`verification.json`を参照する。

**MANDATORY STOP。C1-5/native取得/本格比較/全round RPE/Full-N/Pilotは未実行。**
