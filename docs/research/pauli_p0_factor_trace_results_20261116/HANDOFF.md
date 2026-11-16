# B2/A：P0障害の原因と次の境界

科学SOURCE655e74212、保存carrier9fa9525b3、実行`run01/RESULT.json`。
保存operator再解析3件・factor復元1回・LD9→8→7のtrace3件が完了。retryなし。

- LD7：元残差1.11294e-7の主因はJW `+=`途中除去。JWだけno-filterのbridgeは残差1.06801e-16。
- LD8/9：factor9のweighted Fermion residual加算で128件の非zero部分和が消え、JW加算でも消える。
  JWだけno-filterでは約6.39180e-7残るためFermion側も必要。
- 最初の差は全LDともone_body加算だが、約1e-14係数の消失だけではgate超過を説明しない。
- 完全no-filter shadowは同じ保存DF tailと約1e-19以下で一致。元defaultは保存JWと全項・順序・値で一致。

比較表は[LD_SUMMARY.csv](LD_SUMMARY.csv)、first term/係数は[STAGE_SUMMARY.json](STAGE_SUMMARY.json)、
全イベントと各段階の係数mapsは`run01/LD{9,8,7}/COEFFICIENT_TRACE.json.gz`。
`real_hex`/`imag_hex`を保存し、消失イベントには加算前・incoming・除去前部分和を含む。
イベント数と固有label数、ノルム寄与の加算を混同しない。

最小修正案は[MINIMAL_FIX.md](MINIMAL_FIX.md)。P0専用のFermion/JW構築に局所no-filter集積を導入する方針であり、
OpenFermion global tolerance・元SOURCE・元判定を変更する提案ではない。
新production builder、辞書identity/λ_P、canonical/scheduleの採用は未実施。
旧Pauli IDや旧signal/価格を新辞書に再ラベルしない。

**B2へ返す状態は「原因診断完了・本番P0は元STOPのまま」。**
LD7/8/9の新準備完了やRPE/価格処理へ自動昇格させない。
次の必要事項は、局所修正方針と新SOURCEのP0再資格の承認。signal・価格較正・Q3はその後続。
今回の実行許可は消費済みの診断範囲であり、追加run・他LD・新条件への許可ではない。

研究ロードマップへの影響：P0障害は原因未特定から固定入力で原因説明済みへ進んだ。
運用上の辞書資格・signal/価格coverageは増えていない。ロードマップ本体は統合担当が更新する。
