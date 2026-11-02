# 一体項再配分PF pilot v1：GPT重要科学判断への引継ぎ（2026-10-15）

## 現時点の実行結果と承認境界

固定one-shot H4 finite-step pilotと独立数値監査は完了した。`NO_UNIFORM_REGISTERED_COMPONENT_GAIN`、独立checker PASS。今回の数値実行認可は消費済み、retry0。GPT科学レビューの開始、研究の継続/終了、第二候補への移行は未承認・未判断。公開・GitHub照合後Mandatory STOP。

[結果と限界](qc_onebody_repartition_finite_step_pilot_v1_results_20261015.md)、[固定契約](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/contract.json)、[summary](../../artifacts/qc_onebody_repartition_finite_step_execution_v1/2026-10-15/saved_results_summary.json)、[独立checker](../../artifacts/qc_onebody_repartition_finite_step_pilot_v1/2026-10-15/attempt_1/independent_audit.json)。

## 今回分かったこと

linear H4 / 1.00Å / STO-3G / 保存DF rank12・固定generation順・Nα=Nβ=2 / 36dim、全12density deterministic、先頭pairの一方向mean-zero spin-paired shift、共通h窓8.5784978e-5..0.0027451193の一件development結果。L_D・PR delta探索ではない。generatorはexact-state γとd0を使うがE0/finite-h truthなし。equal-access compact BCH生成の還元所見を上書きしていない。

leading original0.01336651855→response0.00820068355、最大hのraw bias1.00726e-7→6.17977e-8Ha。有限hのraw誤差低下は観測した。最大hでscalar-corrected originalも要求を満たし、responseのcontrolled primitive/query費用比は全4条件約1.00039..1.00074、10%gate未達。overlapの差は丸め水準で有意な改善を主張しない。raw originalはCS2originalと同回路価格で、γ/d0取得を要せず既にeligibleというtieも明記した。

## この固定pilotでは判定していないこと

- 最大登録hで元raw S2が厳しいtargetのdeterministic budgetより十分小さいため、許容最大h/step削減のfrontierを探れていない。固定窓のno-gainと、より大きいhを含む一般不能性を区別する。
- CS2/S4/Richardsonの微小biasは宣言numerical allowance以下で、厳密interval比較や高次誤差順位は確定していない。
- 情報取得、fault-tolerant synthesis、実state preparation、全QPE/RPE confidenceとtotal-resource利益は未評価。static primitive modelの小差をcompiled Tや最終総費用へ変換しない。
- 独占的応答生成改善・新規算法認定、分子移送、第二候補DF等価回転の採否は未判断。

## GPTが別開始承認後に判断する問い

1. 登録範囲で費用gainがない一方、gridが誤差frontierに届いていないという情報価値の限界をどう扱うか。追加計算が判断を変える根拠があるか、現系列を終了するか。
2. 主要誤差補正を持つ強いPF対照と同じexact-state条件に戻したとき、再配分そのものへ追加投資する科学的根拠が残るか。
3. equal-access生成還元、γ/d0取得、finite-h/overlap、回路価格・queryモデルを踏まえ、独立新規性や全費用のどこが依然未解決か。
4. 継続が必要と判断する場合は、新scope/source/上限を改めて設計すべきか。第二候補へ移るか。いずれもCodexで自動開始しない。

ここでは追加pilotやh拡張の実行計画を確定していない。科学的不能性・正式終了・採択をCodexが代行しない。N1STOP/N2HOLD/N3CLOSED/THRIFT現template CLOSED、C1–C4/Track A–B・global manifest既知FAIL・旧結果/認可/STOPを維持する。
