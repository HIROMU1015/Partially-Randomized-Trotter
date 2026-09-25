# P-B/P-C/P-A 研究テーマ選定

最終更新：2026-09-25 JST

## 結論

研究テーマ選定pilotを一巡し、**P-A（DF回路列のjoint synthesis）を暫定主題**、
**P-C（geometry間エネルギー差）を副候補**、**P-B（signal weight）は現H4範囲で停止**とした。

P-Aの選定は確定ではない。project内の強いbaselineに対する改善は確認できたが、広い回路合成文献に
対する新規性監査が未完了である。従ってstatusは
`provisional_pending_prior_art_and_novelty_audit`とする。

## 4問による比較

| theme | 既存baselineとの差 | 明確な差分 | 未使用条件 | 完成形を一文で表せるか | 判断 |
|---|---|---|---|---|---|
| P-A | full共有、event-support、現行`support_run_le_1`を越えるinterval DP | 現行比RZ -7.19% | 未使用列長3/5/8で全て改善 | support/run構造から同値で安いbasis planを予測する | 暫定主題 |
| P-C | absolute energyでなくsigned PF errorの差を利用 | 二重holdout差分bias誤差2.539% | 未使用geometryとdelta | 誤差相殺・破れを予測し差分用PFを選ぶ | 副候補 |
| P-B | energy biasとweight/signalを分離 | 実用的な選択差なし | 選択器を検証できるfailureなし | 反例が得られれば可能 | 現範囲で停止 |

P-Aを優先する理由は、強い比較対象への直接改善、未使用列長での再現、operator同値性、有限候補oracleへの
近さが同時に得られたためである。P-Cは予測gateを通過したが、現時点では滑らかな局所補間を越える
breakdown機構や、軌道・fragment追跡による設計可能性までは示していない。P-Bは一般仮説を棄却せず、
現gridで主題化する根拠がない。

## 次の一件

次は計算拡張ではなく、P-Aの先行研究・新規性監査を行う。監査で独立した差分が残る場合に限り、
次のblind validationを事前登録する。

- 現在のH4 snapshotから独立したDF snapshotまたはevent family
- 少なくとも一つの未使用compiler context（optimization levelまたはcoupling条件）
- 現行policy、full共有、event-support、interval DPの同一比較契約
- operator同値性、RZ/CX/depth、古典DP cost、悪化trajectoryを分離して報告

新規性が既知手法へ還元される場合は、P-Aを主題化せずP-Cへ戻る。P-Cへ戻る場合の次の判別は、
geometryを増やすだけでなく、軌道/DF fragmentを連続追跡した場合と独立生成した場合の相殺・破れを
holdoutすることである。

H12、長RPE総cost、追加$q>32$、$\delta/r$境界拡張は、現時点の次作業に含めない。

## 証拠

- selection artifact：
  `artifacts/research_direction_theme_selection/2026-09-25/theme_selection_pb_pc_pa_v1.json`
- content fingerprint：
  `7daa49c3c30b453d69d52830d0889ce04104cc5a0f02a2c3083517486d575fbb`
- implementation：`src/trotterlib/research_direction_theme_selection.py`
- runner：`scripts/run_research_direction_theme_selection.py`
- test：`tests/test_research_direction_theme_selection.py`
- 専用test：`3 passed`
- 全suite：`582 passed, 2 skipped, 4 warnings`、失敗0

selection artifactはP-B/P-C/P-Aのcontent fingerprintとfile SHA-256を固定し、新しい物理計算や回路compileを
行わずに判断を再構成する。証拠statusはlocal dirty-worktreeであり、外部再現またはimmutable CIではない。
