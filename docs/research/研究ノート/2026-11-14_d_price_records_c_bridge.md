# 2026-11-14 D：C Pauli bridgeへの接続と価格record分類の修正

C `c628e9e8` を基点に新branchを作った。Dの4 callerとtestを、まず8b28dce0とbyte一致のまま取り込み、その上で修正した（固定SOURCE `62ae1bd5`）。
Pauli LD1/m0 pilotは、Cの正式planner（`I.plan`→`price_context`／`pauli_requests`）のcanonical physical候補を正本にした。
旧形式IDは削除した。seed（14788159701244282917）、辞書、D2、sampler、scalar、回路経路は変えていない。

価格recordはB PRICE_KINDSに合わせた。
- Full PF／Truncated p7：DIRECT_COMPILED_DETERMINISTIC／DETERMINISTIC。
- Pauli単一path：DIRECT_COMPILED_FINITE_SAMPLE／FINITE_COHORT_ONLY。
- 旧名はdiagnosticに残した。

人工価格で確認した結果：C guardは正しいsidecarを受理し、各種の不一致を拒否した。第1水準会計では、PauliがFINITE_COHORT_SCENARIO、Full/TruncatedがDETERMINISTIC_DIRECTになった。

D 37件、C 48件のtestがPASS。旧04bead11／8b28dce0は保全した。実sampling・build・compile・価格は0。
[REPORT](../d_price_records_c_bridge_20261114/REPORT.md)。公開照合後mandatory STOP。
