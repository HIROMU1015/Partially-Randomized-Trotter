# C1：保存H6の1経路・独立state-action診断

ユーザーの継続修正許可に基づき、SOURCE `38c8da551225d2d48fd285b6242e5771619f5b86`／[5-call契約](research_c_c1_h6_native_state_action_diagnostic_contract_20261025.json)で、保存CDF14 s1/literalのraw／P0 QPYだけを診断した。直鎖H6・1.00 Å・STO-3G、sector400、ordered S2 h0.1、固定5 probe・full8192の範囲。新しい回路build/compile、fit、DF/PF再計算、shotsや費用計算は0。

[結果・5 probe別値・保存statevectors・STOP](../../artifacts/research_c_c1_h6_native_state_action_diagnostic/2026-10-25/run01/)／[SOURCE・入力hash・call要約](../../artifacts/research_c_c1_h6_native_state_action_diagnostic/2026-10-25/execution_summary.json)。SOURCE46／入力4の照合と人工2 testsはPASS。診断は5/5 COMPLETE、retry/resume0、wall40.278秒。native試行324＋診断5＝累積329 units。

| 比較 | max-entry |
|---|---:|
| raw Aer／保存sector reference | 4.2196691246884366e-15 |
| raw Qiskit Statevector／保存reference | 4.4513748033426296e-15 |
| compiled Qiskit／raw Qiskit | 5.51982533701089e-10 |
| compiled Aer／compiled Qiskit | 7.622883573084989e-15 |

両シミュレータがcompiled側で一致し、そのcompiled/raw差が登録1e-10を超える。少なくともこの経路・5 probeでは、Gaussian精度修正後のraw/reference不一致やAer独自の実行誤差ではなく、compiler出力側へ問題を絞れた。これは有限probeの診断で、全operator上界ではない。診断対象の不一致を観測することが目的であり、`STOP.failed=false`をcompiled correctnessのPASSとは扱わない。

独立した人工5-qubit・150 block回路では、control0の既知identityからの旧P0誤差8.7120884972e-11を再現した。P0処理中のglobal_phaseだけを既存Qiskit/SymEngineの高精度数で集計する試作では、1.0444168237e-12へ低下し、出力ゲート列と角度は旧出力と完全一致した。これは人工入力の技術確認で、H6を再compileした結果ではない。

次の技術修正では、既存P0の全pass・opt1・seed17・basis・approximation_degreeを維持し、declared global_phaseの加算だけを安定化する。raw/reference状態・信号からphaseをfitしたり、科学的閾値を緩和したりしない。長い人工回路でゲート本体不変と位相保持を検証し、新しいSOURCE・契約・fresh outputを固定してから、同一8経路を再評価する。

今回の5-call診断は完了しSTOPを保持する。旧2回のFAIL・SOURCE・STOPを変更しない。未知のglobal合成誤差／CA／shots／総費用は未資格のまま。
