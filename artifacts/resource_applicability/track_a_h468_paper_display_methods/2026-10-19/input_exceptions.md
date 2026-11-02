# 入力・来歴・表示の例外表

公開された比較記録の設定値は記録として引用し、元providerやprivate入力の今回の独立再監査と区別する。旧protocolのPENDINGやA82欠測は当時の履歴であり、下記の後続完成状況を現在の本文へ適用する。元protocol資料は上書きしない。

|ID／旧protocol対応|項目|現在の記録と差|掲載上の限定|
|---|---|---|---|
|PUB_EX01 / EX01/EX03|H4 input lineage|H4 Figure2A: legacy fixed DF12; H4 Table1: tol-only DF10; separate saved Hamiltonian/state; H6/H8 tol-only DF19/28|No relabelling or numerical transfer|
|PUB_EX02 / EX02|SCF/orbitals/integrals|Recorded RHF 1e-9/50, canonical helper order, spin convention; Recorded shared settings, one-half two-body convention; Full provider/runtime/orbital/byte equivalence not re-audited|Not complete procedural identity|
|PUB_EX03 / EX02|DF and projection|H4 threshold1e-8/no final_rank; retain signed generation order/cutoff0; Weighted Hermitian projection full-Fock N8; Recorded same definition, N12/N16; ranks19/28|Engineering check, not interval guarantee or size-only effect|
|PUB_EX04 / EX02|state solver|H4 dense scipy.linalg.eigh, sector36; Lowest vector, canonical phase; Recorded H6/H8 eigsh SA, HF v0, k1, tol1e-12, ncv40, maxiter1000; sector400/4900|H6/H8 provider bindings not freshly re-audited|
|PUB_EX05 / EX02|normalization/residual|H4 normalized eigh vector, no renormalization after saving; H4 residual gate1e-10; loader norm check; H6/H8 recorded one binary64 normalization; absolute1e-9/relative1e-10 residual gates|Retain original saved vectors; no new common solver or ground-state certificate|
|PUB_EX06 / EX06|compiler and control|Recorded Qiskit1.3.0 rz/sx/x/cx opt1 seed17; symmetric_directional; ordinary_ancilla_branch denotes semantics, not alternative lowering; H8 original profile bytes and full old compiler-output identity unverified|Do not infer binary identity from option/hash equality|
|PUB_EX07 / EX04/EX05/EX07/EX09|completion/cohorts/cost evidence|H4 DF10 P and H6 P completed in later fixed results; Table1 six fixed controls directly measured; old domains retain models; H4 coarse non-reproduction and H6 coarse PF-favouring reproduction retained|Old protocol PENDING statements are historical; no COMMON_CORE_READY promotion|
|PUB_EX08 / EX08|A82 completion versus artwork|Original A82 annotation is render-time history; Later Mechanism1ff6ac0b completed all32/64 saved factors; A82 old H8 n32 is not H8 independent n48; no factors transferred to new H4/H6 P|No new points, ratios, or factor calculations|
