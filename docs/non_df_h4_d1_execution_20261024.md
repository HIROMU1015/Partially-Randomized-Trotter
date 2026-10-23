# H4 D1限定診断：実装・実行記録

2026-10-24 JST。今回の利用者指示はadapter準備からD1最大9費用compileまでの一括承認。基点3282f74cd4d219141e4358c215a225dc2e3ec9c3、元H4 SOURCE0c329b30a81accb35c212be8defa66fee0dd0f12。

固定H4/1Å/STO-3G/n8、Pauli184/prefix27・DF tol1e-8/rank10/prefix1＋one-body、S2/T.8/q8/r4/K4、cosine、Qiskit1.3.0 rz/sx/x/cx opt1 seed17、prep除外、DF raw concatenation。元4代表以外のtrajectory、sampling、D2/D3、N/Gは対象外。

## 技術SOURCE固定前の状態

[adapter](../src/trotterlib/h4_d1_adapter.py)は保存gate・event・primitiveを元型へ復元し、元Pauli append、DF basis registry／sequence builder、diagonal append、wrapper／compilerを再利用する。新規部分はdescriptorの接続とphase・scheduleの維持だけで、Hamiltonian／state／lawを再計算しない。

[controller](../src/trotterlib/h4_d1_control.py)、[runner](../scripts/run_h4_d1.py)は新SOURCE／runtime／入力SHA／利用者の承認を結合したone-shot grantだけを受ける。原本SOURCEは変更しない。Call前予約はfull build4、ablation build5、費用compile9まで。Stage Aは独立4process、Stage Bは旧価格の最初の不一致で残るcallを消費しない順序、Stage Cは独立5対象を並列化する。CPU/RAMへの人工上限は追加しない。

全full input hash一致前にcompileしない。Pauli自動gate名だけの疑いでも不一致を迂回せずSTOPする。Full4価格一致前にablationをbuild／compileしない。新STOPは今回のD1だけの記録で、旧native UNKNOWN、C/L BLOCKED、compiler MISSING_EVIDENCE、旧N/G nullを維持する。

[人工adapter検査](../tests/test_h4_d1_adapter.py)は人工2qubitのみで、位相・両control branch・basis inverse／reuse・event順序・QPY roundtrip・sampling/preparation/compile拒否を確認。[authority検査](../tests/test_h4_d1_authority.py)はscope拡張・grant欠測／改変・one-shot消費・予算の非返還を確認する。既存native witness／CallBudget／durable saveを流用し、主agent自身で確認する。独立科学レビューではない。

SOURCE固定後にseal／active contract／新grantを作成し、Stage Aから実行する。実行結果・終了監査・公開状態はこの報告を更新して記録する。
