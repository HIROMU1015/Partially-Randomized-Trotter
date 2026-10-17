# 固定SOURCE・APIの接続明細

2026-10-17 JST。[技術報告](technical_preflight_report.md)の静的根拠。SOURCEをimport・実行していない。以下の「実装あり」は定義・処理経路を読んだ意味で、現在環境での動作、数値correctness、性能の再認証ではない。既存testは定義だけを読み、今回のPASS数へ算入しない。

## Repository内の基盤

全行のSOURCE snapshotは `56f1cc55c59ab7bc42c6b24e0953675de9b29b1d`。ファイルbytes/SHA256とASTで確認した定義行は [evidence_and_checks.json](evidence_and_checks.json) にある。固定snapshotの監査であり、dirtyな親worktreeや全branchの状態を代表しない。

| ID | 固定SOURCE・関数 | 実装されているもの | 今回の生成問題への限界 |
| --- | --- | --- | --- |
| I01 | [df_hamiltonian.py L327][df-h] `DFHamiltonian`、L381 `PhysicalSector`、L938 `expand_sector_state` | scalar・one-body・signed DF-squareのデータ、number/Sz sectorの古典basis制限と展開 | arbitrary encoded register上の回路・state encoderではない。N/SzとS²を区別する |
| I02 | [df_trotter/ops.py L194][ops] `U_to_qiskit_ops_jw`／L213 `..._givens`、L226/246 diagonal primitives、L295 `append_diagonal_primitives` | NatureのJW Gaussianを用いる構造化回路、対角角度とscalar、controlled時のPhaseGate | structured経路はNature依存。fallbackの小規模dense pathとは区別。taper/GSE/Paldus回路生成器ではない |
| I03 | [df_trotter/circuit.py L17][circuit] `build_df_trotter_circuit` | energy shiftを時間に応じたglobal phaseへ接続 | 新encoded adapterでのcontrol化まで自動検証するものではない |
| I04 | [df_rte_tail.py L80][tail] `DFBasisRegistry`、identity policy | basis circuit identity、faithful/extractのidentity扱い | basis hashは同じ物理作用・sector・準備費の証明ではない |
| I05 | [qiskit_time_evolution_grouping.py L29][group] `build_clique_hamiltonian` | grouped Pauli circuit生成の既存経路 | L37–39でidentityをskip。absolute energyには外部scalar会計が必要 |
| I06 | [algorithm_design.py L130][old-design] `gaussian_circuit`、L233 `CircuitAudit.compile`、[construction_comparison.py L104][old-compare] `gaussian_circuit`／L114 `apply_pauli` | 過去のsector/Lie/conditional回転監査、controlled identity、phase-awareなprototype | 前者はStatevector/dense基準からの位相調整を使う。有限規模の監査を効率的な一般生成規則へ流用しない |
| I07 | [rpe_hadamard_interrogation.py L120][rpe] `QiskitRPEHadamardInterrogationBuilder` | 既にcontrolledなevolutionに対するwrapper、constant/identity/RTE位相metadata | L393–401はstate preparation未包含、backend未実行、shots=0等を明示。complete encoded RPE pipelineとは言えない |
| I08 | [mechanisms.py][mechanisms]、[qc_mixed_access_math_v1.py][mixed] | 既存mixed/conditional構成、occupation作用、Gaussianの参照処理 | N1/H3/THRIFT等の重複照合用。停止系列の再実行や新法への改称はしない |
| I09 | [test_df_partial_s2.py][test-s2]、[test_df_rte_tail.py][test-tail] | controlled unitary、identity・constant・onebody・event、dense-free path等を調べる既存test定義 | **今回は一件も実行していない**。新encodingでのvalidation証拠に転用しない |

Gaussianのbasis回路がexactに同じ `U` と `U†` として挟まれるなら、その共通scalarは相殺し得る。`_normalize_u_ops` がgate tupleだけを返すことから、現在の全経路にphase bugがあるとは結論しない。standalone変換、新しいcontrol境界、異なる合成精度のadapterでは改めて規約が必要である。

## 上流依存の固定版

requirementsの版を対応commitに解決し、tag URLとcommit URLのraw bytesが等しいことを確認した。`OF_INDEX` は解決済みcommitから直接取得した。上流SOURCEを保存実行せず、文書検証用一時cacheで読んだ。インストール済み環境がrequirementsと等しいという確認ではない。

| Package | 指定tag | 解決したcommit |
| --- | --- | --- |
| Qiskit | `1.3.0` | `896d1c8b72888a52a46081a6d1f4de8c56db8aea` |
| qiskit-nature | `0.7.2` | `d86d4a0306f8f4af8918c9eb9e3eae4622192f45` |
| OpenFermion | `v1.6.1` | `16c6b0666175cd4381dbbf841b14ba4b2f315a46` |

| 一次SOURCE | API条件・確認内容 | 接続時の注意 |
| --- | --- | --- |
| [Nature ParityMapper][up-parity] | 二qubit削減はblock spin orderのspin parityに基づく。interleavedとの併用に明示的警告 | 固定N/Sz sectorとparity sectorを混ぜない。mode-order adapterはH・state・symmetryと一緒に処理 |
| [Nature TaperedQubitMapper][up-taper] | symmetryのtaper値、Clifford変換、非可換operatorの `None`／filter処理 | 全補助operatorが残るとは限らない。削除をゼロ演算子と解釈しない |
| [Nature BogoliubovTransform][up-gaussian] | N×Nのcreation-onlyとN×2Nのcreation/annihilation入力。JWのみ対応 | 名前から任意encodingやN/Sz/S²の保存を推定しない |
| [Nature BKSF mapper][up-bksf]、[OpenFermion BKSF][up-of-bksf] | 既知BKSFのoperator変換API | GSE論文の任意graph/local-Majorana設計と同じ万能APIではない。mapperの出力はcontrolled oracle全体ではない |
| [OpenFermion sparse_tools][up-sector]、[indexing][up-index] | `jw_sz_indices` はdefaultでeven/odd spin mode、indexに `2**(n_qubits-1-k)` を使う | 他のLSB pathやNatureのblock orderとの同一性を仮定しない。fermionic reorderの符号とendianは別 |
| [Qiskit add_control][up-control] | 定義のglobal phaseを制御位相へ含める経路 | 元回路が捨てたscalarを復元する機能ではない |
| [Qiskit Operator.equiv][up-operator] | global phase同値の比較 | control化前の単独検査では絶対位相の一致を示さない |

公式web docsは検索・案内にも用いたが、版による違いが問題となる断定は上記固定SOURCEを基準にした。新しいAPI呼出しを試す実装や数値試験は作成していない。

## 検索範囲と非検出の意味

固定snapshotのtracked `src/` と `scripts/` に対して、`TaperedQubitMapper`、`ParityMapper`、`BravyiKitaevSuperFastMapper`、`Z2Symmetries`、`binary_code_transform`、`bravyi_kitaev_fast`、`Paldus` の文字列を確認した。結果と対象ファイル数を台帳に保存した。

この名前の検索は、統合adapterを確認できないという限定的な根拠である。別名の実装、generic objectを渡す機能、dependencyの未使用API、他branchの内容、未取得原本まで機能不存在とはしない。特に依存libraryに既知mapperがあることは確認済みであり、「符号化自体が未実装・未解決」とは書かない。

## 必要箇所のみ照合した一次文献

| 文献 | 読解した箇所 | この調査で確認していないこと |
| --- | --- | --- |
| [Tapering, 1701.08213v1][paper-taper] | simulator/isometry条件、particle/spinとZ2 symmetryの節 | 出版版全差分、プロジェクト入力でのsector同定 |
| [GSE, 1810.05274v2][paper-gse] | even algebra、graph/local Majorana、loop codespace、誤り訂正要件の区別 | 新しいgraphの構成・妥当性や費用の数値評価 |
| [Molecular GSE, 2511.09322v1][paper-brown] | 化学graph、edge path、軌道変換、depth指標の定義 | 未提示の新手順への完全包含、総controlled energy費用への移植 |
| [Quantum Paldus Transform, 2506.09151v2][paper-paldus] | 導入、§IV A–C。spin-free sector、group actionとinteracting Hの区別、限定したfast-forwarding | 全94頁の証明・回路検証、こちらの新しいcontrolled実装の保証 |

Paldus以外の未取得原本の回収や、広い候補探索は行わない。読んでいない資料の結論を補完しない。取得と読解は科学的な独立検証・正式受理とは異なる。

[df-h]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/src/trotterlib/df_hamiltonian.py#L327
[ops]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/src/trotterlib/df_trotter/ops.py#L194
[circuit]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/src/trotterlib/df_trotter/circuit.py#L17
[tail]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/src/trotterlib/df_rte_tail.py#L80
[group]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/src/trotterlib/qiskit_time_evolution_grouping.py#L29
[rpe]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/src/trotterlib/rpe_hadamard_interrogation.py#L120
[old-design]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/src/trottertracks/representation_exploration/algorithm_design.py#L130
[old-compare]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/src/trottertracks/representation_exploration/construction_comparison.py#L104
[mechanisms]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/src/trottertracks/representation_exploration/mechanisms.py
[mixed]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/src/trottertracks/representation_exploration/qc_mixed_access_math_v1.py
[test-s2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/tests/test_df_partial_s2.py
[test-tail]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/tests/test_df_rte_tail.py
[up-parity]: https://github.com/qiskit-community/qiskit-nature/blob/d86d4a0306f8f4af8918c9eb9e3eae4622192f45/qiskit_nature/second_q/mappers/parity_mapper.py
[up-taper]: https://github.com/qiskit-community/qiskit-nature/blob/d86d4a0306f8f4af8918c9eb9e3eae4622192f45/qiskit_nature/second_q/mappers/tapered_qubit_mapper.py
[up-gaussian]: https://github.com/qiskit-community/qiskit-nature/blob/d86d4a0306f8f4af8918c9eb9e3eae4622192f45/qiskit_nature/second_q/circuit/library/bogoliubov_transform.py
[up-bksf]: https://github.com/qiskit-community/qiskit-nature/blob/d86d4a0306f8f4af8918c9eb9e3eae4622192f45/qiskit_nature/second_q/mappers/bksf.py
[up-of-bksf]: https://github.com/quantumlib/OpenFermion/blob/16c6b0666175cd4381dbbf841b14ba4b2f315a46/src/openfermion/transforms/opconversions/bksf.py
[up-sector]: https://github.com/quantumlib/OpenFermion/blob/16c6b0666175cd4381dbbf841b14ba4b2f315a46/src/openfermion/linalg/sparse_tools.py
[up-index]: https://github.com/quantumlib/OpenFermion/blob/16c6b0666175cd4381dbbf841b14ba4b2f315a46/src/openfermion/utils/indexing.py
[up-control]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/add_control.py
[up-operator]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/quantum_info/operators/operator.py
[paper-taper]: https://arxiv.org/html/1701.08213v1
[paper-gse]: https://arxiv.org/html/1810.05274v2
[paper-brown]: https://arxiv.org/html/2511.09322v1
[paper-paldus]: https://arxiv.org/pdf/2506.09151v2
