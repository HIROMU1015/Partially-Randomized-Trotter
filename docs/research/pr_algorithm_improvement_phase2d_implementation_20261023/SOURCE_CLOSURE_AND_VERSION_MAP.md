# SOURCE closure・版差・接続方法

実装起点はP0最終 `b8d54708f90ccef275eda3757143100b2d0c677e`。その科学SOURCEはR1起点 `ef2eb7d55ea37bf93e0b85773df20b63811d1f9b` と同系列。既存fileの更新を行わず、新prototypeへ必要機能だけを限定port／adapterで接続した。

| 固定出典 | 内容 | 今回の接続 |
|---|---|---|
| 2978e2fea672b7a1ff20cac74269ec9a610159dc rte.py | RTEComponent、RTETailLike、law/config、event、sign accessor | 起点と同bytes。新prototypeが実際にimportするのは起点の同一rte.py |
| 同2978 df_rte_qiskit.py | sequence builder、basis_plan、中央制御、共有とphase集約 | 起点と異なる。丸ごと置換せずcircuits.pyでtyped sequenceに必要な処理を限定再実装 |
| 同2978 df_partial_s2_repeated.py | raw／optimized、同D境界half-step融合 | 一blockの同じfusion規則を新build_evolutionへ限定port |
| f29eafbf4412193e8e8660d9f051889c2cba684b research_direction_sequence_policy.py | singleton support_run_le_1、long-run full | 新typed router共通policyとして接続、全arm同条件 |
| 同f29 research_direction_structure_pilot.py | support_restricted_unitary_completion | numpyだけに依存する該当helperを限定port。重いpilot moduleはruntime importしない |
| ae7434452450e80ec007d65417335e18a063fa1e construction_comparison.py | arbitrary Pauli、非制御local/parity＋中央control | circuits.pyの任意長branchとexact algebraの参考。toy全体・dense oracle・samplingを移植しない |
| 起点 rpe_hadamard_interrogation.py | strict DF型、H→U→Sdg/H、metadata | strict DF型へ偽装せず新MixedEvolutionResultで同じwrapper recipe／検査を再実装 |
| 起点 df_trotter/ops.py | U_to_qiskit_ops_jw_givens、U.T convention | native経路だけを呼ぶ。dense fallbackを呼ばず、unitary gateを拒否 |

起点df_rte_qiskit.pyのSHA256は `15395beab7f6e8d68589964516c03d1b513b87439490e1cb1d1487fbf4848aac`（16,363bytes）。2978参照版は `c710de23b7bfef525199ab28b897c21b2a78723a01b29da6067ff64354dcd436`（20,313bytes）。反復builderも28,698対30,013bytesで異なる。参照関数が起点にもあると仮定していない。

[REFERENCE_PROVENANCE.json](REFERENCE_PROVENANCE.json)に固定commit/path/hashと確認範囲を保存する。新SOURCEのbytes/SHA256/Git blobは[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。公開されたSOURCE commitは[PUBLICATION.json](PUBLICATION.json)の検証済みpayloadで追跡する。

## Transitive dependency

ASTはnew SOURCEからのimportを追い、親packageの__init__、conditional importも含めた保守的graphを記録する。旧trotterlibの__init__が多数moduleをimportすることも含め、内部SOURCE80fileが到達候補である。これは全てのbranchが実際に実行されるという意味ではない。新packageのroot __init__はdocstringだけで、科学処理や自動re-exportを行わない。

reference.pyはfractions/itertoolsと関数内mpmath/numpyだけを利用し、candidate algebra／canonicalizer／router／momentをimportしない。authorization.pyはstdlibだけ、runnerはverify_packet／claimの後にexecutionをimportする。

外部にはNumPy、SciPy、Qiskit、Qiskit Nature、mpmathに加え、既存package initializerが到達するOpenFermion／PySCF／plotting等の候補がある。実envのimport解決、version、Python patch、wheel/hash、BLAS、Qiskit transpiler、mpmath内部endpoint形式は未確認。AST graphをruntime closure検証済みと呼ばない。shared venv/packageを変更せず、将来専用環境をbindingする必要がある。

全import branchの科学実行・パッケージ初期化を今回は行っていない。未使用のdense fallback codeが既存closureに含まれても、新routerからそのAPIを呼ばないことを静的に確認する。helper SOURCEのidentityとexternal runtimeの資格は別である。
