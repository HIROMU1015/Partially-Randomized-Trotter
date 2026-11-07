# 案M：保存N1のn=0 header binding

2026-11-08 JST／SERVER1。base [d07a76d][M]の人工instrumentを作り直さず、保存operation metadataを既存`SlotLabel`と群適格性判定へ渡す差分を追加した。**人工header上で接続済み、実N1への投入・群集計は未実行。**

## 保存schemaと再利用

固定N1結果は`744e1a09661ebba6841fe5faf456f88c41c47780`、科学SOURCEは`b6dbec9f80da35a88fba28795ad02ca74c9adfa0`。対象は`artifacts/o2_a1_new_target/2026-11-06/n1/`の5ファイルである。実catalogを読み取って群を調べるのでなく、生成serializerと既存builderを静的に確認した。

| 保存ファイル | 使うfield・意味 | 元SOURCE |
|---|---|---|
| `SOURCE.json` | source_commit、scope（幅、m/q/r/K、δ、threshold） | `a1_new_target_n1.run` |
| `CONFIG.json` | config/distributionのtau、time、tail、orders | `make_rte_config`とN1 serializer |
| `LAW.json` | label/orderのexact質量、53-bit grid、products_then_rotation | `a1_new_target_n1.law_payload` |
| `POLICY.json` | declaration/fingerprint、source basis/support→selected definition、操作とhash | `policy_payload`, `make_run_threshold_basis_plan` |
| `PREPARATION.json` | identities、components、basis_operations、D blocks、scalar/identity hex | `a1_law_validation.preparation_payload`, `_block_hash_payload` |

`BasisChangeOperation`、`canonical_basis_hash`、`one_body_diagonal_primitives`、`df_squared_diagonal_primitives`、N1のcanonical JSON/digest、既存`CategoricalLaw`を呼ぶ。これらは保存記述・既知の対角係数からmetadataを作る処理であり、Hamiltonian再生成・対角化・新basis・回路buildではない。

`restore_operations`はgate生成とlocal matrix fingerprintを伴うため呼ばない。`SavedEngine.selected_circuit`はstate/full回路へ接続するため静的対応だけを利用する。既存operation descriptorを複製して一般codecを新設せず、保存basisの`rz`/`xx_plus_yy`とD blockの`rz`/`rzz`に限定する。

## Literal記述への必要最小限の拡張

[m_label_average.py](../../../src/trottertracks/algorithm_codesign/m_label_average.py)へimmutableな`LiteralCircuit`を追加した。既存`circuit_key`がこのmetadataも受け取れるようにし、`qualify_group`と`partition_supplied_rows`本体は変えない。既存Qiskit instrumentの生成処理も変更せず、Literal入力に対して`METADATA_ONLY_NO_QUANTUM_BUILD`を返すguardだけを追加した。

新literalはsystem幅、application順のgate名・binary64 hex parameter・ordered wire、ordered scalar phase termsを保持する。Qiskitのstandard gate classと同じgate/parameter/wire意味論を用いるが、scalarを丸めて一つへ足す代わりに項列を残す。従ってQiskit由来の単一global phase keyと自動同一視しない。近似一致・phase fitting・group都合の簡約をしない十分条件であり、可能な全群を発見する判定器ではない。

inverseはgate列を反転し、RZ/RZZは角を負に、XXPlusYYはthetaだけ負にしてbetaを維持する。phase項も反転・符号反転する。後者は固定Qiskit1.3.0の`XXPlusYYGate.inverse`を静的確認した。幅8をmetadata上で拒否する制限はない。

## n=0の元targetへの対応

新[binder](../../../src/trottertracks/algorithm_codesign/m_n0_header_binding.py)の`bind_n0_headers(records)`は、実保存schemaの5辞書を明示入力として受け取る。対象slot IDは常に`m0/q1/r1/order0/application0/rotation`。order/roundの探索・変更は行わない。

既存`SavedEngine.selected_circuit`のsystem側の順序は、aggregated scalar phase、forward D blocks、selected basisの逆列、signed rotation、selected basisの順列、reversed D blocksである。保存selected basisの順列をSとすると

\[
 U_l=e^{i\chi}A\,S\,e^{-i\phi D_l}\,S^\dagger B
     =e^{i\chi}A V^\dagger e^{-i\phi D_l}VB,\qquad V=S^\dagger.
\]

**案MのVは保存されたSそのものではなくS†。** 名称だけでorientationを決めない。A/Bは行列積記法の順序とapplication順序を区別する。Bのapplicationは元blockリスト順、Aは逆順。各blockは元SOURCE通りbasis inverse→diagonal→basis forwardで、diagonalはRZZ、RZ、scalar phaseの順で記述する。

| 値 | binding規則 |
|---|---|
| A/B | `df_partial_s2._append_block`と同じhalf time δ/2のdescriptor。block phaseはordered項列で保持 |
| V | actual selected定義のadjoint。original basisとselectedを別々にhash/ID照合 |
| φ | 元`_make_event`と同じbinary64演算`atan(saved_tau/(0+1))`。近似一致や理想値へ変更しない |
| signed D_l | 保存coefficient_sign=±1とZ/ZZ support。元rotationのunsigned angle=sign×φに対応 |
| scalar phase χ | 元`_append_aggregated_phases`通り`−δ×1×(constant+extracted_identity)`。別々に丸め直さない |
| event phase | n=0なので`(−1)^0=1`、phase angle=0。n=2へこの値を流用しない |
| identity | generic人工headerではsigned空supportとして保持。固定N1実行のextracted-identity catalogにidentityが混入したらINPUT_OR_POLICY_MISMATCHで止める |
| wires | 保存幅nと0..n−1のQiskit system順。ancilla/readoutはこのmetadata passの対象外 |

対角primitive生成に伴う少量の算術とphi生成は、将来のmetadata-only科学callに含める。今回これらを実N1数値に適用していない。basisの保存matrix hashはbyte provenanceとして照合するだけで、新しいlocal matrixの計算やoperator保証を行わない。

参照した既存12関数について、上のN1科学SOURCEと今回baseのAST一致を確認した。[静的照合記録](../../../artifacts/o2_m_n0_header/2026-11-08/STATIC_SOURCE_CORRESPONDENCE.json)。逆basis・block・scalar・angle/sign・selected policy・diagonal primitive・basis hashの処理に版差は見つからなかった。この一致はbinding実入力のcorrectnessを実行認証したという意味ではない。

## Actual Qと既存群判定

label質量はLAWの`exact`だけをFractionで読む。componentsの理想/float probabilityを使わない。p_g=Σ_g p_l、q_l=p_l/p_g、full conditional context質量=Q(n=0) p_gを別記する。label catalogとlawの完全coverage・正規化/gridを検査し、zero-grid labelは元lawと個別記録に残して群対象からのみ分ける。正labelを落とす処理や価格依存再正規化はない。

群署名はη/slot/policy/law、A/B/Vのliteral、φ/phase、wire順、selected basis IDを含む。original IDが同じでもselectedが異なればまとめない。signed diagonal functionは(sign, sorted support)のsymbolic descriptorで比較する。全zのr(z)列挙は不要で、今回も将来のmetadata passも行わない。

`A=B†`はliteral adjointの一致が明示できる場合だけ`ONLY_KNOWN_DIRECT_MEASUREMENT_STRUCTURE`とする。不一致は非適用の証明ではなく`UNKNOWN_NO_LITERAL_ADJOINT_MATCH`。同情報は群の非自明性とは別欄へ保存する。

## 未検証事項

実H4の非自明群、selected定義の実保存値への適合、正supportの実binding、instrument誤差、実stateの分散、ancilla-free baselineとの比較、full費用はUNKNOWN。今回の人工headerからの変換成功を、これらの取得済み結果として扱わない。科学実行の境界は[実行草案](M_N0_METADATA_EXECUTION_DRAFT.md)へ分けた。

[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/d07a76de8e7a9995b1d55c258556b86fe58b1e3a/docs/research/o2_m_instrument_technical_validation_20261108
