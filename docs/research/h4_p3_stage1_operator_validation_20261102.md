# P3 Stage 1：H4案Aの独立operator一致検証

2026-11-02 JST。ユーザーは案A `H=c_N I+T_eff+V_square`、`T_eff=T_bare+D̂`のStage 1を明示許可した。
共通Gaussian一体保持回路を将来のStage 3に使う方針も示したが、今回はその回路を実装・生成しない。
基点は静的調査commit `eb20e81074aaae3cb365b348f6c3ab9d217d3697`。
[静的調査](h4_df_pauli_matched_partition_static_design_20261102.md)と[再利用・実行準備](h4_df_pauli_matched_partition_execution_readiness_20261102.md)の未確定事項のうち、今回許可された選択だけをこの記録で具体化する。過去の2資料は当時の記録として保全する。

## 固定条件と入力

Linear H4、隣接1.00Å、STO-3G、中性singlet、4 spatial/8 interleaved spin orbital、8 system qubits。
主比較はfull256次元。Pauli184非identity項＋constant、保存DF tol-only1e-8の返却rank10を使用する。
新しい分割prefix・q/r/K・時間発展の条件はない。旧比較T=.8は来歴としてだけ記録する。
保存36次元Nα=Nβ=2 indicesは和operatorのleakage補助検査にだけ使い、主比較を射影しない。
共通raw stateは配列・hashのauditに使用し、状態作用・norm再評価・reference生成は行わない。

入力基準refは`4591b8d75a2a6a55c74a51225f0c61648023c8ee`。固定hashはrunnerのINPUTSおよび実行時source_and_inputs.jsonに記録する。
元生成SOURCEは`0c329b30a81accb35c212be8defa66fee0dd0f12`。
receipt/snapshot.npz/pauli.jsonと両Cモデルを読み、同じMO/spin tensor、DF one-body/λ/G、state、sector indicesの対応を確認する。
receiptの既存full-H差`8.462237051574415e-15`（旧tolerance1e-9）は引用のみ。新T/V一致の証拠や新toleranceとしては使わない。

## 独立構成と最小拡張

| operator | Pauli経路 | DF経路 |
|---|---|---|
| H | 保存Pauli184項＋元constantをPauliHamiltonian.sparse_operatorで再構成 | 保存h_eff/λ/G/c_Nを元df_linear_operatorのfull Fock actionで各basis columnへ作用 |
| T_eff | 保存bare spin_one＋元Wのnormal-ordering収縮Dをfermion→JW→Pauli行列へ | 保存DF one_bodyをnative fermionic excitation tableから行列化 |
| V_square | 保存normal-ordered Wと−DをInteractionOperatorへ明示しJW変換 | 保存Gのnative one-body行列Aを作りΣλ A²を計算 |
| D | 元WからOpenFermionのreordering/contraction helper | 保存h_eff−bareのtensor、およびそのnative one-body operator（補助検査） |

主T/Vは`H−T−scalar`だけでは定義しない。分解閉合は独立構成後の補助検査とする。
Pauliのbare tensor全H、V_normal、Dの再構成と、保存snapshot両full matrixも補助照合する。
Pauli identityを各operator内に残し、物理的核反発c_Nを一回だけ加える。Pauli constantをDF constantへ二重加算しない。
DF square identityは`λ[(Tr G)²+Tr(G²)]/4`を各factorから取得し、native full-Fock trace/JW identityと照合する。
これは既存I/Z/ZZのidentity式のbasis-independent表記であり、DF/Gaussian対角化・回路生成は不要。

[kernel](../../src/trotterlib/matched_partition_stage1.py)、[runner](../../scripts/run_h4_p3_stage1.py)、[人工tests](../../tests/test_matched_partition_stage1.py)。
元`get_chemist_two_body_coefficients`、PauliHamiltonian、df_linear_operator、native excitation table、reverse_bits、spin_lift_unthresholdedを再利用する。
元低rankfactorizationは呼ばない。OpenFermion1.6.1 low_rank.py SHA256は
`de74fb802550417788362845515ed9e56a180e4ff85ab21ab43ba8445156dbbe`。
元Wは既に1/2を含むことを保存MO→spin liftとの対応で確認し、追加の1/2を掛けない。

最小の新規処理は、独立経路を束ねるadapter、native excitation signsをそのまま使うCSR行列化、比較・記録である。
元DF Python actionのbasis column反復はfull Hの独立照合に残す。T/V/Dは同じexcitation tableをvector化して重複するcolumn反復を省く。
Pauli側とDF側のoperator行列や補正の中間結果は共有しない。汎用Hamiltonian framework・approval/grant機構は新設しない。

## 実行前に固定する数値政策

complex128（binary64実虚）、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、OpenFermion1.6.1。
主normは一般行列のspectral normをSciPy SVD、driver=gesddで求める。Hermitian化して差を消さない。
Frobenius normと最大要素差も保存する。SVD失敗時の別solver・追加精度への自動切替はしない。

固定dimension d=256、machine epsilon e、`gamma_d=d e/(1−d e)`、比較operandのnorm scale `s=max(1,||A||₂,||B||₂)`に対し

`tau = 16 gamma_d s`

をengineering consistency resolutionとする。scalar/D-tensor/leakageにも構成元full-Fock次元を使う。
leakageのscaleは元operatorのnormを使う。これはbinary64の次元依存丸めscaleに、10factorの加算・独立JW/native構成・closure等の工程を見込む余裕を加えた診断政策である。
厳密なrounding boundやDF truncation certificateではない。入力は既存binary64値を対象とし、新しい積分誤差の評価はしない。
df_tol1e-8をoperator差の許容値へそのまま足さず、保存df_truncation_value=0も厳密0誤差の証明とはしない。

- PASS：residual ≤ tau/4。境界に余裕がある数値整合。
- UNRESOLVED：tau/4 < residual ≤ 4 tau。tolerance周辺として停止し、追加精度を自動実行しない。
- FAIL：residual > 4 tau。意味のある不一致として原因・途中結果を保全して停止。

この式・係数・bandは実H4結果を見る前に固定する。結果を見てから緩和しない。
判定対象はH/T/V/DのPauli/DF差、D tensor、両閉合、bare/JW全H、normal-order閉合、H/T/Vの各Hermiticity、
保存sectorとの両方向leakage、保存両full matrix、scalar/identity整合の計34 metric job。
独立一致の数値診断であり、数学的厳密等号・全RTE law・実compiled精度の保証ではない。

## 人工検査・実行・資源

人工2/4qubitで9 tests PASS。signed/offdiagonal factor、手計算occupation oracle、native actionとCSR、bit順、
補正欠落/二重加算、1/2誤り、scalar二重計上、Hermiticity/leakage、PASS/UNRESOLVED/FAIL境界を検査した。
factorization/QuantumCircuit生成を禁止した区間で独立operator構成を確認した。実H4の科学実行はSOURCE固定後のみ。

既存専用runtimeを読取利用し、Python bytecodeを保存しない。BLASは小行列ごとに1thread、独立metric jobを空きCPUに応じて並列化する。
これは256次元の小行列の過剰thread競合を避ける実装で、科学計算全般へのCPU/RAM上限ではない。
Pauli/DFの2構成経路は独立taskとして実行し、全threadをjoinする。共有環境・他jobの設定/状態は変更しない。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python -B -m pytest -q tests/test_matched_partition_stage1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python -B scripts/run_h4_p3_stage1.py --execute
```

科学SOURCEは上記code/tests/条件をcommitで識別してから実行する。実際のcommit SHAとcode/input/依存SOURCE hashは出力に保存する。
新規出力はJST日付の`artifacts/df_pauli_matched_partition_validation/2026-11-02/stage1/`、既存なら再実行を拒否する。
matrixは10構成のhash/shapeだけを保存し、巨大な原本やfull行列を複製しない。補正8×8と比較・scalar・status・STOPは保存する。

原本/SOURCE/規約不一致、欠落、数値FAIL/UNRESOLVED、scalar二重計上、sector不整合、未許可の条件変更、共有環境への干渉でSTOPする。
科学実行失敗後の自動retry、案B、Stage 2/3/4、sampling、time evolution、N/C/G、P1/P2再実行は禁止。
P1/P2のSOURCE/result/STOPと元branch/mainを保持し、新しい結果を公開・remote SHA/全文照合後mandatory STOP。
次の担当はユーザー/GPT：Stage 1結果を基に、必要な場合だけStage 2の条件と実行scopeを判断する。
