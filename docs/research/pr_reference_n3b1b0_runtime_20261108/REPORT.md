# 新A N3-B1b-0：Emitter実runtime適合性・strict位相保証

2026-11-08 JST、SERVER 2。固定参照 **4abf269d40b14636ffe0f7e2607aa293d4948b0a**。専用branchは `codex/pr-reference-n3b1b0-runtime-20261108`。

**判定：RUNTIME_EMITTER_COMPATIBLE。** 固定runtime・通常IEEE f64意味論・named ideal Clifford+T gate・宣言したメモリ内の位相経路に限り、既存emitterを変更せずN3-B1aの上界を適用できる。C_phiに本質的な不一致は見つからなかった。限定技術検証を完了し、**mandatory STOP**。

実43基本word、完成辞書、本番provider、全support CT RPE成功、総Tの資格ではない。43 key合成／新guard／科学価格計算は実行していない。

## 固定条件

Linear H4・1.00 Å・STO-3G・DF12、元H厳密ground供給・状態準備費除外。Partial S2・LD3・h=.02・finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、m7/q128/r1・保存73 bits。最終native RZ precision10^-11、b36、指数-36…6、43 key、基本Frobenius要求10^-13を変更していない。

Qiskit1.3.0/compiler A、B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP、D融合／basis共有／scalar・event／X/Yを維持。保存199 shots/axis、energy1.5936e-4 Ha、failure.05、phase.4、alpha1/720も不変更。up_to_phase=False、exact W、相対位相、全連結wordのactual adjointを保持し、phase fit／pi snap／無条件2pi key同一視／後置controlは用いない。

科学入力の正本は[N3-B0](../pr_reference_n3b0_numeric_20261108/REPORT.md) `9c60c43135666a9ab5497a43a4b1695ab371650f`の[FIXED_PARAMETERS](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/FIXED_PARAMETERS.json)、[ERROR_BUDGET](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/ERROR_BUDGET.json)、[coverage](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/BASIC_KEY_COVERAGE.json)。0/43という保存coverageを再選択・再集計していない。

## SOURCEと実runtime

Python3.12.3、Qiskit1.3.0、pygridsynth2.0.0、mpmath1.3.0、numpy1.26.4、x86_64/GPUA100-gleap02。実行環境は[ENVIRONMENT.json](../../../artifacts/pr_reference_n3b1b0_runtime/2026-11-08/ENVIRONMENT.json)。

[検証SOURCE](../../../scripts/verify_pr_reference_n3b1b0_runtime.py)は、旧lower_nativeとexact_word_matrixの**変更していない関数AST**を抽出して実行した。無関係な科学moduleの初期化・runner・guardを起動しないための限定再利用であり、emitterの再実装ではない。CT-1 word_matricesからactual adjointの文字対応もそのまま再利用した。guard定義を検証namespaceへ読み込んでいない。

N3-B1aで固定したQiskit公式commit `896d1c8b72888a52a46081a6d1f4de8c56db8aea`とのPython8ファイル一致を、実際にimportした環境で確認。追加でOperator／circuit_to_instructionの2ファイルも同commitのblobとbyte一致した。使用中accelerate wheelのbinary hashと導入済みdistribution RECORDとの一致も保存した。[SOURCE.json](../../../artifacts/pr_reference_n3b1b0_runtime/2026-11-08/SOURCE.json)に検証SOURCE・再利用関数・保存archive/member・入力hashを記録した。

Rust wheelのbuild flagsは公開されていない。PythonのCFLAGSには調べたunsafe float flagsがなく、fegetroundは読取専用でFE_TONEAREST、ties-to-evenの有限確認も一致した。Rustの全機械語／libm全入力の形式検証を行ったという意味ではない。資格はSOURCE、通常IEEE意味論、固定binary、実runtimeの有限適合確認を対応させた技術判断である。

## C_phiへの対応

| 条件 | 結果 | 根拠・範囲 |
|---|---|---|
| C1：通常binary64演算 | PASS | radix2／mantissa53、fegetround=0、丸めtie確認、SOURCEのf64演算。機械語の形式証明は含まない |
| C2：Pi0/P/Rust周期 | PASS | math.pi=`0x1.921fb54442d18p+1`、P=`0x1.921fb54442d18p-1`。compiled CircuitData setterでtauと両隣のbinary値を確認 |
| C3：named gateと順序 | PASS | Written product逆順、H/S/T/Tdg/X、Sdgを含む全体inverseがSOURCE・exact word・有限診断と整合 |
| C4：W処理 | PASS | 8 residue、wrap／複数wrap、正負setter、inverse、再設定、composeの固定例が一致。一般閉包はN3-B1aの代数証明に依る |
| C5：数値carrierとactual adjoint | PASS | [0,tau]を覆う数値phase、連結全体のinverse。symbolic／control-flow／後置controlは使わない |
| C6：有限経路 | PASS | 43非自明join＋最終join、全体inverse最大1、非zero phase初期設定1、QPY0。上界適用経路のjoinはinplace=True |

各条件のSOURCE根拠とruntime観測を[RUNTIME_COMPATIBILITY.json](../../../artifacts/pr_reference_n3b1b0_runtime/2026-11-08/RUNTIME_COMPATIBILITY.json)に保存した。Rust PIは直接exportされた定数を読んだのではなく、固定SOURCEとcompiled setterのwrap／隣接値から対応を確認した。

## 局所テスト結果

[TEST_INPUTS](../../../artifacts/pr_reference_n3b1b0_runtime/2026-11-08/TEST_INPUTS.json)は人工fixtureと保存archive内の2 memberを固定する。新しい科学的RZ合成入力や43基本keyの代用ではない。

| テスト | 件数 | 結果 |
|---|---:|---|
| W residue／wrap／複数wrap／正負／inverse／再設定 | 24 | 全PASS、期待binary phaseと一致 |
| 数値carrier pair／compose／inverse／setter境界 | 64 | 全PASS、厳密Fractionによるbinary周期参照と整合 |
| 人工6 word＋保存2 wordの正負 | 16 | 全PASS、named gate順／全体adjoint／strict行列診断 |
| 最大43人工chunk、正負、2-qubit埋込み | 2 | 全PASS、単一emissionとcarrier結合の両方を確認 |
| copyのtau境界 | 1 | tau→0を確認。上界適用経路とは分離 |

人工43 chunkは3種類の短い固定wordの反復であり、未取得の43基本wordではない。保存2 wordはCT-4B archiveのlexicographic先頭2 JSONを選び、元の採用word／guard／価格を変更していない。

主なword／composite診断の最大観測operator差は約1.73e-14。診断tolerance1e-10はstrict残余のcertificateに使っていない。exact cyclotomic行列のinterval midpointをnumpyへ変換した数値診断と、SOURCE・代数に基づく上界を区別する。各chunkだけをadjointにして元順で掛ける誤りも、非可換2 chunkの対照で区別できた。

### Copy・Operator・QPYの範囲

SOURCEを追加追跡すると、defaultのcompose(inplace=False)はQuantumCircuit.copy経由でCircuitDataを再構成し、位相を再正規化する。この追加処理を無条件に無視しない。今回、上界を適用する43-carrier／最終joinは**inplace=True**に明示した。既存lower_nativeも元から同経路であり、旧SOURCEの修正は不要。

負の最小subnormalから作ったphase=tauは、再設定／copyで0へ移る。binary周期での差が0でも、真の2piとの差は別に存在する。またOperator(circuit)はto_instruction経由で位相を再設定するため、このcopy境界のOperator差は0と観測された。これをexact scalar差0の証拠にしない。保存位相を直接expで評価した別の浮動診断も[PHASE_TESTS](../../../artifacts/pr_reference_n3b1b0_runtime/2026-11-08/PHASE_TESTS.json)に併記した。

今回の局所emitter経路はメモリ内で完結し、保存するのはword metadata／JSON結果である。QPY往復は不要なので実行していない。QPY、QASM、symbolic phase、任意copy／後処理へ資格を自動移送しない。

## 誤差上界と判定範囲

[N3-B1aの記号証明](../pr_reference_n3b1a_emitter_static_20261108/PHASE_ERROR_BOUND.md)を変更していない。通常IEEE条件の下で固定PのW加算が8状態に閉じ、W数を未取得値で仮定せずにlocal phaseを覆える。

宣言したinplace経路では、43 local変換＋非自明join、最終join、全体inverse最大1、追加再設定最大2の保存十分係数を使える。実テストの追加非zero初期設定は1、QPYは0である。

\[
e_{\rm emit} < [(43+1)64+32+2\cdot16]2^{-53}
=\frac{45}{140737488355328}
<2^{-41}<R_{\rm emit}^{lo}.
\]

最後の比較はN3-B0保存Fractionをそのまま用いて厳密に検算した。局所1-qubit Frobenius境界であり、jointへの移送はspectral embedding／telescopingで扱う。joint Frobeniusへdimension係数なしで流用しない。量子化／基本guard／emitterを二重計上しない。詳細は[EMITTER_BOUND_ASSESSMENT](EMITTER_BOUND_ASSESSMENT.md)。

この資格はemitter部分に限る。未知wordの合成成功／停止、基本guard、実provider、既存C0 wordのguard＋emitter余裕、full-support CT signal／RPE成功、価格は別途未資格。保存199 shotsを再解析してPASSにした結果ではない。

## 保全・再現・STOP

正式共通運用指示2026-10-25、親／worktree AGENTS.md、PROJECT_MAP、N3-B1a全5文書とSOURCE参照、N3-B0、N3-A／N1／CT-4B／CT-5Bの固定資料を適用した。今回の検証は著者Codexによる技術検証であり、独立科学レビューではない。

最終SOURCEは同じ固定caseを再実行してPASSを確認した。初期検証後の差分は各C_phi根拠の出力、copyに関するSOURCE対応、Operator診断の留保を明示するもの。科学条件・基本key・precisionの探索はしていない。

再現は指定venvで `PYTHONDONTWRITEBYTECODE=1 …/.venv/bin/python scripts/verify_pr_reference_n3b1b0_runtime.py`。実処理は約0.25秒の局所算術／小行列で、1 processで完了した。affinity64・空きメモリを確認し、資源cap／共有設定変更を行っていない。旧SOURCE／raw／result／grant／ledger／marker／STOP、旧venvを変更せず、他Track／他ユーザーへ干渉していない。

新規合成／guard／完成辞書／本番provider／科学T価格／PF build・compile／sampling・simulation／N2・SK／b・precision再選択／199-shot積／全18round総T・T-depthは0。

[STOP.json](../../../artifacts/pr_reference_n3b1b0_runtime/2026-11-08/STOP.json)のとおり範囲完了でmandatory STOP。次担当は**GPT／ユーザー**：[NEXT_SCOPE](NEXT_SCOPE.md)に沿って、固定43 keyの取得と最小provider検証を別途指示・承認するか判断する。N3-B1b-1へ自動進行しない。
