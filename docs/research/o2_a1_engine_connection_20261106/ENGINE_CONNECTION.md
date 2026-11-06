# A1 N2補完：保存gate/stateと測定engineの技術接続

2026-11-06 JST、SERVER1 GPUA100-gleap-01。**実装・人工検査のみ。実N1/N2入力での新しい数値上界は未取得、科学実行は未許可。**

対象は保存linear H4/1.00 Å/STO-3G/legacy DF12/8qubit/全4電子number sector/L_D3、m0/q1/δ=.02/r1/K2/coefficient_atol=0、X/Y、理想gate・保存DFモデル限定。
N1結果744e1a09661ebba6841fe5faf456f88c41c47780、N2結果5e9bd08068738cdf1070bdc43bee8194b113ae33、N2科学SOURCE a4141795bd3e23670612d0f0d4a43678457a1a37、契約2b2e6c1bc3b2d201d1a6d1238172187d68aea6abを固定する。
N1 preparation/law/policyとN2 state/spectral witnessのbytes・identityを静的に照合する。旧REPORT・UNKNOWN・STOPを変更せず、新規の実数値計算は別許可まで行わない。

## 実装と再利用

[adapter](../../../src/trottertracks/algorithm_codesign/a1_engine_connection.py)と[人工tests](../../../tests/tracks/algorithm_codesign/test_a1_engine_connection.py)を追加した。共有core・既存SOURCEに変更はない。

| 新adapterの処理 | 再利用する固定SOURCE・関数 | 保持する意味 |
|---|---|---|
| inspect_fixed_bundle | Gitの固定commit bytes、N1 RESULT/SOURCE、N2 SOURCE/STATE_RESULT | H/partition/preparation/tail/catalog/law/policy/stateをbind。vectorやwitnessの再計算なし |
| restore_operations / SavedEngine | DFBasisRegistry.register、describe_basis_change_operations、既存DFDeterministic*Spec | 保存RZ/XXPlusYYのparameterを復元し、局所matrix hash・順序・basis hashの完全一致を要求。eighや新basis構成なし |
| eventの明示構成 | rte._make_event、DFRTEEventSequenceCircuitRequest | 入力は適用順products→rotation。SOURCEへrotation-first配列で渡す。nominal event_probabilityを重みやQにしない |
| selected basis | make_run_threshold_basis_plan、QiskitDFRTEEventCircuitBuilder.build_sequence | 保存support定義とsupport_run_le_1。source basis runが長い場合は既存規則どおり保存full basisを使う。新しい選択・学習なし |
| q1 outer blocks | _append_aggregated_phases、_append_block | boundary_optimizedの既存SOURCEと同じscalar/identity合算、D forward→event→D reverse。q1にないboundary fusionを追加しない |
| X/Y wrapper | QiskitRPEHadamardInterrogationBuilder._build_validated | H→controlled列→H、YのみSdg→H。ancilla最後、controlを重ねない、measurementなし |
| directed scalar計算 | df_correction_comparison.intervals | mpmath1.3.0 iv100dps、正確なbinary64入力変換・endpoint・外向き256bit保存を再利用 |

SOURCE上、boundary_optimized q1はscalar phaseを `-dt*(c+κ)` の演算順で合算する。別々に丸めた位相の和へ置換しない。
RTE event phaseと負係数product signは既存builderのancilla PhaseGateに入り、測定後に再度掛けない。
対象は実際のselected logical gate列であり、full-basis/algebraic referenceとの同値を仮定しない。各gateのbinary64 angleをexact angleとして理想unitaryを定義する。
scalar/identity/event phaseを消したchannel同値でHadamard信号を代用しない。

保存payloadには実行可能gate objectがないため、named gateの小さな復元adapterが必要だった。
既存fixed_signal_evaluationをそのまま使うとpreparation再実行・algebraic reference・MC/費用等を含む経路に入るため、その全runnerは起動しない。
既存statevector実装は人工検査の独立な対照に使う。実engineには、その内部BLAS誤差を根拠なく0にする代わりに、標準Fractionを使う小さな局所gate contractionを採用した。
これはこのgate集合向けのadapterで、汎用数値ライブラリ・新しいsolverではない。

## Norm・Born・測定bitへの接続

engineはN2の保存Qiskit順vector vをprivateに保持する。N2が保証したstateはv/||v||である。
各局所Qiskit Operatorのbinary64実部/虚部をexact Fraction pairに変換し、gate適用の全積和をFractionで行う。
生成されたgate行列の近似誤差は別途intervalで課金し、適用時の積和丸めは生じない。
最終joint vector wについて、Born確率を `p_plus=Σ(ancilla=0)|w_i|² / Σ|w_i|²` とexactに計算する。
初期vのnormとその平方根はこの比で正確に相殺する。丸めgateで生じた最終normの変化は、[誤差計画](ENGINE_ERROR_PLAN.md)の4E項に含める。
raw probabilityのclipや、未知norm誤差の無視ではない。分母≤0、確率範囲外、非有限gate・unsupported gateはSTOPする。

probabilityはengine内部の値であり、CVの特徴・学習・配分には渡さない。将来の測定facadeはengine内部で独立wordを受け、bitだけを外へ出す必要がある。
今回はそのRNG/fresh-bit facade、train/cal/prodは実装・実行しない。純粋なthreshold_countだけがあり、53-bitのkに対してk<ceil(2^53 p_plus)という接続を規定する。
X/Yは別fresh shotとする条件を維持し、bit0→+1、bit1→−1。B_fによるshot weightは掛けずw=1を維持する。

## 人工検査と実入力の境界

人工2-mode DF fixtureの保存形式を復元し、全人工path・X/Yで、既存full repeated+selected+Hadamard builderとgate名・angle hex・qubit位置・順序を比較する。
RZ、XXPlusYY（β≠0）、controlled RZ/RZZ、CZ、Phase、H、Sdgを検査し、人工statevector作用を対照にexact kernelを確認する。
既知phaseのX/Y符号、nonunit normのexact処理、[0,1] guard、grid endpoints、interval包含、局所matrix改変・identity不一致時STOPも確認する。
これらは人工correctnessであって、実分子の少数pathから全supportへ外挿する証拠ではない。

今回の実N1/N2へのアクセスはREPORT/SOURCE/schemaの読解と固定bytes・identityの静的検査だけ。
`from_fixed_bundle`、`SavedEngine.qualify`、`qualify_fixed_bundle`を実データで呼ぶこと、実stateへのprobability計算は開始しない。
N1/N2のactual law・希少control・pilot数、state・spectral witness、旧STOPとC1凍結を維持する。

## 未実行の次回入口

`qualify_fixed_bundle(root,fresh_output)`は**別途明示許可が必要な科学関数**として実装した。import時や人工test/static inspectionからは起動しない。
SOURCE/依存環境lockを照合した後、保存bundleを一回復元し、全局所gate集合と構造的最大列長から一様上界を一回取得する。実path build/state evolution/測定は不要である。
入力・関数・内部回数・出力とSTOPは[ENGINE_ERROR_PLAN](ENGINE_ERROR_PLAN.md)の後半に定める。
得た数値があっても1/32や精度/予算の採択、N3実行許可にはならない。
