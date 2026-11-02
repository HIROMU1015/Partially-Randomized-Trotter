# H4 Pauli central B0/B0I：matched比較の限定実行結果

2026-10-25。**新central B0/B0Iは両軸868 RZと安価だが、cosineの切り捨てbiasが約0.182803で、指定5精度すべての合同shot十分条件を満たさなかった。**
ρ=0という仮想条件でもheadroomが負になるため、B0のN_total/Gはnullである。
これは旧結果の証拠不足によるnullとは区別する。nullを0・∞・費用順位へ読み替えていない。

基点 `2511bf5bcaccb664b34315a883629753c76d8fa1`、専用branch `codex/non-df-h4-pauli-b0-execution-20261025`。
実行前に固定・公開した科学SOURCEは **`268e7e9b242196fac46aba1101a513c5e3cfb645`**。
[active contract](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/active_contract.json)は準備段階の無効計画を別namespaceへ具体化したもの。
準備計画、旧SOURCE・raw・grant・ledger・marker・STOP、並行PR母平均研究には変更していない。

## 固定条件と再利用

Linear H4、隣接1.00 Å/STO-3G、保存Pauli-JW184非identity項＋constant、prefix27 Z/ZZ、tail157切り捨て。
S2/T=.8/q8、system0–7/ancilla8、Qiskit little endian、保存raw rayと共通exact norm。
r4/K4はPRとの比較metadataで、B0ではeventを生成しない。B0/B0Iは同じ1候補とした。
Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。
費用は状態準備を除くfull Hadamard wrapper、sineはSdg-before-final-Hの+Im。

保存D1 SOURCE基準Bのretained-only **非制御body** を再利用した。
432命令のcoefficient/time/label/wire/順序と、元float更新規則・保存metadataのaggregate phaseを照合してPASS。
aggregate phaseは保存値 `0.2651822507334482` と一致した。
新しい非制御bodyは生成せず、D2の `central_control_body` を一度だけ適用し、両軸wrapperを生成した。
generic制御の旧6,524 RZをcentral価格に転用していない。

数値targetはCのexact binary64 operand規約・phase、PR0のexact Pauli作用、Arb bridgeのunitary L2 recenterを再利用した。
各stepの保持forward/reverse halfとconstantを元順序で作用する。PR event、law mean、samplingは使わない。
保存Cのfull-H複素参照**全区間**を使い、`bias=(new_B0_raw_signal−saved_full_reference)/ν`、`ν=<raw|raw>` を評価した。
新full-H参照作用、Hamiltonian/state生成、state再正規化は0。
回路のbinary64位相集約・angle変換とideal targetの差はρの側に残す。

[SOURCE/input/runtime記録](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/run01/SOURCE.json)、
[body照合](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/run01/body_validation.json)、
[control対応](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/run01/control_mapping.json)。
Compilerは既存Python3.11/Qiskit1.3.0環境、Arbは既存Python3.12/python-flint0.9.0環境で、いずれも変更していない。

## 保存回路の実測費用

| B0/B0I full wrapper | RZ | CX | depth | RZ depth | CX depth |
|---|---:|---:|---:|---:|---:|
| cosine | 868 | 1520 | 1606 | 547 | 1056 |
| sine | 868 | 1520 | 1606 | 547 | 1056 |

出典：[cosine metrics](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/run01/cosine/metrics.json)、
[sine metrics](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/run01/sine/metrics.json)。input/output QPYも各directoryへ保存した。
両軸の同値は今回の実測で、一般的な測定軸独立性の証明ではない。
比較対象の保存Pauli central PRは256 trajectory平均1,140.8046875 RZ/shot、SE約1.059925である。
出典は[既存統合表](../artifacts/non_df_h4_data_catalog/2026-10-25/comparison_circuit_prices.csv)のcentral PR両軸。
新しいPR compile・平均・母平均費用推定は行っていない。

## 64 bitの複素包含・bias

[target_64.json](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/run01/target_64.json)が信号・biasのexact有理数端点とνを保存する。
下の小数は表示用の近似で、比較判定には保存端点/有理数を使用した。

| 軸 | B0 physical signal（概数） | signed bias中心 | u_enc | bias区間幅 |
|---|---:|---:|---:|---:|
| cosine | +0.0212013659737 | +0.18280322488925224 | 8.2264e-17 | 1.6453e-16 |
| sine | +0.9713087353054 | −0.015547302420011231 | 8.4229e-17 | 1.6846e-16 |

bias中心は区間両端のexact有理数平均、u_encは両端を覆うexact半幅。
保存full-H参照区間幅、norm換算、全Arb作用/内積誤差を含む。
数値的不確かさが小さいことと、full-Hに対するbiasが小さいことは別である。

数学的なB0はunitaryなので **B=1/log_B=0**。
会計のB_upは既存外向き丸め `scale_bounds(1,0)` により、保存exact値で `1+10^-95` となった。
PRのB・旧PFのlog marginを流用せず、B_upを手動で1にしていない。
実compiled残差ρは未評価であり、数値targetの包含をcompiled精度保証へ格上げしていない。

## 5精度の条件付きN/G

元と同じ `ε_axis=float(ε_sig)/sqrt(2)`、α_axis=.025、`u=u_enc+ρ`、
`h=ε_axis−|b0|−u`、`u/h≤.01`、外向きshot十分条件を適用した。
ここでのρ=0は仮想条件であり、実ρが0という認定ではない。

| ε_sig | B0 cosine | B0 sine | B0合同N_total/G（ρ=0） |
|---|---|---|---|
| .05 | bias超過 | guard成立、N_sin=22337 | null / null |
| .01 | bias超過 | bias超過 | null / null |
| .005 | bias超過 | bias超過 | null / null |
| .001 | bias超過 | bias超過 | null / null |
| .0005 | bias超過 | bias超過 | null / null |

cosine bias区間の下端は最大ε_sig=.05の軸配分より大きく、各条件のheadroomが負である。
追加shotによって、このideal B0 targetの系統biasを減らすことはできない。
sine単独の.05条件ではguard端N_sin=22786も保存したが、合同Gの成立にはcosineが必要である。
条件不成立の理由は保存 `rho_zero.reason=FIXED_GUARD_FAILED`、cosineの負headroomと `BIAS_EXCEEDS_TOLERANCE` で確認できる。

[全結果JSON](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/run01/results.json)、
[比較CSV](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/comparison_table.csv)、
[結果要約](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/result_summary.json)。
保存PRの5行のN/G・guard範囲は引用のみで、再会計0。
B0側の合同guard域が空なので、B0対PRの有限G比・費用順位はUNDETERMINEDのままとした。
これは「B0のGを無限大と置いたPR優位」ではない。
固定matched B0は要求精度を満たす代替候補にならず、安い回路価格だけではPRとの総資源比較を成立させられない、という限定結果である。
PR全lawρ・母平均費用tail・正式certificate未受理は維持する。

## 人工検証・科学call・終了

新人工8種類PASS。Compiler環境5件、Arb環境7件で、共通4件を含む延べ12実行。
2〜3qubitで既知の複素信号包含、Y/負係数/非可換順序、非単位norm=5、参照全区間、
保存body対応・位相不一致拒否、自己bias/allowance・null・昇精度判定を確認した。
[人工記録](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/artificial_checks.json)。主agent自身の技術検査である。

| 科学作業 | 上限 | 使用 |
|---|---:|---:|
| 保存body照合 / 新非制御body生成 | 1 / 0 | 1 / 0 |
| D2 central lift / 両軸wrapper | 1 / 2 | 1 / 2 |
| top-level compile | 2 | 2 |
| B0複素target状態作用 | 3 | **1（64 bit）** |
| shot scalar評価 | 20 | 11 |
| PR N/G再会計 / compiled状態作用 | 0 / 0 | 0 / 0 |

64 bitで10軸条件が全てguard成立またはbias超過に確定したため、96/128 bitは使用しなかった。
独立2 compileとtargetを並行実行し、engine全体は約6.4秒で終了した。
CPU affinity64・十分な空きメモリを観測し、人工的なCPU/RAM制限、環境設定変更、他process操作は行っていない。

親engine native exit0・reap済み、直接3 childも全exit0・wait/reap済み。
[親native終了](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/native_parent_exit.json)、
[children](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/run01/children.json)、
[mandatory STOP](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/run01/STOP.json)。
新科学grant発行/消費は0、失敗・retry・成功済み計算の再実行も0。
新しい終了証拠を、旧H4 native UNKNOWNの補完にしていない。

[保存検証](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/verification.json)と
[公開記録](../artifacts/non_df_h4_pauli_b0_execution/2026-10-25/publication_status.json)を参照。

**mandatory STOP。** この固定候補の結論を得るための追加計算は不要。
次の担当は**ユーザー**：この限定結果を比較研究でどう扱うか判断する。
**Codex**は今回の保存・公開で終了し、prefix/q探索、追加PR計算、R2-C、compiledρ診断、重要レビューへ自動移行しない。
それらが必要なら、対象・条件・計算範囲の別途明示承認を要する。
