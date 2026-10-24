# PR1 sine Arb包含評価・明示実行承認（2026-10-25）

ユーザーは「H4 Pauli PR1 sine Arb包含評価」を正式承認した。基点 `5c549779536bc272eb500d126a94845ae2bb1d6b`。
対象はtrajectory `596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa` のsine一件だけ。
固定linear H4／隣接1.00 Å／STO-3G、Pauli184／prefix27、S2／T=.8／q8／r4／K4、system8＋ancilla8。

保存state/ray・exact ν・32event・order2の43(+)/4(-)→rotation72(-)、order phase−1・constant/global/relative phaseを保持する。
個別C-targetの虚部と実保存sine compiled QPYをArbで包含し、Sdg-before-final-H/+Imを守る。
PR law mean／binary64経験値をtargetの保証区間に流用せず、B_savedを直接差へ一度だけ適用する。

精度64→96→128bit、各precision selected C-target1＋compiled1、初回合計最大6状態作用、残差区間縮約最大3。
保存sine予算 `102330934580088554157/30525376945269386661330944` に上端≤予算／下端>予算が確定すれば即STOP。
未決時だけ昇精度し、128でも未決ならINCONCLUSIVE。幅1e-8は参考で、決定後の高精度化は禁止。

新SOURCE/runtime/入力/active contractを固定して専用未使用one-shot grantを発行・消費する。
中間承認なしで実装・人工検証・限定計算・保存監査を一括実施できる。
技術中断時は失敗call・旧SOURCE/grant/ledger/STOPを保全し、新SOURCE・未使用grantで限定回復する許可がある。成功済み作用を不要に繰り返さない。

cosine/PF/PR0再計算、追加trajectory、新compile/sampling/Hamiltonian/state生成、R2、N_G再解析、正式certificate受理は禁止。
旧SOURCE/results/grants/ledger/markers/STOPを保全し、局所ρをPR全law保証へ昇格しない。
既存コードと監査を再利用し、CPU/RAM/workerへの恣意的上限・共有環境や他ユーザーへの変更をしない。
§8.4/10を適用し、提供版に§10.3がなければ版差を記録して直接指示を優先する。
結果保存・commit/push/remote確認後mandatory STOP。
