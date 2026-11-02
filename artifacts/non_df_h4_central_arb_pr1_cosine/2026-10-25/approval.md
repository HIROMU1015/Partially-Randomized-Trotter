# PR1 cosine Arb包含評価の明示承認（2026-10-25）

ユーザーは「H4 Pauli PR1 cosine Arb包含評価」として、PR index1 cosineのArb包含評価を正式に承認した。
基点は `e90d6cfed1721b59c33c22bb146aebcc1fea3973`。
対象ID `596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa`。

固定linear H4／隣接1.00 Å／STO-3G、Pauli184／prefix27、S2／T=.8／q8／r4／K4、system8＋ancilla8、cosineだけ。
保存32event、order2はposition30でproduct43(+1)→product4(−1)→rotation72(−1)、order phase−1。
保存QPY/raw stateとCのexact operandsを使用し、位相・配置・共通exact ν・B_saved一回適用を維持する。

64→96→128 bit、各precisionのselected C-target1＋compiled1、最大6状態作用、scalar縮約最大3。
保存ρ予算 `19792741170690161541/7631344236317346665332736` に上端≤予算または下端>予算が決まれば即STOP。
判定不能の場合だけ昇精度し、128でも未決ならINCONCLUSIVE。幅1e-8は参考値で、決定後の高精度化は禁止。
科学SOURCE/runtime/入力/active contractを固定して専用の未使用one-shot grantを一回発行・消費する。

技術的中断時は失敗記録・旧SOURCE/grant/STOPを保全し、新SOURCE/未使用grantで限定回復する許可がある。
成功済み作用の不要な再実行はしない。今回はPR0/PF再計算、sine、追加trajectory、sampling、新compile、Hamiltonian/state生成、R2、N_G再解析、正式certificate受理は禁止。
局所数値ρをPR全lawのρへ昇格しない。結果保存・commit/push・remote確認後mandatory STOP。

§8.4/10と共通AGENTSを適用。§10.3が取得版になければ版差を記録し直接指示を優先する。
既存コード・人工検証・native監査を再利用し、独立2作用はprocessごとのArb contextで並行する。
CPU/RAM/worker数への恣意的上限、共有環境や他ユーザーへの変更はしない。
