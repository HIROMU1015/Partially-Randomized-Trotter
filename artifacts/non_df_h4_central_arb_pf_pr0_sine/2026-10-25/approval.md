# PF・PR0 sine Arb包含評価・実行承認（2026-10-25）

ユーザーは基点 `06ca7c838808cbde9e1c1b5030c46f260d929c32` からPF sineとPR index0 sineの実装・人工検証・限定科学計算・終了監査を一括で正式承認した。
固定linear H4／隣接1.00 Å／STO-3G、Pauli184／prefix27、S2／T=.8／q8／r4／K4、system8＋ancilla8。

PF sineは保存C虚部の包含区間全体を再利用し、新しいC-target状態作用0、compiledは各precision最大1・最大3作用。
PR0 sineは保存個別event列の虚部targetと保存compiledを各precision各1・最大6作用。
合計2論理診断・初回最大9状態作用、区間縮約は各対象最大3。B_savedはPF1、PR0の保存値を一度だけ適用する。

64→96→128bit、上端≤保存予算／下端>予算で対象ごとに即STOP。未決時だけ昇精度し、128でも未決ならINCONCLUSIVE。
PF sine予算exact `25651956654414547227/7631344236317346665332736`、PR0 sine予算exact `102330934580088554157/30525376945269386661330944`。
幅1e-8は参考で、判定後の高精度化は禁止。Sdg-before-final-H/+Im、phase・配置・作用順・同一raw ray/exact νを保持し、再正規化・経験点/PR law meanのtarget代用はしない。

実行前に新SOURCE/runtime/入力/active contractを固定し、新しい未使用one-shot grantを一回発行・消費する。
今回のroot grantが2対象を授権し、子のPARENT_AUTHORIZATIONは委譲参照のみ（別grant発行/消費ではない）。各対象・各作用のcall上限は個別非返還ledgerで守る。
意図しない技術中断は失敗call・旧SOURCE/grant/ledger/STOPを保全し、新SOURCE・未使用grantで限定回復する許可がある。成功済み計算は不要に繰り返さない。

既存cosine/PR1再計算、追加trajectory、新compile/sampling/Hamiltonian/state生成、R2/N_G再解析、正式certificate受理は禁止。
旧SOURCE/results/grants/ledger/markers/STOPを保全し、旧4件は保存値の引用だけで6局所一覧へまとめる。
既存コード・検証を再利用。CPU/RAM/workerの恣意的上限、共有環境や他ユーザーへの変更は0。
提供policy§8.4/10/10.1/10.2を適用し、§10.3がない版差を記録して今回の直接指示を優先する。
結果保存・commit/push/remote確認後mandatory STOP。6件が揃ってもPR全law保証や正式certificate受理へ昇格しない。
