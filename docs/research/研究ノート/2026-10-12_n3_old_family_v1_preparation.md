# 2026-10-12 N3旧条件族診断 v1 準備履歴

## 完了済み科学レビュー受領

S1の保存幅差を支持し、旧族の保存coverageを越えた回収／分離診断を限定継続するレビューを受領した。
利用者はCodexで可能な準備を次の科学判断・明示実行指示の地点まで進めるよう指定した。
レビュー第11節は新診断への別認可を必要とするため、source/独立checker/synthetic/input/plan/公開までが今回の範囲。
[準備報告](../n3_old_family_v1_preparation_20261012.md)、[数学・実行契約](../n3_old_family_v1_mathematical_contract_20261012.md)。

## 実装・synthetic

μ0への単調性による一変数等価性と、PSDなWの両端strict負traceを実装した。
原FIVE→old_candidateの対応を静的確認し、NORM不成立を別に判定する契約を保持する。
23 synthetic PASS。dense9 testに恣意的なcheck-count期待の失敗が一度あり、exact certificate PASSの確認へ訂正。raw logを保持。
新しいS1診断0。元OCC勝者の有理上端・係数・U/cはlosslessコピーだけ。元primary/gate/STOPとN1/N2/別Trackを保持する。
