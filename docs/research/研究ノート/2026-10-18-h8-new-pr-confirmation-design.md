# H8新PR独立確認の設計

利用者が公開済み64c85f7a…を保全した上で、H8 2候補の対象・比較・必要nの設計を指示。
[固定条件・標本数案・限界](../track_a_h8_new_pr_confirmation_design_20261018.md)。

各48軌道・192 compileを提案。保存n8のSD4倍normalシナリオの最大46を8-task単位で切上げた案で、
実power/母平均保証ではない。両候補の保存n8は全Taylor次数0だった。
rare可視性の低さから、独立点費用の再現と母平均の保証を分離し、条件付きpilotの移送はしない。
人工11件PASS。新sampling/compile/native build/seed生成0、無効grant・未sealでSTOP。
H6高次PFの13/45/45は別契約。実行は別の利用者承認を必要とする。
