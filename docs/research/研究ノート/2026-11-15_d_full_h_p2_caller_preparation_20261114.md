# Full-H κ2 P-2 caller技術準備（2026-11-15）

実装SOURCE `0a5dcfbe99b42d2d94c1f12d1c8028694767b54c`。

22/24/26 compileの未採択案、rank検査、explicit cell/paired seed、C受渡しを技術対応。新H4計算・保存科学値再解析0。m5 HOLDOUTとA資源順序の整合は採択前に要解決。

独立した研究系列のbranchを使い、旧raw/STOP、決定論PF Stage0と32件計画、他担当worktreeを変更していない。科学計画の採択・実行へ進まない。

[REPORT](../d_full_h_p2_caller_preparation_20261114/REPORT.md)／[HANDOFF](../d_full_h_p2_caller_preparation_20261114/HANDOFF.md)／[次の有限実行案](../d_full_h_p2_caller_preparation_20261114/NEXT_SCIENCE_PLAN.md)。mandatory STOP。

追補：実装SOURCE `6069605d50270111d1510478b08e11b8e5a18d86`。保存JSONのtuple/list表現差だけを吸収するpaired event比較を追加。親receiptのJSON往復toyとmetadata回帰をPASSし、phase/order/scalar/compilerは不変。新H4科学計算なし、STOPを継続する。

追補（D7 CALIBRATIONの実行指示後）：D7採択・較正18 compileの許可を受領し、rank3・固定SOURCE・seed・保存入力を確認した。ただし固定callerがpairedを再samplingすること、非configuration較正5 cellのimplementation evidence出力が空であることをtoy spy／人工recordで確認した。今回の親再sampling禁止・証拠保存と固定SOURCE維持を同時に満たせないため、科学実行前にSTOP。取得0、SOURCE変更0、HOLDOUT/fit/N/G0。[較正STOP報告](../d_full_h_p2_calibration_20261115/REPORT.md)に限定修正案とロードマップへの影響を記載。旧判定・STOPは保全した。
