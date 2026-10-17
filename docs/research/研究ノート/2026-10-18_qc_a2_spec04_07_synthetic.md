# 2026-10-18 A2 SPEC-04〜07の人工実装記録

ユーザーが承認したのはSOURCE実装と人工テスト、技術報告・引継ぎ・GitHub公開まで。基点 `e921d178291ff3cfc3bde5934f99bcee94d3e082`、設計 `7af83266f96b714bb3fa2f1ef5e4bc89faf37db3`。固定仕様の既知式を実装し、既存SPEC-01〜03と科学証拠は保全する。

[今回の技術報告](../qc_a2_spec04_07_synthetic_20261018/technical_report.md)、[テスト記録](../qc_a2_spec04_07_synthetic_20261018/synthetic_test_results.json)、[再開条件](../qc_a2_spec04_07_synthetic_20261018/decision_handoff.md)を正本とする。generator/checkerはCAR作用とoccupation行列、streaming行和と明示B、JW積とtrace抽出、有理式と分母を払った不等式に分けた。schema/Fraction/runtime・SPEC-01〜03 checkerは共有するため、独立開発や形式証明とは主張しない。

科学条件の採択は行わない。人工testのd/alpha/sqrt_bits/root_width/反復予算は人工fixtureの設定だけで、科学契約の既定値ではない。A2 `NOT_ADOPTED`、A1 `CLOSED`、K1-F未承認を維持。実入力certificate、新規科学計算、SPEC-08、QPE/native実装、次の科学レビューは未実施。公開と保全照合後mandatory STOP。
