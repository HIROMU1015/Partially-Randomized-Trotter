# H4非DF/DF：Bの数値精度診断とCの保証手法設計

2026-10-23 JST。利用者のB＋C一括指示に基づく専用branch `codex/non-df-h4-precision-bc-20261023`。基点は調査commit `ebe20403bcb5c4439162614f64d991a769d7ed68`、旧科学SOURCEは `0c329b30a81accb35c212be8defa66fee0dd0f12`。

Bは保存scalar40軸の必要誤差予算だけを計算する。Cは手法設計と人工テストのみ。実H4の高精度/区間信号・certificate生成、shot/G再計算、旧grantのreuse、旧native終了証拠の補充は承認されていない。

## Bの固定仕様

`h=epsilon_axis-abs(saved_bias)-u`、`u/h<=rho`、`rho=.01`を使う。正の `s=epsilon_axis-abs(saved_bias)` に対し、実数代数では `u<=rho*s/(1+rho)`。理想decimal rhoなら `u<=s/101`。s<=0では非負uをどれだけ改善しても正のheadroomを確保できない。

保存JSONのbinary64を `Fraction.from_float` 相当のexact rationalとして解釈し、Decimal100の別式 `u/rho+u=s` と照合する。元のfloat閾値0.01と理想1/100を分け、両者の小さい予算を必要条件の表示へ用いる。表示は80桁を内向き丸め。これらは新たな実測誤差上界・達成可能性・certificate・正式適格判定ではない。

Bの実行前にtechnical SOURCEをcommitし、そのSOURCE/code/runtime SHA、入力10 file SHA、40軸registry、承認原文SHA、B専用grant、固定namespaceと非返却claimを作成する。40軸と独立算術40件、R/H集約2件が予算。異常時は部分結果を保持し、自動retryしない。新しいnative終了記録はB自身のみに限定する。

## 実装と人工検証

- `src/trotterlib/h4_precision_budget.py`：Bのexact算術・独立Decimal検査・必要誤差予算。
- `src/trotterlib/h4_precision_control.py`：stdlibのB専用authority・one-shot・SHA・保存監査。
- `scripts/run_h4_precision_b.py`：prepare/launch/run/verify。Cの実H4実行機能はない。
- `src/trotterlib/h4_enclosure_prototype.py`：人工入力専用のrational区間・有限RTE scalar law・行列指数包含・誤差伝播prototype。H4 loader/assemblerはない。
- `tests/test_h4_precision_bc.py` / `tests/test_h4_precision_b_authority.py`：算術境界、旧会計の人工回帰、partial/no-retry、未承認C拒否、区間包含等。最終33 PASS、FAIL/ERROR/skip0。

[証拠入口](../artifacts/non_df_h4_precision_bc/2026-10-23/README.md)。この時点ではBの実40軸は未実行。SOURCE固定後に許可済みBだけを一回実行し、結果と完成C設計を別commitへ保存する。
