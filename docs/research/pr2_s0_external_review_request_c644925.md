# GPTへの指示 — PR-2 S0 input-reproduction STOP後レビュー

以下のGitHub repository/branch/commit packetを読み、PR-2のS0後判断を独立にレビューしてください。

- repository: `HIROMU1015/Partially-Randomized-Trotter`
- branch: `all-r-coherent-opt2-reoptimization`
- result-prior authorization commit: `e9bffb85f9ed57712bb83150172a6a4662cecf7f`
- S0/S1 source commit: `c644925b50587072784846df09bf02c39e8453e1`
- evidence commit: この文書を含む最新commit

## 最初に読むもの

1. `docs/research/pr2_s0_s1_execution_amendment_v3.md`
2. `artifacts/pr2_s1_s3_preregistration/2026-09-28/pr2_s0_s1_authorization_manifest_v3.json`
3. `docs/research/pr2_s0_reproduction_stop_c644925.md`
4. `artifacts/pr2_s0_s1_validation/2026-09-28/pr2_s0_validation_v1.json`
5. `src/trotterlib/pr2_s0_s1_validation.py`
6. `scripts/run_pr2_s0_s1_validation.py`
7. `tests/test_pr2_s0_s1_validation.py`
8. `artifacts/pr2_s0_s1_validation/2026-09-28/pr2_s0_s1_tests_c644925.xml`

旧pilotとの比較に必要なら次も読むこと。

9. `docs/research/pr2_pr3_minimal_pilot_preregistration.md`
10. `artifacts/pr2_pr3_minimal_pilot/2026-09-27/pr2_pr3_minimal_pilot_v1.json`
11. `src/trotterlib/pr2_pr3_minimal_pilot.py`

## 固定事実

- S0のenvironment gateは完全一致した。
- development expected Hamiltonian hashは
  `d8b4aaf21afcc3935d5b5aa4d0805b358c5ec670d8104d25807c7cd0620a3dc3`。
- observed hashは
  `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。
- S0 statusは`STOP_INPUT_REPRODUCTION_MISMATCH`、`S1_authorized=false`。
- ground energy差は約`1.38e-14` Ha、rank 3/6/9 residual $\lambda_R$差は`1.2e-15`以下だが、
  Hamiltonian/tail byte hashは不一致。
- H4 1.30 Å held-out inputはsnapshotだけfreezeし、signal/cost/rankingは未開封。
- candidate signal、compile、trajectory sampling、quantum shot、S1/S2/S3はいずれも0件。
- S0/S1専用・関連test packetは123 passed。

## レビューしてほしい問い

1. amendment v3と事前登録に照らし、CodexがS0を停止しS1を実行しなかった判断は正しいか。
2. mismatch診断は十分か。追加のread-only比較で原因を特定できるか、それともpilot時にarray snapshotを
   保存しなかったため厳密な原因同定は不可能か。
3. DF factorの符号・縮退部分空間のbasisなど、表現gaugeだけを除くcanonical hashを新設することは
   科学的に妥当か。妥当な場合も、**今回の結果を見た後のgate緩和ではなく、新しい結果前amendment、
   canonicalization test、pilot/S0双方への同一規則の適用**が必要か。
4. 旧pilot artifactだけからcanonical hashを再構成できないなら、PR-2をこの地点で終了すべきか、または
   「pilotを新snapshotで再実行して研究をやり直す」別研究としてのみ再開可能か。
5. 現時点の証拠から、S1へ進む正当化は一切あるか。

## 禁止する助言

- energyや$\lambda_R$が近いことだけを理由に、observed snapshotをpilot snapshotと同一扱いする。
- expected hashをobserved hashへ書き換えて、そのままS1を実行する。
- 旧pilot結果と新snapshotのS1結果を一つの同一入力系列として混ぜる。
- held-out signal/cost/rankingを開封する。
- S2/S3、32/128 expected-cost Monte Carlo、resource winner判定、追加rank/geometry/precision、H12、長RPE、
  final total costへ進む。

## 回答形式

次の順で回答してください。

1. `REVIEW_DECISION`を一つ:
   - `CONFIRM_TERMINAL_STOP`
   - `AMEND_AND_RESTART_FROM_NEW_S0`
   - `REDESIGN_AS_NEW_PILOT`
2. 致命的問題
3. 追加で必要なread-only確認
4. canonicalizationを許す場合の最低要件
5. PR-2の研究価値・negative resultとして残る内容
6. 次にCodexへ許可する作業を、数値実行の有無を含めて一つだけ

不明な点は推測で埋めず、該当file/pathと不足情報を明記してください。
