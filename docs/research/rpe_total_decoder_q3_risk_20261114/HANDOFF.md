# 引渡し：全関数推定量Z1＋T1とQ3評価器

結果は [REPORT](REPORT.md) を参照。**どのB branchにもまだ適用していない。実H4の評価も未承認である。**

## 1. Bへの取込み（技術作業）

**前提**

- 追加は新規3 fileだけで、既存fileは変更しない：
  - `src/trottertracks/rpe_total_decoder.py`
  - `src/trottertracks/rpe_total_risk.py`
  - `tests/check_rpe_total_decoder_risk.py`
- 依存するのは、Bにある既存のmodule（`rpe_deterministic_risk`、`rpe_decoder_fast`、`rpe_decoder_statistics`、`rpe_signal_statistics_caller`）と、test fixture（`check_rpe_deterministic_risk`、`check_rpe_signal_statistics_caller`、`check_pr_rpe_rmse_controller`）だけである。
- Bの固定DP runner（`run_h4_dp_risk_fixed.py`）の保護対象fileには触れない。

**手順**

```bash
cd <Bの統合worktree>
git fetch origin                       # pruneは付けない
git cherry-pick 710aa69b663ef3313439762d25c80a652a03948f
# または:
# git show origin/codex/rpe-accuracy-contract-w0-20261114:docs/research/rpe_total_decoder_q3_risk_20261114/total_estimator.patch | git apply
cd tests
PY=/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python
$PY -B check_rpe_total_decoder_risk.py
$PY -B check_rpe_deterministic_risk.py
$PY -B check_rpe_deterministic_refinement.py
$PY -B check_rpe_decoder_fast.py
```

**適用を確認した対象**

| 対象 | 結果 |
|---|---|
| B本線`e7ce23803`＋patch | 7 file／83件PASS |
| guard branch `67876f228`＋patch | 8 file／92件PASS |

**注意**

- sparse checkoutで使う場合は、`check_rpe_decoder_fast`のために`docs/research/rpe_decoder_r3_20261113`をcheckoutに含める。
- `check_pf_pr_rpe_pipeline`のためには`docs/research/h6_df19_rpe_m7_preparation_20261110`を含める。

**呼出し**

```python
from trottertracks import rpe_total_risk as R, rpe_total_decoder as T
options = dict(<Bと同じoptions>, estimator=T.ESTIMATOR,
               total_state_refinement=dict(source=..., maximum_frontier_states=..., maximum_transition_pairs=...))
records = R.evaluate(config, signal_packet, allocations, evaluation_target=..., statistical_contract=...,
                     integration=..., source=..., options=options)
records[0]['diagnostic']['deterministic_risk']    # 全関数版：MSE_upper/lower、bounds、moments、q3
records[0]['diagnostic']['partial_decoder_risk']  # 旧D.evaluateの出力（無変更）
```

- `state_refinement`を指定すると、旧部分DPも走る。これは整合検査に使えるが、Bの旧DPと同じ計算費用がかかる。
- `q3_limit=dict(MSE_upper=..., source=..., scope='MODEL_MSE_FILTER_ONLY')`を指定すると、limitとの比較を記録する（資格判定ではない）。

## 2. 実H4で最初に行う有限科学評価の案（**未承認。承認されるまで実行しない**）

| 項目 | 案 |
|---|---|
| 目的 | 固定した同じ2配分について、全関数推定量の直接モデルMSE上界・bias²・Varを、旧の上界と並べる |
| 入力 | Bの固定INPUTS（`e7ce23803:docs/research/h4_dp_risk_fixed_results_20261114/INPUTS.json`）をそのまま使う。LD1・LD2の2件、raw bundle、target proxy、ρ=1、EXACT_AFFINE、M=12、X/Y各軸`64,60,…,12` |
| 推定量 | `consecutive_dyadic_nearest_branch_principal_energy_total_v1`（評価option。canonical configは変更しない） |
| 計算 | (a) 補題上界（count表の作成後、秒単位）。(b) 全関数状態DP（追跡範囲はBのrun01と同じ8192／141,559,036）。旧部分DPは再実行しない |
| 件数・負荷 | 2評価（LD1・LD2を各1回、2 processで並列）。Bの旧DPの実績から、それぞれ約2.5時間・約1 GBと見込む |
| 必須gate | 実行前：SOURCE・INPUTSの完全一致。count表・閉包証明・coarse上界がB run01の保存値とexactに一致すること（同じコードの再計算）。実行中：質量保存、遷移allowance。実行後：`lower≤upper`、全関数DPの下界がB run01の採用上界以下であること（`hi=L_U≥L_D`なので、W0命題2によりB run01の上界は全関数版にも有効）。不成立なら停止し、retryしない |
| 出力 | 全関数版のMSEの上界・下界、モーメント区間、zero保持・tie採用の確率、旧で未解決になる確率の区間、`ε_E²`との比較（model filterとしての記録だけ） |
| 実施しないこと | 新しい配分・N探索・MC・価格・G。Minkowski予算への接続。十分Nの資格判定。旧run01の上書き |

これは保存DP結果の再解析ではなく、新しい推定量の新しい評価である。そのため、明示的な科学実行許可が必要になる。

## 3. ユーザーが判断すること

1. D1：Z1＋T1を正式な推定量として採用するか。採用する場合、canonical configの新version、IDのbridge、価格の再結合を誰が担当するか（REPORT §5）。
2. Q3契約：直接モデルMSEで合否判定し、bias²・Varを設計に使い、Minkowskiは参考とするか。
3. §2の有限科学評価（2件）を許可するか。
4. target・raw allowance・samplerのlawの移送項を、どの主張に含めるか（REPORT §8）。
