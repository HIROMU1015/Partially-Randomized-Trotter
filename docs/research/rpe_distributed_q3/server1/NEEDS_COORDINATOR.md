# 共通担当が参照する環境不足・再現記録

## 固定DPのruntime

固定DP view `bd42dd1e95eb1d4152216d62792ad3421b803f7c`、adapter基点
`5829a7288d28395d2b3e89bfb676c23c690b9ddb`。
Python 3.12.3 + numpy1.26.4での再現：

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/rpe-distributed-q3-server1
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
RPE_FIXED_VIEW=/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-q3/routes/b-bd42dd1e9 \
PYTHONPATH=src:scripts:tests /home/AbeHiromu/venvs/trotter-common/bin/python -B \
tests/test_rpe_dp_prefix_reuse.py
```

`ValueError: Reference total DP SOURCE changed; do not silently adapt`。
[rawログ](checks/PREFIX_REGRESSION.log)。guard自体の拒否テスト1件はPASS。
Pauliのdry-checkがPASSでも、この数値constructor互換性は保証しない。
Python3.11.0rc1は今回専用prefixへ準備中。固定guardの変更はしない。

## PR-DFのネストした固定view

```bash
/home/AbeHiromu/venvs/trotter-common/bin/python -B scripts/run_rpe_optimizer_q3_job.py \
 --request docs/research/rpe_distributed_q3/server1/checks/prdf-23160cc87_REQUEST.json \
 --output /tmp/NEW_SERVER1_PRDF_DRY --mode dry-check
```

top-level `route.view` はローカル固定view `23160cc876807c0300be3de811f767b5810640a7` に解決済みでも、
そこから読み込む旧caller `e0b3cad2a024070389f757160f3a20d7da02c5ff:scripts/run_pr_df_ld_split_q3.py`
が、別の絶対pathと固定SHAを要求する。

- path: `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d2-pr-df-ld-split-q3-source-20261115`
- SHA: `27e70efe533862f38f4080aabb9cdd3ad1749fe9`
- 最初の失敗: `FileNotFoundError` (`scripts/run_pr_df_fullh_total_q3_fixed.py`)
- [元ログ](checks/prdf-23160cc87_DRY_PY312.log)

サーバー1では専用の同一SHA viewと未使用pathへのsymlinkで整備した。
他hostで同pathを用意できない場合は共通担当側の移植性課題として扱う。
固定caller/ASTをworker側独自に書き換えて回避しない。

## 接続

ローカル0のqg2をサーバー1から使うとDNS解決失敗。[ログ](checks/SSH_QG2_ATTEMPT.log)。
サーバー1には鍵/agentなし。loopback18763は拒否。
共通HANDOFFのローカル0リレー2本の起動が必要で、現在利用者へ依頼済み。
実装不具合やサーバー2自体へのネットワーク到達不能と断定しない。
