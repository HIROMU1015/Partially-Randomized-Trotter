# 独立offline再現

Python3.10以降・標準library。repository rootから、未作成の/tmp名を使う。

```bash
P=docs/research/track_b_g6_w2c_b3_global_bundle_independent_acceptance_local_20261019
PYTHONDONTWRITEBYTECODE=1 python3 "$P/restore_public_raw.py" --new-directory /tmp/b3-global-public-new
PYTHONDONTWRITEBYTECODE=1 python3 "$P/restore_local_artificial_raw.py" --new-directory /tmp/b3-global-local-new --public-raw /tmp/b3-global-public-new
PYTHONDONTWRITEBYTECODE=1 python3 "$P/run_offline_acceptance.py" --public-raw /tmp/b3-global-public-new --fresh-raw /tmp/b3-global-local-new/fresh_cases --new-output /tmp/b3-global-offline-new
PYTHONDONTWRITEBYTECODE=1 python3 "$P/run_independent_tests.py" --public-raw /tmp/b3-global-public-new --fresh-raw /tmp/b3-global-local-new/fresh_cases --new-output /tmp/b3-global-tests-new
```

保存freshのoffline再読ではruntimeは宣言hashとして扱い、以前の絶対runtime_rootや旧Python実体へアクセスしない。`--verify-current-fresh-runtime`は実測時の同じruntime tree/Pythonが実際に存在する場合だけ用いる。どちらも科学GOは返さず、保存状態の分類は同じ。

単一caseは`PYTHONPATH=src python3 -m trottertracks.algorithm_codesign.g6_w2c_b3_bundle_independent_cli_v1 --case ... --fixed-tree "$P/fixed_reference/bridge" --new-output ...`。比較CSVは完全な二条件bundleだけに行を出す。入力と重なる出力・既存出力への上書きを拒否。

旧623は基点の`docs/research/track_b_g6_w2c_b3_limited_pilot_independent_acceptance_20261019/run_independent_tests.py`。V2独立57とStorage41は別runtimeに固定SOURCE/test/inputをmaterializeした。初回のsymlink参照や欠落依存は診断として保持し、最終成功と区別した。固定受入77126767の公開prepare/run手順と今回fixed_reference/各inventoryを参照。実行済みのabsolute server pathへ書込まない。

新しい人工再生成が必要な場合、`prepare_artificial_runtime.py --new-runtime /tmp/...`はe099 objectの既存Git取得可能性と62 pinned bytesを確認して所有runtimeだけを準備する。実行SOURCEの採択・科学入口の解除は行わない。現artifactのrun specsと固定run_case.pyをfresh専用process/rootで使う。今後の新規科学実行はこの再現手順から許可されない。
