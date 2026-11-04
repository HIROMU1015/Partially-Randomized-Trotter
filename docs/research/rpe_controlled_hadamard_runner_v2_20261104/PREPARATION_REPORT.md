# v2 selector修正・実行前検査

2026-11-04 JST。run01の固定component不在に対し、旧SOURCE/結果を残したままv2へ最小修正した。
今回の修正・再実行指示の範囲で、新SOURCE固定後に[固定12probe](EXECUTION_SCOPE.md)をrun02へ1回実行する。

変更は、保存payloadに存在するc0/c1の固定と、default静的検査での実payload照合。
c0=df-fragment-3:Z0Z2（負）、c1=df-fragment-4:Z0Z1（正・異なるbasis）を
保存ID順/sign/basisだけで選び、[selector.json](selector.json)へ完全metadataを固定した。
有利なsignalやoperatorを探索して選択していない。
科学条件・12profileの配置規則・独立参照・precision・許容差・call上限は維持した。

v1のrunner/adapter/reference/testを新ディレクトリへコピーし、selectorと静的検査・回帰テストだけを最小変更した。
reference_adapter.pyはv1とbyte一致。既存production module、source_inputsの旧入力binding、run01は変更していない。
版を分離した理由は、停止済みSOURCEと科学rawを上書きしないためである。

人工33＋前段人工34＝**67 passed in 6.99s**。
新規7件は、実保存payloadのmembership、5種類のmetadata改変拒否、fake runtimeとの元index/basis hash照合。
実保存payloadを使う追加testは文字列・schema・既存literalの照合だけであり、H4 load/build/数値計算はしていない。
既存の6rの人工pipeline、96-bit参照、両control branch、Re/+Im、spawn、raw/STOP、input不変も再検査した。
主エージェント自身の技術検査であり、独立詳細科学レビューではない。

```bash
PYTHONPATH=src <research-python> -m pytest -q \
  docs/research/rpe_controlled_hadamard_runner_v2_20261104/test_local_probes.py \
  docs/research/rpe_controlled_hadamard_preparation_20261104/test_circuit_semantics.py \
  tests/test_df_rte_qiskit_builder.py::test_controlled_event_is_exact_block_diagonal \
  tests/test_df_rte_qiskit_builder.py::test_every_small_event_matches_dense_reference_with_all_signs_and_phases \
  -p no:cacheprovider
```

default metadata検査はNumPy/Qiskit/flintをimportせず完了し、new_H4_scientific_calls=0。
実H4の回路整合性は、この準備段階では未評価である。
次は今回許可済みのrun02を固定SOURCEで1回実行し、結果・raw・evidence statusを記録して公開照合後STOPする。
