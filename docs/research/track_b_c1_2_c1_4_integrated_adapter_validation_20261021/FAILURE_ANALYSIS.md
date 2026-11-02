# C1-3失敗の静的確認（再実行なし）

失敗は`run_c13.py:60`の`STATE_BYTES_MISMATCH`。固定SOURCE・契約は変更せず、failure・terminalとstdout/stderrをそのまま保存する。実行1、retry 0。C1-4へ進まない。

R1の[保存input identity](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/input_identity.json)にはstate_vector_hash `c9aca811b5c023772d148d0331c82958bac6824b593a367cb65a5f89937f4f63`がある。この値を今回の契約へ転記した。

その定義は固定SOURCEの[`pr2_new_series_validation.py::_digest_record`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/pr2_new_series_validation.py#L235)→[`pr2_s0_s1_validation.py::_array_hash`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/pr2_s0_s1_validation.py#L145)であり、SHA256へ次を順に入力する。

1. `normalized.dtype.str.encode()`
2. shapeのcanonical JSON
3. C orderの配列bytes

一方、今回の[`adapter.py::matrix_hash`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f3a6aa1108bbc8359e585c470e23033458c57466/docs/research/track_b_c1_2_c1_4_integrated_adapter_validation_20261021/adapter.py)は、basisのlocal matrix用の**配列bytesのみ**のhashである。runnerがこの関数でstateをhashし、異なる定義のR1 state_vector_hashと直接比較した。これは今回作成した照合実装の誤りである。artifactの値を書き換えて救済しない。

入力manifest照合と`PROVENANCE PASS`はstate照合より前に完了した。固定NPZ全体のSHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`とR1保存成果のbytesは一致している。したがって今回の停止はNPZ破損やH4 model binding失敗を示す証拠ではない。失敗後にNPZを開き直したり、state hashや数値値を再計算したりしていない。

`restore_r1()`は復帰してstate照合に到達したが、この段階全体のPASS出力はない。H4のschedule出力、Green kernel、P0–P5 preflight、6 selected event、controlled operator比較、X/Y期待値、mean/path/roundは未実施。H4に対して有限bit保証が成立/不成立と判定する材料は得られていない。

元SOURCEのhash関数を**読んで定義差を確認しただけ**であり、旧科学SOURCEをimport/実行/変更して診断したものではない。fixed sourceの補修・同一契約再試行・新grant作成は行っていない。再開には、ユーザー/GPTによる修正範囲と新規実行の明示承認が必要。
