# Track B scripts

BM-0.5の限定記号監査は[audit_bm05_symbolic_equivalence.py](audit_bm05_symbolic_equivalence.py)。
standard-library Fractionと抽象非可換wordsだけを使い、degree3までの三経路を比較する。
physical inputs、行列、Hamiltonian、state、science provider、circuitは扱わない。
stdoutのJSONが[保存report](../../../artifacts/track_b_bm05_equivalence/2026-10-05/formal_word_audit_v1.json)。
本文・scopeは[BM-0.5 packet](../../../docs/tracks/algorithm_codesign/bm05_review_packet_20261005.md)。
このscriptの追加はscience runnerの実行承認ではない。

既存BF science/recovery runnerの契約・result・STOPは
[Track B index](../../../docs/tracks/algorithm_codesign/README.md)を参照する。
過去のone-shot markerやauthorizationをBMへ流用しない。
