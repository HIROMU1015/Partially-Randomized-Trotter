# Track B scripts

[SP-0.5 runner](run_sp05_synthesis_economics.py)はplan（合成0）と、別承認後のみのrunを分離する。
[結果前source review](../../../docs/tracks/algorithm_codesign/sp05_synthesis_economics_preregistration_v1.md)、
[contract／preparation](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/)。
pygridsynth一つ、catalogue一つ、8 target／23 keys。登録計測未実行。全outcomeでSTOP、retry0。


BM-0.5の限定記号監査は[audit_bm05_symbolic_equivalence.py](audit_bm05_symbolic_equivalence.py)。
standard-library Fractionと抽象非可換wordsだけを使い、degree3までの三経路を比較する。
physical inputs、行列、Hamiltonian、state、science provider、circuitは扱わない。
stdoutのJSONが[保存report](../../../artifacts/track_b_bm05_equivalence/2026-10-05/formal_word_audit_v1.json)。
本文・scopeは[BM-0.5 packet](../../../docs/tracks/algorithm_codesign/bm05_review_packet_20261005.md)。
このscriptの追加はscience runnerの実行承認ではない。

既存BF science/recovery runnerの契約・result・STOPは
[Track B index](../../../docs/tracks/algorithm_codesign/README.md)を参照する。
過去のone-shot markerやauthorizationをBMへ流用しない。
