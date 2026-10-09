# G1限定diagnostic source

[source review](../../../../docs/tracks/algorithm_codesign/g1_source_review_20261009.md)と
[固定契約](../../../../artifacts/track_b_g1_result_prior_preparation/2026-10-09/decision_packet_contract_v1.json)を参照。

- `rational_symbolic.py`：stdlib exact polynomial/rational-function kernel。
- `structure.py`：別指示後だけの固定P3独立symbolic audit。importで計算しない。
- `controller.py`：source/runtime/marker gateとA→8 echo→8 solve/verify→STOP。
- [入口](../run_g1_decision_packet.py) / [内部audit入口](../audit_g1_structure.py)。
- [59 off-domain tests](../../../../tests/tracks/algorithm_codesign/test_g1_source_preparation.py)。

既存RA-D0、共通API、旧guard/verifierを変更しない。source準備は実行認可ではない。
本model audit・固定8入力のbackend呼出しは未実施。将来の一回の結果後は必ずGPT G1へ戻す。
