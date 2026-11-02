# 最終人工回帰と障害観測

SOURCE `860fef4a3a8ebb5a46628f4e0364e5934b5af416`で実収集・実行した**481 PASS、fail/error/skip各0**。
保護baseline356＋今回の新SOURCE検査125。275/356/339という過去の別成果を足していない。

| suite | PASS |
|---|---:|
| g6_w2c_b3_admission | 69 |
| g6_w2c_b3_readiness | 30 |
| g6_w2c_b3_driver | 36 |
| g6_w2c_cost_preparation | 48 |
| g6_w2c_b3_f01_provenance | 26 |
| g6_w2c_b3_host_integration | 66 |
| g6_w2c_b3_parallel_candidate | 36 |
| g6_w2c_b3_parallel_overhead | 45 |
| g6_w2c_b3_parallel_codec | 30 |
| g6_w2c_b3_parallel_f01 | 25 |
| g6_w2c_b3_parallel_integration | 70 |

新125＝codec/reference30＋新driver F01 25＋新Host統合70。8方式/layer正常subcaseや1000 random treeを別test件数へ加算しない。
旧356は保護された既存実装の回帰であり、新Host経路を通る125と区別している。全suiteは独立processで実行、別AIによる独立reviewではない。

機械ID/log/SHAはtests/summary.json。追加検査は3 phase、X→Y、seed/bit/event/trace/保存、atomic scope race、
SOURCE/schema/runtime、reserve後spent、retry/resume、未承認host/直接helper拒否、spawn非継承、F01 prefix、unknown/未ACK、STOP保存不能を含む。

同じ最終SOURCEの障害fixtureはreference/fused各17＝34件＋fused parent crash1＝35観測。これは481 testsとは別集計。

| 同期ケース | reserved | completed | failed | unknown | STOP→join ms | STOP後worker推定CPU ms |
|---|---:|---:|---:|---:|---:|---:|
| fused-P1_inflight | 8 | 7 | 1 | 0 | 207.05 | 966.01 |
| fused-P2_inflight | 2 | 1 | 1 | 0 | 118.97 | 403.53 |
| fused-simultaneous | 8 | 0 | 8 | 0 | 59.22 | 263.58 |

worker STOP後CPU推定のtick分解能は80 ms。共有時間/物理例外発生の厳密な同時計測ではない。
parent crashはraw START_RESERVED1、fake ledger ADMITTEDのままspent。8 workerのcleanupとEXITED/Zを確認した（{'3699510': 'EXITED', '3699511': 'EXITED', '3699512': 'EXITED', '3699513': 'EXITED', '3699514': 'EXITED', '3699515': 'EXITED', '3699516': 'EXITED', '3699517': 'EXITED'}）。
started/completed・最終親/worker CPUは不明なのでnull。正常summaryを作らず、独立observationに保存した。
最終readbackは47 archives、32272物理trial行、partial tail 2 fileを検査。
flush未commitの完全行も物理行に含む。hash/chain/state/receipt/fine/coarse残差の全assertion PASS。詳細artifact_verification.json。
物理耐久性の全故障モデル、実G7全domainの同値性、外部再現/CI、正式科学結論を認定したものではない。
