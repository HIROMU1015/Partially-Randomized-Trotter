# Storage / STOP / authority

旧独立Storage V2 consumerでRAM intent/physical state/trusted trace prefix/journal/単一call-result-log budgetを検査。
UNKNOWN非返却、STOP後ENTER禁止、inflight drain、S失敗後P1禁止、P1失敗時S保持、receipt/marker/close/delivery/callerを確認した。
保存22caseの正常2/技術失敗13/証拠不完全7を維持し、missing evidenceを補完しなかった。
追加のfresh Angle失敗もterminal STOP/F01、S失敗後P1禁止を独立確認した。

新FAKE_ONLY scope2件でatomic race32中1winner、double claim/SPENT/ACK unknown/binding mismatch/STOP後再利用の10 check PASS。
原DBをauthority_evidenceへ保存。実authority/store/grantには触れていない。
親とspawn workerでclosed entry/real factory/bitsにboolean/fake grant等を渡す30callを拒否し、core科学moduleのloadとimport試行0。
closed flags false/null/true指定を維持。real store所有・鍵/失効・実atomicity/ACK運用・power durabilityは未確立。
[authority結果](independent_authority_verification.json)、[gate](parent_worker_gate_verification.json)、[保全](pins_and_preservation_verification.json)。
