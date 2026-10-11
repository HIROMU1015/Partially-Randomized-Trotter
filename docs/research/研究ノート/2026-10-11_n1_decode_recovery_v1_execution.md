# 2026-10-11 n1_decode_recovery_v1 実行とSTOP

利用者の新しい明示認可により、固定準備commitから専用branch `n1-decode-recovery-execution-20261011` を作り、一回限定batchを実行した。N1保存real/imag vector decodeを復旧し、旧条件の一回実行を完了。全3 contextのRZ/CX選択はexact fallback、certified objective改善0%。N1_RECOVERY_COMPLETE_AWAITING_SCIENTIFIC_REVIEW。

[新結果](../n1_decode_recovery_v1_results_20261011.md)にscope・一次結果・数学的保証・科学的限界・sourceと実行plan・監査を保存した。新synthetic18件、独立保存checker 26575 checks PASS。旧kernel/input/result/STOPのbyteは保持する。現在の結果を理由に条件を拡張せず、成功を探すretryも0。

中心テーマnull、次段認可false、mandatory STOP。この新結果の科学判断をGPTへ返す。別taskのsource/結果は混合せず、このbranchから他Trackへmergeしない。以下の過去milestoneを現在の判断に合わせて書き換えない。
