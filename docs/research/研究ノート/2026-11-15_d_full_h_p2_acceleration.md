# Full-H P-2の並列化・高速化

ユーザーの高速化指示に対応し、実行中SOURCE d323fee1cを保全した別checkoutでpath/X/Yのprocess並列化、controlled rotation構築再利用、trial外legacy再構築のfast fingerprintを実装した。8 synthetic test PASS。toy実transpileではcompiled-vectorを含む7 gateもPASS。H4の追加compileは0。

稼働中G36223へのcheckpoint切替案は、worker終了の許可が明確でないとして実行前の自動承認レビューが拒否した。workerは停止していない。実装・検証・公開を完了してから、切替に必要な旧worker終了の明示許可を確認する。元のD7条件・最大18実compile、旧SOURCE/raw/STOPを維持する。HOLDOUT、fit、E[C]/N/Gへは進まない。

[REPORT](../d_full_h_p2_acceleration_20261115/REPORT.md)／[TESTS](../d_full_h_p2_acceleration_20261115/TESTS.txt)。H4実性能は未測定。ロードマップ本体は編集していない。
