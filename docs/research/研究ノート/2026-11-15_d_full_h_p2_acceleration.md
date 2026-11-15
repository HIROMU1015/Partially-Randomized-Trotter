# Full-H P-2の並列化・高速化

ユーザーの高速化指示に対応し、実行中SOURCE d323fee1cを保全した別checkoutでpath/X/Yのprocess並列化、controlled rotation構築再利用、trial外legacy再構築のfast fingerprintを実装した。8 synthetic test PASS。toy実transpileではcompiled-vectorを含む7 gateもPASS。H4の追加compileは0。

稼働中G36223へのcheckpoint切替案は、worker終了の許可が明確でないとして実行前の自動承認レビューが拒否した。workerは停止していない。実装・検証・公開を完了してから、切替に必要な旧worker終了の明示許可を確認する。元のD7条件・最大18実compile、旧SOURCE/raw/STOPを維持する。HOLDOUT、fit、E[C]/N/Gへは進まない。

[REPORT](../d_full_h_p2_acceleration_20261115/REPORT.md)／[TESTS](../d_full_h_p2_acceleration_20261115/TESTS.txt)。H4実性能は未測定。ロードマップ本体は編集していない。

## 許可済みcheckpoint引継ぎと14軸公開の追補

ユーザーの明示指示により旧G36223 path0 native checkpoint（X/Y2 compile・7 gate PASS）を保全し、旧worker終了後、新SOURCE887でsampling/compile/数値gateを再実行せずに残検査へ引き継いだ。未着手path1だけを初回取得し、path/process・X/Y/processを並列化した。旧parent STOP_CELL/exit2はCONTROLLED_AFTER_NATIVE_CHECKPOINTであり科学gate失敗ではない。過去の拒否・SOURCE・rawを保持する。

G566/G4528/Q8R566/P2R566の14軸は結果7d0c39066へ公開し、lossless receipt・固定caller元record・個別compile bytes/hash/metrics・C metadata readerによる対応をD1が検査してPASSした。C科学受入は未実施。G36223最終receiptは未取得、全18 compileの範囲を維持する。全較正引渡し又は重大gate失敗でmandatory STOP。fit/凍結、HOLDOUT、長時間外挿、law E[C]/N/Gへ進まない。ロードマップ本体は編集していない。

## 修正版0dcへの切替後のmandatory STOP

利用者の切替許可後、旧887の完了path1と元d323のpath0 nativeをhash固定して保全し、SOURCE0dcへ追加sampling/compile0で引き継いだ。旧manager/workerだけを制御付きで終了し、2:22:05の旧terminal STOPを移管理由付きで保存した。新workerは4:14.99でchild receiptを書かず異常終了、終了signalは未記録で原因未特定。最初の内部stackはfast writer内、2回目の定期dumpは途中で切れている。cgroup OOM記録0、単純人工timer診断はexit0で再現せず。

価格18 compileは保存済み（d323で16、887で2）、最終path検査16軸PASS/2軸UNKNOWN、4 cell完成・G36223未完了。数値gateの不一致を観測したとは言わず、path0残検査の欠落をPASSにしない。G36223完全較正平均recordを作らず、4 cellの完成recordと保存4価格の部分証拠をCへ分けて渡す。科学retry・条件緩和・seed変更0、mandatory STOPを維持。C fit/凍結・HOLDOUT・E[C]/N/Gには進まず、ロードマップ本体は編集しない。最新REPORT/HANDOFFと固定公開locatorを正本とする。

## 監視クラッシュの人工再現と利用者許可後の残検査再開

Python3.11.0rc1＋人工3-qubit FastWriterで定期faulthandler dump有効時にSIGSEGV(-11)、無効時に完了を確認。旧実workerのsignalは欠落しており同一原因と断定しない。利用者の修正・再実行許可を受け、親process観測へ変更、13人工testと元input/hash照合PASS。SOURCE8cce540645b99e515d72888b817782a1b9f97334をpush/remote照合後、隔離checkoutでpath0残検査のみ再開。完了path1と18 compileを再利用、追加sampling/compile0。旧SOURCE/STOP保全、最終受入16PASS/2UNKNOWNのまま。全較正引渡し又は重大gate失敗でmandatory STOP、fit/HOLDOUT/E[C]/N/Gへ進まない。

## 修正後の検証完了と全較正引渡し

SOURCE8cce540645b99e515d72888b817782a1b9f97334の保存path0残検査はwall21:39.15、peakRSS9,530,444 KiB、親/子exit0。18 unique compileを維持し追加sampling/compile0。全5 cell/18軸PASS、FAIL/UNKNOWN/価格未取得0。結果f98cb96384b775bec2c2b9e1e63fe01a6ccdf2b1をpushし、公開commitからraw復号・4軸の原compile bytes/hash・価格/実装証拠/較正record対応を照合。先行14軸の7d0固定locatorを再利用し、全10較正recordをC向けHANDOFFへ結合。D1技術照合でありC科学受入は未実施。旧STOP保全。全較正引渡しでmandatory STOP、fit/凍結・HOLDOUT6・外挿・E[C]/N/Gへ進まずロードマップ本体は統合担当へ渡す。
