# 現行新B N3の正規停止・保全結果

2026-11-07 JST、SERVER1 GPUA100-gleap-01。**承認された親SIGTERM一回による協調停止は成功。強制kill・process group停止・再起動は0。**

## 対象と実際の操作

科学SOURCE=`ee49b53fd6f9a17d715edb1df3aac8cbf5731d7a`、campaign=`A1-N3-B-H4-m0-q1-20261106-run01`、seed20261107。
保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits/全4電子sector、L_D3/m0/q1/δ=.02/r1/K2/atol0、X/Y、N1 actual Q/selected full・N2保証ground・認証済exact engine。

縮小設計5c27684d904aed9f4f9f2787752c917bb46d684cのCURRENT_N3_STATUSに従い、SOURCE/全固定code hash、親PID/start ticks/boot/PGID/SID、UID、自分の所有、既存STOPなしを再確認した。
pidfdで親PID3026040へSIGTERMを一回だけ送信した。受信記録は**2026-11-07 00:06:59.618 JST**（2026-11-06T15:06:59.618051Z）。
STOP_REQUESTEDは同00:06:59.619、POOL_DRAIN_STARTは同00:06:59.620で、STOPが回収より先に保存されている。
その後は読み取り監視だけを行い、既存finalizerが進行中の価格処理を回収するのを待った。新しいshot/compileの再投入やsignal再送をしていない。

00:10:59 JSTの確認で親・全125workerが不在、STOP/PARENT_EXIT/WORKER_EXIT_CODES/DISPATCH_EXIT/outer EXITが保存されていた。
親はhandlerでSystemExit143を捕捉し、semantic STOPとして通常exit1を返した。これをOSからのSIGKILL/OOMや科学的guard失敗と分類しない。
worker125は全てprocess exit0、finalization_errorsは空。既存managerのcleanup126記録は全てIDENTITY_GONE_OR_CHANGEDで、追加signalの送信はなかった。

## 最終取得と費用

| 方式 | 保存bit | 価格取得済み | 既知full RZ | 備考 |
|---|---:|---:|---:|---|
| A1 | 640 | 640 | 3,133,028 | X/Y train各320、rank/range通過 |
| B1-O | 256 | 256 | 1,247,868 | X/Y pilot各128 |
| B1-OL | 443 | 422 | 2,054,713 | X pilot完了、Yの途中で操作停止 |
| B0 | 0 | 0 | 0 | 未着手。完了方式の0費用とはしない |
| 合計 | **1,339** | **1,318** | **6,435,609** | 価格欠測21件あり |

1374 trialを計画/path固定/開始。内訳はbit+価格1318、bitあり価格なし21、bit未保存35。
**総RZはUNKNOWN**。6,435,609は既知価格部分の下限で、未取得費用を0とした完成費用ではない。未完了attemptの古典費用は記録範囲とUNKNOWNを保持する。
実callはexact engine1374、Born1339、selected full build2692、compile666、53-bit words3413。cal/N凍結/prodは未実行。N1 preparation/CDF、N2 solve/spectral、engine再認証は0。
全4方式の精度達成・CV改善・期待費用優位は未判定。今回の停止はユーザー承認による研究scope移行であり、CVの失敗判定ではない。

## 保全方法

[stop artifact](../../../artifacts/o2_a1_n3_stop_and_pilot_preparation/2026-11-07/stop/)へSOURCE、STOP、CALL_COUNTS、ACCOUNTING、ATTEMPTS、CAMPAIGN_LEDGER、exit/intent/操作監査をcopyした。
FULL_STOPPED_RUN.tar.gzにはscience・manager・operatorの計1090ファイルを、cache SQLiteのWAL/SHM・lock・全journal・全データとともに保存した。
RAW_ARCHIVE_MANIFESTは各memberのSHA256/sizeとarchive SHA256を持ち、展開byte hashが原本と全件一致した。SQLiteを開いてcheckpointする操作はしていない。
元worktreeのSOURCE/入力/既存結果を上書きせず、raw journal/cache/新しいSTOPも原位置に残した。旧d97eの397件、初回B import失敗、旧grant/ledger/marker、元main/dirty/untracked、C1凍結を保全した。
今後の752-shotへ、これらの測定・係数・word・cacheを混入しない。旧結果は参考資料・支出履歴としてのみ扱う。
