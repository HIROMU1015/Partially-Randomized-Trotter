# Trusted store技術prototype

所有する/tmp/b3-release-test-*内のTEST_ONLY.sqlite3とintentだけを操作する。workerへstore/grant/credential objectを渡さず、parent側だけでclaim/STOP/receipt linkを扱う。
TestGrantはbuiltin TEST_ONLY型とmanifest全体のbindingを持ち、real authorityへ変換する経路を持たない。実credentialや本番鍵はない。
atomic exclusive intentをO_EXCLで作り、file fsync＋directory fsync後、SQLite BEGIN IMMEDIATE/UNIQUE/FULL synchronous/WALでclaimをcommitする。
intent作成時からscopeをspent扱いとし、ACK不明・commit前死でもintentを削除／refundしない。STOP reconcileはUNKNOWN_STOPPEDを保存し、新規admissionを作らない。
12 process競合one winner、double claim、wrong grant、test revocation、STOP後ENTER、receipt link一回、intent後/commit前/commit後の異常終了を試験する。
実際に終了するのは自分の人工test processだけ。raw DB/intent/snapshot/exit codeをacceptance/raw_artificial.tar.gzへ保存。
fsyncとSQLite FULLを要求した技術事実と、物理power-loss耐久性は別。power loss実験・管理者storage保証は今回未取得。
claimはglobal一回でS/P1を内包し、STOP→SPENTへ戻さず、旧claimはアクセスも再利用もしない。

**LIVE_RELEASE_CANDIDATE_UNAUTHORIZED。実科学未実行。mandatory STOP。次の担当：ユーザー・GPT。**
