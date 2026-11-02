# B-PREとB科学実行の承認境界

現在承認済みなのは契約・SOURCE準備、静的検査、公開だけ。B science、F1、F2は未承認。今回authorization.json、科学grant、ledger、consumed marker、実run IDを発行しない。

future B runnerはF1 runnerへ機能を追加したものではない。標準libraryのverify()が、USER_EXPLICIT_BのAPPROVED_BOUND_ONE_SHOT、承認文/contract digest、固定SOURCE HEADと10file closure、spec/call/schema、recipe policy SOURCE、専用environment、exact11case/DAG、fresh outputと未使用markerを照合する。P0/I/F1用authorization、draft、未binding、件数違反はscience import前に拒否する。

B承認のためにはenvironmentの実lockを先に確定する必要がある。今回のENVIRONMENT_PLANはlockではない。新SOURCE commitは公開receiptで参照し、実行packetではユーザーが承認したexact commitを明記する。SOURCE bytesが同じでも未承認の別HEADを暗黙採用しない。

全preflight後だけconsume()が専用packet領域へO_EXCL marker、fresh outputを作る設計。これはSOURCE定義であり、今回実行しない。science caseごとにattemptを先にimmutable保存し、completedとscientific outputsは別名で保存する。失敗attempt／部分output／STOPを成功fileで上書きしない。同runの自動retry/resumeは禁止。

## 即STOP

SOURCE/spec/env/hash、exact zero/sign/scalar、JW/register/frame convention、arm coverage、finite scalar資格、call ID/DAG/件数が不一致ならSTOP。価格未資格を0として扱う、intervalを無資格で認証扱いする、build/compile/sampling/event/trajectory/F1 validationへ逸脱する、原本・共有環境を変更する、output競合もSTOP。

失敗時はそれまでの記録を保存する。成功してもB完了mandatory STOP。F1/F2、旧C1、RQ-04、追加sampling、算法探索、GPT/Claudeレビューを自動開始しない。旧grant/STOP/失敗runを再利用しない。

科学run前には実CPU/RAM/GPU/diskと他job非干渉を確認する。人工的なCPU/RAM/worker/GPU/wall上限なし。shared venv/package/cache、scheduler/quota/OS、他ユーザーprocess/filesを変更しない。固定済み旧契約の資源条件も上書きしない。
