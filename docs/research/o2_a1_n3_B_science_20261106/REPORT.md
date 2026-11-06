# A1 N3案B：起動失敗・科学開始前STOP

**NOT_STARTED_LAUNCH_IMPORT_FAILURE / mandatory STOP。新規科学計算は0件。**

一条件4方式×X/Y・一campaign、seed20261107を採択し、SOURCEを5f2eac41cbc324643d663b8110b1cf127aed54dfへ公開した。16ファイルの公開本文・blobとtree 9eb2e9bbd1f2fe131fb77c97f15fec3fd88f3269を照合後、自分の一時cronに一回投入した。
しかし追加entryが自分のsrcをsys.pathへ登録しておらず、PYTHONPATHのないcron環境で`ModuleNotFoundError: No module named 'trottertracks'`となった。実装上の不備であり、OS killやOOMを推測する必要のある中断ではない。

## 取得状況と保全

[失敗証拠](LAUNCH_FAILURE_EVIDENCE.json)に保存したcron.logにはimport失敗が1件。dispatch本体・CONSUMED receipt・外側manager・親driverには到達していない。science output/INVOCATION/SOURCEは作成されていない。
科学bundle binding、PCG64 stream生成、word、train/cal/prod、bit、gate/Born、compile、RZ価格取得は**全0件**。B0/B1-O/B1-OL/A1のX/Y精度・CV改善は未判定。新campaignの実現量子RZは0、launcher/preflight/test等の古典費用は生じたが完全な時間・peak計測は未取得でUNKNOWN。

自分の一時cronを、SPECの内容と完全一致することを確認して撤去した。crontabなしへ復帰し、再投入していない。
[manager directory](../../../artifacts/o2_a1_new_target/2026-11-06/n3_campaign_B_manager_run01/)にSPEC、公開照合、cron.log、STOP、全call数、trigger撤去記録を保存した。
旧397件・旧SOURCE/STOP・N1/N2/認証・main dirty/untracked・C1凍結は維持。[PRESERVATION.json](PRESERVATION.json)でmainの3,087ファイル、HEAD/status、旧科学/manager worktreeの不変を確認した。

## 技術修正と残る実行条件

失敗SOURCEは5f2eac41…として保全し、後続の技術差分でentryに自分のroot/srcの登録を追加した。実科学は再投入していない。
PYTHONPATHなし、cwd=/tmp、cron相当の最小環境で`--help`を呼ぶ人工testを追加した。[CORRECTED_TECHNICAL_TESTS.txt](CORRECTED_TECHNICAL_TESTS.txt)：**37 PASS**（新入口13、旧campaign11、管理13）。最初のSOURCE準備時の36 PASSも[TESTS.txt](TESTS.txt)に残す。人工PASSを科学的改善の証拠にしない。

元PROFILE_LOCKは失敗SOURCEの記録として変更していない。修正後の環境候補はCORRECTED_PROFILE_LOCK.jsonに別保存した。既存SPEC/SOURCEの起動不整合検査やcontrol output衝突を回避する処理は追加していない。現在の入口をそのまま再投入しても再実行可能にはしない。
次回は[有限な再投入案](NEXT_EXECUTION_PLAN.md)の明示承認、fresh manager出力先・修正SOURCE/profileの固定・公開照合が必要。今回は科学retry、seed変更、条件緩和、追加campaign、レビューを行わずSTOPする。
