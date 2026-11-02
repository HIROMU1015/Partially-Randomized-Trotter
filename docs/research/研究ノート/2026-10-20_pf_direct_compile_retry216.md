# 2026-10-20 PF直接compile修正・216軸再実行

利用者が「その方針で進めて、再実行まで行って」と承認。H2/H4 shape処理とH8既知scalar入れ子を修正し、分離observerとnative終了監査を接続する。新13人工＋既存境界13検査PASS。旧572 task受入と旧STOP/欠測を保持し、残り216軸だけ新SOURCEで1回実行する。現在はseal前。新公開は未承認。詳細は[報告](../track_a_pf_direct_compile_retry216_20261020.md)。

## 追記：実行起動とtask単位STOP

SOURCE 0c5fcc83ef55045c2c64746aadfaf3128667b058をsealし1回起動。observer199948/PPID1、coordinator200066を分離確認。2026-10-19T20:18:27.474399+00:00のnative確認は成功86/STOP106/未終了24軸。全STOPはcompile前CONTROL_MISMATCH。B0I preparation選択の静的不一致を発見したが、稼働SOURCEは変更せずretry0。残り正常worker継続。終了後の結果監査・commitはCodexが行う。現状は計算待ちであり全体mandatory STOP完了ではない。

## 終了追記：2026-10-20 10:43 JST確認

07:03 JSTに終了していた。今回110成功/106 control STOP/未投入0/unknown0。H8残22は全成功。全108worker回収、coordinator exit0、metadata auditorはpartialを保存して設計どおりexit2、observer STOP。実時間6439.953秒、worker CPU52330.987秒、最大単一RSS6433300480 bytes、標本合計RSS最大41486770176 bytes。38,686件の最終gate PASS。旧572と合計682軸を保存、元global final gate欠測は保持。SOURCE/旧raw/grant/ledger/STOP変更・追加科学実行0。結果転記・構造検証を保存し、commit後mandatory STOP。次担当ユーザー：残106軸への追加対応と公開判断。
