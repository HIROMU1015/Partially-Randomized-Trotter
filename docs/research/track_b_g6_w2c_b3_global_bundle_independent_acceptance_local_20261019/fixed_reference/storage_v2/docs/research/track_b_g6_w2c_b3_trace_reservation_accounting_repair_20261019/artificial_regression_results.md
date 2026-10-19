# 人工検証の区分

|今回の最終実行集合|PASS|FAIL/ERROR/SKIP|
|---|---:|---|
|固定独立runnerの10 suites|623＝旧540＋独立83|0/0/0|
|保護したProducerのcodec/F01/integration/pilot|156＝30＋25＋70＋31|0/0/0|
|修正V2の新unit|41|0/0/0|
|合計・収集ID重複なし|820|0/0/0|

過去報告の481/623/696を足していない。fixed_checker_regressions、focused_tests、storage_unit_final.json/logに実収集IDとSOURCE hashを保存した。focused_tests内の新unit旧40件は開発時点であり、最終の新unitはstorage_unit_finalの41件を使う。全processで禁止科学importを検査した。

別枠の保存matrixは108ケース。正常S/P1の2件、期待されたTECHNICAL_FAILURE40件、期待されたINCOMPLETE_EVIDENCE66件。全caseで期待した状態・counter・credit・STOP/spentを確認した。108をunit PASS数へ足さず、故障runを正常成功と呼ばない。

trace open/hash/write/partial/flush/close、RESERVE/ENTER/COMPLETE/STOP/phase、result reserve/ACK/coalesced、全trial terminal共通経路、state open/truncate/pwrite前後/zero-write、journal、pair保存/flush、summary、receipt、final diagnostic、admission STOP、parent/worker喪失、call/result/log cap、drain/同時failureを検査。CANCELのstate保存故障等、確実な制御が必要な経路はunitで狙った。

途中試行はpreliminary_artifactsへ区別した。初期unitのUNKNOWN注入がCANCELへ正規化されていた点と、ACK faultが初期log ACKに当たるfixtureを修正。最初のmatrixではreceipt marker後に保存失敗が起きる12条件を固定checker共通control検査が拒否したため、markerを保存/STOP確認後へ移した。さらにadmission STOP保存4条件と、cleanup時に閉じたjournalへ再保存しない検査を加えた。

初期development matrixのその12条件は、checker前にrawを保存していなかったため原rawが残っていない。SOURCE commit・fault指定・tracebackを保持し、後から同じ原rawを再生成したとは呼ばない。以降のharnessは検査前にrawを保存し、最終108条件はすべて原rawを保持した。

canonical_compatibility.jsonでは固定旧版/S/P1の全24pairのpayload一致を確認。除外は実測generator_exclusive_nsだけであり、event/evidence/angle/bit消費/identity/chain対象データを落としていない。F01/CPU partitionは既存checkerも再実行した。実G7の費用測定や科学値評価ではない。
