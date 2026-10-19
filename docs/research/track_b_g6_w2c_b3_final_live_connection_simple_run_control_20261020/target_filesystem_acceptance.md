# 対象filesystemの追加受入

最終公開照合で/tmp (device 66320)と予定science outputのproject filesystem (device 66306)が異なることを確認した。
SOURCEを変えず、予定state/outputの最近接既存ancestorと同じfilesystemへ人工専用rootを作り、既存SimpleState 16 testsだけを再実施した。
同時12process一勝者、二重Run/scope、3crash、partial intent、STOP/UNKNOWN/receipt/symlinkの16件すべてPASS。
主受入132＋対象FS追加16＝148 PASS。mutation20、integration10は主受入を維持し、sampling/Bundleは再実施しなかった。
実行SOURCEは測定a1c4f41bと同じ143 pins。SRC/tests/runnerの変更0。

初回はstdin runnerがspawn再importできず、harnessとして失敗した。失敗raw/logをtarget_filesystem_failed_stdinへ保全し、ファイルrunnerで16/16 PASSを確認した。
対象FSでのO_EXCL/SQLite transaction/fsync要求を確認した事実であり、物理power-lossや悪意ある改ざん耐性の保証ではない。
予定science directory、実permission/claim、科学loader/import/bits/samplingは作成・実行0。

[照合JSON](target_filesystem_acceptance.json)・[PASS raw](acceptance/target_filesystem/raw.tar.gz)・[最終公開追補receipt](publication_filesystem_receipt.json)。
**実行未承認、mandatory STOP。次の担当：ユーザー・GPT。**
