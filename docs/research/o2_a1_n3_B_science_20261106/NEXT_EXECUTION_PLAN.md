# 科学未開始の案B：次回の限定投入案（未承認・未実行）

今回の一回投入はimport失敗で終了し、科学campaign・stream・shotは0。旧失敗SPEC/STOP/cron.logと5f2eac41…は上書きしない。
修正entryの人工テスト37件はPASSしたが、これだけで今回のSTOPを解除しない。

次の承認対象は、同じ未実行科学campaign `A1-N3-B-H4-m0-q1-20261106-run01`について、修正入口を使う**新しいmanager投入を一回**行うこと。
候補manager directoryは`artifacts/o2_a1_new_target/2026-11-06/n3_campaign_B_manager_run02`。科学outputは未作成の`n3_campaign_B_run01`、未消費master seed20261107と採択済み科学条件を維持する案。
これは旧397件や旧seed20261106の再利用ではなく、未開始のBを起動するための技術的再投入案である。科学条件・quota・control・engine・compilerは変更しない。

承認後に必要な技術差分は、(1)fresh control先の明示bind、(2)修正entry/testを含むPROFILE_LOCKの固定、(3)候補SOURCEと公開本文/blob/treeの照合、(4)manager経路での一回投入。
修正entryは自分のsrcを明示しており、PYTHONPATHやcwdに依存しない。import時の失敗はconsume receiptより前なので、その場合のat-most-once保証は成立していなかった。次回のtriggerは、この入口の最小環境testを前提に固定する。
科学開始前のimport失敗が再発した場合も、自動修正・再投入せず、triggerを撤去してSTOPする。

追加のseed選別、shot追加、再学習、精度緩和、float kernel、科学retry、追加campaignは含めない。
今回、この案のcontrol変更・lock採択・投入はしていない。別途明示承認までmandatory STOPを維持する。
