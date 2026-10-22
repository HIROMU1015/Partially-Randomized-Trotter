# H6 metadata追加承認と取得STOP

利用者が63候補の再観測と固定H6後続stageを包括承認。原B0I serializerの不足値を推測せず新観測へ固定した。事前人工14PASS、準備2回のgrant前例外を修正してSOURCE3cff9a4baで一度起動。snapshot_restore1後のscalar shape比較不具合でSTOP、signal/metadata0。入力破損ではない。修正458ea00dd・人工5PASSを保存し、ユーザー指定の失敗時STOPに従って科学再実行はしない。旧H2/H4結果・旧H6 STOPを保全。H6共通域比較は未完了のまま。[結果](../track_a_h6_pf_metadata_STOP_20261022.md)。local commit後mandatory STOP、push0。

## 追記：利用者の「修正を行って」に対応

scalar shape修正にgrant前のstdlib NPY header/dtype/SHA検査を追加し、loader本体の読込経路とH2/H4既存規約を人工試験した。人工26件・保存H6の8配列byte照合PASS。旧STOPを保全して科学再実行0。[修正報告](../track_a_h6_snapshot_loader_repair_20261022.md)。local commitのみ、push0。

## 追記：metadata取得成功とCOSTの外部中断

新SOURCE/新grantで不足63観測（旧52値保持・新11 signal）と新102 signal、1380 endpointは正常終了。COSTは187完成pairを残してprocess不在。元の失敗時刻・原因・exit/reapは不明。1122 archiveを保存値監査し、394 compile予約を含む全消費を非返却計上。残22を元手法でfresh取得するとnative piece累積4854が元上限3625を超えるため、再開せずmandatory STOP。旧grant/ledger/marker/STOPと旧H2/H4結果を保全。自分専用一時serviceによる起動分離を人工3件＋3秒実動作で検証したが、追加実分子stageは0。[監査・予算差分](../track_a_h6_pf_cost_interruption_20261022.md)。push0。

## 追記：中断に備えた監督処理修正・実行無効仕様

今回の「修正を行って」は制御修正として実施。新SOURCE 96c1c60cfaf35b9710d90d1ef6f44280b2d73267でspawn失敗/非0終了/TERM/INTのSTOP、子回収、terminal heartbeatとPID start time検査を追加。新10＋既存3人工PASS、自分の一時user serviceで起動元終了後の3秒人工タスク完了を確認。残22の入力/旧control等6504照合項目を未承認仕様へ封印。予算追加は未承認、science/new grant/push0。旧STOPと完成187pairは不変、H6全域比較はUNRESOLVED。[修正と限界](../track_a_h6_pf_recovery_control_repair_20261022.md)。ローカルcommit後mandatory STOP。

## 追記：利用者の再開指示を新grantへ結合

「計算を再開して」を直前のfresh22内部予算案への承認として、新SOURCE bdf6772aade7063c65d69f24dfdc5e303a2819a2・新pack・新grantで残22を一回起動。旧187価格の保存値受入と旧SIGNAL/QUALIFY成功を参照し、旧STOP・消費は不変。22 worker、人工新5＋export1 PASS。正常終了後に別ACCOUNT grantで固定会計、監査、588候補参照統合、local commit/mandatory STOP。開始時点で完了・順位は未確定。push0。[起動記録](../track_a_h6_pf_recovery_execution_start_20261022.md)。
