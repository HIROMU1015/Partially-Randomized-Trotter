# E/T saved recovery v2 startup repair

前版はmarker前にmissing parent directoryで停止、科学処理0。旧source/approval/outer failure保存。
mkdir(parents=True)へ修正し、fake coordinatorでmissing-parent regressionを追加。
31 focused PASS、test layout整理後にpending/startupの2checks PASS。会計ASTとstorageは不変。
新preparation/result namespace v2でsource固定し、利用者の継続認可に基づくrecovery-only直接子で実行する。
元v7のnative22/T39に対するsource-identical metadata再集計だけ。new science/native/T/matrix0。
元one-shot分類/markerを維持、復元完了後mandatory STOP。
