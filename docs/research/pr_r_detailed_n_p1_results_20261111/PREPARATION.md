# 詳細N P1の限定実行SOURCE（2026-11-11 JST）

固定計画 `d33380a8c2fe68d0be1f2c5faaf7fc3b3a956b58:docs/research/pr_r_detailed_n_preparation_20261111/EXECUTION_PLAN.json` のP1_SHORT_MIDDLEだけを実行する。計画原本の未承認/未実行statusは過去の準備状態として変更しない。今回の利用者チャットでP1実行が許可されている。許可証明書・追加grantは作らない。

H4 m0 R2/4/8、m6 R4/128/256、H6 m4 R11/16/32、m8 R16/128/256の全12 ID。保存baseline scalar3→H6 m4 baseline新mean1→新R8の順。新finite mean/内部対照各9、scalar12/24軸、新referenceはH6 T1.6のexpm_multiply/expm各1だけ。その他3時点は保存pairをbinding付きで再利用。旧N/保存signalを変更せず、全候補は同じR_AWARE_EMPIRICAL_ACTION_COUNT_V1で評価する。

B `ac3e1ce87b5ed7d9114abc316545366dd097209e` のRMenuBatch/_law/bound_context_factoryのAST本体を変更せず再利用し、価格/optimizerをloadしない。A RBatch・finite action・normalizer・scalar geometry・整数ceil検査は不変。既存の適応worker配分、プロセス内BLAS1、独立systemの並列化を用いる。power backend/cache変更は導入しない。

6人工/mockテストPASS。real科学は実行SOURCEをcommitし、runnerの--executeを明示したときだけ開始する。fresh directory以外を拒否し、結果・partial/STOPを保存、同じrunへの自動retry/resumeはしない。P2/P3、sampling、compile、価格、G、R最適化、全RPE simulation、詳細科学レビューへ進まない。
