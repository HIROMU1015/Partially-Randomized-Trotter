# C-1承認前の選択と固定項目

C-0.5は実装・toy検証まで。現契約はPROPOSED/approved=false/runtime_authorization=false。以下を勝手に最終採択しない。

|判断|具体案/必要情報|担当|
|---|---|---|
|数値モデル|保存binary64の厳密有理数条件付きモデル＋PCG64 raw integer categoricalを採用するか。canonical B・旧NumPy choiceとの差、1e-12 engineering gateの限定を確認|ユーザーが採否。実H8差分/count計算は科学承認に含める|
|pilot scope|A：両条件singleだけ、B：両条件の3希少層。条件片方だけの選択も可能。未取得層は未知保持|ユーザーがcondition/層/開始順・同時可否を選択|
|本数|相対分散精度η=1/2または1/4、計画信頼90%または95%の案。Normal仮定のchi-square幅基準などでnを決める。n>1だけでは精度十分でない|ユーザーが計画仮定・基準・予算を選び、Codexが完成整数予算を提示|
|validation|選択層ごと1代表、4control probe（4simulation）＋1event oracle＋2非測定wrapperという元検査案。省略/変更も科学判断|ユーザーが対象task・call数・simulation実行を明示承認|
|pilot解析|paired層別mean/variance/SE・独立moments、law診断。S0との新期待費用結合・区間は別解析scope|ユーザーが数値会計範囲を明示承認|
|production|候補95/99%、半幅は保存点PF/PR差の1/4または1/2。S0不確かさ、残余、pilot分散から別cohortのNを決める|ユーザーが後で判断。C-1 pilotから自動進行しない|
|資源bootstrap|初回登録taskを受入できるprivate peak/thread/IO需要の根拠。cold-start前に未測定値を小値へ補完しない|Codexが既存receipt/承認条件から提示し、ユーザーが契約範囲を確認。共有設定は変更しない|
|実行binding|C-0.5 SOURCEを基にhelper/module paths・runtime/library・input/prep・旧seed除外coverage・新registry・fork安全性をfresh照合|Codexがread-onlyと人工検証で封印。unsafe forkならSTOPし代案を提示|
|grant|条件ごと別の承認receipt/grant/非返還ledger/O_EXCL marker/STOP。旧grantの再利用なし|完成契約にユーザーが明示承認した後だけCodexが新grantを作成|
|公開|新code/設計/監査の公開範囲。private原本は除外|ユーザーが別承認。Codexは未承認pushをしない|

現時点では完成前の包括的な科学実行許可を求めない。まず上記の選択を行い、具体的なn/整数call予算/manifest/SOURCEをreview可能にしてから、C-1 one-shot承認を判断する。mandatory STOP。
