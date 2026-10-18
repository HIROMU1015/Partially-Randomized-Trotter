# A82説明側の限定再解析契約案（実行未承認）

状態：`REQUIRES_TECHNICAL_ADAPTER_SOURCE_AND_SEPARATE_EXECUTION_APPROVAL`。
approved=false、runtime_authorization=false、新SOURCE=null、grant未発行、launch_command=null。
今回作ったinternal new_binding_DRAFT.jsonは固定旧bindingへの別の追加案であり、旧契約の改定ではない。

将来の最小対象案はA82_OLD_PR.PRの2軸説明guardだけ。A82_OLD_PR.PFの完了済み説明を保存値として継承し、
A82一比較の新しい説明因子2軸行を別の結果履歴へ保存する。元core32、完了63 side、他31比較の因子を再計算しない。
既存SOURCE b6716396acc5760fde7cf7cc6f2481b183a08f10の数式・field型・binary64演算順序・abs1e-12 guardを維持する。
normalization、signed bias、保存allowance/headroom、alpha、log bound、保存ceilを同じ規則で照合する。

既存PRIMARY32 launcherは全64 sideを再評価するため、この最小scopeへそのまま流用できない。
新adapter・SOURCE固定・人工テスト・runtime closure固定が先に必要。数値実行を今回の回収承認から推定しない。
追加説明は新output・新grant・新ledger/markerで1回だけとし、実行前後のSOURCE/入力/runtime/元core/元説明SHAを照合する。
既存の成功・欠測・STOPは保存し、不一致・追加欠測でfail-closed STOP、自動retry/refundなし。

core/G/shot比/費用比/SEの再計算、shot置換、sampling、signal、compile、統計、ランキング、212比較、H4 S_STATS/Pは0。
新結果を保存してmandatory STOP。公開push、次stage、論文最終freezeは別判断。
次担当はユーザー：この最小追加を望むか判断する。希望する場合、Codexの技術準備後に固定SOURCEと新契約への
別のone-shot実行承認が必要である。
