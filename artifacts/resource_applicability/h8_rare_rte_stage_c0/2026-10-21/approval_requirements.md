# 次に必要な承認・未確定事項

C-0は設計完了・科学実行未承認。以下はユーザーが判断する事項で、経過時間やpreflight PASSで承認したことにはしない。

|判断対象|現在の案・必要な具体化|担当と次の作業|
|---|---|---|
|目的/主張|条件付き感度、モデル仮定下の統計推定、全support厳密保証のどれか。厳密保証は追加前提が必要|ユーザーが要求する保証範囲を指定|
|対象condition/層|H8 p8q2・p7q16、S2_SINGLE第一候補。multi/S4を取得/妥当bound/未知保持のどれで扱うか|ユーザーが取得scopeを選択。無条件の除外はしない|
|数値law|canonical finite lawと保存categoricalのdomain、精度/丸め許容差、全count支持の表現方法|Codexが承認範囲内で本番adapterを完成し人工検証。H8による新数値law評価は別科学承認が必要|
|標本数|目標G半幅h、信頼水準、区間方式、誤差配分、分散/load pilot本数・call予算が未定|ユーザーが精度/予算/phaseを選択。分散不明ならpilotで止め、productionは新契約/別承認|
|S0|既存P n48のSEを含める。必要精度に届かなければS0追加・目標変更・未確定STOPを選ぶ|ユーザーがS0追加可否を指定。Codexは現時点でS0精度を再計算しない|
|入力・prefix復元|保存blocks/state/preparation exportを再利用し、p7/p8の元hash一致を要求。純粋prefix復元を予算へ明記|Codexが元input/current runtime閉包をread-only照合し、新SOURCEへ接続。新Gaussian/SCF/DF/stateは禁止|
|validation call|旧4control probe・event oracleはsimulationを含む。C compile承認から自動許可しない|ユーザーが代表層/本数・simulation/oracleの正確なcall範囲を承認するか別検証方針を選ぶ|
|会計・数値監査|新標本の層別平均/SE・S0結合・区間・残余感度は新保存値解析として予算へ含める|ユーザーがこの解析範囲も明示承認。実行SOURCEと独立算術検査を固定|
|資源/同時実行|固定低worker capなし。fresh CPU/RAM/I/Oから動的admission。条件別grant/marker/STOP|ユーザーが同時実行可否と開始順を承認。Codexは自jobのみ調整|
|one-shot|各実行のSOURCE/input/runtime/seed registry/科学call数と失敗STOPを固定。旧grantは無効|Codexが完成契約を提示し、ユーザーの明示承認後にだけ新grantを発行|
|公開|新SOURCE・設計・結果候補の公開範囲。private raw/seed/path/credentialsは除外|ユーザーがGitHub公開範囲を別承認。Codexは承認までpushしない|

標本数と本番adapter/runtimeが未確定な現契約案を、そのまま有効な科学実行契約として扱わない。現在必要なのは上記目的・精度・pilot範囲等の判断であり、完成前の包括的な実行承認を求めない。C-0終了後はmandatory STOP。
