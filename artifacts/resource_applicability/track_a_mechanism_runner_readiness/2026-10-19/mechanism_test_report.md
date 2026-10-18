# Mechanism technical tests

**64人工／mock検査PASS、実科学データへの数値適用0。** 作者本人の技術検査であり独立レビュー・科学結果・immutable CIではない。

実行：python3 -I -B tests/test_mechanism_analysis.py。全数値fixtureは作成した人工値。public/source schemaやhashの照合は別のmetadata作業であり、テストへ実科学値を読み込ませていない。

|分野|確認内容|
|---|---|
|exact core|軸ごとに異なるshot/単価、巨大整数、model fraction、2つの費用分解・重み和、保存G/削減率/shot和との不一致拒否|
|pairing/cohort|cosine/sine同trajectory、seed/event、replica coverage、n8/n32/n48混入拒否、再利用n48 replica二重計数拒否|
|H4|fixed anchorの誤分類、将来D/P datasetの自動混合拒否|
|欠測|null・False・integer0の区別、missing、display-only、coreと説明の別status、非finite拒否|
|説明|B/log_B、headroom、連続shot比、保存整数の丸め整合性、LOG_DOMAIN_ONLY、overflow/underflow、保存整数を変更しないこと|
|gate/write|未承認grantをcontract読込前拒否、SOURCE不一致、既存SOURCE/STOP配下書込拒否、symlink、run再利用、実データkindの人工入口拒否|
|native/終了|人工CLIの実child正常終了・結果hash、親受入のmock、異常exitでSTOP/予約保持、post入力破損で完了拒否、入力不変|
|統計|保存SEをそのまま転記し、新SE/CI/p値等を生成しない|

人工CLIでは実OS childを起動してreturncode/waitとhashを確認。production受入lifecycleはgate/Popenをmockし、実科学用の有効grantは作っていない。人工mockの成功と科学本実行の成功を混同しない。

最初の追加mock検査3件はfixture側のinput_binding/runtime_profile field不足で失敗し、fixtureを補正して全件PASS。初回57件のPASS、失敗log、最終64件のlogを内部保全した。実値でG不一致を発見・修正した履歴ではない。

既存研究のtest campaign、quantum compile、sampling、実データによる数値再現検査は実施していない。
