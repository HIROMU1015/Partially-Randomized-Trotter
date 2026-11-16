# 利用者入口からの4方式保存比較・再開の受入

**現行公開Skillの4方式同時比較と同じrunの再開はPASS（26/26）。** Full PF・切り捨てPF・PR-Pauli・PR-DFを同じroundへ渡し、構成・L_D・κ・shot列を入力へ埋め込まず、各方式の新規Q3予算0で実行した。期待値は実行版が実際に読む固定catalogから取得し、別版の最新最安値を混在させていない。信号・sampling・compile・fit・実Q3は全て0、job起動も0。

Cの通常定期同期もPASS。更新B2による第2検収は、現行版を先に公開した時点では対象統合commitが未公開であり保留。B2の長時間探索の完了は待たない。

## 入口と固定SOURCE

|対象|SOURCE|
|---|---|
|公開Skill|19252baf39698570bd80a98bf10142cc8f215517|
|Aの既存受入|87d00ccba。`load_skill`と`Program.run`をそのまま再利用|
|現行Skillが指定する実行版|a2e2ef50a108d6f34089a99bd00f11a7085dd9c9|
|problem|同SOURCEの`configs/rpe_method_optimizer/h4_df10_problem.json`、SHA256は期待値snapshotに保存|
|実際のcatalog INPUTS|417a73e7329fd520381dc88cc0de00a11dd0b641、`docs/research/h4_saved_rpe_cost_catalog_20261116/INPUTS.json`（固定20入力）|

この環境では公開Skillは専用toolとして登録されていないため、SkillのSKILL.md/referencesを固定Git blobから読み、指定CLIをそのまま起動した。ネイティブSkill起動を実施したとは称さない。独立したclean detached program worktreeを使い、A・B2・Dの稼働worktreeは変更していない。

保存linear H4/1Å/STO-3G/tol-only DF10、sector(2,2)、h_RPE=.1/M12、元のZ1+T1/Q3・target/law/loss・exact閾値を固定problemから継承。返す価格は元catalogの条件付きcompiled RZ費用であり、正式G・十分N・物理的RMSE・全探索winnerを認定しない。

## 現行版の結果

全26軸の整数配分・MSE区間・IDは[4方式表](current_a2e2ef50a/RESULT_TABLE.md)、exact値は[元catalogからの期待値](current_a2e2ef50a/EXPECTED_FROM_ACTUAL_CATALOG.json)と[受入結果](current_a2e2ef50a/ACCEPTANCE_RESULT.json)を参照。

|方式|保存集合内の最安構成|条件付きRZ費用 exact|判定|再開|
|---|---|---:|---|---|
|Full PF|S2/16h/7、11cf8ca6…|559457386|保存Q3適合|同一結果・停止理由|
|切り捨てPF|p7/Yoshida4/32h/5、dd8bfc24…|372035410|保存Q3適合|同上|
|PR-Pauli|L_D5、4d6e2c9c…|3792207339/7|保存Q3適合|同上|
|PR-DF|L_D7、ce9c7f08…|155652932221/126|保存Q3適合|同上|

全方式の停止理由は`NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET`。次候補は返るが投入しない。現在の別catalogにあるFull PFの480,258,064RZ等を、この旧実行版が読んだ期待値へ差し替えていない。

価格待ちPR-DF候補、NEEDSに載る信号/価格未完了候補は最安点から除外され、他方式は正常に返った。未評価の次配分も保存適合点へ混入していない。

実CLI起動は初回と同一コマンド再開の2回、終了codeは両方0、wallは7.82/7.76s。INVOCATIONSは1→2、方式別launched/new_callsは両方全0、jobs/LAUNCH/OUTCOMEは0。bestと停止理由は各方式で再現し、完了済みjobの再起動も0。program checkoutは実行前後ともclean。新規Q3予算0だけを根拠にせず、この版のprepareが保存metadataのみであるSOURCEと、実起動/出力を合わせて確認した。

4方式差分の判定関数に対する人工6 testもPASS。誤った構成・配分・判定・費用、価格待ち混入、予算外起動、再開時の変化、MSE区間不一致を検出する。新しい科学条件・新規モデルfitは含まない。

## Cの実際の定期同期

C公開 `8b22ca2341f7bdf421fba7a3c018a4ff6d9b281f` の通常catalog `candidates[57]` に、D352539のFull PF/S2/2hが追加された。信号13 round、価格26軸（直接10・条件付き16）、Q3_WAITING、配分/N/MSE/Q3判定/達成費用nullを確認した。

通常cronの13:20:01〜13:20:24 UTC周期が終了0で反映した。手動再同期・scheduler編集は0。PUBLICATIONSでD registryはINTEGRATED。既存p7の二つの評価行は元固定catalogとbyte相当の内容一致で、点の重複なし。registry再掲はsignal_only_historyに保持される。後続C `c7ebeadb…` はこのcatalogを変更していない。[判定](C_PERIODIC_SYNC_ACCEPTANCE.json)・[固定selector/周期log locator](C_PERIODIC_SYNC_LOCATORS.json)を参照。

## 更新版の検収状態

開始時にremoteで確認したB2は`e3b4c1114139258a2c6e5d3e14de8b73fc4f7cda`。D2準備と新κ選択を統合したlocal `ba35de77…` は未公開だったため、固定公開版として使用していない。dirtyファイルや進行中runも取り込まない。

統合commitが公開されたら専用program checkoutと別runを作り、同じ4方式・予算0・initial=allで検収する。catalogのSOURCE差を別記する。不足候補→D2は保存入力又は既存人工fixtureで検査し、実D2処理は起動しない。現行版のPASSを先に公開し、更新版の完了を待って隠さない。

本検査は利用者入口に対する受入であり、Aの価格モデル検証やB2の更新実装、実科学計算のレビューを代替しない。科学raw・旧STOP・他担当runは保全した。ロードマップには「現行4方式の保存比較/同run再開とC実定期同期の受入完了、更新版は未検収」を差分として渡す。
