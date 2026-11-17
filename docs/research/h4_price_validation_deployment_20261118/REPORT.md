# 次回local0通常探索への簡易価格検証の配置

次回local0の`CURRENT_SERVICE`は、新しい固定実装SOURCE **`a18d5a04df5ec8c789f274730bff87a2c0a7cc89`**を選ぶ。通常／absolute κの実CLI previewと、旧bootstrap経由の結果一致を確認済み。

対象は[価格検証の固定引渡し](../h4_price_validation_integration_20261117/HANDOFF.md)の次回用配置。既存runのSOURCE、入力、STOPは更新しない。今回の実取得・signal・compile・Q3は各0件。

## 差分

- `92b3e617a`を、分散preparationの固定改修`f956f5197`と統合した。検証取得は通常探索と同じpreparation queue／空きslotを使用し、保存検証の表示・再利用は追加compileなしで行う。
- 検証workerのSOURCEを`price_validation.source`で固定する。controllerが保存metadataをcommitしても、compile profileが参照するworkerのSOURCEは変わらない。
- `PRICE_MODEL_VALIDATION`用compile profileを追加した。PR-PauliとPR-DFの共通profileはQ3用の個別AST/runtime契約を引き継がず、測定付きwrapper取得用のPython/Qiskit条件を使用する。
- PR-DFの実要求metadata検査で、既ロード済みcontroller packageから固定legacy helperを読めない問題と、保存descriptorから読むscalarの扱いの差を検出した。固定SOURCEのhelper照合と、元の通常入口のdescriptor/scalar処理を再利用して整合した。controllerの既ロードmoduleと衝突するため、DF metadataと将来のDF path childは既存worker同様のlegacy-firstな分離processを使用する。元configuration・信号・科学契約は変更しない。
- fresh PFの`run_pf_preparation_stage.load_provider()`が、固定providerの`MODULE`をexportするよう修正した。Full PFとp7で旧NameErrorを人工再現し、修正後は元native childへの引数、X/Yのreceipt、SOURCE保存が通る。

## 配置・検収

**server2で14件PASS**（既存12保存入力＋PR-Pauli／PR-DFの実要求2件）。Python 3.12.3、Qiskit 1.3.0、NumPy 1.26.4、SciPy 1.14.1と、旧Pauli／DFを含む7固定viewを確認した。実要求の検査には保存CASを参照するinspection profileを使用し、運用profileの実行coreが同一であることを確認した。実queue転送・実compileの検収ではない。

確定したSOURCE、通常入口、profile、PR実入力のmetadata検査結果は`DEPLOYMENT.json`を正本とする。profile検査はPython/compilerの存在・version・SOURCE・入力依存の確認であり、回路取得の成功証拠ではない。

remote 3 hostのprofileと配置packageは生成済みだが、この作業ではremoteへの配置・接続は行っていない。server2の新worker/profileを通常のshared queueに設定済み。Q3用profile coreは旧設定とbase／3 routeとも一致する。

旧local0クライアントは旧Gatewayを読むが、`CURRENT_SERVICE`の新設定から新しい`code_root/scripts/run_rpe_local0_search.py`を起動する。人工検査でこの経路と旧STOP保全を確認した。新runの保存先は専用controller内とし、preparationのROOT-relative出力契約を満たす。

なお既存Gatewayのstrict HEAD条件は変更していない。controllerでmetadata公開commitによりHEADが進んだ後は、固定HEAD不一致による再開拒否が残る。今回の次回初回起動への配置と、この既存再開条件は区別する。旧runの条件を緩和して再開する変更はしていない。

## 実取得の境界

今回は通常探索自体を開始しない。配置したtemplateの`science_approved=false`、compile予算0を維持する。次に利用者がcompileを含む通常探索を指示した際、そのrunの既存preparation compile予算に従い、保存独立検証のないモデルだけを代表1点で検証する。PRは独立2経路×X/Y、同じモデルの別shot・κ・runからの二重投入を防ぐ既存store／共有queueを再利用する。

待機・取得失敗だけで既存適用可能価格のQ3・最適化を止めず、係数・予測・順位・価格資格は自動変更しない。詳細q/R検証は対象を指定した別指示の範囲で実施する。

## 検査記録と保全

人工・回帰103件と追加manifest回帰1件PASS。`TESTS.txt`に人工・保存metadataの検査結果を記載する。実入力検査の内訳、remote hostの未確認事項は`DEPLOYMENT.json`と[HANDOFF](HANDOFF.md)を参照。`PRESERVATION.before.json`／`PRESERVATION.after.json`は旧service・固定設定の保全を記録する。科学実行の許可証明を新設するものではない。

次回の通常SOURCEへの配置であり、新しい価格・MSE・N・Gの研究結果を追加したものではない。ロードマップ本体は変更しない。配置確認・公開後mandatory STOP。
