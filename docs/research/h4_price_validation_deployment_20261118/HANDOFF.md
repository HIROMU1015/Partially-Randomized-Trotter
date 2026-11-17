# 次回通常入口への引渡し

通常childの固定SOURCE：`a18d5a04df5ec8c789f274730bff87a2c0a7cc89`。選択済みSERVICE：`/home/AbeHiromu/projects/rpe-local0-search/price-validation-a18d5a04df5e/SERVICE.json`。

実装の意味と検収範囲は[REPORT](REPORT.md)、配置の絶対パス・固定SOURCE・metadata結果は[DEPLOYMENT.json](DEPLOYMENT.json)を参照。

次回local0の通常起動は従来どおり`Invoke-RpeSearch.ps1`→server2の`/home/AbeHiromu/projects/rpe-local0-search/CURRENT_SERVICE.json`を使用する。既存のrun／旧SERVICE／SOURCE／STOPは上書きせず、新controllerを選ぶ。既存クライアントのthin Gatewayから新しい通常childに渡ることは人工検収済み。

`SERVICE.json`と通常／absolute-κのROUND templateには、固定worker SOURCE、PR-Pauli／PR-DFのnormal-path provider、同一queueのcompile profile、明示的なcompile runtime要件を配置する。科学モデル、Q3 computation SOURCE、固定price registryは引き継ぐ。

今回はcompile予算0・科学起動なし。次のcompileを含む明示指示を受けた担当は、対象runの`executor.distributed.preparation.budgets.compile`をその許可範囲に設定する。`price_validation.compile_approved`はfalseに固定していないため、この既存run予算から取得範囲が決まる。Q3だけの許可からcompileを推定しない。価格適用済み候補のQ3は検証の待機だけでは停止しない。

PRの元SOURCEはPauli `13ab1acfa`、DF `326feadd0`。固定view・元configuration／辞書／law／SOURCE／較正seedとの独立性の照合を保ち、代表1点の事前予測とrequestを固定してから既存workerへ渡す。配置済みprofileが通ることと、実compileを完了したことは別である。初の新規実測は今回行っていない。

fresh PFの未定義`MODULE`は修正済み。固定provider `36ad751d4`の`MODULE=scripts/prepare_pf_candidate.py`をそのまま使用し、Full PFとp7の両native childを人工検収した。PF公式・幅・科学law・価格モデルの変更はない。

remote host向けの固定profileと既存local0用配置packageは同ディレクトリのdeployment配下にある。server2は7 view配置・14件metadata PASS（保存CAS inspection override、運用profile core一致）。実queue転送と実compileは未実施。remoteはprofile/package生成のみで未配置。server2での実入力metadata検査と、remote hostに未配置／未検収の部分を`DEPLOYMENT.json`の状態どおり扱う。既存WSL参加状態、relay、worker、spoolをこの作業で起動・変更しない。

旧Gatewayのstrict HEAD再開条件は保全した。metadata commit後のHEAD更新を許す設計変更は今回含めていない。既存runの固定SOURCE不一致を無視して起動しない。

ロードマップへの差分：次回通常SOURCEに簡易価格検証を配置し、PR compile metadataとfresh PFのchild bindingを確認した。価格資格・十分N・global RMSE・Gの認定は増えていない。配置・公開後STOP。次のcompileを含む通常探索の指示までは実取得0件を維持する。
