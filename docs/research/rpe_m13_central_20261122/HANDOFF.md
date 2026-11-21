# H₄の14ラウンドQ3を中央探索へ接続

- 旧runの固定科学入力は `28d976b99cdf8329adcc1b79327e329e947ef774`。通常Q3のリスク計算は `cbc471c913072de45c25a0ed22d51dff8a9b7034` の同一関数を基点とし、14ラウンド用の変更を `89fcde14f6ac5f4b97be234f9b0822091e7d9ac7` に固定した。
- 中央スケジューラとadapterの固定版は `f8b27885172c3f58dc3976f8d4e93d4d2431ade9`。評価器の参照値と人工Start／Resumeは `0100c216ac6136cace7e9a8d297617de3935921c` の検証条件を引き継いだ。
- Full PFとB0Iの末尾信号は既存の `PF_SIGNALS` で1ラウンドずつ扱う。端末固有の入力パスの解決、bundleの新規checkout検査、受取時のhash照合、予算と所有は既存workerと中央キューが処理する。設定は `executor.distributed.preparation.horizon_extensions` を使う。運用中のrunの継続設定に渡す場合は、同じカタログを `worker_trial.horizon_preparation_catalog` のパスとSHA-256で固定する。
- 信号入力の `configuration` と `signal_packet` は元の `M=12` の固定Git locator。`scripts/run_rpe_pf_horizon_signal.py --mode plan` は13行の物理回路・測定法則・信号の同一性と新しい終端の物理IDを確定する。既存の準備workerへ渡す実行要求は `physical_id` と固定snapshotを含む。終端の独立参照は同じ保存H₄状態の固定2経路、信号は既存のPF数値kernelで得る。
- 終端要求を含む入力bundleはGitオブジェクトだけを取り込み、実行checkoutのHEADを固定providerのまま保つ。RomuTailの固定Pythonで、作業checkout作成からFull PFの実保存入力13行を読み込む `plan` まで通過した。既登録の延長要求は短いhorizonの実行開始時に同じキューへ送り、その完了を待たない。
- Q3用の固定routeに `horizon_risk_source` と `horizon_parent` を指定する。前者は上記科学viewのcommit・ファイル・SHA-256、後者は保存した `configuration` と `signal_packet` のlocator。worker profileでは科学viewを `route_views` から解決し、必要に応じて `route_profiles` を `horizon_risk_commit` で指定する。旧routeには14ラウンド用のフィールドを付けない。
- adapterは全14行と、固定した14個の位相半径を照合する。旧13行に対して、新旧の構成IDに伴うラベルの変更だけを解決した上で、物理schedule、測定法則、raw・corrected・reference・正規化が一致することを検査する。大ショットの既存2,048 shotと資源上限を適用する。
- 総RZ比較へ戻すFull PF・B0Iは、既存の価格・信号registryに新しい構成ID、元構成ID `horizon_source_configuration_id`、固定route、全14ラウンドの28軸分の価格証跡を登録する。片軸欠けの段階では終端価格不足を表示する。PR-DFの新しい終端信号には同じ物理条件の固定参照をPR-DF providerへつなぐ必要がある。

人工キューでは (M=12,13) の通常StartとResumeを検査し、同一job ID・attempt IDと結果のバイト列が維持され、追加起動が0件だった。固定viewを使った通常Q3は全列挙と厳密一致し、大ショット精密化の区間も参照MSEを含んだ。PF/B0Iの実adapter入口には保存13行と人工の終端信号を渡し、14行のdry-checkが通過した。

選択した回帰試験は92件と16サブテストが通過した。旧最適化関数とのAST比較、旧PF template所有照合、人工ホストのsupervisor Python期待値に関する別の失敗は、変更前の `f8e67d806fb845f5985a9c39638f36cae0b74029` でも再現した。
