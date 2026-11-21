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

## 実入力による確認

2026-11-22 03:40 JST、同じrun `search-29795d0e2e2593819e27` の通常Resumeから、Full PFの4th（Yoshida）、要求幅4h、H₄ DF10の1候補を確認した。元構成 `99e539037a07452f2f5a4029549980c82ec72d1b832cfcbcff59313ee3b0ae0e` の保存 (m=0,ldots,12) を再利用した。終端 (m=13) は同じ状態・Hamiltonian・snapshot、(q=2048)、物理ID `a660e98c78346c3fa922) の1件をRomuTailで実行し、独立した参照状態作用2回と信号1件を回収した。

中央の準備job `prep-b2e839b53009097dd0c8d577`、attempt `cd754c59626b425db5cfc65f60c15b90` に成功結果を保存した。この物理IDの予約は1件。入力取込み時にcheckoutのHEADが変わる問題は隔離検収で検出し、固定providerを保持する修正後の要求を実行した。変更前の要求に実行所有は付いていない。旧コンパイル2件は結果を回収し、既存の未確定attempt 4件の所有を保全した。

回収した信号をhash照合し、14行の実信号を使う `pf_b_pf6` で `DRY_CHECK_PASS`。配分は現runの (s=2^{-12/4}) の格子点、位相半径は元の13行が使う固定値1.0を最終行にも明示した検収設定。実入力のQ3評価と追加価格コンパイルは各0件。運用正本の検収記録は `.rpe-central/deployment/fleet-m13-f8b2788517/REAL_TERMINAL_VALIDATION.json`。終端X/Y価格の受入後にFull PFの総RZ比較へ接続する。

## B0I・PR-DFの終端入力

- 2026-11-22 03:53 JST、同じrunの通常Resumeから、B0I p7、4th（Yoshida）、要求幅4h、H₄ DF10の終端信号をRomuTailで1件実行・回収した。物理IDは `2cf98331e5eb7e8a4414`、jobは `prep-b4179ddb93a574a03312e91f`、attemptは `de7d95d25869429e95eb0bea66d8bd54`。保存13信号を再利用し、固定adapterで全14行の実入力の `DRY_CHECK_PASS` を確認した。終端X/Y価格は未取得。記録は `.rpe-central/deployment/fleet-m13-b0i-terminal-n22/REAL_TERMINAL_VALIDATION.json`。
- PR-DF LD7、2nd、κ=1、幅hの同じH₄ DF10入力を通常の `DF_SIGNALS` 準備workerで照合した。元の13ラウンドを再利用し、物理ID `18495467bbe1b54f4c9c` のみを要求できることを、26個の固定Git入力を持つ新規作業領域で確認した。終端の全H参照は同じHamiltonian・状態・snapshot・厳密時刻・結果SHAを照合し、既存providerの独立参照ゲートを維持する。異なる結果SHAでは終端だけが受け付けられなかった。
- 同じPR-DFの終端物理IDの実信号をサーバー2の公開結果で確認した。公開側と中央の構成IDは異なる。物理schedule、法則と元のrawを照合し、準備計画では全14ラウンド再利用、新規信号なしとなった。標準のQ3 adapterへ14行の受付を組み込み、公開の固定REQUESTでdry-checkと保存結果のcollectを実施した。元OUTCOMEのSHA-256 `6872f9c30b2c375f04ed344ccf29b2349beea94bb919957ea9a6fdb4f46ec4f7`、厳密MSE上下界、入力identity、PASS判定が一致した。中央の既存runでのPR-DF Q3採用と総RZ順位付けは、終端X/Y価格および通常キューへの登録後に行う。サーバー2のFull PFとB0IはS2、幅16h/7で、中央が別途取得したYoshida・幅4hとは別の物理scheduleである。 [サーバー2の実信号検収](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ac869a3dc239cf65489aacaf200cb4f113880c28/docs/research/rpe_m13_real_validation_20261122/HANDOFF.md)

## 同じrunのCPU観測

11月22日03:47 JSTの約36秒では、RomuTailの探索本体が平均0.93 CPU、価格較正が1.00 CPUを使用した。同じrunの実ジョブはサーバー1・2・WSLで観測されなかった。当時のPBSは121秒に約3 CPU秒で、予約96 CPUに対して約0.03%だった。

通常Resume後にB0IのM=11、12のQ3を計35件開始し、04:00 JSTまでに全件採用した。内訳はRomuTail 5、サーバー1 9、サーバー2 5、WSL 4、PBS 12。PBSの積算時間から求めた後半の2分では660 CPU秒、平均5.5 CPU、予約に対して約5.7%だった。次の2分は約0.03%で、待機列はいったん空になった。B0I Q3のCPU使用は既存の短いhorizonで生じた。

観測時のPF・B0Iでは価格receipt再照合が22件の保留理由となり、PR-DFには旧実行未確定と既存の価格較正・監査の失敗が残った。実行可能な配分が出た時間帯には各環境で開始できており、短いバッチの後も供給を続けられるかが残る課題である。運用上の利用率は通常runの作業時間から計測し、14ラウンド検収の実測とは区別する。
