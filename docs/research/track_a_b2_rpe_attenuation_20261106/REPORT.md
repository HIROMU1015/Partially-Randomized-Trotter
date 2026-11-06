# Track A B2：固定RPE長roundのnormalization減衰スクリーニング

2026-11-06 JST / SERVER 2（GPUA100-gleap02）/ Codex主エージェント。

**承認された固定スクリーニングを完了した。既存horizon関数からm=0..17の18round、Q_max=131072を算出した。最終roundのnormalizationは86.41007760254907、attenuationは0.0115727242440354。全roundでbinary64 underflow・subnormal・normalization overflowはなかった。** このAはnormalizationの乗数であり、元H stateの実信号半径や総費用改善の判定ではない。**mandatory STOP。**

## 1. 固定SOURCE・入力・実行範囲

| 項目 | 使用したもの |
|---|---|
| library SOURCE | [`66a41ad9219f12377e7d325e272c1a90d66ab7e5`][CONNECTION]。接続実装の既存219人工検査PASSを参照し、接続を再実装・再試験していない |
| DF12入力 | commit `bf5d3b2405b2bec4f88dd0589b75dd737b13e052`、[`artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz`][INPUT] |
| input SHA256 | `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。今回Git objectから読み、file bytesを照合 |
| DF H hash | 保存metadataの`de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。元Hとの数値比較は行っていない |
| 固定条件 | linear H4、1.00 Å、STO-3G、保存DF12、generation-prefix L_D3、partial-S2、r4/K2、δ=.02。決定論fragments0/1/2、random fragments3..11。全signed weightsと保存factor順序を保持 |
| screening条件 | ε_E=.00015936001019904 Ha、β_RPE=.4 rad、成功確率候補95%、Q_m=2^m。ε_E/β/δはhorizon用。95%は記録のみでA計算に不使用。energy保証の採択ではない |
| 実行環境 | Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、OpenFermion1.6.1、OpenFermion-PySCF0.5。環境とsource/runner SHAは[result.json][RESULT]に保存 |

元H/stateの仕様は[前回の対応確認][STATE]、既存資産の整理は[接続可能性調査][PRIOR]を継承する。LOCALで進行する対応検証へアクセス・介入しておらず、その未決定事項を変更していない。旧T=.8の結果も新roundの結果として流用していない。

## 2. 回路を作らず、既存係数・finite lawを再利用

既存`prepare_b2`→`_prepare`はGaussian回路構築を含むため、今回は呼ばなかった。[`build_df_blocks`][OPS]が使う`diag_hermitian(sort='descending_abs',assume_hermitian=True)`と[`exact_df_diagonal_coefficients`][TAIL]だけを直接使い、保存された各tail Gの固有値から同じI/Z/ZZ係数を取り出した。これは固定Gの係数抽出で、新しいDF分解・fit・state solveではない。

既存`extract_df_diagonal_tail`と同じ、coefficient_atol=0、exact identity抽出、fragment ID/support順、`math.fsum(abs(coefficient))`を使った。**非identity成分324、λ_R=0.5832704246339983、抽出tail identity係数=0.09261516589219072、確率和=1.0**。λ_Rは[既存V1–V3の保存値][V123]とも整合する。signed coefficients、fragment別固有値、identity内訳をresult JSONへ保存した。one-body・核scalarは変更・評価せず、identityは決定論側へ移す既存規約を維持した。これらのscalarは今回のλ_Rに加えていない。

その係数だけの非実行用projectionから、既存[`make_rte_config`/`finite_rte_distribution`/`finite_rte_attenuation`][RTE]を使用した。Gaussian basisやoperatorを生成せず、空のbasisを実回路として偽装していない。projection hashは実行可能なB2 tail/preparation hashではない。

K2のpaired finite lawはorder0/2で、補正平均の有限多項式はdegree3：

\[
P_3(-i\tau h)=I-i\tau h-\tfrac{\tau^2}{2}h^2+\tfrac{i\tau^3}{6}h^3,
\quad h=H_R/\lambda_R.
\]

この式を記録しただけで、operator生成・作用・信号simulationは行っていない。SOURCEのnormalizationは

\[
b_2(\tau)=\sqrt{1+\tau^2}+\frac{\tau^2}{2}\sqrt{1+(\tau/3)^2},
\quad \tau=\frac{\lambda_R\delta}{4}=0.0029163521231699912.
\]

1 event（1 RTE short step）の **b=1.0000085051026735**。order0/2の確率はそれぞれ`0.9999957474793058`、`4.252520694391542e-06`。抽出後のλ_Rと固定r/K/δから今回計算したもので、M1のL_D0を転用していない。constructor内部の固定K tolerance helperはそのKを受理する整合用途だけに使い、RTE位相誤差予算として採用していない。

## 3. 必要round範囲と全round結果

既存[`required_rpe_round_horizon`][HORIZON]を、指定ε_E/β/δで呼んだ。出力はM=17、Q_max=131072、nominal分解能`.000152587890625 Ha`、一つ前のQのnominal分解能`.00030517578125 Ha`。SOURCEの最小power-of-two判定を通過した。旧18roundをコピーした結果ではなく、今回の算出結果である。これはalias・共通位相・誤差条件を満たしたenergy保証ではない。

全roundでr4/K2、λ_R/τ/bは共通。N_event=Q_m r、B_m=b^N_event、log A_m=−N_event log bとして保存した。表示は丸め値で、完全なbinary64値・80桁scalar照合値は[JSON][RESULT]と[CSV][CSV]にある。

| m | Q_m | T_m=Q_m δ | N_event | B_m | A_m | log A_m（自然対数） |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | .02 | 4 | 1.00003402084 | .999965980313 | −.0000340202660212 |
| 1 | 2 | .04 | 8 | 1.00006804285 | .999931961783 | −.0000680405320423 |
| 2 | 4 | .08 | 16 | 1.00013609032 | .999863928195 | −.000136081064085 |
| 3 | 8 | .16 | 32 | 1.00027219917 | .999727874905 | −.000272162128169 |
| 4 | 16 | .32 | 64 | 1.00054447243 | .999455823861 | −.000544324256339 |
| 5 | 32 | .64 | 128 | 1.00108924131 | .998911943850 | −.00108864851268 |
| 6 | 64 | 1.28 | 256 | 1.00217966906 | .997825071566 | −.00217729702535 |
| 7 | 128 | 2.56 | 512 | 1.00436408907 | .995654873447 | −.00435459405071 |
| 8 | 256 | 5.12 | 1024 | 1.00874722342 | .991328627018 | −.00870918810142 |
| 9 | 512 | 10.24 | 2048 | 1.01757096076 | .982732446745 | −.0174183762028 |
| 10 | 1024 | 20.48 | 4096 | 1.03545066017 | .965763061886 | −.0348367524057 |
| 11 | 2048 | 40.96 | 8192 | 1.07215806965 | .932698291704 | −.0696735048114 |
| 12 | 4096 | 81.92 | 16384 | 1.14952292632 | .869926103347 | −.139347009623 |
| 13 | 8192 | 163.84 | 32768 | 1.32140295813 | .756771425284 | −.278694019245 |
| 14 | 16384 | 327.68 | 65536 | 1.74610577776 | .572702990127 | −.557388038491 |
| 15 | 32768 | 655.36 | 131072 | 3.04888538712 | .327988714900 | −1.11477607698 |
| 16 | 65536 | 1310.72 | 262144 | 9.29570210380 | .107576597102 | −2.22955215396 |
| 17 | 131072 | 2621.44 | 524288 | 86.4100776025 | .0115727242440 | −4.45910430793 |

1 eventのbは1に近くても、Qの増加で同じnormalizationが多数回積み重なる。固定r/K/δではτとbが変わらず、−log AはQに比例し、各roundで倍になる。λ_Rはτを通じてbを決め、rはτとevent数の双方、Kはpaired lawの打切り範囲を決める。今回はそれらを変更・比較していない。長roundでnormalizationによる乗数が小さくなったことは確認できるが、**Aから実信号半径、必要shot、単回価格、総費用、PF/RTE適格性や95%成功を推定していない。**

## 4. 数値精度とunderflow

主出力は既存関数のbinary64 bと`exp(−Q r log b)`である。積の逐次乗算を避け、全roundにlog Aとlog Bも保存した。binary64で表示不能になった場合のflagと、正のscalarを対数で保持する出力経路も用意したが、今回18roundには該当しなかった。

標準`decimal`の80桁で、同じbinary64 τを入力した上のb_2閉形式・log・expを照合した。最終roundのAはこの照合で`0.0115727242432468234…`、主出力との相対差は最大`6.814102871926307e-11`。binary64で丸めたb自体を固定して80桁log/expで照合すると相対差は最大`1.9016012947367183e-16`で、主な差は1 eventのbの丸めが長い指数へ伝播したものである。

80桁照合はscalar normalizationだけで、DF固有値・λ_Rを80桁で解き直していない。区間演算・厳密誤差certificateでもない。全roundでAは正のnormal binary64値であり、表示の0を物理的な減衰ゼロと混同していない。

## 5. 既存B2 RPE接続へ戻す入力と未採択事項

| 次の接続点 | 今回の結果を使う方法・残る条件 |
|---|---|
| B2 roundのlaw | `B2RPERound.config/distribution`を正式なprefix preparationから作り、同じDF入力・L_D/r/K/δ/λ_R・bを確認する。今回の係数projection hashを実際のtail hashとして挿入しない |
| 元H/stateの信号 | LOCALの同一入力対応確認後、元H stateを指定してoriginal/DF/ideal-tail PF/finite corrected/rawの各round信号を検証する。DF stateやPF有効固有状態へfallbackしない |
| 実半径とnormalization | `B2SignalRecord.bounds`は既存Aと`raw=A*corrected`を照合する。PF半径下限とcorrected誤差diskは未評価。実半径をAだけと置かず、raw半径へ二度Aを掛けない |
| phase・失敗配分 | 元H energy区間、alias/branch、model/PF/RTE/stat/implementation/reserve予算、各軸αと全roundの和は未採択。95%候補を各round・各軸へそのまま複製しない |
| shot・価格・総和 | それらの条件が確定し別承認された場合だけ、既存`account_signal_round`/M1会計とAX native全wrapper・強baselineを使う。今回はこれらを呼ばず、N・価格・総費用はnull相当の未評価のまま |

最小の次検証案は、元H/state bindingと誤差・alias条件の採択後、**まず同じ固定B2の最長roundと短round anchorの実信号・半径・誤差を検査する**こと。事前予算に不適合ならSTOPし、適合する場合に残る全required roundの資格を確認する。最長roundでAが小さいためここを先に見る意味はあるが、A単独でGO/NO-GOを決める閾値は新設しない。

**現在のGOは固定normalizationスクリーニングの完了だけ。** 元Hのenergy成功保証・全RPE総費用評価へのGOではない。次段階では未結合state、非正の半径下限、未定義位相、予算不適合、非有限量、回路意味不一致でSTOP。r/K、δ、prefixの再探索や旧価格での補完は別承認なしに行わない。既存219人工検査の接続は維持し、作り直さない。

## 6. 再利用・実行・公開・STOP

新規コードは[固定screen用の薄いrunner][RUNNER]だけ。既存prepareがGaussian回路を作ることを避け、既存の係数抽出・law・horizon・attenuation関数を呼ぶために必要だった。SOURCE/libraryは変更せず、runner本体のhash、照合した7 SOURCEのhash、input identityをresultへ保存した。実行後にもSOURCE bytesとrunner hashの不変、JSON/CSVの全18行・全field一致を確認した。既存219人工testを再実行したとの主張はしない。

既存専用venvと接続実装用`.venv/extra`を読取利用し、`-B`/`PYTHONDONTWRITEBYTECODE=1`、専用MPLCONFIGDIRを使用。新package導入・共有環境変更なし。利用可能CPU64、科学kernel開始前のMemAvailableは`401349748 kB`。計算は9個の8×8 Gの係数抽出と小さいscalar表なのでworker poolを作る必要はなく、CPU/RAM/workerの人工上限も設けていない。GPU query・他ユーザーprocess操作はない。

実行した新科学計算は、この固定条件の係数・normalization/attenuation・horizonと同じscalarの精度照合のみ。保存stateのdecode、元H/DF生成・fit、state solve、状態時間発展、PF/RTE信号、回路build/compile、sampling、shot・価格・総費用、energy成功認定、GPT詳細科学レビューは行っていない。旧SOURCE/input/raw/result/grant/marker/STOP、M1終了提案、M2保留を保持した。

専用branch `codex/track-a-b2-rpe-attenuation-20261106`へrunner・JSON・CSV・このREPORTの4ファイルをGitHub連携でcommit公開し、remote SHA・tree・4ファイルの本文/bytesとローカル一致を確認する。共有Git認証設定は変更しない。

**mandatory STOP：固定スクリーニング完了・公開照合後に停止する。** 次の担当は**ユーザー**：LOCALの対応検証結果と本診断を踏まえ、必要なら固定B2の信号・半径検証の条件と実行範囲を別途承認する。**Codex**はその明示指示まで追加科学作業を開始しない。r/K再最適化・shot/価格/総費用・詳細レビューへの自動進行はない。

[RESULT]: ../../../artifacts/track_a_b2_rpe_attenuation_20261106/run01/result.json
[CSV]: ../../../artifacts/track_a_b2_rpe_attenuation_20261106/run01/rounds.csv
[RUNNER]: ../../../scripts/run_b2_rpe_attenuation_screen.py
[CONNECTION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/docs/research/track_a_b2_rpe_implementation_20261106/REPORT.md
[INPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf5d3b2405b2bec4f88dd0589b75dd737b13e052/artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz
[STATE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2796d26270cf39e46b40accc4f898fe29d459819/docs/research/track_a_b2_original_state_spec_20261106/REPORT.md
[PRIOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e17f4d39ec3733c5e1f58eda6871d69b06a651c/docs/research/track_a_b2_total_rpe_connection_20261106/REPORT.md
[OPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trotterlib/df_trotter/ops.py
[TAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trotterlib/df_rte_tail.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trotterlib/rte.py
[HORIZON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trotterlib/rpe_target_round_horizon_validation.py
[V123]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/artifacts/pr2_new_series_validation/2026-09-28/pr2_v1_v3_result_v1.json
