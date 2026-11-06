# Track A・B2 P1：固定強PF 8 armの信号適格性（2026-11-07）

固定8 armを実行し、**Full S2のsubsteps 1・2、Yoshida4の1が全18round条件付きPASS**。
残る5 armは独立数値経路の複素信号差・半径差が固定閾値1e-9を超え、最初の不適格roundでSTOP。
PF位相cap超過による失格ではない。閾値変更、再計算、位相fit、spin projectionによる救済は行っていない。
固定集合のスクリーニングを終了し、**mandatory STOP**。shot・branch・価格・総費用へ進んでいない。

## 固定SOURCE・入力・条件

- 実行SOURCE：[`a7940b58c6cb939d77e9f8e44d214b3787b7e101`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a7940b58c6cb939d77e9f8e44d214b3787b7e101)。人工18件PASS後にcommit固定し、tree/作業bytes一致を確認して1回実行した。
- [結果JSON](../../../artifacts/track_a_b2_p1_strong_pf/2026-11-07/run01/result.json)、[全評価round CSV](../../../artifacts/track_a_b2_p1_strong_pf/2026-11-07/run01/rounds.csv)。全140点（失格点を含む）。不適格後の未評価4点は欠測をPASSで埋めていない。
- B2参照：全18round `d7a55ac181de71cd2b2c49644810373cafd0a450`、端点 `ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5`。保存値と原結果・SOURCE・P1条件を既存`validate_saved_bundle`で照合。元H/DF信号を再simulationせずそのまま使用した。
- B2 branch `d424bb8bec1aaf1224a33513a06e64801749a69f`、shot `0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`、P1 `7bc952772e451f27c1654dab6803192f2b204c9f`、既存強PF接続 `66a41ad9219f12377e7d325e272c1a90d66ab7e5`。新しいbranch/shot評価ではない。
- Targetはcutoff前H_phys、linear H4、1.00 Å、STO-3G。保存DF rank12、Nα=Nβ=2、sector36。元H保存基底stateを使用し、厳密stateを研究上仮定、準備費用を除外。DF/PF固有stateと仮定しない。
- H_phys S0 hash：`bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2`。
- 元state S0 hash：`1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db`。private NPZ SHA256：`deb561367e216fd4c62d013c10a4adbbef5da8c3e8139c39170d71f80530a80c`。
- DF NPZ：`bf5d3b2405b2bec4f88dd0589b75dd737b13e052:artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz`、SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。Hamiltonian hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。
- 元operator/state/DF/sector/bit-orderは既存`load_private_inputs`で固定hashと照合。DF snapshotのstateは廃棄。spinは2p=α,2p+1=β、Qiskit index=Σn_q2^q。private HDF5/provenance原本・積分・MO・state配列は公開していない。

P1はε_E=0.00015936001019904 Ha、成功目標95%、β=.4 rad、δ=.02 Ha^-1、m=0..17、Q=2^m、T=Qδ、事前区間[-128,128] Ha、shift=0、U=exp(-itH)、φ=-δE。
予算はmodel/PF/RTE/statistical/implementation/reserve = .02/.06/.04/.24/.02/.02 rad、α_X=α_Y=.05/36。
u_z/u_radius/u_normalization=1e-8、u_phase=1e-8 rad、独立不一致≤1e-9を維持。
**決定論RTEは0、未使用.04 radは予約のまま再配分しない。**
JSONの`P1_conditions`内L_D=3,r=4,K=2は比較元B2の条件であり、強PFの分割条件ではない。
強PFはone-bodyと全12 signed factorを使用し、各armのformula/substepsを全roundで固定する。

## 既存実装と最小差分

- [`b2_rpe_connection.strong_baseline_schedule`](../../../src/trotterlib/b2_rpe_connection.py)：既存`build_strong_baseline`のschedule生成部分をpure helperへ切り出し、元の回路接続も同じhelperを使用。既存Full S2 `ordered_schedule`と`_get_w_list('4th')`による正・負・正Yoshida stage順序、Fraction時間を維持。
- [`b2_p1_strong_pf_validation.py`](../../../src/trotterlib/b2_p1_strong_pf_validation.py)：`prepare_xdf_basis(...,SPIN)`、`native.runs/combined_diagonal`を直接再利用。`StrongBuilder.sequence(relative_fusion=True)`の基底遷移Γ(V_next†V_prev)、diagonal primitive、終端基底を既存CAR helperの`fock_lift`で数値評価する薄いadapter。Gaussian emitter・StrongBuilderの回路sequence自体は起動しない。
- 独立経路は既存`c1_stage3_reference._ordered_exponentials`によるCAR one-body/全signed squared generatorのexpm。各armの基本δ operatorを作り、両経路で既存`np.linalg.matrix_power`によりQ回反復。131072-step回路は展開しない。
- full空間の元generatorと基底変換の両方向sector leakageを確認してからsectorを制限。元generator leakageは全て0。spin構造は既存exact-spin判定で受理し、projectionや別stateへのfallbackをしない。
- 核scalarは各基本δに一度exp(-i c δ)を付与。factor identityは既存diagonal primitiveのglobal phaseを含めて保持。B2のtail identityを別途重複加算しない。Γのvacuum=1規約によりrelative位相を固定し、global phase alignmentを使わない。
- signal record/複素差/円周位相差と`radius_phase_diagnostics`は既存コード。決定論ではnormalization=A=1、RTE disk=0、raw=z_PF、半径下限=|z_PF|-u_radius。
- PF判定値=arg差(z_PF,z_DF)+u_phase、model判定値=arg差(z_DF,z_phys)+u_phase、implementation/numerical判定値=独立経路間arg差+u_phase。いずれもP1 engineering allowanceを置いた数値診断で、厳密enclosureではない。
- 人工18件PASS（4.07秒）：8 armの旧schedule一致、負stage・signed factor・scalar、relative/diagonal作用とCAR expmの一致、未使用RTE予算保持、最初の不適格で停止。private load/回路constructor/shot関数を禁止したテスト。219件・既存B2 scienceは再実行していない。

## 8 armの結果

表の位相値は各armの最終評価roundのPF位相判定値（cap=.06 rad）。独立差の複素値/半径はそれぞれ≤1e-9が必要。
全armを事前指定順で、arm内はm昇順に評価した。失格armの以降roundは実行していない。

| Formula | substeps/δ | 評価m | 最終判定 | PF位相判定 [rad] | 独立複素信号差 | 独立半径差 |
|---|---:|---|---|---:|---:|---:|
| Full_S2 | 1 | 0..17 | 18/18 PASS | 0.0140160170723 | 2.82334160444e-10 | 2.82327716761e-10 |
| Full_S2 | 2 | 0..17 | 18/18 PASS | 0.00350397492399 | 3.74119986513e-10 | 3.74076103427e-10 |
| Full_S2 | 4 | 0..17 | STOP m=17 | 0.000875998927493 | 1.1525746133e-09 | 1.15257392519e-09 |
| Full_S2 | 8 | 0..16 | STOP m=16 | 0.000109508544448 | 1.07431271434e-09 | 1.07430886409e-09 |
| Yoshida4 | 1 | 0..17 | 18/18 PASS | 3.21100570482e-06 | 6.95484075548e-10 | 6.95369428705e-10 |
| Yoshida4 | 2 | 0..17 | STOP m=17 | 2.10093693537e-07 | 1.67252069414e-09 | 1.67205305157e-09 |
| Yoshida4 | 4 | 0..16 | STOP m=16 | 1.62594451935e-08 | 1.36834206296e-09 | 1.36830202724e-09 |
| Yoshida4 | 8 | 0..15 | STOP m=15 | 1.02001782354e-08 | 1.52925680514e-09 | 1.52925672126e-09 |

**失格5 armの最初の原因は全て`independent_numerical_match`**。当該点のmodel/PF/implementation位相capと正の半径条件は適格だが、複素差・半径差が固定独立閾値を超えたためPASSにしない。
基底δ operatorの独立差は約7.66e-15〜1.58e-13、native側unitarity成分残差は約1.21e-14〜3.10e-13で、長い行列累乗後の不一致は主に半径に現れた。これは丸め誤差の累積と整合する観測であり、原因の厳密な証明やPFの物理的失格を意味しない。unitarity補正・高精度再計算・救済は実行していない。

全18round通過armの値：

| arm | 最大PF位相判定 [rad] | 最小PF/raw半径下限 | 最大P1総予約位相 [rad] |
|---|---:|---:|---:|
| Full_S2_s1 | 0.0140160170723 | 0.999999989625532 | 0.360000020011531 |
| Full_S2_s2 | 0.00350397492399 | 0.999999989659251 | 0.360000020015345 |
| Yoshida4_s1 | 3.21100570482e-06 | 0.999999989191306 | 0.360000020022243 |

model位相判定の最大は約1.00096e-8 rad、全評価点のimplementation位相判定は約1.004e-8 rad以下。
総予約位相は既存`allocation.beta_total`（PF .06+RTE .04+stat .24）にmodel/implementation判定値とreserve .02を加算した。
PF観測値に合わせて予算を減らしたり、未使用RTE .04を統計へ移したりしていない。
結果JSONは各armのSOURCE/入力context、各roundの参照SOURCE・3信号・独立PF信号・誤差・bounds・判定・不適格原因を保持する。CSVは140点を同じ順序で保持する。

## 実行環境・証拠の範囲

SERVER 2 GPUA100-gleap02、Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1。CPU affinity64。
実行前の利用可能RAMは約42GiB、load13.56/17.33/19.76。小さいsector行列を使って全体約13.3秒で完了した。
恣意的なCPU/RAM/worker制限や共有設定変更はしていない。再利用venvと他worktree・他ユーザーを変更していない。

今回のPASSは**固定8 arm中の3 armに対する、元H stateの全18round条件付き数値信号適格性**。
元H/DF/state、P1目標・配分、scalar/ordering、no-prep範囲はB2と一致する。
B2の有限RTE減衰はここへ転用せず、決定論A=1とした。8 armは有限baseline集合であり大域最適とは主張しない。
B2の既存7,805,784 shotsは旧固定結果のままで、今回新しいshot数や費用比を算出していない。

strong PFのnearest-branch/energy復元、shot込み95%条件、実機95%成功、厳密数値/実装誤差保証、実H4 controlled全support/全長compiled回路のcorrectness、compiler価格・全RPE総費用は未評価/未認証。
今回のideal relative-Gaussian数値作用の照合や、以前のB2代表controlled 16/16をこれらの証明へ格上げしない。
新SCF/DF/state solve、B2 simulation、sampling、回路build/compile、価格MC、再最適化、GPT詳細レビューは実施していない。

## 次段階とGO/STOP

| 担当 | 別途明示承認後に行う候補作業 | 範囲・条件 |
|---|---|---|
| ユーザー | 次の科学実行範囲を選択・承認 | 3通過armを対象に進めるか、5停止armの数値照合問題を先に扱うかを指定。今回の失格を上書きしない。 |
| Codex | 通過3 armの保存raw=z_PFから既存decoderによるbranch/energy検証と条件付きshot接続 | 同じ18round、区間・符号・P1 allowance/β_stat/αを維持。shotではA=1,disk=0,PF半径下限を既存APIへ渡す。branch不適格・非正半径・条件不一致ならSTOP。今回未実行。 |
| Codex | 停止5 armを再評価する場合の数値経路の技術調査・必要な修正 | 科学パラメータと独立閾値は維持。新SOURCE・限定再評価の明示承認が必要。既存結果を修正しない。 |
| Codex | 実H4 controlled correctnessとfull no-prep価格の限定検証 | 別途範囲を固定して承認。同一compiler、spin-aware/relative fusion、核scalar・identity、X/Y wrapper、費用指標を揃える。回路compile/価格測定を新規承認前に起動しない。 |

費用比較は適格な信号・branch・条件付きshot・各round/X/Yの同一範囲価格が揃ってから行う。
現在の数値照合失格を解決するためのGPT詳細科学レビューは必須ではない。既存手法の接続と数値精度の問題であり、本作業ではレビューを開始していない。厳密な誤差/成功保証へ主張を拡張する科学判断は別途扱う。

**今回の科学実行は終了。結果公開とremote照合の後mandatory STOPを維持する。**
