# 強PFの3 PASS arm：P1保存18roundのbranch・shot会計

2026-11-07 JST / SERVER 2 GPUA100-gleap02 / Codex B（主エージェントのみ）。

**Full S2 s1・s2、Yoshida4 s1の全18roundで条件付きbranch/energy/shot会計PASS。各armのX/Yは9,270ずつ、計18,540 shots。** 5 STOP armは再解析していない。今回の保存平均に対する結果を実機95%成功・厳密数値包含・総回路費用の優位性に格上げしない。**科学実行を終了し、公開照合後mandatory STOP。**

## 固定SOURCEと対象

- 新実行SOURCE：[`909171eb3e4f1f4b0ec9e4234624b7000ecd8215`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/909171eb3e4f1f4b0ec9e4234624b7000ecd8215)。人工確認後、GitHub連携でSOURCEをcommit公開し、local/remote treeと4公開pathの原内容一致を確認して1回実行した。
- 固定信号結果：[`2583ce4957ed44e914ce6571e3e676830439dfa7`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/2583ce4957ed44e914ce6571e3e676830439dfa7)、原科学SOURCE `a7940b58c6cb939d77e9f8e44d214b3787b7e101`。指定JSONの`Full_S2_s1`、`Full_S2_s2`、`Yoshida4_s1`だけを数値評価。
- P1正本 `7bc952772e451f27c1654dab6803192f2b204c9f`。B2 branch `d424bb8bec1aaf1224a33513a06e64801749a69f`とshot `0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`は条件・元H/state identity・原入力blob/SHA256・SOURCEを照合する参照。B2信号・branch・shotを再計算していない。
- 元Hはcutoff前H_phys、linear H4/1.00 Å/STO-3G、保存DF12、元H保存基底state（厳密基底stateを研究上仮定）、Nα=Nβ=2/sector36、interleaved spin・Qiskit LSB、state preparation費用除外。DF/PF stateへ置換しない。
- 元H S0 hash `bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2`、state S0 hash `1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db`、DF H hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。これは公開保存metadataの照合で、private配列・積分の再読込検証ではない。

P1：ε_E=.00015936001019904 Ha、β_RPE=.4 rad、δ=.02 Ha^-1、m0..17/Q=2^m/T=Qδ、energy prior[-128,128] Ha、shift0、U=exp(-itH)、φ=-δE、95%目標。位相配分model/PF/RTE/statistical/implementation/reserve=.02/.06/.04/.24/.02/.02 rad、各軸α=.05/36=1/720。u_z/u_radius/u_normalization=1e-8、u_phase=1e-8 rad、独立照合閾値1e-9を維持。**決定論RTE=0でも.04予算は予約のまま。再配分なし。**

## 各armの結果

参照E_phys=-2.1663874486347607 Haは全3 decoderの完了後に開き、正しさの照合だけに使った。下表のenergyは保存理想複素平均からの推定であり、finite-shot測定結果ではない。

| arm | 最終E_hat [Ha] | 元H energyとの差 [Ha] | branch/energy | 全X shots | 全Y shots | 全18round shots |
|---|---:|---:|---|---:|---:|---:|
| Full_S2_s1 | -2.166382101952369 | 5.34668239194e-06 | PASS | 9270 | 9270 | 18540 |
| Full_S2_s2 | -2.1663861119782135 | 1.33665654722e-06 | PASS | 9270 | 9270 | 18540 |
| Yoshida4_s1 | -2.1663874498558444 | 1.22108367862e-09 | PASS | 9270 | 9270 | 18540 |

全54点で正のPF半径下限を確認。すべてのroundで各軸515、round合計1,030 shotsとなった。A=1、corrected finite-RTE disk=0、raw=z_PF、ρ_PF,lb=abs(z_PF)-u_radiusを既存会計へ渡した。単位半径への置換、半径正規化、既知energyによる位相補正はない。

m0の事前区間によるalias余裕は3 arm共通で0.18159265358979304 rad。全18roundのnearest-branchは一意で、scalar/vectorizedの枝・最終位相が一致。全3 armのm17 branch indexは904。事後oracle枝も全roundで一致し、元H共通位相からの差はβ=.4以内。β<π/3と初期alias条件を満たすという既存decoderの条件付き十分条件を確認した。最終nominal分解能=.000152587890625 HaもP1目標内。既知energyを使った枝の選択・修正・phase fitは0回。

B2の既存**7,805,784 shots**は固定結果の引用であり、強PF側に代入していない。今回の18,540は各armの保存実半径から実APIで算出した値。**shot数の大小は総回路費用の優劣を示さない。** 単価・回路長・compiler価格は未取得。

## 保存した証拠と再利用

[全体summary JSON](../../../artifacts/track_a_b2_p1_strong_pf_branch_shots/2026-11-07/run01/result.json)に実行SOURCE、旧SOURCE hash、保存入力commit/path/blob/SHA256、共通科学条件、arm別energy/shot/STOP状態を記録。

| arm | 全18round JSON | CSV |
|---|---|---|
| Full S2 s1 | [result.json](../../../artifacts/track_a_b2_p1_strong_pf_branch_shots/2026-11-07/run01/Full_S2_s1/result.json) | [rounds.csv](../../../artifacts/track_a_b2_p1_strong_pf_branch_shots/2026-11-07/run01/Full_S2_s1/rounds.csv) |
| Full S2 s2 | [result.json](../../../artifacts/track_a_b2_p1_strong_pf_branch_shots/2026-11-07/run01/Full_S2_s2/result.json) | [rounds.csv](../../../artifacts/track_a_b2_p1_strong_pf_branch_shots/2026-11-07/run01/Full_S2_s2/rounds.csv) |
| Yoshida4 s1 | [result.json](../../../artifacts/track_a_b2_p1_strong_pf_branch_shots/2026-11-07/run01/Yoshida4_s1/result.json) | [rounds.csv](../../../artifacts/track_a_b2_p1_strong_pf_branch_shots/2026-11-07/run01/Yoshida4_s1/rounds.csv) |

各roundはm/Q/T、PF/raw複素値・半径・位相、branch index、energy推定/誤差、PF半径下限、N_X/N_Y/合計、誤差cap・軸別α・条件付き資格を保持する。JSONにはrunner-upとの差・曖昧性・scalar/vectorized一致・事後oracle照合も保持する。

- 新規[`b2_p1_strong_pf_branch_shots.py`](../../../src/trotterlib/b2_p1_strong_pf_branch_shots.py)と[runner](../../../scripts/run_b2_p1_strong_pf_branch_shots.py)は保存JSONの薄い接続のみ。既存の強PF科学SOURCE、signal計算、decoder、shot式は不変更。
- `b2_p1_branch_validation.decode_raw_means`→既存`reconstruct_rpe_phase`/`_vectorized_reconstruction`、`initial_alias`、`posthoc_correctness`を直接再利用。元H energy recordだけでなく、energyを含むB2 branch/shot metadataも全decoder終了後に読み、共有contextをshot前に確認した。
- `radius_phase_diagnostics`で保存boundsとP1 cap/reserveを確認。`b2_p1_shot_accounting.account_saved_round`→`account_radius_round`→既存`_hoeffding_shots`を直接使用。定義・ceil処理は再実装しない。`cost=None`、`config=None`。必須cost_metric識別子は互換metadataに限り、価格関数を呼ばない。
- `aggregate_shots`で各armの18roundと整数和を検査。`verify_serialized_totals`を各armで再利用し、CSVのDecimal逆順和・JSONとの一致を確認。全3 armで9,270+9,270=18,540と一致。αの厳密和は各armで1/20=.05。3 armは独立の候補会計で、3 arm同時成功を95%と主張しない。

公開前に既存manifest checker・REPORTのローカルリンク・`git diff --check`もPASS。実行SOURCEは公開commitと同じbytesを保持している。

人工testは**60件PASS（新規28・既存branch6/shot8/strong PF18）**。最終出力表示修正後、新規28件を再実行して全PASS、既存32件は変更なし。人工testは架空複素信号だけを使い、private/saved science読込とPF kernel・回路constructorを禁止。共有不一致全STOP、一arm STOP後の継続、energyの事後参照・枝の不修正、既存shot式呼出、RTE予約、整数/CSV保存を検査した。

科学実行は保存3 arm/54点の再解析1回、3.230秒。Python 3.12.3、NumPy 1.26.4、SciPy 1.14.1。実行前affinity64、利用可能RAM約492.7 GiB、load約1.29/1.56/4.10。既存vectorized decoderを使う小さいscalar-data処理で、任意のCPU/RAM/worker上限は設けず、GPU・新規依存導入・共有設定変更・他プロセス操作は行っていない。

## 保証の範囲・次の作業・STOP

得られたのは**P1 engineering allowanceと元H基底state仮定、理想独立±1測定、採択prior/branch十分条件に依存する、保存平均のenergy照合と十分shot数**。最小shot数、実機95%成功、数値/実装の厳密enclosure、全support/全長controlled回路correctness、総回路費用は未認証/未評価。元の強PF信号評価は**3 PASS/5 STOPのまま**。5 STOP armの診断・step精度改善・再資格判定には触れていない。

**ユーザー**：次に価格評価へ進むか判断し、対象arm・全round/X/Y・compiler/version/seed/backend・費用指標（CX/RZ等）・whole no-prep scope・価格MC cohort/不確かさを具体的に別途承認する。**Codex**：その承認後のみ、同じ元H/state/DF12/P1とspin-aware/relative fusion・核scalar/identity・ordinary-controlled相対位相・準備除外を維持して、必要なcontrolled correctnessと価格を取得する。今回のcountsと同じarm/round/axisの価格が揃うまでΣN×E[C]を算出しない。5 STOP armの再適格化は別範囲のまま。

今回、新しい実H4 PF信号/演算子、SCF/DF/state、private入力読込、sampling、回路build/compile、価格MC、総費用、再最適化、詳細科学レビューは0件。Codex Aのworktree/SOURCE/途中成果、旧SOURCE/input/raw/result/grant/ledger/marker/STOPとprivate入力を保全した。公開は接続済みGitHub連携を使用し、共有Git設定・認証は不変更。

**3 armの科学実行は完了。公開remote SHA・tree・全変更pathの内容照合後mandatory STOPし、価格・総回路費用・5 arm再適格化へ自動進行しない。**
