# 新HP強PF 5 arm：保存branch・energy復元と条件付きshot会計

2026-11-07 JST / SERVER 2 GPUA100-gleap02 / Codex B主エージェント。

**新HP 5 arm×18roundが全点で条件付きbranch/energy/shot PASS。各armはX/Y各9,270、全18roundで18,540 shots。** 旧3 armは保存結果だけを再利用して全8 arm比較表へ転記した。旧5 STOPは過去の別SOURCEの履歴として保全し、新HP結果で書き換えない。科学実行を終了し、公開照合後mandatory STOP。

## SOURCE・共通科学条件

- 本実行SOURCE：[`509f829f5d58f67887f2ade5301965051f89fa04`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/509f829f5d58f67887f2ade5301965051f89fa04)。人工確認後にcommit固定・公開し、6 pathの原内容とlocal/remote treeを確認して1回実行。
- 新HP信号：[`3e316e071fe671aa61b31bc89f83bf4380d0e92f`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3e316e071fe671aa61b31bc89f83bf4380d0e92f)、科学SOURCE `d401d82ab9e77ab43c6eec3483b3086633dcea6d`、precision adapter SOURCE `75cb38e686ad8b82d00d787531d767e816e1331d`。Full S2 s4/s8、Yoshida4 s2/s4/s8の保存90点のみ新規再解析。
- 旧3 arm：[`55fa5f5eb121ee01ae963da7ab6e5bbe9a507173`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/55fa5f5eb121ee01ae963da7ab6e5bbe9a507173)、branch/shot SOURCE `909171eb3e4f1f4b0ec9e4234624b7000ecd8215`。元信号SOURCE `a7940b58c6cb939d77e9f8e44d214b3787b7e101`／結果 `2583ce4957ed44e914ce6571e3e676830439dfa7`。Full S2 s1/s2、Yoshida4 s1にdecoder・shot APIを再実行していない。
- P1 `7bc952772e451f27c1654dab6803192f2b204c9f`、B2 branch `d424bb8bec1aaf1224a33513a06e64801749a69f`、shot `0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`。原HP/旧結果のcommit/path/blob/hash、SOURCE、P1、元H/state/DF12/sector/order、保存参照の来歴を照合。

Targetはcutoff前H_phys、linear H4/1.00 Å/STO-3G、保存DF12、元H保存基底state、Nα=Nβ=2/sector36、2p=α/2p+1=β・Qiskit LSB、prep費用除外。H_phys S0 hash `bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2`、state S0 hash `1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db`、DF H hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`は全8 armで一致。これは公開保存metadataの照合で、private入力・state・積分の再計算ではない。

ε_E=.00015936001019904 Ha、成功目標95%、β_RPE=.4、δ=.02 Ha^-1、m0..17/Q=2^m、T=Qδ、prior[-128,128] Ha/shift0、U=exp(-itH)/φ=-δEを維持。位相配分model/PF/RTE/statistical/implementation/reserve=.02/.06/.04/.24/.02/.02 rad、各軸α=.05/36。u_z/u_radius/u_normalization=1e-8、u_phase=1e-8、独立照合閾値1e-9も不変更。**決定論A=1、disk=0、raw=z_PF。未使用RTE .04は再配分していない。**

## 全8 armのprovenance付き比較

下表のenergyは保存理想平均からの復元値。すべてP1 energy目標内。元H参照E=-2.1663874486347607 Haは新5 armのdecoderが完了/STOPしてから開き、事後correctness確認だけに使った。既知Eによる枝選択・初期推定・位相補正はない。

| arm | Signal SOURCE・今回の扱い | 信号資格 | Branch | 最終E_hat [Ha] | 元Hとの差 [Ha] | X shots | Y shots | 合計shots |
|---|---|---|---|---:|---:|---:|---:|---:|
| Full_S2_s1 | 旧 a7940b5 / 保存転記 | 18/18 条件付きPASS | 一意PASS | -2.166382101952369 | 5.34668239194e-06 | 9270 | 9270 | 18540 |
| Full_S2_s2 | 旧 a7940b5 / 保存転記 | 18/18 条件付きPASS | 一意PASS | -2.1663861119782135 | 1.33665654722e-06 | 9270 | 9270 | 18540 |
| Full_S2_s4 | 新HP d401d82 / 今回会計 | 18/18 条件付きPASS | 一意PASS | -2.166387114471502 | 3.34163258842e-07 | 9270 | 9270 | 18540 |
| Full_S2_s8 | 新HP d401d82 / 今回会計 | 18/18 条件付きPASS | 一意PASS | -2.1663873650940024 | 8.35407583111e-08 | 9270 | 9270 | 18540 |
| Yoshida4_s1 | 旧 a7940b5 / 保存転記 | 18/18 条件付きPASS | 一意PASS | -2.1663874498558444 | 1.22108367862e-09 | 9270 | 9270 | 18540 |
| Yoshida4_s2 | 新HP d401d82 / 今回会計 | 18/18 条件付きPASS | 一意PASS | -2.1663874487110713 | 7.63105134638e-11 | 9270 | 9270 | 18540 |
| Yoshida4_s4 | 新HP d401d82 / 今回会計 | 18/18 条件付きPASS | 一意PASS | -2.1663874486395285 | 4.76774175695e-12 | 9270 | 9270 | 18540 |
| Yoshida4_s8 | 新HP d401d82 / 今回会計 | 18/18 条件付きPASS | 一意PASS | -2.1663874486350654 | 3.04645197957e-13 | 9270 | 9270 | 18540 |

**各行は独立候補で、保証はすべて条件付き／`not_certified`。** 全8 armを同一SOURCEで実行した結果でも、8 arm同時使用に対する95%保証でもない。価格・総回路費用は全行未評価。shot数だけで費用順位を確定しない。

新5 armは各roundでN_X=N_Y=515、round合計1,030。既存18,540やB2の7,805,784 shotsを代入せず、各保存半径から実APIで取得した。α和は各armで厳密に1/20=.05。JSON/CSV再読込・整数和の照合も5/5 PASS。

新5 armの全branchは一意、scalar/vectorized一致、保存HP位相を直接渡した既存decoderとも枝が一致。m17の枝は全5 armで904。初期aliasと既存strict β<π/3条件、各roundの事後oracle枝・共通元H位相・energy nominal条件を満たす。最終nominal分解能=.000152587890625 Ha。

## HP→scalar精度の接続と留保

[新adapter](../../../src/trotterlib/b2_p1_hp_branch_shots.py)は保存schema接続だけを担当する。HPの80/120桁資格、独立native/CAR差、conversion_error、HP signalとradius/phase記録を既存1e-9で照合し、元と同じ`complex(float(real),float(imag))`を使用。信号の正規化・半径強制・phase fitは行わない。

- `b2_p1_strong_pf_branch_shots.decode_arms`→既存`reconstruct_rpe_phase`/scalar-vectorized照合をそのまま使用。保存HP位相をfloatにした別入口も同じ既存decoderへ渡す。枝距離gapが変換誤差の伝播と**既存**tie許容値1e-12より十分大きいことを確認した。
- 既に選択された枝だけを120桁の保存位相・πでliftし、energyへの変換を比較した。HPで新しいnearest-branch searchを実装しておらず、既知energyを使わない。5 armのenergy目標判定はHP lift/既存scalarで一致し、変換差より目標境界までの余裕が大きかった。
- PF複素値のbinary64変換差は最大約6.87342e-17、選択済み枝のHP/scalar energy差は最大約1.20160e-14 Ha。これは固定保存値の変換診断であり、物理energyや実装の厳密誤差上界ではない。特にYoshida4の極小energy差の下位桁を厳密な精度保証へ読み替えない。
- 保存scalar半径下限を使い、旧`account_arm`→`account_saved_round`→`account_radius_round`→既存Hoeffding/ceilを直接実行。HP半径と保存scalar半径を囲む隣接float両端も**同じAPI**で検査し、全90点でN_X/N_Yの整数が一致。これは入力変換に関する既存binary64式の安定性で、sin/log演算までの厳密interval保証を新たに認定するものではない。新しいshot式・数値閾値は追加していない。

旧SOURCE `55fa5f5…`から`b2_p1_strong_pf_branch_shots.py`と対応testを原内容で再利用した。既存科学SOURCE、HP構成/信号、decoder、shot式は不変更。新規はHP adapter、runner、人工testとPLAN。旧5 STOPのraw/結果/SOURCEを新HP5 PASSに置換していない。

人工**67件PASS（新規25＋既存42）**。架空HP複素値のみで変換・保存資格の拒否、正負位相、枝・energy・ceil安定性、旧3 arm非再計算、全8行比較、arm固有STOP後の継続と失敗数値保存を確認。private load、PF構成・signal pipeline、回路constructorを禁止した。人工PASS自体を実H4結果とは扱わない。

## 成果物・実行

[全体JSON・8 arm比較](../../../artifacts/track_a_b2_p1_hp_branch_shots/2026-11-07/run01/result.json)、[全8 arm比較CSV](../../../artifacts/track_a_b2_p1_hp_branch_shots/2026-11-07/run01/comparison.csv)。各旧行には元commit/path/blob/SHA256、各新行には新科学SOURCEと本実行SOURCE/結果pathを保持する。

| 新HP arm | 全18round JSON | CSV |
|---|---|---|
| Full_S2_s4 | [result.json](../../../artifacts/track_a_b2_p1_hp_branch_shots/2026-11-07/run01/Full_S2_s4/result.json) | [rounds.csv](../../../artifacts/track_a_b2_p1_hp_branch_shots/2026-11-07/run01/Full_S2_s4/rounds.csv) |
| Full_S2_s8 | [result.json](../../../artifacts/track_a_b2_p1_hp_branch_shots/2026-11-07/run01/Full_S2_s8/result.json) | [rounds.csv](../../../artifacts/track_a_b2_p1_hp_branch_shots/2026-11-07/run01/Full_S2_s8/rounds.csv) |
| Yoshida4_s2 | [result.json](../../../artifacts/track_a_b2_p1_hp_branch_shots/2026-11-07/run01/Yoshida4_s2/result.json) | [rounds.csv](../../../artifacts/track_a_b2_p1_hp_branch_shots/2026-11-07/run01/Yoshida4_s2/rounds.csv) |
| Yoshida4_s4 | [result.json](../../../artifacts/track_a_b2_p1_hp_branch_shots/2026-11-07/run01/Yoshida4_s4/result.json) | [rounds.csv](../../../artifacts/track_a_b2_p1_hp_branch_shots/2026-11-07/run01/Yoshida4_s4/rounds.csv) |
| Yoshida4_s8 | [result.json](../../../artifacts/track_a_b2_p1_hp_branch_shots/2026-11-07/run01/Yoshida4_s8/result.json) | [rounds.csv](../../../artifacts/track_a_b2_p1_hp_branch_shots/2026-11-07/run01/Yoshida4_s8/rounds.csv) |

各round JSONは元HP signal文字列、scalar値、保存PF半径下限、HP/scalar変換誤差、branch/energy、N_X/N_Y/合計、全予算・α・変換ceil検査を保持。`aggregate_shots`・`verify_serialized_totals`を再利用し、countsをfloat化せず集計した。価格はnullであり0としていない。

新科学実行は固定保存5 arm/90点の再解析1回、5.543秒。独立5 processと既存vectorized decoderを使用、OS affinity64、開始前MemAvailable約493.4 GiB。任意CPU/RAM/BLAS上限・共有設定変更・他ユーザー/他チャットのprocess操作はなし。旧3 arm再計算、private入力読込、新H/DF/state、HP step/PF simulation、再適格化、測定sampling、回路build/compile、CX/RZ/depth価格、価格MC、総費用、最適化、詳細科学レビューは0件。

公開前の既存manifest checker・REPORTローカルリンク・`git diff --check`はPASS。固定SOURCE6 pathは実行後もcommitと同一bytes。保存済み5 armの整数再読込・ceil安定性status、比較表の新5/旧3区分を確認した。

## 次の担当とmandatory STOP

条件付き十分shot数は最小shot数・実機95%保証ではない。元H基底state仮定、採択prior、P1 engineering allowance、理想独立±1測定に依存する。数値／実装の厳密enclosure、全support/全長controlled回路correctness、費用優位は未認証/未評価。

**ユーザー**：全8 armのSOURCE付き結果を後段の価格評価へ渡す場合、その対象・全round/X/Y・compiler/version/seed/backend・費用指標・whole no-prep scope・価格MC/不確かさの範囲を指定する。**Codex C**：自身の別途許可された価格評価範囲で、対応するarm/round/axisの価格を扱う。本結果だけではHP数値経路と実際のnative emitter/compiled gate列の一致を証明しない。**Codex B**：本作業を公開確認で終了し、価格・総RPE費用・追加最適化を自動開始しない。

Codex A/Cのworktree・SOURCE・途中成果、旧SOURCE/input/raw/result/grant/ledger/marker/STOP、private state/HDF5/MO/provenanceを変更・公開していない。GitHub連携で専用branchへ公開し、remote SHA/tree・全変更内容一致を確認したら**mandatory STOP**。Codex Cの完了は待たない。
