# Track A B2 P1：残り16roundの信号検証

2026-11-07 JST / SERVER 2（GPUA100-gleap02）/ Codex主エージェント。

**m=1から16まで固定順序で16/16 `CONDITIONAL_PASS`。保存済みm0・17と合わせて`ALL_18_P1_CONDITIONAL_SIGNAL_PASS`となった。** 全18roundでP1の成分cap・総枠、独立数値照合、正の半径下限、normalizationを満たした。これは固定入力・engineering allowance・既存nativeの理想的意味を前提とする条件付き信号検証である。nearest-branch成功、shot込み95%成功保証、価格・全RPE総費用改善は未評価。**指定計算を完了しmandatory STOP。**

## 1. 固定SOURCE・入力・科学条件

実行SOURCEは **[`071cb00955379b0bf4da149bd9793d4125968cad`][SOURCE]**。人工テスト後、実行前にcommit公開し、remote/local tree・5変更ファイルのbyte一致を確認した。新規科学実行は1回、3.45秒。m0・17の信号および代表controlled検査は再実行していない。

- 新規16roundの完全な5信号・誤差・半径・数値比較・判定・SOURCEは[`result.json`](../../../artifacts/track_a_b2_p1_remaining_rounds/2026-11-07/run01/result.json)に保存。
- 保存endpointを結合した[`all_18_rounds.json`](../../../artifacts/track_a_b2_p1_remaining_rounds/2026-11-07/run01/all_18_rounds.json)は各roundに`new/reused`を表示。再利用分は[直前成果 `ea2c091…`][PRIOR]の値をそのまま複写し、計算し直していない。異なる実行SOURCEの証拠を由来付きで整理したもので、費用会計の混合context集計ではない。
- P1正本は[`7bc9527…`][P1]、前回SOURCEは`5ef55f923e73e61bada383d0cfe98283e7984e9d`。前回結果・SOURCE・STOPは変更していない。

| 項目 | 固定条件 |
|---|---|
| 対象/state | Linear H4、1.00 Å、STO-3G、cutoff前H_phys、保存DF12、Nα=Nβ=2/sector36。保存元H基底stateを使い、厳密基底状態を研究上仮定。prep費用除外 |
| B2 | generation-prefix L_D3、partial-S2、r4/K2、全signed factors。one-body、一体補正、核scalar、tail identity抽出、spin/JW・bit-order不変更 |
| P1 | ε_E=.00015936001019904 Ha、95%目標、β_RPE=.4 rad、δ=.02 Ha^-1、Q_m=2^m。今回の新規対象はm1..16のみ |
| prior/phase | I_E=[−128,128] Ha、shift0、U=exp(−itH)、φ=−δE。既存係数和に由来する事前区間を保持し、保存energyから逆算しない。branch判定なし |
| phase cap(rad) | model .02、PF .06、RTE .04、stat .24、implementation/numerical .02、reserve .02。再配分なし |
| α/数値余裕 | 各round・X/Yに.05/(2·18)。u_z=u_radius=u_normalization=1e-8、u_phase=1e-8 rad。独立不一致≤u/10=1e-9。事後変更なし |

private loaderをそのまま再利用し、元state・raw H_phys・DF12・sector/orderと前回identityを照合した。保存h/g・MO・AO overlapのhash、HDF5から再構築したraw operator hashも一致。元stateを再正規化・phase変更せず、DF ground stateを破棄して代用しなかった。

| 必須入力identity | 一致した値 |
|---|---|
| state.npz / SHA256 | `deb561367e216fd4c62d013c10a4adbbef5da8c3e8139c39170d71f80530a80c` |
| 元HDF5 / SHA256 | `a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958` |
| full state / S0 array hash | `1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db` |
| raw H_phys / S0 array hash | `bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2` |
| DF12 NPZ / SHA256 | `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。commit `bf5d3b2405b2bec4f88dd0589b75dd737b13e052`、path `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz` |
| DF Hamiltonian hash | `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`（file hashとは別の規約） |

stateはfull256/sector36のcomplex128、2p=α/2p+1=β、Qiskit LSB、OF sectorからbit reverseした保存indexを使用。raw H_phys残差は2.52903e-15 Ha、Hermiticity defectはspectral 1.08379e-15 Ha。rawを黙ってHermitian化せず信号を評価した。H_srcをenergy targetに変更せず、H_solveによる過去のstate生成の来歴も維持した。private3原本の終了時hashは開始時と一致し、権限を含め不変更。配列・private provenance・HDF5は公開対象外。

## 2. 既存実装の再利用・人工検査

前回`b2_p1_signal_validation.py`のround内処理を`evaluate_signal_round`へ取り出した。private loader、mean-only kernel、`qualify_signals`など既存helperの内容は不変更で、`B2SignalRecord.bounds`→既存`radius_phase_diagnostics`の判定・式・予算をそのまま使用する。

新runnerはm1..16だけを昇順に指定し、各結果を保存してから次へ進み、初回不適格なら停止する。新しいPF/RTE式・shot式・最適化は加えていない。主経路は旧B2のspectral作用・half sweep、別経路は既存paired-polynomial/finite moments・expm、Q乗は既存matrix_power。元H/DFの別照合にはfull-space expm_multiplyを使う。全5信号を同じ元H state上で評価し、DF/PF固有stateとは仮定しない。

**今回は回路を作る`prepare_b2`や`controlled_checks`を呼ばない。** [減衰成果 `e69a51d…`][ATT]の固定係数projection・λ_R・identityと既存canonical finite lawを再利用した。このprojection hashと前回の実行可能preparation/tail hashは別fieldで保持し、projectionを回路preparationと偽装していない。

implementation/numericalの比較側は、既存`diag_hermitian`と独立CAR参照の`fock_lift`を使う理想Gaussian/diagonal平均を数値合成し、Q乗する。保存済みsupportだけを合成し、新しいsupport探索・event列挙・回路構築はしない。Gaussian basisの共通vacuum phaseは共役sandwichで消える既存規約を保持し、evolution信号をphase-alignしない。これは前回nativeの意味を前提とする理想平均参照であり、新たな実回路検証ではない。

**新規人工テスト4件PASS（2.98秒）。** すべて人工二modeまたはmock結果。QuantumCircuit構築、private読込、prepare_b2、controlled_checks、shot入口を禁止した状態で、mean-only/独立Gaussian・raw経路の一致、固定順序、数値/予算不適格時の即時停止、抽出したround処理を確認した。既存219件・前回分子endpointの再実行はしていない。

## 3. 新規16roundの結果

全行PASS。phase欄はallowanceを加えた判定値で、RTEは観測位相差だけでなく**複素誤差disk÷PF半径下限の既存arcsin変換**。表示は丸め値。各5信号の実部・虚部・半径・位相と隣接段階の複素差・円周位相差はJSONに保存した。

| m | Q | T(Ha^-1) | PF phase / cap .06 | RTE disk phase / cap .04 | raw半径下限 | B | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | .04 | 2.23453e-7 | 1.00010e-8 | .999931941783 | 1.00006804285 | PASS |
| 2 | 4 | .08 | 4.36906e-7 | 1.00020e-8 | .999863908195 | 1.00013609032 | PASS |
| 3 | 8 | .16 | 8.63811e-7 | 1.00039e-8 | .999727854903 | 1.00027219917 | PASS |
| 4 | 16 | .32 | 1.71762e-6 | 1.00079e-8 | .999455803853 | 1.00054447243 | PASS |
| 5 | 32 | .64 | 3.42523e-6 | 1.00158e-8 | .998911923814 | 1.00108924131 | PASS |
| 6 | 64 | 1.28 | 6.84039e-6 | 1.00316e-8 | .997825051454 | 1.00217966906 | PASS |
| 7 | 128 | 2.56 | 1.36706e-5 | 1.00631e-8 | .995654853300 | 1.00436408907 | PASS |
| 8 | 256 | 5.12 | 2.73311e-5 | 1.01262e-8 | .991328606921 | 1.00874722342 | PASS |
| 9 | 512 | 10.24 | 5.46524e-5 | 1.02525e-8 | .982732426684 | 1.01757096076 | PASS |
| 10 | 1024 | 20.48 | .000109294673 | 1.05050e-8 | .965763041939 | 1.03545066017 | PASS |
| 11 | 2048 | 40.96 | .000218579419 | 1.10100e-8 | .932698271949 | 1.07215806965 | PASS |
| 12 | 4096 | 81.92 | .000437148832 | 1.20200e-8 | .869926084143 | 1.14952292632 | PASS |
| 13 | 8192 | 163.84 | .000874287648 | 1.40400e-8 | .756771407009 | 1.32140295813 | PASS |
| 14 | 16384 | 327.68 | .001748565398 | 1.80800e-8 | .572702973939 | 1.74610577776 | PASS |
| 15 | 32768 | 655.36 | .003497120748 | 2.61600e-8 | .327988702984 | 3.04888538712 | PASS |
| 16 | 65536 | 1310.72 | .006994231396 | 4.23200e-8 | .107576591453 | 9.2957021038 | PASS |

新規16roundのmodel phase判定値の最大は1.00048074e-8 rad、implementation/numericalの最大は1.00037984e-8 radで、それぞれ.02以内。総枠は最大.360000020009 radで.4以内。PF/RTE/statは採択された枠を保持し、model/implは今回の値＋u、reserve .02を別に確保した既存判定である。未使用枠を流用していない。

## 4. 独立照合・normalization・半径

λ_R=.5832704246339983、tail identity=.09261516589219072、τ=.0029163521231699912、b_event=1.0000085051026735で前回と一致。核scalarは2.29310124732 Haを一度だけ含む。raw経路の独立参照は`finite_rte_operator_moments`の**attenuated event mean operatorをstepへ入れてからQ乗したもの**で、主信号にAを掛けただけの値ではない。

新規16roundで独立経路の最大複素差は1.52121e-10、最大位相差は7.52732e-12 rad。各信号の複素差・半径差・位相差、B/A/BAのlog対direct比較が固定1e-9以内だった。`abs(z_raw−A z_finite)`も最大2.51926e-12。raw減衰をPF/finite誤差へ加算せず、raw半径にAを二重適用しなかった。

PF半径下限>0、RTE disk<PF半径下限、raw半径下限`A(ρ_PF,lb−disk)>0`を全roundで確認。小さなbinary64丸めによる半径1超を物理的増幅や優位性と扱わない。観測差＋uはengineering allowance付き数値診断で、厳密なoperator/phase誤差包含ではない。

## 5. 保存済みendpointを含む全18round

| 項目 | 全18roundの判定値・範囲 |
|---|---|
| coverage | 新規m1..16＋保存m0/17。全18点を明示評価した証拠が揃い、補間でPASSにしていない |
| model phase 最大 | 1.00096147e-8 rad（m17、再利用）<.02 |
| PF phase 最大 | .013988452884 rad（m17、再利用）<.06 |
| RTE disk phase 最大 | 7.46399450e-8 rad（m17、再利用）<.04 |
| implementation/numerical 最大 | 1.00049984e-8 rad（m17、再利用）<.02、理想native平均の条件付き範囲 |
| 総位相枠 最大 | .360000020015 rad（m17）<.4、stat/reserveを保持 |
| raw半径下限 最小 | .0115727232601（m17、再利用）、全点で正 |
| 独立複素差 最大 | 3.04242e-10（m17、再利用）<1e-9 |

前回のcontrolled correctness **16/16 PASS**は、その固定SOURCE・入力に対する代表primitive/1 outer step・X/Y wrapperの証拠として引用しただけである。新たなnative回路・全support・全trajectory・長回路correctnessを検証したという意味ではない。今回の実装位相も未確認を0とはせず、理想Gaussian/diagonal平均Q乗との差＋uとして個別に判定し、この証拠水準を維持した。

| 主張 | 今回の状態 |
|---|---|
| 全18roundのP1条件付き数値信号適格性 | **PASS** |
| nearest-branch成功の確認 | 未評価 |
| shot数を含む95%成功保証 | 未評価・未認証 |
| 全RPE総費用改善・強PFとの同条件比較 | 未評価 |
| 全support・全長回路の厳密実装誤差保証 | 未資格。代表native証拠から格上げしない |

## 6. 終了・次の担当と承認範囲

次の候補は、**同じ固定P1・入力と保存18round信号を用いたnearest-branch復元・共通energy target/初期alias整合の限定検証**である。**ユーザー**がその必要性・範囲を判断し、進める場合は保存信号の新規解析とbranch検証を別途明示承認する。**Codex**は承認後にだけ既存decoderで実施する。既知基底energyによるbranch補正や、新しい信号・shot・価格計算をその範囲へ混ぜない。

shot/95%条件、full no-prep価格・価格不確かさ、強PFの適格性・総費用比較は後段の別scopeである。今回Aや半径が正であることだけから、総費用改善やr/K再最適化へ進まない。詳細科学レビューは開始していない。

実行環境はPython3.11.0rc1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、OpenFermion1.6.1、h5py3.16.0。開始時空きRAM約492 GiB、load .35/.41/.61、観測CPU affinity64、BLAS thread変数未設定。任意の資源cap・共有設定変更はせず、共通mean operatorをキャッシュし各roundを逐次判定した。新SCF/DF/state solve、sampling、回路build/compile、shot、baseline、価格・総費用、詳細レビューは0件。

専用branchへSOURCE・公開JSON・REPORTを公開し、remote SHA/tree/全変更内容一致を確認して**mandatory STOP**。旧SOURCE/input/raw/result/grant/marker/STOP、M1終了提案・M2保留、private入力を保全し、後続科学実行へ自動進行しない。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/071cb00955379b0bf4da149bd9793d4125968cad
[PRIOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5/docs/research/track_a_b2_p1_two_round_20261107/REPORT.md
[P1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7bc952772e451f27c1654dab6803192f2b204c9f/docs/research/track_a_b2_signal_conditions_20261106/REPORT.md
[ATT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e69a51de70651ced673c22d2b215683a9f30059b/docs/research/track_a_b2_rpe_attenuation_20261106/REPORT.md
