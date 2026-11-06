# Track A B2 P1：m=0・17の限定信号検証

2026-11-07 JST / SERVER 2（GPUA100-gleap02）/ Codex主エージェント。

**`TWO_ROUND_CONDITIONAL_PASS`。固定P1のm0・m17は、入力identity、代表native correctness、独立数値照合、model/PF/RTE/実装の位相cap、正半径・normalizationを通過した。指定2roundの科学計算を完了してmandatory STOP。** これは理想native平均lawと採択済みengineering allowanceによる条件付き数値結果であり、全support・131072-step実回路の浮動小数点誤差証明、全18roundのbranch/energy/95%成功保証ではない。shot・価格・総費用は算出していない。

## 固定SOURCE・入力・条件

実行前にSOURCE **[`5ef55f923e73e61bada383d0cfe98283e7984e9d`][SOURCE]** をcommit公開し、remote/local tree・全5変更ファイルのbyte一致を確認してから1回実行した。結果の完全な値・SOURCE file SHA256・環境・診断は[`result.json`](../../../artifacts/track_a_b2_p1_two_round/2026-11-07/run01/result.json)に保存した。実行時間35.42秒。

| 固定対象 | 内容 |
|---|---|
| 契約 | ユーザー採択済み[P1 `7bc9527…`][P1]、[実行前PLAN](PLAN.md)。新しい承認証明・grantを作成していない |
| Hamiltonian/state | linear H4、1 Å、STO-3G、cutoff前H_phys、Nα=Nβ=2/sector36。保存された元H ground state。厳密基底状態を研究上仮定し、prep費用除外 |
| B2 | 保存DF12、generation-prefix L_D3、partial-S2、r4/K2、coefficient_atol=0、tail identity抽出。全signed factors、one-body、一体補正、核scalar、spin/JWを維持 |
| RPE | ε_E=.00015936001019904 Ha、95%目標、β_RPE=.4 rad、δ=.02 Ha^-1。全schedule m0..17のうち**m0/Q1/T.02とm17/Q131072/T2621.44だけ**を評価 |
| prior/phase | I_E=[−128,128] Ha、shift0、U=exp(−itH)、φ=−δE。P1の既存係数和S=103.50446476572802を事前根拠とし、保存基底energyから区間を作らない。decoder・branch成功判定は未実施 |
| 予算 | model .02、PF .06、RTE .04、stat .24、implementation/numerical .02、reserve .02 rad。α_X=α_Y=.05/(2·18)を保持。再配分なし |
| 数値判定 | u_z=u_radius=u_normalization=1e-8、u_phase=1e-8 rad。独立照合はu/10=1e-9以下。結果を見て変更していない。厳密包含ではない |
| 環境 | Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、OpenFermion1.6.1、h5py3.16.0。既存専用venv/extraを読取利用。GPU不使用。OS affinity64は観測値で、agentによるCPU/worker capではない。BLAS thread環境変数は未設定のまま |

開始時の空きRAMは約494 GiB、load average .10/.16/1.78。共有設定・他プロセスを変更せず、36次元sectorでの既存matrix-power経路を使用した。回路は8 system qubits＋1 ancilla、代表primitive/1 outer stepに限定し、巨大なm17回路を作っていない。

## private入力照合：PASS

以下は転送済みファイルを読取照合した値。private実体・provenance本文・積分/MO/state配列は公開していない。

| 対象 | SHA256 / 固定identity |
|---|---|
| state.npz file | `deb561367e216fd4c62d013c10a4adbbef5da8c3e8139c39170d71f80530a80c` |
| HDF5 file | `a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958` |
| full state / S0 array hash | `1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db` |
| 再構築raw H_phys / S0 array hash | `bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2` |
| DF12 NPZ file | `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。commit `bf5d3b2405b2bec4f88dd0589b75dd737b13e052`、path `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz` |
| DF Hamiltonian hash / full matrix hash | `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424` / `525cdfbe81b5b4990fbf9477cf97b1cea79cc5439b4615d31a5800969996ef2b`。異なるhash規約を区別 |

公開[`agent_b_input_spec.json`][INPUTSPEC]とprivate provenanceは固定[state成果 `89165cf…`][STATE]に一致。保存h/g・MO・AO overlapは[対応成果 `275813d…`][CORR]のhashに一致した。full state(256)/sector state(36)は`<c16`、basis indexは`<i8`。S0のdtype/shape/bytes hashを使い、M1の別hash規約へ置換していない。

`2p=alpha,2p+1=beta`、Qiskit full index=`sum(n_q*2**q)`、OF sector basis→既存bit reverse→保存Qiskit indicesとfull/sector stateの一致を確認した。stateの再正規化・phase変更、DF ground stateや別SCF stateへの代入はない。snapshot loaderが返す旧DF stateは破棄した。

保存HDF5を読み、既存spin liftとCAR kernelでraw H_physを再構築した。今回のraw state残差は2.52903e-15 Ha、raw Hermiticity defectはspectral norm 1.08379e-15 Ha。raw operatorは黙ってHermitian化せずexpmの診断対象とした。H_solveを使った過去のstate生成とその残差の来歴は維持し、targetをH_solve/H_srcへ変更していない。H_srcはcutoffを含む生成経路の参照であり、今回のenergy targetではない。

3原本の終了時file hashは開始時と一致。ファイル権限664も変更していない。private入力はGit staging/publication対象外である。

## 最小接続・人工検証

新規は`b2_p1_signal_validation.py`、薄いrunner、人工test、PLANの4ファイル。共有コードの変更は`o1_m1_resource_bridge.py`の半径・位相判定を**`radius_phase_diagnostics`へ分離し、従来の会計も同helperを呼ぶ**差分だけ。shot/価格式は変更していない。

- 旧B2 SOURCE `pr2_matched_accuracy_m1_execution.py`はbyte不変更。`_preparation_eigensystems`、`_apply_outer_step`、spectral作用、cutoff helperを再利用し、canonical `finite_taylor_operator`からmean-only経路を構成した。旧`_random_signal_record`/`_base_record`/`_axis_record`、DF固有state・T=.8専用`_target_signal`は呼ばない。
- `prepare_b2`/`B2RPERound`/`B2SignalRecord.bounds`を直接使用。model上界は未証明なのでcontextではNone。`account_signal_round(cost=None)`を呼ばず、signal-only判定で完結した。
- 独立経路は既存M1の`paired_polynomial_action`、finite moments、scipy expmをB2のforward/reverse sweepsへ接続。元H/DFの別照合はfull-space `expm_multiply`、主経路はsector expm。PF/finite/rawは元state上で評価し、DF/PF固有stateと仮定しない。
- nativeの半step・event・outer step・wrapperは既存AX emitter、DF-RTE builder、M1の`controlled_reference`/代表event選択を再利用した。新しいPF/RTE/shot推定式を実装していない。

**人工21件PASS（5.22秒）：新規10＋既存B2 kernel/固定SOURCE検査9＋変更したM1半径・位相会計の回帰2。** 人工二modeのcomplex/signed factors・prefix3、raw/corrected、cap超過拒否、元state不一致拒否、private hash失敗時の早期拒否、shot禁止、native相対位相を確認した。219件の既存接続を作り直したり一括再実行したりしていない。人工PASSと下記の分子科学結果は区別する。

## 2roundの5信号

表示は丸め値。完全なbinary64値はresult.jsonを参照。半径が1をわずかに超える数値は丸め誤差として保持し、物理的増幅や改善とは扱わない。

| m | 信号 | 実部 | 虚部 | 半径 | 位相(rad) |
|---:|---|---:|---:|---:|---:|
| 0 | H_phys | .999061499918 | .0433141937591 | 1.0000000000000002 | .0433277489727 |
| 0 | DF12 | .999061499918 | .0433141937591 | 1.0000000000000002 | .0433277489727 |
| 0 | ideal-tail PF | .999061504541 | .0433140871328 | .999999999999952 | .0433276422462 |
| 0 | finite corrected | .999061504541 | .0433140871328 | .999999999999459 | .0433276422462 |
| 0 | raw | .999027516781 | .0433126136011 | .999965980312121 | .0433276422462 |
| 17 | H_phys | .585901448635 | −.810382312546 | 1.0000000000000722 | −.944804341234 |
| 17 | DF12 | .585901448627 | −.810382312551 | .999999999999724 | −.944804341244 |
| 17 | ideal-tail PF | .574508508762 | −.818498608794 | .999999999619209 | −.958792784127 |
| 17 | finite corrected | .574508471520 | −.818498555961 | .999999934979395 | −.958792784257 |
| 17 | raw | .00664862811694 | −.00947225808259 | .011572723491925 | −.958792784259 |

`exp(−iT E_phys)`は厳密基底状態仮定の別参照として保存し、operatorから求めたz_physと分けた。既知Eで位相を補正したりbranchを選択したりしていない。

| 段階の観測誤差 | m0：複素絶対差 / 位相差(rad) | m17：複素絶対差 / 位相差(rad) |
|---|---:|---:|
| model：DF−H_phys | 7.63278e-17 / 7.63278e-17 | 9.62102e-12 / 9.61470e-12 |
| PF：ideal PF−DF | 1.06726e-7 / 1.06726e-7 | .01398832883 / .01398844288 |
| RTE：corrected−ideal PF | 4.93223e-13 / 9.85323e-16 | 6.46399e-8 / 1.29729e-10 |

normalization段階のraw−correctedの大きな振幅差は、意図した減衰であり、PF/RTE誤差に加算しない。m17のRTE複素差は主に半径差で、位相差そのものより大きい。資格判定では位相差だけで済ませず、既存の複素disk/radiusによるarcsin条件を使用した。

## normalization・数値照合・P1判定

λ_R=.5832704246339983、抽出tail identity=.09261516589219072、τ=.0029163521231699912、b_event=1.0000085051026735。K2の平均は既存paired degree3 polynomial。[減衰成果 `e69a51d…`][ATT]と同じlawである。

| 指標 | m0 | m17 |
|---|---:|---:|
| B | 1.000034020844717 | 86.41007760254907 |
| A | .9999659803126615 | .0115727242440354 |
| `abs(z_raw−A z_finite)` | 1.11239e-16 | 3.55558e-13 |
| PF半径下限（−u） | .999999989999952 | .999999989619209 |
| corrected複素誤差disk（＋u） | 1.00005e-8 | 7.46399e-8 |
| raw半径下限 A(ρ_PF,lb−disk) | .999965960312801 | .011572723260114 |
| model phase＋u / cap .02 | 1.00000001e-8：PASS | 1.00096147e-8：PASS |
| PF phase＋u / cap .06 | 1.16726458e-7：PASS | .013988452884：PASS |
| RTE disk由来phase / cap .04 | 1.00004933e-8：PASS | 7.46399450e-8：PASS |
| implementation phase＋u / cap .02 | 1.00000000e-8：条件付きPASS | 1.00049984e-8：条件付きPASS |
| P1総枠 / .4 | .360000020000：PASS | .360000020015：PASS |

総枠はPF/RTE/statの採択枠、model/実装の今回の値＋allowance、reserveを足した既存判定で、未使用枠の再配分ではない。各成分のcapも個別に確認した。stat .24と全18roundの均等αは将来測定用に確保しただけで、今回は測定・shot式を呼んでいない。

独立照合の最大複素差はm0で2.33148e-15、m17で3.04242e-10。各信号の複素差・半径差・位相差すべてが1e-9以下。Bのlog対direct差はm17でも1.42109e-14で、A、BAも通過した。rawの別経路は**attenuated event mean operatorを各stepに入れてQ乗したもの**で、主信号へAを掛けた値を独立参照と偽装していない。raw半径にAを二重適用していない。

## controlled correctnessと実装位相の証拠範囲

**16/16 checks PASS**：4 deterministic half-step、全tail係数のGaussian sum対CAR、代表controlled event5件、complete outer step2件、両ケースのX/Y wrapper4件。代表eventは既存選択からorder/phase/sign/support-sizeの重複を除く固定規則で選び、RNGを使っていない。order2の負位相も含む。

control-0 identity・control-1の**位相を合わせ直さない**operator対照で、prefix forward/reverse順序、核scalar `2.29310124732 Ha`とtail identityの一回計上を確認。YはSdg→HでIm(z)、XはRe(z)を返した。complete stepのmax-entry差は最大3.69718e-15。全tailのGaussian/CAR再構成差はmax-entry 7.94252e-13で、固定1e-9以下だった。

長roundについては、nativeの実際のGaussian/diagonal programから得たhalf-step operator・全tailの有限平均を用い、その**平均stepのQ乗**をnominal corrected信号と比較した。m17の位相差は4.99845e-12 radで、＋u後も.02枠内。未確認のまま0を代入していない。

ただし、この実装判定は既存canonical emitterの理想的意味と平均lawを前提とする条件付きengineering診断である。代表event5件は全support走査ではなく、全sampled trajectoryの浮動小数点誤差上界、compiler、装置noise、全長回路の実装保証は**未資格**。この限定PASSをそれらへ移送しない。

## 次の16roundへの提案とSTOP

**条件付き数値信号検証としては、同じ入力・P1・既存kernelで残りm=1..16へ進むGOを提案する。実行はユーザーの別途明示承認後のみ。** 2端点から中間roundの位相・半径を補間してPASSとはしない。次回も固定cap/allowanceと独立照合を維持し、不適格時は条件を変えずSTOPする。実装では今回の共通部分を再利用して対象roundだけを固定拡張し、新state solve・新DF・探索は不要。

全18roundのenergy/95%成功保証には全required roundの資格、branch、model/数値誤差の証拠水準、IID測定条件が残る。full no-prep価格・shot・全RPE費用・強baseline価格は全て未評価。A≈.01157でも、減衰だけから総費用改善の可否やr/K変更を判断しない。

条件付きpilotの継続にGPT詳細科学レビューは不要で、開始していない。厳密保証や既存十分条件を超える主張を求める場合は、その論点と承認範囲を別途定める。

**ユーザー**：残り16roundを同じ条件付き水準で検証するか判断し、進める場合は範囲を明示承認する。**Codex**：その承認まで科学実行を停止する。本成果のSOURCE・人工test・公開JSON・REPORTを専用branchへ公開し、remote SHA/tree/全変更内容を照合して**mandatory STOP**。旧SOURCE/input/raw/result/grant/marker/STOP、M1終了提案・M2保留を維持し、private配列・provenance・HDF5を公開しない。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/5ef55f923e73e61bada383d0cfe98283e7984e9d
[P1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7bc952772e451f27c1654dab6803192f2b204c9f/docs/research/track_a_b2_signal_conditions_20261106/REPORT.md
[INPUTSPEC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89165cf70958e7e4676d0580e3acb7faf588772c/artifacts/b2_hphys_ground_state/2026-11-06/run01/agent_b_input_spec.json
[STATE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89165cf70958e7e4676d0580e3acb7faf588772c/docs/research/b2_hphys_ground_state_20261106/REPORT.md
[CORR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/275813d0a12a4bca62a8c972312c80386829de22/docs/research/b2_original_df12_correspondence_20261106/REPORT.md
[ATT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e69a51de70651ced673c22d2b215683a9f30059b/docs/research/track_a_b2_rpe_attenuation_20261106/REPORT.md
