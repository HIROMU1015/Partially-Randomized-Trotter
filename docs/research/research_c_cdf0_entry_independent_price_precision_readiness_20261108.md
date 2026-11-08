# 研究C：CDF0対ENTRY・独立追加価格検証の実行準備（2026-11-08）

**CASE A：ENTRY E2の保存ground原本を取得し、全固定入力の受動検査がPASSした。** 新科学実行は未承認・未実行。主案は、固定CDF0/r64と固定ENTRY E2/r16を、それぞれ新しい128標本で一度だけ評価する案である。旧CDF0比較の`UNRESOLVED_PRECISION_TARGET`も、旧ENTRY対PR-awareの判定も変更しない。公開照合後mandatory STOPする。

統合運用指示`research_project_instructions_unified_2026-10-25.md`の現作業checkout側原文を読み、§8.1/8.2/8.4/10/10.3を適用した。承認証明用ファイル・新grantは追加しない。

## 1. 固定科学条件と旧結果

Linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2（sector36/system256/controlled512）、number-CDF rank8、元frame順序0..7・LD2。線形Z再配置は`tail_linear_Z_to_existing_kappa_keep_frame_identity_v1`。Partial ordered S2/s1、δ=h=.1、K0、RPE m0..12、候補固有own-DF sector ground oracle、状態準備除外。CA=.00159 Ha、DF reserve=CA/8、βtotal=π/6、βstat=π/12、failure=.01、axis=.01/26を維持する。各候補の近似Hamiltonianとown-groundは異なり、同一とは仮定しない。

固定compilerはQiskit1.3.0/P0、optimization_level1/seed17、basis rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false。既存MPFR160 declared phase、phase fitなし、actual inverse、head/X/Y shell、fenced serialを再利用する。本件は有限回路のbinary64 correctnessを保持する計画であり、全wordの正常終了証明・Rust一様誤差保証を追加条件にしない。

次表は保存summaryからの引用で、新しい統計再解析ではない。すべて上記H4・固定rank8/LD2/δ=.1のideal-law/exact-native条件付きfinite-bit価格proxyである。

| 項目 | 通常CDF0 | ENTRY E2 |
|---|---:|---:|
| 候補 | CDF0_STAGE_A_X0_V2_GAMMA0_FINAL_R8 | H4_100_C0_ENTRY_g003_budget64 |
| 選択済みr | 64 | 16 |
| 旧final標本 | 64 | 64 |
| kernel RZ平均 | 11295.15625 | 5512.3125 |
| 公開済みkernel標本分散 | 735023.0228174604 | 177196.72619047618 |
| 公開済みkernel平均SE | 107.16685463109766 | 52.61842687430127 |
| 保存W | 20535472 | 13718624 |
| 保存shell RZ | 60375 | 52913 |
| 保存RPE shots | 17250 | 15118 |
| 保存条件付き総RZ | 231951425282.5 | 75621395471 |

旧CDF0−保存ENTRY差点156330029811.5、名目95% Welch差区間[151717539065.4622,160942520557.5378]、半幅4612490746.037811、半幅/ENTRY=.060994520364。登録5%を超えるため`UNRESOLVED_PRECISION_TARGET`。点値67.3977%減や区間の正符号だけで正式PASSに変更しない。実CA達成、全law価格移送、FT総費用、一般最適性は未資格である。

## 2. 原本・SOURCE・旧標本の来歴

固定GitHub本文とSOURCE blobをローカルgit objectに照合した。対応は[technical.json](../../artifacts/research_c_independent_price_precision/2026-11-08/technical/technical.json)に置く。

- [CDF0結果 cbc517e7](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cbc517e75ad47054e4246748351f56269dac3a0c/docs/research/research_c_cdf0_z_relocation_rerun02_results_20261108.md)：科学コードbde646c2、SOURCE SHA256 `640052dfeed3b886bb95e11abb410e3275cdda1ffe82661e544037c120f94c32`。SOURCE112ファイルを保持し、CPU専用Aer0.15.1でfresh run02を完了した履歴。selection32はr64/r128、最終r64は別namespaceの32×2バッチ。CSV128件はselection64＋final64であり、全128件をfinal標本として扱わない。
- [ENTRY結果 3015d4fb](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/docs/research/research_c_h4_relocated_downstream_results_20261106.md)：科学コードcfc64b70、コードSOURCE `6bd88bc86bd1da2876f87b8ae67be509b15727f01517cea9c7219f91c350c4cc`。許可済みhandoffは旧namespaceとsampling SOURCE `bb102f16c3bf757853f9e9fc98d811dc230592d71cd6bad57bcce0363fa2046e`を維持し、検証済み部分標本を復元した。E2/final_r16はselection32と別の32×2バッチ。公開CSV全800件からE2 finalを取り直して再計算していない。
- [三者比較 70981380](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/709813805d41e470676e311cd63a740af23bf33c/docs/research/research_c_three_method_z_relocated_comparison_20261108/REPORT.md)：通常CDF0はStage A γ0終端、ENTRY E2はC0/γ.03/64更新、PR-aware Pは別のγ.1候補。新PR-aware価格評価は本提案に含めない。比較は保存された有限最適化結果に条件付きであり、同一初期値・同一目的を揃えた因果的な正則化効果ではない。

旧master=2026110501、PCG64、cohortは`integrated:<key>:<selection|final>:r<r>:batch<bb>`。keyは`relocated:<version>:<JU SHA>:<version>:LD2`、CDF0だけ先頭に`cdf0-20261108:`が付く。roleはCDF0/E1/E2=baseline、P=proposed、axisは`INTEGRATED_SELECTION_PRICE_ONLY`または`INTEGRATED_FINAL_PRICE_ONLY`。既存`trajectory_seed`へ(master,cohort,SOURCE)、role、候補identity、round=-1、axis、shot=0..31、occurrence=0を渡す。

同じ候補/representation/r/splitを共有した旧結果は独立な追試ではない。旧CDF0比較は保存E2 final64を再利用しており、今回の双方freshという主案に情報価値がある。新namespaceの相違はseedの衝突・再利用を防ぐが、PCG64のjoint lawや数学的IIDを証明するものではない。

## 3. 統計規則の継承

既存`c_near_r32_statistics.primary`は各armのaffine価格`W×kernel RZ+shell RZ`をSciPy `ttest_ind(equal_var=False)`へ渡し、指定confidenceのWelch差区間と自由度を得る。母集団coverageを検証した区間ではなく、名目の固定N・近似IIDモデルである。

主案では、事前固定したnew-CDF0 minus new-ENTRYの**一つのfamily、一つの主RZ比較、名目95%**とする。旧ENTRY対Pの二主比較family（各97.5%）は変更しない。過去と今回を合わせた同時coverageや、複数の追試から有利な結果だけを選ぶ資格は主張しない。他metricは記述量だけで、D比較・secondary推論・ratio CIは実施しない。

精度基準は`Welch区間半幅 / 今回の新ENTRY総RZ点値 ≤ .05`。分母も新cohortの推定値であり、保存点値を実行時の分母にしない。skewness絶対値≤2、delete-one最大平均移動/各armの個別t区間半幅≤.25を既存helperの定義のまま用いる。後者の分母はWelch差区間半幅ではない。

診断警告→`UNRESOLVED_DISTRIBUTION_DIAGNOSTIC`、精度未達→`UNRESOLVED_PRECISION_TARGET`、その後で差区間の符号による名目低費用判定を行う。符号未確定は同等性ではない。分散0はunresolved。欠測・不正値・correctness failureは例外STOPとし、価格0への置換・除外・cohort選択を行わない。128件完了まで推論せず、一度だけ主比較を保存する。未達でもN/confidence/閾値を変更せずSTOPする。

## 4. 標本数の計画案と限界

公開済みの各kernel分散s²とWのみから、価格分散の計画値`v_a=W_a² s_a²`を作る。設計比較には`2 sqrt(v_C/n_C+v_E/n_E)/保存ENTRY総RZ点値`を使った。係数2は名目95%付近の**計画用近似係数**であり、t臨界値、計算済みCI、新しい検定、保証付き誤差ではない。旧標本行や旧観測min/maxを用いず、分散を再推定していない。6.10%という旧半幅だけを比例換算した設計でもない。

| 計画n(CDF0),n(ENTRY) | 計画比率（CIではない） | 新native件数 | 用途 |
|---|---:|---:|---|
| 64,64 | 6.12547% | 128 | 精度余裕が不足する計画参考 |
| 96,96 | 5.00143% | 192 | 5%付近で余裕がない |
| **128,128** | **4.33136%** | **256** | 主案、各4×32バッチ |
| 128,旧64 | 4.53686% | 128 | 軽量案の計画参考のみ |

128は既存32件バッチの倍数で、比較した同数候補の中で5%未満に余裕を持つ最初の値である。新データの分散・分母・裾は変動するため、128で必ず5%達成するとは主張しない。診断のPASSもcoverage/IID/未観測の裾の保証ではない。精度達成確率や正式powerは算定していない。以上を根拠に、**未承認proposal内で128/armを固定する案を推奨**する。科学政策の採択は別途必要である。

異なるNも合理性がある。標本単価cに対する理想的な配分比は`sqrt(v_C c_E/(v_E c_C))`で、同単価を仮定した公開分散だけの比は約3.049である。ただし実native単価は未計測で、r比4を実行時間比として扱わない。主案は既存の同数NのWelch APIをそのまま使えること、両armを対称に独立追試すること、計画上の精度余裕を優先した。同コスト最適配分とは主張しない。異数N案を選ぶ場合は実行前に新たな固定設計が必要である。

## 5. 主案と軽量案

| 観点 | 主案：双方fresh128 | 軽量案：CDF0 fresh128＋保存ENTRY64 |
|---|---|---|
| 新native/correctness処理 | 256件、うちENTRY128件が追加 | 128件、ENTRY側の新回路処理なし |
| 来歴 | 新namespace同士、旧値はpoolしない | 過去のE2推定と不確かさを再使用 |
| 独立追試としての情報 | 両側価格の再現性と差の新比較 | CDF0側だけの追加検証 |
| 旧CDF0比較との依存 | 候補・r・設計は履歴に条件付くが価格標本は新規 | 同じENTRY標本を再利用するため旧比較と相関する |
| 実装 | 本adapterの主経路 | 説明だけ。実行経路は採用しない |

軽量案でもENTRYのSEを0にしてはならず、保存点値を既知の定数として扱う検定は不適切。新128対旧64の正式統計は今回算定していない。計算負担を減らせる一方、今回の目的である新cohort同士の確認には主案を推奨する。どちらも今回の指示だけでは採択・実行しない。

## 6. 再利用と最小adapter

新規ファイルは`src/trottertracks/c_cdf0_entry_independent_price.py`、既定が受動preflightのrunner、人工テスト、proposal、技術SOURCEとtechnical.jsonである。旧コードは一切変更しない。

- CDF0のreference schema接続は`c_cdf0_relocated.connect`、ENTRYは`c_h4_relocated_downstream.prepare_candidate`。保存J/U・original corrected κ/scalar・保存sector reference・候補固有groundを結び付ける。original9因子、同じH、groundの資格を再検査する。保存groundがなければ既存fallback solveへ入る前に拒否する。
- `configure`、distribution/tail/normalization gate、`native_context`で保存候補・固定r・同一lawのfinite correctness contextを準備する。新しいDF fit/norm、平均PF/RTE精度評価、RPE shot計算ではない。新trajectoryの独立reference行列は既存correctnessに必要な処理として実行proposalに含む。
- `c_pr_integrated_performance.native`→`c_native_process.native`→`c_native_sample.run_sample`を各32件に適用する。4バッチへ拡張するadapterだけを追加し、sampler/build/compiler/evaluatorは再実装しない。旧`native_cohort`はfinalを2バッチに固定しているため128には直接使わない。handoff設定を渡さず、旧価格のrestore経路は空となる。
- 各sampleのraw/reference、compiled/raw/reference、full512全入力出力、control-relative/scalar phase、actual raw/compiled inverseとreturn、X/Y複素信号を維持する。QPY4本、reference NPZ、events/gate/metricsを保存する。故障標本を価格0として扱わず、最初のfailureでSTOP・部分成果を保持する。
- `b2.calibration`と`c_near_r32_statistics.arm_summary/primary`を新規128行だけへ適用する。既存helperの`PR`キーはこのadapterではCDF0を意味すると明示する。D/RPEの再計算やsecondary CIを呼ばない。

総資源は保存bindingの`W×新kernel平均+保存shell`。Nやrを追加乗算しない。rはkernelに内包され、Wは保存shotsの`Σ_m 2 shots_m 2^m`。native価格標本は古典的価格校正であり、量子shot数は増えない。入力summary・affine bindingのidentity/r/サンプル数/適格性を受動照合してから再利用する。

## 7. 新namespace・SOURCE・preflight

新master=2026110803、axis=`INDEPENDENT_PRICE_FINAL_ONLY`、cohort=`cdf0_entry_independent_price_v1:<CDF0|ENTRY>:r<64|16>:final:batch<00..03>`。SOURCEは今回の新manifest digest、role baseline、候補identityは保存JU SHA＋再配置version＋LD2。sample0..31のseed hashを既存関数で導出する。256個の重複と旧登録1024個とのintersectionをチェックした。旧CDF0について未実行rの予約namespaceも含む保守的検査であり、旧1024個すべてを実行したという意味ではない。preflightでPRNG drawは0。

[proposal](research_c_cdf0_entry_independent_price_proposal_20261108.json)は未承認。新SOURCEは旧112ファイル＋新adapter/runner2ファイル。旧契約2本のdigestをanchorにcompiler/target/packages/物理条件/κ/shell/旧seed domainを照合する。入力52本すべてをbyte hashで検査した。ユーザー提供のE2 ground原本も旧raw索引の期待hashと一致する。NPZ/QPYのdecode、旧価格の統計再計算はしない。旧SOURCE、STOP、CSVは保全のhash対象であり新推論の標本poolにはしない。

```bash
python3 scripts/run_research_c_cdf0_entry_independent_price.py
```

既定の受動検査は`PASSIVE_CHECKS_PASS`、execution_ready=true、execution_permission=false、science calls0。`execute_science=False`は数値import前に拒否する。入力が再び欠けた場合にも科学call前に拒否する回帰テストを保持した。`--execute`はこの具体的proposalへの別途科学実行許可を受けるまで使用しない。実行許可の証明書・grant発行は追加しない。

## 8. 保存groundの取得・同一性確認（追記）

初回準備commit `2dba3fb09800e373ccb94027a2a9da1d2c540716`の時点では、旧handoff rawの`E2/A/ground_candidate.npz`は公開raw索引のhashだけがあり、原本不足でCASE Cだった。この履歴と初回preflightはgit履歴およびtechnical.jsonのprevious_stateに保持する。

ユーザーが`/home/AbeHiromu/projects/partially-randomized-trotter/ground_candidate.npz`へ追加した1106 byteの原本を受け取り、旧`publication/result_files.json.gz`に記載された期待SHA256と一致することを確認した。

- SHA256：`6843e56b955a131583950c585287a41f61c08321b48618c1525b83d2361471b4`
- 保存先：`artifacts/research_c_independent_price_precision/2026-11-08/inputs/E2_ground_candidate.npz`
- 処理：原本を変更せず同じbyte列をコピー。NPZ配列のdecode、新ground solve、丸め直し、数値的な再解析は0。

入力52本すべて・新SOURCE114ファイル・旧STOPの受動照合がPASSし、不足入力は0となった。科学adapter/runner/SOURCE digest・固定128/arm・r64/r16・2405-call scopeを変更していない。groundの数値gateは将来の許可済み科学実行時に既存処理が行う。今回のbyte照合を新しいground科学検証とは扱わない。

## 9. 実行scope・callback予算・保存

過去のcall上限を転用せず、実APIから導いた上限は**2405成功callback相当**。callback内部のprimitive算術、2本のactual inverse build、4回のfull512 decodeは親gate callbackに含まれ、追加の独立callbackとして重複計上しない。

| 種類 | 回数 |
|---|---:|
| 256標本×9 callback（sample/build/reference/raw gate/compile/compiled gate/inverse gate/XY/metrics） | 2304 |
| armごとの保存candidate/reference・再配置・sector/9因子/H/ground・law/normalization・full CAR/context・9 Gaussian/4 relative | 74 |
| 各4バッチのcalibration＋全128の検査/calibration/arm summary | 14 |
| 共有κ load、shell load6/gate3、CAR context | 11 |
| 単一primary comparison・result summary | 2 |
| 合計 | 2405 |

native256 build/compile、actual inverse512本、native full512 decode1024回、shell decode6回。新selection/fit/DF norm/PF・RTE科学精度評価/RPE shots再計算/量子shot drawは0。バッチごとに既存spawn/mmap schedulerが実際のaffinity/quota/空きメモリと残標本数からworker数を選ぶ。builder/CAR/shellを再利用する。CPU/RAM/workerへ新しい恣意的上限を置かず、共有環境・他jobは変更しない。

fresh outputは`artifacts/research_c_independent_price_precision/2026-11-08/science/run01`のみ。既存dirがあれば拒否し、retry/resume0。将来実行では契約/SOURCE、各armの新QPY/reference/events/gates/metrics、calibration/summary、主比較、call journal、failureとSTOPを保持し、rawを取得可能な形で索引・hashとともに公開する。科学結果は旧runと別の正式結果とし、旧CSV/summary/STOPへ書き戻さない。

## 10. 人工・回帰検証と今回の境界

**新43＋既存62＝105件PASS**（失敗0、RPE/Dの不要な回帰2件は選択除外）。テスト結果・実行環境・受動preflightはtechnical.jsonを正とする。新テストは人工128行とmock nativeで候補/r/version/SOURCE/旧selection・pool/seed・ID衝突/N変更を拒否し、W/rの二重乗算なし・shell1回・旧固定N helper・failure時の部分保存/STOP/再開拒否を検査した。人工rank8の保存ground接続ではfallback solveを禁止し、両armの実setup APIからcallback内訳を照合した。保存H4は数値decodeしない。

追記の入力復旧後は、存在する原本の受動検査と、欠落を模擬した場合の科学開始前拒否を検査した。関連26件PASS、他18件は今回の変更に無関係なため選択除外した。前回の全人工・回帰105件PASSはその時点の証拠として保持し、今回は全件を再実行したとは扱わない。

既存のCDF0 schema/接続、固定N価格統計、再配置full512/phase/actual inverse/XY、process backendの人工回帰を再利用した。これらのローカルPASSは実H4新結果やimmutable CIとは扱わない。CPU専用Aer0.15.1とthreadpoolctlは以前の専用/tmp overlayを使い、共有venvを変更しない。

今回、新実H4 build/compile/native標本、保存価格の新しい統計再解析・CI・検定、新DF fit/PF/RTE/RPE/shots、旧run retry/resume、P0政策変更、全word保証研究、重要GPT科学レビューは実施していない。公開要約による計画proxy以外の新科学値はない。旧SOURCE/input/raw/結果/grant/ledger/marker/STOPを保全した。

**残るのは、具体的な128/arm設計・SOURCE・2405-call scopeに対する科学政策の採択と、新科学実行の明示許可である。** 本ターンでは承認を求めて開始せず、成果物を公開・照合してmandatory STOPする。
