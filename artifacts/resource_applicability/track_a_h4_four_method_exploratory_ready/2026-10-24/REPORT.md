# H4 Full S2 / Full Yoshida4 / PR S2 / PR Yoshida4：探索的総費用比較の実行準備

2026-10-24 JST。専用branch `codex/h4-four-method-exploratory-ready-20261024`。
**技術準備完了、H4科学実行承認待ち。新しいH4 u/N/G・順位の評価0、compile/sampling/simulation0、grant発行0。**

## 固定候補と再利用

| 方法 | 元候補ID | prefix | q | 元trajectory / 軸 |
|---|---|---:|---:|---:|
| Full S2 | H4_FULL_10_8_2nd | 10 | 8 | 1 / 2 |
| Full Yoshida4 | H4_FULL_10_4_4th | 10 | 4 | 1 / 2 |
| PR S2 | M5_MIN_72b305020499dd68 | 3 | 8 | 48 / 96 |
| PR Yoshida4 | M5_MIN_ab57ad5bd40dc583 | 3 | 4 | 33 / 66 |

選定は既存登録候補からの事後選択。指示されたS2/q8・Yoshida4/q4のPR候補は元登録範囲でprefix3に一意に対応し、Fullは同formula/qのfull-prefix10を対応させた。m5_best、DF12、別独立確認cohortを混入しない。新しいGを見て候補を変更していない。

Linear H4、隣接1.0 Å、STO-3G、tol-only DF1e-8/rank10、T=.8、ε_signal=.0005、α_axis=.025、P=0、system8/ancilla8、usual signed bias、symmetric-directional control、full cosine/sine Hadamard wrapper。PRはprefix3/K4/ABS_TIME_PER_OUTER32。費用は状態準備を除くcompiled RZ/軸で、T/Toffoliや全エネルギー推定費用ではない。

Fullの共通native保存単価はS2各62571 RZ、Yoshida4各93799 RZであり、両方とも実測結果SHAが共通PF inventoryへ結合する。単価不足による追加Full compileは不要。PRは元48/33 trajectoryを全数保持し、原81 receiptのSHA・task・SOURCE/plan・Hamiltonian/state/preparation・compiler/native・両軸・8 correctness PASS fieldを照合した。過去の受入済みpaired-priceファイルとも整数の各要素が一致し、統計を再計算していない。

Fullの保存signal/reference/bias/shot/allowanceは採択済みf75a7c25... dataset、PRは固定8613b3ec... 結果とその原private入力から回収した。Hamiltonian/state/snapshotのbindingは4候補で共通。[保存データ再利用表](saved_data_reuse_table.json)。不足入力なし。保存N/G等があることと、今回の4候補を一つの表で会計・順位評価済みであることを区別する。新しいH4表・順位はまだ出していない。

## referenceとnormalizationの保存差

Full保存referenceは(real,imag)=(-0.16160185891555842, 0.9868560377253799)、PRは(-0.16160185891555753, 0.9868560377253801)。同じHamiltonian/state/input bindingに対応する受入済みの別binary64値であり、完全byte一致を要求しない。各系列のbias・referenceをそのまま使い、0誤差保証や一致証明へ格上げしない。これは実行承認時に明示する限界である。

PR S2の原cost receiptのlog_Bには0.000850509400794277、保存signalには0.0008505094007942772がある。元の受入済み解析は256×binary64 epsilonのclose規則を使っていた。今回は原receiptと受入済み解析SHAへ結合し、両hex値をprivate入力に保持した。shotには保存signal側の値を維持し、normalizationを新規計算したり原receiptを書き換えたりしていない。

H4のu=1e-9および各候補の保存log_B_marginを継承する。H2で新しく評価したdiagnostics由来uへ置き換えない。保存H4 reference精度を条件とし、reference_absolute_errorはnullを維持する。

## SOURCE・再利用とexact予算

- 新SOURCE：`4cbbc1b6b49bd64e765ccc596ec5016a1a4d5503`。
- 契約canonical SHA256：`f2fe519082fd0994de01660ad14810d3f2d2ef7bbd64c470f78eb3183931a408`。
- 固定入力manifest file SHA256：`465d16b6f00c606673dac547bba732334933a1ec15583e616709a575988a1d6a`。
- H2数値SOURCE：`aca689060d302b4e1f37d32d7075c9266b637f56`、H2結果：`9e5ea6b235d7beecc02651b1393b561948e85789`。
- provider：`c31ef424175c02379eafebe8fc6c7548c1d9aabe`。既存H4 shot SOURCEは8613b3ec... の `d0_frozen_shot_rule_v1.py`。

H2の `Backend.account`、`Backend.rank`、探索版 `Backend.statistics` のcode objectをそのまま使い、関数のmetadata namespaceだけをH4へ結合した。元SOURCEは編集しない。H4のsaved allowance取得、候補/入力検証、明示承認とrun管理を追加した。[再利用検査](numeric_reuse_evidence.json)ではH4で使った凍結shot関数と今回providerの `u_aware_shots`・`axis_headroom` はAST一致。`resource_row`・`metric_summary` とtrajectoryのpaired統計もH2と同じ実装。新しい数値算法や区間公式は導入しない。

予算はH4の実経路に基づき、**bootstrap1 / bundle_load1 / allowance4（保存値取得）/ accounting4 / u_aware_shots4 / axis_headroom8 / cost_statistics4 / resource_row4 / metric_summary8 / ranking1 =39 ledger event**。empirical_allowance評価は0。setup/prepare/primitive/reference/signal/sampling/build/lowering/compile/transpile/simulationはすべて0。H2の43を流用していない。

shot整数は保存H4の各軸Nと一致することを受入条件にする。未適格行のN/Gはnullに残す。費用は既存accountの演算順を維持し、statistics側のmean-then-multiplyとの差は元実装どおり記録する。ランキングでは4点の順序とFull/PRの各4次化方向を別に保存し、逆転の有無と4方式最良を区別する。原因帰属はnull、有限SEは母平均保証ではない。名目区間は新規導入せずnull。

[契約案](execution_contract_DRAFT.json)、[入力索引](fixed_input_manifest.json)、[runtime封印](runtime_seal.json)、[出力schema](output_schema_DRAFT.json)、[H2/H4条件対応](H2_H4_condition_map.json)。

## 人工テスト・preflight・保全

人工11 test method（複数の混入・欠測・cohort・axis・unit・未承認拒否ケースを含む）がPASS。封印Python3.11 -I/-S/-Bでも人工fixtureだけを実行し、39件のexact ledger・loaded closureをPASSした。実H4数値を人工fixtureへ流用していない。[test結果](test_results.json)・[封印runtime人工検査](isolated_artificial_smoke.json)。作成者Codex自身の技術検査であり、独立科学レビューではない。

実H4入力を使った処理はJSON/SHA/元field/受入receiptの技術検査だけ。[metadata preflight](metadata_preflight.json)はPASSし、SOURCE/input/runtime計1065 fileを封印した。承認前のscientific kernel呼出し0、有効grant0、科学run_root未作成。旧SOURCE・grant・ledger・marker・STOPを変更せず、別worktreeの通常進行を拒否条件にしていない。[終了保全監査](preservation_audit.json)。

CPU/RAM/workerに人工上限を設けない。今回の小さな39 call算術は依存順と逐次ledgerを保つ単一processで十分で、重いsimulation/compileはない。静的SHA確認は既知の入力領域へ限定する。共有OS/scheduler/venv/他ユーザーprocessに変更なし。[資源観測](resources.json)。

必須参照6 commitはローカルGitで取得可能。3eca... の独立確認、2d4... のS2別系列、6daf... の統合索引は照合範囲を区別し、今回へpoolしない。[参照可用性](reference_availability.json)。指定されたローカル共通指示の可視版には§10.3見出しがなく、今回のユーザー指示に明記された科学再現性優先・過剰一致検査を避ける方針を適用した。原文は変更していない。取り消されたH4非DFレビューの依頼・資料は今回の根拠に使用していない。

## 一回の科学実行承認ゲート

ユーザー：上記4候補、SOURCE/input/契約SHA、39 event、保存H4 reference/allowance条件、有限cohort・事後選択の限界を確認し、H4 one-shot実行を明示承認する。

承認後のCodex：新UNUSED grantを新H4 namespaceに発行し、直前preflight後に一回だけ会計・統計・順位を実行。result/ledger/CONSUMED/completion/STOPを保存し、9 result fileのSHA、exact39消費、source/input/runtime不変性、native exit/reapを確認する。失敗・整数不一致・重大な入力不一致ではpartial/消費/STOPを保持して停止し、同grant再試行や自動追加実行はしない。修正して再科学実行する場合は新SOURCE/契約/grantの承認境界を維持する。

H2/H4を並べられる固定資料を結果取得後に整える。GPT科学レビュー自体は開始しない。旧strict COMPARE・旧正式契約はBLOCKEDのまま、論文主表・Figure2・採択済み共通PF資料は更新しない。private原本・seed・grant・raw・内部絶対pathを公開しない。

**今回の終了状態：H4科学実行承認待ち。mandatory STOP。**
