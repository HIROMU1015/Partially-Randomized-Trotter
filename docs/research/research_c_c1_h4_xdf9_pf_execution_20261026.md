# H4 保存順XDF9：表現・own-ground／alias・決定論S2の限定科学実行

2026-10-26 JST。**95/95 logical units COMPLETE、correctness FAIL／inflight／retry／resume0。表現・ground／aliasと26/26 PFセルがPASSし、s=1／4とも`POTENTIALLY_FEASIBLE`。** 最大PF誤差は各0.06843041535127227／0.004277093283650178（m12）。最終CA・native・合成誤差・shots・総費用は未評価。公開照合後mandatory STOP。

- 科学SOURCE：`9a4cda44a45675c2518b2252a98e78eeeb320942`、tree `35e4bbeff0d095c628a8c116f881c7ff734eaaf7`。科学実行から結果公開のための文書更新まで、固定library／runner／契約は変更していない。
- Branch：`codex/research-c-c1-h4-xdf9-pf-execution-20261026`、独立worktree `/tmp/prt-c1-h4-xdf9-pf-execution-20261026`。
- [固定95-call契約](research_c_c1_h4_xdf9_pf_contract_unapproved_20261026.json)、scope `C1_H4_SAVED_XDF9_REPRESENTATION_GROUND_PF_V1`。
- 契約SHA256：`189f4fa9f8a9d34d7a268857e78ec7459fba4e0ec05e99679fa2b30934535310`。
- SOURCE manifest SHA256：`a3fd7d17f7059240ba70d003b4dee7ad8e3f36358cf8bd64cbbed03fb1bc373d`。
- [機械可読summary](../../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/execution_summary.json)／[全26セルCSV](../../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/pf_rounds.csv)／[raw45・SOURCE・journal・STOP](../../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/run01/)。
- [preflight](../../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/execution_preflight.json)／[postflight・原本hash](../../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/execution_postflight.json)／[既存人工検証・再利用](research_c_c1_h4_xdf9_preparation_20261026.md)。

## 固定条件と実行境界

直鎖H4・隣接1.00 Å・STO-3G、中性singlet、Nα=Nβ=2／sector36。保存OpenFermion1.6.1 eigh由来λ/Gの元順prefix9だけを使用。通常number表現、元g補正κ・核scalar一回・spin／bit／sector順・frame順を維持した。全frame ordered symmetric S2、s=1主対象／4対照、δ=.1、m=0..12、h=.1/s、coherent反復n=s·2^m、PR split非適用。own-XDF9 sector ground oracle、状態準備費用除外。CA=.00159 Ha、DF reserveCA/8=.00019875 Ha、β_total=π/6、β_stat=π/12を変更していない。

元P-H4 SHA256は`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。公開commit `86b8b002704aced8ca239414080f53793de81658`の固定packetと同じbytesをtask内`.external/P-H4.npz`で使用した。保存DF値はSOURCE `f7bc730d12ad704cdb72cc8a82d2bd973d01b2e5`の[prefix9](../../artifacts/research_c_c1_h4_xdf_prefix_budget/2026-10-26/run01/prefix_9.json)と、その元SOURCE／消費済みSTOPを照合して引用した。新しいphysical HamiltonianやDF normは生成・評価していない。

最初の添付指示に従ってread-only preflightだけを完了した後、ユーザーから本scopeの科学実行への明示許可を受け、一度だけfresh run01を実行した。歴史的な`unapproved`名・metadataは維持し、承認証明書・原文保存・新grantは作成していない。実行記録は以下。**run01は消費済みで、このコマンドを再起動してはならない。**

```bash
MPLCONFIGDIR="$PWD/.runtime/matplotlib" TMPDIR="$PWD/.runtime/tmp" \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python \
  scripts/run_research_c_c1_h4_xdf9_pf.py \
  --contract docs/research/research_c_c1_h4_xdf9_pf_contract_unapproved_20261026.json \
  --input .external/P-H4.npz --execute \
  --output artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/run01
```

## 全correctness gate

| Gate | 結果・数値 |
|---|---|
| SOURCE・契約・input・依存版 | PASS。SOURCE24・保存JSON3・原packetを実行前後照合 |
| 保存DF予算 | PASS。保存norm2.0799049651854478e-7＋guard1.4808500737431315e-9、保存sum2.0947134659228792e-7 Haを引用。DF norm再計算0 |
| κ・scalar | PASS。元g由来κ max-entry差2.7755575615628914e-17、核scalar2.29310124732をbaseだけに一回 |
| frame／spin／basis／finite／Hermiticity | PASS。元順F0..F8、exact spin lift、登録sector36とOF→Qiskit permutation。非有限値・不正な射影なし |
| raw CAR対独立一般K | base＋9 frameの10/10 PASS。最大成分差8.43769498715119e-15、許容差1e-10、rtol0 |
| raw合計対保存XDF9 matvec target | PASS。max-entry1.7763568394002505e-15 |
| K合計対同target | PASS。max-entry9.381384558082573e-15 |
| own-ground residual | PASS。1.907144528367971e-15 Ha ≤1e-10 |
| normalization／sector | PASS。差1.1102230246251565e-15 ≤1e-12、sector36 |
| H4 alias | PASS。own-ground energy−2.1663875300259736 Haは[-10,10]内。登録base branch margin1.6179938779914944 rad >0 |
| S2 s1／s4 | 26/26セルが固定sin(π/12)以下 |

10個の完全one-body／square因子はexact spin liftを満たし、spin sectorを保存する。この検証は個々のGaussian gateのsector射影やfull controlled回路の検証ではない。raw CARのsquareと独立I/Z/ZZ係数展開、既存保存λ/G matvecの3経路をsector36で照合した。DF近似誤差とadapter同値性は別判定。groundはXDF9固有のeighで、XDF8／10の状態を流用せず、physical ground overlapはnull／未測定である。

## 全26 PFセル

誤差は無次元operator 2-norm `||W_s^(s·2^m)−exp(−i t_m H_XDF9)||₂`、t_m=.1·2^m。既存`PF_rounds`／`ordered_schedule`／expm／保存した固有対／反復二乗法を再利用した。登録十分条件はerror≤sin(π/12)=0.25881904510252074。数値条件・allowance・閾値の事後変更はない。以下の数値は原本の表示丸めで、CSV／JSONに全保存精度を残した。

| m | t_m | s1 PF誤差 | s4 PF誤差 | 判定（両s） |
|---:|---:|---:|---:|---|
| 0 | 0.1 | 6.245915180524e-05 | 3.901987916538e-06 | PASS |
| 1 | 0.2 | 1.245816334350e-04 | 7.782952674036e-06 | PASS |
| 2 | 0.4 | 2.464970589311e-04 | 1.539941474679e-05 | PASS |
| 3 | 0.8 | 4.725088766988e-04 | 2.951960578741e-05 | PASS |
| 4 | 1.6 | 8.051362288787e-04 | 5.030332732120e-05 | PASS |
| 5 | 3.2 | 9.681454974167e-04 | 6.049236909484e-05 | PASS |
| 6 | 6.4 | 1.288140241312e-03 | 8.049322066110e-05 | PASS |
| 7 | 12.8 | 2.412885948786e-03 | 1.507851133326e-04 | PASS |
| 8 | 25.6 | 4.304084111693e-03 | 2.689646975685e-04 | PASS |
| 9 | 51.2 | 8.593698040855e-03 | 5.370290954790e-04 | PASS |
| 10 | 102.4 | 1.717058253812e-02 | 1.073017686459e-03 | PASS |
| 11 | 204.8 | 3.427960536751e-02 | 2.142265082894e-03 | PASS |
| 12 | 409.6 | 6.843041535127e-02 | 4.277093283650e-03 | PASS |

| 候補 | 最大PF誤差（m12） | 登録十分条件 | 分類 |
|---|---:|---|---|
| XDF9 S2 s1 | .06843041535127227 | 13/13 PASS | POTENTIALLY_FEASIBLE |
| XDF9 S2 s4 | .004277093283650178 | 13/13 PASS | POTENTIALLY_FEASIBLE |

NOT_MET／UNRESOLVEDセルは0。決定論なので補正RTE誤差0・attenuation1は構造上の条件であり、未知の合成誤差・law誤差を0と認定したものではない。最終CA達成はnull、strict rounding enclosureも未資格。

## 既存XDF8と照合するための保存証拠

同じ元input・元順prefix・sector36・S2・s・13round窓・own-target比較の既報最大値を並べる。XDF8の値は[既存報告](research_c_c1_h4_xdf8_pf_execution_20261026.md)／[保存26行CSV](../../artifacts/research_c_c1_h4_xdf8_pf/2026-10-26/pf_rounds.csv)の引用であり、旧operator・ground・PFを再計算／再解析していない。

| 対象 | DF norm＋guard [Ha]（保存値） | PF最大 s1 | PF最大 s4 | PF科学SOURCE |
|---|---:|---:|---:|---|
| XDF8 | 9.533901455091894e-5 | .06842795072754271 | .004276939210418963 | `328f31007ee8069d1f7ba24f91029a3748b59109` |
| XDF9 | 2.0947134659228792e-7 | .06843041535127227 | .004277093283650178 | `9a4cda44a45675c2518b2252a98e78eeeb320942` |

今回のXDF9最大PF誤差は保存XDF8よりわずかに大きい。DF近似精度の改善だけからPF誤差・必要shots・総資源の改善を推定しない。両者のPF targetとown-groundも異なる。rankを1増やしたnative費用と最終shotsの交換関係は、まだ数値評価していない。

## 科学call・資源・保全

| call | 実績／上限 |
|---|---:|
| saved_input_load | 1/1 |
| saved_DF_gate | 1/1 |
| sector_context | 1/1 |
| input_identity | 1/1 |
| xdf_prefix_model | 1/1 |
| xdf_matvec | 36/36 |
| raw_factor | 10/10 |
| K_model | 1/1 |
| representation_gate | 10/10 |
| sector_sum_gate | 1/1 |
| ground_eigenpair | 1/1 |
| ground_alias | 1/1 |
| sector_kernel_reference | 2/2 |
| PF_round_assessment | 26/26 |
| candidate_summary | 2/2 |
| **合計** | **95/95 COMPLETE** |

attempt95／COMPLETE95／FAIL0／inflight0をjournalで照合。fit／optimizer／新DF分解／DF norm／CDF8-XDF8-XDF10新科学評価／回路build-compile／合成／CA／shots／総費用／retry／resumeはすべて0。固定9個のGのinner diagonalizationは登録K_modelの表現変換に含み、新しいouter DF分解ではない。CSV・最大値・要約の整形は保存した今回の結果の報告処理であり、norm／PF／shots solverを追加実行していない。

開始前affinity32 CPU、空きRAM約52.7 GiB、CPU／memory pressure avg10=0を確認。閲覧可能なcgroupのmemory.max／highはmaxで、CPU quota数値は取得できた範囲に現れず、無制限と断定していない。既存resource_planは独立s task数に応じworker2を選択し、job内BLAS1でoversubscriptionを回避した。共有設定・venv・cache・他ユーザーjobを変更せず、恣意的なCPU／RAM／wall capも追加していない。

runner計測wall1.739476810209453秒、process CPU8.372112658秒、peak RSS365880 KiB。既存read-only Python3.11.1／NumPy1.26.4／SciPy1.14.1ほか固定8 packageを使用。既存の同SOURCE人工78 tests＋29 subtests PASSを再利用し、SOURCE変更がないため人工suiteを重複実行していない。

原本45 files／153,943 bytes（26セル、2 kernel、generators、ground／spectrum、SOURCE／契約／journal／STOP等）を保全した。protected SOURCE24＋saved JSON3＝27件、原packet、契約／manifestの実行前後hashが一致。旧H2／H4／H6のSOURCE・raw・grant・ledger・marker・STOPと、旧約27%／新約12%の費用結果は変更していない。ローカル科学実行の公開証拠であり、外部再現・CI結果とは区別する。

## 次に必要な担当・判断・mandatory STOP

XDF9 s1／4はともに、native再資格付けを提案できるPF候補に残る。ただし今回取得したのはsector表現・own-ground／aliasと登録PF十分条件までである。native正しさ・controlled相対位相／actual adjoint／X-Y、P0前後、全13round合成誤差、厳密enclosure、最終CA、shots、総費用・費用順位は未資格／未評価。

**ユーザー／研究C**が、XDF9のnative・合成誤差再資格付けを次に採るか判断する。必要なら**Codex**が既存precision Gaussian／stable-phase P0／shared・fenced serial policyを使う最小技術準備を行い、固定条件と新契約を提示する。実build／compile・数値評価にはその固定範囲への別の明示科学許可が必要。現在のCDF8証拠はtarget・入力・PF・precision／P0・共有／境界・位相予算・horizon等が同一の場合に限り再利用を判断し、XDF8の回路・shotsをXDF9へ移植しない。

今回のscopeに未完了計算・技術blockerはない。結果・SOURCE・input・STOPをcommit／pushしremote照合後、**mandatory STOP**。native／合成上界・全operator・CA・shots・総費用・詳細科学レビューへ自動進行しない。
