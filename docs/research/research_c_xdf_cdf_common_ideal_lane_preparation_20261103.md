# 研究C：保存XDF8/CDF8の共通ideal lane接続・61-call科学実行準備

2026-11-03 JST。HIROMU1015/Partially-Randomized-Trotter。

**技術的GO。実行可能なSOURCE・保存JSON入力・未承認61-call契約を固定した。科学runは未実行。** 保存B2 CDF8/B3 XDF8の候補固有DF/PF/ground/native証拠を再利用し、最新PRの`ideal_exact_native`に対応する反実的exact-native laneへ接続する。新しい実H4のshots・W・総RZ・CIを算出していない。

[未承認科学契約](research_c_xdf_cdf_common_ideal_lane_contract_unapproved_20261103.json) ／ [adapter](../../src/trottertracks/c_xdf_cdf_common_ideal_lane.py) ／ [entry](../../scripts/run_research_c_xdf_cdf_common_ideal_lane.py) ／ [人工test](../../tests/management/test_research_c_xdf_cdf_common_ideal_lane.py) ／ [技術SOURCE](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/technical/SOURCE.json) ／ [passive preflight](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/technical/passive_preflight.json)。今回の技術作業完了後、公開照合してmandatory STOP。

## 1. 比較する問いと今回の境界

同じupdate064 J/U上のPR対強Dと、別DF表現を許すPR対保存XDF/CDFを区別する。[固定監査A][AUDIT]で確認済みの物理問題、input、compiler、精度・会計政策を再利用し、その静的監査を一から繰り返していない。

直鎖H4・隣接1.00 Å/STO-3G、Nα=Nβ=2/sector36、rank8、全frame ordered S2/s1/δ=h=.1/m0..12、own-DF ground oracle・準備費用除外、P0/fenced serialの**保存数値合成lane**の総RZはCDF8=102056444569、XDF8=100267421030だった。[旧評価][OLD]　最新PR064はnumber CDF8/元順LD2・partial S2/s1/r32/K0・同時間窓のideal条件付き会計で、独立価格256件の総RZ点推定100330684089.25を保存している。[価格結果][PRICE]

この三値を同一証拠laneで取得済みとせず、差・比・新CI・検定はここで計算しない。本契約の科学対象は**保存CDF8/XDF8の二候補だけ**である。未来のPR/ENTRY新endpointやAの途中結果は入力に含めない。共通laneが計算できた後のPRとの統計比較、同等性・非劣性認定は別の問いであり、この61-callに含まれない。

## 2. 固定commit・候補・科学SOURCE

起点は指定監査commit `8375433869999c2e10501b1983c11b2b22c5d423`。共通Evidence Map `ba0f280c8be9e7038c5fa6b5fd390a235b113833`を含み、必要な保存metadataとaccuracy/serialコードが利用できる。新しい下流やPR価格コードを実行するためのmergeは不要だった。

| 対象 | 固定identity・SOURCEと役割 |
|---|---|
| 旧共通CA/資源 | 結果commit `54ceff9981d9bc53210db97c743a7d1cf694f362`、科学SOURCE `1aeedd96178a3f50a15bc2701c3c332f4f391b64`。旧86-call数値合成laneを保存する。 |
| B2 CDF8 | `B2_CDF_relative`。元fit SOURCE `da2edb465d5ae6432439fbd1df8e57512ee59c11`、export SHA256 `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08`。 |
| CDF DF/物理表現 | 評価SOURCE `bec41ac7dc7e034bd664817acfdcfed858ddadd0`。保存数値norm＋guardを旧prepared/identityから再利用し、新normを求めない。 |
| CDF PF/ground | SOURCE `6ea5c6a9c10cc4b5411039a0d61a9f19acc4f34f`、保存feasibilityのCDF D/s1、m0..12。 |
| CDF native | SOURCE `880006396326a4d46f4a4030564279e92fadb9cf`、結果`e0c89f611eac92c939edde67890f5cc4391d098e`。B2 routeの固定価格。 |
| B3 XDF8 | `B3_XDF_spin_relative`。元OpenFermion1.6.1 eighのsigned λ/G、rank10 snapshotの元順prefix8。CDFのJ/U exportと同じidentityにはしない。 |
| XDF DF | SOURCE `f7bc730d12ad704cdb72cc8a82d2bd973d01b2e5`のprefix8保存誤差。 |
| XDF PF/ground | SOURCE `328f31007ee8069d1f7ba24f91029a3748b59109`、結果`1f318bb9fc4e412bda647b424278deb80dbc5790`。保存XDF8自身のground/aliasとPF13を使用。 |
| XDF native | SOURCE `b40fce5c62c9fff97f68073cbc02182e4bbf2b9b`、結果`866d3ba327abd5ca38af5d7a9347f6c91294958f`。Spin対応＋relativeのB3。 |
| 最新PR参照 | 下流結果`cca6b487c8e1ed3362f8825e95c937ebbcc87755`/SOURCE`928fda0a6df260b66eba1bbea63b51f2734e7b54`、価格結果`3718d369b3db515b2dd13b2cdbb6c49d57880121`/SOURCE`9837eac59b94113bf057fcc311a27b35e60343f5`。政策の照合・引用であり、今回の科学入力ではない。 |

元H4 inputは`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`、生成SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`、公開`86b8b002704aced8ca239414080f53793de81658`。[旧契約][OLDC]の来歴を保持する。今回の実行はそのNPZを必要とせず、検証済みJSON evidenceだけを読む。

## 3. 共通ideal laneの正確な意味

PRの固定SOURCEでは`c_rte_r_retuning.assess_lanes`→`c_pr_aware_df_stage_c.qualification/assess_round/energy`が共通accuracyへ接続する。[下流SOURCE][NEWSOURCE]でこの呼出し経路、起点内のStage Cのscenario定義を確認した。`ideal_exact_native`はnative synthesisと追加law誤差を0と**仮定**し、PF・補正RTE・attenuation・DF誤差は残す。全lawに対する実装誤差0の実測ではなく、finite-bit価格からideal-law期待価格への移送資格も得ていない。

決定論CDF/XDFはA=1、RTEなし、random tail lawなし。したがって新laneは次の固定入力になる。

| 項目 | 新CDF/XDF ideal lane |
|---|---|
| Native synthesis signal error | 0を反実仮定。旧数値合成誤差を測定0へ変更しない。 |
| PF operator誤差 | 保存候補固有13値を保持。Exact Hamiltonian evolutionに置換しない。 |
| Attenuation/RTE/law | 決定論として1/0/0。PRのr/K・tail・sampling分散を持ち込まない。 |
| Raw signal/phase | 既存`AccuracyTarget`/`RoundEvidence`/`assess_round`を使用。δsignal=εPF、ρ=1−εPF、βdet=asin(εPF)という同じモデル。 |
| Shots | 既存Hoeffding solverがρとβstatから十分整数Nを求め、既存deterministic fixed-N検査でmarginを確認。失敗時にNを事後増加しない。 |
| DF energy誤差 | 保存norm＋guardをenergy側へ一回。PFの無次元誤差へ足さず、除去もしない。 |
| 価格 | 保存P0 kernel/shell metricsを保持。反実exact synthesisを理由にgateを減らさない。 |
| 資格 | `CONDITIONAL_IDEAL_CA_PASS`。`actual_CA_achieved`/rigorous enclosure/保証優位はnull。 |

旧packetの`segment`/`shells`/`synthesis_ready`は旧数値証拠としてそのまま保持する。新laneの0は、これらの残差から推定・丸めた値ではない。旧numerical laneと新ideal laneを別の出力namespace/labelに置く。

## 4. 共通条件と候補固有条件

| 共通条件 | 固定値 |
|---|---|
| 物理系 | Linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、sector36、system8 mode |
| DF/PF | 保存CDF8/XDF8、元frame順、全frame ordered symmetric S2/s1 |
| 時間 | δ=h=.1、m0..12、各roundのkernel反復2^m |
| AccuracyTarget | CA=.00159 Ha、DF reserve=CA/8、βtotal=π/6、βstat=π/12、各axis failure=.01/26、各candidate failure=.01 |
| 状態・alias | 各DF自身のground-state oracleと保存alias gate。状態準備費用除外。 |
| P0 | Qiskit1.3.0/opt1/seed17、rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false |
| Precision | 保存MPFR160 declared phase、Gaussian/relative fusion、位相fitなし |
| 会計 | Head/X/Y shellを含むfenced serial native kernel composition、scheduled serial depth |

各候補の近似Hamiltonian、DF誤差、ground/alias、PF誤差、kernel価格、必要shots、W、合計費用は固有のまま残す。CDF prepared内の旧rank10 XDF groundをXDF8へ流用しない。同じ元物理問題と、同じ近似Hamiltonianを混同しない。

旧CDFのκは8モード直接対角化、B3 XDFはκ/各Gのspin対応対角化である。保存routeの合法な違いを保持し、spin対応化や新しい融合の仮想削減を足さない。元g補正κ・核scalar一回・identity相対位相の所有規約も変更しない。[監査A §3/6][AUDIT]

## 5. 既存pipelineの再利用と最小差分

| 処理 | 再利用先・今回の役割 |
|---|---|
| 保存SOURCE/契約/ground/alias/候補binding | `c1_h4_strong_ca_resources.load_bundle/attach_evidence`と`c1_h4_ca_resources`。既存の全gateを利用し、公開packetとの完全一致を追加確認する。 |
| 保存DF・PF13 | 旧strong-CA packet。元CDF PF/groundとXDF8結果への既存参照を検査してから使用。 |
| Ideal round | `c1_stage3_accuracy.assess_round`。`c_same_ju_strong_d.ideal_plans`と同じ`RoundEvidence`の変換を人工で照合。 |
| 整数N margin | `c1_stage3_deterministic_accuracy.assess_fixed_round`。旧決定論の独立fixed-N検査を維持。 |
| RPE solver/horizon | 既存`rpe_resource_accounting._hoeffding_shots`、`rpe_target_round_horizon_validation`。新しいsolver/誤差式なし。 |
| Energy十分条件 | `c1_stage3_accuracy.assess_energy`。保存DF誤差一回、13round・failure・oracle/aliasを保持。 |
| Kernel/shell・serial | `c1_h6_ca_resources.serial_round_cost`→`c1_stage3_connection.compose_serial_cost`。各X/Y一interrogationの費用を構成。 |
| 総量 | `c1_stage3_accuracy.aggregate_totals/compare_totals`。六指標・shots/executions・scope/coverageを既存engineで処理。 |
| Calls/STOP/IO/resource | 既存`c1_stage3_integrated.Calls/resource_plan/progress`、既存read/save/shaを利用。新しい管理・承認証明frameworkなし。 |

新規実装は短いadapter、一つのCLI、adapter固有人工test。旧runnerや科学SOURCEを編集していない。旧`aggregate_candidate`の`NUMERICAL_PILOT_CA_PASS`というlabelをそのまま新laneへ流用せず、既存aggregation engineの結果にideal資格を付ける薄いschema変換にした。

Wは既存serial recordの`kernel_occurrences`と予定Nから、同じaggregation call内でbookkeepingする。総費用は既存engineが計算する。r、校正N、sqrt13補正を追加する余地はなく、決定論gate価格に新たな統計CIを付けない。Fixed gate priceの点区間は、signal/energy精度の厳密保証を意味しない。

## 6. 保存値の確認と、まだ行っていない評価

今回の静的確認は、指定監査の既存28公開本文・734照合等を再利用したうえで、新adapterの入力bindingに必要な範囲に絞った。二つの保存packet、旧result/契約/SOURCE/STOP、候補別native metrics/DF参照、4種CSVを確認した。[静的記録](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/technical/static_checks.json)

CSVはcandidates2、rounds26、axes52、comparison6行。PF13と旧numerical shots等はJSONの既存fieldとの一致だけを確認し、旧scalarから新しいideal N/W/費用や区間を計算していない。旧DF/PF/ground/nativeのSOURCEは表2の固定来歴へ接続する。

実行入力は**JSON30件だけ**。旧legacy loaderがs4等の文脈metadataを含むJSONを読むことはあるが、新評価は二候補s1に限定し、s4のaccuracy/費用を計算しない。NPZ/QPY/raw/ground vectorはmanifestの実行入力に含めず、decodeしない。

新科学評価に残るのは、各候補のideal round十分N、energy十分条件、W、shell込み六指標総量とCDF/XDF条件付き比較である。これらは今後の**保存数値からの科学的再解析**であり、今回の技術許可では実行していない。

## 7. 人工・mock・回帰検証

**新16＋既存53＝69 unique PASS**。[最終log](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/technical/artificial_tests.txt)。すべて発明したJSON/scalarで、実H4の保存値にaccuracy engineを適用していない。

| 対象 | 検証したこと |
|---|---|
| Identity | CDF/XDF混在、DF/PF/native価格変更、SOURCE違い、ground不適格を拒否。公開packetと再attachmentの一致。 |
| Accuracy | PF保持、counterfactual synthesis0、直接の旧ideal helperとの一致、既存N solver/fixed-N margin、DF一回、energy/failure条件。 |
| Cost | 保存候補固有六指標、head/X/Y shell、s1/2^m、W×kernel＋shellと既存aggregateの一致、校正数の倍率なし。 |
| Qualification | 数値合成laneを変更せず、actual CA/保証優位/PR統計比較をnullに保持。Infeasible/unqualifiedで総量を確定しない。 |
| 実行管理 | Passive import、SOURCE/input drift・欠落・lane不一致、opt-inなしの拒否、fresh/retry/resume0、correctness partial STOP。 |

最初の人工実行では、新fixtureが書き換えたsourceをmanifestへ登録していない一件と、旧testが一時非Gitディレクトリでsource_commitを呼ぶmock不足一件が失敗した。新fixtureを修正し、旧testは実行時のsynthetic Git contextだけをmockして再利用した。**既存科学SOURCEと旧testのファイルは無変更。** 最終69 PASSはローカル技術証拠で、CI/外部再現や科学的性能GOではない。

## 8. SOURCE・入力・passive preflight

| 固定物 | Identity |
|---|---|
| 実装SOURCE commit | `cc051d9e7b80a7818730104d45ff9a9664d8c95b` |
| [新SOURCE manifest](../../artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/technical/SOURCE.json) | 84ファイル、SHA256 `8761deb2a35e904252b6091ae96f872aa218166286bccf0584089703dc6143e5` |
| [未承認契約](research_c_xdf_cdf_common_ideal_lane_contract_unapproved_20261103.json) | SHA256 `6c4ef36ab1b7fd57fc8d516138515012be30909ab8ddfc99806f7df60526b7eb` |
| 入力 | JSON30件、各hashとpathを契約に固定。旧SOURCE・science contract・STOPを入力証拠として保持。 |
| Runtime | 既存evaluation venv、Python3.11.1観測、NumPy1.26.4/SciPy1.14.1/Qiskit1.3.0等の旧9 package pins。Compilerは価格来歴であり、今回は呼ばない。 |
| Fresh output | `artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/run01`。今回未作成。 |

SOURCEは旧strong-CAの31件と人工E2E/回帰で実際にimportされたproject module、および新module/entry/testの和集合。旧31件からのhash差は監査済み`src/trotterlib/rte.py`一件で、opt-in arithmetic backend対応、既定binary64経路は保持される。決定論に新RTE処理を適用する差ではない。旧manifestを更新・再発行していない。

Passive preflightはSOURCE84・JSON30・契約/policy・9 package版本・fresh pathを確認しPASS。NumPy/SciPy/Qiskit/ffsimのimportを禁止したprocessでもPASSし、実H4評価や新shots/W/total/CIを起動しないことを確認した。SOURCE/入力hashは科学的identityのためで、承認証明ではない。

今回実行したCLIは次の**passive commandだけ**。

```bash
PYTHONDONTWRITEBYTECODE=1 \
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
scripts/run_research_c_xdf_cdf_common_ideal_lane.py \
--contract docs/research/research_c_xdf_cdf_common_ideal_lane_contract_unapproved_20261103.json
```

出力は`BYTE_PREFLIGHT_PASS / scientific_calls=0 / maximum_logical_units=61`。将来の別明示科学許可後だけ、同じentryへ`--execute`を追加する。Flagは誤起動防止のopt-inであり、チャット許可を証明する仕組みではない。契約のUNAPPROVED metadataを承認書へ書き換える必要もない。

## 9. 未承認61-call契約・進行条件

| 処理 | 最大logical calls | 再利用/範囲 |
|---|---:|---|
| Saved bundle load | 1 | 既存JSON loaderとSOURCE/契約/status/ground/aliasの検査 |
| Candidate gate | 2 | B2/B3各一回、保存packet完全一致・候補identity |
| Ideal round accuracy | 26 | 各13round、既存solver＋fixed-N margin。PF再計算ではない。 |
| Ideal energy | 2 | 各候補の保存DF誤差・13round十分条件 |
| Round serial cost | 26 | 各候補13回のX/Y pair、既存kernel/shell会計 |
| Ideal aggregation | 2 | 六指標、W bookkeeping、資格/null伝播 |
| Comparison | 1 | CDF対XDFの固定価格/条件付き比較。PR CI/検定なし。 |
| Final summary | 1 | 二候補の結果・call count・未資格を保存 |
| **合計** | **61** | 旧86からround_synthesis26を外しsummary1を加えた実処理内訳 |

これは旧86-call runnerの再実行ではない。新fit・DF/PF norm・ground eigensolve・operator/state-action・Gaussian・native build/compile・sampling・新統計推定は0。69人工PASSやpassive preflightを61 science unitsに加算しない。

全correctness gateを通過した候補だけを評価する。Roundのinfeasibleまたは整数N未資格は、その候補の後続roundと費用を止め、totals/Wはnull。独立して開始済みの他候補は保全して完了できる。これをcorrectness FAILと同じ意味にしない。Correctness例外は既存Callsのstopで新しい依存callを止め、開始済み処理をdrainし、partial journal/traceback/STOPを保存する。

両候補全通過なら61/61、部分STOPなら61未満。Fresh run一回・retry/resume0、失敗標本や不利な候補の置換、旧未使用callsの転用なし。正常・不適格・correctness FAILのいずれもmandatory STOPし、次のPR統計比較や別条件へ進まない。

## 10. 資源・非干渉・保全・次の担当

正式共通運用指示`research_project_instructions_unified_2026-10-25.md`は依頼元workspaceの原本で§8/8.4/10を確認し適用した。起点treeに存在するとの扱いや原本の公開コピーはせず、チャット上の明示許可と科学的identity/STOPを区別する。新grant・承認原文保存・承認証明の仕組みは作らない。

主処理は保存scalarの小さい二candidate chainで、同一candidate内のround→energy→costは依存順、二candidateは既存resource_planから選ぶ並列workerで進める。新しいdense行列・BLAS/GPU workloadはなく、人工的なCPU/RAM/worker/wall上限や共有設定変更を加えない。将来実行時に実空き資源/管理者quotaを再確認し、自taskのcache/tmp/outputだけを使う。速度比を測定したという主張はない。

Codex-Aの実行中worktree、科学SOURCE・契約、raw/journal/STOP、process、cache/lock/markerを読取り・変更・停止していない。Aの途中結果で条件を選ばず、新endpointのidentityを仮定で埋めてもいない。将来固定公開されるA結果との具体的統合は、後続統合担当が行う。

旧numerical synthesis結果・H4入力/raw/科学SOURCE/契約/grant/ledger/marker/STOPと元dirty checkoutは変更0。独立branch `codex/research-c-xdf-cdf-common-ideal-lane-preparation-20261103`、worktree `/tmp/prt-c-xdf-cdf-common-ideal-lane-preparation-20261103`で、新規ファイルだけを追加する。共有PROJECT_MAP・研究概要・Evidence Map・VALIDATION_STATUS・validation manifestは今回編集せず、統合担当へ引き渡す。

**次の担当はユーザー／研究C**：固定SOURCE・JSON30入力・最大61-callの新ideal評価を実行するか、科学実行を別途明示許可するか判断する。許可された場合の**Codex担当**は、その同じ契約をfresh run01へ一度実行し、結果と未資格を公開・照合してSTOPする。今回は科学runを起動しない。

共通ideal laneが用意できても、actual CA、全law signal/synthesis、ideal-law価格移送、hardware/FT費用、分子/rank/r一般性、PR対XDF/CDFの統計的優位・同等性・非劣性は未認定。保存XDF価格がPRの既存区間内にあることを同等性と呼ばない。独立新規性・論文化GO/NO-GO・重要科学レビューは本作業の判断対象外。

Commit/push後にremote SHA/tree、公開本文、主要リンクと新旧SOURCE/入力の保全を照合する。**今回の技術準備は完了、公開照合後mandatory STOP。**

## 固定参照

[AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8375433869999c2e10501b1983c11b2b22c5d423/docs/research/research_c_strong_xdf_cdf_comparison_audit_20261103.md
[OLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/docs/research/research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md
[OLDC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/docs/research/research_c_c1_h4_strong_baseline_ca_resources_contract_binding_v2_20261026.json
[NEWSOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/928fda0a6df260b66eba1bbea63b51f2734e7b54/src/trottertracks/c_update064_r32_downstream.py
[PRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/docs/research/research_c_update064_r32_price_statistics_execution_20261103.md
