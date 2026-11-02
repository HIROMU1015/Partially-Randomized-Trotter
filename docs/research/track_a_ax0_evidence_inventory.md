# Track A AX-0：保存証拠・source・欠測目録

2026-10-09、対象worktree開始HEAD `2a80f1d5d5e5734e51d970b2b6822cd2543fd596`。[研究契約](track_a_ax0_research_contract.md)に従う静的field監査であり、新しい保存値解析・fitではない。

## 1. 利用区分と証拠の扱い

| 区分 | 意味 | AX-1での扱い |
|---|---|---|
| 1 | 保存fieldから直接取得できる | allowlist・hash・join keyを確認して利用 |
| 2 | 保存fieldとsourceの定義だけから導出できる | AX-1の別認可後にのみ導出。今回は式・fieldの存在確認だけ |
| 3 | 別の保存artifactの所在・schema・アクセス範囲の確認が必要 | 今回確認していないfieldを「存在する」と断定しない。確認できなければ欠測 |
| 4 | 新しい科学計算・trajectory再生成・build/compileが必要 | AX-1では禁止。AX-2以降の別判断に回す |
| 5 | 現在のtaskには対応しない、N/A | 無理な数値比較をしない |

`artifacts/validation_manifest.json`の関連result setは`source_present_no_current_ci`。committed保存結果でも、immutable CIや外部再現とは呼ばない。JSON内の当時の`LOCAL_UNCOMMITTED_POSTHOC`、`*_AWAITING_REVIEW`、mandatory stop、research_decision=null等はそのまま保持する。現在の格付けを上書きしない。

## 2. artifactと来歴

以下のpathはrepository rootからの相対path。source commitと結果を保存したcommitは別物である。保存hashは今回のbyte読取で確認し、科学量は再評価していない。

| ID | path・内容 | schema／source来歴 | scopeと利用できるclaim |
|---|---|---|---|
| A | `artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/pr2_matched_accuracy_m1_a_result_v1.json` | **filename v1、schema `pr2_matched_accuracy_m1_a_result_v2`**。実行source commitの専用fieldなし。authorization v1.1のbase `88bad4bd06db398c53c5e77dadb2e5b0bef23226`とrequired_source_hashesで追跡。結果保存commit `3c1831e326c27c5f679b3820997f27916d26ed9f`を実行commitと同一視しない | H4 1.00 Å、STO-3G、DF rank12、prefix0/3/6/9/12、T=.8、q=1/2/4/8、δ=.8/.4/.2/.1、K=2/4、210 signal/candidate records。未校正比較とH4開発 |
| B | `artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/pr2_matched_accuracy_m1_b1_compile_map_result_v2.json` | `pr2_matched_accuracy_m1_b1_result_v2`、source `33f436bb3a7d5b9cefa23604bb22c8d1fb17cd62` | Aと同じ210候補のfull wrapper費用。random 194 cell×32 paired trajectory、deterministic 16 cell。cost照合・sample統計 |
| M2 | `artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json` | 同名schema、source `2978e2fea672b7a1ff20cac74269ec9a610159dc` | H4 1.30 Å、同basis/rank/T。固定B2(L_D=3,q=1,r=4/8,K=2)、B0(L_D=6,q=1)、B1(L_D=12,q=1)、B3(L_D=0,q=8,r=32,K=4)、δ=.8/.1。探索・再最適化なし。条件付きcost移送 |
| PM0 | `artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/`：`summary.json`、`execution_audit.json`、`manifest.json`、`candidate_decomposition.csv`、`same_R_candidate_comparison.csv`、`fixed_q_r_K_rank_presence.csv`、`selector_metric_regret.csv`、`endpoint_N_cost_ratios.csv` | manifest `pr2_pm0_new_output_manifest_v1`、evidence commit `b6e65c6123475add5e620ec1064f361378bead95`。auditの`source_commit_fixed=false` | 既存A/B/M2の事後帰属・same-R監査。純粋なcompiled RZのgate-category分解ではない |
| PM1 | `artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/result.json` | `track_a_pm1_discard_result_v1`、source `fd7552edc0334ccf57ecf501a128c85c8d22822a` | H4 1.00 Å、rank12 target、B0 prefix4/5×q1/2/4/8、8候補。同T/二次PF。近接discardの追加 |
| PM2 | `artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/`：`summary.json`、`manifest.json`、`validation_audit_v1.json`、`claim_audit.json`、`precision_ledger.csv`、`eligibility_boundaries.csv`、`P_envelope.csv`、`representative_decomposition.csv` | source `324435d77b6642dbd44e8d1f178420daf62e77ed`。schemaは各JSONにあり、CSVはheaderで契約確認 | development218、transfer5、保存精度grid302点（.005〜.1）。同じbias/costの再使用。新しい独立信号データではない |
| V01 | `docs/manuscripts/track_a_resource_study_v0_1.md`、同supplement/claim audit、`artifacts/resource_applicability/track_a_manuscript_v0_1/2026-10-05/manifest.json`以下のdisplay CSV・図 | 参照commit `4c23453c541700c6a41ba71fc5ec9323b53858d6`。manifestに22 filesのbytes/hash | 既存claim/evidence mapと図。保存・参照のみ。AX-0で再生成・更新しない |
| H6-old | `artifacts/pf_c_system_size_validation/h2_h6_paper_d6_c_v1.json` | `pf_c_system_size_validation_v2`、provenance commit `8418192ec9844b3f51b1414b9a0126e68a036628`＋source_sha256 | H6、STO-3G、1.0 Å、DF rank11、L_D=5、δ=.250〜.256、number sector924のstate-action。新Track A matched-accuracy/finite-RTE/compiled-costの証拠ではない |

主要JSONのSHA256：

```text
A    1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086
B    71278113c32b26af0dbf6144a626237a0087478212f8a93fc908de3d4d52aee4
M2   f41a92beb57e59cddc8c063b061c40acd4da50cb76ac0698efc2bce004937931
PM0 manifest 18e7e6a4ecd99f8c09e2fb373354659c8d7728279c496bbe17aca472b3afdcaa
PM1  9305857873602d6bc4f45fbc78c4903911d083156620df01e9b23f00e7fdf05b
PM2 manifest 546cdfaf8c77f349f6f55b346e93749ce5956c9a605281d169887257377c843f
V01 manifest 9582fc78e787441066dcdcde1039de0176863cba847b6711aeae3193e6446f7b
H6-old f8f45023a52ab183c6ae95325ad8050e8fc3c753668bdc481c01d237263c294c
```

Aの保存snapshot metadata：Hamiltonian hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`、snapshot SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。M2のsnapshot SHA256 `ad7e3e7165c55dbaa395eef7a1dd74db89e1f7ab29a69ac64333f4aebf8b3e37`。これは保存metadataの引用であり、AX-0でNPZや状態をloadした結果ではない。

Aのauthorization JSON `artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/pr2_matched_accuracy_m1_execution_authorization_v1_1.json`のbyte hashは結果の`execution_authorization_sha256`と一致。moduleのrequired SHA256 `7528ebae754e14ee623d28212a668e4edc237575775a78d0d2bf4d34dc53c293`は、結果保存commit `3c1831e...`のsource blobと一致することを静的に確認した。source byte来歴は追跡できるが、専用の実行HEAD fieldがあるという意味ではない。旧H4全cellの共通compiler/identityはモデル対応2節に記す。

## 3. field単位のAX-1利用表

| 入力・量 | 保存箇所／source定義 | 区分 | 欠測・注意／利用可能なclaim |
|---|---|---|---|
| candidate/task identity | A `candidate_ledger`、各record `candidate`：method/rank/q/r/K/T/δ、compiler/identity、Hamiltonian/state/snapshot hash、fingerprint | 1 | `rank`はprefix L_Dでtarget DF rankとは別。id単独でjoinしない |
| target/raw/corrected/exact-tail signal | A `signal_records`、M2 `candidate_results.signal`：`exact_target`,`raw_mean`,`corrected_mean`,`pf_exact_tail_signal` | 1 | 各complex値のreal/imag。B0ではrandom専用fieldは存在しない |
| bias/normalization/shot | A：`axis_bias`,`axis_allowance`,`axis_shots`,`total_shots`,`normalization_multiplier/log`,`corrected_bias_abs`。PM1 signal、PM2 ledgerにも対応field | 1 | ineligibleを0 shot/0 costにしない。nullと理由を残す |
| finite distribution/λ_R | Aのrandom record、M2 signal：`exact_rte_lambda_r`,`finite_distribution`。後者にorders、probabilities、unnormalized weights、exact normalization、paper bound、residual bound、overflow flags | 1 | deterministic recordにはN/A。ranking proxy λで置換しない |
| 未丸めrandom期待値 | random record `expected_random_applications_exact`、`random_action_integer_policy`、`n_rand` | 1 | ceil前の値を取得できる。新sampling不要 |
| `W_action/W_tail` | random recordの同名field。`pr2_matched_accuracy_m1_execution.py::_random_signal_record` | 1/2 | action index。compiled RZやpaper Pauli rotation数ではない。deterministic action会計はn_det/n_fixedと保存Nから導出する |
| axis wrapper cost | B `compile_map[].compiled_axes.cosine/sine`：`cost`,6metric、scope/control/measurement/preparation/compiler/seed metadata | 1 | full wrapper。state preparationなし、quantum shots未実行。depth総和を実機runtimeと呼ばない |
| trajectory別費用 | B各axis `retained_trajectory_records[].cost`（6metric）、`sample_count`,`metric_statistics` | 1 | random32、deterministic1。n=32は古典cost sampling数で必要quantum shot数ではない |
| trajectory seed/phase/fingerprint | 同record：`trajectory_index`,`trajectory_seed`,`step_seeds`,`constant_phase`,`extracted_identity_phase`,`rte_relative_phase`、actual/evolution/wrapper semantics/provenance fingerprints | 1 | `trajectory_records_truncated=false`の保存範囲を確認。phaseはtrajectory recordにあるが、全event詳細ではない |
| paired covariance | cosine/sineをtrajectory index/seed/step seedでjoin。PM2 `sample_statistics`、M2 `compiled.paired_trajectory_rows` | 1/2 | pairing一致を先に検査。sourceのn−1 variance/covariance定義を再利用。独立axisと仮定しない |
| detailed event列 | B main result trajectory recordにはorder/component index列・basis transition列がない | 3→4 | 別保存artifactの所在と同一fingerprintを確認できる場合だけ利用。seedからの再生成は区分4でAX-1禁止 |
| `benchmark_path` | B axis field例：`medium_q_validation_benchmark`、`benchmark_validation_path=true` | 1 | これらはlabel/boolで、外部JSONの実pathではない。event保存先が示されていると誤認しない |
| basis/relative transition情報 | sourceのfragment順・builder semantics。実際のbasis matrices/Givens/event transitionは主要JSONにない | 2/3/4 | size/orderに基づくstructural proxyは2、別保存matrix metadataは3。再factorization/buildによる補充は4。AX-1のNPZ/runtime読取は自動許可しない |
| PFとfinite biasのsigned分解 | A/M2のcomplex差：corrected−pf_exact_tail、pf_exact_tail−target | 2 | telescopeはexact、絶対値は相殺しうる。数値誤差込みで監査する |
| B0 pure discard/PF | exact truncated-H signal z_Dが必要。Aの`outer_pf_bias_abs`はdiscard+PF。PM1 `pure_discard_bias_abs`,`pure_pf_bias_abs`は**null** | 3→4 | 追加saved z_Dがなければ新exact state-actionが必要。AX-1ではtotal誤差だけを扱い、因果分解を捏造しない |
| matched resource/精度ledger | B `matched_accuracy_compiled_work_no_state_preparation`、PM1 `work`、M2 `work_by_metric`、PM2 CSV `N_real/N_imag/primary_RZ_P0/primary_SE`等 | 1/2 | 保存値との照合後、登録anchorへ再会計できる。今回実施していない |
| .001/.0001 anchor | 保存bias/B/Cに対して登録shot式を適用 | 2 | **保存候補集合内**の適格性・費用は調べられる。新たな高精度最適候補や方式全体の不可能性は判定できない |
| M2 conditional prediction | M2 `predicted_work_by_metric`、`axis_shots`、sourceのdevelopment axis one-shot cost | 1/2 | held-out reference shotを使用。truth-free shot/総費用の検証ではない |
| FT/RPE総費用・chemical accuracy | T/Toffoli、全round schedule、energy window、state overlap/preparation、synthesis error配分 | 5/4 | 現taskにはN/A。将来の別taskで新契約が必要 |

## 4. 再利用sourceと実装gap

| source（特記以外は`src/trotterlib/`） | 既に存在する機能 | AX研究へのgap |
|---|---|---|
| `pr2_matched_accuracy_m1_execution.py`、`pr2_matched_accuracy_m2_transfer_execution.py` | dense finite polynomial signal、raw/corrected、shot、full wrapper compile、paired cost sampling | M1は8-qubit dense guard。generic-size state-action adapter、数値誤差契約、model入力recordが必要 |
| `src/trottertracks/resource_applicability/pm2_precision_analysis.py` | Hoeffding eligibility、精度再会計、unbiased variance・paired covariance | fixed H4 saved-result adapterを新研究のmodel audit schemaへ接続。新science不要な部分と分ける |
| `df_hamiltonian.py` | arbitrary-size DF生成、PhysicalSector number/spin、matrix-free LinearOperator、ground-state solver、workspace制限 | 新runnerのphase/identity・tail構成・数値保証・sector適合性を確認する必要 |
| `df_rte_tail.py` | symbolic DF tail、identity抽出、dense small-system adapter | symbolic経路を使う。dense adapterのmax_qubitだけを緩める設計は禁止 |
| `df_partial_s2.py`、`df_partial_s2_repeated.py` | repeated controlled PF、scalar phase、boundary optimization | state-actionとの合成とspin-sector中間作用の保存性確認 |
| `df_partial_s2_repeated_cost.py`、`df_rpe_hadamard_compiled_cost.py` | canonical unweighted seed hierarchy、full measured wrapper、MC statistics/cache fingerprints | 新target fingerprint・checkpoint・rare-order規則を接続する必要 |
| `df_partial_randomized_pf.py`、`rpe_hadamard_compiled_cost_proxy.py` | deterministic/local compiled proxy、wrapper proxy | analytical/compiled入力混在を表示。旧係数をH6/H8に未検証で移さない |
| `rte_connected_cluster_cost_validation.py`、`rte_order_stratified_cost_validation.py`等 | cluster/boundary/order-stratified費用検証、H4/H5移送の保存結果 | calibration条件が別。所在確認済みでも本研究のtest入力にはしない |
| `pf_c_system_size_validation.py` | PF行列なしのexact-tail `expm_multiply`、Qiskit half-action、sector solve。H6-oldあり | finite polynomial平均経路とは異なる。identity policyも旧`faithful_identity_in_tail`とM1`extract_identity_phase`を揃える |
| `df_gpu_statevector.py` | GPU statevector・template/batch/phase helper | finite-RTE polynomial、sector、reference、cost compileのend-to-end GPU化ではない。import時CUDA初期化等があるためAX-0でimportしない |

H8対応sector/sourceとconfigのrank15等はimplementation capabilityであり、H8の新Track A matched-accuracy結果ではない。H6-oldのrank11、旧H4 rank7、Track A rank12は別のrank policy。古いH4/H5 calibration JSONの存在だけでサイズ移送が完了したとはしない。

## 5. STOPの来歴と再設計への含意

| 保存判断 | 理由（既存文書の記録） | AXで防ぐ問題 |
|---|---|---|
| old S0 `STOP_INPUT_REPRODUCTION_MISMATCH` | Hamiltonian再現の不一致。別rebaseline系列で進んだ | hash/representation一致を最初のgateにする。old系列を混ぜない |
| M1-A `SELECTION_LIMITED` | 小さいselectorが保存候補のfrontierを十分に覆わなかった。後に別認可で全compile mapを実行 | 予測モデルがdirect比較集合を削らない。既存statusは維持 |
| P-A/P-C主研究STOP | 非退化blindで新しいcompiled改善が出ない／geometry移送gate不通過 | mechanism存在やlocal fitだけで一般寄与を主張しない |
| P-D S1 `stop_s1_undetermined_boundary_no_go_decision`、S2停止 | baseline選択の一致、regret0、boundary未決定。理想/解析改善からcompiled総費用を結論できない | 境界未決定を非優位と混同しない。高次PFをbaselineから除外しない |
| R3 `STOP_R3_NO_METHOD_DELTA` | 一般multi-fidelity手法との差分を固定できなかった | proxy/directという構造だけを新方法と呼ばない |
| FR `MECHANISM_ONLY_NO_PRACTICAL_GO` | oracle情報下のmechanismが同情報・固定予算の実用gainへ移らなかった | oracleとoperationalを分ける。機構の改善と選択効用を別metricにする |
| M2/PM1/PM2 mandatory stop | 結果後の研究方針reviewが契約。次段認可なし | AX-0文書追加をscience再開の認可にしない |

出典は[現行synthesis](研究概要・現状.md)とそこからリンクされる個別検証・decision文書。これらの別研究系列を再開・変更する作業はAX-0に含めない。

## 6. AX-1の具体的範囲と前提

実行前にA/B/M2/PM0/PM1/PM2の必要JSON/CSVとsourceの明示allowlist、hash、candidate join key、scope、model version、output directory、古典時間/RAM上限を固定する。runtime cache、registry、NPZ、v0.2、Track Bは対象に追加しない。必要なら独立レビューで読取範囲を変更してから扱う。

AX-1で行う予定の順序は、保存identity/schema/coverage → 単位対応と未校正proxy照合 → unrounded action/normalization/bias/shot/costの分離 → 同一R・q/prefix/Kのgroup診断 → 少数parameter H4校正 → oracle条件のselection regret → H6前の仮説・情報契約freeze。新event・z_D・trajectoryを生成しない。

保存値だけで閉じる成果は、native cost proxyの対応可能範囲、actionとRZの差、条件付き費用誤差、候補集合内の精度・選択感度である。H6/H8の未使用予測、強いsynthesis、operational shotモデル、B0純誤差、event-order機構のdirect検証は新たな段階が必要。欠測を残したcoverage付きの結論を正常なAX-1成果とする。
