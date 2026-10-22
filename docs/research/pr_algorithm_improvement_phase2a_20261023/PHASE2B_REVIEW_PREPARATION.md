# Phase 2B科学レビューの準備資料

状態：**REVIEW_PREPARATION_COMPLETE / REVIEW_NOT_STARTED / NO_SCIENTIFIC_PRIORITY_ASSIGNED**。

ユーザーが明示的に開始を承認した場合に、研究全体の方向性を判断するLevel 1レビューを行うための資料。GPT-6 Pro推奨はユーザー指示の推奨で、自動開始・モデル切替ではない。各課題を設計へ進めるか、統合・縮小・保留するかはユーザー/GPTの判断。レビュー開始承認と将来の設計・実装・科学実行承認は別である。

## 比較対象と共通の判断課題

[7課題の12項目定義](OPEN_RESEARCH_PROBLEMS.md)、[全10候補の再評価](STRONG_BASELINE_AND_OVERLAP_MATRIX.md)、[費用層](BOTTLENECK_AND_LEVER_MAP.md)、[反証材料](PAST_IMPROVEMENTS_AND_FAILURE_MODES.md)、[制約](EVIDENCE_LIMITATIONS.md)を一組で読む。順番・件数は研究価値の採点ではない。

判断してほしいのは、(a)何を解けば価値があるか、(b)既知法との独立差をどこに要求するか、(c)想定機構が相殺要因を変える根拠はあるか、(d)最小識別証拠の取得に価値があるか、(e)どの未知が設計前提を欠くか、である。Codexの仮説を事実として追認しない。

## 課題ごとの独立判断質問

### RQ-01 表現の小さいnormを、同じPR taskの低い実装費用へ変換できる条件

**科学的中心問題：** 同じ元Hamiltonian・sector・状態アクセス・要求精度を保つとき、表現／軌道／可解coreの変更は、normalizationやPF誤差の低下を上回る基底移動・controlled生成費を払わず、PRの測定込み資源を下げられるか。DFを出発時点の必須形式とせず、実行可能な表現の成立条件を問う。

**既に解決／検討された範囲：** Pauli/DF四方式はH2固定比較済み。H4は準備資料あり。 core/pairの強対照負結果、N1のexact fallback、THRIFT現templateの閉鎖、QC-05の小幅proxy差、QC-08の保守保証。UWC候補7件はFOUND_UNVERIFIED。既存の軌道回転やnorm最適化という名称だけでは独立差にならない。

**価値を左右する仮説と未解決：** 確認済み：生成辞書が1-shot費用を変え、normの単調改善と資源順位は一致しない。仮説：同じ物理taskで、残余の統計負担を減らしつつ基底移動の実費も下げる入力構造がある。どの構造か・大きい寄与を持つかは未確定。 QUESTION_DEFINED / MECHANISM_UNRESOLVED。反証材料と強対照は利用可能。実行可能な独立生成差と主要費用寄与は未確定。旧UWCを認証しなくても問い自体は定義可能だが、旧成功の再現研究としては扱えない。

**GPTへの具体的質問：**

1. norm最小化を超えて説明すべき構造は何か。既知orbital/BLISS/RC-DF/GRADEとの独立差を要求する位置はどこか。
2. REP-09の相殺要因を変える理由は何か。同じH・アクセスで区別できる最小の証拠に研究上の価値があるか。

**必須一次資料・固定GitHub URL：** [EV-0197](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/docs/research/representation_attribution_pair_results_20261010.md)、[EV-0200](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/representation_attribution_pair/2026-10-10/attribution_summary.json)、[EV-0198](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/src/trottertracks/representation_exploration/attribution_pair.py)、[EV-0179](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/docs/non_df_h2_process_recovery_validation.md)、[EV-0148](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/docs/research/n1_decode_recovery_v1_results_20261011.md)、[EV-0007](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/src/trotterlib/uwc.py)、[P2-GAP2S](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/096c293a85a0e756574245673ecf29d5245ddca1/docs/research/pr_algorithm_improvement_landscape_20261023/second_environment_search_20261023/STATUS.json)。SOURCEのAST-only確認や選択sectionの範囲も[索引](SOURCE_EVIDENCE_INDEX.json)で確認する。

### RQ-02 非prefix分割・groupingが既知の同情報split選択を超えるための条件

**科学的中心問題：** 単項weight順prefixより自由な決定論／random分割やgroupingは、同じ情報を使う既知のpartition・compact BCH・cost評価にも許したとき、PF誤差・tail統計・実装境界を通じて残る費用差を作れるか。一般探索を広げることと独立した方法差を区別する。

**既に解決／検討された範囲：** PR2/PM/P-Dの既存協調軸。R3は既知選択法との独立差なし。 P-D/B-Fの選択一致、BMの既知同値、R3のmethod delta不足、P-C追跡の限定失敗。normalizationだけ又はleading係数だけの改善はfull task改善ではない。四次PF逆転の原因は本課題の自動調査対象にしない。

**価値を左右する仮説と未解決：** 確認済み：固定PR-2集合では1-shot節約が増えるshotsを上回る条件がある。仮説：weight順では失う相互作用／共有構造を使う分割が、同情報known baseline後にも差を作る。非prefixであること自体は機構の証拠ではない。 QUESTION_DEFINED / METHOD_DELTA_MISSING。保存結果は反証設計に使えるが、新しい具体的差を持つ候補は未確立。既存prefixの勝者を新採択しない。

**GPTへの具体的質問：**

1. 非prefixという自由度以外に何を発見すれば既知partition/SPRINT/compact設計との研究差になるか。
2. P-D/B-F/BMの各失敗条件を変え得る構造は既存証拠から特定できるか。できなければ設計着手を保留すべきか。

**必須一次資料・固定GitHub URL：** [EV-0061](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md)、[EV-0066](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json)、[P2-M2-S](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/src/trotterlib/pr2_matched_accuracy_m2_transfer_execution.py)、[P2-ATTR](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/research/pr2_post_m2_evidence_attribution.md)、[EV-0073](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_pm1_discard_result_validation.md)、[P2-PD](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_pd_fair_comparison.md)、[EV-0081](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md)、[EV-0084](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md)、[EV-0058](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research/r3_prior_art_and_minimal_contract.md)。SOURCEのAST-only確認や選択sectionの範囲も[索引](SOURCE_EVIDENCE_INDEX.json)で確認する。

### RQ-03 既存の基底共有・control・境界融合後に残る回路生成の改善余地

**科学的中心問題：** 同じordered event列とcontrolled coherent meanを保持し、既存run-level full/support選択・同basis相殺・step境界融合を与えた後、異basis間やwrapper全体の生成に削減可能な大きい費用が残るか。何を共有できるかの構造条件を問う。

**既に解決／検討された範囲：** 基底共有・literal相殺は既存。P-A interval固有差は否定的。 support-allの悪化、P-Aのplan同一、Gaussian候補の追加手順欠如、PR-2でbasis非controlと境界融合は実装済み。Taylor高次数が稀というだけで回路利得の上限を新たに推定しない。

**価値を左右する仮説と未解決：** 確認済み：run構造の保持は局所1-shot費用を変える。仮説：これら共通最適化で消えない異basis移動の構造がfull wrapperでも主要費用であり、より安く生成可能。費用占有率と増分削減は未取得。 QUESTION_DEFINED / COST_SHARE_UNMEASURED。既存SOURCEから解消済みの箇所を除外できた。残る大きい費用と固有生成差は不明。新しい回路取得の必要性自体をPhase 2Bで判断する。

**GPTへの具体的質問：**

1. 既存の共有・境界融合を除いた後、相対基底変換に研究問題として残る差はあるか。既知compilerの適用で尽きるか。
2. full wrapperの費用寄与が未分解のまま、何を最初の識別証拠とすべきか。P-AとGaussian closureを繰り返さない停止条件は十分か。

**必須一次資料・固定GitHub URL：** [EV-0027](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_sequence_policy.md)、[EV-0028](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_sequence_policy.py)、[EV-0030](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json)、[EV-0032](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_joint_synthesis_mechanism_validation.md)、[EV-0037](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/artifacts/research_direction_joint_synthesis_mechanism_validation/2026-09-25/pa_forced_support_order2_mechanism_validation_v1.json)、[P2-BUILD](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/df_rte_qiskit.py)、[P2-REPEAT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/df_partial_s2_repeated.py)、[EV-0128](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md)。SOURCEのAST-only確認や選択sectionの範囲も[索引](SOURCE_EVIDENCE_INDEX.json)で確認する。

### RQ-04 古典生成費と有限confidenceを払っても有用なPR samplingの成立条件

**科学的中心問題：** 同じcontrolled first momentを生成するlawについて、非加法なwhole-path量子費用と、weightのmoment/range・bias・有限bit・古典生成／取得費を含めても、既知ISとcanonical RTEを超える実行可能な節約条件はあるか。最適分布の式と安い実装可能性を分ける。

**既に解決／検討された範囲：** Closed P5はRA内部特殊化。G9とH4 C1移植は既に実施。 既知ISへの還元、signed/history方式の保証移送不足、C1の小さいlogical差と処理境界の異なるCPU計測、Closed P5が既存の強い内部特殊化であること、M1-SとPR-3の統計・bias制約。C1旧P5 failureを後のP5R成功で書き換えない。

**価値を左右する仮説と未解決：** 確認済み：同finite meanの辞書/law変更がmoment・acceptance・native費用を交換する。仮説：必要なpath構造を列挙せず安く扱え、その利益が有限confidenceと古典費を戻しても残るclassがある。安い生成則と適用classは未確立。 QUESTION_DEFINED / ACCESS_AND_CONSTRUCTION_UNRESOLVED。旧資料の回収不足より具体的生成差・保証接続が不足する。C1は凍結、G6理論は保持、P5は内部特殊化として扱う。

**GPTへの具体的質問：**

1. 未知なのは既知ISの数式か、情報アクセス・有限confidence・安い生成のどこか。その接続は独立の研究問題となるか。
2. C1でparentがほぼ空でもchild回転が残る反証を踏まえ、どの費用層が変わる根拠があれば新しい問いを検討すべきか。
3. PRHS/qSHIFTのchannel/observable保証をcoherent meanへ移す問題と、同meanのlaw変更を分離できているか。M1-Sを同じ方法の成果として混ぜていないか。

**必須一次資料・固定GitHub URL：** [EV-0201](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md)、[EV-0202](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trotterlib/rte.py)、[EV-0144](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/src/trottertracks/algorithm_codesign/g7_generator.py)、[EV-0117](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/g9_p5.py)、[EV-0115](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md)、[EV-0142](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md)、[P2-C1-ORD](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/result/ordinary/summary.json)、[P2-C1-FULL](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/result/full_return/summary.json)、[EV-0147](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/result/summary.json)、[EV-0139](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a48b3c8244fb5d932e0d6bf4a5de6bb573ce35ed/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/handoff.md)、[EV-0134](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md)。SOURCEのAST-only確認や選択sectionの範囲も[索引](SOURCE_EVIDENCE_INDEX.json)で確認する。

### RQ-05 native回転辞書・配分・合成位置の変更が全wrapperで有用になる条件

**科学的中心問題：** 同じfinite target・精度を生成する回転辞書／allocation／合成配置の選択に、既知Euler pairing・cost-aware IS・共通fusion後も残る資源機構はあるか。加法primitive価格での利益が、全wrapper・複数step・準備費の条件で残る範囲を問う。

**既に解決／検討された範囲：** Closed P5はRA内部特殊化。G9とH4 C1移植は既に実施。 SPのn依存反転、S4の同一proposalとbest-to-bestの交絡、prep費・多stepの未確立、D系列の微小価格margin、C1の小さいlogical差。commonより低いL1をallocationだけで得るという前提は採らない。

**価値を左右する仮説と未解決：** 確認済み：固定native providerで単価と統計負担の交換が費用差を作る。仮説：加法価格に依存しない構造的な節約が、同精度全wrapperと多stepでも残る。既存の約束済み新合成価格やoracleを仮定しない。 QUESTION_DEFINED / WHOLE_WRAPPER_TRANSFER_UNMEASURED。保存された局所機構と強対照はある。新たな独立生成差・頑健な条件は不明。Parent allocation D+E closureは変更しない。

**GPTへの具体的質問：**

1. 既存SP/S4/D系列が既に示した交換関係を超え、研究として未解決な中心問題は何か。
2. 加法primitive価格からwhole wrapperへの差は研究機構か実装上の確認にとどまるか。どの証拠が設計着手の価値を変えるか。

**必須一次資料・固定GitHub URL：** [EV-0086](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)、[EV-0091](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/saved_field_audit_v1.json)、[EV-0131](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md)、[EV-0132](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/parent_allocation.py)、[EV-0133](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/saved_output_audit/saved_output_audit_s4.json)、[EV-0129](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md)、[EV-0115](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md)、[EV-0119](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_affine_T_comparison_v2.json)。SOURCEのAST-only確認や選択sectionの範囲も[索引](SOURCE_EVIDENCE_INDEX.json)で確認する。

### RQ-06 roundごとに表現・splitを変えても同じenergyを推定できる条件

**科学的中心問題：** 時刻／roundごとに表現・rank・splitが異なる信号を、同じ元Hの絶対energy taskへ接続し、既存round別r/K・beta/alpha配分を超える費用節約を得られるか。変更した有効位相・状態overlap・branchをどの情報で整合させられるかを問う。

**既に解決／検討された範囲：** PR2/PM/P-Dの既存協調軸。R3は既知選択法との独立差なし。 同情報の汎用配分への還元、長q移送と状態準備への感度、M1-Sのphase bias、PF校正の一般移送不足。二つのT値やPMのepsilon地図から全round成功を推論しない。

**価値を左右する仮説と未解決：** 確認済み：既存固定構成でもround費用は一様でなく配分が効く。仮説：表現/split自体を変えると、共通energyへのbridgeと変更費を払っても追加の節約がある。現証拠にその実証はない。 QUESTION_DEFINED / MATHEMATICAL_INTERFACE_UNRESOLVED。主な欠落は新しいtarget整合条件で、旧rawを探せば閉じる問題ではない。設計・証明へ進む価値をPhase 2Bで判断。

**GPTへの具体的質問：**

1. 同じ元energyを推定するためのbridgeは既知の多精度推定で尽きるか、未解決の算法問題があるか。
2. round別r/K配分に表現/splitを追加することは科学的に十分な差か。保証の取得費を含めて識別できるか。

**必須一次資料・固定GitHub URL：** [EV-0038](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_wp11_synthesis.md)、[P2-OPT2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_full_opt2.md)、[EV-0039](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_full_opt2.py)、[EV-0072](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_pm2_precision_resource_result_validation.md)、[EV-0134](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md)、[EV-0020](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/second_study_v2/evidence_integration_20261006/approved_closure.md)、[EV-0021](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/research_audits/pf_finite_time_completion_20261016/open_question_and_evidence_map.md)。SOURCEのAST-only確認や選択sectionの範囲も[索引](SOURCE_EVIDENCE_INDEX.json)で確認する。

### RQ-07 接続性がPR表現・分割の選択を変える条件

**科学的中心問題：** 明示した接続性と同じmapping/routing最適化を全方式に与えたとき、PRのbasis移動・演算子分割・表現の選択は、既知compilerへの単純接続を超える資源差を作れるか。

**既に解決／検討された範囲：** routing/Gaussian/mapping自体は既知。世界的未探索としない。 既知routing・Gaussian共有への還元可能性。RZの改善とCX/depthの悪化、shotsによる相殺。routingなしの結果をhardware性能へ外挿しない。

**価値を左右する仮説と未解決：** 確認済み：主要保存値はroutingなし。仮説：接続性で増えるbasis移動費が十分大きいclassでは、表現/splitとの協調に価値がある。支配性も同情報baseline後の差も未測定。 QUESTION_DEFINED / PRIOR_ART_AND_COST_SHARE_UNRESOLVED。Phase 2Bで問題自体の価値を判断する段階。性能資料が揃ったという意味のreview-readyではない。

**GPTへの具体的質問：**

1. これはPRの未解決問題か、既知routing/Gaussian compilerの適用範囲か。何が見つからなければ保留とすべきか。
2. topology-freeで未測定の費用層を調べる価値は、既存証拠からどこまで説明できるか。公正なbaselineの必須条件は何か。

**必須一次資料・固定GitHub URL：** [EV-0027](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_sequence_policy.md)、[EV-0061](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md)、[EV-0179](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/docs/non_df_h2_process_recovery_validation.md)、[EV-0142](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md)、[P2-BUILD](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/df_rte_qiskit.py)。SOURCEのAST-only確認や選択sectionの範囲も[索引](SOURCE_EVIDENCE_INDEX.json)で確認する。

## 十分な事実と不足の境界

|課題|現証拠で判断材料にできる事項|未取得・科学判断に残る事項|
|---|---|---|
|RQ-01|normと生成費の相殺例、H2既存比較、UWC実装／来歴制約|PR向け独立表現差、主要費用を変える構造、大系移送|
|RQ-02|強い同情報対照での選択一致・compact同値、固定prefix支持|非prefixの具体的差と、探索／情報取得費を戻した価値|
|RQ-03|run-level局所改善、P-A plan同一、実装済みcontrol/fusion|異basis移動の未解消な大費用・既知compiler後の独立生成差|
|RQ-04|同mean lawの既知条件、C1 child残存、P5/CTS強対照、M1-Sの限界|安い生成アクセス、全trial finite-confidence、実PR費用への接続|
|RQ-05|SP/S4/D系列の固定費用交換・価格感度|whole-wrapper・多stepでの有用性、独立機構、prep/online費|
|RQ-06|r/K配分は既存、一時刻precision地図は全RPEではない|構成別phaseを共通energyへ接続する条件と追加利益|
|RQ-07|主要比較はroutingなし、既知connectivity生成が存在|PR固有の独立差、routing費の支配性、公正な適用baseline|

RQ-01は旧UWC原本なしでも他の表現反証から問題を定義できる。RQ-02/03の旧H4全来歴は不足のままで、別固定SOURCEの静的事実と混同しない。RQ-04/05はC1凍結・S4 D+E closureを維持する。RQ-06/07の不足は旧GAP回収では埋まらない新しい理論／性能証拠を含む。

## レビューの境界と返す内容の案

設計・証明・実装を始める前に、残す問い、統合する問い、保留する問いと理由、必要な独立差、最小識別証拠がどの判断を変えるかを示す。正式な科学的優先順位・主張の採択はユーザーが判断する。具体的な新算法構成や数値救済scanを、この資料から自動生成しない。

一次文献は[案内](PRIMARY_SOURCE_GUIDE.md)のversion/確認範囲を参照。PR/composite/SPRINT/GRADE/RC-DF/BLISS/partition/IS/connectivityとの既知重複を出発点にする。PRHS/qSHIFTの詳細は既存固定監査に帰属し、全定理の新しい検証済みという扱いにしない。新規性を確定するための網羅調査は未実施。

重要な科学レビュー開始は未承認。今回の公開・技術PASSはレビュー結果でも科学GOでもない。追加の別環境探索を次のToDoにしない。
