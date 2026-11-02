# PR改善の未解決研究課題 — Phase 2Bへの候補

識別子順であり順位ではない。各failure/stop criterionは将来の比較を考えるための案で、既存STOP変更や新たな科学的採否ではない。具体的な新方式・証明・実験契約は作っていない。

## RQ-01 表現の小さいnormを、同じPR taskの低い実装費用へ変換できる条件

由来：OP-I3, OP-II1。関連研究：REP-03, REP-04, REP-08, REP-09, QC-05, QC-08, REP-07。

### 1. Problem statement

同じ元Hamiltonian・sector・状態アクセス・要求精度を保つとき、表現／軌道／可解coreの変更は、normalizationやPF誤差の低下を上回る基底移動・controlled生成費を払わず、PRの測定込み資源を下げられるか。DFを出発時点の必須形式とせず、実行可能な表現の成立条件を問う。

### 2. Existing bottleneck

norm・rank・群数を単独で小さくしても、実際の生成単位、基底変換、残余bias、shot allowanceは同時に良くならない。core/pairの固定toyでは安いwhole-Pauli対照が強い。UWCの旧proseは元run来歴未照合であり、成功実証の出発点にできない。

### 3. Evidence

REP-09は固定3-mode正平方DF2／physical-pair toyでnorm低下と単価増の相殺を保存済み寄与表まで記録する。REP-08はH2四方式の条件付き固定task比較を完了済み。QC-05は同情報の既知一体抽出が強い。QC-08では新しい十分保証が強いcentered対照より鋭くない例、THRIFTではstep減少が登録算術templateの費用余白を作らない例がある。異なる実験の値を合成しない。

### 4. Proposed research freedom

軌道/frame、sector-preserving shift、分解辞書、可解成分と残余の表現、表現取得精度。具体的な最適化則・新分解・目的関数の設計は未着手。

### 5. Strong baseline

同じ入力情報・sector・model誤差枠を持つcollected Pauli、既存DF/圧縮DF、適用できる既知orbital/grouping・BLISS・SPRINT/GRADE系。各表現にdiscard/full deterministic/partial/random endpoint、同じ合法fusion/control/compilerを許す。外部論文の別task費用を数値対照として流用しない。

### 6. Improvement mechanism hypothesis

確認済み：生成辞書が1-shot費用を変え、normの単調改善と資源順位は一致しない。仮説：同じ物理taskで、残余の統計負担を減らしつつ基底移動の実費も下げる入力構造がある。どの構造か・大きい寄与を持つかは未確定。

### 7. Counterevidence

core/pairの強対照負結果、N1のexact fallback、THRIFT現templateの閉鎖、QC-05の小幅proxy差、QC-08の保守保証。UWC候補7件はFOUND_UNVERIFIED。既存の軌道回転やnorm最適化という名称だけでは独立差にならない。

### 8. Scientific value

成立／不成立を分ける構造条件が得られれば、PR向け表現選択について既知のnorm指標を超える知見になり得る。現時点で新算法・独立新規性・論文化可能性を認定しない。

### 9. Smallest distinguishing evidence

まず同じHへのbindingと、既知表現対照が再現できない生成上の差を文書で示すこと。次に、費用の大きい項が実際に変わることとbias/normalizationの相殺を同じtaskで区別する証拠が要る。新信号・軌道最適化・compileが必要なら、その小さい比較範囲を別承認する。旧UWC探索・旧DF復元を前提にしない。

### 10. Failure/stop criterion

新しい記述が既知の同情報表現と同じ生成列へ還元される、又は減らす項が全費用の小部分にとどまり追加生成・統計負担が相殺するなら、この経路の優先度を下げる。旧toyのparameter追加だけで救済しない。閾値・採否はPhase 2Bで未採択。

### 11. Current readiness

QUESTION_DEFINED / MECHANISM_UNRESOLVED。反証材料と強対照は利用可能。実行可能な独立生成差と主要費用寄与は未確定。旧UWCを認証しなくても問い自体は定義可能だが、旧成功の再現研究としては扱えない。

### 12. Provenance

[EV-0179](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/docs/non_df_h2_process_recovery_validation.md)、[EV-0197](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/docs/research/representation_attribution_pair_results_20261010.md)、[EV-0200](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/representation_attribution_pair/2026-10-10/attribution_summary.json)、[EV-0198](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/src/trottertracks/representation_exploration/attribution_pair.py)、[EV-0005](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/notes/uwc_current_implementation_and_results.md)、[EV-0007](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/src/trotterlib/uwc.py)、[P2-GAP2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/096c293a85a0e756574245673ecf29d5245ddca1/docs/research/pr_algorithm_improvement_landscape_20261023/second_environment_search_20261023/SEARCH_REPORT.md)、[P2-GAP2S](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/096c293a85a0e756574245673ecf29d5245ddca1/docs/research/pr_algorithm_improvement_landscape_20261023/second_environment_search_20261023/STATUS.json)、[EV-0168](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md)、[EV-0173](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f7e103bf3a0ffcee6162bd4b8d244f9a2f4502c5/docs/research/qc_a2_current_route_handoff_20261018/achievements_and_limits.md)、[EV-0158](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_mixed_access_feasibility_v1_closure_20261013.md)、[EV-0148](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/docs/research/n1_decode_recovery_v1_results_20261011.md)。固定commit・path・bytes・取得／静的確認範囲は[SOURCE索引](SOURCE_EVIDENCE_INDEX.json)と[一次資料案内](PRIMARY_SOURCE_GUIDE.md)を参照。containing commitと元科学SOURCEを混同しない。

## RQ-02 非prefix分割・groupingが既知の同情報split選択を超えるための条件

由来：OP-I4, OP-II2。関連研究：PR-01, EVAL-01, OPT-03, OPT-04, OPT-05, RB-01, RB-02, QC-05。

### 1. Problem statement

単項weight順prefixより自由な決定論／random分割やgroupingは、同じ情報を使う既知のpartition・compact BCH・cost評価にも許したとき、PF誤差・tail統計・実装境界を通じて残る費用差を作れるか。一般探索を広げることと独立した方法差を区別する。

### 2. Existing bottleneck

安いdeterministic backbone、tail norm、commutator、basis共有は相反し得る。良いscoreを追加しても既知対照の候補選択が同じなら意思決定上の利益はない。候補探索・誤差情報取得を候補側だけ無料にできない。

### 3. Evidence

PR-2 M1/M2とPM-1/2には固定prefix候補内のpartial支持と精度依存があるが、非prefixの優位は測っていない。P-Dは強いB1b/B2/B4が同じ候補を選択。B-Fは同family・同評価budgetでF固有価値なし。BMの三次係数は同じcompact再帰と同値。P-Cは局所pilot後の広いgeometry追跡で固有差・予測が不足した。

### 4. Proposed research freedom

保持／random化する演算子集合、groupの単位と順序、共有basisとcommutator構造の利用、評価に使う情報の範囲。新しい非prefix規則、探索アルゴリズム、証明は作らない。

### 5. Strong baseline

PR原法/composite partition、SPRINT/GRADE、同じ候補domainとDF backendを持つcompact BCH再帰・cache、PR-2近接discardを含むendpoint。全方式へ同じ情報・探索機会・精度保証とlegal circuit最適化を与える。prefixだけに制限した対照を最強としない。

### 6. Improvement mechanism hypothesis

確認済み：固定PR-2集合では1-shot節約が増えるshotsを上回る条件がある。仮説：weight順では失う相互作用／共有構造を使う分割が、同情報known baseline後にも差を作る。非prefixであること自体は機構の証拠ではない。

### 7. Counterevidence

P-D/B-Fの選択一致、BMの既知同値、R3のmethod delta不足、P-C追跡の限定失敗。normalizationだけ又はleading係数だけの改善はfull task改善ではない。四次PF逆転の原因は本課題の自動調査対象にしない。

### 8. Scientific value

既知partitionに含まれない有用な構造条件、又はどの追加情報が設計判断を変えるかを示せれば研究知見になり得る。汎用optimizerや評価台帳の再提案だけの価値はPhase 2Bで別に判断する。

### 9. Smallest distinguishing evidence

同じ情報・domainを持つ既知対照との差を一つの固定比較で識別できる記述が必要。まず差がscoreの集約順やcache禁止に由来しないことを確認する。新splitの性能や候補再ランキングは今回行わず、必要なら限定した同task比較を別途承認する。

### 10. Failure/stop criterion

強い対照が同じscore・候補・再利用を再現できる場合、独立算法経路として進めない。未測定の探索費を0としてしか利得を説明できない場合も優先度を下げる。旧P-D/B-F/BM/R3/P-CのSTOPは維持する。

### 11. Current readiness

QUESTION_DEFINED / METHOD_DELTA_MISSING。保存結果は反証設計に使えるが、新しい具体的差を持つ候補は未確立。既存prefixの勝者を新採択しない。

### 12. Provenance

[EV-0061](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md)、[EV-0066](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json)、[P2-M2-S](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/src/trotterlib/pr2_matched_accuracy_m2_transfer_execution.py)、[EV-0073](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_pm1_discard_result_validation.md)、[EV-0072](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_pm2_precision_resource_result_validation.md)、[P2-ATTR](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/research/pr2_post_m2_evidence_attribution.md)、[P2-PD](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_pd_fair_comparison.md)、[EV-0081](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md)、[EV-0084](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md)、[EV-0047](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_geometry_tracking_breakdown.md)、[EV-0058](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research/r3_prior_art_and_minimal_contract.md)。固定commit・path・bytes・取得／静的確認範囲は[SOURCE索引](SOURCE_EVIDENCE_INDEX.json)と[一次資料案内](PRIMARY_SOURCE_GUIDE.md)を参照。containing commitと元科学SOURCEを混同しない。

## RQ-03 既存の基底共有・control・境界融合後に残る回路生成の改善余地

由来：OP-I1。関連研究：CIR-01, CIR-02, PR-01, RB-12, RB-13。

### 1. Problem statement

同じordered event列とcontrolled coherent meanを保持し、既存run-level full/support選択・同basis相殺・step境界融合を与えた後、異basis間やwrapper全体の生成に削減可能な大きい費用が残るか。何を共有できるかの構造条件を問う。

### 2. Existing bottleneck

単発supportの安さは長いrunの共有を壊し得る。basisの費用占有率はPR-2集約RZからは分解できない。すでに無制御のbasisをさらに無制御化する、融合済み境界を再融合するという前提には改善余地がない。

### 3. Evidence

WP06bは独立短列holdoutでrun policyの局所削減を記録。P-A後続はcompile前planが強い一区間baselineと同一で、interval分割固有寄与を支持しない。actual M2 sourceのDF builderはcontrol_diagonal_onlyを強制し同basisを共有、repeated S2はblock0の境界halfを融合する。Gaussian積生成closureは候補固有の追加手順なしであり性能negativeではない。

### 4. Proposed research freedom

厳密なrelative Gaussian変換、support completionと基底間移動、legalな順序保持共有、scalar/ancilla位相の実装位置。新compiler、回路、合成policyは設計・実装しない。

### 5. Strong baseline

full-basis共有、既存support_run_le_1、one_segment_per_source_run、既存diagonal-only control、同block boundary fusion、全方式共通actual-adjoint/literal inverse cancellation、適用可能な既知partial Gaussian compiler。WとZを別回路にする弱い対照を強制しない。

### 6. Improvement mechanism hypothesis

確認済み：run構造の保持は局所1-shot費用を変える。仮説：これら共通最適化で消えない異basis移動の構造がfull wrapperでも主要費用であり、より安く生成可能。費用占有率と増分削減は未取得。

### 7. Counterevidence

support-allの悪化、P-Aのplan同一、Gaussian候補の追加手順欠如、PR-2でbasis非controlと境界融合は実装済み。Taylor高次数が稀というだけで回路利得の上限を新たに推定しない。

### 8. Scientific value

位相を保つ同一taskの回路生成に、強い既知compiler後にも構造的な費用削減が残る条件を示せれば価値がある。既知compilerの適用研究か独立構成かは未判断。

### 9. Smallest distinguishing evidence

どの既存操作列が消え、強対照ではなぜ同じ削減を得られないかをまず示す。必要なら固定event列に対するoperator/ancilla相対位相とfull wrapper費用の対照を別承認で確認する。compile前planが同じなら科学compileの追加だけで機構を救済しない。

### 10. Failure/stop criterion

差が既存fusion・run共有・既知Gaussian積生成に尽きる、又はfull-wrapperで変える項の寄与が小さく追加費用で消えるなら縮小する。interval DPの名称変更再開はしない。

### 11. Current readiness

QUESTION_DEFINED / COST_SHARE_UNMEASURED。既存SOURCEから解消済みの箇所を除外できた。残る大きい費用と固有生成差は不明。新しい回路取得の必要性自体をPhase 2Bで判断する。

### 12. Provenance

[EV-0027](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_sequence_policy.md)、[EV-0028](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_sequence_policy.py)、[EV-0030](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json)、[EV-0032](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_joint_synthesis_mechanism_validation.md)、[EV-0037](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/artifacts/research_direction_joint_synthesis_mechanism_validation/2026-09-25/pa_forced_support_order2_mechanism_validation_v1.json)、[P2-BUILD](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/df_rte_qiskit.py)、[P2-REPEAT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/df_partial_s2_repeated.py)、[P2-ATTR](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/research/pr2_post_m2_evidence_attribution.md)、[EV-0128](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md)、[EV-0129](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md)、[EV-0130](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/rte_development/accounting.py)。固定commit・path・bytes・取得／静的確認範囲は[SOURCE索引](SOURCE_EVIDENCE_INDEX.json)と[一次資料案内](PRIMARY_SOURCE_GUIDE.md)を参照。containing commitと元科学SOURCEを混同しない。

## RQ-04 古典生成費と有限confidenceを払っても有用なPR samplingの成立条件

由来：OP-II3, OP-I2。関連研究：RB-18, RB-14, RB-06, RB-09, RB-16, RB-17, RB-15, PR-02。

### 1. Problem statement

同じcontrolled first momentを生成するlawについて、非加法なwhole-path量子費用と、weightのmoment/range・bias・有限bit・古典生成／取得費を含めても、既知ISとcanonical RTEを超える実行可能な節約条件はあるか。最適分布の式と安い実装可能性を分ける。

### 2. Existing bottleneck

basis切替や高価格eventを減らすと重み・必要shotsが増え得る。full-path costやnormalizerを知るための列挙・compile oracleを無料にできない。zero試行と失敗を区別し、parent縮約後のchild回転も残る。

### 3. Evidence

時間相関監査はmarginal保存だけでcoherent meanが保てないこと、whole-path ISが既知であることを記録。C1は同じH4 tailでlogical差が小さく全pathのchild回転が残る。G9は同finite P5の条件付きT利益を保存するがCTSにCXの利点がある。B3は限定workloadのみ。M1-Sは別mean/干渉構成でvisibilityとphase biasが交換し、rate比をshot比にできない。PR-3はbias減少がvarianceとbackbone費に負けた。

### 4. Proposed research freedom

proposalの表現と生成アクセス、historyの保持、likelihood計算、normalizerのアクセス形式、zero/rejection処理、再利用可能な前処理。具体的なlaw・相関kernel・samplerや新証明は作らない。

### 5. Strong baseline

同じ情報での既知whole-path IS/cost-aware IS、canonical ordinary/partial/P3+tail、同finite targetのGeneral Full/Closed P5/collected CTS。全方式へ同じ合法cache・selector・前処理を許し、全trialの量子測定と古典生成を含める。branch相関でtargetを変える方式は同mean対照と混ぜず別のbias契約を必要とする。

### 6. Improvement mechanism hypothesis

確認済み：同finite meanの辞書/law変更がmoment・acceptance・native費用を交換する。仮説：必要なpath構造を列挙せず安く扱え、その利益が有限confidenceと古典費を戻しても残るclassがある。安い生成則と適用classは未確立。

### 7. Counterevidence

既知ISへの還元、signed/history方式の保証移送不足、C1の小さいlogical差と処理境界の異なるCPU計測、Closed P5が既存の強い内部特殊化であること、M1-SとPR-3の統計・bias制約。C1旧P5 failureを後のP5R成功で書き換えない。

### 8. Scientific value

抽象的な最適lawから実行可能な生成・認証へ移る際の障害、又は成立する構造classを明確にできれば研究価値があり得る。ISの既知式やFullとP5の名称差は新規性として数えない。

### 9. Smallest distinguishing evidence

既知法との差を持つ生成アクセスと、同finite target・support/likelihood・finite-bit・全trial意味論をまず明記する。次に、減らす量子費用項が実際に十分大きいかを既存証拠で判断する。足りなければ一つの限定classでの生成費と有限confidence比較が必要になり得るが、今回実装・新sampling・N再計算をしない。

### 10. Failure/stop criterion

最適lawの存在説明だけで安い生成がない、meanを保持できない、又はnormalization/range・古典費を払うと利益が消えるなら設計着手の優先度を下げる。H4 degree-zero parentへの期待だけでC1を再開しない。

### 11. Current readiness

QUESTION_DEFINED / ACCESS_AND_CONSTRUCTION_UNRESOLVED。旧資料の回収不足より具体的生成差・保証接続が不足する。C1は凍結、G6理論は保持、P5は内部特殊化として扱う。

### 12. Provenance

[EV-0201](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md)、[EV-0202](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trotterlib/rte.py)、[EV-0131](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md)、[EV-0132](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/parent_allocation.py)、[EV-0115](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md)、[EV-0119](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_affine_T_comparison_v2.json)、[EV-0142](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md)、[EV-0147](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/result/summary.json)、[EV-0144](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/src/trottertracks/algorithm_codesign/g7_generator.py)、[EV-0146](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/src/trottertracks/algorithm_codesign/g6_f3_generator.py)、[EV-0139](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a48b3c8244fb5d932e0d6bf4a5de6bb573ce35ed/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/handoff.md)、[EV-0068](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/pr2_pr3_minimal_pilot_validation.md)、[EV-0070](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/artifacts/pr2_pr3_minimal_pilot/2026-09-27/pr2_pr3_minimal_pilot_v1.json)、[EV-0134](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md)。固定commit・path・bytes・取得／静的確認範囲は[SOURCE索引](SOURCE_EVIDENCE_INDEX.json)と[一次資料案内](PRIMARY_SOURCE_GUIDE.md)を参照。containing commitと元科学SOURCEを混同しない。

## RQ-05 native回転辞書・配分・合成位置の変更が全wrapperで有用になる条件

由来：OP-II4, OP-I2。関連研究：RB-03, RB-09, RB-13, RB-14, RB-17。

### 1. Problem statement

同じfinite target・精度を生成する回転辞書／allocation／合成配置の選択に、既知Euler pairing・cost-aware IS・共通fusion後も残る資源機構はあるか。加法primitive価格での利益が、全wrapper・複数step・準備費の条件で残る範囲を問う。

### 2. Existing bottleneck

cheap角度への移動はnormalizationやshotsを増やす。T価格・CX・depthは同じ方向に動かず、価格取得後の選択とproxy最適化が交絡し得る。小さいparent部分を最適化してもchildとdeterministic費が残る。

### 3. Evidence

SP-1は固定synthetic wrapperで局所gainとn増加後loss/shot capを記録し、actual whole-wrapper compileではない。S4はcommonのL1最小性を保持しつつ非commonの条件付きT利益とN/CX増を記録。訂正KKTは固定価格proxyだけ。D1-D3は保証と合成価格の小差で有限集合の選択が変わる保存例。G9のT-primaryとCTSのCXを両方保持する。

### 4. Proposed research freedom

平均を保存する回転辞書、parent配分、合成を行う位置、誤差配分、合法な隣接融合。具体的な角度・allocation・合成policyの設計や価格探索をしない。

### 5. Strong baseline

common/非commonと同じ既知ISの機会、ordinary/partial/P3+tail/Full/Closed P5/CTS、exact cheap角とactual-adjoint、全方式同じliteral inverse cancellation、適用可能な既知probabilistic synthesis。native保証とprepare/readoutの費用枠を一致させる。

### 6. Improvement mechanism hypothesis

確認済み：固定native providerで単価と統計負担の交換が費用差を作る。仮説：加法価格に依存しない構造的な節約が、同精度全wrapperと多stepでも残る。既存の約束済み新合成価格やoracleを仮定しない。

### 7. Counterevidence

SPのn依存反転、S4の同一proposalとbest-to-bestの交絡、prep費・多stepの未確立、D系列の微小価格margin、C1の小さいlogical差。commonより低いL1をallocationだけで得るという前提は採らない。

### 8. Scientific value

既知部品を越える生成機構、又はどの費用・precision・準備条件で交換が有用かを事前に区別する知見が候補。新規性・主論文価値・有用な最適化則の存在は未判断。

### 9. Smallest distinguishing evidence

辞書変更の効果とproposal変更を分け、既知強対照後にも消える具体的操作・誤差項を先に特定する。必要な将来証拠は一つの同finite targetでの全wrapper費と十分budgetの対応であり、新angle/native取得・再rankingは別承認。S4の同toy探索延長を既定にしない。

### 10. Failure/stop criterion

差が既知IS/Euler/cheap角の直接再現のみ、又は加法模型でしか残らず共有・準備・多stepを戻すと相殺するなら縮小する。合成seed/precision追加で小marginを救済しない。

### 11. Current readiness

QUESTION_DEFINED / WHOLE_WRAPPER_TRANSFER_UNMEASURED。保存された局所機構と強対照はある。新たな独立生成差・頑健な条件は不明。Parent allocation D+E closureは変更しない。

### 12. Provenance

[EV-0086](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)、[EV-0091](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/saved_field_audit_v1.json)、[EV-0087](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md)、[EV-0092](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_sp05_economics_result/2026-10-06/v1/result.json)、[EV-0115](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md)、[EV-0119](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_affine_T_comparison_v2.json)、[EV-0129](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md)、[EV-0130](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/rte_development/accounting.py)、[EV-0131](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md)、[EV-0132](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/parent_allocation.py)、[EV-0133](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/saved_output_audit/saved_output_audit_s4.json)、[EV-0142](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md)。固定commit・path・bytes・取得／静的確認範囲は[SOURCE索引](SOURCE_EVIDENCE_INDEX.json)と[一次資料案内](PRIMARY_SOURCE_GUIDE.md)を参照。containing commitと元科学SOURCEを混同しない。

## RQ-06 roundごとに表現・splitを変えても同じenergyを推定できる条件

由来：OP-III2, OP-I4。関連研究：OPT-01, PR-01, EVAL-01, RB-15, PF-05。

### 1. Problem statement

時刻／roundごとに表現・rank・splitが異なる信号を、同じ元Hの絶対energy taskへ接続し、既存round別r/K・beta/alpha配分を超える費用節約を得られるか。変更した有効位相・状態overlap・branchをどの情報で整合させられるかを問う。

### 2. Existing bottleneck

一時刻の安いsignalが同じenergyの一貫したround列とは限らない。構成によってPF固有位相、discard/model bias、visibilityが変わる。構成更新・情報取得・準備・compiler較正費も必要。

### 3. Evidence

WP系列は既にround別r/Kと統計／誤差配分を扱い、共通最適化が比較を変える。PM-2は一時刻精度と構成の関係を保存したが全RPEではない。M1-Sは固定signalと全energy/RPE接続の不足を明記。PF有限時間研究は系・time窓による移送不足を記録する。

### 4. Proposed research freedom

roundごとの表現・分割・近似構成、構成変更に伴う情報の再利用範囲。具体的なadaptive law、decoder、新しいphase/energy保証は設計しない。

### 5. Strong baseline

固定表現/splitに既存round別r/K・beta/alpha/shot配分を最適化した対照、同じcost/error情報の既知adaptive/multifidelity運用。真のE0やexact spectral oracleを候補だけへ与えない。

### 6. Improvement mechanism hypothesis

確認済み：既存固定構成でもround費用は一様でなく配分が効く。仮説：表現/split自体を変えると、共通energyへのbridgeと変更費を払っても追加の節約がある。現証拠にその実証はない。

### 7. Counterevidence

同情報の汎用配分への還元、長q移送と状態準備への感度、M1-Sのphase bias、PF校正の一般移送不足。二つのT値やPMのepsilon地図から全round成功を推論しない。

### 8. Scientific value

構成が変わるphaseデータを同じenergyへ結ぶ成立条件と、既知schedule配分では得られない差を示せれば候補。単なる一般動的計画や評価体系の再設計を本課題の成果にしない。

### 9. Smallest distinguishing evidence

まず異構成の信号が同じ元energyに結び付く条件と、固定構成＋既知配分にない節約項を示す資料が要る。新しい非自明な保証が必要ならその必要性だけを記録し、今回証明しない。全RPE simulationや既存表のround再会計を行わない。

### 10. Failure/stop criterion

bridgeに未知E0／高価なspectral情報が必要、又は同情報の既存配分で選択・費用が再現されるなら着手優先度を下げる。異なるtargetのcheap signalを混ぜて成功にしない。

### 11. Current readiness

QUESTION_DEFINED / MATHEMATICAL_INTERFACE_UNRESOLVED。主な欠落は新しいtarget整合条件で、旧rawを探せば閉じる問題ではない。設計・証明へ進む価値をPhase 2Bで判断。

### 12. Provenance

[EV-0038](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_wp11_synthesis.md)、[EV-0039](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_full_opt2.py)、[EV-0072](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_pm2_precision_resource_result_validation.md)、[EV-0061](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md)、[P2-ATTR](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/research/pr2_post_m2_evidence_attribution.md)、[EV-0134](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md)、[EV-0020](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/second_study_v2/evidence_integration_20261006/approved_closure.md)、[EV-0021](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/research_audits/pf_finite_time_completion_20261016/open_question_and_evidence_map.md)、[P2-OPT2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_full_opt2.md)。固定commit・path・bytes・取得／静的確認範囲は[SOURCE索引](SOURCE_EVIDENCE_INDEX.json)と[一次資料案内](PRIMARY_SOURCE_GUIDE.md)を参照。containing commitと元科学SOURCEを混同しない。

## RQ-07 接続性がPR表現・分割の選択を変える条件

由来：OP-III1。関連研究：CIR-01, PR-01, REP-08, RB-17。

### 1. Problem statement

明示した接続性と同じmapping/routing最適化を全方式に与えたとき、PRのbasis移動・演算子分割・表現の選択は、既知compilerへの単純接続を超える資源差を作れるか。

### 2. Existing bottleneck

主要保存compiled比較はcoupling mapなし。logical RZ/CX差からrouting後のSWAP/depth費用と順位は決まらない。backendの恣意的選択や候補だけの配置最適化で差を作る危険がある。

### 3. Evidence

WP06b、M2、H2四方式はtopology-freeを明記する。C1はraw logical計数に限定。これらは接続性に関する性能positive/negativeを与えない。プロジェクト内の今回確認資料が薄いことは、世界で未研究である証拠ではない。

### 4. Proposed research freedom

物理接続に合わせる生成単位、配置とbasis移動、分割／表現との関係。接続graphやhardwareを今回選定せず、新mapping/routing法を設計しない。

### 5. Strong baseline

適用可能な既知Gaussian/fermionic mappingとrouting/compiler、同じgraph・layout探索budget・control・fusion・精度を持つPauli/DFのPF/PR endpoint。Kivlichanらのfermionic swap network/Givens準備も適用classでは比較候補。ただし同論文の表現前提を任意DF-PRへ自動移送しない。どの既知法を必須とするかはPhase 2Bで判断する。

### 6. Improvement mechanism hypothesis

確認済み：主要保存値はroutingなし。仮説：接続性で増えるbasis移動費が十分大きいclassでは、表現/splitとの協調に価値がある。支配性も同情報baseline後の差も未測定。

### 7. Counterevidence

既知routing・Gaussian共有への還元可能性。RZの改善とCX/depthの悪化、shotsによる相殺。routingなしの結果をhardware性能へ外挿しない。

### 8. Scientific value

PR特有の構造と接続性の関係を示す知見が得られる可能性はあるが、現在は世界的先行研究との距離も未確定。単にbackendを追加する検証との差を明示する必要がある。

### 9. Smallest distinguishing evidence

まず近接する既知routing/Gaussian生成との対応と、接続性が変える具体的な費用項を明らかにする。研究差が残る場合だけ、将来の一つの明示graph・同task・同最適化budgetの対照を検討する。新compile/routingの承認は今回含まない。

### 10. Failure/stop criterion

既知compilerを同条件で適用すれば全差が再現される、又はgraphの恣意的選択にのみ依存するなら新算法課題として縮小する。未測定を有望さの根拠にしない。

### 11. Current readiness

QUESTION_DEFINED / PRIOR_ART_AND_COST_SHARE_UNRESOLVED。Phase 2Bで問題自体の価値を判断する段階。性能資料が揃ったという意味のreview-readyではない。

### 12. Provenance

[EV-0027](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_sequence_policy.md)、[EV-0061](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md)、[P2-BUILD](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/df_rte_qiskit.py)、[P2-REPEAT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/df_partial_s2_repeated.py)、[EV-0179](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/docs/non_df_h2_process_recovery_validation.md)、[EV-0142](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md)。固定commit・path・bytes・取得／静的確認範囲は[SOURCE索引](SOURCE_EVIDENCE_INDEX.json)と[一次資料案内](PRIMARY_SOURCE_GUIDE.md)を参照。containing commitと元科学SOURCEを混同しない。
