# Track C2：候補比較と将来検証の判断材料

2026-11-06 JST。基準 `720416c5b88b227cf76ff7aa729d91a0a98ed58b`。
[REPORT.md](REPORT.md)の目的・確認深度・既存結果・STOPを共有する。
**事実＝固定資料/SOURCE/文献で確認した内容、仮説＝今回整理した未検証の問い。以下の最小検証・STOP条件は判断用の提案であり、採択・実行許可・正式GO/NO-GOではない。新しい数値閾値は固定しない。**

## 候補間の比較

| 観点 | C2-A：round依存PF | C2-B：非一様合成精度 | C2-C：基底energy向けPF |
|---|---|---|---|
| 独立に変える対象 | PF次数/step幅/構成のroundごとの差 | 固定PF・固定測定に対する合成誤差の配分 | 固定された比較条件でのPF構成・選択規準 |
| 既知との重複 | UWC、round別r/K/誤差/失敗配分、variable-precision PE | 既知のsubroutine別誤差最適化、SP、Parent/ALLOCATION、PR残差/PAI | 固有値摂動、D6/支配分枝、P-D、BF/BM、高次/processed PF |
| 想定する費用経路（仮説） | 各時刻に必要な構成だけを払い、信号とNを保つ | 過剰な合成精度を緩め、N増加より大きくTを減らす | 位相biasを下げてstep数/統計予算を改善し、半径と価格も保つ |
| 主な反証材料 | 最終round支配、早期branch誤り、強い固定構成でも同じ利益 | 既知非一様配分への還元、保守誤差では自由度が消える、N/normalization増大 | 強対照との選択一致、BCH既知同値、半径低下・負時間/内部D費が支配 |
| 現時点で欠けるもの | 既知round最適化後に残る具体的な方法差と共通energy保証 | 既知配分を超える具体的機構とcontrolled meanの全誤差資格 | 目的関数の変更を超える具体的機構と有限時間の位相/半径保証 |
| 並列性 | 他を固定して単独に検証可能 | 他を固定して単独に検証可能 | 他を固定して単独に検証可能 |

## C2-A：RPE round依存PF

**既知研究との差。** [固定文献調査のO4][opportunities]は異roundの表現/splitを同じenergyへ結ぶ問題を既に挙げている。[UWC §II.2.4/III.3](https://arxiv.org/html/2603.22778v2)はround別split、round別表現最適化と古典負担を議論する。既存 `rpe_short_round_optimization.py` はq≤4のr/K探索、`rpe_delta_round_schedule_validation.py` は固定PFのδ候補とround別r/Kを扱う。これを再実行してもPF次数切替の新機構にはならない。

**未検証部分・改善機構（仮説）。** roundごとに次数・step幅・構成を変えることで、必要な精度に対するfull価格を下げられる可能性はある。ただし、各構成のeffective eigenenergyは一般に異なる。各roundをそれぞれのPF固有位相へ合わせるだけでは不十分で、全signalが同じ元Hの位相を許容誤差内で示し、初期aliasと分枝更新を満たす必要がある。短roundを安くする効果が最終roundの重さに比べ小さい可能性も残る。既存H4の寄与率を今回再集計していない。

**既存負結果の適用。** C-Nの一致は固定S2/Y4・参照Aの観測価格cohortに関するもので、round切替を検証していない。一方、同じfusionだけを新しい利点とは数えられない。P-Dの負時間・内部D実装とtail負担の警告は高次roundへも関係する。旧run01やP-Dの未完了範囲をC2-A名で埋めない。

**強いbaseline。** S2/Full Y4だけに固定せず、適合する既知高次PFにもstep・r/K・β/αの同じ最適化機会を与えた最良の固定構成を含める。既存round別r/KとUWCの適合する最適化機会も比較に含む。UWCの保存未資格値や外部rotation proxyを、このH4 native結果と直接比較しない。候補だけにE₀/正確なbias/価格情報を与えない。

**最小検証の提案。** GPT／ユーザーが具体的な方法差と有限な既存PFメニューを選んだ場合、まず共通時刻での位相・半径・normalizationと共通energy/branchの対応を確認する。新設計全探索は前提にしない。次に最適固定構成＋既存round最適化を対照とし、全18roundの成功条件とNを接続する。価格はその後、適合するfull-wrapper providerで取得する。q1/2/q8モデルや旧compiler誤差inventoryを別角度/別構成へ無条件移送しない。

**反証・STOP材料。** 既存r/K配分の言い換えに尽きる、共通energy/branch条件を満たさない、許容biasを戻すと強対照と同じ構成になる、または全N×価格の利益が不確かさを越えて示せない場合は、追加の科学的主張を保留する材料となる。長q価格未資格ならUNDETERMINEDで止め、未取得価格を0にしない。

## C2-B：非一様な合成精度配分

**既知研究との差。** [Hänerほか §I–II](https://arxiv.org/html/1807.02336v1)は、合成精度と費用を異なるsubroutine群へ配分する一般的枠組みを既に扱う。「回転ごとにεを変える」だけでは差分にならない。Mehendaleほかの[App.A.7](https://arxiv.org/html/2312.13282v3)にもTrotter/PE/合成の誤差交換がある。以下は近接するが、同じ最適化変数とは限らない。

| 既存方式/資産 | 実際の変数・接口 | C2-Bとの区別 |
|---|---|---|
| SP synthesis placement | cheap branch/通常合成のplacementとsigned QPD、wrapperでのrange・second moment・T | gateごとの決定論的精度配分だけでなくlaw/weightが変わる。primitive削減と必要Nを分ける |
| Parent/S4 | parent-child係数配分、common angle、cost-aware digital proposal、外部angle/native error入力 | `allocation`は子へ配る係数。`root_bits/probability_bits`は数値/生成精度。任意回転のClifford+T誤差εの最適化器ではない |
| ALLOCATION / 再配分会計 | event/angle/proposal、moment、外部誤差の加重和、native plan | 既知KKT/ISと訂正目的を継承する。外部誤差入力APIがあることは、新配分の最適性やRPE保証ではない |
| 旧C-S | coarse回転＋同位置残差RTE、rangeとliteral gate数に応じた合成precision、strict key | 巨大shotは固定coarse/残差policyの負結果。決定論的な非一様合成一般への反証ではない |

**未検証部分・改善機構（仮説）。** controlled full回路内で、異なる誤差感度と離散T価格に対する過剰精度が残るなら、総Tが減る可能性がある。反復頻度が高いだけで緩められるとは限らない。同じ頻度が費用と誤差蓄積の双方に入るためである。新しい配分式やoptimizerは今回導出していない。実際に自由度が残るか自体が未検証。

決定論的合成だけなら通常は別のsigned normalizationを導入しないが、合成bias・半径変化で統計予算/Nは変わり得る。残差randomization/PAIを使う場合は別途range/momentを払い、既存RTEのnormalizationと二重計上しない。同じsystem channelを実現するだけではcontrolled相対位相は決まらない。

**強いbaseline。** 通常合成に最適化した一様precisionと既知の非一様配分をともに含める。双方に同じ角度情報、同じfull回路での厳密相殺、actual adjoint再利用、Clifford判定、同じ合成tool/誤差規約/費用取得機会を与える。残差を使う候補には、PR原著の残差法、SP、適合するPAI/TETRIS等との条件差も必要。旧Parent proxy版や不利な一様設定だけを対照にしない。

**最小検証の提案。** 承認後に既存固定PFを一つ対象とし、full回路に残る角度key・反復数・actual inverseと、既存の誤差会計に渡せる精度を対応させる。まず既知配分との方法差を確認する。controlled joint operator/meanの誤差とphase/radius/shotの予算が閉じる配分だけを同toolの実T価格へ接続する。単なる連続log価格の推定を実T削減と呼ばない。各keyの精度/phase規約が変われば古いcache価格や誤差証拠を流用しない。

**反証・STOP材料。** 既知配分と同じ、共通のworst-case誤差条件では非一様性の利益が消える、strict相対位相が満たせない、または合成削減よりrange/N増加が重い場合は、追加利益を主張する根拠が残らない。旧C-Sの巨大shotを再現するだけの残差policyへ自動拡張しない。T価格・成功条件未資格なら条件付きのまま停止する。

## C2-C：基底エネルギー推定向けPF設計

**既知研究との差。** 演算子normより固有値誤差を見る方針は既にある。[Mehendaleほか](https://arxiv.org/html/2312.13282v3)は固有値摂動を用いてpartition/step選択を扱い、本repositoryにもD6、支配固有位相、branch weight、survival signalの診断がある。P-Dはenergy biasとtail負担、BF/BMは有限model/compact BCHとの対応を検討済み。高次PFの係数を取り替えるだけでも、目的関数をenergyへ変更するだけでも独立の方法差を確認できない。

**未検証部分・改善機構（仮説）。** 同じ実装費用で位相biasを下げ、元H groundの有効な信号半径を維持するPF構成なら、step数や統計予算を介してGを下げ得る。ただし小δのleading energy係数が小さくても、有限時刻の非支配分枝、beat、半径低下がshotを増やし得る。C_useは経験的包絡であり、新設計の厳密上界ではない。無料の真E₀でfit/選択したoracle診断と、実用的に取得できる情報による選択を分ける必要がある。

**既存負結果の適用。** C-Pのwhole-kernel共役はスペクトルを変えないので、これをPF固有値biasの変更と説明できない。P-Dでは内部Dだけ高次化しても外側D/R splitの次数が変わらない点、負時間でtail exposureが増える点を保持する。P-Dの強対照と同じ候補選択、BFの同budget Lでも到達できた保存結果、BMの既知BCH対応は、この候補の直接の反証材料。PR-3と同じbias/variance/backbone交換を新機構として再開しない。

**強いbaseline。** 最適化した既知ordering/partition、S2・Y4・適合する公開高次/processed PF、compact BCH/有限時間補正、P-DのB1b/B2/B4と同情報の選択を含める。exact-D/R二block価格は診断用で、内部fragment実装を省いた対照を実価格とはしない。全方式に元H ground、同じbias情報・step/精度探索・compiler機会を与える。

**最小検証の提案。** 具体的な既知との差が選ばれた場合、まず同じPF列への単なる共役/再記述/compact同値でないかを既存の解析・BCH記録と照合する。承認された小さな既存メニューについて、元H groundの位相bias、支配分枝重みと実信号半径、finite-RTE normalizationを分離して確認する。内部D/外側split、負時間、control scalar phaseも戻す。fitに用いない既存窓の点などで相殺だけに依存しないかを確認し、全roundのNとfull価格を比較する。係数の新規探索や全grid再実行はこの提案に含めない。

**反証・STOP材料。** 既知PF/compact/processorに還元される、同情報の強対照でも同じ候補が選べる、局所energy係数だけの改善で実信号/長roundの半径が悪化する、必要な情報取得や内部D・負時間の価格を戻すと利益が消える場合は、独立の改善主張を支持しない。保証が経験的modelに留まれば、その条件付きを維持して止める。

## 再利用するSOURCEと検証資産

以下の版は**読取/収録版**。全ての旧runのexecution SOURCEをこのSHAへ置換する意味ではない。過去の実行は原報告のSOURCE/inputへ戻る。SOURCEは静的に読んだ関数・接口に限定し、テスト名/pathの存在は実行PASSではない。

| 固定版・資産 | 再利用する機能・対象 | 適用限界／候補 |
|---|---|---|
| [720416c：rpe_delta_round_schedule_validation.py][schedule]、同 `rpe_short_round_optimization.py` | δ screening、round別r/K、`optimize_rpe_short_round(s)`。対応runner/testは同名 | 固定S2/経験的PF係数・短round等の前提あり。任意のround別PF optimizerではない。A |
| [720416c：PF registry][registry]、`pf_decomposition.py` | `_get_w_list`、`iter_pf_steps`。既知S2/Y4/高次の順序・係数 | 旧Moralesと公開係数版、8桁係数を区別。高次化でfinite-RTE/control資格が自動継承されない。A/C |
| [720416c：rte.py][rte]、`df_rte_qiskit.py`、`df_partial_s2.py` | signed finite分布、heterogeneous occurrence、ordered event/control、basis共有、scalar phase | sampled law、LCU係数、weight、raw/corrected meanを区別。共通adjoint機会を全対照へ。A/B/C |
| [720416c：pf_delta_validation.py][phase] | `paper_d6_perturbative_energy_bias`、`validate_pf_delta_grid`、支配分枝/survival診断。`test_pf_delta_validation.py` | D6 small-sine conditioningと旧δ窓を保持。S2検証を任意PFの保証にしない。C/A |
| [f29eafb：P-D module][pd]、同 `research_direction_pd_realization.py`、`research_direction_pd_fair_comparison.py` | `formula_registry`、`composition_tail_weights`、signed control、内外PF、同情報対照。対応 `test_research_direction_*.py` | exact二block参照は高次finite-RTE full回路そのものではない。P-D旧runは再開しない。A/C |
| [4301c0f：BM auditor][bm]、[BF1保存復旧報告][bf] | 形式degree3のexp/log/BCH同値、強対照・元失敗と保存復旧の区別 | 一般定理や有限時間保証ではない。auditor/runnerも今回再実行しない。C |
| [4ccc183：parent_allocation.py][parent]、[546bca7：再配分accounting][allocation] | `digital_proposal`、`event_statistics`、`error_accounting`、`native_plan`、`adjoint_plan`。訂正Parent/S4、外部誤差とproposal会計 | 係数/proposalの最適化と合成εを混同しない。providerやRPE成功の保証は外部。旧承認機構を複製しない。B |
| [720416c：SP wrapper_accounting.py][sp]、同 `economics.py` | `population_profile`、`axis_budget`、`total_t`。E[W²]、max\|W\|、E[C]、E[W²C]、synthesis bias。Parent/SP関連tests | channel/QPD fixtureの会計で、coherent finite-RTE意味論や実分子のNを自動保証しない。B |
| [720416c：track_c_h4.py][pilot-code]、`track_c_h4_finite_bits.py` | `weights/kernel/signal_task/shot_rows`、`integer_cdf/weight_interval/proposal_budget`、固定入力・誤差/shot接口 | 関数呼出しは科学計算を含み得る。今回import/実行せず、保存条件を再利用する将来の接口として確認。A/B/C |
| [720416c：track_c_compiler_a.py][compiler]、`track_c_compiler_a_18round.py`、`track_c_source_guarantee.py` | `compile_reference`、局所inventory/区間誤差、SOURCE→mean、phase/radius/alias接続 | 既存44ケース/q1/全support資格を再実行しない。新角度集合・反復構成の上界は再適合が必要。既にcontrolledなjoint回路をlowerし、後から新controlを付けない。A/B/C |
| [720416c：track_c_h4_synthesis.py][synthesis] | `strict_guard`、`exact_word_matrix`、actual adjoint/合成precision key、T指標 | 旧opt2前処理や巨大残差policyをそのまま再利用する意味ではない。新keyの取得は別許可。B |
| [720416c：track_c_cn_a_price.py][price]、`rpe_resource_accounting.py` | `maximal_groups/strong/moments/account`、paired full/強対照、共分散、軸別N×価格 | 既存会計のS2/単位半径等の仮定と、最新Track Cでの実半径/weight接口を区別。q≤8較正を長q・別recipeへ移送しない。A/B/C |
| [231d17e：uwc.py][uwc]、同 `partial_randomized_pf.py/grouped_uwc_comparison.py` | `UWCConfig/preprocess_qubit_hamiltonian`、sector保存、表現/split/費用の旧接口 | 原著UWC全機能の実装保証ではない。保存rawのSOURCE/state/grouping結合不足とFOUND_UNVERIFIEDを維持。Aの重複確認 |

DF builder/反復、finite-RTE signal、RPE accounting、Parent native等の既存test/pipelineの対応は[プレフライト再利用表][reuse]を継承する。保存correctnessのPASSと新条件の科学性能を区別し、科学計算を呼ぶtestを静的調査の名目で起動しない。

## 並列に進める場合の依存関係

| 将来の独立pilot | 固定して比較するもの | 共有してよいもの／混ぜないもの |
|---|---|---|
| Aだけ | 合成policy・compiler・元H/state・成功条件 | 既存入力と会計接口は共有可能。Aで得たPF選択を後付けでB/Cの独立結果にしない |
| Bだけ | PF scheduleとmean target・測定・compiler | strict key/adjoint規約は共有可能。A/Cの新角度の価格は自動継承しない |
| Cだけ | round運用・合成policy・情報アクセス | phase/半径診断と強対照は共有可能。Cで変更した構成に旧保証・価格を無条件に貼らない |

共通の入力保全とphase/半径/Nの定義に合意すれば、どれか一つの価格grid完走を他候補の前提にする必要はない。相互結合を調べる段階は別の研究範囲。共有環境への変更やAIサブエージェント委譲は行わない。

**次の担当：GPT／ユーザー**が残す問いと具体的な既知との差、主資源指標、比較情報と成功保証の範囲を選ぶ。**Codex**は選択された設計・実装・科学検証について別許可後のみ進む。本書は新しい科学的GO/NO-GOを付けず、2文書の公開照合後mandatory STOPする。

[opportunities]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/RESEARCH_OPPORTUNITIES.md
[schedule]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/src/trotterlib/rpe_delta_round_schedule_validation.py
[registry]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/src/trotterlib/product_formula.py
[rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/src/trotterlib/rte.py
[phase]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/src/trotterlib/pf_delta_validation.py
[pd]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_energy_tail_pareto.py
[bm]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/scripts/tracks/algorithm_codesign/audit_bm05_symbolic_equivalence.py
[bf]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d2645a09440f50e5b869ef42a1b73a1b625a1af/docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md
[parent]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/parent_allocation.py
[allocation]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/src/trottertracks/algorithm_codesign/rte_reallocation/accounting.py
[sp]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/src/trottertracks/algorithm_codesign/synthesis_placement/wrapper_accounting.py
[pilot-code]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/src/trottertracks/track_c_h4.py
[compiler]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/src/trottertracks/track_c_compiler_a.py
[synthesis]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/src/trottertracks/track_c_h4_synthesis.py
[price]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/src/trottertracks/track_c_cn_a_price.py
[uwc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/src/trotterlib/uwc.py
[reuse]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/track_c_o3_technical_preflight_20261106/REUSE_AND_BASELINES.md
