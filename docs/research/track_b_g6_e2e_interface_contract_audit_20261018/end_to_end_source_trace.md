# E1〜E7：入力から測定・会計までのSOURCE trace

〔静的SOURCE読解〕S番号は[source_inventory.json](source_inventory.json)。行番号は記載の固定commitのもの。科学moduleをimport・実行せず、本文・AST宣言位置・参照callを確認した。以下は**存在する辺と不足する辺**の記録で、全chainを動かした証拠ではない。

## E1 DF入力、D/R、identity

`DFHamiltonian`（S11、324–362）はconstant、one_body、lambdas、g_matrices、metadataを持つ。`DFModel → build_df_blocks`（S09/S10）で対角化と`U_to_qiskit_ops_jw`を使い、`DFBlock(U_ops,eta,lam)`となる。`BogoliubovTransform(U.T)`の規約を名前からadjointへ読み替えない。これらの呼出しは今回行わない。

`prepare_df_partial_s2`（S05、328–495）はpartitionを検証し、one-body correctionと指定prefixをDへ、`randomized_block_indices`をtail抽出へ渡す。`explicit_ordered_partition`をweight順へ無断に変えない。`exact_rte_lambda_r`はSOURCE上のfloat成分L1で、fragment ranking proxyとは別であり、名称のexactを元実数Hamiltonianのcertificateとしない。

`extract_df_tail_from_hamiltonian → extract_df_diagonal_tail → exact_df_diagonal_coefficients`（S04、857–904／563–733／529–556）。既存のI/Z/ZZ展開は`np.dot/np.sum`等の浮動小数点演算。関数名にexactがあっても有理数による係数生成ではない。係数thresholdを適用した後、`faithful_identity_in_tail`か`extract_identity_phase`を処理する。抽出identityをthreshold dropに混ぜず、constantも別保持する。

`DFDiagonalPauliComponent`はcomponent ID、float c/abs/sign、fragment、support、basis ID/hashとoperationsを持つ。`DFBasisRegistry.register`（80–155）は同IDの競合、wire、hashを検査するが、hash一致だけでは元DFへの近似誤差・Gaussian性・strict native certificateは得られない。

`extraction_to_symbolic_rte_tail → prepare_df_rte_event_inputs`（S04、808–854／1086–1099）はfloat p=abs(c)/lambdaとcircuit specsを対応付け、`DFRTEEventPreparation`（S08、180–231）が成分順・sign・support・basisを検証する。ここからG6の正確な有理p/xと入力誤差証拠へ進むbindingは未実装。Λ_R=0は既存deterministic-only経路であり、空pのG6を作らない。

## E2 有理入力・有限bit条件

数式対応は、保持成分に対するΛ_R=Σabs(c_a)、Q_a=sgn(c_a)T_a、p_a=abs(c_a)/Λ_R、x=Λ_R abs(h)、σ=sgn(h)。xは無次元、hは時間、Λ_Rはエネルギー。元raw DF、float抽出成分、G6有理target、実native回路の各層を区別する。

`ReturnKernel.__init__`（S01、138–160）は`Fraction`への変換、p>0、正確なΣp=1、奇数m、0≤x≤1を検査する。floatを`Fraction`が受け取れることは元実数との誤差保証ではない。既存float pを個々にcastしても、正確なΣp=1を保証しない。`isinstance(m,int)`はboolを区別しないため、将来の接続境界には既存RTEの`require_integer_count`に相当する厳格な型条件が要る。今回その実装・テストをしない。

理論の主領域は0<x≤1、m≥1 odd。x=0はidentity fast pathであり、`digital_parent`は拒否する。L=1/m=1は既存形式testの対象。ordinary even cutoff K_evenに対しm=K_even+1は同じfinite targetの対応であって、law・angle・weightの一致ではない。root_bits K_rootとK_evenを同じ記号で実装しない。

S01のisqrt enclosure・dyadic処理とS02の固定H/K、eta/rhoがある。自動H/K selector、元DF誤差の証拠、一般的なangle enclosureのSOURCE接続は未完成。M10の存在論証をselector実装済みとしない。

## E3 Local query → event/None

`ReturnKernel.__init__ → word_series/_mass → parent → digital_parent → rational_event_weight`（S01）と、`FullReturnGenerator.sample`（S02、237–251）が接続される。後者は次数dyadic → raw IID labels → rawが非reducedならNone → local acceptance → child → eventである。事前zeroをreduceして別parentへ再割当てしない。

eventは`word,raw_word,child,ratio,phase_i_power,coefficient,proposal,weight,provider_calls,helper,event_unitary,circuit_time_word`を持つ。`coefficient`はc̃、`proposal`は実dyadic π、`weight=c̃/π`。`ratio=s/a`は有理数、wordはoperator左→右、`circuit_time_word`はreverse。σはeventの実fieldではなくprovider呼出し引数／run文脈にある。

Noneは非reduced rawと受理coin拒否の双方に使う。失敗例外とは異なるが、現在のNone単体は拒否理由・trial ID・入力identityを保持しない。`DeterministicBits`は再現診断用SHA streamで、fresh-IIDの実証ではない。G6局所packet、G7 event記述の成功と、DF circuit実行・測定の成功は別である。

## E4 RTEEventへの接続：構造の近さと明示的なguard

S06 `RTEEvent`（549–643）は、偶数product長、末尾一rotation、phase=(-1)^(l/2)、float probability/coefficient/normalizationの関係を検証する。constructor自体は有限floatのword依存angleを禁止しない。しかし実dyadic有理数、W、root/bit証拠、σ文脈、zero/failure状態の専用fieldはない。全意味論を保存する形式へのbindingは未実装。

ordinary `_make_event`（1594–1659）は次数angle=`atan(tau/(n+1))`、IID確率、共通B_oを作る。さらにS05 `DFPartialS2StepRequest.__post_init__`（499–643）は、全eventに対し、**共通normalization、次数だけで決まるangle、ordinary IID event_probability**を再検証する。S13のrepeated requestもこのone-step validatorを再利用する。

したがって「RTEEventの同名fieldへ入れれば現行PR wrapperも受理する」は支持されない。wrapper契約の分離／拡張やsidecar等の選択は未採択。既存guardを弱める修正を提案・実装しない。

## E5 Controlled circuitとnative

S07 `QiskitDFRTEEventCircuitBuilder._build`（214–391）はapplication列を時間順に保ち、同一basis ID/hashの隣接境界だけを共有する。`_append_basis`（448–458）はgate順を反転して実inverseを使う。`_append_central_application`（472–502）はZ/ZZのproductをCZ列、rotationをcontrolled RZ/RZZへ下ろし、productのsignはancilla phaseへ戻す。

G6のV=(−iσ)^l exp(−iσφQ_child)Q(u)では、reverse(u)を時間順に実行し、最後にchild rotationを置く。parent phase、product sign、child sign、σ、constant/extracted identity phaseを別々に保持する。rotationでsignをangleへ入れた後にproduct位相として二重加算しない。既存の定数位相経路はS05 `_append_step_phases`（875–891）。

S03のhelper IRは別の既存経路で、exact controlled-Qとactual adjointを仮定し、外側control＋helperを使う。DFの直接共役回転はhelperを要求しない構造があるが、両者を同じworkspace/costとして扱わない。どちらを接続するかは未採択。

S17 `NativePipeline.step`はS03のbindを**σを渡さずdefault+1で**呼ぶ。従ってこのpipelineを負時間も含む汎用DF routeとして再利用するにはsigned-time bindingの条件が不足する。旧G8の正時間契約の誤りを認定したものではない。G9 S19は固定3-system-qubit provider例であり、一般DF registryのproviderではない。

## E6 Hadamard・backend・全trial

S14 `RPEHadamardInterrogationRequest`（58–74）は`DFPartialS2RepeatedCircuitResult`を要求する。builderはcontrolled trajectoryにH、sineではS†、H、測定を付け、bit b→(−1)^bを使う。public entryのqは1/2/4に限定される。G6 event辞書やG7 helper付きIRをそのまま渡す一般wrapperではない。

同builder resultは`state_preparation_included=False, backend_execution_included=False, quantum_shots_executed=0, fresh_iid_trajectory_per_hadamard_shot_verified=False`を明示する。S16 compiled-costも測定命令込み回路の**古典的構築・compile**で、量子測定の実行記録ではない。

S17はattempt/accepted/zerosを数え、Noneならcache・provider・readoutを呼ばないが、`quantum_measurements=0`。重み付き測定accumulatorではない。監査対象SOURCEでは、G6のaccepted回路→backend bit→実W→全trial平均の接続を確認できない。確認範囲内の`MISSING_INTERFACE`とし、repository全体・外部環境に一切存在しないという不在証明にはしない。

## E7 Error/cost/PRへの戻し

S02のU/budget、S18の有限provider仮定とresource-failure予算、S15のordinary RPE会計はそれぞれ存在する。しかしS02の固定confidence、S15の±1 outcome/ordinary attenuation・phase仮定をG6へ自動コピーできない。S13はB_oの反復attenuationを保存するが、G6の補正済み全trial meanへさらに同じ補正を掛けない。

input、Taylor、RA、coefficient、proposal、angle、native、統計、PF/multi-stepを[誤差契約](precision_and_error_contract.md)へ、量子前拒否とevent依存費用を[測定・費用契約](measurement_and_cost_contract.md)へ分けた。新しい上界、shots、費用、分子値は計算していない。
