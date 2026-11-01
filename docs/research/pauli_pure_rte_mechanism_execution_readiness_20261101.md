# Pauli純RTE原因分離：実行契約案・再利用対応表

2026-11-01 JST。**状態：DOCUMENTATION_ONLY / DRAFT_NOT_EXECUTABLE。科学実行の承認・新SOURCEの固定ではない。** [検証設計](pauli_pure_rte_mechanism_validation_design_20261101.md)のA〜Dgrid、F/R/E/Empty、M_A〜Cの定義を用いる。本書の数値は既存保存値の引用または将来の限定call範囲の提案であり、今回の再計算ではない。

## 1. 今回許可された作業と実施記録

既存Git・GitHubファイル、SOURCE hash、保存JSON schema、archive索引と選択member、関数・testsの静的読取、2文書の作成・検査・公開だけを行った。新信号・bias差分・normalization/N/G・統計・ランキング・sampling・simulation・Hamiltonian/state/reference生成・回路build/compileは0件。科学モジュールや科学testsは実行していない。

2026-10-25正式共通運用指示§8.1/8.4/10とprojects/AGENTS.mdを適用する。承認はユーザーの対象scopeへの許可であり、承認原文保存・認証・証明packet・grant発行を新規要件にしない。旧個別契約のSOURCE lock、使用済みgrant、one-shot、retry、STOPは保全し、その実行権限を新研究へ移さない。サブエージェントは使わない。

起点は棚卸し公開commit 9a0bc7f17f47823e46c3945734127e9f0d09154c。元main・研究branch・研究概要・manifestを更新せず、新しい独立branchへ本書と設計書だけを追加する。[元棚卸し][LAND]、[U01〜U04][GAP]。

## 2. 固定SOURCE・結果・実入力の対応

### 2.1 H6 Pauliの初期4候補

| 役割 | 固定commit／保存先 | 実際に使うもの |
|---|---|---|
| Hybrid実行SOURCE | 887b69e449cb3c378286f09d0d7626599850fb67 | Pauli adapter、既存law/control、H6限定接続、tight L2、worker。[SOURCE JSON][H6SOURCE] |
| Hybrid解析SOURCE | af891fd45805454c6592bbee5441266f14235ffd | 既存CANDIDATE_RESULTSと確認会計の出所。再実行しない |
| Hybrid結果 | 00958ae1945c780718b2deef07d5dab1096a5cd8 | p78/q8/r16,32,64/K2を本計画で使う。元第4候補p684/q1/r16/K4は除外。[保存表][H6CSV] |
| RTE実行SOURCE | e755ecf581600192fd2745d7066e410cebd933d3 | L0/q8/r64/K2のlaw・tight target・audit・central回路。[実行SOURCE][EVSOURCE] |
| RTE結果 | 10cb3e5cca7834ebf75301bd91bd093ca51d4a6b | H6_p000_q8_r64_K2、確認n48を引用。[報告][EVENT] |
| model | 同RTE結果の artifacts/pauli_h2_h6_comparison/2026-10-25/H6/inputs/model.json | Hybridがコピーしたものと同じ保存byte。12qubit・918項、all_Z_prefix=78。[原本][H6MODEL] |
| reference | 同上H6/inputs/reference.json | raw stateのDuhamel参照、exact_eigenstate_assumed=false。[原本][H6REF] |

既存model SHA256は 6c66a2eeb4d99cf5bbd9137bbcb067275536be3201d5cc047225266dc5f58681、referenceは 9956466a5cf3286af9960b792b455bb8bacc2532a34ac7a28b470475f92fcb7a。Hamiltonian hashは 37b2747eaafacef8b4b470e5f4d6e195ac0602dc8e59953e3b76152d4576c847。この3種のhashは意味が違い、承認証明ではない。

Hybrid SOURCE.jsonのcode18 pathと、RTE SOURCE_e755…jsonのcode25 pathを、それぞれ記録された科学SOURCEのGit file bytesとSHA256照合した（重複pathあり）。記録済みH6入力hashも共通である。これは現HEADの同名コードを実行版とみなした照合ではない。

**現起点HEADにはH6 Hybrid限定moduleとそのworkerが存在しない。** またDF長時間signal/nativeもこのHEADにはない。必要な版は固定commitから参照する。古い実行時の自動抽出package hx_study とA.extractの相対path依存もあるため、将来の隔離実装でimport closureを明示し、同名moduleを偶然拾わない。[H6接続][HX]、[worker][HXSCRIPT]。

### 2.2 圧縮証拠の所在と確認したschema

Hybrid公開data_archives.jsonはH6_evidence.tar.xz.part000/001/002を列挙する。3partと連結archiveの保存SHAを確認し、次の4memberのJSON bytesを索引hashと照合して読んだ。QPYをdeserialize/build/compileしていない。[索引][H6ARCH]

~~~
H6/inputs/model.json
H6/inputs/reference.json
H6/batch01/cases/PR_p0078_q8_r32_K2/spec.json
H6/batch01/cases/PR_p0078_q8_r32_K2/prices/confirmation/0000/result.json
~~~

spec.json内にlawがあり、retained78ラベルはI/Zのみ。resultにはcompact events、body_phase、両軸metrics、input_qpy/compiled_qpy hashがある。control_sidecarは保存resultのfieldにない。r16/64のspec・target・両軸QPYも索引上に存在するが、今回全memberの復元を再検証したとはしない。

RTEはcases_evidence.tar.xz.part000の別archive。選択した次の4memberを索引hashと照合して読んだ。[索引][EVARCH]

~~~
cases/H6_p000_q8_r64_K2/spec.json
cases/H6_p000_q8_r64_K2/law.json
cases/H6_p000_q8_r64_K2/target.json
cases/H6_p000_q8_r64_K2/prices/confirmation/0000/result.json
~~~

RTEのlawはspec内ではなくlaw.jsonにある。target.jsonはphysical_corrected_mean、raw_norm_exact、B_applied_count=1、state_renormalized=falseを持つ。eventはfull application_sequence schema。specのNOT_EXECUTED_AWAITING_PERMISSIONは準備時点の記録として残っており、それだけから未実行と判定しない。完成target・SOURCE・報告・終了状態と対応させる。

将来QPYを復元する場合は既存reversible QPY-aware archiveのencodingを守り、元gzip byteのSHAとarchive内展開QPYのSHAを区別する。選択memberだけを新しい専用作業域へ読み、保護rawへ上書きしない。今回全QPYや全cohortを復元・再認証したわけではない。

### 2.3 DF・Mechanism・旧H4の固定版

| 系列 | 科学SOURCE／結果 | 調査で確認した対応と限界 |
|---|---|---|
| DF長時間signal | e417ffc9f52b13c18f5493193ce510d858664259 | h46_long_time_signal.pyのSHAが155529e2のscientific_provenanceと一致。[SOURCE][DFSIG] |
| DF長時間native | 27ebabdd2943a4de787ec8b976c38e06fda5a497 | h46_long_time_native.pyも同provenanceと一致。[SOURCE][DFNATIVE]、[来歴][LONGPROV] |
| DF B2保存結果 | 155529e20a8fac39e00786c40b57830b71203c4f | H4/H6 tol-only DF10/19。A/C/Dの保存fieldを確認。新しい分解値の算出なし。[records][DFRECORDS] |
| DF B3公開結果 | ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed | 一体保持・二体L0。未公開local最終f794a2aaとの同一視なし。[報告][B3]、[provenance][B3PROV] |
| Track A説明 | b6716396acc5760fde7cf7cc6f2481b183a08f10 | mechanism_explanationの固定版と現起点byte一致。normalization/headroom検査の数理・schemaを参照。[code][MEX] |
| A82後続因子 | 06c94745f50345b2d93714760608d9d11b83e784 | mechanism_a82_factorsも固定版と現起点byte一致。32比較64軸完成を継承。[code][MFACT]、[最終結果][MFINAL] |
| 旧H4共通入力 | 0c329b30a81accb35c212be8defa66fee0dd0f12 | h4_molecular_inputs、Pauli/DF構築・normal-orderingを静的読取。[input SOURCE][H4INPUT]、[DF補正][DFH] |
| D1後続 | 9843b803878db6c3aee7ab5e7c6bb4818572c944 | generic部品9/9測定の出所。D1 adapterは再利用先887b69e4版とbyte一致を確認。純粋な復元・式の部分だけを参照。[実行報告][D1] |
| D2 | 74515d8eb4f073c40fc783681a3775dbca89e9b1 | central control対照の科学SOURCE。control helperはH6実行版887b69e4の同fileとbyte一致を確認。[実行報告][D2]、[helper][CTRL] |

DF provider SOURCE c31ef424175c02379eafebe8fc6c7548c1d9aabeと外部入力・occupation-sectorの依存も記録される。DFのcontext/runnerを呼ぶと別時間のreferenceや多数候補を新評価するため、保存値の読取と純粋なkernel再利用を分ける。元private全event/checkpointは今回取得しない。

## 3. コード・関数・testsの再利用対応

| 機能 | 既存関数・runner・tests | 適合する範囲／必要な最小差分 |
|---|---|---|
| Hとsplitの定義 | PauliHamiltonian.sparse_operator、PauliSignalAdapter.operators [ADAPTER][ADAPTER] | 保存binary64係数・little-endian・constant分離を再利用。Bは全保持和の指数、Cは元half sweepであり切替を小さく追加 |
| ideal Dpoly | sparse_mean_actionのdegree0…K+1 recurrence [ADAPTER][ADAPTER] | 既存戻り値はideal normで割ったraw経路。physical polynomialを出す診断modeが必要。単に保存Bを掛けるだけではDgridと一致しない |
| effective Dgrid | law_descriptor/probability_masses [LAW][LAW]、tight evaluate/event/rotate [TIGHT][TIGHT] | 初期は保存target再利用。CDF53bit masses・binary64 angle・scalar・B onceを維持。旧r≤8 validatorを直接広げない |
| H6候補接続 | Hybrid config/mean_target [HX][HX]、pureRTE h6_mean [RTEEXT][RTEEXT] | 元fixed domainのまま参照。将来diagnostic専用entryへB/C/Dpolyを足す。旧STUDY_STOPを消さない |
| DFの診断 | evolve_batch/context [DFSIG][DFSIG]、旧finite_rte_signal_validation [DFOLD][DFOLD] | A/C/D列・閉合・signed stageが参考。DF block/sectorとPauli項は交換不可。旧context実行は含めない |
| B型toy | finite_rte_phase_amplitude._condition_record [TOYB][TOYB] | 正負時間、対称／非対称、exact h_d/tailの小系oracleとして再利用。H6実データのBではない |
| N/C/G説明 | mechanism_analysis、mechanism_explanation.normalization/explain_side、A82因子 [MEX][MEX]・[MFACT][MFACT] | 既存の会計因子と複素誤差因果は別。DF empirical marginをPauli包含uへ転用しない。旧grant制御runnerを起動しない |
| shot会計 | h4_conditional_cng.shot、h6_rte_fairness.account/scale（H6 SOURCEの依存） | ε_axis、α=.025、guard、整数政策を維持。初期必須batchにN/G再評価は含めない |
| event replay | D1 event_from_record/grouped_events/build/interactions [D1ADAPTER][D1ADAPTER] | full event schemaと順序・phase処理を再利用。Hybrid compact schemaのtoken復元とcentral/両軸partsが必要 |
| central制御 | central_control_body、_tokens、full wrapper helper [CTRL][CTRL] | basis/parityを非制御にしcentral rotation/product/aggregate phaseを制御。D2 full_wrapperはcosine限定なので既存両軸wrapperを使う |
| 保存価格/QPY | price_jobとqpy保存経路 [PRICE][PRICE] | 既存F metrics/input/compiled QPYを利用。旧price_job自体は抽選するのでreplayに使わない |
| DF/Pauli matched | h4_molecular_inputs.spin_lift_unthresholdedと保存tensor [H4INPUT][H4INPUT] | 係数規約の参考。元SCF/JW/DF builderを再実行せず、将来の明示scopeで保存tensorから共通T/Vを定義 |

既存testsは静的に確認しただけで、今回実行していない。将来の最小実装に対応させる候補は以下。

| 検査 | 既存testの再利用 | 必要な追加correctness |
|---|---|---|
| retained/event/phase/norm/B | test_retained_event_phase_exact_norm_and_B_once_enclose_oracle [TESTMEAN][TESTMEAN]、test_raw_ray_and_B_once [TESTVECTOR][TESTVECTOR] | 同一toyでA/B/C/Dpoly/Dgridの複素閉合、ν二重補正・B二重適用の拒否 |
| 可換／非可換 | H6 toy body/meanの既存fixture [TESTHX][TESTHX] | I/Z保持ではB=C、非可換保持ではB/Cを誤って同一視しない、L0ではB=C=A |
| tail exact／polynomial | DF long-time signal tests [TESTDF][TESTDF]、既存B型toy | K0/2、負時間、scalarあり、forward/reverseの順序、raw/corrected閉合 |
| control・位相 | test_noncommuting_order_is_preserved、test_products_negative_and_aggregate_phase、test_pure_scalar_phase_including_wraparound [TESTCTRL][TESTCTRL] | partsのscalar/event所有、F一致、L0 scalar-only R、sine Empty |
| 保存eventと部品費 | test_event_roundtrip_and_order_rejection、test_interaction_signed_counts [TESTREPLAY][TESTREPLAY] | compact/full双方のschema、元event/angle不変、sampling禁止、共有R/Emptyの同一性 |

人工検査を実分子の新しい科学runへ拡張しない。H6の実propagation、回路build、compileは人工検査とは別の科学callである。

## 4. 実行契約案：選べる限定scope

**いずれも未承認・未実装・未実行。** 将来の実装commitは未作成であり、今ここで実行SOURCEを封印しない。下記をまとめて許可する指示は可能だが、許可されなかったpackageを自動実行しない。

### P1：H6 bias分離（第一候補）

| 項目 | 契約案 |
|---|---|
| 科学的な問い | p78/q8の保持PF・外側分割・finite polynomial・effective lawのどこに複素biasが残るか |
| 固定入力 | §2のmodel/reference/raw state/exact ν、T=.8、q8、prefix78または0、K2、ε_sig=.0005 |
| 固定4候補 | H16/H32/H64/R64のみ。prefix・q・K探索、H2/H4、DF新評価は含めない |
| 再利用 | A、4 Dgrid target/law、既存B/N/C/Gの表示値。保存target再実行なし |
| 新科学処理 | 保存Pauli項からH_D/H_Rの部分作用素を構成し、共通B/C各1、Dpoly4、複素Δ_outer/hold/trunc/gridと閉合・headroom関係を評価 |
| 数値的な実装案 | 既存NumPy1.26.4/SciPy1.14.1のcomplex128（中心53bit）sparse action＋Pauli recurrence/rotationを初期診断案とする。scalar/coefのbinary64値を入力として固定。DgridのArb128bit scalar/53bit vector包含と保証を混同しない |
| 新信号scope | 主信号6種類。共通B/Cに各1回のtail-action整合cross-checkを含めるなら伝播実行最大8回。個々のexpm/matvecはこの固定伝播を構成する内部処理として記録する |
| cross-check | 同じtail時間を二つのhalf actionへ分けるsemigroup整合等。これは独立した精度証明ではない。1e-10は既存診断由来の整合目安で、rigorous errorではない |
| 判定 | B=CおよびL0 B=C=A、複素閉合、両軸の符号、global phase、ν/Bの回数。差が数値分解能を超えなければUNRESOLVED |
| 範囲外 | 新A/full-H reference、SCF/FCI/JW/DF生成、sampling、compile、N/G再評価、パラメータ再探索、全lawρ証明 |
| STOP | 入力／law／SOURCE不一致、数値非有限、閉合・位相・norm失敗、精度不足。追加precision/新対象へ自動retryしない |

1e-10より小さいΔや桁間相殺を解像するために、rigorous exponential-action remainderや高精度vectorが必要になる場合は、対象とcall数を改めて範囲化する。初期P1は数値機構診断であり、B/C/Dpolyへ保存Dgridと同じ包含保証を付けた契約ではない。ゼロに近い差は有限bit/数値誤差と区別できない場合がある。

### P1N：条件付きshot感度（P1とは選択可能な追加範囲）

既存のDのB/α/ε_axisを固定し、誤差成分を符号付きで除いた仮想biasを元の十分shot式へ戻す。対象は4候補・両軸・ε=.0005だけ。変更biasの不確かさ政策が先に必要である。新しいNの評価は保存値の引用ではなく科学的再解析なので明示許可を要する。

出力はdiagnostic counterfactual Nと採用したu/ρ仮定。B/Cの実装を持つ別量子推定法のshot保証やG順位は作らない。P1が解像不能ならP1Nを進めない。

### P2：H6 central部品費（第二候補）

| 項目 | 契約案 |
|---|---|
| 問い | 保存Fの保持・event・結合時compiler応答に、操作的な価格差があるか |
| 対象 | H32/H64/R64、各confirmation index0000。価格に基づき代表を変更しない |
| 固定するもの | event列・順序・元angle/sign/order phase、q/r/K、全qubit、scalar所有、両軸full wrapper、compiler profile、測定境界 |
| 再利用 | 既存Fの6価格とinput/compiled QPY、元result metrics。QPYの復元・意味対応を先に検査 |
| 新実装 | 保存event→元SOURCE token→F/R/E/Empty。generic D1ではなくcentral/両軸へ接続。元SOURCEを変更せず診断接続を追加 |
| 新compile | R4＋E6＋Empty2の最大12program。Hybrid r32/r64のRと各軸Emptyは共用。同一入力はcache再利用 |
| 新sampling | 0。seedからのRNG replayも0。保存eventが不足する場合はSTOP |
| 出力 | 各arm RZ/CX/depth/histogram、命令対応、QPY、位相所有、RZ/CXの相互作用残差・contextual increment |
| 判定・STOP | Fの意味・入力binding・価格境界が一致しなければ旧F価格を使わずSTOP。部分回路の精度やNは評価しない |
| 範囲外 | F全cohort再compile、pass別ablation、新たな候補・母平均/tail推定、元Nを掛けたG |

F inputの元SHAは読み取り不変を確認する。再構成QPYのserialization名などの違いと、命令・角度・phase・compiler条件の不一致を区別する。ただし科学的に必要な固定input契約を黙って緩和しない。不一致を通すために元証拠を書き換えない。

同一候補内の4arm比較は同じ保存programのpaired診断だが、異なるfamilyのevent分布を同じlawのpaired標本とは呼ばない。1programの差を全law期待価格へ移さない。母平均を求める追加samplingのnは未設計であり、既存SE・層確率・判別精度から必要になった場合だけ別途決める。

### P3：H4 DF/Pauli matched分割（第三候補・現時点で実行条件未確定）

H4旧共通入力をanchorにする。M_A全非identity、M_B同じTを保持したPauli residual、M_C同じT/VのDF residualを比較する構想である。**初期P1/P2に含めない。**

実行前に未解決の選択は、T_eff/V_squareかT_bare/V_normalか、共通保持blockの実装、identity移管、同精度・shot会計の政策、対象q/r/Kと代表trajectoryである。これらが未定なので、今回P3の科学call数や性能判定表を作り込まない。

H4について静的に揃ったものはgeometry/basis、元MO積分の生成規約、同じ保存state配列、8qubit little-endian、旧S2/T.8/q8、旧compiler/全wrapper境界である。一方、以下は別gateとして残る。

| gate | 保存証拠で分かること | 新科学評価／実装が必要なこと |
|---|---|---|
| full H | receiptにtol-onlyDF10とPauli184の数値一致がある。[H4REC][H4REC] | 新しい共通T/V定義での全H再構成・誤差評価 |
| one-/two-body | builderでone_body_correction吸収を確認。[DFH][DFH] | TとVそれぞれの演算子一致。prefix変更だけでは代替不可 |
| scalar/identity | Pauli constant分離、DF tail identity抽出は既存 | 一度だけ移管する共通規約とphase検査 |
| state/reference/norm | 両Cモデルのstate配列・hash同一。[H4PM][H4PM]、[H4DM][H4DM] | 表現間のprecision allowanceやreference区間を同じ会計へ接続する場合の評価 |
| sector | 保存36次元spin sectorとfull256次元index規約がある | 新しい各armの全中間演算のsector不変性。無断の射影なし |
| price | generic D1とcentral D2は別系列として保存 | M_B/M_C共通保持block、同じcontrol/compile境界の新価格 |

H4 legacyDF12や長時間DF10の別snapshotを混ぜない。H6 weighted-Hermitian completionと別Pauli生成入力をこのanchorへ統合しない。数値的一致を確認しても、M_B/M_Cの辞書・law・normalization・angle・回路差はなお交絡する。

## 5. 将来保存する最小の成果物とprovenance

新しい実行日・専用namespaceを使う案とし、現在は作成しない。

~~~
artifacts/pauli_pure_rte_mechanism_validation/<execution-date>/bias/
    source_and_inputs.json
    signals_and_signed_differences.json
    numerical_checks.json
    REPORT.md
artifacts/pauli_pure_rte_mechanism_validation/<execution-date>/cost/
    source_and_inputs.json
    <candidate>/confirmation_0000/<arm>/<axis>_{input,compiled}.qpy
    metrics_and_interactions.json
    REPORT.md
~~~

biasにはφ/νの出所、Aのraw/physical規約、Dgridのmemberとtarget精度、各信号のalgorithm/precision/error status、複素中心と区間または不確かさ、B適用回数、SOURCE/input/law、全scientific action件数・失敗を保存する。報告だけで足りる場合、巨大な全中間行列・全vectorの重複コピーは不要。

costには元archive/member/hash、cohort/index/eventとangleの対応、scalar/event phase所有、full再利用根拠、compiler profile、RZ/CX/depth/histogramとQPYを残す。平均ではないこと、sampling0、ablationのN/Gなしを記す。可能な重複R/Emptyは入力program hashで共用する。

保存SOURCE/inputのhashは科学的再現性のために使い、承認文のhash・証明書・専用ledgerを作らない。実行SOURCE commit、対象入力、結果、実際の数値精度・runtime、必要なSTOPだけを記録する。過去のgrant/markerは読み取り保全し、新しい承認証明の基盤へ拡張しない。

## 6. 明示許可の対象と残る準備事項

今回の公開は以下の許可を含まない。将来実施するなら、ユーザーは必要なpackageと次を具体的に指定できる。

- **P1**：4候補の最小実装・人工検査、保存A/Dgridを使うB/C/Dpolyと差分の新科学評価、数値診断の精度と上記最大伝播scope。
- **P1N**：P1とは別に、新しい条件付きN感度を求めることと、そのu/ρ仮定。
- **P2**：同じ保存3trajectoryのparts build、両軸最大12compile、保存F再利用、相互作用差の新評価。
- **P3**：T/V定義・保持block・対象条件が具体化した後の別科学scope。H2/H4 bias移送やH6 DF統合は別途必要性を判断する。

承認に証明書やgrant発行は不要。同じ明示scope内の最小実装・人工検査・debugと実行をまとめて許可することは可能であり、処理段階ごとに再承認する仕組みは設けない。将来作る科学SOURCEは実装差分とtestを確認してからcommitを記録し、承認範囲とinputに結び付ける。今の文書commitを未実装の科学SOURCEと扱わない。

実行前に解消すべき技術事項は、(1)B/C/Dpoly診断接続の実装、(2)新信号の数値精度・誤差statusの明記、(3)compact/full event双方からの保存program復元、(4)Fとの位相・命令・compiler一致、(5)旧packageの相対path依存の隔離である。H16/H64の必要memberも、実行時は選択memberだけを復元照合する。今回未取得の全law実ρ・母平均価格tailをP1/P2の暗黙の成果にしない。

資源へ恣意的なCPU/RAM/worker上限を置かず、固定入力・共通B/C・共有R/Empty・保存target/priceの再利用と独立taskの並列化を使う。共有OS、scheduler、driver、共有環境、他ユーザーのfile/processを変更しない。旧実行のresource数値は履歴として保持し、今回の新規scopeの恒久上限にしない。

## 7. 文書検査・公開・mandatory STOP

既存の文書専用one/source_rows/check_linksを再利用して、保存行の文字列、引用commit/path、相互リンク、変更範囲を検査した。固定参照59件のGitHub blob識別子をlocal Gitと照合した。大型JSONはGit data blob経路、binary signal recordは固定raw URLのbyte照合を使用し、contents APIの空本文を欠測と取り違えていない。科学的な式の数値評価、source moduleのimport、science tests、runnerは呼ばない。この検査は主agent自身の技術照合であり、独立科学レビューではない。

公開対象は本書と設計書の2資料だけ。既存SOURCE/input/raw/result/grant/ledger/marker/STOP・元branch・main・論文は変更しない。科学実行SOURCEの新規固定、grantの発行・消費、旧STOP解除は0。GitHub公開後、remote branch SHAと2本文を照合し、結果は最終チャットへ報告する。

**公開・照合後mandatory STOP。次の担当はユーザー/GPT：必要な科学的判別とP1/P1N/P2/P3の実施範囲を判断し、実施する場合だけ明示許可する。Codexは新科学計算・実装・compile・詳細科学レビュー・主要claim採択・論文改訂へ自動移行しない。**

<!-- 固定結果carrierと科学実行SOURCEを本文で区別する。 -->
[LONGPROV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/scientific_provenance.json
[B3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md
[B3PROV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/scientific_provenance.json
[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_d1_source_execution_20261024.md
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_d2_execution_20261024.md
[H6SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/SOURCE.json
[H6CSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/CANDIDATE_RESULTS.csv
[EVENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/docs/pauli_rte_event_extension_20261026.md
[LAND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_scientific_landscape_20261101.md
[GAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_claim_evidence_gap_matrix_20261101.md
[ADAPTER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/pauli_signal_adapter.py
[LAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/h4_pauli_pr_search.py
[CTRL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/h4_d2_pauli_control.py
[HX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/h6_hybrid_fairness_extension.py
[TIGHT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/pauli_vector_tight_enclosure.py
[D1ADAPTER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/h4_d1_adapter.py
[RTEEXT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e755ecf581600192fd2745d7066e410cebd933d3/src/trotterlib/pauli_rte_event_extension.py
[PRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/scripts/pauli_h2_h6_study.py
[HXSCRIPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/scripts/run_h6_hybrid_fairness_extension.py
[H6ARCH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/data_archives.json
[EVARCH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/data_archives.json
[EVSOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/SOURCE_e755ecf581600192fd2745d7066e410cebd933d3.json
[H6MODEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_h2_h6_comparison/2026-10-25/H6/inputs/model.json
[H6REF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_h2_h6_comparison/2026-10-25/H6/inputs/reference.json
[DFSIG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e417ffc9f52b13c18f5493193ce510d858664259/scripts/h46_long_time_signal.py
[DFNATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/27ebabdd2943a4de787ec8b976c38e06fda5a497/scripts/h46_long_time_native.py
[DFRECORDS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/signal_records.json.gz
[H4INPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h4_molecular_inputs.py
[DFH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_hamiltonian.py
[TOYB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/finite_rte_phase_amplitude.py
[DFOLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/finite_rte_signal_validation.py
[H4REC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json
[H4PM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json
[H4DM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/df_native_model.json
[MEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6716396acc5760fde7cf7cc6f2481b183a08f10/src/trotterlib/mechanism_explanation.py
[MFACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/06c94745f50345b2d93714760608d9d11b83e784/src/trotterlib/mechanism_a82_factors.py
[MFINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/resource_applicability/track_a_mechanism_a82_factor_results/2026-10-19/integrated_report.md
[TESTHX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/tests/test_h6_hybrid_fairness_extension.py
[TESTMEAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/tests/test_h6_hybrid_fairness_extension_mean.py
[TESTCTRL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/tests/test_h4_d2_pauli_control.py
[TESTREPLAY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/tests/test_h4_d1_adapter.py
[TESTVECTOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/tests/test_pauli_vector_enclosure.py
[TESTDF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/tests/test_h46_long_time_signal.py
