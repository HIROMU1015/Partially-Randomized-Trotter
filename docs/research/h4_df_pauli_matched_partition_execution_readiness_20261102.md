# P3：H4 matched分割の実行準備・再利用対応表

2026-11-02 JST。**非実行設計。A/Bの選択、q/r/K、保持block、精度政策、科学call/compile件数は未確定。**
[静的調査](h4_df_pauli_matched_partition_static_design_20261102.md)の式・scalar会計・sector制約を前提にする。
今回作成するのはMarkdown 2資料だけであり、新adapter・runner・test・科学SOURCE・数値結果は作らない。

## 1. SOURCE・結果・入力の対応

| 役割 | 固定ref | 読取対象と再利用境界 |
|---|---|---|
| 研究全体 | `9a0bc7f17f47823e46c3945734127e9f0d09154c` | [全体地図][LAND]と[不足表][GAPS]。DF研究が中心、Pauli比較の位置付け |
| P3元設計 | `6f0759df7c1df7fd6d835922b0246753a8a98d6c` | [検証設計][DESIGN]、[実行準備][READY]。分割・保持block・allowance等は未決定 |
| H4入力生成 | `0c329b30a81accb35c212be8defa66fee0dd0f12` | [build_h4_inputs][INPUT]と[DF生成][DFH]。同一MO積分、spin/JW、D吸収 |
| H4保存anchor | `4591b8d75a2a6a55c74a51225f0c61648023c8ee` | [receipt][REC]、[Pauli model][PM]、[DF model][DM]、[旧比較][SYN]。snapshot/state/hashは静的調査§2 |
| H4 C包含SOURCE | `e80d7510880f8db993b777c627bb8e95b05a070d` | [保存C結果報告][CREPORT]。各表現reference・normalization/nominal/implemented meanの候補証拠。新分割には未適用 |
| H6 P1 | SOURCE `a0a3aff915f98456733d6bf09ed641a1be423628`、結果 `bfb4ca75437a9b3605a8f57b5f82100615dd034f` | [bias kernel][BIAS]、[結果][P1]。数値分解方法を参照、旧値・STOPを保持 |
| H6 P2 | SOURCE `11e50e262f95630b14fa4070ad30894ba3f64b74`、結果 `c9073274daee69c9b69d8db3dc332bf6132c2259` | [cost kernel][COST]、[結果][P2]。中央制御・descriptor・両軸会計の骨格。新T/V価格として流用不可 |
| DF B3 | native SOURCE `638e0a989b4471a075549ccb58617b1030e6748f`、公開結果 `ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed` | [h46_b3.py][B3SRC]のone_body保持検査と[公開成果][B3]。別snapshot・時間・候補域であり値は移さない |

旧H4入力のPauli hash `e0dbbe9a…`、DF hash `156939eb…`は異なるdescriptorの識別子である。
hash文字列が等しいことをfull-H一致条件にはしない。full-Hの保存PASSと、同じraw state/hashは別の証拠として接続する。
現branchのbaseはP2結果`c9073274…`。P1/P2のコード・artifact・STOPを再計算・改変しない。

## 2. 再利用対応表

「そのまま」は機能単位の再利用可能性であり、旧実行承認や旧固定runnerを起動してよいという意味ではない。
今回、以下の科学関数・testsは呼び出していない。

| 区分 | 既存処理・SOURCE | 使い方／必要な適合確認 |
|---|---|---|
| そのまま | 保存receipt/snapshot/modelの読取・hash照合、[PauliHamiltonian.from_terms/from_openfermion][PAULI] | label変換・重複統合・constant分離・cutoff0。将来の新残差展開自体は科学計算として許可後に行う |
| 条件付き | [h4_molecular_inputs.spin_lift_unthresholded/build_h4_inputs][INPUT] | index規約の根拠。build_h4_inputsはSCF/FCI/DF生成を伴うため再実行しない。保存tensorを読む小さなloaderで足りる |
| 条件付き | [df_hamiltonian_from_integrals][DFH] | D・rank/tolの定義を参照。既存λ/Gを使い、再DF生成を既定工程にしない |
| そのまま | [DFHamiltonian/df_hamiltonian_to_model][PART]、[one-body Gaussian/diagonal primitives][OPS] | 後者のone_body_correction fieldはh_eff全体。Aの保存one_body basis/specは将来、T/phase/time照合後に共有候補 |
| 条件付き | [prepare_df_partial_s2][S2]、[split_df_hamiltonian_by_ld][PART] | AのM_Cではfragment L_D0に対応し、一体は常に保持。Bの残差D̂追加は既存API外 |
| そのまま | [exact_df_diagonal_coefficients/extract_df_diagonal_tail][TAIL] | I/Z/ZZの係数、identity policy、basis-aware辞書。Aのsquare residual。BではD̂由来I/Zを加える小adapterが必要 |
| そのまま | [DFBasisRegistry・extraction_component_circuit_specs][TAIL]、[DF event builder][EVBUILD] | registered basis、actual adjoint、signed product/order phase、rotated Z/ZZ central制御、同basisのreuse。新辞書を正しく接続する |
| そのまま | [rte.finite_rte_distribution/finite_taylor_operator/finite_rte_operator_moments][RTE] | finite paired-Taylor、corrected/attenuated mean、normの意味。armごとにλ/law/Bを新定義する必要がある |
| 条件付き | [PauliSignalAdapter][PAULI] | M_AのL_D0骨格に適合。M_BのT_eff/T_bareは全Hのabs-prefix選択では表現できない |
| 最小拡張 | [P1 split_operators/propagate][BIAS]、[paired_polynomial_action][POLY] | prefixではなく明示T/V/block/scalarを受ける薄い接続。raw ν・trace/phase政策を合わせる。H6可換保持という結論は移さない |
| 条件付き | [finite_rte_signal_validation][FINITE] | sector/bit-order/leakage・exact-tailと名目meanの独立検査を再利用。関数全体は固有state/control stateを新生成するため、共通raw state固定のP3 runnerとして直呼出ししない |
| 条件付き | [DFSignalAdapter.operators][FIXED] | full/one/square/tail/scalarの構成を参照。Bでは追加D̂と共通Tの分離が必要 |
| そのまま | [D2 central_control_body/tokens][D2]、[両軸hadamard_wrapper/default_compiler][FIXED]、[transpile_and_measure_cost][COMP] | Pauli event部分とwrapper/計測。D2の対応gateはPauliEvolutionと局所productであり、Gaussian混在body全体をそのまま渡せない |
| 条件付き | [D1 event復元・interaction][D1]、[P2部品価格][COST] | 新実行で保存した同じlaw/eventのF/R/E/Empty診断に使える。旧H6件数・価格・代表indexをP3へ移さない |
| 条件付き | [fixed_signal_accounting.fixed_signal_shots][ACCOUNT]、[DF長時間のsignal/native骨格][B3SRC] | 方式別B/bias/allowance/Nとfull-wrapper価格を接続。旧N/G・別snapshot・old grantは流用しない |
| 現状は未対応 | 共通任意T＋任意Pauli残差のP3 arm、案Bの補正一体を含むDF-tail | 既存prefix adapter/DF square-only extractionだけでは完成しない。下記の小さい明示接続を提案する |

`evaluate_fixed_signal`全体はreferenceの新評価・sampling・compile・会計を連続実行し、raw exact νのP3政策もそのままでは満たさない。
本研究では必要な関数だけを取り出して接続し、Stage 1/2の許可で後続compileまで暗黙に走る経路を使わない。
これは新しい管理frameworkを作る提案ではなく、科学処理の依存範囲を明確にした最小接続である。

## 3. 確認した既存testsと将来の追加検査

既存testsはSOURCEを静的に読んだだけで、今回PASSを再取得していない。分子生成を含むtestもあり、一括実行しない。

| 既存test・所在 | 再利用する検査内容 |
|---|---|
| [test_fixed_signal_evaluation.py][TFIX]：`test_supplied_spatial_integrals_give_manual_number_operator_hamiltonian`、`test_supplied_integral_offdiagonal_convention_against_openfermion`、`test_openfermion_to_qiskit_bit_order_with_y` | 1/2・index・spin・JW label/bit順・tiny integral oracle |
| [test_df_hamiltonian.py][THAM]：`test_df_truncation_value_is_not_added_to_hamiltonian_constant`、`test_df_square_block_is_applied_matrix_free` | truncationとscalarの区別、λA²。H3実分子生成testは静的参照のみ |
| [test_df_rte_tail.py][TTAIL]：`test_exact_diagonal_i_z_zz_expansion_and_identity_policies`、`test_identity_global_phase_becomes_controlled_relative_phase` | square→I/Z/ZZ、identity抽出、controlled phase。H2再生成testは新規実行しない |
| [test_df_partial_s2.py][TS2]：`test_identity_policy_one_body_and_constant_variants_match_dense`、`test_shared_diagonal_primitives_match_existing_and_controlled_unitaries` | 一体・scalar・identityの一回適用、±timeとcontrol |
| [test_df_rte_qiskit_builder.py][TEV]：`test_sequence_order_and_cross_event_basis_reuse_preserve_unitary`、`test_controlled_circuit_controls_only_the_central_action` | 順序・actual inverse・basis reuse・central制御 |
| [test_finite_rte_signal_validation.py][TFIN]：`test_operator_moment_acceleration_matches_explicit_partial_s2_mean` | 独立名目mean oracle、sector/leakageとaccelerated mean |
| [P1人工tests][TP1]、[P2人工tests][TP2]、[D2人工tests][TD2] | raw norm、signed scalar、名目多項式、saved event復元、命令対応・F/R/E/Empty・両軸 |

将来の最小追加検査は、A/Bのtoyでfull/T/Vを独立に照合、D̂移管の符号、全trace/native scalar表記の同値、
共通T blockの同一命令・phase、Bの一体残差componentの符号/basis/identity、M_B/C名目meanの一致期待値である。
Gaussian逆変換は逆順＋actual inverse、個別eventのsector外作用、+Im sineをtiny full-space oracleで検査する案。
人工検査は新分子数値の代わりにならず、今回その実装・実行もしない。

## 4. 案A/Bに必要な最小拡張

共通の小さい入力記述を`保存full H参照、T descriptor、V descriptor、scalar所有、保持block descriptor`とする案。
Pauli prefix APIを一般化する巨大frameworkではなく、P3専用の薄いarm adapterとして既存helpersへ渡す。
この記述の数値生成・新実装は将来の許可範囲で行う。

| 項目 | 案A：T_eff/V_square | 案B：T_bare/V_normal |
|---|---|---|
| 入力 | 保存df_one_body、λ/G、元full H | 保存spin_one、df_one_body、λ/G、元quartic tensor |
| 共通保持 | 保存DF一体blockのbasis/specを再利用可能か照合 | bare一体のGaussian block等を新規に用意する必要 |
| Pauli residual | 保存square和を同じJW規約で明示展開・global labelごとに集約 | 元V_normalを展開、またはsquare＋D̂と独立照合。全H prefixの切り替えでは不可 |
| DF residual | 全square fragment、identityを共通規約で抽出 | 全squareにD̂を追加。D̂を対角化した一体I/Z辞書を既存basis-aware componentへ足す最小拡張案 |
| D̂の扱い | T内に一回 | random残差内に一回。T_bareへ置換しただけの既存DF adapterでは欠落 |
| 残る交絡 | 既存B3に近いが、bare二体分割とは違う。辞書norm/law/basis差が残る | 追加一体辞書も費用・lawに影響し、旧B3の直接再現ではない |

案BのD̂辞書は、Hermitian一体演算子の固有basisで`Σd_i n_i=(Σd_i/2)I−Σ(d_i/2)Z_i`を使い、
`DFBasisRegistry`、existing component spec、finite RTE、central Z builderへ接続する案である。
新しい擬似square factorizationやD̂の決定論側への置き戻しで比較条件を変えない。
元Dの返却配列はsnapshotの独立fieldにはない。`df_one_body−spin_one`で捕捉した補正と、依存SOURCEの式・Hermitian化の対応を将来確認する。

共通保持blockの選択肢は次のとおり。**今回どれも正式採択しない。**

| 選択肢 | 解釈・追加条件 |
|---|---|
| 共通Gaussian一体block | 既存DF helpersの再利用が多い。同じbasis命令、actual adjoint、eps、half時間、scalar phaseをM_B/Cへ共有。数値diagonalization/bindingの残差を扱う |
| 共通Pauli PF一体block | 両armへ同じTのPauli分解・順序を挿入可能だが、T内部PF biasが加わり得る。DF native保持と同じ近似方式とは呼ばない |
| exact exp(T)による信号診断のみ | Stage 2の物理分割比較に使える。dense unitaryの汎用合成価格をGaussian保持のnative価格として使わない |

Stage 3ではcommon controlled T half、各controlled residual event列、同じscalar placementを組み立ててから同じwrapperを掛ける案。
Pauli event部分だけD2 centralへ通し、Gaussian block全体をD2へ渡さない。
Gaussian loweringの`BogoliubovTransform(U.T)`経路とdense fallbackの違いを揃え、availabilityによる暗黙の実装変更をしない。
共通保持入力が同じでも、tailとのcompile境界でFull価格は異なる。孤立R共有とFull非加法性を区別する。

## 5. 将来Stage 1：Hamiltonian一致

**目的**：full H、T、Vの3一致を独立に検査し、normal-ordering・D移管・scalar会計を確定する。

- 入力条件：既存H4共通snapshot/modelの固定hash、A/Bの選択、full/sectorの評価範囲、coefficient cutoff・identity規約。新SCF/DF/state生成は不要。
- 再利用：receiptのfull-H PASS、保存Pauli/DF descriptor、spin/JW変換とλG²の既存構成関数。receiptはT/V別の証拠にしない。
- 最小追加：選択T/Vを各側で独立に構成し、共通順序でfull/T/Vを比較。`V=H−T−scalar`だけで両側を定義して独立検査を省略しない。辞書和と移管後scalarも確認。
- correctness/精度：index・符号・1/2・Hermiticity、full256でのoperator比較、T/Vのsector外blockを検討。保存1e-9 toleranceは過去条件であり、新しい科学分解能として自動採用しない。誤差norm・許容差・丸め/DF近似の扱いを実行前に決める。
- 判別：同じ物理分割を実際に実装できたか。ここではsignal・費用・優位性は判別しない。
- 残る交絡：辞書norm、law、Gaussian実装・compile、統計/allowance。
- STOP：原本/hash不一致、T/V/辞書和/scalarの意味ある不一致、未説明のD欠落/二重計上、sector射影で差を隠す必要、未許可の再生成が必要。

ここで初めて新operatorの数値再構成・一致評価が必要になる。今回は行っていない。
共通Hへの補正やPauli full Hの置換を黙って行わず、保存Pauli/DFの小さな既存不一致をどう扱うかも決める。

## 6. 将来Stage 2：ideal signal・finite bias・normalization

**目的**：同じraw φ・ν・reference/時間に対して、分割誤差・保持block近似・finite Taylor・implemented lawの差を区別する。

- 入力条件：Stage 1で対応付けたT/V/scalar、保存共通raw state、T=.8、選択q/r/K、正確なpaired cutoff、共通referenceとallowance政策。新しいstate/referenceを既定生成しない。
- 再利用：P1のexact retained/exact tail対polynomialの分解、`paired_polynomial_action(normalization=1)`、DF exact-tail/operator-momentの検査、保存reference包含・exact νの規約。
- 最小追加：明示T/Vを受ける小さい接続。M_Aの保持なしanchor、M_B/Cの同じT half＋exact V、共通保持実装のideal作用、各方式のfinite corrected meanを分ける。必要なimplemented CDF/angle/normalizationの評価は別に数える。
- correctness/精度：同じT/V/q/r/K/cutoffと同じ保持blockなら、M_B/Cの**補正済み名目多項式mean**一致を期待する。Bの違いによるraw attenuationは別。B/ν二重適用、phase alignmentによる差の除去、sector再正規化をしない。
- sector：T/V和の不変性が確認できればnominal polynomial meanの36次元評価を検討できる。individual Pauli event・Gaussian途中・implemented lawを未確認のまま同sectorへ射影しない。full256評価か、対象演算子ごとのleakage根拠が必要。
- 判別：保持導入でどの近似biasが変わるか、同じ物理分割でもBがどのように異なるか、名目meanと実効law差が解像できるか。
- 残る交絡：M_A/Bは物理分割とlawが同時に変わる。M_B/Cでもencoded basis/finite-grid/roundingの差が残り得る。実compiled精度・費用は未評価。
- STOP：共通nominal meanの不一致を説明できない、reference/ν/phase不整合、precision不足、leakage、未許可のprecision・solver・新referenceへの変更。分解能以下はUNRESOLVEDであり0確定にしない。

既存P1のI/Z可換保持・L0 alias・1e-10工程目安を、新H4分割の実測結果や保証へ移さない。
新しいB/bias/Nが要らないoperator一致だけのscopeならStage 1で止められる。Stage 2を自動追加しない。

## 7. 将来Stage 3：matched native回路・価格

**目的**：同じT block、同じ物理V、同じwrapper/compilerの下で、Pauli/DF残差実装の全体価格を比較する。

- 入力条件：Stage 1/2の対応、選択保持block、q/r/K、各armのlaw、scalar所有/placement、control・basis reuse、P=0、cosine/+Im sine測定境界。
- 再利用：DF deterministic halfとevent builder、Pauli event/D2、両軸wrapper、compiler/metric collector、QPY/hash、P2の部品共有・interaction式。
- 最小追加：共通controlled T half＋arm別controlled residualを合成するarm adapter。新規価格を取る前にoperator/phaseと新law/保存eventの対応を検証。共通T/Emptyは入力program一致で共有する。
- trajectory：代表1本の機構診断か有限cohort価格かを先に選ぶ。lawが異なるM_B/Cに同じseedを渡しただけでpaired科学標本と呼ばない。保存M_Aは入力・law・q/r/K・compiler・個別QPYが一致する場合だけ再利用候補。
- correctness/精度：artificial small full-space oracle、actual adjoint・signed phase、共通T命令、wrapper測定、input/compiled QPY対応。RZ/CX/depth/size/histogramを保存。compiler案は旧比較のQiskit1.3.0/basis rz,sx,x,cx/opt1/seed17/追加backend等なしだが、実行前に選択・固定する。
- 判別：その固定program/cohortにおける辞書・Gaussian/control・compilerを含む価格差。同じ保持programが作れなければ、その実装差も含む結果として報告する。
- 残る交絡：λ・probability・angle・basis switch/reuse・結合非加法性、代表性・rare-event tail、実compiled残差。個別passやHamiltonian表現だけの独立効果ではない。
- STOP：共通T/残差/phase/wrapper/compiler不一致、許可外の候補・sampling/compileが必要、固定原本が欠落、共有環境への干渉。科学失敗のretry/別条件への変更は実行scopeに従い勝手に追加しない。

旧D1はgeneric Pauli controlの価格、D2は同bodyへのcentral化、P2はH6の別分割である。新P3の保持/event価格には転用しない。
B3 nativeのsymmetric-directional controlも旧H4のcentral構成と同一視しない。occupation oracleや会計の再利用と、controlled programの採用は別に照合する。
同じTとVでもR/E分割後の単独価格はFullへ加法的とは限らない。部品診断を選んだ場合にだけ、同じ新trajectoryのF−R−E+Empty等を取得する。
全cohortや全lawの再compileを既定の完成条件にしない。

## 8. 将来Stage 4：同精度N/C/G

**目的**：新bias・B・compiled価格が揃ったarmだけを、同じsignal精度/failure/数値誤差政策で会計する。

- 入力条件：各arm専用のlaw/B/bias、same raw ν/reference/T、ε_sigと軸配分、failure budget、u_enc/ρ・normalization上界の扱い、取得した価格の統計対象。
- 再利用：fixed_signal_shotsの共通Hoeffding会計、DF Track Aのsignal/price分離、探索と別標本確認、Pauli両軸の同trajectory会計。[ACCOUNT][ACCOUNT]、[B3 SOURCE][B3SRC]。
- 最小追加：P3 armの新結果を既存会計関数へ結合する。旧N/G・別armのN・旧平均価格を新候補へ適用しない。quantum shot Nと価格MC標本数を区別。
- correctness/精度：Bは各lawに対応する上界を用い、数値allowanceを共通の意味に合わせる。不足はnull/UNRESOLVEDとし、経験値と保証を分ける。ρ=0仮定を実compiled保証と呼ばない。
- 判別：選んだ有限候補/精度/費用標本条件で、B・bias headroom・単回価格を合わせた資源評価が可能か。
- 残る交絡：候補域、選抜、価格母平均/tail、実ρ、入力/reference差。単一Tのsignal会計は全energy/RPE総費用と同じではない。
- STOP：bias/B/allowance不足、headroom成立を確認できない、価格とlawの対応欠落、母平均/全law保証へ無根拠に拡張する必要。

Stage 3だけの固定program価格から新しい方式順位を作らない。共通T/Vが成立してもsampling law/B/Nが同じとはしない。
Stage 4は独立に実行許可を要する。今回、N/G・新cost倍率・ランキングの計算は0。

## 9. 実行前に選ぶ事項と最小scope

| 選択担当 | 未決定事項 | 判断への影響 |
|---|---|---|
| ユーザー/GPT | A、B、または両方 | B3に近いsquare分割を問うか、bare一体/二体分割を問うか。補正辞書の追加要否 |
| ユーザー/GPT | Gaussian共通保持、Pauli PF共通保持、exact-T信号診断をどこまで使うか | 同じ物理Tと同じ保持回路の区別、内部近似とnative価格の解釈 |
| ユーザー/GPT | Stage 1まで、Stage 2まで、価格Stage 3、同精度Stage 4のどこまで必要か | 許可する科学処理と得られる結論が変わる |
| 実行指示で確定 | 共通targetを保存Pauli/DFのどちらに定義するか、既存差のallowance/転送、raw ν・identity表記 | reference・誤差政策とscalar二重計上を防ぐ |
| 実行指示で確定 | q/r/K、候補/trajectory選定、機構代表か価格cohortか、compiler/lowering/basis reuse | 旧q8/r4/K4やH6 index0000を根拠なく移さない |
| 実行指示で確定 | operator norm/tolerance、精度と昇格条件、sector利用範囲、ε/failure/u/ρ政策 | 解像度に合う結論とSTOP、empirical/certifiedの区別 |

最小scopeの候補は、**選択した分割のStage 1だけ**、または**Stage 1＋Stage 2の事前固定した比較tupleだけ**である。
価格が目的ならStage 3、同精度資源評価が目的ならStage 4を追加する判断材料になる。どれも今回は採択しない。
科学call・compile数は、選択したarm/tuple/代表またはcohort・再利用原本・精度処理を確定してから実装依存の実数として算出する。
P2の最大12件や過去研究のworker/call上限をP3の新上限としてコピーしない。

## 10. 承認・保全・今回の技術確認・mandatory STOP

ユーザーの明示許可は静的SOURCE/metadata調査、設計、文書公開までであり、上記Stageの実行許可ではない。
将来は選択したH4 input・T/V/辞書・block・precision・candidate/law・処理範囲・STOPを定め、新しい明示実行許可を受ける。
許可された範囲の通常の実装/debugはまとめて行い、承認証明書・原文hash保存・grant基盤を新設しない。旧固定契約・使用済みgrant/STOPを再利用・解除しない。

今回の技術確認は、Git object/metadata読取、保存hash/descriptor照合、依存SOURCEの読取、Markdownリンク・引用先・差分検査に限定した。
scientific libraryをimportしてoperatorを組み立てる処理、SCF/FCI/DF再生成、norm/誤差/費用/倍率の新計算、simulation、sampling/RNG replay、QPY回路build/compileは0。
既存testsの科学処理も実行していない。引用した12 test関数は固定SOURCEのAST内で存在確認した。文書間relativeリンク2件、固定commitの参照先path、公式依存SOURCE blobを照合した。これは文書の技術検査であり、独立科学レビューや主要claimの正式採択ではない。

旧P1/P2 artifact subtreeとSTOP、既存科学SOURCE/input/raw/result/grant/ledger/marker、main/元研究branch、共有OS/venv/他jobを変更しない。
CPU/RAM/workerへ恣意的な低い上限を置かず、将来の独立compile等は実際の空き資源に応じて並列化する。共有設定を変更したり他jobを停止したりしない。
新独立branchへこの2文書をcommit/publishし、remote SHA・全文一致を照合する。
**公開照合後mandatory STOP。次の担当はユーザー/GPT：分割A/B/両方、共通保持block、最小Stage scopeの判断。Codexの次の科学実行作業なし。**

[LAND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_scientific_landscape_20261101.md
[GAPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_claim_evidence_gap_matrix_20261101.md
[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6f0759df7c1df7fd6d835922b0246753a8a98d6c/docs/research/pauli_pure_rte_mechanism_validation_design_20261101.md
[READY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6f0759df7c1df7fd6d835922b0246753a8a98d6c/docs/research/pauli_pure_rte_mechanism_execution_readiness_20261101.md
[INPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h4_molecular_inputs.py
[DFH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_hamiltonian.py
[REC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json
[PM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json
[DM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/df_native_model.json
[SYN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_comparison_synthesis_20261025.md
[CREPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_c_reference_evidence_20261023.md
[BIAS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a0a3aff915f98456733d6bf09ed641a1be423628/src/trotterlib/pauli_mechanism_bias.py
[P1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bfb4ca75437a9b3605a8f57b5f82100615dd034f/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/bias/REPORT.md
[COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/11e50e262f95630b14fa4070ad30894ba3f64b74/src/trotterlib/pauli_mechanism_cost.py
[P2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9073274daee69c9b69d8db3dc332bf6132c2259/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost/REPORT.md
[B3SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/638e0a989b4471a075549ccb58617b1030e6748f/scripts/h46_b3.py
[B3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md
[PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/pauli_signal_adapter.py
[PART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_partial_randomized_pf.py
[OPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_trotter/ops.py
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_partial_s2.py
[TAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_rte_tail.py
[EVBUILD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_rte_qiskit.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/rte.py
[POLY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a0a3aff915f98456733d6bf09ed641a1be423628/src/trotterlib/pauli_signal_adapter.py
[FINITE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/finite_rte_signal_validation.py
[FIXED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/fixed_signal_evaluation.py
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/11e50e262f95630b14fa4070ad30894ba3f64b74/src/trotterlib/h4_d2_pauli_control.py
[COMP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/rte_compiled_cost.py
[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/11e50e262f95630b14fa4070ad30894ba3f64b74/src/trotterlib/h4_d1_adapter.py
[ACCOUNT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/fixed_signal_accounting.py
[TFIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/tests/test_fixed_signal_evaluation.py
[THAM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/tests/test_df_hamiltonian.py
[TTAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/tests/test_df_rte_tail.py
[TS2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/tests/test_df_partial_s2.py
[TEV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/tests/test_df_rte_qiskit_builder.py
[TFIN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/tests/test_finite_rte_signal_validation.py
[TP1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a0a3aff915f98456733d6bf09ed641a1be423628/tests/test_pauli_mechanism_bias.py
[TP2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/11e50e262f95630b14fa4070ad30894ba3f64b74/tests/test_pauli_mechanism_cost.py
[TD2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/11e50e262f95630b14fa4070ad30894ba3f64b74/tests/test_h4_d2_pauli_control.py
