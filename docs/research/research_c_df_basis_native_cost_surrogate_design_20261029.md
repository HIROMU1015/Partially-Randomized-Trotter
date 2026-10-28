# 研究C：Basis-aware Native Cost Surrogateの設計・既存検証の整理

2026-10-29 JST。起点`70b5d5de36eedb8cc72a323bc32ece8e9839747b`。独立branch `codex/research-c-basis-native-surrogate-20261029`。

**最初に検証する案はC：basisごとのrun構造＋Z/ZZ＋head差を分離した説明可能な特徴ベクトル。** 現時点では価格を返す式やDF目的関数として採択しない。Aの切替頻度とBの既知overlap案を比較対照とし、保存2 cohort各32件で反証可能な148-call再解析を準備した。

今回は技術調査・数学設計・人工PoCのみ。**実H4の新特徴量、相関/回帰、trajectory再集計、NPZ配列/QPYの数値decode、fit/sampling/build/compileは0。** 実保存ファイルにはhash/metadata/schema照合だけを行った。43人工・回帰PASSは性能適合の証拠ではない。

## 1. 固定研究条件と既報の範囲

直鎖H4・隣接1.00Å・STO-3G、Nα=Nβ2/sector36/system8/full512、通常number CDF8/元frame順LD2。元g補正κ、nuclear scalar一度、tail identity抽出、interleaved2i+σ。Partial ordered symmetric S2、δ=h=.1/s1/r128/K0/m0..12、own-DF ground oracle/状態準備除外、共通CA=.00159Ha/DFreserveCA/8。P0/Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/initially-zero=false/MPFR160 declared phaseを変更しない。

| 保存報告の引用 | PR-aware Proposed | tail-P4 |
|---|---:|---:|
| Tail係数位置 | 216 | 144 |
| λ_R [Ha] | 1.442430 | 1.516166 |
| 主仮定shots | 16,056 | 16,568 |
| Kernel標本平均RZ | 14,076.4375 | 15,047.46875 |
| 条件付き総RZ | 234,117,788,135 | 275,422,425,481.5 |

数値は[M4-Native128報告](research_c_pr_aware_df_M4_native128_execution_20261028.md)と[tail-P4報告](research_c_df_structural_pr_performance_20261028.md)の引用。今回新たに計算した集計値ではない。両者は全13round適格・native32 full512/adjoint/X/Y PASSだが、全law synthesis/implemented law/cost support/母平均優位/actual CAは未資格。二候補はJ/U・projected start・tail law・cohortが異なり、疎化だけの因果効果ではない。

正式共通指示2026-10-25版§8.1/8.2/8.4/10/10.1/10.3と既存契約を維持する。旧raw/SOURCE/STOP/grant/ledger/markerを保全し、旧runnerを再開しない。承認証明・新grant・重要GPT詳細科学レビューは作成/実施しない。

## 2. 既存研究・コードの対応表

| 資料/既存実装 | 既に実施・設計されていたこと | 今回の再利用と未取得部分 |
|---|---|---|
| [DF構造案](research_c_df_structural_design_20261028.md)、[P4 21-call](research_c_df_structural_support_execution_20261028.md)、[449-call](research_c_df_structural_pr_performance_20261028.md) | 固定support/全U可変の1 fit、物理精度、144 atom、new PF/RTE/ground、native32・総資源 | J/U・law・価格の原本を次の対象へ固定。支持数減少が価格減少を保証しない負結果を保全。個別機構の寄与は未分離 |
| [PR-aware設計§7](research_c_pr_aware_df_optimization_design_20261026.md)、[機構整理§7](research_c_pr_aware_df_performance_mechanism_readiness_20261028.md) | λ-only smooth penalty、固定U・凍結非負event-price weighted mass案 | 既存案を再発明しない。後者はbasis/U変化・run・whole-P0を直接表さない。今回その3 fit案を実行しない |
| [M1](research_c_pr_aware_df_mechanism_execution_20261028.md)、`c_pr_aware_df_mechanism.py` | 保存shots×価格の会計的寄与分解。因果分離ではない | 「shots」と「kernel価格」を分ける解釈を再利用。M1を今回再実行しない |
| [M2](research_c_pr_aware_df_M2_same_JU_deterministic_execution_20261028.md)、[M3](research_c_pr_aware_df_M3_s1_r64_repair_execution_20261028.md)、[M4](research_c_pr_aware_df_M4_s1_r128_feasibility_breakeven_20261028.md) | 同J/U決定論対照、s/r変更、新PF/RTE、RPE、affine費用 | 異条件の比較は履歴として参照。新basis-price特徴量の検証や2 cohortの因果分離を完了したものではない |
| `c_pr_aware_df.py`/B1/B2/M4-native/P4-native | 一般J係数→signed atom/law、Gaussian・whole-P0/full512、校正 | 物理係数・sign・identity・frame順を維持。保存metricsを再利用し、新price engine/sampler/compilerを作らない |
| `c1_stage3_connection.split_native_model`、`circuits.build_event_sequence` | Exact basis keyと最大隣接runで共有 | 同じ判定を特徴定義へ使う。近いUのapproximate merge、列符号の自動統一をしない |
| `StrongBuilder`/`PrecisionBuilder` | Head relative Gaussian、exact-zero/LAPACK/atan2の既存Gaussian合成 | SOURCEを静的確認。Tail全basisのrecipeと異basis相対距離は別。今回実UのGaussianを再合成しない |
| `pricing.expected_literal_price/direct_literal_price` | 既知literal token価格のrun状態DPと独立scan | 人工値で整合性を再利用検証。これはP0 whole-kernel価格でもboundでもなく、実価格不足を0/1で埋めない |
| `rte._basis_reuse_intervals`/既存event codec | Exact keyの連続interval、sign/support/順序検査 | 特徴helperで再利用。新samplerなし。旧WP/connected-clusterのbasis mass計算も確認したが別実装routeの性能を今回へ移送しない |
| [圧縮DF接口調査](research_c_compressed_df_interface_survey_20261024.md) | RC-DF/SCDF・一般J/square/shift・固定実装の制約 | 係数正則化・norm低減は既知。忠実RCDF実装/SCDF汎用fit入口の未解決を解消済みとしない。今回外部文献の新規priority認定はしない |

## 3. 現native政策で費用が生じる箇所

**Tailの共有単位はframe名ではなく`basis_key`。** `split_native_model`はspin-liftした宣言rotation tupleの`sha256(repr(rotation))`をkeyにする。`build_event_sequence`は連続した同じkeyを一つのrunとし、そのrunにbasis inverse/forwardを一組置く。異frameでも宣言rotationが厳密一致すれば同じkeyになり得る。`Provider`はsingletonでもfull-frame basisを使い、support-completionへの追加削減を仮定しない。

各tail frameの確率massではなく、共有されるkeyごとに集約したq_bをAに用いる。浮動小数点の近似一致・物理的projector同値・列符号gauge同値では自動共有しない。6 tail frameと異なるbasis keyの個数を同一視しない。

Head半分の順序はone-body/F0/F1で、その内部に`V_following† V_previous`のrelative fusionがある。逆半分も同様。Tailの異basis間およびcross-head/tailでは一般relative fusionをしない。Whole-P0は隣接gateを簡約し得るが、未観測の効果を事前値へ加えない。

K0の中央は各event一つのsigned ZまたはZZ rotation。`append_central`の**未compile emit数**はZでRZ2/CX2、ZZでRZ2/CX4。したがってλ>0・r128の固定条件で中央RZは2r、期待中央CXは`2r+2r p_ZZ`。これはGaussian/head/phaseを除いたraw中央だけの式で、P0後価格ではない。Atom種類144/216はevent数128を変えず、中央RZ数も直接変えない。

同basisの同supportが連続すると角度結合/cancellationの機会はあるが、現tail emitter自体は中央角度を先に総和しない。符号・角度・支持・順序を保ち、非可換なeventを並べ替えない。Scalar/identity位相はcontrolled相対位相として別途一回だけ保持する。

価格増の候補はq_b変化によるrun回数、Uによる各basis recipe、Z/ZZ・sign/support列、head F0/F1、P0境界簡約、有限非paired cohortのばらつき。今回新特徴量を測っていないため、どれが主因かは未判定。Headが主因、Gaussianが何%という値は出さない。

## 4. 三候補の数学仕様

### A：Sampling-weighted exact-basis run

非identity atom aの係数c_aに対し、`λ=Σ|c_a|`、`p_a=|c_a|/λ`、`q_b=Σ_{a:key(a)=b}p_a`。K0 ideal IIDの隣接一致確率とrun開始期待数は

`P_same=Σ_b q_b²`, `E[N_switch]=(r−1)(1−P_same)`,
`E[R_b]=q_b+(r−1)q_b(1−q_b)`, `E[R]=1+E[N_switch]`。

Run長ヒストグラムは実列の観測量、E[R]はlaw上の期待量である。固定列を独立に再samplingしない。λ=0で空tailがcertifiedならrandom event/run期待数は0、P_sameは非適用/nullとする。未知のhead/価格を0にする規約ではない。Identity atomはここへ入れずphaseへ抽出済みを要求する。

この案は共有routeに直接対応し、一般J/負係数もabs massで扱える。ただし各basisが何RZか、head、Z/ZZ、P0を説明しない。同一のrun数でも価格が違い得る。既存run DP・basis-mass研究と重なり、独立の新規性を主張しない。

### B：既存overlap/transition距離

実直交n×n行列に`D_lk=n−Σ_ij |(U_lᵀU_k)_ij|⁴`、`T=(r−1)Σ_lk q_l q_k D_lk`。ここでq_lはframe mass、Aのq_bはkey集約massであり使い分ける。実列には`overlap_path=Σ_{adjacent}D_lk`を定義する。0≤D≤n−1は直交性から従う（binary64の範囲確認allowance1e-8、丸め値をclipしない）。

これは[構造案のA](research_c_df_structural_design_20261028.md)で既に提案済み。列符号/列置換に不変で、signed permutation相対行列なら0。しかし現在のtail共有はexact keyだけであり、Dが小さいことから価格低下を導けない。Head relativeの数学的向きには対応するが、正確なelimination/gate/P0価格を表す式ではない。対照の幾何descriptorとして残し、主目的への採択を保留する。

### C：basis/run＋atom＋headを分離するベクトル（推奨）

まず価格単位を持たない

`x=(q_b,E[R_b],実R_b,run長,Z数,ZZ数,同basis同support隣接数,符号/角度診断,head/basis幾何)`

を保持する。既知のrecipe価格が将来得られた場合だけ、countについての加算proxy `H_k+Σ_b R_b G_b,k+Σ_a N_a c_a,k+φ_k`を検討できる。H/head、G/basis pair、c/中央、φ/phaseを別項とし、今回はH/G価格を**null**にする。Depthは単純加算してwhole-depthと呼ばず、6指標ごとに検証する。既存literal DPの価格を代入してもliteral grammar内の値に過ぎない。

最初の実証では、次の小さい動的投影を事前固定する。`C_composite`の説明変数は**basis_switches、ZZ_count、same_basis_support_pairsの3列のみ**。Aはswitches1列、Bはoverlap_path1列、対照はcohort内定数平均。各cohortに別interceptを置く。Interceptは固定headを含む平均差を吸収するが、head単独価格と同一視しない。

静的q/U/expected-TはDF当たり1観測として保存するだけで、32倍複製して説明変数回帰しない。Raw QPYのopcode censusも全体の診断として保存し、head/basisを誤って分割した価格とはしない。これらを結果後に都合よくCの説明変数へ追加しない。動的3列の不適合が、あらゆる複合surrogateの不可能性を意味するわけではない。

| 条件 | A | B | C |
|---|---|---|---|
| J/Uとlawから定義 | Exact係数/key集約で可 | U・frame massで可 | Law/run/atomは可、価格H/Gは別データが必要 |
| 現native共有との対応 | Exact-key runに直接対応 | Tail一般relativeを使わないため弱い | 共有・中央・headを区別できるがP0を検証要 |
| Head価格 | 含まない | 一部幾何のみ | 未知価格null、cohort差と明示して保持 |
| 一般J/非PSD/負係数 | 符号吸収/abs mass | J制約なし | Sign/support/phaseを保存 |
| 微分可能性 | Exact keyの変化は離散 | 固定ラベルならUに滑らか | 位置/run/gateは離散。未採択smooth近似が必要 |
| 棄却に使う観測 | Within-cohort heldoutで平均予測より悪い/不識別 | 同様、D小でも削減しない例 | 3列が単純A/B/平均を改善しない、rank不足、負価格予測 |
| 新DFへの一般化 | 未検証 | 未検証 | 未検証。独立DF holdoutが必要 |

## 5. Gauge・物理表現と目的関数への接続

列符号を変えるとdensity projectorは不変。列置換はUとJの両方を整合して変換すればDF tensorを保つ。Bはこれらに不変だが、宣言行列のbyte key・Gaussian recipe・中央support labelは変化し得る。**物理的gauge同値と固定native recipe同値は別**である。現保存cohortのgaugeを改変/再canonicalizeして比較しない。近いUを同basisとして共有させるsmooth approximationは、現native政策そのものではない。

将来の`Fγ,η=Fγ+η T_native`は、独立にDF物理予算・PF/RTE/attenuation/RPE・実価格を評価する必要がある。λ-onlyと異なるのは、同じλでもq_b/run・U・head・atom価格が違う点を区別すること。η、scale、T_nativeのscalar化、optimizer条件は今回一切採択しない。

Smooth |c|_εからp/qを近似する場合、exact λ/lawと分けて記録する。Exact keyによるgroupingはUに一般に非滑らかであり、固定groupingの局所近似やBの滑らかな近接度を導入しても、実際に共有できるという保証はない。PrecisionBuilderはexact zero条件でeliminationするため、Uの近さを単純なgate数gradientに置き換えない。

固定U/凍結price weighted massとの違いは、basisが連続する順序・runと中央種類・headを区別する点。RCDFの係数penalty、SCDFのrank-one/shiftは別の表現/用途で、今回一般J・κ/scalar・PR/RTEを変更していない。Price predictorの説明的回帰係数を、optimizer ηや正則化weightへ転用しない。新fitter/gradient/最適化iterationは実装していない。

## 6. 保存2 cohortのSOURCE・schema対応

| 内容 | M4 Proposed | Tail-P4 |
|---|---|---|
| 科学SOURCE commit | `dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad` | `4ac0e5f96a5ad5009f6273d0fc6ff5657ff80aff` |
| Sampling SOURCE digest | `3c0fc544b31b4aad9dd776c83eeb6392d59ee03c5b8a3059433cd1b65d56e12f` | `324fdf9c09fada29b37916ab69569b0389f8beaa575d608bb10900a63c2fa505` |
| 保存run | `artifacts/research_c_pr_aware_df_M4_native/2026-10-28/run01/` | `artifacts/research_c_df_structural_pr/2026-10-28/run01/native/` |
| Label | `proposed_00`〜`proposed_31` | `tail_p4_00`〜`tail_p4_31` |
| JU SHA256 | `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227` | `ec0dfe3d53e8209b343b294a103ab89405ac757d3e26164ae591344d4423b985` |
| Master/cohort | 2026102805/`M4_r128_native_fixed32_v1` | 2026102806/`tail_P4_s1_r128_native_fixed32_v1` |
| 元tail catalog | B1-run01/`proposed_tail.json` | run01/reference/`tail.json` |
| STOP call数 | 332 | 親run01の449 |

対応キーは**cohort＋role＋sample＋JU hash＋sampling SOURCE**。同じsample番号を両cohort間でpairedとしない。

- `samples/{label}_trajectory.json`：seed/cohort/SOURCE_sha256/final_JU_sha256/microsteps/events。P4にはroleもある。各eventはtaylor_order、rotation_component_id、phase、angle、application_sequence等。Appはbasis_hash/id、coefficient sign/abs、supportを持つ。
- `samples/{label}_metrics.json`：保存rz_count/rz_depth/cx_count/cx_depth/total_depth/circuit_size等。科学再解析時にfeaturesとこの固定labelでjoinする。
- `circuits/{label}_raw.qpy`/`_compiled.qpy`：今回bytes hash/QISKIT magicだけ確認。将来はraw opcode censusだけdeserialize、compiledはhash/provenanceの確認のみ。新build/compileやOperator decodeなし。
- Tail catalogにはatom_id/kind/coefficient/word/basis_key/frame_id、componentsにはp/sign/absがある。`df_fragment_id`がnullでもatom_id→catalogのframeへjoinできる。Frame/key/順序の情報を勝手に補完しない。
- 保存exportは両方J/Uがfloat64 `[8,4,4]`。P4にはcompact raw optimizer等の追加配列がある。今回はNPY headerだけ確認しpayloadをdecodeしていない。

[schema対応原本](../../artifacts/research_c_df_basis_surrogate/2026-10-29/technical/schema_correspondence.json)に値を含まない型/metadataを保存。旧JSONはschema抽出のため一件/型だけ読み、features/相関・新価格集計はしていない。全対象のpath/SHA256は次の契約に列挙する。

## 7. 未承認の最小科学実験：保存64件だけ

[固定SOURCE99](../../artifacts/research_c_df_basis_surrogate/2026-10-29/technical/SOURCE.json) SHA256：`7fd98dee2b241bbd991c58fa73c10cb42a12f3b477531ed01b3ab42d3b16fd12`。
[未承認148-call契約](research_c_df_basis_surrogate_contract_unapproved_20261029.json) SHA256：`345f81f7784335af87cbe0eff144997ffdad4aa452d89d7018f3f9027c03fa43`。

対象2DF・各32件、128 K0 events、input267件のhashを固定。Fresh `artifacts/research_c_df_basis_surrogate/2026-10-29/run01`、retry/resume0。新fit/sampling/build/compile/演算子/PF/RTE/shot/総資源再計算は0。既存価格metricsをtargetにする**特徴量の科学的再解析**なので、実行には別途明示許可が必要である。

| Call | 上限 | 実際の内容 |
|---|---:|---|
| saved_bundle | 2 | cohortごとの保存U/tail/SOURCE/STOP読込 |
| identity_gate | 2 | 元run/STOP/provenance検査 |
| static_features | 2 | law grouping、U幾何、expected run/overlap、head F0/F1診断 |
| trajectory_features | 64 | 固定eventの特徴抽出と保存metrics join |
| saved_raw_census | 64 | raw QPY読込/全体opcode count。Head分割価格ではない |
| descriptive_models | 12 | 2cohort×6metric、各A/B/C/meanの4fold＋記述相関 |
| surrogate_qualification/result_summary | 各1 | 主RZ判定・資格の限界・STOP |
| 合計 | **148** | 旧科学call残枠やSTOP済みrunnerを再利用しない |

### 事前登録する統計・判定

各cohort内でsample index mod4の固定4fold（train24/test8）。NumPy OLS/intercept/rcond1e-12、A1列/B1列/C3列/mean-onlyを同じfoldで比較し、heldout MAE/RMSEを6指標別に保存する。Pearson/Spearmanも各cohort内・登録した列だけの記述値で、p値・有意性・新特徴選択・重み探索を行わない。

Constant列/targetの相関はnull、rank不足はUNRESOLVEDで、列drop/ridge/別foldへfallbackしない。欠損や非有限inputはcorrectness FAIL、負の予測値はclipせず記録する。副5指標を主RZへ任意の重みで加算しない。

主RZでCのMAEが**両cohortそれぞれ**mean/A/Bよりstrictに低く、必要なdesignがfull rankで、Cの負予測がないときだけ`RETAIN_FOR_INDEPENDENT_DF_VALIDATION`。それ以外は`SURROGATE_NOT_QUALIFIED`として負結果/不足証拠を保存する。微小なMAE差を有意差や普遍的予測力とは呼ばない。このgateがPASSしても**new_DF_fit_GO=false**であり、目的関数の採択・新fitを許可しない。

同一DF内のU/Jは固定なので、64 trajectoryは64 DFではない。32件をfoldに分けても「新DFへ汎化」の検査にならない。二つのintercept差は静的head/Uの因果分離ではなく、P4効果の独立サンプルも2つでしかない。将来は独立DFを丸ごとholdoutする設計が必要で、その新fit/条件を今回採択しない。

## 8. 実装・人工検証・保全

新helper `c_df_basis_surrogate.py`、限定entry `c_df_basis_reanalysis.py`/`run_research_c_df_basis_surrogate.py`、人工testsを追加。既存のatom normalization、private run interval scan、saved-event restore/符号検査、literal DP、NumPy least squares、SciPy correlations、Calls/I/Oを再利用した。新価格エンジン・Gaussian/compiler・samplerは作っていない。

**新12＋既存M1/metadata回帰31＝43 PASS（1.70秒）**。人工のIID期待値と全列挙、同basis/異frame、128列・Z/ZZ/sign/order、λ0/identity禁止、overlap範囲、一般非PSD Jのgauge tensor不変、exact keyは別という区別、geometry join、固定fold/full rank/rank不足/欠損、literal DP、人工QPY census、mock148-call E2E、partial STOP/hash/freshを確認した。初回のtest側API戻り値とmock contract不足を修正し、最終suiteでPASS。実保存trajectoryを入力したテストはない。

[人工・回帰ログ/verification](../../artifacts/research_c_df_basis_surrogate/2026-10-29/technical/)を保存。Passive preflightはhash/metadata/packagesだけでPASSし、新science output run01は未作成。既存SOURCEは無変更。保護原本1458件をhash確認した。

Task-local venv/cache、既存依存はread-only参照。今回の処理は小さい人工配列・64×128相当のmock/metadata I/Oと4変数OLSで、重い科学負荷はない。将来の148-callもsmall matrix/I/O中心の逐次schema gateで、再compileを含めない。CPU/RAM/worker/wall人工capなし。実行時は空き資源を再確認し、重い独立DF/compile検証を将来設計する場合はその負荷に応じて既存並列化を使う。共有設定・venv・cache・他jobを変更していない。

## 9. Q1〜Q8と次担当

- **Q1**：Run分布・各U recipe・head・Z/ZZ/sign/support・P0境界・有限cohort差が原因候補。寄与は今回未測定で主因を断定しない。
- **Q2**：Head内部relative、tail exact-key run共有が実装済み。Tail一般relativeやcross-head/tail追加融合は未実装政策として除外する。
- **Q3**：Cのlaw/run＋atom＋headを分けたベクトルが最初の候補。単一の価格式・ηは未採択。Aは簡単な反証対照、Bは幾何のみの対照。
- **Q4**：既存overlap案の再利用であり発明とは呼ばない。固定U weighted mass/RCDFとの重複を保ち、順序/run/whole-P0との対応を問う点を追加する。
- **Q5**：2固定DF内の動的event構造と価格の記述的関係、同cohort heldout適合まで。静的U効果、DF間因果、普遍的価格予測は検証できない。
- **Q6**：新DF汎化、最適化によりU/Jを動かしたときの改善、PF/RTE/物理精度・総資源効果には独立DF/新評価が必要。今は実行しない。
- **Q7**：次の一つは固定2cohort各32件の**148-call保存値再解析**。新compile/fitを避け、A/B/Cの予測限界を先に反証する。
- **Q8**：Cが単純対照を上回らなければ価格目的への直結を保留。上回っても独立DF検証の候補として残すだけで、η/scale/solverやfitを自動採択しない。

**ユーザー／研究C**：この仕様・限界と固定契約を確認し、保存値再解析を別途明示許可するか判断する。**Codex**：その許可後だけ固定SOURCE/input/fresh outputで148-callを実行し、負結果/UNRESOLVEDも公開してSTOPする。

技術報告・SOURCE・人工証拠・未承認契約をcommit/pushし、remote SHA/tree・SOURCE・旧原本を照合後、**mandatory STOP**。今回は新しい実H4特徴量・科学計算・重要GPT詳細レビューへ進まない。
