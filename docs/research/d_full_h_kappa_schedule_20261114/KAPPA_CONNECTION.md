# κ型scheduleへの接続：event数・signal・誤差評価・価格

## 1. 原論文の定義と現在の実装の対応

A `510fbacd` が記録した原論文v2（arXiv:2503.05647v2、A.2/B.2/E.1/E.3）の転記を一次参照とした。repo内PDFは本環境にPDF描画系がなく、今回は再読していない（未読を読了扱いしない）。

| 論文側 | 実装側 | 対応・注意 |
|---|---|---|
| E21：r=κλ_R²T²（ランダム部分の抽象Pauli回転総数、configuration全体で共通κ） | B `paper_event_mapper` のFULL_H分岐：`ceil(κ(λ_R T)²)` をevent総数として返す。A `pr_rpe_rmse.paper_event_mapper` も同じ整数 | K2 paired eventは次数0/2とも抽象回転1個（A確認）。event数=回転数の単位 |
| E.3：共通上界B≤exp(1/κ) | `rte._paper_normalization_upper_bound`=exp(τ²)/event。総和は(λT)²/s≤1/κ_eff≤1/κ | finite K2のb(τ)=1+τ²−(7/72)τ⁴+…なのでlogBは包絡より小さい。binary64では1 ULPの余裕が必要（後述） |
| E22：費用 ∝ exp(2/κ)(A+Bκ)、停留条件Bκ²−2Bκ−2A=0 | 未実装（κ optimizerなし） | κ*=1+√(1+2A/B)。A（κに比例しない1回路あたり費用）→0でκ*=2 |
| B.2：N_m=ceil(B²[11+4(M−m)]) | A `pr_rpe_rmse` のnominal N（経験的モデル） | 十分Nではない。今回Nは計算していない |

純RTE（Full-H）には決定論PF部分がなく、Aに当たるのはwrapper・scalar・group境界程度である。このため解析モデルではκ≈2が自然な第一候補になる。既存研究の対応値も近い。T2.0最適化のH4 q8/R448/K2はκ_eff=2.026、H6 q32/R448/K2は2.109、R512確認点は2.316だった。旧4方式screenのR2-ADAPT（実行されず）はm6/m12でκ_eff 2.06/2.03である（[KAPPA_CANDIDATES.json](KAPPA_CANDIDATES.json) `existing_schedule_context`）。ただし、これらは旧会計・旧loweringでのT=2.0単一時刻の点比較である。整数化、finite-K、実decoder risk、新loweringのcompiled価格を含めた最適値は未確認であり、**κ=2を採択しない**。

## 2. λの定義とSOURCE

`prepare_full_h` は、DF二体の対角Pauli成分にone-body Z成分を加えたうえでidentityを抽出し、残る非scalar成分の |係数| の総和をλ_Rとする（`lam = fsum(c.coefficient_abs for c in components)`）。H4の値は21.02794414280327で、9b83 `input_and_validation_summary.json:H4.lambda_R` に保存されている。D `saved_full_h_preparation` は再計算せず転記する。samplerの成分確率はp_j=|a_j|/λ_Rで（検証済み）、正規化のτ=λ_R·dtにも同じλを使う。

κの意味はこのλに結び付く。PR tailのλ、B3のλ（13.72）、一体項を除いたλは使えない。D `check_full_h_preparation` は、one-body L1を含まないλとtail-only λの流用を拒否する。本helperも `FULL_H_INCLUDING_ONE_BODY` のspec以外を拒否する。

## 3. 既存mapperとgroupingの区別

| 既存関数 | κからの出力 | Full-Hへの不足 |
|---|---|---|
| B `paper_event_mapper(spec,row)` | FULL_Hでは総event数 `paper_total_target` のみ、R_outer=None | grouping q／group内Rを持たない。C `prepare` はFull-Hの各roundに `physical_layout(q,R)` を要求する |
| A `pr_rpe_rmse.paper_event_mapper` | q_PFごとのR_outer・signed allocation | 引数名がqPF。Full-Hのgroupingへ使うと意味を取り違えやすい |
| D `full_h_rte_spec(kappa=...)`＋`layout_mapper` | 明示q/Rをそのままlayoutにし、供給κをechoする（`D_FULL_H_EXPLICIT_PHYSICAL_LAYOUT_V1`） | κからq/Rを作らない（D自身はκを生成・逆算しない設計） |

**最小差分**として新module [`full_h_kappa_layout.py`](../../../src/trottertracks/full_h_kappa_layout.py) を追加した。総数はB mapperの値をそのまま使い、exact Fraction式との一致も検査する。そのうえでgroupingとgroup内Rへ分ける。

- 総event目標 r_m=ceil(κ(λ_R T_m)²)。B mapperの値をそのまま使う。
- grouping q_m：既定 `DYADIC_RPE` は q=2^m。現R448と同じgrouping・境界数を保ち、変化をRだけに限定するため。`EXPLICIT` は任意の整数q（例：q=1）を指定する。
- group内R_m=ceil(r_m/q_m)、総event s_m=q_m R_m。r_m≤s_m<r_m+q_mで、最小event数はq（R≥1）。
- Full-Hは正時間S2の単一pieceなのでallocationは[R]だけで、signed piece配分は生じない。finite-Kの次数{0,2}は各eventでlawからsampleされ、決定論的な次数配分規則はない。
- κ_eff=s/(λT)²≥κ、包絡(λT)²/s≤1/κ、τ=λT/sをexact rationalとfloatで記録する。finite normalizer・signal・N・価格はnull。

ideal finite lawの平均は s=qR とKだけで決まる（固有基底でP(x·T/s)^s）。一方、回路identityと価格はqとRを別々に持つ。groupの境界ではRTE sequenceが切り替わり、C価格モデルの `OUTER_BOUNDARY` 項に当たる。q=1と2^mではconfiguration IDもphysical IDも異なり、artificial testで確認した。

κ=2・DYADICのlayout（全roundと他κは [KAPPA_CANDIDATES.json](KAPPA_CANDIDATES.json)）：

|m|T|q|R|s|R448比|κ_eff|τ|(λT)²/s|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|0.1|1|9|9|0.020|2.035|0.234|0.4913|
|3|0.8|8|71|568|0.158|2.007|0.0296|0.4982|
|5|3.2|32|283|9,056|0.632|2.0001|7.43e−3|0.49999|
|6|6.4|64|566|36,224|1.26|2.0001|3.72e−3|0.49999|
|9|51.2|512|4,528|2,318,336|10.1|2.0001|4.64e−4|0.49999|
|12|409.6|4096|36,223|148,369,408|80.9|2.000004|5.81e−5|0.499999|

m0〜m2は整数丸めでκ_eff=2.035になり、m5以降は2.0001以下に収まる。κ=1/4も同じ規則で生成した。

## 4. canonical/physical identityとlaw binding

既存route `D.full_h_rte_spec(kappa=K.kappa_descriptor(κ))` → `K.layouts` → `C.configure`（=B.generate＋D layout_mapper）→ `B._validate` → `C.prepare`（D physical_schedule＋_check_against_b）を、本ブランチでproduction変更なしに通した。

- configuration IDはspecのκ（kind `PAPER_TOTAL_ROTATIONS`、exact value、定義/SOURCE）と全roundのlayoutを含む。κ2=`54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e`、κ1=`f4fec351…`、κ4=`e442b9a5…`。R448（`cfa6aee3…`、κ=None）とは別identity。
- 各roundの `law_binding.kappa` は同じκを持ち、`event_mapping.kind=D_FULL_H_EXPLICIT_PHYSICAL_LAYOUT_V1`。B `select` では、Full-Hの明示layout family（R448）とκ供給familyが分かれる。κ1/2/4は同じkind/definition_sourceを持つので、同じfamily内のinner κ候補として比較できる。
- physical candidate（`FULL_H_`＋cid）はq、R、K、T、s、familyだけで決まり、κを含まない。同じlayoutならκの有無にかかわらず同じphysical IDになる（test確認）。κ候補はR448やT2/light/R2-ADAPTのphysical IDと重ならず、D `saved_asset_status` は全て `NO_SAVED_EXACT_CANDIDATE`。
- κ供給routeでは、B.generateが `_explicit_layout`/`_explicit_preparation` を実行しない。そこで `check_configuration` が生成後にこの2つ、κのecho、q_PF/R_outer=None、event数一致を検査する。R448の明示layout routeと同じ強さの検査になる。
- RMSE policyはR448・PRと同じA S0 policy（`minkowski`、`finite_B_diagnostic`）を変更せずに保持した。`paper_kappa_envelope` への切替はB/ユーザーの判断事項である。

## 5. signal・誤差評価への接続

- **同じH/state/snapshot/target/referenceを使える。** H、state、snapshot、DF10/sector、h=binary64 0.1、M12、P0、元full DF-Hのground-energy targetはR448と同一で、T_mも同じである。保存13 reference pair（`bba15083` PHYSICAL_INPUTS）を同じdescriptorで使え、新しいreferenceは0。builderはκ候補のreference descriptorがR448と同一であることを `C.prepare` で確認した。
- **数値経路は無変更。** `baseline_signal_numerics.evaluate` は任意のq/Rで (P(dt)^R)^q とnormalizer b^{qR} を固有基底で評価し、event列を作らない。κ2 m12のR=36,223でも計算量はR448と同程度である（R448実行ではB0I＋Full-Hの26 meanで4.15 s）。normalizerは異なるτごとに1回で、DYADICでは13回。
- **有限K平均とnormalizer。** corrected=Σ|k⟩⟨k|P(x_k T/s)^s|ψ⟩·e^{−iT(c+s)}、raw=corrected·e^{−logB}（独立伝播）、logB=s·log b(τ)。0<logB≤(λT)²/s≤1/κ_eff≤1/κなのでattenuation≥e^{−1/κ}（κ2で≥0.607）。人工τ gridで、binary64のbがexp(τ²)の丸め値を1 ULP超える例が71点中7点あった。実行gateには既存の `normalizer_one_ulp_log_sensitivity` を余裕として加える。
- **raw信号の減衰と位相。** normalizationは正の実数倍なので、rawとcorrectedの位相は同じである。位相誤差はfinite-K打切り（次数4以上）から生じ、τ_λではなく非scalar H固有値xのx·dtに依存する。κ2はm0〜m4でR448よりevent数が2〜50倍少なくdtが大きいため、R448よりbiasは増える方向になる。その大きさはS1で実測する（今回は算出しない）。
- **K/κ変更の条件。** κを変えるとs、τ、b、logB、bias、raw減衰、physical/canonical IDが変わり、H/state/target/referenceは変わらない。K0ではb≈√(1+τ²)、logB≈(λT)²/(2s)となり、κ=1/logBの対応が成立しない（K0は別identity・別評価）。Kはこの計画ではR448と同じK2に固定し、変数をκだけにする。
- **decoder・MSEへの入力。** Bには既存 `I.signal_input` packet（configuration_id/target/source/means）を渡す。各meanのraw X/Y座標、logB、identity/law_binding（κ）、physical candidate、provider bindingを含む。Bはrawを測定確率に使い、normalizationをN・価格へ二重に乗じない。decoder、prior、branch、RMSE policyはR448/PRと同じものを使う。
- **R448の使い方。** R448は対照系列としてだけ使う。inputs、reference pair、数値SOURCE、caller様式、gate、packet形式は再利用するが、R448のsignal/logB/N/価格をκ候補へ移さない。R448とκの差は価格差だけではなく、raw減衰とshot数そのものが変わる。

## 6. compiled価格への接続

- **新lowering。** Dの `efficient_controlled_zz`（CX·CRZ·CX、`01e068af`）は、provider moduleの可逆swapでFull-H composeにも及ぶ（call-time importのため）。旧Full-H setupはswapを呼ばないので、新callerはcomposeの範囲だけを `efficient_controlled_zz(df_rte_qiskit)` で包み、例外時にも戻す。生成回路に旧 `RZZ.control(1)` が残っていないことを `instruction_parity` 等で記録する。
- **旧Full-H価格との違い。** light（T.8、q16/R16/K4ほか）とT2.0（q8/R448/K2ほか）は旧lowering・別T/q/R/K・別seed family/cohortで取得されている。κ候補とはphysical IDが一致しない。新ZZ価格・平均・上界へ変換せず、旧価格からP0の局所差（17〜19 RZ/回転）を引く操作もしない。
- **C比較guard（`3a8b37f1`）は変更なしでκ layoutを解決する。** `price_implementation_guard.check` はround mappingからlayoutを復元し、D physical_schedule/_check_against_bで照合する。toy κ2 configurationで通過し、PR旧/Full-H新のmixed policyを拒否することを人工testで確認した。必要なsidecarは既存契約のままである：price source、identity、law_binding（κ）、axis、cohort、physical context（split/law/structure/q/R/K）、`rte_rotation_lowering=CONTROLLED_ZZ_CX_CRZ_CX_V1` と `rte_lowering_source=01e068af:efficient_controlled_zz.py`、native/wrapper/scalar SOURCE、`scalar_application=ONCE_AT_TOTAL_T`、measurement。
- **full measured Hadamard X/Y wrapper。** 既存 `build_native_hadamard_wrapper(axis, include_measurement=True)` を使う。P0のためstate準備は除き、ancilla H、basis、control phase、測定は含む。scalarはc_H＋抽出Full-H identity（合計−0.33147781341681615）を総Tで一回。LD2用tail-only scalarは使わない。
- **同一性条件。** compiler（Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、hash `1239304c…`）、control `symmetric_directional`、wrapper、P0、RTE lowering、native compose SOURCE（`af284d85:h46_long_time_native.compose/cost_trial`）、Full-H preparation（9b83）、scalar quantityが、比較する全method（PRも）で同じpolicyに入っていること。compiler hashの一致だけでは同条件にならない（D S1）。
- **既存の価格評価器の再利用性。** C `rpe_mean_price_model.quote_model` はFULL_H_RTEをADDITIVE（RTE_BODY×event_count、OUTER_BOUNDARY×(q−1)、SCALAR、WRAPPER）で扱い、`q_variable=event_count` とする。C `mean_compiled_price.full_wrapper_mean`/`MeanKernel.repeat` は閉状態kernelの冪乗で、q·R規模のevent列を作らずに期待構造価格を評価できる。ただしFull-H用のevent kernel（基底変換の境界状態を含むAccounting）は未整備で、H4 PR用Accountingの流用は未検証。
- **長時間roundの評価法。** κ型ではs∝4^m、論文の経験的Nの重み(11+4(M−m))はκに依存しない。event比例価格を仮定すると、Σ_m N_m C_m の97%がm10〜m12（s=9.3e6〜1.5e8）に集中する（m12 67%、m11 23%、m10 7%；モデル算術）。直接compileできる規模（既存Full-H directは旧loweringでH4最大s=4,096、H6最大14,336。κ2 m6は36,224）から4桁以上の外挿になる。
  - 推奨はgroup分解である。groupは独立に同じlawから生成されるので、E[C_m]≈a＋b·s_m＋c·(q_m−1)。DYADICではgroup内R_m≈κλ²h²2^mとなり、**m12の1 group（R=36,223）はκ2 m6全体（s=36,224）と同規模**で、単一groupは直接compileできる。
  - 外挿は、i.i.d. groupの反復と境界の局所性（compiler opt1の局所pass）という仮定に縮む。仮定は検証可能だが未証明で、**UNQUALIFIED_MODEL scenario**として扱う。短い回路の価格をqやRに比例させる式は検証済みの事実ではない。
