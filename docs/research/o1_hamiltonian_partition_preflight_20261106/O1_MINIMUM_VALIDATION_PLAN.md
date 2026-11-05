# O1：候補ごとの最小検証と既存SOURCEの再利用

2026-11-06 JST。**将来の条件付き計画であり、実行許可・実行契約・正式な算法仕様ではない。** 今回は静的SOURCE読解、保存報告の引用、文献照合、4文書の作成・公開・内容照合まで。新しい科学数値、rank、分子、精度、失敗確率、compiler、scheduleは決めていない。新科学計算・compile・再解析・重要レビューは実施していない。

候補の意味と全体費用への経路は[機構](O1_IMPROVEMENT_MECHANISMS.md)、元条件と結果commitは[比較表](O1_EXISTING_METHODS.md)に示す。本計画を、以下すべてを実施する必須backlogとして扱わない。

## 1. 今回閉じたことと、次に判断が必要なこと

| 区分 | 今回確認したこと | まだ確認していないこと |
|---|---|---|
| 資料 | Iの必須4成果物、Sの必須4資料、正式指示・入口・該当研究文書を読解。Git objectから固定SOURCE/原結果を限定参照 | private raw/全seed/旧実行環境の完全復元。SERVER 1の現物監査はしていない |
| M1 | 同じ物理D/Rを記述する既存設計、S/P代数・router・有限mean・literal価格・F2 actual価格と統計資格のinterfaceを確認 | 新分子辞書、元H state上の全law、保証されたNと全round full費用 |
| M2 | 既存DF/RC-DF/PR部品、J/U係数map、一体/scalarの条件、研究Cのown-DF state選択を確認 | 任意fitted coreの完全残差接続、その元H state上のPF/RTE・native価格 |
| 既存負結果 | F2のN=null、core/pair、P-C/P-D/R3、研究C native fit NO-GO、singleton一時終了を保持 | それらを覆す新しい科学結果や再開判断 |

技術的に実施できる部分が存在することと、強baselineより安くなることは別である。現時点で両候補の科学実行GOは出していない。

## 2. 共通の比較条件

次の科学実行をユーザーが選ぶ場合にだけ、その対象と最小条件を定める。

1. **同じ元H・state・energy task。** 積分規約、scalar、一体補正、sector、spin/JW順、基底変換を対応付ける。state準備は除外するがstate oracleを近似H groundへ増やさない。保存stateを診断に使う場合も、選択algorithmがその古典vectorを無料取得する仮定にしない。
2. **同じ情報と取得機会。** 既知UWC変換、collected Pauli/DF/適用可能なRC-DF、同じcore/辞書候補とerror/価格情報、合法なcontrol/fusionをbaselineにも開放する。未知E₀/gapを片側だけに与えない。fit/collect/情報取得・探索費・cold/hot条件を記録する。
3. **同じ成功要件。** 固定時間の複素信号精度、energy RMSE、指定成功確率を混同しない。全roundのbias・半径・shot・branch/aliasを同じE₀へ結ぶ。理想meanと実装law・native誤差も分ける。
4. **指標別の全価格。** RZ/CX/T/depthを独立に、full controlled X/Y wrapper・scalar・D sweep・boundary・readoutまで数える。Tを求める場合だけ合成精度と実価格が必要。literal回転数をTへ換算しない。
5. **保証の強さを合わせる。** 探索pilotはその段階の誤差・標本不確かさを明示すればよい。最終成功保証を主張する段階では、未観測law/rare event、較正移送、decoderの条件まで必要になる。既存の経験的PF係数をrigorous証明へ格上げしない。追加の承認証明・認証基盤は作らない。

強baselineは単一の勝者を今選ぶことではなく、同じtaskで適用可能な既知法の対照集合を正しく強化することを指す。既存停止familyの再実行は含めない。

## 3. M1の最小検証

| 問い | 既存資産で静的に確認できること | 新科学実行が必要になること | GO/STOP判断に必要な結果 |
|---|---|---|---|
| 同じHRを比較できるか | matched partition設計のT_eff/V_squareとT_bare/V_normal、一体補正・identity方針、S/P exact代数、元入力/SOURCE版 | 選んだ入力のT/V別operator・辞書和・scalar・state対応の数値確認。生成も別許可 | 共通HだけでなくHD/HR・scalarが対応。不能なら辞書単独効果のclaimをSTOPし、比較の交絡を表示 |
| 補正meanを保存するか | paired polynomial、K偶数・degree K+1、ordered event/phaseとactual adjointの既存実装 | 実装law・finite-bit・controlled first moment・元H state信号の確認 | 同じ物理splitとfinite targetの範囲を閉じる。channel一致だけ、位相fit、無断sector射影ならSTOP |
| Nと価格の交換が有利か | F2単回価格、N未資格、literal DPとactual価格の差、既存RPE会計のunit-radius制約 | 対象条件でのPF/finite mean/半径、全round N、law別full価格・希少層の必要な確認 | 同精度成功条件を満たした指標別Gが強対照より低いという、宣言した不確かさに見合う根拠。価格差が識別不能ならUNDETERMINED |
| 既知法との差があるか | 既存two-endpoint/同poolのselector、UWC既知機会との比較 | 必要と判断された限定比較のみ。新selector探索は含めない | 既知法が同じpoolで同じ利益を再現すれば独立新法claimはSTOP。応用の成立条件として残すかはユーザー判断 |

最も強い対照は、同じHD/HR上のglobal collected Pauliとrotated-DF辞書の両端点に、同じcontrol/fusion・canonical finite-RTE・同情報の既知選択を与えたもの。その上でtask全体では適用可能なUWC/強Dとも比較する。既存F2でPPが低単価だったことを尊重し、mixed内点を必ず追加する計画にはしない。

**最小の次段階は、既存の同一物理split候補と状態条件を採るかという判断である。** 分子・精度を今追加せず、新しいfixture・N再計算・samplingを自動開始しない。singleton/PS48を使う案が必要になれば、元の正式一時終了と再開範囲を別途扱う。

## 4. M2の最小検証

| 問い | 既存資産で静的に確認できること | 新科学実行が必要になること | GO/STOP判断に必要な結果 |
|---|---|---|---|
| 適合coreがあるか | 既存DF/RC-DF/研究C保存coreの構造、J/U/一体補正/scalar、元Hとの差の定義、固定STOP | 新しいcore fit/rank選択、残差tensor/Pauli生成は別許可 | 適合する既存coreがないなら、実行準備へ進めず保留。新optimizerを便宜的に作らない |
| 元Hを推定できるか | exact residualを含める記帳と、近似Hをそのまま推定する枝の違い。研究Cはown-DF groundであること | 元H state上のcore＋残差mean、PF位相/半径、残差/合成/finite-bit誤差の転送 | 元H/state/sectorを保ち、元Hのenergy精度・成功条件に接続。近似H groundや未知E₀補正に依存すればSTOP |
| 圧縮利益が残差費用を上回るか | 既存PR-2のbackbone/tail構造、N1 exact fallback、研究Cのnative価格負結果 | 適合core一つと必要な強対照のfull価格・全round N/G。新しい大域探索は最小計画に含めない | 残差λ、PF step、basis費、Nを払った後に指標別利益が残るか。残らなければその条件でSTOP |
| 同情報baseline後の差か | UWC、既知factorization-residual法、同candidate強D、discard+scalarの比較条件 | 選んだ同じ候補集合での必要な公平比較のみ | 同じcore/residual選択で既知法が再現するなら独立新法claimをSTOP。どの有効条件が得られたかだけを報告 |

最も強い対照は、同じcore候補・元H情報を使う既知residual PR、同candidate強D、collected Pauli/UWCの再最適化、適用可能なDF/RC-DF endpoint。DF fitting誤差を無視したcore-only方式を弱い対照にしない。近似Hを許すbaselineなら同じ元Hへの誤差転送を支払わせる。

**研究Cの新cost-aware fitを準備しないという既存技術NO-GOは維持する。** M2は新η、近いbasisの無料共有、weighted-price目的、rank sweepを採択する理由にはならない。将来の検証には、同NO-GOと異なる対象・比較の必要性を具体化したユーザー判断が必要である。

## 5. 再利用する資産と適合性

3環境統合の[再利用地図][reuse]を入口に、次の本文/関数を静的確認した。S=`e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8`は今回読んだ収録snapshot。元科学結果を再現する際には[比較表](O1_EXISTING_METHODS.md)の実行SOURCEと依存closureを使い、異版を無断で混合しない。以下のtest/runnerは所在の確認であり今回のPASSではない。

| 候補・profile | 既存関数／SOURCE・入口 | 入出力・再利用できる範囲 | 適合性と最小差分（未実装） |
|---|---|---|---|
| M1 / MD-INPUT | [prototype.py][prototype]の`make_fragment_alternatives`、`canonicalize_arm`、`make_tail`。SOURCE `9468c708…`。同版`algebra.py`、`fixture.py`、`tests/test_pr_dictionary_f1.py` | rational fragment/frame→S/P atom・collect済tail・RTEComponent、exact rational λ | 元4mode fixtureの精度・JW順・identityを保持。一般分子の素材が適合しなければ、既存serializerへ渡す入力adapterだけを検討。scalar/threshold policyを黙って変えない |
| M1 / MD-CONTROL/F2 | 同SOURCE `circuits.py`の`build_event_sequence`/`append_deterministic`、`gates.py`の`compiled_wrapper`/`component_derived_mean`、`f2_resources.py`の`execute_resources`。`tests/management/test_pr_dictionary_f2_resources.py` | typed atom/law→full-Fock/control/X-Y wrapper・actual単回価格 | phase修正版を使う。旧metadata/phase不具合の再実装は不要。元F2 runnerは固定入力・元契約を持つので再起動せず、必要な純機能の呼出し境界だけ将来検討 |
| M1 / MD-STAT | 同SOURCE `f2_statistics.py`の`require_adopted_statistics`/`qualified_shots`、`intervals.py` | 資格済みrange/statistical入力→N。旧F2では未資格 | 関数があってもgamma/Nを自動認定できない。対象の信号と成功要件に適合する既存統計処理を優先。旧資格STOPを改変しない |
| M1 / MD-INPUT価格 | [pricing.py][pricing]の`literal_rz_count`、`expected_literal_price`、既存二端点`select_dictionary` | canonical次数・component確率・recipe→literal期待価格 | native full価格やdepthの一般oracleではない。F2 actualと区別。新DP/selectorを作らない |
| M1/M2 / RTE-BASE | [S:rte.py][rte]の`finite_rte_distribution`、`make_rte_config`、`iter_rte_events`、`finite_rte_operator_moments`、`compose_finite_rte_occurrences`。`tests/test_rte.py` | symbolic tail→finite law/normalization、補正meanとraw attenuationの区別 | involution/sign/ordered phase/identity/zero-tail・finite-bit条件を維持。deprecatedなmean wrapperを避ける。dense `normalize_involutory_tail`を大系へ強制しない |
| M1/M2 / Pauli | [S:pauli_signal_adapter.py][pauli]の`PauliHamiltonian.from_terms`/`from_spatial_integrals`、`append_pauli_event`、`paired_polynomial_action`/`sparse_mean_action`。`tests/test_pauli_rte_event_extension_mean.py` | 保存Pauli/model/state→回路・独立paired-polynomial mean | NumPy/SciPy/Qiskit依存。from_spatial_integralsやmean呼出し自体が科学処理。今回未実行。新しい共通T/残差入力の接続差だけ将来検討 |
| M1/M2 / DF-CONTROL | [S:df_partial_s2.py][dfprep]の`prepare_df_partial_s2`、`df_rte_qiskit.py`の`QiskitDFRTEEventCircuitBuilder`、`df_partial_s2_repeated.py`の`QiskitDFPartialS2RepeatedCircuitBuilder`。原再利用map固定版`2978e2fea672b7a1ff20cac74269ec9a610159dc`。`tests/test_df_partial_s2.py` | 保存DF partition→one-body/D half/tail/control/反復とmetadata | Sにはexplicit ordered partitionもある。PauliをDF Z/ZZに偽装しない。一般core＋任意残差のtyped接続は別途必要か照合。新builder一式は提案しない |
| M2 / 研究C fit・codec | [S:c_pr_aware_df.py][cpr]の`coefficients`/`make_value_and_grad`/`fit`、`c_rcdf_comparator.py`の`full_l2`/`make_value_and_grad`/`fit`、`c1_ffsim_corrected_codec`との既存接続 | signed J/U・元積分→係数・fit/codec。既存ffsim/JAX loss/ADを再利用 | PRのtail L1型とRC-DFの全frame L2正則化は別。fit wrapperは固定条件を持つ。M2では適合する保存coreを先に探し、新fit自体を最小前提にしない |
| M2 / 研究C下流・native | [S:c_pr_integrated_evaluation.py][ceval]の`Evaluation.assess`、[c_pr_integrated_performance.py][cperformance]の`run`、stage B/B2/M3既存価格経路。`scripts/run_research_c_pr_integrated.py`、`tests/management/test_research_c_pr_integrated_pipeline.py` | 既存入力→DF budget、candidate PF/RTE、wrapper価格、round会計 | hardcoded rank8/H4 sector36・own-DF ground等がある。元H state＋完全残差を表現する最小adapterが必要か未確定。既存固定runnerの引数だけ変えて起動しない |
| M1/M2 / RPE-COST | [S:rpe_resource_accounting.py][accounting]の`evaluate_rpe_round_candidate`/`build_rpe_resource_summary`、[full_opt2][fullopt]の`load_and_validate_full_opt2_evidence`。`tests/test_rpe_resource_accounting.py` | 候補誤差・半径/shot・provider期待価格→round/全round会計 | 旧APIは二次DF・unit-radius specialization。一般半径とerror入力を持つ後続SOURCEを先に照合し、足りない場合だけ該当引数/provider境界を拡張。会計frameworkを作り直さない |

後続18round decoder/enclosure/energyの所在は[統合台帳][ledger]のSUP-RPE-V1/V2/V3R1/ENERGYから辿る。今回その全closureを再読解・再実行した意味ではなく、旧短round APIだけに戻って「接続未着手」ともしない。保存energy bridgeの条件付き保証を、真の分子energyや実装全体へ拡大しない。

新実装が必要になり得る最小境界は、M1では**既存の共通T/HRと辞書入力の接続**、M2では**保存coreと元H残差・元H stateの接続**、共通では**対象signalの半径/誤差を既存会計providerへ渡す部分**である。既存資産で賄える場合は追加不要。いずれも今回実装しておらず、採択前に新router・fit・管理層・承認機構を作らない。

## 6. 将来の判断と停止

科学実行を選ぶ場合、最初から全分子・全rank・全辞書を走らせる必要はない。ユーザーが意味のある対象と既存core/共通splitを選び、最小の比較で成立・反証を識別できる範囲だけを明示許可する。科学条件を変える追加stageはそのとき判断する。現存する固定SOURCE/one-shot/retry/STOP条件は保全する。

- **GOに必要な証拠**：同じ元H energy taskの成功要件を満たし、情報取得・normalization・全wrapperを含めても、同情報の強対照に対する指標別利益を示すこと。pilotの条件付き利益と最終保証を分ける。
- **STOPまたは保留**：mean/target/state不一致、非正半径、誤差予算超過、oracle依存、相殺で利益なし、差が不確かさ以下、全law/価格が未資格、あるいは具体的な追加操作がない。未資格を0にしない。
- **ユーザーの次判断**：M1/M2の既知法応用のどちらに検証価値を認めるか、元H・共通split/core・対象精度/成功確率・資源指標をどう選ぶか。必要な重要科学レビューの開始許可と、新科学計算の実行許可は別である。
- **Codexの状態**：本4文書の公開・remote内容照合まででmandatory STOP。次の実装・数値処理・compile・詳細科学レビューを自動開始しない。

[reuse]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/DUPLICATION_AND_REUSE_MAP.md
[ledger]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/IMPROVEMENT_VERIFICATION_LEDGER.csv
[prototype]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/prototype.py
[pricing]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/pricing.py
[rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/rte.py
[pauli]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/pauli_signal_adapter.py
[dfprep]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/df_partial_s2.py
[cpr]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trottertracks/c_pr_aware_df.py
[ceval]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trottertracks/c_pr_integrated_evaluation.py
[cperformance]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trottertracks/c_pr_integrated_performance.py
[accounting]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/rpe_resource_accounting.py
[fullopt]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/research_direction_full_opt2.py
