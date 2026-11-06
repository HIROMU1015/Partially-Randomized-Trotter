# M1：最小科学検証の接続準備と未決定条件

2026-11-06 JST。**保存入力候補と最小差分は特定したが、H4-A/B・科学条件は未採択、接続は未実装、科学実行は未承認・未実行。** これは完全な実行契約でも新算法仕様でもない。[入力・共通splitの対応](M1_COMPARISON_SETUP.md)と一組で読む。

## 1. 最初に比較できるもの

候補は、公開済みH4 anchorの既存案A（T_eff/V_square）または案B（T_bare/V_normal）を一つ選び、同じ物理splitのP/S二端点を対にするものである。両案を実行する採択はしていない。最小の初段はfull H、T、V、辞書和、scalarの独立したoperator対応確認だけに限定できる。ここでは新時間点、shot、compiler、価格cohortは必要ない。

この初段自体も、保存配列からの数値operator再構成を伴う**新科学計算**である。今回行ったZIP/NPY header・bytes・metadata確認とは区別する。適合する一つの入力から始められるため、新SCF/DF/state生成やnew fixtureを前提にしない。

P0/F2は修正版full operator検査と価格境界の参照に使える。ただしprivate physical/arm/bundle payloadは今回未取得、fixture stateはoccupied(0,1)、価格はprep込み、Nはnull。P0を最小分子energy検証として直接実行可能とはしない。

## 2. 固定SOURCEの機能と必要最小接続

以下は[3環境再利用map][REUSE]のMD-INPUT/CONTROL/STAT、RTE-BASE、DF-CONTROL、RPE-COSTを使った限定照合。`aeb9d184…`は今回読んだ収録snapshotであり、過去runの実行SOURCEに置き換えない。既存関数・testの所在を読んだだけで、今回のtest PASSを主張しない。

| 再利用する機能 | 具体的SOURCE・関数 | 今回確認できた境界 | 必要なら加える最小差分（未実装） |
|---|---|---|---|
| P0辞書素材・collect | F2修正版SOURCE `9468c7089727f5648d9f59cddbaebe0a42f5f0a0`：[prototype.py][PROTOTYPE]の`make_fragment_alternatives`、`canonicalize_arm`、`make_tail` | `canonicalize_arm`はSS/SP/PS/PPと2fragment限定。Pだけglobal wordでcollect、Sはframe/originを保持。`make_tail`はexact rational λとfinite-bit component情報を分離 | H4へP0 selectorを一般化しない。両端点のexplicit HR入力を既存Pauli/DF経路へ渡す小さい接続を検討するだけ |
| P0物理identity・reference | 同SOURCE [reference.py][REFERENCE]の`original_operators`、`polynomial`、`finite_target`、`dictionary_operator`。F1実行SOURCE `8192530b17d6bc4af195c734ff43a6e78c506942`の`f1_saved_execution.SavedF1` | P0は4mode/2粒子、diagonal D、2つのsquare R、faithful identity tail。F1 K2/q2とF2 K0/q1は別law | full operatorとcontrolled phaseの検査考え方を再利用。H4にoccupied stateやP0 cutoffを転用しない |
| 保存DF→block/tail | [df_hamiltonian.py][DFH]の`DFHamiltonian`、[df_partial_s2.py][S2PREP]の`prepare_df_partial_s2`、[df_rte_tail.py][TAIL]の`exact_df_diagonal_coefficients`/`extract_df_diagonal_tail` | 保存one_body/λ/GをDF型へ載せる素材がある。通常preparationはone-bodyを保持しfragmentを分割する。default identity抽出とP0のfaithful policyは別 | Aでは保存配列から一体保持＋全square tailへの接続。BはD̂をtailのI/Zへ接続する差分が必要。DF入力再生成・new factorizationは不要 |
| Pauli residualとmean | [pauli_signal_adapter.py][PAULI]の`PauliHamiltonian.from_terms`、`append_pauli_event`、`paired_polynomial_action(normalization=1)`、`sparse_mean_action` | 既存全H prefix adapterは新HR入力ではない。polynomial作用はsampler/compiled circuitと独立の名目reference | 選んだ共通Tと明示VのPauli入力、および共通T halfを受ける部分だけ接続。旧prefix27を使わない |
| finite-RTE lawとnormalization | [rte.py][RTE]の`finite_rte_distribution`、`make_rte_config`、`finite_rte_operator_moments`、`compose_finite_rte_occurrences`、`exact_enumerated_event_mean_operator` | cutoffは偶数、paired polynomialはK+1次数。補正meanとraw attenuationを別fieldで返す。heterogeneous occurrenceのBも既存対応 | 既存APIを使い、辞書ごとの正しいλ/configを渡す。deprecated `finite_event_mean_operator`を新接続に選ばない。dense small referenceを大系の無料oracleにしない |
| controlled DF/Pauli・actual adjoint | [df_rte_qiskit.py][DFBUILDER]の`QiskitDFRTEEventCircuitBuilder`、DF partial S2/repeated builder、Pauli `append_pauli_event`、P0 [circuits.py][CIRCUITS]の`build_event_sequence`/`actual_adjoint` | typed中央dispatch、basis相殺、event順序、符号・scalar相対phaseを保持する資産がある。任意physical PauliをDF Z/ZZとして偽装しない | 共通保持program＋arm別residualの合成境界だけ接続。同じT operatorでも別programなら、その交絡を表示 |
| no-prep X/Y wrapper | [rpe_hadamard_interrogation.py][WRAPPER]の`QiskitRPEHadamardInterrogationBuilder` | 既存wrapperはsystem prepを含めず、制御を重ねない。公開`build`は短round domainとDF repeated-result metadataを検査する。P0 wrapperはprep込み | P/S双方に同じno-prep wrapper骨格を適用する小さい型/入力境界が必要か確認する。DFの証拠型を偽造しない。P0価格からprep費を差し引く方法は採らない |
| literal期待価格とactual価格 | F2 SOURCE [pricing.py][PRICING]の`expected_literal_price`/`literal_rz_count`、[f2_execution.py][F2EXEC]の`measure_f2_circuits`/`aggregate_metrics` | literal grammarとnative compile価格は別。F2 actual処理はP0 4arm・K0・元one-shot契約へ固定される | 既存metric collectorを再利用し、P/S/no-prep対象の入力接続だけ検討。旧F2 runnerのcontractを編集・再実行しない |
| qualified shots | F2 SOURCE [f2_statistics.py][STAT]の`require_adopted_statistics`/`qualified_shots`、`intervals.py` | η/α/γ/interval資格がないと拒否する。関数存在は資格獲得でない。既存F2はN=null | 新条件で必要なsignal/mean/range/biasを科学的に資格付けた後にだけ入力する。承認証明・認証基盤を作る意味ではない |
| scalar・fixed-T N | [fixed_signal_accounting.py][FIXED]の`fixed_signal_shots` | bias/allowance/log_Bと根拠を受け、UNKNOWNならNはnull。条件付きfixed complex signalの会計で、全energy成功やcertificate検証ではない | 対象H/state/timeの新結果を渡す接続に限る。既存default axis配分を新scopeへ黙って採用しない |
| RPE round/全round会計 | [rpe_resource_accounting.py][ACCOUNT]の`evaluate_rpe_round_candidate`/`RPERoundCostProvider`/`build_rpe_resource_summary` | evaluatorはDFPartialS2Preparation・二次PF、`reference_radius=1`のspecialization。summaryも同じpreparation/δ/guarantee/価格modelを共有する候補だけを合計 | 新辞書に必要な一般半径・error・価格入力を既存後続SOURCEとまず照合。不足するfield/provider境界だけを将来拡張し、会計全体を再実装しない |

`make_fragment_alternatives`から始めるP0経路と、H4のgeneric Pauli/DF経路を、同じmoduleに存在するという理由だけで連結しない。F2のFrame/JW/phase精度とH4のfloating DF/一体補正・sectorの条件を先に合わせる。required dependencyはNumPy/SciPy、Qiskit、Qiskit Nature、OpenFermion、mpmath等の各固定SOURCEの対応版であり、共有venv/cacheを変更して揃えない。

## 3. 検証に使える既存test・runner

| 既存資産 | 既存テストが扱う内容／今回への限界 |
|---|---|
| `tests/test_pr_dictionary_f1.py` | exact serializer、CAR/JW、4辞書のH/sector再構成、exact collect、signed phase/adjoint、X/Yとstate prep、actual-block mean。P0のテストでありH4適合の証拠ではない |
| `tests/management/test_pr_dictionary_compile_phase.py` | 単回転・Gaussian・controlled相対phase。F2修正版の位相保持展開を再利用する入口 |
| `tests/management/test_pr_dictionary_f2_resources.py` | resource-only経路が統計を呼ばずNを0補完しないこと。科学N/優越の保証ではない |
| `tests/test_df_rte_tail.py` / `tests/test_df_rte_qiskit_builder.py` | I/Z/ZZとidentity、selected fragment再構成、全sign/phase、controlled block、basis reuse・異basis非相殺 |
| `tests/test_rpe_hadamard_interrogation.py` / `tests/test_rpe_resource_accounting.py` | 同じtrajectoryのX/Y、相対phase、no-prep scope、短round、union会計、価格MC数を量子shotに掛けないこと |
| `scripts/run_pr_dictionary_f1_saved.py` / `scripts/run_pr_dictionary_f2_resources_saved.py` | 元の固定P0入力・契約・one-shot実行入口。新M1実行のためにそのまま起動しない |

各pathは基点または上表のF2修正版SOURCEで確認した。今回はtest/runnerを一つも起動しておらず、将来必要な検査だけを選ぶ。固定旧runの再実行を技術testと呼び替えない。

## 4. 最小scopeの選択肢

| 将来選ぶscope | 対象・必要処理 | このscopeで判別できること | まだ言えないこと |
|---|---|---|---|
| **operator対応のみ** | H4-AかBのP/S。保存snapshotを用い、full H/T/Vと辞書和・scalar・一体補正を独立経路で確認 | 同一物理splitを実装できるか。A/Bやtarget/allowanceの不整合を見つける | mean精度、N、価格、全RPE利益 |
| 補正mean・signalまで | 上記に共通保持作用、既存paired polynomial、辞書別finite law/B、元H `state`上のraw/corrected signalを接続 | 同じ理想補正meanと、異なるnormalizationの交換。実装lawとの差 | native価格や全round成功。stateをDF_stateへ代替しない |
| no-prep full価格まで | 同じ保持program・wrapper/compilerにP/S residualを接続し、必要な代表／cohort価格を取得 | 定めた価格対象でのN-independent単価差 | 全law母平均、同精度G、元H energy成功。古いprep込み価格や部品和で埋めない |
| fixed-T N/C/Gまで | 対象signalのbias・allowance・Bと費用を既存accountingへ渡す | 宣言したfixed-time条件付き資源比較 | 全RPE energy成功・最終優越 |
| 全RPEへ接続 | 成功条件に必要な全round・branch/alias・PF/finite-RTE/native/DF誤差・半径・N・full価格を接続 | 指定energy精度・成功確率の下での指標別G | 未資格のlaw/価格・外挿をPASSにしてはならない |

scopeを一つずつ形式的に承認し直すことを要求していない。ユーザーが必要な範囲をまとめて明示許可すれば、その固定scope内の通常接続・technical debuggingをまとめて進められる。ただし今回は上表を**一切実行していない**。条件が未決定なので科学call数や契約を無理に固定しない。

## 5. normalizationから全RPEへの接続条件

既存paired lawでは、辞書dごとの`τ_d=λ_R,d δ/r`、finite normalizer `B_K(τ_d)`、raw attenuation `A_m,d=B_K(τ_d)^(-r q_m)`を、補正meanとは分ける。同じHR/time/r/Kで理想多項式meanが対応しても、finite-bit確率、native角度、raw半径、event価格は同じとは限らない。特にK0をidentity-onlyと解釈しない。[O1の費用経路][O1MECH]

全RPEでは、正当な基準半径下界とfinite-mean誤差から観測半径を評価し、その半径・統計位相予算・失敗配分を必要shotへ接続する。正のBは位相を変えなくてもshot/range費を増やす。raw半径での評価とB補正後rangeでの評価を二重計上しない。厳密な元H groundでも、近似PFの固有状態・単一位相・半径1とは限らない。

既存18round成功／energy bridgeは[統合台帳SUP-RPE-*][LEDGER]にあるため未着手扱いに戻さない。一方、新H4-A/B辞書への全law・half-block・state・error-policyの適合は新たに必要。旧legacyDF12の18round scheduleや単位半径APIから、新tol-onlyDF10の共通splitへ価格・Nを移さない。RZ/CX/T/depthは別々に合計し、classical collect/生成・価格較正費は別に記録する。

## 6. 強いbaselineとGO/STOP

M1の機構対照は、同じHD/HRを使うglobal collected Pauliとrotated Z/ZZの二端点である。両側に同じ保持program候補、scalar処理、actual adjoint、合法なbasis/run共有・control軽量化・compiler機会を与える。価格MCなら各lawとの対応と標本不確かさを保持し、同seedをpaired couplingと称しない。

task全体の対照では、同じ元H/stateアクセスと取得情報に対し、適用可能なUWC変換、同candidateの全決定論PF、既存PF family/δ/r/K/β/α最適化、discard＋既知scalar等の機会を揃える。UWCをnorm-onlyな弱い固定点にしない。ただしUWCで表現・splitまで変える比較は**同じHD/HRを隔離するM1機構比較とは別欄**に置く。既知法が同じ情報と辞書poolで同じ選択を再現すれば、独立新法のclaimはSTOPし、応用の有効条件として何が残るかを記述する。[既存文献比較][PRIOR]

| 判定対象 | GOに必要な確認 | STOP／保留になる条件 |
|---|---|---|
| 最小operator検証の開始準備 | ユーザーがA/B、共通target/allowance、scalar/identity方針、operator精度・範囲を選び、その科学処理を明示許可 | 元H/HD/HRの説明不能、D̂欠落/二重計上、未取得原本、未許可の生成が必須 |
| mean/統計資格 | 元H state、共通補正mean、実装law/range/bias、正半径、成功条件に適合した根拠 | phase fitting、sector射影で不一致を隠す、半径1の自動仮定、UNKNOWNを0補完 |
| 価格・全費用 | no-prep full境界、両lawの価格対象、同情報強対照、対象精度でNと全roundの接続 | F2 prep込み価格の流用、孤立prep/保持価格の単純減算、部品和や未観測rare-lawを母平均へ昇格 |
| 性能claim | 不確かさに見合う指標別全Gの差と所定energy成功 | 差が識別不能ならUNDETERMINED、相殺で利益なしならその条件でSTOP。局所λ/単回価格だけで総費用優越としない |

## 7. 次の判断と今回の停止

**ユーザー**が、P0 raw回収の必要性、H4-A/Bのどちらを選ぶか、共通保持／identity方針、元Hへの数値差転送、operatorのみかsignal/価格/全RPEまでかを判断する。新科学実行と重要科学レビューの開始は独立の許可であり、本資料が代行しない。

**Codex**は、この3文書の公開・取得照合でmandatory STOP。後続実装・数値再解析・sampling・build/compileを自動開始しない。M2の結論をM1の未確認条件の証拠にせず、既存SOURCE/raw/result/STOPと停止familyを保全する。

[REUSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/DUPLICATION_AND_REUSE_MAP.md
[PROTOTYPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/prototype.py
[REFERENCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/reference.py
[CIRCUITS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/circuits.py
[PRICING]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/pricing.py
[F2EXEC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/f2_execution.py
[STAT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/f2_statistics.py
[DFH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/src/trotterlib/df_hamiltonian.py
[S2PREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/src/trotterlib/df_partial_s2.py
[TAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/src/trotterlib/df_rte_tail.py
[PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/src/trotterlib/pauli_signal_adapter.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/src/trotterlib/rte.py
[DFBUILDER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/src/trotterlib/df_rte_qiskit.py
[WRAPPER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/src/trotterlib/rpe_hadamard_interrogation.py
[FIXED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/src/trotterlib/fixed_signal_accounting.py
[ACCOUNT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/src/trotterlib/rpe_resource_accounting.py
[O1MECH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/docs/research/o1_hamiltonian_partition_preflight_20261106/O1_IMPROVEMENT_MECHANISMS.md
[LEDGER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/IMPROVEMENT_VERIFICATION_LEDGER.csv
[PRIOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/PRIOR_ART_COMPARISON.md
