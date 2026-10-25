# 研究B：数値的不確かさ調査・限定native／shot費用評価の設計

2026-10-25 JST。**静的技術調査と未承認の実行案。新規科学計算、保存数値の再解析、高精度演算、回路build／compile、MC費用計測、shot数評価は未実行。**

## 1. 起点・引き継ぐ事実・今回の結論

起点は `33715848bfb83d49ec4e54d6eef481012a7441e4`、科学SOURCEは `26e8734aff3969dc8077cb058826d12394be28e1`。[22行報告][RESULT]、[CSV][CSV]、[固定契約][CONTRACT]、[同精度設計][DESIGN]、[16行結果][PREVIOUS]と関連SOURCEを確認した。CSVは保存列の閲覧だけで、新しい差分・順位・fit・統計量を計算していない。

対象はsynthetic 2-mode、one-body＋3 DF因子、N=1／mode0、L_D=2 weight-ranked prefix、T=±0.18、A2 Yoshida4／A3 published Y8m10b、n_D=1、基本q2（SABA2追加q4）、refined f2／f8、r/K=2/2・4/2・2/4、complex128、atol1e-12、phase診断無効。分子geometry／AO basis／分子DF rank探索は非該当。全169 correctness nodeが比較に先行し、22行・1239 dispatch・274 key・RTE関連各96・110 expectationを完了したローカル結果である。[RESULT]

引用値：A3＋outer Yoshida4のcorrected physical signal誤差はr2K2約4.53e-10、r4K2約8.34e-11、r2K4約7.00e-11。u_num、γ、shots、総資源は未資格／未取得。S1のfinite-RTE operator差5.09846e-13に対しnominal bound5.07865e-13、F1は6.62042e-14対6.34772e-14。これらの原因とK4の丸め水準残差は未解決のまま扱う。[RESULT]

推奨は、(A0) 既存の独立高精度部品による小さな精度・原因確認、必要な場合だけ(A1) signalに絞った包含評価、別件の(B0) full controlled wrapperの少数trajectory pilot、の順である。A0一致をcertified u_numにしない。B0の技術的PASSも同精度費用のGOとしない。**本書はこれらの科学実行を許可するものではない。**

## 2. 数値計算経路と誤差の伝わり方

以下はコードから確認した事実。リンクは起点commitのSOURCEであり、今回変更していない。

| 経路 | 実装・処理 | u_numへの関係 |
|---|---|---|
| 入力 | [input_package.py][INPUT] `load_candidate`：DF split／Gaussian準備、sector operator化、normalized tail・state生成 | 元literalのbinary64化、固有分解／基底変換、D/R/scalar分離の数値差。入力hashは同一性の記録で、誤差上界ではない |
| D/R参照 | [dense_backend.py][DENSE] `_validate`：Dはblock和、Rはλ×normalized tail、scalarはFraction和をfloat化 | 丸めた部分和と保存physical Hの違いを無視しない。Hermiticity／normのatol検査は包含ではない |
| 時間・係数 | [outer.py][OUTER] `mathematical_stages`／`DStage.append_operations`：Fraction、SABA係数は`Fraction(1/math.sqrt(3))`、PF係数も既存float由来 | 高精度化時に1/√3やYoshida係数を別の理想値へ置換すると対象列が変わる。凍結された二進係数をそのまま使う |
| 行列指数 | `_exp`：`scipy.linalg.expm(-1j*float(time)*matrix)`、指数cache | 時間変換・引数積・Padé／scaling and squaringの誤差。`exact_*`は参照の役割名で、厳密値ではない |
| D／outer積 | `product_append`が`value=matrix@value`を順次実行。独立数学gateは右積 | 各block指数と積の誤差が累積。異なる積順の一致も共通誤差を除去する証明ではない |
| finite RTE | [rte.py][RTE] `finite_taylor_operator`：K+1次までのmatrix recurrence、`matrix_power(...,r)` | polynomial／power丸め。負時間もsigned polynomialでありinverseではない |
| scalar phase | `physical_scalar_phase`が`np.exp(-1j*float(T)*scalar)`を掛ける | scalar和、時間、複素指数、最終積の誤差。共通phaseでも浮動差へ影響し得る |
| normalization | `finite_rte_operator_moments`の`exp(r*log(B))`、`normalization_product`の`math.prod` | raw／rescaling／corrected estimatorのrangeの不確かさ。**corrected_operatorはpolynomialから直接生成され、N倍rawから生成されない** |
| signal | `finite_rte_signal_validation._signal`：`complex(np.vdot(state,operator@state))` | candidateとphysical referenceの両方、state表現、内積の誤差 |
| 差・norm | `operator_error`／`_check_error`は`np.linalg.norm(A-B,ord=2)`、signalは`abs(z_c-z_phys)` | 近い二つの計算済みO(1)量の絶対誤差が小残差を覆い得る。SVD／absも包含値ではない |

[SciPy1.14.1の公式説明][SCIPY]は可変次数Padé法を示すが、呼出結果に個別の保証付き誤差半径は返さない。[Al-Mohy–Highamの解析][EXPM_PAPER]はアルゴリズムのbackward error・丸め管理の根拠であり、今回のBLAS・入力・全積のforward errorを自動認証しない。

推論上の伝播は、各因子の誤差e_jとnorm上界a_jを使う積の摂動評価に、各matrix multiplyの局所丸めを別途加える形になる。例えば`E_(j+1)≤||Â_j|| E_j+e_j||P_j||+η_j`。γ_k=ku/(1−ku)型の内積／積評価は演算数、complex演算の実装、under/overflow、BLASの前提を指定して初めて使える。単なる「block数×machine epsilon」はheuristicである。

正規化stateを仮定したsignal誤差なら、matrix誤差e_Mに加えstate誤差dの寄与を`||M||(2d+d²)`等で抑え、内積誤差も加える。候補・physicalの両方を含む`u_num≥e_c+e_phys+e_sub+e_abs`が必要になる。近接量の減算自体は正確な場合もあり、絶対誤差が必ず巨大化するわけではない。**既に丸められた二つのoperandの誤差に対し、残差の相対誤差が大きくなる**ことが問題である。Spectral normはmatrix摂動に対して1-Lipschitzだが、入力matrix差やSVD自体の誤差を消してくれない。以上は評価方法の導出で、数値上界は今回計算していない。

## 3. 何を高精度のtargetにするか

最初は**保存されたprimitive arrayと係数を二進有理数として固定したsnapshot target**を使う。H_phys、D各block、normalized tail、λ、scalar、stateを正確にliftし、caseのFraction時間と既存の二進PF／SABA係数を保持する。`Fraction.from_float`相当で実部／虚部をそれぞれliftし、`mp.mpf(str(float))`による短い十進文字列への置換をしない。

保存physical Hを、再構成したD+R+cIへ置換しない。後者との差も入力実装の不確かさとして区別する。90/128桁で新しいDF diagonalizationや別fixtureを生成する必要はない。まず保存2×2入力から独立した数学列を構築し、数値演算の影響を調べる。

このtargetは「元の十進literal／化学モデルの真値」を認証するものではない。元literalをexact decimalとみなす別targetや、厳密unitary基底からの再構築を調べる場合は、snapshotとの入力差・モデル意味を別に記録する。小さい虚部や非Hermiticityを黙って対称化・再正規化しない。将来のnative/full-Fock再構築には、同じDF recipe・basis identityとsnapshotとの整合検査が別途必要である。

## 4. u_numを扱う三つの保証水準

| 方法 | 今使える資産・作業 | 得られるもの／不足 |
|---|---|---|
| Independent high precision | [reference.py][HP]の`precision_context(90/128)`、`exact`、独立CAR／Pauli／controlled helper、mpmathのTaylor／Padé expm | **Empirical precision check**。一致しても共通target誤り・共通実装誤り・未包含tailを排除しない |
| 誤差伝播／a posteriori | expm引数の摂動、積の局所誤差、state expectation、N・phase・norm評価を上記の式で追跡 | 評価前提・各局所bound・外向き評価が揃えばcertifiedへ進める。通常floatでcondition numberや残差だけ計算したものは**Heuristic estimate**または診断 |
| Ball／interval | exact rational入力、complex ball exponential／products／expectation／abs区間 | 全必要演算・入力包含が確認できた範囲だけ**Certified bound**。入力とtargetの意味も含める |

既存実行環境のdistribution metadataだけを確認した：mpmath1.3.0、SymPy1.14.0はinstalled、python-flint／gmpy2は未導入。科学moduleのimport・constructorは起動していない。mpmath intervalの対応は公式資料でもexperimentalとされ、既存[intervals.py][INTERVAL]はscalar log／sqrt／ceil等の限定用途である。これをmatrix exponential／spectral normの保証へ移送しない。[mpmath contexts][MP_CONTEXT]

mpmath1.3.0同梱`matrices/matrices.py::mnorm`はspectral p=2が未実装。Frobenius normを旧`ord=2`の残差へ黙って置換しない。Empirical operator 2-normは高精度SVD（最大singular value）等で計算する案とする。Certified scalar biasにはoperator normの認証は必須ではなく、complex signalのballから直接包含できる。[mpmath matrix SOURCE][MP_MATRIX]

必要なら別の隔離環境でPython-FLINTのarb/acb、`acb_mat.exp`を検証する。公式資料にはrigorous error trackingとcomplex matrix expがあるが、**現環境に接続済みではない**。ライブラリの既存包含演算を小行列に使い、新しい一般保証frameworkは作らない。Installation・version固定・API確認・必要演算のqualificationは次の技術準備に残る。[Python-FLINT][FLINT]、[acb_mat][ACB]

Certifiedなsignal差の絶対値区間[L,U]が得られ、旧観測biasを正確な二進値b64として扱えるなら、`u_num=max(|L−b64|,|U−b64|)`を外向き評価する案がある。そのとき初めて既存規則`b64+u_num≤ε/2`を保証付きで検査できる。Nの上下界と実装γも別に必要。高精度間差を2倍する、atolを足す、複数桁一致だけを理由にこのu_numを返す、という方法は採らない。

研究上の推奨水準：原因探索と費用pilotの設計にはまずempirical checkで十分な情報価値がある。**保証付き適格性／同精度達成を主張する段階では、限定signal包含または成立前提を揃えた誤差伝播が必要**。未資格の間もstructural costは調べられるが、そのGは条件付き参考値である。

## 5. S1／F1のnominal bound超過を静的に追跡

[OuterDenseBackend.perform][OUTERBACK]のtail norm gate・`outer_taylor_bound`・`outer_taylor_total`と、[outer_rte_execution.py][OUTEREXEC]の予算／資格metadataを照合した。

| 原因候補 | 確認できたSOURCE事実 | 現在の分類 |
|---|---|---|
| Boundの丸め方向 | `step_taylor_truncation_residual_bound`は`exp(d log(abs(τ))−lgamma(d+1))`から再帰級数和と幾何tailを計算。停止条件`max(1e-300,1e-16*total)`は残余項の条件で、丸め誤差の包含ではない | 理論式の安全性とfloat実装の上向き保証は別。下向き丸めを排除できない |
| Occurrence／合成 | `rte_occurrence_truncation_from_config`→step bound、r回合成。`compose_truncation_residual_bounds`は`expm1(Σ r_j log1p(ρ_j))` | `log1p/expm1`は小量の評価に適するが外向き丸めではなく、総和も通常float |
| τの評価順 | config側は`λ*(t/r)`、truncation helper側は`(λ*t)/r` | 一般に演算順差の候補。ただし今回r=2/4であり、これを原因とする証拠はない |
| Matrix exponential／polynomial | exact-RはSciPy expm、finiteは別のTaylor recurrence／matrix_power | 両operandの誤差が残差に含まれる。共通誤差相殺もあり得て現資料では寄与分離できない |
| 多数の積 | S1/F1のD literal指数数は報告上180／480。D積・outer積・scalar phaseを経て差を取る | 丸め累積は有力な仮説だが、支配原因とは未確認 |
| Spectral norm | `np.linalg.norm(A−B,2)`の通常浮動評価 | SVDの診断値であり包含値ではない。相対識別力の問題を残す |
| Norm前提 | `outer_tail_bound_check`は`norm≤1+atol`を許容。報告値1.0000000000000002 | `norm(H̄_R)≤1`の厳密証明ではない。保存matrixをexactにliftした場合の前提も別途確認が必要 |
| Signed time／重複 | `outer_taylor_total`は**全**bound依存を`(ρ_j,r_j,1)`として渡す。Node共有後も各物理occurrenceのlistを保持。Signed configは別gateで照合 | 検索した経路では符号落ち・occurrence脱落の明確なバグは見つからない。数値独立検証で確定したという意味ではない |
| Nと参照定義 | finite-RTE誤差はcorrected対ideal-tail、双方に同じ近似Dとscalar。Raw対corrected誤差ではない | N積の丸めをこの残差の直接原因とはしない。参照の構成・D積・入力差を次に分離する |

Exact arithmeticでは、Hermitian H̄のnorm≤1、unitary D、同じordered列なら、step誤差ρ_jから`∏(1+ρ_j)^r_j−1`という評価が成り立つ。保存matrixのHermiticity／unitarityやnormが厳密には未包含の場合、a_jを1に固定せず因子normを含む一般の積摂動評価が必要である。単にτへ補正係数を掛けるだけで全前提を補えない。[RTE SOURCE][RTE]、[原論文Appendix A.2][PAPER]

**現時点の分類**：明確な実装バグ・理論破綻を特定していない。非外向きbound、expm／polynomial／積の丸め、rounded inputと理論前提の差は確認すべき候補。どれがS1/F1の超過を支配したかは未解決。旧atolをboundへ足してPASSとしない。

## 6. 最小追加数値検証案A0／必要時A1

以下は未承認の新規検証で、旧runのretry／resumeではない。

| 条件 | 提案 |
|---|---|
| 対象 | S1、F1、正常対照B3、丸め水準対照R4のT=±0.18、計8既存行。他の条件は旧22行契約のまま |
| SOURCE／入力 | 26e8734afと33715848bの保存入力を固定参照。新reference実装SOURCEを別commitで固定。入力NPZ・graph・結果はread-only |
| 精度・独立性 | mpmath 90桁Taylor expm、128桁Padé expm。既存`precision_context`を再利用し、候補DAG evaluatorを呼ばずcaseと凍結係数から数学順を独立構築 |
| 演算 | 2×2指数・D/outer積、K+1次polynomial・r乗、scalar phase、N、5対象signal、operator 2-norm、同じτの独立positive Taylor remainder／合成bound |
| Scope | 8行×2精度=16 case-precision評価、各5対象、計80 target／80 signal評価要求。共通参照はcache可。入力constructor・DF再分解・回路build・shotsは0 |
| 追加診断 | 行別にcomponent入力差、signed τ、D／outer積、finite polynomial、scalar、N、norm／差のどこからdoubleとの差が現れるかを記録。旧定数を使うreplayと、primitiveから高精度で和・積を行う参照を区別 |
| 安定性基準案 | 90/128桁のsignal・残差norm・boundの絶対差≤1e-40をempirical安定の目安。これはu_numの保証値ではなく、未達はINCONCLUSIVE、追加精度へ自動拡張しない |
| Boundの判定 | 高精度残差が高精度bound以下で安定すれば「empirical整合」。依然超えるなら入力norm／Hermiticity／target定義を調べ「未解決」、実装／理論FAILの断定を留保。区間が比較不能ならINCONCLUSIVE |
| 出力 | 旧b64と各signalの差、90/128間差、stage／全体bound、入力前提を別保存。u_numはcertified結果が得られるまでnull |

A0で浮動の局所化ができても、native候補全てのu_numを評価したことにはならない。Native候補集合の他の行へ保証を移送しない。参照用expm unique key・primitive積・SVD等の詳細call数は**実装後metadata展開で固定してから実行許可対象にする**。旧1239 dispatch等を新検証へ流用しない。旧値を読むだけの比較と新しい高精度評価を区別してscopeに記録する。

A1を必要とする場合は、まずA0の8行だけを対象に256／384 bitのcomplex ballで同じtargetを包含し、signal差・abs区間を返す案とする。旧snapshotをexact liftしたこと、全演算がballから通常floatへ漏れていないこと、必要なnorm／product前提が成立することを確認する。Spectral residualの包含まで必要なら2×2の`E*E`の最大固有値包含等を限定利用し、Frobenius上界をspectralの等値として使わない。包含が広い場合はSTOPし、後から精度や対象を自動追加しない。

A0/A1ともfresh専用namespace（案 `artifacts/research_b_outer_numeric_check/<date>/<fresh-id>/`）に新結果・SOURCE・入力identity・精度・失敗・STOPを保存する。今回は作成していない。旧22行／16行／36行の原本とSTOPは不変。数値検証の許可はnative／shot実行の許可を兼ねない。

## 7. Native・費用コードの再利用と具体的な不適合

| 既存資産 | 再利用できるもの | そのまま使えないもの／最小接続 |
|---|---|---|
| [outer.py][OUTER] | `OuterPlan`／`DStage`／`RTEStage`、signed時間、case r/K、同tau・同familyのD反復融合 | 現状は記述子とdense graph。任意ordered列のnative rendererが必要。異tau近似Dを時間和で一つの近似へ置換しない |
| [df_partial_s2.py][DF]、[df_trotter/ops.py][OPS] | DF preparation／basis registry、`one_body_diagonal_primitives`／`df_squared_diagonal_primitives`／`append_diagonal_primitives`、controlled diagonal・relative phase | `DFPartialS2StepRequest.pf_label`は2ndのみ。A2/A3を一個のpartial-S2 requestに偽装しない。block rendererの小さな共通化／再利用が必要 |
| [df_rte_qiskit.py][DFRTE]／`df_rte_circuit.py` | `QiskitDFRTEEventCircuitBuilder.build_sequence`、signed event、Z/ZZ、basis共有、ancilla-relative identity／event phase | 各R occurrenceへ固有configとr個eventを渡す。旧一律configは使わず、負時間をinverse列へ置換しない |
| [df_partial_s2_repeated_cost.py][REPCOST] | exact／MC streamの確率weight、task seed、streaming／全trajectory費用・workload管理のパターン | streamは単一step_time・configのq反復。一般outerのheterogeneous R列には新しい薄いstream接続が必要。q=16への偽装でF1を表現しない |
| [rpe_hadamard_interrogation.py][HAD] | H→controlled-U→H、YのSdg→H、measurement／bit mapping、同じtrajectoryのX/Y評価 | 型は`DFPartialS2RepeatedCircuitResult`、許可反復は1/2/4、state preparationなし。一般outer結果を受ける小さな共通wrapper部品とmode0準備の挿入が必要 |
| [df_rpe_hadamard_compiled_cost.py][HADCOST] | 両軸を同一trajectoryから生成し個別compile、exact weight一回／MC unweighted、費用統計 | `_compile_hadamard_trajectory_stream_with_plan`はstate preparationが入ると**拒否**する。既存「without_state_preparation」契約は変えず、新full-wrapper用の薄い経路へ接続 |
| [rte_compiled_cost.py][COST] | `plan_compiled_cost_workload`、`TranspiledCircuitCostCache`、`CompiledMetricAccumulator`、整数gate／filtered depth計測 | Plannerは1item当たり同じcircuits数。compiled対象X/Yの2と追加診断buildを分ける。既定cacheはmetricsのみ、persistent hitではcompiled operatorを検査できない |
| [pr_dictionary_validation/gates.py][GATES] | `compiled_wrapper`のofficial-definition展開、actual controlled/operator／Hadamard検査の方法、`shot_count_from_enclosure` | 旧fixture・P0の科学結果／価格／許可は移送しない。scalar ceil helperの資格も再確認が必要 |

重要なcompiler注意点：`rte_compiled_cost.transpile_and_measure_cost`は直接transpileする。一方、既存`compiled_wrapper`にはQiskit1.3のnested equivalence翻訳でphaseを失う問題に対して、`UnrollCustomDefinitions(EquivalenceLibrary(), native+['u'])`で公式definitionを展開する修正がある。[phase回帰test][PHASETEST]も参照した。新経路ではこのphase保持接続を再利用する案とし、直接transpile経路と同じ結果だと仮定しない。共有SessionEquivalenceLibraryを変更せず、global phase fitで不一致を隠さない。

`get_or_transpile`はcompiler callbackを受け取らず既定transpilerに固定されている。Pilotはphase保持compilerと既存metric extractorを使い、実compiled circuitを保持する小さなlocal cacheへ接続する案が簡潔である。将来persistent cacheへ統合するなら、lowering済みactual circuit・compiler／backend／versionとlowering policyをkeyに含める最小拡張に限定する。角度・control・state prep・measurementを省いたkeyや、過去研究の価格cacheを使用しない。

静的に確認した関連testsは、`test_rte_compiled_cost.py`のexact weighting／MC unweighted／cache／controlled identity phase、`test_df_partial_s2_repeated_cost.py`のtrajectory weighting／workload／streaming、`test_df_rpe_hadamard_compiled_cost.py`の共通trajectory・両軸個別compile・measurement、`test_rpe_hadamard_interrogation.py`のY符号・q制限、`test_df_rte_qiskit_builder.py`のfull controlled block・event順・basis共有である。[tests][TESTS] 今回これらは実行していない。数値／回路生成を含むtestを「人工」として無断起動しない。

## 8. 最小native候補集合の提案

下表は既存報告の引用を用いた**設計上の選定**で、今回の新しいrankingや保証付き適格判定ではない。基本集合は6候補×両T=12 case。未評価候補があるため、この集合内でも費用を測る前から最小を認定しない。

| ID | 目的・既存bの引用 | ε=1e-8の比較候補案 | ε=1e-9の比較候補案 |
|---|---|---|---|
| B3：Yoshida4/A2/r2K2 | 短いDの基準、b約6.76e-10、D指数180 | 保留資格付きで含む | 観測bがε/2超、今回の狭い集合では費用基準のみ |
| D1：Yoshida4/A3/r2K2 | D精度の対照、b約4.53e-10、D指数1260 | 含む | 含むが予算余裕とu_numを要確認 |
| R2：Yoshida4/A3/r4K2 | r増加、b約8.34e-11、R short-step24 | 含む | 含む |
| R4：Yoshida4/A3/r2K4 | K増加、b約7.00e-11、R short-step12 | 含む | 含む |
| B4：SABA2/A2/q2/r2K2 | 少ないD/R、b約2.91e-9、D指数90、R4 | 含む | 観測bがε/2超、今回対象外 |
| F1：refined-S2/A2/f8/r2K2 | 強いS2対照、b約3.54e-9、D指数480、R16 | 含む | 観測bがε/2超、今回対象外 |

Native pilotはこの6候補を同じscopeで調べる。ε=1e-9のG比較を後に返す場合は、u_num・γ資格を満たしたD1/R2/R4だけが当面の対象で、適格を今認定しない。未達は無限費用でなく「この登録集合で未適格／未確定」と記録する。

S1のq4はB4よりR/Dが増え、報告b約7.13e-10はε=1e-9の半予算を上回るため、初回native pilotから外す。ただしNは小さく、native全費用でB4等に劣ると証明したわけではない。A2のR1/R3も補助対照として未評価に残す。ε=1e-8では有益な可能性を残すので、**6候補選定を全22行での資源最適性の主張にしない**。

## 9. Full-wrapper／shot費用の定義

全候補で`G=m_X E[C_X]+m_Y E[C_Y]`。Cはmode0 occupation準備、ancilla H、ordinary controlled evolution、X/Y終端、ancilla measurementを含む。現在のmode0準備は対応qubitへのXで実現する案だが、OpenFermion／Qiskitのbit orderを独立確認する。Gaussian basis変更はevolution内の実費である。準備費を除く旧costを定数として足すだけでは、full-wrapper最適化を反映できない。

主指標はshot加重expected RZ count、補助はCX count、RZ／CX depth、total depth、size、measurement count。D kernel／blockとR event分布・次数も併記する。m×平均depthは逐次work proxyであり並列wall timeではない。RZのClifford角分類は丸め近傍を厳密Cliffordとせず「provable Clifford／一般回転／数値未分類」に区別する。Clifford+T合成・T-countを求めるなら別の離散合成法・精度・phase条件が必要である。

固定candidateの全体Nは各stageのB_j^r_jの積で、raw outcomeはO_a∈{−1,+1}、corrected estimatorは`N mean(O_a)`。既存規則を維持し、`s=2ε/5`、`η_X=η_Y=s/√2`、`α_X=α_Y=.025`を単一candidate・Tごとの案とする。

共通の予算は`b+u_num≤ε/2`（50%）、`γ≤ε/10`（10%）、`s≤2ε/5`（40%）。ε=1e-8／1e-9とも同じ規則を適用し、未取得のu_num・γ・sを0として適格化しない。[固定budget metadata][OUTEREXEC]

\[
 m_a\ge\left\lceil\frac{2N^2}{\eta_a^2}\log\frac{2}{\alpha_a}\right\rceil,
 \quad\Pr(|N\bar O_a-N\mu_a|>\eta_a)\le2e^{-m_a\eta_a^2/(2N^2)}.
\]

これは独立・有界な測定に対する[Hoeffdingの標準評価][HOEFFDING]を今回のrangeへ適用した式で、新しいPRの保証ではない。RTE normalizationとHadamard平均の対応は[原論文A23–A29][PAPER]を参照する。**今回はN代入・log／ceil実数値・shot数を計算していない。**

[rpe_resource_accounting.py][SHOTS]の`_hoeffding_shots`はraw座標精度に使うscalar部品として再利用候補で、引数は`η_a/N`。`evaluate_rpe_round_candidate`は半径・phase予算・roundを導入するため、単一complex absolute errorへそのまま使わない。保証用ceilにはNの上界・正確な予算・log／sqrtの外向き評価とceilの一意性確認が必要で、通常floatの`math.ceil`を証明にしない。既存scalar interval helperを限定再利用できるが、P0の固定η／αを今回へ流用しない。

Quantum shotごとに新しい独立RTE trajectoryと測定outcomeを使う。費用MCで同一trajectoryをX/Yに使うことはpairedな古典費用比較であり、量子測定でtrajectoryを多数shot再利用する許可ではない。Varianceは`N² Var(O_a)/m_a`だがNだけから実variance・shot数を確定しない。Variance適応法なら独立pilot・empirical Bernstein等のconfidence定義を別途固定する。初回はbounded worst-caseとする。

γにはfinite-bit probability law、normalization近似、native角度・相対phase・合成・state準備の実装平均のずれを含む。Raw側誤差はNで増幅される。Ideal native RZ／理想samplerを仮定した参考費用と、保証付き実装のGを区別し、γを未評価のまま0にしない。複数candidate・両T・精度を同時に主張するfamilywise budgetは未配分。Cost MCの誤差確率も量子推定のαとは別である。

## 10. Native実行契約案B0：まず少数trajectoryのpilot

**未承認・実装前の案**。上記6候補×T±.18、n_D1、既存fixture／sector／mode0／LD2・signed stageを固定する。Inputは同じrecipeのDF preparation／full-Fock基底を再構成してsnapshot identityへ照合する。2×2 sectorの一致だけで、full controlled phaseを検証したとしない。

Compiler案は既存phase保持profileに合わせる：Qiskit1.3.0、basis `rz/sx/x/cx`、optimization level1、seed17、backend／target／coupling mapなし、layout／routing指定なし、approximation_degree1.0、qubits_initially_zero=False。Official-definition展開を固定し、測定は1bit、初期resetとnoiseは含めない。真の装置時間・fault-tolerant synthesis費用を返すprofileではない。Lowering後のglobal phaseも保持し、共有環境を変更しない。

各caseで次の4つの決定論的代表trajectoryを固定する案：①全order0で最初のcomponent、②全order0でcomponentを循環、③最初のR short-stepだけ最大保持次数Kでcomponentを循環、④Tと逆符号のR stage（複数ならappend順で最初、なければ最終R stage）の最終short-stepだけorder2、他はorder0。各eventのproduct／rotation順と成分signを固定する。③でK4、④でorder2のphaseとYoshida4中央の符号反転を覆う。T<0では中央stageが正時間になるため、単なる「最負stage」の選択では二種類のdurationを覆えない。これらは**確率標本ではない**。

| 高位境界 | 提案するlogical scope／上限 |
|---|---:|
| Candidate×T | 12 |
| 代表trajectory | 48 |
| System U renderer | 48 build要求、4×4 full-Fock |
| Controlled-U renderer | 48 build要求、8×8、control-zero=I |
| 準備・測定込みX/Y wrapper | 96 build要求 |
| Wrapper compile | 96 cache要求、実transpileはcache miss数以下 |
| 独立trajectory reference | full system48、controlled組立48、X/Y wrapper96（入れ子処理） |
| Operator正当性比較 | system48＋controlled48＋未compile wrapper96＋compiled wrapper96、計288要求 |
| X/Y期待値確認 | 未compile／compiled各96、計192要求。Quantum shot simulationは0 |
| 5対象／mean関係 | 12 caseのfull-space analytic mean確認。旧sector結果との対応も検査 |

上表は重なる親子境界であり、合計を独立科学call総数としない。Measurementを外したvalidation copy、QiskitOperator化、基底順permutation、state準備の確認は対応gate内に登録する。Compiled circuit本体を保持し、metrics-only cache hitからoperator PASSを推定しない。

Event meanの関係には、各caseで最初のR stageとTと逆符号のR stage（複数ならappend順で最初、なければ最終R stage）の**single short-step**を独立weighted enumerationで照合する案を加える。24 logical要求で、今回の3成分・K2/K4ならevent matrix項は最大`20×30+4×273=1692`。Qiskit回路を全eventでcompileする案ではない。両TでYoshida4の2種類のR duration、SABA／refinedの同一durationを覆い、同じlawは共有する。R stage全体のr乗、ordered mean、Nの積への接続を別途確認し、選んだ48 trajectoryの平均を真のmeanとしない。

正当性gate案は、absolute operator／complex signal誤差≤1e-12、rtol0、global phase fitなし。Signed probability law、D内部A2/A3順、scalar／event phase、control-zero、full-Fockとsectorの対応、準備とY符号を先に検査し、その後だけ費用表を返す。この閾値は**empirical correctnessでγ certificateではない**。他研究の5e-10 gateを新予算へ無条件移植しない。

`plan_compiled_cost_workload`では48 item×2費用wrapper=96のcompile計画を作り、system／controlled診断buildを別枠96として追加する。Stage別primitive／RTE config／expm／event descriptor／Operator／検証copyの詳細数・instruction上限・出力枠は、renderer実装後にmetadataだけで展開して固定する。**今すぐ実行可能なlauncher契約が完成したとはしない。** Source固定・人工test・科学importを止めたpreflight後に、B0の明示実行許可が必要である。

## 11. Exact／MC expected costと後続scope

構造上、1 short-stepのevent空間は`E_K(M)=Σ_(n=0,2,...,K) M^(n+1)`。既存報告のretained3成分ならE2=30、E4=273。全trajectoryは`∏_j E_(K_j)(M)^r_j`で、B3/D1は30^12、R2は30^24、R4は273^12、B4は30^8、F1は30^32。これらは組合せmetadataで、law確率／費用の数値評価ではない。実行前にゼロ確率supportを確認する必要はあるが、初回からfull exact enumerationを選ぶ根拠はない。[rte.py `iter_rte_events`][RTE]

Full-wrapper費用はevent価格の和では確定しない。D/R境界、同basis共有、合法なD融合、scalar/control、compilerの最適化を含む**full trajectoryをcompile**し、その期待値を取る。Exactならjoint probabilityを一回だけweightし、MCならfresh IID joint lawのsampleをunweightedに平均する。K4の稀な高次数を決定論的pilotで覆っても、IID cost meanの精度が確認できたわけではない。

B0後に別途採択するB1案として、まず各case固定256 IID trajectory、12 caseで3072 trajectory／6144 X/Y compile要求の探索的cost MCを提案する。これはQuantum shot数ではない。既存`CompiledMetricAccumulator(weighted=False)`で平均・unbiased variance・SE・min/maxを保存し、両軸は同じtrajectoryを使う。相対SE2%を診断目標、正規近似95%区間は診断と表示し、厳密なcoverageと呼ばない。K4稀eventの未観測も記録する。未達・順位区間重複ならSTOPし、自動増標本・候補追加をしない。標本数はB0の実測compile負担・必要精度を踏まえて正式契約時に変更可能な**未採択の初期案**である。

Task seedはcandidate／T／sample／occurrence／short-stepに結び、並列度で変えない。独立case／trajectoryのcompileを利用可能CPU/RAMで並列化し、immutable準備情報・実角度のcacheを共有する。BLAS／Qiskit等とのoversubscriptionを避け、worker数を恣意的に小さく固定しない。古典compile時間と量子費用は別記する。Phase保持済み回路のcache、streaming統計、合成済み部品を既存資産から再利用し、意味の異なるangle-free cacheは使わない。

u_num・γ・confidenceが資格を満たしたcandidateについてのみ、将来別許可のscalar shot評価でm_X/m_YとGを出す。Cost meanの不確かさをGへ伝播し、区間が重なる順位を保留する。未資格の候補に保証付きmやGを返さず、参考モデルを出す場合は仮定を明示する。

## 12. 保全・完了状態・次の担当

今回変更するのは本Markdownだけ。独立branch `codex/research-b-outer-uncertainty-native-design-20261025`、起点33715848b。科学SOURCE／launcher／環境／raw／STOP／既存契約は変更しない。新規科学結果や採択済み方針を追加した段階ではないため、synthesisやevidence manifestの科学statusを更新しない。科学moduleは起動せず、SOURCE・tests・資料の読取りとdependency metadata確認、文書検査だけを行った。

A0、A1、B0、B1、shot/G評価はそれぞれ対象・精度・SOURCE・科学call／出力scopeを確定した新しい実行単位とする。Fresh namespace、失敗／部分結果保存、retry／resume禁止、mandatory STOPを維持し、wall-time／CPU／GPU／RAMの人工的上限や承認証明frameworkを加えない。技術準備・mock確認は実行許可を代用しない。現在、これらのoutput directoryは作成していない。

次は研究Bのユーザーが、まずA0の限定高精度確認を行うか、B0のrenderer／full-wrapper接続の実装準備を採択するかを判断する。Codexは採択範囲を実装・人工確認してSOURCEと詳細scopeを具体化し、**高精度科学実行とnative／shot科学実行は各々の明示許可後だけ**開始する。今回の公開・remote照合後はmandatory STOP。重要な方針変更の判断を代行せず、通常作業について研究統括へ報告しない。

[RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/docs/research/research_b_outer_rte_22row_execution_20261025.md
[CSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/artifacts/research_b_outer_rte22/20261025/COMPARISON.csv
[CONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/docs/research/research_b_outer_rte_22row_execution_ready_20261025.md
[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/docs/research/research_b_outer_rte_same_accuracy_cost_design_20261025.md
[PREVIOUS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/docs/research/research_b_outer_splitting_stage1_execution_20261025.md
[INPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trottertracks/b2d_evaluator/input_package.py#L110
[DENSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trottertracks/b2d_evaluator/dense_backend.py#L50
[OUTER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trottertracks/b2d_evaluator/outer.py
[OUTERBACK]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trottertracks/b2d_evaluator/outer_backend.py#L61
[OUTEREXEC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trottertracks/b2d_evaluator/outer_rte_execution.py#L40
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trotterlib/rte.py#L1127
[HP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trottertracks/pr_dictionary_validation/reference.py
[INTERVAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trottertracks/pr_dictionary_validation/intervals.py
[DF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trotterlib/df_partial_s2.py#L426
[OPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trotterlib/df_trotter/ops.py#L226
[DFRTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trotterlib/df_rte_qiskit.py#L150
[REPCOST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trotterlib/df_partial_s2_repeated_cost.py
[HAD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trotterlib/rpe_hadamard_interrogation.py#L120
[HADCOST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trotterlib/df_rpe_hadamard_compiled_cost.py
[COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trotterlib/rte_compiled_cost.py#L107
[GATES]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trottertracks/pr_dictionary_validation/gates.py#L85
[PHASETEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/tests/management/test_pr_dictionary_compile_phase.py
[TESTS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/33715848bfb83d49ec4e54d6eef481012a7441e4/tests
[SHOTS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/src/trotterlib/rpe_resource_accounting.py#L553
[SCIPY]: https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.expm.html
[EXPM_PAPER]: https://eprints.maths.manchester.ac.uk/1300/
[MP_CONTEXT]: https://mpmath.org/doc/current/contexts.html
[MP_MATRIX]: https://github.com/mpmath/mpmath/blob/1.3.0/mpmath/matrices/matrices.py
[FLINT]: https://python-flint.readthedocs.io/en/latest/index.html
[ACB]: https://python-flint.readthedocs.io/en/latest/acb_mat.html#flint.acb_mat.exp
[PAPER]: https://arxiv.org/pdf/2503.05647v2
[HOEFFDING]: https://doi.org/10.1080/01621459.1963.10500830
