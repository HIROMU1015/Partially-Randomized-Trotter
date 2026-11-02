# DF-RTE積部分の生成・圧縮：限定方法差・source・期待費用監査 v1

- 日付：2026-10-13 JST。
- 区分：採択済みGPT方針の下で行うCodex技術監査。新しい科学レビュー・新アルゴリズム採択ではない。
- branch：`track-b-df-rte-product-generation-audit-20261013`。
- base：DF補正closure公開commit `4192077723770f4846cc391ea0da96c1e115f2fa`。
- 入力原本：[GPT方針レビュー](../../research/track_b_post_closure_research_direction_review_20261013.md)。41,995 bytes、SHA256 `caeef1dc9d4f74b7e6b7e7ec3671c3039f42ddbb2232d95dc10f8ea019e21289`。
- 原本公開commit：`c085ccd9226b6342751aaeee305d18997f80ae56`。remote再取得で原本bytes一致を確認。
- source監査対象：`321cd8c3066194fe98741cbebbcdba1515d1deaa`。Track Aのsourceは読取りのみ。
- 成果物：[identity inventory](../../../artifacts/track_b_df_rte_product_generation_audit/2026-10-13/source_identity_inventory_v1.json)、[saved費用の確認範囲](../../../artifacts/track_b_df_rte_product_generation_audit/2026-10-13/saved_cost_scope_v1.json)、[証拠manifest](../../../artifacts/track_b_df_rte_product_generation_audit/2026-10-13/evidence_manifest_v1.json)。

## 1. 到達点と停止位置

**指定された理想構成は、number-conserving Gaussian primitiveという前提の下で成立する。現在記述された追加手順は既知の反射積・partial Gaussian合成の組合せに還元され、固有の方法差は未定義である。** 現builderには異基底wordをまとめる入口を確認できなかったが、repository内に専用実装がないことは独立新規性の証拠ではない。

技術分類は `KNOWN_COMPONENT_PIPELINE_NO_DISTINCT_STEP_DEFINED`。Gate 1の新手法pilot進行条件は充足していない。Gate 2は `UNRESOLVED_CONDITIONAL_COST_NOT_ZERO`：公開保存値だけでは変更可能部分の母平均費用上限を取得できず、小さい・大きいのどちらも認定しない。

**Codexで行える静的監査・式の対応・保存値bookkeeping・公開までを完了し、`STOP_FOR_GPT_SCOPE_DECISION` とする。** これは候補やTrack B全体の科学的終了をCodexが採択したラベルではない。次に必要なのは、独立した追加手順があるか、既知法のapplicationとして別途進めるか、今回の候補を区切るかというGPT側の研究判断である。

Gate 3は条件付きなので、現時点で新fixture・runner・native比較・T合成契約を固定しない。Gate 4、本番分子pilot、H6再開、G10/RA-D0/whole補正closure解除も行わない。**新しい行列・回路・synthesis・sampling・LP・NPZ・GPU処理0、authorization/marker作成0、mandatory STOP。**

## 2. 監査範囲

読み取ったsourceはidentity inventoryの10 Python file。Git objectからtextを読み、ASTと関数・field・callの対応を確認した。moduleのimport、科学runner、既存scientific testsは実行していない。

保存値はH6公開summaryのJSON3件とwrapper CSV1件だけを確認した。orbitals、Hamiltonian行列、NPZ、private runtime、実際のtrajectory event列を生成・復元していない。

一次文献は関連する生成手順・制御・partial basis・reflectorの箇所を確認した。先行研究全体の網羅的な不在証明ではない。数学監査の導出と、実装・数値認証・コンパイル実証を区別する。

## 3. 理想構成の独立照合

### M1：parity primitiveと一粒子反射

真空で1となる第二量子化を `Γ(M)=⊕η ∧ηM` とする。正規直交列 `V=(v1,…,vr)`、`r∈{1,2}` に対し、`R=I−2VV†` は選択軌道で−1、直交補空間で＋1を作用させる。占有数ごとの積から

\[
\Gamma(R)=\prod_{\ell=1}^r(1-2n(v_\ell)),\qquad
Q=\sigma\Gamma(R)
\]

となる。primitiveのscalar sign σはone-particle Rに吸収しない。`Γ(−R)` はparticle numberごとに異なる因子を生み、scalar `−Γ(R)` の代用にならない。

これは**指定されたGaussian basisのZ/ZZ primitive**の式であり、任意DF blockや任意registry operationに対する主張ではない。

### M2：時間順・event phase・終端回転

Q1からQkへ時間順に作用すると、matrix productは

\[
P=Q_k\cdots Q_1=\chi\Gamma(R_k\cdots R_1),\qquad
\chi=\prod_j\sigma_j.
\]

現 `RTEEvent` はproduct数＝even Taylor order k、`selected_component_ids` はproductsの後にrotationを置く。`_make_event()`はsampling側で最初に選んだrotation labelを最後のapplicationへ配置する。label抽選順と回路作用順を混同しない。sourceのevent phaseは `(-1)^(k/2)`。

`unsigned_rotation_angle` は名称に反してrotationのcoefficient signを掛けた実装角度である。既存terminal rotationを維持するとき、そのsignをもう一度掛けない。

sourceの時間順に対応する全eventは

\[
U_e=(-1)^{k/2}\,\chi\,
 e^{-i\theta Q_{\rm terminal}}\Gamma(M),
\]

という形で追跡できる。identity productもsignに含め、identity terminalは既存のscalar phase処理を残す。terminal rank1であっても今回はまとめない。

### M3：active spanと小行列

全selected columnのspanをS、直交basisをWとする。各RはSを保ちS⊥でIなので、積Mも同じ性質を持つ。

\[
M=I+W(M_S-I_d)W^\dagger,\quad
d\le\min(N,\sum_j r_j)\le\min(N,2k).
\]

初期案の範囲はevent内のproduct roleだけ。K2なら `d≤min(N,4)`、K4なら `d≤min(N,8)`。一般語長に対して最大6軌道とはしない。ここでのdはexact span rankであり、浮動小数QR thresholdで捨てるrankではない。

### M4：small unitaryの対角化とcontrolled実装

unitaryな `M_S=Z diag(e^{iφ}) Z†` とcompletion `B=W̃(Z⊕I)` に対し

\[
\Gamma(M)=\Gamma(B)D_\phi\Gamma(B)^\dagger,
\qquad D_\phi=\prod_{\ell=1}^{d} e^{i\phi_\ell n_\ell}.
\]

時系列は `Γ(B)† → Dφ → Γ(B)`。controlled targetはblock-wiseに

\[
(I_c\otimes\Gamma(B))\,
 \operatorname{diag}(I,D_\phi)\,
(I_c\otimes\Gamma(B)^\dagger)
=\operatorname{diag}(I,\Gamma(M)).
\]

中央のoccupation phaseはnative `CP(φ)`に対応する。scalar χとevent phaseはcontrolのrelative phaseに戻す。`e^{iφ n}=PhaseGate(φ)=e^{iφ/2}RZ(φ)`なので、RZだけへ置換してphaseを落とすとcontrolled targetが変わる。

同じbasis circuitとその厳密なinverseを両側に用いれば、そのbasisのscalar global phaseはconjugationで相殺する。これと、中央作用・eventのscalar phaseを捨てることは別である。

### M5：rank2終端回転は一般に非Gaussian

二つのmodeで `Q=(1−2n1)(1−2n2)` とし、`exp(−iθQ)` の真空位相を除くと、single occupationsは各 `e^{2iθ}`、double occupationは1になる。number-conserving Gaussianのdiagonal actionならdouble occupationはsingle二つの積 `e^{4iθ}` でなければならない。従ってgeneric θではGaussianでない。特別角度の可能性を一般rank2へ拡張しない。

これは式による診断であり、新しい2-mode行列生成・数値計算を行った結果ではない。

### M6：誤差と推定量の不変性の条件

各eventのideal作用、p、weightが一致すれば有限meanは一致し、Taylor target・normalization・理想moment/rangeは変わらない。実装誤差は別である。

\[
\|\sum_e p_e w_e(\widehat U_e-U_e)\|
\le\sum_e p_e|w_e|\,\|\widehat U_e-U_e\|.
\]

exact unitaryなMとM′に限り、tensor積のtelescopingとantisymmetric subspaceへの制限から

\[
\|\wedge^\eta M-\wedge^\eta M'\|\le\eta\|M-M'\|
\]

を使える。非unitary float arrayへ直接適用しない。full Fockの保守係数はN。最終Clifford+T近似gateの回路が正確なGaussianだと仮定して、このone-particle boundだけで物理gate errorを覆わない。実際のgate synthesisはfull circuit operator errorを合成して別枠で戻す。

同じshot数を保てるのは、同じ最終bias予算・統計幅が残った場合だけ。今回、数値的なrank/phase/angle certificateや同じaccuracyのnative実装を作ったとは認定しない。

## 4. 現sourceとの対応とaccessの不足

以下のlineは固定commit `321cd8c3066194fe98741cbebbcdba1515d1deaa` のもの。

| 機能 | source / line | 確認した実装・限界 |
|---|---|---|
| role・時間順 | `rte.py` 549–643、1594–1659 | even k products→一つのrotation、Taylor phase、product signとrotation signを分離 |
| sequence builder | `df_rte_qiskit.py` 229–391 | requestをflattenし、基底ID/hashが同じときreuse。rotationを跨いでも同じbasisなら既存reuseを保持 |
| central product/control | 同473–502 | ZまたはZZ、controlledではCZ、negative productはscalar phase |
| basis/control/sign | 同281–362、448–458 | basisはuncontrolled、control相対phaseとuncontrolled global phaseを別に保存 |
| typed basis plan | `df_rte_circuit.py` / builder 423–445 | support限定basisのcolumn-preservation条件を検査。今回cross-basis productの新hookを確認しなかった |
| Gaussian converter | `df_trotter/ops.py` 194–225 | `BogoliubovTransform(U.T)`の汎用変換が既に存在。依存欠如時のdense Fock fallbackは今回呼ばない |
| support completion | `research_direction_structure_pilot.py` 59–119 | Z/ZZ supportの1/2列を保ち、N×N completionを生成。r列だけを直接QRする専用thin compilerと同一ではない |
| P-A union/run DP | `research_direction_joint_synthesis_pilot.py` 67–116、204–352 | 同source-basis run内のsupport union、有限proxy候補内のpartition |
| P-A formal scope | `research_direction_joint_synthesis_formalization.py` 255–310付近 | arbitrary completion・relative basis synthesis・noncontiguous groupingは旧最適性の範囲外 |
| orbital access | `df_rte_tail.py` 48–155、349–381 | registryはoperation metadata/runtime gatesを保存。N×r orbital Vはtyped fieldにない |
| `basis_unitary` | 同363–381、480–496 | guarded **many-body** reference。名前だけでN×N orbital matrixとして呼んではいけない |
| H6費用保存 | `h6_matched_cost_port_v3.py` 18–110 | rawにはtrajectory/events/digest、axis-paired wrapper costを保存するが、公開summaryにはraw本体がない |

registryは任意operation listを登録できる。全registered basisがnumber-conserving Gaussianであることを型だけで保証していない。候補の有効範囲には、実際のruntime basisと `Γ(U)`、support column、signのsource-bound対応が必要。

従って既存の `DFRTEEventCircuitRequest` を受け取るだけでは、提案が前提とする低rank orbital列へ直接到達できない。将来実装するなら、B固有sidecarまたは小さい独立adapterの入力規約・identity・precisionを固定する必要がある。共通APIを今回変更せず、DF modelを再対角化して未承認のorbitalsを取得することも行わない。

検索は固定sourceの `src/`・`scripts/`・`tests/` と上記call chainに限定した。keyword未検出だけでrepository全体・他branch・外部libraryに実装がないとは断定しない。

## 5. 既知構成への具体的な還元

### 5.1 rank2も通常のrank1反射列へ分解できる

primitive内のorthonormal columnsに対して

\[
I-2VV^\dagger
=\prod_{\ell=1}^{r}(I-2v_\ell v_\ell^\dagger).
\]

同一primitive内ではprojectorsが互いに直交するので、この分解はexact。同じgroupの列を展開しても、異なるprimitive groupの元の時間順は保つ。

### 5.2 arbitrary dense columnsに対するcompact WYの再帰

時間順v1,…,vs、`H_j=I−τ_j v_jv_j†`、`τ_j=2`とする。

\[
M_s=H_s\cdots H_1=I-X_sT_sX_s^\dagger,
\quad X_s=[v_1,\ldots,v_s],
\]

と書くとleft multiplicationの直接展開から

\[
T_s=
\begin{pmatrix}
T_{s-1}&0\\
-\tau_s v_s^\dagger X_{s-1}T_{s-1}&\tau_s
\end{pmatrix}
\]

となる。v間が従属でもこの表現は成立し、rank認証に頼って列を削除する必要がない。`X=WC`へQRすると `M_S=I−CTC†` であり、レビューのactive-span reflection積と同じMを生成する。

これはcompact WY型の既知reflector-product表現への数学的還元である。[Schreiber/Van Loan](https://epubs.siam.org/doi/10.1137/0910005)、[LAPACK ZLARFT](https://www.netlib.org/lapack/explore-html/d7/d0d/group__larft_ga99d53d027cc196a5e5f7e2bbacdf4979.html)を対照とする。

**ZLARFT自体をarbitrary dense DF columnsにそのまま呼べるとはしない。** 公式routineはDIRECT、STOREVとtrapezoidal storage/implicit diagonalの規約を持つ。上の一般再帰との関係、列の正規化、tau、順序をadapterで照合する必要がある。実routineを今回呼んでいない。

### 5.3 partial QR・Gaussian合成への還元

Wを座標部分空間へ運ぶpartial Givens QR、d×d unitaryの通常分解、occupation phase、逆変換は既知の構成部品である。Mottaのversion固定PDFはpartial basisのcount

\[
g(N,d)=Nd-d(d+1)/2
\]

をsupplement pp.7–8「Implementing basis rotations」で扱う。今回の使用は選択spanに対するpartial transformationという手順の対応であり、MottaのHamiltonian truncation/分子性能を候補へ移したものではない。[Motta et al., arXiv:1808.02625v2](https://arxiv.org/pdf/1808.02625v2)

Campsのfusion/turnover、Kökcüのcontrolled free-fermion/Q/P blockは、一般Gaussian subcircuitを既に圧縮できる対照である。「Hadamard controlが既知法にない」という差は作れない。ただし異なるcompilerの個々の出力gate・T数まで同じと証明したものではない。[Camps et al., v2](https://arxiv.org/pdf/2108.03283v2)、[Kökcü et al., v2 Sec.IV.2, Appendix C](https://arxiv.org/html/2303.09538v2)

JiangのSlater determinant準備には特定状態のoccupied-row gauge freedomがある。これを任意入力full-unitaryの不要gateとして使わない。今回のphase規約はvacuumを含む作用とcontrol targetで照合する。[Jiang et al., v3 Sec.III, Appendix A](https://arxiv.org/html/1711.05395v3)

### 5.4 claim単位の対応表

| 提案の処理・claim | 既知の具体処理 | repository上の対応 | 残る追加手順・判定 |
|---|---|---|---|
| parity積のGaussian閉包 | exterior-power representation / number-conserving Gaussian | Z/ZZ基底conjugationは既存 | 既知の出発点、独立claimではない |
| DF lowrank反射の積 | rank1展開＋compact WY/active-span積 | word全体の専用pathは確認できない | implementation gap。現仕様に固有の新recurrenceなし |
| N×N/Fock denseを避ける | partial QR＋small matrix、F3Cのlocal compression | 小規模dense fallbackは既存、cost pathはGaussian converter | 全Fock回避だけで独立差にはならない |
| 共有spanで一回basis開閉 | partial rotation＋small unitary diagonalization | 現reuseは同ID/hash・P-Aは同基底run | baselineへ同thin accessを戻す必要。現候補は既知部品pipeline |
| controlを中央へ限定 | conjugationのcontrolled diagonal、controlled Gaussian compression | 現builderもbasis無制御＋central control | control対応自体のdeltaなし |
| 相対signを保持 | scalar phase ledger | 現builderで実装済み | 正当性の必須条件、無料の資源削減ではない |
| finite law/weightを保つ | deterministic event-wise implementation substitution | 現event metadataを保存可能 | sampling再設計・normalization改善は今回含まない |
| cold classical costを削減 | known WY/partial QRとの演算・memory比較 | dedicated input sidecarとcache binding未実装 | 改善があり得るが、実測・bit complexity・certificate未取得 |
| 全PR expected resourceを削減 | law-weighted full-wrapper費用比較 | conditional rare費用未公開 | Gate 2未確定、数値勝利・不利益とも言えない |

## 6. 同じthin accessを与えた資源モデル

元primitiveのrankをrj、全列数s＝Σrjとすると、each-primitive partial basisの粗いrotation modelは `2Σg(N,rj)` にcentral Z/CZを加える。active-span構成の粗いmodelは `2g(N,d)+O(d²)` とcentral phases。両方に同じboundary fusion・coupling・数値precisionを戻す。

`d≈s≈2k`ならどちらもO(Nk)。thinを与えないO(kN²) baselineに対する改善を、強い対照を超える漸近改善としない。s≫dの繰り返しspanには集約余地があるが、実際のRTE lawでその頻度・costが十分かは未取得である。

existing support completionはfull N×Nを作る実装である。理論上のthin O(Nr)を実装済み・本番認証済みと表現しない。一方、それを候補だけに与えてbaselineをfull basisへ固定することもしない。

central Z/CZは既存ではCliffordだが、集約後のφは一般角度となり得る。basis開閉回数が減ってもRZ/T合成が増える可能性がある。RZ数をactual Tへ換算した新性能は今回取得していない。

古典処理の `O(Nsd+sd²+d³)` 等は演算モデルに限る。密なcompletionを実際に生成すればmemory O(N²)が残り、rank・QR・phase認証とbit complexityも別に必要。実装されていないimplicit basisを無料として計上しない。

## 7. Gate 2：保存lawで確認できたこと・不足

### 7.1 対象と算術の範囲

参照はknown/development/post-hocのTrack A H6：linear1.00 Å、STO-3G、12 modes、Nα=Nβ=3、sector400、tol-only1e−8 DF rank19、generation order、weighted Hermitian projection、T0.8、selected prefix5、q1/2/4/8、r4/4/4/2、K2/4、独立32 trajectory/cell。Bの新held-out・独立science replicationとは呼ばない。

`1−saved P(all order0)` と `RΣk p_k` だけをbinary64の記述的算術で集計した。新しいRTE distributionを評価・samplingしていない。pchangeは変更可能trajectoryの**上包**にすぎず、identity/redundancyとcompiler事情からactual beneficial probabilityは分からない。

| q/r | K | 少なくとも一つnonzero orderの確率（約） | expected product slots/trajectory（約） |
|---|---:|---:|---:|
| 1/4 | 2 | 0.063793 | 0.130757 |
| 1/4 | 4 | 0.063965 | 0.131485 |
| 2/4 | 2 | 0.032973 | 0.066918 |
| 2/4 | 4 | 0.032996 | 0.067012 |
| 4/4 | 2 | 0.016697 | 0.033658 |
| 4/4 | 4 | 0.016700 | 0.033670 |
| 8/2 | 2 | 0.016697 | 0.033658 |
| 8/2 | 4 | 0.016700 | 0.033670 |

### 7.2 改善上限の式は成立するが、今回数値化できない

law/weight/誤差budget/shot規則を同じに保ち、Ω外で同じfull wrapperなら

\[
\Delta G_x=\sum_a n_aE[C^0_{a,x}-C^1_{a,x}]
\le\sum_a n_aE[1_{\Omega}C^0_{a,x}].
\]

既知のpΩだけでは右辺を計算できない。必要なのはconditional whole-wrapper費用、または全supportに対する証明済みCmaxである。5,000,000 instruction等の実行STOP capは、そのまま母集団のCmaxにはならない。同じcompilerの局所費用を加算してpost-fusion wrapper費用の上限とすることも、証明なしにはできない。

公開wrapper CSVにはcell/phase/replica/seed/axis/event digest/各費用があるが、actual event labels/order列・span dはない。integrated group registryにはrawのpath/hashがあり、raw本体ではない。**hashだけからorder条件と費用をjoinしない。** 3/8 cellの未観測nonzero order、全K4の未観測order4はcost 0ではない。

不足は、(i) orbital sidecarとruntimeの同一性、(ii) saved event列とwrapper費用の対応、(iii) rare/unobserved conditional cost、(iv)研究上のmateriality/適用範囲。公開summaryだけから「改善上限が5%未満」等のgateを作らない。5%などの閾値は今回レビューで未固定。

Track Aのrare-cost作業をBのrunとして自動開始・再採点しない。既存rawの必要最小限公開で埋まる情報と、新compileが必要な未観測条件を区別する。実際の公開・取得の追加範囲は次の判断後に固定する。

## 8. ここまでの技術検証と未検証

| 項目 | 今回の実施 | 未検証事項 |
|---|---|---|
| 式(1)–(4a)、sign・順序 | 独立した式の導出とsource対応 | 新candidate circuitの数値operator残差 |
| WY対応 | 一般dense列へのleft-product再帰を導出 | LAPACK adapter・rank certificate・実runtime性能 |
| controlled phase | block-wise algebra、PhaseGate/RZ規約を照合 | compiled strict full-unitary error |
| thin fairness | Motta partial QRとexisting helperを区別 | 公平なnative/T実測 |
| Gate 2 | 8保存lawの記述的算術、公開field確認 | conditional population cost上限 |
| metadata監査 | 原本bytes、Git identity、AST位置、saved counts・schemaを検査 | 数学証明やscience再現を自動testで認証したものではない |

metadata verifierはこの監査folderのstdlib専用scriptであり、scientific moduleをimportしない。実行結果・件数は [metadata validation](../../../artifacts/track_b_df_rte_product_generation_audit/2026-10-13/metadata_validation_v1.json) を正本とする。既存のfocused PASS件数を今回実行した件数として流用しない。

## 9. 次の判断・実行承認の境界

1. **Codexで完結した事項**：式・source・既知処理の対応、access gap、保存field不足、利益上限の適用条件、provenanceと公開。
2. **GPTの判断が必要な事項**：現在の既知部品pipelineとは別に固有手順を定義できるか、独立新規性を主線とするか、application/engineering studyへ目的を限定するか、候補を区切るか。
3. **後続の別実行承認が必要な事項**：新しいmatrix/control fixture、B adapter/runner、native compile、T synthesis、登録sampling、H6 conditional cost・分子validation。現時点でこれらの実行契約・authorizationは作成していない。

将来GPTが実装comparisonを採択した場合の技術準備項目は、同基底反復／異基底独立span／異基底共有span／空word・negative scalar／rank2 terminal、実在のthin/WY/controlled圧縮対照、共通strict phase・error allocation、cold/cache会計、有限call/output/memory上限である。fixtureのmatrix/seed/row数や勝利閾値はここでは選ばない。新準備契約では利用者のperformance政策に従いwall capを原則nullとし、旧凍結契約は変更しない。

**結果を見つけるための条件追加や、別sampling/Hamiltonian候補への自動移行は行わない。公開確認後mandatory STOPし、GPT側へ方法差・着地点の判断を戻す。**

## 10. 一次資料の確認箇所と限界

| 一次資料 | 取得・確認した関連箇所 | 技術的用途・限界 |
|---|---|---|
| [Camps et al., 2108.03283v2 PDF](https://arxiv.org/pdf/2108.03283v2) | Intro pp.1–2、Sec.3 fusion/turnover、Sec.8 numerical implementation | 既知Gaussian/subcircuit compression。特定compilerのT最適性や本candidateとの出力完全一致は未認定 |
| [Kökcü et al., 2303.09538v2 HTML](https://arxiv.org/html/2303.09538v2) | II.2、III.1、IV.2 Eq.(42)–(43)、Appendix C.1/C.2 | controlled対応の既知性。全Fock制御phase・gate set・precisionの本source adapterは別に必要 |
| [Schreiber/Van Loan, 1989](https://epubs.siam.org/doi/10.1137/0910005) | 出版社abstract | compact WYの直接prior。論文全文へのアクセスを主張しない。一般再帰は本監査で展開 |
| [LAPACK ZLARFT](https://www.netlib.org/lapack/explore-html/d7/d0d/group__larft_ga99d53d027cc196a5e5f7e2bbacdf4979.html) | Purpose、DIRECT/STOREV、Further Details | 複素reflector表現と順序。dense Vのdrop-in適用をしない |
| [Motta et al., 1808.02625v2 PDF](https://arxiv.org/pdf/1808.02625v2) | p.2 gate count、supplement pp.7–8「Implementing basis rotations」 | partial QRとrank依存count。2021出版社本文は取得エラーのため同じ本文とは断定しない |
| [Jiang et al., 1711.05395v3 HTML](https://arxiv.org/html/1711.05395v3) | III Eq.(7)–(13)、Appendix A Eq.(77)–(81) | one-particle表現・state preparationの境界。occupied-state gaugeをfull-unitaryへ移さない |

Mottaのversionless PDFは取得本文にv2表示があり、明示v2 URLでも確認した。v3を取得済みとはしない。関連検索で取得した別タイトルのHuggins PDFは、このmethod-delta表の根拠に採用していない。

取得先の不整合・未取得箇所を隠して全文確認済みとは書かない。上の直接対応が、他に公表済みのDF-RTE専用実装が存在しないという網羅的な証明にはならない。
