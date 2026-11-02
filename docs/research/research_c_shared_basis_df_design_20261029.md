# 研究C：Shared-basis DFの代数的同値性・PR政策の切り分け

2026-10-29 JST。起点`a8437b6ae88e133c561ac002dd41ee3008dfc91f`。独立branch `codex/research-c-shared-basis-design-20261029`。正式共通運用指示2026-10-25版の§8.1/8.2/8.4/10/10.1/10.3を実ファイルで確認した。依頼された数学・SOURCE調査と人工PoCの技術報告であり、別の詳細GPT科学レビューではない。

**技術提案：一般対称JのShared-basisを独立したDF分解アルゴリズムとして新設する案はNO-GO。** G個の共有groupは通常CDFのouter rank **G以下**と同じ表現集合であり、専用fitterも重複fitも不要である。一方、同じHamiltonianに対するfragment／atom化政策がPRに与える影響は別の問いとして残る。本PoCはその機構を確認したが、独自性・実H4の資源改善・科学実行GOを認定しない。Shared-basis一般やPR向け低rank DF全体を否定する結論ではない。

## 1. 出発点・既存結果の位置付け

以下は保存報告の引用で、新たな保存H4の数値decode・再集計ではない。直鎖H4・1.00 Å・STO-3G、number CDF8／元順LD2、partial ordered S2 δ=h=.1／s1／r128／K0／m0..12、own-DF ground-state oracle、状態準備除外、既存P0、ideal-exact-native仮定の有限校正点比較である。

| 保存DF | λ_R (Ha) | 期待basis切替 | Kernel平均RZ | 条件付き総RZ／M4 |
|---|---:|---:|---:|---:|
| PR-aware γ=.1 | 1.4424 | 58.39 | 14076.44 | 1.000 |
| Tail-P4 γ=.1 | 1.5162 | 65.24 | 15047.47 | 1.176 |
| 通常CDF γ=0 | 2.2098 | 80.46 | 17454.19 | 2.678 |

[M4-Native128](research_c_pr_aware_df_M4_native128_execution_20261028.md)、[tail-P4](research_c_df_structural_pr_performance_20261028.md)、[第三DF修正実行](research_c_df_basis_third_repair_execution_20261029.md)を確認した。第三holdoutでA MAE369.79がB303.33を上回り登録転移判定は不成立。[旧148-callの複合C](research_c_df_basis_surrogate_repair_execution_20261029.md)も未資格。回帰係数をηへ移さず、今回の設計をそれらの事後救済にしない。Actual CA、全law合成誤差、費用support、母平均優位の旧未資格も維持する。

## 2. 再利用対応と一次資料

| SOURCE／資料 | 適合性・再利用・今回の境界 |
|---|---|
| [C-DF接口調査](research_c_compressed_df_interface_survey_20261024.md)、[PR-aware目的設計](research_c_pr_aware_df_optimization_design_20261026.md) | 一般J、元g補正κ、number規約、外部通常CDFを強対照とする位置付けを継承 |
| ffsim0.0.83 `linalg/double_factorized_decomposition.py`／`util.py` | 既存rank指定・tensor loss/AD・real上三角Jと反対称回転pack。人工ADで実関数を使用し、fitter/minimizeを例外mockで遮断 |
| `c1_ffsim_compat.py`／`c1_ffsim_corrected_codec.py`／`c1_real_orthogonal_codec.py` | batch-logm互換・real Schur列符号gauge・qualified x0。既存codec回帰を再利用。本体・venv・旧callerを編集しない |
| `c_pr_aware_df.py`／`c_pr_aware_df_pilot.py` | Jからtail Z/ZZへの線形写像、loss/AD・fit接続。新fitter・新penalty・γ/ηを実装しない |
| `c1_h6_initialization.reconstruct` | 人工tensorの同値性に既存contractionを使用。H6診断entryは起動しない |
| `c1_general_k_adapter.py`／`c1_stage1.py` | `_matrix/_spin_lift`、一般K係数と独立raw CAR、scalar/κを人工full16で照合 |
| `c1_molecule_extension.py` | `factor(max_vecs=rank)`、SectorCAR／rank_assessmentは将来通常CDF7の再利用先。今回はfit・rank探索・実sector評価なし |
| `c_df_structural_support.py` | compact injection／pullbackの既存例。共有Uの冗長fitterには流用せず、固定supportとgroup制約を混同しない |
| `c1_stage3_connection.py`／`c_pr_aware_df_stage_b.py` | prefix・identity抽出・basis key・原frameを保持。人工arm→既存atom/law/referenceをそのまま通す |
| `c1_stage3_reference.py`／`trotterlib/rte.py` | 部分PFのtail和、固定paired-Taylor多項式、正規化・attenuation。人工K0/K2だけで照合し実H4を評価しない |
| `c_pr_aware_df_stage_b2.py`／`c_df_structural_pr_native.py` | full CAR reference・law metadata・native32・P0校正の既存下流を静的確認。旧runnerは起動しない |
| `c_df_basis_surrogate.py`／`c_df_basis_switch.py` | exact-key、law descriptor、run intervalを再利用。未資格価格予測を採択しない |
| NativeBuilder／PrecisionBuilder／StrongBuilder／Gaussian helper | 隣接run・中央fsum・head相対融合・位相を静的照合。新compiler・Gaussian合成器・H4 build/compileなし |

[ffsim固定SOURCE](https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/linalg/double_factorized_decomposition.py)は一般対称diagonal Coulomb行列と軌道回転の既存CDFを定義する。通常CDFの表現・最適化は既存機能であり、その再実装を寄与としない。[Cohn–Motta–Parrish, 2021](https://arxiv.org/abs/2104.08957)もC-DFを時間発展に用いる既存研究である。可変のwebドキュメントは版表示が変化したため、実装判断には固定commitとlocal0.0.83 SOURCEを使った。これは限定的な一次資料照合で、網羅的新規性レビューではない。

## 3. 表現集合の同値性：Q1／Q2

`T(U,J)_{pqrs}=Σ_ij U_pi U_qi J_ij U_rj U_sj`と置く。互いに素な非空group `G_a` が全R frameを覆い、`U_l=V_a (l∈G_a)`なら、Jに関する線形性から

\[
\sum_{l=1}^{R}T(U_l,J_l)=\sum_{a=1}^{G}T(V_a,K_a),
\qquad K_a=\sum_{l\in G_a}J_l.
\]

一般実対称Jは和で閉じる。逆に、任意の通常G-frame CDFに対して、各group内で`J_l=w_l K_a`、`Σw_l=1`、`U_l=V_a`とすれば同じtensorを得る。従って、groupの固定サイズを含めても

\[
\boxed{\mathcal S_{\mathrm{shared}(R,G)}=\mathcal S_{\mathrm{CDF}\le G}
\subseteq\mathcal S_{\mathrm{CDF}\le R}}.
\]

Group数は有効outer-rankの**上限**であり最小rankではない。K_a=0、別groupのVがさらに同一、他の表現冗長性ならもっと少なくなる。「常に実効rank7ちょうど」とは呼ばない。また、この包含が対象nや特定物理tensorで真に狭いか、H4で予算を落とすかは今回証明・測定していない。パラメータ数の減少だけを実際の表現多様体の次元やDF誤差増加と同一視しない。

**CDF8でtail F2/F3だけを共有し、残り6 frameを独立にする例はCDF≤7と同じ集合**。Head F0/F1を自由にできるなら通常CDF≤7全体に対応する。Headを別途凍結する場合は「同じheadを凍結したCDF≤7」が対応baselineで、自由なCDF7より狭くなり得る。Head/tailをまたいでgroup化すればsplit政策も変わるため、今回のPoCは禁止する。

### 冗長性と最適化

各Jの独立entry数を`m=n(n+1)/2`、回転局所パラメータ数を`k=n(n−1)/2`とする。共有Uのまま別Jを残すと名目変数数`R m+G k`、groupパラメータでは`G(m+k)=G n²`。少なくとも`(R−G)m`方向の

\[
J_{l_1}\mapsto J_{l_1}+\Delta,\quad
J_{l_2}\mapsto J_{l_2}-\Delta,\quad\Delta=\Delta^T
\]

がtensorを変えない。純tensor lossのJ勾配はgroup内で同じになり、この差方向はnull。共有回転への勾配は元各frameの勾配の和であり、既存ADのchain ruleで扱える。H4サイズn=4/R8/G7なら128→共有U付き122→冗長Jを除いた112という**名目パラメータ数**で、null Jは10方向。Gaugeや零group等の追加不識別性もあり得る。

この数学的冗長性と、局所optimizerが異なる軌跡・停止点を返すことは別である。冗長変数で一つのfitが良くなる可能性を排除しないが、それを新しい表現能力・大域最適性と主張できない。Fragment別のL1/L2/smooth penaltyを足すとloss自体がsplitに依存し、冗長な分割でpenaltyが変わり得る。比較ではgroup Kへ同じ目的を適用するか、目的変更として別要因にする。

例外はJにrank-one・別support等の追加制約を課す場合で、同じUの和が単一のその制約内Jへ戻らないことがある。**今回の一般対称J CDFの仮定にはその例外を持ち込まない。** 近接する別Uを同じVへ置換する操作も、この同値性の適用外である。

## 4. Spin・κ・scalarの保持

元gからの`κ_pq=h_pq−(1/2)Σ_r g_prrq`を固定し、merged g_DFから作り直さない。各number fragmentは

\[
D_l=\tfrac12\sum_{ij\sigma\tau}J_{lij}n^{U_l}_{i\sigma}n^{U_l}_{j\tau},
\quad K_{ab}=J_{a//2,b//2}/2,\quad a=2i+\sigma.
\]

同Uでの和はD_l／spin K／I/Z/ZZ係数の各段階で線形。従って元κと核scalar一回を保持すれば、full-Fock Hamiltonianも同じであり、任意sectorへの制限も同じである。I係数も必ず加算してcontrolled相対位相に残し、λからのみ除外する。Spin順・Uの向き・1/2を変更しない。

PoCは既存raw CARと一般Kの両経路を、人工2 spatial orbital／full16で比較した。Tensorの二乗を別squareとして追加したり、JをPSD化したりしない。Mathematical equalityとbinary64丸め後の等値は区別する。PoCは既存fsum型の和を用い、再構成残差を人工testで確認。実分子のguardや許容差を変更していない。

## 5. A保持／B統合／C通常低rankの関係：Q3

| 対象 | A：共有U・別J | B：K=ΣJへ統合 | C：通常CDF≤G |
|---|---|---|---|
| 同じJの和を使うHamiltonian | 同一 | 同一 | BのJ/Uをそのまま用いれば同一 |
| 表現集合 | Cと同じ、冗長 | Cと同じ | 既存通常CDF |
| Ordered全frame PF | 元因子列 | 統合位置・隣接性で変化し得る | 同じJ/U・順序ならBと同じ |
| Prefix D/R | 元splitを保持 | 境界変更を別要因として禁止 | 比較相手と物理的D/Rを合わせる |
| Tail atoms／λ | Fragment別に絶対値を取る | 同basis・同supportの係数を先に加算 | Bと同じatom化なら同一 |
| 有限PR law・native価格 | 別fragment labelを保持 | Lawの変更は別評価が必要 | 同policyならBにない固有の利点はない |

### Ordered PFと部分PFを区別する

同Uの密度fragmentは可換なので、**隣接して同じ時間を持つ**`exp(−itD1)exp(−itD2)=exp(−it(D1+D2))`は合法な局所rewrite。通常のordered symmetric S2でも、その共有pairが隣接すれば両halfを対応付けられ、中心にかかるpairは可換性を使ってまとめられる。一方、別の非可換因子を挟む同basis fragmentをまとめるには、その因子を横切るため一般には別PFになる。人工`[Z,X,2Z]`対`[3Z,X]`でHの和は同じだがS2が異なる反例を確認した。一般のscheduleでは単なるgroup名だけで時間係数の対応を省略しない。

現在の研究Cの**partial** referenceは`H_D`のordered halfと、**tail全体の和H_Rへのexact exponential**を組み合わせる。従って今回のtail内統合でH_D/H_Rとidentityを同一に保てば、tail frameが非隣接でもそのPF referenceは変わらない。これはtailをorderedな各frame exponentialで近似する全frame PFとは異なる。Head内の非隣接統合やhead/tailをまたぐ統合へ拡張しない。

### 係数の相殺・有限law

同じ厳密basisかつ同じPauli support sでは`c_Gs=Σ_l c_ls`なので

\[
\lambda'_R=\sum_{G,s\ne I}|\sum_{l\in G}c_{ls}|
\le\sum_{l,s\ne I}|c_{ls}|=\lambda_R.
\]

符号が一致する係数だけなら等号。符号相殺なら小さくなり、零になったsupportをexact-zero規則で除く。異なるbasisの係数をこの式で相殺しない。

Fragment labelのlawは統合前後で異なる。一方、同一signed physical Pauliだけの同符号統合なら、labelを忘れた理想IIDのpushforward lawは同じになり得る。「統合したら必ず別の物理law」とも主張しない。相殺でλが変わるとτ=λh/r、signed-atom確率、order重みとnormalizationが変わる。物理Hを保つだけで生trajectory／raw平均信号が同じとは限らない。

重要な限定等値として、既存`finite_taylor_operator`は偶数Kについて次数K+1までを保持するため、h/r/Kを固定し同じH_Rを使うと

\[
\left[\sum_{n=0}^{K+1}\frac{(-i\tau H_R/\lambda_R)^n}{n!}\right]^r
=\left[\sum_{n=0}^{K+1}\frac{(-ihH_R/r)^n}{n!}\right]^r.
\]

従って**normalization-corrected有限RTE referenceと、そのPFに対する誤差はこの条件では不変**。Raw平均は別のnormalizationで減衰し、K0では`A=(1+(λh/r)^2)^(−r q/2)`が変わる。この条件を人工K0/K2で既存referenceにより確認した。Kを残差基準から選び直す、D/Rを変える、h/rを変える等ではこの等値を移送しない。λ=0は正規化除算を使わず、既存no-random-tail分岐を別確認する必要がある。

また、λが下がってもbasisごとのqが変わり、期待切替`(r−1)(1−Σq_b²)`は増え得る。人工反例もPASSした。固定rの中央event数がatom catalogの数とともに減るわけでもない。以上の利益・不利益は**fragment／atom化政策**の効果であり、Shared-basis特有の新表現能力ではない。

## 6. 現native SOURCEで共有できること：Q4

| 実装箇所 | 実際の規則・限界 |
|---|---|
| `split_native_model` | `sha256(repr(block.rotation))`。Spatial Uを既存`_matrix`→`_spin_lift`でinterleaved spinへ変換。同じ配列を同じ経路で複製すればfragment ID/Jに依存せず同じtail key |
| Signed zero／gauge | −0と+0はtuple等値でもreprが異なる場合がある。列符号はdensity projector、列置換はJの対応置換でtensorを保つがkey一致を保証しない。転置・微小回転も別key。PoCはcanonicalizationしない |
| `native.runs` | 決定論側は隣接`rotation == rotation`のtuple比較。Tailのrepr hashより弱いsigned-zero区別であることを記録し、両方を無条件に同じ判定と呼ばない |
| `combined_diagonal` | 同じbasisのtime-weighted I/RZ/RZZをfsumしexact-zeroだけ削除。固定full ordered PFの中央係数相殺は可能 |
| `build_event_sequence`／`_basis_reuse_intervals` | Tailの隣接exact-key runごとにB⁻¹を一回、全中央操作、Bを一回。k個runなら境界のk−1対を除ける。別keyを跨ぐ共有はしない |
| Tailの中央操作 | 現emitterはrun内の中央eventを順にemitする。動的な同supportの角度を事前合算する最適化は実装していない。後のWhole-P0が消す可能性と保証を分ける |
| `StrongBuilder`／`PrecisionBuilder` | Head内部は異basis間にも相対Gaussian `V_next† V_prev`を使い、analytic phaseと終端basisを保持。別basisの費用がゼロになるという意味ではない |
| Stage B `build_kernel` | Head両half・tailは別segment。Tailにも同じprecision basisを使うが、cross-head/tail相対融合は明示的に無し。終端復元・境界を無料扱いしない |
| Whole-P0 | RZ/CX/depthは全回路のsimplificationに依存。Static fragment数・run数からcompiled削減率を保証しない。旧P0、spin規約、合法融合を両armで揃え、今回は変更しない |

物理的なdensity-projector同値と、現recipeで同じkeyになることは別である。近いUを同一視したり、codecのgaugeをkey共有増加へ転用しない。新PoCはexact-key不一致・非直交・不正partition・split横断を明示的に拒否する。非連続groupはHamiltonianだけを比較する`representation`モードでのみ許し、PF／PR／費用同値性を返さない。

## 7. 候補A/B/C・強baseline：Q5／Q6／Q7

| 候補 | 技術判断 |
|---|---|
| A：F2/F3共有、別JのCDF8 | CDF≤7の冗長表現。共有回転6変数を減らしてもJのnull10方向を残す。独立方式・専用fitは推奨しない |
| B：groupごとV/K | 既存通常CDF≤7そのもの。ffsimのrank指定・corrected codec・pack/unpack・loss/gradient・L-BFGS-Bを再利用すればよい |
| C：PR-aware grouping／atomization | 同一HのD/R・係数相殺・law・run・価格の共設計として問いは残る。ただし既存CDF7の同じcanonical atom化にも適用できる。共有方式固有の利点とは認定しない |

公平な評価では、現最良保存PR-aware CDF8と通常CDF8に加え、**適格な通常CDF7**を必須対照にする。今回確認した関連CDF保存記録では、同条件・物理sector予算まで資格付けた通常CDF7を確保していない。XDF7 prefixやCDF8の一frame削除をCDF7 fit結果として代用せず、新しい数値を補完しない。

将来比較するなら以下の2段階を分ける。

1. **表現／fit対照**：同じ元H・DF予算、明示したrank上限、同じ初期group K/V、corrected codec・solver・loss・guard。BとCを同一初期値・目的にすれば、同じ最適化問題を二度走らせる必要はない。Headを凍結するならbaselineも同様に凍結。独立初期値・冗長パラメータ・別penaltyを変えた局所解差を構造優位と呼ばない。
2. **実装政策対照**：同一の適格な低rank J/Uと同一D/Rに対し、fragment labelを保持した場合とcanonicalな同basis同support加算を比較。物理H・独立CAR・identity位相をまず一致させる。固定PF/RTE条件で変わるのは主にlaw/normalization/trajectory価格で、各armのλ・減衰・shotsを取り直す必要の有無を式から判断する。

H4条件は直鎖1.00 Å/STO-3G、Nα=Nβ2/sector36、通常number CDF、元g補正κ、scalar一回、general symmetric J、real orthogonal U、CA=.00159 Ha、`||H−H_DF||sector,2+guard≤CA/8=.00019875 Ha`。Tensor lossを物理予算PASSに代用しない。各DFのown-ground oracle・状態準備除外、元prefix LD2、δ=.1/m0..12、既存s/r/KとP0・failure/phase配分を比較前に固定する。Rank7の物理予算不適格は、そのrank/初期化/solver条件の結果であり、Shared-basis全般の不可能性とは呼ばない。

現時点で、低rank baselineにも同じgroupingを適用した後に残る**独立したPR改善機構は未確立**。相殺によるλ低減自体は代数的に説明でき、製造した重複fragmentを改めて統合するだけの実分子実験は推奨しない。

## 8. 人工PoC・回帰・変更範囲

新規SOURCEは純粋人工PoC [`c_df_shared_basis.py`](../../src/trottertracks/c_df_shared_basis.py)、評価testとAD testの3ファイル。分解法、optimizer、Gaussian、compiler、sampling、科学runnerは追加していない。既存SOURCEを一切編集せず、`reconstruct`・一般K/raw CAR・既存law/PF/RTE・exact-key/run helperを呼ぶ。旧indexからgroupへの対応も返し、無断で元frameをsortしない。新しいatom IDや科学runへの登録adapterを実装済みとは扱わない。

| 試験 | 内容 | 結果 |
|---|---|---|
| 新人工20＋旧switch11 | 人工2/4 orbital、rank8→7、非PSD/相殺、tensor/full CAR/κ/scalar/spin、signed zero/列符号/列置換/転置/near-U、隣接run、PF反例、K0/K2、λ減少かつswitch増加、不正group/split/配列のSTOP | 31 PASS／1.63秒 |
| 新AD1＋旧AD1＋corrected codec15 | 実ffsim0.0.83のpack/unpack/loss/gradient、group内J勾配とnull方向、共有U勾配の和、既存codec det±/πの非回帰。Fitter/minimizeは例外mock | 17 PASS／3.83秒 |
| 旧一般K独立回帰 | 係数・spin・scalar・参照接続・passive境界 | 13 PASS＋29 subtests／.26秒 |
| **合計** | **新21＋既存40** | **61 unique PASS＋29 subtests** |

[人工ログ・環境・SOURCE・verification](../../artifacts/research_c_shared_basis_df/2026-10-29/technical/)。初回人工testで、2D QRがdet=-1の対称な反射になり「転置すると別basis」のfixtureになっていなかったため、明示的な非対称回転へ修正した。AD testの符号配列の記述も修正した。科学的な閾値変更や実分子再実行はない。

これは技術的検証で、実H4の新fit・数値再解析・PF/RTE・sampling・ground・native build/compileは全て**0**。人工データの既存小行列／JIT経路は数秒で完了し、重い分子処理を直列化したわけではない。新たなCPU/RAM/worker/wall capなし。旧依存環境はread-only、cacheはtask-local。CIや外部独立再現の結果とは呼ばない。

## 9. 未承認契約案の扱い・科学実験を推奨しない理由：Q8

依頼§10/§11/§12の「同値なら新fitterや重複fitを推奨しない／実験不要の理由を報告する」を適用する。**Shared CDF8実効7と通常CDF7を別々にfitして、新分解法固有の利点を測る科学実験は提案しない。** 数学的に同じ変数K/Vと目的を二重に実行するためである。現時点で物理予算を満たしたCDF7の証拠もなく、実H4に固有の価格削減見通しは得ていない。

[未承認の設計判定案](research_c_shared_basis_df_contract_unapproved_20261029.json)はこのNO-GOと必要条件を機械可読化した**非実行案**で、`executable=false`、科学call上限**0**、fresh scientific output未指定、fit/optimizer/retry/resume0。実行可能な科学契約が完成したとは扱わず、例示条件を採択済みのpilotへ格上げしない。承認証明や新grantではない。

次の優先方向の提案は、Shared-basisという新方式ではなく、**通常CDFのPR-aware rank政策**を検討すること。ユーザー／研究Cが必要と判断した場合、既存fitterによる通常CDF7一条件のDF予算評価から別契約を具体化する。初期値とfit回数・call上限を固定し、DF精度→同じ低rank問題との差→PR law/PF-RTE→native価格→条件付き資源の順で必要性を判断する。今回はrank7 fitやその科学契約を先回りして確定・実行しない。

Fragment／atom化共設計を次の主題とするなら、低rank・canonical atom化にも適用した強対照の後に、何が独立して残るかを先に定義する必要がある。Switch-only、旧複合surrogateのη採択、別γ/r探索、新native最適化へ戻らない。

## 10. 回答・公開・mandatory STOP

- **Q1**：一般対称J・共有G groupの表現集合は通常CDF≤Gと同値。
- **Q2**：新DF表現としての独立価値は確認できない。冗長変数の最適化効果とPR実装政策は別問題。
- **Q3**：λは非増加、相殺なら減少し得る。生law/減衰は変わり得るが、同一H_D/H_R・固定h/r/Kなら現在のpartial PFと補正RTEは同じ。全frame ordered PFは順序条件が必要。
- **Q4**：Tailは隣接exact-key runのGaussian対を共有、中央eventは順にemit。Head相対融合とは別で、head/tail境界の無料融合やWhole-P0削減を仮定しない。
- **Q5**：Atom化・正規化・D/R政策にPR固有の問いは残るが、今回低rank CDFと区別できる独立改善は未確立。
- **Q6**：専用fitter不要。将来必要なら既存通常CDFのrank、合法grouping／atom化を用いる。
- **Q7**：同じCDF7 J/U・目的・D/R・P0にも同じ最適化を適用した後の差が必要。表現能力の差はない。
- **Q8**：独立Shared-basis DFとしての実分子実験はNO-GO提案。通常低rank CDFのDF予算とPR資源の交換条件を次の検討候補とする。

SOURCE・人工証拠・本報告・非実行案をcommit/pushし、remote SHA/tree・SOURCE・旧原本の保全を照合する。**公開照合後mandatory STOP**。ユーザー／研究Cがこの位置付けと次の研究方向を判断し、Codexは別の具体的依頼と必要な実行許可を受けた後だけ対応する。新fit／保存値再解析／PF-RTE／native／別rank探索／他分子／詳細GPT科学レビューは自動開始しない。
