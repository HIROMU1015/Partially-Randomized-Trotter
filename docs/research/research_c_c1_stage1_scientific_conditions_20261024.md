# C1 Stage 1 科学条件案（2026-10-24、未採択・未実行）

本稿は、同じ元物理Hamiltonianに対する**C-DF近似誤差**と**一般K adapterの表現誤差**を分ける、初回1条件の提案である。推奨は保存P-H4、number表現、元gから補正したκを保持、`max_vecs=8, tol=1e-8, optimize=True`、通常L-BFGS-Bの1 fit。条件の採択、技術実装、科学実行はいずれも今回の承認に含まれない。

状態は `C1_STAGE1_DESIGN_PREPARED_ADOPTION_AND_IMPLEMENTATION_REQUIRED`。実行可能なSOURCE・依存lock・数値allowanceの係数は未確定であり、科学GOではない。[実行準備報告](research_c_c1_stage1_execution_readiness_20261024.md)と[未承認契約案](research_c_c1_stage1_contract_draft_20261024.json)を合わせて読む。既存の共通研究状態・C4終了・Gaussian singleton保留を変更しない。

## 1. 根拠と今回の調査範囲

- C1 adapter基準SOURCE：[`31705b2caddc8bee316ff241055dcc23b18ba8bc`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/31705b2caddc8bee316ff241055dcc23b18ba8bc)。[adapter](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31705b2caddc8bee316ff241055dcc23b18ba8bc/src/trottertracks/c1_general_k_adapter.py)と[前回技術報告](research_c_c1_general_k_adapter_preparation_20261024.md)を静的確認した。前回の人工unit/mock 13件PASSは実CAR・実分子Hamiltonian・実GaussianのPASSではない。今回は再実行していない。
- P-H4：[入力回収報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8a0cd683f65e189f5900ce002af51fce2c53059/docs/research/research_c_c1_input_ffsim_readiness_20261024.md)。今回はGit blobのbyte SHA256と保存来歴を照合し、配列値の演算・再解析を行っていない。
- ffsim v0.0.83、commit `558d3517385778c733f1aa226bb61b8f30f4b0f3`：[Hamiltonian API](https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/hamiltonians/double_factorized_hamiltonian.py)、[fit SOURCE](https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/linalg/double_factorized_decomposition.py)。通常C-DFを対象とし、RCDF/SCDF、正則化、別one-body政策を含めない。
- 共通運用は `research_project_instructions_unified_2026-10-23.md` §8.4、§10、§10.3。今回の変更は独立worktreeの新規3文書のみ。fit、実積分による行列・norm・residual、環境構築、科学コードの追加は0。

## 2. 初回物理条件の候補

| 項目 | 推奨候補（正式採択前） |
|---|---|
| 元入力 | P-H4、保存MO h/g/cをそのまま使用。SCF/積分の再生成なし |
| 分子 | 直鎖H4、H座標 `(0,0,z)`、z=0,1,2,3 Å、隣接1.00 Å |
| 電子・軌道 | charge 0、spin_2S=0、4空間/8 spin軌道、実canonical RHF MO、STO-3G、symmetry=false、active/frozen-core指定なし |
| 主評価sector | Nα=Nβ=2、次元 `C(4,2)^2=36` |
| 表現確認 | 8 spin軌道full Fock次元256をadapterのbaseおよび各frameの同値性にだけ使用 |
| X-DF対照 | 同じpacketの保存実rank10、旧生成 `df_rank=None, df_tol=1e-8`。再分解しない |
| split/時間 | `L_D`、delta窓、PF/RTE時間・分割はStage 1に該当しない、未採択のまま |

固定input commitは `86b8b002704aced8ca239414080f53793de81658`、pathは `artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`、SHA256は `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。物理入口は `one_mo[4,4]`、`eri_chemist[4,4,4,4]`、`nuclear_repulsion[]`、MO identityは `mo_coeff[4,4]`。元spatial gに1/2を前掛けしない。保存 `spin_two` は既にInteractionOperator用の1/2込みであり、spatial入口へ使わない。

配列orderingはspatial chemist g、spin mode `a=2p+σ`（α=0, β=1）。referenceの共通state indexはmode aをbit aに置くlittle endian。OpenFermion/既存PhysicalSectorのmode 0最上位bitとのbit reversalは一度だけ行い、36個のsector stateを同じ順序へ写す。spin-block/interleavedの軌道permutationとstateのbit reversalは別操作である。

## 3. Hamiltonianとκ・constantの帰属

以下は固定SOURCEの代数式であり、今回評価していない。

\[
H_{phys}=c+\sum_{pq\sigma}h_{pq}a^\dagger_{p\sigma}a_{q\sigma}
+\tfrac12\sum_{pqrs\sigma\tau}g_{pqrs}a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}.
\]

\[
C(g)_{pq}=\sum_r g_{prrq},\qquad \kappa=h-\tfrac12C(g_{original}),
\]
\[
H_{CDF}=c+\sum_{pq\sigma}\kappa_{pq}a^\dagger_{p\sigma}a_{q\sigma}
+\tfrac12\sum_{lij\sigma\tau}J^l_{ij}n^l_{i\sigma}n^l_{j\tau},
\quad n^l_{i\sigma}=\sum_{pq}U^l_{pi}\overline{U^l_{qi}}a^\dagger_{p\sigma}a_{q\sigma}.
\]

推奨は**ffsim factoryと同じ元g由来のκを保持する政策**。fitはgのJ/Uを更新する。`g_fit=Σ_lij J_lij U_lpi U*_lqi U_lrj U*_lsj` とすると、normal-orderedへ戻したone-bodyは `κ+0.5 C(g_fit)` であり、一般に元hと同一ではない。この違いも定義したH_CDFの近似誤差に含める。元h固定を目的にκを再補正する別政策は今回の比較へ混ぜない。[固定API SOURCE](https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/hamiltonians/double_factorized_hamiltonian.py)

adapterは `enable_general_k=True, z_representation=False, one_body_role="ffsim_corrected_kappa"` で取り込む。`K_(2i+σ),(2j+τ)=J_ij/2`、`V_(2p+σ),(2i+τ)=δ_στ U_pi` として1つの元Jを1つのframeに保つ。`D_l=Σ_ab K^l_ab n^V_a n^V_b` は対角と両順序込み。Jをrank1/PSDと仮定せず、負の係数も保持する。

modelはκと元nuclear constant cを各1回だけ所有する。frameのnumber→I/Z/ZZ展開で生じるidentity係数はそのframeが所有し、cへ移して再加算しない。Z表現への変換、scalar除去、signed-square再分解、threshold pruningを行わない。

## 4. rank/tol/optimizerを1案へ絞る理由

| 比較案 | 初回の意味と制約 | 判断 |
|---|---|---|
| `max_vecs=10` | 保存X-DFと同じouter上限。4軌道の実対称pair空間次元10まで許すため、ほぼ非圧縮の初期化で終わり一般Jの効果を調べにくい可能性。実R=10は保証されない | 保守的な資格確認案だが今回は非推奨 |
| **`max_vecs=8`** | rank10より小さい上限を事前に課し、比較的軽い圧縮で表現と近似誤差を切り分ける。収束・精度は未保証 | **初回1条件の推奨** |
| `max_vecs=6`以下 | より強い圧縮。近似誤差や非線形fitの難しさが初回資格確認に重なる | 必要性を結果後に別検討、今回のcallに含めない |

8はSOURCEと実験目的に基づく初回の設計判断であり、最適rankの推定ではない。8 frameと10 squareのnative費用は未比較で、20%の回路削減を意味しない。同outer rankでも一般J frameとsquareの費用・近似精度は等価ではない。最終Rが小さい、またはJが実質rank1でも追加fitは行わず、その性質を報告する。

| fit field | 推奨する固定候補 |
|---|---|
| API | `ffsim.linalg.double_factorized(g, ..., return_optimize_result=True)` を1回 |
| 設定 | `optimize=True, max_vecs=8, tol=1e-8, method="L-BFGS-B", options={}, callback=None` |
| 初期化・support | `cholesky=True, diag_coulomb_indices=None`、全対称J support、通常Cholesky初期値のみ |
| objective | `0.5 ‖g−g_fit‖_F²`、実tensorの全要素、正則化なし |
| precision | JAX `jax_enable_x64=True`を実測確認、fit/export float64、operator complex128 |
| solver版 | **SciPy 1.14.1を候補として固定**。ffsimの依存許容範囲内だが独立環境での解決・import確認は未実施 |
| 終了資格 | `OptimizeResult.success is True`、status/message保存、x/fun/jac/J/U/κ/cが有限、実Rとshape・規約が整合 |
| 付帯記録 | nit/nfev/njev、最終tensorのmax-absolute/Frobenius残差、実R、初期化設定、device・dependency lock |

`tol=1e-8` はffsim既定の初期分解閾値を維持する選択である。保存X-DFの同じ数字との等精度を意味しない。SOURCEではmodified Cholesky初期化をrank上限または残差で止め、そのRで非線形fitする。`optimize=True`では `cholesky=False` も初期化を変えない。`tol`は最終optimizerの停止条件、最終tensor残差、sector normの保証ではない。[固定fit SOURCE](https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/linalg/double_factorized_decomposition.py)

SciPy 1.14.1で `options={}` とした実効既定は `maxcor=10, ftol=2.220446049250313e-9, gtol=1e-5, maxfun=15000, maxiter=15000, maxls=20`。`jac=True`でJAX勾配を使うため数値差分epsは使わない。ftolは目的値の相対減少、gtolは投影勾配の停止基準であり、物理Hamiltonian誤差ではない。solver内部の反復・評価上限はこの1 fitの停止政策であり、CPU/RAM/wall-time上限ではない。[SciPy公式文書](https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.minimize-lbfgsb.html)、[v1.14.1 SOURCE](https://github.com/scipy/scipy/blob/v1.14.1/scipy/optimize/_lbfgsb_py.py)

factoryはOptimizeResultを返さないので、factoryでfitした後にstatus取得のため再fitしてはならない。将来の薄い接続処理はlower-levelを1回だけ呼び、factoryと同じκ式と元cを組み合わせてexportする。SOURCEはsuccessを強制しないため、呼出側で未収束を保留・STOPにする。rank/tol/seed/初期値/optimizerのsweep、restartは0。

## 5. 近似誤差A、表現誤差B、X-DFの別評価

`P`を36次元sectorへの制限とし、主指標は絶対spectral norm（Hartree）とする。

\[
\epsilon_{DF,CDF}^{sector}=\|P(H_{phys}-H_{CDF}^{rawCAR})P\|_2,
\qquad
\epsilon_{DF,XDF}^{sector}=\|P(H_{phys}-H_{XDF}^{saved})P\|_2.
\]
\[
\epsilon_{adapter}^{sector}=\|P(H_{CDF}^{rawCAR}-H_{CDF}^{K})P\|_2.
\]

前2者はconstant/one-body/二体項を全て含む近似誤差の**測定・報告**であり、今回根拠のある絶対DF許容値は導けていない。初回案では絶対精度gateを新設しない。tensor残差も別に報告し、energy計算は初回に含めない。後者は同一C-DFの表現誤差なので事前固定した数値allowance内を要求する。

同値性の見落としを避け、full256でκ+cのbase 1組と各frame R組も比較する。固定電子数sectorだけでは `α(N−4)` 型の補正誤りやframe間の相殺を隠せるためである。vacuum・1粒子sectorも含むこの検査でdensity対角、1/2、index、U/V向き、I/Z/ZZを確認する。H_physとX-DFのfull norm・full diagonalizationは追加しない。full行列の生成が必要な経路でもsector値以外の物理誤差評価を増やさない。

## 6. 独立referenceの構成案

| 経路 | 入力・再利用 | 独立性の条件 |
|---|---|---|
| H_phys | 元h/g/c → 保存生成SOURCEのunthresholded spin lift → OpenFermion InteractionOperatorの係数を無閾値でFermionOperator term mapへ渡す → 既存sparse変換 → 36sector | fit/adapter/saved DFを読まない。SCF関数を呼ばない |
| H_CDF raw CAR | exportの元κ/J/U/c → 既存generic `car_annihilator` の8-mode bit作用 → bilinear `a†a` → 上記density-product式 | adapterのK/V/Z係数を使わない。raw JとUを独立にspin展開 |
| H_CDF K | adapterのκ/c、frame K/Vと `general_k_coefficients` → OpenFermionのbilinearから `Z_a^V=I−2Σ_pq V_pa V*_qa a†_p a_q` を構成 → I/Z/ZZの積和 | raw CAR assembly helperを呼ばない。別のsecond-quantized行列経路でcandidate係数を評価 |
| H_XDF saved | 保存κ/λ/G/c → 既存DFHamiltonianと `df_linear_operator` → sector36本のbasis action | 元Hを共有referenceにし、再分解・再hermitize・ground-state solveを行わない |

低レベルのNumPy/SciPy等を共有しても、raw J/U density-productとadapter Z係数からの経路は分ける。raw reference入口はexportのraw配列であり、`model.reference_input()`を照合用metadataとして使う場合も、そこからcandidate K/Zを逆算してreferenceを作らない。C4の4-mode保存reference、square fixture、PASSはH4へ流用しない。generic CAR bit作用だけを再利用する。

OpenFermion v1.6.1の通常 `get_sparse_operator(InteractionOperator)` は内部のFermionOperator変換でsymbolic `+=` を使い、`_issmall(...,EQ_TOLERANCE)` による係数削除を含む。このshortcutをそのまま使わない。InteractionOperatorの非ゼロ係数を一意なterm mapへ無閾値で渡し、`get_sparse_operator(fermion_operator,n_qubits=8)` の既存JW sparse処理を使う薄い接続案とする。K側のbilinearもsymbolic加算による削除を避け、行列上でI/Z/ZZを積算する。数値linear algebraやJW algebraの再実装は不要だが、この係数保持glueと微小係数の人工確認は未実装である。[変換SOURCE](https://github.com/quantumlib/OpenFermion/blob/v1.6.1/src/openfermion/transforms/opconversions/conversions.py)、[symbolic加算SOURCE](https://github.com/quantumlib/OpenFermion/blob/v1.6.1/src/openfermion/ops/operators/symbolic_operator.py)、[sparse SOURCE](https://github.com/quantumlib/OpenFermion/blob/v1.6.1/src/openfermion/linalg/sparse_tools.py)

この設計はUの列が回転生成演算子を定める**代数的な規約**を検査する。実 `BogoliubovTransform(V.T)`、Givensのゲート順序、controlled/adjoint/位相は検査しない。それらのGaussian実回路資格はStage 2に残る。

## 7. 保存X-DFの静的な適格性と残る確認

P-H4生成SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12` の [h4_molecular_inputs.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h4_molecular_inputs.py) は1回のSCF由来h/g/cをPauliとDFの両経路へ渡す。保存は `df_one_body[8,8]`、`df_lambdas[10]`、`df_g_matrices[10,8,8]`。同じMO、interleaved spin、元nuclear cの来歴がある。生成時の `.hermitized()` 適用も履歴として維持する。

OpenFermion v1.6.1の [low_rank.py](https://github.com/quantumlib/OpenFermion/blob/v1.6.1/src/openfermion/circuits/low_rank.py) のnormal-order補正は、この保存spin tensor規約では `−0.5 C(g_original)` のspin liftに対応する。従ってκ保持政策と構造上整合し、**初回対照は保存rank10を再利用できる候補**である。第4戻り値truncation_valueはscalarとして足さない。

科学実行時はpacket hash、h/g/cとMOの来歴、λ/G shape、保存κと元gからのκの差、constant帰属を確認する。κの数値一致は未測定であり、生成時hermitizationも考慮した事前allowanceで判定する。保存の旧full-H/energy/sector PASSは新しい `ε_DF,XDF^sector` の値ではない。その新規norm callを契約に含める。政策・identity不一致ならX-DF比較を保留してSTOPし、fresh X-DFを無承認で作らない。

## 8. 数値allowanceの設計と確定前の事項

primaryはbinary64/complex128、unit roundoff `u=2^-53`、sector d=36、full D=256。CARの±1とbit作用は厳密に定義できるが、U/J/κの収縮、回転のunitarity誤差、差行列の形成で絶対roundoffが生じる。微小な差行列の相対条件数に依存した一律relative toleranceを使わず、元の項の大きさと演算経路からabsolute allowanceを作る。

提案する事前固定の形は `β_route(d)=γ_m_route(d) B_route(d)`、`γ_m=m u/(1−m u)`、`τ_eq(d)=β_raw(d)+β_K(d)+β_SVD(d)`。Bは相殺前の係数絶対和・中間行列absolute row sum等、mは薄いwrapperの演算graphから得るdot長・加算深さ・複素演算の伝播係数である。**観測したadapter残差からBやmを決めない**。入力に依存するBの算出規則と計上callもfit前に固定する。

例えばraw側の粗いscale候補は `|c|+2Σ|κ_pq|+2Σ_lij |J_lij| s_li^2 s_lj^2`、`s_li=Σ_p|U_lpi|`。K側は展開係数の絶対和と `||Z_a^V|| ≤ 1+2(Σ_p|V_pa|)^2` を使えば、unitarityを仮定せずscaleを作れる。これらはallowance全体の厳密認証を意味しない。Hermiticity、J対称性、U†U、Nα/Nβ保存も各scaleと演算経路に応じて別のabsolute/Frobenius gateを定める。

spectral normは差行列の最大singular valueで評価する案とし、`scipy.linalg.svd(...,compute_uv=False,lapack_driver="gesdd")` を使う。小さな非Hermitian成分を黙ってprojectして落とさない。eigvalshは入力のHermiticityを検査しないため、この用途で無条件に置き換えない。[SVD公式文書](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.svd.html)、[eigvalsh公式文書](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.eigvalsh.html)

LAPACKのbackward stabilityは入力scaleに対する絶対誤差の扱いを支持するが、実wrapperのmや安全係数を自動的には与えない。[LAPACK error analysis](https://netlib.org/lapack/lug/node97.html)に従い、行列形成とSVDの誤差を分ける。m、β_SVDの係数、構造診断gateの数値は**未確定の技術blocker**。別承認の最小実装と人工入力による経路確認で導出・固定してから、科学契約を再凍結する。旧C4の1e-10はコピーしない。実H4の結果を見て閾値やreference精度を動かさない。

DF近似誤差はこの表現同値性gateとは別の測定値である。測定のroundoff scaleも併記し、X-DF値が数値floor近傍なら比率や「C-DF≤X-DF」を必須判定にしない。物理誤差budgetを後に統括が採択するまでは、値の報告とStage 2適格性の保留を両立させる。

## 9. Stage 1の判定、残る採択、境界

| 判定 | 提案する条件と扱い |
|---|---|
| 技術的STOP | SOURCE/input/環境lock hash不一致、schema/補正/constant不整合、status取得不能・非有限値、独立reference欠落、未承認実行要求、call超過、共有環境への干渉 |
| 科学的STOP/保留 | 物理sector・規約の意味論不一致、未収束、U/V・Hermiticity/保存則・adapter絶対同値性FAIL、X-DF比較条件不成立。後に採択した物理誤差gateがあればそのFAILも含む |
| Stage 1表現GO | fit資格・構造gate・sector合計およびfull base/各frameの同値性gateが全てPASSし、CDF/XDFのsector DF誤差を別々に報告できた範囲 |
| Stage 2への適格性 | 上記GOとは別。物理誤差budget/比較目的を統括が採択して判断。C-DF≤X-DFは候補比較目標に留め、初回必須gateにしない |

失敗時はraw出力、partial evidence、原因、call ledger、STOPを保存して終了する。tol/rank/seed/optimizer/precisionの自動変更・再fit・restartなし。成功も資源優位、PR性能、新規性、end-to-end総費用の結論を意味しない。

統括・ユーザーが採択すべき事項は、P-H4/sector/full確認範囲、κ保持、保存X-DF、rank上限8/tol/solver、表現資格とDF誤差報告を分ける政策、allowance固定後の契約とcall規則である。Codexには別途、薄い接続・検証処理と独立環境の準備を承認する必要がある。**固定SOURCE/input/契約/call/STOPを指定する新たな明示承認があるまで科学実行しない。**

Stage 2の実Gaussian/Partial-S2/controlled/adjoint/X/Y/compile/native費用、Stage 3のD/R/PF/finite-RTE/sampling/shot/confidence/総資源は今回の契約外。共通policyへ埋め込まず、Stage 1結果確認後の別採択とする。本3成果物の公開・remote照合後はmandatory STOP。
