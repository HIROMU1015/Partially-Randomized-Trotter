# 研究C：固定U・tail演算子保存型DF係数再分解LPの技術準備

2026-11-01 JST。**技術準備完了。実H4のarray decode／LP／PF-RTE／shots／native／新fitは0。公開照合後mandatory STOP。**

## 1. 目的と引継ぎ

起点611a618674aba7146b0103cc34f156c9bc89bc16から、候補A＝固定U・固定head・固定tail合計tensorの下でexact Pauli tail λを最小化する係数再分解を準備した。従来の条件付き総資源評価の継続は今回の対象ではない。

ユーザーが指定した`research_c_df_algorithm_design_scientific_review_20261101.md`原本は、アクセス可能なworkspace／添付領域の検索で見つからなかった。読んだとは扱わず、今回の自己完結した指示文の数学仕様・研究方針を引き継いだ。共通運用指示2026-10-25版§8/10とAGENTS.mdを確認し、承認証明・承認原文保存・新grant・重要GPT科学レビューは作成／実施していない。

LP／L1／uniform shiftそのものの新規性は主張しない。実用上の問いは、現在のPR-tailでPF／補正K0 RTEを保持しながらλを下げられるか、という限定問題である。候補Bのouter-U更新／variable projection／joint最適化は実装していない。

## 2. 固定対象・SOURCE

| 対象 | Identity／条件 |
| --- | --- |
| 実装commit | 22aedb27d70f93ca43687bb9120f7598dc0d4e9f |
| SOURCE105 SHA256 | 8a9dbedd1a97702ff3a76a3939d98ee3d4447d8b0f7bf548e7ca752e692fdcfd |
| 未承認契約SHA256 | bdee0d3c656e580fadbd36e4d54ae83c0c7671666ffbba21d3a9987aef8bdcff |
| 元PR-tail fit SOURCE | e2f7dfd44357510d54ab9c6c9c5032e0072ab376 |
| 元PR-tail export SHA256 | 3fdb0cb77afc770294cbe7aebf47d8a4529e3567ef9e886895495f4490b7ee21 |
| 元h/g/scalar snapshot SHA256 | 1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db |
| 保存reference SOURCE | d6b17b79461a39bd8ba2e6b7389a1f4fad4d74e6 |
| 物理条件 | 保存直鎖H4／隣接1.00 Å／STO-3G、Nα=Nβ=2／sector36 |
| 表現 | 通常number CDF、一般実対称J、実直交U、8 frameの元順 |
| 分割 | head frame0/1、tail frame2..7、L_D=2、元g補正κ／nuclear scalar一回 |
| PF/RTE | partial ordered symmetric S2、h=0.1／s=1／r=128／K=0 |
| DF予算 | CA/8=0.00019875 Ha、CA=0.00159 Ha |

保存exact λ=**0.9545432846581517 Ha**はユーザー指示と保存metadataの引用・identity照合であり、今回の再計算値ではない。元DF予算は保存PASS、元optimizerの共通収束資格はUNQUALIFIEDのまま。旧PR-tail、旧RC2、γ=0、元optimizer再開とは区別する。

新規は[線形作用／LP](../../src/trottertracks/c_fixed_u_tail_lp.py)、[科学接続runner](../../src/trottertracks/c_fixed_u_tail_lp_runner.py)、[CLI](../../scripts/run_research_c_fixed_u_tail_lp.py)、[人工テスト](../../tests/management/test_research_c_fixed_u_tail_lp.py)。[SOURCE105](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/technical/SOURCE.json)は既存依存集合と新4ファイルを固定し、旧libraryへの編集は0。

## 3. 線形作用とLP

Tailの対称Jをffsim互換のframe-major上三角順にpackする。変数jは6×10=60個。Unpack時はoff-diagonalを対称位置の両方へ置き、

\[
 A_{U,R}j=\sum_{tij}U^t_{pi}U^t_{qi}J^t_{ij}U^t_{rj}U^t_{sj}
\]

を作る。明示Aのoff-diagonal列は`P_i⊗P_j + P_j⊗P_i`であり、片方だけにしない。全256 tensor entryを制約に使う。`apply`は既存`c1_h6_initialization.reconstruct`、随伴は同じcontractionの転置で、packed off-diagonalには両寄与を足す。作用・随伴は明示Aの構築と分離した。初回H4では256×60の小行列を使い、大規模matrix-free solverを新設しない。

Tは既存`c_pr_aware_df.coefficients`から構築する。K_ab=J_(a//2,b//2)/2、Z_a=−row_sum(K)/2、ZZ_ab=K_ab/2という既存spin規約を保持する。216個の非identity係数に対する**exact L1**で、entrywise Σ|J|やsmooth L1へ置換しない。微小な非ゼロを閾値で剪定せず、非ゼロJが全ゼロ係数へunderflowした場合は明示的に拒否する。

\[
 \min_{j,u}\mathbf1^Tu,\qquad Aj=b=A j_0,\quad Tj-u\le0,\quad -Tj-u\le0,\quad j\in\mathbb R^{60},\ u\ge0.
\]

合計276変数、256等式、432不等式。元j0は理想代数上のfeasible pointであり、実装では既存contractionで作ったbとのfull-row残差も確認する。jの符号をSciPy既定の非負変数へ誤って制限しない。PSD／rank-one／frame順変更／headへの移送／rank探索は0。係数0になるframeがあっても8 frameをその位置に保持する。

SolverはSciPy1.14.1同梱のHiGHS dual simplex、`method='highs-ds'`、presolve=True、primal／dual feasibility tolerance各1e-10、**1回**。新依存・独自LP solverは不要だった。maxiter／time_limit／worker上限は追加しない。[SciPy公式仕様](https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.linprog-highs-ds.html)を確認した。

## 4. Dual・conditioning・判定

\[
 \max_{y,z}b^Ty,\qquad A^Ty=T^Tz,\quad\|z\|_\infty\le1.
\]

SciPyのeqlin marginalをyとし、`Tj-u<=0`のmarginalをα、`-Tj-u<=0`をβとすると、**z=β−α**。自由jのstationarityとuのdual slack `1+α+β>=0`をこの符号で確認する。原marginals／residual／x／slack／conはNPZへ保存し、unbounded変数のinf residualを不正なJSON値にしない。診断のy/zは有限のJSON listとしても保存する。

独立にprimal equality、epigraph／u非負、dual equality／box／inequality sign／u-dual、complementarity、primal値、dual値、gap、solver funとの一致、元候補feasibility、λ差を確認する。Solver successだけでは採用しない。

数値条件は[固定契約](research_c_fixed_u_tail_preserving_lp_contract_unapproved_20261101.json)に全係数を記載した。主な条件：

- full-row残差≤1e-10＋1e-10 maxabs(b)、独立tensor保持gate≤1e-10。既存CAR／PFと同じbinary64数値pilotの成分精度を使う。
- primal／dual残差≤1e-9。Objective関連残差・gapは1e-9＋1e-9×objective scale以内。Solver指定精度に一桁の評価余裕を持たせ、物理的保持gateは別に残す。
- `dual_dot_scale=Σ|b_i y_i|`と、`g_obj=8 eps64 (2 rows+2 j+u) max(1,|primal|+λ+dual_dot_scale)`を保存し、g_objもobjective allowance以内とする。大きな相殺を含むdual dotを小さいgapだけで資格付けしない。
- Strict改善のpilot判定は`λold−λnew > 1e-8 +1e-8 max(λold,λnew)+10(|gap|+g_obj)`。LP gap／丸めguardと区別できない差を成功にしない。

全singular value、`eps64 max(A.shape) σmax`による診断rank／near-null数、retained conditionを記録する。Retained condition>1e12は未資格。**SVDは診断のみ**で、制約削除やtruncated-nullspace探索には使わない。記録の`equality_rows_dropped=False`はadapter側の行削除なしを指し、HiGHS内部presolveの簡約なしを意味しない。保存Uをpolar／sign gauge／logmで変更せず、近似null方向をexactとは扱わない。

これらは事前固定したengineering判定で、丸めを囲ったprimal feasibility／dual lower bound／厳密最適性certificateではない。Full-row KKTが数値的に良好でも、固定binary64 Uのexact零空間を証明したことにはならない。

## 5. 保存する構造と不変性

全Uとhead Jは元bytesのまま出力する。κは元gから同じ補正、scalarは同じ一回、元frame順も維持する。新exportはJ/Uのみとし、古いoptimizer x/jacを新候補の値として捏造しない。

Identityは既存一般K規約で各frame `ζ_t=Σ_ij J_ij/2+tr(J)/4`。正確な直交Uではtail tensorの二つのtraceからΣJとtr(J)が保存されるので、ζも保存される。実装ではその理想関係だけに依存せず、identity係数・full256 tail CAR・sector36 raw／一般Kを独立に確認する。

Head、tail合計H_R、ζが保持されれば、partial PF `exp(-ihζ) R_h exp(-ihH_R^0) L_h`は理想的に不変。K0では`τR=(h/r)H_R^0`なので、normalization補正済みRTEも不変になる。既存reference_kernels／finite_rte_corrected_operatorを再利用し、新PF/RTE simulatorは作らない。Zero λでは既存deterministic-only経路を使い、H_R^0/λを評価しない。

λ低下時のideal K0 attenuation `A_m=[1+(λh/r)^2]^(-r2^m/2)`の非減少と、既存RPE/Hoeffding条件付きshotsの非増加を人工例で確認した。実H4ではλをまだ計算していないため、A／shots改善が起こるとは言わない。一般LPで各fragmentが変わるので、**全frame決定論S2の不変性は主張しない**。同じtail tensorでも全frame S2が変わる人工例を確認した。

Uniform shift `J_t+a_t11^T, Σa_t=0`は、正確な直交Uで主LPに含まれる安価なreference family。既存Pauli mapと独立な`Σ_i|s_i+na|+¼Σ_i|J_ii+a|+Σ_i<j|J_ij+a|`を人工検証し、主LP値がこのfeasible referenceより悪くないことを確認した。各frame独立最適化でゼロ和を仮定せず、headへ移送せず、実分子用の追加uniform-shift LP系列は作らない。

## 6. 再利用・人工検証

| 既存資産 | 適用／新規との差分 |
| --- | --- |
| C1 `reconstruct`／ffsim v0.0.83 real symmetric pack | Contractionはそのまま再利用。J-only上三角layoutは実ffsim関数と人工配列で完全一致を確認。U更新を起こすrotation codecは呼ぶ必要がなく、再実装もしない |
| PR coefficients／general-K adapter | T、identity、既存plan／lawのmass整合。旧smooth目的やfitterは変更0 |
| SectorCAR／full raw CAR | 新6 frameのgeneral-K比較、旧新full256 tail合計、保存sector mappingへ接続 |
| Stage B／C1 reference・RTE | 原g補正κ、head／phase、保存PF/corrected kernel照合、zero λ経路を再利用 |
| C1 accuracy／Hoeffding | 条件付き単調性の人工検証のみ。新しいshots式やRPE estimatorは0 |
| SciPy linprog／Calls／atomic JSON／resource_plan | 既存solverと実行保全の最小接続。承認管理や独自solverを作らない |

[主試験log](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/technical/pytest.txt)：新17＋既存63＝**80 PASS**（10.05秒）。[一般Kの独立回帰](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/technical/general_k_regression.txt)：**13 PASS＋29 subtests**。合計93 PASS＋29 subtests。

人工検証は作用／随伴、off-diagonal二重計数、I/Z/ZZ展開、det±1、既知の正負最適解、strict改善／改善なし、dual符号／gap、uniform shift包含、重複U／rank-deficient map、相殺の大きいdual、微小非ゼロ／underflow拒否、非直交U拒否、U無補正、部分PF／K0 RTE、全frame S2の非不変性、zero λ、条件付きshots単調性を含む。End-to-endでは50-call正常系、solver未収束の9-call未解決分岐、保持FAIL時STOP、fresh/no-resume、元入力保全を確認した。Actual ffsimは人工Jのpack/unpackのみで、optimizer iterationは0。

[実保存入力passive preflight](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/technical/passive_preflight.json)は、array load／FixedU／LP／PF/RTE等を呼出し禁止にした状態でPASS。SOURCE105／入力9／lock／NPY header／保存λ metadataを照合し、numpy/scipy/qiskit/ffsim/jaxのimportすら0。実H4 decode／科学callは0、fresh run01は未作成。

## 7. 未承認50-call最小実験

第一対象は指定された保存PR-tailだけ。RC2への適用は、この固定候補に追加自由度が残るかという最初の問いの前提ではない。後でLP後の競争的優位を主張するならRC2にも同じ後処理を与える公平性を別arm／別契約で検討する。今回はRC2 LP=0。

| Logical science call群 | 最大数 |
| --- | ---: |
| Evidence／input／export／保存reference／固定入力gate | 5 |
| LP構築／solver 1回／primal-dual診断 | 3 |
| Candidate export／独立tensor保持 | 2 |
| 旧新model／law／coefficient map各2 | 6 |
| SectorCAR context／新raw6／一般K gate6 | 13 |
| Full CAR context／旧新full frame12／tail合計gate | 14 |
| 固定head・identity／DF保持 | 2 |
| 新PF/RTE kernel1／保存kernel照合1 | 2 |
| Ideal attenuation13round profile | 1 |
| Candidate資格／最終分類 | 2 |
| **合計** | **50** |

元g／物理Hamiltonian・旧DF normを再計算せず、保存guard-inclusive DF値へ新旧sector operator差の2-normと事前定義guardを加えたnumerical triangle boundで予算判定する。新guardは`8 eps64×2(n^4+rank n^3)(d+1) max(1,Σ||G_old||F+Σ||G_new||F)`。旧guardを変更せず、これは追加の丸め証明ではない。保持gate前に旧DF判定を新候補へ単純転記しない。

PF／補正RTEの新数値gateは**1 h-step**を保存旧kernelへ比較する範囲。全13roundの新PF/RTE誤差・ground／energy／CAは計算・転記しない。Attenuation13行はideal K0の診断のみで、shots／W／native／総費用は0。

Fresh outputは`artifacts/research_c_fixed_u_tail_lp/2026-11-01/run01`。新candidateは別export SHA256と親SOURCE/inputを持つ係数後処理で、元optimizerの続きではない。Original/U/head/identity/tensor/operator/PF/RTE/DF-budget mismatchは最初のFAILで依存callを止め、開始済みfull-CAR taskを回収してpartial evidenceを保存する。Solver／KKT／conditioning未資格はUNRESOLVEDで後続採用なし。Retry/resume=0、追加LP／fallback／別rank／別U／RC2追加は0。

[環境・資源計画](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/technical/environment.json)：SciPy1.14.1など既存evaluation環境をread-only利用し、新依存はない。256×60のA、疎T／epigraphはH4で小規模。Full CAR12因子は実空きCPU/RAMから既存resource_plan(12,128)で並列化し、並列時BLASの過剰多重化だけをprocess-localに調整する。CPU/RAM/worker/wall人工上限・共有環境変更は0。

## 8. Q1〜Q10と次の判断

| 問い | 回答 |
| --- | --- |
| Q1 正しい固定U LPか | 指定の全tensor等式とexact Pauli L1 epigraphを実装し、作用／随伴／独立展開／既知解で人工確認。実H4での成功は未評価 |
| Q2 元候補はfeasibleか | 理想上は定義からfeasible。実行時に独立contractionとA j0の残差を確認し、未確認のまま採用しない |
| Q3 exact L1か | 既存spin K→Z/ZZのTを使用。Identity除外、row sumを保持しΣ|J|／smoothingへ置換しない |
| Q4 primal/dual/gapは | SciPy marginalsからy,zを取り、full-row primal／dual／box／補完性／gap／相殺guard／conditioningを独立診断。厳密certificateは未取得 |
| Q5 head/tail/phaseは | U／headをbytes固定し、独立tensor、full256 CAR、sector36一般K、identityを別gateで検証する構造 |
| Q6 PF/RTEは確認したか | 人工例でpartial PF／補正K0 RTE／zero λを確認。全frame決定論S2への誤った一般化も反例で排除した |
| Q7 再利用範囲は | 既存contraction・Pauli map・CAR・model/law・PF/RTE・accuracy・solver・出力/STOP。新規は線形作用、LP診断と研究固有dispatchのみ |
| Q8 最小新計算は | 保存PR1候補、HiGHS-ds 1回と保持gateの最大50 units。実行には固定SOURCE/契約への別明示許可が必要 |
| Q9 λが減らない時は | Numerical optimality＋全保持PASSなら限定問題の改善余地が小さいというpilot。未収束／KKT不良／conditioning不良なら判断保留で、改善不能とは言わない |
| Q10 Bへ進む条件は | 数値最適性と保持を資格付けしても改善余地が乏しい時に研究CがBの価値を判断。未解決なら先に数値問題を切り分け、strict改善ならAの下流価値を先に判断。自動移行しない |

LP改善があっても保持FAILなら候補Aの成功ではない。数値的に明確な改善と保持PASSが得られた時だけ、条件付きshots／新candidate固有native価格を取得する価値を次に判断する。方法上の独立新規性／汎化／実CA／総資源優位は別問題のまま。

[保全postflight](../../artifacts/research_c_fixed_u_tail_lp/2026-11-01/technical/postflight.json)で旧native729／shots92／Stage B69／共通fit37／JSON修復7、旧SOURCE／contract／STOPの不変を照合する。Commit/push後にremote SHA/tree・公開本文・SOURCEを確認する。

**次の担当：ユーザー／研究Cが未承認50-call契約の実H4実行を明示許可するか判断する。Codexは許可後のみfresh runを一度実行する。今回の公開照合後mandatory STOPし、実H4 LP・追加fit・PF/RTE・shots・native・B実装・重要レビューへ自動進行しない。**
