# 研究C：PR-aware DF最適化の初期設計・技術検証

対象repository：HIROMU1015/Partially-Randomized-Trotter。調査起点：`54ceff9981d9bc53210db97c743a7d1cf694f362`。本書は依頼の20261026識別子を使用する。

**推奨は、通常CDFのテンソル目的に、固定PR tailの実際の非identity I/Z/ZZ係数massを滑らかに抑える項を追加し、物理sector誤差は独立に判定する方式である。** 実ffsim fitterへの人工接続を確認した。科学的有効性・新規性の採択はしていない。実分子decode、新fit、実optimizer iteration、保存結果の新規再解析、PF/RTE/shot/cost評価、回路build/compileはすべて0。

初回は[未承認Stage A契約案](research_c_pr_aware_df_pilot_contract_draft_20261026.json)の**最大30 logical units、同一初期値からの2 fit**だけを提案する。Stage B/Cは契約外。科学実行には対象・SOURCE・条件に対する別途の明示許可が必要で、証明書・承認原文保存・新grantは要求しない。正式運用指示2026-10-25版§8.1/8.4/10/10.3に従う。

## 1. 既存の結果と未実施の範囲

- [強baselineの保存報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/docs/research/research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md)：直鎖H4・隣接1Å・STO-3G、sector36、CDF8 B2対XDF8 B3、全frame ordered S2 s1、δ=.1/m0..12、PRなし、own-DF ground oracle・状態準備除外、共通precision/P0/serial。条件付き総RZはCDF 102,056,444,569、XDF 100,267,421,030、XDFが約1.753%少ない。この保存scalarの引用であり再集計していない。基底構築と相対Gaussian融合を無視したbaselineに戻さない。決定論の順位をPRへ移さない。
- [過去PR費用報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5d7faf7121cc281b7e2f38c2e04c224e594ae53d/docs/research/research_c_c1_stage3_cost_compat_execution_20261025.md)：保存H4 CDF8/XDF10・登録8候補の限定native校正／楽観的shotsでは、kernel削減がshots増を補えない例がある。全law費用supportは未資格。この事実は回路単価だけの最適化への注意であり、PR一般のNO-GOではない。
- [H6修正fit保存報告](research_c_c1_h6_corrected_fit_execution_20261025.md)：直鎖H6・1Å・STO-3G、sector400、rank14、修正codec、L-BFGS-Bでテンソル残差が改善してもsector誤差は初期値より約4.31%増加した。DF予算は保持した。テンソル目的と物理精度の一致を仮定しない。
- 今回以前に確認したのは、通常CDF、codec修正、決定論の強いnative baseline、固定DF下のPR law・誤差・限定費用接続である。今回の代理目的による実分子の最適化、同精度PR総資源削減、独立新規性は未実施／未確立。

上記2件の指定GitHub報告はconnectorで取得しlocal blobとも照合した。過去の実験SOURCE・raw・grant・ledger・marker・STOPとH2/H4/H6の結果は変更していない。

## 2. 先行研究の確認範囲

[RCDF, arXiv:2212.07957v3](https://arxiv.org/html/2212.07957v3)の目的式・regularization節・λ式を確認した。一般対称JのCDFに係数正則化を加え、用途に応じたnormを小さくする考えは既知である。測定・qubitization向けの目的と、PRの減衰・誤差・shots加重native費用は一致しない。係数penalty追加そのものを新規性と呼ばない。

[SCDF, arXiv:2403.03502v1](https://arxiv.org/html/2403.03502v1)の§2.2/2.4・目的式・付録Aの関連箇所を確認した。対称性shift、構造化分解、qubitization向け費用最適化との違いを確認した。今回はrank-one制約やsymmetry shiftを導入せず、通常CDFの一般J・元κ・scalarを維持する。

[PR論文, arXiv:2503.05647v2](https://arxiv.org/pdf/2503.05647v2)のAppendix A（paired Taylor weights）、VII.B（因子化PauliとGaussian）、C.3（軌道基底・norm最適化）を確認した。表現・軌道基底・λを改善する方向自体も既存研究と重なる。今回の問いは、既存general-J CDFを固定prefixの実PR atom係数へ結び、物理sector gateと強baselineを通したshots加重費用で改善仮説を反証できるかである。これを独立した新規性と認定していない。

確認はこれらの関連節・式に限定した技術調査で、全論文の精読、網羅的な新規性レビューではない。

## 3. SOURCE・関数・テストの再利用

| 既存資産 | 確認内容と今回の扱い |
|---|---|
| ffsim 0.0.83 `double_factorized_decomposition.py` / `util.py` | 固定2 source hashを照合。既存tensor loss、JAX value/gradient、pack/unpack、SciPy L-BFGS-Bを再利用。Hamiltonian CDFには任意lossのpublic callbackがないためprocess-localなfactory接続だけ追加。t2用regularizationは別問題なので代用しない |
| `c1_ffsim_corrected_codec.py` / `c1_real_orthogonal_codec.py` | 保存J/Uを初期値として渡す実ffsim route、real Schur＋列符号gauge、qualified x0照合、finally復元をそのまま再利用。det=-1/π branchの旧codecへfallbackしない |
| `c1_stage1.py` / `c1_stage1_execution.py` | 元g補正κ、input whitelist、byte/JSON I/O。旧one-fit runnerやgrant経路は起動しない |
| `c1_h6_initialization.py` | 汎用のtensor再構成・tensor metricsだけ呼ぶ。H6診断・旧codec・optimizer entryは起動しない |
| `c1_molecule_extension.py` | 既存SectorCAR・raw density・general-K経路、`rank_assessment`・`representation_gate`をH4 sector36へ使用。旧rank ladderは不使用 |
| `c1_general_k_adapter.py` | number表現、interleaved spin、一般対称J→K→I/Z/ZZ。非PSD Jも許す。rank-one XDFへの変換・微小係数thresholdはしない |
| `c1_stage3_connection.py` | original-frame-prefix split、identity抽出、FRAME_Z atom、有限RTE再構成、occurrence独立trajectory。人工データでtail mass・正規化・attenuationを照合 |
| `rte.py` / `c1_stage3_reference.py` | paired-order weight・有限Taylor law・補正RTE／PF独立reference。Stage B/Cの再利用先で、今回実分子評価しない |
| `df_partial_randomized_pf.py` | 旧weight-ranked splitを確認。今回の「元frame順prefix L_D=2」と同一視せず、weight順に並べ替える処理は呼ばない |
| `pr_dictionary_validation/prototype.py` / `pricing.py` | 係数・exact mass・finite-bit support、literal basis遷移の期待費用DP。新θでlaw identityを再構成する必要がある。literal recipe価格をP0全回路価格とは呼ばない |
| `c1_stage3_qualification.py` / `cost_compat.py` | 全support資格と校正点推定の区別を維持。標本最大値を上界にしない |
| `c1_stage3_accuracy.py` / `rpe_resource_accounting.py` | 振幅／位相room→Hoeffding shots、DFとenergy条件、未資格nullの伝播を再利用予定。新しいshot式は実装しない |
| `rpe_short_round_optimization.py` | 固定DFの短round r/K最適化でありDF最適化ではない。既存探索gridを流用して候補拡張しない |
| NativeBuilder / `c1_h4_strong_baseline.py` / H6 precision/P0 | scalar、actual adjoint、Gaussian方向、既知のspin対応基底・相対融合・精度修正を公平baselineとして維持。Stage Aではbuild/compileしない |
| 既存corrected-codec・general-K tests | 下記28 tests＋29 subtestsを無変更再利用 |

新実装は2 module・1 runnerと人工testだけで、optimizer、codec、CAR、Gaussian合成器、sampling法、shot推定器を作り直していない。

## 4. パラメータ・物理規約

実軌道に限定し、`θ={J_l,U_l}`、Jは一般実対称、Uは実直交。ffsimの`U=exp(A)`、A実反対称を用い、det=-1はdensity projectorを保持する列符号gaugeで処理する。frame順・rankを固定し、frameの選別・sortは目的関数内で行わない。H4 rank8/norb4では128変数（回転48＋J上三角80）。PSD・rank-one square制約は加えない。

ffsimの再構成と元のlossは

\[
g^\theta_{pqrs}=\sum_{lij}U_{lpi}U_{lqi}J_{lij}U_{lrj}U_{lsj},\qquad
f(\theta)=\tfrac12\|g-g^\theta\|_F^2.
\]

元gを使う`κ_pq=h_pq−(1/2)Σ_r g_prrq`を固定する。新gθからκを作り直さない。核scalarを一回保持し、number fragmentは`D_l=(1/2)Σ_ijστ J_lij n_iσ n_jτ`。spin modeは`a=2i+σ`。

\[
K_{ab}=J_{a//2,b//2}/2,\quad
c_{Z_a}=-\tfrac12\sum_bK_{ab},\quad
c_{Z_aZ_b}=K_{ab}/2\ (a<b),\quad
c_I=(\sum_{ab}K_{ab}+\operatorname{tr}K)/4.
\]

この線形係数写像のみJAX用に記述し、独立した既存general-Kの係数と人工照合した。identityは決定論の正確な位相へ抽出し、λ_Rには含めない。J変更に伴うidentityの値自体は変わるので凍結しない。異なるUの係数を根拠なく相殺しない。

## 5. DFから最終資源への依存関係

\[
\min_{\theta,\pi} C_{PR}(\theta,\pi),\qquad
C_{PR}=\sum_{m,a\in\{X,Y\}}N_{ma}(\theta,\pi)
\mathbb E_{\tau\sim p_\theta}[C_{P0}(\theta,\pi,\tau,m,a)].
\]

共通精度・失敗率・oracle仮定・状態準備除外の下で比較する条件付きinterrogation費用である。初回はπを固定しθだけを変える。未適格候補を費用順位へ混ぜない。

| θによる変化 | 伝播・検証すべき量 |
|---|---|
| gθ / J / U | tensor残差と物理sector誤差は別。Uは非可換性・PF誤差・Gaussian価格へ影響 |
| 非identity tail係数 | λ_R=Σ|c|、atom確率・符号・order重み・basis identityを再生成。固定J・固定splitなら係数massはUに依存しない |
| λ_R | RTEのnormalizationとattenuationへ。K>0ではTaylor orderの分布と期待event数も変わる |
| PF/RTE誤差 | commutator・有限次数・位相roomに依存し、λだけでは決まらない |
| native価格 | UのGaussian遷移、決定論head、trajectory全体、合法融合・compilerへ依存。単一event価格の和でP0費用を代用しない |
| finite-bit law | θ変更ごとにprobability/supportを再確認。正係数のunderflowを黙って削らない。ideal lawとの距離は別資格 |
| 信号room | attenuationと全誤差から正半径・位相roomを確認し、その後shotsを算出 |
| 最終費用 | shots×期待native費用。合成誤差・law差・費用supportが未資格なら保証付き優位は未資格 |

既存paired RTEではh=δ/s、τ=λ_R h/rとして

\[
w_{2k}=\frac{|\tau|^{2k}}{(2k)!}\sqrt{1+(\tau/(2k+1))^2},\quad
B_K=\sum_{k=0}^{K/2}w_{2k},\quad A_m=B_K^{-r n_m},\ n_m=s2^m.
\]

ここでKは既存APIの偶数Taylor cutoff（例0/2）で、sumは偶数order 0..K。K=0なら`A_m=(1+τ²)^(−r n_m/2)`で、固定πのλ低下はAを改善する。この代数関係を人工データで既存実装と確認した。K=0でもrandom atom eventが無くなるわけではない。λ=0のexact tailは既存lawの別分岐で扱い、ε付きmassからzero tailを認定しない。

既存精度接続は、無次元operator/signal誤差を`d=A(εPF+εRTE)+εsynth+εlaw`、`ρ=A−d`、`βdet=asin(d/A)`とし、正半径・位相条件を要求する。統計roomに対する座標許容値`e=ρ sin(βstat)/√2`から、軸ごとの十分shots処方`N=ceil[2 log(2/αaxis)/e²]`を使用する。誤差の資格・fixed-N条件が前提で、情報理論的下界ではない。DFのHa誤差はenergy側で別に接続し、このdへ直接加算しない。未知の合成／law誤差を0と認定しない。人工反例ではattenuationが改善してもPF誤差増でshotsが増え、λ目的だけの限界を確認した。

## 6. 第一候補：固定prefixのsmooth tail mass

\[
M_\epsilon(J)=\sum_{l\ge L_D,s\ne I}\sqrt{c_{ls}(J)^2+\epsilon^2},\quad
F_\gamma=f/f_*+\gamma(M_\epsilon/M_*)^2,
\quad f_*=\max(f(\theta_0),10^{-14}{\rm Ha}^2),\ M_*=M_\epsilon(J_0).
\]

- 変数はJ/U全部。κ・scalar・rank・frame順・L_D固定。初期共通scaleを一回計算しbaseline γ=0、提案γ=.1で同じ値を使う。
- ε=1e−8 Haを提案する。`sqrt(c²+ε²)`からεを引かず、常にexact mass以上。`0≤Mε−λ_R≤Tε`。H4 rank8/LD2ならT=6×(8+28)=216なのでgap≤2.16e−6 Ha（構造からの計算で、実H4値ではない）。出力にはexact massも必ず残す。
- 元tensor objective/gradientはffsimを再利用し、penaltyだけ既存parameter unpackとJAX ADで加える。係数生成はO(R n²)、新しいexpensive operator／shotsをoptimizer内で呼ばない。tensor contractionとJAX expm/gradientが主要負荷。全体はUも動く非凸問題で大域最適性を要求しない。
- 物理制約は`||Hθ−Hphysical||sector,2 + guard ≤ CA/8`。実装はこの制約を内包するsolverではなく、初期・終了の独立sector判定による棄却方式。f低下から予算PASSを推定しない。最適化途中の予算内保持も保証しない。
- 改善仮説：固定PR政策のλ低下により減衰負担が小さくなる。Uの変化によるPF誤差増、deterministic cost増、shotsのceil境界、basis遷移価格変化で最終総費用は悪化し得る。
- 最小反証条件：両fitが収束し物理予算を満たした上で、exact λ_Rのbaseline比削減が1%以上か。未達ならこの登録penalty/初期値でのStage A仮説は支持されない。通過してもPR費用の支持ではない。

この案は、事前の実回路価格calibrationが要らず、実際のgeneral-K tail係数とpaired-RTE式に対応し、既存fitterへの差分が小さいので第一候補とする。sector normの直接AD、離散RZ count、全RPE shotsをそのままJAX lossへ組み込む案は初回に適さない。

## 7. 第二候補：固定U・凍結価格weighted mass

第一候補が物理誤差／非凸探索／native価格との不整合で不適格な場合だけ検討する代替案で、未実装・未承認である。

\[
\min_{J=J^T} f(J,U_0)/f_*+
\gamma\left[\frac{\sum_{l,s\ne I}w_{ls}\sqrt{c_{ls}(J)^2+\epsilon^2}}{W_*}\right]^2.
\]

U、rank、frame順とPR政策を固定し、事前に両arm共通のnative/literal recipeで得た非負event価格wを凍結する。W*は同一初期値のweighted mass。価格不足を0や一律1で埋めない。価格取得には別の限定科学許可・契約が必要で、今回の30-callには含めない。

固定Uではgθと係数がJに線形で、tensor二乗誤差と非負smooth weighted mass二乗は凸になる。既存packのJ変数と既存optimizer/ADを使い、別solverは作らない。追加計算はO(T)だが、価格calibrationとそのidentity管理を要する。UのGaussian再合成を抑えたまま安価なtailへmassを移す仮説である。

ただし凍結event価格はtrajectoryの境界融合・P0全回路・sampling分布変化・shotsを表さず、λ増加による減衰悪化もあり得る。最小pilotでは共通物理予算を守った上でweighted massとexact λを両方比較し、独立に得た期待native価格とattenuationで仮説を反証する。新γ探索やStage A失敗時の自動fallbackは行わない。RCDFのweighted regularizationと重なるため、この案もpenalty自体の新規性は未確立。

## 8. 実装と人工検証

SOURCEは `src/trottertracks/c_pr_aware_df.py`（係数・目的／AD・process-local loss factory）と`c_pr_aware_df_pilot.py`（既存のfit/sector処理へのStage A接続）、入口は`scripts/run_research_c_pr_aware_df.py`。実装commitは`1deb62f8b`、全科学SOURCE bytesは[manifest](../../artifacts/research_c_pr_aware_df/2026-10-26/technical/SOURCE.json)で固定する。

既存ffsimの初期化routeとSciPy minimizeを呼ぶが、実行process内だけ`_make_compressed_value_and_grad`を置き換える。γ=0はnormalized tensor loss、γ>0はpenalty ADの加算。factory・codec・minimize bindingはfinally復元し、同一process内で並列fitしない。optimizer直前のqualified x0、元目的／penalty／combined objectiveとgradient、終了x/jac・停止条件を保存できる。新旧J対称性・有限値・U直交性を検査する。

| 試験process | 内容 | 結果 |
|---|---|---:|
| fit専用env | 新目的9＋実worker2＋既存codec15。general-K符号、γ0一致、finite-difference gradient、det±/π、factory復元、実ffsimのx0・mock minimize | 26 PASS |
| evaluation env | 人工30-call end-to-end、初期予算/表現STOP、optimizer不成功/終了予算未達/改善なし、未評価null、fresh拒否、byte不一致、mass→既存RTE、shots反例 | 10 PASS |
| evaluation env・別process | 既存general-K独立CAR回帰（旧testのpassive import条件を維持） | 13 PASS＋29 subtests |

**計49 unique tests＋29 subtests PASS。** 実ffsim接続は人工配列だけでoptimizerをmockしnit=0。既存sourceを変えず、H2/H4のgeneral-K・codec共有部の非回帰を確認した。H2/H4の実結果を再評価した意味ではない。

初回の人工テストで、float等値比較の1.7e−18程度の差、subprocess mockによるgit probe干渉、同一pytest processのoptional import干渉を検出し、人工testの比較／mock境界／process分離を修正した。科学数値allowanceは変更していない。最終logは[technical](../../artifacts/research_c_pr_aware_df/2026-10-26/technical/)に保存。CI・外部再現の証拠ではない。

再検証時は各環境のPythonを明示し、下記の3試験を別processで実行する。`FIT_PY`はffsim/JAX環境、`EVAL_PY`はOpenFermion/Qiskit1.3環境。`TMPDIR`／MPL/Numba/JAX cacheはtask-localへ設定する。

```bash
"$FIT_PY" -B -m pytest -q --tb=short -p no:cacheprovider tests/management/test_research_c_pr_aware_df.py tests/management/test_research_c_pr_aware_df_workers.py tests/management/test_research_c_c1_ffsim_corrected_codec.py
"$EVAL_PY" -B -m pytest -q --tb=short -p no:cacheprovider tests/management/test_research_c_pr_aware_df_pilot.py
"$EVAL_PY" -B -m pytest -q --tb=short -p no:cacheprovider tests/management/test_research_c_c1_general_k_adapter.py
```

passive検査は同runnerへ`--contract <draft> --input <保存input>`だけを指定し、`--execute`／`--worker`を付けない。fit環境の版照合はmoduleの`preflight(..., fit=True)`で行う。今回の両PASSを保存したが、将来の科学実行時にも再照合する。

## 9. 推奨pilot：Stage Aだけ

| 条件 | 未採択の推奨案 |
|---|---|
| 入力 | 保存P-H4・直鎖1Å/STO-3G、SHA256 `8a9c4542…`、Nα=Nβ2/sector36 |
| 両fit初期値 | 保存rank8 CDF export `c33f782d…`のJ/Uを同一に使用、faithful codec後にtensor／sector確認。新XDF初期分解なし |
| 変数／条件 | 通常number CDF J/U、rank8、元順prefix LD2、ε1e−8、γ0/.1、元g補正κ・核scalar一回 |
| optimizer | ffsim0.0.83＋既存L-BFGS-B、options=None、同じ初期scale・修正codec、各arm1 fit、最大2 fit |
| tol | 1e−8はffsim initializerのtruncation tolでありSciPy optimizer tolではない。今回は保存初期分解を使いtruncateしない |
| 精度 | CA=.00159 Ha、DF reserve=.00019875 Ha。既存sector operation-scale guard8はengineering pilotで厳密enclosureではない |
| 新しい条件の意味 | γ0でもlossを正規化するためSciPyの絶対停止scaleが旧生loss fitと変わり得る。既報H4通常CDFの再現ではなく、対にした新baseline fitとして比較する |
| 成功条件 | 初期表現・予算PASS、両optimizer success、両終了DF予算PASS、提案exact λが新baseline比1%以上小さい |
| 非成功 | correctness不一致は依存処理STOP。有限のoptimizer不成功／終了予算未達は診断後`STAGE_A_NOT_MET_OR_UNRESOLVED`。λbaseline=0は削減率未定義。初期値fallback・retry・rank変更なし |

callはprepare6＋baseline fit5＋proposed fit5＋sector/比較14＝**30**。内訳は契約の`call_limits`とrunnerの`PREPARE/FIT/EVALUATE`が一致する。optimizer内部nfev/njev・callbackをlogical callとして無制限再起動する意味ではない。1 fit内の内部反復は既存SciPy options=None、全実績を保存する。

実行順序はbyte/package preflight→保存h/g/cとJ/U→codec/tensor/shared scale→初期raw-CAR/general-Kと物理sector gate→独立processのpaired fit→両終了tensor/sector/表現→比較→mandatory STOP。fresh `artifacts/research_c_pr_aware_df/2026-10-26/run01`専用、retry/resume0。全52/260旧PFセルなどを再利用した新評価は含めない。

fit専用環境とevaluation環境は既存venvをread-only使用しpackage versionsを契約に記載した。fit側Qiskit2.4.2はffsim依存で、P0 compiler環境ではない。evaluation側は既存Qiskit1.3.0。実行時に両envのpassive preflightを行う。input取得先・export来歴・fixed ffsim2source hashは契約を参照。

資源は実行直前のCPU affinity・空きRAM・quota・pressureを再確認し、2個しかない独立fitを別processで並列化する。JAX x64と既存tensor ADを用い、cache/tempはtask-local。sector36の小行列評価は初期gate依存のため逐次、既存小行列BLAS設定を用いる。fit側JAX/BLASのthread競合は実際の空き資源に応じ自jobだけで調整し、速度向上率は未測定。CPU/RAM/worker/wallの科学契約上の人工上限は設けず、共有env・他jobは変更しない。

## 10. Stage B/Cと研究A/Bへ返す境界

| 段階 | 準備後に必要な最小判断・資格 | 今回の状態 |
|---|---|---|
| A | 上記paired fit＋物理予算＋exact λ。γ/ε/初期値・normalized baselineの採択 | 30-call案・人工接続まで。実行未承認 |
| B | 新θごとにtail/dictionary/finite-bit law再生成、固定πで減衰とPF/補正RTEを評価、native価格を両arm共通に校正 | scope/callをA結果後に別固定。0 call |
| C | 全law合成／law差／費用supportを資格付け、全13roundでCA/RPE shots→期待総RZ、CX/depth比較 | 未資格・未承認。0 call |

将来の固定πの推奨は元frame prefix LD2、partial S2、δ=.1、s4/r16/K0、m0..12、CA=.00159、βtotal=π/6・βstat=π/12、candidate failure=.01・axis=.01/26、own-DF oracle・状態準備除外、ideal finite-RTE IID law。比較両armに同じ政策とcompilerを適用し、提案だけr/K/compilerを選び直さない。これはB/Cの実行契約ではない。

既存PR tail `FullFrameProvider`は旧native Gaussian providerを使い、強baselineの高精度／相対Gaussian融合がPR全trajectoryへ自動適用されるわけではない。Stage Bでは同精度・同じ合法fusionの対応を先に確認する必要がある。決定論B2/B3の保存metricsを新PR価格として流用しない。一般相対融合の未観測効果や有限sample最大値を全law boundと扱わない。

- **研究C内**：固定split・law形式のままJ/Uを変え、係数・確率を正しく再生成する。DF／回路実装の公平性を担当する。
- **研究A/B／統括へ**：pを独立に最適化してreweightする、分割／frame順／s/r/K／PF方式や統計failure配分を共同最適化する場合。既存の保証を無断変更しない。
- **研究C／ユーザーへ**：第一候補の科学的情報価値・γ等の採択、Stage A実行許可、結果後のB/Cを行う価値の判断。penaltyの技術PASSを新規性やPR資源優位へ昇格させない。

最初に必要なのはStage A案の採否と、その固定範囲の実行許可である。Codexは許可後に限り30-call範囲を実行し、結果を返してSTOPする。今回の公開照合後もmandatory STOPし、新科学計算には進まない。
