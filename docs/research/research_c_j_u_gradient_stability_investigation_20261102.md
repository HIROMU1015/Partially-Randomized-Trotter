# 研究C：Reduced-objective型J/U最適化の数値安定性調査と次pilotの準備

2026-11-02 JST。Repository: HIROMU1015/Partially-Randomized-Trotter。

## 1. 結論・今回の実施境界

**次の実H4限定pilotには、残差の拡張精度評価と、事前登録した追加refinementを使うreduced方式を推奨する。** Joint J/η方式へ自動切替はしない。人工例では同一点の勾配評価精度が改善し、旧stabilityで停止する一部の例も通った。一方、改善後も未資格となる例、inner correctness gateで止まる例、jointでも一次条件に届かない例を保存した。実H4で有効かは未検証である。

今回行ったのはSOURCE/保存JSONの確認、数学的な誤差関係の整理、人工データでの実装・比較・回帰テスト、未承認契約の準備だけである。**実分子のNPZ decode、新しいH4 gradient/Hessian/残差/条件数/sector評価、fit、PF/RTE/ground、量子回路、shots・総費用は0件。** 詳細科学レビューも行っていない。

起点は結果commit `a4c0fcec7d03d72a34c0f304280c758a6fa2ae5c`。旧科学SOURCE `e1582a5620e61caec52f0ee284c8cad8c0202580`、[旧1051-call契約](research_c_j_u_reduced_contract_unapproved_20261102.json)、[38-call結果とSTOP](research_c_j_u_reduced_execution_20261102.md)は変更していない。旧allowance `1e-6`を緩和せず、旧FAIL/未資格をPASSへ読み替えない。以下の採用案は**別SOURCE・別契約の数値pilot**である。

## 2. 保存証拠で判明すること・判明しないこと（Q1）

引用対象は保存直鎖H4・隣接1.00Å・STO-3G、number CDF8、Nα=Nβ2/sector36、元frame順LD2、元qualified J/U/x0、α=γ=.1、元g補正κ/scalar、固定f*/M*/P0である。PF時間窓はこの最適化には非適用。数値は旧JSON/報告からの引用で、新しい配列評価ではない。

| 保存point | 独立J∞ | 物理U₂ | 登録refinement U差 | 結果 |
| --- | ---: | ---: | ---: | --- |
| PR初期 | 1.3249031e-7 | 2.954975096 | 5.6058350e-6 | REDUCED_GRADIENT_UNQUALIFIED |
| RC2初期 | 6.2833529e-8 | .867029585 | 2.1699547e-7 | 数値admission PASS |
| RC2未受理trial1 | 2.7930604e-7 | 79.26087532 | 8.5692603e-6 | REDUCED_GRADIENT_UNQUALIFIED |

全6 inner solveのJ一次条件、PR初期の共通head gauge/rank条件、同一点の微分整合、終端codec/sector表現、両診断終点のDF予算はPASSしている。PR raw Newton-CGは初回nit481、refinement nit2。RC2はaugmented `gelsd`/residual correctionのrank80が通り、保存correction normは初期4.3873143e-11、trial1で1.0388668e-10である。これらは保存値であり、新規にHessianや感度係数を推定したものではない。

したがって直接のSTOP理由は「固定1e-6でのrefinement stability未資格」である。J solverのiteration不足やPR追加nullspaceと断定できない。小さいJ移動に比べ大きなU勾配変化が観測されているが、実H4での真のinner誤差、混合微分による増幅、残差/ADの丸めの寄与は分離できていない。RC2のtrialは未受理で、objective上昇を受理ステップの非増加違反とは扱わない。受理U更新は0で、勾配rebound回避の成否も未判定である。

[保存JSON抜粋と元trace参照](../../artifacts/research_c_j_u_stability/2026-11-02/technical/saved_JSON_observations.json)に確認範囲を記録した。二つの近似勾配が近くても真のenvelope gradientへの近さは証明できず、差が大きいだけで後者が使用不能とも断定できない。旧gateは真の誤差評価ではなく、登録した経験的stability検査である。

## 3. 数学的構造と誤差の分離（Q2・Q3）

既存isometric座標xは上三角Jに対し対角重み1・非対角√2で、理想的な厳密演算では||x||₂=||J||F。既存再構成作用A(U)、tail Pauli作用T、目的関数、gradient/Hessianを再利用する。

RC2は

\[
F(x,U)=\|A(U)x-g\|^2/(2f_*)+\alpha\|x\|^2/P_0,
\quad H_{xx}\succeq\mu I,\quad \mu=2\alpha/P_0.
\]

exact inner residual r=F_x(x̂,U)なら、d=||r||₂/μに対し||x̂−x*||₂≤d。混合微分も必要で、exactな直交Uと既存isometric physical tangentの下では、保守的に

\[
\|G_U(\hat x)-G_U(x^*)\|_2
\leq {4d\over f_*}
\{\|A\hat x-g\|_2+\sqrt R(\|\hat x\|_2+d)\}.
\]

再構成tensorの4個のU因子を微分すると、単位physical tangentで||dA[η]x||≤4||x||、||A||≤√Rとなる。G_U差を、dAに掛かるxの差と残差の差に分解して上式を得る。ここでRはouter rankである。

この関係を実装したが、**floating residualを代入した値は厳密enclosureではない**。有限精度Uの非直交性やJacobian/adjoint/丸めの誤差も包含していない。`UNRESOLVED_FLOATING_RESIDUAL_AND_ARITHMETIC`として保存し、admissionの代替証明に使わない。人工RC2のscale1ではplug-in約1.68e-15だが、scale1e-9では約1.48e4で実用性を失った。正則化でμ>0でも、固定した小さいf*と混合微分が外側誤差を増幅するため、J∞≤1e-5だけからU誤差≤1e-6は導けない。

PRでは既存smooth tail mass二乗を維持する。共通headの普遍gauge、tail T injectivity、head rank/nullspaceに関する既存十分条件は各Uで確認する。追加nullや曖昧なsingular gapは未資格STOPである。Gaugeは消さず、全方向SVD座標も既存の可逆な計量として使い、ridge/PSD/符号/gauge固定制約を足さない。Quotient上の曲率やinner solutionの感度と、全空間の一様強凸性は別物で、**PRにRC2のμを転用しない**。H4初期構造gate PASSを、他のUでの一様曲率保証にも転用しない。

## 4. 最小の数値改善・旧SOURCEとの差分

| 変更 | 採用処理と維持条件 |
| --- | --- |
| 残差/累積 | 既存tensor再構成・係数adjointをNumPy longdoubleで実行。入力・solver出力はbinary64のまま |
| U勾配 | 既存ffsim/JAX x64の残差を明示U配列で微分し、拡張精度残差と縮約。その後、既存指数chartのJacobianへpullback |
| 同一点 | exactな入力J/U配列をbinary64/拡張精度双方に渡す。参照評価だけのweighted pack→unpackを挟まない |
| Inner RC2 | 既存augmented gelsd＋既存residual correctionを2回。正規方程式へ置換しない |
| Inner PR | 既存Newton-CG/analytic Hessian/可逆SVD計量、maxiter1000・xtol1e-12を維持し、同じ目的でrefinementを2回 |
| Admission | 最後の2 refinement間のphysical/chart勾配差の大きい方≤元の1e-6。旧coarse→refined差も観測値として残す |
| 一次条件 | 最終J/Uについて、同一配列でのbinary64・拡張精度の**両norm**に元marginを加えて元1e-5以下であることを要求 |
| 外側 | 既存SciPy BFGS/line search・全trialでJを再solve・受理点の非増加・physical tangent判定。未資格trialでarm停止 |

この環境のlongdoubleは**有効64 bit仮数**（binary64は53）であり、128-bit quadと呼ばない。拡張精度が利用不能なら明示STOPし、binary64へ黙ってfallbackしない。Jacobianとchart自体の計算はbinary64で、任意精度ADやvalidated gradient oracleではない。

同じ数学的配列点であっても、binary64の`x=D*j; j=x/D`はoff-diagonalを1ulp動かすことがある。人工J=.19でこの境界を検証し、新参照では不要なround-tripを除いた。**この人工事実を、旧H4のcodecやSTOP原因が確定したという意味には使わない。** 実H4でこの追加丸めが支配したかは評価していない。終端native ADの比較は実際にdecodeされたJ/U配列で行い、codec前の物理candidateを置換しない。ここでnativeはparameter ADで、量子回路ではない。

追加refinementは事前固定したinner処理であり、retryではない。最後の2点の一致を使うのは新しい数値admission案で、旧規則を事後変更するものでも、inner真値への誤差保証でもない。同じbiased gradientを2回得る可能性を残し、`rigorous_gradient_error_bound=null`を維持する。

## 5. 人工比較・改善しない場合（Q4）

[人工study原本](../../artifacts/research_c_j_u_stability/2026-11-02/technical/synthetic_study.json)。すべてseed41/42のinvented tensor/J/Uで、分子積分や保存H4配列を使わない。n=4/rank8/LD2例でも、H4と同じなのはサイズと引用したscale定数だけである。

### 独立参照による評価精度

2次元・rank3・LD1の人工roundoff/cancellation例について、binary64配列をexact入力とする独立80桁Decimalの積の微分を参照にした。旧ADを繰り返して同じ結果が出ることと、精度が高いことは区別する。

| f*（人工） | 旧U勾配とDecimalの差norm | 新U勾配とDecimalの差norm |
| ---: | ---: | ---: |
| 1 | 4.14949e-18 | 2.12469e-21 |
| 1e-9 | 4.14949e-9 | 2.12469e-12 |
| 1e-12 | 4.14949e-6 | 2.12469e-9 |

Decimalは人工小行列専用の独立referenceで、本番optimizerや新AD基盤ではない。この一例の精度改善をH4での達成精度や一般的な改善倍率へ外挿しない。

### Refinement admissionの成功・負結果

| 人工例 | 旧pair U差 | 新final pair U差 | 新判定 |
| --- | ---: | ---: | --- |
| RC2 n4/rank8/LD2/seed41/f*=2.1839632053e-9 | 9.59473e-6 | 4.56789e-7 | 数値admission PASS |
| RC2 同seed42 | 1.57676e-5 | 1.06262e-6 | REDUCED_GRADIENT_UNQUALIFIED |
| PR n2/rank3/LD1/seed41/f*=1e-10 | 1.00175e-6 | 2.72028e-10 | 数値admission PASS |

PRの旧値は境界直上であり、この成功を頑健な全域改善と呼ばない。PR n4/rank8/LD2のseed41/42では旧・新とも既存innerの`J_POINT_CONSISTENCY_MISMATCH`で停止した。これも隠さず保存した。旧H4では同gateがPASSしているので、H4でその不具合が再発したとは報告しない。

### Outer最適化とjoint代替

次はn2/rank3/LD1/seed41。旧/新reducedはいずれも外側64iteration・128pointまで、新方式は各pointにinner solveを1回増やす。Jointはidentity座標BFGS64と既存full-direction SVD座標BFGS256を人工対照にし、最後に既存J solve/refinementを行った。異なる古典予算の比較である。

| Arm / f* | 旧reduced | 新reduced | 新reduced保持終点F / U₂ | Joint64 / Joint-SVD256 の最終J/U一次 |
| --- | --- | --- | --- | --- |
| RC2 / 1 | BOTH_MET | BOTH_MET | .01510540 / 6.57379e-6 | 両方MET、F≈.01501433 |
| RC2 / 1e-9 | BOTH_MET | BOTH_MET | .01692925 / 2.87356e-6 | 両方U未達、U₂ .21661 / .16309 |
| RC2 / 1e-12 | J_UNQUALIFIED | J_UNQUALIFIED | .39974216 / 9.94977 | 両方未達 |
| PR / 1 | BOTH_MET | BOTH_MET | .004902096 / 6.73026e-6 | 両方MET、F≈.0049020963 |
| PR / 1e-9 | J_UNQUALIFIED | J_UNQUALIFIED | .008626170 / .01470677 | 両方U未達、U₂ .27640 / .20232 |
| PR / 1e-12 | J_UNQUALIFIED | J_UNQUALIFIED | .27568525 / 4.67919 | 両方未達 |

PR scale1e-9の保持終点JはMETだが、その後のtrialのinner Jが未資格でtrajectoryを止めた。終点のJ値をtrajectory全体の資格と読み替えない。Jointはenvelope admissionを要しないが、悪条件下の同時一次収束は別問題である。

負担も別に記録した。RC2 scale1e-9は25点・受理23stepで、旧50回/新75回のlinear solve、新方式の試験wall .167秒/旧 .078秒。PR scale1は12点・受理9stepで、旧nfev/njev/nhev=2315/2265/1007、新2878/2688/986、wall .176/.231秒。JIT初回、cache、異なるsolve数を含む観測で、速度倍率や同予算での優位ではない。Jointのnit/nfev/njevと最後のJ solveも原本へ別記した。古典負担をlogical unitだけで比較しない。

人工conditioning例ではRC2 augmented conditionがscale1で4、scale1e-9で約1.22e5へ増え、混合微分FD normも約1.27から1.27e9へ増えた。PRのnormal-product Hessianの最小floating eigenvalueは小さく、scale1e-9では丸めで負値も出た。これを厳密な負曲率や強凸性否定とせず、正規方程式を避ける理由、有限精度の条件評価の限界として保存した。

## 6. テスト・SOURCEの再利用（Q7）

**既存186＋新34＝220 unique pytest items PASS。** Fit環境185、fit-worker回帰2、評価環境33。ローカル人工・回帰結果で、CI/外部再現の認定ではない。[実行logと件数](../../artifacts/research_c_j_u_stability/2026-11-02/technical/test_results.json)。

目的/係数gradient・既存Hessian、RC2残差boundの成立条件、PR普遍gauge/追加null、det±/π回転、same-point、physical/chart、J再solveを伴うreduced FD、inner不足/outer上限/未資格STOPを検証した。小scale1e-9のreduced方向FDはRC2差約1.86e-7、PR約3.19e-7で、人工FD用の事前テスト許容差内。これは科学契約の1e-6を変更するものでも、厳密gradient誤差証明でもない。

人工E2Eでは実ffsim/JAXのworkerをmock分子packetへ接続し、独立SectorCAR/general-Kの親評価、入力hash不一致、表現FAIL時partial STOP、fresh output再利用拒否、旧statusの保全、call計上を確認した。正常な人工経路37 units、表現故障33 unitsを記録するが、これは**人工テスト内**のcallであり、実H4科学callではない。H2/H4関連の旧codec・same-point・ffsim経路の回帰もPASSした。

作業branchは `codex/research-c-j-u-stability-preparation-20261102`、独立worktreeは `/tmp/prt-c-j-u-stability-preparation-20261102`。新SOURCEは6ファイルのみ：

- `src/trottertracks/c_j_u_precision.py`：既存再構成/adjoint/ffsim-JAX微分のprecision接続、RC2誤差関係の浮動小数点診断。
- `src/trottertracks/c_j_u_stable.py`：追加refinementと新数値admission。SciPy outer driverは既存がoracleを直指定するため最小コピーし、optimizerは再実装しない。
- `src/trottertracks/c_j_u_stable_pilot.py`：既存snapshot/worker/sector/codec/STOPを再利用する専用接続、旧reduced JSON比較。
- `scripts/run_research_c_j_u_stable.py`：既定passiveの入口、venv Python実パスを保持。
- `tests/management/test_research_c_j_u_stability.py` / `test_research_c_j_u_stable_pilot.py`：人工参照・比較・E2E。

`c_j_u_reduced*`、`c_separated_j_u*`、`c_pr_aware_df`、`c_rcdf_comparator`、`c_fixed_u_tail_lp`、ffsim/SciPy本体、両venvは無変更。独立worktree/branchで、旧raw1879件（従来1825＋直近54）、旧SOURCE12 manifest・各参照ファイル、旧契約/STOPをhash照合し、すべて一致した。[保全確認](../../artifacts/research_c_j_u_stability/2026-11-02/technical/preparation_postflight.json)／[技術STOP](../../artifacts/research_c_j_u_stability/2026-11-02/technical/STOP.json)。科学結果が変わっていないため科学manifest130件・VALIDATION_STATUS・研究概要・過去noteは更新せず、今回の技術報告と必要な入口索引だけを加える。

## 7. 選定と次の未承認実験（Q5・Q6）

Reduced方式を継続する技術的理由は、目的・閾値・PRの数学的意味を変えず、独立参照で評価精度改善が確認でき、追加refinementで数値admissionを得る人工例があること。一方、jointは簡単な例ではより低いFへ達するが、悪条件例では追加予算を与えても一次資格が得られない。今回の比較からjointの普遍的優位は支持できず、二方式を実H4で一括実行する契約にはしない。

[未承認1563-call契約](research_c_j_u_stable_contract_unapproved_20261102.json)で以下を固定した。

| 項目 | 新pilot条件 |
| --- | --- |
| 対象 | 保存直鎖H4/1.00Å/STO-3G、number CDF8、sector36、元順LD2、一般signed対称J/実直交U |
| 目的・scale | 元α=γ=.1、ε1e-8、f*=2.1839632053132894e-9、M*=2.2093719370722074、P0=4.410884790420646 |
| 初期値/物理規約 | 同じ元qualified J/U/x0、各arm独立、元g補正κ/scalar/spin/frame。旧終点warm-startなし |
| Inner | RC2既存gelsd、PR既存Newton-CG、各Uで初回＋2 refinement、tol/目的不変 |
| Outer | SciPy BFGS、maxiter64/max distinct points128、same-η byte cache、未資格trialで停止 |
| 数値資格 | 同一点cross-precision/既存gradient/value gate、両precisionで元J/U一次条件、final pair≤1e-6、PR構造/RC2 rank |
| DF | 終端の独立sector norm＋既存engineering guard≤.00019875Ha、一般K/raw CAR表現gate |
| 比較 | 旧joint/分離/修正版/旧reducedの保存JSON。Fは同role内、exact tail λ/一次資格/DFを別々に判定 |
| Fresh | `artifacts/research_c_j_u_stability/2026-11-02/run01`、今回未作成 |
| 新最適化 | PR/RC2各1 trajectoryのみ。ffsim joint fitter0、再始動/別方式fallbackなし |
| Retry/resume | 0/0。旧残枠1013は転用しない |

| Call内訳 | 各arm最大 | 全体最大 |
| --- | ---: | ---: |
| Load/setup/元identity/保存history | 4 | 8 |
| Inner solve（refinementを含む） | 3×128=384 | 768 |
| Precision state | 2×128=256 | 512 |
| Reduced admission/構造/RC2 bound診断 | 128 | 256 |
| Outer driver（nested callと別） | 1 | 2 |
| 終端独立/ tensor/ export | 3 | 6 |
| Parent load/初期資格/sector/physical/final/DF/表現/比較 | — | 11 |
| **合計** | **776** | **1563** |

科学callは費用計測単位で、内側nfev/njev/nhevやJAX呼出し数の上限とは別である。実際のinner/outer raw、gradient、refinement全点、受理/未受理、時間/CPU/RSSを保存する。旧runの再実行や新rank/strength/初期値探索を含めない。H4のHessian sweepはこの最小契約にも加えていない。

最大call数は `11+2*(7+1+128*(3+2+1))=1563` と新処理から積み上げた。

SOURCE manifest SHA256は `1d22c3426742af2221822b6af7fa79d5ff7f9b095d4dffdf736c6475f9b2f509`、契約SHA256は `c255a2bc553a7f3738ae92b632f727c0fead34acb46c64983a6600e00b0f2baf`。

SOURCE139ファイル、input31ファイルを固定し、既存の2つの依存lock/ffsim sourceを再利用する。両環境で配列decodeなしのpassive preflight PASS。既存plannerは実空き資源を確認して独立2armを並列化し、小行列BLASの過剰threadを避ける。直近観測worker peakから余裕を持つRAM見積りを使うが、CPU/RAM/worker/wallの恣意的上限は設けない。

Correctness FAILでは依存処理を止め部分証拠を保存する。数値未資格は失敗したtrialと最後の受理点を区別して保存し、後者のexportは診断だけで、自動fallback成功と扱わない。両一次/DF/新admissionが揃わなければqualified tail比較はnull。Optimizer flag・物理一次条件・DF予算・厳密保証は独立の判定である。成功してもPF/RTE/native/shotsへ自動進行しない。

## 8. Q1〜Q7への回答とSTOP

| 問い | 回答 |
| --- | --- |
| Q1 原因 | 旧JSONから直接の停止gateは確定。小f*による残差丸め/混合感度は人工例で確認。実H4での各寄与と真のinner誤差は未分離 |
| Q2 RC2強凸性 | 厳密算術のinner→outer誤差関係を利用可能。ただしfloating plug-inは未保証・しばしば過大で、実H4の有用なboundは未取得 |
| Q3 PRの改善 | Gauge/目的不変でprecision/追加refinementの人工改善あり。構造条件と微分精度を分離し、追加null/未資格はSTOP |
| Q4 改善と限界 | 独立Decimal誤差と一部admission改善。一方scale1e-12、n4PR inner整合、RC2 seed42等は解消せず |
| Q5 方式選定 | 改善reducedの数値pilotを第一候補。Jointは人工代替として保持し、自動fallback/実H4同時比較なし |
| Q6 次の計算 | 新SOURCE/1563-call契約のPR/RC2各1 trajectory、終端一次/DF/旧JSON比較。ユーザーの別途明示許可が必要 |
| Q7 再利用/保全 | 既存solver/AD/再構成/codec/sector/pipelineを再利用、旧SOURCE無変更、新6ファイル。旧raw1879/旧STOP・科学manifestを保全 |

残る主要blockerは、実H4で新方式が数値admissionと同時一次条件を確保するか不明な点、真のgradient誤差enclosure未取得、悪条件でinner precisionが不足し得る点である。技術preflight上の未解決依存不整合はない。

**ユーザー／研究C**は新SOURCE・契約を確認し、この数値pilotを実行するか判断する。**Codex**は別途明示許可後に限りfresh runで上記範囲を実行する。今はcommit/push・remote SHA/tree/公開本文/SOURCE/保全照合後に**mandatory STOP**。新規H4計算や詳細科学レビューへ進まない。

## 9. 技術的根拠として参照した一次資料

- [Higham, Iterative Refinement and LAPACK (1995)](https://www.netlib.org/lapack/lawnspdf/lawn104.pdf)：残差の評価精度とrefinementの関係を確認した。適用条件のある解析で、本実装の収束保証として引用しない。
- [Carson・Daužickaitė, A comparison of mixed precision iterative refinement approaches for least-squares problems (v2, 2025)](https://arxiv.org/html/2405.18363v2)：least-squaresでは非零残差とconditioningがrefinementの成立条件に関係する点を参照した。新solverは導入していない。
- [Español・Jeronimo, Convergence Analysis of a Variable Projection Method for Regularized Separable Nonlinear Inverse Problems (2024)](https://arxiv.org/abs/2402.08568)：abstractでinexact inner solve/近似Jacobianに停止条件が必要という位置づけを確認した。この論文のLSQR/variable-projection解析をPR-tail/BFGSへ転用しない。
