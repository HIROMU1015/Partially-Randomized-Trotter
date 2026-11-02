# 研究C：PR reduced stability未資格の切り分け・qualified-only step技術準備

2026-11-02 JST。起点 `fc4ee5712d475812c03e647472f41901bdfa0165`。研究C・保存直鎖H4/1.00Å/STO-3G、number CDF8、元順LD2、Nα=Nβ=2/sector36の前段結果を扱う技術調査。

## 1. 結論と今回の範囲

**実H4の7.3109e-6の原因は未確定**。保存JSON/JSONLから、J基準・構造・同一点微分は通過し、最終refinement pairのU勾配差だけが未資格だったことを確認した。J差・混合Hessian・physical/chart個別差は未保存であり、NPZをdecodeして補っていない。

人工検証では、J一次残差が小さくF変化が表示されなくても、混合曲率によりU勾配差が大きくなる例、丸め改善の成功/失敗、gaugeと非gaugeの違いを確認した。Precision lossは原因の確定材料でも資格の代替でもない。

次の技術推奨は**候補B単独：innerと全資格基準を維持し、未資格trialを棄却する、1回だけのqualified-only Armijo step**。既存SciPy BFGS/Wolfeへ架空の値やgradientを渡して続行する変更ではない。新SOURCE・人工試験・[未承認60-call契約](research_c_pr_guarded_step_contract_unapproved_20261102.json)を準備した。方式Bの正式採択・実H4での改善は未判断である。

**新しい実分子decode・勾配/Hessian評価・再最適化・科学callは0**。許可範囲は保存JSONの整理、静的SOURCE調査、人工検証と技術実装まで。旧18-call/54-call・過去のSOURCE/raw/契約/STOPを変更せず、公開照合後mandatory STOPする。

## 2. 保存証拠から分かったことと不足

[固定実行報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc4ee5712d475812c03e647472f41901bdfa0165/docs/research/research_c_pr_reduced_limited_execution_20261102.md)をGitHubから取得し、local同一commitの保存値と照合した。[JSONだけの整理結果](../../artifacts/research_c_pr_stability_investigation/2026-11-02/technical/saved_log_findings.json)に出典hashを記録した。

| 保存scalar/判定 | 初期point0 | 未受理trial point1 |
| --- | ---: | ---: |
| Initial J solve F before→after | 1.099999999998→.0122676444298 | 158736947.173→1.646654647446 |
| Refinement F before/after | 両段とも同じ表示値 | 両段とも同じ表示値 |
| Independent J∞ solve/refine1/refine2 | 3.20233e-7 /7.60432e-8 /7.60039e-8 | 1.03924e-6 /7.46977e-7 /1.18793e-7 |
| Raw status solve/refine1/refine2 | 0 /2 /0 | 0 /0 /2 |
| nit solve/refine1/refine2 | 591 /1 /1 | 297 /1 /1 |
| Latest physical/chart shiftの大きい方 | 4.236925151e-10 | 7.310894271e-6 |
| Legacy coarse→refine1の保存観測 | 1.554135074e-6 | 4.761814276e-7 |
| Extended対binary64 U差 refine1/refine2 | 5.05019e-8 /5.06655e-8 | 4.20660e-7 /1.85067e-7 |
| J一次・rank/nullspace・chain | PASS | PASS |
| 最新pair stability ≤1e-6 | PASS | 未資格 |

Coarse→refine1は旧binary64 coarse経路との観測であり、最新pairの拡張精度physical/chart差と同じ測定ではない。これを同一系列の単調な収束指標と扱わない。保存U勾配のnorm同士の差からvector差のnormも復元できない。

静的SOURCE上、最新pairは同一のV配列・同じ目的・同一点の残差/微分規約を使う。以前の異なる評価経路によるchart/physical整合FAILは今回の2点ではない。旧trial2には未到達で、全trajectory解決を意味しない。

Metric行列の保存singular rangeは両点とも約2.6e-11～6.0523e4。ただしこれは`[A/sqrt(f*);sqrt(2γ)T/M*]`のmetric診断であり、最小値には既知head gaugeが含まれる。**物理quotient Hessianの条件数や支配方向を測った値ではない**。変換Wはfloor1の全方向可逆変換で、方向を削除/固定していない。

未取得：①個別のphysical/chart最新差、②extended coarse→refine1の同規約差、③ΔJとそのgauge成分、④実H4のH_UJ/quotient Hessian・condition、⑤真のgradient誤差上界。今回これらを再計算しない。次案には既存計算済み配列のphysical/chart個別差とΔJ Frobeniusだけを追加し、新Hessian診断は含めない。

## 3. 原因候補の人工切り分け

[人工結果・独立参照](../../artifacts/research_c_pr_stability_investigation/2026-11-02/technical/artificial_mechanisms.json)。対象は解析toyとseed41等のinvented tensor/J/Uのみ。実H4の配列・fitを使っていない。

### 3.1 小さいJ残差とU感度

局所的には`Δ(∂U F)=H_UJ ΔJ+O(||ΔJ||²)`。J残差からΔJを制御するには、gaugeを除いたH_JJの下限等が必要で、J∞≤1e-5だけではU誤差≤1e-6を導けない。小さいf*は残差由来の勾配/混合曲率を1/f*で増幅するが、単独で実H4の原因を確定しない。

解析toy `F=1+(a j-b u)^2/(2f*)`、a=1e-6/b=1/f*=1e-12、u=0、Δj=1e-8では、J残差1e-8に対してU勾配差.01となる。目的差はDecimal80で5e-17、binary64のFは同じ1。解析H_UJ=-1e6は独立FDと一致した。**J gateとF停滞がU精度の保証にならない機構例**であり、実H4の混合Hessian測定ではない。

PRのLD2 universal gauge `ΔJ_head0=c11^T, ΔJ_head1=-c11^T`は、正確な直交Uならtensor/目的/U微分を変えない。既存DF mapとADを使う人工試験でこれを確認し、非gauge移動ではFが変化する対照も確認した。実H4のΔJ分解は未取得で、移動量をそのまま物理誤差と解釈しない。

### 3.2 数値評価とinner改善の限界

2軌道invented DF、f*=1e-12の同一binary64配列における独立Decimal80 product-rule参照では、既存extended U誤差2.12469e-9、binary64経路の誤差4.14949e-6を観測した。同一経路の内部一致だけでなく参照精度の改善を確認したが、既存extended方式にも残差があり、誤差enclosureではない。この改善経路は実H4でもすでに使われている。

別の解析toyでは、既存Newton-CGがsuccess=trueのままj=1e-8を残し、J gateを満たしてもU誤差.01が残った。静的SOURCEにはCG方向curvatureが機械epsilon以下の場合の打切りがあり、このtoyでは精度と停止規則を分ける必要がある。これを実H4のprecision lossの原因と転用しない。

| 人工inner比較 | U誤差基準1e-6 |
| --- | --- |
| Newton-CG、元の定数付きF | 未資格 |
| Newton-CG、定数を拡張精度で引いてからcast | 未資格 |
| 既存SciPy trust-exact、元F | 未資格（raw失敗） |
| trust-exact＋定数を引いた同一数学的目的 | PASS、j=0 |

**人工問題では閾値を変えず改善した例があるが、centeringだけ/solverだけでは改善しなかった**。実PRのsmooth tail・head gaugeでの堅牢性、raw-value規約・conditioningへの影響は未確立。したがって候補Aを実H4用innerへ接続せず、Cの組合せも採用しない。Toyの定数除去/solver変更は人工比較だけで、次の科学契約では禁止する。

4軌道rank8のinvented DFでは、実Newton-CGがstatus2/precision lossを返しながら独立J∞8.49581e-7でMETの例も得た。保存実H4のprecision lossは初期refine1にも存在し、その初期点はstability PASS。**Precision lossは今回のH4不安定性の十分な原因証拠ではない**。

## 4. Outer政策A/BとSciPy API

固定SciPy1.14.1の`_minimize_bfgs`、`_line_search_wolfe12`、`scalar_search_wolfe1`を確認した。[API/SOURCE確認](../../artifacts/research_c_pr_stability_investigation/2026-11-02/technical/scipy_api_findings.json)。BFGSはWolfe探索で値と勾配を要求し、独自の未資格例外をtrial棄却通知として扱わない。実SciPy BFGSによる人工試験でも例外が伝播して停止した。

公開`line_search.extra_condition`は、値とgradientを取得してstrong Wolfeを満たした後の追加条件である。未資格gradientの代わりにはできない。[SciPy1.14.1 line_search](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.optimize.line_search.html)、[BFGS options](https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.minimize-bfgs.html)。

そこで小さい**外側step制御**だけを追加した。新BFGSや独自Newton法は作らず、内側の既存Newton-CG/Hessian/map/AD/state/qualificationを再利用する。BFGSの継続を装わない。

1. 元initial eta=0のnumerically admitted g0だけから`d=-g0`を作る。
2. `α0=min(1,1.01/||g0||)`。これはH0=IのSciPy1.14.1最初のBFGS問い合わせscaleと同じ静的規則で、実H4のtrial identity再現を今回測ったわけではない。
3. `α_i=α0*0.5^i, i=0..7`を事前登録。各点で元Jから3 inner/2 state/元admissionを実行する。
4. 未受理trialの`J_UNQUALIFIED`/`REDUCED_GRADIENT_UNQUALIFIED`だけを明示棄却し、gradientをdirection/slope/受理/更新に使わない。探索用Fはnullのまま、架空のpenalty/NaN/ゼロgradientを返さない。
5. Qualified pointだけで、実測`F_trial<=F0+1e-4*α(g0·d)`かつ`F_trial<F0`を確認し、最大1 stepを受理する。
6. Correctness FAILは即例外STOP。Point上限/zero step/非降下/8試行終了は明示STOP。未資格初期点なら探索しない。

未資格点の近似Jで得たFが高いことも、正確な`min_J F`への下界証明ではない。単純なvalue-only棄却の正当化に使わず、**資格が不足した点として棄却する**。資格付き点のF/gも数値pilotなので、これは観測されたArmijo降下であり、厳密なΦや真のgradientへの保証ではない。Strong Wolfe/BFGS curvature updateは使わず、その収束理論も主張しない。

### 人工A/B比較

`F(j,u)=(j-u)^2/2+u²/2`、j*=uの解析toy。大きいtrialではrefine1/2のJ残差8e-6/0がともに1e-5内でもU差8e-6で未資格、小さいtrialでは1e-8/0で資格を満たすよう、inner精度を制御した。これは方策を検証する人工oracleで、実H4のerror profileを推定したものではない。

| 人工ケース | A：未資格STOP | B：資格付きstep探索 |
| --- | --- | --- |
| 大trial未資格・小trial資格あり | initial＋1 trialでSTOP | initial＋2 trialsで資格付きArmijo受理 |
| 全trial未資格 | initial＋1 trialでSTOP | initial＋8 trialsで上限STOP、受理0 |
| Correctness FAIL | 最初のtrialで即STOP | 同じく即STOP、次αを調べない |

Genuine invented DFでも、元のJ/rank/nullspace/stability gateを全て通した1 step受理とcall計上を検証した。いずれも人工結果であり、実H4で進む保証ではない。資格領域に穴がある/8点内に有効点がない場合は進まない。母問題の最適性や全域安定性を示さない。

## 5. 実装・検証・旧原本保全

- `c_pr_guarded_step.py`：既存`c_pr_reduced_limited.Oracle`を継承し、同じgateにpairの個別norm/ΔJ診断だけ追加。小さいArmijo制御は旧SciPy APIに未資格trial protocolがないため新規実装。
- `c_pr_guarded_step_pilot.py` / `run_research_c_pr_guarded_step.py`：既存入力loader、Calls、raw/JSON/NPZ出力、resources/STOPを再利用した独立entry。既定はpassive preflight。
- 新人工test3ファイル：機構、政策、E2E。旧SOURCE154/default/契約を変更しない。全方向metricと元J初期化、2 refinement、raw jacのpre-step/accepted/endpoint規約、独立stateを保持する。

[試験結果](../../artifacts/research_c_pr_stability_investigation/2026-11-02/technical/test_results.json)：**251 unique pytest items PASS（新25＋既存226）**。fit241、既存評価venv10。初回24は重複で加算しない。FD・解析Hessian・Decimal・gauge・raw精度・資格付き受理・未資格/FAIL・trial上限・Calls/STOP・fresh拒否を検証した。旧適合回帰を変更後に実行し、過去PASSの転記ではない。CI/外部再現とは呼ばない。

[passive preflight](../../artifacts/research_c_pr_stability_investigation/2026-11-02/technical/passive_preflight.txt)は160 SOURCE/4入力・既存依存lock/ffsim SOURCE/versionを照合してPASS。実H4配列は閉じたまま。[保全postflight](../../artifacts/research_c_pr_stability_investigation/2026-11-02/technical/preparation_postflight.json)で旧raw2423、旧SOURCE16組、旧18-call/54-call STOP・契約を保全する。

人工計算はvectorized NumPy/SciPy/JAX/既存cacheと専用venvを再利用し、fit/evaluationの独立試験を並列に処理した。小行列BLAS thread調整は既存に合わせ、逐次依存するoptimizerを並列化せず、共有環境・他job・cache/OSを変更しない。恣意的なCPU/RAM/worker/wall capやspeedup主張はない。

## 6. 推奨する新60-call科学実験（未承認）

[固定契約案](research_c_pr_guarded_step_contract_unapproved_20261102.json)。**B単独・1回の資格付きstepの可否**を最初に調べる。H4/1Å/STO-3G、number CDF8/LD2、Nα=Nβ2、元qualified J/U/x0、γ=.1/ε1e-8、f*=2.1839632053132894e-9/M*=2.2093719370722074、元κ/scalar/spin/frameを維持する。

J/U一次1e-5＋元margin、gradient atol1e-8/rtol1e-7、value atol1e-10/rtol1e-8、stability1e-6、元PR rank/nullspaceは不変。Newton-CG条件も不変。変更案は**outerのWolfe BFGS/未資格即全体STOPを、1方向・8候補のqualified-only Armijoへ置き換えること**で、ユーザー／研究Cの採否を要する。

| Logical unit | 最大 |
| --- | ---: |
| saved_load / initial_identity / objective_setup | 各1＝3 |
| outer_driver（BFGSではなく1 step controller） | 1 |
| inner_J_solve（initial＋8 trials、各3） | 27 |
| precision_state（各2） | 18 |
| reduced_admission＋既存配列のpair診断（各1） | 9 |
| classification / export | 各1＝2 |
| **合計** | **60** |

最大9 distinct U点、27 inner（うちrefinement18）、受理U更新≤1。実VG/Hessian/iterationsも別途保存する。登録trialはアルゴリズムの新評価点であり、旧runのretry/resumeではない。新namespace・新契約・fresh `artifacts/research_c_pr_guarded_step/2026-11-02/run01`を使い、retry/resume0。旧未使用36 unitsを転用しない。

SOURCE manifest SHA256：`ce9073667a287714d8b66e3e2abcf0104a21b86a1e68567e8b84323f56773047`。契約SHA256：`5407c96b715dd22c0332cc02b63c0ddb1c72aa791951acc6c64539879dc2073c`。

新しい混合Hessian/conditionやgauge分解、DF物理normは最初の実験に含めない。DF予算CA/8=.00019875Haは維持するが未測定/null。RC2、joint、paired、PF/RTE/ground/native/shots/総資源も対象外。成功時も1 stepの数値的前進だけ、未資格継続時も上限で公開STOPする。

## 7. Q1～Q10と次の担当

| 問い | 回答 |
| --- | --- |
| Q1 実H4の7.3109e-6の原因 | 保存scalarでは直接のstability未資格を確認。ΔJ/H_UJ/個別差がなく原因は未確定。人工では感度・丸め・内側停止の機構を分離した |
| Q2 小さいJ残差でもU差が残る理由 | H_UJとquotient inner精度の組合せが必要。解析/FD/DecimalとDF gauge対照で機構確認、実H4の感度は未測定 |
| Q3 Precision lossの関係 | trial最終段で共起する一方、初期資格PASS側にもあり、因果確定も資格代用も不可 |
| Q4 Inner改善 | 人工toyでは閾値を変えず改善例あり。ただし単独変更で失敗例もあり、元PRへの堅牢なinner修正は未選定。実H4 innerは変更しない |
| Q5 STOP政策の改善余地 | A停止/Bが小stepを受理する人工例あり。Bでも全未資格は有限上限STOPし、correctness FAILは継続しない |
| Q6 Guard付き探索 | 既存BFGSに未資格gradientを渡して継続できない。資格付き数値値/勾配だけの1 Armijo stepを最小実装。Wolfe/厳密Φ保証ではない |
| Q7 次の最小H4 | 元初期点＋最大8 halving trials、資格付き1受理stepの可否と個別pair差/ΔJ norm。新60 units・DF評価なし |
| Q8 Jointへ移る条件 | 有効stepを得られない、inner資格が維持できない、gauge/conditioning対応が複雑化する等が続く場合に研究Cで比較判断。Jointも微分精度問題を自動解決しない |
| Q9 再利用/追加 | 旧inner/map/AD/Hessian/state/資格/Calls/STOPを再利用。新規は小さいouter制御とdormant entry/人工試験だけ |
| Q10 判断/承認 | ユーザー／研究CがBFGS/Wolfe→1 Armijo stepの変更案と60-call範囲を判断。別途明示許可後のみCodexが実H4で一度実行 |

Branch `codex/research-c-pr-stability-investigation-20261102`。Commit/push後、remote SHA/tree・公開報告/SOURCE/契約/test結果と旧raw/STOPの一致を照合する。**今回の技術準備完了後はmandatory STOPし、実H4科学実行へ自動進行しない。** 承認証明書・原文保存・新grantは不要。研究方針の正式変更や重要GPTレビューも今回行わない。
