# 研究C：PR inner勾配整合FAILの人工原因切分け・最小修正候補・10-call診断準備

2026-11-02 JST。起点は固定結果commit `9c9eafe74493a5bd4bfa59e41db0e8bf80785e6f`。

## 1. 結論と今回の実行境界

**人工問題では、同じ数学的残差を異なる浮動小数点経路で評価していることが、今回と同型の勾配整合FAILを生む主な機構だった。** Isometric重みやoff-diagonalのchain ruleを変えず、両経路を同じdecoded J配列へ揃えると元allowance内へ戻った。片方の残差だけを拡張精度にしても解消しなかった。実H4の失敗点で同じ機構が支配したかは未検証である。

最小修正候補として、既存のtensor再構成/adjoint/Pauli写像・可逆SVD計量・解析Hessianを再利用する、opt-inの同一配列点評価器を実装した。独立Decimal参照、有限差分、既存SciPy Newton-CGへの人工接続も確認した。**旧solver、旧paired runner、目的関数、PR gauge、数値allowanceは変更していない。** 全paired実行を再起動する準備とは区別する。

RC2は保存JSONの確認だけで独立診断した。64更新でFは低下し続けたが、保存U勾配は26更新で増大しており単調ではない。直接の終了理由はiteration上限だが、単なる反復不足か更新/conditioningの制限かはまだ判断できない。RC2の新しい継続条件は採択しない。

次は**保存PR trial2の勾配整合だけを評価する未承認10-call契約**を第一候補とする。新しいinner solve/fit/最適化trajectoryは0。失敗trialのUが直接保存されていないため、保存済み2点を使う既存BFGSのdry trace replayを明記した。実際の復元・実H4評価は今回行っていない。

今回の許可範囲はSOURCE・保存JSON/ZIP metadata、人工・mock検証、修正候補と新契約の技術準備である。**実H4 NPZの数値decode、新gradient/Hessian/残差/条件数/operator、再fit、RC2 continuation、PF/RTE/native回路/shots/総費用、新詳細科学レビューは0件。** 旧480-call STOPと未使用1083 unitsは保全する。

## 2. 旧実H4で確認済みの事実

引用条件は保存直鎖H4・隣接1.00Å・STO-3G、number CDF8、Nα=Nβ2/sector36、元frame順LD2、元qualified J/U/x0、α=γ=.1、ε1e-8、固定f*/M*/P0、元g補正κ/scalar。PF時間窓は今回の係数診断に非適用。

[480-call固定報告](research_c_j_u_stable_execution_20261102.md)と保存traceでは、PR trial2の`chart_vs_physical_gradient`差2.3969496e-7、元allowance比23.9694が直接の停止predicateである。raw jac対last-step-startはPASS、objective比較もPASSだった。Raw jac対endpointがFAILでもpre-step検査とのOR条件は満たすので、そのFAILだけを停止原因とはしない。

新stabilityは初期point0/未受理trial1でPASS、trial2はinner整合FAILのためrefinement/admissionに進んでいない。受理PR U更新0、raw Newton-CG success=trueを物理一次資格や正しさPASSに読み替えない。今回保存配列を再評価してこのFAILを解消したわけではない。

[保存JSON・archive member確認](../../artifacts/research_c_pr_coefficient_point/2026-11-02/technical/PR_saved_metadata.json)はmetadataだけで、`.npy`の数値内容を読んでいない。`inner_raw_2_solve.npz`はx0/x/z/W/raw jac等を持つが、outer η2/U2は含まない。一方、envelope0/1とquery1のUは保存されている。将来の契約で「失敗点のUを読み込むだけ」とするのは誤りなので、その不足を処理する入口を別に用意した。

## 3. 旧SOURCEと数学的chain rule

既存isometric座標はx=D upper(J)、Dは対角1・非対角√2。実対称Jのoff-diagonalを動かすと、tensorには二つの対称な項が入り、既存adjointはこれらを足してDで割る。`FixedU.matrix`、`FixedU.adjoint`、`Coefficients.pack/unpack/vg`を確認した。符号や係数2の公式ミスは人工検証で検出していない。

旧inner chartは、可逆なWについてx=x0+Wzとし、以下を使う。

\[
r_{affine}=(Ax_0-g)+(AW)z,\qquad c_{affine}=Tx_0+(TW)z.
\]

独立original-coordinate経路は、実際にbinary64で作ったxをdecodeし、`reconstruct(J,U)-g`とadjointを計算する。実数の厳密演算なら同じだが、次の違いが残る。

- `A@x0+(A@W)@z`と`A@(x0+W@z)`の丸め・加算順。
- Matrix列を作るUの積と、直接tensor再構成の積・縮約順。
- Weighted x→Jのbinary64除算による配列点の移動。
- `T@x`と実際のJからのPauli係数計算の丸め。
- Physical gradientをbinary64へcastしてからWへ縮約する場合のcancellation。

旧gateはchain ruleの公式だけを比較しているつもりでも、残差や係数点が違えば別の数値oracleを比較する。小さいf*で残差差が勾配へ増幅され、F一致のallowance内でも微分一致は落ち得る。**同じ物理関数を表すことと、浮動小数点で同じ評価点・残差を使うことは区別が必要である。**

## 4. 人工再現・切分け・精度検証

対象はseed41/42のinvented n4/rank8/LD2、f*=2.1839632053132894e-9。サイズとscaleがH4と同じでも、g/J/Uは保存分子データとは無関係である。旧の既存Newton-CGをそのまま人工実行して得た終点で比較した。

| 人工例 | 旧chart/physical勾配差 | affine残差対actual-point拡張残差の差max | physical側だけ拡張精度 | 同一残差・binary64の両chain差 | 新同一配列・拡張精度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed41 | 1.3628551e-7、FAIL | 3.7516106e-16 | 1.3358621e-7 | 6.9050149e-15 | 4.4694752e-18 |
| seed42 | 4.6203292e-7、FAIL | 6.1287238e-16 | 3.4144538e-7 | 4.9274427e-14 | 2.6237693e-17 |

Ablationでは、共通のactual-point残差を使うだけでもこの2例の元gateを通る。したがって主な修正は**同一配列点・残差経路への統一**であり、longdoubleだけの効果と説明しない。拡張精度は、既に導入済みの残差/gradient縮約の精度をinnerにも揃える補助である。実H4での各寄与の割合は測っていない。

内部二経路の一致だけでは共通biasを排除できないので、独立80桁Decimalの積・係数微分も確認した。こちらはinvented n2/rank3/LD1、γ=0のtensor項単体で、binary64配列をexact入力とする人工参照である。

| 人工f* | 旧chart勾配対Decimal差norm | 新chart勾配対Decimal差norm |
| ---: | ---: | ---: |
| 1 | 3.46600e-17 | 2.46217e-20 |
| 1e-9 | 1.12846e-12 | 7.79171e-16 |
| 1e-12 | 3.56851e-11 | 2.46395e-14 |

PR penaltyを含む一般signed Jの方向FD、gradient→Hessian FD、既存HessianのW pullback、head gauge、列符号でdetを反転したUのdensity保持も検証した。新しいridge/符号/PSD制約、rank切捨てはない。既存SciPy Newton-CGへvg/hessを渡す人工試験ではFが低下し、終点の新整合gateが通ったが、実H4のoptimizer成功や同時一次資格は主張しない。

[人工study原本](../../artifacts/research_c_pr_coefficient_point/2026-11-02/technical/artificial_study.json)。Longdoubleの有効仮数はこの環境で64 bitであり、quad128ではない。新Jacobian/gradientの厳密enclosureや一般精度保証はなく、`rigorous_gradient_error_bound=null`を維持する。

## 5. 最小修正候補の内容と未接続範囲

`c_pr_coefficient_point.CoefficientChart`は既存W/x0を保持し、各zについてbinary64のxとJを一度decodeする。以降のF・残差・Pauli係数をそのJ配列に揃える。既存matrix写像の方向微分経路と、既存tensor adjointのoriginal-coordinate経路を同じ点で比較する。Physical gradientはWへの縮約後にbinary64へcastする。Hessianは既存解析式を同じ点で評価する。

固定gradient atol1e-8/rtol1e-7、value atol1e-10/rtol1e-8は不変。拡張精度が使えない、入力点/metricが違う、非有限値や整合FAILなら明示STOPする。偽の勾配を与える人工faultも拒否した。RC2はこのclassから明示的に除外し、旧gelsdやouter政策を変更しない。

旧`c_separated_j_u_consistent.solve_j`はclassを直接指定しているため、今回その固定SOURCEや旧paired runnerを差し替えていない。新classはstandaloneの修正候補として人工Newton接続と一つの保存点診断に用いる。**全paired経路が修正済みとする段階ではない。** 最小H4診断で意味を確認してから、必要なら別の技術作業でinner callerへ接続する。Joint J/ηへ全面移行する判断もしていない。

## 6. RC2：保存64更新の独立した技術診断

[保存ログ診断](../../artifacts/research_c_pr_coefficient_point/2026-11-02/technical/RC2_saved_log_diagnosis.json)はJSONの既存値を読み比較しただけで、新gradientやstepを配列から計算していない。

| 受理更新 | 保存F | 保存物理U₂ |
| ---: | ---: | ---: |
| 初期 | .02919382875 | .867029595 |
| 1 | .02862458549 | .210488068 |
| 8 | .02569388426 | .179591603 |
| 16 | .02222704826 | .033082972 |
| 32 | .02029232506 | .010538449 |
| 48 | .01815933167 | .008280116 |
| 64 | .01742887277 | .004762078 |

全64受理点の保存nonincrease gateはPASSで、最後の保存F減少も2.46556e-5と正だった。一方、保存U normは26更新で前の受理点より増加している。Fの降下とU勾配normの単調降下は同じ条件ではない。Solverはsuccess=false/status1、nit64/nfev75/njev75でiteration上限。これは直接の停止理由であり、モデル/更新方式が十分だから反復を増やせば収束するという証明ではない。

Outer step norm、line-search α、inverse-Hessianの十分なscalar記録はない。今回はη NPZからstepを作り直さず、これらをnullとした。強いcoupling、chartのconditioning、BFGS計量が収束速度を制限したかは未確定。RC2の上限変更や最適化continuationは今回実施・採択せず、必要なら別の独立契約で扱う。終端DF未評価の状態も変更しない。

## 7. 次の最小実H4検証：未承認10-call契約

[固定契約案](research_c_pr_coefficient_point_contract_unapproved_20261102.json)。対象は旧PR trial2の一つの保存inner終点だけ。新しいfit/inner solve/refinement/科学的最適化trajectoryは0で、RC2 continuationや1563-call paired再実行は含めない。

失敗点のouter Uは直接保存されていない。既存SciPy1.14.1 BFGSへ、保存query0/query1の値・勾配だけを渡し、第3queryを受け取った瞬間に例外でcaptureする。新しい科学的objective値は返さず、受理callbackが先に起きたらSTOPする。これは**明示的に1回のdry trace-replay driver**であり、「optimizer APIを一度も呼ばない」とは記載しない。保存η0/η1、query1 U、失敗時x0/W/xをexact array equalityで照合してから評価へ進む。人工replay/E2Eは通ったが、実H4でこのidentityが成立するかは未検証である。

| 科学logical unit | 最大 | 内容 |
| --- | ---: | --- |
| saved_load | 1 | 6 saved NPZ＋PR trace読込。新fitなし |
| outer_trace_replay | 1 | 保存2点だけのdry driver、第三query capture、受理0 |
| trial_geometry | 1 | 既存JAX x64のU0 exp(η/√2) |
| coefficient_setup | 1 | 既存A/T・cached Gram・可逆metricを構成 |
| point_identity | 1 | 保存点/metric/coordinateの一致。不一致ならSTOP |
| legacy_evaluation | 1 | 旧affineと独立経路の勾配差を記録 |
| stable_setup | 1 | 同じWを使う拡張精度写像の準備 |
| stable_evaluation | 1 | 新二経路の勾配整合＋旧physical objective一致 |
| classification | 1 | 保存点の数値整合だけを判定 |
| export | 1 | 計算済みvectors/result/STOP保存、数値再評価なし |
| **合計** | **10** | **Retry/resume0、新しいinner solve/fit0** |

旧FAILの再現は既知不一致を測る診断項目で、新しいcorrectness FAILを無視する許可ではない。再現しなければ`INCONCLUSIVE_LEGACY_MISMATCH_NOT_REPRODUCED`を保存してSTOPする。新gateでFAIL/非有限値/入力不一致なら部分証拠を保存してSTOPする。追加点、探索、fallback、閾値変更はしない。

PASSが示すのは、復元identityを通った一つの保存点における新旧objective整合と新微分経路の数値一致だけ。厳密gradient保証、inner一次収束、PRの性能、DF予算、後続paired実行許可ではない。既存constructor内のGram cacheは構成するが、独立したHessian/条件数sweep、sector/operator、PF/RTE/ground/native/shots/総費用は含まない。

SOURCE144ファイル、入力8ファイル（7 evidence＋既存fit lock）、ffsim sourceと依存versionを固定した。SOURCE manifest SHA256は `68a424484e78eab66dfc3c2949a7dd95b35d9ad16c9152aebc3066369a0d78ed`、契約SHA256は `ffd41ecc990939a2eb87e419de5951dd03904422fb24807492fa3fe32f0793a9`。Fresh outputは `artifacts/research_c_pr_coefficient_point/2026-11-02/run01`、今回未作成。実行前に実空き資源を確認し、CPU/RAM/wallの恣意的上限を設けない。共有環境・venvへ変更しない。

## 8. テスト・SOURCE・保全

**既存220＋新19＝239 unique pytest items PASS。** 内訳と実行logは[テスト証拠](../../artifacts/research_c_pr_coefficient_point/2026-11-02/technical/test_results.json)。人工E2Eでは正常10 units、保存点不一致5 units、偽の新勾配8 units、旧FAIL非再現6 unitsでのSTOPを検証した。これらは人工データ内のcallであり、実分子科学callではない。既存H2/H4/codec・reduced/point-semanticsの回帰も再利用した。CI/外部独立再現とは表現しない。

新SOURCEは以下の5ファイルである。

- `src/trottertracks/c_pr_coefficient_point.py`：opt-inの同一配列点chart。既存DF/Pauli/adjoint/metric/Hessianを再利用。
- `src/trottertracks/c_pr_coefficient_diagnostic.py`：既存Calls/STOP/exportと保存-only replayを接続する診断専用入口。
- `scripts/run_research_c_pr_coefficient_point.py`：既定はpassive、科学実行は明示`--execute`。
- `tests/management/test_research_c_pr_coefficient_point.py` / `test_research_c_pr_coefficient_diagnostic.py`：人工再現・独立精度・FD/故障/replay/E2E。

`c_j_u_stable*`、`c_j_u_precision`、既存inner/outer solver、ffsim/SciPy、旧contract・run・STOPは無変更。旧raw2378件（1879＋499）、旧SOURCE13組、旧科学manifest131 recordsと人工準備原本を保全する。今回は技術調査のみなので、旧科学qualification・validation manifest・研究概要・過去noteを上書きせず、必要な索引と本報告を追加する。

作業branch `codex/research-c-pr-gradient-consistency-preparation-20261102`、独立worktree `/tmp/prt-c-pr-gradient-consistency-preparation-20261102`。[preflight/保全確認](../../artifacts/research_c_pr_coefficient_point/2026-11-02/technical/preparation_postflight.json)と[技術STOP](../../artifacts/research_c_pr_coefficient_point/2026-11-02/technical/STOP.json)を保存する。

## 9. 次の担当・許可・mandatory STOP

**ユーザー／研究C**が、修正候補を実失敗点で検証する10-call診断の実行を判断する。**Codex**は別途明示許可後にだけ、その固定SOURCE/契約のfresh runを一度実行する。承認証明書・承認記録・新grantは不要。

今回の技術作業から実H4評価、paired再実行、RC2 continuation、Joint移行、詳細GPT科学レビューへは進まない。旧480-call STOPを維持し、残り1083 unitsも旧1051-call残枠も再利用しない。commit/push・remote SHA/tree・公開本文・SOURCE/保存原本を照合後、**mandatory STOP**する。
