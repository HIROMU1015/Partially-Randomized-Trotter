# 研究C：改善reduced J/U pilot・PR inner整合FAILで部分STOP

2026-11-02 JST。Repository: HIROMU1015/Partially-Randomized-Trotter。

## 1. 結果と結論の範囲

ユーザーの明示的な科学実行許可に基づき、科学SOURCE `12e0079a264dc0f5831791efc06c0aa159b88908` と[固定1563-call契約](research_c_j_u_stable_contract_unapproved_20261102.json)を変更せず、fresh run01でPR/RC2各1 trajectoryを実行した。**480/1563 logical unitsを試行し、478 COMPLETE・2 FAIL。Retry/resume0。PRの一つのinner correctness例外が、そのcallと包むouter driverの2件に記録されている。**

PRは初期point0と未受理trial1で新しいfinal-refinement-pairの数値admissionを通ったが、次の未受理trial2のinner J solveで`J_POINT_CONSISTENCY_MISMATCH`となった。実際に停止を決めたpredicateは`chart_vs_physical_gradient`である。**PRの受理U更新は0、outer最適化の有効性は未判定。**

開始済みの独立RC2 workerは固定64 outer iterationsまで完了し、75点すべてが新admissionを通った。自己Fは初期 .029193828754771826から終端 .017428872773088928へ低下したが、J∞3.67648e-8に対して物理U₂ .0047620784で、U一次閾値1e-5を満たさず`U_UNQUALIFIED`である。

親runnerはworker例外を受け、終端snapshot load・新DF sector norm・一般K/raw CAR・比較callへ進まなかった。**両armの新終端DF予算、qualified tail比較、PF/RTE/CA/量子native/shots/総資源はNOT_RUN/null。** Tensor残差の小ささをsector DF予算PASSへ代用しない。保存初期値のDF予算PASSは初期資格に限り再利用した。

[実行summary](../../artifacts/research_c_j_u_stability/2026-11-02/execution_summary.json)／[資格評価済み77点のprofile](../../artifacts/research_c_j_u_stability/2026-11-02/point_qualification_profile.json)／[生call・NPZ・trace・STOP原本](../../artifacts/research_c_j_u_stability/2026-11-02/run01/)／[保全postflight](../../artifacts/research_c_j_u_stability/2026-11-02/execution_postflight.json)。新summaryは保存JSONの投影・journal集計であり、STOP後の追加配列評価ではない。

## 2. 固定条件・SOURCE・preflight

| 固定物 | 条件／identity |
| --- | --- |
| 科学SOURCE | `12e0079a264dc0f5831791efc06c0aa159b88908`、139ファイル無変更 |
| 契約SHA256 | `c255a2bc553a7f3738ae92b632f727c0fead34acb46c64983a6600e00b0f2baf` |
| SOURCE manifest SHA256 | `1d22c3426742af2221822b6af7fa79d5ff7f9b095d4dffdf736c6475f9b2f509` |
| 分子／表現 | 保存直鎖H4、隣接1.00Å、STO-3G、通常number CDF8 |
| Sector | Nα=Nβ2、36次元 |
| J/U・split | 一般signed実対称J、実直交U、元frame順LD2/head0/1・tail2..7 |
| 元input SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| 元qualified J/U/x0 SHA256 | `78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856` |
| 目的 | PR smooth tail mass二乗γ=.1、RC2全ordered J L2 α=.1、ε=1e-8Ha |
| 固定scales | f*=2.1839632053132894e-9、M*=2.2093719370722074、P0=4.410884790420646 |
| 物理規約／予算 | 元g補正κ・scalar一回・spin/frame順、DF予算CA/8=.00019875Ha |
| 数値手順 | Binary64 J/U/solver、longdouble residual/adjoint（有効64 bit仮数）、既存JAX x64 Jacobian |
| Refinement | 各UでJ初回solve＋2 refinement、最後2点のmax(chart,physical)差≤元1e-6 |
| Outer | SciPy1.14.1 BFGS、各arm64 iterations/128 distinct points、既存physical一次閾値1e-5とmargin |
| Fresh | `artifacts/research_c_j_u_stability/2026-11-02/run01`、今回消費済み |

初期値は両armとも同じ元qualified snapshotで、旧最適化終点のwarm-startはしない。新しいridge/gauge固定/PSD/符号制約、tail tensor保存等式、rank/strength/scale変更はない。PFのδ・時間窓は今回の最適化に非適用。

[実行preflight](../../artifacts/research_c_j_u_stability/2026-11-02/execution_preflight.json)でSOURCE139、入力31、契約、旧raw1879、旧SOURCE12組と実環境を照合した。Fit環境はffsim0.0.83/NumPy1.26.4/SciPy1.14.1/JAX0.4.38、評価環境はQiskit1.3.0/OpenFermion1.6.1等の固定依存を維持。両専用Pythonは既存venv実パスを使用した。[同一SOURCEの220人工・回帰PASS](../../artifacts/research_c_j_u_stability/2026-11-02/technical/test_results.json)を再利用し、実行のための重複テストやSOURCE修正は行っていない。

契約の歴史的UNAPPROVED metadataは変更せず、チャットの実行許可に基づいて実行した。承認証明書・承認原文保存・新grantは作成していない。

```bash
# 実行履歴。run01は消費済みで、再起動しない。
PYTHONDONTWRITEBYTECODE=1 \
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_j_u_stable.py \
 --contract docs/research/research_c_j_u_stable_contract_unapproved_20261102.json \
 --execute \
 --output /tmp/prt-c-j-u-stable-execution-20261102/artifacts/research_c_j_u_stability/2026-11-02/run01 \
 --fit-python /tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python
```

## 3. PR：新stabilityは通ったがinner chart整合でSTOP

| PR point | 自己F | J∞（両precisionの大きい方） | 物理U₂ | 新final pair差 | 状態 |
| --- | ---: | ---: | ---: | ---: | --- |
| 0、初期U | .012267644429815034 | 1.3273234e-7 | 2.954975107 | 4.1721025e-10 | 数値admission PASS、U未達 |
| 1、未受理trial | 1.6466546577971732 | 3.2604647e-7 | 22.40703239 | 0 | 数値admission PASS、U未達 |
| 2、未受理trial | inner raw fun17.695576779649574 | 未資格・独立state未取得 | 未取得 | 未取得 | inner correctness FAIL |

Point0/1はPR共通head gauge/rank十分条件も数値PASS。Point0の旧coarse→first-refinement勾配差は5.6058350e-6のままであり、**旧38-call結果は引き続き未資格**。今回の4.1721e-10は、拡張精度評価＋追加refinementによる別規則の数値結果で、旧規則の遡及的PASSでも真のenvelope gradient誤差上界でもない。

Point1のrefinementは両方ともNewton-CG raw status2（precision loss）、nit0を返した。独立J一次条件と登録数値admissionは通るため、このraw flagだけでSTOPする契約ではない。同じ返却点の勾配差0は、真のinner解への近さを保証しない。改善案の限界が残る例として保全する。

Point2の既存Newton-CGはraw success=true/status0、nit945/nfev1594/njev1594/nhev945を返したが、整合gateは以下の結果だった。

| Inner整合検査 | 保存maxabs | 判定と停止上の意味 |
| --- | ---: | --- |
| raw jac対last-step-start | 0 | PASS、raw jacのpre-step意味の検査は通過 |
| raw jac対last-accepted/endpoint | 1.0972255e-7 | FAILだが、pre-step PASSとのOR条件なので単独の停止理由ではない |
| chart対独立physical-chain gradient | 2.3969496e-7 | **FAIL、最大allowance比23.9694、直接の停止predicate** |
| chart対独立physical objective | 1.7763568e-14 | PASS |
| raw fun対endpoint objective | 0 | PASS |

Gradient consistencyの元allowanceはatol1e-8/rtol1e-7のまま。旧修正前の「pre-step jacをendpointと誤認した検査」が直接再発したという説明はしない。今回pre-step意味は保存証拠で通り、別の必須検査が落ちている。座標変換、残差の丸め、強いconditioning等の寄与は、保存JSONだけでは確定できない。停止後にJ/U/transform配列をdecodeして再評価したり、検査を無効化・緩和して続行したりしていない。

PRのtraceにaccepted callbackは0件。Point1/2のobjective上昇を受理点の非増加違反とは扱わない。PRの`optimization.json`、終端native診断、final export、tensor/sector評価は未作成で、raw outer success/nit等もSciPy return前の例外のため取得不能/nullである。

## 4. RC2：新admission全75点PASS、64反復後もU未達

| RC2測定 | 初期point0（2 refinement後） | 終端point74 |
| --- | ---: | ---: |
| 自己F | .029193828754771826 | .017428872773088928 |
| Exact tail λ / Ha | 8.28582979616644 | 8.108752128251535 |
| J∞（両precisionの大きい方） | 5.9232572e-8、MET | 3.6764824e-8、MET |
| 物理U₂（両precisionの大きい方） | .867029595、NOT_MET | .0047620784、NOT_MET |
| 新final pair U差 | 4.8803791e-9 | 3.9846369e-9 |

全75点で`NUMERICAL_PILOT_ADMITTED`、225回のgelsd/correctionを完了し、受理U更新64件の非増加gateが通った。初期値と終端を比べたF/U残差は改善したが、両一次条件へ収束したという結果ではない。BFGS rawはsuccess=false/status1、`Maximum number of iterations has been exceeded.`、nit64/nfev75/njev75である。

最初のtrial1は新pair差4.6891355e-7で数値admissionを通り、旧coarse pair観測は7.2556790e-6だった。旧SOURCEのtrialと同一配列だとは主張せず、両規則の意味を区別する。新SOURCEで外側を進める数値的能力は確認できたが、厳密gradient上界は全点null。終端のRC2解析bound floating plug-inは約6838.49で、`UNRESOLVED_FLOATING_RESIDUAL_AND_ARITHMETIC`。保証に利用していない。

終端native parameter/codec診断はPASSし、実decode点でのJ-gradient差7.30546e-8は元native allowance1e-6/1e-7内、projector差1.30798e-15、decode tensor差Frobenius9.01257e-16。元物理candidate J/Uをdecode値へ置換していない。量子回路build/compileではない。

終端tensor Frobenius6.39982937e-11/max-entry1.25902622e-11は既存worker callで保存した。**親のDF sector norm/engineering guardは未実行**なので、これをDF予算PASS、一般K/raw CAR PASS、最終CAへ昇格させない。親比較callも未実施で、旧joint/分離/旧reduced終点との新比較をSTOP後に手動実行していない。

## 5. 科学call・実負担・停止範囲

| Call | Attempts | COMPLETE | FAIL |
| --- | ---: | ---: | ---: |
| 親snapshot/初期資格/context/physical | 4 | 4 | 0 |
| 両worker load/setup/identity/history | 8 | 8 | 0 |
| Inner J solve（refinement含む） | 232 | 231 | 1 |
| Precision state | 154 | 154 | 0 |
| Reduced admission | 77 | 77 | 0 |
| Outer driver（nested innerとは別） | 2 | 1 | 1 |
| RC2終端独立/tensor/export | 3 | 3 | 0 |
| **合計** | **480/1563** | **478** | **2** |

Parent4＋PR18＋RC2 458。PRの7 inner試行のうち6完了、RC2は225完了。New optimization trajectoryはPR/RC2各1で、ffsim joint fitter0。内側solve/勾配/資格検査をouter driver1件へ隠していない。生nfev/njev/nhev・VG/Hessian・rank・timingは[solver work集計](../../artifacts/research_c_j_u_stability/2026-11-02/solver_work_summary.json)と各trace/NPZに保存した。未返却metadataを架空の0へ補完しない。

PRでcorrectness例外が生じた時点でそのarmは停止した。独立に開始済みのRC2は既存executorの回収規則に従って完了し、新しいtask/候補は投入していない。親はworker回収後に例外を受け、依存する終端load/DF/表現/比較7 unitsを開始しなかった。親`failure.first_failure=null`はworker launchが親Calls外であるためで、正しさPASSを示さない。原因tracebackはPRの`failure.json`と二つのFAIL journalに保存されている。

| Process | Wall / 秒 | Process CPU / 秒 | Peak RSS / KiB |
| --- | ---: | ---: | ---: |
| Parent | 6.08099454 | 1.300744858 | 275892 |
| PR | 2.64132075 | 5.914896482 | 469980 |
| RC2 | 4.58941579 | 8.583030315 | 545436 |

実行前affinity32 CPU、空きRAM約56.1GB、CPU/memory pressure avg10=0。既存plannerで独立2 worker、tiny matrix BLAS1をJAX poolと協調した。SOURCEにあるmemory estimateはquotaでなく、恣意的なCPU/RAM/worker/wall上限は追加しない。共有venv・cache・全体設定・他ユーザーjobは変更していない。短いwall timeは早期FAILと固定上限を含み、旧方式に対する速度倍率ではない。

## 6. 取得できた判断と未取得

| 研究上の問い | 結果と限界 |
| --- | --- |
| 新しいreduced数値admissionは進めるか | PR初期/第一trial、RC2全75点は通過。真のreduced gradientへの保証は未取得 |
| PRの同時一次資格 | correctness FAILで停止、受理U更新0。初期J MET/U NOT_METは診断のみ |
| RC2の同時一次資格 | J MET、U NOT_MET。64 accepted updates後も未達 |
| U再増大を回避できたか | PRは未判定。RC2の初期対終端U残差は低下したが、一般的な回避・収束保証ではない |
| 旧分離型より自己F/λが改善したか | 親の保存終点比較call未実施。両arm qualified比較はnullで、手動の追加比較なし |
| DF予算維持 | 新終端sector評価NOT_RUN/null。旧/初期DF PASSを終端へ転用しない |
| 制限要因 | PRはinner chart/physical-chain consistency、RC2はouter64上限でのU一次未達 |
| 新方式の総合有効性 | 未確立。PR-aware DFの十分最適化・RC-DFに対する優位・PF/資源性能は主張できない |

次の最小の提案は、**Codexが人工条件と今回の保存JSONで、inner Jのchart/physical-chain整合を調査すること**。Raw jacのpre-step意味とは分け、座標pack/unpack、affine残差とdirect残差、gradient縮約の違いを検討する。これは次作業案で、今回FAIL後に実配列診断やSOURCE修正を実行したという意味ではない。実H4配列を使う新診断や再最適化、RC2のiteration増加には、範囲を定めた新SOURCE/契約と別途明示許可が必要である。

**ユーザー／研究C**が次の技術調査・限定科学診断の優先度を判断する。今回の未使用1083 units、旧1051-call残枠、旧runを継続/別条件へ転用しない。数値閾値を緩和せず、追加fit・joint fallback・新rank/strength探索・PF/RTE/ground/native回路/shots/総費用・重要科学レビューには進まない。

## 7. 保全・公開・mandatory STOP

実行branch `codex/research-c-j-u-stable-execution-20261102`、独立worktree `/tmp/prt-c-j-u-stable-execution-20261102`。SOURCE139/input31/契約・allowanceは無変更。旧raw1879、旧SOURCE/契約/grant/ledger/marker/STOP、技術準備の人工証拠を保全する。新raw499件に生solver/gradient/transform/η、call journal、failure、RC2診断export、resources、STOPを保存した。PRの未取得終端や新DF値を作らない。

研究概要・validation status・manifestへ部分FAILを追加し、dated noteは追記だけとする。旧報告/旧STOPは上書きしない。commit/push後、remote SHA/tree・公開本文・SOURCE/input・新raw/STOPを照合して**mandatory STOP**。Retry/resumeは0で、今回の承認から次の科学計算へ自動進行しない。
