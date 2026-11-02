# 研究C：PR-only reduced経路の限定検証準備

2026-11-02 JST。Repository: HIROMU1015/Partially-Randomized-Trotter。

## 1. 完了範囲と科学実行境界

起点 `73908270bad2d4ad76f12cb85013621b9a9900ff` から独立worktree/branchで、修正PR innerを使うreduced経路の限定runner・人工検証・未承認科学契約を準備した。**新規実H4 decode/数値再解析/科学call/fitは0**。将来のfresh run01は未作成で、公開照合後mandatory STOPする。

前段の[8-call結果](research_c_pr_inner_point_execution_20261102.md)は旧trial2 Uを固定した1 innerの限定結果である。J基準METとraw precision lossを保全し、この1点からouterの安定性・J/U同時収束・DF予算を推定しない。今回の契約案は**元のqualified J/Uから新規に開始**し、trial2 Uや旧終点をwarm-startに使わない。旧480/10/8-callのSOURCE・結果・STOPと未使用枠は再利用しない。

## 2. 再利用と最小差分

| 処理 | 再利用／今回の差分 |
| --- | --- |
| 同一点vg/Hessian・raw jac検査 | `c_pr_inner_point_solver.solve_j` / `c_pr_coefficient_point.CoefficientChart`をそのまま使用 |
| 初回solve＋2 refinement | 既存opt-in `PRReducedOracle`の3呼出しを再利用。旧Newton-CG/allowance/残差/座標変換を変更しない |
| 独立state | `c_j_u_stable.state`のsame-array extended/binary64 J/U・forward/pullback検査 |
| rank/nullspace | `c_j_u_reduced.applicability`。既知のU非依存head gaugeだけを認め、追加/曖昧nullは拒否 |
| 外側 | 既存と同じSciPy BFGS、固定original-U exponential chart、exact eta cache、callback受理規約 |
| 新core | `c_pr_reduced_limited.py`：既存oracleを継承し、admission内訳を追加。短いBFGS orchestrationはPR専用・限定範囲のため分離 |
| 新pilot/CLI | 保存original g/J/U・scales →限定outer →分類/診断export。旧paired parent、CAR/DF・native経路は起動しない |

旧defaultや固定SOURCE149ファイルへのpatch・runtime差替えはない。新規はcore/pilot/CLI/人工test2ファイルの5ファイルだけ。新分解法・optimizer・Gaussian/CAR・精度管理framework・承認証明基盤は作らない。

## 3. 判定を分離する

各distinct U点の順序は、**初回J solve →refine1 →state1 →refine2 →state2 →admission**。初回solveは毎回同じ元Jから開始する。2 refinementは直前Jを使い、既存条件のまま実施する。

1. **Correctness**：raw jacのpre-step/accepted/endpoint規約、独立chart/physical-chain value/gradient、same-array binary64 value、独立stateの微分検査を維持。不一致・非有限値・直交性・目的増加は例外STOP。生データは検査前に保存する。
2. **Raw optimizer flags**：3 innerすべてのsuccess/status/message/iteration/evaluation情報を保存。precision lossを正常収束と呼ばず、raw successだけで資格を決めない。
3. **J一次**：最終innerの独立J gateと最終precision stateのJ gateが両方MET。初回/1回目refinementのJ状態は別に保存する。従来同様、最終refined Jが判断対象である。
4. **Gradient stability**：最終2 refinementのphysicalおよびchart U勾配差の大きい方≤既存`1e-6`。最初のcoarse差も観測として保持する。
5. **構造**：元のPR head rank/nullspaceとtail injectivityの数値guardを維持。rank guard256・gap1024を変更しない。
6. **Admission**：上記J・stability・構造に加え、既存block/refinementのnonincrease検査を通過した場合だけ`NUMERICAL_PILOT_ADMITTED`。既存内側J状態も明示的に確認する。

`raw_solver_flags`、`inner_J_statuses`、`admission_gates`、`inherited_admission_status`を分けて保存する。J基準METだけでreduced資格を付与しない。これは最新pairに基づく**数値pilot**であり、真の勾配誤差enclosureや厳密なenvelope最適性の保証ではない。状態のpartial U勾配とqualified reduced-gradientも区別する。

## 4. 推奨する最小科学契約

[未承認54-call契約](research_c_pr_reduced_limited_contract_unapproved_20261102.json)。まだ実行していない。

| 項目 | 固定案 |
| --- | --- |
| 対象 | 保存直鎖H4・隣接1.00Å・STO-3G、number CDF8、元frame順LD2、Nα=Nβ=2 |
| 初期値 | 保存original qualified J/U、eta=0。毎回original J初期化、2 refinement |
| PR目的 | γ=.1、ε1e-8Ha、f*=2.1839632053132894e-9、M*=2.2093719370722074 |
| κ/scalar | 元g補正・scalarを保持。今回のpilotでは再評価しない |
| Inner | SciPy1.14.1 Newton-CG、解析Hessian、maxiter1000/xtol1e-12 |
| Outer | 同BFGS、maxiter2、gtol1e-7/norm2/xrtol0/c1=1e-4/c2=.9 |
| 限定規模 | 受理U更新≤2、distinct eta≤8（初期点込み） |
| J/U基準 | 元の1e-5とengineering margin。gradient atol1e-8/rtol1e-7、value atol1e-10/rtol1e-8 |
| Stability/構造 | final pair U差≤1e-6、元rank/nullspace guard、無変更 |
| 最大 | 24 inner（うちrefinement16）、1 outer、54 logical units |
| Fresh output | `artifacts/research_c_pr_reduced_limited/2026-11-02/run01`、未作成 |
| Retry/resume | 0、追加refinement/rank変更/別solver/fallbackなし |

8点/2更新は前進を観測するための科学範囲案で、CPU/RAM/wall capではない。全域安定性や同時一次収束を要求する実験ではなく、未資格・point上限で受理更新0になる結果も許容する。実行前にユーザーがこの固定範囲を別途明示許可する必要がある。

| Logical units | 上限 |
| --- | ---: |
| saved_load / initial_identity / objective_setup | 1ずつ＝3 |
| outer_driver | 1 |
| inner_J_solve | 3×8＝24 |
| precision_state | 2×8＝16 |
| reduced_admission | 8 |
| classification / export | 1ずつ＝2 |
| **合計** | **54** |

Inner unitはsolver内のVG/Hessian・raw検査を含む。実iteration/VG/Hessian回数も保存し、1 unitを1数値評価とは扱わない。キャッシュhitは新solveを計上しない。新ffsim joint fit・RC2継続0。κ/scalar、DF物理norm、CAR、PF/RTE、ground、native build/compile、shots、総費用は含めない。DF予算CA/8=.00019875Haは変更せず未判定/null。

## 5. STOP・終点と取得できる結論

- 初期pair未資格：BFGSを始めず、initial pairを診断保存。旧初期値へfallbackしない。
- Trial未資格：最初の未資格点でSTOP。最後の受理終点と未受理stop trialを別に保存する。
- 9点目の要求：計算前に`OUTER_LIMIT_UNQUALIFIED`。未受理trialを更新数へ入れない。
- 2更新または限定BFGS return：raw状態を保持。`ACCEPTED_U_ADVANCE_OBSERVED`は受理U更新≥1を表し、収束/性能優位ではない。
- Correctness FAIL：raw/trace・partial failure・call/STOPを保存して依存する後続を停止。追加exportで旧点へfallbackしない。
- 正常・未資格・FAILのいずれでも公開照合後mandatory STOP。

最初に確認するのは、修正inner＋2 refinementの整合性、rank/nullspaceとstabilityの数値資格、U更新の実受理、旧整合FAIL再発の有無である。評価点範囲を越えた安定性、PR/RC2比較、最終tail λ、DF物理予算やPR性能の結論は得ない。

## 6. 人工検証・環境・保全

[試験結果](../../artifacts/research_c_pr_reduced_limited/2026-11-02/technical/test_results.json)：**226 unique pytest items PASS（新26＋既存200）**。fit専用215、評価専用10に加え、実BFGS起動数・callback更新上限の仕上げ後に新26項目を全PASS確認した。重複25項目と先の39項目は加算しない。実データを開かない人工NPZ E2E、2/4軌道・rank8、全3 solve dispatch、FD/cache、raw precision lossと資格分離、rank/nullspace/J/stability未資格、raw gradient/value偽装、初期/受理後trialのFAIL、fresh outputとretry禁止を確認した。

最初の回帰batchは旧sector試験2件がfit環境のOpenFermion欠如で失敗した。既存の評価venvで両件をPASS確認し、環境を改変していない。test path誤記によるcollection0件もlogへ残した。最終適合batchは全PASSで、CI/外部再現とは表現しない。

[人工study](../../artifacts/research_c_pr_reduced_limited/2026-11-02/technical/artificial_study.json)はseed41のinvented tensorsのみ。受理2更新、point上限、extra-head-null未資格を区別する。人工の極小stability allowanceだけでは、pairが数値的に同一なら未資格にならない観測もそのまま保存し、検査を強制FAILさせたとは書かない。stability不一致時のSTOPは別の人工fault testで検証した。

[passive preflight](../../artifacts/research_c_pr_reduced_limited/2026-11-02/technical/passive_preflight.txt)はhash・metadata・package/ffsim SOURCE・拡張精度の確認だけでPASS。SOURCE154（旧149＋新5）、入力4、既存fit lock/環境を固定した。SOURCE manifest SHA256は`e261fe6a816cf8c4368a08d85598d6ea839ca7bbcf6b67265b44e6c255d92026`、契約SHA256は`f0b8eba5dba70a687156d9cf3f8f3cb55ffd34eaf4f599ad72757e66c4f54c61`。

[保全確認](../../artifacts/research_c_pr_reduced_limited/2026-11-02/technical/preparation_postflight.json)：旧raw2401・旧SOURCE15組をhash照合。旧科学inventory・研究概要・dated note・STOPは変更しない。今回の公開は技術準備で、新しい科学結果をmanifestへ追加しない。独立branchは`codex/research-c-pr-reduced-limited-preparation-20261102`。

将来実行時は実空き資源を確認し、依存付き1 trajectoryの既存vectorized NumPy/SciPy/JAX/longdouble CPU経路を使う。小行列BLASのthread調整は既存に合わせ、人工的なCPU/RAM/worker/wall上限や共有設定変更を設けない。独立armの並列化は今回PR-onlyなので非適用。速度比較・speedupは未評価。

## 7. 次の担当と許可

技術的接続と人工検証は完了。残る科学的制約はraw precision loss、J閾値近傍でのinner解精度、実H4でのrefinement安定性・構造資格・受理更新が未検証であること。これらをテストPASSで解消済みにしない。

次は**ユーザー／研究C**がこの54-call案を判断する。別途明示許可後に**Codex**が固定SOURCE/契約/入力とfresh outputを再照合し、PR-only限定実行を1回行い公開STOPする。今回その科学実行、RC2延長、Joint方式移行、詳細科学レビューを開始しない。
