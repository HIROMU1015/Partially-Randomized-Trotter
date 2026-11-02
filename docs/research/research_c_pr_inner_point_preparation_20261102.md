# 研究C：PR inner solverへの同一配列評価器接続・人工検証・限定8-call準備

2026-11-02 JST。Repository: HIROMU1015/Partially-Randomized-Trotter。

## 1. 完了範囲と結論

**検証済みの同一decoded J評価器を、既存SciPy Newton-CGのobjective・gradient・Hessianへopt-inで接続した。** Raw jacのpre-step規約と独立終端勾配検査を保ち、reduced経路の初回J solveと2回のrefinementでも新adapterを使う接続を人工検証した。旧moduleへの実行時patchや旧defaultの差替えは行わない。

次の科学案はpaired1563-callを再実行せず、**保存trial2のUを固定し、元JからPR inner solveを1回だけ行う8-call契約**とした。これで旧停止Uにおけるinner経路全体の整合と独立J一次条件を確認する。Outer U更新、refinement、RC2 continuation、新DF sector normは含めない。実行には別途明示許可が必要で、今回は実行していない。

起点は10-call結果commit `0e120c13672dce34d95352f10403939caa04c277`。[固定診断報告](research_c_pr_coefficient_point_execution_20261102.md)の保存点identity PASS・旧差2.3755165e-7・新差9.1846165e-18は既存値の引用だけで、新たな保存配列評価ではない。この一点のPASSを厳密gradient保証やPR optimizer収束へ昇格させない。

**今回の実分子NPZ数値decode・新fit・gradient/Hessian/sector等の科学評価は0件。** 旧480-call/10-call、未使用枠、SOURCE/raw/STOPを保全する。新GPT詳細科学レビューやJointへの方針変更は行わない。

## 2. 既存資産との接続

| 要素 | 再利用・最小差分 |
| --- | --- |
| 物理目的・座標 | 既存PR smooth mass二乗、一般signed対称J、x=D upper(J)、同じhead/tail split |
| 計量 | `c_pr_coefficient_point.CoefficientChart`が既存可逆SVD W/x0を保持。方向切捨て・ridge・gauge固定なし |
| F/gradient/Hessian | 同classのvg/hessを既存SciPy `minimize(method='Newton-CG',jac=True,hess=...)`へ直接渡す |
| 既存制御 | maxiter1000、xtol1e-12、原value/gradient allowance・独立一次margin・signed nonincreaseを維持 |
| 原本 | `c_separated_j_u_consistent.py`、`c_j_u_stable.py`、`c_pr_coefficient_point.py`等の固定SOURCEを変更しない |
| Reduced接続 | 新`PRReducedOracle`で初回＋refine1＋refine2の3呼出し先だけを新PR solverへ接続。State/構造guard/final pair admissionは既存処理を再利用 |

旧inner関数とreduced evaluatorはchart/solve関数を直接指定しており、factory注入ができない。固定SOURCEを編集・monkeypatchせず、必要なSciPy呼出しと記録処理だけを新moduleに明示的にadapter化した。Optimizer、DF再構成、AD、Hessian数式を新規に作ってはいない。

新APIは`c_pr_inner_point_solver.solve_j`と`PRReducedOracle`。旧`solve_j`や`StableReducedOracle`/paired runnerのdefaultはそのままで、RC2入力は新APIが明示拒否する。新reduced classは既存の`fun`/`evaluate` interfaceを継承し、人工試験でSciPy BFGSへ接続した。旧paired entryを修正済みと扱わない。

## 3. Raw jac・独立勾配・停止条件

SciPy1.14.1 Newton-CGはraw jacが最後のstep前の点を表す場合があるため、raw x/z/J/jac/Wとlast-step-start/last-accepted座標を保存する。返却jacはそれらとendpointのいずれか正しい点との一致を要求し、endpointのJ一次条件は別途新鮮な独立計算値で判定する。Raw jacを終点勾配で上書きしない。

終端で新chart方向微分と既存tensor-adjointのoriginal-coordinate微分を比較する。値は同じdecoded J配列を使い、physical gradientは拡張精度のままWへ縮約する。加えて同じJ/U配列での既存binary64値とJ勾配を観測し、J一次判定は拡張精度・binary64の大きいnormに元marginを加えて1e-5以下かを確認する。参照評価のための追加pack/unpackは挟まない。

| 条件 | 維持する基準 |
| --- | --- |
| Gradient consistency | atol1e-8、rtol1e-7 |
| Objective consistency/nonincrease | atol1e-10、rtol1e-8 |
| J一次 | ∞norm＋元engineering margin≤1e-5 |
| Reduced final pair（人工接続のみ） | max(physical/chart差norm)≤元1e-6 |
| U直交性 | 元1e-10、polar補正なし |
| Raw flag | success/status/message/nit/nfev/njev/nhevをそのまま保存し、一次資格と独立判定 |

不正なraw jac、非有限値、独立微分/値不一致、非増加違反、入力異常は明示FAIL。未資格のinner/reduced pointでは既存の未資格STOPを維持し、旧solverへのfallback、別の点/条件、追加fitへ進まない。`rigorous_gradient_error_bound=null`を維持する。

## 4. 人工検証の結果と残る限界

[人工solver原本](../../artifacts/research_c_pr_inner_point/2026-11-02/technical/artificial_study.json)。以下はinvented g/J/Uだけであり、n4/rank8/LD2/f*が同じでも実H4のデータではない。

| 人工例 | Raw status / nit | F初期→終端 | 独立J∞ | Chart/physical差max |
| --- | --- | --- | ---: | ---: |
| n2/rank3/LD1、seed41、f*=1 | status2 precision loss /74 | .25424280→.04155968 | 8.38249e-10、MET | 1.14650e-20 |
| n4/rank8/LD2、seed41、f*=2.1839632053e-9 | status2 precision loss /397 | 1.6923704e9→34.83452475 | 8.49581e-7、MET | 4.01972e-18 |
| 同seed42 | status0 success /553 | 7.7000742e8→100.39297362 | 4.60013e-6、MET | 3.19474e-17 |

旧SOURCEで人工の整合FAILを生じるサイズ/seedでも、今回の全inner経路は新検査を満たした。一方、**raw precision lossは解消したとは言えない**。J一次がMETでもraw success=falseとなる例を保存し、すべて正常収束と報告しない。実H4での反復数、J一次、精度・安定性、DF予算への影響は未検証。

Reduced接続は各ηで新solver3回・既存precision state2回・admission1回を明示計上し、同じηのcacheで再solveしない。η±hの両側でJを再solveする有限差分、既存SciPy BFGSの1更新、構造未資格/J一次未達/point上限でのSTOPを確認した。旧moduleの関数objectも変更していない。これらは接続の人工検証で、実H4のouter最適化結果ではない。

既存239項目と新17項目、**計256 unique pytest items PASS**。新試験はvg/hessの全dispatch、実Newton-CG pre-step jac、悪いraw gradient/非有限endpoint、reduced FD/cache/line-search/未資格、人工NPZを読むpilotの正常8-callとidentity/inner/state故障STOPを含む。[テスト件数・log](../../artifacts/research_c_pr_inner_point/2026-11-02/technical/test_results.json)。従来のoff-diagonal/chain rule/Hessian FD・Decimal/codec・H2/H4回帰も再利用した。ローカル試験でCIや外部再現ではない。

## 5. 次の最小PR科学案：固定U2・1 inner solve、最大8 units

[未承認契約](research_c_pr_inner_point_contract_unapproved_20261102.json)。10-call診断でU2そのものが保存されたため、今回はdry BFGS replayを繰り返さず、その固定Uを再利用する。旧Newtonの失敗終点Jやzはwarm-startに使わず、同じ元qualified Jを既存isometric座標へencodeして開始する。

| 項目 | 固定案 |
| --- | --- |
| 分子・表現 | 保存直鎖H4/1.00Å/STO-3G、number CDF8、一般signed実対称J |
| U・frame | 保存trial2 Uを固定、実直交、元順LD2。新U更新0 |
| 目的 | PR γ=.1、ε1e-8Ha、f*=2.1839632053132894e-9、M*=2.2093719370722074 |
| 物理規約 | Nα=Nβ2/sector36、元g補正κ/scalar/spin/frameを維持。今回はsector評価しない |
| Inner | 修正adapterのSciPy Newton-CGを最大1回、maxiter1000/xtol1e-12。Refinement0 |
| Outer/RC2 | 新outer trajectory0、U更新0、RC2 continuation0、ffsim joint fitter0 |
| Guard | 前節の元gradient/value/一次/直交性条件を維持、拡張精度なしならSTOP |
| 予算 | DF予算CA/8=.00019875Haを変えず、今回の測定・PASS判定は未実施/null |
| Retry/resume | 0/0。旧480/10-callの枠やSTOPを再利用しない |
| Fresh | `artifacts/research_c_pr_inner_point/2026-11-02/run01`、今回未作成 |

| 科学logical unit | 最大 | 内容 |
| --- | ---: | --- |
| saved_load | 1 | 元g/Jと保存U、過去の点identity/resultを読込 |
| saved_point_identity | 1 | 保存資格の再利用、J/U schema・直交性 |
| objective_setup | 1 | 同じPR目的と既存係数写像/通常cache |
| inner_solver | 1 | Newton-CGを1回。生rawと独立終端/point検査を保存 |
| independent_state | 1 | 既存same-array J一次・partial U微分診断 |
| tensor_metrics | 1 | 既存Frobenius/max-entry残差、sector normではない |
| classification | 1 | Correctness、raw flag、独立J資格を分離 |
| export | 1 | 計算済みJ/U/gradientの診断保存、再評価なし |
| **合計** | **8** | **係数optimizer最大1回、refinement/outer0** |

契約のpolicyは共通DEFAULTのsnapshotであり、outer/inner_refinements等の未使用fieldはreduced APIとの互換記録に限る。このpilotの実際の範囲はinner1回・refinement0・outer/U更新0である。

Inner1 unitには多くのsolver iterationsやVG/Hessian評価が含まれるため、nit/nfev/njev/nhevだけでなく実VG/Hessian/独立adjoint/binary64診断回数・wall/CPUを別に保存する。8 unitsを8回の目的関数評価とは扱わない。呼出し回数を根拠なく旧1563-callから流用したものではない。

Correctnessが通った上で、innerと既存独立stateのJ gateが両方METなら`INNER_CORRECTNESS_PASS_J_MET`、そうでなければ`INNER_CORRECTNESS_PASS_J_UNQUALIFIED`。Raw successやU部分勾配は別フィールドに残す。新しいU探索をしていないため、U一次未達はこの固定U試験のcorrectness FAILではない。Partial U gradientをqualified envelope gradientとも呼ばない。

この実験が通っても、「旧停止Uの固定係数問題で修正innerが検査を満たした」という範囲だけである。新しいreduced outerが旧BFGSと同じquery順序を辿ることや、全trialで安定・収束することは証明しない。DF sector norm/一般K-CAR/PF/RTE/ground/native回路/shots/総費用も対象外で、診断endpointをDF-qualified候補として下流へ渡さない。

SOURCE149ファイル、入力6ファイル、既存fit依存lockとffsim sourceを固定した。SOURCE manifest SHA256は `ed011ca32bd1b5b0da2cc684746353c1e89ffee262a85ce9d61e414b2c9dcc11`、契約SHA256は `9c4dce1dd21a5763ec855633febf828bcf7db3a738aa1565eace4939ae2d92e0`。固定入力は元g/J NPZ、成功した10-call診断のU/result/identity、既存fit lockである。実行時は利用可能CPU/RAMを確認し、単一の依存付き固定U問題に既存SciPy/NumPy/JAX/longdoubleを使う。CPU/RAM/worker/wallの恣意的上限や共有環境変更は加えない。

## 6. 新SOURCE・保全・公開・STOP

新SOURCEは5ファイルだけである。

- `src/trottertracks/c_pr_inner_point_solver.py`：PR Newton-CG adapterとopt-in reduced class。
- `src/trottertracks/c_pr_inner_point_pilot.py`：保存U固定・1 solveだけの独立pilot。既存Calls/STOP/export/state/tensorを再利用。
- `scripts/run_research_c_pr_inner_point.py`：既定passive preflight、科学実行は別許可後のみ。
- `tests/management/test_research_c_pr_inner_point_solver.py` / `test_research_c_pr_inner_point_pilot.py`：人工接続・故障・回帰。

旧SOURCE、旧480-call/10-callの原本・STOP、旧raw2388件、旧SOURCE14組、現在の132科学recordsを保全する。今回は技術準備なので科学的qualificationやvalidation manifest・研究概要・過去noteを変更せず、必要な入口索引と技術報告だけを加える。[保全postflight](../../artifacts/research_c_pr_inner_point/2026-11-02/technical/preparation_postflight.json)／[技術STOP](../../artifacts/research_c_pr_inner_point/2026-11-02/technical/STOP.json)。

作業branchは `codex/research-c-pr-inner-point-preparation-20261102`、独立worktreeは `/tmp/prt-c-pr-inner-point-preparation-20261102`。Commit/push後、remote SHA/tree・公開本文・SOURCE/契約/原本を照合して**mandatory STOP**する。

次の担当は**ユーザー／研究C**：この8-call PR-only案を採択し、実行を明示許可するか判断する。**Codex**は別途明示許可後にのみfixed-U2の1 solveを実行し、correctness・raw状態・独立J一次条件を報告する。今回の技術作業で実H4評価、paired再実行、RC2上限変更、Joint移行や重要科学レビューへ進まない。
