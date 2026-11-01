# 研究C：Qualified-only Armijo限定実行・8 trialで受理なし

2026-11-02 JST。Repository: HIROMU1015/Partially-Randomized-Trotter。

## 1. 結果と結論の範囲

ユーザーの実行許可後、固定科学SOURCE `d03351ea00a9352b1d637e672c0dc679201b523a` と[60-call契約](research_c_pr_guarded_step_contract_unapproved_20261102.json)を変更せず、fresh run01へ一度だけ実行した。**60/60 logical units COMPLETE、correctness FAIL0、retry/resume0**。

結果は`GUARDED_TRIAL_LIMIT_NO_ACCEPT`。登録8 trialのうち**2点はreduced-gradient数値未資格、6点は数値資格を通ったがArmijo不成立**。受理U更新0、distinct U点9、J solve27（うちrefinement18）。追加のtrial/縮小/refinement/別solverは実行しない。

未資格gradientを使用せずに後続の登録stepを評価する機構は実H4でも動いたが、**登録した8点から資格付き降下stepを得られなかった**。これはbounded grid内の限定結果で、より小さいstepや別方向にも降下点がない証明ではない。Reduced方式全体のNO-GO、全域安定性、J/U同時収束、DF物理予算やPR性能の結論へ拡張しない。

[result](../../artifacts/research_c_pr_guarded_step/2026-11-02/run01/result.json)／[optimization](../../artifacts/research_c_pr_guarded_step/2026-11-02/run01/optimization.json)／[全trial・inner trace](../../artifacts/research_c_pr_guarded_step/2026-11-02/run01/optimization_trace.jsonl)／[summary](../../artifacts/research_c_pr_guarded_step/2026-11-02/execution_summary.json)／[STOP](../../artifacts/research_c_pr_guarded_step/2026-11-02/run01/STOP.json)。

## 2. 固定SOURCE・条件とpreflight

| 項目 | 条件 |
| --- | --- |
| 科学SOURCE | `d03351ea00a9352b1d637e672c0dc679201b523a`、SOURCE160無変更 |
| 契約SHA256 | `5407c96b715dd22c0332cc02b63c0ddb1c72aa791951acc6c64539879dc2073c` |
| SOURCE manifest SHA256 | `ce9073667a287714d8b66e3e2abcf0104a21b86a1e68567e8b84323f56773047` |
| 分子／表現 | 保存直鎖H4・隣接1.00Å・STO-3G、number CDF8、signed実対称J・実直交U |
| Split/sector規約 | 元frame順LD2、Nα=Nβ=2/sector36。今回sector演算はしない |
| κ/scalar | 元g補正κ・scalar・spin/frameを保持、数値再評価なし |
| 初期点 | 同じ元qualified J/U、eta=0。全distinct U点で元Jから開始し、初回solve＋2 refinement |
| PR目的 | γ=.1、ε1e-8Ha、f*=2.1839632053132894e-9、M*=2.2093719370722074 |
| Inner | 既存SciPy1.14.1 Newton-CG、maxiter1000/xtol1e-12、既存解析Hessian・同一点評価器 |
| 方向とstep | 資格付き初期chart g0からd=-g0、α0=min(1,1.01/||g0||)、α_i=α0/2^i、i=0..7 |
| Acceptance | 数値資格PASSかつF_trial≤F0+1e-4 α(g0·d)、F_trial<F0。受理≤1 |
| 数値条件 | 元J/U一次1e-5＋margin、gradient atol1e-8/rtol1e-7、value atol1e-10/rtol1e-8、latest-pair U差1e-6、元rank/nullspace guard |
| Fresh | `artifacts/research_c_pr_guarded_step/2026-11-02/run01`、今回消費済み |

[preflight](../../artifacts/research_c_pr_guarded_step/2026-11-02/execution_preflight.json)でSOURCE160/input4・契約・既存fit依存lock/ffsim SOURCE/version、旧raw2423/旧SOURCE16組とfresh run01を照合した。元g SHA1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db、元initial SHA78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856は不変。251人工・回帰PASSの同一SOURCEを再利用し、環境を変更していない。

承認はチャット許可に基づく。歴史的`unapproved` metadataを変更せず、承認証明書・原文保存・新grantを作成しない。今回の限定実行を、将来のouter政策の無条件採択や長期最適化の許可とは扱わない。

```bash
# 実行履歴。run01は消費済み、再起動しない。
PYTHONDONTWRITEBYTECODE=1 \
/tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python -B \
 scripts/run_research_c_pr_guarded_step.py \
 --contract docs/research/research_c_pr_guarded_step_contract_unapproved_20261102.json \
 --execute
```

## 3. 全8 trialの結果

初期F0=.012267644429815034、g0·d=-8.731877899141525、α0=.3417964492063899。以下のFは元と同じnormalized objectiveであり、Haのenergy誤差ではない。

| Trial i / point label | α | 最終F（未資格点は診断値のみ） | Armijo上限 | 数値資格／棄却理由 |
| --- | ---: | ---: | ---: | --- |
| 0 /1 | .341796449206 | 1.646654647446 | 未使用 | stability未資格 |
| 1 /2 | .170898224603 | 3.194460895037 | .012118418187 | 資格PASS、Armijo不成立 |
| 2 /3 | .085449112302 | 11.597902607183 | 未使用 | stability未資格 |
| 3 /4 | .042724556151 | 14.241736632357 | .012230337869 | 資格PASS、Armijo不成立 |
| 4 /5 | .021362278075 | 2.303257782715 | .012248991149 | 資格PASS、Armijo不成立 |
| 5 /6 | .010681139038 | .454478656936 | .012258317790 | 資格PASS、Armijo不成立 |
| 6 /7 | .005340569519 | .094187819484 | .012262981110 | 資格PASS、Armijo不成立 |
| 7 /8 | .002670284759 | .020543518022 | .012265312770 | 資格PASS、Armijo不成立 |

未資格point1/3について、探索decisionのobjectiveはnull、`gradient_used=false`を維持した。上表の診断Fは既存raw stateからの引用であり、資格を与えたりArmijo判定へ使ったりしていない。残る6点は実測した資格付きF/gradientだけで判定した。架空のpenalty/NaN/ゼロgradientは使用しない。

最小αのpoint8でもF=.020543518021918193はF0=.012267644429815034より高く、固定条件で受理できなかった。新たな9番目trial、α範囲拡張、方向変更は行わない。未受理trialのF上昇を、受理更新のnonincrease違反や性能悪化と読み替えない。

`accepted_labels=['0']`、`terminal_label='0'`、`last_trial_label='8'`。資格を通った初期点がterminalで、U更新を1回した意味ではない。`diagnostic_terminal.npz`は初期Uとその最終refined Jを保存する。未受理点を終点/warm-startへ置き換えず、各trialのJ/U・eta・勾配は独立に保存した。

## 4. 資格・pair診断・raw solver flags

全9点で最終inner Jと独立最終state JはMET、head rank/nullspace・tail injectivity・block/refinement nonincreaseはPASS。初期点＋6 trialが`NUMERICAL_PILOT_ADMITTED`、point1/3だけがlatest-pair stability未資格だった。

| Point | Physical pair U差 | Chart pair U差 | ΔJ Frobenius | 最終独立J∞ |
| --- | ---: | ---: | ---: | ---: |
| 0 初期 | 4.23692515e-10 | 4.23692515e-10 | 1.47877927e-17 | 7.60039191e-8 |
| 1 未資格 | 7.31089427e-6 | 7.29031643e-6 | 4.90976494e-13 | 1.18792628e-7 |
| 3 未資格 | 4.93620065e-4 | 4.93606701e-4 | 7.09652263e-12 | 5.70529565e-7 |
| 8 最小step | 5.59348484e-9 | 5.59348447e-9 | 4.27200334e-13 | 8.89315998e-8 |

未資格2点ではphysical差もchart差も1e-6を超えており、chart pullbackだけの未資格ではない。これらは同じU配列の既存refinement出力から測った登録診断である。ΔJには未分離のgauge成分と丸めがあり、ΔU勾配/ΔJの比をH_UJの測定や厳密感度上界にはしない。実H4の混合Hessian・quotient condition・gauge分解・真のgradient誤差は未測定のまま。

27 innerのraw内訳はsuccess/status0が15件、precision loss/status2が11件、maxiter/status1が1件。**全innerが正常収束/J基準PASSしたとは報告しない**。初回solveではpoint3がJ∞2.1368464（status1/nit1000）、point4が2.7832005e-5（status2/nit726）でNOT_METだったが、各点で登録済みの2 refinement後には最終J基準を通った。追加refinement/solver retryはない。初回J状態は観測し、最終refined Jが判定対象という固定規則を維持した。

合計nit6007、nfev11339/njev11334/nhev6018。Raw外の終点・点照合を含む実VG11420、Hessian6018、独立adjoint54、same-array binary64診断27を保存した。Logical unitを1数値評価として費用換算しない。

27 solveのchart/physical-chain整合差max2.2204460493e-16は元allowance内。独立stateの同一点value/微分・直交性検査も通過した。Raw jacは元のpre-step/accepted/endpointのOR規約で判定し、個別のFalse観測を上書きしていない。Correctness FAIL0は合成correctness判定を表し、すべての下位観測やreduced資格のPASSを意味しない。

初期terminalのJ∞7.6003919128e-8はMET、physical U₂2.9549751184はNOT_METで、J/U同時一次収束ではない。数値資格を通ったtrialのgも真のreduced gradientへのenclosureとは呼ばない。今回のouterはBFGSでなく登録済みの1方向step controllerで、outer BFGS起動0・outer raw=nullである。

## 5. Call・資源・保全

| Logical units | Attempts / COMPLETE |
| --- | ---: |
| saved_load / initial_identity / objective_setup | 各1 /1 |
| outer_driver | 1 /1 |
| inner_J_solve | 27 /27 |
| precision_state | 18 /18 |
| reduced_admission＋pair診断 | 9 /9 |
| classification / export | 各1 /1 |
| **合計** | **60 /60、FAIL0** |

Registered trials8、distinct U点9、初回J solve9＋refinement18、guarded controller1、受理U更新0。ffsim joint fitter/RC2継続/retry/resume0。Trialはこの新契約内の登録評価であり、旧18-callや旧未使用36 unitsの転用ではない。

実行前CPU affinity32、MemAvailable54861844KiB、CPU/memory pressure avg10=0を確認。逐次依存する方向/halving/innerは順序を保ち、既存vectorized NumPy/SciPy/JAX/longdoubleと小行列BLAS調整を使用した。CPU/RAM/worker/wallの恣意的上限や共有venv/cache/OS・他job変更はない。

保存resourceはwall25.7805732209秒、process CPU107.990513984秒、peak RSS454700KiB。Process CPUはimport/preflight等を含み、wall比をspeedupとは扱わない。Matched-budget benchmarkではない。

新raw72ファイルに全point/solve・trial decision・raw counters・SOURCE/契約・terminal・journal/resources/STOPを保存。[postflight](../../artifacts/research_c_pr_guarded_step/2026-11-02/execution_postflight.json)でSOURCE160/input4、旧raw2423/旧SOURCE、技術準備原本を照合する。研究概要・validation status・manifestに今回の限定結果を追加し、dated noteは追記のみ。

## 6. 未資格事項・次の判断・STOP

未資格trialを安全に棄却して登録探索を続けることは確認できたが、受理U更新は未達。今回の範囲では、stabilityだけを通せば前進できるわけではなく、資格付きtrialの降下条件も残っている。どの小stepで真のΦが降下するかや、数値inner精度がF/gへ与える誤差を今回の結果から保証しない。

DF予算CA/8=.00019875Haは維持したがsector operator誤差は未評価/null。PR性能・最終tail比較・PF/RTE/ground/native/shots/総資源は対象外。新Hessian/condition・追加stepを計算しない。

次は**ユーザー／研究C**が、局所step scale、初期方向微分とinner数値資格の関係をさらに切り分けるか判断する。候補は**Codex**による保存ログと人工問題での最小後続条件の準備で、単なるtrial上限拡張やJoint移行を自動採択しない。追加実H4評価には新しい固定範囲と別途明示許可が必要。

Branch `codex/research-c-pr-guarded-step-execution-20261102`、独立worktree `/tmp/prt-c-pr-guarded-step-execution-20261102`。Commit/push後、remote SHA/tree・公開報告/結果/SOURCE/契約/STOP・成果物を照合して**mandatory STOP**。今回の結果から追加科学計算へ自動進行しない。
