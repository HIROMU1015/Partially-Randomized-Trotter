# 研究C：頑健性・有限最適化政策の統合準備（担当A）

2026-11-05 JST。技術準備のみ。実H4 fit・保存実配列の新解析・PF/RTE・shots・native compile・重要科学レビューは実施していない。新科学実行は統合契約案に対する別の明示許可を要する。

## 再利用する実装と原SOURCE

主実装は `beb8de5cd8b71f80bb69556f92299435ff431c3e` にある近接weighted ENTRY対PRの有限recipe、最適化科学SOURCEは `70a98adfb701a209db2c65ea60fb789ff62b1fe7`、実装SOURCEは `9e9fc8391590aeb7f9d36b76d90b5a25bbe2fe5e`。保存結果・SOURCE・契約・STOPの意味は元のままとする。収集main `546bca7ed2cece59b714b986a264d7b0b0871a01` には同名SOURCEの別版が混在し、`c_near_practical.py` / `c_near_stable.py` が通常srcに存在しなかった。収集版の同名ファイルを無条件に組み合わせず、固定SOURCEのclosureを使用する。

| 機能 | 既存実装 | 統合準備の差分 |
|---|---|---|
| PR/ENTRY目的、重み・smoothing | `c_pr_aware_df.py`, `c_near_objective.py` | 無変更。raw objectiveをarm間の勝敗にしない |
| J Newton-CG / 2 refinement / HVP | `c_pr_inner_point_solver.py`, `c_pr_coefficient_point.py` | 無変更。1 U点ごとcold J＋2 refinement、maxiter1000/xtol1e-12 |
| Uの独立接空間・数値admission | `c_j_u_precision.py`, `c_pr_guarded_step.py`, `c_near_stable.py` | 無変更。J/数値stability/物理予算を区別 |
| recentered bounded Armijo | `c_u_step_control.py` | 無変更。方向32 halving、c1=1e-4、収縮.5等を継承 |
| paired bookkeeping | `c_near_practical.run_arm` | task-local step/policy/max_points/arm_limitsの任意引数だけ追加。既定16更新/128trial/129点は保持 |
| 寸法・候補接続 | `f2a9baea` の `c_near_transfer.recipe_arm/objective_specs` | recipe_optionsを転送。新optimizerやHamiltonian再構成なし |
| tensor-only initializer restore | `c1_real_orthogonal_codec`, `c_pr_aware_df.fit`, `c_common_convergence.fit` | 既存gamma0 fitへの薄い接続。旧fitter/codec/installed packageを変更しない |
| 新adapter | `c_integrated_robustness.py` | 固定セル展開、U摂動、上記APIへの接続、16prefixの保存・計測 |

初期規約は元順の通常number CDF8・signed symmetric J・実直交U・LD2。κは元gから一度補正し、核scalar一回、tail identity位相を保持。PR/ENTRY共通の保存直鎖H4・隣接1.00 Å・STO-3G・電荷0/singlet・Nα=Nβ=2/sector36を用いる。最適化段階にはPF時間窓は非適用。下流共通anchorは別段のpartial S2/s1/r32/K0/δ=h=.1/m0..12である。

## 事前固定した最小要因

- **初期値3個**：C0は保存tensor-only CDF8のqualified neutral start。P1/P2はC0のUへ固定PCG64 seed2026110501/2026110502から右乗算exp(A)の摂動を加え、各1回、両目的に中立なtensor-only restoreを実行した点。Jは摂動時に保持し、全8frameを合わせたskew generator Frobenius normを.05 radに固定する。同じrestore点をPR/ENTRYに共有する。C0＋独立2方向により単一起点依存を調べるが、初期値分布を統計的に代表するとは主張しない。
- **強度3個**：relative initial penalty contributionを{.03,.1,.3}に固定。既存.1を中心に約3倍の下側・上側を置き、局所的な強度依存を調べる。結果後に範囲や強度を追加しない。PRのγとENTRYのβは各armの自己初期massで正規化し、その点での寄与が選んだ同じ値になるよう固定する。最終誤差、curvature、gradient負担の一致とは主張しない。
- **有限予算2個**：同一trajectoryの16受理更新prefixと最大64受理更新の終端。64は既存StepPolicyの既定水準、16は既存比較の更新数anchorであり、4倍の更新機会を調べる。total trial上限は両prefixとも512、U点513、各方向32halvingを共通にする。**16/512は旧16/128の再実行ではない。** この共通trial政策により16prefixを64軌道から正確に共有でき、二重最適化を避ける。16前に有限停止した場合は同じ最後の適格点を両budgetに保存し、停止理由を明示する。

開発側は3start×3strength×2arm＝18trajectory、最大36endpoint。移送側は開発で選定・凍結したmethod別strengthを各3startに使うため6trajectory、最大12endpoint。移送先を見た強度・seed・摂動幅・予算・solver政策の変更を認めない。

## tensor-only restoreの固定政策

既存`c_common_convergence.OPTIONS`をそのまま用いる：L-BFGS-B、ftol1e-15、gtol1e-6、maxcor10、maxls20、maxiter15000、maxfun15000、jac=True、callback=None、boundsなし。`c_common_convergence.fit(c_pr_aware_df, ..., gamma0 spec)`が最終hopのoptionsだけをprocess-localに渡す。1start1fit、warm restart/代替seed/追加fitなし。専用processを使い、既存factory/codecのprocess-local bindingsをfinally復元する。

摂動後・restore前は元HのDF予算に入るとは限らない。この段階は有限binary64・対称J・直交U・codec/tensor/表現correctnessを要求する。restore後には独立value/gradient照合とsector DF予算を要求する。生success=falseやtensor-only一次未達でも有限に終了し、独立correctness・物理予算を通れば中立な有限初期化として使用できる。success/停止/残差を隠さず保存する。restore後DF不適格は当該startの全セルを欠測にし、代替を増やさない。correctness FAILは依存処理STOP。

開発のtensor loss scale f*=2.1839632053132894e-9 Ha²は旧neutralの物理スケールを保持する。PR/ENTRYそれぞれのsmooth initial massは各paired restored startで一度だけ固定し、endpointで再正規化しない。移送側のf*は移送neutral C0から一度だけ求める正のtensor lossで、旧H4の数値を流用しない。exact zero/非有限のscaleは未解決として停止し、任意のfloorへ置換しない。

## 科学適格性・記録・候補選択

DF reserve=CA/8=.00019875 Ha（CA=.00159 Ha）。既存sector operator2-norm＋`sector_operation_scale_guard8_proposal_v1`を保持する。guardは数値pilotの工程尺度であり厳密enclosureではない。General-K/raw CAR max-entry1e-10、J∞/物理U接空間L2各1e-5、latest-pair stability1e-6、直交性1e-10、既存rank/nullspaceと値/勾配整合を継承する。U一次未達だけで有限適格DF候補を消さない。

各prefixはJ/U配列、元input/SOURCE、start seed、neutral restore、objective spec、accepted update数、停止理由、tensor誤差、sector誤差とguard、独立J/U一次残差、自己Fのtensor/penalty成分、exact λ、ENTRY envelope、相殺gapを保持する。16prefixは受理直前の独立assessmentと物理評価をそのまま再利用し、別の候補を作らない。高budget終端が数値不適格でも、既に適格な16prefixの診断・exportは保全する。

古典負担は各prefixまでのnfev/njev/nhev/nit、実VG/Hessian回数、inner/refinement数、棄却trial数、raw status別件数、thread CPU/wall、JAX/共有shape cacheの有無を保存する。親process CPUとthread CPUの違い、cache初回費用、並列重なりを併記する。同じ更新機会であり、同CPU時間・同FLOPs・同真の最適化誤差を保証する比較ではない。共通のwork軸で累積負担と性能の推移を記述し、未登録の同時間補間や追加fitで勝敗を変えない。

候補は各固定recipeで最後の独立・物理適格受理点。best-of-rejected-trialsにしない。各start/budgetのstrength選択は統合計画の共通条件付き総RZ training政策に従う。自己F、λ単独、PRだけに有利な資源指標で選ばない。選択後の下流評価には独立native cohortを用いる。入力hash・candidate J/U file hash・objective/start/budget/SOURCEを渡し、同じ値が偶然得られても別科学候補を無断で同一扱いしない。

## 科学callの上限と積算

1 U点に既存3 inner solve＋2 precision state＋1 reduced admissionを数える。AD内部演算、LAPACK個々の呼出し、VG回数をこのlogical unitへ重複加算せず、実数を別保存する。

| 1trajectoryの種別 | 最大units |
|---|---:|
| optimization driver | 1 |
| inner J solve | 3×513＝1539 |
| precision state | 2×513＝1026 |
| reduced admission | 513 |
| independent terminal | 64受理時＋最終1＝65 |
| physical assessment | refined initial1＋受理64＝65 |
| endpoint mass diagnostics | 16prefixと64terminal＝2 |
| endpoint export | 2 |
| **合計** | **3213** |

開発18trajectory＝57,834、移送6trajectory＝19,278。共通input/context/spec setup・neutral restore・downstream・統合集計は別に積算する。低高が同一点なら重複診断を再実行せず再利用し、最大値は減る。旧645/1629の未使用callや既存STOPを利用しない。

新規restored startは各5units：摂動（またはtarget neutral XDF initializer）1＋faithful codec1＋gamma0 fit1＋restore後physical1＋export1。初期VG、raw終端、独立terminal VGは既存fit1unit内の固定hookで記録し、その実評価回数を別保存する。開発P1/P2は10units、移送C0/P1/P2は15units。target H/g/scalar生成・context/spec共通処理はさらに別。

`neutral_restore`はcodec/fit/physicalの3unitをcall callbackで囲み、外側driverが摂動/XDF生成とexportを各1unitで囲む。`recipe_cell`は旧armのcall callbackを継承する。これは既存logical call accountingへの接続であり、新しい承認証明システムではない。

## 資源と所要時間の見通し

保存16更新paired runは32CPU affinityの環境で全体wall92.766s、PRのarm wall90.872s、ENTRY52.816s、実VG計数は40,805/42,103、Hessian20,037/20,924だった（保存報告の引用）。9paired組×4倍の更新機会という粗い比例モデルは最適化約56分を示すが、強度・初期値でline searchや内solve負担が変わるため実行時間の保証ではない。新tensor-only restoreの時刻は未測定で、追加費用は不明。native下流は別見積り。

実行前にCPU affinity/quota/空きRAM/GPU/pressureを確認し、独立start/strength/armを資源に応じて並列化する。JAX x64、既存vectorized tensor map/AD/HVP、shared immutable g/CAR contextとshape cacheを使う。arm内のrefinement/line searchは依存順を保つ。JAXとBLAS/OpenMPの多重thread競合を自job内で協調し、process-global hookを持つneutral fitterは専用processとする。CPU/RAM/worker/wallの人工capなし、共有環境・他job・installed package変更なし。速度向上率は未測定。

## 人工・回帰検証

新`tests/management/test_research_c_integrated_robustness.py`は9件：固定18/6セルと3213call算術、paired固定seed/非破壊摂動/直交性、自己初期penalty寄与一致、gamma0 fitterへの一回委譲とDF不適格、既存arm wrapperのtask-local policy、16prefix/早期停止共有、raw workload集計、高budget不適格時の適格16prefix保全、実ffsim/codec/最終hop optionsを通るgamma0 restore（optimizerはmock、反復0）。全て人工配列・mock optimizerのみ。

固定SOURCE closureをA専用worktreeに復元し、旧near practical/near objective/near stableも回帰した。fit環境では**新9＋旧40＝49PASS、6.98秒**。人工CAR worker1件はevaluation環境で別process実行し**1PASS、1.94秒**。合計50unique PASS。最初のテスト作成時にはmock raw status欠落を修正した。収集mainの古いcoefficient chart/Armijoと組み合わせた場合に旧回帰が失敗することも確認し、数値閾値を緩和せず元closureの組に戻した。これは実H4の失敗やretryではない。

```bash
PYTHONPATH=src:tests/management /tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python -B -m pytest -q -p no:cacheprovider tests/management/test_research_c_near_practical.py tests/management/test_research_c_near_objective.py tests/management/test_research_c_near_stable.py tests/management/test_research_c_integrated_robustness.py -k 'not actual_new_runner_CAR_worker'
PYTHONPATH=src:tests/management /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/management/test_research_c_near_practical.py -k actual_new_runner_CAR_worker
```

ローカル人工PASSは科学的性能・科学GO・CI・外部再現を示さない。A専用worktreeには回帰用の固定closure復元ファイルがあるが、統合へ渡す新規差分はadapter、新test、本書、およびrun_arm/recipe_armのtask-local引数だけである。旧SOURCEを新しい科学SOURCEに付け替えない。統合公開照合後mandatory STOP。次の担当はユーザー：統合案の科学条件と実行範囲を判断し、実行する場合だけ別の明示許可を与える。Codexはその許可前に実H4へ進まない。
