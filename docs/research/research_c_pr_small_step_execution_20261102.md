# 研究C：小stepによる1回の資格付きU更新・限定科学実行

2026-11-02 JST。固定科学SOURCE `a426d62f27182526baa9479cd7cf8a953f92a9dc` と[19-call契約](research_c_pr_small_step_contract_unapproved_20261102.json)を変更せず、fresh `artifacts/research_c_pr_small_step/2026-11-02/run01` で一度実行した。**19/19 COMPLETE、correctness FAIL/retry/resume 0。単一trialを独立終端確認後に1回だけU更新として受理した。** 公開照合後mandatory STOPする。

## 1. 対象・固定範囲

保存直鎖H4・隣接1.00Å・STO-3G、number CDF8・signed symmetric J/real orthogonal U・元frame順・LD=2、Nα=Nβ=2/sector36の入力規約。元qualified J/Uのeta=0を新たに評価し、γ=.1、ε=1e-8Ha、f*=2.1839632053132894e-9、M*=2.2093719370722074を維持した。元g補正κ・scalar・spin/frame順は変更せず、operator/sector normは今回評価しない。

Scope `C_PR_QUALIFIED_ONE_SMALL_U_UPDATE_V1`。契約SHA256 `fac4da5ded5d0c9d6cf899c65ffcd58a9f82b1dfb8066bca8ff71f7a7c4fa53d`、SOURCE manifest SHA256 `095f434a129529534fd184af4d5a4af8bc2cec34e0b7d6f2f8eacef63529cca8`。歴史的なunapproved filename/metadataは変更していない。

[技術準備](research_c_pr_small_step_preparation_20261102.md)の既存同一decoded J/Newton-CG/Hessian、元Jからsolve＋2 refinement、独立J/gradient検査、rank/nullspace、latest-pair stability、値尺度、raw/STOP保存をそのまま再利用した。旧49-callの診断点を受理点・warm startとして採用したのではなく、今回の初期点と1つのtrialを新たに計算した。

## 2. 受理規則と結果

[全optimization情報](../../artifacts/research_c_pr_small_step/2026-11-02/run01/optimization.json)／[実行集計](../../artifacts/research_c_pr_small_step/2026-11-02/execution_summary.json)。初期点が数値資格を通った後、候補方向d=-g0、s=g0ᵀd、αF=F0/(-s)、α=αF/8を計算し、正方向の1 trialのみを評価した。

| 項目 | 実測値 |
| --- | ---: |
| F0 | .012267644429815034 |
| 候補方向微分s | -8.731877899141525 |
| αF | .001404926245134638 |
| α（登録q=1/8） | .00017561578064182975 |
| Ftrial | .010792467936921136 |
| F0−Ftrial | .0014751764928938975 |
| Armijo bound（c1=1e-4） | .01226749108425966 |
| comparison_scale | 2.2267644429815037e-10 |
| Ftrial＋comparison_scale | .010792468159597581 |

`Ftrial+comparison_scale <= Armijo bound`かつ`Ftrial+comparison_scale < F0`を満たした。ここでFは正規化tensor lossとPR penaltyからなる自己目的関数であり、エネルギー誤差ではない。比較尺度は元のvalue allowanceと既知のrefinement/cross-path差から作る工程的な尺度で、未知inner biasを包む区間ではない。

Provisional passの後、**同一trial J/U/etaを既存stateで1回だけ独立再評価し、そのPASSを確認してから受理eventを記録した**。新しいJ solve/refinementを追加していない。保存trace上も`independent_terminal_assessment`が`accepted_U_update`より先である。

結果statusは **`QUALIFIED_SMALL_ARMIJO_STEP_ACCEPTED`**。Point0は初期状態、point1が今回受理したU更新、受理回数は1。`diagnostic_terminal.npz`は今回のpoint1のJ/Uを保持する。次の方向・trial・U更新・外部outer optimizer/BFGSは実施しなかった。

## 3. 数値資格・独立終端・raw状態

[結果](../../artifacts/research_c_pr_small_step/2026-11-02/run01/result.json)／[全trace](../../artifacts/research_c_pr_small_step/2026-11-02/run01/optimization_trace.jsonl)。終端値は受理直前に同一配列を独立再評価した結果。

| 項目 | 初期point0 | 受理point1 |
| --- | ---: | ---: |
| 自己目的F | .012267644429815034 | .010792467936921136 |
| 独立J∞（stable/binary64の大きい方） | 7.600391912793852e-8 | 1.222707730341721e-7 |
| J一次基準（元1e-5＋margin） | MET | MET |
| 物理座標partial U勾配₂ | 2.9549751184186754 | 2.669694288051786 |
| U一次基準（元1e-5＋margin） | NOT_MET | NOT_MET |
| Latest-pair U shift | 4.2369251510155467e-10 | 2.7856999425731587e-10 |
| Raw status solve/refine1/refine2 | 0/2/0 | 0/0/0 |

両点で最終inner J、独立J、PR rank/nullspace、full numeric rank、latest-pair stability、block/refinement nonincreaseの全admission gateがPASS。PR head quotient rank19/universal nullity1・tail injectivityを元のguardで評価し、stability1e-6・元のgradient/value allowanceを変更しなかった。

追加の独立終端確認は`INDEPENDENT_TERMINAL_J_MET`。再評価Fとtrial F、再評価chart gradientとtrial gradientの差はともに0、元allowance内でPASS。独立physical-U pullback整合差2.220446049250313e-16、直交性残差1.3322676295501878e-15。終端のstable J∞=7.27277681155054e-8、binary64 J∞=1.222707730341721e-7、stable physical U₂=2.669694288051786、binary64 U₂=2.6696942727672455を別保存している。

Raw Newton-CG6件はstatus0が5件、status2 precision lossが1件（初期refine1）。全6件でJ基準MET、chart/physical-chain比較はPASS、最大差1.2984818884506884e-19。すべて正常optimizer収束したとは表現しない。Raw jacの元pre-step OR endpoint規約を維持し、endpoint単独の不一致観測も削除していない。

U勾配は低下したが、**U一次条件は未達、J/U同時一次収束は未達**。J∞は初期より大きくなったが、登録閾値を満たしている。J/Uすべての勾配が改善したという主張ではない。独立再評価・latest-pair一致はいずれも真のreduced gradientの厳密誤差上界ではない。

## 4. 科学call・資源・証拠保全

[Call journal](../../artifacts/research_c_pr_small_step/2026-11-02/run01/science_calls.jsonl)／[STOP](../../artifacts/research_c_pr_small_step/2026-11-02/run01/STOP.json)。

| Logical units | 実績 |
| --- | ---: |
| saved_load / initial_identity / objective_setup | 各1＝3 |
| small_step_driver | 1 |
| inner_J_solve | 6 |
| precision_state | 4 |
| reduced_admission | 2 |
| independent_terminal_assessment | 1 |
| classification / export | 各1＝2 |
| **合計** | **19/19 COMPLETE、FAIL 0** |

2 distinct U点・1 trial・1受理、inner6（refinement4）、外部outer optimizer/BFGS 0、新ffsim DF fit 0、RC2継続/retry/resume 0。Inner nit合計1027、nfev/njev各1985、nhev1028、実VG2003/Hessian1028/independent adjoint12/binary64 same-array6を保存した。新ffsim fit0や外部optimizer0は、inner Newton-CGの反復0を意味しない。

[Preflight](../../artifacts/research_c_pr_small_step/2026-11-02/execution_preflight.json)は固定SOURCE170/input4・依存lock・ffsim SOURCE/version・fresh outputでPASS。準備時297 unique人工/回帰PASSを不変SOURCEの技術証拠として参照し、今回新たにテストやCI/外部再現を実施したとは主張しない。

実行前affinity32、load約1.07、MemAvailable約54.7GB、CPU/memory pressure avg10=0。既存vectorized NumPy/SciPy/JAX/longdoubleと小行列BLAS調整、task-local cacheを使用し、初期/trial/refinement/独立終端の逐次依存を維持した。恣意的なCPU/RAM/worker/wall上限、共有環境や他ユーザーjobへの変更はない。

[資源実績](../../artifacts/research_c_pr_small_step/2026-11-02/run01/resources.json)：runner wall5.910896秒、process CPU33.986175秒、peak RSS451528KiB。環境確認やpublication時間を含まず、speedup比較でもない。

[Postflight](../../artifacts/research_c_pr_small_step/2026-11-02/execution_postflight.json)でSOURCE170/input4、旧SOURCE18組＋今回170件、旧raw2554件、旧契約/49-call・60-call STOP、準備技術証拠を照合する。新raw25件とそのhashを保存した。SOURCE・入力・閾値は無変更、科学結果manifestの旧136件は保全して新1件を追加、当日noteは追記のみ。Runner終了後は計算済みJSON/JSONLの報告集計とhash照合だけで、NPZの追加数値decodeや科学queryは0。

## 5. 結論の範囲と次の担当

今回確認したのは、保存H4/1Å/STO-3G、number CDF8/LD2・γ=.1の元初期点から、登録された小step規則で**1回の数値資格付きU更新を受理できたこと**。旧診断の減少点を後付けで採用せず、cold initializationを含む既定経路で再評価し、値尺度を含むArmijoと独立終端確認を通した。

複数更新での安定性・収束性、真のreduced勾配精度、厳密な降下保証、未知inner解誤差は未資格。DF物理予算CA/8=.00019875Ha、PF/RTE/ground/native/shots/CA/総資源、PR性能優位は未評価/null。自己目的Fの低下をこれらの達成へ昇格させない。

次は**ユーザー／研究C**が、小step尺度を使う複数U更新の限定検証について技術準備へ進むか判断する。採択後の候補は**Codex**が、同じinner/J/rank/stability規則と独立終端確認を保つ更新回数・step初期化・不適格STOP条件を人工検証し、新しい固定範囲を準備する作業。今回の受理点から自動継続せず、追加実H4計算・RC2/Joint/paired再開・重要科学レビューを実施しない。

Branch `codex/research-c-pr-small-step-execution-20261102`。SOURCE/raw/call/STOP・報告をcommit/pushし、remote SHA/tree・公開成果物を照合する。**公開照合後mandatory STOP。**
