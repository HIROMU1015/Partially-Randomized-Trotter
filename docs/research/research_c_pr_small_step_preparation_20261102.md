# 研究C：小stepによる最大1回の資格付きU更新試験・技術準備

2026-11-02 JST。起点 `662ef9656962545c9dbcb8f1face65cbb0ed8811`。**最小接続・人工/回帰297 unique PASS・未承認19-call契約の準備まで完了した。実H4の追加decode、inner solve、U更新、科学callは0。** 公開照合後mandatory STOPする。

## 1. 保存結果と今回の目的

[固定49-call結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/662ef9656962545c9dbcb8f1face65cbb0ed8811/docs/research/research_c_pr_local_direction_execution_20261102.md)では、保存直鎖H4・隣接1.00Å・STO-3G、number CDF8/元順LD2、γ=.1、元qualified J/Uについて、3段階の正方向小stepすべてで自己目的Fの低下を観測した。全7点は既存J/rank/nullspace/stabilityの数値資格内だったが、診断点のU受理は0で、真のreduced勾配誤差・厳密降下保証は未資格だった。

今回はこの報告と既存JSON/ソースを参照しただけで、保存NPZを新たにdecode・数値再解析していない。旧結果からの明示的な後続設計であり、独立held-out検証や未観測の性能改善とは扱わない。

## 2. 実装・既存資産の再利用

新規は以下の5ファイルのみ。旧165 SOURCE/default/runner/科学契約は無変更。

- `src/trottertracks/c_pr_small_step.py`：1 trialの提案、値尺度を含むArmijo、独立終端再評価、最大1回の受理。
- `src/trottertracks/c_pr_small_step_pilot.py`：元入力loader、preflight、call計上、raw/export/STOPの最小接続。
- `scripts/run_research_c_pr_small_step.py`：既定はpassive。実行は別許可後の`--execute`のみ。
- `tests/management/test_research_c_pr_small_step.py` / `test_research_c_pr_small_step_pilot.py`：人工unit/E2E。

`c_pr_guarded_step.Oracle`、`c_pr_reduced_limited`、`c_pr_inner_point_solver`の同一decoded J評価器/Newton-CG/Hessianをそのまま使用する。各etaで元Jから初回solve＋2回refinement、独立J/physical/chart state、PR head quotient rank/nullspace・tail injectivity、latest-pair stability、raw jacのpre-step OR endpoint規約、nonincreaseを維持する。

値尺度は`c_pr_local_direction.value_profile/comparison_scale`を再利用。追加の終端確認は既存`c_j_u_stable.state`を同一J/U/etaで1回だけ呼ぶ。新fitter、数値計算library、Gaussian/CAR、外部outer optimizer、認証・承認管理基盤を実装していない。

旧Armijo runnerの初期stepはSciPy初回問い合わせ尺度、最大8 trialsである。そこを改変・再起動せず、今回の小step・単一trial・独立終端確認という別scopeを独立entryにした。

## 3. 事前固定した受理規則の提案

[未承認19-call契約](research_c_pr_small_step_contract_unapproved_20261102.json)、scope `C_PR_QUALIFIED_ONE_SMALL_U_UPDATE_V1`。

初期点eta0=0を、新たに元の手順で評価する。保存49-callのtrial J/Uや最小F点を初期値・受理点として流用しない。初期点が数値資格を通った場合だけ、候補方向d=-g0、s=g0ᵀd、αF=F0/(-s)、**α=αF/8**を使う。新初期F0/g0から算出し、旧数値alphaを直接固定しない。

q=1/8は旧49-callで減少を観測した3尺度のうち最大の尺度である。今回の新提案として事前固定し、残り2尺度、負方向probe、backtracking、adaptiveな候補追加は含めない。αFは曲率boundや降下保証ではない。

試すのはeta0+αdの**1点だけ**。その点でも元Jからsolve＋2 refinementと全admissionを実施する。provisional passには、c1=1e-4を維持し、次の両方を要求する。

```
Ftrial + comparison_scale <= F0 + c1 * alpha * s
Ftrial + comparison_scale < F0
```

`comparison_scale`は元の`value_atol+value_rtol*max(|Fa|,|Fb|)`と、双方の既知のrefinement F変化・extended/binary64差・独立AD差の最大値。工程的に減少が明瞭な場合だけ通す規則であり、未知inner biasを含む厳密なF区間ではない。旧gradient/value/stabilityの閾値を緩めていない。

## 4. 受理・独立終端判定・未資格の分離

| 段階 | 判定と動作 |
| --- | --- |
| 初期数値資格 | 既存J一次・rank/nullspace・stability・correctnessを維持。未資格ならtrialなし |
| 初期F/方向 | F0<=0、zero方向、既にJ/U一次METはtrialなしで明示STOP。非有限値はcorrectness FAIL |
| 単一trial | 同じ既存資格を満たす点だけF比較へ進む。未資格なら追加探索なし |
| Armijo＋値尺度 | 不成立なら未受理。別alphaへのfallbackなし |
| 独立終端確認 | provisional pass後、同じJ/U/etaを既存stateで新たに1回評価。J/physical U/chart/binary64勾配・値・直交性・一致を保存。Jを再solveしない |
| 一致FAIL | 値/chart勾配/直交性等の不一致では依存処理STOP、部分証拠を保存、受理・終端exportなし |
| 独立J未資格 | U更新未受理。元Uの診断状態を残し、試行点は別証拠として保持 |
| 全受理条件PASS | 初めて`accepted_U_update`を記録し最大1回だけ採用。以後の方向更新・trial・BFGSは0 |

原子回路やHamiltonianの確認ではなく、自己目的関数の物理座標partial U微分の独立再評価である。U一次METをU更新受理の必須条件にはしない。U一次がNOT_METであっても降下stepを受理し得る一方、その未達を隠したりJ/U同時収束と呼んだりしない。

受理後の`result.json`には独立J/U一次判定、physical/chart勾配の数値state、元refinement/rank/stability資格、raw optimizer success/status/messageを分けて残す。Raw precision lossも保全し、raw successやJ MET単独でreduced資格を与えない。

`diagnostic_terminal.npz`は、実際に1回受理した場合だけtrial J/Uを保持する。未受理なら元Uのrefined initialを診断用に保持する。これは不適格なtrialから初期値へ新しくfitし直すfallbackではなく、trialを採用しない動作である。いずれもDF物理予算適格候補とは扱わない。

## 5. 将来の限定実行条件と科学call

保存H4/1Å/STO-3G、Nα=Nβ=2/sector36規約、number CDF8/元順LD2、γ=.1/ε=1e-8Ha、f*=2.1839632053132894e-9/M*=2.2093719370722074、元g補正κ・scalar・spin/frame順を保持。J/U一次閾値1e-5＋元margin、gradient atol1e-8/rtol1e-7、value atol1e-10/rtol1e-8、stability1e-6、元rank/guardも維持する。

| Logical units | 最大 |
| --- | ---: |
| saved_load / initial_identity / objective_setup | 各1＝3 |
| small_step_driver | 1 |
| inner_J_solve（初期＋trial、各3） | 6 |
| precision_state（各2） | 4 |
| reduced_admission（各1） | 2 |
| independent_terminal_assessment（provisional pass時のみ） | 1 |
| classification / export | 各1＝2 |
| **合計** | **19** |

初期数値未資格やno-trial STOPでは通常12 units、trial未資格/Armijo不成立で18、独立終端確認まで進めば19が上限。Correctness FAILでは依存処理を停止し、完了済み/失敗/未実施を区別する。

新ffsim DF fitは0だが、将来の許可された実行には**最大6回のinner Newton-CG**が含まれる。反復上限/xtolは既存の1000/1e-12、実nit/VG/Hessian等を記録し、optimizer反復0とは表現しない。外部outer optimizer/BFGS、RC2継続、新rank探索は0。最大2 distinct U点・4 refinement・1受理、retry/resume0。

Fresh outputは `artifacts/research_c_pr_small_step/2026-11-02/run01`。旧49/60-callの再実行・残枠転用はしない。Correctness FAILは部分証拠を保存してSTOP。未資格・降下不成立も単一trialで終了し、第二trial・追加refinementへ進まない。

DF物理予算CA/8=.00019875Haは変更せず**未評価/null**。CAR/sector operator/PF/RTE/ground/native/shots/CA/総費用、新たな性能比較、重要科学レビューは契約外。未知inner解誤差・真のreduced勾配誤差・厳密降下保証もnullを維持する。

## 6. 人工・回帰テストと環境確認

[集計](../../artifacts/research_c_pr_small_step/2026-11-02/technical/test_results.json)：**297 unique items PASS（新26＋既存271）**。Fit側287 PASS/10 deselected、評価側10 PASS/29 deselectedで重複なし。新26件の先行smoke PASSは重複計上していない。ローカル人工検証であり、CI/外部再現ではない。

Invented DFの実Newton-CG→refinement→独立終端state→受理をE2E確認した。未資格initial/trial、降下不成立、比較尺度に埋もれる減少、post J未資格、identity/非有限値/independent mismatchのSTOP、raw precision-loss flagとJ資格の分離、19-call/fresh/retry0、未受理時の初期状態保持も検証した。旧chart/gradient/Hessian/raw jac/codec/reduced/rank・人工sector parentの回帰も通した。

初回の広い回帰suiteでは、fit venvにテスト用threadpoolctlがなく収集5 errorsとなった。既存評価venvの3.6.0モジュールをtask-localなtest-only領域へコピーした。その後、人工sector parent10件はfit venvにopenfermionがないため環境不適合となり、既存の評価venvへ分離して全件PASS。両失敗log/XMLは技術履歴として保全した。テスト用環境分離だけを修正し、SOURCEや科学環境の依存lockを変更していない。

[Test command](../../artifacts/research_c_pr_small_step/2026-11-02/technical/test_commands.json)／[test-only依存の来歴](../../artifacts/research_c_pr_small_step/2026-11-02/technical/test_dependency.json)。Fit側は既存ffsim0.0.83/NumPy1.26.4/SciPy1.14.1/JAX0.4.38、評価側は既存Qiskit1.3系/OpenFermion環境をread-only再利用。新しい科学runnerにthreadpoolctlを要求していない。

[Passive preflight](../../artifacts/research_c_pr_small_step/2026-11-02/technical/passive_preflight.json)はSOURCE170/input4/既存依存lock/ffsim SOURCE/versionでPASS、実run01は未作成。SOURCE manifest SHA256は `095f434a129529534fd184af4d5a4af8bc2cec34e0b7d6f2f8eacef63529cca8`、契約SHA256は `fac4da5ded5d0c9d6cf899c65ffcd58a9f82b1dfb8066bca8ff71f7a7c4fa53d`。

人工テスト開始時のaffinity32、MemAvailable約54.7GB、load約1.27、CPU/memory pressure avg10=0を確認し、fit/評価の独立suiteを並列実行した。逐次依存するinner/refinement/独立終端確認の順序は維持し、既存vectorized NumPy/SciPy/JAXと小行列BLAS調整を再利用する。CPU/RAM/worker/wallの恣意的上限、共有venv/cache/設定、他ユーザーjobへの変更はない。

## 7. 保全・未資格事項・次の担当

[保全postflight](../../artifacts/research_c_pr_small_step/2026-11-02/technical/preparation_postflight.json)で、旧raw2554件・旧SOURCE18組・元SOURCE165件・旧49/60-call結果/契約/STOP・136科学recordsを保全する。今回の成果は技術準備だけで、科学結果manifest・研究概要・研究noteを変更していない。

技術的blockerは残っていない。実H4で今回の単一trialが数値資格・受理条件・独立終端確認を満たすかは未検証。成功しても1回の数値的U更新を示すだけで、J/U同時一次収束、真の勾配精度、DF予算、λ/PR性能・資源優位の確立ではない。

次は**ユーザー／研究C**が固定19-call契約の実行可否を判断する。別途明示許可後のみ**Codex**が固定SOURCE・入力・環境を再照合してfresh runを一度実行し、受理/未受理・J/U/raw状態を報告して公開STOPする。承認証明書・承認原文保存・新grantは不要。複数U更新・solver変更・RC2/Joint/paired比較は今回提案に含めない。

Branch `codex/research-c-pr-small-step-preparation-20261102`。Commit/push後、remote SHA/tree・公開SOURCE/契約/報告/テスト証拠を照合する。**mandatory STOP。未承認の実H4診断・U更新を開始しない。**
