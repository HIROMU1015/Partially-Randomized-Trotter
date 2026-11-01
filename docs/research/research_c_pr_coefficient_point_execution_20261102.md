# 研究C：保存PR trial2・10-call診断で旧FAIL再現／新同一配列経路PASS

2026-11-02 JST。Repository: HIROMU1015/Partially-Randomized-Trotter。

## 1. 結果と結論の範囲

ユーザーの明示許可に基づき、固定科学SOURCE `565de95fc3ba4d19feaf25a17cbeecde69977a48` と[10-call契約](research_c_pr_coefficient_point_contract_unapproved_20261102.json)を変更せずfresh run01へ実行した。**10/10 logical units COMPLETE、correctness FAIL0、retry/resume0。**

保存済みquery0/query1だけを使うdry BFGS replayでquery2を復元し、登録した保存点identityをすべて通過した。その同じinner終点で、旧経路はgradient差2.3755165e-7・allowance比23.7551で不一致を再現。修正候補は差9.1846165e-18で元allowanceを満たした。旧physical objectiveとの差7.10543e-15も元allowance内である。

結果は`NEW_SAVED_POINT_NUMERICAL_CONSISTENCY_PASS`。**保存1点の数値整合の検証であり、PR optimizerの成功、J/U同時一次収束、DF予算、PR性能、厳密なgradient誤差上界を意味しない。** 新J solve/fit/最適化trajectory0、RC2 continuation0。1563-call paired実行や次のsolver接続へ自動進行しない。

[結果原本](../../artifacts/research_c_pr_coefficient_point/2026-11-02/run01/result.json)／[summary・call実績](../../artifacts/research_c_pr_coefficient_point/2026-11-02/execution_summary.json)／[STOP](../../artifacts/research_c_pr_coefficient_point/2026-11-02/run01/STOP.json)／[保全postflight](../../artifacts/research_c_pr_coefficient_point/2026-11-02/execution_postflight.json)。

## 2. 固定SOURCE・条件・事前確認

| 固定物 | 条件／identity |
| --- | --- |
| 科学SOURCE | `565de95fc3ba4d19feaf25a17cbeecde69977a48`、144ファイル無変更 |
| 契約SHA256 | `ffd41ecc990939a2eb87e419de5951dd03904422fb24807492fa3fe32f0793a9` |
| SOURCE manifest SHA256 | `68a424484e78eab66dfc3c2949a7dd95b35d9ad16c9152aebc3066369a0d78ed` |
| 対象 | 保存直鎖H4、隣接1.00Å、STO-3G、number CDF8、元順LD2 |
| Sector条件 | Nα=Nβ2/36次元を維持。今回sector operator評価はしない |
| 変数／目的 | 元の一般signed実対称J・実直交U、PR γ=.1、ε1e-8Ha |
| Scales | f*=2.1839632053132894e-9、M*=2.2093719370722074 |
| 物理規約 | 元g補正κ・scalar・spin/frame規約不変。今回は評価対象外 |
| 固定DF予算 | CA/8=.00019875Haを変更せず、今回の適格性判定は未実施 |
| 勾配allowance | atol1e-8/rtol1e-7 |
| 値allowance | atol1e-10/rtol1e-8 |
| 数値精度 | binary64入力・既存JAX x64、longdouble有効仮数64 bitの残差/adjoint縮約 |
| Fresh output | `artifacts/research_c_pr_coefficient_point/2026-11-02/run01`、今回消費済み |

[実行preflight](../../artifacts/research_c_pr_coefficient_point/2026-11-02/execution_preflight.json)で、SOURCE144/input8（7 evidence＋既存fit lock）、契約、固定ffsim source、依存環境、旧raw2378、旧SOURCE13組を照合した。ffsim0.0.83/NumPy1.26.4/SciPy1.14.1/JAX0.4.38等を保ち、既存専用venvを変更していない。[同一SOURCEの239人工・回帰PASS](../../artifacts/research_c_pr_coefficient_point/2026-11-02/technical/test_results.json)を再利用し、追加技術準備・重複テスト・詳細科学レビューを挟まなかった。

契約の歴史的UNAPPROVED metadataは変更せず、チャットの実行許可に基づいて実行した。承認証明書・原文保存・新grantは作成していない。

```bash
# 実行履歴。run01は消費済みで、再起動しない。
PYTHONDONTWRITEBYTECODE=1 MKL_NUM_THREADS=1 \
/tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python -B \
 scripts/run_research_c_pr_coefficient_point.py \
 --contract docs/research/research_c_pr_coefficient_point_contract_unapproved_20261102.json \
 --execute
```

## 3. 保存点の復元・identity

旧trial2のU/outer ηは直接保存されていないため、既存SciPy1.14.1 BFGSへ保存query0/query1のF/gradientだけを返すdry replayを1回使用した。第三queryを受け取った時点でcaptureし、新しい科学的objectiveは返さず、受理stepも0だった。新たな目的関数最適化trajectoryやinner solveではない。

[identity原本](../../artifacts/research_c_pr_coefficient_point/2026-11-02/run01/point_identity.json)は`SAVED_TRACE_AND_FAILED_POINT_IDENTITIES_PASS`。固定SOURCEの検査で、以下がexact array equalityを通った。

- Replayの最初の2 queryと保存η0/η1。
- 復元query1 Uと保存query1 U。
- 失敗inner点のx0、可逆metric W、x=x0+Wzと保存raw。

点やSVDの符号を後から揃えるfallback、近似identityへの緩和は行っていない。復元η/U、保存x/z、新旧比較に用いた診断vectorを[diagnostic.npz](../../artifacts/research_c_pr_coefficient_point/2026-11-02/run01/diagnostic.npz)へ保存した。新しくfitしたJ/Uのexportではない。

## 4. 旧経路・修正候補の比較

| 診断 | 保存値 | 判定 |
| --- | ---: | --- |
| 旧chart gradient対独立physical-chain差max | 2.3755165000e-7 | 旧不一致を再現、allowance比23.7551 |
| 修正候補chart対既存tensor-adjoint chain差max | 9.1846164751e-18 | PASS、元allowanceのまま |
| 旧chart objective | 17.695576779649574 | 元の評価経路 |
| 旧physical objective | 17.695576779649556 | 元の独立経路 |
| 修正候補objective | 17.695576779649564 | 同じ物理目的関数 |
| 新objective対旧physical差 | 7.1054273576e-15 | PASS、atol1e-10/rtol1e-8 |
| 厳密gradient誤差上界 | null | 未資格 |

旧480-call報告の勾配差2.3969496e-7と、今回2.3755165e-7は同じ数値ではない。勾配vectorのbitwise再現を主張せず、**座標点identityを通った上で、登録した旧勾配一致gateの不合格を再現した**と報告する。差の末尾の原因を確認する追加再計算はしていない。両者とも元allowanceを明確に超えるという判定は一致する。旧FAILを上書き・再分類しない。

修正候補は同じdecoded J配列と残差を両微分経路へ渡し、既存matrix写像による方向微分と既存tensor adjointのoriginal-coordinate微分を比較した。元のW、目的・PR gauge・off-diagonal規約・rankを変更せず、physical gradientのW縮約まで拡張精度を保つ。[人工原因切分け](research_c_pr_gradient_consistency_investigation_20261102.md)で支持された修正方針が、この保存失敗点でも元allowance内の数値整合を得た。

ただし二経路は同じprimal残差を使うため、9.18e-18を真のgradientへの誤差と読まない。実H4の独立高精度参照やvalidated enclosureは今回含めず、残差共通化と拡張精度の各寄与を実H4で個別に測ってもいない。新方式の全U領域の安定性、inner optimizerの進行、outer reduced-gradient資格、DF精度への影響は未検証である。

## 5. Call・資源・未実施範囲

| Logical unit | Attempts / COMPLETE |
| --- | ---: |
| saved_load | 1 / 1 |
| outer_trace_replay | 1 / 1 |
| trial_geometry | 1 / 1 |
| coefficient_setup | 1 / 1 |
| point_identity | 1 / 1 |
| legacy_evaluation | 1 / 1 |
| stable_setup | 1 / 1 |
| stable_evaluation | 1 / 1 |
| classification | 1 / 1 |
| export | 1 / 1 |
| **合計** | **10 / 10、FAIL0** |

旧経路の既知不一致は診断対象として記録したもので、callの実行例外ではない。新しいcorrectness gateや点identityのFAILは0。各内部処理は固定契約のcomposite logical unitに従い、exportで勾配を再計算していない。新J solve/fit/最適化trajectory、RC2 continuation、retry/resumeはすべて0。

保存resourcesはwall .555108139秒、process CPU5.99373765秒、peak RSS341760KiB。Process CPUはimport/preflight等を含む保存値で、phase wallとの比を速度倍率とは扱わない。実行前affinity32 CPU、空きRAM約52GiB、CPU/memory pressure avg10=0。単一の依存付き診断点なので、既存CPU NumPy/SciPy/JAX x64/longdouble経路を用い、小行列BLASの過剰threadを避けた。CPU/RAM/wallの人工上限や共有環境変更は追加していない。

新Hamiltonian/sector DF norm/一般K-CAR/PF/RTE/ground/native回路/合成誤差/shots/総費用は未実施。旧RC2未収束・旧PR受理U更新0・旧終端DF未評価の状態も変更しない。新方式のF/λ改善や同CA資源優位はこの診断から主張しない。

## 6. 次の最小作業とSTOP

今回の結果は、**同一配列点の数値評価器を既存PR inner solverへ接続する技術準備に進む根拠**になる。次の候補担当はCodexで、既存Newton-CGのvg/hess・raw jacの点意味・独立終端検査を保った最小接続と人工回帰を行い、その後の最小科学契約を設計する。これは次作業案であり、今回すでにsolverを差し替えたという意味ではない。

ユーザー／研究Cがその技術接続を採択する。今回の結果から1563-call paired実行、追加fit、RC2 continuation、Jointへの全面移行、詳細GPT科学レビューへ自動進行しない。新しい科学計算には固定した新範囲と別途明示許可が必要である。

実行branch `codex/research-c-pr-coefficient-point-execution-20261102`、独立worktree `/tmp/prt-c-pr-coefficient-point-execution-20261102`。SOURCE144/input8/契約・数値allowanceを保ち、旧raw2378と旧SOURCE/STOP、旧480-call結果と未使用1083 units、技術準備原本を保全する。新raw10ファイルへSOURCE/契約/identity/旧・新診断/生vectors/call/resources/STOPを保存した。

研究概要・validation status・manifestへこの限定数値結果を追加し、dated noteは追記だけとする。commit/push後、remote SHA/tree・公開本文・SOURCE/input・新raw/STOPを照合して**mandatory STOP**。旧run・残枠を再利用しない。
