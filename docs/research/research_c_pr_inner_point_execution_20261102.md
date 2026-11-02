# 研究C：PR固定trial2 U・修正inner1回、正しさ/J一次PASS・raw precision loss

2026-11-02 JST。Repository: HIROMU1015/Partially-Randomized-Trotter。

## 1. 結果と結論の範囲

ユーザーの明示許可後、固定科学SOURCE `ab5d80bd64eeb82bab9dd8f11ef0a6c7c9a838bf` と[8-call契約](research_c_pr_inner_point_contract_unapproved_20261102.json)を変更せずfresh run01へ実行した。**PR inner solver呼出し1回、8/8 logical units COMPLETE、correctness FAIL0、refinement/U更新/outer最適化/RC2継続/retry/resumeはすべて0。**

保存trial2 Uを固定し、同じ元qualified Jから新同一配列評価器を使うNewton-CGを実行した。終端chart/physical-chain勾配差1.05214e-17は元allowance内で、独立J∞9.8354345e-6も元margin込みで閾値1e-5を満たした。結果は`INNER_CORRECTNESS_PASS_J_MET`。

ただし**Newton-CGのraw success=false、status2（precision loss）、nit708**である。正しさの検査と登録した数値J一次条件を満たしたことを、optimizerの正常収束や厳密な最適性と読み替えない。J一次残差は閾値に近く、より強い精度・汎用安定性を示す結果でもない。

この実行は、旧停止Uで修正innerの全呼出し経路が検査を通ることを確認した。旧BFGSの全trajectoryを再実行したものではなく、reduced gradient資格、J/U同時一次収束、DF予算、PR性能や資源優位は未評価/nullである。

[result原本](../../artifacts/research_c_pr_inner_point/2026-11-02/run01/result.json)／[summary](../../artifacts/research_c_pr_inner_point/2026-11-02/execution_summary.json)／[optimizer生metadata](../../artifacts/research_c_pr_inner_point/2026-11-02/run01/optimizer.json)／[STOP](../../artifacts/research_c_pr_inner_point/2026-11-02/run01/STOP.json)／[保全postflight](../../artifacts/research_c_pr_inner_point/2026-11-02/execution_postflight.json)。

## 2. 固定SOURCE・条件とpreflight

| 固定物 | 条件／identity |
| --- | --- |
| 科学SOURCE | `ab5d80bd64eeb82bab9dd8f11ef0a6c7c9a838bf`、149ファイル無変更 |
| 契約SHA256 | `9c4dce1dd21a5763ec855633febf828bcf7db3a738aa1565eace4939ae2d92e0` |
| SOURCE manifest SHA256 | `ed011ca32bd1b5b0da2cc684746353c1e89ffee262a85ce9d61e414b2c9dcc11` |
| 分子／表現 | 保存直鎖H4・隣接1.00Å・STO-3G、number CDF8、一般signed実対称J |
| U／split | 保存trial2の実直交Uを固定、元frame順LD2/head0/1・tail2..7 |
| Sector規約 | Nα=Nβ2/36次元を維持、今回はsector operator評価しない |
| PR目的 | γ=.1、ε1e-8Ha、f*=2.1839632053132894e-9、M*=2.2093719370722074 |
| 物理規約 | 元g補正κ・scalar・spin/frameを保持。今回はκ/scalarの数値再評価なし |
| 元g/Jの入力 | 元input SHA1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db、元initial SHA78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856 |
| 保存U2入力 | 成功10-call診断の`diagnostic.npz`、SHA8bd146df4bb19a54251f215950e97d601cd0ff8a52bc778d66da7e3c3cef1d6c |
| Optimizer | 既存SciPy1.14.1 Newton-CG、解析Hessian、maxiter1000・xtol1e-12 |
| 数値allowance | gradient atol1e-8/rtol1e-7、value atol1e-10/rtol1e-8、J∞1e-5＋元margin規則 |
| Fresh | `artifacts/research_c_pr_inner_point/2026-11-02/run01`、今回消費済み |

DF予算CA/8=.00019875Haは変更せず、今回の測定・PASS判定はしない。PF時間窓はこの固定U係数最適化に非適用。旧失敗終点J/zをwarm-startにせず、同じ元qualified Jを既存isometric座標へencodeして開始した。新しい制約、ridge、rank探索、U更新、gauge固定はない。

[実行preflight](../../artifacts/research_c_pr_inner_point/2026-11-02/execution_preflight.json)でSOURCE149/input6、契約、固定依存・ffsim source、旧raw2388/旧SOURCE14組、fresh outputと実空き資源を照合した。ffsim0.0.83/NumPy1.26.4/SciPy1.14.1/JAX0.4.38等の既存専用環境を再利用し、変更していない。[同一SOURCEの256人工・回帰PASS](../../artifacts/research_c_pr_inner_point/2026-11-02/technical/test_results.json)も再利用し、新しい科学実行の前に不要な追加準備・レビュー・重複テストを挟まなかった。

契約の歴史的UNAPPROVED metadataは変更せず、チャット許可に基づいて実行した。承認証明書・承認原文保存・新grantは作成していない。

```bash
# 実行履歴。run01は消費済みで、再起動しない。
PYTHONDONTWRITEBYTECODE=1 \
/tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python -B \
 scripts/run_research_c_pr_inner_point.py \
 --contract docs/research/research_c_pr_inner_point_contract_unapproved_20261102.json \
 --execute
```

## 3. Newton-CGの全経路とraw jac検査

| 保存optimizer情報 | 値 |
| --- | --- |
| Method | `PR_Newton_CG_same_decoded_point_v1` |
| Raw success/status | false /2 |
| Raw message | `Warning: Desired error not necessarily achieved due to precision loss.` |
| nit/nfev/njev/nhev | 708 /1319 /1319 /709 |
| 実VG/Hessian呼出し | 1322 /709 |
| 独立adjoint／same-array binary64診断 | 2 /1 |
| 目的F（同じ固定U2） | 5822231.60178163 →17.695576779649564 |
| Signed nonincrease | 元allowance内でPASS |
| 計量方向削除／追加正則化 | 0 /なし |

実VG回数はsolver nfevにfresh endpoint・raw jac点照合等を加えたもので、inner1 logical unitへこれらを隠して費用を1評価と扱わない。生x/J/z/jac/W、gradient0、last-step-start/last-accepted座標は[inner_raw.npz](../../artifacts/research_c_pr_inner_point/2026-11-02/run01/inner_raw.npz)、各iterationと診断は[inner_trace.jsonl](../../artifacts/research_c_pr_inner_point/2026-11-02/run01/inner_trace.jsonl)へ保存した。

| Raw/独立整合チェック | 差max | 結果 |
| --- | ---: | --- |
| raw jac対last-step-start | 9.3691086e-6 | FAIL観測 |
| raw jac対last-accepted | 0 | PASS |
| raw jac対endpoint | 0 | PASS |
| chart対独立physical-chain gradient | 1.0521414e-17 | PASS |
| chart対独立physical value | 0 | PASS |
| raw fun対endpoint value | 0 | PASS |
| same-array binary64 value | 1.4210855e-14 | PASS |

Raw jacにはpre-stepだけでなくaccepted/endpointの意味を持つ場合もあり、固定規則はこの3点のいずれかに一致することを要求する。今回はaccepted/endpointが通るため、last-step-startのFAIL観測だけでは停止しない。**すべての下位チェックがPASSしたとは書かず、登録された合成correctness gateがPASSした**と報告する。Raw jacやFalse観測を上書きしていない。

旧停止点でのchart/physical不一致を起こさず、修正したvg/hessを使うinner全体が終端まで進んだことは確認できた。ただしprecision lossの原因を新たに数値診断したり、追加refinement/別solver/閾値緩和で解消したりしていない。Raw status2はそのまま残る。

## 4. 独立J一次条件・partial U・tensor

| 独立測定 | 結果 |
| --- | ---: |
| Extended J∞（独立state） | 9.7155189402e-6 |
| Binary64 J∞（同じ配列点） | 9.8354344622e-6 |
| 一次判定に使う大きい方 | 9.8354344622e-6 |
| 元engineering margin | 1.0001e-8 |
| 数値J一次条件 | MET、margin込みで≤1e-5 |
| Partial U勾配₂ | 1204.2512757342、NOT_MET |
| U直交性max | 6.66134e-16 |
| Tensor Frobenius/max-entry | 9.6733762e-7 /2.2074432e-7 |
| Actual rank | 8 |
| Reduced gradient資格／DF予算／PR性能 | null／null／null |

独立stateは既存NumPy/JAXの同一点value・forward/pullback検査もPASSした。Inner solver内のJ gateと独立stateのJ gateが両方METのため、登録された`INNER_CORRECTNESS_PASS_J_MET`となった。**閾値に近い工程的数値基準の達成**であり、厳密stationarityや十分なinner解精度の保証ではない。

Uは元の未受理trial2の値を固定したので、partial U勾配が大きくても新しいU更新の失敗とは解釈しない。J/U同時一次条件は満たしておらず、このpartial derivativeをqualified envelope gradientと呼ばない。F低下も同じ固定Uにおける係数最適化のbefore/afterに限る。初期Uでの旧最適化候補や異なる目的のRC2と性能順位を付けない。

Tensor残差は登録callで測定したが、新sector operator norm/一般K-CARは測っていないため、DF予算PASSへ代用しない。κ/scalarの変更や新Hamiltonian生成も行わず、診断endpointを下流のDF-qualified候補として採用しない。[diagnostic_endpoint.npz](../../artifacts/research_c_pr_inner_point/2026-11-02/run01/diagnostic_endpoint.npz)はこの限定診断の保存物である。

## 5. Call・計算資源と未実施範囲

| Logical unit | Attempts / COMPLETE |
| --- | ---: |
| saved_load | 1 /1 |
| saved_point_identity | 1 /1 |
| objective_setup | 1 /1 |
| inner_solver | 1 /1 |
| independent_state | 1 /1 |
| tensor_metrics | 1 /1 |
| classification | 1 /1 |
| export | 1 /1 |
| **合計** | **8 /8、FAIL0** |

科学的な係数最適化は1回のみ。Refinement0、new U updates0、outer optimization0、ffsim joint fitter0、RC2 continuation0、retry/resume0。既存policy snapshot中のouter/refinement fieldsは、このpilotでは未使用である。出力時に勾配を再評価していない。

保存resourceはwall4.2536499165秒、process CPU20.181210792秒、peak RSS431072KiB。Inner blockのwall2.75813131秒/CPU6.875593521秒、独立stateのwall1.39559098秒/CPU8.784449889秒も保存した。Process CPUはimport/preflight等を含む値で、wallとの比を速度倍率として扱わない。旧solverとのmatched-budget benchmarkも行っていない。

実行前affinity32 CPU、空きRAM約52GiB、CPU/memory pressure avg10=0。依存付き単一点のため既存CPU NumPy/SciPy/JAX/longdouble経路と小行列BLAS調整を使い、CPU/RAM/wallの人工上限・共有venv/cache/OS設定変更は追加しなかった。

DF sector norm、PF/RTE/ground、native回路build/compile、合成誤差、shots、総費用・最終CAは未実施。旧480-callや10-callの残枠・結果・STOPを再利用/上書きしない。

## 6. 次の最小研究判断・公開・STOP

この結果は「実際の保存trial2 Uで、修正innerが終端の正しさと登録J数値一次条件を通った」という進展である。一方、raw precision loss、閾値近傍のJ残差、全trialのreduced-gradient資格、outer U最適化とDF精度は未解決である。今回の結果だけでpaired1563-callへ自動復帰しない。

次の候補は、**CodexによるPR-only reduced経路の限定検証設計**。既存の2 refinement/final-pair admissionと新innerを組み合わせ、どのU点・どこまでの更新を確認するか、J数値資格・raw flag・reduced資格を分離して固定する。これは次作業案で、今回その実計算や契約変更を行ったという意味ではない。**ユーザー／研究C**が方針を判断し、追加科学計算には新しい固定範囲と明示許可を与える。RC2の反復延長・Joint移行・重要科学レビューも今回の範囲外である。

実行branch `codex/research-c-pr-inner-point-execution-20261102`、独立worktree `/tmp/prt-c-pr-inner-point-execution-20261102`。SOURCE149/input6/契約・数値allowanceは無変更、旧raw2388と旧SOURCE/STOP・技術準備原本を保全する。新raw13ファイルへraw solver/trace、state/tensor、SOURCE/契約、call journal、診断endpoint、resources、STOPを保存した。

研究概要・validation status・manifestへこの限定結果を追加し、dated noteは追記だけとする。commit/push後、remote SHA/tree・公開本文・SOURCE/input・新raw/STOPを照合して**mandatory STOP**。追加fit/refinement/U更新や新しい研究段階へ自動進行しない。
