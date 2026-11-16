# Full PF/S2/2h：通常探索から実Q3一件・再開・C同期

通常の公開PF補完経路から、対象configurationだけに対し実Q3を1件取得した。結果は**採択Q3モデル適合**、MSE上界約6.563770861285207e−7 Ha² ≤ 元exact τ約2.539561285063813e−6 Ha²、条件付き費用956,592,146 RZ。同じround・総予算1で再開し、追加起動0、配分・結果・停止理由の再現、原job全file不変を確認した。Cの14:10定期同期にも同じ配分・exact MSE判定・条件付き費用で反映され、通し検収はPASSした。

対象は保存linear H4/1Å/STO-3G/tol-only DF10、Full PF・S2・要求幅2h=.2、h_RPE=.1、M12、P0除外、configuration `9e96ed75f7f10b5f993ba9e2841684e52ad97b7d4e9df07b7b0070f02c852981`。元のZ1+T1 decoder、DF10 solver proxy、IID deterministic raw、全事象平方loss、exact τを引き継ぐ。B2の新κ/D2統合版検収とは別で、D352539の公開済みPF経路を使用した。

## 入口・固定入力と重複照合

- 実入口：`scripts/run_rpe_method_optimizer.py --problem ... --round .../ROUND.json --run-dir .../run01`。
- 既存code/input基点：352539e2749989b4ddc3ab74c31f486982bfe265。実Q3実行SOURCE：90a52a211f6acc4d7b01e1017572c15224ac25de。元runner/optimizer/adapter/problemのbyte一致はFIXED_INPUTS.jsonに記録。
- 保存13信号・26価格を固定Git registryから読む。pf_completionは起動せず、新しい信号・sampling・build・compile・fitは0。
- §10.5：[DUPLICATE_PREFLIGHT.json](DUPLICATE_PREFLIGHT.json)。B保存Full PF 244 REQUEST、B2既知領域、D旧round、C保存catalog、関連processを照合し、対象configuration/配分の同等Q3や進行中jobは確認範囲内に無かった。別幅16h/7のFull PF成果は流用しない。
- 対象はFULL_PF一方式・configuration一件、総Q3予算1。ROUNDへshot列や研究者が選んだseedは埋め込まず、通常探索器のB_PRICE生成規則で配分を得た。

## 実際に提案・評価した配分

X/Y各m0〜m12：`[59,55,51,47,43,39,35,31,27,23,19,15,11]`。

これは[proposal_only](proposal_only/ROUND_RESULT.json)と実[REQUEST](run01/jobs/FULL_PF/22bbe4a8d5aee8bfa9b8/REQUEST.json)の一致を確認した出力である。配分IDは`RPE_OPT_22bbe4a8d5aee8bfa9b8`。total viewとbindingも元worker/C受入で照合した。

価格は測定付きfull controlled Hadamard wrapperの1shot compiled RZ。保存direct10軸＋条件付きモデル16軸を使い、[PRICED_AXES.json](PRICED_AXES.json)へm/axis/physical ID/N/単価/N×単価/元price locatorを保存した。全26項のexact和は通常探索器のquote 956,592,146と一致する。長qモデルの未知誤差・価格CIはnullのまま。正式E[C]・G・winnerを認定しない。

## Q3・件数・資源

|項目|結果|
|---|---|
|実job|FULL_PF/22bbe4a8d5aee8bfa9b8 一件|
|実API call|SCIENTIFIC_CALL_START一回、API_RETURN/RESULT/OUTCOME照合済み|
|数値gate|有限binary64 decoder closure、質量・残余・loss・DP scope、process終了と保存gate通過|
|判定|MODEL_BOUND_MEETS_ADOPTED_THRESHOLD|
|MSE上下界|両端とも表示約6.563770861285207e−7。exact分数は原OUTCOME/API、CLIは30桁の外向き丸め|
|初回停止|Q3_BUDGET_EXHAUSTED、未実行の次要求を別欄に返す|
|対象外job|0|
|追加信号/sampling/回路build/compile/fit|全て0|
|科学retry/回収|0。技術エラーなし、原worker完了済み|

元PF worker→API/partials保存→pinned validatorの実入口を人工2 testで先に確認し、元AST guardは変更していない。C publisherは人工PASS/FAIL/取違え拒否の3 testを通過した。FAILはINFEASIBLEとして方式最安/達成費用へ入らないことも検査した。これらは人工入力の著者・委譲先実装検査であり、追加H4科学callではない。

Python3.11.0rc1の専用環境、固定PF view cbc471c913072de45c25a0ed22d51dff8a9b7034、元8-worker/outcome_batch8のDPを使用。これは元人工benchmarkで8/16が同等だった設定の再利用で、新しい資源capを設けていない。実worker wall150.047s、job約152.3s、CLI wall165.17s。観測process tree peak3,970,072KiB（約3.79GiB）、worker単体peak606,748KiB。DP transition87,623,455、round段階logで進行を観測し、faulthandler等の非同期dumpや共有環境変更は行っていない。速度倍率は未測定。

## 同一runの再開

同じROUND、run01、通常CLI引数、総予算1で再実行した。[RESUME_CHECK.json](RESUME_CHECK.json)：

- 累積LAUNCHは1→1、新規callは1→0、消費済みは0→1。
- 構成・X/Y配分・MSE上下界/判定・exact費用・停止理由は一致。
- job下の全原file（API_RETURN、partials、CASE_INPUTS、REQUEST、OUTCOME、SOURCE、終了記録等）はhash不変。
- 初回の親出力はfirst_snapshotへ保存。予算やSOURCE条件を緩めず、次候補の892,832,582RZ配分は投入していない。

## Cへの公開と資格

原worker成果は55c174a3f04f560a57bd30abd19b1197b2edea67に保全。metadata ledger/input viewを固定した後、既存のgeneric PF形式で[SUMMARY](SUMMARY.json)・[C_RESULT_JOIN](C_RESULT_JOIN.json)をf2a6930ed638d98db09b357412caaa7eef2d44aeへ公開した。raw CASE_INPUTS/API/結果は変更していない。C branchやschedulerの変更も無い。

ローカルの既存C consumerは、同じconfiguration・配分・26価格・Q3判定・条件付き費用をFEASIBLEとして受理した（C_CONSUMER_CHECK.json）。通常周期の反映はC_PERIODIC_SYNC_CHECK.jsonを正本とする。同期前の自前consumer確認を、定期同期済みの証拠に読み替えない。

ロードマップ差分はHANDOFF.md。今回の結論はこの一構成・自動提案一配分の通し検収まで。新κ選択、別構成/配分、価格モデルの長q保証、全体最適性を認定せず、次の探索へ自動進行しない。

## 通常定期同期の実確認

Cの14:10:01〜14:10:25 UTC周期が終了0で反映した。公開 `5fbe4782725ae3ba36e2b431d199e39ae1a29b7b`、`docs/research/h4_saved_rpe_cost_catalog_20261116/run01/CATALOG.json#candidates[57]`、point `7338c82f5538c9968acecee3`。同configuration/配分、FEASIBLE・Q3モデル適合、956,592,146RZ、26価格軸を確認し、MSE上下界は原OUTCOMEのexact分数と一致した。join/summaryはINTEGRATED。Cの新科学callは0、手動同期・scheduler/C branch変更は0。固定locator・周期logはC_PERIODIC_SYNC_CHECK.json。
