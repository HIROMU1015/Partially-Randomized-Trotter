# 初回の有限compiled価格取得計画：候補依存adapter

2026-11-13 JST。**未承認・未実行。対象候補・round・path数・価格精度/confidenceは未採択。** 本文は実行前に埋める有限scopeと判定条件であり、そのまま起動できる凍結runではない。旧P1/P2・旧window診断の予算やSTOPを再開しない。

## 1. まず受け取る入力

1. A/探索担当の有限physical候補集合と、Bのcomplete V2 configuration。各対象のm・X/Y、L_D/formula/ΔPF/q/R/K・piece配分とlawを固定する。
2. 復元する元H/state/snapshot・input SOURCE、使用provider/runtime、新adapterを含むexecution commit。preparation hash既知なら指定し、未取得なら初回prepareでresolved hashを保存する。
3. B/ユーザーが指定した目的：候補adapterの初回移送診断か、finite-cohort pilotか、平均価格CIの取得か。技術診断を平均価格の精度保証へ読み替えない。
4. 必要なpath数n_j又は、その選び方に必要な既定精度・confidence・range/variance根拠。今回は新しい2%/95%等を採択しない。

対応外の候補は価格不足として返し、探索空間から除外しない。標準PR-Y4や空tail等が必要なら、同じ科学runでrecipeを偽装せず別の最小技術接続を先に提案する。

## 2. 最小の実行scopeの形

対象Jは外部比較に必要なcandidate/roundの有限集合。全L_D×formula×K gridや全13round一律高精度化を開始しない。取得軸A_j、normal pathのindex集合I_jを事前に固定する。

\[
\text{新compileの最大数}=\sum_{j\in J}|A_j|\,|I_j|.
\]

cache hitがあれば実compile数は減るが、その余剰で別path/round/候補を追加しない。両軸を指定する場合は同じpathに対する個別応答として記録する。過去のX=Yから片軸を省略しない。mean/共分散の計画が両軸を要するなら、初めから両軸を指定する。

初回の候補移送を診断する場合、必要な比較に含まれる新L_Dやsigned/K条件を問うcellを選ぶ。固定L_D1 replayを行うか、実際の最大使用domain点を含めるかも、対象/予算を提示してユーザーが決める。人工PASSを理由に新H4条件の自動実行、m7→m8→m9の連続compileを始めない。

## 3. 1 pathの固定手順

- 元loaderでH/state/snapshot/inputを照合し、専用processのNativeRuntimeを作る。input_sourceとexecution_sourceはrequestと一致させる。
- from_b又はfrom_a_jobでrequestを生成。required preparation/source・profile/lowering・signed allocationを実行前に照合する。
- runtime.contextで新L_Dのhead/tail/λ/scalarを構築し、resolved preparationを保存する。過去C1のsystem-only contextは呼ばない。
- 新cohort/index/outer/pieceの元seed規則でnormal sample_requestsを一度呼ぶ。forced rareを追加しない。保存path replayを指定する場合はその元event/seed/digestを明示復元し、normal新標本数へ足さない。
- Adapter.acquireで同じpathのS、指定軸C、D、fingerprint/6指標/cache参照を取得する。元scalar一回・measurement・新ZZ・固定compilerを維持する。
- fresh directoryに元fast出力とCANDIDATE_PATHを保存。runner側でcontext/samplingを含むpipeline wall・各compile時間・peak RSSを記録する。fast TECHNICAL_TIMINGだけを全pipeline時間としない。

SOURCE・input・seed・J/A_j/I_j・出力先・最大compile数・停止/再試行条件を固定してから科学実行する。今回科学runner/launch管理layerは追加しておらず、上記を現在実H4で実行していない。

## 4. 判定と結果の引渡し

技術的PASSは、指定path/軸の会計が一致し、正しいcontext・grouping/law・lowering/profile・measurement・scalar・保存digest/fingerprintが保持されること。過去経路との科学replayを範囲に含める場合は、同一event/seedのS/C/6指標/fingerprint一致を別途判定する。

pilotの分散・平均を取得する場合も新しい科学計算なので、その集計scopeを明示する。mean/CIが必要ならtarget lawと同時confidence・rare/lawの扱い、既存E[S]を使うか直接Cを使うかを先に定める。標本min/maxやnのfloorだけで価格CIを認定しない。新L_Dに旧E[S]や残差を移植しない。

十分な証拠がなければBにはmean=null/UNQUALIFIED/UNKNOWNを返す。path CやSをmean/厳密区間へ変更しない。N/RMSE/G・候補winner算出はこの価格取得runの自動後続にしない。outer-window/fitを予算内の代用品として追加しない。

不一致、対象外条件、実行不能、保存不能は部分結果を保全してSTOP。無断retry・新seed・path増量・候補変更はしない。一時的な共有負荷だけで正常workerを再起動せず、実際の異常/資源不足には対応する。並列度は利用可能資源と依存から選び、他ユーザー/共有環境を変更しない。

## 5. 実行前に埋める有限表

| 欄 | 現在の状態 |
|---|---|
| candidate/configuration・round/axes | 未指定。A/B/ユーザーが有限JとA_jを選ぶ |
| execution SOURCE/input references | 本成果の公開commitを基にrun scopeへ固定。実inputは今回復元していない |
| cohort/index/seed・n_j | 未指定。既存carrier規則、新namespace、有限index一覧を固定する |
| compile/価格集計予算 | 対象ごとの取得軸数×path数の有限和と、許可する集計内容を具体化する |
| accuracy/confidence/用途 | 未採択。診断・pilotと正式mean/CIを区別して指定する |
| STOP | 対象内の結果を保存後mandatory STOP。次の価格精密化/最適化へ自動進行しない |

ユーザーの対象・SOURCE・有限範囲に対する明示チャット承認で十分。追加grant/承認証明書は要求しない。今回の技術実装の承認を新科学実行へ拡張しない。

**次：A/B/ユーザー**が上表を埋める。**D**がその固定範囲のみを承認後に実行し、必要な場合だけ**C**のE[S]/featureを利用する。現在はcommit/push・remote照合後mandatory STOP。
