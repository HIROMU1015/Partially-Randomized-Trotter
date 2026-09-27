# Codex用：PR-2再基準化・検証パッケージ

## 1. 目的

S0のhash mismatchを理由に旧実験を停止したことは維持する。一方、この停止を部分ランダム化研究の科学的失敗とは扱わない。旧入力の回復可能性を有限の範囲で調べ、必要なら新しい固定入力を使う別seriesとして、同じPR-2 RQを検証する準備を整える。

レビュー基点は `bf5d3b2405b2bec4f88dd0589b75dd737b13e052`。
旧authorizationは `e9bffb85f9ed57712bb83150172a6a4662cecf7f`。
旧sourceは `c644925b50587072784846df09bf02c39e8453e1`。

この文書は新しい作業計画であり、旧S0のstatusやauthorization値の書換え指示ではない。旧S1 runnerを直接実行しない。

## 2. 並行して行ってよい準備

### A. read-only evidence inventory

既知のログ・cache・MolecularData等、合理的に特定できる保存先だけを一巡確認する。

- 旧pilotに使ったone-body、lambda、g、constant、basis、state等の配列が残っているか。
- hash入力が何を含み、旧時点のfield別情報がどこまで残っているか。
- source/package以外のBLAS、thread、solver設定等が残っているか。
- 回復可能性の判定と、未確認・アクセス不可を明記する。

hash一致まで再生成する探索をしない。計算結果やheld-out performanceは開かない。

### B. new-series amendment草案

旧系列を閉じたまま、新しい固定入力に対するseries_id、source、manifestを設計する。

- old series ID、old expected/observed hash、old STOPを保持。
- 旧pilotの結果は既知、S0のenergy/lambda差は既知、candidate性能は未評価と記す。
- 旧入力が回復できればその原本を候補にする。
- 回復できなければ、保存済みS0 development snapshotを新入力候補にする。旧入力と同一と主張しない。
- observed snapshotを選ぶ理由は性能ではなく、既に保存された最初の入力であること。複数再生成から安いものを選ばない。
- rank6、control3/9、共通task、corrected estimator、既定grid、cost scopeを、hash不一致を理由に変更しない。
- 1.30 Å held-outは保存物を維持し、signal/cost/rankingを開かない。再生成もしない。

A/Bは新しい研究用数値実行なしに準備する。

## 3. 実行前に固定する最小項目

1. 新seriesが旧pilotのcontinuation claimではなく、新入力での同じRQの検証であること。
2. 採用input artifact、raw file SHA-256、array hash、metadataのmanifest。
3. S0'で行う既存配列の数値検査と、S1'で行うsignal/sampling/compileの範囲。
4. 閾値、失敗分類、各stageのcounter、原本非上書き。
5. S0'通過時だけS1'に接続し、S1'後は必ず停止する条件。
6. S2/S3、expected-cost MC、resource winner、held-out開封を許可しないこと。

manifestが自分自身の最終commitを埋め込む循環は避け、specification/source commitと、そのmanifestを保存したcommitを分けて記録する。

## 4. 改訂固定後の検証（複数をまとめて実施）

### V1. snapshot integrity・model validation

新分子buildではなく、採用済みdevelopment snapshotを読む。

- 保存配列のshape、dtype、Hermiticity、finite値、constant・one-body・DF構成を検査。
- load→同じdata hash、同じinputを再loadして同じdata hashであることを検査。
- state normとtarget Hに対するresidual、sectorの整合を確認。
- N_alpha=N_beta=2だけをsingletの証明として扱わない。必要なら記述をN=4、S_z=0へ限定する。
- 旧energyとの差が小さいことを旧入力同一性の受理条件にしない。

数値処理は既存development配列に対する決定論的検査。held-outの性能評価なし。

### V2. partial構造の検査

rank3/6/9について同じsnapshotからgeneration/weight prefixを構成する。

- ordered indices、集合、各fragment、tailを別々に記録。
- constant/one-body/identityを含むH_D+H_Rの再構成。
- sampling coefficientの符号、確率和、tail identity phase。
- G/Wが同一なら重複を統合し、異なれば別候補を保持。
- prefix境界をまたぐ縮退factorの混合を、無害なgaugeとして消さない。

これはresource比較や最適rank選択ではない。

### V3. 必要な実装修正・unit tests

現在のコードはS0 PASS側で `_actual_wrapper_probe` を呼ぶ一方、payload/validatorはtrajectory countを0へ固定している。今回のSTOPはprobeより前なので旧結果を書き換える必要はない。新seriesでは実行したwrapper probeのsamplingを正しく数える。

- input-only、synthetic test、wrapper probe、candidate evaluationのcounterを区別。
- build成功とoperator/control意味論の数値検査を別fieldにする。
- frozen raw inputの改変検出、artifact上書き禁止、非承認stageのguardをtest。
- corrected estimatorの既知multiplier、finite truncation、shot式を既存の正しい規約で検査。

canonical hashは必須にしない。採用する場合だけ、許す変換・不許可変換を定義して正負testを追加し、raw hashを残す。

### V4. 既定の軽量S1'

新しいS0'が正しく通った場合のみ、改訂で固定したS1'へ進む。

- T=0.1、q=1を主にcorrected/raw mean、target、control、Re/Imを確認。
- rank6と既定control、既定(r,K)の範囲を広げない。
- q=8 toy wrapper testを新しいH4 q=8 performance結果と混同しない。
- compile smoke testは既定の小規模上限を守り、32/128 trajectoryによる期待cost推定は行わない。
- K=2/4の高次eventがsmokeの一trajectoryに出ないことはあり得る。対応するunit testの範囲を明記し、1 sampleで全event正当性が確認されたとしない。
- 完了時にはresource winnerを判定せず、S1' summaryを書いて停止。

## 5. 旧pilotをどこまでやり直すか

PR-3の再試験や、旧PR-2 qDRIFT screening全体の再実行は不要。新入力に必要なのは、target、分割、誤差参照と、将来の比較baselineが同一入力から作られること。

旧pilotの数値は新inputのbaselineに転記しない。新系列でcost比較を行う段階では、その新inputで対応する費用を評価する。その段階は今回未承認。

## 6. 途中で止める条件

- 旧入力回復不能：Aを終了して新series経路へ。研究全体のSTOPではない。
- snapshot破損、再構成不整合、controlバグ：実装／入力問題として止め、同じ科学条件で修正可能か報告。
- taskやtargetの変更が必要：勝手に変更せず、具体的理由を報告。
- G/W同一：研究失敗ではなく、手法同値性として整理。
- S1'で候補がaccuracy不適格：correctness違反と分けて記録。S1の短時間結果だけで資源研究全体のnegativeを確定しない。

## 7. GPTへ返すpacket

一つの統合packetにまとめる。

- evidence inventoryと旧入力回復可否。
- old STOPを保ったnew-series amendment、source/specification/manifestのcommit。
- 新入力のfile/array/model/algorithmのhash対応。
- 実際に見た情報と、未開封のheld-out項目。
- V1–V3結果とtest log。独立再現かlocal validationかを明記。
- 実行許可された場合のS1' summary。
- performed numerical operationsのcounter、deviationと理由。
- S2/S3は未実行・未承認、automatic_next_stage=null。

次回判断は『部分ランダム化を評価する準備が整ったか』と『本来の資源RQを測る次の有限計画は何か』を中心にする。新しいhashの工夫そのものを研究の着地点にしない。
