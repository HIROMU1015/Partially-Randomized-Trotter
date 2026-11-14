# 有限価格取得の開始条件と段階（今回未実行）

**現在の状態は[保存4件binding完了の追補](ADMISSION_UPDATE.md)を参照。** Cのp7受入・新版CELL_PLAN・共通指示の取得は解消し、SOURCE22ccf3972へ接続済み。policy/許容差は未採択で科学実行STOP。以下は初版0501096bd時点の記録を保全したもの。

改訂SOURCE `2b9983590a642e67fda794521baac7e5eaf0496f`は**Stage 0用**。
`P7_CELL_PLAN/P7_ADMISSION=None`なので、実行flagを指定してもscience child前でSTOPする。
この技術PASSをStage 1の開始許可に読み替えない。

## 現在の不足

1. Cのp7 M5/m4保存receipt正式受入SOURCE。現remote2fa88065はFullの受入だけ。
2. C新版CELL_PLAN固定SOURCE。Dの提案表はCの正本ではない。
3. ユーザーによる比較policyの正式採択。
4. ユーザーによるholdout許容差・分母の正式採択。Cの2%案を自動採択しない。
5. `research_project_instructions_unified_2026-11-14.md`正本の所在と全文確認。
6. 実行直前の重複照合・SOURCE/input/provider/import依存・出力先・実行資源の確認。

今回調査した既知範囲以外のprivate価格・実行中jobがあれば、そのmetadataを照合する。
未知を0とせず、元証拠不足または追加既存価格の候補は当該cellをSTOPして報告する。
過去のsparse checkoutによる入力import失敗を踏まえ、tracked references等を実行worktreeに展開し、
元SOURCEのbytesと実restore moduleまでのimport-only経路を本番前に確認する。
今回は実preparationの成功やruntime scalar gateの通過を新しく確認していない。

## 条件が成立した後の担当

**Codex D（技術）**：C正本を固定blobで接続し、提案表から2 action以外の変更がないことを照合する。
両routeの保存binding・C guard・model group・重複検査を確認し、新しい実行SOURCEをcommitで固定する。
前の最大34件承認内から重複2件を除いた32件であること、元条件とSTOPを維持していることを報告する。
不足条件が残る状態では開始しない。新候補やscience条件の追加はしない。

**Stage 1（D、開始条件成立後のみ）**：

| route | CALIBRATION新規 | 保存較正 |
|---|---:|---:|
| Full PF p10 | 10 wrapper | M5/m4 X/Y 2件 |
| Truncated p7 B0I | 10 wrapper | M5/m4 X/Y 2件 |
| 計 | 20 | 4 |

既存`--execute-approved-stage CALIBRATION`を使用する。今回はコマンドを起動していない。
stage namespaceは従来の`…/calibration/run01/`、privateもroute別の同stageである。
旧namespaceが既にあれば上書きせずSTOPする。別runへの自動変更・retryはしない。
既存native/context/checked_trial/wrapper/compiler/cacheをそのまま使用し、元数値gate不一致なら部分保存STOP。
並列workerは既存のidle CPU/available RAM観測規則を使用し、共有設定・他者processを変更しない。

公開工程は従来どおり、RESULT/PRICES/SIDECARS_UNBOUND/IMPLEMENTATION_EVIDENCE/C_INPUTSをcommit/push、
その実commitへ`--bind-stage-evidence CALIBRATION --evidence-commit ...`で束縛、C guard、commit/push・remote照合。
この時点で**mandatory STOP**。Dは実fitしない。

**Stage間（C、別途科学実行承認が必要）**：保存4＋新規20の24較正価格を12 groupに分離し、
各m4/m5で`C(q)=a+bq`をfitする。係数・design・SOURCE・許容差を凍結公開する。
Dが実データfit・係数凍結を代行しない。

**Stage 2（D、前Stageの条件・次段指示成立後のみ）**：Full6＋p7 6＝12 wrapper、m6だけ。
既存`--execute-approved-stage HOLDOUT --frozen-models COMMIT:PATH`を使用する。
CALIBRATIONの公開済み7ファイルが不変であることとCの凍結modelをcompile前後に照合する。
既存stage outputの上書きや、m6の較正混入は拒否する。
公開・bind・guard・remote照合後**mandatory STOP**。

**Cのholdout判定**は別途科学実行承認後。Dは判定・再fit・model昇格を行わない。
m7〜m12の外挿、全qでのaffine性、最終candidate順位、N/Gへ自動進行しない。

受入不成立・SOURCE不整合・元証拠不足の場合は原因を保全してSTOPする。
旧34件計画への自動復帰、p7 M5/m4の再compile追加、p8/別round/別候補、Pauli B0の再実行は行わない。
