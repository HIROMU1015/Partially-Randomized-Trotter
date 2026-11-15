# C→B/D1/統合担当：D7受入と次の固定shot評価案

- branch：`codex/m5-m4-receipt-admission-20261114`。
- 入口：[ADMISSION.json](ADMISSION.json)。D1完全SHA `06956c64380107392f1b24fe54b9273f0f5ae1c0`。原raw `b5fac0ee4eac83f027f883e5ea5630623a98bf29`、元実行 `9cbabc617e95a81810924413c1b2b06ced329514` を保持。
- **6/6軸別receipt、4/4記述的比較を受入、保留0**。m5は2 path/軸の平均、m6は1 path/軸。X/Y共有の3 pathを6独立trajectoryとして数えない。
- **Bのκ2固定1.5倍配分へ26/26軸接続**。`rows` の結合キーは `(original_configuration_id, physical_id, m, axis)`。精度側は `total_configuration_id,total_candidate_id,binding_id,allocation_id` も保持。original/total viewのIDをreceipt生成IDへ付け替えない。
- `rows[k].receipt_admissions` → `receipt_admissions[j].original_record.locators` から公開元価格・compile/evidence・raw manifest/decoded selectorへ戻る。`rows[k].holdout_admissions` → `comparisons[j]` からb516の観測前予測とD1の保存exact比較へ戻る。`B_join` は6c5569c34の固定Q3/N正本。
- m5-X/Yの保存符号付き相対誤差は各−0.445257%、m6-X/Yは各＋0.134608%。再計算なし。`DESCRIPTIVE_COMPLETE`、`NOT_ASSESSED_NO_ADOPTED_TOLERANCE`、`tolerance=null`。既存モデルの `CALIBRATED_UNVALIDATED/qualified=false` は変更しない。
- 元7数値gateは各cell index0のみ。全pathのscalar/event/XY/parity gateというD1の実施scopeを維持。consumerが数値検証を再実行したものではない。
- G566/G4528/G36223/Q8R566/P2R566は非configuration較正cell。q/Rの一方が実roundと同じでも、そのroundの価格観測として扱わない。P2R566はG566の親2本の連結、独立親標本ではない。m3 P1はwiring単一pathのまま。

## 次に承認する案

**[F1：0新path・0新compileの固定shotモデル総額診断](NEXT_SCIENCE_PLAN.md)** を推奨する。対象m0〜12×X/Y、既存m5/m6の4予測を再利用、残り最大22予測を元exact係数から評価。26軸の非負性/domainを先に検査し、通過時だけ条件付きモデル総額・必要誤差予算を算出する。併せて保存m6モデル部分額の診断を行い、少数roundの母期待部分和検証へ絞れるか判断する。負値をclamp・補正せず、全26軸が有効でなければモデル総額はnull。

正式比較の対象lawはBと同じideal finite-K2 Full-H law。D7はPCG64/CDF有限cohortで、law差・標本誤差・移送・共有係数誤差は未評価。F1はモデルscenarioの診断で、正式E[C]/CI/winnerではない。PF基準593,555,352 RZは1002f1dfeから転記し、その条件を保持する。D7の小誤差から全roundの2%/CIを設定しない。

D1実測はm5 2path/4compileでwall22:05.61、m6 1path/2compileで1:26:15（元GNU RSS等はADMISSIONを参照）。後半roundの実測はなく、巨大compileを最初の前提としない。F1後に必要精度と取得対象を限定し、標本数・law・停止規則を別案として確定する。

## ロードマップ差分・STOP

「Full-H D7独立holdout6 receipt/4比較をCへ受入。Bのκ2・1.5倍Q3固定配分を26軸へ接続。次は0-compileの固定shotモデル診断案。全26軸の正式E[C]/CI/Gは引き続き未資格」。ロードマップ本体は統合担当が更新する。

今回：保存受入・metadata接続・人工テスト・計画のみ。新予測/期待価格/加重額/fit/Q3/sampling/compile/N/Gは0。F1未実行・別途明示承認が必要。今回の公開・remote照合でmandatory STOP。
