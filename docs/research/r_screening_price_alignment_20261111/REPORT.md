# AのR候補とCの価格Phase 1：統合・整合確認

2026-11-11 JST。**Cの8セルはA/Bと同じ科学recipeへ結合できる。うちH4の新R8/q8・q16はA自動menu外なので、C固有の明示校正・独立検証laneへ保持する。Cの候補は変更しない。** Aのnormalization24件、Cの最大80 wrapper、Aの長時間N例は別の実験範囲であり、小q価格校正を長時間候補へ採用できるとはしない。

今回は技術調査・固定保存metadata照合・計画統合・人工不一致テストだけ。**実normalization/N/C、新しい保存値の科学的再解析、fit/sampling/compile、実G/R最適化は0。** 既存のB共通pipeline/producer/数値会計は変更していない。

成果：[8候補CSV](C_PHASE1_CANDIDATES.csv)／[機械可読照合](ALIGNMENT.json)／[依存関係・実行案](INTEGRATED_PLAN.json)／[HANDOFF](HANDOFF.md)／[metadata CLI](../../../scripts/plan_r_screening_price_alignment.py)／[テスト](TESTS.txt)。

## 固定SOURCEと確認方法

- A：`f98982118324bcc8402ed513b5c720b37bb03e87`のH4/H6_METADATA_PROPOSAL、EXECUTION_PLAN、REPORT/HANDOFF。
- B：`ac3e1ce87b5ed7d9114abc316545366dd097209e`の固定52行N/C coverage、現行のtask/identity/recipe、R builder/request_context。
- C：`8457c443554af5fadaa3ba36f97f6316ca3c5050`のMINIMAL_SCIENTIFIC_PLAN、MODEL_DESIGN、SAVED_PRICE_COVERAGE、SOURCE_INDEX。固定Cの`describe_request`関数本体をそのまま読み、Bの実moduleを渡して両軸のdescriptorを照合した。

既存branch `codex/r-variable-rpe-cost-pipeline-20261111`を再利用。A/Cの作業中tree、未公開成果、private arraysや進行状況にはアクセスせず、指定Git objectだけを読む。新frameworkやnormalizer/価格計数器は作っていない。

## 候補の照合結果

共通：Linear H4/H6、1Å、STO-3G、保存tol-only DF10/DF19・weighted completion、保存H/state/snapshot/fragment/scalar/preparation。H4はS2/δ=.1/PR p1/K2、H6はm5_best/δ1.6/PR p7/K2。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、symmetric_directional、測定付きfull wrapper/P=0。h/M/prior/decoder/α/βを変更しない。

| 系 | m / T | qPF | R | 今回C modelでの役割 | A自動menu |
|---|---|---:|---:|---|---|
| H4 | 3 / .8 | 8 | 4 | 保存校正・探索n9 | 内 |
| H4 | 4 / 1.6 | 16 | 4 | 保存校正・既存n33 | 内 |
| H4 | 3 / .8 | 8 | 8 | 新校正n8 | **外：明示C lane** |
| H4 | 4 / 1.6 | 16 | 8 | 新独立検証n8 | **外：明示C lane** |
| H6 | 4 / 1.6 | 1 | 16 | 保存校正・探索n9 | 内 |
| H6 | 5 / 3.2 | 2 | 16 | 保存校正・既存n8 | 内 |
| H6 | 4 / 1.6 | 1 | 32 | 新校正n8 | 内 |
| H6 | 5 / 3.2 | 2 | 32 | 新独立検証n8 | 内 |

全IDはCSV/JSONに保存。H4の2追加点は`6e1efff1ba1f7ac95a3b` / `201d498a3fa5eaa3b5c5`。近い自動候補R16/32等へ置換しない。A自動menuのPR51+35=86候補を保持し、C校正用2候補をR探索範囲へ自動追加もしない。

H4配分は[R]。H6は元順11 signed pieceを保ち、R16は`[1,2,1,2,1,2,1,2,1,2,1]`、R32は`[3,3,3,3,2,5,2,3,2,3,3]`。全eventはqPF×Rで、11を再乗算しない。H6 qPF1を理由にm0..3の別δ/p5/R32へ結合しない。qRPEはH4ではqPF、今回H6では16qPFという別fieldである。

法則定義のA `89c91c692:src/trotterlib/rte.py` とC `4cb61ef463:src/trotterlib/rte.py`は同一blob。ただしA screeningのlaw tag（event定義SOURCE）とB統合のlaw tag（A mean provider SOURCE）は表記・参照先が異なる。raw law_bindingは同一ではなく、そのままB RMenuBatchへ渡さない。SOURCE一致を確認してB request_contextへ対応付け、元A宣言も保持する手順をHANDOFFへ記した。これは有限samplerとideal lawの距離の証明ではなく、その資格は未取得。

## 保存成果・新実験・依存関係

| 範囲 | 保存成果でできること | 別途科学実行が必要なこと |
|---|---|---|
| CのR0 4セル | 固定候補のN・direct C metadataを再利用。校正は明示の探索/既存cohort | C−Sのpath/SOURCE対応、必要な校正・残差評価はC実行scope。平均を無関係なpathと結合しない |
| CのR1 4セル | 条件とID/allocationの照合のみ | 新compiled C。C価格pilot自体に詳細Nは不要。対応N×Cを求めるなら新N/phase/radiusも必要 |
| A normalization24 | 既存SOURCE/入力/prefix/配分metadata | log B/丸め感度等の新実系scalar評価。保存Nがあるbaselineも含む元24件を勝手に減らさない |
| A長時間N例 | m12の旧R N・signal資料を保全 | H4 R8/H6 R32の新mean/Nと、両baselineの同R-aware allowanceによるscalar再解析 |
| 長q・未使用R | 現在の欠測/domainを表す | Cの別の有限検証。Phase 1の小q 8セルはm12 q4096/q256や大R bandを資格しない |
| 全law/不確かさ | 保存nominal量・shared SOURCE/paired XYを保持 | rare寄与、model q/R systematic、cross-R/round covariance、選択後CIと全RPE成功の科学資格 |

具体的な実行順序は次の通り。各packageは未承認で、まとめて許可する場合も対象・unitを明示する。C Phase 1の校正→係数固定→holdout/診断は、許可された同一scope内で進める工程であり、工程ごとの不要な追加承認は要求しない。

1. **A0**：元の24候補をnormalization-onlyで記録。閾値・予算Fは既定なし。位相/radius/NのPASSも費用順位も作らない。後回しqueueを全比較menuの代わりにしない。
2. **C1**：A0とは論理的に独立した小q価格検証。保存R0の2セル＋新R1/qLでsystem別3係数を校正し、係数等を固定してから新R1/qHを独立検証。新32 whole-pathのX/Yで最大64 wrapper、別のrare対応診断で最大16、計最大80。forced pathを無条件cohortへpoolしない。
3. **A1**：必要なら元のm12 N例（新mean2/内部対照2、新R scalar2＋保存baseline scalar2、XY計8軸）。4点ともA0の24件に含まれ、同SOURCEのnormalization記録を重複評価しない。C1の8セルとの重複は0で、C1の小q modelを価格として代入しない。
4. **任意の同点N/C join**：小qで先に費用接続を見たい場合に限る追加案。Cの新R4セルの新mean/N4＋baseline同policy scalar4（XY計16軸）。C1のための必須Nではなく、自動追加しない。保存2経路referenceの再利用資格を先に確認する。
5. **C2/B**：C1結果とAのN側・実使用候補に基づいて、長q/R方向の有限価格検証を改めて指定する。全13round full compileや新Rの無断外挿は既定にしない。必要なN/C・同policy・全RPE/不確かさ資格と別の実費用実行許可が揃ってから、Bの既存exact会計へ渡す。

A0とC1の共通候補はH6 m4/R16・R32の2点。A0をC用に拡大する場合は追加scopeを明示する。共有できるのは同条件のSOURCE付きnormalization等であり、compiled価格・signal/Nを相互代用しない。

価格精度は将来の固定Nと候補費用差に対するΣN|ΔC|、共有ΣC等から要求を具体化する。今回その重み・差・CI・優先順位の実数値は求めていない。観測max/標本SE/αtotalを保証へ格上げせず、N最小だけでRを採択しない。

PF対照の長q価格も既存domain/検証資格を確認する必要があり、PRのPhase 1ではその不足を解消しない。

## 検査・次の担当・STOP

[CHECKS.txt](CHECKS.txt)に既存core/過去証拠の保全とmetadata・試験SOURCEを記録した。

[7技術test PASS](TESTS.txt)。C矩形・2つの明示H4点、H6時刻/δ、24/80の別unit、旧cohortの明示選択、menu非変更、recipe/law/compiler不一致拒否、法則宣言の差、科学stack/数値入口の非起動を確認。旧83件は既存記録を再利用し、数値coreが不変なので重複実行していない。独立科学レビューではない。

**ユーザー**がA0/C1/A1等の実行scopeを判断。**A**は許可後normalization/N、**C**は未変更のPhase 1を許可範囲で実施し、未確定のseed/rare配置・path対応・必要精度を実行前に固定する。**B**はHANDOFFに従い新成果を受け入れ、coverageと未資格を保持する。新しい実G/R比較は別許可後だけ。

SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOP、他担当tree/process/環境を変更していない。commit/push・remote SHA/主要成果物照合後**mandatory STOP**。科学実行・R最適化へ自動進行しない。
