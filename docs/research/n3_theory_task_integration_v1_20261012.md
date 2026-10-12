# N3 理論・文献・幅指向縮約task：統合準備完了 v1

2026-10-12 JST。完了済み[GPT holistic review原文](../../artifacts/n3_theory_task_integration_v1/2026-10-12/review_intake.md)第10節を受領。
ユーザーの「次のGPTレビューや実行の明示指示が必要な地点まで進める」に従い、関連技術作業を統合した。
今回の状態は **N3_INTEGRATION_PREPARED_GPT_RESEARCH_DECISION_REQUIRED_MANDATORY_STOP**。
詳細なGPT研究レビュー・中心テーマ採択・新science batchは実施しない。

## 完了した準備

- [claim map・証明・独立保存算術](n3_theory_claim_map_v1_20261012.md)：全幅排除、旧OR下端、既存OCC/CAR/Schur/Fantope/改善機構、包含・有限generator限界を整理。
- [一次文献対応](n3_literature_correspondence_v1_20261012.md)：数学・RDM・Schur・sparsification・active-space taskの入力／保証／取得費用を比較。SQuISH全文を新たに取得。原典二件の全文未取得も明示。
- [幅generator・縮約task仕様案](n3_width_reduction_v2_spec_20261012.md)：入力、有限提案と厳密checker、cold/warm/OR、単調性、未判定、mode選択、誤差会計、費用、将来sourceのfreeze要件。
- 新[stdlib checker](../../src/trottertracks/representation_exploration/n3_theory_implications_v1.py)、[CLI](../../scripts/verify_n3_theory_implications_v1.py)、[30 synthetic tests](../../tests/test_n3_theory_implications_v1.py)。checker sourceは`370c42a6`で固定してから保存値へ適用。

## 独立照合の範囲

公開base `26ed4b959bf5b59d23e2131d8be785ee701f0bc2`、旧source13／plan／saved inputs等34 filesをSHAで固定。
元pack/ledgerからのlossless binding、保存dualとexact PSD・旧対象trace、r,b条件、既存NORM root区間、保存OCC scalar／幅／勝者IDを確認した。
二入力とも全幅排除補題が成立。r,b,q、改善区間とforced-energyの保存証拠からの下界など14項目がGPT付録と一致した。
新しい数学的仮説・数値証拠の探索ではなく、レビューで導出済みの帰結の独立照合。
OCCの全ZとU,c proofは以前のimmutable source-bound監査を継承し、今回の小さいcheckerで再生成したとは言わない。
[exact結果](../../artifacts/n3_theory_task_integration_v1/2026-10-12/saved_analytic_checks.json)、[レビュー対応](../../artifacts/n3_theory_task_integration_v1/2026-10-12/review_correspondence.json)、[identity](../../artifacts/n3_theory_task_integration_v1/2026-10-12/saved_identity.json)。

Scope：S1 star、実一体＋非負unordered density、scale1.25、ν4、full10→active9、一外部mode8、元δgoal=.005。
今回δは保存OCC正有理上端、同じ保存U/c。geometry/basis/DF rank/L_D/PF時間窓は適用なし。
登録旧OR下端に対する達成幅改善は約20.424087%／0.343340%。同じS1の事後developmentであり、独立移送・真誤差・量子費用の改善ではない。
元primary NO_ADDITIONAL_ACCEPTANCE_OBSERVED、追加保証0、2-base未達、過去STOPを保持する。

## 実行していないこと

新入力、κ/μ/W/τ/Z探索、OCC生成、δ scan、固有値／E_forced計算、U/c再取得、science batch再実行、ground/gap、compile/shots、N1/N2再開は0。
幅v2 generator／selectorの科学実装は仕様案であり、未実装。scienceの実行planやauthorizationを作って認可済み扱いにはしない。
小さいexact照合はCPU1／thread1。保存値の読取り・matrix/trace再構成を再利用し、新しいproducer・数値ライブラリは起動しない。
新branchだけをHIROMU1015へ公開しroot dirty/BentoSlide/別Trackをstageしない。旧source/結果／manifest既存entryの変更は0。
global manifestの既存description欠落FAILは別問題として保全する。local testをimmutable CI・外部科学再現と表現しない。

## 次の担当と停止理由

次はGPT／ユーザーによる、理論・算法仕様・一次文献対応をまとめた科学的判断。
レビュー第11節の「理論・algorithm仕様・先行研究対応が揃い、独立した研究差分があるか判断できる」に達したため、同じ一点診断や追加scanを開始せず停止する。

判断してほしい事項：

1. Fantope/Schur等の既知primitiveを踏まえ、独立した理論・保証生成／縮約taskの寄与が具体的に残るか。
2. 同情報の一般SDP/RDM等の強い対照を必要とするか、利用情報・保証・費用をどう固定するか。
3. 幅と縮約決定を結ぶtask、およびcold取得費用の研究上の評価基準を採択できるか。
4. 独立した限定検証の情報価値があるか。あるならclass／入力／予算／比較／停止条件を採択し、その後に別の実行認可へ進むか。

新規性・主要claim・中心テーマ・次の科学契約をCodexで確定しない。
[共通指示のsnapshot](../../artifacts/n3_theory_task_integration_v1/2026-10-12/workflow_intake.md)第2.1・3・11.4・12節に従い、GPTレビュー開始と新計算実行は別承認。
N3はGPTレビューの優先継続候補という判断を受領した状態、中心テーマは未採択。N1停止、N2保留、next_stage_authorized=false、Mandatory STOP。
