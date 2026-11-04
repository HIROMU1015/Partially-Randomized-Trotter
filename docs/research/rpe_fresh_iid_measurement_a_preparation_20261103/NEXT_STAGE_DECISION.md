# 候補Aの次段階判断

2026-11-03 JST。**技術準備完了／科学実行は未承認・未実行。**

1. Adapterと限定runnerは準備できた。既存sampler、seed helper、trajectory作用を再利用し、
   fresh request→条件付きmean→仮想±1測定を接続した。人工テスト18件PASS。
2. 同一event列の自然な再出現と、request/streamの意図しない再利用を分けた。
   役割・q・axis・shotに結び付く新公開seed recipeを固定した。旧private seedは使わない。
3. H4/1Å/STO-3G/legacy DF12、physical B保存state、Partial L_D3/δ=.02/r1/K2、
   q1各軸510・q8各軸454、合計1,928 trajectory/仮想shot、8,284eventの一回だけを計画した。
   state solve・controlled/measurement回路build・compile・費用再解析は0。
4. 比較する4保存mean、旧N/α/β/ε、SOURCE/input、RNG、12診断の式とSTOPを
   [SCIENTIFIC_EXECUTION_PLAN](SCIENTIFIC_EXECUTION_PLAN.md)で固定した。
   不一致未検出でも同等性はUNDETERMINEDとし、V2成功率を再評価しない。
5. 実行前に残るのはユーザーの明示的な科学実行許可と実際の空き資源の確認。
   新numeric contextと分布平均の実H4整合性は未検証であり、今回の人工PASSでは埋まらない。
   新namespaceのseed値選び直し、結果後のN/閾値変更、科学的retryはしない。
6. **ユーザー**：上記1回の新sampling・state-action・仮想測定を実施するか判断し、
   実施する場合は明示許可する。**Codex**：その許可後に公開固定SOURCEで実行・記録しSTOPする。
   承認証明書・grant発行等は不要。独立詳細レビューは別許可。

V1/V2/V3/R1と候補BのSOURCE・結果・正式status、M06-FとR2/R3のSTOP、
他研究・H8のworktree/jobは保全した。
今回新しい実H4 sampling・state-action・仮想shot、NPZ数値decode、回路build/compileは0。

成果物：[実装報告](IMPLEMENTATION_REPORT.md)、[実行計画](SCIENTIFIC_EXECUTION_PLAN.md)、
[adapter](fresh_iid_adapter.py)、[runner](run_fresh_iid_a.py)、[人工tests](test_fresh_iid_a.py)、
[検査記録](VALIDATION.json)。公開・remote照合を終えたらmandatory STOP。
