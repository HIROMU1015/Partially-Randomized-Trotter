# GPTへのTrack A完成原稿v0.1レビュー依頼

## 依頼と境界

限定的な条件付きresource/applicability studyとしての完成原稿を評価してください。
今回のreviewは研究方針・新規性・投稿可能性の判断であり、科学計算を実行する依頼ではありません。
原稿を読まずに新しいpilotや一般的な追加実験一覧を勧めず、現在の結論を変える未取得情報があるかを特定してください。

保存CSV/JSON、原稿、図、関連sourceと一次文献だけを使い、
分子NPZ/NPY/pickle、runtime/cache/registry、state、GPUへアクセスしないでください。
signal評価、sampling、circuit build/compile、全repository tests、旧science runnerは実行禁止です。
Track Bの探索と統合せず、原稿の限定scopeを維持してください。

## 読む順序

1. [通し原稿v0.1](track_a_resource_study_v0_1.md)：AbstractからConclusion、4主図とReferences。
2. [Supplement](track_a_resource_study_supplement_v0_1.md)：比較集合、6metrics、SE、selector/P、provenance。
3. [執筆者claim audit](track_a_resource_study_claim_audit_v0_1.md)：独立reviewではない。
4. 必要な数値に限り[223候補元精度CSV](../../artifacts/resource_applicability/track_a_manuscript_v0_1/2026-10-05/original_precision_223_candidates.csv)、
   各表示CSVと[asset manifest](../../artifacts/resource_applicability/track_a_manuscript_v0_1/2026-10-05/manifest.json)。
5. [固定claim/evidence map](../research/track_a_post_pm2_claim_evidence_map.md)と、そこが指すvalidation文書・保存JSON。

結果収録commit：5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc。
設計commit：d45c4006d440ba053517d6f646844a61b18dfd15。
原稿作成時点はlocal uncommitted。利用者のcommit/push指示により本bundleへ収録する。
公開handoffでは別途確定するcommitを示し、
設計commitに原稿が含まれているとは主張しません。
identityは[verification audit](../../artifacts/resource_applicability/track_a_manuscript_audit/2026-10-05/verification.json)へ収録します。

## 評価してほしいこと

- 主RQとC1〜C3は、単なる既知trade-offの図解以上の定量的情報を与えるか。
- 最接近GüntherらのDF・truncation比較との差分は十分か。partial/DF/cost×shots自体の新規性は主張しない。
- development218の設定感度と固定5構成transferを分離し、method一般の最適性を避けられているか。
- bias/headroom、B、shots、actual wrapper費用を区別する図・説明は明快か。
- second-order実装class、保存状態、RZ proxy、32 cost標本、受理規則という限界を含め、投稿に足るscopeか。
- 強いcontrolled synthesis等の未評価baselineはclaim縮小で対処できるか。
  追加測定が必要なら、どの具体的claimをどの結果で撤回・維持するのか一件だけ示せるか。

## 回答形式

先頭に次のいずれか一つを示してください。

- RESOURCE_APPLICATION_PAPER_PLAUSIBLE
- CLOSE_AS_TECHNICAL_RESOURCE_NOTE
- ONE_TARGETED_VALIDATION_REQUIRES_SEPARATE_REVIEW

その後、中心的な判断理由、過大claimまたは論証の穴、最小の原稿修正、
最接近一次文献との差分、投稿scope案を述べてください。
現証拠ではどの形式も難しい場合はその理由を明示し、positive result探しを提案しないでください。
追加検証案を出す場合も実行認可とはしません。
全結果でmandatory STOPし、原稿修正・新計算の認可を利用者へ戻してください。
