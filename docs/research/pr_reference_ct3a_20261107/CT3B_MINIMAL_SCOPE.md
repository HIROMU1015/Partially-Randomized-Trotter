# CT-3B：実施判断へ戻す不足条件

2026-11-07 JST、固定参照 `6b0b6669bbf58ef22e789bfef134a1a6da9f2343`。[REPORT](REPORT.md)、[候補](FALLBACK_POLICY_CANDIDATES.md)、[停止/上界](POLICY_TERMINATION_AND_BOUNDS.md)。

## 現在の判断

**CT-3Aで`READY_FOR_LIMITED_IMPLEMENTATION`に達した候補はない。CT-3Bの実装scope、入力key集合、precision、word/価格取得件数は未設定とする。** 不足する条件を仮定したまま、実装が承認済み・開始予定であるとは扱わない。

SK型でstrict生成集合とideal exact-realの十分定数を記号的に具体化できたが、実netの全域coverage・eta0/t0・有限精度誤差/停止が不足する。数論法の条件付き2k式も現backendの全入力capにはならない。有限key成功を積み増すだけでは、この全support資格の不足を解消しない。

## 次にGPT／ユーザーが判断すること

| 担当 | 判断する事項 | 必要な根拠 |
|---|---|---|
| GPT／ユーザー | strict生成集合のkernel書換えとc=12の導出を採用可能な補題とするか | [Codex導出§2–3](POLICY_TERMINATION_AND_BOUNDS.md)の数学的確認。独立レビュー未実施 |
| GPT／ユーザー | SKのnet/有限精度保証を揃える費用に研究上の価値があるか | T比較に使えるcapを得る見込み。今回は改善率や上界の桁を推測しない |
| GPT／ユーザー | 別途の保証付き既存net/実装資料を供給するか、限定的な証明/構築作業を依頼するか | strict SU(2) coverage、実word、eta0/t0、誤差/終了の証拠が必要 |
| ユーザー | 条件が揃った後、新規実装・net/word生成・数値検証を許可するか | 対象SOURCE、入力/precision、取得する証拠、STOPを別途固定した明示承認 |

科学レビューと科学計算の許可は別。CT-3A自体は正式な採用/新規性GO・NO-GOレビューではない。今回の調査完了から次の研究を自動開始しない。

## 条件が揃った場合に再利用できる資産

以下は将来の設計材料であり、今から追加する実装の仕様ではない。

| 既存SOURCE/資産 | 使える部分 | そのまま移せない部分 |
|---|---|---|
| [track_c_h4_synthesis](../../../src/trottertracks/track_c_h4_synthesis.py) | exact binary target、exact cyclotomic word matrix、strict Frobenius guard | backend完了保証、cache即return、新fallbackのword長/精度規約 |
| [track_c2_b_verify.lower_native](../../../src/trottertracks/track_c2_b_verify.py) | written productの順序、global W、actual adjoint、SX/CX/X | reset/readoutを含むwhole処理、C2-BのBDB†誤差重み |
| [pr_reference_ct1](../../../src/trottertracks/pr_reference_ct1.py) | record validation、保存local/whole正確性の対象定義、joint/reset/測定、取得wordのprovenance | stage_bの無期限待ち/待機shutdown、q1 precision/有限成功の全域移送 |
| [pr_reference_ct0](../../../src/trottertracks/pr_reference_ct0.py) | literal angle/role、全support phase規則、保存のprecision headroom | 新precision採択や全域word価格の自動認定 |
| [pr_reference_rotation_v3](../../../src/trottertracks/pr_reference_rotation_v3.py) | native分類、role/出現数の対応 | 全support未知角度の価格、T-depth |
| [pr_reference_transfer_error](../../../src/trottertracks/pr_reference_transfer_error.py) | 旧raw disk/位相算術/aliasの規約 | synthesis diskの二重計上、新位相算法 |
| [track_c2_b_strong_baseline](../../../src/trottertracks/track_c2_b_strong_baseline.py) | 有限価格/実行可能解/下界の区別という会計上の教訓 | 旧optimizer/precision候補/LP下界を新A baselineへ貼ること |
| Qiskit discrete_basis | 既存SK再帰構造の参考 | strict coverage、有限精度保証、既定depth10 netの資格 |

必要な拡張があるなら、返却recordへbackend/fallback route・入力・epsilon・実word・phase・guard・T数と、そのcapの根拠を対応させる最小adapterを検討する。承認証明書やgrant発行基盤を新設する理由はない。旧cache/record/SOURCEを上書きせず、異なるpolicyの科学的provenanceを混同しない。

## 後の限定実行で固定すべき事項

実装候補が資格できた後に限り、ユーザーへ戻す案に次を含める。

- 対象：compiler後の実binary64、strict RZ、W/actual adjoint、既存B′/Reference構成を維持。全実角を保証する理論と、有限の実装検証集合を分ける。
- precision：CT-0の共通十分境界とCT-1のq1用precisionを区別し、正式な候補を事前固定。今回は新gridや採択値を設定していない。
- 証拠：netの全域証明、丸め込みの再帰/guard完了、capを保つ実word選択、有限遷移controller、signed targetとphase/order、保存誤差会計との整合。有限テストPASSは全入力証明の代用ではない。
- 検証対象：人工のphase/符号/境界対象と必要な保存targetを事前登録し、word/compile/価格をどこまで取得するか有限に限定する。今はkey、件数、実行順序を新たに選ばない。
- 公平性：B′/Referenceで同じprecision・取得/選択機会。未知価格を0埋めせず、最適word存在と実採用word価格を分ける。

これらは追加作業に必要な条件の一覧で、**未確定のCT-3Bを既定計画にしたものではない**。新しいnetやfallbackの実装が必要な理由は、現資産にstrict全域coverageと有限精度停止/上界の証拠がないためである。現在のコードに小さなhookを足すだけで全域保証を取得できるとは主張しない。

## STOPと承認境界

現在はCT-3Aの静的範囲を完了し、全support資格に必要な証拠が不足する地点で**mandatory STOP**。

将来の限定実行でも、strict phase/guard/cap/停止条件の不成立、必要入力の欠損、固定科学条件の変更要求、事前範囲の完了でSTOPする。backendが返らない場合の遷移を未保証のまま実行することや、成績改善のためにnet/precision/対象keyを自動拡張することはしない。

CT-3Bには別途の明示的な実行承認が必要。ユーザーのチャット承認で足り、承認を証明する新規artifactは不要。CT-3Bの実装/取得を承認されたとしても、その承認は全18round期待T、T-depth/FT depth、precision/alpha/shot最適化、別PF/law/gadgetや停止済みTrackの再開へ自動拡張しない。

次担当は**GPT／ユーザー**。Codexによる新実装・科学計算は未開始。
