# P1-final-delta-checklist

[入口](README.md) / [handoff](handoff.md)。**将来の追加確認項目であり、確認作業・採択・実装・実行の自動承認ではない。** 現在の状態は`AWAITING_FUTURE_PUBLIC_FIXED_DRIVER_AND_INSTRUCTION`。Codex Aのlive成果を参照せず、完成時期・future commit/schemaを推測しない。

静的基準は公開準備`c8cbeb18f76aef2cb1329d8c810036372fa07cc0`、前回監査`ff0bed24ba8fdb361a2890646ef84347ae37d457`。補助SOURCE SHAは`35a4a0d8b28fd1a50791cc2afd09ac86800a1f37b488dc4cee5906864ea7f0e1`。この基準から**変わったSOURCE/fieldと新driver接続だけ**を確認する。

| ID | 公開後に必要な差分 | 固定済み基準／確認・STOP条件 |
|---|---|---|
| D01 | 最終driver SOURCE commit、manifest、起動入口、candidate/schema/SOURCE差分 | 同repositoryの公開固定commit/bytesで取得。G7/kernel/F3の既存pins、補助SOURCEの変更理由を確認。未公開なら保留。live worktreeを調べない |
| D02 | 方式集合・input・policy | P5_general_order、旧4方式、固定law/N/root/ceil/bias/confidenceの継承。Closed P5/CTSが追加されたら別scope/承認を要求。p/x/m一致だけでG9と統合しない |
| D03 | 採択されたcost estimand、Layer A/B、setup/observer/output schedule | BCのnull/proposalから何が変わったか確認。観測全workloadか母期待μか。stage exclusive、total/profile二重計上なし。CPUの統計保証は量子αと別 |
| D04 | X/Y・arm間のstate/cache/取得共有 | per-arm/layer setup回数、state不変、bitstream/seed、cold/warm/pool、lookup/hash/flush、parallel干渉の扱い。両側同機会・無料項なし |
| D05 | Angle τ/K_phi/J、enclosure・RZ誤差配分、actual adjoint/phase | 旧RZ ε、CQ allowance0、3ρ budgetとの結合。G9のevent2εをCQ0へ誤適用しない。新budget/physical errorなら未承認変更としてSTOP |
| D06 | 実call予算・counter・failure/保存接続 | 保存Nのverbatim load、採択scheduleの全trial数と初回correctness範囲、budget/reference再実行なし、失敗prefix保全、one-shot/retry/STOP。本照合自体はclaimを発行・消費しない |
| D07 | Provider/encoding/basis/sector、direct/helper、native誤差 | BC nullを何にbindしたか。G9 contextならR08/R11条件を確認。opaque operator_idだけで同一視しない。未選定ならC1は保留 |
| D08 | Price/schema/cache/certificate binding | 9-field requestと具体provider/tool/precision/optimization manifest、RZ full key・19既存資産、forward hash/actual adjoint、event/primitive粒度、未知nullを確認。metadata guardだけを数値再認証と呼ばない |
| D09 | G9費用座標との接続 | per-event/per-trial/two-axis、G7/G9 N/policy、T/CX/1Q/outer項、workspace・depth・C2のscope。既存summaryで粒度不足なら追加B2/P2を今回行わず判断へ返す |
| D10 | 採択・承認・公開・保全 | 正式契約と科学実行承認を区別。旧SOURCE/raw/契約/claim/STOP不変、新driver未実行/準備完了/実行済みを分離。公開SHA/bytesと保全receiptを確認しmandatory STOP |

## 合格・保留の意味

- D01〜D10の対応が閉じれば、公開driverへの**技術的なP1照合**を完了できる。実入力correctness・実費順位・科学的GOの完了ではない。
- G9 providerが採択されない場合、G9既存native比較は独立scopeの引用として残せるが、新C1との直接結合は保留する。
- 不一致は消去せず、`CONTRACT_DIFFERENCE`、`ADDITIONAL_BINDING_REQUIRED`、`ADDITIONAL_EVIDENCE_REQUIRED`等として記録する。新数値評価を差分確認へ混ぜない。

## 繰り返さない事項

G7/G9の基本数式・方式、16/22 axes・3ρ/8ρ差、既存G9のT-primary/CX結論、Closed P5の形式係数確認、先行研究調査、19キーの過去取得は固定証拠を再利用する。旧数値verifier・sampler・compilerを「確認」の名で実行しない。

最終公開版で当該field/SOURCE/hashが変わった場合だけ影響範囲を読む。科学条件の新規採択、追加B2/P2/P3、新native取得は別途ユーザー判断・必要な実行承認へ戻す。**今回の作業はここでmandatory STOP。**

<!-- FIXED_REFERENCES -->
