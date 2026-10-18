# GPT・ユーザーへの引き継ぎ

2026-10-18 JST。**G6全生成の部品については、AKのGreen特殊化、WANのordinary次数/IID生成とEuler、Taylor補償/CTSの収集、SPSのlibrary枠組み、ISの実proposal補正に具体的対応がある。指定資料だけでは、return収集後のparent/child lawから有限bitのbias/range・U予算までを同じaccessで再現する先行の全手順は確定できず、独立性も未確立である。**

## 完了した技術整理

[最重要の逐次表](generation_pipeline_alignment.md)はP1〜P12×7経路の84セルを扱う。元のM1〜M12 Evidence MatrixとD1〜D5を参照し、再作成・改変していない。具体的に次を整理した。

- ordinaryは既に非列挙。有限cutoffとsignを揃える対応があるが、個々のraw eventの回転位置・積順序を無断で同一視しない。
- Greenから有限係数を得る対応と、その係数をpositiveなparent/child samplingへ変換する対応を分離した。
- CTSのPauli収集とlayer samplingを強い対照として保持した。一般odd formal wordをPauli回転軸にする置換は認めない。
- `W=c̃/π`は既知IS型補正であり、ISが本質的に未知B_RAを要求するとは扱わない。係数・実π・認証Uの取得が別の比較対象である。
- 量子前の古典rejectと、量子error-detection後のZeroFill/Discardを区別した。moment、全trial N、期待量子回数NZ、event依存価格、hard capを分離した。
- finite-bitのmeanはM̃。係数bias、native/angle誤差、統計保証、算術countと全runtimeの境界を維持した。

[既知部品の組合せ監査](known_method_composition_audit.md)の結論は「既知部品からG6へ還元できる可能性はあるが、指定資料だけでは完全な還元を確認できない」。これは完全非同値や独立新規性の主張ではない。

## 科学レビューで判断すべき具体的な問い

1. 既知のGreen/Euler/rejection/ISを同じA0/A1 accessで許したとき、C1〜C3の全接続は既知手順のroutineな特殊化として十分説明できるか。それとも既存の正値域・suffix配分・有限bit保証の結合に原著的な構成価値が残るか。
2. 論文の中心を「同じevent lawの生成」「同じ有限meanと有限precision保証」「条件付きquery構成」のどの水準へ置くのが適切か。単なるlawの違いだけを貢献としない一方、普遍的資源改善を必要条件に追加しない。
3. 新しい還元証明が必要なら、どの先行algorithm、target、access、出力law、precision条件に対する命題が必要か。既存M1〜M12の証明と追加比較命題を区別する。

ここで科学レビューを実施・採択していない。新しい同値性/非同値性定理を証明していない。私的Claude/GPT原本を公開せず、強い資源NO-GO・修正後の性能GOのどちらも採択していない。

## 残る最小作業と担当

| 担当 | 次の作業案 | 承認と境界 |
|---|---|---|
| ユーザー | 本監査を使い、上の限定された科学レビューを開始するか判断する | 今回はレビュー開始未承認。公開完了から自動開始しない |
| GPT | 開始が明示承認された場合、既知合成への還元のroutine性・中心claimの価値/範囲を判断する | 認定すべきは独立構成貢献であり、Codexの技術statusを科学GOへ読み替えない |
| Codex | 具体的な未取得の先行algorithm/補足/公開SOURCEが指定された場合、その箇所だけ追加照合する | 今回の文献取得は完了。次の技術調査は別指示後。無制限探索へ拡張しない |
| Codex | 科学判断で新実装・数値証拠が必要と決まった場合、別scopeを扱う | 今回未承認。実装、科学実行、回路、保存値再解析には別の明示許可が必要 |

現時点で欠けているのは、単なる一次本文の取得ではなく全接続の比較判断である。新しい性能実験や未採択分解を、資料不足の代わりに追加しない。条件付き主張候補は[remaining_independent_claim_candidates.md](remaining_independent_claim_candidates.md)。

## 公開・保全とSTOP

作業基点はd4163acd、非Pauli Packetは128a0576からread-only参照。取得元は[source_inventory.json](source_inventory.json)で区別する。公開・保全記録は[verification.json](verification.json)と[publication_receipt.json](publication_receipt.json)。receiptが検証した内容commitと、そのreceiptを追加する子commitを区別し、自己hash循環を避ける。最終HEAD・全公開bytesの照合は完了チャットでも報告する。

既存2 Packetと科学SOURCE/results/grant/ledger/marker/claim/STOPは変更しない。親worktreeのtracked bytes・直下untracked bytes、登録済み他worktreeのHEAD/index/lockの比較範囲を明示し、非登録環境やすべてのuntracked内容まで監視したとは主張しない。

**公開・remote照合・保全確認後、mandatory STOP。** 重要科学レビュー、論文執筆、新SOURCE、新科学計算、他branch統合は自動開始しない。
