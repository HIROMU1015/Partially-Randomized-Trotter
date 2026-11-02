# C1-1-D0 GPT・ユーザーhandoff

**`HASH_MISMATCH_CAUSE_PARTIALLY_IDENTIFIED`。診断1回、retry 0。D0 actual payloadを保存し、fragment 6・7のweight bit差を特定した。C1-1全体は再開していない。**

- Expected：`722ec0077b4505d963c03bccdd6311f9dd8b3d3a5ed1b9117f7a6f8fa85a7450`。
- D0 actual：`16b35c1dde1a6807d39c733defd87e315e4ac7317a831b240d0f7214e744a834`。canonical UTF-8 1,834 bytesのSHA256と一致。
- 保存H4 1.00 Å / STO-3G / DF12 / 8 qubit / prefix3の必要配列・H hash・順序は保存参照と一致。D0ではstate/sectorを読まず、basis/G6 exact λp/モデル差は未実行。
- 原因の到達点：同じsnapshotの既存rank9 B2-G/B2-W fragment記録と比べ、index6のweightはhex末尾 `...af0e` → `...af0f`、index7は `...2787` → `...2785`。113比較record中4件、独立のfield差は2つ。新しいrank/split探索なし。
- 未確定：元の完全なpartition payload、9..11の旧weight/proxy sum、旧binary/BLAS等。保存fragment記録は元の直接payloadではないため、2fieldだけが唯一の原因とは言わない。特定runtimeの責任や元C1-1のactual値も確定していない。
- Logger修正の人工4ケースと実診断で証拠保存を確認。raw stderr空、terminalとprocess exit2一致。SOURCE修正/再実行なし。
- 元C1-1のSOURCE/失敗/raw/auth/STOP、科学SOURCE等を保全。185pinsと旧実worktree26filesを照合。共有環境・他processへの変更なし。

## 次の担当・作業・承認（提案、未実行）

| 担当 | 次の作業 | 必要な判断・承認 |
|---|---|---|
| ユーザー・GPT | 元payload/実行環境の追加回収が必要か、派生float metadata差を保持した新partition来歴を検討するか判断 | identity受理方針と再開範囲。旧expected hashを黙って変更・無効化しない |
| Codex（別指示後） | その方針に沿った新C1-1 SOURCE/契約を準備し、比較前のbasis/component/identity証拠保存とloggerを固定 | 新しい準備指示。D0/旧C1-1成果物は変更しない |
| ユーザー | basis/exact入力/model差を含むC1-1再解析の再開可否を明示承認 | 新SOURCE/input/environment・必要処理・one-shot/retry/STOPを固定した別の実行承認 |
| Codex（その承認後のみ） | 許可されたC1-1残りの処理を実施して保全・公開する | 現在は未承認・未実行。adapter/native/sampling等はさらに別承認 |

技術的診断と科学GOを区別し、原著価値・資源優位・投稿GOは認定しない。

Branch：`track-b-c1-1-d0-partition-hash-20261021`。基点 `4cfe1c257709f0ac09351eadce3952b81403cf93`、D0実行SOURCE `835ae6f57eae43ba8452b71eae0ac9bce3dc2828`、新authorization `7e3ebb620cf3796b6c651c5dc6dfa03236730230`。最終HEADはpublication receiptおよび最終チャットを参照。

レビュー最小セット：[README](README.md)、[hash identity](result/hash_identity.json)、[canonical payload](result/partition_payload.canonical.json)、[float provenance](result/float_provenance.json)、[field比較](result/field_comparison.json)、[terminal](result/terminal.json)、[保全確認](preservation_verification.json)。**公開・保全確認後mandatory STOP。**
