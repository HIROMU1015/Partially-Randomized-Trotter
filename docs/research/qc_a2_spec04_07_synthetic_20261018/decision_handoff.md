# GPT・ユーザー向け引継ぎ：A2 SPEC-04〜07人工実装完了

**技術分類：`SPEC04_07_IMPLEMENTED_SYNTHETIC_PASS`。Mandatory STOP。** [報告](technical_report.md)／[SOURCE・独立性](source_api_and_scope.md)／[結果JSON](synthetic_test_results.json)／[索引](evidence_index.json)／[公開receipt](publication_receipt.json)。

## 今回完了したこと

1. SPEC-04：明示determinantからsigned平方の変分Uとexact Wを生成し、別occupation行列経路で検査。
2. SPEC-05：固定K0 raw BCH辞書、証拠付きdyadic sqrt上界、row K_SOS。generatorのstreamingとcheckerの明示Bを分離。
3. SPEC-06：centered capとgrouped相殺を保持する構造Pauli L1の両経路、各groupのmin。checkerは独立行列・traceで全係数を検査。
4. SPEC-07：同一d/α/contextでRF0/RF1/EW2と強化centered対照を比較。strict domain/open endpoint/退化/未解像/不一致を保存。科学的GOは判定しない。
5. 新60件＋旧無変更38件、合計98 methods・2,629 subtests PASS。静的AST/import/literal検査PASS。実入力は0件。

## 技術的に解決した点と残る限界

符号・添字一致の多重寄与・sqrt上界の丸め方向・Pauli支持集合・tail内相殺・centered cap係数・domain/確率境界・反復予算・証拠改変を検査した。未解決の人工test failure、SPEC-04〜07の未実装stageはない。重大な新しい科学問題は今回発見・認定していない。

一方、checkerの形式的正当性、platform/runtimeの認定、実入力での成功・費用は未検証。共通runtime/schemaとSPEC-01〜03 checkerは共有する。独立occupation/trace checkerは指数的であり、実分子に安価な証拠取得が可能と結論しない。I/O failureのpartial recordは返すが、OS killを回復するscientific runnerはない。full systemの運用準備完了とは異なる。

root resolution・d/α・trial・情報取得/償却・適用分子/geometry/basis/rank/sector/順序等の科学条件は今回採択していない。人工test設定はこれらの既定値ではない。原分子との対応、exact-ground access、合成/controlled scalar、alias/unwrap/readout、state preparation/native cost、best-known baseline envelope、総資源改善、研究価値は未確立または未評価。

## 次の担当・必要な別承認

| 担当 | 次に判断・実施する内容 | 現在の権限 |
|---|---|---|
| ユーザー | この技術成果を確認し、次に依頼する限定scopeを決める | 科学方針・A2採択・実入力・SPEC-08は未承認 |
| Codex | 追加の具体的な技術修正が依頼された場合のみ、今回の固定SOURCEと旧SPEC-01〜03を基に対応 | 自動探索・実入力評価・次期契約準備はしない |
| GPT／ユーザー | 重要な科学判断が必要とされた場合、別の開始承認後にレビュー・採否を判断 | 今回は新科学レビューを開始していない |
| 将来の実行担当 | 固定target・必要証拠・費用/算術条件・保存・source/contract/input identityと別実行承認が揃った場合のみ評価 | 現時点で実入力科学実行不可 |

SPEC-08・QPE/nativeを次の既定作業として提案・開始しない。科学条件・実入力certificate生成に進むには、それぞれの範囲をユーザーが新たに明示する必要がある。認可のない状態で低水準functionを実データに適用してはいけない。

**状態保持：A2 `NOT_ADOPTED`、A1 `CLOSED`、K1-F未承認。** 既存SOURCE/raw/科学契約/authorization/grant/ledger/marker、親worktreeと他研究を保全。公開照合後mandatory STOPを維持し、追加計算や研究採択へ自動移行しない。
