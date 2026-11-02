# GAP-001 / GAP-008 第2環境の資料探索報告

2026-10-23 JST。環境ID `ENV-LOCAL-02`。前回の[探索報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dfdfa6460f9e7d25a9595132a672e8de2a5dc7e/docs/research/pr_algorithm_improvement_landscape_20261023/local_environment_search_20261023/SEARCH_REPORT.md)とGAP別報告・来歴JSONを確認した。対象は未発見原資料の所在と来歴だけである。

| GAP | 今回の判定 | 発見・不足 |
|---|---|---|
| GAP-001 | `FOUND_UNVERIFIED`、未解決 | 指定JSON 3件とtheta sweep JSON/CSV 4件を発見。独立した原本identityと元科学runへのbindingは未確定 |
| GAP-008 | `NOT_FOUND_IN_SEARCH_SCOPE`、未解決 | 今回重点の修正後canonical集約/split/indexと指定元anchor rawは未発見。関連資料のみ存在 |

GAP-008の判定対象は今回新たに求められた原資料である。前回確定した旧無効化37件・旧screeningのhistorical identityを取り消す意味ではない。

## GAP-001

次の指定資料の保存ファイルが見つかった。

- `H2_H6_2nd_grouped_uwc_alpha_bliss_quadratic_theta_0p01_gpu.json`
- `H3_bliss_sector_scaling_diagnostics.json`
- `H2_H6_2nd_grouped_uwc_alpha_simple_shift_gpu.json`

theta sweepのJSON/CSV 2組も見つかった。独立した期待raw hashがないため、ファイル名一致や保存proseとの対応を原本のbytes一致とは扱わず、全体を`FOUND_UNVERIFIED`とする。

固定公開noteとの保存値照合では、BLISS比較表10行、H3診断time-grid 4行、H3 theta 6行が公開表の表示値と一致した。別sweepのJSON/CSVでは80行とtheta0 10行の保存キー・alphaが一致した。これは保存済み資料の対応検査で、再fit、性能再評価、再ランキングではない。

保存条件にはHamiltonian hash、sector、grouping、fit入力、backend profileがある。H3は旧N_target=2と区別し、修正後3の保存metadataだった。SOURCE snapshotは存在するが、原runの実行commit・正確なcommand・当時の依存版・H/state/grouping payloadへの独立joinは未確認。新しい科学的結論や再現性の成立を主張しない。

発見原本はGit管理外の非公開保存資料として扱った。bytes/SHA256、算出Git blob OID、内部パス、保存条件の詳細は非公開記録に分離した。既存Git object内の原本identityは確認できず、算出OIDを実在するcommitted blobと表現しない。この公開用branchに原本やprivate fingerprintを収録しない。

## GAP-008

| 今回対象 | 判定 |
|---|---|
| 修正後canonical Cgs集約表 | `NOT_FOUND_IN_SEARCH_SCOPE` |
| 修正後split table / index | `NOT_FOUND_IN_SEARCH_SCOPE` |
| `H3_H14_df_screening_anchor_cgs`系の元raw | `NOT_FOUND_IN_SEARCH_SCOPE` |
| 元科学runの入力manifest、実行SOURCE、command、生成log、runtimeとのbinding | `NOT_FOUND_IN_SEARCH_SCOPE` |
| 別refinement/max-L_D資料、補助log | `FOUND_UNVERIFIED`、元canonical/anchorへのbindingなし |

旧screening logの保存12行は既存stale結果の保存行に対応していた。新たな科学的成果・修正後結果として採用しない。別refinement資料も原本の代用にしない。旧`DO NOT USE / stale`、既存STOP、無効化履歴を保持する。前回確定済みの旧37ファイルを回収・再評価していない。

## 探索範囲と未調査範囲

非重複の既知研究保存領域7 root、ファイル名234378件を確認した。主PR repository、同一研究の登録worktree32件、関連DF/高次PF研究repository、本人の既知worktree保存先、既知backup/data、登録された一時worktreeを含む。旧root aliasは二重計上しない。ignore対象・隠しファイルも含め、依存環境・Git内部filesystem走査を除外した。対象pathの既存Git履歴は別途確認した。

関連と判断したarchive13件は展開せずmember一覧を読み、対象raw/canonical名は見つからなかった。関連archive中のファイル内容・nested archiveや通常の圧縮数値ファイル内部は検査していない。SOURCEは読み取りのみで、import・実行していない。

別host、SSH、他ユーザー領域、未知の保存先、アプリcache、未知symlink先、Git unreachable/dangling object・reflogは未調査。確認したroot・archive一覧に権限エラーはなく、未調査範囲を`ACCESS_UNAVAILABLE`とはしない。不存在を別環境へ一般化しない。

## 情報の分離と停止

公開用記録は本報告と公開用scope/status/verificationだけ。原本・private fingerprint・内部絶対パス・詳細run metadataはローカル非公開記録に分離し、GitHubへ公開・転送していない。

次環境へ引き継ぐGAP-IDは **GAP-001、GAP-008**。GAP-001の残課題は発見資料の元run来歴、GAP-008は修正後canonicalと元anchorの原本・来歴である。次環境の選択と探索指示はユーザーに委ねる。

**mandatory STOP。** 科学計算、simulation、fit、再ランキング、再実験、新しい科学的GO、原本回収・公開・移送、元研究の再開を自動開始しない。既存SOURCE/raw/grant/ledger/marker/STOPおよび研究statusを変更していない。
