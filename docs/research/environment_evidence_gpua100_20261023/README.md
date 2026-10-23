# GPUA100環境：既存PRリソース評価の証拠所在索引

2026-10-23 JST。repository: `HIROMU1015/Partially-Randomized-Trotter`、hostname: `GPUA100-gleap-01`。
既存資料の所在・byte identity・公開状態を整理したもの。科学結果、研究方針、検証status、候補順位は更新していない。
新規科学計算、再実験、SOURCEのimport/実行、signal/shot/costの再計算、再fit、再ランキング、論文執筆は0。

## 整理担当の入口

まず採択済み共通PF資料を参照し、独立confirmation、S2比較、非DF pilotをそれぞれのcohortとして読む。

|資料|固定commit|結果への入口|
|---|---|---|
|採択済みH2/H4/H6共通PFデータ|`f75a7c25d584b69d90de454731a7f343a421e059`|[既存README・dataset・候補/endpoint索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f75a7c25d584b69d90de454731a7f343a421e059/artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22/README.md)|
|共通PF最終結果・SOURCE来歴|`90f1cd55a0d963953696bd0a27d9d041ff939bb7`|[既存最終統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/90f1cd55a0d963953696bd0a27d9d041ff939bb7/artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/README.md)|
|H246再構成独立confirmation|`3eca46148070ba4d4b0f5815ef45943407cc959e`|[既存結果・入力・終了監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eca46148070ba4d4b0f5815ef45943407cc959e/artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/README.md)|
|固定S2再計算・原H2回収の比較|`a590214d6d70962aa02b9e0a2bce56810859522f`|[既存データ・物理診断・cohort](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a590214d6d70962aa02b9e0a2bce56810859522f/docs/progress/track_a/figures/pr_pf_s2_recomputed_20261022/README.md)|
|C/N/Gの保存値会計|`b81d4096e806efb6523304764d3a0680b7663320`|[既存定義・図・留保](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b81d4096e806efb6523304764d3a0680b7663320/docs/progress/track_a/figures/pr_pf_analytic_vs_circuit_CNG_20261022/README.md)|
|非DF H2の分離process結果|`c8990de7721545faff88b636108034dedb80a463`|[既存固定4方式比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/artifacts/non_df_h2_process_recovery/2026-10-23/comparison.json)|
|H2共通政策の不足資料・他環境metadata受領|`cc7404ed189f8c768ecc50a2066485975c9c68f4`|[既存受領報告・未解決事項](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc7404ed189f8c768ecc50a2066485975c9c68f4/artifacts/resource_applicability/track_a_pf_order_reversal_h2_metadata_receipt/2026-10-23/REPORT.md)|

上記は既にGitHubから取得できた資料であり、再登録していない。各commitの既存説明にある過去時点の「push0」は、その過去作業の記録である。今回のremote確認時点の可用性とは別に保持した。

## 機械可読索引

- [artifact_index.csv](artifact_index.csv)：23系列・2,834参照。系列ID、資料の役割、固定commit/path、Git blob OID、SHA256、bytes、公開状態、実取得の有無を結ぶ。科学数値の転記表ではなく、保存原値への参照表。
- [series_manifest.json](series_manifest.json)：系列ごとのSOURCE・入力・結果・条件・終了監査への入口、環境との関係、cohortと検証scopeの留保。
- [recorded_source_bindings.csv](recorded_source_bindings.csv)：22件の記録済みSOURCE関連commitを、manifest自身のcommit/path/SHAとJSON pointerへ結合。結果commitと科学SOURCE commitを同一視しない。旧SOURCE/input_SOURCEも別行で保持する。
- [worktree_catalogue.csv](worktree_catalogue.csv)：調査時に登録されていた既存worktree33件。branch名不在だけでは未公開としない。別branchから取得可能な歴史commitはcommit APIで確認した。
- [remote_retrieval_checks.json](remote_retrieval_checks.json)：結果CSV/JSON、採択済み圧縮dataset、入力snapshotを含む12ファイルを固定commitで実取得し、decoded byteのSHA256・サイズ・Git blob OIDを照合。主要SOURCE8 commitもAPIで取得可能と確認。
- [gaps_and_requests.json](gaps_and_requests.json)：優先した回収要求と、未取得・FAIL・未解決事項。既存の判断を変更しない。
- [private_custody_summary.json](private_custody_summary.json)：private原本の保全範囲と公開資料だけによる検証の限界。内部絶対パスやseed原本は含まない。
- [publication_review.md](publication_review.md)、[verification.json](verification.json)、[SHA256SUMS](SHA256SUMS)：新規候補の開示判断、技術照合結果、新規候補ファイルのSHA。

2,794参照は既存公開commitとlocal Git blobが一致した。40参照は既存ローカル限定commitの資料で、GitHub URLを付けていない。
後者は `c4cb5f3c9007f84b8d3798be17398ee2ea498e0f` の技術追補と、`8f0ff62b2bfc6676c31773261b160e3935ffcdae` の既存公開準備metadata。
今回のAPI取得は422だった。原本文、配置draft、原結果の追加公開は行わず、所在とfingerprintだけを公開候補へ整理した。

## 条件・定義を保つ読み方

科学数値を引用する場合は、対象ファイル内のHamiltonian/model、geometry、basis、DF rank/rank policy、分割 `L_D`、delta窓、cohort、compilerと受理scopeを必ず一緒に読む。
系列IDだけでこれらを補完しない。共通PF、独立confirmation、原H2探索、新cohort、旧rank固定研究、ordinary-control非DF pilotは別条件である。
WP11/baseline履歴の実行hostは今回の所在だけから推定せず、historical scopeを維持する。他環境から受領したH2 projectionも、このhostで原探索を実行した証拠にはしない。

固定時間C/N/Gでの `C_a` はcos/sin軸ごとの状態準備なしcompiled RZ/shot、`N_a` は保存bias・normalization・allowanceと統計式に基づく十分shot数、`G` は軸別 `N_a C_a` の和。
ここでいう回路費用 `C_a` と、PFの決定論的bias係数 `C` は別量である。
有限MCの費用SE/名目区間、QPE/RPEの統計失敗率、deterministic PF biasも混ぜない。
経験的allowanceを厳密certificateへ、十分Nを実測最小shotへ、`C_use`を厳密上界へ格上げしない。
独立 `N_circuit` がないCNG系列ではG00/G01のNは固定であり、N比・G10/G11を0や同一性の実証で埋めない。
保存bias・u・margin・費用・単位は変更していない。新しいratio、順位、最適点は計算していない。

共通PFの有限登録gridにおける既存採択と、最終全RPE/energy資源評価・一般的優位性・母平均保証・最終evidence freezeを区別する。
数値条件や結果の正本はリンク先の既存資料であり、この索引は新しい科学的結論を作らない。

## 調査・検査の範囲

自身の主repository、登録済みworktree、既知の実行/private保存領域、既存GAP回収報告を調査した。
`git ls-remote`によるremote ref確認、固定commitのGit treeとlocal byte照合、GitHub connectorからの実取得、stdlibによるJSON/CSV読取りとhash計算のみを使用した。
2,834参照の全件でlocal byteと記録commitのGit blobが一致。12ファイル以外は個別remote再ダウンロードを実施しておらず、remote commit identityとGit blob照合の範囲である。
既存SOURCE・結果・private control/raw等の50,159ファイルを前後SHAで確認し、50,158件が一致した。
不一致1件は今回の編集対象外だった主worktreeの `AGENTS.md` で、push失敗時の報告規則の追加を検出し、その変更を保全した。
既存worktreeのtracked状態の変化も同じguide1件だけだった。科学SOURCE・結果・grant・ledger・marker・STOPの変更は検出していない。

他host/SSH、他ユーザー領域、共有環境全体、未知の外部symlink、未参照保存先、unreachable Git objectは探索していない。
private archiveの展開・内容再解析、seed replay、科学runnerの再開は行わない。独立科学レビュー、CI、第三者再現の結果ではない。
この作業は新しい科学的validation pathや科学status更新ではないため、元の研究概要・validation manifest・科学SOURCE/resultを変更していない。新しい所在索引だけを追加した。

## 公開状態と停止

専用branch `codex/pr-evidence-gpua100-gleap-01-20261023`。新規候補はローカルcommitまで。
今回新規作成したファイル単位の公開承認は確認できないため、依頼第5節に従ってpushしない。
既存公開ファイルの複製、private原本、seed復元情報、内部絶対パス、未公開grant/receipt、raw/archiveを追加していない。
[MANDATORY_STOP.json](MANDATORY_STOP.json)は今回の整理作業だけの終了記録であり、元の科学STOP/grantを解除しない。

**mandatory STOP。** 公開、追加回収、科学実行、論文作成、既存研究の再開へ自動進行しない。
