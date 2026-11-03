# サーバー2：固定SOURCEと保存成果の追加収録

2026-11-03 JST。ローカル側で原blobまたは固定SOURCE contextを確認できなかったserver2の197要求を、保存済みの原本と最新mainに照合した。比較mainは `0c49a88134066177816feca2741f36f6b88d23ba`。server1の69要求は今回の対象外である。

## 今回の配置

| 区分 | ファイル数 | 扱い |
|---|---:|---|
| mainにない研究報告・時系列記録 | 13 | 既存pathを上書きせず、元blob・modeで追加 |
| SOURCE・test・固定artifact・過去の索引等 | 213 | 保存原版 `.original`。現行APIや別cohortへ採用しない |
| 合計 | 226 | 7770303 bytes。重複を除いた配置ファイル数 |

[全ファイルの出典・blob・mode・配置対応](server2_main_collection_20261103_files.tsv)と[保存原版の入口](../archive/server2_main_collection_20261103/README.md)を参照する。要求行数、ファイル数、研究数は同じではない。既存mainの同一blob・modeは追加していない。

## 追加した報告の入口

- [h4_trajectory_workers.md](../h4_trajectory_workers.md)
- [track_a_h246_higher_pf_confirmation_run01_results_20261021.md](track_a_h246_higher_pf_confirmation_run01_results_20261021.md)
- [track_a_h2_m5_existing_pf_comparison_results_20261020.md](track_a_h2_m5_existing_pf_comparison_results_20261020.md)
- [track_a_h2_m5_phase_guard_20261020.md](track_a_h2_m5_phase_guard_20261020.md)
- [track_a_h4_mc_supplement_20261019.md](track_a_h4_mc_supplement_20261019.md)
- [track_a_h8_confirmation_parallel_resume01_20261019.md](track_a_h8_confirmation_parallel_resume01_20261019.md)
- [2026-10-18-h8-new-pr-confirmation-design.md](研究ノート/2026-10-18-h8-new-pr-confirmation-design.md)
- [2026-10-18-h8-new-pr-confirmation-run01.md](研究ノート/2026-10-18-h8-new-pr-confirmation-run01.md)
- [2026-10-19-h8-confirmation-parallel-resume01.md](研究ノート/2026-10-19-h8-confirmation-parallel-resume01.md)
- [2026-10-19_h4_mc_supplement_adoption.md](研究ノート/2026-10-19_h4_mc_supplement_adoption.md)
- [2026-10-20_h2_m5_comparison_completion.md](研究ノート/2026-10-20_h2_m5_comparison_completion.md)
- [2026-10-20_h2_m5_phase_guard.md](研究ノート/2026-10-20_h2_m5_phase_guard.md)
- [2026-10-21_h246_confirmation_run01.md](研究ノート/2026-10-21_h246_confirmation_run01.md)

## SOURCE・版・公開範囲

元の科学SOURCE・入力identity・runtime・status・quarantine/missing・実行契約・STOPは各原記録を使う。この収録commitや保存用の参照を科学SOURCEへ付け替えない。固定SOURCE contextは対応表に識別子を記すが、元commitの全体を公開したことやGitHubから取得可能なことは主張しない。公開履歴の親は比較mainのみであり、原本のローカル履歴をmergeしていない。

Python SOURCEとtestは、保存原版から元のcontextへ戻るための資料であり、mainの実行pathへの新しい採用ではない。固定artifactも別のSOURCE・入力・cohortの既存main版へ混ぜない。追加報告の相対リンクが保存元の配置を前提とする場合は、対応表から該当する保存原版を参照する。私有の元contextについて取得可能な公開URLを作っていない。

B3のlocal版 `f794a2aa4c76bc024540c808bf9400d9f523bb76` と既知の公開版 `ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed` の分岐は維持した。同じ研究名だけでSOURCEや結果を同一視しない。圧縮rawを展開して科学値の同一性を新判定していない。

H8のlocal-only・共有未承認資料、非共有の原本対応、SOURCE withheldの指定、私有保存先を含む原版は公開対象に含めない。大きなraw・NPZ・QPY・checkpoint・cacheや既存bundleをこの収録へコピーしていない。5 MiBはデータの配置判断の目安であり、科学実行の条件を変更するものではない。個々の非公開保持・大データ保持・残件は非公開結果報告で管理する。

## 技術確認と限界

追加した全原版は出典のGit blob・modeと照合した。Python 57件はASTのみ、JSON 73件は構文のみを確認し、表形式は構造の読取りに限定した。新しい案内と対応表のリンク・参照・重複を確認した。

Pythonのimport、元の研究runner/test、科学計算、数値再解析、回路build/compileは0回。SOURCE全体の実行互換性、main全体の回帰検査や科学性能の再認定は行っていない。過去の技術PASSや原結果を今回の科学実行結果として扱わない。

元repo・worktree・SOURCE・入力・raw・既存bundle・grant・ledger・marker・STOPは変更せず、jobを停止・凍結していない。枝全体のmerge、削除、履歴だけをつなぐmergeは行っていない。
