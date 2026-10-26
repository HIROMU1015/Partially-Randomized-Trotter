# 次のGroup 1：139件の保全と大量整理準備

2026-10-27 JST。PUBLIC_REDACTED。今回の実行範囲は非破壊保全・検証・計画まで。**139 worktreeは全件維持し、削除0。利用終了を確認済みとは扱わない。**

## 正確な集合・既存資産の再利用

固定commit `2caa236fa7332872b5a346f0744318d48d59f6ad` のLOCAL_234_FOLLOWUP.csvでTECHNICALLY_CLEAR_PENDING_OTHER_ENVとなった214 IDから、既存BATCH_75_INVENTORY.csvの正確な75 IDを差し引いた139件。重複0、Task Aの57件との交差0、Group1保護20件との交差0を確認した。元分類は変更せず、新しい状態を別fieldに記録した。過去75件の所有者判断・3環境照合を139件へ転用していない。

元の系列はTrack B116、旧Track B G6 6、Research C C1 4・C4 5、DF-native Track A 1、表現アルゴリズム3、その他4。既存のrecent_changed_paths・SOURCE/result/STOP公開locator・研究subtree、非公開原本と保全manifestを再利用した。recent pathは保全scope選択の入口であり、branch名/連番から対象ID集合を作っていない。研究単位の表示は既存系列とbranch/packet名を用いた作業分類で、科学的包含や成果採択ではない。

正式運用指示2026-10-25版の§8.1・§8.4・§10を実在ファイルで確認した。既存CSV/Git/匿名化helper、exclusive-copyとstable source SHA/Git blob verifierを再利用し、必要な差分だけ追加した。科学module/runner/verifierの実行・compile・sampling・数値解析・詳細科学レビューは0。

## 実際に保全したものとGit保持だけのもの

選定scopeは各IDの既存recent変更索引と固定SOURCE/result/STOP locatorに現れる研究packet・文書・結果/controlのsubtree＋最小限のcontext。repository全体を139回コピーしていない。独立cohort・失敗・中断・grant/ledger/marker/STOPは元bytesとID/branch/完全SHA/元相対pathのbindingを維持し、後続成功に置換しない。宣言hash全件の再解決やscience runtimeの可搬性・全研究依存の閉包を認定したものではない。

- **新規実コピー5,889 files／503,215,888 bytes**。本人所有・owner-onlyの新規非公開領域で空き容量・衝突を確認し、exclusive create・source前後の安定SHA-256／保存bytes／Git blob一致を検証した。source hardlink0・上書き0・失敗0。保存fileはowner read-only。新archiveだけへの書込みで元checkoutを変更していない。
- **既存archive 514固有版を再利用**。新規分と合わせた6,403固有file versionsに対し、139件の**26,390版別binding**でコピー元stable SHAと保存bytes・source inode独立性を確認した。同じbytesを反復コピーせず、各ID/commit/path/SOURCE対応を非公開完全manifestへ残した。同一blobは同一科学結果を意味しない。
- 通常コード・script/test・設定**981 bindingはGit参照のみ**で保持し、実コピー済みとは記さない。各139件は今回観測時点で同名GitHub tipと完全HEAD一致。将来もlocal/remote branch・objectsを保持する。
- sparse checkoutで未materializeだった文書/manifest等**953 bindingはGit参照のみ**。skip-worktree tagとtracked blob/refを確認し、元checkoutへmaterializeせず、保全コピー済み/内容hash検証済みとは扱わない。必要時は保持Git版から取得する。Git objectの永続保持と科学的に必要な固定版の取り違え防止は別条件。
- **15 binding・9固有の大きなtracked checkpoint/result（計384,562,915 bytes）を未コピーで保留**した。該当8 IDは下記。現在のGitHub branch/commit保持とは区別し、実ファイルの追加custodyを必要とするかを対象限定で判断する。勝手な大規模raw移送・独立backupはしていない。

公開[NEXT139_ARCHIVE_MANIFEST.csv](NEXT139_ARCHIVE_MANIFEST.csv)では、新規コピー、既存archive再利用、Git参照のみ、未コピー大容量保留をentry_type/actual_new_file_copied/bytes_preserved_in_archiveで区別した。新規物理コピー単位は5,889行で、26,390 bindingの完全対応は非公開原本にある。コピーしていないcode/sparse/大容量fileにSHA検証済みの表記をしていない。公開の相対pathは保持された公開Git版のtracked pathだけ。

保全copyは同一ローカルdisk上で別障害域backupではない。外部private raw/未記録原本の全custodyや独立backupを保証しない。元checkoutなしに選定した保存資料は読めるが、dynamic import・共有runtime/venv/cache・固定path条件を解除していない。.git/venv/共有環境/再生成cacheは一括コピーしなかった。Track A 1件のMatplotlib font cacheは既知schemaのcacheとして区別し、科学原本とは扱わない。

## 判断と次の大量整理単位

| 主判断 | 件数 | 残条件 |
| --- | ---: | --- |
| ARCHIVED_VERIFIED_CLEANUP_CANDIDATE | 131 | 選定scope保全済み。所有者の研究単位利用終了、A/B実path照合、具体的契約/custody、直前preflightと削除実行指示が必要 |
| PRESERVATION_ACTION_REQUIRED | 8 | 大きなcheckpoint/resultの実ファイル保全scope/custodyを別途判断。worktreeは維持 |
| VERSIONED_ONLY_PENDING_DEPENDENCY | 0 | 主判断として該当なし。ただしcode/sparseはGit参照保持であり、archive範囲外を全面保全済みとは言わない |
| KEEP_REQUIRED | 0 | 今回の既存索引と限定差分では具体的な必須物理保持義務は新たに未検出 |
| UNRESOLVED | 0 | 既知scopeの準備判断として該当なし。未記録契約・外部依存等のscope不明は各行に別fieldで維持 |

追加保全判断8 ID：**LOCAL-WT-0002, LOCAL-WT-0095, LOCAL-WT-0096, LOCAL-WT-0097, LOCAL-WT-0099, LOCAL-WT-0100, LOCAL-WT-0101, LOCAL-WT-0103**。0002の7 checkpoint/native IRは約251 MB、残るG10関連7 IDの2固有result版は約134 MBで、重複bindingを含む。これらは元checkoutとGit参照を維持したまま、既存の非公開保管領域に対象fileのみコピーする必要性を確認する案。容量合計は上記384,562,915 bytesで、同diskの追加コピーを独立backupと称さない。外部private rawの追加探索や転送は提案対象に無制限拡張しない。

| 作業単位 | 合計 | 選定scope保全済み候補 | 追加保全判断 |
| --- | ---: | ---: | ---: |
| 既存アルゴリズム開発／中間実装・診断・停止履歴 | 4 | 3 | 1 |
| DF_NATIVE_TRACK_A | 1 | 1 | 0 |
| REPRESENTATION_ALGORITHM_HISTORY | 3 | 3 | 0 |
| TRACK_B_D_E_HISTORY | 16 | 16 | 0 |
| TRACK_B_DF_CORRECTION | 19 | 19 | 0 |
| TRACK_B_HANDOFF_AND_DIAGNOSTICS | 9 | 9 | 0 |
| TRACK_B_G_SERIES | 23 | 16 | 7 |
| TRACK_B_G6 | 6 | 6 | 0 |
| TRACK_B_M1_M1S | 13 | 13 | 0 |
| TRACK_B_PARENT_ALLOCATION | 15 | 15 | 0 |
| TRACK_B_RA_D0 | 7 | 7 | 0 |
| TRACK_B_BF_R1_SP_EARLY_HISTORY | 14 | 14 | 0 |
| RESEARCH_C_C4_CIRCUITS | 5 | 5 | 0 |
| RESEARCH_C_C1 | 4 | 4 | 0 |

**優先はTrack B116件中の候補109件**。そのうちD/E 16、DF correction 19、M1/M1S 13、parent allocation 15、handoff/diagnostics 9を先にまとめると72件。次にRA D0 7、BF/R1/SP等14、G系列の保全済み16を個別契約とともに確認できる。G10の追加保全7件は別枠。全116件が一括削除可とはしない。

旧G6 6件は名称B3を現在のDF-native B3へ混同せず別群。Research C 9件は現在研究/別Codexの予定を特に研究担当へ確認する群として後順位。可視processにpath一致がなくHEAD/statusが旧snapshotと同じでも、利用終了と再開予定の不在は認定しない。その他候補7件も研究単位で判断する。

候補群の正確なID集合（削除指示ではない）：

- 既存アルゴリズム開発／中間実装・診断・停止履歴：LOCAL-WT-0046, LOCAL-WT-0270, LOCAL-WT-0330
- DF_NATIVE_TRACK_A：LOCAL-WT-0021
- REPRESENTATION_ALGORITHM_HISTORY：LOCAL-WT-0042, LOCAL-WT-0044, LOCAL-WT-0045
- TRACK_B_D_E_HISTORY：LOCAL-WT-0054, LOCAL-WT-0055, LOCAL-WT-0056, LOCAL-WT-0057, LOCAL-WT-0058, LOCAL-WT-0059, LOCAL-WT-0060, LOCAL-WT-0061, LOCAL-WT-0062, LOCAL-WT-0063, LOCAL-WT-0064, LOCAL-WT-0065, LOCAL-WT-0085, LOCAL-WT-0086, LOCAL-WT-0087, LOCAL-WT-0088
- TRACK_B_DF_CORRECTION：LOCAL-WT-0066, LOCAL-WT-0067, LOCAL-WT-0068, LOCAL-WT-0069, LOCAL-WT-0070, LOCAL-WT-0071, LOCAL-WT-0072, LOCAL-WT-0073, LOCAL-WT-0074, LOCAL-WT-0075, LOCAL-WT-0076, LOCAL-WT-0077, LOCAL-WT-0078, LOCAL-WT-0079, LOCAL-WT-0080, LOCAL-WT-0081, LOCAL-WT-0082, LOCAL-WT-0083, LOCAL-WT-0084
- TRACK_B_HANDOFF_AND_DIAGNOSTICS：LOCAL-WT-0089, LOCAL-WT-0133, LOCAL-WT-0134, LOCAL-WT-0170, LOCAL-WT-0171, LOCAL-WT-0172, LOCAL-WT-0173, LOCAL-WT-0174, LOCAL-WT-0190
- TRACK_B_G_SERIES：LOCAL-WT-0090, LOCAL-WT-0091, LOCAL-WT-0092, LOCAL-WT-0094, LOCAL-WT-0098, LOCAL-WT-0104, LOCAL-WT-0105, LOCAL-WT-0106, LOCAL-WT-0107, LOCAL-WT-0108, LOCAL-WT-0127, LOCAL-WT-0128, LOCAL-WT-0129, LOCAL-WT-0130, LOCAL-WT-0131, LOCAL-WT-0132
- TRACK_B_G6：LOCAL-WT-0109, LOCAL-WT-0110, LOCAL-WT-0111, LOCAL-WT-0112, LOCAL-WT-0113, LOCAL-WT-0114
- TRACK_B_M1_M1S：LOCAL-WT-0135, LOCAL-WT-0136, LOCAL-WT-0137, LOCAL-WT-0138, LOCAL-WT-0139, LOCAL-WT-0140, LOCAL-WT-0141, LOCAL-WT-0142, LOCAL-WT-0143, LOCAL-WT-0144, LOCAL-WT-0145, LOCAL-WT-0268, LOCAL-WT-0269
- TRACK_B_PARENT_ALLOCATION：LOCAL-WT-0146, LOCAL-WT-0147, LOCAL-WT-0148, LOCAL-WT-0149, LOCAL-WT-0150, LOCAL-WT-0151, LOCAL-WT-0152, LOCAL-WT-0153, LOCAL-WT-0154, LOCAL-WT-0155, LOCAL-WT-0156, LOCAL-WT-0157, LOCAL-WT-0158, LOCAL-WT-0159, LOCAL-WT-0160
- TRACK_B_RA_D0：LOCAL-WT-0161, LOCAL-WT-0163, LOCAL-WT-0164, LOCAL-WT-0165, LOCAL-WT-0166, LOCAL-WT-0167, LOCAL-WT-0169
- TRACK_B_BF_R1_SP_EARLY_HISTORY：LOCAL-WT-0177, LOCAL-WT-0178, LOCAL-WT-0179, LOCAL-WT-0180, LOCAL-WT-0181, LOCAL-WT-0182, LOCAL-WT-0183, LOCAL-WT-0184, LOCAL-WT-0186, LOCAL-WT-0189, LOCAL-WT-0191, LOCAL-WT-0192, LOCAL-WT-0194, LOCAL-WT-0197
- RESEARCH_C_C4_CIRCUITS：LOCAL-WT-0320, LOCAL-WT-0325, LOCAL-WT-0326, LOCAL-WT-0327, LOCAL-WT-0329
- RESEARCH_C_C1：LOCAL-WT-0321, LOCAL-WT-0322, LOCAL-WT-0323, LOCAL-WT-0324

## 依存・原本・契約の未確認事項

139件のHEAD/branchは原snapshotと一致し、tracked clean・新untrackedなし。既知guide/cache以外の新重要ignored資産は未特定。可視本人UID processの対象path一致0。既存ローカル234件scopeでSOURCE/path必須束縛を確認しなかった評価を再利用し、新規/HEAD変化checkout最近control/docのpath差分照合でも新一致0。未知namespace・他所有者process・未記録再開・shared mount/alias・固定契約の不存在証明ではない。

**A/Bの139実path照合は今回未実施**。以前の75件結果を流用して非依存としない。両環境へSSH/自動転送せず、次のID/branch/完全HEAD・非公開正規path・SOURCE/provider/input/STOP参照の最小表を非公開原本へ準備する。ユーザー自身が必要に応じ受け渡す場合も方法を強制しない。B3のjob/契約/保存実体は一切変更しない。

BF/R1/SP/G6の固定SOURCE・独立確認・失敗・STOP・grant消費/retry条件はGit版と選定packetへ保持したが、再実行権限や物理path責任を解除した意味ではない。具体的な旧SOURCE/provider/input契約や現在job束縛が見つかったIDはKEEP_REQUIRED等へ戻す。元20保護IDと旧SOURCE11件は今回集合から除外され、保持を確認した。

## 次の担当と実行承認

1. **ユーザー/研究担当**：上記研究単位で将来workspace利用終了と再開/他Codex利用予定を判断する。139件全部を終了確認済みにせず、IDを1件ずつ再棚卸しする方式へも戻さない。
2. **A/B担当**：各自身の既存SOURCE/provider/input・固定path/共有実体・現在/再開jobを、受領した最小対応表と対象scopeだけで確認する。BのDF-native L_D=0 B3と旧G6のB3を区別し、jobや共有環境へ変更しない。
3. **該当研究/ローカルCodex（次の指示後）**：8 IDの大きなtracked bytesのcustody必要性と具体的コピー案を扱い、他群は実依存とowner判断を対応付ける。sparse metadataやcodeは保持Git版から取り出せるが、今回実copyと混同せず必要なら対象限定で補う。
4. **ユーザー**：残条件が閉じた研究単位の正確なID集合と、checkout/登録だけの通常removeを別に明示指示する。
5. **ローカルCodex（別の具体的削除指示後）**：直前にHEAD/branch/status/untracked/ignored leaf・利用/lock・実path/common/backlink・保持ref・archive可読性・SOURCE/path差分を確認。合格IDへ通常removeを個別適用し、拒否はforce/prune/clean/gc/手動強制削除で回避しない。異常IDだけ保留し共通異常では全削除停止。

今回新たな削除実行承認を求めて進める作業はしない。必要な次段階を具体化した計画である。研究branch/main・科学SOURCE/raw/grant/ledger/marker/STOP・原本・archive・共有Git/runtimeを保持する。**139件の削除0、旧SOURCE3版55宣言の状態変更0、A/B接続・B3干渉・新規科学計算0。匿名化公開後mandatory STOP。**
