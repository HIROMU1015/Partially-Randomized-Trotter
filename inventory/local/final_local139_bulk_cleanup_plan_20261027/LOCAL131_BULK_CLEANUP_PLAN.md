# LOCAL131件：研究単位による大量整理の実行前計画

PUBLIC_REDACTED_FINAL_LOCAL139_BULK_PLAN。2026-10-27 JST。
固定LOCAL保全入力`96cf3dae3bd03d6f64f384fefedc6e88e782eef6`、A139入力`8da04de4aae1ccf9c0e0327411ce9081cf78f491`、B_PATH_139入力`ee30e8b435be6b15bba201de923b61eb02d9c54e`。

**既存131件をすべてBULK_APPROVAL_CANDIDATE_PENDING_OWNERとした。研究単位の利用終了・再開/他Codex利用判断と、具体的な削除実行指示、操作直前preflightが残る。今回139件のworktree削除・登録解除は0。** 所有者判断済み0、一般的UNKNOWNを理由とした全件再調査要求0。既知3環境scopeで必須物理依存を検出したIDは0だが、全環境の依存不存在を証明した意味ではない。

## 正確な131件と追加8件の分離

元集合は`96cf3dae3bd03d6f64f384fefedc6e88e782eef6`の[NEXT139_PRESERVATION_STATUS.csv](../next_group1_139_preparation_20261027/NEXT139_PRESERVATION_STATUS.csv)でARCHIVED_VERIFIED_CLEANUP_CANDIDATEだった正確な131 ID。139件の集合は234件technical-clear214から既存75件を引いたものと一致する。今回の大容量保全で8件の状態が進んでも、131件の集合は拡張していない。

下表の「131対象ID」列の和集合が、将来**一度の具体的実行承認で扱える最大131 ID**である。研究単位の利用終了と適用される固定条件が揃った単位を選び、その和集合を承認対象にできる。72件ごとの小分け承認・実行を必須にはしない。通常removeは各IDへ逐次適用し、個別異常はそのIDだけ保留する。保全条件を解消した追加8件は右列の別枠で、将来131件の指示へ自動追加しない。

| 研究単位 | 131対象数 | 正確な131対象ID | 別枠の追加候補8件のID | 所有者の現在判断 |
| --- | ---: | --- | --- | --- |
| Track B D/E履歴 | 16 | LOCAL-WT-0054, LOCAL-WT-0055, LOCAL-WT-0056, LOCAL-WT-0057, LOCAL-WT-0058, LOCAL-WT-0059, LOCAL-WT-0060, LOCAL-WT-0061, LOCAL-WT-0062, LOCAL-WT-0063, LOCAL-WT-0064, LOCAL-WT-0065, LOCAL-WT-0085, LOCAL-WT-0086, LOCAL-WT-0087, LOCAL-WT-0088 | なし | OWNER_DECISION_PENDING |
| Track B DF correction | 19 | LOCAL-WT-0066, LOCAL-WT-0067, LOCAL-WT-0068, LOCAL-WT-0069, LOCAL-WT-0070, LOCAL-WT-0071, LOCAL-WT-0072, LOCAL-WT-0073, LOCAL-WT-0074, LOCAL-WT-0075, LOCAL-WT-0076, LOCAL-WT-0077, LOCAL-WT-0078, LOCAL-WT-0079, LOCAL-WT-0080, LOCAL-WT-0081, LOCAL-WT-0082, LOCAL-WT-0083, LOCAL-WT-0084 | なし | OWNER_DECISION_PENDING |
| Track B M1/M1S | 13 | LOCAL-WT-0135, LOCAL-WT-0136, LOCAL-WT-0137, LOCAL-WT-0138, LOCAL-WT-0139, LOCAL-WT-0140, LOCAL-WT-0141, LOCAL-WT-0142, LOCAL-WT-0143, LOCAL-WT-0144, LOCAL-WT-0145, LOCAL-WT-0268, LOCAL-WT-0269 | なし | OWNER_DECISION_PENDING |
| Track B parent allocation | 15 | LOCAL-WT-0146, LOCAL-WT-0147, LOCAL-WT-0148, LOCAL-WT-0149, LOCAL-WT-0150, LOCAL-WT-0151, LOCAL-WT-0152, LOCAL-WT-0153, LOCAL-WT-0154, LOCAL-WT-0155, LOCAL-WT-0156, LOCAL-WT-0157, LOCAL-WT-0158, LOCAL-WT-0159, LOCAL-WT-0160 | なし | OWNER_DECISION_PENDING |
| Track B handoff/diagnostics | 9 | LOCAL-WT-0089, LOCAL-WT-0133, LOCAL-WT-0134, LOCAL-WT-0170, LOCAL-WT-0171, LOCAL-WT-0172, LOCAL-WT-0173, LOCAL-WT-0174, LOCAL-WT-0190 | なし | OWNER_DECISION_PENDING |
| Track B RA D0 | 7 | LOCAL-WT-0161, LOCAL-WT-0163, LOCAL-WT-0164, LOCAL-WT-0165, LOCAL-WT-0166, LOCAL-WT-0167, LOCAL-WT-0169 | なし | OWNER_DECISION_PENDING |
| Track B BF/R1/SP履歴 | 14 | LOCAL-WT-0177, LOCAL-WT-0178, LOCAL-WT-0179, LOCAL-WT-0180, LOCAL-WT-0181, LOCAL-WT-0182, LOCAL-WT-0183, LOCAL-WT-0184, LOCAL-WT-0186, LOCAL-WT-0189, LOCAL-WT-0191, LOCAL-WT-0192, LOCAL-WT-0194, LOCAL-WT-0197 | なし | OWNER_DECISION_PENDING |
| Track B G系列 | 16 | LOCAL-WT-0090, LOCAL-WT-0091, LOCAL-WT-0092, LOCAL-WT-0094, LOCAL-WT-0098, LOCAL-WT-0104, LOCAL-WT-0105, LOCAL-WT-0106, LOCAL-WT-0107, LOCAL-WT-0108, LOCAL-WT-0127, LOCAL-WT-0128, LOCAL-WT-0129, LOCAL-WT-0130, LOCAL-WT-0131, LOCAL-WT-0132 | LOCAL-WT-0095, LOCAL-WT-0096, LOCAL-WT-0097, LOCAL-WT-0099, LOCAL-WT-0100, LOCAL-WT-0101, LOCAL-WT-0103 | OWNER_DECISION_PENDING |
| 旧Track B G6 | 6 | LOCAL-WT-0109, LOCAL-WT-0110, LOCAL-WT-0111, LOCAL-WT-0112, LOCAL-WT-0113, LOCAL-WT-0114 | なし | OWNER_DECISION_PENDING |
| Research C C1 | 4 | LOCAL-WT-0321, LOCAL-WT-0322, LOCAL-WT-0323, LOCAL-WT-0324 | なし | OWNER_DECISION_PENDING |
| Research C C4 | 5 | LOCAL-WT-0320, LOCAL-WT-0325, LOCAL-WT-0326, LOCAL-WT-0327, LOCAL-WT-0329 | なし | OWNER_DECISION_PENDING |
| DF-native Track A | 1 | LOCAL-WT-0021 | なし | OWNER_DECISION_PENDING |
| 表現アルゴリズム履歴 | 3 | LOCAL-WT-0042, LOCAL-WT-0044, LOCAL-WT-0045 | なし | OWNER_DECISION_PENDING |
| その他の開発・中間検証 | 3 | LOCAL-WT-0046, LOCAL-WT-0270, LOCAL-WT-0330 | LOCAL-WT-0002 | OWNER_DECISION_PENDING |

Track B116件のうち本計画内109件。そのうちD/E16＋DF correction19＋M1/M1S13＋parent allocation15＋handoff/diagnostics9＝**優先72件**であり、残るRA D0 7＋BF/R1/SP14＋G系列の保全済み16＝37件も同じ最大131集合へ含めた。優先順であって72件ずつ処理する制約ではない。旧G6 6、Research C9、その他保全済み7も上表の別単位として含めた。

## 研究単位で回答する所有者判断表

現在は14単位すべてOWNER_DECISION_PENDING、139 IDすべて利用終了未確認。過去75件の利用終了方針やA/B照合完了を転用しない。ユーザーは単位ごとに次のいずれかを回答でき、139件を1件ずつ再回答する必要はない。

- WORKSPACE_END_APPROVED：この単位の専用checkoutを今後作業/再開/別Codex用に使わない。資料・Git・契約は保全する。
- WORKSPACE_KEEP：専用checkoutを作業場所等として維持する。用途や例外IDがある場合だけ示す。
- OWNER_DECISION_PENDING：まだ判断せず維持する。

この表の判断は**workspace用途の終了判断**であり、削除実行許可と同義ではない。後日の一つの明示指示に、単位ごとの終了判断と正確な131件以内の実行範囲を併記することは可能。承認証明書・grant・署名・保存済み承認文等を要求しない。

各単位の判断材料は以下のとおり。具体的なSOURCE/result/STOP locatorとGit保持refは[139件統合CSV](LOCAL139_THREE_ENVIRONMENT_RECONCILIATION.csv)のID行、実pathと操作・復旧情報は非公開表で解決する。

| 研究単位 | 全ID数 | 131対象 | 別枠8件 | A元主判断内訳 | B元主判断内訳 |
| --- | ---: | ---: | ---: | --- | --- |
| Track B D/E履歴 | 16 | 16 | 0 | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":16} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":16} |
| Track B DF correction | 19 | 19 | 0 | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":19} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":19} |
| Track B M1/M1S | 13 | 13 | 0 | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":13} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":13} |
| Track B parent allocation | 15 | 15 | 0 | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":15} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":15} |
| Track B handoff/diagnostics | 9 | 9 | 0 | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":9} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":9} |
| Track B RA D0 | 7 | 7 | 0 | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":7} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":7} |
| Track B BF/R1/SP履歴 | 14 | 14 | 0 | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":13,"HISTORICAL_REFERENCE_ONLY":1} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":14} |
| Track B G系列 | 23 | 16 | 7 | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":16,"PRESERVATION_OR_CONTRACT_GATE":7} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":23} |
| 旧Track B G6 | 6 | 6 | 0 | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":6} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":6} |
| Research C C1 | 4 | 4 | 0 | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":4} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":4} |
| Research C C4 | 5 | 5 | 0 | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":5} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":5} |
| DF-native Track A | 1 | 1 | 0 | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":1} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":1} |
| 表現アルゴリズム履歴 | 3 | 3 | 0 | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":3} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":3} |
| その他の開発・中間検証 | 4 | 3 | 1 | {"PRESERVATION_OR_CONTRACT_GATE":1,"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":3} | {"NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE":4} |

全単位で、選定研究packetの実ファイル保全＋コード/sparseのGit参照保持を確認した。上表の8件は今回大容量9版15 bindingも独立実ファイルへ保全した。どの単位も、各SOURCE/input/result・独立cohort・失敗/診断・grant/ledger/marker/STOPと固定one-shot/retry/resumeを保持する。代表文書や同じGit blobだけで他証拠を置換しない。

現在の可視本人UID processのcwd/引数に139対象path一致は0、139件のHEAD/branch/status/既知ignored leafは旧記録と一致し、lockなし。既知scopeで具体的な物理worktree保持義務は未特定。ただし、将来再開・他Codex利用の可能性は**未確認**であり、ownerが上表の研究単位で判断する。特にResearch C C1/C4の9件は現行研究や別Codexの作業予定との関係を確認する。Track B/G6の旧B3とサーバーBの現在のDF-native L_D=0 B3を混同しない。

## A/B依存確認の扱い

Aは130 NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE、1 HISTORICAL_REFERENCE_ONLY、8 PRESERVATION_OR_CONTRACT_GATE。Bは139 NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE。すべて正確なID/branch/完全HEADがLOCALと一致し、両方のscopeで139件の元checkout必須物理依存は未検出だった。両環境の未受領・実パス照合未実施という前回条件は解消した。A/B再SSHや同じ全件検索は不要。

AのLOCAL-WT-0184は過去参考commitへの歴史参照であり、固定SOURCEの物理供給義務ではない。Aの8 gateはローカルの未コピー条件を転記した元判断。今回はLOCALで大容量保全を完了したが、A元主判断と副条件は書換えず、LOCALの解消状態を別fieldへ追加した。Bの元大容量副条件も同様に履歴として残した。

未記録mount/alias/export/bind・跨host実体、別namespace/他job、相対/動的path、未索引契約・外部private custodyは残余リスク。今回の最小表に含まれないLOCAL common Gitやguide先の直接照合までA/B完了と表現しない。具体的兆候があるIDだけ限定確認を提案し、一般UNKNOWNを理由に全件監査を無期限に繰り返さない。具体的な固定契約や共有jobへの危険が判明した場合は、そのIDを保留し、単なる削除承認で契約保護を解除しない。

## 保全・Git保持・独立backup

従来の研究packet26,390 bindingは、5,889新規実files／503,215,888 bytesと既存archive514固有版、合計6,403固有版へ対応する。今回そのmetadata不変と過去SHA/Git blob検証を再利用し、無条件再コピー・再hashはしなかった。コード981 binding、sparse未materialize953 bindingはGit参照のみであり、独立実コピー済みとは扱わない。保持Gitから正しい版を取り出せる条件を維持する。全139件の同名remote tipは今回も完全HEAD一致。

元checkoutから独立した保全filesの閲覧、Git object/refによる版保持、科学的再実行/再現性、独立障害域backupは別条件。保存先は同じローカルdiskで独立backupではない。外部rawや科学runtimeの全custody/可搬実行までは保証しない。必要な新しい具体的未保存原本や破損兆候が出たIDだけ保全scopeへ戻す。

## 別枠8件の保全解消

**LOCAL-WT-0002, LOCAL-WT-0095, LOCAL-WT-0096, LOCAL-WT-0097, LOCAL-WT-0099, LOCAL-WT-0100, LOCAL-WT-0101, LOCAL-WT-0103**。今回9固有files／384,562,915 bytesを1版1コピーで保全し、15版別bindingを個別保持した。source前後SHA-256・保存bytes・Git blob一致、regular/readable・source inode非共有、上書き0。8件はPHYSICAL_ARCHIVE_VERIFIEDになり、将来の追加候補としてはowner判断・直前preflight・別の具体的実行範囲が残る。科学SOURCE/path契約の解除、将来再開不用、独立backupを認定した意味ではない。

131件の承認集合へこの8件を自動追加しない。追加8件のcheckoutも今回全維持。9固有fileの個別Git/実copy/bytes/検証・契約残条件は[大容量custody CSV](LOCAL139_LARGE_FILE_CUSTODY.csv)にある。

## 将来の一括実行方法

1. ユーザーが研究単位の利用終了と正確な131件以内のID集合について、LOCALの対応checkout directory＋worktree登録だけを除去する範囲を明示する。上表の和集合を一度の実行指示にできる。WORKSPACE_KEEP/PENDING単位や例外IDは除外し、追加8件や保護20件/旧SOURCE11件へ拡張しない。
2. ローカルCodexが非公開対応表で正規完全path、owner、Git common・admin/gitfile backlink、登録lockを操作直前に照合する。主checkout/common Git/archive/別worktreeや予期しないmount実体を対象にしない。実行sandboxとnative namespaceの表示差は同一LOCAL inode/Git identityで根拠を確認し、未知aliasを勝手に許容しない。
3. HEAD/branch、tracked/untracked、ignoredの**配下内容**と重要資産差分を確認する。normal removeがignoredを消し得るためGit clean判定だけに依存しない。既知guide/cacheと科学原本・raw/checkpoint/STOPを区別する。
4. 現在job/他Codex・再開予定・SOURCE/provider/input/path依存の変更、保持local/remote branch・固定commit/object・SOURCE/result/STOP refs、archive/manifest可読性・source独立性を必要な差分範囲で確認する。既存検証を再利用し、変化した必要fileだけhashする。owner回答や過去snapshotを将来の直前確認にしない。
5. 合格した明示承認IDへ、通常の`git worktree remove`を個別・逐次適用する。この操作はdirectoryと登録を除去する。branch・重要commit・common objects・原本・archive/manifest・科学契約・他worktreeを維持する。今回はコマンドを実行しない。
6. 各IDの登録/directory消失と保持資産を確認してから次へ進む。新重要資産・dirty/ref drift・lock・job/契約疑義・normal remove拒否はそのIDだけ保留する。`--force`、prune、clean、reset、gc、手動強制再帰削除、無断unlockで回避しない。共有Git/保全資料の共通障害や他jobへの影響・広範囲の同一性不整合があれば全削除STOP。
7. 個別preflight、正常removeの一回性、結果・保留理由、保持確認、元path/branch/完全HEAD/common・manifestと復旧情報を非公開で保存する。公開失敗時に成功済みremoveを再実行しない。並行登録追加/branch前進はIDごとの除去と分ける。

操作前preflightは全139件で将来未実施。現在のHEAD/status/process確認を、未来の合格判定や削除許可へ格上げしない。

## 復旧と保持条件

非公開表へ元正規path・branch/完全HEAD・common/admin backlink、版別archive/manifest、実行参考文字列と再checkout参考文字列を保存した。保持branchが記録HEADで、元pathが空き、branchが他worktreeで使用されていないことを確認して通常のGitで再checkoutする案。競合をforce/resetで上書きせず、復旧自体も具体的操作指示の範囲で行う。今回は復旧0。

tracked checkoutの再作成とignored cache/guide/runtimeの完全復元は別。保全資料はcheckoutなしで参照できるが、科学実行環境の再構築やSOURCE欠落の補完をこの計画に含めない。固定SOURCE/controlを変更せず各独立cohort/失敗/STOPと元の実行制限を保持する。

## 承認対象の具体形

最大候補は上表「131対象ID」の和集合131件のみ。操作は対応する**LOCALの登録済みlinked checkout directoryとworktree登録だけの通常Gitによる除去**。研究branch/remote branch・main・共通Git/objects・原本/archive・科学SOURCE/raw/control・A/B/B3環境には及ばない。14単位をまとめて判断し、必要な単位を一つの指示にまとめられる。

現在すべてOWNER_DECISION_PENDINGかつ削除許可なし。ユーザーの実際の判断を待つ。所有者が判断していない事項の自動承認、receipt/grant等の新設はしない。

**mandatory STOP。139件全worktreeを維持。今回は非破壊保全・3環境統合・大量削除計画・匿名化公開まで。**
