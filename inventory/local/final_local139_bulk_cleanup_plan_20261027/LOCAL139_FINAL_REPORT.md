# LOCAL139：3環境統合・大容量8件の保全・最大131件の削除計画

2026-10-27 JST。PUBLIC_REDACTED。固定入力観測：`2026-10-27T03:30:41.539940+09:00`。
Repository：HIROMU1015/Partially-Randomized-Trotter。整理専用branch：`codex/pr-environment-inventory-20261026`。

**139件のID・branch・完全HEADを3環境で全件対応付けた。既知scopeで元checkout必須物理依存を確認したIDは0。既存131件は全てBULK_APPROVAL_CANDIDATE_PENDING_OWNER。大容量8件は9固有file／384,562,915 bytesを非破壊保全し、PHYSICAL_ARCHIVE_VERIFIEDへ進めたが、131件へ自動追加せず別枠にした。所有者判断は14研究単位・139 IDすべて未回答。worktree削除・登録解除は0。**

## 実際に確認した固定資料と指示

開始時の現在remote HEADは指定`8da04de4aae1ccf9c0e0327411ce9081cf78f491`と一致した。既存独立公開cloneだけへ取得し、以下を読んだ。GitHub連携で取得したA/B資料とcloneの内容も一致した。

- LOCAL139準備：`96cf3dae3bd03d6f64f384fefedc6e88e782eef6`、[状態CSV](../next_group1_139_preparation_20261027/NEXT139_PRESERVATION_STATUS.csv)、同directoryのarchive manifest・計画。
- A139：`8da04de4aae1ccf9c0e0327411ce9081cf78f491`、[A報告](../../server_a/local139_actual_path_reconciliation_20261027/SERVER_A_LOCAL139_PATH_REPORT.md)と139行CSV。
- B_PATH_139：`ee30e8b435be6b15bba201de923b61eb02d9c54e`、[B報告](../../server_b/local139_actual_path_reconciliation_20261027/SERVER_B_LOCAL139_PATH_REPORT.md)と139行CSV。
- 既存234 followup、75件最終計画、Batch1/Batch2〜4削除結果、非公開139正規path表・完全保全binding/manifest・SOURCE/result/STOP索引。

正式運用指示`research_project_instructions_unified_2026-10-25.md`をローカル実在ファイルで読み、§8.1・§8.4・§10を適用した。ユーザーの今回指示は資料確認・非破壊保全・計画・匿名化公開までの実行許可で、削除/研究契約変更/科学レビュー/科学計算には拡張していない。承認証明書・receipt・新grant等を作っていない。

## 3環境の元判断と統合

| scope | 完了ID | 元主判断 | 必須物理依存の確認 |
| --- | ---: | --- | ---: |
| LOCAL既存索引＋今回限定差分 | 139 | 選定scope保全131・大容量保全待ち8から今回進捗 | 0 |
| A139 | 139 | NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE 130、HISTORICAL_REFERENCE_ONLY 1、PRESERVATION_OR_CONTRACT_GATE 8 | 0 |
| B_PATH_139 | 139 | NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE 139 | 0 |

A/B両方で139元checkoutの必須物理束縛を検出していない。ただし、A/Bの異なる主判断・副条件を統合stateで置換しなかった。CSVのA_main_decision_original/B_main_decision_original等へ元値を残した。Aの8 gateとBの大容量副条件は、LOCALの15 binding・9版未コピー条件を引き継いだもの。今回LOCALで解消したことを別fieldに示し、Aが保全copyを再検証したかのように書き換えていない。

**LOCAL-WT-0184**はAの歴史参照1件。A確認の公開source manifestのTrack_A_reference_only/commitが過去の参考commit`4c23453c541700c6a41ba71fc5ec9323b53858d6`を記しており、checkout HEAD`09fe527274e8de5c0d808ada5890fa06a59f78ea`とは別。物理directory指定や固定SOURCE供給契約として扱わず、Gitと元記録を保持する。BにはこのA判断を代入していない。

A/Bの実パス未受領・照合未実施という前回条件は139件で解消済み。A139は既存候補/限定metadata・既知A alias/process/Git pointer、B_PATH_139は保存path occurrence/SOURCE-runtime宣言・既知B3 chainと限定metadataのscopeである。一般的な未知alias・mount/共有実体・別namespace・未記録job/再開・private custody等は各行の副条件・残余リスクへ保持する。最小表にないLOCAL common Git/外向きguide先の直接照合まで完了とは言わない。両hostへの接続や同じ全件調査は今回0。

## 大容量8件：必要性・実コピー・custody

対象は**LOCAL-WT-0002, LOCAL-WT-0095, LOCAL-WT-0096, LOCAL-WT-0097, LOCAL-WT-0099, LOCAL-WT-0100, LOCAL-WT-0101, LOCAL-WT-0103**。前回保留の15 bindingを正確に抽出し、9固有tracked blob、384,562,915 bytesへ対応付けた。9版すべてで、固定commit/pathのGit blob・サイズ、保持common Git objectの存在、各同名GitHub refの完全HEAD一致、現在materialized sourceの本人所有/regular/可読性を確認した。正しい版は保持Git objectから取り出せるが、既存保全manifestには同じ版の独立実ファイルがなかった。

従来の保全scopeが選定checkpoint/resultを元checkoutなしに閲覧することを目標としていたため、その未コピー箇所を閉じる**1固有版1コピー**を必要と判断した。Gitでtrackedだから不要、または全研究rawを重複コピーすべき、とはしていない。既存の科学契約が新しいbackup先を要求しているという認定でもない。具体的な元path必須契約・現在jobによる当該checkout必要性は、既存LOCAL/A/B scopeと現在可視process/metadata差分で未検出。未記録再開・研究担当の将来利用は未確定のまま。

| 内容 | 結果 |
| --- | ---: |
| 対象worktree | 8、全て維持 |
| 固有tracked大容量版 | 9 |
| 版別binding | 15、個別対応を維持 |
| 新規実コピー | 9 files |
| 新規保存bytes | 384,562,915 |
| 元source前後SHA-256・保存bytes・Git blob一致 | 全9版・全15対応 |
| 元checkoutへのhardlink/symlink | 0 |
| 上書き・copy error | 0 |
| 新しい科学計算・checkpoint再生成 | 0 |

本人所有・owner-only ACL/permissions、空き容量・新規namespaceの非衝突を確認し、既存と同じ方式の非公開資料領域に新規copyをexclusive createした。source安定性、dest可読regular file、source inode非共有を確認。旧archive/manifestへ書き込まず新規manifestへ15 bindingを記録し、保存fileをowner read-onlyにした。

8 IDはすべて**PHYSICAL_ARCHIVE_VERIFIED**。GIT_VERSIONED_CUSTODY_ACCEPTABLEだけで済ませたID、PRESERVATION_ACTION_REQUIRED/FIXED_CONTRACT_GATE/UNRESOLVEDとして残ったIDは今回0。これは元checkoutの利用終了や科学的再実行可、固定path契約の解除ではない。将来の追加候補としてowner判断・preflight・別の具体的削除範囲が必要で、今回の最大131件には含めない。

Git ref保持、実bytes保全、科学的再現性、独立backupは別条件。新規copyは同じローカルdiskにあり独立障害域backupではない。科学runtimeの再起動・checkpoint再開・研究結果の再計算による検証はしていない。9版以外のraw転送やshared storage設定変更は0。

## 既存131件と所有者判断

| 判断 | 固定131件内の件数 |
| --- | ---: |
| BULK_APPROVAL_CANDIDATE_PENDING_OWNER | 131 |
| BULK_APPROVAL_CANDIDATE_OWNER_CONFIRMED | 0 |
| TARGETED_CHECK_REQUIRED | 0 |
| PRESERVATION_ACTION_REQUIRED | 0 |
| KEEP_REQUIRED | 0 |
| UNRESOLVED | 0 |

131件は選定scope保全とLOCAL/A/B限定結果が整合し、現在の具体的阻害要因がない。前回75件の所有者利用終了方針は一切転用せず、全139 IDのowner_workspace_decisionをOWNER_DECISION_PENDINGとした。B照合完了でもowner判断は完了しない。各研究単位の再開・他Codex利用の可能性も未確定。特にResearch C9件を現行研究や他Codex予定から自動的に終了認定しない。

Track B固定131件内109と優先72の構成、他の旧G6 6・Research C9・その他7の正確なID集合は[最大131件計画](LOCAL131_BULK_CLEANUP_PLAN.md)にある。14研究単位でまとめて回答できるOWNER_DECISION_PENDING / WORKSPACE_END_APPROVED / WORKSPACE_KEEPの判断表と、保全・固定SOURCE/STOP・A/B主判断・現在利用・将来予定・具体的保持必要性・ユーザー判断項目を作成した。今回は回答を代行していない。

## 限定ローカル確認と残条件

139件の正規path/owner・common Git・登録、HEAD/branch・tracked clean/untrackedなし・既知ignored leaf不変・lockなしを確認した。可視本人UID processのcwd/引数に対象path一致0。既存control206・既知editable pointer12の差分と、前回以降にHEAD変化したcheckoutの最近control/doc62件を文字列照合し、新path参照0・読取エラー0。旧64件削除で利用不能になった旧metadataを未読のまま読了扱いにはせず、既存scopeと今回可読差分を区別した。

従来archive6,403固有版のmetadata不変を確認し、過去の新規5,889 files／503,215,888 bytes・514版再利用、26,390 bindingの安定SHA/Git blob/source inode検証を再利用した。code981/sparse953 bindingは保持Git参照のみというscopeを維持し、コピー済みとは表現していない。今回全archiveの無条件再hash/再コピーは0。

現在具体的な保持義務を検出していないことと、未知の固定契約・alias/mount・他namespace/job/手動再開・外部raw custodyの不存在は別。これらをscope外残余リスクとして保持し、具体的兆候のあるIDだけ限定確認を追加する。同じ全件監査を永久に繰り返す条件にはしない。固定SOURCE/pathやshared jobへの具体的な危険が判明した場合、一般的な削除承認だけで契約を解除しない。

Bの最新固定観測は03:05 JSTに既存DF-native L_D=0 B3がactive/running・SCIENTIFIC_ACQUISITION_RUNNINGだったという報告。ローカルから新たな状態照会や停止/再開/科学job操作はしていない。旧Track B/G6/W2C B3やResearch B pauseと区別し、B3のSOURCE/provider/input/raw/cache/Git/runtime/grant/ledger/marker/STOPと実行制限を保持する。

## 次回の最大一括承認範囲

候補は固定131 IDのまま。14研究単位のうちworkspace終了が実際に判断された単位を選び、その正確なID和集合を一度の具体的実行指示で扱える。全該当単位が終了した場合の最大は**131件**。優先72件を先にすることはできるが、72件刻みの承認/実行を必須にしない。追加8件は別の明示対象にしなければ含めない。

後日の承認対象はLOCALの指定linked checkout directoryとworktree登録の通常Gitによる除去だけ。branch/remote ref、固定commit/SOURCE/result/STOP、原本/archive/manifest、main/common objects、他worktree・A/B/B3は維持。実操作直前にはHEAD/status/untracked/ignored配下重要資産、実path/common/backlink/lock、job/再開・SOURCE/path差分、保持refとarchive可読性/独立性を確認する。一般UNKNOWNの全面再調査や全files再copyを既定にしない。

通常git worktree removeを承認IDごとに適用し、異常IDだけ保留、共通Git/保全/jobの共通障害では全STOPする将来計画を作成した。force/prune/clean/reset/gc/手動強制削除/無断unlockで拒否を回避しない。操作後は登録/directory除去と保持資産、元path/branch/完全HEAD・manifest・再checkout手順を記録する。復旧は必要時の具体的指示内で通常Gitにより行い、ignored/runtime全体まで完全復元できるとはしない。

## 成果物・原本保護・未実施事項

- [LOCAL139_THREE_ENVIRONMENT_RECONCILIATION.csv](LOCAL139_THREE_ENVIRONMENT_RECONCILIATION.csv)：139件の元分類・LOCAL/A/B元判断/副条件・新判断・owner/残条件・131/追加8分離。
- [LOCAL139_LARGE_FILE_CUSTODY.csv](LOCAL139_LARGE_FILE_CUSTODY.csv)：9固有fileの8 ID・15 binding、Git保持、既存archive不在、必要性、copy bytes/hash検証とcustody残条件。
- [LOCAL131_BULK_CLEANUP_PLAN.md](LOCAL131_BULK_CLEANUP_PLAN.md)：正確な131 ID集合・14研究単位owner判断表・Track B109/優先72・最大一括承認・preflight/通常Git/STOP/復旧。
- 本報告：3環境統合・保全・限界・次回の具体的操作範囲。

新規の本人所有非公開領域に実path/管理backlink・保全先/15binding manifest・固定契約記録・131件将来操作文字列・全139件復旧情報・14単位判断表を保存する。旧原本・manifest・archiveは上書きしない。公開へ内部絶対path、username/host/IP、private raw/非公開archive位置、credential/秘密鍵、非公開対応表やprivate file内容を追加しない。公開するcontent SHAは今回保全した公開tracked bytesのhashだけで、内部pathの単純hashではない。

既存のCSV/Git/manifest/匿名化/コピーhelperを再利用し、新処理は3環境join・正確な9版の追加保全・131件と8件の分離・研究単位判断と計画の生成に限定した。今回科学的採否・詳細科学レビュー・計算/compile/sampling/再解析は0で、科学レビュー用詳細研究記録は作成しない。

139 checkout・Group1保護20・旧SOURCE3版55宣言の11件・Research B供給環境・その他LOCAL/主Git/環境・A/B worktreeを維持する。削除済み64件の復旧/再操作も0。公開は既存独立cloneで新規directoryの4資料だけを、最新remoteの並行更新を保持する通常fast-forwardで追加する。既存inventory/研究branch/mainは変更しない。commit/remote SHAとGitHub読み戻し結果は非公開PUBLICATION_RESULTと完了通知へ保存する。

**mandatory STOP。全139 worktreeを維持。owner判断待ち14単位/139 ID、次回最大131 IDの具体的削除指示と直前preflightが必要。追加8件を自動的に131件へ増員しない。**
