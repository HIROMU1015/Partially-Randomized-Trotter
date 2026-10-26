# サーバーAから見たローカル75件の依存確認（2026-10-26）

固定入力は `950a866f4756e29c06a691c6b80e7b0e431777eb` の [75件一覧](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/950a866f4756e29c06a691c6b80e7b0e431777eb/inventory/local/batch_cleanup_75_20261026/BATCH_75_INVENTORY.csv)。同commitの保全manifest・計画・報告、234件followup、統合資料、サーバーA原棚卸し・12件followupも参照した。これは匿名化した公開用コピー。内部の詳細原本・照合索引は非公開で別途保全している。

75/75件の資料照合を完了した。内訳は図表7、QC18、N1/N3 16、Research B34。LOCAL-WT-0283・0286は対象外。環境ID同士を番号で結び付けず、環境・branch・完全HEAD・記録済みSOURCE/result commitを区別した。

| 主判定（相互排他的） | 件数 |
| --- | ---: |
| CONFIRMED_PHYSICAL_DEPENDENCY | 0 |
| NO_DEPENDENCY_IN_VERIFIED_SCOPE | 0 |
| PATH_VERIFICATION_NOT_POSSIBLE | 75 |
| ACTIVE_OR_RESTART_DEPENDENCY_UNRESOLVED | 0 |
| PRESERVATION_DEPENDENCY_REQUIRED | 0 |

75件とも正式なローカル実パス・別名の対応表がなく、物理照合は未完了。物理依存の検出0は依存不存在の証明ではない。主判定の実行未解決0は、別fieldに残した実行・再開未検証75を解消した意味ではない。追加保全・固定契約対応が具体的に必要と確定した対象は0件だが、全75件で可搬性・private custody等の確認は残る。既存のSOURCE・結果・失敗・STOP・Git履歴の保持条件は全75件で継続する。

## 確認範囲 S1 と論理参照

既存の非公開メタデータ索引2,016代表を再利用し、SOURCE/control/manifest/handoff/STOP等と既存文書・運用スクリプトを合わせた2,135テキスト資料を静的照合した。大きい索引2件と圧縮payload1件は対象外。最終テキスト照合の読取エラーは0。全履歴検索・rawの数値解析・実行試験は行っていない。既存Git object内の公開SOURCE関連JSON34件も読み、記録commitを照合した。公開メタデータ中のcommitは、記録fieldに従いSOURCE、結果、その他参照を分け、文書を含むcheckoutのHEADを科学SOURCEと同一視していない。

74件でサーバーAに取得済みのHEAD一致Git参照がある。これはキャッシュされたGit参照で、現在のGitHub tipや実行依存の証明ではない。LOCAL-WT-0207は同じ照合で一致せず、固定入力の公開履歴内保持（ancestor）の記録を引き継いだ。新たな全履歴探索はしていない。

| LOCAL ID | サーバーAの既存資料で検出した関係 | 対応資料数 |
| --- | --- | ---: |
| LOCAL-WT-0203 | 完全HEADで固定された公開図表資料への参照 | 1 |
| LOCAL-WT-0204 | 完全HEADで固定された公開図表資料への参照 | 1 |
| LOCAL-WT-0205 | 公開メタデータに記録されたSOURCE/result commitとの共通参照 | 13 |
| LOCAL-WT-0206 | 公開メタデータに記録された文書・図表入力commitとの共通参照 | 13 |

上表の資料は重複し、合計17資料。4件の論理参照は元worktreeの物理依存を意味しない。0206に関する過去の固定commit比較は、現在のprovider SOURCEを指定した記録とは別である。残る71件ではS1の具体的SOURCE/control等に照合キーの一致を検出しなかった。71件を物理依存なしへ格上げしていない。相対パスだけの一致はcommit未確定として扱う設計で、今回その追加一致はなかった。

共有Git・object alternates・venv/editable installの既存参照も確認した。サーバーA内の保護対象であり、LOCAL75との物理的な接続関係は実パス不明のため確定できない。同一所有者のprocess引数スナップショットに対象の論理キー一致はなかったが、別の引数表現・他process・再開計画を含めた不使用確認ではない。現在のjobや再開に不要とは判定しない。

## 未確認条件と返却質問

CSVの `U1;U2;U3`、`Q1;Q2;Q3` は全75件について、それぞれ同じ行のID・branch・完全HEADに適用する。所有者の「専用worktreeを再利用しない」方針は最新依頼から全75件に適用済み。古い固定報告の所有者確認待ちは履歴であり、再承認は求めない。

- U1 / Q1：元worktreeの正規実パスと別名・mount/symlink関係を照合できるか。合意済みの非公開手段で対応表を共有するか、ローカル側で照合し、ID別の非機密な一致結果を返す。branch名やSHAからパスを推測しない。
- U2 / Q2：各IDについて、現在job・再開予定、固定絶対パス、SOURCE/provider/input契約、editable install、venv/cache、共有object storeが元worktreeに依存するか。公開commit参照だけ、過去の記録だけ、実行時に必要、を区別する。
- U3 / Q3：元worktreeにしかない科学ファイル・未管理ファイル・private原本があるか。別保全先の同一性、SOURCE/STOP保持、相対参照・再開時の可搬性をID別に確認する。68件の並行保全の完了・内容・新保存先は今回検査しておらず、完了を仮定しない。

LOCAL75の対応表を送るための既承認の非公開共有方法は確認できなかった。既存の別目的のprivate転送方針は今回の許可へ流用しない。公開GitHubへの実パス記載や他環境へのSSHは行わず、上記の照合条件を残した。

`B1`：サーバーB担当による後日のID別照合が全75件で必要。現在のDF-native L_D=0 B3のSOURCE/provider/input・job・再開予定と元worktreeのパス/別名を確認する。旧Track B/G6/W2C B3とは別に扱う。今回はサーバーBへの接続・操作を行っていない。

## 既存保全条件と作業境界

今回の一致資料が所在するサーバーA worktreeは、従来12件followupの対象と重ならなかった。従来のKEEP_REQUIRED 3・UNRESOLVED 9を変更せず、再調査もしない。今回、新たなLOCAL元worktreeの物理依存を示す根拠は得ていない。

サーバーAのmain・共有Git、SOURCE provider、H2/H4/H6等のraw/control/STOP、private custody、共有venv/cache、未公開commit、永続handoffは継続保護する。新たな数値的・科学的結論を出していない。`research_project_instructions_unified_2026-10-25.md` はサーバーAで取得・読了できた資料として扱っていない。

公開コピーから内部絶対パス、ホスト/IP、private原本の保存場所・非公開ファイル名、process詳細、環境構成の具体値を除外した。branch・完全HEAD・研究系列・分類・件数・確認限界を残し、内部証拠の場所と照合詳細は非公開原本に保持する。

研究worktreeの切替・削除・移動、branch削除、共有Git/venv/cacheの変更、SOURCE/raw/grant/ledger/marker/STOPの変更、科学計算は行わない。公開は専用branchへの本ディレクトリ2ファイル追加だけとし、既存公開成果物を保持する。今回の照合はworktree削除の承認を与えない。公開後はmandatory STOP。
