# LOCAL Group 1・234件：研究単位の判断資料

PUBLIC_REDACTED_FOLLOWUP。元分類KEEP_EVIDENCE 166／HOLD_UNKNOWN 68は変更しない。**削除・登録解除・移動の許可ではない。**

| 新判断状態 | 件数 |
| --- | --- |
| TECHNICALLY_CLEAR_PENDING_OWNER | 0 |
| TECHNICALLY_CLEAR_PENDING_OTHER_ENV | 214 |
| PRESERVATION_ACTION_REQUIRED | 0 |
| KEEP_REQUIRED | 7 |
| UNRESOLVED | 13 |
| 合計 | 234 |

## 研究単位ごとの結論

| 研究系列 | 計 | 他環境確認待ち | 維持必須 | 未解決 |
| --- | --- | --- | --- | --- |
| 原稿・図表・索引 | 7 | 7 | 0 | 0 |
| QCアルゴリズム履歴 | 18 | 18 | 0 | 0 |
| N1/N3履歴 | 16 | 16 | 0 | 0 |
| Research B B2D Outer | 36 | 34 | 0 | 2 |
| Track Bアルゴリズム履歴 | 131 | 116 | 7 | 8 |
| 旧Track B G6 | 6 | 6 | 0 | 0 |
| Research C C1 | 4 | 4 | 0 | 0 |
| Research C C4回路 | 7 | 5 | 0 | 2 |
| DF-native Track A | 2 | 1 | 0 | 1 |
| 表現・アルゴリズム履歴 | 3 | 3 | 0 | 0 |
| その他アルゴリズム・中間検証 | 4 | 4 | 0 | 0 |

「他環境確認待ち」214件は、今回の既存索引scopeでローカル技術確認を終了したもの。所有者による利用終了・再使用・履歴保持役割の判断も残る。他環境の実パス参照を確認できていないため、ユーザー判断**だけ**を残す対象は0件。7維持・13未解決にも他環境条件が残り、全234件についてOTHER_ENV_SCOPEは未解決である。214は整理許可数ではない。

## 最初の検討群：文書・図表7件

| ID | branch | 保持先とのGit関係 | 判断 |
| --- | --- | --- | --- |
| LOCAL-WT-0203 | codex/pr-all-methods-hchain-figures-20261022 | ANCESTOR | TECHNICALLY_CLEAR_PENDING_OTHER_ENV |
| LOCAL-WT-0204 | codex/pr-higher-pr-comparison-figures-20261022 | ANCESTOR | TECHNICALLY_CLEAR_PENDING_OTHER_ENV |
| LOCAL-WT-0205 | codex/pr-resource-slide-outline-20261020 | ANCESTOR | TECHNICALLY_CLEAR_PENDING_OTHER_ENV |
| LOCAL-WT-0206 | codex/pr-resource-slides-clean-q-model-20261021 | ANCESTOR | TECHNICALLY_CLEAR_PENDING_OTHER_ENV |
| LOCAL-WT-0207 | codex/pr-slides-pf-cost-change-20261021 | ANCESTOR | TECHNICALLY_CLEAR_PENDING_OTHER_ENV |
| LOCAL-WT-0208 | codex/pr-resource-slides-current-20261022 | NOT_ANCESTOR | TECHNICALLY_CLEAR_PENDING_OTHER_ENV |
| LOCAL-WT-0209 | codex/pr-resource-slides-supplement8-20261022 | NOT_ANCESTOR | TECHNICALLY_CLEAR_PENDING_OTHER_ENV |

GitHub公開HEAD/公開祖先、clean、対象内の新しい未追跡科学資産なし、専用venvなしを確認。7件中6件には既存control索引の実パス一致がなく、0208の2参照は過去の文書公開manifestの作業pointerだった。科学SOURCEや現在の環境providerの束縛と区別した。図表・原稿は保持ref上のversioned成果として残し、専用checkoutの将来の再使用・履歴役割を所有者がまとめて判断できる。

**ユーザーの判断事項**：0203〜0209の専用checkoutを今後使う予定と、過去の作業場所を履歴として残す必要性。公開成果を保持refから再取得する方針で足りるか。これは利用終了方針の判断であり、実際の登録解除・branch削除・ファイル削除は別の操作として指示する。

## 第2群：QC18件・N1/N3の16件

全34件をTECHNICALLY_CLEAR_PENDING_OTHER_ENVとした。QCでは17件が既存strategy handoffの祖先、1件が同一HEAD。N1/N3は14件がN3 closureの祖先、1件が同一HEAD、1件はその祖先ではない。祖先でないものを破棄・同じ科学結果と認定しない。いずれも自身の保持refを残し、別SOURCE・別cohort・失敗/STOPを独立に保持する。

**保護する成果**：各固定HEAD/SOURCE/control、QC strategy handoff、N3 closure、旧N1回復・N3診断/失敗の記録。**ユーザーの判断事項**：過去のpreparation/診断checkoutの再使用予定と、代表文書＋保持refで追跡する方針の可否。他環境の物理参照は担当回答後に閉じる。

## 第3群：Research B 36件

34件はローカル技術確認終了、LOCAL-WT-0283/0286はUNRESOLVED。36HEADすべては既存の最終成果/pause handoffのGit祖先として確認したが、raw/独立実行結果まで包含されたとは判断しない。0283はExact-R readiness、0286はFull DF-PF readinessの物理SOURCE供給役割が記録されており、その終了は確認できていない。

**保護する成果**：対象外の最終handoff、科学SOURCE、各実行STOP・個別cohort・結果ref。**ユーザーの判断事項**：pause後の中間survey/preparation/結果参照checkoutを使う予定と保持役割。0283/0286は単なる利用終了の一括判断に含めず、記録されたSOURCE供給責任を別に確認する。Research Bのpauseを、サーバーBで実行中のDF-native B3停止の根拠にしない。

## 第4群：Track B 131件

116件はローカル技術確認終了、7件はKEEP_REQUIRED、8件はUNRESOLVED。大部分の実パス一致は過去のGit worktree一覧や保全snapshotだったため、それだけを物理的なSOURCE依存と扱わない。一方、固定SOURCE/provider、凍結実行契約、source/runtime付きone-shot記録の参照は解除しなかった。

**保護する成果**：BF/R0/R1/SP0.5/SP1、RA、G系列、M1-S等の各固定ref・独立入力/失敗/STOP・source/review/direct-childの来歴。**ユーザーの判断事項**：116件を用途別（旧準備・調査、実装/検証、結果/handoff参照）にまとめ、再使用・歴史保持の終了を判断する。7維持と8未解決を同じ一括終了へ含めない。

## 残りの系列と現在の研究活動

旧Track B G6の6件、Research C C1の4件、表現系列3件、その他4件はローカル技術確認終了。Research C C4の7件は5件が終了、0319/0328は出力namespaceの責任が未解決。Track Aは0021が技術確認終了、0020は記録されたraw namespaceの保管責任が未解決。現在の別Codex作業や再開予定を、process未検出だけで終了扱いにしていない。

旧Track B/G6/W2CのB3と、現在サーバーBで実行中と申告されたDF-native L_D=0 B3は別文脈。後日のB担当照合まで実参照を推測しない。Research Cの4＋7件は、現在の研究/別Codex予定の確認を特に所有者へ残す。

## 維持7件・未解決13件の対象別の理由

| ID | 研究系列 | 新判断 | 具体的な障害 |
| --- | --- | --- | --- |
| LOCAL-WT-0020 | DF-native Track A | UNRESOLVED | 公開済み実行記録のraw_namespaceが候補内を指す。現時点でGit外資産は未検出だが、原本/namespace保管責任の終了は未確認 |
| LOCAL-WT-0093 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | 既存source_preparation記録が、このcheckoutと同じ固定SOURCE commitをproviderとして直接指定。物理SOURCE参照の責任を未解除 |
| LOCAL-WT-0102 | Track Bアルゴリズム履歴 | UNRESOLVED | 保存observer source内にこのROOT定数が残る。凍結監視コードの再使用・歴史保持・path責任の扱いが未確定 |
| LOCAL-WT-0162 | Track Bアルゴリズム履歴 | UNRESOLVED | source review/authorization準備記録が物理worktreeを保持。後続記録との役割移行が明確でない |
| LOCAL-WT-0168 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | 凍結実行契約・guardがcompile入力/出力rootを直接指定し、fixture変更・reconstructionを認めない。既存path条件を維持 |
| LOCAL-WT-0175 | Track Bアルゴリズム履歴 | UNRESOLVED | exclusive B worktreeという既存保護記録がある。専用開発環境としての役割を終える判断は未確認 |
| LOCAL-WT-0176 | Track Bアルゴリズム履歴 | UNRESOLVED | 複数の既存READMEがB worktreeとして指定。共有venv実参照は未検出だが代表開発環境の役割終了は未確認 |
| LOCAL-WT-0185 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | R1の固定SOURCE/runtimeと実行worktreeをinvocation/終了記録が直接束縛。結果後STOPと固定pathの保持責任を維持 |
| LOCAL-WT-0187 | Track Bアルゴリズム履歴 | UNRESOLVED | 旧SOURCE/契約の修正待ち記録の保管責任と後続reviewへの移行が未確定 |
| LOCAL-WT-0188 | Track Bアルゴリズム履歴 | UNRESOLVED | 独立SOURCE review準備worktreeという記録が残る。旧レビュー用環境の役割終了が未確認 |
| LOCAL-WT-0193 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | SOURCE_FROZENのmanifestがこのworktreeを指定。直接child・固定SOURCEの契約を今回解除しない |
| LOCAL-WT-0196 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | SP0.5のone-shot実行記録に固定SOURCE/runtime・実行worktreeが残る。停止・grant消費と固定参照を維持 |
| LOCAL-WT-0198 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | SP1のSOURCE/review/direct-childと実行worktreeを既存記録が束縛。固定参照・STOPの保管責任を維持 |
| LOCAL-WT-0199 | Track Bアルゴリズム履歴 | UNRESOLVED | 旧SP1契約proposalがworktreeを記録。提案の再使用・自己拘束の扱いは所有者判断が必要 |
| LOCAL-WT-0200 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | FROZEN_SOURCE_PREPARATIONのmanifestがこのworktreeを指定。旧STOP・source identity条件を維持 |
| LOCAL-WT-0201 | Track Bアルゴリズム履歴 | UNRESOLVED | design review pendingの契約がworktreeを記録。設計提案/履歴を保持する役割が未確認 |
| LOCAL-WT-0283 | Research B B2D Outer | UNRESOLVED | Exact-R readinessが候補の物理worktreeを参照。最終handoff後も固定SOURCE供給元として必要か未確認 |
| LOCAL-WT-0286 | Research B B2D Outer | UNRESOLVED | Full DF-PF readinessが科学SOURCEと候補worktreeを記録。最終handoff後のSOURCE供給責任を未確認 |
| LOCAL-WT-0319 | Research C C4回路 | UNRESOLVED | 既存実行scopeの出力namespaceが候補内を指定。現状Git外成果は未検出だが出力/STOP保持責任を未確認 |
| LOCAL-WT-0328 | Research C C4回路 | UNRESOLVED | Rustiq実行scopeの出力namespaceが候補内を指定。役割終了・将来再使用と他環境の責任を未確認 |

KEEP_REQUIREDは「永久にcheckoutを削れない」という科学的判断ではなく、現在残る固定参照・契約を本調査が解除しないという扱い。UNRESOLVEDも、停止済み・Git公開済みだけで物理path/保管責任の終了を認定しなかったもの。source/review/output役割を原本で具体化し、必要な場合にだけ別の保持方針・契約判断へ進む。

## 次の担当と、操作を分離した順序

1. **ユーザー/各研究の所有者**：まず図表7件、次にQC/N1/N3、Research Bの34件、Track Bの116件など、研究用途単位で利用終了・再使用・履歴役割を判断する。対象ID集合はCSVで確認できるため234件を一件ずつ再調査しない。
2. **サーバーA/B担当**：候補の物理directory/固定SOURCEへの実参照を自身の既存記録で確認する。Aの12件を再調査せず、A追跡結果KEEP_REQUIRED3/UNRESOLVED9を維持。Bは実行環境に干渉せず後日照合する。
3. **ローカルCodex（別途指示後）**：利用終了・他環境・SOURCE/path・custody責任が閉じたIDだけ、完全pathは非公開原本で解決し、branch保持・共通Git保持・原本保全とcheckout登録解除を別の操作として具体化する。公開IDだけで実行先を選ばない。
4. **ユーザー**：実際の対象別の登録解除/削除・データ移動/保管先変更・branch整理を別途明示指示する。今回の技術判断や利用終了の回答だけで、未指定の操作は実行しない。
5. **該当環境のCodex（具体的操作の許可後）**：BEFORE_ACTION_DELTAをその直前に確認してから対象操作へ進む。このgateは今回全234件でNOT_DONE_FUTURE_ONLYのまま。

**mandatory STOP。今回、worktree/branch削除、登録解除、移動、科学計算、SOURCE/raw/grant/ledger/marker/STOPの変更を行わない。**
