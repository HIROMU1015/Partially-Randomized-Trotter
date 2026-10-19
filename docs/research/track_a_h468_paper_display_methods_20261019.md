# H4/H6/H8掲載用図表・caption・Methods整備（2026-10-19）

利用者は統合commit `a77141ba5ff0752d6409e32fda9e4649d0919fdb`を受け入れ、既存証拠だけを使う掲載用整備を指示した。
配置は**主表＝固定独立確認6条件、Figure 2A＝legacy・探索系列、Figure 2B＝H8独立n48**として固定した。
これは表示・文書の固定であり、最終evidence freezeや詳細科学レビューの開始ではない。

## 掲載用成果物

[掲載packet](../../artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/README.md)から各資料へ移動できる。

|資料|整備内容|
|---|---|
|main_table.md / main_table.csv|独立確認6条件。CSVは前段保存表のbyte同一copy。exact費用・保存paired SE・候補・PF・L_D/δ・n・点方向を維持|
|figure2_caption.md|英語の掲載用caption。Figure2Aのlegacy/探索、2BのH8 n48を区別し、A82後続完成と原注記の履歴を説明|
|methods.md|英語Methods。入力生成、solver、正規化/残差gate、control/compiler、費用・cohort・希少event・公開監査範囲を明記|
|input_exceptions.csv / .md|入力・来歴・表示の8例外。旧protocolの歴史的PENDING/欠測から後続完了への対応も明示|
|h8_public_evidence_status.csv / .md|公開証拠14項目。SOURCE規則、PF公開receipt、PR完了申告、未検証事項を分離|
|evidence_linkage.csv|主表6条件→既存CEM→既存台帳→SOURCE/cohort→Figure2 aliasの対応|
|publication_manifest.json|原図・表・caption・Methods・CEM・台帳のpath/SHA対応。表示固定と科学freezeを区別|
|documentation_evidence_ledger.csv / public_read_receipt.json|実際にGitHubで取得した17公開資料の固定commit/blob/SHAと用途|

主表の条件はlinear H4/H6/H8・隣接1 Å・STO-3G・tol-only1e-8 DF10/19/28、T=.8・P=0、通常signed bias/allowance倍率1。
H4はL_D3/δ.8/n1108とL_D2/δ.1/n78、H6はL_D4/δ.4/n48とL_D4/δ.1/n104、H8はL_D8/δ.4/n48とL_D7/δ.05/n48。
各系の前者はε=.05、後者は.0005。主表での固定PF対照は直接実測で、費用は解析shot×compiled RZの保存値。
H4粗精度は **NOT_REPRODUCED・PF低費用**、細精度はREPRODUCED・PR低費用。
H6粗精度は **PF優位方向のREPRODUCED**、細精度はPR優位方向のREPRODUCED。
H8は両方POINT_REPRODUCED・PR低費用という保存点結果を維持する。母平均順位を認定した表ではない。

H4旧固定DF12を主表のDF10へ改称せず、探索n8・FP n32・旧確認union n32・新独立確認を混合しない。
H8主表と2Bは既存Mechanism32内H8_CONFIRM_1/3と同じ証拠であり、重複した独立証拠として数えない。

## Figure 2とA82

元commit eb33de67の34ファイルは全byteを保持し、PNG/SVG/PDF・入力表・plot値・原captionを変更していない。
今回更新した掲載用captionは別管理の文書であり、元画像内の歴史注記を削除するものではない。
旧A82欠測注記は描画時点の履歴。後続1ff6ac0bで固定32比較・64軸の説明因子が完成した事実を掲載captionに追記した。
後続保存因子表を別に参照し、原図へ新しい点を足していない。
A82は旧H8 n32 comparisonであり、H8 n48の説明欠測と誤記しない。旧因子を新H4/H6 Pへ移していない。

## 入力生成・solver・正規化の差

H4公開SOURCEと保存policy比較記録を照合した。記録上はSCF・canonical MO helper・spin/integral規約・DFのtol-only政策等に一致する部分があるが、元provider/runtime/全生成規則の完全同一性を今回認定していない。

- H4 DF10：sector36の完全行列にdense scipy.linalg.eigh。最低vectorの位相を固定。eighの正規化vectorを使い、保存後再正規化なし、残差gate1e-10。
- H6/H8：保存比較記録はeigsh SA・HF初期vector・k1・tol1e-12・ncv40・maxiter1000、sector400/4900。一度のbinary64正規化、absolute1e-9/relative1e-10の残差判定を記す。

後者の具体値は公開された比較記録からの引用であり、今回元provider/保存入力の私有witnessを再監査した値ではない。
公開generic solverに正規化stepがあることと、特定runの全引数・元bytesを再確認できたことは区別した。
保存状態を統一手順で生成し直さず、同じtol名やrankだけで同一性・系サイズの因果効果・scalingを主張しない。

## H8で確認できたこと・未検証のこと

GitHubの固定公開SOURCE、結果CSV、paired-axis projection、historical design、selected-PF receipt、既存公開auditを実際に読取確認した。
17資料のremote blob IDを固定Git objectと照合し、その同一objectのSHA256を記録した。科学moduleは実行していない。

**公開記録を確認できた事項**：主表の保存mean/SE/n/固定PF/点判定、公開axis CSVのschema、SOURCE上のsymmetric_directionalとscalar/reference/cohort guard。
固定PF対照B0I p11/q1 S2・p15/q4 S4には、両軸の保存実測単価とexit/reap/compiler/fingerprintを含む公開selected-axis receiptがある。
そのreceiptは選択された決定論jobの記録に限定される。

**公開証拠だけでは再検証できない事項**：後続PR確認の実seed除外・grant消費・全runtime/provider/入力bytes・全worker native exit/reap・最終byte照合、原profile bytes、旧compiler出力の完全同一性。
PR公開報告のローカル完了記載は保持するが、PFの終了証拠をPRの完了証明へ移送しない。
historical designのunapproved状態は設計時点の記録であり、後の実行grantの有無を証明しない。
確認した公開projectionの不足を「私有原本が存在しない」という断定にしない。非公開資料へアクセス・公開していない。

未観測のS2/S4 event費用・分散は未知のまま残す。観測0・有限SE・点順位は母平均保証やCIを与えない。
既存RZ metricをFT/non-Clifford/T費用へ読み替えない。

## 構造検証と保全

新checker `python3 scripts/check_h468_paper_display.py` は、配置、exact表copy、主表/CEM/台帳の6対応、caption/Methodsのscope、A82後続完成、H8 PF/PR監査区分、旧資料hashを検証する。
新構造テスト13件と既存統合構造テスト11件がPASS。保護対象197ファイルにFigure2原本34ファイル・既存統合bundle・既存SOURCE/結果/grant/ledger/STOP等を含む。
テストは使い捨てmetadata上の改変拒否で、科学計算・統計解析・raw再集計ではない。Codex自身の技術チェックであり、別担当の独立科学レビューではない。

今回のsampling・compile・新統計・費用比/CI・機構分解・科学callは0。H2混合、共有環境変更、他job操作、AI delegationも0。
公開範囲は前段で承認された公開情報由来の新統合資料と文書/構造コード。今回private資料や新科学結果を追加公開しない。

## 最終採用前の判断

配置は今回の利用者指示で固定済み。次は**ユーザー**が掲載caption・Methodsの文言、入力手順差とH8公開監査限界を明示した形での最終原稿採用を判断する。
母平均優位・完全入力同一性・完全なH8公開実行監査を必要とする主張を採る場合は、その不足を解消できる証拠の要否を別に判断する。追加の解析・sampling・compileを自動開始しない。
**Codexは文書commit・承認範囲の公開とremote SHA確認後、mandatory STOP。** 最終evidence freeze・詳細科学レビューは未開始。
