# O2全成果の文書統合：作業報告

2026-11-08 JST／SERVER1 GPUA100-gleap-01。base `89facf729535901a79c111721e5ab1714bd9a4be`、専用branch `codex/o2-results-integration-20261108`。

**O2の既存成果を改善機構ごとに統合した。新しい科学計算・保存科学値の再解析・アルゴリズム実装・旧test実行・重要科学レビューは0。** 本作業は成果の再認証や新しいGO/NO-GO・新規性・論文化可能性の判断ではない。

## 成果物

- [O2_RESEARCH_SUMMARY.md](O2_RESEARCH_SUMMARY.md)：目的、機構別の系譜、固定条件と主要数値、限定成果/負結果、共通知見、今後の新規O2探索領域。
- [O2_EVIDENCE_INDEX.md](O2_EVIDENCE_INDEX.md)：39の主要stage/資料入口。科学SOURCEと結果/報告収録commit、条件profile、証拠レベル、現在の状態、修復/中断/再利用を区別。
- 本REPORT：今回の文書作業と未解決事項・保全・公開・STOP。

新directoryは要求されたMarkdown3点のみ。SOURCE/raw/manifestを複製して新しい管理台帳を作らず、必要な索引・概要・当日ノートと機械可読manifestに文書統合の項目だけを追記する。既存科学的statusは変更しない。

## 確認範囲と再利用

正式共通指示2026-10-25（SHA256 `e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720`）、適用AGENTS、PROJECT_MAP、研究概要・現状、研究目的/方法と証拠statusの従来の読み順を維持した。

指定の固定7成果（3571477、2b71117、dc42c3a、b6305ee、62c450a、4e610af、89facf7）を中心に、3環境台帳のO2関連stageと再利用mapからR0–G5、G6–G10、D1–D3、Parent/S4、B3、C1、sampling自由度調査を追跡した。さらにN1/N2/旧engine、N3中断/新B停止、752-shot、3control理論、SCV224/Signed metadata/448、A2/J1と最新LCUの一次REPORTを確認した。全repository・全raw・全private環境の再監査はしていない。

既存総括の再利用により、初期R0/G1/G4-Aの理論成果、G5の限定class終了、G9の局所T成果、Parentのh=0/異proposalの限定利益、C1の正式凍結を保持できた。後のS2/SCV/A2の負材料へ単純に上書きしない。研究Cの別H4/CDF8やO1/O3・C-S等の数値をO2の改善成果へ加えていない。

752-shotは完走したがX/YともINCONCLUSIVEである。397bitの中断・1339bitの正規停止をその完走結果へ混入せず、SCVの共通paired評価価格も二重計上しない。旧G9/G10失敗版、S4訂正前proxy、S2 v1転送STOP、N2当初engine UNKNOWN、A2当初アクセス未完成と後続LCU成功をそれぞれ別履歴として残す。

## UNKNOWNと今回決めないこと

全PR＋RPE最終資源優位、実H4の期待費用改善、真のCV相関/母分散差、全support価格/normalizerの安価な取得、未測定online古典費、他入力/precision/provider/compilerへの一般化は未確立のまま保持する。

B3の原報告記載SOURCE `c7e8806cf308131b5629ac54b627d46c360799a3` は、local objectがなくGitHub commit APIも422で取得できなかった。固定結果commitの報告引用は可能だが、そのSOURCEを実行Git commitへ独立に結合できたとはしない。証拠索引にUNKNOWNを明記し、推測で別commitへ補完しない。これ自体を科学結果のFAILへ変換しない。

旧snapshot `4301c0f…` 等を報告収録先として使う箇所は元の最初の公開commitとは区別する。外部再現/CI・priority・研究重要度の定量ランキングは今回認定しない。将来領域の表は必要な新性質を整理したもので、新アルゴリズムの採択や実装開始ではない。

## 技術確認・保全・公開

技術確認はUTF-8/改行、Markdownの参照/相対link、固定Git commit/path、引用表示文字列の原REPORTへの存在、変更範囲、既存文書のappend-only、manifest旧項目不変、元mainのHEAD/status/既存2,136ファイルのbyte保全に限定する。既存Phase 2Aの `git/blob_oid` helperを再利用する。数値の再fit/集計/分散/十分N/費用計算や科学module/test/runnerは起動しない。

公開前の技術確認はPASS。39固定文書path、110 Markdown参照先、38の引用表示文字列の原REPORT内存在、manifest旧17項目の不変を確認した。B3のSOURCE commit未取得は上記の例外として保持する。これは科学値の検算・SOURCE closureの再認証ではない。

元main・dirty/untracked、旧SOURCE・raw・結果・grant/ledger/marker/STOP・C1凍結、並行worktree/process/cacheを変更しない。旧artifactを新資料へコピーせず、共有OS/systemd/cgroup/scheduler/venvや他ユーザーへの操作も行わない。通常の文書作業だけで、恣意的なCPU/RAM/worker制限を新設する必要はなかった。

専用branchでcommit/公開し、remote SHA、全変更本文・blob・treeと固定参照を照合する。公開後の固定commitは最終チャットと作業用照合記録を正本とし、自己SHA追記用の追加commitは作らない。

**公開照合完了後mandatory STOP。新科学計算・保存値再解析・新規実装・旧研究再開・重要科学レビューへ自動移行しない。**
