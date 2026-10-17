# 調査範囲・保全・公開の記録

2026-10-17 JST。文献取得・文書作成の技術記録。科学的レビュー結果ではない。

## 1. 固定対象と権限

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。originは同ownerのSSH remote。
- 基点：`81d6103ec89d485cb7336c3a90bd95dad66992aa`。GitHub connectorでもcommitを取得して実在を確認。
- 独立branch：`m1s-prior-art-survey-20261017`、worktree：`/tmp/prt-m1s-prior-art-20261017`。
- 適用指示：ユーザーの今回の文献調査指示、AGENTS.md、PROJECT_MAP.md、2026-10-16版共通運用指示。
- commit/pushは今回明示依頼された公開作業。GPTレビュー開始・科学計算・新規証明への承認ではない。
- 主作業checkoutの未commit編集を持ち込まず、shot・cost feasibility branchへmergeしない。

## 2. 検索と一次資料確認

英語の公開web検索、arXiv版履歴・PDF、出版社、著者論文の参考文献と、基点のP0／G0-pre／
Checkpoint 2後／interstep correlation監査を使った。二次サイトは発見の手掛かりに限り、
技術比較は原論文または既存監査が指定する一次箇所に戻した。

実際に使った主な検索語群：

| 軸 | 検索例 | 主な採用資料 |
|---|---|---|
| 共有branch・同符号／逆符号 | `"shared randomness" "Hamiltonian" simulation`; `"qDRIFT" "forward" "backward"`; `"qDRIFT" "shared" "sign"` | MLMC、COS、SCU、SYK、既存protocol監査 |
| 相関したword／path | `"correlated" "qDRIFT"`; `"random product formulas" "palindromic"`; `"Hamiltonian simulation" "same" "branches" "random"` | COS、PRHS、qSHIFT、Kim–García-Pintos |
| coherent平均・位相 | `"randomized" "Hadamard test" "Hamiltonian" coherent`; `"coherent" "randomized Hamiltonian" "control"`; `"randomized" "phase estimation" "visibility" "bias"` | MF、KGW、PR、Gu、YC |
| noise／controlの意味論 | `"randomized compiling" Wallman Emerson 2016`; `"coherent control of quantum channels" interference Abbott` | RC、Abbott |
| 低深さ・shot・費用 | 論文名検索：WBC、qSWIFT、qFLO、Importance sampling、Resource-Optimal IS | WBC、qSWIFT、qFLO、KISS、IS |
| 統計・decoder・平均化 | KLY論文名／Erratum DOI、Hoeffding論文名、Davies 1974論文名 | KLY v3、Hoeffding、Davies（abstractのみ） |

調査は2026-10-17時点で取得できた指定版に固定。完全なsystematic review、全citation graphの閉包、
特許調査、priorityの法的審査ではない。検索で一致を発見しなかったことを不存在の証明に使わない。

今回PDF取得28論文、追加の別Erratum／Davies書誌2件、計30文献レコード。
出版社のYC本文HTML・SIも別経路で確認した。PDFの全pagesを読了した件数ではない。
関連箇所・未確認proof・版差・失敗経路は[bibliography.md](bibliography.md)と
[literature_retrieval.json](literature_retrieval.json)を正本とする。

## 3. 実施した作業と実施していない作業

実施：read-onlyのGit資料閲覧、文献検索、公開PDF/HTML取得、pdftotext／紙面render、
書誌・式・定理statementの比較、保存された科学値の文章引用、ファイルhash・link・Git tree照合、
この調査文書の作成と公開。

新科学計算、保存結果の科学的再集計／境界再評価／shot forecast／fit／ランキング、
Hamiltonian・Pauli・state・行列のdecode／再生成、simulation、sampling、回路build、
native compile、source import・科学runner・既存numerical testの起動は**0**。
新規理論・還元定理・証明は**0**。GPTの新規性判断・論文化評価は**未開始**。

文献PDFの取得・ハッシュ・ページ表示と、研究の数値解析は別作業である。
使用したPythonはstdlibによる文書取得・metadata・file identity用に限る。
GPU・BLAS・Qiskit／Numba等を起動しない。取得物の一時保存先は`/tmp`。
科学計算性能policyは、新たな科学runを許可する根拠にはしていない。

## 4. 保全検査の境界

1. 共有checkoutのHEAD／index bytesと、追跡ファイル＋提供レビュー原本等のhashを事前記録。
2. 全worktreeの既存HEADを事前記録。他worktreeのruntimeやjobを操作しない。
3. 本branchの基点差分を調査directory・新研究ノート・発見用index追記だけに限定。
4. 継承したSOURCE／科学結果／契約／authority／grant／ledger／marker／validation manifest／科学statusのGit tree identityを比較。
5. 共有checkoutの事前記録と終了時を比較。並行作業由来の変化があれば、こちらの改変や全環境静止と混同せず報告。

新規検証pathを追加していないため、runner/library/test indexの変更や既存validation manifestの更新は行わない。
科学的なdecision・validation statusも変更していない。
本調査の証拠inventoryは独立した[repository_evidence.json](repository_evidence.json)と
[literature_retrieval.json](literature_retrieval.json)に記録する。

保全の最終結果は[preservation_audit.json](preservation_audit.json)、文書技術検査は[document_checks.json](document_checks.json)に記録する。
これらのPASSは文献中の定理・M1-Sの数値coverage・科学的新規性の認定ではない。

事前／終了時比較では、共有checkoutのHEAD・indexは一致し、確認した1,332 pathのうち
1,331は同一、`AGENTS.md`に並行更新を観測した。改訂された性能方針・完了後の担当明示規則を読み、
今回の科学計算禁止とmandatory STOPを維持した。既存他213 worktreeのうち、
shot-cost feasibility preparationのHEADが`81d6103e…`から`fb5519bf…`へ進んだ。
旧commitはancestorとして保全されている。本調査はそのファイル・branchを変更せず、更新を戻さない。
「全workspaceが静止していた」という保全主張はしない。
本調査branchで継承した保護対象7,099 Git object記録の対象pathは基点から無差分である。

## 5. GitHub公開同一性

調査payloadをcommit/pushし、remote branch SHAと固定commitのGitHub file bytesを確認する。
公開後receiptを別子commitへ追加する場合、receipt自身のhashを自己参照させず、
payload SHA／filesのbytes・SHA256・blob、継承証拠のidentityを記録し、
receipt子commitのremote HEADとreceipt bytesを最後に別途確認する。

`publication_receipt.json`が公開時の実測記録。存在する前に確認成功を宣言しない。
repositoryの公開設定はGitHub connectorで`public`を確認した。
基点の必須34ファイルはGitHubから実際に取得してblob・byte数一致を確認し、
うちT1a/T1b・主要レビュー・raw3系列等の9ファイルはbase64 decode後の全bytes一致も確認した。
記録は[base_github_identity.json](base_github_identity.json)。
branch公開はmainへのmerge、PR作成、他branchの更新を含まない。

**公開確認後mandatory STOP。次のGPTレビュー・科学計算・native費用作業は開始しない。**
