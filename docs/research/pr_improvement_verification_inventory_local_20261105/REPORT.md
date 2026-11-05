# PRアルゴリズム改善研究：検証棚卸し・重複排除（LOCAL）

2026-11-05 JST。**LOCALの公開証拠に基づく暫定整理。新科学計算・数値再解析・詳細科学レビュー・研究方向の正式採択は実施していない。**
既存coverageの統合結果・評価、研究正本、SOURCE/raw/result/grant/ledger/marker/STOPは変更しない。

## 成果物

| ファイル | 内容 |
|---|---|
| [IMPROVEMENT_VERIFICATION_LEDGER.csv](IMPROVEMENT_VERIFICATION_LEDGER.csv) | 旧70研究への継承索引＋42段階・関連記録、計112行・40列。目的・算法・差・条件・検証層・元status・SOURCE／結果／文書commit・関係・STOP・未知・再利用 |
| [IMPROVEMENT_GAP_MATRIX.md](IMPROVEMENT_GAP_MATRIX.md) | RQ-01〜07と高次PF/FR/access/研究C/RPE等の追加系列。検証済み・部分・未検証・資料不足を区別 |
| [DUPLICATION_AND_REUSE_MAP.md](DUPLICATION_AND_REUSE_MAP.md) | copy、再引用、条件変種、修正版別実行、独立確認、後続更新、不明の対応と再利用候補 |
| [REMAINING_VALIDATION_MAP.md](REMAINING_VALIDATION_MAP.md) | 将来の主張ごとに既知と不足を対応。実行せず、優先順位・採択を決めない |
| [REPORT.md](REPORT.md) | 本書。調査範囲、固定資料、限界、検査と終了条件 |

112行は実験数でも独立確認数でもない。旧70行は研究familyの索引で、同じ原結果を段階別・関連証拠別に参照する行もある。
科学的な実験数・勝率・新しい性能順位は算出していない。

## 基点・作業場所・許可範囲

観測した公開mainは [`546bca7ed2cece59b714b986a264d7b0b0871a01`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/546bca7ed2cece59b714b986a264d7b0b0871a01)。GitHubのmain refと既存LOCAL objectの対応を確認した。
独立作業branchは `local/pr-improvement-verification-inventory-20261105`。
元のdirty checkoutを切り替えず、共有Git objectを参照する分離cloneで新規専用directoryだけを追加した。
この作業cloneを独立backupとは呼ばない。main・既存branchの上書き、force push、branch/worktree削除は行わない。

最初に実在する `PROJECT_MAP.md`、`AGENTS.md`、正式共通指示 `research_project_instructions_unified_2026-10-25.md` を確認した。
研究概要を読んだ上で、main内の概要・status/manifestをそのcheckoutの保存時点として扱い、統一索引と後続の固定記録を併用した。
IDEのactive fileは調査対象・許可の根拠に使用していない。sub-agentは使用していない。

許可されたのは文書/SOURCE静的読解、既存結果引用、provenance/metadata照合、台帳作成、専用branchへの文書公開である。
科学moduleのimport、科学runner/test、科学値の平均・比・SE・fit・N/G再計算・再ランキング、simulation、sampling、Hamiltonian/state/reference生成、build/compileは行っていない。

## 既存119,304行台帳との差

固定[coverage PROTOCOL](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/968218c2b0c6280af6c1c02822603e3312aa410c/docs/research/pr_cross_host_coverage_audit_20261102/PROTOCOL.md) §2は、アルゴリズム改善・Return Aggregation・新event/sampling生成・理論的新規性を対象外とする。
改善系列に含まれるbaseline等はSUPPORT_ONLYに限るという境界も維持する。

既存[Phase 2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/604f0235840feb18d368f99ea0828f370ce70fc8/docs/research/pr_cross_host_coverage_audit_20261102/integration/REPORT.md)と[Phase 3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/REPORT.md)の公開報告・索引・metadata処理を参照した。
119,304という値は既存報告の引用。入力inventoryやCANONICAL_CATALOG/RECORD_LINKS/UNRESOLVED/RELATIONSHIPSを再生成・再統合していない。
既存coverageのclaim、充足評価、不足分類、原IDを更新しない。

今回追加したのは改善方式の目的・構成・方法差・検証段階・失敗/修正/終了・再利用の地図である。
Track A、Pauli-JW、研究C、RPEの既存実験は関連証拠として参照し、この台帳を理由に新しい科学実験へ加算しない。
「研究B/B2-D」「Track B」「Track BのH4 C1」「研究CのC1」は同名部分があっても別系列として読む。

## 取得・確認した資料

| 資料群 | 実際に確認した範囲と役割 |
|---|---|
| 運用・入口 | PROJECT_MAP、AGENTS、10/25正式共通指示、研究概要全文、VALIDATION_STATUSの関連記述、manifestの構造・入口。旧記述を全branch最新仕様としない |
| 10/23 landscape/Phase2A | STUDY_CATALOGとSOURCE_AND_PROVENANCE_INDEX、STUDY_SCOPE/RESEARCH_PROBLEMS、主な問題・失敗・reuse資料。70カードの目的/条件/結果/STOP等を継承。全70のSOURCE/rawを今回独立に再読した意味ではない |
| Phase2C/2C-R1 | 主設計、既知法対応、独立差未確立/限定追加案の同値判断。新規性の再審査なし |
| Phase2D一式 | P0/I/B-PRE/B-ENV/B、F1旧停止・修正版、F2-PRE/資格STOP・旧資源失敗・修正版、要約JSONの元status/SOURCE。科学SOURCEを起動していない |
| Gaussian後続 | 保存F2寄与分析・静的費用原因、PS48結果/固定条件、選択規則調査、一時終了handoff。寄与表を今回再集計していない |
| Track B | G6〜G10の結果・旧失敗/修正版、v1.2/資源利用範囲への入口、C1-R1/R2/5A/P5R/凍結、BS/SP/R/RA-D0/D1〜D3/Parent/M1-S等の旧カードと主な総括 |
| 11/1〜2の横断資料 | pr_repository_scientific_landscape、claim_evidence_gap_matrix、RESEARCH_RESULTS_INDEX、coverage Phase1〜3/後続bindingへの入口。元coverageの再評価なし |
| 11/3〜4 main収録 | main_collection、SOURCE別版保存案内、priority7、unlisted34の本文/対応表を入口に固定snapshotを照合。収録と新実験/実行採用を区別 |
| 研究C後続 | 11/3 Evidence Map、005/064のSOURCE/結果、11/4固定word pilot。候補間・校正cohort・有限correctness・全law未資格を分離 |
| RPE後続 | V1/V2/V3/R1、11/5固定資源統合の条件付きenergy接続。固定splitの後続成果をround別表現変更の実証にしない |

外部の高次PF/DF repositoryに属する旧索引は、元の固定repository/commit/path/URLを継承した二次参照である。
今回そのrepositoryの全原本を取得し直したとは記載しない。

## 更新関係の要点

| 記録 | 科学SOURCE | 結果固定commit | 今回反映した状態 |
|---|---|---|---|
| 旧F1 | `31e8391a56fbb1a0e64f477f78e242655329de99` | `594bab9e1441699474f262b075390db3ab8723c6` | metadata技術停止・部分結果。actual transpiler未到達 |
| 修正版F1 | `8192530b17d6bc4af195c734ff43a6e78c506942` | `b0dad8cc2f7a4cb821f53185c4a65f3a42e8f381` | 登録correctness全件PASS。価格fallbackは最適選択でない |
| 旧F2 | `9b1e103102f918bb157dea90c7eaf3acb5d88b6c` | `06ce7e5b1002de3cfaec6c17ea901f589c05f8f6` | raw全件PASS/compiled3件FAIL。期待資源未完了 |
| 修正版F2 | `9468c7089727f5648d9f59cddbaebe0a42f5f0a0` | `dc7378ab34015287215bcf4ce52147e46d94ecd9` | 位相保持修正後全件PASS。実1-shot期待費用取得。認証Nは未資格 |
| PS48 | `770e53addda63a03fa8c6c5d5168f7062067c572` | `1e20458d742a8fac62ce0d0264188b1a7772ee28` | 同operator/lawの48wrapper同値性PASS。一律費用削減不支持 |
| singleton一時終了 | 科学実行なし | `69f8e1cac23a4924ab81aebe399657a705024f6a` | 正式一時終了を記録。旧成功/局所改善は保持 |
| G9 v2 | `0ef2b92738750a3c0d187743dd4b7c9b927db802` | `78cb65be373b0808641b5a5e44d9b69913bf84e9` | 旧API停止後の別run完了。T/CX交換で全座標支配ではない |
| G10 v3 | `b9ed01455351628c9073748f5ba5751aa794b789` | `4301c0f87a16be45b620304c3fa935d5aa2d721a`（保存carrier） | v1/v2技術停止と別に完了。G9 m5/key再利用を分離 |
| C1 P5R | `2fbd83a44a0deb76c6444ed7920741c6b5e2e748` | `bb9059bacb697e1105e63c1b3826b0c34886ad15` | P5-only完走。旧Ordinary/Full完走を再利用し3方式の分子logical観測完成 |
| C1凍結 | 科学実行なし | `185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e` | 追加研究一時凍結。G6数学的方法論は維持 |

結果commitは固定された保存版を示し、常に最初の生成commitとは限らない。
台帳の `source_commit_role`、原SOURCE manifest、相対pathを併用し、carrierを科学SOURCEへ付け替えない。
F1/F2/PS48の条件と精度・compiler、G9/G10のfinite target、C1のtail-only条件は台帳とGapに保持した。
表のPASSはlocal登録範囲の保存判定であり、immutable CIや今回の独立再現ではない。

## SERVER 1/2：取得済みと未取得

取得したのは以下の固定公開資料である。

- [SERVER 1 11/5資源証拠差分](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0a2b54650527094306f619e3d08dbb2bbc94ba14/docs/research/pr_server1_evidence_delta_20261105/REPORT.md)とDELTA.tsv。
- [SERVER 2 11/5資源証拠差分](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5b0971a2d9fc6ef474008d26d14af85e324b4b24/docs/research/pr_server2_evidence_delta_20261105/REPORT.md)とDELTA.tsv。
- [後続3環境資源統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd/docs/research/pr_research_paper_evidence_synthesis_20261103/RESEARCH_SYNTHESIS.md)とSTATUS_AND_OPEN_QUESTIONS。

これらは主に資源評価の追補で、今回の**改善専用追加調査報告は未取得**。
GitHubの公開branch名の確認、`improvement`/`algorithm`のbranch検索、成果物名のcode検索では、その専用報告を特定できなかった。
未公開・不存在・他環境未実施とは断定しない。照会時に取得できた範囲の暫定整理である。
直接SSHや他環境filesystemの探索は行っていない。

SERVER 1の境界比較、SERVER 2のH8長時間探索/確認やC1既存参照は関連証拠として照合した。
H8 Stage Cの新PR確認と保存PF価格の再利用、旧H8の同cohort alias、B3主要表とraw全量の差を区別する。
後続資源統合で既に反映された結果を再度「新しい改善実験」として登録しない。

## 台帳のSOURCEとUNKNOWNの読み方

旧70索引では、原カタログのprimary/saved-result/implementation参照を保存した。
単独の実行SOURCEが今回の読取だけで確定しない場合、`science_source_commit=UNKNOWN` とし、候補参照を `source_commit_role` と `source_paths` に残した。
これはSOURCE欠落の認定ではない。後続stage行に直接の実行SHAがある場合はそちらを読む。

重要なUNKNOWNは、改善専用SERVER報告、旧UWC原runとのidentity、修正済み旧DF入力、全履歴cohort/private rawの独立性、各候補の全law/母平均/最終費用保証である。
別repositoryの旧PF/access系列は今回継承範囲に限る。11/5の全稼働研究を完全censusしたとは主張しない。
資料不足・未実施・実行未完了・技術失敗・限定負結果・既知同値・正式保留を区別した。

## 検査・効率・原本保全

検査対象は文書とmetadataに限定した。既存 `metadata_integration.py` の固定Git lookup・CSV・関係ラベル・公開情報検査、旧Phase2Aの文書検査方式を静的に確認して再利用した。
既存43列の全観測統合処理を再実行せず、新40列の研究/段階索引へ一時的な標準library adapterを使用した。
科学SOURCEのimportやテスト起動は行っていない。

確認項目はUTF-8/RFC4180の往復、40列・ID一意性・関係先、固定commit/pathの実在、5要約JSONのstatus/SOURCE literal一致、選択報告のmain/固定元blob比較、相対リンク、公開情報、差分範囲である。
外部repositoryの継承URLは新規取得検査に含めず、UNKNOWN/継承区分を維持した。
保存科学値を算術で再検算するvalidatorや科学testは実行していない。

Git object再利用、batch metadata lookup、必要なCSV/JSONだけの読取を用い、大容量raw/観測台帳を複製・再展開しなかった。
CPU/RAM/worker/wallの人工上限、新規依存導入、共有設定・他ユーザー/job操作はない。速度向上倍率は測定していない。
元checkout・旧SOURCE・結果・STOPへ書き込む操作は行わず、出力branchの変更を新規5ファイルに限定する。
公開資料に内部絶対path/host/IP/資格情報、private raw、承認原文、grant/markerの内容を転載しない。

検査の具体的な件数・結果は下記「最終文書検査」に記載する。
公開後のremote SHAと5成果物の取得照合は最終チャットでも報告する。本書自身を含む最終commitのSHAを自己参照で埋め込むためだけの追加commitは作らない。

## 最終文書検査

| 検査 | 結果・範囲 |
|---|---|
| CSV | UTF-8、RFC4180往復bytes一致、112行/40列、ID一意、関係先実在：PASS |
| 行の種類 | 継承索引70、段階記録32、関連参照10。独立実験数へ換算しない |
| 実行SOURCE欄 | 明示40桁の23異なるcommitはLOCAL Gitでcommit objectとして実在。実行時内容の再検証ではない |
| 当該repositoryの固定参照 | 台帳URLの234異なるcommit/pathをLOCAL Git objectで解決。全234の新規HTTP取得確認ではない |
| 外部repository | 継承した28固定URLは今回新規取得未検査として分離。当該repoでobject未発見だから外部資料不存在とは判断しない |
| status/SOURCE文字列 | 旧F1/修正版F1/旧F2/修正版F2/PS48の5要約JSONとliteral一致：PASS |
| 選択報告の保存copy | main以外の固定文書35件のうち33件はmain同pathと同blob。064価格報告と11/5最終資源統合の2件は基点main同path未収録で、固定元を使用 |
| 相対リンク・公開情報 | 5成果物の相対リンク切れ0。内部絶対path/host/IP/credential等の検出pattern一致0。公開本文も目視確認 |
| 差分・保全 | 新規専用領域の5ファイルだけを公開対象にする。既存SOURCE/raw/正本/coverage等の変更なし。公開前にGit差分/whitespace検査 |
| 科学検証 | 新しい科学値・科学PASS・再現性認定を生成していない。科学test/runner/compilerの実行なし |

これは作者自身の文書・metadata検査で、独立科学レビューやCI科学再現ではない。
公開照合ではこの5成果物を固定remote SHAから取得し、ローカルの内容とGit blobを比較する。

## 終了と次の担当

**専用branchへの公開、remote SHA・成果物取得照合後mandatory STOP。**
Codexは新アルゴリズム設計・実装・科学計算・保存値再解析・詳細科学レビュー・旧STOP解除へ進まない。
既存研究正本・coverageの科学的判断を変更していない。

次の担当はユーザー。必要なら改善専用SERVER報告の照合対象や、今後判断する科学的な問いを別途指定する。
これは今回の完了を保留する承認依頼ではない。今回の棚卸しに続く自動実行作業はない。
