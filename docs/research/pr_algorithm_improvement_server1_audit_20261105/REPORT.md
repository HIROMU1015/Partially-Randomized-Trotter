# SERVER 1：PRアルゴリズム改善研究の追加証拠調査

観測日：2026-11-05 JST。対象：`HIROMU1015/Partially-Randomized-Trotter`。

**既存の改善研究の固定資料・SOURCE・実行結果・旧失敗と後続更新を追加整理した。新しい科学結果は作成していない。** SERVER 1で確認したGit保存コピーと、実行環境・非公開原本の現存確認を区別した。

## 引渡す成果物と単位

| ファイル | 内容 |
|---|---|
| [SERVER1_IMPROVEMENT_EVIDENCE.csv](SERVER1_IMPROVEMENT_EVIDENCE.csv) | 72行。47既存研究カード、15個のPhase 2段階、7後続・重点項目、3保存文書の対応。実験数・独立cohort数ではない |
| [SERVER1_EVIDENCE_GAPS.md](SERVER1_EVIDENCE_GAPS.md) | SERVER 1未確認、現存する非公開資料、調査範囲外を分離 |
| [SERVER1_REUSE_MAP.md](SERVER1_REUSE_MAP.md) | 再利用対象、関数、条件、依存、SOURCE版差と検証境界 |
| [REFERENCE_INDEX.csv](REFERENCE_INDEX.csv) | 固定commit/path/blobで参照を識別。本文読解とGit tree metadataだけの確認を分離 |
| [CODE_ASSETS.csv](CODE_ASSETS.csv) | 93件のSOURCE・runner・testの静的AST目録。過去版10件を含む。test実行記録ではない |
| [INPUT_COHORT_BINDINGS.csv](INPUT_COHORT_BINDINGS.csv) | B、F1旧/修正版、F2旧/修正版、PS48の公開済みSOURCE/input/run識別情報を許可列だけ転記 |
| [FIXED_REF_RETENTION.csv](FIXED_REF_RETENTION.csv) | 固定object、観測した公開ref、既存bundle headerの対応。全復元の認証ではない |
| [GITHUB_REFERENCE_CHECKS.json](GITHUB_REFERENCE_CHECKS.json) | GitHub connectorで実取得した代表SOURCE・結果の固定blob照合 |
| [AUDIT_METADATA.json](AUDIT_METADATA.json) | 読取範囲、metadata検査、保全確認、公開対象のhash |

47カードのうちOPT-01は、前回にも登録されたRPE接続baselineの索引補助である。今回の72行を前回100,878行へそのまま加算しない。カードの元status・outcome・条件・STOPは継承し、今回の整理用列と区別した。`NOT_BOUND_IN_THIS_AUDIT`はSOURCE不存在・未実施の判定ではない。

## 調査基点・適用指示・前回との境界

原証拠checkoutのHEADは`0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`。既存dirty/untrackedを維持した。公開資料の比較基点は、取得済みobjectと`git ls-remote`で一致を確認したmainの`546bca7ed2cece59b714b986a264d7b0b0871a01`である。これは閲覧・収録のcarrierであり、過去の科学SOURCEへ付け替えない。

PROJECT_MAPを最初に読み、研究概要、原checkoutと関連既知worktreeのAGENTS、出力基点の固定AGENTSを確認した。正式共通運用指示は主rootの`research_project_instructions_unified_2026-10-25.md`を実取得し、特に§8.1、§8.2、§8.4、§10を適用した。個別研究のone-shot・旧STOP・原本保護は維持した。primary agentだけで実施し、別AIによる独立レビューとはしていない。

前回の[SERVER 1 Phase 1 REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2de84ed7af00e73339886429aabab0bf8180e9f0/docs/research/pr_cross_host_coverage_audit_20261102/inventory/SERVER_1/REPORT.md)は、100,878観測行と改善方式自体の除外を記録する。[統合REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/604f0235840feb18d368f99ea0828f370ce70fc8/docs/research/pr_cross_host_coverage_audit_20261102/integration/REPORT.md)もその除外とSUPPORT_ONLYの境界を保持している。今回、両REPORTの本文・範囲・索引・保全・STOPを確認し、巨大な列挙表や分割CSVの再作成は行っていない。Phase 3の科学評価を引き継ぐ作業でもない。

[統一索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/RESEARCH_RESULTS_INDEX.md)、[改善研究70カード](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9bdcf1d44061daceffe896637e2c92fcb3353b4/docs/research/pr_algorithm_improvement_landscape_20261023/STUDY_CATALOG.json)、対応SOURCE索引、Phase 2A/2C/2D、algorithm_codesignの原報告、[11月1日の科学地図](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/pr_repository_scientific_landscape_20261101.md)と[claim–evidence matrix](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/pr_repository_claim_evidence_gap_matrix_20261101.md)を探索入口として再利用した。

## SERVER 1で実際に確認した範囲

本人UIDと対象originを確認できた既知3 Git domain、および同common Gitの登録worktree・既存監査/保全rootに限定した。公開root IDは`S1I-GIT-01`（主Git）、`S1I-GIT-02`（前回監査clone）、`S1I-GIT-03`（既存収録clone）。WP11 worktreeは主Gitと同じdomainである。旧aliasの一つは主rootへ解決し、別研究環境として数えなかった。旧SERVER_A/B等のhost全体の対応は確定していない。

再利用した非公開metadataは、既存repository inventory、SOURCE/custody catalog、archive所在表、10月23日のファイル名索引、11月3日の収録判断・backup検査、11月4日のdelta照合、11月5日の既存worktree保全記録。今回はこれらから候補を限定し、関連rootの直下目次・明示されたpathのmetadataだけを確認した。古いファイル名索引は当時の観測であり、現在の全域不存在を証明しない。

固定Git object内の文書・SOURCEは再利用可能だった。47カードの要約・条件・既存判定を転記し、75のカード関連Markdownを機械的に取得して固定参照を解決した。重点系列の実行報告・要約・provenanceは追加で読み、93件のコードをimportせずAST解析した。全参照の科学的内容を再監査したという意味ではない。大きなraw・QPY・配列・圧縮case表はmetadata参照にとどめ、展開・decode・再集計しなかった。

## 確認した追加証拠と後続更新

| 系列 | 保存記録から確認した状態と読み分け |
|---|---|
| Phase 2A / 2C / 2C-R1 | 課題定義、fragment別S/P辞書設計、既知法との対応を保持。`NO_METHOD_DELTA_FOUND_WITHIN_SCOPE`は限定的な方法差調査であり、性能実験の負結果ではない |
| Phase 2DのP0/I/B-PRE/B-ENV/B | 静的契約、実装のみ、環境identity、実際の入力bindingを別段階化。Iのtest定義は実行証拠ではなく、Bの完了もoperator/信号/compileのPASSではない |
| F1旧停止→修正版 | 旧metadata setterの技術エラーと部分結果を保持。修正版は登録correctnessを完了したが、interval未資格の`UNPRICED_FALLBACK`を認証済み最適選択へ格上げしない |
| F2資格gate→限定資源計測 | 統計資格付き経路のblockedを、別契約の1-shot資源計測で解消したことにはしない |
| F2旧停止→位相保持修正版 | 旧compiled operatorの3件FAILと、後続の全登録数値検査・資源集計完了を分離。旧失敗の書換え、同精度総資源優位、認証Nの取得とは扱わない |
| F2保存寄与分析→PS48→Gaussian終了 | 保存caseの記述的分解、同じbaseline/referenceを使う限定48-wrapper、一律適用の費用負結果・局所改善例、選択規則未確立、方向の一時終了を別記録として保持 |
| Track B BF/BM/SP/R/RA/G1–G10 | 係数familyの負結果、既知compact同値、synthetic native tradeoff、固定辞書で強いCTSに負ける結果、return集約・finite-bit・on-demand構成を分離。異なるdegreeを同exponential精度の順位へ並べない |
| Parent allocation / M1-S / C1 | 訂正KKT前後、同じ保存信号の後続解析、H4 logical pilotの回復runと凍結を保持。C1で分子接続の記録はあるが、strict native・十分N・全wrapperの完成へ拡張しない |
| 非prefix / sampling | RQ-02は具体的method deltaと非prefixの増分検証が未確立。後続sampling調査の`NO_DISTINCT_FREEDOM`と人工テストは、H4科学samplerの検証や一般no-goではない |

混在辞書の数値検証に言及する範囲は、**4 mode・2粒子・occupied(0,1)、決定論one-body一つ、`L_D=1`、残差density-square二fragmentの合成fixture**。分子geometry・AO basis・分子DF rankは非該当。F1は`T=1/5,q=2,delta=1/10`一点、`r=2,h=1/20,K=2`、F2/PS48は`T=1/5,q=1,delta=1/5`一点、`r=2,h=1/10,K=0`であり、lawや価格を同一条件として転用しない。Qiskit 1.3.0、opt1、seed17、`rz/sx/x/cx`、free topology、approximation_degree=1、qubits_initially_zero=False、phase fittingなし、修正前後の接続版を区別する。[F1修正版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b0dad8cc2f7a4cb821f53185c4a65f3a42e8f381/docs/research/pr_algorithm_improvement_phase2d_f1_rerun_20261024/EXECUTION_REPORT.md)、[F2修正版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc7378ab34015287215bcf4ce52147e46d94ecd9/docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md)。

## SERVER 1固有資料・コピー・保全

`S1I-DOC-01..03`として、G6/H4後のレビュー原稿とTrack B前後期の案内資料二つの現存原本を確認した。いずれも原checkoutでは未コミットだが、現在の固定mainの保存原版とGit blobが一致した。以前の収録台帳の`PREPARED_NOT_COMMITTED`を現在の未公開状態として引用しない。これは保存文書の対応で、独立の実験・レビュー再採択ではない。各公開pathはCSVにある。

改善研究の非公開raw/QPY/grant/marker/詳細logについて、原報告の「保全済み」と、SERVER 1の現存原本まで確認できた状態を分離した。**今回、Track B・混在辞書の新しい非公開科学run原本をSERVER 1で追加同定したとはしていない。** 見つからないことを未実施へ置換しない。公開snapshotだけの行にはその限界を記入した。

WP11の既存compute log二つとbackup検査metadataは現存・本人所有を確認した（`S1I-WP11-01`）。これはRPE/cost baselineの補助資料で、前回観測と別の改善実験ではない。checkpoint/rawの再読・再hash・復元はしていない。

`S1I-BUNDLE-01`は既存保全bundle。980 refのheaderを読み、Phase 2関連の固定先端が含まれることを確認した。bundle全体の展開・payload検査・全SOURCE closure復元は未実施。branch/checkout不在を証拠不在とせず、固定objectと保存tag/headerへ接続した。これを全原本の独立媒体backupと呼ばない。

観測前後で既存3 Git domainのworktree登録、HEAD、index hash、dirty/untracked一覧とlocal branch/tagは一致した。一方、監査対象外の進行中系列に対応するremote-tracking ref一つの更新を観測した。更新主体は未同定で、今回の読取処理からfetch/ref更新は行っていない。巻戻しや他作業の停止はせず、全refが不変だったとは報告しない。保存raw全体のbytes不変をhashで証明したという主張もしていない。

## 検査・公開・mandatory STOP

CSVのID、JSONセル、参照先、固定blob、SOURCE版差、入力bindingをmetadataとして検査する。GitHub connectorで代表4ファイルを固定commitから実取得し、既存objectのblobと一致した。これは保存資料の取得照合であり、人工テストPASS・数値correctness・資源優位・外部再現の再認証ではない。mainのVALIDATION_STATUS/manifestも確認したが、旧snapshotの登録状態を全改善系列の一括判定として使わなかった。

公開は既存研究から分離した新branch `codex/server1-improvement-evidence-audit-20261105`、新namespace `docs/research/pr_algorithm_improvement_server1_audit_20261105/`だけを対象とする。既存main・研究branch・監査台帳・normative文書・validation manifestは更新しない。出力Gitは独立index/branchを使い、既存objectのalternates参照は独立backupと扱わない。最終commit自身のSHAと公開後の全ファイル取得照合は最終チャットで示す。

科学計算、保存値の数値再解析・再ランキング、sampling、simulation、RNG replay、研究runner/test再実行、回路build/compileは0件。原SOURCE/input/raw/result/grant/ledger/marker/STOP、他job・共有venv・scheduler・OS設定を変更していない。原Gitのcheckout/reset/clean/gc/prune、branch整理、LOCAL/SERVER 2へのSSHも行っていない。

**公開・取得照合後にSERVER 1追加調査mandatory STOP。横断統合、科学レビュー、新規実験、旧run再開へ進まない。** LOCALには固定commit/path/blob、公開input hash、元run/cohort識別情報と未結合項目を渡す。独立性の横断確定と統合台帳への採否はLOCAL側の作業として残す。
