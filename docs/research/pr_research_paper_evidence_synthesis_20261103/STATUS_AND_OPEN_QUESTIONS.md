# Status・未保証事項・次の判断

2026-11-03 JST。LOCALが統合担当。対象はLOCALに限定せず、GitHub公開済み3環境の固定成果、Phase0–3、G01/G02、研究別結果を優先した。これは新しい独立科学レビューでも、主要claimの正式採択でもない。

## 1. 原statusを残した追補

[CLAIM_EVIDENCE_MAP.csv](CLAIM_EVIDENCE_MAP.csv)はPhase3の21claimを基本行として保持し、R02のV1/V2/V3/R1を4部品行、既存全体matrixのTrack B B04/B05/B06/B11を4行追加した29行。行数は実験数ではない。`phase3_original_*`は当時の文字列・coverage行の投影、`followup_evidence_separate_from_phase3`は後続、`current_formal_adoption`は新採択なしを表す。Phase3原CSV・A–E分類は更新していない。

Phase3のcoverage44行のうちHIST-UWC/HIST-DFSCREENは21claim外のhistorical lookup。対応CSVへ新しい肯定claimとして加えず、旧coverageへの固定参照と本資料の未保証欄で保つ。Track B行もPhase3へ遡って存在したように扱わない。[Phase3][P3REPORT]、[原claim][P3CLAIMS]、[原coverage][P3COV]

| 系列 | 旧status／記録 | 現在の追補と守る境界 |
|---|---|---|
| A01–A12、R01/R02、F01、P01–P05 | 大半はLIMITED_RESULT_OR_LOCAL_VALIDATION_REPORTED、USER_FORMAL_ADOPTION=UNKNOWN。A04はM1 local review記録あり | 限定保存結果を論文化候補へ整理しただけ。technical PASS・既存local reviewを全研究の正式受理や外部再現へ変えない |
| A13 H8 rare | 層別finite sample＋条件付き残余感度のみACCEPTED_DOCUMENTED、EXISTING_STOP | 正式採択済みの限定scopeを維持。未知残余を理由に未完了・sampling必須へ戻さない |
| A11/G01 | 追加7候補のSOURCE/input/価格対応がEVIDENCE_UNLOCATED | 7物理候補すべてMATCHED。旧642／S2 q32追加22に含まれない。既存33/40と拡張探索33/40は別集合。確認価格を探索へpoolしない |
| A08/A09/A11/A12/G02 | 候補・cohort・共有・回復の対応が限定的 | 100対応行は98候補／比較行とH8 alias2。88 DOCUMENTED_DISJOINT、10 RECORDED_SEPARATE、2 REUSED。同seed継続・aliasは独立実験でない。全seed履歴／raw証明ではない |
| R02/V1 | 旧Phase3全roundモデルの未保証 | 共通E0で両18round PF残差は予算内。状態と厳密半径は条件付き。Full半径7点不一致は保持 |
| R02/V2 | 全18round分枝・成功率等は当時未検証 | 保存平均・IID・alias条件下で両候補CONDITIONAL_SUCCESS。全RPEを未検証へ戻さない。実機・状態準備・厳密joint decoder保証へ拡張しない |
| R02/V3 | 直接価格・移送域が不明瞭 | 72軸を対応。最終Partial直接q1/2/8、Full q1/2/16/32。A0 r32を最終r1へ流用しない。長q費用は外挿 |
| R02/R1 | 元モデルのPartial低費用点値 | 元値を再現、約4.858%点選好は維持したが、共分散UNKNOWN・モデル差・長q感度により総合UNDETERMINED。R2/R3は順位決定目的で非推奨 |
| Pauli | 旧H2 Hybrid優位／旧H6候補集合等 | 最新保存表では純RTE14/15、H4粗精度はHybrid。H6 Hybrid追加後も方向維持。P0・仮想ρ0・確認後選抜と共有価格を保持 |
| Track B C1 | 古い資料のsynthetic中心／分子接続未完成 | 3方式各1024分子pathのlogical観測は完成。一時凍結。strict native全wrapper・共通認証N・分子全task優位は未取得 |

G01/G02の件数・V1–R1の科学結果は原資料の保存結論を引用した。G02の再利用2行の元ラベルは`REUSED_OR_OVERLAPPING`。今回新たな実験同一性判定、科学的再分類、成功率／費用／平均／SEの計算は行っていない。[G追補][GREPORT]、[G01表][G01]、[G02表][G02]、[V1][V1]、[V2][V2]、[V3][V3]、[R1][R1]

## 2. 証拠の読取範囲と取得限界

全体landscapeとclaim/gap matrix、研究カード／SOURCE索引、Phase3固定索引・選択group、G01/G02、研究別validation、旧原稿・補足・図表設計を再利用した。固定Git objectのREPORTと、主要JSON/CSVの必要field・行を照合した。大きな全rawのdecode/hashや、全119,304観測・全カード・全依存SOURCEの再監査ではない。取得一覧・blob・読取範囲は[VALIDATION.json](VALIDATION.json)。GitHub固定URLがある資料と、他hostのprivate原本を直接検査したことは区別する。

| 不足・所在候補 | 今回確認した範囲 | 本稿候補への影響・対応 |
|---|---|---|
| H8詳細audit/provenance | 公開主結果・主表は取得。詳細候補commit `659e555714c8ccf3a6ec7fd1cb0dbf05cfcc342d` と `96aa3878a344f479ee064118cb90002a770db627` はGitHub照会で404、LOCAL固定object未取得。既存2026-11-02 inventoryにSERVER2対応候補あり | 現存を直接確認していない。保存主表の限定点比較は書けるが、詳細audit／完全独立性まで確認済みとはしない。強いclaimを採る場合はそのSERVER2候補だけread-only照合し、必要な匿名metadataのみ公開する候補 |
| H6 final seal | `f79366f68caf8585cc36d2f2b0fefcb1a4bd0d1f` の公開READMEは取得。scientific execution未実施・承認待ちのreadiness資料 | n48/104等の提案契約を、実行済み最終seed監査の回収と扱わない。G02のnullをこの資料で置換しない [H6 readiness][H6SEAL] |
| DF B3公開／private最終版のcustody | 公開 `ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed`、保全記録のprivate最終版 `f794a2aa…` の差を元provenanceが記録 | 公開結果の取得域負結果を使う。private最終版・全raw同一性まで認定しない。必要なclaimだけに追加custody metadata回収を限定 [保全記録][SRC_B3PROV] |
| 高次PR・H6等の全履歴seed排除 | G02の列挙cohortと保存監査まで。一部旧履歴未回収 | 記録された別価格cohortの方向確認まで。全研究履歴とのdisjointや外部再現を主張しない |
| UWC原JSON／旧DF screening修正raw | 既存Phase3の未発見・失効を継承し、無関係な全域再探索はしない | prose UWCやinvalidated旧DF37表を現行の性能根拠にしない。現在の限定claimの必須資料ではない |

SERVER1/2 inventoryは所在候補の根拠であり、今回SERVERの現在状態を直接読んだ意味ではない。現claimに必要な公開一次結果を取得できたため、直接SSH、全サーバー再棚卸し、転送・private証拠の新規公開は実施していない。H8資料候補は未解決として残し、資料がないことを「実験未実施」へ変換しない。SERVER2主checkoutの配置意図・別媒体backup・branch継続用途確認はこの研究統合とは別課題。

## 3. 記述・資料回収・追加科学を分ける

以下はユーザー指定の6区分による作業整理であり、Phase3 A–Eの新評価ではない。

| 区分 | 現時点の具体例 | 担当・進める条件 |
|---|---|---|
| 1 既存証拠で記述できる | 固定task／費用定義、旧H4事例、B0I・強PFの保存正負、DF B3負結果、G01対応 | Codexの本統合で整理済み。実際の原稿化はユーザーが範囲を選んだ後の次の指示 |
| 2 限定を伴えば記述できる | 長時間33/40・別価格方向、Pauli14/15、H8層別finite sample、RPE条件付き18round成功と点費用モデル | 原集合・cohort・仮想ρ0・状態／IID／alias・外挿・未知共分散を本文へ残す。新科学計算は必須でない |
| 3 必須保存証拠の未取得なら技術回収 | 完全なH8 audit／全履歴独立性／private B3最終版custodyを重要claimにする場合 | まずユーザーが強いclaimの必要性を選ぶ。Codexが該当host候補だけread-only照合し、SOURCE/STOPを変えず最小匿名投影を検討。現在の限定claimに一律必須の未回収資料とは扱わない |
| 4 主張を強める場合だけ追加検証 | 実compiled全lawρ、母平均tail／選抜保証、同条件最強family、状態準備、最終長q価格 | 新科学解析・compile等なら別明示許可。全サーバー再調査や全研究再計算を先に要求しない |
| 5 重要な新科学判断 | 主張の正式採択、独立paper/note、Pauli分割、RPEの主要性、Track B構成独立性、保証水準 | ユーザー担当。必要と判断した場合だけ別の詳細科学レビューを指示。本資料は正式判断しない |
| 6 今回の候補では要求しない | RPE R2/R3の順位目的の再開、DF B3追加探索、H8残余一律sampling、H12、新algorithm、noise/実機/FT/実通貨一般化 | 現STOP維持。論文化のための標準追加実験として設定しない |

内部path欠落、未公開の終了metadata、古い書誌名などは、条件付き点比較の符号を直ちに変える問題と同一視しない。一方、候補集合、actual/model価格、状態・law、独立性、半径・保証、同じ推定taskかという違いは、使うclaimの強さを直接変え得る。管理項目の完全性だけを目的に新しい巨大auditを作らない。

## 4. 既存STOPと未保証事項

| 研究 | 保持する境界 |
|---|---|
| H8 rare | 2026-10-21の層別finite sample＋条件付き感度という正式終了。S2_MULTI/S4母平均は未知のまま、0補完・追加samplingなし |
| DF B3 | 保存負結果と終了scope。新候補、再sampling、retry／別namespaceでの再開なし |
| 長時間／高次／確認cohort | 元SOURCE・入力・nominee・retry/resume・価格共有・停止記録を変更しない |
| RPE V1/V2/V3/R1 | 各段の固定条件・結果とmandatory STOPを維持。R1によるR2/R3非推奨を記録し、勝敗を決めるためだけに再開しない |
| Track B C1 | 実分子logical観測の完成と一時凍結。strict native・deterministic halves・共通N取得へ自動進行しない |
| その他の固定契約 | grant/ledger/marker、one-shot・retry/resume、SOURCE/raw/QPY、archiveを保全。今回の許可で既存契約を改変しない |

未保証のまま残るのは、全law実compiled精度、未観測価格tail、選抜後の母平均winner、実状態準備、環境横断の全原本復元、表現／law単独の因果効果、最終energyでの頑健な費用優位、普遍的算法新規性である。これらを現論文候補の暗黙の結論にしない。新しい独立GPT/ClaudeレビューはNOT_STARTED。

## 5. 技術的整合性・原本保全・公開

固定commit、CSV列・ID、旧status/coverage投影のliteral一致、参照先、公開情報の検査がPASS。90固定blobを照合し、主要16資料のGitHub返却blobも一致した。[検査器](validate_synthesis.py)のPASSは文書・metadataの技術整合性であり、新しい科学的PASSではない。科学runnerはimportも実行もしていない。

証拠treeは`GIT_OPTIONAL_LOCKS=0`／`GIT_NO_LAZY_FETCH=1`で読み、独立出力Git/indexに本ディレクトリだけを追加した。公開前の元worktreeのHEAD・tracked/untracked状態は開始時と一致し、元SOURCE・結果・旧原稿・図・Phase0–3/G01/G02・V1–R1・STOPを編集していない。refには追加1・更新1を観測した。本taskから元repoへのref更新操作はなく、観測差は巻き戻さず残した。原本全rawのbyte監査や共有環境全体のatomic凍結を行ったという意味ではない。

公開するのは本統合の匿名文書・metadataだけ。内部host/IP・絶対保存path・資格情報・private raw・seed値の逆引きは掲載しない。原資料のprivate pathは転記せず、公開固定URL・relative path・commitへ置き換えた。既存公開fileの全本文を転載せず、必要なclaimと出典を記した。

専用branch `codex/pr-research-paper-evidence-synthesis-20261103`へcommit/pushし、remote SHAと公開全file本文を照合する。最終SHAと照合結果はチャットで報告する。本fileへ自己参照の最終SHAは埋め込まない。

**公開・照合後mandatory STOP。** 次の担当はユーザー：DF中心範囲、Pauli／RPE／Track Bの配置、正式claimの強さ、次に原稿化・限定資料回収・独立レビューのどれを指示するかを選ぶ。Codexは新科学計算、完成原稿、詳細科学レビュー、投稿、branch整理、既存STOP解除へ自動進行しない。

<!-- fixed-reference-definitions -->
[G01]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/G01_CANDIDATE_BINDINGS.csv>
[G02]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/G02_COHORT_BINDINGS.csv>
[GREPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/REPORT.md>
[H6SEAL]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f79366f68caf8585cc36d2f2b0fefcb1a4bd0d1f/artifacts/resource_applicability/track_a_h6_pr_confirmation_final_seal/2026-10-19/README.md>
[P3CLAIMS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/CLAIM_SCOPE.csv>
[P3COV]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/COVERAGE_MATRIX.csv>
[P3REPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/REPORT.md>
[R1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fde197c64da671f1d428f0d3e8fdb762dc21930/docs/research/pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/REPORT.md>
[SRC_B3PROV]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/scientific_provenance.json>
[V1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md>
[V2]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md>
[V3]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc60beb51930d4311d2ded84440682e583ccafc7/docs/research/pr_rpe_v3_cost_transfer_preflight_20261103/REPORT.md>
