# H8 G04詳細監査の統合とO1証拠資料の追補

2026-11-03 JST、LOCAL、主エージェントのみ。既存資料の技術照合と文書更新である。新科学計算・保存数値の再解析・sampling・simulation・回路build/compile・独立詳細科学レビュー・原稿本文／図表作成・中心claimの正式採択は行っていない。

**G04の現在の追補状態は `ANONYMIZED_COMPLETION_METADATA_AVAILABLE_WITH_REMAINING_LIMITS`。** SERVER2公開の最新版6ファイルにより、H8固定2条件の詳細終了metadata、科学SOURCE／保存結果／cohort、192軸の匿名provenanceへ接続できる。過去のアクセス不足はこの範囲で解消した。raw全体の第三者再監査やH4/H6との完全な監査同等性を認定した意味ではない。**p8の `saved_exact_export_match=false` は未解決として保持する。** [SERVER2報告][A_REPORT]、[終了監査][A_AUDIT]、[公開範囲][A_SCOPE]

## 1. 固定入力と今回の読取範囲

| 入力 | 固定commit | 使用範囲 |
|---|---|---|
| SERVER2 G04最新版 | `37694741ee356fd09b5dfcb56130c55ab381044c` | REPORT、H8_CONFIRMATION_EVIDENCE、COMPLETION_AUDIT、AXIS_PROVENANCE、PUBLICATION_SCOPE、VALIDATIONの6ファイル。GitHub返却blobとLOCAL固定objectを照合 |
| H4/H6/H8比較資料 | `713c45f84c8d7a41ceaef93381bd083e22189085` | 指定4資料、coverage36行、既存validator／VALIDATION。旧fieldをliteral保持 |
| PR全体の論文化統合 | `2ce2b5f161e94567fec322d694fd8b01d03cd854` | 指定5資料、特にA07/A08/A13とO1、旧取得制約、本文／SI設計。旧採択statusを保持 |

必要な範囲だけ、元比較表・192軸価格表・固定design・SOURCE登録11ファイル・resume／結果報告・主表／protocol・Mechanism／Figure2B・H4/H6終了記録・Phase3 G04へ戻った。公開Git objectとmetadataの照合であり、SERVER2のprivate実体、raw、配列、QPY、運用receipt全体を新たに取得したわけではない。科学SOURCEは静的に読んだだけでimport／実行していない。既存の研究概要・現状と研究索引は前段で読んだものを導線として再利用し、全repositoryの再棚卸しは行わない。[旧比較報告][B_REPORT]、[旧統合][C_REPORT]

## 2. 固定2比較と同一cohortの接続

共通条件はlinear H8、隣接1.00 Å、STO-3G、tol-only DF threshold1e-8、保存rank28、16 spin modes／sector4900、T=.8、P=0。二次partial PF、無条件finite-RTE、通常signed bias・allowance倍率1、元DF順序、full-Fock scalar一回、reference不回転、ordinary ancilla／symmetric-directional control。Qiskit1.3.0、opt1、rz/sx/x/cx、transpiler seed17、測定込みfull Hadamard wrapper、状態準備除外。

| 主表ID | 元scope／固定PR | ε・δ・n | 固定PF | 同一結果のalias |
|---|---|---|---|---|
| H8_DF28_P_1 | H8_CONFIRM_P8_Q2／B2_p8_q2_2nd_r4_K4 | .05／.4／48 | B0I_p11_q1_2nd | H8_CONFIRM_1、Figure2B |
| H8_DF28_P_2 | H8_CONFIRM_P7_Q16／B2_p7_q16_2nd_r4_K4 | .0005／.05／48 | B0I_p15_q4_4th | H8_CONFIRM_3、Figure2B |

元比較表のprimary data record1/3、主表JSONの同ID、Mechanismのcontrast_id、Figure2Bの保存比をliteral照合した。両条件の保存方向はPOINT_REPRODUCED・PR低い標本点費用。cohort名のUNCONDITIONALとFIXEDはこの確認のaliasであり、文書数・図数を独立実験数に加算しない。探索n8、旧union6/n32、条件付き希少pilotを確認n48へpoolしない。Nは保存analytic shots、nは古典価格標本、SEはpaired軸価格の保存標準誤差である。[二条件表][A_EVIDENCE]、[元比較][ORIGINAL_COMPARISONS]、[主表][MAIN]、[Mechanism][MECHANISM]、[Figure2B保存値][FIGURE2B]

## 3. 終了・hash・192軸の照合

- SOURCE登録11ファイルのcommit／path／blob／bytes／SHA-256をCOMPLETION_AUDITのSOURCE_closureと照合した。実行SOURCE `745a674546f2cb6258526fbfde1d9a00601689f8`、元dataset/seed SOURCE `1549b6f891a15ff48379996c34b344118ba2606c`、結果公開commit `e6eea5387a6e84d1e39a9adb3b61975acc74e200`は役割を分ける。
- H/state/snapshot/compiler/profileの公開identifierとpreparation hashを固定designへ結んだ。公開保存identityの一致であり、private入力配列の独立復元・新しいreference計算ではない。
- AXIS_PROVENANCEのscope／replica／axisを元paired_metricsへ一対一に結び、candidate・cohort・再利用flag・RZをliteral照合した。各scopeでreplica0..47×cosine/sine、全192軸が対応する。保存compiler hash・fingerprint・task-result hashとworker exit0／reapedのmetadataを確認した。fingerprintを元回路から再導出していない。
- 終了監査のdriver exit0／reaped、各48受入・96軸、保存N／G／SD／SEとnullを原公開結果へ接続した。privateな原result/native/terminal/seal/resume/registryのhashはSERVER2が保存照合を記録した科学的identityとして扱い、LOCALがそれら全rawのhashを再計算したとは記述しない。

新しい平均・SE・G・比・順位・信頼区間は一切計算していない。検査対象は保存token・キー集合・参照・SOURCE/blobの整合性である。[匿名軸provenance][A_AXES]、[元192軸価格][ORIGINAL_AXES]、[固定design][DESIGN]、[SERVER2検査範囲][A_VALID]

### Resumeと再利用

p8は元SOURCEで完了していたreplica0の1trajectory／2軸を再利用し、修正SOURCEで残47trajectory／94軸を完了。p7は修正SOURCEで48trajectory／96軸を完了した。最終受入は各48であり、全192軸が修正SOURCEによる新compileではない。新規trajectory親47/48と、control等を含む新規child94/101も別の数である。

元p8のNATIVE_START_IDENTITY STOP、元p7の中断、未完了予約の非返却、同datasetでのresumeを消さない。旧起動報告のRUNNINGは当時の時点記録、後続結果と今回の匿名終了metadataは完了時の記録として併存する。今回retry・grant利用・job操作をしていない。[resume起動記録][RESUME]、[元結果報告][RESULT_REPORT]、[終了監査][A_AUDIT]

## 4. H4/H6との比較と旧状態

[追補CSV](H468_COVERAGE_ADDENDUM.csv)は旧coverageの36行・先頭32列をすべてそのまま保持し、H8関連8行に明示的な更新を付けた。旧original_status／gap_class／保存N_G_SE等と、新addendum_status／出典／確認事項／未確認事項を別欄にする。残りの行は旧証拠の保持であり、新たな同水準PASSを発行していない。

| 観点 | H4 DF10 | H6 DF19 | H8 DF28の現在の追補 |
|---|---|---|---|
| 固定2精度 | 旧主表の粗PF低費用・非再現／細PR低費用を保持 | 粗PF低費用方向再現／細PR低費用を保持 | 固定2候補の保存点方向は両PR低費用。主表・Mechanism・Figure2Bは同cohort |
| 終了・回収 | 公開completion、native exit0、worker回収記録 | 両driver exit0／reaped、native completion記録 | 作者による匿名の終了・原結果hash・192軸対応が公開。raw receipt全体の第三者再実行ではない |
| seed/index | cutoff付き除外・collision0、将来未登録を含まない | 保存cohortと公開終了資料。全履歴最終seed sealまで閉じていない | 登録index・paired binding・限定master照合と、2026-10-20派生seed監査receiptの転記を区別。全履歴／数学的独立性は未認定 |
| 入力・保証 | H/state/referenceの元条件を保持 | weighted-Hermitian completion等の元条件を保持 | 保存bindingに加えp8 export flag未解決。三系の完全同一生成規則・母平均winnerを認定しない |

H4/H6の公開completionも作者の技術証拠であり、外部再現／独立reviewではない。H8の匿名公開によって監査項目を形式的に同じPASSへ揃えない。[H4終了][H4_COMPLETION]、[H4 seed範囲][H4_SEEDS]、[H6終了][H6_COMPLETION]、[旧protocol][PROTOCOL]

旧Phase3のNOT_ACCESSIBLE、Cの404記録、Bの422／未取得記録、元結果の詳細公開制約、前4ファイル版の保留は履歴として保持する。現在は固定6ファイルを使える、という後続日付付きの追補である。NOT_ACCESSIBLEを科学未実施へ読み替えず、今回のmetadata取得を科学的PASSへ読み替えない。[旧Phase3 coverage][PHASE3_COVERAGE]、[旧統合status][C_STATUS]

## 5. 残存事項とO1への影響

p8 `saved_exact_export_match=false` の比較対象・原因は公開元資料だけでは確定しない。preparation hash／固定design一致は別の検査であり、このflagをtrueへ変えない。p7の保存flagはtrue。元profile bytes、compiled回路からのfingerprint再導出、全履歴seed独立性、未観測希少eventの費用・分散、母平均優位、三系の完全共通入力生成、最終energyの頑健資源優位も未保証のまま。[残存事項](STATUS_AND_REMAINING_GAPS.md)

O1ではH8固定2精度を本文の限定比較へ、強化184の条件依存性を別domainの比較へ置ける。SOURCE／cohort／終了監査／192軸／希少層／p8制約はMethods・SI・captionの限定へ対応付けた。旧CのA08「詳細監査への公開アクセス制約」はこの匿名scopeで更新でき、A07の強化対照やA13の希少層終了scopeは変えない。長時間・高次PRがH8でも検証済みになったわけではない。[O1用対応](O1_H8_CLAIM_EVIDENCE_UPDATE.md)

**現在のO1論文に、新しいH8科学計算は必須ではない。** この判断を変える新科学証拠は今回ない。新規計算の提案・新しい実行条件は追加しない。原稿採用・中心claimの強さはユーザー判断のままである。

## 6. 技術検査・公開・STOP

既存B／C validatorのCSV・固定Git・literal検査を再利用し、この追補のキーと列に限った[小さな検査器](validate_addendum.py)を作った。旧36行全field、6固定比較、元status、SOURCE登録、2条件／192軸／alias、false/null、公開情報・リンクを検査する。[VALIDATION.json](VALIDATION.json)のPASSはmetadataの整合性であり、独立科学レビューやscientific PASSではない。

元worktreeのAGENTS、親規約候補、PROJECT_MAP、正式共通指示2026-10-25の§8.1／§8.4／§10を確認。読取はGIT_OPTIONAL_LOCKS=0／GIT_NO_LAZY_FETCH=1。独立出力Gitに本ディレクトリだけを追加し、原SOURCE／input／raw／QPY／結果／grant／ledger／marker／STOP／別branch／worktreeを変更しない。SSH、全raw hash／decode、科学runner、新しい管理基盤、共有環境・他ユーザーのjob操作は行わない。非公開path・host/IP・資格情報・seed値を公開しない。

公開前の観測で元HEAD・tracked状態は開始時と一致した。ref・未追跡状態には並行する変化があり、そのまま保持した。状態一覧の一致確認は原本全rawのbyte監査ではなく、共有checkout全体の凍結も行っていない。

専用branch `codex/h8-g04-o1-evidence-integration-20261103`へcommit／通常pushし、remote SHAと全公開本文を照合する。最終SHA／照合結果はチャットで報告する。**公開・照合後mandatory STOP。** 次の担当はユーザー：O1の正式採択と原稿作成を指示するか判断する。Codexは原稿本文・図表・詳細レビュー・科学実行へ自動進行しない。

<!-- fixed-reference-definitions -->
[A_AUDIT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/COMPLETION_AUDIT.json>
[A_AXES]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/AXIS_PROVENANCE.csv>
[A_EVIDENCE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/H8_CONFIRMATION_EVIDENCE.csv>
[A_REPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/REPORT.md>
[A_SCOPE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/PUBLICATION_SCOPE.md>
[A_VALID]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/VALIDATION.json>
[B_REPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/713c45f84c8d7a41ceaef93381bd083e22189085/docs/research/h8_h46_validation_coverage_gap_20261103/REPORT.md>
[C_REPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ce2b5f161e94567fec322d694fd8b01d03cd854/docs/research/pr_research_paper_evidence_synthesis_20261103/RESEARCH_SYNTHESIS.md>
[C_STATUS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ce2b5f161e94567fec322d694fd8b01d03cd854/docs/research/pr_research_paper_evidence_synthesis_20261103/STATUS_AND_OPEN_QUESTIONS.md>
[DESIGN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_design/2026-10-18/plan_PROPOSED.json>
[FIGURE2B]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb33de67475e7d33eed56801b74a2a277147e4be/artifacts/resource_applicability/track_a_mechanism_figure2_a82_readiness/2026-10-19/figures/plot_values.csv>
[H4_COMPLETION]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/completion_audit.json>
[H4_SEEDS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/seed_refresh_projection.json>
[H6_COMPLETION]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cccfcb2051dacf29e91473e2e1152acef1574bd/artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/native_completion.json>
[MAIN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.json>
[MECHANISM]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1ff6ac0b6ffed6231dac5f655275b620598d0201/artifacts/resource_applicability/track_a_mechanism_results/2026-10-19/comparison_metadata.csv>
[ORIGINAL_AXES]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/paired_metrics.csv>
[ORIGINAL_COMPARISONS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/comparisons.csv>
[PHASE3_COVERAGE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/COVERAGE_MATRIX.csv>
[PROTOCOL]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/protocol_scope_status.csv>
[RESULT_REPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/docs/research/track_a_h8_new_pr_confirmation_results_20261019.md>
[RESUME]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/be2e5475f4dbb5a07e947b053e8c419109f1eda9/docs/research/track_a_h8_confirmation_parallel_resume01_20261019.md>
