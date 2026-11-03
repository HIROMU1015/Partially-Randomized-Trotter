# G04で解消したアクセス不足と残存制約

2026-11-03 JST。新しい統合追補。旧Phase3・H468比較B・PR統合Cの固定文面・status・rawは変更しない。最新監査入口はSERVER2 commit `37694741ee356fd09b5dfcb56130c55ab381044c` の6ファイル版である。[公開範囲][A_SCOPE]、[SERVER2検査][A_VALID]

## 1. 過去と現在の状態

| 観測・文書 | 当時の状態 | 本追補での扱い |
|---|---|---|
| 元H8結果・protocol・Phase3 | 結果は公開、詳細completion／private rawにアクセス制約。対応coverageはNOT_ACCESSIBLE | 当時の科学結果・監査状態・公開制約をそのまま保存 |
| C統合 `2ce2b5f1…` | 詳細候補のGitHub404、SERVER2所在候補のみ | 404は当時の応答。元private候補全commitの公開を今回確認したとはしない |
| B比較 `713c45f8…` | 限定照会で新補完未取得、旧候補GETは422。G04 metadata回収が残る | 旧資料は履歴。現在の監査参照に本追補を重ねる |
| SERVER2の前4ファイル版 | 匿名詳細監査の公開保留という履歴 | 今回はその版でなく、明示指定の新6ファイル版を採用。過去記録を遡及修正しない |
| 最新6ファイル版 | 終了・SOURCE／結果hash・192軸provenanceが匿名公開 | **ANONYMIZED_COMPLETION_METADATA_AVAILABLE_WITH_REMAINING_LIMITS**。アクセス不足はこの公開scopeで解消 |

G04の「匿名詳細終了metadataが取れない」という問題と、private raw全体の第三者再監査ができないという限界は別である。前者の解消を後者の解消へ拡張しない。旧G04やPhase3のA–Eを新しい科学分類へ置き換えない。[旧coverage][PHASE3_COVERAGE]、[元結果][RESULT_REPORT]、[最新終了監査][A_AUDIT]

## 2. 確認できた事項と残る事項

| 項目 | 確認できること | 依然として未確認・未保証 |
|---|---|---|
| H/state/reference | 公開binding、snapshot・preparation等のidentifierを固定designへ対応付け。referenceの保存literal一致は作者監査が記録 | private配列・生成過程の独立復元、三系の完全共通入力生成規則 |
| p8 export flag | **`saved_exact_export_match=false`** が保存されている。`matches_fixed_design_hash=true` と別のfield | falseの比較対象・原因・科学的同等性はUNRESOLVED。trueへの訂正根拠はない |
| p7 export flag | 保存値はtrue | p8のfalseを解消する証拠ではない |
| 固定候補／対照／保存N/G/SE | 元primary record1/3、主表、最新対応表のliteral接続 | 新しいbias保証・母平均winner・全184実測最良。p8副評価PF nullはそのまま |
| SOURCE／hash | 登録11公開SOURCEのblob／bytes／SHA-256、原result等の保存hashへの参照 | private全receiptのLOCAL再hash、元SOURCE全依存の完全監査、独立科学再実行 |
| 192軸 | scope／replica／axisと元価格表の対応、RZ・再利用flag、保存compiler/fingerprint/task-result hash・終了metadata | 元compiled回路からのfingerprint再導出、再compile一致、原profile bytes完全同一性 |
| Resume／終了 | p8旧1＋新47、p7新48、driver exit0/reapedという作者metadata | 失敗なし／retryなしという歴史改変。旧STOP・中断・未完了予約は保持 |
| seed/index | 登録index/pair等の限定照合と、2026-10-20保存派生seed監査の範囲が明示 | 全履歴seed独立性・数学的確率独立性。seed値やRNG replayは取得／実行しない |
| 希少event | 既存層別有限標本と条件付き残余感度の正式終了 | 未観測S2_MULTI/S4の費用・分散上界・母平均優位。0補完しない |
| 物理域・最終応用 | H8固定T.8・2精度の点確認 | H8長時間・高次PR、統一size scaling、状態準備込み最終energyの頑健資源優位 |

これらの未保証はmetadataの公開だけで変わらない。元hashの存在は完全な原本復元・科学的同等性・第三者再現を意味しない。[終了監査のlimits／preparation][A_AUDIT]、[192軸][A_AXES]、[元比較][ORIGINAL_COMPARISONS]

## 3. 比較表・O1への反映

[H468_COVERAGE_ADDENDUM.csv](H468_COVERAGE_ADDENDUM.csv)の先頭32列はBの36行を全field保持する。追加列には日付、新status、出典、現在確認／未確認、LOCALの検査範囲、同cohort再利用、export flag、旧Phase3 coverage、旧C採択statusを記録した。

更新の中心はH8固定2比較、INPUT_H8、SIGNAL_H8、NATIVE_H8、COHORT_H8、H8_G04、H8_PROFILE_UNRESOLVEDの8行。H4/H6やH8広域比較の旧statusを新PASSへ揃えない。H8_PROFILE_UNRESOLVEDは未解決のままであり、G04アクセス解消と両立する。

C A08の詳細監査アクセスに後続証拠を付けられる。C A07の強い対照による正負と、A13の希少層終了は維持する。本文・Methods・SI・captionの参照要件は[O1証拠対応](O1_H8_CLAIM_EVIDENCE_UPDATE.md)に整理した。これは正式採択や原稿本文ではない。

## 4. 追加作業・STOP

**現在のO1論文に新しいH8科学計算は必須ではない。** G04への必要な匿名metadata回収は指定6ファイルの範囲で完了し、形式的なSERVER1/2再調査、元確認の再sampling／再compileは要求しない。残るp8 flag・原profile等は制限として開示し、新しい計算案・実行条件は追加しない。

H8希少RTEの正式終了・STOP、元固定確認のSOURCE／retry／resume／grant／ledger／marker／STOP、他研究の停止条件を保全する。H8長時間・高次PRや最終energyの保証が完成したことにはしない。[H8正式終了政策][CLOSURE]、[既存の追加計算不要判断][B_PLAN]

次の担当はユーザー：O1の正式採択と原稿作成の別指示を判断する。詳細科学レビューと新科学実行は、それぞれ独立の明示許可が必要。**本成果物のcommit／push・remote SHA／全公開本文照合後mandatory STOP。** Codexは原稿本文・図表作成、レビュー、科学実行、既存STOP解除へ自動進行しない。

<!-- fixed-reference-definitions -->
[A_AUDIT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/COMPLETION_AUDIT.json>
[A_AXES]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/AXIS_PROVENANCE.csv>
[A_SCOPE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/PUBLICATION_SCOPE.md>
[A_VALID]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/VALIDATION.json>
[B_PLAN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/713c45f84c8d7a41ceaef93381bd083e22189085/docs/research/h8_h46_validation_coverage_gap_20261103/H8_MINIMAL_NEXT_PLAN.md>
[CLOSURE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md>
[ORIGINAL_COMPARISONS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/comparisons.csv>
[PHASE3_COVERAGE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/COVERAGE_MATRIX.csv>
[RESULT_REPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/docs/research/track_a_h8_new_pr_confirmation_results_20261019.md>
