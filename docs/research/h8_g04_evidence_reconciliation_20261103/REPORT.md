# H8固定独立確認とG04：SERVER2証拠対応調査

2026-11-03 JST。既存資料の読取、identity/hash・保存tokenの照合、公開可能なmetadataの整理を実施した。新科学計算、保存値の数値再解析、sampling、回路build/compile、詳細科学レビューは実施していない。担当はprimary Codexのみ。

**H8の結果・固定SOURCE・cohort対応に加え、終了・原結果hash・192軸の匿名provenanceをリポジトリへ収録した。G04の「詳細監査metadataを取得できない」という障害はこの範囲で解消した。H4/H6との完全な監査同等性や母平均優位は認定しない。** 前版の公開保留は、匿名化監査metadataの収録をユーザーが許可したため本版で更新した。対象GitHubリポジトリはpublicであることを確認済み。過去の固定結果・監査statusは変更していない。

## 固定2条件と保存値

共通条件はlinear H8、隣接1.00 Å、STO-3G、tol-only 1e-8 DF rank28、16 spin modes/sector4900、T=.8、P=0、通常signed bias、allowance倍率1。PRは二次partial PF・r4/K4、元DF順序・無条件finite-RTE lawを維持する。scalarはfull-Fockで一回、reference不回転、ordinary ancilla/symmetric_directional control。Qiskit 1.3.0、opt1、basis rz/sx/x/cx、seed_transpiler17、測定込みfull Hadamard wrapper、状態準備費用除外。

|主表ID / Mechanism ID|PR候補 / δ|ε / 元n|固定PF / RZ単価（各軸）|PR平均G exact / paired SE|固定PF G|
|---|---|---|---|---|---|
|H8_DF28_P_1 / H8_CONFIRM_1|B2_p8_q2_2nd_r4_K4 / .4|.05 / 48|B0I_p11_q1_2nd / 42268|9840910801/8 / 3175066.353534034|1267321444|
|H8_DF28_P_2 / H8_CONFIRM_3|B2_p7_q16_2nd_r4_K4 / .05|.0005 / 48|B0I_p15_q4_4th / 668636|546493203592571/6 / 144211830223.71762|94193016652860|

これは[元比較表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/comparisons.csv)のprimaryデータ行1/3のliteral転記である。方向は両方POINT_REPRODUCED・PR低費用。N（保存analytic shot）はPRが(7659,8832)/(79863816,99933856)。Cは[保存192軸表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/paired_metrics.csv)のRZ列を参照する。新しい軸平均C、G、SE、差・比・順位は計算していない。SEは同じtrajectoryのcosine/sineを組にした費用の保存標準誤差で、量子shotのSEではない。副評価は同じ標本の記述的再引用であり、p8の固定PF不適格・比較nullを保持する。

[二条件CSV](H8_CONFIRMATION_EVIDENCE.csv)に固定入力identity、SOURCE、cohort、N/C/G/SE参照と公開監査状態を保存した。inputのHamiltonian/state/snapshot識別子は、公開済みの[固定design](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_design/2026-10-18/plan_PROPOSED.json)のbindingから転記した。[終了監査metadata](COMPLETION_AUDIT.json)には、保存referenceのliteral一致、signal/accounting/law hashと固定input bytesの照合範囲も記録した。新しいreference計算や配列decodeは行っていない。識別子や本監査projectionだけで、第三者がraw入力・生成過程の全検査を独立再実行できるとはしない。

## SOURCE・候補固定・版の役割

|役割|固定object|今回の位置付け|
|---|---|---|
|実行SOURCE|745a674546f2cb6258526fbfde1d9a00601689f8|元結果と公開SOURCEを結合。短縮SHAの推測補完ではない|
|元dataset/seed SOURCE|1549b6f891a15ff48379996c34b344118ba2606c|[公開resume起動記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/be2e5475f4dbb5a07e947b053e8c419109f1eda9/docs/research/track_a_h8_confirmation_parallel_resume01_20261019.md)が保持する元登録の来歴。修正後実行SOURCEと区別|
|探索結果|64c85f7a833b5950ef2ccae623708b73081edcd6|探索n8による候補選択の履歴。n48へpoolしない|
|候補固定design|434e67a8bb205711d22d5bf1aa5d143fd08b7765|2候補・PF・ε・各48を新確認標本を見る前に固定した設計。DESIGN_ONLY_UNAPPROVEDは当時の設計statusで実行許可を意味しない|
|結果公開|e6eea5387a6e84d1e39a9adb3b61975acc74e200|比較4行と192軸projection、限定結果報告|
|公開回収索引|9f33ad0b11b4febe48b802000c38c5a0e3fd267c|[既存18資料の索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9f33ad0b11b4febe48b802000c38c5a0e3fd267c/docs/research/track_a_h8_evidence_recovery_public_index_20261020.md)。詳細PR実行receiptの代替ではない|

SERVER2の明示索引に従い、候補OID `659e555714c8ccf3a6ec7fd1cb0dbf05cfcc342d` と `96aa3878a344f479ee064118cb90002a770db627` のcommit実体と指定資料を別cloneで確認した。これらは結果資料のlocal版で、科学SOURCEや新しい実験ではない。両版の比較表は公開表とbyte一致した。軸表は96aa版が公開版とbyte一致し、659e版は追加provenance列を持つが公開列の全行literalは一致する。異なる版を同一bytesと記録せず、今回、必要な科学的hashと監査metadataは[COMPLETION_AUDIT.json](COMPLETION_AUDIT.json)、192軸のmeasurement SOURCE・fingerprint・worker状態・task-result hashは[AXIS_PROVENANCE.csv](AXIS_PROVENANCE.csv)へ匿名化して収録する。元の内部path・seed値・raw全体は収録しない。candidate object全体の完全性、全祖先・backupの復元性まで再認定していない。

公開[起動記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/be2e5475f4dbb5a07e947b053e8c419109f1eda9/docs/research/track_a_h8_confirmation_parallel_resume01_20261019.md)のRUNNINGは2026-10-19の時点記録であり、[後続結果報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/docs/research/track_a_h8_new_pr_confirmation_results_20261019.md)の完了記述と区別する。元p8のidentity list/tuple受入STOP、元p7の所有run中断、固定datasetでのresumeを保存する。公開起動記録はp8の完全1軌道再利用、残47/48軌道・新190軸、最終192軸の計画を記録し、結果公開は完了と同じ2軸の再利用を報告する。起動時予算と実完了receiptを同一視せず、全192軸を修正SOURCEで新規compileしたとも、失敗・retryなしとも記述しない。旧STOP解除・再試行・grant再利用は今回行っていない。

## 主表・Mechanism・Figure 2B

[主表JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.json)のH8_DF28_P_1/2は、[Mechanism metadata](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1ff6ac0b6ffed6231dac5f655275b620598d0201/artifacts/resource_applicability/track_a_mechanism_results/2026-10-19/comparison_metadata.csv)のH8_CONFIRM_1/3とPR/PF/ε/nが一致する。cohort名INDEPENDENT_UNCONDITIONAL_CONFIRMATION_N48とINDEPENDENT_FIXED_CONFIRMATION_N48は同じ確認結果のaliasである。[Figure 2B保存値](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb33de67475e7d33eed56801b74a2a277147e4be/artifacts/resource_applicability/track_a_mechanism_figure2_a82_readiness/2026-10-19/figures/plot_values.csv)もこの2比較を参照し、主表に保存されたR_N/R_Ceff/R_Gとliteral一致する。別の独立追試として加算しない。2Aのlegacy/探索cohortや新H4 DF10/H6確認へ機構因子を転用しない。

## H4/H6との監査水準の比較

|項目|H4 tol-only DF10 P|H6 tol-only DF19確認|H8 tol-only DF28確認|
|---|---|---|---|
|固定確認結果|n1108/n78、粗PF低費用・NOT_REPRODUCED、細PR低費用|n48/n104、粗PF低費用方向を再現、細PR低費用|各n48、両方PR低い標本点費用を再現|
|公開終了監査|[completion audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/completion_audit.json)・native exit0・1186 worker回収・2372軸|[native completion](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cccfcb2051dacf29e91473e2e1152acef1574bd/artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/native_completion.json)・両driver exit0/reaped・152軌道/304軸、source preservation/ledger|終了・SOURCE/原結果/terminal hash・全192軸対応の作者metadata監査を収録。raw receipt全体の第三者再検証は未達|
|seed/index監査|[seed refresh](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/seed_refresh_projection.json)はcutoff時点の明示除外70270、collision0、future_unregistered_campaigns_covered=false|公開SOURCE/input/registry識別と終了資料。今回新たな全履歴seed監査は行わない|登録seed/indexの照合と除外集合に対するmaster非重複を確認。既存の派生seed/旧登録監査は2026-10-20の保存receiptとして区別し、再実行していない|
|今回の公開status|PUBLIC_COMPLETION_RECEIPTS_VERIFIEDを元資料の範囲で保持|同左|過去のstatusを保持し、現在はPUBLISHED_AUTHOR_METADATA_AUDIT_LIMITED_COMPLETION_AND_LITERAL_JOIN|
|保証しない事項|母平均優位・全support分散・未登録将来seed独立性|母平均優位・全履歴の数学的独立性|同左に加え、原profile bytes・旧compiler出力完全同一性・p8保存export一致flagの未解決|

過去の固定[protocol_scope_status](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/protocol_scope_status.csv)の状態を更新・昇格していない。H4/H6の公開監査も作者の技術証拠であり、外部再現・immutable CI・別reviewerの合格を意味しない。系間でSCF/orbital/integral/DF/state生成規則が完全に同じとの認定も行わない。

固定候補と探索を分離した設計、探索n8/旧n32/希少pilotをpoolしない方針は公開資料で確認できる。seed不一致・namespace差だけで確率独立性を証明しない。今回の終了・pair照合は原本metadataを用いた限定検査である。SOURCEの検査機能の存在だけをPASSの根拠にせず、全履歴seedとの非重複・数学的独立性を推測でPASSにしない。

## 収録した終了監査と残存gap

[COMPLETION_AUDIT.json](COMPLETION_AUDIT.json)は各条件のdriver exit0/reaped、全48登録task/96軸との対応、原結果・native・terminal・seal・resume・registryの科学的identity、固定inputとSOURCEの照合、保存N/G/SEを収録する。[AXIS_PROVENANCE.csv](AXIS_PROVENANCE.csv)は既存192軸のscope/replica/axisをキーに、measurement SOURCE・compiler hash・fingerprint・task-result hash・exit/reapを結ぶ。保存値のliteral照合であり、新平均・SE・G・順位は計算しない。

p8は元SOURCE 1549b6f8…で完了した1軌道を再利用し、修正SOURCE 745a6745…で残47軌道・94軸を完了した。p7は48軌道・96軸を修正SOURCEで完了した。新規trajectory親は47/48、新規の軸/control childは94/101。旧p8受入STOPと旧p7中断、未完了予約の非返却を保持する。新規完了数と最終受入数を混同しない。

登録seed/index/trace/両軸の一致、各48masterの一意性、保存除外集合とのmaster非重複は2026-11-03の限定照合で確認した。既存の2026-10-20監査は旧除外4136、新A登録master128、合成除外5000、確認task96、旧探索登録592行・旧確認登録192行のmaster/派生欠落0を記録する。後者は保存監査の転記で、今回の派生seed再導出や全履歴保証ではない。seed値は収録しない。

**p8の保存preparation資料にある `saved_exact_export_match=false` は未解決として明示する。** 固定designのpreparation hashと実行側の保存identityの一致は別の確認であり、このexport照合flagをtrueへ変更しない。p7の同flagはtrue。falseの比較対象・原因や科学的同等性は今回新たに認定しない。p8に関して入力復元が全面的に同一とする記述は避ける。

[固定Phase 3 G04](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/REPORT.md)は当時の状態として変更しない。現在は終了・public/private原結果対応の匿名metadataへの接続が可能になった。一方、raw receipt全体を公開していないため、第三者による全rawチェックの再実行やH4/H6との完全同等性は認定しない。G04への今回の補足状態は **ANONYMIZED_COMPLETION_METADATA_AVAILABLE_WITH_REMAINING_LIMITS** とする。

|残存項目|性質|LOCALへ渡す内容|
|---|---|---|
|p8 saved_exact_export_match=false|保存資料の比較flag不一致・意味未確定|COMPLETION_AUDITのpreparation節を参照。SOURCE/input identity一致と分離し、全面PASSにしない|
|原profile bytes・元compiled回路からのfingerprint再導出|原資料未取得／確認不能|保存fingerprintを公開しても再導出したことにはしない|
|全履歴seed独立性、母平均優位、未観測supportの費用・分散|科学的未保証|CI/p値null、未知費用を0補完しない。監査metadata収録で保証が増えたとはしない|
|raw・内部path・seed一覧|今回の収録対象外|匿名projectionと科学的hashを使う。rawの自動送付・新しい認証/承認証明基盤は作らない|
|希少RTEと元確認監査の区別|研究scopeの区別|正式終了方針862d732f…を維持。S0/S2_SINGLEと残余S2_MULTI/S4_PRESENTの保存解析を元n48実行監査の代わりにしない|

## 成果物・完了境界

本ディレクトリはREPORT、H8_CONFIRMATION_EVIDENCE、PUBLICATION_SCOPE、VALIDATIONと、新たなCOMPLETION_AUDIT、AXIS_PROVENANCEの6ファイル。原本・科学SOURCE・input・結果・STOPは保全し、primary Codexが既存照合結果の転記と匿名化・構造検査を実施した。科学runnerをimport/起動せず、新しいsampling/compile/数値再解析は0。

**MANDATORY STOP。** 次の担当はLOCAL Codex：別途依頼された統合作業の中で本6資料をG04の補足として使用する。固定Phase 3の過去状態、今回の取得可能範囲、残る未確認を区別する。今回のCodex自動継続作業はなく、新科学計算や希少RTE研究を開始しない。
