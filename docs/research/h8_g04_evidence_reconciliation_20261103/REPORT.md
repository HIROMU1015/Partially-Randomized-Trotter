# H8固定独立確認とG04：SERVER2証拠対応調査

2026-11-03 JST。既存資料の読取、identity/hash・保存tokenの照合、公開可能なmetadataの整理を実施した。新科学計算、保存値の数値再解析、sampling、回路build/compile、詳細科学レビューは実施していない。担当はprimary Codexのみ。

**H8の結果・固定SOURCE・主表・Mechanism・Figure 2Bの対応は確認できた。H4/H6と同水準の公開終了監査が揃ったとは判定しない。G04の詳細audit接続は公開制約により未解消である。** 原本の限定照合と非公開補足は所有環境内で保全し、その監査内容・hash一覧をこの公開成果へ移していない。

## 固定2条件と保存値

共通条件はlinear H8、隣接1.00 Å、STO-3G、tol-only 1e-8 DF rank28、16 spin modes/sector4900、T=.8、P=0、通常signed bias、allowance倍率1。PRは二次partial PF・r4/K4、元DF順序・無条件finite-RTE lawを維持する。scalarはfull-Fockで一回、reference不回転、ordinary ancilla/symmetric_directional control。Qiskit 1.3.0、opt1、basis rz/sx/x/cx、seed_transpiler17、測定込みfull Hadamard wrapper、状態準備費用除外。

|主表ID / Mechanism ID|PR候補 / δ|ε / 元n|固定PF / RZ単価（各軸）|PR平均G exact / paired SE|固定PF G|
|---|---|---|---|---|---|
|H8_DF28_P_1 / H8_CONFIRM_1|B2_p8_q2_2nd_r4_K4 / .4|.05 / 48|B0I_p11_q1_2nd / 42268|9840910801/8 / 3175066.353534034|1267321444|
|H8_DF28_P_2 / H8_CONFIRM_3|B2_p7_q16_2nd_r4_K4 / .05|.0005 / 48|B0I_p15_q4_4th / 668636|546493203592571/6 / 144211830223.71762|94193016652860|

これは[元比較表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/comparisons.csv)のprimaryデータ行1/3のliteral転記である。方向は両方POINT_REPRODUCED・PR低費用。N（保存analytic shot）はPRが(7659,8832)/(79863816,99933856)。Cは[保存192軸表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/paired_metrics.csv)のRZ列を参照する。新しい軸平均C、G、SE、差・比・順位は計算していない。SEは同じtrajectoryのcosine/sineを組にした費用の保存標準誤差で、量子shotのSEではない。副評価は同じ標本の記述的再引用であり、p8の固定PF不適格・比較nullを保持する。

[二条件CSV](H8_CONFIRMATION_EVIDENCE.csv)に固定入力identity、SOURCE、cohort、N/C/G/SE参照と公開監査状態を保存した。inputのHamiltonian/state/snapshot識別子は、公開済みの[固定design](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_design/2026-10-18/plan_PROPOSED.json)のbindingから転記した。referenceの値・raw入力bytes・生成過程の完全性を、これらの識別子だけで公開再認証したとはしない。

## SOURCE・候補固定・版の役割

|役割|固定object|今回の位置付け|
|---|---|---|
|実行SOURCE|745a674546f2cb6258526fbfde1d9a00601689f8|元結果と公開SOURCEを結合。短縮SHAの推測補完ではない|
|元dataset/seed SOURCE|1549b6f891a15ff48379996c34b344118ba2606c|[公開resume起動記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/be2e5475f4dbb5a07e947b053e8c419109f1eda9/docs/research/track_a_h8_confirmation_parallel_resume01_20261019.md)が保持する元登録の来歴。修正後実行SOURCEと区別|
|探索結果|64c85f7a833b5950ef2ccae623708b73081edcd6|探索n8による候補選択の履歴。n48へpoolしない|
|候補固定design|434e67a8bb205711d22d5bf1aa5d143fd08b7765|2候補・PF・ε・各48を新確認標本を見る前に固定した設計。DESIGN_ONLY_UNAPPROVEDは当時の設計statusで実行許可を意味しない|
|結果公開|e6eea5387a6e84d1e39a9adb3b61975acc74e200|比較4行と192軸projection、限定結果報告|
|公開回収索引|9f33ad0b11b4febe48b802000c38c5a0e3fd267c|[既存18資料の索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9f33ad0b11b4febe48b802000c38c5a0e3fd267c/docs/research/track_a_h8_evidence_recovery_public_index_20261020.md)。詳細PR実行receiptの代替ではない|

SERVER2の明示索引に従い、候補OID `659e555714c8ccf3a6ec7fd1cb0dbf05cfcc342d` と `96aa3878a344f479ee064118cb90002a770db627` のcommit実体と指定資料を別cloneで確認した。これらは結果資料のlocal版で、科学SOURCEや新しい実験ではない。両版の比較表は公開表とbyte一致した。軸表は96aa版が公開版とbyte一致し、659e版は追加provenance列を持つが公開列の全行literalは一致する。異なる版を同一bytesと記録せず、追加列・詳細監査・hash一覧・非公開補足を公開しない。candidate object全体の完全性、全祖先・backupの復元性まで再認定していない。

公開[起動記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/be2e5475f4dbb5a07e947b053e8c419109f1eda9/docs/research/track_a_h8_confirmation_parallel_resume01_20261019.md)のRUNNINGは2026-10-19の時点記録であり、[後続結果報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/docs/research/track_a_h8_new_pr_confirmation_results_20261019.md)の完了記述と区別する。元p8のidentity list/tuple受入STOP、元p7の所有run中断、固定datasetでのresumeを保存する。公開起動記録はp8の完全1軌道再利用、残47/48軌道・新190軸、最終192軸の計画を記録し、結果公開は完了と同じ2軸の再利用を報告する。起動時予算と実完了receiptを同一視せず、全192軸を修正SOURCEで新規compileしたとも、失敗・retryなしとも記述しない。旧STOP解除・再試行・grant再利用は今回行っていない。

## 主表・Mechanism・Figure 2B

[主表JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.json)のH8_DF28_P_1/2は、[Mechanism metadata](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1ff6ac0b6ffed6231dac5f655275b620598d0201/artifacts/resource_applicability/track_a_mechanism_results/2026-10-19/comparison_metadata.csv)のH8_CONFIRM_1/3とPR/PF/ε/nが一致する。cohort名INDEPENDENT_UNCONDITIONAL_CONFIRMATION_N48とINDEPENDENT_FIXED_CONFIRMATION_N48は同じ確認結果のaliasである。[Figure 2B保存値](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb33de67475e7d33eed56801b74a2a277147e4be/artifacts/resource_applicability/track_a_mechanism_figure2_a82_readiness/2026-10-19/figures/plot_values.csv)もこの2比較を参照し、主表に保存されたR_N/R_Ceff/R_Gとliteral一致する。別の独立追試として加算しない。2Aのlegacy/探索cohortや新H4 DF10/H6確認へ機構因子を転用しない。

## H4/H6との監査水準の比較

|項目|H4 tol-only DF10 P|H6 tol-only DF19確認|H8 tol-only DF28確認|
|---|---|---|---|
|固定確認結果|n1108/n78、粗PF低費用・NOT_REPRODUCED、細PR低費用|n48/n104、粗PF低費用方向を再現、細PR低費用|各n48、両方PR低い標本点費用を再現|
|公開終了監査|[completion audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/completion_audit.json)・native exit0・1186 worker回収・2372軸|[native completion](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cccfcb2051dacf29e91473e2e1152acef1574bd/artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/native_completion.json)・両driver exit0/reaped・152軌道/304軸、source preservation/ledger|結果表・SOURCE・公開報告のlocal完了記述。詳細receiptからの公開再検証は未達|
|seed/index監査|[seed refresh](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/seed_refresh_projection.json)はcutoff時点の明示除外70270、collision0、future_unregistered_campaigns_covered=false|公開SOURCE/input/registry識別と終了資料。今回新たな全履歴seed監査は行わない|公開SOURCEと起動記録の除外policy、scope/replica/axis/cohort/reuse列。actual registry・derived seedの公開監査は未達|
|今回の公開status|PUBLIC_COMPLETION_RECEIPTS_VERIFIEDを元資料の範囲で保持|同左|REPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTSを保持|
|保証しない事項|母平均優位・全support分散・未登録将来seed独立性|母平均優位・全履歴の数学的独立性|同左に加え、原profile bytes・旧compiler出力完全同一性・詳細PR終了の公開再認証|

[protocol_scope_status](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/protocol_scope_status.csv)の状態を更新・昇格していない。H4/H6の公開監査も作者の技術証拠であり、外部再現・immutable CI・別reviewerの合格を意味しない。系間でSCF/orbital/integral/DF/state生成規則が完全に同じとの認定も行わない。

固定候補と探索を分離した設計、探索n8/旧n32/希少pilotをpoolしない方針は公開資料で確認できる。seed不一致・namespace差だけで確率独立性を証明しない。公開SOURCE内の検査機能を全実行でのPASSの証拠に置き換えず、全履歴seedとの非重複・数学的独立性を推測でPASSにしない。

## 解消した資料探索と残存gap

追加科学計算なしで、候補OIDの実在と版差、完全SOURCEとdataset SOURCEの役割、固定候補/PF/ε、保存値参照、主表/Mechanism/Figure 2Bの同一cohort対応、既存公開回収索引を整理できた。原本の限定metadata照合と詳細な対応先は所有環境の非公開補足で保全した。private由来の新監査表・hash一覧・運用来歴は本公開成果へ含めない。

[Phase 3 G04](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/REPORT.md)の「限定点表の対応がある」と「詳細終了auditへ公開接続できない」は別の行である。前者の対応を補足できても後者はNOT_ACCESSIBLEを保持する。今回の結果は、詳細資料が存在しない・未実行だったという判定ではない。

|残存項目|性質|LOCALへ渡せる内容／必要判断|
|---|---|---|
|詳細PR completion・public/private raw対応・actual seed/index監査|資料の公開制約|本4成果物と固定公開索引。非公開補足は送付しない。ユーザーが必要なfield・相手・経路と既存公開拒否の扱いを具体的に判断する必要がある|
|原profile bytes・元compiled回路からのfingerprint再導出|原資料未取得／公開資料で確認不能|既存の未確認表示を維持。設定一致から出力完全同一を推定しない。新compileは解消手段として実行しない|
|全履歴seed独立性、母平均優位、未観測supportの費用・分散|科学的未保証|資料不足だけの問題と扱わず、今回認定しない。CI/p値null、未知費用を0補完しない|
|希少RTEと元確認監査の区別|研究scopeの区別|[正式終了方針](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md)を保持。S0/S2_SINGLE層別有限標本と残余S2_MULTI/S4_PRESENT条件付き感度解析の完了を、元n48実行監査のPASSへ代用しない|

## 成果物・完了境界

公開するのは本REPORT、[二条件CSV](H8_CONFIRMATION_EVIDENCE.csv)、[公開範囲](PUBLICATION_SCOPE.md)、[検査記録](VALIDATION.json)の4ファイルのみ。原SOURCE・input・raw・QPY・grant・ledger・marker・STOP・元worktreeを変更せず、既存checkerの方針を使ったstdlibの構造/literal照合をprimary Codexが実施した。科学runnerや集計関数をimport/起動していない。

**MANDATORY STOP。** 今回の公開・照合が終了した後のCodex自動作業はない。次の担当はユーザー：詳細監査をLOCALへ渡す必要がある場合にのみ公開/共有可能な最小範囲と経路を判断する。LOCAL Codex：別途指示を受けた場合に本成果をG04の補足として読み、詳細auditを公開PASSへ昇格しない。希少RTE追加sampling/compile、新H8探索、保存値再解析は開始しない。
