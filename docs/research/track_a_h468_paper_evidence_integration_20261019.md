# H4/H6/H8論文証拠統合：固定独立確認の採用と表示案

2026-10-19。利用者の指示により、H4 tol-only DF10 Pの2結果を正式採用した。
ε=.05は **NOT_REPRODUCED・PF低費用**、ε=.0005は **REPRODUCED・PR低費用**。
この資料は既存結果の文書統合であり、新規科学計算・統計解析・raw費用再集計は行っていない。
最終evidence freezeと詳細科学レビューは開始せず、文書統合後mandatory STOPとする。

## 固定資料とGitHub照合

|資料|結果commit|科学SOURCE・役割|
|---|---|---|
|H4 tol-only D190|[406f7057](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/406f70570323366da185355550a3900388717fbb)|84db3909：登録190候補内の保存PF最良5精度|
|H4 tol-only A50|[85cb5b4d](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/85cb5b4d1808434165c421115ed09dab3b63dd00)|84db3909：保存FP n32と新規49候補n8を分離|
|H4 S_STATS|[d728d088](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/d728d088c010089c6b672eef0521261791b2d209)|8752489b：既存標本数計画。母平均・実powerの保証ではない|
|H4 DF10独立P|[821cc0f6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/821cc0f6ced57168293ea166644024867b064ffb)|3c664ca4：n1108/n78の別独立確認|
|H6独立確認／採用|[3cccfcb2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3cccfcb2051dacf29e91473e2e1152acef1574bd) / [877e7c28](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/877e7c2834609c75f585c0cf7561ae7dca5db8d1)|bf0c26da：n48/n104|
|H8独立確認|[e6eea538](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e6eea5387a6e84d1e39a9adb3b61975acc74e200)|745a6745：n48/n48。詳細監査の公開制限あり|
|Mechanism完成|[1ff6ac0b](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/1ff6ac0b6ffed6231dac5f655275b620598d0201)|既存32比較・64軸の説明完了。新H4/H6 Pへの転用不可|
|Figure 2原本|[eb33de67](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/eb33de67475e7d33eed56801b74a2a277147e4be)|render SOURCE 3de17acd。全34ファイル保全|
|共通プロトコルv1|[9036d8a8](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/9036d8a87ad1c38e83b9ff654b412dde7041e9d3)|ADOPTED_V1_DOCUMENT_ONLY。全系の科学gate合格を意味しない|

[GitHub照合receipt](../../artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/github_verification.json)に、依頼10commitと科学/render SOURCE9commit、実取得47資料、SOURCE87ファイルの照合を記録した。
認証済みGitHub connectorのcommit/tree/contents GETで実際に確認し、固定ローカルGit objectとGit blob SHAを照合した。
receiptのSHA256は、その同一blobのローカルbytesから取得したもの。H6の大規模paper manifestはリモートでは先頭2行とblob identityを確認し、全1572行は固定Git objectから読んだ。この取得範囲をreceiptに残した。
公開済み成果物のみを新worktreeへbyte同一で取り込み、科学コードのbranch全体はmergeしていない。

## 採用する点比較

[主表案](../../artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.md)は6比較。
候補ID、PF ID、平均のexact fraction、paired SE、保存方向、出典JSON pointer/CSV recordを
[JSON](../../artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.json)に保持する。削減率や差・比・CIを新しく計算しない。

|系・入力系列|ε|PR候補|L_D / δ|n|保存方向|低い点費用|
|---|---|---|---|---|---|---|
|H4 tol-only DF10|.05|B2_p3_q1_2nd_r4_K2|3 / .8|1108|NOT_REPRODUCED|PF|
|H4 tol-only DF10|.0005|B2_p2_q8_2nd_r2_K4|2 / .1|78|REPRODUCED|PR|
|H6 tol-only DF19|.05|B2_p4_q2_2nd_r4_K2|4 / .4|48|REPRODUCED|PF|
|H6 tol-only DF19|.0005|B2_p4_q8_2nd_r4_K2|4 / .1|104|REPRODUCED|PR|
|H8 tol-only DF28|.05|B2_p8_q2_2nd_r4_K4|8 / .4|48|POINT_REPRODUCED|PR|
|H8 tol-only DF28|.0005|B2_p7_q16_2nd_r4_K4|7 / .05|48|POINT_REPRODUCED|PR|

全行はlinear chain・隣接距離1 Å・STO-3G・tol-only 1e-8、T=.8、P=0、通常signed bias・allowance倍率1。
full cosine/sine Hadamard wrapper、symmetric_directional control、scalar original+tail一回、reference不回転。
compilerはQiskit1.3.0・rz/sx/x/cx・opt1・seed17・topologyなし。
費用は保存analytic shots × 実測compiled RZで、SEはpaired trajectory費用の保存標準誤差。
量子shot実行・雑音・FT T費用・最終energy/RPE総費用の測定ではない。
主表6比較の固定PFは直接実測。既存domain全体にはmodel値もあり、全domain実測最良とは主張しない。

H6粗精度はPF低費用の事前方向を再現しており、REPRODUCEDをPR優位と読み替えない。
H4粗精度の非再現は正式な採用結果であり、失敗扱いや追加samplingの理由にしない。
H8の2行は既存Mechanism32内のH8_CONFIRM_1/3と同一確認標本。
cohort名INDEPENDENT_UNCONDITIONAL_CONFIRMATION_N48 / INDEPENDENT_FIXED_CONFIRMATION_N48はこの固定結果のaliasとして結合し、追加の独立証拠とは数えない。

H4 legacy DF12 P（8a18b277、旧4185軌道・8370軸、主10＋診断4の方向再現）は別系列。
今回DF10 Pの「両方再現」と書く根拠にはならない。探索n8・旧FP n32・旧確認・新Pの標本は混合しない。
H4 D/A/S_STATSの値は固定結果の再利用であり、候補を選び直していない。

## 希少eventと不確かさ

H4 DF10 Pの保存trajectory strataは、粗精度S0_ONLY=1071/S2_SINGLE=37、細精度71/7。
いずれもS2_MULTI=0/S4_PRESENT=0。保存event次数0/2/4は粗4395/37/0、細1241/7/0。
K2のTaylor4はsupport外。K4で観測されないTaylor4や複数S2の費用・分散は未知であり0で補完しない。
H6は保存event次数0/2が粗374/10、細3327/1で、K2のsupport範囲と希少成分の不確かさを残す。
H8は公開報告のTaylor0が384/3072、Taylor2/4は両方0。trajectory strataは公開結果に保存されていないため導出しない。
全系で有限標本の点順位と保存SEのみを採用し、母平均優位・全support分散・実power・CI/p値を保証しない。

## プロトコルgateと技術監査の区分

[scope別状態](../../artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/protocol_scope_status.csv)を参照。
共通tol-only政策は入力生成手順の同一性を証明しない。H4保存状態のdense eigh・明示的な事後再正規化なしと、H6/H8 eigsh・正規化の違いを維持する。
SCF/orbital/integral/DF/state生成の全規則が同一とは確認していない。COMMON_CORE_READYへの自動昇格、系サイズによる因果・scalingの主張はしない。
既存Mechanism32のM_CORE/M_EXPL完了を新H4/H6 Pの4比較へ転用しない。

H4公開completion auditはnative exit0、1186 trajectory/2372 compile軸/5680 RTE event、全worker回収を記録する。
H6公開result/native completion/ledger/source preservationは152 trajectory/304軸/3712 eventの完了を記録する。
今回これらを再実行せず、保存監査receiptとSOURCEをGitHubで照合した。
H8は公開結果表とSOURCEを照合したが、詳細監査・hash一覧・運用来歴は公開報告にある従前の公開拒否により非公開。
したがってH8はREPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTSとする。詳細監査を今回公開したり、原compiler outputの同一性を認定したりしない。
全体をimmutable CI・外部再現・別担当の独立レビュー合格とも表現しない。

## 更新した成果物

[統合bundle](../../artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/README.md)に保存した。

- paper_manifest.json / paper_data_manifest.csv：旧1572行の全cellを保持し、H4 DF10 PのPR/PF計4行を追加して1576行。H6/H8の既存行を再追加しない。
- claim_evidence_matrix.csv：旧8行を保持し、H4 DF10/H8 Pの4主張、legacy H4 PとA82の2 scoped主張を追加して14行。current_overlay JSONで完成MechanismのCE1–8を結合する。
- evidence_ledger_v3.csv：旧256行を保持し、保存値とscope/完了状態/公開監査不足の出典を追加。新たな比較結果の計算はない。
- Figure 2更新案：原本と数値を保全し、表示・captionと将来必要な数値作業だけを分離する。

SOURCE、結果、旧grant/ledger/STOP等の取り込み175ファイルを保存SHA256で保全した。
新scriptはstdlibだけを使い、科学runnerをimportしない。検証対象は固定値の転記、cohort、rank、alias、出典hash、旧行の保存、原図の不変性。
実行方法は `python3 scripts/integrate_h468_paper_evidence.py --check`。
拒否テストは `python3 -m unittest discover -s tests -p test_h468_paper_evidence_structure.py -v`。
技術チェックはCodex自身が実施した構造検証であり、別担当による独立科学レビューではない。

## Figure 2と次の判断

[詳細更新案](../../artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/figure2_update_proposal.md)を参照。
推奨案は既存2Aをlegacy/探索のmechanism図として保持し、2BはH8 n48確認の分解を保持、独立P6主表を別に置く構成。
新H4 DF10/H6確認を機構分解の2Aへ置換するには、そのcohortの保存軸集計とPF対照を結合した別数値作業が必要。
未保存のH4比・削減率や新4比較の因子・log・誤差表示は今回計算していない。必要範囲を更新案に記録してSTOPする。

ユーザーが主表・原図の配置、入力手順差の例外表示、H8公開監査不足の扱いを判断する。
新しい機構分解や図の数値更新を望む場合、ユーザーが対象入力と数値処理範囲を別途承認し、その後Codexが実施する。
H2統合・最終evidence freeze・詳細科学レビューは今回の対象外。**mandatory STOP**。
