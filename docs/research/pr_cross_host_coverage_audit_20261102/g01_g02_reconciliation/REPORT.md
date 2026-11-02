# PR研究Coverage監査：G01/G02限定証拠照合 addendum

LOCAL Codex、2026-11-03T02:56:10+09:00。既存資料・metadataの読取と技術照合だけを行った。**新しい科学計算、保存値の科学的再解析・再ランキング、詳細科学レビュー、主要claim採択は行っていない。** Phase3等の原成果物を更新せず、独立した追補として保存する。

## 結果

**G01は7候補すべてMATCHED。** 元長時間研究の登録条件と未価格状態、スクリーニング、実行SOURCE、保存input binding、endpoint/N、探索・確認価格、拡張比較集合への接続を確認した。Phase3の `EVIDENCE_UNLOCATED` は当時の限定探索での状態であり、今回指定された固定実測資料から対応が解けた。Phase3の原表を上書きしない。

**G02は必要な価格cohort/indexの対応を具体化したが、全対象を無条件の独立性保証へ格上げしていない。** 100行の対応表は、候補・比較別98行とH8再引用alias2行である。88行は保存cohort/index・保存seed監査の限定範囲で `DOCUMENTED_DISJOINT`、10行は `RECORDED_SEPARATE`、2行は `REUSED_OR_OVERLAPPING`。これは独立cohort数でも独立実験数でもない。Phase2の99関係を99cohortとして認定していない。

- [G01_CANDIDATE_BINDINGS.csv](G01_CANDIDATE_BINDINGS.csv)：7候補の条件・原登録・SOURCE・input・endpoint・価格・比較・catalog対応。
- [G02_COHORT_BINDINGS.csv](G02_COHORT_BINDINGS.csv)：A08/A09/A11/A12と追加7候補の固定nominee、cohort、index、共有/再利用/回復、独立性の限定範囲。
- [validate_bindings.py](validate_bindings.py) と [VALIDATION.json](VALIDATION.json)：metadata/literal検査。科学的PASS・新しい統計保証ではない。

## 固定資料と読取境界

Phase3はcommit `235e56ee7d5cac4682775a62bc2a8262f15047dd` の [REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/REPORT.md) とGAP_CANDIDATES/COVERAGE_MATRIX/CLAIM_SCOPE/EVIDENCE_INDEX/EVIDENCE_GROUPSを読取。G01/G02の対象と未解決理由を保持した。
Phase2はcommit `604f0235840feb18d368f99ea0828f370ce70fc8` の全3catalog part・全2RECORD_LINKS partと、関係/SOURCE/参照/系列/未解決索引を再利用した。全groupの科学再監査はしていない。

G01の直接資料は、元結果 `155529e20a8fac39e00786c40b57830b71203c4f` の [長時間REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md)、スクリーニング `2e518b86e7c052dafb20dece0de7ffa6c7c85f8f` の [REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e518b86e7c052dafb20dece0de7ffa6c7c85f8f/analysis/h46_unpriced_screening_20261026/REPORT.md) / [選定表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e518b86e7c052dafb20dece0de7ffa6c7c85f8f/analysis/h46_unpriced_screening_20261026/promising_candidates.csv)、実測結果 `bb97a93d65992fdb53e642dde1584dd7d5b811d7` の [REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb97a93d65992fdb53e642dde1584dd7d5b811d7/analysis/h46_seven_cost_readiness_20261027/REPORT.md) / [PLAN](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb97a93d65992fdb53e642dde1584dd7d5b811d7/analysis/h46_seven_cost_readiness_20261027/PLAN.json) / [保存会計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb97a93d65992fdb53e642dde1584dd7d5b811d7/analysis/h46_seven_cost_readiness_20261027/results/candidate_costs.csv) / [execution_provenance](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb97a93d65992fdb53e642dde1584dd7d5b811d7/analysis/h46_seven_cost_readiness_20261027/results/execution_provenance.json) / [END_AUDIT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb97a93d65992fdb53e642dde1584dd7d5b811d7/analysis/h46_seven_cost_readiness_20261027/results/END_AUDIT.json)。

実測execution SOURCEは **`cab73373593ba7c113c1af018b1dc3077c8e78fa`**、provider SOURCEは **`c31ef424175c02379eafebe8fc6c7548c1d9aabe`**。元結果carrier、スクリーニング解析版、実測execution SOURCE、provider、結果公開commitを別の役割として記録し、一つのSOURCEへ正規化していない。`PLAN.json` はexecution SOURCEと結果commitで同一blob/bytes、保存provenanceのPLAN SHA-256とも一致する。

G02では上記に加え、[H4/H6/H8主表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.json)、[高次固定確認](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/fixed_evaluation/results.json)、[長時間nominee計画](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/confirmation_selection.json) / [seed監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/confirmation_seed_independence.json)、固定B3公開版 `ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed` の [nominee計画](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/confirmation_selection.json) / [completion](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/completion_provenance.json) / [provenance](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/scientific_provenance.json) と、直接リンクされた公開paired/index projectionを読取した。

実際のAGENTS.md、PROJECT_MAP入口、正式共通運用指示 `research_project_instructions_unified_2026-10-25.md` の§8.1/§8.4/§10を確認し、primary agentのみで実施。別agent・独立reviewerは使っていない。固定PROTOCOLのPhase0担当記述と実担当SERVER2との差も、前段のprovenance注記のまま保持する。

## G01：7物理候補の一意な接続

共通条件はlinear H4/H6、隣接1Å、STO-3G、tol-only DF1e-8、rank10/19、保存Hamiltonian/state/reference/scalar/fragment順、H6 weighted-Hermitian completion、P0、元finite-RTE/shot政策。symmetric-directional control、actual adjoint、cosine/+Im sineの測定付きfull Hadamard wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。候補別δ=T/qの保存値はCSVに保持する。

| Candidate ID | 系 | T | 保存PF名 | prefix/q/R_outer/K | 探索n / 確認n（保存値） | 照合 |
|---|---|---:|---|---|---|---|
| d748ee85d02193720b34 | H4 | 0.8 | 2nd | 3/1/4/6 | 427 / 13734 | MATCHED |
| 205f25b8cb63949b4b18 | H4 | 0.8 | 2nd | 3/1/8/2 | 1127 / 36295 | MATCHED |
| 53acfe65c49018ff2506 | H6 | 1.6 | 2nd | 5/4/4/2 | 49 / 390 | MATCHED |
| 6eb81b93d048deeeb63f | H4 | 0.8 | 4th(m5_best) | 2/1/32/4 | 38 / 1216 | MATCHED |
| 54510520e99d85af4690 | H4 | 1.6 | 2nd | 3/8/4/6 | 39 / 1247 | MATCHED |
| cf094355c78d50ae45f7 | H6 | 1.6 | 2nd | 5/4/8/2 | 143 / 1151 | MATCHED |
| 45f9fc0a53c1a37c7345 | H4 | 2.0 | 4th(m5_best) | 3/1/64/2 | 33 / 800 | MATCHED |

各候補は元 `candidate_precision_accounting.csv` の5精度行に登録され、価格状態はすべて `NOT_ACQUIRED`。したがって**旧642価格候補に含まれない**。元q32拡張の22価格候補とも別IDである。7候補を旧642やq32追加22の別名として扱っていない。

`PLAN.saved_endpoints` の適格行と、元表・探索/確認の保存N_cos/N_sinをliteral照合した。原報告の単位は **7物理候補、19適格precision endpoint、探索/確認を分けた38会計行**。同じ候補の価格は複数精度で共有し、独立標本数を増やさない。G/SE/単価は保存行への参照を記録し、新しい平均・G・SE・順位を算出していない。

H4入力は保存済み10/22再構築入力の「7科学配列hash一致、NPZ容器は異なる」という元記録を継承する。H6は元snapshot/receipt byte一致の記録を継承する。今回private NPZや科学配列を再読込・再hashした意味ではない。PLANのbindingが元scientific_provenanceへ一致することと、容器byteの同一性を区別した。

### 旧集合・拡張集合・確認の分離

[保存extended_point_summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb97a93d65992fdb53e642dde1584dd7d5b811d7/analysis/h46_seven_cost_readiness_20261027/results/extended_point_summary.json) は `original_saved_count=33`、`extended_exploration_point_count=33`、`total_cells=40`、`independent_confirmation_pooled=false`。旧33/40と追加後33/40は**異なる候補集合の保存結果**として併存する。数え直し・書換えをしていない。

原実測REPORTは探索価格追加で7セルのPR点最良が改善し、PF/PR方向は全40セルで不変と記録する。各候補が保存 `extended_cells.csv` の最良IDとして使われた行だけをCSVへリンクし、単なる追加候補と最良に選ばれた候補を区別した。確認価格は探索の最小値へpool/代入せず、[primary_comparisons](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb97a93d65992fdb53e642dde1584dd7d5b811d7/analysis/h46_seven_cost_readiness_20261027/results/primary_comparisons.csv) の固定参照比較として保持される。旧PRの探索参照と、利用できる保存独立参照の欄も別に残る。

Phase2では、7候補それぞれに元登録5精度のcanonical group/元recordへの対応がある。一方、この追加実測価格の固定fileに対応する行は固定Phase2 catalogに登録されていない。CSVで**旧登録条件へのリンク**と**新実測未登録**を分け、世界共通experiment IDや新canonical groupを勝手に作っていない。Phase3のG01/A11行には独立addendumとして接続する。

## G02：確認できた独立性の範囲

| 対象 | metadata対応 | 現時点で使える表現と制約 |
|---|---|---|
| A08 H4の2固定比較 | 公開paired projectionのcandidate/replicaと固定n、SOURCE契約、保存seed refreshのknown coverageが対応 | 保存除外監査の列挙範囲と固定cohort/indexに限定したDOCUMENTED_DISJOINT。cohortラベルだけでlegacyDF12と統合しない |
| A08 H6の2固定比較 | 確認replica0–47、0–103とcandidate/結果が対応。Q2 STOPと別dispatchのQ8を分離 | RECORDED_SEPARATE。公開初期designのnはnullで、final n seal/探索seed除外の閉じた対応を再構成していない |
| A08 H8の2固定比較 | 確認replica0–47、両軸pairedの公開projectionと固定候補が対応 | RECORDED_SEPARATE。designの事前freeze政策と実行最終seed監査を混同しない。元詳細audit制約を維持 |
| A09 高次の6保存群 | 118標本の公開sample IDと固定候補/旧shotを対応。新stage contractの別namespace・既知除外監査を記録 | RECORDED_SEPARATE。旧proposal02・全旧seed除外は未回収。新群相互のindex照合を全歴史的独立へ拡張しない |
| A11 長時間B2の39候補 | 固定nominee/n、探索と確認の全公開trial locator suffixをcandidate/cohort/indexへ対応 | 当該価格pairのDOCUMENTED_DISJOINT。確認59比較は39候補の価格共有を含み、独立cohort数ではない |
| A12 DF B3の40候補 | 固定64 nominee viewと40確認価格群、探索/確認のcandidate/cohort/indexが対応 | 当該価格pairのDOCUMENTED_DISJOINT。精度/公式viewを独立標本にしない。B2 baselineに確認価格がない比較は探索参照 |
| 追加7候補 | execution SOURCEと同じPLANで全7候補のn/2cohortを先に固定。全公開trial suffixのindex集合とnが対応 | DOCUMENTED_DISJOINTはこのrunの探索対確認に限定。保存seed-sequence digestはPLANとEND_AUDITで一致、seed再生成はしていない |
| H8主表↔Mechanism alias2件 | H8_CONFIRM_1/3は同じ保存確認への引用 | REUSED_OR_OVERLAPPING。追加の独立確認と数えない |

ここでDOCUMENTED_DISJOINTは、**列挙された保存標本ID/cohort/indexと元seed監査が支える範囲**である。実seedを新しく再構成して非衝突を証明したわけではない。既存PROTOCOLのindependence_status、元scientific/technical/acceptance/validity状態を変更・置換していない。

全系列でcosine/sineは同じtrajectoryのpairであり、2n独立標本ではない。複数精度や公式別viewは同じ価格群を再利用する。signal/reference/shotの再利用と、価格だけの別標本確認を分ける。確認後minと探索固定nomineeを混同せず、保存 `fixed_before_sampling` / `FIXED_BEFORE_INDEPENDENT_CONFIRMATION` / no-topup/no-poolingの記録を保持した。

### 回復・再利用を新しい独立標本にしない

長時間研究の中断後の完成trial再利用・同seed未完了index継続、B3の106/133完了記録や後続1691探索trial再利用・event-only再構成は、同じ固定cohort/indexの保全・継続である。元STOPを維持し、回復回数を独立確認数に加算していない。追加7候補のrepresentative試行やcompile cache hitも固定n内の処理で、追加標本ではない。

追加7候補のCOMPLETIONは全worker join、END_AUDITのcontroller_exit_codeはnull。失われたtool sessionの終了codeを0で補わない。ここで新しいworker起動・seed replay・QPY復元を行わず、保存された終了状態を引用した。

## 残る照合事項と研究記述への影響

1. **A08 H6**：原確認担当環境（現alias UNKNOWN）が、最終n固定plan/sealと探索→確認の匿名seed/index除外対応を提供すれば、公開design n=nullと最終実行の接続を強められる。必要なのはmetadataであり、再samplingではない。
2. **A08 H8**：原確認担当環境（現alias UNKNOWN）の最終seed registry/除外coverage/終了監査の公開可能なprojectionが必要。今回はG02に必要な公開cohort対応までで、G04等の別不足候補の調査へ広げていない。
3. **A09**：旧proposal02/旧全seed除外の未回収を維持する。元担当が対象探索cohortに限定した匿名coverageを示せれば分離をより厳密に記述できるが、全歴史的seed探索を完成条件にしない。
4. **A11/A12**：固定before-samplingの保存宣言とcandidate/nは確認したが、全candidateの正確なfreeze UTCを独立に復元したわけではない。必要なら元担当が保存plan hashと取得開始に結び付く公開metadataを提示する。現在の限定価格方向確認には、その時点限界を明記する。

指定表とのcandidate条件・保存N・SOURCE/PLAN対応に重要な不一致は見つからなかった。H6初期designのn=null、H8 designのPROPOSED状態は、後続実行結果との時点差として分離した。これらを独立性の完全な確認済みへ推定補完していない。

G01の原結果・実測対応は解消したため、「7候補の実測証拠が未発見」という制約を今後の記述へ持ち越す必要はない。G02の表現は、対象を明示した「保存cohort/indexと既知除外記録の範囲での別価格標本確認」または「記録された別cohortの確認」と使い分ける。**全候補最適性、母平均winner、全歴史的seed非重複、選抜後同時保証、実ρ全law保証、最終energy/RPE資源優位**は自動成立しない。G03以降の解消・主要claim正式採択・論文改訂は行っていない。

## 技術検査・再利用・保全

Phase2のmetadata_integration/integrate_catalog/verify_outputs、Phase3のcheck_coverageの固定参照・CSV/selector・literal照合を再利用した。追加7候補のprepare/execute/finish、長時間native、B3 seed/confirmation処理、A08契約を静的に確認した。元prepare/finish/verify処理はseed再構成や数値会計を含むため実行・importせず、metadata専用の小さな検査だけを作成した。科学kernel、NumPy/Qiskit等をimportしていない。

検査は候補ID/条件、PLANのSOURCE間byte一致、保存Nの逐語一致、19 endpoint/38会計行の構造、cohort/indexの一意性・範囲、固定public参照/blob、対象catalogと元record対応、公開情報、変更範囲を対象とする。G/SE/差/平均/費用順位/33/40の再計算は0。private seed値・入力配列・raw/event/QPYの再取得・再hashは0。結果はVALIDATION.jsonを参照。

全3catalog partをstreamingで条件選別し、対象98candidate IDに関連する868groupの検索リンクと元観測IDだけを再利用した。これらは実験/cohort数ではない。関連しないgroupの科学再監査は行わない。CPU affinity32、MemAvailable約52GiB、専用一時領域空き約221GiBを確認。独立資料/partのmetadata処理を並列化し、恣意的CPU/RAM cap、共有設定・他ユーザー/jobへの変更はない。

証拠GitではGIT_OPTIONAL_LOCKS=0/GIT_NO_LAZY_FETCH=1を用い、checkout/reset/merge/clean/gc/prune/fetch、branch/worktree/archive削除を行っていない。専用出力Git/indexだけに変更を作成。元科学SOURCE/input/state/reference/raw/result/grant/ledger/marker/one-shot/retry/STOP、Phase0–3固定成果物、他branch/worktree、整理保全資料は無変更。並行する別作業は非atomic観測として扱い、凍結・巻戻しをしない。

公開版に内部host/IP/絶対path/資格情報/private seed/非公開rawを含めない。公開済みcohort名、candidate ID、index範囲、固定Git URLだけを使い、元trial locatorの私有rootを除いた。LOCALからSERVER1/2へSSH・転送・直接操作をしていない。

## 公開・mandatory STOP

新branch `codex/pr-coverage-g01-g02-reconciliation-20261103` に、このaddendumだけを通常commit/pushする。remote SHAと全公開file本文を照合し、最終SHA・照合成否をチャットへ報告する。force push、main merge、既存branch変更はしない。

**保存・技術検査・公開照合後にmandatory STOP。** G03以降、追加科学計算、詳細科学レビュー、論文執筆・主要claim採択へ自動移行しない。

次の担当は**ユーザー**：この限定表現を採用するか、上記残るmetadataのどれを原担当へ別途依頼するか判断する。Codexの自動継続作業なし。
