# O2データ辞書

全CSVセルはUTF-8文字列。値の数値再解釈・再丸めをしない。JSON有理数文字列とdecimal表示は同じ型へ強制変換しない。
原SOURCEの定義が優先し、列をそろえたことは同条件・同単位・同精度の認定ではない。

## 科学量・資源量

| 用語 | 定義・単位・注意 |
|---|---|
| N_cos / N_sin | 要求signal精度の元十分条件に基づく解析上の量子shot数。実機で測定した回数ではない |
| n | 古典的に価格を求めたtrajectory標本数。Nと別。複数精度に同じ価格を再利用してもnは増えない |
| C_cos / C_sin | 元compilerでの測定付きfull Hadamard wrapperの1shot価格。通常compiled RZ。native T/CX・logical数・物理runtimeと換算しない |
| PF誤差係数C | 価格Cとは別量。元delta窓のC_use等は経験的包絡で、全域の厳密上界ではない。別Hamiltonian/rankへ移送しない |
| G | 元保存 `N_cos*C_cos+N_sin*C_sin`。RZ×shot。有理数列G_exact等をそのまま転記。今回新しく評価しない |
| G_SE | 元のG価格標本のSE。多くは両軸paired会計。N/B/bias等のsystematicを含むとは限らず、母平均上界や厳密CIではない |
| L_D | DFは二体fragment prefixと別枠one-body、Pauliは元係数順非identity項prefix。DF p3とPauli p3は同じ保持部分でない。P3共通Gaussian保持には単純な整数prefixを強制しない |
| q | 元SOURCEのouter PF step数。T/qの新計算はしない。delta未保存ならNOT_DERIVED |
| r / R_outer | Pauli rは1outerのevent数。高次DFのR_outerはsigned pieceへの元ABS_TIME_PER_OUTER総配分。各occurrenceのrとは一般に別。元allocationを保持 |
| K | finite paired-Taylorの最大偶数event次数。元lawでは0,2,…,K、多項式次数K+1。PF orderと混同しない |
| PF order | S2/Y4/m5_best等の元公式名・次数。4th(m5_best)を独自の新数値へ変換しない |
| B / log_B | RTE正規化とその元保存log。補正signal・range/shot会計の役割を分ける。log_BからBを新計算しない。attenuationを新算出しない |
| bias_cos / bias_sin | 元referenceと補正signal等の差。signed/absolute、nominal/grid、数値allowanceを原列名・projection_mapで識別 |
| headroom | 元の精度からbias/u/rho等を控除した保存値。今回引き算しない。P3 engineering候補と実装後の真のheadroomは別 |
| epsilon_signal | 保存有限時間複素signalの精度。axis配分は元SOURCEのまま。epsilon_energy（Ha）やRPE成功保証と混合しない |
| alpha | 元のaxis/total失敗確率予算。通常axis .025等だが、共通値を未知の研究へ補わない |
| rho | native/compiled等の残差。仮想rho=0とactual rho=0の証明は別。P3 finite-bit診断だけではcompiled rho仮定を解除しない |
| P=0 | 初期状態準備費用を含めない。準備が実際に無費用であるという実測ではない |

## 証拠・欠測・共有

| 値・語 | 意味 |
|---|---|
| DIRECT | 元の保存実compiled価格。PRの有限標本平均を母平均保証としない。処理全体の実機実測でもない |
| MODEL / EXTRAPOLATED | モデル・外挿価格。DIRECTと別に保持し、未価格を0にしない |
| EXPLORATION / CONFIRMATION | 元cohort・選抜時点のラベル。別cohortをpoolしない。確認後min、固定nominee、復旧・同seed継続も区別 |
| CONDITIONAL | 指定モデル、数値allowance、rho等の仮定に条件付き。仮定成立の実装保証ではない |
| INCONCLUSIVE | 原比較の不確かさ込み診断が未確定。点費用が低くても確定改善としない |
| UNKNOWN / NOT_ACQUIRED | 元データの未確認・未取得状態。そのまま参照し、0や失敗へ変換しない |
| null | 元JSONのnullを文字列として保存。例：決定論PFのsample_SE_G。未知の母平均tail bound等も別のnull |
| 空文字 | 元CSVに存在する空セル。候補のbaseline診断空欄等を保存。ゼロでない |
| NOT_RECORDED | 転記対象元recordにそのfieldがない。別資料にも存在しないという主張ではない |
| NOT_PROJECTED | TABLE_REFERENCE行では値を複製せず元表へ進む |
| SEE_STUDY_AND_SOURCE / SEE_SOURCE | 当該レコードからは未転記。元study契約・field定義を読む。数字を推測補完しない |
| shared_price_group | 元study・candidate・cohort・結果版内の価格共有キー。複数precision行の価格標本再利用を示す補助で、seed独立性の新認定ではない |
| alias_of | 転記は同じ元表のsubset view。別観測・新実験ではない。TABLE_REFERENCEとの共存を二重計上しない |

## ファイルの列

`study_catalog.csv` のstudy_idは既存artifact/runnerのscope名、representation/molecule/geometry/decomposition/time_domain/precision_domainは条件の範囲。
複数系のstudyでも `condition_partition` のキーで分離する。
Hamiltonian_identity/state_identity/reference_identityとidentity_refsは保存bindingへの入口で、配列の再生成ではない。
method_family、compiler_contract、cost_scopeは各studyの意味を示す。compiler_contractは価格を作ったcompiler条件で、後日の会計readerのPython版ではない。
source_commitは記録済みの複数SOURCE役割やbinding locatorのJSON配列。空配列とsource_resolutionは未転記を明示し、結果SHAをSOURCEとして代入しない。
source_availabilityはそのIDのLOCAL_COMMIT_PRESENTとRECORDED_ID_ONLYを区別する。境界Rの完全SOURCE非公開という既存制約を維持し、公開summaryの記録と原本取得を混同しない。
result_commit/result_pathは主入口、result_refsは全関連ref_id、research_statusは元保存statusまたは正本への案内で、今回の採択ではない。
original_claim_componentは既存35行mapのID。study_id_basis、limitationsは由来と未保証事項。

`candidate_results.csv` のrecord_kind/source_row_unitが行の単位。TABLE_REFERENCEはcandidate値の行でなく、表／JSON領域への索引。
LITERAL_EXTRACTはcandidate×precision×cohort、またはcandidate/cohort aggregateの保存値。
candidate_idは元candidate ID、P3は元armラベルP/DF、表索引はSEE_ORIGINAL_RECORD_ID。
representation/method_familyとPF_order/L_D/q/r_R_outer/K/T/deltaは原意味を優先し、未転記はstudyへ戻る。
epsilon、eligibility、original_decision、cohort_id、price_kindは元判定／価格条件を識別する。
ref_id/result_commit/result_path/row_selectorが元recordを指定し、projection_mapは出力列から元列／JSON fieldへの対応。
JSON fieldに `{"json_pointer":"/scope/comparison_ids/PR"}` を使う場合はdocument全体からの絶対pointer。
N/C/G/SE/n/B/log_B/bias/headroomはprojection_mapにあるfieldだけをliteral照合する。任意の列間算術はしない。
coverage_note、SOURCE_refs、shared_price_group、source_identity_key、alias_ofは転記範囲・出所・二重計上防止用。

`comparison_catalog.csv` のcomparison_idは元比較名または元表/節の参照ラベルで、新しい比較結果のIDではない。
study_idは主入口、related_study_idsは複数研究にまたがる既存総括の参照。
record_origin/new_comparison_created/new_numeric_evaluationで保存比較の参照であることを示す。
candidate_or_arm_selectors、fixed_conditions、changed_parameters、data_refs、row_selectorが対照の定義。
saved_direction_or_decisionは元の結論の引用で、再ランキングでない。
cohort_and_sharing、resource_factors、observed_cost_or_signal、mechanism_interpretation、unseparated_factorsで観測・説明・未分離要因を分ける。

`df_pauli_matching.csv` はcomparison_ids/study_idsへの参照に、H/state/reference、time、precision、shot、deterministic/randomized部分、law、q/r/K、PF、control、compiler、metric、cohort、finite-bit/rhoの一致・相違を文字列で記録する。
comparison_classはMATCHED（指定目的で直接対応）、CONDITIONAL_MATCHED（仮定付き対応）、CONTEXTUAL（異条件の参考）、NOT_COMPARABLE（同じ比較へ統合不可）。
資料不足の行はcomparison_class空欄＋verification_state=UNCONFIRMED、unconfirmed_itemsへ不足を記載する。
classification_scope、comparison_purpose、changed_together、permitted_useがラベルの適用範囲。
causal_representation_onlyは全行NOT_ESTABLISHEDで、matchedラベル自体が表現単独の因果効果を保証することはない。

## 元データへのselector

CSVは `{"type":"csv_rows","where":{"candidate_id":"...","cohort":"confirmation"}}` の完全一致。where空なら元表全体への参照で、再出力・再集計はしない。
JSONは `{"type":"json_pointer","pointer":"/PR_summary"}`。空pointerはdocument全体。
text型は元reportの節・表へのlocator。巨大表・rawを再複製する命令ではない。
SOURCE_INDEXのcommit/path/blob/bytes/urlを先に確認し、別版の同名ファイルを使わない。
Phase2 canonical/series/reference/source索引も固定refとして保持し、全観測の再統合・新canonical採番は行わない。
