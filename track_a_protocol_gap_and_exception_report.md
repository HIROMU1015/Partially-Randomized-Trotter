# Track A共通プロトコル：差異・例外・最小追加検証 DRAFT

2026-10-19。DRAFT_NOT_ADOPTED。これは既存証拠の静的比較であり、現行契約の変更・新しい計算・正式な論文採否ではない。[プロトコル](docs/research/track_a_pr_resource_evaluation_protocol_v1_DRAFT.md)と[要件別CSV](track_a_protocol_compliance_matrix.csv)を併読する。根拠IDは専用evidence_catalogへ対応する。

## 1. 現時点の見取図

|系列|D|A|P|機構分解|共通条件への差分|
|---|---|---|---|---|---|
|H4 legacy DF12|登録104、費用71候補/保存全条件不適格33、登録内対照評価完了|新A16/n8・旧n32 anchorを別保存|9候補・主10/診断4比較が固定nで完了。主5εを含む|既存core/説明はPRE_D_P・旧subset16。新P cohortへ未接続|固定rank政策がtol-only案と異なる。D/P完了を取り消さずlegacy例外で扱う|
|H4 tol-only DF10|FPはq4の固定5構成。B0を含む強化探索Dではない|FPのPR1点/32本は主探索Aの代替ではない|新A後の事前固定独立Pは未確認|今回のcore32にない|F/FP完了。Codex Bの主比較準備はユーザー申告、branch/SOURCE/完了証拠未照合|
|H6 tol-only DF19|強化136。中間精度PF単価はmodelを含む|新A16/n8・旧union n32を分離して5ε会計|新Aの独立確認は準備中とのユーザー申告。完了証拠は指定資料にない|core/説明は保存32比較のH6対象で完了。将来の新Pは対象外|全入力生成規則の同一性未確認。非相殺の旧B0I12欠測はExtendedへ残す|
|H8 tol-only DF28|強化184にmodel残存。新Pの固定PF2点は直接実測|新A16/n8、旧union6/n32と別|事前固定2候補×n48、通常両端で点方向再現|coreは独立n48を含む。説明は全体63/64 side、A82_OLD_PR.PRだけ欠測|元profile file bytesと旧compiler出力の完全同一性未確認。新PのcontrolはSOURCEでdirectional確認|

上表の件数は元報告の転記であり再集計ではない。`D/P完了`と`共通入力/掲載scopeでCore全体充足`は異なる。現段階で共通tol-only main panel全系完成とは認定しない。

## 2. 入力政策の差異（EX01 / EX02）

EX01：H4 legacyは固定DF12である。H4 D/P（E_H4D/E_H4P）が完了しているので、その保存入力に限定した主結果を条件付きで掲載する案は可能。tol-onlyと書き換える、H4 Fのrank10結果へ費用・状態を読み替えることはできない。

EX02：H4 Fは新SCF・canonical orbitals・新DF・新sector状態を生成した別snapshot。E_F_SOURCEでOpenFermion `low_rank_two_body_decomposition(...truncation_threshold=1e-8)`、final_rankなし、signed weights・全returned fragments・generation順・cutoff0、weighted Hermitian projectionを確認した。H4のF/FP正常終了もE_H4F_AUDITで確認した。

一方、H6/H8の全SCF初期化/収束政策、MO順/位相・積分factor、library/runtime、projectionの次元依存engineering budget、sector/state生成/縮退処理までをH4 Fと同一とする照合は未完了。政策名やrank10/19/28から一致を認定しない。

**最小次作業は既存receipt/SOURCEの読取join**。整合manifestを回収して差分を明示する。これで解決するなら再SCF/再DFは不要。実際に相違があり共通化を採択する場合だけ、新入力系列とその系列に結合したD/A/Pを別承認で設計する。DF rankだけを変えた因果実験とは呼ばない。

## 3. H4の更新と機構表の対象（EX03）

論文準備E_PAPERと既存Mechanism E_CORE/E_EXPLは、H4 `old_plus_direct_subset16`、旧n32選択・推定anchor、および新A探索n8を対象としている。後続のE_H4Dは登録104のeligible集合を評価し、E_H4Pは新4185 trajectoryの独立cohortを完了している。E_H4Dに代表winner不変との報告があっても、別cohortの機構表まで自動更新済みとはしない。

既存Mechanismのcore32/32と後続説明63/64は、それぞれの固定scopeでの有効な完了証拠。旧core内に残る説明statusは履歴で、E_EXPLが別結果として後続改善を報告している。新D/Pの値を旧tableに上書きしない。

最小対応は掲載するcohort/対照版を先に選ぶこと。旧scopeの機構説明をそのまま掲載するなら計算追加なし。新Pを説明図へ使うと決めた場合のみ、既存保存入力のjoin後、必要な限定機構解析を別承認する。Pのsampling/compileを再実行する理由にはならない。

## 4. H4 tol-only主比較とH6独立P（EX04 / EX05）

EX04：H4 F/FPはDF政策代表panelとして完了しているが、FPは固定q4の5構成で、強化B0/B0I/full S2/S4とPRの登録探索全体を満たさない。PRのn32を独立確認済み最良候補へ改称しない。新しい主比較の準備はCodex B担当とのユーザー申告として記録し、完了SOURCEの照合前はPENDING。Codex Bの契約を本案で書き換えたり、同じ計算を別に起動したりしない。

EX05：H6は新Aと既存強化Dの通常5精度会計があるが、新Aの固定候補に対する独立P完了を指定資料から確認していない。Codex A準備中という情報はPENDINGとして分離する。旧union8/n32内の点選択も、高次PFのordinary別系列の確認も、その代替にはしない。

最小対応はA/Bから既存の準備SOURCE・固定対象・n設計・承認境界の資料を読取取得すること。重複実装/計算を避け、正式プロトコル採択後、必要な差分だけを担当側の新契約として扱う。今回の文書は担当への連絡・指示・条件変更を実行しない。

## 5. H8のcontrolとprofile（EX06）

E_H8P_SOURCEの`compile_one`は`control_policy='symmetric_directional'`を明示する。`ordinary_ancilla_branch`は低branch I/高branch Uという意味論のguardであり、ordinary loweringへの変更を意味しない。この点は共通control案との一致をSOURCEで確認した。

ただしE_H8P/E_A/E_AUDITは、元profile file bytes・旧compiler出力の完全同一性を未確認とする。compiler options/hash一致をその未確認の解決と混同しない。またPのexit/reap/最終byte照合は報告書に完了記載がある一方、詳細private witness/hash一覧は今回の固定公開projectionに含まれず、原本を再監査していない。

最小対応は既存原profile/receiptの回収・SOURCE binding照合。P2件を再samplingする前に既存証拠で解決できるか確認する。回収できなければ、明示例外の条件付き比較として掲載するか、そのidentityの強さを必要とする主張を保留する。全面再compileを自動要求しない。

## 6. Model・欠測・scope（EX07 / EX08）

EX07：H6/H8の強化PF集合にはmodel単価が残る。H8 Pの固定対照2点の直接実測やH6の一部直接測定を、全登録点の実測へ一般化しない。Core最低案として固定モデルを認めるか、主P対照には直接実測を必須にするかは採択待ち。モデル込み最良はモデル条件付きのまま表示する。

EX08：E_EXPLの既知欠測はH8 ε=.05の旧union側 `A82_OLD_PR.PR`（B2_p7_q2_2nd_r4_K2）のnormalization/signed_bias。これが未取得でもcoreの費用恒等式は成立済みであり、費用自体の欠測とは異なる。新A/n48のp8の説明欠測と誤記しない。

最小対応は、主図にこの旧PRの定量的機構説明が必要かを先に判断すること。必要なければ欠測を併記して既存63 sideで一区切りにできる。必要なら既存原signalの所在回収・metadata joinを先に行い、その後の数値説明検査のみ別承認。B=1/margin=0/bias推定で補完しない。

## 7. 独立確認の設計と非再現（EX09）

H4 Pは9候補に異なる固定nを設計し、主5εの10比較と診断4比較を完了した。H8は各n48・両端2比較を固定して完了。どちらも元の契約に対してCOMPLETEだが、共通の最低n・検出力・CI保証を採択したわけではない。H8固定designのn48/無pooling/早期成功停止なしを確認したが、今回その分散・rare費用に基づく必要nを計算・再評価していない。

H8方式を共通化する場合、手順だけを継承し各候補の標本設計を事前に行う。既存契約のCI/p値nullや母平均未確定を変えない。非再現、tie、PR不利、比較不適格は有効な科学結果であり、同じ対象で有利になるまで再計算しない。技術的中断の修正と科学的非再現を区別する。

## 8. Extended / Guarantee / m5の扱い

H6非相殺の旧B0I12分解欠測、H6/H8の新A倍率10保存shot欠測は、主案の通常signed/倍率1とは別。必要な頑健性主張にだけ追加検証を提案し、主Core不足を隠す理由にも、無関係なCoreを止める理由にもしない。

希少event費用上界、母平均優位、厳密bias、非Clifford/FT指標、RPE全体、全距離/時間/状態の頑健性は未保証。大nや観測0を根拠にGuaranteeへ昇格しない。条件付き希少pilotは無条件期待費用の完成ではない。

m5_bestのpilotは本チャットで保留された別契約。今回のDRAFTで再開せず、H4 legacy入力の既存pilot契約をtol-onlyに変更しない。新PFを主比較へ採用する場合の版・数値次数証拠・native費用系列・対照集合との整合は採択判断AD03として残す。

## 9. 計算を追加せず一区切りにする案

現存するlegacy/固定集合/モデル条件付き/有限標本の主張へ限定し、例外ID・未確認・Pの有無・説明欠測を表示する。共通tol-only全系比較や母平均保証を主張しなければ、既存成果を保存したまま研究を一区切りにする判断ができる。採択待ち項目を勝手に緩和してCOMMON_CORE_READYを作らない。

次の担当は**ユーザー**：共通tol-only主比較を目標に追加差分を行うか、legacyを含む条件付き主比較へ限定するか、必要な例外・検証階層を選ぶ。今回のCodex作業はlocal commit後mandatory STOP、push/新科学計算0。
