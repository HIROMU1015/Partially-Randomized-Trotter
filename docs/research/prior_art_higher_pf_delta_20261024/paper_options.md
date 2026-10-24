# 論文構成案A/B：未採択の比較資料

両案は同じ採用済み高次PR証拠を異なる問いへ配置する案である。ここでは優先順位や正式採否を決めない。
新規性を「高次＋PR」「次数で費用が変わる」「C×N」に置かず、具体的な比較結論とその成立範囲に置けるかを検討する。

## 案A：既存PR資源評価論文への組込み

**中心的な問いの候補**：合理的な決定論的対照、native compiled費用、独立費用確認を用いたとき、PRの資源上の利得はどの条件で残るか。
高次系列は、この問いを次数/q/partitionの自由度へ広げる重要な追加事例とする。

|予定section|目的|Main textに置く証拠候補|Supplementary/限定|
|---|---|---|---|
|1. Introduction/related work|既存PR・composite・DF・cost-aware評価との差分を限定|P01/P03/P08と「適用条件の検証」という問い|旧39文献索引、今回10文献の版・本文箇所|
|2. Fixed signal task and accounting|何の精度/資源を比較しているか明示|P=0、compiled RZ、保存bias/u/shots、cohort区分|solver/representation/compiler差、経験的allowance、energy/FTへ非外挿|
|3. Deterministic baselines and existing confirmations|PR有利を前提にせず対照依存性を示す|既存6条件主表、B0Iや共通PF588の採用結果を各scopeで引用|legacy DF12/new DF10、旧cohort/探索/独立、全PF最適ではないこと|
|4. Higher-order configuration study|今回の追加成果の中心配置|異なるqのFull/PR方向差、H4/H6独立標準Y4、同一qの否定的対照を一組として示す|H2探索のidentity制約、H4 .005の弱さ、5精度の全方向表|
|5. Why unit circuit cost is insufficient|具体的C/N/G寄与の説明|H6探索でC増でもN削減が上回る保存結果|旧探索因子を新独立群の機構確認へ移送しない。全操作の因果分解は未実施|
|6. Discussion and limitations|読者が使える限定と未確立事項|比較候補域、精度政策、qの交絡、samplingの限界|母平均/rare tail/厳密bias/実用選択則/FT/先行研究の完全最適化は未確立|

**既存主表・Figure2との関係**：固定1923fce8…の主表とFigure2A/Bは保持する。
高次標準Y4の新2比較を旧6条件主表の行へ差し替えたり、legacy/探索のFigure2へ同じcohortとして足したりしない。
案A採択後の編集案として、section4の追加表/図（仮称H）を別途設計できる。図なしでも保存結果表と対照の文章で構成可能で、新図は実証の前提ではない。

**主な改善**：旧S2中心の適用条件研究に、PF構成を変えるとFull/PRの費用方向も変わり、その解釈は同一q対照で制限される、という具体例が加わる。
独立標本確認を伴うため探索最小値だけへの依存も一部減らせる。ただし旧claim全体を遡及的に独立確認済みとしない。

**査読上の弱点**：多くの系列を一つの実験のように見せる危険、話題の拡散、既知アルゴリズムの再実装という批判がある。
対処は各図表にtask/DF/cohort/対照域を付けることと、追加結果が既存の中心的な問いへ答える役割を明示すること。
高次研究を入れるために、旧H4の非再現や対照でPR優位が消える結果を削らない。

## 案B：高次PF・PR資源効率を独立した論文テーマにする

**中心的な問いの候補**：固定量子化学signal taskで、PF次数・q・deterministic/random partition・目標精度の構成選択は、実装単価と必要shot数を通じて費用方向をどう変えるか。
表題候補は「Configuration-dependent resource trade-offs in higher-order partially randomized time evolution」のように構成依存性を明示する。
「order-induced reversal」「new optimal PF」「universal high-order advantage」は現証拠に合わない。

|予定section|中心となる内容|直接使える固定証拠|不足・過大主張の境界|
|---|---|---|---|
|1. Question and prior art|既知の高次PR・次数最適化を出発点にし、今回の追加問いを述べる|P01 App.A3/D1、P03 §5、P21/P25/P32/P33との比較|「高次化の初提案」「精度crossoverの初発見」としない|
|2. Fixed implementation and candidate configurations|S2/Y4、q、prefix、finite K、tail law、control/metricを定義|CONTRACTと探索元SOURCE。H2はq4/2、H4/H6はq8/4|qとpartitionを最適化済みの一般法ではなく登録固定構成。standard Y4とm5_best/P33新4次を区別|
|3. Precision-dependent cost directions|4方式の探索費用方向と精度依存性|PRIMARYの15方向/30比較|離散精度点間の境界、漸近性、H4 .005の頑健性は認定しない|
|4. Same-q controls|主比較の解釈を制限する否定的対照をMainに置く|SAME_Qのq4/q8・H4/H6・ε=.001|同一qでの逆転なし。qが唯一の原因とも次数の因果効果とも断定しない|
|5. Independent cost-direction confirmation|新Y4標本と既存独立S2で方向が残るか|ADOPTIONの2比較、4群、固定family/近似区間|回路費用標本の確認であり、signal/allowanceを新たに独立実証したことにはしない|
|6. Accounting explanation and scope|C/N/G、signed pieces、RTE配分、normalization/headroomを説明|保存CNGとSOURCE記述|会計恒等式は新理論ではない。操作別compiler因果や未測定条件の予測則は未確立|
|7. Discussion|新しい具体的知見と一般化に必要なものを区別|本差分Matrixと保存限界|全PF域・全分子・energy/FTへの一般化は未達|

**Main/Supplementary**：Mainは精度方向、同一q対照、独立2比較、C/N/Gの代表説明。
Supplementaryは全endpoint、候補/条件、元SOURCE・契約・終了/修正履歴、finite-RTE/shot政策、未観測strata、既存論文への参照。
旧H468六条件やgeometry全表は新独立論文の新成果として複製せず、背景または既存結果への参照に留める。

**理論説明の必要範囲**：高次signed stageの追加、tailへのABS_TIME配分、保存normalization/bias/headroomとshot規則、full-wrapperのCとの関係を既知理論に沿って説明する。
既知の会計とSOURCE構造により既存結果を記述するために、新しい定理は必須とは限らない。
一方、独立した「一般選択法」や「予測原理」を売りにするなら、P01/P03/P08/P21/P32を超える具体的criterionと未使用条件での支持が必要になる。
これは追加計算の既定要件でも実行指示でもなく、主張の選択に依存する不足である。

**重複の境界**：P33の決定論係数探索・新4次・energy評価を新規成果として再利用しない。
案Aと案Bを両方発表するなら、この高次PRデータと同一q/独立確認の主たる掲載先を決める必要がある。
同じ2比較を異なる論文で新独立証拠として数えない。旧PR論文側には参照と最小限の文脈を残す等の整理はユーザー判断とする。

## 案A/Bの比較

|観点|案A：既存PR論文に組込む|案B：独立した高次PR論文|
|---|---|---|
|中心課題の明確さ|PRの利得の成立条件という広い問い。高次結果の役割を短く接続する必要|次数/q/partition/精度に焦点を絞れる。ただし何を新しく答えるかの明示が必要|
|先行研究との差分|baseline・native費用・独立確認を含む複数の具体的適用条件|高次構成の具体的方向差・同一q否定対照・独立費用確認の組合せ|
|既存証拠で支持できる範囲|採用済み限定結果を重要な追加事例として引用できる|同じ限定的な実証結果を中心にしたcase studyは構成できる|
|独立した新規性|論文全体の証拠を補強する位置付け。高次部分だけの独創性に依存しにくい|一般原理は既知で、単独論文の追加的な説明力が査読の焦点になる|
|主な査読上の弱点|異なるcohort/DF/controlの混在、主題の過密化|既知手法の代入・狭い固定q条件・普遍選択則なしという批判|
|既存論文との重複|高次結果の主たる掲載先を本論文にまとめられる|旧PR論文/P33と結果・問い・図表・新規性の境界を切る必要|
|原稿変更規模|既存稿の追加sectionと関連研究/議論の調整。今回は未着手|新しい論旨/Methods/結果構成が必要。今回はoutlineのみ|
|新科学計算の必要性|限定的な追加事例の記述には新計算不要|限定case studyの記述には必ずしも不要。一般選択則等を主張する場合に限り不足を検討|
|論文全体の一貫性|合理的対照と費用政策への依存という共通問いで接続|構成依存性と同一q対照を一貫した中心に据えられる|
|説明力|「どのPR利得が残るか」の証拠が増える|「高次の利得をq/partitionと切り離して解釈できない」という限定を正面に出せる|

どちらにも先行研究との同一条件再実行比較、厳密母平均/tail、普遍的method-selectionの証拠はない。
独立Y4確認の完了は両案に同じだけ寄与し、案Bの独創性を自動的に確定するものではない。

第三案として、既存PR論文のSupplementaryに限定し、本文では構成依存の注意点と独立確認への参照を置く方法もある。
本文の主題を拡散させにくい反面、否定的対照を含む重要な高次知見の可視性が下がる。これも未採択。

## 図表候補と保存入力

|候補|主な入力|今回できる記述・不足|配置候補|
|---|---|---|---|
|精度別の方向マップ|PRIMARY/PRIMARY_COMPARISONS|15保存方向を離散点として示せる。境界補間・新rankingは不要/未承認|A追加図、B主図|
|異なるqと同一qの並列表示|PRIMARYとSAME_Q/SAME_Q_COMPARISONS|q/候補/cohortを明記。same-qはε=.001のH4/H6のみ|A高次section、B主図|
|独立2比較の差と近似区間|ORIGINAL/COMPARISONS|保存差/区間をそのまま使用可能。G比のCIはnullなので作らない|A追加表、B主表/図|
|探索と独立の4群費用|元結果のindependent_groups_4.csvとquoted_full_and_exploration_8.csv|mean/SE/nを別cohortで表示可能。poolingや新差SEは行わない|B主図、A補足|
|C比/N比/G比の分解|CNG_CSV、main_C_N_G_6.csv|既存分解fieldの範囲で使用可能。新独立群の分解・新寄与率は未算出|A説明補足、B説明図|
|一般的な選択則/連続相図|現保存入力だけでは不足|未使用条件の予測や境界推定には内容を固定した別数値解析/科学承認が必要|現時点で採用しない|

今回は描画・数値加工とも0。保存値だけの描画を行う場合も次の明示指示を待つ。
新しい比・CI・分解・反実仮想が必要になった場合は、その最小範囲だけ別契約で承認を得る。
