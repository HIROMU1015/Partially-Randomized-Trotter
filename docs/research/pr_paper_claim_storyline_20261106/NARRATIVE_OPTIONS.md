# S1/S2/S3：中心課題Aから作る論文構成候補

2026-11-06 JST。中心課題Aに沿う研究構成メモ。既存証拠の引用・対応付けのみ。新しい数値解析、詳細科学レビュー、原稿本文・Abstract・題名の執筆、Main Claim/構成の正式採択は行わない。

## 0. 編集判断の基準と暫定優先

このメモの中心は、方式を採用してから成果を並べることではなく、Aのどの問いにどの比較が答えるかである。Aに最も直接的な証拠はDFの強PF envelope・精度/時間・負結果。PauliにもH4の直接対決定論証拠があり、Hybrid対純RTEだけの研究として扱わない。[OLDMAP][COMPS]

**暫定的な編集優先はS1、次いでS2。S3はAへの直接比較を主に置けることと増分性の説明が条件。** これは資料の直接性・説明負担による編集提案であり、科学的な優劣、新規性の正式認定、正式採択ではない。S2は「二表現を同一要因として比較する」案や、未検証の統一G機構を要求する案Xとは異なる。ユーザーがAの幅と、純RTE端点をPR familyに含める記述範囲を決める。

各案は、本文を執筆する前の節の役割メモである。見出しは作業用で、論文の題名/Abstract/完成本文ではない。追加の数値比較表・図を今回作っていない。

## 0.1. 指定10軸での編集上の比較

- **中心課題との一致**：S1は多時間・多精度の強PF対照へ直接対応。S2はPauli P03を直接結果に置く場合に対応。S3はH4対決定論を核にし、Hybrid/pure比較だけにしない。
- **科学的新規性**：S1は強対照下のcompiled条件境界、S2は各実証差分を共通問いで扱う意義、S3はPauli固定実装・局所機構が候補。どれもPR/二乗則自体は新規でない。
- **科学的重要性**：S1は「いつ使う意味があるか」、S2は「保持の価値をどこまで一般化できるか」、S3は「Pauliで非空保持を入れる意味は何か」を明確にできる。単なる件数・大削減率の列挙を避ける。
- **証拠の強さ**：S1は強PF・長時間・別価格確認・負結果が同一系列にある。S2はそれにPauli H4の直接結果と最新内部境界がある。S3のAへの直接比較の主範囲はH4で、機構情報は別の局所証拠。
- **正負の結果**：S1はH2/粗精度/B3、S2は両系列の保持境界、S3はpure端点がHybridを上回る結果を積極的に置く。勝つ結果だけを選ばない。
- **論理的一貫性**：S1は最短の一系列。S2は二つのMethodsを一つの問いに接続する編集負担がある。S3は機構Bが主問Aを置き換えない構成が必要。
- **比較の妥当性**：S1も異なるDF rank/stateをpoolしない。S2は表現のみの因果を言わない。S3は旧primary/最新min/pure端点とHybridを区別する。
- **図表**：S1はV01/V02/V03を核、S2はその核と旧Pauli主表/V05/V07、S3は旧主表/V05と必要な機構図。残りはSI/不採用でよい。
- **追加作業**：三案とも限定的な構成整理は新科学計算なしで可能。強い統計・因果・時間/次数一般化を選ぶ場合だけ必要証拠を別判断する。
- **想定される弱点**：S1はincremental noveltyとDF CX点比較、S2は条件差・紙幅・統合の科学的意義、S3は先行重複・狭いT/orderと一部結果がpure端点であること。

## 1. S1：DFを主な実証にする

**中心問い/Main Claim候補**：固定DFの有限時間signal課題で、保存された強いPF envelopeに対するPR資源の競争力は、精度・時間・保持・次数・価格指標に依存する。正負の条件を同じ比較規則で示す。A10/A11/A12を主軸とし、A14とB1を補強、A09/B2-aを補助にする。[SYN][REAN]

**評価軸**：Aへの一致は最も直接的。強baselineと負結果が一つの論旨に入り、論理も明確。新規性は新原理でなくfinite-K・強対照・測定付きcompiled価格と条件境界という実証差分候補である。先行の予想を定量確認した面が大きく、証拠の強さだけで重要性・出版価値が確定するものではない。異なるDF rank/state・取得集合を分離できることが比較妥当性の条件。

**図表/追加作業/弱点**：最小核はV01/V02/V03。V04は補強、V08/V09とPauli/P3はSI候補。既存RZ図のままでも限定結果を説明できる。B1は保存結果の文章参照とし、将来のCX追加描画は別指示。現在の構成整理に新計算は不要。単一compiler、有限候補、希少order、DF CX SE未取得、incremental noveltyが弱点。

### S1-1. 問い・task・資源単位

- **目的**：Aを有限時間signal資源の問いへ操作化する。
- **問い**：Q1/Q2/Q5
- **方法・方式**：Full/B0/B0I/B2/B3とN,n,C,Gを定義。
- **直接証拠**：A01/A02/A03/F01、元Methods/data dictionary。
- **図表候補**：V07は定義文脈としてSI任意。
- **先行関係**：PR/DF適用・shot会計は既知。
- **限定**：energy/RPE/FT/runtimeへ外挿しない。
- **次節への接続**：どの決定論対照を入れると問いに答えられるかへ。

### S1-2. 強い決定論対照の必要性

- **目的**：分母によって利益評価が変わることを示す。
- **問い**：Q1/Q3
- **方法・方式**：B0+I、高次、共通PF取得域を保持。
- **直接証拠**：A06/A07/A10;H246_common588、H6_p7_measured。
- **図表候補**：V01本文。旧B0I/強化表は必要部分をSIへ参照。
- **先行関係**：truncation/保持範囲探索は先行あり。
- **限定**：MODELをDIRECTへ格上げせず、全決定論法の最強性としない。
- **次節への接続**：同じ規則で精度/時間に沿う主要結果へ。

### S1-3. 精度・時間と確認

- **目的**：利益がある条件とない条件を対応付ける。
- **問い**：Q1/Q2
- **方法・方式**：元nominee・候補域・独立価格cohortを維持。
- **直接証拠**：A11.BASEとA08;元40セル、別価格方向確認。A14.H8_Cは別条件補強。
- **図表候補**：V02本文、V04は本文任意/またはSI。
- **先行関係**：小分子の具体的compiled実証が差分候補。
- **限定**：33/40は成功確率でない。旧642/追加7/境界を合算せず、H8をscaling点にしない。
- **次節への接続**：有利条件の裏側としてrandomization範囲の負結果へ。

### S1-4. B3と不利な条件

- **目的**：負結果を中心Aの境界として使う。
- **問い**：Q3/Q4
- **方法・方式**：one-body保持B3をB2とPF envelopeに比較。
- **直接証拠**：A12.BASE、B3_fixed_negative、B1のFull PF限定例外。
- **図表候補**：V03本文。
- **先行関係**：全randomizationの一般原理は既知;有限域の具体的負結果。
- **限定**：全RTE不可能性でもPauli純RTEの否定でもない。RZ/CXとbaselineを指定。
- **次節への接続**：どの要因が関係するかを説明できる範囲へ。

### S1-5. 次数・splitと会計上の競合

- **目的**：観測の理解に必要な補助情報を置く。
- **問い**：Q4
- **方法・方式**：元C/N/G分解、matched条件、R/K、finite-K定義。
- **直接証拠**：A09/A11、C_N_G_decomposition.csv、既存B2-a再現結果。
- **図表候補**：V08/V09とB2-a保存表はSI;短い本文参照。
- **先行関係**：Günther A.3とcomposite先行を明記。
- **限定**：normalizationだけでG原因を確定しない。B2-b未実行、普遍selector未実証。
- **次節への接続**：近似以外のmetric/implementation境界へ。

### S1-6. metric感度・外挿限界

- **目的**：中心claimをどこまで拡張できるかを区切る。
- **問い**：Q5/Q6
- **方法・方式**：B1固定N/CX、Pauli内部/P3は別scope、RPEは条件付き接続。
- **直接証拠**：B1保存結果、P05/F01/R02。
- **図表候補**：V05/V06/V07はDiscussion参照またはSI。
- **先行関係**：資源指標・表現別の先行と区別。
- **限定**：DF CXは点比較、Pauli→DF因果・全energy優位を言わない。
- **次節への接続**：結語では保存scopeのAへの回答だけを整理する。

## 2. S2：DF＋Pauliを共通の問いで扱う

**中心問い/Main Claim候補**：同じ有限時間signal資源の問いを、各保存表現/実装内の強い決定論対照とrandomization端点に対して調べると、保持の価値と利益の適用域は条件付きである。DFとPauliの順位を直接競わせるclaimは置かない。A10/A11/A12にP03とP01/P04を加える。

**評価軸**：Aへの一致は、Pauliの対決定論比較P03を主に置く場合に成立する。V05のHybrid/RTE比較だけを追加すると問いBの比重が増し、Aへの回答が薄くなる。正負の結果を一貫して扱え、非空保持が常に有利とは限らないことを具体化できる。一方、両表現を同時に出すことで新たな単独因果効果が得られるわけではない。新規性は各実証差分の合成であり、共通設計則を仮定しない。

**図表/追加作業/弱点**：V07を比較ルール、V01/V02をDF、既存H4 primary表とV05をPauli、V03をDF境界として配置。V06/P1/P2は原則SI。各々のMethods・候補・選抜・ρ仮定を説明する紙幅が必要。追加計算はこの限定構成には不要。完全matchedな表現比較が必要な論旨なら現証拠では成立せず、そのclaimを採らないことが代替となる。

### S2-1. 共通の問いと比較の規則

- **目的**：同じ問いと異なる実装条件を最初に示す。
- **問い**：Q1/Q6
- **方法・方式**：共通G定義とrepresentation別split/law/control定義。
- **直接証拠**：study_catalog、df_pauli_matching、A02/F01/P05。
- **図表候補**：V07本文候補。
- **先行関係**：一般partial/composite・測定価格は既知。
- **限定**：同一分子名やL_D=0をmatchedとしない。
- **次節への接続**：表現ごとの内部比較のMethodsへ。

### S2-2. 二つの内部比較のMethods

- **目的**：混合せず再現条件を示す。
- **問い**：Q1/Q2/Q5
- **方法・方式**：DF envelope、Pauli106決定論域/141PR域、最新内部比較を別に定義。
- **直接証拠**：A10/A11、P03/P01、固定SOURCE/cohort資料。
- **図表候補**：既存条件表を参照;新しい数値比較表は不要。
- **先行関係**：先行query/STAR/compiled Gの目的関数を区別。
- **限定**：Pauli primaryと最新確認後min、DF旧/拡張域を統合しない。
- **次節への接続**：最初にDFでAへ答える。

### S2-3. DFの利益と境界

- **目的**：強baselineの下で正負を示す。
- **問い**：Q1/Q2/Q3
- **方法・方式**：DF元精度/時間系列、B2とB3。
- **直接証拠**：A10/A11/A12とA14補強。
- **図表候補**：V01/V02/V03本文;V04任意。
- **先行関係**：既知理論の上でのcompiled条件境界。
- **限定**：全域最適・母平均・サイズ単調則でない。
- **次節への接続**：別条件で同じ問いをPauliに尋ねる。

### S2-4. Pauliの対決定論利益と保持端点

- **目的**：Aへの直接証拠とHybridの限界を分ける。
- **問い**：Q1/Q2/Q3
- **方法・方式**：旧固定H4 primary、最新Hybrid/RTE比較を順に扱う。
- **直接証拠**：P03;次にP01/P02/P04。
- **図表候補**：既存H4 primary表への参照＋V05本文。
- **先行関係**：PR/分割/資源則の重複を先行へ帰す。
- **限定**：大削減の一部はpure端点。T.8/S2/仮想ρ0・非対称探索。
- **次節への接続**：共通する資源感度と、共通化できない部分へ。

### S2-5. 費用構造とmetric依存

- **目的**：共有できる解釈の水準を定める。
- **問い**：Q4/Q5
- **方法・方式**：B1/CX、保存N/C/G、P1/P2、B2-aを既知枠組みとして参照。
- **直接証拠**：元B1/B2-a、CNG、P1_bias/P2_parts。
- **図表候補**：V08/V10/V11はSI;本文は保存結論の短い参照。
- **先行関係**：Günther A.3を明記し、新しい二乗設計理論にしない。
- **限定**：全G因果説明やmetric別winner再選択は未実施。
- **次節への接続**：統合の意義と交絡をDiscussionへ。

### S2-6. 横断解釈と限界

- **目的**：共通問いで得た知見と未分離を同時に置く。
- **問い**：Q6
- **方法・方式**：P3共通保持、D2 control対照、unmatched系列を区別。
- **直接証拠**：P05/F01とmatching台帳。
- **図表候補**：V06 SI、V07再参照。
- **先行関係**：表現差と実装差を区別する。
- **限定**：直接DF/Pauli最適winner・表現のみの因果主張は置かない。
- **次節への接続**：結語は各系列内の回答と、その共通性の限界に限る。

## 3. S3：Pauliを主な実証にする

**中心問い/Main Claim候補**：保存Pauli-JW実装で、非空保持Hybridは強い決定論familyおよびpure RTE端点に対して、どこで資源を節約するか。H4 P03の直接比較を核に、P01/P02/P04を保持の境界、P1/P2を限定機構として使う。pure RTEの大きな利益を、そのままHybridの利益と呼ばない。

**評価軸**：Aへの直接性はH4固定対決定論比較にある。最新15条件は主にHybrid対pure RTEの問いであり、強いPF envelopeに対するCX頑健性の代用品ではない。局所的機構材料は豊富だが、新原理・定量的全G因果・全time/order則は未成立。高い削減率と高い新規性を同一視しない。Pauliを主要対象とする編集上の選択肢として成立するが、現時点のAへの広い回答はDFより狭い。

**図表/追加作業/弱点**：既存H4 primary表＋V05を核とし、P1/P2は必要な一方だけ本文、残りSIでもよい。V07は定義、V06は条件付き補足、DF V01/V03はDiscussionまたはSI。追加計算なしで限定構成を作れる。T=.8/S2・確認後選抜/共有価格・仮想ρ0・先行重複が弱点。これを理由にPauli時間/高次実験を自動必須化しない。独立Main Claimの増分性はユーザー判断のまま残す。

### S3-1. Pauliで保持を入れる価値という問い

- **目的**：AとHybrid/pure端点の区別を示す。
- **問い**：Q1/Q3
- **方法・方式**：Full/B0/B0I/Hybrid/pure RTE、中央controlの定義。
- **直接証拠**：Pauli study/比較台帳、P03/P01。
- **図表候補**：V07本文候補。
- **先行関係**：PR/Pauli適用・測定価格の先行を明記。
- **限定**：energy/FT/全表現の最適性を言わない。
- **次節への接続**：公平な保存比較のMethodsへ。

### S3-2. 条件・選抜・価格会計

- **目的**：主結果の母集団と選抜政策を明らかにする。
- **問い**：Q1/Q2/Q5
- **方法・方式**：H2/H4/H6のH/state/reference、T.8/S2、N/n、rho0、cohortを別に保持。
- **直接証拠**：P02/P03/P04と元SOURCE入力参照。
- **図表候補**：既存候補/条件表の参照。
- **先行関係**：先行のquery/STAR費用をcompiled RZと区別。
- **限定**：旧primaryと確認後min、逐次延長を同一の保証にしない。
- **次節への接続**：Aへ最も直接的なH4対決定論結果へ。

### S3-3. 保存決定論対照への利益

- **目的**：Aのpositive/negativeを直接支える比較を核にする。
- **問い**：Q1/Q2
- **方法・方式**：固定H4 primary対保存106構成のB0/B0I/Full。
- **直接証拠**：P03.BASE / Pauli_H4_PR_vs_deterministic。
- **図表候補**：既存H4主表を本文参照。
- **先行関係**：大きな利益だけを新規性にしない。
- **限定**：粗精度のHybridと厳精度pure端点を区別。最新15比較と選抜が違う。
- **次節への接続**：なぜ非空保持を常に選ばないかへ。

### S3-4. Hybrid対pure RTEと指標感度

- **目的**：保持の利益が一律でないことを示す。
- **問い**：Q2/Q3/Q5
- **方法・方式**：最新H2/H4/H6内部比較、B1固定N/CX。
- **直接証拠**：P01/P02/P04と既存B1。
- **図表候補**：V05本文;CXは将来の追加表示案のみ。
- **先行関係**：既知randomization範囲探索の具体例。
- **限定**：5精度を独立実験にせず、B1から強PFへのCX優位を推定しない。
- **次節への接続**：N/C競合の局所的説明へ。

### S3-5. N/C競合とP1/P2

- **目的**：保持を入れる負担の観測を説明する。
- **問い**：Q4
- **方法・方式**：共通outer split診断、Full/R/E/Empty価格の操作的対照。
- **直接証拠**：P05/F01、P1_bias/P2_parts。
- **図表候補**：V10またはV11を補助本文候補、残りSI。
- **先行関係**：構造既知部分と固定programの実証差分を区別。
- **限定**：UNRESOLVEDや単一indexを母平均因果寄与にしない。
- **次節への接続**：他実装へ移ると何が変わるかをDiscussionへ。

### S3-6. 適用範囲とDFの文脈

- **目的**：Pauli結果の外への一般化を制限する。
- **問い**：Q6
- **方法・方式**：P3/D2とDF B3・strong PFの別条件を提示。
- **直接証拠**：P05/F01、A10/A12を文脈だけに使う。
- **図表候補**：V06 SI、V01/V03 SIまたはDiscussion参照。
- **先行関係**：先行とincrementalな実装差分を区別。
- **限定**：DF/Pauli一般優位、T/order外挿、selector、新理論を主張しない。
- **次節への接続**：結語は固定PauliのAへの回答まで。

## 4. 案を選ぶ判断単位

- **Aへの最短で直接的な回答を優先**：S1。主要結果のstrong baseline・精度/時間・負結果が同一系列にそろう。Pauliを消す理由ではなく、主論旨に必要な証拠の直接性による。
- **異なる実装で保持の価値を問い直す意義を重視**：S2。Pauli P03をAの直接証拠として置き、P01を端点の境界として置く説明を引き受ける。因果的なDF/Pauli比較を求めない。
- **Pauli内部の方式境界と局所機構を主問にしたい**：S3。Aの強対照比較が主となる範囲と、Bの機構情報が補助となる範囲を選び直す必要がある。新規性が不足すると判断される場合も、無理に新理論を追加しない。

追加S4は設けない。今回の既存証拠から、Aへの直接性と一貫性で上記三案を越える独立した構成を仮定する必要はない。小さなresource noteへの分量縮小はS1/S3の編集幅であり、新中心claimではない。

いずれも、新計算を実行しなくても既存証拠の範囲で節構成を確定する判断材料は作れる。必要な次作業は、ユーザーによるclaimの強さ・主対象・Pauli/pure端点の扱い・図表の本文/SI配置の選択、その後の別指示による詳細outline/執筆である。科学的不足を解消したとみなすことや、S1/S2/S3の正式採択は行わない。Mandatory STOP。

[SYN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd/docs/research/pr_research_paper_evidence_synthesis_20261103/RESEARCH_SYNTHESIS.md>
[OLDMAP]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd/docs/research/pr_research_paper_evidence_synthesis_20261103/CLAIM_EVIDENCE_MAP.csv>
[STATUS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd/docs/research/pr_research_paper_evidence_synthesis_20261103/STATUS_AND_OPEN_QUESTIONS.md>
[CAT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2779710a86d380bb484accc51d3dccba183370b8/docs/research/o2_data_catalog_20261105/README.md>
[COMPS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2779710a86d380bb484accc51d3dccba183370b8/docs/research/o2_data_catalog_20261105/comparison_catalog.csv>
[MATCH]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2779710a86d380bb484accc51d3dccba183370b8/docs/research/o2_data_catalog_20261105/df_pauli_matching.csv>
[FIG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/README.md>
[FM]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/FIGURE_MANIFEST.csv>
[DISPLAY]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/DISPLAY_DATA.csv>
[PRIOR]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/788ebad04e620055a80c75b1b3bef4a15712d4c2/docs/research/o2_claude_review_followup_20261106/PRIOR_ART_MATRIX.md>
[TRACE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/788ebad04e620055a80c75b1b3bef4a15712d4c2/docs/research/o2_claude_review_followup_20261106/SOURCE_FORMULA_TRACE.md>
[REAN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/29c3bf0ecc687d458df55cf967ff38f990130b81/docs/research/o2_b1_b2a_saved_reanalysis_20261106/REPORT.md>
[B1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/29c3bf0ecc687d458df55cf967ff38f990130b81/docs/research/o2_b1_b2a_saved_reanalysis_20261106/B1_CX_SENSITIVITY.csv>
[B2A]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/29c3bf0ecc687d458df55cf967ff38f990130b81/docs/research/o2_b1_b2a_saved_reanalysis_20261106/B2A_NORMALIZATION_RESULTS.csv>
[FIXED]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/29c3bf0ecc687d458df55cf967ff38f990130b81/docs/research/o2_b1_b2a_saved_reanalysis_20261106/B1_FIXED_COMPARISONS.json>
[SRC_LONG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md>
[SRC_PF588]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_pf_common_final_20261022.md>
[SRC_B3]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md>
[PAULISYNTH]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md>
[PAULI_ADDENDUM]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f6eb34dad982f8f753a618c609b528fb3c20d142/docs/research/pauli_h2_h4_h6_paper_evidence_addendum_20261102.md>
[PRI_PA-E3]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/artifacts/non_df_h4_pauli_pr_search/2026-10-25/best_by_precision.csv>
[H6_P1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bfb4ca75437a9b3605a8f57b5f82100615dd034f/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/bias/REPORT.md>
[H6_P2]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9073274daee69c9b69d8db3dc332bf6132c2259/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost/REPORT.md>
[H6_P1_BINDINGS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bfb4ca75437a9b3605a8f57b5f82100615dd034f/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/bias/source_and_inputs.json>
[H6_P2_BINDINGS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9073274daee69c9b69d8db3dc332bf6132c2259/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost/source_and_inputs.json>
[P3_STAGE4]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a1d3f32aba6416fd1e200cd66a97d1a825b5dd73/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage4/run01/REPORT.md>
[H8_C]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb07156a37b1938ed3aed95c13abbb545466c573/docs/research/h8_stage_c_execution_20261105/REPORT.md>
[G01]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/G01_CANDIDATE_BINDINGS.csv>
[G02]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/G02_COHORT_BINDINGS.csv>
[CNG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/C_N_G_decomposition.csv>
