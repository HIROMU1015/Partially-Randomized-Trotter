# 研究C：論文構成・図表・追加証拠の準備

2026-11-05 JST。**未採択の原稿設計。** 文献・式・固定数値・candidate台帳は[証拠整理](research_c_related_work_novelty_evidence_20261105.md)に集約し、本書では重複転載しない。新科学計算・保存結果の再解析・重要科学レビュー・論文化GO認定は行っていない。

## 1. 中心の問いとタイトル候補

中心の問いは、**物理精度を保つ一般CDFの表現自由度を、固定PR-tailの実atom係数に合わせて最適化すると、近接するentrywise目的より下流の条件付き量子資源を改善できるか。その利益は有限最適化政策・初期値・正則化・未使用geometryに対してどこまで残るか**、である。

現在あるのは単一H4における有限手順比較と、候補・rに依存する同J/U決定論比較である。一般性・目的だけの因果効果・独立新規性・保証付き総資源優位の答えはまだない。

タイトル候補は以下とする。採択ではなく、A〜D結果を受けて主張の強さに合わせて選ぶ。

1. **PR-Aware Double Factorization: Tail-Mass Objectives and Conditional Resource Tradeoffs**
2. **Representation Optimization for Partially Randomized Quantum Chemistry: A Finite H4 Case Study**
3. **From Mapped DF Tail Mass to Phase-Estimation Resources: Evidence and Limits**

現段階のデータ範囲に最も忠実なのはcase studyを明記する案2。A〜Dで頑健性や移送が支持されても、より広い適用範囲や独立新規性をタイトルだけで先取りしない。

## 2. 現時点で書けるストーリー

PRの費用は、決定論headの負担とrandom tailによる減衰・shots・kernel単価の釣合いで決まる。CDFは同じ物理Hamiltonianを近似する複数の係数／軌道表現を許す。そこで、entrywise正則化ではなく、実際のPR atomへの写像後tailを抑える目的を既存CDF fitへ接続する。

保存された近接比較では、共通初期化・有限政策・共通DF admissionを通るPR/ENTRYの終点が異なり、PR側は小さいmapped λを得た。候補固有のPF/RTE・shots・native価格まで接続すると、kernel単価の増加を含めても条件付き主RZの低い点値が得られた。これは単にλが下がったという報告より具体的である。

ただしJ/U同時一次収束と公平retuningは未完了で、外部原法への勝利でもない。同J/U強Dに対しては、r128で不利、別r32条件で未解決、別candidate/r32でnominalな支持がある。この条件依存性を本文へ残し、PR一般の優位とはしない。最後にA〜D結果を置き、追加利益がどの範囲で残るか、または消えるかを記述する。

この順序なら、厳密保証の完成や技術監査自体を研究の主題に据えず、**目的の定義→有限比較→下流tradeoff→頑健性と限界**を中心にできる。厳密保証・実装の資格は、対応するclaimの強さを定める条件として示す。

### 中心claimの暫定文案

| 対応ID（証拠整理§6） | 原稿に使える限定表現 | 条件が整うまで使わない表現 |
|---|---|---|
| C1 | 「固定number-CDF tailの非identity atom係数を用いた正則化目的を具体化し、entry上界との関係を示す」 | 「signed J、frame U、係数写像、tail集中を初めて発明した」 |
| C2 | 「登録した有限H4政策で、共通DF予算を満たす終点間の条件付き資源差を観測した」 | 「公平に最適化されたRC-DF/UWCを上回る」「実CAを保証して総資源を削減した」 |
| C3 | 「shots/Wの改善とkernel単価増加が競合する具体例を示す」 | 「λだけで総資源を予測できる」「このtradeoffの原理自体が新しい」 |
| C4 | 「同J/U比較でも、candidateとrにより主RZの結果が異なる」 | 「PRは決定論PFより一般に低費用」「全指標で優れる」 |
| C5 | 「頑健性と未使用geometry移送は未取得の検証課題である」 | 成功・再現性・汎化を結果取得前に本文やabstractへ書くこと |

## 3. 章構成と証拠の置き方

| 章 | 伝える内容・主な問い | 現在使う材料／未取得結果の位置 |
|---|---|---|
| 1 Introduction | PRの分割だけでなくDF表現を見直す理由。先行研究に残る具体的な問い | 証拠整理§2–4。CDF/RC-DFの自由度とUWC/PRの目的を先に認め、固定tail辞書と実資源で追加利益を問う。世界初とは書かない。 |
| 2 Background and related work | number-CDF、PR head/tail、finite RTE、RPE、費用scope | 原論文L1–L9と固定SOURCE。測定shots／RPE shots、qubitization λ／tail λ、native RZ／FT Toffoliを区別。 |
| 3 PR-aware DF objective | T写像、identity／κ規約、smooth tail mass²、ENTRY envelope、J/U処理 | 証拠整理§4とE1前段。既存の代数関係を提示し、非凸U問題と固定U問題の違いを説明。新収束定理は置かない。 |
| 4 Numerical setup and comparison policy | 系・rank・LD・時間・精度・failure・oracle・compiler・有限予算・baseline | E1の固定条件／契約／SOURCEと、比較I「目的能力」・比較II「有限recipe」の区別。A〜D担当の実際の固定政策は公開後に追加。 |
| 5 Finite objective comparison and downstream resources | PR/ENTRY終点、物理admission、λ/B0/相殺gap、PF/RTE、shots/W、kernel、shell込み費用 | E1/E1-FIT。現状の主たる限定比較。名目的差区間の対象、N32/arm、未収束を同じ章に記す。 |
| 6 Mechanism and deterministic comparison | row-sum相殺という観測、単価とshotsの競合、同J/U強Dの正負・未解決 | E2/E3/E4/E5。candidate・r・cohortごとにpanelを分ける。λのみの因果ablationとは呼ばない。 |
| 7 Robustness, fair retuning, and transfer | 追加利益が初期値・強度・予算・未使用H4で残るか | **A〜D placeholder**。結果も図も未挿入。結果不成立時にも保持できる章として設計。 |
| 8 Discussion and limitations | 既知原理との関係、具体差分の意義、負結果、適用範囲、未資格 | 証拠整理C1–C5・残論点。物理誤差／数値／sampling価格／全law／実CAを分ける。新規性や投稿GOはユーザー/GPTの別判断。 |
| 9 Conclusion | 取得済み範囲だけの結論と次の研究問い | A〜D反映後に書く。現在は有限H4観測と残る条件を述べる仮文まで。 |
| Supplement | 完全条件、数理対応、収束診断、旧負結果、資格・provenance | 固定JSON・Markdown表・旧SOURCEへリンク。全rawの再転載や新監査体系を作らない。 |

### 原稿の数理部分の限界

Tの定義と $B_0-\lambda_R\ge0$、相殺のない符号領域での同値は既存設計の式を再利用できる。これを最適化解の分離・収束率・普遍的資源改善定理へ拡張しない。固定Jの直接U penaltyは0でも、Jを再solveしたreduced目的と下流PF・Gaussian価格はUに依存するため、「Uは関係しない」とは書かない。

UWCのsoft costにはdeterministic countとtail²が入り、Cはtensor fitと固定frame-tailを使う。この差はMethod末尾とDiscussionで説明する。RC-DFのsigned行和写像が既知であることは、Cの式を導入する箇所で引用する。FROの残差tensor L1、RC-DF/ENTRYのJ-entry penalty、Cのmapped atom massを同じL1という語で混ぜない。

## 4. 主図・主表の候補

以下は**図の設計表であり、今回新しい図や数値は生成していない**。確認範囲（baseの`artifacts/`・`docs/`、E1固定tree・報告の画像リンク）では研究C専用の完成PNG/SVG/PDFは見つからなかった。既存Markdown表・JSON・一部CSVを再利用する。他branch・別環境の全画像が不存在とは断定しない。

| ID・配置 | 何を伝えるか | 固定SOURCE・保存データ | 既成図表／作成に必要な作業 |
|---|---|---|---|
| F1 主図・方法模式図 | J/U→number→T写像・head/tail→PF/RTE→attenuation・shots/W→kernel＋shell。UがPF/価格にも効くこと | [PR SOURCE][PR-SRC] `coefficients/smooth_mass`、[ENTRY SOURCE][ENTRY-SRC]、既存RTE/RPE会計 | 完成画像未確認。式・依存関係の模式図は文書作図だけで可能。新科学計算不要。未取得量は実測値のように描かない。 |
| F2 主図・PR対ENTRYの比較panel | λ／B0／相殺gap、shots/W、kernel単価、総RZを並べ、proxyと総費用の対応を示す | [E1-FIT][FIT]の保存表、[E1報告][PAIR]、[E1 result][PAIR-JSON] | 保存表を直接使える。異なる単位は別panel・明記。新比率や相殺寄与率を作らない。総RZ差の保存区間だけ使用し、PR/ENTRY比CIは作らない。 |
| F3 主図または補足・round別の接続 | 後半roundでのPF/RTE、A、radius、shotsの候補差を示す | E1 [PR refs][PR-REF]／[ENTRY refs][ENTRY-REF]、報告のm0..12表 | 表は既存。保存済み各値の転記で作図可能。新fit、外挿、誤差帯、log-slope等の再解析は別許可が必要。今回は図仕様のみ。 |
| F4 主図・同J/U強D比較 | update005/r128不利、005/r32未解決、064/r32支持を並べ、条件依存性を示す | [E3報告][DET]、[E5 primary][005-JSON]、[E4 primary][064-JSON] | 保存比／区間／判定をそのまま使う計画。r128は保存点値、r32はnominal CIと明示。candidateや精度政策の異なるpanelを一本の改善曲線として結ばない。 |
| T1 主表・setup | 物理系、basis/rank/LD/時間、誤差予算、failure、oracle、P0、最適化政策、native N | E1-FIT/E1と各SOURCE・契約。完全identityは証拠整理§5へ | 既存表を短縮再利用。δ=.1は単一条件で、delta掃引と表記しない。 |
| T2 主表・related-work差分 | 同じ自由度／既知の目的／Cの特定mask・辞書・近似予算の違い | 証拠整理§2–4の一次論文対応 | 同表を要約して原稿用に使う。外部論文の資源値を横並びランキングにしない。 |
| T3 主表・qualification | evidence／ideal／assumed、有限native correctness、名目的CIとactual CAの区別 | E1 lane表、E3数値pilot、E4/E5資格 | 既存表あり。定理、engineering guard、観測残差、標本CIの意味を別欄にする。 |

F2はC1/C2/C3、F4はC4の支持範囲を示す。F4の肯定panelだけを主図に採用して負結果・未解決を隠さない。図を見ただけで対象candidateと条件が分かるcaptionを付ける。

### Captionに残す条件

H4／1.00 Å／STO-3G／number CDF8／元順LD2、δ=h=.1、s/r/K、m0..12、candidate label、RPE失敗予算、状態準備除外・own-DF ground、P0/fenced serial、native N、laneと区間種別を記す。完全SHAはT1または補足の対応表に置く。別geometryや異なるrankの図では、その都度実際の条件に差し替える。

## 5. 補足図表と既存データの使い方

| ID | 内容・科学的な役割 | 保存資料と追加作業の境界 |
|---|---|---|
| S1 表 | E1の16更新、J一次MET/U NOT_MET、trial/refinement、nfev/njev、raw失敗を含む有限予算 | E1-FIT本文と`artifacts/research_c_near_practical/2026-11-03/run01/result.json`。既存集計値のみ引用。新収束率のfitやoptimizer再実行なし。 |
| S2 表／図候補 | update005の5更新とupdate064の別64更新。研究経路の区別 | E2報告、update064最適化結果`ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6`。既存表を再利用。異なる軌跡を一つの公平な収束曲線へ合成しない。 |
| S3 表 | tail-P4、CDF7、surrogate、switch移送、shared-basis、round割当の負結果と射程 | 証拠整理§5.4、10月31日総括と11月3日Map。新ランキング不要。分子/rank/LD/rの違いを残す。 |
| S4 表 | ideal／assumed laneと有限標本価格の差、actual CA null | E1/E4/E5の保存資格・感度表。未資格量を0に置かない。 |
| S5 数理補足 | T写像、entry上界、同値符号領域、identity/κ、RC-DFとの規約対応 | 既存near-objective設計と固定source。新定理・新科学数値なし。 |
| S6 再現用一覧 | scientific SOURCEと閲覧commit、candidate、input packet、結果JSON、契約／STOP | 証拠整理§5の台帳と固定SOURCE manifestを参照。過去のgrant／ledgerやSTOPを更新しない。 |

CDF7の`presentation/calibration6.csv`／`conditional_comparisons24.csv`、P4の`PF_RTE_rounds13.csv`等の保存表もある。現在のE1/E4の主結果を代用するものではなく、必要な負結果の提示に限って原系列の条件で使う。

係数ごとの符号図、frame別mass再配分図、相殺の因果寄与分解は、上記の保存scalarだけでは作れない。新しいJ/U decodeや数値再集計が必要なら、この作業の対象外として明示許可を受ける。保存gap表だけで因果的な改善機構を実証したとのcaptionは付けない。

## 6. A〜D統合担当への引渡し

ここでのA〜Dは依頼文の四つの検証軸を整理するための呼び名である。別担当のphase名・契約・結果schemaを新しく定義するものではない。担当のSOURCE・契約を編集せず、未公開worktreeやjobへ書き込まない。以下は原稿が受け取る情報の対応であり、追加実験の要求・新しい合否閾値・実行許可ではない。

| 軸・placeholder | 科学的な問い | 受け取る保存情報 | どのclaimを支持／弱めるか・反映先 |
|---|---|---|---|
| A 初期値・正則化・有限予算の頑健性：**結果未取得** | E1の差は一つの初期値・強度・停止政策に強く依存するか | 登録した全arm/起点/強度/予算と失敗も含む結果、候補identity、J/U到達度、DF誤差、exact λ/B0、実計算負担 | C1/C2/C5。複数登録条件で利益が保たれれば有限recipeの頑健性を支持。特定seed/強度だけなら主張をその条件へ狭める。§4/7、F5/S1。 |
| B PR／ENTRYの公平retuning：**結果未取得** | 近接対照に同等の調整機会を与えても追加利益が残るか | 各armの実際の強度・選定規則、共通物理admission、到達物理誤差帯、予算・到達度、固定後の候補と共通下流指標 | C1/C2。ENTRYが追いつく／逆転すれば「目的固有の追加利益」は弱まる。差が残ってもUWC/RC-DF原法への勝利ではない。§5/7、F5。 |
| C 未使用H4 1.15 Å移送：**結果未取得** | 1.00 Åでの構成・政策が未使用geometryでも有用か | geometry/basis/rank/LD、学習・選定／held-outの区別、移したものと再調整したもの、全candidate固有のDF/law/PF/RTE/価格 | C5。成立してもH4の一geometry移送まで。不成立なら単一条件case studyへ限定。geometryを用いた後でheld-outと呼び直さない。§7、F6。 |
| D 物理精度→PF/RTE→shots→native→総資源の統合：**結果未取得** | proxy利益はtask費用まで届くか、同J/U強Dにも残るか | 各candidateの資格・lane、13round等の実際の範囲、shots/W/shell、独立価格・不確かさ、同J/U D、下流retuningと校正の区別 | C2/C3/C4。λのみ改善・価格/PF悪化で総利益消失なら資源claimを弱め、tradeoff結果として残す。区間重複／精度未達は未解決。§5–7、F2/F4/F7。 |

### 未取得図表のplaceholder

| ID・想定配置 | 図として問うこと | 必要データ／現在の扱い |
|---|---|---|
| F5 主図候補 | 初期値・強度・有限予算でPR/ENTRY差はどの程度変わるか | A/B担当が登録した全結果と比較可能性。表示形式は結果schema取得後に決める。空のplaceholderのみで、予想分布・成功曲線を置かない。 |
| F6 主図候補 | H4 1.15 Åへ何が移送できたか | C/Dの固定held-out結果と、実際の元条件との対応。新geometryに旧λやnative価格を流用しない。 |
| F7 主図または主表候補 | fair retuning後の物理精度と条件付き総資源の関係 | B/Dの保存frontierや選定済み結果が存在する場合だけ利用。今回frontierを再計算しない。非適格candidateや失敗も表示方針へ含める。 |

「結果未取得」は本作業が固定公開結果を受領していないという意味であり、別担当の進捗を監査・断定したものではない。別担当が既に取得した場合も、その結果commit・SOURCE・比較条件・STOPを確認してから反映する。

### 結果別に変わる論文の形

| 将来受領する結果 | 可能な原稿の位置付け候補 | 依然として言えないこと |
|---|---|---|
| 頑健性・fair retuning・移送と下流利益が揃って支持される | 固定tailの目的設計と追加資源利益を中心にした方法＋検証の候補 | 文献の独立新規性・一般分子優位・actual CA・投稿GOの自動認定 |
| ENTRY retuningで差が消える | 汎用正則化で説明できる範囲と、写像目的の限界を示すcase study | PR目的に頑健な追加利益があるという中心claim |
| λ差は残るが下流差が消える | proxyと総資源の不一致を中心にした機構／tradeoff報告の候補 | λ改善を資源改善と言い換えること |
| 1.00 Åだけ支持、1.15 Å不成立 | 対象系・geometryを明記した有限観測と移送限界 | transferable method、広いrank/basis/分子への一般化 |
| 数値資格・到達度・価格精度が不足 | 未解決点を残した技術・研究記録として整理 | 不足を成功扱いし、都合のよいarmだけで結論を作ること |

これらは結果の受け皿であり、今回どの路線を正式採用するかは決めない。重要科学レビューや追加科学実行は別の明示許可による。

## 7. 論文に書けること・まだ書けないこと

**今書けること**は、SOURCE上の目的・規約、一次文献との重複と具体差分、E1–E5の条件付き有限観測、正負・未解決の併存、再現用原本の所在と適用範囲である。価格校正の「独立」は旧cohortと分けた固定Nモデルの意味であり、外部研究者の再現とは書かない。

**まだ書けないこと**は、完全J/U収束・大域最適、fair retuning後の一般優位、未取得A〜Dの成功、UWC/RC-DF/SCDF全体への勝利、λや相殺だけの因果効果、全law合成・実装law信号距離・validated CI coverage・actual CA、状態準備／実backend／noise／FT実時間を含む最終総資源である。原稿用図の未取得部分を予想値で埋めない。

本文の限定語は読みやすく統一し、重要な条件をcaptionとsetupへ置く。すべての段落に長い監査語を繰り返す必要はないが、abstractや結論から「有限H4」「条件付き」「baselineの範囲」「未検証の一般性」を落とさない。

## 8. 次の担当と停止点

| 担当 | 具体的な次の作業 | 現在の許可との関係 |
|---|---|---|
| A〜D統合担当Codex | 既に担当している準備・実行の範囲で得た固定結果とSOURCE・candidate・資格を返す。本書§6を原稿側の受取項目として参照 | この文書は担当の実行許可・科学条件を増やさない。 |
| ユーザー／GPT | 固定A〜D結果の受領後、C1–C5のどれを中心にするか、追加利益の重要性・文献上の独立性・論文化を判断 | **将来の判断事項**。重要レビューを自動開始しない。 |
| Codex（後続依頼時） | 許可された固定結果の引用・出典照合を行い、§7とF5–F7のplaceholderを実際の結果へ更新する | 本作業の公開後に自動継続しない。保存値の新集計・新科学計算が要る場合はその範囲の別許可が必要。 |

今回の完了は、2冊と入口リンクを指定branchへcommit/pushし、remote SHA／tree・公開本文を照合した時点とする。新科学結果・科学的採択・証拠資格の変更はないため、研究概要・研究ノート・中央manifestの科学statusを更新しない。

**MANDATORY STOP：公開照合後はここで停止する。** 論文本文の全面執筆、科学計算／再解析、A〜D条件変更、新アルゴリズム、正式新規性／論文化GO、重要GPTレビュー、投稿・対外公開判断へ自動的に進まない。今回許可されたGitHub資料公開と、学術論文の投稿判断は別である。

## 固定データ参照

[PR-SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/70a98adfb701a209db2c65ea60fb789ff62b1fe7/src/trottertracks/c_pr_aware_df.py
[ENTRY-SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/70a98adfb701a209db2c65ea60fb789ff62b1fe7/src/trottertracks/c_near_objective.py
[FIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ceb0a8825785ebd4f21241d8ef0f20cb321bf69/docs/research/research_c_near_objective_practical_comparison_execution_20261103.md
[PAIR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/docs/research/research_c_near_practical_r32_downstream_execution_20261103.md
[PAIR-JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/artifacts/research_c_near_practical_r32/2026-11-03/run01/result.json
[PR-REF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/B/references.json
[ENTRY-REF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/artifacts/research_c_near_practical_r32/2026-11-03/run01/ENTRY/B/references.json
[DET]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38b8d0df39b8943178ecef611194819cff1c2169/docs/research/research_c_same_ju_strong_d_execution_20261102.md
[005-JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/459c8472d66689bf149a60fd6cc7311ff5237c65/artifacts/research_c_r32_price_statistics/2026-11-03/run01/primary.json
[064-JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/artifacts/research_c_update064_r32_price_statistics/2026-11-03/run01/primary.json
