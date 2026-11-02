# Track A PRリソース評価：共通実験プロトコル v1 正式版

2026-10-19。**ADOPTED_V1_DOCUMENT_ONLY。基本方針と明示条件を文書基準として採択。新科学計算・数値再解析・既存契約への自動適用は承認しない。**
本プロトコルは有限時間の複素signalを同じ精度で推定する、PRと決定論的PFの資源比較を対象とする。主要比較への採用条件と研究を一区切りにする基準であり、energy/RPE全round、実backend/noise、状態準備、FT T資源、母平均保証のプロトコルではない。

入口：[充足表](../../track_a_protocol_compliance_matrix.csv)、[差異・例外](../../track_a_protocol_gap_and_exception_report.md)、[採択判断](../../track_a_protocol_adoption_decisions.md)、[統合報告](track_a_pr_resource_protocol_v1_adoption_20261019.md)。根拠IDは [evidence_catalog.json](../../artifacts/resource_applicability/track_a_protocol_v1/2026-10-19/evidence_catalog.json) の固定commit・file・SHAへ解決する。取得したGit objectを読んだ証拠であり、remote最新一致、private rawの再監査、実行の再現とは区別する。

## 2026-10-19 暫定追補 CT01：compile前後の状態差閾値

ユーザーの「一旦 －７ にして」により、**compile前後の同じ入力状態に対するraw状態ベクトル2-norm差の許容閾値を、従来の1e-9から暫定1e-7へ変更する**。判定はfiniteかつ厳密に`norm(after-before) < 1e-7`。Track Aの全系列・PF公式・PR/決定論PF・cosine/sineに共通の文書基準とし、m5だけの例外にはしない。AD08の「既存閾値維持」は、この検査だけ本追補を優先する。

理由は、固定raw閾値の超過と、測定signal・資源見積りへの影響を区別した上でのユーザーによる暫定採択である。1e-7を理論的誤差上界・厳密bias保証とは認定しない。native/occupation/reference/normalization/closure、DF閾値、shot用allowance、modelの2%等は変更しない。compiler・回路・係数・制御の変更や、費用影響を使う新しい合否方式の導入も含めない。

**既存PASSとcompile前の検査は再利用し、閾値変更を理由とする再検査・再compileは要求しない。** 旧SOURCE・grant・ledger・marker・STOPは保持する。旧FAILを新基準で扱う場合は、同じ対象・入力・検査の保存値との照合を別記録にし、旧STOPを書き換えない。未実施軸の補完や既存費用の自動受理はしない。今回は文書追補だけとし、封印済み実行コード・契約や進行中runへ自動適用しない。他の採択事項と88行の履歴的充足状態は維持する。

## 1. 状態と比較単位

`COMPLETE`は行に記載した既存対象・cohort・endpoint・対照集合に対する充足、`PARTIAL`は一部のみ、`PENDING`は作業/判断待ち、`NOT_VERIFIED`は未照合、`NOT_APPLICABLE`はその主張には適用しないことを表す。不一致の有無は別の`target_relation`列で表示する。過去stageの正常終了を、v1のCore全体完了や個別科学結果の新たな受入へ読み替えない。

比較キーは、系、geometry、basis、入力生成policy版、Hamiltonian/state/snapshot hash、sector/bit順、DF政策と実rank、fragment順序、PF公式ID/係数版、prefix/q/r/Kとr配分政策、T/δ、bias、allowance、shot会計版/α、control/native/preparation/runtime/compiler identity、cohort、PF対照集合版、結果commit。candidate IDやrank数だけではjoinしない。

H4 legacy DF12、H4 tol-only DF10、H6 DF19、H8 DF28を別系列にする。新しいプロトコルの開始日を理由に旧結果を上書きせず、例外採択と元結果の状態を分ける。

## 2. 採択された共通実験条件

|分類|主比較の基本条件|一致の受入方法と現在の限定|
|---|---|---|
|系・geometry|線形H4/H6/H8、隣接1.0 Å|座標・charge・multiplicity・spin構造とhashを保存。geometry系列は別panel|
|basis|STO-3G|basis名だけでなくlibrary版・軌道数・integral規約を記録|
|DF|tol-only、閾値1e-8、rank自動|`final_rank`指定・config fallbackなし。rank値を同じ数へ強制しない|
|入力|SCF・軌道・DF・状態生成規則を照合|SCF実装/初期化/収束、MO順/位相、spinorbital/二電子積分係数、DF correction・signed weights・順序・cutoff・projection、sector/state solver・正規化・残差・縮退処理をmanifest化|
|時間|T=.8|時間単位・δ=T/q・signed substep順をSOURCEへ結合|
|状態準備|P=0|同じ保存状態を両手法に使用。状態生成が無料という物理的主張ではない|
|精度|ε=.05/.01/.005/.001/.0005|複素signal精度。軸配分ε/√2を保存。energy精度へ換算しない|
|主会計|通常signed bias、allowance倍率1|empirical u/log margin・数値診断・headroom規則を固定。厳密bias certificateは別|
|回路|full Hadamard wrapper、cos/sin、symmetric-directional|元constant＋必要なtail identityを一回適用、referenceを回転しない。低branch I/高branch Uの意味論を保持|
|compiler|Qiskit1.3.0、rz/sx/x/cx、opt1、seed17|backend/target/coupling/layout/routingの指定有無、Python/runtime、basis lowering、native policy、prep/fingerprintも照合|
|費用|両軸compiled RZ×保存解析shotの合計|PRはpaired trajectory費用の有限標本平均、PFは実測/modelを明示。RZは非Clifford数やFT T countではない|

既存主系列のshot配分はα_total=.05/α_axis=.025として資料に記録されている（E_PAPER）。AD04により既存会計の基本方針として採択する。SOURCE固有のallowance/guardは変更しない。費用標本のSE・α_C・family sizeとは異なる。

H4 legacy DF12は共通DF政策に不一致であり、D/Pが完了してもtol-onlyへ改称できない。H4 Fはtol-only1e-8/final_rankなし・generation順・cutoff0・weighted Hermitian projection・新SCF/新stateをSOURCEで確認した（E_F_SOURCE）。H6/H8との**全生成規則の同一性はNOT_VERIFIED**。閾値とrank policy名の一致だけでは入力gateをPASSにしない。DF政策を揃えても分子サイズの因果効果や漸近scaling則は成立しない。

## 3. Core：主論文の最低検証基準

以下の基本手順を採択する。具体的候補数・n・新しい閾値・主張familyは個別契約で固定する。条件付き項目はAD05/08/09/10/11の限定を守る。v1の文書採択だけで既存結果の受入statusを引き上げない。

### D：決定論的PF対照

- full PF、B0、B0+I（B0I）とS2/S4を登録する。公式名だけでなく係数原本/版/合成順を固定する。AD03によりm5_best等の追加PFの主比較採用は保留し、既存S2/S4比較条件を維持する。数値係数の次数残差が非ZEROというだけで棄却せず、保存精度・数値次数検証・適用窓を記録する。
- 系ごとのprefix/q/次数の候補一覧と拡張・選択規則を費用評価前に固定する。同じ候補数を全分子へ強制しないが、PRに有利な対照だけを選ぶことは認めない。既存強化集合を後から弱めない。
- 適格性はELIGIBLE/INELIGIBLE/UNDETERMINEDを分ける。全対象endpointで保存不適格の候補をcompileしないことは認める。欠測/未確認をFalseとして除外しない。
- `DIRECT_MEASURED`、`VALIDATED_MODEL_ESTIMATE`、`MISSING`を軸別に保持する。モデルはSOURCE/input/control/compiler/preparation、次数、prefix/q、軸の適用範囲を明記し、fit前に検証点・閾値・STOPを固定する。旧2%は該当する既存契約で維持する条件で採択した。compile前後の状態差だけは暫定追補CT01に従い、他のnative/closure閾値は各SOURCE/契約のままとする（AD08）。再fitや追加anchorを自動実行しない。
- 固定集合の最小費用・ties・境界属性・証拠範囲を記録する。未知候補がある全登録最良はnullとし、別登録subsetの結果はsubsetと表示する。モデルを含む最小値はモデル条件付き。全格子直接実測や大域最適性を主張しない。

### A：PR探索

- prefix、outer PF、q、r、K、RTE配分と元の無条件lawを登録する。rはpiece当たり/outer合計等の意味を必ず付け、R、signed substep、normalizationまで結合する。
- 共通5εは同じ候補のsignal/単価を再利用して会計する。精度ごとに重複compileを要求しない。費用探索n_E、選択目的、ties、予算とboundary対応を事前固定する。
- 各系の候補域はDF rank・既存適格性・実行可能性・既存boundaryを根拠に疎な格子で設計する。p/r/Kの端、次数別prefix端、非直積q-r schedule端、物理的q=1下端、singletonを区別する。端にwinnerがあっても自動拡張しない。AD06により登録内最良と境界を明示する。拡張は個別の事前規則・別承認で決め、自動では行わない。
- n8探索、H4旧n32 fixed anchor、H6/H8確認union内選択、選択を事前固定した独立cohortを別ラベルにする。新旧から無登録のunion winnerを作らない。候補数・選択後効果と有限標本性を示す。

### P：事前固定候補の独立確認

AD07によりH8の「探索後に候補・対照を固定し、新cohortを全部取得してから判定する手順」を共通手順として採択する。**n=48を全系へ強制しない。H8候補IDも移送せず、標本数は候補ごとに個別設計する。**

1. 原則ε=.05と.0005の主対象を探索結果から固定する。PR候補、PF候補・費用の版、保存shot、期待方向、主要/診断役割を新sampling前に封印する。両端が不適格なら不適格を保存し、代替endpointは別の事前判断として記録する。両端以外の主張は「探索値」と明示するか、重要性に応じExtended確認を別判断する。
2. 候補ごとの分散/SE・識別したい差・未観測event・設計仮定を評価してn_Cを事前設計する。既知n8分散だけを母分散と認定しない。固定予算の点順位再現と、より強い差推論を区別する。未観測rare費用を0補完しない。数値のn/目標power/誤差/検定familyは個別の事前判断・承認事項であり、AD07は一律値を採択していない。
3. 新namespace・新seedと旧master/derived seedの除外照合、元の無条件law、独立cohort、cos/sin pairingを守る。異なるseedだけで統計的独立性が数学的に証明されたとはしない。探索・union選択標本・条件付きpilotとのpoolingは禁止。
4. 固定n完了後、両軸費用から平均・paired SE・rare event観測を記録する。候補内で複数εを共有する場合は同じtrajectoryを独立標本と数えない。候補間pairingを仮定しない。
5. PR有利・不利の両方向について、`POINT_REPRODUCED`、`POINT_NOT_REPRODUCED`、`EXACT_POINT_TIE`等の判定規則を事前に定める。非再現でも正常完了した実験として残す。科学的な非再現はtechnical failure/retryの理由ではない。
6. 点方向の再現、近似CIの診断、母平均保証を分ける。区間/有意性を使う場合は事前に比較family・手法・仮定を固定し、0を含む場合は母平均順位未確定。近似区間が0を含まなくても、未観測tailを含む厳密保証へ昇格しない。

### M：費用分解と説明

各手法m・軸aで `G_m=Σ_a N_m,a C_m,a`。PRのCは同一cohortの軸別標本平均であり、GのSEはpaired trajectory総費用から扱う。

`R_G=G_PR/G_PF`、`R_N=Ntot_PR/Ntot_PF`、`Ceff_m=G_m/Ntot_m`、`R_Ceff=Ceff_PR/Ceff_PF` とし、`R_G=R_N R_Ceff`。また `w_PF,a=N_PF,a C_PF,a/G_PF`、`R_N,a=N_PR,a/N_PF,a`、`R_C,a=C_PR,a/C_PF,a` として `R_G=Σ_a w_PF,a R_N,a R_C,a` を保存値と照合する。

単純平均の回路比とshot合計比の積で代用しない。Ceffはshot配分にも依存し、純粋な回路だけの因果量ではない。0 denominator・不適格・欠測にはnullと理由を残す。実測/model区分は分解後も維持する。

normalization、empirical margin、signed bias/u、headroom、α、整数ceilは、保存会計と同じ式・演算順・許容差に結合して説明する。例えば既存の `h_a=ε/√2−|bias_a|−u_a`、`Nstar_a=2 exp(ell_upper)^2 h_a^(-2) log(2/α_a)` と丸め因子は、そのSOURCEが採用した規則の下でのみ使う。step bと累積Bを同一視しない。説明のbinary64/log整合と費用coreのexact恒等式を別statusにする。

保存G/core照合のPASSが、説明入力の欠測を解消するわけではない。説明のない点は費用比だけの主張へ限定し、normalization/headroomの定量的説明を付けない。AD09/10の条件付き採択により、説明未検証点は会計coreの範囲に限定し、例外ID・captionで欠測/未検証を明示する。定量的機構説明が必要な主張には、対応cohortの検証済み説明を使う。

### T/S：技術・統計的受入

- Native低/高branch、scalar一回、独立event/occupation oracle、Hadamard両軸を確認する。compiler変更・新loweringは対応する人工/実native controlを再登録する。compile前後の状態差には暫定追補CT01の共通1e-7未満を適用する。他の技術検査の既存1e-9/1e-8等は維持する。閾値だけの変更で、既存PASSやcompile前検査を再実行しない。
- SOURCE/input/preparation/元law/compiler/runtime/fingerprint/trajectory seed・event・両軸・shotのidentityを結合する。同じcompiler optionsだけで同一費用系列と認定しない。原profile bytes未確認とSOURCE上のcontrol一致は別の項目。
- nはtrajectory数。cos/sinを2n標本としない。SEの定義、探索/確認の役割、rare event各層の観測/未観測を記録する。条件付きpilotの平均を無条件平均へ無承認で混ぜない。未観測は0費用・分散0・上界0ではない。
- plan/grant/ledger/marker、予約と完了、native exit0、worker wait/reap、結果SHA、最終byte照合、mandatory STOPを記録する。報告文だけの終了確認と、実receiptを取得した確認を分ける。過去のSTOPを遡及PASSにしない。

## 4. Extended / Guarantee

|階層|対象|Coreとの関係|
|---|---|---|
|Extended|他εの独立確認、非相殺、allowance感度、DF政策/rank感度、T/距離、別state、compiler感度、探索拡張|必要な頑健性主張に限定して別判断。全変数直積を義務化しない|
|Guarantee|未観測rareを含む母平均費用差、妥当な全support tail上界、厳密bias certificate、必要なら選択後保証|適切な定理・仮定・証拠が必要。大n、ゼロ観測、経験的SE、モデル誤差PASSだけでは成立しない|

**Core完了後にExtendedやGuaranteeへ自動進行しない。** 保存DFと元電子Hamiltonianの誤差保証、非Clifford/FT資源、RPE全roundは別の対象・承認とする。

## 5. 主図・主表への採用規則（条件付き）

|表示区分|条件|許容する主張|
|---|---|---|
|COMMON_CORE_READY|採択された入力条件とCoreが掲載scopeで充足、欠測のない比較、選択/確認が区別可能|その固定集合・endpoint・有限標本・単価証拠の下での比較|
|CONDITIONAL_CORE_PANEL|相違/未確認/model/限定集合を例外IDでユーザーが承認し、凡例・caption・行に併記|条件付きbenchmark比較。common完全一致や全格子最良を名乗らない|
|EXPLORATORY_OR_RESTRICTED|P未完了、説明欠測、比較集合未解決等を保持した掲載scope|探索点・利用可能subset・説明のない会計値など、証拠のある部分のみ|
|NOT_READY_FOR_REQUESTED_CLAIM|identity不一致、未知候補を除いた大域最良、異cohort混合、説明の逆算補完等|その主張の掲載を保留。既存結果そのものを無効化しない|

AD10により区分と明示例外の方針を条件付き採択した。今回いずれの系もCOMMON_CORE_READYへ自動認定せず、個別図表の最終採否は別判断とする。低費用になったかどうかを受入条件にしない。PR不利や非再現も同じ基準で掲載する。

不適格は`INELIGIBLE`、真の欠測は`MISSING`、適格性未確定は`UNDETERMINED`、未照合は`NOT_VERIFIED`、推定は`MODEL_ESTIMATE`、説明入力欠測は`EXPLANATORY_INPUT_MISSING`。null/空欄を0へ変えず理由を付ける。全登録集合が未解決でも、明示して固定したsubsetの記述は別に保存できる。

分子サイズに関しては、各保存入力・状態・探索domainの有限benchmark差を記述できる。同じDF政策だけから、サイズの単独因果効果、普遍的優位、漸近指数、H10/H12の費用/必要nの外挿を主張しない。

## 6. 停止・一区切り・将来のH10

技術的STOP（identity破損/nonfinite/native不一致/実プロセス異常等）と、科学的な非再現/不利/不適格/解釈保留を区別する。新attemptは個別契約・明示された修正/実行承認に従い、既存消費/STOPを保持する。v1の正式化で旧retry契約を変更しない。

共通入力・比較集合・cohortの不足が残る場合は、その主張を保留するか、固定subset/legacy/モデル条件付きの例外をユーザーが採択して一区切りにできる。ゼロ欠測・有意差・PR優位を得るまで計算することを完了条件にしない。残るrare/厳密bias/Extendedを列挙してSTOPし、追加計算なしで限定論文化へ進む判断も有効とする。

H10等の追加では、入力policy/manifest照合 → D/A登録と疎なsignal適格性 → 必要単価の実測/事前検証モデル → 5ε会計 → 両端の固定P候補/n設計 → 新cohort → 同じ掲載scopeのM/T/S → STOP、を最低手順とする。過去の候補数・rank・n48・worker数をコピーしない。資料で充足済みなら重複計算をせず、最初はmetadata gapを確認する。各科学stageは独立に承認する。

## 7. 今回の権限と次の担当

今回の正式化作業は静的SOURCE/資料照合、保存結果の転記、metadata整理、文書・構造/人工テストだけ。科学module import、sampling、compile、signal/SCF/DF/state生成、保存値の新統計・比計算・shot再会計・rankingは実行しない。既存Codex A/B、m5 pilot、grant/ledger/marker/STOPは変更しない。

独立した読取/構造processは並列化可能。将来の計算も依存関係・candidate/trajectory単位で並列化し、inner thread/COW/cache等を適切に使う。歴史的なCPU/RAM/worker上限を恒常化せず、共有設定・他利用者を変更しない。一時PSI/CPU競合・無関係なworktree更新だけで正常な計算を止める契約へ戻さない。

local commit後にmandatory STOP。文書基準の正式化だけを完了し、push・科学計算・数値再解析・旧契約への自動適用は行わない。**次の担当はユーザー**：公開の要否と残る個別stageの承認を判断する。

## 8. 採択履歴と最新証拠の扱い

DRAFT `9c54d89f6154525e3c7c814a4da5ecdc37453b8a` からの文書正式化。AD01/02/04/06/07/12は基本方針採択、AD05/08/09/10/11は既存区分・閾値・欠測・例外を維持する条件付き採択、AD03は追加PF保留である。[採択判断表](../../track_a_protocol_adoption_decisions.md)と機械可読adoption_recordが判断scopeを定める。旧DRAFTは歴史資料として保全し、条件付き採択を全細目の無条件採用と解釈しない。

- H6独立確認 Phase 0：`6d290fc8fb656f4db148e2c9f91b757d80ed83c2`。adapter SOURCE `1ab88d7c13e7f635bc00bf93a8ecac3d0fb2fd46` の技術準備完了。固定案はε=.05のp4/q2/r4/K2（PR不利方向）とε=.0005のp4/q8/r4/K2（PR有利方向）。最終n・seed registry・科学SOURCE・call数は未seal、launch_ready=false。**Phase 0完了は独立P実行完了ではない。** 既存旧union n32や高次PF別系列の確認を代替にしない。
- A82原signal回収：`bd41b3760ff89e428380f55585d78094a9330a56`。対象は `A82_OLD_PR.PR`、旧union n32のp7/q2/r4/K2。signal SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe` とD0解析SOURCE `20f89bc32b161ee22e4a1ed709f9e3414c19633a` を区別する。原signalとnormalization/signed_bias fieldの回収・metadata同一性を報告。**説明数値検証は未完了**であり、既存説明63/64 sideの完了件数は変更しない。statusは `RECOVERED_SAVED_FIELDS_NOT_NUMERICALLY_VALIDATED` とし、歴史結果の欠測statusも上書きしない。

両commitは固定Git objectとして読み、科学SOURCE/STOP/契約をmerge・改変・実行していない。後者報告のH4 P未完了という記述は当該資料の歴史的scopeであり、別に取得済みのH4 P完了 `8a18b277…` を取り消さない。進捗は主題・固定commit・役割ごとに判断し、資料の時刻だけで一括上書きしない。
