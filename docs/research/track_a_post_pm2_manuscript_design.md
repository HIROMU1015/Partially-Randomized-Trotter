# Track A 限定資源比較研究の原稿・図表設計

2026-10-05。[主張・証拠対応表](track_a_post_pm2_claim_evidence_map.md)に基づき、
現在の保存証拠で完成させる原稿の構成と主要4図を設計する。
図の生成・通し原稿の執筆はまだ行っていない。追加計算の計画・authorizationではない。

## 原稿が答える問い

固定DF表現、二次PF、canonical finite-RTE、対称軸配分の十分shot規則における
finite-time coherent-signal推定の条件付き資源比較を、一つの事例研究としてまとめる。
中心は、精度適格性、shot負担、回路費用、測定込み資源、固定構成移送が別の判定量であること。
partialが勝つ条件を追加探索せず、新規計算キャンペーンをここで区切る。

日本語仮題：
「DF-prefix時間発展における精度条件付き資源比較：残差切断・ランダム補完・固定構成移送の事例研究」

英語仮題：
Accuracy-conditioned resource trade-offs in DF-prefix coherent-signal estimation:
a case study of truncation, randomized completion, and frozen-configuration transfer

H4 linear 1.00/1.30 Å、STO-3G、固定DF rank12、8 system qubits、保存参照状態、
T=0.8を冒頭とMethodsで明示する。DF rankとprefix長L_Dを区別する。
development218候補とM2固定5構成は別domain。
delta=T/q、q=1/2/4/8、登録r/K、compiler、state-preparationを除いた費用scopeは
[根拠表の共通条件](track_a_post_pm2_claim_evidence_map.md#採用する範囲)に従う。

## 本文の構成

| 章 | 論証する内容 | 根拠・対応図 | その章で避ける飛躍 |
|---|---|---|---|
| Abstract | taskと限定scope、近接discard反証、B2内の精度依存、固定移送の限定を短く述べる | C1〜C3 | 新algorithm、一般最適性、energy推定全体へ一般化しない |
| 1 Introduction / Related work | 既知のpartial・DF・費用と測定負担の接続を認め、本実装taskの定量的比較という差分を示す | W1〜W7 | 「先行研究はdiscard／測定費用を扱っていない」としない |
| 2 Task and methods | 保存H/state、B0〜B3、候補登録、finite normalization、axis headroom、整数shot、軸別compiled費用を定義する | E1〜E3、PM-2契約 | certified ground state、truth-free選択則、C_eff固定流用を主張しない |
| 3 Development comparison | 同じε=0.05で、近接discardを含むN・C・Gの違いを示す | C1、図1 | 未登録baselineを補間せず、全決定論法への優位にしない |
| 4 Precision and resource competition | 全登録候補の精度曲線とsame-R組で、headroom／normalization／回路費用の競合を説明する | C2・C5、図2・図4 | method切替、q単独の因果効果、未取得の誤差相殺機構にしない |
| 5 Frozen-configuration transfer | 元M2の独立transferと、使用済み5構成のPOSTHOC精度感度を明確に分ける | C3、図3 | held-out再最適化、未知geometry全般の設計則にしない |
| 6 Discussion / Limitations | 不確かさ、axis規則・candidate域・状態・geometry・compiler依存性、P感度を記す | C4・C6、補足 | formal CI、一般的精度限界、物理実行時間へ読み替えない |
| 7 Conclusion / Reproducibility | 同一taskでの条件付き資源比較を要約し、固定commitと保存表への導線を示す | 全根拠 | 完成を理由に次pilot・Track B統合を認可しない |

Methodsの説明式は以下の役割に限定する。

1. s_a=ε/√2−b_aが正なら現行規則で適格。
2. N_a=ceil(2B² log(2/α_a)/s_a²)でanalytic shot burdenを決める。
3. G_RZ=Σ_a N_a(C̄_a,RZ+P)。本文primaryはP=0、P≥0は補足の仮想感度。
4. h_a=s_a/(ε/√2)を使い、連続式のB²・h_a^−2・Cの関係を説明する。

連続式は新しい定理ではなく説明補助。図と正式費用は保存された整数shotを維持する。
stage名・承認履歴・hash列を本文の物語にせず、再現性付録で示す。

## 図の共通規則

- 元データは結果commit 5a1adffad780f0ec4272f5e8bb94713f9ff0f2bcで固定する。
  図1〜3はPM-2の保存CSVと元JSON、図4は既存PM-0の保存CSVを使う。
- 候補選択の表示処理は既存の全集合・登録fingerprintを維持する。新しいbias評価・sampling・compileはしない。
- εは固定302点のみ。log横軸の隣接点を線で結ぶ場合はvisual guideと明記し、exact crossoverは推定しない。
- ineligible／MISSINGは線を切る。0費用に置き換えず、未登録構成を描かない。
- B0/B1/B2/B3の色を全図で共通にする。細かなr/Kの色分けで厳密winnerを示唆しない。
- engineering ±2SEをformal confidence band・family同時CIと呼ばない。
  点最小構成の標本SEは、その表示構成に条件付けた費用変動で、選択手順の不確かさ全体ではない。
- geometry、T、DF rank、L_D、delta/q、ε domain、費用scopeはcaptionか共通条件表で確認できるようにする。

## 図1 元精度のshot負担と回路費用

目的：accuracy適格、少ないshot、安い回路、低いGが同じではないことを示す。

入力：[PM-2代表decomposition](../../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/representative_decomposition.csv)と
[precision ledger](../../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/precision_ledger.csv)。
domain=development、ε=0.05、P=0の次の4構成を固定する。

| 表示構成 | N_total | 共通axisの平均RZ | G_RZ |
|---|---:|---:|---:|
| B0 L_D5 q1 r0 K0 | 25,910 | 8,866 | 229,718,060 |
| B1 L_D12 q1 r0 K0 | 18,471 | 20,168 | 372,523,128 |
| B2 L_D3 q1 r4 K2 | 20,563 | 6,359.71875 | 130,774,896.65625 |
| B3 L_D0 q8 r32 K4 | 36,032 | 31,993.53125 | 1,152,790,918 |

表の数値は読者向け表示に丸めている。描画にはCSVの保存値を使い、丸めた表から再計算しない。
B0 L_D5 q1の保存candidate_idはPM1-B0-rank5-q1-r0-K0。PM-1追加候補のprefixを落とさず、
candidate_fingerprintと併せて元M1候補から区別する。

パネルA：N_real・N_imagを並べ、解析shot burdenを表示。
パネルB：C_cosine,RZ・C_sine,RZを軸別に表示。
パネルC：G_RZを表示し、random構成だけ元paired標本に由来する±2SEを添える。
保存値がこの4件ではaxis共通でも、一般に共通費用として処理しない。
B0 L_D4 q1を使う場合は、全図の候補を変えず「適格だが高資源」の補助例として付録へ置く。

Captionの結論：このdevelopment条件ではB2はB1より多いshotを要しても1-shot費用が小さい。
近接B0 L_D5にも低いprimary点推定が残る。C1の登録domain内の観測であり、実行量子shotや一般優位ではない。

## 図2 developmentの精度依存とB2内の設定変化

目的：精度による設定変更とmethod変更を区別する。

入力：[precision ledger](../../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/precision_ledger.csv)の
development全218構成×302点。候補fingerprintを保持し、各εのaccuracy-eligibleなmethod別primary点最小を表示する。

パネルA：横軸ε、縦軸G_RZ(P=0)、両軸log。B0/B1/B2/B3のmethod別点最小曲線。
methodが適格候補を持たない点は欠測。必要なSEは表示した構成の保存値に限定する。
パネルB：全候補のprimary点最小に属するB2のqを離散stripで表示。
q4 r4 K4→q2 r4 K4→q1 r4 K2を注記し、境界は隣接表示点間の幅として示す。
ε=0.05に縦線を入れ、図1との対応を示す。

不確かさの補助表示は、同じB2 family内の別r/Kとendpoint比較を区別する。
既存の「全302点で他候補と区間が重なる」という結果だけから、全点でmethod間が区別不能とは書かない。
描画前に保存claim auditと表示構成のSEの対応を確認し、formal winner検定を追加しない。

Captionの結論：固定表示点ではmethod最小はB2のまま、望ましい離散化設定が変わる。
同じbias・B・32 cost標本を再利用したPOSTHOC曲線で、302件の独立検証ではない。

## 図3 元M2固定5構成の適格性と費用順位

目的：元precisionでのtransfer支持と、別precisionでの適格性・競争力を分ける。

入力：[precision ledger](../../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/precision_ledger.csv)と
[eligibility boundaries](../../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/eligibility_boundaries.csv)のM2元5構成だけ。
B2 L_D3 q1 r4/r8 K2、B0 L_D6 q1、B1 L_D12 q1、B3 L_D0 q8 r32 K4を追加・除外しない。

パネルA：5構成のstrict eligibilityを横軸εのstripで示す。
パネルB：同じ5構成のG_RZ(P=0)曲線を描き、ineligible領域を欠測として残す。
元のtransfer精度ε=0.05を示し、元TRANSFER_SUPPORTEDの判定と混同しない。

B2 r4のε_min≈0.00525637654710と、表示費用最小がB3からB2へ変わる
0.00667938131366〜0.00674641423837の間を別々に表示する。
総複素bias約0.003725912<0.005でも現規則でε=0.005は不適格、という注記を添える。

図中またはcaptionに「fixed five / posthoc / symmetric-axis Hoeffding rule」と記す。
1.30 Å、T=0.8、二次PF、B2/B0/B1はdelta=0.8、B3はdelta=0.1を明記する。
この境界は受理規則依存で、方法一般の精度限界ではない。
held-outの未登録q2/q4やPM-1 B0 L_D5のcurveは描かない。

## 図4 同じR=qrで見るbiasと費用の競合

目的：normalization／random action期待値を揃えた既存比較で、qの役割を説明する。

入力：[PM-0 same-R CSV](../../artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/same_R_candidate_comparison.csv)。
既存報告で示したM1 development、B2 L_D3、K2、T=0.8、R=8の4件に固定する。
56 groupから有利な組を新しく探さず、この例の適用範囲を限定する。

| q / r | delta | total bias abs（概数） | N_total | C_eff,RZ（概数） | G_RZ（概数） |
|---|---:|---:|---:|---:|---:|
| 1 / 8 | 0.8 | 0.00725187 | 19,489 | 6,809.1875 | 132,704,255.188 |
| 2 / 4 | 0.4 | 0.00173267 | 15,693 | 12,446.9063 | 195,329,299.781 |
| 4 / 2 | 0.2 | 0.000428445 | 15,015 | 23,791.3125 | 357,226,557.188 |
| 8 / 1 | 0.1 | 0.000106825 | 14,858 | 46,332.6563 | 688,410,606.563 |

パネルA：qに対するtotal biasとNを別軸または別小図で表示。
パネルB：同じqに対するC_effとGを表示。
図内に「同一L_D,K,T,R：tau、normalization、random action期待値は同一」と注記する。
数値表示はCSVの保存値から行い、上の丸め値を入力にしない。

Captionの結論：この組ではq増加でbiasとNが減る一方、1-shot費用とGが増える。
これは既存比較の資源項の説明であって、q一般の単調則や誤差相殺の実証ではない。
n_detはaction proxyで、fusion後のblock数ではない。RZを未保存のdet/random/basis別に割り当てない。

## 補足図表

| 補足 | 保存入力と表示範囲 | 守る限定 |
|---|---|---|
| S1 共通P感度 | PM-2 P_envelope、PM-0共通5構成の結果。元ε=0.05でdevelopment218とM1/M2共通5構成を別panel | 共通仮想RZ-equivalent P/shot。candidate集合差をgeometry効果にしない |
| S2 候補・費用・適格性の台帳 | development218、M2固定5、元ε=0.05、6 compiled metrics | INELIGIBLE、NOT_REGISTERED、MISSING、測定されている1-shot costを区別 |
| S3 selector帰属とq8比較 | 既存PM-0の同domain表・metric別regret | primary regret0とPareto一件欠落を併記。旧SELECTION_LIMITEDは撤回しない |
| S4 再現性・provenance | 根拠表E1〜E6、source/result/input hash、既存test audit | local testsはimmutable CIではない。新testやscience runは要件にしない |

## 図生成・原稿化へ進む際の確認

次の原稿化作業で、保存CSVから表示用の選択・整形と静的図を生成する。
新しいscience runner、source authorization、精度grid、candidate、統計検定を追加しない。
保存reference gate・manifestの書換え、科学データの再計算を図生成に混ぜない。

図生成前後は入力の明示allowlist、commit blobとhash、行選択／missing規則、
軸・label・captionのscopeを確認する。全repository tests、旧science runnerを呼ばない。
不整合が具体的に見つかれば、図のために値を補正せず停止し、原因を報告する。

完成確認は以下に限定する。

1. 同一estimand、accuracy規則、費用scopeが本文で比較できる。
2. C1〜C3の数値が固定artifactへ戻れ、事前検証とPOSTHOC感度を区別している。
3. C4の訂正、適格境界／原理限界の違い、欠測の扱いがAbstractと本文で一致する。
4. point estimates、engineering intervals、analytic shots、family内／method間の比較を分離している。
5. 先行研究との差分を限定実証研究として説明し、未評価baselineへの優位を主張しない。
6. 通し原稿と図が揃った後、独立した原稿レビューへ渡す。

投稿先・採択可能性は未固定。原稿レビューを次のpilot認可に読み替えない。
今回はこの設計までで停止する。旧status／manifest不変、追加科学計算0、Track B変更0。
