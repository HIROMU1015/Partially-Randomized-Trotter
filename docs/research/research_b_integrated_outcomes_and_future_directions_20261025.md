# 研究B：成果統合・科学的到達点・今後の発展候補

2026-10-25 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。
統合作業の固定起点は `e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d`。本書は既存証拠の技術的整理であり、新しい科学計算、数値再解析、独立科学レビュー、研究方針の正式採択を行ったものではない。

## 1. Executive Summary

研究Bでは、**部分ランダム化時間発展のD内部を高次化した利得が、outer D/R分割とfinite-RTEを経て物理的な複素coherent signalへ残るか、さらに回路単価とshot負担を含めても有益か**を段階的に調べた。これは既存研究の問いを整理した表現であり、研究目的を変更するものではない。

固定synthetic 2-modeモデルでは、D内部の高次化、outer分割変更、RTE精度変更がsignal誤差へ与える影響を切り分けた。Full controlled Hadamard wrapperの実装・数値correctnessとnative費用計測も成立し、有限IID標本、未観測rare費用の感度、解析的な十分shot数まで接続した。一方、**保証付き同精度の資源優位、新規算法としての独自性、実分子・全RPEへの一般化は未確定**である。[36行][D36]、[16行][O16]、[22行][R22]、[B0][NATIVE]、[shot加重][SHOT]

- D内部の改善はA0現partial-S2／A1 D-S2に対するfull signal改善として観測されたが、A3 Y8m10bとA4 YP8m8の差は数値識別不能だった。Processorを含む短いD runの構造費ではA4の端点負担が大きい。
- Outer Yoshida4とSABA2では、固定条件のcorrected signal誤差がouter-S2より小さかった。Yoshida4ではfinite-RTEとnormalizationの負担も増え、D内部A2/A3の違いが表面化した。SABA2の一般四次保証を確認した結果ではない。
- A3を内部対照にするとr/K変更によるfull signal改善が観測された。A2ではD近似の影響が残り、RTE成分の改善とfull複素誤差の改善は単調に対応しなかった。
- Native-v5のfull wrapperは固定correctnessを通過した。B1では3,072 trajectory／6,144 wrapperを完了したが、全51,200 short-stepがorder0で、高次数費用は未観測だった。
- Shot加重評価は24条件・48軸別shot数・144資源行・24比較を完了した。S0条件付きではB4対B3、D1対R2の低費用方向が±2SE矩形感度でも残り、D1対R4は±1SEで識別不能だった。`u_num`／`gamma`／rare費用が未取得のため、最終科学状態は **`INCONCLUSIVE_QUALIFICATIONS_AND_RARE_COSTS`** のままである。

既存成果だけで、誤差の層別診断と条件付き資源評価をまとめた技術報告・発表資料を作れる。保証を追加せず現在の限定研究を一区切りにする選択肢も成立する。継続する場合の方向A〜Eは§14の候補であり、採択はユーザーが判断する。

## 2. 研究目的と、対象を取り違えないための定義

[研究B初期調査][INITIAL]の目的は、量子化学時間発展について目標精度を維持しつつ資源を削減する方法を、PR改善・代替方式の両面から調べることだった。その後の[B2限定設計][B2DESIGN]は、D半時間発展にpublished processed PFを入れるnested構成を、既存partial-S2と同配置の通常PFに比較する問いへ具体化した。後続のouter・RTE・native・shot評価は、この限定構成の精度と費用を調べた系列である。

本体の[研究目的][PURPOSE]にある最終目標は、状態準備を範囲外とする全RPE interrogationのshot数×期待compiled costの総和である。今回の研究B到達点は、**状態準備込み・単一複素signalの限定会計**であり、この全RPE目的関数の達成ではない。状態準備の扱い、精度の単位、対象時間・confidenceが違う費用を同一視しない。

三つの層を分ける。

| 層 | 問うもの | 対応する証拠 |
|---|---|---|
| D内部PF | 非可換D blockの近似、processorの端点費 | 36行と静的構造費、後続A2/A3対照 |
| Outer D/R・finite-RTE | D/Rのordered分割、近似tailのmean、normalization | 16行、22行、A0高精度参照 |
| 実装・sampling・資源 | 相対位相を保つcontrol、native単価、費用MC、shot会計 | B0、B1、rare感度、24条件 |

物理的な基準は同一入力状態の `z_phys=<psi|exp(-i H_phys T)|psi>`。比較するのはscalar／identity phaseを戻したcorrected meanの複素signalであり、raw meanのX/Y・半径、operator誤差を別に保持する。各誤差成分は異なる行列差・複素差で、normや絶対値を単純加算・差引きして誤差の等式や因果寄与率にはしない。[D36][D36]、[O16][O16]、[R22][R22]

**名称の衝突に注意する。** 36行の「A0 baseline」「A1 D-S2」と、後続の「A0高精度検証」「A1数値包含案」は別物である。22行の候補B1=S2とB1費用MCも別。後続候補B4=SABA2は、初期調査の分類名B4乱択PFを意味しない。以下では段階名と候補名を併記する。

## 3. 科学モデル・固定条件・比較可能性

共通モデルはsynthetic 2-mode、one-body＋3 DF因子、**N=1／mode0 occupation、L_D=2 weight-ranked prefix**。Dはone-bodyとDF因子0,1、Rは因子2。分子geometry／AO basis／分子DF rank探索は非該当であり、実分子の基底状態を検証した系列ではない。入力・sector・state・D/R identityは各公開原本へ保存されている。36／16／22行ではH identity残差6.16645e-17、D/R commutator norm 0.00428043が報告され、必要な非可換性を確認した。[D36][D36]、[O16][O16]、[R22][R22]

| 段階 | 固定した比較窓・設定 | 比較上の制約 |
|---|---|---|
| D内部36行 | δ=±0.09、q=2,3、A1〜A4のn_D=1,2、r/K=2/2、complex128、atol1e-12 | q間で全時間T=qδが異なる。A0は元partial-S2、A1以降はnested配置 |
| Outer16行 | T=±0.18、q2、refinedのみf2、D内部A2/A3、n_D1、r/K=2/2 | 同じTでouterを比較。Stage1のq3の値と直接同条件ではない |
| Outer/RTE22行 | 同じT・n_D1、候補別q/f/r/K | 変更する反復数・RTE条件の構造費も変わる。資源同一ではない |
| A0高精度 | 22行の保存入力、B3/R4/S1/F1の両T、90/128桁 | 保存binary64モデルをexact lift。全候補の精度保証ではない |
| B0／B1 | 6候補の両T、n_D1、native-v5 full prepared wrapper | B0は4決定論的代表、B1は256 IID/case。標本をpoolしない |
| Shot24条件 | 同じ6候補×両T×ε=1e-8/1e-9 | 主biasは22行、単価はB1、A0は補足。新signal・sampling・compileなし |

Complex128段階のcorrectness atol1e-12は実装照合の受理値で、`u_num`や`gamma`の上界ではない。解析的な36/16/22行は位相診断を無効にして複素signalへ直接比較し、native照合もphase fitなしで行った。Compiler復旧時の位相診断は§8に分ける。A0高精度は別精度の補足で、旧結果の置換ではない。[A0][HP]、[shot][SHOT]

Shot段階が利用した6候補を固定する。

| ID | Outer | D内部 | q/f | r_RTE/K |
|---|---|---|---|---|
| B3 | Yoshida4 | A2 Yoshida4 | q2 | 2/2 |
| D1 | Yoshida4 | A3 published Y8m10b | q2 | 2/2 |
| R2 | Yoshida4 | A3 | q2 | 4/2 |
| R4 | Yoshida4 | A3 | q2 | 2/4 |
| B4 | SABA2 | A2 | q2 | 2/2 |
| F1 | Refined-S2 | A2 | q2、f8 | 2/2 |

その他の22行候補はB1=S2/A2/q2、B2=refined-S2/A2/q2/f2、R1=Yoshida4/A2/r4K2、R3=Yoshida4/A2/r2K4、S1=SABA2/A2/q4。特記以外r/K=2/2。これらをB1 MC／shotの追加費用評価済み候補には数えない。[R22][R22]、[B1][MC]

## 4. 段階間の依存関係と主要結果の俯瞰

36行は「D内部を改善してもfull residualが残る」ことを示し、静的費用はprocessorを含む短runの負担を整理した。16行はouterを動かしてfull signalへの影響を確認し、22行はそこで見えたD近似・tail精度・normalizationを比較した。A0は小残差の数値識別を補足した。B0はordered列をcontrolled native wrapperへ接続し、B1はその単価をIID標本化した。Rare感度とshot会計は、既存費用標本を利用し未知寄与を残した。

これは各報告の診断対象をつないだ説明である。単なる実行順を因果証明とは扱わず、各改善の記述比較は対応する固定実験内に限定する。

| 評価軸 | 主な対象・範囲 | 主要な既報数値 | 証拠の種類／成立しない主張 |
|---|---|---|---|
| D内部精度 | A0〜A4、36行 | A4のD half誤差約1.3–2.9e-15、A3/A4 full signal差最大約1.46e-15 | 固定条件の経験値。A4対A3の優劣INCONCLUSIVE [D36][D36] |
| Outer精度 | 4 outer×A2/A3、16行 | Corrected signal：S2約2.27e-7、Yoshida4約4.53–6.76e-10 | 同じTの経験値。収束次数・同資源優位ではない [O16][O16] |
| RTE精度 | r/K変更等、22行 | D1→R2→R4のb：4.53e-10→8.34e-11→7.00e-11 | 固定A3対照の観測。数値保証付きε適格性は未確定 [R22][R22] |
| 数値的安定性 | A0高精度、8行×90/128桁 | 比較指標の精度間差最大1.843847384e-89、全8件nominal bound整合 | EMPIRICAL_CONSISTENCY。包含証明ではない [A0][HP] |
| Native正当性 | B0、48代表／96 wrapper | 288 operator／192 expectation通過、compiled operator最大4.426725459e-13 | 固定回路の経験的correctness。γ保証ではない [B0][NATIVE] |
| Native期待費用 | B1、3,072 trajectory／6,144 wrapper | RZ/CX relative SE最大約0.0800%、51,200抽出すべてorder0 | 有限IID推定。rareを覆う母平均順位ではない [B1][MC] |
| Rare寄与 | 8 case、16境界／80 SE corner | B4のp_rare約1.39635e-5、B3/D1約1.82896e-4 | CONDITIONAL_SENSITIVITY_ONLY。未知費用を残す [Rare][RARE] |
| Shot加重 | 24条件、48 shot／144資源／24比較 | 各軸約1.096e18／1.096e20 shot、ε=1e-8／1e-9 | 固定統計モデルの十分数と条件付き会計。End-to-end未資格 [Shot][SHOT] |

全て公開commitに記録されたローカル検証で、CIによる再生成や独立外部再現とは区別する。異なる精度・費用単位の行を横断した新しい改善率・ランキングは本書で計算していない。

## 5. D内部高次PF：改善が残る範囲とprocessor費

36行のA0は既存partial-S2のforward/reverse sweepで、前後を独立したD-S2半stepに置換したA1とは異なる。A2はYoshida4、A3はpublished Morales Y8m10b、A4はpublished processed Morales YP8m8である。A4は `P K^n P†` を実現し、採用した実現列のactual inverse・published列の作用順を独立照合した。[B2設計][B2DESIGN]、[実行][D36]

入力・非可換性・A4数学参照4件・published列規約4件の全gate後に36行を完了した。登録941 DAG dispatch・448 exponential keyも一致し、A4数学参照の最大operator差1.89162e-15、published full列参照最大6.69489e-16だった。初回SOURCE54008c69の旧入力contract参照を修正した758a05edで完走し、保存後のCLI表示のFraction変換修正は科学再実行を伴っていない。科学完了とprocess表示の技術障害を区別する。[D36][D36]

以下は36行報告のn_D=1の引用。δ=±0.09、q=2,3を含むfull値のmin–maxであり、同じT一点の表ではない。

| 方式 | D half operator誤差 | Full physical operator誤差 | Corrected physical signal誤差 |
|---|---:|---:|---:|
| A0現partial-S2 | — | 4.007258e-6–5.974889e-6 | 1.807640e-6–2.694640e-6 |
| A1 D-S2 | 2.083856e-7 | 1.339230e-6–1.996814e-6 | 5.999438e-7–8.944059e-7 |
| A2 Yoshida4 | 6.406136e-11 | 5.097543e-7–7.600520e-7 | 2.267377e-7–3.380549e-7 |
| A3 Y8m10b | 1.110794e-15 | 5.094998e-7–7.596725e-7 | 2.266387e-7–3.379070e-7 |
| A4 YP8m8 | 2.888666e-15 | 5.094998e-7–7.596724e-7 | 2.266387e-7–3.379070e-7 |

[36行報告][D36]が掲載したA4のsignal誤差減少は、A0比87.460–87.462%、A1比はn_D1で62.220–62.223%、n_D2で29.163–29.166%。A2との差はn_D1で0.04369–0.04377%、n_D2で0.002731–0.002735%という小さい記述差だった。A3/A4差は丸め水準で、符号や順位を採用していない。A0からの改善にはnested配置変更も含まれるため、その全量をYP8m8固有の効果とはしない。

高次3法のfull誤差が近いことから、この固定条件ではD内部精度を上げるだけで大きなfull利得が続くとは確認できなかった。Outer誤差の残存を次に検査する理由となるが、全モデルでouter律速という結論ではない。

### 端点費を含む既存の構造比較

[静的費用報告][STRUCT]はD block数m=3として、次を記録した。新しい数え上げは行っていない。

| D半runの方式 | 反復kernelのS2数 | Processor片側／actual inverse | q2・n_D1のfull列：literal D指数→合法融合後 |
|---|---:|---:|---:|
| A2 | 3 | 0／0 | 60→51 |
| A3 | 21 | 0／0 | 420→339 |
| A4 | 17 | 20／20 | 1140→755 |

Rが障壁となるので、q回のfull列でA4のprocessor pairはq+1組残る。全executionを無料の一組で挟む会計にはできない。既報のS2 word償却条件は孤立D半runでn>10、合法融合full列でn>5(1+1/q)。q2ならn≥8、q3ならn≥7で、今回n=1,2はこの条件に届かない。これは構造数の交点で、同精度やnative gate費の交点ではない。[静的費用 §3][STRUCT]

A4を後続B0/B1へ追加しなかったことを、A4のnative劣位が実測された証拠にはしない。長いD runで内部精度が必要になる用途の可能性は残るが、今回の結果だけではその用途・同精度n・processorの実価格を確定できない。

## 6. Outer splitting：full signalへの改善とnormalization

16行は同一T=±0.18、q2、n_D1、r/K=2/2でouter-S2、refined-S2/f2、Yoshida4、SABA2をA2/A3内部対照と比較した。以下は[16行報告][O16]の表示値を引用する。各行のraw／corrected X/Y・半径は同報告の原本にある。

| Outer | Corrected signal誤差：A2 | 同：A3 | Normalization積N | R occurrence |
|---|---:|---:|---:|---:|
| S2 | 2.26738e-7 | 2.26639e-7 | 1.000055855727 | 2 |
| Refined-S2/f2 | 5.66296e-8 | 5.66234e-8 | 1.000027927648 | 4 |
| Yoshida4 | 6.75687e-10 | 4.52693e-10 | 1.000365893033 | 6 |
| SABA2 | 2.90506e-9 | 2.80198e-9 | 1.000027927648 | 4 |

同報告が記載したouter-S2に対するsignal誤差比はrefined約1/4.00、Yoshida4はA2約1/335.6・A3約1/500.6、SABA2はA2約1/78.05・A3約1/80.89。既存値の引用であり、資源budgetを揃えた改善率ではない。

Yoshida4ではA2のD近似operator差約1.35e-9に対しA3は約1.57e-14で、内部PFの違いがfull値にも現れた。A3＋Yoshida4でもideal-tail signal差約7.0036e-11に対しfinite-RTE signal差約4.5818e-10が残り、tail精度を調べる具体的な理由となった。この小さいA3のD残差は、後のA0高精度診断が識別限界を補足している。

Raw physical signal誤差はS2約5.57312e-5、Yoshida4約3.64737e-4。**Corrected meanの改善とraw観測信号の改善は同じではない。** 正負stageを持つYoshida4列ではRの絶対時間・RTE負担・Nが変わる。負時間だけを独立した原因として測ったablationではない。[O16 §3–4][O16]

SABA2は既知の弱摂動向け正係数分割として比較した。||R||/||D||=0.07680309と二重commutator診断は記述値で、弱摂動保証の受理閾値ではない。単一|T|の結果からSABA2一般四次やYoshida4実装全体の収束次数は認定しない。SABA2の位置づけは「今回のsignal誤差改善と比較的短い構造を持つ候補」までである。

## 7. RTE精度：r/K変更が有効な層

22行ではrをshort-step数、Kを既存paired finite-Taylor lawのcutoffとして扱う。r/Kを変えるとbiasだけでなく、event分布・normalization・回路構造も変わる。以下は同報告のT=+0.18の引用で、全候補q2・n_D1、F1のみf8。負Tの全値は原報告参照。[22行 §4–5][R22]

| ID | Corrected physical operator誤差 | Signal直接bias b | Finite-RTE signal差 | N |
|---|---:|---:|---:|---:|
| B3 | 1.75192e-9 | 6.75687e-10 | 4.58175e-10 | 1.000365893033 |
| D1 | 5.28691e-10 | 4.52693e-10 | 4.58176e-10 | 1.000365893033 |
| R2 | 1.77540e-10 | 8.33775e-11 | 5.72715e-11 | 1.000182932417 |
| R4 | 1.43716e-10 | 7.00360e-11 | 2.28878e-16 | 1.000365893522 |
| B4 | 5.96822e-9 | 2.90506e-9 | 3.80092e-12 | 1.000027927648 |
| F1 | 7.95421e-9 | 3.53826e-9 | 6.22221e-14 | 1.000006981853 |

A3対照D1→R2→R4ではtail高精度化がfull signalへ残った。一方、A2のr4K2（R1）のb=5.84380e-10、r2K4（R3）は5.92676e-10で、RTE成分の縮小だけではfull誤差の単調減少をいえない。A2のideal-tail signal差約5.92676e-10、D近似signal差約5.22697e-10が報告されており、複素差の相殺も含み得る。位相診断で原因を確定した結果ではない。[R22 §4][R22]

Yoshida4のr2→r4でNは約1.000365893→1.000182932へ低下する一方、RTE short-step数は12→24。K2→K4ではshort-step数12のままevent supportが変わり、Nは僅かに増えた。B3のliteral D指数180に対しD1/R2/R4は1260。これらはplan構造数で、native費を代用しない。[R22 §5][R22]

S1=SABA2/q4とF1=refined/f8でも観測bはそれぞれの対照より小さかったが、step数を増やすため費用同一ではない。K4の約1e-16のfinite-RTE差、S1/F1のnominal bound超過はcomplex128の数値識別問題を残し、A0高精度の対象となった。

## 8. A0高精度・B0 native・B1 IID費用

### A0：小残差の経験的確認

保存binary64 primitiveを正確にliftし、90桁Taylor／128桁Padéと独立数学順の積で8行を確認した。全比較指標の精度間差は1e-40以内（最大1.843847384e-89）で、nominal Taylor boundと整合した。以下はT=+0.18の既報値。[A0 §3][HP]

| ID | 旧complex128 finite-RTE operator差 | A0高精度の同差 | A0 nominal bound |
|---|---:|---:|---:|
| S1 | 5.098455973e-13 | 5.077665782e-13 | 5.078650737e-13 |
| F1 | 6.620415605e-14 | 6.347082459e-14 | 6.347720334e-14 |
| R4 | 3.589109754e-16 | 5.523329131e-16 | 5.528097987e-16 |

S1/F1の旧bound超過は高精度では再現せず、通常floatのoperator評価側の数値誤差と整合した。演算・蓄積・normの寄与別ablationではない。R4のD近似差も旧約1.57e-14に対し約2.30e-19で、小残差を旧精度だけで解釈できないことを示した。

高精度化しても保存モデルのH identity差約4.40e-17、R Hermiticity差約4.61e-18、tail normの厳密な1との差は保持する。丸め方向を保証した包含ではなく、**`EMPIRICAL_CONSISTENCY`、`u_num=null`、`certified=false`**。旧22行を書き換えず、shot評価でもA0値を選択的に主biasへ差し替えていない。[A0 §4–5][HP]、[Shot][SHOT]

### B0：相対位相を保つfull wrapperの成立

B0はD/R stage、RTE event、scalar phase、controlled evolutionを、mode0準備＋ancilla＋X/Y終端＋measurement込みのwrapperへ接続した。System4×4、controlled/wrapper8×8を独立数学列へ照合し、phase fitなし・complex128・operator spectral atol1e-12／rtol0を維持した。[B0][NATIVE]

初期attemptはcompiled operator不一致で停止した。保存診断を追加して公式gate展開・phase集計を修正し、最終v5ではRY/Uの微小角を保つ恒等式とnative1q determinant由来の離散phase集計を用いた。同じD1回路で従来／balanced評価の誤差は修正後3.820996400e-13／5.210662172e-13となった。許容差緩和やfull operatorへの位相合わせはしていない。修正の個別寄与の完全なablationやγ証明ではない。[B0「再開後の独立D1診断」][NATIVE]

最終SOURCE `b190fe28b98d2f94e1509e19cd3d36a63b2bb99e` の全12 case・48代表で288 operator／192 expectationが通過し、その後96 wrapper費用を保存した。Compiled operator最大4.426725459e-13、compiled signal最大2.504663144e-13。各候補の代表RZ範囲はB3 3994–4070、D1 27322–27398、R2 27346–27494、R4 27322–27398、B4 2020–2066、F1 10624–10822。決定論的代表であり、期待値・全support上界ではない。

旧失敗・STOPは履歴として保全されている。A0報告の「B0は未実行」やB0旧履歴の「未解決」はその時点の記録で、**現在のB0はCOMPLETE**。後続B1はv5を使い、この実装問題を未解決のまま費用測定へ進めたものではない。

### B1：期待費用の有限標本と、残ったcoverage

B1は各candidate/Tにつき256 IID trajectory、全3,072本。X/Yは同じ古典trajectoryを使うpaired 6,144 wrapperで、独立512本/caseではない。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、approximation1、`qubits_initially_zero=False`、topologyなし、angle-preserving native-v5を固定した。State preparation・control・measurement込みであり、T-count／実機wall timeの価格ではない。[B1][MC]

以下はT=+0.18・X軸・full wrapper単価の既報mean±SE。Yは同じ表示値、負T・全metricは原報告にある。後続では全標本がS0だった点を踏まえ、S0条件付き参考統計として利用する。

| ID | RZ mean ± SE | CX mean ± SE |
|---|---:|---:|
| B3 | 4096.89453 ± 1.59814 | 2788.84375 ± 1.26439 |
| D1 | 27423.63281 ± 1.65513 | 18772.04688 ± 1.29393 |
| R2 | 27548.05078 ± 2.32664 | 18868.60937 ± 1.80706 |
| R4 | 27424.37109 ± 1.70766 | 18772.65625 ± 1.34921 |
| B4 | 2087.82422 ± 1.42353 | 1416.09375 ± 1.13304 |
| F1 | 10896.96484 ± 2.53092 | 7429.21875 ± 2.02413 |

原報告の参考normal区間では両T・両axisのRZ/CXについて `B4 < B3 < F1 < {D1,R4} < R2` の群間差が分離し、D1/R4は区間が重なった。これは既報比較の引用で、母平均の保証付き順位や本書の新ランキングではない。SE最大約0.0800%は観測標本の指標で、未観測tailを覆わない。

全18,432 operator／12,288 expectationを通過し、compiled operator最大4.550538909e-13。一方、51,200 short-stepは全てorder0でorder2/4未観測。B0には高次数を含む決定論的代表があったが、rare条件付き無作為費用標本を供給したものではない。B1の科学状態は **`INCONCLUSIVE_RARE_EVENT_COVERAGE`** である。[B1「Rare event coverage」][MC]、[Rare「B0の代表費用」][RARE]

古典実行は32 processで並列化し、wall50分46.95秒、平均30.48コア相当、実transpile6,078／cache hit66だった。B0の逐次pilot（wall11分13.74秒）とは仕事量が違うので速度倍率は出していない。この実績は継続時のcompile負担の参考になるが、分子系の時間見積りではない。

## 9. Rare感度とshot加重：何を条件付きで会計したか

### 未観測高次数を未知費用に残す

S0は「全short-stepがorder0」、rareは「少なくとも一つ非0 event」で、K4最高次数だけを指さない。最小rare解析はB3/B4/D1/R4の両T、保存lawの確率・既存S0統計を用い、未知 `u_i=E[C_i|rare]` を非負変数として残した。費用差のaffine境界と±1/2SE矩形感度を保存した。[Rare][RARE]

既報の単価RZ境界は、+Tで `u_B4*=1.43828287e8+13.09810605*u_B3`、`u_D1*=4035.81150+1.00000267647*u_R4`。境界は経験S0平均固定下の等費用線で、rare費用の推定ではない。B4/B3のS0差は±2SEでも残り、D1/R4は±1SEで符号が変わった。有限上界のない未知費用を残すため、無条件母平均順位は確定しない。状態は **`CONDITIONAL_SENSITIVITY_ONLY`**。

追加rare sampling／compileは実施せず、現在の条件付き主張のために自動的に必須とはしない判断を原報告が記録している。この保留をcoverage保証取得と読み替えない。

### Shot24条件のモデルと精度資格

最終評価はSOURCE `9bebdc46ebd60039f315da9b6c6e57bd9c004b85`。主biasは22行の共通corrected physical signal直接誤差。各axisのoutcome±1、**quantum shotごとにfresh IID full trajectory**を仮定し、normalizationの上界Uを用いた。実際の量子shot実行は0件。[Shot][SHOT]、[固定設計][SHOTDESIGN]

予算は `b+u_num≤ε/2`、`gamma≤ε/10`、`s=2ε/5`、`alpha_X=alpha_Y=1/40`。既報式は `m_a=ceil(4 U²/s² ln(2/alpha_a))`。95%は単一(candidate,T,ε)のX/Y union boundで、全candidate/T/εのfamilywise保証は未配分である。十分数であり最適shot数ではない。

保存N_usedと、signed binary64 τ・有限paired Taylorに対する90dps outward scalar区間[N_lo,N_hi]を分け、`U=max(N_hi,abs(N_used))`を使った。Rescaling biasのscalar上界も別記録した。この限定scalar enclosureはsignal全体のu_numやfinite-bit sampler／native実装のγ保証ではない。

| ID（両T・各axis） | m_axis：ε=1e-8 | m_axis：ε=1e-9 | p_rare（表示丸め） |
|---|---:|---:|---:|
| B3 | 1096308481839662540 | 109630848183966253904 | 1.82895756403e-4 |
| D1 | 1096308481839662540 | 109630848183966253904 | 1.82895756403e-4 |
| R2 | 1095907502690109238 | 109590750269010923789 | 9.14535898003e-5 |
| R4 | 1096308482913175258 | 109630848291317525775 | 1.82896245917e-4 |
| B4 | 1095567849372165468 | 109556784937216546703 | 1.39635269131e-5 |
| F1 | 1095521956053880197 | 109552195605388019638 | 3.49090794495e-6 |

上表は[最終報告][SHOT]からの既存整数・表示値の引用。X+Y合計ではない。Exact Fraction端点がscalar上界の正本で、表の丸め値には上界性を主張しない。R2/F1のrare確率も最終評価で取得済みの値である。

観測bだけの点予算では、ε=1e-8は全6候補・両Tが以内。ε=1e-9はD1/R2/R4が以内、B3/B4/F1が超過した。しかし全24条件で `u_num=null`、`gamma=null`、`end_to_end_qualified=null`。Budget外の参考費用を、同精度達成候補の価格とはしない。

### 単価とshotを接続したS0参考費用

既報の `G_S0=m_X*mean_X+m_Y*mean_Y` を引用する。以下はT=+0.18、ε=1e-8、両axis合計の表示値。全144行はRZ/CX count・各depth・total depth・sizeを含む。[Shot「S0条件付きshot加重資源」][SHOT]

| ID | G_S0 RZ | SE_S0 RZ | G_S0 CX | SE_S0 CX |
|---|---:|---:|---:|---:|
| B3 | 8.9829204e21 | 3.50411e18 | 6.1148661e21 | 2.77232e18 |
| D1 | 6.0129523e22 | 3.62906e18 | 4.1159908e22 | 2.83709e18 |
| R2 | 6.0380231e22 | 5.09956e18 | 4.1356501e22 | 3.96073e18 |
| R4 | 6.0131141e22 | 3.74425e18 | 4.1161245e22 | 2.95830e18 |
| B4 | 4.5747062e21 | 3.11914e18 | 3.1028536e21 | 2.48265e18 |
| F1 | 2.3875728e22 | 5.54536e18 | 1.6277745e22 | 4.43497e18 |

Cost MCのSEは同じtrajectoryのX/Y covarianceを保持したn256の値。Quantum measurementの95%予算とは別で、SE欄はconfidence intervalではない。Depth加重値は逐次work proxyで実機wall timeではない。大きなshot数・Gは上記の厳しいsignal精度と保守的なbounded-outcomeモデルでの会計であり、実行可能なbackend資源やエネルギー精度達成を保証しない。

最終会計も `G_hat(u)=(1-p)G_S0+p*m_X*u_X+p*m_Y*u_Y` と未知rare平均を残す。真のGにはS0母平均と経験平均との差もある。`unconditional_G/SE/winner`はnullである。

原報告の24比較では、RZ/CX・両T・両εにおいてB4/B3とD1/R2のS0経験寄与差の低費用方向が±2SE矩形でも残り、D1/R4は±1SEで符号未確定だった。R2のN低下によるshot減少は、その保存S0単価増を相殺しなかったという限定結果である。候補間独立性や共分散ゼロを仮定した有意差検定ではない。

Shot加重後の+T・ε=1e-8・RZ境界は、B3/B4が切片1.4402660e8・傾き1.31069607161e1、R4/D1が4.0359583e3・1.00000267744、R2/D1が6.3883046e5・4.99848340320e-1。ここで未知費用は各候補のshotで加重したrare両axis平均である。**単価だけの旧境界とは係数・定義が異なる。** 詳細式・符号規約は[最終報告][SHOT]参照。新しい差分・比・境界を本書で計算していない。

## 10. 科学的知見をどこまで統合できるか

**内部高次化の価値はfull taskで評価する必要がある。** D単独誤差の小ささだけではfull signal利得や低費用を決められなかった。36行のA3/A4識別不能とprocessor構造費は、短いD runでYP8m8独自の利得を確認できなかったという利用可能な限定結果である。YP8m8一般の無効性を意味しない。

**誤差を小さくする層は条件で変わる。** Outer変更でfull signalが動き、その後A3対照のr/K変更でも動いた。一方A2ではD近似が残った。各段階内の対照から得た知見であり、未実行条件の支配項や一般的な最適設定を推定するものではない。

**Corrected mean、raw信号、量子shot、古典費用推定は異なる。** Normalizationを戻すとmeanの比較は改善し得るが、統計range・shot負担とnative単価を払う。B1の費用MC標本数は、quantum shot数ではない。Full prepared wrapperとprep除外RPE指標も混ぜない。

**精度改善と資源改善は一致するとは限らない。** A3は内部誤差対照として有用だった一方で長いD列を持つ。r増加・K変更・refinementも精度・normalization・event費を同時に変える。B4/B3、D1/R2、D1/R4については既報の条件付き比較を利用できるが、全候補の同精度最小費用・最適性は決まっていない。

**未観測tailの扱いが主張範囲を決める。** S0標本の小さいSE、B0の数代表、高次数の小確率だけでは無条件期待費用を保証できない。未知費用を変数に残した感度は、追加samplingをしなくても現在言えることを明確にする成果である。[D36][D36]、[構造費][STRUCT]、[R22][R22]、[B1][MC]、[Rare][RARE]、[Shot][SHOT]

## 11. 主張の強さの分類

| 分類 | 利用できる内容 | 条件・留保 |
|---|---|---|
| A：固定条件で確認 | 36/16/22行のoperator・corrected signal値、入力非可換性、A4数学／published列照合 | Synthetic・固定窓の経験的評価。一般保証・収束次数ではない |
| A：固定条件で確認 | A0の90/128桁整合、B0/B1実現回路のcorrectness、B1の標本統計 | 高精度一致は包含ではなく、実現回路PASSは全support／γ保証ではない |
| A：固定モデルで評価 | Scalar normalization区間と解析的な十分shot整数 | 指定IID・bounded outcome・confidence配分の条件下。End-to-end保証ではない |
| B：条件付き支持 | S0単価／shot加重会計、未知rare費用のaffine感度、±SE矩形での方向 | S0経験平均・unknown rare・統計モデルを明記。母平均順位やCIではない |
| C：未確定 | Certified u_num、γ・finite-bit law、rareを含む無条件期待費用、費用母平均順位・同時保証 | Nullを維持。未取得を0や既存atolで埋めない |
| C：未確定 | 同精度end-to-end資源優位、分子系・全RPEへの一般化、全候補最適性、一般誤差保証、先行研究に対する新規性 | 本統合による正式採択・GO認定なし |
| D：現在保留 | A1数値包含、追加rare sampling/compile、大規模全support費用、分子系、energy/RPE拡張 | 個別主張の必要性と追加許可をユーザーが判断 |

実行COMPLETEと科学的GO、経験的PASSとcertificate、有限標本平均と母平均、条件付きGと無条件Gを区別する。最終のINCONCLUSIVEは、固定計算の未完了や方式の科学的FAILを意味しない。

## 12. 未解決事項と、追加証拠で変わる主張

| 未解決事項 | 現在ある根拠 | 主張を強めるために必要な証拠 |
|---|---|---|
| u_num | A0高精度の経験的整合、scalar N区間 | 対象を限定したsignal包含。入力丸め・H identity／Hermiticity差・積・exponential・norm等の誤差会計 |
| γ | 実現回路のatol内照合、v5のphase保持 | Target mean対実装meanの差。有限bit確率・角度・normalization・合成誤差を必要範囲で含める |
| 無条件費用 | B1 S0標本とp_rare、unknown affine式 | Rare条件付き費用推定／上界等の根拠と、その不確かさ。B0代表最大を上界にしない |
| Cost MCと候補選択 | Mean/SE、矩形感度、X/Y covariance | 比較に必要な費用不確かさと同時confidence／選択規則。量子測定のαと別管理 |
| 同精度総資源 | 観測bias予算と参考shots×S0費 | 上記の精度・実装・母平均費用資格を同じtaskへ接続。Budget外候補は再認定しない |
| 実分子・エネルギー | 分子DF／sector／RPE既存pipeline | 対象Hamiltonian、状態・overlap、DF誤差、全時間・全roundの精度と費用 |

これは不足証拠の整理で、新しい保証方式の採用や実行契約ではない。全項目を取得しないと既存限定結果を報告できないという意味でもない。

## 13. 既存成果だけで利用できる研究資料

| 利用先 | 既存証拠で構成できる内容 | 併記する制約 |
|---|---|---|
| 技術報告 | 三層の誤差診断、A4端点構造、native接続、S0条件付き費用とrare感度 | 固定synthetic model・限定時間・経験的／条件付き評価 |
| 論文の限定結果候補 | 既知高次PFのPR接続、outer/RTE対照、phase保持実装の検証、条件付き会計 | 新規性・掲載可能性・中心主張の正式判断は別の科学レビュー／ユーザー判断 |
| 発表・研究概要 | 本書の条件表・主要結果表、各原報告の比較表・correctness／資源実績 | 図表を使う場合も元条件・SOURCE・単位・未確定資格を付す |
| 再利用可能な実装資産 | Ordered D/outer plan、RTE law、native-v5、paired MC、scalar intervalと会計 | 固定fixture用adapterを一般分子対応済みとはしない |

本書の候補主張は「この固定範囲で、局所精度改善の残り方と費用上の留保を対照実験・条件付き会計で示した」までとする。「新算法の新規性が確立」「論文化可能」「SABA2／Yoshida4／YP8m8が最良」は正式採択しない。

追加数値を取得せず、現在の経験的結果と未取得保証を明示して研究を一区切りにできる。発表用には既存報告の数値表を引用し、新しい科学的再集計・図の再描画は本作業で行っていない。

## 14. 今後の発展候補A〜Eと再利用可能な技術

### 方向A：現在の限定研究を一区切りにする

本書と8段階の固定報告・原本を、誤差の層別診断と条件付き資源評価として利用する。保証付き総資源優位を中心主張にしなければ、u_num／γ／rareを未確定のまま明示できる。追加科学計算は必要ない。新規性や論文中心主張の判断を求める場合だけ、別途そのレビューを依頼する。

### 方向B：実分子へ展開する

固定起点のSOURCEを静的に確認した結果、再利用候補は以下である。今回はimport・関数呼出し・Hamiltonian生成・回路生成をしていない。

| 既存資産 | 確認した接口・役割 | Syntheticから変更・検証する点 |
|---|---|---|
| [df_hamiltonian.py][DFHAM] | `build_df_h_d_from_molecule`、`df_hamiltonian_from_integrals`、`PhysicalSector`、`solve_df_ground_state`、`df_linear_operator` | Geometry/basis/charge/spin、DF rank/tolerance、切捨て誤差、入力stateとsector。旧snapshotとdefault rankを同一視しない |
| [df_partial_s2.py][DFPREP] | `prepare_df_partial_s2`、ordered DF block／Gaussian basis／tail preparation | Split・basis・scalar identityを固定。高次D／signed outerへの接続と非可換性を再検証 |
| [finite_rte_signal_validation.py][SIGNALS]、[pf_c_system_size_validation.py][SIZE] | 小sector dense参照、state-action／ground residual・PF signal診断 | Dense参照をそのまま大型化せず、適切なsector／matrix-free参照を選択。既存S2結果を高次outerへ移送しない |
| [df_rte_qiskit.py][DFRTE]、[df_partial_s2_repeated.py][REPEAT] | DF eventのsigned control／phase、既存反復builder | Bのheterogeneous outer occurrenceへadapterが必要。共通stepのattenuationをそのまま使わない |
| [b2d_evaluator/input_package.py][INPUT]、[b0_native.py][B0SOURCE]、[b1_native.py][B1SOURCE] | 現Bの入力identity、独立参照・native描画、paired MC・cache・process並列化 | 現入力は2mode/N1/LD2に固定。B0独立CAR・4×4入力、state[1]、因子0/1/2の直指定も一般化が必要 |
| [rte_order_stratified_cost_validation.py][STRATA]、[connected-cluster cost][CLUSTER] | 次数条件付き費用評価、snapshotを固定した較正／holdoutの構造 | 既存経路はK2・order0/2が前提で、R4/K4へ直接流用できない。Signed多stage・full prepared wrapperへの適合と新較正も必要 |

したがって「分子Hamiltonian基盤がある」ことと「Bの分子実行経路が完成している」ことは異なる。まずモデル・DF snapshot・L_D・sector/state・比較PF・T・誤差の意味を限定し、量子state preparationの価格と古典参照の作り方を決める必要がある。N1/mode0だけの今回の結果を、相互作用する多電子分子の性能根拠へ一般化しない。

分子系では準備する状態のoverlap、DF切捨て・丸め、sector規約、compilerとcontrol、行列／回路サイズ、rare supportが追加論点となる。既存分子snapshotを再利用する場合も正確なidentityを確認し、DFを再生成しただけで同じ入力と扱わない。実行資源・所要時間は未評価。重い段階へ進むなら利用可能CPU/RAM/GPUを調べ、独立条件の並列化・既存matrix-free/cacheを選ぶ。今回のB1実績を分子系speedup予測には使わない。

### 方向C：保証付き同精度資源比較を目指す

§12のうち、ユーザーが必要とする主張に対応する証拠だけを追加する。A1は少数signalの数値包含、γはfinite targetと実装meanの差、rare対応は費用母平均／上界、cost MCは費用推定の不確かさを扱い、別の問題として接続する。Confidenceも単一X/Y、費用推定、候補選択・多条件を区別する。

既存[scalar intervalを使うshot接続][SHOTSOURCE]はnormalization／log／ceil会計の再利用資産だが、matrix signal包含の代替ではない。B0/B1のv5・correctnessもγの証明を供給しない。必要な精度／費用保証が揃った条件について初めて、保証付き同精度会計や母平均比較を検討できる。全候補最適性や全分子保証はそれでも別の証拠を要する。

### 方向D：既存方式の組合せと資源条件を改善する

既存のD内部A2/A3、outer-S2/refined/Yoshida4/SABA2、r/K、n_D/qを組み合わせる候補は、ordered plan・signed RTE・native描画を再利用できる。検討点は、D近似を十分小さくする条件、negative-timeを含むR絶対時間とN、低費用outerのbias、目標εごとの精度予算と費用の競合である。[Outer SOURCE][OUTERSOURCE]、[構造費][STRUCT]、[R22][R22]

今回のB4はε=1e-8で観測bias予算内だったがε=1e-9では超過した。D1/R2/R4もε適格性保証は未取得。したがって既報単価順をそのまま新しい目標精度の最適方式にしない。A4の長run用途、低費用outerの別設定などは探索候補であり、新PF係数設計・最適化・数学的正当性・新規性を本書で認定しない。新しいparameter gridや保存値の新しい最適化にも別許可が必要となる。

### 方向E：量子化学エネルギー推定へ接続する

[rpe_resource_accounting.py][RPERES]にはfresh-IID／attenuation／round配分と資格区分、[df_rpe_hadamard_compiled_cost.py][RPECOST]には短roundのpaired interrogation費用接口がある。ただし既存の費用範囲はstate preparationなしで、今回のprepared wrapper・異種outerにはそのまま対応しない。

単一signalのεを化学的精度のenergy誤差へ置き換えず、対象state・固有位相分枝・overlap・半径下界、alias／branch selection、全roundの時間とconfidence、state preparation・measurement・必要なoracleを明示する。既存のH-chain/RPE検証は再利用候補の技術基盤で、Bの全RPE評価済み証拠ではない。実機noise、Clifford+T／physical qubit価格も今回のnative RZ/CXからは得られない。

### 発展候補の比較表

負担はSOURCE・既存実績からの定性的整理で、計算時間・RAM・回路数の新しい見積りはしていない。

| 方向 | 既存成果の再利用 | 新たに必要な証拠 | 技術的負担 | 科学的判断 | 実行許可 |
|---|---|---|---|---|---|
| A：一区切り | 本統合・8報告・既存表と原本 | 限定報告なら追加科学証拠不要 | 文書編集中心。最も着手しやすい | 経験的・条件付き到達点を成果範囲とするか | 追加計算なし。詳細科学レビューは別開始許可 |
| B：実分子 | DF／sector／matrix-free／Gaussian／native基盤 | 新snapshot・多電子state・分割・全correctness・精度費用 | 入力とB0独立参照の変更が必要。規模未評価 | どの分子・状態・精度を代表taskにするか | 新分子計算・build/compile・評価を固定して別許可 |
| C：保証比較 | A0・scalar interval・v5・S0統計／rare式 | u_num、γ、rare費用、費用統計と同時保証 | 主張範囲次第。包含／実装law／費用の接続が必要 | 保証が研究結論に必要か、どこまで限定するか | A1／γ／追加費用評価は別許可 |
| D：方式改善 | Ordered outer/D plan、r/K law、native描画 | 新設定のfairな精度・N・単価・shot比較 | 既存方式なら再利用度は高い。探索規模未評価 | 既知法の応用改善か、別原理を求めるか | 新grid／PF探索／新再解析は別許可。新規性判断も別 |
| E：energy/RPE | RPE accounting・短round費用接口 | State/phase/radius・全round誤差／成功率・準備を含む費用 | Task定義と複数層の接続が必要。規模未評価 | 単一signalからenergyへ目標を広げるか | 新energy/RPE科学評価は別許可 |

技術的にすぐ利用できるのはAの限定報告である。B〜Eへ進む場合は、得たい科学的主張を先に選ぶことで追加証拠を絞れる。これは選択肢の提案で、優先研究方針の正式変更ではない。

## 15. ユーザーの次の判断と停止点

ユーザーが判断する事項は、(1)現在の条件付き到達点で一区切りにするか、(2)継続する場合に分子適用・保証・方式改善・energy接続のどの主張を優先するか、(3)独立科学レビューや論文中心主張の検討を別途依頼するか、である。既知法の限定応用結果としてまとめることと、独立新算法の新規性を要求することも分ける。

Codexの今回の担当は本書と索引の最小更新、文書検査、commit/push、remote SHA・公開内容照合まで。新しいSOURCE候補や科学実行契約は作らず、科学SOURCE・入力・raw・過去のgrant/ledger/marker/STOP・共通指示を保全する。

**公開確認後mandatory STOP。** ユーザーによる次の対象指定・必要な実行許可までは、新規計算、保存値再解析、MC、A1包含、γ評価、分子検証、PF探索、詳細科学レビュー、論文最終採択を開始しない。通常の研究Bの統合報告として、研究統括への送信は行わない。

## 16. SOURCE・公開結果・確認範囲

### 一次報告と実行SOURCE

下記A〜Hを指定GitHub commitから実際に取得した。各公開報告のblobは統合作業起点の同じpathと一致することを確認し、本文を根拠に数値を引用した。関連する静的構造費報告も公開版と起点SOURCEを参照した。結果公開commitは科学実行SOURCEではない。

| 証拠 | 科学／解析SOURCE | 公開結果commit・報告 |
|---|---|---|
| A：D内部36行 | `758a05ed6aac58a5945af5a5330c792a842b174d` | `51b1387996b1b53feb550efa7cb98aadcf5f9189` [報告][D36] |
| 関連：静的構造費 | 既存SOURCEのdescriptor整理、科学実行なし | `3f5d60122f158b3320fadab86498963950f00734` [報告][STRUCT] |
| B：Outer16行 | `b85dd328423de5b1159f11939835f04db90f3187` | `b1d9f46ee583d86666deb433a4fc4c20dad20225` [報告][O16] |
| C：Outer/RTE22行 | `26e8734aff3969dc8077cb058826d12394be28e1` | `33715848bfb83d49ec4e54d6eef481012a7441e4` [報告][R22] |
| D：A0高精度 | `1e3921ba16e5d531e9a3010cffa9ff717609a93a` | `31662b49157b15f2d1c03eef1c2faa07e2f905e7` [報告][HP] |
| E：B0完了 | `b190fe28b98d2f94e1509e19cd3d36a63b2bb99e` | `0db9b3ed189eb6cd9d66e070993dbd0e72d29104` [復旧・完了報告][NATIVE] |
| F：B1 IID | `11e2532a62d54a65f8fad5489be6b135f826011b` | `bef63edbd8cc564444b44e4e8cf23baa90f05fe5` [報告][MC] |
| G：Rare感度 | `a8cf99156af692f0facaa1885ab9ed52b28c6f4c` | `cae3cd784dbdfcf2fe96902345379aca4b9606c7` [報告][RARE] |
| H：Shot24条件 | `9bebdc46ebd60039f315da9b6c6e57bd9c004b85` | `e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d` [報告][SHOT] |

各原報告からresult/CSV・入力・環境・実行履歴・STOPへたどれる。数値的に不明な箇所を埋める新解析は必要とせず、今回rawの新しい差分・比率・順位・統計を計算していない。数値表は上記報告の掲載値を引用し、表示丸めと精度間の違いを維持した。

全体の背景には[研究概要・現状][SYNTHESIS]を全文参照し、[研究目的][PURPOSE]、[VALIDATION_STATUS][STATUS]、[manifest][MANIFEST]の研究B各entryと照合した。旧未完了entryを削除・更新せず、B0 completeなど後続の証拠を優先した。初期調査・B2設計は目的と対象の確認に用い、過去の候補提案を今回の正式採択とは扱っていない。

SOURCE確認は§14の関連関数・固定fixture制約・費用scopeの静的読取りに限定した。科学moduleのimport、数値テスト、runnerのinspect/preflight/execute、依存installは行わない。既存の文書・SOURCE・索引を直接利用し、新しい生成frameworkや承認証明資料は作らない。必要な確認はリンク先、引用した値と条件、差分が本書・索引だけであることに限る。科学的status・manifest・旧報告・研究記録の本文は変更しない。

[D36]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/docs/research/b2d_stage1_execution_20261025.md
[STRUCT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f5d60122f158b3320fadab86498963950f00734/docs/research/research_b_post_b2d_stage1_static_cost_outer_splitting_20261025.md
[O16]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1d9f46ee583d86666deb433a4fc4c20dad20225/docs/research/research_b_outer_splitting_stage1_execution_20261025.md
[R22]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/docs/research/research_b_outer_rte_22row_execution_20261025.md
[HP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31662b49157b15f2d1c03eef1c2faa07e2f905e7/docs/research/research_b_outer_a0_execution_20261025.md
[NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0db9b3ed189eb6cd9d66e070993dbd0e72d29104/docs/research/research_b_outer_b0_recovery_20261025.md
[MC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bef63edbd8cc564444b44e4e8cf23baa90f05fe5/docs/research/research_b_outer_b1_expected_native_cost_20261025.md
[RARE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cae3cd784dbdfcf2fe96902345379aca4b9606c7/docs/research/research_b_outer_rare_event_minimal_sensitivity_20261025.md
[SHOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/docs/research/research_b_outer_shot_weighted_execution_20261025.md
[INITIAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/docs/research/research_b_time_evolution_initial_survey_20261024.md
[B2DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/docs/research/research_b_b2_limited_design_20261024.md
[PURPOSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/docs/research/研究目的・研究課題.md
[SYNTHESIS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/docs/research/研究概要・現状.md
[STATUS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/VALIDATION_STATUS.md
[MANIFEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/artifacts/validation_manifest.json
[SHOTDESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/docs/research/research_b_outer_shot_weighted_resource_design_20261025.md
[DFHAM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trotterlib/df_hamiltonian.py
[DFPREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trotterlib/df_partial_s2.py
[SIGNALS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trotterlib/finite_rte_signal_validation.py
[SIZE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trotterlib/pf_c_system_size_validation.py
[DFRTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trotterlib/df_rte_qiskit.py
[REPEAT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trotterlib/df_partial_s2_repeated.py
[INPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trottertracks/b2d_evaluator/input_package.py
[B0SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trottertracks/b2d_evaluator/b0_native.py
[B1SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trottertracks/b2d_evaluator/b1_native.py
[STRATA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trotterlib/rte_order_stratified_cost_validation.py
[CLUSTER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trotterlib/rte_connected_cluster_cost_validation.py
[SHOTSOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trottertracks/b2d_evaluator/shot_weighted.py
[OUTERSOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trottertracks/b2d_evaluator/outer.py
[RPERES]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trotterlib/rpe_resource_accounting.py
[RPECOST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/src/trotterlib/df_rpe_hadamard_compiled_cost.py
