# G6全生成手順の逐次対応表

2026-10-18 JST。**P1〜P12×7経路＝84セル**。元のM1〜M12 Evidence Matrixは参照し、再作成しない。以下は文献・SOURCEに明示された処理と既存恒等式の対応で、新しい同値性・非同値性証明ではない。

statusは、その行で明記した部品に対する判定である。`DIRECTLY_DOCUMENTED`でも全algorithmの一致を意味しない。`NOT_ESTABLISHED`は確認した固定節の範囲で対応根拠が不足することを意味し、文献全体・全関連研究に処理が存在しないという不在証明ではない。

| status | 意味 |
|---|---|
| DIRECTLY_DOCUMENTED | その処理が一次本文又は固定SOURCEに明示 |
| ROUTINE_REPARAMETERIZATION | 既存の式・変数・有限cutoffの直接対応。新非自明定理なし |
| PARTIAL_OVERLAP | 部品が重なるが全law/保証/取得の一致は未確認 |
| NOT_ESTABLISHED | 対応の根拠不足 |
| DIFFERENT_TARGET_OR_ACCESS | target、process、oracle又はprecision条件が直接一致しない |
| REQUIRES_NEW_SCIENTIFIC_PROOF | 非自明な全joint-law還元・非還元等が必要となる場合。今回証明せず返す |

## 1. G6/G7の既存処理と位置

| 段階 | 既存処理 | 参照M | 固定SOURCE位置 |
|---|---|---|---|
| P1 | 正の有理p、Q²=I、x/m/σとcoherent finite targetを固定 | M1/M12 | K __init__ 138–146; MATH §1; BIT §1 |
| P2 | 切断Green F/G係数を取得 | M2 | K 138–160; MATH §5 |
| P3 | 指定reduced uのa_uと全childを局所query | M1/M3 | K 162–202; BIT §2 |
| P4 | b_l/B_oの次数とIID pのraw proposal | M6 | K 204–212; G 216–225,237–239 |
| P5 | 非reduced rawは量子前zero、別parentへ再配分しない | M6/M7 | K 11–18; G 240–241; MATH §8 |
| P6 | a/s/d、local acceptance、child lawを接続 | M4/M5/M6 | K 114–133,194–202,214–236; G 242–251 |
| P7 | ordered Q(u)、suffix依存角s/a、parent phaseを返す | M5/M12 | K 229–236; G 79–90; CP 2–34 |
| P8 | dyadic実πと近似c̃を使いW̃=c̃/πで補正 | M9 | K 45–110,214–254; G 242–251; BIT §§3,5 |
| P9 | ideal Mとdigital M̃、bias/range/momentを分離 | M7/M9/M10 | MATH §8; BIT §§4–6; U §7.4 |
| P10 | B_RA不要のUと外向き区間、十分全trial数を会計 | M8/M10 | G 19–41,266–284; U §7 |
| P11 | setup/local算術、係数bit長、selector等を分離 | M3/M11 | BIT §§2,5; K/G selector; C04 |
| P12 | controlled-Q・実adjoint・helper・strict native条件 | M12 | CP 2–58; BIT §6; N04 |

K/MATH/BITは当初G6、G/CPとUは後続G7/既存公開U保証である。Uを当初G6証明へ遡及して含めない。G6単体は局所packet、G7はbits interfaceからevent/Noneを返す。fresh uniform bitsのモデルとDeterministicBits診断を区別する。

## 2. 行の読み方と共通条件

各行は、上のG6処理・M番号・SOURCE位置、下の「既知処理/一致条件/一次locator」、[access/精度/費用profile](access_precision_complexity_comparison.md)の同名方式を合わせた一つの対応recordである。10項目（target/output、共通oracle、前処理/normalizer/取得、有限bit/bias/range/confidence、controlled位相、計算量、出典、不足）を[evidence_index.json](evidence_index.json)の各セルに明示的に展開している。

WBCをWAN、Zhao–YuanをMTと略す。PTSCは別経路として分け、計7経路を扱う。A0/A1/A2は全方式で公平に認める。未知oracle・未取得費用は無料にしない。理想event/同じ有限meanと実digital lawは別々に確認する。

## WAN：Wan–Berta–Campbell ordinary

| 段階 | 対応status | 一次資料の実際の処理 | 同じ条件への対応／残る不足 | 一次locator |
|---|---|---|---|---|
| P1 | `DIFFERENT_TARGET_OR_ACCESS` | Pauli凸結合を入力 | 一般Qの代数適用と安価なPauli回路accessは分ける。A0/A1を同等に与える | [WAN] App.C Lemma2, Algorithm2 input |
| P2 | `NOT_ESTABLISHED` | Green return queryの指定なし | ordinaryの係数取得と、return収集用F/Gは別。A1を追加可能だが文献記載としない | [WAN] App.C (C4)–(C5) |
| P3 | `PARTIAL_OVERLAP` | 次数とIID labelから非列挙 | 非列挙性は既知。指定uのa_u/child queryまで同じではない | [WAN] Algorithm2 steps3–4 |
| P4 | `ROUTINE_REPARAMETERIZATION` | even次数重みとIID p | 1 microstep、n≤m−1へ切断し、scalar b_n/B_oを対応。全G6 lawの一致ではない | [WAN] (C4)–(C5), Algorithm2, App.E.2 |
| P5 | `NOT_ESTABLISHED` | raw語を適用するordinary経路 | 非reduced rawを量子前zeroにするG6の規則は確認できない | [WAN] Algorithm2 steps4–7 |
| P6 | `PARTIAL_OVERLAP` | 隣接Taylor次数のEuler pairing | 次数角は既知。return後a_u/s_u、acceptance、child条件lawは未対応 | [WAN] (C4)–(C5) |
| P7 | `PARTIAL_OVERLAP` | Pauli積・回転・scalar位相を列化 | 原文のrotation位置とG6のleft rotationを無断に同一視しない。個別event全写像は未確定 | [WAN] (C4), Algorithm2 steps5–8 |
| P8 | `NOT_ESTABLISHED` | 理想次数/label確率のsampling | G6のroot区間・実dyadic π・係数近似補正の同じ保証は未確認 | [WAN] Algorithm2; App.E.2 |
| P9 | `PARTIAL_OVERLAP` | truncation biasとbounded estimator | 同じfinite cutoffを指定できるが、G6 M̃とrange/momentの保証は別 | [WAN] App.E.1–E.2 |
| P10 | `PARTIAL_OVERLAP` | LCU weightに基づくsample会計 | Uと事前zeroによる全trial/NZまで同じ予算は確認できない | [WAN] Lemma2; App.E.1–E.2 |
| P11 | `PARTIAL_OVERLAP` | 効率的な次数/label draw | 全辞書列挙へ制限しない。Green局所query・bit/selectorを含む全費用分離は未対応 | [WAN] Algorithm2; App.E.2 |
| P12 | `DIFFERENT_TARGET_OR_ACCESS` | Pauli用controlled実装 | generic CQ providerへ同条件で戻す必要。共有角合成も対照へ認める | [WAN] App.C (C5)後; App.E.3 |

## AK：Aomoto–Kato Green

| 段階 | 対応status | 一次資料の実際の処理 | 同じ条件への対応／残る不足 | 一次locator |
|---|---|---|---|---|
| P1 | `ROUTINE_REPARAMETERIZATION` | cyclic free productのwalk | Z2、p_bar=p/2、spectral変数の既存対応。quantum targetまでは含まない | [AK] §1 (1.1)–(1.8); MATH §5 |
| P2 | `ROUTINE_REPARAMETERIZATION` | Green multiplierとspectral shift | 既存M2の特殊化・形式切断として対応。新Green定理ではない | [AK] §1 (1.6)–(1.14), Lemma1.1 |
| P3 | `ROUTINE_REPARAMETERIZATION` | reduced pathのmultiplier積 | 指定word係数は既存M2/M3の切断・代入で対応。量子eventではない | [AK] (1.6)–(1.11); K word_series/_mass |
| P4 | `DIFFERENT_TARGET_OR_ACCESS` | Green/resolventの記述 | ordinary Taylor次数proposalの生成algorithmは対象外 | [AK] §1 pp.62–65 |
| P5 | `NOT_ESTABLISHED` | groupのminimal expression | raw rejectで全trialに0を記録する規則は記載確認できない | [AK] (1.6)とLemma1.1 |
| P6 | `NOT_ESTABLISHED` | return/first-passage情報 | a/s/dへのpositive配分、acceptanceとchild samplingは未確認 | [AK] §1 (1.6)–(1.14) |
| P7 | `DIFFERENT_TARGET_OR_ACCESS` | group積とresolvent | rotation角・controlled scalarを指定しない。演算子語をreorderする根拠にもならない | [AK] §1 pp.62–65 |
| P8 | `NOT_ESTABLISHED` | Greenの解析式 | dyadic proposal・局所root補正・実weightのalgorithmは未確認 | [AK] §1 pp.62–65 |
| P9 | `DIFFERENT_TARGET_OR_ACCESS` | Green係数とスペクトルが対象 | coherent mean、digital bias/range/momentは当該定理の出力でない | [AK] §1 Lemma1.1 |
| P10 | `NOT_ESTABLISHED` | resolventの正規化条件 | U、Bernstein trial、量子呼出し会計の対応は未確認 | [AK] §1 (1.12)–(1.14) |
| P11 | `PARTIAL_OVERLAP` | 係数を得る既知再帰 | G6の限定算術countはK/BITの既存証拠。AKがG6全bit/runtimeを解析したとはしない | [AK] Lemma1.1; BIT §2 |
| P12 | `DIFFERENT_TARGET_OR_ACCESS` | 量子providerを扱わない | 一般controlled-Qとnative費用は別途必要 | [AK] §1 pp.62–65 |

## MT：Zhao–Yuan modified Taylor

| 段階 | 対応status | 一次資料の実際の処理 | 同じ条件への対応／残る不足 | 一次locator |
|---|---|---|---|---|
| P1 | `DIFFERENT_TARGET_OR_ACCESS` | Pauliと追加可換関係を利用 | 一般involutionだけの入力では追加情報がない。合法情報は双方に許す | [MT] §2 (1)–(2), §4.2 |
| P2 | `NOT_ESTABLISHED` | 高次項のcancellation/吸収 | 自由積Green queryそのものの記載は未確認 | [MT] §4.2 (24)–(29) |
| P3 | `PARTIAL_OVERLAP` | 低次・identityに寄与を再配分 | 全formal uへの局所queryを同じ入力で与えることは未確認 | [MT] (25)–(28) |
| P4 | `DIFFERENT_TARGET_OR_ACCESS` | modified remainderのLCU化 | ordinary b_l/B_oを提案するG6とはtarget・selectorが同じでない | [MT] (28)–(29) |
| P5 | `NOT_ESTABLISHED` | 収集して係数へ吸収 | 非reduced rawを別parentへ流さずzeroにするlawは未確認 | [MT] (26)–(29) |
| P6 | `PARTIAL_OVERLAP` | identity/先頭項への吸収 | short-step全uの正値性、suffix childのa/s/d lawは未確認 | [MT] (26)–(28); MATH §§3–6 |
| P7 | `PARTIAL_OVERLAP` | phase付きunitary積をLCUに使う | G6のparent scalarとsingle-child rotationへの直接一致は未確認 | [MT] (25), (28)–(29) |
| P8 | `NOT_ESTABLISHED` | modified LCU係数 | 同じfinite Mの実πと有限root近似weightの保証は未確認 | [MT] §4.2; BIT §§3–5 |
| P9 | `DIFFERENT_TARGET_OR_ACCESS` | 高次tailを含むmodified target | 元の同じP_mに戻した全係数/meanを自動同値としない | [MT] (24), (28)–(29) |
| P10 | `DIFFERENT_TARGET_OR_ACCESS` | LCU誤差とnormalization | G6のUと全trial/事前zeroの予算とは直接同じでない | [MT] §4.2 (24)–(29) |
| P11 | `PARTIAL_OVERLAP` | 追加関係で係数を簡約 | 収集の取得費用・追加情報を認めつつ、G6局所bit countとの分離は未確定 | [MT] §4.2; BIT §2 |
| P12 | `DIFFERENT_TARGET_OR_ACCESS` | LCU routineで実装 | G6のconditional CQ/回転IRと同じquery/phase価格を要確認 | [MT] (29)後のLCU実装説明 |

## PTSC：Paired Taylor-series compensation

| 段階 | 対応status | 一次資料の実際の処理 | 同じ条件への対応／残る不足 | 一次locator |
|---|---|---|---|---|
| P1 | `ROUTINE_REPARAMETERIZATION` | zeroth-order S0=IのTaylor入力 | 一般PF remainderと分離し、同じ係数・microstep・cutoffへ固定する | [PTSC] IV.A (42)–(44) |
| P2 | `NOT_ESTABLISHED` | Taylor補償係数 | 自由積Green return再帰は当該箇所で未確認 | [PTSC] IV.A (42)–(50) |
| P3 | `PARTIAL_OVERLAP` | 次数とlabelの積表現 | 生成時の非列挙機会を認める。return局所queryそのものではない | [PTSC] (48)–(49), IV.C |
| P4 | `PARTIAL_OVERLAP` | paired先頭項と残余次数のlaw | 全even pairのordinary envelopeとは異なる。scalar law一致を仮定しない | [PTSC] (47)–(49) |
| P5 | `NOT_ESTABLISHED` | 補償項をsampling | 非reduced rawの量子前zero規則は未確認 | [PTSC] (48)–(49), IV.C |
| P6 | `PARTIAL_OVERLAP` | Iと先頭項のEuler pairing | 全return後のparent/child配分は別。Euler identity単体のみ直接対応 | [PTSC] (45)–(48) |
| P7 | `PARTIAL_OVERLAP` | 回転とphase付き積 | 角度・順序を同じcoherent targetで照合する必要。suffix law一致は未確認 | [PTSC] (45)–(49) |
| P8 | `NOT_ESTABLISHED` | LCUの次数/label確率 | G6のfull-support dyadic/root bias/range接続は未確認 | [PTSC] IV.A (48)–(50) |
| P9 | `ROUTINE_REPARAMETERIZATION` | finite cutoffのLCU式 | S0=I、m≥3でcutoff=m、係数・signを同じ有限和へ合わせる範囲。m=1は先頭Euler式で別扱い、digital lawも別 | [PTSC] (42), (45), (48)–(50) |
| P10 | `PARTIAL_OVERLAP` | normalizerとsampling資源 | G6のU・prequantum zero・全trialの同じ保証は未確認 | [PTSC] Prop.3, IV.C |
| P11 | `PARTIAL_OVERLAP` | 次数/label生成を利用可能 | 全辞書型の対照へ弱めない。Green局所queryと同bitモデルでの分離は未確定 | [PTSC] (49), IV.C |
| P12 | `DIFFERENT_TARGET_OR_ACCESS` | Pauli/LCU access | generic CQと同じnative/phase条件の対応が必要 | [PTSC] IV.A (45)–(49), IV.C |

## SCU：SCU / CTS including layered sampling

| 段階 | 対応status | 一次資料の実際の処理 | 同じ条件への対応／残る不足 | 一次locator |
|---|---|---|---|---|
| P1 | `DIFFERENT_TARGET_OR_ACCESS` | Pauli展開と左右propagator | 同じfinite operator側を切り出し、channel左右drawとは分離。共通A1も許す | [SCU] II.3 Thm1 (5)–(7) |
| P2 | `NOT_ESTABLISHED` | Pauli係数の収集 | Greenを合法に与えてよいが、G6のF/G接続が本文にあるとはしない | [SCU] Methods IV.2; Suppl.Note5 |
| P3 | `PARTIAL_OVERLAP` | layer別独立sampling | 非列挙・layeringの既知性を認める。全cross-layer returnを保持する同lawは未確認 | [SCU] Suppl.Note3 |
| P4 | `PARTIAL_OVERLAP` | Pauli/rotation群の確率 | 収集後normalizer/層のlawでありordinary-envelope raw proposalと同じではない | [SCU] (6), Methods IV.2 |
| P5 | `NOT_ESTABLISHED` | layerでsampleした積を用いる | G6のraw非reduced判定→量子前zeroという条件は未確認 | [SCU] Suppl.Note3 |
| P6 | `PARTIAL_OVERLAP` | 収集した虚係数をEuler回転化 | 一般odd formal wordは単一involutionと限らない。G6のsingle child＋suffix配分が不足 | [SCU] Thm1; Suppl.Note5 (S14)–(S15) |
| P7 | `PARTIAL_OVERLAP` | signed Pauli回転とunitary積 | Pauli固有収集・phaseを保つ必要。全event順序/角度の写像は未確定 | [SCU] (5)–(6), Suppl.Note5 |
| P8 | `NOT_ESTABLISHED` | 収集係数による理想確率 | 同じ相対丸め・実π補正機会を認めるが、既存のG6型bit保証とは認定しない | [SCU] Methods IV.2; Suppl.Note3/5 |
| P9 | `PARTIAL_OVERLAP` | truncated operatorと左右平均 | 同じfinite cutoffでoperator比較可能。channel一致やG6 M̃/biasまで自動移送しない | [SCU] (4)–(7); Suppl.Note5 |
| P10 | `PARTIAL_OVERLAP` | LCU/L1と層sampling overhead | 層normalizer等を課金する。同じU/Bernstein/量子前reject予算は未確認 | [SCU] (6)–(7), Suppl.Note3 |
| P11 | `PARTIAL_OVERLAP` | 全展開とlayeringの複数経路 | 既知法を全高次表に制限しない。同Green accessの追加費用込み比較は未確立 | [SCU] Methods IV.2; Suppl.Note3 |
| P12 | `DIFFERENT_TARGET_OR_ACCESS` | 左右unitaryのSCU実装 | 同じcontrolled coherent taskとworkspace/phaseを揃える必要。対照から除外しない | [SCU] (4), Fig.1, (6)–(7) |

## SPS：Sparse Probabilistic Synthesis

| 段階 | 対応status | 一次資料の実際の処理 | 同じ条件への対応／残る不足 | 一次locator |
|---|---|---|---|---|
| P1 | `DIFFERENT_TARGET_OR_ACCESS` | 指定process libraryと分解係数 | operator Mではなくprocessが基本対象。control拡張dictionaryは検討可能 | [SPS] II.1 (1)–(2) |
| P2 | `NOT_ESTABLISHED` | library/係数solution | Green情報の取得・切断再帰は当該手順で未確認 | [SPS] II.1–II.2 library formulation |
| P3 | `NOT_ESTABLISHED` | library係数のsampling | 全formal wordの局所queryを同じ入力から実現する記載は未確認 | [SPS] Statement1; II.2 |
| P4 | `PARTIAL_OVERLAP` | abs係数/L1でlibrary選択 | ordinary envelopeに限らない。libraryとL1取得を無料にしない | [SPS] Statement1 (2) |
| P5 | `NOT_ESTABLISHED` | signed measurement後処理 | G6と同じraw事前zeroの実行経路は未確認 | [SPS] Statement1; App.B.1 |
| P6 | `NOT_ESTABLISHED` | 指定dictionaryを確率分解 | return後a/s/dやsuffix角libraryを既存SPSが生成する根拠は未確認 | [SPS] II.1–II.2 |
| P7 | `DIFFERENT_TARGET_OR_ACCESS` | process matrixは共通global phaseを消す | control拡張で保持すべきoff-diagonal作用の一致を別に要確認 | [SPS] II.1 process definition; III.3 Statement3 |
| P8 | `PARTIAL_OVERLAP` | signed L1 weightを付ける | 一般weight原理は対応。実dyadic π・root biasの同じ保証は未確認 | [SPS] Statements1–2; App.B.1–B.2 |
| P9 | `PARTIAL_OVERLAP` | process/complex signalの平均とvariance | 複素信号も扱うので単にchannelという理由で排除しない。M̃のoperator条件は別 | [SPS] Statements2–3; App.B.2 |
| P10 | `PARTIAL_OVERLAP` | L1によるmeasurement overhead | G6のUと全trial/reject回数の同じconfidence保証は未確認 | [SPS] Statement2; App.B.2 (17)–(19) |
| P11 | `DIFFERENT_TARGET_OR_ACCESS` | offline library/convex解を取得 | 同じp/Greenだけから局所event辞書を得る算法とは未確認。弱いlibraryへ固定しない | [SPS] II.2; library setup |
| P12 | `DIFFERENT_TARGET_OR_ACCESS` | 取得済みgate近似library | 一般Qのcontrolled/strict phase/native取得を同条件へ戻す必要 | [SPS] II.1; III.1; III.3 |

## IS：Resource-Optimal Importance Sampling

| 段階 | 対応status | 一次資料の実際の処理 | 同じ条件への対応／残る不足 | 一次locator |
|---|---|---|---|---|
| P1 | `DIFFERENT_TARGET_OR_ACCESS` | 既知ensemble・proposal・費用 | 拡張coherent taskでも検討可能。event係数/角度を最初から無料にしない | [IS] II (1)–(7) |
| P2 | `NOT_ESTABLISHED` | 既存ensembleへproposal変更 | return係数のGreen queryを生成する手順ではない。A1を双方へ追加可 | [IS] II Thm1 (9)–(13) |
| P3 | `NOT_ESTABLISHED` | ensemble重みと費用のquery | a_u/childを局所取得する既存関数は未確認 | [IS] II Thm1; III Thm2 |
| P4 | `PARTIAL_OVERLAP` | 任意qと費用付きq* | G6のordinary proposalも選択の一例として比較可能。q*の取得/正規化は有料 | [IS] (4), (9)–(13) |
| P5 | `DIFFERENT_TARGET_OR_ACCESS` | quantum detection後ZeroFill/Discard | G6の古典事前rejectへ物理費用をそのまま移さない | [IS] IV.1–IV.2 (28)–(34) |
| P6 | `NOT_ESTABLISHED` | 取得済みlawを再weight | a/s/dの構成とpositive parentの取得はIS式から自動では得られない | [IS] II (4); Thm1 |
| P7 | `PARTIAL_OVERLAP` | 既存回路自体を保つ再sample | 位相を含む元eventが与えられた後の話。G6角度/orderを作る手順ではない | [IS] III Thm2 (22)–(23) |
| P8 | `ROUTINE_REPARAMETERIZATION` | 実p/qのimportance補正 | 未正規化c̃/実πへの書換えは同じ既知恒等式。これだけで丸め/bias保証を追加しない | [IS] II (4); K rational_event_weight |
| P9 | `PARTIAL_OVERLAP` | 二次momentと既存bias保存 | 新しいc̃−cを消さない。有限bit range・supportはBITの条件を別に要する | [IS] II (6)–(7); III (23) |
| P10 | `PARTIAL_OVERLAP` | 費用×moment、ZeroFill会計 | §IIのCLT型net-costと有限Bernstein Nは別。未知normalizer/取得費用を落とさない | [IS] II (7)–(13); IV (28)–(34) |
| P11 | `PARTIAL_OVERLAP` | proposal最適性の式 | q*を書けることは同accessで効率生成できる証拠でない。classical取得を課金 | [IS] II Thm1 (9)–(13) |
| P12 | `DIFFERENT_TARGET_OR_ACCESS` | 既存quantum circuitの費用c | G6 provider/phaseを同条件で与えた後に比較。quantum後rejectを古典前拒否に読み替えない | [IS] II (5); IV.1 execution loop |

## 3. 直接対応がある部品と全還元を区別する

以下の部品は`DIRECTLY_DOCUMENTED`として確認した：AKのGreen式、WAN/PTSC/CTSのEuler式、WANのordinary次数/IID生成、CTSの独立layer生成、ISの重み補正、SPSのlibrary選択、固定Kの局所packet、固定Gのsample/zero/U/budget接続。これらの部品の明示性と、表の各P段階全体への対応statusは別である。機械索引のDC1〜DC8が追跡する。

[AK]からのGreen特殊化、[WAN]/[PTSC]/[SCU]のEuler式、[IS]の実proposal補正は直接に追える。たとえば`πW̃=c̃`は既知の重み恒等式の対応であり、それ自体を独立新規性としない。G6内の既存SOURCEに当該処理が明示されることもDIRECTLY_DOCUMENTEDである。ただし、その存在を先行アルゴリズムの全lawとして認定しない。

既知Green＋Euler＋rejection＋ISをすべて共通に認めても、比較を閉じるには、有限Taylorへのsigned収集/正値性、suffixごとのpositive配分、zero結果と実π/c̃、U/bias/range/費用を同じscopeで接続する所在又は直接写像が必要である。これらはG6側の既存M1〜M12に証拠があるが、先行手順への全還元は未確認。新しいjoint-law同値定理や一般的非同値定理が必要なら、`REQUIRES_NEW_SCIENTIFIC_PROOF`として停止して返す。[組合せ監査](known_method_composition_audit.md)参照。

同じfinite meanへの対応、同じevent lawへの対応、同じ保証への対応の三つを混同しない。部分対応のセル数から、全同値・独立性・性能のscoreや順位を作らない。

## 固定資料・一次文献

[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[CP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
[C02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/docs/research/track_b_g6_construction_paper_evidence_20261018/mathematical_claim_evidence_matrix.md
[C03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/docs/research/track_b_g6_construction_paper_evidence_20261018/prior_art_and_method_delta.md
[C04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/docs/research/track_b_g6_construction_paper_evidence_20261018/implementation_and_validation_evidence.md
[N02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/128a05765985c1c6aec3a1d0ce35e19ffbb3b3f6/docs/research/track_b_g6_nonpauli_application_survey_20261018/application_feasibility_matrix.md
[N03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/128a05765985c1c6aec3a1d0ce35e19ffbb3b3f6/docs/research/track_b_g6_nonpauli_application_survey_20261018/pr_rte_connection_and_access.md
[N04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/128a05765985c1c6aec3a1d0ce35e19ffbb3b3f6/docs/research/track_b_g6_nonpauli_application_survey_20261018/controlled_native_feasibility.md
[IS]: https://arxiv.org/html/2603.13495v1
[SCU]: https://arxiv.org/html/2407.21095v2
[SPS]: https://arxiv.org/html/2402.15550v2
[PTSC]: https://arxiv.org/pdf/2212.04566v2
[WAN]: https://arxiv.org/pdf/2110.12071v2
[MT]: https://arxiv.org/pdf/2103.07988v2
[AK]: https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf

SOURCE入口：[K] / [G] / [CP] / [MATH] / [BIT] / [U]。既存Matrix [C02]、方法差 [C03]、実装scope [C04]。非Pauli適用表 [N02]、接続 [N03]、controlled条件 [N04]。

SOURCEの参照ID・hash・symbol位置は[source_inventory.json](source_inventory.json)。文献の固定版、今回実際に読んだ範囲と既存Packet本文hash一致は[literature_scope.json](literature_scope.json)。私的原本は使用・公開していない。
