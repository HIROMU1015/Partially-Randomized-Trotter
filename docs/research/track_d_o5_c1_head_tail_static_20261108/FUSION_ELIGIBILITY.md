# 局所書換えの適格性

これは実装前の技術条件の整理。数値correctnessや新しい一般保証を取得したものではない。SOURCEは[SOURCE_INDEX](SOURCE_INDEX.md)。

## 優先候補：同じsystem上の隣接Gaussian橋

head出口のbasis回路をG_H、次のtail入口のbasis回路をG_Tとする。時間順の

`… head-center → G_H → G_T.inverse() → tail-center …`

に必要な橋の行列は **G_T†G_H**。両basisがancillaに作用せず、同じwire順を持ち、その間に非可換演算がなければ、橋だけを同じ行列のrelative Gaussianへ置換する局所恒等式が候補になる。head/tail中心のmodeは変えないため、uncontrolled/ordinary/directionalの違いだけを理由にこの**basisだけ**の書換えを拒否する必要はない。

既存 `StrongBuilder.transition(model, previous, following)` はV_following†V_previousを合成し、実basis emitterの真空位相差を

φ=φ(G_H)−φ(G_T)−φ(G_relative)

でdeclared global phaseへ補正する。この既知実装を、H4 Providerと同じfull-frame/gaugeから得たhead/tail端部に接続する案である。新しいGaussian合成器や新干渉gadgetは不要。コードに有限精度行列積・合成があるので、数式上の恒等性だけで生成回路の許容誤差内一致を認定せず、既存full-input checkerを次回使う。

| 条件 | 適格 / 不適格の扱い |
|---|---|
| 同じ**実回路**Gとactual G.inverseが隣接 | 数学的にはI。raw専用削除は未実装だがP0との重複が強く疑われ、全接口での増分はUNKNOWN |
| 異なるexact rotation、共通emitter/gauge、同system、隣接 | relative化は合法候補。head内では既存、head→tailでは未接続。POTENTIAL_GAIN、compiled追加効果UNKNOWN |
| basis IDだけ同じ、rotation/gauge/wires不一致 | 同basis相殺としてはINELIGIBLE。ID一致で数値同値を代用しない |
| rotationが近いだけ、縮退部分空間が似るだけ | exact同一扱いはINELIGIBLE。新しい近似規則や閾値を追加しない |
| generic singleton support completion / 違うprovider | full-frame transitionへ無条件接続不可。実際の回路rotationとgaugeが取得・適合するまでUNKNOWN |
| Gaussian間にsystem回転・entangler等がある | そのまま跨いで融合する案はINELIGIBLE。可換性の個別根拠が必要 |
| 間にancillaのみのP(event phase) | system Gaussianとは可換だが、phaseを一回保持する必要がある。削除やglobal化はbranchを変える |
| global cutの両側・別register・CSWAP越し | 現gadgetを維持した局所橋としてはINELIGIBLE |

## Control modeを合算する場合の別条件

固定した**同じgenerator**のU(t)=exp(−itD)について、modeの枝は次のとおり。Dのidentity成分も含む。

| mode | ancilla=0 | ancilla=1 |
|---|---|---|
| uncontrolled | U(t) | U(t) |
| ordinary | I | U(t) |
| directional | U(−t) | U(t) |

同frameであっても、generator/supportが違えば単一timeの表をそのまま使わない。basis橋の除去と中心角度の合算は別の操作である。

| 隣接mode（time a,b） | 枝0 / 枝1 | 既存mode一つへの合算条件 |
|---|---|---|
| U,U | U(a+b) / U(a+b) | uncontrolled(a+b) |
| O,O | I / U(a+b) | ordinary(a+b) |
| D,D | U(−a−b) / U(a+b) | directional(a+b) |
| D,U またはU,D（同generator） | U(b−a) / U(a+b)（Dのtime=a） | a=bならordinary(2a)。既存step半sweep規則 |
| O(a),U(b) | U(b) / U(a+b) | 一般には単一ordinary/directionalへ無断変換不可 |
| O(a),D(b) | U(−b) / U(a+b) | 一般には単一modeへ無断変換不可。branch別の会計が必要 |

既存 `mixed_head` はsame modeかつexact same rotationだけをgroup化する。mode間の一般合算は新しく提案しない。とくにdirectional half＋uncontrolled halfをdirectional fullへ置換すると枝0が変わるので棄却する。head-tailに同basisがあってもtail ordinaryをuncontrolledやdirectionalへ置換しない。

## 保持する対象とchecker

StrongはW0†W1=Vを保持し、現whole-pathがdirect-controlled Vに一致する条件では両branchも照合する。basis橋の局所全unitary一致ならこの関係を保存できる。単一入力の期待値やglobal phaseをfitした一致で代用しない。

ReferenceはV=AB、A_actual_adjoint=A†、各halfのvacuum ray、既知χ、全入力complex coherence/X/Y observable、最終reference leakageを維持する。B6bのwhole-V-ray信号条件だけを使って既存half-ray checkerを緩めない。既存gadgetの二CSWAP、同一cut、readoutのX:H / Y:Sdg→H、ancilla最後・bit0:+1/bit1:−1を保持する。

event/applicationの順序・signed angle・support・normalization・attenuation、scalar/extracted identityを一回計上する条件も不変。normalization/attenuationをunitaryへ乗算しない。RZ由来のvacuum gaugeとancilla Pのrelative phaseは別物であり、Gaussianのsingle-particle行列だけの比較では不足する。

次回checkerはC7-2の独立exterior lift/occupation diagonal、C3/C5のAB/actual inverse/phase/sign、Track Dの全入力isometry/coherence/X/Yを再利用する。component abs1e−10、observable Frobenius/signal2e−10、rtol=0、phase fitなしという既存条件を候補にし、実行前に固定する。今回は呼んでいない。

イベント並替え、head schedule/PF次数、Hamiltonian分割、CDF/DF表現、RTE law/cutoff/normalizationの変更はC2。C1の局所同値変換へ混ぜない。
