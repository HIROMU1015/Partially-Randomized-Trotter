# F1：一般有理入力の事前有限bit精度・入口契約

2026-10-18 JST。保証付きの**十分条件**であり、必要条件・最小精度・H/K選択手順ではない。与えられたK_root/Hについて以下を満たせば、既存数学とSOURCE guardを接続できる。数値代入・selector・checker・testは作成していない。根拠は[MATH]§§1,4–8、[BIT]§§3–6、[G7C]、[K] / [G]の静的読解。

## 1. 入口仕様と保証の層

| 入力/分岐 | 必要条件・型 | 固定SOURCEとの差 |
|---|---|---|
| p table/labels | 有限L≥1、明示Rat、分子/分母は整数、分母>0、正確なp_i>0とΣp_i=1。零係数labelは呼出し前に除外し、対応表を保持。bit長H_pも記録 | K138–145/G107–110はFraction化後に正値/和検査。floatからのcoercionも受け付け得るため、明示Ratという入口仕様の実装には追加validatorが必要。元物理係数の保証とは無関係 |
| x/m/arm | 非identityでは有理0<x≤1、整数odd m≥1。bool等を整数入力の代用にしない。mの数値とbinary符号長を区別 | K144/fullはm=1可。G105 canonicalのordinary/partial_return_tail/closed_P3_tailはm≥3で、型検査もKと完全には同一でない。入口で域外を拒否 |
| x=0 | 有理zeroを認識しM=Iとして別経路。乱数・parent query・weight除算・angle取得を行わない。identity結果と非identityのrandom zeroを区別 | K constructorはx=0可、digital_parent221–222は拒否。fullにidentity dispatcherなし、canonicalもx=0拒否。数学的identityは既存、接続は未実装 |
| η/ρ/H/K_root | η,ρは明示有理で0<η<1、ρ>0。H,K_rootは正整数。K_rootはroot区間bit数でありTaylor次数/cutoffのKとは別 | Kの低水準guardは包括validatorではない。G fullのconstructorにρの一括範囲検査はない。固定defaultsの使用だけでは本preflightの証拠にならない |
| τ/σ/Q | 要求angle誤差τ>0、σは±1。Q_i†=Q_i、Q_i²=I、controlled-Qとactual adjointはexact queryモデルの前提 | generatorはσをevent fieldに保持せず、CPの引数として渡す。symbolic ratioと物理angle証拠は別。Qの性質は整数labelから検証できない |
| 直接helper呼出し | reducedな合法label word、even parent長≤m−1、childは空parentなら全i/非空ならi≠first(u)、group/index/packet/law/source/inputの一致 | K `_word`162–168/parent194–202に検査。G canonical event162–191のindexing等は包括的な入口ではない。負index・外来packetを含む任意の呼出しを既存APIが拒否すると認証しない |
| RNG/誤差/confidence | 各H-bit整数はfresh independent uniform。既存budgetを使うならその精度・bias・測定モデルと正の残余幅を満たす | G59–72のSHA-counterは診断用。G266–284は固定α規約であり一般α APIではない。型・memory・bit供給・serialization等の実行failureは数学的precision保証と別 |

L=1/m=1/x=1は既存数学域であり新しい境界定理は不要。以下はFullのvalid parent全体を対象とし、canonicalの違いは§5で扱う。Kのpure-parent規約は維持するが、外来s=0 packetをfullのchild前提経路へ自由に渡せるとはしない。域内ではL≥2なら各valid parentにchildがあり、L=1なら到達するeven reduced parentは空語だけである。

## 2. 使う既存の一様下界と記号

`n_o=(m+1)/2`、`t_min=x^(m−1)/(m−1)!`、`p_min=min_i p_i` とする。[BIT]§5の引用は次のとおり。

- order q_l≥q_min:=t_min/3。
- label≥p_min、有るchild≥p_min/2。
- valid-parentの条件付き受理A_u≥1/(2√2)>1/4。
- d_u≥d_min:=t_min p_min^(m−1)/2。

全trialのZはA_uと別であり、digital受理率をideal Zへ置き換えない。B_o≥1はb_0≥1からの直接帰結、b_l≥t_minは0<x≤1のt_lの単調性からの直接帰結である。

提供された固定bit数に対し、`Δ=2^(-K_root)`、`h=2^(-H)` と書く。root_interval45–52は外向き幅≤Δである。以下のP0〜P5はこれらの値に対する**受入述語**であり、K/Hを返す式ではない。全parent集合を入力・列挙する必要はない。

## 3. normalized intervalと丸めをつなぐ共通の受入述語

normalizedな理想law q_i≥q_min>0を、同じlawの区間[z_i^lo,z_i^hi]が包み、ある有理ζについて

`0 <= ζ < 1`, `(1−ζ)q_i <= z_i^lo <= q_i <= z_i^hi <= (1+ζ)q_i`

を満たす場合を考える。これはSOURCEが扱う区間の直接的な誤差会計であり、新しい確率法則ではない。midpoint正規化後のv_i/q_iは `(1−ζ)/(1+ζ)` と `(1+ζ)/(1−ζ)` の間にある。既存largest-remainders処理のfloor+1調整で各要素の絶対丸め差は≤hである。従って以下を **R(ζ,q_min)** と定義する。

```
0 <= ζ < 1
h/q_min <= (1−ζ)/(1+ζ) − (1−η)(1+ζ)
h/q_min <= (1+η)(1−ζ) − (1+ζ)/(1−ζ)
```

この直接代入により、SOURCE K79–81の `q̃_i≥(1−η)z_i^hi` と `q̃_i≤(1+η)z_i^lo` が全iで満たされ、q̃_i>0も従う。rhsが非正なら証拠不成立としてSTOPであり、閾値を変えない。ζ=0の正確な一点区間なら条件は `h<=η q_min` になる。

ここで行ったのは、既存midpoint正規化とfloor/余り調整の誤差を既存guardに代入する通常導出である。新しい丸め法・H/K十分定数・選択アルゴリズムではない。真のlawを囲む根拠も必要で、Σlo≤1≤ΣhiというSOURCE検査だけを任意外来区間の正当性証明としない。

## 4. Fullの全到達parentに対するP0〜P5

| ID | sampling前に確認する十分条件 | SOURCE guard/閉じる保証 | 入力と導出の型 |
|---|---|---|---|
| **P0 domain** | §1の非identity条件、正確なp/x、η/ρ、正整数K_root/H、fresh-bitモデル、全helperの合法呼出し | K138–145/162–168/194–202、G216–251。分母0や域外を数学的zeroにしない | 入口条件は既存保証とSOURCE直接照合 |
| **P1 order** | `e=n_o Δ<1`, `a=Δ/t_min`, `ζ_o=(a+e)/(1−e)` とし、R(ζ_o,q_min)が成立 | K208–212のroot正規化区間が正supportを持ち、K55–82の相対guardを通る | b_l≥t_min、B_o≥1、root誤差≤Δと総誤差≤n_oΔの直接代入。下記説明参照 |
| **P2 labels/children** | `h<=η p_min` かつ `h<=η p_min/2` | label G222、全valid parentのchild K230–231の一点区間dyadic guard | [BIT]§5の既存一様下界にR(0,·)を適用。childなしの別枝に架空のlawを要求しない |
| **P3 coefficient** | `Δ<d_min` かつ `Δ/2<=ρ(d_min−Δ)` | K224–227：root.lo>0、半幅≤ρ root.lo。全uでabs(d̃_u−d_u)≤ρd_u | root.lo≥d_u−Δ≥d_min−Δ、半幅≤Δ/2の直接代入 |
| **P4 acceptance** | `Δ<1/4` かつ `(2−η)Δ+h<=η/4` | K228–229/K85–94：positive lo、hi≤1、下向きthreshold≥(1−η)hi。全valid uで同じ条件 | A_u≥1/4の既存下界、root幅、downward floorの誤差だけを合成 |
| **P5 law/identity** | 実際のorder/label/child lawが同じp/x/m/K_root/H/η/ρとSOURCEに結び付き、独立H-bit integersをそのまま使う。x=0は§1で分岐 | K97–110のbit-map、G240–244の正当なzero、K239–254の実π補正を保持 | lawのすり替え・新normalization・retryを許さない呼出契約 |

P1の直接会計：各b_lのroot端点の誤差は≤Δ、総和端点の誤差は≤n_oΔ。従ってnormalized区間の相対端点は `(1−a)/(1+e)` 以上、`(1+a)/(1−e)` 以下であり、ζ_oが§3の対称的な保守的誤差を与える。P1が成立すれば全orderで正loがあり、total.loも正である。

P4の直接会計：真のAを囲む区間を[lo,min(1,hi)]とし、下向き丸め値をÃとすると、`Ã>=A−Δ−h`、`hi_clipped<=A+Δ`。従って `(2−η)Δ+h<=η A` は `Ã>=(1−η)hi_clipped` の十分条件であり、既存のA≥1/4を代入した。positive loもΔ<1/4で閉じる。clipを誤差の隠蔽に使っていない。

P0〜P5と既存正値/係数lemmaの下では、**未訪問を含む全到達valid parentで、列挙なしに既存の数学的precision guard条件が満たされる**。これは提供されたK/Hへの条件付き仕様完成であり、SOURCEがこのpreflight証拠を生成/検証済みという意味ではない。bit数を選ぶ手順や結果を見た増精度は作っていない。固定defaultsがP1〜P4を満たすと評価したこともない。

## 5. canonicalの域と全到達law

canonicalの3 armはG105–143のodd m≥3を保持する。Greenやfullのparent辞書を要求しない。

- group rootの正loと `hi−lo<=2ρlo`、normalized group lawの相対guardはconstructor137–143の有限group列で検査される。group数はSOURCE148–151/119–136に対応する有限の入力依存列であり、全parent宇宙ではない。
- IID labelはP2のlabel条件。`conditional(excluded)`153–156は1−p_excluded>0と合法なexcludedが前提で、使用する全iについてp_i/(1−p_excluded)≥p_min。従ってh≤ηp_minで全到達conditional lawの一点区間guardを閉じる。
- distinct groupがあるときのfirst_distribution158–160と、root_collectedがあるときのroot_child128–129は、SOURCEに明示された有限L個の正確な有理値である。それぞれ正値/和1を確認し、**その最小値q_min,first / q_min,root**に対するh≤ηq_minを追加する。不存在groupの分母1−χ等を評価しない。
- 各有限lawの根拠は[G7C]の既存canonical構成とSOURCE式であり、Full用のp_min/2下界を無断で全canonical分布へ転用しない。必要情報はp/x/m/armと有限group/root区間だけで、全語角辞書は不要。

これらはconstructor後のsample/eventで作り直すlawにも同じ入力identityの下で適用される。全到達lawの条件を一度のpreflight証拠へ結ぶ入口は未実装。Full/Partialのnormalizer順序やP_Uの真偽とは関係しない。

## 6. Angle・weight・confidenceの接続

K129–130/235/G251はexact rational tangent s/aを返す。既存[BIT]§6はs/a≤2x/(l+1)≤2、half-angle変換とatan交代級数によるenclosure可能性、angle誤差τからstrict involution回転誤差≤τを与える。未来のangle出力はφを囲むrad区間と、代表値の最大誤差≤τ、source/input/parent/σとprecisionのidentityを伴う。**このenclosureのSOURCE実装、native rotation取得、strict physical certificateは今回作らない。**

実dyadic π>0、c̃=d midpoint由来、W̃=c̃/πでπW̃=c̃。係数bias≤ρB_RA≤ρB_o、meanは一般にM̃。angleを近似した実装meanは更に別の誤差項を持つ。Mや元Hamiltonianへの完全一致とはしない。

既存budgetはU/B_oのhi、digital moment/range、固定α_axis=1/320・16 axes family≤1/20、共通factor2 bias、正remainingを使う。[G]258–284。0<η<1、ρ>0、primitive_error≥0等の前提を入口で確認し、精度failureが起こらない全law条件、fresh IID bits/測定モデル、全N完了が必要。任意αの新policy、最適N、failure budgetは作らない。

## 7. 三状態・failure証拠・未完了

| 状態 | 発生点 | 必要な扱い・保存 |
|---|---|---|
| A 正当なzero | G240–241 raw nonreduced、243–244 coin不受理 | 全NのX=0。rawをreduce先へ再配分しない。試行番号/経路/消費bits/古典費用prefixを保持 |
| B technical failure | K65–81のsupport/relative guard、K92–93のacceptance精度、K226–229のroot不足等。constructor時とtrial中を区別 | reason、関数/段階、source/input/policy hash、K/H/η/ρ、区間/不等式の証拠、trial/parent、成功prefixと消費資源を保全してSTOP。P条件が通ったのに数学的guard違反なら契約/実装不整合としてSTOP |
| C invalid input/契約違反 | p/domain/m/型/σ、word/index、bit範囲、law/packet identity、非正remaining | 不正項目と原因を記録しfail closed。既存ValueError型だけではB/Cを分類できないので入口/段階を保存 |

B/Cをzeroに変換しない。未完了prefixは完全な固定N confidence結果でなく、完走runだけへの事後conditioningの保証も主張しない。memory不足、供給器failure、serialization等は別のtechnical原因としてSTOPし、本数学precision条件で実行全体の無故障を保証したとはしない。再試行・refinement・新claimは採用しない。

**F1仕様は完成、入口/preflight/identity/angle/原因別保存の接続は未実装。** 将来の実装受入条件は[別表](implementation_acceptance_requirements.md)。本仕様は科学契約採択/実行許可ではない。**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G7C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
