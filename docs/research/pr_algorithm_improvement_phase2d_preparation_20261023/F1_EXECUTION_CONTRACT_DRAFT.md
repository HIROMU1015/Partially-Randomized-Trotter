# F1実行契約案：正しさの限定検証

`DRAFT_NOT_AUTHORIZATION`。候補条件とcallを記載した文書であり、実装・科学実行を許可しない。現在 `PREPARATION_INCOMPLETE`。

## 1. 目的と対象

元のPhase 2Cのfragment別Pauli／frame辞書が、既存finite-RTE samplerへ正しく接続し、同じH／finite corrected targetを維持するかを検査する。SS/SP/PS/PPと既知四候補全列挙を[公平性契約](COMPARISON_AND_FAIRNESS.md)に従って比較する。F1は資源優位性・独立新規性の実証ではない。R1反可換pair案は対象外。

[入力候補](FIXTURE_AND_INPUT_IDENTITY_DRAFT.md)の4mode／2粒子／one-body block1／残差2fragment、`T=.2,q=2,delta=.1,r=2,h=.05,K=2`。q2はD macro境界の融合、r2は同じoccurrence内のbasis run継続、K2はn0とn2のproduct-child order／paired phaseの両方を含める診断設定。最適なr/Kの主張ではない。full16／controlled32次元でsector外Pauli作用を隠さず検査できる。

## 2. SOURCEと参照の分離

起点commitは `ef2eb7d55ea37bf93e0b85773df20b63811d1f9b`。既存RTEComponent/RTETailLike/law/eventを再利用し、任意Pauliのtyped router、共通sequence policy、wrapper adapterを追加する**予定**。回路recipe参照2978版／support policy f29版と起点の版差を[接続案](SOURCE_IMPLEMENTATION_PLAN.md)に明記した。新SOURCE commit／closure／envはまだ存在せずnull。

候補経路はexact coefficient map→canonical atom→既存event law→typed actual circuit。独立参照はCAR occupation行列とexterior-minor Gamma(V)、直接Pauli matrix、独立factorial polynomial／matrix expから作る。candidateのcanonical R、moment関数、router、価格transitionをそのままreferenceへ流用しない。

## 3. 別々に報告するgate

1. **H-ID**：exact係数入力、各S/P fragment、全armの `cI+D+R` を独立Hへ照合。I位置、係数collect、full register、2粒子sector restrictionsを別report。
2. **LAW**：各armのp、lambda、w_n、B、角度、event probability／coefficientを独立式と比較。全 `E_a=L_a+L_a^3` 一microstep eventのmetadataを列挙。抽選はしない。
3. **FINITE-MEAN**：actual component control1 blockからproduct／rotationの加重momentを再構成し、product→child順でone-microstep meanを作る。各nの `Vbar_n Pbar^n` にw_nを加え、四microstepとD/cを合成して独立T3 targetへ照合する。候補Rを同じ多項式関数へ入れただけの一致は不合格。factorization検査はcomponent branchの接続を検査するもので、full sequence fusionの一般証明ではない。
4. **CONTROLLED**：[8motif×4arm](MOTIF_CATALOG_DRAFT.md)でraw／optimized／actual adjointを検査。control0=I、off-diagonal blocks=0、control1=予定U。ancilla相対phase、負product／child、paired phase、identity/scalar、D境界、同／異basis、support/fullを独立照合。global phase fitting禁止。
5. **SIGNAL**：32optimized trajectoryに初期state prepとX/Y wrapperを付け、未compileとcompile後の複素信号を独立参照と比較。bit0=+1、bit1=-1、Y=+Im。ordinary wrapperは[回路順序](SOURCE_IMPLEMENTATION_PLAN.md)による。
6. **CATALOG-MEAN / PRICE**：4label68eventのraw／optimized controlled operatorとB加重mean、literal price直接列挙とDPを別gateにする。price PASSをcompiled予測の認証へ昇格しない。

`H-ID_PASS`、`LAW_PASS`、`FINITE_MEAN_PASS`、`CONTROLLED_PASS`、`SIGNAL_PASS`、`PRICE_PASS` は別々に保存する。PO-01–09の一般的な数学証明が済んだという表示をしない。

## 4. 科学call範囲

規範的な機械台帳は [CALL_INVENTORY_DRAFT.json](CALL_INVENTORY_DRAFT.json)。入力binding前のLを推測しない。

| 作業 | 登録件数候補 |
|---|---|
| physical fixture／catalogの入力生成 | 別々の2case、binding承認が必要 |
| 4arm coefficient map／H-ID | 各4case |
| 一microstep event metadata | `sum_a (L_a+L_a^3)` records |
| component controlled probes | `3 sum_a L_a` builds＋operator検査 |
| physical motifs raw／optimized／adjoint | 96 builds＋operator検査 |
| motif full X/Y wrappers | 64 builds、64 compile、未compile／compile後各64 signal検査 |
| catalog raw／optimized | 136 controlled builds＋operator検査、actual corrected means2件 |
| physical actual corrected mean | 4arm、各one-stepと全体を1report |
| catalog literal価格 | 68直接count、DP1、aggregate比較1 |
| physical literal候補価格 | 4unique、selector1、既知四候補探索1 |

各独立参照caseは90digits／128digitsの2評価をあらかじめ登録し、適応的precision増加や失敗retryにしない。有限mean reference4、motif reference32、catalog reference68、component unsigned reference `sum L_a` が独立のlogical reference cases。1case内の両precisionを台帳で別evaluationとして数える。scientific callの追加やthreshold変更には改訂と承認が必要。

## 5. Compiler・閾値・合否

限定compileは64full wrappersのみ。Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／topology-free。F1は限られたmotifの正しさ確認であり、全trajectoryのactual平均費用取得ではない。

[数値gate](SEMANTICS_AND_NUMERICAL_GATES.md)のdimension/precisionを考慮した候補値を承認前に固定する。H1e-11、controlled1e-10、mean/signal2e-10、compiled5e-10を主に用いる。rtol0、phase合わせなし。検査漏れ、未binding hash、未登録call、変換lossをPASSとしない。

全別gateの合格、原本保全、call逸脱なし、4armの適格性、意味のある構成差の記録を満たせば `F1_VALIDATED_WITHIN_REGISTERED_SCOPE` を提案できる。mixed最良は要求しない。差が残らない場合も有効な結果だがF2適格を自動採択しない。

## 6. 出力とSTOP

将来の新規science output namespaceは承認時に一意に固定する。入力、SOURCE/env closure、case ID、hash、数値residual、gate別結果、失敗case、timing/memory、count実績、STOP reportを保存する。全巨大event matrixは保持せず、metadataをstreamし、失敗caseの必要範囲と集約値を保存する。保存容量はLとliteral gate数をbinding後に見積もる。

H/mean/order/sign/adjoint/phase/sector/scalar不一致、hash不一致、原本保全違反、未承認条件変更、call逸脱は即STOP。同じ科学runを自動retryしない。失敗したbuild／compileもattemptとして消費し、再実行で置換しない。終了は成否にかかわらずmandatory STOP。F2、旧C1、RQ-04、レビュー自動開始の権限は一切生じない。
