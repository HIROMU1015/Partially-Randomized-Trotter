# F1のmotif・小catalog固定候補

静的な生成規則だけを定義する。event／trajectory／回路は今回生成していない。件数と選択規則をSOURCE/inputと一緒に採択してから実行する。coverage不足を別motifで救済しない。

## 1. 4armそれぞれ8件

stable atom IDはtyped kind、origin fragment、Pauli/support、basis identityのcanonical recordの辞書順とする。canonical化後の各armで、I=最初のidentity、u=最初のnonidentity、s=最初の負係数nonidentity、v=uと異なるsource basis keyを持つ最初のnonidentityを選ぶ。uとsは同じでもよい。I/s/vが存在することは今回確認していない。欠落は `INPUT_COVERAGE_FAILURE`。binding段階で不足が分かったら停止して契約改訂を求める。別fixture探索やmotifの事後追加はしない。

`E0(c)` はn=0のchild c。`E2(p1,p2;c)` は時間順にproduct p1、product p2、最後にchild cを適用し、paired phase -1を伴う。SOURCEのsampled index順へ渡す場合は `(c,p1,p2)`。各行の四eventを順にmicrosteps `(macro0,0),(macro0,1),(macro1,0),(macro1,1)` に配置する。各macrostep前後に固定D半step、外部scalarを含める。

| ID | 四event（左から時間順） | 主な識別対象 |
|---|---|---|
| M0 | E0(u), E0(v), E0(u), E0(v) | n0、異basis接続 |
| M1 | E0(s), E0(u), E0(s), E0(u) | 負child、sign一回適用 |
| M2 | E2(u,v;s), E0(u), E0(u), E0(u) | n2、非自明order、paired phase |
| M3 | E2(I,s;I), E0(v), E0(v), E0(v) | identity product/child、負product |
| M4 | E0(u), E0(u), E0(u), E0(u) | 同basis run、support/full、D境界 |
| M5 | E2(u,u;u), E0(u), E0(u), E0(u) | 三連続同basis、実際の相殺 |
| M6 | E0(u), E0(u), E0(v), E0(v) | macro境界half-step融合、basis run分割 |
| M7 | E0(I), E0(I), E0(I), E0(I) | residual I、外部scalar、ancilla phase |

計32 full controlled trajectories。各々でraw版・共通optimized版・optimized版のactual `.inverse()` adjointを一つずつ作り、**96 controlled circuit build**。adjointは順序を逆にして各gateをadjointにした実回路であり、角度だけ逆にしない。optimized正方向32件へX/Y prep/readout wrapper各一つ、**64 wrapper build・64 compile**。32件が違うtyped recipeを使うことと、実際のoptimized列に差が残ることを別欄で報告する。M0–M7は確率平均へ混入する標本ではない。

追加の各component product／child-θ0／child-θ2のcontrolled probeは `3 sum_a L_a` 件。[call台帳](CALL_INVENTORY_DRAFT.json)を参照。long Pauli branchが実fixtureで一つも存在しなくてもcatalogのCで任意長dispatcherを検査できる。ただしphysical motifの必須I/s/v欠落をcatalogで代替しない。

## 2. 別の4-label catalog

system/ancilla mapping、JW、B0は[fixture仕様](FIXTURE_AND_INPUT_IDENTITY_DRAFT.md)と共通。D=0、外部scalar `c_cat=-1/31`、`T_cat=h_cat=1/20, q_cat=r_cat=1, K_cat=2`。

| label | 実係数 | unsigned operator | recipe |
|---|---|---|---|
| A | +1/8 | B0 Z0 B0† | frame F、support {0} |
| B | -1/10 | B0 Z1 B0† | frame F、support {1} |
| C | +1/12 | X0 Y2 Z3 | physical Pauli、局所Clifford＋parity |
| I | +1/20 | I | ancilla相対phase |

これは通常の4involutionであり、反可換pair grouping／unitary partitioningは使わない。FとCのbasis keyは区別する。Iはliteral run境界。n0のchild4件、n2のordered `(child,product1,product2)` 64件で、**合計68event**。lexicographic A,B,C,I順、n0→n2順で全件を一度列挙する。

68件のraw／optimizedを各一回buildし、**136 controlled build・136 operator検査**。同じcatalogのB加重actual block meanをrawとoptimized各一回、独立CAR/Pauli参照のT3へ照合する。外部scalarはその参照と回路の両方に同じ規約で加える。catalog compileは行わない。

## 3. Literal priceの検査

価格単位は**未compileの固定literal recipeに含まれるRZ token数**。basis entry+exit、support singleton、central role/n、identity・負product・paired event・外部scalarの潜在phase reserveを別recordにする。全armとcatalogに同じproviderを使う。phase reserveは一寄与につき独立のancilla phaseをRZへ分解するliteral数で、actualのphase集約後countを予測しない。

各provider値は実行SOURCEのliteral recipe定義から一度bindingし、結果を見た手調整は禁止する。今回gate列を生成していないので、full/singleton等の数値価格とそのhashは未確定。これを0や仮の単価で埋めない。最終F1承認のblockerである。

有限状態はPhase 2C §7.2の `EMPTY / SINGLE(atom,role,n) / MULTI(basis)`。same-basis runのsingletonはsupport-restricted completion、長さ2以上はfull basisを一度entry/exitする。identityとD境界でflush、microstep境界だけではflushしない。catalogではr1なので68列の各終端でflush。別実装の列走査でliteral planを直接数え、確率付き有限列挙平均とDPを比較する。DPを自分自身のtransitionで列挙する循環検査にしない。

catalogは68件の直接price record、DP1回、集約比較1回。整数の各recipe token countは完全一致を要求する。期待値の丸めgateは[数値契約](SEMANTICS_AND_NUMERICAL_GATES.md)による。このPASSは**literal planと価格DPの整合性だけ**で、compiled費用の予測認証でも資源優位性でもない。
