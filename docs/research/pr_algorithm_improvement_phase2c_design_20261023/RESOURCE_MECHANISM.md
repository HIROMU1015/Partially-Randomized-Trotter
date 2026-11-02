# 資源削減の機構、成立条件、相殺

本書に新しい測定値・改善倍率はない。設計上の可能性であり、basis／childのactual native費用が支配的と認定したものではない。

## 1. 実際に変える操作

S側のatomはGaussian frameを出入りしてZ/ZZの制御product／rotationを行う。Pへ替えたfragmentでは、同じF_lを別の係数辞書で表し、各sampled atomは物理register上のPauli wordとなる。そのeventではGaussian frameの出入りを局所Clifford／parity回路へ置き換える。Pauli wordが長ければparity費用が増え、係数のcollect／分布も変わる。単一の元atomを同じ確率で安いgateへ置換したという意味ではない。

child rotationは各finite-RTE eventに一つあり、order0でも存在する。従って辞書変更は長いparentが出る場合だけに限定されない。ただしorder0の確率、S/Pの出現率、basis／central占有率、compiled利得は今後の対象であり、旧H4等から割合を移していない。

## 2. 全費用層への影響

| 層 | 固定／変化 | 削減の可能性と相殺 |
|---|---|---|
| model・係数 | H_ref固定、理想の新ΔH=0、既存ΔH_input維持。辞書数・係数・λは変化 | 同一Pauliのcollectでλが下がる場合も、Pauli展開で上がる場合もある。norm単独で選ばない |
| PF | H_D、R、split、順序、δ、q固定 | 理想では同じPF target。新しいPF係数改善を主張しない。有限bit差と粗いcertificate差は別 |
| finite RTE | r/K固定、λにより次数law・角度・Bは変化 | 理想のcorrected polynomialは同じRから作る条件。B増で統計負担が増え得る |
| variance／shots | 同じ複素signal allowanceとconfidence | canonical samplerの補正rangeはB_tot依存。state依存varianceを取得済みとしない。Nのceilも数える |
| 1-shot logical | frame費をPauli local/parity費へ交換 | 既存full/support共有後の差を対象にする。S runが長いとframe共有を失う可能性がある |
| actual control／RZ／CX／T／depth | 全armで同じ合法制御とcompiler | 設計価格はliteral RZ等。CX増、phase、layout、角度簡約でRZ選択とactual順位が逆転し得る。Tへの換算は未実施 |
| classical preprocess | S/P係数map、全体collect、候補評価DP、cache | coefficient algebra・memory・精度が重ければ量子節約を消す。既存mapを持つ対照へもcacheを与える |
| sampling／保存 | component数、離散law構築・lookup、metadata量が変化 | alias等の既存手順を使っても構築費は無料ではない。新ISや全path再重み付けは導入しない |
| sector／実行空間 | target sectorは同じ。P atom単独の粒子数保存は要求できない | full registerでは操作可能でも、sector内に縮約したsimulation／encodingは使えない場合がある。中間postselectionで補わない |
| prep/readout | 同じ物理状態・wrapperとその単価 | Nが変われば総準備・読出し費も変わる。1-shot共通だから総費用で相殺するとは限らない |

## 3. 改善を期待できる必要条件

Pへ替えるfragmentに、shared Gaussian経路より安くなるPauli実装構造が必要である。同時にλ/B/Nの悪化、他fragmentのbasis run破壊、parityと中央制御の追加費、古典係数／価格取得費が利得を食い潰さない必要がある。端点のどちらかが常に最良なら混在選択の実用範囲は狭い。これは失敗ではなく最小検証で切り分ける条件である。

score上の十分な改善候補でも、実compiler後に効果が消える可能性がある。Φには異basis相対Gaussian合成、可換Pauli cluster、phase全体の相殺、D/R境界の追加簡約を完全には含めていない。その限界は両armに共通で、実測時の合法最適化を禁止する理由にならない。価格を補修するために新たなnative価格oracleや大量pilotを無承認で取得しない。

## 4. 根拠として使った既存結果の強さ

`EV-0179`はH2 1.00 Å/STO-3G・保存DF rank3・T=.8/q4（δ=.2固定、delta scanなし）・r2/K4・ordinary-control full Hadamard・prep除外・topology-free Qiskit1.3/opt1/seed17、PR各64 paired trajectoriesの比較で、小さいBのDF側が回路単価で不利になった固定例。Pauli `L_D=7/14` とDF `L_D=1/3` は同じ物理splitではないので、本案の因果的な純表現効果ではない。保存記録の引用であり、immutable CIや外部再現の認定ではない。

`EV-0197`のsynthetic 3-mode core/pairは、N節約がbasis／central費用に相殺された限定証拠。`P2-ATTR`の固定H4 post-M2はshots増と1-shot減の交換を示すが、det/random/basis/centralのcompiled占有率は復元できない。本書へ既存倍率を転記し、新候補の改善予測にしていない。

`EV-0027/0028`のrun共有と、`P2-BUILD/REPEAT`の実SOURCEは省略できない強い既存部品。P-A追加intervalのnegativeやGaussian closureを、全辞書変更のno-goへ拡張していない。[出典の固定identity](SOURCE_EVIDENCE_INDEX.json)。

## 5. 古典計算の実行可能性

係数mapは低次数fermion構造を利用する。N軌道なら一般の四次係数・Pauli mapは高次の多項式規模になり得て、全Fock行列を避けるだけで安価とは言えない。各候補のcollect・確率再計算、finite-state moment伝搬のatom×state遷移、精度、q/r/K、候補数を別に記録する。計算量を「最大2+2M回」だけで隠さない。

fragment別map作成や同じ価格入力は独立にbatch・cacheできる。選択passの更新は逐次依存を持つ。将来の実行ではavailable CPU/GPU/memoryを確認し、dominant workloadを適切に高速化・並列化する。恣意的なCPU・GPU・memory・wall制限は設定しない。科学call／候補／出力の範囲は別契約で固定し、共有環境と他jobには変更しない。今回これらの科学実行は行っていない。
