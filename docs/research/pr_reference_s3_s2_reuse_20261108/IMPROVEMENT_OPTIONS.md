# S3構造保証と保存S2 recordの再利用条件

2026-11-08 JST。基準f0b4b143e9c01115cd010f4dd9f88354ba06c3b4。[REPORT](REPORT.md)。SOURCEで確認した仕様、保存metadata、条件付き数理式、未実行の提案を分ける。

## 1. 共役式と資格の段階

既存の[証明](../pr_reference_noncentral_reuse_20261108/IMPROVEMENT_OPTIONS.md#2-共役blockの条件付き保証とinverse-defect)を再利用する。Unitary B,D,A,D_tilde,Cに対して

`U=B D B†`, `V=A D_tilde C`, `eta=||C-A†||_2`,

`eps_f=||A-B||_2`, `eps_i=||C-B†||_2`, `eps_D=||D_tilde-D||_2`。

D=exp(-itH)に対しW=diam(spec H)を使えるなら、

`||V-U||_2 <= eps_D + min(eps_f+eps_i, min(2,|t|W)eps_f+eta)`。

これは条件付き数学的十分式で、今回新しくeta／W／epsを評価したものではない。SOURCE上のinverseで十分なのではなく、**実際に使うA/Cの関係**が必要である。

| 資格の段階 | 確認事項 | 今回の状態 |
|---|---|---|
| SOURCE inverse | 順序反転＋gate.inverse、同じbasis ID/hashとwire | 既存SOURCEで確認 |
| 各RZのnegative | 同じpositive sequence全体のactual adjoint、Wを含む | 現provider／lower_nativeの仕様を確認 |
| Logical instruction product | 保存native列を同じwordへ置換したA/Cが逆か | 88 pairで限定構文系では証明できず。一般の不一致とはしない |
| Full native basisのstrict phase | Native global phaseを含めC=A†か | 保存local loweringだけでは未資格。SOURCE phaseで代用しない |
| Emitted／consumer | 同じnamed列、word選択、inverse、wire／fence／join、terminal | 既存限定資格を引用。新しいpair／overlay経路は未検証 |
| Full-lawの共役誤差 | 全context、center、support、旧compiler diskとの接続 | 未資格 |

静的構文系は、RZ wordをWを含むhashの不透明記号として扱う。同一wordの逆だけを相殺し、一般の異word等価性を判断しない。残ったtokenを「eta>0」と解釈しない。現在はどのpairにもeta=0を付与していない。

## 2. 改善候補A：現wordのまま保証を鋭くする

### A-I：現在のinverse defectを別に囲う

Current A/Cを保持し、instruction-body defectを保存wordのexact matrixとinterval評価で囲う候補。新合成もword変更も不要である。小さいetaが資格できれば上式の第二項が通常加法より有利になる余地がある。大きい／広い上界なら、この証明経路は不十分なまま残る。演算子不一致やRPE失敗の判定ではない。

Native global phaseの対応が未取得なら、測れるのは**instruction bodyのeta**であってfull strict etaではない。欠けたprefactorを0にしない。既存N5-B1のcommon-joint-scalar分離を使う場合も、全control構成後・同じnamed gate列・uncontrolled basis embedding・同一terminal sinkという条件を満たす必要がある。未知scalarを理由なくphase-fitで除く方法ではない。

### A-II：同じ近似basisを使い、帰りを全体のactual adjointにする

新しい構成保持adapterでC=A†を**構成として**保証する候補。Positive wordとその全体inverse、W、wire順を再利用できる。Eta=0の前提を満たす経路だが、現consumerの個別RZ置換だけでは得られない。

実装する場合、SOURCE basisをapproximate bodyへ対応させ、return側をそのactual adjointとして生成する必要がある。現在の独立なinverse loweringを書き換えるので、旧d_old／compiler資格をそのまま移さない。既存C2の抽象原理とinverse実装は参考になるが、Full Y4の係数・価格・科学結果は再利用しない。今回はadapter実装・build/compileをしていない。

### A-III：Center／supportに依存する係数

Dのdeterministic diagonalと、RTE runのproducts＋rotationを区別する。RTE builderではproductsがrotationより前にあり、identityはactive basisをflushする。同basisの共有とD境界融合は既に実施されているため、その削減を二重計上しない。

Productsを含むcenterがnear-identityとは限らず、全runへ小角係数min(2,|t|W)を強制できない。SOURCE全component／orderを覆う有限support条件又は安全なsupersetが必要で、期待run数や有限例最大を全support保証に使わない。Controlled centerでは、basisが両branchに同じuncontrolled操作として作用する条件を保持し、SOURCE scalar／event／vacuumのrelative phaseを消さない。

## 3. 改善候補B：保存frontierのfinite overlay

43 literalは3つのSOURCE RTE basisのnative位置に対応する。候補は**新角度の近似ではなく、同じbinary64 target literalへの既存strict record**である。元spec／record ID／sequence hash／保存guard methodが適合する場合は再合成を避けられる。

| 再利用できる部分 | Overlayに必要な最小差分／資格 |
|---|---|
| C0 299、85 composite、43基本word | 保持。選択literal以外を勝手に再構成しない |
| 保存S2の非劣ID | そのまま固定候補として使う。Frontier再計算／新record rankingをしない |
| 元backend strict specとguard | Original exponentとhashを保持。Transport用carrier keyをbackend仕様へ偽装しない |
| Positive sequence／negative inverse | 同じpositiveをlower_nativeへ渡し、negative inversionは一度だけ。Wを含む全inverseを保持 |
| 現lower_native／consumer | Named gate列とwire／reset／measure／joinを再利用。新policy経路の局所fixtureが必要 |
| Stored T/T†/W | 原字段のまま。違うwordのemitter保証や総Tを未検証で移さない |

現DictionaryProviderのconstructorはC0 cache exponent=-11、epsilon=1e-11、b36／43 basicを固定する。例えば既存-13 recordを使う場合、cacheを無断改変するのでなく、literal→record tupleを持つ有限overlay adapterが必要。元のcache recordを-11 specとして保存してはならない。

同じliteralは両arm・全roleで同じselectorへ接続する。今回Dへの43 literalのhitはなかったが、whole consumerや未照合経路まで不存在と一般化しない。候補だけへ無料の真E0／gap／ground古典記述を与えず、同じrecord pool・取得機会を強B′にも与える。

## 4. Dの精度／T tradeoffを扱う条件

保存D frontierでは現wordは非劣である。よって同じ有限pool・同じguard/T意味なら、strictにguardを改善しTを増やさないpointを、既存結果に反して存在すると扱えない。一方、より高精度で高T、より低Tで緩いguardという選択肢は残る。どの点を使うかはjoint予算と全体費用に依存し、precision labelや局所Tだけで決めない。

一つの例としてD key `0x1.0009c3e615d69p-8`には、保存frontier上にT=120と100の異なるpointが残る。値は引用であり、今回新たなguard/T比較を行っていない。Recordごとのexact guard、TとprovenanceはSTATIC_FINDINGS内の`D_saved_frontiers_quoted`を参照する。同一pointの別archive記録も保持する。

## 5. 総RPE費用へ接続するために残ること

43候補がS1のworst-case frame maxを決めるbasisへ効くかは未評価である。新overlayを選ぶ前に、対象literal→template→SOURCE多重度／maxの対応を資格する必要がある。S3による保証改善とword選択は独立の要因として比較し、合成精度／近似basis構成を同時に勝手に変えない。

完成policyを得た後に、同じraw単位の旧diskとsynthesis diskを一度だけ加算し、X/Yの位相・半径・保存Nの十分条件へ接続する。N_B626／N_R630は異なるため、共通frameの1-shot価格を総費用差から消さない。全18roundでは各保存N・PF／r／bitsを使い、m17だけから外挿しない。LP下界や未実行frontierの点を実行可能な強B′価格にしない。

新規合成は、選んだ有限policy・保証・予算で既存record／構造保証が不足すると特定されてからの候補である。今回の構文不成立や43候補の非採択は、合成必須という証拠ではない。Domain、central、emitter、shell、terminalをunknown／非zeroで保ち、実際の誤差、保存上界、条件付き数理保証、有限数値結果を区別する。
