# 一次文献・方法差：包含と同費用の手順同値を分ける

〔一次〕固定9本文の該当節を今回再取得した。version、bytes、SHA256、実際の読解範囲は[literature_scope.json](literature_scope.json)。全証明の独立検証、網羅的priority調査、著者実装の再実行ではない。〔既存〕G6/G10、Parent prior-art、D1–D3対応監査を[証拠索引](evidence_index.json)から引用する。以下は新保証・新生成規則の提案ではなく、既存構成の対応整理である。

## 1. タスクと強い対照の共通条件

対象は M=Pₘ(−iσxΣpᵢQᵢ) の**first operator moment**。ordered product・ancilla relative phaseを保持する。ensembleが同じchannelを作ること、同じobservableだけが一致すること、同じcoherent operator meanを作ることは別である。全exponentialに対する近似として比較するときは、有限cutoffとnative biasを共通taskに戻す必要がある。

対照にも同じp/Q、involution関係、Pauli情報、Green query、DF情報があればそれを与える。layer sampling、通常のcollection、同じ回路合成器・精度・seed・cache、価格取得とIS、partial/P3の配分機会も等しく許す。ordinaryに全語列挙を強制せず、CTSを常に全Pauli展開する方式として扱わず、SPSに不利な固定libraryだけを与えない。一方、全語価格表・controlled-Q・可変角合成・state accessを無料oracleとして与えた比較から実装上の利益を主張しない。

| 一次資料・確認箇所 | 既知の内容／数学的包含 | Track Bの具体的構成との対応 | 同情報・同費用で残る問い |
|---|---|---|---|
| [Wan–Berta–Campbell v2](https://arxiv.org/pdf/2110.12071v2)、App.C(C4)–(C5), Algorithm2、有限cutoff、Hamming-weight phasing | 隣接Taylor次数のEuler pairing、order＋IID labelの非列挙生成、normalizationとdepthの交換 | ordinaryは直接の近接構成。RAはreturn収集後のparentごとに係数と角度が変わる | IID生成自体は新しくない。局所query追加が利得になる条件と、共通角の一括合成を対照にも許した価格が不足 |
| [Aomoto–Kato 1988](https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf)、pp.62–65、(1.6)–(1.14), Lemma1.1 | 自由積のGreen multiplierと再帰 | G6のZ2 involutionへの特殊化は既存監査で代数対応済み | 新しいGreen定理ではない。有限alternating Taylor質量・正値性・samplingとの結合全体の先行性は別 |
| [Zhao–Yuan v2](https://arxiv.org/pdf/2103.07988v2)、§4.2(24)–(29) | 高次のidentity・低次数unitary寄与の吸収、modified Taylor | returnを吸収する原理は既知。G6はQ²=Iだけのformal reduced wordで同じ有限Pmを保持 | 残余を再構成する近似、反交換の利用、LCUとHadamard平均の差を揃えず完全同値としない |
| [SCU/CTS v2](https://arxiv.org/html/2407.21095v2)、Theorem1、Suppl.Note3/5 | Pauli collection＋Euler、layer別samplingもあり、完全展開を避ける経路がある | G9のmatched CTSは同じP5の限定operator specialization。一般RAはPauliに追加還元しない | channel左右独立drawとcoherent first momentのnormalization冪を整合する。層間returnを局所queryで取得する同等経路・そのbit/cost差は未決 |
| [Resource-Optimal IS v1](https://arxiv.org/html/2603.13495v1)、Thm1(9)–(13)、Thm2(21)–(23)、IV.1/2 | 固定protocolの費用×weight二次moment、bias不変、ZeroFill/Discard | 固定辞書で qᵢ∝cᵢ/√Cᵢ は既知解。未知normalizerを避ける拒否も新原理ではない | zero-C、range・ceil・bias・hのある有限confidence目的はleading proxyの最適性と別。outer配分が辞書と角度を変えることはfixed-law ISだけでは完結しない |
| [SPS v2](https://arxiv.org/html/2402.15550v2)、Statements1–3、(1)–(12)、App.B.2/B.3 | gate processのsigned辞書分解、L1・residual・T費用と反復の交換 | 固定角libraryへの包含、signed controlled-processへの埋込みは既存Parent監査に整理済み | operator平均を同じ量子processと同一視しない。連続zが角度・columnを変える全探索、情報取得・sampling law・native実装の完全同値は未確定 |
| [PTSC v2](https://arxiv.org/pdf/2212.04566v2)、§IV.A(42)–(50), Prop.3 | identityと先頭項のEuler pairing、finite compensation・sampling | first-pairの近接例。R0.5に既存対応がある | 全隣接pair・全formal returnを同じfinite targetで生成する手順との完全同値は未証明。D3に独立PTSC armはない |
| [PR v2](https://arxiv.org/pdf/2503.05647v2)、App.A.2(A18)–(A29) | RTEのphase-sensitive mean、normalizationと反復、PR全taskへの接続 | Track Bはrandom部分の有限operatorと条件付きnative比較 | 分子DF、deterministic側、長時間、prep、RPE/QPE全体の資源利益はTrack Bでは未取得 |
| [Ross–Selinger v3](https://arxiv.org/pdf/1403.2975v3)、§7–9の該当algorithmと条件 | Rz近似のT費用・算術条件、phaseを含む問題区分 | 保存pygridsynth価格は固定toolの合法sequence価格 | 最小Tの証明、別seedでの同順位、whole-circuit最適性にはしない |

## 2. A：既存構成を演算手順として照合

〔既存：G6監査・技術ノート、SOURCE静的確認〕

1. p tableを読む（入力自体に少なくともL個の情報）。Fᵢ,Gの有限級数を準備し、reduced parent uとchildの係数を局所取得する。既存の算術countはsetup O(Lm²)、係数保存 O(Lm)、有効parent query O(m³+Lm²)。有理演算countはbit演算・秒数・量子費用ではない。
2. ordinary order-envelopeからraw IID語を選び、非reducedなら**zero**。語をreduceして再割当てする手順ではない。局所受理とchild選択を行い、受理時にparent-dependent回転とordered wordを記述する。
3. idealでは受理率 Z=B_RA/B_ord、全試行分母の二次momentは B_ord B_RA、rangeは B_ord。accepted-only平均にしてはならず、未知Zを既知値として十分Nへ代入できない。取得可能な上界Uを使う既存予算を区別する。
4. finite-bitでは実dyadic確率πに対し有理weight c̃/πを付ける。πW=c̃の有理等式は成立するが、元のirrational係数cと同じmeanではない。既存の係数bias、角度・native誤差、range増加を別々に課金する。固定precision不足はfail closed。
5. P5専用経路はroot、長さ2、長さ4の群でnormalizerを求め、局所wordを選ぶ。群setupのO(L²)と、全selector・root精度・cache・online合成の費用は同一ではない。G9は汎用kernelとは別の具体的なfast pathである。

〔技術推論〕これは「operatorをcollectした」という一文より具体的な構成成果である。しかしordinaryのIID samplingも非列挙であり、CTSもlayer samplingを持つ。**非列挙であることだけでは対照に対する計算量分離にならない。** G6のGreen特殊化とfinite-bit構成が有効であることと、全結合の独立した方法差が同情報で残ることは別の問いである。

native利益の既存証拠はcheap I1が明示された小syntheticに限定され、そこではCTSにも同じPauli情報を与えられる。これを「Pauli情報を得られない対照より有利」というaccess分離には使えない。逆に一般involutionをPauliへ高価に展開する方式だけを対照に据えて勝利を作らない。A1（既知原理への包含）は部分ごとに確認済み、A2（生成・取得の独立差）は全結合について未決、A3（実資源への寄与）は固定例のみ、A4（既存証拠だけで論文の核を支持）は不足と整理する。

## 3. Parent allocation：辞書変更とISを混同しない

〔既存：10/14方針・Parent対応監査・S4完成レビュー〕

非負配分 Σzᵢ=a に対し cᵢ=√(zᵢ²+bᵢ²)、φᵢ=atan2(bᵢ,zᵢ)として、既存の平均保存式は

Σᵢ cᵢ ζ exp(−iσφᵢQᵢ)Q(u) = ζ(aI−iσΣᵢbᵢQᵢ)Q(u)。

回転軸Qiがinvolutionであることが必要で、任意wordをinvolutionとしない。z=0のpure endpointでもancilla相対位相と順序は保存する。共通角 zᵢ=abᵢ/Σb のL1最小性は線形分割・Euler・三角不等式からの既存結果。

固定正価格のinner ISについて (Σcᵢ²/qᵢ)(ΣqᵢCᵢ)≥(Σcᵢ√Cᵢ)² と等号lawは既知。outer zではcと角度・価格も変わる。S4の訂正KKTは**旧commonの固定価格**に対する Σ√Cᵢ√(zᵢ²+bᵢ²) のsurrogateで、native費用の最適解ではない。旧raw-C proxy20 lawを訂正KKT20 lawとして再利用しない。

SPSのprocess Ad_C(V)にはoperator平均以外のdiagonal actionもある。既存監査のsigned off-diagonal embeddingは数学的包含の道具であり、通常のunitary gate processを同じlaw・回路・費用で実現する証明ではない。有限libraryのLPはlibrary内の主張で、連続角・辞書取得を自動的に最適化しない。このため「SPSに包含されるから全手順既知」「ISと違って辞書を変えるから新アルゴリズム」のどちらにも飛躍がある。

## 4. B：費用モデルの既知部分と未取得の方法論

〔既存：D3/S4〕実lawのmoment ν、range R、全bias βに対して残余統計幅 s=ε−β>0を用い、同じ既存Bernstein十分N規則を比較する。Nは実測shotsでも必要最小shotsでもない。ISを変えても同一implemented decompositionのbiasが改善したとはしない。

S4の資源座標jでは Gⱼ(hⱼ)=2N(E[Cⱼ]+hⱼ)。D系列は受理率・境界相殺を含み、Gᵢᵣⱼ=2Nᵢᵣ(rZᵢʳdᵢⱼ−Sᵢᵣⱼ+Zᵢʳhⱼ)。hを課す回数が異なるため系列をそのまま合算しない。Tはnative1Qの部分集合で、T+1Qのscoreを作らない。depthはgate countから自動取得しない。expected costと高確率hard capも別。

〔一次＋技術推論〕SCU補足4.1にはspectral/statistical errorの分離とshots×CNOTが既にあり、SPSにはlibrary・反復・Tの交換、ISには固定protocolのnet costがある。共同会計の必要性自体を新方法論とはしない。D3/S4は価格や保証が方式順位を変える**具体的な事後診断**を付加しているが、未取得入力での選択を予測する独立selectorはない。

S4の約14%／19%は保存best-to-bestで配分とproposalが同時に変わり、RA固有の効果でもない。単一のh=0切片を一般的選択境界にしない。保存pairの交点は全70 lawの最適包絡線ではない。多stepの頂点に関する係数質量・二次moment下界も、最適化後の総T比の下界ではない。これらから新定理や新しい実用閾値を本調査で構築していない。

**技術分類**：Aの全結合を既知と断定する証拠は不足するが、独立資源貢献の立証も不足。Bは診断例を持つが一般的な新方法論は未成立。追加provider・r・seed・精度探索を自動開始しない。
