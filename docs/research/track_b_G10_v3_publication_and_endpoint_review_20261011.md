# Track B：構成ノートv0.1の論文化可能性・新規性・研究の着地点に関するレビュー

- 作成日：2026-10-11（JST）
- 開始承認：利用者の「レビューを開始して」。対象は直前に提案した論文化・着地点のレビュー。
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 対象branch：`track-b-g10-v3-scientific-review-intake-20261010`
- 対象commit：`73a7abd6547830d8c09c95637c978855af83a01e`
- 文書本体のparent commit：`f41d5e3d6c12a46c7ea387c064346e160a79936a`
- 主対象：`docs/manuscripts/track_b_return_aggregation_native_resource_note_v0_1.md`
- 本レビューの状態：完了。G10の科学的結果・順位の再レビューではない。
- 新規科学実行・repository変更：なし。

## 1. 最終判断

**現時点では、独立した新規アルゴリズム論文として投稿へ進むことや、そのための追加性能探索を推奨しない。一方、一般構成・P5の具体的実装・native資源上の限定的な否定結果を、自己完結的な技術・方法ノートとして完成させる価値はある。**

したがって、次の作業は新しい科学計算ではなく、既存のG6/G9/G10資料を使った一回の原稿統合と、現系列の区切りの明確化とする。実務担当はCodex。m9、新しいp/x/provider/seed/precision、proposal最適化、再合成、DF/分子への展開、G11は認可しない。

この判断は「native費用で負けた理論構成は論文にならない」という基準によるものではない。一般構成の内容は存在する。しかし、既知の構成要素に対する結合全体の優先性・非自明性と、それを主貢献とする科学的理由が、現在の原稿と確認した証拠では十分に切り出されていない。原稿を整えることと、新しい研究貢献を獲得することを分ける。

また、「技術ノート」と「独立した文書」は対立しない。技術ノートも単独で読める成果物にできる。ここで採択しないのは、**優れた一般PRアルゴリズムを確立したという主要claimで研究・投稿を進めること**であり、記録の価値や将来の出版可能性を一律に否定することではない。査読採択を予測・保証する判断でもない。

### 今回のレビューで確定すること

| 対象 | 今回の判断 |
|---|---|
| G10の既存科学的結論 | 維持。今回の対象はその再判定ではない |
| 一般full returnの性能上の主役化 | 採択しない |
| 一般構成と有限bit処理 | 報告すべき具体的な技術内容がある |
| P5の少数群による実装 | 原稿中で明示する価値がある。単なる実装ログに埋めない |
| 固定辞書のsampling-only救済限界 | 強い適用範囲限定の事例として残す。新しい普遍的最適化定理とはしない |
| 現在のv0.1をそのまま独立原著として投稿 | 推奨しない |
| 既存証拠に限定した自己完結的な技術ノートへの完成 | 推奨 |
| 追加科学計算を論文化のために開始 | 今回は不要・未認可 |
| 次の文書完成後の同一レビュー反復 | 原則不要。実質的な新claim・新方針が生じた場合だけ再判定 |

これは本レビューの研究上の推奨であり、既存のauthorizationやmandatory STOPを書き換える操作ではない。

## 2. 資料、取得状況、前回レビューとの分担

### 2.1 今回の正本

主対象は構成ノートv0.1 [R1] とclaim/evidence表 [R2]。一般構成の証明・仮定にはG6数学監査 [R3]、有限bitと計算量にはG6 access監査 [R4]、P5実装にはG9結果前契約の導出部分 [R5] を用いた。先行研究との対応は、G6文献監査 [R6] を読み、関連する一次文献 [P1–P6] を別途参照した。

GitHubのbranch情報からHEADが`73a7abd…`、そのparentが`f41d5e3…`であることを確認した [R9]。引き継ぎの一部リンク・公開receiptがparentを指すことは、文書本体と後続の公開記録を区別したものであり、別の科学実行やref誤りと解釈しない。[R7,R9]

レビュー開始を妨げる必須資料の欠落は確認していない。今回の判断に不要な巨大raw結果の再取得や、原結果136分割の再結合を新たな開始条件にはしなかった。

### 2.2 固定された科学的来歴

| 役割 | Identity |
|---|---|
| Science S3 | `b9ed01455351628c9073748f5ba5751aa794b789` |
| Authorization / 実行HEAD A3 | `53a7bc4ca8051bfd76343e98f3122c0f198e37d0` |
| G10 v3原結果 | `fcd3ea6217bc00b667180cec149a70102d75f07e` |
| 取得性改善 | `e65e3c0680fff4cfc4243ed5d6c87429cd5ce753` |
| 文書化本体 | `f41d5e3d6c12a46c7ea387c064346e160a79936a` |
| 今回のレビュー対象・公開引き継ぎ | `73a7abd6547830d8c09c95637c978855af83a01e` |

原結果は66,842,494 bytes、SHA256 `64a867dd6cc8f5f535880607616dea72f13af1360c7468f91eef44f79c543a2f`。本レビューではその新規再ハッシュを実施せず、既存レビューと公開資料のidentityとして扱う。[R7,R10]

### 2.3 今回実施したこと・していないこと

今回実施したのは、原稿の主張と根拠の対応、一般構成・P5実装の技術的中身、最接近文献との関係、原稿の自立性、追加研究の必要性、研究の着地点の判断である。文献は関連する定理・式・sampling手順を確認したが、全関連研究・全版の網羅的priority調査ではない。

G10の17行の順位、全10,936 event、57,021保存監査条件を独立に再計算したわけではない。native synthesis、回路・Hamiltonian行列計算、samplerの新実行、budget再最適化、新規入力の検証をしていない。3図については原稿本文・captionが与える意味を確認したもので、今回PNG/SVGの画素・レイアウト品質の全点検を実施したとはしない。

文献PDFはWan–Berta–CampbellのAlgorithm 2、Zhao–Yuanの§4.2関連ページを画面でも確認した。Aomoto–Katoについては取得できた解析テキストとG6の対応導出を参照した。PDF画像取得は成功しておらず、当該対応の全数式を独立に再証明したとはしない。

## 3. 現在の研究には何が残っているか

### 3.1 一般の全形式returnを、局所生成と有限bitへ接続した構成

資料に記載される一般対象は

\[
M_m=P_m(-i\sigma xR),\qquad R=\sum_{i=1}^{L}p_iQ_i,
\]

である。仮定は正の有理数p_i、\(\sum_i p_i=1\)、Hermitian involution \(Q_i^2=I\)、奇数m、\(0<x\le1\)、\(\sigma=\pm1\)。x=0はidentityとして別扱いする。[R3,R4]

ここで「全return」とは、**隣接した同labelを消去する関係\(Q_i^2=I\)の下で、同じreduced wordへ帰着する全形式語の寄与**である。実際のPauli演算子の交換・反交換、偶然の一致など、追加の物理的代数関係に由来する全相殺まで取り込んだという意味ではない。

資料の係数は

\[
a_u=\sum_{\substack{|u|\le n\le m\\n-|u|\ \mathrm{even}}}
(-1)^{(n-|u|)/2}\frac{x^n}{n!}P_n(u),
\qquad
M_m=\sum_u(-i\sigma)^{|u|}a_uQ(u).
\]

G6は、挿入上界からshort-stepでの正値性を示し、既知free-product母関数を局所係数計算へ用いている。偶数parent uと奇数child iuを、\(s_u=\sum_i a_{iu}\)、\(d_u=\sqrt{a_u^2+s_u^2}\)、\(\phi_u=\arctan(s_u/a_u)\)でまとめ、

\[
V_{u,i}=(-i\sigma)^{|u|}e^{-i\sigma\phi_uQ_i}Q(u)
\]

に対する正係数ensembleを構成する。回転generatorは単一のQ_iであり、一般の奇数語がinvolutionだと仮定しない。[R3]

一般生成法は、ordinary envelopeによるproposal、非reduced raw語のzero化、局所acceptance、child選択を組み合わせる。**raw語をreduceしてその行き先へ無条件に再配分する方式ではない。** 理想的なglobal normalizerを生成器の入力に必要としない一方、その未知量を無料のshot削減情報に使わない。[R3,R4]

さらに、有限bitの実proposal \(\pi_e\) と近似係数 \(\widetilde a_e\) に対し、重み \(\widetilde W_e=\widetilde a_e/\pi_e\) を有理数として補正する。確率丸めを補正した平均は\(\widetilde M\)であって、一般には元のMへの厳密一致ではない。係数近似biasとnative誤差の費用を別に残す。[R4]

**評価：これは単なるアイデア名ではなく、報告可能な具体的構成である。** 一方、その各部品の既知性と、結合したアルゴリズム全体の新規性は異なる。今回の確認で、全く同じ有限target・分布・access・有限bit保証を備える単一の先行方式を特定したわけではない。しかし、それをもって独立新規性や非自明性を確定したとも言わない。

なお、ordinary RTEも非列挙であることは、この構成の技術的価値を消さない。ordinaryと全形式return集約後では出力する分布が異なるため、ordinaryのsamplerをそのまま呼べば目的の集約分布が得られるわけではない。「指定した集約分布を全列挙せず構成すること」と「既存RTEより一般に速く、安くすること」は別の主張である。前者の具体的な構成内容を残し、後者の未証明な優位を加えない。[R3,R4,R6,P1]

### 3.2 計算量の材料も存在する

G6の記録には、前処理\(O(Lm^2)\) rational operations、\(O(Lm)\)個の係数保持、有効parentの一試行query \(O(m^3+Lm^2)\)という算術量がある。係数bit長とroot/probability精度も別会計される。[R4]

従って「古典処理について何も示されていない」は不正確である。正しい限定は次の通り。

- 算術operation数をそのままbit operation数や実時間と同一視しない。
- mについての多項式性は、mを数値として扱う評価であり、log mだけに多項式という主張ではない。
- controlされたQ_i、word依存回転の合成、cache・測定などの実費は、この係数queryの算術量に含まれない。
- 本番の固定H=160、K=256が全入力で常に十分だとはしない。一般のprecision選択可能性の議論と、固定設定で不足時にfail closedする実装を分ける。

これらは原稿を自己完結させるために既存資料から移すべき事項であり、原稿化のために新しいbenchmarkを必須化する理由ではない。

### 3.3 P5の具体的な少数群構成は、原稿で過小評価しない

G9の導出は、P5の同じ理想ensembleを、root、ordered pair、長さ4のfirst-labelに依存する群で表す。群数は

\[
1+L(L-1)+L=L^2+1
\]

以下であり、群構築は\(O(L^2)\)の有理算術operation、local conditionalは\(O(L)\)として説明されている。全child表を作らず、短いcompletion DPを使用する。[R5]

これは、汎用full生成器を単にm=5で呼んだというだけではない。normalizerを扱いやすい形へまとめた具体的な実装選択であり、native費用の成果が残った構成を読者が追うためにも重要である。

**評価：P5の式、群分け、計算量は技術ノートの主要な構成内容として示す。** ただし、return吸収の考え方が既知であることからP5のこの全手順まで既知と断言しない。逆に、今回それと同じ式を見つけなかったことから「世界初のP5アルゴリズム」とも書かない。現状では、具体的に提示できる構成と、その優先性の確定を分ける。

## 4. 最接近文献との関係

本節の文献側の説明は一次本文に基づく。右側の評価は、本研究資料との比較によるレビュー判断である。

| 既存要素 | 一次資料で確認した内容 | 現研究との関係・扱い |
|---|---|---|
| ordinary RTEのpairingと生成 | Wan–Berta–CampbellのLemma 2／Algorithm 2は次数とIID labelから回転を含む語を生成する [P1] | 非列挙性・一回転という形だけを新規性にしない。return集約後の局所分布とは別に比較する |
| 高次寄与の吸収 | Zhao–Yuan §4.2は高次寄与をidentity・低次数unitaryへ戻す [P2] | 吸収原理は既知。ただしmodified Taylorの近似targetと固定finite P_mの同一性を無検証に仮定しない |
| free-product母関数 | Aomoto–Kato §1とG6対応導出 [P3,R3] | 母関数自体は既知式の特殊化。有限Taylor samplerへの接続を説明対象にする |
| CTSの収集・sampling | Peetz–Smart–Narang Theorem 1／補足Note 3,5 [P4] | Pauli収集と層samplingの両方がある。CTSを必ず全語列挙する方法として扱わない |
| cost-aware importance sampling | Cugini–Atif–Subaşı Theorem 1は費用と二次momentの積を扱う [P5] | 基本不等式とproposal最適化は既知。G10の固定辞書・native費用への適用を区別する |
| PRのRTE位置付け | PR元論文のRTE導出・involutionの仮定 [P6] | 一般involutionを使うことだけを、既存PR/RTEからの独立した新規性としない |

### 4.1 重要なのは「何が同じで何が違うか」である

比較のためには、少なくとも対象演算子、使用する代数関係、出力する分布と重み、必要な入力情報、係数・角度取得費用、finite-bit誤差を対応させる必要がある。

本研究の全形式returnとCTSの全Pauli収集は異なる。同じfinite P_mへ特殊化してoperator meanを比較できても、dictionary、相殺の範囲、古典取得の課題は同一ではない。[R6,P4]

また、CTSの層samplingは大きい展開を避けられる一方、層間の簡約を取り逃す。これに対して本研究は指定finite step内の全形式returnを保持するという差候補を持つ。しかし、これはPauliの全簡約を保持することや、CTS family全体より低費用であることを意味しない。[R6,P4]

### 4.2 現時点で独立原著の主張として弱い箇所

現原稿は一般構成を定義しているが、最接近の既存手順ではどの同一課題が処理できず、本構成がどの保証を追加するのかが、独立した主命題として十分に表に出ていない。

例えば「非列挙」はordinary RTEにもある。「全return」は既知Green計算への接続を説明しなければ、既知数式を記号変換しただけという反論が残る。「低いnormalization」はnative総費用の改善と同じではない。「有限bit」は正しい実装に必要な内容だが、それを組み込んだことだけで新しい複雑性上の優位を示したとはならない。

一方で、既知部品の組合せから具体的な新アルゴリズムが得られることは論理的に可能である。本レビューはその可能性を排除しない。**ただし、現在の資料から、既知部品の整理を超える中心貢献を確立したと認定するところまでは進めない。** このため、原著化のための追加性能探索より、現内容を正確な技術ノートとして完成させる方を優先する。

## 5. 固定辞書下界は、どの種類の成果か

### 5.1 新しい普遍的IS最適化定理ではない

固定された正係数a_j、native価格T_j、proposal q_j、共通準備費用hに対し、

\[
M_2(q)=\sum_j\frac{a_j^2}{q_j},\qquad
C(q;h)=\sum_jq_j(T_j+h)
\]

を考える。\(\sum_jq_j\le1\)の残りは量子実行前のzero trialとする。このとき

\[
M_2(q)C(q;h)\ge\left(\sum_j a_j\sqrt{T_j+h}\right)^2
\]

はCauchy–Schwarzの形であり、既知importance samplingのcost–moment関係と同じ基本構造である。[P5,R11]

さらに、\(T_i,T_j,h\ge0\)なら

\[
\sqrt{(T_i+h)(T_j+h)}\ge\sqrt{T_iT_j}+h
\]

である。両辺の二乗の差は\(h(\sqrt{T_i}-\sqrt{T_j})^2\ge0\)となる。この代数からaffineな下界が得られ、登録Bernstein十分予算の正のrange項とceilingを落とすことで、原稿の下界へ接続する。

この段落は下界の性格を説明するための既知不等式の整理であり、新しいG10の数値検算・proposal構築ではない。これを「新しいresource-optimal samplerを提案した」または「あらゆる量子推定の最低shot数を証明した」と表現しない。

### 5.2 残る価値は具体的な比較クラスの分離

G10の意味がある点は、その一般原理を、固定native合成列と有限精度・confidence policyを持つ辞書へ適用し、**full側の任意proposalに対する下界と、対照側の具体的な登録予算を比較したこと**である。[R1,R10,R11]

既存結果では、m=7のfull下界からP5＋tailの登録費用を引いた差は、表示値で

\[
5{,}415{,}782.443240-5{,}537.20852461h
\]

であり、\(0\le h\le970\)で正である。約978.0709はこの下界による分離の端であり、実行可能なlawの勝者が逆転した点ではない。今回はこの数値を再計算したのでなく、前回の採用済み科学レビューと現在の原稿の主張として扱う。[R1,R10]

**評価：既知原理の具体的適用だが、単なるcanonical順位表より強い限定的な事例である。** 原稿のresource-limits節の中心に据える価値はある。ただし、新規最適化定理として独立原著を支える根拠には置かない。

## 6. 限定否定結果だけでも何を言えるか

論理的に、「係数質量またはsupportが小さくなれば、同じ精度でnative総費用が必ず減る」という普遍的含意を否定するには、条件のそろった反例が一つあれば足りる。多数の追加系を並べなければ反例として成立しない、という条件は課さない。

しかし、その一例から改善・悪化が一般的にどの程度起きるか、装置や合成backendを変えても同様か、一般fullが全入力で不要かまでは分からない。ここでは反例の存在と、一般的な性能傾向を分ける。

さらに、量子回路の単価と必要試行数を同時に考える必要自体は、既存cost-aware ISの動機にもある。[P5] したがって原稿の新規性を「ノルムだけでは費用は決まらないと初めて発見した」という一般論に置かない。

報告すべき具体性は、同じfinite first operator moment、比較可能なnative実装、係数質量と推定momentの分離、root回転価格の変化、任意proposal下界、CTSとの別資源trade-offを一つの追跡可能な事例として残すことである。[R1,R2,R10]

一つのdevelopment providerであることは、本ノートの主張を限定する理由であって、現在の反例・構成資料を価値のないものとする理由ではない。DF、分子、PR全体、QPEの追加実験を、今回の技術ノートの完成条件として要求しない。

## 7. v0.1の原稿としての評価

### 7.1 既にできていること

v0.1は、各次数内で同じfinite targetを比較すること、idealとfinite-bit/nativeの差、NとK、T/CX/1Q、下界端点とwinner crossover、binding数と実時間の違いを明記している。現在のclaim/evidence表とも対応しており、前回レビューの重要な限定を維持した研究記録として有用である。[R1,R2]

特に、0≤h≤970を固定辞書・固定policyに限定し、m=3/5や他のcompilerに広げていない点、CTS全体への支配を主張していない点は維持すべきである。

### 7.2 そのまま投稿用原稿とは判定しない理由

主に不足しているのは、単独の読者が研究内容を確認できる構造である。

**第一に、一般構成の根拠が別資料への参照に寄っている。** §2には集約係数とparent pairingがあるが、正値性、実際の生成手順、unknown normalizerの扱い、finite-bit meanとbias、計算量の結び付きが本文・付録だけで完結していない。[R1,R3,R4]

**第二に、P5が結果表の名前に近い扱いになっている。** P5の具体的な群構成と\(L^2+1\)の根拠を示せば、低次数の実装が一般器とどう異なるかを読者が理解できる。これは既存G9に資料があり、新しい理論を作る要請ではない。[R1,R5]

**第三に、一次文献との関係がGPTレビュー・内部監査を介している。** 内部レビューは研究管理上の資料であって、既知性や数学的成立の最終的な根拠ではない。原稿本文は、原論文・定理・式と自分たちの構成を直接対応させる必要がある。[R1,R2,R6]

**第四に、運用上の判断と科学的主張が混在している。** source/commit/STOPや57,021監査項目の情報は重要だが、学術的内容の主役ではない。本文から削除して証拠を失うのではなく、再現性・来歴の付録に移す。数学的正しさの根拠は証明、native結果の根拠は入力・費用モデル・保存値である。

この四点は、v0.1が依頼された「採用レビューに沿う構成ノートの初稿」であることと整合する。文書化作業の失敗と判定しているのではなく、研究記録から独立した読者向け文書への次の整理範囲を特定している。

### 7.3 完成させても自動的に原著の新規性が増えるわけではない

証明と文献を本文へ移すことで、内容の確認可能性は高まる。しかし、それによって既知の内容が新しくなったり、一般的な性能分離が得られたりはしない。

したがって「文書の不備を直せば独立原著として投稿十分」とは判定しない。今回認めるのは、**既存成果を技術ノートとして完成させるだけの中身があること**である。優先性や主要な新規性を新たに確定したと主張する場合には、その追加主張を支える具体的な根拠が別に必要となる。

## 8. 研究の着地点と代替案

| 選択肢 | 判断 | 理由 |
|---|---|---|
| 一般fullのnative優位を主題とする原著 | 採択しない | G10の主要比較がその追加利益を支持していない |
| 一般構成の新規理論論文として直ちに投稿 | 現段階では推奨しない | 構成材料はあるが、最接近研究からの中心的な新規差を十分に切り出していない |
| 構成とnative限界の自己完結的な技術・方法ノート | **推奨** | 現在の証拠に見合い、一般構成・P5・固定classの限界を保存できる |
| Track AやDF研究へ直ちに統合 | 今は採択しない | 別研究のtask・cost指標・claimへ自動的に移せない |
| 負結果として全資料を破棄 | 採択しない | 構成、finite-bit処理、具体的実装、限定的分離には再利用価値がある |

技術ノートとして仕上げた後に、著者が独立のtechnical reportとして公開するか、別研究の補足に使うか、研究室内の記録に留めるかは、公開範囲・著者間の判断として別に決められる。このレビューでは投稿先、著者順、公開日を決めず、外部への送信・投稿も行わない。

**本レビューの推奨は、独立原著への格上げを目的に新計算を始めず、現在の有限Taylor return集約系列を技術ノートとしてまとめて区切ることである。** これはPR内部改善という研究領域全体の終了判断ではない。

## 9. 最終ノートで据える中心メッセージ

以下は本レビューが提案する原稿の位置付けであり、v0.1に既に完全な形で書かれているという主張ではない。

> 有限Taylor演算子について、involution関係による全形式returnの集約を、局所生成・有限bit処理と結び付けて記述する。P5では少数群の構成として具体化する。固定nativeモデルの比較を通じ、係数質量・supportの削減と、準備費用・合成費用・十分予算を含む総資源の改善を区別し、追加集約の利益が得られない場合と、samplingだけではその差を埋められない比較クラスを示す。

このメッセージは、一般構成を破棄せず、同時にnative優位のない一般器を性能上の主役へ戻さない。

### 推奨する自己完結的な構成

1. **対象と問い。** 同じfinite operator meanの複数実装、形式returnの定義、native総費用を別に評価する目的。
2. **構成。** G6の正値性・parent pairing・局所生成、finite-bit mean/bias、input/accessと算術量。長い証明は付録でよい。
3. **P5特殊化。** 群構成、normalizer、生成方法、一般器との関係。
4. **評価モデルと既存結果。** 同一次数の比較、合成・位相・予算、native資源と準備単価感度。
5. **固定辞書の限界。** 下界の導出とclass、具体的な分離、成立範囲外。
6. **関連研究と限界。** 既知要素とこのノートが明示する組合せ、未確定の優先性、古典scaling/PR全体への未検証性。
7. **付録。** 入力、source/result、失敗runの保持、保存監査、取得・再現手順。

これは章番号の強制ではない。原稿が一つの読者向け文書として成立し、証拠の所在が分かればよい。3図の追加を義務付けず、既存図の役割とcaptionを本文の主張へ対応させる。下界・native trade-offを主図、m3境界を補助図にする構成は一案に留める。

## 10. 追加検証は必要か

### 10.1 今回の完成方針には、新しい科学計算は必要ない

一般構成のproof材料、finite-bit/access、P5実装、同一taskのnative結果、固定辞書下界は、既に別資料に存在する。現在のclaimのために必要なのは、これらを正しい区別のまま統合することである。[R1–R5,R10,R11]

「理論構成がある」と書くために量子実機や分子の実験を要求しない。「固定モデルで単調改善しない例がある」と書くために多数の新providerを要求しない。「同じfixed-dictionary policyで分離がある」と書くために最適proposalを実際に探索する必要もない。

追加scopeが必要になるのは、現在の主張を拡大するときである。例として、入力群全体での優位、全体PR/QPEの資源削減、新しいrepresentation、一般native合成モデルでの保証、既存法に対する別の計算量分離を主張する場合には、その主張に対応した研究判断が必要になる。

この区別により、単に「論文にするにはデータを増やす」という無期限の検証へ進まない。

### 10.2 文書整備中に見つかった問題の扱い

引用不足、記号不統一、すでに証明された補題の記載漏れ、図captionやリンクの問題、既存保存値の転記ミスは、Codexが採用scope内でまとめて修正してよい。

一方、既存の一般証明に本質的な反例が見つかった、必要な仮定を変える、結果を別の予算規則で再評価する、主要比較を差し替える場合は、単なる文書修正ではない。その具体的内容を報告して新しい研究判断へ戻す。この条件は形式的な細分化ではなく、claimの意味が変わる境界である。

## 11. 次の担当と完了条件

### Codexへ渡す作業範囲

既存scopeを維持したまま、ノートを一回の作業で自己完結化する。G6数学・finite-bit/access、G9 P5導出、G10評価・下界・来歴を本文と付録へ整理し、一次文献との対応を明記する。原結果・source・contract・authorization・marker・STOPとTrack Aを保護する。

本レビューを採用した研究判断として保存する場合も、科学的sourceや原結果の分類を変更しない。文書の新versionを作ることは、新しい科学的実行承認を作ることではない。

### 完成の目安

- 読者がチャットやGPTの判断文を読まずに、対象・仮定・アルゴリズム・保証・比較classを追える。
- 一般構成、P5特殊化、固定native事例が区別される。
- ideal exact mean、finite-bit bias、native誤差、statistical budgetが混同されない。
- 既知原理を自分たちの新定理として書かず、未確定の優先性を肯定も否定も断言しない。
- 「最適」「一般に優れる」「PR/QPEを改善」の未支持claimがない。
- 必要な根拠への固定参照と、原結果不変の来歴が残る。

これらを満たしたら、現在の成果の文書化は完了として扱う。同じG10の結論や同じ論文化論点を、versionが増えるたびにGPTへ戻して再承認させない。著者が別の主要claimや投稿方針を具体的に提案する場合には、その新しい点だけを対象に必要性を判定する。

## 12. 前回GPT再現ZIPの扱い

公開引き継ぎは、前回レビューの別添ZIPがCodexへ渡されておらず、独立対数certificateなどを再実行したと主張していないことを明記している。[R2,R7,R8]

この会話環境には当該ZIPと旧レビューMDが存在する。本レビューで旧MDのbytes・SHA256を照合したところ、46,672 bytes、SHA256 `0253b9483b3f7b0082ef849cd164d70fc7e2e81b8bbca1e89c5af8b6c5cf910a`であり、repositoryのreview-intake identityと一致した。ZIPのファイル一覧は確認したが、内部の検算scriptを再実行していない。

これは、「成果が未pushで科学レビュー不能」という不足ではない。既存の下界source・保存有理数・corrected/final監査が参照できるため、今回の論文化・着地点判断を妨げない。将来、独立検算の再現一式を公開物の一部とするなら、そのZIPを実際に渡してidentity付きで収録するか、再現範囲を現在のsaved-only検算へ限定して正確に書けばよい。提供・再実行していないscriptを再現済みと表記しない。

本レビューは新たな数値検算を実施していないため、新しい「科学計算再現ZIP」を作成したかのような添付は行わない。

## 13. 判断の確かさと限界

現資料の範囲、既知構成要素、P5の具体性、原稿の自己完結性不足、現在のclaimで新規科学計算が不要なことについては、比較的明確に判断できる。

構成全体のpriority、査読者が認める非自明性、特定journalの採択可能性は、今回の限定文献調査から確定できない。したがって「同じアルゴリズムが既にあるので新規性ゼロ」とも、「完全に新しく投稿できる」とも述べない。

この不確実性があっても、研究の次の行動を決めることはできる。現状では、追加性能探索によって独立原著化を目指すより、具体的構成と適用限界を自己完結的に残す方を選ぶ。これは採択率の推測ではなく、現在の主張・証拠・未解決点に対する研究上の優先順位の判断である。

**最終結論：現在の成果には単なる失敗ログを超える内容がある。ただし、一般full-returnの優位を示す新規アルゴリズム論文としては進めない。一般構成、P5の少数群実装、固定nativeモデルでの限界を技術ノートとして完成させ、この検証系列を区切る。Mandatory STOPと新規科学実行未認可は維持する。**

---

## 参考資料

以下のrepository参照は原則としてレビュー対象commitで固定する。文献側は今回参照した版を示す。URLは資料の識別用であり、全ページ・全versionの精読を意味しない。

### Repository

[R1] 構成ノートv0.1。blob `bd172dc84c5b812d779ba24a1646695a544e93b9`。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73a7abd6547830d8c09c95637c978855af83a01e/docs/manuscripts/track_b_return_aggregation_native_resource_note_v0_1.md

[R2] Claim/evidence表。blob `3ac112f5446126257e3b668d0fedcad3aba48170`。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73a7abd6547830d8c09c95637c978855af83a01e/docs/tracks/algorithm_codesign/g10_v3_claim_evidence_map_20261010.md

[R3] G6独立数学監査。blob `2f621562dadfe1fc52367a680372b8d34fff4ef6`。全形式return、正値性、Green対応、parent pairing、zero-fill。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73a7abd6547830d8c09c95637c978855af83a01e/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md

[R4] G6 finite-bit/access監査。blob `4b89c388161b77a640ccbe1916e66397ceeb4234`。当時のnative未確認と、後のG9/G10のexplicit provider確認を混同しない。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73a7abd6547830d8c09c95637c978855af83a01e/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md

[R5] G9 P5 matched-native結果前契約。blob `44207f1610967ef4df9f4f9415c0d3c85fa1ed93`。P5導出と少数群構築。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73a7abd6547830d8c09c95637c978855af83a01e/docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md

[R6] G6 prior-art/method-delta監査。blob `d1f75217af67a2c269648d4ef03118b1febe7b94`。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73a7abd6547830d8c09c95637c978855af83a01e/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md

[R7] 文書化引き継ぎ。blob `5ddee3694d695de7d922f4f7aa3b01317509d34f`。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73a7abd6547830d8c09c95637c978855af83a01e/docs/tracks/algorithm_codesign/g10_v3_scientific_review_documentation_handoff_20261011.md

[R8] 採用レビューidentity。blob `ba108df95d04c225b585b484cb85b83c8eddf0c0`。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73a7abd6547830d8c09c95637c978855af83a01e/artifacts/track_b_g10_v3_scientific_review_intake/2026-10-10/review_intake_identity.json

[R9] Live branch情報（確認時HEAD `73a7abd…`）。
https://api.github.com/repos/HIROMU1015/Partially-Randomized-Trotter/branches/track-b-g10-v3-scientific-review-intake-20261010

[R10] 採用済みG10 v3科学レビュー。今回その結論を再計算していない。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73a7abd6547830d8c09c95637c978855af83a01e/docs/research/track_b_G10_v3_scientific_review_20261010.md

[R11] 固定辞書下界source `g10_saved.py`。既存レビュー時に参照した関数`affine_policy_lower`。今回の下界説明はこの固定式と[P5]の対応であり、新規source実行ではない。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/src/trottertracks/algorithm_codesign/g10_saved.py

### 一次文献

[P1] Kianna Wan, Mario Berta, Earl T. Campbell, *A randomized quantum algorithm for statistical phase estimation*, arXiv:2110.12071。Lemma 2、Appendix C、Algorithm 2、truncation記載を参照。
https://arxiv.org/abs/2110.12071
https://arxiv.org/pdf/2110.12071

[P2] Qi Zhao, Xiao Yuan, *Exploiting anticommutation in Hamiltonian simulation*, Quantum 5, 534 (2021), arXiv:2103.07988v2。§4.2、Eqs. (28)–(29)前後。
https://arxiv.org/abs/2103.07988
https://arxiv.org/pdf/2103.07988

[P3] K. Aomoto, Y. Kato, *Green functions and spectra on free products of cyclic groups*, Annales de l'Institut Fourier 38(1), 59–85 (1988)。§1のGreen multiplier・spectral shift。G6によるZ2特殊化対応と区別して参照。
https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf

[P4] Joseph Peetz, Scott E. Smart, Prineha Narang, *Quantum Simulation via Stochastic Combination of Unitaries*, arXiv:2407.21095v2（2026-07-06）。Theorem 1、Methods IV.2、Supplementary Notes 3,5。
https://arxiv.org/abs/2407.21095
https://arxiv.org/html/2407.21095v2
出版物：npj Quantum Information 12, 52 (2026), DOI 10.1038/s41534-025-01168-w。
https://www.nature.com/articles/s41534-025-01168-w

[P5] Davide Cugini, Touheed Anwar Atif, Yiğit Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1（2026-03-13）。Theorem 1、Eqs. (9)–(12)、ZeroFill/Discardの関連記載。
https://arxiv.org/abs/2603.13495
https://arxiv.org/html/2603.13495v1

[P6] *Phase estimation with partially randomized time evolution*, arXiv:2503.05647v2（2026-07-10）。RTEとinvolution仮定の位置付けに限定して参照。PR全体の新しい費用分析を今回行っていない。
https://arxiv.org/abs/2503.05647
https://arxiv.org/pdf/2503.05647v2
