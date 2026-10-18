# G6の全生成手順と既知法：D1〜D5

〔既存文献監査の再利用〕前回の[固定比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/research/track_b_paper_claim_survey_20261018/prior_art_method_comparison.md)と、[G6方法差監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md)を土台に、M1〜M12の構成だけを照合する。既取得の固定version本文をhash確認して再利用した。実際の節・版・読解範囲は[literature_scope.json](literature_scope.json)。全証明の再検証、追加の網羅的priority探索、著者SOURCE実行ではない。

## 比較する生成手順と入力

対象は同じ有限奇数次数Taylor operator Mで、channelや無限exponentialに置き換えない。G6/G7の経路は、明示p読込 → F/G切断係数 → ordinary次数envelope → raw IID語 → 非reducedなら事前zero → 局所parent/child係数 → 受理coin → child → phase/ordered word/回転のevent記述 → 実dyadic確率によるweight補正、である。Uは予算用であり、B_RAをsamplerへ渡す経路ではない。

対照にも同じp/Q、involution関係、利用可能なPauli/DF情報、Green query、有限精度、state、制御providerを与え、layering、streaming、cache、回路簡約、共通角合成、cost-aware ISを許す。追加のoracleには双方で取得費用を付ける。ordinaryを全語列挙に限定しない。CTSを全Pauli辞書の事前展開に限定しない。SPSに弱い固定libraryだけを強制しない。

## G6の具体的部品との照合

| 既知資料・固定節 | G6で重なる部分：D1 | D2生成／D3アクセスで未確定の部分 | D4精度／D5計算量の留保 |
|---|---|---|---|
| [Aomoto–Kato 1988](https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf)、pp.62–65、(1.6)–(1.14), Lemma 1.1 | M2の自由積Green再帰。Z2特殊化の対応は既存G6監査済み | 既知kernelだけでは、有限signed Taylor係数からparent受理・childまでの全lawを指定していない。構成全体の先行性は別問題 | 切断・有理演算の実装を新しいGreen定理としない |
| [Wan–Berta–Campbell v2](https://arxiv.org/pdf/2110.12071v2)、App.C(C4)–(C5), Algorithm2 | M5のEuler pairing、M6のordinary order＋IID生成に近接 | ordinaryは既に非列挙。G6はreturn収集後のsuffix依存係数・角度と拒否を追加 | 共通角の合成を対照にも許す。局所query追加が速いという証拠はない |
| [Zhao–Yuan v2](https://arxiv.org/pdf/2103.07988v2)、§4.2(24)–(29) | M1の高次寄与の低次・identityへの吸収原理 | 同じ有限Pm、一般involution、実際のlawで還元されるかを区別する | tail再構成、LCUとcoherent信号、精度予算の相違を消さない |
| [SCU/CTS v2](https://arxiv.org/html/2407.21095v2)、Theorem1、Methods IV.2、Suppl.Note3/5 | collection、Euler、layer別samplingという近接部品 | Pauli collectionと一般formal wordは同一でない。層間returnも保持する局所queryを同じ入力で組み込んだ全手順の同値性は未確定 | channelの左右drawとfirst operator momentのnormalizationを揃える。非列挙性だけの分離はない |
| [Resource-Optimal IS v1](https://arxiv.org/html/2603.13495v1)、Thm1(9)–(13)、Thm2(21)–(23)、IV.1/2 | M7/M9のimportance補正、費用と二次moment、zero outcomesの会計 | G6の拒否は量子前の古典処理。論文中の量子error-detectionのZeroFillと物理経路まで同じとはしない | 最小momentだけで費用最適性を決めない。実確率、range、bias、ceil、費用取得も揃える |
| [Sparse Probabilistic Synthesis v2](https://arxiv.org/html/2402.15550v2)、Statements1–3、App.B.2 | signed辞書分解、L1、libraryと費用の交換 | process-levelの包含だけでphase-sensitive operatorの全生成・取得手順が同一とは言えない | 固定library内の最適性を未知のword依存角library生成へ移送しない |
| [PTSC v2](https://arxiv.org/pdf/2212.04566v2)、IV.A(42)–(50), Prop.3 | identityと先頭項のEuler pairing、有限補償とsampling | 全return、even suffix、ordinary-envelope拒否の全結合との同値は未確定 | 異なるtarget・cutoffの比較で新規性や速度差を作らない |

## D1〜D5の結論と必要な証拠

**D1：数学的包含。** Green再帰、Euler式、収集、importance補正、外向き丸めは既知部品を含む。M4の正値域、M5のsuffix pairing、M6/M8/M9の接続は既存の具体的な論証・SOURCEを持つが、部品の既知性から全構成の同値／非同値を断定しない。

**D2：全生成手順。** 残る候補は、一般involutionの同じfinite targetに対し、局所query・事前zero・finite-bit weightを閉じた手順にした構成保証である。「同じものを文献で見つけなかった」は独立性の証明ではない。既知法の許された特殊化を同じaccess/精度で逐次対応付け、何が残るかを判断する必要がある。今回は新しいsampler・同値性証明を作らない。

**D3：情報アクセス。** G6は全語表とB_RAを要求せず、明示p、local query、既知B_o、認証上界Uで進む。一方、p読込、root/確率丸め、controlled-Q、word依存角取得は必要。局所係数アクセスが、既知法に許されない特別oracleになっていないかが中心的な比較条件。general Qの安価なcontrolled実装は未取得。

**D4：有限精度。** 主張できるのはM̃に対するπW=c̃と、Mへのbias・range・精度取得条件の既存保証。元Mのexact sampler、物理RNG、strict native位相certificate、任意入力での自動bit選択までは実装されていない。既知法にも同じ丸め・補正の機会を与えたうえで、接続の構成的価値を判断する。

**D5：計算量。** 全formal辞書を作る実装に対して、G6の局所queryはその前払いを避ける。しかしordinaryも非列挙、CTSもlayering可能。G6のO(Lm²) setup等は有理演算countで、既知samplerに対する下界分離でも実測優位でもない。現SOURCEの線形selector、確率丸めのsort、任意精度整数、取得/cacheを別に記録する。[実装証拠](implementation_and_validation_evidence.md)参照。

## 強い反論への現在の回答範囲

「既知のGreen＋Euler＋rejection＋importance correctionをつないだだけ」という反論に対し、現資料は**どの有限targetを、どの仮定・有限bit lawで生成するか**を具体化できる。ただし、その接続の非自明性・独立した原著貢献の認定は不足する。逆に、部品が既知という理由だけで全law・全精度・全accessの完全同値も認定しない。

量子総資源で普遍的に勝つことは、本構成論文の必要条件に追加しない。独立した保証として価値があるか、資源改善を狙う別主張が成立するかは分ける。資源NO-GOとする場合も、量子前拒否、U、費用付きISを含む条件を落とさない。

〔既存レビューの扱い〕今回の指示§2/§7で共有されたClaude v2の強い資源NO-GOとGPTの修正を、単一の検証済み結論には統合しない。私的原本の取得は[metadata](review_input_metadata.json)で確認できる。原本公開、新しい独立数学レビュー、論文化GOは本書の範囲外である。
