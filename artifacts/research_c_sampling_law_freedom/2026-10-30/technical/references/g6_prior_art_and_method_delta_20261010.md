# G6：claim単位の一次文献・同値性監査

2026-10-10 JST。新method採択判断はGPT/利用者に帰属する。
同じ入力・対象mean・law・weight・処理量が一致するかを区別した。
「この確認範囲で同じ結合algorithmを特定しなかった」は新規性の証明ではない。

## 一次本文の確認範囲

| 文献・固定版 | 関連本文の確認箇所 | 本回の取得状況 |
|---|---|---|
| Aomoto–Kato, 1988 | pp.62–65, §1 Eq.(1.6)–(1.14), Lemma 1.1 | [一次PDF](https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf)の該当全文。全スペクトル定理の精読ではない |
| Zhao–Yuan, modified Taylor | §4.2 Eqs.(24)–(29), Appendix C Eq.(70)の実装文脈 | [arXiv:2103.07988 PDF](https://arxiv.org/pdf/2103.07988)。本文の低次数吸収箇所 |
| Wan–Berta–Campbell | Lemma 2, Appendix C Eqs.(C4)–(C5), Algorithm 2、truncation記載 | [arXiv:2110.12071 PDF](https://arxiv.org/pdf/2110.12071)。ordinary sampler本体まで確認 |
| Phase estimation with partially randomized time evolution | §IV Eq.(27)–(29), Appendix A.2 Eqs.(A18)–(A26) | [arXiv:2503.05647v2 PDF](https://arxiv.org/pdf/2503.05647v2)。今回v2該当本文を取得。44頁全体の精読とはしない |
| Peetz–Smart–Narang CTS | Theorem 1, Eqs.(5)–(6), layered/Markov sampling本文 | [出版本文](https://www.nature.com/articles/s41534-025-01168-w)。関連節を確認 |
| 同CTS supplement | Supplementary Note 3（層sampling）, Note 5 Eqs.(13)–(15)（Euler pairingの証明） | [7頁補足PDF](https://media.springernature.com/original/springer-static/esm/art:10.1038%2Fs41534-025-01168-w/MediaObjects/41534_2025_1168_MOESM1_ESM.pdf)を今回取得して関連全文を確認 |
| Cugini–Atif–Subaşı | Theorem 1 Eq.(9)–(10), §IV.1–IV.2 Eq.(28)–(34) ZeroFill/Discard | [arXiv:2603.13495v1本文](https://arxiv.org/html/2603.13495v1)。ISとzero outcomesの処理 |
| Koczor, Sparse Probabilistic Synthesis | §II.A–B Eq.(1)–(4), Statements 1–2 | [arXiv:2402.15550v1 PDF](https://arxiv.org/pdf/2402.15550)。process-level dictionary/mean/normを確認 |

GPT G5レビューはCTS補足未取得、PR v1本文のみ、Aomoto–Katoは抄録確認だった。
今回の関連全文確認はその不足を閉じるための新しい文献監査であり、旧レビューが全文確認済みだったと書き換えない。
論文全文・図のcopyrightedコピーをrepositoryへ追加していない。

## Claim比較表

| Claim | 先行研究で既知の内容 | G6との具体的対応・差 | 今回主張しない内容／残る条件 |
|---|---|---|---|
| Return/identity absorption | modified Taylorは高次数寄与をidentityや低次数へ吸収する | G6は同一有限P_mの全形式returnをQ_i²=Iだけで集約。modified Taylorのtail再構成と対象をそろえる必要がある | absorptionやm=3式だけの新規性なし。G6のnon-enumerative lawが当該手順と同じかは別問題 |
| Free-product母関数 | Green multiplierの乗法性と他因子returnによるspectral shift | Z2、p_bar=p/2、zeta=1/zでF_i/Gが**代数的に同じ** | この再帰・非列挙係数query自体を新しいrandom-walk定理と呼ばない |
| Euler pairing | Wan et al.とordinary RTEは隣接Taylor次数を単一involution回転でpair | G6はreturn集約済みeven suffixをparentにし、child全質量s_uでpair。角度はlengthだけでなくparentに依存 | Euler identity・one rotation/word構造自体は既知。新角度取得が有利とは未判定 |
| 非列挙ordinary RTE | 既知samplerはorderとIID labelから生成。全word費用表は不要 | G6も非列挙だが局所coefficient query O(m³+Lm²)を追加 | ordinaryを全列挙evaluatorで代用したclassical勝利を主張しない |
| CTS、Pauli collection | CTSはcos/sinのPauli係数をcollectしてsigned rotationsへ変換 | G6はfree-product形式語であり、Pauli/anticommutationによる追加相殺を利用しない。一般odd語はinvolutionではない | Pauli contextのCTSに勝つ保証なし。両方同じfinite targetへ特殊化して比較する必要 |
| CTS Markov/layering | 補足Note 3は層別samplingで非列挙のunbiased product estimatorを示すが、cross-layer cancellationを取り逃す | G6の局所queryは指定finite step内の全returnを保持するという限定差候補 | 「CTSは必ず全L^m語を展開する」は誤り。non-enumerative CTSと費用・errorを一致させること |
| Generic rejection/zero-fill | success coinとimportance weightで全試行平均を作る原理は既知。Discardは追加normalizerが要る | G6のcoinは量子測定前のclassical local d/(b_l p(u))。known envelopeを使う | zero-fillやnormalizer回避だけでmethod noveltyとはしない |
| Cost-aware IS | fixed mean/protocolでq∝coefficient/sqrt(cost)が既知 | G6の変更はmean representationを先にcollectする部分。さらにcost-aware proposalを使う余地は既知IS | native cost accessなしに「resource optimal」と呼ばない。IS baseline追加・実行は本回未認可 |
| Sparse synthesis | operatorではなくprocess matrix上のdictionary meanとL1 optimizationを提供 | G6はphase-sensitive operator meanと形式word生成。controlされた相対位相を保つ必要 | channel equalityをHadamard operator equalityへ読み替えない。dictionary取得・既知L1原理は新規性なし |

各外部claimの根拠は上表の一次本文・式番号。free-productの対応とG6側の差分は本監査の推論である。
文献にG6の全結合samplerが載ると断定せず、generic要素が既知であることと区別する。

## 「単なる記法変更」への回答

1. F_i/Gは既知free-product Green kernelの直接の特殊化であり、**ここは記法変更と判定**。
2. Pairingとzero-fillは既知構成原理。独立新規性の根拠にはしない。
3. 今回確認した該当節からは、固定finite P_mの全return集約、even-suffix child pairing、ordinary envelope rejection、
   有限bit補正weightを**一体として同じlaw**にした記載までは特定していない。
   ただし既知手順の組合せでも容易に得られる可能性があり、priority・非自明性・科学的価値は未確定。
4. 残る差候補は「同じ有限meanで、returnを取り逃さず、全表/未知B_newを要求せずに生成し、
   word依存rotation・control・classical costを支払っても価値が残るか」だけ。
   その資源差はG6では測定していない。既知componentsの存在から性能勝利を推測しない。

## R0–G5、過去STOPとの関係

- R0/G1/G4-Aのdegree-local・固定辞書・固定policy内の限定分離を保持する。
- G4-B/G5のCTS優越と固定toy同辞書閉鎖を保持する。G6はそのtoyに別scoreを付けない。
- B-Fのfinite-task objective、BM-0.5のcompact-BCH同値性、SPのgate synthesis placementを復活させない。
- G6の形式word集約は高次数からの**signed return**を先にcollectするため、旧非負degree matching classとは対象が違う。
  違う対象であることは新規性・優位の証拠ではない。
- G5のmissing accessを全word tableで埋める案は採らない。局所prototypeはtable/B_newを読まないがnative costsは未取得。

**技術判定**：`COMPONENTS_KNOWN_COMBINATION_PRIORITY_AND_RESOURCE_VALUE_UNRESOLVED`。
研究上のGO/縮小/STOP、新性能pilotの必要性・scopeはGPT/利用者へ戻す。G6後mandatory STOP。
