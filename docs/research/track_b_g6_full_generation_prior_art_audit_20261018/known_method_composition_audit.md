# 既知部品の組合せによる再現：対応監査

2026-10-18 JST。これは既存input/output・SOURCE・一次記述の対応表であり、新しいsamplerを構成する作業ではない。[P1〜P12の逐次表](generation_pipeline_alignment.md)と[access/精度条件](access_precision_complexity_comparison.md)を合わせて読む。

## 1. 最も強い反論の置き方

「既知の自由積Green係数を切断し、得た有限Taylor係数を収集し、Euler回転にpairし、ordinary envelopeからrejectionし、実proposalでimportance補正する。したがってG6は既知部品の結合である」という説明は、**G6のSOURCEの部品構成の記述として支持できる**。既存K/Gは実際にこの種の演算・制御フローを持つ。

しかし「部品構成が記述できる」と「既知の独立資料が、その入力・中間係数・law・有限bit保証・費用scopeまで同じ生成手順を明示している」は異なる証拠である。本監査では後者を確定できない。反対に、後者が未確定であることから結合の非自明性・独立新規性も認定しない。

比較対象に共通Green query、streaming、cache、正当なPauli情報、同じcontrolled providerと丸め機会を認める。G6だけへ強いoracleを与えない。ordinaryを全語列挙、CTSを必ず全高次Pauli表へ固定しない。既知法の側へG6の完成済みevent lawを丸ごとoracleとして与え、それを「先行法の生成手順」と呼ぶ循環も避ける。

## 2. 接続点ごとの証拠

| 接続点 | 直接確認できる既知対応 | G6側の既存証拠 | 全先行手順へ還元したとするための不足 |
|---|---|---|---|
| B1 Green→局所return係数 | AK §1のmultiplier/Green特殊化。次数切断と有限係数の代入は既存M2/M3の直接対応 | K `__init__/word_series/_mass`、MATH §§2,5、BIT §2 | AK自体はTaylor係数のsigned収集・正値域・回転lawを指定しない。係数queryの既知性とevent生成の既知性を分離 |
| B2 収集→positive parent/child | Euler式はWAN(C5)/PTSC(45)/SCU(S15)と対応 | M1/M4/M5、MATH §§2–6。short-step正値性、even suffixとchild前置、s/a | 既知Euler式だけではa_uの正値域・全childへの配分は指定されない。これらはG6で既存証明済みだが、先行法の同じ手順としての所在は未確認 |
| B3 parent→proposal→量子前zero | WANのordinary order/IID proposal、一般的なrejection/weight原理 | M6/M7、K `digital_parent`、G `sample` | 非reduced rawを別parentへreduceして再配分しないlawと、A_u・child lawを組み合わせた全出力の対応資料が不足 |
| B4 ideal→実dyadic law | 実proposalに対する係数/πというISの補正は直接対応 | M9/M10、Kのroot/rounding/weight、BIT §§3–5 | full support、root誤差、係数近似、range、必要bit長を同じtargetで閉じる先行保証が未確認。先行法に丸めを禁じて差を作らない |
| B5 moment→U→budget | weighted estimator・二次momentと費用を区別する既知会計 | M8/M10、既存公開U §7、G `upper_normalizer/budget` | Uの当該式と有限bit κB_oUを、既知lawへ同情報で接続した対応が不足。未知B_RA/Zを無料使用しない |
| B6 event→controlled provider | 単一involution回転、実adjoint、phase保持という条件付き構成 | M12、K/G event記述、CPの既存IR、非Pauli Packet N03/N04 | 任意のQに安価なnative providerを与える一般実装は未成立。同じquery条件なら比較可能だが、物理費用まで同一とはしない |

ここでの「不足」は**G6の既存M1〜M12が未証明という意味ではない**。不足しているのは、別の既知アルゴリズムの手順・保証として同じ接続を追える比較証拠である。また、すべての不足が新定理を要するとも判定しない。既存文献の明示箇所で埋まるもの、直接的な式対応で済むもの、非自明な還元命題が必要になるものを分ける。

## 3. 比較経路別の到達点

| 経路 | 確認済み | 全生成再現についての判定 |
|---|---|---|
| WAN ordinary＋AK | ordinary非列挙proposalとGreen係数に直接対応あり | **部分対応**。return収集後のpositive parent law・事前zero・dyadic保証までを既知手順の特殊化として追う証拠は未確認 |
| Zhao–Yuan modified Taylor | identity/低次への高次寄与吸収が明示 | **部分対応＋target差**。§4.2のmodified remainderを同じ有限P_mとみなさない。一般formal return全体と同じという対応は未確立 |
| PTSC zeroth-order | Eulerのidentity/先頭項pairingと次数・label sampling | **部分対応**。同じ有限cutoffへ揃える余地はあるが、全return/suffix lawは別の照合を要する |
| CTS＋共通Green | Pauli係数収集とlayer samplingが明示 | **部分対応**。既知法にもGreen queryを許すが、それだけで一般odd formal wordをPauli回転の軸にはできない。G6はsingle Qとsuffixをpairする |
| SPS＋controlled process dictionary | libraryとsigned weightの一般的枠組み | **target/access条件付き**。controlled-systemのobservableとして検討でき、対照から除外しない。ただしlibraryの取得・局所生成・位相を保つ分解を別に確認する必要がある |
| IS＋取得済みevent係数 | `W=c̃/π`という補正の直接対応 | **直接対応する部品**。ISは既存event集合のproposalを変えるが、それだけでreturn係数とparent角度を生成しない。cost-optimal proposalの取得も無料でない |
| 上記すべてを許す強い対照 | 構成の各部品を既知原理で説明可能 | **全還元はNOT_ESTABLISHED**。同時に、独立性もNOT_ESTABLISHED。追加の還元をここで考案・証明しない |

## 4. 停止して返す比較命題

下表は新しい研究計画ではなく、何を示せば全還元・非還元の判断が変わるかという不足の特定である。

| 問い | 現在確認できる範囲 | 必要な追加根拠の型 |
|---|---|---|
| 任意の許可p/x/mで既知の手順を特殊化すると同じeventのsupport・phase・実π・c̃・zero結果を得るか | K/Gの出力仕様はある。先行手順からの全写像は未確認 | 明示した先行algorithmと直接置換の照合。非自明なjoint-law等式が必要なら**REQUIRES_NEW_SCIENTIFIC_PROOF**として別判断へ返す |
| 同じoperator meanなら十分か、同じlawまで貢献の判定に必要か | mean同値とsampling同値は別である | 論文の中心claimの科学的な範囲判断。Codexが基準を採択しない |
| 同一accessで既知合成が同じbit/bias/range/予算を保つか | G6既存条件と先行法側の条件を列挙済み | 既存保証の明示的特殊化で済むか確認。新しい一様lemmaや計算量分離なら別の理論承認が必要 |
| すべての既知合成では再現不可能、またはG6がより速いか | この監査からは言えない | 一般的非同値性・下界・優位性定理は新しい科学的証明。今回実施しない |

**結論：既知部品からG6へ還元できる可能性はあるが、指定資料だけでは完全な還元を確認できない。** その不確定性を独立新規性へ読み替えず、構成保証の原著貢献としての価値はGPT・ユーザーへ返す。新しい同値性定理、sampler、性能実験は作成しない。
