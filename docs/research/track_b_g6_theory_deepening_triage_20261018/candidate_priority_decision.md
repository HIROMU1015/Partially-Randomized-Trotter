# 未解決課題の比較と優先2候補

2026-10-18 JST。**技術調査上の提案はC1→C2。正式な研究採択ではない。** C1は構成保証の使用条件と取得費用を閉じる問題、C2は追加return集約の意味と限界を変え得る比較問題である。C1が既存lemmaのroutineな仕様化だけで完結するなら、そのことを成果の種類に反映し、新定理として膨らませない。3件目を無理に選ばない。

## 比較の基準

科学的結論が変わる可能性、既知法との差、保証・成立限界、実現可能性、他成果への波及を用いる。点数の加算による擬似的な科学ランキングはしない。成功しやすさやtest追加の容易さだけでは選ばない。

| 候補 | 結論を変える点／新規性への寄与の可能性 | 実現可能性・依存 | 今回の優先判断と理由 |
|---|---|---|---|
| **C1 既存有限bit保証の実効的な全費用契約**（T1–T3/T5） | 「十分精度なら可能」の既存保証を、精度の指定・SOURCE停止条件・whole online bit費用まで追える範囲にする。追加lemmaが残るかも明確化 | 既存下界・K/G・query契約を利用可能。自動selector/native取得の実装は未承認 | **優先1**。全ての比較の前提であり、隠れた取得費用やdomain不足が比較・主張を変え得る。単なるH/K実装単独を優先したのではない |
| **C2 一般次数で追加returnが判断を変える成立域／限界**（T4/T5） | normalizer改善が既知の強い対照後にも、認証予算・期待量子費用・古典費用のどこへ残るか。否定結果でも主張を限定できる | m=3は既知、一般mは未解決。実効費用についてC1に依存。既存P3/P5 closed対照を利用 | **優先2**。小さいbound改善を超えた条件・限界を狙える場合に限る。資源の普遍的勝利は要求しない |
| C3 既知法への全還元を独立課題化（T5） | 還元ができれば独立性を弱め、保証・取得上の不足が明確なら差の候補を限定できる | A03 B2/B4/B5が既に特定済み。新しい還元証明の承認が必要な場合あり | **C1/C2の比較義務へ統合**。84セルの再調査や「未発見」の積み増しを独立投資にしない |
| C4 domain・境界の整合（T1） | x=0別経路、m=1/Canonical domain等の過大主張を除く | 既存文書・SOURCEだけで範囲を確定可能 | **通常の前提整備**。L=1/m=1/x=1の既存保証を再証明して新成果としない |
| C5 H/K selector・angle interfaceだけの実装（T2） | 一般入力への使いやすさ・技術的再現性を増す | C1の仕様が先。既存finite-bit存在保証は既にある | **C1の実装枝として保留**。実装できるだけでは新しい科学的保証とは限らない |
| C6 正値性・Uの鋭さだけを改善（T4） | 比較予算・range・適用域を変えるほどなら意味がある | 現域の正値性と三次のstrict差は既知。一般等号・最適性は未確立 | **条件付き保留**。新しい小定数だけで意思決定が変わらない場合は投資停止。x>1最大域探索も現scopeから自動採択しない |
| C7 DF→G6→PR/RTE adapter（T6） | 物理入力・coherent推定量へ保証を接続し、応用可能性を明確化 | 係数・primitiveは既存。入力誤差、ordered event、全trial、angle/nativeが未接続 | **別の応用研究候補**。一般有理入力の構成保証に必須ではなく、分子性能の問いを先行させない |
| C8 一般controlled provider/native性能（T6/M12） | exact query仮定から特定providerの実費・誤差へ進む | 物理モデル・providerを別途固定する必要。既存IRは条件付き | **別scopeとして保留**。任意Qを安価に実装する一般課題へ広げない。既存STOPを解除しない |

## C1：既存有限bit保証を実効的な全費用契約として閉じる

1. **現在の到達点**：BIT §§2–6とM9〜M12は、supportの一様下界、必要精度の多項式的取得可能性、πW̃=c̃、係数bias/range、angle enclosureの処理可能性を与える。K/Gは固定精度で検査しfail closedする。G7はU.hiを使うbudgetと条件付きprovider IRを持つ。[BIT] [K] [G] [CP]
2. **未解決課題**：合法な入力・誤差条件から、どのH/K・root幅・angle精度を用い、どのAPI検査を満たし、setup／各reject／accepted event／全N trialに何のbit処理と取得費用が発生するかを一つの明示契約にできるか。既存polynomial論証を否定する問いではなく、未閉鎖の具体的な処理・定数・モデルの特定である。
3. **新しく得られる成果**：一般有理入力で保証を使える実効的な構成仕様、必要な停止条件、全費用の適用範囲。追加の非自明lemmaが実際に必要で証明された場合だけ、その保証を新しい理論成果と数える。
4. **反対の結果の価値**：既存lemmaの直接適用で全て閉じれば「新しい精度存在定理」の候補を消せる。未認証angle/provider、bit/selector費用が支配的なら、安価な全生成という主張を限定できる。固定bitの失敗だけを一般構成の不可能性とはしない。
5. **科学的新規性への寄与**：条件付き。H/Kを自動化する実装単独はroutineになり得る。価値が残るのは、既知法と同じaccessでの全費用保証、実際に追加が必要な保証・成立限界を明確にできた場合。独立新規性は本調査で認定しない。
6. **技術的実現可能性**：高い静的出発点がある。p_min、t_(m−1)、root/roundingの既存下界・検査を利用できる。ただしSOURCEを実行して全入力成功を検証したわけではない。新しいselector、角度計算、cache方式はここで設計しない。
7. **依存関係**：正の有理入力、0<x≤1、有限奇数m、uniform bits、exact controlled-Qという既存条件を保持。DF入力誤差・物理provider・長時間PRは含めない。一般αや別budget規約を入れる場合は拡張条件を明示する。
8. **必要な次作業**：まず[W1](minimal_next_work_proposals.md)の既存lemma→API条件→費用項の差分表だけ。各穴を「直接仕様化」「非自明な新証明」「実装」「物理取得」に分類する。具体的な全上界の証明・実装は別段階。
9. **承認境界**：本回は選別とW1の提案まで。後続の静的照合は通常のCodex技術作業として明示指示後に可能。新理論、新SOURCE/科学test、sampling・compile等はそれぞれ別承認。新grantを発行しない。
10. **停止条件**：既存lemmaの転記とroutineな仕様化だけで閉じたら、仕様完成としてC1の新規理論投資を止める。新oracleが必要、global表が必要、元入力の変更が必要ならscopeを広げず返す。保証違反の疑いは未確認と確定を分け、性能作業を停止する。

## C2：一般次数の追加return保証が変える判断と限界

1. **現在の到達点**：B_RA≤U≤B_o、ideal/digital momentとrange、Uによる十分N、量子前zeroは既存。m=3,L≥2にはB_RA<U<B_partial<B_oとpartial/full差の既存論証がある。P5には既存closed非列挙生成もある。[G7C] [U] [P5]
2. **未解決課題**：一般奇数m≥5で、既存partial＋ordinary tail、closed-P3＋tail、利用可能なclosed-P5等に対し、追加returnの正規化差が、有限bit・U認証・同じ残余精度・event価格・全trial古典費用を戻すと何の条件下で残るか。UとB_partialの順序、Nの順序、期待量子費用の順序は別の命題にする。
3. **新しく得られる成果**：追加集約に意味がある明示条件、保証が弱まる条件、または比較不能・成立しない範囲。単なるtoy順位や小さいU改善ではなく、比較判断を変える成立条件／限界が対象。
4. **反対の結果の価値**：追加normalizer差が精度・range・取得費用で消える条件、強い低次数対照で説明できる範囲が分かれば、資源主張を縮小できる。否定的な比較結果も構成保証の価値自体は反証しない。
5. **科学的新規性への寄与**：一般次数・明示accessで、既知のmoment原理から直接出ない成立条件や限界が残る可能性。ただし既知のcost-aware ISや既存比較で説明し尽くせれば独立性を弱める証拠になる。どちらも検討対象。
6. **技術的実現可能性**：中程度・理論的には未判定。既存U/partialの式とSOURCEから出発できる。全parent和、観測variance、真のZ、全word角価格を無料oracleとして使わない。一般比較を新しい数値探索で代用しない。
7. **依存関係**：C1の費用・precision契約に依存する部分がある。normalizerに関する数学命題だけなら先に定式化できるが、その結果を実費優位へ直結しない。非列挙ordinary、CTS layering、合法cache/Green queryを比較側にも認める。
8. **必要な次作業**：まず[W2](minimal_next_work_proposals.md)で、既存式・仮定・未知量を固定した最小の比較命題と、正負どちらの結果なら判断が変わるかを記述する。一般mの証明、漸近展開、反例探索は実施しない。
9. **承認境界**：比較命題案は今回作成できる。命題の採択・新証明の範囲はGPT/ユーザー判断。新計算は別実行承認、保存値の再集計も別承認。旧G7/G9/G10の実行枠を再利用しない。
10. **停止条件**：改善が既知三次の再記述、小定数だけでN/成立域を変えない、弱いbaselineでしか現れない、未知normalizer/価格oracleを要する場合は当該投資を止める。特定命題の否定はその範囲の結論として保存し、別の有利な入力を自動探索しない。

## 非優先候補の10項目と投資停止の条件

| 項目 | C3 既知還元 | C4 境界整合 | C5 単独実装 | C6 上界の鋭さ | C7 DF adapter | C8 physical provider |
|---|---|---|---|---|---|---|
| 現在の到達点 | B1〜B6と84セルが既存 | L=1/m=1/x=1は保証済み | dyadic/weight・atan可能性は既存 | 正値性、U、三次比較は既存 | R成分とcontrolled primitiveは既存 | 条件付きIR、phase/adjoint規約は既存 |
| 未解決点 | 同accessで保証まで移せる具体的先行手順 | identityの呼出側と各armのdomain | 自動H/K・angle実装接続 | 一般域の鋭さが判断を変えるか | 入力誤差・law・推定量・nativeの接続 | 特定物理Qのstrict誤差・費用 |
| 新たに得る成果 | 還元範囲または追加保証の限定 | 過大主張の除去、明確なAPI | 技術的完成度、使用可能範囲 | 有効なら成立域・認証予算の改善 | 実データへ保証を接続 | query条件を特定nativeモデルで具体化 |
| 反対の結果の価値 | routine還元なら独立性を弱める。未発見は独立性証明でない | domain外なら誤った適用を防ぐ。定理反例ではない | 費用・精度不足なら実装範囲を限定 | 小改善・不成立なら上界改良の投資を止める | 接続費用・入力誤差で応用を限定 | 重いproviderなら資源主張を限定 |
| 新規性への寄与 | 非自明な保証差が残る場合に条件付き | 通常は文書/API整備 | 単独ではroutineになり得る | 判断・適用域が変わる場合だけ条件付き | 別の応用成果。演算子クラス自体は既知 | 特定providerの方法差が必要 |
| 実現可能性 | 静的出発点あり、還元証明は未判定 | 既存SOURCEで確認可能 | C1確定後の技術作業として可能性あり | 未判定。既存式を再利用 | 複数interface・誤差認証が必要 | access/model依存で未判定 |
| 依存関係 | C1/C2と同じtarget・access・precision | 主定理を拡張しない | C1の仕様と誤差契約 | C2の比較・意思決定に依存 | 採択済D/R、物理入力、C1 | 物理model・provider・誤差scopeの別固定 |
| 必要な次作業 | 対象限定の資料照合、必要時のみ新還元証明 | 静的照合・記述修正 | 実装・science testは別承認後 | 新証明／反例／計算の必要性を先に判定 | interface仕様、後に実装・科学検証 | 仕様調査、後に取得・合成・認証 |
| 承認境界 | 本回は問題の限定だけ。新証明・科学判断は別 | 本回はrisk記録だけ。SOURCEは変えない | 本回は実装禁止。次の明示指示が必要 | 本回は新保証・探索禁止 | 本回はadapter・分子計算禁止 | 本回は回路・合成禁止 |
| 停止条件 | 84セルの反復になるなら独立課題化しない | 既存の域と責任が明確になれば終了 | routine完成を新理論と数えない | N/成立域/比較を変えないなら停止 | 新D/R選択や未承認入力が要れば返す | 一般Qを無料native化する前提が要れば返す |

## 既知法との差を判断する最小の追加問い

C1では、A03のB2→B4→B5について、**取得済みの局所係数から同じfinite mean・bias/range・認証予算を、同じ有料accessで得る先行手順が明示されるか**に限定する。同じevent lawの再現はさらに強い別の問いで、lawが違うだけを独立性の根拠にはしない。既知法へ丸め・Green・IS・streamingを禁止しない。

C2では、既知のnormalizer／momentの式だけで比較が閉じればその直接帰結と記録する。追加の定理・下界・反例が必要なら未解決として返す。「全法で不可能」「G6は必ず速い」という広い命題を採択しない。資料追加だけで済む穴と新しい科学的証明を要する穴を区別する。

前のGPTレビューでの構成的価値の評価は再審査していない。新たに判断すべきなのは、この2候補のどの限定命題に投資するかである。普遍的量子総資源改善・実用的分子結果は構成研究の必要条件にしない。

<!-- FIXED_REFERENCES -->

[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[CP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G7C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
