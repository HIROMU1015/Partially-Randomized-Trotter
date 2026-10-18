# T1〜T6：何をさらに明らかにできるか

2026-10-18 JST。M1〜M12は[既存Matrix][C01]を参照し、再証明・再作成しない。`DOCUMENTED`は固定資料に既存保証がある、`IMPLEMENTATION_GAP`はSOURCE接続不足、`CONTRACT_GAP`は保証を使うための仕様不足、`OPEN_THEORETICAL_QUESTION`は追加命題が未確立、`NOT_RETRIEVED`は未取得を表す。未取得を不適用・偽と判定しない。

| ID / 対象 | 既存の到達点 | さらに明らかにする問題 | 不足の型・必要証拠 | 解決で増える成果／関連候補 |
|---|---|---|---|---|
| TG01 / T1 量化・境界 | L=1の形式Green、m=1、x=1、全正pについて既存M1〜M7。x=0は別扱い | 各数学domainと各APIのdomainを混同しない。特にidentityの呼出側処理 | `CONTRACT_GAP`/`IMPLEMENTATION_GAP`。K 138–146,214–224、G 103–111,216–251の静的対応。新しい正値性証明ではない | 適用域の誤記を防ぐ。単独の新研究としては選ばない。C4、C1の前提 |
| TG02 / T1 挿入・support | 挿入coverは上界、0<x≤1で正値。valid parent、i≠first(u)、pure-parentの数理規約がある | x>1・負p等への延長は未保証で、現証明の十分域と真の最大域を区別。現scopeのsupport欠落があるかはAPIに照合 | 現域は`DOCUMENTED`、域外は`OPEN_THEORETICAL_QUESTION`。MATH §§1,3–7、K 162–202。域外不成立の反例は今回未導出 | 既存正値域を新しい未解決問題にしない。最大域探索は現目的上優先しない |
| TG03 / T2 精度存在→有効仕様 | BIT §5にorder、label、accept、child、dの一様下界と多項式bit数の存在論証 | 既存lemmaを使うH/K・root幅・丸め認証条件の実効的な仕様と、全合法入力での停止条件の接続 | `CONTRACT_GAP`/`IMPLEMENTATION_GAP`。既存下界とK 45–110,214–236を照合。存在保証の再証明ではない | 手選び精度のfixtureと、一般入力で使用できる構成契約を分離する。C1 |
| TG04 / T2 全誤差の接続 | πW̃=c̃、M̃へのmean、Mへの係数bias/range、atan処理可能性、条件付きG7 budgetがある | angle enclosure→strict Rz/actual adjoint、必要η/ρ/τと残余s、入力誤差のどこまでを対象にするか | `IMPLEMENTATION_GAP`/`CONTRACT_GAP`。BIT §§3–6、CP 2–58、G 266–284。DF入力の誤差は別C7 | SOURCEが実際に認証する条件を明示。新たな一般誤差lemmaが必要なら別理論作業。C1 |
| TG05 / T3 局所算術→全bit費用 | setup O(Lm²)、局所parent/全child O(m³+Lm²)、係数bit長の既存評価 | selectorのlaw再検査・走査、sort、isqrt、全reject、angle取得、初期化とonlineを含む費用scope | `CONTRACT_GAP`。BIT §§2,5–6、K 55–110、G 237–251。具体的な総上界の新証明は未実施 | 非列挙がどの入力・費用モデルで意味を持つかを限定できる。C1。mについての多項式をlog mについての多項式と呼ばない |
| TG06 / T3 memory・繰返し | kernel O(Lm)係数、現在のlocal packet。G7 full sampleはpacket一回で、過去wordの増大cacheを要しない | root/有理数の一時領域、分布表、N試行のbit/CPU/取得費用、合法cacheの費用と再利用範囲 | `CONTRACT_GAP`。K 138–160,194–236、G 237–251、BIT §2/6。全時間・memoryの数値実測なし | 単一queryの算術countをend-to-end計算量へ格上げする誤りを防ぐ。C1 |
| TG07 / T4 正値性・normalizerの鋭さ | B_RA≤U≤B_o。m=1は等号、L=1はU=B_RA、m≥3,x>0でU<B_o。m=3,L≥2ではB_RA<U<B_partial<B_o | 一般奇数m≥5で、既存の強いpartial/closed対照と認証予算にどの順序・限界があるか | `OPEN_THEORETICAL_QUESTION`。G7C「U」「強い三次対照」、U §7。三次結論の一般化は不可 | 成立条件や明確な負結果で追加集約の役割を狭められる。C2。微小なU改善だけはC6として保留 |
| TG08 / T4 moment→実費 | ideal E[X²]=B_o B_RA、全trial、量子前zero、digital上界κB_oU、hard cap Nが既存 | 認証可能なN、digital acceptance、eventごとのprovider/angle価格、reject古典費用を合わせて何を比較できるか | `CONTRACT_GAP`/`OPEN_THEORETICAL_QUESTION`。MATH §8、G7C finite-bit/provider、G budget。未知B_RA/Zや全価格表を入力にしない | 「momentだけが改善」「予算だけが改善」「費用の成立域」を判別できる。C2、C1依存 |
| TG09 / T5 既知法への還元 | P1〜P12と接続B1〜B6の監査済み。Green/Euler/ordinary/rejection/ISは既知 | B2のpositive suffix law、B4/B5の有限bit・U予算を、同じA0/A1と費用で保証する先行手順の具体的対応 | `NOT_ESTABLISHED`。A03 §§2–4。全法一般の非還元・下界は別の新証明。G6 lawを無料oracleとして渡さない | 還元は独立性を弱め、取得・保証の実質的追加は支持材料になり得る。どちらも価値。C3をC1/C2の比較義務に統合 |
| TG10 / T5 強い低次数対照 | P3の非列挙closed法、G9 P5Closedのgroup/DP生成が既存 | m=3/5でGreen法が必須という説明はできない。一般mの保証と低次特化の取得費用を区別 | `DOCUMENTED`＋一般比較は`NOT_ESTABLISHED`。G7C、P5 11–123、C02。P5のreference列挙は利用しない | 既知・既存特化へ還元される範囲を明確化。C1/C2のbaseline条件 |
| TG11 / T6 DF入力 | 既存RのGaussian共役Z/ZZ、符号・係数・controlled primitiveが確認済み | float/物理係数→正有理p,x、identity shift、入力誤差とfinite Mのbinding | `IMPLEMENTATION_GAP`/`CONTRACT_GAP`。N03 §§1–4、N04 §§1–3 | 実データへの保証の接続。一般有理入力の理論成立とは別。C7は応用候補として分離 |
| TG12 / T6 adapter/native | ordered event、phase、全trial estimator、native誤差に必要な各条件は既知 | 現PR/RTE adapter、word依存angle、実測interfaceとstrict native費用の統合証拠 | `IMPLEMENTATION_GAP`/`NOT_ESTABLISHED`。N03 §4–5、N04 §5、CP | 実用性能研究として意味があるが、T1〜T5の数学保証の必要条件ではない。C7/C8 |

## 新しい成果との境界

TG03/TG04/TG05を「一般有限bit保証が存在しない」と言い換えない。既存の一様下界・有限精度処理可能性・多項式性は到達済みである。追加価値があるのは、その保証が実際の全処理と明示した費用モデルまで追える範囲や、追えない限界を示せた場合。単なるselector実装・test追加だけなら実装完成として扱う。

TG07/TG08は普遍的資源改善を要求しない。一般次数の成立条件・証明できない比較・明確な負結果も研究成果になり得る。指定した比較命題の不成立からG6の構成全体を否定せず、前のレビューで認められた限定的な構成価値とも分離する。

TG09で原著価値を再認定しない。既存84セルを再走査して文献数を増やす代わりに、[優先順位](candidate_priority_decision.md)では不足する比較命題を具体的な後続条件へ落とす。

固定出典：[MATH]、[BIT]、[K]、[G]、[CP]、[G7C]、[U]、[P5]、[C02]、[A03]、[A04]、[N03]、[N04]。M番号は[C01]の既存Matrix、節・行番号は各固定版を指す。

<!-- FIXED_REFERENCES -->

[A03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f17a7548b7eef1976f45f5d09bdc979e2e0a81b/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/known_method_composition_audit.md
[A04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f17a7548b7eef1976f45f5d09bdc979e2e0a81b/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/access_precision_complexity_comparison.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[C01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/docs/research/track_b_g6_construction_paper_evidence_20261018/mathematical_claim_evidence_matrix.md
[C02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/docs/research/track_b_g6_construction_paper_evidence_20261018/implementation_and_validation_evidence.md
[CP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G7C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[N03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/128a05765985c1c6aec3a1d0ce35e19ffbb3b3f6/docs/research/track_b_g6_nonpauli_application_survey_20261018/pr_rte_connection_and_access.md
[N04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/128a05765985c1c6aec3a1d0ce35e19ffbb3b3f6/docs/research/track_b_g6_nonpauli_application_survey_20261018/controlled_native_feasibility.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
