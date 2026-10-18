# W1-1〜6：既存lemmaとAPI・費用の間の差分義務

2026-10-18 JST。M1〜M12やBのI01〜I18を再掲する表ではなく、**保証を呼出し条件として使う際の差分**を記す。行番号は固定SOURCE [K]（G6）、[G]（G7 generator）、[CP]（provider）である。各行の「数理」と「実装」を別分類にし、一方の成立から他方の完成を推測しない。[機械索引](obligation_index.json)も同じIDを使う。

`CLOSED_BY_EXISTING_LEMMA`=既存明示保証、`ROUTINE_SPECIFICATION`=直接照合、`IMPLEMENTATION_MISSING`=接続なし、`NOT_VERIFIED`=追加解析/証拠未取得、`PHYSICAL_PROVIDER_UNCERTIFIED`=物理未認証、`OUT_OF_SCOPE`=応用等の対象外。今回、必須W1条件に対して新しい非自明定理が不可欠と確認した行はない。理由は[gap文書](remaining_proof_vs_implementation_gaps.md)。

## W1-1：入力・domain

| ID | 数学的根拠・必要条件 | SOURCE：入力→出力／実装状態 | 費用・未閉鎖事項／分類 | 次作業と解消時の効果 |
|---|---|---|---|---|
| D01 正有理入力 | [MATH]§1、[BIT]§1：有限正p、正確なΣp=1、零label事前除外。Q_i²=I・Hermiticityはoracle前提 | [K]138–160 `ReturnKernel(p,x,m)`、[G]103–143 `CanonicalGenerator`。Fraction変換後に正値/和を検査。零labelを自動除去/再labelしない。F coercion自体は入力表現を厳密なRat型に限定しない | p全read・有理変換・sum・比較。ラベル対応と明示有理入力の契約は`ROUTINE_SPECIFICATION`、数学域は`CLOSED_BY_EXISTING_LEMMA`。一般Qの性質をSOURCEが検証したとはしない | Codexが入力表現・対応を明文化する通常仕様。物理DF変換は[B-ERROR]参照の`OUT_OF_SCOPE` |
| D02 arm/degree境界 | [MATH]§§1,5–8はL=1/m=1/x=1を含む。新しい境界証明不要 | [K]144は整数の正odd m、0≤x≤1。[G]216はKを使うfullでm=1可。[G]105はordinary/partial_return_tail/closed_P3_tailすべてodd m≥3、109は0<x≤1。`make_generator`254–255はfull以外をcanonicalへ渡す | arm別域は`ROUTINE_SPECIFICATION`。canonicalのm検査はKと同じ型検査ではなく、下位range等で型例外もあり得る。boolがPython intであることも強い型契約の代わりにならない | APIごとの域と例外の記述。全armをm≥1対応と称さず、既存域を変更しない |
| D03 identity/pure parent | x=0 identity、s=0 pure-parentは[MATH]§§1,6で別扱い | K constructorはx=0可、`digital_parent`221–222は拒否。G7 full sample237–251にidentity返却分岐なし、canonicalはx=0拒否。K230–231/245–251にpure-parent規約、full sample246/250はchildを選ぶ経路 | identityの数理は`CLOSED_BY_EXISTING_LEMMA`、呼出側dispatcherの明示接続は`IMPLEMENTATION_MISSING`。pure-parentを含む任意packetがG7 fullで使えるという追加domainは未主張 | 呼出側identity責任の仕様化→許可後の実装。域内valid parentと実際の到達性を無視してpure-parent枝をbugと断定しない |
| D04 mの符号化・低水準呼出し | [BIT]§2/5の多項式性は次数m自体を変数に含む。mのbinary符号長だけに対する保証ではない | K147–160等はm+1配列と次数loop。K `_word`162–168は範囲/reduced、`parent`194–202はeven≤m−1。G `conditional`153–156はexcludedと1−p_excluded>0、`first_distribution`158–160はχ<1が呼出前提。L=1のconstructorが該当groupを省くこととpublic helperの任意呼出しは別 | `ROUTINE_SPECIFICATION`。m、L、入力総bit長、η/ρ/τ、K/Hを別費用変数にする。低水準helperは包括validatorでない | 型・domain・到達経路を明示。入力長だけのpoly-time APIという過大主張を除く |

## W1-2：order・label有限bit生成

| ID | 数学的根拠・必要条件 | SOURCE：入力→出力／実装状態 | 費用・未閉鎖事項／分類 | 次作業と解消時の効果 |
|---|---|---|---|---|
| P01 root/normalized order | [BIT]§5：q_l≥t_(m−1)/3。Kのb_l rootは正確有理radicand、K≥1整数。総root区間から正規化された同じ真のlawのenclosureを得る | [K]45–52 `root_interval(value,K)`はscaled整数division/isqrt、外向き区間。208–212はb_l区間→[lo/total.hi,hi/total.lo]。[G]221はその区間をdyadicへ。root helper自体は新規乱数を取らない | enclosureとsupport可能性は`CLOSED_BY_EXISTING_LEMMA`。K/H指定と入力型の整合は`ROUTINE_SPECIFICATION`、自動指定は`IMPLEMENTATION_MISSING`。isqrtのbit費用は別 | 既存下界と実検査を結ぶ仕様。十分K/Hの新定数やselectorは今回作らない |
| P02 dyadic order/labels | 0<η<1、正support、同一normalized lawを囲む区間。既存lemmaは十分な有限精度の存在 | [K]55–82：lo>0、Σlo≤1≤Σhi、midpoint再正規化、floor+largest remainders sort、全iで q̃_i≥(1−η)hi_i かつ q̃_i≤(1+η)lo_i。[G]222/_law75–76は正確pの一点区間。Σ端点検査だけでは任意入力区間が正しいlawを囲む証明にはならない | 検査の意味は`ROUTINE_SPECIFICATION`、所要精度の存在は`CLOSED_BY_EXISTING_LEMMA`。labelはroot K非依存だがHは必要。fixed defaultは一様成功の証拠ではない | 呼出し元の区間provenanceと失敗原因を仕様化。広い区間と粗いHの両方を区別 |
| P03 bit-map | 各drawはfresh uniform H-bit整数、正しい登録law。同じHでdyadic countsが整数、和1 | [K]97–110：bit範囲/和、counts非負整数検査、累積線形走査。[G]238–239はorder/各labelへ再呼出し、59–72はSHA-counter診断用 | mapは実装済み、randomnessの前提は`ROUTINE_SPECIFICATION`。production fresh-bit接続は`IMPLEMENTATION_MISSING`。検査/走査は毎drawに課金 | deterministic traceを確率保証にしない。新sampler/高速selectorは今回設計しない |

## W1-3：parent・child・acceptance

| ID | 数学的根拠・必要条件 | SOURCE：入力→出力／実装状態 | 費用・未閉鎖事項／分類 | 次作業と解消時の効果 |
|---|---|---|---|---|
| L01 local係数/weight root | [MATH]§§4–7、[BIT]§5：valid even reduced u、a>0、d≥t_(m−1)p_min^(m−1)/2、abs(d̃−d)≤ρd、ρ>0 | [K]194–202 parent→a,masses。214–227 root→midpoint、lo>0と半幅≤ρloを検査。`Parent.s/d2`はpropertyで再計算。G `packet`227–228 | 数理`CLOSED_BY_EXISTING_LEMMA`、実guard対応`ROUTINE_SPECIFICATION`。ρ>0等を全体入口で一括検査する完全な契約はなく、局所guardだけで代替しない | root幅・ρ・supportの出典を保存する仕様。全parent表を要求しない |
| L02 acceptance/child | [BIT]§5：A≥1/(2√2)>1/4、有るchildはq_i≥p_min/2。Aと全trial Zは別。u非空のchildはi≠first(u) | [K]228–231：A² rootを[lo,min(1,hi)]、[K]85–94で下向きdyadicと q̃_A≥(1−η)hi。childは正確有理qの一点区間→同dyadic検査。pure-parentはKで別枝 | 数理`CLOSED_BY_EXISTING_LEMMA`、局所検査は存在。選定K/Hが**全到達parent**で通る接続は`IMPLEMENTATION_MISSING`。child law再構築・sortはonline | 新しい一様存在定理ではなく、既存一様下界→事前固定設定の接続が必要。実装許可前には具体式を作らない |
| L03 zero/failure | [MATH]§8のraw nonreduced/coin rejectだけが正当なzero。精度例外をzeroに置換する根拠はない | [G]240–244のNone対、[K]65–81/88–93/226–229の例外、入力/word/bit不正例外を区別。G full sampleはpacket1回、accepted eventのため2回目のlocal queryなし | 分類は`ROUTINE_SPECIFICATION`。現在の例外は多くが同じValueErrorなので型だけで精度と不正を判別する保証はない。全trial保証の前提を満たす事前policyは未接続 | 固定Nの全lawで精度成功を保証する既存条件を事前に満たす必要。未完了prefixや完了runへの事後conditioningを無条件confidenceとしない。retry/refinementは提案しない |

## W1-4：mean・bias・range・budget

| ID | 数学的根拠・必要条件 | SOURCE：入力→出力／実装状態 | 費用・未閉鎖事項／分類 | 次作業と解消時の効果 |
|---|---|---|---|---|
| E01 actual proposal/mean | [BIT]§§3–4：π>0、W̃=c̃/π、πW̃=c̃。各係数はd midpoint由来。symbolic exact angleのもとmeanはM̃ | [K]239–254はorder×実label×accept×childとtarget。G sample249–251、_event79–90はπ/c̃/Wを保持。π再正規化やweight clipなし | `CLOSED_BY_EXISTING_LEMMA`＋`ROUTINE_SPECIFICATION`。正確Fractionの連鎖・出力処理費用が残る。係数biasは≤ρB_RA≤ρB_o、消えない | MとM̃を別targetとして保持。物理誤差・Taylor remainderをこの保証へ追加しない |
| E02 U/range/moment | [U]§7、[BIT]§4：B_RA≤U≤B_o。digital κ=(1+ρ)²/(1−η)^(m+2)、range≤(1+ρ)B_o/(1−η)^(m+2)、moment≤κB_oU | [G]19–41はraw-already-reduced DPとU区間、223–225はB_o/U、266–271はhiを使用。U.loはB_RA下界でなく、U.hi≤B_o.hiの任意endpoint順序保証も使わない | `CLOSED_BY_EXISTING_LEMMA`。root/DP費用はsetupに課金。未知B_RA/Z、観測variance、真のsignalを入力しない | 上界endpointと対象identityを保持すれば既存budget式の接続条件を閉じられる。新しいNや比較結果は作らない |
| E03 confidence/残余幅 | [G7C]有限bit節の既存Bernstein規約。0<η<1、ρ>0、primitive_error≥0、正の残余幅、whole-law precision、fresh IID/測定モデルが前提 | [G]258–284：ln640上界、共通factor2 bias、remaining>0、ceil N、α_axis=1/320、16 axes family≤1/20。一般α引数なし。`budget`はgeneratorが満たした全前提を再検証するvalidatorではない | 既存規約`CLOSED_BY_EXISTING_LEMMA`、前提のAPI仕様は`ROUTINE_SPECIFICATION`。任意α拡張は`IMPLEMENTATION_MISSING`（今回不要）。budgetの戻り値だけでSOURCE全体の無例外/IIDを認証しない | 既存confidenceを保持する最小仕様。一般αを勝手に採択せず、例外prefixを十分N完了としない |

## W1-5：angle・controlled provider

| ID | 数学的根拠・必要条件 | SOURCE：入力→出力／実装状態 | 費用・未閉鎖事項／分類 | 次作業と解消時の効果 |
|---|---|---|---|---|
| A01 tangent→angle | [BIT]§6：s/a≤2x/(l+1)≤2、half-angle atanと交代級数による有限精度処理可能性、angle誤差τ→strict回転誤差≤τ | [K]129–130/235、[G]251は正確有理ratioだけ。[CP]13–15もratioとsignを記述するIR。任意入力のangle enclosure APIなし | 存在`CLOSED_BY_EXISTING_LEMMA`、SOURCE接続`IMPLEMENTATION_MISSING`。angle取得・bit/一時領域費用の具体化は未了 | 別許可後の通常enclosure実装。symbolic ratioを角度計算済み/native済みとしない |
| A02 phase/sign/adjoint | [MATH]§§1,6、[U]§9、[G7C]provider節。even parent scalar、σ±、word逆時間順、actual adjoint、helper cleanup、exact CQ | [CP]2–20はZ phaseとreversed word、signed Rz、実adjoint。23–34はouter+helper=2、回転部分CQ2/H4/Rz2/CX2。37–58は外部取得sequenceのactual inverseとscalarを保持。σはprovider引数でevent fieldではない | query意味論`CLOSED_BY_EXISTING_LEMMA`＋`ROUTINE_SPECIFICATION`。wordのCQは別途追加。cache取得・strict row証拠はこのSOURCEが作らない | 既存条件を保持。DF/nativeの実費・誤差認証は[B-COST]参照`PHYSICAL_PROVIDER_UNCERTIFIED`、実取得は`OUT_OF_SCOPE` |

## W1-6：全処理費用

| ID | 数学的根拠・必要条件 | SOURCE：入力→出力／実装状態 | 費用・未閉鎖事項／分類 | 次作業と解消時の効果 |
|---|---|---|---|---|
| C01 setup/local | [BIT]§2：setup O(Lm²)有理演算/O(Lm)係数、word O(length(u)m²)、parent/全child O(m³+Lm²)、係数bit長 O(m(H_p+H_x+log m)) | [K]147–160、21–28、170–202。[G]19–41/221–225のlaw/root/U setupも別途ある。Cの再利用とlocal系列再利用はあるが全語表なし | 限定countは`CLOSED_BY_EXISTING_LEMMA`。全setup/local周辺への費用座標の対応は`ROUTINE_SPECIFICATION`。配列係数数はpeak bytesではない | [費用表](classical_cost_obligation_map.md)へ分離。係数bit評価をisqrt/weight/angle/全処理の一括保証にしない |
| C02 onlineの隠れた処理 | [K] / [G]静的な呼出し列。law再検査/線形走査、sort、scaled整数isqrt、Fraction除算/正規化、reduced判定、packet、再sum・weight処理 | K45–110/121–133/177–192/239–254、G237–251。canonical比較側も153–212でconditional lawの再構築、eventでの再計算あり。既知対照に同じ最適化機会を認めるが新cache設計なし | 操作の所在は`ROUTINE_SPECIFICATION`。具体的全bit上界/peak memory/CPU実測は`NOT_VERIFIED`。新非自明定理が必要と確認したわけではない | 別許可後、既存bit lemmaと標準算術費用を適用する追加解析。今回新O記法・定理を作らない |
| C03 全trial/provider | [MATH]§8/[G7C]：全N、ideal NZは期待、hard cap N。digital acceptanceをideal Zへ置換しない | G237–251は有限の1trial、Noneも古典処理を消費、acceptedだけCPに渡す条件。Gのgeneratorに全N測定loop/accumulatorはない（実装監査は[B-COST]に委譲） | 費用記法`ROUTINE_SPECIFICATION`、全N具体費用`NOT_VERIFIED`、物理費用`PHYSICAL_PROVIDER_UNCERTIFIED`。rejectionの期待回数だけで全費用を保証しない | setup+全trial古典費用とaccepted event依存量子費用を別座標で記録する。全量子/native/分子結果は未認証のまま |

ここまででW1の**充足条件の差分仕様**は作成した。未実装policyや具体的な全bit費用上界まで完成した意味ではない。新しい非自明な命題を証明したり、W2のFull/Partial比較を始めたりしない。**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[B-COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/measurement_and_cost_contract.md
[B-ERROR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/precision_and_error_contract.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[CP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G7C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
