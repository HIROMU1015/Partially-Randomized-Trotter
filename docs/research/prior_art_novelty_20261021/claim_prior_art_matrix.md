# Claim–Prior Art Matrix

16主張の判定。新規性は「世界初」を認定する尺度ではなく、本調査で確認した一次文献との具体的差分。高/中/低は主観的な質的評価で、確率や査読採択率ではない。科学的妥当性は [査読想定](novelty_and_reviewers.md) と別に読む。数値は保存結果の転記に限定。

|ID|主張候補|直接先行研究・箇所|既知／本研究の追加|種類・新規性|支持scope|批判／既存証拠による応答|追加確認|
|---|---|---|---|---|---|---|---|
|C01|PRという新しいsimulation法|P01 §V Eqs.(32)–(39), Fig.7; P03 §§4–5|deterministic prefix + random tailの混合は既存。追加：固定SOURCEで実装・適用・検証した|再現・実装比較／方法新規性：低|v2 protocolとSOURCEの構成に対応|査読：先行法の実装ではないか。応答：原理の新規性では反論不可。実装依存の資源問いに絞れば説明可能|新しい構成・定理があるとの主張には、その具体的差分が必要|
|C02|canonical finite-RTEが新アルゴリズム|P02 Lemma2/Fig.1; P01 App.A2 Eqs.(A20)–(A29)|paired Taylor LCU、normalization、有限次数化が既知。追加：有限K・次数/符号/identity・controlのSOURCE検証|再現・実装比較／低|rte.pyのpaired order weightとtruncation説明を静的確認|査読：Taylor打切りの工学実装に過ぎない。応答：現証拠ではその批判に同意。独自な打切り理論や保証は本報告で認定しない|finite remainderの新しい改善定理があるなら原理と既知系を別途照合|
|C03|DF-native/Givens/basis共有が新しい|P10 Eqs.(3),(8), Fig.1; P01 §VII B Eqs.(42)–(43), Fig.10|low-rank、basis rotation、partial randomizationへの適用が既知。追加：現native full wrapper/control/scalar・境界規則の具体的コスト|実装比較・新定量値／低〜中|H4 B0I実測、NATIVE_MIXED counterの保存結果|査読：標準的なDF回路とcompiler最適化。応答：個別実装の費用・方向差は示せる。一般的新合成法の主張は反論不可|新しい合成法としては専用定理/既存synthesisとの対応が不足|
|C04|compiled RZ×analytic shotsという新会計法|P01 App.E Eqs.(E21)–(E23); P08 §3.3 Eqs.(38)–(39); P26 Eqs.(17)–(18)|単価×反復数とcost-aware partitionは既存。追加：full HadamardをQiskit1.3.0で測り、固定signal精度へ接続|実装比較・新定量値／方法新規性：低|protocol v2、H468確認、theory/compiled保存結果|査読：既存会計の代入に過ぎない。応答：原理は同意。実測が比較結論を変える具体例として反論可能|rotation angle/T合成・backend/runtimeへの外挿は別証拠が必要|
|C05|normalizationとbias headroomによるshot競合の発見|P01 §V/App.B/E; P02 Lemma2; P28 Table1/Eqs.(5)–(8)|LCU重み/誤差余裕によるsample増は既知。追加：各保存候補で何が支配するかの定量対応|既知理論の実装検証／低|Mechanism core32/説明32の保存因子|査読：恒等式の図示で新しいmechanismではない。応答：代数の新規性は主張しない。特定の条件差を説明する証拠としては有用|新法則を主張するには独立予測・非自明な判定則が必要|
|C06|B0+IによってPR全面優位が崩れる|P01 App.D2 Figs.19–20とApp.Eのphase補正; P10 truncation|discard誤差・phase補正の原理は既知。追加：固定signalに対するscalar保持対照を実測し、H4旧全面優位の例外を特定|新しい定量的知見／中〜高（局所）|H4 legacy DF12・6距離・T=.8/P=0・S2・q1/2/4/8。保存397/1812、240軸ΔRZ=0|査読：正しいphaseを戻しただけ、baseline欠陥の修正では。応答：新法ではなく、合理的対照が資源結論を変える検証例として反論可能。先行PR全体の反証ではない|別入力/実装へ一般化する証拠、単なる既知補正を超える問いの整理が必要|
|C07|truncated PF・細prefix・S4による公平比較|P04 Fig.2 caption（truncation/次数）; P01 App.D2; P21 §§IV–V; P27 Fig.1|truncationと高次/構造化PFの重要性は既知。追加：H2 D70/H4 D190/H6 136/H8 184という登録対照との比較|実装比較・定量的知見／中|protocol v2候補表、D/P結果。最良固定PF単価は実測、domain全体は条件付き|査読：PF q上限やformula集合が弱く、PR側の自由度と非対称。応答：登録集合での結果としては反論可能。全PF一般最適との主張は反論不可|H6/H8 PF q≤8とPR q16個別点、ordering/representation/SPRINT未比較を明記|
|C08|探索PR優位の独立確認・H4非再現|P08 bias/variance/実験設計; P14 concentration; P01/P25は同一の費用標本確認設計とは未認定|独立holdout/selection biasは一般統計として既知。追加：候補・PF・nを固定した別cohortと、PR優位が再現しない事例を保存|新しい定量的知見／中〜高|H4 DF10 ε=.05、L_D3/q1/r4/K2、n1108、PF B0I p5q1。別の細精度n78も保持|査読：非再現はsampling誤差/選択効果の当然の結果では。応答：一般論は既知だが、実費用結論が変わる量的事実は追加。特定原因の断定は不可|selection-adjusted保証や母平均順位は追加証拠が必要。今回は計算しない|
|C09|希少eventとMC費用の問題を明確化|P02 Fig.1（高次Pauliは稀）; P03 §1（variable gate count問題）; P14; P30 v2 §2.3|rare support、有限標本、二種類のsampling誤差は既知。追加：保存S0/S2群のfull PR費用、経験的MC変動、未観測supportを明示|新しい定量診断／理論新規性：低|MC補足bee9ae99、H4 Pと同じcohort。H8 n48高次0の保存報告|査読：bootstrapでは未観測tailを扱えず、mean優位を保証しない。応答：その限界は認める。観測分布の記述として有効、full-support保証は未確立|別担当の希少RTE既存証拠調査が判断材料となり得る。今回介入せず|
|C10|理論gate countとcompiledで優劣方向が変わる|P18 explicit circuit resource comparison; P01 App.E; P25 App.C gridsynth|解析的単価と実装単価の差は既知。追加：S4修正後にも残るnative固定pairの方向差を示す|新しい定量的知見／中|H4 legacy DF12 1Å/T=.8/P=0、PR p3q8r2K4 n32対PF p12q1 S4、ε=.05/倍率1。保存比.8716321313/1.0393357766|査読：理論モデルが異なり、有限標本/実装選択のartifactでは。応答：特定NATIVE_MIXEDモデルの実装適用限界として反論可能。P01のToffoli式の反証とはしない|same-event分解・compiler因果分離・他実装移送の証拠は不足|
|C11|q費用モデルは比較結論に頑健|P01 App.Eの解析単価; P08 Eq.(39)の期待cost|モデル検証・固定価格置換という手続きは既知。追加：100重複scope/40 endpointでmixed→direct単価置換の感度を保存|新しい定量検証／中|80b7a35c。最良候補2・最良費用/全順位各50・PR/PF方向0変更。H6/H8旧n32、P=0/T=.8|査読：独立100実験でもmodel-only誤差でもない。応答：その区分を明記し、限定価格政策の方向頑健性として反論可能|profile/runtime/S4仮説・旧最終gate欠測を除く保証はない|
|C12|PR有利/不利な精度領域の定量化|P01 Fig.21/App.E; P25 §V B/Figs.3–4; P04 comparison plots|ε依存の方式crossoverは既知。追加：固定DF/complex-signal・native回路・登録候補での領域と非単調な例外|新しい定量的知見／中|H4 B0I geometry map、v2独立8条件、別H6 T panelの条件を混合しない|査読：狭い格子のbias floorによるだけでは。応答：登録domainの適用境界として説明可能。普遍cross-over/漸近則の反論は不可|候補格子/ε定義/時間条件が違う先行研究との同一条件比較は未実施|
|C13|Mechanism因子分解が新理論|P01 Eqs.(E21)–(E23); P02/P28 normalization会計|費用=shots×単価、相対比の積は恒等的。追加：保存32比較のheadroom/normalization/有効単価を整合させる説明|既知理論の整合検証／低|1ff6ac0b説明32/64軸。H4 legacy/旧H6 cohort/H8 n48等の出典差を維持|査読：因子の恒等式を科学的mechanismと呼べるか。応答：新法則では反論不可。どの因子が特定条件を説明するかの記述なら有用|予測可能なcriterionや独立データでの説明力は未確立|
|C14|一般分子・energy/RPE・T/FT資源へPR優位を拡張|P01 full-QPE/Toffoli; P12 Tables I–III; P26 §V; P38 architecture resources|それぞれ固有の誤差/状態準備/合成/architecture会計が必要。追加：現在は固定T小水素鎖のno-prep RZ資源|現証拠では支持不可／新規性判定以前|v2の明示scope、H8回路未取得、入力生成手順差|査読：重要な実用problemを評価していない。応答：限定benchmarkとしては説明可能。全体優位への反論は不可|energy精度接続・state準備・角度合成/FT・一般分子は別の証拠が必要|
|C15|4次化が本PR研究の新アルゴリズムになる|P21 higher-order/processed PF; P33 new S4の公開別成果; P27/P32/P36|高次PFとrandomizationの組合せは既知。追加：進行中研究完了後、bias低減とtail/負時間/compile/shots競合を比較できる可能性|現状は未確定／将来の定量知見候補|現行独立表のPRはS2。S4はdeterministic対照。進行中結果未取得・未介入|査読：既存S4の単純差替えではないか。応答：新構成だけでは弱い。公平最適化で非自明な総費用差が残るなら補強可能|完了結果と来歴・原論文対照が必要。今回採択も実行も行わない|
|C16|本研究全体が新しい一般method-selection方法論|P03 partition theory; P08 cost-aware sampling; P21 formula比較; P25 regime研究; P26 UWC|費用を最小化するpartition/representation/精度配分は既知。追加：対照強化・実装単価・固定cohortの検証を一体にしたcase evidence|現状は検証workflow/実証研究／方法新規性：低〜中|v2、B0I、theory/compiled、独立P、q感度を同一scopeへ無理にpoolしない|査読：検証checklistの集積で一般予測能力がない。応答：再現性と具体的反例には価値。普遍選択則としての反論は現証拠で不可|移送可能な説明/criterion・既知最適化との差分が不足。不存在検索だけでは補えない|

## 読み方と保留

P番号の正式著者・版・DOI・本文確認範囲は [参考文献](references.md)、対象・指標は [論文比較表](papers_comparison.md)、固定Git/SHAは [SOURCE一覧](audit/source_inventory.json)。abstract評価しかしていない文献の未確認節に、同等対照が存在しないとは判断しない。

C06/C08/C10が比較的強い定量候補、C07/C11/C12が補助的な頑健性/適用条件候補。C01–C05/C13の一般原理は中心的新規性にしない。C14は適用範囲の未到達、C15は進行中の別研究が完了してからの条件付き論点。いずれも正式な方針採択ではない。

**MANDATORY STOP。** 新しい統計処理・対照探索・科学レビューを開始しない。
