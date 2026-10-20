# H8希少RTE：条件付きsampling・標本数・推論の設計

2026-10-21。C-0設計のみ。H8 p8/q2/r4/K4・p7/q16/r4/K4、linear 1 Å/STO-3G/tol-only DF28、T=.8/P=0、二次partial、通常signed/allowance1を維持する。以下は将来の契約案で、実データによる計算・標本数決定・新保証の採択を実施していない。

## 原法則と条件付け

元SOURCE `745a674546f2cb6258526fbfde1d9a00601689f8` の `rte.py:_make_event/iter_sample_rte_events`、`scripts/h6_dba_sampling.py:count_law/draw/steps`、`h8_rare_tail_selection.py:conditional_counts/place_orders/component_events` を取得した。[SOURCE目録](../../artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21/source_provenance.json)。

R=qr個のeventの元次数確率をp0,p2,p4とする。次数countの元multinomial weightは

`W(n2,n4)=R!/(n0!n2!n4!)*p0^n0*p2^n2*p4^n4, n0=R−n2−n4`。

登録層の全count cell上でWを正規化してcountを抽選し、ラベル付きR位置からn2+n4個を一様・重複なしで選び、最初のn2個をorder2、残りをorder4に割り当てる。同じcountの各order wordを等確率にする。全R位置を順序を保ってq個のr-event occurrenceへ分割する。stepやevent位置を任意に固定しない。

|層|count支持|必要な扱い|
|---|---|---|
|S2_SINGLE|n2=1,n4=0|全R位置で一様。主たる追加情報候補|
|S2_MULTI|n2≥2,n4=0|n2=2へ固定しない。全支持を保持|
|S4_PRESENT|n4≥1,n2≥0,n2+n4≤R|n4=1への固定、order2の排除をしない|

各eventでは元component確率を元の検証・正規化順序で使い、rotation indexを先に、続いてk個のproduct indexを独立抽選する。生成は固定`_make_event`へ委譲し、実適用順products→rotation、符号/identity位相、angle、basis/reuse metadata、元のevent coefficient・normalizationを保つ。層条件付きprobabilityは**費用標本の抽出確率として別field**に記録する。量子event係数を層質量で割る・量子shotを変更する操作はない。

元`h6_dba_sampling.steps`は全count・component・request組立の参考になる。旧`h8_rare_tail_selection.validate_scope`はq4/q8・r4に限定されるため直接呼ばない。新SOURCEのadapterでq2/q16・同じr4/K4と新cohortを登録し、旧SOURCEは変更しない。元無条件`native.steps`は新条件付き取得へ流用して再抽選を繰り返す方式にしない。`DFRTEEventSequenceCircuitRequest`→`DFPartialS2StepRequest`→元native loweringへ接続する。

## 数値law・小さい支持・乱数

段階Bのcanonical有限lawと保存binary64 categorical lawの丸め診断を混同しない。Bの層質量は固定保存値として参照する。将来samplerで使うpのdomain（canonical高精度近似か保存categoricalの数値表現か）、正規化、precision、Bとの許容差、count分布の独立照合を実行前に一つに固定する。現在契約の`law_numerical_convention_and_tolerance`は未確定で、科学実行gateを閉じている。差を未評価のままπ_Bと別lawのμを結合して厳密期待値と呼ばない。

既存count APIはlog-spaceで桁落ちを避け、正のcount cellのunderflowを検出すると拒否する。大きいRの極小cellが実数として正でもbinary64で表せない可能性がある。今回はH8のcount lawを計算していないため、実際にどのcellが該当するかは判定していない。失われた支持を0として進めない。必要なら新SOURCEで任意精度weight/cumulative categoricalを導入し、精度増加・独立算術・全支持保持を人工検証してから、変更と数値モデルの採用を別承認へ含める。C-0のtoy Fraction oracleはこの構造の検査であり、本番adapter完成の証拠ではない。

新seed namespaceはcondition・stratum・phase（variance pilot/production）・replica・SOURCE・input binding・承認epochを含む。order/count/position streamとq個のcomponent streamを区別し、親で決定的なregistryを封印する。全master/派生streamを旧P・旧pilot・他condition・他stratum・新phaseとの間で衝突検査する。元PCG64/component選択APIを維持し、worker起動順・PID・wall clockをseedに使わない。実seedはprivateで、公開候補にはregistry digest・非衝突receiptだけを置く。条件付き新cohortは旧Pのevent列を再現する主張をしない。

## paired会計とS0の扱い

同じevent列で両軸のfull measured Hadamard wrapperを作り、`G_i=Ncos*RZ_i,cos+Nsin*RZ_i,sin`。保存analytic shotsは条件別契約に固定する。軸間の共分散はpaired Gに内包し、軸別SEを独立として加算しない。RZ/CX/depth/size等は別欄で保存するが、主費用GはRZ×shotのまま。

原独立P n48は両条件ともS0のみ。既存S0 mean/SEを固定保存値として参照し、新希少標本とrowをpoolしない。層ごとの母平均推定を、同じ無条件lawの期待値の項として結合する設計である。旧H8 p7/q4,q8の条件付きpilotは別条件で、今回p8/q2またはp7/q16の層平均・分散として代入しない。

全ての必要層が評価された場合の推定量案は `Ehat=Σ_s π_s*Gbar_s`。独立strata・各層内iidというモデルの下で `Vhat=Σ_s π_s²*s_G,s²/n_s` とする。S0項も含める。各層n>1が分散推定の最低条件で、n1では分散/SE=null。未知層、未検証のupper bound、singleton分散を0補完しない。S0の標本抽出・cohort選択の条件、乱数独立性と有限計算の限界は明記する。

multi/S4を未取得のまま残すなら `π0*μ0+πsingle*μsingle+πmulti*μmulti+π4*μ4` の未知項を保持する。数学的に適用できるMsが別途得られた場合だけ`πs*Ms`で残余を制限できる。非負性による下側制約と、PR優位に必要な上側制約を区別する。小さい質量や過去最大値だけで未知項を消さない。

## 標本数を決める手順

1. ユーザーが推定目的（条件付き感度/近似統計推論/厳密保証）、許容絶対G誤差h、信頼水準、層別誤差配分、S0追加取得の可否を選ぶ。現時点では全て未確定である。
2. 既存静的資料でmulti/S4の制限が成立するかを先に確認する。成立すれば不要なcompileを削減できるが、現時点のTier Rには全native閉包の不足があり、数値残余上界は未採択である。
3. 同じ固定条件の分散とcompile負荷がない場合、**対象層・pilot本数・call数を別途承認したvariance/load pilot**を実施し、そこで終了する。別q/prefixのpilotから勝手に必要nを算出しない。pilot本数もC-0では未設定。
4. 得たσs・費用csに基づくcost-aware allocation案は`n_s ∝ π_s*σ_s/sqrt(c_s)`。目標varianceから整数nを決める。S0項・残余bound・law数値誤差に配分した分を先に差し引く。残る許容varianceが非正なら希少層追加だけで目的は満たせず、S0の別取得・目標変更・判定不能STOPをユーザーが選ぶ。
5. productionのn・同一条件/層・call予算・受入精度を新SOURCE/新契約で固定し、**別の明示承認・新grant**で実行する。適応pilotはprimary production平均へpoolせず、探索情報として分離する。原PのS0は別anchorとして指定したまま扱う。

この二段階案は、今回または将来pilot承認からproductionを自動許可するものではない。既存の妥当な分散/boundで事前にproduction nを固定できる場合はpilot不要だが、その根拠を提示する。nを結果の見栄えや符号が確定するまで増やす規則は採用しない。

## 受入・STOP・科学的主張

構造受入はcount支持、同event両軸、全予算予約、SOURCE/input/runtime/compiler/preparation一致、整数metrics、全native exit/reapを要求する。未完了taskを黙って除いて平均を出さない。失敗・law underflow/数値不一致・実OOM/破損なら部分結果を保全してSTOP、同grantのretry/resumeはしない。

固定Nでの近似区間（例：モデル仮定下の正規/Student近似または層別bootstrap）は、方式・alpha・多条件/多層比較の扱いを事前指定し、有限標本での厳密coverageを主張しない。bootstrapは未取得の支持を新たに生成しない。近似区間がPF固定費用と分離した場合もモデル条件付きの統計判断とする。重なる場合は未確定と報告して、追加標本へ自動移行しない。

厳密な全support上界・非漸近coverageには、妥当な全support range/分散制御等の別証拠と推論契約が必要である。既存SE・保存最大・新pilot結果・Tier R L1のみから厳密保証を採択しない。性能・数値照合PASSも科学的保証ではない。
