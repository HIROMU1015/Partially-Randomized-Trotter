# 候補生成・BのN・D/Cの価格を接続する仕様

2026-11-13 JST。[REPORT](REPORT.md)／[有限科学案](MINIMAL_SCIENCE_PLAN.md)。**設計仕様であり、新しい値付き価格packetや実行runnerではない。**

## 1. 所有者と既存API

| 所有者 | 入出力・再利用するAPI |
|---|---|
| 候補探索担当 | 外部で有限menuとcomplete V2 configurationを生成。既存pr_rpe_rmse_controller.generate又はsource-bound schedule adapterの結果を渡す。Dは候補L_D/ΔPF/κ/Mを選ばない |
| BとN供給担当 | 同configurationの全round N/RMSE資格又は根拠付き許容集合。assess / evidence.planで欠測・UNQUALIFIED/INFEASIBLEを維持。DはNを補完しない |
| C | 必要になった候補のE[S]・featureと数値enclosure/law根拠。既存MeanKernel、structural_feature_meansの条件を満たす部分を使用 |
| D | physical requestに対する価格取得・残差/価格区間と誤差domain。compiled_price_fast、新ZZ、元compose/wrapper/compiler、compiled_residualを再利用 |
| B/探索担当 | 供給された適格な区間とNで有限menuの除外/比較。既存exact_scalar/exact_value、V2 assess/cost/selectを再利用。実G計算は別途承認範囲 |

所有者名は引渡し先を示し、他チャットやworkerへのメッセージ送信・実行指示ではない。

## 2. 探索側 → 価格request

metadataは既存V2 configuration、row.identity、row.law_bindingをそのまま保持する。
canonical IDをDで別規則から再生成したり、同じqRだけに短縮したりしない。

| 項目 | 必須内容 |
|---|---|
| request anchor | source commit/path/row selector、configuration_id、m、X/Y。source_candidate_idがある場合はphysical_candidateとの対応も保存 |
| physics | H/state/snapshot/decomposition/rank・fragment順/one-body方針、L_Dとdeterministic/randomized scope、preparation hashとcomponent/basis辞書SOURCE。未構築ならpreparation hash=nullのNOT_EVALUATED |
| schedule | T、q_RPEとq_PFの区別、実ΔPF、formula_id、順序付きsigned_weights・piece別allocation・event_count、K、κの定義SOURCEと実mapping。丸め直しで候補を変更しない |
| event law | 各pieceのorder/component/sign/rotation law、basis sharing、ideal P/具体的sampler Q、PRNG/CDFとseed ensemble、正負timeの処理、normal/conditional strata |
| price profile | compiler hashに加えlowering_id、control policy、wrapper/measurement/outcome convention、P=0、RZ gate_count/shot、scalar_identity_once_at_total_T、状態準備除外 |
| role・need | exploratory proxy / compiled mean interval / final mean、対象domain、要求absolute幅又は候補差予算、同時confidence family、利用するNのSOURCE/資格 |

元configurationのprofileにはlowering_idが常にあるとは限らない。旧profileを歴史上書換えず、price request/evidenceの明示profile extensionで必ず照合する。compiler hashだけでは新旧loweringを区別できない。将来profile本体へ組み込むなら、B側で新configuration identityとして生成する。

同じphysical priceを複数configurationで共有するには、元record参照と「回路/law/全price profile同一」のbridge証拠が必要。候補ID・N・global RMSE契約だけが変わる場合も、元recordをcopyしてidentityを書換えるだけでは不足。bridgeはN資格を移送せず、cache hitも平均のconfidenceを増やさない。

## 3. request → 既存pipeline

1. Bの_validate/assess/evidence.planで外部scheduleとsource/lawの整合を確認。値不足なら不足のまま返す。Dの処理をN供給fallbackとして呼ばない。
2. C1既存contextに一致するH4 L_D1/S2/δ=.1等は、p1_efficient_zz.load_c1_backend→compiled_price_fast.make_backend_classの既存部品を利用できる。P2の消費済みrunner/予算を再利用して実行しない。
3. 別候補はcontextのpreparation/head/scalar/lawを候補単位で作る薄いadapterが必要。既存prepare、canonical_tau、schedule、sample_requests、compose、native wrapper、transpile_and_measure_costを呼ぶ。今回そのadapterは未実装。system-only cacheを流用しない。
4. S用blocksは元recipeのsigned time/half sweepで生成し、tailと同じeventsを使う。順序・binary64加算・operations表現を維持し、scalar/suffix後のfinishを一回にする。fast RZ projectionは同じH4 wire/wordに対し再利用可能。
5. X/Y両軸を個別にbuild/compileし、fingerprint/6指標・event digest/seedを対応付ける。現新loweringはCONTROLLED_ZZ_CX_CRZ_CX_V1、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。別profileは別record。
6. 構造kernelを使う場合はCが新candidateのstate/境界・closure・precisionを確認する。fixed26値の書換えや旧kernelのinput guard解除で一般化しない。

個々の科学runでは専用SOURCE/input/有限scopeを固定してから実行する。今回この手順を実入力で起動していない。

## 4. D/C → B：二つの出力を区別する

**既存値付きrecord**はquantity=FULL_WRAPPER_MEAN_COMPILED_PRICE、identity/law_binding、axis/measurement/units、classification、expectation、source、cohort/n_paths、mean.rz_count、model/domain、uncertaintyを維持する。S/local-window/path Cは別quantity。combined残差は既存combine/b_price_recordを使っても自動でQUALIFIED_EXPECTATIONにはならない。

**bound-only提案record**は点meanがなくても扱う別のevidence sidecarとする。現在のV2 costに通すためmidpointを捏造しない。必要な項目は次の通り。

| 欄 | 内容 |
|---|---|
| quantity | FULL_WRAPPER_MEAN_COMPILED_RZ_INTERVAL、parent request/identityとtarget law/profile、point meanとは別 |
| endpoints | lower/upperはexact整数又は{numerator,denominator}。unknown upperはnull＋upper_status=UNKNOWN_UNBOUNDED、数学的には+∞。区間を作れないreasonを保持 |
| evidence | DERIVED_PATHWISE_BOUND / SIMULTANEOUS_MEAN_INTERVAL / UNQUALIFIED_HEURISTIC等、定義/導出/検証SOURCE、使用domain、対象law・rare/missing扱い |
| error/coverage | numerical、MC、calibration/model、window、law、rareの包含・unknown、confidence/family/stopping ruleと同時被覆。単にflagをtrueにして証明の代わりにしない |
| dependence | path/mean-estimator共分散と共有校正・X/Y・candidate/round、又は依存性を仮定しない同時box。未知covarianceはnull |
| qualification | bound成立の仮定・条件とUNKNOWN範囲。unqualified feature/modelは除外に使わない。普遍的[0,+∞]も情報不足であることを明示 |

これは新schemaの実装・登録ではなく、Bが必要時だけ追加する最小sidecar仕様。既存enumを再定義しない。現BV2はmeanが必要なのでinterval-only選別を既に実装したと主張しない。

## 5. B → D：精密化request

Bはfeasible incumbent/競合候補、比較のγ又はτ、qualified N又は𝒩、price coverage不足を返す。各cellのpriorityは|N_x−N_b|×幅等の**条件付き式**、新しい実数値の重み付けは別科学scope。N未資格なら正式Gへの影響順位を作らない。

要求は「L_D1全26軸を2%へ」ではなく、「この競合比較を区別するため、このphysical cellの同時幅をtまで、domain/law/bias予算込みで取得」とする。必要なconfidence、有限path/compile数、SOURCEは実行前に固定する。上界未取得・biasが大きい場合はUNKNOWNで返し、無断で増分runしない。

最終比較には共通target/保証条件のwhole-schedule N資格、使用price domainの同時被覆、未探索menuの明示が必要。finite-cohort scenarioは参考値に留め、正式winner/最適性と区別する。
