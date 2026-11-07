# N4-A：RZ role・literal key・coverageの対応

2026-11-08 JST。[REPORT](REPORT.md)。基準002ba1141fcdbfb9bd59b8173fb594da9a89388b、固定H4/DF12・Partial S2/LD3/h=.02/K2・q128/r1/m7だけを扱う。

## 1. 証拠と分類の意味

[CT-0 KEY_COVERAGE.json.xz](../../../artifacts/pr_reference_ct0/2026-11-07/KEY_COVERAGE.json.xz)のm7/B′・Reference role_rowsとactual inverse D supplement、[PHASE_SUPPORT](../../../artifacts/pr_reference_ct0/2026-11-07/PHASE_SUPPORT.json)のr1 event superset／m7 vacuum境界を読み、固定C0と照合した。

C0 hitは同じpositive binary64 hexが299採用tableに存在すること。missで保存L以内ならDICTIONARY_DOMAIN_COVEREDという**role再利用候補**とする。新roleのprovide/emit・certificate生成ではない。全保存numeric supersetに域外keyは見つからなかった。これをCを外れた任意compiler出力の域保証とはしない。

新しい到達集合、trajectory、角度、価格、確率を計算していない。absで正負を同じpositive tableへ対応させる以外、π snap／近接float統合／2pi同一視を行わない。隣接hexは異なるkeyのまま残す。

| 分類軸 | 今回の意味 |
|---|---|
| C0_FIXED_WORD | 既存同literalのwordと既存資格を参照できる |
| DICTIONARY_DOMAIN_COVERED | missだが固定provider域内。roleへの実接続は別資格 |
| OUTSIDE_DICTIONARY_DOMAIN | 保存supersetでは該当なし |
| CONDITIONAL_ON_COMPILER_CONTEXT | SOURCEからnative keyへの全support移送はCT-5B条件Cが前提 |
| EMITTER_CONTEXT_UNQUALIFIED | whole role consumerのreset/readout/phase等が未接続。旧emitterの不具合認定ではない |
| SOURCE_SUPPORT_UNKNOWN | b2のcanonical到達性など、必要なfinite-bit cellの証拠を今回確定していない |
| UNQUALIFIED_FULL_SUPPORT_CT_RPE | 局所word／候補coverageだけからRPE成功を認定しない |

機械可読表は[ROLE_COVERAGE.json](../../../artifacts/pr_reference_n4a_coverage/2026-11-08/ROLE_COVERAGE.json)。keyは保存台帳をunionしたmetadataであり、件数は物理gate多重度ではない。

## 2. D deterministicとactual SOURCE inverse

[pilot.symmetric_fused](../../../src/trottertracks/pr_reference_interference_pilot.py)、[transfer.assemble](../../../src/trottertracks/pr_reference_transfer_qualification.py)、[V3 dparts／diagonal](../../../src/trottertracks/pr_reference_rotation_v3.py)を照合した。Dの構成・時間引数・境界多重度はevent非依存。ReferenceのF側はSOURCE inverseを使う。

保存m7 D_unfusedのpositive keyはB′159、Reference155で全てC0内。signed境界補正はそれぞれ17／27 positive key、保存actual SOURCE inverse D supplementは160 positive keyで、これらもC0内。集合は重なり、signed補正は物理出現ではないので加算しない。

Dの全support native価格固定はC下の局所template移送に依る。SOURCE固定だけを根拠にnative価格を保証しない。SOURCE inverseのloweringは正方向native列の単純inverseと同一視せず、保存V5 inverse supplementを別に保持する。

## 3. 共通RTE basis frame

[DF RTE builder](../../../src/trotterlib/df_rte_qiskit.py)はcomponentのbasis_id／basis_hashから固定runtime_operationsを選び、SOURCE inverseでは順序と各gateを反転する。連続runのbasis共有はcontrolに依存しない。occurrence間splitとactual SOURCE inverseを含むpair multiset共通性は[CT-5B §4](../pr_reference_ct5b_20261107/STRUCTURAL_CERTIFICATES.md)を再利用する。

m7保存台帳は両armで同じ**216 positive key**。**132がC0、84が辞書域内miss**。84の全リストはJSONのroles／non_C0_domain_candidate_keysへ保存した。これらのcanonical確率・到達性を新しく計算していない。

指定key `0x1.08ba32d420f2cp+2`はC0なし・域内。保存basis_inverse:df-basis-cb0c4fc6dd74e883da97770cd31022dd8e2e647befe0f44f88d4018e214391e2にも存在する。両armの代表保存recordは[INTEGRATION_TARGETS](../../../artifacts/pr_reference_n4a_coverage/2026-11-08/INTEGRATION_TARGETS.json)。

同一policyとCの下でframe価格差は取消せるが、frame実価格を旧C0の値や0として埋めない。近似frameと逆側の誤差が取消すとは主張しない。

## 4. RTE中央

SOURCEのunsigned rotationはorder0/2に対するatan式、component signを含むfull-angle thetaを使う。B′では±theta/2、Referenceでは±theta。今回は再計算せず、[保存4 key](../pr_reference_ct5b_20261107/FINITE_KEY_AND_UNKNOWN_PRICE.md)を引用する。

| 用途 | 保存positive literal |
|---|---|
| Ref order0 full | 0x1.7e3c19bcd3089p-6 |
| B′ order0 half | 0x1.7e3c19bcd3089p-7 |
| Ref order2 full | 0x1.fdaaba4d44069p-8 |
| B′ order2 half | 0x1.fdaaba4d44069p-9 |

全てC0_FIXED_WORD。SOURCE half/full・符号・F†の向きを維持する。近似したfull wordを分割してhalf wordへ流用せず、同じliteral tableを使う。

## 5. ProductとB′ event

productはZ/ZZ supportからZまたはCZを置く。保存templateのπ／π/2 keyはC0内。SOURCE product-Z cancellationは実gate出現を変え得るが、新しいRZ角を生成する規則ではない。保存T=0を誤差0やgate不存在へ読み替えない。

event符号は、先にevent.phaseを集め、その後productsの負係数ごとにfloat(pi)を**逐次**加算する。identity抽出済みの固定tailではr1/K2の非zero supersetは保存b1/b2/b3。

| phase key | C0／辞書候補 |
|---|---|
| 0x1.921fb54442d18p+1 | C0 |
| 0x1.921fb54442d18p+2 | 辞書域内miss：b2 |
| 0x1.2d97c7f3321d2p+3 | C0 |

SOURCEは位相0ならPを省略する。b2はorder2かつ負productが1個の場合の分岐だが、canonicalでそのlabel組に非zero 73-bit cellがあるかは今回未確定。安全なsuperset内にある以上、到達性を先に解くためのsamplingや追加合成は不要である。

Referenceではこのevent符号を独立P roleとして置かず、lane global phaseとvacuum補正へ反映する。B′ eventの価格非負性と、B′でそのwordを実行できることは別。b2を無料のidentityへ置換していない。

## 6. Vacuum・scalar・wrapper・CSWAP

vacuumはF/Gの実命令順RZ/RZZ減算、Qiskit remainder、最後のbinary加算を保つSOURCE recurrenceで生成される。m7保存上界82.49247654978845は既存providerのLと同じhex。到達key全体は未列挙だが、区間包含とN3-B1b-1の既存provider資格で扱える。新しいphase算法や期待分布を作らない。

B′ scalarは保存m7 literal key1種でC0内。Referenceの物理scalarはlane/vacuum補正へ入る。wrapperのH/Sdg、Ref CSWAPの全保存RZ keyもC0内。B′にCSWAP roleはない。

native SX/X/CXはRZ lookupと別の旧lower_native経路を使う。SX=HSH、X、CXの既存意味論を維持し、別gadgetや新controlを後置しない。

## 7. 残る資格

Cのtemplate閉包を保てば、SOURCE上のfinite numeric rolesとvacuum域に既存C0／辞書を対応させられる見込みがある。一方、role固有のnative consumer、全joint phase／reset／readout、各実位置の誤差・全RPE成功は今回未実行・未資格。

保存local loweringにSOURCE/global phaseが欠ける場合は0補完しない。局所native gate productの置換とSOURCE unitary再構成を分け、元のwhole-native metadataを使う。**追加基本合成が必要という結論ではなく、既存provider接続の追加資格が必要という結論**である。
