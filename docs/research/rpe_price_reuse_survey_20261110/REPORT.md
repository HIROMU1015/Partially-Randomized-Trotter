# RPE総費用評価：アルゴリズム改善系列の価格成果再利用調査

2026-11-10 JST。**静的技術調査・次回実行案。新しい科学計算・数値再解析は未実施、公開照合後mandatory STOP。** [固定SOURCE・成果物索引](SOURCE_INDEX.md)。以下の数値は保存欄／原報告からの引用で、新しい平均・残差・shot・総費用・順位ではない。

## 1. 結論と既存成果との差分

- **作り直さず接続できる。** DのB4-2保存N、C6a〜C7-4の価格・共分散、B/Cの13-round会計が揃い、同日の`SavedCDF8Study`は保存N・単価・総値を読取専用で返す。V3/R1にも固定NのPF/PR価格評価・感度結果がある。同じfit・残差検証・会計を再実行しない。
- **q64は完了済み。** 旧FAIL→同4回路の数値修正→残7 pathの限定完了を区別する。11月9日の最終記録ではraw8・post32・paired/χ8が完了。旧「残7未開始」を現在のcoverageとしない。一方、旧B/C総費用にはこの直接価格がまだ入っていない。[D64]
- **主な不足は長qの絶対期待価格と外挿bias。** D旧会計の約99.4〜99.5%はq≥64のproxy。legacy DF12では末尾3roundの保存寄与率がPartial .9139324437497927、Full .8302589410503727。小qの追加compileだけで判別が完成するとはいえない。[DTOTAL][V3]
- **DのStrong/Referenceは制御回路の比較で、PF対PRではない。** 同じPR law・同じ理想測定・条件付き共通Nでの制御差を分離する技術として再利用する。CDF8の係数・改善率・NをDF10/19/12へ移植しない。
- **当面Nを固定する。** 保存済み直接C、保存モデルĈ、保存残差、未取得価格域を別列に置く。価格側の判断がつく前にshot再配分やPR再最適化を同時に行わない。後述の新しい数値評価は提案だけである。

## 2. 科学条件の対応：qだけではjoinしない

共通してlinear H-chain／隣接1Å／STO-3G、状態準備費除外。ただしH・DF・stateは系列別の固定入力である。各価格にはHamiltonian/DF/state・分割・law・時間・builder/cut・compiler・軸・scope・cohort・SOURCEを対応付ける。

| 系列 | q・時間、PF/RTE | Hamiltonian・DF・分割・state | 回路・compilerと移送制約 |
|---|---|---|---|
| D C6a〜C8/B4/B-C | qはh-step反復数、T=.1q、ordered partial S2/s1/r32/K0、h=δ=.1 | 元H、number-CDF rank8、元frame順LD2、Nα=Nβ=2。元H厳密ground仮定。packet/CDF/lawは索引参照 | Strong symmetric-directional対whole-path一gadget Reference。full Hadamard X/Y。P0=Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=False/topology-free/宣言phase保持。C1追加融合OFF |
| D r16後続 | T=.1q、同S2/s1だがr16/K0 | 同CDF8/LD2、**r変更でlaw・attenuation・Nが変わる** | r16固有価格q2/4/8と保存Nあり。r32のA/F/S・208−224qを転用不可。[D16] |
| 有限時間H4 DF10 | **T=.8固定、δ=.8/q**。PF S2/S4、PR S2、q/r/Kは登録候補別 | tol-only1e−8/rank10、generation順prefix。保存H/state hash、spin sector36 | symmetric-directional、cos/sin whole wrapper、scalar+tail identity一回、reference不回転。Qiskit1.3/opt1/seed17。信号推定精度εの費用で、RPE energy精度ではない。[PROTO] |
| 有限時間H6 DF19 | 同T=.8固定。PF S2/S4、PR登録A16は疎なq/r/K集合 | tol-only1e−8/rank19、prefix、保存state/sector400。H4とsolver等の工程も完全同一とは未認定 | 同opt1の規約でもprofile/input/providerまで同一とは限らない。全PF格子にはモデル価格が残る。独立Pの固定PF実測を全格子の実測としない。[PROTO] |
| legacy H4 DF12 V1/V2/V3/R1 | qはδ=.02の反復数、T=.02q、18round q1..131072。Partial S2/r1..32/K2、Full S2/r0/K0 | 固定DF12、LD3対LD12、8qubit／4電子sector70。保存DF-H E0・physical ground。PF有効固有状態／単位半径仮定とのgapを保持 | boundary-optimized controlled partial-S2、identity位相抽出、全Hadamard・measurement込み、**opt2**。opt1のDやDF10とは別。V1半径不一致・V2条件付き成功は価格資格とは別。[V1][V2][V3] |
| Track C | T_m=.02·2^m、外側step数はT_m/h。Dのordered/tuned構成・Full Y4等、PR tail P3、h/resolution/αをround別選択 | DF12/LD3を基点とする元保存Hermitian-H ground、N=4 sector70、finite-bit weighted P3 | compiler Aのmodular価格。whole-path直接価格ではない。CDF8 K0や旧unweighted K2とは別。[COPT] |
| 新A | T_m=.02·2^m、固定h=.02のq=2^m、Partial S2/LD3/K2。独自Reference/B′・readout等 | 固定H4/DF12と当該state/lawの仮定を保持。Q後続のcanonical保証とDのB4条件を分ける | compiler Aのnative DAG、回転分類、有限strict-Tは別価格。DのP0後価格やCのmodular depthと同一視しない。[NEW_A][JOINT] |

DのStrongはGaussian共有、directional control、head relative/tail共有、q−1 step境界融合、scalar集約を含む。Referenceはactual A†、χ、二CSWAP cascade、追加8qubit、q−2内部融合とcut一境界損失を含む。既存最適化を再度差し引かない。q1はevent16後cutで、q≥2の中央complete-step cutとは別である。

## 3. q別coverageと保存N

### 3.1 D r32：直接価格と旧13-roundモデルを分離

NはB4-2の保存十分shot数で、表は一軸分。NX=NY、保存総shot14,738は必要最小値ではない。13round／εE=.00159 Ha／failure=.01／βtotal=π/6／βstat=π/12／αaxis=.01/26、prior[-10,10]、ideal IID/exact-native等の条件付き。B6b理想同値性だけでactual成功を認定しない。[DB4][DTOTAL]

| q | 保存N/軸 | 取得済み直接whole-path価格 | Ĉ・旧B/Cで実際に用いた価格 |
|---:|---:|---|---|
| 1 | 512 | Stage A n32、両方式X/Y | 保存直接平均。q≥2 affineで置換しない |
| 2 | 512 | C6a n16、各4 wrapper | 保存直接平均／Fの校正点 |
| 4 | 512 | C6a n16 | 保存直接平均／Fの校正点 |
| 8 | 512 | C6b n32、凍結F検証あり | 保存直接平均。後にA/Sのfitにも使用 |
| 16 | 513 | C7-1 n16、凍結F検証あり | 保存直接平均。後にA/Sのfitにも使用 |
| 32 | 514 | C7-3 n8、凍結F検証あり | 保存直接平均。後にA/Sのfitにも使用 |
| 64 | 516 | **C8-1 n8、全32post・paired/χ完了** | A/F/S予測と既存独立domain診断あり。**旧総費用は依然proxy** |
| 128 | 521 | このD条件の直接価格未取得 | 保存A/F/S外挿 |
| 256 | 530 | 同上 | 保存A/F/S外挿 |
| 512 | 549 | 同上 | 保存A/F/S外挿 |
| 1024 | 589 | 同上 | 保存A/F/S外挿 |
| 2048 | 679 | 同上 | 保存A/F/S外挿 |
| 4096 | 910 | 同上 | 保存A/F/S外挿 |

nは価格標本whole-path数でありNではない。X/Y・両方式は同じpathを共有し、独立nを倍増しない。C6a〜C7-3は計88 path、q64はそのfitに未使用の別8本。q64を加えた再fitは未実施。D r16はq2/4/8各n4だけが別coverageであり、q16以降の価格は未取得／長qモデル未fit。[D74][D64][D16]

保存直接価格の例（q64、一軸）：Strong→ReferenceはRZ 336797.375→333473.75、CX 128880→114752、total depth 190931.875→86706.25、幅9→17。全law母平均の保証ではない。[D64]

### 3.2 legacy DF12と有限時間DF10/19

| 回路族 | 同条件の保存直接価格 | 最終RPE scheduleとの関係／モデル価格域 |
|---|---|---|
| DF12 Partial r1/K2 | q1/2較正・q8 holdout、各n8 | 最終m0/1/3に直接価格あり。q4未測定、最終q16..128もr1の直接域外 |
| DF12 Partial r2/4/K2 | 各rのq1/2/8、各n8 | 最終r2=q256/512、r4=q1024は外挿 |
| DF12 Partial r8/16/K2 | 各rのq1/2/8、各n32 | 最終r8=q2048/4096、r16=q8192/16384は外挿 |
| DF12 Partial r32/K2 | 最新q1/2/8各n32、別legacy q16/32各n8 | A0のq16/32はr32。最終q16/32の**r1**へ移せない。最終r32=q32768..131072は外挿 |
| DF12 Full r0/K0 | q1/2較正・q16/32 holdout、各n1 | q4/8直接未測定。q≥64未検証。δ=.01のq8をδ=.02へ移さない |
| DF10 H4有限時間 | D190：PF登録q1/2/4/8/16・S2/S4。A50：PR p1/2/3 r4 K2/4とcentral p2 r2/8 K2/4・同q集合 | T=.8の候補格子。FP n32・新A n8・独立Pを混合しない。独立PはPR q1/r4/K2/p3(n1108)とq8/r2/K4/p2(n78)、各固定PFも直接実測 |
| DF19 H6有限時間 | 強化D136の登録q1/2/4/8にはmodel残存。PR A16は登録p3/4/5・q/r/Kの疎集合（q16個別点を含む） | 独立PはPR q2/r4/K2/p4(n48)、q8/r4/K2/p4(n104)、PFはq1/S2/p7、q4/S4/p11の直接値。未登録qや別prefixを検証済みとしない |

DF12の最終18round単価は、holdout点でも一律に `Ĉ(q)=(2−q)C̄1+(q−1)C̄2` を使用した保存モデルである。「直接値がある」と「その直接値を総費用へ採用した」は別。V3の72軸PRICE_EVIDENCE_MAPを再利用し、新表を再構築しない。[V3]

DF10/19の有限時間比較は保存signal bias・analytic N・compiled RZの既存接続として再利用できる。しかしT=.8でδを変えるq掃引は、RPEのδ固定でTが増える掃引ではない。有限時間のNや単価係数を18roundへコピーすることはできない。CXの同条件全格子をこのRZ主表から推定しない。[PROTO]

## 4. 保存モデル残差・共分散・数値修正

- **F**＝C6a q2/q4の凍結affine、**A**＝C7-4全5q affine、**S**＝design[1,q,sqrt(q)]。A/Sとbasis/lane-runの設計にq2..32の88本を使用。leave-one-q-outはモデル選択にも使った探索CVで、q32を独立holdoutへ戻さない。basis/lane版の未来joint feature law・E[max]は未取得で、保証付き長q予測はない。[D74]
- C6b/q8・C7-1/q16・C7-3/q32は凍結Fに対する一次12量が探索的幅内。ただし別population-line診断はq8/q16で4量、q32で6量が幅外。確認的モデルPASSとして再引用しない。[D6B][D71][D73]
- C7-4のpaired RZ countはall-q affineで小残差、paired total-depthのLOQ q平均RMSEはaffine191.27214→basis-run47.975012。方式別絶対価格とpaired差は別応答であり、異なるbest designを共分散なしに混ぜない。[D74]

q64に対する**既存**paired total-depth残差（観測−予測）：

| 固定model | 予測paired差 | 観測paired差 | 保存残差 |
|---|---:|---:|---:|
| A | −103542.753（表示丸め） | −104225.625 | −682.872228 |
| F | −97814.4375 | −104225.625 | −6411.1875 |
| S | −104149.536（表示丸め） | −104225.625 | −76.0890234 |

q64 CXのpaired差は全8本で−14128＝208−224q、整数残差0。しかし絶対CX平均の残差はStrong/Reference共通でA +455.122、F +5024.625、S +8.171（原報告丸め）。**paired差の一致だけでは総費用に必要な両絶対価格は確定しない。** Sの点残差が小さくても、A→Sを採択しない。[D64]

C8-1旧sample00の4postは2e−10のobservable閾値でFAIL。保存QPYのままAer simulation fusionを無効化して4postを通し、その後別SOURCEで残7を完了した。閾値緩和や価格対象P0の融合変更ではない。旧FAIL/STOPを残し、現在未完了なのはq≥128・全law/tail・外挿bias・actual RPE成功保証等である。[D64]

共分散はC6aのpaired統計、C7-4のq内36量・共有bootstrap係数、q64の方式/軸/資源28×28とpaired12×12を再利用する。q64 n8のrank上限7、X/Y依存、SD0を母分散0としない。D B/Cは共有係数によるround間依存を既に伝播し、Fのq2/q4直接平均との共有も別component化した。A/Sの直接平均とfit係数のcross-covariance、q1 joint covariance、model bias/law距離等は不足し、**total covariance/SD/CIはnullのまま**。既知blockの単純和で埋めない。[D74][DTOTAL]

DF12ではA0 r32/q16約3.345%、q32約4.911%のselected RZ点誤差、q32標準化残差約2.012、full-basis補助約5.598%の非gating FAILを保存。R1は共有anchor・未知PSD相関・長q感度を既に解析し、点選好Partial／総合UNDETERMINED。局所r1/q16,32の寄与は約0.0147124%で、順位決定目的のR2/R3は非推奨だった。同じ再解析・低寄与補完を新計画として繰り返さない。[V3][R1]

## 5. 総費用への接続と高寄与域

資源gごとに

\[
G_{g,\mathrm{RPE}}=\sum_{m,a}N_{m,a}C_{g,m,a},\qquad
\widehat G_g=\sum_{m,a}N_{m,a}\widehat C_{g,m,a}.
\]

ランダム回路ではCは同じlawに対する一shotの期待価格を指す。直接価格標本とその保存平均は母平均そのものとは区別する。N固定の価格差は `δG_g=ΣN(C−Ĉ)` と分解できるが、**今回はこれを新たに数値評価していない**。quantum shot Nと古典価格標本n、信号normalization B/rangeと価格を分離し、attenuationをCへ二重乗算しない。widthはpeakで、Nやround数を掛けない。

将来の最小接続表は既存行を参照して、`study/m/axis/q/T/r/K/N_origin/C_direct_origin/Ĉ_model_origin/saved_residual/price_scope/compiler/uncertainty/missing`を分ければよい。新たな管理frameworkやapproval記録は不要。Dのprices52・contributions・paired_rounds、DF12のV3/R1表を使う。DF12のnormalization Bは元signal gridの`finite_distribution_normalization`、attenuation・r/K/qは同じpoint、N/α/モデル単価は最終opt2のround/axesから参照する。有限時間H4 PのNは`comparisons[*].saved_shots`であり、別RPEのNを新しく付与しない。

D旧Aの末尾roundの既存寄与行の例（X軸、Reference−Strong、**proxy下の保存値**）：

| m / q | 保存N | 保存paired RZ寄与 | 保存paired CX寄与 |
|---|---:|---:|---:|
| 10 / 1024 | 589 | −35373257.7929682 | −134979952.00000027 |
| 11 / 2048 | 679 | −81866440.29557249 | −311351376.00000066 |
| 12 / 4096 | 910 | −219850963.0078091 | −834739360.0000017 |

出典はA/paired_rounds.csv。新しい順位・寄与率や両軸合計を算出していない。絶対総資源の約99.4〜99.5%が長q proxyという既存分解と併せ、高qのmodel biasを軽視しない。legacyは前記末尾3roundの保存寄与率を使う。q64の接続完了だけで高qの不足が解消したとはいえない。[DTOTAL][V3]

Track CのD/RTE/wrapper分解、独立holdout、共有cellの係数を先に合算して共分散を取る方法も再利用できる。ただしC最終P3は既にh/resolution/αを共同選択済みで、同じ探索を未実施扱いしない。新Aのmax-plus/native DAG、有限strict-T、Q後続のNはそれぞれ条件付き別価格／別測定であり、DやPF/PRへ改善率を掛け合わせない。[COPT][NEW_A][JOINT]

## 6. 再利用可能な技術と直接移植できないもの

| 再利用する技術・既存コード | 今回の用途／制限 |
|---|---|
| C6a `describe/paired_statistics`、C7-4 `event_features/design/residual_summary` | whole-path pairing、q別ddof1共分散、低次元model・特徴の分離。**既存出力を読む**。fit/bootstrap/特徴再集計は今回起動しない |
| B4保存row、B6b `read_b4/price_catalog/aggregate_registered`、`track_d_bc_total` | N/C/総費用の既存接続とscope照合。`evaluate/model_value`は新数値処理なので読取と区別 |
| 同日`track_d_resource_interface` | `SavedCDF8Study.common_shots/select`は保存fieldの選択。`aggregate_supplied`は既存`c1_stage3_accuracy.aggregate_totals`への薄い算術接続。再実装不要 |
| D native count／wire max-plus／有限P0 bridge | 固定native列の厳密計数、位相・zero/special angle・cut/CSWAP界面の扱い。q4/q8固定8wrapper・56価格照合済み。**任意q/lawのP0後期待価格は未資格**。[BRIDGE] |
| DF12 V3 metadata表・R1固定plan | N固定の共有校正、holdout条件照合、systematic感度、低情報価値の追加計算回避。保存結果で満たす部分は終了扱い |
| Track C P3層別価格MC／独立holdout／共分散、Full/Truncated対照 | 方法・コードの再利用候補。218成分、weighted range、256-bit law、compiler AをCDF8 K0やDF10/19へ変えた数値は未資格 |
| Track B finite-bit moment/range／return会計 | weightedとunweightedのN/単価接続を区別する方法。G6/G7やC1-P5の係数・N・有限toy利益をこのRPE価格として直接利用しない |

再利用の単位は**コード／方法／同条件の数値**を別にする。条件差がある数値係数は方法の参考に留める。SOURCE・runner・test・成果物の具体的pathは索引に集約した。

## 7. 新compileなしで可能なこと・追加実行が必要なこと

**今回までの静的作業で完了できるもの**：保存N/C/Ĉ/残差/寄与の所在・schema照合、条件別coverageの確認、既存結論の引用、read-only APIのSOURCE確認。旧13round会計・DF12 R1を再実行しなくても提示可能。

**新compileなしでも別の科学実行許可が必要なもの**：q64直接価格を旧13roundに差し込む新G評価、固定Nで新しい価格感度／誤差許容幅を算出すること、保存価格からの新共分散推定・再fit・新N/α配分・候補再選択。軽いscalar処理でも今回は行わない。

**新たな科学計算が必要になり得る部分**：D q≥128の独立直接価格、DF10/19をRPE時間域へ延長するsignal/N/価格、変更したDF/LD/r/K/h/PF/law/control/compilerの価格資格、全law/actual誤差資格。既存native bridgeの同条件適用が成立する範囲を先に確認し、全長qのfull compileを既定にしない。未取得領域を0や資格済みへ埋めない。

## 8. 最小の次の実行案（未承認・未開始）

**E1：固定Nで価格だけを更新する単一の条件付き評価**を具体的な承認候補とする。主目的は、最新直接価格と旧model価格の接続を最小差分で示すこと。PF/PRの最終順位や長q資格の完成ではない。

1. **ユーザー**：D r32/CDF8の13roundを対象とするか、legacy PF/PRの保存R1提示で十分かを選ぶ。DF10/19を新しいRPE taskにする場合は、その別条件を先に定める。元H/DF12/PF-PRとD/control比較を同じ評価に混ぜない。
2. **Codex（実行許可後）**：Dを選ぶ場合、SOURCE基点`1f5ea971f`のresource interface／`8828b298`の会計、B4 `07d18bfe`の固定N、q64 `fe361b206`の保存平均を使用。**q64だけを有限直接平均へ置換**し、q1..32の保存値とq128..4096の保存A/F/Sを固定した全13roundの条件付き価格感度を一campaignで評価する。旧総値と差分を別namespaceへ保存し、旧結果を更新しない。q64診断を再実行・再fitしない。
3. 同じcampaignでN変更0、signal/sampling/build/compile0、fit/bootstrap0、新PR探索0。A/F/Sを都合のよい資源別に選択せず、全6資源とpeak幅を併記。q64は有限n8、長q期待価格・cross-covariance・biasの未資格、total CI=nullを明記し、公開照合後STOPする。q64置換は小さい接続確認であり、高寄与長q問題の解決を期待しない。
4. **ユーザー**：既存長q感度・E1の限定結果を踏まえ、価格資格の追加が目的に必要かを判断する。必要なら高寄与q域の反証／資格目的・対象builder・q/path集合・精度・STOPを別途固定して承認する。r16/新A Q/Closed P3/shot配分変更はこのE1へ追加しない。legacyのR1/R2/R3を機械的に再提案しない。

既存の全13round点推定と保存残差の提示だけで目的を満たすなら、**E1自体を省略できる**。新しい計算を必須にしない。Nの再配分・PR再最適化は、価格側の不確かさと目的が明らかになってから必要性を判断する。

## 9. 調査範囲・branch・終了条件

既存横断調査`2e79d80d`と同日統合計画`1f5ea971f`を再利用し、追加分を本調査に限定した。前者のcleanなworktree／branch **`codex/pr-rpe-cost-refocus-survey-20261108`**を継続。D統合枝・q64保全枝・r16枝・他担当worktreeへmerge/checkout/修正せず、固定Git objectを読んだ。新branchは不要と判断した。

projects/AGENTS、worktree/AGENTS、PROJECT_MAP、正式共通運用指示2026-11-09の実行／再利用／branch方針を適用した。checkoutに研究概要がないため公開統合`141fa7624`の研究概要を取得し、旧時点の本文と後続の個別結果を区別した。status/manifestの旧q64未完了記述は、最新完了資料で補って読む。全raw再監査・別サーバー照会・外部再現・独立科学レビューではない。

**今回の新signal／shot再算定／価格再fit／trajectory sampling／回路build/compile／新G数値再解析／科学test／研究module importは全て0。** JSON/CSVは保存fieldの読取・転記に限り、平均・合計・差・比・順位・共分散を再計算していない。SOURCE/raw/結果/grant/ledger/marker/mandatory STOPを変更せず、非公開event/QPY/NPZや内部保存先を新たに公開しない。

公開対象は本報告・固定索引と入口リンクのみ。commit/push後、remote SHA・tree・公開本文を照合して**mandatory STOP**。次担当は**ユーザー**：上記E1を行うかと、その科学実行範囲を判断する。**Codex**：明示許可まで新しい科学処理を開始しない。

[DB4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d18bfe1d5145d6f8de026c0d994d5cb1c783c8/docs/research/track_d_b4_2_rpe_shots_20261106/RESULT.md
[D6B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fed5f458be2f625de4e1b8c98b8cc5da6e551b14/docs/research/track_d_c6b_q8_20261107/RESULT.md
[D71]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86c5c9d48dbddc085f388b6521be8cd92ee49d0d/docs/research/track_d_c7_1_q16_20261107/RESULT.md
[D73]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/acea43987edae0d5eb0c186a89aa8cd7139fbd7b/docs/research/track_d_c7_3_q32_20261107/RESULT.md
[D74]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64072ba81bb70db1df1b34e94adb08e78926f24b/docs/research/track_d_c7_4_model_stability_20261107/MODEL_REPORT.md
[D64]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe361b206a6572142775ac01972545d312e5fb14/docs/research/track_d_c8_1_q64_20261108/COMPLETION_RESULT.md
[D16]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d68cf3c19685da586ceadbbac8193f55b9ba0fa0/docs/research/track_d_r16_price_calibration_20261110/REPORT.md
[DTOTAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8828b298975c88a2c0d1fd57ff6ed905233840f5/docs/research/track_d_bc_total_20261108/REPORT.md
[PROTO]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1f001cfc4e8e0531279a26613be29a52c7c0640c/docs/research/track_a_pr_resource_evaluation_protocol_v2.md
[V1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md
[V2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
[V3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc60beb51930d4311d2ded84440682e583ccafc7/docs/research/pr_rpe_v3_cost_transfer_preflight_20261103/REPORT.md
[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/141fa7624d74aa5deacbd752d09df516cad9206f/docs/research/pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/REPORT.md
[COPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md
[NEW_A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/224440f740051ed5794e8db87a57908fc02dd02f/PROJECT_MAP.md
[JOINT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1f5ea971f8ceac54f2dc00e71824ab8dabf95837/docs/research/pr_rpe_joint_integration_20261110/REPORT.md
[BRIDGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c002153a98a2f2d209a4a60f035c6bd7e36293c9/docs/research/track_d_bridge_run_reuse_20261110/REPORT.md
