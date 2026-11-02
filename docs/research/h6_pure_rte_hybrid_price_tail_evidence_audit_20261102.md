# H6純RTE／Hybrid：価格tail・母期待compiled costの既存証拠調査

2026-11-02 JST。B専用branch：`codex/h6-pauli-price-tail-audit-20261102`、読取基点：`71432728ee0140fb3c1f8bf30ecf51b2faba86ec`。

**保存価格標本と仮想ρ=0における純RTE低Gは既存結果の範囲で報告できる。真の母期待価格・母平均Gの順位とtail保証は未確定である。**
固定3候補すべてでOrder2を観測し、eventとFull価格の対応も保存されている。しかし、Order2の価格寄与、component組合せによる高価格層、未観測trajectoryの寄与を制限する証拠は未取得である。
希少eventの存在は高価格tailの実証ではなく、未観測寄与が0という証拠でもない。

本書はSOURCE・保存結果・既存統計の読取、archive/memberの所在・SHA・schema確認と技術的整理である。
新しいsampling、RNG replay、回路build/compile、QPY decode、状態作用、保存価格再集計、新平均・分散・SE・信頼区間・相関・条件付き価格・tail確率・数値上界、N/G・順位の再計算は行っていない。
詳細科学レビュー・主要claimの正式採択ではない。公開照合後mandatory STOPとする。

## 1. 目的・固定条件・調査Aとの境界

linear H6、隣接1.00 Å、STO-3G、中性singlet、保存Pauli-JW918非identity項＋constant、system12 qubits。
T=.8、S2、q8、K2、P=0、central-rotation control、cosine／+Im sineのFull13-qubit Hadamard wrapper。
Qiskit1.3.0、basis `rz,sx,x,cx`、optimization level1、seed17、backend/coupling/layout/routing追加指定なし。DF rankは適用外。
保存53-bit実効CDF・binary64 event角度、価格・B・bias・shot会計を保持する。
主価格は状態準備を除く**Fullのcompiled RZ count**であり、CX/depth等の個別記録もある。event-only価格や実行時間・FT総資源へ読み替えない。

| ID | 固定candidate | L_D | r | event位置数/trajectory | 比較の意味 |
|---|---|---:|---:|---:|---|
| R64 | `H6_p000_q8_r64_K2` | 0 | 64 | 512 | 両比較で同じ保存標本を共用 |
| H32 | `PR_p0078_q8_r32_K2` | 78 | 32 | 256 | R64との主要family比較。rも異なる |
| H64 | `PR_p0078_q8_r64_K2` | 78 | 64 | 512 | R64との同q/r/K対照。prefix・残差辞書・lawも異なる |

[完了した調査A][A]（commit `0e1c7aae386986f6c1bb9483548671655447f516`）を参照した。
Aは条件付き全law誤差転送式を確認したが、native相対位相とcompilerの一様数値誤差評価が未完成で、全law実装誤差上界は未確定である。
Bではこの不足を再調査・評価せず、価格標本と母期待値の接続だけを扱う。Aの完了と[局所診断][LOCAL]のengineering確認は、価格tailを解決しない。
[局所STOP][LOCALSTOP]も維持する。

## 2. SOURCE・結果・保存標本の対応

| 対象 | 科学SOURCE | 固定結果commit | 既存の読取入口 |
|---|---|---|---|
| R64 | `e755ecf581600192fd2745d7066e410cebd933d3` | `10cb3e5cca7834ebf75301bd91bd093ca51d4a6b` | [event拡張報告][RREPORT]、[candidate_results][RROWS]、[verification][RVERIFY]、[archive manifest][RARCH] |
| H32/H64 | `887b69e449cb3c378286f09d0d7626599850fb67` | `00958ae1945c780718b2deef07d5dab1096a5cd8` | [元REPORT][HREPORT]、[candidate_results][HROWS]、[price_statistics][HSTATS]、[comparison_details][HCOMPARE]、[archive manifest][HARCH] |
| Hybrid統計・比較 | 解析SOURCE `af891fd45805454c6592bbee5441266f14235ffd` | 同上 | [解析SOURCE][HANALYSIS]、[sampling_design][HDESIGN]、[confirmation_plan][HPLAN]、[event_orders][HORDERS] |
| P2の部品価格 | `11e50e262f95630b14fa4070ad30894ba3f64b74` | `c9073274daee69c9b69d8db3dc332bf6132c2259` | [固定index0000のREPORT][P2] |
| 既存対応表 | — | `b3bab573cf17b9dd8ddc1dca5f1a6a6c935ef6d1` | [調査書][INVREPORT]、[INVENTORY.json][INV] |

元SOURCEの`run_pauli_rte_event_extension.py`、`run_h6_hybrid_fairness_extension.py`、`pauli_h2_h6_study.py`、
`pauli_rte_event_extension.py`、`h6_hybrid_fairness_extension.py`、`h4_pauli_pr_search.py`、`rte.py`、`rte_compiled_cost.py`を静的に確認した。
Hybridの実価格workerは`pauli_h2_h6_study.price_job`、R64はevent拡張runnerの`price_worker`を使う。
いずれも同じbodyから両軸wrapperを作り、`transpile_and_measure_cost`でFullをcompileして価格を保存する。汎用統計APIの存在だけを、当該結果に統計が保存されている証拠とはしていない。

### Archive内の保存範囲

| 対象 | manifest namespace / candidate prefix | confirmation index | 保存内容 |
|---|---|---|---|
| R64 | `cases` / `cases/H6_p000_q8_r64_K2/` | `0000`–`0047` | `law.json`、各`prices/confirmation/NNNN/result.json`、両軸input/compiled QPY |
| H32 | `H6` / `H6/batch01/cases/PR_p0078_q8_r32_K2/` | `0000`–`0063` | lawを内包する`spec.json`、各result、両軸input/compiled QPY |
| H64 | `H6` / `H6/batch01/cases/PR_p0078_q8_r64_K2/` | `0000`–`0063` | 同上 |

全確認indexについてresultと4 QPYのmanifest登録を確認した。各resultはseed/index/cohort、event列、orders、Full両軸のRZ/CX/depth/size/phase、QPYのraw/gzip hashを持つ。
Hybridは`order/products/rotation/angle/phase/signs`というcompact event schema、R64は`application_sequence`を含むfull event schemaで、R64にはstep seedsとcontrol sidecarもある。
保存lawの順序を使ってcomponent IDをlabel・符号へ対応付けられる。seedだけからeventを復元する必要はない。
ただしeventごとの**compiled価格**は保存されておらず、価格は結合したtrajectory全体に付く。

既存INVENTORYとP2の`selected_members`／`recover_member`を再利用し、3 law/spec・3 index0000 result・R64の確認plan/decisions/COHORT_STOPの計9 JSON memberだけを選択読取した。
archive part、結合archive、選択memberのSHAは一致した。全archive展開・QPY読戻し・元rawの書換えはしていない。
全確認resultの科学値を今回再監査・再集計したわけではなく、全体のevent集計・QPY検証は元verificationの既存証拠として引用する。

## 3. 標本の生成・選択・独立性

**価格の標本単位はtrajectoryである。** 同じevent列を共有するcosine/sineを独立2標本と数えない。
方式間の同じindexもpaired標本ではない。候補別seed addressが異なり、R64を2比較で独立な別cohortとして数えない。
両軸RZが保存標本で同値でも、両programが同一、または全lawで価格同値と新しく証明したことにはならない。

| 項目 | R64 | H32/H64 |
|---|---|---|
| 探索価格 | n8。大event数探索からR64をnominate | 各n128。適格な固定4候補すべてを確認へ |
| 確認価格 | n48。探索とは別seed tag | 各n64。探索とは別seed tag |
| 標本数決定 | 事前の48→96→192→384延長規則。保存n048判定は全5精度`extend=false`、延長0 | 探索前にn_E設計、探索結果でn_Cを決め確認前に一度固定。確認top-upなし |
| 選抜 | 全5精度で探索nomineeはR64。確認集合minもR64 | 全5精度の探索nomineeと確認後minはH32。H64も確認済みの対照 |
| pool | 探索と確認を混合しない | 同左 |

H6のseed addressは両元SOURCEとも`SeedSequence([20261025,6,L_D,q,r,K,tag,index])`、探索tag101／確認tag202。
保存confirmation/index0000 seedはR64=`1281903131106297017`、H32=`9082285277357406555`、H64=`10692284186540933136`。
今回はseedを再導出・再選定せず、保存記録と生成関数を参照した。address分離や旧衝突検査はPRNG stream独立性の数学的証明ではない。
Hybridの`config.cost_samples=8`はlegacy descriptor metadataであり、実確認標本数64を8へ読み替えない。

R64は確認データで延長可否を判断する設計で、実際に延長しなかったことだけから固定nの厳密coverageを認定しない。
Hybridは確認前固定nでも、探索による候補選抜、確認後min、旧RTEとの再利用比較を含む。
全5精度は同じ価格標本を共有し、5独立実験・選抜補正済み母平均順位とは扱わない。
R64のplan/decisions/終了は[RARCH]の`confirmations` namespace、指名は[confirmation_nominations][RNOM]に保存されている。
P2とselected implementationのindex0000診断も同じ保存標本の再利用であり、独立した価格確認の追加本数ではない。

## 4. 保存価格分布・既存統計の取得状況

下表は保存文字列・数値の引用であり、raw価格からの再計算ではない。価格単位は両軸それぞれのFull compiled RZ。

| 候補 | n | 保存平均exact（cos=sin） | 保存SE（各軸） | 保存済み観測範囲 | 保存sample variance／cos-sin covariance |
|---|---:|---|---|---|---|
| R64 | 48 | `149519/48` | `11.652073811073278` | 集計欄は未取得。raw配列からmin/maxを新計算していない | 独立した分散/共分散欄は未取得。paired G_SEは保存済み |
| H32 | 64 | `299249/64` | `6.034241832887364` | `4576`–`4776`（各軸） | 対角・交差とも`1044007/448` |
| H64 | 64 | `432931/64` | `8.302351027455133` | `6605`–`6911`（各軸） | 対角・交差とも`17786999/4032` |

R64は[RROWS]の`price_statistics.{n,values,mean_exact,SE}`、Hybridは[HSTATS]の`candidate/confirmation`から引用した。
Hybridの`covariance_exact`はsample covariance行列で、対角はsample variance。R64にも価格配列はあるが、新しい分散・共分散を評価していない。
Hybrid保存nominal95 halfwidthはH32=`12.058470098033485`、H64=`16.590924656402322`。
いずれも`population_mean_upper=null`、`nominal_t95_is_tail_bound=false`である。
R64も[RVERIFY]で`population_tail_guaranteed=false`。標本最大を全lawの最大価格として代用しない。

`rte_compiled_cost.CompiledMetricAccumulator`には分散・SE・min/max処理がある一方、今回の元runnerは上記の保存schemaを使う。
共通libraryの能力を理由に、R64未保存統計を補完しない。
[既存Hybrid解析][HANALYSIS]は両軸の共分散を含むshot加重G_SEを保存している。
`saved_paired_covariance_reanalysis.json`の対象は旧r8のHybridであり、R64の共分散資料とは扱わない。

## 5. 実効law・Order0/2・component support

[RVERIFY]の`observed_orders`と[HORDERS]を照合した。次は観測event回数で、trajectory数でもlaw確率でもない。

| 候補 | 確認n | event位置数/trajectory | 保存Order0 | 保存Order2 |
|---|---:|---:|---:|---:|
| R64 | 48 | 512 | 24563 | 13 |
| H32 | 64 | 256 | 16374 | 10 |
| H64 | 64 | 512 | 32765 | 3 |

**K2ではorder=[0,2]であり、Order4はsupport外である。未観測tailにOrder4を加えない。**
13/10/3件が何本のtrajectoryへ分かれたか、何件が価格へ寄与したかはこの集計だけでは分からず、今回は新たに分類・集計していない。

保存`order_mapping.exact_masses`をそのまま引用する。名目`order_probabilities`や観測頻度と区別する。

| 候補 | Order0の53-bit実効質量 | Order2の53-bit実効質量 | 保存angle[order0,order2] |
|---|---|---|---|
| R64 | `9003777501913741/9007199254740992` | `3421752827251/9007199254740992` | `[0.02756704744929354,0.009191085413279683]` |
| H32 | `9002910591840289/9007199254740992` | `4288662900703/9007199254740992` | `[0.03086302564478063,0.010290579627970677]` |
| H64 | `9006126365457415/9007199254740992` | `1072889283577/9007199254740992` | `[0.015435188428266028,0.005145426034031545]` |

componentごとのlabel/sign/coefficient_abs、nominal選択確率、binary64 CDF、53-bit exact massesが順序付きで保存されている。
R64は918成分、H32/H64は840成分。配列長対応を確認し、保存mass文字列に0/負の分子はない。
従って既存IID 53-bit uniform-wordモデルでは全登録成分が到達可能である。新しい確率・support総数は計算していない。
個々のcomponent確率の正本は当該lawの`component_mapping.exact_masses`であり、名目係数比で置換しない。
例えばR64先頭成分`ZIIIIIIIIIII`の実効質量は`216261314304793/9007199254740992`、
Hybrid先頭成分`XXYYIIIIIIII`は`8812733459675/2251799813685248`。component IDは候補の辞書内indexで、方式間の同じIDを同じPauli項と仮定しない。

H32/H64のcomponent列・component CDF/masses・保持列は保存JSONで同一。rの違いによりorder確率とevent角度が異なる。
R64は保持prefixがなく、辞書・正規化されたcomponent law・order lawも異なる。
`rte.iter_sample_rte_events`はorderを選び、componentをorder+1個選ぶ。Order0はrotation1個、Order2はproduct2個＋rotation1個で、componentの重複選択も許す。
draw順と実際のapplication順は区別して保存される。
53-bit到達可能性を、固定seed集合が全組合せを観測したという証拠へ拡張しない。

[HDESIGN]にはH32/H64の`probability_any_nonzero_per_trajectory`としてそれぞれ`0.11478095708121804`／`0.059167660001485045`が既に保存されている。
これは既存実効massから作られた標本設計上の値の引用で、観測頻度・高価格確率・tail寄与・分散上界ではない。
元欄も`not_a_tail_or_variance_bound=true`である。R64の同種確率は今回新計算していない。

## 6. 希少event・価格tailに関する到達点

| 問題 | 既存証拠から確認できること | 未取得・未検証 |
|---|---|---|
| Order2の多いtrajectory | 生成構造上productが増える。保存eventとFull価格のjoinは可能 | 多いほど高価格という単調性、条件付き平均、Order2寄与率、相関 |
| 特定component組合せ | 項別Pauli support・符号・application順・角度が保存され、回路構造を静的に追える | 全component組合せの価格、特定組合せが高価格層を作るか、その確率と寄与 |
| 隣接eventのcancellation/fusion | central制御ではbasis/parity/rotation/product列を結合してFullをcompileする | どの隣接patternが価格を上下させるかという全law分類・寄与 |
| 保持部とeventの結合 | P2固定programに正負の非加法性の実例あり | 全trajectoryのinteraction分布、保持部だけの因果配賦 |
| 同じ次数内の変動 | componentのX/Y/Z配置・support幅、繰返し/隣接順・角度・境界が異なり得る | 次数固定での価格分布・条件付き分散・tail上界 |
| 未観測trajectory | 有限support内でも未検証の組合せを排除する価格保証がない | 観測最大を超える価格の存在、その大きさと母平均寄与 |

非加法性は[P2]の保存値を引用できる。index0000のFull RZはR64=`3081`、H32=`4732`、H64=`6735`（両軸）。
H32はOrder2を1個含み、H64/R64は全Order0である。
`F−R−E+Empty`のRZはH32 cosine/sine=`89/91`、H64=`72/75`、R64=`-3/0`。
Hybrid保持単独は`2500 RZ`。H64/R64は同じevent位置数でもevent-only価格が異なる。
これらは固定programの削除操作に対する価格で、Order2の独立価格や母平均tailの測定ではない。

Order2が高価になる可能性はあるが、productの相殺・rotation融合などを経るため、次数だけで価格は決まらない。
また**Order0のみの層にもcomponent列由来の価格変動は残る**。Order2を追加観測するだけで全tailが閉じるとは限らない。
一方、有限辞書・有限event位置数・固定Kのlawなので、正常にcompileされる全経路の価格は有限support上の量である。
ここでいうtailは希少な高価格部分を指し、無限supportのheavy-tailや無限期待値を実証した意味ではない。
全supportの存在と有限性だけでは、数値的に有用な最大価格・期待値・順位は得られない。

## 7. 既存資料だけで支持できる費用比較

次のGは元candidate_resultsからのexact文字列の引用である。新しい費用比・SE・順位は作っていない。

| 候補 | ε_sig=.05の保存G | ε_sig=.0005の保存G |
|---|---|---|
| R64 | `570714023/6` | `45657735316457/48` |
| H32 | `3435079271/32` | `63691899506611/32` |
| H64 | `973661819/8` | `72211615818205/32` |

[HREPORT]の主要family比較は全5精度でR64の条件付き点Gが低く、保存H32/R64比は約`1.129/1.152/1.183/1.489/2.092`。
同q/r/K対照のH64についても[HCOMPARE]の既存`primary_comparisons[].versus_RTE`は全5精度で`REFERENCE_LOWER_EMPIRICAL_G`である。
これは固定標本・仮想ρ=0・保存bias/B/shot条件における結果であり、母平均winner、prefixだけの因果効果、大域最適性ではない。

| 主張段階 | 現在の状態 | 必要な証拠 |
|---|---|---|
| A：保存標本と仮想ρ=0における条件付き低G | 既存結果で報告可能 | 保存Full価格・N/G・条件と限界の明記。追加tail実行は必須でない |
| B：真の期待compiled costでも純RTEが低い | 未確定 | 同じlaw・compilerにおける母平均価格/tailの制限、選抜・標本設計を反映した比較根拠 |
| C：実装精度を含む総費用でも純RTEが低い | 未確定 | Bに加え全law実装誤差・必要shot数の評価。A調査の未解決事項を参照し、本Bでは扱わない |

単回価格の母平均順位と、候補ごとに異なる保存Nで重み付けした母平均Gの順位も区別する。
ρ=0のNを固定した母平均Gを調べても、実ρを含むCが確定するわけではない。

## 8. 母期待価格へ進む方法と不足証拠

軸aについて`μ_X,a = E_{ω∼P_X}[C_X,a,compiled(ω)]`を対象とする。
`P_X`は保存実効law、Cは保存profileによるFull価格。保存標本平均はμそのものでも、その上界でもない。

| 方法 | 現在できる技術的整理 | 不足する証拠・限界 |
|---|---|---|
| 1. 既存標本の経験的評価 | 保存平均/SE/Hybrid観測範囲とcohort設計を引用。両軸依存を維持 | 未観測層の価格をSEは制限しない。名目t95や2 combined-SEはtail・逐次観察・選抜後同時coverage保証ではない |
| 2. 価格上界による評価 | 有限event構造、Pauli support、保持列、central emission、Full wrapper、固定compilerが特定済み | 全到達経路または各層を覆うcompiled RZ上界が未取得。native命令数・孤立event価格・標本最大からの直接代用は不可 |
| 3. 希少層を分けた評価 | 保存eventで層を定義でき、同じresultのFull価格と対応付け可能 | 実効lawの層質量、層内条件付き価格/上界、残余層の制御が未取得。既存raw再分類も新数値再解析として今回は実行しない |
| 4. 現結果で終了 | 主張Aを条件付きcase studyとして維持 | B/Cを主張しない限り追加tail検証は不要 |

価格上界を作る場合は、元`append_pauli_event`／D2のbasis変換・parity・CRZ・controlled products・scalar phase、
保持回路とwrapperの構造から出発し、**現行compilerの到達lowering/最適化が出力するgate数**へ接続する必要がある。
1Q融合・basis translationなどの到達規則と最大出力長、全対象での正常終了、backend/topology条件、価格指標の一致が必要である。
単に「opt1は最適化なのでRZが増えない」と仮定しない。
この価格個数の議論はAのoperator誤差証明と別であり、Aの一様数値誤差証明をBの前提に新しく要求しない。
現資料には上記を閉じた価格上界がない。今回は証明実装・数値上界構成へ進んでいない。

残余層の質量と価格上界があれば未観測寄与を制限する経路はあるが、質量だけでは費用寄与は制限できない。
全候補の上界だけが得られても順位は通常決まらず、R64上側とHybrid下側など、比較を分離する両側の根拠が必要である。
保存Nを使うGについては軸別価格を同じNで一回だけ重み付けし、両軸・候補間・精度間の依存と選抜の扱いを固定する必要がある。新しい損益分岐・N/Gは未計算である。

## 9. 追加samplingなしで可能な検証案

以下は案であり、今回実行した数値解析ではない。

1. **追加科学計算なし**：元SOURCE・law・manifest・cohort設計・既存統計の参照整備、event/価格joinのschema確認。本調査で主要な所在不足は解消した。既存証拠を報告するだけならここで終了できる。
2. **追加sampling/compileなしの保存値再解析**：固定3確認cohortのevent記録をOrder2個数・位置や事前定義のcomponent分類で価格と結び、条件付き記述統計を作る余地がある。ただし新平均・分散・相関・寄与率を作るため別の明示実行許可が必要で、未観測層の保証にはならない。R64の未保存の記述統計補完も同様である。
3. **回路生成なしの構造的検討**：現行compilerの出力gate数を制限できる範囲を静的に絞る案。samplingや全law実装誤差評価とは別課題で、数値上界・期待値の実評価は別許可にする。有用な比較幅が得られる見通しがなければ、大きな証明基盤を作らず未確定で終了できる。

保存標本の再解析だけでは未観測component組合せを価格付けできない。
追加samplingなしで母平均順位が必ず確定するとも、追加samplingが必ず必要とも、現証拠から断定しない。

## 10. 追加samplingが必要な場合の限定案（未実装・未承認）

主張Bを目標にし、保存値整理や有用な価格上界だけでは比較を分離できない場合に限って検討する。
対象はR64/H32/H64の保存law・Full wrapper・compilerを維持し、新候補・分子を追加しない。

- 排他的で全lawを覆う層を事前に定義する。第一候補はtrajectory内Order2数が0、1、2以上の層。Order2位置、product/rotationのPauli supportや境界patternは、必要な場合だけ追加分類する。Order0層のcomponent tailも残す。
- 各層の質量`w_X,s=P_X(S_s)`は保存53-bit実効massから対応付け、層内は元lawを条件付けた分布から取得する。記号的な対応は`μ_X,a=Σ_s w_X,s E[C_X,a|S_s]`である。今回はwも条件付き価格も数値評価していない。
- 希少層を過剰samplingする場合、取得本数の比率をwとして使わない。異なる提案分布を使うならsupportと尤度比を固定し、適正な重みを要する。強制Order2を任意位置へ置くだけでは元の条件付きlawにならない。
- 毎trajectoryの両軸は同一event列で価格化し共分散を保持する。方式間indexをpaired化せず、旧探索/確認と新cohortの役割を分ける。
- 未取得層・高Order2数を切り捨てず、残余質量と価格の上界等で扱う。全lawの高信頼順位を目標にするなら、全層を覆う誤差/確率配分と選抜・逐次停止の扱いが必要で、単なる「希少eventを一度観測」で終了しない。
- 科学SOURCE/入力、層・条件付きlaw、標本数/seed・精度・compile件数・統計/停止規則を実行前に限定する。既存STOPのresumeではなく、新しく許可されたscopeにする。今回は件数を推定して実行契約を作っていない。

目的は「未観測層の費用寄与が、固定3候補の母期待価格またはρ=0・保存N条件付き母平均Gの比較を左右し得るか」を判別することである。
全lawρや実装込み必要Nへ自動移行しない。

## 11. 結論表・未確認事項・STOP

| 調査項目 | 既存証拠 | 不足する証拠 | 条件付き報告を維持する場合 | 母平均の順位まで主張したい場合 |
|---|---|---|---|---|
| 固定標本平均 | 3候補の確認平均・SE・raw対応。Hybridは分散/共分散/範囲も保存 | R64の独立した分散/共分散/範囲集計欄は未取得 | 追加実行不要。未取得欄はそのまま明記 | 記述統計の補完だけでは不足。law全体に妥当な推論根拠が必要 |
| Order2寄与 | 13/10/3件の既存観測、event/Full価格join可能 | 条件付き価格・寄与率、未観測多発層の価格/質量 | 追加実行不要。高費用tailの実証とはしない | 保存値再解析は候補。残余制御や追加層別価格が必要かを限定判断 |
| Component由来の価格tail | 全登録成分に正の保存実効mass、ordered dictionary、P2の非加法性 | 未観測組合せの価格と母平均寄与、全経路/層別上界 | 追加実行不要。存在・ゼロ寄与のいずれも断定しない | 構造的上界または妥当な層別等の追加根拠。新実行は別許可 |
| 母期待価格 | 固定law/Full価格の定義と保存標本推定値 | 母平均上界/下界・tail/coverage保証 | 追加実行不要。標本平均と区別 | 追加根拠が必要。必ず新samplingと決めず、上界経路も検討 |
| 純RTE／Hybridの母平均順位 | 既存有限標本・ρ=0ではR64低G。H32/H64の既存比較あり | 母平均を分離する両側の根拠、選抜/逐次設計への対応 | 追加実行不要。現claim Aを保持 | 新解析・有用な上界・必要時の追加価格のうち、具体scopeを別許可。順位は未確定のまま |

既存証拠ではtailが順位を覆すとも、覆せないとも判定できない。Aの全law誤差課題を重複して調べる必要はない。
**停止理由は資料の所在不明ではなく、母平均・tailの新しい数値証拠が未取得で、その生成は本Bの許可範囲外であること。**
主張Aを使い続けるための追加科学計算は不要である。主張B/Cを自動採択せず、今回の文書公開をもって終了する。

次の担当は**ユーザー**：条件付き報告Aで終了するか、母平均順位Bまで必要かを判断する。
Aで終了する場合、**Codexの次の作業なし**。Bを求める場合だけ、**Codex**が選択された限定scopeを次の指示に従って扱う。
新しい数値再解析・層質量/上界/期待値評価・sampling/compileには具体的なSOURCE・入力・条件・範囲に対する明示実行許可が必要で、今回開始していない。

## 12. 技術照合・provenance・保全

P2の選択archive読取関数と`restore_pauli_h2_h6_evidence.recover_member`だけをAST抽出して使用し、科学moduleやrunner全体はimport・実行しなかった。
`saved_rte_cost_diagnostics.py`と`audit_h8_rare_tail_selection_evidence.py`も再利用候補として読んだ。
前者はH2固定schemaと価格再集計を含み、後者はH8固定資料・別lawを扱うため、そのまま実行・移植していない。H2/H8のtail結果をH6へ転用しない。
[旧H6希少次数調査][OLDH6]のREADME・構造catalogも確認した。同じH6でもDF tol-only rank19／別prefix・q/r/Kの研究であり、本Pauli3候補の価格証拠ではない。
保存event/価格join・次数層別・native構造boundとcompiled上界の区別は参考になるが、旧確率・費用感度・native boundを移植せず、旧解析scriptも実行していない。
新しい数値解析module、sampling/compiler runner、汎用tail framework、承認証明/grant管理機構は追加していない。

| SHA照合した選択member | SHA256 |
|---|---|
| R64 `law.json` | `5f83761e9633d7b854481e6a1f83bdc58f2d82b521c7a78269abdba65e8f715d` |
| H32 `spec.json` | `8f7fdc7427588db16cb1f8ecd5800f2148e1122d1ab1a6923110bac5e0fff364` |
| H64 `spec.json` | `c1df4c492df4f74c5bc9311bf107ea799da687b93f3b1ec01af1a1553bf8f578` |
| R64 confirmation/0000 `result.json` | `a378219d36ab926f309193f708fb6afe709ca8a6992fc6b7b3d7173eda7252a7` |
| H32 confirmation/0000 `result.json` | `78776c6137b1f5a78d16ae3302998e1744c8049c7dc1d0f0c87babc8afe9bdf2` |
| H64 confirmation/0000 `result.json` | `1a044dd2cc294364c8a57176c8616b612c5b1ec61258dd0a240c89f9eb696acb` |

R64 `cases` archive SHA256は`94d2330087b81fd727ccd597ab42faf3111a01296dbb3155a0243d07e8ec7203`、
Hybrid `H6` archiveは`b2a2b92bb834530dce4aeb66ef06babd5b290ecf871d8a380dd2e317c7e701bf`。
R64確認履歴archiveは`d93449ea8bd2d325686e514acc714c0da6ba29a02717751c168ba08198585b3c`。
member名・part・encodingの正本は各固定manifestにある。科学的来歴のためのhashであり、ユーザー承認の認証記録ではない。

GitHub connectorからA調査書、Hybrid REPORT/price_statistics、RTE報告、既存証拠調査書、局所STOPを固定refで実取得し、返却blob SHAとlocal固定Git blobの一致を確認した。
元SOURCE/raw/input/result/grant/ledger/marker/STOP、P1/P2/P3・H6追加確認・selected implementationの旧証拠は変更しない。
独立worktreeだけに本書を追加し、Aのbranch/worktreeや元branchへ書き込まない。共有設定・library・他利用者のfile/processも変更していない。sub-agentは使用していない。
科学値の再解析や新testsは0であり、文書整合性・保存schema・hash確認を科学的tail検証のPASSと数えない。
commit/push後にremote SHAと本書の公開本文を照合し、最終チャットに結果を報告する。

**mandatory STOP：公開後、新sampling・compile・数値再解析・全law誤差評価・N/G更新・詳細科学レビューへ進まない。**

[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e1c7aae386986f6c1bb9483548671655447f516/docs/research/h6_whole_law_transfer_feasibility_20261102.md
[LOCAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/71432728ee0140fb3c1f8bf30ecf51b2faba86ec/artifacts/h6_selected_implementation/2026-11-02/run01/REPORT.md
[LOCALSTOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/71432728ee0140fb3c1f8bf30ecf51b2faba86ec/artifacts/h6_selected_implementation/2026-11-02/run01/STOP.json
[INVREPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3bab573cf17b9dd8ddc1dca5f1a6a6c935ef6d1/docs/research/h6_pure_rte_hybrid_implementation_evidence_inventory_20261102.md
[INV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3bab573cf17b9dd8ddc1dca5f1a6a6c935ef6d1/artifacts/h6_pure_rte_hybrid_evidence_audit/2026-11-02/INVENTORY.json
[RREPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/docs/pauli_rte_event_extension_20261026.md
[RROWS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/candidate_results.json
[RVERIFY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/verification.json
[RARCH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/data_archives.json
[RNOM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/confirmation_nominations.json
[HREPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/REPORT.md
[HROWS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/candidate_results.json
[HSTATS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/price_statistics.json
[HCOMPARE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/comparison_details.json
[HARCH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/data_archives.json
[HDESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/sampling_design.json
[HPLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/confirmation_plan.json
[HORDERS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/event_orders.json
[HANALYSIS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af891fd45805454c6592bbee5441266f14235ffd/scripts/analyze_h6_hybrid_fairness_extension.py
[P2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9073274daee69c9b69d8db3dc332bf6132c2259/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost/REPORT.md
[OLDH6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/artifacts/resource_applicability/track_a_h6_rare_order_cost_analysis_v1/2026-10-13/README.md
