# 幅×shot配分探索：保存compiled価格のcoverage

2026-11-15 JST、Codex D（主agentのみ）。**別幅12候補のうち、保存価格だけで全13round・X/Yを加重できる候補は0。全候補に部分的な直接証拠があり、312候補軸中100軸への対応を確認した。残る212軸は未取得・未検証。別幅へ採択済み閉包を移送した行は0。**

新規価格予測、private QPYの再監査、RZ再計数、build/compile、fit、Q3/N/Gはすべて0。現行156軸はB2の固定表を参照し、再集計・再監査していない。成果は[B向けHANDOFF](HANDOFF_B.md)、[機械可読表](PRICE_COVERAGE.json)／[CSV](PRICE_COVERAGE.csv)、[最初の有限案](NEXT_FINITE_PLAN.md)。計画公開で**mandatory STOP**。

## 最初の3問への答え

1. **保存価格だけで任意の候補shot配分を加重できる別幅：なし。** 現行6候補には固定の直接96＋条件付き導出60があるが、別幅の全26軸は揃わない。同じ物理schedule・axis・wrapper/compilerの1-shot価格はshot配分に依存しないので、配分ごとの再compileは不要。ただし、旧閉包の採択文言は現行6候補・元固定配分の条件付き比較であり、新配分での評価・主張にはB/Cがその前提と用途を明示する必要がある。旧Gや順位を移送しない。
2. **一部roundの不足を埋めれば結合できる別幅：12候補すべてに部分価格がある。** 価格欠測が最少なのは両routeのS2 2h・Y4 4h・Y4 32h/5（各m0〜m4の10軸あり、m5〜m12の16軸不足）。ただし、これらには完成した全13round信号packetもない。**現段階で信号側が既に13round揃う別幅はB0I p7/S2 1hだけ**で、価格はm0/m3/m4の6軸、20軸不足。価格不足だけを埋めれば新meanなしにBの候補となるが、Q3成立・十分shot数は未判定。
3. **新信号も価格も要るため後回しにできるもの：完成packetのない11候補。** 特に粗いS2 8h/3・Y4 32h/5（各route）は、BのQ3可否が出るまで長roundの価格を先行取得しない。細かいFull S2 1h、Full/p7 M5 8hなども、より小さいPF誤差と増えるstep費用のtradeoffをBが探索対象に残す場合だけ進める。「後回し」は劣位の認定ではない。T2のenergy shift、構造piece数、未検証価格式をcompiled価格の上下界やQ3の代用にしない。

最初の有限案は、**Bが候補に残した場合のB0I p7/S2 1h・m1/m2・X/Y、最大4 compile**。既存6軸を保全・再利用し、q2/q4の直接点を加える。成功しても10/26軸までで、長roundや新モデルを認定しない。これなら信号の新規取得を待たずに技術準備でき、巨大回路を前提にしない。[対象ID・判定・STOP](NEXT_FINITE_PLAN.md)。

## 現行156軸は固定参照

正本：`1002f1dfea34de439ca1b704a0192bca473c5387:docs/research/pf6_allround_cost_20261115/INPUTS.json#axes`。直接96軸、条件付き60軸の区別、各元receipt、original/total configuration・candidateの対応をそのまま使う。`AXIS_TERMS.json`の価格出所も参照可能だが、今回はN×Cや総額を計算しない。

条件付き価格の正本は `e4458cd7548977e6e11a5a7b0c8dca0d049535d4:docs/research/pf_closure_prices_20261115/run01/PRICES.json#rows`。`PF_FIXED_COMPILER_CLOSURE_CONDITIONAL_60_AXES_V1`、`MODEL_PREDICTION`、非直接観測、元仮定、誤差上界/CIなしを維持する。元12モデルの資格flagを変更しない。現行signal6候補の完成はA `1adb7e65a9e7c3adc32491d853f31d124393ead7:docs/research/pf_shortlist_mean_completion_20261115/PACKET_INDEX.json`へ参照する。

今回の表では現行156行を複写せず、`current_156_reference`に固定locatorと6 configuration IDだけを収めた。現行の採択済み係数や60価格の再評価は0。

## 調査した幅と不足round

幅はh=.1（**元binary64 hのexact有理数**）に対する比。旧T2 `5fb2c32e39e63267ea1748861269c362a93a1b86:docs/research/truncated_pf_optimization_plan_20261114/PLAN_METADATA.json` のFull10/p7を使用した。S2は1h/2h/**16h/7**/8h/3、Y4は4h/**16h/3**/32h/5、M5は8h/**16h**。太字が現行。M5について旧有限ladderには現行より粗い幅がなく、今回新たな幅を発明していない。探索空間の永久制限でもない。

| route | 別幅 | 現行との関係 | 直接証拠があるround（X/Y各別） | 軸数 | 価格不足round（X/Y各別） |
|---|---|---|---|---:|---|
| Full / S2 | 1h | 細 | m0,m3,m4 | 6 | m1,m2,m5〜m12（20軸） |
| p7 / S2 | 1h | 細 | m0,m3,m4 | 6 | 同上 |
| Full / S2 | 2h | 細 | m0〜m4 | 10 | m5〜m12（16軸） |
| p7 / S2 | 2h | 細 | m0〜m4 | 10 | 同上 |
| Full / S2 | 8h/3 | 粗 | m0〜m2 | 6 | m3〜m12（20軸） |
| p7 / S2 | 8h/3 | 粗 | m0〜m2 | 6 | 同上 |
| Full / Y4 | 4h | 細 | m0〜m4 | 10 | m5〜m12（16軸） |
| p7 / Y4 | 4h | 細 | m0〜m4 | 10 | 同上 |
| Full / Y4 | 32h/5 | 粗 | m0〜m4 | 10 | m5〜m12（16軸） |
| p7 / Y4 | 32h/5 | 粗 | m0〜m4 | 10 | 同上 |
| Full / M5 | 8h | 細 | m0〜m3 | 8 | m4〜m12（18軸） |
| p7 / M5 | 8h | 細 | m0〜m3 | 8 | 同上 |

100は**候補scheduleへの軸別対応数**で、重複しない実測100件ではない。uniqueなphysical ID×axisは68件。違う要求幅でもearly roundのT/q/actual幅が同一なら元価格を共用できる。例えばY4 32h/5のm4はactual幅16h/3で現行m4と同じ直接点だが、m5はactual幅32h/5へ変わる。このm4の共用は別幅への価格モデル移送ではない。

表の「直接証拠あり」は同物理量の元trial・SOURCE・axisの所在確認であり、**新しいconfigurationのC guardを実行した意味ではない**。JSONの`consumer_binding`で、同物理pointの新metadata bindingが必要なもの、旧receiptから新sidecar/C guardが必要なものを区別した。元SOURCE/cohortを付け替えず、consumer受入後に利用する。保存価格の再取得は不要で、受入不明の項目を新compileで潰さない。

分類は312軸中、`DIRECT_SAME_PHYSICAL_EVIDENCE`100、`MISSING_VALIDATION_OR_ACQUISITION`212。採択モデルの新適用0、元trial未回収の`SAVED_LOCATOR_BINDING_PENDING`0。後者0は既知一致の元記録を解決できた範囲であり、未列挙archiveが存在しないという主張ではない。Full M5 8h/m4等、catalogに物理候補の行はあってもpriceが欠測なら「回路が保存済み」とは数えない。

## 根拠・数量の整合

価格quantityは保存linear H4 / 1Å / STO-3G / tol-only DF10、P0除外、Full一体項＋fragment0..9又はB0I p7一体項＋fragment0..6の測定付きfull controlled Hadamard wrapper1 shotのcompiled全RZ数。元fragment順・completion・basis、signed S2/Y4/M5 piece列、Full c_H / p7 c_H＋tail identityをtotal Tに一回、symmetric_directional制御を保持する。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、coupling mapなし。H/state/snapshot/profile/scalarの根拠は表の6 `contexts`へ共有した。

metadata joinではT/q/actual幅をbinary64 hexで比較し、元IDもscheduleから再確認する。元T2 roundのqとexact actual比を読み、保存CELL_SCOPEのsigned weightsと元nativeの演算順 `T*weight/q` を使った1 outer-stepの時間metadataを保存した。反復はq回というdescriptorで持ち、巨大gate列を構築しない。scalar位相や回路作用は再計算しない。qだけ、要求幅の表示だけ、compiler hashだけでは受入しない。

根拠は次の順で再利用した。

- **現在の同一physical点**：B2固定入力の軸別元receipt。新configurationへ結合するためのmetadata参照で、現在の156軸を再認定しない。
- **T=.8の追加2 cell（Full/p7 S2 1h・m3）**：公開PF commonの `native_results.json.gz` の元trial（Full index131、p7 index45）、保存result SHA、H/state/prep/scalar、軸別compiler・fingerprint・既存gateを照合した。旧SOURCE監査はe7dca5f73の`receipt_source_checks`を継承する。
- **T=1.6の追加6 cell**：Full/p7 S2 1h・S2 2h・Y4 4hの旧long-time `EXPLORATION`。固定公開carrierのexact physical metadataと既知の元result JSON6件を照合した。JSONだけを読み、QPY/cache circuit/statevector/NPZは開いていない。元7 gate、total scalar、prep hash、入力binding、元native file SHAが現基準と一致。元JSONのpath・SHAを固定公開carrierと一緒に[RECEIPT_CHECKS](RECEIPT_CHECKS.json)へ記録した。axis別価格は元保存値を転記しただけで、RZを数え直していない。
- **検索scope**：long-time5 carrier（746 record）、Full catalog136候補、現行B固定表、Aの既存710保存mean index。旧PF common、M5、pilotの同一quantity確認は既存coverageとSOURCE監査を継承し、網羅的監査を繰り返していない。他rank・別control・別wrapper・旧RTE/Pauli/JW価格は参照quantityとして除外する。

[SOURCE_INDEX.json](SOURCE_INDEX.json) に固定SOURCE/path、[表](PRICE_COVERAGE.json)に各receipt selector・元trial・適用条件を保存した。旧private JSONは読取専用、raw/STOP/凍結JSONを変更しない。新しい重要科学レビュー、QPYの独立再監査とは呼ばない。

## 別幅へ閉包を使うために残る根拠

旧監査606881b04と採択c3f761f0の根拠は、**現行の固定実幅**でのbit同一piece列、compiled周期本体と端効果、最終ancilla run、障壁・局所passの持続を用いるq方向の延長である。Full/p7・公式・axisごとに独立の元係数を使っている。

新しい実幅では角度が変わり、1q最適化、相殺、障壁、周期本体、端、終端の数値枝が変わり得る。元係数a/bの流用、公式piece数比例、m6/m9の2%を新領域の誤差上界にすることはできない。installed Rustと参照sourceの対応・規模依存分岐等の旧条件も残る。今回は新幅での終端角余裕や閉包を計算しない。

将来別幅のq延長を採用する場合は、候補のscalar・piece列・compiler/sourceを固定したうえで、その幅のcompiled本体/端/終端の証拠と誤差の扱いをCが別に確認・採択する必要がある。既存q点にQPYが保存されていれば再利用監査を先に提案できるが、それは新しい科学監査なので別承認。全round直接compileを既定にせず、根拠が揃わない領域はUNKNOWNとする。

## 信号coverageと検査

Aの710保存record index関数をSOURCEから再利用し、元meanの数値一致やpower・biasを再計算せず、物理ID・gate・locatorだけを索引化した。現行6packetの同物理roundも参照できる。別configurationの全packetへ自動コピーしない。

| 別幅（両route。ただし注記あり） | 現行SOURCEの保存meanがあるround | 旧meanのみ・資格待ち | 新meanが必要 |
|---|---|---|---|
| S2 2h | m0〜m3,m12 | m4 | m5〜m11 |
| S2 8h/3 | m0〜m3,m12 | なし | m4〜m11 |
| Y4 4h / p7 | m0〜m3,m12 | m4 | m5〜m11 |
| Y4 4h / Full | m0〜m3,m12 | m4,m6 | m5,m7〜m11 |
| Y4 32h/5 | m0〜m5,m12 | なし | m6〜m11 |
| M5 8h | m0〜m3,m12 | m4 | m5〜m11 |
| S2 1h / Full | m0,m12 | m3,m4,m6 | m1,m2,m5,m7〜m11 |
| S2 1h / p7 | 完成packet13round | 旧値の再資格化不要（元packet資格維持） | なし |

この表はmeanの存在・元診断であり、Q3のPASS、十分N、完成packetの新認定ではない。T2のstep/eigenspectrumがあることも全round mean完成を意味しない。全SOURCEと残るroundはJSONの`signal`に収めた。

`python3 -B docs/research/d_pf_width_price_coverage_20261115/check_metadata.py`：4 metadata/人工差分check PASS（[TESTS](TESTS.txt)）。18幅configから現行6を除いた12候補、312軸、各source selectorとactual幅、ID別名の同等性とq-only/1ULP幅差の拒否、X/Y元locatorの分離、model未昇格、4件有限案のscopeを確認した。元研究pipeline・libraryへ変更0。pure stdlib/Git読取・小JSON joinなので科学worker並列化は不要、既存710 index関数とGit reader cacheを再利用した。資源上限や共有設定を変更していない。

ロードマップ差分はHANDOFFへ渡し、本体を編集しない。全候補の費用順位・最適幅・Q3配分は未計算。commit/push・remote一致確認後**mandatory STOP**。
