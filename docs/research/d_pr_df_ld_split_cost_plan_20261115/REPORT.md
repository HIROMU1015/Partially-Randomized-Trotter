# PR-DFのL_D選択：候補横断のモデル費用比較計画

2026-11-15 JST。保存H4 linear 1.00 Å/STO-3G/tol-only DF10。保存資料・SOURCE・metadataの照合と設計のみ。

1. **保存資料だけで比較できる範囲**：LD1とLD2は、それぞれ自身の保存ideal finite-RTE信号と同じQ3精度条件で、保存V0配分への適合が確認済み。LD1には26軸の凍結モデル参照価格がある。LD2の同候補・現loweringの価格は確認範囲で未同定なので、**二分割の費用順位はまだ比較できない**。LD3/5/7/9にも保存descriptor・旧signal候補があるが、現行13roundのconfiguration/Q3/価格表はない。これらを劣位として除外しない。
2. **追加する一作業の推奨**：LD2の**m0/m3/m6、fresh各4 path、X/Y最大24 compile**で、prefix固有のfull-wrapper Cモデルを整える。その26軸参照を凍結した後、LD1の既存26軸を再利用し、候補ごとの2配分・最大4 Q3で二分割の最初の有限比較へ結ぶ一つの段階的pilotを提案する。新しい信号やLD1単価の再計算は不要。今回pilotは実行しない。
3. **その後に見る結果**：各分割が自身の信号で適合した配分の**最小モデル参照総費用**、選ばれたN、Q3の適合/不適合/不確定、round別加重費用と長round外挿への依存を見る。LD2側でtail削減の利点を調べる価値が残れば、descriptorがある隣接LD3を次の候補にする。順位が未検証の長round価格に依存するなら、別L_Dの前にその価格確認を優先する。LD1/LD2のどちらかを代表・最良と固定せず、二点の傾向で他分割を枝刈りしない。

[候補・保存入力・52軸正本](CANDIDATES.json)／[L_D別CSV](CANDIDATES.csv)／[最初の有限比較案](NEXT_FINITE_PLAN.json)／[統合HANDOFF](HANDOFF.md)／[静的照合](CHECKS.json)。既存D2 LD1 branch `codex/d2-pr-df-ld1-m3-acquisition-20261115`、起点fb50f41eを再利用。LD2専用branchの原成果は固定参照だけで結ぶ。

## 1. 今回のL_Dの定義と比較対象

**L_Dは決定論側へ置く二体DF fragmentの本数**であり、one-body correctionは数えない。元保存順のfragment index `0..L_D−1` をheadへ、`L_D..9` をrandom tailへ置く。H4の非零one-body correctionは別blockとしてheadの先頭へ入り、scalarはc+s_Rを総Tで一回処理する。fragmentを今回重み順へ並べ替えたり、headを同じ本数の別fragment集合へ変更したりしない。

固定経路はcarrier155529e2 `h46_new2_native.prepare` → provider c31ef424 `pr2_matched_accuracy_m1_execution._prepare` → `generation_partition` → `prepare_df_partial_s2(..., partition_policy='explicit_ordered_partition')`。`pr2_s0_s1_validation.generation_ranked_fragments` は元index順を返す。一般APIのdefault weight_ranked_prefixを今回の現契約と取り違えない。`df_partial_s2.py:373–423` がone-body別blockとDF head、426–434がtailを構成する。

最初の探索では、S2/Δ=.1/K2、元exact共通κ、h=.1/M12、元H/state/snapshot・fragment/basis順、制御・wrapper/compilerを保持し、**L_Dだけを外側変数**にする。Rはprefix固有λ_Rと既存κ/整数化規則に従う各scheduleの値で、独立したR最適化ではない。将来の幅・公式・κ/Kの選択はこの一回で最適化したことにしない。

`L_D=0`ではDF fragmentのheadは0でもone-body headが残るので、one-bodyもrandom化するFull-H pure RTEとは異なる。一方、現在のsignal contextは `1<=prefix<rank`、価格NativeRuntimeは正のtail λを要求する。0と10は現在のPR adapterの境界定義から要る。特にL_D=10をλ=0のままK2/κ規則へ投入せず、Full PFを別の決定論endpoint参照として保持する。

## 2. L_D別の候補・保存入力・不足

全桁ID、各roundのq/R/physical ID、元canonical/total用ID、信号・Q3・価格locatorはCANDIDATES.jsonに保存した。1と2は既存configurationを読むだけで、新IDやscheduleは生成していない。他分割の未作成IDはnull/CSV空欄。

| L_D | 現行候補・schedule | 同精度で使う保存信号/Q3/N | 現条件の価格・構造・モデル | 配分を選び直すための不足 |
|---:|---|---|---|---|
| 0 | 現signal adapterのdomain外。one-body headは残る | 現行13round/Q3未同定。Full-H pure RTEを転用しない | 現価格表未同定 | 境界の候補定義、λ/scalar出典、schedule/信号/価格/Q3 |
| 1 | config **a4dda8f7…**、S2/.1/K2。A a9abf8de CONFIGURATION_LD1.rounds | 13round信号、a82b0f0e Q3でV0適合。allocation PR_DF_LD1_TOTAL_Q3_SAVED_CONTROL_01 | m0/m3/m6較正、m4独立holdoutの直接C/S。E_I[S]26軸、85dab58dのC_ref26軸を再利用。長roundは未検証外挿 | 新しいNだけを自身のsignal/Q3へ照合。価格再計算・再fit不要 |
| 2 | config **1fa33556…**、S2/.1/K2。A同CONFIGURATION_LD2.rounds | 13round信号、同Q3でV0適合。allocation PR_DF_LD2_TOTAL_Q3_SAVED_CONTROL_01 | 同候補C/S/E[S]/評価済みモデルは26軸で未同定。m12旧T2はsampling前STOP | prefix2固有の参照価格。その後自身のN/Q3を選ぶ |
| 3 | 保存λ/scalar descriptorあり。現在のschema/APIで構成可能、現configuration/scheduleは未作成 | 旧signal候補あり、T=.8/1.2/1.6/2のみの旧grid。現13round/Q3/Nは未同定 | 旧prefix3価格あり。例0469d792…はT2/q8/R32/K4/4thで別候補。現単価へ不採用 | 現行configuration/qR、13信号とtotal view、prefix固有価格、独自配分Q3 |
| 4 | 低水準prefix構成は可能だが、保存contextの対応λ/scalar descriptorなし | 現13round/Q3未同定 | 同条件価格/モデル未同定 | descriptorとprovenanceを新しく整えてから候補を定義する |
| 5 | descriptorあり。現configuration/schedule未作成 | 旧signal候補あり、現13round/Q3/N未同定 | 旧prefix5価格あり。例05777e93…はT1.2/q4/R32/K4/4thで別条件 | LD3と同じ種類の不足 |
| 6 | 低水準APIあり、保存descriptorなし | 現13round/Q3未同定 | 同条件価格/モデル未同定 | LD4と同じ定義入力の不足 |
| 7 | descriptorあり。現configuration/schedule未作成 | 旧signal候補あり、現13round/Q3/N未同定 | 確認した旧価格5台帳でPR-prefix7価格は未同定。B0I p7はtailを捨てる別方式 | configuration/qR、13信号、固有価格、独自配分Q3 |
| 8 | 低水準APIあり、保存descriptorなし | 現13round/Q3未同定 | 同条件価格/モデル未同定 | LD4と同じ定義入力の不足 |
| 9 | descriptorあり。現configuration/schedule未作成 | 旧signal候補あり、現13round/Q3/N未同定 | 確認した旧価格5台帳でPR-prefix9価格は未同定 | LD7と同じ種類の不足 |
| 10 | PRの空tail境界は要定義。Full PF endpoint参照config **11cf8ca6…**は別に保存済み | Full PF S2/要求16h/7にはb4296b11の適合配分がある。PR/.1/K2のNへ読み替えない | Full PFの同候補156軸価格体系からの条件付き参照。新PR価格モデルではない | PRの空tail対応を作るか、Full PF endpointとして比較するかを明示する |

LD1/LD2の保存V0は、X/Y各軸でm0..12順に **[64,60,56,52,47,43,38,34,30,25,21,17,12]**。同じNで両方適合した事実は再利用できるが、同じNを今後も強制しない。両候補のqは保存scheduleを参照し、R列は次のとおり。

```text
LD1 R = [2,4,8,16,32,64,128,255,510,1020,2040,4080,8160]
LD2 R = [1,2,3,5,10,20,40,80,160,320,640,1280,2559]
```

A a9abf8deの旧menuは既にLD3/5/7/9のdescriptorを保留候補、4/6/8を保存descriptorなしと区別していた。SOURCE_INDEX7a8bc7f1が指定するH4_contextのhash・prefix keyだけを読取照合し、その数値配列/descriptor本文を再生成・新公開していない。

旧carrier155529e2の公開価格5台帳とsignal_recordsを**metadataだけ**で照合した。H4/PRの旧価格はprefix1/2/3/5に、旧信号は1/2/3/5/7/9にあった。旧信号はT=.8,1.2,1.6,2.0の散在gridで、現13roundの代わりにはならない。件数はcatalog record数であり、独立path数ではない。旧価格のT/q/R/K/公式と旧ZZ lowering、wrapper/provenanceを保ち、同名prefixだけで新価格へ移さない。LD2の旧134088ad…はq4/R4、現m3はq8/R5。公開追補d8f04ce2のreceiptも引き続き不採用である。旧rank12のLD3/6等のcost検証や別snapshotの結果も現DF10の数値入力にはしない。

## 3. 推奨する共通価格規則

**候補固有のoffset＋affine full-wrapper参照**を一案として推奨する。

```text
C_ref(L,m,a) = B_L(m,a) + θ0[L,a] + θq[L,a] q[m] + θE[L,a] q[m]R[L,m]
T_ref(L,N)   = sum_(m,a) N[m,a] C_ref(L,m,a).
```

- **LD1**：B_1は保存E_I[S]、θは3cac53daの凍結残差モデル。実際に使う単価は**85dab58dに保存された26個のexact C_refをそのまま再利用**する。構造項/残差の再加算、m4観測への差替え、単価再予測・再fitをしない。
- **LD2**：B_2を**モデルのoffsetとして0**と定義し、θをLD2自身のfull Cのcell平均へfitする。これはE[S]=0又はheadが無料という意味ではない。LD2の構造期待値を新たに求める前に、同じcompiled単位で探索用参照を作れる。実測Sは同path診断として保存しても、fit済みの最終Cへもう一度加えない。
- **将来のL_D**：同条件の完成した固定参照があれば先に再利用。欠測分は候補自身のC又は、根拠がある場合だけ構造offset＋残差を較正する。係数をprefixごと・軸ごとに持ち、LD1から転用しない。モデルの基底とoffsetの選択は観測前に固定する。

単位はP=0の**測定付きfull controlled Hadamard wrapperのcompiled RZ/shot**、symmetric-directional、c+s_R総T一回、newZZ、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17で共通。X/Yは同pathの別軸で、path数を倍にしない。normalization B、価格標本数、event数を最終C_refへ再乗算しない。

**headとtailの反映**：LD1では構造offsetがhead/tail・basis共有・literal相殺を含み、θがそのcompiled残差を近似する。LD2ではprefix2のhead sweepとouter境界の平均費用がθqに、選択されるZ/ZZ・basis・order/product/phaseなどtail内容の平均的影響がθEと切片/outer項に含まれる。qRは負荷指標であり、独立eventを別compileして足す厳密加法則ではない。θqを純粋なhead費、θEを普遍的event費と同定したと主張せず、prefix固有の全wrapper観測で境界相殺も一緒に較正する。

共通なのは8-modeのgate構成部品、wrapper/newZZ/compilerと規則の適用方針である。headのfragment数/基底/対角角度、残るtailのcomponent分布・basis遷移、λ/τ/order分布、R列、c+s_R、head-tail/outer境界はprefix依存する。そのため同じcoefficientの転用は採らない。LD1の構造offset付き参照とLD2のfull C参照の根拠の非対称性も結果に併記する。

### この規則で比較できる主張

これは**探索用モデル参照費用**であり、同じlawの厳密E[C]と混同しない。LD1のSは保存I、較正残差/LD2実価格は元normal sampler Qの有限cohortに由来する。I/Qを一致と仮定せず、正式E_I[C]、CI、全path保証は未認定のまま残すが、それらを今回の探索用価格規則の採用条件にはしない。F0やm9-X下界の再証明を依存作業に戻さない。

同条件実測はcalibrationと有限cohort診断に使い、他prefixや長roundの母平均へ直接リラベルしない。LD1 m0/m3/m6の較正、m4独立holdout、P1/replay参照は元役割を保持する。LD2には独立holdoutも長round観測もまだない。参照値が全26軸で有限・正、identity/profileが一致、fitが定義できることを逆価格配分の技術条件とする。負/零/非有限の予測をclipして進めず、その場合は部分成果を保存して停止する。全path保証や任意の2%・他方式の許容差を追加しない。

## 4. 最初の有限比較：LD1保存＋LD2三点価格pilot

目的は**LD2を加えると、各分割が自身の配分を選ぶモデル費用比較を初めて行えるか**である。LD3等を先行するとconfiguration/schedule・13信号・Q3も必要になる一方、LD2は信号とV0適合が既にある。取得の少なさを理由に選び、低費用を予想してLD2を選んでいない。

| 段階 | 保存再利用／新規対象 | 後続案の上限 |
|---|---|---:|
| A：LD2価格取得 | m0 ce7ef02e… q1/R1/T.1、m3 334b1fd3… q8/R5/T.8、m6 fe234d98… q64/R40/T6.4。各normal fresh4 path・X/Y。入力H/state/snapshot・schedule・元sampler・native/新ZZ/compiled pipelineを再利用 | 12新path、24新compile。各cell8件 |
| B：LD2参照の凍結 | 各軸3 cellのexact算術平均へ(1,q,qR)をfit、B_2=0。LD1の26単価をcopy再利用 | 6 cell-axis平均、2 fit、LD2 26参照価格。LD1 fit/単価再計算0 |
| C：候補別配分・Q3・有限比較 | 下記A_SHAPE/B_PRICE各1本をLD1/LD2で別々に生成。自身の保存signalを用いる。保存V0 Q3は2件再利用 | 最大4新vector・4新Q3。比較表は基準2＋新4の最大6行。新しい報告用総額は最大5（LD1旧V0総額は再利用） |

**追加信号0、LD1追加価格0、全round compile sweep0。** これは後続の一つの段階的有限pilotの上限で、今回の実行件数は全て0。新seed列や配分も今回は生成していない。後続は実装・入力・cohort/seed・出力先とSOURCEを科学開始前に固定する。

4 path/cellは既存LD1 P2/m3の有限pilot規模を引き継ぐ上限であり、母平均精度や信頼水準から導いたnではない。同じpathのX/Yを個別receiptへ結び、event digest、cohort/index/seed、S/C、fingerprint、metrics、lowering/scalar/preparation、実行SOURCEと結果公開commitを別々に保存する。B0/W・Pauli・旧prefix価格を較正へ足さない。実際に観測したorder2だけを記録し、未観測ならそのまま残す。

### 実装入口の具体的な差分

prefix別の取得にはe8273676の **ld12_price_requests.request_for → NativeRuntime.context/sample → Adapter.acquire** を優先する。contextはrequestのprefixとsplitでprepare/cacheし、prefix2のhead/tail/λ、corrected c+s_Rを照合したcontextを、元fast価格backendへ渡す。旧C1 `Backend.context('H4')` はprefix1固定なので、LD2 callerからそのまま呼ばない。m3 runnerのPLANだけLD2へ書き換える案は採らない。 さらにAdapterは `fast.make_backend_class(...,axes=...,observer=...)` を呼ぶ。Dの26d4f23e/e8273676にある対応版（blob44b0f8cc）を揃える必要があり、現在LD1 checkoutの旧版（blob6c3d7be5、引数base/scienceのみ）へそのまま接続できない。現在のSOURCEを変更したのではなく、後続の固定実装で整える差分である。元T2/m12 STOPは再開せず、今回はm0/m3/m6だけを選ぶ新しい有限caller/入力を固定する。

入力・preparation・request・lowering・compiler・fingerprint検査はP2 fresh/m3と同じ範囲を保ち、元経路で未実施の数値作用gate/full-wrapper statevectorをPASSとしない。実行前の技術検証は人工入力/fake compilerで実際のprefix2入口、保存結果再利用、axis/receipt対応、総scalarを確認する。保存H4演算子の数値再検査suiteは使わない。観測は協調的stage logと親から行い、Python3.11.0rc1で既知不具合のある定期faulthandlerを流用しない。独立pathの並列化と既存fast writer/fingerprintを使え、過去CPU/RAM上限を恒常的制限にしない。

### 三点だけで長roundへ適用する際の明示的近似

LD2では**同じprefix・S2/.1/K2・共通κ schedule上で、full-wrapperの平均的価格を固定(1,q,qR)で近似する**。m0/m3/m6で三係数を決めても、m7..m12の角度・order分布・basis共有・compiler相殺の変化は検証できない。特にm7..m12はq64/R40を超える外挿である。三点fitの残差0もモデルの確認にはならない。

初回はこの近似付きモデル比較を許す設計とし、**LD2の独立価格holdoutは今回提案の取得枠に含めない**。旧Cの四cell案のm4 holdoutは未採択の後続候補として保持する。LD1 m4は既存独立holdoutのままで、LD2のvalidationを代用しない。モデル合否の数値許容差は設けず、NOT_ASSESSED_NO_ADOPTED_TOLERANCEを残す。順位が長round外挿に依存すると判明した場合に、そのroundの別有限確認を次の判断にする。

## 5. 各分割が別々にNを選ぶ規則

既存Bの06f4210e `run_pf6_cost_budget_q3_fixed.py:70–104`の有限breakpoint法を再利用する。モデルは観測前に固定し、候補Lごとに保存V0の参照総額T0_Lと、**自身のT0_Lの3/4**という一つの試行予算を用いる。PF金額はこの予算を決める条件にしない。

```text
i=(m,a), cap_i=V0[m], beta=3/4
w_i = cap_i              (A_SHAPE)
w_i = cap_i/C_ref[L,i]   (B_PRICE)
N_i(lambda)=max(1,min(cap_i,floor(lambda*w_i)))
候補自身の sum_i N_i(lambda) C_ref[L,i] <= beta*T0_L を満たす最大breakpointを選ぶ。
```

軸ごとの価格を使い、X/YやLD1/LD2に同じvectorを強制しない。今回初回menuでは各NはV0以下・1以上に限る。余剰予算の使い切り、結果を見たλの変更、別βや追加vectorを自動で試さない。元V0、A、Bの3点/LDの範囲であり、V0を超えるshot振替や大域最適配分を探索しない。

**必要な技術変更**：既存generatorはpricesを整数に限定する。LD1単価はexact有理数なので、その関数の価格parse/重み/積和を正のFractionへ対応させる小変更が必要。元単価を整数RZへ丸めない。人工の有理価格・breakpoint端点・欠測/零価格・再利用経路で確認し、decoder/DPの数式は変えない。価格側は `price_model_validation._solve/predict` のexact算術を再利用する。同moduleの汎用fitはrecords間のlaw辞書の完全一致を要求する。m0/m3/m6の実際のτ/order等のpoint lawを同一と偽らず、各cellの元law receiptを保持したまま、prefix2固定scheduleのモデルfamilyを宣言する薄いfit wrapperを用意する。既存domain guardを緩めない。toleranceを必須とするvalidateへ仮の値を渡さない。

精度はa82b0f0eの採択Z1＋T1 total decoder、保存DF10 solver energy proxy −2.1663874486347634 Ha、全事象平方loss、εE=0.00159360010199040 Haのexact二乗閾値、EXACT_AFFINE_OF_BINARY64_RAW、shots/axes/round間IIDを固定する。各分割自身のlaw・normalizationを持つsignal packetを使う。新Nではcount/closure preflightをやり直し、旧Nの閉包PASSを流用しない。riskへAのδE²や価格残差を足さず、normalizationを価格へ乗算しない。

保存V0と一致するvectorや同一候補の同等Q3があれば再利用してcallを減らす。新規Q3は各unique candidate/vector一回、最大4。適合のみから参照総費用が最小の行を候補別に選び、同額・不適合・UNKNOWN/INCONCLUSIVEも残す。不確定を消して全menu最小と表示しない。新配分でMSEが必ず単調改善するとは仮定しない。

## 6. 次のL_Dを選ぶための読み方

| 有限比較で見るもの | 次の研究判断 |
|---|---|
| 各Lの最小適合T_refと選ばれたN、A/B/V0のQ3状態 | LD1代表の固定比較から、二つのLが自身のNを選ぶ有限比較へ進んだと言える。どちらが低いかを今回は予測しない |
| round別N×C_ref、較正/補間/外挿のrole、prefix別モデルの根拠 | 差がm7..m12の未検証価格に依存するなら、まずLD2側の該当長roundの有限確認を別計画にする |
| LD2を増やした際のhead/outer項とqR項の釣合い、候補固有のQ3応答 | 次の隣接split **LD3** は保存descriptorがあり、one-fragment追加の比較を設計しやすい。必要ならconfiguration→13信号→価格→独自N/Q3の有限案を作る |
| LD1が安い、又はLD2が安いという二点の結果 | その方向がLD3/5/7/9でも続くとは限らない。未評価Lを棄却せず、次の候補選択とモデル検証の優先度に使う |

厳密な母期待値・CIがなくても、この有限menuのモデル参照費用を比較して次の取得対象を選べる。一方、有限pathのばらつき、I/Q差、未観測event、長round移送、モデル族の非対称性は結果の解釈に残す。探索用の順位を、全L_D又は各方式を最適化した際の保証付きwinnerへ昇格しない。

方式間の参考値は最新b4296b11の **Full PF 615,304,502 RZ、B0I p7 435,481,090 RZ**。これは現6幅×N0/A/Bという方式ごと9点の有限集合内最小適合費用で、96直接＋60条件付き閉包価格を引き継ぐ。今回これらとの差額・比・新ランキングは算出していない。旧85dab58d/fb50f41eが参照した593,555,352/838,693,580 RZや旧STOPを上書きしない。

## 7. §10.5・成果物・STOP

前回LD2照合8dc41b35/公開追補d8f04ce2、LD1下界fb50f41e、A旧二prefix menu、C旧四cell案を再利用した。関連origin refをfetchし、既知job名・出力先・関連Git履歴を確認した。今回のPR-DF候補横断価格/配分比較の同等完了物・同名jobは確認範囲で未同定。新しいBのPF予算/同shot幅比較、B2のPauli再配分、AのPauli S0+、CのFull-H価格案は別scopeである。他担当の進行を凍結せず、その結果も待たない。

次回実行前は、この計画公開後に増えた**同LD2 physical/axis/law/cohort/seed/lowering/wrapper/compilerのreceipt・fit/26軸表、同NのQ3、進行job、出力/cache、実行SOURCE/入力差分**だけを追加照合する。同等結果は出所付きで再利用し新規件数を減らす。取得済みか解決できない項目は保留し、旧T2のunused枠や旧STOPを再利用しない。raw台帳のrecord数を新規path数と同一視しない。

後続pilotで入力/SOURCE/scalar/axis等の不一致、取得・保存失敗、固定cell欠測、rank失敗、非正/非有限参照があれば新規投入を停止し部分保存。retry、seed差替え、追加path、観測後のfeature変更・clip・再fitで補わない。Q3不適合/不確定は有限結果として保存し、追加vectorの許可としない。終了後は公開照合でSTOP。LD3/他L、幅/κ/K変更、独立価格holdout、正式E[C]/CI/Gは別判断である。

ロードマップ差分：目的を「LD1対PFを保証付きで決着」から、**同精度Q3モデル下でL_Dごとの参照費用とNを選ぶ有限探索**へ広げる。保存価格・信号を持つLD1/LD2を初手に、descriptor-readyな3/5/7/9と定義入力不足の分割を区別する。全support下界や厳密期待価格は初回pilotの待ち条件ではない。ロードマップ本体は統合担当が更新する。

今回の検査はmetadataのID/locator/公開catalog条件、LD1参照26値の元JSONとの一致、SOURCE上のprefix/domain/呼出し経路、文書/JSON/CSVの整合に限定した。**sampling・信号生成・build/compile・fit・価格予測・shot配分生成・Q3・総額再計算・再ランキングは全て0**。科学module・保存H4演算子回帰suiteも起動しない。原結果、凍結価格、m4独立holdout、m9下界UNKNOWN、LD2原receipt/T2 STOPを保全し、commit/push・remote一致確認後 **mandatory STOP**。有限案へは進まない。
