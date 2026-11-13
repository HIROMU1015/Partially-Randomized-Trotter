# 後続の有限価格取得案（未実行・未承認）

2026-11-13 JST。[評価方式・資格条件](LONG_TIME_PRICE_EVALUATION_PLAN.md)／[HANDOFF](HANDOFF.md)。今回許可された設計作業は科学実行を含まない。以下は**選択可能な有限案**で、科学scopeの採択、実行許可、価格精度の認定ではない。チャットで選ばれた有限範囲の明示承認で足り、新grant・承認証明書は作らない。

## まず決められる技術pilotと、まだ決められない平均取得

| 案 | 対象candidate / round / axis | path数と最大新compile数 | 得るもの・限界 |
|---|---|---|---|
| T2（初回推奨候補） | PR-LD2 / m12 / X,Y | normal fresh 1 path、最大2 compile | 最大qとLD2の実preparation/native/hash/compile実行可能性、軸別path S/C/D・stage/RSS。平均の分散/CIは得ない |
| T1（代替） | PR-LD1 / m12 / X,Y | normal fresh 1 path、最大2 compile | 旧LD1 m6から最大使用domainへの技術challenge。m12負荷UNKNOWN。LD2の確認にはならない |
| T12（両prefixの技術確認が必要な場合だけ） | 上記二点 | 各1 path、合計最大4 compile | prefix別domainを直接確認。T2成功を理由にT1へ自動追加する許可ではない |
| M（正式比較に必要なmean取得） | Bが必要性を示す有限セル集合J | n_jを後で明示固定。最大Σ_j \|axes_j\| n_j compile | target-law mean又はfinite-cohort診断。現在はJ/n/精度/confidence未定で実行不能 |

T2を先に挙げる理由は、同じ最大qで固定metadataのevent数がLD1より少なく、新prefixの接続も問えるためである。価格/RAM/timeが必ず小さい保証や、正式G上の優先順位ではない。LD1への移送証明もない。Nの供給を待たず技術pilotを選ぶことは可能だが、ユーザーはその情報のための巨大compile負担を選ぶ必要がある。T1/T2/T12のどれも現在は未採択。

最大使用点を安全に投入できる根拠がなければ、**未実行STOP**も正当な選択肢である。別案は、target-m12のR/角度を保った短outerの技術challenge又は人工stream改善だが、前者は元m12 candidate価格ではなく別window診断となる。window長・path数・caps・最大compile数を別途具体化して承認を受けるまで開始しない。m11へ無断で縮小したり、m7から順に試したりしない。

## T案で固定するSOURCE・event定義

| 項目 | 固定・取得方法 |
|---|---|
| 物理menu | A `a9abf8de75bf3432fcd714e868e5c14c8884b8f8` のMENU/CONFIGURATION_LD1又はLD2。H4 DF10/S2/ΔPF=.1/K2/h=.1/M12/P0・元exactκ・元H/state/snapshot/fragment順。A91349のsignalを再呼出ししない |
| metadata/adapter | D `748be6875b5007707d864e16aa7a4fc4250f3714` の `request_for/from_b`、候補Runtime/Adapter。SOURCE一覧はREPORT。後続caller・必要な小修正を含む**新実行commitを実行前に固定**。本計画や旧metadata publicationのkeyを科学runへ再ラベルしない |
| compiler/lowering | Qiskit1.3.0/NumPy1.26.4、rz/sx/x/cx、opt1、seed17、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、CONTROLLED_ZZ_CX_CRZ_CX_V1、元symmetric_directional/full measured wrapper/P0 |
| 入力復元 | 元loaderでH/state/snapshot・decompositionを照合。prefix別λ/scalarの実値、preparation/component/basis/tail hash、fragment順をRuntimeで確認。新state solveやDF作成を追加しない |
| cohort | 提案名 `D_LONG_TIME_FEASIBILITY_V1`、選択candidateごとindex0だけ。過去P1/P2/development/replayと別role。将来Mのfresh validation cohortへ混合しない |
| seed規則 | 元carrier `trial_seed`：SHA256(JSON(['H46_LONG_TIME_20261025', cohort, physical_id, index, outer, piece]))の先頭16hexをuint64化。全seedとevent digestを保存。同一数値seedだけでLD1/LD2間の共通乱数法と称しない |
| sampling | 元normal sampler、PCG64/NumPy choice、finite K2、q/R/piece allocation・signed timeのbinary64表現不変。forced rare、再seed、失敗path置換なし |
| path | X/Yは同じevent列で個別にfull wrapper/compile。scalar一回、control/measurement保持。n=1を2標本と数えない |

各compile要求軸とcache hit/missを記録する。cache hitでcompileが減っても未使用budgetを別path/round/retryへ振り替えない。旧pathは同一性の参照又は正確なreplayに限り、fresh全law nを増やさない。科学path原本は新専用directoryへ保存し、公開時に既存private event方針を守ってdigest/参照を残す。

## 実装と費用の測り方

SOURCEを固定する前に、既存runnerから選択requestだけを呼ぶ最小callerを作り、人工入力で範囲/例外保存を確認する。新しい探索controller、N solver、統計認定器、承認管理は作らない。Cの共通統合は科学取得依存graphを持たないため、CのnamespaceからRuntimeを自動起動しない。

```python
# 後続で承認された有限runの接続例。今回は呼び出していない。
req = L.request_for(root, selected_label, 12, axes=('X', 'Y'),
                    execution_source=fixed_new_execution_source)
# 元loaderで復元・照合したhamとreqのbinding/input/executionからRuntimeを作る。
# runtime.context/sample、Adapter.acquireは承認されたindex0だけ。
```

全pipelineの入力復元、prepare/context、sampling、event変換/write/digest、構造S、compose、軸別wrapper/hash/compile/6指標/mutation、保存のwallを分ける。既存`TECHNICAL_TIMING`を再利用し、外側の全path wall・worker peak RSS、stage境界RSS/実測同時peak、event/word数、cache hit、compiler versionを追加観測する。`max_instructions=len(evolution.data)+4`は現在のwrapper整合budgetで、事前のRAM保証ではない。

初回Tのwall/peakはUNKNOWN。m6の881秒やevent倍率から予算時間を認定しない。metadataとnative emitterの静的構造から保持物を点検し、実際の利用可能RAM/storageと自己processの観測に照らし投入判断する。全compiler中間の確かな見積りがなければ「実行可能性を探る試行」であることを残す。真の資源不足時は部分原本を保全してSTOPし、成功pathだけに条件付けた平均を作らない。

独立path/process並列、同prefixのpreparation再利用、選択軸順次解放、元fingerprint/cacheを優先する。CPU/RAM/workerに固定の小上限は置かず、実際のメモリ/I/Oと非干渉で並列度を決める。可逆module差替えのある同processで科学callをthread並列しない。Tの一pathという科学範囲はCPU制限ではない。他者のjob/共有設定に手を触れず、一時的なPSI/競合だけで正常workerをkill/restartしない。

後続費用は、測定対象と同じdomainのpath stage実績が得られた後、各jobの仕事量和と実際の重なり・同時peakを用いて評価する。T2の実測をLD1へ一律換算しない。request不足やsource不一致なら、samplingより前にSTOPする。

## M案：必要セル・n・資格の確定手順

Mを実行可能な有限計画にするには次の入力が必要である。

1. **B/A・ユーザー**：同target/decoder/全事象lossの主張水準、全N又は根拠付きN集合、比較する候補/配分familyとそのSOURCE。nominalやsignal局所gateを十分Nとしない。
2. **B・ユーザー**：比較margin又は許容regret、価格confidenceと同時family、各誤差の予算。価格2%/95%や旧n=8を自動採択しない。
3. **A/C/D**：target P/Qとsignal lawの整合、同candidate E[S]の適用可否・数値enclosure、law/rareとmodelの未知項。Tのpath価格だけでは埋まらない。
4. **D・ユーザー**：使うmean estimator/CIの前提、必要な校正・独立holdoutの区別、固定path上限・最大compile数・停止規則、実測domainの費用。

直接C法と同lawのE[S]+平均D法を、取得費×推定分散・S数値幅・law補正で比較する。small-nのSDだけで将来nを保証しない。たとえば真の範囲幅Wが証明され、IIDの固定n/一セルfailure αとMC半幅eを選ぶ場合、Hoeffdingの設計式は n≥W² log(2/α)/(2e²)。これは**条件付きの式**で、現在W/α/eもnも代入していない。有用な範囲が得られず近似CIを選ぶ場合、その前提と「経験的/未資格」の主張水準を承認範囲に含める。

pathごとX/YのC/S/D、nと標本分散/共分散を保存する。Nが固定された診断ではN_X D_X+N_Y D_Yの共分散項を含める。個別軸の同時boxでBへ渡す方法なら軸独立性は不要。共有model・round/candidate共分散を不明のまま0にしない。

finite-cohort平均と有効CIは別成果物とし、law/systematic/window誤差が囲えないときはqualified=falseで止める。normal pathの全law CIがrareを含むなら二重加算せず、全order0観測だけなら未観測conditional costをUNKNOWNのままにする。

セル集合Jに各n_j paired pathを選ぶならcompile上限は2Σ_j n_j（両軸の場合）。単軸指定はΣ_j |axes_j|n_j。実行前に**具体的な有限整数**へ固定して別途承認を受ける。n不足を見ての無断増分、失敗retry、別round置換はない。familyを追加するなら新scopeと同時被覆予算を設計する。

## 長時間modelを選ぶ場合の追加条件

model採用はTとは別判断。対象prefix・最大q/R・K/law・signed time/capsを含む適用域、feature/係数の同定条件、calibration sourceと独立holdout family、tail/discrepancyの許容幅を先に定義する。P1/P2やTを見て選んだmodelの検証に同じpathを再利用しない。

必要なら使用長時間域（例：実際に使用するm12）を直接holdoutに選び、m7→m8→m9の順次compileを既定にしない。prefix二種は別domainである。m12の検証がm10/11の誤差も囲むという証明がなければ、それらが正式比較に必要になった際、追加参照又は別の有効envelopeが必要となる。window長・各drawのprefix数・axes・direct holdoutを列挙し、最大compile数をΣ_draw(|window_lengths|×|axes|)+Σ_direct |axes|へ具体化するまで科学実行しない。

独立holdoutのmean errorには参照側MC誤差もある。一pathの一致や有限点の最大誤差からuniformな長時間boundを宣言しない。モデル選択/再fitに使ったholdoutは新しい独立検証にならない。full referenceが不能かつtail/閉包も未証明なら、modelを正式長時間価格として採択せずUNKNOWNを返す。

## 保存・判定・STOP

| 状況 | 出力・次の扱い |
|---|---|
| Tが成功 | 軸別path C/S/D・6指標・fingerprint/seed/digest/全SOURCE、stage/wall/RSS。**技術実行可能性の一観測のみ**。mean/CI未資格でSTOP |
| SOURCE/λ/scalar/law/時刻/grouping不一致 | 先行・部分結果/例外を保持しmandatory STOP。tolerance緩和・fallbackなし |
| OOM・実際に継続不能・compiler失敗 | 失敗pathと実行段階を保存。C=0/欠測除外せず、retry・新seed・短roundへ自動移行しない |
| Mで幅不足又は未知law/rare/model項 | finite-cohort診断と不足理由のみ。標本を無断追加せず未資格STOP |
| model holdout失敗/適用域外 | 同じholdoutで再校正してPASSにしない。長時間価格UNKNOWNでSTOP |
| 適格mean/同時区間が得られた | HANDOFF仕様でC/Bへ供給する。供給だけでN資格・正式G・winnerを認定せず、別承認の会計までSTOP |

今回の完了時はcommit/push・remote SHA/tree/主要成果物照合後 **mandatory STOP**。次は**ユーザー**がTの有限案又は先にB入力を待つ案を判断し、科学実行する場合だけ対象・SOURCE・path/compile上限を明示承認する。**B/A**はN/claim/law契約、**C**は必要なSとbinding差分、**D**は選ばれた範囲のcaller/価格取得を担当する。自動進行しない。
