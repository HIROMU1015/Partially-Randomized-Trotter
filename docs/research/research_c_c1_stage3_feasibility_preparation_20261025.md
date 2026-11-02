# C1 Stage 3：feasibility-only限定実行の技術準備

2026-10-25 JST。起点 `6b754579f0bcb7e360ab7050c2ec78cfe546c4a2`。
Branch `codex/research-c-c1-stage3-feasibility-preparation-20261025`。

**feasibility-only経路、人工検証、新未承認契約を準備した。実H4の科学callは0。POTENTIALLY_FEASIBLEはCA達成・資源優位の判定ではない。**

[未承認限定科学契約](research_c_c1_stage3_feasibility_contract_unapproved_20261025.json)／[SOURCE・人工結果・STOP](../../artifacts/research_c_c1_stage3_feasibility_technical/2026-10-25/)。正式共通指示2026-10-25版は前作業で読んだものと同じSHA256 `e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720`である。許可は別途のチャット指示で扱い、新しい証明書・承認記録・grantは作成しない。

## 1. 限定entryと除外範囲

旧統合SOURCE・entry・契約・結果を変更せず、`scripts/run_research_c_c1_stage3_feasibility.py`に専用の`--mode feasibility-only`を追加した。省略時も同じ限定modeであり、full modeへ切り替える引数はない。

```bash
# 技術的byte preflightだけ。NPZ／QPY decode・科学評価は行わない。
python scripts/run_research_c_c1_stage3_feasibility.py --mode feasibility-only \
  --contract docs/research/research_c_c1_stage3_feasibility_contract_unapproved_20261025.json \
  --input /path/to/saved/P-H4.npz

# 次回、固定SOURCE・契約に対する別明示科学許可を受けた後だけ使用する。
python scripts/run_research_c_c1_stage3_feasibility.py --mode feasibility-only \
  --contract docs/research/research_c_c1_stage3_feasibility_contract_unapproved_20261025.json \
  --input /path/to/saved/P-H4.npz --execute \
  --output artifacts/research_c_c1_stage3_feasibility_science/<date>/<fresh-run-id>
```

限定modeは、保存入力／ground／alias、PF operator誤差、補正RTE operator誤差、attenuation、候補分類だけを実行する。QPY読込、Gaussian回路、native回路build／compile、event sampling、合成誤差資格、X/Y信号、RPE、shot数、総費用、最終資源順位付けには接続しない。`shots`／`total_resources`／`final_resource_ranking`／`CA_achieved`は0ではなく未評価の`null`として保存する。

## 2. 固定条件と再利用

保存P-H4（直鎖1.00 Å・STO-3G、Nα=Nβ=2、sector36／full256）、通常CDF number・元g補正κ・保存rank8／保存XDF rank10を使用する。各DF Hamiltonianのsector ground-state oracle、状態準備除外、ideal finite-RTE IID lawを維持する。新fit／optimizer0。

CA=0.00159 Ha、DF reserve CA/8、δ=0.1、m=0..12。決定論ordered symmetric S2のs∈{1,4}、PRは元outer順L_D=2・s∈{1,4}・r∈{4,16}・K∈{0,2}、合計20候補。候補の列挙順序は旧`integrated.candidates()`をそのまま使用する。βtotal=π/6、βstat=π/12、alias窓[-10,10] Ha、既存数値toleranceも変更しない。親契約から科学条件だけを抽出し、その一致をpreflightで確認する。旧compiler／費用パラメータは限定契約の実行条件へ含めない。

| 再利用元 | 限定経路での用途 |
|---|---|
| `c1_stage3_integrated` | 候補順序、既存十分条件predicate、call journal、resource plan、partial progress |
| `c1_stage3_molecular` | 保存入力読込、matrix schema、ground／sector埋込／alias、数値条件・保存DF誤差の説明 |
| Stage 1／2・一般K・CAR | 保存fit export loader、κ／scalar保全、一般Kの値表現、保存CDF generator、未保存XDF generator11の独立CAR構成 |
| `c1_stage3_connection` | 元frame順のtail・I項分離と既存有限RTE configuration。native circuit emitterは呼ばない |
| `c1_stage3_reference` | PF／補正有限RTE kernel、既存attenuation、full256 matrix power・operator norm。signal／RPE helperは呼ばない |

新SOURCEは`src/trottertracks/c1_stage3_feasibility.py`と専用CLI、人工testだけである。旧backendの`prepare()`にはQPY読込とnative専用CAR atom catalogが含まれるため、そのまま再起動しない。数値準備部分だけを新subclassに明示し、既存helperを呼ぶ。旧固定SOURCEを変更せず不要処理を確実に省くための最小の分岐であり、新fitter・Gaussian合成器・RTE・数値計算ライブラリは実装していない。

## 3. 実行順序と科学call上限

**最大639 logical units、16種類。** 旧full上限5,637をこの限定経路へそのまま適用しない。各unitの内部処理は契約に記載し、matmulごとの二重計上はしない。

| 処理 | 上限 |
|---|---:|
| 保存XDF input／保存fit export読込 | 1＋1 |
| 保存operator読込（sector3＋CDF full10） | 13 |
| ground eigenpair（physical／CDF／XDF sector） | 3 |
| CDF／XDFの値表現準備 | 2 |
| CAR context／XDF full generator／XDF full sum | 1＋11＋1 |
| ground埋込・alias等／一般K tail分離 | 2＋2 |
| PR有限RTE configuration | 16 |
| arm×roundのexact evolution cache | 26 |
| 20候補のkernel reference | 20 |
| 20×13 round reference | 260 |
| roundのfeasibility判定 | 260 |
| 候補の全round・DF reserve集約 | 20 |
| 合計 | **639** |

保存準備63、kernel内のRTE設定16、kernel20、round reference260、round判定260、候補集約20。科学開始前のbyte／schema／環境metadata確認は科学call0。saved fit SOURCE、親評価SOURCE、今回の限定SOURCEは独立に保持する。

将来の実行では、まず保存XDF whitelist・κ/J/U/constant・13行列を読み、必要なXDF generator11だけを構成する。3sector groundとfull埋込／aliasを検査し、ground NPZを保存する。物理DF operator normはStage 1保存値＋元のengineering allowanceを再利用し、新しく測り直さない。厳密な区間証明ではない。

次に26 exact evolutionsをcacheし、20候補×13roundのPF／補正RTE誤差・減衰を評価する。中間evolutionをsector36へ無資格射影せず、旧定義のfull256を保持する。candidateごとに全13roundの結果を保存し、登録順で分類表を出力してSTOPする。分類別件数は集計するが、candidateの資源選択・順位付けはしない。

## 4. 分類の意味とSTOP

旧十分条件を維持する。A>0の場合、既知部分について

```text
εknown = εPF,operator + εRTE,corrected,operator
εknown <= sin(βtotal - βstat)
```

を検査する。raw signal単位では既知誤差はA εknownであり、位相条件の相対尺度へ戻すとAが消える。未知の合成誤差を0へ置き換えていない。最終CA精度、statistical shots、total resourcesへは進まない。

- `INFEASIBLE`：少なくとも1roundで登録の十分条件が満たせない、binary64でobserved attenuationが0、または保存DF誤差がCA/8を超える。物理的実現不可能という必要条件ではない。
- `POTENTIALLY_FEASIBLE`：全13roundの既知PF/RTE部分と保存DF reserveが通る。合成誤差・実sampler誤差等の資格は意味しない。CA達成・総資源優位とも扱わない。
- `UNRESOLVED`：必要なfeasibility値が欠け、既知の不適格理由もない。欠けた誤差を0で補わない。

既存referenceのunderflow policyを保持する。A=0の場合、巨大な補正後matrix powerとnormを作らず、誤差欄は未測定の`null`となり得る。非有限値・負の誤差・shape／alias不一致等は未資格として隠さずFAILとする。

候補INFEASIBLE／UNRESOLVEDでは他の登録済み候補を続ける。最初のcorrectness FAIL／call超過は新規依存callを止め、既開始callを安全にdrainする。完了済みround値のprogress JSONL、候補JSON、科学call、ground NPZ、failure／resource／STOPを保存する。retry／resume0、条件・閾値の事後変更や登録外探索はない。preflight不一致は科学call0で起動拒否し、既存outputへ書き込まない。

## 5. 人工検証・資源・完了境界

新規人工16件と既存Stage 3 reference／統合回帰44件、計**60 tests PASS**。人工・mock入力だけを使った。新限定modeの人工試験では、全639 callの一致、20候補×13round、全潜在適格／全不適格／全未資格／混在、欠測と非有限値の区別、途中FAILの数値partial保全、fresh output、数値条件変更拒否、別full modeの引数拒否を確認した。

実保存backendには2mode／full4次元の人工NPZだけを渡し、読込先をtemp領域へ限定した。QPY読込、QuantumCircuit、transpile、NativeBuilder、native専用atom catalog、旧full prepare、fit、sampling、shot／費用評価を禁止するmockを置いた状態で、全20候補×13roundと639 callが通った。別の非可換人工例で、有限K2 RTEのPF・補正誤差・減衰と独立3-exponentialの一致を確認した。旧回帰の人工回路testは元のtechnical検査であり、新限定modeや実H4のcompileではない。

既存task専用venvをread-onlyで再利用し、依存追加・共有環境変更なし。利用可能CPU／cgroup quota／RAMからworker数を選び、共有行列準備後にも空き資源を再確認する。26 exact evolutionsと独立candidateを並列化し、BLAS thread数をprocess内で調整する。旧の保守的dense buffer見積もりを再利用しており、固定CPU／RAM／worker／wall capではない。matrix power・arm別exponential cacheを使い、native atom catalog・QPY・回路・全trajectory列挙を省く。登録NumPy／SciPyのCPU経路を用い、未検証のGPU高速化は主張しない。

今回の実H4 decode／operator・PF/RTE評価／回路build・compile／shots・総費用計算／新fitはすべて0。旧SOURCE・科学結果・grant・ledger・marker・STOPを保全する。人工PASSから科学的GOを出していない。

次はユーザー／研究Cが限定SOURCE・639-call契約を確認し、feasibility-only科学実行をチャットで別途明示許可するか判断する。Codexは許可後だけ評価を実行する。取得できるものは260セルの誤差／減衰と20候補の分類・根拠であり、CA達成や資源順位ではない。今回remote照合後は**mandatory STOP**。
