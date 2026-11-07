# 新A N3-B0-numeric：有限辞書パラメータと保存word coverage

2026-11-08 JST、SERVER 2。固定preflight commit **`385c24a00fb9b47d6639fea1e7aecb0c37d55b23`**、科学入力基準`79dfa01e6796d1ecab50e30694f5b3f73b95f4ab`。専用branch `codex/pr-reference-n3b0-numeric-20261108`。

**固定scopeの計算・指定検算を完了し、mandatory STOP。** 主仕様は **b=36、j=-36…6、43基本key、基本wordの一様Frobenius要求10^-13**。固定3 archiveには選択範囲のliteral keyに一致するrecordがなく、**再利用候補0/43、取得必要43/43**となった。

資格は`PARAMETER_SCOPE_FIXED`、`DICTIONARY_ACQUISITION_REQUIRED`、`PRICE_BOUND_CONDITIONAL`。実emitter誤差・完成word policy・全support CT RPE成功は**UNQUALIFIED**。基本角度表の確定を実辞書の完成とは扱わない。新規合成／guard／word policy実装／T価格集計には進んでいない。

## 1. 固定条件・選択規則

Linear H4・1.00 Å・STO-3G・DF12、元H厳密ground供給・状態準備費除外。Partial S2・LD3・h=.02・finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、m7/q128/r1/73 bits、最終native RZの一様strict epsilon=10^-11を保持する。

Qiskit1.3.0/compiler A、rz/sx/x/cx・全結合、B′4-CX controlled-RZZ、Reference F/G・actual inverse・vacuum P・2 register・8-CX CSWAP、D融合・basis共有・scalar/event・X/Yは不変更。energy1.5936e-4 Ha、failure.05、phase.4、alpha1/720、199 shot/axisも変更しない。

[PARAMETER_SELECTION_RULES](../pr_reference_n3b0_preflight_20261108/PARAMETER_SELECTION_RULES.md)と[NUMERIC_EXECUTION_SCOPE](../pr_reference_n3b0_preflight_20261108/NUMERIC_EXECUTION_SCOPE.md)をそのまま用いた。保存Lを厳密有理数へ変換し、sqrt2を192-bit区間で1回評価した後、全判定を整数／Fractionで行った。

b=0…1074の各候補についてN_b／J_b／M_bを求め、固定量子化上端Q_b^upの下で`M_b*10^(-p_b) < epsilon-Q_b^up`を満たす最小正整数p_bを求めた。tuple(p_b,M_b,b)の辞書式最小を、**cache照合前**に一意選択した。固定上端に対する条件付き仕様の選択であり、実T最適性や真の誤差に対する一般の最適bを証明したものではない。

## 2. 主仕様と誤差要求

| 項目 | 結果 |
|---|---:|
| b | **36** |
| j_min / j_max | **-36 / 6** |
| N_b | **5668839823158** |
| M_b／最大基本word使用数 | **43** |
| 内部p／予定要求exponent | **13 / -13** |
| 各基本wordのFrobenius割当 | **1/10000000000000** |
| 基本誤差割当和 | **43/10000000000000** |
| 選択tuple(p,M,b) | **(13,43,36)** |
| 評価したb／条件付き成立候補 | **1075 / 1039** |

保存角域Lは82.49247654978845 rad。上端到達や一様角分布を仮定せず、SOURCEのIEEE演算順・remainderを変更しない。選択した43 positive keyの全表と整数比は[FIXED_PARAMETERS.json](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/FIXED_PARAMETERS.json)。各2^jをbinary64と整数比で照合した。負角には正側連結word全体のactual adjointを要求するが、今回はwordを生成していない。

量子化Frobenius上端の50有効桁による外向き表示は

```text
[5.1448789686149941146202353564502048486962735310114e-12,
 5.1448789686149941146202353564502048486962735310115e-12]
```

emitterへ要求するstrict残余R_emit^loは、厳密には

```text
584610679685680278574326123259095118223625605117136488901731578927
/1053122916685571866979180276836704323188950954005491112543109775360000000000000
```

であり、その外向き表示は

```text
[5.5512103138500588537976464354979515130372646898850e-13,
 5.5512103138500588537976464354979515130372646898851e-13]
```

である。正本は[ERROR_BUDGET.json](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/ERROR_BUDGET.json)のFraction。`Q_up + M*g_alloc + R_emit_lo = epsilon`を厳密に検算した。

全基本wordが各割当を満たすうえで、**実e_emit < R_emit_lo**を別途資格できることが必要。ここで算出した残余は実emitter誤差の観測値／保証値ではない。Wの浮動global_phase処理に対してe_emit=0を置かず、実装資格はUNQUALIFIEDのままとした。より小さい実保存guardを理由に、missing keyの割当を緩める再最適化も行っていない。

## 3. 保存3 archiveのcoverage

| 固定pool | metadata確認record | guard方式資格あり | 方式未資格 | 先行poolとの重複 |
|---|---:|---:|---:|---:|
| CT-4B strict_records | 299 | 299 | 0 | 0 |
| 旧C2-B raw_synthesis | 7603 | 7599 | 4 | 111 |
| CT-1 strict_records | 268 | 268 | 0 | 97 |

重複208件の科学的内容は一致し、一意record keyは7962件。旧C2-Bのmethod未記載4件は未資格のまま保持し、補作／再guardしていない。JSON整形ではなくspec／word／phase／保存誤差等を対応させた。既存T／Tdaggerの文字数確認はmetadata整合だけに使い、価格和／最大値を求めていない。

**選択した43 keyにliteral一致するrecordは0件**であり、精度不足を理由に候補を排除した結果ではない。全43 keyがACQUISITION_REQUIRED。coverage表のstatusは`NO_ELIGIBLE_EXISTING_BASIC_KEYS`とし、非ゼロの部分再利用があるかのようなラベルを付けない。[BASIC_KEY_COVERAGE.json](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/BASIC_KEY_COVERAGE.json)に全key・照合結果・missingを保存した。

N3-Aで見つかった既存`0x1.0000000000000p-61`は今回の指数域外である。これを含めるためにb／precision／key集合を変えていない。CT-4Bの最終native C0表と全採用wordも保存archiveと照合し、不変更を確認した。内部要求がより厳しいことを理由に、D／CSWAP／scalar／中央回転の既存価格を置換しない。

## 4. 指定検算とSOURCE

[CHECKS.json](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/CHECKS.json)の指定検算はPASS：固定入力27件のhash、sqrt2区間の厳密二乗包含、全1075候補の整数floor／bit length、各pの1段粗い要求との比較、辞書式選択の一意性、43 keyの厳密整数比、budget恒等式、既存record metadataと重複整合、zero／正負／最近接tie／角域両端の算術検査を完了した。端点検査は到達trajectoryの観測ではない。

全bの判定表は[CANDIDATE_DECISIONS.json.xz](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/CANDIDATE_DECISIONS.json.xz)。選択候補だけの基本keyを生成し、候補全体の角度catalogueは作っていない。

新しい[会計SOURCE](../../../scripts/run_pr_reference_n3b0_numeric.py)は固定blob読取・整数／Fraction・sqrt2区間・保存metadata照合だけを行う。CT-0の`positive/validate_record`をASTから取り出して再利用し、旧module全体／旧runner／Qiskit／pygridsynthをimport／実行していない。新しいword policy、丸め／連結adapter、emitterは実装していない。

[SOURCE.json](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/SOURCE.json)、[INPUTS.json](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/INPUTS.json)、[ENVIRONMENT.json](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/ENVIRONMENT.json)に実行SOURCE hash、固定入力／選択文書のhash、既存runtimeを保存した。指定venvのPython3.12.3／mpmath1.3.0で約2秒。利用可能affinity64に対する小さい有限算術・読取処理なので1 processで完了し、独自資源cap・共有package／設定変更は行っていない。

## 5. 価格・未資格・次担当

[PRICE_FEASIBILITY](PRICE_FEASIBILITY.md)のとおり、全43基本wordの実T価格が未取得で、B_dictと優先cacheを含むB_Pは未評価。N1閾値は保存値を引用するだけで、T削減率・全support優越を認定しない。実word取得／停止・実emitter・全support CT RPE成功・18round総T／T-depthはUNQUALIFIED。common frame／B′ b2 eventの既存欠損・STOPも維持する。

新規合成、strict guard実行、word連結／policy実装、回路build/compile、sampling、真空位相分布、T価格和／最大値／期待値、N3-B1、N2、SK、199-shot積／全18round総T／T-depthには進んでいない。旧SOURCE/raw/result/grant/ledger/marker/STOP、他ユーザー・別Track・共有環境は変更していない。

[STOP.json](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/STOP.json)のとおり**承認範囲完了でmandatory STOP**。次担当は**GPT／ユーザー**：固定された43 key／内部要求とemitter残余を評価し、未資格を解消する最小の次段階を判断する。Codexは新規取得やN3-B1を自動開始しない。
