# B/D向け：ideal source lawとCDF/PCG64の接続

[REPORT](REPORT.md)／[数式と条件](MATH.md)／[静的診断metadata](LAW_CONTEXT.json)／[SOURCE](SOURCE_INDEX.json)／[次の有限計画](SCIENCE_PLAN.json)。新実H4数値解析は未実施。

## 受入時の区別

|項目|状態|
|---|---|
|finite-K/source samplerの生成構造|同じSOURCE・support・τ・metadataのもとで対応確認|
|C de2427 / 元71ba49 / provider c31efのrte.py|全blob一致|
|厳密実数/IIDモデルの数学的生成law|同じ入力なら一致。解析的重みとbinary64重みの差は別途|
|実H4のP/Q/F/G距離|未評価、null|
|CDFを一様53-bit乱数へ適用する確率|既存cdf_lawで有限exact診断可能。今回は人工入力のみ|
|固定seed PCG64のjoint-law距離/IID保証|UNKNOWN。CDF計算だけから距離0にしない|
|S・C−Sへのlaw systematic|未知。全supportの範囲とjoint距離が必要|
|平均compiled価格の資格|false、今回新price recordなし|

## Codex D：paired residual

固定参照Dはaf6f27d7、P0は442cafa1。P0はcontrolled-ZZのlocal lowering診断で、full-wrapper cohortやsampler joint lawの検証ではない。後続のD公開SOURCEでsampler変更がある場合、本結果の参照先を自動的に付け替えない。

1. Cの26 E[S]のSOURCE・interval・quantity・元Accountingを保持する。DがS′を使う場合、元Sと混ぜない。
2. physical candidate/round/q/R/τ/finite-K・component/basis順・H/state/snapshotとcompiler/control/full wrapperを照合する。
3. C/Sは同じevent列・digest・seed/cohort/indexと同じaxisでpairにする。X/Yは一つのpathからのpaired応答。
4. PCG64 Generatorのreset単位、NumPy version、seedの組み立てとnamespaceを記録する。元sample_requestsはouter/piece別seed、axis共通。未来のD runnerの実provenanceはD側で示す。
5. `event_probability`のmetadataはnominal。実選択確率はCDF閾値由来であり、metadata値やγ16 parityだけからlaw一致としない。
6. normal cohortとforced rareを分ける。未観測order2の条件付き残差を0にしない。poolにはlaw weight/条件付き推定と不確かさが必要。
7. IID/gridモデルを採用するなら明示する。それはPCG64の数学的独立証明ではない。有限cohort CI、law systematic、長q/R model systematicを別々に供給する。

Iを構造側のlaw、Qを残差標本化lawとする。I-targetならmixed値のlaw biasはE_Q(C−S)−E_I(C−S)、Q-targetならE_I S−E_Q S。目標lawと補正対象を明記し、同じ誤差を二重加算しない。

## Codex B：共通APIは変更不要

bf515a97の`compiled_residual.combine`は構造/残差のidentity・law_binding・profileを照合し、law_context.lawの差又は未宣言をreasonへ残す。値が計算可能でも`UNQUALIFIED_COMPONENT_SUM`、qualified=falseを維持する。この技術調査を理由にreasonを消したり`QUALIFIED_EXPECTATION`へ昇格しない。

LAW_CONTEXT.jsonは価格providerのrecordではなく診断packet。既存law_binding/IDを改名せず、各componentのlaw_contextと別添のlaw対応証拠へ本packetを参照する。最低限、次を区別する。

- `SOURCE_GENERATIVE_CORRESPONDENCE_CONFIRMED`：SOURCEの対応。
- `EXACT_LAW_EQUALITY`：証明できる場合だけ。今回の実H4判定はUNKNOWN。
- `BOUNDED_LAW_DISTANCE`：対象law、TV上界、条件、SOURCE、scopeが得られた場合だけ。今回はnull。
- `UNKNOWN_SYSTEMATIC`：PRNG joint-law、価格range、rare等。nullを0にしない。

不明な価格systematic、cohort variance/XY covariance、cross-round covariance、total CIは互いの代替ではない。本packetは26 E[S]をcopy/改変せず、新E[C]/N/Gも出力しない。

## 最小の次段階

ユーザーが必要と判断すればSCIENCE_PLANのF0を一括承認する。Cは保存324 component確率と13 order分布のみを使い、元exact bytesとCDF helperでcategorical TV→条件付きpath TVを評価できる。新sampling・compile・E[S]は不要。ただし保存科学数値の新解析なので今回は実行していない。

F0を行っても固定seed PCG64のIID証明、未知の価格range、有限cohort/rareの精度保証は得られない。global boundが大きい場合のcost-aware検証やeffective-grid kernelによるE[S]評価は別の科学scope。自動追加しない。

担当：**ユーザー**がF0と採用する確率モデルを判断、**C**が許可された有限診断、**D**が自身の残差証拠、**B**が未資格情報を保持して接続。今回Cは公開照合後mandatory STOP。
