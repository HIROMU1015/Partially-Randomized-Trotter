# C1-2/3 SOURCE・実行前監査

基点はR1 `2937e6b655ba3cd77f982d6901d766355a8b92c1`。追加は本namespaceだけ。共通運用指示2026-10-17、§8/8.4とprojects/AGENTS.mdを適用し、主agent自身が実装・確認する。別agentによる独立reviewではない。指示の公開・one-shot承認を使い、旧grantは再利用しない。

- sampler: 固定243eea8の`return_aggregation.py`、`g7_generator.py`とF3 facadeは変更しない。正式event queryはsampling trialと区別する。`AllTrials`はvalid zeroを分母に含め、technical failureでは終端し、formal queryをestimatorに入れない。
- provider: 新規`adapter.py`の`g6_df_sidecar_v1`。既存DF builderのbasis appendとHadamard wrapperのregister作成部を利用する。RTEEvent/DFPartialS2RepeatedCircuitResultへの偽装はない。Z/ZZ中心のcontrolled積・rotationは新adapterで接続する。G7のhelper経路と同じ費用とは扱わない。
- 規約: Q=sign B P B†、語の回路時間順は逆、長さ2のparentはancilla Z、childは2 sigma sign phiのcontrolled RZ/RZZ。一般の奇数語をinvolutionとみなさない。B†はreverse gate order + actual inverse。scalarはcontrolled branchのPで保持する。
- R1復元: 係数はexact pとλ、符号はQに一度だけ入れる。保存gate型RZ/XXPlusYYのconstructor、descriptor、local matrix hash、basis hashを照合する。再対角化・DF再分解は呼ばない。独立referenceは保存hex行列を用いる。
- 有限bit: fixed H=2048/K=4096、eta=rho=10^-12。旧160/256を十分とみなさない。F1/F3の全pに対するP0–P5をone-shot内で照合し、不足時はSTOP。angleは既存F3の固定256bit/64項/tau=10^-12でfail closed。float angleへの丸め量も別記する。これはnative strict error証明ではない。
- 数値基準: 固定G10 `g10_comparison.py`のfinite first moment/strict controlled比較にある1e-10。Frobeniusをspectralの上界として使う。浮動小数点診断であり厳密証明ではない。X/Yは診断値を保存し、新しいrigorous PASS基準を作らない。
- Schedule: 固定`df_partial_s2.py::_build_parts/build_step`のforward half→tail→reverse halfを静的に照合する。q=1、T=4/5、tailのr=4各1/5、sigma=+1、m=5。少数eventをsubstepで検証し、全trajectory・PF誤差を認定しない。constantと抽出identityは可換scalarとして診断上4分割し、合計時間Tを保つ。
- 人工確認: 2 qubit、3 label、負sign/非可換basis、正負sigma、root/長さ2/4、strict controlled branch、actual adjoint、X/Y、scalar、source/model/Evidence、raw/coin zerosとtechnical failure、63 artificial eventsのideal/digital first moment。digital meanはroot近似により有限Taylorと厳密同一ではない。

`seal_contract.py`はJSON/bytes/AST・環境metadataのみを扱う。H4 Green係数・operator・state評価は`run_c13.py`の一回だけ。SOURCE commitの直後にauthorization receiptだけのcommitを作り、実行時に両者とmanifestを照合する。result directoryのexclusive作成で再実行を拒否する。失敗を含むstdout/stderrを保持する。

既存SOURCEはruntimeでは検証済みnamespace pathから選択submoduleだけimportし、広い__init__ exportを回避する。package・共有cache・OS設定を変更しない。環境は固定venv、critical runtime bytesとpackage metadata/RECORDを照合する（hermetic OS保証ではない）。全古典/保存費用やnative費用の優位は今回測定しない。
