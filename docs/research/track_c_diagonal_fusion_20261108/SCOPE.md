# Track C：run内対角gate融合

基点2745d57bea5c28604739bf9c01338e14e649c467。固定H4/1Å/STO-3G/DF12/LD3・Full12、保存元H exact ground、18round/δ0=.02、epsilon1.5936e-4Ha、alpha.05、phase.4、prep除外、Qiskit1.3/compiler A/seed17を維持。旧SOURCE・raw・grant・ledger・marker・STOPは変更しない。

共同isometryの最大same-basis runだけを対象に、同じPauliの角度をexact rationalで集約し、controlled diagonalをparity多項式として合成する。既存central-control emitterとQiskit GraySynthを再利用する。単一gate loweringと、角度集約・複数parity共有の効果を分離し、既存compilerが削減済みの価格をbaselineとする。sampling law・return aggregation・論理event列・符号・normalization・相対位相は保持する。

既存384標本を今回trainingと明記し、必要な独立holdoutを別seedで生成。D内部BCH係数/h/resolutionは固定し、Fullにも同じdiagonal合成とα最適化を与える。source/native/RPE資格の既存証拠を再利用し、新しい角度丸めを区間/有理数で会計する。RZ/CX/sizeは具体的modular回路の加算価格、depthはserial上界と実DAGを分ける。

合理的な限定検証を一括継続し、成果commit/pushとremote照合後mandatory STOP。別Track・物理条件・RTE法則変更・詳細GPT科学レビューへ自動進行しない。
