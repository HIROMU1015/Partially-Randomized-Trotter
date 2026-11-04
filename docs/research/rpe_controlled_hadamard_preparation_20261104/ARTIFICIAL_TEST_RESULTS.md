# 人工回路・mockテスト結果

2026-11-04 JST / LOCAL。**34 passed in 2.50s**。
新規人工30件＋既存人工4件。実H4入力のload/build/state-action/compile/数値評価/samplingは全て0。
主エージェント自身による技術検証であり、独立詳細科学レビュー・immutable CI再現ではない。

## 実行とfixture

Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、pytest9.0.3。
既存research venvをread-onlyで利用し、パッケージ・共有設定を変更していない。
2-mode・4振幅の手書きDF行列、非対角one-body、3fragment、constant .17から
既存prepare/builderを呼ぶ。H4 snapshotやstate NPZの読込はしない。
乱数を使わずevent index列を直接指定し、SCF/DF因子化・ground-state solve・transpileを呼ばない。

```bash
PYTHONPATH=src <research-python> -m pytest -q \
  docs/research/rpe_controlled_hadamard_preparation_20261104/test_circuit_semantics.py \
  tests/test_df_rte_qiskit_builder.py::test_controlled_event_is_exact_block_diagonal \
  tests/test_df_rte_qiskit_builder.py::test_every_small_event_matches_dense_reference_with_all_signs_and_phases \
  -p no:cacheprovider
```

| 新規test | 件数 | 確認した性質 |
|---|---:|---|
| forced_noncommuting_pipeline | 7 | r=1,2,4,8,16,32/K2とr3/K4、q_probe2。occupation-energyによるF/R、signed eventとの一致。raw/optimized/uncontrolled/whole-gate control、両Hadamard軸 |
| support_policy | 1 | 複数eventのsupport_run_le_1とfull basisが同じoperator、実際のsupport選択あり |
| order_and_sign_mutations | 1 | 非可換product逆順、rotation符号逆転、F/R交換、relative phase誤りを検出 |
| categorical_mean | 1 | 不均等有限lawの全16 trajectory和と平均演算子の積、channelとの区別、追加B除算の誤り、normalized ray |
| paired_component_law | 1 | 不均等人工component/order lawの10event列挙と非列挙paired式、各eventのcontrolled block。実CDF payloadではない |
| all_rounds_wrapper_metadata_domain | 18 | q=2^m/m0..17のbenchmark metadata受理。回路buildなし |
| B phase/sign error detection reuse | 1 | 既存B evaluate_jobに人工phase circuitを渡し、負sine・誤conjugate/relative phase検出 |
| **新規計** | **30** | SOURCEは[test_circuit_semantics.py](test_circuit_semantics.py) |
| 既存controlled_event test | 3 | faithful identity rotation/product、ZZ controlledを独立dense event式と比較 |
| 既存every_small_event test | 1 | 人工小tailの全event、n0/n2、I/Z/ZZ、±sign/phase |

大きなqの確認は2×2人工平均行列のbinary powerだけ。
巨大qのQiskit回路や実H4 mean評価を呼んでいない。
pipeline比較はatol1e-11・rtol0、B evaluator再利用は既存atol1e-10。
人工PASSからH4の同じ許容差内一致を推定しない。

## 技術修正と限界

最初の人工実行は28 PASS/1 FAIL。F/R交換の検出量が約2.05e-5で、
任意に置いた「差>.001」のfixture強度条件だけが過大だった。
検出条件を`差>1000*ATOL`へ変更し、正しいoperatorの一致基準1e-11は変更していない。
科学結果・実H4の合否基準変更ではない。
既存testとの同時収集ではPYTHONPATH未指定でcollection errorが出たため、
既存環境に合わせ`PYTHONPATH=src`を指定して上記最終実行を行った。

既存codeを再利用し、新規は強制profile・occupation-energy参照・分布平均の人工fixtureだけを追加した。
既存test helperにはsamplingを呼ぶものがあるため、そのhelper全体を再利用せず明示eventを構成した。
また広い既存suiteには実分子やartifact再解析が含まれ得るため、今回必要な人工testだけを選択した。
小さいfixtureの約2.5秒に対してprocess起動を加える利点はなく、逐次実行した。
CPU/RAM/workerの人工上限は追加せず、実科学runのbackend・資源選択は未実施である。

共通basis runtime、DF preparation、Qiskit自身の浮動小数算術は依存として残る。
任意component/run/大規模qや高精度の数学的誤差証明、compiled equivalenceは保証しない。
今回、旧SOURCE・保存結果・grant/ledger/marker/STOPを編集せず、別包含jobを呼び出していない。
