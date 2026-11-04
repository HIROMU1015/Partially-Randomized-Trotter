# RPE候補A：係数閾値の修正版と別の1回実行計画

2026-11-03 JST。**技術修正・人工検証のみ完了。修正版のH4科学実行は未承認・未実行。**
run01の停止を解除せず、新しい科学実行を自動開始しない。
この文書と修正版runner/testを含む公開commitを、新しい固定SOURCEとして用いる。
full SHAは完了チャットで示す。許可を得ても別のcommitへ無断で移さない。

## 問題と修正

元SOURCE `09237b48ea22ad10d860e67e4a2c1465aad65e41`の1回試行は、sampling前の
preparation検査で停止した。停止結果の固定commitは
`07a4a824714de3be71f4d85e8381598bc269ba68`。
[停止報告](../rpe_fresh_iid_measurement_a_execution_20261103/REPORT.md)と
[run01原RESULT](../../../artifacts/rpe_fresh_iid_measurement_a/2026-11-03/run01/RESULT.json)を保持する。

保存signalの`coefficient_atol=1e-12`に対して旧runnerは引数省略で既定値0.0を使用していた。
修正版は次の2点を変更した。

1. `prepare_df_partial_s2(..., identity_policy="extract_identity_phase", coefficient_atol=1e-12)`を明示する。
2. preparation照合で停止する場合、例外に実際のpreparation hash・component数・deterministic block数を残す。

旧runnerを変更せず、
[`run_fresh_iid_a_v2.py`](../rpe_fresh_iid_measurement_a_preparation_20261103/run_fresh_iid_a_v2.py)を追加した。
科学helper、sampling・測定・seed・統計診断・出力処理は旧runnerから再利用した。
固定旧ファイルを直接編集せず、実行可能な別版を残すためにrunnerだけをコピーした。
module説明、`describe()`の修正版ラベル・閾値metadata、`prepare_context()`の上記2点以外は同じコードである。
adapterと既存libraryに変更はない。

## 人工検証と限界

[`test_fresh_iid_a_v2.py`](../rpe_fresh_iid_measurement_a_preparation_20261103/test_fresh_iid_a_v2.py)の
3 tests PASS（約0.010秒）。実際の修正版`prepare_context()`を呼び、
H/state読込・DF依存moduleを人工fixtureへ置換して、閾値の受渡し、停止時の診断値、
downstream処理の未到達、metadata-only CLIを確認した。実H4のDF処理やRNGは呼んでいない。

```bash
python -m unittest -v test_fresh_iid_a_v2.py
```

同じdirectoryの既存`test_fresh_iid_a.py`も18 PASS（約0.250秒）。
こちらは元runnerと共通adapterの人工testsであり、実H4数値照合ではない。
修正版の共通helperは元runnerと同一であることを静的に照合した。
既存科学source・元A/B成果物・run01・manifestの変更がないこと、JSON/リンク・差分も確認する。

**修正後に保存preparation hash/218 componentが再現されるか、周辺平均が保存meanと整合するかは未検証**。
今回の人工PASSを科学的GO・統計的同等性・RPE成功の証明としない。

## 別の1回に固定する科学条件

元の[実行計画](../rpe_fresh_iid_measurement_a_preparation_20261103/SCIENTIFIC_EXECUTION_PLAN.md)の
SOURCE `09237b48ea22ad10d860e67e4a2c1465aad65e41`版を継承し、
runner・SOURCE commit・出力先と係数閾値の明示だけを本書の指定へ更新する。

| 項目 | 固定条件 |
|---|---|
| Model/geometry/basis/rank | linear H4/隣接1.00 Å/STO-3G/保存legacy DF rank12 |
| State | B commit e22d7718c47358ff650dbc04e8e72f6ab1c8bf41のphysical ground state、8 qubit、4電子sector70 |
| PF/RTE | Partial L_D3、単一δ=.02、r1/K2、extract_identity_phase、coefficient_atol=1e-12 |
| Round/axis/N | m0/q1のcosine/sine各510、m3/q8の各454、計1,928 trajectory・仮想shot |
| 保存target Re/Im | q1=(.9989255707627144,.04330819375415813)、q8=(.939502685390124,.3393522369363202) |
| Preparation | hash 89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919、218 retained component |
| SOURCE carrier/mean | carrier4591b8d75a2a6a55c74a51225f0c61648023c8ee、V2 commit78c49d0b5f1d71e63b5959e78813237923bbc648のattenuated event-mean |
| RNG | 同じ公開namespace rpe_fresh_iid_measurement_a_20261103_v1_public、PCG64、legacy key/role別seed派生 |
| 判定 | 元計画の12比較・family η=.05、Hoeffding h=sqrt(2 log(2×12/.05)/N)、既存数値許容差 |

保存H/state/RESULTのbyte hash、physical fingerprint、N/α/β/ε、符号・normalization規約、
有限標本の3水準、欠測・非有限・範囲逸脱のSTOP条件をすべて継承する。
q1の保存α=2.8897922699779505e-8、q8は2.2180178685403194e-7、
β=(.014050012886294481,9.899584668737594e-8,.38594988811785885)、
ε=(.266146579872968,.26589319587097715)は変更しない。
run01はRNG/samplingに未到達なので、seedを選び直さず同じ固定recipeを用いる。
新しい公開namespaceでSTOPを迂回する計画ではなく、新SOURCEに対する別の実行許可を必要とする。

call範囲はDF preparation/config各1、218 signed sector operatorの一度の構成、
sampler/条件付きsignal/仮想shot各1,928、event位置計8,284、診断12。
新SCF/DF再生成・state solve・controlled wrapper build・compile・価格評価は0。
Full、他round、q>8、r≥2、全18round検証、重要科学レビュー、C/DやR2/R3は追加しない。
不一致なしでも科学statusはUNDETERMINEDで、統計的同等性や成功率の証明へ進めない。

許可後は空きCPU/RAM/共有負荷を再確認し、既存sector cache・固定shot chunk/worker並列処理を使う。
worker数は空きCPU/memoryに従い、job内BLAS threadだけを調整する。
恣意的なwall-time/CPU/RAM上限は追加しない。実H4速度向上の倍率は未測定。

## 出力・STOP・次担当

**新SOURCEに対するユーザーの明示許可後だけ**、実行専用worktreeで次を1回使用する。
run01は残し、fresh run02だけを使う。既存directoryへのresume/retryはしない。

```bash
python docs/research/rpe_fresh_iid_measurement_a_preparation_20261103/run_fresh_iid_a_v2.py --run --output-dir artifacts/rpe_fresh_iid_measurement_a/2026-11-03/run02
```

新SOURCE、固定入力、seed recipe、versions/context、結果をRESULT.jsonへ、
生成できたrawをshots.jsonlへ保存する。内部host/path・秘密情報を公開しない。
**成功・不一致・未判定・中断のいずれでもmandatory STOP**。
閾値・hash/218条件を緩めず、seed/N/診断規則を変更せず、追加sampling・自動retry・別候補移行を行わない。

次担当はユーザー：この公開SOURCEとrun02の1回実行を明示許可する。
Codexは許可後にそのSOURCEを固定して実行・記録・公開照合し、停止する。
過去のrun01許可はこの修正版の実行許可へ転用しない。承認の証明機構は要求しない。
