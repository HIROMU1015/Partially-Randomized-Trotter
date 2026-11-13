# compiled価格pipelineの高速化：技術引継ぎ

2026-11-13 JST。[REPORT](REPORT.md)／[P2提案](P2_MINIMAL_PLAN.md)。
**人工検証済みのopt-in API。実H4の新build/compileは未実施。**

## 実装と再利用

追加は[src/trottertracks/compiled_price_fast.py](../../../src/trottertracks/compiled_price_fast.py)。
`make_backend_class(base, science)`で固定C1のBackendを継承し、`price_requests`だけを拡張する。
`context`の入力/準備/SOURCE照合、`acquire`/`acquire_many`等は継承する。
H4の元Accounting wordだけを対象とし、H6へ暗黙に適用しない。

構造価格：

- `price_accounting(Accounting)`は元classを継承し、native/folded counterをRZ-only projectionへ置換。
  NativeInstructionの型・arity・有限値・wire検査を保持し、使用しないdepth frontierを更新しない。
- gate expansion、basis対相殺、pending-Zの足し算順、literal zero、相対phaseは元メソッドのまま。
  `finish()`の全出力も人工テストで旧Accountingと一致する。
- `structural_pair`はHから本体・scalarまで一度走査し、**finish前**の状態をdeepcopyして
  X/Yそれぞれのsuffixを評価する。pending basis/Zを途中でflush/resetしない。
  copy対象はcounter・pending状態・featuresで、event全列やQuantumCircuitではない。

回路・hash・compile：

- carrier `155529e20a8fac39e00786c40b57830b71203c4f`のcomposeを固定C1 loaderからそのまま使う。
  providerのtail builder/PF/native構築・wrapperも変更しない。
- 既存[scripts/h46_b3_fast_fingerprint.py](../../../scripts/h46_b3_fast_fingerprint.py)の
  `use_fast_fingerprint`を再利用。今回そのfileを編集しない。
  policy=`qiskit_recursive_numeric_circuit_v2`と最終SHA256のcanonical bytesは不変。
- 現在のoperation/parameter/definition/control/bit配置/global phaseをcallごとに調べる。
  object IDだけの永続cacheや事後の不変性の仮定でmutation checkを省略しない。
  alias差替えは単一threadのprocess内だけで、例外時も元へ戻る。
- `build_native_hadamard_wrapper(include_measurement=True)`による両軸を個別にbuildし、
  各軸でfingerprint→cache照合またはcompile→再fingerprintを実施する。
  新loweringの有効化を要求し、X/Yの同価格からcompileを省略しない。
- compiler設定はQiskit1.3.0、rz/sx/x/cx、opt1、seed17。
  `transpile_and_measure_cost`を変更せず、元の6指標を取得する。
  C1同様、全pathのprocess並列と競合するnested transpile poolは使わない。
- cache filenameは元wrapper SHAのまま。cache内fingerprint/compilerが違えばfallbackせず失敗。
  旧cacheは読み取り専用で、新しい永続cache・共有設定・auth機構は追加しない。

`events_PRIVATE.json`、`STRUCTURAL.json`、軸別started/metrics、resultの形式はC1を維持する。
追加の`TECHNICAL_TIMING.json`は新規の取得directoryにのみ書く。
compose/wrapper時間には内部fingerprint・走査が含まれるため、別の内部profileを加算しない。

## 接続例（構築だけ。今回は科学実行しない）

プロジェクトと既存providerを使える同じ専用環境で、`src`と`scripts`をPYTHONPATHに含める。
runtime/provider位置とNumPy1.26.4/Qiskit1.3.0をC1 contextが検証する。

```python
from trottertracks import p1_efficient_zz as P
from trottertracks import compiled_price_fast as F

modules = P.load_c1_backend(root)  # pinned 1a87e533, no molecular input loading
science = modules['r_price_phase1_science']
Backend = F.make_backend_class(science.Backend, science)
backend = Backend(root, execution_plan)  # lazy constructor
```

将来の承認済みrunnerでは、元contextを確立し、provider moduleに
`efficient_controlled_zz(df_rte_qiskit)`を有効化した範囲内で
`backend.price_requests(cell,index,new_directory,requests,seeds,laws,conditional=False)`を使う。
既存P1 SOURCE/runner/PLANを改変・再実行する方法ではない。
新しい科学SOURCEと有限範囲の固定はその実行の前提で、今回その許可はない。

## 同値性の確認内容

[新テスト](../../../tests/test_compiled_price_fast.py)は分子ファイルとsamplerを禁止し、
[人工fixture](../../../tests/compiled_price_artificial.py)を使う。
任意の8-wire行列・明示event列を使い、H4 Hamiltonian・保存trajectoryは使わない。
8 system wireは元Accountingの固定control=8と整合させるためである。
Gaussian loweringとcompose/scheduleは元carrier関数を固定blobからロードする。

9項目で以下を確認した。

- 正/負時間、scalarあり/なし、Z/ZZ、order0/2/product、basis reuseを含む構造出力の完全一致。
- 同じrequestからの旧/新Backendで、event bytes/digest、S、X/Y wrapper fingerprint、
  compiled回路の再帰fingerprint、compiler設定hash、測定1個、全6価格指標が一致。
- full-wrapper canonical bytesの一致、phase変化の検出、evolution変更の拒否。
- cache hitの個別再利用、compiler不一致時の失敗、compileによる入力改変の検出。
- lowering非有効時とrequest配分不一致時の拒否、counterのwire/type検査。

既存lowering10項目・fast hasher8項目・残差adapter13項目・mean price15項目を再利用。
残差adapterの「科学library未import」検査は専用processで実行する。
同じpytest processにcompileテストを混在させた初回の失敗を
[TEST_COMBINED_IMPORT_CONFLICT](TEST_COMBINED_IMPORT_CONFLICT.txt)に保全した。
import状態の検査が理由であり、旧テストを変更・無効化していない。

## 人工再現コマンド

環境：`/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python`。
専用providerが前述runtime位置に存在すること。出力先は未使用のパスにする。

```sh
PYTHONPATH=src:tests:scripts python -m pytest -q --disable-warnings tests/test_compiled_price_fast.py tests/check_efficient_controlled_zz.py tests/test_h46_b3_fast_fingerprint.py
PYTHONPATH=src:tests:scripts python -m pytest -q --disable-warnings tests/check_compiled_residual.py
PYTHONPATH=src:tests:scripts python -m pytest -q --disable-warnings tests/check_mean_compiled_price.py
PYTHONPATH=src:tests:scripts python scripts/benchmark_compiled_price_artificial.py --output /tmp/d-price-artificial-new.json
```

benchmarkはbaseline/accelerated各3回×2サイズ、毎回fresh spawn processで交互に実行。
自分のbenchmark同士の資源競合を避けるため順次測る。CPU affinityやRAM上限は設定しない。
wallはimport/fixture構築を除いたprice取得、RSSはimport/fixtureを含むprocess lifetime peak。
段階別RSS peak・実H4倍率は未測定。tracemallocなし。
人工fixtureの明示order2は性能・技術テスト用で、物理lawの頻度を模擬しない。

## 残る負荷と適用限界

元composeはpieceごとにtailとpartial evolutionを生成し、そのfingerprintを計算してから全体へcomposeする。
wrapperごとにも元evolutionのgate走査/hashがある。これらの検査を省略していない。
Qiskitの既定copy、tail内basis append、event.to_dict/JSON出力、全回路の保持・走査は残る。
既存fast writerもcallごとのO(L)走査が必要で、常に全体が小さな定数メモリになるわけではない。

X/Yのprocess並列は将来の大pathでは候補になるが、evolutionの受渡し・複製とRSS増を伴う。
まずpath間並列を使える既存C1の仕組みを維持した。今回X/Y並列化やgate emitterの別実装は追加しない。
人工で十分な改善が得られたため、実測のないcopy=False置換やintermediate hash省略へ開発を広げない。
次に必要な技術証拠はP2提案内の実移送・実測であり、今回はmandatory STOP。
