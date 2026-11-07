# Signed support H4 metadata-only：採択済み一回実行

2026-11-07 JST / SERVER1。ユーザーが契約 `3ac50a98970ad471661cba452423791c7917cae2` のmetadata-only一回を明示許可した。新しい科学SOURCEを公開・照合した後、固定2入力だけを一度処理する。grant/承認証明書は作らない。

保存linear H4/1.00 Å/STO-3G/legacy DF12/8qubit/全4電子sector参照/L_D3/m0/q1/δ=.02/r1/K2/coefficient_atol0、N1 actual finite-grid Q/selected full。軸非依存の全5層moment一組だけを取得する。

- [固定設定](EXECUTION_SETTINGS.json)：N1 commit・2入力path/hash/blob、H/partition/preparation/tail/policy、a,b、feature version、outputを固定。
- [環境/SOURCE lock](PROFILE_LOCK.json)：b23b3c0の14 import SOURCE、新しい入口・CLI・テスト・設定・人工結果を固定。実行SOURCE SHAは公開後のcommitをCLIへ明示して保存する。kernel/document commitで代用しない。
- [入口](../../../src/trottertracks/algorithm_codesign/scv_signed_h4_metadata.py)、[CLI](../../../scripts/tracks/algorithm_codesign/run_scv_signed_h4_metadata.py)、[人工テスト](../../../tests/tracks/algorithm_codesign/test_scv_signed_h4_metadata.py)。既存kernelは無変更。

## 最小接続と計測

LAW/catalog各1読込・hash検査後、signed binder1回、保存5層/label順照合、SignedControl1回、計算済みmoment一組、exactなΣ1の符号分類1回。全path列挙・CDF生成・新moment実装はない。

I/O・外側invariant検査・atomic保存/fsync・STOPのみ追加した。既存campaign.save/appendの方式を最小限に再利用し、engineをimportするcampaign本体は使わない。局所call ledgerは再実行や承認を管理するシステムではない。

既存科学kernelをpatch/改変せず、`sys.setprofile`でPython関数入口だけを観測する。内部のJSON decode、from_record、Metadata、cells、pattern_mass/joint_meanの実観測回数を記録し、契約から静的に導いた期待値と区別する。既存SOURCEを変更しない。観測中の禁止関数はbody開始前に拒否し、call超過もSTOPする。

正常経路の内部期待値は、scalar binder1、from_record1、categorical2、occurrence1、ControlBank3、scalar/signed Metadata各1、SignedControl post-init1、LAW/catalog JSON各2、cells2、pattern_mass10、joint_mean80。Componentはcatalog各行一回であり、科学的な全path数とは違う。

source/env/inputの不一致、h1正層/schema/support/moment不整合、負variance、call超過、保存不全・中断はUNKNOWN_ERRORとして部分記録とSTOPを保全する。Σ1=0はINELIGIBLE_CONSTANT_H1、Σ1>0は構造的ELIGIBLE。どの結果でもretry/resume・後続scienceなし。

## 人工検証と境界

新入口20件＋既存106件の計126件を実科学入力前に確認する。手書きのdyadic人工LAW/catalog・固定metadataだけを使用する。正負featureとexact保存、全5層、非定数/定数分類、hash/H/prep/feature/schema/label/sign/support不整合、Source/env/hash/collision preflight、観測call回数、禁止body非起動、部分記録・保存失敗・中断・STOP・再実行拒否が対象。

[人工log](ARTIFICIAL_TESTS.log)、[結果](TEST_RESULTS.json)。既存の人工回路・mock engineテストは人工入力だけで、実H4をsmoke testへ投入しない。今回の人工PASSは実H4の性能証拠ではない。

## 実行コマンドと固定出力

出力は未使用の `artifacts/scv_signed_support_h4_metadata/2026-11-07/run01/`。既存ならrun02へ変更せずSTOP。科学worker/manager/GPUは起動せず、共有venv/package/設定/他jobへ操作しない。CPU/RAM/thread上限は追加せず、実資源を記録する。

```bash
MPLCONFIGDIR=/tmp/scv-signed-H4-metadata-cache-20261107/mpl \
NUMBA_CACHE_DIR=/tmp/scv-signed-H4-metadata-cache-20261107/numba \
XDG_CACHE_HOME=/tmp/scv-signed-H4-metadata-cache-20261107/xdg \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python -B \
scripts/tracks/algorithm_codesign/run_scv_signed_h4_metadata.py \
  --source-commit <published-science-SOURCE-SHA>
```

上記placeholderは公開照合済みの新SOURCE SHAへ置き換える。コマンドを一回だけ実行し、例外や中断で科学retryしない。SIGTERM/KeyboardInterrupt等のPython例外は記録・STOPするが、SIGKILL/host消失時にfinalizer完了を保証しない。未RETURNや未保存値はUNKNOWNであり、再計算で埋めない。

SOURCE/input/result/calls/REPORT/STOPをatomic保存し、exact Fraction文字列を正本にする。N2 state・engineは参照identityのみ。Born/state作用、RNG/shot、build/compile/price、OLS/train/rank/paired/calibration、相関・残差分散・RZ改善は全0。旧224/752 bit・係数・cacheを使わない。

公開SOURCEでの一回取得後、結果と原本保全を公開照合してmandatory STOPする。Σ1>0でもfresh train/paired条件は未採択であり、別途ユーザー判断と明示許可が必要。
