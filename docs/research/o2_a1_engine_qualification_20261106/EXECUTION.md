# A1 N2総engine誤差：許可された一回の限定数値認証

2026-11-06、SERVER1 GPUA100-gleap-01。ユーザーが固定24d6bdfa11ca5964c83b2639f060f4655ab1d4beの`qualify_fixed_bundle`一回を明示許可した。
保存H4/1.00 Å/STO-3G/legacy DF12/8qubit/全4電子sector/L_D3、m0/q1/δ.02/r1/K2/coefficient_atol0、X/Y、N1 actual Qとselected policy、N2保証ground、理想gate・保存DF限定。
旧計画・SOURCE・結果・STOPは変更しない。1/32の採択、N3、Control/pilot変更を含めない。

[固定kernel](../../../src/trottertracks/algorithm_codesign/a1_engine_connection.py)とPROFILE_LOCKの全538 SOURCE・依存環境を保持する。
[最小runner](../../../scripts/tracks/algorithm_codesign/run_a1_engine_qualification.py)は旧kernelを一回だけ呼び、禁止されたsolve/eigh/state作用/Born/path/compile等をfail-only guardで拒否する。gate計算や入力値はpatchしない。
旧kernelは前回計画の表示上界0.000000781を使う。今回の明示指定に従い、そのraw `UNIFORM_ERROR.json`を保全した上で、同じ保存EとN2 `STATE_RESULT.json` の `coordinate_state_error_upper.exact` を一度だけ合成し、`ENGINE_ERROR_CERTIFIED.json`を最終証明とする。
再qualification・state誤差の再計算ではない。新runnerのhashも入力前に公開固定する。

一回のbundle復元、D half-step4局所列とscalar列、保存basis/inverse・RTE/wrapperの全局所gate集合、unique key毎一回のinterval検査、一様δ*/L_max/Eの合成を許可する。
NPZ load・preparation/CDF・solve/eigh・実state norm/作用・実Born評価・実path build/列挙・RNG/shots・train/cal/prod・compile/variance/N/費用は0件。
不整合・未対応gate・非有限interval・E≥1・出力衝突でSTOPし、SOURCE変更や再試行で救済しない。

旧private cache設定をそのまま使い、共有venv/package/cacheや他jobは変更しない。CPU/RAM/workerの人工上限を新設しない。

```bash
MPLCONFIGDIR=/tmp/a1-engine-connection-cache-20261106/mpl \
NUMBA_CACHE_DIR=/tmp/a1-engine-connection-cache-20261106/numba \
XDG_CACHE_HOME=/tmp/a1-engine-connection-cache-20261106/xdg \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <existing-opt2-python> -B \
scripts/tracks/algorithm_codesign/run_a1_engine_qualification.py \
 --output-dir artifacts/o2_a1_new_target/2026-11-06/engine_qualification_run01
```

fresh出力を要求し、外部の一回invocation記録を結果へコピーする。SOURCE・gate証明・厳密総上界・STOP・REPORTを保全・commit/push・remote照合後mandatory STOPする。
