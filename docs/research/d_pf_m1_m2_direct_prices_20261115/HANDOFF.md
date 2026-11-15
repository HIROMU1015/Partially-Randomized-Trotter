# C向け：決定論PF 6候補のm1/m2直接価格

**24/24軸有効。m1/m2各12軸、各round再利用0・新規実compile12・有効12・保留/欠測/UNKNOWN0。全作用gate・C implementation guardを通過し、直接証拠coverageは72/156→96/156。mandatory STOP。** モデルのPASS/FAIL・2%判定は行っていない。

正本は[run01/AXIS_PRICES.json](run01/AXIS_PRICES.json) の24行（CSVも同名）。m0のAXIS_PRICES形式を使い、`(configuration_id, physical_id, m, axis)` でCの固定alignment `0b4b2d941cd669aaa45231e1236bc2d3158a9d12:docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json` とjoinする。各行にalignment selector、T/q/actual幅のbinary64 hex、取得区分、RZ・6指標、fingerprint、gate状態、trial/receipt/sidecarの固定selectorがある。scope外の行を合成しない。

保存linear H4 / 1.00 Å / STO-3G / tol-only DF10、Full一体項＋fragment0..9 / B0I p7一体項＋fragment0..6、元H/state/snapshot、P0除外。m1は全公式T=.2/q1/actual幅.2、m2はS2 T=.4/q2/actual幅.2、Y4/M5 T=.4/q1/actual幅.4。S2のm1/m2は同幅でも別ID・別wrapper・別total scalar phaseであり、価格を共有しない。

比較quantityは、元symmetric_directional制御進化とX/Y基底変換・最終測定・scalar位相を含むfull controlled Hadamard wrapper全体のcompiled RZ数。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17。Fullはc_H、p7はc_H＋tail identityをtotal Tに一回。元native / signed piece / fragment順 / scalar定義 / wrapper / compilerを変更していない。

## 固定locator

| 種類 | commit | path |
|---|---|---|
| 実行SOURCE | `a3cbe70e4afb0948188403f48cf4086bb419d8bf` | `scripts/run_pf_m0_prices.py`（m1/m2限定拡張）、元Full/p7 caller / helper |
| Full raw | `bf8a7f0428a7efd45c3888e288b3e2095f5e4c34` | `docs/research/d_full_pf_shortlist_prices_20261114/m1_m2_direct/run01/RESULT.json` |
| p7 raw | 同上 | `docs/research/d_truncated_p7_t3_prices_20261114/m1_m2_direct/run01/RESULT.json` |
| Full bound | `eea0bfab986221040165f2cc9c5f81cc094023f1` | `docs/research/d_full_pf_shortlist_prices_20261114/m1_m2_direct/run01/{PRICES,SIDECARS,GUARD_CHECK}.json` |
| p7 bound | 同上 | `docs/research/d_truncated_p7_t3_prices_20261114/m1_m2_direct/run01/{PRICES,SIDECARS,GUARD_CHECK}.json` |

sidecarのimplementation proofはraw公開commitに束縛してある。native policyは `H46_LONG_TIME_NATIVE_COMPOSE`、RTEは `NOT_APPLICABLE_NO_RTE`。C guardの固定SOURCEは既存c628e9e8447c11c984064b922cbce6c52dc72d48、policy ID `PF_M1_M2_DIRECT_IMPLEMENTATION_ONLY_V1` は元profile / native / scalar条件のimplementation確認用であり、新しい科学的精度基準ではない。compiler hashだけを同一性根拠にしていない。

価格recordは `DIRECT_COMPILED_DETERMINISTIC` / `DETERMINISTIC`、`EXACT_DETERMINISTIC_CIRCUIT_PRICE`、`diagnostic.model_qualified=false`。元index0の7作用gate・許容差を全12 cellで保持し、X/Yを別compileした。価格が一致した軸・roundも別receiptのまま受け取る。片軸補完、cache-only価格、科学retry、MC平均化は0。

再現はREPORTのmetadata checkerを使う。現在のcallerは元m0 SOURCE729fe5370から有限scopeを拡張した新SOURCEであり、旧SOURCEと旧結果を再ラベルしていない。m0のSOURCE同一性checkerは元c95353a44 checkoutでの検査として保持する。今回の検査は新SOURCEとm1/m2の公開receiptへ束縛する。

## 資源・coverage・ロードマップ差分

12物理cellをFull/p7各6 workerで並列実行し、contextはroute内で共有した。取得wall102.512 s（binding・報告・公開は別）、route wall Full99.738 / p7 73.782 s、最大worker RSS576.74 MiB。child/worker全終了code0・signalなし。開始時idle CPU約63.84 / RAM約489 GiBから配置し、共有設定・他processへ変更なし。段階時間・観測は[run01/TIMING.json](run01/TIMING.json)、[取得概要](run01/ACQUISITION.json)とroute別progress JSONL。速度倍率は未測定。

統合担当へ：**m1に12軸、m2に12軸の有効な直接証拠が増加した。6候補のm0～m6が直接価格で揃い、既存m9を含むcoverageは96/156。モデル追加判定は未実施。** これは旧m3の現在consumer向けmetadata binding等を自動完了にしたものではない。Cは今回の24行だけを既存表へoverlayし、保存m0・Stage1/2・m6/m9・元凍結JSONを変更しない。

m7/m8/m10～m12の60軸、モデル適用domain・誤差、MSE/N/G/winnerは未評価・未資格のまま。取得価格を別roundへ移送したり、m6/m9の2%規則を短roundへ適用したりしない。ロードマップ本体の更新は統合担当へ渡す。今回namespace / raw / cache / QPY / log / STOPを保全し、commit/push・remote照合後**mandatory STOP**。追加科学計算へ自動進行しない。
