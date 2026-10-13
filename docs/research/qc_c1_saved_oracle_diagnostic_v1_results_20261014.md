# C1固定保存oracle・selector診断 v1：結果と独立監査（2026-10-14 JST）

利用者の今回の明示承認により、公開済み固定sourceと実行契約を使った保存データ診断を一回実行した。通常stateの16 contextすべてで、未選択24macrocycleを含む保存eligible候補のoracleにも登録5%費用優位はなかった。bad-state controlの16 contextはeligible候補不足により未判定。独立checkerは32 context PASS。C1全般の不能性、研究の継続・終了、新規性・C4移行は判断していない。

## 認可・固定identity・scope

- Repository `HIROMU1015/Partially-Randomized-Trotter`、branch `qc-c1-saved-oracle-execution-20261014`。
- 準備commit `e82a5202098ad95f2eb7280279d2c97cc06190bc`、固定source `657513ad06661aed84950d5907f3304dcdd5f1f7`、旧研究base `cc992b579290233ab5a7d8408d4a81c6d295f36d`。
- [今回の新規認可](../../artifacts/qc_c1_saved_oracle_execution_v1/2026-10-14/authorization.json)、実行前認可公開commit `a7139218`。[旧準備target](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/approval_target.json)は未発行の準備履歴として不変。消費済み940-call認可は流用していない。
- [実行計画](../../artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14/execution_plan.json)、[固定数学・比較契約](qc_c1_saved_oracle_diagnostic_contract_v1_20261014.md)、[preflight](../../artifacts/qc_c1_saved_oracle_execution_v1/2026-10-14/preflight.json)、[実launch receipt](../../artifacts/qc_c1_saved_oracle_diagnostic_v1/2026-10-14/attempt_1/launch_receipt.json)。
- 非分子2qubit Pauli Hamiltonian 4入力、24macrocycle、6h（h=1/(8Λ2^s),s=0…5、Λ=fragment係数L1）、通常state |00>16/bad state |11>16、energy target10^-5/10^-6 model units、controlled価格3/6。geometry/basis/DF rank-policy/L_D/固定粒子sectorは非該当。
- single/subdivision12、macro24、S4三、C2三の保存42methods×6h×32contexts=8,064行。branch/overlap/precision eligibility、保存費用、selector選択とprecision情報は変更・再生成しない。

## 指標と厳密な判定

C_Mは保存eligible macro24の最小費用、C_Bはsingle/subdivision/S4/C2の強い対照の最小費用、C_Kは同じC1catalogのsingle/subdivision12+macro24の最小費用。macro_ratio=C_M/C_B。method_regret=C_saved_selector/C_K−1は、選択method内のhが既に事後oracleで最適化された限定的なmethod指標であり、運用可能なprecision選択のregretではない。jointの保存precision選択はmetadataとしてコピーした。

保存binary64費用を有理数へ厳密に写し、最小値・比・regretを計算した。5%gateはstrict 19/20。費用ごとの隣接1ULP区間は丸め感度のみを扱い、物理的誤差・eigensolver誤差・費用モデル不確かさを保証しない。tiesは保存順最初と全同費用行pointerを保持。nullを0や無限大へ置換していない。

## 結果

| 事前固定分類 | 通常16 | bad-state16 |
|---|---:|---:|
| A：oracleにも5%優位なし | 16 | 0 |
| B：oracle優位あり、selectorが未選択 | 0 | 0 |
| oracle優位がknown compositionと同値 | 0 | 0 |
| eligible不足・境界・保存不足による未判定 | 0 | 16 |

通常stateのmacro_ratioは1.082687339〜5.331029315。全16件で1を超える。5%gate付近の数値境界例はない。bad-state16件ではC_M/C_B/C_Kと各selectorのeligible点がなく、比・regretはnull。これは優位なし16件の証明に加算しない。

通常stateの明細（全行分類A、exact費用・ratio・regret・row pointer/tieは[一次結果](../../artifacts/qc_c1_saved_oracle_diagnostic_v1/2026-10-14/attempt_1/diagnostic.json)）：

| 入力 | 価格 | target | C_M/C_B | C_M method,h index | C_B method,h index | regret empirical / joint / box |
|---|---:|---:|---:|---|---|---|
| S1_weak | 3 | 1e-05 | 3.248057548 | M01_1_3, 0 | S0_m1, 0 | 2.57363688 / 2.57363794 / 0 |
| S1_weak | 3 | 1e-06 | 1.082687339 | M01_1_3, 0 | S0_m3, 0 | 1.38242836 / 0.191214286 / 0.333333233 |
| S1_weak | 6 | 1e-05 | 3.255703285 | M01_1_3, 0 | S0_m1, 0 | 2.61186555 / 2.61186662 / 0 |
| S1_weak | 6 | 1e-06 | 1.085235921 | M01_1_3, 0 | S0_m3, 0 | 1.40791418 / 0.203957196 / 0.333333233 |
| S2_balanced | 3 | 1e-05 | 1.624029606 | M01_1_3, 1 | S0_m4, 0 | 0.131780117 / 0.786819482 / 2.33778406e-16 |
| S2_balanced | 3 | 1e-06 | 2.659255041 | M01_1_3, 2 | C2_0, 0 | 0.509043577 / 0.19121436 / 0.333333231 |
| S2_balanced | 6 | 1e-05 | 1.627852476 | M01_1_3, 1 | S0_m4, 0 | 0.159814452 / 0.805933826 / 2.75410177e-16 |
| S2_balanced | 6 | 1e-06 | 2.665514771 | M01_1_3, 2 | C2_0, 0 | 0.546422776 / 0.203957271 / 0.333333231 |
| S3_signed | 3 | 1e-05 | 1.624029602 | M01_1_3, 1 | S0_m4, 0 | 4.57645186e-16 / 4.57645186e-16 / 4.57645186e-16 |
| S3_signed | 3 | 1e-06 | 5.318509856 | M01_1_3, 3 | C2_0, 0 | 1.46446674e-15 / 1.46446674e-15 / 1.46446674e-15 |
| S3_signed | 6 | 1e-05 | 1.627852472 | M01_1_3, 1 | S0_m4, 0 | 4.04357733e-16 / 4.04357733e-16 / 4.04357733e-16 |
| S3_signed | 6 | 1e-06 | 5.331029315 | M01_1_3, 3 | C2_0, 0 | 1.29394664e-15 / 1.29394664e-15 / 1.29394664e-15 |
| S4_strong | 3 | 1e-05 | 1.624021436 | M01_1_3, 1 | S0_m1, 2 | 0.852695202 / 0.976725701 / 0 |
| S4_strong | 3 | 1e-06 | 2.659074668 | M01_1_3, 2 | C2_0, 0 | 0.235141051 / 0.317828434 / 0.333332605 |
| S4_strong | 6 | 1e-05 | 1.627844287 | M01_1_3, 1 | S0_m1, 2 | 0.885826451 / 1.0136798 / 0 |
| S4_strong | 6 | 1e-06 | 2.665333974 | M01_1_3, 2 | C2_0, 0 | 0.257228746 / 0.342464711 / 0.333332605 |

regretには大きなmethod選択の差と、binary64保存値の微差による約10^-16〜10^-15の値が含まれる。後者を科学的なselector失敗の大きさと解釈しない。selectorのcatalog内regretがあっても、今回の通常16contextには強いC_B対照を5%以上上回るmacrocycleがないため、分類Bではない。

## 実行と独立監査

[保存一次結果](../../artifacts/qc_c1_saved_oracle_diagnostic_v1/2026-10-14/attempt_1/diagnostic.json)、[独立監査](../../artifacts/qc_c1_saved_oracle_diagnostic_v1/2026-10-14/attempt_1/independent_audit.json)、[報告用集約](../../artifacts/qc_c1_saved_oracle_execution_v1/2026-10-14/result_summary.json)。報告集約は診断outputだけを読み、元comparisons/sealを再処理しない。

worker1/thread1、stdlib JSON/Fraction、GPU・数値libraryなし。CPU利用可能32、事前memory約55GB、固定memory上限256MiB。短い保存表の走査・有理数比較が主処理で、固定sourceとscopeを守りserial実行した。速度改善倍率は主張しない。
runner全体0.18秒/peak RSS27,684KiB、checker全体0.07秒/peak RSS26,116KiB。[runner resource](../../artifacts/qc_c1_saved_oracle_execution_v1/2026-10-14/runner_resource_use.txt)、[checker resource](../../artifacts/qc_c1_saved_oracle_execution_v1/2026-10-14/checker_resource_use.txt)。独立checkerは別のpool sortとcross multiplyを用い、診断の評価関数を呼ばず、入力SHAと32context/8,064保存行、最小値pointer、tie、ratio/gate、sealed-method frontier、regret/classificationを検査した。独立実装監査であり外部科学再現ではない。

attempt1/retry0、独立audit1。eigensolver/PF/compileはいずれも追加0回。出力2MiB、log1MiB、memory256MiB上限を満たす。詳細保存総量・保全照合は[static integrity](../../artifacts/qc_c1_saved_oracle_execution_v1/2026-10-14/static_integrity.json)に記録する。新しい人工test・数学探索・一般文献調査は実施せず、固定sourceで完了済みの34人工test結果を参照した。

## 支持範囲と未解決事項

この結果は固定catalog・6点grid・保存eligibility・条件付きHeisenberg費用proxyの範囲で、通常16件の未達を「有利なmacrocycleをselectorだけが見逃した」とは説明できないという診断を支持する。grid外、catalog外、他Hamiltonian、異なる価格・精度、改善後のselectorには移送しない。
既知compositionとの同値性、真値なしの選択、近似state情報取得費用、分子Hamiltonianへの移送、QPE/RPE全体費用、独立新規性は解決しない。全基底エネルギー総費用はUNKNOWN。C1全般の数学的不能性も証明していない。

## 保全・引継ぎ・停止

固定source16/input25、旧science source20/一次9、既存証拠・準備計338files、旧manifest140entries/non-result fields、root既存131pathsのbyte/status、global manifestの既存description欠落FAILを保全する。旧結果/source/認可/STOP、N1 STOP・N2 HOLD・N3 CLOSED・THRIFT現template終了を変更しない。
次は[GPT判断資料](qc_c1_saved_oracle_diagnostic_review_handoff_v1_20261014.md)を使う重要科学判断の別開始承認。研究の正式継続・終了・再設計/C4移行をCodexは決めず、自動追加処理を開始しない。GitHub公開とremote再取得照合後Mandatory STOP。
