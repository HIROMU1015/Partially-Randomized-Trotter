# Cへの引渡し：Pauli LD1/m0 B0 run02

**実接続PASS、有限1 pathのみ。B1・law平均・N・Gには未着手、mandatory STOP。**
実行SOURCEは`9abbcc563aaa8a2650a5f799bc394f67d9c42534`、raw結果は
`8502dd33ad16dc012fb3558668ef3e4cd8a4405c`に固定した。
[復旧REPORT](RECOVERY_REPORT.md)に条件、人工test、全数値gate、性能を記載した。

## 使用する公開データ

- [PRICES.json](run02/PRICES.json)：元C canonical identityを保持したX/Yの直接価格record。
- [SIDECARS_PUBLISHED.json](run02/SIDECARS_PUBLISHED.json)：上記価格と組み合わせるsidecar。
- [IMPLEMENTATION_EVIDENCE.json](run02/IMPLEMENTATION_EVIDENCE.json)：raw結果commit固定、selector `m0X` / `m0Y`。
- [C_GUARD_CHECK.json](run02/C_GUARD_CHECK.json)：Cの既存guardの実record metadata確認。追加科学call 0。
- [RESULT.json](run02/RESULT.json)、[SOURCE.json](run02/SOURCE.json)、[CHECKS.json](run02/CHECKS.json)、
  [RECEIPT_TRACE.json](run02/RECEIPT_TRACE.json)：数値gate、入力/実行SOURCE、fingerprint、private archive所在。

rawの`SIDECARS.json`はproducerが出力したpublish前placeholderのまま保全している。
consumerは`SIDECARS_PUBLISHED.json`を使用する。変更は証拠commit locatorだけで、price/physical context/axis/cohortは同一。
raw結果commitにあるproofを読み、Cの`price_implementation_guard.check`がm0X/m0Yを受け入れた。
guard関数を差し替えたり、mock proofで実recordを通したりしていない。

## identity・quantity

元configurationは`d905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27`、
physical IDは`RMSE:62a9a6b68c2c88951f1055d1da7f9b6a89da94a91092c9488cf5fce503288e54`。
B新Z1＋T1/Q3 IDへの付替えは禁止。同じPauli LD1/m0/T.1/qPF1/R3/K2、cohort
`D_PAULI_LD1_M0_WIRING_V1`、index0、seed`14788159701244282917`である。

classification=`DIRECT_COMPILED_FINITE_SAMPLE`、expectation=`FINITE_COHORT_ONLY`、
n_paths=1、diagnostic.law_expectation=false。既存`FULL_WRAPPER_MEAN_COMPILED_PRICE` quantityおよび
`mean.rz_count` fieldはschema上の入れ物であり、法則平均や平均の信頼区間を認定していない。
X/Yは同一trajectoryのpaired応答で、独立な2 pathとして数えない。
両軸1380 RZ、578 CX、depth834。Sと残差Dは未取得。

route=`PR_PAULI_TAIL`、lowering=`PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`を必須とする。
DF controlled-ZZ用loweringと混在させない。compiler profileはQiskit1.3.0/rz,sx,x,cx/opt1/seed17、
symmetric_directional、P0、full_Hadamard_with_measurement。
scalar quantityは`HAMILTONIAN_CONSTANT_PLUS_SAVED_DF_TAIL_IDENTITY_sR`、
resolved/expectedとも2.7108187593826942、一回適用。回路中のhead/tail順、signed time、測定も元callerのままである。

guard用policy `PAULI_B0_IMPLEMENTATION_CHECK_ONLY`はこの実装照合専用で、
方式間の正式比較policy・価格精度・confidenceの新採択ではない。
測定descriptor内のlaw_contextはCの既存ideal-law templateを保持するが、
価格recordのlaw_contextはsaved_samplerを使用し`ideal_sampler_distance_qualified=false`である。
template中のtrueを実sampler-law距離の資格証拠と解釈しない。

## private原本と保全

原本は`/home/AbeHiromu/projects/d-pauli-ld1-m0-20261114-private/run02/`にある。
`events_PRIVATE.json`、`child.json`、`job.json`、compiled_cacheの各receiptとwrapper/compiled QPYを保全した。
公開CHECKSのpath・bytes・SHAで対応する。X/Yのfingerprintとmeasurement末尾を確認済み。
旧run01はb5fc2f97d3ab31fb4e1648c8a77eed051d378491に保存され、raw/private jobのSHAは不変。
今回の手動復旧一回と旧run01の失敗を混合せず、SOURCE・入力・既存STOPを保持する。

## 次の担当と未取得証拠

- **Codex C**：上記record/sidecar/proofを有限sampleの実装証拠として参照する。law平均/model資格へ昇格しない。
- **ユーザー**：B1、追加path/round、平均価格model検証の要否を判断する場合は、対象・seed/cohort・SOURCE・出力先・有限科学call範囲を別途明示する。
- **Codex D**：今回の公開・照合で停止。追加sampling/compile、再試行、mean/CI/fit、N/Gは実行しない。

未取得：law平均/CI、rare寄与、ideal lawとsampler lawの差、別構成/長時間への移送、全RPEの価格とG。
