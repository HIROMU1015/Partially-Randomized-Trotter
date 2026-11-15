# C2 → 後続担当：PR-Pauli LD2 m9 完了HANDOFF

状態：**m9有限cohortの記述的比較を公開・mandatory STOP**。
H4直線1.00 Å/STO-3G/tol-only DF10、LD2/S2/step0.1/K2 normal、q512/R526/T51.2。
cohort=`D_PAULI_LD2_M9_LONG_NORMAL_N2_V1`、path0/1、X/Yは同一pathの別軸。

予測2件は観測前にpush・remote照合済み。X/YともP=`242777084125865/90343469`。
個別価格は各軸path0=2678974、path1=2680853、O=`5359827/2`。
P−Oと絶対誤差は`1328803831867/180686938`、相対と絶対相対は
`1328803831867/484225364419863`（符号付き表示**+0.274418%**）。
採択数値許容差なし、モデルPASS/FAILなし。Oは有限2 path平均でありE[C]ではない。

## 固定参照

以下のpathの共通prefixは`docs/research/d_pauli_ld2_m9_longpoint_results_20261115/`。

| 内容 | 固定commit | path / selector |
|---|---|---|
| 観測前2予測・binding・1024 seed | `3f6e6604a4877dc882fd1ee7538a854464160a5b` | `PREDICTIONS.json`、`predictions` / `binding` |
| 科学実行SOURCE | `650d8d98486c6ff479629ffa0daf93e6e976b48f` | `scripts/run_pauli_ld2_m9.py`（repo root基準） |
| raw・元4 receipt・子終了/資源 | `a7fdd200da8d499e93965d63b5b2e5b01e0b8940` | `run01/RESULT.json`、`COMPILE_RECEIPTS.json`、`PROCESS_EXITS.json`、`path*_SUMMARY.json` |
| lossless原path | `a7fdd200da8d499e93965d63b5b2e5b01e0b8940` | `run01/RAW_PATHS.json`、`PATH0.json.gz`、`PATH1.json.gz` |
| canonical有限価格・evidence・並列照合 | `c9986c82f866bfa57a7142124746f89fc42f9838` | `run01/PRICES.json`、`IMPLEMENTATION_EVIDENCE.json`、`RAW_VALIDATION.json` |
| 2軸canonical guardと結合sidecar | `adeae8f831b0cf7ffbf3448ad39b3c0ea259a655` | `run01/C_GUARD_CHECK.json`、`SIDECARS_PUBLISHED.json` |
| exact比較2件・REPORT | `adeae8f831b0cf7ffbf3448ad39b3c0ea259a655` | `COMPARISON.json`、`REPORT.md` |

各個別receiptのselectorは`p0X,p0Y,p1X,p1Y`。元QPYはreceiptの`artifacts_private`に所在・bytes/hash。
rawとexecutionのSOURCEを区別する。`RAW_VALIDATION.raw_source`の短縮`a7fdd200d`は上表のfull SHAへ解決する。
原13ab caller/core/receiptと、b74eモデル・714247 packet・採択2d513計画のfull locatorは[HANDOFF.json](HANDOFF.json)。

## 読取方法（科学実行なし）

C2 branchをfetchして任意のcheckoutから、固定SHAのJSONを読む。rawはgzip JSON、manifestのsha256を先に確認する。

```python
import gzip, hashlib, json, subprocess
B = "docs/research/d_pauli_ld2_m9_longpoint_results_20261115/"
def blob(sha, path):
    return subprocess.check_output(["git", "show", sha + ":" + path])
def read(sha, name):
    return json.loads(blob(sha, B + name))
comparison = read("adeae8f831b0cf7ffbf3448ad39b3c0ea259a655", "COMPARISON.json")
receipts = read("a7fdd200da8d499e93965d63b5b2e5b01e0b8940", "run01/COMPILE_RECEIPTS.json")
for m in read("a7fdd200da8d499e93965d63b5b2e5b01e0b8940", "run01/RAW_PATHS.json"):
    raw = blob("a7fdd200da8d499e93965d63b5b2e5b01e0b8940", m["path"])
    assert hashlib.sha256(raw).hexdigest() == m["sha256"]
    path = json.loads(gzip.decompress(raw))
print(comparison["rows"], sorted(receipts))
```

再取得・再fitの実行指示ではない。path欠測なら1 path平均へ置換しない。
本件は有効4/予定4・欠測0、preexisting再利用0、新規compile4、cache hit0、retry0。
元7 gate/bit-orderと2軸guardは成立し、技術未解決事項はない。技術受入を価格精度のPASSへ昇格しない。

## 資源・追加高速化

取得wall3825.209秒、worker wall3820.240/3806.267秒、peak RSS7,538,860/7,601,256 KiB、両exit0。
2 path並列とevent行列memo/fast fingerprintを適用した。path内X/Y・状態vectorは逐次処理が残る。
利用者の追加指示で科学SOURCEを変えずraw2 path並列照合を追加し、別PID各約4.7秒・全体7.310秒で確認。
準備4＋追加公開2の人工testはPASS。全体speedup比は未測定。詳細は[REPORT](REPORT.md)と
[追加高速化](PUBLICATION_ACCELERATION.md)。

## ロードマップへの差分とSTOP

- 「m9未検証」を「m9-X/Y独立有限cohortの記述的比較済み（2 path/4 receipt、+0.274418%）」へ進める証拠を追加。
- q512の長round証拠が1点増えた。全13round誤差上限・母平均・CI・正式E[C]・N/G・winnerは未資格のまま。
- 自然eventは各path269312件すべてorder0、非ゼロD2 aggregate phase step0。実LD2 order2 product・非自明位相は未観測UNKNOWN。
- b74eの2モデル/元4予測、旧raw・m5/m6比較/STOPは保全。旧cohortへの加算・書換え0。LD1/Full-Hへの移送0。
- 希少event調査・追加取得・再fitは保留。ロードマップ本体は統合担当が編集する。

**ここでmandatory STOP。後続の新しい科学取得・数値再解析には、対象範囲の明示指示が必要。**
