# PR-LD2 m12：T2単一path実行

2026-11-13 JST。**T2はpreparation/contextのscalar binding不一致でmandatory STOPした。保存入力の復元は成功したが、normal sampling 0、生成path/event 0、native evolution/full wrapper build 0、新compile 0/2、retry 0。X/YのS/C/Dは未取得（null）。** 巨大回路の実行可能性を確認できたという結果ではない。

対象：Linear H4/1.00 Å/STO-3G、保存tol-only DF10/sector(2,2)、deterministic one-body＋prefix2/tail2…9、S2/ΔPF=.1/K2/h=.1/M12/元exactκ/P0。A a9ab固定configuration、q4096/R2559/T409.6/10,481,664 logical event。cohort D_LONG_TIME_FEASIBILITY_V1/index0。元λ/scalar・fragment順、full measured symmetric-directional X/Y、新ZZ、Qiskit1.3.0/NumPy1.26.4/rz,sx,x,cx/opt1/seed17・元compiler hashを保持する。

[PLAN](PLAN.json)／[T2_PATH_RESULT（部分STOP）](run01/T2_PATH_RESULT.json)／[RESULT](run01/RESULT.json)／[WORKER_STOP](run01/WORKER_STOP.json)／[保全確認](run01/CHECKS.json)／[実行log](EXECUTION.txt)／[技術確認](TESTS.txt)／[HANDOFF](HANDOFF.md)／[NEXT_SCIENCE_PLAN](NEXT_SCIENCE_PLAN.md)。

## 固定SOURCEと実行

科学実行SOURCEは **`8291f16ceb6b88cc8a43dbe508db419b9a26f373`**。caller、PLAN、observer追加と人工確認をcommit/pushしremote SHA一致後、一回だけ起動した。基点/固定計画は39678f5a0、A menuはa9abf8de、signal出典91349e7は再計算していない。physical ID=`f73c8ae19a25ef249f0a`、configuration ID=`1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae`。実科学SOURCE/PLAN/コードは停止後も変更していない。

既存ld12_price_requests/candidate_price_adapter、C1の保存入力loader、元prepare・normal sampler・Accounting/native compose/wrapper/fingerprint/cache/compilerを再利用するcallerを用意した。fast価格処理の変更は任意stage observerと指標抽出の計時のみ。新compiler・modelはない。新11＋既存29＝distinct40人工項目PASS。observer付き人工回路の旧新fingerprint/compiled指標/価格同値性と、scope/source/context/部分保存/無retryを確認した。実H4の同値性・性能を人工PASSから認定していない。

## 停止原因と分かる範囲

`candidate_price_adapter.NativeRuntime.context` の既存検査が次の例外を出した。

```text
ValueError: Prepared scalar differs from the requested total scalar
```

requestのexpected scalarはA固定MENU/tail descriptor由来の **0.2223759094107018**。runtimeは元保存Hからprepareした後の `ham.constant + prep.extracted_identity_coefficient` を使い、両者の`float.hex()`完全一致を要求する。この検査は元D748からの条件で、今回追加・緩和したものではない。

SOURCE/read-only provider検査、保存snapshot/receiptのbytes/SHA、元loaderのH/state/sector/array identity検査が通過した。runtimeのLD2/8wire/H hash/fragment順/DF10と正λ・binary64 λ照合もscalar gateより前にあり、例外はそこを通過した後である。ただしctxがreturnする前に停止したため、preparation/component/basis/tailの完成inventoryと実scalar値は公開保存に到達していない。

**実scalarの値・差の大きさはUNKNOWN。** 既存例外は比較した値を含まず、worker終了後に再preparationしていない。1 ULP等の丸め差か、構成・scalar定義の差かを今回確定できない。差を科学的に無害と仮定せず、expected scalarの置換、tolerance化、guardの無効化、sampling/compileの先行を行わなかった。

## 取得できた時間・メモリ

| 観測 | 結果 | 限界 |
|---|---:|---|
| 全pipeline wall（preflight開始〜親の結果保存直前） | 4.694852 s | 成功pathの時間ではなく、この失敗attemptの時間 |
| preflight（SOURCE/input bytes等） | 2.004265 s | metadata/bytes確認 |
| input復元 | 2.306109 s | scientific import・保存array/state照合を含む |
| context構築 | worker開始後2.349018 sに開始、worker STOP記録時2.643402 s | 正常終了時刻/完全なcontext stage時間は未取得 |
| worker peak RSS | 354,508,800 bytes（354.509 MB） | ru_maxrss、1 MB=10^6 bytes。full build/compile peakではない |
| 親processの観測high-water | 30,351,360 bytes | workerと加算した同時peakではない |
| sampling/event変換・保存/S/native/hash/wrapper/compile/metrics | 未着手・未測定 | 0価格/0所要時間という意味ではない |

resource STOPではなく科学的binding gateによる停止だった。ホストMemAvailableは実行前約488.3 GBで、10秒monitorの初回観測とstage/停止時RSSを保存した。短時間終了のためresource journalのworker観測は初期値のみで、peakはworkerのru_maxrssを使う。m6又は今回の短い失敗時間からm12の実性能を外挿していない。

| 軸 | C | S | D | fingerprint / compiled metrics / cache |
|---|---|---|---|---|
| X | UNKNOWN | UNKNOWN | UNKNOWN | NOT_REACHED |
| Y | UNKNOWN | UNKNOWN | UNKNOWN | NOT_REACHED |

10,481,664は計画metadataの位置数で、処理済みevent数ではない。seed rule/list digestはPLANに固定したが、samplerは未呼出しなのでevent digestはない。cache照合も未着手で、hit/missを取得済み0件とみなさない。

## 保全と結論

SOURCE、private snapshot/receiptの停止後SHAは開始前と一致。旧private event/cache、過去のSOURCE/結果/input/KEEP/grant/ledger/marker/科学契約/STOPは変更しない。新private directoryは作成したがpath生成前で空のまま保全した。完了したX/Yファイルはなく、部分結果はmetadata・stage/資源・例外である。

今回はLD2 scalar整合が長時間取得より前の技術的障害と判明した。full compileの大規模性能・実行可能性は未測定のまま。次の最小案は**新しい承認範囲のpreparation-only診断1回、sampling/build/compile 0**で実scalarと元定義を記録すること。今回のunused2 compile枠は再利用しない。mean/CI・全law資格・fit/outer-window校正・他round外挿・N/RMSE/G/winnerは未取得。最終commit/push・remote照合後 **mandatory STOP**。
