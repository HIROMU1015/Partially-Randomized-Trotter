# O2-S2 複数event有限table評価：途中停止報告

2026-11-06 JST。専用branch：`codex/o2-s2-multi-event-validation-20261106`。

**科学評価は技術的転送エラーでSTOPした。3,969 pair価格と元law P0の2cellは保存済み。qA/qBの4cellは未取得で、sampling変更の追加利益は判定できない。** 不利なIS結果やT=0による停止ではない。六cell完了とは報告しない。

科学実行SOURCEは [16d10faabb1f6aa4bda42f3aef81d89bf9ca9832][EXEC]。設計正本は [5eb90e8241e69e85a1ca3e79938a367700c24a48][DESIGN]。今回のユーザー指示がこの固定条件の最小接続・人工確認・限定数値評価・公開を許可した。旧研究の実行許可やSTOPは流用していない。

## 1. 固定条件と出力

3-system-qubit synthetic operator `R=(1/5)Q0+(3/10)Q1+(1/2)Q2`、`Q0=Z0`、`Qi=Vi† Zi Vi`、`V1=R_XX01(π/4)`、`V2=R_XX12(π/4) R_ZZ01(π/4)`、`R_P(θ)=exp(-iθP/2)`。D1保存row 10／D2 Closed P5、degree5、r=2、τ=5/7、x=5/14、σ=+1、state=`000`。有限targetは `[P5(-i(5/14)R)]²`、時間順i→j、行列順VjVi。分子geometry/basis/DF rank/`L_D`/δ窓は非該当。

ε_axis=1/200、α_axis=49/24000、H=160、K=256、η=ρ=10^-12、primitive error=10^-6、interval dps指定100。科学的数値は厳密有理数と既存sqrt/log enclosureを使い、浮動小数点は時間・表示だけ。旧D2 state-uniform biasとremainingはD1保存値との一致を確認した。D3保証/合成case変更なし。六cellは代替forecastの比較であり、同時familywise実験ではない。

| 成果物 | 内容 |
|---|---|
| [SOURCE.json][SOURCE] | 固定入力commit/path/hash、旧SOURCE、実行コードhash、条件、価格境界・STOP |
| [RESULT.json][RESULT] | STOP理由、保存済みP0のM2/L/N・費用、setup時間とメモリ、未測定項目 |
| [pairs.json][PAIRS] | 全3,969 ordered pairのi/j、Aij、元P0/W0、加法・境界T/CX/1Qと相殺depth。IRは複製せずD1固定binding参照 |
| [law_P0.json][LAW] | 元の実joint lawと係数・weight。元lawをcanonicalで再生成していない |
| [correctness.json][CORRECT] | 科学実行前の人工38件PASS |
| [transfer_fix_correctness.json][FIXCHECK] | STOP後の巨大Fraction転送の人工回帰1件PASS。科学再計算なし |

D1入力は `fc96571881f301435a655f057709df4d6c1678e0` の `events/row_10.json`、cache、resource map。D2結果は `d1c59b28f4370cc7be65cff60d1ec877fd1491f5`。各参照とbyte identityはSOURCEに保存した。G9/G10 x=5/7のIR/価格は使っていない。

## 2. 再利用・correctnessと最小差分

実行前にD2 boundary、Parent allocation、RA finite-bit、g8 log/bounds、D1 accounting、Parent整数I/O contextの6 moduleが固定SOURCEと同じbytesであることを確認した。既存boundary人工19件、Parent proposal人工13件、新pair接続6件の合計38件PASS。phaseful adjoint、非可換順序、literal相殺、wire/key、joint係数、実q/weight、support/zero-price/precisionと共通targetの接続を確認した。人工検査は登録入力・科学runner・新synthesisを起動しない。

追加した [pair adapter][ADAPTER] はD2の `gate/simplify/cancellation_depth/native_price` とsuffix cache手法を使い、個別pair価格を保持する。既存D2 aggregate出力では個別価格が得られないため必要だった。新sampler/optimizerはない。Parent `reweight_events/event_statistics` と既存Bernstein scalar式を接続し、r1 ledger全体をr2へ転用しない。

科学実行では保存event ID、phase field、IR reduction、保存step価格、cache sequence hash/count/strict errorを確認した。pair価格に負値はなく、T=0のpairもなかった。qA/qBを新たに評価するための条件は、この段階では満たされた。全law終了後に予定した旧D2平均とのexact anchor照合は、途中STOPにより未到達である。

## 3. 六cellの取得状況

主指標T、補助CX/1Qを別々に示す。価格はcontrolled保存IRに、XのH,H、YのH,Sdg,Hという既存測定wrapperを加えた既知gate成分。system状態準備は除外。T/CXのwrapper追加は0、1QはXで2・Yで3。Tは1Qに含まれるので足さない。

以下の費用は両axis合計の**既知成分の期待費用**。表示は丸め、正本はRESULTの厳密有理数。ancilla準備/readout等に未定義の共通費h_metricがあるなら各cellへ `2N*h_metric` が残る。測定時間をgateへ換算していない。

| Cell | N/axis | T費用 | CX費用 | 1Q費用 | 未知共通hの係数 |
|---|---:|---:|---:|---:|---:|
| P0/A | 773,966 | 429,626,133.047707 | 23,509,517.974704 | 1,111,501,995.747995 | 1,547,932 |
| P0/B | 773,966 | 427,870,692.302325 | 19,998,636.483940 | 1,105,697,758.230249 | 1,547,932 |
| qA/A | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 |
| qA/B（強い対照） | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 |
| qB/A | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 |
| qB/B | 未取得 | 未取得 | 未取得 | 未取得 | 未取得 |

P0の取得済みM2≈1.3460905381983308、L≈1.1602114196121027。保存実joint qから同じBernstein式でNを計算した結果であり、旧保守的Nをbaselineへ一方的に代入していない。ここでは算出Nが旧production Nと同じ整数になった。真の信号や実現varianceを使っていない。

P0の固定lawではBの既知価格・費用がAより低い。この比較で未知共通hの係数は同じである。**sampling変更の効果、qB/B対P0/B・qA/Bの優劣は不明。** IS側のmoment/range/ceilによる利益消失も、追加利益の成立も判断できない。値の欠落を0や同等性の証拠にしない。

## 4. 技術停止の原因と修正準備

エラーは `ValueError: Exceeds the limit (4300) for integer string conversion`。CPython 3.10の `Fraction.__reduce__` は `(Fraction, (str(self),))` を返す。ISの大きな厳密有理数をProcessPoolから返す際、標準の整数文字列変換制限に触れた。ファイル出力時の `integer_io_limit` は用意していたが、私の接続実装ではprocess間転送の範囲へ適用できていなかった。

qA/qBのtaskは起動されたが、親processへ有効な数値結果を受け取れなかった。正常終了したlawとして保存・引用できない。whole poolの終了を待ってRESULTへSTOP理由を書き、科学runnerの再起動・別namespace評価はしていない。

[RETRY_TRANSFER_FIX.patch](RETRY_TRANSFER_FIX.patch) は、既存の整数I/O contextをpool全体へ適用し、今回のLinux環境でfork childにも変換方針を継承させる最小修正案。private process内で完結し、数値精度・H/K・law・回路価格・比較条件を変更しない。**実行済みSOURCEへ未適用であり、科学再実行も未実施。** `git apply --check` と、5,001桁の人工Fractionを往復転送し元の制限を復元する回帰1件はPASSした。科学的な有限bit精度不適格の修正ではない。

## 5. 古典費用と評価限界

実行全体wallは12.554529秒、親CPU 0.441815秒、終了child CPU合計24.905727秒。pair表取得は0.092255秒。親peak RSS 95,304 KiB、終了childの最大peak RSS 108,008 KiBで、後者は同時総RSSではない。pair表7,948,641 bytes、P0 law表7,207,348 bytes。失敗したIS taskを含む実行費であり、成功六cellのsetup費とはしない。

CPU affinity32、観測MemAvailable 51,830,824 kB、低いloadを確認した。GPU driverは利用できず、GPU向けでない厳密有理数計算をCPUで処理した。IR/suffix cacheを共用し、主な3 lawの独立有理数計算を3 processで並列化した。旧D2のworker/RAM/wall上限を新研究へ転用せず、共有設定・他者processを変更していない。速度倍率は未測定。

P0のstatistics/N処理は約0.427223秒。qA/qBの個別normalizer/proposal時間は親へ未取得。onlineは `NOT_MEASURED`。既存 `dyadic_index` の線形table走査、weight lookup、境界処理等は無料にしない。古典費を含む全実行時間の優位は判定しない。

Bernstein十分shot数は保証に基づくforecastで、実測shot数・最小shot数ではない。有限bit q/weightとlog enclosureの扱いを保つ設計だが、IS数値未取得なのでその結果の保証完了を主張しない。既知 `a/√C` の新数理原理、非列挙生成の効率、全RA/CTSへの優位、分子PR＋RPE全round費用の改善は主張できない。

## 6. 次の最小作業と保全・STOP

**ユーザー**が、同一科学条件で転送修正後に六cell評価を再実行することを明示許可する必要がある。今回指示§14は「停止時は有効な途中結果・原因・不足条件を保存し、同じ科学評価を別namespaceやretryで無断継続しない」と指定しているため、技術修正が条件不変でも自動retryしない。

許可後の**Codex**の提案作業は、修正済み新SOURCEと本途中結果への参照を固定し、必要な人工転送確認を済ませ、同じ六cellだけを評価してqB/B対P0/B・qA/Bを確定すること。分子・非列挙sampler・追加r/degree/time/precision/optimizer・重要科学レビューは追加しない。T=0は今回の停止理由ではなく、regularizerや別指標を持ち込む必要は生じていない。

今回はSOURCE、valid partial raw/result、correctness、修正案と本報告を公開し、remote SHA・取得内容を確認してmandatory STOPする。元dirty checkout、main、旧branch、固定SOURCE/input/raw/results、契約/grant/ledger/marker/STOPは保全した。新H/state/reference、科学signal matrix、RNG/量子shot、build/compile/synthesisは実行していない。STOP後は資料整理・人工転送test・公開だけを行った。

[EXEC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/16d10faabb1f6aa4bda42f3aef81d89bf9ca9832
[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/5eb90e8241e69e85a1ca3e79938a367700c24a48
[SOURCE]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v1/SOURCE.json
[RESULT]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v1/RESULT.json
[PAIRS]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v1/pairs.json
[LAW]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v1/law_P0.json
[CORRECT]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v1/correctness.json
[FIXCHECK]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v1/transfer_fix_correctness.json
[ADAPTER]: ../../../src/trottertracks/algorithm_codesign/s2_multi_event.py
