# 並列化・高速化手法の再利用索引

## 決定論PF 保存compiled QPYの閉包監査（2026-11-15）

[報告・規則](research/d_pf_qpy_closure_audit_20261115/REPORT.md)、[入口](../scripts/pf_qpy_closure_audit.py)（規則・code固定 `d3338c7ef`）。

| 対象・手法 | 実装 | 条件・注意点 | 検証・実測範囲 |
|---|---|---|---|
| 長qの重いcompileの代わりに、保存compiled QPYで本体の周期構造を0-compileで照合する | wire token（51d324eaのwire比較を流用し、parameterをhexで符号化）を整数に置き換え、numpyで周期を抽出。group単位でprocess並列 | 固定幅・決定論PF・index0の保存QPYがあり、Qiskit 1.3.0である場合に限る。直接価格は得られない。監査qの外への延長は論証による | 実対象56件で、12 workerのwallは100.5 s、workerの最大RSSは1.47 GiB。最大5.75M命令で65.6 s。人工test 18件。m12を12 compileする場合の見積り（約12 h）の代わりに閉包の前提を確認した。速度倍率は比較対象がないため未定義 |

## 決定論PF m1/m2（2026-11-15）

[実測・範囲](research/d_pf_m1_m2_direct_prices_20261115/REPORT.md)。m0の元trial / fast context / fork worker / 協調stage logをそのまま使い、同じcallerの有限selectorと出力namespaceをm1/m2へ拡張した。実行SOURCE `a3cbe70e4afb0948188403f48cf4086bb419d8bf`。12物理cellを2 route各6 workerで実行、X/Y別24 compile・元作用gate/C guard全PASS。取得wall102.512 s、最大worker RSS576.74 MiB、同条件baselineとの速度倍率は未測定。m0回帰4＋新3人工/mock testで、同じactual幅でもT/qが違うS2 cellを混同しないことを確認した。旧m0は元固定SOURCEで再現する。新namespace専用・科学retryなし、他round/モデル/N/Gへ転用しない。

## 決定論PF m0（2026-11-15）

[実測・適用条件](research/d_pf_m0_direct_prices_20261115/REPORT.md)、[入口](../scripts/run_pf_m0_prices.py)。既存PF fast contextをFull/p7の元trialへ最初から適用し、別contextの2 route・6物理cellをprocess並列化した。各cellのevolutionはX/Yで共有し、両軸を個別compileする。新規12価格と元作用gate/C guard12軸が有効。取得wall100.876 s、最大worker RSS575.16 MiB。同条件baseline再実行をせず、速度倍率は未測定。

元`pf_price_stages._cell`を使い、公開multiprocessing APIで終了code/signalを記録。compose / fingerprint / measured wrapper / compile＋保存 / statevector gateを変更せずcontext内でSTART/END/ERRORを観測する。人工fork入口で計測と復元、異常終了時の部分保存・追加投入停止も検査した。metadataの`_meta()`をnative childより先に呼ぶとprovider解決が変わるため、metadata親と`install()`先行の科学childをexecで分離する。固定SOURCE / native provider bytesはchild内のstdlib Git読取で照合する。新規namespace専用、旧runのresume/retryには使用しない。

共通手法集は[Full-H系列の固定版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aa09e0ee56f6190bb278058de38074e9fea0d477/docs/PERFORMANCE_METHODS.md)を参照する。本文を複製せず、このbranchで追加したPF手法を以下に記録する。科学実行範囲や稼働SOURCEを変更する許可にはならない。

## 決定論PF m9（2026-11-15）

SOURCE `7ed93d24dbd2c9d49d8951549c976c9eb463d4e5`。[技術報告・適用条件](research/d_pf_m9_holdout_20261115/ACCELERATION.md)。

| 対象・手法 | 実装 | 条件・注意点 | 検証・実測範囲 |
|---|---|---|---|
| 非重複Full/p7 cellをprocess並列化 | [p7入口](../scripts/run_pf_m9_p7_fast.py)、既存`pf_price_stages.run_cells` | physical ID・出力・context・SOURCEを分離し、重複jobと重大STOPを照合。科学対象やcompile数を増やさない | 実H4でFull3＋p7 3 workerの同時起動を確認。m9全12軸の元作用gate/C guard完了。[実行結果](research/d_pf_m9_holdout_20261115/REPORT.md)を参照 |
| 同一signed S2の再loweringを削減 | [compose_reusing_parts](../scripts/pf_native_price_fast.py) | 元composeを使い、一回のcompose内だけ、block参照と型/hexを含む全builder引数の一致で再利用。copy=Trueによる命令順・scalar一回を保持し、template hashを再確認。RTE/candidate間への無条件転用不可 | 人工4-mode DFのS2/Y4/M5、負時間Full/B0I、両軸hash・compiled全6指標・作用が一致。[最終4 test](research/d_pf_m9_holdout_20261115/FAST_TESTS_FINAL.txt)。実H4の同条件速度倍率は未測定 |
| canonical bytesを保持した高速fingerprintを通常入口へ接続 | [accelerated_native](../scripts/pf_native_price_fast.py)、既存[h46_b3_fast_fingerprint](../scripts/h46_b3_fast_fingerprint.py) | worker内で元cost_trialのcompose/wrapper/hash/mutation検査をすべてcontext内に置き、aliasを復元。compiler/lowering/測定は不変 | 元cost_trialの7 gateを人工入力で確認。部品の存在・別経路のPASSだけでは入口への適用確認にならない |

初期のm9入口は、既存fast helperがあっても未接続で、Full→p7を逐次実行していた。稼働FullのSOURCEを保全し、未開始p7だけを新SOURCEで修正した。今回の再開は保存結果の確認・binding・判定であり、既存runを再build/再compileする操作ではない。各入口への実際の適用とSOURCEを確認して再利用する。

人工composeの単発時間比を実H4全体へ外挿しない。CPU使用率やcache存在だけを有効価格・gate PASSとせず、対応trialと作用gate、公開receiptを確認する。今後の新規runnerでは主要段階の開始/終了・所要時間を開始前に観測可能にする。現在の稼働SOURCEへ後付け変更は行わない。

完了追記：実p7 m9でcompose 17.145/22.545/33.305 s、builder実生成1/2/6を記録。Full/p7は異なるprefixなのでwall比を高速化倍率にしない。判定データSOURCE `389ae157371b267a819357da1212f3115f7350f9`。元の計測範囲・制約は保持する。
