# 監視・serializationの実装案（未実装・未承認）

対象sourceはrun05の19件。今回は環境差分の解決案を固定してSTOPし、source・tests・科学結果を変更しない。
run05の75 local testsは旧hostの実装証拠で、production成功でも新hostでの再検証でもない。

## 静的に確認した残る問題

`circuits.number()`はndarrayを`tolist()`してから全要素をexact辞書へ変換する。
array metadataの共有はcall-localであるが、最初の変換は巨大なPython containerを作る。
`numerical_fingerprint()`は全`serialize()`終了後にhashへ渡すため、hashの64KiB分割だけではこの区間を分割できない。
`OwnedRun._watch()`はdriver内のthreadで1秒周期。GIL/GCによるdriver停止と観測I/O遅延が、同じ5秒理由にまとめられている。
旧停止stackの実log・journalsは未受領なので、GIL/GC/I/Oの寄与を測定済みとは扱わない。

## Streamingの変更案

- ndarrayのdtype/shapeと行優先の値を遅延走査し、巨大な`tolist()`・exact中間treeを避ける。
- gate/definition/control state/phase/ordering/Unicode/signed zeroと旧canonical bytesを保持する。
- 回路全体のinstruction/parameter処理も遅延走査し、64KiB単位のencode/hashへ接続する。
- 同じmatrixのencoding再利用は1回の呼出内だけ。上限付きchunk cacheとして明示計上し、global cacheにしない。
- 小さい回路は旧serializerのcanonical bytes、8-system+ancilla人工回路は同じshapeの全serialize→hashを比較する。
- 新sourceの追加moduleを19件に加え、旧/new blob・byte hash・seed再結合を保存する。

## 監視を分離する案と未承認費用

独立observerを候補とする。単にthreadへの`sleep(0)`追加で独立した監視が成立したとは扱わない。
observerはstdlib中心のprocessとし、AS256MiB/RSS64MiBを検証前の上限候補とする。これは採用済み上限ではない。
既存12-worker admissionの120GiBにAS候補256MiBを明示加算すると120.25GiBとなり、追加roleとresource契約のreviewが必要。
CPU時間・全run elapsed wall（起動/終了を含む）・RSS/AS・全log/journal bytesをdriver/worker/observerについて計上する。
monitor/logを増やす場合、旧3.5GiB/560000inode planningを根拠なくそのままPASSにせず、serialized record cap・周期・72h件数・filesystem allocation・journal overheadから再積算する。
総10GiB charge/72h/74784 actual capを独断で増やさず、追加roleを無料扱いしない。

observerのUID/PID/starttime/親子関係をnew-hostで確認し、pidfd等を含む確実なownership検証後だけown-runを停止する。
old-host PIDは停止対象にしない。FD/pipeはprivate、bounded frame、不要FDは各processで閉じる。
phase通知にはsequence・phase・monotonic timestampを付け、driver退出/EOF・worker死・observer死・欠測でfail-closed。
最初のfailureをdurableに保存してからown-runを終了し、driver/pool/observer終了とFD closeを確認する。
監視自己故障の扱いとoutput chargeの同期・予約手順も実装前に固定する。

追加roleを避ける代案は、既存owned workersへbuild/serialize/hashを移す二段階protocol。
識別前予約・重複owner・compile-only actual invocations・候補/trajectory/axis順を変えない設計が必要で、現時点では実装済みではない。

## 限定testsの草案

新environmentを選定してから、256×256人工complex parameterと8-system+ancilla回路で旧/new byte/digest、dtype/shape、signed zero、control/phase、nonfinite STOPを検査する。
12 workersのowner/RSS/AS・worker死・欠測・pressure/OOMはmock/fixtureで検証し、12実worker stressを起動しない。
GIL占有/重いGC・観測I/O遅延を別ケースで注入し、interval・observation duration・staleness・phase・最初の停止理由を別fieldへ保存する。
5秒制限・fail-closed・own-run限定は維持。大規模campaignは作らない。
追加transpile予定0、今回実施0。必要になれば旧累積28/64を引き継ぎ、件数/capを事前固定する。

source修正・新tests・新SOURCE_COMMIT/plan固定は未実施。旧source/plan/auth/reviewと旧承認記録を保持する。
