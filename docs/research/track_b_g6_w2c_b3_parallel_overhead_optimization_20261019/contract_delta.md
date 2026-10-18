# 正式採択前の判断と既存6項目の差分

**今回の全方式は非採択。6項目を途中の承認gateとして扱わず、人工実装・修正・測定を最後まで実施した。** 科学実行や本番統合へ進む承認は含まない。

ユーザーが判断する中心は次の**2点**に集約できる。

1. **費用比較の意味**：同時競合下のpair CPUを科学比較の主指標にするか。逐次S0と同じ実行条件ではない。cached observerは同じcall/returnとexclusive partitionを維持するが、observer effectと実装費用は小さくなる。そのSOURCE・計測方式を固定し、逐次/並列・baseline/instrumentedを区別して採用する必要がある。
2. **失敗時の並列処理と保全**：P1・8 workerのcooperative in-flight drain、parent到着順、喪失trial/未ACK bytesの保守的課金を受け入れるか。必要ならP2/capを選ぶ。START_RESERVED/ENTER gate、call/byte上限、spent/retry0は緩めない。親STOP後の新規許可停止は、物理的な例外発生瞬間の全worker即時停止とは異なる。

その判断後に行う本番SOURCE接続・schema/host/admission/pin/独立承認bindingの実装と技術検証は、別途依頼される後続作業。実入力pilot/実authority/本番実行はさらに明示的な実行承認が必要。今回自動で着手しない。

| 既存6項目 | 今回の差分案 |
|---|---|
| Dispatch | 8 pair・3 barrier・X/Y共有・fresh processは同じ。主候補は8 worker |
| Start/call | RESERVE＋ENTERを維持。前trial COMPLETEを次予約messageへ同梱。未来call block予約なし |
| STOP/prefix | cooperative/未知結果の保証範囲は同じ。親がcommit/ACKを観測するmessage単位を明記 |
| CPU estimand | concurrent条件の判断は残る。同範囲のcached observerと実装費をSOURCE固定。OFF診断を研究計測に代用しない |
| Global保存 | 原bytes事前予約、state pwrite、全体上限は同じ。write ACKの到着遅延と返却しない未知creditを追加規定 |
| SOURCE/認証 | 新module/observer/transport/policyを新manifest/schema/pinへ含める。本番hostや旧v2を今回書き換えない |

G7 input/law/N/seed/bitstream、phase/weight/word、誤差/confidence、R1/A6 τ=10^-6・Kφ32・J32、math-only、科学対象、31,710,848 callsの正式予算は変更しない。実scientific samplingで同値性を実測したとは主張しない。
