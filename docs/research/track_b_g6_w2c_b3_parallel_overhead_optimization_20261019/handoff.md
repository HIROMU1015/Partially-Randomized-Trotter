# Q1–Q8・ハンドオフ

状態：人工技術検証完了・非採択。最終測定/回帰SOURCE `77f8667fc0898a3615a4efa834d971b3c339df71`。公開/保全照合はpublication_receipt.json/verification.jsonを参照。

1. **Q1 主因**：管理pathを含むfine callbackの繰返しPath/label/contextmanager/row生成・locals参照と、trialごと5往復のIPC。observer単独でCPU約61/57/35%、IPC単独で約11/10/5%削減（共通Counter条件）。状態更新spanは約0.03–0.09 CPU秒で主因ではなかった。効果を単純加算しない。
2. **Q2 実装**：code identity/label/rowとself要否をcacheする同範囲observer、前commit/write ACKを次messageへ同梱するIPCを実装。clock検査の管理呼出しが減る以外、core operation/call数・event/bit/traceを維持。予約/ENTER/pwrite/全体byte予約は残す。
3. **Q3 実測**：下表。旧P1比CPU66.6/63.0/40.2%減、wall50.6/44.8/29.0%減。元S0比wall4.65/3.88/3.15倍。cached S0比ではCPU追加が残るため、演算量削減とは解釈しない。
4. **Q4 意味論**：global one-shot/spent、call cap、retry0、F01、byte cap、3phase gateを維持。P1は開始済み7件、P2 cap2は1件をdrain。親commit/ACK観測のmessage単位は変更し、未ACK bytes/結果不明を隠さずcredit返却しない。逐次R1と同じSTOPとは主張しない。
5. **Q5 テスト**：同じ最終SOURCEで**356 PASS（既存311＋追加45）**、skip0。99性能/27診断/17障害も同SOURCEで実施。runtime SOURCEを測定後に変更していない。
6. **Q6 最有望**：**改善P1＋cached observer＋coalesced IPC、8 worker**。今回の全workloadでafter1/2/4/8のうちwall最小。CPU最小を重視するならcached S0/少数workerとも比較する。共有負荷・観測条件の制約は残る。
7. **Q7 ユーザー判断**：重要判断は2点。(a) concurrent pair CPUと新observer実装を科学比較の主estimandとして採るか、(b) cooperative in-flight/parent到着順/未知credit保守課金を受け入れるか（必要ならP2/cap）。その後の正式schema/host/SOURCE/pin/binding接続は別依頼の技術作業。6個の途中承認gateは設けていない。
8. **Q8 実行境界**：実G7/F3 import/sampling、新科学計算、実科学保存値の数値再解析、実authorization/grant/claim発行消費、本番correctness256/31,710,848 callsは全て0。正式R1、旧SOURCE/結果/ledger/marker/STOP、他worktreeを保全。ローカルad3117d2やGMP/nativeは統合していない。

| 負荷 | S0 wall / CPU秒 | 旧P1 wall / CPU秒 | 改善P1 wall / CPU秒 | CPU削減 | wall削減 |
|---|---:|---:|---:|---:|---:|
| light | 13.8067 / 11.9682 | 6.0168 / 18.4206 | 2.9703 / 6.1490 | 66.6% | 50.6% |
| mixed | 7.0756 / 6.6512 | 3.3021 / 10.2070 | 1.8227 / 3.7741 | 63.0% | 44.8% |
| heavy | 6.5647 / 6.4287 | 2.9307 / 8.3151 | 2.0818 / 4.9703 | 40.2% | 29.0% |

CPUは親＋全worker、wallはhost/STOPまで、同じ人工集合の中央値。共有負荷上昇を含む今回の対照値で、実G7予報ではない。

**次の作業（未着手）**：**ユーザー**が上記2点の採否を判断。依頼を受けた**Codex**が別scopeで本番接続のSOURCE/schema/guard/admissionと人工回帰を準備する。実入力pilot/実authority/本番は別承認まで行わない。今回の公開・保全確認後、**Codexはmandatory STOPして待機**する。
