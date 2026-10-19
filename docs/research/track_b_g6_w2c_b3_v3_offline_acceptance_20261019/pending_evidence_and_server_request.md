# Pending証拠とサーバー担当への最小引継ぎ案

固定producer SOURCEの修正を要する不整合は今回の公開47件から確定していない。以下は証拠不足であり、技術的不可能・NOT_APPLICABLEとは区別する。元SOURCE/rawは変更していない。

1. **障害30件の完全contract不足。** 公開で得られるのは正常12件＋障害5件のみ。30件はBUNDLE_BINDING/Hostの保存digest一致まで。対応する生成時のcontract原bytesを新しい証拠版に追加できればformal/runtime/source/schema digest再計算を拡張できる。今回再生成して一致した扱いにはしない。
2. **P08: 全47件にper-trial親gate履歴なし。** statefile/controlではSTOP後のENTER grant全履歴を再構成できない。次版で必要ならrun/bundle digest、seq、global offset、RESERVE、ENTER permission、terminal、STOPのordered traceを予算内で保存する仕様判断が必要。今回新producerやtraceを考案・実装していない。独立normalized人工fixtureが検査可能な境界を示すのみ。
3. **P07/P11: private approval/SQLite/store historyなし。** source静的構造・69件の旧人工Admission回帰・v3保存retry0/resume false/STOPPED-SPENT申告を区別した。実v3 atomic claim proofではない。必要ならscope、claim、formal digest、store identity、approval/reservation/spent/STOP/retry拒否の既存人工証拠を新しい公開版へ添付する。全caseに同じARTIFICIAL_SCOPE_ONEが現れるが、別tmp store間の実reuseの可否は判定できない。非再利用を推定しない。
4. **Historical Python/private input bytes不足。** 保存manifest内SHAの形状・相互結合、公開SOURCE/schema/fixture bytesは照合した。過去Python3.12実体、private adoption原file等は未取得。ローカルPython3.10の別bytesを代用しない。
5. **P06: 11件pending。** log cap/worker loss/dispatch/crashではworker END/通知の保存不足がある。log capで前半noticeが失われ後半だけ残る場合は保存subsequenceを検査する。欠落を補完しない。
6. **親crash: P04/P09 pending、P10は障害35件pending。** 成功summary/receiptがないことは想定どおり。偽成功を作らず、未知のCPU/ACK/exit履歴は保留する。

以上は依頼案であり、今回サーバーへ送信・実行・接続していない。ユーザーとGPTが、どの不足を埋めるか、今後の正式採択に必要な範囲を判断する。科学的な優越性・正式GOのレビューを実施した資料ではない。
