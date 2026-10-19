# Q1–Q10：独立技術受入と次担当

**結論：固定Storage V2の人工技術受入を完了した。正常2件、技術失敗40件、証拠不完全66件を独立に確認し、科学実行は閉鎖したままmandatory STOPする。** 同じサーバー上の別セッション・別worktreeによる技術検証であり、異なる物理環境での外部再現や第三者査読ではない。

**Q1. 固定SOURCE等は照合できたか。** できた。公開596781daeba253418ba35f4b7964312a54caf233／測定45bf46b4a833e71f138903fb511f2741e54043b7を区別してSOURCE/schema/inventoryを検証した。最初の82公開ファイルと補足6ファイルはGit blobと一致。108ケースの4part archiveと758 memberのbytes/SHA256も一致した。測定SOURCEの直接URLは404のため、公開履歴bundleを新設bare Gitに復元し、15測定SOURCEファイルの公開bytes一致も確認した。

**Q2. V2独立adapterと独立性。** 新しいidentity/trace/adapterとacceptance harnessを実装した。旧global core・pair/F01・CPU partitionを使い、V2の意味論を別実装した。独立consumerはProducer/scheduler/Admission/Producer補助checkerをimportせず、Producerの保存済み判定は結果比較だけに用いた。旧V1検証器は変更していない。

**Q3. 108件の結果。** 正常 `VERIFIED_ARTIFICIAL_COMPLETE` 2、`TECHNICAL_FAILURE` 40、`INCOMPLETE_EVIDENCE` 66。全件で申告分類と一致した。66件を正常PASSへ変えていない。

**Q4. 予約会計修正。** 固定旧rawは予約state1/report reserved0を `GLOBAL_STATE_COUNTER_BINDING` として独立拒否。V2のS/P1保存rawはreserved1/started0/cancelled1。さらに新規外部writer故障S/P1で実write例外時点のcounter/cursor/ticket/stateの一致を観測し、保存rawも別consumerで再確認した。両方ともtrace不完全なのでINCOMPLETE_EVIDENCE。固定人工条件で修正を確認した。

**Q5. RAM/physical/prefix/UNKNOWN/STOPの証明範囲。** 82件は物理stateからcounts、shadow hash、cursor等を照合。14 state保存不完全＋12 report欠損は同じ証明へ格上げしない。trusted prefix後のtail、未保存RAM、未知entry時刻、snapshotだけからのatomic historyは未確認のまま残す。P1 drainの8開始/7COMPLETE/1FAILを確認。STOP後ENTER、重複、UNKNOWN成功化を拒否する。

**Q6. Summary/receipt/diagnostic/marker。** schema/identity/hash/prefix、snapshot境界、byte credit/ACKとmarker順序を結合して検査した。diagnosticはpre-closeの非成功記録で、receiptの代用にはしない。markerはstorage close前に出るため、後続close失敗では残り得るが最終受入は拒否する。単一ファイルの存在を成功の証明にしない。

**Q7. 費用と予算。** 保存reportの96件は独立cost boundary/partitionを検証、12件はCPU/wall未知をnullで維持。parent＋worker CPUとouter wall、coarse/fine代替、ACK/physical/reserved、trace自己課金、diagnostic、RAM shadowとstate保持/累積pwriteの区別を確認した。個々の親保存処理のexclusive cost、observer単独overhead、unique memory peakは未取得として残す。候補予算は無断変更していない。

**Q8. Testsと新不具合。** 原文の旧623件は622 PASS/ラベル期待値不整合1 FAIL。元SOURCEを保全し、その期待値だけを明示補正した別versioned harnessでは623 PASS。新独立57件もPASS、重複しないversioned回帰は680 PASS/FAIL・ERROR・SKIP各0。51 negativeは57件の内数。公開108 rawと新規writer故障2件、Producer既済820件を加算していない。新しいProducer不整合は今回のscopeで検出しなかった。consumer実装上の不具合は修正・再検証し、初期記録も保持した。

**Q9. Production bridge前の残件。** 66件と歴史pendingの証拠限界を維持する。V2のRAM/physical/prefix/UNKNOWN/marker/close意味論の正式契約採択、旧test期待値の保守、二条件global bundle/one-shot/call budgetを結合するproduction bridge、CPU比較estimand、正式SOURCE/承認/監視条件の判断が必要。n=256・二条件16,896 callsは未採択候補のまま。

**Q10. 禁止科学実行等は0か。** 実G7/F3 sampling/import、新科学計算、保存科学値数値再解析、science SOURCE採択、実authorization/grant/claim発行・消費、quantum native build/compile/synthesisはすべて0。旧grant/claim再利用、既存STOP解除も0。新fake storeだけを人工検証に使用した。

公開内容・remote HEAD・公開bytes/SHA256の記録はpublication_receipt.json。保全監査はpreservation_verification.json。元dirty/untrackedの内容全件hash snapshotを取得したとは主張せず、親statusの前後一致、固定の使用SOURCEと旧consumerのbytes一致、および書込scopeを区別して記録した。

**次の作業（未実行）**：**ユーザーとGPT**が本packetを確認し、V2の正式技術契約差分・旧test保守・production bridge準備の範囲を判断する。科学実行には別の明示承認が必要。**Codex**はcommit/pushと公開照合後mandatory STOPし、次段階へ自動移行しない。
