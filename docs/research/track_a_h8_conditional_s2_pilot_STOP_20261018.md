# H8 PR s2条件付きpilot run01：metadata不一致によるSTOP

2026-10-18 JST。利用者が一回認可したq4→q8の条件付きpilotを起動したが、q4 workerの保存preparation receipt照合でSTOPした。新しい科学sampling・native compileは0件、q8は未起動。全owned processはnative退出・回収済み。今回のone-shotは消費済みであり、自動retry/resume・新しい有効grant発行は行わない。

## 固定実行と結果

Linear H8、1.00 Å、STO-3G、保存tol-only DF rank28、16 modes、sector4900、T=.8、prefix7、S2、r4、K4。Aはq4/R16、Bはq8/R32、各s2条件付き8本・cosine/sine paired。state preparation費用を除外し、whole native symmetric-directional Hadamard wrapperを測る計画であった。

| 項目 | 状態 |
| --- | --- |
| 実行SOURCE | `3db51df7bdb4e28968f0bd4d5af22d8e4553fe56` |
| 実行前REVIEW資料commit | `f62fb6f7b6e0607166b9d4d5f4682865b356346d` |
| 科学実行試行・消費one-shot | 1 |
| 起動worker | q4の8 process、各own CPU1・内部thread1 |
| 実conditional trajectory / native compile / control probe | 0 / 0 / 0 |
| q4 accepted / q8 accepted | 0/8、0/8（q8未起動） |
| driver | native exit1、wait4/reap済み |
| 全8 worker | native exit2、全wait4/reap、cleanup error0 |
| 最終gate | STOP、mandatory STOP |
| 停止後SOURCE/input/runtime stat | PASS |

最初の記録は `WORKER_FAILED:B2_p7_q4_2nd_r4_K4_7:KeyError:'partition_hash'`。後続のcleanupで最初の理由を上書きしない。`native_exit.json.first_STOP=null`はwitness自身に例外がないという意味であり、driver成功を表さない。driver exit1、terminalとfinal gateのSTOPを結合して判定した。

科学sampling receipt、trajectory receipt、compile entry、transpile start、control validation、cost resultはすべて0件。adapterのnative準備がsamplingより先に行われ、そこで停止したことと整合する。native preparationは保存DFの回路用表現のmaterializeであり、SCF/DF/stateや新signalの生成は行っていない。

## 原因と限定修正

私が実装したadapterは、旧H8CoreSignalPortが保存しない独立field `partition_hash` を `ref[field]` で要求していた。人工fixtureの回路接続検査ではこの旧receiptのschema差を捕捉できなかった。

元science SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe` のreceiptは `cell_id / preparation_hash / lambda_r / constant / extracted_identity` の5項目。元 `df_partial_s2.py` はpartition hashをpreparation hashのcanonical payloadに含める。この来歴を[schema監査](../../artifacts/resource_applicability/track_a_h8_conditional_s2_pilot_stop/2026-10-18/receipt_schema_provenance.json)へ固定した。

実行済みcheckoutを変更せず独立cloneで次を修正した。

- 旧5項目をgrant/lockより前にschema検査する。
- 元preparation hashの完全一致、lambda/scalar、partition順序とthresholdを維持する。
- standalone partition hashが保存されている場合は、その完全一致も要求する。欠測値を捏造しない。
- 失敗run01の非返却予約を将来用draftのcarryへ加算する。旧ledger・上限を変更しない。

修正SOURCEは `7522d74be3f59a8be1b435b9d05980bf6a1485b3`、closure8。sampling law、元native constructor、compiler、driver/witness/poolは変更なし。[旧新blob/SHA](../../artifacts/resource_applicability/track_a_h8_conditional_s2_pilot_stop/2026-10-18/source_change_audit.json)を保存した。このSOURCEの科学実行は0件で、準備状態は未承認・未seal、有効grantなし、absolute commandはnull。

副次的に3 workerで `BasisFingerprintError`、4 workerでcleanup stop signalが記録された。fingerprint errorがcleanup signalを包んだものかは証明できておらず、独立した不備の可能性も未解決として残す。今回のmetadata修正によって実H8のnative経路が完走すると保証しない。

## 検査と資源証拠

実行前SOURCEは人工22検査と限定契約16検査に合格していた。16件は22件に含まれる再確認であり38件の独立coverageではない。人工2 full-wrapper compileのみであり、実H8 compileではない。過大な初期人工fixtureを科学前に中断した記録も保全している。

修正SOURCEは限定人工20件（既存16件＋receipt regression4件）PASS、0.340秒。旧receiptにpartition fieldがない正常系、保存hash改変、明示partition不一致、必須field欠測を検査した。最小childの所有・first STOP・exit/reap回帰もPASS。新SOURCEの無効grant CLIはexit1で科学import/lock前に拒否された。重いnative人工campaignや実H8の再準備は繰り返していない。検査は主担当によるもので、別担当の独立reviewではない。subagent0。

元compiler `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、profile `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12` を保持。実行前にprovider265件＋runtime34,273件、計1,508,760,538 bytesのSHA照合PASS。既存venv・共有設定を変更していない。旧hostとのcompiler output完全同一性は未認定のまま。

fresh gateはCPU重複なし・有効空きメモリ・PSI/OOM観測・filesystem block/inodeを確認してPASSした。空き520,286,720,000 bytes、219,263,064 inodes、約2GiBのown allocation probe成功・削除済み。数値quotaはUNKNOWNであり、無制限とは扱わない。8 workerのRSS peakは約418–432MB、driver native receiptのpeakは432,443,392 bytes。これらを同時RSS合計とは解釈しない。停止後のdriver/worker9 PIDは/procに不在。共有jobへsignalや変更を行っていない。

## ledger・carryと実消費

sampling/compileの実到達は0である一方、dispatch済みq4の予約は返却しない。

| 単位 | 旧known | 旧held | run01 held | STOP後known+held | 仮に新full pilotを別承認した場合の最大 | 元cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compile | 1572 | 22 | 16 | 1610 | 1642 | 1722 |
| trajectory | 768 | 11 | 8 | 787 | 803 | 843 |
| occurrence（outer q call） | 3056 | 11 | 32 | 3099 | 3195 | 4219 |
| control probe | 360 | 4 | 4 | 368 | 376 | 376 |

primitive2172、reference_matvec535088、signal92、normalization72も維持。過去carryのlisted compile2289・trajectory1112を含む現在の記載範囲subtotalは3899/1899。仮の新full pilot後は3931/1915だが、これは新しい実行承認ではない。歴史的unknown actual=null、返却0・resetなし。後続の別認可stageは別namespaceで保全し、このsubtotalを全研究の総消費とは呼ばない。[ledger監査](../../artifacts/resource_applicability/track_a_h8_conditional_s2_pilot_stop/2026-10-18/ledger_audit.json)。

## 科学的に分かったこと・未取得事項

今回の実s2費用標本は0本であり、q4/q8のaxis RZ・paired費用・平均・SD・SEはいずれも未取得（null）。元確認n32・探索n8・元bias/normalization/shot・既存費用比較・PR削減率を変更していない。n40として混合していない。s4費用はnullであり0とはしない。希少層の母平均保証と選択後効果は未解決。新しい科学結論は得られていない。

条件付きlawの人工検証は、N2>0のBinomial(R,p2/(p0+p2))＋一様位置配置が元s2 whole-trajectory lawと等価であることを対象とした。実samplingが成功したという証拠とは区別する。今回以外のsampling、s4、決定論compile、signal、fit、費用再会計は0。

## 証拠入口・公開・次の担当

[STOP bundle](../../artifacts/resource_applicability/track_a_h8_conditional_s2_pilot_stop/2026-10-18/README.md)に実行binding、旧新closure、native exit/reap、first STOP、予約、無効draft、検査、private原本hashを保存する。使用済みgrant・lock・NPZ・raw cache・元承認全文はprivateのまま。旧SOURCEと旧準備資料は過去時点の記録として保持する。

科学run SOURCEの公開前pushはHTTPS認証がなく失敗した。修正SOURCE・STOP報告も専用branchへnon-force pushを試みる。成功を確認するまではGitHub公開済みとは扱わない。認証設定は変更しない。ローカル最終commitとpush結果・コマンドは最終チャットで報告する。

次の担当：ユーザーが停止証拠を確認し、新しいone-shot実行を認可するか判断する。再実行が別途承認された場合に限り、Codexが新しいbinding・未使用output/control/lock・grant・fresh gateを結合する。今回は新grantを発行せず、科学retry/resumeせずmandatory STOPする。
