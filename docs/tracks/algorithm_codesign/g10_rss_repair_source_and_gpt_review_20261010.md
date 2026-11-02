# Track B G10 v2：RSS限定修正source・GPT実行前レビュー

今回の成果は**実行前sourceの準備と非科学的validation**である。本番G10は実行していない。
新authorization A2、登録用marker、v2 resultは作成していない。科学実行認可はfalse、mandatory STOP。
研究方針、新規性、科学的結果の採択はGPT側へ戻す。

## 固定証拠と作業分離

- Repository: `HIROMU1015/Partially-Randomized-Trotter`
- 準備branch: `track-b-g10-rss-repair-source-preparation-20261010`
- 準備base: `21f66137ab205fb0d71ee6ee001844f4c2d725e3`（RSS原因調査）
- 旧science S: `05c5ef23fce775a822ab5686f5da2f0d77675864`
- 旧authorization A: `f5cd0755424d1b11e2249cc115518d84fb8bb8d3`
- 旧technical result R: `e429c99d77b3222c5cca62750b2d111f87e4cb50`
- 本資料と同じ公開commitをS2レビュー対象とする。完全SHAは公開時のhandoffで指定する。

旧S/A/Rのsource・contract・authorization・result・marker・STOPは不変。
元runの分類は `G10_TECHNICAL_INCONCLUSIVE`、17行の科学的利用は引き続き禁止。
今回のI/O検算は保存済みtechnical JSONのbytesを再現するだけで、17行を再採点していない。
別Track、root worktree、共通library/API、旧branchの履歴を変更していない。

採択した指示は [利用者指示の原文](g10_rss_repair_user_instruction_20261010.md)。
attachment原文をこのpathだけへbytesのまま保存し、identityを新contractに記録した。
原因調査の確信度は [旧技術報告](g10_rss_failure_technical_investigation_20261010.md) を維持する。
元RSS例外の正確な発生行は未確定であり、serializationが唯一の原因とは断定しない。

## source変更と意味論の境界

| 対象 | 修正 | 維持した内容 |
| --- | --- | --- |
| 新 `g10_io.py` | 型を遅延変換するencoder、bounded UTF-8、逐次bytes/hash、exclusive出力 | Fractionの精度、indent=2、ensure_ascii=false、allow_nan=false、末尾改行 |
| 新 `g10_degree_matched_native_v2.py` | 旧scientific bodyをcollectへ移し、局所参照解放とI/O診断を挿入 | algorithm呼出し・順序・引数・全row/native IR/budget/cache項目 |
| lifetime | decode後old_raw解放、m5 rebudget後old解放、row完成後pendingとloop local解放 | m5のdeepcopyと元G9 budgetのdeepcopy、rowが保持するevent/native情報 |
| 正常出力 | guard内でstream/write/flush/fsync/close/identity検算とcompletion protocol | 同じlogical payloadのJSON bytes、元の上限値 |
| 失敗出力 | 巨大result再serializationを廃止、bounded failure receipt | consumed marker、retry=0、technical prefix禁止、STOP |

旧runner、`g10_comparison.py`、generator/reference、native/matrix、numeric、共通guard/launchは変更していない。
特にm5 deepcopy自体は削除せず、旧graphの不要な寿命だけを短縮した。
`pending`のevent参照をclearしても、完成rowのbindingが必要なeventを保持する。
科学計算が終了するとcollect frameが消え、loop localの参照を出力まで保持しない。
失敗時にはfull resultをclearし、短い例外位置を取り出してtracebackの大きなframe参照も解放する。
`del`がRSSを必ず下げるという主張、強制GCによる本番成功の主張は行わない。

[runner差分](../../../artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/runner_diff_v2.patch) は
関数移動も含む。読みやすさのため、別途AST検査でscientific bodyの一致を確認した。
検査で除去するのはlocal `del`、`io.snapshot`、`pending.clear`だけ。
削除変数のallowlistと位置も照合し、field削除は拒否する。
classical stage clockの計算式も一致し、科学関数・baseline・confidence式の置換はない。
最終protected hashだけは新I/O moduleの逐次checkerへbindingを変更し、同じledger/prefix/hash規則を維持した。
旧G10の66 MB JSONを新ledgerに追加しても、科学graphと全file bytesを同時保持しないための変更である。
synthetic full/prefix/tamper検査で旧checkerとの一致を確認した。旧共有helperとlaunchは不変。
実際の時間値、source/contract/protected-countのprovenanceは新実行で変わる。
本番実行なしの静的確認であるため、全本番payloadの動的同値性を新runで確認したとは言わない。

## JSON型とschemaの適用範囲

適用範囲はstring-key dict、list/tuple、Fraction、JSON scalar。
Fractionは旧`serial`と同じ`str(Fraction)`、tupleは同じJSON arrayへ変換する。
Unicode、浮動小数、key順序、indent、新行も同じlogical payloadで旧方式とbytes一致を確認した。
nonfinite、未対応型、cycleは拒否する。cycleの例外型まで旧再帰関数と同一とは主張しない。

非文字列keyは拒否する。旧`serial`の`str(key)`とJSONEncoderのkey変換は一般には同値でない。
保存JSONの全keyが文字列であることに加え、sourceのpayload生成箇所を静的に確認した。
runnerのdegree-keyは`str(m)`、cacheは文字列ratio、cost座標は`T/CX/1Q`、
`g10_reference.cts_events`のcertificateはtuple-key targetを `k[0]+':'+str(k[1])` へ明示変換する。
transientなPauli targetのtuple-key dictをそのままresultへ渡す実装ではない。
必要条件に反するfuture schemaは黙ってkeyを変換せずtechnical failureとする。

全containerコピーとJSONEncoder.encodeの`list(chunks)`/joinを使わない。
bounded bufferから1 write最大32,768 bytesでUTF-8を出力する。
validationのworkspaceはancestorのdepthに比例し、shared acyclic subtreeを許可する。
CPython iterencodeが一つの巨大escaped string tokenを構成する可能性は残る。
登録G10のstringは短く、任意サイズのstring tokenまでメモリが完全boundedとは主張しない。

## 出力と失敗記録の技術契約

[contract v2](../../../artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/contract_v2.json)
の旧契約からの変更はpath/schema/base/branch等のbookkeepingと`io_revision`のみ。
`p=(1/5,3/10,1/2)`、`x=5/7`、`m=3/5/7`、3-qubit provider、全arms、
共通finite operator、native費用、confidence/error/precision/seed/samplingは全て旧契約と一致。
RSS512 MiB、AS1536 MiB、wall1200 s、CPU900 s、per-key30/20 s、
output134,217,728 bytesと他のcapsも同一。

1. fresh v2 directoryのexclusive markerを後日の正式実行だけで消費する。
2. 正常時は `.partial` をexclusive作成し、validation/encoding/writeを監視する。
3. bytesとSHA256を逐次更新し、flush/fsync/close後にdiskをbounded読み直しで照合する。
4. exclusive hardlinkで `result_v1.json` へ昇格する。既存fileを上書きしない。
   成功した今回作成分のtemporary nameだけをunlinkする。
5. success STOPを監視下で保存し、最後に `COMPLETED.v2` を作成する。
   最後のclose後もguard checkを行う。
6. scientific completionにはtoken、matching success STOP/result identity、failure receipt不在を要求する。
   `.partial`だけ、final JSONだけ、成功statusだけでは完成結果と扱わない。
   `verify_completed()` はこの必要条件をread-onlyで照合する。

aggregate output上限にはmarker、telemetry、payload、receiptを含める。hardlinkは同じdataを二重計上しない。
32 KiBをterminal用に予約し、各receipt16 KiB、telemetry32 KiBを上限とする。
旧上限の拡大ではなく、housekeeping込みでより厳密に会計する変更を明示した。
同じpayloadに対するJSON schema/bytesは維持するが、上限ぎりぎりのpayloadには予約分の余裕が必要。

guard failureを含む失敗時はfull resultを再encodeしない。technical receiptへ
短いreason、最大4 frameのfile/line/function、stage/phase、current/peak RSS、wall/CPU、
source/auth/contract/tool/marker identity、row数、tracked partial bytes/hash、STOPを保存する。
tracebackはこれらを抽出してから解放する。
`ru_maxrss`がstickyなので、既に失敗したpeak checkを繰り返してreceiptを阻害しないよう、
**失敗確定後の小型receiptだけ**periodic alarmを停止する。その間も元AS/CPU制限はguard exitまで維持する。
科学処理、大きなserialization、出力retryはこのwindowで行わない。

失敗receiptはbest effort。OS kill、OOM、disk full、receipt I/O failure、guard entry/exit failure時の
完全な記録は保証しない。markerは保持し、completion条件未成立のprefixは利用禁止。
SIGALRMがwriteとhash bookkeepingの間に発火した場合、tracked identityがdisk全体に一致しない可能性がある。
失敗後に巨大fileを再hashせず、`identity_reverified_after_failure=false`と明記して後日のread-only監査へ渡す。
STOP保存後のtechnical failureでもfailure receiptがcompletionを無効化し、既存STOPを書き換えない。
最終guard failure時は自分が今回作成したcompletion tokenを先にunlinkし、receipt I/O failureでも無効化する。
旧token、marker、STOPは削除しない。OS kill等でhandlerへ到達しない場合はこの無効化も保証しないため、
外側processの正常完了receiptとread-only監査を確認するまで科学的完成を認定しない。
`verify_completed()`はfile identity/completionの必要条件だけを検査し、外側process auditを代替しない。

## 非科学的validation

47件のfocused testsは本番runnerを呼ばず、人工dict/tuple/Fractionとtemp fileだけを使う。
JSON同値性、非finite/未対応/非文字列key/cycle拒否、marker保持、output cap、short write、
write/flush/close/fsync failure、guard failureの各段階、exclusive collision、
partial/final/completionの区別、bounded receipt、traceback解放、pending alias/m5 copy、
direct authorization-only child条件とpending拒否を検査する。
既存のmatrix/samplingを含むsemantic suiteは今回再実行していない。
旧sourceのsemantic evidenceは旧commitに保持し、今回の変更は静的差分とI/O focused testsで検査した。

保存JSONと人工typed treeは、それぞれ同じruntimeの独立processで新旧方式を一回ずつ検算した。
各I/O診断はRSS512 MiB、AS1536 MiB、wall/CPU60 sという本番より短い診断上限で行った。
新方式の計測には実際のvalidation/stream/write/flush/fsync/close/disk hashを含めた。
診断のtemp payloadを科学resultへcommitせず、production marker/tokenも作成していない。

| I/O-only対象 | 旧方式peak RSS | 新方式peak RSS | bytes / SHA256照合 |
| --- | --- | --- | --- |
| 保存G10 technical JSON | 510.48 MiB | 255.50 MiB | 66,842,493 bytes、`b62695c…14dfe1`で一致 |
| 人工50,000行Fraction/tuple/Unicode | 153.50 MiB | 42.73 MiB | 8,371,227 bytes、`b99e1ef7…f55458d`で一致 |

保存JSON計測のwallは旧約1.92 s、新約2.80 s。低メモリ化には追加validationとdisk検算の時間が伴う。
前回技術調査の508.91/253.25 MiBとの差は今回のimports/guard/実際のwriter等を含む計測条件の差であり、
どちらも本番G10のlive Fraction/native graphを再現してはいない。

機械可読根拠は
[validation summary](../../../artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/validation_summary_v2.json)、
[source/protected audit](../../../artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/source_preparation_audit_v2.json)、
[source manifest](../../../artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/source_manifest_v2.json)、
[evidence inventory](../../../artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/evidence_manifest_v2.json)。
protected ledgerは1,300 paths（append-only indexはbase prefix）を保存し、違反0。
旧G10のresult/marker/STOP、old113critical source、旧A、RSS監査、Track A共通manifestを保護した。
NPZはaccess前に拒否する。科学run/synthesis/matrix/circuit/sampling/LP/DF/molecule/GPUは0。

## GPTレビューと次の実行条件

今回の結果が示すのは、主要なserialization増幅経路を除去でき、保存JSONでbytesを維持できたこと。
**本番G10がRSS512 MiB以内で完了することは未確認で、保証しない。**
live graph、matrix/synthesis cache、allocator、Fraction、aliasing、future I/O failureが残る不確実性。
本番再試行を行わず、row spooling、cap引上げ、誤差条件・出力field削減も採用していない。

GPTはS2のscientific AST/contract差分、pending launch、completion protocol、
sticky-peak failure window、aggregate cap/terminal reserve、source manifestをreviewする。
承認後も、この準備だけでは実行しない。
別の明示one-shot指示を受けた場合だけ、S2の**直接子A2**に新authorization JSONと任意新receiptをcommitし、
新branch remote=A2、clean、manifest/runtime/protected hash、fresh v2 marker不在を確認する。
旧S/A/R/markerは残す。旧Aの認可をv2へ流用しない。
認可JSONのstatusは既存gateの `APPROVED_FOR_ONE_G10_RUN`、source=S2、runs1/retry0/STOPtrue、
contract v2 SHAと当該利用者の新実行指示を固定する。

次に必要なのはGPTの実行前レビューと利用者の別途実行承認。
今回の準備完了時点でmandatory STOPし、科学的結論と次stageの判断をGPTへ戻す。
