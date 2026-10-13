# D3 v1一回実行の技術停止・保存監査・GPT引き継ぎ

記録日：2026-10-14 JST。固定契約のnamespace/dateは2026-10-13を保持する。

**最終分類は `D3_TECHNICAL_INCONCLUSIVE`。一回起動、retry=0、mandatory STOP。**
最初の二つの合成keyが精度引数の型変換で停止した。完成sequenceは0件、登録比較は0/48行である。
このrunから保証変更・価格変更の効果、方式の優劣、研究継続可否は判断できない。

## 1. 固定来歴と対象

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。
- 実行branch：`track-b-d3-one-shot-execution-20261013`。
- 固定source S：`c7d2c826433dcf3b1bb30c98fc05a3bf13970ae1`。
- authorization-only直接子A／実行開始HEAD：`db584d9fb3f80c07ae55d5e1fe477bff2e386af2`。
- [authorization receipt](d3_execution_authorization_receipt.md)：利用者の「次の作業に進んで大丈夫です」「続きを行って」を原文で記録。直前の完了報告が示したD3一回実行の範囲へ限定。
- [固定execution contract](../../../artifacts/track_b_d3_source/2026-10-13/v1/execution_contract_v1.json)：SHA256 `8dba8caf24bb8424e4d589bcd589e2bff65e80039cd385e34c0432c26993418a`。
- [authorization](../../../artifacts/track_b_d3_source/2026-10-13/v1/authorization.json)：SHA256 `6d55b4d7296350654c6006dead5a5c0faca55dbf5691c63d9eff3aaf7692a9c3`。
- [one-shot marker](../../../artifacts/track_b_d3_result/2026-10-13/v1/one_shot_consumed.json)：SHA256 `f1db465691c4349ce192d356735e426c968e99b49b24e9abaa0c5ace8a0b783f`。

Sの公開確認HEAD `6150e14075e9fb25ecffd72e42e78576c373003d` はAのparentとして使用していない。
Aの変更はauthorization JSONとreceiptの二つだけ。旧D1/D2、旧authorization、旧markers/STOPは保持した。
結果公開commitはAに続く証拠commitであり、実行HEAD Aとは区別する。正しい公開SHAはbranchのremote確認記録と最終報告に示す。

契約対象は既知developmentの明示3-system-qubit involution provider、
`p=(1/5,3/10,1/2)`、`tau=5/7`、`sigma=+1`、初期状態`000`、finite `P5(-i tau R/r)^r`、`r=1,2`。
`Q0=Z0`、`V1=R_XX01(pi/4)`、`V2=R_XX12(pi/4)R_ZZ01(pi/4)`、`Qi=Vi†ZiVi`、`R=sum p_i Q_i`。
分子・geometry・DF rank・軌道basis・deterministic split L_Dはこの明示synthetic domainには適用しない。
六方式、旧/改善Taylor保証×旧/新合成価格の4ケース48行を予定していた。
新価格はstrict `1e-7`の36 keys、全ケースのnative誤差会計は旧`1e-6`固定。
保存D1 law/IR、保存D2 literal境界相殺額を再利用し、219,303 pairsは再実行しない契約である。

## 2. 実際の終了状態と証拠

| 項目 | 保存記録で確認した状態 |
| --- | --- |
| runner起動 | 外側one-shot entry一回、内部ledger一回 |
| 合成worker | key 000=`1/28`、key 001=`1/20`の二つを起動 |
| backend予約token | 2件。合成完了数やbackend本体への到達数ではない |
| worker結果 | 二つとも`KEY_TECHNICAL_INCONCLUSIVE`、`saved=null` |
| 完成sequence／strict guard | 0／0 |
| registered rows／新価格cache | 0／なし |
| pair、provider/signal matrix、circuit build、native compile、量子shots、sampling、LP、DF、分子、GPU | 0 |
| 外側exit／signal | 1／なし |
| outer resource failure | `null`。保存記録にcap超過はない |
| retry／次stage | 0／未認可 |

[最終terminal](../../../artifacts/track_b_d3_result/2026-10-13/v1/terminal.json)、
[inner terminal](../../../artifacts/track_b_d3_result/2026-10-13/v1/inner_terminal.json)、
[外側process receipt](../../../artifacts/track_b_d3_result/2026-10-13/v1/outer_process_receipt.json)、
[STOP](../../../artifacts/track_b_d3_result/2026-10-13/v1/STOP.json)を原記録のまま保持する。
最終terminalのrows/new_synthesis_callsは`null`であり、完了した48行/36合成を主張していない。
保存ファイルの棚卸しから登録rows0、完成sequence0と確認した。

外側の`status_consistent=false`は、sourceがこのfieldを`status == COMPLETE`として記録するためである。
最終、内側、外側effective、外側inner_statusの四つの分類は同じtechnical inconclusiveで一致する。
raw fieldは書き換えず、監査では「failure labels一致」と「complete受理flag false」を分離した。

資源receiptはwall **1.2855492429807782秒**、CPU（外側および子を含む記録値）**1.8808714秒**。
wait4 child peak RSS **163,852 KiB（160.01171875 MiB）**、
live観測aggregate peak **363,675,648 bytes（346.828125 MiB）**、live単一process peak **151.9921875 MiB**。
短い失敗runの値であり、36 keys/48行完了時の性能・必要資源を推定する値ではない。
監査開始前のraw出力は**25ファイル、48,190 bytes**、最大30,776 bytes。監査・文書は別に追加する。

## 3. 停止原因：型変換。科学的なnegativeではない

二つのworkerに保存された例外は同じ：

    TypeError: cannot create mpf from Fraction(1, 10000000)

- [D3 worker](../../../src/trottertracks/algorithm_codesign/rte_robustness/worker.py)はprecisionを`Fraction(1, 10**7)`として渡す。
- [共通numeric helper](../../../src/trottertracks/algorithm_codesign/rte_reallocation/numeric.py)の`synthesize()`は、予約callbackの後、`gridsynth_gates(theta, mp.mpf(epsilon)/4, cfg=cfg)`を呼ぼうとする。
- 固定mpmath 1.3.0の`ctx_mp_python.py`の`mpf_convert_arg`はこのPython `Fraction`を受理せず、保存されたTypeErrorを投げる。
- Pythonは関数本体へ入る前に引数式を評価する。この例外とsourceの呼出し順序から、二つとも`gridsynth_gates`本体へ到達していないと判断できる。これは保存例外と静的経路に基づく判定であり、backend-entryを直接測定したcounterではない。
- workerの`synthesis_calls=1`はcallbackで加算された予約数。raw tokenにも「completion claimではない」と明記されている。実際に完成した合成2件と解釈しない。

準備時の63 focused testsは人工scalar/mockに限定しており、実backend接続のprecision型変換を検証していなかった。このテストの不足を実行失敗が明らかにした。
source未push、runtime/tool identity不一致、phase guard違反、資源cap、科学的bias不適格による停止ではない。
新しい角度・backend・matrixの再計算による原因再現は行わず、保存記録とsourceだけで特定した。

### 記録上の限界

key 000には個別process receiptがある。key 001はfailure result自体が保存されているが、個別process receiptはない。
最初の失敗を受けたacquisitionの`finally`が残りprocessを終了・waitする経路では、そのreceiptを書かないためである。
二つの保存例外とfailure result identityは確認できるが、key 001固有のexit/peak RSSを推測で補わない。
全体のstatus・資源は外側receiptを使う。この記録不足が型変換failureの直接原因ではない。

## 4. 保存値・provenance監査

[saved-only audit](../../../artifacts/track_b_d3_result/2026-10-13/v1/saved_output_audit/saved_output_audit_v1.json)：**999 checks PASS、FAIL 0**。
[監査script](../../../scripts/tracks/algorithm_codesign/audit_d3_failed_saved_outputs.py)はstdlibで保存bytes/metadataとASTだけを読む。
本番runner、数値backend、science moduleをimport/呼出しせず、登録再会計、型変換の数値再現、sequence生成、pair評価は追加しない。

確認した範囲：critical source/input **416 hashes**、protected old evidence **386 hashes**、raw25 filesのbytes/SHA、
S→A直接子とauthorization-only差分、job/started/token/resultのidentity、markerの不変性、
inner→outer→terminal→STOPのhash links、失敗分類・retry0・prefix不使用、欠測範囲。
raw failure outputsは[監査前snapshot](../../../artifacts/track_b_d3_result/2026-10-13/v1/saved_output_audit/raw_output_snapshot_v1.json)に固定した。
[prelaunch record](../../../artifacts/track_b_d3_result/2026-10-13/v1/saved_output_audit/prelaunch_metadata_verification_v1.json)は、当時のremote=A・clean・新namespace未存在・runtime identity一致を記録する。
監査PASSは失敗証拠の整合を意味し、D3科学比較の完了・成功を意味しない。

## 5. 最小の技術対応案と承認境界

今回は修正・retryを行わない。次に限定修正が認可された場合の候補は、
helper内で`Fraction`のnumerator/denominatorを固定precisionのmpmathへ明示変換し、
`epsilon`の有理数identity、key、error threshold、`epsilon/4`要求値を保持する型adapterである。
epsilonをdecimal文字列へ置換するだけでは保存key表記とvalidationが変わり得るため、無検証で採用しない。

必要な登録外validationは、実helperへFractionを渡しbackend本体をmockしたAPI接続test、
既存string入力との互換性、要求epsilonと保存identity、strict phaseとerror pass判定、失敗時token意味論。
登録keysやproviderで試行しない。必要なら別の限定変更として、終了された並行workerの個別receipt保存も検証する。

同じ数学・誤差保証・比較条件を維持する型adapterの設計とfocused testsは技術事項である。
その修正だけを理由に研究方針全体のGPT詳細レビューを追加する必要はない。
ただし、**この一回のauthorizationは消費済みで、現在は修正・再実行を開始しない**。
修正へ進む指示を受ける場合も、新しいsource/manifestを固定し、原v1のresult/marker/authorization/STOPを保持する。
新しいregistered runには新しいresult namespace、別の明示実行承認、authorization-only直接子、remote/clean/preflightが必要。
capやprecision、候補、保証、baselineを変えて失敗を救済しない。

## 6. GPT側へ渡す範囲・停止

D3の中心比較は未取得であり、P5/partial、r選択、保証×価格の作用に新しい科学的結論はない。
既存D1/D2の科学的扱いは変更していない。
GPT/利用者へ返すのは、明確なAPI型変換failureと、同じ科学契約で限定修正・新one-shotを用意する必要性の判断材料である。
Codexによるalgorithm採択、研究継続判定、実験拡張、追加source修正、実行は行っていない。

**原結果・marker・source・契約・旧証拠不変。結果と監査を公開後mandatory STOP。**
