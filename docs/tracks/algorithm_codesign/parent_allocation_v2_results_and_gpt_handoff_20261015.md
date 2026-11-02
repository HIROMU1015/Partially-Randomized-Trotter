# Parent allocation v2：CPU上限による技術停止・保存監査

2026-10-15 JST。最終分類は **PARENT_ALLOCATION_TECHNICAL_INCONCLUSIVE**。
固定S2の直接子A2から一回だけ起動し、総CPU上限3600秒で停止した。retry=0、marker消費済み、mandatory STOP。
最終resource map・比較分類は未完成であり、保存prefixを配分利益・非利益や研究の継続判断に使わない。

## 正本と実行identity

| 項目 | 固定値 |
|---|---|
| Science source S2 | `a874f443411e03ad4d9d329834f09073a64a6621` |
| Authorization-only直接子A2／実行HEAD | `2f4886c9f0389311acbaa57724d8c2a428e51809` |
| branch | `track-b-parent-allocation-v2-one-shot-execution-20261015` |
| Contract SHA256 | `a197f8e92056fa8f6c33c8a2d4d4d653d711343a221c97e21ac78c7a686de896` |
| Authorization SHA256 | `9cab18b4d8e78bd50e9c099644e4f3c2d1ca972ce7eb9521bb65ee7949325727` |
| Source manifest SHA256 | `ca871c9eb4720cde3e726aa9d8f47706babd5dd719f933214cd9a82c85393af4` |
| 新marker SHA256 | `eeb6f2961532a0c02814e0f6dab72090767f5cd23bcdf9be1b3e8a7f3b85fba1` |
| 旧v1 marker SHA256（不変） | `257112f17667e42550b434ced6aea4cb926ebad939adbbba0e0c806d4e01ad7c` |

利用者の原文「次の作業に進んで大丈夫です」を、直前の固定S2 one-shot承認待ち報告への承認として
[authorization receipt](parent_allocation_v2_execution_authorization_receipt_20261015.md)とJSONへ記録した。
A2の唯一のparentはS2で、diffはv2承認JSONとreceiptの2 pathのみ。公開確認HEAD `0cf6a90735e6e8d304d1c37ad79898d2e6921d9b`は実行に使用していない。
launch前にremote A2一致、clean、164 critical hashes、140旧保護ファイル、固定runtime、新namespace未存在を確認した。

正本は[原出力ディレクトリ](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v2/)。
[terminal](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v2/terminal.json)、
[outer process receipt](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v2/outer_process_receipt.json)、
[STOP](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v2/STOP.json)、markerは実行終了時のbytesを保持する。
本報告や監査はこれらを置換しない。

## 固定条件と到達段階

既知development用3-qubit distinct-basis synthetic。p=(1/5,3/10,1/2)、x=tau=5/7、状態000、sigma=+1、r=1、共通finite P5。
Q0=Z0、V1=R_XX01(pi/4)、V2=R_XX12(pi/4)R_ZZ01(pi/4)、Qi=Vi†ZiVi。
分子geometry・basis・DF rank・split L_Dを持つ分子DF実験ではない。
primitive精度1e-6、K256/H160、rho=eta=1e-12、strict phase guard 100dps、axis精度1/200、alpha_axis=1/40を維持。
合成worker 2、numeric threads 1、全科学条件・guard・資源上限は変更していない。

| 段階 | 保存された事実 |
|---|---|
| source／runtime／authorization gate、exclusive marker | 通過、runner起動一回 |
| 使用旧unique sequenceのstrict guard | 20件 |
| 旧common価格に基づく候補・C* freeze | 15構成。新価格による候補選び直しなし |
| inventory | unique 36＝旧20＋新16。新key上限18内 |
| 新合成 | 16 worker結果すべてKEY_COMPLETE、各一回、各sequence・error・T/T†・1Q・W記録あり |
| 登録root controlled診断 | ideal／realized計72件の保存記録 |
| event binding | 全15構成、計2562 bindingを保存 |
| law／row | law_00〜60、row_00〜60の61組を保存。次のlaw_61.json.partialを保持 |
| 最終resource_map／evidence_manifest／比較結果 | 未生成。terminal coverage=null |
| quantum shots／RTE sampling／trajectory／LP／DF／GPU／旧runner再試行 | 0 |

61は保存済みprefix行数であり、全比較の完遂数ではない。contract上のlaw数は最大75で、inapplicable proposalを許す。
未完遂runから最良候補、allocation gain、negative evidenceを作らない。
v1で停止したowner PID受渡しは今回は16 keyの完了まで通過したが、v2全体の科学実行成功とは区別する。

## 停止理由・内外status・資源

停止理由はouter receiptとterminalに共通する **aggregate CPU cap**。
外側ownerが上限を観測し、ledger subprocessへSIGTERMを送った。

| 項目 | 保存値 |
|---|---|
| public runner（toolで観測）のexit | 1 |
| outer receiptのexit_code（ledger subprocessのexit） | -15＝SIGTERM |
| inner terminal | 未保存、inner_status=null |
| final classification | PARENT_ALLOCATION_TECHNICAL_INCONCLUSIVE |
| sampled aggregate CPU | 3600.0600000000004秒 |
| completed ledger tree CPU | 3285.188201秒 |
| outer process CPU | 314.922547077秒 |
| completed CPU合計 | 3600.110748077秒 |
| wall | 3284.2323729647323秒（約54.74分） |
| sampled aggregate peak RSS | 392,511,488 bytes（約374.33 MiB） |
| completed child peak RSS | 165,124 KiB |
| 新16 worker CPU合計 | 7.935340999999999秒 |
| 全原出力の最終bytes | 19,699,323 bytes（監査追加前） |

sampled CPUとreaped CPUの差は計測時点・粒度の異なる値であり、同一値とは扱わない。
3600秒を最初に超えた観測で停止している。RSS・output・logの上限超過を停止理由とする記録はない。
inner terminal欠落は正常完遂とは扱わず、outer statusを保持する。
[public runner観測記録](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v2/saved_output_audit/public_runner_completion_observation.json)でpublic exitとledger exitを明示的に分けた。

## 技術的なボトルネックの整理

**確認済み**：最後の保存位置は`laws/law_61.json.partial`で、停止理由は総CPU上限。新16合成の実CPU合計は約7.94秒。
外部read-only snapshotでは、合成完了後にevent JSON、続いてlaw JSONのpartial bytesが増え続けている。
実行中のsource、上限、出力方式は変更していない。

**静的sourceと観測からの有力な説明**：
[EvidenceWriter.write/check/permit_bytes](../../../src/trottertracks/algorithm_codesign/rte_development/evidence.py)は、
`JSONEncoder.iterencode`の細かなfragmentごとにbudget確認とflushを行う。
budget確認には[Guard.check](../../../src/trottertracks/algorithm_codesign/rte_development/launch.py)の結果tree走査と、writerのtotal_bytes走査が含まれる。
この反復に加え、周期的guardと[外側ownerの/proc・出力tree監視](../../../src/trottertracks/algorithm_codesign/parent_experiment/runner.py)が動く。
JSON保存時の大量の短いfragmentと繰返し走査を、優先して調べる技術的ボトルネックと考える。

CPU上限と保存段階の特定は確信度が高い。**各関数がCPUの何％を使ったかは未測定**であり、全CPUをserializationに帰属しない。
function profiler・stack tracebackは取得していない。native費用会計や有理数演算の寄与も、今回の証拠だけでは完全分離できない。
[技術帰属記録](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v2/saved_output_audit/technical_stop_attribution.json)と
[外部snapshot](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v2/saved_output_audit/external_progress_snapshots.jsonl)を参照。

## 保存値・provenance監査

[監査JSON](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v2/saved_output_audit/saved_output_audit_v2.json)と
[実行したread-only script](../../../artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v2/saved_output_audit/audit_saved_output_as_executed.py)を保存した。
**148項目PASS**。A2直接親・remote・source/contract/authorization/manifest、worker ownerとfreeze binding、
sequence identityと保存strict guard、root診断、marker/STOP/terminalのhash chain、全complete chunkのUTF-8・lossless復元、旧marker、親worktreeを照合した。
原出力315ファイルのpath・bytes・SHA256を索引化した。

164 source/input critical hashesと140旧保護ファイルは不変。親HEAD/index、69 status entriesと130既存ファイルのbytes/hashも不変。
operator guard、matrix、合成、runner、N計算を監査中に再実行していない。
未完遂runのlaw moment・費用を新しく再採点せず、prefixの科学的比較も行っていない。
既存78件のfocused testsは固定sourceに保存された事前証拠を参照し、今回は再実行していない。

## 次の技術作業と判断の境界

最小の修正候補は、科学値・JSON内容・誤差会計を変えずに、出力fragmentを固定サイズのbufferへまとめ、
budget走査とflushを適切なblock単位にすること。周期的CPU/RSS guard、hard CPU上限、byte/log/call上限、exclusive保存・partial保全は維持する。
これは**未実装・未採択の技術案**であり、速度改善率も未測定。
まず登録外の代表的なサイズの人工JSONで、bytes/SHA一致、UTF-8 chunk復元、上限・中断時の保存、CPU負担を検証する必要がある。
CPU capの拡大だけを解決策としない。

今回S2を変更・再起動していない。修正するなら新source S3、更新manifestと保護ledger、新namespace、必要な契約versionが必要。
科学条件を維持する技術準備について追加GPT科学レビューを必須とする具体的な論点は現在ないが、今回のone-shot承認は新しい実行を認可しない。
再実行は、新source固定後の別途承認とauthorization-only直接子が必要。旧A1/A2・marker・prefix・STOPは保持する。
保存prefixの利用方式や科学契約を変える判断は独断で行わない。

GPTには、**まだ配分の資源価値を判断できる完成比較がない**ことと、この技術停止原因・修正候補を返す。
研究上の採否・新規性・科学的GO/STOP・追加実験は判断しない。科学レビューの開始承認とも混同しない。

結果・監査・公開確認をcommit/pushした後、**mandatory STOP**。追加修正、再実行、次stage、GPTレビューを自動開始しない。
