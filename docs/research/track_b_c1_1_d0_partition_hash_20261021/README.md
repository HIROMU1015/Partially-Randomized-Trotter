# C1-1-D0：partition hash原因診断

**`HASH_MISMATCH_CAUSE_PARTIALLY_IDENTIFIED`。D0のactual hash/payloadを取得し、同じ保存H4のfragment 6・7のweightにbit差を特定した。元の完全なhash payloadがないため、hash不一致の唯一の原因までは確定していない。実行1回・retry 0。公開・保全確認後mandatory STOP。**

C1-1全体の再実行・basis再構成・G6 exact λ/p取得は行っていない。前回C1-1の`REANALYSIS_INCONCLUSIVE`、SOURCE、raw、authorization、STOPは変更していない。原著価値・資源優位・投稿GO・adapter実装GOを判定する資料ではない。

## Hashと原因の確度

| 項目 | 値 |
|---|---|
| Expected partition hash | `722ec0077b4505d963c03bccdd6311f9dd8b3d3a5ed1b9117f7a6f8fa85a7450` |
| D0 actual partition hash | `16b35c1dde1a6807d39c733defd87e315e4ac7317a831b240d0f7214e744a834` |
| Canonical payload bytes / SHA256 | 1,834 / actual hashと一致 |
| Hamiltonian hash | `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`、保存V1と一致 |
| 分類 | `HASH_MISMATCH_CAUSE_PARTIALLY_IDENTIFIED` |
| Process exit / terminal expected exit | `2 / 2`（固定契約上の正常な診断mismatch outcome） |

[Hash identity](result/hash_identity.json)、[hash対象の正確なUTF-8 bytes](result/partition_payload.canonical.json)、[可読payload](result/partition_payload.json)、[float provenance](result/float_provenance.json)、[field比較](result/field_comparison.json)、[分類原記録](result/diagnosis.json)。canonical fileは末尾改行なし、可読JSONとは別bytes。Actual/payload/float証拠をexpected hashとの比較より先に保存した。

異なったfieldは次の2つ。各fieldが既存rank9のB2-G/B2-Wに記録されているため、比較record数では4件となる。新しいrank計算はしていない。

| Original fragment index / D0 payload path | 保存JSONをfloat読込したweight | D0 weight | 保存hex → D0 hex |
|---|---|---|---|
| 6 / `/randomized_fragments/3/weight` | `0.016458751994078695` | `0.0164587519940787` | `0x1.0da902630af0ep-6` → `0x1.0da902630af0fp-6` |
| 7 / `/randomized_fragments/4/weight` | `0.0001322875009981928` | `0.00013228750099819275` | `0x1.156d4fb1d2787p-13` → `0x1.156d4fb1d2785p-13` |

保存値のpointerは `/stages/V2/ranks/2/methods/{B2-G,B2-W}/deterministic_fragments/{6,7}/weight`。target rank3では同じlabelがtailに属する。計113件のfield比較中4 recordが不一致。他の比較可能な保存fieldは一致した。対象rank3のdeterministic metadataだけを見ると、この差は見えない。

このweightは既存 `df_fragment_weight` の `abs(lambda) * norm(g, fro)**2` に由来し、**DF fragmentのランキング用派生float値**である。G6 exact normalizer λ_Rではない。SOURCEは丸めないweightとtail sumをpartition payloadへ含める。[固定SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_randomized_pf.py#L180)、[hash定義](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_s2.py#L210)。

ただし元結果の値は `_fragment_record` が出力したmetadataであり、元のpartition dataclass全fieldやcanonical bytesそのものではない。元の9..11のweight、proxy sum等も指定結果にはない。したがって「この2fieldだけを置換すればexpected hashになる」「特定のBLAS/CPU/versionが原因」「元C1-1の未保存actual hashもD0と同じ」とは認定しない。欠落値をactualで埋めたhash探索、runtime切替、閾値緩和、再試行はしていない。

## 入力と最小実行範囲

Linear H4、1.00 Å、STO-3G、DF rank12、8 system qubit、生成順prefix3を固定した。対象snapshotは `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz`、8,652 bytes、SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。

constant/one_body/lambdas/g_matrices/metadata_jsonだけをread-onlyで1回取得し、必要な配列digest・constant hex・H hashを保存V1に照合、全一致。[入力記録](result/input_identity.json)。state/sector配列はD0では読んでいない。deterministic `[0,1,2]`、randomized `[3,4,5,6,7,8,9,10,11]`、ld=3、explicit ordered policyは保存値と一致した。

既存 `generation_partition` → `generation_ranked_fragments` → `split_df_hamiltonian_by_ld` を使用。`_partition_hash`内のpayload代入式を固定ASTから評価し、同じserializationのbytesのSHA256をunmodified `_partition_hash`の返値と照合した。`prepare_df_partial_s2`全体を呼ばず、basis構築前のhash計算だけを切り出した。[実行前SOURCE監査](SOURCE_AUDIT.md)、[診断SOURCE](diagnose_partition.py)。既存科学SOURCE自体は変更していない。

[Python関数call観測](result/scientific_call_observations.json)：generation_partition 1、generation_ranked_fragments 1、split 1、df_fragment_weight 12、H hash 1、partition hash 1。prepare/basis/DF matrix再構成/ground-state生成の対象関数は0。これは指定されたproject Python関数の観測であり、C拡張を含む全call証明ではない。SOURCE経路も静的に確認し、SCF、新DF分解、basis対角化、Gaussian構築、registry serialize、exact G6 λ/p、Hamiltonianモデル差、adapter、新回路/compile/synthesis、native費用、N、sampling、Pilot、Full-Nは行っていない。

## 実行前固定・logger・結果保全

| 項目 | 固定来歴 |
|---|---|
| Branch | `track-b-c1-1-d0-partition-hash-20261021` |
| Base（旧C1-1） | `4cfe1c257709f0ac09351eadce3952b81403cf93` |
| 科学SOURCE参照 | `243eea8e840f3c472753f7935e510391626fd325` |
| D0実行SOURCE commit | `835ae6f57eae43ba8452b71eae0ac9bce3dc2828` |
| 新authorization commit | `7e3ebb620cf3796b6c651c5dc6dfa03236730230`、SOURCE直後・receiptのみ追加 |
| Run ID | `track_b_c1_1_d0_h4_partition_hash_20261021_one_shot_v1` |
| 承認 | 今回のユーザーD0指示によるPhase A+Bの明示承認。旧receipt再利用なし |
| 実行 / retry | `1 / 0` |

[契約](execution_contract.json)、[SOURCE/input manifest](source_manifest.json)、[環境identity](environment_identity.json)、[新authorization](authorization_receipt.json)、[run identity](result/run_identity.json)、[command](result/launch.json)を参照。旧結果に記録された主要6ファイルSHAも全一致した。[静的比較](static_reference_audit.json)。

Runtimeは前回C1-1と同じ既存隔離環境を変更せず使用。Python 3.11.0rc1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0等。主要packageのMETADATA/RECORDとPython実体SHAを固定し、dependency追加・版探索なし。今回の[NumPy build metadata](result/numpy_build_metadata.json)はOpenBLAS64等を記録するが、元PR-2のbinary/BLAS実体との同一性を証明するものではない。

実行前CPU affinity64、MemAvailable `488644688 kB`、storage free `431582806016 bytes`。[資源観測](result/prelaunch_resources.json)。独自CPU/RAM/worker上限は設定せず、既存BLAS既定値を使用。記録されたpackage build設定にあるthread値は今回新設した制限ではない。逐次依存と最初の技術失敗でSTOPする小規模診断であり、不要なbasis処理を省いた。wall約2.584秒、own CPU約16.106秒、peak RSS `341416 KiB`（self/Linux。総量子資源や性能benchmarkではない）。

新loggerはeventとdetailsをnestし、stage引数衝突を解消。stdlib人工4ケースが成功。[チェックSOURCE](check_logger.py)、[人工証拠](artificial_logger_checks.json)。実one-shotでも[stdout](result/raw_stdout.log)に`MANDATORY_STOP`行があり、[stderr](result/raw_stderr.log)は0 bytes、二次logger失敗なし。[failure](result/failure.json)はhash mismatchを記録し、例外がないためtracebackを捏造せずnull。[terminal](result/terminal.json)のexit2と[outer process exit](result/process_exit.json)は一致した。科学例外が起きた場合の一次traceback保持は人工ケースで確認した。診断後のSOURCE修正・再実行なし。

[保全確認](preservation_verification.json)で185固定ファイル不変、旧C1-1実worktreeの26ファイルもbytes/SHA一致。非公開レビュー原本は不変・未公開。既存科学SOURCE/raw/claim/ledger/grant/marker/STOP、共有環境、他processを変更していない。全変更は新namespaceへの追加のみ。

[Run output manifest](run_output_manifest.json)に原ログ等を含む19出力のbytes/SHA256を収録。runner自身のinventoryは、outer log等のcopy前の時点の記録として保持。最終公開HEAD・全公開bytesの照合はpublication receiptと最終確認で記録する。

## 前回C1-1への影響と次の条件

今回得たのは**D0のactual hashと派生metadata差**。前回未保存のactual値の回収ではなく、前回結果や失敗分類を遡って変更しない。保存DF入力の内容が変わった証拠やoperator不一致の証拠にはならず、operator一致を証明したわけでもない。basis/exact model/model bindingは未完のまま。G6原著価値、資源優位、投稿GOを認定しない。

再開前にはユーザー・GPTが、元payload/実行binaryの追加回収が必要か、または今回の差を保持した新しい来歴のpartitionをどの契約で受け入れるか判断する必要がある。expected hashを黙って差し替えたり判定を無効化してPASSにしない。再開する場合は新SOURCEで比較前の証拠保存・loggerを固定し、basis/exact model/model差の必要範囲を明記した新契約と別の明示実行承認が必要。D0から自動再開しない。

担当と必要な承認は [handoff.md](handoff.md)。Codex本人による静的確認・人工確認・一回診断であり、別reviewer/agentによる独立再現ではない。**公開・保全確認後mandatory STOP。**
