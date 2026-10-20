# C1-1-D0 実行前静的監査・診断仕様

対象は保存H4 1.00 Å / STO-3G / DF rank 12 / 8 system qubit / 生成順prefix3。前回C1-1 `4cfe1c257709f0ac09351eadce3952b81403cf93` を基点とする新branch/namespaceのみへ追加する。今回のD0指示はPhase AとPhase Bの両方を明示承認している。旧SOURCE/失敗/raw/auth/STOPは変更せず、新しいSOURCE commitと、その直後の新authorization-only commitで1回・retry 0を固定する。

## 既存SOURCEとの対応

固定科学参照は `243eea8e840f3c472753f7935e510391626fd325`。[元PR-2結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/artifacts/pr2_new_series_validation/2026-09-28/pr2_v1_v3_result_v1.json) のprovenanceに記録された主要6ファイルのSHA256も全一致。[機械記録](static_reference_audit.json)。

| 既存定義 | hashへ至る役割 |
|---|---|
| `pr2_s0_s1_validation.generation_ranked_fragments` | original index順、rank=index、stored λをfloat化、`df_fragment_weight`を取得 |
| `generation_partition(h,3)` | 上記を `split_df_hamiltonian_by_ld` へ渡す |
| `split_df_hamiltonian_by_ld` | 入力順prefix3と残りを保持、`float(sum(randomized weight))`をranking proxyとする |
| `df_fragment_weight` | `abs(float(lambda)) * np.linalg.norm(g, ord="fro")**2`、丸め直しなし |
| `df_hamiltonian_hash(h, weight_rule=partition.weight_rule)` | exact array content SHA、constant hex、12桁丸めのnorm/λ、metadata、weight ruleをcompact sorted JSONでhash化 |
| `df_partial_s2._partition_hash` | H hash、ld、両群の全 `RankedDFFragment` field、proxy sum、weight rule、hash policyをcompact sorted JSONのUTF-8 bytesへ写してSHA256 |
| `prepare_df_partial_s2` | partition検査 → H hash → partition hash → model/basis構築。この関数全体はD0では呼ばない |
| `pr2_new_series_validation._method_record` | B2-Gにはexplicit ordered policy、B2-Wにはweight-ranked policyを渡す。D0はB2-Gと生成順を固定 |

[生成順](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/pr2_s0_s1_validation.py#L269)、[分割](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_randomized_pf.py#L234)、[weight](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_randomized_pf.py#L180)、[H hash](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_randomized_pf.py#L990)、[partition hashとprepare順](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_s2.py#L210)、[元_method_record](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/pr2_new_series_validation.py#L448)。

`RankedDFFragment` の全fieldは `rank, original_index, lam, weight, weight_rule`。`partition_hash_policy`は今回 `explicit_ordered_df_partition_v1`。`partition_policy`というAPI引数自体とは異なる文字列。SOURCEのJSON規則は `sort_keys=True, separators=(",", ":")`、既定ensure_ascii=True、UTF-8、hash対象には末尾改行なし。可読JSONとはbytesが違うので両方保存する。

NPZ全体SHA、各配列digest、Hamiltonian hash、partition hashを区別する。同じ配列contentでも、norm/weight/sumの算術実行やJSONへ渡る値の差をpartition hashが検出し得る。NumPy/SciPyのversion文字列だけでbinary/BLAS/CPU算術実体まで同一とは言わない。小数表示が似ていてもfloat.hexで一致しなければbitwise一致ではない。現時点で特定fieldやruntimeを原因と断定しない。

## D0の最小経路と証拠

1. 固定NPZを1回read-only openし、constant/one_body/lambdas/g_matrices/metadata_jsonだけを取得する。state/sector配列は開かない。既存 `_array_hash` とH hashを保存V1へ照合する。
2. 既存 `generation_partition(h,3)` を1回呼ぶ。H hashには同じweight ruleを使い、partition ruleと同一であることを検査する。
3. 固定 `_partition_hash` の最初のpayload代入式をASTでそのまま評価し、canonical bytesを保存する。unmodified `_partition_hash` を1回呼んだ値とbytesのSHA256を照合する。旧科学SOURCEは変更せず、別payload/serializationを探索しない。
4. Expected比較より先にactual hash、payloadの全field、λ/weight/proxyのvalue/repr/hex/typeと依存関係をdurable fileへ保存する。
5. 対象B2-Gの順序/policy/rank、保存H hash、保存fragment metadataをfield比較する。同じsnapshotの他rankに既に保存されたfragment recordも参照し、0..8のfieldの比較範囲を補う。新しいrank計算・探索・再分類は行わない。

元のcomplete partition payloadやcanonical bytesは指定結果に保存されていない。`_fragment_record`の保存λ/weight/indexは比較可能だが、元partition dataclass payloadそのものではない。9..11のweightやproxy sumなどの原値はこの結果では不足する。欠落fieldをactualで埋めて旧hashの再現を試すこと、field置換・hash探索・閾値緩和は行わない。具体的不一致が見つかっても一意なhash原因を裏付けられない場合はPARTIALLY_IDENTIFIEDに留める。

新SOURCEは既存moduleのimportを行うが、その中のbasis/回路/分子生成等の定義は呼ばない。前回と同じ軽量package namespaceでexport-only `__init__`を迂回する。対象Python関数のprofile guardで禁止関数の呼出しを拒否し、件数を保存する。これはC拡張内部までの全call証明ではない。

## Loggerと事前確認

`emit(event_name, details)` に詳細をnestし、`stage`とterminal statusを引数展開しない。一次例外のtype/message/tracebackを保持し、loggerや保全の二次障害は別ファイルへ保存する。通常hash mismatchもfailure記録とterminalを保存し、exit 2と対応させる。matchはexit 0、技術失敗はexit 1。stdout/stderr/outer process exitは専用runtimeへ直接captureし、終了後に新resultへbytesをcopyする。

[人工logger checks](artificial_logger_checks.json)はstdlibのみの4ケース：stage付きmatch、stage付きmismatch、一次例外、一次例外＋終端logger例外。全成功。科学module import/call、NPZ open、basis構築は0。実際の科学入力を使った試走はしない。構文は `ast.parse` で確認した。

実行前固定対象は [契約](execution_contract.json)、[SOURCE manifest](source_manifest.json)、[環境identity](environment_identity.json)、[保全baseline](preservation_baseline.json)。既存隔離runtimeをそのまま使い、依存版の探索や追加はしない。CPU/RAM/workerへ独自上限は置かず、逐次依存・一回限定・最初の技術失敗でSTOPする小規模診断として実行する。basis/λp/モデル差/adapter/native/samplingへ進まない。

Codex本人による静的確認と人工確認であり、別agent/reviewerの独立レビューではない。正式共通運用指示2026-10-17、projects/AGENTS.md、worktree/AGENTS.mdを適用した。ユーザーの新namespaceのみへの追加指示を優先し、既存status/index/ledger/STOPは書き換えない。
