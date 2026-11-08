# 保存q4単一pathへの最小接続と再利用

## 追加と再利用

| 処理 | 使用箇所 |
|---|---|
| 不変cut選択 | 人工基点a54d0405の `track_d_cut_selection.select_cut`。score/係数/tie/candidateはbyte不変 |
| 特徴 | 固定64072ba8の `event_features` をGit bytesから一時import。SOURCE SHA256を照合、旧価格解析/fit runnerは呼ばない |
| 入力binding | 新 `track_d_cut_h4_saved.checked_payload/checked_record`。archiveと全memberを既存公開hashへ結び、保存q4_sample00の4×32 eventだけを受け付ける |
| 選択入口 | `run_track_d_cut_h4_select.py`。SOURCE/file hash/依存versionを固定してからdecode、call開始→decisionを一回保存。価格は開かない |
| 保存event | 既存 `track_d_h4_reference.stored_events` のlossless serializerを使い、順序・符号・phase・normalizationを再生成せず保存値のまま照合 |
| 中央cut再利用 | 新 `track_d_cut_h4_reuse` と薄い `run_track_d_cut_h4_reuse.py`。選択が中央2の場合だけ旧request/old PASS/P0/QPY/価格の資格を確認。非中央を中央価格で代用しない |
| 全資源 | 既存 `track_d_minimal_reference.metrics` で保存compiled4本を確認。新しい価格providerやcompileはない |
| 構造 | 既存C7-2 `weighted_depth` とQiskit DAG。保存A†/B/Reference coreとXの2方式だけ、追加compileなし |

既存C6a CLIはsamplingと32 path/128 compileを起動するため、今回呼んでいない。今回必要だった追加は「提供archiveからlossless inputを結ぶ」「既存selectorを一回呼ぶ」「一致する旧価格を再利用する」という接続だけ。C3 `prepare_steps/physical_segment`、Reference gadget/control_one、StrongBuilder、compiler/checkerのSOURCEは変更していない。非中央cutを作る新adapterは結果上不要だったため追加していない。

## 人工検証

実event投入前にarchive mismatch・member mismatch/欠落・event byte mismatch・lossless memberを人工5件で確認した。decision後の再利用部分は別の人工4件で、別path/非中央を拒否、4 wrapper欠落を拒否、既存error threshold/rtolを緩めないことを確認した。9件PASS、記録は `artifacts/track_d_cut_h4_q4/2026-11-08/technical_completion/`。分子信号や資源改善の証拠として人工PASSを扱わない。

## 適格な旧結果の対応

旧資料は `artifacts/track_d_c6a_paired_price/2026-11-06/run01/`。`event_bindings.json`、`payload_archive.json`、`prices.json`のrequest/event digest、precompile/paired/4 postcheckとcompiler recordを、今回のarchive全20 memberと照合した。旧raw core/wrapper/QPY/参照NPZのSHA256は一致し、選択は同じcut2。ファイル配置の差で再構築する必要はなかった。

値の出所は古い科学実行であり、今回SOURCEで全256入力simulationを新たに走らせたという主張ではない。旧correctnessの精度はabs1e−10/observable Frobenius2e−10/rtol0/phase fitなしで、全入力・leakage・χを含む。再利用判定と保存全7資源の一致を [REUSE_RESULT](../../../artifacts/track_d_cut_h4_q4/2026-11-08/run01/REUSE_RESULT.json) に残した。

8 QPYを読み取って、同じ幅(9/17)、bound回路、ancilla末尾→bit0、XのH・YのSdg→H readout、全7資源を確認した。A†を再合成したりVへgateを作用させたりしていない。Canonical意味論は、同一request/cutとbyte一致する保存回路を使用し、元builder/最適化/phase compiler・PASSが不変であることから継承する。別IRを価格だけで同等扱いしたのではない。

## SOURCEと原本保全

選択codeは実event decode前の `SOURCE.json` のfile SHA256で固定した。再利用code/旧証拠は個別価格処理前の `SOURCE_REUSE.json` に追加固定した。source commit基点と実行file hashを区別する。結果公開commitは実行後に確定し、その公開file bytesが実行時hashと一致することを検証する。

実行の再現入口は上記2 runnerだが、今回は一回完了済みで再実行しない。選択入口は既存directoryを拒否し、再利用入口は開始marker/STOP/失敗のある出力を拒否する。既存STOPを解除・上書きせず、提供raw archiveと展開物をpublic treeへ追加しない。

前回preflight01の全記録は保全し、今回はrun01へ保存した。旧SOURCE・C6a event/QPY/結果、元main/dirty/untracked、C1凍結と他job資源を変更していない。q64の未完了結果を取得・最適化用データにしていない。
