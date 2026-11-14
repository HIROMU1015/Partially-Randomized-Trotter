# D7較正の継続用メモ（チャットを一旦終了、計算は継続）

ユーザーが計算待ちのチャット終了を許可した。科学workerは停止・再起動していない。**再開時に再投入しないこと。** 最終報告・結果公開は未完了。

## 固定SOURCEと範囲

- worktree：`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d-full-h-p2-caller-prep-20261114`
- branch：`codex/d-full-h-p2-caller-prep-20261114`
- SOURCE／固定入力commit：`d323fee1ca8cd6893c4b705f990cdc931bdb5c87`。push済み。HEADはこのまま。
- 旧STOP：`df0e11fd1`。旧SOURCE・旧STOP資料は無変更。
- `FROZEN_JOBS.json` のD7較正5 cell・9 path・最大18実compileだけ。HOLDOUT、fit、E[C]、N/Gなし。無断retry・別seed禁止。
- 限定修正のmetadata 11 test／toy 2 testはPASS。親event直接復元でsampler/sample_events 0、位相・順序一致、子X/Y fake compile 2を確認済み。

## 最終観測時の状態

| cell | 状態 | 既存session | 科学child PID（次回はcommandも照合） |
|---|---|---:|---:|
| G566 | 完了・4 compile PASS | 終了済み | 終了済み |
| P2R566 | 完了・2 compile PASS | 終了済み | 終了済み |
| Q8R566 | 実行中。compile record 3、path0 trial PASS | 47841 | 3153060 |
| G4528 | 実行中。compile record 2、path0 trial PASS | 90597 | 3154188 |
| G36223 | 実行中。compile record 0、RSS約8.6 GB | 7375 | 3155960 |

観測時のSTOP記録0。G566のX/Y RZはpath0 82,785、path1 87,375。paired P2R566のX/Y RZは169,888、`SAVED_PARENT_EVENTS_NO_DRAW`／`PAIRED_EVENTS_IDENTICAL`。価格は有限cohort観測で、law期待値ではない。

public出力：本dirの `run01/<cell>/`、console・time・実行jobは `run01/` 直下。
private：`/home/AbeHiromu/projects/d-full-h-p2-calibration-20261115-private/resume01/<cell>/`。native trialは `cost/<cohort>/<physical_id>/<index>/result.json`、異常は `STOP.json`、compiler原記録は `compiled_cache/*.json`。

各cellを別jobで、G566→paired→Q8→G4528→G36223の順に投入した。G566資源実績・失敗なし・十分な空き資源を観測して、独立cellの実行を重ねている。native/wrapper/compilerは固定のまま。全jobは既に投入済みであり、新たな科学jobは不要。

## 再開後に残る作業

1. 上記既存jobの終了状態、public RESULT、private STOP、console/timeを確認する。重大失敗なら結果・原因を保全してSTOPし、retryしない。正常workerを不要に中断しない。
2. 全取得後、各path/軸のPRICES・IMPLEMENTATION_EVIDENCE・native個別compile recordのhash/metrics対応とcorrectnessを確認し、予定／再利用／実行／PASS／FAIL／未取得を確定する。
3. raw `RESULT.json` は大きいため元bytesを保全したままgzipで公開する。G566・P2R566は `RESULT.json.gz` と `RAW_RECEIPT.json` を準備済み。元rawはscoped `.gitignore` で除外され、localに残っている。大RのgzipがGitHubの100 MiB制限を超える場合は64 MiB以下のbyte partsへ分割し、RAW_RECEIPTに順序・各hash・全stream hashを記す。元rawと復号bytesのhash一致を確認する。数値の再解析ではない。
4. SOURCEを変更せず、取得結果をcommit/pushして結果SHAを確定する。その後、本dirの `READ_RECEIPT.py:bind` で圧縮receiptを明示復号して既存 `P.records` と照合し、Cの既存 `evidence` readerで小さい公開JSONを読む。固定callerの `bind_calibration_publication` は未圧縮RESULTを想定するため、gzip公開には直接使わない。
5. 公開SHA付きsidecar／C向けHANDOFF・固定locator・親子依存・実際に通過した検査・未確認事項・最終REPORTを作成する。D1の検査とC担当の科学的受入未確認を区別し、canonical round guardは使わない。
6. 必要な索引と研究ノートを追記し、commit/push、remote SHA/treeを照合。ロードマップへの影響を報告するが本体は編集しない。全較正引渡し後、または重大gate失敗でmandatory STOP。

結果取得中に科学SOURCEや入力を変更しない。既存C fit・m5/m6 HOLDOUTへ自動進行しない。

## 並列化・高速化指示後の追補

Q8R566・G4528も完了（既存session 47841/90597ともexit 0）。元の較正は4 cell・14実compile PASS。両cellのRESULT gzip化とbyte一致検証も完了。旧SOURCE/HEADはd323fee1cのまま。

高速化SOURCEは別worktree `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d-full-h-p2-acceleration-20261115`、branch `codex/d-full-h-p2-acceleration-20261115`、commit `f8d648ab99c4c1eb84723dacc1452b8e3f525482` に公開済み。remote SHA/tree一致・clean。path/X/Y process並列、gate memoization、legacy fast fingerprint、保存checkpointのsampling/compile再実行なしの継続を実装し、8 synthetic test PASS。実H4には未適用。

旧workerを停止してcheckpointへ切り替えるscript案は自動承認レビューが拒否し、作成・実行していない。高速化指示だけでは進行中worker終了の許可が明確でないという理由。旧workerは停止していない。新entryは旧workerが生きている間の同一path起動を拒否する。**ユーザーに必要なのはcheckpoint保存後の旧worker終了と新SOURCEへの引継ぎの明示許可のみ。** 拒否を迂回しない。

最新観測：`{"observed_utc": "2026-11-14T18:56:56.156014+00:00", "compile_records": 2, "checkpoints": []}`。旧G36223 session 7375、child PID3155960。既存processを確認し、再投入しない。高速化側REPORTに具体的引継ぎ計画あり。

## 最新追補：許可済みcheckpoint引継ぎ実施（2026-11-15）

上記の未適用・旧worker継続の状態は過去の観測である。ユーザーの「引き継いでよい」を受け、SOURCE `887c3b9d04d15be96637b6634265424e7dd50339` のcontrollerで旧G36223 path0 native checkpoint（両compile・7 gate PASS）を保全し、旧workerを終了した。旧parentのSTOP_CELL/exit2は `CONTROLLED_AFTER_NATIVE_CHECKPOINT` であり、科学gate失敗ではない。path1未着手を確認済み。旧RESULT/raw/QPYとhashは変更していない。

新worktree：`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d-full-h-p2-acceleration-20261115`。HEAD/source887を固定。新session48999、path worker初期PID3247396（p0保存結果の残検査）と3247398（p1初取得）。新privateは `/home/AbeHiromu/projects/d-full-h-p2-calibration-20261115-private/acceleration01/G36223`。出力は `docs/research/d_full_h_p2_acceleration_20261115/run01/G36223`。**再投入しないこと。** 残新compileはX/Y各1の計2、path0はsampling/compileを再実行しない。

元D1 branchは旧worker終了後に887へfast-forwardした。公開資料・結果commitは元D1 worktreeで行い、実行中の新worktreeHEADは変更しない。G566/P2R566/Q8R566/G4528の14軸価格・個別compile原記録・lossless gzipは取得済み。G36223完了後の証拠公開、C metadata readerでの照合、HANDOFF、remote照合が残る。Cの科学的受入・fit/凍結・HOLDOUT・law期待値・N/Gはいずれも未実施。

## 最新追補：path1完了・path0長時間化と再発防止修正

G36223新privateの `path_result_1.json` はPASSで保存済み（約1.04GB、X/Y各RZ5572406、両compile完了）。path0はSOURCE887のworker PID3247396で残検査を継続している。外部nonblocking stack取得は通常OS権限で拒否され、root/sudoや共有設定変更はしていない。CPU稼働・STOP未検出だけでは検査の前進や最終PASSを断定しない。総残時間は不明。

元D1 worktreeだけでcheckpoint wrapperの高速fingerprint適用漏れを修正した。既存checkpoint toy testへ旧writer禁止を追加し、修正前は実wrapper内部の旧writer呼出しでFAIL、修正後は9 test PASS。詳細は `../d_full_h_p2_acceleration_20261115/RELIABILITY_FIX.md`。**修正は実行中のacceleration worktreeには適用していない。新たな科学jobは起動していない。再投入しないこと。**

共通AGENTSにも、実経路の適用確認・長時間処理の観測・根拠付き見積り・小さい再発検査を追加した。既存のSOURCE固定・科学許可・保全・mandatory STOPは変えていない。全較正の取得・公開・C引渡しはまだ未完了。
