# P1→P2：SOURCE・raw・再現条件

科学計算は`STUDY_STOP.json`でmandatory STOP済み。以下は実行済み研究の依存順の説明であり、再実行の指示ではない。数値再解析・再計算には別の許可scope、fresh checkout/cache/outputを使い、保存STOPを削除・編集しない。

## 保存物とbyte確認

研究rootは`artifacts/track_c_df_cancellation_validation/2026-11-08`。REPORTの表は`P1_RESULTS.csv`、`P2_POINTS.csv`、`P2_ROUNDS.csv`、`COMPARISON.csv`、`RESULTS.json`に対応する。`P2/POLICY.json`は独立照合前の凍結policyで、`P2/FINAL.json`と`FINAL_CHECK.json`が完了状態。source欄はpublication commitではなく実load時のSOURCEを指す。

`RAW_INDEX.json`は全raw memberのbytes/SHA256、`ARCHIVE_PARTS.json`は分割archiveのbytes/SHA256と連結後のhash。`ARCHIVE_VERIFICATION.json`はtar全memberと原fileの照合結果。番号順に連結すると標準tar.gzになる。元file bytesは変更せず、tar headerのみ標準化した。展開先は空のscratchとし、公開checkoutや原rawを上書きしない。

```bash
RAW_BASE=artifacts/track_c_df_cancellation_validation/2026-11-08
RESTORE_DIR=$(mktemp -d /tmp/track-c-p1p2-raw.XXXXXX)
cat "$RAW_BASE"/raw.tar.gz.part??? | tar -xz -C "$RESTORE_DIR"
```

part/連結hashを照合してから展開し、RAW_INDEXの全entryを再照合する。これはbyte検査であり科学再計算ではない。rawにはP1新4点、旧n0を含む最終5点表、P2固定pool/cohort、16個のHkeep spectral配列・証拠、全157資格、凍結・独立結果、人工test、失敗logも含む。旧pickle・自分のreplay cacheは非公開原本として保全し、公開rawには含めない。公開済みの元INPUT_ARRAYSを重複梱包しない。

## 固定参照の復元

元科学研究は`da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13`、静的調査・今回の基点は`12e74dd68568f493ed39937f8856f26ce48d33e1`。元研究の[REPRODUCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_df_truncation_20261108/REPRODUCE.md)に従って元rawを別scratchへ復元する。

参照rootは必ず`<reference_checkout>/artifacts/track_c_df_truncation/2026-11-08`のdirectory構造に置く。`prepare_model()`はこのrootの`parents[2]`をpublic replayのcheckout rootとして使う。必要なのはH4/H6のINPUT.json/INPUT_ARRAYS.npz、保存scan・native価格・資格・ordering結果・独立check等の元raw一式。SOURCE.jsonは元RAW_INDEXと入力hashを固定する。元H/state/B/sp/Pauli係数/gate IRからreplayし、新SCF・DF fit・初期状態生成を行わない。新資格の既存ground residual補助関数には固有分解が含まれる。

旧runner/STOPは再開しない。再利用writerは新rootの`checks/`へredirectし、参照rootは読取専用にする。

## 実load SOURCEと環境

`SOURCE_HISTORY.bundle`は基点12e74ddからcode tip `edb013f09de01d6bdb9184cc8da7347eb94baa9a`までの元Git objectを保全する。公開snapshotはGitHub App経由のためcommit identityが異なる場合がある。SOURCE.jsonのstage表とbundle内commitが科学SOURCEの対応先であり、公開treeの一致を別途確認する。

```bash
git bundle verify artifacts/track_c_df_cancellation_validation/2026-11-08/SOURCE_HISTORY.bundle
git bundle list-heads artifacts/track_c_df_cancellation_validation/2026-11-08/SOURCE_HISTORY.bundle
```

環境はPython3.12.3、numpy1.26.4、scipy1.14.1、mpmath1.3.0、Qiskit1.3.0、OpenFermion1.6.1、openfermionpyscf0.5、PySCF2.7.0、pytest8.4.2。ENVIRONMENT.jsonを参照。project専用venvで共有環境は変更しない。各BLAS1thread、workerはtask数と実affinityに応じる。今回affinity64、P2資格64worker。

## 実行したstageの依存順

以下を実行するのは別の科学実行許可があるfresh scopeに限る。保存namespaceは全entry pointのSTOP guardで再開を拒否する。cacheはcheckout内`.cache/df_cancellation_validation`なのでfresh checkoutを使う。既存cacheがある場合、prepare_modelは再生成しない。

```bash
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TRACK_C_CANCELLATION_REFERENCE=/absolute/reference_checkout/artifacts/track_c_df_truncation/2026-11-08
export TRACK_C_CANCELLATION_OUT=/absolute/fresh_output
```

1. `python -m trottertracks.track_c_df_cancellation_validation p1-plan`：整数5点と総時間を固定。n0再利用、新規4点のみ。
2. 同moduleの`p1-qualify` → `p1-independent` → `p1-finish`。最終finishはlogical rational hとbinary64のSOURCE差、scalar時刻transferを含む。初期FINALを保存し追加会計した技術履歴はREPORTとbundle参照。指定有限5点の解釈成立を確認してP2へ進んだ。
3. `python -m trottertracks.track_c_df_cancellation_conservative plan`：個別directed計算前に固定poolと157点cohortを保存。
4. 同moduleの`keep` → `qualify` → `freeze` → `independent` → `finish`。16 keep、157点、6policyを扱う。referenceの同一点の資格・独立証拠・実価格を確認して再利用し、新しい必要点だけを検査。
5. `python -m trottertracks.track_c_df_cancellation_results`：完了結果のCSV/JSONと限定ablationを出力。今回のexport・最終check後にSTUDY_STOPを保存した。STOP後はこのexportも実行しない。

人工testは次の2file（計13 PASS、`final_artificial_tests.log`）。分子の性能結果や独立研究者による再現とは別の確認である。

```bash
python -m pytest -q tests/test_track_c_df_cancellation_validation.py tests/test_track_c_df_truncation.py
```

独立expm/Schur/native X/Yは別評価経路という意味で、別AI/研究者によるレビューではない。P1新4点・P2新6点を照合。P2の参照再利用を含むunique照合点は43。新価格MC/実機shotは0。既存Full/PR policyとPR独立価格holdoutをそのまま用い、weighted推定量は保存規約を維持した。

保存した位相・ρ・β・α・整数N・全18round decoder/alias条件が科学的再現の判定対象であり、候補の大域最適性やplatform間のbitwise同順位は主張しない。過去logのFileNotFoundError、初期FINAL、修正履歴を消さずに保全した。P3/P4・別分子・無制限探索・新family・詳細GPTレビューへ進まずmandatory STOP。
