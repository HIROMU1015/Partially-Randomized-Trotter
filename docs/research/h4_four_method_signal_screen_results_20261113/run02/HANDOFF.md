# A → B：run02の4局所recordと未完了screen

2026-11-13 JST。科学SOURCE`256caad3414a80de02fb68eca89c13c957ed6354`を変更せず、ユーザー承認でimport経路を補正して1回再実行。**全体状態はSTOP_NUMERICAL_OR_BINDING_FAILURE。4セル保存・6数値gate失敗・9未開始・1空例外未解決。** 詳細は[REPORT](REPORT.md)。

## 受渡し

- [result.json](result.json)：固定実装の原結果。4 recordと16 errorを上書きせず保存。
- [PLAN.json](PLAN.json)：元20セルと保存40 locator。前段の固定PLANと同一。
- [SCREEN_STATUS.json](SCREEN_STATUS.json)：20 candidateを完全に分け、保存4件のsignal/local/numericalをcopy。失敗値と未開始値はnull、科学的PASS/FAIL・十分N/global RMSE/winnerもnull。
- [SCREEN_SUMMARY.csv](SCREEN_SUMMARY.csv)：保存4件の診断fieldを表示・転記し、他16件は空欄。空欄は0ではない。
- [EXECUTION.json](EXECUTION.json)／[起動ログ](../RUN02_EXECUTION.log)：同じSOURCE、NumPy1.26.4、`src/`＋`scripts/`、fresh run02。run01を保全。
- [TECHNICAL_CHECKS.txt](TECHNICAL_CHECKS.txt)：7件のmetadata・SOURCE・保存field整合性確認。新mean、normalizer、reference、decoderや価格を呼ぶ検査ではない。

保存4件はFull PF/prefix10/m6/T6.4のS2・標準Y4×Δ.1/.4。raw=corrected、B1/logB0/attenuation1。これらに元SOURCE付きreference・allowance・binding/profileが付く。旧HP型NやPR nominalをbaselineのNへ流用しない。

6 gate失敗はF2/F4のm12各Δ、F5-FINEのm6/m12。normとunitarityの合成gateで、閾値1e−9は変更していない。失敗時の残差scalarは元実装が保存せず、どちらが超えたかは不明。新たに復元していない。失敗をPF公式そのものの科学的敗北やglobal RMSE FAILへ読み替えない。

空例外のR4-ADAPT/m12は取消と整合するが、例外型未保存のため未解決扱い。normalizerは保存record上0、実総call数の確定値はnull・範囲0〜1。Full-Hの他5件は明示的に未開始。新referenceは0、保存40の再計算も0。

## Bへの制約と次の有限案

4件だけを`LOCAL_DIAGNOSTIC_SAVED_NOT_RMSE_QUALIFIED`として参照する。全scheduleの科学的資格、十分N、global RMSE bound、canonical configuration、C/G/winnerを生成しない。m12・M5・B0I・Full-H RTEの情報が不足し、全13round候補の採択材料は揃っていない。旧40セルは元SOURCEと既存資格を保持した表示だけに使い、今回新しい比較・再診断を追加しない。

次は**A**が閾値を変えずに数値安定性と失敗診断保存をSOURCE・人工入力で調べる技術案。必要なら**ユーザー**が代表1失敗セル（例F2-COARSE/m12）の有限な実数値診断を別途承認する。独立経路照合とmacro-step/powerのnorm・unitarity・spectral orthogonalityを対象にし、新reference0、候補不変とする。固定SOURCEの無断変更や全20再走はしない。

**B**は4件の限定観測と未評価状態を保持して次の計画へ接続する。C/Dの価格条件、law、8次PF、M/κ/δ最適化に拡張しない。今回の部分成果公開・照合後、Aはmandatory STOP。
