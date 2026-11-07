# O2-A2：物理的なLCU測定アクセスを具体化、資源改善は不支持

2026-11-08 JST／SERVER1（GPUA100-gleap-01）。base `4e610af3c0a0b9745cff63b39b57ee1f97657cf6`、専用branch `codex/o2-a2-executable-access-20261108`。

**実行可能なLCUアクセスを具体化し、最小adapterと人工9件PASSを得た。一方、当該LCUは成功率/scaleを戻すとnoise床が残り、単純coherent分岐よりM2が大きい。資源改善の科学候補としては残さず、新しいO2科学実行の一時保留を技術提案とする。** O2全体の正式終了・一般NO-GOではない。

## 到達点

| 項目 | 結果 |
|---|---|
| 物理的アクセス | `EXECUTABLE_WITH_MINIMAL_DELTA`：明示的system gate列に追加controlを付け、既知2-unitary LCUでχ/βを成功blockへ格納 |
| 追加操作 | 元Hadamard aに加えblock b/herald h、PREP/逆PREP、制御RY、fine/coarse二重制御、bのRY測定基底、X/Y readout |
| 無料oracle | state逆準備/reflection、coherent sampler、S/p/真signal oracleは不要 |
| 測定/失敗 | success flagと既知β²・λで全attemptのzero-fill estimator。失敗prefixも有料 |
| 階層 | 既存K2のapplication metadataを最後のrotation labelのn=0 eventへ写す二level候補を記述。未採択、実N1で生成/評価なし |
| 改善条件 | `COST_ADVANTAGE_UNSUPPORTED`。今回のLCUでは `M2=β²(d²+ζ⁻²)>4`、小差分だけではnoiseが減らない |
| 人工correctness | 新規9件PASS。full success/failure block、X/Y/phase、全attempt mean、toy有限law、unsupported拒否 |
| 実科学 | H4数値計算・Born・RNG/shot・実compile/price・variance/N/総費用・保存raw再解析0 |

詳細は[A2_EXECUTABLE_ACCESS.md](A2_EXECUTABLE_ACCESS.md)、[A2_MEASUREMENT_AND_COST.md](A2_MEASUREMENT_AND_COST.md)、[A2_IMPLEMENTATION_AND_ARTIFICIAL_TESTS.md](A2_IMPLEMENTATION_AND_ARTIFICIAL_TESTS.md)。

## 前回から進んだこと

前回の一般的なdilation存在・UNKNOWNを繰り返さず、既知LCUの実際のcontrol順、二つの追加ancilla、成功/失敗block、測定基底、bit decoderを作成した。前回STOPや `ACCESS_MODEL_INCOMPLETE` の履歴は保全し、今回の技術到達点として区別する。

SCVはmetadata補正、通常MLMCはpath couplingを使う。今回A2は量子状態と測定observableを変えるため、アクセスとしては異なる。ただし実行可能になったことは、測定noiseを減らせたことを意味しない。論文のscaled augmented boundを、成功率・LCU scaleを省略して移送しなかった。

## 再利用と未資格事項

既存RTEのevent/application/phase、DFのcontrolled/uncontrolled gateアクセス、Hadamard bit/X/Y規約、既存matrix比較helperを再利用した。独立のsampler・solver・decoder・管理frameworkは作っていない。静読SOURCEと固定資料のblobは[SOURCE.json](SOURCE.json)、技術保全確認は[TECHNICAL_CHECKS.json](TECHNICAL_CHECKS.json)。

旧engine認証は追加double-control/RY/scale/joint-gridを覆わず、新adapterの総数値誤差はUNKNOWN。実H4 selected fullへのbindingとnative価格も未資格。これらを0補完して成功扱いにしない。人工のparameter値はtest fixtureで、科学条件の採択ではない。

## 次の判断

このLCUを実H4で試しても、既に式で確認できたnoise負担を再確認することになり、初手の新科学runを勧めない。実行可能な回路部品は保存するが、**改善の実行候補は残さないためNEXT_MINIMAL_SCIENCE_DRAFT.mdを作成しない**。

再検討には、未知state情報なしに、全attemptのrange/momentと全block費を改善する具体的な追加性質が必要。新しい非自明な理論の採択・重要科学レビューは今回開始していない。将来その仕様が成立する場合だけ、独立SOURCE・対象・有限call・誤差・費用・STOPを定め、別途の科学実行許可を受ける。J1/R1の同じ静的探索やO1/O3/C-Sへの自動移行はしない。

## 保全・公開・STOP

適用AGENTS/PROJECT_MAP、前回から変更のない正式共通指示2026-10-25と概要、および固定6成果物を確認。変更は新module/test/資料と必要な索引・概要・当日ノート、manifestの独立技術項目のみ。旧科学的statusと全SOURCE/raw/grant/ledger/marker/STOP、C1凍結、元main/dirty/untrackedを維持する。共有環境・他ユーザー・他process/job/cacheは操作しない。人工処理は小規模で、共有venvを変更せず恣意的資源上限も設けない。

専用branchで公開し、remote SHA、全変更本文/blob/treeと固定参照を照合する。公開後の固定SHAは最終チャットと作業用照合記録を正本とする。公開照合後 **mandatory STOP**。新しいscience・条件採択・詳細科学レビュー・別研究へ進めない。
