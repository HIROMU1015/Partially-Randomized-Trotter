# 公開A0 normalization実結果24件の受入・C1接続（2026-11-11）

公開済みA0実結果24件を、既存B `529e81fc20969bd8265ae98ba71ad18fa8f3eca7`の`A0Intake`とCLIを**変更せず**取り込んだ。SOURCE9件・入力hash7件・全candidate/identity/law・HANDOFF/resultが一致し、`normalization_evidence_usable=true`、全24件の診断共有資格が有効となった。C1固定8セルとの接続は指定H6の2件だけで成立した。**新しい科学計算・保存値の数値再解析・N/C/G評価・R最適化は0**。既存47技術testと公開fixture検査がPASSした。

## 入力と受入状態

入力は[Aの公開commit `4745bb676568bb11d46552be42ffc6f4fba75385`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4745bb676568bb11d46552be42ffc6f4fba75385/docs/research/pr_r_normalization_phase0_results_20261111/REPORT.md)に固定した。

- HANDOFF：`docs/research/pr_r_normalization_phase0_results_20261111/run01/HANDOFF.json`
- result：`docs/research/pr_r_normalization_phase0_results_20261111/run01/result.json`
- A科学実行SOURCE：`6a75ebf967aaae2698ce6d295bf6fbec9d5e9929`
- B受入仕様：[既存REPORT](../rpe_normalization_intake_20261111/REPORT.md)・[既存HANDOFF](../rpe_normalization_intake_20261111/HANDOFF.md)。今回adapter、R可変N/C pipeline、科学kernelとテストSOURCEは変更していない。

既存CLIの公開SHA/path引数を使い、fresh directoryに[INTAKE.json](INTAKE.json)を出力した。受入結果のstatusは`A0_DIAGNOSTICS_ONLY`、completionは`COMPLETE_RECORDED`、元root statusは`PHASE0_NORMALIZATION_RECORDED`。元`normalization_evidence_valid=true`、SOURCE/input保全ともtrue、`evidence_blocks=[]`。入力hashの照合対象は公開resultと固定PLANであり、Aのlive入力を開いた検査ではない。

| 診断recordの状態 | 件数 |
|---|---:|
| 正常に記録済み | 24 |
| 欠測 | 0 |
| STOP record / root STOP | 0 / false |
| 数値未資格 | 0 |
| overflow / underflow表示 | 0 / 0 |
| 診断消費・共有資格が有効 | 24 |
| C1との共有対象 | 2 |

件数は保存statusとrecord存在の整理であり、新しい数値指標ではない。全表示は元の`FINITE_BINARY64`、normalization statusは`NORMALIZATION_SCORED_NOT_N_QUALIFIED`。値の表示品質を科学的PASSにしない。

## 条件照合と値の保全

既存adapterで24 ID、system/m/T/qRPE/qPF/δ/R_outer、formula/prefix/K、signed weights・piece allocation/event・scalar規則、H/state/snapshot、completion/fragment順、RPE configuration、compiler/control/full measurement wrapper/P=0を照合した。Linear H4/H6・1Å/STO-3G・保存tol-only DF10/DF19、H4 S2/δ=.1/prefix1/K2、H6 4th(m5_best)/δ=1.6/prefix7/K2、元のstate/sectorとh=.1/M12等を保持する。

HANDOFF/resultのSOURCE/status/records、固定科学SOURCE hash9件、resultの固定input hash7件と保全booleanが一致した。Aの元law宣言を維持し、B N provider用lawとの差を自動aliasしていない。normalization methodと保存tail SOURCEも既存仕様に従い一致した。SOURCE照合は科学的来歴の確認であり、利用者許可の証明機構ではない。

log B、2log B、attenuation、normalization由来shot係数、丸め感度、小τ主項/max event τ、全allocation/event、精度status、SOURCE/request/lawは元recordとJSON同値のまま保存した。exp/log、誤差・近似、丸め感度を再計算していない。異常時のnull/overflow/underflow保持は既存人工テストで回帰確認し、今回の正常値へ置換していない。1 ULP感度は厳密誤差区間やCIではない。

## C1接続

既存Bの候補整合commit `328cb26fa893e49197d15e0e075e3d9b43213980`とC計画commit `8457c443554af5fadaa3ba36f97f6316ca3c5050`を使用した。C1はH4 qPF8/16・R4/8、H6 qPF1/2・R16/32の元8セルを維持する。

| Candidate ID | 条件 | C1 role | 診断共有 |
|---|---|---|---|
| `2d2af6ca830dff14a695` | H6 m4/T1.6/qPF1/δ1.6/R16/p7/K2 | 保存校正 | true |
| `4269eafc8c0ef8922c23` | H6 m4/T1.6/qPF1/δ1.6/R32/p7/K2 | 新校正 | true |

固定C1の予定metadataから既存B関数でrequestを再構成し、Cの保存`describe_request`によるidentity/axis join_keyが既存照合表と一致することを確認した。そのrequestと元law、A科学SOURCEを`A0Intake.for_c1`へ渡し、上記2件の`share_eligible=true`と元診断の同値性を確認した。[C1_LINKS.json](C1_LINKS.json)にsource/full request/診断recordを保存した。これは予定候補との技術接続であり、C1実科学実行の証拠を取り込んだものではない。

他6セルは同APIがA0範囲外として拒否し、診断値を補完しなかった。H4 R8/q8とR8/q16は`EXPLICIT_C_CALIBRATION_VALIDATION_ONLY`のまま保全した。C1のsampling/compile/fit/価格評価を呼ばず、C候補やA0の範囲を変更していない。

## N/C coverageと残る未資格

[COVERAGE.json](COVERAGE.json)はA元PR menu全86件とC1 8セルを別laneで保持し、A0診断24件/範囲外62件を区別する。元menuを削除せず、追加のR上限・採択・順位も付けない。固定N/C在庫はB `ac3e1ce87b5ed7d9114abc316545366dd097209e`の既存coverageだけを参照し、保存数値を再集計していない。このpin以降のC科学実行状況は主張しない。

- **N**：A0の全24件は`phase_radius_N_status=NOT_EVALUATED`、N/C/G=null。別成果として同一candidateの固定R Nがあるのは6件で、新R18件には同一候補Nがない。既存6件の条件付きround PASSは出典/statusだけを保持し、R-aware allowanceの共通比較資格を新たに付けない。保存H4巨大Nや旧N原本は開いて計算せず、変換・上書きもしない。
- **C**：A0内で保存direct/Affineの有限scenarioがあるのはH6 m4/R16だけ。残る23候補に適格な保存Cを補完しない。固定R baselineの長qは未検証、新Rは固定Rモデルのdomain外という既存区別を維持する。C1全8セルでは保存校正4/新校正2/独立検証2の元roleを保全し、ここから新価格を生成しない。
- **期待値と不確かさ**：有限cohort価格は全law期待価格ではない。長q、大R/未観測allocation、希少event、actual samplerとの距離、共有校正誤差・cross-R共分散、long-q systematic、total CI、全RPE成功は未資格。未知共分散/CI等はnullのまま。

`N_records=[]`、`C_records=[]`、N/C/G・推奨R・科学的PASS/FAILはnull、`scientific_R_adopted=false`、`full_RPE_qualified=false`を保持した。normalizationを十分Nへ変換せず、shot係数をCに再乗算しない。小qのC1が長時間m12の価格資格を与えるとは仮定しない。

## 技術検証・完了範囲

[TESTS.txt](TESTS.txt)：既存の受入8＋候補整合7＋R選択12＋provider統合8＋PF/PR pipeline12＝**47件を再実行して全PASS**。テストSOURCEの追加・変更はない。[VERIFY_INTAKE.py](VERIFY_INTAKE.py)は今回の公開fixtureに対する診断無変更・最終API/C1接続・86候補保全・数値entry非起動の技術検査とcoverage書出しだけを行う。通常実行で保存した派生JSONとも照合できる。numpy/scipy/qiskitとnormalization producerはimportしない。

原本、旧SOURCE/input/raw/結果/grant/ledger/marker/KEEP/mandatory STOPを保全し、A/Cのbranch/worktree/process/cacheに触れていない。独立科学レビューや新しい科学計算は実施していない。

次は、**ユーザー**が必要な有限signal/N・R依存C/長q/全law/共分散の科学scopeを判断し、**Codex A/C**が許可範囲の成果を公開する。**Codex B**は指定された公開SHA・条件付きN/C recordを既存pipelineへ取り込む。A0完了を新N/C/G実行の許可にはしない。[HANDOFF](HANDOFF.md)に今後の受入条件を記した。commit/push・remote SHA/tree/主要成果物照合後mandatory STOPする。
