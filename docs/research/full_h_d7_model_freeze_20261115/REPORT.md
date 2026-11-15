# Full-H κ2 D7：10較正record受入・2モデルfit・4予測の観測前凍結

2026-11-15 JST。**全5 cell・10軸別較正record、9 path・18軸の保存価格/receipt/実装証拠を受入。保留0。X/Y別のrank3・3係数モデルを各1件、5 cell等重みのexact有理数最小二乗でfitし、m5/m6×X/Yの4予測を固定した。holdout価格開封0、新sampling/compile0。**

正本：[run01/FROZEN_MODELS.json](run01/FROZEN_MODELS.json)の`models.X/Y`と`predictions`4行。[PREDICTIONS.csv](run01/PREDICTIONS.csv)は正本の転記。[INPUTS.json](INPUTS.json)は受入/元recordと18path-axisの固定locator、[PREOPEN_CHECK.json](PREOPEN_CHECK.json)は実行前、[PUBLICATION_PRECHECK.json](PUBLICATION_PRECHECK.json)は公開前metadata確認。

## 固定SOURCE・重複確認

科学実行SOURCEは**`f46fa120b4cddcaac3edcf21e3731a0d9020b2b8`**、`scripts/freeze_full_h_d7_models.py`。固定SOURCE/INPUTSをcommit後、fresh run01で2fit/4predictを一回だけ実行した。

- D1最終引渡し：`e628c0d69c7da559369d99f323e3656a96c0728a:docs/research/d_full_h_p2_calibration_resume_20261115/HANDOFF_C.json`。
- 先行4 cell：`7d0c39066d4ed900cfe7f5de6f4c7c5a3b2ec41f`。完成G36223：`f98cb96384b775bec2c2b9e1e63fe01a6ccdf2b1`。G36223最終検証SOURCE：`8cce540645b99e515d72888b817782a1b9f97334`。
- D7 cell/holdout設計入口：`a43050c786c741d267ddf322232d888780efc03c`。実採用metadataはD1引渡しが指す`7d0c3906…:…/FROZEN_JOBS.json#plan`。A設計/特徴量は`7a0c9468b2499fac9fa1cd279ae2980c9dab048e:src/trottertracks/finite_price_design.py`。
- exact fit/predict：`c628e9e8447c11c984064b922cbce6c52dc72d48:src/trottertracks/price_model_validation.py`を既存A `fit_identified → c_model`経由で無変更再利用。依存`rpe_mean_price_model.py`も固定Cと一致。

共通指示§10.5に従い、開始時のC HEAD/公開ref `9edfa5d3…`・clean、関連公開refとC結果path名、関連Python job名、既知D1 private root/resume01のnamespace名を確認。同じD7入力/式/軸の受入済みfit・予測・進行job、M5/M6取得を確認範囲で認めず、metadataを実行直前/公開前に保存した。他担当/環境に変更なし。未知archive/別hostの全記録を網羅した主張ではない。

## Cの受入範囲

保存H4/linear1Å/STO-3G/tol-only DF10、元state/H/snapshot、Full-H pure RTE（one-bodyを含む非scalar全Hをrandomize、deterministic headなし）、finite K2、κ2の固定D7 domain。元Full-H preparation、fragment/component順、normal sampling law、full controlled measured Hadamard wrapper/P0、新ZZ、元scalarとその総T一回適用、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17・symmetric-directional controlを維持。

各recordのroute/law/implementation/profile/quantity/kind、FULL_H_GROUP_V1特徴量、cohort/step_seeds、n_pathsをplanへ照合。path_axis_bindingsからPRICES、IMPLEMENTATION_EVIDENCE、元compiler JSON bytes/hash・fingerprint/metrics/compiler、scalar/event/XY parity、raw receipt manifest/decoderを対応付けた。専用非configuration recordを使い、較正cellへ架空round IDは付けず、canonical-round専用guardは呼んでいない。

D1の公開後raw hash/再生成一致検査は上流証拠として再利用した。Cは巨大gzip rawを再復号して回路/状態/価格を再生成せず、公開された個別price・実装証拠・compiler recordの一致を受入確認した。これはCによる独立な全raw再実行・作用検証ではない。全pathの保存phase/parity/XY/scalarと、**各cell index0の7数値gate、index1はその数値gateなし**という元の範囲を保持する。

G36223の現受入は完成4軸のみ。p0のproducerは`d323fee1ca8cd6893c4b705f990cdc931bdb5c87`、p1は`887c3b9d04d15be96637b6634265424e7dd50339`、最終validatorは`8cce5406…`。旧partialや再開時再利用4軸を別価格として加算せず、18軸の内数として保存した。

P2R566はG566 path0/group0→path1/group0の親event直接連結。元親selector/source/seed/orderと`independent_third_sample=false`を保持した。D7所定の1較正行として使用するが、親2観測と独立な第3親標本とは数えない。

## 入力とfit

| cell | q | R | path数 | 保存X平均RZ | 保存Y平均RZ |
|---|---:|---:|---:|---:|---:|
| G566 | 1 | 566 | 2 | 85080 | 85080 |
| G4528 | 1 | 4528 | 2 | 693571 | 693571 |
| G36223 | 1 | 36223 | 2 | 11125307/2 | 11125307/2 |
| Q8R566 | 8 | 566 | 2 | 692816 | 692816 |
| P2R566 | 2 | 566 | 1（paired） | 169888 | 169888 |

保存PRICE_CALIBRATION_RECORD_V1の軸別meanをそのまま用い、Cはcohort平均を再集計していない。`s=qR`、`C_hat=a+b*s+c*(q−1)`、features順は`one,event_count,outer_boundary_count`。各軸5行・3列、rank3、`EXACT_NORMAL_EQUATION_LEAST_SQUARES`。5 cellをそれぞれ1行とし、path数による追加重み付け・X/Yのpoolはない。P2の依存関係を独立観測の仮定へ置き換えていない。

| axis | a | b | c |
|---|---|---|---|
| X | −52444240256483/20115111656 | 3090493494163/20115111656 | −653520358735/10057555828 |
| Y | −52444240256483/20115111656 | 3090493494163/20115111656 | −653520358735/10057555828 |

2モデルを別々に保持。`calibration_residuals`は既存fitが返す**各cellの絶対残差**で、入力順はmodel.calibration/input_locatorsのG566,G4528,Q8R566,P2R566,G36223。exact分子・分母、元SOURCE/cohort/step_seedsを保存した。係数の符号制約・clip・再fitは追加していない。較正残差を独立holdoutの成功判定としない。

## 観測前の4予測

| cell / round | axis | q | R | s | 元physical ID | exact予測RZ | 表示用RZ |
|---|---|---:|---:|---:|---|---|---:|
| M5_ROUND / m5 | X | 32 | 283 | 9056 | FULL_H_bc8aab34256b73b68025 | 27894546580642075/20115111656 | 1386745.798764761 |
| M5_ROUND / m5 | Y | 32 | 283 | 9056 | FULL_H_bc8aab34256b73b68025 | 27894546580642075/20115111656 | 1386745.798764761 |
| M6_ROUND / m6 | X | 64 | 566 | 36224 | FULL_H_4ec8d73571760a6830b8 | 111815248527103419/20115111656 | 5558768.47413625 |
| M6_ROUND / m6 | Y | 64 | 566 | 36224 | FULL_H_4ec8d73571760a6830b8 | 111815248527103419/20115111656 | 5558768.47413625 |

比較用はexact比。表示丸め値を再利用しない。M5は各軸2path、M6は各軸1path、後続compileは合計6で今回0。cohort/step_seedsは固定planからmetadataとして保持。holdoutの価格/receiptは読んでいない。

各モデルは`PRICE_LINEAR_FEATURE_MODEL_V1`、`CALIBRATED_UNVALIDATED`、`qualified_for_candidate=false`。元design_sourceにはA提案時の`PROPOSAL_NOT_ADOPTED`という履歴が残るが、利用者の本指示とD1固定plan.adopted=trueを適用し、D1 gateが参照する元design_sourceを改名していない。母law E[C]、係数/有限cohort/共有依存CI、sampler/ideal距離、rare、長時間移送は未知のまま。許容差・モデルPASS/FAIL、N/G/winnerは追加していない。

## 技術確認と終了点

人工新5＋既存exact model回帰3＝8 test method PASS。synthetic pricesだけで2軸の過決定fit、path重みなし、保留軸の非起動、元D1 frozen_model_gateとseed/physical/validateの読取、holdout価格reader拒否を確認した。metadata準備では抽出関数の依存定数P1.COHORT不足を補い、実際の入口を通過した。科学fitはその修正後に固定SOURCEで一回のみ、科学retryなし。

実数値段階wallは約0.165秒（metadata受入/テスト/公開は含まない）。小さいexact系で、既存保存reader cacheとC/A helperを再利用し、重いprocess/circuit pipelineは起動していない。作者の技術確認であり独立科学reviewではない。

ロードマップ差分：Full-H D7の「全較正引渡し→C受入/fit/予測待ち」を解消し、D1の独立m5/m6取得（別指示の6compile）待ちへ進む。モデルの予測性能・母law資格・CI・長round・N/Gは未解消。ロードマップ本体は統合担当が更新する。

**mandatory STOP：固定モデルと4予測の公開、D1 metadata readerでの固定JSON読取、remote照合、HANDOFFで終了。holdout取得/開封/判定、許容差新設、他round、E[C]/CI/N/G/winnerへ進まない。**
