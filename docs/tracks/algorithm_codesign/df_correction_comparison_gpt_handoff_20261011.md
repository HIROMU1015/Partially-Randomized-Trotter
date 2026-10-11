# GPT/利用者へ：DF whole / local Pauli 比較契約の確認

2026-10-11 JST。今回の作業は**原本固定と事前契約のみ**。研究判断、比較・compile・Clifford+T合成は行っていない。mandatory STOP。

branch `track-b-df-correction-comparison-preregistration-20261011`、base `8a5e21c8f6ecf98ee285ec1fad49079ffb5ed97f`。
入力原本commit `f6a8f022947bb9efe1f2c4fffeb0eeda051b2903`。
事前契約の固定commitはこの文書を取得するGitHub commitとpublication receiptに指定する（自己参照SHAを埋めない）。

## 今回公開したもの

- [GPTレビューMD原本](../../research/track_b_df_sparse_correction_post_audit_scientific_review_20261011.md)、[ZIP原本](../../../artifacts/track_b_df_correction_comparison_preparation/2026-10-11/inputs/track_b_df_sparse_correction_post_audit_review_20261011_reproducibility.zip)。
  ZIP50,156 bytes、9files、SHA256 `bb14c641f9b7158cfdfd9efdc17d9c123f65b7ae1335c28c7847ba0ba4b951a4`。CRC／manifest／外部MD一致、GitHub再取得byte一致。
- [比較契約](df_correction_comparison_preregistration_20261011.md)、[native/T実行計画](df_correction_comparison_execution_plan_20261011.md)、[contract JSON](../../../artifacts/track_b_df_correction_comparison_preparation/2026-10-11/contract_v1.json)。
- [未決事項](../../../artifacts/track_b_df_correction_comparison_preparation/2026-10-11/open_items.json)、[準備scope](../../../artifacts/track_b_df_correction_comparison_preparation/2026-10-11/preparation_scope.json)、[source identity](../../../artifacts/track_b_df_correction_comparison_preparation/2026-10-11/reference_source_manifest.json)、[runtime確認](../../../artifacts/track_b_df_correction_comparison_preparation/2026-10-11/runtime_identity_preparation.json)。
- [取得性・provenance inventory](../../../artifacts/track_b_df_correction_comparison_preparation/2026-10-11/evidence_manifest.json)、[metadata整合確認](../../../artifacts/track_b_df_correction_comparison_preparation/2026-10-11/preparation_validation.json)。これは新しいmath/physics PASSではない。

## 指定条件と具体化した案

CDの係数0.45/0.35/0.25、seed6121107、t1、δ1/8,1/16,1/32、γ1.25、s1、cutoff4/P5、complex ε1e−4、α0.05を保持した。
初期状態を非vacuumの6basis superpositionとして提案した。
X/Y各0.025をfamily budgetと解釈し、A/B×3δの同時比較では各axis/taskへ1/240を割り当てる案にした。
この二点は利用者による契約確認対象。暗黙に実行していない。

共通axis最大bias：algorithmic2.5e−5、digital5e−6、native5e−6、numeric1e−6。
algorithmic項だけfixed certified式による未使用分をstat幅へ返し、その他はmaximum額を控除する規則。
最大δの保存A bound約1.8052e−5は過去のfloat記録への参照で、新しいcertificateやA/B比較ではない。
Bのweighted meanにはnon-contraction factorを戻す。新bound・QR/Pauli組立の認証は将来のgateで必要。

CD全PRの回路単価・prep/readout費用は未取得。記号変数と損益分岐だけを出す計画。
Eはseed7101107の元生成順序、r3/r6、u1/128、control有無の局所center費用に限定する。
**E単価×CD頻度をPR実測総費用としない。** CD native budgetが未検証なら精度保証もnative/digital実装条件付きのenvelopeとして報告する。

## native/T上限

native代表compile最大22、T target最大64。2worker・library thread1、既存runtimeをread-only使用。
pygridsynth2.0.0/mpmath1.3.0、dps100、up_to_phase=false、全target共通ε_gate=2.5e−11、seed/key固定、retry0。
T per-key wall30s/CPU20s、phase wall1200s/CPU900s、aggregate RSS1536MiB。
native wall600s/CPU450s/RSS2560MiB、全one-shot wall2400s/CPU1800s/RSS3072MiB。
wall capは今回の明示要求に応じた案。旧G10のRSS512MiB等の契約は不変。

64keysは全whole回路Tを保証しない。固定位置selectionで得るactual部分Tと、未取得の全Tを分け、coverage不足ならT_INCONCLUSIVE。
全RZの無制限合成、sample平均Tで全量を埋めること、結果後のprecision/target変更は認めない。
literal RZ anglesは将来のnative stageで固定規則から生成し、合成前にinventoryをexclusive freezeする。現在はselection仕様まで。

## 実行前に残ること

既存R1 angle型はπ/atan用でraw dyadic radiansに未対応。新しい小adapter、phase accounting、hard parent resource guardが必要。
比較runnerは未実装。source-bound semantic tests、input/Pauli/angle interval guard、control位相、64-key inventory completenessも未検証。
契約確認後に、必要なfuture source準備の範囲を利用者が決める。source S固定・別authorization-only child・明示one-shot指示までscience開始なし。

今回ZIP codeは未実行、synthetic NPZ未load、新行列/信号/trajectory/compile/T synthesis/LP/分子/GPU/旧run再開は全て0。
77＋15件の旧監査を再実行・増額していない。原本、旧API/source、結果・STOP・authorization・markerは保持。
science_execution_authorized=false、runs0、source_commit=null。新execution markerも作っていない。
資料取得性確認後にmandatory STOP。研究採択・新たな詳細科学レビュー開始を自動認可しない。
