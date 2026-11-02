# SOURCE接続監査

科学SOURCEの固定参照は `243eea8e840f3c472753f7935e510391626fd325`。新branch基点は `068f0ae8771fccf28c529bda12a84abe8b7e7648`。指定13 SOURCEはworking bytesと固定Git blobを照合し、[static_provenance.json](evidence/static_provenance.json)にSHA256を保存する。R1保存値はidentity/schema・既存記載の引用に限定した。

## 接続状態

| 層 | 分類 | 確認した接続・残件 |
|---|---|---|
| Sampler | VERIFIED_EXISTING | [G7] Ordinary/Partial/P3/Full、[P5] Closed P5。新sampler法は追加していない |
| 理想finite law | CONDITIONAL_EXISTING | 正有理p、Hermitian signed involution、0<x<=1、m=5、元のordered word/位相。全方式の理想meanは同じP5(-ixR)。Full/P5の理想accepted ensemble一致は[M§5] |
| Digital Full | VERIFIED_EXISTING / CONDITIONAL_EXISTING | F3 `prepare`は固定precisionでP0-P5をsource/lawへ結合。新人工fixtureでも利用。IID bits/物理provider/完走はcaller条件 |
| Digital canonical 4方式 | IMPLEMENTATION_REQUIRED / CERTIFICATE_REQUIRED | G7/P5は実proposal/補正weightを返しlocal guardを持つ。F3は`CANONICAL_PREFLIGHT_NOT_IMPLEMENTED`で拒否。新receiptはbit trace再生・source/state/実event検査まで。Full preflightを代用しない |
| DF operator | VERIFIED_EXISTING / CONDITIONAL_EXISTING | 固定C1 adapterの`append_q`、DF registry、actual reverse+inverseを再利用。人工5方式のlogical controlled接続を確認。R2のH4は固定6 eventの旧限定結果のみ |
| Whole-path | IMPLEMENTATION_REQUIRED → 人工最小接続済み | [whole_path.py](whole_path.py)：step順、weight積、zero、全trial分母、first failure、未完走meanなし。実H4 runner、永続transaction/measurement受入・全schedule bindingは未実装 |
| 統計 | CONDITIONAL_EXISTING | D1/D3の独立step/全trial・非unitary mean条件。新codeはIIDやconfidenceを認証せず、Nを計算しない |
| 誤差 | CONDITIONAL_EXISTING / CERTIFICATE_REQUIRED | Taylor・finite-bit・telescopingは既存保証。DF/R1 binding、PF定数、state/readout、uniform nativeの実入力certificateが未取得 |
| Native | CERTIFICATE_REQUIRED | strict controlled相対位相・actual adjoint・全supportのDF/rotation/scalar証拠が必要。6 event float一致はcertificateではない |
| 費用 | VERIFIED_EXISTING / CERTIFICATE_REQUIRED | D1全列挙とG8 observed-onlyを確認。logical/native/期待event/全trial/classicalを分離。認証価格はUNKNOWN |
| Evidence | 人工接続済み / IMPLEMENTATION_REQUIRED | exact Evidence digest、source/model/labels/precision/schedule/method/terminalを接続。productionの耐障害保存・実provider証明・RNG/測定来歴は未取得 |

必要SOURCE自体の欠落を確認した欄はない（`MISSING_SOURCE` 0）。研究上のUNKNOWNは未取得certificateと区別する。`PROOF_REQUIRED`は今回0件で、既存保証の通常適用と実装不足を新定理不足に読み替えていない。将来、新しい非列挙exact期待価格oracle等の保証が必要と判明した場合は別途判断する。

## 既存保証を接続する条件

有限bitでは実πW=α_midがexactに成立する一方、root近似meanと理想finite meanは異なる。[M§4]のroot bias、range、second momentは各方式の実丸め・positive support・係数relative誤差条件を満たす必要がある。ηによるproposal変更をweightで補正した後、同じ変更を追加mean biasとして二重加算しない。angle enclosureは実回転/native保証とは別である。

D1 `error_budget`はepsilon_T=3x^6/6!、finite meanのnorm<=1+epsilon_T、各stepの一様実装差eという条件で、Taylor積差 `(1+epsilon_T)^r-1`、実装積差 `(1+epsilon_T+e)^r-(1+epsilon_T)^r` を使う。meanをunitaryとしない。独立microstepではordered meanの積とweight二乗/最大値の積を既存D1/D3と同じ条件で扱える。不同stepは各認証boundを対応させる必要がある。deterministic halvesとscalarも実scheduleへ結合し、外側PF差は別に課金する。

D1 `validate_contract`は旧p、tau、r、rowsを固定する。旧runnerを編集してR1へ流用しない。`whole_budget`の旧alpha・誤差係数を無断移植しない。confidenceにはpositive残余幅、認証M2/range、全family/axis failure配分、fresh IID・測定条件、全N完走が必要であり、今回どれも新Nへ評価していない。

## targetと比較方式

microstep P5 → ordered finite meanの積 → outer S2を含む近似signal → 保存DF指数task → digital/native/statistical estimatorを別層として扱う。R1と保存DFのbindingが未閉鎖である以上、前二層の接続を最終指数taskの同精度比較と呼ばない。[旧C1-4]

| 方式 | normalizer / ideal law / digital接続 | 実入力までの残件 |
|---|---|---|
| Ordinary | iid raw偶数語+child、ordinary group root和B、dyadic group/label・weight | canonical preflight、実schedule/provider/誤差 |
| Partial Return | root+distinct degree2+ordinary degree4 tail、group root和B | 同上。distinctとraw iid tailの合法性を分離 |
| Closed P3 + tail | collected root、fixed-first degree2、ordinary degree4 | 同上。degree4 tailをFullのchild除外へ誤変換しない |
| General Full | local Green parent、ordinary envelope B、U、raw/coin zero、全normalizer不要 | 実H4全path・uniform native・保存終端 |
| Closed P5 | root/ordered-pair/first-label群、completion、群normalizer | group/conditional finite-bit認証、実DF/provider接続 |

Full/P5の同理想eventには同provider単価を認める。digital proposal/係数近似/moment/range/認証N/selector/setup/保存は一致を仮定しない。通常最適化された全対照にも合法cache・selector・basis reuse・completion DP・共通相殺機会を認める。

[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_generator.py
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g9_p5.py
[M§5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L336
[M§4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L212
[旧C1-4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/C1_4_ERROR_CONFIDENCE.md

追加固定根拠：[F3入口とevent guard](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g6_f3_generator.py#L305)、[D1 accounting](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/accounting.py#L36)、[方法ノートD1-D3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L670)、[固定DF adapter](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f3a6aa1108bbc8359e585c470e23033458c57466/docs/research/track_b_c1_2_c1_4_integrated_adapter_validation_20261021/adapter.py)、[旧R2限定correctness](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/CORRECTNESS_REPORT.md)。
