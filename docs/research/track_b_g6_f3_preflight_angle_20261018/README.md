# Track B G6/G7 F3：事前有限bit保証の実装・限定人工検証

2026-10-18 JST。**F3_TECHNICALLY_VERIFIED**。対象を非identityの`full_return`に限定し、F1のP0〜P5を提供済みH/Kの正確な受入述語として接続した。全到達valid parentの根拠は固定F1の一様lemmaであり、人工例を全parentの証明へ置き換えていない。全parent列挙・角辞書・自動selectorは使用しない。

入口、x=0 identity、実lawとSOURCEのbinding、理由別terminal failure、固定精度のangle enclosure、旧event意味論の保持を実装した。**最終SOURCEに対する人工テスト63件PASS、FAIL/ERROR/SKIP各0**。一般定理、実分子、physical provider、native費用、資源優位、新規性の認定ではない。

| 資料 | 内容 |
|---|---|
| [実装契約](implementation_contract.md) | API・exact入力・event・failure・費用・出力境界、A01〜A10対応 |
| [preflight証拠](preflight_certificate.md) | P0〜P5/R、全parent十分条件、evidence・identity |
| [angle実装](angle_enclosure_implementation.md) | half-angle/交代級数、固定K_phi/J、τ、signed time |
| [人工検証報告](focused_test_report.md) | T1〜T6、全63件、4回の限定実行と不足 |
| [ユーザー/GPTへの引継ぎ](handoff.md) | 完成範囲・次の担当・承認・mandatory STOP |
| [固定資料索引](evidence_index.json) / [SOURCE索引](source_inventory.json) | 旧commit/path/SHA256、読了・実行scope |
| [実装索引](implementation_inventory.json) / [管理検査](verification.json) | 新SOURCE/testのhash・行位置・静的/保全確認 |
| [公開receipt](publication_receipt.json) | payload公開bytes/SHA256とreceipt子commitの検証方法 |

SOURCEは[contract](../../../src/trottertracks/algorithm_codesign/g6_f3_contract.py)、[precision/angle](../../../src/trottertracks/algorithm_codesign/g6_f3_precision.py)、[generator wrapper](../../../src/trottertracks/algorithm_codesign/g6_f3_generator.py)。testは[F3専用ファイル](../../../tests/tracks/algorithm_codesign/test_g6_f3_preflight_angle.py)だけ。既存G6/G7 SOURCE・tests・raw・契約・grant・ledger・marker・claimは変更しない。既存索引も上書きせず、この入口で新旧を対応させた。

基点`dd03eb08b6d768f5578d743ab5ad390bd9c93b07`、branch`track-b-g6-f3-preflight-angle-implementation-20261018`。独立した専用cloneを作業領域に使用し、親/他Codexのworktree・indexを共有しない。新SOURCE/testだけを追加する。正式科学契約・新execution authorizationは作成しない。

新規科学計算、保存科学値再解析、分子/DF、backend、回路build/compile/synthesis、科学レビュー、新証明、H/K selector、新sampling law、DF/PR/RTE adapter、全trial estimator、P_U/W2b/W2cは全て0件。既存のpure rational処理は、今回明示許可された人工入力と新wrapper内でのみ呼んだ。

固定仕様：[F1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/preflight_precision_contract.md)、[受入A01〜A10](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/implementation_acceptance_requirements.md)、[F2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md)、[既存angle監査§6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md)。ここでの技術PASSは科学的GOではない。**公開・保全確認後mandatory STOP。次の作業を自動開始しない。**
