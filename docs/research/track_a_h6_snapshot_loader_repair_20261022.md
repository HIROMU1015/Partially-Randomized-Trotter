# H6 snapshot loader修正・grant前検査

2026-10-22 JST。**loader修正と人工26件、保存H6の8配列byte照合がPASS。科学再実行は行っていない。**

修正SOURCE `f942a5f66482db797902d4c88c262a70da6a8492`、branch `pf-common-h6-metadata-execution-20261022`。基点 `2f786bf4a998b8c7b47ecc743fe0275fb82edb1b`。旧METADATA stageの失敗SOURCE・grant・ledger・marker・STOPは保全し、H6共通PF比較は引き続き未完了。

## 修正内容

前回の0次元scalar shape修正をloader本体の読込経路へまとめ、原H6 receiptはNPYのshape（scalarは`[]`）、移行H2/H4 receiptは従来のcontiguous配列shapeという定義を維持した。dtypeとpayload SHAの検査を緩和せず、欠落・余分な配列も拒否する。

新しいstdlib専用 `scripts/pf_common_snapshot_receipt.py` は、NumPy変換を挟まず元NPY headerのshape/dtypeとpayload SHA、archive SHA・固定binding SHAを検査する。H6 packの新規root作成前と、`Stage`によるgrant/marker/ledger作成前の両方でこの検査を行う。誤ったshape receiptを与える人工試験で、METADATA root自体が作成されないことを確認した。読込時のSHA・dtype不一致を科学call消費後まで持ち越さない。

原H6 loaderと同じ16 MiB入力上限を保持。重複ZIP member、配列名相違、原C-order相違、header/payloadサイズ相違も拒否する。PF係数・候補・科学的観測定義・call予算・compiler政策は変更していない。

## 検証結果

| 検査 | 結果 |
|---|---|
| 新NPY/loader/grant前拒否の人工test method | 11 PASS |
| 既存metadata再帰・closure・one-shot・cache | 5 PASS |
| 既存common native・整数会計・one-shot | 10 PASS |
| 保存H6のheader/dtype/SHA | 8配列 PASS |
| 消費済みprivate packと旧公開結果の保全 | SHA/byte一致 PASS |

人工入力は数値scalar、Unicode scalar、complex vectorを含む。loaderの配列読込関数を通し、H6と移行H2/H4の双方のshape規約を確認した。shape/dtype/SHA改変、配列欠落、重複ZIP entry、Fortran-orderを拒否し、stdlib checkerの別processでNumPyがimportされないことも検証した。

保存H6の原snapshot SHAは `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`。検査はNPY headerとbyte hashの読み取りだけで、NumPyによる実H6配列復元、Hamiltonian/state構築、oracle、preparation、signal作用を行っていない。したがって実H6数値loader全体・63件取得が成功したとは主張しない。

新しい分子pack/grant・snapshot_restore・metadata・compile・PR samplingは全て0。以前のsnapshot_restore1回の消費とSTOPは不変。旧H2/H4結果と旧H6 STOPを読み取り参照しただけで、再計算・再ランキング・旧receipt書換えなし。

## 保存

[検証結果](../../artifacts/resource_applicability/track_a_h6_snapshot_loader_repair/2026-10-22/validation.json)、[原H6 byte照合](../../artifacts/resource_applicability/track_a_h6_snapshot_loader_repair/2026-10-22/original_H6_byte_preflight.json)、[旧結果・grant保全](../../artifacts/resource_applicability/track_a_h6_snapshot_loader_repair/2026-10-22/preservation_checks.json)、[修正SOURCE一覧](../../artifacts/resource_applicability/track_a_h6_snapshot_loader_repair/2026-10-22/source_inventory.json)。人工ログはbundleへ保存した。

実装：`scripts/pf_common_snapshot_receipt.py`、`scripts/pf_common_run_native.py`、`scripts/pf_common_run_contract.py`、`scripts/h6_pf_metadata_completion.py`。人工：`tests/test_pf_common_snapshot_receipt.py`。

技術修正と検証をローカルcommitし終了。消費済みgrantのretry/resume、新しい分子stage起動、GitHub pushは行っていない。
