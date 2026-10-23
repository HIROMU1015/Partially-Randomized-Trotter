# H4 C reference縮約修正・一般状態の人工回帰

2026-10-23 JST。利用者の「修正を行って」に基づき、referenceの右側state作用欠落を修正した。修正専用branchは `codex/non-df-h4-c-reference-fix-20261023`、基点は失敗runの保存commit `49db6f628e5ae51b5f5afa014e29d771e6d0d1eb`。修正実装SOURCEは **`e80d7510880f8db993b777c627bb8e95b05a070d`**。成果物・sealを収録する後続commitと区別する。

**修正・人工検証・新技術SOURCE固定まで完了。修正版による実H4計算、新科学grant、有効化、N/G会計、pushは行っていない。** 以下の検査は主担当Codex自身の技術検査であり、別担当の独立科学レビューではない。

## 修正内容と原因

`src/trotterlib/h4_ball_evidence.py` のreferenceを `signal(state, exp(-iTH) * state)` に修正し、必要な `ψ† exp(-iTH) ψ` を評価する。旧実装は `(ψ† exp(-iTH))[0,0]` を返していた。signal helperにも、同じ行数を持つ非空の列vector2本だけを受け入れるshape検査を追加した。未作用の正方行列・row vector・複数列・行数不一致を拒否する。

科学法則、候補、閾値、normalization式、call予算、precision進行、実行権限、native監視の実装は変更していない。旧SOURCE `49c18ccc35f14041df45bfeff455ba5e67543182` が実際に参照するworktreeは編集せず、別worktreeに修正版を作成した。

## 人工検証

| 検査 | 結果 | 範囲 |
|---|---|---|
| 既存110件＋追加14件の明示suite | 124 PASS、FAIL/ERROR/skip 0 | 人工・mockのみ。B保存40軸の再解析なし |
| 新規14件を旧SOURCEに適用 | 14 test、subtestを含む30 assertion FAIL、ERROR/skip 0 | 欠陥を検出するための期待どおりの失敗。旧コード・旧ログは未変更 |
| 同じ14件を新SOURCEに適用 | 14 PASS | 上の124件に含むため件数を加算しない |
| 既存FLINT backend回帰 | 13 PASS | 人工1×1/2×2行列のみ |
| frozen旧runtime→新productionの接続 | 4人工model PASS | 非真空・複素状態それぞれの2 qubit Pauli/DF、計80人工軸 |
| 無効契約CLI | 5 PASS | inspectはdispatchなし。grant/execute/worker/bridgeは権限gateで拒否 |

新しいreference回帰は、全4基底状態を64/96/128 bitでproduction `tier` に通し、別の256 bitスカラー解析式と保存上下端を照合する。複素重ね合わせ、global phase不変性、非可換なX/Z人工Hamiltonian、非単位normのencoded state保持、PF/PR両biasへのreference反映を含む。intervalの幅も検査し、広い包含やmidpointの一致だけでは通過させない。非真空状態の正常referenceには追加のdense指数callを必要としないことも確認した。

前回の110 PASSは初期basis stateだけでは欠陥を検出できなかったという履歴であり、書き換えていない。今回の追加検査と修正版のPASSが実H4の精度達成やcertificate受理を証明するわけではない。

## 新SOURCE・runtime・無効契約

[SOURCE](../artifacts/non_df_h4_c_reference_fix/2026-10-23/SOURCE.json)はcode19、private runtime1,050、旧科学source738、旧runtime6,999、入力16ファイルをSHAで結合する。private環境は本worktreeの `.c-reference-fix-venv`、Python3.12.3、python-flint0.9.0/FLINT3.6.0、NumPy1.26.4。共有環境・旧runtimeを変更していない。

[C契約](../artifacts/non_df_h4_c_reference_fix/2026-10-23/C_execution_contract.json)は `execution_authorized=false`、`grant_valid=false`、`dispatch_enabled=false`、grant/approval未設定、未消費のまま。科学入力は元SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`、BのSOURCE/結果は `2bd212288e8530e519c4a8135257df37f7608662` / `b0f94185f93cd5614e8f183613a2a9313a5b9d4f` を保持する。新namespaceは将来用の予約文字列で、directoryを作成していない。

将来の対象は従来のlinear H4/1.00 Å/STO-3G・8 qubit・保存共通状態Nα=Nβ=2、Pauli184項/PR prefix27、DF tol-only1e-8/rank10/PR prefix1＋one-body。S2、T=.8/q8/r4/K4、4方式×ε=.05/.01/.005/.001/.0005×cosine/sineの40軸、設計比較目標Bの必要u上限/2を保持する。precisionは64/96/128、call最大予算もop2/prob2/reference6/nominal12/implemented PR6/normalization6/dense39のまま。契約差分は実装SOURCE・seal・将来namespace・状態説明・旧SOURCE識別だけで、科学条件の差分はない。

## 保全と残る制約

[保全監査](../artifacts/non_df_h4_c_reference_fix/2026-10-23/preservation_after.json)は既知の2,166ファイルを照合した。元科学source/raw/入力/grant/ledger/marker/STOP、B結果・消費済みgrant、原因調査、前技術package、失敗C runの105-file inventoryとそのmanifestを保全した。検証対象は元の絶対pathであり、履歴inventoryを修正後worktreeの新しいindexやコードへ読み替えていない。

失敗C run01のgrant `8af94df114e6485c7a5f10d12e77352347aa21b2e9525647a8ad8ebc988f4554` は消費済み、STOPは `FAILED_NO_RETRY` のまま。Pauli20軸rawの不受理・DF最終20軸未取得・科学受理0軸も保持する。旧SOURCEの `EXECUTION_HOLD.json` は解除していない。

旧20精度行のN/G=null・40軸UNDETERMINED、元H4のnative終了未証明・監視消失原因UNKNOWNは独立した未解決事項である。DF event範囲、literal compiled wrapperへの転送、IID uniform-word仮定/PCG64独立性、正式certificate受理に関する制約も残る。この修正や人工テストで経験的証拠をCERTIFIEDへ格上げしていない。

## 次の作業と停止

**ユーザー**：新SOURCEと無効契約を確認し、固定40軸の新しい実H4数値証拠候補生成を実行するか、その範囲を明示承認する。公開pushとN/G再会計は別の判断対象。

**Codex**：承認後にだけ、修正版SOURCEをfresh byte gateで検査し、新しいapproval/active契約/未使用namespace/新grantで一回実行する。消費済みrunのresume・grant再利用・旧結果上書きは行わない。

現段階はlocal commitのみ。**mandatory STOP。実H4再実行・新grant有効化・科学解析へ自動移行しない。**
