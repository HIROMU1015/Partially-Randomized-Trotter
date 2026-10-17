# 検査方法と限界

[契約メタデータ検査script](check_contract_metadata.py) は、固定SOURCEの`preflight`からauth／pinsを参照しない文だけをASTで選び、同じschema・正値・Rat形式・trial・inventory・STOP条件の検査へ適用した。選択行と除外行は [検査記録](static_inspection.json) に保存する。実行許可を仮造してfull preflightを通したものではなく、auth／pinsの結合とguard順序は静的に確認した。

追加の検査はユーザー指定値との完全一致、SOURCE31 file bytes/SHA256とAST構文、SOURCE/input/base commitの順序、保存NPZの**container bytesだけ**の照合、canonical checkout・ledger非衝突の読み取り専用確認である。`policy_validate`などのcontract表現検査では採択済みRatの正値・範囲を確認するだけで、Hamiltonian・sqrt witness・U/W・K_SOS・Cbar2・認証stepを評価しない。module importは定義の読込みだけで、generator／checker／adapter／runnerを呼び出さない。

管理用scriptの作成途中で、許可する文字列操作のAST whitelist不足（`startswith`、`split`、`re.fullmatch`）と既存manifestの`container.sha256`への参照誤りを修正した。これらは新しい検査scriptの問題であり、固定SOURCEや採択条件の不受理ではない。修正後の全68 named checksと、固定SOURCE由来の16個の直接require検査がPASSした。途中も数値generator/checkerや科学入力decode・claimは実行していない。

Phase Aの134人工test methods／6,777 subtests PASSは [旧記録](../qc_a2_pilot_phase_a_20261018/synthetic_test_results.json) の継承参照に限る。今回は人工数値testや実入力correctness testを再実行していない。static PASSは科学実行成功・定理の新しい検証・プラットフォーム認定・CIを意味しない。実入力finite/Hermiticity、exact intake、checker受理、実資源費用は将来の別途認可済みone-shotの対象である。

環境観測は時点情報であり、他jobの利用終了や実行時の余裕を保証しない。実行前のorigin／SOURCE／入力／index／ledger・claim／空き資源・非干渉の再確認は必要だが、それを今回の科学実行承認と扱わない。全成果の公開・保全確認後はmandatory STOP。
