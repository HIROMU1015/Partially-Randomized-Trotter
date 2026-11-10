# tests の役割

`tests/` は、ライブラリAPI、数値恒等式、成果物schema、ガード条件の回帰検査を置く。
基本的に `src/trotterlib/<name>.py`、`scripts/run_<name>.py`、
`tests/test_<name>.py`、`docs/<name>.md`、`artifacts/<name>/`を一組として読む。

テスト通過は、実装が期待した規約を満たすことを示す。一方で、次を単独では意味しない。

- 大きな系でも同じ近似精度になること
- 実量子backendやnoise下で成立すること
- 未検証のパラメータ範囲への外挿が正しいこと
- 最終的な総コストや手法間の優位性が確定したこと

研究上の証拠statusは[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)と
[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)を参照する。

最新のRPE長round検証は`test_rpe_target_round_horizon_validation.py`と
`test_rpe_delta_round_schedule_validation.py`、`test_rpe_delta_compiled_cost_validation.py`を、
対応する検証文書・artifactと一組で読む。最後の検証は中央RTEブロックだけを扱い、
Hadamard 1 shot全体や最終総コストの検証ではない。

## 過去のtest SOURCEの保存

[G6 formal test SOURCE](tracks/algorithm_codesign/test_g6_return_aggregation.py)は元snapshotの内容を保存したもの。
[元の保存記録と今回の収録範囲](../docs/research/main_collection_20261103.md)を参照。今回そのtestを再実行しておらず、mainでの新しい科学的検証結果を意味しない。

## main集約の版差と原SOURCE（2026-11-03）

[索引の統合・SOURCEの版差・保存原版](../docs/research/main_collection_versions_20261103.md)から、同名ファイルの各公開版とmainの配置を確認できる。
過去のSOURCE・test・結果は元snapshotの条件で読む。科学実行と元の研究testの再実行は行っていない。

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](../docs/integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。

## サーバー照合からの成果収録（2026-11-03）

[収録ファイル・保存版と確認範囲](../docs/research/main_server_results_collection_20261103.md)から、過去の公開成果と元SOURCEを辿る。
既存mainの科学ファイルは上書きせず、研究test関連の原版は実行contextと分けて保存する。新科学計算は行っていない。


## C-α sector訂正後の非科学的接続確認（2026-11-10）

`python tests/test_calpha_h4_step1_mean_contract.py -v` はstdlib/mockによる8件。科学moduleのimport、配列、人工行列simulationを使わず、二点scope・出力保全・例外保存・定義域を確認する。実H4の数値性能を検証するtestではない。

## C-α macrostep・区間処理の人工確認（2026-11-10）

`test_calpha_macrostep_bounds.py`の8件と`test_calpha_interval_bounds.py`の5件は、非可換stage order・局所次数・正確な慣性count・指数／log級数の包含・複素共役・scalar geometric sumを確認する。分子入力や旧信号の再simulationはなく、研究保証の正式採択・実機性能・総費用のPASSとは区別する。

## C-α固定gate lawと会計の人工確認（2026-11-10）

`test_calpha_sampling_law.py` 4件、`test_calpha_rpe_accounting.py` 3件、`test_calpha_price_connection.py` 3件は、gate意味論・CDF・既存branch／Hoeffding・両軸価格接続・basis取消の期待命令数を人工例だけで確認する。分子信号・旧runの再実行を含まない。
