# F1 compile cacheのmetadata修正（2026-10-24）

## 修正と確認結果

Qiskit 1.3.0では `QuantumCircuit.metadata` に辞書だけを設定できる。
F1最初のcompileで発生した `TypeError: Only a dictionary is accepted for circuit metadata`
に対し、[compiled_wrapper](../../src/trottertracks/pr_dictionary_validation/gates.py)の
cache用コピーを `metadata=None` から `metadata={}` に変更した。
元の回路は変更せず、説明用metadataとnameをcache識別から除く既存の方針を維持する。
角度、global phase、演算子順序、compiler設定、P0の科学条件・許容差は変更していない。

[管理用回帰テスト](../../tests/management/test_pr_dictionary_compile_metadata.py)2件を、
既存F1専用Python 3.11.9／Qiskit 1.3.0環境でローカル実行した。
修正前は両件とも上記TypeErrorで失敗し、修正後は両件PASS（0.017秒）となった。

- Qiskitの実際のmetadata descriptorをproxyに適用し、空の辞書の受理と元データの保全を確認。
- 説明用metadata／nameだけが異なる場合のcache再利用と、payload・phase・SOURCE identityが異なる場合のcache分離を確認。

QPY出力とtranspileはmockに置き換え、payloadも数値計算を行わない文字列を使用した。
実際のQuantumCircuit構築・QPY serialization・transpile・F1科学callはすべて0件。
この確認はmetadataのAPI不整合を対象とし、実回路のcompile成功や科学的正しさを認定しない。

## 保全と次の実行

修正branchは `research/pr-algorithm-improvement-phase2d-f1-metadata-fix-20261024`。
起点は停止結果の公開commit `594bab9e1441699474f262b075390db3ab8723c6`。
旧科学SOURCE `31e8391a56fbb1a0e64f477f78e242655329de99`、
[停止結果](pr_algorithm_improvement_phase2d_f1_execution_20261024/handoff.md)、
保存B入力、raw結果、使用済みgrant・marker・STOPは保全する。
元workspaceへ修正を混入させず、独立cloneで作業した。

旧F1-PRE manifestは旧SOURCEの記録として維持する。
修正版SOURCEのbindingは新たに固定する必要があり、旧manifestや使用済みgrantを流用して実行しない。
本作業では新しいgrantを発行せず、F1のretry／resume／科学実行を行っていない。
F1の判定は **F1_TECHNICAL_FAILURE_PARTIAL_RESULTS_MANDATORY_STOP** のまま。

次の担当はユーザー／GPT：修正版による新しいF1科学実行の承認・範囲を判断する。
承認後の担当はCodex：修正版SOURCE・保存入力を照合し、新しいgrantと専用outputを用意して実行する。
F2へは移行せず、**mandatory STOPを維持する**。
