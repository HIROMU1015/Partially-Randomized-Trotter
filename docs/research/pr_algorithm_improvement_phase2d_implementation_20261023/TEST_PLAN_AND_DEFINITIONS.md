# Test定義と実行境界

**TESTS_DEFINED / TESTS_NOT_EXECUTED / SCIENTIFIC_VALIDATION_NOT_PERFORMED**。

[tests/test_pr_dictionary_f1.py](../../../tests/test_pr_dictionary_f1.py)に20個のtest関数を定義した。pytest／collection／新module importを今回実行していない。静的ASTで関数名・signature・importを読んだだけである。

対象はserializerと不正fraction、q/r/time規約、exact JWとCAR、全四armの元H、P collectと微小非零、exact-zero tail、typed任意長Pauli、interval endpoint／ceil、component controlled product/rotation、support completion、8motifとfusion／actual adjoint、prep込みX/Y、actual component由来mean、価格DPと直接有限列挙、unknown price拒否、selector／既知全列挙／tie、未資格fallback、二precision reference、compile後full-wrapper operator、P0/I authorization拒否である。

これは自動的な将来pytest実行計画ではない。testをそのまま一括実行すると登録callとの重複や余分なbuildが生じ得る。将来、ユーザーが承認する最終F1 call inventoryへtestの実行単位・回数・case IDを対応させ、runner検証と二重実行しない必要がある。未承認testやmockを「軽い確認」として実行しない。

test helperのtruncation_tolerance=1.0は既存APIを明示cutoff K2で呼ぶための未採択診断値で、bias allowanceではない。runnerではこの値を既定にせず、final contractからarm別bindingを必須にしている。testを科学実行する前にAPI値とその根拠を最終契約へ固定する。unbound_test_input等はtest code中の識別子であり、実際の科学input SHAではない。

現在のtestコードを動的に確認した証拠は0件。構文PASSでruntime API整合、Gaussian convention、relative phase、数値gateを認証しない。F1失敗時の自動retry／threshold緩和／追加motifは禁止のまま。
