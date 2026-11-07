# Signed support H4 metadata-only：科学SOURCE準備

2026-11-07 JST / SERVER1。ユーザーが契約3ac50a9の一回実行を明示許可した。

[実行仕様](EXECUTION.md)、[固定設定](EXECUTION_SETTINGS.json)、[人工検証](TEST_RESULTS.json)、[SOURCE/環境lock](PROFILE_LOCK.json)を準備した。b23b3c0のsigned kernelを変更せず、薄いI/O・整合確認・記録・保存・STOPを接続する。

公開SOURCE SHA・本文/blob/treeと固定input/環境を照合後、固定H4/1.00 Å/STO-3G/DF12/8qubit/全4電子参照/L_D3/m0/q1/δ.02/r1/K2/atol0のmetadata-only一回を実行する。N1 actual Q/selected fullを維持し、N2 state/engineはロードしない。

ここまでは人工入力のみで実H4 moment未取得。ELIGIBLE/INELIGIBLE/UNKNOWNを区別し、いずれも結果保全・公開照合後mandatory STOPする。科学retry、fresh train、paired評価、Born・shot・compileへ進まない。
