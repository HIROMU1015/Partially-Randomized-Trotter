# Header binderの人工correctness

2026-11-08。**新規10件＋既存適格性の関連3件がPASS**。旧12件の測定instrument suiteや他研究suiteは全面再実行していない。量子simulation、state/Born、RNG、shot、compile、実N1入力の群集計は0。

## Fixtureと検証

[test_m_n0_header_binding.py](../../../tests/tracks/algorithm_codesign/test_m_n0_header_binding.py)は、N1 serializerのfieldに合わせた**発明したheader**を用いる。実LAWのlabel/質量や実D係数をコピーしない。fixtureは2/8 system wire、2つのD block、異なるoriginal/selected basis、signed/identityとzero-grid labelを持つ。matrix provenanceは人工文字列のhashであり、実matrix計算・matrix認証ではない。

| 新規test | 確認した内容 |
|---|---|
| saved schema→SlotLabel | selectedの逆列をVへ、thetaだけ反転しbeta維持、stable slot/context/law/policy、signed support、actual Fraction質量 |
| context順序・phase | Bはforward、Aはreverse、block inverseとRZZ/RZ順、ordered global phase、Literalをquantum buildへ渡すと拒否 |
| 既存群判定との接続 | 共通複数label受理、A/B/V違いの拒否、全正mass保持、exact q、signed functionの区別 |
| identity・adjoint・幅 | 空support/sign保持、A=B†の明示structure、8-wire metadataの受理 |
| singletons | 非自明群が無い場合を当該slotだけの結果とし、全massを保存 |
| 不適格header | config/slot/policy/hash、basis名/操作、sign/support、D order、scalar欠落/非有限、actual law不正等を具体的理由で拒否 |
| mock pass正常経路 | source/input preflight各1回、binding/partition各1回、全出力、counts、STOP、出力衝突で処理前拒否 |
| preflight/binding失敗 | 入力blob・source commit不一致でSTOP、失敗label/取得済みpartial記録、再試行なし |
| source preflight | 実行commit/file bytes/envの一致、変更時拒否。全部mock sourceで検証 |
| 実行境界 | 新binder/入口にQuantumCircuit/Operator/Statevector/Born/compile/RNG呼出なし |

`m_label_average.py`にLiteral受理を追加したため、既存のexact law/partition、近似/phase-equivalent入力拒否、zero/actual mass/wire拒否の**3件だけ**を回帰確認した。instrument build、readout、数値kernel本体のASTは前段と同一であり、Literal専用build拒否guardだけが追加されている。前段の12件PASS証拠は固定commit d07a76dのまま保全する。

## 人工で確認したmass例

人工label a,b,c,zeroのactual質量は1/8,3/8,1/2,0。保存component probabilityは意図的に異なる値にして、binderがそれを使わないことを確認した。a,bの適格群はmass=1/2、q=(1/4,3/4)、cは別群としてmass=1/2。人工Q(n=0)=3/4に対し各群のfull context massは3/8。zero記録は残り、正mass総和は1である。

これはheaderの変換と会計のcorrectnessであり、実H4の群・相関・分散/N/資源の新数値ではない。人工の`NONTRIVIAL_GROUP_FOUND`は実H4の同判定へ流用しない。A=B†情報も既知直接測定の比較を残すためであり、優位性の認定ではない。

## 保存SOURCE・実行

最終記録は[artificial_run02/SOURCE.json](../../../artifacts/o2_m_n0_header/2026-11-08/artificial_run02/SOURCE.json)、[TEST_RESULTS.json](../../../artifacts/o2_m_n0_header/2026-11-08/artificial_run02/TEST_RESULTS.json)、[RUN.txt](../../../artifacts/o2_m_n0_header/2026-11-08/artificial_run02/RUN.txt)、[STOP.json](../../../artifacts/o2_m_n0_header/2026-11-08/artificial_run02/STOP.json)。Python3.12.3、NumPy1.26.4、Qiskit1.3.0。新10＋関連3、0 failure/error、test入口の測定区間0.1162秒、peak RSS 461,208 KiB。import前の全wall時間を測った値ではない。

人工run01は新9＋関連3がPASS。非自明群なしの分岐とsource依存の固定範囲を追加した最終run02もPASS。前回記録を上書きせず保全した。追加差分に根拠がある再検証だけを行い、旧suiteを拡大再試験していない。既存namespace import時のdocstring SyntaxWarningは旧ファイル由来で、旧SOURCEを直していない。

人工入口は[run_m_n0_header_artificial.py](../../../scripts/tracks/algorithm_codesign/run_m_n0_header_artificial.py)。実科学入口[run_m_n0_metadata.py](../../../scripts/tracks/algorithm_codesign/run_m_n0_metadata.py)はmock以外では起動していない。公開commitの最終コードをSOURCE hashと照合するが、このlocal人工結果をCI/外部再現・実N1資格とは呼ばない。
