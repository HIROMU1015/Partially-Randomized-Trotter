# C-S V0：新しい接続境界の人工correctness

2026-11-08 JST。**最終18 test PASS、失敗0、0.136秒。実科学call0。**

旧same_position_residual.pyと既存9testはbyte不変。元9件の順序/非可換隣接/slot挿入/controlled phase/coherent schemaの範囲を再利用し、再実行していない。F1/F2・P3・Parent/SP・SCV/C1等の全面suiteも実行していない。

新testは[test_cs_v0.py](../../../tests/tracks/algorithm_codesign/test_cs_v0.py)。新規binding/strict角境界だけを対象に、2×2人工operator、2system schema、作成した有理角を使った。

| 差分 | 人工確認 |
|---|---|
| 保存schema/basis/role | D centralとFIXED basis/pを分離、first positive R、stable site、selected basis一意対応、欠落/異常STOP |
| 元R eventとの型境界 | invented tau=.25の元_make_event処理を再利用し、binary64の符号/2倍をexact Fractionへ保持 |
| coarse | πを保存、正負tie、exact差、S0/S1/S2共通表、政策domain |
| finite table/error | sampleを禁止して既存2event/actual q/w接続、100digit独立人工operatorとの差を既知tail＋root幅が包含、exact回転との差は非zero |
| strict phase/角型 | mixed π+radian、scaled atan、2π=−I、bare Tのprojective誤りを検出、unsupported/float拒否 |
| positive/negative | 一つのpositive key、Wを含むactual adjointを独立2×2計算と照合 |
| native error/cache | exact式へのstrict interval guard、失敗guardSTOP、別precision/tool/runtime/cacheと改変receipt拒否 |
| literal局所費 | exact IZ parity CX ladder、event scalar、native_accounting、primitive guardを別途付与。full compileなし |
| error合成 | finite/coarse/weighted nativeを合成、不足event誤差・負誤差を拒否 |
| 境界/保存 | CALL_START→RETURN/STOP、8key cap・duplicate/retry/禁止call拒否、static preflightに科学activation optionなし |

最初に追加した14件も初回PASS（0.079秒）。selected policy binding、局所費/error、static preflightと元R event境界を仕上げ、18件へ拡張後の最終実行もPASS。失敗/debugによるscience条件変更なし。その後、FIXED partnerのrole型も明示的にFIXEDにする小修正を最終確認で追加し、対応するbinding test1件だけ再実行してPASSした（[局所追試log](ARTIFICIAL_BINDING_FOLLOWUP.log)）。残る17件の対象は不変として再利用。全3 invocation、18種類のtestであり、反復分を独立件数へ加算しない。最終logは[ARTIFICIAL_TESTS.log](ARTIFICIAL_TESTS.log)。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python -m unittest discover -s tests/tracks/algorithm_codesign -p test_cs_v0.py -v
```

固定13入力とruntimeを実際に読む**静的preflight**も[PASS](STATIC_PREFLIGHT.json)。D/R binder/resolver/coarse/residual_tableを実H4へ呼ばず、state vector・Born・RNG・合成・回路compile・価格・N/Gを計算しない。実providerはimport確認のみ、合成呼出0。SOURCE hashを公開前に固定した。

残る未検証はH4でのfirst slotの実適格性、実strict acquisitionの成立/価格、実finite mean/error/range、局所cap成立。人工PASSは分子性能・strict価格・改善・CI/独立再現を証明しない。full controlled列・measurement engine・N/Gへの変更伝播はV0にも含めない。
