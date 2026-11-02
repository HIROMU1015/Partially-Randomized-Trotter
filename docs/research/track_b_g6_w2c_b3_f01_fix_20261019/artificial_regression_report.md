# 人工回帰検証：新26件＋既存84件

[入口](README.md) / [機械可読結果](artificial_test_results.json)。以下は今回実際に実行したローカル人工テストであり、immutable CI・外部再現・実G7 correctnessではない。

| Suite | 件数 | PASS | FAIL | ERROR | SKIP |
|---|---:|---:|---:|---:|---:|---:|
| 新F01 regression | 26 | 26 | 0 | 0 | 0 |
| 既存B3 driver | 36 | 36 | 0 | 0 | 0 |
| 既存補助SOURCE | 48 | 48 | 0 | 0 | 0 |
| 合計 | 110 | 110 | 0 | 0 | 0 |

既存2 suitesのSOURCEは不変。各suiteを独立Python processで実行し、終了時にG7 generator/ReturnKernel/F3 precision-generator/G9等の科学moduleがsys.modulesにないことも確認した。new mockのchild failure検査は自分のprivate childとtempだけを使用する。FakePortは人工fixtureで、実承認やclaimを作らない。

## 新規coverage

| 失敗・境界 | 検証内容 |
|---|---|
| 正常sample→次sample例外 | 前eventがinflightへ漏れない。現在trial IDと未取得、前完了prefixのcounter/digestが対応 |
| X完了→Y初回sample例外 | X最終eventはlast_completedだけ。Yのinflightはnull |
| Y bits factory／identity例外 | axis=Y、trial ID未開始、result/event未取得、X完了prefixを保持 |
| Pair初期／最終identity例外 | 未開始／全trial完了を区別。前axisのeventをinflightにしない |
| Resultアクセス・validation／event validation | 取得済み結果・未観測event・現在event・未完了を区別 |
| Angle・evidence例外 | 現eventだけをinflightへ保持し、完了counter/digestを増やさない |
| Serialization・candidate digest例外 | コピー上の候補だけが変わり、最後の完了prefixは不変 |
| Emit前失敗・部分write | 完了済みrowと末尾の不完全bytes、候補eventと前完了traceを区別 |
| Full-row write後flush例外 | fileに完全な行があってもemit未return／trial未完了とする |
| File hash更新例外 | write返却byte数は残し、未対応hashはnull。成功hashと偽らない |
| Failure record／frozen descriptor例外 | 原例外を保持しsecondary原因を別記録。STOP/retry0を保持 |
| Child failure＋close例外 | trial provenanceをchild logへ伝え、cleanupで原原因を隠さない |
| 複数trial・ZERO・commit後memory例外 | stale ID/event混入なし。ZERO完了と、commit後の別処理失敗を区別 |
| 正常8ケース | 修正前の人工referenceと正常出力・trace・counter・bit数・query数・coarse記録が一致 |

共通assertionで、失敗axis/trial ID、result取得状態、last_completedと完了数、null/inflight、完了traceのdomain、trial_completedとemit_returnedの関係も検査する。各test ID・ログ・bytes/SHA256はmachine結果へ記録した。

## 正常系referenceの来歴

基点`e4af6c06...`の未修正driverで、編集前にFakeGenerator/FakeSession/FakeClock/Bitsだけの4方式×A/Bを実行した。各axisのtoy4 trialを使い、完全prefixとcompact保持の両方を含めた。referenceは新fixture JSONとして保存した。[fixture](../../../tests/tracks/algorithm_codesign/fixtures/g6_w2c_b3_f01_normal_reference.json)

修正後testはこの固定referenceを読み、summaryのexact equality、JSONL全bytesのSHA256、chain、行数/bytes、H_bits_supplied、packet/event query数、fake-clock coarse stageを照合した。referenceを修正版で更新して合わせる処理は行わない。fixtureのSOURCE SHAも旧`d7947af1...`に固定する。

これは実G7のlawを再現したfixtureではなく、正常実行pathの回帰検査である。実law/科学入力の成立性は将来の明示承認されたcorrectnessに残る。CPUの実費・速度倍率・性能優位も測っていない。

## 実行記録・限界

SOURCE変更中の既存84・新26の確認もすべてPASSし、最終SOURCEでは3 suitesを独立実行して上表を確認した。新testに共通assertionを補った後、その26件だけ再確認した。繰返しテストは同じ開発検証であり、別の独立再現として数えない。

外部atomic admission、実storageの電源断耐性、OS kill/OOM後の完全保存、real G7 sampling/angle成立性、31,710,848 callのruntime、C1/nativeを検証していない。旧実行契約・claimの発行/消費は0。**mandatory STOP。**
