# 研究C：H4再配置native計算のprocess並列化

2026-11-06 JST。技術実装・人工入力の検証のみ。基点は `a67036a6baafc00deaa38a1b0b801d313d519ad5`、branchは `codex/research-c-h4-relocated-process-parallel-20261106`。

実行中の固定SOURCEの科学計算は元worktreeで継続し、停止・再起動・SOURCE変更をしていない。本branchでの新しい実H4計算は0。新backend用の[未承認契約](research_c_h4_native_process_contract_unapproved_20261106.json)は実行許可を追加しない。旧結果・SOURCE・契約・STOPを保全する。

## 変更と再利用

旧nativeは32 threadを使うが、kernel build・P0 compile・actual adjoint buildがprocess内の共有Qiskit mutexとPython実行により直列化されていた。標本本体を `c_native_sample.run_sample` へ抽出し、既存thread経路と新 `c_native_process` が同じ関数を呼ぶ。抽出した標本本体はa670のtask本体とAST一致を確認した。

- `spawn` の独立processが1標本のsampling・build・compile・reference・raw/compiled/実raw逆/実compiled逆・return・XY・費用6指標・保存を完結する。Qiskitの共有状態をprocess間で共有しない。
- CAR atom、head指数行列、shell全行列を候補ごとに一度保存し、workerが読み取り専用mmapで読む。event cacheはworker内、Gaussian/relative transition cacheは再利用する。
- warmed Qiskit contextはcohortごとに一度保存する。spawnへはその小さなpathだけ渡し、大きなpickleを各起動pipeへ送ることによる直列待ちを除く。
- worker数は既存のaffinity/quota/空きメモリ/未処理32標本から決める。spawnの常駐分を見込むため観測RSSと既存dense scratch見積を加算する。固定CPU/RAM上限は追加しない。BLAS/Aerはprocessごとに1 threadで協調し、親の環境変数は終了時に戻す。
- 各標本は固定indexで一度だけ投入する。workerの最初の異常で共有STOP eventを立て、他workerは次のstageへ進まない。実行済み部分と失敗を保存し、親に同じstageの失敗を伝播する。自動retry/resumeはない。call集計の親への反映は完了記録の転送だけであり、数値処理を再実行しない。

既存 `native()` の既定はthreadのまま。新契約の `native_backend: process` でopt-inし、再配置済み候補に結び付いたprepared contextを必須とする。candidate、r、selection32/final64、統計・会計・correctness・STOPの科学条件は旧契約と機械比較で一致した。SOURCE digestを用いる将来の乱数domainは新SOURCEになるが、方式の一致テストでは同じ固定domainを指定した。

## 人工入力による結果

乱数seed724の対称JとQR実U、n4/rank8/元順LD2、線形Z再配置、r32、512次元の全入出力検査を使用した。保存分子J/U/κ/QPY/科学結果は読み込まない。

| 人工r32・同一8標本 | 旧thread | 最終process |
|---|---:|---:|
| wall秒 | 168.36 | 22.68 |
| 平均CPUコア | 1.06 | 7.33 |

同一8標本で **7.42倍**。乱数列、参照行列、raw/compiled/actual adjoint/XY検査と6指標は一致。[benchmark8.json](../../artifacts/research_c_h4_native_process/2026-11-06/technical/benchmark8.json)。

本番と同じ32標本・32workerの人工検証は **23.07秒、平均29.25 CPUコア**。32の異なるworker PIDを確認した。全32標本の全512 raw/compiled/実逆回路/return/相対位相/XYを検査し、最初の8標本は旧thread方式の乱数列・参照・各検査・費用6指標と一致した。[production32.json](../../artifacts/research_c_h4_native_process/2026-11-06/technical/production32.json)。

大きな起動pipe転送を残した開発版では32標本が62.56秒、平均9.97コア、実際に標本を処理したPIDは19だった。最終方式で起動時の待ちを減らした。これは32標本の旧threadとの直接速度比ではない。

時間にはworker起動・context保存・各標本出力・pool終了を含む。候補準備の共通setupは別記する。同じホストの単一人工fixtureの局所測定であり、実H4全体の速度、r8/16/64/128、将来ホストでの倍率を保証しない。CPU秒は親と全子processの合計、RSSは各processの観測値であり合計RAM量ではない。実測から科学的な資源優位・actual CA・全law資格を主張しない。

## 検証と来歴

既存118件＋新規4件＝**122 unique tests PASS**。全体121件は231.25秒、新backend最終4件は23.27秒（3件は重複）。位相誤りの検出、部分結果・mandatory STOP、retry/resume拒否、既存出力の保全、重複index拒否、固定32件と128 QPY保存、mmap読取専用、別process、既存adapter互換を検証した。主agent自身の確認であり独立レビューではない。既存warningのみ。

[tests.xml](../../artifacts/research_c_h4_native_process/2026-11-06/technical/tests.xml) ／ [process_tests_final.xml](../../artifacts/research_c_h4_native_process/2026-11-06/technical/process_tests_final.xml) ／ [verification.json](../../artifacts/research_c_h4_native_process/2026-11-06/technical/verification.json) ／ [新SOURCE](../../artifacts/research_c_h4_native_process/2026-11-06/technical/SOURCE.json)。既存eval venvを読み取り利用し、共有package/OS/他processを変更していない。サブagentなし。

既存のpassive CLIで新契約を指定できる。配列decodeや科学起動は行わない。

```sh
python3 -B scripts/check_research_c_h4_relocated_downstream.py --contract docs/research/research_c_h4_native_process_contract_unapproved_20261106.json
```

人工benchmarkの再現は既存eval環境で `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <eval-python> scripts/benchmark_research_c_native_process.py --output .cache/h4_process_benchmark_r32 --samples 8 --r 32`。32標本のdriverは[production32_driver.py](../../artifacts/research_c_h4_native_process/2026-11-06/technical/production32_driver.py)。fresh人工出力を使い、上記8標本の結果と照合する。

公開照合後、この技術作業はSTOP。Codexは別に承認済みの元科学計算を完了・公開・mandatory STOPする。新SOURCEへの実H4切替・二回目の科学実行・H2/H6には進まない。
