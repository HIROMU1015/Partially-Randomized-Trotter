# T2 scalar不一致：preparation-only診断

2026-11-13 JST。基点752b9747aに対し、ユーザーが承認したPR-LD2/m12の保存入力復元1回・context/prepare1回だけを行う。sampling/event/path/native evolution/full wrapper/compile/T2 retryはすべて0。科学実行前版であり、実scalarはまだ取得していない。

[PLAN](PLAN.json)／[人工確認](TESTS.txt)／[次の判断](NEXT_STEPS.md)。既存T2のread-only preflightと保存input loader、元NativeRuntime.contextを再利用し、例外の同一frameのprep/scalar/λを捕捉する。guard、演算順序、prepare body、expected scalarは無変更。失敗後にprepareを再呼出ししない。

SOURCE読解では、Aの保存descriptorはtail identity係数を保持し、A signalではhamiltonian constantを別に加える。D748の専用selectorはdescriptor値をexpected total scalarへ渡している。定義の取り違えが疑われるが、同一preparationで実値を取得してから結果を確定する。今回は修正・新scalar採用・T2へ進まない。

診断caller・PLAN・人工確認をcommitで固定してから一回実施し、値・decimal/hex・差/ULP・取得可能なpreparation/component/basis hashと時間を新run01へ保存する。既存T2 run01、SOURCE/input/raw/cache/KEEP/grant/ledger/marker/STOPは保持する。結果公開照合後mandatory STOP。
