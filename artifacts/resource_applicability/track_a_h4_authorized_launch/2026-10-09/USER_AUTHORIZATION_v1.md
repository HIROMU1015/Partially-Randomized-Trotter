# Explicit user H4 one-shot execution authorization — 2026-10-09 JST

利用者の最新指示を実行認可の根拠として固定する。

起点branch：track-a-h4-native-proof-seal-20261009
起点REVIEW：5de745199748a6523ba84b07a0ea61b2b831815b
SOURCE：6bd1ba01cd71ec3e2071082963c9f07478dada9a

承認内容：
・固定済み候補environment/compilerを、既存venvを変更せず本番採用。旧compiler output完全同一性の未検証は保持。
・worker12：CPU [2,4,5,6,8,9,10,11,12,13,14,15]、driver：CPU16、observer：CPU18。
・own-run限定affinity、内部thread1。
・observer AS256MiB/RSS64MiB、admission120.25GiB。
・累積actual上限だけ74784→74804へ変更。
・carry20件・165214360 bytes・5466.188392877579秒を保持。
・10GiB/72h等の他上限、科学条件、compiler optionsは維持。
・H4 signal/compile mapを一度実行し、完了またはfail-closed STOP後に停止。
・自動retry・入力再生成・次stage進行・旧partial/cache再利用・GPU query/use・追加準備campaignなし。

既存proof受領・技術seal・回帰testsを繰り返す必要なし。最終digest/binding/sealを再結合して独立最終review。
最終artifact固定後、SOURCE/profile/input/carry、fresh CPU・memory・pressure/OOM・容量・inode・quota、unused output/control・one-shot lockを直前検査。
全条件PASSなら追加承認待ちでSTOPせずそのまま一度起動。不合格なら起動せず具体原因を報告。
作業はhome配下、共有環境・既存venv・他jobを変更しない。
起動確認後run ID/PID/log/SOURCEを報告し、既定監視下で完了またはSTOP後に終了する。

独立review PASSの実reportとfinal artifact commitを確認するまで起動しない。これは追加利用者承認ではなく、この指示の実行条件である。
