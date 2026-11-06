# B2 P1：m=1..16の固定継続

2026-11-07 JST。ユーザーは本16roundの科学実行を明示承認済み。
起点は`ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5`、前回SOURCEは`5ef55f923e73e61bada383d0cfe98283e7984e9d`。
P1の入力・予算・allowance・r/K/prefix/deltaを変更しない。

既存のprivate loader、B2 mean-only kernel、5信号評価、資格判定を直接再利用する。
回路を作るpreparation/controlled_checksは呼ばず、保存済み係数projectionと有限law、
既存のexterior-power参照による理想Gaussian/diagonal平均の数値合成を用いる。
これは新しい回路・support探索ではなく、前回native証拠を前提とする平均operatorの接続である。
projectionを実行可能なB2回路preparationと偽装せず、hashは別名で保持する。

実行前に最小SOURCE差分・人工testをcommit固定し、入力identityを照合する。
m1から16へ逐次進み、各roundの5信号・数値照合・成分cap・正半径・normalizationを判定。
最初の不適格で停止し、後続roundを実行しない。全PASS時だけ保存m0/17をコピーで結合する。
m0/17信号と代表controlled16件は再計算しない。private原本は不変更・非公開。

branch成功、shot、価格、総費用、baseline、sampling、回路build/compile、詳細レビューは範囲外。
結果・REPORTの公開とremote SHA/tree/内容照合後、mandatory STOP。
