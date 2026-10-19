# 受入条件・STOP境界

1. 新しいfactor専用の明示実行承認とgrantが、固定SOURCE/contract/binding/delta/core/説明SHAおよび出力先と一致する。
2. 固定SOURCE5 file・全14入力・runtime68 file・保存sideの来歴が実行前に一致。旧core32、他31比較、PR/PF sideは再計算0。
3. 1 comparison/2 axes/各7 factor＋7 log factorの出力schema、正の有限factor・有限log、入力digestとscopeが一致する。
4. marker/ledger/claimは新規で排他的。科学factor pass1回。既存出力や旧grantの再利用を拒否する。
5. native親/workerのexit0/wait/reap、worker/結果SHA、実行後SOURCE/input/runtime/controlが一致する。
6. PASSならdeltaを保存してmandatory STOP。不一致・欠測・gate/native失敗なら非zero/STOPを保存しretryしない。科学的な同定義/整合条件を緩めない。

資源を観測し、小さい1比較処理に適した親/workerを用いる。CPU/RAM/workerの恣意的な恒常上限は設けない。無関係worktreeのHEAD/statusや一時的PSIは停止条件にしない。共有環境・他jobには書き込まない。

今回の準備STOPは実解析を開始した証拠ではない。grant false、実factor式呼出し0、科学output未作成、旧比較因子31/32を維持している。
