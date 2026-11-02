# v1からv2への変更と保存

v1 commit `7c1a3d43f61c5501a9e79206b7c60933f94b1077`を唯一の起点とする。
zero-plan SHA `805874dcbe1466adbd92f2300b90a88a2e111e8d43543759fae4b28a9eb1d758`、
plan fingerprint `09ce44081a5a20f2e1f3dba2fced3bf8d8732251ff88393e8a190d0fa820bb88`、
公開manifest SHA `68cc9c28faaf6ab1b43799d5ffc18ece633d244634f5146239dbc93770c530f7`を照合した。
公開前manifest hash145ee4fa…を起点にしていない。

| 項目 | v1からの具体化 |
|---|---|
| D1 | minao/通常CDIIS8/cycle1/damp0、明示conv_tol1e-9/max_cycle50、raw gradientとextra-cycle独立gate、MO/積分/spin/DF規約、no rescue |
| D2 | 生成順継承、tie時の返却order保持と有意prefix tie STOP、sign/null縮退freezeと有意縮退STOP、sector36/bit順序/dense eigh、既存gate閾値維持 |
| D3 | master seed採用案20261006、actual master/input/sourceはnull、実seed未生成 |
| D4 | ASとRSSを区別。headroom16GiBと8+8w+16GiB admission、64GiB→最大5 workers、wall累積72h/output総10GiB、固定run/root案 |
| 認可 | 入力生成専用source-bound plan/認可/review/launch→freeze STOP→input-bound signal plan/別result-prior認可/review/launch→map STOP |
| status | v2 prepared awaiting review。レビュー4判断を閉じるまでは契約完成としない |
| 検査 | 新規320 synthetic JSON checks。旧129件は保存・runner再実行0 |

科学scope、6距離、218 templateとby-method counts、32 paired trajectories、74,784 logical/actual cap、内部1/最大12 workers、精度/uncertainty/identity/cache/digest/STOPは不変。
v1 record wire formatsをv2 schema filenamesから参照できるようbyteコピーしたが、v1ファイル自体は変更しない。
v1 validatorをそのまま継承し、v1検査runnerは呼ばない。

DF rank12をnonzero eigenvalue countへ言い換えない。small/null fragmentsを除外して候補scopeを変えない。
PySCF未確認のMole precision属性は固定しない。installed sourceで確認した明示controlだけを採用案にする。
null eigenspaceのfreezeは一回の具体basisを再利用する規約で、cross-host bitwise生成保証ではない。

旧v1 manifestの33件はbase commitのblobで完全照合する。
v1配下26 fileと保存証拠はworktreeでもbyte-identical。8 documentary indexes/noteは今回だけ更新しv2 manifestへ記録する。
旧247 source、公開準備25 files、保存6 JSON、validation status/manifestは不変。
原稿/図/Track B/旧結果を変更せず、snapshot/runtime/checkpoint/cacheをmaterialize・commitしない。
今回の公開認可は科学認可ではない。新science source/runner/testsの実装も別の明示指示待ち。
