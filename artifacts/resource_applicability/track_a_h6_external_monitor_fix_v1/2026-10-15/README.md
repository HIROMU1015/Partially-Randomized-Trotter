# H6独立資源観測修正・run03

SOURCE `b73580dd7402df5d8ac0a95789ec351c05209049`。run02はnative wait2、OBSERVATION_STALE、compile0で消費済み。既存metadata launcherからdriver/6compile childのmemory・CPU/AS/RSS/affinity/thread・PSI/OOM・fs/inode/quotaを1秒観測する。追加role/process0、5秒監視と資源・科学・compiler上限を維持。新14＋既存70の84限定tests PASS、2process以内のGIL保持C実動作12.22秒中も最大観測間隔.259秒、native0/reap。旧失敗fixtureを含む全私有証跡を保持。

新grantは旧run01/run02/診断carryを保持。source/profile/input bytesと未使用outputを照合し同担当binding確認。サブエージェント0、独立reviewとは扱わない。最終認可/sealを固定してfresh gate後にrun03を一度実行する。次stage未認可。
