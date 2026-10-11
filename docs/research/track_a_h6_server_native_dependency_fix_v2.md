# H6 native依存修正 v2：一回Phase A起動

2026-10-12 JST。最新利用者指示に基づき不足依存を補完し、index447を新grantで一度起動する。
旧run/消費済みgrant/旧環境・旧449結果・H4は保全。自動retry/resume・Phase B142・確認32・H8へ進まない。

| 固定項目 | Identity |
|---|---|
| branch | track-a-h6-native-dependency-fix-20261012 |
| 修正SOURCE | 6a924e52cf1b6541e2780e9d1b756b8049683166 |
| science SOURCE | 5a903aea2492c2d211c1e15e7737df0d50694c12（224 paths不変） |
| runtime closure | 233 paths |
| manifest digest | 4a5d5544fb75fe5aa9879391d97117266d1cfa61388e5be14622c098b465c98a |
| profile SHA256 | 4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75 |
| task | index447 / B1_p19_q8_4th / replica0 / seed17585593421502146214 |

## 修正と検査

前回はqiskit-nature不足でGaussian準備がImportErrorとなり、evolution build/full-wrapper transpileに未到達。
科学コードの代替は導入せず、science requirementsに固定されているQiskit Nature0.7.2・Aer0.15.1と、
Qiskit1.3で使用可能なAlgorithms0.3.1・psutil5.9.8を、新しい専用venvへwheel bytes/hash固定で追加した。
[PyPI Nature0.7.2公式metadata](https://pypi.org/pypi/qiskit-nature/0.7.2/json)の依存条件を参照。
旧47 distributionsのversion/hashは維持し、51 distributions/34273 critical filesを新profileへ固定。
旧profile33343 filesは変更0。既存venv・H4・共有環境へinstall/update/設定変更なし。
新venvは `/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv`。
既存home専用CPython3.11.0rc1 prefixとprivate build libsをread-onlyで使用する。

旧environment gate9項目・compiler hash 1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b を維持。
NatureのBogoliubovTransformを、固定scienceの実import関数からpre-grant gateで検査し、そのsource bytesをprofileへbinding。
分子・array・回路構築・transpileを依存確認中に実行しない。
Coordinator RSS/VmSize/VmHWM/VmPeakを観測journalへ追加。失敗workerのterminal high-water marksとcoordinator wallも保存する。
新v2 namespace/lockを用い、旧output・grantへ書かない。

限定51 tests PASS（6.01秒、local、immutable CI/独立reviewではない）。実inputs/追加transpile0。
不足/異version/transitive import失敗/profile binding、旧grant/一回gate、32GiB継承、監視I/O遅延・欠測・RSS・cleanupを含む。
通常1 test driver/internal thread1、cleanup fixtureだけ最大4 tiny owned processes。AS2GiB、wall120秒、file16MiB。
旧科学benchmarkの再実行、stress campaign、追加候補探索なし。
旧native binary bit identityと実H6 compiled outputの完全同一性は未検証として保持する。

## 実行条件と公開状態

linear H6/1Å/STO-3G/DF rank19/sector400/保存state/T0.8、92候補/4精度/seed/normalization/shot/PF・PR/native lowering/compiler不変。
旧33 frozen inputs/5220 raw/449 groups/902 wrappers/92 signal/84 B3をbyte/Git/reuse契約で再照合。
Science SOURCE224とscientific plan digestは不変。今回compile2/control_probe4/trajectory0/occurrence0の一groupだけ。
Worker1 CPU20/21、AS32/32GiB。Coordinator CPU22/23、soft2/hard32GiB。内部thread1、nice10、aggregate RSS40GiB/job RAM64GiB。
これは資源予約ではない。fresh host216.25GiB/effective64GiB、running host168.25GiB/effective16GiB、PSI/OOM・監視1秒/max5秒は維持。
Total/phase wall capは既存scientific planのnullを保持。output2GiB、journal32MiB、worker log1MiB、worker output32MiB。
Fresh filesystem10GiB/100000 inodes・quota・CPU/H4非重複・output/lock未使用を検査し、起動時もrunnerが再検査する。

旧run05などH4 carryへ変更を行わない。H6旧STOP/失敗attemptの未記録actualを0と推定せず保全する。
利用者はscheduler/正式規則が特にない旨と、他ユーザーに著しく迷惑をかけない範囲での直接起動を確認した。
最新の「実装不足を補い開始」指示を新grantの根拠とする。承認範囲・資源上限を拡張しない。
静的manifestのH6_NOT_AUTHORIZED/DRAFT_NOT_AUTHORIZATION flagsは、一回grantがない場合のgateを維持する。
新approved grantだけが今回の一回起動を認可する。成功/失敗後mandatory STOP。

既存origin branchはe51f19b649a7feb3f7f84d1e22cf49fc57e0e412と照合済み。
新SOURCE/REVIEWはlocal immutable commit。HTTPS pushは既知のcredential未設定で、設定変更は行わない。
REVIEW自身のSHAはcommit後のgit rev-parse HEADまたはlaunch receiptを参照。未公開をremote PASSとは記録しない。

資料入口は [準備v2](../../artifacts/resource_applicability/track_a_h6_server_preparation_v2/2026-10-12/README.md)。起動後のPID/時刻/状態はlaunch receiptとruntimeのprogress/ResourceTraceで確認する。
未完了の143中447のみに着手し、成功後の残142と独立32確認は別認可を要する。
