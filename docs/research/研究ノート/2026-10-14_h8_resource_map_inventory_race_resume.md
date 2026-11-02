2026-10-14 03:19 JST、監視競合修正済みの8 worker版を再開。run ID `H8_CORE_RESOURCE_MAP_20261014_resume02`、coordinator PID643626。92 signal/126 complete groupを再利用し残466。8同時回路処理・8 memory grant・STOPなしを確認、原SOURCEと科学seedを固定。最新利用者指示でown-runの実行に必要な運用条件調整も認可され、compile1722/trajectory843/occurrence4219を採用。共有環境・他ユーザーへ変更なし。以下の承認待ち表記は旧REVIEW時点の履歴。

# H8 8 worker：監視競合修正とcomplete-only再開02

SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、branch `track-a-h8-resource-map-resume02-20261014`。利用者「修正して再開させて」により停止済みrunの監視バグを修正し再開準備を完成。科学seed source `cb8a56c6b2a772827275f9200b02638030f33f72` を保持。旧SOURCE `c6692964c948071ed00363b5b711fbca8930ad61` の265 closure中262 runtime不変、変更3はinventory/contract/technical-handoffだけ。科学adapter/compiler/input/state/DF/gates/shot/science-gridを変更しない。

## 原停止と修正

旧resume01は2026-10-13 23:56:29 JSTにnative exit1を観測（正確な退出時刻は未記録/null）。first STOPはgroup0126の `.pending_diagnostic_0013.json` がinventoryの列挙後、stat前にAtomicWriterによって削除されたFileNotFoundError。8worker動作中の監視競合で、RSS/OOM/科学信号/コンパイル失敗ではない。全worker回収/native wait-reap済み、cleanup error0・残存0。

inventoryはlstatでfile/directory symlinkを拒否し、FileNotFoundErrorかつ `.pending_*.json` だけは既知のatomic一時fileとして継続する。存在する一時fileのbytesは計上し、消えた一時fileのinodeは保守的に残す。通常成果/cache/log欠落・permission/他OSError・directory欠落はSTOP。walk errorも伝播する。write-capの走査もH8専用AtomicWriterへ同じ対応を局所結合し、旧共有module/globalを変更しない。従来exclusive publish・fsync・5秒observer・own-run cleanup・first STOP保持は不変。

## 引継ぎと予算

祖先run02の75＋resume01の51＝126/592探索group、signal92/92を旧directoryのbytes/SHA/task/source/seed/paired metrics/compiler/native receiptに結合して受理。旧manifestの祖先file hashesも再照合し、二段階継承で最初の75と4 signal groupが欠落しないよう修正した。新SOURCEへの変更は固定親c6692964とruntime変更3箇所だけを許し、異なるscience/input/compilerを拒否する。旧grant・旧lock・partial/runtime/cacheを保持、新output/run01/control01/新grantを使う。残探索466、その後のtop2/union最大8×独立32を既存core契約どおり実行予定。

親ledger known compile254/trajectory120/occurrence120、held compile22/trajectory11/occurrence11/control4を継承し返却/reset0。登録済み592探索と最大256確認の本体はcompile1700/trajectory832/occurrence4208/control360。必要最小上限補充はcompile1708→1722（+14）、trajectory835→843（+8）、occurrence4211→4219（+8）。control376の中にheld4も収まる。carry lineage2289 compile/1112 trajectoryは維持し、累積枠はcompile4011/trajectory1955案。科学標本数・候補・shotは不変。現時点はこの最小補充の承認待ちでapproved=false/runtime_authorization=false、起動0。

CPU20/21/24/25/26/27/28/29、coordinator22/23、内部thread1。準備RSS1GiBから9GiB normal/24GiB heavyへ予約し構築前にgrantを待つ8worker方式を保持、AS32・worker予約72・aggregate76/job admission96GiB。未測定/rare/largeはheavy、常時8heavyは許さない。wall capなし・各phase成功時だけ次へ、自動retry0・全core後mandatory STOP・q16/transfer/追加pilot/次stageなし。共有環境/venv/H4/他job/GPUに変更なし。

## 限定検証と実行方法

44 local artificial/mock PASS（8.52秒）、37関連外deselected。単一test process/thread1/AS2GiB/wall60秒/log16MiB、実worker0/実NPZ/実transpile禁止。消えたpendingとpublished計上、その他欠落/permission/symlink拒否、write-cap競合、二段階signal引継ぎ/祖先改変拒否、旧budget上限保持、seed/compiler/source拒否、8同時mock admission/cleanup/failed reservationを確認。旧科学検証・feasibility・signal計算を繰り返していない。これはlocal regressionでCI/外部独立reviewではない。

固定SOURCE/profile/input/native318・fresh CPU/memory/PSI/OOM/filesystem/inode/quota・own-run退出・未使用output/lockをread-only確認、条件付き技術review合格。認可後auth02を再結合しfresh gateを再検査し、`launch_track_a_h8_resource_map_v1.py --manifest <auth02/manifest.json> --grant <auth02/grant.json> --control-directory <control01>` を既存Python3.11.0rc1/venv・-I -P -Bで一度実行する。run ID `H8_CORE_RESOURCE_MAP_20261014_resume02`。private入口 `/home/AbeHiromu/projects/h6-handoff-evidence/20261014/h8-resource-map-resume02-v1/`。

[受領・資源・review](receipt_and_review.json)、[source closure](source_closure.json)、[限定tests](limited_tests.txt)。NPZ/runtime/checkpoint/cache/credential/internal SSHはGitへ含めない。
