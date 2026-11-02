# 実施した環境検査

主結果は **PASS_ENVIRONMENT_MANAGEMENT_ONLY**。NumPyのimport・C API ABI identity・環境file/native再照合が通過した。RTE／mixed prototypeのruntime正しさ、科学的な誤差・性能を検証した結果ではない。

| 検査 | 実際の結果 |
|---|---|
| interpreter起動 | CPython3.11.9、GCC11.4.0、x86_64、SOABI cpython-311-x86_64-linux-gnu |
| 実binary | venv symlinkの実体、libpython、build configをprivate receiptへ固定 |
| Python file tree | base prefix 3471 file、venv 925 file。pyc/__pycache__を除く。再照合で同一 |
| package閉包 | runtime distributionはNumPy1.26.4だけ。pip/setuptoolsは専用baseのbootstrap側にあり、venv siteへ入れていない |
| wheel | 指定cp311/cp311/manylinux_2_17_x86_64 wheel、18,252,005 bytes、PyPI SHA256と実bytes一致 |
| installed RECORD | hash付き917 fileでhash/size一致。RECORD自身は個別SHA固定 |
| wheel→installed | RECORDを除く914 fileを逐byte照合。pip生成entry point／INSTALLER／direct_urlはinstalled RECORDと実treeで別固定 |
| NumPy import | 専用venv由来、version1.26.4。import時のupstream初期化を許容し、ユーザーの研究用array／数値演算は呼ばない |
| C API ABI | runtime `0x1000009` と導入headerのNPY_ABI_VERSION `0x1000009` が一致。既存extensionのimport成功も確認 |
| native | 実mapped ELF 55件（interpreter／stdlib extension／NumPy extension／BLAS／system dependencies）をSHA固定。全DT_NEEDEDをmapped SONAMEへ一意に解決 |
| BLAS | bundled OpenBLAS、wheel metadata `0.3.23.dev`。実library hashも固定。研究用BLAS計算・速度比較は0 |
| 分離 | -I／-B、system site false、user site false、external sys.path0、pth0。PYTHONHOME/PYTHONPATH/PYTHONUSERBASE canaryが混入しない |
| 環境管理再照合 | wheel/RECORD、base/venv全file、native、SOURCE10、helper、isolationを別processで再照合してPASS |
| 科学処理・権限 | B11call0、F1/F2 0、研究SOURCE import0、grant/marker0 |

検査時刻、個別hashとnative解決先は[lock](ENVIRONMENT_LOCK.json)とprivate receiptへ保存。公開側では内部絶対pathをlogical IDへ変換した。OS library versionは実ファイルの所有packageをdpkg-queryでread-only確認した値であり、推測していない。

## 実施commandの形

実pathを含むcommandとreceiptは非公開で保全する。以下の変数は表示用placeholderで、科学runnerの起動指示ではない。

```bash
"<CPYTHON_PREFIX>/bin/python3.11" -I -B -c 'import sys; print(sys.version)'
"<CPYTHON_PREFIX>/bin/python3.11" -I -B -m venv --without-pip "<VENV_PREFIX>"
"<CPYTHON_PREFIX>/bin/python3.11" -I -B -m pip --isolated --python "<VENV_PREFIX>/bin/python" install --no-index --no-deps --no-compile --require-hashes --only-binary=:all: --find-links "<PRIVATE_WHEELHOUSE>" -r "<PRIVATE_REQUIREMENTS>"
"<VENV_PREFIX>/bin/python" -I -B scripts/management/pr_binding_environment.py collect --source-root "<FROZEN_SOURCE_ROOT>" --wheel "<PRIVATE_WHEEL>" --private-dir "<PRIVATE_RECEIPT_DIR>"
"<VENV_PREFIX>/bin/python" -I -B scripts/management/pr_binding_environment.py verify --source-root "<FROZEN_SOURCE_ROOT>" --wheel "<PRIVATE_WHEEL>" --private-dir "<PRIVATE_RECEIPT_DIR>"
readelf -d "<ACTUALLY_MAPPED_ELF>"
dpkg-query -S "<ACTUAL_SYSTEM_LIBRARY>"
```

追加の技術probeは、NumPy C API versionとheaderの一致、-Iでの環境変数canaryの無視。NumPyをimportする技術processは5件（初回collector、ldd診断、修正collector、verifier、ABI probe）。B.CONTEXTSの5callとは別であり、B logical callsはいずれも0件。

## 検査手順で判明した点

初回collectorは、wheel同梱libを単独lddしたときの`not found`を未解決nativeと誤判定して終了した。実際のNumPy importは成功しており、親extensionのRPATHによってlibgfortran／libquadmathがprocessへmapされていた。旧collector SOURCEと診断記録をprivate保全し、管理helperだけを実mapping＋readelf方式へ修正した。科学SOURCEは無変更。実際の解決先をSHA固定し、再照合は通過した。科学callのretryは0件。

offline pipでは`--isolated`によりPIP_CACHE_DIRの環境変数が採用されず、既存cacheへ書けない旨の警告後にcacheを無効化して導入した。既存cacheを変更していない。今後の再構築手順は`--cache-dir <PRIVATE_CACHE>`を明記し、共有cacheを使わない。

## 資源と限界

構築前に32 logical/affinity CPU、約67.1GB RAM（約56.6GB available）、約373.7GBの空きdisk、低いloadを確認した。python-buildは`MAKE_OPTS=-j32`、file hash／readelfはaffinityに基づく32worker。実CPU占有率・最大RSS・speedup倍率は取得しておらず主張しない。科学計算の並列実行は0件。OS／scheduler／quota／共有envやcache／他processを変更せず、wall/CPU/RAMへの人工capも追加していない。

未検査は固定RTE APIの科学実行、係数map／law／input、回路、F1/F2、数値精度、第二環境での独立再現。これらを本環境検査のPASSに含めない。
