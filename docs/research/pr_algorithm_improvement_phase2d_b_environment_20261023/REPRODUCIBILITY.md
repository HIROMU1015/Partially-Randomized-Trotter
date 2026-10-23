# 専用環境の再利用・再構築

今回作った専用環境を再利用する場合、ユーザー向け非公開引継ぎにあるinterpreter・venv・wheelhouse・receiptを使う。別のvenvへcopyして同じidentityとは扱わない。-I/-Bで管理helperを使い、full file/native treeと固定SOURCEをread-only再照合する。再検査では新しい自己所有0700のreceipt directoryを用意し、既存のenvironment_receipt_private.jsonとenvironment.jsonをbyte同一でコピーする。その新directoryを--private-dirへ指定してverifyし、verification receiptをexclusive保存する。実venv・base・wheel・SOURCE rootは元の固定実体を指定する。既存verification receiptを上書き・削除しない。

構築手順の再現は、同一version・source/wheel・build toolchainと分離方針を再現するもの。絶対prefix／build日時／toolchain差を含むため、他hostでbit-identicalなCPython binaryを得られるとは主張しない。実binary hashが変われば新しいlockと承認対象になる。

1. 自己所有の新しいprivate root、cache/temp/build/wheelhouse、専用prefixを用意する。共有venv／cache／pyenv/shimsへ書かない。空き資源と非干渉を確認する。
2. pyenv/pyenv `6393a4dfce90615792751c6567c07a118a961ff9` のpython-buildと3.11.9 definitionを使う。指定source archiveの実SHA256を検査する。今回sourceは `9b1e896523fc510691126c864406d9360a3d1e986acbda59cda57b5abda45b87`。upstream sourceは[Python公式配布](https://www.python.org/ftp/python/3.11.9/)。
3. `PYTHON_BUILD_BUILD_PATH`、`PYTHON_BUILD_CACHE_PATH`、`TMPDIR`、`PIP_CACHE_DIR`はprivate root内へ、`PYTHON_BUILD_SKIP_MIRROR=1`、`PYTHON_CONFIGURE_OPTS=--enable-shared`とする。今回のMAKE_OPTSは実affinity32に合わせた`-j32`。再構築時はその時の資源と非干渉から決め、恣意的な低いcapやwall capを設けない。
4. `python-build -k 3.11.9 <CPYTHON_PREFIX>` を起動する。OS/root/global installなし。private baseにはCPython同梱ensurepipのbootstrap用pip/setuptoolsが導入されるが、B venvへは導入しない。
5. `"<CPYTHON_PREFIX>/bin/python3.11" -I -B -m venv --without-pip <VENV_PREFIX>`。system-site-packages=falseを確認する。
6. [PyPI NumPy1.26.4](https://pypi.org/project/numpy/1.26.4/)のcp311/cp311/Linux x86_64 wheelをprivate wheelhouseへ取得する。filename、bytes、SHAはlockで固定。checksumなし・別version・source buildへ切り替えない。
7. requirementsは次の一行。bootstrap側pipから`--isolated --cache-dir <PRIVATE_CACHE> --python <VENV_PREFIX>/bin/python install --no-index --no-deps --no-compile --require-hashes --only-binary=:all: --find-links <PRIVATE_WHEELHOUSE> -r <PRIVATE_REQUIREMENTS>`でoffline導入する。今回の実導入ではcacheは無効だったが、再構築手順は明示したprivate cacheへ統一する。

```text
numpy==1.26.4 --hash=sha256:666dbfb6ec68962c033a450943ded891bed2d54e6755e35e5835d63f4f6931d5
```

8. B-PRE固定HEADのSOURCE rootと新しい0700 private receipt directoryを指定して環境helperをcollect／verifyする。collect/verifyは環境metadata、NumPy import、C ABI/native、file identityだけ。研究SOURCEはimportしない。今回の検査結果を他環境のPASSへコピーしない。
9. full private receipt、B用7field environment record、公開summaryのcanonical digestを区別する。個別native libraryは実mapped path／SONAMEで解決し、standalone lddだけでwheelのRPATHを判断しない。
10. 環境固定が終わったらSTOP。B科学実行は新しい明示承認とfresh packetを必要とする。

公開lockはversion/source/build/relative file/native SHAを提示し、private receiptは実体位置・exact build configuration・absolute native resolutionを保全する。venv／binary／private cacheそのものはGitHubへ公開しない。
