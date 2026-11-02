# SOURCE/runtime/input/import closureの技術封印

[DIAGNOSTICS_contract.json](DIAGNOSTICS_contract.json)はnull/templateではなく、このホストのcanonical pathと実SHAを持つ実契約。canonical JSON digestは `f71e52ce6f42cc17d1a97f0cbdfac311da57c80b49ee8ea69ebe34fb4d45bfd9`。[preflight](preflight.json)はこのdigestでexit0/science_GO=false。科学import・実H2演算・grant発行は0。

## 固定SOURCEと入力

| 役割 | 固定commit | file数 |
|---|---|---:|
| runner/control/backend/diagnostics/launcher/binder/policy/reference excerpt | 35444654f9159a5d85debdbc577feab0590e2610 | 8 |
| acquisition kernel/plan/formula | 4ec26921c6872ed6e8667f167074e1e405cf1a3f | 8 |
| scientific provider package | c31ef424175c02379eafebe8fc6c7548c1d9aabe | 252 |

acquisitionとproviderは固定Git objectから新しい自分専用assetsへbyte抽出した。runnerは専用worktree内の実file。各fileについてGit blob SHA1・SHA256を[source_index.json](source_index.json)へ、canonical path/role/SHA256を契約へ保存する。preflightはGit cat-file batchで全SOURCEのbyteを照合する。無関係なworktreeのHEAD/statusは実行gateにしない。

assets rootは `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h2-diagnostics-binding-assets-20261023`。source snapshotはその `acquisition/` と `provider/`。旧SOURCE/旧private原本は変更していない。assetsはlocal専用で、raw snapshot/receiptの中身はGitに新規公開しない。

| 入力identity | SHA256 |
|---|---|
| tol_only_snapshot.npz（3222 byte） | 62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa |
| tol_only_receipt.json（7759 byte） | 63acdc965e57670531fa689824c8efddd489374e95a03c05ca723a038d6ed2c5 |
| 保存Hamiltonian | d0bc260db6fc0a485fe518050075c18ccc0dff532253dcc719a52a61f41afa80 |
| 保存state | 080b7ad2a46f38ed51e2989eb399ceda4cae95fca688722959ee65463cf174bf |

NPZはhash/copyのみで配列を開いていない。Hamiltonian/state hashは保存receiptと固定bindingの宣言照合であり、今回新たに計算していない。科学承認後のsetupで配列shape/dtype/array SHA、sector/state、H/state hashを再確認する。新prep/rotation/operation hashは科学prepare2回の出力であり、今回推測して埋めていない。既存prefix1/3 prep hashも新receiptへ代入しない。

old_referenceはcc7404edの保存excerpt全byteと照合。元FP result/acceptance/completionとsnapshot等の既知原本9件も当時のSHAに一致。[原本照合](preservation_observation.json)、[旧完了run314 fileの照合](historical_completed_preservation.json)を保存した。

## runtimeを新しく封印する範囲

Pythonは既知専用venvの `.../h6-environments/20261012/native-dependency-v2/venv/bin/python`。canonical targetは `/home/AbeHiromu/projects/h6-environments/20261011/python-3.11.0rc1/bin/python3.11`、SHA `ebc2fae9fa0845b8c1fc09ed10196c0e93c26cb8fbd2a84d64548326ef93209b`。stdlib-only probeとMETADATA読取だけで版/配置を確認した。NumPy1.26.4、Qiskit1.3.0、SciPy1.14.1。保存PR Python3.12とのruntime一致証明ではなく、今回のfresh binding候補。

runtimeの静的許可集合は53 package roots、10,450 file/973,067,154 byte、ELF1130 entry。全契約はSOURCE/入力/manifestを含む10,721 file。完全inventoryとELF graphは[runtime_closure.json](runtime_closure.json)、主要情報は[runtime_closure_summary.json](runtime_closure_summary.json)。canonical manifestはassets/runtime_closure.jsonで、Gitの監査コピーとのbyte hashも同じ。

source ASTの絶対importからpackage単位の推移閉包を作り、同梱resource/extension/.libs/dist-info、stdlib、interpreter、ELF interpreter/NEEDED依存をhashした。providerのpackage初期化がplot/chemistry等をimportするため、直接の線形代数kernelより広い集合になる。package単位の保守的許可集合であり、実際に全fileがloadされたとの主張ではない。旧runtime manifestの無関係な12,202 fileをそのまま必須にしていない。tests/docs/examples/cacheを除き、共有venvやOS設定は変更しない。

readelfはwheelのRPATH/RUNPATH、$ORIGIN、同梱ライブラリ候補を静的に追う。これはdynamic linkerの歴史的実load観測ではない。Numbaの任意TBB pluginはlibtbb.so.12未導入なので許可集合から外し、attempted importを拒否する。TBB導入やthreading policy変更は行っていない。DIAGNOSTICSのsnapshot restore/QR/線形代数にそのpluginを使うSOURCE呼出しはない。必要になれば予算内で勝手にfallback修正せずSTOPする。

[Python構文監査](python_syntax_audit.json)は選択Python3.11自身のstdlib ASTで、sealed Python source7,560 fileをparseし構文error0。packageの科学コードは一行も実行していない。import時の任意副作用や全関数の動作成功をこの検査で認定しない。

## import・native混入防止

launcherは `-I -S -B` と専用の未存在cache prefixを必須にし、sys.pathを封印stdlib/siteとrunner scriptsへ設定する。site/.pth実行、PYTHONPATH、既存unchecked pycを避ける。LD_LIBRARY_PATH/LD_PRELOADは空を要求し、今回のstdlib preflightだけは子process環境から除去して起動した。共有の環境変数・設定を変更したものではない。

通常importのPathFinderでorigin/hashをloader実行前に照合する。直接SourceFileLoaderもget_codeをprocess内で一時的に検査し、sealed sourceからcodeを作るので無検査cacheを使わない。guardはfinallyで復元。backend import後、setup前と実行完了時にsys.modules originを検査し、/proc/self/mapsのexecutable file-backed mappingもcanonical path/hashがsealedであることを要求する。消去済みmappingも拒否する。

`loaded_closure_complete=false` と `scientific_imports_observed=false` を明示している。これは不足をtrueで覆ったものではなく、新しい `STATIC_CLOSED_IMPORT_SET_V1` の区別。静的許可集合を封印し、集合外への実行を拒否する方式である。科学import自体が未承認なので、実科学loadの動的attestationは今回存在しない。予期しない依存が要求されればbootstrap1を消費してSTOPし、正常な計算成功とは扱わない。

## liblzma不一致の範囲

[静的監査](liblzma_static_audit.json)では、選択Python3.11の `_lzma` RUNPATHが既知build-dependencies側を指し、canonical liblzma.so.5.4.5のSHAは `696e868dd0700a19a6d65fc01608ec2d70d3cb91f65710e89180cd2e688f30cb`。当時の旧runtime manifestと一致した。このfileをfresh closureへ含めた。Pillow/libtiffにも静的依存候補がある。

NumPy/SciPyの132 native rootsからNEEDEDをたどった146 ELFの静的graphにはliblzma依存経路がなかった。したがって直接のBLAS/線形代数kernelへの依存経路はこのgraphでは見えない。ただしpackage初期化・画像/圧縮処理を含むruntime全体で無影響と証明したわけではない。

受領資料のPython3.12 `${OTHER_ROOT_894d731f69bb}/lib/.../liblzma.so.5.4.5` 不一致は別path/runtimeの観測として保持する。今回の3.11 file一致で旧PR結果のruntime問題が解決したとは言わない。当時のworker loaded receiptがないため、旧科学結果への実影響は未判定。旧結果を無効とも再認定ともしていない。

## 実行権限・残る限界

技術preflight PASSは科学GOではない。ユーザーの政策採択、fresh runtimeとexact contractの明示承認、新grantが揃うまで実行不可。新科学run_root/cache/CONSUMED/ledgerは未作成。PR Cの当時runtime identityは別のBLOCKERであり、このfresh runtime sealでは埋まらない。
