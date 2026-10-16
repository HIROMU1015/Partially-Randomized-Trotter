# M1-S Gate 0：runtime・resource技術契約候補

2026-10-16 JST。**CANDIDATE / NOT ADOPTED / SCIENTIFIC EXECUTION NOT AUTHORIZED**。

対象は採択済みH4 1.00 Å/STO-3G/8 spin orbitals/DF rank12、保存順G3/6/9のoracle診断のみ。科学実行、benchmark、NumPyのimportによる動作確認は今回実施していない。CPU/RAM/環境の観測はOS・distribution metadata・関連binaryのfile hashのみ。[runtime候補JSON][RUNTIME]、[precision設計][PREC]を参照。

## 1. 固定候補

| 項目 | 候補 |
|---|---|
| Python | 既存`.venv311/bin/python`、**3.11.0rc1**を正確にpin。final版3.11.0と混同しない |
| NumPy | **1.26.4**。metadataで確認、importは未検証 |
| Build identity | NumPy initializer/config/core extension、OpenBLAS/gfortran/quadmathの計6 fileのSHA256をruntime候補に記録 |
| CPU | workers=1。G3→G6→G9を順次処理 |
| Threads | OPENBLAS/OMP/MKL/NUMBA/RAYON各NUM_THREADS=1（RAYONはRAYON_NUM_THREADS） |
| GPU | 使用しない。前段のdriver通信失敗を保持し、利用を仮定しない |
| Memory | **4 GiB = 4,294,967,296 bytes**、RLIMIT_ASのvirtual address space |
| Result JSON | **256 MiB = 268,435,456 bytes**、newline含むserialized result上限 |
| Log | **256 MiB**。Python/native stdout・stderrを同じrunner.logへFD単位で保存 |
| Context | **4 MiB**以内。科学moduleの初期化前にexecution_context.jsonを保存 |
| Pauli table | 最大**65,536**。非identity tailは最大65,535 |
| Global contribution cap | **1,000,000**、3分割・準備処理を通算する_combine寄与数 |
| Scientific wall-time | **上限なし**。tool polling timeoutと別 |
| Output directory | `artifacts/track_b_m1s_gate0_result/2026-10-16/v2_once`候補。未作成、採択・別実施承認後にだけ排他的に作成 |

Version/build file pinは主要な算術依存の同一性を強めるが、OS/libc/CPU命令経路も含めた完全な再現性certificateではない。実環境のavailabilityと実効上限はlaunch時に保存する。3.11.0rc1のまま進めるか別のPython版を準備するかも候補の受理事項で、無断upgradeはしない。

## 2. 計数上限の根拠：科学データを評価しない静的見積り

8 modeのone-body JW supportは、identityとZが最大9、異なる2端点のXX/XY/YX/YYが`4*binom(8,2)=112`で、計121以下。各bilinearは最大4寄与。A_l²のmap積は121²=14,641寄与以下で、12 blockを一度だけ作成する。3分割のD/R集約は各回、base121＋全12 blockの寄与を上から数えればよい。

SOURCEの全_combine寄与数の静的包絡は

`64*4 + 13*64*4 + 122 + 12*121² + 3*(121+12*121²) = 706,837`。

これはPauli文字列や係数を生成して数えた値ではなく、loopの上限と支持数からの整数計数である。1,000,000 capには293,163寄与の余裕があり、今回の固定SOURCE・rank/splitsを正常に処理する包絡を超えている。SOURCEのloopを増やしたらこの根拠を更新する。native BLAS命令数のcapではなく、展開・集約の全寄与数のcapである。

## 3. Memory・保存量の根拠

256×256 complex128行列は1 MiB。12 block参照を共有し、basis/bilinear/block mapを再利用する。全Pauliのdense行列はcacheしない。flip/phase cacheはそれぞれ最大256 mask、各256成分に限られる。matrix部分は数十MiB程度の構造であり、Python辞書・list・JSONが主要な追加消費になる。

最悪の3×65,535 eventについて、tableとexpectation gateの各ASCII JSON recordに640 bytesずつを割り当てる静的包絡なら約240 MiB、残りの固定gate/configはその差分余裕を使える。256 MiB result capを候補とする。これは実保存サイズの測定ではなく、現在の固定key/indent/word・有限binary64の表記長に基づく設計見積りである。形式変更時は再評価する。

LogはPythonの高水準writeだけでなくnative FDも捕捉するため、SOURCE v2ではresultと同じRLIMIT_FSIZEを適用する。OSの同一processにおけるper-file上限という仕組みに合わせ、設定検査でlog/result capの同値を要求する。logは256 MiB、resultは256 MiB、contextは4 MiB、SHAは65 bytesで、合計**516 MiB＋65 bytes**以内を設計上限とする。directory metadata等は別。巨大ログを成功と扱わない。

4 GiB address-space capは小行列だけでなく、map集約のPython object、全row/gate、JSON str/UTF-8 bytesの一時的な重複、NumPy/BLASのmappingの余裕を含む技術候補である。RAM使用量とvirtual address spaceは異なる。確認時は32 logical CPU、host RAM約67.1 GB、available約54.7 GBだったが、これは予約ではない。current cgroupの別の上限や他jobの使用があれば、launch前に再確認する。benchmarkによる十分性・speedupは未確認。

RLIMIT_ASとRLIMIT_FSIZEの意味は[Python resource公式仕様](https://docs.python.org/3.11/library/resource.html)に従う。既存soft/hard limitが小さい場合はより小さい上限を維持し、実効値を結果に記録する。制限不足をthreshold緩和や自動retryで解消しない。

## 4. SOURCE・契約・承認の保護

v2固定SOURCE：`94e6089eba37b022a79eb262a2aa5025f5e8ddcd`。旧v1 SOURCE `4bd39ca6...`とその結果・承認・markerは不変。新module `m1s_gate0_v2`と専用runnerを追加し、既存runnerには接続しない。

将来のexecuteは、SOURCE固定commit→採択済み最終契約commit→**別の明示科学実施承認commit**の順序とhashを照合する。v2はscope採択記録のkind/hash/条件を別にbindingし、それだけではpermitを発行しない。scope記録は科学実行authorizationではない。旧PR2/他Trackのauthorizationだけでなく旧Gate0 v1 schemaもv2 guardで受け付けない。新authorization/markerは今回作成しない。

デフォルトはstatus表示のみ。help/schema/validate-configはstdlibのみでNPZを開かず、scientific moduleをimportしない。候補JSONは9個の精度値と最終契約adoptionが未解決なのでexecution_ready=false。metadata/build hashを照合した後でのみNumPyをimportできる設計である。

## 5. 失敗の意味と保存

| 種別 | 保存・停止方針 |
|---|---|
| 環境／SOURCE／契約／入力hash不一致 | guard段階のtechnical拒否。科学処理を起動しない |
| 再構成／basis／Hermiticity／phase不一致 | globalまたはsplitのtechnical STOP。条件を変更しない |
| 精度要求・state残差不適格 | technical gateの記録。研究全体の勝敗ではない |
| λ/W未定義、小分母、ρ精度不足 | NA/UNDETERMINED/PRECISION_UNRESOLVEDを区別 |
| Domain未認証 | oracle labelと未認証欄を保存。追加spectrumへ自動移行しない |
| Memory／log／output上限 | MemoryError等を可能な範囲で保存。SIGKILL/SIGXFSZやserialization時の不足では結果を保存できない場合がある |

科学処理前のcontextとnative logは、完全なresultを保存できない場合の証拠であり、科学的な成功の代用ではない。出力dirが存在すれば起動を拒否する。source/contract/geometry/split/state/toleranceを変えての自動retry、旧結果上書き、空directory削除による暗黙再試行は認めない。失敗後の取扱いは保存証拠を踏まえて別途判断する。

## 6. 未実施と停止

今回の追加検査はSOURCE/設定/拒否経路の静的確認のみで、旧30/79件を一括再実行していない。memory/log FD captureや全数値処理のruntime correctnessは未検証である。最終契約の採択と別実施承認が揃った最初の数値gateへ明示的に残す。**新科学計算・benchmark・数値correctness・新marker0、mandatory STOP。**

[RUNTIME]: ../../../artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/runtime_candidate.json
[PREC]: m1s_gate0_precision_error_budget_design_20261016.md
