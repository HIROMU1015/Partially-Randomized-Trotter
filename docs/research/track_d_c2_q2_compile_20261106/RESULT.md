# Track D C2：固定q=2 whole-pathのcompile検証

2026-11-06 JST。**固定q=2の4 wrapperはcompile・postcheckすべてPASS。Referenceは強いsymmetric-directionalよりCXを4100→3860、total depthを6096→3014へ削減し、RZは10773→11114、qubitsは9→17へ増加した。** 実行SOURCE `e0befa619a14b956ad64349a86ec2f261962883c`。[実行前固定PLAN](PLAN.md)に従い、C1の同一raw bytesを使った。retry/resume0、`TECHNICAL_PASS_MANDATORY_STOP`。この一fixtureの結果を全law・長q・全RPE費用へ外挿しない。

## 固定条件とSOURCE

linear H4・隣接1.00 Å・STO-3G・Nα=Nβ=2、number-CDF8・元frame順LD2・ordered S2/s1/r32/K0、各h=δ=.1、q=2、総時間.2。sample00→01の全64 eventsと両stepのhead／逆順headを保持した。V=U01 U00、B=U00、A=U01、cutは最初のcomplete h-step終了後。χ=−.4574879096539928 rad、scalar／identity各2回。元H厳密groundを仮定し、状態準備費を除外する。ground vectorは使用していない。

| 役割 | 固定identity |
|---|---|
| C2実行SOURCE | `e0befa619a14b956ad64349a86ec2f261962883c`。人工回帰2件PASS後にcommit・公開、全8 fileのbytesを照合してから実行 |
| [C1結果](../track_d_b1_c1_validation_20261106/RESULT.md)／科学SOURCE | `d3913866e6ade4ab97ac02645fed91b2d5fcfa75`／`8c6eeb02e89f6cf14870789719493e75482f62f8` |
| [Stage A結果](../track_d_h4_paired32_reference_20261106/RESULT.md)／科学SOURCE | `3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0`／`2dd012dc77dcc877d5918c7d212819eb79b97557` |
| 元C SOURCE／保存carrier | `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`／`3649b75af191916daf44abb91e14d34c43b83e8f`。役割を混同しない |
| 元h/g/scalar packet SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| J/U endpoint SHA256 | `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a` |
| C1 input archive SHA256 | `2cef1b138af95b75f50d80ee89fdc7bb7aee13d133fa05b38d91a82f80ec01d9`。member11件を同一bytesで復元 |

元SOURCE143・保存入力16・保存cohort96 binding・追加law・固定10 package pins等を既存preflightで照合した。C1 metadata／raw archive、q1比較用のStage A保存CSVも固定commitと照合した。旧SOURCE・入力・結果・grant・ledger・marker・STOPは保全し、今回の新規C2許可と旧STOPを区別する。

## 再利用・公平性

追加は[明示runner](../../../scripts/run_track_d_c2_q2.py)と[人工回帰](../../../tests/check_track_d_c2_q2.py)。旧Reference adapter、位相保持compiler、correctness関数は変更していない。C1 raw QPY10件とreference／observable NPZを復元し、回路の再生成・cut変更・新head構成は行っていない。

strong symmetric-directionalは保存core00→01のまま、各stepのGaussian非制御・directional center・head relative fusion・tail basis共有・global phaseを保持した。Referenceも同じC1の一回gadgetであり、短いgadgetを二回並べる方式ではない。両方式へ同P0を適用し、そのcompilerが可能な最適化を許した。新しいcross-step手動最適化・別baseline構成・別cut探索は追加していない。

P0はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation1／initial-zero false／topology-free。既存`compile_one`→`compile_checked`→MPFR160 declared-phase accumulation→位相保持unrollを再利用した。MPFR160はcompiler phase蓄積の精度であり、全数値照合やRZのT合成精度ではない。状態準備なしのH→core→YのみSdg→H→ancilla測定を含むfull wrapperを、2方式×X/Yで各一回compileした。

## Correctness

C1の22項目＋paired X/Y PASSは、同じraw／reference bytesの資格として引き継いだ。追加のprecompile small-operator検査ではAB=V、actual A†とinverse、vacuum ray／χ、cached A/B/adjoint、強いsymmetricの両branch=diag(I,V)を照合した。PASS後だけ4 wrapperをcompileした。

postcheckは既存`verify_one`／`check_compiled`で、directの全512 operatorと、Reference／ancilla=0の全256 system入力列・各131072出力振幅を検査する。Referenceは保存raw isometryのbranch作用（actual A†とχを含む）へ照合し、保存CAR由来X/Y測定演算子とraw observableへ比較する。A†単体のcompileは追加しない。17-qubit全Operatorも作らない。

さらにcompiled方式間のpaired X/Yと、vacuum入力のX+iY=e^{iχ}を照合した。追加precompile10項目、4 wrapperのpostcheck18項目、paired X/Yと両方式のvacuum位相がすべてPASSした。

| 保存checkからの最大値 | 数値観測 |
|---|---:|
| 追加precompile10項目の最大成分差 | 1.03122226202e−14 |
| postcheck18項目の最大成分差／数値Frobenius差 | 1.84961533211e−12／2.90761156450e−11 |
| Compiled vs CAR observable：最大成分／数値Frobenius差 | 6.97331081767e−13／1.06140665307e−11 |
| paired X：最大成分／数値Frobenius差 | 7.09987624251e−13／1.08516594097e−11 |
| paired Y：最大成分／数値Frobenius差 | 3.20132810824e−13／2.63421285248e−12 |
| vacuum位相：symmetric／Referenceのcomplex差 | 8.86528900660e−15／6.87741002662e−13 |
| Reference leakage：最大成分／入力列L2 | 1.24399689650e−13／1.39538301560e−13 |

complex128/binary64、component／unitarity abs1e−10、signal／observable数値Frobenius2e−10、rtol0、global phase fitなしを維持した。閾値緩和はない。これはreference/ancilla=0での数値isometry／誘導observable検証であり、任意の17-qubit入力、丸め誤差の厳密包囲や全lawの実装誤差保証ではない。[precompile記録](../../../artifacts/track_d_c2_q2_compile/2026-11-06/run01/precompile_checks.json)／[paired・χ記録](../../../artifacts/track_d_c2_q2_compile/2026-11-06/run01/paired_observables.json)を保存した。

## 資源・q1との比較

RZ／CX count、各depth、total depth、qubits、sizeを別々に記録する。主資源指標や重みを今回選んでいない。q1比較は同じsample00／01の保存X/Y価格を使用し、新しいq1 compileや母平均再推定を行わない。

X/Yを別々に各一回compileし、以下の資源値は両軸で同じだった。各行は1 full-wrapperの価格であり、4件の平均や独立標本の母平均ではない。

| q2方式（X/Y共通） | RZ count | CX count | RZ depth | CX depth | Total depth | qubits | size |
|---|---:|---:|---:|---:|---:|---:|---:|
| symmetric-directional | 10773 | 4100 | 2624 | 1996 | 6096 | 9 | 21970 |
| Reference | 11114 | 3860 | 1357 | 860 | 3014 | 17 | 22263 |
| Reference−symmetric | +341 | −240 | −1267 | −1136 | −3082 | +8 | +293 |

q2でもCX・各depthの減少とRZ／qubit増加のtradeoffを観測した。Referenceは追加8-qubit register、前後2 CSWAP cascade（各8対）、χ補正、A†、固定cutとlane間の分配を含む実wrapper価格である。単一の総資源winnerを宣言しない。

Stage Aの保存q1個別価格を以下に転記する（同じP0、各行X/Y共通）。

| q1保存path／方式 | RZ count | CX count | RZ depth | CX depth | Total depth | qubits | size |
|---|---:|---:|---:|---:|---:|---:|---:|
| sample00 symmetric | 5709 | 2140 | 1387 | 1034 | 3207 | 9 | 11594 |
| sample00 Reference | 6129 | 2136 | 753 | 495 | 1679 | 17 | 12244 |
| sample01 symmetric | 5356 | 2040 | 1292 | 990 | 3003 | 9 | 10911 |
| sample01 Reference | 5767 | 2032 | 718 | 475 | 1597 | 17 | 11538 |

q1ではCX差がsample00で−4、01で−8、RZ差が+420／+411だった。今回のq2ではCX差−240、RZ差+341となった。qubit幅はq1/q2ともdirect9／Reference17である。qubitsへqやshotsを掛けた値をpeak幅としない。[比較JSON](../../../artifacts/track_d_c2_q2_compile/2026-11-06/run01/resource_comparison.json)に両軸のq2価格・差と、引用したq1の4 axis/path行を保全した。

q1のReferenceは各h-stepのevent16後cut、q2はcompleteなstep境界cutである。q・path長・cut・全長でのfusion機会が同時に異なり、価格差をqだけの因果効果とはしない。q1 full-wrapperの単純和やq倍をq2価格の代用にしない。

保存C1 rawを両方式同P0でcompileした範囲の結果であり、他のwhole-path構成・cut・compiler最適化に対する最適性は未評価。routing／接続制約、RZのfault-tolerant合成精度・T費用、reference registerの実機維持／reset費は含まれていない。これらを0と認定していない。全law母平均、long-q価格、同じenergy精度・成功確率でのNや全RPE総費用も未評価である。

## 保存物と実行・保全

[SOURCE.json](../../../artifacts/track_d_c2_q2_compile/2026-11-06/run01/SOURCE.json)／[conditions](../../../artifacts/track_d_c2_q2_compile/2026-11-06/run01/conditions.json)／[result](../../../artifacts/track_d_c2_q2_compile/2026-11-06/run01/result.json)／[STOP](../../../artifacts/track_d_c2_q2_compile/2026-11-06/run01/STOP.json)を保存した。各wrapperのcompile設定・時間・postcheck・observableと、C1継承記録、資源比較を同run01に置いた。

raw QPY10、compiled QPY4、C1 reference／observable NPZ1とcompiled observable NPZ4を[lossless archive](../../../artifacts/track_d_c2_q2_compile/2026-11-06/run01/circuit_payload.tar.gz)へ保全した。[member索引](../../../artifacts/track_d_c2_q2_compile/2026-11-06/run01/payload_archive.json)に全19 memberのpath／bytes／SHA256を記録し、[元37出力inventory](../../../artifacts/track_d_c2_q2_compile/2026-11-06/run01/inventory.json)との一致を確認した。archiveは6,763,615 bytes、run01で展開すると元pathを復元する。元ファイルはlocalで保持し、非公開環境path・cacheは公開しない。

[資源記録](../../../artifacts/track_d_c2_q2_compile/2026-11-06/run01/resources.json)：affinity64 CPU、開始時idle46.4692 CPU相当、MemAvailable=492381167616 bytes。4独立process、direct postcheck各1 thread、Reference2件は各22 threads、大きいGramも同threads、小行列／compilerはBLAS1。実空き資源とtask数から配分し、CPU/RAM/worker/wallの人工cap、共有設定・共有venv・他jobの変更はない。

全wall=208.251126秒。4 compileは各約6.98〜7.27秒、direct postcheckは各約39.15秒、Reference postcheckは各約175.1秒。Reference worker最大RSS記録は9519732 KiB。aggregate peak・child合算CPU・並列speedupは測定していない。Python3.12.3と固定10 pinsはSOURCEに記録した。

実行コマンド（一回だけ。変数はtask用の既存venv／復元入力を表し、再実行指示ではない）：

```bash
"$TRACK_D_PYTHON" -B scripts/run_track_d_c2_q2.py \
  --origin "$TRACK_D_ORIGIN" --saved "$TRACK_D_SAVED" --execute
```

compile4、retry/resume0。実行後もSOURCE・入力・復元rawのbytesは一致した。新sampling、ground solve、DF refit、量子Born shots、N評価、B2、別path／q、全RPE費用評価は0。Aer snapshot内部のshots=1は量子shot実験ではない。旧grant・ledger・marker・STOPを再利用・改変していない。

## 終了範囲

成果物を専用branch `codex/track-d-c2-q2-compile-20261106`へcommit/pushし、remote SHAと公開tree／blob／bytesを照合して終了する。

**mandatory STOP。B2、新sampling、別path／q、長q compile、N、全RPE総費用、追加科学検証・詳細科学レビューへ進まない。次担当はユーザー／GPT**：固定q2でのCX／depth減とRZ／qubit増、および全law・長q・実装誤差の不足を踏まえ、継続の要否と次の明示許可の範囲を判断する。技術PASSを全RPE優位や方式の最終採択とは呼ばない。
