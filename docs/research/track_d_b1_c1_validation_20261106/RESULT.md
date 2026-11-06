# Track D B1/C1：元Hへのoperator誤差接続とq=2 correctness

2026-11-06 JST。**B1は保存13 roundで元H向けoperator十分条件が成立し、C1は固定q=2 fixtureのraw correctnessがPASSした。** 実行SOURCE `8c6eeb02e89f6cf14870789719493e75482f62f8`、基点 `5f94972adf5a10e96e9400d3919c00d14e7a718a`。[実行前固定PLAN](PLAN.md)に従う。一回の実行を完了し、`TECHNICAL_PASS_MANDATORY_STOP`。新sampling・ground solve・DF refit・compile・N算定・価格評価・retry/resumeは0。全law実装・全round測定・全RPE費用の保証ではない。

## 固定条件と再利用

linear H4・隣接1.00 Å・STO-3G・Nα=Nβ=2、number-CDF8・元frame順LD2・ordered S2/s1/r32/K0、h=δ=.1。初期状態は元Hの厳密groundを仮定し、準備費を除外する。B1は同じsector36の保存13-round m0..12、C1はsample00→01を連結したq=2の一fixtureだけを対象とした。

| 役割 | identity |
|---|---|
| 新科学実行SOURCE | `8c6eeb02e89f6cf14870789719493e75482f62f8`。人工回帰3件PASS後、実行前にcommit・公開し全9 fileのbytesを照合 |
| Stage A SOURCE／結果 | `2dd012dc77dcc877d5918c7d212819eb79b97557`／`3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0` |
| 再利用元C SOURCE | `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。143 SOURCE・16保存入力・追加lawの固定hashを照合 |
| C r32保存評価の科学SOURCE | `4df8eca29c59ffd55e2dd898f05da16bb246ddbe`。数値欄の生成元。carrier／今回の実行SOURCEと区別 |
| 保存carrier | `3649b75af191916daf44abb91e14d34c43b83e8f`。保存events／CAR／ordinary QPY計96 bindingとB1入力4件をbyte照合 |
| 元h/g/scalar packet SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| J/U endpoint SHA256 | `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a` |
| Cohort | `near_practical_PR_r32_fixed32_v1`。固定sample ID先頭2件の昇順という結果非依存規則。母平均標本としての選択ではない |

[新module](../../../src/trottertracks/track_d_b1_c1.py)／[明示runner](../../../scripts/run_track_d_b1_c1.py)は、既存Stage A preflight／saved_record／path_phase／control_one／reference_gadget／run_prechecks、既存accuracyの値検査、既存CPU Aer StateActionsを再利用した。旧関数やown-DF専用guardを変更せず、original-H用の小さい誤差接続と保存shellの取り外しだけを追加した。[人工回帰](../../../artifacts/track_d_b1_c1_validation/2026-11-06/technical/synthetic_tests.txt)は3件PASS。人工回帰は分子入力・sampling・compileを使っていない。

## B1：state-vector不要の元H向け十分条件

保存DF sector normと元engineering guardの和は `D_H=8.060251994885292e−9 Ha`。同じHermitian HとH_CDF、共通不変sectorに対する標準Duhamel評価

\[
\|e^{-itH_{CDF}}-e^{-itH}\|\leq\min(2,tD_H)
\]

を保存PF／normalization-corrected RTE operator誤差へ加えた。新しいmatrix normやexponentialは計算していない。

\[
\epsilon_m=\epsilon_{PF,m}+\epsilon_{RTE,m}+\min(2,t_mD_H),\qquad
\Delta_m=A_m\epsilon_m.
\]

exact original-H ground上のreference `A_m exp(-it_mE0)`に対して、`rho_lb=A_m-Delta_m`、`beta_sys<=asin(epsilon_m)`（epsilon<1）を評価した。これはpartial PFの半径を無条件に1と仮定した評価ではなく、exact-H信号までのoperator距離を使う条件付き評価である。DF差はこの誤差へ一回だけ入れ、旧own-DF energyへのWeyl加算を重ねていない。保存own-DF signal・N・ground vector、保存physical vectorは使用していない。

**全13 roundで正半径と、旧phase配分との比較による十分条件が成立した。** 旧βtotal=π/6、βstat=π/12は履歴診断だけであり、新目標を採択していない。energy精度・成功確率、N、decoder成功率は新しく設定・算定していない。

| 項目 | m0／q1 | m12／q4096 |
|---|---:|---:|
| t | .1 | 409.6 |
| DF evolution error | 8.06025199489e−10 | 3.30147921711e−6 |
| original-H corrected operator error ε | 7.81377797167e−5 | .120627072398 |
| raw signal error Δ | 7.81347269828e−5 | .102788719884 |
| observed radius lower | .999882796671 | .749331105632 |
| deterministic phase upper / rad | 7.81377797962e−5 | .120921543224 |
| 旧βtotal−βsys−βstat / rad | .261721250019 | .140877844576 |

全13行は[B1/result.json](../../../artifacts/track_d_b1_c1_validation/2026-11-06/run01/B1/result.json)。最小の旧phase slackは `.14087784457561353 rad`。旧phase配分を維持する場合の追加raw implementation error総和の余裕は最小 `.11775611966884127`で、これは `A[sin(βtotal−βstat)−ε]` の診断値である。合成・law誤差にこの値を勝手に割り当てたり、未知誤差がこの値以下だと認定したりしていない。

**資格と不足：** 保存binary64 sector operator norm＋engineering guardを有効なpremiseとした、ideal finite-RTE IID／exact nativeの数値的十分条件。厳密な丸め誤差包囲ではない。implemented joint-law／fresh-IID bridge、全law compiled／synthesis・Reference補正誤差、選定した目標でのoriginal-H alias／horizonとfailure配分は未資格である。従ってactual RPE successや同じNを利用できる保証は出していない。

この範囲では、元H向けの正半径・phase十分条件を得るためにB2のstate-vector数値検証へ進む必要は生じなかった。ただしB1の成立は、未確認の実装誤差や全RPE成功率を補うものではない。B2は実施していない。

## C1：保存2 stepのwhole-path Reference

固定append順はU00→U01、V=U01 U00、B=U00、A=U01。cutは最初のcomplete h-step終了後であり、全64 events、head／逆順head、scalar／identity各2回の会計を保った。保存ordinaryのcontrol=1制限からA/Bを作り、保存独立CAR01 @ CAR00をtargetとした。strong symmetricはStage A保存raw X wrapperのH/core/H/measureからcoreだけを取り出して連結した。各stepの既存relative Gaussian・basis共有・融合・global phaseを保持し、対照を弱くした価格比較は行っていない。

whole Vを一回の既存Reference gadgetで扱い、χは両pathの既知χのfsum、A†は実際のA.inverse()とした。χ00=−.2240560438590352、χ01=−.23343186579495756、全χ=**−.4574879096539928 rad**。短いReference gadgetを二回並べる方式は使用していない。

既存22項目と追加paired X/YがすべてPASS。saved ordinary／独立CAR、AB=V、actual A†／inverse、unitarity、両halfの真空rayとχ、direct両branch、Reference complex coherenceとX/Y observableを確認した。

| C1の保存checkからの値 | 観測値 |
|---|---:|
| 既存22項目の最大成分差 | 2.04281036531e−14 |
| 既存22項目の最大数値Frobenius差 | 2.05581519614e−13 |
| paired X observable：最大成分／数値Frobenius差 | 1.36557432348e−14／1.12141250513e−13 |
| paired Y observable：最大成分／数値Frobenius差 | 6.63358404151e−15／8.24661116333e−14 |
| reference-lane leakage：最大成分／入力列L2 | 0／0（数値観測） |
| χを省いた負対照のX／Y最大成分差 | .272274301896／.453225029597 |

Reference／ancilla=0で**全256 system入力列・各131072出力振幅**を既存CPU Aer double state-actionで評価した。17-qubit全Operatorは作っていない。directは全512 operator、Referenceは入力isometry／complex coherenceから誘導したX/Yを照合した。元H ground一点の検査に依存せず、任意system入力に対応する数値observable検査である。任意reference初期状態や任意のgadget反復channelの保証には拡張しない。

complex128/binary64、component／unitarity abs1e−10、signal／observable数値Frobenius2e−10、rtol0、global phase fitなしを維持した。閾値緩和はない。これはlocal numerical evidenceであり、丸め誤差の厳密包囲ではない。cost compilerは呼んでおらず、compiled回路のcorrectnessは未検証である。state-actionは既存の小gate行列変換を使い、full-wrapperのcompile・最適化・価格付けをしていない。

## 保存物・実行記録

[SOURCE.json](../../../artifacts/track_d_b1_c1_validation/2026-11-06/run01/SOURCE.json)に元143 SOURCE・16入力・96保存binding、追加B1入力、Stage A archiveとmember identity、今回の固定SOURCE／再利用部品、Python3.12.3／固定10 package pinsを記録した。実行前後のSOURCE／入力は一致した。

- [B1/result.json](../../../artifacts/track_d_b1_c1_validation/2026-11-06/run01/B1/result.json)：全13 row、履歴診断、資格・不足。
- [C1/result.json](../../../artifacts/track_d_b1_c1_validation/2026-11-06/run01/C1/result.json)：fixture条件、全check、paired observable、fusion来歴。
- [raw回路・referenceのlossless archive索引](../../../artifacts/track_d_b1_c1_validation/2026-11-06/run01/payload_archive.json)：10 QPY（A/B/actual A†、ordinary q2、2 core、2方式×X/Y wrapper）とreference／observable NPZ1件。archiveは2,659,624 bytes。run01で展開すると元pathを復元し、全memberのbytes／SHA256は元inventoryに一致する。
- [run result](../../../artifacts/track_d_b1_c1_validation/2026-11-06/run01/result.json)／[STOP](../../../artifacts/track_d_b1_c1_validation/2026-11-06/run01/STOP.json)／[元17出力のinventory](../../../artifacts/track_d_b1_c1_validation/2026-11-06/run01/inventory.json)。cacheや内部環境pathは公開しない。

[実資源記録](../../../artifacts/track_d_b1_c1_validation/2026-11-06/run01/resources.json)：affinity64 CPU、開始時idle観測7.0295 CPU相当、MemAvailable=497042550784 bytes。独立2 workerのうちB1は短いscalar処理、C1は実空きCPUに基づく7 threads、小行列BLAS1、大きいGramも7 threadsを使用した。CPU/RAM/worker/wallの人工cap、共有venv／OS／他jobの変更は0。wall=239.441919秒、B1=4.497482秒、C1=235.973623秒。C1 worker peak RSS=4363288 KiB。aggregate peakや並列speedupは測定していない。

実行入口は次の一回のみ（変数はtask専用の既存venv／復元入力を表し、再実行指示ではない）。

```bash
"$TRACK_D_PYTHON" -B scripts/run_track_d_b1_c1.py \
  --origin "$TRACK_D_ORIGIN" --saved "$TRACK_D_SAVED" --execute
```

旧SOURCE・関数・入力・raw／result・grant／ledger／marker／STOPは保全した。新sampling・ground solve・DF refit・量子Born shots・N算定・compile・価格比較・全round費用評価・retry/resumeはいずれも0。Aer snapshot内部のshots=1は量子shot評価ではない。

## 成立範囲とB2/C2の必要性

| 論点 | 今回の到達点／残ること |
|---|---|
| B1→元H | 保存norm／guardを前提として、元H厳密ground向けの正半径とphase余裕をstate-freeで接続できた。own-DFの信号・Nは転用していない |
| C1→whole-path | 固定sample00→01・q2・明示cutの一回gadgetについて、raw X/Yの数値的な同値性を確認。別pair／cut／qや全law・compile後へは拡張しない |
| B1とC1の結合 | B1はideal lawの13-round誤差条件、C1は一つのq2実装fixture。これらだけで全roundの実装測定分布・同じN・成功率を保証しない |
| B2 | 今回のoperator十分条件は成立したため、正半径・phase接続のための自動fallbackは不要だった。追加の元H signal／decoder統計が必要かはユーザー／GPTが判断する |
| C2 | compile後correctness・q依存価格を得るには別のC2実行が必要。まず今回の固定fixtureに限定したP0 compile等を検討できるが、対象・精度・費用scopeの選択と明示許可が先。未実施 |
| 全RPE費用 | 主資源指標、共通energy精度・成功確率、全law誤差／期待価格／long-q coverage、routing・合成等が未結合。Stage AのCX／depth減を全費用優位としない |

## 終了・次の担当

専用branch`codex/track-d-b1-c1-validation-20261106`へSOURCEと成果物をcommit/pushし、remote SHA・公開Git tree/blob／bytesを照合して終了する。

**mandatory STOP。B2／C2、別fixture／cut／q、新sampling、compile、価格比較、全round費用、ground solve、DF refit、詳細科学レビューへ進まない。次担当はユーザー／GPT**：条件付きB1とC1の結果を踏まえ、次の科学範囲と必要な明示許可を判断する。技術PASSを方式の採択・科学的GOとは呼ばない。
