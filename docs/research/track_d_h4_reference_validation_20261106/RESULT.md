# Track D：実H4・保存word02のReference移送結果

2026-11-06 JST。**固定scopeのcorrectnessと6 full-wrapper compile後照合がすべてPASS。Referenceは強いsymmetric-directional対照よりCX/depthが小さく、RZ/qubitsが増えた。** 新しい科学実行は完了し、`TECHNICAL_PASS_MANDATORY_STOP`。方式全体の採択・全lawの優位は認定しない。

## 固定条件・SOURCE

[PLAN](PLAN.md)／[元の静的調査](../track_d_real_pr_transfer_static_20261106/REPORT.md) §7に従った。対象はlinear H4・隣接1.00 Å・STO-3G・Nα=Nβ=2、number-CDF8、元順LD2、partial ordered S2/s1/r32/K0、h=δ=0.1の**1 h-stepだけ**。保存word02 `alternating_frames` の32eventを使用し、event16直後の固定cutからV=ABを構成した。別word/cut、再sampling、DF再fit、ground solve、精度変更、retry/resumeは0。

| 来歴 | 固定版・保存物 |
|---|---|
| 新実装・科学実行SOURCE | `b21b5faff1fcbcabbb7f674c4b6d8818b8bb08ea`。実行前にcommit固定。新adapter・runner・check harness・人工回帰・PLANを含む |
| 元C科学SOURCE | `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。別checkoutから優先import。元manifest143 file・保存入力16件を実行前後にhash照合し一致 |
| 元C固定結果 | `3649b75af191916daf44abb91e14d34c43b83e8f`。保存word02、独立full256 CAR reference、raw QPYを再利用。旧runner・旧4-word科学実行・grant・STOPは再利用しない |
| 元h/g/scalar | packet SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。元gからのcorrected κ、nuclear scalar一回を維持 |
| 固定CDF endpoint | J/U SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。保存independent referenceのJ/Uとexact一致 |
| 完全な入力・部品identity | [SOURCE.json](../../../artifacts/track_d_h4_reference_validation/2026-11-06/run01/SOURCE.json)に元commit/path/hash、143 source・16入力、新SOURCE/再利用部品、package版、保存word/QPY/referenceのbytes bindingを記録 |

初期systemは**元H厳密基底状態を仮定**し、状態準備費を除外した。保存own-DF numerical groundは検証入力に使わず、新しいground vectorも計算していない。以下の測定演算子検査は全256 system入力basis列を使うので、この初期状態仮定に依存する一点のstate検査ではない。

## 再利用と強い対照

ordinaryは元 `prepare_arm` / `build_kernel` / StrongBuilder(relative_fusion=True)を使用し、保存rawと位相を含めて一致した。Gaussian basisは元から非制御で、中央だけを制御する。

symmetric-directionalは、前半headをuncontrolled、後半逆順headの中央を既存AX2B `append_directional_diagonal` とし、中央32-event tailはordinary-controlledのまま保持した。元と同じrun grouping・diagonal fsum・解析位相補正付きrelative Gaussianを使った。head境界融合を外した弱い回路との比較ではない。元pipelineのcross-head/tail relative fusion=false、tailのfull-frame provider・basis共有規則を全方式で維持した。

ReferenceのA/Bは、元のdiagonal-only制御回路をancilla=1へgate単位で制限して構成した。既存base gate・global phaseを保持し、dense gateへ置換していない。人工Track Dのwrapper/metrics、2層CSWAP recipe、C系列の位相保持compileを再利用・8-modeへ限定拡張した。A†は実際のA.inverse()であり、単なる角度反転ではない。

保存metadataから、scalar=2.29310124732、extracted identity=0.041217410629575235を一度だけ会計した。signed event rotation・event.phaseを含むpath位相は **χ=−0.23343186579495756 rad**。normalization/attenuationをunitaryやobservableへ乗算していない。[conditions](../../../artifacts/track_d_h4_reference_validation/2026-11-06/run01/conditions.json)／[relative fusion記録](../../../artifacts/track_d_h4_reference_validation/2026-11-06/run01/fusion.json)を保存した。

## correctness

実行前の[人工回帰7件](../../../artifacts/track_d_h4_reference_validation/2026-11-06/technical/synthetic_tests.txt)はPASS。scalar/actual inverse、未知ancilla操作拒否、relative融合付きdirectional同値性、既存2-mode gadgetとの一致、complex observable、Reference register配置を確認した。人工回帰で分子入力・sampling・compileは実行していない。

実H4の[precompile検査](../../../artifacts/track_d_h4_reference_validation/2026-11-06/run01/precompile_checks.json)は**27項目PASS、最大成分差2.08722e−14**。保存CAR reference/保存raw/ordinary/symmetric、両control branch、A B=V、actual A†、unitarity、真空rayとχ、X/Y測定演算子を照合した。χを省いた負対照はX/Yでそれぞれ最大成分差0.0827683/0.232888を検出した。

Referenceは既存 `c1_h6_state_action.StateActions` を使い、全256 system basis入力から**各131072出力振幅**を計算した。raw coreのbranch出力・入力isometry・complex coherenceからX/Y observableを構成し、compiled X/Y wrapperもそれぞれ全256入力列でrawに照合した。初期reference/ancillaは0で固定する。**17-qubit全Operatorは生成していない。** direct対照は全512 operatorでcompile前後を比較した。

6 wrapperすべてでpostcompile PASS。全postcheckの最大成分差9.59659e−13、CARに対するobservableの最大成分差3.09753e−13・最大数値Frobenius差4.79218e−12。compiled Referenceのreference-lane leakageは最大成分6.43421e−15、入力列ごとの最大L2約1.079e−14だった。

精度はcomplex128/binary64、component/元unitarity abs=1e−10、signal allowance=2e−10、rtol=0、global phase fitなし。observableの数値Frobenius差も2e−10以下を確認した。これらは数値観測であり、丸め誤差の厳密包囲・17-qubit全入力operator保証・全law一様保証ではない。元研究のStage C ready=false、whole-law bound=null、actual CA=nullを変更しない。

## 6 full-wrapperの資源

全arm同一P0：Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation1、initial-zero false、topology-free。Cの `compile_checked` → MPFR160 declared-phase accumulation → F2 phase保持unrollを使用した。MPFR160はcompilerのdeclared phase蓄積であり、全数値評価を160-bitで行ったという意味ではない。

H → core → YのみSdg → H → ancilla測定を含み、状態準備を除外した。**X/Yは別々に一回compileし、下表の費用は両軸で同じだった。** 過去のkernel-only価格と今回のwrapper価格を混ぜていない。

| 方式（X/Y各1件） | RZ count | CX count | RZ depth | CX depth | total depth | qubits | size |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary＋既存共有・融合 | 6473 | 3074 | 2945 | 2530 | 6542 | 9 | 12812 |
| symmetric-directional＋既存共有・融合 | 4482 | 1794 | 1682 | 1200 | 3952 | 9 | 9273 |
| Reference | 4721 | 1734 | 858 | 555 | 1955 | 17 | 9562 |

強いsymmetric対照との差は**RZ +239、CX −60、depth −1997、qubits +8**。Referenceの前後2層・各8対CSWAP、P(χ)、固定cutと両laneの全回路費を含む。CXの削減幅はordinaryを対照にした差より小さい。depthは減る一方RZ/qubitは増えるため、主資源指標未固定のまま単一winnerを宣言しない。RZからT合成価格を推定していない。routing、reset/維持費、全law期待価格、shots/N、全RPE総資源は未評価。

## 実行・保全・公開

一回のwallは329.223秒。parent CPUは265.605秒（child合算ではない）、parent peak RSSは4007668 KiB、Reference postcheck workerの最大RSSは5750960 KiB。aggregate peakは測定していない。32 CPU・空きRAM約49 GiBを確認し、6 compile processを並列化。postcheckは4 direct task各1 threadと2 Reference task各14 threads、raw Referenceは32 Aer threads、大きいGram contractionも並列化した。small matrixはBLAS1。GPU driver利用不可のため既存CPU Aer doubleを使用し、CPU/RAM/worker/wall人工cap、共有venv/OS/cache/他job変更は0。実測対照のないspeedupは主張しない。

出力は[run01](../../../artifacts/track_d_h4_reference_validation/2026-11-06/run01)の42ファイル・14,618,767 bytes（task-local Matplotlib cacheを除く）。raw/compiled wrapper12 QPYとA/B/actual A†の3 QPY、保存CAR reference/生observable、compiled observable6件、各compile/check記録、SOURCE/conditions/保全確認/result/STOP/inventoryを保存した。compileは6件、sampling/quantum shots/ground solve/DF fit/retry/resumeは0。Aerのsnapshot plumbingのshots=1を量子shot実行と呼ばない。

[result.json](../../../artifacts/track_d_h4_reference_validation/2026-11-06/run01/result.json)／[STOP.json](../../../artifacts/track_d_h4_reference_validation/2026-11-06/run01/STOP.json)／[inventory](../../../artifacts/track_d_h4_reference_validation/2026-11-06/run01/inventory.json)。元143 SOURCE・16保存入力と回収した旧結果bytesは実行後も一致し、元SOURCE treeに変更なし。旧grant/STOPは変更・消費していない。

実際の実行コマンド（作業directoryは本branchの隔離checkout）：

```bash
PYTHONPATH=/tmp/prt-track-d-h4-deps-20261106 \
  "/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" -B \
  scripts/run_track_d_h4_reference.py \
  --origin /tmp/prt-track-d-h4-origin-source-20261106 \
  --saved /tmp/prt-track-d-h4-saved-result-20261106 \
  --output artifacts/track_d_h4_reference_validation/2026-11-06/run01
```

`--origin` は元科学SOURCEのcheckout、`--saved` は元結果commitからSOURCE.jsonの `saved_word_bindings` に従って回収した8件と、そのbindingを保存した `identity.json`。元ファイルを再生成せずGit bytesを使用した。

専用branchは `codex/track-d-h4-reference-validation-20261106`。SOURCE commitと本結果をcommit/pushし、remote SHAと全変更ファイルの公開bytes/Git blobを照合して終了する。

**mandatory STOP。次担当はユーザー／GPT**：固定wordでのCX/depth削減とRZ/qubit増加を踏まえ、主資源指標と継続の要否を判断する。Codexはその判断と新たな明示許可なしに、別word/cut、全law/期待価格、shots、全RPE、重要科学レビューへ進まない。
