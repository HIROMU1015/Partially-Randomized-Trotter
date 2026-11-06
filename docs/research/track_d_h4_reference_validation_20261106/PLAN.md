# Track D：実H4・固定word02の限定移送検証

2026-11-06 JST。ユーザーの明示指示により、[固定静的REPORT](../track_d_real_pr_transfer_static_20261106/REPORT.md) §7の実装・数値correctness・PASS時6 wrapper compileだけを実行する。承認証明や新grantは作らず、既存SOURCE・入力・結果・grant・STOPを保持する。

対象はlinear H4/1.00 Å/STO-3G/Nα=Nβ=2、number-CDF8、元順LD2、partial ordered S2/s1/r32/K0、h=δ=0.1。保存word02 `alternating_frames` の32eventをそのまま使用し、event16の直後で切る。Bは前半head＋最初16event、Aは後16event＋逆順head＋scalar/identity一回。独立draw、別word/cut、DF再fit、ground solveを行わない。初期状態は元H厳密基底状態を仮定し、準備費を除外する。保存own-DF stateを使わず、任意system入力の測定演算子を数値照合する。

## 固定SOURCE・保存入力

- 調査：`a9a74b1386ba713ed3648658dd94be4d992a492e`。
- 元科学SOURCE：`ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。元manifest143 file/保存入力16件を、隔離checkoutで元hashに照合する。元runner/4word実行・grant・出力を再起動しない。
- 元結果：`3649b75af191916daf44abb91e14d34c43b83e8f` のword02 events、full256 CAR reference、raw QPY等をGit bytesで回収する。
- 元h/g/scalar packet SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、J/U endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
- 元SOURCEは別のread-only利用treeから優先importし、新adapterと人工Track D shell/metrics・AX2B directional helperだけを新branchから利用する。新SOURCEは実行前にcommit固定し、実際の混合来歴をSOURCE.jsonへ保存する。

## 既存機能の再利用・限定変更

ordinaryは元 `prepare_arm` / `build_kernel` / StrongBuilder(relative_fusion=True)を無変更で使用。元raw回路とfull CAR referenceの位相込み一致を必須とする。symmetric-directionalは同じrun grouping、diagonal fsum、relative Gaussianと解析位相補正を保持し、後半headの中央だけ既存 `append_directional_diagonal` へ接続する。tailは元ordinary-controlのまま。

Referenceのuncontrolled A/Bは、既存diagonal-only制御回路のancilla=1制限を既存gate/base_gateで構成する。未知ancilla操作は拒否し、dense gateへ置換しない。2-mode adapterの2 CSWAP層を8-modeへ拡張し、actual A.inverse()とP(χ)を使う。head内部の既存融合・tail basis共有を全方式で保持する。元pipelineのcross-head/tail relative fusion=falseを変えない。

χは元scalar＋extracted identity、保存event.phase、signed rotation角から取得し、物理Hの真空energyだけで代用しない。B/attenuationをunitaryへ適用しない。

## correctnessと比較

1. 正確な保存word・sign/basis/角度・J/Uとidentity/normalizationを束縛し、保存raw、ordinary、symmetricをdiag(I,V)へ全512成分行列で照合する。A B=V、actual A†、unitarity、真空rayと解析χを検査する。
2. Referenceは既存 `c1_h6_state_action.StateActions` のCPU Aer doubleを使う。256個の全system basis入力、reference/ancilla=0から、17-qubit全出力振幅を評価する。17-qubit全Operatorは作らない。raw coreのisometry/coherenceからX/Y測定演算子を構成し、χ欠落の負対照も調べる。
3. 全precompile correctness PASS後だけ3方式×X/Yの6 full-wrapperを各一回compileする。既存 `compile_checked` → MPFR160 declared-phase → F2 phase保持unroll/P0経路を全armへ適用する。H、YのみSdg、H、ancilla測定を含む。state preparationは含めない。
4. compiled directは全512 operator、compiled Referenceは全256入力列のstate-action/isometryとX/Y observableをraw/CARに照合する。global phase fitはしない。Referenceの全17-qubit入力に対するoperator保証とは呼ばない。

binary64、component abs/元unitarity abs=1e-10、complex signal allowance=2e-10、rtol=0。測定演算子差の数値Frobeniusも2e-10以下とする。厳密丸め包囲・全law保証には昇格しない。P0：Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation1、initial-zero false、topology-free。Gaussianは `nature_jw_exact_zero_lapack_atan2_v1`。

RZ count/depth、CX count/depth、total depth、qubitsを別々に報告し、主指標を固定しない。direct9、Reference17 qubits、前後の各8 CSWAPとP(χ)、分割による融合損失を費用に含める。旧kernel価格を新full-wrapper価格と直接比較しない。

## 実行・STOP

実CPU/RAM/GPUを確認する。利用可能32 CPU、RAM約62.5 GiB/空き約49 GiB。GPU driver利用不可なので既存Aer CPUを使用する。6独立compileをprocess並列化し、postcheckは4つの小direct行列と2つのReference state-actionへCPUを配分する。小行列は既存timingのBLAS1方針、大きいGram contractionは利用可能threadsを使う。CPU/RAM/worker/wall人工capを設けず、共有環境・他job・共有venvを変更しない。threadpoolctl3.6.0の不足分だけをtask-local `/tmp` dependencyへ取得した。quantum shotsは0で、Aerのsnapshot plumbingにshots=1を使うこととは区別する。

fresh output一回、retry/resume0。SOURCE/入力/word/phase不一致、欠損、公平なsymmetric構築不能、correctness/compile後検査FAIL、共有環境干渉で新callを停止し、開始済み処理だけ回収して証拠を保持する。別word/cut、再sampling、再fit、精度緩和、fallback・無断retryへ進まない。6 compile以外の分子compile、全law/shot/N/全RPE総費用、新研究方向・重要科学レビューは対象外。完了/FAILどちらも公開・remote SHA/bytes照合後mandatory STOPし、ユーザー/GPTへ研究判断を戻す。
