# Track D B1/C1：実行前固定scope

2026-11-06 JST。基点`5f94972adf5a10e96e9400d3919c00d14e7a718a`の[準備REPORT](../track_d_bc_static_preparation_20261106/REPORT.md)と今回のユーザー指示に基づく。B1/C1の必要最小限の実装と科学評価だけを実行する。実行前にSOURCEをcommit固定する。旧SOURCE・入力・結果・grant・ledger・marker・STOPを保全し、新grantや承認証明は作らない。

## 固定入力と再利用

linear H4・隣接1.00 Å・STO-3G・Nα=Nβ=2、number-CDF8、元frame順LD2、ordered S2/s1/r32/K0、h=δ=.1。元H厳密groundを仮定し状態準備費除外。元H packet `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、J/U endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。

Stage A実行SOURCE`2dd012dc77dcc877d5918c7d212819eb79b97557`、結果`3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0`。元C SOURCE`ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`の143 SOURCE・16保存入力、追加law、carrier`3649b75af191916daf44abb91e14d34c43b83e8f`由来の保存96 bindingを既存Stage A preflightで再照合する。B1のidentity／references／law／旧契約も同carrierとbyte照合する。C1は同じcohortのsample00と01だけを使用する。

## B1：state-free operator transport

対象は保存13-round m0..12、q1..4096、t=.1q。新しいhorizonの採択ではない。保存DF sector norm＋元engineering guardをD_Hとし、同じHermitian H／H_CDFと共通不変sectorに対する標準Duhamel評価 `d_H(t) <= min(2,t D_H)`を使用する。保存PF／normalization-corrected RTE operator誤差に加えて

`eps_original = eps_PF + eps_RTE + min(2,t D_H)`

を評価する。Exact original-H groundに対するraw referenceは`A exp(-itE0)`。既存accuracyのraw単位変換と同じく`Delta=A eps_original`、`rho_lb=A-Delta`、`beta_sys<=asin(eps_original)`（eps<1）とする。DFはここに一度だけ入れ、旧own-DF energyへのWeyl加算を重ねない。保存binary64 normとguardをpremiseとする数値的十分条件であり、丸め誤差の厳密包囲ではない。

phase余裕をparametricに記録し、旧契約のβtotal=π/6／βstat=π/12との比較だけを履歴診断として併記する。新energy精度・成功確率・主資源指標は採択せず、N、decoder成功率、energy費用を新評価しない。保存own-DF signal／Nは使用しない。追加synthesis／law誤差を0とせず、ideal IID／exact nativeの条件付きstatusと実装未資格を分ける。十分条件不通過なら理由を保存しB2へ進まない。新matrix norm／exponential／ground vector計算は不要。

## C1：保存sample00→01のq=2、一回のwhole-path gadget

結果非依存の固定規則は「保存sample IDの先頭2件を昇順にappend」。U00を先、U01を後に作用し、V=U01 U00、B=U00、A=U01。cutはcompleteな最初のh-step終了後（その逆順head・scalar/identity込み）。全64 eventsと両stepのheadを保つ。Stage Aのevent16 cutを全qへ移す案ではなく、q2用の明示固定cutである。別pair／別cut／q>2へ展開しない。

物理Uは保存ordinary rawのcontrol=1を既存`control_one`で制限して得る。強い対照はStage A archive内の保存symmetric raw X wrapperから既知のH/core/H/measure shellだけを外し、core00→01を連結する。各stepのrelative Gaussian・basis共有・融合・scalar位相をそのまま保持する。新しいcross-step price最適化は行わず、価格比較はしない。独立保存CAR targetはCAR01 @ CAR00である。

既存`reference_gadget(A,B,chi)`を一回使い、actual A.inverse()、両stepのpath依存χのfsum、scalar/identity各一回を保持する。短gadgetの反復を代用しない。既存`run_prechecks`／`track_d_h4_reference_checks`でsaved raw／CAR、AB、actual adjoint、真空ray／χ、unitarity、direct両branch、Reference complex coherence・X/Y induced observable、方式間paired X/Y、χ欠落負対照を確認する。

Reference/ancilla=0で全256 system入力列・各131072出力振幅を既存CPU Aer doubleで検査する。17-qubit全Operatorは作らない。complex128、component／unitarity abs1e−10、observable数値Frobenius／signal2e−10、rtol0、global phase fitなしを維持する。raw QPYとreference／observableを保存する。新compile／compiled照合／資源価格は0。Aer snapshotの内部shots=1は量子Born samplingやRPE shot評価ではない。

## 並列化・保全・STOP

B1の短いscalar処理とC1のstate-actionを独立な2 workerで並列投入する（AI sub-agentではない）。初期観測はaffinity64 CPU・MemAvailable約484 GB。開始時にaffinity CPUの短いidle観測と空きRAMを取り、C1のthreadsを実際の空き資源から選ぶ。small matrixは既存BLAS1、大きいstate-action／Gramはそのthreadsを使う。CPU/RAM/worker/wallの人工capを新設せず、共有環境・他jobには触れない。

出力は`artifacts/track_d_b1_c1_validation/2026-11-06/run01`一回、retry/resume0。実行前後に同じSOURCE／入力を照合し、SOURCE、B1 rows、C1 checks／QPY／NPZ、実資源・inventory・result／STOPを保存する。固定入力・位相・correctness不一致は新規投入を止め、開始済み処理とpartialを回収する。閾値緩和・phase fitting・pair置換・無断retryはしない。

成功・不成立とも専用branch`codex/track-d-b1-c1-validation-20261106`へ保全・公開し、remote SHA／公開bytesを照合してmandatory STOP。B2、C2、sampling、full-wrapper compile、価格比較、全round費用集計、ground solve、DF再fit、詳細科学レビューへ進まない。次担当はユーザー／GPTであり、継続の科学判断と必要な明示許可を戻す。
