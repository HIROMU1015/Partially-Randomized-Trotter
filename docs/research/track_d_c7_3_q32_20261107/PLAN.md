# Track D C7-3：q32/n8探索的paired価格の実行前固定PLAN

2026-11-07 JST。今回のユーザー指示による新規の限定科学実行。過去STOPを再開せず、新namespace/専用branch `codex/track-d-c7-3-q32-exploratory-20261107` を使用する。基点はC7-2最終 `709d6ed9c234bf6d5a8848d3a30be69d88fd6476`。適用AGENTS/PROJECT_MAP、正式共通指示2026-10-25、旧SOURCE・研究契約・結果/STOPを確認し保全する。サブエージェントは使わない。

## 1. 固定条件・実行件数

linear H4/隣接1.00Å/STO-3G/Nα=Nβ=2、number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1。元H厳密groundを仮定し準備費除外、ground vector/solve/DF fitは使わない。

元science SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。packet `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、law `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`。元SOURCEは既存固定exportから読み、既存preflightでcommit bytes/科学入力を確認する。Git worktreeと仮定しない。

|対象|固定値・上限|
|---|---|
|q/cut|32 / complete step16直後|
|path ID|q32_sample00..07、fresh 8本|
|occurrence|各32、順0..31、合計256|
|events|各occurrence32 ordered K0、合計8192|
|raw correctness|8 path|
|compile/postcheck|path raw PASS後だけ2方式×X/Y、最大32 full-wrapper|
|最終paired X/Y・χ|8 path|
|単体A/B/shellの追加compile|0|

同じordered pathを両方式・X/Yへ共有。全eventを一回だけ保存してhash固定した後にCAR/回路/価格を評価する。追加標本・prefix/旧cohortの流用・短path反復・結果依存選択なし。

## 2. Seed・モデルをsampling前に固定

cohort/domain `track_d_c7_3_q32_exploratory_paired_price_v1` / 同名`:independent_holdout:q32`、master2026110703。master tupleは(master,domain,C7-2基点)。既存trajectory_seedへ`proposed, endpoint, 4, C7_3_PAIRED_PRICE_ONLY, sample, occurrence`を渡す。NumPy1.26.4 PCG64、既存configure/sample_occurrence/iter_sample_rte_eventsを再利用。完全表は[seed_plan.json][SEED]。C6a/C6b/C7-1の既使用seedと非重複を確認し、domain名のunusedを未使用とは扱わない。seed分離はideal joint-IIDの証明ではない。

C6a結果 `e9d4e2f602d5f9e014507ca7e391719bab017b88`、science SOURCE `6b6667026ec8b670f610e41141d195883a9eb513` の保存係数48と共有校正共分散を再fitせずq32へ評価し、[frozen_model.json][MODEL]へ事前固定する。q8/q16はfit点にしない。packet/law/price boundaryと入力ファイルhashを保存する。

CX候補は`208−224q`、q32予測−6960、整数残差`observed ΔCX+6960`。全8 path・X/Yで厳密0なら観測一致、非零があれば候補違反。いずれも全path公式は`CX_GLOBAL_FORMULA_UNQUALIFIED`。5%幅で整数反例を隠さない。

一次familyは6資源×X/Yのpaired差12量。RZ/CX count・RZ/CX depth・total depth・size、qubitsは別幅制約。絶対価格/方式間・資源間・axis間共分散を保存するが、一次PASS familyを増やさない。X/Yを16独立pathとしない。

事前対照予測価格Sに対しτ=1+.05S、nominal classical α=.05、z=Φ⁻¹(1−.05/24)、n8/ddof1。一次は固定数値予測へ条件付け、`abs(observed paired mean−prediction)+z*s8/sqrt8<=τ`を探索的幅内とする。確認的95%保証ではない。

別population-line診断は同じerrorへ`z*sqrt(s8²/8+vcal)`を使用。q2/q4からの線形重み(−14,15)、独立校正集合という条件でvcalへ196Cov(mean2)+225Cov(mean4)を伝播し、method covarianceを維持する。cross-q独立性は旧設計を条件とする仮定で、無条件に証明された0ではないことを記録する。一次へ校正分散を二重加算しない。law/IID/tail/モデルbias/interval coverageが未資格なので両診断とも名目的・探索的、確認的PASSを認めない。

## 3. Whole-path・公平な対照

V=U31…U0=A B、B前16/A後16、Reference gadget一つ。strongはuncontrolled Gaussian、directional control、head relative fusion、tail full-frame同basis共有、31境界の合法half-sweep融合、scalar/identity集約。Referenceはactual A.inverse、analytic wholeχ、二つのCSWAP cascade/追加8qubits、30内部境界融合とcutの1境界損失。符号/位相/順序/weightを保持し、normalization/attenuationはunitaryへ重複適用しない。

価格境界は`single_hadamard_interrogation_without_state_preparation`。実full-wrapper価格を使い、局所和・fenced kernel＋shell・単純lane式へ代用しない。弱いbaseline・別cut・短gadget反復なし。

## 4. 既存実装と最小差分

C3 prepare_steps、C5 wrapper/_checked_price、B2/C6a CAR/head/atom cache、StrongBuilder、Reference adapter/actual inverse/χ、Stage A raw checker、C2 compile/postcheck、C3 pairedχ、C6b BatchedStateActionsとallocator、C7-1統計/凍結モデル/並列schedulerを再利用する。

旧q16 helperはq/nとclass既定値が固定なので、新Q32Request/seed関数を作り、**同じSOURCEを読んだ非公開module instance**でdomain定数・binding・request型だけを束縛して既存関数をそのまま呼ぶ。通常の旧q16 module/guardとファイルは変更しない。表示用q16ラベルだけq32へ正規化し、計算式を再実装しない。schedulerも独立module instanceでworker callbackと進捗上の予定8/32を束縛し、旧main/raw workerを起動しない。新しいGaussian/sampler/compiler/simulatorはない。

sampling前は旧prepare_commonが無条件にhead回路を構築するため、その入力検査部分だけをlaw-only contextへ分離し、既存prepare_arm/configureを再利用する。全eventの保存・hash固定より前にhead/whole-path回路をemitしない。workerの完全context/CAR/回路構築はhash固定後に行う。

構造特徴は既存raw/compiledだけから取得：tail内basis run/異basis遷移、support/sign/zero集計、head/tail/cut、scalar/χ、lane raw metrics、CSWAP・ancilla依存、compiled wire終端時刻とcritical-path特徴。C7-2 max-plusを再利用し既存metricsへ照合。filtered-out gateもwireを重み0で同期する。total DAGのlongest-path分類とfiltered wire時刻を区別し、compile不要predictorとは呼ばない。ordered event payloadや追加単体compileは公開/実行しない。

## 5. Correctness・P0・資源運用

独立CAR/ordered積、AB=V、actual adjoint/unitarity、vacuumχ、scalar/identity/signed rotation、strong各branch、complex coherence、X/Y observable、宣言位相を各pathで検査。raw PASS後のみその4 wrapperをcompile。全256 system入力state-actionを使い、17qubit全Operatorを構築しない。

P0：Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation1、qubits_initially_zero=false、topology-free、既存位相保持。component1e−10/observable Frobenius・signal2e−10/rtol0、phase fittingなし。CAR/raw/P0/paired X/Y/actual inverse/χ/reference leakageとSOURCE/input bindingを確認する。失敗後に精度やthresholdを変更しない。

実空きCPU/RAM/cgroup/負荷と完了q16の工程別peakを観測し、長さ比2を初期memory admissionの見積にだけ使う。実際の空きmemoryで投入し、恒常的な上限や厳密memory boundとしない。独立pathとraw PASS後compileを既存pipelineで並列化、cacheとbatched/native処理を使う。schedulerの旧80枠は今回最大40 taskより大きく非拘束で、科学task数/CPU/memoryが実投入数を決める。小行列BLAS/nested compilerの過剰並列を防ぐ。実行中SOURCE/既存環境/他jobに変更しない。

## 6. STOP・保全・公開

SOURCE/input/law/seed/domain/cut/phase/correctness/fair baseline/P0/価格定義の不整合、証拠破損、範囲外処理・共有環境干渉が必要なら依存科学処理を停止しpartialを保全。無断retry/resume・標本除外・seed/cut/threshold変更なし。価格候補反例/幅外は有効結果として固定診断を完了する。

出力 `artifacts/track_d_c7_3_q32/2026-11-07/run01` はfresh一回のみ。旧SOURCE/結果/grant/ledger/marker/STOPを保全。今回のSOURCE/PLAN/seed/modelを実行前commit/pushし、remoteとbytes照合後に一回起動する。実行中は自分のHEADを進めない。

公開は科学scalar/検査/価格/共分散/CX残差/構造feature/hash索引/SOURCE/STOP。raw event/QPY/NPZ/参照配列はlocalに保全し追加公開しない。内部資源観測/絶対path/資格情報を公開しない。RZ→T換算、実機routing/noise、N/全13-round総費用/actual RPE成功評価、q≥64/再fit/詳細科学レビューは対象外。

正常完了または検証不成立時に結果とSTOPを保存、公開可能成果を専用branchへcommit/pushし、remote SHA/tree/public bytesを照合して**mandatory STOP**。次の担当はユーザー/GPT：q≥64/価格モデル/必要追加証拠を判断する。Codexは自動進行しない。

[SEED]: ../../../artifacts/track_d_c7_3_q32/2026-11-07/technical/seed_plan.json
[MODEL]: ../../../artifacts/track_d_c7_3_q32/2026-11-07/technical/frozen_model.json
