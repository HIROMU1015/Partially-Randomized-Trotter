# 次段階：有限parameter・Gaussian support certificateによる接続

2026-11-04 JST。**選択案は長q回路を作らず、全18roundへ一様なtransferを構成する有限audit。**
今回固定したのは対象・数式・入力・call範囲・判定条件である。
新helperは人工検証済みだが、H4向け統合runnerは未実装のため
DESIGN_FIXED / SCIENTIFIC_RUNNER_NOT_READY。実行commandはまだ提示しない。
この資料は新しい科学実行許可ではない。

## 1. 対象・成功時の主張

追加trajectoryは0本。選定する対象は6rのlawと18保存tupleに共通する全component/basis、
4 deterministic half block、融合するblock0のfull primitive、18 scalar parameterである。
局所probeを増やしても全word保証にはならないため、finite operator certificateを優先する。
最大q=131072を含むがq回のbuild/state-actionは不要。

成功時に言えるのは、固定H4/1Å/STO-3G/保存legacy DF12/L_D3/δ=.02、
8 qubit・N4 sector70・保存normalized ray・K2・cutoff1e-12・identity抽出、
同一signal τ/uniform53 law、support_for_source_basis_run_le_1、
ordinary-controlled boundary_optimized・ideal Hadamardというモデルについて、
任意合法wordのreferenceとの差を上界化し、そのIID平均に全36軸の既存局所十分条件を転送できた、という限定結果。
PCG64数学的IID、Float64仮想測定、compiled/noisy、状態準備、decoder/α/E0、
全体RPE成功保証・費用は含めない。

## 2. SOURCE・入力の固定

productionは82b2df6af80eb923302d35e369c4572ed9fa0a9fと同一のsrc。
C実行SOURCE7279f700a、categorical SOURCEf4905fff5とのbyte一致は
[FIXED_BINDINGS.json](FIXED_BINDINGS.json)に記録した。
本commitのconnection_adapter.py・人工test・証明が追加SOURCEである。
統合runnerを技術実装した後、そのrunnerを含む新commit SHAを公開してから科学実行許可の対象にする。
既存run02や全18categorical runnerを再起動して代用しない。

| 固定入力 | SHA256 / 意味 |
|---|---|
| 保存H snapshot | 13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b |
| B physical_state.npz | 793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4 |
| source_replay.json | ad7fd3617a524207fbc86a77de471ed6222b010046822a11bd9675edcace1f08 |
| 全18固定tuple | fixed_inputs.json、SHA256 72644832c1c8a159809efc30ff2909aaa576e49d69197d7ee007bf8a19d57673 |
| 全18 RESULT | d2e7a7db6、SHA256 410221788791280c7f69ed71ad188c13a7bf0db7167b87aac7a01738fc9ba568 |
| m17元RESULT | 34840ce9b、fixed_inputsのM17RESULT（96-bit enclosureを再利用） |
| C run02 RESULT | 82b2df6af、SHA256 5792eb42bcb2bcdd691d8978e5f4711ea71653c2451c1c210d9ba71ae6946077 |
| C run02 parameters | 同commit、SHA256 9b1c302e9ed77c71dd700dc34614a5ba8506f0b2f7e23115706e6f3476b82f85 |

各入力の完全path/blob、law/source依存はbindingsのfixed_inputs/evidenceへ固定。
H scientific hashは56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5、
preparation hashは89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919、
tail hashは65b31bcf432b703c523753ee4607236a2e9530d93401dcec9255e686cc3f04af。
stateの再load/再正規化・norm²再計算は不要。保存state-ray/provenanceと既存norm²包含をそのまま使う。
support training fingerprintは9dcdc4a28866dc223afb0ea212df645a3ed4a28e5c6dad6ddfc0ec1fcb28ba9e。
N/α/β/ε、r/K/τ、CDF、θ、成分係数・符号は変更しない。

## 3. 新しい科学call：許可対象にする最大範囲

これは実施数ではなく、後の承認対象案である。今回全て0。

| 操作 | 新規callの範囲・目的 |
|---|---|
| 保存H load | 1。constant/eigenvalue/etaを得る。SCF/DF再因子化/新しいH作成0 |
| DF preparation | 1。保存hash・cutoff・component/basis全件を照合 |
| support登録・selected descriptor取得 | 1。既存register_support_restricted_basesを使い、全selected hashをrun02記録と照合。欠けていたgate class/wire/parameter列を保存 |
| half primitive生成 | 4 block各1。保存exportのphase/RZ/RZZ payloadへexact hex一致が必要 |
| full primitive生成 | block0のみ1。halfとの差d_bをexact rationalで評価 |
| scalar parameter / boundary bound | 18 tuple各1。raw a,b、SOURCEと同順序のg、d_s、(q−1)d_bをFractionで評価。18個のparameter監査でありtrajectory signalではない |
| local controlled gate template監査 | RZ/RZZのcontrol constructorを172回以下。det halfの最大116＋block0 full最大8＋6r×2order×2sign×2support=48。各unique parameterはcache。class/definition/global phase、half-angle/doubling条件を確認。CZ/Phaseは既存SOURCEの任意parameterに対する恒等式を照合し、H4 parameterの追加constructorは呼ばない。回路全体のbuild/compile/Operator評価0 |
| Gaussian一粒子包含 | full11 basis＋selected support最大218 basisを各1回、8×8、96 bit。class/adjacent topologyを先に確認 |
| component operator差 | 218成分以下。support列のnorm差に両radiusを加え、CAR bound Δ_Sを作る。重複keyはcache |
| transfer / strict判定 | 18個のΔ_extra、36軸の保存interval拡張・保存εとのexact rational比較を1 pass |
| law/CDF/atan/mean/trajectory/sampling/shot | 新規0。保存literalの比較のみ |
| 70×70/256×256 evolution、q-sized circuit、transpile | 全て0。H4 local gateの数値Operatorも0 |

既存export_adapter.export_basisはlocal Operatorを評価するので、そのまま呼ばない。
runtime class、wire、parameter、canonical descriptorを読み出す最小exportを統合runnerに実装する。
full basisと全componentのclass/descriptor/hashは既存verify_runtime_basisの条件を全件へ拡張して照合する。
selected basis ID/hashはSOURCEを固定したrun02のsupport_proofsと対応させる。
CPU/RAM/workerの恣意的上限やwall-time capは設けない。
call数・入力・precision・STOPはこの範囲を維持する。

## 4. 回路構成と独立参照・判定

1. production SOURCEのfinite parameterを取得する。新trajectoryは生成しない。
   half primitiveと保存exportが一致すれば、保存categorical referenceのF/Rへのparameter bridgeになる。
   一致しなければ自動で代替referenceを作らずSTOP。
2. scalar再結合とblock0融合は生成されたbinary64をFractionへexact変換して差を評価する。
   mathematical energyから再計算した信号やlocal residualを参照にしない。
3. basisはruntime class/parameterからanalytic RZ/XXPlusYYを96-bit Acbで評価する。
   productionのbinary64 local matrixやcontrolled出力をcertificateの参照にしない。
   raw/fullとselectedの一粒子column差をCARで全Fock operatorへ転送する。
4. local Qiskit control templateのexact half-angleとzero global phase等を確認し、
   sourceで示した恒等式を全使用parameterへ適用できた場合だけΔ_lowering=0。
   未対応definitionはmissing evidenceであり0にしない。
5. Δ_extra=min(2,Δ_boundary+3qr max Δ_S+Δ_lowering)を18個構成する。
   existing I_catにはπ64が含まれるので二重加算しない。
6. 保存各軸のU_cat_upper、shot_half_width.upper、εから
   U_cat_upper+Δ_extra+shot_half_width.upper < ε
   をexact rationalで判定する。εの解釈は既存ball_adapter.classify_intervalのdecimal rationalと同じ。
   保存I_cat±Δ_extraも出力する。shot幅/α/N/βの再計算・再配分は行わない。
   この36軸strict判定自体が新しい科学的再解析であり、別途承認前には実行しない。

精度はAcb96 bit固定・1 pass。SOURCE identity、有限性、descriptor coverage、unitarity構造、
lowering、全36軸strict条件を全て満たせば
CONDITIONAL_IDEAL_CONTROLLED_CATEGORICAL_LOCAL_CONDITIONS_CONFIRMEDと記録できる。
厳密上界がεを超えた場合も、真の誤差が大きいと断定せずUNDETERMINEDにする。
新結果namespaceで原parameter/selected descriptors/上界/未判定理由を保存し、旧結果は書き換えない。
全体RPE_tolerance_certified=false、scientific_status=UNDETERMINED、両U=nullを維持する。

主な計算は最大229個の8×8 analytic basis積。
full basisと同一selected basisをcacheし、独立basisを並列化できる。
統合runner準備時に人工fixtureでcall数と並列結果の一致を確認する。
科学起動前にCPU affinity/RAM/共有負荷を確認し、basis数・実測された人工timingからworker/threadを選ぶ。
BLAS/FLINTの過剰並列を避け、共有環境を変更しない。
小行列なのでGPUや大きな行列backendへ無条件に展開しない。未計測のspeedupは主張しない。

## 5. mandatory STOP条件

- 今回は技術成果のcommit/push・公開照合完了でSTOP。科学callを開始しない。
- 統合runner未公開、SOURCE/入力hash/既存statusが不一致、対象条件が変わる場合は起動しない。
- preparationやselected basis hashが既存run02と不一致、support未被覆、未知gate/nonadjacent XXPlusYY、
  非有限値、保存primitive不一致、lowering前提不成立、包含が有限でない場合はUNRESOLVED/STOP。
- 96 bitの幅や一様majorantが大きくstrict条件を満たさない場合はUNDETERMINED/STOP。
  自動precision追加・trajectory sampling・別support policy・別referenceへ移らない。
- 科学callの上記範囲を超える作業、retry/resume、旧run01/run02再実行は行わない。
- 一回の許可範囲内で既存入力のbyte確認等を何度も承認し直す仕組みや、新規grant/承認証明は作らない。
- 成功・失敗を問わず原結果を保存し、公開照合後STOP。Dのα/energyやFloat64 bridgeへ進まない。

## 6. 担当と次の判断

**推奨：この有限certificate案を採用し、大量trajectoryや巨大q circuitの追加probeは選ばない。**

- Codexが次にできる技術作業：本計画の統合runner、selected descriptor export、
  全component coverage・call accounting・strict判定を人工fixtureで接続し、専用branchへ公開する。
  科学SOURCEが未完成である点を明記し、H4をloadせずに準備できる。
- ユーザー：公開された統合runnerのSOURCE SHAと本scopeに対して、別途明示的に科学実行を許可する。
- Codex：その許可後だけ固定finite auditを1 pass実行し、限定的なideal circuit結果と残存gapを報告する。

今のhelper/testへの承認だけをH4 executionの許可とは解釈しない。
準備済みSOURCEと実行許可を混同せず、本作業はここでmandatory STOP。
