# H2・H6 Pauli-JW：PF/B0/PRの有限候補・条件付き資源比較

2026-10-25。基点 `fbfbfd21a252a4a32f52e3b40e50acd91b3c626a`、専用branch
`codex/pauli-h2-h6-resource-comparison-20261025`。実行許可は今回のユーザー指示による。
新しい承認証明・grant・認証基盤を作成せず、旧run/STOPは再開していない。

**H2/H6とも、凍結したPR finalistの独立確認平均を用いると、5要求精度すべてで
登録決定論的PF集合より低い条件付きGが得られた。** H2の確認集合最小はprefix1/q1/r4/K4、
H6はε=.05でprefix684/q1/r8/K4、他4点でprefix78/q8/r8/K2。
これは仮想ρ=0、有限探索、保存標本価格による比較であり、実compiled精度や母平均資源優位の認定ではない。
研究上の主要claimの正式採択・詳細科学レビュー・論文本文の執筆はしていない。

## 1. 固定条件と入力の対応

|項目|H2|H6|
|---|---|---|
|分子・geometry・basis|Linear、隣接1.00 Å、STO-3G、中性singlet|同左|
|system / ancilla|4 / wire4|12 / wire12|
|非identity Pauli項|14|918|
|先頭all-Z prefix / 最初の非対角項|10 / 11|78 / 79|
|Pauli Hamiltonian hash|`4eca732f1835e5d52ba2672277edbe6ee911114ef9b336d7e820c23ed3abc840`|`37b2747eaafacef8b4b470e5f4d6e195ac0602dc8e59953e3b76152d4576c847`|
|raw state hash|`4dc87df6c0466c1e5559ffcf3603b3a45ca7d1453b0657603188c120fe297b11`|`ff1ba6309c6b17a263195eea718f1d5243cbe6b3422c6a9d1fbb0fe8b212ff0c`|
|状態の来歴|旧non-DF H2 pilotの保存raw state|新しい同一MO積分のPauli full-H、Nα=Nβ=3 sector最低固有vector|
|新規SCF / FCI / DF生成|0 / 0 / 0|1 / 1 / 0|

全方式で各分子内の同一Hamiltonian・raw ray・full-H reference区間・exact raw normを共有した。
constantを常に保持し、項はabs係数降順/label tie-break、cutoff0。B0/B0Iは同じ回路・信号なので重複しない。
係数に小さい非零値があっても、918項から都合よく落としていない。

H2は旧pilot SOURCE `b5ed2c6e2399b55a0c24eb8437f842892a6703ef` の入力だけを再利用。
旧mutable-borrow失敗、旧grant、旧STOP、未受理価格を今回の成功へ読み替えていない。
H6の確認した保存snapshotはDF rank19のprimitive/stateで、元の非DF MO積分が含まれなかったため、
許可範囲で新RHF/JW入力を作成した。新state energy −3.2360662798923525 とFCI −3.236066279892345、
full-H residual経験値約4.45e−15を照合。400次元sector sparse eigensolverを使い、4096次元dense指数は作成していない。
旧H6 DFの状態・価格・biasを移植していない。

両分子共通：S2、T=.8、q=1/2/4/8、PR r=2/4/8、K=2/4、既存paired finite Taylor RTE。
central-rotation制御、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/Target/coupling/layout/routing指定なし。
full cosine/+Im sine Hadamard wrapperのcompiled RZ数を価格とし、state preparationを含めない。
Sdg-before-final-H、constant/global/relative phase、little endian、同じtrajectoryの両軸を保持した。

入力と詳細版情報は [`H2/inputs`](../artifacts/pauli_h2_h6_comparison/2026-10-25/H2/inputs/)、
[`H6/inputs`](../artifacts/pauli_h2_h6_comparison/2026-10-25/H6/inputs/)、
[`runtime_versions.json`](../artifacts/pauli_h2_h6_comparison/2026-10-25/runtime_versions.json)。

## 2. 候補集合、探索、独立確認

開始前の全体条件・追加規則は [`study_plan.json`](../artifacts/pauli_h2_h6_comparison/2026-10-25/study_plan.json)。
具体的candidate/law/input/code SHAは各batchのplan/specに固定し、過去runをresumeしていない。

- H2：初期103構成、追加12＋5、計120。決定論的は全prefix1–14×q4種の56構成（B0 52、full PF4）、PR64構成。
- H6：初期94構成、追加73＋48、計215。B0110、full PF4、PR101。初期B0 prefixは1/78/79/230/459/688/826/910/914/918。
- H6のPR初期prefixは0/78/79/230/459/688。保持数/q骨格、K2対照、r2/8対照から開始した。全直積ではない。
- 各追加は精度別上位候補のprefix近傍・隣接q/r・K切替。第2roundは第1roundの>1%改善/境界に基づき実施。登録集合の大域最適性は証明していない。
- PRは適格候補ごと8本の探索価格で選抜し、精度別上位2のunionを**確認前に凍結**。H2は2候補、H6は4候補。
- 別seedの確認を48本から開始し、差/SE規則によりH2は各96本、H6は各48本で終了。探索と確認をpoolしていない。
- seedは `SeedSequence([20261025, molecule_size, L_D, q, r, K, cohort_tag, index])`。探索101、確認202。

H2の探索nomineeはprefix0/q1/r2/K4だったが、独立確認集合の最小はprefix1/q1/r4/K4になった。
両方を `PR_exploration` / `PR_confirmed_nominee` / `PR_confirmation_set_minimum` として保存した。
主表の「最良PR」は**凍結確認集合内の観測平均G最小**で、確認値を見てからの選択を含む。
名目t区間は選択・逐次観察・全候補同時coverageを保証しない。探索全体の真の母平均最小とは呼ばない。

価格は実compileの実測で、モデル価格ではない。候補ID/trajectory index単位の重複登録は除いた。
Qiskitは各native processで使い、同じprocessの複数threadからcompiler/global equivalence libraryへ触れない。
独立jobは利用可能affinity64とjob数に合わせて並列化。小さいベクトル/行列を多数のprocessで処理するため、
各processのBLAS/OMPは1threadとし、入れ子の過剰並列を避けた。OS・共有環境・他processは変更していない。

## 3. 信号と数値誤差の評価

Hamiltonian/state/sampler/control/会計は既存実装を再利用した。H4のArb mean接続は数学的oracleとして再利用し、
実H4への新科学callは0。H2/H6 production用に、matrix-free binary64中心ベクトル＋L2誤差の接続を追加した。
新しいRTEアルゴリズムやcompiler familyは作成していない。

数値処理の根拠は次の通り。

1. 初期stateと係数を保存binary64値のexact数として扱い、exact raw squared norm νを計算する。状態を再正規化しない。
2. Pauli作用は符号・実虚交換・index permutationで実装。2/3qubitの全Pauli/signを既存C作用と照合した。
3. scalar trig/係数/interval endpointはArb128bitで包含する。vectorのcomplex積・和はIEEE binary64のγ₈ₘでL2 roundoffを覆い、subnormal加算項も保持する。fast-math/float32は使わない。
4. 共通入力誤差eを、全Pauli rotationではunitaryとして1倍、PR event平均ではcontractionとして1倍伝播する。局所演算誤差だけを別に加える。
5. 実CDF/uniform53 word法則のexact massを既存C規約で取得する。M=ΣpⱼPⱼ、event平均F=Σπₖ(−1)^(k/2)[cosθₖ M^k−i sinθₖ M^(k+1)]。独立drawの積の期待値を用い、非可換productを勝手に交換しない。Fはunitary eventの凸平均なので||F||≤1。
6. Retained operandはexact coefficient×saved step time、PRのnative step scalar/rotation角は既存binary64値。B_savedは法則固有のbinary64値を一度だけ適用する。law meanを標本signalで置換しない。
7. raw-state referenceはDuhamel bound `||(exp(-iHT)−exp(-iET))ψ||≤T||(H−E)ψ||` から包含する。ψがexact eigenstateであるとは仮定しない。

reference半幅はH2約2.56690e−14、H6約1.34277e−11。PRのcorrected mean、bias、数値allowanceを同じνで評価した。
上記の数値包含は記載した算術モデルとIID uniform-word法則を前提にする。正式certificate受理やcompiled ρ保証とは別である。

初版はunitary内のcos/sin分岐を別々に伝播しており、正しいが過度に広い上界だった。
H6全215候補を別SOURCEで精密化し、旧target/rawを変更せず `target_refined.json` を追加した。
旧・新区間の重なりを全件確認。full PF q8/.0005は数値未確定から条件付き適格へ解決した。
これは科学条件や精度閾値の変更ではない。

人工検証：旧central-controlの12試験、初版vectorの4試験、精密版vectorの4試験を実行してPASS。
後2組は既存Arb96bit mean oracle、raw-ray scale、B一回、phase/sign/測定両成分、Duhamel区間を確認した。
作者自身の技術検査であり、独立科学レビューではない。

## 4. 条件付きshot/G会計

各軸のεは元のbinary64 `ε_sig/√2`、α_axis=.025。
`h=ε_axis−|bias|−u_enc−ρ`、`N=ceil(2 B_up² ln(2/α_axis)/h²)`、`u_enc/h≤.01` を維持する。
既存のoutward exp/log・exact rational ceil・safe-integer gateを再利用した。
`G=N_cos C_cos+N_sin C_sin`、主表は仮想ρ=0。B0/PFのB=1、PRは各candidateのB_saved。
rho guard内の会計端点も保存し、独立guard範囲で全10比較はPR低Gのままだった。
実ρがそのguard内にあることは未証明である。

H2は全600精度行が適格/不適格に確定。H6は1075行のうち1行だけnullを維持した。
対象はprefix0/q1/r4/K4・ε=.0005で、bias区間精度の問題に加えてB≈202009と大きい。
仮にbias=u=0でも、同じ十分shot式の下限は2,861,126,492,304,307,821、既存safe-integer domainを超える。
保存平均価格を掛けた楽観的G下限も約5.08565e20で、表の候補より大きい。
これは不明値を∞へ変えたものではなく、nullを保った別の下限診断である。母平均費用の下限ではない。

## 5. 主結果

以下の表は保存JSON/CSVからの表示。価格・bias・B・u・軸別N・G・SEの全fieldは
[`H2_H6_best_comparison.csv`](../artifacts/pauli_h2_h6_comparison/2026-10-25/H2_H6_best_comparison.csv)、
全1675精度行は [`registered_candidate_precision_table.csv`](../artifacts/pauli_h2_h6_comparison/2026-10-25/registered_candidate_precision_table.csv)。

### H2（独立確認集合の観測最小）

| ε_sig | 最良決定論的 L/q | G_D（RZ） | PR L/q/r/K | G_PR（RZ） | PR/D | 名目95%比区間 |
|---|---|---:|---|---:|---:|---|
| 0.05 | 14/1 (PF) | 1353240 | 1/1/4/4 | 625812.5 | 0.462455 | [0.431927, 0.492982] |
| 0.01 | 14/1 (PF) | 45976320 | 1/1/4/4 | 15645748.44 | 0.340300 | [0.317836, 0.362764] |
| 0.005 | 14/1 (PF) | 293617350 | 1/1/4/4 | 62585996.88 | 0.213155 | [0.199084, 0.227226] |
| 0.001 | 14/4 (PF) | 1.518367778e+10 | 1/1/4/4 | 1565249433 | 0.103088 | [0.096283, 0.109893] |
| 0.0005 | 14/4 (PF) | 7.773339301e+10 | 1/1/4/4 | 6264001704 | 0.080583 | [0.075264, 0.085903] |

PR価格/shotは両軸同値（個別にcompileして確認）：H2 prefix1/r4は24.21875 RZ、確認n96。

### H6（独立確認集合の観測最小）

| ε_sig | 最良決定論的 L/q | G_D（RZ） | PR L/q/r/K | G_PR（RZ） | PR/D | 名目95%比区間 |
|---|---|---:|---|---:|---:|---|
| 0.05 | 724/1 (B0) | 182956812 | 684/1/8/4 | 172546742.4 | 0.943101 | [0.942779, 0.943423] |
| 0.01 | 830/2 (B0) | 1.070808301e+10 | 78/8/8/2 | 7776692907 | 0.726245 | [0.724595, 0.727895] |
| 0.005 | 830/4 (B0) | 6.548679536e+10 | 78/8/8/2 | 3.194529064e+10 | 0.487813 | [0.486704, 0.488921] |
| 0.001 | 864/8 (B0) | 2.989511807e+12 | 78/8/8/2 | 1.004541674e+12 | 0.336022 | [0.335259, 0.336785] |
| 0.0005 | 864/8 (B0) | 1.248676181e+13 | 78/8/8/2 | 5.648322276e+12 | 0.452345 | [0.451317, 0.453373] |

PR価格/shotは両軸同値（個別にcompileして確認）：H6 prefix684は約7969.4583 RZ、prefix78は約3122.7083 RZ、各確認n48。

### H4保存値を含むPR/Dの一覧

| ε_sig | H2 | H4（固定PR） | H6 |
|---|---:|---:|---:|
| 0.05 | 0.462455 | 1.147837 | 0.943101 |
| 0.01 | 0.340300 | 0.468458 | 0.726245 |
| 0.005 | 0.213155 | 0.281686 | 0.487813 |
| 0.001 | 0.103088 | 0.135617 | 0.336022 |
| 0.0005 | 0.080583 | 0.145333 | 0.452345 |

1未満は保存標本平均を用いたPR低G。H4のPRは未最適化の歴史的baselineで、H2/H6との同一探索比較ではない。


## 6. 希少event、比較の範囲、残る不確かさ

Order0/2/4の実法則確率と探索/確認の観測数は `rare_event_evidence.json` に対応する。
K2にはorder4がsupportにないが、K4の未観測order4や未観測componentを費用0にはしていない。
SE/名目Student-tは有限確認標本のばらつきで、PR母平均費用tailの厳密上界ではない。
価格がほぼ一定の候補でも、この区別は維持する。

H2/H6とも有限確認平均では5点すべてPR低G。H4の保存固定PRはε=.05で最良B0より高G、他4点で低G。
ただしH4のPRはprefix27/q8/r4/K4固定の256本で、今回H2/H6のL_D/q/r/K最適化と同じ探索範囲ではない。
分子間でstate・reference・candidate集合・cohortを混ぜず、分子サイズの普遍則や探索範囲外の最適性を主張しない。

DF-nativeの保存価格は、H2でgeneric対central control、H6で新Pauli入力対旧DF入力の差がある。
同条件の補助G比較へ無理に接続せず、旧DF結果は別系列の参考資料として保持した。新DF探索は0。
旧 `codex/h2-accepted-compare-20261023` の資料も確認した。同資料はDF rank3・prefix1/K4/outer32等の
限定DIAGNOSTICS受入経路であり、今回のPauli central系列と同一ではない。旧identity/COMPARE状態を変更していない。
H4の6局所Arb保証もH2/H6へ転用せず、PR全law実装残差、実compiled精度保証、正式certificateは未解決のまま。

追加の303点gridは今回行わず、承認された主要5精度の比較を完了した。
PR/B0の連続精度での交点、未探索候補、別geometry、エネルギー/RPE全体への一般化は評価していない。

## 7. SOURCE・保全・保存・公開

新native SOURCE履歴：
`d17780fcb57bcff344325a8632f718c0cc56f1dc`（開始計画/入力コード）、
`0880c38e3e8a6ea57882ec66e6366a8b86e408e2`（入力/初版評価/探索）、
`a8c7c7dca5a03aae1b008a09048d89c216bbf89c`（相関誤差精密化/確認会計）。
各runの実コード・入力hashは対応plan/SOURCEに保存した。
入力生成開始時は基点commit＋新コードSHAを固定しており、初回git commitはauthor未設定で失敗した。
Codex名義を当該commandにだけ設定し、同じコードをd17780fcbへ保存した。入力再計算やSOURCE hashの事後書換えはしていない。

[`verification.json`](../artifacts/pauli_h2_h6_comparison/2026-10-25/verification.json)：
記録されたnative child 2372件は全exit0/reaped。
H2 target120、H6 target215＋精密化215、reference residual2、価格trajectory1818、実軸compile3636。
全QPYのcompressed/uncompressed SHA照合PASS。旧H2入力、旧H4結果/STOP、旧grant/ledger/markerは保全した。
過去のH2失敗やH4監視UNKNOWNを今回のnative PASSへ格上げしていない。

raw/spec/law/target/QPY/価格/実行log/監視・wait記録は分割圧縮archiveと `data_archives.json` に保存する。
QPYは元byte列をtarへ入れ、元のdeterministic gzipコンテナを可逆に復元できることも確認した。
[`restore_pauli_h2_h6_evidence.py`](../scripts/restore_pauli_h2_h6_evidence.py)で空directoryへ復元し、全member SHAを検証できる。
新SOURCEのnative Git履歴も `SOURCE_history.bundle` に保存し、GitHub公開commitのmetadata差と科学的code/input同一性を区別する。
公開後のcommit/remote照合は `publication_status.json` と最終チャットに記録する。

## 8. 終了と次の担当

計算・比較・保存・公開照合後にmandatory STOP。新しい科学実行、R2-C、正式certificate、主要claim採択へ移行しない。
ユーザー/GPT：この有限候補・条件付き結果をどう研究上解釈するか判断する。
Codex：その後の明示指示がある場合だけ、判断に必要な範囲を実施する。現段階の追加科学計算は不要。
