# H4非DF/DF：残存certificate gateの原因・証拠充足性監査

2026-10-23 JST。対象は `codex/non-df-h4-c-reference-evidence-20261023` の結果commit **`86b8b002704aced8ca239414080f53793de81658`**、C科学SOURCE **`e80d7510880f8db993b777c627bb8e95b05a070d`**。旧科学SOURCEは `0c329b30a81accb35c212be8defa66fee0dd0f12`。今回の専用branchは `codex/non-df-h4-certificate-gate-audit-20261023`。

**DF rangeのfalseは表現別に設定された証明留保であり、保存eventの範囲逸脱を検出した判定ではない。** 全40軸の数値設計目標達成は保存証拠として確認できるが、literal wrapper/compilerへの保証転送と受理層は未完了。今回は既存証拠の照合・静的調査・人工テストだけを行い、新しい実H4計算、数値再解析、certificate生成・正式受理、N/G会計を行っていない。

## 固定対象・取得と保全

linear H4/隣接1.00 Å/STO-3G、8 system qubit、共通保存状態Nα=Nβ=2。Pauli-JW184項/PR prefix27、DF tol-only1e-8/rank10/PR prefix1＋one-body。S2、T=.8/q8/r4/K4、ε_sig=.05/.01/.005/.001/.0005のcosine/sine、4方式40軸を維持する。他系列のH2 S2/Yoshida4費用逆転研究を統合していない。

GitHub接続で基点branch、基点commit/tree、C SOURCE commit/tree、指定報告書、40軸CSV/JSON、normalization、completion、run内契約/消費済みgrant/STOP/ledger/両final result、保証設計・production/compile source、旧pilot4caseの結果/validation等、**37ファイル**を実取得した。base64 fileは復号後のbytesと固定Git blobを照合、報告書UTF8はblob SHAと改行正規化内容を照合した。run01の直下listingも全memberのGit objectと照合し、大きなmodelの内容は固定ローカルGitと保全SHAで確認した。[GitHub照合記録](../artifacts/non_df_h4_certificate_gate_audit/2026-10-23/github_verification.json)。

元の既知2,347ファイルを絶対pathで保全対象にした。元H4のSOURCE/input/raw/grant/marker/STOP、B、旧C失敗run、修正技術package、完成C runとそのinventoryを含む。他worktreeの通常作業を凍結する契約は追加しない。現在の正常C終了と、元H4のnative終了未証明・原因UNKNOWNを分けて保持する。

## A：DF-native PR event範囲の原因

### 実際の判定経路

[exporter:90](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80d7510880f8db993b777c627bb8e95b05a070d/src/trotterlib/h4_legacy_export.py#L90)は次を設定する。

```python
exact_event_range_proved = (representation == 'pauli_jw')
```

Cの[engine](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80d7510880f8db993b777c627bb8e95b05a070d/src/trotterlib/h4_ball_evidence.py#L394)はこの値を `range_plus_minus_saved_B_certified` へ転記する。eventのnormを計算し、不等式違反を検出してfalseにする分岐ではない。AST構造検査でこの経路を確認した。[静的証拠](../artifacts/non_df_h4_certificate_gate_audit/2026-10-23/static_flag_path.json)。

`TARGET_MET`は保存biasからの数値誤差上端と設計targetの比較である。rangeフラグや正式受理の判定ではない。exporterの `circuit_correspondence_proved=True` もencoded native local matricesと固定lawに対するscopeであり、同じcaseの `correspondence_scope` はliteral compiled wrapper未certifyを明示する。前者だけを読んでcompiler gate成立と解釈しない。

原因は二層に分かれる。

1. **実装上の直接原因（確認済み）**：Pauliだけをtrueとするproof policyを保存している。DFのexact encoded norm証明・literal測定への保証転送が接続されていない。
2. **留保が必要な数学的理由**：CはDF基底の `Operator(gate).data` とinverseをbinary64のexact値として取り込む。これらの丸め済み行列を、解析的なunitary gateそのものと同一視できない。SOURCEはbasis defectとnorm増幅を伝播し、unitarityを無断で仮定していない。実H4でunitarity違反・range超過が起きたかは今回計算しておらず、**UNKNOWN**。

### DFのblockとeventは異なる

HamiltonianのDF fragmentは `λ_l A_l²`（A_lはfermionic one-body）であり、一般にはPauli involutionでもunitaryでもない。そのままnorm1と扱わない。HamiltonianがHermitianなら理想的な `exp(-it λ_l A_l²)` はunitaryだが、有限PRのcorrected mean多項式自体にunitarityは要求しない。

DF PRは対角tailからsymbolic componentを抽出し、理想gateでは `P_j = sign_j U_j Z_support U_j†`（supportはZ/ZZ、identityは別phase）を使う。U_jがunitaryならP_jはHermitian involutionとなる。Pauli-JWのP_jはexact signed Pauli permutationなので、このU_jの丸め問題を持たない。

保存DF modelはPF11 block、PR2 block＋324 tail component。PR tailの9基底とdeterministic側2基底を合わせて11基底、局所操作名はRZとXXPlusYY、異なる局所行列hashは148種類だった。これはJSON metadataの重複除去であり、行列積・norm・固有値を新計算した結果ではない。[構造記録](../artifacts/non_df_h4_certificate_gate_audit/2026-10-23/saved_model_structure.json)。

### 係数・確率・normalization・測定範囲

paired有限lawではk=0,2,4、τ=λ_r δ/r、重み `w_k=|τ|^k/k! sqrt(1+(τ/(k+1))²)`、`B_K=Σw_k`、nominal `B=B_K^(qr)`。sourceのeventは

`W_e=(-1)^(k/2) V_j(atan(τ/(k+1))) P_jk ... P_j1`

であり、productを先に、rotationを最後に作用する。符号はsigned product/rotationとancilla phaseへ保存する。理想component確率とorder確率を掛けたevent確率に対し、LCU係数/確率は一様なB_Kとなる。ただし**binary64 CDF・53 bit格子で実装する確率と、nominal式は別**。Cはeffective massesと `Σ π_k (-1)^(k/2) R_k M^k` を用い、保存Bで補正したimplemented meanとの差を記録している。sampling lawやBを変更しない。

測定の定義をさらに分ける必要がある。

- **literalな二値測定**：ancilla bit b∈{0,1}を `Y=B_saved(1-2b)` と記録するなら、支持集合は定義からexactに{−B_saved,+B_saved}。この事実はDF/Pauli共通で、CDFがどのeventを選ぶかやCの丸め済み行列normに依存しない。PFはB=1。この抽象測定範囲は静的に確認可能であり、DF falseを「物理的な測定bitが範囲外」と解釈しない。
- **Cのencoded surrogate**：`U_m Z U_m_inverse` やそのrotation・積をbinary64行列のexact値で扱う場合、norm1やP²=Iは別の証明が必要。Cの値をliteral測定の期待値へ転送する誤差が未接続である。このscopeに対するDF falseは保持する。

Born probabilityを持つ理想回路・正規化stateという前提と、有限精度statevectorのnormや確率をそのまま厳密値扱いすることも区別する。支持集合の証明だけではCのmean/biasとの対応は完成しない。

両PRで `saved_log_B_upper_encloses=true` が保存されているのはnominal log_B上界の包含である。保存 `B_saved_difference` と実測定のB_savedスケールをshot用上界に結ぶ照合は別項目。上界を増やしたりB_savedを置換したりしない。

**実H4に対してFAILED_CONDITIONと断定できる新しい事実は今回0件**。falseは証拠不足・proof接続未実装を表し、range逸脱や方式不適格の判定ではない。人工rational反例で「丸め行列をexact unitaryと同一視できない」という推論上の問題だけを確認した。

## B：数値包含からliteral compiled wrapperへの転送

[対応表CSV](../artifacts/non_df_h4_certificate_gate_audit/2026-10-23/wrapper_transfer_map.csv) / [JSON・source位置](../artifacts/non_df_h4_certificate_gate_audit/2026-10-23/wrapper_transfer_map.json)に9境界を記録した。

| 境界 | 確認済みの定義・保存物 | certificateに不足するもの |
|---|---|---|
| 入力→C | common state hash、n=ψ†ψ、full256 H、basis順序、SOURCE/input/runtime | 同じrayの正規化状態と実際の測定準備の明示的接続 |
| 数学的信号 | full-H reference、nominal P5、実装law mean、saved B、signed biasと上端 | literal gate/measurementに同じsignalを移す誤差 |
| Pauli evolution→control | 一項PauliEvolutionGate、signed product、global phase、gate.control(1) | synthesisの位相・角度丸めとcompiler全変換の保証 |
| DF evolution→control | basisは非control、diagonalだけcontrol、basis inverseの逆順、identity/符号をancilla PhaseGate | 丸めlocal行列から理想RZ/XXPlusYYへの包含、control=0分岐のbasis cancellation・reuse上界 |
| q-stepとevent順序 | fixed S2の前半/PR r回/逆半、raw concatenation | 全event supportに通用するcompositional proof、phase加算誤差の接続 |
| control→wrapper | ancilla n、H、cosineはH、sineはSdg-H、bit測定 | 証明済みcontrolled Uとnormalized stateを用いてp0−p1=Re/Imを接続 |
| compile前後 | 固定basis rz/sx/x/cx、optimization1、seed17、layout不変check、設定hash | postcompiled program/変換receipt、exact等価性または一様な保証誤差 |
| 出力→補正signal | ±1のbit observable、固定B_saved乗算、PF B1 | 保存C biasとの結合、保存上界とのscalar binding |
| certificate→N/G | 旧会計・epsilon/alpha・guardを保存 | proof validator・正式受理・その後の別承認会計 |

generic controlでevolutionのglobal phaseを落とすと、controlled相対位相が変わる。DFのconstant/extracted identity/eventの符号も、system global phaseではなくancilla PhaseGateとしてcontrol後に残す。compilerの後の**回路全体の**global phaseは測定確率に影響しないが、controlする前のsystem phaseを捨てることとは異なる。

### 旧operator・compiled probability証拠の再利用

旧pilot SOURCEの実行では、PF各1軌道、PR各256軌道、計514軌道/1,028 wrapperに対するコンパイル前後の固定state ancilla確率比較を行っていた。保存された最大差をそのまま転記する。

| case | 保存軌道 | compiled probability最大差 | trajectory operator検査 | forced order |
|---|---:|---:|---:|---|
| pauli_jw_PF | 1 | 2.4444890556196697e-12 | 1 | なし |
| pauli_jw_PR | 256 | 1.3922196728799463e-13 | 3 | 0/2/4 |
| df_native_PF | 1 | 5.0616733027197824e-12 | 1 | なし |
| df_native_PR | 256 | 2.2315482794965646e-13 | 3 | 0/2/4 |

これらは旧固定1e-9閾値内という**経験的検証結果**で、厳密な包含上界ではない。forced orderも各orderの固定例であり全component組合せの証明ではない。今回514軌道のevent_record/trajectory_record/seed/価格/設定/input wrapper fingerprintを読取照合した。[保存証拠一覧](../artifacts/non_df_h4_certificate_gate_audit/2026-10-23/legacy_compiler_evidence.json)。新compileやsampling replayは0。

PR production eventの保存orderはPauli 0:8157/2:35、DF 0:8167/2:25、両方order4観測0。DFで観測したcomponent IDは160で、完全tail324を網羅していない。forced order4と実sampling未観測を混同せず、有限標本を全event保証へ拡張しない。

`TrajectoryAudit.join` の `wrapper_fingerprints` は `cost.actual_circuit_fingerprint`、すなわち**pretranspile wrapper**のfingerprintである。cache key・compiler設定・countsは保存されるが、postcompiled programのfingerprintやcertificateではない。compiler cacheのpersisted payloadもmetricsであり、回路本体は返り値/in-memory保持にとどまる。今回の既知4case directoryにQPY/QASMがないことを確認した。filesystem全体にないとは主張しない。

したがって、旧保存最大差とhashを接続する作業は解決できたが、過去のcompiled binaryを厳密に再検証するための実体はこの保存集合にない。将来の再compileは新しいprogramを生成する行為であり、失われた旧programの回収証明にはならない。旧の科学値・native欠落・STOPを事後に置換しない。

## C：40軸の受理条件と充足性

[全40軸マトリクスCSV](../artifacts/non_df_h4_certificate_gate_audit/2026-10-23/certificate_gate_matrix.csv) / [JSON](../artifacts/non_df_h4_certificate_gate_audit/2026-10-23/certificate_gate_matrix.json)、[gate別必要条件・SOURCE位置・不足](../artifacts/non_df_h4_certificate_gate_audit/2026-10-23/gate_catalog.json)に14項目を対応させた。全4方式×5精度×両軸を個別行に持ち、formal_certificate_accepted=false、N/G=nullを保持する。

| gate | Pauli PF | Pauli PR | DF PF | DF PR |
|---|---|---|---|---|
| 保存数値enclosure/full-H reference/encoded bias | VERIFIED | VERIFIED | VERIFIED | VERIFIED |
| PF構造・PR nominal log_B保存上界 | VERIFIED | VERIFIED | VERIFIED | VERIFIED |
| 二値測定の支持集合 | VERIFIED | VERIFIED | VERIFIED | VERIFIED |
| exact encoded event norm | VERIFIED | VERIFIED | MISSING_EVIDENCE | MISSING_EVIDENCE |
| 実装lawと統計モデル | VERIFIED | ASSUMPTION_DEPENDENT | VERIFIED | ASSUMPTION_DEPENDENT |
| literal wrapper転送 | MISSING_EVIDENCE | MISSING_EVIDENCE | MISSING_EVIDENCE | MISSING_EVIDENCE |
| compiler転送 | MISSING_EVIDENCE | MISSING_EVIDENCE | MISSING_EVIDENCE | MISSING_EVIDENCE |
| IID/confidence条件 | ASSUMPTION_DEPENDENT | ASSUMPTION_DEPENDENT | ASSUMPTION_DEPENDENT | ASSUMPTION_DEPENDENT |
| SOURCE/input/runtime/保存来歴 | VERIFIED | VERIFIED | VERIFIED | VERIFIED |
| B_saved→shot用上界の接続 | VERIFIED | EVIDENCE_EXISTS_UNCHECKED | VERIFIED | EVIDENCE_EXISTS_UNCHECKED |
| 最終結合誤差予算 | EVIDENCE_EXISTS_UNCHECKED | EVIDENCE_EXISTS_UNCHECKED | EVIDENCE_EXISTS_UNCHECKED | EVIDENCE_EXISTS_UNCHECKED |
| certificate受理実装 | MISSING_EVIDENCE | MISSING_EVIDENCE | MISSING_EVIDENCE | MISSING_EVIDENCE |

VERIFIEDは指定された**scopeでの既存証拠の確認**であり、新しい正式certificateの発行・独立科学レビューを意味しない。DF PFのstochastic normalizationは不要だが、丸めbasis→literal gate/measurementへの転送はPFにも残る。実H4にFAILED_CONDITIONは追加認定していない。元H4 native欠落はUNKNOWNのまま独立管理する。

PCG64列そのものの独立性は証明していない。固定契約のIID uniform-word・fresh independent shotという仮定を明示して条件付き保証を設計する。**無条件RNG独立性の証明を新必須gateに追加しない。** Cost MC256標本は量子測定shotではなく、母平均費用・希少event費用の保証も別問題である。

受理を妨げるのはフラグだけではない。Cは正式受理を常にfalseとし、既存 `fixed_signal_shots` は `CERTIFIED` というラベルを受け取ってもproofを検査せず `certificate_verified=False` を返す。全proofとSHA・対象scope・仮定を照合する受理層が別に必要。ラベルやフラグだけをtrueへ変えてはならない。

旧N/Gがnullとなった実行経路は、旧empirical allowanceとheadroom/比率guardで保留されたものだった。今回のC rangeフラグが直接旧会計のnullを発生させたわけではない。certificate受理、固定guard確認、N/G再会計を独立段階として維持する。

## 解決範囲と次段階の最小案

| 区分 | 解決する内容 | 今回の状態 |
|---|---|---|
| 1：既存照合・接続 | flagの原因、40軸・normalization・SOURCE、514 event/価格join、二値出力の支持集合、IID仮定の整理 | 調査・照合完了。保存包含/旧log_Bを再計算せず再利用可能 |
| 2：技術修正・人工test | encoded norm/readout rangeを分離したschema、phase/control/basis証明checker、formal受理層、compiler pass証明の接続 | 別SOURCEの実装案のみ。旧実装を変更していない |
| 3：新しい数値証拠が必要になり得る範囲 | 理想局所gateと丸め行列の差、保存Bと上界の接続が既存endpointだけで証明できない場合、transfer結合上界 | 実H4 numerical診断は未実行。必要性と最小対象を固定して別承認 |

[最小契約案](../artifacts/non_df_h4_certificate_gate_audit/2026-10-23/minimal_next_contract_DRAFT.json)は非実行DRAFT、execution_authorized/grant_valid/dispatch_enabledはfalse。全40軸を最初から計算し直す案ではない。

1. **技術段階T**：rangeのscope分離・受理checker・symbolic phase/control/compiler規則を実装し、人工/mockで拒否系を確認。実H4科学call0。技術SOURCE/runtime seal後STOP。
2. **必要なら局所診断L**：保存DF modelの11基底・148 distinct local gateだけを対象に、RZ/XXPlusYYの解析的2×2/4×4 gateと保存forward/inverse行列の誤差を外向き評価する。最大forward148＋inverse148、basis合成11、symbolic support324の検査。full256 reference/law mean/bias、dense exp、sampling/RNG replay/compile、SCF/FCI/DF/state solve、N/Gは0。精度と実装SOURCEを先に固定し、同じ上界を両軸・5精度へ再利用する。既存semantic allowanceに含まれる項を二重加算しない。
3. **compiler段階K**：まず固定compilerの分解・rewriteをsourceとgate定義から証明する道を検討する。旧program未保存のため、source/設定/有限標本最大差だけで全旧programへの保証を完了したとはしない。この道で証明できない場合、新programの保存・pass receipt・family全体の保証を対象にする別契約をユーザーへ提示する。PFなら2表現×2軸の4wrapper実体が最小候補だが、PRに256×2軸の盲目的再compileを課さない。PRの正確な追加workloadが未確定のまま実行を承認済み扱いしない。
4. **受理段階A**：transferと保存B upper bindingを解決した後、保存C上端と追加誤差の結合を固定target/guardで検査し、既存IID仮定を明示した条件付き受理を別承認で行う。単にTARGET_METから受理しない。正式受理後もN/G再会計は別承認・別STOP。

将来の科学診断はnew SOURCE/input/runtime seal、未使用namespace・新grant、call前予約・非返却ledger、native wait/reap witness、partial/receipt/STOP保存、retry/resume禁止、mandatory STOPを条件とする。既存C grantは消費済みで再利用しない。誤差が固定予算を超える・証明不能・未対応gate/pass・非有限・hash不一致なら保留STOPとし、B/allowance/guard/lawを変更しない。

## 検証・終了状態

人工1 system qubit/2 qubit wrapperの14 testはPASS、FAIL/ERROR/skip0。DF signed product・order0/2/4・identity/control phase・basis reuse、Pauli符号、cosine/sine確率、global→relative phase、bit出力範囲、exact rationalの丸め行列反例を確認した。浮動小数点の人工診断を厳密certificateへ格上げしていない。sampling/RNG replay/compile entrypointを呼ばない構成で検査した。

構造検証はGitHub/byte/AST/JSON/CSV/hash/joinだけを使う。科学moduleの数値評価、保存40軸の数値再解析、reference/law/biasの再計算、新高精度/区間計算、sampling/compile、certificate生成・受理、shot/Gは0。主担当自身の検査であり、別AI/別担当へ委任していない。[検証記録](../artifacts/non_df_h4_certificate_gate_audit/2026-10-23/structural_audit.json)。

今回の報告・補助資料・保全と終了監査を専用branchへcommitする。2026-10-23の更新共通指示「コミット許可に含まれるプッシュ許可」に基づき、この調査branchのpushも承認済みとして行う。公開状態は[公開記録](../artifacts/non_df_h4_certificate_gate_audit/2026-10-23/publication_status.json)と最終チャットで報告する。元H4 native未証明・監視原因UNKNOWNと今回C native PASS、旧20条件N/G=null・40軸UNDETERMINED、40軸数値目標達成・正式受理0軸を保全する。**mandatory STOP。新しい数値診断・科学計算・certificate正式受理・N/G会計へ自動移行しない。**

次の担当は**ユーザー**：T/L/Kのどの範囲を次に進めるか、compiler保証の対象を旧保存結果か新programかも含めて判断・承認する。**Codex**は承認後だけ、その範囲の新技術実装または限定診断を行う。
