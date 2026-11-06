# A1 N2補完：全supportのengine誤差保証と次回実行仕様

**技術設計・人工検査のみ。実N1 gate/stateを用いた以下の数値評価は未実行。総b_engineはUNKNOWN。**
対象は保存H4/1.00 Å/STO-3G/legacy DF12/8qubit/全4電子sector/L_D3、m0/q1/δ=.02/r1/K2/coefficient_atol0、N1 actual finite-grid Qとselected policy、N2保証state、X/Y、理想gate・保存DFモデル限定。
固定commitと接続処理は[ENGINE_CONNECTION](ENGINE_CONNECTION.md)、現在の結果は[REPORT](REPORT.md)。

## 誤差の対象と定義

一次targetは保存selected logical gate列のbinary64 parameterをexact angleと解釈したideal unitaryと、保存DFの全4電子groundに対するattenuated複素mean。
Qやselected policyを新しいproposalやfull-basis参照へ置換しない。DF/PF/RTEの物理biasをengine丸め誤差と混ぜない。

| 項 | 扱い・必要な評価 |
|---|---|
| b_state | N2の保存保証 `≤0.000000781` をそのまま再利用。solve/eigh/残差・gapの新評価なし |
| b_gate | 実Qiskit局所binary64 matrixと、同じgate・exact stored angleの理想matrixとの差δをintervalで検証。全局所gate集合と最大列長で一様合成。**実数値はUNKNOWN** |
| gate適用の積和丸め | Fractionの実/虚部によるexact局所contractionなので、この実装では0。NumPy/Statevectorの丸めを0と仮定した意味ではない |
| 初期norm | v/||v||をtarget stateとする。Bornの分子/分母で||v||²が正確に相殺するので、sqrtの近似を導入しない |
| 最終norm | 近似gate productを適用したjoint vectorをBorn比で正規化する差を4Eに含める。未知項として捨てない |
| b_Born | 全平方和・加算・除算をFractionで行うため算術丸めは0。正分母と厳密な[0,1]検査が必須。float probabilityへ変更した場合は保証無効 |
| b_grid | 独立uniform k∈{0,…,2^53−1}、U=k/2^53、U<p_plusという**条件付き**モデルで座標誤差≤2/2^53。RNG独立性・fresh bit実装自体は未確認 |
| その他 | 同じ保存Qとgate列・ordering・phaseのbindingを必須とする。未知gateや変更matrixでSTOP。実機・noise・compileは対象外で、compile誤差0の実機保証とはしない |

N2で未保証だったgate/Born項は、異なる丸め実装へ適当に小さいallowanceを付けるのでなく、上記の具体的な実装で扱う。
N3で高速な浮動小数点kernelへ変更する場合、このexact算術の0項と本証拠は移植できない。精度・費用・実時間の優位は未評価。

## 局所gateの検証

復元対象basisは保存RZ/XXPlusYY。half-stepとevent/wrapperにはRZ/RZZ・ordinary controlled RZ/RZZ、CZ、Phase、H、Sdgを使う。
既存registryと同じ `qiskit.quantum_info.Operator(gate)` で**局所1–3qubit**matrixを得る。Qiskit1.3のcontrolled RZZの定義展開もこの局所matrixに含めて課金する。新しいnative compileではない。
各binary64 matrix成分はexact rationalとして固定し、解析的な同じgateの成分を既存mpmath1.3.0 iv100dpsで包む。
半角の除算もexact rational angleから行い、controlled gateではcontrolが局所least-significant bitである同じembeddingを使う。
XXPlusYYのβとqubit順序は固定Qiskit1.3 SOURCEに合わせる。[公式v1.3定義](https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/qiskit.circuit.library.XXPlusYYGate)とinstalled SOURCEを確認した。

成分差のreal/imag endpointの最大絶対値をr_ij,s_ijとし、
δ_g = upper sqrt(Σ_ij(r_ij²+s_ij²))
とすれば、Frobenius norm上界なのでoperator norm差もδ_g以下。
既存interval helperの外向き256bit endpointを用いる。point値でintervalを置換せず、非有限・未対応function/gateでSTOPし、precision retryはしない。
[mpmath intervalの基本規約](https://mpmath.org/doc/current/contexts.html#arbitrary-precision-interval-arithmetic-iv)を使う。同機能がexperimentalであることを記録し、使うsin/cos/sqrt・四則演算を固定版の人工検査で確認する。形式検証済み数値ライブラリという認定ではない。

## 全supportを覆う有限inventoryと列長

有限集合Gには以下を含め、unique gate名・exact parameter hexで重複を除く。同じkeyでmatrix hashが異なる場合はSTOP。

1. 保存full basis（D含む）・全saved support basisの全operationと各inverse。
2. 各D blockを既存 `_append_block` で半stepとして一回生成した局所gate列、および既存 `_append_aggregated_phases` のscalar/identity phase列。
3. n=0/2のrotation angle、符号±、support長1/2について既存 `_append_central_application` が生成するcontrolled rotation/product。
4. n=2のevent phase πと最大2個の負product signによるPhaseGate。SOURCEと同じfloat加算順でπ、π+π、(π+π)+πを含める。tailはidentity抽出済みなので他のidentity phaseは中央にない。
5. HとSdgのwrapper gate。

D block jの半step gate数をd_j、scalar列長s、保存basisの最大operation数をBとする。全q1・両軸について
L_max = 2 Σ_j d_j + s + 3(2B+2) + 1 + 3
で抑える。D forward/reverseは同じ半step、中央は最大3 applications、各applicationはenter/exit basisと最大2 CZ、集約phaseは最大1、wrapperは最大3 gate。
既存のbasis reuse/cancellationは列を短くするため、このno-cancellation capを超えない。run threshold=1は列長によるbasis選択だけで、新しいgate型・angleを導入しない。

従ってδ*=max_{g∈G}δ_gとすれば、全positive-label path・X/Yの全gateを覆う。
これは232^3等のpath全列挙や少数pathの経験的一致を用いない構造的包含である。
実行時も、gate key・matrix hash・列長・global phase/classical scopeを検査し、未認証variantは拒否する。
N1 zero-grid labelの再正規化、control変更、代表pathだけの最大誤差を一様上界とすることはしない。

## 誤差の合成と正規化の根拠

U_jをideal local gate、Ghat_jを固定binary64局所matrixとし、||Ghat_j−U_j||≤δ*。
全U_jがunitaryなので、積の摂動を順に展開すると
E = (1+δ*)^L_max−1
が全joint vector作用差の上界となる。adapterはこの式を既存directed intervalで上側評価し、巨大なexact分子/分母の指数を保存しない。
E<1を要求すれば、normalized入力に対する計算出力wのnormは少なくとも1−E>0。
理想出力uとの距離がE以下なら、| ||w||−1 |≤Eより、
|| w/||w||−u || ≤ ||w−u|| + |1−||w|| | ≤ 2E。
よってnorm1のZ observableに対する座標差は4E以下。初期vのnormはBorn比で相殺し、最終正規化の効果はここで課金済みである。

したがって、このexact kernelと条件付きuniform-wordモデルに対し
b_engine ≤ 0.000000781 + 4E + 2/2^53
を取得できる。gate生成・作用・正規化・Born・gridの必要な項は上の表に対応する。
各path/axisの同じ一様上界なので、確率和1のN1 actual Qで平均しても同じ上界が使える。
未知の実δ*・L_max・Eを今回代入していない。総b_engineは現在UNKNOWN。
草案の1/32、ε/α、train/pilot/cal/prod数は採択・受理閾値にしない。

## 別途許可が必要な限定科学実行（提案）

**次の作業だけを一回実行する別承認が必要。N3の許可ではない。**

| 項目 | 入力・関数・回数 |
|---|---|
| 固定入力 | N1 PREPARATION/POLICY/CONFIG/LAW/RESULT/SOURCE、N2 STATE_RESULT・保存STATE_VECTOR・spectral witnessのidentity。NPZはloadしない |
| 実行SOURCE | 本adapterの公開固定commitとPROFILE_LOCK。実入力前にclean worktree・全profile・旧payload bytesを照合 |
| 関数 | `qualify_fixed_bundle(root,fresh_output)` を**一回** |
| 内部の復元 | `from_fixed_bundle`一回。保存named parameter・local matrix/basis hashの対応を確認。DF preparation/diagonalizationなし |
| D block局所列 | N1の保存D block4件に対するhalf-step `_append_block` 各一回、aggregate scalar列一回。全interrogation列はbuildしない |
| gate評価 | 上記の全保存basisとinverse、固定D/RTE/wrapper variantをinventoryし、unique gate keyごとに`certify_gate`一回。個数は保存catalogから有限に決まり、実数を結果に記録する |
| 一様合成 | `SavedEngine.qualify(0.000000781)`内のδ*・構造的L_max・E・b_engine評価を一回 |
| **0回を維持** | N1 preparation/CDF、state solve/eigh、実state norm/信号/作用評価、実path build/全列挙、RNG/shots、compile/費用、CV variance/N |
| 出力 | fresh directoryのSOURCE、UNIFORM_ERROR（D half-step/phase列・cap根拠含む）、GATE_CERTIFICATES（exact matrix pair・hash・interval誤差）、STOP |

N2の保証済みstate/spectral witnessを再利用し、state vectorやgapから新しい上界を再計算しない。今回未許可の `probability` を実state/pathに呼ぶ必要は、この一様qualificationにはない。
実行中、identity/profile/復元matrix/selector不一致、未対応gate、非有限interval、E≥1、出力衝突でSTOP。threshold緩和、代替kernel、basis再生成、科学retryをしない。
数値取得後も、結果保全・commit/push・remote内容照合でmandatory STOP。N3・精度/予算採択・C1解凍は自動実行しない。
この限定評価の数値が得られ、総engine誤差と実行モデルを採択した後に、測定facade・独立stream・条件付き取得・予算等を含むN3 scopeを別に判断する。
