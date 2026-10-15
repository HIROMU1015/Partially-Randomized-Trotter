# Native DF理論ゲート数モデル A/B — 2026-10-16

実装・数式・人工検証の技術資料。**実H6/H8への適用、科学的採用、実行認可ではない。**
モデルAは構造最適化を無効にした構成、Bは明示したGaussian境界相殺だけを含む。
固定した分解のRZ/CX数を数える。既存コンパイラの出力や母平均を保証しない。

- [独立カウンタ](../../src/trotterlib/native_df_gate_model.py)
- [人工テスト](../../tests/test_native_df_gate_model.py)／[標準ライブラリ専用runner](../../scripts/check_native_df_gate_model_synthetic.py)
- [検証receipt](../../artifacts/resource_applicability/track_a_native_df_gate_model/2026-10-16/synthetic_validation_v1.json)
- [固定SOURCEとSHA-256](../../artifacts/resource_applicability/track_a_native_df_gate_model/2026-10-16/source_provenance_v1.json)
- [作業範囲・保全監査](../../artifacts/resource_applicability/track_a_native_df_gate_model/2026-10-16/technical_audit_v1.json)

## 1. SOURCEと再利用

Repository: `HIROMU1015/Partially-Randomized-Trotter`。
起点branch `track-a-h6-fixed-delta-t-dependence-readiness-20261015`、commit
`043f58ac9336d662a3181caff508526568f1e57a`（以下043）。
独立worktree/branch `track-a-native-df-theoretical-gate-model-20261016`。

| 固定版・path | 確認した構造／今回の利用 |
| --- | --- |
| [043: df_trotter/ops.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/src/trotterlib/df_trotter/ops.py) | `DiagonalEvolutionPrimitives`、one-body/DF squared、Gaussian runtime operations。既存primitiveリストをduck-typeで受ける。DF係数の再生成はしない。 |
| [043: df_partial_s2.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/src/trotterlib/df_partial_s2.py) | `runtime_basis_operations`、inverse/diagonal/forward、S2 half、構造上界。新しいpreparationは作らない。 |
| [043: df_rte_circuit.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/src/trotterlib/df_rte_circuit.py) | event/requestとbasis registryの契約。新規requestやtrajectoryを生成しない。 |
| [043: df_rte_qiskit.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/src/trotterlib/df_rte_qiskit.py) | product→rotation、basis ID/hash照合、identity barrier、列末尾のphase集約。`existing_rte_occurrence`は既存event metadataを受ける。 |
| [043: rte.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/src/trotterlib/rte.py) | canonical finite-RTEの全偶数次数、component独立選択、係数符号とevent符号。標準ライブラリによる重み評価と条件付き期待値を実装。 |
| [043: rte_compiled_cost.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/src/trotterlib/rte_compiled_cost.py) | transpile後のRZ/CX/depth/size等。呼ばない。理論分解と異なる評価対象として残す。 |
| [043: product_formula.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/src/trotterlib/product_formula.py) | unmerged S2、Yoshida S4係数。負時間を保持するscheduleのみ独立実装。 |
| [87: ax2b_native_df_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87b6b9f46b7e8ba9abe132ee21478c114b6f46f7/src/trottertracks/resource_applicability/ax2b_native_df_v5.py) | ordinary/symmetric-directional controlled diagonal、単一scalar ancilla P、Hadamard wrapper。**043には存在しない別SOURCE**。read-only参照し、srcへコピーしない。 |
| [87: ax2a_control_plan.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87b6b9f46b7e8ba9abe132ee21478c114b6f46f7/src/trottertracks/resource_applicability/ax2a_control_plan.py) | forward uncontrolled / reverse directional、およびglobal Yoshida schedule。 |

最新版共通指示は[server原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b78e5f39917d92d6c60694d3e0b0d06394d7eb5/docs/research/provenance/research_project_instructions_server_original_20261012.md)
（50,268 bytes、SHA-256 `709524f856a3117c85042309c390a3697d9b2c8c8d79af6b6c290e9d20814bcf`）を参照。
主担当が実装と技術チェックを実施した。別担当の独立科学レビューではない。

既存コードをimportして科学依存を起動する代わりに、既存producerの出力を受ける小さなadapterを追加した。
`basis_from_runtime`は既存の `(gate, ordered_qubits)` を読み、数値パラメータと名前だけをIRへ写す。
`diagonal_from_primitives`は元のRZ/RZZリストとphaseをそのまま使う。
`existing_rte_occurrence`は保存済みeventの順序・角度・符号を使う。
これらの実分子への接続検証は今回実施していない。

## 2. 何を厳密に数えるか

| レベル | 本資料の意味 |
| --- | --- |
| primitive exact count | 与えられた数値metadataから作る抽象gate列の命令数・gate別histogram。選択したA/B構成に対してexact。 |
| decomposition-specific count | `explicit_rz_sx_x_cx_v1`で各primitiveを置き換えた整数RZ/CX/SX/X数。分解を固定した条件付きexact count。 |
| structural bound | ゼロ係数も最大数に含める高水準命令上界。lowered RZ/CX上界とは呼ばない。 |
| expected count | 指定した有限確率法則と境界規則に対する期待値。確率評価・phase費用の限界を別に表示。 |
| compiled count | 実環境、basis、optimization level、seed、routing等に依存する別の測定量。今回は取得も再解析もしない。 |

回路IRは小さなgate記述であり、Qiskit circuitではない。state準備、measurement、classical処理、connectivity routing、fault-tolerant合成は含めない。
wrapperはancilla H・任意Sdg・Hまでを数える。測定を付けるなら別に1 measurement/shotを記録し、RZ/CXへ換算しない。
RZ数は回転命令数であって、回転合成精度を指定したT/Toffoli数ではない。

## 3. 明示分解 profile

以下は**時間順**。global phaseもIRと人工unitary比較で保持する。

- `P(a) = exp(ia/2) RZ(a)`、`Z=P(pi)`、`Sdg=P(-pi/2)`。
- H: `RZ(pi/2), SX, RZ(pi/2)`、global phase `pi/4`。
- SXdg: `SX, X`。
- RZZ(a): `CX(u,v), RZ_v(a), CX(u,v)`。
- CRZ(a): `RZ_v(a/2), CX(c,v), RZ_v(-a/2), CX(c,v)`。
- CRZZ(a): `CX(u,v), CRZ(c,v,a), CX(u,v)`。
- CZ: `H_v, CX(c,v), H_v`。
- RXX(a): Hを両qubitへ、RZZ(a)、Hを両qubitへ。
- RYY(a): SXを両qubitへ、RZZ(a)、SXdgを両qubitへ。
- XX+YY(theta,beta): `RZ_0(beta), RXX(theta/2), RYY(theta/2), RZ_0(-beta)`。

最後の式は[IBMのXXPlusYYGate行列定義](https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.0/qiskit.circuit.library.XXPlusYYGate)と同じqubit順序を使う。
各分解は1〜3 qubitの独立な解析行列oracleと比較した。global phaseを捨てた一致ではない。

| primitive | raw | RZ | CX | SX | X |
| --- | ---: | ---: | ---: | ---: | ---: |
| RZ / P / Z / Sdg | 1 | 1 | 0 | 0 | 0 |
| H | 1 | 2 | 0 | 1 | 0 |
| RZZ | 1 | 1 | 2 | 0 | 0 |
| CRZ | 1 | 2 | 2 | 0 | 0 |
| CRZZ | 1 | 2 | 4 | 0 | 0 |
| CZ | 1 | 4 | 1 | 2 | 0 |
| RXX | 1 | 9 | 2 | 4 | 0 |
| RYY | 1 | 1 | 2 | 4 | 2 |
| XX+YY | 1 | 12 | 4 | 8 | 2 |

このGaussian分解は効率最適を主張しない。beta=0でも分解に含まれる2本のRZを数える。
Qiskit Natureの[0.7.2 num-conserving Bogoliubov source](https://github.com/qiskit-community/qiskit-nature/blob/0.7.2/qiskit_nature/second_q/circuit/library/bogoliubov_transform.py)
はRZとXX+YYを使うことを確認したが、**これはH6/H8の実行環境versionの同定ではない**。
runtime gate listを正本とする。opaque dense `unitary`や未知gateは費用を推測せず拒否する。

## 4. DF diagonalと制御

primitiveを `(phase=phi, rz, rzz)` とし、exact nonzero RZ項数をa、RZZ項数をb、
`g=1[phi != 0]` とする。もとのappend処理と同様、diagonalではexact zeroだけを省く。
微小値を閾値で落とさない。Gaussianの明示zero gateやRTEのzero-angle rotationは保持する。

Gaussian forwardをG、inverseをGinvと記す。どちらもuncontrolled。
各blockの時間順序は `Ginv, diagonal, G`。
Gaussian費用は `C(Ginv)+C(G)` として別々に計算し、一般には単に `2 C(G)` としない。

| diagonal mode | raw | RZ | CX | phaseの扱い |
| --- | ---: | ---: | ---: | --- |
| uncontrolled | a+b | a+b | 2b | circuit global phase phi |
| ordinary | a+b+g | 2a+2b+g | 2a+4b | ancilla P(phi) |
| directional | 3a+5b+g | a+b+g | 2a+4b | ancilla RZ(2phi) |

directionalのRZ項は `CX(c,u), RZ_u(-a), CX(c,u)`。
RZZ項は `CX(u,v), CX(c,v), RZ_v(-a), CX(c,v), CX(u,v)`。
このblockは `diag(D^-1,D)` を実装する。
S2で前半をuncontrolled、逆順後半をdirectionalとすると、ancilla0の積はI、ancilla1の積は所定のS2になる。
一般にblockが非可換でも順序が逆なので成立する。S4は各S2 pieceへ適用する。

数え上げ自体は元producerの浮動小数演算順序を尊重する。例えば代数的にRZ係数をまとめた式でzero判定し直さない。
人工DF例 `eta=(1,2), lambda=.4, time=.3` は占有数上の
`exp[-i .12 (n0+2n1)^2]` と制御phaseまで一致した。これは人工例であり分子結果ではない。

元native v5の高水準block上界も別APIで保持する。systemサイズn、basis命令数m、
pair上限p（one-bodyでは0、DF squaredではn(n-1)/2）に対し、

\[
U_U=2m+n+p,\quad U_O=2m+n+p+1,\quad U_D=2m+3n+5p+1.
\]

`structural_instruction_bound`はこれらだけを返す。`U_D`等をRZ数へ流用しない。

## 5. PFとwrapper

S2は各outer stepで全blockのforward halfと逆順reverse half。
`pf_schedule`はtime=T/q/2とordinary/ordinary、またはuncontrolled/directionalを返す。
S4は `S2(w1*T/q), S2(w0*T/q), S2(w1*T/q)`、
`w1=1/(2-2^(1/3))`, `w0=-2^(1/3)/(2-2^(1/3))`。
piece間やstep間を事前にmergeしない。負のtimeを絶対値へ変えない。

native参照版のscalarは全trajectory末尾の単一ancilla `P(-c*T)`。
legacy partial builderのstepごとのconstant/extracted-identity Pを加算して二重計上しない。
APIの`scalar_phase`はこの**実際にemitする角度**であり、scalar係数そのものではない。

cosine(X) wrapper増分は `(raw,RZ,CX,SX,X)=(2,4,0,2,0)`。
sine(Y)はSdgを加えて `(3,5,0,2,0)`。測定とstate準備は別。

## 6. モデルA/Bと境界

**A:** 各nonidentity application/blockでGinvとGをemitし、optional Gaussian reuseを無効にする。
phaseの列末尾集約はnativeの構成規則としてA/B共通で保持する。
したがって既存requestでreuseが有効なら、そのraw命令数をAが再現するとは主張しない。

**B:** 隣接する `..., G, Ginv, ...` だけを相殺する。
basis ID、source hash、全操作名・数値パラメータ・順序・qubit配置、uncontrolled scopeの一致を要求する。
同じID/hashなのに操作が違う場合は相殺を避けて進むのではなくエラーにする。
supportが同じという理由で同じbasisとはみなさない。
中央演算は交換しない。同軸回転のmergeや近似identity除去は今回採用しない。

RTE builderに合わせてidentity applicationはphaseが0でもactive basisを閉じる。
occurrence末尾でもbasisを閉じ、集約Pを置く。basis-planがapplication/order/historyで変わる場合は、
その選択を追加状態に持たない限り下記iid DPを適用できない。
`existing_rte_occurrence`のadapterは元basis ID/hashが一致するregistered basisに限定し、
support-restricted certificateの検査を代行しない。

deterministic PFのBは隣接block/piece/stepで同じbasisなら相殺可能。
PR whole-trajectory期待値のBは**各prefix half・tail occurrence・outer-step境界を閉じる**限定版。
`partial_prefix_circuit`がその固定費用を作る。tailを取り除いたPFのB費用を代用すると、
存在しない隣接境界を相殺してしまうため使わない。
この閉境界版Bは、さらに境界をまたぐ合法な相殺を加えたモデルより保守的になり得るが、
Qiskit compiled費用に対する大小関係は保証しない。

## 7. finite-RTE期待値

ここでtauは既存finite-RTE distributionに渡す**event用のパラメータ**である。
分子のTそのものと同一視しない。既存normalizationとtauの接続は後段で照合する。

全偶数次数 `k=0,2,...,K` の重みと確率は

\[
w_k={|\tau|^k\over k!}\sqrt{1+(|\tau|/(k+1))^2},\qquad
b_K=\sum_k w_k,\qquad p_k=w_k/b_K.
\]

tau=0ではp0=1。他の正重みをsample未観測という理由で除外しない。
`canonical_order_probabilities`はlog重みをbinary64評価し、得られた値をFractionで正規化する。
**真の超越関数確率に対する丸め誤差certificateではない**。underflowで正の次数を失うとエラー。
上流で高精度評価した確率を使う場合も、全法則をFractionとして渡せる。
選択済み法則に対する数え上げは有理数演算で行う。

component iの確率をpi_iとする。条件付きkでk個のproductと1個のrotationを同じcomponent法則から独立選択する。
物理的適用順序は**productsの保存順序、最後にrotation**。
rotation角は `theta_k=atan(tau/(k+1))`、central rotationは係数符号s_iを含む `RZ/RZZ(2*s_i*theta_k)` を制御化する。
productのZ/ZZはsupportごとにCZを置く。負係数productは別にpiの位相を足す。
event符号は `(-1)^(k/2)`。identity product/rotationにも元の位相を残す。

### Aの線形期待値

phi集約Pを除いたcomponent費用を `cP(i,k), cR(i,k)` とすると、

\[
E[C_{A,event}^{body}]=\sum_k p_k\sum_i\pi_i
\{k\,c_P(i,k)+c_R(i,k)\}.
\]

nonidentityでは各費用にGinv/Gを含める。identityでは中央gateとbasis費用は0。
1 occurrenceにr eventあるbody費用はr倍。A実装はこの線形性を使い、event列やr回のDPを作らない。
0次でもrotationが1個あり、2次はproduct2個、4次はproduct4個＋rotation1個となる。

### Bのbasis遷移DP

状態sはactive basis（またはnone）。nonidentityの次basis=bでs!=bなら、旧sを閉じbのinverseを開く。
s=bなら境界費用0。identityはsを閉じnoneへ。各applicationの中央費用を加える。
状態確率P_sと、その状態へ到達する費用の重み付き和M_sを保持する。
選択確率v、遷移費用dに対して

\[
P'_{s'}\mathrel{+}=vP_s,\qquad M'_{s'}\mathrel{+}=v(M_s+P_sd).
\]

次数ごとにk product→1 rotationの遷移を合成し、p_kで混合する。r event後にactiveを閉じる。
全component列の指数的列挙は不要。状態数はregistered basis数+1、遷移数は概ね
`O(r * sum_k(k+1) * component数 * (basis数+1))`。
中央費用と単一gate loweringをcacheする。Fractionの分母成長は別の計算費用となる。
default最大1,000,000 DP遷移、超過時は近似へ黙って切り替えず失敗する。
今回の人工suiteは1秒未満のためprocess並列化は行わず、科学計算の性能改善を主張しない。

### ancilla集約P：exact部分とbound部分

元builderはevent signを先に足し、その後application phaseを順に足す。
最後にfloatの `phase != 0.0` の場合だけPを**1個**emitする。2piでもmodulo削除しない。
明示event列のカウンタはこの順序と規則を保持する。

identity rotationがない場合（またはtau=0）、全phaseは非負のpiの和である。
`p_plus=sum_{s_i=+1} pi_i` とすると、1 eventの位相が0である確率は

\[
z=\sum_{k:\,k/2\ {\rm even}}p_k\,p_{plus}^k,
\qquad E[\#P_{occurrence}]=1-z^r.
\]

ここまでexact（指定法則に対して）。identity rotationの `-s_i*theta_k` がある一般の場合は、
piや角度の相殺・浮動小数加算を安易に判定しない。
`0 <= E[#P_occurrence] <= 1` と報告する。
したがって**CX/SX/X期待値はexact、raw/RZ期待値は最大1/occurrenceの幅**。
`ExpectedCost.exact`はこの幅が0のときだけtrue。
symbolic phaseの相殺とruntime floatのzero判定を同一視しない。

### R=qrとwhole trajectory

q個の閉じたoccurrenceが同じlaw、各r event、R=qrの場合、

\[
E[C_{trajectory,axis}]=C_{prefix,closed}+C_{scalar}+C_{wrapper,axis}
+q\,E[C_{tail(r)}].
\]

`expected_partial_trajectory`に`partial_prefix_circuit`の固定費用（scalarは一度だけ）を渡す。
phaseがboundならwhole-trajectory raw/RZ幅は最大q。CX/SX/Xは変わらない。
返値のphase_probability欄は**1 occurrenceのP確率**でありq倍した確率ではない。
occurrence間の独立性は元sampling lawとの接続時に確認する。異なるlawならoccurrenceごとの期待値を加える。
この量は理論分解の期待値であり、compiled RZ母平均やrare-event compiled費用の上界ではない。

## 8. 人工検証と実施範囲

再現command（依存追加不要、実分子データ不要）:

```bash
python3 -B scripts/check_native_df_gate_model_synthetic.py
```

27 tests PASS、failure/error/skip 0。詳細な名称、Python version、実測秒数、source SHAはreceiptに保存。
検証は主担当によるlocal technical checksであり、immutable CIや独立科学レビューではない。

検証には次を含む：固定lowering全種類の解析unitary、Gaussian inverse、
制御DFのoccupation eigenvalueとancilla位相、非可換S2/S4のordinaryとdirectionalの一致、
raw/分解費用/structural boundの区別、wrapper差、basis/barrier/hash拒否、
次数0/2/4と正負component、複数eventのDPと小規模全列挙の一致、
identity phase bound、確率10^-18の次数4を落とさないこと、underflow/遷移上限の拒否。

Python標準ライブラリだけを使用。既存`trotterlib/__init__.py`は科学依存を読み込むため変更せず、
standalone runner/testは`importlib.util.spec_from_file_location`で新moduleだけをloadする。
通常のpackage importは元packageの依存環境が必要。実環境のimport接続は未検証。

## 9. 後段の確認と比較設計（今回未実行）

1. 別途許可された小規模Qiskit環境で、runtime Gaussian gate種・順序・inverse・phase・basis hash、
   selected basis plan、control policy、scalar、wrapperを固定SOURCEと照合する。
   opaque gateならその分解profileの追加を先に検証し、経験的費用で埋めない。
2. 実分子への適用前に同じsource/input/candidate/axis、tau/K/component law、
   original normalizationとbasis-reuse requestをbindingする。新しいsamplingやcompileを自動実行しない。
3. 同じsignal target、bias、経験的数値allowance、normalization、shot会計から得た**同じN_axis**を固定し、
   `G_theory=sum_axis N_axis E[C_theory,axis]` と
   `G_compiled=sum_axis N_axis mean(C_compiled,axis)` を比較する。
   まず回路費用モデルだけを替え、shotモデルまで同時に変えない。
4. A/B各modelのcandidate rankingとcompiled ranking、PR/PF比、identity phase幅、希少次数の寄与を別々に表示する。
   explorationと独立confirmationの標本をpoolしない。compiled finite-sample winnerをpopulation winnerと呼ばない。
5. theoretical expectationに全次数を含めても、未観測次数のcompiled母平均は認定されない。
   分解profileの差、限定最適化、routing、compiler残差を科学的説明の候補として区別する。

未解決：一般identity phase期待値の厳密計算、history依存のbasis plan、閉境界をまたぐ追加相殺、
多数eventでのFraction DPの実用性、実Qiskit接続、depth/routing、state準備Pとの統合。
これらは勝手な近似で埋めず後段に残す。既存のsignal定義・normalization・shot数・科学的比較条件は変えない。

今回のモデルは明示された構成の技術基盤として完成。コードのために同じ科学reviewを繰り返す必要はない。
ただし新しい分解profile・境界scopeを用いてPRの優位性や候補選択を科学的に主張する前には、
GPTが比較契約への採用と主張範囲を判断する必要がある。本資料はその判断を代行しない。

実H6/H8数値評価・保存値再解析・native build/compile・sampling・サーバー接続・新grant・ledger変更・原稿改訂はすべて0。
**MANDATORY_STOP / H6_NOT_AUTHORIZED / H8_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION / next_stage_authorized=false**。
この未認可表示は今回の新しい適用に関するもので、別環境の既存認可や稼働jobを失効・変更しない。
