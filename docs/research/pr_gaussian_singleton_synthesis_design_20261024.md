# Gaussian singleton合成：選択部分空間のGivens合成を推奨

**推奨は、選択した軌道部分空間だけをrectangular Givensで合成する候補A。support2を優先し、support1も同じ方式の対照とする。** 現行の「選択列を固定→残りの列を補完→正方unitary全体を分解」を、既存Slater determinant合成のGaussian部分で置き換える設計である。新しいPRアルゴリズムの認定ではなく、既知合成法を固定controlled演算子へ接続する実装改善候補である。**実装可能性をSOURCEから確認した段階で、同値性の数値検査・資源削減量は未測定。採択・実装・科学実行は未承認。**

## 対象と、今回確認した範囲

ここでF1は**残差fragment F1**。[寄与分析](pr_algorithm_improvement_phase2d_f2_attribution_20261024.md)のcommitは `41c4e79f1c58af0f2be634f46dbeba8dcd36acc5`、測定SOURCEは `9468c7089727f5648d9f59cddbaebe0a42f5f0a0`。
4 mode・2粒子・occupied(0,1)、L_D=1、残差density-square 2fragmentの固定合成Hamiltonian。geometry／AO basis／分子DF rankは非該当、rank探索なし。T=1/5、q=1、δ=1/5の一点、r=2、h=1/10、K=0、faithful identity tail。
Qiskit1.3.0／Nature0.7.2、opt1、seed17、rz/sx/x/cx、topology-free、approximation_degree=1、qubits_initially_zero=False、位相保持修正版compilerが基準である。

今回行ったのはSOURCE・既存文書・保存atom metadataの読取り、既知手法との技術的照合と設計だけ。数値再集計、行列生成・分解、回路の実装・build／compile、simulation、law生成、新grantは0件。既存SOURCE・raw・grant・STOPには書き込んでいない。

以下は既存寄与分析の転記で、再計算ではない。PSでframe成分が1個の群にはIFとFPが含まれる。

| fragment F1のsingleton群 | 確率 | 条件付きRZ／CX／depth | RZ寄与 |
|---|---:|---|---:|
| support1 | 0.205245 | 68.826819／32.066843／87.218050 | 14.126339 |
| support2 | 0.184951 | 84.011489／37.499467／104.013145 | 15.538043 |

これらは共通D・相手event・prep/readoutも含むfull-wrapper費用で、Gaussian部分単独の価格ではない。どのGivensやcompiler passが何割を占めるかは保存集計から分離されていない。

## 現行の費用発生機構

`BasisProvider.basis(singleton=True)` は `support_completion` で選択列を保持し、残りを標準ベクトルの二重再直交化で埋める。その行列を `native_gaussian → U_to_qiskit_ops_jw_givens → BogoliubovTransform(U.T)` へ渡す。
Natureのnumber-conserving分解は正方行列全体を消去し、RZとXXPlusYYを出す。XXPlusYYの公式定義にもCX・RY・位相操作があり、native化でRZ・CX・depthになる。補完の古典CPU処理そのものが量子費用なのではなく、**選ばれた補完行列の分解を往復すること**が費用機構である。[現行回路SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/circuits.py#L14-L94)、[固定BogoliubovTransform](https://github.com/qiskit-community/qiskit-nature/blob/0.7.2/qiskit_nature/second_q/circuit/library/bogoliubov_transform.py#L196-L259)、[XXPlusYY定義](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/circuit/library/standard_gates/xx_plus_yy.py#L113-L165)。

- support1は選択1列のnumber parityを回転する。現行補完はその列を個別に固定するため、補空間の自由度をnative費用へ最適化していない。
- support2は選択2列のjoint parityを回転する。中心部にparity compute/uncomputeが加わり、Gaussian変換も異なる。現行は各列の保持を要求するが、**このZZ演算子に必要なのは2列の張る部分空間**であり、列を個別に固定する条件は強い。
- F1の元frameは(2,3)、(0,2)、(0,1)のplane rotationから作る。ただし回路は補完後の行列を再分解するので、元rotation数を実際のGivens数や追加費用と同一視できない。
- 隣接同frame runのfull basis共有・singletonのsupport completion・actual adjointは既にある。K0/r2では短いpair内の共有であり、新候補は **run長1のbranchだけ**を対象とする。

## 同じoperatorを保てる自由度

元の一粒子frameをU、選択supportをS、選択列をV=U[:,S]、P=VV†とする。Γをnumber-conserving Gaussian表現として、対象のunsigned operatorは

\[
Q_S=\Gamma(U) Z_S\Gamma(U)^\dagger
    =\exp\!\left(i\pi\sum_{ij}P_{ij}a_i^\dagger a_j\right).
\]

support1では選択列の位相を変えてもPは同じ。support2ではV→VW（Wは2×2 unitary）でもPは同じで、**個々のnumber演算子を変えても、その2軌道のjoint parityは変わらない**。補空間内のunitaryも自由である。この標準的な共役・parityの関係を、候補接続の同値条件に使う。実際の候補回路について成立を検査した、という意味ではない。

この自由度はSlater determinant合成で使われるoccupied-spaceのunitary自由度と対応する。既知の基礎は [Jiang et al., arXiv:1711.05395v3 §III](https://arxiv.org/html/1711.05395v3#S3)。本候補はその再利用であり、独立新規性を主張しない。

## 候補A：rectangular GivensのGaussian部分を使う

**一つに絞るならこれを推奨する。** support1はrank1、support2はrank2として同じ処理を使う。ここでrankは合成用の選択部分空間の次元で、実験の2粒子stateを変更する意味ではない。

1. 元の選択列Vだけから、Slater APIのcreation coefficient行列Vᵀを渡す設計とする。今回のframeは実数。転置／共役・gate方向・little-endianの対応は、下記のprojector条件と独立referenceで確認する。丸めた新frameを作らない。
2. 固定Nature0.7.2の `_prepare_slater_determinant_jw` は、占有初期化Xのprefixと、その後の隣接XXPlusYYのGaussian部分を生成する。**初期化prefixを使わずGaussian部分Gだけを取り出す**限定adapterを設計する。任意のXを名前で一括削除する処理にはしない。
3. 内部基準supportをT={0,…,|S|−1}とし、Gの一粒子作用Wが `W Π_T W† = P` を満たすことを合格条件にする。support1はZ₀、support2はZ₀Z₁を中心部の基準にする。
4. 時間順に **G† → controlled exp(−i sθ Z_T) → G**。sは既存の係数符号、θは当該保存caseの値。actual adjointを同じGから作る。Gをancilla制御しない。
5. 保存atomのword・support・basis key・係数・case IDは元のSのまま保持する。Tは回路recipe内部の参照先で、physical selected operator Q_Sを変えるものではない。wrapperのstate preparation X0X1も変更しない。

固定helperは選択行間の回転を古典処理だけで行い、その後に必要なGivensを出す。support2ではこの行間自由度を利用できること、未選択列の正方completionを合成しないことが改善期待の理由である。**Givens段数が減り得ても、compiled RZ/CX/depthの削減は保証されない**。元の特殊角・zero構造・full-wrapper fusionが有利な場合もある。[固定helper SOURCE](https://github.com/qiskit-community/qiskit-nature/blob/0.7.2/qiskit_nature/second_q/circuit/library/initial_states/utils/givens_rotations.py#L27-L81)、[Slater APIの行列規約](https://github.com/qiskit-community/qiskit-nature/blob/0.7.2/qiskit_nature/second_q/circuit/library/initial_states/slater_determinant.py#L44-L77)。

**必要な新規接続は小さいが、現行APIへそのまま渡すことはできない。** `U_to_qiskit_ops_jw_givens` はsquare matrix専用である。将来のadapterはsingletonについてGと`reference_support=T`を返し、中央生成の参照先も接続する必要がある。共有runは既存full frameへ戻す。cacheは元frame・元support・policyを保持し、同じTだから異なるoperatorを同基底として融合しない。

### 他の候補と制約

| 候補 | 具体的方法 | 扱い |
|---|---|---|
| B：中心と可換な端部因子だけを除く | 例えば現行basisがB=G_b ΦでΦがdiagonal orbital phaseなら、ΦはZ_Sと可換なのでB Z_S B†=G_b Z_S G_b†。中央側のΦ/Φ†を合成前に省ける。共通HamiltonianのD部分を除く意味ではない | 小さい接続で済む可能性。ただしcompilerが既に消している可能性があり、削減量不明。今回の最小対照には追加しない |
| C：同support／補空間内の因子をまとめる | support2の内部U(2)、補空間内unitaryが中央に接する因子として取り出せれば共役から除ける | 候補Aが利用する自由度。途中に別因子が挟まる場合、単に未選択modeを触るgateだからという理由では削除できない |

G†とGは中央回転を挟むため丸ごと相殺できない。固定2粒子sectorでだけ成り立つ「選択2modeを補集合2modeへ置換」は採用しない。一般にはtotal parity因子が必要で、契約は全Fock空間のoperator一致だからである。

このfixtureには構造的な簡約候補もある。元frameはU=R₂₃ R₀₂ R₀₁なので、`IIIZ`では中央Z₃と可換な右端R₀₁・R₀₂を共役から外せ、`IIZZ`では右端R₀₁を外せる。これは演算子等式から分かる候補で、native費用は未計測。現行completionも同じ簡約を実現している可能性があり、新しい削減とは認定しない。逆に候補Aのcanonical Tへ揃えることで不要なmode移動が入る場合もあるため、**Tの選択は費用に無関係ではない**。実装時の固定規則を明示し、wire置換を無料として扱わない。元factorを直接使う場合もR₀₂は非隣接fermionic操作であり、裸のqubit XXPlusYYへの置換ではJW parityを保てない。こうしたfactor単独の候補を最小対照へ自動追加しない。

## 再利用できる実装と未確認事項

| 資産 | 再利用範囲・注意点 |
|---|---|
| `BasisProvider`、`build_event_sequence`、`append_central`、wrapper | singleton branchだけに候補を接続。演算子順序・符号・identity/scalar位相・sampling law・full wrapperを維持 |
| Nature0.7.2 Slater helper／`givens_matrix` | 固定版のGaussian suffixと数値kernelを再利用。private APIなので版を固定。状態準備の1状態一致だけでは今回のoperator検査の代わりにならない |
| OpenFermion1.6.1 `givens_decomposition` | 同じrectangular分解の公開APIによる代替。角度・qubit順・位相をQiskitへ接続するadapterが別途必要で、初回候補は増やさない。[固定SOURCE](https://github.com/quantumlib/OpenFermion/blob/v1.6.1/src/openfermion/linalg/givens_rotations.py#L232-L286) |
| 既存support completion／WP06-b | 現行baselineとして再利用。既に列制限を導入した資産であり、「未使用列を補完するだけ」を新改善として重複実装しない。[履歴の固定SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_structure_pilot.py#L59-L121) |
| `reference.py`、`gates.py`、F2 raw/reference/CSV | 元frameの独立CAR/exterior reference、既存operator/signal gate、保存baseline費用、位相保持compileを利用。新規Hamiltonianやbaseline全件再生成は不要 |

未確認なのは、actual Gがprojector条件を固定許容差で満たすか、conventionと符号・相対位相が一致するか、compiled費用が減るかである。Nature helper内の`np.isclose`は既存の数値受理閾値の保証ではない。tiny entryの省略がP0 gateを通らなければ不合格とし、許容差緩和や位相fitで救済しない。共有環境・依存packageの変更は不要であり、行わない。

## 最小対照実験案：PSの固定48 wrapper

**以下は次回承認の候補であり、実行登録・実装・buildを今回行っていない。** 候補A一種類だけ、同じPS arm・同じ保存F2 contextで比較する。

- 対象atom：保存PS index20～25のfragment F1成分。support1は`IIIZ / IIZI / IZII`、support2は`IIZZ / IZIZ / IZZI`。wordは保存SOURCEと同じqubit順表記。
- 相手：保存PS index0の`P:collect:IIII`と、index2の`P:collect:IXXI`。identityだけでなく、主要寄与のFP境界を含める。費用を見て相手を探索する選び方ではない。
- 両方のoperator順序とX/Yを使う：**6 atom × 2 partner × 2 order × 2 axis = 48 full-wrapper要求**。support1／support2は各24要求。独立なevolutionは24、候補basisは6種類としてcacheできる。単位が違う件数を科学call総数へ合算しない。
- baselineは同caseの保存raw IR・独立reference・compiled資源CSVを再利用し、再build／compileを省く。θは当該PSの保存raw IRへbindingし、丸めたλやF1/K2の角度から再構成しない。公開lawにθの独立欄がないため、保存QPYからの値の読取り・対応確認も将来の接続範囲に含める。元caseと角度をbindingできなければ実行開始しない。

| 検査・計測 | 次回の合格条件・比較方法 |
|---|---|
| basisの条件 | 既に作る6個のGから一粒子作用とactual adjointを確認し、元VのPとWΠ_TW†を照合。候補の状態準備一致だけに依存しない |
| raw／compiled operator | 各48 wrapperを保存された元の独立referenceへ照合。4 system mode全16次元／ancilla込み32次元を対象とし、sector-only・global phase alignmentを使わない |
| controlled位相・信号 | 既存raw operator 1e-10、raw X/Y signal 2e-10、compiled operator/signal 5e-10、rtol=0を維持。control枝の相対位相、negative係数、元のscalar/identity位相を含めて検査 |
| 回路費用 | 同一compiler接続で候補48要求をcompileし、caseごとにbaselineとの差を保存。RZ・CX・total/RZ/CX/quantum-only depth・sizeを別々に比較し、恣意的scoreにまとめない |
| 古典費用 | basis生成・compile・検査のCPU/wallとcache実績。恣意的なCPU/RAM/wall capを設けず、必要な独立処理は並列化。共有環境を変更しない |

保存referenceのhash・SOURCE・input・角度が一致すれば既存二精度照合済みtargetを使える。追加の高精度reference再計算が必要と分かった場合は、件数・範囲を確定して承認に含める。今の資料だけで追加計算を暗黙に許可しない。

この48要求は接続の正しさと限定contextでの資源差を見る最小案であり、全F2の期待改善率やPPを上回るかは判定しない。高費用のsingleton経路を改善できる兆候が得られて初めて、全該当trajectoryの置換・期待費用再集計を別途検討する。

## 次の担当と承認・mandatory STOP

次は**ユーザー／GPT**が候補Aと48-wrapperの限定対照案を採択するか判断する。進める場合に必要なのは、(1) singleton専用adapterの実装承認、(2) 内部基準support T・JW順序・保存角度・case・referenceと固定許容差・compilerをbindingした科学実行範囲の明示承認である。過去のgrant／STOPを再利用せず、新SOURCE・専用outputと失敗時STOP条件を固定する。予測削減率や新規性を事前認定しない。

**今回の成果は静的調査・候補設計のみ。公開・保全照合後mandatory STOP。** 新しい実装、科学計算、回路build／compile、数値再集計、γ・N研究、重要科学レビューへ自動移行しない。
