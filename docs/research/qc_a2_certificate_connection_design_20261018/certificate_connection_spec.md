# A2 固定DFモデル：認証情報接続仕様

2026-10-18 JST。**設計のみ。`DESIGN_READY_FOR_IMPLEMENTATION_PLANNING`。** 本資料は後続実装の入出力・検査仕様であり、certificate、採択済み科学実行契約、A2採択記録ではない。実入力は未選定、全認証値は未生成。新しいRF/EW2定理・証明を構成していない。

[SOURCE/API接続表](source_api_connection.md)／[比較・費用仕様](comparison_and_cost_design.md)／[判断用ハンドオフ](decision_handoff.md)／[証拠索引](evidence_index.json)を相互参照する。

## 1. 正本と対象の限定

基点は[preflight `c62fe99d…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/c62fe99d29e7edcaf0cb655d789d297890709a29/docs/research/qc_a2_practical_preflight_20261017)。そこで確認済みのAPI機能・不足は再調査せず、接続仕様へ具体化する。数学は[EW2監査 `db181958…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/db1819587df36ee8833b37227ca0083faa0408ec/docs/research/qc_relative_form_ew2_step_audit_20261017)、RF0/RF1 `051df466…`、K0 `58d86727…`、K0.5 `01bdc56a…` の固定記録を使う。

今回の暫定前提は、保存binary64係数を正確な数として解釈した

\[
H_{\rm DF}=cI+Q+\sum_{\ell=1}^{r}\lambda_\ell A_\ell^2,
\qquad Q=\sum_{pq}q_{pq}a_p^\dagger a_q,
\quad A_\ell=\sum_{pq}(g_\ell)_{pq}a_p^\dagger a_q
\]

と固定N sectorである。c・λは実、q・gはHermitian、全項number-conservingである。spin sector、一般SOS/BLISS、平方内shift探索は初期設計の対象外。これは元電子Hamiltonianの保証ではない。元integralや核反発の由来が不明でも固定DF演算子の数学的対象は定義できるが、原分子への対応は `UNVERIFIED_ORIGINAL_MODEL_CORRESPONDENCE` として残す。

初期の単位規約はc・q・λをHa、gとAを無次元と宣言する。別の因子scaleを使うデータは、単位/scaleを証明可能に写す明示adapterなしで受け入れない。factorの再scaleでλやgを数値的に作り直す処理を暗黙に入れない。u_ellは無次元、q上下端・L/U/W/f/ΛはHa、K_SOSはHa²、Cbar2はHa³、hはHa⁻¹、ηと確率は無次元である。

採用する認証方法は、K0の既知scalar PSD化と、標準的なCAR・三角不等式・変分原理・正確な係数演算を使う。新しい有限step保証やPSD分割最適化を研究するものではない。以下に示す粗い経路は、認証手順を閉じるための方法であり、鋭い値やstep優位を予測するものではない。

## 2. 基本データ型とidentity

### 2.1 型

科学的実数をJSON浮動小数点で受け渡さない。次は文書上の型定義であり、実行用schema/runnerの実装ではない。

| 型 | 表現・検査 |
|---|---|
| `Nat` / `Int` | 十進整数文字列。先頭の不要な0・`+`なし。Nat≥0。無制限精度の整数として読む |
| `Rat` | `{num:Int, den:Nat}`、den>0、gcd=1、0は0/1。単位は外側Quantityで指定 |
| `ComplexRat` | `{re:Rat, im:Rat}`。Hermitian対角のimは厳密に0 |
| `Interval` | `{lo:Rat, hi:Rat, lo_closed:bool, hi_closed:bool}`、lo≤hi。無限endpointは専用`POSITIVE_INFINITY` tagを使い、数の∞を混入しない |
| `Quantity` | `{value:RatまたはInterval, unit:HAまたはHA2またはHA3またはINV_HAまたはDIMENSIONLESS}` |
| `Digest` / `ObjectRef` | SHA256小文字64桁、対象種別、固定commit/blobまたは保存bytesのdigest。ファイルpathだけをidentityにしない |
| `MatrixRef` | shape、dtype、endian、C-order規則、raw payload digest、正確な係数列のdigest、係数意味論version。配列payloadは別保存可能 |
| `WitnessRef` | 証拠種別、親object IDs、証拠payload digest、生成SOURCE/version、検証SOURCE/version、根拠文書・式、生成/検証費用ref |
| `Verdict` | §9のenum、理由code、対象IDs、不成立/欠落した検査、部分証拠ref。単一boolのPASSだけにしない |

有限binary64は、sign/exponent/fractionを整数としてdecodeし、subnormalを含め正確な有理数へ写す。複素値は実部・虚部を別decodeする。NaN/±Infは拒否。+0/−0は数学的0へ正規化するがraw bytes hashは保持する。このdecode自体にfloat加減算・`real_if_close`を用いない。入力とdtypeが一致しないときは推測変換しない。

CARは`{a_p,a_q†}=δ_pq`、その他の反交換子0。canonical occupation basisは`|n〉=(a_0†)^n0 … (a_(n−1)†)^n(n−1)|vac〉`、整数bit indexは`Σ_p n_p 2^p`と定義し、a_pのoccupation符号は低いmode indexの占有prefixから取る。行列は明示したmode順の[p,q]。OpenFermion/Qiskit等の外部vector・gate順はこの整数規約と一致すると仮定せず、permutation/bit mappingをadapterへ必須記録する。今回既存APIのbit orderingを修正・再検証したわけではない。

### 2.2 内容identityの階層

`model_id → sector_id → partition_id → pf_id → certificate_bundle_id → comparison_id` を親参照付きで固定する。

| object | hashへ含める内容 |
|---|---|
| `model_id` | 規約version、mode数、c/q/全λ/gの正確な係数、block順序、fermion mode順序、CAR convention、energy unit、固定DFを対象とする宣言 |
| `sector_id` | model_id、固定N、mode数、occupation-bit/JW ordering、number-conservationの宣言文法version。検証witnessはID形成後にbindする |
| `partition_id` | 上記IDs、全F_jの式と順序、全scalar shift、L、宣言するcap値と取得method ID。証拠のdigestは後でbundleへbind。cap変更でも別partition certificate identityにする |
| `pf_id` | partition_id、表示順の対称S2積、時系列との対応、stageの正確な1/2・1、absolute scalar復元規約、定数shiftを含む定義 |
| `certificate_bundle_id` | 全上位IDs、L/U/W/f/Λ/K_SOS/Cbar2の値と証拠digest、算術規約・method/dictionary versions。検証報告と費用は後付けenvelope |
| `comparison_id` | bundle_id、d・α_spec等の共通task条件、対象保証version、情報共有policy、step enclosure規則、比較/費用scope |

canonical内容はUTF-8、ASCIIの固定key、key辞書順、空白なし、整数/有理数の上記正規形、重複key拒否、配列順序保持でserializeする。content hash欄自身や後付けの計測時刻はそのobjectのhash入力から除外し、envelopeに置く。raw archive hash・生成commit・元データref・変換履歴は別のprovenance envelopeに保存する。意味の異なるobjectを同じhash namespaceへ混在させない。

hash依存を循環させない。model/sector/partition/PFの宣言を先に作り、それらのIDを参照する証拠payloadを生成し、値と証拠digestからbundle core IDを作る。その後にbundle_idを参照するverifier report・費用・生成時刻のenvelopeを付ける。個別証拠は親の宣言IDsを持つが、まだ形成されていないbundle_idや自分自身のhashをpayloadへ含めない。費用ref・検証報告ref等は、下記各Certificateの保存recordに含めてもidentity coreからは除外する。envelopeの完全性は別の保存file digestで追跡する。

同じ表示でもmode permutation・JW bit順・λ/g再配分・scalar移動・PF順序・capが変われば再bindが必要。他bundleから数値だけを流用しない。`U`等が数学的にはPF順序に依存しなくても、比較bundleへは同じcontextを検証して明示的に再bindする。

## 3. モデル受入と既存APIへの入口

入力`ModelIntake`は、raw archive/array identities、DFHamiltonian field対応、constant provenance、元データref（不明なら理由付きnull）、変換履歴、単位、mode orderを必須とする。固定DFのconstantに既に含まれる核反発・電子scalarをもう一度加算しない。核反発の分解が不明ならその内訳を創作せず、c全体を対象scalarとする。

generatorは正確な係数表を出力し、verifierはraw bitsから独立にdecodeしてq=q†、全g=g†、c/λの実性、shape/rank/mode数、全blockの存在を検査する。**tolを設けず厳密比較**する。失敗時は`INVALID_MODEL_FOR_GUARANTEE`。既存`hermitized()`を通して黙って修復しない。対称化した別対象が必要なら、変換内容とrawからの差を明示する別modelとし、後続でその選択の承認を受ける。

規約の混用は次の段階で止める。

- 初期入力は`DFHamiltonian`の**全constant/全onebody＋signed squares**。`DFModel`のfield名だけから内容を推定しない。
- `df_hamiltonian_to_model`のfull値payloadと、`df_decompose_from_integrals`のspatial/correction-only payloadは、異なるtag付きadapterとする。既存callerに不具合があると認定しない。
- `df_truncation_value`は元データ変換のmetadataで、cや`constant_correction`へ加算しない。固定DFモデル内では「モデルに追加する誤差項」でもない。
- spin-orbitalとspatial chemist、tensor factor・normal orderingを明示する。初期certifierは再DF分解をせず、既に固定されたq/g/λを正本とする。
- `select_blocks`によるH_Dだけの入力を、全Hの保証へ流用しない。全block inventoryと元model_idが一致することを検査する。

`SectorWitness`は0≤N≤mode数と、全基本項がa†a・その平方・identityである文法証拠を持つ。古典indicesの抽出・基底外skipで代替しない。native個別gate全てのN保存を主張する必要はないが、logical fragment全体と採用したGaussian変換の演算子対応は別途必要である。

## 4. PSD分割、L、f_j、Λ

### 4.1 capの生成と独立検証

実装計画を未指定の数値eigensolverに依存させないため、最初に**正確な係数三角上界**を定義する。これは既知の`||a†_p a_q||≤1`と三角不等式の適用である。新しい相対形式保証ではない。

Hermitian一体係数xに対し、generatorは

\[
s_x=\sum_{pq}(|\operatorname{Re}x_{pq}|+|\operatorname{Im}x_{pq}|)
\]

をRatとして出力し、verifierが全項を再加算する。これを一体演算子の安全なcapとして使う。複素絶対値のfloat sqrtは不要。固定Nでより鋭くできる可能性は残るが、このfull-Fock capもN sector上で有効である。粗さを隠さない。

一体Qの区間を`q_minus≤Q≤q_plus`、平方生成子を`−u_ell I≤A_ell≤u_ell I`とする。必ず実装できる初期経路は`q_minus=−s_q, q_plus=s_q, u_ell=s_gell`。検証付きspectral enclosure等の鋭い経路を追加する場合は、証拠種別と独立checkerを別versionへ固定し、同じ情報を対照へも公開する。未認証のeigh出力をこの証拠にしない。

### 4.2 K0のsigned PSD化を保存形式へ写す

次は[K0 §6.1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md)の既知の外側scalar shiftの帳簿である。

| fragment | 保存する正確な式 | cap f_j | Lへのscalar寄与 |
|---|---|---|---|
| 一体 | `F_0=Q−q_minus I` | `q_plus−q_minus` | `q_minus` |
| λℓ≥0 | `F_ell=λell Aell²` | `λell uell²` | 0 |
| λℓ<0、μℓ=−λℓ | `F_ell=μell(uell² I−Aell²)` | `μell uell²` | `−μell uell²` |

\[
L=c+q_{\rm minus}-\sum_{\lambda_\ell<0}\mu_\ell u_\ell^2,
\qquad H_{\rm DF}=LI+\sum_jF_j,\qquad \Lambda=\sum_j f_j.
\]

verifierはcap証拠、符号、全block inventory、identity・一体・平方の係数帳簿を正確に再構成する。PSD性にはHermitian Aとそのcapを使い、signed項をそのまま正の平方と扱わない。Lはこの帳簿と一致する**正確なRat**で保存する。Lだけを下向きに丸めると和の恒等式が壊れるので禁止。下向き表示が必要なら表示専用enclosureとし、certifierへ戻さない。別のLを採用するなら対応するPSD identity補償を分割に明示し、全ID・K_SOS・Λを更新する別設計が必要となる。

上向きcapを使う場合も、その値をnegative-square fragmentのidentity係数に**実際に使う**。細いcapで定義したfragmentへ粗いLだけを後付けしない。零λ/零capのfragmentは初期inventoryへ残し、無断で削除・再順序化しない。Mはこのinventory長。一体のみでもM≥1。uとqのcap変更は同じ回路へscalarを戻せても証拠identityを変える。

`LowerCertificate`はmodel/sector/partition IDs、c、q上下端、全u・λ・符号、scalar寄与、L、cap/PSD/恒等式witness refs、算術規約、units、generator/verifier versionsと費用refを保存する。`FragmentCapCertificate`は全F_j式・f_j・共通sectorと検証結果を保存する。Λは検証済みf_jの正確な和であり、既存phase-rate helperとは別物である。

### 4.3 指数化とscalarを二重に数えない

各fragmentを`F_j=B_j+s_j I`と記録する。一体はB=Q、正符号平方はB=λA²、負符号平方はB=λA²、外側sは上表から決まる。既存Gaussian/DF blockはBの指数化に対応し、PSD用sはscalar台帳へ置く。

正stage S2で各fragmentの総時間係数が1であることを検査し、`L+Σs_j=c`を確認する。物理回路は元Qとsigned平方のS2にcのphaseを付ける既存経路を使える。F_jのscalarも個別に付ける経路を使うなら、外側Lと合わせて同じcになることを検査する。両経路を混ぜてscalarを二重加算しない。controlled時は既知scalarもancilla相対位相または明示したframe復元となり、黙って消さない。

数値Gaussian対角化・回転角を実装する際の残差/合成誤差は、上のexactなlogical恒等式と別の`CircuitRealizationRecord`へ置く。今回その誤差を評価・認定していない。

## 5. 変分上界UとW

初期のtrial state型は、mode orderingが既知の**単一occupation determinant**とする。入力`TrialDeterminant`はoccupied modeの昇順list、mode数、N、bit/JW convention、全係数のmodel_idを持つ。occupied listは空/満占有も型として許し、重複・範囲外・個数不一致を拒否する。具体的な占有を今回選ばない。結果を見てtrialを選び直すpolicyも採択していない。

generatorは既知CARのoccupation-bit符号でQと各Aをこのdeterminantへ作用させ、疎な遷移振幅表を正確なComplexRatで保存する。verifierは同じraw係数とoccupied listから独立に遷移を再構成し、正規化=1、N保存、遷移先の重複集約を確認する。Hermitian Aについて既知の期待値恒等式を使い、

\[
U=c+\langle D|Q|D\rangle+
\sum_\ell\lambda_\ell\|A_\ell|D\rangle\|^2
\]

を正確なRatとして検証する。signed λをabsへ置換しない。これが同じsectorでの変分上界。ground solverやFCIを必要とせず、HF energy metadataも入力値にしない。既存HF determinantの候補listを使う場合も、占有情報と上記expectationを再検証する。

`UpperCertificate`はtrial型、occupied list、basis convention、norm、遷移表ref、q/g/c/λ refs、expectation内訳、U、全context IDs、generator/verifierと費用refを持つ。より一般のtrial stateはこの初期仕様へ黙って追加しない。正規化や期待値の検証付き拡張として別の承認・実装仕様を要する。

この初期経路では各一体作用をmode対で列挙でき、cap/Hermiticity/単一determinant期待値の入口は概ねO((r+1)n²)個の係数処理と疎振幅集約で済む。これは算術操作とデータ構造の静的説明であり、整数bit長・sorting/hash・出力費を含むruntime見積りではない。多体ground diagonalizationをUの必須前処理にしない。

verifierが同じcontextのLとUを受理した場合に限り、`W=U−L`を正確に形成する。U<Lなら算術/identity/証拠の不整合として拒否。Wを表示用float差分で作らない。任意にG≤ΛIから区間を狭める場合も、Wそのものと有効幅を別fieldに保存し、両方式へ同じpolicyを適用する。初期比較のWはU−Lを維持する。

## 6. K_SOSの証拠

[K0 §4 Eqs.(6),(8),(9)](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md)の**raw辞書**を正本とする。添字は固定PF表示順で1,…,Mへ写す。保存fragment IDがF_0を含んでも、辞書用位置と明示的に対応させる。

- 各j<M、k>jに`κ_jjk += 1/24`。
- 各j<M、k,l>jに`κ_klj += −1/12`。
- 同一tupleへの寄与は正確なRatで集約するが、別tupleを無断に同一視しない。raw件数とdictionary digestを保存する。

各raw項(a,b,c)で、K0のB加算式

\[
|\kappa_{abc}|\{f_b\sqrt{f_af_c}(e_ae_c^T+e_ce_a^T)
+f_c\sqrt{f_af_b}(e_ae_b^T+e_be_a^T)\}
\]

を使う。sqrtには`SqrtUpperWitness{z:Rat,u:Rat}`を用意し、verifierがz≥0、u≥0、u²≥zを正確に確認する。generatorは整数/有理数のroot enclosureでuを作れる。未採択のpや丸めtolを今回設定しない。zeroはu=0の証拠を使える。上界の粗さ・bit長を記録し、float sqrtをそのまま証拠としない。

streaming行和なら、各項でr_aへ二つの寄与、r_cへ第一寄与、r_bへ第二寄与を加える。添字が一致するときも、それぞれの加算を実施する。対角で二回になる寄与をdeduplicateしない。`K_SOS_upper=max_a r_a`を出力する。この値はexactなK0行和値の認証上界であり、平方根を含む真の行和値そのものと区別する。

`KSOSCertificate`はdictionary version/固定K0参照、全f、全sqrt witnessのdigest、行和vector、maximum、位置対応、全context IDsと費用refを持つ。verifierはraw辞書・witness条件・全行和を再生成して確認する。行和streamはK0既存のO(M³)算術/O(M)作業領域。保存witness/transcriptを全て展開すればその出力はO(M³)にもなるため、式・入力digest・重複しないsqrt witnessを保存して再生成する方式を初期候補とする。bit計算・certificate保存量は別会計である。

可換性や相殺を使ってraw辞書を短縮する場合は、完全な同値証拠とversionが別に必要。最初のK_SOSはrawのままでよく、強い対照側の同じ構造利用を禁じない。新しいK_SOS理論を作らない。

## 7. 強い交換子対照Cbar2の証拠

[固定EW2監査 §5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/db1819587df36ee8833b37227ca0083faa0408ec/docs/research/qc_relative_form_ew2_step_audit_20261017/mathematical_audit.md)の同順序S2について

\[
C_2=\sum_{j<M}\left\{\frac{\|[F_j,[F_j,R_j]]\|}{24}
+\frac{\|[R_j,[R_j,F_j]]\|}{12}\right\},\quad R_j=\sum_{k>j}F_k
\]

を対象とする。Cbar2はenergy³。1/24と1/12を交換しない。scalar shiftは交換子から厳密に落とせるが、元H/controlled phaseから削除したという意味ではない。

generator/verifierは、少なくとも次の二つの既知norm評価を同じ情報から許す。

| 手順 | generatorの出力 | verifierの検査・価格 |
|---|---|---|
| centerしたcapの標準norm積 | F_j−f_jI/2、R_j−(Σ_{k>j}f_k)I/2のcapsを使い、標準`||[A,B]||≤2||A||||B||`を二回適用した上界 | 正確なcap/係数、同じ交換子、centerの合法性を再計算。元の粗いuncentered capだけに対照を固定しない |
| 構造を保持した正確な係数展開 | 上の**各grouped double commutator全体**を正確なCAR/JW規約でPauli係数へ展開・集約し、係数絶対値和のnorm上界を出す | 同じraw coefficientsから正確な積と符号を再構成し、identity/cancellation/Pauli supportを照合。Hermitian double commutatorのPauli係数は実であることを確認。full-space上界としてsectorにも有効。展開膨張・bit長・作業領域を計測 |

構造経路は、個別tripleのnormを先に足して相殺を消す方式と区別する。Pauli文字列の可換判定・exact zero・同じ係数の集約を利用する。既存float normal-order/JW結果だけを証拠にしない。実装はまだなく、exact algebra generatorと独立checkerが必要である。既存ライブラリは仕様照合・将来のproposal生成の参考としてのみ利用できる。

各grouped normには、検証済みの二経路の小さい上界を使える。どの経路も結果と費用を保存し、同じ情報を両方式へ公開する。利用可能なsector witnessやより鋭い既知norm情報が別途取得できるなら、同じ価格で追加する。初期二経路をbest-known norm法とは主張しない。

`CommutatorCertificate`はgrouped式のdigest、順序、各normの全候補/採用値、構造証拠digest、center caps、正確な1/24・1/12和、Cbar2、共通IDs、費用refを持つ。構造経路が未完なら粗いcapだけの値を保存してもよいが、`STRUCTURAL_BASELINE_INCOMPLETE`を付け、比較上の優位認定は保留する。都合のよい粗さを選んでrelativeを勝たせない。

## 8. 共通bundleの生成・検証・保存

`CertificateBundle`はcontext、ModelIntake、SectorWitness、Lower/Upper/FragmentCap/KSOS/Commutator Certificates、L/U/W/全f/Λ/K_SOS_upper/Cbar2、error/scalar/cost ledger refs、個別verdictを持つ。値は全て証拠から再構成可能とする。certificate生成と、生成された証拠を再検証する処理を分離する。generatorの成功flagをverifierの入力真実として扱わない。

文書上の処理順は次のとおり。

```text
固定rawと規約を受け取る
  → raw identityとexact decode、Hermiticity/number-conservationを検査
  → 宣言済みcap手順とK0 shiftでPSD台帳を提案
  → 独立verifierが台帳・L/f/Λを受理または拒否
  → 明示trialからU証拠を提案し、独立に検証、同一contextでWを形成
  → K0 raw行和と強いgrouped交換子証拠を生成・別々に検証
  → 全contextの一致と証拠完全性を検査してbundleをseal
  → 承認済みtask条件がある将来段階でのみstep比較へ渡す
```

これは実行可能な科学コードではない。入力や途中証拠が不成立でも、partial bundleと理由、生成/検証ログ、SOURCE・入力hash、費用を保存する。verifierは再DF分解・ground solve・simulationを暗黙fallbackとして呼ばない。検証に必要な演算も将来の科学実行範囲へ明示し、dry-runという名前で勝手に実データへ適用しない。

## 9. verdict・誤差台帳

| verdict | 意味と下流の扱い |
|---|---|
| `VERIFIED_EXACT_FOR_FIXED_DF` | 指定の正確な係数と論理モデルについて全必要証拠をcheckerが受理。checker正当性・実装検証の前提は別に残る |
| `INVALID_MODEL_FOR_GUARANTEE` | Hermiticity・文法・sector等の前提不成立。tolで修復しない |
| `IDENTITY_MISMATCH_INCOMPARABLE` | model/sector/分割/PF/task条件が異なる。他の値を自動流用しない |
| `INVALID_CERTIFICATE` | 誤ったscalar、上界の逆向き、不正なsqrt witness、期待値や辞書の再計算不一致等 |
| `INCOMPLETE_CERTIFICATE` | 必須証拠またはそのchecker未実装/未完。未取得は0と書かない |
| `UNRESOLVED_ARITHMETIC` | 指定したenclosure/bit予算で境界を分離できない。閾値緩和・精度変更は結果を見て自動で行わない |
| `TECHNICAL_ERROR_OR_INTERRUPTED` | I/O・資源・SOURCE/依存・保存等の技術失敗。部分成果を保全してSTOP |

bundleとstep比較のverdict、科学的GO/NO-GOは別fieldである。型schemaの受理やhash一致を`VERIFIED_EXACT_FOR_FIXED_DF`としない。異なる実装経路の数値一致だけでも証拠にはならない。

`ErrorLedger`は、(i) raw binary64→正確係数：exact decode、(ii) 元分子→固定DF：範囲外・未認証、(iii) certificate arithmetic：exact Ratまたは検証済み有向enclosure、(iv) Gaussian/回転合成/controlled実装、(v) state、(vi) QPE readoutを別項にする。固定DFをtargetと定義したことによる(i)のdecode誤差ゼロを、(ii)のモデル誤差ゼロと表現しない。

certificate coreはexact Ratを既定の実装設計とし、BLAS/IEEE/trig/logの未検証仮定へ数学的certificateを依存させない。平方根・step rootは検証可能な有理数enclosureを使う。これはplatformの認定ではない。generator/checkerのcorrectnessは未検証であり、今回はcertificate自体も生成していない。

## 10. 保証・QPEへの出口

bundleが受理され、同一のd>0、0<α_spec<1が後日採択された場合のRF0/RF1/EW2・centered Cbar2の接続は[比較仕様](comparison_and_cost_design.md)で定義する。inverse Gやground vectorを古典証拠の入力にしない。exact ground accessは条件付き理論入力であり、determinantを使ったU証拠とは役割が異なる。

実QPEには固定hのS2 powers、absolute scalar/frame、branch/alias、readout、合成・状態準備の品質/費用が別に必要。certificateだけでそれらを満たしたとしない。**実装計画へ進める設計は完成したが、実装・certificate生成・科学実行は未承認。Mandatory STOP。**
