# Track B／M1-S Gate 0 v5：tail局所誤差上界のGPT独立数学レビュー

**レビュー日**：2026-10-16（JST）  
**実施者**：GPT-6  
**区分**：Level 2／既採択Gate 0科学scope内での独立数学・数値誤差契約レビュー  
**開始承認**：ユーザー「レビューを開始して」  
**判定**：`CONDITIONAL_MATHEMATICAL_ACCEPTANCE_OF_TAIL_LOCAL_BOUND_AND_V5_DESIGN; PLATFORM_COVERAGE_NOT_CERTIFIED; EXPERIMENT_NOT_EXECUTED; USER_ADOPTION_AND_SEPARATE_EXECUTION_AUTHORIZATION_REQUIRED; MANDATORY_STOP`  
**権限**：本書はGPTの独立科学的判断である。ユーザーの最終契約採択、科学実施承認、grant／marker発行、GitHub更新ではない。

---

## 0. 要旨と最終判定

**新規のtail局所係数誤差上界**

\[
\boxed{\alpha_R=\gamma_{713+r}S_R},\qquad
\gamma_k=\frac{ku}{1-ku},\quad u=2^{-53},\quad
S_R=\sum_{\ell\in R}|\lambda_\ell|\left(\sum_{p,q}(|\Re g_{\ell pq}|+|\Im g_{\ell pq}|)\right)^2,
\]

に関して、**指定された複素スカラー演算・`math.fsum`の条件付き誤差モデルの下では、導出の主要段階は数学的に整合し、Gate 0 v5に採用する候補として条件付き受理を推奨する**。固定SOURCEの語支持上界、局所群和の項数、DAGの積・和の累積gamma index、実部射影・非identity tailへの制限を独立に再構成したところ、本レビューで採択を阻む反例や指数の欠落は見いだされなかった。

加えて、著者が提示した比率条件 `S_R/lambda_hat <= 8`、`L/lambda_hat <= 2` における `e_W < 2^-24`、`e_v < 2^-24` の**解析的十分条件も、記載された保守的な2冪評価に整合している**。これは実H4でv5予算値を評価したという意味でも、全品質gateが通るという意味でもない。

ただし、次は**受理しない／未確定**とする。

- 実機・固定Python／NumPy／BLASの算術がすべて宣言された `gamma`／`fsum`／underflowモデルで覆われるという**プラットフォーム数値certificate**。
- v5の**実入力での**`e_W,e_v`算定・しきい値達成・W/v/ρのcorrectness／結果／分母精度・rho headroom判定。
- 数値oracleが**厳密な全Fock基底固有状態**であること、非縮退性、pair-sum条件、Jensen floorの無条件適用。
- finite-step／finite-RPE／shot／native回路／最終総資源の優位性、あるいはM1-Sの科学的GO／NO-GO。
- 係数上界の導出そのものが独立した新アルゴリズム・独立した出版上の新規性であるという判断。

**推奨する次段階**：ユーザーがこの限定的な条件付き受理と `CONDITIONAL_ENGINEERING_NOT_CERTIFIED` の区分を理解した上で、v5最終契約の採否を**別途明示的に判断**する。採択後も科学実行には新しいv5専用の**別途明示承認**が必要。v4の `TECHNICAL_STOP` と一回実行済みのauthorization／markerを保全する。

---

## 1. 対象・固定証拠・取得状況

Repository：`HIROMU1015/Partially-Randomized-Trotter`。

| 資料 | 固定識別子・レビューでの取扱い |
|---|---|
| [v5実験契約候補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47da54279838a6b3e1497f36204ba7e53c4857f9/docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v5_candidate_20261016.md) | 公開commit `47da54279838a6b3e1497f36204ba7e53c4857f9`、本文を取得 |
| [条件付き証明原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5a185cc0c3aae0e477bfc52b93061e86feb7dec/docs/tracks/algorithm_codesign/m1s_gate0_tail_local_bound_proof_20261016.md) | 証明固定commit `b5a185cc0c3aae0e477bfc52b93061e86feb7dec`。v5 refからも同文書・内容を確認 |
| [v5 SOURCE: `budget.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69de022dc8142f703423b1ce0bd42f05e123954e/src/trottertracks/algorithm_codesign/m1s_gate0_v5/budget.py) | SOURCE固定commit `69de022dc8142f703423b1ce0bd42f05e123954e` の算定式、Fraction／上向き変換、global/local台帳、law伝播を読解 |
| [v5 SOURCE: `science.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69de022dc8142f703423b1ce0bd42f05e123954e/src/trottertracks/algorithm_codesign/m1s_gate0_v5/science.py) | 係数生成 `_sum/_combine/_multiply/_one_body_pauli`、`prepare/split`、状態・再構成・law／metrics／STOP経路を読解 |
| [v5 SOURCE: `control.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69de022dc8142f703423b1ce0bd42f05e123954e/src/trottertracks/algorithm_codesign/m1s_gate0_v5/control.py) | 固定17項目、proof/source/入力/runtime、採択契約と別authorizationのguardを読解 |
| [v5 machine契約候補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47da54279838a6b3e1497f36204ba7e53c4857f9/artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/execution_contract_v5_candidate.json) | 採択・coverage・実行はfalse、17項目とSOURCE/proof hashを確認 |
| [モデルJSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5a185cc0c3aae0e477bfc52b93061e86feb7dec/artifacts/track_b_m1s_gate0_tail_bound/2026-10-16/v1/tail_local_model_candidate.json) | gamma index、support count、適用仮定を確認 |
| [SOURCE–証明対応表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47da54279838a6b3e1497f36204ba7e53c4857f9/artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/source_proof_alignment.json) | AST／参照SOURCE比較の静的PASSとその限界を確認 |
| [公開receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47da54279838a6b3e1497f36204ba7e53c4857f9/artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/publication_receipt.json) | 122件SOURCE検査、96件文書監査、計算0等はCodexの監査記録として扱う |
| [v4 one-shot結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c937ed37e73bb9b0bb39f2fa7116d255a3215269/docs/tracks/algorithm_codesign/m1s_gate0_v4_one_shot_results_and_stop_20261016.md) | `TECHNICAL_STOP`、W/v/ρ=null、旧authorization消費済みを保持 |
| [直前の原因分析](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a058ef14d4ea162547ec73b707a4662ec8fcd7d6/docs/tracks/algorithm_codesign/m1s_gate0_precision_budget_cause_and_options_20261016.md) | v4の全H尺度過大評価、事後的感度分析を確認 |
| [v3のGPT限定レビュー原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47da54279838a6b3e1497f36204ba7e53c4857f9/docs/research/track_b_m1s_gate0_v3_gpt_limited_scientific_review_20261016.md) | 既存の品質・coverage・科学scope判断を参考。今回は新規tail上界を独立検討 |
| [統合共通研究指示](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47da54279838a6b3e1497f36204ba7e53c4857f9/docs/research/research_project_instructions_unified_2026-10-16.md) | 2026-10-16版。取得済みローカル原文も確認 |

v5 SOURCE commitのREADME内に固定した `proof_commit < SOURCE_commit < 将来のユーザー採択contract < 別実行authorization` の関係を確認した。GitHub公開ファイルは実際に取得したが、**receipt掲載の全49ファイルを今回独立再ハッシュしたわけではない**。SOURCE動作を実行していない。GitHubの内容を書き換えていない。

### 固定科学scope

H4 linear、1.00 Å、STO-3G、8 spin orbitals、保存DF rank12、保存DF順prefix G3/G6/G9、非冗長signed Pauli-tail、`cutoff=0`、HR identityをHDへ移動、保存数値sector-ground vectorによるoracle診断。主量

\[
W=1-\langle h^2\rangle_\psi,\quad v=\operatorname{Var}_P\langle P_{\mathrm{signed}}\rangle_\psi,\quad
\rho=\frac{W+v}{2W},\qquad h=H_R/\lambda_R
\]

はsmall-step leading attenuationを考えるための**保存数値oracle量**であり、true ground、finite RPE signalやnative資源の指標ではない。v4で到達した再構成等132 gateのPASSをv5全体のPASSに自動移送しない。

---

## 2. 条件付き算術仮定と証明の射程

1. `u=2^-53`、`gamma_k=ku/(1-ku)`、`ku<1`。複素乗算を8実演算の保守的な相対・絶対誤差包絡で評価し、各`_multiply` leafの誤差をbox normで加算する。
2. `math.fsum`の長さmの群和について、**独立の仮定として**絶対誤差 `<= gamma_(m+32) * sum(abs(input))` を置く。これはCPython文書そのものが全機種でこの定数の厳密な上界を保証する、という意味ではない。
3. IEEE binary64 round-to-nearest、未会計のoverflow／underflow／FTZなし、exact Hermitian入力 `one_body/g_l`、実の `lambda_l`、固定SOURCE／演算順・runtime/build/thread等を条件とする。
4. 絶対誤差のtargetは**保存binary64値を正確な入力値とみなした場合の理想Pauli係数と、SOURCEの算出係数との差**であり、分子積分誤差・DF近似誤差・状態生成誤差を含まない。

標準的なgamma記法自体はHigham–Maryの公開論文で確認できるが、同論文の確率的`~gamma`緩和は**今回使っていない**。複素BLAS、`fsum`、underflowの全実行経路はプラットフォーム上で未認証である。**条件付き数学的補題の受理**と**実実装のrigorous numerical certificate**を区別する。

外部参考：Higham & Mary, *A New Approach to Probabilistic Rounding Error Analysis*, SIAM J. Sci. Comput. 41 (2019), DOI [10.1137/18M1226312](https://doi.org/10.1137/18M1226312)；[Python 3.11 `math.fsum`文書](https://docs.python.org/3.11/library/math.html#math.fsum)；[NumPy `matmul`文書](https://numpy.org/doc/1.26/reference/generated/numpy.matmul.html)。

---

## 3. Pauli代数・支持数・群和長の独立検証

### 3.1 Pauli wordの積

単一Pauli語をグローバル位相を除いた二値ベクトル `(x,z) in F_2^{2n}` で表す。積のPauli語ラベルは`(x_a xor x_b, z_a xor z_b)`で一意に決まる。従って出力語wと左語aを固定すると、右語bは**一意**。有限支持集合A, Bからwへ寄与する組数は

\[
\#\{(a,b)\in A\times B:a\cdot b\propto w\}\le\min(|A|,|B|).
\]

この結果は乗算順を入れ替える主張ではない。位相±1/±iは積の係数へ残される。

### 3.2 8 spin orbital one-bodyの支持

`a_p^dagger a_q` はp=qなら `I,Z_p` の2語、p≠qなら端点X/Yと間のZ列から4語以内。したがって全64個のbilinearを合計した語の**union**の上界は

\[
1+8+4\binom82 = 121.
\]

一つの語に寄与するbilinearは64組を超えず、各`bilinear`辞書は語を一意keyにしている。`A_l*A_l`の**一つの語**への寄与も121を超えない。最後に各DF blockから同じ語へ寄与する項は、tail block数rを超えない。

これは`_combine`の**一つのwordに対するfsum群長**の上界であり、積全体で列挙される組数やresource capとは異なる。実測counter `27,781` を数学的な保証として代入していない点は妥当である。

**判定**：Pauli語の一意性、`121`支持、`64/121/r`群長上界は、固定された8モードSOURCEと整合。条件付き証明で使用してよい。

---

## 4. 累積gamma indexを独立再構成

スカラー複素積について `b(fl(zw)-zw) <= gamma_8 b(z)b(w)` を条件付き原始評価とする。`_multiply`は各wordで最大8回のPauli位相乗算と `phase*ca*cb` の2乗算を含み、leafの上界は **`10*8=80`** と置ける。`fsum`はgroup長mに `m+32` を加算する。

| 段階 | gamma indexの再構成 | 監査結果 |
|---|---:|---|
| JW bilinear `B_pq` | `80+(4+32)=116` | 一語あたり最大4寄与 |
| `A_l = Σ g_lpq B_pq` | `116+8+(64+32)=220` | gの複素乗算と最大64項の和 |
| `A_l²` | `2*220+80+(121+32)=673` | 同じAを二回使う相関誤差もノルムで包む |
| `lambda_l A_l²` | `673+8=681` | DF係数乗算 |
| r-block HR集約 | `681+(r+32)=713+r` | r個のblockから各wordへ高々r項 |

`gamma_k=ku/(1-ku)`の積不等式

\[
(1+\gamma_a)(1+\gamma_b)\le 1+\gamma_{a+b}
\]

は `ku<1` の条件下で確認でき、`(1-au)(1-bu)>=1-(a+b)u`から従う。Pauli convolutionについて係数mapのbox-L1がsubmultiplicativeなので、`Ahat*Ahat-A*A` の差を

\[
\left[(1+\gamma_{220})^2-1\right]G_l^2
\]

で囲い、計算側の乗算・群和誤差 `gamma_233` を加えると **`gamma_673 G_l²`** に収められる。誤差独立性や異なる乱数の仮定は必要ない。

したがって実lambda乗算、r個のblock和まで含めると、**各tailの全Pauli係数mapのbox-L1誤差**について

\[
\|\widehat C_R-C_R\|_{1,\mathrm{box}}\le\gamma_{713+r}S_R
\]

という条件付き導出を認められる。rはG3=9、G6=6、G9=3なので、indexはそれぞれ **722、719、716**。M=65536やresource cap=1,000,000をgamma指数に置き換える必要はない。

**判定**：本レビューで見つかった範囲では、indexの欠落・二乗時の相関の落とし忘れ・全H尺度の再混入はない。**原始浮動小数点誤差モデルが成立することを条件に、新係数補題を数学的に条件付き受理する。**

---

## 5. 実部射影・符号付きlaw・identity移動

保存raw `g_l`がexact Hermitianでlambdaが実数なら、数学的な`A_l²`はHermitianで、Pauli係数は実数となる。`Re`射影は `box`ノルムに関して縮小写像だから、複素係数mapから実係数への射影で誤差は増えない。

同一wordの正負係数は `_combine`で**符号付き和を取ってから**絶対値でnormalizerへ反映する。真の係数がzeroで計算係数がnonzeroとなる場合、または小さい係数の符号が反転する場合も、exactとcomputedのsupport unionでL1誤差を数える。任意の小係数cutoffは使っていない。

`rreal.pop(identity)` によってHRの非identity係数は変化しない。HDに加えるscalarの追加`fsum`誤差は**既存のglobal/phase台帳**へ残すが、identityを除いたHRのsampling lawには加えない。この分離は正しい。ただし、HDのglobal台帳を新しい `alpha_R` で置き換えることはできない。

正負signed eventを `(Pauli word, sign)` の二channelに表すと、一つのwordに対する**unnormalized mass**のL1差は `|c_hat-c|`。符号反転例でも成り立つ。normalizerの誤差 `delta_lambda=alpha_R+gamma_(M+32)L`、正の `lambda_lo=lambda_hat-delta_lambda` を前提に、

\[
e_c\le\frac{\alpha_R}{\lambda_{lo}}+
  \frac{\delta_\lambda L}{\lambda_{lo}\widehat\lambda_R}
\]

が得られる。これは真のsigned event lawと計算側の正規化lawを結び付ける妥当なL1の包絡である。

**判定**：signed event・ゼロsupport・identity移動と新係数補題は整合。`lambda_lo<=0`は従来どおり `PRECISION_UNRESOLVED`、閾値の自動緩和を許さない。

---

## 6. W・v予算と2^-24の解析的十分条件

### 6.1 Wとvの意味

保存状態に対し、理想signed Pauli lawの `h=Σ_j p_j P_j` には `||h||<=1`。したがって `0<=W<=1`。また

\[
v=\sum_jp_ja_j^2-\left(\sum_jp_ja_j\right)^2,
\quad a_j=\langle\psi|P_j|\psi\rangle
\]

は非負。実際、`P_j psi= a_j psi + xi_j`、`xi_j`はpsiに直交、という分解から Jensen の分散不等式により `v<=W` が確認できる。これは任意の正規化保存状態での代数事実であり、exact eigenstateの認定を必要としない。

`budget.py`で、coefficient由来 `e_c` にmatrix assembly・matvec・divisionの上界を加えた `y` から `e_W` を構成し、`ep`（law／確率除算）、`ea`（Pauli expectation）、`em`（mean）、最後のcentered variance加算から `e_v` を構成する。**v5では `alpha` を `alpha_R` へ取り替え、その他の伝播式をv4から保持したことをSOURCE上で確認した**。

この部分の主張は、**従前v4の条件付きforward errorモデル全体を前提にした整合性**であり、BLAS・normなど後続演算を今回すべて独立に実測認証したものではない。

### 6.2 十分条件の紙上追跡

著者の領域条件は `S_R/lambda_hat<=8`、`L/lambda_hat<=2`。両者は本レビューで新規数値評価していない。前者は科学的な新しい切り捨てgateではなく、`e_W/e_v`が一定以下となることを説明する*十分条件の領域*である。

`M=65536`、`d=256`、`r<=12`、`u=2^-53` と `gamma_k<=2ku`（`ku<=1/2`）を使うと、`gamma_(713+r)<=2^-42`、`gamma_(M+32)<=2^-35` 等の粗い上界が成立する。すると

- `alpha_R/lambda_hat <= 2^-39`、`s/lambda_hat <=2^-34`、正の分母 `lambda_lo` を確保できる。
- normalized coefficient/lawの基本誤差 `e_c < 2^-30`。
- v4から保持したmatrix assembly・matvec・divisionを含む `y < 2^-26`。
- 従って `e_W < 2^-25+2^-52+2^-36+2^-48 < 2^-24`。
- `ep<2^-29`、`defect<2^-33`、`ea<2^-37`、`delta_center<2^-28` 等を代入すると `e_v < 7*2^-27+2^-55 < 2^-24`。

最後の `2^-24` はbinary64で正確に表現でき、真の有理数予算が厳密にこれを下回れば上向きbinary64変換後も上限を超えない、という丸め上の説明も整合する。

**判定**：仮定したratio領域での**解析的十分条件は整合する**。しかし、これは「全3分割の実行gateが通る」「rhoの分母が正に分離する」「visibility改善がある」のいずれも含意しない。過去の保存値で `S_R/lambda_hat`が概ね4.6–5.4と報告された事実は参照可能だが、今回v5計算に代入したり、事後的にPASSを認定していない。

---

## 7. SOURCE接続・採択・STOPの独立監査

### 7.1 適切に分離されたglobalとlocal

`m1s_gate0_v5/budget.py`では、従来のglobal上界を保持したまま `tail_block_scales` とsplitごとの `tail_models` を構築している。`alpha_R` は保存`lambda_l, g_l`のraw値とsplitのrから **係数表生成前** にFraction演算で計算する。`global_limits`は元v4の残差・Hamiltonian再構成・phase品質を継承し、`law_limits`のみ新しい局所係数上界に置換する。

`science.py`では`prepare`でlocal台帳を作り、splitの`rcomplex→real_map→identity pop`の演算順序を保持している。各splitが独立のlocal予算を用い、結果や実測counterを公式へ入力しない構造は妥当。`W/v/ρ`未算定時の `null`／statusを保持する設計も従来と整合する。

ただし **static/AST検査122件PASSは数学証明の代替ではなく、今回の独立読解も実動作試験ではない**。

### 7.2 Guardとprovenance

`control.py`はprecision17条件・runtimeをpinし、proof固定commit、SOURCE固定commit、将来の採択済み契約、さらに別authorizationという履歴順を要求する。v4 authorityを使い回さず、新規output namespaceへ排他保存する。現候補の `adopted=false`、`coverage_model_adopted=false`、`adoption_record=null` は正しい。

旧v4の`TECHNICAL_STOP`、W/v/ρの`null`、消費済みauthorization／markerは、v5理論の採否にかかわらず変更しない。v5実行で後からv4をPASSと再分類することも不可。

### 7.3 条件付き受理後の科学的解釈

今回の導出は**実装依存の係数算術上界の精緻化**であり、tailを小さくした結果の必然的な科学性能、Pauli lawの新しいアルゴリズム特性、長step・長RPEに関する主張ではない。v4が停止した原因に対し、予算算定の方法を結果後に修正したものなので、同じH4 development条件でのv5結果は**事後的な技術改訂後の評価**として扱う。未使用系での独立transferの証拠に読み替えない。

---

## 8. 残る留保、停止条件、次段階

| 項目 | 今回の科学判断 | 次の担当 |
|---|---|---|
| tail局所係数補題 `alpha_R` | **指定算術モデル下で条件付き数学的受理を推奨** | ユーザーが採否判断 |
| 既存 `e_W/e_v`伝播と2^-24十分条件 | v4由来の誤差会計を条件として整合、解析的十分条件は妥当 | 採択後の数値correctnessで確認 |
| IEEE／BLAS／fsum／underflow coverage | **未認証。rigorous certificateとしては受理しない** | Codexは実行時に仮定と依存環境を監査し、限界を記録 |
| v5 SOURCE／static guard | 新設計と整合。現時点で実数値未評価 | ユーザー採択後、別実施承認を得たCodex |
| 固定H4で `e_W/e_v` threshold pass | **未評価。科学的／技術的PASSを宣言しない** | 別実施承認下のv5 one-shot |
| rho取得・小分母・phase/shot/native資源優位 | **未検証。科学GO／NO-GO保留** | 将来の結果後に再判定 |

### 推奨順序

1. **ユーザー**：本レビューの限定的・条件付き受理を踏まえ、v5の17条件・局所係数モデル・`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`・旧絶対残差品質基準・runtime/caps・all-splits/one-shot/failure/STOPを**正式採択するか**明示判断する。採択をしない場合は修正点を固定し、変更が科学scopeへ及ぶかを別判断する。
2. **Codex**（採択を受けた場合）：新規の採択記録・machine契約・SOURCE/hash/プロトコルを別commitに固定して公開する。これだけでは計算しない。
3. **ユーザー**：v5の固定SOURCE・契約・成果物・one-shot範囲・resource/STOPを明示した**別途科学計算実施承認**を与えるか判断する。
4. **Codex**（実施承認がある場合に限る）：初回正しさgate、新局所予算、各splitのW/v/ρ可否を承認範囲で評価し、全3行と未到達理由を保存してmandatory STOP。失敗時の後付け閾値変更・自動retryなし。

### 本レビューにより不要な作業

- v5採択のためだけの重複したGPT一般研究方針レビューや大規模先行研究探索。
- 係数層の誤差上界を改善するための別の未承認Pauli厳密有理数再構成（現在のconditional engineering目的では必須ではない）。
- v4の失敗結果を上書きしての再実行、数値目標の緩和。

---

## 9. 監査と参考情報

**今回実施したこと**：固定GitHubの証明・数学的仮定・SOURCE／契約・監査原本の取得、主要係数演算の独立した紙上検証、group/support上界・累積gamma index・算術誤差伝播・十分条件の突合、出典の公開ドキュメントとの比較、Markdownレビュー記録作成。

**今回実施していないこと**：凍結NPZの科学配列decode、実入力への新budget数値代入、W/v/ρの計算、Pauli係数生成、独立数値correctness、科学simulation、eig/gap、回路生成・compile、GPU/CPU benchmark、SOURCE/contract/result/authorization/markerへのGitHub書込み。

**限界**：本レビューは、SOURCE固定版と文書に現れた実装・条件付き数値誤差モデルに対する独立数学レビューであり、全Python/native演算が特定hardwareで仮定した誤差上界を満たすとの独立検証ではない。v4既存伝播式のすべての実行経路を新たな厳密数値ライブラリにより検証したものでもない。数値計算を伴うcoverage確定や新規科学主張には別の契約・承認が要る。

### 一次資料・参考文献

1. [Gate 0 tail局所誤差証明（GitHub固定commit）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5a185cc0c3aae0e477bfc52b93061e86feb7dec/docs/tracks/algorithm_codesign/m1s_gate0_tail_local_bound_proof_20261016.md)。  
2. [v5最終実験契約候補（GitHub固定commit）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47da54279838a6b3e1497f36204ba7e53c4857f9/docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v5_candidate_20261016.md)。  
3. [v5 SOURCE `budget.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69de022dc8142f703423b1ce0bd42f05e123954e/src/trottertracks/algorithm_codesign/m1s_gate0_v5/budget.py)、[v5 SOURCE `science.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69de022dc8142f703423b1ce0bd42f05e123954e/src/trottertracks/algorithm_codesign/m1s_gate0_v5/science.py)、[v5 SOURCE `control.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69de022dc8142f703423b1ce0bd42f05e123954e/src/trottertracks/algorithm_codesign/m1s_gate0_v5/control.py)。  
4. Higham, N. J. and Mary, T., *A New Approach to Probabilistic Rounding Error Analysis*, *SIAM Journal on Scientific Computing* **41** (2019), A2815–A2835. [DOI](https://doi.org/10.1137/18M1226312)。引用するのは古典的 `gamma_k` 表式であり、確率的緩和ではない。  
5. [Python 3.11 `math.fsum`公式文書](https://docs.python.org/3.11/library/math.html#math.fsum)。  
6. [NumPy 1.26 `matmul`公式文書](https://numpy.org/doc/1.26/reference/generated/numpy.matmul.html)。  
7. [v4 one-shot結果・TECHNICAL_STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c937ed37e73bb9b0bb39f2fa7116d255a3215269/docs/tracks/algorithm_codesign/m1s_gate0_v4_one_shot_results_and_stop_20261016.md)。

**最終status**：`GPT_LEVEL2_V5_INDEPENDENT_MATHEMATICAL_REVIEW_COMPLETED_CONDITIONAL_TAIL_BOUND_ACCEPTED_FULL_PLATFORM_CERTIFICATE_NOT_ACCEPTED_FORMAL_ADOPTION_PENDING_NO_NEW_SCIENCE_EXECUTION_MANDATORY_STOP`。
