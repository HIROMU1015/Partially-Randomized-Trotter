# N13再設計batchの独立科学レビュー：N3方向付き保証、N1技術停止、N2修正

- 作成日：2026-10-11
- 文書種別：ユーザーの明示的な開始承認を受けて実施したGPT科学レビュー
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- 対象branch：`n13-redesign-preparation-20261011`
- 科学source：`bbcb0176d43bfd1f7a10a87fa8f38e200d61ec46`
- 結果commit：`e8eff10260f337ad0e957f31e7f7d36bfb0be719`
- 公開資料の固定commit：`10d91e518520e6f3fd853c85d8eff078357b7edb`
- レビュー状態：完了。今回の新結果に対する判断であり、旧結果レビューの再実行ではない。
- 中心テーマ：未採択。N3を次の科学的検証の優先候補とする。
- 実行状態：本レビューではrepositoryを変更せず、Codexの次batchやN1再実行を実行していない。

## 1. 結論

N3について、宣言された適用classにおける方向付き保証の数学的成立を支持する。同じtrial上界U・discarded下界cを使うnorm-only保証が不受理である一件を、係数から構成した方向付き保証が5 active modesで受理したことを、固定sourceから独立に再構成できた。

これは「試験がPASSした」以上の機構確認である。ただし、一般分子への適用、未知入力への移送、最小active space、量子総資源削減、独立論文としての新規性はまだ示されていない。

N1は保存vectorのdecode不備による技術停止であり、今回の新selectorについて科学的な肯定・否定の判断はできない。入力adapterと実保存形式のpreflightを修正し、元の候補・比較条件のままN1だけを一回再実行する価値がある。

N2は固定Sの誤差尺度の不一致を解消した補助修正として一区切りとする。量子回路資源の改善を得た結果ではない。

| 対象 | 科学的判断 | 次の扱い |
|---|---|---|
| N3 | 方向付き係数保証の成立と一件の追加受理を支持 | 優先的に方法を一般化・移送検証する。ただし中心テーマは未採択 |
| N1 | 新selectorの科学結果は未取得 | decoder・実形式testを直し、同じ条件でN1のみ一回再実行 |
| N2 | weighted-L1適格性の回復を支持 | 修正完了として保持し、追加charge/circuit scanへ進まない |
| 旧A-core/B/B′/C、Track A/B | 今回の結果から状態を変更しない | 既存source・結果・STOPを保持 |

## 2. レビュー対象と証拠の読み方

### 2.1 参照した正本

以下はすべて固定commitの資料である。ユーザー環境のローカル絶対pathを取得証拠とは扱っていない。

| ID | 資料 | 固定参照 |
|---|---|---|
| R1 | 今回の結果報告 | [n13_redesign_results_20261011.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10d91e518520e6f3fd853c85d8eff078357b7edb/docs/research/n13_redesign_results_20261011.md) |
| R2 | 結果前scope・数学的証明 | [n13_redesign_limited_scope_v1_20261011.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10d91e518520e6f3fd853c85d8eff078357b7edb/docs/research/n13_redesign_limited_scope_v1_20261011.md) |
| R3 | 固定science source | [n13_redesign.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bbcb0176d43bfd1f7a10a87fa8f38e200d61ec46/src/trottertracks/representation_exploration/n13_redesign.py) |
| R4 | N3一次結果・採否・真値診断 | [run1/n3.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10d91e518520e6f3fd853c85d8eff078357b7edb/artifacts/n13_redesign/2026-10-11/run1/n3.json) |
| R5 | N2一次結果 | [run1/n2.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10d91e518520e6f3fd853c85d8eff078357b7edb/artifacts/n13_redesign/2026-10-11/run1/n2.json) |
| R6 | N1失敗 | [run1/failure.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10d91e518520e6f3fd853c85d8eff078357b7edb/artifacts/n13_redesign/2026-10-11/run1/failure.json) |
| R7 | N1入力形式の静的監査 | [n1_failure_input_schema_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10d91e518520e6f3fd853c85d8eff078357b7edb/artifacts/n13_redesign/2026-10-11/n1_failure_input_schema_audit.json) |
| R8 | 保存監査 | [saved_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10d91e518520e6f3fd853c85d8eff078357b7edb/artifacts/n13_redesign/2026-10-11/saved_audit.json) |
| R9 | 実行前の数学・数値解釈 | [n13_redesign_execution_20261011.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10d91e518520e6f3fd853c85d8eff078357b7edb/docs/research/n13_redesign_execution_20261011.md) |
| R10 | 独立diagnostic監査 | [independent_diagnostic_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10d91e518520e6f3fd853c85d8eff078357b7edb/artifacts/n13_redesign/2026-10-11/independent_diagnostic_audit.json) |
| R11 | 公開後の別取得記録 | [remote_retrieval_result_commit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10d91e518520e6f3fd853c85d8eff078357b7edb/artifacts/n13_redesign/2026-10-11/remote_retrieval_result_commit.json) |

GitHub connectorを通して本文とsourceを読めたため、レビューに必要な資料の再pushは不要である。GPTコンテナからの直接raw downloadはDNS制約で利用できなかった。これはGitHub上の欠損・未pushではない。したがって、本レビューを「全repositoryを新環境へcloneして、全保存監査をそのまま再実行した外部再現」とは記述しない。

### 2.2 証拠の強さを分ける

R8の660 checksは、IR数0、N1 science rows0、N3 records29、N2 fixtures2を対象としている。`worst_absolute_native_residual=0`はN1のnative回路が検証されたことを意味しない。対象回路が存在しないからである。

18 synthetic testsのPASSと、実保存packに対応できなかったN1の技術失敗は両立する。テストの正常終了を入力schemaの完全な適合や科学仮説の支持へ読み替えない。

N3の採否はbinary64 literalを正確な有理数に変換して行っている。ここでいう厳密性は、その有限係数Hamiltonianに対する有理数比較の厳密性である。分子積分の誤差、物理模型の妥当性、実機雑音まで保証したものではない。

### 2.3 GPT側で今回追加した確認

付録Aのコードはrepository moduleをimportせず、固定source R3の係数literalを明示的に転記した。元の4入力以外のHamiltonian、元grid以外のlambda、native compile、量子shotsを追加していない。

- rational LDLとは別に、leading principal minorsで一体行列の正定値性を確認。
- 有理数inverse quadratic formからkappaと採否条件を再計算。
- 4入力・3方法の最終active数を報告結果と照合。
- 明示的なoccupation演算からmany-body行列を別に構成し、CAR恒等式を有理数として照合。
- certificateの選択を終えてから、同じ固定入力の固有値を診断。
- 採択済み証拠点が第7節の行列不等式も満たすことを有理数で確認。

これは固定入力の独立算術・意味論確認であり、source全体の再実行、primary JSON全bytesの再照合、未知入力の検証とは異なる。

## 3. N3の数学的契約

### 3.1 適用対象

当stageのclassは、実係数のnumber-preserving one-bodyと非負density相互作用である。

\[
H=d\Gamma(h)+\sum_{i<j}J_{ij}n_i n_j,\qquad h=h^T,\quad J_{ij}\ge0.
\]

固定粒子数をnuとし、一つの外部mode dを空に固定した空間Pと、同じ全粒子数sectorでnd=1の空間Qへ分ける。

\[
H=\begin{pmatrix}A&B\\B^\dagger&C\end{pmatrix},\qquad
 a=\lambda_{\min}(A),\quad E_0=\lambda_{\min}(H).
\]

aとE0は証明上の記号であって、selectorに入力する真値ではない。

重要な限定：一般のexchange、pair hopping、attractive densityへは、そのまま流用できない。外部modeが複数残る段階では、この実装は方向付き式ではなく旧scalar保証を使っている。今回の「軌道数」はsource上のfermionic modesであり、空間軌道数と無条件に同一視しない。

### 3.2 結合の方向を保持するCAR恒等式

active一体行列をhA、v_i=h_{i,d}とすると、PからQへのhoppingは

\[
B^\dagger=a_d^\dagger\sum_i\overline v_i a_i.
\]

P上では外部が真空なので、CARから

\[
BB^\dagger=d\Gamma_\nu(vv^\dagger)
\]

を得る。右辺は、結合がactiveのどの一体方向にあるかを保持する。

方向を捨てると、0<nu<=active modesの非自明sectorでは

\[
\|BB^\dagger\|=\|v\|^2=\beta^2
\]

である。このscalarだけでは、低エネルギー方向と強い結合方向が同じかどうかを判別できない。

### 3.3 one-body行列からの相対form bound

\[
M_\lambda=h_A-\lambda I\succ0,\qquad
\kappa=v^\dagger M_\lambda^{-1}v,\quad L=\nu\lambda.
\]

weighted Cauchy–Schwarzにより

\[
vv^\dagger\preceq\kappa M_\lambda.
\]

二次量子化は正値性を保ち、active density相互作用VAは非負なので、

\[
BB^\dagger\preceq\kappa d\Gamma(M_\lambda)
\preceq\kappa(A-LI),\qquad A\succeq LI.
\]

従って、active全体の最大結合normではなく、activeのエネルギーに相対的な結合の強さを使える。hAの固有分解を説明用に書けば

\[
\kappa=\sum_j\frac{|u_j^\dagger v|^2}{\epsilon_j-\lambda}.
\]

これは新しい未知ground-state oracleではなく、小さい一体行列のresolvent quadratic formである。同じvのnormでも向きによって値が変わる。

### 3.4 取得するUとc

UはP内の明示されたnormalized trial stateの期待値による上界である。現実装はdeterminantと直接hopでつながる二determinantの等重み重ね合わせを使用する。したがって「真の基底状態を不要とする」は正しいが、「試行状態に関する情報を一切使わない」は正しくない。

一体下界ellにより

\[
C\succeq cI,\qquad c=h_{dd}+(\nu-1)\ell
\]

を得る。非負density相互作用は下界を下げない。

Uとcはmany-body eigensolverから取得していない。現trial集合の全列挙はnu=2・active最大5で安いだけであり、一般fillingの全determinant列挙を効率的と主張しない。

### 3.5 Schur採否条件の独立導出

目標幅delta>0について

\[
c>U,\quad D=c-U+\delta>0,\quad t=\kappa/D\le1,
\quad t(U-L)\le\delta
\]

なら、

\[
0\le a-E_0\le\delta.
\]

証明：z=a-deltaとするとC-z>=D I。従って

\[
\begin{aligned}
A-zI-B(C-zI)^{-1}B^\dagger
&\succeq A-aI+\delta I-t(A-LI)\\
&=(1-t)(A-aI)+[\delta-t(a-L)]I\\
&\succeq[\delta-t(U-L)]I\succeq0.
\end{aligned}
\]

Schur補元によりH>=zI、変分原理によりE0<=a。t<=1、非負density、c>Uという現実装の十分条件を落としてはいけない。この証明は標準的なSchur補元と相対form評価の応用であり、短い導出だけを新規定理と呼ばない。[L1,L2]

補足：数理的にはc-U+delta>0だけで進められる範囲もあるが、今回のcodeはc>Uというより保守的な契約である。本レビューでは契約を変更して結果を救済していない。

## 4. N3の固定結果と独立照合

### 4.1 採否の対照

| 固定入力 | legacy scalar | 同じU/cのnorm-only | direction | 解釈 |
|---|---:|---:|---:|---|
| original_gapped | 4 | 4 | 4 | 最初の旧scalar段階で受理。方向法による追加削減ではない |
| original_small_gap | 6 | 6 | 5 | 方向付き係数保証が追加受理 |
| strong_external_coupling | 6 | 6 | 6 | 当該promotion経路の5-mode候補は不受理 |
| near_external_energy | 6 | 6 | 6 | 外部エネルギー下界条件も不成立 |

数字は最終active modes。GPT独立コードはこの12件の最終active数を再現した。[R1,R4]

### 4.2 小gap例で何が変わったか

active=[0,1,2,3,4]、external=5。丸め表示で

\[
U=-0.25,\quad c=0.36,\quad\beta^2=0.0041,\quad\delta_0=0.004.
\]

同じU/cのnorm-onlyは

\[
\frac{\beta^2}{c-U+\delta_0}
=0.006677524429967428>0.004
\]

で不受理である。

方向付きはlambda約-0.27の既存grid点で

\[
\kappa=0.008258558932006846,\quad L\simeq-0.54,
\quad t=0.013450421713366198,
\]

\[
t(U-L)=0.0039006222968761975<0.004
\]

を満たす。全値はsourceのbinary64 literalを表す有理数から再計算した。lambda=-27/100へ無断置換したものではない。

「U/cを改善したから受理できた」ではなく、そのU/cを共有した対照も不受理なので、追加受理は結合の扱いの差に帰属できる。

### 4.3 境界余裕とtruth診断

採否余裕は

\[
\delta_0-t(U-L)=9.937770312380262\times10^{-5},
\]

目標の約2.4844%である。この値は数値丸めより圧倒的に大きいが、広い入力範囲への頑健性を示すほどの余裕とは限らない。

有効な三grid点の右辺は

| lambda概数 | SPD | 方向条件右辺 | 採否 |
|---:|---|---:|---|
| -0.30 | 真 | 0.004160137922 | 不受理 |
| -0.27 | 真 | 0.003900622297 | 受理 |
| -0.24 | 真 | 0.004290950842 | 不受理 |
| -0.21 | 偽 | 評価しない | 不受理 |
| -0.18 | 偽 | 評価しない | 不受理 |

受理後の独立固有値診断ではa-E0=0.0002621922779750818で、保存値0.00026219227797563693との差は倍精度の丸め範囲である。真値診断は全証明量と採否の確定後に実行した。

保証幅0.004はこの実誤差の約15倍で、最適な幅ではない。今回得たのは一件の安全な受理であり、tight boundではない。

### 4.4 対照入力の意味

強結合の場合、同じpromotion規則が残す5-mode候補はactive=[0,1,2,3,5]であり、外部は4になる。その実際の縮約誤差は約0.0097603390715。低外部energy例の5-mode誤差は約0.0443378387037である。いずれも0.004を超える。

これらの不受理は当該候補に対して妥当である。ただし、全ての別5-mode選択が不可能と示したわけではない。また、全空間へのfallbackは正確性を守る仕組みであり、縮約成功の件数へ数えない。

### 4.5 開発データと独立性

original_small_gapは前レビューで5 modesなら十分と診断済みのdevelopment入力である。source内の採否にtrue ground/gapが流入していないことと、方法開発の際にその入力を見ていたことは別である。

有理数証明は、その入力についての保証を正当に与える。開発入力だったことによって証明が無効になるわけではない。一方、成功率・頑健性・一般化可能性を示すには、設計に使っていない入力での追加検証が必要である。

## 5. N3でまだ主張できないこと

### 5.1 一般の量子化学Hamiltonianへの適用

現在のBBdaggerの一体化は、外部占有を変える項がone-body hoppingだけであることに依存する。一般の二体項が外部占有を変えると、Bには三つのactive fermion演算子を含む項も生じ、BBdaggerが単純なone-body行列へ落ちる保証はない。

また、VA>=0を使うため、任意のexchange/pair-hopping/attractive densityを含む分子積分へそのまま適用できない。軌道変換も、density-onlyという形を一般には保たない。

この制約は「量子化学に使えない」とする不可能性ではない。正値分解、制御された残差、別のdomination証明が必要という未解決課題である。現時点で無断に一般分子へ拡張してはいけない。

### 5.2 真のmany-body gapを取得したという主張

c-Uは、指定されたP/Q分割と取得した上・下界に基づく分離の十分条件であり、真のmany-body spectral gapそのものではない。

### 5.3 時間発展の保証

0<=a-E0<=deltaから、任意状態の時間発展誤差がT delta以内と結論できない。スペクトルの他の部分、状態写像、時間発展の差を別に扱う必要がある。

### 5.4 量子資源の総削減

active5で受理したことは、JWのsystem modesを6から5へ減らせる一例である。QPE/PR全体の回路・shot・状態準備・精度配分を含む資源削減は未評価。

active側のエネルギーaを推定して|eA-a|<=epsilonAが成り立つ事象では

\[
E_0\in[e_A-\epsilon_A-\delta,\ e_A+\epsilon_A].
\]

区間中心eA-delta/2を使えば半幅はepsilonA+delta/2。したがって、最終半幅epsilon_totalを求めるならepsilonA+delta/2<=epsilon_totalを満たす配分が必要である。同じ成功確率で比較し、状態準備・overlap・simulation誤差の前提もそろえる。

0.004は本batchの縮約幅であり、最終アルゴリズムの全誤差や化学的精度の達成を意味しない。

### 5.5 計算量

現certificateは最大5x5の有理数行列を扱う。固定されたlambda一点のLDL/solveは算術操作数でO(m^3)が自然だが、有理数bit growth、候補外部mode数、trial-state生成、一般fillingでの試行集合増大は別である。

多項式に多数のdeterminantを列挙しながら「many-body計算不要」と主張しない。証明上はUが任意の検証可能な試行状態上界でよいので、将来は固定サイズのtrial集合などへ変更できるが、その場合の有効性は未検証。

## 6. 先行研究との差と研究の着地点

文献確認は2026-10-11時点の限定調査であり、全世界のpriorityを証明するものではない。以下の比較は、本文を確認できたFeshbach–Schurの該当節、または各論文の公開要旨・公式出版情報に基づく。全文を確認していない論文の全定理に対する不存在証明は行っていない。

| 先行方向 | 既知内容 | 今回との差分候補と限界 |
|---|---|---|
| Feshbach–Schur / relative form bound [L1] | resolvent/Schurを使う固有値・固有vectorのrigorous bound。相対form normも使用する | 方向情報を使うこと自体は新規でない。N3はfermionic CARからsmall coefficient matrixへ落とし、active-space受理へ使う構成が候補 |
| Schrieffer–Wolff [L2] | low/high subspaceをdecoupleする有効Hamiltonian、基底エネルギー近似の評価 | 現N3は仮想遷移補正を有効Hへ加えずPHPを使い、その省略幅を証明する。新しいSW理論とは呼ばない |
| SQuISH [L3] | Hamiltonianと近似ground-state wavefunctionを使った反復的sparsification | N3は真のground stateを使わず係数から証明するが、trial上界Uは使う。SQuISHの全手法に優位と示していない |
| AVAS [L4] | atomic valence projectorからactive orbitalsを生成 | 自動active-space選択という主題は既知。N3の貢献候補は、その縮約をenergy errorで認証する生成・受理手順 |
| autoCAS [L5] | DMRGを利用するactive-space選択と大きいvalence空間への運用 | 同じtask・同じ情報・同じ保証の比較ではない。現在の狭い模型を分子実績より優位としない |

現段階で独立論文に十分な主成果が確立したとは判断しない。Schurの短い証明と一つの既知development入力だけでは不足する。

一方、次のような着地点は具体的な研究候補である。

> 宣言したHamiltonian classに対し、many-body真値を使用せず小さい係数行列から方向付きの縮約certificateを生成し、scalar基準では拒否する有意味な入力族を安全に受理する、active-space選択アルゴリズム。

その上で、古典取得費用、失敗条件、未知入力移送、同精度・同成功確率での量子資源を評価する。PRへの接続は外部前処理として成立する。PR特有の新規性を無理に追加する必要はなく、別Trackの内部sampling/return改良を重複させない。

## 7. 次段へ進むための具体的な方法論案：small-matrix certificate生成

現在はlambda五点から選ぶ。一般入力へ移す際、少数の固定gridによる取りこぼしと、物理的な縮約不能を区別しやすくするため、同じ保証を小さい行列不等式の実行可能性として扱う案がある。

D=c-U+deltaを固定し、scalar kappaとmuに対して

\[
\begin{cases}
0\le\kappa\le D,\\
\kappa h_A-\mu I-vv^\dagger\succeq0,\\
\kappa U-\nu\mu\le\delta D
\end{cases}
\]

を求める。これは二つのscalarを変数とするLMIと線形不等式である。

既存の採択点はmu=kappa lambdaとしてこの条件を満たす。現条件の全ての有効なstrict-SPD証拠点を含む。semidefinite境界やkappa=0の扱いも可能な直接十分条件として書いているため、既存の五点探索と全領域で「完全同一の選択集合」とまでは呼ばない。

証明：一体LMIとVA>=0から

\[
BB^\dagger\preceq\kappa A-\nu\mu I.
\]

z=a-deltaと置いてSchurを取ると

\[
\begin{aligned}
A-zI-B(C-zI)^{-1}B^\dagger
&\succeq(1-\kappa/D)(A-aI)
+\left[\delta-(\kappa a-\nu\mu)/D\right]I\\
&\succeq\left[\delta-(\kappa U-\nu\mu)/D\right]I\succeq0.
\end{aligned}
\]

今回の独立チェックでは、新しいlambda最適化を行わず、既存の採択済み証拠点に対してこの行列の全31 principal minorsが非負であることを有理数で確認した。数値では

\[
\mu\simeq-0.0022298109116418484,
\quad \delta D-(\kappa U-2\mu)\simeq6.1017909718\times10^{-5}.
\]

この再定式化そのものも、新規SDP理論ではない。価値を示すには、入力からのcertificate生成が固定gridより有用か、scalar対照以上の適用範囲を持つかを検証する必要がある。

実装時はfloating solverのSUCCESSを証明とせず、返された証拠を有理数・外向き区間等で検証する。境界で有理数証拠が得られない場合に未判定を返してよい。探索の成功・不成功と、証明のsoundnessを分離する。

## 8. N1の評価と技術修正・再実行の判断

N1の新selectorは、係数PF上界込みの科学行へ到達していない。従って、前batchの結果に新たな肯定・否定の証拠を加えない。[R1,R6]

R3の`prepare_factors`は`ungauged_eta`を数値listとして添字アクセスするが、実保存packは`real`/`imag`を持つvector objectである。R7では44 vectorsが全てこの形式で、長さ4・imaginary成分ゼロだったと確認している。

今回修正が必要なのは、科学モデルよりも入力契約の境界である。推奨する次のCodex作業は以下をまとめた一単位である。

1. 読み込み境界でvector/matrix形式を明示decodeする。長さ・shape・finite・実数spectrum条件を確認する。
2. `ungauged_eta`だけでなく、同型の全vector fieldを調べる。虚部を無条件に捨てない。
3. 実保存packの形式をそのまま使う回帰testと、launch前の全context/schema検査を追加する。
4. 元の44 factor候補・178 pair・q集合・比較対象・指標・誤差予算は変えず、修正sourceを新commitへ固定する。
5. N1だけを一回再実行し、失敗runと新runを分離して保存する。完了済みN3/N2を再実行しない。

デコードされた係数値が元packと一致することを検証する。期待外の虚部や値変更が必要と判明した場合は単なるdecode修正を越えるので停止・報告する。

細かなtest修正ごとにGPTレビューを挟む必要はない。source修正・実形式preflight・既定N1一回実行をCodexへまとめて任せてよい。ただし、今回のレビューを科学実行したこととして扱わない。

再実行で全候補がPF上界により不適格、exactへfallback、または総費用差がない場合も、正常な科学結果である。成功するまでqや候補を増やしてはいけない。

## 9. N2の評価

固定Sに対するweighted entry-L1 fitは、意図された有理数入力でprimal=dual=1/500を示し、Frobenius fitのentry-L1=9/2500とは採否が異なる。binary64入力では別のprimal/dual sandwichを保持し、約1.67e-16のgapを記録している。[R1,R5]

これは誤差尺度の不整合を解消した有効な補助修正である。一方、追加S探索、charge回路合成、FT資源の改善は行っていない。

N2は修正完了として保持する。今回の適格性回復を理由にsigned-charge全体の大規模scanを再開する判断はしない。

## 10. 次のN3検証で判別すべきこと

### 10.1 目標

「original_small_gapを再び5で受理する」だけでなく、宣言したclassにおける係数certificate生成・選択法として成立するかを判別する。

### 10.2 有益な比較

- 現在のlambda五点保証。
- 同じU/c・同じ候補active spaceのnorm-only。
- 利用可能なより良い係数scalar下界・trial上界。方向法にも同じ改善を公平に適用する。
- 提案するsmall-matrix生成法。

方向法が全ての入力でscalar法を支配すると仮定しない。最終的な運用では、複数の有効certificateのうち一つでも成立すれば受理できる構成が自然である。

### 10.3 入力と情報の分離

既存4例は回帰検査として保持する。新たな検証には、設計に使用していない入力を結果前に固定し、結合normと方向、one-body尺度、粒子数・mode数の影響を区別する。特定の軌道IDやactive数5を正解としてhard-codeしない。

真値はselectorを閉じた後の評価専用とする。証明は各入力に有効なので、実験上のholdoutはsoundnessの代わりではなく、方法の有用性を測るために必要である。

### 10.4 何を記録するか

受理・不受理だけでなく、受理余裕、下界取得の失敗、試行上界の保守性、行列サイズ、算術時間、最大分子分母bit長、未判定率を残す。次に資源まで評価する場合は、縮約幅の予算とactive solverの予算を同時に扱う。

### 10.5 複数modeの削減

一度に多外部を扱う一般式を無理に流用しない。同じclassが保存される逐次削減では、各段のエネルギー差上界を足し合わせることで全体幅を管理できる。ただし、各段で別に0.004を使いながら全体幅を0.004と報告してはいけない。逐次証明の有用性と費用は将来検証である。

一般の化学二体項へ拡張する前には、どのCAR・正値性・下界の条件が壊れるかを明示する。現classの結果をそのまま分子へ外挿しない。

## 11. 研究の着地点と方針判断

現時点でN3を優先するのは、N1が失敗したからの消去法ではない。N3では、提案した方向情報が同じU/cの対照を越えて一件の受理を変え、係数だけの小さい計算で保証を構成できたという、新しい正の機構証拠があるためである。

一方、今後の結果が、既知Schur不等式を特定toyへ適用しただけに留まるなら、独立主研究としての評価は下がる。その場合は一般化する理由のないtoy scanを増やさず、既知のcertificate primitiveとして整理する。

目指し得る成果は、入力class、取得できる証明量、計算量、受理条件、失敗時の扱いが明確な保証付き縮約構成である。量子資源改善を主張するには、active solver・状態準備・最終精度まで接続する。純粋なclassical証明法として成立する可能性と、PR外部最適化として資源を下げる可能性は別に評価する。

旧A-core/B′等の小例で負けたことから、Hamiltonian表現研究全体を否定しない。逆に、小gap一件で成功したことからN3の中心テーマ採択を急がない。

## 12. 次の担当と停止条件

直近の担当はCodex。

- N1：入力adapter・実形式preflight・同条件一回再実行を一単位で実施する価値がある。
- N3：本レビューで整理したsmall-matrix生成と、同classでの限定移送を具体的scopeへ落とす。必要な小規模な実装・検証はユーザーの実行指示後に行う。
- N2：補助修正完了として保持。
- 共通：旧source、failure、N3/N2の一次結果、Track A/B、既存STOPを保持する。N1の再実行がN3の新研究方針の承認を代行することはない。

再レビューを要する節目は、N1の科学結果が新たに得られたとき、N3の生成法・移送で重要な結果が出たとき、または一般二体・多外部・主task/指標を変更するときである。単なるschema修正や文書整理では同じ科学レビューを繰り返さない。

成功条件は、N3が真値なしに有効な証明を生成し、明示した対照との差を複数入力で示せること。失敗はsoundness違反、証明量が取得できないこと、有効例がdevelopment一件から広がらないこと、取得古典費用や量子費用が利益を消すこと等である。ただし、一件の不受理を物理的な縮約不能と混同しない。

既存manifestのdescription欠落は局所結果と区別した保守事項。global checkerをPASSと称さず、今回資料が取得可能であることから不要な再pushや科学再計算は要求しない。

## 13. 参考文献

[L1] G. Dusson, I. M. Sigal, B. Stamm, “The Feshbach–Schur map and perturbation theory” (2021), arXiv:2105.02058; DOI 10.4171/ECR/18-1/5. 本文pp.2–3のrelative form norm・Main Theorem 1.1をPDF画像と本文で確認。https://arxiv.org/abs/2105.02058

[L2] S. Bravyi, D. P. DiVincenzo, D. Loss, “Schrieffer–Wolff transformation for quantum many-body systems” (2011), Annals of Physics 326, 2793–2826; arXiv:1105.0675. 公開要旨と出版情報を確認。https://arxiv.org/abs/1105.0675

[L3] D. B. Chamaki et al., “Self-consistent Quantum Iteratively Sparsified Hamiltonian method (SQuISH): A new algorithm for efficient Hamiltonian simulation and compression,” arXiv:2211.16522. Hamiltonianと近似ground-state wavefunctionを利用する点を公開要旨で確認。https://arxiv.org/abs/2211.16522

[L4] E. R. Sayfutyarova, Q. Sun, G. K.-L. Chan, G. Knizia, “Automated Construction of Molecular Active Spaces from Atomic Valence Orbitals,” JCTC 13, 4063–4078 (2017); arXiv:1701.07862; DOI 10.1021/acs.jctc.7b00128. https://arxiv.org/abs/1701.07862

[L5] C. J. Stein, M. Reiher, “autoCAS: A Program for Fully Automated Multiconfigurational Calculations” (2019), J. Comput. Chem. 40, 2216–2226; DOI 10.1002/jcc.25869; arXiv:1904.00097. https://arxiv.org/abs/1904.00097

[L6] A. M. Childs et al., “A Theory of Trotter Error,” arXiv:1912.08854. N1既定PF上界の背景。今回N1の科学実行は未到達であり、この文献の存在を新実装の正しさの代用とはしない。https://arxiv.org/abs/1912.08854

## 付録A. 今回の独立チェックの実行方法と範囲

Python 3、NumPy、SymPyを使用。コードを`independent_n3_review_checks.py`として保存し、次を実行する。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python independent_n3_review_checks.py
```

同じディレクトリへJSONを生成する。GitHubへの書き込み、production sourceのimport、量子回路合成、LP/SDP最適化、新しいHamiltonian入力は行わない。最大5x5のrational inverse/minorと15次元の事後固有値診断を含む。

以下にコードと実際の主要出力を収録する。

```python
#!/usr/bin/env python3
"""Independent review checks for the fixed N13 inputs (no repository imports).

The input literals and lambda/trial rules are transcribed from frozen source
bbcb0176d43bfd1f7a10a87fa8f38e200d61ec46, n13_redesign.py.
This is not a byte replay of run1/n3.json and is not a new scientific batch.
Only the four previously fixed Hamiltonians are used. Selector calculations use
exact rationals representing the binary64 literals; many-body eigensolves are
post-certificate diagnostics only. No circuit compilation or quantum shots.
"""
from __future__ import annotations
import itertools as it
import json
import hashlib
from pathlib import Path
import numpy as np
import sympy as sp

Q = sp.Rational
GOAL = Q(1,250)
RHO = [Q(0),Q(1,20),Q(1,10),Q(3,20),Q(1,5)]

def binary(value):
    a,b = float(value).as_integer_ratio()
    return Q(a,b)

def pack(x):
    x=Q(x)
    return {'rational':[int(x.p),int(x.q)],'float':float(x)}

def build_input(name):
    tail = ['6','8'] if name=='original_gapped' else ['.4','-.1' if name=='near_external_energy' else '.6']
    h=sp.diag(*[binary(x) for x in ['-.2','-.1','.1','.25',*tail]])
    vals=[(1,2,'.2'),(0,3,'.1'),(2,4,'.08'),(3,5,'.16' if name=='strong_external_coupling' else '.04'),(4,5,'.20' if name=='strong_external_coupling' else '.05')]
    for i,j,v in vals:h[i,j]=h[j,i]=binary(v)
    return h,{(0,1):binary('.3')}

def determinants(n,nu):
    return [s for s in range(1<<n) if s.bit_count()==nu]

def fock_sector(h,nu,density=None):
    """Direct occupation algebra, not the project/JW helpers."""
    n=h.rows;states=determinants(n,nu);index={s:i for i,s in enumerate(states)}
    H=sp.zeros(len(states))
    for col,state in enumerate(states):
        for q in range(n):
            if not ((state>>q)&1):continue
            removed=state^(1<<q);sq=(-1)**((state&((1<<q)-1)).bit_count())
            for p in range(n):
                if (removed>>p)&1:continue
                sp_=(-1)**((removed&((1<<p)-1)).bit_count())
                H[index[removed|(1<<p)],col]+=sq*sp_*h[p,q]
        for (i,j),v in (density or {}).items():
            if ((state>>i)&1) and ((state>>j)&1):H[col,col]+=v
    assert H==H.T
    return H,states

def coefficient_trials(h,den,active,nu):
    dets=list(it.combinations(active,nu))
    dval=[sum((h[i,i] for i in d),Q(0))+sum((v for (i,j),v in den.items() if i in d and j in d),Q(0)) for d in dets]
    vals=list(dval)
    for a,b in it.combinations(range(len(dets)),2):
        left=set(dets[a])-set(dets[b]);right=set(dets[b])-set(dets[a])
        if len(left)==len(right)==1:
            p,q=next(iter(left)),next(iter(right))
            if h[p,q]!=0:vals.append((dval[a]+dval[b])/2-abs(h[p,q]))
    return min(vals),len(vals)

def legacy(h,den,active,nu):
    ext=[i for i in range(h.rows) if i not in active]
    trial=sorted(active)[:nu]
    U=sum((h[i,i] for i in trial),Q(0))+sum((v for (i,j),v in den.items() if i in trial and j in trial),Q(0))
    vn=sum((abs(h[i,j]) for i in range(h.rows) for j in range(i+1,h.rows)),Q(0))+sum(map(abs,den.values()),Q(0))
    off=sum((abs(h[i,j]) for i in range(h.rows) for j in range(i+1,h.rows) if i in ext or j in ext),Q(0))
    rhs=Q(0);valid=True
    for pos,orb in enumerate(ext):
        free=[i for i in range(h.rows) if i not in ext[:pos+1]]
        c=h[orb,orb]+sum(sorted(h[i,i] for i in free)[:nu-1],Q(0))-vn-(len(ext)-1)*off
        b=sum((abs(h[orb,i]) for i in active),Q(0))
        if c<=U:valid=False
        else:rhs+=b*b/(c-U+GOAL)
    return {'accepted':bool(valid and rhs<=GOAL),'U':pack(U),'rhs_at_goal':pack(rhs)}

def direction(h,den,active,nu,method):
    d=next(i for i in range(h.rows) if i not in active)
    m=h.extract(active,active);v=h.extract(active,[d]);U,nt=coefficient_trials(h,den,active,nu)
    r=min(m[i,i]-sum((abs(m[i,j]) for j in range(m.rows) if i!=j),Q(0)) for i in range(m.rows))
    span=max(m.diagonal())-min(m.diagonal())
    entries=[]
    for lam in sorted(set(r+x*span for x in RHO)):
        M=m-lam*sp.eye(m.rows)
        # Sylvester leading principal minors, independent of the source LDL.
        minors=[M[:j,:j].det() for j in range(1,m.rows+1)]
        ok=all(x>0 for x in minors)
        ent={'lambda':pack(lam),'spd':bool(ok),'leading_minors':[pack(x) for x in minors]}
        if ok:
            k=(v.T*M.inv()*v)[0]
            ent['kappa']=pack(k)
            # The rank-one Schur relation is certified without a Fock eigensolve.
            assert k>0
            extM=M.row_join(v).col_join(v.T.row_join(sp.Matrix([[k]])))
            assert extM.det()==0
        entries.append(ent)
    ell=max([r]+[Q(*e['lambda']['rational']) for e in entries if e['spd']])
    c=h[d,d]+(nu-1)*ell;beta2=(v.T*v)[0]
    denom=c-U+GOAL
    normrhs=beta2/denom if denom>0 else None
    chosen=None
    for ent in entries:
        if not ent['spd']:continue
        lam=Q(*ent['lambda']['rational']);L=nu*lam;k=Q(*ent['kappa']['rational'])
        if c<=U:ent['accepted']=False;continue
        t=k/denom;rhs=t*(U-L)
        ent.update(L=pack(L),t=pack(t),rhs=pack(rhs),margin=pack(GOAL-rhs),accepted=bool(t<=1 and rhs<=GOAL))
        if ent['accepted'] and chosen is None:chosen=ent
    accepted = (c>U and normrhs<=GOAL) if method=='norm_only' else chosen is not None
    return {'accepted':bool(accepted),'active':active,'external':[d],'U':pack(U),'c':pack(c),'ell':pack(ell),'beta2':pack(beta2),'norm_rhs':pack(normrhs) if normrhs is not None else None,'trial_count':nt,'points':entries,'chosen':chosen}

NAMES=['original_gapped','original_small_gap','strong_external_coupling','near_external_energy']
EXPECTED={'original_gapped':{'legacy_scalar':4,'norm_only':4,'direction':4},'original_small_gap':{'legacy_scalar':6,'norm_only':6,'direction':5},'strong_external_coupling':{'legacy_scalar':6,'norm_only':6,'direction':6},'near_external_energy':{'legacy_scalar':6,'norm_only':6,'direction':6}}


def run():
    results=[];proof_checks=[]
    for name in NAMES:
        h,den=build_input(name);methods={}
        for method in ['legacy_scalar','norm_only','direction']:
            active=list(range(4));history=[]
            for stage in range(3):
                ext=[i for i in range(6) if i not in active]
                if not ext:cert={'accepted':True,'full_space':True}
                elif len(ext)==2 or method=='legacy_scalar':cert=legacy(h,den,active,2)
                else:cert=direction(h,den,active,2,method)
                history.append({'active':active.copy(),'certificate':cert})
                if cert['accepted']:break
                promote=max(ext,key=lambda j:(sum((abs(h[i,j]) for i in active),Q(0)),-j))
                active=sorted(active+[promote])
            assert len(active)==EXPECTED[name][method]
            methods[method]={'selected_active':active,'history':history}
        # Certificate choices are complete before any many-body eigenvalue diagnostic.
        H,states=fock_sector(h,2,den);Hf=np.asarray(H,float);E=float(np.linalg.eigvalsh(Hf)[0])
        stage=methods['direction']['history'][-1] if name=='original_gapped' else methods['direction']['history'][1]
        active=stage['active'];ext=[i for i in range(6) if i not in active]
        p=[j for j,s in enumerate(states) if all(not ((s>>d)&1) for d in ext)];q=[j for j in range(len(states)) if j not in p]
        A=H.extract(p,p);B=H.extract(p,q);C=H.extract(q,q)
        a=float(np.linalg.eigvalsh(np.asarray(A,float))[0]);cc=float(np.linalg.eigvalsh(np.asarray(C,float))[0])
        diag={'stage_active':active,'full_energy':E,'active_energy':a,'discarded_energy':cc,'width':a-E}
        if len(ext)==1:
            v=h.extract(active,ext);dg,_=fock_sector(v*v.T,2)
            assert B*B.T==dg
            diag['CAR_BBdagger_exact']=True
            cert=stage['certificate']
            min_eigs=[]
            for point in cert['points']:
                if not point['spd']:continue
                k=Q(*point['kappa']['rational']);lam=Q(*point['lambda']['rational']);L=2*lam
                dom=k*(A-L*sp.eye(A.rows))-B*B.T
                eigmin=float(np.linalg.eigvalsh(np.asarray(dom,float))[0])
                assert eigmin>=-1e-12
                min_eigs.append({'lambda':float(lam),'dominance_min_eig_diagnostic':eigmin})
                if point.get('accepted'):
                    k=float(k);c=float(Q(*cert['c']['rational']));U=float(Q(*cert['U']['rational']))
                    z=a-float(GOAL)
                    schur=np.asarray(A,float)-z*np.eye(A.rows)-np.asarray(B,float)@np.linalg.solve(np.asarray(C,float)-z*np.eye(C.rows),np.asarray(B.T,float))
                    proof_checks.append({'input':name,'lambda':float(lam),'Schur_min_eig_diagnostic':float(np.linalg.eigvalsh(schur)[0]),'width_within_goal':a-E<=float(GOAL)})
            diag['dominance_diagnostics']=min_eigs
        results.append({'input':name,'methods':methods,'post_certificate_diagnostic':diag})
    # Exact review algebra: the current rule is a feasible two-scalar LMI witness.
    rec=next(x for x in results if x['input']=='original_small_gap')['methods']['direction']['history'][1]['certificate']
    w=rec['chosen'];k=Q(*w['kappa']['rational']);lam=Q(*w['lambda']['rational']);mu=k*lam
    U=Q(*rec['U']['rational']);c=Q(*rec['c']['rational']);D=c-U+GOAL
    h,_=build_input('original_small_gap');active=[0,1,2,3,4];m=h.extract(active,active);v=h.extract(active,[5])
    R=k*m-mu*sp.eye(5)-v*v.T
    # PSD via all principal minors in fixed 5x5, used only as an independent check.
    principal=[R.extract(s,s).det() for size in range(1,6) for s in it.combinations(range(5),size)]
    assert all(x>=0 for x in principal)
    assert 0<k<=D and k*U-2*mu<=GOAL*D
    lmi={'kappa':pack(k),'mu':pack(mu),'lambda':pack(lam),'D':pack(D),'affine_goal_margin':pack(GOAL*D-(k*U-2*mu)),'all_31_principal_minors_nonnegative':True,'scope':'same accepted witness; no new optimizer, lambda grid, or Hamiltonian'}
    return {'status':'INDEPENDENT_FIXED_INPUT_REVIEW_CHECKS_PASS','science_source':'bbcb0176d43bfd1f7a10a87fa8f38e200d61ec46','publication':'10d91e518520e6f3fd853c85d8eff078357b7edb','provenance':'Input literals transcribed from fetched source. Not a byte replay of all repository artifacts. Exact rationals represent the original binary64 literals.','original_production_reexecuted':False,'new_Hamiltonian_inputs':0,'new_lambda_grid_points':0,'new_compilations':0,'quantum_shots':0,'certificate_results':results,'schur_diagnostics':proof_checks,'equivalent_LMI_witness':lmi}

if __name__=='__main__':
    out=run();here=Path(__file__).resolve();out['script_sha256']=hashlib.sha256(here.read_bytes()).hexdigest()
    dst=here.with_suffix('.json');dst.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    summary=[]
    for x in out['certificate_results']:
        row={'input':x['input'],'active':{m:len(d['selected_active']) for m,d in x['methods'].items()},'width':x['post_certificate_diagnostic']['width']}
        if x['input']=='original_small_gap':
            c=x['methods']['direction']['history'][1]['certificate'];w=c['chosen']
            row.update(U=c['U']['float'],c=c['c']['float'],lambda_=w['lambda']['float'],kappa=w['kappa']['float'],L=w['L']['float'],t=w['t']['float'],rhs=w['rhs']['float'],margin=w['margin']['float'],norm_rhs=c['norm_rhs']['float'],relative_margin=w['margin']['float']/float(GOAL))
        summary.append(row)
    print(json.dumps({'status':out['status'],'summary':summary,'lmi':out['equivalent_LMI_witness'],'output':str(dst)},ensure_ascii=False,indent=2))
```

## 付録B. 実際の主要出力

```json
{
  "status": "INDEPENDENT_FIXED_INPUT_REVIEW_CHECKS_PASS",
  "script_sha256": "9c4c074ec767436a196559549d90becb00a60e57faafc2c92050745b92b256bf",
  "summary": [
    {
      "input": "original_gapped",
      "selected_active": {
        "legacy_scalar": [
          0,
          1,
          2,
          3
        ],
        "norm_only": [
          0,
          1,
          2,
          3
        ],
        "direction": [
          0,
          1,
          2,
          3
        ]
      },
      "post_certificate_energy_width": 0.0006408249812011602
    },
    {
      "input": "original_small_gap",
      "selected_active": {
        "legacy_scalar": [
          0,
          1,
          2,
          3,
          4,
          5
        ],
        "norm_only": [
          0,
          1,
          2,
          3,
          4,
          5
        ],
        "direction": [
          0,
          1,
          2,
          3,
          4
        ]
      },
      "post_certificate_energy_width": 0.0002621922779750818,
      "accepted_witness": {
        "lambda": {
          "rational": [
            -97277751951202719,
            360287970189639680
          ],
          "float": -0.27
        },
        "kappa": {
          "rational": [
            209476089609093792008256005739033326043597766053305917609692676048035,
            25364726622855339863074884922038424496573145181388689046309100083740672
          ],
          "float": 0.008258558932006846
        },
        "L": {
          "rational": [
            -97277751951202719,
            180143985094819840
          ],
          "float": -0.54
        },
        "t": {
          "rational": [
            8728170400378908000344000239126388585149906918887746567070528168668125,
            648914256101382436616852282167282452884695163773642395449388659632594312
          ],
          "float": 0.013450421713366198
        },
        "rhs": {
          "rational": [
            45597494556816266170750209541807359588378926409963821951449807761965205642576285889125,
            11689800007894354207842179468477877238196030772974591295898930711375193605619037144875008
          ],
          "float": 0.0039006222968761975
        },
        "margin": {
          "rational": [
            145213184345143832577313541513018670550649585241817904018239385441946097487482836296879,
            1461225000986794275980272433559734654774503846621823911987366338921899200702379643109376000
          ],
          "float": 9.937770312380262e-05
        }
      },
      "U": {
        "rational": [
          -9007199254740993,
          36028797018963968
        ],
        "float": -0.25
      },
      "c": {
        "rational": [
          2026619832316723,
          5629499534213120
        ],
        "float": 0.36
      },
      "norm_rhs": {
        "rational": [
          10644208559996397532001683545207625,
          1594035135570192061782736042450747392
        ],
        "float": 0.006677524429967428
      }
    },
    {
      "input": "strong_external_coupling",
      "selected_active": {
        "legacy_scalar": [
          0,
          1,
          2,
          3,
          4,
          5
        ],
        "norm_only": [
          0,
          1,
          2,
          3,
          4,
          5
        ],
        "direction": [
          0,
          1,
          2,
          3,
          4,
          5
        ]
      },
      "post_certificate_energy_width": 0.009760339071496094
    },
    {
      "input": "near_external_energy",
      "selected_active": {
        "legacy_scalar": [
          0,
          1,
          2,
          3,
          4,
          5
        ],
        "norm_only": [
          0,
          1,
          2,
          3,
          4,
          5
        ],
        "direction": [
          0,
          1,
          2,
          3,
          4,
          5
        ]
      },
      "post_certificate_energy_width": 0.04433783870372138
    }
  ],
  "schur_diagnostics": [
    {
      "input": "original_small_gap",
      "lambda": -0.27,
      "Schur_min_eig_diagnostic": 0.0037392786550703564,
      "width_within_goal": true
    }
  ],
  "equivalent_LMI_witness": {
    "kappa": {
      "rational": [
        209476089609093792008256005739033326043597766053305917609692676048035,
        25364726622855339863074884922038424496573145181388689046309100083740672
      ],
      "float": 0.008258558932006846
    },
    "mu": {
      "rational": [
        -4075472616940267847256515950961967595723804558914690678458537419250813396112413321433,
        1827721173872732928594619116893919443304160440630420981249606921611863928456878848212992
      ],
      "float": -0.0022298109116418484
    },
    "lambda": {
      "rational": [
        -97277751951202719,
        360287970189639680
      ],
      "float": -0.27
    },
    "D": {
      "rational": [
        2765210171205484509,
        4503599627370496000
      ],
      "float": 0.614
    },
    "affine_goal_margin": {
      "rational": [
        435639553035431497731940624539056011651948755725453712054718156325838292462448508890637,
        7139535835440363002322730925366872825406876721212581958006277037546343470534683000832000000
      ],
      "float": 6.101790971801481e-05
    },
    "all_31_principal_minors_nonnegative": true,
    "scope": "same accepted witness; no new optimizer, lambda grid, or Hamiltonian"
  }
}
```

完全な独立チェックJSONのSHA-256：`6a54e4d8785bba243c1c433bbe0838178384b0998459e949d22c2c8f03bcf106`。コード実行により同内容を再生成できる。異なる線形代数環境では事後の浮動小数点固有値診断の末尾桁が変わり得る。有理数certificateと浮動小数点診断を混同しない。
