# N3 occupation-aware保証：限定移送設計の科学的レビュー

- 作成日：2026-10-12 JST
- レビュー対象：`HIROMU1015/Partially-Randomized-Trotter`
- 固定commit：`c2955ba0ef391eaf9102c711c378ab73b766fc9f`
- 対象設計：`n3_occupation_transfer_design_v1_20261012.md`
- レビュー状態：**完了**
- 科学的判断：**設計の主軸を支持。下記の限定的な補正後、実装と固定8入力の単発比較へ進む価値がある。**
- 直近の担当：**Codex（契約補正・実装・synthetic検査・source-bound実行planの固定）**
- 新8入力の科学実行：**今回のレビューでは実行していない。本レビュー承認を科学batchの実行承認へ読み替えない。**
- 中心研究テーマ：未採択。N1の現候補scan停止、N2の保留を維持。

本記録は、固定資料に書かれている内容、GPTが今回追加した数学的検討、外部一次資料との照合、推奨する設計変更を区別する。既存T3の結果や新8入力を、都合のよい結果へ変更するものではない。

## 0. 結論とその理由

現在の設計は、前回レビューの中心的な問いに対応している。

> occupation-aware条件が認証できる縮約のうち、強化された外部下界を同じように与えた既存方法の安全なORでは認証できないものが残るか。

この問いは、単に新しい式が正しいか、solverが終了するか、受理数が旧式より一件増えるか、という問いと異なる。前回のT3では、新しい条件が旧LMI族を厳密に上回る同情報の例と、より強い外部下界なら既存NORMでも通る例の両方が得られていた。今回の比較設計はこの交絡を直接扱っている。[D1][D2][D5]

判断は以下とする。

| 項目 | 判断 |
|---|---|
| 実one-body＋非負density、外部1mode、固定粒子数での保証の骨格 | 妥当。今回の導出確認で、成立範囲内の反例・証明破綻は認めなかった |
| 新8入力 | 変更しない。4 base×同normの2方向という限定比較に使う |
| 三下界×四判定 | 維持。raw結果と、保存証拠を継承した運用結果を分ける |
| c*での主比較 | 既存方法の保存済み証拠をc*へ持ち越す規則を明記してから使用する |
| 2-base基準 | 維持。0-base、1-base、未判定多発を同じ意味にしない |
| 条件族の厳密な分離 | 旧LMIの全κに対するseparatorと、NORMの厳密な不成立を確認した場合に限る |
| 生成費用 | 因子実験の実測と、単独運用経路から再構成した費用を区別する |
| 新8入力への実行 | 修正・実装・source固定と、別の明示的実行認可の後に一回だけ |
| 一般新規性・実分子・量子総資源の優位 | 未判定。このbatchの結果だけでは確定しない |

現v1の方向を全面的に差し戻す理由はない。ただし、**そのまま実行すると、有限探索による既存証拠の取りこぼしをOCCの追加効果へ混ぜる余地がある**。この点と、ν=1に関するテスト記述、結果分類、費用・予算の定義を実装前に補正する。

## 1. 使用した資料と確認の範囲

### 1.1 固定資料

以下はすべて固定commit `c2955ba0ef391eaf9102c711c378ab73b766fc9f` を対象としてGitHub接続から参照した。

| ID | 資料 | 役割 |
|---|---|---|
| D1 | [限定比較設計v1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/docs/research/n3_occupation_transfer_design_v1_20261012.md) | 仮説、入力、比較、費用、予算、STOPの契約案 |
| D2 | [数学検証・比較結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/docs/research/n3_occupation_redesign_mathematical_audit_20261012.md) | CAR、Schur証明、包含関係、T3の位置づけ |
| D3 | [入力生成source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/scripts/prepare_n3_occupation_transfer_inputs.py) | 新8入力の事前recipeと出力規約 |
| D4 | [独立証拠checker source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/src/trottertracks/representation_exploration/n3_occupation_certificate_audit.py) | 保存証拠の検証方法。新generatorではない |
| D5 | [独立証拠監査JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/artifacts/n3_occupation_redesign_preparation/2026-10-12/independent_witness_audit.json) | T3の756 checks、7 PSD行列、監視範囲、実行情報 |
| D6 | [数学fixture tests](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/tests/test_n3_occupation_certificate_audit.py) | 9 testsの内容、CARの9 subcases、改変拒否等 |
| D7 | [新入力pack](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/artifacts/n3_occupation_redesign_preparation/2026-10-12/proposed_transfer_inputs.json) | 固定係数。今回のGitHub読取りは冒頭とblob識別子。全内容はD3から同一byte列を再構成して照合 |
| D8 | [入力作成receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/artifacts/n3_occupation_redesign_preparation/2026-10-12/input_preparation_receipt.json) | 入力SHA、48,201 bytes、科学評価0 |
| D9 | [STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/artifacts/n3_occupation_redesign_preparation/2026-10-12/MANDATORY_STOP.json) | 現在の未認可範囲、N1/N2、中心テーマ未採択 |
| D10 | [固定commit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/c2955ba0ef391eaf9102c711c378ab73b766fc9f) | 親はN3 publication `64cf5ed227b7e4a74b37453cbe733db5f5d51f56` |

主要blob識別子：

- 設計書：`fb928f23a3de23ff84801ebee8053b06a3aee4fa`
- 入力生成source：`aadd3de942d76c603e8f761915edaecfc70efa05`
- 独立checker：`6247127affba04da342f40195f08c3c50c62c6b0`
- 新入力pack：`40f9406d77353d1e17973851931bc803e45ea621`
- 新入力SHA-256：`c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4`

前回のGPTレビュー `n13_transfer_recovery_scientific_review_2026-10-12.md` は今回もローカル添付を確認した。140,675 bytes、SHA-256 `ef84c62a8e160c8822a833e8a748ba44438a7aa7131352d36c153ec05c4bf1ec`。今回の設計判断は前回のT3検討を取り消さず、その次段の比較設計を対象とする。

### 1.2 今回、実際に行った作業

固定資料・sourceの読取り、保証の導出確認、比較集合・判定規則・費用経路の分析、一次文献の限定照合を行った。

さらに、GitHubから読んだ入力生成sourceのローカルコピーについてGit blob SHAを確認した上で、純粋な係数構成関数 `make_inputs()` のみを実行した。`main()` は呼んでいない。その出力を元sourceと同じ規約で直列化し、D7のGit blob SHA、D8のSHA-256とbyte数のすべてに一致した。各baseの2方向について、active一体行列、density、外部対角成分、結合L2 normの一致を確認した。

**この確認で新8入力のU、c、κ、保証条件、受理結果、ground energy、gapは計算していない。** 入力同一性・schemaの照合であり、科学batchではない。

今回新たに作った小さい例では、ν=1の帰着、証拠の下界間継承、κ=0のNORM埋込み、separatorの端点条件、正のscale変換を有理数で検査した。これは一般的な設計検査の5項目であり、新8入力でも新たな移送入力集合でもない。コードと出力は付録に収録する。

### 1.3 行っていない作業

Codexの9 tests・756 checksを今回のGPT環境でそのまま再実行したわけではない。保存監査と対応sourceを確認した。新generatorは未実装なので、その性能・完全性・実際の出力は未確認である。既存12入力の再scan、T3の新しい再探索、新8入力の保証生成、many-body対角化、quantum compile、sampling、N1/N2実行は行っていない。リポジトリへの書込みも行っていない。

公開・保存監査はimmutable CIや外部科学再現と同一視しない。既存global manifestのdescription欠落FAILが残るとの報告をglobal PASSへ変更しない。ただし、主要資料を取得できない問題とは区別する。

## 2. 現在の到達点の科学的意味

D5には、T3で新条件の有理数証拠、同じ情報を使う旧LMI族のseparator、より強い下界を使うNORM対照が検証された記録がある。

| 比較 | 保存監査で支持される内容 |
|---|---|
| 同じU,c_occ,δ | OCCの証拠は受理、NORMは不成立、旧LMI族は全κで不可能 |
| 同じU,δだがc_dhへ強化 | NORMも受理 |
| 新8入力 | 係数固定だけ。受理結果は未取得 |

この結果から分かるのは、**同情報での条件族の分離**であって、**強い既存の情報取得法まで含めた算法上の独立利益**ではない。D1が後者を問う設計になったことは妥当である。[D1][D2][D5]

新しい実行に情報価値があるのは、現在未取得の「強い共通下界の下でも追加受理が残るか」という結果が得られるためである。既存T3をもう一度検査することだけでは、この問いには答えられない。

## 3. 数学的骨格の確認

### 3.1 保証の対象

実対称one-body係数hと非負density相互作用を持つ

\[
H=\sum_{ij}h_{ij}a_i^\dagger a_j+\sum_{i<j}J_{ij}n_i n_j,\quad J_{ij}\ge0
\]

を固定粒子数νのsectorへ制限する。外部1mode dが空のP、占有されたQに分ける。active mode数をm、1≤ν≤mとする。

\[
H_\nu=\begin{pmatrix}A&B\\B^\dagger&C\end{pmatrix},\quad
A=d\Gamma_\nu(h_A)+V_{A,\nu},\quad V_{A,k}\succeq0,
\]

\[
C=h_{dd}I+d\Gamma_{\nu-1}(h_A+\operatorname{diag}J_{id})+V_{A,\nu-1}.
\]

実係数v_i=h_idに対するCARから

\[
BB^\dagger=d\Gamma_\nu(vv^T).
\]

途中indexの外部modeを除去するときのfermionic parityを省略しない。D4/D6はこの点を小さい有理行列の一致として扱っている。一般複素係数への拡張を今回の実入力classの検証と混同しない。[D2][D4][D6]

保証は固定ν-sector内の

\[
0\le a-E_0(H_\nu)\le\delta,\qquad a=\lambda_{\min}(A)
\]

である。全粒子数sectorを自由に変えたground energy、全状態の時間発展operator error、QPE全費用、任意の逐次軌道削除を保証しない。

### 3.2 固有値和の下界証拠

\[
\sigma_k(G)=\sum_{j=1}^k\lambda_j^\uparrow(G)
=\min_{0\preceq\gamma\preceq I,\,\operatorname{tr}\gamma=k}\operatorname{tr}(G\gamma).
\]

固有basisで固定k個を占有することから、\(\lambda_{\min}(d\Gamma_k(G))=\sigma_k(G)\)。Ky Fan原理は既知であり、その利用自体を新規性としない。[L1]

有理数τと有理対称行列Zが

\[
Z\succeq0,\qquad G-\tau I+Z\succeq0
\]

を満たせば、\(0\preceq\gamma\preceq I\)から

\[
\operatorname{tr}(G\gamma)\ge k\tau-\operatorname{tr}(Z\gamma)
\ge k\tau-\operatorname{tr}Z.
\]

したがって \(k\tau-\operatorname{tr}Z\) は有効な下界になる。数値固有分解はτ/Zのproposalであって証拠ではない。k=0は厳密zero、k=mはtrace恒等式を使える。PSD境界、zero pivot、丸め、bit capで証拠が得られないことを数学的不可行性とはしない。

### 3.3 OCC条件とSchur証明

\(a\le U\)、\(C\succeq cI\)、\(c>U\)、\(\delta>0\)、\(D=c-U+\delta\)を仮定する。

\[
K_\kappa=\kappa h_A-vv^T,
\]

\[
0\le\kappa\le D,\qquad
\underline\sigma_\nu(K_\kappa)\ge\kappa U-\delta D.
\tag{OCC}
\]

このときκ≥0とV_A≥0から

\[
\kappa A-BB^\dagger\succeq (\kappa U-\delta D)I.
\]

z=a−δと置くと、\(C-zI\succeq DI>0\)。よって

\[
\begin{aligned}
A-zI-B(C-zI)^{-1}B^\dagger
&\succeq A-aI+\delta I-BB^\dagger/D\\
&\succeq (1-\kappa/D)(A-aI)+\kappa(U-a)I/D\\
&\succeq0.
\end{aligned}
\]

Schur補元によりHν≥zI、変分原理によりE0(Hν)≤a。したがって所望の幅が得られる。c>Uは今回の契約のまま維持する。ここを事後的に緩める必要はない。[D2]

この導出は有限次元のblock正値性による。Feshbach–Schurの既存理論と関連するが、既知の固有空間projectionや真のgapを入力へ導入したものではない。[L2]

## 4. 既存方法との包含関係とテスト記述の訂正

### 4.1 旧LMIからOCCへの埋込み

旧証拠

\[
\kappa h_A-\mu I-vv^T\succeq0,\quad
\kappa U-\nu\mu\le\delta D,\quad0\le\kappa\le D
\]

があれば、τ=μ、Z=0を使える。したがって旧LMI族はOCCの数学的可行集合に含まれる。FIVEはその旧族の有限候補生成であり、独立したより広い条件族ではない。[D1][D2]

### 4.2 NORMからOCCへの埋込み

κ=0ならK0=−vvᵀで、1≤ν≤mに対し

\[
\sigma_\nu(K_0)=-\|v\|^2.
\]

τ=0、Z=vvᵀで厳密な証拠が得られる。よってOCCは

\[
\delta D\ge\|v\|^2
\]

というNORM条件を含む。既存証拠のfallbackを数値proposalへ再投入して発見し直す必要はない。ただしfallback成功をOCC固有のgainと数えない。[D2]

### 4.3 必須訂正：「ν=1へのnorm帰着」

D1のsynthetic項目にあるこの記述は、OCC全体がν=1でNORMと同値だという意味では正しくない。

正しい関係は次である。

- **κ=0で、OCCがNORMに一致する。**
- **ν=1で、OCCの数学的条件族が旧one-body LMI条件族に一致する。**

ν=1ではσ1=λ_minなので、OCCは

\[
\kappa h_A-vv^T-(\kappa U-\delta D)I\succeq0
\]

であり、一般にはκ>0の効果が残る。

具体的に、ν=1、h_A=diag(0,1)、v=(0,1/10)ᵀ、U=0、c=1、δ=1/250とする。D=251/250、κ=1/100でKκ=0なのでOCCと旧LMIは成立する。しかし

\[
\delta D-\|v\|^2=-187/31250<0
\]

でNORMは不成立である。これは新8入力と無関係の小さい反例であり、今回有理数で検査した。

現在のD6にこの誤った同値性を強制するproduction testが存在する、と主張しているわけではない。新generatorは未実装であり、**次のテスト仕様の文言を訂正する**という指摘である。

なお、数学的な族の一致と、有限の旧generator・有限のOCC generatorの発見集合の一致も別である。テストで数値探索の完全一致まで仮定しない。

## 5. 最も重要な補正：下界を強くしたとき、既存の証拠を失わせない

### 5.1 現v1の未規定部分

D1は三つのc行ごとに有限探索を行い、c*行のBASE-ORを主対照にする。またstandalone経路は弱い下界から早期受理する。[D1]

しかし、c0またはc_occで既に得られた旧法の証拠を、c*行へ継承する規則が明記されていない。有限探索では数学的可行集合の包含がそのまま発見集合の包含にはならない。

例えば、弱いc行で旧法の証拠が得られ、c*行では新しい有限proposalがそれを再発見できず、OCCだけが成功する場合、c*行のraw結果だけで追加gainを数えると、既存の早期終了算法は既に成功していたのに「OCCだけが認証した」と報告してしまう。

これは実装済みのバグを発見したという意味ではなく、**未実装の有限生成器に対して必要な比較規則が未規定**だという意味である。新8入力でこの事象が起きるかどうかは計算していない。

### 5.2 継承できることの証明

同じ入力、U、δでc2≥c1、Dj=cj−U+δとする。

旧証拠のκ、μを固定すると、PSD行列\(\kappa h_A-\mu I-vv^T\)は変わらない。さらに

\[
D_2\ge D_1,\quad
\delta D_2-(\kappa U-\nu\mu)
=\delta D_1-(\kappa U-\nu\mu)+\delta(c_2-c_1).
\]

したがって0≤κ≤D1で有効だった旧証拠は、c2でも有効である。

OCCも同じκと同じτ/Z証拠を固定すればKκは変化せず、affine余裕だけがδ(c2−c1)増える。NORMも同じ単調性を持つ。**新しいeigensolver探索なしで、保存証拠を再検証できる。**

c_occの有限証拠下界がc0以上になるとは常に仮定しない。一方、定義上c*≥c0,c_occなので、少なくともc*への継承は可能である。

### 5.3 推奨する実装・採点上の明確化

rawの96 primary recordsはそのまま保持する。各行の独自探索結果は上書きしない。

主対照には、一次の既存方法が各c行で既に生成した保存候補をc*で再検証した結果を含める。少なくとも、弱い下界での全受理証拠は必ず継承する。PSD部分を既に検証済みで、弱いcではaffine/box条件だけを満たさなかった候補も、c*で条件を再確認できる。この再利用の規則を結果前に固定し、新しいκ探索は行わない。

比較を形式化すると、一次のNORM/FIVE/OLD-DIRECTが生成した証拠候補集合を\(\mathcal W_{\rm old}\)として、

\[
B^*(i)=\bigvee_{w\in\mathcal W_{\rm old}(i)}\operatorname{Verify}_{c^*(i)}(w).
\]

一次のgain判定はv1のraw_OCC-at-c*を維持し、

\[
g(i)=\operatorname{raw\_OCC}_{c^*}(i)\land\neg B^*(i)
\]

とする。OCC側の別c行からの継承、既存方法からのfallback、運用unionも別fieldで記録するが、これらをraw_OCC-at-c*へ混ぜない。この定義を採れば、少なくとも既存証拠の単なる再発見失敗をOCCの一次gainとして数えることを避けられる。

主要な区分は、独自生成、同方法の別下界からの継承、他方法の既存証拠fallback、最終運用採否である。具体的なschema名はCodexの裁量とするが、元証拠のID・生成段階・検証先c・入力/ソースidentityを追えるようにする。

この補正は新入力追加・精度変更・探索予算追加ではない。保存済みの合法情報を既存対照から不必要に捨てないための、結果前の比較定義の追補である。

## 6. 三つの外部下界の妥当性と限界

D1の

\[
c_0=h_{dd}+(\nu-1)\ell,
\]

\[
c_{\rm occ}=h_{dd}+\underline\sigma_{\nu-1}(h_A+J_d),
\]

\[
c_{\rm dh}=\min_{|S|=\nu-1}C_{\rm diag}(S)
+\underline\sigma_{\nu-1}(O),\quad O=h_A-\operatorname{diag}h_A
\]

は、必要な下界証拠と完全な対角列挙があれば安全である。特にc_dhの対角項はh_dd、active対角、外部density、active内densityを正しい粒子配置で一度ずつ数える。単に一つの試行occupationの値を下界として使ってはいけない。上界Uの「一つのtrialが存在すればよい」性質とは逆である。[D1][D2]

c*=max(c0,c_occ,c_dh)は、**今回用意する三つの有効な下界の最大**であって、すべての係数下界の最適値ではない。成分の証拠生成が失敗し、残りの有効成分へfallbackした場合は、どの成分が利用できなかったかを保存する。

c0、c_occ、c*は必ずしも三つの異なる数値ではない。重複を三つの独立した成功として数えず、証拠と費用の共有を明記する。

c0→c_occの採否差は常に「改善」とは呼ばない。有限の認証下界が逆転した場合はそのまま報告し、c*で単調な証拠継承を確認する。比較対象のU,c,δをOCCだけ有利に変更しない。

## 7. 新8入力の妥当性

### 7.1 入力同一性とpaired比較

入力生成sourceとpackのhash一致を今回確認した。構成は次のとおり。[D1][D3][D7][D8]

| Base | active m | ν | scale | δ=scale/250 | active determinant数 | 外部側対角occupation数 |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 6 | 2 | 1 | 0.004 | 15 | 6 |
| R1 | 7 | 3 | 3/2 | 0.006 | 35 | 21 |
| L1 | 8 | 3 | 3/4 | 0.003 | 56 | 28 |
| S1 | 9 | 4 | 5/4 | 0.005 | 126 | 84 |

各baseのedge_pairとdistributed_threeは、結合vの方向が異なる一方、active h_A、density、h_ddが同じで、\(\|v\|_2\)も同じである。これらは係数比較だけで確認でき、U/c/採否の計算は不要である。

したがって共通U・共通下界の構成が同じように成功する場合、NORMの数学的採否は各pair内で同じになる。方向情報を用いる方法の差を調べるための、適切なpaired controlである。ただし、有限proposalの品質による差と、条件族の厳密な差はseparator等で分ける必要がある。

全baseでν≥2であり、前回の追加成功が一粒子sectorのT1だけだった問題に対し、今回は多粒子入力で問いを立てている。この設定は保持してよい。[D1][D3]

### 7.2 何をこの8入力から推定しないか

4 baseは同じrecipeから作った意図的な合成入力であり、実分子分布からの無作為標本ではない。m、ν、graph、scale、対角構造などがbase間で同時に変わる。96 method recordsを96独立標本とせず、8入力、4 baseとして報告する。[D1][D3]

今回のδはscale/250であり、前回の全例δ=0.004固定と同一条件の再試験ではない。旧12入力と新8入力の受理率を直接つないで改善率を主張しない。新8入力の内部で同じU,c,δを与えた方法比較が一次の根拠になる。

一様にh,J,v,U,c,δ,κをs>0倍すると、Kκと判定のaffine余裕はs²倍になる。したがって理想的な数学的可行性は一様scale変換で不変である。scale違いそれ自体を新しい物理条件での移送証拠として過大評価しない。なお、dyadic量子化やE=max(1,max|Gij|)を使う有限proposalの結果まで完全にscale不変とは仮定しない。

### 7.3 ceiling/floorの扱い

新8入力がすべて既存BASE-ORで受理される可能性、c*>Uの前提が多くで成立しない可能性、双方が有限予算で証拠を得られない可能性は残る。今回それらの結果を事前計算していない。

その可能性だけを理由に、結果を見ながら入力を選び直すべきではない。各入力でc*の前提、BASE-ORの採否、OCCの採否、未判定理由を記録して、実際に追加受理を示せる機会がどれだけあったかを解釈する。

事後にc*>Uの入力だけを残して元の8入力の分母を変更したり、2-base基準を下げたりしない。これは小さな限定判別であり、その範囲で追加利益が示せなかったことと、一般に利益が存在しないことは区別する。

## 8. 2-base基準と結論の分類

### 8.1 基準は保持する

2方向で2件成功しても同じbaseなら1 baseと数える。第5節のg(i)について、

\[
G=\#\{b:\exists i\in b,\ g(i)=1\}
\]

を用い、G≥2を有限pipelineの基準とする。この2という値は、単一baseの例だけで継続を決めないための研究上のgateであって、統計的有意水準や母集団への一般化確率ではない。[D1]

### 8.2 「基準未達」と「追加gainゼロ」を分離する

D1はG<2を一つのNO_OCCUPATION_SPECIFIC_TRANSFER_GAINラベルへまとめる。これを自然言語でも「gainが一件もない」と解釈すると、G=1の結果を失う。

機械ラベルの具体的名称は任せるが、少なくとも以下の情報を分ける。

| 観測結果 | 支持される報告 | 支持されない報告 |
|---|---|---|
| G=0、対象比較が完了 | この有限scopeで追加受理を観測しなかった | OCC族は数学的に役立たない |
| G=1 | 1 baseで追加受理、事前2-base基準は未達 | 追加受理0、移送成功 |
| G≥2、separatorが不足 | FINITE_PIPELINE_TRANSFER_GAIN | 旧条件族すべてに対する厳密分離 |
| G≥2、うち2 baseで適切なseparatorとNORM不成立を確認 | CERTIFIED_FAMILY_TRANSFER_SEPARATION（同じ入力・U,c*,δの指定族に限る） | 全既知算法・全下界に対する優位、量子総資源優位 |
| 下界生成・proposal・有理cap等の未判定が多い | finite未達と、族の価値についての判断不能を併記 | 物理的縮約不能、強い科学的否定 |
| soundness/source/schemaに重大違反 | 定義されたSTOP、影響範囲を特定 | 失敗を正常な負の科学結果として混ぜる |

未判定率だけを示すのではなく、どのstageがどの判別を妨げたかを残す。c≤U、数値proposal失敗、bit cap、共有予算枯渇、入力不整合は同じ理由ではない。

## 9. 旧LMI族のseparatorの意味

旧LMIはμを最小許容値\(\mu_0=(\kappa U-\delta D)/\nu\)へ下げることで、

\[
M_\kappa=M_0+\kappa R\succeq0,
\]

\[
M_0=\delta D I/\nu-vv^T,\quad R=h_A-U I/\nu,\quad0\le\kappa\le D
\]

と等価になる。[D1][D2]

W≠0、W≥0で

\[
\max\{\operatorname{tr}(WM_0),\operatorname{tr}[W(M_0+DR)]\}<0
\]

なら、traceはκに対してaffineなので全区間で負である。Mκ≥0ならPSD同士のtraceは非負になるため矛盾する。この証拠は、一点の探索失敗より強い。端点の片方だけが負では足りない。W=0、非PSD、数値toleranceによる負判定を受け入れない。

ただしNORMはν>1で一般に旧one-body LMI族の部分集合ではないため、WだけではNORMを除外したことにならない。CERTIFIED_FAMILY_TRANSFER_SEPARATIONでは、同じ有効c*でNORMのscalar条件が厳密に不成立であることも確認する。FIVEは旧LMI族の部分集合なので、旧族全体のseparatorがあればその族としては除外される。

16回以内というseparator proposal予算は有限であり、生成失敗は「separatorが存在しない」という結論ではない。具体的なproposal構成・丸め・候補順は実装時に結果前固定する。

一次ledgerを固定してから条件付きseparator段階へ進み、secondary診断が一次の候補・閾値・下界やraw採否を変更しないようにする。secondaryで得た別の情報は診断として保存し、一次pipelineの結果へ後から混ぜない。

## 10. 計算費用と資源予算

### 10.1 今回の費用に含めるもの

trial U、c0/c_occ/c_dh、κ探索、τ/Z proposal、有理PSD検査、証拠生成・保存、独立checker、cache、bit長を分ける。c_dhの対角全列挙とUのtrial列挙を無料扱いしない。[D1]

等しい最大eigenvalue呼出し数は、等しい計算時間を意味しない。OCCには追加のτ/Z構成と有理PSD検査がある。受理件数だけで効率的算法とは結論しない。

### 10.2 因子比較と単独運用の費用を区別する

三下界×四判定の因子比較では、別方法・後段の比較のためにも計算を行う。一方、単独BASE-ORは所定順序で早期受理すれば後段を呼ばない。[D1]

したがって、因子比較の総wallをそのまま単独算法の費用として課してはいけない。反対に、単独算法ではまだ得ていない後段や他方法のcacheを無料で使わせてもいけない。

原始処理の依存関係と実行eventを保存し、所定の単独経路で必要だった処理だけを課金する。対称に、OCC運用にもU/c取得、BASE-OR fallback、OCC固有部分を課金する。継承する証拠を取得した元の処理も費用から消さない。

ログから再構成した費用は「経路別再構成見積り」と表示する。実際のcold standalone実行時間と同じ精度で測定したとは表現しない。今回の承認範囲にないbenchmark再実行を追加しなくても、この区別を明記すれば比較は進められる。

同じbaseの2方向の間でも一体行列などは共有可能だが、batch共有利益と単一入力のcold費用は区別する。

### 10.3 予算の数え方を実装前に一意にする

D1は最大m=9、2case workers、各thread1、GPU0、850数値eig/case、512 exact PSD/case、180 dyadic spectral pairs/case、4096-bit guardなどを置く。[D1]

180は3下界×10κ×6τ/Z候補と一致する。これはOCC側だけで180組になる。外部下界の固有値和証拠まで同じcounterへ含めるなら、その分の定義が必要である。外部下界に何組使うか自体もsource固定前に明確にする。

ここで単にcapを増やすよう要求しているのではない。**counterが何を数え、どのstageへ予約され、cache hit・候補失敗・継承・fallback・独立checkerをどう扱うかを固定する**必要がある。

共通のcase capを先に実行する方法が使い切り、後に実行するOCCや旧対照が予定budgetを受け取れなかった場合、その失敗を方法固有の科学的失敗として採点しない。段階別予約または最大呼出し会計によって、規定比較が順序だけで変わらないようにする。

最悪の呼出し・PSD・byte・bit会計は新8入力の受理を計算しないsynthetic検査で確認する。既存の24 fixture上限の範囲にまとめ、細かな技術手順はCodexに任せる。

有理bit監視については、明示的な有理演算の中間値、結果、Fraction内部の一時整数まで監視したかを混同しない。現在のD5は監視範囲を限定して報告している。新実装の「全演算guard」という要件は、その実装範囲をsourceとtestで説明する。測っていない量まで測定済みとしない。

worker address-space上限と、parent/helperを含むaggregate RSSは別の指標である。総8 GiBの会計対象を明記する。wall/CPU time上限を今回新たに追加しない。将来の実行時間やspeedup倍率を、保存T3証拠一件の0.0246秒から予測しない。

## 11. このbatchが研究として判別するもの

### 11.1 正の結果が得られた場合

主比較で追加受理が複数baseにあり、旧族のseparatorも確認できれば、前回の単一development例から一段進む。

> 指定した非負density・一外部modeのclassにおいて、同じ係数由来U/c/δを与えた既存条件族では認証できない縮約を、occupation-aware条件が有限証拠として認証する。

これは検討に値する限定claimである。費用が受理coverageに見合うかを併せて評価し、次に生成法の有用性・証拠費用・入力classのどこを主要貢献にするか判断する。

ただし、今回のbound集合が全係数下界を尽くすわけではなく、入力は人工的で、下界・Uには小系のoccupation列挙を含む。正の結果だけで普遍的な新規性、多項式scaling、実分子での量子優位を確定しない。

### 11.2 負または限定的な結果の場合

強い既存BASE-ORだけで入力を十分認証できるなら、今回の用途で追加のOCC生成器を使う根拠は弱くなる。OCCを主題として採択せず、数学primitiveまたは保証ライブラリとして保持する判断が合理的である。

一方、共通下界がほとんど取得できなかったり有限proposalがcapで止まったりした場合、OCCの数学的価値について強い否定はできない。ただし、それを理由に自動で成功例探索やprecision追加を繰り返すこともしない。固定scopeの未達を保存し、追加作業の情報価値を改めて判断する。

1 baseだけの厳密分離は、移送gate未達でも一つの数学例として価値を保持する。2-base基準を後から下げて中心テーマを採択することはしない。

### 11.3 今回はtruth計算を追加しなくてよい

今回の主要taskは、機械検証できる十分条件による追加受理の有無である。valid certificateが所定の誤差幅を保証するため、すべての新入力のmany-body ground energyを計算しなければこの比較ができない、ということではない。

従って、D1の「many-body truthなし」を維持する。代わりに、certificateが得られなかった入力について実際には何件が縮約可能だったか、真の誤差にどれほど近いか、というrecall/tightnessは未評価とする。

## 12. 先行研究との位置づけ

今回の外部照合では、LiangほかのKy Fan trace最小化原理の式(1.1)と、DussonほかのFeshbach–Schur定理1.2を一次PDFの該当ページで確認した。D2の基礎数学との対応はあるが、これらを「新しい理論」として扱う根拠にはならない。[L1][L2]

近接するHamiltonian圧縮研究SQuISHは、Hamiltonianと近似ground-state波動関数を利用して反復的に切り詰める方法である。現在のN3の入力情報・保証taskとは同一ではないため、そのまま今回の数値BASE-ORの代わりにするものではない。しかし「Hamiltonianを小さくする」こと一般を新規性にするのも適切ではない。[L3]

新規性の候補は、指定classでどの情報を小さい係数証拠に保持し、既存条件族にない追加認証を、どの費用で生成できるかにある。今回の照合は基礎定理と近接研究の限定確認であり、世界初・既存研究の完全網羅を主張しない。今回のGOは新規性確定ではなく、研究を判別する限定実行へのGOである。

## 13. 代替方針との比較

| 方針 | 今回の採否と理由 |
|---|---|
| 数学監査だけをさらに繰り返す | 採用しない。新generatorの追加利益という未取得結果に答えない |
| v1を無修正で実行する | 採用しない。既存証拠の継承、ν=1記述、分類・費用定義が未整備 |
| 新入力を増やす、成功が出るまで条件を変える | 採用しない。現在の8入力の結果を未取得のまま固定する意味を失う |
| 直ちに実分子・一般二体・多外部へ拡張する | 採用しない。現在の証明範囲と主要claimを越える |
| N1/N2を再開する | 採用しない。このN3設計監査から再開の新根拠は得られていない |
| 限定補正後に実装し、別認可後に固定8入力を一回比較する | 採用する。現在残る問いに直接答え、費用・結果の範囲を限定できる |

## 14. Codexへ渡す作業範囲

次を一つの実装準備作業として進めてよい。途中の通常の技術判断を、毎回GPTへ戻す必要はない。

1. 固定commitを基点として設計書へ追補する。第5節の証拠継承と主対照、第4.3節のν=1訂正、第8節の未達内訳、第10節の費用・counter定義を反映する。元v1と変更理由を残す。
2. version付きgenerator、runner、独立checkerを実装する。既存保存T3専用checkerを新8入力のgeneratorと誤用しない。
3. 合成・境界・改変拒否テストを既定の小さいfixture scopeへまとめる。新8入力や旧12入力の受理を実装チューニングの目的で走らせない。
4. raw/transport/fallback/operationalの区分、同一input/U/c/δへの証拠binding、正確なNORM判定、旧証拠のOCCへの埋込み、下界間継承、primary seal後のsecondary診断を検査する。
5. source/input/planのhash、stage別予算、実行環境、ログ・出力・メモリ上限を固定する。入力packのbyte列は変更しない。
6. 必要な公開権限の下で専用branchへ成果をcommit・pushし、remoteから参照できる証拠索引を報告する。既存のdirty差分、旧source/results/STOP、別研究の作業、秘密情報を保護する。

この範囲の実装が今回の科学的判断をそのまま満たすなら、同じ比較設計の重要レビューをもう一度要求する必要はない。新8入力の科学実行は、source固定後の明示的な実行認可を受けてattempt1/retry0で行う。

実行を後から明示認可する際には、既定範囲の単発batch、独立保存監査、公開・remote再取得、結果報告までを一つの作業としてまとめることができる。結果が負でも追加scanを自動で行わない。

次にGPTの科学判断が必要なのは、新8入力の結果が得られたとき、または実装中に証明の破綻・重大な比較変更・新入力やprecision/予算の実質変更が必要になったときである。単にsourceが固定できたという技術進捗だけで同じレビューを繰り返さない。

## 15. 最終判断

**N3の限定比較は継続する。ただし、既存証拠を正しく再利用する強い対照と、誤解のない判定・費用会計を実装前に固定する。**

今回の判断は、数学の再検査を無期限に続ける判断でも、現在のN3を中心テーマとして採択する判断でもない。必要な補正は具体化できており、次の担当はCodexである。

新8入力の受理結果は未知のまま保持した。旧結果、2-base基準、入力数、粒子数・density class、δ policy、N1/N2、Mandatory STOPを勝手に変更していない。重要な科学的レビューはこの文書で完了している。

## 参考文献

- [L1] Xin Liang, Li Wang, Lei-Hong Zhang, Ren-Cang Li, *On Generalizing Trace Minimization*, arXiv:2104.00257. [一次PDF](https://arxiv.org/pdf/2104.00257)。式(1.1)をPDF第1ページで確認。ここで利用するKy Fan原理の出典。
- [L2] Geneviève Dusson, Israel Michael Sigal, Benjamin Stamm, *The Feshbach-Schur map and perturbation theory*, arXiv:2105.02058. [一次PDF](https://arxiv.org/pdf/2105.02058)。取得PDFの第4ページの定理1.2を確認。ここでの有限次元occupation projectionの保証は第3節で独立に導いた。
- [L3] Diana B. Chamakiほか, *Self-consistent Quantum Iteratively Sparsified Hamiltonian method (SQuISH): A new algorithm for efficient Hamiltonian simulation and compression*, arXiv:2211.16522 (2022). [一次abstract](https://arxiv.org/abs/2211.16522)。Hamiltonianと近似ground-state情報を使う近接研究として位置づけた。今回、同手法との実験比較を行ったわけではない。

D2記載のarXiv HTML URLは今回のweb取得ではcache missだったため、L1/L2はPDFの該当ページを確認した。これはGitHub側の研究資料欠損とは無関係である。

---

# 付録：今回の非production検査

以下はレビューの再利用用の記録である。新8入力の科学batch、未認可のgenerator、既存T3の再探索を実行するコードではない。

## A. 入力同一性の照合結果

入力生成sourceのGit blobを照合してから、その純粋関数だけを呼び、出力packのGit blobとSHA-256を確認した。下記はその結果である。結合normの比較はschema確認であって、NORM保証条件の評価ではない。

```json
{
  "scope": "coefficient identity and schema only; no guarantee predicates evaluated",
  "source_git_blob_sha": "aadd3de942d76c603e8f761915edaecfc70efa05",
  "pack_git_blob_sha": "40f9406d77353d1e17973851931bc803e45ea621",
  "pack_sha256": "c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4",
  "pack_bytes": 48201,
  "bases": [
    {
      "base": "C1",
      "m": 6,
      "nu": 2,
      "determinant_count": 15,
      "external_diagonal_count": 6,
      "equal_norm_pair": true,
      "common_active_h_and_density": true
    },
    {
      "base": "R1",
      "m": 7,
      "nu": 3,
      "determinant_count": 35,
      "external_diagonal_count": 21,
      "equal_norm_pair": true,
      "common_active_h_and_density": true
    },
    {
      "base": "L1",
      "m": 8,
      "nu": 3,
      "determinant_count": 56,
      "external_diagonal_count": 28,
      "equal_norm_pair": true,
      "common_active_h_and_density": true
    },
    {
      "base": "S1",
      "m": 9,
      "nu": 4,
      "determinant_count": 126,
      "external_diagonal_count": 84,
      "equal_norm_pair": true,
      "common_active_h_and_density": true
    }
  ],
  "new8_bound_evaluations": 0,
  "new8_acceptance_evaluations": 0,
  "eigensolver_calls": 0,
  "ground_solves": 0,
  "repository_writes": 0,
  "source_main_called": false
}
```

## B. 小さい数学的設計検査のコード

このコードのPASSは一般定理の代用ではない。一般的な導出は本文に示した。新8入力のファイルを読まず、NumPy・eigensolver・many-body対角化・optimizerを使わない。下界継承の例では同一のphysical C=2に対してc=1とc=2を使う。

```python
"""Small exact design checks. Does NOT read or evaluate the eight transfer inputs.

This is review-support code, not an implementation of the proposed generator.
It uses only rational arithmetic and diagonal/rank-one illustrative fixtures.
"""
from fractions import Fraction as F
import json
from pathlib import Path


def enc(x):
    if isinstance(x,F): return [x.numerator,x.denominator]
    if isinstance(x,dict): return {k:enc(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [enc(v) for v in x]
    return x


def diagonal_old_matrix(h_diag,v,nu,U,delta,c,kappa):
    D=c-U+delta
    return [[kappa*(h_diag[i]-U/nu)+(delta*D/nu)-v[i]*v[i] if i==j else -v[i]*v[j]
             for j in range(len(v))] for i in range(len(v))]


def exact_psd(M):
    n=len(M)
    if n==0 or any(len(r)!=n for r in M) or any(M[i][j]!=M[j][i] for i in range(n) for j in range(n)):
        return False
    R=[r.copy() for r in M]
    for k in range(n):
        p=R[k][k]
        if p<0:return False
        if p==0:
            if any(R[k][j]!=0 for j in range(k+1,n)):return False
            continue
        for i in range(k+1,n):
            for j in range(i,n):
                R[i][j]-=R[i][k]*R[k][j]/p
                R[j][i]=R[i][j]
    return True


def run():
    out={}
    # nu=1: OCC/old-family can certify while NORM rejects.
    h=[F(0),F(1)]; v=[F(0),F(1,10)]
    U,c,delta,kappa=F(0),F(1),F(1,250),F(1,100)
    D=c-U+delta; beta=sum(x*x for x in v)
    K=[[kappa*h[i]-v[i]*v[i] if i==j else -v[i]*v[j] for j in range(2)] for i in range(2)]
    assert K==[[0,0],[0,0]]
    occ_margin=delta*D-kappa*U
    norm_margin=delta*D-beta
    old=diagonal_old_matrix(h,v,1,U,delta,c,kappa)
    assert 0<=kappa<=D and occ_margin>0 and norm_margin<0 and exact_psd(old)
    out['nu1_counterexample']={'nu':1,'h_A_diagonal':h,'v':v,'U':U,'c':c,'delta':delta,'D':D,'kappa':kappa,
                              'K':K,'norm_margin':norm_margin,'OCC_margin':occ_margin,'old_LMI_PSD':True,
                              'statement':'nu=1 reduces OCC to the old family, not generally to NORM'}
    # Take the physical external one-body diagonal to be 2: in nu=1, C=2.
    # Hence both c=1 and c_stronger=2 are valid lower bounds for the SAME H.
    # A fixed successful witness transports upward (same input/U/delta).
    C_exact=F(2)
    mu=F(0); c_stronger=F(2)
    assert c <= c_stronger <= C_exact
    old_aff_weak=delta*D-(kappa*U-mu)
    D2=c_stronger-U+delta
    old_aff_strong=delta*D2-(kappa*U-mu)
    assert old_aff_strong-old_aff_weak==delta*(c_stronger-c)>0
    assert 0<=kappa<=D2
    # With fixed kappa and spectral witness, the OCC affine margin has the same increment.
    assert (delta*D2-kappa*U)-occ_margin==delta*(c_stronger-c)
    out['witness_transport']={'physical_C_in_this_nu1_fixture':C_exact,'c_from':c,'c_to':c_stronger,'fixed_kappa':kappa,
                              'margin_increment':delta*(c_stronger-c),'new_proposal_calls':0,
                              'PSD_matrix_unchanged_for_fixed_mu_or_spectral_witness':True}
    # At kappa=0, sigma_nu(-vv^T)=-||v||^2 for 1<=nu<=m.
    v3=[F(1,5),F(2,5),F(0)]; nu=2
    Z=[[x*y for y in v3] for x in v3]
    K0=[[-x for x in row] for row in Z]
    tau=F(0); lower=-sum(Z[i][i] for i in range(3))
    assert exact_psd(Z) and all(K0[i][j]+Z[i][j]==0 for i in range(3) for j in range(3))
    assert lower==-sum(x*x for x in v3)
    out['norm_embedding']={'nu':nu,'tau':tau,'Z':Z,'certified_lower':lower,
                           'statement':'kappa=0 with tau=0,Z=vv^T reproduces NORM'}
    # A separator must be nonzero PSD and negative at BOTH endpoints.
    W=[[F(1),F(0)],[F(0),F(0)]]
    M0=[[F(-1),F(0)],[F(0),F(1)]]
    MD=[[F(-2),F(0)],[F(0),F(3)]]
    def tr_product(A,B):return sum(A[i][j]*B[j][i] for i in range(len(A)) for j in range(len(A)))
    assert exact_psd(W) and sum(W[i][i] for i in range(2))>0
    endpoint=(tr_product(W,M0),tr_product(W,MD))
    assert max(endpoint)<0
    for t in (F(0),F(1,7),F(1,2),F(1)):
        Mt=[[(1-t)*M0[i][j]+t*MD[i][j] for j in range(2)] for i in range(2)]
        assert tr_product(W,Mt)<0
    bad_MD=[[F(1),F(0)],[F(0),F(3)]]
    assert not max(tr_product(W,M0),tr_product(W,bad_MD))<0
    out['endpoint_separator']={'endpoint_traces':endpoint,'one_negative_endpoint_is_insufficient':True}
    # Positive energy rescaling: kappa,delta,U,c scale by s and K/margins by s^2.
    s=F(3,4)
    assert (s*delta)*(s*D)-(s*v[1])**2 == s*s*norm_margin
    scaled_old=diagonal_old_matrix([s*x for x in h],[s*x for x in v],1,s*U,s*delta,s*c,s*kappa)
    assert scaled_old==[[s*s*x for x in row] for row in old]
    out['scaling']={'s':s,'matrix_and_margin_scale':s*s,'mathematical_acceptance_invariant':True,
                    'finite_rational_proposals_not_proved_scale_invariant':True}
    return {'status':'PASS','scope':'five illustrative protocol checks; no held-out guarantees or new transfer runs',
            'checks':out,'new8_bounds':0,'new8_acceptance_tests':0,'eigensolver_calls':0,
            'many_body_diagonalizations':0,'generator_runs':0}


if __name__=='__main__':
    result=enc(run())
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
```

## C. 数学的設計検査の出力

```json
{
  "status": "PASS",
  "scope": "five illustrative protocol checks; no held-out guarantees or new transfer runs",
  "checks": {
    "nu1_counterexample": {
      "nu": 1,
      "h_A_diagonal": [
        [
          0,
          1
        ],
        [
          1,
          1
        ]
      ],
      "v": [
        [
          0,
          1
        ],
        [
          1,
          10
        ]
      ],
      "U": [
        0,
        1
      ],
      "c": [
        1,
        1
      ],
      "delta": [
        1,
        250
      ],
      "D": [
        251,
        250
      ],
      "kappa": [
        1,
        100
      ],
      "K": [
        [
          [
            0,
            1
          ],
          [
            0,
            1
          ]
        ],
        [
          [
            0,
            1
          ],
          [
            0,
            1
          ]
        ]
      ],
      "norm_margin": [
        -187,
        31250
      ],
      "OCC_margin": [
        251,
        62500
      ],
      "old_LMI_PSD": true,
      "statement": "nu=1 reduces OCC to the old family, not generally to NORM"
    },
    "witness_transport": {
      "physical_C_in_this_nu1_fixture": [
        2,
        1
      ],
      "c_from": [
        1,
        1
      ],
      "c_to": [
        2,
        1
      ],
      "fixed_kappa": [
        1,
        100
      ],
      "margin_increment": [
        1,
        250
      ],
      "new_proposal_calls": 0,
      "PSD_matrix_unchanged_for_fixed_mu_or_spectral_witness": true
    },
    "norm_embedding": {
      "nu": 2,
      "tau": [
        0,
        1
      ],
      "Z": [
        [
          [
            1,
            25
          ],
          [
            2,
            25
          ],
          [
            0,
            1
          ]
        ],
        [
          [
            2,
            25
          ],
          [
            4,
            25
          ],
          [
            0,
            1
          ]
        ],
        [
          [
            0,
            1
          ],
          [
            0,
            1
          ],
          [
            0,
            1
          ]
        ]
      ],
      "certified_lower": [
        -1,
        5
      ],
      "statement": "kappa=0 with tau=0,Z=vv^T reproduces NORM"
    },
    "endpoint_separator": {
      "endpoint_traces": [
        [
          -1,
          1
        ],
        [
          -2,
          1
        ]
      ],
      "one_negative_endpoint_is_insufficient": true
    },
    "scaling": {
      "s": [
        3,
        4
      ],
      "matrix_and_margin_scale": [
        9,
        16
      ],
      "mathematical_acceptance_invariant": true,
      "finite_rational_proposals_not_proved_scale_invariant": true
    }
  },
  "new8_bounds": 0,
  "new8_acceptance_tests": 0,
  "eigensolver_calls": 0,
  "many_body_diagonalizations": 0,
  "generator_runs": 0
}
```

## D. 添付コード・検査結果のhash

- `protocol_math_checks.py`：5,254 bytes、SHA-256 `fd80268ca58aa49ba7a446ea68cd6fa45590fea254b77dc59fcfc3ac8db55a63`。
- `protocol_math_checks.json`：2,950 bytes、SHA-256 `d1bdd31263a4d76fb81b872dd950425fad3e889fe742551cb2b34a0a79453642`。
- `input_identity_check.json`：1,334 bytes、SHA-256 `5357abd612ba80bc8dae120e0c8541088a3a1a6065b90b0165ffd0142270dccc`。

入力生成sourceは固定GitHubファイルD3を使用した。本文書ではsourceを新規の研究成果として複製せず、固定blobへの参照と出力identityを残した。
