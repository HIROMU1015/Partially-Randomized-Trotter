# N3旧条件族との厳密分離：研究全体の再評価・解析的補強・次段方針

- 作成日：2026-10-12（Asia/Tokyo）
- 対象プロジェクト：研究／Hamiltonian simulationの表現・構成・前処理アルゴリズム探索
- レビュー対象：S1二例の旧条件族診断、および独立表現探索からN1・N2・N3へ至る研究経緯
- 最新診断source：`1aa12ab0b777b28f1428d9bb8417af93c4376aae`
- 最新診断結果：`86c69983544064d1ed7ee7910634c541c7b982b7`
- 最新公開記録：`26ed4b959bf5b59d23e2131d8be785ee701f0bc2`
- レビュー状態：ユーザーの明示的開始承認を受けて実施・完了
- 運用：更新済み共通指示の「研究全体の俯瞰と再評価」を適用。科学レビュー承認と新規計算の実行承認を分離する。

**この文書は新しい科学batchの実行認可ではない。既存の結果、事前gate、認可、source、Mandatory STOPを変更しない。**

## 0. 総合判断

**N3は、誤差保証付きactive-space縮約の理論・前処理アルゴリズムとして優先的に継続する研究候補とする。ただし、中心テーマの最終採択、新規性、独立論文の完成、量子資源優位は未確定である。**

次は「S1の一点に関する未確定事項を一つずつ診断する」段階をいったん終え、(i)これまでの理論と証拠を統合し、(ii)既知原理から自明に得られる部分と独立の研究差分を分け、(iii)所定の誤差予算で何を縮約するアルゴリズムなのか、を一つの研究構成として具体化する。

今回の判断の根拠は次の三点である。

1. 最新結果は、有限保存候補が弱かっただけという説明を排除した。同じ固定情報と指定幅で、NORM・FIVE・全旧LMIのκ/μ条件族を除外する有理数証拠がある。
2. 本レビューの追加解析により、保存された同じWから、NORM閾値以下の全幅で旧LMI族を排除できることを確認した。したがって同じ固定U,cでは、NORMを含む登録旧対照全体の最小幅の下端を解析的に確定できる。これ以上、同じ旧対照の最適幅を探索する必要はない。
3. 一方、占有制約、固有値和、Schur補元、半正定値表現は既知原理である。プロジェクト内の旧条件族を超えたことだけで、文献上の新規性や実際の量子計算上の利益まで確定したとはいえない。

| 項目 | 今回の方針 |
|---|---|
| N3の現在の位置付け | 理論・保証生成・縮約前処理の優先候補として継続 |
| 元の固定8入力の追加保証0、2-base未達 | そのまま保持 |
| S1の保存幅差 | 有効な事後的結果として保持 |
| 同じU,cでの登録旧対照最適幅 | 本レビューの解析的帰結としてNORM下端に一致すると確認 |
| S1旧対照の追加最適化scan | 現時点では不要 |
| 独立入力／実分子／多外部／一般二体への新計算 | 今回認可しない |
| N1 | 現構成の探索停止を維持 |
| N2 | 補助修正完了・追加探索保留を維持 |
| 親リポジトリの別Track | 参考のみ。契約・成果・停止位置を混同しない |
| 次の担当 | Codex：理論の独立照合、文献対応資料、幅指向generatorと縮約taskの仕様をまとめて準備 |

## 1. 研究の出発点と現在位置

### 1.1 当初の目的を変更しない

この研究の目的は、既存手法の性能比較だけを行うことではなく、Hamiltonian simulationにおける新しいアルゴリズムの構成、または本質的な改善を得ることである。PR内部だけの改良は別Trackで扱い、この系統ではHamiltonianの表現、構成、前処理、必要に応じた内外共同設計を対象としてきた。

このため、N3が現在扱っているactive-space射影誤差の保証は、量子計算の前処理という意味では元の研究範囲に入る。しかし、現状のN3が実証したのは古典計算で生成・検証できるエネルギー誤差certificateである。PRの新しい乱択則、時間発展回路の改善、最終的なQPE/RPE資源削減を実証したわけではない。

**「当初の目的との接続が残る」ことと、「当初の目的を達成した」ことを分ける。** 本レビューは目的を単なる既知法のbenchmarkへ置き換えない。同時に、現在の数学的成果だけで量子アルゴリズム全体の改善を宣言しない。

### 1.2 研究経緯を統合した見取り図

以下は、各段階の固定資料に基づく整理である。全過去batchの再実行や、全履歴の再審査を行ったものではない。

| 段階 | 試みた改善 | 得られた結果／問題 | 現在の研究への意味 |
|---|---|---|---|
| 独立表現探索A/B/C | square-factor/core再構成、補助空間を用いる表現、near-integrable構成 | 局所proxyの改善がPF誤差・native費用・同精度比較で維持されない例。B/Cも固定小系で強い直接対照を超える全体利益は未実証 | 表現上のきれいさや係数proxyだけで新算法の利益を判定しない |
| N1：縮退生成・frame構成 | 近い固有値をcluster化し、群内gaugeでbasis回路を削減 | 初期のbasis削減はmodel errorとshots増に相殺。復旧後の費用込みselectorは3 context×RZ/CXの全てでexactへ戻り、改善0% | 技術停止を解消した上で、現構成の科学的判断材料が得られた。単なるscan拡大の根拠はない |
| N2：signed charge | density相互作用を小さいcharge構成へ変換 | 小系ではcompute–phase–uncomputeの全費用が直接実装を上回る。初期fit目的と採否規範の不一致は技術課題として扱われ、補助修正は完了 | charge数やrankだけを目標にしない。修正完了自体は新規資源優位ではない |
| 初期N3 | 係数から外部sectorの下界を取得して射影誤差を保証 | 高い外部energyでは縮約を認証。弱い下界では不受理。方向情報への拡張を検討 | 真のground/gapを運用predictorへ入れず保証できる可能性 |
| N3直接generator・12入力 | norm-onlyを超える方向付き保証を未使用入力へ移す | direct5件、norm4件、five4件。direct受理集合は旧二法のORと一致。normとの差は一粒子T1に限定、事前2-base未達 | 強い既存対照と相互作用する多粒子系での検証が必要 |
| OCC再設計・固定8入力 | 固定粒子数の固有値和と外部下界強化 | c_occ+NORMだけで8/8、OCCも8/8、追加保証0。元の採否taskは飽和 | 下界取得の利益と判定条件の利益を分離。実験目的未達は保持 |
| 保存証拠の幅評価 | 新規証拠探索をせず保存certificateの保証下端を比較 | S1二方向でOCCが約20.424%／0.343%狭い。他6例のraw OCCはBASEより広い | 単一δでの採否では見えなかった情報。ただし保存poolの偏りが残る |
| 最新旧族診断 | S1の保存OCC幅を全旧LMIが達成できるか | 二方向で厳密dual分離、NORM不成立。FIVEは旧族に包含 | 保存候補の取りこぼしだけでは説明できない条件族の差を確認 |

初期A/B/Cと親リポジトリのTrack A/Bは別の分類である。ここでは前者をこの研究の履歴として、後者を必要な評価上の参考としてのみ扱う。[P1–P4]

### 1.3 共通して見えてきた研究上の教訓

N1/N2では「少ないbasis回路」「少ないcharge」といった中間量が、同じ科学taskでの全費用改善にならなかった。N3でも「一つの旧条件より強い」だけでは、強い下界と旧法ORを用意した時の利益にならなかった。

最新の分離は、この懸念のうち**登録旧条件族の取りこぼしに依存するのではないか**という点を解消した。しかし、より広い既知法との比較、情報取得費用、縮約後の量子計算までの接続は未解決である。

したがって、次に解くべき問題は、同じ形式の小さな反例を一つ増やすことではない。これまでの個別成果を、**入力情報→certificate生成→縮約可否→誤差会計→計算上の目的**という一つの構成へまとめることである。

## 2. 参照資料、取得状況、独立確認の範囲

### 2.1 固定された最新資料

| 役割 | 固定identity |
|---|---|
| Repository | HIROMU1015/Partially-Randomized-Trotter |
| 診断source | 1aa12ab0b777b28f1428d9bb8417af93c4376aae |
| 診断authorization commit | f40b1bc3fe1e3cffcfdc67b9d8e3c2dcbc9450ee |
| 診断結果 | 86c69983544064d1ed7ee7910634c541c7b982b7 |
| 診断公開記録 | 26ed4b959bf5b59d23e2131d8be785ee701f0bc2 |
| 診断plan SHA-256 | ef3c66d2e01e47027d9d382afc324aab9b7263aab3d550c3576c31e0f5bd6ff5 |
| 入力fingerprint | 81e90cb760d6e6e9774aad808ed40953bae0b461c18532bff78d7ead6e4e584d |
| 元の幅評価payload | 652667187ce83c7c36041820a0a752669134865f |
| 元のOCC generator source | 83f51cd8a4848572c66a5535c67d0d0a5f3868cd |
| 元の固定8入力primary seal | 8b1e75e89da78c13b82f24896db387adec6e76597ec12669a7549279592d0d69 |

GitHub connectorにより最新結果報告、数学契約、二つの完全な小さい入力、二つの完全なdual証拠、公開監査、および過去の主要結果を取得した。[P1–P12]

### 2.2 今回自分で検査したこと

公開connectorが返した有理数を標準ライブラリのみのコードへ転記し、次を28条件で照合した。

- 両入力のcanonical instance SHAが公開値と一致すること。
- 同じh_A、U、c、νと指定v、δを使っていること。
- 保存Wが公開されたrank-one／凸混合表現と全entryで一致し、PSD、trace1であること。
- 保存された二つの端点traceを元行列から再構成し、完全一致・厳密負であること。
- 保存NORM幅の上下端が多項式で正しくrootを囲むこと。
- 保存OCCのκ,sに関するboxとscalar保証条件が成立すること。
- 第5節の新しい全幅除外条件r<0、b>β²/νが厳密に成立すること。

これは**保存データの小さい厳密算術照合と解析的帰結の検証**であり、新しいκ、μ、W、OCC証拠、入力の探索ではない。新しい固有値計算、optimizer、many-body ground solve、回路compile、科学batchは0。

### 2.3 検査していないこと

保存OCCの全Z行列・全240候補・元の2,873,507条件をGPT側で再実行したわけではない。OCC証拠のPSDと旧U,cの有効性については、既存のsource-bound独立監査を根拠とする。今回のコードはその監査を置き換えない。

containerからのrawネットワーク取得はDNS制約により失敗した。GitHub connectorの必要資料取得は成功しており、上記算術はその返却値を用いた。ローカルへリポジトリ全体をダウンロードして全ファイルのraw bytesを再確認したとは主張しない。転記された入力の内容はcanonical instance hashで照合した。

GitHubへの69 blobsの公開・再取得、独立保存監査PASSは公開報告の記録として確認した。同一local環境での保存再検証であり、外部環境での科学再現、immutable CIではない。既存global manifestのdescription欠落FAILも維持する。[P5]

## 3. 最新結果そのものの評価

対象：star型S1、実対称one-body＋非負unordered density、一外部mode8、full10→active9、固定ν=4。元のδ_goal=0.005、scale1.25。化学geometry／basis／DF rank／L_D／PF時間窓は適用なし。[P4]

共通値は

\[
U=-\frac{1951}{1064},\quad
c=-\frac{233352716459809}{183068686024704},\quad
g=c-U=\frac{13609964515120091}{24348135241285632}>0,\quad
\beta^2=\|v\|^2=\frac1{576}.
\]

二つのδは、前段階で保存されたOCCの正の有理上端をそのまま使っている。新しいδ調整や丸めで選んだものではない。

| 入力 | 保存OCC δ（表示） | δ(g+δ)−β² | tr(WMδ(0)) | tr(WMδ(g+δ)) |
|---|---:|---:|---:|---:|
| S1_edge_pair | 0.002457958833301265 | −0.000356135481461 | −0.000280006092587 | −0.0670430164266 |
| S1_distributed_three | 0.003078217499141555 | −0.00000599340822845 | −0.0000312782815474 | −0.00511496743798 |

表の小数は表示であり、判定には公開された有理数を用いた。W≽0、trW=1、両端負により全κ区間を除外できる。数値探索の失敗、候補数、solver statusは不可行性の根拠ではない。

edge_pairのWはactive座標0の射影e0e0^Tである。distributed_threeのWはactive座標1とcoupling方向の二つのrank-one PSDの正の凸混合である。[P6–P9]

**今回の最新結果を、指定された同じU,c,δと旧条件族の範囲内で支持する。** また、全κ/μを除外したという意味で、前段階の「保存旧証拠がなかっただけではないか」という疑問は解消した。

## 4. 保証の数学的構造と証明範囲

### 4.1 現在のHamiltonian class

\[
H=\sum_{ij}h_{ij}a_i^\dagger a_j+\sum_{i<j}J_{ij}n_i n_j,
\qquad h=h^T,\ J_{ij}\ge0.
\]

一つの外部mode dの空／占有で固定ν-sectorを分ける。

\[
H_\nu=\begin{pmatrix}A&B\\B^\dagger&C\end{pmatrix},\quad
A=d\Gamma_\nu(h_A)+V_A,\quad V_A\succeq0,
\]

\[
C=h_{dd}I+d\Gamma_{\nu-1}(h_A+\operatorname{diag}J_{id})+V_{A,\nu-1},
\quad BB^\dagger=d\Gamma_\nu(vv^T).
\]

固定sectorでCARと外部空射影を用いる。外部indexが途中にある場合のfermion符号は基底同型に含める。全Fock空間のidentityと取り違えない。[P10]

上界a=λ_min(A)≤U、外部下界C≽cI、c>Uを持つとする。σν(G)をGの下からν個の固有値和とし、Kκ=κh_A−vv^T、D=c−U+δとする。

### 4.2 OCC十分条件の再確認

\[
0\le\kappa\le D,\qquad s\le\sigma_\nu(K_\kappa),\qquad
s\ge\kappa U-\delta D
\]

ならκV_A≽0より

\[
\kappa A-BB^\dagger\succeq(\kappa U-\delta D)I.
\]

z=a−δとするとC−zI≽DI>0。したがって

\[
\begin{aligned}
A-zI-B(C-zI)^{-1}B^\dagger
&\succeq A-aI+\delta I-BB^\dagger/D\\
&\succeq(1-\kappa/D)(A-aI)+\kappa(U-a)I/D\succeq0.
\end{aligned}
\]

Schur補元と変分原理により

\[
0\le a-E_0(H_\nu)\le\delta.
\]

これは固定ν-sectorの基底エネルギー射影差である。状態overlap、operator time-evolution norm、励起状態誤差、QPE全費用を同時に保証したものではない。

### 4.3 有理数の固有値和certificate

\[
Z\succeq0,\quad K_\kappa-\tau I+Z\succeq0
\]

を有理数PSDとして検証すれば、rank-ν射影Πについて

\[
\operatorname{tr}(\Pi K_\kappa)\ge\nu\tau-\operatorname{tr}(\Pi Z)
\ge\nu\tau-\operatorname{tr}Z=:s.
\]

その最小値がσνなのでsは有効な下界である。提案に浮動小数計算を使っても、実際の認証は有理数PSDとscalar不等式に依存する。有限proposalが全ての有効証拠を発見することまでは保証しない。

### 4.4 旧条件族の定義を固定する

本研究のOLDは

\[
0\le\kappa\le D,\quad K_\kappa-\mu I\succeq0,\quad
\kappa U-\nu\mu\le\delta D.
\]

μ≥μ0=(κU−δD)/ν。可行μをμ0へ下げるとPSDを保つため、全μを含む可行性は

\[
M_\delta(\kappa)
=\kappa\left(h_A-\frac{U}{\nu}I\right)-vv^T+rac{\delta(g+\delta)}{\nu}I\succeq0
\]

とκ∈[0,g+δ]に等価である。FIVEの有効候補はこの条件族に含まれるが、NORMは別条件としてORに含める。[P11]

ここでいう「全旧条件族」は、この定義における全κ/μである。他の下界、別のstate情報、別のLMI、文献上の全アルゴリズムを含むものではない。

## 5. 今回のレビューで得た追加帰結：登録旧対照の最小幅

**本節は公開診断の結論を単に転記したものではなく、保存証拠から今回導出した新しい解析的帰結である。元報告は指定δでの分離までを主張し、最適幅は未判定としていた。元報告を変更しない。**

### 5.1 全幅を除外する十分条件

\[
R=h_A-\frac{U}{\nu}I,\quad
r=\operatorname{tr}(WR),\quad b=v^TWv,
\quad W\succeq0,\ \operatorname{tr}W=1.
\]

保存Wが

\[
r\le0,\qquad b>\frac{\beta^2}{\nu}
\tag{A}
\]

を満たせば、NORMの閾値

\[
\delta_N=\frac{\sqrt{g^2+4\beta^2}-g}{2}
\]

以下の全幅0<δ≤δ_Nに対し、全κ≥0で

\[
\begin{aligned}
\operatorname{tr}(WM_\delta(\kappa))
&=\kappa r-b+\frac{\delta(g+\delta)}{\nu}\\
&\le-b+\frac{\beta^2}{\nu}<0.
\end{aligned}
\]

よってOLDは不可行であり、FIVEも不可行。NORM自身の数学的受理下端はδ_Nである。したがって

\[
\boxed{
\inf\{\delta>0:\mathrm{NORM}\lor\mathrm{FIVE}\lor\mathrm{OLD}_{\rm all\ \kappa,\mu}
\text{が同じ固定情報で認証する}\}=\delta_N.
}
\tag{B}
\]

この証明はκの有限探索に依存せず、許容区間をさらに広げた全κ≥0さえ排除する。有理数出力ではδ_Nが無理数の場合、上から近づく安全な幅を用いるため「下端／infimum」と表現する。

### 5.2 二つの保存Wで条件(A)が成立する

| 入力 | r=tr(WR) | b=v^TWv | β²/ν−b |
|---|---:|---:|---:|
| S1_edge_pair | −0.118915673191989 | 0.000625000000000 | −0.000190972222222 |
| S1_distributed_three | −0.009044876481795 | 0.000463807707268 | −0.000029779929490 |

全値は厳密有理数で照合し、小数は表示だけである。新しいWの構成・探索をしたのではなく、公開されたWをそのまま用いた。両端の負性だけから(A)が自動で導けるとは仮定せず、h_A,v,UとWからr,bを実際に厳密照合した。

### 5.3 幅改善率の新しい解釈

登録対照の数学的最小幅の下端は両入力で

\[
\delta_N\simeq0.0030888226617.
\]

保存されたOCCの正の有理certificate幅をδ_sとすると、達成された改善率は

\[
1-\frac{\delta_s}{\delta_N}.
\]

保存NORM root区間の外向き多項式検証から得られる値は、edge約20.4240870225%、distributed約0.3433399621%である。

**今回の解析を根拠に、同じ固定U,cと登録された対照族に限れば、この数値は有限保存BASEだけでなく、NORMを含む旧対照全体の最小幅に対する達成改善として説明できる。**

ただし、次を必ず区別する。

- OLD LMI単独の最適幅を求めたわけではない。NORMを含むORの下端が求まった。
- 理想OCC族の最小幅を求めたわけではない。保存certificateが達成した幅を用いた。
- より強いU,cを取得する別アルゴリズム、一般のSDP、文献上の全手法に対する最適性ではない。
- 20.424%／0.343%は真の物理誤差や量子資源が減った割合ではない。
- 元のδ_goal=0.005での追加保証0、2-base基準未達を変更しない。

### 5.4 次の作業への影響

同じ固定S1情報において登録旧対照の最小幅を数値探索することは、もはや優先課題ではない。Codexに必要なのは本補題と保存有理数の独立照合・記録であり、新しいold-family batchやδ scanではない。

一方、文献上の別の保証との比較や、別のU,c取得法の利益まで本補題で解決したわけではない。それを調べる場合は別の科学的比較として扱う。

## 6. 何がOCCを強くしているか、どこが既知か

### 6.1 固定粒子数の占有制約を戻すという説明

Hermitian Kについて

\[
\sigma_\nu(K)=\min_{\gamma\in\mathcal F_\nu}\operatorname{tr}(K\gamma),\qquad
\mathcal F_\nu=\{\gamma:0\preceq\gamma\preceq I,\ \operatorname{tr}\gamma=\nu\}.
\]

このFantopeの変分表示は既知である。最大固有値和の公式を−Kへ適用すれば最小固有値和の式となる。[L1,L2]

一方、旧条件族が使う最良のscalar固有値下界は

\[
\nu\lambda_{\min}(K)
=\min_{\gamma\succeq0,\ \operatorname{tr}\gamma=\nu}\operatorname{tr}(K\gamma).
\]

こちらはγ≼Iを課さない。形式的に同じ最小一体方向へν個分の重みを置けるため、固定fermion粒子数で使える情報より粗い。

したがって、旧条件からOCCへの改善は「固定粒子数sectorの占有上限を捨てずに下界へ残す」と説明できる。これは両条件族の差の明確な解釈である。ただし、**Pauli占有上限や固有値和そのものを新発見と主張しない。**

ν=1ではtrace1かつPSDならγ≼Iは自動的なので、理想OCCと旧LMI族は一致する。κ=0ではOCCとNORMが一致する。有限proposalの保存集合まで一致することとは別である。

### 6.2 固定δでの小行列SDP表示

同じU,c,δを固定すると、OCCは

\[
0\le\kappa\le D,\ Z\succeq0,\ K_\kappa-\tau I+Z\succeq0,\
\kappa U-\nu\tau+\operatorname{tr}Z\le\delta D
\]

というm×m行列の凸可行性問題として表せる。exactな固有値和と同値な実数証拠は、Kを対角化してZ=(τI−K)_+とし、適切なτを選べば構成できる。有限bitの有理generatorでは境界・縮退・丸めによる未判定が残り得る。

この表示は理論整理・algorithm設計に役立つが、SDP化自体を新規性にはしない。研究として重要なのは、与えられた情報・誤差予算・古典計算費用のもとでどのcertificateを生成できるか、その構成が既知法を実質的に超えるかである。

### 6.3 NORMに対する改善機構

既存数学契約で導出した

\[
E_{\mathrm{forced}}(u)=u^Th_Au+\sigma_{\nu-1}(h_A|_{u^\perp}),\quad u=v/\|v\|
\]

は結合方向uを占有する最小一体energyであり、相互作用込みの真のground energyではない。E_forced≤Uなら理想OCCもNORMより狭い幅を得られず、E_forced>Uなら理想条件族にはNORMを改善する方向が存在する。[P10]

保存証拠だけでもs≤σν(Kκ)≤κE_forced−β²より

\[
E_{\mathrm{forced}}-U\ge\frac{\beta^2-q}{\kappa},\qquad q=\kappa U-s,
\]

が得られる。S1二方向の右辺は約0.24553998、0.01884604で正である。この確認は新しいE_forced固有値計算ではない。

ただし、この機構はNORMとの差の説明であり、旧LMIとの分離は最新のdual証拠と第5節によって別に支えられている。NORMとの差だけで旧法全体への優位を推定しない。

## 7. 先行研究との関係と新規性の判定

### 7.1 今回確認した一次資料

| 資料 | 実際に確認した内容 | N3へ適用する位置付け／限界 |
|---|---|---|
| Liangほか, On Generalizing Trace Minimization, arXiv:2104.00257v1 | PDF第1節・式(1.1)等。Ky Fan trace最小化の既存性 | 最小固有値和の原理は既知。N3のactive-space算法との同一性を示す論文ではない |
| Vuほか, Fantope Projection and Selection, NeurIPS2013 | 公式PDF第2節、Fantope定義・式(2)・rank-d射影の凸包表示 | 0≼γ≼I,trγ=νの表現が既知。sparse PCAのアルゴリズムをN3の直接資源対照とはしない |
| Dusson–Sigal–Stamm, arXiv:2105.02058v1 | PDFの仮定(A)–(C)、Theorem1.1と1.2、Schur map | Schurによる固有値誤差評価は既知。孤立固有空間やgap仮定を現在のoccupation射影へ無条件移植しない |
| Chamakiほか, SQuISH, arXiv:2211.16522 | arXiv公開abstract。PDF全文取得は成功していない | Hamiltonianと近似ground-state wavefunctionを用いる反復的疎化・simulation/compression。具体的保証の全文比較は未了 |
| Greiner–Gauss–Eriksen, arXiv:2406.11343v1 | HTML本文2.3節を中心にscreening誤差推定・reference active space設計 | lognormal model、Monte Carlo等によるempirical構造を使う方法。現在のexact rational certificateと保証意味論が異なる。任意手法全体の保証欠如とは言わない |

PDFはwebの解析本文を参照し、可能な箇所はscreenshotで式・仮定を確認した。Fantopeのscreenshotはツール側で失敗したため、第2節は取得済み解析本文に基づく。表・グラフの数値をscreenshotなしで新たに採用したものではない。

### 7.2 新規性について現在言えること

**確認済み：** プロジェクトで定義されたNORM＋FIVE＋旧LMI条件族に対して、OCCがより狭い幅を与える有限入力と有理数証拠がある。第5節では同じ情報下の比較下端まで解析的に定めた。

**未確定：** 文献上、同じclass・同じ利用情報・同じ誤差意味論・同程度の取得費用で、この構成を包含する保証生成法が既にあるか。今回の対象文献確認は、同じ手法が存在しないことの証明ではない。

内部対照を数学的に超えたことは重要だが、「全旧法を超えた」「世界初」「既知理論を本質的に改善」とはまだ表現できない。

### 7.3 独立論文の候補として必要な中身

主張を次のように分解することを勧める。

1. 所定のone-body＋非負density classで、利用するtrial上界・外部下界を明示した射影誤差certificate。
2. 占有制約を残す条件と、捨てる条件の関係、NORM改善の可能／不能機構、厳密分離を説明する理論。
3. 有限計算・有限bit・明示未判定を持つ証拠生成法。適用可能性と情報取得費用も含める。
4. どの入力modeをどの誤差予算で削減するかという構成則、または限定classで有意義な量的優位を示す理論。
5. 上のclaimと既知研究の定理・条件を対応させた比較。

単一の数値成功例を別baseへ増やすだけでは、この一式は成立しない。一方、実分子実験がなければ必ず論文にならない、とする理由もない。非自明な理論と算法の差が閉じていれば限定classの方法論研究は選択肢になる。ただしその完成度・先行性はまだ確認が必要である。

## 8. 研究の本来の目的への接続と、まだ埋まっていない部分

### 8.1 何を縮約するかを決める算法へ

現在の証拠は、指定された一外部modeを空に射影してよいかを判定する。入力に応じてmodeを選び、縮約Hamiltonianとcertificateを返す全体構成は未評価である。

当初の前処理研究との接続を明確にするには、少なくとも「入力class・粒子数・誤差予算・候補mode集合→共通U/cの取得→安全な保証選択→縮約Hamiltonianと検証可能証拠」というtask仕様が必要である。mode選択に真のground/gapや事後truthを使わない。

この記述は研究方針の提案であり、今ここで候補mode追加や新しいselection計算を実行したという意味ではない。

### 8.2 誤差幅が縮むことと量子計算が安くなること

同じHamiltonian・同じ射影なら真の射影誤差は変わらない。変わったのはその上界である。より厳しい許容誤差で同じ縮約を採用できる、または適切な全体誤差会計で他の誤差へ割ける余地が増える可能性がある。

しかし、現在の結果は1modeの削減であり、PF回路depth、T count、state preparation、Hadamard wrapper、shot数、QPE全費用を評価していない。10→9というmode数差だけから最終資源削減率を導かない。

元のS1運用費用の再構成見積りはδ_goal=0.005向けである。新しい狭い幅をcold生成する費用とは異なる。事後監査3–6秒のwallをgenerator費用と混ぜたり、20.4%の保証幅縮小を20.4%の量子費用削減へ変換したりしない。

### 8.3 適用範囲と古典費用

- c>Uは厳しい外部sector分離の前提である。満たさない入力の不受理を物理的縮約不能としない。
- m×m PSDでcertificateを検査できることと、必要なU,cを大系で安く取得できることは別である。
- 現方式のdeterminantや外部対角occupationの列挙は、一般fillingでは組合せ的に増える。単に行列が小さいという理由で全pipelineを多項式と呼ばない。
- 非負density classは一般の量子化学二体Hamiltonianではない。一般軌道回転でdensity形式が保たれるとは限らず、N1のframe改善をそのままN3へ合成できない。
- 多外部modeや逐次削除ではBの構造、sector、誤差の合成を再検討する必要がある。本レビューはその拡張を証明していない。
- 有理数監査はsoundnessを支えるが、独立データ数や新規性を増やすものではない。大きいcheck数を科学的効果量として扱わない。

## 9. 代替研究方針の比較と優先順位

| 方針 | 現在の根拠 | 主な弱点 | 判断 |
|---|---|---|---|
| N3の理論・証拠生成・縮約taskを統合 | 新しい厳密分離と解析的baseline最適下端があり、未解決点が具体的 | 既知原理との差、一般的取得費用、全体taskとの接続は未了 | **最優先の継続候補** |
| S1だけで新しい最適幅・κ scanを繰り返す | 個別数値の精密化 | 第5節で旧OR下端は決着。局所問題の目的化になりやすい | 進めない |
| すぐ独立8入力や多数分子へ展開 | 経験的coverageを増やせる可能性 | 新規性・task・generator目的が曖昧なままでは同じ問題が残る | 今は優先しない |
| N1現候補探索を再開 | native構成基盤とframe機構 | 費用込みselectorで0%。新しい機構なしでは成功探索になる | 停止維持 |
| N2現charge構成を再開 | 構成・符号・work registerの基盤 | 小系full costの負担、修正完了を超える新しい根拠なし | 保留維持 |
| 全面別テーマへ移る | 元の広い研究目的には他の選択肢がある | 現在得た分離を研究構成として評価する前に放棄する理由も乏しい | N3統合で差分・taskが成立しない場合の選択肢 |

これまで費やした作業量を継続の理由にはしない。N3を優先する理由は、現在の候補の中で、数学的な差と次に判別すべき問題が最も具体化しているからである。N1/N2の一般的不可能性を認定して順位を付けたわけではない。

## 10. 次のCodex作業：一つの統合パッケージにまとめる

### 10.1 理論と既存証拠の独立整理

第5節のr,b条件、全幅排除、登録旧OR下端の式を独立に確認する。保存W/h/v/U/c/δと既存OCC証拠へのbindingを保ち、必要な有理数照合と反例fixtureを整備する。

結果は元の報告を上書きせず、新しい解析的帰結として追補する。必要な検算は保存値・既存式から行い、新old-family optimizerや新S1 batchを追加しない。

既存のOCC十分条件、Fantope表示、NORM改善可能／不能条件、旧法包含・分離、finite-generatorの限界を一つのtheorem/claim mapへ整理する。個々の式を毎回別のGPT reviewへ戻さない。

### 10.2 文献上の対応を閉じる

本レビューの一次資料とその参照文献について、取得可能な全文の定理・式・入力情報・費用・保証対象を対応表にする。Codexは情報収集・同値変形・条件差の抽出を進め、未取得の全文は未取得と記録する。

特に、(i)固定粒子数のspectral bound、(ii)Schurによるactive-space error、(iii)RDM／SDPで同じ証拠を表現する方法、(iv)Hamiltonian sparsification／active-space選択の実際のtaskを区別する。単に関連語が一致するだけの論文を同一法としない。逆に分野が違うというだけで、同じ数学的構成を無視しない。

新規性・主要claimの最終採否はGPT／ユーザーへ戻すが、文献一本ごとの資料取得や軽微な式確認では戻さない。

### 10.3 幅指向generatorと利用taskの仕様

元generatorの目標幅での早期終了と、幅を小さくする研究目的の違いを解消する仕様を作る。数学的に可能な証拠族、有限予算の生成器、保存証拠の再利用を分ける。

最低限、入力、利用できるU/c、出力certificate、失敗／未判定、classical取得費用、NORM・旧法との同情報比較、縮約決定に使う誤差予算を定義する。運用では既存法との安全なORを維持し、そのfallbackを新しいOCC利益に数えない。

固定δでの小行列可行性・δに関する単調性は設計の材料になる。どの数値法・cache・bit管理・solverを使うか、syntheticの細部はCodexに任せる。目的は毎回準備→細部reviewを繰り返すことではなく、科学的な論点が閉じた仕様と技術検証を一括で揃えることである。

### 10.4 次の実験案は必要性が分かるものに限定

独立データの追加を計画するなら、今回のS1と既存8入力はdevelopmentとして固定し、結果後のthreshold変更を確認的成功へ混ぜない。

評価項目は「一つのδで何件通ったか」だけでなく、どの縮約決定が変わるか、同情報の保証幅、取得費用、未判定、既存法のcoverageを結び付ける。どの規模・入力・予算が必要かは、そのtaskから導く。今回のレビューでは任意の新しい2-base数値gateや新batch上限を追加しない。

新たな入力、固有値計算、certificate探索、量子compile、registered診断を行う場合は、仕様と固定sourceを用意し、別の明示実行承認を得る。元のone-shot認可や本レビュー承認を流用しない。

## 11. 次にGPTへ戻る条件と研究の終点

次の重要レビューは、次のいずれかの科学的状態になった時にまとめて行う。

- 理論・algorithm仕様・先行研究対応が揃い、独立した研究差分があるか判断できる。
- 既知の定理／算法が現在の主張を実質的に包含すると分かった。
- 新しい承認済み検証で、縮約決定・保証幅・取得費用の関係に重要な新結果が得られた。
- 数理の前提、保証意味論、主要対照、対象classを変える必要が出た。

通常のsource修正、synthetic PASS、資料公開ごとに詳細reviewを要求しない。新たな重要レビューを行う際は、開始承認を得る。

### 継続を支持する条件

文献との差が具体的に残り、利用する情報・費用を含む保証生成／縮約taskが閉じ、既存対照では得られない有意義な保証または理論があること。実分子での成功のみを絶対要件とはしない。

### 縮小・終了・別方向への転換を支持する条件

主張の本体が既知理論の直接利用に尽きる、より妥当な同情報対照で差が消える、certificate情報の取得費用が目的とする削減に見合わない、あるいは狭い幅が実際の縮約判断を変えるtaskを構成できない場合。これらを確認した場合は保証primitive・negative結果・検証基盤として保存し、中心テーマ探索を別方向へ戻す。

これは事前にN3継続を決めつけるものでも、文献調査一件で自動終了するものでもない。科学的差分と目的への接続を明示して判断する。

## 12. 主張の境界と最終結論

### 支持する主張

1. 指定されたS1二例、固定U,cと保存OCC幅で、NORM・FIVE・全旧κ/μ族への厳密分離がある。
2. 保存された同じWは第5節のより強い条件を満たすので、この固定情報では登録旧ORの最小保証幅の下端はNORM下端に一致する。
3. 保存OCC証拠はその下端より約20.424%／0.343%狭い幅を達成する。
4. 旧対照との差は有限候補の不足だけに依存しない。
5. N3を理論・保証生成・縮約前処理の研究候補として継続する根拠がある。

### 支持しない主張

- 元の独立移送／2-base gateを達成した。
- OCCが全既知算法に勝つ、あるいはOCC最適幅が求まった。
- 真の射影誤差が20.424%減った。
- 量子回路・T count・shot・QPE総資源が同率で減る。
- 一般二体、任意軌道回転、多mode逐次縮約へ同じ証明を適用できる。
- コード検査数の大きさから新規性や一般化を認定できる。
- 文献上の優先性を網羅的に確認した。

**最終判断：N3を優先候補として継続する。ただし点ごとの診断を積み重ねる進め方から、誤差保証付き縮約算法としての理論・先行研究差分・利用taskを統合する段階へ移る。**

このレビューは最新の分離を受けて評価を前進させるものであり、元の否定的結果を消すものではない。今回新しい科学batchは実行せず、Mandatory STOPとN1停止・N2保留を維持する。

## 13. 参照資料

### プロジェクト一次資料

P1. 初期独立表現A/B/Cの履歴（当該固定commitの冒頭）。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c24dcede10ed9726e1e0b1030eca749ba7bc5cba/docs/research/研究概要・現状.md

P2. N1/N2構成・N3係数下界の初期feasibility。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/docs/research/hamiltonian_algorithm_design_results_20261011.md

P3. N1復旧後の固定結果。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/docs/research/n1_decode_recovery_v1_results_20261011.md

P4. 最新旧族診断結果。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86c69983544064d1ed7ee7910634c541c7b982b7/docs/research/n3_old_family_v1_results_20261012.md

P5. 最新公開・保存再取得監査。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/26ed4b959bf5b59d23e2131d8be785ee701f0bc2/docs/research/n3_old_family_v1_diagnostic_publication_20261012.md

P6. edge入力。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86c69983544064d1ed7ee7910634c541c7b982b7/artifacts/n3_old_family_v1/2026-10-12/inputs/S1_edge_pair.json

P7. distributed入力。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86c69983544064d1ed7ee7910634c541c7b982b7/artifacts/n3_old_family_v1/2026-10-12/inputs/S1_distributed_three.json

P8. edge完全dual。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86c69983544064d1ed7ee7910634c541c7b982b7/artifacts/n3_old_family_v1/2026-10-12/diagnostic1/S1_edge_pair.json

P9. distributed完全dual。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86c69983544064d1ed7ee7910634c541c7b982b7/artifacts/n3_old_family_v1/2026-10-12/diagnostic1/S1_distributed_three.json

P10. 保存幅の数学契約と改善機構。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/docs/research/n3_saved_width_v1_mathematical_contract_20261012.md

P11. 旧族の数学・診断契約。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aa12ab0b777b28f1428d9bb8417af93c4376aae/docs/research/n3_old_family_v1_mathematical_contract_20261012.md

P12. edge保存幅summary。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/case_summaries/S1_edge_pair.json

P13. distributed保存幅summary。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/case_summaries/S1_distributed_three.json

P14. 固定8入力の元結果。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c/docs/research/n3_occupation_transfer_v1_results_20261012.md

P15. 保存証拠の事後幅結果。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/docs/research/n3_saved_width_v1_results_20261012.md

P16. 初期N3直接法・限定移送結果。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3e41b76fbb21c238c25572620e967d131571458/docs/research/n3_direction_generator_v1_results_20261011.md

既存GPTレビュー：n13_transfer_recovery_scientific_review_2026-10-12.md、n3_occupation_transfer_design_scientific_review_2026-10-12.md、n3_occupation_transfer_v1_results_scientific_review_2026-10-12.md、n3_saved_width_v1_scientific_review_2026-10-12.md。これらは経緯・判断履歴であり、一次データの代替にはしない。

### 外部一次資料（確認範囲は第7節）

L1. Xin Liang, Li Wang, Lei-Hong Zhang, Ren-Cang Li. On Generalizing Trace Minimization. arXiv:2104.00257v1, 2021.
https://arxiv.org/abs/2104.00257
https://arxiv.org/pdf/2104.00257

L2. Vincent Q. Vu, Juhee Cho, Jing Lei, Karl Rohe. Fantope Projection and Selection: A near-optimal convex relaxation of sparse PCA. NeurIPS 2013, Sec.2, Eq.(2).
https://papers.nips.cc/paper_files/paper/2013/file/81e5f81db77c596492e6f1a5a792ed53-Paper.pdf

L3. Geneviève Dusson, Israel Michael Sigal, Benjamin Stamm. The Feshbach-Schur map and perturbation theory. arXiv:2105.02058v1, 2021, Theorems1.1–1.2.
https://arxiv.org/abs/2105.02058
https://arxiv.org/pdf/2105.02058

L4. Diana B. Chamaki, Stuart Hadfield, Katherine Klymko, Bryan O'Gorman, Norm M. Tubman. Self-consistent Quantum Iteratively Sparsified Hamiltonian method (SQuISH): A new algorithm for efficient Hamiltonian simulation and compression. arXiv:2211.16522, 2022. 本レビューの確認は公開abstract中心。
https://arxiv.org/abs/2211.16522

L5. Jonas Greiner, Jürgen Gauss, Janus J. Eriksen. Error Control and Automatic Detection of Reference Active Spaces in Many-Body Expanded Full Configuration Interaction. arXiv:2406.11343v1, 2024.
https://arxiv.org/html/2406.11343v1

## 付録A. 保存値の独立算術照合

以下のコードはGitHub connectorで取得したexact rationalの転記に基づく。入力のcanonical instance SHA、保存Wの全entry、保存端点traceを照合する。リポジトリscience moduleをimportせず、新しい証拠や数値固有値を生成しない。

保存OCCの全Z・PSD検証は本コードに含まれず、公開されたsource-bound監査を根拠とする。コード実行結果は28条件PASSである。コードと実出力をそのまま以下に収録する。

### check_saved_dual_implications.py

- UTF-8 bytes: 7846
- SHA-256: `d7b272cc36b32eb97a5770105ba9b390038bb6b64a93e81f7acf9d2fcecc180f`

```python
"""Read-only exact arithmetic on values printed by the GitHub connector.
No repository code imports, new candidate search, eigensolver, or batch execution.
The two W matrices are reconstructed in their ALREADY SAVED factored forms.
"""
from fractions import Fraction as F
from decimal import Decimal, localcontext
from pathlib import Path
import hashlib
import json

def enc(x):
    if isinstance(x, F): return [x.numerator, x.denominator]
    if isinstance(x, dict): return {k: enc(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [enc(v) for v in x]
    return x

def trprod(A,B):
    return sum((A[i][j]*B[j][i] for i in range(len(A)) for j in range(len(A))),F(0))
def outer(v): return [[x*y for y in v] for x in v]
def add(A,B): return [[x+B[i][j] for j,x in enumerate(row)] for i,row in enumerate(A)]
def scale(a,A): return [[a*x for x in row] for row in A]
def dec(x): return str(Decimal(x.numerator)/Decimal(x.denominator))
def canon(x): return json.dumps(enc(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
checks=[]
def check(name,cond):
    if not cond: raise AssertionError(name)
    checks.append(name)

U=F(-1951,1064)
c=F(-233352716459809,183068686024704)
g=c-U; nu=4; m=9; beta2=F(1,576)
diag=[F(-769,1332),F(-1607,2664),F(-69,148),F(-1163,2664),F(-325,1332),F(-47,296),F(119,1332),F(613,2664),F(79,148)]
row0=[diag[0],F(-5,76),F(1,16),F(-5,72),F(5,76),F(-1,16),F(5,72),F(-5,76),F(1,16)]
h=[[F(0) for j in range(m)] for i in range(m)]
for i in range(m): h[i][i]=diag[i]; h[0][i]=row0[i]; h[i][0]=row0[i]
I=[[F(i==j) for j in range(m)] for i in range(m)]
R=add(h,scale(-U/nu,I))
t=F(237159687664232761596842197143239424527144293709308117720769424915284840810354147627875072523358074497811006849252782017282311836375,323614294843401893550331839528627842504099180360415783030339531681817550103280657290252696079550288653741950481397115801500425125888)
inputs=[
 dict(id='S1_edge_pair',v=[F(1,40)]+[F(0)]*7+[F(1,30)],
      delta=F(74086643592581502428451423294952443073313,30141531497122887216658795977091164634349568),
      instance_sha='c692d497e58f6a4939d3a85936767e4782cced42565b8bd50dc0bac471e7a926',
      endpoints=[F(-25438887306611909871668477235771857223236961534803722741306652604174635531091062857639,90851192099205107883201462535919860439682746210034384708236202981073196243367842178662400),F(-2030312654762469762564525976673713436947369774607405554951537059085338547969959783926413,30283730699735035961067154178639953479894248736678128236078734327024398747789280726220800)],
      kappa=F(1042523116060419733,718774500528331161600),s=F(-1110377215,274877906944)),
 dict(id='S1_distributed_three',v=[F(0)]*3+[F(1,36)]+[F(0)]*2+[F(1,36),F(0),F(1,72)],
      delta=F(46391094852685011415862451268664783392177,15070765748561443608329397988545582317174784),
      instance_sha='09683cee9fcaa1a5d2d3be6119c761719141f7cc61a1d8297a39830a4d1973a7',
      endpoints=[F(-5830329039487146317274545261994554776801922131813647452559968119888561455647298314096251795234261910573296127064411767688903906007,186401833829799490684991139568489637282361127887599491025475570248726908859489658599185552941820966264555363477284738701664244872511488),F(-5830329039487146317274545261994554776801922131813647452559968119888561455647298314096251795234261910573296127064411767688903906007,1139856530892143085539146994508716933630863249890113289318419737422045280739126758580139340886739857483588302908945323429213327327232)],
      kappa=F(243823299571786253,766692800563553239040),s=F(-635862191,274877906944))
]
base_lo=F(1454716336665867269946106497416100894011,470961429642545112760293687142049447411712)
base_hi=F(93101845546615505276550815834630457260993,30141531497122887216658795977091164634349568)
check('g positive',g>0)
check('NORM root bracket polynomial',base_lo*(g+base_lo)<=beta2<=base_hi*(g+base_hi))
check('NORM root bracket 128-bit precision',base_hi-base_lo<=F(1,2**129))
rows=[]
for record in inputs:
    name=record['id']; v=record['v']; delta=record['delta']; V=outer(v)
    instance=dict(id=name,U=U,c=c,delta=delta,h_A=h,nu=nu,v=v)
    check(name+': exact instance hash',hashlib.sha256(canon(instance)).hexdigest()==record['instance_sha'])
    check(name+': beta squared',sum((x*x for x in v),F(0))==beta2)
    if name=='S1_edge_pair':
        e=[F(1)]+[F(0)]*8; W=outer(e)
        check(name+': recorded W form PSD',W[0][0]==1 and sum(sum(row) for row in W)==1)
    else:
        e=[F(0),F(1)]+[F(0)]*7
        check(name+': mixture positive',0<t<1)
        W=add(scale(t,outer(e)),scale((1-t)/beta2,V))
        saved_W33=F(86454607179169131953489642385388417976954886651107665309570106766532709292926509662377623556192214155930943632144333784218113289513,728132163397654260488246638939412645634223155810935511818263946284089487732381478903068566178988149470919388583143510553375956533248)
        # Complete nonzero pattern of the saved W: t at (1,1),
        # W33=W36=W63=W66, W38=W83=W68=W86=W33/2, W88=W33/4.
        literal=[[F(0) for _ in range(m)] for _ in range(m)]; literal[1][1]=t
        for i in (3,6):
            for j in (3,6): literal[i][j]=saved_W33
            literal[i][8]=literal[8][i]=saved_W33/2
        literal[8][8]=saved_W33/4
        check(name+': all saved W entries match factorization',W==literal)
    check(name+': W trace one',sum(W[i][i] for i in range(m))==1)
    D=g+delta; M0=add(scale(delta*D/nu,I),scale(-1,V)); MD=add(M0,scale(D,R))
    endpoint=[trprod(W,M0),trprod(W,MD)]
    check(name+': recorded endpoint fractions',endpoint==record['endpoints'])
    check(name+': endpoints strict negative',all(x<0 for x in endpoint))
    check(name+': NORM false at target',delta*D<beta2)
    k=record['kappa']; s=record['s'];q=k*U-s
    check(name+': OCC scalar box',0<=k<=D)
    check(name+': OCC scalar sufficient margin',delta*D>=q)
    check(name+': saved OCC target narrower than NORM',delta<base_lo)
    r=trprod(W,R); b=trprod(W,V)
    check(name+': extrapolation slope nonpositive',r<0)
    check(name+': extrapolation intercept negative at NORM threshold',beta2/nu-b<0)
    # New analytic implication: for every 0<d<=dN, k>=0,
    # tr(W M_d(k))=k*r-b+d(g+d)/nu<=-b+beta2/nu<0.
    # NORM is attained at dN. The best width of NORM OR all OLD is dN.
    row=dict(id=name,instance_sha=record['instance_sha'],r=r,b=b,
             beta_squared=beta2,old_exclusion_margin_at_norm=beta2/nu-b,
             saved_delta=delta,recorded_endpoints=endpoint,
             q=q,forced_energy_lower=(beta2-q)/k,
             improvement_lower=1-delta/base_lo,improvement_upper=1-delta/base_hi,
             saved_W_t=t if name.endswith('three') else None,
             proof_scope='same stored U,c,h_A,v,nu; NORM OR FIVE OR complete unmodified OLD kappa/mu family')
    with localcontext() as ctx:
        ctx.prec=45
        row['display']={key:dec(row[key]) for key in ('r','b','old_exclusion_margin_at_norm','saved_delta','q','forced_energy_lower')}
        row['display']['improvement_percent_lower']=dec(100*row['improvement_lower'])
        row['display']['improvement_percent_upper']=dec(100*row['improvement_upper'])
    rows.append(row)
output=dict(status='EXACT_SAVED_DUAL_IMPLICATIONS_PASS',checks=len(checks),check_names=checks,rows=rows,
            new_eigensolvers=0,new_optimizers=0,new_certificate_searches=0,
            original_files_modified=False,producer_imported=False,
            limitation='OCC spectral PSD validity inherited from source-bound audits; this script checks saved duals and scalar consequences, not every historical proof')
out=Path(__file__).with_name('review_proof_checks.json');out.write_text(json.dumps(enc(output),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=output['status'],checks=len(checks),rows=[dict(id=r['id'],**r['display']) for r in rows]),ensure_ascii=False,indent=2))
```

### review_proof_checks.json

- UTF-8 bytes: 7452
- SHA-256: `f7408d90f05e12a81555c2086e453fc01509716e190632882bee192dab2d384d`

```json
{
  "status": "EXACT_SAVED_DUAL_IMPLICATIONS_PASS",
  "checks": 28,
  "check_names": [
    "g positive",
    "NORM root bracket polynomial",
    "NORM root bracket 128-bit precision",
    "S1_edge_pair: exact instance hash",
    "S1_edge_pair: beta squared",
    "S1_edge_pair: recorded W form PSD",
    "S1_edge_pair: W trace one",
    "S1_edge_pair: recorded endpoint fractions",
    "S1_edge_pair: endpoints strict negative",
    "S1_edge_pair: NORM false at target",
    "S1_edge_pair: OCC scalar box",
    "S1_edge_pair: OCC scalar sufficient margin",
    "S1_edge_pair: saved OCC target narrower than NORM",
    "S1_edge_pair: extrapolation slope nonpositive",
    "S1_edge_pair: extrapolation intercept negative at NORM threshold",
    "S1_distributed_three: exact instance hash",
    "S1_distributed_three: beta squared",
    "S1_distributed_three: mixture positive",
    "S1_distributed_three: all saved W entries match factorization",
    "S1_distributed_three: W trace one",
    "S1_distributed_three: recorded endpoint fractions",
    "S1_distributed_three: endpoints strict negative",
    "S1_distributed_three: NORM false at target",
    "S1_distributed_three: OCC scalar box",
    "S1_distributed_three: OCC scalar sufficient margin",
    "S1_distributed_three: saved OCC target narrower than NORM",
    "S1_distributed_three: extrapolation slope nonpositive",
    "S1_distributed_three: extrapolation intercept negative at NORM threshold"
  ],
  "rows": [
    {
      "id": "S1_edge_pair",
      "instance_sha": "c692d497e58f6a4939d3a85936767e4782cced42565b8bd50dc0bac471e7a926",
      "r": [
        -168533,
        1417248
      ],
      "b": [
        1,
        1600
      ],
      "beta_squared": [
        1,
        576
      ],
      "old_exclusion_margin_at_norm": [
        -11,
        57600
      ],
      "saved_delta": [
        74086643592581502428451423294952443073313,
        30141531497122887216658795977091164634349568
      ],
      "recorded_endpoints": [
        [
          -25438887306611909871668477235771857223236961534803722741306652604174635531091062857639,
          90851192099205107883201462535919860439682746210034384708236202981073196243367842178662400
        ],
        [
          -2030312654762469762564525976673713436947369774607405554951537059085338547969959783926413,
          30283730699735035961067154178639953479894248736678128236078734327024398747789280726220800
        ]
      ],
      "q": [
        1055372336755393164917,
        764776068562144355942400
      ],
      "forced_energy_lower": [
        817091680161655692449,
        3327733786464859787736
      ],
      "improvement_lower": [
        19015201954034002848099392539678014143391,
        93101845546615505276550815834630457216704
      ],
      "improvement_upper": [
        19015201954034002848099392539678014187680,
        93101845546615505276550815834630457260993
      ],
      "saved_W_t": null,
      "proof_scope": "same stored U,c,h_A,v,nu; NORM OR FIVE OR complete unmodified OLD kappa/mu family",
      "display": {
        "r": "-0.118915673191988981462665673191988981462665673",
        "b": "0.000625",
        "old_exclusion_margin_at_norm": "-0.000190972222222222222222222222222222222222222222",
        "saved_delta": "0.00245795883330126497416592532082392995321983255",
        "q": "0.00137997562965012609857336952555702610268336235",
        "forced_energy_lower": "0.245539977832684128799777828911642150814817344",
        "improvement_percent_lower": "20.4240870225426535550454705615653150869077177",
        "improvement_percent_upper": "20.4240870225426535550454705615653151247623659"
      }
    },
    {
      "id": "S1_distributed_three",
      "instance_sha": "09683cee9fcaa1a5d2d3be6119c761719141f7cc61a1d8297a39830a4d1973a7",
      "r": [
        -70178982558650879604384560335275140504290380756194567233657501676060670736981510677798548630517470401417288174599166895583753438264191,
        7758976333165403799762756184538381151878281948321328813935420611603257581276257039191098641203297720762117004741977248456774192818290688
      ],
      "b": [
        86454607179169131953489642385388417976954886651107665309570106766532709292926509662377623556192214155930943632144333784218113289513,
        186401833829799490684991139568489637282361127887599491025475570248726908859489658599185552941820966264555363477284738701664244872511488
      ],
      "beta_squared": [
        1,
        576
      ],
      "old_exclusion_margin_at_norm": [
        -5551033468318658565906682503231457350930091561003719551985223846078321767106345339814449536304641992495456011795054833843007008041,
        186401833829799490684991139568489637282361127887599491025475570248726908859489658599185552941820966264555363477284738701664244872511488
      ],
      "saved_delta": [
        46391094852685011415862451268664783392177,
        15070765748561443608329397988545582317174784
      ],
      "recorded_endpoints": [
        [
          -5830329039487146317274545261994554776801922131813647452559968119888561455647298314096251795234261910573296127064411767688903906007,
          186401833829799490684991139568489637282361127887599491025475570248726908859489658599185552941820966264555363477284738701664244872511488
        ],
        [
          -5830329039487146317274545261994554776801922131813647452559968119888561455647298314096251795234261910573296127064411767688903906007,
          1139856530892143085539146994508716933630863249890113289318419737422045280739126758580139340886739857483588302908945323429213327327232
        ]
      ],
      "q": [
        1411362789291059904237,
        815761139799620646338560
      ],
      "forced_energy_lower": [
        44002705749533460907,
        2334851916699425158728
      ],
      "improvement_lower": [
        159827920622741222412956648650445216175,
        46550922773307752638275407917315228608352
      ],
      "improvement_upper": [
        319655841245482444825913297300890476639,
        93101845546615505276550815834630457260993
      ],
      "saved_W_t": [
        237159687664232761596842197143239424527144293709308117720769424915284840810354147627875072523358074497811006849252782017282311836375,
        323614294843401893550331839528627842504099180360415783030339531681817550103280657290252696079550288653741950481397115801500425125888
      ],
      "proof_scope": "same stored U,c,h_A,v,nu; NORM OR FIVE OR complete unmodified OLD kappa/mu family",
      "display": {
        "r": "-0.00904487648179488559163142074824191082905174116",
        "b": "0.000463807707268102522825753037174474502677405000",
        "old_exclusion_margin_at_norm": "-0.0000297799294903247450479752593966967248996272219",
        "saved_delta": "0.00307821749914155495875880301668702633053299322",
        "q": "0.00173011770288265971004556878133813178638795461",
        "forced_energy_lower": "0.0188460370590594955939696271443504157952566843",
        "improvement_percent_lower": "0.343339962133653716521647657377667453928775144",
        "improvement_percent_upper": "0.343339962133653716521647657377667501335932226"
      }
    }
  ],
  "new_eigensolvers": 0,
  "new_optimizers": 0,
  "new_certificate_searches": 0,
  "original_files_modified": false,
  "producer_imported": false,
  "limitation": "OCC spectral PSD validity inherited from source-bound audits; this script checks saved duals and scalar consequences, not every historical proof"
}
```
