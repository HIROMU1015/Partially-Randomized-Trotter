# N3保存証拠の保証幅：科学レビューと限定継続方針

- 作成日：2026-10-12 JST
- 対象：N3 saved-width v1 の固定8入力・有限保存証拠集合に対する事後評価
- 結果commit：`652667187ce83c7c36041820a0a752669134865f`
- 公開commit：`ce1e9512ac225a234a5ce0f8bce03950dae1f29c`
- 評価source：`3f295df0263059ce16d866f84c1132f8bace0670`
- 元generator source：`83f51cd8a4848572c66a5535c67d0d0a5f3868cd`
- レビュー状態：明示的な開始承認を受けて実施・完了
- 科学的判断：**N3の即時終了は勧めず、S1の既存条件族との分離／回収可能性を判別する限定継続を推奨。中心研究テーマの採択、実分子等への拡張は見送り。**
- 実行認可：**このレビューは、新たな診断・最適化・科学batchの実行承認ではない。既存Mandatory STOPは維持する。**

## 1. 結論と判断の変更点

前回は、元目標幅でNORM・強対照・OCCが全8入力を認証し、OCC固有の追加認証が0件だったため、新規入力の追加ではなく、保存証拠に幅の利益が残るかを調べる方針だった。今回、その利益がS1の二方向で確認された。

したがって、現時点で「幅の利益も残らなかったのでN3を整理して終了する」という分岐は選ばない。一方、20.42%という一つの数値だけで中心テーマ採択へ進むことも勧めない。次に判別すべき問題は、**保存候補ではなく旧条件族全体と比較したときにも差が残るか**である。

| 論点 | 結論 |
|---|---|
| 保存証拠の保証幅 | S1の二方向でOCCが厳密に狭いという報告を支持する |
| 元の事前登録結果 | 追加保証0件・2-base基準未達を変更しない |
| 「2勝6敗」の意味 | raw保存poolの比較。fallback込み算法が6例で悪化するという意味ではない |
| S1の改善機構 | 同じU・c・結合normでqが減少している。NORMとの差には説明可能な機構がある |
| 旧LMI条件族との分離 | 未確定。S1には保存済みOLD-DIRECTのPSD-valid証拠がない |
| 原generatorの目的 | 元δgoalでの認証。幅最小化用の公平な最適化実験ではない |
| 研究の次段 | S1二例に限定した旧条件族の到達可能性／分離診断を優先する |
| 新規性・量子資源優位 | 未確定。既存数学の利用と新しい構成上の貢献を分ける |

ここでの限定継続は、成功するまで入力を追加することではない。現に得られた二例の利益が、対照の保存coverageの限界に依存するかを解くためのものとする。

## 2. レビューの根拠と実施範囲

### 2.1 使用した固定資料

主な正本は次のとおり。

- [結果報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/docs/research/n3_saved_width_v1_results_20261012.md)
- [結果前比較仕様](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/docs/research/n3_saved_width_v1_comparison_specification_20261012.md)
- [数学契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/docs/research/n3_saved_width_v1_mathematical_contract_20261012.md)
- [Aggregate summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/aggregate_summary.json)
- [S1 edge_pairの保存スカラー・区間・証拠ID](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/case_summaries/S1_edge_pair.json)
- [S1 distributed_threeの保存スカラー・区間・証拠ID](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/case_summaries/S1_distributed_three.json)
- [公開・remote再取得監査報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce1e9512ac225a234a5ce0f8bce03950dae1f29c/docs/research/n3_saved_width_v1_evaluation_publication_20261012.md)
- [幅算術source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f295df0263059ce16d866f84c1132f8bace0670/src/trottertracks/representation_exploration/n3_saved_width_v1.py)
- [元generator source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/83f51cd8a4848572c66a5535c67d0d0a5f3868cd/src/trottertracks/representation_exploration/n3_occupation_transfer_v1.py)：特に`margins`、`old_candidate`、`search`、`dyadic_candidates`、`generate_method`、`operational`。

以上の本文はGitHub connectorで取得した。全文の再認証を目的として元巨大ledgerを再実行したわけではない。全候補のPSD再検査と有限集合の完全性は、リポジトリに保存されたCodexの監査を根拠とする。

### 2.2 今回GPT側で行った確認

1. 数学契約と幅算術sourceの整合性を検討した。
2. S1の保存スカラーを用いて、共通g、q=κU−s、公開幅区間の多項式端点条件、縮小率区間、strictな大小を28条件で照合した。これは付録の独立スカラー算術であり、元の幅評価runnerやPSD checkerを呼んでいない。
3. 元generatorの停止条件とμの構成を読み、事後幅比較の限界を具体化した。
4. 既存変分表示から、保存証拠が与える結合方向占有エネルギーの下界を解析的に導出した。
5. 次の旧条件族診断の数学的目的と、何をもって分離／回収／未判定とするかを整理した。

新Hamiltonian、新κ・μ・τ/Zの探索、固有値計算、E_forcedのスペクトル評価、ground-state計算、compile、量子sampling、元batch再実行は行っていない。

スカラーは取得したJSONから明示的に転記した。保存ファイル全bytesのGPT側一致や、全240候補の最小化の再計算を主張しない。プログラム実行環境からの補助的なraw GitHubダウンロードはDNS解決不能だったが、GitHub connectorで主要本文・スカラー・sourceは取得できた。これはremote未公開や資料欠損を示すものではない。

### 2.3 監査数の意味

報告された独立保存監査は全8件PASS、旧checker conditions合計5,284,307である。公開資料には82 blobsのremote再取得・identity一致も記録されている。これらは保存証拠と対応関係の確認であり、8件の独立科学実験、汎化性能の証拠、外部環境での科学再現、immutable CIを意味しない。GPT側の28スカラー条件と混同しない。

## 3. 実験対象と二つの評価を区別する

対象は実対称one-body＋非負unordered density、一外部modeをemptyへ射影する固定粒子数Hamiltonianである。

| Base | Graph | Full→active modes | ν | 元δgoal |
|---|---|---:|---:|---:|
| C1 | chain | 7→6 | 2 | 0.004 |
| R1 | ring | 8→7 | 3 | 0.006 |
| L1 | ladder | 9→8 | 3 | 0.003 |
| S1 | star | 10→9 | 4 | 0.005 |

各baseに結合normが同じ二方向がある。geometry・化学basis・DF rank/policy・L_D・PF/time-evolution windowは適用なし。数値をHartreeやchemical accuracyへ勝手に読み替えない。

保証対象は

\[
0\le a-E_0(H_\nu)\le\delta,
\qquad a=\lambda_{\min}(A).
\]

同じ射影P、同じHamiltonianのまま保証幅が狭くなったのであり、物理的な射影誤差自体が20.42%減ったわけではない。

### 3.1 元の採否実験

元δgoalのもとでは強いBASE-OR、NORM、OCCが8/8を認証した。OCCによる追加受理0件、2-base gate未達である。

### 3.2 今回の保存幅実験

同じ8入力をposthoc developmentとして、固定保存証拠が含意する幅の下端を比較した。新しい独立移送や、元gateの再試験ではない。

| Input | BASE saved | OCC raw all saved | 狭いpool | OCC幅縮小率（表示） |
|---|---:|---:|---|---:|
| C1_edge_pair | 0.00117914072202 | 0.00118821565104 | BASE | −0.769622% |
| C1_distributed_three | 0.00114127558648 | 0.00118821565104 | BASE | −4.112947% |
| R1_edge_pair | 0.00272056818527 | 0.00273324112293 | BASE | −0.465820% |
| R1_distributed_three | 0.00272056818527 | 0.00273317645277 | BASE | −0.463442% |
| L1_edge_pair | 0.00115517774857 | 0.00118334516364 | BASE | −2.438362% |
| L1_distributed_three | 0.00115517774857 | 0.00118323524342 | BASE | −2.428847% |
| S1_edge_pair | 0.00308882266170 | 0.00245795883330 | OCC | 20.424087% |
| S1_distributed_three | 0.00308882266170 | 0.00307821749914 | OCC | 0.343340% |

表は報告値の丸め表示。分類の根拠は有理区間のstrict inequalityである。

S1の絶対幅差はedge_pairで約0.000630863828397、distributed_threeで約0.0000106051625571。0.343%は小さいが、報告された区間比較では丸めノイズに相当する差ではない。一方、実用上重要な差かはtask依存であり、strictだから大きな資源価値があるとも言えない。

二件は同じS1 baseなので、二つの独立なbaseで再現したと扱わない。star graphだけの効果ともしない。m、ν、graph、係数構成等はbase間で同時に変わっている。

## 4. 比較の数学：何が厳密で、何が有限集合に限定されるか

共通の係数trial上界Uと外部下界c=c_starを保持し、g=c−U>0とする。固定κの証拠が

\[
K_\kappa=\kappa h_A-vv^T,
\qquad Z\succeq0,\quad K_\kappa-\tau I+Z\succeq0,
\]

を満たせば、

\[
s=\nu\tau-\operatorname{tr}Z\le\sigma_\nu(K_\kappa).
\]

元のSchur保証の十分条件は

\[
\delta>0,\quad\kappa\le g+\delta,\quad
\delta(g+\delta)\ge q:=\kappa U-s.
\]

したがって固定証拠wの下端は

\[
\delta_w=\max\left\{0,\kappa-g,
\frac{\sqrt{g^2+4\max(q,0)}-g}{2}\right\}.
\]

これは**この十分条件を満たす幅の下端**であり、真の誤差でも条件族全体の最小幅でもない。δ_w=0はδ>0契約でのinfimumとして扱う。

NORMはκ=0、s=−β²、β²=||v||²であり、

\[
\delta_N=\frac{\sqrt{g^2+4\beta^2}-g}{2}.
\]

FIVE/OLD-DIRECTの保存証拠はs=νμとして同じ式を使う。PSD部分はδと移送先cに依存しないため、元c0行でaffine条件に落ちた保存証拠をc_starで使うこと自体は正当である。元κを固定し、xからκを作り直さないことが重要であり、比較仕様はこれを明示している。

### 4.1 有限保存集合と条件族全体

保存集合S_B、S_Oと条件族F_B、F_Oを区別する。

\[
\delta_B^{saved}=\min_{w\in S_B}\delta_w,
\quad
\delta_O^{saved}=\min_{w\in S_O}\delta_w.
\]

保存集合は条件族全体の部分集合なので、

\[
\delta_B^{family}\le\delta_B^{saved},\qquad
\delta_O^{family}\le\delta_O^{saved}.
\]

よってδ_O^{saved}<δ_B^{saved}から、δ_O^{family}<δ_B^{family}は直接には導けない。現結果は前者を認証している。

### 4.2 運用unionなら「6敗」ではない

OCC条件族はNORMと旧証拠を埋め込める。保存証拠の運用にもBASE fallbackを認めれば、

\[
\delta_{union}^{saved}=\min(\delta_B^{saved},\delta_O^{saved}).
\]

現表では、unionはS1二件で改善し、残り六件はBASEと同じ幅になる。「raw OCCの保存poolが六件で広い」と「OCCを併用すると六件で保証が悪化する」は異なる。

ただし、保証幅の非劣性から取得時間・メモリ・検査費用の非劣性は導けない。fallbackのコピーを新規OCC利益へ数えない方針も維持する。

## 5. S1で何が改善したか

二方向の共通値は

\[
U=-\frac{1951}{1064},\quad
c=-\frac{233352716459809}{183068686024704},
\]

\[
g=\frac{13609964515120091}{24348135241285632}
\simeq0.558973587925638,\quad
\beta^2=\frac1{576}.
\]

| 保存勝者 | κ | q=κU−s | β²−q |
|---|---:|---:|---:|
| S1_edge_pair | 0.00145041750270 | 0.00137997562965 | 0.000356135481461 |
| S1_distributed_three | 0.000318019550193 | 0.00173011770288 | 0.00000599340822845 |

両者ともκ<gなのでbox制約は幅の下端を決めない。q>0のrootが幅を決め、同じgでrootはqの厳密単調増加関数である。したがってq<β²がNORMより小さい幅を説明する。

この帰属は下界の片側優遇ではない。幅比較では双方が同じ保存c_starを使い、BASE側も全三下界行の保存PSD-valid候補を使っている。OCCの勝者は元c0の不受理行に保存された証拠であり、強い下界で再利用した結果である。

### 5.1 保存証拠だけで得られる機構の解析的帰結

以下は本レビューの解析的な導出であり、新しいE_forced固有値計算の報告ではない。

u=v/βとし、Πu=uを満たすrank-ν射影の最小一体エネルギーを

\[
E_{forced}(u)=\min_{\Pi u=u,\ \operatorname{rank}\Pi=\nu}
\operatorname{tr}(\Pi h_A)
=u^Th_Au+\sigma_{\nu-1}(h_A|_{u^\perp})
\]

とする。射影の一部集合で最小化するため、

\[
\sigma_\nu(K_\kappa)
\le \min_{\Pi u=u}\operatorname{tr}(\Pi K_\kappa)
=\kappa E_{forced}(u)-\beta^2.
\]

s≤σν(Kκ)と合わせ、κ>0なら

\[
\boxed{E_{forced}(u)-U\ge\frac{\beta^2-q}{\kappa}.}
\]

S1の保存スカラーから右辺を有理数で評価すると、

\[
E_{forced}(u_{edge})-U\ge
\frac{817091680161655692449}{3327733786464859787736}
\simeq0.245539977832684,
\]

\[
E_{forced}(u_{distributed})-U\ge
\frac{44002705749533460907}{2334851916699425158728}
\simeq0.0188460370590595.
\]

これにより、少なくとも保存証拠が与える範囲では、結合方向を占有する一体エネルギーがUより高いという機構が確認できる。新しいスペクトル計算でE_forcedそのものを求める必要はなく、この正値だけを再確認する診断を次の必須作業にしない。

ただし、これらはE_forcedの値そのものでも、真の相互作用込み基底エネルギーとの差でもない。また、二つの下界値の差だけでE_forcedの真の大小順序まで決められない。NORMに対する機構を示すものであり、旧LMI族を含むBASE全体の分離証拠でもない。

### 5.2 縮約の採否を変え得る区間

保存OCC上端と保存BASE下端の間には、OCCだけで認証できる正の目標幅がある。説明例として、

- edge_pair：δ_req=1/400=0.0025
- distributed_three：δ_req=77/25000=0.00308

はいずれも保存OCC幅より大きく、保存BASE幅より小さい。保存スカラーのbox・二次条件でも確認した。

これは、幅差に「同じ外部modeの削除を、より厳しい許容幅で認証する」という意味があることを示す。ただし、結果を見て選んだ説明例であり、新たな事前登録成功例、独立移送、元δgoalの変更として数えない。既存条件族全体もその目標幅で認証できないと示したわけではない。

## 6. 元sourceから分かる重要な比較限界

### 6.1 OCCは元goalで最初に受理した候補で停止する

`dyadic_candidates`は0と1を含む候補を昇順で返す。`generate_method`のOCC分岐は、候補の認証に成功すると`break`する。

元goalでκ=0が通れば、それ以降の正κのPSD証拠を保存せずに終了し得る。報告ではC1/R1/L1のraw OCCの保存最小がκ=0の有限精度spectral証拠であり、その下界はNORMの厳密identityより弱かった。

したがって、これら六例の幅が広いことは、OCC条件が粒子数を保持したために本質的に劣化したことを意味しない。元の認証taskに十分だった時点で保存が終わったこと、有限精度の下界が持つslackの影響で説明できる。これは原taskでは意図された停止であり、単純なバグと呼ばない。

S1は元c0で不受理だったため、その行に正κの多数の証拠が残った。保存数の差は、幅比較用に等しい探索機会を与えた実験ではないことを示す。

### 6.2 旧直接法のμは元goalのaffine閾値に合わせて作られている

元sourceでは各原下界行jで

\[
D_j=c_j-U+\delta_{goal},\qquad
\mu_0=\frac{\kappa U-\delta_{goal}D_j}{\nu}
\]

として旧PSD条件を検査している。ゆえに保存証拠のs=νμ0とqは

\[
\boxed{q_{OLD,saved}=\kappa U-\nu\mu_0
=\delta_{goal}(c_j-U+\delta_{goal})}
\]

となる。各行でqはκに依存しない。特に同じc_jのまま幅を読み出せば、受理された旧直接証拠は元δgoalの閾値に貼り付く。弱いc_jの証拠を強いc_starに移すことで幅が狭くなることはあるが、保存μはそのκで取り得る最大値を記録しているわけではない。

PSDに余裕があるなら、未保存のより大きいμでより狭い幅を認証できる可能性がある。現評価は新μ生成を禁止しているため、それを調べていない。これは評価仕様違反ではなく、**原証拠の保存方法が幅最適化比較に適していない**という限界である。

S1ではさらに、OLD-DIRECTのPSD-valid保存証拠は0件である。BASE最小を実際に与えているのはNORM。有限探索で保存されなかったことから、旧条件族全体の不可行性は導けない。

### 6.3 現結果を無効にはしない

この指摘は20.424%や0.343%の保存幅差を取り消すものではない。固定された保存集合の最小を比較するという実験目的には整合している。差の主張を、保存集合から条件族・幅最適化済み算法へ無断で拡張しないための指摘である。

## 7. 次に最も情報価値がある検証

### 7.1 推奨：S1二例の旧条件族到達可能性／分離診断

新しい入力を増やす前に、現S1二例で保存OCCが認証した幅を、旧LMI族も別κ・μで認証できるかを調べる。

固定する科学的内容は、元の二つのHamiltonian、各v、ν=4、保存U、保存c_star、保存OCC証拠と正の有理上端δ_O^hiである。新しいU/c、OCCの再最適化、Hamiltonian生成、真のground/gap、実分子、他base探索はこの診断に混ぜない。

比較点の一案は、各caseの**既に保存された正のOCC保証上端δ_O^hiそのもの**とすること。丸め表示を代入しない。これは新しい固定診断であり、既存の消費済みone-shotの再開ではない。

### 7.2 一変数の旧LMI族として書ける

δ=δ_O^hi、D=g+δとし、旧条件

\[
\kappa h_A-\mu I-vv^T\succeq0,
\quad\kappa U-\nu\mu\le\delta D,
\quad0\le\kappa\le D
\]

を考える。許される最小μ0=(κU−δD)/νを使うと、旧条件族の可行性は

\[
\boxed{M_\delta(\kappa)
=\kappa\left(h_A-\frac{U}{\nu}I\right)
-vv^T+\frac{\delta D}{\nu}I\succeq0,
\quad\kappa\in[0,D]}
\]

に等価である。

あるμ≥μ0でPSDなら、μをμ0まで下げると正の単位行列を加えるのでPSDを保持する。逆はμ0を採ればよい。したがって、この一変数族を調べれば、保存されなかった別μの可能性も含められる。

数値探索は候補生成に使えても、科学的分類は次の証拠に基づく。

- **回収証拠**：有理κとμ0、元係数から構成したMδ(κ)の厳密PSD。
- **分離証拠**：有理W≽0、W≠0で、区間両端のtraceが厳密負。

\[
\operatorname{tr}(WM_\delta(0))<0,
\qquad
\operatorname{tr}(WM_\delta(D))<0.
\]

traceがκのaffine関数なので、両端が負なら全区間で負。Mδ(κ)がPSDならPSDなWとのtraceは非負でなければならず、矛盾する。

NORMの不成立は別にδD<β²で確認する。元FIVEの有効証拠は同じκ・μの旧条件に含まれることをsource・数学契約で対応させる。これらが揃って初めて、指定したNORM/FIVE/OLDの条件族を含む対照との分離とする。

### 7.3 結果の解釈

| 結果 | 言えること | 言えないこと |
|---|---|---|
| 旧法の厳密回収証拠が得られる | 旧族も保存OCC幅に到達する。現保存差だけではその幅での族分離を支持しない | OCC族が他の幅・入力でも価値を持たない |
| 厳密separatorとNORM不成立が得られる | 同じU/c/δのもとで、指定旧族全体に対する分離が成立する | 最適旧法に対して20.42%という量的差そのもの、全既知算法への優位、新入力への移送 |
| どちらも得られない | 固定診断では未判定 | 旧族不可行、OCC無価値、探索予算の無条件増加 |

δ_O^hiで旧法が回収した場合、より小さいOCC族の未探索幅まで否定するわけではない。ただし現二例を根拠とした族分離の主張は支持されなくなる。その後の追加最適化を自動で続けない。

separatorがδ_O^hiで得られても、旧族の最小幅が保存NORM幅と等しいとは限らない。したがって20.424%をそのまま「最適旧族に対する改善率」としてはならない。量的な最適幅差を主張するには別の上下界が必要である。

### 7.4 なぜこの順序か

現時点で新baseを追加して同じ保存差が増えても、旧条件族のcoverageという疑問は残る。E_forced>Uの正値だけを新しい固有値計算で確認する必要もない。本レビューの保存証拠から既にその正値の下界が得られる。

また、まず量子compileやcoldベンチマークへ進んでも、比較対象の算法的差が未確定のまま費用だけを詳細化することになる。最初に旧族との回収／分離を判別する方が、研究の進路に直接関係する。

## 8. 費用対効果と研究task

### 8.1 現在の費用データで言える範囲

元producerのS1経路再構成値はBASE-OR約0.0968〜0.0994秒、OCC operational約0.3908〜0.3924秒だった。しかしこれは元δgoal=0.005の受理経路である。今回の狭い幅をcoldに生成する費用ではなく、0.0030888対0.0024580の比較費用として直接移植できない。

今回の約6.15秒も、保存集合の抽出・再検証・幅算術の費用であり、OCCだけのonline証拠生成時間ではない。生成、検証、公開監査、文献調査、量子実行の費用は区別する。

### 8.2 幅が小さいことの実用的意味

独立に定められた投影誤差予算ε_projが、旧法の保証幅とOCCの保証幅の間にあるとき、OCCの証拠により同じmode削減を認められる。この段階の最小の実用taskは「与えられたmodeを削除してよいか」であり、一般のmode選択最適化とは異なる。

今回の20.42%は、より厳しい投影予算に対応できる可能性として意味がある。ただし、元目標幅ではどちらも同じ10→9 modeの削減を認めている。今回だけで追加のmodeやqubitが減ったわけではない。

誤差予算を他の量子アルゴリズム段階へ再配分できる可能性もあるが、その全体誤差契約・状態準備・失敗確率・費用関数は未指定である。幅比からPF/QPE/RZ/CX/T費用比を作らない。複数mode削減へも単純に外挿しない。

### 8.3 将来の費用比較に必要なもの

族分離が残った場合に、同じ投影task・許容幅・合法入力情報で、下界とtrial取得、証拠生成、厳密検証、失敗率を含めて比較する。古典秒数と量子gate数を根拠なく一つの値へ足さない。理論・保証算法としての貢献と、量子資源削減の実証は別の到達点としてよい。

## 9. 先行研究と新規性の位置づけ

今回の外部確認は、次の一次資料の書誌・公開abstractを中心とした限定確認である。HTML全文は取得できなかったため、本文定理までの網羅的な新規性監査を完了したとはしない。

1. Xin Liang, Li Wang, Lei-Hong Zhang, Ren-Cang Li, *On Generalizing Trace Minimization*, arXiv:2104.00257v1（2021）、Linear Algebra and its Applications 656, 483–509（2023）。[一次資料](https://arxiv.org/abs/2104.00257)
   - 公開abstractはKy Fan trace最小化原理の一般化を扱うと明示する。
   - 最小固有値和の変分表示を使っただけでN3の新規性は成立しない。
2. Geneviève Dusson, Israel Sigal, Benjamin Stamm, *The Feshbach-Schur map and perturbation theory*, arXiv:2105.02058v1（2021）。[一次資料](https://arxiv.org/abs/2105.02058)
   - 公開abstractはFeshbach–Schur法による明示的な固有値・固有関数の厳密評価を扱う。
   - Schur補元を使うこと自体を新発見と扱わず、対象射影と追加仮定の差を調べる必要がある。
3. Diana B. Chamaki et al., *Self-consistent Quantum Iteratively Sparsified Hamiltonian method (SQuISH): A new algorithm for efficient Hamiltonian simulation and compression*, arXiv:2211.16522v1（2022）。[一次資料](https://arxiv.org/abs/2211.16522)
   - Hamiltonianと近似ground-state wavefunctionを使った反復的な疎化・資源削減を扱う。
   - N3は係数＋明示trial上界に基づく指定mode削除の保証であり、同じtaskの直接対照と自動的には扱えない。N3もUを使うので「状態情報ゼロ」とは言わない。

この限定調査から、N3と同一の定理・算法が未発表と結論しない。逆に、一般原理が既知というだけで、特殊classの非自明な構成や条件族分離が無価値とも言わない。

狙い得る研究の形は、単なる二例の幅比較ではなく、**占有情報が必要になる条件、既存条件との分離、保証生成と縮約決定の方法、取得費用と限界**を統合したものである。現在はそのうち保存証拠による限定的な幅差と機構の一部があるだけで、主要claimの完成ではない。

## 10. 代替方針の比較

| 方針 | 判断 | 理由 |
|---|---|---|
| 今すぐN3を中心テーマにする | 見送り | 1baseのposthoc知見で、旧族のcoverageと資源価値が未解決 |
| N3を直ちに全面終了する | 現時点では勧めない | 保存幅のstrict改善とNORMとの差を説明する機構がある |
| 新しい合成baseを大量に追加する | 見送り | 対照の族全体との比較という疑問を解決しない |
| 原8入力を幅最適化で全面再実行する | 直近作業にはしない | まずS1の固定幅に対する旧族可行性という小さな判別が可能 |
| 同じ保存幅評価を再び実行する | 不要 | 既に回答済みで、新しい科学的情報を増やさない |
| S1二例の旧族分離／回収を判別する | 推奨 | 現利益の帰属を直接変える情報が得られる |
| 実分子・多外部・一般二体へ拡張する | 見送り | 成立仮定の拡張と実用価値の課題が同時に増える |
| 証拠primitiveとして保存し、別テーマへ戻す | 診断後の有力選択肢 | 旧族に回収され、具体的な研究差分も残らない場合 |

## 11. Codexへの次作業：技術工程をまとめる

次に委ねる推奨単位は、**S1二例の旧条件族診断の準備と、承認後の限定実行・証拠整理**である。

### 固定する科学的要件

- S1二入力の元係数、保存U/c_star、元結果、保存OCC勝者を保持する。
- 比較目標を、各保存OCCの正の有理上端に固定する案を実行planに明記する。
- 旧法の任意κ・μに関する可行性を対象とし、有限保存集合だけを再比較しない。
- primal、dual、未判定を分ける。数値optimizerの成否を数学的不可行性としない。
- 同情報のNORM不成立とFIVE⊂旧族の対応を確認する。
- 旧結果の再分類・元2-base gate変更・新入力・新OCC探索・true ground/gap・量子compile・他研究の再開をしない。

### Codexの裁量にする技術事項

数値候補生成器、簡単な解析的separatorの優先、独立有理checker、有限予算の具体値、合理的なresource制約、synthetic/adversarialテスト、出力分割と索引、source/hash/preflight、費用記録はCodex側で設計・確認する。数学的対象を変えずに解決できる技術的な問題ごとに、GPT詳細レビューを要求しない。

source・plan・独立テスト・証拠取得性をまとめて準備し、新しい実行には別の明示認可を必要とする。消費済みのsaved-width one-shotを流用しない。成功・回収・未判定・技術停止のどの場合も旧証拠を保持し、自動再試行をしない。

### 次にGPTへ戻す条件

- 厳密な旧族回収証拠または分離証拠が得られた。
- 限定診断が未判定で終了し、追加計算の情報価値の判断が必要になった。
- 既存保証の前提・意味論・比較の公平性に実質的な問題が見つかった。
- 新しい科学契約や主要な対象拡張が必要になった。

通常の実装完了、source version変更、focused testsのPASS、公開receiptだけを新たな詳細科学レビューの理由にしない。

## 12. 主張できる範囲と研究の着地点

### 現在支持される主張の例

> 実対称一体項と非負density相互作用からなる固定粒子数・一外部modeの既存8入力に対し、同じ保存trial上界と外部下界を使用して有限保存証拠を事後評価した。S1の二つの結合方向では、raw OCC保存証拠がNORM/FIVE/OLD-DIRECTの有限保存集合より厳密に小さい保証幅を与え、幅差は約20.424%および0.343%だった。元目標幅での追加認証0件という登録結果は変わらない。

### 現時点で支持されない主張

- 最適化した既存条件族に対して20.424%改善した。
- star型の一般Hamiltonianで同じ利益がある。
- 二つの独立base・未使用データで移送を確認した。
- 真の投影誤差や量子gate費用を20.424%削減した。
- 認証するだけで追加mode削減や逐次縮約が保証される。
- 全既知のactive-space法・疎化法より有利である。
- 本算法の新規性や独立論文としての主要貢献が確定した。

### 最終方針

**N3は即時終了ではなく限定継続。ただし中心テーマは未採択。次の焦点は、現S1の幅改善が旧条件族全体を相手にしても残るかである。**

厳密分離が残れば、機構と使用taskを踏まえた独立検証・文献差分・費用評価へ進む価値が増す。旧族が同幅を回収し、他の非自明な構成差も残らなければ、現N3を保証primitiveとして保存し、中心テーマ探索を別方向へ戻すことを検討する。有限診断の未判定は数学的不可能性とせず、追加計算を自動化しない。

N1の現候補探索停止、N2保留、個別研究のMandatory STOP、旧source/result保全は維持する。

---

## 付録A. 独立スカラー照合コード

以下は本レビュー中に実行したコード。入力はGitHubの二つのS1 summaryから転記した既存の有理数である。新しいmatrix証拠や候補を生成せず、PSD有効性は公開済み監査を根拠とする。全保存poolの最小値そのものを再計算するコードではない。

```python
"""Read-only arithmetic on S1 scalar values transcribed from fixed GitHub summaries.
Not a producer run, a new certificate search, a PSD replay, or a family-separation test.
"""
from fractions import Fraction as F
from decimal import Decimal, localcontext
from pathlib import Path
import json

REPO='HIROMU1015/Partially-Randomized-Trotter'
COMMIT='652667187ce83c7c36041820a0a752669134865f'
U=F(-1951,1064)
c=F(-233352716459809,183068686024704)
g=F(13609964515120091,24348135241285632)
beta2=F(1,576)
B_lo=F(1454716336665867269946106497416100894011,470961429642545112760293687142049447411712)
B_hi=F(93101845546615505276550815834630457260993,30141531497122887216658795977091164634349568)
DATA={
'S1_edge_pair':dict(
 k=F(1042523116060419733,718774500528331161600),
 s=F(-1110377215,274877906944),
 q=F(1055372336755393164917,764776068562144355942400),
 lo=F(2315207612268171950889106977967263844657,941922859285090225520587374284098894823424),
 hi=F(74086643592581502428451423294952443073313,30141531497122887216658795977091164634349568),
 rel_lo=F(19015201954034002848099392539678014143391,93101845546615505276550815834630457216704),
 rel_hi=F(19015201954034002848099392539678014231969,93101845546615505276550815834630457260993),
 example=F(1,400)),
'S1_distributed_three':dict(
 k=F(243823299571786253,766692800563553239040),
 s=F(-635862191,274877906944),
 q=F(1411362789291059904237,815761139799620646338560),
 lo=F(92782189705370022831724902537329566740065,30141531497122887216658795977091164634349568),
 hi=F(46391094852685011415862451268664783392177,15070765748561443608329397988545582317174784),
 rel_lo=F(159827920622741222412956648650445216175,46550922773307752638275407917315228608352),
 rel_hi=F(319655841245482444825913297300890520928,93101845546615505276550815834630457260993),
 example=F(77,25000))}

def dec(x):
    with localcontext() as ctx:
        ctx.prec=35
        return str(Decimal(x.numerator)/Decimal(x.denominator))
def pair(x): return [x.numerator,x.denominator]
checks=0
def require(cond,msg):
    global checks
    checks+=1
    if not cond: raise ValueError(msg)

require(g==c-U and g>0,'common gap')
require(0<B_lo<=B_hi,'baseline interval')
require(B_lo*(g+B_lo)<=beta2<=B_hi*(g+B_hi),'baseline polynomial endpoints')
require(B_hi-B_lo<=F(1,2**129),'baseline width')
rows=[]
for name,d in DATA.items():
    k,s,q,lo,hi=[d[z] for z in ('k','s','q','lo','hi')]
    require(k>0 and q==k*U-s,'q identity')
    require(k<g,'box inactive')
    require(0<lo<=hi<B_lo,'strict saved-width ordering')
    require(lo*(g+lo)<=q<=hi*(g+hi),'OCC polynomial endpoints')
    require(hi-lo<=F(1,2**129),'OCC enclosure width')
    require(q>0 and q<beta2,'strict q improvement')
    require(d['rel_lo']==1-hi/B_lo,'relative reduction lower endpoint')
    require(d['rel_hi']==1-lo/B_hi,'relative reduction upper endpoint')
    req=d['example']
    require(hi<req<B_lo,'illustrative target in gap')
    require(k<=g+req and req*(g+req)>=q,'stored OCC scalar proof at target')
    require(req*(g+req)<beta2,'NORM fails at target')
    L=(beta2-q)/k
    require(L>0,'forced occupancy lower increment')
    rows.append(dict(
        case=name,
        source=f'https://github.com/{REPO}/blob/{COMMIT}/artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/case_summaries/{name}.json',
        baseline_width_interval=[dec(B_lo),dec(B_hi)],occ_width_interval=[dec(lo),dec(hi)],
        absolute_improvement_interval=[dec(B_lo-hi),dec(B_hi-lo)],
        relative_improvement_percent_interval=[dec(100*d['rel_lo']),dec(100*d['rel_hi'])],
        kappa=dec(k),q=dec(q),beta_squared=dec(beta2),
        q_decrease=dec(beta2-q),
        analytic_forced_energy_minus_U_lower_bound={'rational':pair(L),'decimal':dec(L)},
        illustrative_target=dec(req),
        qualification='Targets are posthoc explanatory consequences, not new preregistered results. Forced-energy bound follows analytically from valid saved spectral proof; no E_forced eigensolve.'))
output=dict(status='S1_PUBLISHED_SCALAR_IDENTITIES_AND_POLYNOMIAL_ENDPOINTS_PASS',
            checks=checks,common_U=dec(U),common_c=dec(c),common_g=dec(g),rows=rows,
            limitations=['Only two winning scalar witnesses and their baseline interval were checked.',
                         'No PSD matrices were replayed; PSD validity is supported by repository audit.',
                         'No minimum over all saved candidates was recomputed; summary minima are repository results.',
                         'No GitHub byte-identity claim for manually transcribed scalar data.',
                         'No new eigenvalue calculation, optimization, certificate search or registered batch.'])
path=Path(__file__).with_suffix('.json')
path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(output,ensure_ascii=False,indent=2))
```

## 付録B. スカラー照合の出力

以下のdecimalは表示用。厳密な区間の差が表示桁以下の場合、low/highが同じ文字列に丸まる。等しい有理数だったことを意味しない。判定はコード中のFraction比較による。

```json
{
  "status": "S1_PUBLISHED_SCALAR_IDENTITIES_AND_POLYNOMIAL_ENDPOINTS_PASS",
  "checks": 28,
  "common_U": "-1.8336466165413533834586466165413534",
  "common_c": "-1.2746730286157157292376991148825522",
  "common_g": "0.55897358792563765422094750165880115",
  "rows": [
    {
      "case": "S1_edge_pair",
      "source": "https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/case_summaries/S1_edge_pair.json",
      "baseline_width_interval": [
        "0.0030888226616986066697373317044505298",
        "0.0030888226616986066697373317044505298"
      ],
      "occ_width_interval": [
        "0.0024579588333012649741659253208239300",
        "0.0024579588333012649741659253208239300"
      ],
      "absolute_improvement_interval": [
        "0.00063086382839734169557140638362659987",
        "0.00063086382839734169557140638362659987"
      ],
      "relative_improvement_percent_interval": [
        "20.424087022542653555045470561565315",
        "20.424087022542653555045470561565315"
      ],
      "kappa": "0.0014504175026995517634442717166790969",
      "q": "0.0013799756296501260985733695255570261",
      "beta_squared": "0.0017361111111111111111111111111111111",
      "q_decrease": "0.00035613548146098501253774158555408501",
      "analytic_forced_energy_minus_U_lower_bound": {
        "rational": [
          817091680161655692449,
          3327733786464859787736
        ],
        "decimal": "0.24553997783268412879977782891164215"
      },
      "illustrative_target": "0.0025",
      "qualification": "Targets are posthoc explanatory consequences, not new preregistered results. Forced-energy bound follows analytically from valid saved spectral proof; no E_forced eigensolve."
    },
    {
      "case": "S1_distributed_three",
      "source": "https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/652667187ce83c7c36041820a0a752669134865f/artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/case_summaries/S1_distributed_three.json",
      "baseline_width_interval": [
        "0.0030888226616986066697373317044505298",
        "0.0030888226616986066697373317044505298"
      ],
      "occ_width_interval": [
        "0.0030782174991415549587588030166870263",
        "0.0030782174991415549587588030166870263"
      ],
      "absolute_improvement_interval": [
        "0.000010605162557051710978528687763503489",
        "0.000010605162557051710978528687763503492"
      ],
      "relative_improvement_percent_interval": [
        "0.34333996213365371652164765737766745",
        "0.34333996213365371652164765737766755"
      ],
      "kappa": "0.00031801955019346119516243153038304538",
      "q": "0.0017301177028826597100455687813381318",
      "beta_squared": "0.0017361111111111111111111111111111111",
      "q_decrease": "0.0000059934082284514010655423297729793247",
      "analytic_forced_energy_minus_U_lower_bound": {
        "rational": [
          44002705749533460907,
          2334851916699425158728
        ],
        "decimal": "0.018846037059059495593969627144350416"
      },
      "illustrative_target": "0.00308",
      "qualification": "Targets are posthoc explanatory consequences, not new preregistered results. Forced-energy bound follows analytically from valid saved spectral proof; no E_forced eigensolve."
    }
  ],
  "limitations": [
    "Only two winning scalar witnesses and their baseline interval were checked.",
    "No PSD matrices were replayed; PSD validity is supported by repository audit.",
    "No minimum over all saved candidates was recomputed; summary minima are repository results.",
    "No GitHub byte-identity claim for manually transcribed scalar data.",
    "No new eigenvalue calculation, optimization, certificate search or registered batch."
  ]
}
```
