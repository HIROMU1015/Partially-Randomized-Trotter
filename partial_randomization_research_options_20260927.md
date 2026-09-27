# 部分ランダム化を中心に据えた研究候補と事前検証計画

作成日：2026-09-27  
位置付け：複数候補の比較計画。2026-09-27に文献監査、PR-2/PR-3最小pilot、強制STOP、PR-2主研究契約、S1--S3結果前事前登録とdry-runまで完了した。PR-2は新手法ではなく、まず限定的な資源研究として進める。一般的新規性または最終資源優位性の確定は意味しない。  
過去の結果の参照：候補案の初版は`HIROMU1015/Partially-Randomized-Trotter`、`71169d817c165b76a9e25fc6f6a16ade28ffe069`を参照した。採択時には2026-09-27のworking treeにある研究概要、P-A--D、R3、FRの現行判断と再照合した。working treeはimmutableな外部証拠ではない。  
今回実施：一次文献の重点調査、PR-1共通評価record、PR-2/PR-3結果前事前登録と最小pilot、固定5項目による方針再選択、PR-2 S1--S3の実装監査・結果前事前登録・dry-run。後半では分子計算、signal評価、compile、samplingを追加実行していない。追加grid、H12、長RPE、最終総costも実行していない。

## 0. 今回の結論

**部分ランダム化を研究の中心に戻す。ただし「新しい一般定理」を必須条件にはせず、既存手法の有用な拡張、表現・実装と組み合わせた資源評価、適用条件の実証も独立した成果候補として扱う。**

主題候補は次の6件とする。番号は旧P-A/B/C/D、R3、FRとは別の新しい整理番号である。

| 新ID | 候補 | 成果の中心 | 最初の不確かさ |
|---|---|---|---|
| PR-1 | 表現・実装をそろえた部分ランダム化の資源比較 | どの条件で、なぜ、どれだけ有用かという判断材料 | 既存のPR/SPRINT/UWC評価が答えていない比較か |
| PR-2 | 圧縮した決定論Hamiltonian＋残差のランダム補完 | 圧縮精度と確率的補完の設計・損益分岐 | 圧縮残差が実装可能な表現でも安いか |
| PR-3 | tailに限定したRichardson外挿／qFLO拡張 | backboneを保ってtail精度を改善する実用的手法 | 外挿bias低減が統計増幅とbackbone再実行を上回るか |
| PR-4 | tailに限定したMLMC | 階層差分推定と部分ランダム分割の資源設計 | 量子測定・差分状態準備まで含めた費用で利益が残るか |
| PR-5 | 高精度random-tail方式の使い分け | qDRIFT/RTE/qSWIFT等の部分ランダム化内での比較 | 平均channelとcoherent信号を同じタスクで比較できるか |
| PR-6 | 安価な決定論部分を利用する相互作用描像との融合 | 大きい部分を正確・安価に扱う設計の実装評価 | tail変換費用がnorm上の利益を打ち消さないか |

候補整理時点では、**PR-1を資源評価としての軸候補、PR-3を小さく実装して試せる拡張候補、PR-2をSPRINT等に直接つながる表現・圧縮候補**として残した。これはpilot前の履歴であり、現行判断は0.1節のPR-2 resource-study contractを優先する。MLMCは有望だが、実装上の前提を確認する前に第一候補へ固定しない。PR-5/6は、対応する処理や表現に明確な利益の理由がある場合の次候補とする。

全6件の数値pilotを連続して行う計画ではない。文献対応を整理し、最初の比較を共用できる2案程度に絞る。PR-1はPR-2/3/5の評価基盤として兼用できるため、各々独立の巨大workflowにしない。

### 0.1 プロジェクト全体への採択状態

pilotのstatusは`PR23_MINIMAL_PILOTS_COMPLETE_MANDATORY_STOP_SELECT_PR2`である。6候補をすべて
実行せず、[事前登録](docs/research/pr2_pr3_minimal_pilot_preregistration.md)どおりPR-2/PR-3を各一条件だけ
実行した。PR-2は`GO_PR2`、PR-3は`STOP_PR3_VARIANCE_BACKBONE_DOMINATES`、5項目比較は8対6で
`SELECT_PR2_PRIMARY_CANDIDATE`となった。`STOP_AFTER_PR2_PR3_MINIMAL_PILOTS`が有効であり、
追加数値は許可されていない。詳細は[最小pilot検証](docs/pr2_pr3_minimal_pilot_validation.md)を参照する。

その後、[PR-2主研究契約](docs/research/pr2_primary_research_contract.md)と
[S1--S3結果前事前登録](docs/research/pr2_s1_s3_preregistration.md)を文書だけで固定した。pilotの
rank-$r$圧縮はrank-12 generation-prefixと一致するが、通常PRのweight-ranked prefixとの同一性は
S0 gateで確認する。一致なら統合、不一致なら両方をbaselineとして残し、順序差だけをmethod deltaとは
しない。core claimをcoherent-signal taskにおけるresidual-aware rank stoppingの資源crossoverへ限定し、
rank 6をdevelopment anchor、H4 1.30 Åを独立条件とした。現行statusは
`PR2_S1_S3_PREREG_V1_FROZEN_FOR_EXTERNAL_REVIEW_EXECUTION_NOT_AUTHORIZED`で、次に許されるのは外部レビュー
だけである。input freeze、identity gate、runner/test完了前にS1を開始しない。

PR-3の三gate監査は同日完了し、substatusを
`PR13_STAGE_A_AUDIT_PASSED_PILOT_PREREGISTRATION_NEXT`とした。通過scopeは固定した外側PFでの
低次数tail-only線形信号外挿と資源crossoverであり、qFLOの漸近深さ保証をComposite回路へ移送する
主張ではない。詳細は
[PR-1＋PR-3 Stage A選定契約](docs/research/pr1_pr3_stage_a_contract.md)を正本とする。

1. **PR-1を共通評価軸として採用する。** Pauli型とDF-native型の表現・sampling単位・制御化・
   count/depth・shot負担を、同じtaskと費用scopeで比較できる契約を先に作る。
2. **PR-3を第一拡張候補とする。** qFLO／Richardson外挿をpartial tailへ接続できるかを、
   coherent signalのestimand、統計増幅、backbone再実行費用の三点から監査する。
3. **PR-2を代替候補として残す。** PR-3でcoherent signalへの正当な接続または新しい比較差分を
   固定できない場合、圧縮残差のsampling表現と費用を先に調べる。
4. PR-4は測定可能な差分推定器と物理費用の監査まで、PR-5/6は明確なmethod deltaまたは
   primitive上の利益が見えるまで保留する。

上記はpilot前に採った順序の履歴である。実行後のPR-2契約では、SPRINT/GRADE等との差、中心主張、
rank-6 development anchor、H4 1.30 Å独立条件、scope progression、完成基準を固定した。追加rank、
geometry、PR-3条件、PR-4--6、H12、長RPE総costは引き続き開始しない。

### 0.2 既存の研究系列との関係

| 既存系列 | 新PR系列で再利用するもの | 維持する境界 |
|---|---|---|
| 総cost・WP00--WP11・M06-F/A0 | controlled回路scope、compiled-cost proxy、shot・状態準備感度、公平再最適化の規約 | 最終総costとPR優位性は未確定のまま。既存H4点順位を一般結論にしない |
| P-A/P-B/P-C/P-D | 強いbaseline、basis共有、geometry holdout、finite-tail比較、停止条件の経験 | 各停止済みclaimを再開・改名しない。新RQで必要な実装だけ再利用する |
| R3 | 一般split/error/cost最適化との重複監査 | 新しいselectorの方法論的新規性を主張しない |
| FR | 平均演算子・channel・coherent signal、正scalar、半径、controlの意味論 | `TECHNICAL_NOTE`分類とFR-R2停止を維持し、PR-3のためにFR計算を再開しない |
| VALIDATION_STATUS・manifest | 既存artifactの再利用可否とprovenance | 方針採択だけでは証拠statusを変更しない |

従って、新PR系列は過去の検証を捨てる別プロジェクトではない。過去に止めた主張と、再利用できる
実装・比較規約・証拠を分離し、その上に別のRQを置く。

---

## 1. 新規性の評価基準を改める

### 1.1 完全に新しい理論は必須ではない

次の成果は、新しい一般定理がなくても研究候補になり得る。

- 既存法を部分ランダム化へ接続し、実装手順と誤差・統計を整合させたうえで、意味のある資源改善や適用条件を示す。
- 同一精度・同一測定タスク・同一gate scopeで、それまで直接比較されていない手法や表現を比較し、選択が変わる理由を明らかにする。
- 既知の圧縮や誤差軽減を組み合わせ、従来の実装では見えない費用の競合を示す。
- 特定のモデル族・化学的タスクについて、従来の評価を改善し、設計判断に使える再現可能な結果を与える。

ただし「既存法を一度動かした」「分子を一つ置き換えた」「理想oracle費用では良かった」だけでは弱い。新しい説明、実装上の有用性、適切な対照との差、独立条件での検証のいずれかが必要である。論文採録・学位審査を保証するものではない。

### 1.2 既知の定理が使えることは、必ずしもNo-Goではない

既存のMLMCや外挿の定理が適用できるなら、正当性を新しく証明する負担を減らせる。そのうえで、決定論部分を残すと実装費用や最適分割がどう変わるかが未解決なら、拡張・資源評価として意味がある。

したがって、単なる `lambda -> lambda_R` の置換で解析が成立したことだけでは停止しない。**それ以外の実装・用途・資源上の新しい知見が残るか**を評価する。逆に、既知理論に含まれることを初の理論発見とは呼ばない。

### 1.3 「部分ランダム化が本質」の基準

記法を

\[
H=H_D(s)+H_R(s)
\]

とし、sを分割または圧縮設定とする。L_Dに限定しない。

部分ランダム化を中心にするとは、結果がD/Rの選び方に依存し、少なくとも次の競合を調べることをいう。

\[
\text{毎回実行する決定論部分の費用}
\quad\leftrightarrow\quad
\text{tailの誤差・統計負担・実装費用}.
\]

完全ランダムでも成立する手法を使うことは許容する。重要なのは、partial版を研究対象として、端点との違いと分割依存を調べることである。「H_Dがゼロだと手法そのものが数学的に定義不能」という強い条件は不要。

---

## 2. 文献調査の結果と、既知として扱う範囲

| 文献ID | 内容 | 今回に関係する点 | 確認範囲 |
|---|---|---|---|
| W1 Güntherら PR | 部分ランダム化とsingle-ancilla phase estimation、量子化学資源評価 | PR・固有値精度・shotと回路の交換関係は既知。一般のPR優位性を初めて調べる研究とはしない | 出版社abstract、arXiv・公開研究機関情報 |
| W2 Hagan–Wiebe Composite | deterministic TrotterとqDRIFTの合成、partitioning、channel error | ハイブリッド化自体は既知。既知理論を部分タスクへ流用できる | 出版社abstract・論文書誌 |
| W3 Pocrnicら | real/imaginary time、local composite、具体的gate比較 | 定数倍改善・モデル別比較も独立した研究の形の先例 | 著者所属機関の公開記録、arXiv abstract |
| W4 SPRINT/GRADE | 圧縮分解、近可積分PF、残差処理、randomization、resource estimate | **圧縮Hamiltonian＋残差をqDRIFT/RTEで処理する構図は既に提示されている** | PDF Fig.1、Eqs.(2)–(5)、Sec.IV、ページ画像 |
| W5 RC-DF | 正則化した圧縮DF、Hamiltonian近似と資源 | 圧縮率・残差の設計に使える。元の圧縮法を新規としない | Quantumの本文入口・abstract・summary |
| W6 UWC/STAR評価 | PR、orbital optimization、BLISS、早期FTQC資源 | PRのFT資源評価全般を新規とはしない。具体的表現・費用モデルの比較差を限定する | arXiv v2 introductionとabstract |
| W7 qFLO | qDRIFT観測量へRichardson外挿、浅い回路と統計の交換関係 | tail-only外挿の基礎。高精度状態を準備する方法ではない | PDF Theorem 1、Sec.3、関連研究、ページ画像 |
| W8 MLMC-qDRIFT | coupled hierarchy、variance減衰、測定方法 | partial partitionとの組合せをSec.5で将来方向として明記。量子shot noiseを含む特殊な差分測定が必要 | v2 Sec.2.2–2.3、3、4、5 |
| W9 qSWIFT | 高次randomized channelとobservable推定 | 高精度tail方式の第一比較候補 | PRX Quantum/著者機関abstract、arXiv |
| W10 PRHS | 相関sliceとquasi-probabilityによる高次randomized simulation | 将来比較候補。ただし本文未取得で詳細条件・費用は未監査 | arXiv索引の公式abstract。PDF/HTML直接取得に失敗 |
| W11 interaction-picture hybrid | 相互作用描像とrandomized/coherent手法の合成 | 絵を変えたpartial法自体は既知。分子の具体的primitive費用まで評価する余地を確認する | Quantum abstract・scope |

**重要な読み替え**：SPRINTが似た設計を提示していることは、PR-2を新アルゴリズムの発明として扱えない理由になる。一方、その設計を特定の推定タスクで実装・比較して、圧縮と補完の有用な使い分けを明らかにする研究まで自動的に禁止する理由にはならない。

MLMCの将来課題にpartialが書かれていることも、世界初性の証明ではない。接続の正当性、測定の費用、新しい結果を確認する必要がある。

---

## 3. PR-1：表現・実装依存性を明らかにする資源評価

### RQ

> Pauli項を選別する部分ランダム化と、DF fragment構造を使う部分ランダム化は、同じ推定精度・回路scopeで何が異なるか。決定論側のbasis共有・並列性と、tailのsampling負担を入れると、どこで部分ランダム化の利益が残り、どこで失われるか。

### 研究として残すもの

最良L_Dの一覧ではなく、**選ぶべき表現・sampling単位・実装方式の条件を説明するbenchmarkと資源図**を残す。新しい最適化アルゴリズムがなくてもよい。

候補となる比較軸は、Pauli/DF、primitive/fragment sampling、論理count/depth、state preparationの感度など。ただし全直積にせず、最初は一つの主差分を選ぶ。

推奨する最初の主差分は、**Pauli型PRとDF-native型PRを同じcoherent信号の誤差基準・制御化・回路指標で比較すること**。既存論文が同一の比較を十分に行っている場合は、別の費用項や未解決用途を特定する。単なる同じ結果の再現だけを主論文にしない。

### partialである理由

決定論に置くと一度のbasis変換内で多数の操作を共有できるが、ランダム化すると実行回数が減る代わりに分布・basis切替・測定負担が変わる。この競合を分割で制御することが中心となる。

### 最小事前検証

1. 既存H4をreproduction anchorにし、別の小さいHamiltonianを一つ追加する。既に使用したH5/H6やstretch geometryを自動的に未使用と呼ばない。
2. 各系でtailなし、tailが大きい候補、中間の少数候補を比較する。すべてのprefixを必須にしない。
3. まず同じ物理時間Tの複素信号を比較し、各手法にdelta・PF・RTE設定を選ぶ同等の機会を与える。
4. 決定論側には既知のbasis fusion・有力な高次PF等を適用する。random側だけ最新最適化を与えない。
5. 該当候補の短いcontrolled回路だけをcompileし、誤差・信号減衰・shot-factorと合わせる。

### 観測量

同じ精度・成功確率に必要なtotal logical cost、1-shot depth、total shots、basis-change割合、random-tail weight、deterministic work、古典生成・compile負荷。

state preparationは除外した層を作ってよいが、共通準備費Pを用いる感度

\[
G(P)=G_{\mathrm{no\text{-}prep}}+P N_{\mathrm{shots}}
\]

を分けて持つ。Pごとに再最適化しない曲線は固定設計の感度と明記する。Pを勝手に具体的なhardware値へ固定しない。

### 着地点

「DF化・共有回転・shot増加のどれがPR利得を左右するか」「count最小とdepth最小で分割が違うか」を、複数の独立条件で説明する。保証がempiricalならその範囲を明示する。

### 論文化の条件と停止判断

有利・不利の機構と独立条件での予測、または資源設計を変える情報が残れば研究候補。既存比較の再現のみで、追加の判断材料がない場合はbenchmark基盤として残す。一条件でPRが負けても全方向を捨てない。誤差棒だけを縮めるための再compileはしない。

---

## 4. PR-2：圧縮決定論Hamiltonianとランダム残差補完

### RQ

> Hamiltonianを高精度まで圧縮する代わりに、より粗く安価な圧縮表現を決定論側で実装し、その残差をランダム側で補うと、同じ元Hamiltonianに対する精度をどこまで低costで得られるか。

\[
H=\widetilde H_\theta+\Delta H_\theta,
\qquad
H_D=\widetilde H_\theta,
\qquad H_R=\Delta H_\theta.
\]

### 既知部分と今回の貢献候補

SPRINT/GRADEは、rank factorizationのresidualを明示し、remainderをqDRIFT/RTE等で処理する選択肢を既に提示する。RC-DFも既存である。[W4,W5]

従って、本案は「残差をrandomにする方法を初めて発明する」研究ではない。**圧縮度・残差表現・近似精度の実装上の選択を、ground-state energyまたはcoherent signalという指定taskで比較する応用・資源研究**を狙う。

### 部分ランダム化の役割

粗くしたH_Dを安く使いながら、ΔHを捨てない。圧縮の利点とランダム補完の負担を調整することが主問題となる。

ただし現行pilotでは、直接生成したrank-$r$ Hamiltonianがrank-12 Hamiltonianの先頭$r$ blockと一致することを確認している。この条件の「圧縮＋残差」は、単純なDF-prefix splitと同一である。従ってcore研究ではこの同一性を明示した資源評価とし、H_D自体を再最適化する非prefix圧縮は別のmethod-delta gateを通過した場合だけ扱う。

### 最小事前検証

- 小さい同一Hについて、既存の圧縮法で3程度の圧縮度を作る。
- 各点でΔHを明示的に再構成する。one-body補正、核反発、basis、正規順序を一致させる。
- 記録するのはrank、H_D実装work、ΔHの係数1-normと項数、各sampled residualのcost、分割PF誤差。
- 有望な1〜2点だけ、有限RTEを含む同じタスクのsignalと費用を評価する。

### 対照

1. 残差を捨てる圧縮法。ただし同じ元Hへの総誤差に圧縮biasを含める。
2. 許容精度まで厳密に圧縮した決定論PF。
3. 通常DF-prefix PR。現行prefix圧縮＋random residualはこれと同じ候補として一度だけ数える。
4. 非prefixの再最適化圧縮＋random residual。別契約のmethod-delta gate通過時だけ追加する。

ΔHをさらに圧縮・切断する場合は、その二次近似誤差を別予算へ入れる。H=H_D+H_Rと定義したことだけで実装誤差やfinite-RTE biasがゼロになるわけではない。

### 重要なリスク

小さいFrobenius残差や小さいenergy errorから、samplingに用いる1-normが小さいとは言えない。残差が多数の高価なPauli項へ広がると利益は失われる。符号の打消しをsamplingで無料利用できるとはしない。

### 着地点

「残差を捨ててよい領域」「rank-12まで決定論化する領域」「prefix residualをrandom補完する領域」を同じcoherent-signal精度で比較し、controlled wrapper、finite sampling、shotを戻した後の損益分岐を示す。新しい圧縮optimizerはcore研究の必須条件ではない。非prefix圧縮を扱う場合だけ、通常prefix PRに対する独立したmethod deltaを要求する。完成条件は[PR-2主研究契約](docs/research/pr2_primary_research_contract.md)を正本とする。

---

## 5. PR-3：ランダムtailに限定した外挿

### RQ

> 決定論backboneとouter PFを保ったまま、tailのrandomized discretizationだけを外挿すると、全Hamiltonianをrandom化して外挿する場合や通常PRより良いprecision–depth–shotの交換関係を得られるか。

### 出発文献

qFLOは、qDRIFTの異なるstep数の観測量をRichardson外挿で組み合わせる。[W7] そのままRTEの個々のunitaryが高精度になるわけではなく、observableまたは線形signal estimatorのbiasを減らす方法である。

最初はqDRIFT-tailで既知展開を利用する。RTE-tailへ変える場合は、normalization補正後の平均とcutoff誤差がどのparameterで展開できるかを改めて確認する。

### 最小設計

同じ分割s、同じH_Dの回路、同じouter step数q・deltaで、tail refinement rのみを変える。全回路の複素平均信号をz_rとする。

適用範囲内で

\[
z_r=z_\infty+c_1/r+c_2/r^2+\cdots
\]

なら、最初は

\[
z_{\mathrm{ext}}=2z_{2r}-z_r
\]

を比較する。z_infはこの固定backbone/outer-PFのtail-exact信号であり、元のexact-H信号とのPF biasは残る。

外挿はRe/Im、または選んだobservableへ適用する。`2 arg(z_2r)-arg(z_r)`という操作を無条件に代用しない。branch、信号半径、非線形変換によるbiasは別に扱う。

### 測定負担

独立推定に対して

\[
\operatorname{Var}(\widehat z_{\rm ext})
=4\operatorname{Var}(\widehat z_{2r})
+\operatorname{Var}(\widehat z_r)
\]

という統計増幅がある。cos/sinは別々に評価し、複素推定の誤差規約をそろえる。

一般の重みw_l、1-shot variance v_l、cost c_lで、予算最適化した連続sample数の費用は

\[
G\propto\varepsilon_{\rm stat}^{-2}
\left(\sum_l |w_l|\sqrt{v_lc_l}\right)^2
\]

となる。この標準的な配分式自体は新規性ではない。deterministic backboneは各試行で実行するので、その費用をc_lに含める。

### 最小事前検証

1. 非可換な小モデルと既存の小さい分子で、1〜2個のpartial splitを選ぶ。
2. r,2r,4rの3levelで、tail-exact参照へのbias、最終exact-Hへのbias、半径を分ける。
3. まずexact meanで外挿の挙動を検査し、次に物理的な有限shotを含む予測を作る。exact meanだけの改善をresource gainとしない。
4. 通常partial-qDRIFT、partial-qFLO、full-qFLOを比較し、有望なら既存partial-RTEとも比較する。
5. 全法にTを共通にし、精度を満たす設計を各自で選べるようにする。既存に不利な固定rや粗いbackboneを強制しない。

### 部分ランダム化の独自な実装問題

tailを小さくすると外挿bias係数は減る可能性があるが、backboneの反復費用が増える。どこまでH_Dへ移すべきかは、通常PRと外挿PRで異なる可能性がある。率自体が既知定理からそのまま得られても、具体的な分割・費用・crossoverの結果には価値があり得る。

### 着地点

tail-only extrapolationの正しいestimandと実装、bias/variance/backbone costの分解、通常PR・full extrapolationと比べた適用域を示す。最初から新しいRichardson定理や全RPE解析を要求しない。固定時間の信号研究で閉じる場合は、energy-estimation全体の利得と呼ばない。

---

## 6. PR-4：MLMCと部分ランダム化

### RQ

> 部分ランダム化でtailだけをfine/coarseに分けると、実際に測定可能なlevel correctionのvarianceと費用はどう変わるか。共通deterministic backboneを利用した実装で、通常PRに対する利益が残るか。

### 文献が保証・提案する範囲

MLMC-qDRIFT v2 Sec.5は、deterministic–randomized partitionと組み合わせ、減らしたrandom componentへMLMCを適用する方向を明記する。[W8] これは有望な出発点であるが、未実施性や世界初を保証する文ではない。

同論文Sec.2.3は、乱数列を共有しても別々の量子測定ではshot noiseがlevelとともに減らないこと、単純な制御superpositionの測定でも十分でないことを説明し、augmented difference-stateとscaled observableを導入している。

### 前の提案の補正

「shared random tailを一つ追加すればよい」「statevector上でvarianceが落ちたらGO」という扱いはしない。

独立のHadamard測定で条件付き平均をμ_f(ω), μ_c(ω)とすれば、単一shotの条件付き分散は

\[
\operatorname{Var}(Y_f-Y_c\mid\omega)
=(1-\mu_f(\omega)^2)+(1-\mu_c(\omega)^2).
\]

μ_f−μ_cが小さくても、各μが±1に近い等の条件がなければ、分散は一般にO(1)のままである。全分散はさらにpathwise varianceを含む。

シミュレータで真の測定確率を使い、両levelへ同じ一様乱数を入れるcouplingは、物理的な測定回路の実装を示したことにはならない。

### 最初に行う事前検証は「推定器と費用」の文献・回路監査

- 出力を通常observableにするか、Hadamardのcos/sinにするかを固定。
- 使用するfine/coarseの周辺分布が正しいことを確認。
- 論文の差分状態・補助qubit・nonunitary dilation・postselection・normalizationを、何で実装するかを書き出す。
- 共通H_Dの物理実行回数を数える。unitaryが両枝で同じなら共有可能な箇所があるが、全backboneが無料で消えるわけではない。
- 1 correctionの期待costに成功確率・再実行・readoutを戻す。

ここが定まったら、三level程度でpath varianceとquantum-shot varianceを別に測定する。RTEへ移す場合はqDRIFTのcoupling定理をそのまま使わず、新たな周辺分布と平均目標を確認する。

### 設計指標

\[
G_{\mathrm{ML}}\simeq
\varepsilon_{\rm stat}^{-2}
\left(\sum_l\sqrt{V_lC_l}\right)^2
\]

におけるV_lは実装した推定器の全variance、C_lは補助実装込みの費用である。両者をtail 1-normだけで置換しない。

### 着地点

測定まで含むpartial-MLMCの具体的構成と、splitごとのvariance–cost、通常PRとのcrossoverを示す。新しいvariance指数が得られなくても、既知法の実装・分割設計を改善できれば拡張研究になる。

### 撤退・縮小の判断

物理推定器の実装費用を含めると有利な条件が見つからず、その限界も既知評価の再現に留まる場合は、main themeにしない。逆に、単にlambda_Rへの置換で証明できるという理由だけではNo-Goにしない。

---

## 7. PR-5：tail方式を高精度化・選択する

### RQ

> 大きい構造化部分は決定論で処理し、残ったtailをqDRIFT、RTE、高次randomized方式のどれで処理するのが、要求精度・時間・補助qubit・実装primitiveに対して適切か。

### 出発文献

qSWIFTは高次randomized channelとobservable estimationを与える。[W9] PRHSは相関sliceとquasi-probabilityによる高次化を公式abstractで報告するが、今回本文を取得できなかったので、具体的なcost優位やsingle-ancilla接続を確認済みとはしない。[W10]

最初の実装候補は、一次論文と既知の回路が確認できるqSWIFTと既存RTEの二つで十分。PRHSやqSHIFT等を一度に全実装する必要はない。

### 部分ランダム化が効く場所

全Hamiltonianを高次random法に掛ける代わりにtailへ限定すると、高次補正を構成する対象、係数norm、必要なterm情報、補助回路の費用が変わる。これを実測・resource modelで明らかにする。

外側PFが二次のままなら、tailを高精度化してもouter error floorが残る。最初に同じouter参照でtail固有の比較を行い、次にouter精度の費用を加える。最初から全部を高次化しない。

### 最小事前検証

小さい二つの問題、少数split、二つの精度要求で、通常PRとhigh-order-tail PRを比較する。各手法のsystematic error、shot variance、quasi-probability重み、ancilla、controlled primitiveを含める。

相対的なquery count改善を、そのままlogical gate、runtime、物理qubit削減と同一視しない。qSWIFTが与えるchannel/observableの保証と、coherentなphase signalに必要な保証を先に対応付ける。

### 着地点

「どのtail方式をどの領域で使うか」という選択図と機構説明を残す。既存方式の比較・partial版の実装として、対象・保証・費用が一貫し、他の研究者が方式を選ぶ材料になることを目指す。

---

## 8. PR-6：相互作用描像との融合

### RQ

> 大きくても安価に時間発展できる部分をH_Dに置き、残差だけをランダム化すると、通常のpartial PFより有利になるか。相互作用描像でのtail変換費用を含めると、何を決定論側へ置くべきか。

\[
U(T)=e^{-iH_DT}\,
\mathcal T\exp\!\left[-i\int_0^T H_R^{(I)}(s)ds\right],
\quad
H_R^{(I)}(s)=e^{iH_Ds}H_Re^{-iH_Ds}.
\]

### 既知部分

Rajput–Roggero–Wiebeはinteraction pictureとrandomized/coherent simulationのhybrid法を既に提示する。[W11] この式や組合せ自体は新規ではない。

### 部分ランダム化の役割と差分候補

決定論側をnormだけでなくfast-forward可能性で選ぶ。量子化学ならone-body Gaussian部分を出発点にできる。一方、相互作用を多数足したH_Dを、同じ費用で正確に実装できるとはしない。

候補となる成果は、DFの具体的なbasis change、共役変換、time samplingを含めた実装の比較である。unitary共役でoperator normが変わらなくても、Pauli項数・LCU weight・実装costは保存されないことに注意する。

### 最小事前検証

同じ小Hamiltonianで、(i) lab-frame partial PF、(ii) one-bodyをH_Dとしたinteraction-picture random tail、を比較する。最初に1〜数個のsampled conjugated tailを実際に構成し、必要なGaussian rotationとキャンセルを数える。

作用素上の有利さではなく、回路costと時間依存simulation誤差の両方を評価する。任意の重要fragmentをH_Dへ追加する場合は、H_Dの内部近似費用も戻す。

### 着地点

「大きい項を決定論へ」よりも「安く正確に動かせる構造を決定論へ」という設計が有効な条件を、具体的な分子・モデルに対して示す。一般理論が直接適用できても、実装可能性と資源が新しい判断材料を与えるなら研究候補になる。

---

## 9. 全候補で先に守る四つの接続

### 9.1 channelの保証とcoherent信号を区別する

通常observable

\[
\operatorname{Tr}(O\,U\rho U^\dagger)
\]

と、phase estimationの

\[
z(T)=\langle\psi|U(T)|\psi\rangle
\]

は異なる。random-unitary channelの保証だけでは平均演算子のglobal/relative phaseは決まらない。

observable手法をHadamard信号へ移す一般的な出発点は、ancillaを含むcontrolled-Hamiltonian

\[
H_c=|1\rangle\langle1|\otimes H
\]

へ適用し、初期ancilla|+>に対するX/Y観測量として扱うことである。ただし、それぞれの手法がこの分解・primitive・制御化を許すか、費用がいくらかを確認する。system側の保証をそのまま移送したと書かない。

observable推定だけを成果にすることも選択肢であり、最初から全案にRPEを強制しない。これはユーザーの目的を勝手に変更する決定ではなく、研究候補としてのscopeの選択肢である。

### 9.2 同じ物理時間・目標量を比較する

各PFでdeltaを変える場合はT=q deltaを共通にする。同じ1 stepの比較を総taskの比較と呼ばない。

signal精度ε_z、energy精度ε_E、RMSE、絶対誤差、信頼度を別に定義する。fixed-T signalが改善したことを、そのまま全ground-energy推定の改善としない。

### 9.3 減衰・符号・外挿の負担を戻す

normalization B、quasi-probability重み、外挿係数、postselection失敗、量子shot noiseは、最終タスクの費用へ戻す。log Bの大きな相対変化が、同じ割合のshot削減とは限らない。

### 9.4 既存の最適化を両者へ公平に与える

deterministic側にも適切な高次PF、time step、basis reuseを与える。random側にも正しい分布、finite setting、variance配分を与える。

plain second-order、最小biasだけを選ぶ方法、内部work無料のB1aを、唯一の資源baselineにしない。一方、異なる研究問題の全アルゴリズムを無制限に実装する必要もない。主張に対応する最も近い対照を固定する。

---

## 10. 過去の検証をどう使うか

| 過去の内容 | 今回の利用 | 今回への拘束ではないこと |
|---|---|---|
| P-DでB1b/B2/B4が一致 | 当初はsimple modelを利用し、結論を左右する候補だけ詳しく測る | すべてのpartial-resource研究が無価値という意味ではない |
| P-A interval新規性が消失 | 既知のrun/basis共有を公平baselineとして再利用 | grouped/DF samplingの資源評価を禁止しない |
| P-Bの現H4でweight差が小さい | その固定範囲ではsignal診断を再利用できる | 別method・別stateでも不要とはしない |
| P-C stretch予測の破綻 | calibrationとholdoutを分け、誤差係数を外挿しすぎない | compressed-residualや他用途を排除しない |
| FRがtechnical note相当 | finite mean・phase・radius・controlの正当性確認へ利用 | FR理論の新規性を新研究の必須条件にしない |
| 高いcompile精度を追った経験 | 候補差が費用model誤差より大きいか先に見る | 全候補の同じ高精度化はしない |

上表の過去結果は、参照commitの研究概要とこの会話で確認済みの範囲を利用する。未取得の最新branch状態を推定したものではない。[R1]

---

## 11. 推奨する初期進行計画

### 段階A：候補ごとの1ページ比較案を作る

六件を数値実装する前に、各候補について次だけ埋める。

- 一文のRQと対象タスク。
- 最も近い一次文献と、既知の部分。
- 新しい理論／拡張／実装・資源評価のどれを成果にするか。
- partial splitを変えたとき何が変わるか。
- 一番大きい技術的不確かさ。
- その一件だけを調べる最小検証。
- 完成時に示す図・比較・利用できる結論。

ここで完全な新定理を要求しない。同時に、未読の論文のfuture workだけで新規性確定にしない。

### 段階B：計算を共用できる二案程度を選ぶ

推奨の組合せは次のいずれか。

**資源評価＋既存手法拡張：PR-1＋PR-3。**
同じ小系、分割、controlled signal、費用定義を共用できる。外挿という変更の寄与を、通常PRとの同一条件で評価できる。

**量子化学表現を重視：PR-1＋PR-2。**
残差の表現・費用を調べ、その結果をresource studyへ直接つなげる。SPRINTの残差処理と何を異なるtaskで比較するかを先に固定する。

PR-4はこの間に測定回路・実装費用の監査だけ進め、見通しが立ってから数値候補へ入れる。PR-5/6を無条件の次順番にしない。

### 段階C：最小pilotで問うことを一つにする

- PR-1：表現・実装をそろえると、partialの損益分岐がどれだけ変わるか。
- PR-2：圧縮で減るdeterministic workに対して、残差のsampling costは許容可能か。
- PR-3：外挿で節約するfine-tail workが、追加shotとbackbone再実行を上回り得るか。
- PR-4：物理測定でのvariance減衰と、その実装costを同時に示せるか。
- PR-5：高次tailの補正費用を入れても、既存RTEより良い領域があるか。
- PR-6：time-dependentな共役tailの回路費用まで入れて、interaction-pictureの利益があるか。

「有利な結果が出たか」だけでなく、費用の支配項と次に必要な証拠を記録する。意味のある不利領域や誤解を正す実装結果も評価する。

### 段階D：一つを主成果に選んだら、研究主張を簡単に固定する

主張例：

> 部分ランダム化のtailへ既存の外挿を適用し、deterministic backboneの費用を含めた精度・深さ・shotの交換関係を評価した。通常PRと全random外挿に対し、有利・不利な条件を説明し、独立した小系で確認した。

または、

> DF表現とsampling単位をそろえた同一taskの資源比較により、部分ランダム化の利益がtail normだけでなくbasis共有と制御化に依存する範囲を示し、実装選択に使える比較を提供した。

これは達成済みの文章ではなく、採用後に埋める主張の型である。

その後は、主張に不足する証拠だけ追加し、毎回新しいRQへ作り直さない。

---

## 12. 最小限の事前登録と、止め方

各pilotで固定すべきものは、以下の8項目程度でよい。別々の巨大gate文書を際限なく増やさない。

1. 元Hamiltonian、表現誤差、参照状態、既参照/未使用の区別。
2. 対象タスク、物理時間、誤差単位、成功確率。
3. 何をdeterministic、何をrandomにするかと、sampling/measurementの意味論。
4. 比較方法と、それぞれに許す最適化範囲。
5. 費用scope、control、ancilla、state preparation、重み・成功率の扱い。
6. 最初の候補数・sample数・実行時間上限と、境界時の一段拡張条件。
7. 正しさ、資源差、機構理解を別々に判定すること。
8. pilot後に止まり、テーマ採用か、追加の一件か、保留かを選ぶこと。

閾値は結果に合わせて変えない。ただし任意の「50%改善gate」を研究価値全体の定義にもしない。固定予算でPF名が同じ、あるいは一条件で不利だったことだけで分野全体を停止しない。

技術bugは仮説失敗ではない。対照が弱かったと分かった場合は旧結果を保存し、比較を修正して効果の意味を正す。結果に都合のよい対照だけを残さない。

---

## 13. 何がそろえば完成か

### 拡張・方法型の完了条件

- 既存法をpartialへ接続した操作・推定器が明確。
- その接続の正当性を、既知定理の適用条件または必要な補題で説明。
- 最も近い通常法、全random法、必要なdeterministic endpointと公平に比較。
- 訓練例だけでない条件で、利益・不利益の理由を説明。
- 費用に抜けがなく、適用できない条件を明示。

### 資源評価型の完了条件

- 過去にない比較条件・費用項・用途を具体的に特定。
- 共通タスクと再現可能な設計選択。
- 単に大きい/小さいでなく、どの構造が違いを生むか説明。
- 結論を変える不確かさを調べ、結論に影響しない細部を無限に精密化しない。
- 原稿の主張と直接計算／model依存の範囲が一致。

「すべての既知法に勝つ」「必ず新しい漸近次数」「一般最適性定理」「H12まで実行」は共通必須条件ではない。反対に、部分ランダム化という名前を含むだけでは成果にならない。

---

## 14. 文献一覧と根拠の扱い

以下は2026-09-27の確認対象。プレプリントの主張は著者の報告として扱い、今回独立に完全証明・実装を検証したとはしない。

**W1** Jakob Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum **7**, 020332 (2026). DOI: 10.1103/ynxb-p2xq.  
https://arxiv.org/abs/2503.05647  
https://journals.aps.org/prxquantum/abstract/10.1103/ynxb-p2xq

**W2** Matthew Hagan and Nathan Wiebe, *Composite Quantum Simulations*, Quantum **7**, 1181 (2023). DOI: 10.22331/q-2023-11-14-1181.  
https://arxiv.org/abs/2206.06409  
https://quantum-journal.org/papers/q-2023-11-14-1181/

**W3** Matthew Pocrnic, Matthew Hagan, Juan Carrasquilla, Dvira Segal, Nathan Wiebe, *Composite QDrift-Product Formulas for Quantum and Classical Simulations in Real and Imaginary Time*, Physical Review Research **6**, 013224 (2024).  
https://arxiv.org/abs/2306.16572  
https://doi.org/10.1103/PhysRevResearch.6.013224

**W4** Pablo A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741v1 (2026). Fig.1、Eqs.(2)–(5)、Sec.IVを確認。  
https://arxiv.org/abs/2606.30741  
https://arxiv.org/pdf/2606.30741

**W5** Oumarou Oumarou, Maximilian Scheurer, Robert M. Parrish, Edward G. Hohenstein, Christian Gogolin, *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*, Quantum **8**, 1371 (2024). DOI: 10.22331/q-2024-06-13-1371.  
https://arxiv.org/abs/2212.07957  
https://quantum-journal.org/papers/q-2024-06-13-1371/

**W6** Shota Kanasugi, Riki Toshio, Kazunori Maruyama, Hirotaka Oshima, *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*, arXiv:2603.22778 (2026). v2 introductionのUWC/PR/STARの位置付けを確認。  
https://arxiv.org/abs/2603.22778  
https://arxiv.org/html/2603.22778v2

**W7** James D. Watson, *Randomly Compiled Quantum Simulation with Exponentially Reduced Circuit Depths*, arXiv:2411.04240。取得PDFのTheorem 1と関連研究を確認。  
https://arxiv.org/abs/2411.04240  
https://arxiv.org/pdf/2411.04240

**W8** Pegah Mohammadipour and Xiantao Li, *MLMC-qDRIFT: Multilevel Variance Reduction for Randomized Quantum Hamiltonian Simulation*, arXiv:2604.26865v2 (2026). Sec.2.3のmeasurement問題、augmented estimatorとdilation、Sec.5のpartial拡張案を確認。  
https://arxiv.org/abs/2604.26865  
https://arxiv.org/html/2604.26865v2

**W9** Kouhei Nakaji, Mohsen Bagherimehrab, Alán Aspuru-Guzik, *High-Order Randomized Compiler for Hamiltonian Simulation*, PRX Quantum **5**, 020330 (2024). arXiv版の題名は*qSWIFT: High-order randomized compiler for Hamiltonian simulation*。  
https://arxiv.org/abs/2302.14811  
https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020330

**W10** Davide Cugini, *Pathwise Random Hamiltonian Simulation*, arXiv:2608.29756 (2026). 今回はarXivの公式検索abstractまでを確認し、PDF/HTML本文を取得できなかった。そのため具体的拡張の採択根拠にはしていない。  
https://arxiv.org/abs/2608.29756

**W11** Abhishek Rajput, Alessandro Roggero, Nathan Wiebe, *Hybridized Methods for Quantum Simulation in the Interaction Picture*, Quantum **6**, 780 (2022). DOI: 10.22331/q-2022-08-17-780.  
https://arxiv.org/abs/2109.03308  
https://quantum-journal.org/papers/q-2022-08-17-780/

**R1** 本プロジェクトの研究概要、固定commit `71169d817c165b76a9e25fc6f6a16ade28ffe069`。P-A/B/C/D、R3、FRの停止範囲と再利用可能基盤を確認した。  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/71169d817c165b76a9e25fc6f6a16ade28ffe069/docs/research/研究概要・現状.md

以上は候補選定のための重点調査であり、全類似研究を網羅した新規性証明ではない。実行する候補を選んだ後に、その最も近い2〜3文献の定理・回路・benchmarkとの具体的対応を深める。

---

## 15. 2026-09-27現在の実行位置

本計画の「文献監査 → PR-2/PR-3最小pilot → 必須STOP」までは完了した。PR-2を主題候補、PR-3を
`STOP_PR3_VARIANCE_BACKBONE_DOMINATES`とし、続いて追加数値なしで
[PR-2主研究契約](docs/research/pr2_primary_research_contract.md)と
[S1--S3結果前事前登録](docs/research/pr2_s1_s3_preregistration.md)を作成した。

実装監査で、pilotが確認したgeneration-prefixと、現行通常PR実装が使うweight-ranked prefixの同一性は
未記録と判明した。このためS0 identity gateを置き、一致すれば一つのB2へ統合、不一致ならB2-G/B2-Wを
別baselineとして残す。full-random $L_D=0$もB3 strong baselineへ追加した。順序差だけを新手法とはしない。

S1--S3ではrank 6を主anchor、rank 3/9をcontrol、H4 1.0 Åをdevelopment、H4 1.30 Åをindependent
transferとする。finite-RTE候補は$r=1,2,4,8,16,32$、$K=2,4$、共通complex-signal errorは0.05、
primary costは状態準備なしfull Hadamard wrapperのshot込みexpected compiled RZである。各stage後に必ず
停止し、S2/S3へ自動進行しない。

現行statusは`PR2_S1_S3_PREREG_V1_FROZEN_FOR_EXTERNAL_REVIEW_EXECUTION_NOT_AUTHORIZED`である。
[dry-run manifest](artifacts/pr2_s1_s3_preregistration/2026-09-27/pr2_s1_s3_dry_run_manifest_v1.json)では、
分子計算、signal評価、compile、trajectory sampling、量子shotを全て0件と記録した。次に許されるのは
事前登録の外部批判レビューだけであり、development/independent snapshot、identity gate、PR-2固有の
signal/shot runnerとtestを満たすまでS1を開始しない。
