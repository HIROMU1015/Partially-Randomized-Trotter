# N3 occupation-aware v1：固定8入力の結果・研究方針の科学レビュー

- 作成日：2026-10-12 JST
- 対象：N3の固定8入力・結果commit `e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c`
- 公開確認commit：`86f57f05a36e7e6639e827486d300bd049bf2438`
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- Branch：`n3-occupation-transfer-v1-execution-20261012`
- レビュー開始承認：ユーザー「レビューを開始して」。本結果と研究方針に対する承認。
- 適用指示：`research_project_instructions_gpt_codex_review_approval_2026-10-12.md`
- 実施状態：結果・研究方針レビュー完了。全保存certificateの独立再検証、最小保証幅の数値比較、追加実験は実施していない。
- 計算実行状態：本レビューで新しい科学batch、固有値計算、最適化、登録条件の再評価、回路compile、sampling、既存checkerの再実行は行っていない。数式の導出と資料の読み取りを行った。

## 0. 結論

**固定8入力の検証は完了し、登録されたOCC固有の追加保証は0件である。2-base基準未達を維持する。現OCCを中心研究テーマとして採択し、新しい入力や実分子へ拡張する判断は出さない。**

ただし、強い対照が8/8を認証したことは、既存法で認証できない8例をOCCも救えなかったこととは異なる。今回の主比較では、対照が不受理になる入力が0件で、追加受理の余地がなかった。したがって、一般的な条件族の無価値・数学的同値・理論上の改善不能までは結論できない。

次に推奨するのは、**新しい移送batchではなく、既存証拠の事後診断と研究上の位置づけの整理**である。固定された証拠から認証可能な幅を比較し、今回の二値的な採否では見えなかった利益が残るかを調べる。生成器の再起動、κ探索、τ/Z再生成、入力追加、δgoalの変更実行を伴わない形を優先する。この事後解析を、元の移送成功や新しい独立検証として数えない。

N3は「主研究としての拡張を見送り、証拠・数理による見極めに作業を縮小」とする。N1の現候補scan停止、N2保留、他Trackの独立性を維持する。新たな数値的な事後解析の実行にも、その範囲の明示承認が必要であり、本レビュー承認を実行承認に読み替えない。

## 1. 研究目的と、今回の判断の位置

この研究の目的は、新しいHamiltonian simulationの構成・本質的な算法改善を得ることであり、単なるテスト基盤の完成や比較表の作成を主要貢献に置き換えることではない。

N3は現在、一外部modeを空に固定する縮約について、係数由来の情報から基底エネルギー差の保証を生成する構成を検討している。これまでの経緯は次のように整理できる。

1. 旧直接generatorは、初期12入力でNORMまたは五点法の安全な併用を超える受理集合を示さなかった。
2. T3 developmentでは、同じ外部下界を使ったOCC条件と旧LMI族の厳密分離を示す保存証拠が得られた。
3. しかし、同じT3に強い外部下界を与えると、NORMでも認証できた。条件式の強さと、情報取得も含めた算法の利益は別問題になった。
4. その切り分けのため、外部下界三行・判定法四種類・既存証拠の継承を備えた固定8入力の比較を実施した。
5. 今回は、その強い対照とOCCがともに8/8を認証し、OCC固有の追加受理が0となった。[R1–R4]

今回の判断は、単に「負の結果なので研究終了」とするものではない。一方、「まだ可能性があるので同型の入力を増やす」とするものでもない。**何が観測され、何が測定設計上まだ見えていないかを分け、追加作業を行う情報価値があるかを判断する。**

## 2. 資料の取得範囲と証拠の限界

### 2.1 実際に参照した資料

GitHub接続で、固定結果報告、機械summary、compactな一次result、公開・取得報告、独立保存監査の記録、比較契約、以前の数学監査を参照した。直前の二つのGPTレビュー記録も、ローカルに提供された原文の関連箇所を参照した。

| 資料 | 主な用途 |
|---|---|
| 固定結果報告 [R1] | 科学scope、登録結果、下界別採否、費用・限界 |
| `result_summary.json` [R2] | 方法別・下界別の保存集計、事後費用の定義 |
| `result.json` [R3] | 正式status、分母、gain、gate、STOP |
| 数学監査 [R4] | OCC条件、Schur導出、T3の既存分離証拠 |
| 追補v2 [R5] | 継承済みB*、raw gain、fallback、費用会計 |
| 公開・再取得報告 [R6] | 固定payload、remote照合、監査の適用範囲 |
| `independent_saved_audit.json` [R7] | 5,008,702 checks PASSという保存監査記録 |

### 2.2 全certificate ledgerは、この環境では本文を取得できなかった

`primary_ledger.json`のGitHub blobは `24946eb4f76486e698e9e1033ca04264e4dfc595` と参照できたが、file取得は本文が空、raw取得はサイズ制約、blob取得はエラーとなった。別の取得経路も試したが、本文の完全取得・SHAの独立再計算には至っていない。

これは「GitHubに未公開」と判明したという意味ではない。公開報告はremoteでの取得・保存監査を記録している。**今回の取得限界は、詳細ledgerの本文アクセスであり、compactな一次resultや結果報告の欠落ではない。**

そのため、本レビューは、報告された結果の意味・比較設計・研究方針の判断を扱う。次のことは確認済みと主張しない。

- 本レビューによる全5,008,702条件の独立再実行。
- 個別の全κ・τ・Zを用いた再認証。
- 保存候補から得られる最小保証幅の数値的な大小。
- 外部環境での科学batch再現やimmutable CI。

戦略判断に必要な主結果は取得できており、この境界を明示したうえでレビューを完了する。個別証拠の含意を次に評価するためには、既存ledgerを薄い閲覧用資料へ正確に抽出する作業が有用である。既存の同じ大きいファイルを再pushする必要はない。

### 2.3 固定identity

| 役割 | identity |
|---|---|
| frozen source | `83f51cd8a4848572c66a5535c67d0d0a5f3868cd` |
| source fingerprint | `41b951dae4968a2b679e61c9e61e761dc8253babeb2c610ccc1886fe3646fcb6` |
| execution / authorization commit | `a4311eece09275e4a27259db3fe12e98974dbadc` |
| result payload | `e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c` |
| publication commit | `86f57f05a36e7e6639e827486d300bd049bf2438` |
| plan SHA256 | `a825ece44d9bf8266a9c83d390572139d3799b748419b644515d79230e5c0d0f` |
| input SHA256 | `c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4` |
| primary seal SHA256 | `8b1e75e89da78c13b82f24896db387adec6e76597ec12669a7549279592d0d69` |

上記hashは取得資料に記録されたidentityである。本レビューが全対象bytesを新たに取得して独立hash検証したという表ではない。[R1,R3,R6]

## 3. 固定されたtaskと数学的意味

### 3.1 Hamiltonian class

\[
H=\sum_{ij}h_{ij}a_i^\dagger a_j+\sum_{i<j}J_{ij}n_i n_j,
\qquad h=h^\mathsf T,\quad J_{ij}\ge0.
\]

固定粒子数νのsectorを、一つの外部modeが空のP側と、占有されたQ側に分ける。Active mode数をmとする。既存証明の対象は1≤ν≤mであり、今回の実験はν≥2である。

\[
H_\nu=\begin{pmatrix}A&B\\B^\dagger&C\end{pmatrix},\qquad
A=d\Gamma_\nu(h_A)+V_{A,\nu},\quad V_{A,\nu}\succeq0.
\]

外部との一体結合ベクトルをvとすると、CARから

\[
BB^\dagger=d\Gamma_\nu(vv^\dagger),\qquad \|B\|^2=\|v\|^2
\]

を得る。外部占有側は

\[
C=h_{dd}I+d\Gamma_{\nu-1}(h_A+\operatorname{diag}J_{id})+V_{A,\nu-1}.
\]

これらの恒等式、ordering、外部densityの数え方は、以前の数学監査で取り扱われている。[R4]

### 3.2 保証対象

\[
a=\lambda_{\min}(A)\le U,\qquad C\succeq cI,\qquad c>U,
\]

のもと、保証したい量は

\[
0\le a-E_0(H_\nu)\le\delta.
\]

Uは明示trialのRayleigh上界であり、真のaを算法へ渡したものではない。cも真のgapではなく、係数から構成した外部sector下界である。

これは、任意状態での時間発展operator norm保証、状態準備の成功確率、QPE全誤差、最終的な量子gate資源を直接評価するtaskではない。[R1]

### 3.3 入力scope

| base | graph | full modes | active m | ν | external | scale | δgoal |
|---|---|---:|---:|---:|---:|---:|---:|
| C1 | chain | 7 | 6 | 2 | 1 | 1 | .004 |
| R1 | ring | 8 | 7 | 3 | 4 | 1.5 | .006 |
| L1 | ladder | 9 | 8 | 3 | 0 | .75 | .003 |
| S1 | star | 10 | 9 | 4 | 8 | 1.25 | .005 |

各baseに同normの二結合方向があり、計8入力。同一base内でUと外部下界は共通。geometry、化学basis、DF rank、L_D、PF窓は適用外である。4baseは意図的な合成入力であり、iidの分子標本ではない。[R1]

## 4. 保存結果の整理

### 4.1 主比較

B*は、三下界行のNORM/FIVE/OLD-DIRECTにより取得したPSD-valid候補を、同じc_starへ継承して再判定した安全なORである。OCC gainは `raw_OCC_at_c_star AND NOT B*` と定義される。自己継承、旧法からのfallback、運用unionをraw gainに含めない。[R5]

| 方法 | c_starで保証した入力数 |
|---|---:|
| NORM | 8/8 |
| FIVE | 4/8 |
| OLD-DIRECT | 6/8 |
| raw OCC | 8/8 |
| 継承を含む強い既存対照B* | 8/8 |
| OCC固有gain | 0/8、0/4 base |

正式statusは `NO_ADDITIONAL_ACCEPTANCE_OBSERVED`、事前2-base基準は未達。これを変更する理由はない。[R1,R3]

FIVEやOLD-DIRECTだけに比べてOCCの件数が多いことは、主対照に対する利益ではない。NORMを落とした対照へ事後的にすり替えない。

### 4.2 下界別の件数

| 外部下界 | NORM | FIVE | OLD-DIRECT | OCC |
|---|---:|---:|---:|---:|
| c0 | 6/8 | 3/8 | 4/8 | 6/8 |
| c_occ | 8/8 | 4/8 | 6/8 | 8/8 |
| c_star | 8/8 | 4/8 | 6/8 | 8/8 |

したがって、今回の8/8を得るために、c_starの最大化やOCC条件自体は必要でなかった。**取得済みのc_occをNORMへ渡すだけでも8/8である。** 特にS1二方向がc0で不受理、c_occで両方法とも受理となった変化は、共通下界の強化として説明できる。[R1,R2]

この観察は「固定粒子数情報に意味がない」という結論でもない。粒子数情報は、外部下界の構成側では有用に働いた。しかし、その効果をOCC独自の方向付き判定条件の効果に付け替えることはできない。

### 4.3 不受理・未判定・secondary

96 raw record中、NORMの条件不成立2件、有限探索による証拠未取得23件。Whole-case未判定は0である。個別のFIVE/OLD-DIRECT等で証拠が見つからなかったことは、条件族全体の数学的不可行性ではない。[R1,R3]

主比較のgainが0なので、全8入力がsecondaryの対象外となり、separator提案・secondary eig/PSDはいずれも0だった。**これはseparator生成が8件失敗したことではない。** 正常な分岐で未実行である。[R1]

### 4.4 監査

保存監査は5,008,702 checks PASS。これは、保存された有理数証拠・binding・元行列再構成等を調べた記録である。5,008,702個の独立Hamiltonian、統計標本、科学実験を意味しない。[R7]

公開報告は42blobs / 12,898,275 bytesのremote取得・照合と、remote exportに対する同じstdlib checkerのPASSを記録する。これは取得性と保存証拠監査であり、外部環境における科学batch再実行やimmutable CIとは区別する。[R6]

## 5. 追加保証0件をどう解釈するか

### 5.1 結論を二つに分ける

**観測された結論：** この固定入力・固定δgoal・取得情報・有限generatorでは、OCC固有の追加受理は観測されなかった。登録基準未達である。

**そこから導けない結論：** 既存法が認証できない入力にもOCCが無効である、理想OCC条件族がNORMと同値である、あらゆるδやHamiltonianで新条件の価値がない、という一般結論。

強い対照が不受理の入力は0件だった。従って「既存法が不受理だった入力のうち何割をOCCが救うか」という条件付きの能力は、この主比較では評価できていない。元の主指標の分母8を0へ変更するのではなく、**主指標の情報上の限界を説明する**ものである。

### 5.2 これはtechnical failureへの再分類ではない

計画通りの入力・判定・保存が完了し、全入力が主比較へ寄与している。対照の受理率が上限へ達したからといって、元結果をtechnical inconclusiveや成功へ変更しない。数学的保証の比較の正当性と、入力集合が追加効果を識別できるかは別問題である。

### 5.3 前回のGPT設計判断にも限界があった

前回レビューでは、ν≥2、同norm二方向、共通下界、強いbaseline、結果前固定を評価した。しかし、それらは比較の公正性・適用範囲を支えるものであり、今回の入力が主対照に対する追加認証の機会を持つことまでは保証していなかった。

入力の採否を結果前に密かに計算すべきだった、という意味ではない。**理論的にどの領域で差が生じるかを明確にせず、固定した二値閾値だけで研究価値を見極めようとした点に、設計側の限界があった。** これをCodexの実装失敗や条件違反へ転嫁しない。

### 5.4 同じ採否でも、保証の強さが同じとは限らない

あるδgoalで双方が通ったことは、双方の最小認証幅が同じという意味ではない。例えばOCCがより小さい幅を保証できる可能性、異なる結合強度まで保証できる可能性は、今回の二値結果だけでは分からない。

ただし、その可能性を既に観測した利益のように扱ってもいけない。今回、個別保存証拠の最小幅は数値的に比較していない。新しいδで再計算して元の成績を置き換える提案でもない。

## 6. 計算費用から言えること

報告された単独運用経路の再構成見積りは次の通り。[R1,R2]

| input | BASE-OR wall見積り s | OCC operational wall見積り s |
|---|---:|---:|
| C1_edge_pair | .007305 | .008166 |
| C1_distributed_three | .007384 | .008158 |
| R1_edge_pair | .012135 | .013246 |
| R1_distributed_three | .012387 | .013374 |
| L1_edge_pair | .018222 | .019608 |
| L1_distributed_three | .018776 | .020035 |
| S1_edge_pair | .096838 | .390802 |
| S1_distributed_three | .099440 | .392357 |

同じ最終採否に対して、凍結したOCC運用経路の見積りは全入力で大きい。従って、**今回の固定taskにOCC経路を追加する計算上の利益は示されていない。**

ただし、この表はevent依存とfirst-observed primitive費用から再構成した値であり、cold standalone benchmarkでも量子資源費用でもない。固定した下界取得順序、fallbackの証拠埋込み、探索の呼出し順序に依存する。これを「OCCは理論的に何倍遅い」「どんな実装でも不利」と読み替えない。

NORMで通るならOCC探索を呼ばない、という運用は可能である。しかし、その最適化は不要な処理を省くもので、OCC固有の認証能力を実証するものではない。今回、速度差を縮めるためだけに実装を改修して別benchmarkを開始することは勧めない。

全batch wall 7.660213秒は保存checkerを含み、単独経路の合計時間とは会計範囲が異なる。producer eig/PSDは2272/719、denied callsなしで、主比較の0件を資源予算枯渇に帰属させる記録はない。一方、RSSはsamplingで、最大観測間隔の空白がある。瞬間peakの厳密な非超過を主張しない。[R1]

## 7. 既存証拠から保証幅を読み出す解析式

**本節は、本レビューで導出した解析的な整理であり、保存resultに既に報告された新しい数値結果ではない。** この式に基づく8入力の数値的な事後評価は未実施である。Codexによる独立した数式確認を経て利用する。

### 7.1 固定されたOCC証拠

同一input、U、cを固定し、g=c−U>0とする。保存されたκ≥0、τ、Zについて、

\[
K_\kappa=\kappa h_A-vv^\dagger,\qquad
Z\succeq0,\quad K_\kappa-\tau I+Z\succeq0,
\]

が検証され、

\[
s=\nu\tau-\operatorname{tr}Z\le\sigma_\nu(K_\kappa)
\]

が得られたとする。この固定証拠が認証するδは

\[
\delta>0,\qquad \kappa\le g+\delta,\qquad
\delta(g+\delta)\ge\kappa U-s
\]

を満たす。q=κU−s、q_+=max(q,0)と置くと、δの許容域の下端は

\[
\boxed{
\delta_w=\max\left\{0,\ \kappa-g,\
\frac{\sqrt{g^2+4q_+}-g}{2}\right\}.
}
\]

δw=0の場合、元のδ>0契約では下端はinfimumとして扱う。正のδwでは、上記条件に等号を含めて検査できる。

### 7.2 導出

δ≥0でδ(g+δ)は単調増加する。q≤0ならこの条件は自動で成立し、q>0なら二次式の非負root以上が必要十分である。残りのκ≤g+δはδ≥κ−gと等価である。その共通部分の下端が上式となる。

κとKκの単位はそれぞれenergy、energy²であり、qもenergy²、δwはenergyである。

### 7.3 NORMと旧法にも同じ会計を適用できる

β²=||v||²とすると、NORMの認証幅の下端は

\[
\delta_{\rm N}=
\frac{\sqrt{g^2+4\beta^2}-g}{2}.
\]

NORMのOCC埋込みκ=0、τ=0、Z=vv†ではs=−β²となり、δw=δNである。

旧法のPSD-valid証拠κ,μにはs=νμを用いて、同じ式で認証幅を評価できる。これにより、OCCだけに事後的に有利な証拠集計を許すことなく、保存された旧法全候補と比較できる。

### 7.4 保存集合について比較する量

同じU,cを与えたとき、NORMおよび保存された全旧法PSD-valid候補についてδwの下端を取り、既存証拠集合の基準とする。OCC側は独自生成されたPSD-valid候補の集合を別に集計し、fallbackと自己継承を識別する。

これは**固定された有限保存証拠集合の含意**であり、再最適化した最良generator、OCC条件族のglobal optimum、真の射影誤差ではない。結果を見て保存されていないκやτ/Zを生成して補充しない。採用証拠しか残っていない場合と、全PSD-valid候補が残っている場合は、coverageを区別する。

### 7.5 重要な保全・数値条件

- 元input、source、U、c、実際のκ、τ/Zまたはμ、元のδgoal、元のstatusを変更しない。
- 固定するのはκそのものである。元のx=κ/(c−U+δgoal)を固定して別δに対するκを作り直すと、別のKκ・別探索になってしまう。
- 派生したδproofと元δgoalは別fieldに保存し、元証拠IDと元matrix hashを参照する。
- 平方根を通常の浮動小数点で計算した値を厳密なcertificateとして出さない。有理数区間、または安全側の有理数δを用い、κ≤g+δとδ(g+δ)≥qを厳密に確認する。
- ほぼ同じ幅の大小を、表示丸めだけで勝敗にしない。区間が重なる場合は大小未判定とする。
- 新たな解析による大小は事後診断として報告し、登録されたgain=0と2-base未達を維持する。

## 8. 理想OCCがNORMより強くなり得る機構の解析

**本節も今回の解析的整理であり、新しい独立実験、既報性の確定、または実装済みの有限generatorの完全性証明ではない。** 以下の命題は、既知のKy Fan変分原理を使って導く。新しい定理としての優先性は主張しない。[L1]

### 8.1 一体空間における結合方向の強制占有エネルギー

v≠0、β=||v||、u=v/β、Q_u=I−uu†とする。Q_uの値域u⊥上に制限した一体行列をh_perpと書き、

\[
E_{\rm forced}(u)=u^\dagger h_Au+\sigma_{\nu-1}(h_{\perp})
\]

と定義する。これは、結合方向uを必ず占有するν粒子Slater部分空間について、一体エネルギーを最小化した値である。相互作用を含む真のactive ground energyではない。

正確には、一体空間のrank-ν直交射影ΠでΠu=uを満たすものの中で

\[
E_{\rm forced}(u)=\min_{\Pi^2=\Pi=\Pi^\dagger,\;\mathrm{tr}\Pi=\nu,\;\Pi u=u}
\operatorname{tr}(h_A\Pi).
\]

固定νにおける一体固有値和の変分表示を使えば、u方向1粒子とu⊥方向ν−1粒子へ分けた上の式を得る。

### 8.2 NORMを超えられない十分条件

\[
E_{\rm forced}(u)\le U
\]

なら、最小化射影Πを一つ選び、任意のκ≥0に対して

\[
\begin{aligned}
\sigma_\nu(\kappa h_A-vv^\dagger)
&\le\operatorname{tr}\{\Pi(\kappa h_A-vv^\dagger)\}\\
&=\kappa E_{\rm forced}(u)-\beta^2\\
&\le\kappa U-\beta^2.
\end{aligned}
\]

従って

\[
\sigma_\nu(K_\kappa)-\kappa U+\delta(g+\delta)
\le -\beta^2+\delta(g+\delta).
\]

右辺はNORM marginである。NORMが不成立なら、正確な固有値和を使う理想OCCも成立しない。これは有限探索が見つけられないという問題ではなく、**このUに対するOCC条件族そのものの限界**である。もちろん、別U、別の保証族、別Hamiltonian全体の不可能性ではない。

### 8.3 正確な固有値和を使う族での、厳密な幅改善の条件

\[
\phi(\kappa)=\sigma_\nu(\kappa h_A-vv^\dagger)-\kappa U.
\]

これはrank-ν射影Πごとのaffine関数の最小値なので凹で、φ(0)=−β²である。κ=0の最小化射影はΠu=uを満たすもの全体であり、右微分は

\[
\phi'(0+)=E_{\rm forced}(u)-U.
\]

この微分関係は次のように確認できる。κ>0の最小化射影Πκを取り、κ→0の部分列極限を考える。射影集合はコンパクトで、極限Π0はuを占有する。差分商は

\[
\frac{\phi(\kappa)-\phi(0)}{\kappa}
=\operatorname{tr}(h_A\Pi_\kappa)-U+
\frac{\beta^2(1-u^\dagger\Pi_\kappa u)}{\kappa}.
\]

最後の項は非負なので下極限はE_forced−U以上。一方、κ=0でE_forcedを達成する固定射影を変分trialに使うと、上極限はE_forced−U以下となる。従って右微分が一致する。

g>0、β>0、1≤ν≤mを固定し、NORMの幅δNを第7節のrootとする。E_forced>Uなら、小さい正κでφ(κ)>−β²となる。δ=δNではOCC marginが厳密に正となり、κを十分小さく取ればκ<g+δNも満たす。δをδNからわずかに減らしても両条件は保たれる。

逆にE_forced≤Uなら第8.2節より、δ<δNでOCCは成立しない。したがって、正確な固有値和を使う理想族について

\[
\boxed{
\text{同じU,cでNORMより厳密に小さい認証幅が存在する}
\iff E_{\rm forced}(u)>U.
}
\]

これは、登録されたδgoalでOCCが必ず追加受理する条件ではない。どれほど小さくできるか、固定bit/call予算のgeneratorが証拠を発見するかは別である。v=0ならNORMの幅は0なので、正の幅におけるstrict improvementの議論は適用しない。

### 8.4 NORMとの改善であり、強対照B*への優位とは限らない

ν=1でもこの改善機構はあり得るが、その場合OCC族は旧one-body LMI族と一致する。従ってE_forced>Uだけでは、新条件が旧法に対して独自の利益を持つことを示せない。

この解析は、NORMとの差がそもそも生じ得るかを整理するためのもので、主対照B*をNORMだけへ縮小する提案ではない。旧法全体との厳密分離には、従来どおり旧条件族の証拠やseparatorとの比較が必要となる。

### 8.5 現8入力については未評価

本レビューではE_forcedを8入力について計算していない。全例でE_forced≤Uだった、または一部で大きかった、とは主張しない。ここで導いたのは、追加の入力総当たりを提案する前に改善機構を考えるための解析式である。

この量の新規数値評価は、保存データの単なる閲覧とは異なる。実施する場合は、その範囲を別に明示して認可を受ける。現在の新規eig禁止を暗黙に解除しない。

## 9. 科学的貢献と先行研究との距離

### 9.1 既に残っているもの

N3には、係数からのtrial上界・外部下界、CARに整合した一外部mode分解、有理PSD証拠、証拠継承・fallback、有限generatorと条件族の区別という再利用可能な構成が残っている。T3の同情報での旧LMI族との厳密分離も、今回の0件によって取り消されるものではない。[R4,R5]

ただし、これらをすべて新規の科学的発見として数えることはできない。Schur法による固有値評価、Ky Fanの固有値和・trace最小化は既存の理論である。[L1,L2]

### 9.2 文献との比較で、今回確認したこと

| 一次資料 | 参照した内容 | N3へ直結させない点 |
|---|---|---|
| Liang et al., On Generalizing Trace Minimization [L1] | Ky Fan型trace最小化の既存性、固有値・固有vectorを用いる理論的枠組み | OCCの具体的手順がその論文で全て既報だとは判定していない |
| Dusson et al., The Feshbach–Schur map and perturbation theory [L2] | Schur系手法による厳密な固有値・固有関数評価 | 同論文の射影・gap条件を現在のoccupation射影へ無条件に流用しない |
| Chamaki et al., SQuISH [L3] | Hamiltonianと近似ground-state wavefunctionの双方を利用する反復的なHamiltonian疎化 | N3と同じ入力情報、同じ保証意味論、同じ資源評価だとは扱わない |
| Motta et al., Low rank representations… [L4] | 電子構造のlow-rank分解とtruncationを量子simulationのgate費用へ接続する先行研究 | 現N3のclassが一般の電子構造HamiltonianやDF全体にそのまま適用可能とはしない |

今回の外部確認では上記一次資料の公開書誌・abstractを再確認した。全文の網羅的な新規性調査を完了したわけではない。書誌上、SQuISHの参照先arXiv:2211.16522の原論文はChamaki、Hadfield、Klymko、O'Gorman、Tubmanによるものであり、本資料はこの取得した情報に従う。

N3もtrial Uを使うので「状態情報を一切使わない」とは表現しない。正確には、登録された係数trial上界を用い、真のmany-body ground stateやgapを運用入力へ流入させない構成である。

### 9.3 論文化の候補と未充足点

「新しいOCC generatorが広く既存方法より有利」とする主要claimは、今回の証拠では支持されない。

一方、将来の着地点として、特定classに対する保証族の厳密分離・改善不能条件・証拠生成費用・具体的な縮約決定を統一する理論／算法研究は考えられる。ただし第8節の整理を含め、既報性と非自明な差分をまだ確定していない。単に標準的変分原理を使ったことを主要新規性にしない。

実分子での勝利や一般Hamiltonian全体への定理を、研究成立の絶対条件として追加しない。理論的に意義のある限定classでの定理・構成でも研究になり得る。しかし現状の資料だけでは、その主要貢献を採択する根拠が十分ではない。

### 9.4 量子アルゴリズムとの接続

現在は、指定された一つのmodeを削除してよいかを認証する段階であり、削除modeを選ぶ算法や多mode逐次縮約、縮約後の状態準備・時間発展・測定費用を完成させた段階ではない。[R1]

例えば、active energy aの推定値をe_A±ε_Aで得た成功事象上では、N3幅δにより

\[
E_0\in[e_A-\epsilon_A-\delta,\ e_A+\epsilon_A]
\]

とできる。従って、より狭いδには誤差予算上の意味があり得る。しかしそれを量子費用の減少へ変えるには、active側の算法・誤差・初期状態等の追加契約が必要である。本式は区間包含の推論であり、新しいQPE費用の測定結果ではない。

## 10. 代替方針を比較した判断

| 方針 | 利点 | 問題・今回の判断 |
|---|---|---|
| 同型の合成入力を増やす | 手順は既存generatorで実現可能 | 今回の主比較が飽和した理由を解決せず、成功例探索になりやすい。現時点では非推奨 |
| δ、振幅、外部energyを変えて勝てる領域を探す | 差が出る機構を調べられる可能性 | 元の独立検証とは異なる新しい探索的実験。今は実行しない |
| bit/call予算やκ optimizerを強くする | 発見漏れを減らす可能性 | 主比較ではOCCも既存法も成功している。登録goalに対する0件を改善する根拠がない |
| cold timingやfallback最適化 | 固定運用の余計な費用を減らせる | 算法固有の認証利益がまだなく、中心課題から外れやすい。主研究作業にしない |
| 実分子・一般二体・多外部へ拡張 | 応用への距離を縮める可能性 | 現classの証明・入力情報が自動移送しない。証明と価値が未整理のまま規模だけを増やさない |
| 既存証拠の幅・機構を整理する | 新入力や新探索をせず、二値指標で見えない利益を判別できる | 事後診断であり、保存集合の限界がある。次に優先する限定作業 |
| 直ちにN3全体を無価値として終了する | 無駄な追加計算を避けられる | 飽和した主比較だけから一般的な改善不能までは言えない。全面終了の論拠にはしない |
| N3を主研究から外しprimitiveとして保持する | 成果を保全し、新テーマへ研究資源を振り向けられる | 保存証拠の診断でも独自利益や応用機構が残らなければ、有力な着地点 |

**今回の選択は、主研究採択・追加batchは見送り、既存証拠と解析による見極めへ縮小すること。** その後も根拠が増えなければ、N3を保証primitiveとして保持して中心テーマ探索を別方向へ移す。これは一般数学の否定ではなく、現在の研究資源配分の判断である。

## 11. 次のCodex作業：既存証拠と研究差分の整理

以下は本レビューからの次作業の提案であり、既に実行した内容ではない。既存batchの認可を流用せず、ユーザーがこの範囲を承認した後にCodexへまとめて委ねる。

### 11.1 まず既存資料を使える形にする

固定payloadの`primary_ledger.json`から、case/row/method/witness ID、元input/source、U/c/δgoal、κ/μ/τ/Z、PSD-valid状態、元の採否、origin・継承先、費用の参照を、losslessな小さいファイルへ抽出する。

大きいファイルを再pushするのではなく、元のJSON pointer、blob・SHA256、抽出valueとの対応を示す。元の有理数を表示丸めで置き換えない。既に公開されている部分は再公開せず、必要な抽出物だけを追加する。これは情報取得上の問題への対応であり、新しい科学結果の生成ではない。

### 11.2 保存証拠の事後的な保証幅比較

第7節の式を独立に確認し、保存されたPSD-valid証拠だけに適用する。同一caseのUと比較先cを全方法で共通にし、NORMおよびFIVE/OLD-DIRECTの全保存候補とOCCを比較する。

元のδgoalと元のprimary判定を不変で保存し、派生した幅の表を別artifactとする。変えるのは元実験の条件ではなく、保存証拠から新たに読み出す事後的な評価量である。新たな数値解析の実行として、範囲の明示承認を必要とする。

保存集合から必要情報が得られない場合は不足とcoverageを報告し、generator、eigensolver、κ探索、τ/Z生成を起動して埋めない。最小幅の差がない場合にも、それをOCC族全体の不可能性としない。

### 11.3 数理と研究差分の整理

第8節の解析を独立に確認する。まず証明・境界条件・既知原理との関係を整理し、現8入力に対する新規固有値計算は自動実行しない。

同時に、既存文献・既存研究資料に基づき、OCC固有の幅改善がどの研究taskで意味を持つか、同じ情報の旧法に吸収されない差分が何かを整理する。新しい算法候補は「候補」として、数学的入力・出力・保証・比較対象・古典費用・未解決点を明示する。実分子や他Trackの条件を無断で現在のclassへ移植しない。

### 11.4 完了報告と保全

関連する抽出・独立確認・保存解析・文献整理を可能な範囲で一つの依頼にまとめる。通常のsource修正やテストのたびにGPT詳細レビューを要求しない。

最終報告には、元primary結果、事後診断の結果、未取得・未判定項目、source/result identity、取得性、計算実行の認可範囲、STOP状態を含める。原source・入力・結果・認可・marker・STOP、N1/N2、他Trackを変更しない。

新input、新δgoalでの再実行、precision/探索予算変更、登録batchのretry、実分子・多外部・一般二体への拡張は、この次作業へ含めない。

## 12. 次の科学的判断点と着地点

### 保存証拠に固有の幅改善がある場合

改善は有限保存証拠集合についての事後的な知見である。元の2-base移送gateは未達のままとする。差の大きさ、取得費用、旧法の証拠coverage、保証幅を縮めることの研究上の意味を評価する。直ちに独立移送や量子資源優位とはしない。

再設計候補を採択する根拠になり得るが、次に独立入力・主指標を固定すること自体が新しい科学設計になるため、重要な判断としてGPTへ戻す。

### 幅改善も確認できず、具体的な適用機構も残らない場合

現在のN3は主研究候補から外し、保証primitive・下界・検証基盤・限定否定結果として保存する方針を勧める。「世界のどのHamiltonianでも無価値」ではなく、現方向を継続する根拠が十分に増えなかったという研究判断である。

### 証拠が不足する場合

不足を明示し、保存資料だけでは判別できないことを報告する。欠損を仮定で補わず、未取得を不可能性としない。既定契約内で補える資料はCodexがまとめて整備する。新しい科学計算が必要なら、その情報価値と範囲を別途判断する。

### レビュー・実行承認

今回の結果・研究方針レビューは完了した。同じ内容の再レビューは不要。次の重要な科学結果または契約変更が生じた段階で必要性を再判定し、新しいレビュー開始承認を得る。

一方、本レビューは次の計算を自動認可していない。元one-shotは消費済み、retry=0、Mandatory STOPを保持する。通常の準備作業を細分化して形式的な承認待ちにすることと、必要な実行承認を保持することは区別する。

## 13. 支持される主張・支持されない主張

| 主張 | 判定 |
|---|---|
| 固定8入力の比較が完了し、OCC固有の追加受理は0だった | 保存結果が支持 |
| c_occ＋NORMでも8/8だった | 保存集計が支持 |
| 既存法が認証できない8入力でOCCが失敗した | 誤った解釈 |
| 今回のraw有限探索の不成功が、旧族の不可行性を示す | 不支持 |
| 以前のT3での同情報・旧LMI分離が今回否定された | 不支持 |
| 保存監査PASSは500万件の独立科学検証を意味する | 不支持 |
| OCCがより小さいδを保証できる余地はない | 今回未評価 |
| 保存証拠を固定したときのδ下端は第7節の式で表せる | 本レビューの解析的導出、数値適用は未実施 |
| 理想OCCのNORMに対するstrict幅改善は第8節の条件で整理できる | 本レビューの解析的導出、新規性・有限実装の完全性は未認定 |
| 今回OCC operationalの再構成費用見積りは大きかった | 保存表が支持 |
| OCCはどの実装でも本質的に遅い | 不支持 |
| 実分子・QPE・PFの総量子資源を削減した | 評価対象外 |
| 現時点でOCCを中心研究テーマとして採択すべき | 根拠不足のため見送り |

## 14. 以前の判断から変えたこと

前回は、強化対照に対して独自の追加受理が残るかを判別するため、固定8入力の限定比較を進めることを支持した。

今回、新しい結果として、主比較の双方が8/8、c_occ＋NORMでも8/8、固定OCC運用費用の追加が観測された。そのため、**同じ比較の入力数拡大や実分子拡張を支持する段階ではない**と判断を更新する。

同時に、前回の二値比較には対照の飽和という識別上の限界が現れた。従って「0件ならOCC一般を終了」という短絡はせず、保存証拠の保証幅と数理機構を利用して、研究上の余地を最小限の追加作業で見極める方針へ縮小する。

## 15. 参考資料・参照範囲

### 固定repository資料

- [R1] 固定結果報告：
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c/docs/research/n3_occupation_transfer_v1_results_20261012.md
- [R2] 機械summary：
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c/artifacts/n3_occupation_transfer_v1/2026-10-12/execution_audit/result_summary.json
- [R3] Compact一次result：
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c/artifacts/n3_occupation_transfer_v1/2026-10-12/run1/result.json
- [R4] 数学監査：
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/docs/research/n3_occupation_redesign_mathematical_audit_20261012.md
- [R5] 実装前追補v2：
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/83f51cd8a4848572c66a5535c67d0d0a5f3868cd/docs/research/n3_occupation_transfer_amendment_v2_20261012.md
- [R6] 公開・remote取得報告：
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86f57f05a36e7e6639e827486d300bd049bf2438/docs/research/n3_occupation_transfer_v1_publication_20261012.md
- [R7] 独立保存audit：
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c/artifacts/n3_occupation_transfer_v1/2026-10-12/run1/independent_saved_audit.json
- [R8] 詳細primary ledger：本レビュー環境では本文完全取得に至らず、内容を独立検査済みと扱わない。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c/artifacts/n3_occupation_transfer_v1/2026-10-12/run1/primary_ledger.json

### 一次文献

- [L1] Xin Liang, Li Wang, Lei-Hong Zhang, Ren-Cang Li, “On Generalizing Trace Minimization,” arXiv:2104.00257, 2021; Linear Algebra and its Applications 656, 483–509 (2023)。本レビューでは公開書誌・abstractを確認。Ky Fan変分原理の背景として参照。
  https://arxiv.org/abs/2104.00257
- [L2] Geneviève Dusson, Israel Michael Sigal, Benjamin Stamm, “The Feshbach–Schur map and perturbation theory,” arXiv:2105.02058, 2021; EMS Press, DOI 10.4171/ECR/18-1/5。公開書誌・abstractを確認。現在のoccupation射影への直接的な定理適用とはしない。
  https://arxiv.org/abs/2105.02058
  https://ems.press/books/ecr/187/3518
- [L3] Diana B. Chamaki, Stuart Hadfield, Katherine Klymko, Bryan O'Gorman, Norm M. Tubman, “Self-consistent Quantum Iteratively Sparsified Hamiltonian method (SQuISH): A new algorithm for efficient Hamiltonian simulation and compression,” arXiv:2211.16522, 2022。公開abstractにおけるHamiltonian＋近似ground-state wavefunctionの利用を参照。
  https://arxiv.org/abs/2211.16522
- [L4] Mario Motta et al., “Low rank representations for quantum simulation of electronic structure,” arXiv:1808.02625, 2018。公開abstractのlow-rank factorization/truncationと量子回路資源の接続を背景として参照。N3の実分子適用の証拠ではない。
  https://arxiv.org/abs/1808.02625

---

**最終状態：科学レビュー完了。登録結果は追加保証0のまま。中心研究テーマ採択と新batch拡張は見送り。次はCodexによる既存証拠・数理・研究差分の整理を提案し、その実行は別の明示承認に従う。**
