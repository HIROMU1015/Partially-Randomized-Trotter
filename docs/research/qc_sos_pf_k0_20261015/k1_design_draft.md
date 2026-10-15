# SOS-PF K1：条件付き検証草案（未認可・実行用契約ではない）

2026-10-15 JST。[K0監査](technical_audit.md)の全体状態は `UNDETERMINED_DATA_REQUIRED`。この文書は科学レビュー用の設計案であり、実装・入力開封・数値実行を許可しない。中心的な分割生成規則がまだ特定されていないため、現時点ではlaunch-readyではない。

## 1. 数値に進む前に必要な科学判断

次の二つを別々に判断する。

1. **分割生成研究**：新しい規則Rが、元Hの係数・粒子数・同じ古典予算から、厳密かつ指数化可能なPSD分割を生成する。そのRを、既知SOS Gram SDP、PSD化DF、平方完成、一体再配分、Gram因子回転から区別して固定できるか。Rが未定なら性能比較へ進まない。旧Dや閉じた再配分系列をRとして再開しない。
2. **保証研究**：既存の分割を固定して、取得可能な情報だけで有用な有限hのground-energy certificateを与えること自体を、独立成果として検討するか。これは「新しい分割アルゴリズム」の目的から主題を変える可能性があるため、GPTの科学判断が必要。今回採択しない。

一般SOSを一度解いてgapが小さいと示すだけでは、どちらも満たさない。逆に既知部品の組合せであるという理由だけで統合構成を排除もしない。

## 2. 反証する中心仮説

分割生成研究を選ぶ場合の仮説は、**事前固定したRが、同じ元H・情報予算・精度・成功確率に対し、PSD化DFと強いPF対照を上回る利用可能なhと量子費用を実現し、その差をSOS gapだけでなくK・実bias・step単価に分解して説明できる**こと。

保証研究を選ぶ場合の仮説は、**同じ分割・同じ回路で、既知のsector/低energy/commutator評価より有用な有限hを、真のground数値を設計に渡さずに認証できる**こと。実biasは固定回路なので同じである。

成功基準の主指標・必要改善幅・不確かさの扱いは、どちらを選ぶかの科学判断後に固定する。2倍改善を正式gateにしない。単なる「上界違反なし」を有用性の成功としない。

## 3. 最小比較の二軸

| 軸 | 実験 | 独立な診断 |
|---|---|---|
| I：保証 | 一つのPSD分割に通常／sector commutator、適用可能な既知低energy評価、提案評価を適用 | bound値、有限h有効条件、認証可能h、証拠生成・検証費用。同じ回路の実biasは変えない |
| II：分割 | A=PSD化DF、B=規則Rの構造化PF-SOS、必要な場合だけC=一般Hermitian平方を、同じ保証法で比較 | 元Hへの残差、下界、G、K、M、幅、実固有位相bias、overlap、step価格 |

各分割の強対照はS2、CS2、S4、利用可能なnear-integrable PF。二分割が本当に可解ならHejazi custom PFも価格に戻す。純processorの固有値不変を守る。Rの情報が対照にも使えるなら同じ入力として与える。CS2のd取得に価格・誤差・失敗確率を入れ、oracle dの比較は診断層へ分ける。block-encoding/QSVTを自動追加しない。

## 4. 入力とblind性の案

新しい分子Hamiltonianは生成しない。既に保存された最大2入力を利用する案とする。

- Development候補：既存のlinear H4、1.00 Å、STO-3G、8 spin orbitals、DF rank12、N_alpha=N_beta=2の固定snapshot。ファイル・Hamiltonian・DF順序・係数精度・sector・既存ground情報のhashを別契約で指定する。このK0では科学arrayをdecodeしていない。
- Holdout候補：別の保存H4 geometry、例えば既存1.30 Å snapshot。存在、正確なrank・geometry・basis・sector・byte hashを確認してから指定し、R・パラメータ・判定規則をdevelopmentで固定するまでground truthを開かない。この文書だけで当該snapshotを正式選定しない。
- この2入力は別geometryへの限定transferであり、独立分子やサイズscalingの証拠にはならない。過去に同じ入力を他研究で使った履歴も明示し、「未知の分子」と呼ばない。
- 部分ランダム化のL_D探索は行わない。全保持DFの決定論PFを対象とし、L_Dは適用外（既存prefixを暗黙に変更しない）。

Rへ渡すのはH係数・sector・許可した古典上界／下界と予算だけ。真のE0、exact ground vector/RDM、finite-h bias、励起gapはsealed diagnostic側に保持する。ground stateの量子利用可能性と古典的な無料truth accessを分ける。

## 5. 科学的指標と有限step診断

記録を少なくとも次に分ける。

1. 元Hへの厳密同値／別分解誤差、Gram PSDとsector保存、複素実化前後のrank・一体残余下界。
2. `G=U−L`、各f、K、`K G` と既知上界。元E0から測った真のgapは診断専用。
3. dとfinite-h bias、h⁴残差、branch projector/overlap。係数・上界・実値を同じ列に混ぜない。
4. 認証可能なhと、経験的に目標を満たすhを別々に保存。gapをtruthから取得した保証はoracle層であり利用可能certificateへ数えない。
5. controlled Givens・n/nnまたはRZ/RZZ・scalar処理・step間fusionの静的会計。T価格は同じ合成精度規約の記号式に限り、compile/T合成は別scope。
6. QPE/RPEの統計・alias・overlapを含む量子resource proxyと、古典時間・メモリ・証拠検証価格を別resourceとして記録。K1小行列の実行時間を本番SOSの計算量としない。

h gridの候補は8点まで、H係数と許可された分割normだけから共通gridをtruth開封前に固定する。例として最大のnorm和Lambda_maxに対する `h_j=2^(−j)/Lambda_max, j=0,...,7` があるが、確定値は別契約で決める。全点を保証区間内と見なさず、Cauchy半径・gap条件を各点で検査する。順序は固定、h refinement・追加gridは0。branch不明、上端到達、剰余支配は「未判定」とし、grid端だけでstep節約なしを断定しない。

## 6. 提案する上限（採用前・実行未認可）

| 項目 | 上限案 |
|---|---|
| 保存入力 | 2、最大8 modes、sector次元70以下。必要以上のspin混合なし |
| 表現 | 各入力最大3。R未固定ならB/Cを作らない |
| PF kernel | 各表現最大6。CS2の共有行列は重複しても上限内で数える |
| h | 共通最大8点、refinement 0、事後拡張0 |
| PF spectrum診断 | 最大288 cells（2×3×6×8）、H参照diagonalization最大2 |
| SDP・optimizer | **この最小案では0**。必要ならsolver・変数・呼出数・証拠精度を固定する別の明示認可が必要 |
| 新分子生成・compile・T合成・量子shot・MC | すべて0 |
| 科学attempt | 一回、retry 0、入力／gateの事後緩和0 |
| aggregate memory | 1 GiB案。実装後の静的見積もりで確定、超過時停止 |
| output / log | 64 MiB / 4 MiB案。過大な行列dumpなし |
| phase／total wall-time cap | **設けない**。toolのpoll timeoutと科学run上限を混同しない |

これは回数予算の草案で、科学実装やauthorization manifestを発行するものではない。SDPを0にすることでRを準備できないなら、既知法の数値だけで代用せず、実行前に草案を差し戻す。

## 7. 性能準備と参照照合

将来の実行が明示認可された場合に限り、保存timingとstatic sourceから支配負荷を推定し、CPU数・空きメモリ・GPU可否を確認してbackendを固定する。小sectorでは既存sector／vectorized経路を優先し、dense全Fockを作らない。独立input/kernel cellの並列はaggregate memory内、初期案は最大2 process・BLAS/OpenMP各1 thread。実機資源に応じて結果前に確定する。GPU導入だけのために追加runを起こさない。

最適化経路と必須referenceの一致を、別契約内のcorrectness cellsで先に確認する。cache keyはHamiltonian・sector・分割・順序・h・precision・source・compiler規約を含み、truthとselectorを分離する。同一hのpaired比較を保持する。実資源使用と限界を記録し、比較計測なしのspeedup倍率を主張しない。

## 8. STOP条件と情報価値

- 入力同一性、厳密再構成、PSD、sector保存に失敗したら停止。閾値を後から緩めない。
- Rが既知のconstant shift／平方完成／一般SOS因子化だけで、独立した生成・保証差がないと判明したら、性能探索を広げない。
- PF-native化でGが悪化した場合、元の一般SOS下界へ戻して利益を水増ししない。
- gapやdがoracle情報でしか取得できず、その価格を含む改善を評価できない場合は、診断結果と未閉鎖を残して停止。
- `K G` が減っても実bias・認証可能h・step単価のいずれも改善しない場合、元の算法仮説の支持とはしない。
- 保存入力2件で見込みが得られても研究正式採択、追加geometry、rank、Gram rotation、SDP階層、候補B/Cへ自動移行しない。

K1には「gapの良さがPF性能へ移らない」「fragment増で利益が消える」という反証の情報価値はあり得る。しかし**未固定Rを救う探索**や既知補題の数値再確認だけなら価値は低い。採択基準・新規性の対象・具体Rが未確定の現在は、K1へ進まずGPT科学判断を先に置く。

**K1 NOT AUTHORIZED / NOT STARTED。K0のMandatory STOPを維持する。**
