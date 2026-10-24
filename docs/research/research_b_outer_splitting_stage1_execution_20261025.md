# 研究B：Outer D/R Splitting固定16行の科学実行結果

2026-10-25 JST。**最終状態：COMPLETE。全correctness gate・16行比較・登録scopeを完了し、mandatory STOP。**

## 1. SOURCE・実行・保存

科学SOURCE **`b85dd328423de5b1159f11939835f04db90f3187`** を変更せず、[固定契約・command](research_b_outer_splitting_stage1_execution_ready_20261025.md)で一回実行した。ユーザーは固定科学実行と意図しない技術的中断時の最小修正・fresh attemptを許可したが、今回は**中断・SOURCE修正・再実行とも0件**。追加許可を使う必要はなかった。承認証明書等は作成していない。

| 保存対象 | 場所 |
|---|---|
| 実行原本 | `/tmp/prt-research-b-outer-stage1-science-20261025/artifacts/research_b_outer_splitting/20261025/once-16rows` |
| 公開原本（同一bytes） | [once-16rows](../../artifacts/research_b_outer_splitting/20261025/once-16rows/)：result／inputs／binding／graph／events／partial_results／status／STOPとinputs.npz、計9ファイル |
| 全16行の閲覧用表 | [COMPARISON.csv](../../artifacts/research_b_outer_splitting/20261025/COMPARISON.csv)（保存JSONからの列抽出。数値再計算・新規fitなし） |
| process実績 | [stdout／stderr／time](../../artifacts/research_b_outer_splitting/20261025/execution_logs/attempt-1/) |

結果公開branchは `codex/research-b-outer-stage1-results-20261025`。実行SOURCEと結果公開commitを区別する。旧B2-D Stage 1のSOURCE・raw・grant・ledger・marker・STOP・契約は変更しない。以下は同じ実行内の登録診断の報告であり、追加科学実行や別条件の再解析ではない。

## 2. 固定範囲と完了確認

Synthetic 2-mode、one-body＋3 DF因子、N=1／mode0 occupation、L_D=2 weight-ranked prefix。分子geometry／AO basis／分子DF rank探索は非該当。
T=±0.18、基本q=2／δ=±0.09、refined-S2のみf=2・4 step。Outer-S2／refined-S2／outer Yoshida4／SABA2、D内部A2 Yoshida4／A3 published Morales Y8m10b、n_D=1。全signed R stageでr_RTE=2／K=2、tolerance metadata=1.0、独立occurrenceのanalytic moments、samplingなし。Complex128、correctness atol=1e-12、phase診断無効、記述的診断のみ。

| 登録境界 | 実績 |
|---|---:|
| 比較行 | 16／16 |
| DAG dispatch | 906／906 |
| 共有exponential key／backend.expm | 490／490 |
| RTE config／moments | 各24／24 |
| Signal expectation | 80／80 |
| 入力高位境界 | 10／10 |
| D数学gate／outer数学順gate／rescaling gate | 40／8／16、全PASS |
| Input gate／weak診断 | 各1、完了 |
| Events | 3074（枠3100以内） |
| 失敗・未完了・保存失敗 | 0 |

Attempted／completedは全登録境界に一致、failedは空。130個のgate参照名は共有後66 nodeに対応する。保存eventsでも全66 nodeの完了が全104比較dispatch（operator_error88＋signal16）に先行することを確認した。入れ子の境界を足して独立科学call総数とはしない。

### 入力とcorrectness

実選択DF因子は[0,1]、残差[2]、sector basis[1,2]、占有mode[0]。実Hamiltonian・state・ordered D／tail・配列identityは[inputs.json](../../artifacts/research_b_outer_splitting/20261025/once-16rows/inputs.json)とnpzへ保存した。

- H=D+R+cIの残差：**6.16645e-17**。
- D内部3対のcommutator norm：0.023895、0.003920、0.01607445。D/Rは **0.00428043**。必要な非可換性を確認。
- D内部の独立数学列照合最大：**5.55382e-17**。
- 独立outer数学順照合最大：**2.30586e-16**。
- Raw／corrected rescaling整合最大：**6.69036e-16**。

全correctnessがatol1e-12以内。これは実装した列・meanの整合検査であり、同じatolを改善率の信頼区間や誤差保証へ転用しない。

## 3. Operator／corrected coherent signalの比較

以下は上記固定synthetic入力・全条件におけるT=±0.18の値。符号間の差は表示桁未満のためまとめた。16行の全精度値、X/Y・半径・raw／correctedはCSV／raw JSONを正とする。

| D内部 | Outer | Corrected physical operator誤差 | Corrected physical signal直接誤差 | D近似operator差（ideal-tail対exact outer） | Finite-RTE operator差 |
|---|---|---:|---:|---:|---:|
| A2 | S2 | 5.09754e-7 | 2.26738e-7 | 2.55080e-10 | 3.24973e-11 |
| A2 | Refined-S2 | 1.27309e-7 | 5.66296e-8 | 1.59422e-11 | 4.06158e-12 |
| A2 | Yoshida4 | 1.75192e-9 | 6.75687e-10 | 1.34530e-9 | 4.89611e-10 |
| A2 | SABA2 | 5.96822e-9 | 2.90506e-9 | 2.65496e-10 | 4.06170e-12 |
| A3 | S2 | 5.09500e-7 | 2.26639e-7 | 4.28989e-15 | 3.24970e-11 |
| A3 | Refined-S2 | 1.27293e-7 | 5.66234e-8 | 4.78524e-15 | 4.06153e-12 |
| A3 | Yoshida4 | 5.28691e-10 | 4.52693e-10 | 1.57089e-14 | 4.89611e-10 |
| A3 | SABA2 | 5.70444e-9 | 2.80198e-9 | 2.44826e-15 | 4.06151e-12 |

同じD内部PFのouter-S2に対するcorrected signal誤差の比は、refined-S2が約1/4.00、Yoshida4がA2で約1/335.6・A3で約1/500.6、SABA2がA2で約1/78.05・A3で約1/80.89だった。これは同じTの記述比較であり、同じ資源budgetの改善率ではない。

**今回確認できたこと：** Outerの変更によるoperator改善がfinite-RTE corrected physical coherent signalにも残った。元S2で見えにくかったD内部PFの差がYoshida4では現れ、A2のD近似差は約1.35e-9、A3は約1.57e-14だった。A3を内部精度対照に残す判断材料となるが、長いA3列の費用優位は未評価。

**残った制約：** A3＋Yoshida4ではexact-D/R outer operator誤差は約1.43712e-10、近似D＋ideal-tailもほぼ同程度だが、finite-RTE差が約4.89611e-10で、corrected full誤差は約5.28691e-10になった。Signalでもideal-tail約7.0036e-11に対しfinite-RTE差約4.5818e-10が残る。Outerを改善した後のtail精度が次の制約になり得る。この観測を別入力・時間窓へ一般化しない。誤差normは向きの違う行列差であり、表の項を単純加算・差引きして等式にしない。

## 4. Normalization・raw信号・構造的費用

| Outer | R occurrence／execution | Normalization積 N | Raw physical signal誤差（A2／A3でほぼ同表示値） |
|---|---:|---:|---:|
| S2 | 2 | 1.000055855727 | 5.57312e-5 |
| Refined-S2 | 4 | 1.000027927648 | 2.78574e-5 |
| Yoshida4 | 6 | 1.000365893033 | 3.64737e-4 |
| SABA2 | 4 | 1.000027927648 | 2.78492e-5 |

全occurrenceのN_jをorderedに保持し、rawからcorrectedへのrescalingを全行で確認した。**Yoshida4のcorrected mean改善を、raw信号の誤差改善とは扱わない。** 長い正負stageを含む今回のYoshida4ではfinite-RTE差とNも大きくなった。負時間だけに因果帰属はしない。
補正後mean、attenuatedなraw mean、必要shot数は別物である。Shot／variance／native費は今回計算しておらず、誤差が小さい列を総資源最小と認定しない。

CSVにはD run数、kernel反復、literal D block指数数、負R stage数、実D/R時間和と絶対R時間も残した。Literal D指数数はA2でS2／refined／Yoshida4／SABA2が60／120／180／90、A3が420／840／1260／630。追加の隣接block融合・native合成後のgate数ではない。全行で公称Tとfloat係数由来の実時間和を区別し、係数を補正していない。

## 5. SABA2と弱摂動診断

||D||=1.08120585、||R||=0.08303995、比=0.07680309、||[D,[D,R]]||=0.007707435、||[R,[R,D]]||=0.0003567221。
今回の入力ではSABA2のfull operator／corrected signal誤差がS2・refined-S2より小さかった。ただしweak条件の自動判定閾値は設けておらず、このnorm比だけで保証の成立を認定しない。単一の|T|、固定q／n_D／r／Kの実行であり、SABA2一般四次、Yoshida4の実装全体の収束次数、普遍的なouter律速は確認していない。

## 6. 技術履歴・環境・保全

Attemptは1件、exit0、SOURCE差分0、技術的中断0、retry／resume0。完了後の追加科学計算0。Matplotlibの既定cache先が非書込で専用一時cacheへfallbackする警告だけがあり、停止・数値FAIL・環境修正は発生しなかった。
既存のCPython3.11.9環境を再利用した。依存version・実interpreter／import元・環境変数はbindingへ保存した。32 CPU、開始時空きRAM約52.7 GiBを確認し、既存2×2 sector・vectorized演算・共有cacheを利用。新たなCPU／thread／RAM上限や共有環境変更はない。
Whole-process wall **7.44秒**、user CPU12.92秒／system CPU3.00秒、max RSS368936 KiB。これは報告・公開作業全体の時間ではなく、科学processの実績である。速度倍率は主張しない。

原本9ファイルとstdout／stderr／timeを変更せずコピーし、byte一致を確認した。Input package内の既存Gaussian basis／sector処理は今回の許可された科学入力範囲で実行したが、native PF full wrapper／controlled回路のbuild・compile、sampling、追加grid、高精度化は行っていない。準備時の人工105件PASSは技術的証拠として区別し、今回はSOURCE不変のため人工suiteの不要な再実行はしていない。

## 7. 次の研究判断とSTOP

**研究Bのユーザー／判断担当：** Outer改善がfull signalに残るという限定結果と、Yoshida4で表面化したD近似・finite-RTE・normalizationの負担を基に、次を判断する。たとえば同精度でのRTE条件とnative費用を扱う限定比較へ進むか、R occurrenceが少ないSABA2を費用面で残すかが論点である。これは次段階の提案で、正式な採択・scientific GOではない。
**Codex：** 今回の結果・machine-readable evidenceを公開・remote照合し、mandatory STOPする。次の実装・科学計算は別途の具体的指示・必要な実行許可後だけ行う。RTE条件変更、別fixture、追加時間、native compile、Stage 2へ自動進行しない。通常の研究B作業として統括報告はしない。
