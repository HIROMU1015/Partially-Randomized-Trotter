# 案E：強いbaselineと全attempt費用

2026-11-08。発明した1/2 system qubit・2-label回路のみ。分子/H4/DF/L_D/δ条件は非該当。[人工結果](../../../artifacts/o2_e_label_erasure/2026-11-08/artificial_run01/TEST_RESULTS.json)、[実装・人工検証](E_INSTRUMENT_AND_ARTIFICIAL_TESTS.md)を正本とする。

## 四つの効果を分ける

1. 正係数coherent平均：通常label mixtureのM2=1からp_sへ変化。
2. Flagをsuffixより先に読む：同じM2のまま失敗時の後続処理を省略。
3. Pool/層別配分：今回新しいpool選択・配分最適化はしていない。
4. 追加アクセス：二つの二重制御W、PREP/逆PREP、pool ancilla、flag測定とfeedback。

共通Bは一度、Aは成功時一度だけ。SELECTでA W0 BとA W1 Bのfull pathを二重に構築しない。ただし二つのcontrolled W blockは両方gate列として実行する。controlの非active branchを無料呼出しと数えない。

## 同じtargetのbaseline

| 方式 | 取得・decoder | M2／variance | 案Eとの差 |
|---|---|---|---|
| 通常label別Hadamard | 古典label jをp_jで取得し、cB→cW_j→cA、X/Y±1 | M2=1、V=1−μ² | 各attemptのWは一つ |
| 標準positive LCU・earlyなし | 同じPREP/SELECT/逆PREP、全attemptにA/readout、最後にflag、失敗0 | M2=p_s、V=p_s−μ² | 失敗でもsuffix費を払う |
| 標準positive LCU・同じearlyあり | 同じflagをAより先に読み、成功だけA/readout、失敗0 | 上と同じ | **案Eと操作・全attempt推定量・論理分岐費用が同じ** |
| 案E | 上記最小adapter | 同じ | **同じearly LCUに対する増分利益0** |
| label層別化 | 各labelの独立±1平均をp_jで加重 | Σp_j²σ_j²/n_j | pilot費も必要。Eより小さい場合あり |
| 適用可能なdirect/grouping | 同じHermitian座標observableを既知回転で測る | observableのM2に依存 | ancilla/controlなしで済む例を含む |

独立baseline builderはacos(√p)、adapterはatan2(√(1−p),√p)で同じPREPを記述する。微小な丸め差を除き、全branch state・classical bit・実行block数が一致した。**案E固有の操作的追加利益は今回の構成に存在しない。**

正係数LCUは[Childs–Wiebe](https://arxiv.org/abs/1202.5822)の既知構成に属する。同じearly flagを標準LCUにも与えると、追加のoracleや省略構造が残らない。LCU一般やO2全体のNO-GOではない。

## 強い対照が勝つ明示人工例

発明した1 system qubit、A=B=I、W0=Z、W1=X、p=2/5、ψ=|0⟩、X座標。S=(2/5)Z+(3/5)Xは既知Hermitian observableで、両Wは非可換。

| 指標 | 値 |
|---|---:|
| 同じtarget平均μ | 0.4 |
| 通常mixture variance | 0.84 |
| 案Eのp_s／variance | 0.76／0.60 |
| 理想Neyman層別の固定N variance係数 | 0.36 |
| ancilla-free grouped observable variance | 0.36 |

direct対照はRY(−atan2(3/5,2/5))後のZ測定で±√[(2/5)²+(3/5)²]を返す実回路の人工Born分岐で確認した。未知state期待値を回路の無料入力にしない。Y座標はこの例でoperatorとして0。層別0.36はσ0=0・σ1=1を知る理想配分式で、有限pilot費や整数配分の保証ではない。direct/groupingと最適層別への一般優越は主張しない。

## 全attempt会計

以下は一つの選択した資源座標の記号的費用。T/CX/RZ/depth・measurement/latency・古典時間を混ぜない。compiled wrapperの価格がprimitive和になるとも仮定しない。

- C_state：fresh入力状態。既存no-prep指標で除外する場合も全attempt消費数を別記する。
- C_pre：H_a、controlled B、必要な共通準備。
- C_pool：PREP、二つの二重制御SELECT、逆PREP、flag測定。
- C_ff：early branch判定・feed-forward。latency等は別座標。
- C_suffix：成功時controlled A、軸変換、signal測定。YはSdg＋H、XはH。X/Yは別々の有料実行。
- C_class：pool確率/metadata処理、古典label生成、記録/cache。量子費と分ける。

固定contextでの記号的実行分解は

\[
\bar C_E=C_{\rm state}+C_{\rm pre}+C_{\rm pool}+C_{\rm ff}
             +p_s C_{\rm suffix},
\]
\[
\bar C_{\rm LCU,late}=C_{\rm state}+C_{\rm pre}+C_{\rm pool}+C_{\rm suffix},
\quad \bar C_{\rm LCU,early}=\bar C_E.
\]

earlyなしとの差は、同じblock価格モデルなら (1−p_s)C_suffix−C_ff。**省略suffixそのもの**は (1−p_s)C_suffixだが、feedback負担まで相殺されるとはしない。古典処理や非加法compile差も別途残る。

通常PRは共通controlled A/B＋wrapperの他、p C_W0^(1)+(1−p)C_W1^(1)を払う。Eでは両Wの二重制御・pool操作を払い、単独Wより高価になり得る。入力状態費はNが変われば相殺されない。将来prefix-only flag校正を使う場合もfresh stateとC_pre＋C_poolを全校正attemptで払う。今回校正campaignなし。

人工dynamic列挙では非可換2-system例のp_s≈0.7286369に対し、E/early LCUのA実行期待回数はp_s、late LCUは1。両方式ともB=1、SELECT_W0=1、SELECT_W1=1、fresh state=1を全attemptで保持した。失敗prefix費を除去せず、未取得bitを成功0bitと取り違えない。

外側contextがランダムなら E[p_s C_suffix]を保持し、E[p_s]E[C_suffix]へ分離しない。successのみの再正規化・retryなし。未知p_sは人工参照解析にだけ現れ、instrumentや費用最適化へ無料で渡さない。

V_raw/2≤V_Eより、同じ資源座標で期待attempt費がrawの2倍以上なら、そのvariance低下だけでcost×variance利益は出ない。有限ε/αの十分shot数証明ではない。range[-1,1]は不変で、HoeffdingだけではN減少は自動ではない。

実H4のpool・PREP精度・一様誤差・native価格・動的hardware対応・十分N・期待総費用は全てUNKNOWN。人工logical block数をcompiled T/CX/RZ/depthと呼ばず、今回実compile/native synthesisは0。
