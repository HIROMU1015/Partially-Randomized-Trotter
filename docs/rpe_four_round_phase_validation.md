# RPE 4段の物理信号・分岐復元検証

## 目的

限定4段資源集計で未確認だった次の二点を、同じ固定H4条件で検証した。

1. $q=8$の物理基底状態上のfinite-RTE複素信号、信号半径、厳密二項失敗率。
2. $q=1,2,4,8$の測定結果から、直前の推定位相に最も近い分枝を逐次選ぶRPE位相復元。

これは固定した4段のend-to-end検証であり、この成果物自体は目標エネルギー精度から必要round数を
決める検証や最終総コスト最適化ではない。必要round数は後続の
[目標round診断](rpe_target_round_horizon_validation.md)で別途評価した。

## 条件

- H4鎖、原子間距離1.0 Å、STO-3G、8量子ビット、DFランク12。
- 保存Hamiltonian snapshot、$L_D=3$、$delta=0.1$、$r=4$、$K=2$。
- $q=1,2,4,8$。
- $(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.02,0.02,0.36)$ rad、
  $\beta_{\rm RPE}=0.40$ rad。
- 各軸shot数は229、207、186、164。各軸の$\alpha$は限定4段集計で採用した
  cost感度重み配分を固定した。
- 実backend・noise・状態準備は含めない。

## 検証方法

### 物理信号と厳密二項確率

sector内のfull-$H$基底状態に対し、finite-RTEの解析的平均演算子から各$q$の複素Hadamard信号を
計算した。保守的半径下界を使って座標許容幅を求め、採用済みshot数に対するcosine/sine座標
失敗率と統計位相失敗率を二項分布から厳密に計算した。

### fresh IID trajectory監査

4段8軸の全1,572 shotについて、shotごとに異なるseedでRTEイベント列を生成した。各trajectoryを
sector内で基底状態へ直接作用し、その条件付きHadamard平均から測定結果を生成した。この1回分の
明示的4段実験はsamplerと信号規約の監査であり、稀な失敗率の推定には使わない。

### 分枝復元

round $q$で得た主値位相$\theta_q=\arg(\widehat X_q+i\widehat Y_q)$に対し、

$$
\phi_{q,k}=\frac{\theta_q+2\pi k}{q},\qquad k=0,\ldots,q-1
$$

を候補とし、直前roundの推定値に円周上で最も近い候補を選んだ。最終許容差は
$\beta_{\rm RPE}/q_{\max}=0.05$ radとした。

解析信号、明示的fresh-IID測定1回分、解析的周辺Bernoulli分布から生成した10万回の4段実験の
三経路で同じ復元器を確認した。

## 結果

### $q=8$信号

- 観測信号半径：$0.993219824261$。
- 保守的半径下界：$0.993219766664$。
- exact信号に対する系統位相差：$1.06819\times10^{-4}$ rad。
- cosine座標失敗率：$1.11404\times10^{-3}$。
- sine座標失敗率：$2.29891\times10^{-18}$。
- 統計位相失敗率：$2.07129\times10^{-6}$。

各座標失敗率は$q=8$各軸への割当$\alpha=0.0133220$以内だった。系統位相差も
$\beta_{\rm PF}+\beta_{\rm RTE}=0.04$ rad以内だった。

### 4段合成

- 8軸のいずれかが座標許容幅を超える厳密確率：$1.33625\times10^{-3}$。
- 4段のいずれかが統計位相予算を超える厳密確率：$2.14873\times10^{-6}$。
- 解析信号から復元した最終位相誤差：$1.33523\times10^{-5}$ rad。
- 10万回の周辺分布シミュレーション：分枝選択失敗0件、最終位相失敗0件。
- 最終位相失敗率の片側95%上限：$2.99569\times10^{-5}$。
- 10万回中の最大最終位相誤差：$0.0437442$ radで、許容差0.05 rad以内。

明示的fresh-IID監査では1,572 seedに重複はなく、trajectory平均と解析信号の差は最大
1.925標準誤差だった。実際に得た1回分の4段測定結果からの最終位相誤差は
$1.10437\times10^{-3}$ radで、許容差0.05 rad以内だった。この単一成功は補助診断であり、
合否は厳密二項確率と10万回の周辺分布検証に基づく。

## 判定と限界

固定H4条件について、PF/RTE信号、半径、shot数、失敗確率、1 shot cost、4段集計、分枝選択、
最終位相復元を一続きに接続できた。全判定を通過した。

ただし、次は未評価である。

- 目標エネルギー精度から必要round数を決めることは、この成果物の範囲外である。後続診断では
  `CA/10`暫定条件に$q_{\max}=32768$が必要と分かり、固定設定の単純外挿を棄却した。
- $q>8$の物理信号、cost proxy、分枝復元。
- 他の$L_D,\delta,r,K$、分子サイズ、compiler条件への移送。
- 実backend、noise、状態準備。
- 最終全round総コストと決定論PFとの比較。
- clean checkoutまたはCIによるimmutableな再生成。

PF係数が経験的包絡であるため、この結果も`empirical_screening`の範囲であり、厳密な成功保証ではない。

## 成果物

- 結果：`artifacts/rpe_four_round_phase_validation/2026-09-20/`
- 実装：`src/trotterlib/rpe_four_round_phase_validation.py`
- 実行：`scripts/run_rpe_four_round_phase_validation.py`
- テスト：`tests/test_rpe_four_round_phase_validation.py`
