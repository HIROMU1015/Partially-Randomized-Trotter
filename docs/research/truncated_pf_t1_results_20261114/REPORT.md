# Truncated PF T1：保持Hamiltonianの切断固有値シフト（p=0〜9）

2026-11-14 JST。Claude Code（主agentのみ。独立レビューではない）。ユーザーが承認したT1（[SCIENCE_PLAN](../truncated_pf_optimization_plan_20261114/SCIENCE_PLAN.md) @ `c3ae31f3`）を一回だけ実行した。

**結果の概要**：全10構成で数値gateと状態対応がPASSした。UNKNOWNは0件。

**今回実施しなかったこと**：PF step、H_eff、signal、価格、N、Gはすべて0件。T2以降、新しいcompile、追加prefixにも進んでいない。正本は[run01](run01/)、一覧は[SHIFTS.csv](SHIFTS.csv)。

## 実行条件・SOURCE

| 項目 | 内容 |
|---|---|
| 対象 | 保存H4 DF10（tol-only 1e−8）、weighted-Hermitian completion、sector(2,2)/36次元、元H/state/snapshot |
| 方式・target | 主対象はB0I。共通targetは元のfull DF-H |
| 実行caller | `fd984d8836ec5c82311048347f709353a4d179de:scripts/run_h4_truncated_t1.py`（実行前にcommitして固定）。人工テスト6件PASS（[CALLER_TESTS.txt](CALLER_TESTS.txt)） |
| 再利用した実装 | C numerics `5cc7c07e`（`86f37d25`とbyte一致）の`load_shared_h4`、`Context`、`kernels.matrix_only_tail_coefficients`。extractorが依存する`decompose.py`と`df_rte_tail.py`はcore `e417ffc9`とbyte一致。群の定義はAの評価器`878000d8d`の`_groups` |
| 元Hの固有系 | Aのcache `69ee0ff8`（`reference_comparison`）をSOURCE付きで再利用。元Hの新しいeighは0件 |
| 数値policy | A `9af1ac8e`：残差1e−9、Gram 1e−9、energy cluster 1e−8 |
| 実行環境 | Python 3.12.3、NumPy 1.26.4、SciPy 1.14.1。CのB0I signal実行と同じvenv |
| 処理件数 | context load 1、extractor 10、保持Hの固有系10。所要3.0秒 |
| 実行後の入力照合 | snapshot・receipt・contextの3 SHA256がINDEXと一致（[POSTRUN_INPUT_CHECKS.json](POSTRUN_INPUT_CHECKS.json)） |

元Hについての確認：
- 復元したfull Hと`定数I＋Σ行列`の差は0。
- cacheの残差1.3e−14、Gram欠損1.8e−13。
- 元Hの最低群は単独（多重度1）。保存stateの重みは1。次の群は−1.93376 Ha。
- 比較には元Hの行列最小値`−2.1663874486347643 Ha`を使う。保存solver proxy `−2.1663874486347634 Ha`との差は9e−16 Ha。

## 結果（表示は丸め値。正本はprefix_p.json）

保持Hは`H_ret(p)=c_0 I＋one-body＋fragment 0..p−1＋c_tail(p) I`。
- `δE_B0I=E_g(H_ret)−E_g(H)`。E_gは、元Hの基底群と最大trace重なりを持つ群のエネルギー。
- `δE_B0=δE_B0I−c_tail`は解析的に導いた値（新しい計算はしていない）。
- 比はε_trot=7.968e−4 Haに対するもの。

| p | blocks | δE_B0I (Ha) | δE_B0 (Ha) | \|δE_B0I\|/ε_trot | c_tail (Ha) | λ_R | ‖tail−c_tail I‖（sector） | 最大trace重なり | 保存stateの重み |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | +4.861e−1 | −4.194 | 610 | 4.680 | 13.72 | .9150 | .6926 | .6926 |
| 1 | 2 | −2.462e−2 | −4.423e−1 | 30.9 | .4177 | 2.863 | 1.108 | .9098 | .9098 |
| 2 | 3 | −1.524e−1 | −3.748e−1 | 191 | .2224 | 1.603 | .9482 | .8949 | .8949 |
| 3 | 4 | −7.647e−2 | −1.691e−1 | 96.0 | .09262 | .5833 | .5119 | .98376 | .98376 |
| 4 | 5 | −2.673e−2 | −4.233e−2 | 33.5 | .01560 | .08717 | .03605 | .99984 | .99984 |
| 5 | 6 | −1.701e−2 | −2.593e−2 | 21.4 | .008913 | .05440 | .03084 | .999983 | .999983 |
| 6 | 7 | −1.988e−2 | −2.406e−2 | 24.9 | .004184 | .02208 | .02176 | .999963 | .999963 |
| **7** | 8 | **+8.336e−6** | −3.697e−5 | **.0105** | 4.530e−5 | 3.130e−4 | 2.180e−4 | 1−1.6e−9 | 1−1.6e−9 |
| 8 | 9 | −2.753e−6 | −1.498e−5 | .00346 | 1.223e−5 | 8.327e−5 | 8.311e−5 | 1−7.8e−10 | 1−7.8e−10 |
| 9 | 10 | −2.946e−8 | −8.139e−8 | 3.7e−5 | 5.193e−8 | 1.561e−7 | 1.561e−7 | 1−2e−14 | 1−2e−14 |

**数値検査（全p PASS）**
- 保持HのHermiticity欠損は0。固有残差は1.4e−14以下、Gram欠損は2.5e−13以下、重み和の誤差は6.7e−16以下。
- Weyl：`|δE|≤‖tail−c_tail I‖_sector≤λ_R`が全pで成立。
- 対応：全pで、最大trace重なりの群が保持Hの最小群と一致し、重なりは>0.5（最小でp0の.6926）。2番目の重なりは最大でp0の.1435。保持Hで次の群までのgapは.23〜.39 Ha。

**tail係数**
- 保存済みの6構成（p∈{1,2,3,5,7,9}）は、既存extractorで再計算した値とc_tail・λ_Rとも**bit一致**（相対差0）。
- p∈{0,4,6,8}は既存extractorの値を使用した。
- 実行ではB0I signalと同じく保存値を使った。

## 観察（T1の範囲内）

1. **p7で急に小さくなる。**
   - fragment 7〜9のλ_Rは合計3.1e−4で、fragment 6のλ_R（約0.022）より2桁小さい。
   - p≤6ではδE_B0Iが1.7e−2 Ha以上（ε_Eの11倍以上）。p7〜9では1.05%·ε_trot以下。
   - DF10 tol-onlyの表現は「主要7 fragment＋小さい3 fragment」という構造で、p7が自然な切断点になっている。
2. **λ_Rだけでは評価が非常に保守的。** p7の実シフトはWeyl上界λ_Rの1/37、sector normの1/26。oracleを使わないL1判定では、ε_trotの39%を使う扱いになる。
3. **単調ではない。** |δE|は、p6（1.99e−2）がp5（1.70e−2）より大きく、p1（2.46e−2）がp2（1.52e−1）より小さい。切断誤差はprefix長に対して単調ではない。fragment順はtail重みでは単調だが、基底エネルギーへの寄与では単調でない。
4. **符号。** p7は+（エネルギーを上げる）、p8・p9は−、p1〜p6は−、p0は+。AのFull PF shiftはS2が+、Y4とM5が−である。ただし、p7〜9では`|δE_trunc|≤8.3e−6`なので、L2（絶対値和）とL3（符号付き）の差は高々2|δE_trunc|≤1.7e−5（ε_trotの2.1%）で、相殺の影響は小さい。
5. **B0IとB0。** 全pでB0Iの方がシフトが小さい（p7ではB0が4.4倍大きい）。T2でB0を別に評価する必要はない。
6. **既存記録との一致。** p7のδE_B0I `+8.335886e−06` とsector norm `2.180416e−04` は、先行Track C（`da8b2c3f`）のH4採択Truncated（coeff_L1_2）の記録値（E0(Hkeep)−E0(H) `+8.335886e−06`、norm(Hdrop) `2.180416e−04`）と表示桁で一致した。
   - 先行の記録はDF12・70次元sectorと表記されている。表現ラベルとsectorの違いがなぜ整合するのかは確認していない。
   - 一致は既存記録との照合であり、先行研究の資格を本研究へ移すものではない。

## T2で優先して評価すべき候補（提案。T2は別途承認が必要）

切断誤差だけを理由に、どの候補も除外しない。以下は優先順位である。

| 区分 | 候補 | 根拠 | 提案するcell |
|---|---|---|---|
| **A（主）** | **p7** → p8 → p9。3公式×Aと同じ幅梯子（S2 {h,2h,16h/7,8h/3}、Y4 {4h,16h/3,32h/5}、M5 {8h,16h}＋early cell） | \|δE_trunc\|≤1.05%·ε_trot、対応は明確、相殺の影響は小さい。p7はブロック数8/11で、同じ幅ならstepあたり最大約1.38倍安い（構造仮説）。p8（9/11）とp9（10/11）は差が小さい | 14 cell×3＝42 cell。加えて、p7/S2/hから13roundをspectral再構成し、保存B0I signal（`367dc98d`）と照合する |
| B（任意の診断） | p4〜p6のうち、PFシフトが逆符号（＋）になるS2で、梯子の最も粗い8h/3 | \|δE_trunc\|=1.7〜2.7e−2 Ha（ε_Eの11〜17倍）。相殺するにはPF側に同程度の＋シフトが必要。Full PFのS2では8h/3で+9.5e−4、.8でも+8.75e−3（Aの記録）なので1桁足りない。この見込みは「保持HのPFシフトはFull PFと同程度」という仮説に依存するので、仮説を確かめたい場合だけ実行する | 最大2 cell（例：p6・p5のS2 8h/3） |
| C（T2では推奨しない） | p0〜p3 | \|δE\|=.025〜.49 Ha（ε_Eの15倍以上）。p0〜p2は重なりが.69〜.91で、混合が大きい。梯子内のPF相殺では届かない。不可能だと証明したわけではない | 0。同じprefixのPRとの説明軸として値だけ保持する |

**T2で確かめる仮説（区分A）**
- p7では、L2でもPFに使える予算が`ε_trot−8.3e−6≈7.88e−4`残る。
- 保持HのPFシフトがFull PFと同程度なら、各公式の最も粗い予算内候補はFull PFと同じ（S2 16h/7、Y4 16h/3、M5 16h）になる見込み。
- 保持Hはブロックが少ないので、PFシフトが小さくなる可能性もある。
- M5の16hが予算内なら、32hの境界challenge（T2b）を別承認で検討する。
- これらは仮説で、T2で確かめる。

**T2の件数（提案）**：区分Aで42 cell、保存signalとの照合1組、区分Bで任意に最大2 cell。T2b、価格、全round signal、risk、Gは別承認とする。

## 資格と保全

- 今回の値は、保存DF10 Hに対するspectralな切断診断である。全RPEのRMSE、十分N、Gの認定ではなく、target包含区間や厳密なground同定も行っていない。`rpe_bias/sufficient_N/G`はnull。
- W0のQ1/Q2契約は変更も採択もしていない。
- 固定SOURCE・input・raw・結果・grant・ledger・marker・KEEP、他担当のworktree・process・branch、共有環境は変更していない。
- 実行は一回で、retry、閾値変更、追加prefixはない。

**mandatory STOP**：次はユーザーがT2（推奨は区分Aの42 cell＋照合）の承認可否を判断する。
