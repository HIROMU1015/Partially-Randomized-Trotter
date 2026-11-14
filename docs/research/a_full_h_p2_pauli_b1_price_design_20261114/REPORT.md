# Full-H κ2 P-2の較正設計の修正と、PR-Pauli B1の期待価格検証計画

2026-11-14 JST。担当はA（主agent Claude Code）。

**今回行ったこと**
- 技術設計と人工testだけを行った。
- 新しいsampling、compile、実価格の評価・再解析、fitは0。
- 候補・件数・精度条件は**提案**であり、正式な採択ではない。
- 作業は作者自身によるもので、独立した査読ではない。

**成果物**

| file | 内容 |
|---|---|
| [A1_FULL_H_P2_PLAN.md](A1_FULL_H_P2_PLAN.md) | Full-H P-2 |
| [A2_PAULI_B1_PLAN.md](A2_PAULI_B1_PLAN.md) | PR-Pauli B1 |
| [HANDOFF.md](HANDOFF.md) | D/Cへの引継ぎ |
| [DESIGNS.json](DESIGNS.json)、[BUILD_DESIGNS.py](BUILD_DESIGNS.py) | 候補設計（metadata、exact rank、人工SE） |
| [TESTS.txt](TESTS.txt) | test結果 |

**実装**
- [`src/trottertracks/finite_price_design.py`](../../../src/trottertracks/finite_price_design.py)：取得前の識別可能性検査と、Cのfitへ渡す前の照合。
- test：[`tests/check_finite_price_design.py`](../../../tests/check_finite_price_design.py)。

## A1：Full-H κ2 P-2

**現行案は実行できない**
- 現行案はC/Dの18 compileで、較正点はq=1のgroup cellだけである。model \(E[C]=a+b s+c(q-1)\)の第3列が0になり、rankは2で、cが識別できない。
- 既存の`price_model_validation.fit`もsingularとして拒否する（人工recordで確認）。ただし拒否するのは**価格を取得した後**である。
- **識別の条件**：較正点の\((q-1,\;qR)\)が一直線上に並ばないことが必要十分である。
  - q=1だけの設計、qが1種類の設計、Rが固定の設計は従属する。
  - 一般の共線配置も検出する。
- `check_plan`は、この条件を取得の**前に**検査する。あわせて、領域の混在、較正とholdoutの混在、wiring cohort、law期待値の宣言も拒否する。

**改訂案**（提案。holdoutはm5 n2・m6 n1で共通）

| 案 | 内容 | compile数 | SE(c)（人工） |
|---|---|---:|---:|
| **D7（推奨）** | 現行案に、q8・R566の独立cell（n2）と、R566の2 pathを連結したpaired q2を加える | 24 | 7.8σ |
| D6（最小） | 現行案に、q8・R566の独立cellだけを加える | 22 | 7.6σ |

**改訂案の効果と注意**
- **予測精度**：m12の予測の相対SEは、bが支配的なのでρあたり約0.4%。m6は、63境界分のcの不確かさが効き、約1.3%。
- **paired q2の役割**：境界の加法性を、event由来の雑音なしで直接検査する。groupをまたぐcompiler最適化は、未検証の仮説として残っていた。
- **holdout判定をH-dで行う場合**：m6をn2にする（26 compile）。
- **P-1の扱い**：P-1の単一path価格（m3全round）は、配線の証拠に限る。較正点・E[C]には使わない。

**期待価格の資格に必要な証拠**
- interval contract
- 事前宣言した許容差でのholdout通過
- 境界加法性
- sampler lawの距離
- 低頻度eventの寄与
- 外挿範囲

**仮定の扱い**：group境界・i.i.d.・長時間外挿は、未検証のまま維持する。

**§10.5の照合**
- 予定したcellはすべて、保存記録がないか、条件が異なる（旧Full-Hは旧lowering）。いずれも新規取得候補である。
- P-1は計画外として、配線の証拠としてのみ引用する。

## A2：PR-Pauli B1

**B0の確認**
- LD1 m0、1 path、3 event（order2が1）。
- 辞書は184項（`dbdde5a1…`）、route・loweringはD2、compilerは`1239304c…`。
- 価格はX/YともRZ 1380。
- 位置付けは配線の証拠である。

**既存のPauli価格の照合**

| 証拠 | 判定 |
|---|---|
| B0 | LD1 m0 cohortのpath0として、事前に宣言すれば部分的に再利用できる |
| P3（案A residual、K4、T0.8） | 再利用不可 |
| Track AのPauli-JW（全JW H、Pauli head） | 再利用不可 |
| PR DF-tailのE[S]・C1・P2 | 再利用不可 |

**価格に影響する特徴量**
- 回転の重み（CXのladder）、X/Yの基底変換、target配置
- order-2のproduct（制御単一qubit Pauli）
- stepごとのancilla位相
- **隣接eventの組の相殺**：opt1による。価格はeventのi.i.d.和ではない。
- DF headの項（stepごと）

**既存B1案（各2 path、8 compile）で分かること**
- 分かる：LD2の配線（初回）、m0での2 path間の差、B0の再現性。
- 分からない：law平均・CI・傾き・head項・外挿。設計のrankは1である。
- **2 pathで期待価格の精度は保証されない。**

**最小の有限検証の候補**（提案）

| 段階 | 内容 | compile数 |
|---|---|---:|
| B1-W | LD2の配線 | 2 |
| B1-V（任意） | m0の追加path。B0を再利用するとCの8から6に減る | 6 |
| B1-CAL | 辞書ごとに、1 step cell（R_m6/m9/m12）、8 step cell、paired 2 step、holdout m5/m6 | 辞書あたり24 |

B1-CALでは、\(a+h q+b s\)がrank 3で識別できる。

**期待価格modelの候補**
- M-lin（上の線形model）
- M-struct：Pauli構造kernel＋残差
- M-feat：labelのclass別特徴量

未資格の仮定は[A2](A2_PAULI_B1_PLAN.md) §8に列挙した。

## 必要な次の科学実行範囲（いずれも未承認）

| 対象 | 範囲（提案） | 前提 | 期待される研究上の効果 |
|---|---|---|---|
| Full-H P-2 | D7（24 compile、H-dを使う場合は26）。段階的に取得する | Dのcaller拡張（cell指定・seed連結）、許容差（A8）の宣言 | 全13 roundのE[C] modelが初めて識別できる。group分解の外挿を支える境界加法性を直接検査できる。m10〜m12（D KAPPA_CONNECTIONのmodel算術ではG_paperの約97%）の価格scenarioの根拠を確かめられる |
| Pauli B1-W | LD2 m0、2 compile | Dのprefix引数化 | LD2の価格経路を開く |
| Pauli B1-CAL | 辞書あたり24 compile（LD1を先に） | Dのcell・seed拡張、許容差 | Pauli tailのPRの全roundの価格model。DF tailやFull-Hと、同じ枠組みで比較できる |

**未解決の条件**
- 比較policy（A9）
- holdoutの許容差（A8）
- interval contract
- sampler lawの距離
- 大きいcell（s≈36k）のcompile資源

## 保全とSTOP

- 既存のSOURCE・raw・結果（P-1、B0、S-1、C/Dの計画）は変更していない。
- B・C・Dのbranch・worktree・processには触れていない。
- 新branch `codex/a-full-h-p2-pauli-b1-price-design-20261114`は、P-1再検証の`a04470c0a`から作成した。
- **mandatory STOP**：P-2、B1、追加sampling、実compile、価格fit、N/Gへは進まない。
