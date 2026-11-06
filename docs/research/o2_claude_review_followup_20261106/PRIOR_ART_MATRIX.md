# MC2に関する一次文献対応

2026-11-06 JST。限定的な文献調査。採否・独立新規性の正式判断はユーザーへ戻す。以下のsection・式・定理を実際に確認した。全文の全証明や関連文献網の網羅的読了は主張しない。

## 最も重要な対応

**Güntherらv2の付録A.3に、signed高次PFへRTEを挿入し、絶対時間に比例配分する直接の先行記載がある。** Claudeの調査対象D/Eだけではこの記載を見落とす。二乗構造が本研究で初めて出たという説明は、この一次資料と両立しない。これは重複する構造の特定であり、MC2全体や実証の価値についての正式判定ではない。

同論文の記号を本調査へ対応させると、同論文のouter反復s→本研究q、stage時間δ_ip→w_iδ、総randomized rotation/event数r→本研究n_ev、絶対時間δ̃→δΣ|w_i|。式(A38)の指数はq λ_R² Σ_i(w_iδ)²/r_i。式(A39)はceil付き絶対時間比例配分を選び、式(A40)はλ_R² δ̃² q²と整数丸め項を持つ。従って(Σ|w_i|)²に相当する依存が直接含まれる。

同論文は無限RTEのnormalization**上界**と定数shot overheadに必要な資源を評価する。本研究の式は固定finite Kの正確なβ_Kを小τ展開した**近似**である。この違いを残す。有限KでもK2以上のleading係数が同じことは、SOURCE定義からの対応であり、二乗構造を新原理へ変えるものではない。

## 文献別の比較

| 一次資料・固定版 | 確認した箇所 | MC2との区分 | 重複／差分 |
|---|---|---|---|
| Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647v2（2025初出、2026改訂） | App. A.2 (A18)–(A29)、A.3 (A32)–(A41), Lemma A.3; App. D, Fig.16; App. E.3 (E19)–(E23); §VII.B | `DIRECT_PRIOR_ART` | signed PF×RTE、絶対時間・比例配分・二乗資源が直接先行。有限K、実装整数配分、DFの特定candidate/cohortでのcompiled費用検証が差分候補 |
| Hagan & Wiebe, *Composite Quantum Simulations*, arXiv:2206.06409v3（2022初出、2023改訂） | §5, Def.7 (57)–(58), Lemma8 (59)–(63), (64)–(65), §5.2–5.3 | `CLOSELY_RELATED` | 高次outer compositeとtail randomizationが先行。qDRIFT channel誤差・query上界であり、finite-RTEのβ_Kと測定込みcompiled Gは同一ではない |
| Pocrnic et al., *Composite QDrift-Product Formulas for Quantum and Classical Simulations in Real and Imaginary Time*, arXiv:2306.16572v1（2023） | §II.4–5 (11)–(16); §V.2–V.4, Props. V.1–V.4, Figs.4–6; App.Bの誤差指標 | `PARTIALLY_OVERLAPPING` | inner/outer次数と分割の同時選択、数値cost競合が先行。有限RTE normalizationやHadamard shot費用の実証とは異なる |
| Kanasugi et al., *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*, arXiv:2603.22778v1（2026） | §II.2–II.2.4 (13)ほか; App.A; App.B (52)–(56); §IIIの表現 | `CLOSELY_RELATED` | 分割・randomized資源・測定overhead・RPE資源最適化が先行。任意のsigned係数列に対するfinite-Kの実装検証とは区別 |

### Günther：アルゴリズムと資源の意味

deterministic H_lとinvolutory tail H_Rを分離。App.A.2のpaired RTEは無限次数でunbiasedなHadamard信号を作り、B²がsampling overhead。A.3は負時間を明記し、B≤exp(λ_R² Σδ_i²/r_i×outer反復)を合成、比例配分とceilを扱う。Dの高次比較は化学benchmarkの有限精度で一貫した改善がないとの数値判断で、任意の高次PRの不利を証明していない。E.3は主にS2を採り、Toffoli等の価格とshotを含める。DF適用自体も既知である。残る差分候補は、保存m5の有限K・整数配分と強い対照を同じfull-wrapper compilerで扱う条件付き実証。一般的新規性の確定はできない。[固定v2 PDF](https://arxiv.org/pdf/2503.05647v2)

### Hagan–Wiebe：Υによる上界との違い

Def.7はSuzuki再帰でA/B channelを合成し、負stageを含む。Lemma8はchannel誤差をΥで合成し、再帰の各時間係数の絶対値を抑えて上界化する。qDRIFT誤差はλ_B²t²/N_B型。これはsigned compositeにrandomization費用が加わる先行理論だが、読んだ式にはΣ|w_i|を保持したfinite-RTE β_Kの等式はない。ここから精緻な配分式を導くことと、文献が明記した式を区別する。同論文の5段再帰の係数評価を、本研究の3段Yoshidaの|w|>1へ無条件に移せない。価格はexponential query数／diamond誤差の上界で、実compiled gateや測定shot込みではない。[固定v3 §5](https://arxiv.org/html/2206.06409v3#S5)

### Pocrnic：次数と分割の相互作用は既知

Trotter側AとqDRIFT側Bを組み合わせ、inner 2kとouter 2lを独立に扱う。§Vはtrace distanceを用いた全channelの数値評価で、分割とN_Bを最適化し、時間・次数で選好が変わる。infidelityとtrace distanceの違いも明示する。従って「高次と分割を同時に考える」という広いMC2は先行と重なる。費用はe^(−iH_jt)等の回数であり、finite KのRTE信号減衰・cos/sin headroom・量子shotを掛けた実装RZ価格ではない。調査箇所では二乗絶対時間によるfinite-RTEの定量式は確認していない。未確認を非存在証明にしない。[固定v1 §II.5/§V](https://arxiv.org/html/2306.16572v1)

### Kanasugi：RPEと指標の先行

Pauli LCUを主に用い、qDRIFTの固有値依存減衰とRTE等のnormalizationを区別。App.B (54)–(56)はdet/random costを分け、N_stage、λ_R²δ²2^(2m)、protocol係数γを使う。RPEのerror配分・測定数・実装アーキテクチャの資源へ接続する。高次pの表記があるだけで、任意のsigned w_iの絶対時間因子を省略できるとは読まない。本研究のQiskit全RZと、同論文のSTAR上のrotation・物理資源を換算しない。DFのGivens角度を理由にPauliを選ぶ記述は、表現／実装の条件付き性にも関連する。[固定v1 §IIとApp.B](https://arxiv.org/html/2603.22778v1)

## 補助的な2025–2026年の確認

- Casares et al., arXiv:2606.30741v1, §III.5とApp.F.1：fragment順序のrandomization、near-integrability、symmetry protectionと高次PFを扱う。tailをpaired-RTEで置換するlawとは異なる。`PARTIALLY_OVERLAPPING`。今回の保存m5とその最適化係数列を同定する根拠にはしない。[固定v1](https://arxiv.org/html/2606.30741v1)
- Peetz & Narang, arXiv:2501.13922v1, main (10)–(13)、supplementのrandom-unitary approximation：高次Zassenhaus補正のsamplingであり、固定signed-S2 stageのtail normalizationではない。`DISTINCT_WITH_EVIDENCE`は確認した構成についてのみ。[固定v1](https://arxiv.org/html/2501.13922v1)

検索語はhigh-/higher-order randomized Hamiltonian simulation、RTE normalization、negative-time composite、Casares product formulasを用い、一次資料へ戻った。一般的な検索結果や二次レビューを主張の根拠にしていない。全2025–2026年文献の網羅性は保証しない。Casares/SZEは指定4文献の調査を置き換えるものではない。

## ユーザーへ戻す新規性の判断単位

| MC2の要素 | 今回の証拠に基づく扱い |
|---|---|
| 高次deterministic PF＋randomized tailという構成 | 既知のcomposite構造 |
| signed時間、絶対時間比例配分、絶対時間の二乗 | Günther A.3に直接先行あり |
| PF次数と分割・時間の競合 | composite理論／数値研究と重複 |
| finite K、最低1 step付き整数配分とSOURCEの対応 | 本調査で技術的に整理。単独の新理論とは未認定 |
| DF10/19、S2/Y4/保存m5、B2/B3、強いPF対照、測定付きcompiled価格での具体的観測 | 実証差分候補。B1/B2の確認後に科学的意義と論文の位置付けを判断 |
| 二乗則から最良Gの変化を一意に予測すること | 未検証。先行構造の存在だけでも、SOURCE整合だけでも成立しない |

「GüntherがS2をbenchmarkに使う」ことと「高次×RTEの理論を扱わない」ことは異なる。案XやMC2の正式採否、中心新規性を何に置くかは本追補では採択しない。
