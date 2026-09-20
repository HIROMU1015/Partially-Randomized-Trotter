# RPE $\delta$候補の中央RTEブロック・コンパイル後コスト検証

## 1. 目的

[`rpe_delta_round_schedule_validation.md`](rpe_delta_round_schedule_validation.md)では、H4の
暫定`CA/10`条件で$\delta=0.01,0.0125,0.02$のround別$(r_m,K_m)$ scheduleを構成した。
成分作用数だけの暫定proxyでは$\delta=0.01$と0.02の差が約0.87%だったため、本検証では
既存の境界補正型回路コストモデルを実際の短時間幅とイベント列長へ拡張し、各scheduleの
**中央$\widetilde U_{\rm RTE}$ブロック**のコンパイル後コストを比較する。

これは状態準備なしHadamard interrogation全体や最終総コストの評価ではない。決定論DF half sweep、
決定論部とRTE部の境界、制御化の追加コスト、補助量子ビットのHadamard・測定軸変更・測定は含めない。

## 2. 固定条件

- H4 chain、原子間距離1.0 Å、STO-3G、8 qubit
- DF rank 12、$L_D=3$、固定した同一Hamiltonian snapshot
- Qiskit 1.3.0、basis gates `rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなし
- $K_m\in\{0,2\}$、選択schedule上の$r_m\in\{1,2,4,8,16,32\}$
- 短時間幅$\delta/r$は0.02から0.000390625まで
- 主指標RZ数、補助指標RZ深さ、CX数・深さ、全体深さ、回路サイズ

## 3. 推定方法

長さ$r$のRTEイベント列について、1--3イベントの局所係数を用いて

$$
\widehat C_r^{(3)}
=\sum_i\kappa_1(e_i)
+\sum_i\kappa_2(e_i,e_{i+1})
+\sum_i\kappa_3(e_i,e_{i+1},e_{i+2})
$$

と推定する。各roundでは有限Taylor分布から次数0・2の解析確率を求め、局所パターンの期待出現回数で
係数を重み付けする。$r=32$では後述の長さ検証で1--3イベント式が不通過だったため、4イベント係数を
追加する。4イベント係数は、全次数0と次数2が1か所の5パターンを較正した。次数2が2か所以上の
4イベント窓は解析確率で監査し、本scheduleでの最大期待窓数が$7.64\times10^{-13}$以下だったため、
数値上無視した。

round $m$の中央RTEブロック代理値は、1回のRTE occurrenceの期待コストを$q_m$倍し、さらに
暫定の両軸shot数を掛ける。partial-$S_2$反復を一体コンパイルした値ではないため、反復境界の追加相殺は
この段階では含まない。

## 4. 短時間幅をまたぐ係数再利用の検査

局所イベント列を同一に保ち、回転角だけを12個の短時間幅へ変えて個別にコンパイルした。
構造キャッシュによる同一視は無効にし、実際に得た6指標を比較した。

| 局所長 | trajectory数 | metric比較数 | same/different境界 | 最大差 | 判定 |
|---:|---:|---:|---:|---:|:---:|
| 1--3イベント | 280 | 3,080 | 295 / 105 | 全6指標で0 | 通過 |
| 4イベント | 160 | 1,760 | 355 / 125 | 全6指標で0 | 通過 |

したがって、このcompiler条件とサンプリング範囲では、既存の短時間幅0.02の局所係数を
schedule中の短時間幅へ再利用できる。これは別compiler、coupling map、別Hamiltonianへの不変性を
意味しない。

## 5. イベント列長の検証

短時間幅0.02の固定較正係数を、係数較正に使っていない一体回路へ適用した。

| イベント列長 | 使用モデル | 主RZ最大点誤差 | 全6指標最大点誤差 | RZ予測側95%半幅 | 判定 |
|---:|---|---:|---:|---:|:---:|
| $L=4,6$ | 1--3イベント | 2.140%以下 | 2.140%以下 | 2%以下 | 通過 |
| $L=8$ | 1--3イベント | 0.945% | 2.319% | 1.856% | 通過 |
| $L=16$ | 1--3イベント | 3.195%以下 | 3.556%以下 | 2.377%以下 | 点誤差通過 |
| $L=32$ | 1--3イベント | 7.966% | 7.966% | 2.701% | 不通過 |
| $L=32$ | 1--4イベント | 4.175% | 4.175% | 係数較正の不確かさは残る | 点誤差通過 |

$L=8,16,32$の全次数0条件は各長さ300標本である。次数2が1か所の診断は各位置2標本に限られ、
その95%上側診断は精密ではない。$L=32$の1--3イベント式は全次数0のRZ誤差7.966%、
$z=3.151$で外れたため、低確率次数2の標本不足だけでは説明できない。4イベント較正は各パターン
100標本で、$L=32$の全次数0 RZ誤差を4.175%へ下げた。ただし独立係数評価全体の95%診断は
5%を超えるため、4イベント式を厳密な5%保証とは扱わない。

この結果から、本scheduleでは$r\leq16$に1--3イベント式、$r=32$に条件付き4イベント式を使う。

## 6. round scheduleへの接続結果

各roundの解析Taylor確率、$q_m$、両軸shot数を用い、中央RTEブロックだけを全roundで合計した。

| $\delta$ | round数 | $q_{\max}$ | 最大$r_m$ | RZ数代理値 | 最小値からの増加 | RZ較正95%半幅 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 18 | 131072 | 32 | $7.9877\times10^{11}$ | 0% | 2.933% |
| 0.01 | 19 | 262144 | 16 | $9.2021\times10^{11}$ | 15.203% | 2.117% |
| 0.0125 | 19 | 262144 | 32 | $1.2626\times10^{12}$ | 58.073% | 2.932% |

RZ数、RZ深さ、CX数・深さ、全体深さ、回路サイズの全6指標で$\delta=0.02$が最小だった。
5%の暫定モデル許容差を考慮しても、中央RTEブロック代理値では0.02だけが最小候補に残る。
成分作用数proxyで近接していた0.01との差は、round数、$q_{\max}$、イベント境界相殺を含む
コンパイル後局所モデルへ置き換えると15.2%へ広がった。

## 7. 結論と次の検証

このH4固定snapshot・compiler条件では、短時間幅の変更自体による局所コンパイル指標の変化は
観測されず、イベント列長32にだけ4イベント補正が必要だった。中央RTEブロックの比較では
$\delta=0.02$を次の優先候補、0.01を比較対照、0.0125をPF独立grid由来の感度候補とする。

ただし、現時点で$\delta=0.02$を最終採用とはしない。次は0.02と0.01について、選択された
round別$(r_m,K_m)$を使う**制御付きpartial-$S_2$反復回路のコスト代理**を検証し、決定論部分、
RTEとの外側境界、制御化、Hadamard wrapperを含む1 shotコストへ接続する。大きい$q$の一体回路を
直接コンパイルした結果、状態準備・noise・実backend、または最終総コストはまだ得ていない。

## 8. 機械可読な結果

- `artifacts/rpe_delta_compiled_cost_validation/2026-09-20/h4_ld3_shortstep_0p02_to_0p000390625_angle_invariance_n20_v1.json`
- `artifacts/rpe_delta_compiled_cost_validation/2026-09-20/h4_ld3_shortstep_0p02_to_0p000390625_k4_angle_invariance_n10_v1.json`
- `artifacts/rpe_delta_compiled_cost_validation/2026-09-20/h4_ld3_s0020_transfer_l8_zero300_rare2_v1.json`
- `artifacts/rpe_delta_compiled_cost_validation/2026-09-20/h4_ld3_s0020_transfer_l16_l32_zero300_rare2_v1.json`
- `artifacts/rpe_delta_compiled_cost_validation/2026-09-20/h4_ld3_s0020_k4_n100_l16_l32_validation_v1.json`
- `artifacts/rpe_delta_compiled_cost_validation/2026-09-20/h4_ld3_delta_0p01_0p0125_0p02_round_schedule_rte_block_cost_v1.json`

これらはdirty worktree上のlocal evidenceであり、immutable CIまたは外部再現結果ではない。
変更後のlocal全テストは`494 passed, 4 warnings`で、warningは既存grouped-UWCテストに由来する。
