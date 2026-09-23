# WP05-b / WP05-bR full-scope拡張

## 目的

WP05-aで$q=4$まで検証した`support_run_le_1`のcomplete controlled partial-$S_2$／
Hadamard wrapper cost proxyを、独立$q=8$と比較対照$\delta=0.01$へ拡張した。
WP05-bの8 trajectory結果で事前5%基準をわずかに超えた唯一の条件
$(\delta,r,q)=(0.02,32,8)$は、WP05-bRで各$q$ 32本のfresh trajectoryを使って再検証した。

この検証は状態準備なしの1 interrogation costである。backend実行、量子shot、noise、
長$q$直接回路、最終総costは含まない。

## 固定条件

| 項目 | 条件 |
|---|---|
| 物理系 | H4直鎖、1.0 Å、STO-3G、8 qubit |
| DF・候補 | rank 12、$L_D=3$とtail-free $L_D=12$ |
| policy | WP06-bの`support_run_le_1`。production既定値はfull basisのまま |
| RTE | $K=2$、$r=1,2,4,8,16,32$ |
| WP05-b | $\delta=0.02$のfresh $q=8$、$\delta=0.01$の$q=1,2,4,8$、各8 trajectory |
| WP05-bR | $\delta=0.02,r=32$、$q=1,2,4,8$、各32 fresh trajectory |
| wrapper | controlled partial-$S_2$、cosine/sine、ancilla Z測定、状態準備なし |
| compiler | Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなし |

WP05-bは$\delta=0.02$のWP05-a $q=1,2,4$をfingerprint検証後に再利用し、
新たに960本のrandomized measurement-bearing wrapperと10本のdeterministic wrapperを
transpileした。WP05-bRは512本のrandomized wrapperをfresh seedでtranspileした。

## WP05-bの結果

$q=1,2$ affine式を固定し、$q=4,8$を直接holdoutした。

| 条件 | selected RZ | selected全6 metric | full basis RZ |
|---|---:|---:|---:|
| $\delta=0.01$、$q=8$ | 2.650% | 2.804% | 4.321% |
| $\delta=0.02$、$q=8$ | 5.084% | 5.392% | 8.995% |

$\delta=0.01$のselected $q=4$ RZ最大誤差は2.235%だった。
$\delta=0.02$のWP06-b中央RTE additive bridgeを$q=8$へ延ばしたRZ残差は2.031%で通過した。
選択policyの直接RZ変化はfull basis比$-8.607\%$から0%で、直接点で平均RZを悪化させなかった。

5%を超えた箇所は$\delta=0.02,r=32,q=8$だった。selected RZ差は結合標準誤差の
約1.02倍、full basis RZ差は約1.34倍であり、小標本変動と$q$非線形性を区別できなかった。
このためWP05-b単独は`requires_refinement`とした。

## WP05-bRの結果

同じ$q=1,2$ affine式を変更せず、全て新しいseedの32 trajectoryで$r=32$だけを再測定した。

| 判定量 | 最大誤差 |
|---|---:|
| selected $q=8$ RZ | 0.516% |
| selected $q=8$ 全6 metric | 0.537% |
| full basis $q=8$ RZ | 0.829% |
| selected $q=4$ 全6 metric診断 | 1.094% |

全判定が5%以内となり、WP05-bRは通過した。従ってWP05-bの単一点超過は、少なくとも
このfresh 32標本追試では再現しなかった。これを一般的な正規性や全$q$での線形性の証明とは
解釈せず、H4・$q\leq8$・1 compiler条件の運用検証とする。

## 判断

WP05のdecision bridgeは、$\delta=0.01,0.02$、$q\leq8$、$r\leq32$の宣言範囲で完了した。
WP01-D/C07ではWP05-a/bをprovider入力とし、$r=32,\delta=0.02$の$q=1,2$較正だけを
WP05-bRの32標本へ置換する。長$q$は直接回路でなくaffine外挿なので、共有較正誤差と
model discrepancyを全roundへ伝播する。

## 成果物

- WP05-b：
  `artifacts/research_direction_full_scope_extension/2026-09-22/wp05b_q8_delta_0p01_full_scope_extension_v1.json`
  （fingerprint `363ac90a...03bfda`）
- WP05-bR：
  `artifacts/research_direction_full_scope_replication/2026-09-22/wp05br_r32_32trajectory_replication_v1.json`
  （fingerprint `7bfeddac...bb472`）
- runner：
  `scripts/run_research_direction_full_scope_extension.py`、
  `scripts/run_research_direction_full_scope_replication.py`
- tests：
  `tests/test_research_direction_full_scope_extension.py`、
  `tests/test_research_direction_full_scope_replication.py`

いずれもsource・upstream hashを記録したlocal dirty-worktree evidenceであり、immutable CIや
外部再現結果ではない。専用testは合計`4 passed`、変更後のlocal全suiteは
`530 passed, 4 warnings`だった。warningは既存grouped-UWC test由来である。
