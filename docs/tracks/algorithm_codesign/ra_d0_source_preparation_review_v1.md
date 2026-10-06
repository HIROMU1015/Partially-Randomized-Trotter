# RA-D0 static table / LP certificate準備・source review v1

2026-10-06 JST。利用者の[formal design原文](inputs/ra_d0_formal_design_v1_user_input.md)を受領し、
§16の準備だけを実施した。基点は数学監査
`8a04c148a66d23dbc1f045086a95a5e19a6372dc`、独立branchは
`track-b-ra-d0-source-preparation-20261006`。

**判定：`REVISE_RA_D0_NUMERICAL_BASELINE_AND_EXECUTION_CONTRACT`。**
登録table最適化は未実行、`RUN_READY=false`、development authorizationなし。
原文のRQ・候補・精度・accuracy・strict witness・科学分類を変更していない。
実装済みkernelと静的資料を公開し、数値baseline規約／query実行契約の採否をGPTへ戻す。

## 1. 固定入力と境界

| 入力 | 固定commit / identity | 今回の扱い |
|---|---|---|
| R1結果 | `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b`、result SHA256 `f726ad70cb2643533f0d037b518cde1b702724adb4e6571fea25e26e4bfdd61e` | 保存eventのみの抽出、分類は不変 |
| R1.5 attribution | `af3d014d0a0cfcbbd25bb544f6544652fec92942` | known / post-hoc設計入力 |
| 数学監査 | `8a04c148a66d23dbc1f045086a95a5e19a6372dc` | fixed-n canonical LPとcertificateの根拠 |
| GPT formal design | [byte-exact原文](inputs/ra_d0_formal_design_v1_user_input.md) | 未修正、入力hashはmanifest参照 |

対象は2-qubit distinct-basis controlled、finite P3、p=(3/4,1/4)、x={1/8,1/4}。
sigma=+1が設計側、-1は同一保存tableの符号control。
分子／DF／geometry／held-out／blind replicationは含まない。
旧R1 source・authorization・result・one-shot marker、R1.5、数学監査、Track Aを変更していない。

## 2. 静的candidate監査

[candidate table](../../../artifacts/track_b_ra_d0_preparation/2026-10-06/candidate_table_v1.json)は
各xにO0/O2/P2/P3/A0/A1/A2×既存三precisionの21 columns。
新angle、合成、native lowering、circuit buildを行わず、保存a/b、IID law、IR hash、cost、strict errorから抽出した。

- 全18 sign-pairでevent別T/CX/1Q/strict errorとworkspaceが完全一致。
  符号でIR/phaseそのものが一致するとは仮定しない。
- ordinaryとPTSC-K0のO0はD、conditional law、IR、cost、error、phaseが一致し、共有aliasとした。
  A0/A2はdegreeが異なるので同angleでも統合しない。
- A1は保存odd/complement記録を保持。新大角度合成を生成しない。
- d_j=2 sum pi delta_event。旧coefficient rounding biasはcolumn dへ転記しない。
- workspaceは保存controlled contextのuniform peak=1。今回除外0。
- 保存126列のsequence hash、T+TdaggerとTdagger subset、error-pass flagを照合。
  strict operator errorの再評価は行わない。

native費用は既存R1の加法的primitive T/CX/1Qであり、whole-circuit optimumではない。

## 3. 入れ子関係：idealと数値lawを分ける必要

[semantic audit](../../../artifacts/track_b_ra_d0_preparation/2026-10-06/semantic_audit_v1.json)で、
全18 profilesのideal degree contributionsの和がt=(1,x,x²/2,x³/6)にexact rationalで一致した。
ideal real arithmeticでは次が成立する。

1. B0_ideal→B1：各groupの全weightを元precisionへ置く。
2. B1→B2：一representationのtheta=1とする。
3. B2→B3：sum z=y、group q=z wからDq=yt。共有O0だけを合算する。

B2は係数段階のwhole-representation混合を再canonicalizeしたもの。
実際のrepresentation選択確率はtheta_r B_r / sum(theta B)である。
thetaのまま旧samplerを選んで異なるB_rを出力する非canonical lawとは区別する。

ただし原文の**保存B0そのもの／ideal B1／全出力qのdyadic丸め**を同時に文字どおりの集合包含とすることはできない。
保存B0はgroup normのinterval midpointを使う。非平方のradicandについてmidpoint²≠a²+b²であり、
ideal weightそのものではない。これは旧R1を無効にする指摘ではない。R1は既存rounding biasを含めて認証済みである。

さらにordinaryの二groupのnorm比の二乗は両xで非平方rational。
norm比がirrationalなので、B1のexact group mass ratioを保ちつつ、全qを2^60分母rationalにすることは不可能。
mean residual xiだけでは**representation membership residual**の許可範囲は定義されない。

**必要な最小修正案（未採用）：**

- `B0_saved`はimmutable reproduction／anchorとして保持し、数学包含の始点を`B0_ideal`と明記する。
- B1/B2/B3の共通numerical mean条件は現delta_numを維持する。
- B1/B2の実装lawについて、固定largest-remainder丸めから導けるmembership許容幅を結果前に明文化する。
  例えば各三precision groupのq丸め誤差≤3/2^60、y丸め誤差≤1/(2·2^60)なので、
  B1ではgroup差≤(3+w_g/2)/2^60という取得値に依存しない上界を構成できる。
  B2ではz_rの表現規約も含め同様に上界を導き、全baselineに対称適用する。
- B2 objective lowerは、このnumerical baseline全体を含む**外側集合**から取る。
  B3のdyadic primal upperがそのlowerより小さい場合だけstrict witnessとする。

今回のB2 compilerはideal membership intervalsのouter relaxationまで実装した。
quantized B2 samplerのmembership certificateは未採用規約のため閉じていない。
literal B0_saved⊂B1および数値pipeline全体をPASSとは報告しない。

## 4. LP／certificate実装とsource確認

[B専用kernel](../../../src/trottertracks/algorithm_codesign/ra_d0/)は既存共通APIと分離した。

- 100桁Decimalのcorrectly-rounded lnと隣接値、integer isqrtによる100-decimal-place sqrt enclosure。
  ell、kappaの上界をFractionへ戻す。float epsilonでcertificateを決めない。
- B3 inner LPはD upper/lower両側のworst residualをr_kで押さえ、sum r≤y delta_numとconfidenceへ戻す。
- B1は三representationそれぞれを固定するouter LP、B2はそれらのwhole mixture outer LP。
  両方のinterval membership／meanを外側へ緩める。
  outer optimum lowerはideal B2へのsafe lowerであり、outer解をB2実装lawと呼ばない。
- min c*x, A*x≤b, H*x=f, x≥0のdualをexact rationalで確認する。
  近似dualのstationarity負残差は、証明済み変数upper boundを掛けてlowerから控除する。
- infeasibilityはexact Farkas certificateを用意した。
  solverのstatus=2だけではB2-only infeasibility witnessにしない。
  正規化separation LPによる証明書取得も実装し、返ったrayをexact rationalで再確認する。
- largest remainderは固定2^60、tieはindex順。yはnearest / half-up。
  丸め後のlawだけをmean、bias、confidence、同query capsへ再認証する。
  負のnominal qをclipせず棄却し、denominatorを増やさない。
- registered-domain solver呼出しはsourceで明示拒否する。
  最適化runner、authorization、one-shot markerは作成していない。

有限domainにはy≤2/(sum t−delta_num)というrational upperを使った。
sum Dq≤sqrt(2)<2と||Dq−yt||_1≤y delta_numから導き、dual誤差補正に使用する。
一次資料は[SciPy linprog公式仕様](https://docs.scipy.org/doc/scipy-1.16.2/reference/optimize.linprog-highs.html)。
solver flagは候補取得にのみ使い、科学的可否はcertificateへ戻す。

技術候補backendは隔離venvのPython 3.12.3、SciPy 1.16.2、NumPy 2.5.3。
[35 focused tests](../../../tests/tracks/algorithm_codesign/test_ra_d0_preparation.py)がPASS。
synthetic LPは人工2変数問題と人工infeasible問題の証明書取得だけ。実tableのLPはschema構築だけでsolve=0。
[検証記録](../../../artifacts/track_b_ra_d0_preparation/2026-10-06/focused_verification_v3.json)にidentityを保存した。
これはimmutable CI、外部再現、general proof、RA-D0 resultではない。

## 5. Grid／queryの結果前recipe

[grid artifact](../../../artifacts/track_b_ra_d0_preparation/2026-10-06/shot_grid_query_recipe_v1.json)。

| x | n_min | n_max | primary anchors | grid全点 |
|---|---:|---:|---:|---:|
| 1/8 | 477,822 | 3,247,176 | 9 | 395 |
| 1/4 | 613,085 | 3,200,447 | 9 | 342 |

lower boundにはell下界とh_max上界を使う。数値mean許容も戻し、
y≤sqrt(2)/(sum t−delta_num)とする。この修正は今回の整数n_minを変えない。
全cost minimumがpositive（T=28.5、CX=2.25、1Q+5/2=76.625）なので、
原文の全座標Pareto upper boundを構成できた。
**一般にzero-cost座標を省略して全3座標dominanceとは言えない**ため、実装はその場合reviewへ拒否する。
今回は省略を使わず、registered domainを変えていない。

整数shotsの切上げ倍率は次点/(直前点+1)の最大を厳密に計算した。
両xで1.005以下。nominal adjacent-grid比との差を保存している。
これはcore feasible setの倍率coverageであり、各epsilon-constraint budgetのfeasibility保存や
有限query集合で全3D frontを解いた保証ではない。

query recipeは原文どおりB0同n資源＋B2 single-resource minimaからのみbudgetを構成する。
**B2 minima、budget実値、B3 point、witnessは未取得。**
それらは最適化を要するため、result-prior artifactに生成規則とplaceholderだけを固定した。

実行前に次を確定する必要がある。

1. B2 minimumからbudgetを固定する方法：丸め／membershipを戻したcertified feasible upperを使う案。
   最小値のnominal floatをcapへ直接使わない。B3実行より前に全budgetをfreezeする。
2. B2 infeasibleのcertificateが得られない場合はtechnical inconclusiveとする規約。
   B3-only feasibilityはdescriptiveとして保存し、§11のprimary strict objective flagと混ぜない。
3. 全queryが完了しても`NO_REGISTERED_WITNESS`は「certified strict witnessが得られない」の意味に限定する。
   lower/upperが重なるだけで真の改善不存在を証明したとは言わない。

## 6. 実行量gate

各coordinateのbudget setは最大9 B0＋1 B2 minimum=10値。
737 points×3 objectives×10²で、最大221,100 paired queries。
二classとB2三minimaを含む最大LP呼出しは
**737×(600+3)=444,411**。B0 feasibilityの事務処理はsolver不要である。
  この数はprimary/minimum LPだけ。各infeasible問題に一つのFarkas補助LPを認める方式では、
  保守的な合計上限は888,822呼出しになる。補助LPは別途固定した証明書取得であり、元queryのretryではない。
これは結果前の機械的上限であり、実際のsolver呼出し数を測った値ではない。

原文は最適化のwall/CPU/RSS／output／per-call上限をまだ固定していない。
全coverage recipeの採用、query縮小の必要性、計算資源上限はGPT側で決める。
Codexは無断でanchor-onlyへ縮小したり、処理できたprefixだけからnegative分類を出さない。
今回のbackendはsynthetic用だけで、future development runtime identity／resource guard／source-bound
authorization-only childとexclusive markerは別の実行contractで閉じる。

## 7. 先行研究の境界

原文のMorisaki–Fujiiを一次資料で特定した。
[Optimized Randomized Hamiltonian Simulation via Average-Error Analysis, arXiv:2609.36694v1](https://arxiv.org/html/2609.36694v1)、
2026-09-29、Importance-sampled qDRIFT／Average-error-optimized sampling、Eqs.(2)–(3),(16)–(20)。
任意term probabilities、varianceによるleading channel error、Haar平均基準のHilbert–Schmidt samplingを扱う。
この範囲をRA-D0の新規性とはしない。

[Resource-Optimal Importance Sampling](https://arxiv.org/abs/2603.13495)、
[Structure-Aware Variance Reduction](https://arxiv.org/abs/2606.23544)との境界は前回数学監査を継承する。
degree-local finite-mean representation／phase-preserving finite precision／finite-confidenceの組合せの
publication priorityや新規性は今回確定していない。世界初／resource優位の新claimはない。

## 8. 停止

今回の準備をcommit/pushしてGPTへ返す。
数値baselineとquery実行契約の修正採用→source freeze／review→別authorizationの前に
登録RA-D0 optimizationを実行しない。
全old STOP、R1 science classification、one-shot制限は保持する。
**mandatory STOP。研究方針・RQ・必要な追加検証の範囲はGPT判断。**
