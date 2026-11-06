# Track C2-B：保存2精度候補による強baseline限定検証

**数値上の構造的配分の利益は残ったが、判定は「最適性未確定」。**
旧貪欲policy差0.547022%に対し、通常配分を整数計画で強化すると旧構造的配分の差は0.547020%。
両armを同等に強化し、同じalpha/cap選択機会を与えた実行可能解では差は**0.514072%**。
ただし通常armの厳密な総T下界が構造armの実現値を下回るため、さらに良い通常配分・alpha選択で差が消える可能性を排除していない。
科学実行を完了しmandatory STOP。候補・PF・fusionの拡張は行わない。

基準commit：`78004943ed5b718c12841e80f55fddf1d1644364`。
branch：`codex/track-c2-b-strong-baseline-20261106`。
SERVER 2 `GPUA100-gleap02`、2026-11-06 JST。ユーザーの限定数値実行承認後に実施。
実行範囲は[SCOPE.md](SCOPE.md)。旧pilotのSOURCE・raw・結果・STOPは保全した。

## 固定条件・保存入力

Linear H4、1.00 Å、STO-3G、DF12。元Hの厳密groundを供給し、状態準備費を除外。
Full Y4、LD12、h=.02、18round、delta0=.02、energy精度1.5936e-4 Ha、alpha総和.05、位相予算.4 rad、compiler A。
旧固定保存行列モデルと元の積分Hamiltonianとの無条件な包含の未資格は維持する。
Hamiltonian、PF schedule、測定方式、actual adjoint、controlled相対phase、fusion、誤差規約は変更していない。

基準commit内の次の保存値だけを用いた。

- `artifacts/track_c2_b_pilot/2026-11-06/raw_synthesis.tar.xz`：7,603 strict record。
- `PRECISION_REQUESTS.json.xz`：72条件＝18round×2 phase cap×2誤差方式。
- `verified_final/RESULTS.json.xz`：採用された補正後の配分・rho/beta/alpha・整数shot・総T。

bundle hash、全recordのword hash・T count・strict上界のrequested epsilon包含、全72条件の必要keyを確認した。
欠損・代替推定・0補完はない。[INPUT_CHECK.json](../../../artifacts/track_c2_b_strong_baseline/2026-11-06/INPUT_CHECK.json)。
初回補正前結果と`verified/`中間結果は採用していない。

## 再利用と今回の変更

[新module](../../../src/trottertracks/track_c2_b_strong_baseline.py)は既存
`track_c2_b_pilot.exponent/allocate/phase_disk_upper/shots`と
`track_c2_b_verify.allocation_lp_lower`を呼ぶ。元moduleのrun/合成/回路関数は呼ばない。
既存RA-D0 LP backendも確認したが、別研究の固定domain・実行契約・LP目的なので流用・変更しなかった。
今回の追加はSciPy 1.14.1/HiGHSの0–1 MILPと、保存値だけの比較・下界計算である。

各登録caseは従来どおり、`e=floor(log10(E/sum(error_weights)))`と`e+1`の2択。
この同じ登録規則を両armへ適用する。構造誤差重みが違うため、同一capでも実際のeはarm間で異なる場合がある。
cross-round/keyの候補poolを作ったり、保存されている別caseの精度を新候補へ足したりしていない。
同じstrict word・実T価格・回転出現多重度を使い、帰りのactual adjointを両側分課金する。
最適化制約は旧来のnominal requested-epsilon上界で維持し、より小さい実guard値へ誤差規約を置換していない。
実guard上界も全配分でnominal以下か確認した。

各caseは、共通precisionから1 decade緩和するkeyを選ぶ単一制約の0–1 knapsackへ帰着する。
目的は実T削減量の最大化。負・ゼロの利益を持つ緩和は不要なので基準precisionに保つ。
MILPの提案解は有理数で誤差予算を確認し、常に旧貪欲解以上の実行可能解だけを採用した。
浮動小数点solverの最適性宣言だけで厳密最適とはしていない。

## 旧cost-sensitive alphaとの対応比較

以下の最初の3行は各armの旧alpha・旧cap scheduleを固定した比較。
4行目は両armに同じ旧2つのcost-sensitive alphaを初期値として与え、登録capの選択と
旧来の解析的cost-sensitive alpha更新を交互に行い、整数shot総Tが最小の実行可能解を保持した比較。
cap scheduleが反復・循環した時点で打ち切り、全cap/alphaの最適性は主張しない。

| 比較 | 通常arm 総T | 構造arm 総T | 構造armの数値削減率 |
|---|---:|---:|---:|
| 旧貪欲policy同士 | 493,576,769,775,156 | 490,876,793,924,848 | 0.547022% |
| 強化通常 vs 旧構造、旧alpha/cap固定 | 493,576,756,183,556 | 490,876,793,924,848 | 0.547020% |
| 両arm MILP強化、旧alpha/cap固定 | 493,576,756,183,556 | 490,876,793,817,392 | 0.547020% |
| 両armへ同じalpha/cap再最適化機会 | 493,413,295,481,472 | 490,876,793,817,392 | **0.514072%** |

通常の配分だけの改善は13,591,600 T、構造の配分だけの改善は107,456 Tであり、旧差を消さなかった。
最後の行の総shotは通常11,616、構造11,610。全18roundのT・shot・CX・rho/beta・disk・alphaを
[ROUND_TABLE.tsv](../../../artifacts/track_c2_b_strong_baseline/2026-11-06/ROUND_TABLE.tsv)に保存した。
T、CX、RZ、depthを合算していない。新しい全回路価格・深さを取得したものではない。

同一alpha vectorを両armへ渡した比較も独立に保存した。各roundのcap選択機会は両armで同じ。

| 共通alphaの出所 | 強化通常 総T | 強化構造 総T |
|---|---:|---:|
| 旧通常cost-sensitive alpha | 493,436,317,669,956 | 490,899,822,955,024 |
| 旧構造cost-sensitive alpha | 493,413,295,481,472 | 490,876,793,817,392 |

[COMPARISONS.json](../../../artifacts/track_c2_b_strong_baseline/2026-11-06/COMPARISONS.json)に
旧条件対応・共通alpha・再最適化の全round値を分けて保存した。

## 下界・最適性gap

全72 MILPでsolverは最適性を報告し、exact feasibilityの修復は0件。
独立の有理fractional-knapsack下界（整数T savingのgcd格子で強化）と一致し、
**厳密なcase最適性を確認できたのは27/72条件**。残る45条件ではsolver表示と厳密証明を区別する。
各caseの実行可能T、厳密下界、gap、solver gap/node数は
[ALLOCATIONS.json.xz](../../../artifacts/track_c2_b_strong_baseline/2026-11-06/ALLOCATIONS.json.xz)に保存した。

cost-sensitive alphaに旧一様alphaのLP総額を流用していない。
各caseの新しいT下界と、誤差会計から得るC下界を使い、全alphaと登録2capを包含する
連続shotのLagrange緩和から新しい全18round下界を得た。
Nを各軸shot、C=4/(rho^2 sin(beta)^2)とすると、必要なround失敗予算の下界は
`4 exp(-N/C)`。lambda>=0に対し

`sum_m min_cap inf_(N>=0) [2 T_lower N + 4 lambda exp(-N/C_lower)] - .05 lambda`

は総Tの下界になる。内側の解はN=0または`C_lower log(2 lambda/(T_lower C_lower))`。
lambdaの探索は下界を強くするためだけに使い、取得したbinary64 lambdaを正確な入力として
192-bit区間評価し、整数総Tに対する下界を外向きに求めた。
これはcase精度選択・cap選択・整数shot・連続alphaを緩和したものであり、実行可能baseline価格ではない。
X/Yは同一価格・同一誤差条件の旧対称モデルで扱った。

| arm | 実現総T 上界 | 厳密緩和下界 | 未確定gap（上界比） |
|---|---:|---:|---:|
| 通常 | 493,413,295,481,472 | 486,238,229,821,631 | 1.454170% |
| 構造 | 490,876,793,817,392 | 477,243,409,728,884 | 2.777354% |

構造の実現総Tは通常の下界を下回っていない。従って、より良い通常配分・alpha選択を排除できず、
「強い通常baselineに対する改善を数学的に確立」とは言えない。
一方、通常の実現値が旧・新構造の実現値以下になった結果でもないので、「改善消失」も認定しない。
**判定はOPTIMALITY_UNDETERMINED（最適性未確定）**。
対象は登録2精度と固定回路だけで、一般の精度配分全体・世界的新規性・別PFへの優越を主張しない。

## 検証・実行・停止

4つの小規模knapsackを全列挙と照合し、貪欲法よりMILPが改善する人工例と、新しい全alpha下界の人工例を検査してPASS。
保存結果では72配分の有理誤差予算、18roundのalpha総和.05以下、整数shot×T総和、旧rho/beta/shotの一致を確認しPASS。
これは主agent自身の技術検査で、第三者による独立科学レビューではない。
[SOURCE.json](../../../artifacts/track_c2_b_strong_baseline/2026-11-06/SOURCE.json)、
[TECHNICAL_CHECKS.json](../../../artifacts/track_c2_b_strong_baseline/2026-11-06/TECHNICAL_CHECKS.json)。

72独立MILPを利用可能64 CPU affinityでprocess並列化し、1 MILPにつき1 threadとして多重oversubscriptionを避けた。
人工RAM上限は設けず、SERVER 2の空き資源を確認した。科学本体約5.65秒。
新規合成0、compile/build0、sampling0、precision候補追加0、AI subagent0、共有設定変更0。
既存のraw・SOURCE・grant・ledger・marker・STOPを変更していない。
[SUMMARY.json](../../../artifacts/track_c2_b_strong_baseline/2026-11-06/SUMMARY.json)、
[STOP.json](../../../artifacts/track_c2_b_strong_baseline/2026-11-06/STOP.json)。

次の判断担当はユーザー／GPT：未確定gapを詰める価値があるかを判断する。
Codexは追加探索を開始せず停止する。fusion最適化・別PF・候補拡張は本実行の範囲外。
