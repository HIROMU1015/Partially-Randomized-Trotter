# Truncated PF：誤差評価方法と有限最適化計画

2026-11-14 JST。Claude Code（主agentのみ。独立レビューではない）。

**作業条件**
- branch：`codex/truncated-pf-optimization-plan-20261114`。基点はCの最新`3a8b37f1a282c0c06ac1a363b070f4376e9ff41a`（B0I signal `367dc98d`を含む）。Cの作業branchを並行作業と競合させないため、新しいbranchにした。
- 実施した範囲：既存成果の調査、metadataのみの計画生成、人工toy検証。
- **実施していないこと**：新しい実H4/H6科学計算、保存科学数値の再解析、回路build/compile/sampling、価格fit・予測、N/MSE/G、契約・予算の変更、A/B/Dへの干渉。
- 段階計画：[SCIENCE_PLAN](SCIENCE_PLAN.md)。全scheduleのmetadata：[PLAN_METADATA](PLAN_METADATA.json)。toy検証：[TOY_CHECKS](TOY_CHECKS.json)。

**参照資料について**：指示書にある「PRレビュー.md」はこのサーバーで見つからなかった。代わりに、11/13のClaude統合レビュー（§4.2のB2/B6、S1/S4、A4）とCの11/13設計`rpe_baseline_optimization_design_20261113`を基準にした。

**根拠の表記**：【既存記録】【実装】【数学】【人工】【桁評価】【提案】。

## 0. 要旨

1. **既存の実装で計画全体を表現できる。【実装＋人工】**
   - Aの評価器はすでに`TRUNCATED_PF`を受け付ける。
   - Cのpipelineは、任意のp∈0..9、B0/B0I、3公式のTRUNCATED_PF configurationを生成できる。
   - 99 configuration・154 step cellを既存generatorで生成し、c546のB0I p7 S2/.1とAのFull PF 3候補のIDを再現した。
   - 新しく必要なcodeは、caller（Aのscan callerにprefixとscalarを加えたもの）と、約30行の分解helperだけ。
2. **誤差を3つに分ける。【数学】**
   - `δE_trunc=E_g(H_ret)−E_g(H)`：Hamiltonian切断による固有値シフト。
   - `δE_PF|ret=E_g(H_eff)−E_g(H_ret)`：保持Hに対するPF近似の固有値シフト。
   - `δE_total=δE_trunc+δE_PF|ret`：元のfull DF-Hに対する合計シフト。
   - 対応は最小値どうしではなく、群の最大trace重なりで取る。toyでは恒等式の誤差が1e−12未満で、評価器の合計値とも一致した。
3. **符号相殺とoracle。【数学＋人工＋既存記録】**
   - Q1（Minkowski）でもQ2（直接MSE）でも、合計シフトはtarget（oracle）に対する符号付きの値として評価される。相殺は自動的に効く。
   - 絶対値の和（L2）やWeylのλ_R（L1）を使うのは、相殺に頼らない頑健性の判定にあたる。
   - toyでは、L3で選べる幅がL2の3.3倍になった。相殺点ではtailを±10%変えると合計シフトが62〜142%変わり、符号も反転した。
   - 先行Track C（H4 DF12）の最良Truncatedも相殺を含んでいた。相殺に依存する候補には`CANCELLATION_DEPENDENT`を付けて区別することを推奨する。
4. **推奨する有限探索。【提案】**
   - T1：全prefix p=0..9の切断シフト（36次元の固有系10件）。
   - T2：`|δE_trunc|<ε_E`を満たすpについて、3公式×Aと同じ幅の梯子でH_eff cellを評価（14 cell/p、最大140）。
   - T3以降：構造proxyで選んだ最大6 templateだけに、価格・全round signal・risk・Gを順に進める。
   - Δ・p・公式の全直積は前提にしない。最終的な比較はG_RPEで行う。
5. **公平比較に残る課題。** PRには共通H_effが未確立なので、Q1ではPRとの扱いが非対称になる（Q2なら対称）。ほかに次が残る。
   - PRも同じprefix帯（LD 5〜7）で探索すること。
   - 同じoracle水準を使うこと。
   - W0のdecoder/契約判断。
   - Dにtruncatedの価格経路がまだないこと。

## 1. 既存成果と再利用可能な実装

| 成果 | 内容・SOURCE | 今回の扱い |
|---|---|---|
| Truncated signal 26round | `367dc98d`：B0I p7 S2/Δ.1の13 raw（直接expm＋行列累乗。元定数とtail identityの位相を総Tで一回）。全数値gate PASS、UNQUALIFIED | T2の実装照合（p7/S2/hのspectral再構成と一致すること）に使う。位相ドリフトの再解析はしていない |
| 旧B0 p7 S2/.1、旧B0I p7 Y4/.4 m3/m4 | `155529e2`の保存signal。scalar規則と数値backendが異なる | B0IやH_effの代わりには使わない |
| AのH_eff評価器 | `pf_effective_hamiltonian.evaluate`（最新`878000d8d`）。KINDSに`TRUNCATED_PF`、target≠retainedを許す。scalarは`INCLUDED_ONCE_AT_STEP_TIME`。分枝はPRINCIPALかENERGY_WINDOW。元Hの固有系cacheを再利用できる | 合計シフトの算出にそのまま使う。toyでも動作を確認した |
| AのFull PF cell（p=10） | `71f04be7`ほか。S2/Y4/M5の各幅（.05〜.8、16h/7、8h/3、16h/3、32h/5、16h）のδE、重み、二時点照合 | 同じ幅の梯子の対照として再利用する（再計算0） |
| Cのpipeline | `baseline_signal_pipeline.configure/prepare`がTRUNCATED_PF（B0/B0I、0≤p<10）を受け付ける。`baseline_signal_numerics.Context.pf_step`（`matrices[:p+1]`）と`tail_scalar`（保存値、なければ既存extractor） | 計画metadataの生成に使用した。T2/T4でもそのまま使う |
| C 11/13設計 | `rpe_baseline_optimization_design_20261113`：自由度p、B0/B0I、3公式、Δ。targetはE0(H)。相殺とnorm上界の区別。除外規則 | 本計画はこの枠組みを具体化したもの。内容を重複して書かない |
| 価格 | D receipt 73件（Full PF・Full-H）にtruncatedは含まれない。保存B0I p7 m3価格（`155529e2`、旧native）。C guardではTruncatedは`NOT_APPLICABLE_NO_RTE` | T3の前に、Dのtruncated builder／receiptが必要 |
| 先行Track C（11/08） | `da8b2c3f`：H2/H4/H6のFull・Truncated・PR比較。18round、δ0=.02、高確率型（失敗率.05）、compiler A、旧DF12。H4/H6ではTruncatedが最安。prefix I0/I1と係数L1 dropのfamily、S2/Y4×h 11点、round別policy、符号相殺を明示 | 条件が違うので数値は流用しない。family定義と「切断・PF・合計の符号付き位相」の記録方法を再利用する |
| 旧Track A B0I map | DF rank12、T=.8の単一時間signal精度。identityを保持しても全240軸でRZ数の差は0 | B0とB0Iの価格同値性を確かめる仮説として使う（T3で1組検査） |
| 保存prefix metadata | 下表 | λ_RとscalarはT1で既存extractorを再実行して照合する |

保存されたprefix metadataの値（`7a8bc7f1`のSOURCE_INDEXが指すH4 context、sha256 `8396605f…`。読取のみで、配列は開いていない）【既存記録】：

| p（保持fragment数） | 1 | 2 | 3 | 5 | 7 | 9 |
|---|---:|---:|---:|---:|---:|---:|
| λ_R（tailの非identity重み） | 2.863 | 1.603 | .5833 | .05440 | 3.130e−4 | 1.561e−7 |
| c_tail（tailのidentity係数） | .4177 | .2224 | .09262 | .008913 | 4.530e−5 | 5.193e−8 |

p∈{0,4,6,8}は保存されていない。

## 2. 切断・PF・合計エネルギー誤差の評価方法

**定義【数学】**

- H：保存full DF-H（`c_0 I＋Σ_(j=0..10) M_j`。M_0はone-body）。
- 保持H：`H_ret(p)=c_0 I＋Σ_(j≤p) M_j＋c_tail(p) I`（B0I）。B0では`c_tail=0`。
- 1 stepのunitary：`U=e^(−iΔ(c_0+c_tail))·S_f(Δ; M_0..M_p)`。Aの評価器に`scalar_policy=INCLUDED_ONCE_AT_STEP_TIME`で渡す。direct signalの「総Tで一回」と`(e^(−iΔc))^q=e^(−iTc)`で整合する。
- `E_g(X)`：Hの基底群と最大trace重なりを持つ、Xの群のエネルギー。最小値を無条件に対応させない。

**3つのシフト**

| 量 | 計算 | 意味 |
|---|---|---|
| δE_trunc | `E_g(H_ret)−E_g(H)`（T1、固有系1件/p） | Hamiltonian切断の固有値シフト。PFにもΔにも依存しない |
| δE_PF\|ret | `E_g(H_eff)−E_g(H_ret)`（H_effとH_retの対応も重なりで取る） | 保持Hに対するPF誤差 |
| δE_total | 評価器の`minimum_branched_energy_minus_reference_matrix_minimum_ha`（ただし重なりで対応を確認する） | 元targetに対する合計。`δE_trunc+δE_PF|ret`と恒等的に一致 |

【人工】toy（6次元、4 fragment）で、全p・S2/Y4について恒等式の誤差が1e−12未満。評価器の値と重なり対応による値も一致した。p=0では保持Hが単一ブロックなので`δE_PF|ret≈0`になる（PF誤差が生じない）ことも確認した。

**B0とB0I**
- 固有値は`δE_B0=δE_B0I−c_tail`で厳密に一致する（toy誤差2.7e−15）。B0を別に計算する必要はない。
- c_tailはfull-FockのPauli identity係数である。sector(2,2)内で定数になるZ/ZZの組合せは含まないので、λ_Rはsector内の切断を過大に見積もる。
- sector traceを使う規則（仮称B0S）は新しい方式になるため、今回は対象外。
- p≤5ではc_tailが.009〜.42 Haあり、B0は大きくずれる。B0Iを主とする。

**分枝**
- scalarを含めた位相`Δ(E+c)`が±πに近いと、主値logが別の分枝になる。
- 【人工】Δ=1.0のtoyでは、PRINCIPALの「最小エネルギー」は折り返されたmodeで、合計シフトは−5.5 Haになった。保持HのGershgorin範囲から決めたENERGY_WINDOW（targetは使わない）では正しい値（0.0055 Ha）に戻った。
- M5 16hではAが承認したwindow `[−2.5,1.0]`を使い、window端のclearanceのgateで判断する。windowを自動で変えない。

**early round**
- 粗い幅では、初めのroundの実幅がh・2h・4h・8hになる（metadataで確認）。round別のcellが必要。
- 【人工】T=.2をstep 1回（.2）で計算する場合、幅.2のcellで再構成すると誤差4e−16、幅.1のcellで代用すると誤差5.8e−5になった。

**状態の重なり**
- 入力stateは保存full-H基底の近似。H_eff基底群への重みwが1未満だと、signalは混合になる。
- 位相の揺れは`arcsin((1−w)/w)`以下（toyで成立）。wとこの上界を各cellで記録する。
- 小さいpでは重なりが下がり得る（toyのp=0で0.78）。w≤0.5または対応が曖昧ならUNKNOWNにする。

**資格**
- これらはspectral診断である。全RPE RMSE、十分N、Gの認定ではない。
- raw signalの直接riskを評価する場合は、切断とPFの効果が既に含まれているので、δEを加算しない。

## 3. 符号相殺とoracle依存性

| 水準 | 判定に使う量 | 必要な知識 | 相殺 |
|---|---|---|---|
| L1（oracleなし） | `λ_R(p)＋|δE_PF|ret|` | 保持Hの固有系（PF側）。切断側はWeylの上界`|δE_trunc|≤‖H_tail−c_tail I‖≤λ_R` | 使わない（最も保守的） |
| L2（絶対値の和） | `|δE_trunc|＋|δE_PF|ret|` | 元Hと保持Hの固有値（oracle） | 使わない |
| L3（符号付き合計） | `|δE_total|` | 同上 | 使う（Aや先行Track Cの判定と同じ） |

【人工】toy（p2、S2）では`δE_trunc<0`、`δE_PF|ret>0`。予算内で選べる最も粗い幅はL2で.15、L3で.5、L1では予算内の幅がなかった。Δ=.35付近で合計は98%相殺した。tailを±10%変えると、その点の合計は62%・142%変化し、符号も反転した。相殺点は細かいモデル差に敏感である（toyの例示であり、H4での一般化はしない）。

**契約との関係【数学】**
- Q1（Minkowski、`|δE|≤ε_trot`）でも、Q2（直接モデルMSE、bias²＋Var）でも、評価は保存target（oracle）に対する符号付きの量になる。Q2のDP riskは、rawに含まれる符号付きbiasをそのまま使う。
- したがって相殺は、どちらの契約でも自動的に効く。これはFull PF・PRも同じで、全方式が同じoracle targetで資格判定されるなら対称である。

**推奨【提案】**
- 主比較はL3で行う（既存規則と一致）。
- L3では通るがL2では落ちる候補に`CANCELLATION_DEPENDENT`を付ける。
- その候補には、隣接幅でもL3が成立することと、L2との差を必ず併記する。
- 「切断で精度を保ちながら安くなる」と主張する場合はL2を、「事前知識なしで設計可能」と主張する場合はL1を、それぞれ別に示す。

**oracleの範囲**：H_effのスペクトル診断自体が古典対角化を使う。これはFull PFでも同じで、本研究では全方式共通の「設計用oracle」として扱う。区別が必要なのは、符号まで利用した微調整（相殺）である。B0Iのc_tailはHから得られる既知定数で、oracleではない。

## 4. 推奨する有限探索範囲

| 自由度 | 範囲 | 理由 |
|---|---|---|
| prefix p | T1：0..9全部（固有系10件、安価）。T2：`|δE_trunc|<ε_E`のp | λ_Rの閾値だけでは除外しない。p5のλ_R=.054はε_Eより大きく、Weyl上界からは判定できない。厳密値が必要 |
| 方式 | B0I（主）。B0は解析的に導く | §2 |
| 公式 | S2・標準Y4・M5 best（既存の3つ） | 8次以上は対象外（pipelineの規約） |
| Δ | S2 {h,2h,16h/7,8h/3}、Y4 {4h,16h/3,32h/5}、M5 {8h,16h}（＋early cell）。境界challengeは別承認で1点ずつ | Aと同じ格子なのでp=10と対で比べられる。δE_truncが予算の一部を使うと、S2では最適幅がh〜2hへ下がる見込み |
| 予算・相殺 | Q1/Q2の両方の判定と、L1/L2/L3を記録 | §3。契約が未採択（W0）のため |
| 価格・G | 構造proxyで上位から最大6 templateを選び、q-affine 2点＋holdout 1点 | 最終的な判断は`G=ΣN·E[C]`で行う。step数・局所誤差・proxyだけで確定しない |

**事前の桁評価【桁評価、確定ではない】**
- 1 stepの構造費用がブロック数（one-bodyを含めp＋1）に比例するなら、p7はFull PFの約8/11になる。同じ幅なら最大約1.38倍安い（11/13レビューA4の上限1.38と同じ構造）。
- p7のWeyl上界λ_R=3.13e−4は、ε_trot=7.97e−4の約39%にあたる。L2で判定すると、S2の許容幅はFull PFの16h/7より狭くなり、利得は一部相殺される見込み。
- M5ではPF誤差が非常に小さく、切断がほぼそのまま予算を使うため、ブロック数の比がそのまま効く見込み。
- いずれもT2/T3で確かめる仮説であり、これを根拠に候補を除外しない。

**探索しないもの**（SCIENCE_PLANに記載）：係数単位のdrop、round別のprefix、並べ替え、B0S、h/M。先行Track Cでは係数L1 dropが最良だったので、prefix familyでTruncatedが有望と分かった段階で、第2段として提案する。

## 5. 必要な科学実行の件数・SOURCE・STOP

[SCIENCE_PLAN](SCIENCE_PLAN.md)の表を正本とする。上限は次のとおり。

- T1：context 1、scalar・λ_Rの抽出4（ほか保存6件の照合）、固有系10。
- T2：最大140 cell（各cellでPF step 1、Schur 1、二時点のmean各1組）。加えて、p7/S2/hと保存13 raw signalの照合1件。
- T2b（別承認）：最大3×|P*|。
- T3：最大36 wrapper（6 template）＋B0/B0Iの価格同値検査1組。
- T4：最大78 mean。
- T5：最大6のDP。
- T6：集計。

**SOURCE**：
- Aの評価器`878000d8d`、PF kernel `dd24dd329`、C numerics `5cc7c07e`（`86f37d25`とbyte一致）。
- 保存input：INDEX `7a8bc7f1`。元Hの固有系cache：`69ee0ff8`。reference pair：`bba15083`。
- extractor：`e417ffc9:scripts/h46_new2_core.py:matrix_only_tail_coefficients`。
- 数値policyはAのPLANと同じ（1e−9、位相guard 1e−8、energy cluster 1e−8）。

**STOP**：
- SOURCE・inputの不一致、またはp7/S2/hと保存B0I signalの照合不一致：全体STOP。
- 数値・分枝・対応のgateが不成立：そのcellだけUNKNOWN（科学的な不成立とはしない）。
- 閾値の緩和、windowの変更、retry、梯子の自動拡張はしない。
- 各段階の後はmandatory STOP。

## 6. PRとの公平な比較に残る課題

1. **契約の非対称**：Q1では、Truncated/Full PFはH_effでε_trotを示せるが、PRは共通H_effが未確立。Q2なら全方式をrawのDP riskで対称に扱える（W0のD3）。
2. **探索範囲**：PRは現在LD1/LD2でκも固定。Truncatedが有望なp帯（≳5〜7）では、PRのLD5/7（λ_Rが小さくtailが安い）を探索しなければ公平でない。同じprefixの対照は説明用で、両方式を同じprefixに拘束しない。
3. **oracle水準**：TruncatedのΔやpを相殺込み（L3）で選ぶなら、PRのκやLDも同じtarget知識で調整できることを明記する。
4. **価格経路**：Dにtruncated builder／receiptがない。PFとTruncatedは同じdeterministic native・lowering・profileで取得する。PRの新ZZ loweringとの混在はC guardで拒否される。
5. **decoder/契約**：W0のzero/tie、未解決loss、資格契約は全方式に同じものを適用する。
6. **表現の範囲**：保存DF10の順序に沿ったprefixに限る。係数dropやround別policy（Track Cで有効だった）は未探索として明記する。shell（h=.1/M12）の固定はPF側を不利にし得る（11/13レビューA6）。
7. **一般化**：1つのgeometryでの相殺は他の条件へ移らない。

## 7. 検証・限界・保全

**metadata**：[PLAN_METADATA.py](PLAN_METADATA.py)で既存のB/D/C generatorを使い、99 configuration（p0..10×9幅）と154 cellを生成した（約18秒）。数値callは0、H4の配列・contextは読んでいない。c546のB0I p7 S2/.1とFull PF S2/.1、およびAのFull PF 3候補のIDが一致した。

**toy**：[TOY_CHECKS.py](TOY_CHECKS.py)で9/9 PASS（約0.5秒）。凍結した評価器`878000d8d`とkernel `dd24dd329`を使い、次を確かめた。
- 分解の恒等式、Weyl上界、B0とB0Iの関係。
- 分枝window、L3≤L2≤L1、相殺で選べる幅が広がること。
- round別の実幅が必要であること、混合による位相の上界。

分解helperは検証用のoracleで、srcには入れていない。

**限界**：
- H4での実際の`δE_trunc`、`δE_PF|ret`、価格はすべて未取得（T1以降の新しい科学計算が必要）。
- λ_R・c_tailは保存metadataの引用で、Weyl上界として使えるのは、λ_Rが既存extractorの非identity 1-normであることをT1で照合した後になる。
- 桁評価は仮説である。

**保全**：
- 固定SOURCE・input・raw・結果・grant・ledger・marker・KEEP・STOPは変更していない。
- A/B/Dのworktree・process、共有環境も変更していない。
- 新設したのはこのdirectoryと索引の各1行だけ。

**mandatory STOP**：次はユーザーが、T1/T2の承認可否と、W0の契約判断（Q1/Q2）を決める。
