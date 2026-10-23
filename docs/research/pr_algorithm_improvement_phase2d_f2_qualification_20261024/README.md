# Phase 2D-F2：統計資格の確認と実行前STOP

**F2_QUALIFICATION_BLOCKED_BEFORE_EXECUTION_MANDATORY_STOP**。
F2科学実行の明示承認は受けている。ただし共通finite targetに対する有限ビット実装差を覆う
誤差予算γの保証を既存資産から確定できず、指示§3・§6に従って科学処理開始前にSTOPした。
科学的FAILや実行中の技術エラーではない。新grantの発行・消費、研究回路build／compile、
F2 law／reference生成、保存結果の科学的再解析はすべて0件。旧one-shotを再利用していない。

[機械可読記録](qualification.json)に計画・実績・資格範囲・SOURCE hashをまとめた。
今回の新しい成果は、既知の統計式の適用条件の確認と、限定scalar区間経路の技術確認である。
SS／SP／PS／PPの性能順位、必要N、資源優位性はまだ得ていない。

## 固定資料・比較条件

起点はF2-PRE `909b13a6f5f618fc3c1ba54ce3e56764e784d12f`、
F1正常結果 `b0dad8cc2f7a4cb821f53185c4a65f3a42e8f381`。
[F2準備と既存SOURCE](../pr_algorithm_improvement_phase2d_f2_preparation_20261024/README.md)、
[P0数値規則](../pr_algorithm_improvement_phase2d_preparation_20261023/SEMANTICS_AND_NUMERICAL_GATES.md)、
[F1結果](../pr_algorithm_improvement_phase2d_f1_rerun_20261024/EXECUTION_REPORT.md)を確認した。
正式共通指示2026-10-23の§8.4・§10・§10.3と最新のユーザー指示を適用した。

対象は4 mode・2粒子・occupied(0,1)、L_D=1、残差density-square 2fragmentの合成Hamiltonian。
geometry／AO basis／分子DF rankは非該当。T=1/5、q=1、delta=1/5の一点、r=2、h=1/10、K=0。
state、sector、scalar、D/R分割、full wrapperとcompiler・P0閾値は変更していない。
F1/K2のlaw・信号・価格をF2結果へ流用していない。physical同精度比較は対象外。

SOURCE103file、保存B入力7fileと登録case計画のread-only照合はPASS。
専用CPython3.11.9環境の固定依存50件も一致した。
科学SOURCEのclosureは `f2ee499081ad336d57d507916cc602fb3649137f7fdbfd384dc550d6ab14d400`。
今回、科学SOURCE・既存draft契約・旧packetの資格フラグを変更していない。

## 統計式に適用できる既存結果

各arm／axisの各shotについて、freshなtrajectoryと測定を独立に繰り返し、補正後の出力を
Y∈[-Btot,Btot]とする。独立性はshot間の条件であり、同一shotのtrajectoryと測定結果が
互いに独立という仮定ではない。後者は一般に依存する。
K0、q1/r2ではBtot=B²=1+(lambda*h)²で、有限ビット確率をexact有理数として正規化した
同一lawを費用平均と仮想測定の両方に使う必要がある。

Hoeffdingの原著Theorem 2、式(2.6)を[-Btot,Btot]の両側へ適用すると、
axis平均のずれtの確率は `2 exp(-N*t²/(2*Btot²))` 以下。
共通finite target z_Kと実装された複素平均μとの差が `|μ-z_K|<=γ` で保証され、
s=η-γ>0なら、各axisでt=s/√2、二つのaxisをunion boundして

```
P(|z_hat-μ| >= s) <= 4 exp(-N*s²/(4*Btot²))
N_X=N_Y=ceil(4*Btot²/(η-γ)² * log(4/α_arm))
```

を得る。これは既存adapterの式と一致する。Nは十分数であり最小必要数の認定ではない。
根拠：[Hoeffding (1963), 原著p.16](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)。

4arm同時保証ならα_arm=α/4とし、さらにarm間でunion boundする。
axis間・arm間の独立性はunion boundに不要。αを各armへそのまま使う場合は各arm個別の保証であり、
4arm同時失敗率をα以下とは呼べない。η=.025／α=.05と適用範囲は依然として採択回答待ちであり、
実行承認から採択を推測していない。sampler実行と量子測定は0件なので、IIDを実測で立証したとは扱わない。

## 有限ビット保証が残る理由

必要なのは、全4armについて、実際に使う正規化済みlawとcompiled回路のX/Y期待値からなるμと、
共通z_Kとの差を覆うγ、およびそのη内の配分である。γ=0や任意の小さい値を置いていない。

| 既存処理・証拠 | 確認できたこと／保証として不足すること |
|---|---|
| f2_execution.pyの確率 | binary64値をFraction化して正規化し、理想確率とのL1差を記録する。SOURCE自身がDIAGNOSTICS_NOT_A_CERTIFICATEと明記。 |
| circuits.pyのframe生成 | support completionはNumPy再直交化と通常のunitarity検査。native Gaussianは既存BogoliubovTransformへ接続する。適用可能な誤差certificateはない。 |
| df_trotter/ops.py・compile | 既存Gaussian分解と固定Qiskit transpileを再利用するが、角度・分解・compileの厳密な合成誤差を返さない。 |
| gates.py・reference.py | 通常数値のmatrix／signal gate、90/128桁の一致。これらの評価自身の誤差を覆うenclosureではない。 |
| P0の数値規則 | engineering acceptanceであり、gate数依存の厳密forward-error boundではないと明記。 |

確率誤差だけを扱うことや、二精度結果の近さをγの証明へ読み替えることでは不足する。
F1でのPASSも、q/r/Kが異なるF2のこの資格を自動成立させない。
新しい非自明な定理や未承認の追加回路・数値検証を行わず、未解決として返す。
physical bias保証は今回の主比較に不要だが、この実装差の問題はfinite-target比較にも残る。

## Scalar interval／log／ceilの限定技術確認

既存のintervals.scalar_context／rational_interval／endpointsとgates.shot_count_from_enclosureを再利用した。
productionの区間演算器やN処理は再実装していない。
追加は[技術test](../../../tests/management/test_pr_dictionary_f2_scalar_intervals.py)のみ。
既存mockは実区間演算を呼ばないため、今回許可されたscalar確認には独立したtestが必要だった。

mpmath1.3.0の実装で、正実数の四則・整数平方とlogの端点にfloor／ceiling方向を使う経路を確認した。
公式文書はivの包含性を説明する一方、interval対応全体をexperimentalとしているため、
任意の関数やmatrix expを一括資格認定しない。
[公式文書](https://mpmath.org/doc/current/contexts.html)、
[固定tagの区間SOURCE](https://github.com/mpmath/mpmath/blob/1.3.0/mpmath/libmp/libmpi.py)を参照。

実環境で次の5testがPASSした。

1. 正負・大きい整数を含むexact有理数変換と四則・平方の包含。
2. log9入力の独立照合。候補α=.05の個別／4arm同時保証で使うlog(80)・log(320)を含む。
3. exact整数ceil、整数へ片側から接する区間、整数境界を跨ぐ区間の拒否。
4. NaN／無限端点の拒否。
5. 研究fixtureとは無関係な人工有理数によるN式と独立oracleのceil一致。

logの独立oracleは、x=2^k*y、1<=y<2に還元し、z=(y-1)/(y+1)について
`2 Σ(j=0..m-1) z^(2j+1)/(2j+1)` をexact Fractionで計算する。
正の打切り残差は `2*z^(2m+1)/((2m+1)*(1-z²))` 以下。m=256の両端を使い、
この証明済み区間全体が既存logの出力区間に含まれることを確認した。
これは高精度点近似二つの一致とは異なり、ここで検査した定数の包含確認である。

**確認済みの範囲は固定90桁・positive rational arithmetic・上記二つの候補log定数・既存ceil規則**。
将来α=.05と既存scopeのいずれかを正式採択する場合に再利用できる。
別のα、任意関数、回路／meanの誤差保証へ拡張しない。
F2のB・lambda・Nは評価しておらず、実入力でのceil解決は実行時確認として残る。
未採択の旧契約を資格取得済みに書き換えず、根拠の最終bindingも未実施。

最終technical testはwall0.151秒、user CPU0.147秒、system CPU0.004秒、peak RSS29,260 KiB。
これはF2科学計算費ではない。F2の生成・build・compile・集計費は未測定である。

## 計画と実績、保全、次の担当

| 単位 | 登録計画 | 今回の実行 |
|---|---:|---:|
| ordered trajectory | 6,837 | 0 |
| full-wrapper build要求 | 13,676 | 0 |
| compile要求 | 13,676 | 0 |
| reference精度評価 | 27,356 | 0 |
| armごとのshot資格評価 | 4 | 0 |
| quantum measurement draws | 0 | 0 |

13,676はX/Y13,674＋共通診断2。異なる単位を科学call総数へ合算しない。
4armすべてのactual RZ・CX・depth・N・N加重費用は **未取得(null)**。
0費用や同率として補完せず、科学的PASS／FAILも判定していない。
辞書生成は保存B再利用を予定しており、F2の実測生成費としては計上しない。

旧B/F1のSOURCE・raw・grant・ledger・marker・STOP、F2-PRE原本とユーザーworkspaceを保全する。
read-only check-onlyはSOURCE/input/planを通過し、旧packetの統計未採択・interval未binding・
最終契約未採択を返した。今回は資格確認のtestと記録だけを独立branchへ追加する。
非公開raw・承認情報・認証情報・内部pathは公開しない。

**次の担当はユーザー／GPT**：η／α／失敗確率範囲を正式採択し、共通finite targetとの差を覆う
γの保証根拠・値・配分を決める。新たな保証取得に科学計算等が必要なら、その範囲を判断する。
すでにF2科学実行の承認はあるため、同一範囲の着手承認を重複して要求するものではない。
Codexは未解決事項が解消された指示後、最終根拠と契約をbindingし、新one-shotのpreflightを通してから実行する。
現時点では **mandatory STOP**。追加実験・physical同精度比較・重要科学レビューへ自動移行しない。
