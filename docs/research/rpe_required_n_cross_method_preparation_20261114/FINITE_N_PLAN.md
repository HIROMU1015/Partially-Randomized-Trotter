# Z1＋T1/Q3採択事項と必要Nの有限実験案

未採択・未実行。保存Pauli2結果を出発点とし、再計算しない。これは最小shot総数を求める計画ではなく、将来の`Σ N E[C]`比較へ渡せる少数のwhole-vectorを検証する案である。

## 正式採択前に確定する契約

| 判断 | 現在の証拠と不足 |
|---|---|
| 推定量 | Z1＋T1 `consecutive_dyadic_nearest_branch_principal_energy_total_v1`は診断opt-in。zero保持、tie argmin、binary64演算順、主値energy、priorをbranch選択へ追加しない元規則を採択するか。旧部分decoderの未解決lossと併用しない |
| energy task / target | 保存full DF-H・保存state・solver proxy −2.1663874486347634 Haに条件付く診断と、物理ground RMSE保証を区別。ground認証・target包含区間・state誤差は未取得。DF前Hamiltonianの保証には自動移送しない |
| lossと予算 | 全事象の`(E_hat−E_proxy)^2`を直接評価する契約を今回以後にも採択するか。元Minkowskiのεqpe/εtrotを保持した分離契約と、直接モデルriskの判定閾値τは別判断。τ=εE²やτ=εqpe²をBが自動採択しない |
| 物理近似 | rawモデルはPF/finite-K/normalizationの位相ずれを含み得る。直接riskへAのδE/δE²を再加算しない。分離予算へ接続するには別の対応定理・包含が必要。PF H_effをPR finite-KのHermitian generatorに流用しない |
| law | 保存ideal finite-K raw、`EXACT_AFFINE_OF_BINARY64_RAW`、shots/axes/roundsのIIDはモデル条件。実PCG64/CDF・sampler lawとの距離・依存性は未資格 |
| 数値 | exact二項質量・整数/有理会計と、binary64 raw/decoder/engineering allowanceを分離する。per-transition allowanceはcertificateでない。境界margin・target/law数値包含が未認証ならそのscopeを明示する |
| Nの呼称 | モデル内でτを満たした**評価済みwhole-vector**と、物理taskに対する十分Nを分ける。資格を出すproducerは科学的根拠と入力移送が整うまで追加しない |

原論文nominal、旧HP型N、旧MCの経験的RMSEはこの資格の代用にならない。決定論的主経路にMC seed・反復数・MC confidenceを導入する必要はない。将来の価格sampling区間のconfidenceは別入力。

## 最初の有限案：1つの共通scaleを2構成へ

1. 保存control vectorを`n_m^(0)`とする。base2結果の再利用はSOURCE・全契約が一致する場合だけ。新採択契約が元2診断と異なる場合、同一結果として再ラベルせず、再利用可能性を先に判断する。
2. **推奨する次の承認単位**は、LD1/LD2それぞれ1本、計2 Q3 call。新候補は全13round・両軸共通で`N_m,X=N_m,Y=max(1, ceil(3 n_m^(0)/4))`というscale族の1本を提案する。整数化・配分ID・SOURCEを実行前に固定し、現在は数値vectorを生成・評価していない。
3. 3/4は控えめな配分縮小を調べる最初の有限challenge案であり、最適scaleや成功の予測ではない。LD1/LD2を同じscale族で評価すれば、異なる辞書/rawの下で同じshot policyを比較できる。価格がない段階でLDや配分のcost winnerは出さない。
4. 両結果を保存したらSTOP。さらに縮小を調べる価値がある場合だけ、共通scale1/2の各1本・計2 callを**次の別承認単位**として提案する。今回の2件からその実行許可を推定しない。

この段階順は多数vectorの総当たりを避ける。初回は最大2新評価だけで、「同じ有限縮小案に対してモデルriskを評価できるか」という判断が可能になる。各構成で別scaleに自動分岐しない。shot数変化でcount格子・parity・zero/tieが変わるため、riskの厳密単調性を仮定した二分探索・未評価配分への補間は行わない。

価格の適格domainと差が揃った後で、共通scale以外のcost-aware familyを別に提案できる。roundごとの独立最適化や26変数自由探索を今採択しない。有限menu内の採用、探索族内の最小、全shot空間の最適という主張を分ける。

## 判定順序と有限範囲

1. metadata：元Pauli SOURCE7142476、元/total canonical、raw/target/law/辞書/κ/K/13round、allocation IDと正確整数を照合。
2. 外部入力：ユーザー採択のmodel閾値τ・loss・target・数値scopeを確認。既存control結果へ新filterをかける場合も、保存値の新解析として承認範囲へ含める。
3. kernel：既存`T.evaluate`→Q3と高速backend9e294778を再利用。旧部分DPを呼ばず、両構成の独立jobを並列化できる。
4. 有限範囲は現実行と同じρ=1、round outcomes4225、verification pairs55043173、frontier8192、transition pairs141559036を初期案とする。縮小vectorの全count・隣接検査を包含することをmetadata段階で確認する。これは科学追跡範囲でありworker/CPU/RAMの上限ではない。
5. 保存後、認められたモデル・数値scope内で上界≤τなら`MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`。下界>τなら当該**条件付きモデル**の候補不適合。区間がτをまたぐ、又は上界だけ超過する場合は`INCONCLUSIVE`。保守的上界超過を物理的精度未達と断定しない。
6. 完了2件の保存・公開後STOP。UNKNOWN/失敗なら部分成果を保全し、無断retry・ρ/shot/追跡範囲変更をしない。片側失敗後の他構成継続可否は有限実行指示で事前固定する。

質量不保存、非有限、source/canonical不一致、閉包不成立、追跡範囲超過、残余loss会計不能、上下界矛盾、保存不能はUNKNOWNとしてSTOP。残余を0にしない。有効上界のminとbias²/Varの診断をそのまま保存する。

実測済みbase2件の高速版負荷は既存REPORTにあり、新vectorの計算時間保証へは外挿しない。独立job・count batch・exact集約・遅延約分を再利用し、共有環境の設定を変えない。

## 実行前の重複照合（11/14版§10.5）

以下は計画時の[READINESS](READINESS.json)・[適用一覧](SIGNAL_APPLICABILITY.md)を再利用する。実行までに計画・SOURCE・他担当の取得状況が変われば、影響する項目だけを一次記録で再確認する。照合完了は科学実行許可ではない。

| 予定する項目 | 既存成果の扱い・新規候補との差分 |
|---|---|
| Pauli base2件・coarse/lemma | caf1417の元runと固定SOURCEをそのまま再利用し、新規取得から除外。今回これらを再計算しない |
| 新whole-vectorのPauli risk | 同じraw/辞書/normalization/target/lawは再取得不要。新allocationでのriskだけが別の科学評価候補。実行前に同一allocation・decoder・契約の保存結果又は他担当の実行中runがないか確認する |
| PR-DFのtotal適用 | 旧partial DPは推定量・lossが異なるためtotal評価へ転用しない。保存13rawは再利用候補のまま、旧gridを保つ契約bridge不足を解消するまで該当分を保留 |
| Full-H κ2・Truncated p7 | 保存13rawを再利用し、新しいsignal取得から除外。次のriskは明示された新decoder/shot/loss契約の評価だけ。R448や旧DFとの同一視は拒否 |
| Full PF全round入力 | 確認したSOURCEではH_eff・二時点mean・限定catalogが利用可能。全13packet未同定を「過去に未実施」と数えない。関連一次記録・SOURCE・actual幅/物理IDを確認し、保存値を受入可能なroundは除外、差分を特定できたroundだけ新規取得候補とする。証拠不足なら取得も保留 |
| 価格・G | 今回の対象外。将来は候補×round×X/Y×cohortごとに元receipt・測定操作・law/compiler/loweringを照合する。保存単一pathを全law期待値へ昇格しない |

IDだけの相違は実パラメータ・結果に寄与するSOURCE差分を確認し、契約上受入可能な場合だけ元識別子付きで再利用する。証拠不足・矛盾は該当項目をSTOPし、回収すべき原記録を示す。独立検証が必要ならその目的と既存証拠では満たせない範囲を明示する。元runの再読・同cohort再集計を独立反復と数えない。

## 別途承認が必要な科学範囲

- **Pauli N1**：採択target/loss/τ/数値scopeの下で、提案scale3/4×LD1/LD2の計2評価＋許可された閾値filter。base2診断の再計算なし。価格・G・十分Nの自動認定なし。
- **他方式初回**：Full-H κ2とp7の各1外部指定whole-vector・計2診断を別案とする。ベクトルやrisk/law条件はPauliから自動採択しない。現在はmetadata readyのみ。
- **Full PF**：上記照合で既存成果を受入できない具体的なround・条件を特定した後に限り、採択PF menuの差分signal計算、又は必要なroundの保存固有系からのspectral再構成を新規候補とする。取得済みか判断できなければ該当分は保留。その後で別途承認範囲とdecoder/model riskの有限範囲を定める。新eigh不要でもspectral mean評価は新しい科学計算である。
- **PR-DF**：旧fixed-A gridを保つ版付きcanonical契約bridgeを先に技術整備する。科学的にdecoderを採択したことにはしない。raw再取得やPauli scheduleへの変換は不要。

ユーザーの判断と有限科学実行許可を待つ。**mandatory STOP**。
