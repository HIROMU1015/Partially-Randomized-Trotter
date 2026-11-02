# H8 PR：候補選択後の統計評価設計（実データ解析は未承認）

[統合報告](track_a_h8_rare_tail_selection_readiness_20261018.md)に固定資料・seed/pairの監査結果。ここでは方法の成立条件を設計し、実H8へ平均・区間・bootstrap等を新たに計算しない。

## 推定対象と再利用単位

候補j、trajectory i、endpoint e=(ε, allowance倍率, bias区分)について、保存shotを固定し、

`G_{j,i,e}=N_c(j,e) C_{j,i,c}+N_s(j,e) C_{j,i,s}`。

CはP0のnative full-wrapper RZ。cosine/sineは同じtrajectoryに属し、このpairを標本単位とする。量子測定shotを回路費用標本数に数えない。ε・倍率・bias区分が変わっても同じCを使うため、その行を独立sampleとして増やさない。

- j_E：探索n8のみで固定した候補。確認n32はj_Eについて独立streamの評価に使える。
- j_C(e)：確認union6の同じ点費用で選んだ最小候補。平均の標準SEをそのまま選択補正済みintervalにできない。
- min_j μ_j(e)：有限候補内の母平均最小。観測minimumやj_Cの母平均とも違う。
- 選択手続そのものの期待性能・regret：探索・確認を繰り返す実験を含む別estimand。保存6候補/n32だけから広い探索全体の性能を自動推定しない。

ε=.0005は元4精度探索に入っていないため、探索事前winnerを遡及的に作らない。後続D0で固定union6を使ったことと、.0005専用の独立選択実験を区別する。

## 方法別の成立条件

| 方法 | 必要データ・仮定 | 技術的適用可能性 | 限界 |
|---|---|---|---|
| 探索事前固定候補の確認平均/SE | j_Eの保存時点、別phase n32、両軸pair、固定shot。候補内IID/有限分散のモデル | 保存データから可能。新native不要 | 有限標本の記述/標準誤差でありtail保証なし。未観測高費用で分散を過小評価し得る |
| union6同時区間 | 固定family、各候補に有効なmarginal interval、family誤差配分 | Bonferroniは候補間共分散を要しない | marginal interval自体が正しくなければ同時保証にならない。観測最大をsupport上界にしない |
| bounded Hoeffding + union bound | 独立な候補内標本、全supportの既知費用上界B、familyサイズM、事前α | 人工kernelのみ準備。H8のBは未取得のため現状実行不可 | 元instruction STOP capは完了費用母集団の上界ではない。失敗を費用0として切り捨てない |
| 多変量正規/t近似・max-t | 有限分散・近似精度、依存design/共分散の正当化、tail評価 | 仮定を採用する場合だけ後続案 | n32/未観測層だけで近似精度を保証しない。候補間共分散は現在null |
| 事前固定familyのtrajectory cluster bootstrap | 候補内IID・bootstrap妥当性、候補間stream設計の仮定。全endpointの同じ再抽選index | 人工実装あり。実行すれば探索的感度に限る | empirical supportしか復元せずs4/multiの未知費用は出現しない。勝者頻度は母平均最良確率ではない |
| 選択を考慮した推論 | 選択event・候補集合・依存・分布モデル。しばしばnormal/exponential-family等 | 研究設計案のみ。既知手法をそのままH8へ適用できるとしない | 選択条件・未観測tail・推定対象を固定する必要。現状の費用分布への厳密な正当化なし |
| 新独立holdout | 既存n32で選んだ候補を新標本前に固定、別seed/law、明示予算 | 新sampling/nativeが必要な代替案 | 既存n32を見た後のsplitを未使用holdoutとは呼べない。有限holdoutでもtail上界は出ない |

同時に有効な各μ_j区間[L_j,U_j]が得られる場合に限り、選択されたj_Cの母平均にもその区間が適用でき、有限候補最小には[min L_j,min U_j]を接続できる。これはcoverageの前提を満たした場合の数式であり、現在のH8で区間を構築した結果ではない。

全supportについて0≤G≤Bが証明済みなら、Hoeffding/union boundの半径案は`B sqrt(log(2M/α)/(2n))`。今回は人工値で式を検査しただけ。H8で未証明Bを観測最大や既存compile capで代用せず、PROVEN_ALL_SUPPORT_BOUNDがなければ拒否する。

## family・解析範囲の未承認案

主解析をunion6・通常bias・5精度・倍率1へ限定する案と、通常/非相殺×5精度×倍率1/10の全endpointへ拡張する案を分ける。最大familyは6×20の候補平均。ただし保存不適格条件を0平均へ変えず、固定eligibility maskの定義とfamilyの数え方を承認前に確定する。

α=.05のfamily誤差、bootstrap反復2000はproposalであって固定科学契約ではない。α_axis=.025という元shot規則とは別の統計目的であり、shot/allowanceへ流用しない。反復数はMonte Carlo出力安定性を別に調べて決める。H8でのbootstrap比較や収束検査は今回0。

元n32の全候補は候補別PCG64 stream設計を持つが、共通replica番号だけで同時paired bootstrapを行わない。candidate内のtrajectory indexは一度だけ再抽選し、両軸および全ε/倍率/biasに共用する。候補間は独立stream仮定を明示した別抽選とするか、依存不明として手法を停止する。ゼロ共分散を保存証拠の実測値として発行しない。

## 必要な入力schemaとgate

1. immutable selection receipt/hash/SOURCE、union6、探索/確認登録、seed導出、event digest、compiler/profile。
2. `(candidate, trajectory_id)`が一意な192行。cosine/sine metric、axis fingerprint、SOURCE、phase、seedを保持。探索/条件付きpilotを混ぜない。
3. endpoint→候補別保存shot・eligibility・bias namespace。費用標本をendpoint数で複製しない。
4. 探索j_Eの確認評価と確認後j_Cを別出力。未知tail平均・population winner certificate・厳密biasはnull/false。
5. 手法・仮定・family/α・resampling seed/反復・停止基準・出力名を別契約で固定し、新統計実行承認を要求。

欠測/非finite、seed/event/axis/compiler不一致、phase混合、未承認・SOURCE drift、誤ったpaired認定、未知tailの0補完でSTOP。数値費用がPRに不利でも異常とはしない。結果に都合よく候補を除外・fit条件を緩めない。

## 参照と今回の判断の区別

[Efron (1979), Bootstrap Methods: Another Look at the Jackknife](https://blogs.helsinki.fi/bk-club/files/2012/05/Efron_Bootstrap_AS1979.pdf)はempirical resamplingの原典。[Fithian, Sun, Taylor, Optimal Inference After Model Selection](https://arxiv.org/abs/1410.2597v4)は選択条件付きerrorとdata splittingを扱う。これらをH8への適用証明として引用しない。[Bentkus, On Hoeffding's inequalities](https://arxiv.org/abs/math/0410159v1)はbounded independent variablesの不等式の前提を確認する参考。Hoeffding(1963)出版社ページは403で本文未取得と記録した。

本設計で述べたH8への限界・優先順位は、固定SOURCE監査と数式の前提からの技術的判断。新しい数値推論・追加科学レビュー・方法の科学的採択ではない。
