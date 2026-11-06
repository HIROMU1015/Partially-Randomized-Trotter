# SCV 224-shot予備診断：科学実行契約草案

2026-11-07 JST。**未採択・実科学未実行。今回の実装許可を科学実行許可と扱わない。**
設計正本84e021bの第一候補を、[準備済SOURCE](SCV_IMPLEMENTATION_REPORT.md)へ対応付けた仕様案。

## 研究目的と固定target

最終目的は、元Hamiltonianの厳密基底状態を仮定するPR＋RPEで、所定エネルギー精度・成功確率に必要な総compiled資源の削減。状態準備費を除外する。
今回はその前段の**層内metadata CVがB1-OLへ追加効果を持ち得るか**の探索。エネルギー精度、成功確率、期待費用の保証付き比較ではない。

| 項目 | 固定対象・再利用元 |
|---|---|
| 保存モデル | linear H4、1.00 Å、STO-3G、legacy DF12、8 system qubits、全4電子number sector |
| 分割・PR | L_D=3、m0/q1/δ=.02/r1/K2、coefficient_atol=0、X/Y |
| finite target | N1 `744e1a09661ebba6841fe5faf456f88c41c47780` のactual finite-grid Q/selected full policy。全5正層とscalar/identity/event phaseを維持 |
| 状態 | 元Hの厳密groundが理論基準。実装にはN2 `5e9bd08068738cdf1070bdc43bee8194b113ae33` の保存保証stateを再利用。H_D ground/PF有効状態へ変更しない |
| engine | `fdf00cee3c87d804845497afc89ac6a71f3b3f59` のexact Fraction kernel/保存証明。b_engineの保存strict上界を再利用し、再認証しない |
| 範囲 | 理想gate・保存DFモデル限定。状態準備、実機ノイズ、全18round、連続理想lawへの同一性保証を含まない |

旧hash・state・価格・bias・shot数を移植しない。saved-state数値誤差と理想ground仮定を区別する。
独立uniform53-bitモデルとPCG64実装の差は従来の条件付き保証を維持する。

## 群選択・identityの先行固定

1. 公開された新activation SOURCEと環境/旧SOURCE、N1 LAW/PREPARATION/catalog、N2 state/witness、engine証明のbyte identityを照合する。今回の [SOURCE.json](../../../artifacts/scv224_preparation/2026-11-07/SOURCE.json) が参照先とhashを保持する。
2. LAW/catalogを一回だけpure binderへ渡し、既存label→fragment、全5正層、products→rotation末尾labelを確認する。CDF/preparation再生成0。
3. 正質量fragmentを文字列辞書順に並べ、非空proper subsetからν_1∈[1/8,7/8]、|ν_1−1/2|最小、同点はbit i=canonical index iの整数mask昇順で一群Sを選ぶ。rule変更・別featureへのfallback禁止。
4. exact ν_h、E[u²|h]=ν_h、Σ_h=ν_h(1−ν_h)、S、law/catalog/H/preparation hash、control identityを保存し、train開始前に凍結する。実H4ではこの段階自体が**新しい科学数値評価**であり、今回実施していない。
5. 適格群なし・対応不一致なら測定前STOP。νやΣが得られた後も応答との相関は未知。

未知bit、旧397/B1339/752のbit・係数・word・cache、state/Born/真のpath信号/真varianceを群選択やCVへ渡さない。

## 固定quota案と独立性

| 段階 | X | Y | 取得方法 |
|---|---:|---:|---|
| h1 train | 32 | 32 | fresh直接Q_1 |
| h0 evaluation | 4 | 4 | fresh直接Q_0 |
| h1 evaluation | 64 | 64 | fresh直接Q_1 |
| h2 evaluation | 4 | 4 | fresh直接Q_2 |
| h3 evaluation | 4 | 4 | fresh直接Q_3 |
| h4 evaluation | 4 | 4 | fresh直接Q_4 |
| 合計 | 112 | 112 | **224 physical shots** |

評価160件を両推定器へ共通利用するpaired設計。別々の二campaignではない。allocation probe/calibrationは0、固定quotaを学習後に変えない。
各軸のintercept OLSは一回、rank2・finite slope・D_1=2+|λ_1|≤4を提案。実際のbinary64 slopeをFractionへ固定してguardし、clip/refit/追加測定を行わない。
他層λ=0は測定前固定。学習失敗後のfallbackではない。X train→Y train→両係数凍結→X eval→Y evalを提案する。

新campaign候補 `O2-SCV224-H4-m0-q1-20261107-v1`、seed候補 **20261109**。いずれも未採択・科学PRNG未生成。
旧06/07/08系列から区別する候補であり、異なるseedだけを数学的独立性の証明としない。

既存`Domain`を取得tagとして維持する：

- train：A1/X, Y/train/1 の2 domain。既存workerはこのtagをB1-OL条件付き取得へ写す。
- eval：B1-OL/X, Y/prod/0…4 の10 domain。`prod`はfresh診断評価のschema名。
- 12 tokenを辞書順で固定、各tokenにpath child→bit childの順でPCG64/SeedSequence子streamを与える（24子stream）。campaign/purposeは新SOURCE/outputにbind。
- 一trial一claim、path word≤64、独立measurement word1。範囲外word、budget超過、stream/handle/trial再使用はSTOP。

このseed/domain/spawn順、range guard、quota、campaignを次回明示採択する必要がある。承認証明書・grant発行は要求しない。

## 指標と解釈

各軸でB1-OL=Σp_h B̄_h、SCV=Σp_h[B̄_h−λ_h(ū_h−ν_h)]を同一評価recordから合成する。
層別raw/residual sample variance、feature count、all-equal、rank、係数、rangeを保存する。

\[
\widehat V_{OL}=\sum_h p_h^2 s_{B,h}^2/n_h,\quad
\widehat V_{SCV}=\sum_h p_h^2 s_{Z,h}^2/n_h.
\]

paired共分散Σp_h²s_BZ,h/n_hと、差の分散点推定Vhat_OL+Vhat_SCV−2Covhatも保存する。
h≠1の差は標本ごとに0。h1の非零相関・改善の有無は取得までUNKNOWN。
raw/residual全同値や低出現countを明示し、sample variance0を母分散0や十分powerと扱わない。
軸間の不一致・不確かさを保留として記録できるようにし、結論を得るための追加shotは行わない。
科学的GO/NO-GOの閾値は採択しない。新しいε/α、Bernstein/Hoeffding十分N、期待費用改善、CV一般優位性は主張しない。

## 全費用と価格

固定価格profileはQiskit1.3.0、basis=(rz,sx,x,cx)、optimization level2、seed17、selected full no-prep。
state準備だけを除外し、全train/evalのfull interrogationを課金する。cache hitも量子RZを全額計上。

- 実physical campaign支出：train64＋eval160を一回ずつ。
- B1-OL単独を同quotaで行う参照：eval160のRZ。probeなしの固定quota版baselineで、最適配分baselineを超えたとはしない。
- SCV手法全費用：train64＋eval160のRZ。学習を無料にしない。
- 二つの参照費を合算して実支出にしない。Vhat×各手法参照実現RZは探索的point scoreだけで、期待費用保証ではない。

driver wall/fit/補正、metadata setup、path生成、exact engine、price/cache/compileのworker時間、parent/worker RSSとouter wallを別記録。
並列worker時間の和とwallを区別する。古いTrialCost.classical=0 placeholderを実測0と扱わない。
未取得価格・未完了tail・未計測費用はUNKNOWN。bitを価格処理前にfsyncする既存workerを維持。

## 有限science call範囲案

| 操作 | 次回の最大範囲案 |
|---|---:|
| LAW/catalog群選択・ν/Σ表 | 一回、metadataのみ、測定前freeze |
| saved bundle decode | 一回 |
| preparation/CDF・state solve/eigh/spectral・engine再認証 | 0 |
| exact engine/Born/measurement | 各224 trial以下 |
| selected full build | 448以下 |
| compile | 224以下 |
| path word / trial・測定word / trial | 64 / 1 |
| 総word | 14,560以下 |
| science retry・追加campaign | 各0 |

実pathの全列挙、未取得pathのcompile、別feature、B0/B1-O、別分子/roundへの拡張は0。
少数shotというだけでexact engineが軽いとは保証しない。既存resource admission/qualified managerの非干渉条件を確認し、人工的なCPU/RAM/worker上限を新設しない。
float/GPU kernel置換なし。自動再起動なし。既存manager仕様の適合性をSOURCEへ固定する。

## 起動前の最小activation・STOP

1. ユーザーが224-shot科学scopeと上の未採択設定を明示採択・実行許可する。
2. 新activationは既存saved-service loader、pcg64 stream、qualified manager/runtime/poolを生成し、準備済`connect_legacy_pipeline`と`Driver`へ渡す薄い接続に限る。新SOURCE/依存環境/input/設定を実入力前に公開照合する。
3. fresh output/cache、無既存STOP・無既存invocationを確認。SOURCEはgrant/承認証明ではなく科学来歴、invocationは二重実行防止。
4. metadata数値段階を一回、control/table identityを保存→train→freeze→fresh paired評価。RNG/engine/価格のcallは同じtrial journalにbindする。
5. group不存在、identity/support/mean/phase/bit/stream/word/価格不一致、rank/range失敗、call cap超過、非干渉不成立はSTOP。条件緩和・再fit・追加shot・再測定・科学retryで補完しない。
6. STOP要求をworker回収前にdurable保存し、既存pool終了/reconcile/attempt分類で全tailを保全する。旧実験は再開しない。
7. 完了・中断いずれもcommit/push、remote本文/blob/tree照合後mandatory STOP。保証付き本検証・追加campaignへ自動進行しない。

今回の成果は人工検証済のprepared driverと草案。実group/ν、science seed/stream、実bundle接続、実測/compileは次回の明示科学許可が必要。
