# Signed supportの次回科学scope：未決事項と限定差分

**未採択の準備案。今回の科学実行なし。新しいshot数・seed・数値range guard・精度/失敗確率・予算は設定しない。**

[実装](SIGNED_SUPPORT_IMPLEMENTATION.md)と[人工correctness](SIGNED_SUPPORT_CORRECTNESS.md)が完了した。次は、signed supportが元のB1-OL後に残るpath依存変動を説明し、学習費込みで追加利益を持ち得るかを問う。旧SCV224の科学契約を複製して採択するものではない。

## 維持する既存target

将来も固定保存linear H4/1.00 Å/STO-3G/legacy DF12/8 system qubits/全4電子sector/L_D3、m0/q1/δ=.02/r1/K2/coefficient_atol=0.0、X/Yを対象とする案。N1 `744e1a09661ebba6841fe5faf456f88c41c47780` のactual finite-grid Q・5正層・selected full policy、N2 `5e9bd08068738cdf1070bdc43bee8194b113ae33` の保証ground、engine認証 `fdf00cee3c87d804845497afc89ac6a71f3b3f59` のexact engineを再利用する。

N1/N2/engine認証を再計算せず、同じ測定規約・targetを保持する。旧224/752/中断experimentのbit・係数・stream・cacheを新学習/評価へ混入させない。新しいcampaign/output/cache/SOURCEを別途固定する必要がある。

## Signed supportへ変更するために必要な差分

| 項目 | 今回できたこと | 次回の実H4で必要なこと・採択待ち |
|---|---|---|
| 特徴 | f(l)=保存sign×supportサイズ/2。係数絶対値・角度・state/信号は不使用 | 保存catalogとlawの完全identityを照合し、この定義だけを適用する |
| metadata・moments | exact一位置和と全5層ν/m₂/Σを人工lawで確認 | 実H4の新特徴・ν/m₂/Σ・h1非定数性の限定評価を別途明示許可する。結果UNKNOWNのまま開始し、定数なら測定前STOPする仕様を定める |
| 活動層 | h1だけfit、他4層λ=0 | h1への限定を最初の科学問いとして採択するか判断。別control/活動層への自動変更なし |
| 学習 | 既存OLS一回、rank2/finite、Fraction freeze | 新しい独立train量と取得法、rank不足時の停止を先に固定。旧group balance規則は使わない |
| range | SOURCE box[-1,1]、actual λからwidth=2+2|λ|、全値包含検査 | guardの要否・数値・超過時処理を学習前に採択。旧indicatorの幅や旧guardを移植しない |
| fresh評価 | 共通recordによる全5層paired baseline/SCV、明示quota必須 | 全5正層の新しい配分と取得量、trainから独立したstream方針・seedを事前固定。過去quotaを自動採用しない |
| 実行接続 | pure adapterのみ。科学runner/managerなし | 既存pipelineへsigned purpose/input/feature/rangeを接続する最小activationと人工テスト、固定SOURCE/環境を先に公開照合する |
| 費用 | 既存会計の人工確認。train・共通評価・cache課金を区別 | 全stageのselected full compiled RZ、学習費・古典moment/feature/fit/cache/engine時間・memoryを記録。新しい価格取得call範囲を固定する |
| 診断 | paired mean/variance/covarianceの計算接続のみ | X/Y相関・残差分散・有望性判定規則を測定前に定義し、探索的診断と保証を分ける |
| finite confidence | 今回はcalを拒否するadapter | af47cf9の未完了接続に加えsigned rangeを反映する独立cal・層別bounds・失敗配分・十分整数quotaの別設計/承認が必要 |

## 最小の段階案とcall境界

1. **Metadata段階**：既存保存law/catalogのidentity照合後、SignedControlを固定入力に対して生成し、全5層momentとh1の正質量feature非定数性を保全する。新規ν/Σ等の実科学数値評価なので、今回の実装許可には含まれない。必要な入力・関数・有限call範囲・出力・STOPを次回指示で定める。失敗を別featureへ切り替えて救済しない。
2. **Fresh学習段階**：採択済み条件の独立conditional trainからh1のOLSを一回fit/freeze。rank・range・actual係数を先行保全。clip/refit/追加shotを自動実行しない。Born/state/真のpath信号はCV側へ公開しない。
3. **共通fresh paired評価段階**：凍結後にB1-OLとSCVを同じ全5層recordで評価し、baseline・補正差・分散診断・全費用を記録する。学習費を除外しない。未取得価格や失敗trialはUNKNOWNで保全する。

段階をまとめて承認するか、metadata段階後に測定仕様を決めるかはユーザー判断。有限scope・データ量・seed/stream・guard・診断条件・科学callの内訳は、実行前に具体化して採択する必要がある。本書は空欄を勝手に埋めない。

## 公平性と結論の範囲

共通評価は対応のある比較で、独立した二method campaignではない。SCVのphysical費はtrain＋共通評価、baseline referenceは共通評価費と分けて示す。配分を固定したpaired診断と、それぞれを最適配分したpolicy比較は別問題。小さなsample variance点差や実現RZ差を期待費用・精度保証に読み替えない。

metadata CVはpath固定時のHadamard shot noiseを消せない。oracle最適係数の可能性、有限trainで凍結した係数の実効性、guard/calibration費まで含めた十分N、全費用改善を区別する。今回の人工PASSから実H4の相関や一般的SCV優位を推定しない。

## STOPと次の承認

identity/source/feature/support/stream/weight/価格の不整合、h1定数・rank不足、非finite、不正residual range、事前採択したguard/有限call境界、共有環境非干渉不成立を停止条件として先に固定する。旧STOPを解除せず、新しいseedによる科学retryや事後control選択を許可しない。

次の承認対象は、**実H4にsigned supportを適用する限定metadata評価と、それに続けるかを決めたfresh train/paired評価・必要価格取得の具体的有限scope**。shot数・seed・range guard・配分・失敗処理を含めてユーザーが判断する。今回、grant発行・承認証明書・新manager・重要科学レビューを要求/開始しない。
