# N5-A終了後：最小実行scopeの候補

2026-11-08 JST。[REPORT](REPORT.md)。**提案・未実行。N5-Aの許可は新しい科学数値評価／provider／合成／compileへ及ばない。Mandatory STOPを維持する。**

## 1. 先に返す判断

**GPT／ユーザー**：短roundの既存policy移送を先に閉じるか、全18roundの主な障害である域・precision・emitterを先に扱うか判断する。全18round一括実行を既定案としない。新しい重要科学レビューは、その開始承認がある場合だけ担当GPT／Claudeが行う。

今回の静的照合で解消した項目は、保存条件表、literal／域／guard候補、S<=1のSOURCE構造、m7結果の限定、総Tでのshot重みの注意。残る実計算をこのscopeのまま進めない。

候補L/Dは、評価規則や変更を許すpolicyがまだ未固定の設計判断候補であり、本書だけを数値実行仕様にしない。候補Sは既存policyと1 roundの保存入力に限定できる具体案である。いずれも実行開始はしていない。

## 2. 候補L：長roundのprecision／emitter実行可能性の限定検討

| 項目 | 提案 |
|---|---|
| 担当 | GPT／ユーザーが科学条件を選び、承認後にCodexが限定技術検証 |
| 最初の対象 | 保存m17・両arm×X/Y。新しいN／alpha／precisionはまだ選ばない |
| 固定入力 | 本N5-A公開commit、CT-0のm17 cap・旧disk／counts／shots、保存logical record候補、C0/85 certificate、N3/N4の量子化・guard・emitter・shell SOURCE |
| 必要な作業 | 既存guardとemitterを正しい単位で接続する、有限の実行可能性再会計のscopeを具体化。新しい角／basic keyの設計や取得前に、保存証拠で何が不足するかを確認 |
| 承認 | 保存値の新しい誤差和・precision実行可能性数値評価には別途明示承認。単なる文書の静的確認だけなら追加科学実行に含めない |
| 判定 | 既存証拠で保存Nの条件を資格できる／上界が足りない／新policyまたは保証補強が必要、を区別。必要精度を保存十分capから断定しない |
| STOP | 固定した有限再会計の完了、保存誤差の不足、現証明の不適合、新規合成・emitter設計・新条件が必要となった時点 |

これは合成runnerの起動提案ではない。現在のemitter項がcapより大きいケースでは、基本wordを再合成するだけの案を先行させない。どの保証・allocation／policyを許すかは科学判断へ戻す。

## 3. 候補D：最初のvacuum域境界m8の限定検討

| 項目 | 提案 |
|---|---|
| 担当 | GPT／ユーザーが域／policy変更の可否を判断し、Codexがその固定scopeの調査／検証 |
| 対象 | m8/q256/r2、保存75 bits／199 shots/axis。全18roundへ拡張しない |
| 固定入力 | 保存vacuum包絡114.42740240900511、現provider L、43基本word／C0、r2の中央・event・scalar metadata、SOURCE recurrence／C_phi |
| 作業 | 新たなreachable角のsamplingではなく、現証明で域が不足する理由と、既存wordを再利用する場合の必要条件を具体化 |
| 承認 | 新しい包絡や構成可能性の数値評価、provider適用、policy実装、合成、guardはそれぞれ具体的な固定入力・call範囲への別途承認が必要。今回未固定のkeyやprecisionを推測して進めない |
| 判定 | 域内の既存証拠で閉じる／証明不足／新しい域・policyが必要を区別。包絡超過を到達証拠としない |
| STOP | この1 roundの固定検討完了、保証不足、未許可のpolicy／合成／compile／samplingが必要な時点 |

新規基本wordの必要性は未確定。単にj範囲やbを伸ばす、modを入れる等を採択していない。m11以降のB′ scalarの域外literalも、最終的な全18round policyには別途扱う必要がある。

## 4. 候補S：短roundの既存policy移送を1 roundだけ確認

| 項目 | 提案 |
|---|---|
| 担当 | ユーザーの明示承認後にCodex |
| 対象 | m0/q1/r1・両arm×X/Y、保存66 bits／192 shots/axisのみ |
| 固定入力 | 現43基本word／C0／85 certificate、保存m0 B′ scalar key `0x1.86e032ee41d40p-5`、保存vacuum域、q1のF/G SOURCE、CT-0 m0条件、N4 consumerと会計SOURCE |
| 作業 | 域内の既存providerによる必要最小限のscalar role接続と保存データ再会計。既存q1特別分割を維持し、shell＋両軸max＋旧diskで192の十分性を確認 |
| 承認 | Provider/certificate呼出し、必要最小限の局所技術確認、保存入力RPE再会計への明示実行承認が必要。新規基本合成／guard／native compile／sampling／価格集計は既定scopeから除外 |
| 判定 | 固定policyとSOURCE条件下で192が十分／未資格。新しいNやalphaを探索しない |
| STOP | 1 roundの指定確認完了、SOURCE/phase/域/予算不一致、対象外の科学callが必要となった時点 |

この短round検証が成功しても、長roundの域・precision問題は解消しない。全18round目的への優先順位は候補L/Dとの比較でGPT／ユーザーが判断する。

## 5. 価格計算は別の限定scope

Round別pointwise条件や総Tを評価する段階では、適合するprecision／実行policy・RPE資格を先に固定する。次に1つのr変更round等のD／scalar／中央word価格・vacuum capを対象として、保存価格の再利用と必要な新計算を分けた有限scopeを作る。現在は対象key・policyが未確定なので、価格取得や改善率の数値計算を開始する案を固定していない。

統括兼Dのnative CX/RZ/depth、q64検証、13-round adapter、回路再設計を上記候補へ含めない。将来そのSOURCE／公開証拠への接続が必要なら同成果を引用し、重複実装を避ける。必須資料が未取得なら欠損を記録し、推測で通過させない。

**次担当はGPT／ユーザー。各候補の採否と、必要な限定実行承認を判断する。Codexは全18round、合成、価格、N5後続へ自動移行しない。**
