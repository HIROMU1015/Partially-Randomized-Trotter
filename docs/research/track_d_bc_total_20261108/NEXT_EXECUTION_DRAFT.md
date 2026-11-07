# 次段階案：長q支配を踏まえた最小検証（未実行）

今回のA/F/S点推定は全6資源のReference減少で一致したが、絶対総資源の約99.5%が未検証長q価格に依存する。従って、条件付き会計の実装を繰り返すことより、**長q価格の独立証拠とモデル精度**を得ることが次の主要な判断材料になる。[結果](REPORT.md)。

## 1. まずユーザー/GPTが主張水準と必要精度を決める

現在の結論は同じ保存ideal十分N・同じ物理条件・A/F/S外挿モデル内の点推定。RZの削減約1.1%はCX約11%・total depth約54〜55%より小さい。どの資源を重視するか、方式別の価格biasに対してどの程度の精度が必要かを決める。今回の結果から新しい主資源・許容幅・確認的成功条件や標本数を自動採択しない。

## 2. 最小の新価格検証候補

ユーザー/GPTの新しい明示許可後のみ、C7-4の既存案に沿うq64/cut32の独立whole-path domainを先に検証する。fixture/seed/law/cut/価格モデルと診断familyを結果前に固定し、強い対照・既存融合/位相・一つのwhole-path Referenceを保つ。先頭固定pathで長q数値安定性をpreflightし、raw correctness PASS後だけ最大範囲内のP0/postcheckへ進む。既存SOURCE/精度は無断緩和せず、失敗後のpath除外・追加標本・retryをしない。

q64で新domainへの局所移送を調べてもq4096までが検証済みにはならない。結果とモデル機構に応じ、q128以上または複数の長qのどの証拠が必要かを次に判断する。全roundを大量compileすることを自動必須にしない。モデル選択・fit更新を行うなら、今回のA/F/S固定結果とは別版とし、fitに使った標本を独立holdoutへ戻さない。

## 3. 既存統計の追加利用は別scopeで選ぶ

点推定を変更せず校正不確かさを明確にする場合、直接平均とモデル係数/複数scenarioのjoint情報の回収または明示した共通再標本化が候補。ただし新bootstrapや旧科学価格の再解析は今回行っておらず、別許可が必要。joint校正が揃っても長q model bias・law距離を解決したことにはならない。

## 4. Actual保証と方式別N

actualで同じenergy/failureを主張するには、B5/B6bのfinite-bit/joint-law、native/compiled/χ/actual inverse/leakage、全使用roundのraw誤差を正半径・phase/coordinate予算と旧整数Nへ接続する必要がある。両方式を同じ保守的budgetへ含められれば共通N、異なるbudgetが必要なら方式別Nを同じtarget/failureで評価する。そのN算定や新boundは今回行っていない。

actual保証完成を理想条件付き資源研究の絶対前提にはしない。一方、actualを対象にする段階では同じNが無条件に使えると推定しない。routing/T/noiseや元H ground solveへ自動拡張しない。

## 5. 別条件・元PR最適化への接続

今回のReference対strong比較を、別CDF/別Hamiltonianの削減や決定論PFに対する最適性へ移送しない。長q価格・Nの資格と主資源の採択後、同じ精度/成功条件でPRパラメータ/決定論PF対照を接続する別計画が必要になる。

**次の担当：ユーザー/GPT**が優先資源・主張水準、q64等の独立検証かjoint不確かさ処理かactual資格かを判断し、対象SOURCE/input/科学call範囲・STOP条件を採択する。**Codex**はその新しい実行許可後のみ進む。今回のcampaignは公開照合後mandatory STOP。
