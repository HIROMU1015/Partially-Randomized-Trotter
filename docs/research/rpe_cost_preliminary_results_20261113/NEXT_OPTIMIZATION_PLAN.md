# NEXT_OPTIMIZATION_PLAN：P0〜P4を受けた優先事項

2026-11-13 JST。[数値packet](DECISION_PACKET_V0.md)の条件付き結果を根拠とする提案。**新しい科学実行や候補の正式採択はしていない。**

## 1. 優先する研究判断

| 優先 | 担当 | 次の有限案 | 可能になる判断・現在の限界 |
|---|---|---|---|
| 1：全Nの資格 | ユーザー＋A/B | 元13roundを維持し、全事象loss/claim・target/raw/law移送・数値誤差を決める。既存whole-shot adapterとrisk研究を再利用 | 目標精度を満たすshot配分を定義できる。今回のnominalやS3点RMSEは十分Nになっていない |
| 1：baseline数値安定性 | A | A699178の失敗値保存とmacro-step/powerをSOURCE/人工入力で調査。必要なら例F2-COARSE/m12の代表1セルだけを別科学許可で診断 | baselineを数値gate失敗だけで除外せず、同RMSE比較に進めるか判断。全20再走・閾値緩和は自動実施しない |
| 2：長時間価格domain | D、law/構造はC | m12まで使うcandidateに対し、最大domainのcompiled価格資格を供給する有限案。既存価格検証計画の独立holdout/解析的囲い/必要セル直接CIを使う | 固定proxyのm10〜12が96.9198%なので、このdomainの未知が研究上重要。ただしnominalを正式なcompile予算重みにせず、必要N又は許容集合も併せて精度設計 |
| 2：少数whole-N配分 | B/A | リスク法の準備後、外部供給1候補×最大3 whole配分等の前提固定案。新MC/解析は別の明示許可 | shot総数でなく価格付き配分の比較が可能かを検証。早いroundもbranch依存があるので独立最適化しない |
| 3：初期PR対照1本 | A＋B＋C/D | 現構成をanchorに、共通κを1点変える対照を第一の提案候補とする。値/倍率はAとユーザーが決め、全schedule・law/domainを固定する | event価格とnormalization/bias/Nの競合を直接調べる。今回測ったκ感度ではなく、構造費用集中を受けた実験設計上の優先案。最適κや改善方向は未判定 |

1の二件は別担当の並行準備として扱え、D/Cの既存研究を止めない。Bから他チャットへ実行指示・作業変更を送っていない。価格methodの技術設計はN資格を待たず進められるが、正式なG精度への十分性は適格N又は有効なN集合に条件付ける。

## 2. 他の設計自由度

- **L_D/分割**：固定L_D1のP0〜P4では他prefixの優劣・感度は未測定。Aがhead/tail/λ/準備SOURCEを固定した1対照を供給できる段階で検討する。現tail価格が大きいというだけでL_D増を改善と断定しない。
- **決定論PF構成・幅**：S2/Y4/M5×.1/.4の既存baselineを保全する。短roundのθは異なるが全RMSE Nと長時間Cがないため順位を付けない。強いbaselineを残し、PFのqをPRと無理に一致させない。
- **PRのPF幅/formula**：現S2/.1以外の全N/Cは必要。PR-Y4はBで未接続という技術制約も残る。Track Aの有限T winnerへ置換しない。
- **finite K**：現K2から変更した時のraw/normalizer/価格law・必要Nは未取得。Kの1対照を調べる必要が生じたときだけ追加し、L_D×幅×κ×Kの直積探索を先に行わない。
- **h/M/budget**：今回固定。全13roundの不足がある間に未採択の探索変数へ自動追加しない。

共通κの対照を最初の提案にしたのは、現scheduleでS/T²と隣接S比が長時間の構造増大を記述し、event数とNの競合をwhole-scheduleで試せるため。これはL_D/K/幅より大きな改善が得られるという科学的結論ではない。担当の供給可能domainが異なる場合は、ユーザーが1対照を選び直す。

## 3. 追加価格精度が必要になる条件

同じRMSE claimを満たすNと同domainの価格区間が揃い、費用差区間が比較境界を跨ぐときに限り精密化する。既存`ΣN b`、共有価格に対する`Σ(N_left−N_right)C`、同時boxを再利用する。未知covarianceを0にせず、finite cohortと全lawmeanを分ける。

短roundθの例としてS2/.1 m4は約1.01127、Y4/.4 m4は約0.75985であり、必要N比/compiled補正が結論を変え得る。これを解決するのは同taskのNとCであり、一律の価格2%/95%採択ではない。D pilotの4.21%/11.09%という標本D/S比をモデル誤差上界やm12補正率にしない。

最大使用domainでの検証ができない場合は、解析的囲い・domain限定・必要round直接測定のどれかを選ぶ。m7/8/9を順番にcompileする必須段階は設けない。小q PF affineとPR T²を長時間compiled価格の認定へ代用しない。

## 4. 省略・延期できるもの

既取得PR全signal/構造kernelの再実行、73価格receiptの再監査、S3 trialの目的不明な再解析、新しい大規模model/solver、新8次PF、全prefix/K/R格子、一律の高精度compileを追加しない。これらを省いても、未知をnullに保てば研究の現状を正しく表示できる。

一方、全N/target/law資格、baseline長時間signalの数値問題、最大domainの平均compiled Cとその不確かさは、正式G比較の重要な不足であり省略してwinnerを作れない。

## 5. 今回の終点

P0〜P4は完了。本結果を公開照合してSTOPし、新signal/MC/compile/fit/PR候補最適化へ進まない。次は**ユーザー**が上記risk方針と必要な有限科学案を判断し、各**A/B/C/D**担当が許可された範囲だけを実行する。承認証明書や新grant発行は不要。
