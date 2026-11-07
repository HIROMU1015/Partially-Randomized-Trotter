# Signed support fresh pilot：具体的有限契約案の報告

2026-11-07 JST / SERVER1。**設計と4 Markdownの公開だけ。新しい科学計算・保存値の数値再解析・SOURCE実装・人工テスト/benchmarkは0。提案値は未採択。**

## 第一候補と研究上の目的

metadata-only結果b4512675の構造的ELIGIBLEを踏まえ、次は独立trainで凍結したh1 scalar係数がfresh評価で層内残差分散を減らすか、学習費込みで有望なpoint傾向があるかを問う。既知のΣ1>0を相関/改善証拠へ格上げしない。

対象は保存linear H4/1.00 Å/STO-3G/legacy DF12/8qubit/全4電子sector/L_D3/m0/q1/δ=.02/r1/K2/coefficient_atol0、N1 actual Q/selected full、N2保証ground、認証済みexact Fraction engine、理想gate、X/Y、full no-prep。

| 項目 | 第一候補（採択待ち） |
|---|---|
| Train | h1だけ64 shots/軸、計128。軸別intercept OLS各一回、Fraction freeze |
| Paired評価 | 全5層(8,128,8,8,8)/軸、計320。baseline/SCVで同じfresh recordを共有 |
| 総physical上限 | **448 shots、一campaign** |
| Seed/stream | **20261110、PCG64/SeedSequence、12 domain/24子stream**、token辞書順、path/bit分離 |
| Guard | **D1=2+2abs(λ1)≤6**。他4層λ=0。rank/finite/guard失敗は全campaign STOP |
| 評価解析 | full＋trial ID固定half A/B、各軸3、計6回。追加shot/価格なし |
| 価格 | Qiskit1.3.0、rz/sx/x/cx、opt2、seed_transpiler17、selected full no-prep |
| 主要上限 | decode1、engine/Born各448、selected full build≤896、compile≤448、fit2、word≤29,120、retry0 |

train32/軸・同evalの384-shot案を低費用の代替として比較したが、第一候補はscalar学習とh1診断を厚くする448案。保存momentから最適train量や検出力を求めておらず、旧224条件の無条件移植でもない。将来の費用/RSS/wallは未評価で、旧実現値から新性能を推定していない。

## Paired診断と費用の読み方

全5層のmean、raw/residual sample variance、V̂=Σp²s²/n、paired covariance、平均差variance、signed feature count・rank/λ/rangeを保存する。固定half診断は方向の不安定さや小さな点差を保留する補助screenで、CI/有意差ではない。X/Y両軸改善なし/片軸/両軸、all-equal/不安定、guard失敗、分散point改善と学習費未回収を区別する。

同quotaではSCVの追加量子費用はtrain分。共通evalは一度だけ課金し、B1-OL参照=eval、SCV参照=train＋eval、実physical=train＋evalを分ける。cache hitも毎shot課金。variance×実現RZのpoint scoreが低くても、精度保証付きshot削減・期待総RZ改善とは呼ばない。最適B1-OLへの優越や全round RPEの結論は対象外。

## 再利用と必要な最小差分

既存signed.fit_h1/SignedFrozenCV/paired_evaluate、sampler/Domain/Streams、SavedEngine/Prices、worker/group/journal/reconcile、費用会計、qualified持続managerを再利用する。新OLS/engine/回路builder/会計framework/managerは不要。

重要な未実装差分は、**保存momentのtyped復元、新purpose/quota/guardの薄いcollector・orchestration、STOP/call予算/timer境界、既存managerへのactivation**。通常SignedControl(metadata)はmomentを再計算するため使わず、固定resultをidentity付きで復元する。旧Driverのindicator balance・range・224-capはsignedへ適合しない。これらを小さいadapterで扱い、旧SOURCEは変更しない。

Wは実資源・affinity・pressure・残taskに応じて設定する案で、恣意的な低いCPU/RAM/worker上限を置かない。parent一回decodeをforkで共有し、worker再decodeや未認証O1/O2 kernelへの差替えをしない。独立namespace/cacheを使い、他job/共有設定は操作しない。今回はmanager/workerを起動していない。

## 成果物と根拠

| 文書 | 内容 |
|---|---|
| [SIGNED_SUPPORT_FRESH_PILOT_SCIENCE_DESIGN.md](SIGNED_SUPPORT_FRESH_PILOT_SCIENCE_DESIGN.md) | 既知の構造的適格性、理論条件、train/配分/guardの理由、point診断と限界 |
| [SIGNED_SUPPORT_FRESH_PILOT_EXECUTION_DRAFT.md](SIGNED_SUPPORT_FRESH_PILOT_EXECUTION_DRAFT.md) | 第一候補の具体値、Stage A–D、finite call/word、費用/出力/STOP、採択事項 |
| [SIGNED_SUPPORT_REUSE_AND_ACTIVATION_PLAN.md](SIGNED_SUPPORT_REUSE_AND_ACTIVATION_PLAN.md) | fixed input/hash、実在APIと不適合箇所、保存control復元、次回の最小adapter/人工検査 |
| 本REPORT | 次の判断・保全・実行境界 |

固定資料はmetadata結果b4512675/科学SOURCE71778dae、signed実装b23b3c0、旧SCV224 SOURCE996e434e/結果ba483ab3、基盤560c6092、confidence接続af47cf9、N1/N2/engine。正式共通指示§8.1/8.2/8.4/10/10.3、適用AGENTS/PROJECT_MAP、研究概要・現状、既存契約/STOPと関連test SOURCEを読んだ。既報126 PASSは引用で、今回は一件も実行していない。新しい重要科学レビュー・一次文献再調査も開始していない。

## 次の判断・承認・mandatory STOP

ユーザーが上表と詳細草案のscope、seed/domain、guard、診断、価格/call/STOPを採択するか判断する。採択後のSERVER1 Codex担当は、最小実装・人工検査・新SOURCE/env/input/manager固定公開と、明示許可された一campaignだけの実行。実装/人工確認と科学実行の許可を今回の設計依頼から推定しない。追加のgrant/承認証明書は要求しない。

次回はidentity・stream・rank/nonfinite/range/guard・価格・保存・call上限・非干渉不成立で全campaign STOP。STOP前のin-flightだけを回収し、bit/価格欠測・未完了・全train費を保全する。完了でも公開照合後STOP。科学retry・追加shot・別seed/control・cal/保証付きproduction・重要科学レビューへ自動進行しない。

今回は独立worktree `/tmp/scv-signed-support-fresh-pilot-contract-20261107`、branch `codex/scv-signed-support-fresh-pilot-contract-20261107`、base `b4512675ae1d650858256547d359210361d77f68`。**本directoryの4 Markdownだけ**を追加する。元main・既存dirty/untracked・旧SOURCE/raw/結果/grant/ledger/marker/STOP・C1凍結と並行O1/O2作業を保全し、旧索引/概要/manifestの科学的statusは書き換えない。

4文書の公開SHA・本文/blob/treeと旧原本無変更を照合後、mandatory STOP。実装・人工検証・科学計算へ進まない。

公開前の保全照合はPASS：元main既存3,087ファイルとHEAD/status、旧SOURCE2,923、metadata完了output9と科学SOURCE lock20（元のlive worktreeも一致）、旧停止研究1,086・前pilot29・SCV224 live raw530ファイルのbytesを確認した。検査はbyte/来歴と文書リンク・表構造・提案予算の四則検算のみで、実H4値の新集計や人工test suiteの実行を含まない。別O1/O2の作業領域・process・cacheは調査/操作していない。
