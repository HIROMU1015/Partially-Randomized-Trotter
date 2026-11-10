# H6 DF19 RPE D1：実装・人工検証・実行準備

2026-11-10 JST。**実装準備と人工テストを完了。実H6の新しいphase/radius/N・部分Gは未承認・未実行。公開照合後mandatory STOP。**

今回の指示書§8は「実装準備・人工テスト・実行計画作成」を許可し、実H6の数値評価には別途明示許可を要求する。この境界を適用した。D1数値結果、全RPE費用、PF/PRの科学的優劣は報告しない。

[実行計画](D1_PLAN.md)／[前回H6準備・保存4点](../h6_df19_rpe_preparation_20261110/REPORT.md)／[固定H4実装・計画][h4-report]。既存H6 branch `codex/resource-eval-h6-df19-rpe-preparation-20261110`を`99ad77ef949b70192918a8700b092dd33c17af8c`から継続し、H4のworktreeには書き込んでいない。

## 実装とH4からの差分

- [h4_df10_rpe_connection.py](../../../src/trottertracks/h4_df10_rpe_connection.py)と[元15件test](../../../tests/check_h4_df10_rpe_connection.py)を固定`e91021ae2db0fba755ba7a6404c18e7e003c129f`からファイル単位で取り込んだ。branch mergeや履歴変更は行っていない。
- `inspect_saved_point/bind_saved_round`の入力層に`expected_system/expected_rank/expected_binding`を追加し、defaultはH4/DF10のまま保持した。H6をH4に偽装しない。既存集計には明示的な`partial_rounds=True`の場合だけ非zero開始の連続部分roundを渡せるようにし、defaultのm0開始条件は保持した。ΣNCの算術は変更していない。
- [h6_df19_rpe_connection.py](../../../src/trottertracks/h6_df19_rpe_connection.py)はH6の保存H/snapshot/state/rank政策、固定4 candidateの構成を照合し、同じH4算術へ接続する薄い層。H4の`assess_supplied_round/failure_budget/aggregate_supplied_axes`、既存geometry/Hoeffding/horizonを再利用する。別solver、fit、sampler、回路builderは追加していない。
- `prepare_conditions`はcallerが明示したprior/h/β/energy/failure条件を受ける。prior不明ならhorizon計算前に未資格、alias余裕超過なら不成立を返し、hやβを自動変更しない。成立時だけ既存horizon関数と全軸一様αを用いる。実H6 targetでは今回呼んでいない。
- `assess_supplied_round`はstate/referenceを含むsupplied corrected diskとraw誤差・provenanceが必要。未知誤差を0にしない。DF-H targetのため`own_DF_target_no_DF_energy_shift`を固定する。
- `supplied_price_axes`は明示した1 cohortからX/Y価格欄をcopyするだけ。探索n9と確認n33をpoolせず、nやBを費用の倍率にしない。`aggregate_partial`はPFとPRを別々に扱い、h=.8のm0/1又はh=.1のm3/4の部分費用を既存集計へ渡す。included/missing roundsとpartial-onlyを明示し、欠けたm0..2や全horizon成功を無視しない。

上記入力・coverage変更以外のH4関数が同じASTであること、15件testが元ファイルと同じbytesであることを確認した。既存RPE decoderは元テストでtoy位相に対して再利用しており、新decoderは作っていない。

## 条件と未確定事項

| 項目 | 現在の扱い |
|---|---|
| 固定target | Linear H6／1.00Å／STO-3G／保存tol-only DF19。weighted-Hermitian completion、fragment順、scalar、H/state bindingを保持。保存DF-H自身のground energyを対象とし、元Hとの差をenergy reserveへ追加しない |
| 精度・失敗確率・資源 | 1 kcal/mol（既存CA定数）、各PF/PRでαtotal=.05、保存数値state、P=0。compiled RZが主指標、CX/各depth/size/peak幅は別量 |
| state証拠 | 前回の保存residual等を維持。`ground_state_certified=False`。小さいresidualだけで厳密ground、reference誤差0、unit radiusを認定しない |
| h・β候補 | h=.8が第一、.1が代替。βtotal=.4／βstat=.2が初期候補。今回は実H6へ適用・採否判定していない。失敗時に配分を変えてPASSへ誘導しない |
| prior／alias | 確認した既知H6 receiptには、数値E0から独立したenergy priorの根拠を見つけられなかった。保存E0から狭いpriorを新設しない。外部priorの出典と`max(abs(I))*h+βtotal<π`が閉じるまでh選択は未確定。[-10,10]をh=.8へ無条件に入れない |
| horizon／α | 実H6のM/Tmaxは未計算。承認後、採択h/β/εEから既存horizonで決め、α=.05/[2(M+1)]を初期配分とする。丸めでα和が上限を超えないよう下向き調整する。旧13/18round・αaxis=.025を自動採用しない |
| law・reference・disk | ideal/actual区分、fresh IID、reference/state誤差、normalization marginのraw誤差等を明示する。経験的allowanceは数値pilotでありrigorous boundではない。不足すればN=null |

ここでのprior不足は、入力・条件の未資格であり、H6物理計算が失敗した結果ではない。新たなstate solve・Hamiltonian生成・spectral certificateを今回の必須作業として増やしていない。

## 固定4候補と保存価格

| T | PF | PR | 価格の扱い |
|---|---|---|---|
| .8 | `76601611270ed9f0677d`：Full19/m5_best/q_PF1 | `2606efdab20e3b8522e5`：prefix5/m5_best/q_PF1/R32/K2 | 同candidate/compiler/full-wrapperの保存直接価格。PR探索n9・確認n33は別cohort |
| 1.6 | `a4f5ffc0f320644dadfa`：Full19/m5_best/q_PF1 | `2d2af6ca830dff14a695`：prefix7/m5_best/q_PF1/R16/K2 | 同上。PRのprefix/R変更を維持し、同一回路の単純時間倍率としない |

前回[保存projection](../h6_df19_rpe_preparation_20261110/SAVED_POINTS.json)は元のbytesを保持したまま、今回は4点に`inspect_saved_point`だけを適用した。新N・Gはnull。保存signal/reference/u/log B/margin/単価は再計算していない。q_PFとq_RPE、Cの単価とN、価格標本数nを区別する。

部分費用は別途実行許可後、Nと価格の条件が揃う場合のみ補助的に算定できる。未知のCI/covariance/peak幅はnullのままとする。全roundのG、価格model再fit、候補再最適化は今回の対象外。

## 人工テストと技術照合

**計25件PASS**：元H4の15件を変更せず再利用し、H6差分10件を追加した。

- [H4回帰15件の出力](h4_df10_ARTIFICIAL_TEST_RESULTS.txt)：元のphase/radius/Hoeffding、B一回、axis-N、union、decoder、direct/model、欠測、旧等N集計の一致。
- [H6差分10件の出力](h6_df19_ARTIFICIAL_TEST_RESULTS.txt)／[test source](../../../tests/check_h6_df19_rpe_connection.py)：passive import、固定binding/4候補、round別PR分割、H4 default拒否、q_PF/q_RPE、prior不明・alias超過時の非実行、toy horizon/一様α、未知diskのN=null、cohort分離、部分会計と欠測early round、PF/PR・energy targetの混在拒否。

数値testは人工signal・人工価格・人工energy精度／priorだけを使う。実保存H6のN・phase・radius・horizonをtestへ混入していない。実データに行ったのは入力copy/identity照合だけである。人工PASSを科学的性能・energy成功の証拠としない。

既存H6専用venv（Python3.11.0rc1）を変更せず使用した。H4保存testの元Python3.12とruntimeは異なるが、同じテスト本文がPASSした事実を示す。依存install、OS/共有設定変更、他process操作は0。再実行は適切な研究用Pythonで次の2つのみ（実H6データは読まない）：

```bash
python -B tests/check_h4_df10_rpe_connection.py -v
python -B tests/check_h6_df19_rpe_connection.py -v
```

## D2/D3への課題とSTOP

H4/H6共通に、外部priorと初期alias、数値state/referenceのerror disk、全roundのα/βとhorizon、law/実装の資格が必要。h=.8なら保存.8/1.6以降のT≥3.2、h=.1ならm0..2とm≥5等の未評価点を別に扱う。保存時間を補間・冪乗して埋めない。H6の全時間域の価格は未資格で、H4/DF12/CDF8の数値係数を移植しない。

次の科学scopeは[実行計画](D1_PLAN.md)の固定4点のscalar評価だけを候補とし、D2/D3・追加signal・compile・全RPE費用へ自動進行しない。

**実施済み**：実装、25人工test、4点の非数値inspection、保存SOURCE・入力・価格の対応確認、実行計画作成。**未実施**：実H6の新phase/radius/N/部分G、horizonの実target評価、eigenproblem、finite mean、sampling、build/compile、fit、RPE測定simulation、詳細科学レビュー。旧SOURCE/raw/結果/grant/ledger/marker/KEEP/STOPは不変。

公開remote SHAと主要成果物の全文bytes照合後**mandatory STOP**。次の担当：**ユーザー**が別途のD1科学実行を許可し、prior根拠・reference/state/lawの資格とh選択を判断する。**Codex**はその許可範囲の4候補をまとめて評価し、不足条件は未資格として報告する。承認証明書・grant発行は要求しない。

[h4-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e91021ae2db0fba755ba7a6404c18e7e003c129f/docs/research/h4_df10_rpe_preparation_20261110/REPORT.md
