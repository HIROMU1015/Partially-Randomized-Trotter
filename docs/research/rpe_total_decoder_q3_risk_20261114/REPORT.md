# 全関数RPE decoder（Z1＋T1）と直接モデルMSE（Q3）評価器の技術実装

2026-11-14 JST。Claude Code（主agentのみ。独立した査読ではない）。

W0（`104dacdc6`）の提案のうち、全関数decoderとrisk評価器を実装し、人工データで検証した。

| 項目 | 内容 |
|---|---|
| branch | `codex/rpe-accuracy-contract-w0-20261114`（W0と同じ研究線。基点`3a27f99fe`はB `ce0517de6`とI1/I2/E1修正を含む） |
| 実装commit | `710aa69b663ef3313439762d25c80a652a03948f` |
| 追加したfile | 新規3 file：`src/trottertracks/rpe_total_decoder.py`、`src/trottertracks/rpe_total_risk.py`、`tests/check_rpe_total_decoder_risk.py` |
| 既存file | **1行も変更していない**（`git diff 3a27f99fe -- 既存decoder・DP・controller`は空） |
| 証拠 | [人工検証](SYNTHETIC_EVIDENCE.json)（[script](SYNTHETIC_EVIDENCE.py)）、[回帰](REGRESSION.json)、[取込みpatch](total_estimator.patch)、[取込みと科学評価案](HANDOFF.md) |

**実施していないこと**：

- 実H4/H6の新しいDP・MSE・N計算
- 保存DP結果（B run01）の数値再解析
- Monte Carlo
- error contract・canonical ID・既存decoderの変更
- 正式契約への反映
- Bの作業・processへの介入

**この実装だけを根拠に認定していないこと**：Z1＋T1の精度優位、十分N、Q3による物理的な誤差保証。

## 0. 要旨

1. **Z1＋T1を別versionの推定量として追加した。**
   - ID：`consecutive_dyadic_nearest_branch_principal_energy_total_v1`
   - 旧decoderが解決する履歴では出力がbit一致する（人工7,136履歴＋無作為5,200試行）。
   - zero/tieでも必ず推定値を返し、zeroで保持したround・tieでargminを採ったround・旧decoderならどのstatusだったかを記録する。
2. **Q3型の直接モデルMSE評価器を追加した。**
   - Bの`evaluate`・`_bound`（count表、閉包証明、coarse上界、部分DP）は変更せずにそのまま呼ぶ。
   - その上で、全関数版専用に次を加えた：
     - 更新幅補題による軽量上界
     - zero/tieを状態遷移として扱う状態集約DP（状態依存の残余上界と、E[Ê]・E[Ê²]を含む）
   - 完成した有効な上界どうしのminだけを取る。
3. **人工の全数列挙とDPがexactに一致した**（W0の8法則、計11,748履歴）。
   - 一致した量：MSE、平均、分散、bias²、旧decoderの解決分寄与η、未解決確率、roundごとのzero・tie質量。
   - 部分追跡（64通り）でも、上界・下界とモーメント区間がすべて真値を挟んだ。
4. **主値境界guardを外すと上界が破れることを確認した**（負の対照）。
   - 補題上界：0.0405 < 真値0.1384
   - DP残余上界：0.0982 < 真値0.1384
   - family上端がL_D未満のときは、Bの部分DP上界も全関数版の真値を下回る（0.00965 < 0.0240）。そのため統合から除外する。
5. **軽量上界と状態集約DPの役割**（人工M=12での比較）
   - 補題上界はcount表の作成後0.1秒程度で求まるが、緩い（coarseの約1/20〜1/3）。
   - DPは数十〜百秒かかるが、上界と下界の幅を相対0.2〜2.5%まで詰め、bias²と分散も分けて出せる。
   - したがって、補題は保証の床（常に有効な上界）、DPは設計・合否判定の主手段という分担になる。

## 1. 新decoderの仕様・versionと旧decoderとの差分

| 項目 | 旧（`..._energy_v1`、部分関数） | 新（`..._energy_total_v1`、Z1＋T1） |
|---|---|---|
| count→位相 | exactなFraction→binary64、`math.atan2`、参照`wrap` | 同じ（`F._wrap`、`math.atan2`をそのまま使う） |
| 候補・選択 | `F._nearest`（scalar規則、最小距離、完全同値はlowest branch ID） | 同じ関数を1遷移kernel `T.step`として使う |
| zero（X=Y=0）、m=0 | `UNRESOLVED_ZERO_COORDINATES`、energy=None | θ̂_0=+0.0（符号付きゼロを区別し、+0.0を明示する） |
| zero、m≥1 | 同上（tieより優先） | θ̂_m=θ̂_(m−1)（bitごと保持） |
| tie（2番目との距離差≤1e−12） | `UNRESOLVED_BRANCH_TIE`、energy=None | 計算上のargmin（同値はlowest ID）を採用し、停止しない |
| 出力 | `DECODED_...`/2種のUNRESOLVED | 常に`DECODED_TOTAL_ZERO_HOLD_TIE_ARGMIN_NOT_RMSE_QUALIFIED`。`Ê=fl(−θ̂_M/h)`。範囲は`[−π_f/h, π_f/h]` |
| 診断 | — | `zero_hold_rounds`、`tie_argmin_rounds`、`partial_decoder_status`（旧ならどのstatusか）、`phase_path` |
| API | `F.decode_coordinates_batch`、`S.decode_counts` | `T.decode_coordinates_batch_total`、`T.decode_supplied_coordinates_total`、`T.decode_counts_total`（priorは旧と同じく診断用で、選択には使わない） |

- **差が出る範囲**：旧decoderで未解決になる履歴だけ（W0命題1）。
- **保全**：旧decoderのstatus、部分関数としての動作、保存結果、canonical ID、error contractは変えていない。
- **binary64の扱い**：zero・tieの判定、演算順序、候補生成、wrapの端点（`+π_f`を返し得る）は旧と同一である。
- **状態の同一視**：DPと決定論riskは、旧と同じく状態をbinary64のhexで同一視する。符号付きゼロも別状態として扱う。

## 2. Q3型risk評価器（`rpe_total_risk`）

### 2.1 構成

`evaluate(..., options)`：

- 入口：`options.estimator`に新IDを明示しない限り拒否する。
- 呼出し：Bの`D.evaluate`を無変更で呼び、その結果を`partial_decoder_risk`として保持する。
- `deterministic_input.options`に推定量IDが入るので、既存のN packet照合（`D.compare`）がそのまま推定量の違いを区別する。
- 全関数版の結果は`deterministic_risk`に入れる。`D.compare`のmodel filterはこの値をそのまま使える。

| 部品 | 数学的な内容 | 適用条件 |
|---|---|---|
| 旧coarse上界（B） | `J_M+P(not all good)·max(L_D,hi)` | 閉包証明が必要。`bad_loss≥L_D`なので全関数版にも有効（W0命題2） |
| 旧部分DP（B、任意） | 上界`η+u·hi+r·L_bad`、下界η | 上界は`hi≥L_D`のときだけ全関数版に有効（未満なら自動で除外）。下界はη（lo項は使わない） |
| 更新幅補題上界`lemma_bound` | `J_M+Σ_f P(F=f)·ℓ_f` | 全関数版に限る。閉包証明が必要。f=0,1は常にL_D。f≥2では、round f−1のgood oracle推定のθ*からの最大円周距離（exactな有理数）に`π(2^(1−f)−2^(−M))`とallowanceを足した半径を使う。主値guard`|θ*|+R<π`が不成立ならL_D |
| 全関数状態DP`total_state_dp` | zeroは保持、tieはargminとして遷移させる。状態は（旧で未解決かのflag, binary64 hex）。質量は共通分母の整数 | 保持上限・遷移組上限で落とした状態sは、`d(s,θ*)+π(2^(−m)−2^(−M))+allowance`（guardが不成立ならL_D）で囲む。完全追跡なら、MSE・E[Ê]・E[Ê²]・bias²・分散がexactに求まる |
| 統合 | 完成した有効な上界の**minだけ**を取る。下界は有効なもののmax | F分割（coarse・補題）とDPの状態分割の項は足さない。`lower>upper`、または全関数DPと旧DPの一致検査（ηと未解決確率）が不成立ならUNKNOWN |
| モーメント（設計診断） | `bias=E[Ê]−E*`、`Var=E[Ê²]−E[Ê]²`。残余があるときは、状態ごとの終端区間から平均の区間を作る。bias²の区間はそこから、Varの区間は`MSE区間−bias²区間`から求める | 合否判定には使わない（`DESIGN_DIAGNOSTIC_...`） |
| UNKNOWN | — | 以下の場合：入力欠落、M>12、物理未資格（Bの理由をそのまま継承）、loss未採択、count範囲外、非有限、主値範囲外、質量不保存、allowance超過、有効上界なし |

### 2.2 証明済みの部分と、工学的なallowanceの区別

**【数学】実数演算で証明済みの部分**

- 更新幅：`d(θ̂_m,θ̂_(m−1))≤π/2^m`。zero保持なら0。
- 初回unsafeの分割と半径。
- 主値guardの下では円周距離が直線距離に一致する。
- `|Ê−E*|≤|θ̂−θ*|/h+|fl(x)−x|`、`|fl(x)−x|≤2^(−53)|x|+2^(−1074)`。
- πを`float(π)<π<nextafter(float(π))`で挟む。

**【工学的allowance】`TRANSITION_ALLOWANCE=10^(−12)` rad/遷移（宣言値。認証ではない）**

- binary64の候補計算による超過分を見込む値である。
- DPで実際に計算した遷移は、すべて実行時にallowance内であることを検査する（超過したらUNKNOWN）。
- 人工のstress（12round×242,000遷移、計算上の中点を含む）では、最大超過は1.29e−15 radだった。
- 未計算の遷移（補題、DPの残余）は、このallowanceに依存する。
- 出力の`transition_allowance`欄で、証明部分とallowanceを区別して記録する。

### 2.3 Q3での役割分担（`q3_summary`）

| 用途 | 使う量 | 出力 |
|---|---|---|
| 合否判定 | 有効な直接モデルMSE上界`≤ε_E²` | `within_limit`。明示されたlimit（SOURCEと`MODEL_MSE_FILTER_ONLY`）がある場合だけ。資格・十分Nではない |
| 設計診断 | DPのbias²とVar（exact、または区間） | `design_diagnostics` |
| 従来のMinkowski条件 | 結合しない | `minkowski_reference`：`SEPARATE_REFERENCE_NOT_COMBINED`、`delta_E_added=False` |

- **δEの扱い**：直接riskはrawの位相ずれ（PF・finite-K）を既に含む。そのため**δEを加える経路は持たせていない**。
- **主張の範囲**：`claim_scope`は「供給されたraw平均・供給target・IID exact-affine count lawに条件付く」である。
- **物理的な保証との区別**：`physical_total_error_guarantee=False`。target・rawの数値allowance・実samplerのlawへの移送項は`UNKNOWN`と明記する。

## 3. 人工全数列挙・回帰テストの結果

**新規test（11件PASS、約15秒）**：

- 旧decoder解決履歴でのbit一致
- zero/tie時の推定値と履歴
- DP＝全数列挙
- 部分追跡での区間の妥当性
- 補題とguardの負の対照、stress
- Bの旧上界・下界との整合
- 偶奇・±π・binary64境界
- UNKNOWN処理
- minだけの統合、binding
- Q3の役割分担
- 旧decoderが今もzero/tieで停止すること

保存されたH4数値・MC・保存データの読取は、mockで禁止した。

[SYNTHETIC_EVIDENCE.json](SYNTHETIC_EVIDENCE.json)（14検査すべてPASS、wall 137秒、benchは4 processで並列）：

| 法則（h=10、人工loss[0,20]） | 履歴 | 旧で解決 | any zero | 旧で未解決 | 全関数exact MSE | Bのcoarse | Bの部分DP [下,上] | 補題 | 採用した上界 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| B_fixture_like | 81 | 56 | .321 | .408 | .01565 | 16.95 | [.01105, 8.173] | .08378 | 全関数DP（exact） |
| zero_tie_rich | 729 | 392 | .578 | .707 | .02400 | 19.69 | [.009647, 14.15] | .09318 | 全関数DP |
| mixed_parity | 576 | 480 | .158 | .225 | .02066 | 14.65 | [.01599, 4.525] | .05474 | 全関数DP |
| four_rounds | 6561 | 2744 | .548 | .705 | .01141 | 18.00 | [.004424, 14.11] | .07116 | 全関数DP |
| five_rounds_N1 | 1024 | 1024 | 0 | 0 | .02577 | 16.64 | [.02577, .02577] | .05563 | 部分DP＝全関数DP（同値） |
| near_principal_edge | 729 | 392 | .414 | .544 | .1410 | 16.76 | [.06736, 10.96] | .3467 | 全関数DP |
| odd_only | 1024 | 1024 | 0 | 0 | .03282 | 18.35 | [.03282, .03282] | .08035 | 同値 |
| principal_edge_wrap | 1024 | 1024 | 0 | 0 | .1384 | 10.39 | [.1384, .1384] | .2038 | 同値 |

- **全数列挙との一致**：全8法則で、MSE・E[Ê]・E[Ê²]・Var・bias・η・未解決確率・any zero・roundごとのzero/tie質量がexactに一致した。
- **Bの旧上界・下界との整合**：Bの旧上界・下界は、全関数版の真値を挟んだ。
- **補題**：全例で真値以上だった。
- **遷移幅**：全数列挙の経路で、遷移の超過は最大4.4e−16 radで、allowance内だった。
- **部分追跡**：保持上限・組上限の64通りすべてで、上界・下界・平均・分散の区間が真値を含んだ。
- **奇数N**：どちらかの軸が奇数Nならzeroは起きない（N=1..199の全count）。
- **符号付きゼロ・±π**：符号付きゼロは別の状態として扱う。`atan2(+0,−1)→wrap→−π`、wrapの端点は`+π_f`。いずれも検査した。

**回帰**（[REGRESSION.json](REGRESSION.json)）：

| 対象 | 結果 |
|---|---|
| 本branch | 33 file／367件PASS（B回帰一式＋決定論risk・DP・decoder関連＋新規） |
| B本線`e7ce23803`＋patch | 7 file／83件PASS |
| guard branch `67876f228`＋patch | 8 file／92件PASS |
| W0の`SYNTHETIC_CHECKS.py`（再実行） | 59/59 PASS |

途中の失敗はsparse checkoutにdocs fileがなかったことだけで、checkout後に再実行してPASSした。

## 4. 更新幅上界と状態集約DPの役割分担

人工M=12の4法則で比較した（h=0.1、H4型の幾何、架空のrawとN、loss familyの上端=L_D）：

| 法則（X/Y各軸のN） | 旧coarse | 補題 | 全関数DP [下, 上] | 残余質量 | 遷移組 | 補題の時間 | DPを含む時間 | bias²／Var |
|---|---:|---:|---|---:|---:|---:|---:|---|
| 偶数N（12…4） | 551.7 | 28.09 | [.6688, .6733] | 3.0e−5 | 1.13M | 0.14 s | 111 s | 4.4e−4／[.668,.673] |
| 奇数N（13…3） | 619.2 | 26.09 | [.7175, .7187] | 5.0e−6 | 0.89M | 0.13 s | 68 s | 1.1e−4／[.717,.719] |
| 小N（全4） | 916.9 | 330.2 | [23.73, 24.33] | 2.4e−3 | 0.43M | 0.09 s | 49 s | .59／[23.1,23.7] |
| 偶数N、保持64 | 551.7 | 28.09 | [3.0e−4, 1.202] | 5.1e−2 | 0.04M | 0.13 s | 6 s | 区間のみ |

保持上限は4096、最後の4行目だけ64。単位はHa²。

| | 更新幅補題 | 状態集約DP |
|---|---|---|
| 必要な入力 | count表、閉包証明、`first_unsafe`、round f−1のgood oracle推定（候補数はroundごとのgood count位相数） | count表のみ（閉包証明は不要） |
| 評価負荷 | count表の作成後O(Σgood outcomes)。秒未満 | O(保持状態×count位相群)。H4の実規模では、Bの旧DPと同程度（1ベクトル約2.4時間）と見込む |
| 出力 | 常に有効な上界（f≤1とguard不成立はL_D） | 完全追跡ならexactなMSE・bias²・Var。部分追跡なら状態依存の区間 |
| 弱点 | `P(F≤1)·L_D`が残る。unsafeが多いと緩い | 高価。残余の上界はallowanceに依存 |
| 役割 | 保証の床。安価な事前filter | 主たる値。設計診断。合否判定にはこのminを使う |

## 5. 新decoder versionと既存configuration・priceとのbinding差分

| 対象 | 現状（今回は変更しない） | 正式採用するときに必要な差分（未実施、ユーザーの判断） |
|---|---|---|
| canonical config `rmse_policy.error_contract.decoder` | 旧ID。Bの`_bound`は旧IDを要求する | 新IDにするとconfiguration ID（spec全体のhash）が変わる。round candidate ID（`RMSE:`hash）、identity、law_bindingも連鎖して変わる |
| N packet（`B.assess`） | `rmse_evidence.error_contract`がspecと一致する必要がある。新推定量は`deterministic_input.options.estimator`で区別する | 正式なN packetには新config versionが必要。旧configから新configへのbridge（物理的に同一の候補の対応表）も必要 |
| 価格record・cohort | candidate ID・identityで結合する | 物理回路は同一でも、canonical IDが変わるので再結合が必要（値は再計算しない）。cohortのkeyも新candidate IDになる |
| 旧の結果 | B run01、W0、DPの結果 | 旧IDのまま保全。比較は別IDの結果を並べて行う |

## 6. Bの最新技術統合branchへの取込み

既存fileを変更しない追加3 fileなので、conflictは起こらない。

- 手順：`git cherry-pick 710aa69b663ef3313439762d25c80a652a03948f`（または[patch](total_estimator.patch)の`git apply`）。
- 適用を確認した対象：
  - B本線`codex/rpe-decoder-statistics-20261113`（`e7ce23803`）
  - guard接続branch（`67876f228`、`3a27f99fe`系）
- 詳細は[HANDOFF](HANDOFF.md)。

## 7. 実H4で必要となる最初の有限科学評価（案、未承認）

[HANDOFF §2](HANDOFF.md)：

- 対象：Bの固定2ベクトル（LD1/LD2、`e7ce23803`のINPUTS）。
- 方法：同じ条件に全関数推定量を当てる。補題を実行し、全関数DPは同じ追跡範囲（8192／141,559,036）で実行する。
- 合計：2評価。
- 新しいN、配分、MC、Gは含まない。

## 8. 残るUNKNOWN（target・raw allowance・actual sampler law）

| 項目 | 状態 |
|---|---|
| target | `E*`は保存solver energyの条件付きproxy（ground未認証）。DF10 Hの基底か、DF前の分子Hかの差、`d_target`の包含区間は未取得 |
| raw allowance | rawはbinary64のideal finite-K平均で、EXACT_AFFINEではそのbinary64値をexactな確率として扱う。rawの数値誤差η（認証）と、実数decoderへの包含は未取得 |
| actual sampler law | 評価はIID二項のideal law。実sampler（PCG64/CDF・経路sampling）とのlaw距離は未資格。W0 N8のTV移送は未実装 |
| binary64 | 遷移allowanceは宣言値で、区間演算による認証ではない。検証済みの範囲はM≤12、q≤2^52 |
| 実H4での効果 | 全関数版のMSE、bias²・Var、旧上界からの改善幅は未評価（科学実行が必要） |

## 9. 保全・STOP

- 固定SOURCE・input・raw・結果・grant・ledger・marker・KEEP・STOP、Bのbranch・process、共有環境は変更していない。
- 検証用の隔離clone（push先は無効化）は作業後に削除した。
- merge・ref削除・prune・gc・履歴書換えはしていない。
- **公開とremote照合の後はmandatory STOP**。次は、ユーザーがD1（推定量の採否）、Q3契約、§7の有限科学評価の許可を判断する。
