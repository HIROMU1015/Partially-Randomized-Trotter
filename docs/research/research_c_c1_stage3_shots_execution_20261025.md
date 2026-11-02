# C1 Stage 3：楽観的shots下界スクリーニング限定再解析

2026-10-25 JST。固定公開SOURCE `f8fd2d373fc81353330e732c46b3b5d471e7c683`。
Branch `codex/research-c-c1-stage3-shots-science-20261025`。

**保存データ再解析は正常完了。282/282 logical units、20候補・260roundセル・520軸欄を保存。潜在的適格14候補の楽観的shots合計を取得し、前段不適格6候補は合計nullを保持。科学的FAIL0、retry／resume0、mandatory STOP。**

[全原本・要約・CSV](../../artifacts/research_c_c1_stage3_shots_science/2026-10-25/)／[20候補JSON](../../artifacts/research_c_c1_stage3_shots_science/2026-10-25/candidate_summary.json)／[候補CSV](../../artifacts/research_c_c1_stage3_shots_science/2026-10-25/candidate_summary.csv)／[260round CSV](../../artifacts/research_c_c1_stage3_shots_science/2026-10-25/round_shots.csv)／[520軸CSV](../../artifacts/research_c_c1_stage3_shots_science/2026-10-25/axis_shots.csv)。巨大整数はJSON／CSVの十進文字列を原本とし、表計算ソフトで読み込む場合も文字列型を用いる。

## 固定条件・再利用・意味

直前の準備方針に同意し次の作業へ進めるというユーザーの指示を、直前に提示した固定SOURCE・契約・最大282-callの保存データ限定再解析に対する許可として実行した。承認証明書・承認記録・新grantは作成していない。[固定契約](research_c_c1_stage3_shots_contract_unapproved_20261025.json)の準備時点のunapprovedファイル名・metadataは無変更で保持する。

保存P-H4（直鎖1.00 Å・STO-3G、Nα=Nβ=2、sector36／full256）、CDF保存rank8 number・元g補正κ／XDF保存rank10。CA0.00159 Ha、DF reserve CA/8、δ0.1、m0..12。決定論ordered symmetric S2はs1/4・全frame、PRはL_D2・s1/4・r4/16・K0/2の元順序20候補。各DF sector ground-state oracle、状態準備費用除外、ideal finite-RTE IID model、βtotal=π/6・βstat=π/12・alpha_axis=0.01/26・alpha_total0.01を保持。

保存feasibility結果のcommitは`2eef01a4256df4c146324ef571c0074cd0ea5b08`。その23 scalar JSONだけを読み、新しくPF/RTE・operator・Hamiltonian・回路を作らず、既存精度geometry・Hoeffding helper・十分条件分類・call／partial保存を再利用した。新SOURCEの変更は0、fit／optimizer0。

既知のPF/RTE operator誤差とAを保持し、追加合成誤差・law誤差が**仮になければ**、既存十分shots算定式が要求する楽観値を求めた。

```
e = epsPF + epsRTE
rho_opt = A * (1 - e)
epsilon_axis = rho_opt * sin(beta_stat) / sqrt(2)
N_axis = ceil(2 * log(2 / alpha_axis) / epsilon_axis**2)
N_candidate = sum_m (N_X + N_Y)
```

この値は採用済みRPE/Hoeffding十分shots算定式内の楽観的必要数であり、情報理論的・普遍的な下界ではない。未知の実装誤差を実際に0と認定せず、`actual_synthesis_signal_error`・`actual_law_signal_error`・`guaranteed_actual_shots`・`CA_achieved`はnullのまま保存した。理想条件での算定値を、実回路の保証付きshots・CA達成や無条件資源優位とはしない。

## 候補別結果

表は登録順であり、回路資源順位ではない。数値は13round×X/Yの合計、A=1対照は同じPF/RTE誤差を残して減衰だけを除いた診断で、新候補ではない。log10欄は元Aでの合計／A=1診断合計の桁差。大整数の表記は表示用で、全桁はJSON／CSVに保持する。

| 候補 | 判定 | 楽観的合計shots | 同誤差・A=1合計 | 減衰による増加 log10 |
|---|---|---:|---:|---:|
| `CDF_D_s1_r0_K0` | 算定対象 | 13,606 | 13,606 | 0 |
| `CDF_D_s4_r0_K0` | 算定対象 | 13,314 | 13,314 | 0 |
| `CDF_PR_s1_r4_K0` | 前段不適格・合計なし | — | — | — |
| `CDF_PR_s1_r4_K2` | 算定対象 | 2.578781e46 | 13,618 | 42.2773 |
| `CDF_PR_s1_r16_K0` | 前段不適格・合計なし | — | — | — |
| `CDF_PR_s1_r16_K2` | 算定対象 | 84,482,108,034,548 | 13,616 | 9.79272 |
| `CDF_PR_s4_r4_K0` | 前段不適格・合計なし | — | — | — |
| `CDF_PR_s4_r4_K2` | 算定対象 | 73,469,641,849,976 | 13,314 | 9.7418 |
| `CDF_PR_s4_r16_K0` | 算定対象 | 58,356 | 14,460 | 0.605917 |
| `CDF_PR_s4_r16_K2` | 算定対象 | 573,646 | 13,314 | 1.63434 |
| `XDF_D_s1_r0_K0` | 算定対象 | 13,600 | 13,600 | 0 |
| `XDF_D_s4_r0_K0` | 算定対象 | 13,314 | 13,314 | 0 |
| `XDF_PR_s1_r4_K0` | 前段不適格・合計なし | — | — | — |
| `XDF_PR_s1_r4_K2` | 算定対象 | 8.075661e25 | 13,606 | 21.7734 |
| `XDF_PR_s1_r16_K0` | 前段不適格・合計なし | — | — | — |
| `XDF_PR_s1_r16_K2` | 算定対象 | 610,901,826 | 13,606 | 4.65224 |
| `XDF_PR_s4_r4_K0` | 前段不適格・合計なし | — | — | — |
| `XDF_PR_s4_r4_K2` | 算定対象 | 534,391,020 | 13,314 | 4.60355 |
| `XDF_PR_s4_r16_K0` | 算定対象 | 21,408 | 13,740 | 0.192589 |
| `XDF_PR_s4_r16_K2` | 算定対象 | 46,502 | 13,314 | 0.543163 |


PRのs4/r16/K0は、CDF58,356・XDF21,408 shotsという楽観値だった。同じ設定のA=1診断値は14,460／13,740。K2ではs4/r16でCDF573,646・XDF46,502、s1/r4で約2.58e46／8.08e25と、同じ「潜在的適格」の中でもshots算定値に大きな差がある。CDF s1/r4/K2ではA=1診断13,618に対して約42.28桁、XDFの同設定では13,606に対して約21.77桁増えている。これは固定十分式での減衰負担であり、量子実験の実測ではない。

決定論s1／s4の算定値はCDF13,606／13,314、XDF13,600／13,314。回路1回当たりの費用と合成誤差を今回評価していないため、shots差から総資源の優位性を推論しない。CDFとXDFの物理DF近似精度の差も解消したことにはしない。

前段不適格6候補も全13roundを保持した。round単位では247セルが算定対象、13セルが登録十分条件不適格である。軸欄は494件に算定値、26件に不適格理由とnullを保存した。前段不適格候補の短roundに有限な値が得られても、候補全体の完全な合計としては表示しない。今回新しい閾値による除外、候補追加、優先対象の正式選定は行っていない。

## Preflight・精度・call・資源

| 項目 | 実績 |
|---|---|
| 固定SOURCE11・保存scalar入力23点のbyte照合 | 実行前後PASS、旧SOURCE／保存結果を変更しない |
| 科学契約SHA256 | `fe0f3c8eee182da01e31f2dc84fcf557580e937190fc4a69183f6e63b4bb8a9f` |
| SOURCE manifest SHA256 | `1866f4645e91eae549b113624655a591cb44ec588921f92f005b25d9575894c9` |
| 環境 | read-onlyの既存専用venv、Python3.11.1、NumPy1.26.4／SciPy1.14.1／既存5依存版PASS |
| 直前の人工回帰 | 21 PASS、4.97秒 |
| saved_evidence_load | 1／1 |
| optimistic_round_assessment（X/Y・A=1診断を内包） | 260／260 |
| candidate_shots_summary | 20／20 |
| screening_summary | 1／1 |
| 合計 | **282／282 logical units**、attempt／COMPLETE各282、FAIL0、重複label0 |
| 新fit／optimizer／retry／resume | すべて0 |
| PF/RTE再計算・新operator・実回路build／compile・実shot実行 | すべて0 |

int64を超える軸算定値は6件で、同じ式のexact-rational ceil経路を使い、任意精度整数の全桁を文字列で保存した。残る488軸は旧binary64 Hoeffding helperを使用。coordinate underflowは今回0セル、clip0。これはbinary64入力・sin／log定数に対する数値pilotであり、厳密な誤差区間付き下界ではない。科学callを追加して高精度で再評価したわけではない。

独立worktree `/tmp/prt-c1-stage3-shots-science-20261025`、fresh output `artifacts/research_c_c1_stage3_shots_science/2026-10-25/run01`で専用entryを1回実行した。affinity32 CPU・空きRAM約52 GiBを確認。260 scalar行だけなのでprocess起動・serializationを避ける逐次loopを用い、CPU／RAM／worker／wallへの人工上限は設けなかった。既存venv・共有設定・他ユーザーのprocessを変更していない。wall0.9905秒、process CPU9.2879秒、peak RSS328,548 KiB。speedup比較はしていない。

結果整理は保存JSON／progress／call journalの整合と十進整数の和、CSV表示だけを扱い、shots式・PF/RTEを再実行していない。SOURCE／契約／runtime／call／resources／各候補・roundのprogress／STOPを保存した。これは固定SOURCEによるlocal scientific evidenceで、外部独立再現ではない。旧SOURCE・科学結果・grant・ledger・marker・STOPを保全する。

## 次の研究判断とSTOP

今回の結果は「潜在的適格14」という分類だけでは見えなかった、固定十分式内の減衰によるshots負担を示す。後続対象の判断には、上の候補別shotsと既知PF/RTE余裕に加え、全lawで有用な合成誤差上界、実装finite-bit law誤差、費用support、回路1回当たりの費用を扱う方針が必要である。これらは未資格／未評価のまま。最終総費用・資源順位・CA達成は得ていない。

次はユーザー／研究Cがこの表を確認し、どの候補の合成誤差資格・費用検証を準備するか判断する。Codexはその具体的な依頼後に技術準備を行い、追加科学計算や回路build／compile・総費用評価は別明示許可後にだけ実施する。今回commit・push・remote照合後は**mandatory STOP**。retry／resume0の当該runを再実行しない。
