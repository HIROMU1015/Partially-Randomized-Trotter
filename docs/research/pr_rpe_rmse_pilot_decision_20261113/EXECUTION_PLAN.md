# B：条件付き・限定科学pilot実行計画

**未承認・未実行。** 本計画は[REPORTの提案A](REPORT.md)またはユーザーが指定する別の1系・固定(M,κ)に対応する。現時点では[既存template](../pr_rpe_rmse_pilot_preparation_20261113/PILOT_INPUT_TEMPLATE.json)の未定欄はnullのまま。既存N/RMSE/preflight/controllerを再実装しない。

## 科学入力とSOURCE

| 入力・処理 | 固定SOURCE／関数・保存先 |
|---|---|
| 外側設計・H/state/snapshot・profile | A `ee15bbd29ec571dd2526feecdb81ed757f783557` の `P1_MATCH_METADATA.json`、元保存task。H4 S2/p1/δ.1/K2/h.1、H6 m5/p7/δ1.6/K2/h.1とearly p5/R32を保持 |
| λ_R | 保存SOURCE_INDEX `7a8bc7f132e166390c943e6b32c92cc6abe1baf4` → T20 context。H4 prefixes.1.lambda_R、H6 regular prefixes.7.lambda_R。context hashは前段REPORTに固定済。再DF/対角化で生成しない |
| 保存mean/数値allowance | P1公開 `71ba49bb28b295aa28e51101bc3b1bf07d500767`、科学SOURCE `a1a79213a9b5331145e20e8778a3be9d9fc4e497`、`docs/research/pr_r_detailed_n_p1_results_20261111/run01/result.json` の対象record |
| 保存reference pair | 同P1 resultのreferences。H4 T.1/6.4、H6 T1.6/25.6。値の存在と数値gateの合格は区別する |
| κ/M/model policy | A model `510fbacd01a2d855202de4c045d8ba1972e5ff3d`、`pr_rpe_rmse.schedule/rmse_policy/evaluate/mean_evidence`。B final `f09af82482aad11460c4bb73d3e13d7ada40be6f` のA adapter |
| 保存照合・二時点scope | ee15 `pr_rpe_rmse_pilot.pilot_inventory/prepare_pilot`、既存CLI `--two-time-pilot` |
| 数値pair gate | `pr_r_detailed_n_p1.checked_reference`。新referenceへfallbackしない |
| 局所phase/radius | ee15 `supplied_disk_geometry`。原SOURCE付きのraw disk/empirical allowanceだけを用い、未知はnull |
| Bへの結果 | A `b_handoff` → ee15 `handoff_supplied_results` → B `n_from_a`。十分N/global boundのnullは維持 |

現行固定SOURCEの既知normalization/numerical range/closure/入力binding・STOP条件を使う。旧raw/結果/markerは読取専用、新成果は別の結果領域に保存する。旧P1 executeや消費済runを再実行する計画ではなく、その数値leaf/保存値を利用する新たな限定scopeである。新grantや承認証明書は要求しない。

## 初回推奨S0：保存再解析だけ

ユーザーが、採択する条件または決定式と、**その具体化・保存再解析を含む科学実行**を明示的に承認した場合だけ実施する。

1. **一つの条件を具体化**する。提案Aなら固定SOURCEのλ_1とhを使ってκ_Aの記号式を一回具体化し、指定εE/εqpe/εtrotとM_empを具体化する。実λのinterval sweep・探索・最適化は含めない。literal M/κをユーザーが指定した場合はそれを使い、都合よく変えない。
2. 元Aのexact Fraction/grid規約で全0..Mのmetadata scheduleを生成し、BのA専用経路と照合する。全roundは構成とidentityの確認だけで、科学評価はPR二時点に限定する。H6 early override、signed weight/順序・scalar・prefix・qRPE/qPFを保全する。
3. 完全なphysical candidate/H/state/snapshot/profile/law/SOURCEを照合する。提案Aではm0/R2 `53e00d8b07d1547d19a7`、m6/R128 `2a8c6718220eec59d419`だけを読む。保存reference Tとbindingも照合する。候補が異なれば別Rの保存meanを使わずSTOPする。
4. 対象二候補の独立reference pair、raw/corrected closure、exact-tail norm、原R-aware数値allowance・logB marginを再確認する。保存P1のu/logB marginやstate仮定を保持し、旧α/βに依存するshot/統計判定を新RMSE条件へ流用しない。gate不成立・SOURCE不一致・不明な数値条件はSTOPし、許容値を変えない。
5. 保存されたfull-H referenceとmeanからbias・phase/信号radiusの**局所診断**を行う。元conditional diskを参照する場合は、その旧SOURCEと経験的・状態条件付きという意味を明記する。新しいεtrot保証や実装/ground/law誤差の上界にしない。未知誤差はnullで、実装誤差0のlaneへ統合しない。
6. 保存whole logBを使うfinite_B_diagnosticと固定Mで、PR二候補・X/Y別の経験的nominal Nを評価する。有限normalizerやHamiltonian actionの再計算はS0に含めない。A.evaluateへ渡すmean_providerは対象二requestの保存meanだけを返し、他roundは未供給とする。旧Hoeffding Nを入力せず、他policyも同時算出しない。
7. supplied-result handoffで全schedule形のpacketに入れる。二候補以外のNは未評価、十分N/actual N/global RMSE bound/C/Gはnull。model budget・envelope・数値条件の不成立も結果として報告し、retryや候補差替えなしで公開照合後STOPする。

S0が得るのは保存finite-K信号に対する局所診断と経験的Nmモデルである。全RPE成功・RMSE保証・十分N・実資源優位・最適M/κを判定しない。離れた二時点を連続m=0,1としてdecoderへ渡さず、新しいdecoder samplingもしない。

### S0の科学call範囲

| 処理 | 対象・最大数 |
|---|---|
| 固定λ/SOURCEによる選択式の具体化 | 採択された1系・1組(M,κ)のみ。κ探索なし |
| 保存数値gate・局所bias/phase/radius/logB診断 | 指定PR二候補のみ。既存値の再解析 |
| 経験的nominal N_X/Y | 2候補×2軸のみ |
| 新finite mean / 内部exact-tail / 新full-H reference | **0 / 0 / 0** |
| finite normalizerの新計数・H/state arrayの新action | **0**。保存JSON/scalarで不足すればSTOP |
| P2/P3、sampling、compile、C/G、全horizon科学評価 | **0** |

## 代案S1：固定候補の不足だけ取得

S0ではなくユーザーが明示的にS1を採択する場合だけ、同じ固定二候補について下記を許可範囲に含められる。現P1の適合組では全mean/referenceのfieldがあるため、最初からS1を必要とはしない。具体化時の実不足を先に列挙する。

| 不足と理由 | 既存関数 | 最大追加call |
|---|---|---:|
| その固定candidateのmean/独立raw/correctedが未取得 | `pr_rpe_rmse.existing_finite_mean` → `h6_df19_rpe_d2.evolve/half_step`。元finite RTE規約を保持 | finite mean **2候補** |
| PR内部exact-tail/norm証拠が欠け、保存から回収不能 | 元evolveの独立内部column。meanと共有する内部対照であり別R候補ではない | 内部対照 **2** |
| 同H/state/Tの独立reference pairがない | `h6_df19_rpe_d2.reference_pair` のexpm_multiply/expm | 最大2時点×2経路＝**4 action** |

不足がなければ追加call0。R違いのmeanは流用せず、同一T/H/stateなら保存referenceを優先する。fresh action内で必要なfinite normalizer評価はその固定候補分だけとし、独立したnormalization sweepは行わない。新SCF/DF fit/eigensolve/state生成、trajectory、回路build/compileは含めない。

**欠測**と、保存FAIL・数値未解決・SOURCE/law不一致を区別する。後者はS1でも自動fresh/retryの理由にせずSTOPする。閾値変更、q/R/K/prefix探索、別のκへの変更、科学的retryは承認範囲外。新規mean取得後もRMSE資格は未知のままで、S0と同じ限定診断・packetを返す。

## 分岐・依存・終了

- 条件が未採択：本提案と元templateを提示してSTOP。今回の状態はここ。
- 1系/M/κ/予算/policy/有限scopeが明示採択・実行承認：その一括scopeだけを進める。処理stageごとの追加承認やgrant発行を要求しない。
- 実κの一致・numerical gate等が不成立：原因を保存し、候補/条件/閾値は自動変更しない。
- nominal算出ができてもstate/finite-K/alias/実lawの資格が不足：UNQUALIFIEDを保持する。これを科学的PASSで埋めない。
- 成果をcommit/pushしremote SHA/主要成果物を照合：mandatory STOP。B/Cの価格・全G統合・全round統計評価は別の作業scope。
