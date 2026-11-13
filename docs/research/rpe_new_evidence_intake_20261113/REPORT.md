# A/C/Dの新証拠受入・4方式のV2整合

2026-11-13 JST。B基点`f0c4bb559d2a64dabe88e2eba31ff1dadad6fca8`、既存branch `codex/r-variable-rpe-cost-pipeline-20261111`を継続。技術実装・公開metadata照合・人工/回帰テストのみ。今回の最終成果は**run02**。run01はA実結果公開前の技術受入記録として保全する。

## 結果

| 担当・公開SOURCE | 正式受入状況 |
|---|---|
| A準備 `ac873e7fcc42e937084fa153476cbd87565702b4` | m1専用scope・科学SOURCE・保存carrier・packet仕様を確認 |
| A実結果 `3e035c0d61386010f59ad7ea792bc7a5cefd18ac` | 作業中に公開された固定commitを取得し、実result/N_HANDOFFを正式受入。m1 nominal **60/60**は保存値の転記だけ |
| A元S0 `75b5c89bddf8a1ffa4a8ac99b2c06d6e05fa012d` | m0=64/64・m6=38/38と元SOURCEを保全。旧S0-only validatorは無変更 |
| C診断 `a637ed1b813dc3952339cb05f36beb79168ace23` | 物理価格状態・状態縮約SOURCEの診断metadataを受入。K0/K1の新実結果は未公開・未受入、平均S/Cは作成しない |
| D物理request `0536646b843781a0e0fb6629473526c05f16a30e` | 公開の人工requestをBの正式formula/layout/外部identity_mapへ接続 |
| D保存coverage `92ff17dededea6e1ee273b2eee6a08e292ee9e5b` | 136 Full PF候補の分類を受入。新receipt照合結果・更新V2 mapは未公開、既存分類を変更しない |

正本はB030f3f548の保存configuration、ID=`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。対象PRはH4 linear 1Å/STO-3G、保存tol-only DF10/元H/state/snapshot、S2/L_D1/ΔPF=.1/K2、h=.1/M12・13round、元exactκ、minkowski/finite_B_diagnostic、元prior/decoder/control/compiler/full wrapper/P0。今回これらの科学条件を変更・採択し直していない。

## A：m1を新scopeで受入

新`rpe_diagnostic_intake.validate_m1_packet`を追加した。既存`h4_s0_v2_intake.validate_A_v2_packet`で元S0 packetを確認したうえで、新packetの全13canonical identity/law、元physical ID、policy/H/state/snapshot/profile、数値SOURCE・adapter SOURCE、previous S0 locator、roundごとの`diagnostic_source`を照合する。

m1はT=.2/qPF2/R4/8event、physical `c00e00dc1e9c414fb7d9`。科学SOURCEはac873e7fc、数値carrierは89c91c692のH4 batch `run01/result.json:records[5]`、元科学90dcc3ed2。resultとpacketのnominal/model status/物理候補も一致した。数値gateやphase/radius、shot式はBで再実行していない。

`newly_reanalyzed_rounds=[1]`、`preserved_S0_rounds=[0,6]`を検査し、m0/m6を今回再解析した結果とは扱わない。nominal診断があるroundは0/1/6の64/60/38、他10roundはNOT_EVALUATED。十分N_X/Yは全26軸null、global bound=null、科学モデルUNQUALIFIED。旧HP型N・nominalを十分Nへ昇格しない。元S0 validatorは新m1値を引き続き拒否する。

## C：kernel/構造Sとcompiled Cの分離

公開run02/RESULT、CHECKS、REDUCTIONと前回正式価格requestを使用。m0/q1/R2とm6/q64/R128のphysical/law/B正本・元SOURCEが対応する。元診断SOURCEはeeccdf18、縮約SOURCEは2b3b0ee7。初回失敗・run01/STOPを含む既存結果は変更しない。

SOURCE由来の縮約条件が記録されていても、元`state_equivalence.status=UNKNOWN`を保持する。物理transition/reward parityはnull、kernel構築/K0/K1成功・平均S/Cの新しい証拠は未取得。Cの実行branch/processには介入していない。

将来K0/K1の結果を受ける小さな構造診断envelopeを用意した。kind/quantityは`MEAN_STRUCTURAL_RZ_COUNT_NOT_COMPILED_C`、SOURCE/identity/law/profile/checks/不確かさを保持し、K1はK0 PASSを前提とする。部分取得・nullは残し、平均Sのcell別値を転記するだけで、`compiled_price_records=[]`、E[C−S]/E[C]/G=null。Cの実公開formatが異なる場合はSOURCE付きの最小adapterを追加する。

既存`V2MeanPriceLookup`とB.assessのquantity検査を再利用し、構造Sの価格供給を拒否する。C1 MODEL false、finite cohortと全law期待、実samplerとの差、rare/長q systematic・covariance/CIの未資格を維持する。今回はK1値を作っていない。K0/K1は実行段階名であり、Taylor cutoff K=2を変更する指定ではない。

## D：物理SOURCEとcanonical V2の対応

新`rpe_d_evidence_intake`は**固定Git上のD既存metadata module**と`full_pf_saved_inventory.for_packet`を再利用する。Dのrunning worktreeやprivate context/rawを読まない。Bの既存generator・標準4次・κなしmapping・外部`requests/validate_requests`を再実装せず使う。旧source viewは保全し、canonical viewをB自身に生成させる。

- 標準4次：Dの`YOSHIDA4_STANDARD_H46_M5_PLAN_BINARY64`とBの`PF_STANDARD_YOSHIDA4_BINARY64_V1`を別欄で保持。元af284d85の`h46_m5_plan.weights('4th')` SOURCE byte、binary64 signed列・実幅・scalar・元physical候補を照合した後だけ結合する。m5へaliasしない。
- Full-H：Dのκなしphysical kindからBの`FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1`へ技術的に接続。DのSOURCEと元kindも保全し、mapping sourceは0536646bへ固定。q/R grouping、component law、全H one-body/preparation/λ/SOURCE、K、scalar一回、`priced_equivalence_claim=false`を保持。κはNoneのままで、κ由来scheduleと同一視しない。
- 外部identity_map：A/C requestが同一physical候補・canonical全schedule identity・law・SOURCE/profileを参照する。元の非null V2 IDは一致を確認し、元nullのpending欄を過去の成果として書き換えない。同event数・異groupingは別候補。新native hookは不要だった。

公開D人工例3件を接続した（[D_ARTIFICIAL_BRIDGES](run02/D_ARTIFICIAL_BRIDGES.json)）。**人工例のcompilerはMOCK_COMPILER**なので、保存科学価格への結合は`D_SAVED_COMPILER_PROFILE_MISMATCH`として拒否した。profileを自動修正して価格を流用しない。Full-H例はFull PF coverageの対象外。実Full PF/RTEの比較menuは生成・採択していない。

## 保存coverageと4方式の不足

既存D分類を転記し、physical ID/Full rank/formula/SOURCE・X/Y別status・報告countを検査した。receiptの再照合、raw signalや価格の数値再解析は実施しない。

| Full PF保存分類（H4 DF10/H6 DF19、2nd/標準4th/m5） | 候補数 |
|---|---:|
| signal EXACT / CONDITIONAL | 134 / 2 |
| reference present（EXACT / CONDITIONAL） | 136（110 / 26） |
| compiled direct metadata EXACT / CONDITIONAL / missing | 56 / 17 / 63（X/Y各同じ分類） |

独立reference pairはD報告上T=2.0だけ。fixed-deltaの2件はraw signal不足。17件はreceiptの追加照合が必要。これらは保存SOURCEの分類で、現RMSE/価格policyへの資格認定ではない。136候補に共通h/M/prior/policyを勝手に付けず、全候補のV2 configuration IDは未採択理由付きnullを保持する。

[既存plannerの更新結果](run02/EVIDENCE_PLAN.json)には、固定PRの新m1 packetとC診断参照だけを渡し、costを呼ばない。他方式の実menuを捏造せず、4方式の不足は[METHOD_GAPS](run02/METHOD_GAPS.json)で分けた。

| 方式 | 利用可能になった範囲 | 残る不足・担当 |
|---|---|---|
| Full PF | 136保存coverage、標準4次を含むD→B接続 | **ユーザー/D**：正式formula/幅/M/RMSE比較条件。**A**：元H基準の十分N/global契約・off-grid mean/reference。**C**：17 receipt、63欠測・grid外C。保存DIRECT/MODELは別 |
| Truncated PF | 既存V2/H4 B0=p7の区分を保全 | **ユーザー/A/C**：比較menu、元Hへの切捨て誤差、RMSE N・価格資格。B0をFull PFへ昇格しない |
| PR | nominal診断はm0/m1/m6。Cは局所価格状態のSOURCE診断 | **A**：残10roundの診断・全13round十分N/global RMSE/finite-K/κ資格。**C**：K0/K1公開、E[C−S]と適格E[C]、law距離・長q/rare・共分散/CI。全26軸の適格Cは欠測 |
| Full-H RTE | 全H one-body/明示layoutと外部V2接続 | **ユーザー/D**：正式preparation/qR/K/M/policy。**A**：normalization/signal/N/global契約。**C**：grouping・law/profile付き平均C。DF B3をFull-Hへ転用しない |

全方式でG/winner=null、4方式比較・科学的優位性は未評価。既存固定R/長q modelや構造Sで不足を埋めない。

## 検証・境界

[TEST_RESULTS](TEST_RESULTS.json)／[TEST_LOG](TEST_LOG.txt)：**人工・回帰106件PASS**（新13＋前回93）、failure/error=0。公開A3e035c0dの実packet/resultもmetadata fixtureとして受入PASS。SOURCE/law/profile不一致、m1 scope、元S0拒否仕様、巨大整数、S→C混同、D formula/source/grouping、coverage非採択、未資格G=nullを確認した。作者自身の技術確認であり、科学的保証・独立reviewではない。

旧8次検証methodとS0 producer suiteは今回対象外。新科学signal/N/RMSE・normalization・保存数値再解析・sampling/build/compile/fit・実G/最適化は0。旧B core/S0 validator/C providerおよびA/C/D科学SOURCE・input/raw/結果/grant/ledger/marker/KEEP/STOPは無変更。

新実行許可をこの技術受入から導かない。Cの既承認科学作業の進行も本BのSTOPでは変更しない。成果をcommit/pushしremote SHA/tree/主要artifactを照合した後、**mandatory STOP**。
