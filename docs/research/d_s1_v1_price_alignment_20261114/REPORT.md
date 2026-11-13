# S1/V1：compiled価格の比較条件と長時間検証のD引渡し

2026-11-14 JST。**旧Full-H価格と新PR価格は同じ実装条件の価格ではない。旧Full-H価格を新loweringの正式比較へ渡さず、同じ条件の証拠がないセルはUNKNOWNにする。** 巨大m12 compileは前提にしない。A/Cの最新Full PF shortlistと保存receipt接続を再利用し、最初の代表候補をS2/.2のm5とする条件付き案を具体化した。

[S1のdomain・修正案](S1_DOMAINS.md)／[V1の根拠・精度](V1_LONG_TIME.md)／[有限追加案](FINITE_VALIDATION_PLAN.md)／[C/B HANDOFF](HANDOFF.md)／[セル・seed](REPRESENTATIVE_CELLS.json)／[SOURCE確認](SOURCE_CHECKS.json)／[確認ログ](CHECKS.txt)。production変更・新preparation/sampling/回路build/compile・価格再集計/fit/E[C]/Gは**0**。

## 今回の固定SOURCE

| 役割 | 固定commitと参照対象 |
|---|---|
| 監査 | `51d324ea9c83bc12f1e866da0fa409cef926f74a`、`pr_resource_repo_wide_audit_20261114/REPORT.md,FINDINGS.md` のS1/V1/E4 |
| D基点 | `01e068af7e06d9f1f5621a5b3800c30eb32ecb06`。e827 scalar修正済みselectorと既存取得/fast/新ZZを保持 |
| A最新Full PF shortlist | `3ae055fa7a696b13320e00ec79a5ba15c4ed182e`、`h4_full_pf_heff_scan_results_20261114/HANDOFF.md,C_HANDOFF.json` |
| A後続の未実行計画 | `d48d8732ee7ef521aab5d67a0945439364f5a2f8`、`h4_full_pf_followup_plan_20261114/REPORT.md,HANDOFF.md`。追加幅5候補と保留M5/32h、未承認の計画 |
| C最新価格coverage | `d7c09ef2e2521bb1815d1d11970529fc7030c2e3`、`full_pf_candidate_price_evidence_20261114/REPORT.md,HANDOFF.md,MINIMAL_VALIDATION_PLAN.md` |
| CのPRモデル検証案 | `3ae2e605dd6efcc63ed3742fe3076a8be3e6afae`、`rpe_price_evidence_validation_20261113/VALIDATION_PLAN.md`。モデル基盤は1914d076 |
| CのFull-H/B0I signal | `367dc98d832ecb4d769f2a3da3d174bc697331a5`、`truncated_fullh_signal_results_20261114/HANDOFF.md` と固定Full-H `PLAN.json`。signalは価格ではない |
| B risk/N状況 | `ce0517de6e7b58cf4bf4c69f99f998d86099c406`、固定2 DP診断input。engine `06db9aefec13bb5c5be6d7c0f81f29b3a10b929c`、前のrisk結果6106e4d9。最適N/全候補の資格packetではない |
| Full PF receipt | `e7dca5f7301c8633b17a59b6a7161331303aef2d` の完成済みD receipt/coverage。C d7c09がAの現3候補へ接続済み |
| Full-H旧価格 | light `9b83b96b83f62142c9d87858dc24a94db7ac5d61`（実行ac33da5f）、T2.0 `7a8bc7f132e166390c943e6b32c92cc6abe1baf4` |
| native SOURCE | provider `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、carrier `155529e20a8fac39e00786c40b57830b71203c4f`、receipt側compose/costの参照af284d85 |

調査時にA/B/C remote HEADが表の3ae055/d7c09/ce0517であることを読取確認した。既存D branch `codex/d-compiled-residual-c-minus-s-20261113`を継続。C/Dの分担は、本D成果をCの固定計画に合わせて文書で引き渡す形とし、Cのvalidator/model・BのN/decoder・Aの候補生成を重複実装していない。別agentへの委任・他担当worktreeへの変更も0。

公開前の再照会でAがd48d873へ進んだため後続計画も読解した。元3候補を上書きせず、exact h相対幅の追加案を別menuとして保持している。新しい科学結果・価格・Nではない。P-S2取得を判断する前にC/Bが元S2/.2を比較familyへ残すか確認し、新幅へ旧係数を流用しない。C/Bの観測remoteは引き続きd7c09/ce0517。

## S1：確認した差と影響

旧providerはcontrolled ZZ rotationを `RZZGate(2u).control(1)` でappendする。新D subclassはcontrolled/rotation/二wireの場合だけ `CX(s0,s1)・CRZ(2u)(anc,s1)・CX(s0,s1)`へ置換し、Z/product/uncontrolledその他を元実装へ委譲する。P0/P1の同unitary・control/global phase確認は既存証拠を再利用し、今回compileを繰り返さなかった。旧Full-H light/T2.0は元cost_trialへ入り、新ZZ切替を呼ばない。

compilerは両方ともQiskit1.3.0、rz/sx/x/cx、opt1、seed17、hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。このhashにはDの外側のlowering選択が含まれない。同じcompiler hash、同じ作用素、同じevent数だけでは同じ価格quantityにならない。

旧Full-Hは元candidate/source/law/cohort/軸/profileでの有限標本証拠として保全できる。新ZZの価格、現Cの全13round平均、又は新価格の厳密上界にはならない。P0の局所17〜19 RZ差を全eventへ掛けて保存価格から引く処理もしない。compilerの境界相殺・contextが違うため、その変換式を証明していない。

Full PF・Truncated PFはRTE event builderを通らない決定論経路。RTE専用loweringの変更はその経路では非該当だが、deterministic native/scalar/control/wrapperのSOURCEを別途保持する。全methodに同じ文字列IDを貼る修正はしない。旧Full PF receiptは元native policyで再利用し、Cが既に整理した同一性とconsumer受入条件を保持する。

**今回S1比較guardをproduction実装したとはしない。** C担当へ、recordのSOURCEに束縛したlowering解決、RTE有無による適用区分、comparison familyの共通policy検査を渡す。現モデルが未資格で正式Gを拒否されることと、将来DIRECTを受ける際のlowering検査が十分であることは別である。

## V1：今ある証拠と最小の次段階

AはFull PF S2/.2、標準Y4/.4、M5 best/.8を残している。C d7c09が保存5round/10軸を対応済み（S2/Y4 m3,m4、M5 m3）。そのreceipt監査や78軸coverageをDで再生成していない。Cの新4round/8 wrapper案のうち、**S2/.2 m5/T3.2/q16/X,Y**を最初の候補にする。Aの優先順位と一致するが、compiled最安・G勝者の採択ではない。

この一セルは保存m3/m4を校正入力として受入・係数固定した後のholdout入力候補に過ぎず、m12 q2048への移送保証ではない。M5 m4校正と他2公式m5はCの元8件案の別選択肢として保持し、自動追加しない。PRの新6 pairedセル案と旧DのLD2接続案も未実行・未採択のまま参照し、全案を合算したmenuを作らない。

Full-Hを新ZZで比較に残す場合の最初の技術候補は、現C固定scheduleのm0/T.1/q1/R448/K2、fresh normal1 path・X/Y最大2 compile。旧light/T2.0の候補と一致せず、新SOURCE・cohort/seedが必要。この1 pathは新loweringでの実配線と回路の作用/価格の証拠であり、law平均・CI・model精度を資格化しない。

P2のLD1 m6の既存path時間846〜890秒・peak2.75〜2.82GB、compile各約242秒という記録を参照できる（[P2 HANDOFF](../d_p2_direct_pilot_20261113/HANDOFF.md)）。計時境界/負荷/候補の違いがあるのでFull PF/Full-Hやm12の費用へ比例換算しない。旧LD2/m12 T2はscalar gateでsampling前STOPしており、巨大回路を実際にbuildして実行不能を測定した結果ではない。高roundの大きな負荷予想と実測済み性能を区別する。

監査の「m10〜12が約97%」は元nominal N×理想構造E[S]の診断に基づく。適格N×E[C]による正式Gの寄与率ではなく、今回再集計もしていない。必要精度はBの同taskの候補別N/許容N集合・比較margin/regret・price law/主張水準を受けて決める。Bが固定DP診断を進めていることから、既にその入力が揃ったとは扱わない。

## 技術確認・残るUNKNOWN

新 [check_source_routes.py](check_source_routes.py) は標準libraryのみで固定Git blobを読み、旧/新ZZ分岐、元Full-H呼出し、Full-H独自scalar/one-body、PF経路、測定付きXY/代表作用検査、receipt側とcarrierのcompose/cost関数同一性を**10 SOURCE assertionで確認**。A/Cの既存physical selectorだけをcopyし、元Full-H seed関数を再利用して整数seedを計算した。量子library・分子input・price値の読込み/算術はない。作者自身の技術確認で、回路作用の新検証や独立科学レビューではない。

旧P0/P1/P2、D e827/01e068、Cのcoverage/model testとreceipt監査は元記録を参照し、重複再実行していない。監査I2の共有module版問題は今回のS1/V1とは別で、科学call前には選択runnerの実loader経路を人工入力で確認する必要がある。旧固定guardを緩和して解決しない。

UNKNOWNは、修正SOURCEでのLD2 scalar実通過、新ZZ Full-Hの直接価格、全候補の適格N、長q/Rのmodel誤差包絡、law/rare・有限標本/共有誤差、全13round E[C]と正式Gである。短いholdout成功・未知を0とした費用から正式優劣を認定しない。

旧SOURCE/input/raw/結果/cache/grant/ledger/marker/KEEP/STOPは不変。Git ref削除・prune/gc/mergeは0。技術成果をcommit/pushしremote SHA/tree/主要成果物を照合した後 **mandatory STOP**。
