# S2・Yoshida4・m5_best決定論PFの共通探索域：照合・追加実行準備

2026-10-22 JST。**候補照合・共通域設計・metadata実装・人工検証・追加予算の固定を完了。新規実分子signal・回路生成・compile・shot再計算・費用再ランキングは0。実行grantは無効、mandatory STOP。**

branch：`pf-common-domain-preparation-20261022`。基点：`3eca46148070ba4d4b0f5815ef45943407cc959e`。準備実装SOURCE：`1fe0e60d28e7fc09821ee3d5a014fd5f8c976c19`。これは実行可能な本番dispatcherのsealではない。実分子backendを持たないcallback基盤と固定Git readerを用意した段階で、実行承認後には独立したSOURCE・input/runtime/preparation closure・fresh資源・stage別grantを結合する。

## 共通域と比較単位

linear H4/H6、隣接1.0 Å、STO-3G、tol-only DF threshold1e-8、生成順・signed weights・後段cutoff0、実rank10/19。T=.8、P=0、full cosine/sine Hadamard wrapper、symmetric-directional control、元constant＋B0I tail identityのscalarを一回適用する。Qiskit1.3.0、rz/sx/x/cx、opt1、seed_transpiler17、backend/coupling/layout未指定、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

| 系 | B0/B0Iの共通prefix L_D | Full prefix | q | 各公式の候補数 | 3公式計 |
|---|---|---:|---|---:|---:|
| H4 DF10 | 1,2,3,4,5,6,7,8,9 | 10 | 1,2,4,8 | 76 | 228 |
| H6 DF19 | 5,6,7,8,9,10,11,12,13,14,15 | 19 | 1,2,4,8 | 92 | 276 |

公式はS2=`2nd`、Yoshida4=`4th`、`4th(m5_best)`。元のbinary64係数・signed順序を保持し、係数最適化はしない。δ=T/q=.8/.4/.2/.1。Fullはrankごとに一度だけ登録し、full-rank B0/B0I aliasを重複計上しない。prefix単調性やB0/B0Iの同費用は仮定しない。

H6 p3/4のexact-tail診断は正式な決定論PF登録ではないため、今回のprefix和集合へ加えない。

**合計504候補**。現在の同一科学入力系列の決定論登録から、公式間のprefix和集合を取った。H6のS4 p5..10も格子へ含める。旧m5計画のH4 DF12 p11をDF10へ移さない。

H4 S2/S4の既存q16は38候補を登録inventoryに保持し、今回の共通q<=8域へは含めない。q16のm5追加は予算外。したがって共通域内の最良PFと、過去に利用可能だった全PFの最良を同義にしない。

## 取得済み候補・再利用の区分

GitHub connectorでm5関連4branchのHEADと後続直接compile等の固定commitを確認し、58件の固定GitファイルをJSON/CSVとして読んだ。各commit/path/blob/SHA256は[source catalog](../../artifacts/resource_applicability/track_a_pf_common_domain_preparation/2026-10-22/inventory.json)に保存した。入力配列やQPYをdecodeせず、保存科学値を再計算していない。

| 共通域内の区分 | H4 | H6 | 計 |
|---|---:|---:|---:|
| 候補総数 | 228 | 276 | 504 |
| 保存raw signalを読める候補 | 180 | 146 | 326 |
| 保存適格性・shotのみ参照でき、原signal metadata未回収 | 0 | 28 | 28 |
| signal/固定endpointの既存取得数 | 180 | 174 | 354 |
| 新signalが必要な候補 | 48 | 102 | 150 |
| 直接費用pairの保存記録あり（系列接続未確認を含む） | 170 | 165 | 335 |
| 共通m5 native系列で受入済み費用pair | 28 | 36 | 64 |

「取得済み」は同じ公式・prefix・q・modeで数えた候補数であり、許可された最終費用ランキングの件数ではない。S2/S4の旧profile費用、後続PF直接compile、m5 native費用の原来歴と受入範囲を分けた。同じcompiler hashだけでwrapper・provider・runtimeの同一性を認定しない。旧H6 model費用は別区分で保存し、主比較の直接単価へ代用しない。

既存H4 D190の全登録、H6強化136、m5の現行64決定論signal、tol-only pilotの全18登録をjoinした。H6 pilotのYoshida4 B0/B0I p5/q2という追加2登録も保持した。重複を除いた現行登録は392候補（共通域外のH4 q16を含む）。旧H4 DF12の104登録と、古いm5 screen提案144行は別表へ保全し、採用・実行済み登録と混同しない。DF12の旧取得subsetを全域取得と表示しない。

H2は今回のDF10/DF19統一の対象外で、既存結果を変更しない。既存PRの候補・SOURCE・n・cohort・seed・raw・独立確認を変更しない。

## 固定した追加call予算（実行は未承認）

| 将来stageの上限 | H4 | H6 | 計 |
|---|---:|---:|---:|
| 新signal候補 | 48 | 102 | 150 |
| B0基礎signal | 24 | 51 | 75 |
| B0I解析scalar overlay | 24 | 51 | 75 |
| 独立forward signal照合 | 24 | 51 | 75 |
| 共通政策でのpaired shot/適格性評価 | 1,140 | 1,380 | 2,520 |
| 新compileを予約し得る最大候補pool | 149 | 224 | 373 |
| 最大compile軸（cosine/sine） | 298 | 448 | 746 |

signal不足の内訳は、H4 m5が48、H6 m5が56、H6 Yoshida4が46。H6 Yoshida4 p5/q2の保存2信号は再計算しない。B0Iは同じB0と保存snapshotから得るtail scalarを用いた一回の解析回転で得る。full-H referenceは回転しない。B0I回路費用をB0費用とaliasしない。

基礎75 signalにはsigned S2 piece step計2,409（H4 990/H6 1,419）、独立forwardは別callとして75回を予約する。新SCF/積分/DF/state/full-H reference/tail spectrum/PR samplingは0。signal contextは各系1 snapshot restore・1 saved reference read・1 full preparation・1 occupation oracle、追加prefix preparationはH4 6/H6 9を上限とする。各baseのforward経路は既存の独立occupation工程へ結合し、block-level作用の上限も機械契約に明記した。これらの実行bindingは承認後に固定する。

compile pool 373候補のうち、**217候補は旧直接費用のidentity接続待ち**（H4 97/H6 120）、**156候補は直接費用pair未取得**（H4 52/H6 104）。217件全ての接続に合格すれば、未取得側の上限は156候補312軸へ下がる。この156件にもsignal未取得150件が含まれるため、全てを実際にcompileするとは限らない。

poolの候補IDと条件分岐は事前固定した。同一εで両軸ELIGIBLE、かつidentity一致の受入済みpairがない候補だけを取得する。費用を見て候補を選ばない。適格性unknownを不適格へ落とさず、保存全endpoint不適格の77候補は、将来uを減らさない政策の下で新規compile対象外にする。残る受入済み64候補との重なりを除き、最大373を得た。

最大compile stageはnative evolution373、signed S2 piece6,374、low/high/独立occupation作用各373、wrapper生成746、uncompiled/compiled wrapper作用各746、compile746。context restore/preparation再構成は最大373でcache共有により削減可能。新sampling/reference/signal/shot/fit/rankingはこのcompile stageでは0。系別詳細と全IDは[未承認契約](../../artifacts/resource_applicability/track_a_pf_common_domain_preparation/2026-10-22/execution_contract_UNAPPROVED.json)に固定した。上限を使い切るための追加実行・予算間の振替・retry/resumeはしない。

## 適格性・shot・比較の政策案

主比較は通常signed bias、ε=.05/.01/.005/.001/.0005、allowance倍率1、α_axis=.025、元の凍結binary64 shot式とDecimal別算術照合を用いる。現行m5系列の経験的allowance式を全公式へ適用し、さらに保存u以上とする案を固定した。floor1e-9、安全倍率10、roundoff倍率64、u/headroom上限.01、log_B margin floor1e-12。これは未承認の共通政策案で、旧u=1e-11等を遡及変更したものではない。

保存shotはsignal・ε・bias・u・normalization margin・α・整数丸め規約が完全一致した場合だけ再利用する。64 m5候補の保存320 endpoint行も参照結合したが、政策一致を検査するまで無条件再利用しない。旧政策と共通政策のendpointを混ぜない。shot計算の上限2,520は全504×5の保守的な枠で、再利用・不適格判定により実際の算術callは少なくなる。

比較は二つに分ける。

主PR系列は既存のH4/H6 outer32/K4 EXPLORATIONの24群（各系3公式×q1/2/4/8、prefix3/5、無条件law）に固定する。通常signedの保存120 endpointとそのcohort/nを参照し、新統計・価格再評価は0。旧piece-r2 pilotや旧疎格子のPRは別の履歴panelであり、この主比較へpoolしない。[PR入力binding](../../artifacts/resource_applicability/track_a_pf_common_domain_preparation/2026-10-22/PR_input_binding_DRAFT.json)を参照。

1. 同じ公式のPR対PF：保存された固定PR候補と、その公式だけの共通PF格子最小を比較する。3公式×2系×5精度＝30 scope。
2. 3公式全体のPF対PR：同じ入力・共通格子の全3公式最小を分母にする。2系×5精度＝10 PF最小scope。保存PRの固定候補・探索選択の来歴を明示し、PRの再探索・統計再評価・確認標本の混合はしない。

費用は両軸のcompiled RZと固定analytic shotの積の和。状態準備P=0、有限Tの複素signal資源であり、energy/QPE/RPE全round最適化ではない。未解決候補があれば関連する最小値は集合全体としてUNRESOLVEDとし、取得済みsubsetだけの最良へ置換しない。新しいPF分母との比較を、以前の独立確認で確認済みとは呼ばない。

非相殺biasは成分欠測を一覧化する補助区分に留める。m5の旧B0/B0IにはPF/discard別成分がなく、今回の主signed比較の予算から新referenceや非相殺再ランキングへ拡張しない。経験的allowance・未観測PR rare event・母平均順位の限界を維持する。

## 比較可能性を阻む条件と実行順序

H6の28候補では、保存済み適格性・shotの原参照はあるが、共通uを算定するためのraw norm/closure/stage metadataが未回収。既存rootの対象名とexport ZIP一覧も確認し、ZIPはexporterコード4件のみで原signalではなかった。原metadataのSHA付き回収が必要で、欠測を推測・0補完したり、別予算のsignal再生成を暗黙実行したりしない。該当IDは[execution_blockers.json](../../artifacts/resource_applicability/track_a_pf_common_domain_preparation/2026-10-22/execution_blockers.json)に列挙した。回収できなければ共通域の完全比較はSTOPする。

H4の再生成snapshot容器は旧SHAと異なる一方、科学配列・Hamiltonian/state hashは旧保存値と一致する。この既存移行receiptを新packへ結合し、旧原本復元とは表現しない。H6は原snapshot SHAが一致する。旧profileと共通native系列の同一性は、値の一致する入力・preparation・係数・scalar/control・provider/runtime・compilerの証拠で接続する。profileタグや真偽flagだけでは接続しない。初回直接compile572軸のglobal final gate欠測など、既存受入の限界を消さない。

順序は入力/metadata・旧費用identity gate → H4/H6独立signal → 共通政策での適格性/shot → 適格かつ費用未解決の固定poolだけcompile → 終了監査 → 別途承認された範囲の保存値比較。各stageは新規grant・ledger・exclusive marker・独立STOPを持ち、成功や承認を次stageへ暗黙継承しない。使用済みgrant再利用、科学call消費後のretry/resume、候補追加、標本追加、q拡張はしない。

科学入力不一致、metadata欠測、allowance未解決、対の欠測、compiler/runtime不一致、native数値gate失敗、予算超過、保存SHA不一致、exit/reap不明は該当stageでSTOPする。compile vectorはCT01 <1e-7、他のnative gate <1e-9、phase alignmentなしという共通m5系列の既存閾値を保持する。旧STOPをPASSに書き換えない。

## 資源・実装・人工検証

準備readerはstdlibだけで58固定Git読取を並列化し、配列decodeやscientific importを行わない。新libraryは正規化・共通域・両軸適格性join・identity receipt・call cap・動的worker選定と人工callback one-shot基盤、runnerはmetadata生成/読取再現専用である。新23＋既存係数12＋signal人工18、計53件PASS。旧signal回帰は最初にhistorical provider import不足で停止したため、復元済み固定providerをPYTHONPATHへ明示して18件PASSを確認した。元科学実装は変更していない。

実行時workerはpending・実idle CPU/cgroup・空きRAM/観測RSS・空きscratch/観測live bytesの最小から選び、固定の低いCPU/RAM/worker/wall上限を置かない。compiler processとBLAS内側threadは1とし、外側の独立候補を並列化する。今回のaffinity/RAM/disk観測と、前回のRSS/保存量は[資源見積もり](../../artifacts/resource_applicability/track_a_pf_common_domain_preparation/2026-10-22/resources_and_estimates_DRAFT.json)に記録した。これは実行時fresh gateの代わりではない。

旧m5費用取得の3worker平均throughputを最大poolへ単純外挿するとH4約27・H6約196 worker時間の粗い容量proxyとなる。prefix/q・新旧環境・I/Oが異なるため所要時間の上界でも確定見積もりでもない。workerのRSSは旧同系列でH4約0.52 GiB、H6約1.67 GiBを観測したが、新格子の最大RSS・rawサイズは未測定。大きいcandidateは命令数proxyと保存値でsize別にadmitし、実測で資源見積もりを更新する。資源不足時は未起動taskを待たせるかSTOPし、科学条件や共有設定を変えない。

新しい専用rawだけを可逆圧縮し、原SHA/圧縮SHAとbyte復元検査後に新規一時fileを解放する設計。旧raw/grant/ledger/marker/STOPを変更せず、他ユーザーのprocess/cgroup/設定を操作しない。

## 保存・再確認

[bundle README](../../artifacts/resource_applicability/track_a_pf_common_domain_preparation/2026-10-22/README.md)、[504候補CSV](../../artifacts/resource_applicability/track_a_pf_common_domain_preparation/2026-10-22/common_candidates_504.csv)、[段階別契約](../../artifacts/resource_applicability/track_a_pf_common_domain_preparation/2026-10-22/stage_sequence_UNAPPROVED.json)、[共通比較政策案](../../artifacts/resource_applicability/track_a_pf_common_domain_preparation/2026-10-22/comparison_policy_DRAFT.json)。

metadataだけの再現確認：

```bash
python3 scripts/prepare_pf_common_domain.py --check --output artifacts/resource_applicability/track_a_pf_common_domain_preparation/2026-10-22
```

今回は追加科学call0、有効grant0、push0。専用branchへローカルcommitしてmandatory STOPする。次はユーザーによる共通政策・固定追加call予算の明示承認と、未解決metadata/identity gateの解消である。詳細科学レビュー・production・論文本文作成へ進まない。
