# C-S V0：次回の限定科学実行草案

2026-11-08 JST。**DRAFT_NOT_ADOPTED／SCIENCE_NOT_AUTHORIZED。今回の技術SOURCE公開後STOP。**

## 採択候補を一組に限定

対象は保存linear H4/1.00 Å/STO-3G/legacyDF12/8system/全4電子sector参照/L_D3/m0,q1/δ.02,r1,K2/coefficient_atol0、N1 actual Q/selected full。N1 744e1a0、N2 5e9bd08、保存D metadata fdf00ce、入力13件をSOURCE_AND_CHECKSに固定する。H/DF/partition/preparation/tail/CDF/state/認証の再計算なし。

技術SOURCEは**本directoryを公開したcommitの全tree**。最終チャットの固定SHAを使う。実行前にclean source worktreeとremote SHA/treeを照合する。別途の承認証明/grant/packetは不要。

| 項目 | 一つの提案 |
|---|---|
| 局所選択 | 保存canonical順のfirst eligible Dとfirst positive R/order0、各controlled logical gateのjoint factor0。最大2spec、他factor/basis/phaseはFIXED |
| coarse | π保持＋radianを最寄り1/256rad、tieは絶対値増加側、同表を3arm共有 |
| residual | exact有理差、r_res1/K_res2、ordinary degree3、root256/proposal160、η=ρ=10^-12、samplingなし |
| tool | NATIVE_PROFILEのstrict pygridsynth2.0.0、seed0、dps100、runtime/package/source hash固定 |
| precision/cap | 全primitive strict≤10^-6、finite≤10^-12、選択joint local mean bound≤3×10^-6。full error budgetは資格付与しない |
| native scope | 同じIZ parity ladder、T/T†・CX・Clifford・serial depth・W・古典費。literal localのみ、full compile0 |
| full science | state/Born/RNG/shot/N/G/full interrogation/18round/V1/他改善は全0 |

これらは**技術的に具体化した第一候補**。coarse格子、residual r/K/精度、局所error capとstrict profileをまとめて次回ユーザーが採択し、最大2specのscience実行を明示許可する必要がある。現在の依頼で採択されたことにしない。

## Stage A：静的preflight

専用branch/worktree、新しい空のoutput/cacheを使用。元main・旧STOP・grant・ledger・marker・C1凍結を保全し、旧合成cacheを流用しない。CPU/RAM/workerの恣意的capを設定せず、実際の空き資源/非干渉を確認する。

static-only入口は以下。SOURCE/13入力bytes、runtime/tool profileの照合だけを実施し、成功してもscienceへ進まない。

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/cs-v0-numba-cache PYTHONPATH=/tmp/cs-v0-provider-20261108:src /home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python scripts/tracks/algorithm_codesign/cs_v0_preflight.py
```

専用overlayが失われていれば、NATIVE_PROFILE/TOOL_IDENTITYと同じwheel/package/sourceを専用領域へ復元し、importとhashを照合する。共有環境を変更しない。runtime不一致のままfallbackしない。

## Stage B：一回の限定局所取得（今回未許可）

以下のAPI接続は実装済み。実行時には薄いserializerから呼び、既存rte_development.evidence.atomic_json/dumpsで各attemptをexclusive/fsync保存する。自動science起動CLIは設けていない。実行許可後のactivationはこの順序をそのまま使い、科学条件やkernelを変えない。

1. `Calls.invoke(binding,...)`一回の中でbound入力を読む。`bind_slots(...,policy=...)`と`resolve_r_angle(c,distribution)`を一回使用し、D/R各first joint factor0を固定。θhex/Fraction、coarse、residual、joint word、元slot/role、partner、selected basis、固定境界を保存。不存在・不一致は代替探索せずSTOP。
2. 各selected specのtarget/coarse/residualをPauliの±1固有空間の**2×2局所表現**で比較する（最大2組、6 rotation表現）。既存operator_target/interval行列を使い、実H4 full行列やstateを作らない。
3. `residual_table`を最大2回、各2eventで列挙。係数/proposal/weight、root幅、B_d/range/moment、truncation/finite meanを保存。実N1 pathの列挙/RNGなし。
4. unique positive target/coarse/event-angle requestsを固定し、各key一attempt。`request`→`acquire`→`checked`→負角なら`time_letters(actual adjoint)`。exact fast path以外はpygridsynth一回とindependent strict guard。失敗後の追加callなし。

## Stage C：局所結果（今回未許可）

`local_native`→既存native_accountingで同じparity ladderの局所価格を最大2組取得。coarse＋finite eventの各枝・補正係数和から局所加法期待価格を保存する。`local_bounds`でfinite＋B_d coarse error＋weighted native errorを合成し、事前局所capを検査する。

S0通常、S1両selected specへのcoarse/residual、S2 Dだけを記述する。これは**選択2spec内の資格**で、全H4のS1/S2実行結果ではない。通常価格とnative error、finite/range負担を並記し、局所Tが低くてもfull改善・energy精度・N/G改善としない。未取得basis/wrapper/partner/full費用はUNKNOWN。V1の強baseline/共通full予算は後続別設計。

## 有限call budgetと耐久記録

| call群 | 最大 | 備考 |
|---|---:|---|
| binding | 1 | R angle resolver1、選択D1/R1。実入力の再準備なし |
| local operator comparison | 2組 | 各target/coarse/residual 3表現。full9qubit matrix0 |
| residual finite mean/error/table | 2組 | 各2order、計最大4event、RNG0 |
| positive acquisition/guard | 8key | 2target+2coarse+2order×2residual。各one attempt、backend≤8、fast/zero/重複は減数のみ |
| local native price | 2組 | counts/literal schedule、実compile0 |
| original path sampling/state/Born/shot/全N/G | 0 | 不許可 |

CallsはSTARTをproviderより先にwriterへ渡し、RETURN/例外STOPを保存する。durable writerは既存atomic_json。source/input/profile/slot metadataと全callのunique token、失敗、取得sequence、guard、局所価格、古典CPU/wall/peak memoryを残す。未知のcall、duplicate、上限超過もSTOPを保存。新規outputはexist_ok=False相当で予約し、既存runへのresume/retryはしない。

8keysはfactor0のみ・r1/K2・2spec・IZ ladder・固定phaseがexactという**この政策の計数**。両controlled factorを替える、別K/追加gateを要する場合は条件不一致STOP。負角・event scalarにbackendを足さない。

## Stage D：保存・公開・mandatory STOP

新outputへ結果、SOURCE/environment/input hash、slot/angle/coarse/residual/table、exact/proposal/weight、strict rows/tool/price/費用、error内訳、call journal、失敗時prefix・UNKNOWN・STOPを保存。専用branchでcommit/push、公開SHA/body/blob/treeを照合しmandatory STOP。

SOURCE/input/profile/role不一致、unsupported angle/gate、selected basis不成立、phase喪失、strict/finite/local cap不成立、finite law/support不能、非有限interval、duplicate/key/callcap不整合、output衝突、共有非干渉不成立で即STOP。science retry・別slot・精度/r/K/格子/seed変更・追加合成なし。

この草案はV0のみ。real state/Born、fresh quantum shot、original/residual joint-law sampling、full compile、N/G、V1、C-N/C-P/SCV、旧STOP解除は自動開始しない。
