# T2 scalar不一致：preparation-only診断

2026-11-13 JST。**原因はDのrequest接続でtail-only identity係数をtotal scalarとして渡したこと。AのdescriptorとDの抽出identity係数はbinary64で完全一致するが、Dの合計scalarは別途Hamiltonian定数を含む。小さい丸め差が主因ではない。** 修正・新scalar採用は行っていない。

[DIAGNOSTIC_RESULT](run01/DIAGNOSTIC_RESULT.json)／[PLAN](PLAN.json)／[人工9項目](TESTS.txt)／[実行log](EXECUTION.txt)／[次の判断](NEXT_STEPS.md)。SOURCE **`244609df5b704723cae5ccd01f78468bca9588b3`** をcommit/push・remote SHA照合後、一回実行した。入力復元1回、context1回、fresh Runtimeのpreparation cache1件。元contextの一つのprepare呼出しを使い、scalar guardの例外frame内の値を保存した。sampling/event/path/native evolution/full wrapper/compile/T2 retryは**すべて0**、mandatory STOP。

## 実値とbinary64

| quantity | decimal（round-trip表示） | float.hex() |
|---|---:|---|
| Hamiltonian定数 c_H=`ham.constant` | 2.29310124732 | `0x1.25845777d3feap+1` |
| tail identity s_R=`prep.extracted_identity_coefficient` | 0.2223759094107018 | `0x1.c76d0552b2a1ep-3` |
| 元contextが実際に計算したtotal scalar `c_H+s_R` | **2.515477156730702** | `0x1.41fb27ccff28cp+1` |
| request expected scalar | **0.2223759094107018** | `0x1.c76d0552b2a1ep-3` |
| A保存descriptor scalar | 0.2223759094107018 | `0x1.c76d0552b2a1ep-3` |

各値の完全なdecimal展開・元binary64値・hexはDIAGNOSTIC_RESULTの`scalars`に保存した。`prep.constant_coefficient`もc_Hと同じ値/hexだった。

- total−requestのbinary64減算結果：**2.29310124732**、hex=`0x1.25845777d3feap+1`。
- 保存された二つのbinary64入力の厳密差：`41308839691812689 / 18014398509481984`。
- 符号付き隣接binary64表現のstep差：**15,666,813,000,271,982**。定義はscriptのordered encodingで、符号付きzeroを区別する。
- requestでの1 ULP：`2^-55`。厳密差をこの固定幅で割ると **82,617,679,383,625,378 ULP**。異なる指数binを跨ぐため、隣接表現のstep数とは別の量である。
- D抽出s_R−A descriptor：binary64減算0、厳密差0、ULP step差0。**tail identity側に観測上の丸め不一致はない。**
- 元順の加算の丸め残差：`stored_total − exact(c_H+s_R) = 1/18014398509481984 = 2^-54`。これは合計をbinary64へ丸める際の差で、今回の約2.293の不一致の説明ではない。

`ham.constant + A_descriptor.scalar`を診断式として評価した値は、元contextのtotalとhex一致した。この値はrequest/runtimeへ代入せず、元guardは同じValueErrorで停止した。差が科学的に無害という判定はしていない。

## 発生箇所とscalar定義

保存Hを `H=c_H I+H_D+H_R`、randomized tailのfull-Fock I/Z/ZZ展開で抽出したidentity係数をs_Rとする。deterministic one-bodyとprefix2は元headに保持し、tailからだけidentityを抽出する。

| SOURCE / 関数 | scalarの意味・演算 |
|---|---|
| carrier155529e2 `h46_new2_core.matrix_only_tail_coefficients` | 元順tail fragmentsの`exact_df_diagonal_coefficients`でsupport空の係数を集め`math.fsum`。**c_Hは加えない** |
| carrier155529e2 `h46_long_time_signal.context` | 上記戻り値を`prefixes[p].scalar`としてH4_context.jsonへ保存 |
| A a9ab MENU / provider fefd548e `load_h4_context` | 保存prefix2 descriptorをそのまま保持。値はs_R |
| A provider `make_context` / signal計算 | tail matrixは`sum(tail blocks)−s_R I`。最後に`exp(-i T (ctx.constant+s_R))`を一回適用 |
| provider c31ef424 `_prepare` / `extract_df_diagonal_tail` | 元順partition、`extract_identity_phase`、atol0。I係数を抽出し`prep.extracted_identity_coefficient`へ設定。c_Hは`prep.constant_coefficient`に別保存 |
| D NativeRuntime.context | **c_H+s_R**を元順で一回加算し、expected scalarとhex比較 |
| D748 `ld12_price_requests._request` | `expected_scalar=item['tail_descriptor']['scalar']` と渡す。**s_Rをtotalの欄へ結ぶ取り違え** |
| carrier `compose` | scalar一回会計はancillaの`p(-scalar*T)`。このscalarはc_H+s_R |

A保存descriptorとDの**抽出identity係数**は同じ物理量であり、今回の数値も完全一致した。Dの**合計scalar**とは定義が異なる。D selectorでのquantity取り違えが、今回のguard不成立を十分に説明する。元A signalがc_Hを落としているという結果ではない。Aの保存signal数値の再解析・再計算もしていない。

## 物理context・hashと確認限界

固定対象：Linear H4/1.00 Å/STO-3G、保存tol-only DF10/sector(2,2)、同H/state/snapshot、deterministic one-body＋prefix2、tail fragments2…9。元S2/ΔPF=.1/K2/h=.1/M12/exactκ/P0のm12 requestを用いた。q4096/R2559/T409.6はmetadataとして保持し、eventを展開しなかった。元prepare内部のGaussian basis loweringを含むpreparationだけを実行し、native evolution・event回路・full wrapperは構築していない。

| 項目 | 観測 |
|---|---|
| λ_R | 1.6030956277929231、hex=`0x1.9a64799dbb2f8p+0`。requestと既存context gateで一致 |
| fragment順 | deterministic=[0,1]、randomized=[2,3,4,5,6,7,8,9] |
| 抽出 | identity8個、randomized component288個、threshold drop0、`extract_identity_phase`/atol0/descending_abs/fermionic_gaussian_jw |
| H hash | `c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0` |
| preparation hash | `cdb4e1bf6beedfbf35a287c94503dd3d7e76c66dac61eb77e7917983e82be7a1` |
| tail hash | `b56b9a570e4f5a8b19f8c530d0efd808e898a74a8c9ffc5dbeb8b2d3c2195a7c` |
| component metadata hash | `b5402ce1e2f591719744528019edc405acd6bd5aace00bf56e20ab41b15e8d68` |
| tail basis metadata hash | `49aea56ef166f4fabd62e4aba3fbea5899f3ff61e13ce7c47fb71a4703f19302`。head3/tail8 basisのID/hashも保存 |

partition hash、個別component ID、basis定義metadataとhash規則、extraction/normalization診断はraw JSONを参照。component/basis集合のSHAは既存`D._key`を使った診断用content hashで、compiled circuit fingerprint/cache keyではない。Aの保存scalar/λ descriptorには同形式のcomponent/basis hashがないため、A/Dの全operator一致を追加認定したものではない。

## SOURCE・実行・STOP

- 停止結果基点：`752b9747a363eb4f1e510092230ef8dc243187ca`。元T2実行SOURCE：`8291f16ceb6b88cc8a43dbe508db419b9a26f373`。
- A menu：`a9abf8de75bf3432fcd714e868e5c14c8884b8f8`。保存descriptor index：`7a8bc7f132e166390c943e6b32c92cc6abe1baf4`、H4_context SHA=`8396605fa0c5ee34f29b436ee16e0f731cfaa6cd94f3304033106a85e9f5d52f`。A signal provider SOURCE：`fefd548ecd56f60d72031b3d385f6f5e6ee95d06`。
- Dのadapter基盤：`0c6e16342b3c804fa0bd89a4028d6b5bea7b0322`、request/guard追加：`748be6875b5007707d864e16aa7a4fc4250f3714`。今回これらのbodyは変更していない。
- 元prepare/loader/carrier：`155529e20a8fac39e00786c40b57830b71203c4f`、provider：`c31ef424175c02379eafebe8fc6c7548c1d9aabe`。NumPy1.26.4/Qiskit1.3.0の既存version gateを保持。

時間：input復元 **2.273191秒**、context（例外捕捉込み）**0.268758秒**、診断run_once wall **2.903016秒**。最後のwallはpassive preflightと最終JSON書込を含まない。process peak RSS **375,607,296 bytes（375.607 MB）**。CPU/RAM/workerの人工上限を追加せず、他process・共有設定へ変更しなかった。

既存のguard/prepare/body/演算順序・expected scalar、T2 run01、保存input2ファイルのSHA、旧SOURCE/raw/cache/grant/ledger/marker/KEEP/STOPを保全。新しいH/state/DF生成、sampling、full native build、compile、E[C]/Gは0。hash/元SOURCEの事後確認は[CHECKS](run01/CHECKS.json)に保存した。**原因特定と修正提案は別であり、結果commit/push・remote照合後mandatory STOP。**
