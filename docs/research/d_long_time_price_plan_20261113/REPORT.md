# PR-LD1/LD2の長時間compiled価格：評価・資格付け設計

2026-11-13 JST。**長時間の直接full compileを参照経路として実行可能性を確認し、同lawの構造平均＋paired残差を使えるセルだけで分散を下げる。短window・特徴量モデルは、長時間の誤差根拠が得られるまで正式価格にしない。** 現在、両候補のm12の適格なE[C]・区間はUNKNOWN。固定L_D1価格の高精度化や全52軸のcompileを先行条件にしない。

[方式・資格条件](LONG_TIME_PRICE_EVALUATION_PLAN.md)／[初回の有限科学案](FINITE_PRICE_SCIENCE_PLAN.md)／[B/C HANDOFF](HANDOFF.md)／[人工API確認5件](TESTS.txt)。今回はSOURCE読解、既存metadataの確認、人工metadataテストのみ。実H4 preparation/sampling/build/compile、保存科学値の新規数値解析、fit、新E[C]/区間/G計算は**0**。既存52軸requestは再生成していない。

## 判断の根拠と対象

B `23d74855135e718554d824a269427de9943213d1` の既存REPORTは、固定nominal N×ideal E[S]でm10〜m12が約96.92%、m12が約66.20%と記載する。これは**固定LD1の条件付き構造proxy**の引用で、正式Gの寄与率、LD2の寄与率、必要価格精度・nの配分根拠ではない。元JSON/CSVの再集計はしていない。

A `91349e7` のREPORT/HANDOFFでは固定LD1/LD2の全13round signalが供給済みだが、十分N・global RMSE・actual-law資格は未取得である。したがってBから有効なN又はN集合と比較契約が来るまで、正式Gに必要な価格セル・精度は確定しない。

対象はLinear H4/1.00 Å/STO-3G、保存tol-only DF10/sector(2,2)、同H/state/snapshot、元順prefix1/2とtail、one-body deterministic、S2/ΔPF=.1/K2/h=.1/M12・元exact有理数κ・P0。Cは**測定を含むcontrolled full Hadamard X/Yのcompiled RZ数/shot**で、状態準備費は除外する。元scalarは全Tで一回。新ZZ、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、symmetric_directional、target/backend/couplingなしを保持する。

| 既存metadataのm12 | q_PF | R_outer | logical event数 | 現在の証拠 |
|---|---:|---:|---:|---|
| PR-LD1 | 4096 | 8160 | 33,423,360 | D748の保存coverage。C/time/RSSの測定ではない |
| PR-LD2 | 4096 | 2559 | 10,481,664 | 同上。eventが少ないことからcompiled価格の優越は導けない |

## 長時間取得の処理上の課題

Fをevent数、Lをnative gate列長とする。FとLは同じでなく、order2のcomponent列・basis/head/controlでLが変わる。現在のbackendはstreaming compilerではない。

| 処理・確認した既存API | 保持物と長時間での負荷 | 再利用と残る課題 |
|---|---|---|
| `NativeRuntime.context` / pinned `prepare` | prefix別preparation・component/basis、signed piece別head | 同prefixのpath/roundでpreparationを再利用。head角度・full contextは候補request別。LD1/LD2は別Runtime |
| carrier `sample_requests` | 全q×pieceのrequest、各event・basis plan、seed/law | normal lawを維持。seedだけの保存は、将来の正確なpath復元・component SOURCE照合の代用にならない |
| fast `event_dicts/write/digest` | event objectに加え全dict列・JSON化・digest用表現 | O(F)以上の走査・一時記憶/I/O。既存原本のreplayは新IID標本ではない |
| `structural_pair` | 元Accountingのpending basis/Zと全eventの走査 | X/Y共通本体とRZ投影は既に高速化済み。Sの意味・順序を変えず再利用 |
| carrier `compose` | tail→partial circuitを順に全evolutionへcompose | 全evolution QuantumCircuitを保持。head/tailを局所compileして足す方式には変えない |
| efficient ZZ | `CX(s0,s1) CRZ(2u)(anc,s1) CX(s0,s1)` | `CONTROLLED_ZZ_CX_CRZ_CX_V1`。旧CCX経由の価格と混在しない |
| `build_native_hadamard_wrapper` | 完成evolutionを検証し、X又はYのwrapperへcompose | evolutionと選択軸wrapperが同時に存在。YのSdg・両軸のH/measurementを保持。軸は順次解放 |
| fast canonical fingerprint / mutation check | 定義も含む全内容の走査 | call内interningを再利用。full hashを不要とする閉包はない。event数/ID/角度なしkeyは不可 |
| `transpile_and_measure_cost` | full wrapper、compiler DAG/translation中間、transpiled circuit | opt1は全回路文脈を使う。RZ/CX count・filtered depth・total depth/sizeの走査も必要。定数メモリ保証なし |
| cache | full pre-transpile wrapper fingerprint＋compiler | 既存read-only cacheを照合、hit元を保存。現在のfast経路は新missの共有cache登録を自動では行わない。cache hitは新標本でない |
| 並列化 | pathごとの上記保持物が重なる | 独立pathはprocess並列。可逆lowering/hashのmodule差替えがあるため同processのthread競合を避ける。利用可能資源と実測RSSから並列度を決める |

現fastの`TECHNICAL_TIMING`はprice call内部であり、入力復元/preparation/sampling・全process peak RSSを網羅しない。後続runnerは既存stage記録にこれらを外から補う。hash/JSONとcompiler中間のpeakを混同しない。独立workerのpeakの単純和は同時peakの実測値ではない。

P2 REPORTのm6 replayではpath約881秒、worker peak約2.794 GB、各軸compile約242秒という履歴がある。旧P1とのpath時間比3.56は単回・計時境界/並列負荷差込みの観測である。これをm12のtime/RAM/価格へ比例外挿しない。新adapterでの実LD2・m12の実行可能規模は未測定。

## 評価手法の到達点

| 方法 | 確認された範囲 | 今回の判断 |
|---|---|---|
| 直接full compile | 旧固定LD1 m0/m6のpath参照、人工候補adapterの同値性。任意の新LD2実preparationは未検証 | domain内のpath Cを得る参照。巨大回路の実行可能性・全law平均資格は別問題 |
| 元Accounting / C kernel | Sの定義と元LD1全26 E[S]。compiler閉包ではない | 同law・同境界を確認したcontrol variate用。LD2へ旧値を移さずCに適合性を依頼 |
| 解析的compiled counter | 部分native/1q/CX bridgeに限定人工・別CDF8系列の証拠 | 現H4 DF10/K2/新ZZ/全opt1のcounterはない。全compiler再実装を新設しない |
| paired D=C−S | 同path取得部品と有限P2観測 | lawを合わせ、valid mean intervalが作れる範囲で使用。旧pilotのC/S比を移植しない |
| outer-window / feature model | connected恒等式、人工反例、低コストfeature API | 長距離tail・未測定prefix/q/R/lawでの誤差が未証明。正式m12価格には使わない |

S・event数はCの厳密な上下界ではない。旧pilot全order2未観測はrare寄与0の証拠ではない。モデルfit/新係数の採択もしていない。最も小さい次の案は、**選択された最大domainの一path・X/Yによる技術pilot**であり、平均価格認定ではない。候補・費用負担の選択肢と最大compile数は有限計画に示す。

## Cとの互換確認と今回の差分

C5ccはD0c6を固定ロードする。D748の`expected_scalar`省略時は旧request内容を保持するが、指定するとB identity/lawは同じままD context/request keyが変わる。C `_requests` の再構成はこのkeywordを渡さないので新requestを拒否する。keyword転送だけでは、schedule全体のscalar anchorとpathのresolved scalar照合が不足する。人工metadata5件でこの差を再現した。

これは既存C統合の再実装ではない。production moduleは変更せず、C担当へ小差分の仕様を[HANDOFF](HANDOFF.md)に渡す。テスト内AST変更はkeyword互換を試す限定実験であり、正式binding修正やscientific qualificationではない。

## 読解した固定SOURCE

| SOURCE | 主な確認対象 |
|---|---|
| `91349e703e0d09ef702ccd108745dfaf1acc29f7` | A signal REPORT/HANDOFF。科学値ファイルの再解析なし |
| `a9abf8de75bf3432fcd714e868e5c14c8884b8f8` | A固定MENU/configuration。D748のSOURCE-bound selector定義を再利用 |
| `748be6875b5007707d864e16aa7a4fc4250f3714` | 現基点、ld12_price_requests / scalar追加 / coverage/HANDOFF |
| `0c6e16342b3c804fa0bd89a4028d6b5bea7b0322` | candidate_price_adapter、from_a_job/from_b、Runtime/Adapter・quantity |
| `24b0a6eb1d806391a3c6ad454fb0c018148d8545` | 価格探索REPORT/HANDOFF、同時区間/候補差/N集合の設計 |
| `39ddd59edf54a3a901e5302207b2765ddbd03fbb` | P2 REPORT/HANDOFFと既存SOURCE。raw/数値JSON/CSVを再解析せず |
| `5a67cea7705e80ff295306606611d774724ae2e2` | compiled_price_fast、元S/compose/wrapper/fingerprint/cache |
| `952ce0113a27b179e4e6b889db124bdf37742fa1` | 方式REPORT/SOURCE、outer-windowの恒等式・未証明tail、compiler反例 |
| `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f` / `37cbc331b6cb9d38e83584f8e5b007da173bc5cd` | 既存全26構造平均・feature/kernelの元domainと、D既存HANDOFFで整理した再利用条件。新しい構造平均/feature値は取得せず |
| `5cc7c07e112ce97205948465c94441cba0042416` | C REPORT/PRICE_INTEGRATION_HANDOFF、rpe_d_price_connection / provider |
| `23d74855135e718554d824a269427de9943213d1` | B P0〜P4 REPORT/HANDOFFの既存記述のみ |
| `b9b9ffa6233db9dd2b413dc07b27e7039324c666` | rpe_shot_allocation の既存mean/profile/interval binding |
| C1 `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7` / carrier `155529e20a8fac39e00786c40b57830b71203c4f` / provider `c31ef424175c02379eafebe8fc6c7548c1d9aabe` | 固定Backend・trial_seed/sample_requests/compose、native wrapper、compiler・Accounting |

既存branch `codex/d-compiled-residual-c-minus-s-20261113` を継続。旧SOURCE/input/raw/results/KEEP/grant/ledger/marker/契約/STOPは変更しない。新しい計画・人工確認をcommit/pushしてremote照合後 **mandatory STOP**。ユーザーの有限科学承認まで新しい実取得へ進まない。
