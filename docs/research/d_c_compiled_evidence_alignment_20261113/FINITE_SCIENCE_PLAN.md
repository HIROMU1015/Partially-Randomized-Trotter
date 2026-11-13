# 最初の有限direct証拠案（科学実行未承認）

**推奨はE0だけを最初に判断する。** 旧LD2/m12 T2の再実行・未使用枠の移転ではない。今回行ったのはmetadata生成と人工回帰まで。E0/E1のSOURCE欄が未確定のまま実行できるcallerを作っていない。

## 選択案と目的

| 項目 | E0：初回提案 | E1：別保留案 |
|---|---|---|
| candidate / round | PR-LD2 / m0 | PR-LD2 / m4 |
| 物理ID | `ce7ef02e7926c1a6c924` | `cef49970538ca248d77e` |
| T / qPF / R / K | .1 / 1 / 1 / 2 | 1.6 / 16 / 10 / 2 |
| signed weights / allocation | [1.] / [1] | [1.] / [10] |
| piece時間 | .1 = `0x1.999999999999ap-4` | .1 = `0x1.999999999999ap-4` |
| 論理event位置 | 1 | 160 |
| axes / normal path数 | X,Y / 1（index0） | X,Y / 1（index0） |
| cohort | `D_C_PRICE_BRIDGE_LD2_M0_V1` | `D_C_PRICE_CHALLENGE_LD2_M4_V1` |
| 最大新compile | 2 | 2 |
| 目的 | source-bound totalの実guard、prefix2、measured XY、同path S/Cと出力形式 | 複数outerとPF/RTE・outer境界を含む同じ取得経路 |
| 統計的役割 | developmentのdirect pathのみ | 別development path。未使用holdoutの資格なし |

cohort/indexと**outer/pieceごとの全整数seed**は [FINITE_CELLS.json](FINITE_CELLS.json) に固定した。E0はseed `13095741627497599946`。E1は16 seedを全記載している。元carrier `155529e20a8fac39e00786c40b57830b71203c4f:scripts/h46_long_time_native.py:trial_seed`を既存 `p1_efficient_zz.trial_seed_function`で読み、元JSON表現からSHA256先頭16桁をuint64化する。同じpathのX/Yでevent列を共有し、別seedにしない。normal samplerのK2・order/component/joint law、元exactκからのq/Rを保持し、forced rareは0。

元A configuration IDは `1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae`。H4 DF10、保存元H/state/snapshot、prefix2、S2/ΔPF=h=.1、K2/M12/元exactκ/P0、新ZZと元compiler条件を固定する。descriptorのtail identityと保存constantを新selectorで束縛する。総event位置はRZ数・生成application数・time/RSSの予測ではない。

## SOURCE・出力先と承認前の残条件

| 項目 | 固定するもの |
|---|---|
| production基点 | D `e827367629ca6c2920d9d908fc1631314399db86`、本成果の公開commit、A a9ab、C1914、元provider/carrier/compiler/input bindings |
| 実行SOURCE | **未確定・未承認**。この技術成果を基に、選択cellだけのcaller差分を人工テストし、科学call前に新commitの完全SHAを固定する。そのSOURCEと有限scopeをユーザーが明示承認する。技術metadata tag/IDを実行SOURCE/IDへコピーしない |
| E0公開出力案 | `docs/research/d_c_price_evidence_acquisition_20261113/e0/run01/` |
| E0非公開出力案 | `/home/AbeHiromu/projects/d-c-price-evidence-20261113-private/e0/run01/` |
| E1出力案 | 上記の`e0`だけを`e1`にした別出力先。E0承認では使用しない |
| 現状 | 出力先が未存在であることをmetadata確認。今回作成・実行していない |

現時点で必要なのは、C/BがE0の接続証拠を必要とするかの確認と、ユーザーによる上記cell・axes・path・cohort/index/全seed・完全実行SOURCE・出力・call範囲の別途承認。E1はCの必要な検証方向を確認した後に別途判断する。Cのモデル仮説/予測をfreezeしていない今、E1をモデルholdoutと呼べない。**モデルがlaw平均を予測する場合、freeze後でも1 pathの差だけから精度を判定できない。** mean/分散/CI・fitは本案のcall範囲に含めない。

## 承認後に固定するcallerとcall予算

`ld12_price_requests.request_for`で選択requestを生成し、既存saved input loader、`NativeRuntime`、`Adapter.acquire`、fast observer/資源記録を呼ぶ小さいcaller差分だけを使う。旧 `run_ld2_m12_t2.py` はm12/q4096固定なので再実行・scope guard変更で流用しない。新しいrequest builder/compile pipelineは作らない。Cのmetadata namespaceはscience backend起動用ではなく、Dの既存provider環境を用いる。

**E0のみの場合の上限**：保存H/state入力復元1回、実prepare1回、normal sampling1 path、signed request batch1、full controlled evolution1、full measured wrapper2、cache missで新compile最大2。sample内部のcontext→acquire内部contextは同runtimeのcache lookupを再利用し、2回目のprepareをしない。scalar gateはsample内部の乱数/event生成より前に評価される。postcheckを追加する場合はその科学callを別に明示し、元cost_trialのstate/vector科学検証を暗黙に増やさない。

E1を**別途**承認する場合も1 restore/prepare、1 path、16 signed request batch、evolution1、wrapper2、新compile最大2。入力復元/prepare共有を目的としたE0+E1一括実行は未採択。いずれもretry/resume、別path/seed/round、追加sampling/compile、forced rare、fit/平均/CI/E[C]/N/RMSE/Gは0。cache hitなら新compileを減らし、そのfingerprint/compiler/physical provenanceを記録する。cache利用自体も新wrapper生成等の科学scopeを要する。

## 保存と判定

- source/input/binding、configuration/physical/round/law、request/context key、実preparation/head/tail/component/basisの取得可能hash、λ、fragment順、c_H・s_R・totalのdecimal/hexを同じ呼出しで記録。runtime完全一致guardの通過/停止を明示する。
- cohort/index/role、元seed列、normal/conditional=false、event digestとprivate event原本、order0/2・Z/ZZ、piece/outer順を保存する。
- 軸ごとのC、S、D、六指標（RZ/CX count/depth、total depth、size）、measurement付きwrapper fingerprint、compiler/lowering/source/cache参照を保存する。X=Yと観測されても片軸compileを省略しない。
- 元fast observerのevent/structural/compose/wrapper/fingerprint/compile/metrics/mutation時間、restore/sample/全体wall、process high-water peak RSSと実資源観測を保存。compose内部hashを外側時間へ二重加算しない。未知の巨大回路性能をこの小セルから外挿しない。
- 成功は**当該1 pathの取得成功**。Cの`bind_results`で受動mappingしてもmean/CIはnull、qualified=false。平均モデル誤差の閾値やconfidence、価格精度を新採択しない。
- scalar/SOURCE/入力/cache整合の不一致又は継続不能時は同一呼出しの部分結果を保全してSTOP。guard/tolerance/phase/source/seed/roundを自動修正せず、診断のための2回目prepareも行わない。

CPU/RAM/worker数に固定capを追加しない。一pathの元process-local lowering/hash swapは単一threadで扱い、共有環境・他者process/設定を変更しない。実際のOOM/継続不能と一時的競合を区別する。小セルの正確なwall/RSSも現時点ではUNKNOWNなので、架空の予算時間を示さない。

E0/E1からm12や他roundへ外挿しない。最大使用domainの主張が必要になった段階でのみ、Bの比較差・N/必要精度とCの誤差証拠に基づき、最大域参照又は別の妥当な価格区間の有限案を検討する。今回の技術成果を公開照合後 **mandatory STOP**。
