# H4 保存RPE結果の自動比較：B・B2・D・D2・統合担当への引渡し

同じコマンドでINPUTS.jsonの固定commit/path一覧を読み、公開済みQ3判定と保存条件付き費用を照合して方式別最安点を返す。対象は **指定した保存候補集合内の最安適合点**。新しい信号・価格・N・Q3や全配分の探索は実行しない。

## LD9 m6直接価格とLD7 PF予算4配分（2026-11-16、現行入力）

基点bc6409309で未反映だった次の2成果を追加し、従来と同じ集計コマンドを再実行した。実行SOURCE `71c77b07b2a72022df3bc1d855da42438c21928b`。

- `23160cc876807c0300be3de811f767b5810640a7:docs/research/d_pr_df_ld9_m6_price_20261116/HANDOFF.md` を入口として、同commitの `C_HANDOFF.json` / `COMPARISON.json` を使用。元機械可読公開d85150cd1b51f87da4e767b32cf52f532765d358、実行/比較SOURCE 8f0a82842481270e0581658b5294ecd59e79104f、原LD9 raw 6392aefdad0438e619cba0c0e9e55d170eb99a49を保持。
- `7ccb3e63b687bd13426f3e53d96b93d918707641:docs/research/pr_df_ld7_pf_budget_q3_results_20261116/C_RESULT_JOIN.json` の4配分。既存3配分のprice model/indexを再利用し、新規に数値探索やQ3を実施しない。

### LD9は同じV0配分のままm6価格を追記

既存LD9 V0のpoint ID `8b2586c294661c9da0db082d`、allocation、Q3適合を保持し、候補数は増やしていない。`conditional_cost=null`、全round用の `price_axes=[]` を維持。m6の2軸は `partial_price_axes` にDIRECT_FINITE_COHORT_MEANとして別記し、全13round価格と区別する。

|軸|LD9保存平均 RZ/shot|LD7保存平均 RZ/shot|保存差 LD9−LD7 RZ/shot|
|---|---:|---:|---:|
|X|492573|1575349/4 = 393837.25|394943/4 = +98735.75|
|Y|492573|1575349/4 = 393837.25|394943/4 = +98735.75|

各軸4 path。LD9 cohort=`D_PR_DF_LD9_M6_DIRECT_PRICE_20261116_V1`、LD7 cohort=`D_PR_DF_LD7_PRICE_CAL_20261116_V1`。分割内X/Yは同じ4 pathを共有し、分割間は別cohortであり、indexによるpaired解析・pool・再平均はしていない。平均・差・比の再計算も行わず、D2のexact保存結果を転記した。

同じLD9 configuration `ea087a7e3b311be7efae27558347e9eb471f616b8b55f59cbef8e506dda7a833`、m6 physical `09fa97187d6eee510d0c` を現行V0へ照合。LD7 physicalは `e7194af048772d7160c6`。T6.4/q64/R1のrequest identity/lawと元configuration、compiler/control/full measured wrapper/P0、cohort別4path×X/Yの完全性を確認し、16軸それぞれの元receiptのRZ整数・fingerprint・compilerを公表表と照合した。原SOURCEのnew-ZZ lowering、scalar期待値、path seed/event digest、元gate statusはrequest/pathのlocatorを保持。元full-wrapper statevector/numerical-action検証は未実施のままで、新しい科学的資格は付けていない。

追跡方法：同じLD9点の `partial_price_source` → C_HANDOFF、`partial_price_axes[*].comparison_source` → COMPARISONの軸別保存平均/差、`path_sources` → LD9の元4行 → receipt/path_receipt/execution_source。比較側LD7の8行も同C_HANDOFFのcohortsから原 `5c5d87ab32d03bb107a9a6bfed3cba253aef2aa0` のreceiptへ戻れる。通常の26軸MODEL providerとは混ぜず、qualified_for_candidate=false。m6の一部分だけから全law E[C]・CI・全round総額を作っていない。

### LD7 PF予算4配分を不適合点として追加

|配分|保存exact費用 RZ|Q3判定|point ID|
|---|---:|---|---|
|LD7_P7_S2__A_SHAPE|704871771/2|モデル不適合|c737e2d277b64fc4633559fb|
|LD7_P7_S2__B_PRICE|24616794490/63|モデル不適合|bb8ce66de85bb3c9d34ab9b4|
|LD7_FULL_PF_S2__A_SHAPE|22577997965/42|モデル不適合|d5099640f810c274c2d423b7|
|LD7_FULL_PF_S2__B_PRICE|34533437597/63|モデル不適合|50d124c114f7171f6d66e454|

固定INPUTSのcase/price_terms、元Q3 RESULTのallocation・configuration/total view/binding・保存判定を照合。各26軸の価格SOURCEは既存LD7モデルと同じことを確認した。費用・予算・保存比較・m7〜m12割合は元recordへの参照/転記で、新規集計・予測はない。price_evidenceに既存モデル/較正/holdoutのlocatorを再利用し、26価格のMODEL・qualified_for_candidate=falseを維持。4件とも数値失敗ではなく保存Q3のモデル不適合として選択から除く。既存V0/A_SHAPE/B_PRICEの3配分と重複しない。

**PR-DFの保存集合内最安はLD7 A_SHAPE・155652932221/126 RZ（約1,235,340,731.912698）のまま**。LD9 V0はm6価格あり・全round費用未取得の価格待ち。その他の旧42点と全方式の最安値はbc6409309版から不変、LD9自身の配分・Q3・null総額も維持。

現行は固定入力18件、SUMMARY展開を含む集計入口21件、候補47点（適合32／モデル不適合13／価格待ち2）。重複除去4件、旧信号のみ履歴1件。技術/保存metadata回帰10 PASS（部分価格を先に読んでも配分は1件、誤physical ID・Q3判定改変を拒否）。既存Git reader/cache、make、元価格/Q3索引、最安比較器を再利用し、この2schemaのconsumer差分だけ追加した。§10.5の現在catalogと関連履歴に同等反映はなく、今回未反映分のみ追加。他担当worktreeや原科学データに変更なし。

ロードマップ差分：LD9はm6のnormal4path・X/Y有限直接平均とLD7比較を追跡可能になったが、13round費用はnull。LD7のPF予算4列は全てモデル不適合として一覧へ追加、既知最安は不変。本体更新は統合担当へ委ねる。新signal/sampling/build/compile/fit/配分探索/Q3は各0、公開・remote照合後mandatory STOP。

## PR-DF LD9共通V0の追加（2026-11-16）

公開版 `ede7d80707249222b3ff676394c4a7abf65fb979` を基点に、`778118926c85158f947928c1e744d565464aab17:docs/research/d_pr_df_ld9_v0_results_20261116/C_HANDOFF.json` を16番目の固定入力として追加し、従来と同じコマンドを再実行した。実行SOURCE `f7b9e70850296147ca0e5ef9cc544f3f315b0831`。

追加したのは `LD9_V0` の1点、point ID `8b2586c294661c9da0db082d`。状態は **Q3適合・価格待ち**、conditional_cost=null、price_axes=[]。保存MSE上界の表示は4.28410123342266259328471×10⁻⁷ Ha²で、正本はC_HANDOFFと元Q3_RESULTの同一exact有理数。値・判定の再評価はしていない。

原configuration `ea087a7e3b311be7efae27558347e9eb471f616b8b55f59cbef8e506dda7a833`、total view `4fe264b899b1ecf51f6047c31bf3a4fd422d3c795a36defcf249cdfe502adbab`、binding `1d868513437a4efac8fe26b3be8061716318de290342f846ac22627249dfe078`。X/Y各 `[64,60,56,52,47,43,38,34,30,25,21,17,12]`。固定Q3 INPUTSと元RESULTについてconfiguration・total view・binding・allocation・law・保存判定・MSE上界を照合し、既存makeのsnapshot/精度/target/P0/profile検査を再利用した。

`run01/CATALOG.json` の `configuration_source` / `case_source` / `signal_source` / `q3_source` が、元configuration、固定入力、LD9信号、`560ef11e734644ea9d0df00fc49bd90a75dda2a8:docs/research/d_pr_df_ld9_v0_results_20261116/q3/LD9_V0/RESULT.json` へのlocator。`shared_V0_reference` はC_HANDOFFのsame_V0_LD7への参照だけで、同じ整数shot vectorの比較根拠を保持する。LD7とは別の物理候補・価格domainであり、再登録や価格移送は行っていない。

**PR-DFの指定保存候補集合内最安はLD7 A_SHAPE・155652932221/126 RZ（約1,235,340,731.912698）のまま**。LD7 V0は1件、LD7全体は既存3点のまま。LD9は価格未取得なので最安値の選択対象外。全42既存点と全方式最安はede7d8070版と完全一致。Pauliの不適合3点とFULL_PF__B_PRICE、粗幅Y4 A_SHAPEの価格待ち/nullも保全。

現行は固定16入力（参照SUMMARY等を含む読取文書19）・候補43点、適合32／モデル不適合9／価格待ち2。重複除去4件、旧信号のみ履歴1件。人工/保存metadata回帰9 PASS。新schemaを既存reader・makeへ接続する最小差分のみ追加し、LD9を先に読んでもLD7や他split価格を取り込まないことを確認した。§10.5の関連履歴と基点catalogではLD9接続はなく、今回不足していた1点だけを追加。他担当worktree・原結果・SOURCE・STOPは変更していない。

ロードマップ差分：PR-DF LD9共通V0はQ3適合・価格待ちとして追跡可能、固有費用はnull。最安適合点・既存価格資格に変更なし。本体更新は統合担当へ委ねる。新signal/sampling/compile/fit/配分探索/Q3は各0。公開・remote照合後mandatory STOP。

## PR-Pauli LD3のPF予算4配分（2026-11-16）

固定 `fb4d78781e2a4c63c9d334af246807785e669d6c:docs/research/pauli_ld3_pf_budget_q3_results_20261116/C_COMPARISON_JOIN.json` を15番目の入力に追加し、従来と同じコマンドで比較表を更新した。現行実行SOURCE `fe462c24d5c046efcef7afcdc43038319707114d`。recordsの4件は次のとおり。

|配分|保存Q3判定|保存exact費用 RZ|point ID|
|---|---|---:|---|
|P7__A_SHAPE|モデル不適合|30009021696/77|07dbc27e672c3da723a64ae9|
|P7__B_PRICE|モデル不適合|456463851968/1155|7b90bfc7c2de659a43961666|
|FULL_PF__A_SHAPE|モデル不適合|10729281172/21|9fe5181d82bc7c5472aba4ad|
|FULL_PF__B_PRICE|適合|625936421828/1155|06f67f6f4ae4a7390d403c96|

**PR-Pauliの指定保存候補集合内最安はLD3 FULL_PF__B_PRICE**。条件付きモデル参照費用は約541,936,295.95 RZ、X/Y各 `[48,45,42,39,36,33,29,26,23,19,16,8,2]`。保存SUMMARYの符号付き差T−PFを転記し、Full PF S2より17,521,090.05 RZ低く、p7 S2より145,973,617.95 RZ高い。exact差はそれぞれ `-20236859002/1155`、`168599528738/1155`。比較基準はこの結果で固定された559457386、395962678 RZを保持し、過去のLD2の予算や判定SOURCEは更新しない。

各recordのoriginal configuration・total view・binding・allocationを固定INPUTSのcaseと照合し、元API_RETURNのtotal configuration/binding/配分も一致することを確認した。4件×26軸について、original/total candidate・law・N・モデル参照単価・固定weighted termの元保存表との対応を照合。Q3判定・MSE区間・exact総額・PFとの差は同directoryのSUMMARYから転記したもので、新しいQ3・価格予測・N×C・差額計算は行っていない。

価格は既存A `f25a05f8a` の26軸を再利用したMODEL参照。古いVPで記録されたNは1shot価格の属性ではないため、新配分のNは今回のINPUTS/APIと照合する。全26軸のMODELとqualified_for_candidate=false、E[C]/CI/正式G未資格を保持。直接取得したのは元の別較正cell Q8R660の独立2path×X/Yのみで、新4配分を新しい価格観測や独立cohortとして数えていない。

候補IDからの追跡は `run01/CATALOG.json` のpoint IDを起点にする。`result_sources` はC_COMPARISON_JOIN.records、`case_source` は固定INPUTS、`q3_source` は元API_RETURNのrecord selector、`price_axes[*].price_source` は元PRICE_26AXISのモデル単価、`physical_context_source` は旧B索引の物理対応、`price_handoff_source` はA結合表。`price_evidence`には既存の係数・scalar/lowering・別較正cell/元receiptのSOURCEを再利用した。`saved_PF_comparisons` に今回の保存exact差とPF基準を保持する。

LD3 VPとLD2 OWN-Bは独立した適合点としてそのまま残す。安い3点はモデル不適合として費用付きで一覧へ残し、最安適合点の選択から除く。既存38点は旧3eed9dff5版と完全一致し、他方式最安・粗幅Y4 A_SHAPE価格待ち/null・B_PRICE不適合と旧UNKNOWN履歴も不変。

固定15入力（参照SUMMARY等を含む読取文書18）・候補42点、適合32／モデル不適合9／価格待ち1、重複除去4件、旧信号のみ履歴1件。人工/保存結果の接続・回帰8 PASS。入力順序を入れ替えても旧VPを保持し、新規4配分が1点ずつ追加される。configurationや価格資格の改変を拒否する。既存Git reader/cache、価格受入、make/選択器を再利用し、今回の公開joinだけを最小拡張した。§10.5の関連履歴に同じ4点のcatalog接続はなく、旧Pauli最安はLD3 VPだった。他担当のworktree・原結果・STOPは変更していない。

ロードマップ差分：PR-Pauliの保存集合内最安をLD3 VPから同じLD3のFULL_PF__B_PRICEへ更新できる。Full PF S2より条件付き参照費用は低く、p7 S2より高いが、MODELの科学的資格は引き続きfalse。全空間最適性や方式winnerは認定しない。ロードマップ本体は統合担当へ委ねる。新signal/sampling/compile/fit/配分探索/Q3は各0、公開・remote照合後mandatory STOP。

## 粗幅p7 Yoshida4 A_SHAPEの追加（2026-11-16）

固定 `8855f420a9fe2577efc77f099c5a39ab0cb9b7e6:docs/research/d_p7_y4_coarse_a_shape_q3_20261116/C_RESULT_JOIN.json` を14番目の入力として追加し、LD7接続を含む同じコマンドを再実行した。現行実行SOURCEは `4a0c5370fba0f2541ab4f17c737cdbf5da50af64`。既存の固定配分Q3 readerをそのまま利用でき、今回のA_SHAPE追加によるlibrary変更は0行。

|粗幅32h/5・標準Yoshida4・p7|保存Q3|条件付き総費用|point ID|
|---|---|---|---|
|B_PRICE|モデル不適合|未取得・null|既存行を保持|
|A_SHAPE|適合・価格待ち|未取得・null|8c95bb669cc1766e4e3427e0|

両点のconfigurationは `dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0`、total viewは `97ffe0a66f2dc14fa4b3264e885428216a2716c7a8a288085ddfe3b8a0e622ec`。配分ID・整数vectorが異なる独立した比較点として保持した。A_SHAPEは `B0I_P7_Y4_32H_OVER5_A_SHAPE_SAVED_INTEGERS_01`、X/Y各 `[37,34,32,29,27,24,22,19,17,14,11,9,6]`。26行のoriginal candidate・m/axis/N、固定配分SOURCE、total view、exact閾値・保存判定を照合した。`q3_source` と `q3_axis_sources` が元Q3結果と26軸へのlocator。

A_SHAPEのprice_axesは空、conditional_costはnull、最安値の選択対象外。現幅16h/3の価格を転記していない。旧B_PRICEの不適合・費用nullと元UNKNOWN履歴を保持。p7の保存集合内最安はS2・395962678 RZのまま。

**現行結果**：固定14入力（参照SUMMARYを含む読取文書17）・候補38点、適合31／モデル不適合6／価格待ち1。PR-DF最安は下記LD7 A_SHAPE・155652932221/126 RZで、費用の95.945359%は長roundモデル外挿。既存LD1/LD2の6点とPauli LD3 VPを含む非LD7の従来34点は68b072c5a版と完全一致。重複除去4件、旧信号のみ履歴1件。追加後の回帰7 PASS、新科学call0。

ロードマップ差分：粗幅p7 Y4は、B_PRICE不適合に加えA_SHAPEをQ3適合・価格待ちとして併記できる。Dの粗幅固有価格が公開されるまでは費用null。LD7の価格接続・最安点更新と合わせて統合担当へ引き渡す。ロードマップ本体は更新せず、公開・remote照合後mandatory STOP。

## PR-DF LD7の価格・Q3接続（2026-11-16）

固定入力 `fd010b111e95411702be51adc73e56b0bba6c093:docs/research/d_pr_df_ld7_price_results_20261116/C_HANDOFF.json` を13番目の入力に追加し、従来と同じコマンドを再実行した。実行SOURCE `dfae17d15e06694efb0e433fd7353fe1260a6ca7`。C_HANDOFFから同directoryのSUMMARYへ進み、全9点のうちLD1/LD2の6点は元 `8ec7c882806809460d38a638a891bafcf18fafd1` のmenuとexactに同一であることを確認し、既存consumerの結果を再利用した。6点を新しい配分・観測として追加していない。

|LD7配分|保存Q3|条件付きモデル参照費用（exact RZ）|point ID|
|---|---|---:|---|
|V0|適合|23363612385/14|c6ea64134e2684239de8bd55|
|A_SHAPE|適合|155652932221/126|ab1d8980b9d136b73e43b0f3|
|B_PRICE|適合|52109807273/42|a0e29a565bc0c8262d3cc814|

**PR-DFの指定保存集合内最安はLD7 A_SHAPE**。表示約1,235,340,731.912698 RZ。選択配分はX/Y各 `[48,45,42,39,35,32,29,25,22,19,16,12,9]`。V0は旧「Q3適合・価格待ち」の同じconfiguration/配分へ価格を接続し、旧状態とSOURCEをprice_availability_historyへ保持。A_SHAPE/B_PRICEの2点だけが候補数の増分である。

原configuration `ce9c7f08234089f0b50fd786c20403fe02c5de9073e89d6d35a12ee9d966c5e1`、total view `f00769918f264994d62080929bced34a3ac576e2aa47a1d783abd32360a7a9f4`、binding `a122e536aabe6f84b8d6e2eac6e342051676477e7a4f26d743582c540adcb9b8` を使用。各点の保存Q3 RESULTとそのINPUTSから同じ配分・total view・configuration・精度契約を照合し、各点26軸について価格のphysical/canonical ID、m/axis、q/R、law、Nを元configurationと照合した。保存Q3判定・MSE上界と保存総額を転記し、Q3・配分探索・価格予測・N×C再計算は行っていない。

価格は `C_ref=alpha+beta*q`（R=1、測定付きfull controlled wrapper）の保存MODEL参照。全26軸のprice_kind=MODEL、qualified_for_candidate=false。m0/m6の較正cell、m4の独立holdoutの元roleとreceiptを保持し、holdout判定はNOT_ASSESSED_NO_ADOPTED_TOLERANCEのまま。新しい価格資格・E[C]/CI/正式Gは認定しない。

**最安点の費用の95.9453590253993779917152%はq>64のm7〜m12へのモデル外挿**（SUMMARYの保存内訳を転記）。exact割合は `149341764653/155652932221`、m7〜m12部分額は `149341764653/126 RZ`。これは未知の誤差を0とする保証ではない。モデルの外挿範囲と有限cohort由来の未資格状態は変えていない。

追跡方法は `run01/CATALOG.json` の上記point IDを起点にする。

- `configuration_source` / `case_source` / `allocation` / `q3_source`：元configuration・配分・Q3原API保存結果。
- `price_axes[*].price_source`：`4326422a4633ed46968d6e348fe5ffaa24fbb468:docs/research/d_pr_df_ld7_price_results_20261116/AXIS_PRICES.json` の26行。元lawとモデル適用区分を保持。
- `price_evidence.model_source` / `calibration_sources[axis]`：同commitのFROZEN_MODELS.json、各軸のm0/m6較正record。そこからpaths[*].receipt/path_receiptで元のcompile・event証拠へ戻れる。再build/再compileなし。
- `price_evidence.holdout_source`：`c160ffb2f3d1e83de80db4ec6eeac97271569897:docs/research/d_pr_df_ld7_price_results_20261116/HOLDOUT_COMPARISON.json`。記述的比較の保存結果へ参照し、新しい合否を付けない。
- `saved_cost_breakdown`：m0〜m6、m7〜m12の部分額と後半割合の保存exact値・表示値。

固定13入力（参照するSUMMARYを含む読取文書16）・候補37点、適合31／モデル不適合6／価格待ち0。重複除去4件、旧信号のみ履歴1件。LD7以外の34点は旧68b072c5a版と完全一致、PR-DF以外の方式別最安も不変。PR-Pauli LD3 VPとLD2 OWN-B、p7粗幅Y4のモデル不適合・費用null・旧UNKNOWN履歴を含めて保全した。

技術検収7 PASS。既存の固定Git reader/cache・Q3照合・価格binding・exact最小値比較を再利用し、この公開schemaの接続差分だけを追加。新入力が先に来ても既存V0と6再掲点が重複しないことを検査した。§10.5の関連code/成果物履歴では同じLD7価格接続は未実装で、旧V0は価格待ちだった。他担当worktreeや科学結果は変更していない。

ロードマップ差分：PR-DF LD7の3配分を価格付きQ3適合点とし、保存集合内のPR-DF最安をLD2 B_PRICEからLD7 A_SHAPEへ更新できる。後半95.945359%の外挿依存と価格の未資格を併記する。ロードマップ本体は統合担当へ委ねる。新signal/sampling/compile/fit/配分探索/Q3は各0、公開・remote照合後mandatory STOP。

## PR-Pauli LD3 VPへの価格接続（2026-11-16）

Aの固定 `f25a05f8a92aab43cfbe30fc80b974054156aac1:docs/research/a_pauli_ld3_cal_price_20261116/HANDOFF_BC.json` を12番目の入力に追加し、上記と同じコマンドを再実行した。実行SOURCEは `e9446ec014fd5cb85c5854bef95eac7c72c7255e`。BのQ3正本は `2de33a4f4daeef7ca899dc98689974bb0eaadaf4:docs/research/pauli_ld3_vp_q3_results_20261116/C_RESULT_JOIN.json`。新しい配分探索を待たず、指定された保存集合だけで比較している。

- original configuration：`dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847`
- total view：`6757998a50bec243e1cd14d1b968457c673d383c47cb95c6b52e8b0ed7e1a0eb`
- binding：`122900d26d9412a569496e8ef98ff425614eac9885e43164bab400991980a858`
- allocation：`PAULI_LD3_USER_VP_MATCH_LD2_01`。X/Y各 `[48,45,42,39,36,33,29,26,23,19,16,13,9]`

26/26行について、original/total configuration、original/total candidate、binding、allocation、m/axis/NをAとBの索引で照合。q/Rとphysical IDは元configurationの全13roundおよびAの元価格行にも照合した。Aの保存BINDINGからsplit/dictionary・compiler/control/full measured wrapperとscalarの出所を保持した。新しい予測、価格平均、N×C、Q3は計算していない。保存exact総額 **173557305044/165 RZ（1,051,862,454.81）** を転記し、既存 `PAULI_LD3_VP` を同一配分の価格付きQ3適合点へ更新した。

機械可読正本は `run01/CATALOG.json` の `candidates[point_id=9c28225e02fab223cd269594]`。`q3_source`/`q3_axis_sources` がB、`price_axes[*].price_handoff_source` がAの結合行、`price_source` が保存モデル適用行へのlocator。`price_availability_history` に旧PRICE_WAITINGとそのSOURCEを残す。`price_evidence` に係数SOURCE、scalar/lowering、別較正cellと4軸receiptの元locatorを保持する。

**価格の意味**：直接取得は非round較正cell Q8R660の独立2path×X/Yのみ。13round×X/Yは26件とも `MODEL`／`MODEL_APPLICATION_FROM_Q8R660_DECOMPOSITION`、`qualified_for_candidate=false`。較正cellをroundのdirect receiptへ付け替えず、期待compiled価格E[C]・CI・正式Gの資格は与えない。Aの旧LD2比較基準やBの事前固定予算も改変しない。

保存集合内のPR-Pauli最安点はLD2 OWN-BからLD3 VPへ更新した。LD2 OWN-B（220695144335642870/90343469 RZ）は独立したQ3適合点として残る。Full PF=559457386、B0I p7=395962678 RZを含む他方式の最安値は不変。PR-DF LD7 V0は引き続き価格待ち・費用null。粗幅p7 Y4のモデル不適合・費用nullと旧UNKNOWN履歴も保持した。

現行は固定12入力（依存するSUMMARYを含む読取文書14）・候補35点：適合28／モデル不適合6／価格待ち1。重複4件を統合し、旧信号のみ履歴1件を保全。LD3以外の既存34点は旧015f14be9版と完全一致。6 test PASS（価格入力がQ3索引より先の場合、identity/配分/Rの改変拒否、実CLIのsource追加、既存回帰を含む）。metadata不一致0、新signal/sampling/compile/fit/配分探索/Q3は各0。

§10.5の差分確認では既存catalogのLD3は価格待ちで、関連code/成果物の履歴に同じA価格接続は見つからなかった。既存reader・Q3受入・比較器を再利用し、この公開schemaの接続だけ追加した。他担当worktree・private QPY・科学結果は変更していない。

ロードマップ差分：PR-Pauli LD3共通VPを「Q3適合・価格待ち」から「Q3適合・26軸モデル参照費用あり」へ変更できる。保存集合内のPR-Pauli最安点をLD3 VPへ更新する。全law E[C]とモデルの科学的資格は引き続き未取得。ロードマップ本体は統合担当へ委ねる。公開・remote照合後mandatory STOP。

## 再実行するコマンド

作業branchは `codex/h4-saved-rpe-cost-catalog-20261116`。旧C価格branchの固定6ad23def0から分離した専用worktreeで、他担当branchは変更していない。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/c-h4-saved-cost-catalog-20261116
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B scripts/collect_saved_rpe_costs.py --inputs docs/research/h4_saved_rpe_cost_catalog_20261116/INPUTS.json --output docs/research/h4_saved_rpe_cost_catalog_20261116/run01
```

`INPUTS.json.sources` に `{"commit":"固定SHA","path":"repo内のSUMMARY/C_RESULT_JOIN/HANDOFF.json"}` を追加して同じコマンドを再実行する。既存SUMMARY群と同じproducer形式ならcode変更は不要。臨時に追加する場合は `--source 固定SHA:repo内path` を繰り返してもよい。Git objectが未取得なら先に通常のfetchで取得する。runnerはネットワークや未公開worktreeを探索しない。入力略SHAは40桁に解決して出力へ保存する。

出力は指定directoryの `CATALOG.json`（候補・方式別最安・完全locator）、`REPORT.md`（比較表）、`INPUTS.json`（実際の入力一覧）。この生成directoryを同じコマンドで更新する。元研究の固定Git blob、原API、SOURCE、価格/配分・旧STOPは変更しない。過去版はGit履歴へ保持する。

## 初回固定入力と追加指示

初期6成果（b4296b110、cbc471c91、738df2262、cdbce383f、8ec7c8828、6ad23def0）に、作業中の明示指示によるa006b8e15、2de33a4f4、0defa286c、7349f6f9bを追加した。完全SHA/pathとH4 snapshot・H/state・精度条件はINPUTS.json。開始後の新成果を自動的に追跡せず、以後は明示された追加記録ごとに固定入力集合を拡張する。

- Full PF/S2・m12各1shot・531533828 RZは保存のMODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD。数値失敗に読み替えずモデル不適合として残す。m12=2の559457386 RZを最安適合点として選ぶ。
- p7/S2はm12各1shot・376203472 RZをモデル不適合として残し、2shot・395962678 RZを4点の最安適合点として選ぶ。
- Full PFの「2shotが最安」というB2結果の範囲は、m0〜m11固定、m12のX/Y同数・各軸1shot以上。全配分の最適性へ拡張しない。catalogの選択範囲は引き続き保存集合内。
- 初回はPR-Pauli LD3共通VP、PR-DF LD7共通V0をQ3適合・価格待ちとして残した。その後LD3には上記A価格を接続した。LD7は費用nullで順位に入れない。LD3の旧信号のみ状態はhistoryへ残す。
- PR-Pauliの事前固定予算・比較は旧615304502 RZの基準と元SOURCEをhistorical_*へ転記し、新Full PF最安値へ付け替えない。
- Full-Hは同じconfiguration・26軸・固定1.5倍配分のQ3をB6c5569c3から照合したうえで、C6ad23def0のモデル参照総額を表示する。m0実測1pathと残24モデルの資格は保持する。

## 再利用・照合と選択規則

既存 `baseline_signal_pipeline.blob` の固定Git読取と、B/B2/DのSUMMARY・C_RESULT_JOIN・INPUTS・候補/価格対応表を再利用する。既存summarizerの確定済み最安値をそのまま集めるだけでなく、全保存行を共通形式へ写し、今回の入力集合に対して最小値を比較する。新しいoptimizer/decoder/N solver/価格計算器はない。

比較条件はsnapshotだけでなくH/state、basis/geometry/tol policy、h/M、energy prior、元full DF-H solver proxy、epsilon/採択閾値、total decoder/loss、P0、compiler/control/wrapperを固定する。方式固有のideal finite-K law・split・K/κは属性として保持し、方式間で同じlawを強制しない。候補内ではcase/configurationのlawを照合する。

各点で、原configuration・total decoder view・canonical candidate・physical ID・round/axis・shot vectorを照合し、26価格の元sourceへ戻せるようにする。新配分の価格termに記録されたNはQ3配分と一致させる。1shot価格の元receiptに付いた過去のNは価格自体の属性ではないため、物理条件が同じ別配分への価格再利用は拒否しない。今回N×Cを新しく計算せず、保存済み条件付き総額とその配分のsourceを照合する。

選択対象は、同条件・保存Q3適合・全26軸価格対応あり・保存総額ありの点だけ。費用比較はexact有理数で行い、表示の丸め値を使わない。同率最安はすべて返す。Q3待ち/価格待ち/モデル不適合/判定保留/metadata不一致を区別する。metadata不一致を科学的不適合にしない。

HANDOFF→SUMMARYの重複参照は同一documentとして読み、同一configuration・配分・評価契約・価格locatorの点も一つに統合する。alias labelと元sourceは全て残す。保存判定/費用が矛盾する重複は選択せず不一致を表示する。

現在対応するのは今回のPF6、決定論PF m12比較、Pauli配分比較、PR-DF split比較、信号packet、価格のないQ3軸索引、Full-H混合参照総額の公開schema。未対応schemaは明示的に停止し、価格や資格を推測しない。将来のD2等は、同じ形式の保存総額＋同じ配分のQ3結果を追加すれば同じコマンドで比較できる。26軸価格だけからの新規model予測・配分生成・G計算は自動では開始しない。

## 検証・§10.5・性能

関連の旧集計 `summarize_pf6_cost_budget_q3_fixed.py`、Pauli summarize.py、`summarize_pr_df_ld_split_q3_resume.py` とCの保存会計を調査した。方式固有の集計とreceipt読取は存在したが、snapshot/精度をqueryにして全方式を横断し、入力追加だけで更新するconsumerは確認範囲で未実装だった。H2やnon-DF旧catalogは別対象のため転用しない。同等成果/同名C jobは確認範囲で0、他担当private rawやprocessには介入していない。

人工・保存結果の技術検収はTESTS.txt。実行入口の同じcommandを2回使い、INPUTSのsource追加だけでp7最安値が435481090→395962678へ更新されることを確認する。snapshot/精度/law/compiler/axis/N/physical ID不一致、欠測、判定保留、exact比較と重複も検査する。保存値の読取と比較であり、新規科学callはない。

`Reader.document` は固定commit/pathをcacheし、複数HANDOFFから同じ大きなJSONを反復読取しない。既存B2と同じprocess-local整数桁上限の解除で長いexact MSE分子・分母を読む。OSや他環境の設定は変更しない。性能方法の短い索引はdocs/PERFORMANCE_METHODS.md。今回追加のworkerや資源上限は不要。

ロードマップ本体は変更せず、保存集合の最安点と未解決点を今回の成果へまとめる。正式E[C]/G/十分N/全空間最適性・方式winnerは認定しない。commit/push・remote一致確認後mandatory STOP。

## 初回公開結果

実行SOURCE `f889ac16e04f6f7bb2786c18c181c7eb025693de`。固定10入力の34点を収録し、適合27・モデル不適合5・Q3適合/価格待ち2、metadata不一致0。重複4件を統合し、LD3の旧信号のみ状態1件はhistoryへ保全。比較表は [run01/REPORT.md](run01/REPORT.md)、機械可読正本は [run01/CATALOG.json](run01/CATALOG.json)、確認は [run01/CHECKS.json](run01/CHECKS.json)。

保存集合内の最安費用はFull PF=559457386、p7=395962678、PR-Pauli LD2 OWN-B≈2442845584.51、PR-DF LD2 B_PRICE≈46341978336.21 RZ。Full-Hの固定シナリオ≈1246382808920.55 RZも同配分Q3を照合済み。LD3 VPとLD7 V0は価格nullで選択対象外。5 test PASS、新規科学call0。固定commit/pathを追加した同じコマンドで更新可能な状態で公開し、remote照合後mandatory STOP。

長いMSE exact分子・分母はCATALOG内では十進文字列として転記し、標準Python JSON readerでも読める。値を丸めた表現ではない。条件付き費用と比較にはexact有理数を使い、元SUMMARYの保存判定を再評価しない。

## 粗幅Y4復旧Q3の追加（2026-11-16）

p7標準Yoshida4・粗幅32h/5 B_PRICEの復旧Q3（8a2f5c46d）をモデル不適合・費用nullとして追加。旧UNKNOWNは34783cd09の実行履歴へ接続。固定11入力35点（適合27/モデル不適合6/価格待ち2）、従来34点と全方式最安値は不変、p7最安395962678 RZ。新科学call0、技術検収5 PASS、公開後STOP。

新しい入力は `8a2f5c46d201b26bdaf2aad7ca556e31083779aa:docs/research/d_p7_y4_coarse_q3_recovery_20261116/C_RESULT_JOIN.json`。新しい固定配分Q3索引の読取を最小拡張し、従来と同じcommandで集計した。実行SOURCE `e121cd556a5ace0487e22ca1363f562e01da622d`。候補 `B0I_P7_Y4_32h_over_5` の `q3_source`/`saved_recovery_assessment_source` が確定判定、`execution_history` が旧UNKNOWNと元理由、`q3_axis_sources` が26軸のphysical/canonical/配分への索引。数値失敗をモデル不適合へ付け替えたものではなく、復旧runの保存判定を採用した。費用推定・Q3再実施なし。旧a03adda2fの34点・旧UNKNOWN/STOPはGit固定参照で保全。ロードマップ本体は更新しない。
