# B2 PR-DF configurationとDF信号providerの共通入口

保存LD1・LD2・LD7・LD9の全52 roundを再利用と判定し、既存bundleを完全一致で再構成できた。LD7・LD9の新しい公開bundleは、元のPR-DF `accept_saved` で各13 meanを受け入れ、`statistics_executed=False` を確認した。人工構成では物理条件が変わるroundだけが取得要求になり、κ宣言のみの変更は全roundを再利用する。入口17検査・既存価格入口17検査がPASS。新規の実configuration・実信号・sampling・build・compile・fit・Q3は0件。

現在の入口SOURCEは `2a5e085e56d917607debcc18cec4faf7fa5a3539`。保存bundle・再現表の公開commitは `0bcd1df98c7896fb91896cc23e8d8c6105611408`。初回の組立SOURCE `dadc7317e` と、後続のround別precision判定を区別して保全した。数値providerのSOURCEは従来の `fefd548ecd56f60d72031b3d385f6f5e6ee95d06` のまま。

## 接続した既存機能

- `pr_rpe_signal_plan.plan`：Bのcanonical validatorと元mapperのphysical rowを使う、whole-scheduleの受動planner。
- `pr_rpe_signal_provider`：既存の `load_h4_context`／`saved_h4_reference`／`evaluate`。本入口が実行するのは保存referenceの読取だけで、H4 contextの復元やevaluateは呼ばない。
- LD7/LD9 callerのbundle変換：同じraw・corrected・exact-tail・reference、normalization、allowance、bias、状態根拠をそのまま射影する。旧callerのLD7出力名差し替え・global変更を共通入口へ持ち込まない。
- `rpe_signal_statistics_caller.accept_saved`／`bind_bundle`：元の `A_BUNDLE_REQUIRES_EXISTING_K_ACCEPT_SAVED` 受入を無変更で使う。Q3 workerやAST guardは起動しない。
- 価格は既存の `pr_price_rules.PriceRules` とB2の公開済みMODEL接続を使う。価格規則の再実装・再fitはない。

追加したのは `src/trottertracks/pr_df_signal_entry.py` と受動CLI `scripts/prepare_pr_df_signals.py`。B2のconfiguration本体、Git locator、または生成結果のconfiguration欄を受け取る。候補・scheduleの生成や探索は行わない。固定providerは別package namespaceで読み込み、B2側の新しい `trottertracks` を上書きしない。

## round別の判定

| decision | 意味 |
|---|---|
| REUSE | 保存recordとH・状態・分割・tail・scalar・law・実scheduleが一致。元SOURCEと数値欄を保つ |
| REQUEST | 保存recordがなく、既存providerへの具体的なjobと保存Full-H referenceを提示できる。数値gateは未評価 |
| UNSUPPORTED | 現SOURCEでは扱えない条件、SOURCE不一致、保存reference不足、またはprecision条件不成立。理由を返す |

照合には、H/state/snapshot・basis/rank、元fragment順、one-bodyを別headとするprefix、prefix固有のλ_Rとtail identity s_R、そのSOURCE、Hamiltonian定数cの独立したSOURCE、scalar一回適用、law/event recipe、T/q/実step/R/K、符号付きpieceとevent配分を含める。Rやevent総数だけでは照合しない。LD7とLD9はともにR=1でも別keyとなる。κやconfiguration IDは、これらの実条件が同じなら照合キーに含めない。異なるSOURCEを自動的に同等認定することはしない。

保存tail descriptorのJSONとhash、Hamiltonian定数の保存記録を読むだけで、保存H/P0演算子の数値再検査は行わない。cとs_Rを取り違えず、元providerの `exp(-i T (c+s_R))` 一回適用を維持する。保存recordの数値gate・scalar flag・identity・law・contextを照合するが、gate自体を再実行したことにはしない。

## 対応範囲とprecision

元providerの実装対応は `2nd`（S2）と `4th(m5_best)`、K=2/4/6、非空のdeterministic prefixとDF tail。分割ごとの保存descriptor、固定H4 linear 1.00 Å／STO-3G／tol-only DF10、保存Full-H referenceが必要。今回の保存再現はS2/K2のLD1/2/7/9。他の公式・Kの実信号性能を確認したとは扱わない。m5_bestのpiece・配分は元structure関数を使う。人工m5例のmetadata経路は確認したが、H4のm5信号や近似誤差を計算していない。

未対応には、Pauli tail、PF endpoints、未知公式/K、H/state/snapshot・fragment順・head/tail政策・λ/scalar/law SOURCEの不一致、保存referenceのないT、canonical/physical rowの不整合を含む。b>1、closure、unitarity、raw範囲等の数値gateは、REQUEST後に元providerが初めて検査する。REQUESTはgate PASSではない。

IDEで共有されたAの `08812cd30`／SOURCE `4e5b8bef6` も静的に確認した。DF providerもcanonical有理数と物理候補のbinary64 T/stepを厳密比較する。表現できない場合、旧whole-schedule plannerは最初の不一致で止まるため、本入口ではB validatorを保った同一job射影を用い、該当roundだけを `ACTUAL_STEP_OR_T_NOT_BINARY64_EXACT` にする。正常jobが元plannerと完全一致することを検査した。人工T=1/8、q=3、step=1/24では1 round未対応・残り12 round再利用となる。qが2の冪かどうかという推測では判定しない。

一方、Aで修正されたPauli入口の「1 roundに1つ」というnormalizer上限は、今回の固定DF経路にはない。DFの `_macros` はpieceごとに元 `finite_rte_distribution` を使う。本作業でPauliのkernel・入口・検証結果を変更したり、その人工系の数値検証をDFの検証へ移したりしていない。

## 保存・取得後の組立

`stage()` は保存recordまたは外部で取得済みのSOURCEを読むだけで、REQUESTを実行しない。同一物理条件でIDが変わったrecordは、数値・allowance・状態/reference根拠・数値SOURCEを保持し、B identity/contextだけを新しいcanonical configurationへ結び直す。元recordへのreuse provenanceを残す。

元DF intakeは全meanのraw carrier commitが一つであることを要求する。このguardを緩めず、混合・reboundの場合は `CONFIGURATION.json` と13個のtop-level raw carrierを専用出力へ保存し、commitで固定した後に `assemble()`／CLI bundleへ渡す。欠測・未対応・gate失敗・identity不一致では部分結果と理由を保存し、完全bundleを作らない。自動retry、再sampling、別roundの補完はない。旧rawを上書きしない。

## 証拠・重複照合・ロードマップ

[SAVED_REPLAY.json](SAVED_REPLAY.json) は4構成52 round、[ARTIFICIAL_CHECKS.json](ARTIFICIAL_CHECKS.json) はκ宣言のみ・Δ変更・R変更・LD7/9差の人工判定。保存bundleは初回公開commitで固定した。元intakeによる公開blobの読取結果は [PUBLISHED_INTAKE_CHECK.json](PUBLISHED_INTAKE_CHECK.json)。初回15検査の記録を保全し、precision追加後の17検査と価格17検査を [TEST_RESULTS_FINAL.txt](TEST_RESULTS_FINAL.txt) に分けた。いずれもD2自身の技術検査で、独立レビューやCによる科学的資格付与ではない。

§10.5の差分照合では、B2の生成入口 `2fa2dbfac`／受入 `4cb648c3b`／価格接続 `47f86eb20`、Pauli共通入口 `93a5ef3b9`／更新 `08812cd30`、DFの元planner/provider、LD7/LD9の保存caller・bundle・Q3 intake、関連Git履歴と対象job名を確認した。Pauli入口は辞書・native job・packet形式が異なるため流用せず、DFの既存機能を接続した。同等のDF共通入口・同一取得jobは確認範囲で未同定。B2やAのworktree・job・出力は変更していない。

ロードマップ上は、B2が生成済みDF候補の不足roundを機械的に取り出し、後続の取得後に既存intakeへ渡せるようになった。新規取得、Q3、価格不足の較正、モデル資格の判断は進めていない。価格入口・固定SOURCE・原結果・旧STOPを保全し、公開確認後に終了する。
