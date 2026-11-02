# H2/H4/H6 費用逆転：同一q対照・H2精度補完の一括実行準備

**A・Bを一つの固定契約に封印し、metadata-only preflightと人工検査を完了した。実数値解析は未実行、実行grantは未発行。ユーザーの一回の明示承認待ち。** Cは結果を用いる固定統合手順、Dは未承認の独立確認準備として分離した。旧strict COMPAREはBLOCKED、旧STOPを維持する。

## 承認対象と識別子

| 項目 | 固定値 |
|---|---|
| SOURCE commit | `1c5bb8a65a7aa7c90707b783eb5d6cf97f4d0bfe` |
| 契約SHA256 | `bae3aa50a3bda959c94c0149d3abb7f8d0cbbdc491cf0800e62d155622dfd2e5` |
| 入力manifest SHA256 | `a8b55834c76ab24ea4efec81da5c957059c5ef81dd0dba43ff1bf20683770d83` |
| 基準commit | `364970df33859e108d90b6ceaacde467db969fb8` |
| A | H4/H6×Full/PR×q4/q8のS2/標準Yoshida4、ε=.001：8比較・16候補endpoint |
| B | 固定H2 4候補×ε=.05/.01/.005/.001：16新endpoint・8比較 |
| C | 元22比較を再計算せず引用し、Bの8比較を追加。主比較15系精度セルと同一q4セルを別表へ |
| D | H4/H6標準Y4/q4独立確認の静的準備のみ。新標本数は未決定 |

linear H2/H4/H6、隣接1 Å、STO-3G、tol-only DF1e-8、rank3/10/19、T=.8、α_axis=.025、P=0、USUAL_SIGNED。Full prefix3/10/19、PR prefix1/3/5、K4・ABS_TIME_PER_OUTER32、symmetric-directional、状態準備を除くfull Hadamard cosine/sine軸のcompiled RZ。H2主比較はFull q4/q2・PR q4/q2、H4/H6主比較はq8/q4で固定。同一q対照を主比較へ置き換えない。[全候補・契約](execution_contract_DRAFT.json)。

Aは前回の適格8比較だけを用い、保存N/G・paired単価・保存SEを保持する。Bは受入済みH2 DIAGNOSTICSのsignal/bias/reference/empirical allowance/log_B_marginと固定単価だけを使う。H2 ε=.0005を再会計せず、別allowance/referenceの共通PF表を移植しない。不適格/欠測はnullを保持して比較未成立とし、0補完・候補差替え・cohort混合・再ランキングをしない。

## 既存コードとexact call graph

元科学SOURCE `2e59a55beec8159c7832ff4ff02513b8976c0eb9`の`h246_order_cost_analysis_backend.decompose`をそのまま使用する。旧`Backend.account`の数値命令・globals・他定数を保持し、ε定数と出力suffixだけを別functionへ置換した。元SOURCEは無変更。`u_aware_shots`/`axis_headroom`はprovider `c31ef424175c02379eafebe8fc6c7548c1d9aabe`の固定4ファイルを専用領域へ回収した。科学backendのconstructor/setup/prepare/nativeを呼ばない。

| subscope | 外側call | 内部call | 合計 |
|---|---|---|---:|
| 共通 | bootstrap1 / bundle_load1 / integration_summary1 | なし | 3 |
| A | decomposition8 | exact軸別C/N/G分解。shot callなし | 8 |
| B | account16 / decomposition8 | u_aware_shots16 / axis_headroom32 | 72 |
| 一括 | 上記のみ | 二重計上なし | **83 event** |

内部headroomは各endpointでreal/imag各1回、shotは各endpointで1回。人工実行でも同じ83の呼出数が一致した。Cのintegration_summary1は既計算のdirection/比/差を対応付ける処理で、元22比較を再分解しない。新sampling/build/lowering/compile/transpile/simulation/signal/reference/Hamiltonian/allowance再計算・標本統計再推定・global rankingは全0。

Bの新εではNが変わるため、旧ε=.0005のGのSEを新Gへ転記しない。新GのSEはnull、保存単価の分散/SEとnを引用する。既存H2主精度・H4/H6には保存G SEを引用する。新しいG SE/差SE/CI/検定/母平均保証は今回含めない。[結果schemaと統合手順](result_schema_and_integration.json)。

## 検査と受入・STOP

人工12件がPython3.12および実行予定の封印Python3.11でPASS。実行runtimeの閉じたimport集合でも、人工fixtureの83イベントがPASSした。[人工検査](technical_checks.json)、[封印runtime人工検査](runtime_synthetic_smoke.json)、[metadata-only preflight](metadata_preflight.json)。H2/H4/H6/rank/公式/m5/q/prefix/cohort/政策/ε/軸pair混入、欠測0補完、旧endpoint再会計、未承認実行を拒否した。一次担当の技術確認であり、別担当の科学レビューではない。

受入条件は固定入力・政策で評価し、適格性を正しく保存すること。A/Bの適格な比較では元の軸別exact積・加法cross項の恒等式が成立し、保存値/新Gの演算順と8 ulp以内で整合すること。期待方向の再現は受入条件にしない。83 eventの台帳・SOURCE/input・結果SHA・marker・exit/reap・保全が整合して完了とする。専用未使用root/新grantを承認後に一回だけ使用し、エラー・予算超過・実入力/SOURCE破損では部分STOP、retryなし。別worktreeのHEAD・資料更新や一時的な競合だけで停止しない。自動追加解析・採択・詳細レビュー・freeze・旧strict解除なし。

SOURCE/input/runtimeの封印は1,007 file。既知1,792 fileを保全対象とし、既存科学SOURCE・元input/result/grant/ledger/marker/STOP・既知論文主表/Figure2を保持する。[出典と再利用一覧](SOURCE_and_inputs.json)、[保全監査](preservation_audit.json)。CPU/RAM/workerの低い人工上限を設けず、軽量な数値処理は固定台帳の逐次依存に従い、既知file SHA照合は空きaffinityに応じたthreadで処理した。共有環境/他ユーザーprocess/共有venvの変更0。

## C・Dの手順と次の担当

Cでは主比較の3系×5登録精度を固定候補のまま表にし、Full/PRのdirection・G比・適格性・n/保存SE・探索/確認の範囲を示す。同一q結果は別表にする。C/N/Gのexact分解、保存bias/log_B/uとBのheadroom/Nを対応付け、元代表RTE6軌道の範囲を越えるoperation寄与を主張しない。H4 ε=.005の弱い探索順位、未観測tail、事後候補固定、empirical allowance、H6旧parent不明を保持する。補間・交差ε推定・候補再選択・独立再現/因果関係への格上げをしない。

[D準備結果](independent_confirmation_preparation.json)：独立S2はH4 q8/prefix3 n25、H6 q8/prefix5 n23の保存paired単価がある。Hamiltonian/state/DF/基底・law・preparation・compilerが対応し、native数値SOURCE5ファイルが元探索取得SOURCEと一致する。H4の再構築NPZ容器SHA差は科学配列一致記録と区別した。静的準備上は既存単価の再利用候補として成立し、独立S2の再samplingを既定にしない。ただしε=.001へのG/SE再会計は今回の予算外であり、独立確認実行を選ぶ場合の別契約が必要。

標準Y4/q4（H4 `M5_MIN_ab57ad5bd40dc583`、H6 `M5_MIN_37ff3428bb58f5cd`）の独立標本は前回の固定調査範囲で未取得。m5_bestは代用しない。必要なら新取得の最小候補はこの2群、軸数は`2×(n_H4+n_H6)`。nは未決定であり、求める主張/MC精度・family方針・保存分散の限界・希少tail・新cohort/seed除外範囲・非適応STOPを固定して決める。元c028/8825のsampling/native/compile pipelineを再利用候補とする。A/B/Cの結果を見て追加確認の科学的必要性を判断し、今回その実行を提案・起動しない。

共通運用指示はプロジェクトに提供された2026-10-23正式版の実体を使用（SHA `050e5dc7055a1c114c35eb17913f3ee050092b57444dcffb70949e896548ac28`）。§8.1/8.4/10/12を適用し、存在しない節を読了としていない。private原本/seed/grant/内部絶対pathは公開しない。

**次はユーザーがこの固定A・Bパッケージの数値実行を一回承認する。Codexは承認後、A・Bの一回実行→C統合→結果に基づくD準備完成→終了監査を一括実施しmandatory STOPする。** 現在は承認待ちで新実数値call0・有効grant0。[STOP](STOP.json)。
