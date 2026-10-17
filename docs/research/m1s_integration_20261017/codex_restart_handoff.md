# M1-S：Codex再開用ハンドオフ

2026-10-17 JST。**記録統合完了後mandatory STOP。新規科学実行の承認なし。**

これは次期研究の実施指示・採択契約ではない。将来のCodexが、既存成果を壊したり同じ作業を重複実行したりせず、
ユーザーが指定した範囲で再開するための記録である。
[統合サマリー](integrated_research_summary.md)、[固定証拠索引](evidence_and_results_index.md)、
[最新GPT原本](../m1s_integrated_reassessment_claude_prior_reviews_stage_s_20261017.md)を併読する。

## 1. 最初に読むこと・確認すること

1. ユーザーの新しい指示と対象repository/branchを確認する。IDEのActive file/Open tabsは実施指示ではない。
2. `PROJECT_MAP.md`、`docs/research/研究概要・現状.md`、本サマリー・証拠索引を読む。
3. [共通運用指示2026-10-17](../research_project_instructions_unified_2026-10-17.md)を適用する。
   旧2026-10-16版を現行正本としない。旧実験契約の固定条件は履歴として保全する。
4. `git status`、remotes、`git worktree list`、関連branch HEADをread-onlyで確認する。
   親のdirty編集を持ち込まず、他Codexのworktree・jobへ書き込まない。
5. commit/push先は`HIROMU1015/*`のみ。quration系へcommit/pushしない。
6. 新作業の範囲を、記録整理、理論研究、契約準備、詳細レビュー、科学実行に分け、実際に受領した承認へ結び付ける。

今回の独立branch：`track-b-m1s-integrated-handoff-20261017`。
今回のworktree：`/tmp/prt-m1s-integrated-handoff-20261017`。
Stage S基点：`f2bfb8770e6858de56a63e50e68f07ff2d6464b6`。
先行研究調査：`4301c0f87a16be45b620304c3fa935d5aa2d721a`。

**二基点は線形な親子ではない。** 先行研究調査branchへ切り替えるとStage Sやshot-cost後続資料が存在しない。
調査を取り込むためにそのtreeで現在のtreeを置換したり、保護対象を巻き戻すmergeを行ったりしない。
本資料では固定commitリンクとhash索引で連結し、調査branchをmergeしていない。

## 2. 研究の現在地を短縮せず引き継ぐ

| 項目 | 現在の証拠 | 読み替えてはいけないこと |
|---|---|---|
| T1a/T1b | 固定model/r・finite damping・小stepの適用域付き数学的受理 | 一般分子規模・任意r運用・資源保証 |
| 共鳴 | 非縮退対象でのJensen参照floor受理、単一指数には(SP)等が必要 | 共鳴なしの認定、全保存oracleへの適用 |
| Gate 0 v4 | TECHNICAL_STOP、W/v/ρ null、one-shot消費 | 科学的NO-GO、未消費としての再利用 |
| Gate 0 v5 | 全3分割W/v/ρ取得、17精度条件を通過 | 有限step/RPE/総費用GO |
| 独立Gate 0 | 照合可能範囲PASS、全scope UNDETERMINED | 未保存full-H/HDとの直接一致 |
| 有限step | 108/108行、G3/G6各16visibility higher、G9絶対未解像 | phase biasも改善、一般的効果 |
| γ V1 | G3/G6各16内部tradeoff境界、G9未解像保持 | 絶対/最適shots、多round保証 |
| π/3 gate | G3 16・G6 T4 8成立、G6 T1 8不成立、G9 16未照合 | RPE成功・実用精度・科学GO |
| Stage S | G3 T4 8局所整数witness、他主対象16はcap直下同数 | 全γ最適/不存在、native総費用 |
| 最新レビュー | 新primitiveの根拠弱い、全定理priority未決、成果保全優先の推奨 | 正式終了/継続/Stage C採択 |

全分子数値のscopeはlinear H4 1.00 Å、STO-3G、8 spin orbitals、保存DF rank12、G3/G6/G9、
保存数値sector oracle。有限step以降はδ=1/8,1/16 Ha^-1、R=2,4、T=1,4 Ha^-1、非zero24cell・72行・48対照、zero36を含む108行。
元norm²、非冗長signed Pauli-tail、identityのHD移動、固定保存入力を維持する。

## 3. 再開時に必ず保持する解釈修正

- HD=0の完全乱択部分は方向制御qDRIFT。HD≠0の全体は決定論／qDRIFT混成対称積。
- 負枝は同順sign reversal、全wordのadjointではない。`A X A`を`A X A†`にしない。
- `w=W−v=E Var_k(P_j−h)`。生の`E Var_k(P_j)`からは`Var_k(h)`を差し引く。全H固有state≠h固有state。
- (SP)のscalar等式と、非縮退だけでの共鳴Jensen floorを分ける。sector内gapだけでは全space条件を認定しない。
- `ρ=(W+v)/(2W)`は指定先導rate比。shots/native費用/全資源の因子ではない。
- raw-H総biasには元のモデル差が含まれる。η=|T|Bを再加算しない。新実装誤差は別問題。
- 同じ十分shot設計と最小必要shotsを区別する。PER_METHOD_ROW 95%を全行同時coverageとしない。
- cap直下はfloor+1。cap同数は全γで削減不能の証明ではない。cap連続supremumは全domain整数比最大値ではない。
- **整数shot削減は総費用改善の必要条件でも十分条件でもない。** 同数でも単価が下がる経路を除外しない。
- 共通per-shot状態準備費と共通固定費は別。前者はshotsが異なると相殺しない。
- G6の小さなresolved差を消さず、G9の絶対未解像を相対率で救済しない。
- finite→γ→π/3→Stage Sは同じ保存信号の解析系列。独立実験4回と数えない。

出典は最新GPT §§3–11、統合サマリー§§2–8。このハンドオフで新しい数学的結果を追加していない。

## 4. 未解決事項と再開候補

以下は既存レビュー・packetに挙げられた候補の整理で、優先順位・研究契約・数値条件の新規採択ではない。
必要な証拠は選ぶclaimに依存する。内部記録のために新しい分子計算や全RPE実装を必須にしない。

| ID | 候補課題 | 何が未解決か／必要な証拠 | 担当・承認境界 |
|---|---|---|---|
| R0 | 既存記録の利用・追記 | 取得済み原本、固定links、時点差の維持 | ユーザーの文書指示によりCodex。科学実行は含まない |
| R1 | 定理statementのprior art対応 | MFの倍化枠組みへの包含と、有限減衰C-sec全statementの直接系を区別。Davies全文・全仮定/量化/定数の対応が不足 | 調査/理論の範囲をユーザーが指定。重要な科学受理の新レビューは別途開始承認 |
| R2 | 有限δ・r変更の理論 | 固定r・δ→0の定理と、固定δ・多数gate/有限HD角の運用との差。T1c未実施 | 新しい理論研究の明示指示が必要。既存T1bの黙示拡張は不可 |
| R3 | 一般DF event／HD内部PF | involution以外の正規化、対称性、precision、control、内部誤差 | 科学scopeの採択と、実装準備指示が必要。今は準備しない |
| R4 | 共鳴・state拡張 | 非冗長Pauliの多rate例、縮退/一般guiding state、state-specific定数、reachable sectorの証拠 | 理論候補。存在や実用性を推測しない |
| R5 | Task A論理総費用の限定比較 | 共通task/精度/confidence/利用情報、γ policy、実行可能schedule、全wrapper単価、HD/準備/Fのscope、強い対照をclaimに応じ選ぶ | ユーザーの継続方針・新scope採択→Codexの必要準備→契約採択→別科学実施承認 |
| R6 | Task B多round energy RPE | target overlap/energy prior/alias、raw位相→E*T、全dyadic round、成功条件、残余位相予算、費用集計 | Task Aのwitnessでは承認されない。新しい設計・採択・実行承認が必要 |
| R7 | 別条件での一般化 | 別分子/geometry/split/law/時間/精度に必要な情報と対照、探索とheld-outの分離 | 一般性を主張する場合の候補。無条件の必須追加実験にはしない |

**Stage CのSOURCE実装、native費用検証、新規契約準備は今回行っていない。**
R5/R6は次の一つの判断単位にtaskと費用をまとめるという最新GPTの推奨を記録しただけ。
細切れgateをさらに追加すること、陽性cellだけを独立検証集合と呼ぶことを既定の次作業にしない。

## 5. Stage C／Task Bの不足情報を正確に扱う

| 要素 | 取得済み | 未取得・未認証 |
|---|---|---|
| state | 保存vector、norm、basis、solver絶対残差品質 | target overlap、全space ground/非縮退、状態準備回路・品質・支払い頻度・費用 |
| HD | 保存DF prefix、dense計算対象、理論のexact half | native分解、内部PF/synthesis誤差、control位相、価格 |
| tail | 保存Pauli law、係数/support/符号の独立照合 | 実装native whole-wrapper単価、相対位相、ancilla、routing |
| signal | 保存理想coherent mean、条件付き丸め半径、モデル差台帳 | 選ぶ実装に対する誤差・物理平均へのbridge |
| 統計 | fresh-IID/Hoeffding/X/Yの十分設計 | 実運用でのIID、全round confidence ledger、最適性 |
| decoder | π/3限定比較 | 全round時間列、符号・prior・initial branch、energy精度/RMSE/失敗率の区別 |
| 強い対照 | ordinary/independentの内部比較 | 決定論PF、PR-RTE、同情報の再最適化等、主張に適したbaseline |

未知を0と置かない。論理資源比較のscopeなら、実機SPAMを測定済みと主張せず明示的な理想仮定を置く選択はあり得る。
実機保証を対象外なのに必須に追加して研究範囲を膨らませることもしない。具体的なscopeは未採択。

## 6. 採択済み／未採択／消費済みの区別

過去に採択・実行されたもの：S0理論scope、Gate 0 v4/v5、独立照合、有限step、γ V1、π/3 gate、Stage Sの各個別契約。
対応する実施authorizationはそれぞれの過去one-shotの根拠で、現在の新実行許可ではない。

現在未採択のもの：最新Claude/GPTの研究縮小・終了・他track転換の推奨、資源路線の次scope、Stage C、Task B、
新しい実用γ・metric・baseline・削減閾値・精度policy、論文化。今回の成果記録指示はこれらを自動採択しない。
既存technical note/addendumの`DRAFT`も黙って正式論文・採択定理集へ変更しない。

finite-stepとγ V1の旧失敗は保存されている。後続の「修正して完了まで」というユーザー指示の範囲で
新S/C/A・新出力へ移った履歴であり、今回にも持続するretry権限として転用しない。
Stage Sは初回正常完了、消費marker1・grant0・retry0（既存報告）。
全旧markerは読み取り対象で、削除・未消費化・再利用しない。

## 7. 証拠を回収する最短経路

- 理論本文は[T1a](../../tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md)、
  [T1b](../../tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md)、
  [Checkpoint 2](../track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md)。
- 最新結果は[Stage S報告](../../tracks/algorithm_codesign/m1s_stage_s_one_shot_results_and_stop_20261017.md)。
- 最新判断は[Claude原本](../m1s_prior_art_novelty_and_research_value_claude_review_20261017.md)、
  [GPT原本](../m1s_integrated_reassessment_claude_prior_reviews_stage_s_20261017.md)。
- SOURCE/契約/raw/監査/markerは[索引](evidence_and_results_index.md)と[evidence inventory](evidence_inventory.json)。
  pathと固定commitの両方で同定し、branch名だけで最新とみなさない。
- [provided_originals.json](provided_originals.json)に今回受領した原本のbytes/SHA256。
  旧Claude inner原本は未回収。旧GPTでの引用を原本確認済みと表記しない。
- hashだけを確認する場合、NPZ配列decode、Pauli生成、科学module import、科学runner/testの起動をしない。
  metadata監査と科学再解析の違いを守る。

## 8. 実行へ進む場合に初めて必要となる手続

この節は将来の手順であり、今回の許可ではない。

1. **ユーザー**が再開対象と主張scopeを選び、必要な科学方針を採択する。
2. **Codex**がその指示範囲でのみ既存資料を回収し、不足技術を準備する。新契約/SOURCEが必要なら別namespace。
3. **ユーザー**が具体的な契約を採択する。重要な新科学判断のレビューが必要なら開始承認を別途出す。
4. **ユーザー**が固定対象の科学計算/保存再解析/native compile等を別途明示承認する。
5. **Codex**が新S/C/A identityと入力・未使用output・科学call範囲を照合し、許可された一回を実行する。
   failureを保全し、契約に反するretry/閾値緩和をしない。完了・失敗後に公開監査してSTOPする。

同じレビューをもう一度開催することを、単なる記録修正の必須手続にしない。
commit/push・公開bytes一致は科学的妥当性・CI再現・platform認定ではない。

## 9. 計算資源・環境とSTOP

最新版共通指示§8.4とユーザーAGENTSを適用する。将来の新規作業へCPU1/2GiB/CPU160・wall320等の旧数字を一律流用しない。
重い処理では実際の空きCPU/RAM/GPU・dominant workloadを確認し、並列化/バッチ/cacheを合理的に用いる。
科学条件、演算精度・乱数・provenanceを変える高速化は独断で採用しない。性能優位を実測なしに断定しない。

共有設定、OS、共通venv/cache、他ユーザーのprocess/files/quota等を変更・停止しない。
旧契約の既採択capsは保全する。物理quotaや非干渉を無視する無制限占有も許可ではない。
今回行うのは文書・Git・hashの操作だけであり、科学性能benchmarkは0。

承認外scope、新しい科学的選択、入力/SOURCE不一致、未取得証拠を埋めるための新計算、
消費済marker再利用が必要になった時は自動実施しない。科学的negative/unresolvedを技術修正でpositiveへ変えない。

**この記録作成の完了後、Codexはmandatory STOP。次の担当はユーザーで、必要なら具体的な再開作業を指示する。**
研究終了・継続・論文化を今回採択したとは扱わず、新計算・新レビュー・Stage Cへ移行しない。
