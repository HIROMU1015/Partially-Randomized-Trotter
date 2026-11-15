# p7／標準Yoshida4／32h/5：信号補完完了・Q3はUNKNOWN STOP

**保存7 roundを再利用し、m6〜m11の6 complex meanを一回ずつ取得、全13 roundの数値・物理入力gateを通過した。Q3 workerは1回起動したが、Accelerator初期化のAST SOURCE guardで停止し、科学的Q3 API呼出しは0件。粗幅MSEは未取得（UNKNOWN）、粗幅総費用も未評価。retryせずmandatory STOPとする。**

保存linear H4 / 1.00 Å / STO-3G / tol-only DF10、B0I p7、一体項＋fragment0..6保存順、標準Yoshida4、PF要求幅32h/5、h_RPE=.1、m0〜m12、P0除外。元configurationは `dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0`。新step/固有系/reference/回路build・compile・価格・fit・別配分は各0。

正本：[RESULT_SUMMARY.json](RESULT_SUMMARY.json)、[SIGNAL_PACKET_INDEX.json](SIGNAL_PACKET_INDEX.json)、[HANDOFF_B](HANDOFF_B.md)。原結果・旧STOPは変更していない。

## 同じshot配分での現幅・粗幅比較

X/Yとも **`[59,55,51,47,43,39,35,31,27,23,19,7,2]`**。現幅Yoshida4のB_PRICE列をそのまま使い、S2の末尾13/4等へ変更していない。decoder/target/loss/law/exact τは元評価契約を継承した。

| PF要求幅 | MSE下界 Ha² | MSE上界 Ha² | 判定・証拠 |
|---|---|---|---|
| 現幅16h/3 | 2.418371981954278179229700e−6 | 2.418371981954278179229706e−6 | 保存 `MODEL_BOUND_MEETS_ADOPTED_THRESHOLD` |
| 粗幅32h/5 | **未取得/null** | **未取得/null** | **UNKNOWN：Q3 API開始前のSOURCE guard停止** |

現幅の数値は `b4296b110b129d226dbef2743c14f725a27c3bf6:docs/research/pf6_cost_budget_q3_results_20261115/SUMMARY.json#rows[15]` の保存表示値の引用。exact正本の元API・原結果は再計算していない。τの元exact入力は同commitのINPUTS.jsonで、表示は `0.000002539561285063813282041692 Ha²`。上界≤元exact τ／下界>τ／閾値跨ぎ／入力・gate不成立という規則を保ち、今回粗幅は後者のUNKNOWNである。2%価格判定やMSEによるFAILと混同しない。

新信号を用いたQ3の数値閉包、質量、残余loss、MSE上下界の評価はまだ実行されていない。現幅のPASS・閉包や少ないqから粗幅の適合を推定しない。今回、粗幅を棄却したわけでも、shot増で救済できると認定したわけでもない。

## 信号6件・保存7件の取得とgate

| 新規m | T | q_PF | physical ID | 元数値gate |
|---|---:|---:|---|---|
| 6 | 6.4 | 10 | `77d24715893869133820` | PASS |
| 7 | 12.8 | 20 | `7e23a1be7c7ffb2e536d` | PASS |
| 8 | 25.6 | 40 | `443fdeda245dc18375fc` | PASS |
| 9 | 51.2 | 80 | `f0a99e91997c44c59ad4` | PASS |
| 10 | 102.4 | 160 | `61d8331798b0671efa95` | PASS |
| 11 | 204.8 | 320 | `b1bd9b1c025f7348bf92` | PASS |

全新規点のactual幅は元hのexact有理数×32/5。保存step `5fb2c32e…/cell_p7_4th_32o5/OPERATORS.npz#PF_step_without_global_scalars` を既存AのSavedStepContext/load_stepsへ渡し、元baseline_signal_numerics.evaluateのmatrix power・norm/unitarity・finite/outcome-range gateを使った。新しいPF step・expm・固有値・referenceは算出していない。

Hamiltonian constant `2.29310124732`、保存tail identity `4.5302064712666724e−05` を読み、元kernelの順序どおり総Tへのscalar phaseを一度ずつ適用した。符号付きY4順序・fragment順・原H/state/snapshotを維持した。raw=corrected、B=1/log_B=0、各scalar_onceとgates_passedを確認した。

m0〜m4は現幅と同一の元A round record（`06a15ee45…/B0I_P7_Y4_16h_over_3/round_0..4.json`）から再利用。元domainの完全一致と保存packetのraw/corrected/reference/normalizationを照合した。m5/m12は元T2の粗幅固有direct meanを既存stage_recordでSOURCE付き再利用した。保存値へscalarを再適用せず、新しい独立標本や新取得として数えていない。

入力状態や保存stepを一度読み、36次元の軽い6 matrix powerは元C.executeの逐次保存/STOP経路を使った。各meanは約0.0008〜0.0012 s、signal全体wall **3.726370 s**、peak RSS **505,316 KiB**。入力load/初回importが大部分で、worker起動・state複製を増やさなかった。同条件速度倍率は未測定。

## 単一SOURCE intakeへの技術接続

最初に保全したraw packetは、元SOURCEを残した保存recordと新recordのSOURCEが混在していた。既存Bのsaved-baseline single-source guardがmetadata intakeで拒否したため、科学実行を追加せず、C.b_meanによる**共通binding-only view**を作成した。

各bound meanは元record locator・元numerical SOURCE・元input SOURCEをprovider_binding.input_sourceに保持する。raw/corrected/reference/normalization/bias/uncertaintyを全13行で変更していない。原packetと原recordは別の固定commitで保全した。binding viewは同じ1候補・13round packetの消費側表現で、2個目の科学packet取得ではない。既存の単一SOURCE guard・physical binding guardは緩和せず、このviewはQ3 metadata intakeを通過した。

## Q3停止の実値と原因

Q3実行SOURCE `278fb5214f1091efba2a05b305bc05190cd18ff6`。元worker/原API保存処理と8-worker並列DPを設定したが、実際には外側worker1件のみ起動し、DP worker生成や科学APIより前に停止した。

停止例外：`ValueError: Reference total DP SOURCE changed; do not silently adapt`。

| AST識別 | SHA-256 |
|---|---|
| 既存guard期待値 | `3d043e716246e57a8b05784b0d5c0f6777aaf43410fed8c4dd8b68df6dd590b8` |
| Python3.12.3の実AST dump | `eb3150a5b99a661352e16c2c594a4c2edb9380d4e36ee56d303c005c854c9679` |
| 空のtype_params fieldだけを除いた診断コピー | 既存guard期待値と一致 |

同じPython3.12で、再利用DP関数のASTは元kernel `710aa69b663ef3313439762d25c80a652a03948f` の関数ASTと一致する。Python3.12で追加された空のFunctionDef.type_paramsがdumpへ入ることがguard不一致の原因と特定できた。過去B実行のPython版をこの資料だけから断定したり、過去のQ3結果を無効としたりはしない。

事前testはSOURCE照合・moduleロード・metadata intakeまでで、AcceleratorコンストラクタのAST照合まで通していなかった。この検査不足を、静的なschema差の回帰検査と次回必須のconstructor事前確認へ反映した。診断はコピーしたASTに限り、runtimeのast.dump・SOURCE guard・数値kernelを変更していない。診断一致だけで互換修正の科学的無害性を自動認定しない。

`WORKER_PERSISTENCE_ERROR.json` の **scientific_evaluation_calls=0 / api_return_saved=false** と、API/partialsの不存在を確認した。外側process/子processのOS終了codeは0だが、監督結果はerrorsあり・receiptなしのUNKNOWN。OS正常終了を科学PASSにしていない。Q3監督wall0.026638 s、Q3 peak RSSは未記録/null。retry・別runtimeでの再投入は0。

原停止記録・診断は `a4da9fb1fc7c6de65b81d81229ae42d7cf59104c` の [q3_run01](q3_run01)／[Q3_STOP_DIAGNOSTIC](Q3_STOP_DIAGNOSTIC.json)。

## 費用基準と未評価領域

粗幅のcompiled単価・全費用は **null/未評価**。今回compile/価格計算はしていない。同公式現幅の **435,641,568 RZ** と、**計画時点**のp7最安適合点 **435,481,090 RZ** は、将来粗幅が適合した場合の比較基準として別々に保持する。

差分照合時にはBの後続現幅S2成果も見つかった。`93ab9dbbe610d66eba39ea9299b070a13a867a3e:docs/research/p7_s2_m12_two_results_20261116/HANDOFF.md` は別配分で395,962,678 RZの保存適合点を報告している。これは粗幅Y4の同等信号/Q3ではなく、指定された二つの基準や配分を置換していない。今後の最良点challengeではこの後続点も別に扱う必要がある。

hの5513q+307や現幅の価格係数を粗幅へ適用していない。追加価格pilot・shot増・別配分・N/G・winnerは未実行。

## 固定SOURCE・成果物・STOP

- 採択計画：`d01c780d035eb096186ef1b8aa6d5e580bca2a50`。
- 信号実行SOURCE：`3179680ed9b4cff251b3401cabea2f6100cf47fb`、SIGNAL_INPUTS/caller。原signal公開 `fff936fa0cddc1ff13cb1edcf38fdde1b4a1a8b2`。
- binding-only SOURCE：`51aba420b24026830eda9e9e4bc9a1a68f3ffe76`。bound packet公開 `f23c3f10ed56d0c218e49efbdbfb31d12ffa8a53`。
- Q3実行SOURCE：`278fb5214f1091efba2a05b305bc05190cd18ff6`、Q3_INPUTS/caller。停止結果公開 `a4da9fb1fc7c6de65b81d81229ae42d7cf59104c`。
- SOURCE・元13physical ID・保存record・scalar規則・raw値・数値gateの対応はSIGNAL_PACKET_INDEXとBINDING_PROVENANCEへ保存。

§10.5：既存coverageを引き継ぎ、関連D/B/A/T2の公表namespaceと同ユーザーの関連processを差分確認。対象6点・同configuration完成packet・同配分Q3の後続同等結果/進行jobは確認範囲で0。確認範囲の外まで不存在を主張しない。詳細[PREFLIGHT](PREFLIGHT.json)。

人工/metadata4 test PASS＋停止後の静的AST診断1 test PASS。signalの実数値gateは新6/保存7でPASS、Q3結果はUNKNOWNという区別を維持する。原grant/ledger/marker/KEEP・旧STOP、他担当branch/worktreeは保全。ロードマップ本体は変更せず、HANDOFF_Bへ「信号完成、Q3未取得」の差分を渡す。

今回のretry=0条件に従い、科学作業は停止済み。復旧案は[NEXT_STEPS](NEXT_STEPS.md)に記載。結果をcommit/push・remote SHA/tree/成果物照合した後 **mandatory STOP**。未使用のQ3上限を自動転用しない。
