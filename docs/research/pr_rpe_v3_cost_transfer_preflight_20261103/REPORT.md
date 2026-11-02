# RPE V3：長q compiled-cost移送・不確かさの技術的preflight

2026-11-03 JST / LOCAL / 主エージェントのみ。観測開始2026-11-02 20:28 UTC（11-03 05:28 JST）。

**最終18roundの価格はq1/2 affine予測である。条件が一致する直接compiled価格はPartialのq1,2,8とFullのq1,2,16,32に限られ、q>32の全roundは未検証の長q外挿を含む。A0はr32の回路族を支持するが、最終scheduleのr1/q16,32の直接価格ではなく、最終選択roundとA0 holdoutの完全一致はない。保存点差約4.858%・全保存区間重複を維持する。**

追加compileの必要性は現資料から確定しない。費用比較をさらに判別する場合の第一候補は、固定schedule・保存価格の共有誤差と長q systematicの判別力を調べる限定再解析。それ自体も別途ユーザー承認が必要。既存の局所compiler精密化STOPは解除していない。今回の新科学計算・費用再会計・compile・samplingは0。

## 成果物と対象

- [PRICE_EVIDENCE_MAP.csv](PRICE_EVIDENCE_MAP.csv)：Partial／Full各18round×cosine/sine、合計72軸行。保存q/r/K/N/α、単価とSE、較正とholdout、直接価格の有無、A0の条件、固定JSON selector。
- [UNCERTAINTY_SCOPE.md](UNCERTAINTY_SCOPE.md)：統計SE、経験的残差、scenario、共有誤差、長qの未保証部分を分離。保存費用・区間・寄与率を転記。
- [MINIMAL_V3_VALIDATION_PLAN.md](MINIMAL_V3_VALIDATION_PLAN.md)：情報価値に基づく定性的な追加候補、再利用コード、既存STOP、別承認の境界。
- [EVIDENCE_INDEX.json](EVIDENCE_INDEX.json)：44固定ファイルのcommit/path/blob/byte数/SHA-256/URL。
- [validate_metadata.py](validate_metadata.py)：標準ライブラリだけのmetadata転記・照合。科学moduleのimport、価格計算、fit、compileは含まない。

対象はlinear H4、隣接1.0 Å、STO-3G、8 system qubit、legacy DF12。Partial L_D3、Full L_D12、δ=.02、m0–17、q1–131072、保存coherent opt2の選択済みr/K/N/α/βを正本とする。controlled partial-S2、boundary_optimized、identity位相抽出、cosine/sine全Hadamard wrapper、measurement込み、状態準備なし。Qiskit1.3.0、basis rz/sx/x/cx、optimization level2、transpiler seed17、backend/coupling/layout/routingはnull。[OPT2DOC][OPT2DOC]、[OPT2JSON][OPT2JSON]、[COMPSRC6][COMPSRC6]

snapshot参照SHA-256は `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。価格回路は状態準備を含まないため、物理H基底状態／PF有効固有状態の利用可能性まで価格identityで検証したとはしない。V2の物理平均・IID測定・alias条件下の両候補 `CONDITIONAL_SUCCESS` と、V1の厳密単位半径・状態identity等の未保証をそのまま引き継ぐ。[V1][V1REPORT]、[V2][V2REPORT]

## 1. 直接価格、較正、holdout、外挿

すべて固定 `4591b8d75a2a6a55c74a51225f0c61648023c8ee` の保存価格とSOURCEに対応付けた。次表のnは**各qの価格trajectory数**で、量子shot Nではない。

| 最終回路族 | 保存された同条件直接q | 標本数・出典 | 最終選択roundとの対応 |
|---|---|---|---|
| Partial r1/K2、δ.02 | 1,2較正、8固定holdout | 各n8、initial | m0/q1、m1/q2、m3/q8で直接価格あり。m2/q4は未測定。m4–7/q16,32,64,128は同rのq8より先 |
| Partial r2/K2、δ.02 | 1,2、8 | 各n8、initial | m8–9/q256,512は外挿 |
| Partial r4/K2、δ.02 | 1,2、8 | 各n8、initial | m10/q1024は外挿 |
| Partial r8/K2、δ.02 | 1,2、8 | 各n32、fresh extension | m11–12/q2048,4096は外挿 |
| Partial r16/K2、δ.02 | 1,2、8 | 各n32、fresh extension | m13–14/q8192,16384は外挿 |
| Partial r32/K2、δ.02 | 最新1,2、8、別legacy16,32 | 最新n32、legacy holdout n8、A0で照合 | m15–17/q32768,65536,131072は同族の短q検証を参照する長q外挿 |
| Full r0/K0、δ.02 | 1,2較正、16,32 holdout | 各n1、legacy opt2を再使用 | m0,1,4,5で直接価格あり。m2/q4、m3/q8は未測定。m6以後/q64–131072は外挿 |

Partialで最終tupleの直接価格が存在するのは3round×2軸、Fullでは4round×2軸。ただし実際に使ったモデル単価は、直接holdout価格への置換ではなく、**その点でもq1/2予測**である。q1/2で保存予測と較正平均が一致しても、別の独立実験ではない。

Full δ.01にはq8直接価格があるが、今回のδ.02へ移さない。旧opt1のq4/q8、別rのq16/q32も同条件のopt2直接証拠として追加しない。q4等の「未測定」は固定価格証拠内での未発見を意味し、すべての非公開rawに実験が存在しないと断定するものではない。

CSVでは較正式の外挿と、他holdoutに挟まれたq範囲を別欄にした。q>2はq1/2 fitの外であり、q4がq8 holdoutより小さくても直接検証済みではない。q>32の全軸は明示flagを付けた。長q物理信号のV1/V2検証は長q価格の検証に流用していない。

## 2. A0が実際に支持する範囲

A0は新compileを行わず、最新fresh q1/2 n32のfitを、既存opt2 q16/32 n8の固定holdoutへ適用した。L_D3、δ.02、r32、K2、同snapshot/compiler/wrapper、`support_run_le_1`、RZ、各軸という限定条件である。q16約3.345%、q32約4.911%は既存5%点誤差gate内。selected q32の標準化残差2.012152530848661は1.96を上回り、統計的同一性とはしない。full-basis q32は約5.598%で非gating診断FAILを保持する。最新較正に対する全metricの検証域はq8まで。[A0][A0]

最終q16/32はr1、最終r32はq32768以後なので、A0と最終選択tupleの直接一致はない。後半r32回路族の短qでの経験的支持としては使えるが、その長q価格を保証しない。A0、LEGACY、M08や別解析に同じholdoutが登場しても独立価格取得として数えていない。

## 3. 保存不確かさと費用順位を妨げる要因

保存点費用はPartial `1263314146772.9375`、Full `1327822299806.0`、点比 `.9514180827942961`。状態準備なしの条件付き点選好はPartial、約4.858%低い。local5%、per-r selected、per-r full-basis補助、transfer25%の**全保存区間は重なる**。undeterminedは同等性の証明ではなく、現不確かさ下で優位が確定していないという意味。[OPT2JSON][OPT2JSON]

主要な未解決要因は、共有q1/2較正のsampling成分と依存、有限holdoutの経験的model discrepancy、q>32の未評価systematic。5%/25%は感度scenario、per-rは短q残差を長q費用へ移す経験的帯で、厳密な全長q同時CIではない。Fullのn1/SE0・短q残差0は未知systematicを消さない。異なる種類の幅に同じ信頼水準を与えて結合していない。

coherent保存per-r scenarioはq8診断に基づき、後続A0のq32残差へ自動更新されていない。これは異なる解析版の関係であり、今回A0を代入して新しい区間を計算していない。保存 `best/scenario_intervals` と `comparison/scenarios` の最下位桁の丸め差もそのまま記録した。[不確かさの定義と全保存区間](UNCERTAINTY_SCOPE.md)

比較上重要な後半3roundの保存寄与率はPartial `.9139324437497927`、Full `.8302589410503727`。両者とも最終qは直接価格域を大きく越える。これによりr32とFull tail-free長q回路族を候補に挙げられるが、具体的な次q・標本数の数値最適順位までは確定できない。

## 4. 局所追加検証の情報価値と最小候補

M06-Fは51task監査と局所precision gate通過後も区間が重なるため、追加局所compiler精密化を停止済み。今回そのSTOPを維持した。既存long-q holdoutは条件付き候補である。[OPT2DOC Research routing][OPT2DOC]、[旧V3候補][OLDPLAN]

単にq1/2のSEを下げる目的だけでは、未知の長q systematicや未測定状態準備が残るため、費用順位を解決できるとは期待できない。**最小推奨はR1：保存価格・固定18roundを使った共有誤差と判別力の限定再解析**。これは今回未実行であり、ユーザー別承認を要する。結果から新compileの価値が認められる場合だけ、主要r32／tail-free回路族の例えばq64/128の限定holdoutを候補にする。qとnは未固定で、最大q131072の直接compileを前提にしていない。

その中間qのr32 probeは回路族への検証であり、最終q64/128のr1 roundとは区別する。短qを通過しても最終長qの保証へは進まない。r1/q16,32の補完は、R1で比較上の情報価値が確認された場合だけの別候補。STOP変更にはsystematic構造を調べる科学的理由と期待する判別力を示し、ユーザー判断へ戻す。[候補ごとの条件・再利用・承認](MINIMAL_V3_VALIDATION_PLAN.md)

## 5. 今回維持できる結論と価格だけでは言えないこと

| 判断の層 | 今回言えること |
|---|---|
| 条件付き点推定 | 固定H4/DF12/δ.02、V2の条件付き18round energy task、準備なしcompiled-RZモデルではPartial点推定が約4.858%低い |
| 保存不確かさ内 | 全区間が重なり、頑健な価格順位・最終総資源優位は未確定 |
| 長q移送が限定された場合の可能性 | 同じ固定taskの準備なしcompiled-RZ比較の解釈が進む可能性。区間が分離するかは新解析・測定結果次第で、今は予告できない |
| 価格検証だけでは言えないこと | 状態準備込みの優位、PF有効固有状態入力・単位半径下界の厳密正当化、実backend/noise、別分子/rank、一般最適性 |

保存shot数はPartial13588、Full11162なので、共通準備costも一般に相殺しない。保存point break-even `26590335.133166734` RZ-equivalent/shot、P=0で既に全scenario重複という境界を引用するだけで、新しい準備価格・総費用を評価していない。暫定CA/10・energy成功条件を新しい正式claimへ採択していない。

## 6. 固定資料・SOURCE・再利用の確認範囲

必須のM06-F文書／coherent JSON／最終監査／A0／legacy opt2／compiler-transfer解析、initial/fresh32 manifest・aggregate・provenance、V1/V2 REPORT、旧MINIMAL_VALIDATION_PLANを固定Git objectから取得した。規範文書は既に全文確認した固定 `9a0bc7f17f47823e46c3945734127e9f0d09154c` の研究概要・目的・方法・評価計画と、V1/V2後続成果を優先して再利用した。VSTATUSとvalidation_manifestは旧証拠statusへの索引として参照し、最新V2の状態へ勝手に書き換えていない。

SOURCEは6主要入口に加え、affine/paired price/統計/runner/validator/testsの関連部分を静的に確認した。主な対応は次のとおり。科学関数は今回呼び出していない。

| 固定SOURCE | 確認した処理・再利用判断 |
|---|---|
| OPT2SRC | `compute_full_opt2_randomized_cell`、`compute_full_opt2_deterministic_cell`、`evaluate_proxy_cell`、`_measured_model_half_width`、`_comparison`。将来の限定検証で再利用候補 |
| OPT2EXTSRC | `_load_combined_points`のfresh置換、`_direct_rz_measurements`、`_seed_audit`、`evaluate_coherent_extension_analysis`。最後の関数はH準備・再探索を含むので今回呼ばない |
| LINEAGESRC | `_require_matching_context`、`_source_contract`、A0固定holdout照合。r32/domainと再fit禁止を確認 |
| TRANSFERSRC／TRANSFERCOMPUTE | 同trajectory再compileとfocused再集計、旧mixed contextの分離、Full providerを確認 |
| DECISIONSRC／ABLATIONSRC／PRESRC | `_axis_model`、`_pair_model`、`AffineAxisCostModel`、affine SE、保守的総和と1.96係数。新しい数値適用なし |
| COMPSRC6／FITSRC | `_statistics`、`_compile_paired_requests`、`_finite_inputs`、`fit_affine_holdouts`。両軸・policy pairing、wrapper measurement境界を確認 |
| OPT2TEST／EXTTEST／A0TEST／TRANSFERTEST | 保存schema、raw条件付き再構築、tamper、scope検査の定義を読んだ。科学importやrebuildを含む既存testsは今回実行しない |

新しいmetadata専用処理だけを追加した理由は、既存科学validatorのimport/rebuildが価格評価やfitを含み得るため。既存JSONのキー・selectorとGit object読取を再利用し、科学モデルを再実装していない。CSVの価格・SE・N等は原欄からの単純転記に限る。

M06-F初期36＋fresh15の51/51 PASS、seed非重複・aggregate再構築一致は**保存監査の引用**。今回51taskを再監査・再compileしていない。aggregate2本はtask/result fingerprintの索引で、個別価格を全保持するrawではない。既知LOCALの各runのtasks/checkpoints/worker_results directoryは未発見。raw全hash・全decode・他環境照会は実施せず、任意raw再構築を完了済みとしない。

carrier commitは保存物の所在で、実行SOURCE commitそのものではない。initial computeは `26aa95c7b3f5ecd1f358280e58073ca2ee44ea5c`、extensionは `7753eb71844e429facdb6f32c2def6e37379541f`、どちらもdirty provenance。10項目のcompute SOURCE hashを記録したprovenanceと過去の照合結果を利用し、全推移依存のclean再生成・immutable CI・外部再現へ格上げしていない。[INITPROV][INITPROV]、[EXTPROV][EXTPROV]、[AUDIT][AUDIT]

## 7. 技術検査・原本保全・公開

実作業LOCALのAGENTS.md／PROJECT_MAP.md、出力親V2 commit側のAGENTS.mdとPROJECT_MAPの入口・M06-F部分、正式共通運用指示 `research_project_instructions_unified_2026-10-25.md`、特に§8.1・§8.4・§10を実際に確認した。LOCAL現行指示と出力親の文書は同一版ではなく、正式共通指示は出力親treeにない。取得したLOCAL原本を適用し、未取得のGit固定版を読了とはしていない。親directoryのAGENTSは存在せず、docs配下にも追加AGENTSなし。

新成果物は独立したGit作業領域・専用branch `codex/pr-rpe-v3-cost-transfer-preflight-20261103` にだけ作成。親はV2固定commit。元研究treeではGIT_OPTIONAL_LOCKS=0/GIT_NO_LAZY_FETCH=1を用い、元HEAD・dirty/untracked・refsの観測と固定object読取だけを行った。科学SOURCE/input/raw/QPY/result、grant/ledger/marker/STOP、他研究branch/worktree/archiveは変更しない。SSH・SERVER1/2調査、fetch・prune・cleanupは行わない。

公開前の観測で元HEADは不変、開始時statusの全行も保持されていた。一方、別の整理結果TSVの追加、別local branchの更新、remote-tracking refの削除を観測したため、**元status・全refsが不変とは報告しない**。本作業からそれらの変更・復元・停止操作は行っていない。固定44ファイルのobject identityは照合済みで、これらの並行する変化を根拠に科学再計算や保全処理を繰り返していない。

技術検査は44固定ファイルのbyte/blob/SHA-256、CSV72軸の欠落・重複・原欄一致、auditとcoherentの直接価格表一致、A0/r1取り違え等8件のmetadata人工検査でPASS。既存科学module import・新価格fit・費用再会計・compile・samplingは0。**このPASSは価格モデルの科学的妥当性・費用順位の保証ではない。**

公開対象は匿名化した6成果物だけ。内部host/IP/絶対path/資格情報/非公開rawを新成果物に含めない。既存証拠の相対path・固定hashと公開URLを参照する。元の非公開provenance本文やrawをコピーして公開しない。公開時には専用branchのcommit/push、remote SHA、公開6ファイルの全本文を照合し、その成否を最終チャットで報告する。本文一致を未確認のまま公開完了とはしない。

## 8. 次の担当・必要な承認・mandatory STOP

今回の完成条件はpreflight資料の作成・技術照合・公開であり、費用不確かさを解消することではない。現在の条件付き結果を維持するだけなら追加作業は不要。

ユーザーは、準備なしの固定18round費用差をさらに追究するか、現在のundeterminedで止めるかを判断する。追究する場合の最小候補はR1の限定保存値再解析で、**ユーザーによる別途明示承認後にCodexが担当**する。追加compileはR1の情報価値判断と既存STOP変更を別途確認する候補にとどめる。

**既存M06-Fの局所精密化STOPを維持。公開・remote本文照合後mandatory STOP。V3科学再解析、compile、モデル再較正、費用再会計、詳細科学レビュー、中心claim正式採択へ自動移行しない。**

## 固定参照

[A0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/m06f_a0_proxy_lineage_reconciliation_v1.json
[AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_fresh32_audit_20260925_065827.json
[COMPSRC6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_scope.py
[EXTPROV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_fresh32_20260924_232418/run_provenance.json
[INITPROV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_initial_20260924_023012/run_provenance.json
[OLDPLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/953b3bbcd05c495ff24d29c7b7e26ea0714907b7/docs/research/pr_rpe_energy_bridge_readiness_20261103/MINIMAL_VALIDATION_PLAN.md
[OPT2DOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research_direction_full_opt2.md
[OPT2JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json
[V1REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md
[V2REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
