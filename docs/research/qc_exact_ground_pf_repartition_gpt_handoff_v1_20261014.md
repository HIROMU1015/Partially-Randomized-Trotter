# 一体項再配分：GPT重要科学判断への引継ぎ

2026-10-14 JST。Codexの限定技術監査を完了した。GPT科学レビューは未開始。第一候補の正式採択・継続・終了・第二候補への移行を判断していない。[技術監査](qc_exact_ground_pf_repartition_technical_preaudit_v1_20261014.md)、[受領方針レビュー原文](../reviews/quantum_chemistry_pf/qc_exact_ground_state_pf_direction_rereview_2026-10-14.md)。

## 1. 今回確認したこと

| 項目 | 技術所見 | 判定の限界 |
| --- | --- | --- |
| 二分割PSD二次形式 | BCH符号、Gram/range、二重交換子は整合。E0入力なし。Mに加えてbも1-RDMへ下がる | 二分割可解class限定、finite-h/gate優位ではない |
| 一般DFの単一局所更新 | 三次t項消去、線形≤3-body/二次≤2-body。一般位置のQはPSD保証なし | 複数非可換方向の同時更新や全logの二次性にしない |
| 最初のdensityとの更新 | 変化分のground期待値を一体二重交換子へ還元、1-RDM応答表をO(n³)時間/O(n²)保存で生成可能 | gamma取得と初回d0は別費用。新規性・強compact対照に対する効率差は未確定 |
| 形式検査 | 抽象指数/logからの独立word演算27/27 PASS | 科学H/state/expectation/eig/PF性能/optimize/compileは0 |
| 既知法への還元 | FFF自由度、state-specific BCH目的、quadratic solve、再帰/cachingは既知 | 1-RDM局所応答表の明示記載を確認できなかったことを初出性証明にしない |
| 実装class | density frameとGaussian+diagonal classを保持 | H0 dense化、追加linear位相、fusion/control/synthesisの実費UNKNOWN |

## 2. 方針レビューから精密化した点

F0=Aを外側に置く固定S2で、最初のdensity fragment F1=Bから同frameのXを移す。尾部Tは任意で[X,T]を0と仮定しない。

\[
D(X)-D(0)=\tfrac1{12}[A,[A,X]]+\tfrac18[H,[X,A]]+\tfrac1{24}[X,[A,X]].
\]

exact ground上で中項が消え、残りはA,Xが一体なら1-RDMだけで取得できる。これは「一般DFの全主要誤差を1-RDMで評価できる」という主張ではない。絶対d0は初回のcompact BCH等による取得が必要。E0を入力・位相補正の正解として与えない。

二分割の1-/2-RDMという元上限、一般位置の3-/2-body上限は誤りとして撤回する必要はない。特定orderingとground恒等式によるより狭い応答を確認した。全N null方向、PSD条件、縮退groundの制限、finite-h残差・overlapの区別を維持する。

## 3. Phase A/Bの扱い

Phase Aで「FFFの再配分を数値的に試すだけ」より具体的な**情報rank・応答表・再利用規則**を閉じたため、この限定classについてPhase Bの静的整理まで進んだ。production sampler/collector/optimizerは作らない。一般位置や第二候補を追加しない。

一方、標準compact BCHにground恒等式と合法cacheを加えた対照が同じ表を同じ費用で作れるかは未決。三点の既知評価で二次式を復元する対照も許す。「式がきれい」「主要係数が下がる」だけで新規性・全費用優位を認定しない。この科学的判断をGPTへ戻す。

## 4. GPTが判断する事項

1. FFFの既知自由度・state-dependent PF目的・compact BCHに対し、この1-RDM応答生成に独立した算法上の寄与が残るか。
2. 初回d0・γ取得を含めた生成費用差を、同情報・同自由度の強い既知対照と比較する合理性があるか。
3. scalar-corrected S2、S4、既知外挿/再分割、適用可能なcorrected PFを含めても、finite-h bias/overlap/controlled総費用へ改善がつながる機構があるか。
4. [最小pilot案](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/minimal_pilot_design.json)を具体化する科学的価値があるか。第一候補を継続するか、別途第二候補を検討するか。

これはCodexの採択推薦・実行GOではない。未知であることだけを継続理由にしない。既知compact対照が同じ情報費用に達する可能性は、明示して判断材料に含める。

## 5. 最小pilot案と承認境界

案は一般DFの固定Hamiltonian一件で、(i)応答表の一致/取得費用、(ii) finite-h error/overlap、(iii)同情報controlled資源の三つを分離して判別するもの。toyだけで一般DFの利益を認定しない。E0はtruth評価側に隔離し、候補選択はd0/γ/係数/保証intervalだけを用いる。

入力pack/SHA、source、正確な状態供給、spin/number制約、directions/bounds、h/energy/synthesis/confidence、memory/output/log/call capsは未固定。新規数値取得は行っていない。案をexecution-ready/preregisteredと表示しない。GPT科学判断の別開始承認、その後に必要ならsource/input固定と別の実行承認が必要。

厳密状態仮定を外してstate-preparation研究へ戻したり、無料の全2-RDM/真E0を設計へ入れたりしない。block-encoding/qubitization、Track A/Bの作業、C1–C4や終了済み系列を起動しない。

## 6. 公開と停止

Branch `qc-exact-ground-pf-onebody-preaudit-20261014`、base `5f35988d3145641739fb83da98567772183046ef`。[原レビューSHA](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/received_review_identity.json)、[文献](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/literature_registry.json)、[scope/status](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/method_status.json)、[保全索引](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/preservation_index.json)、[再取得receipt](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/publication/receipt.json)。

387旧files/146旧manifest entries/non-result fields/global FAIL/root135pathsを維持。受領レビューに記されたC1現系列終了・C2対照保持・C3/C4現構成投資見送り、N1STOP/N2HOLD/N3CLOSED/THRIFT現template CLOSEDを継承し、旧source/結果/認可/STOPを書換えない。今回の科学数値calls0、C1–C4再監査0、Track A/B科学作業0、GPTレビュー0。公開・GitHub bytes一致確認後Mandatory STOP。次は**GPT重要科学判断の別明示開始承認**。

## GitHub公開確認・最終STOP

[再取得receipt](../../artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/publication/receipt.json)：payload `6711a709ad13144972474effe1faf3933cd7f7fb` の26変更blobs、387保護files、固定3参照、受領レビュー38821bytes/SHAをGitHubの独立cache（local alternatesなし）から一致照合した。形式checkerを再実行せず、新科学数値calls0。receiptと本追補を後続metadataとして公開し最終リンクを照合する。独立新規性・finite-h/総費用優位・正式採択は未判定。Mandatory STOP、次はGPT重要科学判断の別明示開始承認。
