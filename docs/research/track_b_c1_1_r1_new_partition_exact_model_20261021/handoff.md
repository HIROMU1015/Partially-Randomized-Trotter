# R1 GPT・ユーザーhandoff

**R1_LIMITED_REANALYSIS_COMPLETE。実行1回・retry0。D0の新partition来歴でbasis/exact model packetと限定DF数値整合性を取得した。旧C1-1/D0結果・STOPは保持。科学的GOや原著価値は判定しない。**

1. D0/R1 canonical payloadは1,834 bytesで完全一致、hash `16b35c1dde1a6807d39c733defd87e315e4ac7317a831b240d0f7214e744a834`。旧hash `722ec007...` との不一致は新来歴の履歴として保持。
2. 指定H4 1.00 Å / STO-3G / DF12 / 8qubit / prefix3のNPZ・全必要配列・state/sector・H hashは一致。stateの厳密ground性/物理準備を認定していない。
3. 13 basis registry定義を取得。操作列、descriptor、qubit、parameter、局所行列hex/hash、fragment/component対応を保存。`BASIS_RECONSTRUCTED_PROVENANCE_DISTINCT`。旧操作列がなくbyte-exact/native実装同一性は未確認。
4. 324 labelの新binary64係数をexact dyadicとして採用。全p>0、Σp=1、λp sign=cがexactに成立。model SHA `e96ea0faea31bc6a784dfb8dc6f7cf4856789b0256e4a1c53b63ce087c86dc49`。exact λと全pはcore/label chunks。旧係数hexとは135差、ID順・signは一致。旧実行入力と同一ではない。
5. 全DF再構成の相対spectral残差 `4.609337135740557e-13`、同じ対象の既存 `1e-10` 内で`MODEL_BINDING_NUMERICALLY_SUPPORTED`。rigorous operator norm/bindingは`MODEL_BINDING_UNCLOSED`。旧係数を新basisに置いたhybridは元旧operatorの再現ではない。
6. 草案T=4/5、q=1、r=4、m=5ではx≤1をexact確認。実schedule/time/signの固定前なので全occurrenceはCONDITIONAL。G6 sampler/provider/event/native/N/samplingは未実施。
7. 232固定ファイル、旧C1-1/D0実worktree61files、非公開原本・環境の非変更を確認。process/terminal exit0一致、raw stderr空、logger二次障害なし。SOURCE修正・retryなし。

## 次の担当・必要な承認（提案、未実行）

| 担当 | 作業 | 必要な判断・承認 |
|---|---|---|
| ユーザー・GPT | 新来歴の技術結果を読み、model/provider/schedule/confidence/費用のどこを次に閉じるか判断 | 科学的価値と検証の必要性・範囲。R1成功だけで実装・論文GOを認定しない |
| Codex（別指示後） | 採択された範囲のG6–DF adapter/strict provider仕様・証拠要件・契約を準備する | 新たな準備・実装指示。旧packet/STOPは保持 |
| ユーザー | 後続実装・科学実行・新証明の開始をそれぞれ明示承認する | 新SOURCE/input/environment、実schedule、誤差保証・STOPを固定。R1 receiptの再利用不可 |
| Codex（該当承認後のみ） | 承認された後続作業を実施する | 現在はadapter/native/sampling等へ進まずmandatory STOP |

Branch `track-b-c1-1-r1-new-partition-exact-model-20261021`。基点 `8188c8a50f79ea9d17fddb987c651650b58aa22c`、R1 SOURCE `418a76987782ce50be670b95b8f998043b3f40f5`、新authorization `3fb433dcd1879e43fe444795e65dbd711ee5e1b4`。最終公開HEADはpublication receiptと最終チャットを参照。

レビュー最小セット：[README](README.md)、[partition](result/partition_identity.json)、[basis index](result/basis_registry_index.json)、[component比較](result/component_comparison.json)、[exact core](result/exact_model_core.json)、[model identity](result/model_identity.json)、[数値binding](result/model_binding.json)、[short-step](result/short_step.json)、[保全確認](preservation_verification.json)。core内のchunk pathsはresult directoryからの相対path。**公開・保全確認後mandatory STOP。**
