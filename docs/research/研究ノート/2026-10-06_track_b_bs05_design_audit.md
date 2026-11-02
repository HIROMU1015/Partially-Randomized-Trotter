# 2026-10-06：Track B BS-0.5 docs-only設計監査

受領GPT判定`REVISE_BLOCK_SYNTHESIS_DESIGN_BEFORE_IMPLEMENTATION`に従った。
[入口](../../tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md)、
[ordinary形式仕様](../../tracks/algorithm_codesign/bs05_ordinary_finite_rte_baseline_v1.md)、
[pilot amendment](../../tracks/algorithm_codesign/block_synthesis_pilot_amendment_v2.md)。

現candidateはfinite Pauli多項式取得＋同辞書operator LCUで、generic sparseにも同処理を許すと
別最適化・情報access・古典取得手順が定義されていない。現new-method claim/専用armを外した。
これはscience equality resultではなく設計同一性。design/applicationとしての価値は未採択でGPTへ戻す。
既知制約緩和以上の追加知見もないと判断された場合はpilotなしのroute closureを推奨する。

O coherent operator / C full controlled channelを別比較へ整理。ordinary K=2,r=1のevent分布・phase・B2・
controlled chronological列・precision候補・raw T/Clifford/CX/ancilla ledger・十分Bernstein shotsを数式で定義。
実synthesis sequence/count/error・tool identity・accuracy/capsは未取得/未freezeで、実行contract未完成。
primaryは多資源Pareto、Pauli T=0だけではwinnerにしない。
12 target案のcommuting4をcontrolへ変更、noncommuting/boundary8がperformance。candidateはdeltaなしで除外。
Wada et al., arXiv:2512.06260v1を利用者返信で同定し、指定PDFの§II/III定義・定理statementを確認した。
grouping自体は既知、linear task/joint Φとのtarget差・group ancilla/実装costを未閉事項として残した。

base `2c8c022db39c3582d6175fcf25a6752037727a21`から独立docs-only branch/worktreeを作成。
旧v1/JSON、SP result/marker、BF/BM closure、Track A source/result/statusを変更しない。
実装・science・tests・matrix・solver・library生成・合成・compile・trajectory・GPU・NPZ/DF/Hamiltonianは0。
必要文書だけcommit/push、固定GitHub commitを提示し、mandatory STOP。次scope判断はGPT側。
[監査manifest](../../../artifacts/track_b_bs05_design_audit/2026-10-06/audit_manifest_v1.json)。
