# 相対時間発展block追加技術監査：GPT判断用資料

2026-10-13 JST。基準 `25cbc6e79480230afb841201b6917b987cc2a3d7`。
[技術監査・対応表・費用境界](qc_relative_block_method_cost_audit_v1_20261013.md)、[受領指示](../../artifacts/qc_relative_block_method_cost_audit/2026-10-13/instruction_intake.md)、[機械可読status](../../artifacts/qc_relative_block_method_cost_audit/2026-10-13/audit_status.json)。完了済み方針レビューの再実施ではなく、方法差・詳細実装への投資判断の材料。

1. **方法差は残ったか**：現構成は `REDUCES_TO_KNOWN`。THRIFT相対積→occupation pairのSU(2)→affine係数・関数の可逆計算→回転→逆算という具体的対応を示した。枝列挙回避、平均phase相殺、共通Δ近似、共有θはstrong compressed法にも適用できる。独立した操作列・前処理・計算量・費用改善は未特定。先行研究の網羅的な初出判定・全実装の不能性は証明していない。
2. **今回閉じた技術点**：known constant-load adderならdetuning往復はz(4w−6) Toffoliという条件付き静的会計。共有θの誤差はmin(1,|t rho|)ηθで抑えられる。全occurrenceからfractional precisionを指定する式、許容function費用B_F、controlled step数比でのbreak-evenを明記した。全対照へ同じ利益を与え、新規算法・実測speedupとしない。
3. **未閉鎖点**：取得したQualtranの固定sourceではArcSin等は資源式/古典参照と完成回路分解を区別する必要がある。atan2/hypotの本domainのuniform誤差・全garbage・control・T/CX/depth/ancillaは `UNKNOWN`。分子M_eff/z_j/DF ranks/coupling/PF定数/step、ground-energy全task費用は未評価。
4. **追加投資を判断する材料**：現構成だけを完成してもknown compressed法との差は生まれない。独立算法を目的に継続するなら、先に具体的な差と同機能の費用条件が必要。既知法の化学実装・適用条件研究として扱う選択も研究目的・価値の科学判断になる。技術的に作れるという事実だけではGOを出さない。
5. **追加情報で変わる判断**：primitiveの精度付き実費用はB_F内かという実装可否、分子係数/stepはnative/DFとの同品質利益、ground branch/overlap/statisticsは全energy task利益を判別する。これらの性能情報だけでは現constructorの独立新規性を証明できない。必要性を先にGPTが判断し、後でscopeと明示実行認可を固定する。

Phase 1を完了し、独立差が見つからなかったため詳細function backendへ進むPhase 2は起動していない。新数値計算・compile・synthetic・既存13再実行・GPTレビューは全て0。
既存H2/H4案は変更せず未認可。第二候補自動移行なし、N1 STOP/N2 HOLD/N3 CLOSED、旧source/results/auth/STOPと既存global manifest FAILを保持。

**次の境界**：GPTによる継続・縮小・終了／追加投資の重要科学判断。レビュー開始には改めて明示承認が必要。Codexは今回その判断・レビューを代行せず、Mandatory STOPを維持する。量子化学ground-energy低コスト化という目的は変えない。


公開済み固定資料：payload `eba04a87dc790d7688b0fce0b906ef12151b781a`。[GitHub監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eba04a87dc790d7688b0fce0b906ef12151b781a/docs/research/qc_relative_block_method_cost_audit_v1_20261013.md)、[公開receipt](../../artifacts/qc_relative_block_method_cost_audit/2026-10-13/publication/receipt.json)。27変更blobsと基準13優先blobs、生成source3件、追加リンク34件をGitHub再取得で確認。新科学計算・GPTレビュー0、Mandatory STOP。
