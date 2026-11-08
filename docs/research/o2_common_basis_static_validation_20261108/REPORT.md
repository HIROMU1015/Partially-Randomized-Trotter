# 共通basis label周辺化：限定静的技術調査

2026-11-08 JST／SERVER1 GPUA100-gleap-01。base 1bf88b251fd09caa0a227972ee0f1b56c5888e6a、独立worktree /tmp/o2-common-basis-static-20261108、branch codex/o2-common-basis-static-20261108。

**正確unitaryのsupport列保存なら、各labelのselected共役Z/ZZを元共通basisで表せる。** 現literal十分条件はこの書換えを判定しない。一方、保存gateへの一様誤差保証は未取得。同じ共通frame/readoutを既知groupingにも与えると独自の利益は残らない。

総合判定は **C：保存finite targetへの必要な数値保証は未成立**。理想条件下のbaseline比較ではB相当の既知groupingへの還元も成立する。実科学ready・新規性・科学的GOの認定ではない。

## 今回確認したこと

1. [演算子接続](COMMON_BASIS_OPERATOR_CONNECTION.md)：必須SOURCE5ファイルが固定39d8390と同じblobであることを照合し、completionの列copy、Gaussianの転置/JW規約、逆→対角→順の演算意味論を確認した。
2. 現keyはselected ID、A/B/V literal列等を要求し、original IDを同値証明に使わない。support/signを変えること自体は許すが、そのsupportに応じた別selected frameを元frameへ戻す証明は持たない。
3. 旧e98589aeの正232 label・232 singleton・非自明群0は、保存H4/1Å/STO-3G/DF12/L_D3/m0q1/δ.02/r1K2/atol0/actual Q/selected fullのn=0唯一slotに限定された正しい結果として保全。バグ/誤判定にせず、新群・件数を取得していない。
4. 共通frame V_f=Γ(O_f)† と、A/B/scalar/event phase、actual q/外側質量を保持すればbranch型測定でRe/Im Zを保存する。A=B†は一般には仮定せず、該当時の強いancilla-free対照も明示した。
5. 理想演算と保存角の理想gate、丸めmatrix、exact Fraction演算を分離。atol=1e-10やmetadata hashをsignal誤差上界へ昇格させない。旧certificate、新instrument、selected→fullの差は別問題。
6. [baseline・全費用](COMMON_BASIS_BASELINE_AND_COST.md)：selected PRのdiagonal_only・basis最適化を維持し、full basis/readout/feedback・古典d計算・群取得・誤差/校正費とM2/variance/rangeを比較した。通常PRへの改善条件は記号的に残るが、同じgroupingへの増分利益は0。
7. 独自改善を目的とするH4実験へは進めない。既知groupingの実用対照として別途評価するなら、まず一様target差・新測定誤差・公平なfull費用取得の有限仕様が必要。元basisが使えそうというだけで新群集計やinstrumentを推薦しない。

## 再利用・不足・境界

既存MのSlotLabel/資格・header binder・readout、support completion/sequence policy、DF Gaussian/Hadamard/actual-law・engine/accountingのSOURCEを静的に再利用した。既存Mの人工correctness・旧metadata契約/STOP、O2統合/非同値性選別を読解し、旧39 stageの再調査・test再実行は不要とした。

一次文献はHugginsのbasis rotation groupingとMitarai–Fujiiの条件付き直接測定の本文を確認した。前者の出版社本文はredirectで未取得のため同題arXiv本文を使用。論文のenergy/noise/postselection性能を複素PRへ移植せず、未知state情報・coherent PREP・reflectionを無料にしない。重要科学レビューや正式採択は行わない。

UNKNOWNは、保存selected/full列のoperator差、実common frame群/質量、new gate/readout/Born/grid誤差、hardware/動的profile、実variance/N/full価格・期待費用。次に実装・人工testが必要かは具体的誤差取得仕様の後に判断する。今回は紙上/SOURCEだけで理想接続と既知法への還元を判別でき、追加人工testも不要だった。

**新実装・runner/pipeline・人工test・科学計算・H4 metadata数値再解析・実state/Born/RNG/shot・回路build/compile/native synthesis・旧結果再集計は全て0。** 旧qualify_groupを緩めず、N1/N2/engineを再計算せず、既存code/certificateを編集しない。

実N1の新群・差の数値認証、実回路・Born/shot/価格、保存科学値再解析には対象input/SOURCE・有限call・新allowance/STOPを明記した別途許可が必要。新しい誤差予算、別slot/basis/control/sampling政策は今回採択しない。真空参照CSWAP、O1/O3/C-S、M/D/E旧研究、C1凍結解除へ自動移行しない。

## 運用・保全・公開

正式共通指示2026-10-25（SHA256 e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720）と適用AGENTS/PROJECT_MAP、既読研究概要・目的の関連箇所/最新追記、既存STOPを確認。§8.1/8.2/8.4/10/10.1/10.3と今回の静的限定指示を適用する。grant/承認証明・管理framework・重要レビュー記録は新設しない。

元main・dirty/untracked、O2統合/M/D/E/R0–G10/Parent/C1/A1/SCV/A2/J1/O1/O3/C-Sの旧SOURCE/raw/結果/grant/ledger/marker/STOP、他worktree/process/cacheを維持。既存索引/概要/当日ノートへ追記し、manifestの旧科学的statusを維持して今回の静的項目だけを追加する。共有OS/scheduler/cgroup/venv/cacheや他ユーザーを操作せず、文書作業へCPU/RAM/workerの恣意的上限を設けない。

専用branchでcommit/公開し、remote SHA・公開本文・blob/tree・固定SOURCE/旧結果と元main保全を照合する。最終SHAは最終チャットと作業用照合記録に示し、自己SHA追記だけのcommitを増やさない。

**公開照合後mandatory STOP。理想接続が成立しても、新しい群分け・instrument・実科学・重要レビューへ自動進行しない。**

公開前の文書技術確認PASS：新3文書の15参照、固定Git8 path、必須SOURCE5 blob一致、旧manifest25項目不変、既存Markdown5点追記のみ、元main HEAD/statusと既存2,136ファイルbyte保全を確認した。これは文書/byte照合で、演算子数値試験や科学計算ではない。
