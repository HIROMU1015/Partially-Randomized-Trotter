# R2-CS C0.5：PRアルゴリズム研究の横断調査

2026-10-24。研究識別子R2-CS-20261024。C0固定8cb41c40…の後続。新しい科学レビュー・数値再解析ではなく、既存結果・SOURCE・式・終了記録の照合である。

**提案はB：有用な構造はあるが、独立した構成選択研究の差分は未確立。** Aの新選択則・新しい安全除外条件は特定できなかった。一方、G6/P5の既存数理・生成構造まで「R3と同じで無価値」とする根拠もない。構成選択法としてのGO/STOPはGPT・ユーザーへ返す。**C0の16 endpoint P1は保留**し、契約実装・実行へ進めない。

## C0から修正する理解

1. **PR-2はM1-AのSELECTION_LIMITEDで止まった研究ではない。** 後続M1-B1は210 cellのcompile mapを完了し、M2は固定5構成のH4 1.30 Å transferをSUPPORTEDとした。PM-0では旧16件がprimary RZ点最小を保持したことを確認している。欠落したのはactual Paretoのもう1点であり、「RZ最良候補を落とした」とは言えない。元の結果前barrierは撤回しない。
2. **同じR=qrの構造分離は既存の保存結果で確認済み。** 同じsplit/K/T/Rの56 groupでtau・normalization・期待random applicationsが一致した。しかしsignal、shot、native価格、Gは一致しない。この関係は新しい発見や一般最適化則ではなく、利用可能なlaw-level情報である。
3. **Track Bには後続の正式closureがある。** BFの限定negative、BMのcompact BCH同値、Gaussian案の方法差不足、Parent allocationの限定成果保持＋探索終了、C1の探索後一時凍結を確認した。旧READMEの実行待ちや旧technical failureを現行状態にしない。
4. **G6は単なるproxy選択とは別の既存構成を持つ。** 局所Green query、全normalizerを使わない生成、P5少数群、有限bit補正・全trial estimatorは保持される。しかし分子PRの認証native総費用・事前G順位・安価なbias評価まで完成していない。
5. **より強いnormalizerの普遍的優位は既に反証済み。** P_Uのm=7反例と、固定辞書のsampling-only下界がある。normalizationが下がるというだけでnative費用・N・総Gの低下を結論しない。

旧資料は変更せず、上記は本C0.5における参照範囲と帰属の補正とする。[研究別crosswalk](algorithmic_evidence_crosswalk.csv)、[現行状態・原典への入口](algorithmic_evidence_crosswalk.md)

## 持ち込める情報と限界

|構造|事前に得られるもの|残る障害|
|---|---|---|
|S2のR=qr条件付き同値class|固定lambda/p/Kとmicrostepから法・normalization・raw操作数の再利用|deterministic挿入・boundary融合・bias/headroom・compiled価格は別|
|finite paired law、signed piece、整数allocation|時間/cutoff/法から係数と次数確率。既存RTE関数を再利用可能|高次pieceの整数配分はR総量だけで一致しない。DF入力/p取得も費用がある|
|G6の局所query/P5群とmoment上界|Q_i二乗=Iと正有理p等の前提で、状態・signal・全event列挙を使わず取得できる係数情報|native word価格・有限bit/selector/setup費用・物理DF binding・signal誤差の接続が残る|
|局所cluster、basis/support、q費用モデル|短い回路calibration後の条件付き価格予測|calibration自体がcompile。prefix/compiler/snapshot外の保証なし。全G選択保証ではない|
|誤差情報との結合|旧FR/BF/BM/PM/D3が必要情報と反例を提供|Oracle半径/正解biasを使わない、安価で有用な新しい保証は未取得|

詳しくは [structural_hypotheses.md](structural_hypotheses.md)。数理的に使える構造、取得可能性、取得が安価か、有用な選択になるか、独立新規性の5段階を分けた。

## 第2研究の残る可能性と次の判断

R3、SPRINT、cost-aware IS、一般multi-fidelityを超える新しい安全除外・選択則は今回の資料から認定できない。G6の既存成果は別研究に属し、第2研究の新規性として再計上しない。[一次文献とR3差分](prior_art_and_R3_delta.md)

追加数値実行を直ちに勧めない。先に、既存G6のlaw-level情報から**候補全signal/全native価格を知らずに使える具体的な判定命題**を一つ記述できるかを判断する。既存U上界・IS下界の再提示だけなら新しい差分にはならない。具体命題がない場合は構成選択法の探索を停止・縮小する判断が妥当な候補となる。[decision packet](decision_packet.md)

現段階では新しい科学検証契約を発行しない。P1保留を解除する根拠もない。研究継続を選ぶ場合でも、新機構・task・access・保証・比較classを先に固定し、必要な最小検証だけを別SOURCE/入力/承認へ戻す。

## 範囲・保全

remoteで指定4 Track B branchとPR-2 branchのHEADを確認し、C0以外の固定Git treeも参照した。全commit/rawの回収はせず、現行closure→結果→必要なSOURCE/式の順で絞った。探索範囲とversionは [source_catalog.json](source_catalog.json)、[調査範囲](scope_and_state.json)。正式運用指示の指定SHA原本は引き続き未取得で、手元版の§10.3欠落を記録した。

新実データ解析・ratio/regret/ranking・fit・signal・sampling・compile・simulation・有効grant・詳細科学レビューはすべて0。人工の抽象word/整数allocation検査と文書・状態・SOURCEの技術確認のみ。[technical_checks.json](technical_checks.json)

固定資料51件・22研究系列を整理し、技術検査27件はPASS。既存の抽象word fixture 9件と係数変異control 6件、人工整数allocation等を含む著者自身の検査である。再実行はworktree起点で次のとおり。

```bash
python3 -B docs/research/r2_pr_structure_c05_20261024/technical/check_c05.py
```

第1研究の採用結果、CEM/paper manifest/原稿/主表/Figure 2、C0、他Track BのSOURCE/raw/結果/grant/ledger/marker/STOPへ変更していない。新規成果物は本ディレクトリのみ。

**次担当：GPT・ユーザーがBという整理と、構成選択法の縮小/終了または具体命題に限った追加調査を判断する。Codexは公開・照合後mandatory STOP。**
