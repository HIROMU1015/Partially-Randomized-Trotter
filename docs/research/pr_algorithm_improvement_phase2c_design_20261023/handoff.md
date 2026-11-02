# Phase 2C引継ぎ：設計を完成し、実行前で停止

2026-10-23 JST。主案は**残差fragment別のPauli／frame辞書選択**。同じF_lを既存回転基底Z/ZZ辞書で残すか、固定registerのPauliへ展開するかを選ぶ。物理H、sector、state、scalar policy、D/R split、q/r/K、既存finite-RTE samplerを固定する。

手順は、保存係数から二辞書を作成、選択fragment間で同一Pauliをcollect、canonical λ/p/次数law/Bを更新、既存run共有を含むliteral期待価格とNを評価し、二端点から一passずつ反転選択、typed sampler/circuit planを出力するもの。[主設計](ALGORITHM_DESIGN.md)は入力・情報取得・データ構造・状態遷移・擬似コード・fallbackまで記載した。

削減対象は典型的なchild rotationにも含まれるGaussian frame出入りとPauli実装構造の交換である。basis費が支配的とはしていない。λ/B/Nの増加、run共有の喪失、parity・制御・phase・prep/readout、古典係数／価格取得費が相殺し得る。Φは定義したliteral planの価格で、actual compiled費の予測・上界ではない。

**原PRの残差Pauli化・異軌道involution、既存toyの異種Primitive、既知cost-aware辞書選択との重複が大きい。** 新しい統合selectorの手順は具体化したが、独立した方法差は未確定。分類は `KNOWN_METHOD_APPLICATION_CANDIDATE` / `INDEPENDENT_METHOD_DELTA_UNESTABLISHED`。独立新法としての現在のclaimは `NO_DEFENSIBLE_NEW_METHOD_YET`。全手順が特定既知算法と完全同値だという証明は行っていない。[比較表](METHOD_DELTA_AND_PRIOR_ART.md)を参照。

数学・実装上は、同一Hの係数代数、mixed finite mean、order／sign／actual adjoint／ancilla phase、finite bit、価格伝搬、統計条件、sectorと実行空間の整合が残る。PO-01～09は全て未実行。個別Pauliが粒子数sectorを出る場合、sectorだけに制限した実行を使えず、途中postselectionで補うこともできない。[意味論](SEMANTICS_AND_PROOF_OBLIGATIONS.md)。

最小案は、4mode/2粒子、one-body D block1、残差density-square2fragmentの固定synthetic fixtureでF1の構成差・full-register制御meanを調べること。MCを始めず、小候補列挙と固定motifを使う。F1に意味のある差が残った場合だけF2へ進み、同fixtureのq1/r2/K0全有限trajectoryを候補に、同precision・同wrapperでN加重費と古典費を比較する。[検証計画](MINIMAL_DISCRIMINATING_VALIDATION.md)は未採択・未実行であり、新規実装／input生成／build/compileを要する。

今回実施したのは固定SOURCE・結果・文献の読解、設計、文書管理検査とGitHub公開。新しい科学計算、raw再集計、sampling、回路build/compile、native価格取得、新実装、新しい非自明な証明、grant発行・再利用は0件。固定公開資料のidentityと実際の読解範囲は[SOURCE_EVIDENCE_INDEX.json](SOURCE_EVIDENCE_INDEX.json)。文書管理検査は科学的検証ではない。

公開先は `HIROMU1015/Partially-Randomized-Trotter`、branch `research/pr-algorithm-improvement-phase2c-design-20261023`、起点 `fbb83680247a4565336ca177278255e32843cfb2`。payload commit・公開検査は[PUBLICATION.json](PUBLICATION.json)。最終remote HEADは公開後のチャットと外部管理記録で確認し、自己参照hashを文書へ埋め込まない。

既存SOURCE・科学input・raw・研究記録・review・原稿・grant・ledger・marker・STOPと、他branch/worktreeを変更しない。ユーザー指定のprivate Phase 2Bレビューは読解に使い、原本を公開していない。

**次の担当と承認：** ユーザーが、追加設計を求めるか、既知法のPR接続としてF1を採択するか、別方向へ移るかを判断する。採択後、Codexが対象SOURCE/input・検証範囲・call inventory・精度・STOPを固定した実装／実行契約を用意し、必要な別途明示承認後に着手する。詳細GPT/Claude科学レビューを自動開始しない。必要性が生じた場合は範囲と開始承認を別に扱う。

**Mandatory STOP：今回の設計・公開・管理検証をもって停止する。次の実装・科学検証・proof・compileを開始しない。旧研究のSTOPも維持する。**
