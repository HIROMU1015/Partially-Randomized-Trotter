# PRリソース評価全体の実装・公開ライブラリ活用監査（2026-11-13）

PRリソース評価系列の現行コードとその直接依存を、横断的に調査した。

- 対象：A `169d3316`、B `3216d27a`、C `83d219cc`、D `e7dca5f7`/`af6f27d7`、および共通の物理・回路SOURCE。
- 実施したこと：SOURCEの読解、人工入力・公開metadataでの軽い技術計測、Qiskitのgate定義のintrospection。
- 実施していないこと：コード変更、ライブラリ導入、環境更新、科学計算・再解析・simulation、実回路のcompile、固定SOURCE・結果の変更。

改善案は自動で実装していない。

詳細：
- [IMPLEMENTATION_AUDIT](IMPLEMENTATION_AUDIT.md)：領域別の現状と判定
- [IMPROVEMENT_CANDIDATES](IMPROVEMENT_CANDIDATES.md)：20候補の評価
- [RECOMMENDED_PLAN](RECOMMENDED_PLAN.md)：変更する価値があるもの6件

計測script：[decoder](bench_decoder_artificial.py)、[candidate_id](bench_candidate_id_tests.py)。

## 結論

**全体として、物理計算・compiler profile・保存値の扱いは妥当で、公開ライブラリへの置換が必要な箇所はほとんどない。**

- 研究の意味を左右する部分は、固定SOURCEとして維持するのが正しい。対象：binary64の演算順、Decimal ceiling、PCG64のlaw、compiler設定、数値gate。
- SciPy・Qiskit・NumPyは既に適切に使われている。
- statsmodels・pydantic・numba等を新たに導入する利益は小さい。

優先的に扱うべきなのは次の3点。

1. **RTE tailの制御ZZ回転が、高コストの汎用control形になっている（研究上の判断事項）。**
   - PR・Full-H RTEのcontrolled tailは`RZZGate(2θ).control(1)`を使い、Qiskit 1.3.0では**CCX·CRZ·CCX**に展開される（gate定義で確認、transpileはしていない）。
   - 等価な`CX·CRZ·CX`より、1回転当たりCXが10個多く、T型・H由来のRZも多い。
   - 決定論PFは効率的な形を既に使っているため、**PRとPFの比較条件が揃っていない可能性**がある。構造S（C1）は安価な形で数えており、これがC−Sの主成分である可能性もある。
   - 小回路での同値性とcompile数の確認（要承認）の後に、価格定義をユーザーが判断する（R1）。
2. **共通の`candidate_id`が、毎回SOURCEをparseしている（技術、効果が大きく、リスクが小さい）。**
   - 1回7.3 ms、B `_validate`の約90%を占める。
   - memo化すると0.005 msになり、`check_rpe_new_evidence_intake`は75 s→16.5 sになった（実測、全件PASS）。
   - 数行の変更で、全担当のpipeline・testが速くなる（R2）。
3. **RPE decoderのvectorized分岐選択が、O(反復·q)のmemoryを使う（AのS3の前提）。**
   - M=12・2000反復で125 MBの一時配列が必要。10⁵反復では約6 GB/round（推定）。
   - 等間隔の候補に対する閉形式では、同じ結果で180倍速い（人工位相での実測）。
   - S3の実装前に、同値性test付きの新関数を追加すべきである（R3）。

その次に価値があるもの：

- **固定SOURCEをexecで再利用する際の自由変数の事前検査**（R4）。S0a run01の`NameError`によるSTOPの再発防止と、A `NodeTransformer`の一致件数検査。
- **共有module 3版の所有・固定方針と、既存の誤検出FAILの解消**（R5）。
- **D inventoryのcache**（R6、任意）。
- **S3の区間計算**：導入済みSciPyの`binomtest().proportion_ci`を使う。statsmodelsは不要。

## 変更不要と判断した範囲

| 範囲 | 判断理由 |
|---|---|
| PF/PR/RTEの物理核（eigh/spectral apply、finite RTE、spectral power、expm_multiply、Decimal ceiling） | 小次元（36/400）で高速。演算順と数値gateが結果の意味に関わる |
| nominal shot式・Monte Carloのbinomial生成 | 妥当 |
| compiler（opt1のpass構成）・fingerprint cache・経路単位のprocess並列 | 科学profileそのものであり、並列化も既に適切 |
| C `MeanKernel`／K0/K1の区間演算 | 結果は既に出ている。python-flint `arb_mat`は任意の独立照合に限る |
| `same()`・identityの埋め込み・git helper・データ検証 | 測定上の負担が小さく、統合は形式的な共通化にとどまる |

## 引渡し

- **ユーザー／GPT**：R1〜R6の採否と担当を判断する。R1は価格定義に関わる科学判断である。
- 実装と科学検証は、承認後に別の作業として行う。
- 今回、他担当のbranch・worktree・process・結果には介入していない。新しいbranchは`codex/pr-resource-implementation-audit-20261113`（docsのみ）。
- 公開後はmandatory STOP。

## 参考（外部資料）

- python-flint（PyPI／docs）：bindingはMIT、FLINT/ArbはLGPL v2.1+。CPython 3.11以上のLinux wheelがある。https://pypi.org/project/python-flint/ 、https://python-flint.readthedocs.io/en/latest/
- statsmodels `proportion_confint`（beta=Clopper–Pearson、wilson、jeffreys等）：https://www.statsmodels.org/v0.14.4/generated/statsmodels.stats.proportion.proportion_confint.html
- Hypothesis（MPL-2.0）：https://pypi.python.org/pypi/hypothesis
- Qiskit 1.3.0のlevel 1 pass構成、`RZZGate.control`、`add_control`、CCX定義は、研究venvに導入済みのqiskit 1.3.0のSOURCEで確認した（web資料は版が異なるため根拠にしていない）。
