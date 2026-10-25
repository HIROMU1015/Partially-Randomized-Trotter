# 研究B B-Next 3：Exact-R 4 caseの限定科学実行結果

2026-10-26 JST。最終状態：**COMPLETE（記述的pilot）／MANDATORY STOP**。
科学SOURCEは[e76c0474806dca42598fa92a6564cded10896f98][SOURCE]。準備・契約は[7056267ed版][READY]。
SOURCE変更、技術的中断、修正、retry／resumeはなかった。原本のruntime statusは`COMPLETED_DESCRIPTIVE_MANDATORY_STOP`、scientific_statusは`DESCRIPTIVE_ONLY`である。

## 1. 固定条件と今回の実行範囲

Synthetic 2-mode、one-body＋3 DF因子、N=1／mode0 occupation、L_D=2 weight-ranked prefix。Dはone-body＋因子0,1、残差Rは因子2のidentity分離後。分子geometry／AO basis／分子DF rank探索は非該当。D内部A2 Yoshida4、n_D=1、q=2、T=±0.18、complex128、operator spectral 2-norm atol1e-12／rtol0／phase fitなし。

E1/E2はouter Yoshida4、E3/E4はSABA2、各正負T。新しい4 Exact-R caseだけを実行し、対応するB3/B4 finite-r2-K2の4 caseは[既存22行][R22]から参照した。Finite-RTEの再評価、追加sampling、shot/G、rare費用、A1／γ、full DF-PF、n_D探索、分子展開、R2-CS再開は行っていない。

Exact-Rは単一DF因子のFull-Fock指数をGaussian往復＋DF二乗対角primitiveで実装する。Rのidentity補正`+c_ext*t_R`と全体scalar-onceを保持し、RTE normalizationは1。D内部PFとouter splittingは近似のままである。

## 2. 原本・実行コマンド・環境

原本：[exact-r-once/](../../artifacts/research_b_outer_exact_r/20261026/exact-r-once/)。7 raw file、234,019 bytes。結果branchへbyte一致で複製し、元runは上書きしていない。
[Result](../../artifacts/research_b_outer_exact_r/20261026/exact-r-once/result.json)、[binding](../../artifacts/research_b_outer_exact_r/20261026/exact-r-once/binding.json)、[status](../../artifacts/research_b_outer_exact_r/20261026/exact-r-once/status.json)、[STOP](../../artifacts/research_b_outer_exact_r/20261026/exact-r-once/STOP.json)。

実行したコマンドは以下。**完了済みで再起動しない。**

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-bnext3-exact-r-science-20261026/scripts/run_b2d_exact_r.py \
  --execute \
  --source-commit e76c0474806dca42598fa92a6564cded10896f98 \
  --output /tmp/prt-bnext3-exact-r-science-20261026/artifacts/research_b_outer_exact_r/20261026/exact-r-once
```

CPython3.11.9、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Qiskit Nature0.7.2、OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0。既存venv・DF／B0／wrapper／compilerを使用した。Compilerはnative-v5、rz/sx/x/cx、optimization1、seed17、approximation1、qubits_initially_zero=False、ordinary control、free topology。Mode0準備と測定を含む。

開始直前32 CPU、MemAvailable55,121,812 kB、load約1.21を確認した。既存84 D block／12 tail／50 exponential cacheを共有し、compile要求8／actual transpile8。最初のFAILで後続compileを起動しないcase順を保持した。追加のprocess poolは使用せず、CPU／RAM／worker／wall-timeへの人工上限や共有環境変更はない。

[Process記録](../../artifacts/research_b_outer_exact_r/20261026/execution_logs/)ではwall **14.34秒**、user20.02秒、system3.02秒、CPU160%、最大RSS398,732 KiB、exit0。速度倍率は測定していない。StderrはMatplotlibが書込不可の既定cacheから一時cacheへ移したwarningのみで、科学処理・保存の中断はなかった。

## 3. 入力・phase・独立correctness

既存snapshotのHamiltonian／sector／state／split identityと一致した。Fragment2のλ=0.22、抽出identity=0.05127099999999999、全体scalar=0.221271。H恒等式残差1.1102230246251565e-16、R恒等式残差5.785335646522553e-17、state差0。Full-Fock DR commutator normは0.004280429999999991で非zeroだった。

| 検査 | 完了 | 最大誤差・実績 |
|---|---:|---|
| Signed R単独／controlled R | 12 operator | 2.4706208321691707e-16 |
| Full system／controlled | 8 operator | 3.740761669321236e-14 |
| Raw prepared X/Y wrapper | 8 operator | 1.25554640322064e-14 |
| Compiled prepared X/Y wrapper | 8 operator | 7.106330112796073e-14 |
| Raw／compiled expectation | 16 | Compiled最大3.68594044175552e-14 |
| 既存ideal_tail／physical signalとの整合 | 4 case、各2照合 | Ideal-tail最大1.8529835144581194e-15、physical差0 |

Control=0 identity／control=1の相対位相を含む明示8×8参照に対し、global phase fitなしで通過した。独立参照は元g／λのCAR行列とSciPy expm、未融合の数学的PF積、明示H/Sdg／occupation embeddingを使い、Gaussian回路を自身の参照にしていない。

全257登録境界が予定＝試行＝完了、failed={}、events514、evidence_write_failures=[]。全36 operator／16 expectationと保存signal整合が完了した後に4 signal結果・8費用を解放した。入力検査はこの36 operatorのほかに実施している。技術準備時の人工30件PASSは過去の技術的証拠として区別し、SOURCE不変なので不要なsuite再実行はしなかった。

## 4. Physical operator／複素coherent signal

主指標bは、独立したExact-R＋近似D/outerの数学参照と、同じH/stateの`exp(−iHT)`との直接複素signal絶対誤差。Compiled expectationは別に上のcorrectnessで照合した。Compiled X/Yから新しいbiasやγ certificateを再計算していない。

| Case | T | Exact-Rのb | 保存finite-RTE correctedのb | Exact-R Full-Fock operator誤差 | Exact-R N1 operator誤差 |
|---|---:|---:|---:|---:|---:|
| E1／Yoshida4 | +0.18 | 5.926752262417758e-10 | 6.756873465875987e-10 | 1.4882069553427571e-9 | 1.4882069553427563e-9 |
| E2／Yoshida4 | −0.18 | 5.926752262417758e-10 | 6.756873465875987e-10 | 1.4882069553427571e-9 | 1.4882069553427563e-9 |
| E3／SABA2 | +0.18 | 2.9044769538026594e-9 | 2.905057732889878e-9 | 5.966194040900029e-9 | 5.9661940409000265e-9 |
| E4／SABA2 | −0.18 | 2.9044769538026594e-9 | 2.905057732889878e-9 | 5.966194040900029e-9 | 5.9661940409000265e-9 |

| Case | X | Y | 半径 | X直接誤差 | Y直接誤差 | 半径直接誤差 |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 0.9704563129624344 | −0.2294182048502242 | 0.9972051785291439 | 3.477829135789534e-11 | 5.91653948056603e-10 | 1.0227119151551278e-10 |
| E2 | 0.9704563129624344 | +0.2294182048502242 | 0.9972051785291439 | 3.477829135789534e-11 | 5.91653948056603e-10 | 1.0227119151551278e-10 |
| E3 | 0.9704563132793785 | −0.2294182013678318 | 0.9972051780364232 | 2.8216584624374264e-10 | 2.890738454164321e-9 | 3.904495615714154e-10 |
| E4 | 0.9704563132793785 | +0.2294182013678318 | 0.9972051780364232 | 2.8216584624374264e-10 | 2.890738454164321e-9 | 3.904495615714154e-10 |

全数値は[result.json](../../artifacts/research_b_outer_exact_r/20261026/exact-r-once/result.json)のセルを転記した。[SIGNALS.csv](../../artifacts/research_b_outer_exact_r/20261026/SIGNALS.csv)も4 rowの直接列抽出だけで、新しい比・改善率・統計量を計算していない。

**この固定範囲では、RTEを除去してもD／outerに由来するfull誤差が残る。** 新回路のsignalは旧ideal-tailと固定atol内で整合し、Full-Fock primitiveによるその対照の実装が成立した。Finite-RTE correctedの点値との違いは上表の範囲で記述し、独立した数値不確かさ包含や全方式の精度順位へ格上げしない。Normalization1は物理signal半径1を意味しない。

## 5. Native full-wrapper単価

| Case | Axis | RZ count | CX count | RZ depth | CX depth | Total depth | Circuit size |
|---|---|---:|---:|---:|---:|---:|---:|
| E1 | X | 4114 | 2800 | 2666 | 2800 | 6231 | 8674 |
| E1 | Y | 4114 | 2800 | 2666 | 2800 | 6231 | 8674 |
| E2 | X | 4114 | 2800 | 2666 | 2800 | 6231 | 8674 |
| E2 | Y | 4114 | 2800 | 2666 | 2800 | 6231 | 8674 |
| E3 | X | 2100 | 1424 | 1360 | 1424 | 3177 | 4428 |
| E3 | Y | 2100 | 1424 | 1360 | 1424 | 3177 | 4428 |
| E4 | X | 2100 | 1424 | 1360 | 1424 | 3177 | 4428 |
| E4 | Y | 2100 | 1424 | 1360 | 1424 | 3177 | 4428 |

[WRAPPERS.csv](../../artifacts/research_b_outer_exact_r/20261026/WRAPPERS.csv)は8 wrapperのmetric／gate／expectationの直接列抽出。全wrapperがmode0準備・ordinary control・readout・measurement1を含む。

これは**決定論的Exact-R回路の単価**である。旧B0代表は期待値ではなく、B1の既存推定はS0条件付きでrare費用が未観測。今回B1標本を再集計せず、finite-RTE期待費用との比・母平均順位・同精度最小費用・shot加重Gは新たに評価していない。Exact-RがPRより総資源で優位という結論は得ていない。

## 6. 限界・判断に残ること

4次元Full-Fockの実装検査とnative単価を取得できたが、物理モデルは2-mode／1電子・単一DF残差の特殊なものに限られる。多電子・多数tail・実分子や全RPEへの移送は未検証。

Atol1e-12は実装correctnessの受理値で、u_num／γ上界ではない。両者とshots／Gはnullを維持した。誤差が小さいことや単価を測れたことから、保証付き適格性・総資源優位・独立算法的新規性・研究継続の正式GOを認定しない。旧shot評価のINCONCLUSIVEやR2-CS終了は変更しない。

ユーザーが許可した技術復旧例外は今回使用しなかった。科学的FAILを技術的不具合と解釈した修正、許容差／precision変更、追加attemptは0である。

## 7. 保全・公開・STOP

結果branchは`codex/research-b-bnext3-exact-r-results-20261026`。新報告、7 raw file・3 process log・2直接抽出CSVと、必要な研究記録／索引を公開する。旧原本・契約・SOURCE・grant／ledger／marker／STOP、共通指示、元dirty workspace、共有環境には変更しない。本run後に追加科学計算を開始していない。

Scope／保存／表／SOURCE・remote内容を照合して**mandatory STOP**。次の担当は研究Bの**ユーザー**で、この限定結果の解釈や追加比較の必要性を判断する。**Codex**は新しい指示と必要な科学実行許可がある場合だけ、別scopeを準備・実行する。Full DF-PF／n_D調整／shot/G／追加MC／詳細レビューへ自動移行しない。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e76c0474806dca42598fa92a6564cded10896f98
[READY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7056267ed6e49b573353c815a89bbba7da9ae3fa/docs/research/research_b_bnext3_exact_r_execution_readiness_20261026.md
[R22]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/docs/research/research_b_outer_rte_22row_execution_20261025.md
