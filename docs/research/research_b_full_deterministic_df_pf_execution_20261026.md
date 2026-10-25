# 研究B B-Next 4：Full deterministic DF-PF 4 caseの科学実行結果

2026-10-26 JST。最終状態：**COMPLETE（固定qの記述的pilot）／MANDATORY STOP**。
科学SOURCE：[8811ef089d588a529aca001f9e40cef1b2f6e1c7][SOURCE]。準備・固定scope：[0e9214e1a版][READY]。
SOURCE変更・技術的中断・修正・retry／resumeは0。原本のruntimeは`COMPLETED_DESCRIPTIVE_MANDATORY_STOP`、scientific_statusは`DESCRIPTIVE_ONLY`。

## 1. 物理条件と実行範囲

Synthetic 2-mode、one-body＋3 DF因子、N=1／mode0 occupation、同じphysical H／state。分子geometry／AO basis／分子DF rank探索は非該当。Full DF-PFはone-body、DF0、DF1、DF2の全4 blockを順に扱い、D/R outer splitting・finite-RTEを使わない。共通入力の結合に旧L_D=2 weight-ranked snapshotを使ったが、Full PFの分割パラメータにはしていない。

FD1/FD2はFull S2、FD3/FD4はFull Yoshida4、各q=2・T=±0.18。Full PFに内部n_Dはない。Complex128、operator spectral 2-norm atol1e-12／rtol0／phase fitなし、ordinary controlled、mode0準備・readout・measurement込みX/Y、native-v5を固定した。

新評価は4 case・8 wrapperだけ。旧B3/B4 finite-r2-K2の4 caseとExact-R E1〜E4は保存セルを参照し、再計算していない。q／n_D探索、新係数、sampling／MC／rare費用、shot数／G、A1／γ、分子・全RPEへの展開、R2-CS再開は行っていない。

## 2. 原本・実行・資源

[原本7ファイル](../../artifacts/research_b_outer_full_df_pf/20261026/full-df-once/)は286,657 bytes。元runから結果branchへbyte一致で複製した。[Result](../../artifacts/research_b_outer_full_df_pf/20261026/full-df-once/result.json)、[binding](../../artifacts/research_b_outer_full_df_pf/20261026/full-df-once/binding.json)、[status](../../artifacts/research_b_outer_full_df_pf/20261026/full-df-once/status.json)、[STOP](../../artifacts/research_b_outer_full_df_pf/20261026/full-df-once/STOP.json)を保全する。

以下を一回実行した。**完了済みで再起動しない。**

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-bnext4-full-df-science-20261026/scripts/run_b2d_full_df_pf.py \
  --execute \
  --source-commit 8811ef089d588a529aca001f9e40cef1b2f6e1c7 \
  --output /tmp/prt-bnext4-full-df-science-20261026/artifacts/research_b_outer_full_df_pf/20261026/full-df-once
```

旧AX-2A/AX-2Bの固定抜粋によるcontrolled Full PF、既存DF system builder、B0 prepared wrapper／compilerを使用した。全blockのDF二乗・Gaussian actual inverse・制御相対位相を既存primitiveで描画する。旧H2/H4/H6のstate／compiler価格は移送していない。

CPython3.11.9、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Qiskit Nature0.7.2、OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0。Native basis rz/sx/x/cx、optimization1、seed17、approximation1、qubits_initially_zero=False、ordinary control、free topology、phase保持native-v5。共有venv・compiler設定は変更していない。

起動直前32 CPU、MemAvailable55,096,840 kB、load約1.06を確認。独立exponential32とblock probe48のcache、既存compile cacheを使用し、compile要求8／actual transpile8。小さい4 caseを最初のFAILで後続compileを起動しない順に処理した。Process poolや人工的なCPU／RAM／worker／wall-time capを追加していない。

[Processログ](../../artifacts/research_b_outer_full_df_pf/20261026/execution_logs/)はwall **8.53秒**、user14.13秒、system3.04秒、CPU201%、最大RSS422,420 KiB、exit0。速度倍率は測定していない。StderrはMatplotlibの一時cache warningのみで、科学処理・保存の中断はなかった。

## 3. Hamiltonian・scalar・独立correctness

One-bodyとDF0/1/2の順序を保ち、全体scalarは**元のc0=0.17だけを一度**付けた。旧分割の`c0+c_ext`やExact-Rのstage補正を混ぜていない。全H恒等式残差は5.551115123125783e-17、旧入力へのstate差は0。Control=0 identity／control=1の全phase、正負stage、prepared X/Yを独立参照へ照合した。

| 検査 | 完了数 | 最大誤差／結果 |
|---|---:|---|
| 全4 block・signed時間・control有無 | 48 operator | 6.95031955626931e-16 |
| Full system／controlled | 8 operator | 3.796714005176106e-15 |
| Raw prepared wrapper | 8 operator | 2.4431213812753793e-15 |
| Compiled prepared wrapper | 8 operator | 2.1208576964133056e-14 |
| Raw／compiled expectation | 16 | Compiled最大1.4210854715202004e-14 |
| 保存physical signal整合 | 1登録closure | PASS。個別残差はこのclosureの出力へ保存していない |

独立参照は同じ元DF入力のFull-Fock CAR行列とexpm、canonical S2／既知Yoshida4の数学積、明示H/Sdg・occupation embeddingである。Nativeの未融合列と非制御system builderの融合列を、ともにこの参照へ検査した。Global phase fittingで差を隠していない。

全224登録境界が予定＝試行＝完了、events448、failed={}、evidence_write_failures=[]。全72 operator／16 expectationと保存整合を先行させ、その後に4 signal結果・8単価を解放した。入力検査は72 operatorのほかに実施している。準備時の人工39件PASSは技術証拠として区別し、SOURCE不変のため不要なsuite再実行はしなかった。

## 4. Full PFのsignal・operator誤差

主指標bは独立数学参照PFのsignalと`<mode0|exp(−iH_phys T)|mode0>`の直接複素絶対差である。Compiled X/Yは別のcorrectness照合を通過したが、compiled signalから新しいb／γを再計算していない。

| Case | T | b | Full-Fock operator誤差 | N1 sector operator誤差 |
|---|---:|---:|---:|---:|
| FD1／S2 | +0.18 | 1.80763525850668e-6 | 4.007241911665479e-6 | 4.0072419116654795e-6 |
| FD2／S2 | −0.18 | 1.80763525850668e-6 | 4.007241911665479e-6 | 4.0072419116654795e-6 |
| FD3／Yoshida4 | +0.18 | 1.947702126906798e-9 | 5.058448211881605e-9 | 5.058448211881605e-9 |
| FD4／Yoshida4 | −0.18 | 1.947702126906798e-9 | 5.058448211881605e-9 | 5.058448211881605e-9 |

| Case | X | Y | 半径 | X直接誤差 | Y直接誤差 | 半径直接誤差 |
|---|---:|---:|---:|---:|---:|---:|
| FD1 | 0.9704564635034172 | −0.22941640289988577 | 0.9972049104748141 | 1.5050620449930108e-7 | 1.8013586844944918e-6 | 2.679520586523765e-7 |
| FD2 | 0.9704564635034172 | +0.22941640289988577 | 0.9972049104748141 | 1.5050620449930108e-7 | 1.8013586844944918e-6 | 2.679520586523765e-7 |
| FD3 | 0.9704563130976367 | −0.2294182023134588 | 0.9972051780771084 | 1.0042400244714145e-10 | 1.9451114607882403e-9 | 3.4976432861100193e-10 |
| FD4 | 0.9704563130976367 | +0.2294182023134588 | 0.9972051780771084 | 1.0042400244714145e-10 | 1.9451114607882403e-9 | 3.4976432861100193e-10 |

全値は[result.json](../../artifacts/research_b_outer_full_df_pf/20261026/full-df-once/result.json)の転記。[SIGNALS.csv](../../artifacts/research_b_outer_full_df_pf/20261026/SIGNALS.csv)も4 rowの直接列抽出で、新しい比・改善率・統計量を計算していない。Normalizationは全case1で、物理signal半径が1という意味ではない。

## 5. Native full-wrapper単価

| Case | Axis | RZ count | CX count | RZ depth | CX depth | Total depth | Circuit size |
|---|---|---:|---:|---:|---:|---:|---:|
| FD1 | X | 406 | 276 | 264 | 276 | 619 | 866 |
| FD1 | Y | 406 | 276 | 264 | 276 | 619 | 866 |
| FD2 | X | 406 | 276 | 264 | 276 | 619 | 866 |
| FD2 | Y | 406 | 276 | 264 | 276 | 619 | 866 |
| FD3 | X | 1186 | 820 | 776 | 820 | 1815 | 2518 |
| FD3 | Y | 1186 | 820 | 776 | 820 | 1815 | 2518 |
| FD4 | X | 1186 | 820 | 776 | 820 | 1815 | 2518 |
| FD4 | Y | 1186 | 820 | 776 | 820 | 1815 | 2518 |

[WRAPPERS.csv](../../artifacts/research_b_outer_full_df_pf/20261026/WRAPPERS.csv)は8 wrapperのmetric／gate／expectationの直接列抽出。全wrapperで準備・ordinary control・readout・measurement1を含む。これは決定論的単価で、shot加重総費用ではない。

## 6. 保存finite-RTE／Exact-Rとの固定点対応

以下は今回の新4 caseと、実行scope内で引用した保存8 caseのセルを並べたもの。各行のbは正負Tで同じ保存値で、決定論的単価はX/Yでも同じだった。新しい費用比や順位統計は算出していない。

| 方法・対応case | T | 物理複素signal誤差b | RZ/CX単価（各prepared axis） | 根拠 |
|---|---|---:|---|---|
| Full DF-S2、FD1/2 | ±0.18 | 1.80763525850668e-6 | 406／276 | 今回 |
| Full DF-Yoshida4、FD3/4 | ±0.18 | 1.947702126906798e-9 | 1186／820 | 今回 |
| Outer Yoshida4／A2／Exact-R、E1/2 | ±0.18 | 5.926752262417758e-10 | 4114／2800 | [保存Exact-R][EXACT] |
| SABA2／A2／Exact-R、E3/4 | ±0.18 | 2.9044769538026594e-9 | 2100／1424 | [保存Exact-R][EXACT] |
| Outer Yoshida4／A2／finite-r2-K2、B3 | ±0.18 | 6.756873465875987e-10 | 本表では期待単価を比較しない | [保存22行][R22] |
| SABA2／A2／finite-r2-K2、B4 | ±0.18 | 2.905057732889878e-9 | 本表では期待単価を比較しない | [保存22行][R22] |

Full Yoshida4とouter-Yoshida4／Exact-Rでは、固定点のnative単価とsignal誤差にtradeoffがある。Full S2は単価と近似誤差の両方を残す別の点である。**q=2でも指数列と仕事量は一致せず、同資源比較・同精度最適化済み比較ではない。**

旧B0代表を期待値にせず、旧B1のS0推定を無条件母平均にせず、rare費用を0補完しない。今回B1 sample再集計・新MC・shot数／Gは行っていない。全方式の最小資源や一般的優位は認定しない。

## 7. 到達点・限界・STOP

既存Full DF-PF builderを研究Bの同じ入力・prepared wrapper・native-v5へ接続し、4 caseの独立correctness・signal・単価を取得できた。対象は2-mode／1電子の特殊なtoyで、多電子・実分子・全RPEへ一般化しない。u_num／γ／shots／Gはnullを維持し、atolをその保証へ流用しない。

科学的実行はCOMPLETE、記述的pilotとして扱う。独立算法の新規性・継続GO、同精度総資源優位、旧shot資格の解消を認定しない。旧SOURCE／raw／契約／grant／ledger／marker／STOPとR2-CS終了は保全した。共通指示・元dirty workspace・共有環境にも変更していない。

結果branchは`codex/research-b-full-df-pf-results-20261026`。新raw7 file・process3 log・直接抽出CSV2件・本報告と必要な研究記録を公開し、SOURCE／remote SHA／公開内容を照合して**mandatory STOP**する。追加科学計算・新q・新PF・分子・shot段階へ自動移行しない。

次の担当は研究Bの**ユーザー**で、固定点結果の解釈と追加比較の必要性を判断する。**Codex**は新しい指示と必要な科学実行許可後だけ次のscopeへ進む。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/8811ef089d588a529aca001f9e40cef1b2f6e1c7
[READY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e9214e1a8f68b8c32ca2b06f69a743311f63e68/docs/research/research_b_full_deterministic_df_pf_readiness_20261026.md
[EXACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58502f5549d350a8da8cbc23a6e674c4670feb9b/docs/research/research_b_bnext3_exact_r_execution_20261026.md
[R22]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/docs/research/research_b_outer_rte_22row_execution_20261025.md
