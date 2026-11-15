# PR-Pauli LD3 段階2：configuration dd5b881d の13 round信号

段階1の正本 [HANDOFF_B](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e77c37989489bc562651b5f1be11411f6b71f92c/docs/research/df_head_pauli_tail_ld3_stage1_20261116/HANDOFF_B.md)（`e77c37989`）に従い、LD3のm0〜m12の信号を取得した。

## 冒頭の答え

| 問い | 答え |
|---|---|
| 取得した信号 | **13/13 round**。すべて `LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED`、局所数値gateはすべてPASS |
| 再利用できた保存信号 | 0件（13個のphysical IDの保存信号・後続成果・進行中jobはなかった） |
| Bへの引き渡し | [MEAN_HANDOFF.json](MEAN_HANDOFF.json)（commit `cdbce383f`）。LD1・LD2と同じpacket形式 |
| LD2との比較（信号の値だけ） | 補正後信号の偏り \|corrected − reference\| は、全roundでLD2の**0.9986倍**。正規化 log B（後半0.0411）とattenuation（0.960）もLD2とほぼ同じ |

**読み取れること・まだ言えないこと**
- fragment 2をheadへ移しても、head側のPF bias（S2、Δ＝0.1）は増えなかった。全roundでLD2より0.14%小さい。段階1で懸念した「headのbiasが増えてshotが増える」方向の兆候は、信号の上では見えない。
- 正規化の大きさも、共通κのためLD2とほぼ同じである。したがって、1 shotあたりの信号の質はLD2とほぼ同じで、tail event数だけが約0.157倍になった、という状態である。
- ただし、Q3で適合するか、必要なshot数がLD2と同じかは、まだ評価していない（BのQ3が必要）。headの費用とLD3のevent単価も未評価である。

## 1. round別の結果（LD2は保存値）

| m | R_m LD2 | R_m LD3 | \|corr−ref\| LD2 | \|corr−ref\| LD3 | 比 | log B LD2 | log B LD3 | attenuation LD3 | 有限RTE bias LD3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 1 | 1.3374e-05 | 1.3355e-05 | 0.9986 | 0.02099 | 0.00660 | 0.99343 | 2.0e-08 |
| 1 | 3 | 1 | 2.6747e-05 | 2.6711e-05 | 0.9986 | 0.02808 | 0.01319 | 0.98689 | 3.9e-08 |
| 2 | 5 | 1 | 5.3492e-05 | 5.3419e-05 | 0.9986 | 0.03375 | 0.02639 | 0.97396 | 7.9e-08 |
| 3 | 9 | 2 | 1.0697e-04 | 1.0682e-04 | 0.9986 | 0.03753 | 0.02646 | 0.97388 | 2.0e-08 |
| 4 | 17 | 3 | 2.1387e-04 | 2.1357e-04 | 0.9986 | 0.03975 | 0.03530 | 0.96531 | 1.2e-08 |
| 5 | 33 | 6 | 4.2764e-04 | 4.2703e-04 | 0.9986 | 0.04095 | 0.03532 | 0.96530 | 2.9e-09 |
| 6 | 66 | 11 | 8.5526e-04 | 8.5404e-04 | 0.9986 | 0.04095 | 0.03853 | 0.96220 | 9.5e-10 |
| 7 | 132 | 21 | 1.7106e-03 | 1.7081e-03 | 0.9986 | 0.04095 | 0.04036 | 0.96044 | 2.7e-10 |
| 8 | 263 | 42 | 3.4212e-03 | 3.4163e-03 | 0.9986 | 0.04111 | 0.04036 | 0.96044 | 6.8e-11 |
| 9 | 526 | 83 | 6.8423e-03 | 6.8325e-03 | 0.9986 | 0.04111 | 0.04085 | 0.95997 | 1.4e-11 |
| 10 | 1,052 | 165 | 1.3685e-02 | 1.3665e-02 | 0.9986 | 0.04111 | 0.04110 | 0.95973 | 1.5e-12 |
| 11 | 2,103 | 330 | 2.7369e-02 | 2.7330e-02 | 0.9986 | 0.04113 | 0.04110 | 0.95973 | 1.1e-11 |
| 12 | 4,205 | 660 | 5.4732e-02 | 5.4654e-02 | 0.9986 | 0.04114 | 0.04110 | 0.95973 | 1.9e-11 |

- \|corr−ref\| は、有限RTEで補正した信号と保存H4 referenceとの差である。ほぼすべてがS2の決定論的部分の誤差（`outer_PF_or_discard_bias`）で、Tにほぼ比例して増える。
- 有限RTE bias（corrected − exact_tail）は、全roundで8×10⁻⁸以下だった。
- 全13行の値（raw・corrected・exact_tail・reference・log B・B・attenuation・closure・empirical u など）は [ROUND_SUMMARY.csv](ROUND_SUMMARY.csv) にある。列の並びはLD1・LD2の全round版と同じ。

**数値gate**（閾値1e−9。referenceの組は1e−10）

| 項目 | 13 roundの最大値 |
|---|---|
| raw–corrected closure | 1.06×10⁻¹⁰ |
| power unitarity | 1.13×10⁻¹¹ |
| norm error | 4.8×10⁻¹⁴ |
| head unitarity | 1.1×10⁻¹⁵ |
| 保存reference組の差 | 9.3×10⁻¹² |

- 有限normalizerは11組（m0〜m2はR＝1で同じτを共有）。いずれも有限で、b > 1 である。
- empirical u は m12で1.56×10⁻⁷（LD2のm12は9.80×10⁻⁷）。

## 2. 評価の経路と固定した入力

既存のP1全round経路（`7084fbc9a` のcompletionと同じ組み立て）をそのまま使った。

| 要素 | 内容 |
|---|---|
| job | P1の `job()`（`05447d1d4`）。configuration `dd5b881d…`（`421258d40`）から |
| reference | 保存H4 reference（`bba15083` PHYSICAL_INPUTS の各roundの reference_pair）。新しいreference計算は0 |
| mean | 固定providerの `evaluate`（kernel `e8641b84f`、`DIRECT_EXPM_HEAD_PAIRED_TAYLOR_MATRIX_POWER_V1`）。sector36 |
| context | P1 `load_contexts` と同じ組み立て。common operatorは `4220f3006`、LD3の `Pauli_nonidentity` は `50640b74c` の `LD3_OPERATORS.npz` |
| 辞書・λ・scalar | 辞書sha `be4ee9cd…`、λ_P＝0.8137790393725127、scalar＝保存s_R 0.09261516589219046（`KEEP_SAVED_sR_AND_c_ONCE`、総Tで一回） |
| κ・schedule | 段階1で固定したexactな共通κと13 roundのR_m（1, 1, 1, 2, 3, 6, 11, 21, 42, 83, 165, 330, 660） |

**contextの同一性の確認**
- 科学callの前に、同じ組み立てで保存LD2の入力からcontextを作った。それを固定P1の `load_contexts` が作るLD2 sector36 contextと比べ、全arrayとdescriptorが完全に一致した。
- LD3 contextの辞書identityも、configurationの辞書identityと一致した。

## 3. 既存成果との照合（§10.5）

- 照合した範囲：2026-11-16に更新されたremote branchの全file、Track Aの保存signal archive（`155529e2`、11,220件）、稼働中のprocess。
- 結果：13個のphysical IDは、段階1の資料以外に現れなかった。archiveにもprefix 3で同じq・Rの記録はなかった。関連する稼働中のjobもなかった。
- そのため、13 roundすべてを新しく取得した（上限13のとおり）。
- なお、physical IDは辞書の種類を含まない。仮に旧PR-DFの記録が同じIDを持っていても、lawが異なるので同等な信号ではない。

## 4. 実装・検証・実行

- 新しいfileは3つ（凍結commit `8c8ce542e`）。既存のprovider、P0・P1、completion、段階1のmoduleは、byte単位で変えていない。
  - [`src/trottertracks/df_head_pauli_tail_ld3_signals.py`](../../../src/trottertracks/df_head_pauli_tail_ld3_signals.py)
    - context builder
    - 実行順序の制御（m0→m12、最初のFAILで停止）
    - packetの組み立て（B intakeの条件を事前に確認）
    - B向けhelper `b_intake_dictionary_identity`
  - [`scripts/run_df_head_pauli_tail_ld3_signals.py`](../../../scripts/run_df_head_pauli_tail_ld3_signals.py)：`--execute` と `--assemble`
  - [`tests/check_df_head_pauli_tail_ld3_signals.py`](../../../tests/check_df_head_pauli_tail_ld3_signals.py)
- テスト（作成者自身の技術検査で、独立レビューではない）
  - 新テスト：5件PASS。
    - 256次元の人工配列で、builderとP1 `load_contexts` の一致を確認した。
    - 人工の13 roundを固定providerで通し、packetがB intakeの条件を満たすことを確認した。
    - 最初のFAIL、またはidentity不一致で停止することを確認した。
    - 実際の入口 `execute`・`assemble` を人工入力で通した。
    - 保存arrayは読み込んでいない（npzはすべて人工のものに差し替えた）。
  - 既存テスト：P0 5件、P1 6件、completion 6件、段階1 9件、すべてPASS。
- 実行：凍結 `8c8ce542e` → raw `26bac4f32`（1回、3.75秒） → packet `cdbce383f`。
  - retry・閾値の変更・fallback・再正規化は0。
  - 実行環境はP0・P1と同じプロジェクト専用venv（Python 3.11.0rc1、numpy 1.26.4、scipy 1.14.1）。

## 5. 行っていないこと・保全

- 新しいpath sampling、回路のbuild・compile、価格計算、Q3、N、shot配分は、いずれも0件。
- 旧SOURCE、原結果、旧STOPは変更していない。ロードマップ本体も編集していない（影響は [HANDOFF_B](HANDOFF_B.md) に記載）。
- 新しい性能手法は導入していない（13 meanで3.75秒）。

**mandatory STOP。**
