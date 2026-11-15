# PR-Pauli LD5：configuration 4d6e2c9c の13 round信号

LD5段階1の正本 [HANDOFF_B](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf0a930fd75fb2eb693622edcf3b7204e30f5eff/docs/research/df_head_pauli_tail_ld5_stage1_20261116/HANDOFF_B.md)（`bf0a930fd`）に従い、LD3段階2の固定経路（`8c8ce542e`）をprefix 5へ接続して、m0〜m12の信号を取得した。LD5の価格（較正cell）は、別branchの [a_pauli_ld5_cal_price_20261116](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/codex/a-pauli-ld2-expected-price-plan-20261115/docs/research/a_pauli_ld5_cal_price_20261116) にある。

## 冒頭の答え

| 問い | 答え |
|---|---|
| 取得した信号 | **13/13 round**。すべて `LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED`、局所数値gateはすべてPASS |
| 再利用できた保存信号 | 0件（LD5の13個のphysical IDの保存信号・後続成果・進行中jobはなかった） |
| Bへの引き渡し | [MEAN_HANDOFF.json](MEAN_HANDOFF.json)（commit `ac46d100e`）。LD1〜LD3と同じpacket形式 |
| LD3との比較（信号の値だけ） | 補正後信号の偏り \|corrected − reference\| は全roundでLD3の**1.0020倍**。後半roundのlog Bは0.0388（LD3は0.0411）、attenuationは0.962（LD3は0.960） |

**読み取れること**
- fragment 3・4をheadへ移しても、head側のPF biasはほとんど変わらなかった（0.2%増）。
- 正規化の大きさは、event数が減った分だけやや小さい。
- 1 shotあたりの信号の質は、LD3とほぼ同じである。

**未評価**：Q3の適否（VP、FULL_PF__B_PRICEのどちらも）。Bの次の段階である。

## 1. round別の結果（LD3は保存値）

| m | R_m LD3 | R_m LD5 | \|corr−ref\| LD3 | \|corr−ref\| LD5 | 比 | log B LD3 | log B LD5 | attenuation LD5 | 有限RTE bias LD5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1.3355e-05 | 1.3381e-05 | 1.0019 | 0.00660 | 0.00004 | 0.99996 | 5.0e-13 |
| 1 | 1 | 1 | 2.6711e-05 | 2.6762e-05 | 1.0019 | 0.01319 | 0.00008 | 0.99992 | 1.0e-12 |
| 2 | 1 | 1 | 5.3419e-05 | 5.3521e-05 | 1.0019 | 0.02639 | 0.00015 | 0.99985 | 2.0e-12 |
| 3 | 2 | 1 | 1.0682e-04 | 1.0703e-04 | 1.0020 | 0.02646 | 0.00030 | 0.99970 | 4.0e-12 |
| 4 | 3 | 1 | 2.1357e-04 | 2.1399e-04 | 1.0020 | 0.03530 | 0.00061 | 0.99939 | 8.0e-12 |
| 5 | 6 | 1 | 4.2703e-04 | 4.2788e-04 | 1.0020 | 0.03532 | 0.00121 | 0.99879 | 1.6e-11 |
| 6 | 11 | 1 | 8.5404e-04 | 8.5572e-04 | 1.0020 | 0.03853 | 0.00243 | 0.99758 | 3.2e-11 |
| 7 | 21 | 1 | 1.7081e-03 | 1.7115e-03 | 1.0020 | 0.04036 | 0.00485 | 0.99516 | 6.4e-11 |
| 8 | 42 | 1 | 3.4163e-03 | 3.4230e-03 | 1.0020 | 0.04036 | 0.00970 | 0.99034 | 1.3e-10 |
| 9 | 83 | 1 | 6.8325e-03 | 6.8460e-03 | 1.0020 | 0.04085 | 0.01941 | 0.98078 | 2.6e-10 |
| 10 | 165 | 1 | 1.3665e-02 | 1.3692e-02 | 1.0020 | 0.04110 | 0.03882 | 0.96193 | 5.1e-10 |
| 11 | 330 | 2 | 2.7330e-02 | 2.7383e-02 | 1.0020 | 0.04110 | 0.03882 | 0.96193 | 1.3e-10 |
| 12 | 660 | 4 | 5.4654e-02 | 5.4762e-02 | 1.0020 | 0.04110 | 0.03882 | 0.96192 | 3.2e-11 |

- 全13行の値は [ROUND_SUMMARY.csv](ROUND_SUMMARY.csv) にある。列はLD1〜LD3の全round版と同じ。
- 数値gate（閾値1e−9。referenceの組は1e−10）の最大値：closure 2.3×10⁻¹³、power unitarity 1.6×10⁻¹¹、norm error 8.7×10⁻¹⁴、head unitarity 1.2×10⁻¹⁵、保存reference組の差9.3×10⁻¹²。
- 有限normalizerは3組（R＝1・2・4に対応）。いずれも有限で、b > 1（1.0000379、1.0000095、1.0000024）。
- m0〜m10ではR＝1（1 stepに1 event）のため、log Bはqに比例して増え、m10以降は一定になる。

## 2. 評価の経路と固定した入力

| 要素 | 内容 |
|---|---|
| 経路 | LD3段階2（`8c8ce542e`）のcontext builder（P1 `load_contexts` と照合）、record検査、B向けhelper、要約の列を無変更で再利用した。prefixと固定入力だけを引数にした（`df_head_pauli_tail_split_signals.py`、凍結 `c67469ad4`） |
| job | P1の `job()`（`05447d1d4`）。configuration `4d6e2c9c…`（`39a1fc7df`）から |
| reference | 保存H4 reference（`bba15083`）。新しいreference計算は0 |
| mean | 固定providerの `evaluate`（kernel `e8641b84f`）。sector36 |
| operator | common operatorは `4220f3006`、LD5の `Pauli_nonidentity` は `6255fbc42` の `LD5_OPERATORS.npz` |
| 辞書・λ・scalar | 辞書sha `2947784c…`、λ_P＝0.061570372477616604、scalar＝保存s_R 0.008913337788299013（`KEEP_SAVED_sR_AND_c_ONCE`、総Tで1回） |
| κ・schedule | 段階1で固定したexactな共通κ。R_m＝1×11、2、4 |

**照合**
- 科学callの前に、保存LD2の入力で作ったcontextが、固定P1の `load_contexts` のcontextと完全に一致した。
- LD3のspecでこの経路を通すと、公開済みのLD3 MEAN_HANDOFF（`cdbce383f`）が完全に再現した（テスト）。

## 3. 照合・検証・実行

**§10.5の照合**
- LD5のconfiguration・辞書・13個のphysical ID・cell IDは、段階1の資料以外に現れなかった。関連する稼働中のjobもなかった。
- 13 roundすべてを新しく取得した（上限13）。

**テスト**（作成者自身の技術検査で、独立レビューではない）
- 新テスト4件PASS：LD3 packetの再現、LD5の固定入力とB helper、実行順序と停止、実際の入口（stub評価）。
- 既存テストもPASS：LD3段階2の5件、P1の6件。

**実行**
- 凍結 `c67469ad4` → raw `1e3985e7e`（1回、3.8秒）→ packet `ac46d100e`。
- retry・閾値の変更・fallbackは0。実行環境はプロジェクト専用venv（Python 3.11.0rc1）。

## 4. 行っていないこと

- 新しいpath sampling、回路のbuild・compile、Q3、shot配分：いずれも0件。
- 固定SOURCE・原結果・旧STOPは変更していない。

**mandatory STOP。**
