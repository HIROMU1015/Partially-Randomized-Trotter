# H6 Pauli-JW Hybrid PR：大event数への固定4候補検証

2026-10-27。科学SOURCE `887b69e449cb3c378286f09d0d7626599850fb67`、解析SOURCE `af891fd45805454c6592bbee5441266f14235ffd`。
固定linear H6/隣接1.00 Å/STO-3G/中性singlet、保存Pauli918項＋constant、同じraw ray/full-H reference/exact norm。S2/T=.8、central-rotation、full cosine/+Im sine、P=0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。
実行対象はp78/q8/r16,32,64/K2とp684/q1/r16/K4の4物理候補だけ。H2/H4、純RTE追加探索、旧成功cohort再取得、Hamiltonian/state/reference生成は0。

**固定4候補の追加後も、全5精度で保存純RTEの条件付き点Gが低かった。新Hybridは全5精度でprefix78/q8/r32/K2が探索nomineeと確認後minの両方になり、r64では再びGが上昇した。** Hybrid/RTEの保存比は約1.129/1.152/1.183/1.489/2.092。全5行で差は記述的2 combined-SEの近接域外だった。

Hybridのr拡張で旧比較の差は縮小したが、この追加集合では順位逆転しなかった。最優先だった大event数方向の探索機会を補完した結果であり、全prefix/q/r/Kの公平性や大域最適性・無条件の方式優位の証明ではない。旧14/15の保存結果はそのまま維持する。

## 確認標本による条件付き比較

仮想compiled残差ρ=0、有限候補・有限価格標本による比較。GはRZ×shot、nは古典価格標本数で量子shot Nとは異なる。全5精度は同じ価格標本を共有する。新集合の確認後minと探索nomineeを区別する。旧14/15は元集合の結果として保存し、今回の拡張結果で上書きしない。

| ε_sig | 新Hybrid確認min / n | G新Hybrid | 拡張Hybrid最小（旧含む） | G純RTE | H/R | G SE Hybrid / RTE | 記述的判定 |
|---|---|---:|---|---:|---:|---:|---|
| 0.05 | PR_p0078_q8_r32_K2 / 64 | 1.073462e+08 | PR_p0078_q8_r32_K2 | 9.5119e+07 | 1.128547 | 138534.1 / 355807.7 | REFERENCE_LOWER_EMPIRICAL_G |
| 0.01 | PR_p0078_q8_r32_K2 / 64 | 2.740373e+09 | PR_p0078_q8_r32_K2 | 2.377928e+09 | 1.15242 | 3536548 / 8895018 | REFERENCE_LOWER_EMPIRICAL_G |
| 0.005 | PR_p0078_q8_r32_K2 / 64 | 1.125697e+10 | PR_p0078_q8_r32_K2 | 9.511726e+09 | 1.183483 | 1.452752e+07 / 3.558012e+07 | REFERENCE_LOWER_EMPIRICAL_G |
| 0.001 | PR_p0078_q8_r32_K2 / 64 | 3.539825e+11 | PR_p0078_q8_r32_K2 | 2.377965e+11 | 1.488595 | 4.56827e+08 / 8.895154e+08 | REFERENCE_LOWER_EMPIRICAL_G |
| 0.0005 | PR_p0078_q8_r32_K2 / 64 | 1.990372e+12 | PR_p0078_q8_r32_K2 | 9.512028e+11 | 2.092479 | 2.568646e+09 / 3.558125e+09 | REFERENCE_LOWER_EMPIRICAL_G |

純RTEは保存H6_p000_q8_r64_K2・確認n48。旧Hybrid anchorはp78/q8/r8/K2とp684/q1/r8/K4・各n48を再利用した。INCONCLUSIVEは2 combined-SE以内という記述的規則であり、厳密な母平均winnerの信頼保証ではない。

## Signal・normalization・回路価格の内訳

| 候補 | 確認n | B_saved | C_cos / C_sin | ε=.05 N_total | ε=.0005 N_total | 適格性 |
|---|---:|---:|---:|---:|---:|---|
| PR_p0078_q8_r16_K2 | 64 | 1.627257 | 3648.453 / 3648.453 | 37326 | 692113127 | 0.05:CONDITIONAL_GUARD_ELIGIBLE; 0.01:CONDITIONAL_GUARD_ELIGIBLE; 0.005:CONDITIONAL_GUARD_ELIGIBLE; 0.001:CONDITIONAL_GUARD_ELIGIBLE; 0.0005:CONDITIONAL_GUARD_ELIGIBLE |
| PR_p0078_q8_r32_K2 | 64 | 1.276169 | 4675.766 / 4675.766 | 22958 | 425678278 | 0.05:CONDITIONAL_GUARD_ELIGIBLE; 0.01:CONDITIONAL_GUARD_ELIGIBLE; 0.005:CONDITIONAL_GUARD_ELIGIBLE; 0.001:CONDITIONAL_GUARD_ELIGIBLE; 0.0005:CONDITIONAL_GUARD_ELIGIBLE |
| PR_p0078_q8_r64_K2 | 64 | 1.129735 | 6764.547 / 6764.547 | 17992 | 333594110 | 0.05:CONDITIONAL_GUARD_ELIGIBLE; 0.01:CONDITIONAL_GUARD_ELIGIBLE; 0.005:CONDITIONAL_GUARD_ELIGIBLE; 0.001:CONDITIONAL_GUARD_ELIGIBLE; 0.0005:CONDITIONAL_GUARD_ELIGIBLE |
| PR_p0684_q1_r16_K4 | 64 | 1.013431 | 8035.438 / 8035.438 | 21083 | None | 0.05:CONDITIONAL_GUARD_ELIGIBLE; 0.01:BIAS_INELIGIBLE; 0.005:BIAS_INELIGIBLE; 0.001:BIAS_INELIGIBLE; 0.0005:BIAS_INELIGIBLE |

全4候補のlaw/meanは既存tight L2（scalar128bit・binary64 vector中心53bit）を各1作用。Bを一回適用し、raw normを共有、再正規化なし。数値allowance・guard・shot十分条件・任意精度ceilは既存のまま。新science retryや追加precision作用なし。
Bias不適格のN/G=nullと、価格標本が取得済みであることを分ける。全候補が少なくともε=.05で適格なため価格を取得した。全精度・両cohortのbias/u/N/C/G/SEはCANDIDATE_RESULTS.csvへ保存。

## 標本設計と選抜

旧paired価格から分散・共分散を再計算し、新lawの非zero event確率と併せてsampling前にn_Eを固定した。旧SE由来のn要求は3、希少event観測の設計要求を含む共通n_Eは128となった。探索上限128、確認下限64/上限1024はsampling前固定の有限科学budgetで、CPU制限ではない。
p684の非zero eventを90%の名目確率で少なくとも一度見る設計要求は346本で探索上限を超える。観測目標は達成保証とせず、未観測寄与を0としない。
確認対象は適格4候補すべて。探索paired分散、名目t係数、1%半幅目標、旧比較側SE floorからn_Cを決め、確認開始前に候補ごとのn_C・seed/indexを一度だけ固定した。確認結果を見たtop-upは0。探索・確認をpoolしない。
探索nomineeは精度ごとに凍結し、nomineeの確認Gと確認後minをCOMPARISON_RESULTS.csvで別列にした。5条件を独立な5実験と数えない。名目SEは希少eventの母平均tailや選抜後同時coverageの上界ではない。

| ε | 探索前規則からのnominee | nominee確認G | 確認後新集合min | r64境界にminが残るか |
|---|---|---:|---|---|
| 0.05 | PR_p0078_q8_r32_K2 | 1.073462e+08 | PR_p0078_q8_r32_K2 | False |
| 0.01 | PR_p0078_q8_r32_K2 | 2.740373e+09 | PR_p0078_q8_r32_K2 | False |
| 0.005 | PR_p0078_q8_r32_K2 | 1.125697e+10 | PR_p0078_q8_r32_K2 | False |
| 0.001 | PR_p0078_q8_r32_K2 | 3.539825e+11 | PR_p0078_q8_r32_K2 | False |
| 0.0005 | PR_p0078_q8_r32_K2 | 1.990372e+12 | PR_p0078_q8_r32_K2 | False |

## 希少eventと検証の範囲

| 候補/cohort | n | order0 / order2 / order4 event数 |
|---|---:|---|
| PR_p0078_q8_r16_K2/exploration | 128 | 16344 / 40 / 0 |
| PR_p0078_q8_r16_K2/confirmation | 64 | 8181 / 11 / 0 |
| PR_p0078_q8_r32_K2/exploration | 128 | 32754 / 14 / 0 |
| PR_p0078_q8_r32_K2/confirmation | 64 | 16374 / 10 / 0 |
| PR_p0078_q8_r64_K2/exploration | 128 | 65527 / 9 / 0 |
| PR_p0078_q8_r64_K2/confirmation | 64 | 32765 / 3 / 0 |
| PR_p0684_q1_r16_K4/exploration | 128 | 2048 / 0 / 0 |
| PR_p0684_q1_r16_K4/confirmation | 64 | 1024 / 0 / 0 |

K2のorder4はsupport外、K4の未観測order4/componentは費用寄与0の証明ではない。実compiledρ、母平均費用tail、全パラメータ大域最適性、FT/QPE/RPE全体は未保証。q/r/Kを揃えてもprefix変更はtail/law/B/biasを変え、純粋な保持部の因果効果ではない。
新trajectory 768、両軸compile 1536、mean状態作用4。native child 772件のexit0/reapを回収。3072 QPYのcompressed/raw SHAと代表32 compiled QPYのwire/basis/費用を照合。これは全compiler数値証明ではない。
Phase Aの人工/mock27件と、再利用したbody/control/shot/paired処理を使用。科学SOURCE/kernelは保全し、新namespaceだけで実行した。実測free CPUに応じた低優先度workerを使用し、共有設定・他者process/ファイルは変更しない。
科学raw/QPYは既存QPY-aware形式で可逆圧縮し、元bytesのSHA復元を全件検査。原rawは無変更。内部絶対pathを含むoperational logはlocalに保全し、公開版PUBLIC_NATIVE_AUDIT.jsonには終了・回収証拠を保存した。

## データ・終了・次の担当

- CANDIDATE_RESULTS.csv / candidate_results.json：全endpointと会計。
- COMPARISON_RESULTS.csv / comparison_details.json：一次対照、nominee、確認後min、G差・両側SE。
- SOURCE.json / ANALYSIS_SOURCE.json / sampling_design.json / confirmation_plan.json：固定SOURCE・設計。
- verification.json / PUBLIC_NATIVE_AUDIT.json / data_archives.json / archive_restore_verification.json：保存証拠。
科学計算はSTUDY_STOP.jsonで終了。公開・remote照合後mandatory STOP。旧14/15結果は保全。主agentの技術照合であり、詳細科学レビュー・中心claimの正式採択はしていない。
次の担当はユーザー/GPT：この拡張有限集合の条件付き比較を研究上どう扱うか判断する。境界・近接・希少eventの追加検証が必要なら、対象を限定して別承認する。Codexは新候補や詳細レビューへ自動移行しない。
