# PR-Pauli 4th(m5_best)・K=2：headとtailのRTEの合成を人工系で確認

想定読者：B（取込み・identity）、B2（候補とparameterの探索）、統合担当。
- 数値の正本：[VERIFICATION.json](VERIFICATION.json)
- テストの出力：[TESTS.txt](TESTS.txt)
- SOURCE：`4e5b8bef6`（branch `codex/pr-r-dependent-n-model-20261111`）

## 冒頭の答え

| 問い | 答え |
|---|---|
| 指定した4次合成を、コードが正しく評価しているか（今回の問い） | **人工系では正しく評価していた。** <br>providerのraw・補正後・exact-tailの信号とlog B、pieceごとの符号付き時間・event数・event時間・τ・bが、独立に書いた参照と一致した。<br>対象は21 round、負の時間のpieceを含む。差は、信号が最大1.5e-12、log Bが6.8e-15、pieceの値が相対2.7e-16以下 |
| 合成公式による近似誤差（別の問い） | 同じ人工系で、決定論的な合成（tailは厳密）の誤差の次数は約4だった（S2は約2）。これは公式の性質であり、上の評価の正しさとは別の結果である |
| 負の時間のpiece | m5_bestの中央のpiece（w=−0.2771）で確認した。<br>headの半時間・tailの時間・event時間・τが負になり、bは\|τ\|で決まる。各roundの負のτについて、回転形式のevent則の平均×bが、K+1次の多項式と1.8e-15以内で一致した |
| 見つけた実装上の不足（修正済み） | ① 正規化子の上限が「1 roundに1つ」で、m5_bestでは足りない（LD3のm5で44個が必要）。<br>② 実step幅T/qがbinary64で正確に表せないround（例：Δ_PF=正確な1/5ではq_PF=3）は、providerが拒否するのに、入口が取得要求にしていた。<br>どちらも入口（`pauli_signal_entry`）を直し、回帰テストを足した |
| 計算 | H4の新しい信号・compile・Q3は0件。人工系だけを使った |

## 既存の検査との照合（追加したのは不足分だけ）

| 確かめたいこと | 既存の検査 | 不足 | 今回の追加 |
|---|---|---|---|
| m5_bestの合成をproviderが評価する | `check_pr_rpe_signal_provider.py` の `test_noncommuting_prefix_signed_K_and_sequential_reference`：K=6だけ、1 qubit。参照はBのweights・配分と `finite_rte_distribution` をそのまま使う | K=2が無い。参照がB・RTEの部品を使い回している。Pauli経路でない | K=2、3 qubit、Pauli経路。weights・配分・多項式・bを、参照側で別に書いた |
| m5_best・K=2 | 同じfileの `test_legacy_finite_RTE_kernel_matches_small_artificial_point`：旧kernelとの比較で、M=1・2×2 | τがごく小さい。負のpieceの値を見ていない | \|τ\|は最大1.66、eventは1 pieceあたり最大512、qは最大16 |
| 負の時間でのevent則の平均 = Taylor多項式 / b | `check_df_head_pauli_tail_contract.py` の `test_event_mean_K0_K2_sign_order_…`：1 qubit、dt=±0.08、K=0・2 | 実際の合成で現れる負のτでは確かめていない | 各roundの中央pieceのτ（−0.003〜−1.66）で、自前の回転形式のevent列挙を3 qubitの辞書に当てた |
| headの順序・配分・符号の誤りを検出できるか | なし | — | わざと誤らせた5種の参照が、すべて検出できることを確かめた |
| 4次であること | `test_h46_new2_core.py`：別の公式（new_2） | m5_bestでは無い | 別の問いとして、次数だけを測った |

## 人工系と参照計算

**系**：3 qubit。headとtailは互いに非可換である（交換子のFrobenius norm 3.06）。
- head：1体項（Z・X）と、10個の2体「fragment」（ランダムなPauli積の和）。L_D=1・2・3。
- tail：符号付きPauli辞書 A = Σ c_l P_l。7項、λ = Σ|c_l| = 3/2（正確）。
- 状態：ランダムな規格化vector。定数0.17、s_R=0.013（一度だけ掛ける）。

**job**：Bの固定版の `generate`/`validate` とP1の `job` で作った。辞書の記述子は合成のPauli用で、providerはH4の実行と同じPauli分岐（辞書・law・identityの照合）を通る。

**参照**（`scripts/verify_pr_pauli_m5_k2_composition.py`）：providerの合成（`_macros`/`evaluate`）も、RTEの部品（`finite_taylor_operator`/`finite_rte_distribution`）も呼ばない。次の内容を明示して書いた。

| 要素 | 参照での書き方 |
|---|---|
| 公式の重み | h46_m5_planのTAILを書き写した。逆順のTAIL＋中央（1−2ΣTAIL）＋TAILの11 pieceで、中央だけが負。Bのweightsとbitまで一致した。ΣW=1、Σw³=−4e−17 |
| pieceの符号付き時間 | 各outer step・各piece jで t_j = T w_j / q |
| eventの整数配分 | 1 outerあたり、各pieceに1つ、残りを\|w\|比で配る。最大剰余は有理数で厳密に計算し、同点は小さいindexを先にする。R=11〜3000で、Bの固定関数と全件一致した |
| headの順序 | exp(−i t_j/2 H_0)、exp(−i t_j/2 H_1)、…、exp(−i t_j/2 H_LD) → tail → 逆順 |
| tail（3列） | exact：exp(−i t_j A)。<br>補正後：P(dt)^c_j。<br>raw：(P(dt)/b)^c_j。<br>P(dt) = Σ_{n≤K+1} (−i dt A)^n/n! は固有値で計算した |
| 正規化子 | b(τ) = Σ_{k偶数≤K} \|τ\|^k/k!·√(1+(\|τ\|/(k+1))²)、τ = λ·dt。log B = q Σ_j c_j log b |
| 進め方 | 状態vectorを、q回のouter stepで逐次に進めた（macro^qは使わない） |
| 負のpieceのevent則 | (k, l_1…l_k, l)のeventを全列挙した。重み \|τ\|^k/k!·√(1+(τ/(k+1))²)·Πp、符号 (−1)^{k/2}Πs、回転角 arctan(τ/(k+1))（τの符号付き） |

## 人工系で確認できた範囲

**round**：5種類の設定で、計21 roundを確かめた。

| L_D・h・Δ_PF・κ | q | R_outer（1 pieceあたりのevent） | 中央pieceのτ | 信号の最大差 |
|---|---|---|---|---|
| 1・1/10・1/10・2 | 1,2,4,8 | 11（1） | −0.042 | 3.3e-14 |
| 2・1/2・1/2・6 | 1〜16 | 11〜54（最大9） | −0.208〜−0.023 | 1.1e-13 |
| 3・1/4・1・12 | 1,1,1,2,4 | 11〜108（最大18） | −0.208〜−0.023 | 5.6e-14 |
| 2・1・4・40 | 1,1,1,2 | 90〜2880（最大512） | −0.028〜−0.003 | 1.5e-12 |
| 2・1・4・1/4 | 1 | 11（1） | −0.42、−0.83、−1.66 | 7.4e-15 |

- 21 roundすべてで、providerの数値gateがPASSした。
- 一致の基準は、信号の差が1e-11以下、pieceの値が相対1e-14以下、pieceのevent数が完全一致であること。全roundで満たした。
- **誤りの検出力**：参照を次の5通りにわざと誤らせると、providerとの差は、そのroundの一致の水準の1.8e5倍以上になった。
  - headの順序を逆にする
  - 負のpieceの符号を落とす
  - 配分を1つずらす（R>11のround）
  - tailをheadの前に置く
  - bをλなしで計算する

  小さいstepではheadの順序の誤りの影響は小さいが（例：4.6e-9）、そのroundの一致の差（8.9e-15）とは明確に区別できた。
- **event則**：負のτ 21件と正の対照1件で、event則の平均×bとK+1次の多項式の差は最大1.8e-15、確率の和は1±1.1e-15だった。providerのrawは、この平均のc_j乗の積と一致している。
- **近似誤差（別の問い）**：L_D=2、T=1.6で、qを1→2→4→8にした。決定論的な合成とexp(−iTH)の差の次数は、m5_bestで5.0・4.1・4.0、S2で2.7・2.2・2.1だった。評価の正しさとは独立の、公式の性質である。H4での大きさは、各roundの `outer_PF_or_discard_bias` に出る。

**正規化子の上限（修正済み）**
- providerは、b(τ,K)を (τ.hex(), K) ごとにcacheし、run全体の上限を超えると `Finite-normalizer scope exhausted` で止める。
- m5_bestでは、1 roundに最大6個の異なるτが現れる。palindromeの対は同じτになり、負の中央が1個加わる。
- 人工系の5 roundで比べた。
  - 上限が「round数」（5）：全roundで停止した。
  - 上限が「piece数」（55）：全roundがPASSし、実際に使ったのは22個だった。
- H4のLD3を4th(m5_best)にしたconfigurationでは、異なるτが44個になる。入口の旧上限は13（round数）なので、途中で止まっていた。
- 入口は、上限をpiece数×round数にした（`E.normalizer_budget`）。S2では従来どおり13のままである。

**binary64で表せない実step幅（修正済み）**
- P1の `job` は、実step幅T/qをfloatにして渡す。providerは、それをBのidentityの有理数と厳密に比べる。
- そのため、T/qがbinary64で正確に表せないround（qが2の冪でない）は、`Foreign B whole-schedule identity` で拒否される。人工系のq=7で確かめた。
- これは公式によらず、S2でも同じである。
- 入口は、このroundを `ACTUAL_STEP_OR_T_NOT_BINARY64_EXACT` として未対応にするようにした。
- **前回のHANDOFFの訂正**：Δ_PF=正確な1/5の人工例でm2–m12を「取得要求」としていたが、実際にはproviderが拒否するroundだった。今は未対応と出る（q_PF=3, 5, …）。
  - binary64の0.2では、qが2の冪になり、取得要求のままである。
  - 入口のVALIDATION.json・PLANS.jsonは、この版で作り直した（[../pauli_signal_entry_20261116/](../pauli_signal_entry_20261116/REPORT.md)）。

## H4の実信号取得に残る条件（B・B2へ）

1. **実行の承認**：4th(m5_best)・K=2のPR-Pauliの13 roundは、新しい科学計算である。入口の `--execute --approve-new-means` は、利用者の承認を得てから使う。
2. **実step幅**：T/qがbinary64で正確に表せる必要がある。h=binary64の0.1の系では、qが2の冪になるΔ_PFに限られる。例：0.1・0.2（binary64）は可、3/10や正確な1/5は不可。
   - これ以外のΔ_PFを使うには、P1の `job` かproviderのidentity照合の変更が要る（Aの作業で、別の承認が必要）。
   - 入口のplanが、該当roundを未対応として示す。
3. **正規化子の上限**：入口では修正済み（piece数×round数）。旧来の分割専用runnerは、m5_bestには使わない。
4. **価格側との一致**：Dのcompile・価格で、信号と同じ構造を使うことを、価格の取得時に照合する。
   - headの順序：1体→fragment 1…L_D→tail→逆順
   - 11 pieceの符号付き時間と、中央の負のpiece
   - eventの配分：`ABS_TIME_PER_OUTER_INTEGER_MIN_ONE_PER_PIECE`
   - 今回は信号側だけを確かめた。
5. **eventの最小数**：m5_bestは、1 outerあたり最低11 event（1 pieceに1つ）である。R=11の下限があるので、λが小さいL_D（例：LD5）では、全roundが下限になる（LD5のm5で全13 roundがR=11）。費用の比較はB2の担当。
6. **H4での数値gate**：providerの数値gate（exact tailのunitarity、headのunitarity、raw・補正後の閉包、rawの範囲、bの精度）は、H4のsector36で初めて評価される。人工系のPASSは、H4でのPASSを意味しない。
7. **検証の区分**：入口の対応区分では、4th(m5_best)・K=2は引き続き `SUPPORTED_NOT_VALIDATED_FOR_PAULI` である。今回の結果は「人工系で評価の正しさを確認済み、H4では未実行」にあたる。K=4・6は今回の対象外。
8. **近似誤差は別に評価する**：m5_bestの外側PFの偏りは、H4では各roundの `outer_PF_or_discard_bias` と、RMSE側の評価で見る。今回の一致は、その大きさについて何も言わない。

## 変更したfile

- 新規：`scripts/verify_pr_pauli_m5_k2_composition.py`（参照と比較）、`tests/check_pr_pauli_m5_k2_composition.py`（6件）。
- 修正：`src/trottertracks/pauli_signal_entry.py`（`normalizer_budget`、binary64のstep照合、planの `request_normalizer_budget`）、`scripts/run_pauli_signal_entry.py`（上限の適用）、`scripts/validate_pauli_signal_entry.py`（Δ_PFの人工例の判定）、`tests/check_pauli_signal_entry.py`（7件）。
- テスト：新しい6件、入口の7件、既存の `check_pr_rpe_signal_provider.py`・`check_df_head_pauli_tail_contract.py`・`check_df_head_pauli_tail_split_signals.py` はすべてPASS。
- 入口の検証も作り直した：保存LD1/2/3/5の再組立てと、Bの `bind_packet` はPASS。

## 範囲と保全

- 人工系とBのmetadataだけを使った。H4の信号・配列の計算、compile、Q3は0件。
- providerとP1のkernel（`e8641b84f`）は変更していない。保存された原結果・packet・旧STOPも変更していない。
- 検査の性質：作成者自身による技術検査で、独立レビューではない。
- mandatory STOP。
