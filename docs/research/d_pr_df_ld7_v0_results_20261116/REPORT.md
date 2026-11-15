# PR-DF LD7：configuration・13信号・共通V0 Q3（2026-11-16 JST）

**LD7は指定V0で適合した。** 全13 roundで元mapperの出力がR=1・S2単一pieceとなり、13信号の数値gateを通過した。共通V0でのMSE上下界はLD7が `[4.28414160318294721301353E-7, 4.28414160318294721365383E-7] Ha²`。次の判断は **LD7固有価格の較正と候補別配分の比較**。価格取得・独自A/Bは実行せず、公開後mandatory STOPとする。

今回新規はconfiguration **1件**、信号 **13 round**、Q3 **1件**。sampling・回路build・compile・fit・価格予測・配分生成は各 **0件**。LD1/LD2のV0 Q3 **2件**とFull-H reference pair **13件**は保存成果を再利用した。

## 同一V0での比較

X/Yとも `[64,60,56,52,47,43,38,34,30,25,21,17,12]`。候補ごとの信号を使い、X/Yをpoolしていない。

| L_D | MSE lower (Ha²) | MSE upper (Ha²) | 判定 | 取得 |
|---:|---:|---:|---|---|
| 1 | 4.72462588934653801951643E-7 | 4.72462588934653960729534E-7 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | 保存V0再利用 |
| 2 | 4.72409731948051054908399E-7 | 4.72409731948051194818804E-7 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | 保存V0再利用 |
| 7 | 4.28414160318294721301353E-7 | 4.28414160318294721365383E-7 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | 新規Q3一回 |

exact値は[SUMMARY.json](SUMMARY.json)、表示表は[V0_COMPARISON.csv](V0_COMPARISON.csv)。LD1/LD2は `8ec7c882806809460d38a638a891bafcf18fafd1` のSUMMARY.menu[0]/[3]を再利用し、原Q3 locatorを[REUSED_V0.json](REUSED_V0.json)に残した。LD7の原結果は `8ebf3c982` の `q3/LD7_V0/RESULT.json`、API_RETURNと一致する。

元のtotal decoder `consecutive_dyadic_nearest_branch_principal_energy_total_v1`、保存Full-H solver条件付きproxy `-2.1663874486347634 Ha`、全事象のsquared-energy loss、exact閾値τ `15136964828156312001 / 5960464477539062500000000 Ha²` を維持した。数値gate不成立・UNKNOWNや上界だけでは判定できない状態との取り違えはない。今回のLD7は閉包検査とtotal DPを完了した上での適合である。

## 条件・configurationと信号

保存H4 linear 1.00 Å / STO-3G / tol-only DF10、元H/state/snapshot、元fragment順。PR-DF LD7のone-bodyは別の決定論headでL_Dに数えず、DF fragment 0..6をhead、7..9をtailにした。S2・Δ=.1・K2、元のpaper κを維持した。対象lawは `ideal_finite_RTE_law`。

- configuration ID：`ce9c7f08234089f0b50fd786c20403fe02c5de9073e89d6d35a12ee9d966c5e1`
- 固定configuration：[generation/CONFIGURATION.json](generation/CONFIGURATION.json)（公開 `12efc9dca`）。元B/A mapperによる各physical ID・exact T・q/R・allocationは[SCHEDULE_GATE.json](generation/SCHEDULE_GATE.json)と[SCHEDULE.csv](generation/SCHEDULE.csv)。m0..12のTは.1..409.6、q=2^m、全R=1、signed_weights=[1]、allocation=[1]。
- 保存tail descriptor：λ_R=`0.0003129539143720037`、s_R=`4.5302064712666724e-05`。s_Rはtail identityで、Hamiltonian定数cとは別の出所である。
- c=`2.29310124732` は固定snapshotのconstant memberと元receiptから読取。c+s_R=`2.293146549384713`（hex `0x1.2585d37d4573ap+1`）。入力hash・array記録・各operandのlocatorを[SCALAR.json](generation/SCALAR.json)に保存し、実providerがロードした値とも信号取得前に一致確認した。

信号は変更しない `fefd548ecd56f60d72031b3d385f6f5e6ee95d06:src/trottertracks/pr_rpe_signal_provider.py` のdirect-expm head / paired finite-Taylor / matrix-power経路で取得。総scalarを総Tで一回だけ適用し、raw/correctedを独立伝播する。旧信号をLD7へ読み替えていない。保存reference pairは `bba150834ca30bdb186fbfd58c55a8d51177b855:.../PHYSICAL_INPUTS.json` のround別locatorを使い、新しいreference actionは0件。

全13 roundのraw・corrected・exact-tail・B/log_B/attenuation、物理ID、元referenceとSOURCEの対応は[ROUND_INDEX.json](ROUND_INDEX.json)、詳細gate/allowanceは[signals/m0.json](signals/m0.json)〜m12.json。信号raw公開は `3ce3e5dde`。各roundはX/Yで共有される複素信号であり、26本の独立pathではない。共通のmacro/head cacheを利用し、最初のmacro計算後12 roundは既存cache hitだった。巨大なevent列の生成やtrajectory samplingは行っていない。

## SOURCE固定・照合・実行確認

全ての40桁commit/pathは[SOURCE_LOCATORS.json](SOURCE_LOCATORS.json)。生成・信号caller SOURCEは `04e37b3f7`、生成入力は `42c4c6308`、信号入力は `009365d5c`。Q3 caller SOURCEは `1cbc24cfc`、Q3入力は科学call前に `dcc9c93a1` で固定した。実行codeと結果公開commitを区別する。

§10.5の計画時調査に対し、fetch後の公開成果・関連job・SOURCE・出力先の差分を確認した。同等の現行PR-DF LD7成果/jobは確認範囲で未同定。新physical IDを旧signal archiveとも照合し、13件とも一致なし。PR-Pauli LD3と決定論p7の並行作業は別候補・別lawであり採用しない。[DELTA_CHECK.json](DELTA_CHECK.json)、[PHYSICAL_DELTA_CHECK.json](PHYSICAL_DELTA_CHECK.json)、[Q3_DELTA_CHECK.json](Q3_DELTA_CHECK.json)を参照。

生成前の最初のSOURCE guardは、後日のPauli変更を含む71424761 viewを拒否した。この時点でB.generate未到達、configuration/信号/Q3は0件だった。停止log・元入力を[technical_preflight](technical_preflight/NOTE.md)に保全し、fefdそのものの別viewに修正してSOURCEを再固定した。期待hashの変更・guard無効化はない。Q3準備時の短縮commit引数もmetadata入口で拒否され、40桁参照で通過した。科学callのretryは0件。

人工検証は[TESTS.txt](TESTS.txt)。非可換なtoy head/tail・scalar一回適用、実worker起動→実Accelerator constructor→子process→toy評価→保存を確認した。実Q3もPython **3.11.0rc1**、元kernel `710aa69b...` とaccelerator `9e294778...`、元AST hashのまま実行した。count-only上限137,923,114、実DP transition 94,048,712、契約cap141,559,036。閉包は `VERIFIED_FINITE_BINARY64_SUPPORT`。API_RETURN・RESULT・1 call・exitcode・13 identity/law/scalar対応の保存結果照合は[CHECKS.json](CHECKS.json)。主担当自身の照合であり、別担当の独立レビューではない。

## 資源と限界

| 段階 | wall | CPU / peak RSS | 終了 |
|---|---:|---|---|
| configuration生成・検査 | 3.448 s | user 4.484 s / system 5.662 s、43,576 KiB | 0 |
| 保存入力restore＋13信号 | 5.123 s | user 15.667 s / system 6.843 s、358,736 KiB | 0 |
| LD7 Q3 worker・保存 | 231.652 s | worker CPU 219.438 s、child CPU 200.679 s、worker peak 787,280 KiB | 0 |

Q3のprocess-tree sampled peak RSSは5,160,232 KiB、単一完了childのpeakは568,284 KiB（tree合計ではない）。親supervisor wall231.720 s、親から見た全Q3 entry wall233.845 s。詳細は各PROCESS.json、[RESOURCE_FINAL.json](q3/LD7_V0/RESOURCE_FINAL.json)、[COMPLETION.json](q3/COMPLETION.json)、段階/round別の協調logに保存。数値gate失敗0、欠測0、UNKNOWN0、保留0。

今回の結論は、保存H/state・数値信号・ideal finite-RTE rawのIID測定仮定・指定decoder/target/lossの下で、**この共通V0にLD7が適合する**ことまで。ground state証明、信号の厳密数値誤差保証、実PCG64への移送、厳密E[C]/CI/G/winner、全L_Dの最適性・単調性は認定しない。費用優劣は今回評価していない。

次の有限判断とロードマップ差分は[HANDOFF.md](HANDOFF.md)。LD7固有価格→価格モデル固定→独立価格holdout→候補別配分/Q3という計画の後半は未実行で、別指示を要する。元raw・凍結価格・m4独立比較・過去STOPは変更していない。
