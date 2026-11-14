# C向け：決定論PF 6候補のm0直接価格

**12/12軸が有効。再利用0、新規実compile12、保留0・欠測0・UNKNOWN0。元作用gateとC guard完了。m0の直接価格だけを引き渡し、mandatory STOP。**

正本は[run01/AXIS_PRICES.json](run01/AXIS_PRICES.json) の12行（CSVも同名）。C alignment `0b4b2d941cd669aaa45231e1236bc2d3158a9d12:docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json` と `(configuration_id, physical_id, axis, m=0)` でjoinする。alignment行selectorも保持した。Full/p7、S2/標準Yoshida4/M5をIDだけでなく元physical descriptorへ結び付け、X/Yを別recordで受け取る。

対象条件はlinear H4 / 1.00 Å / STO-3G / 保存tol-only DF10、Full一体項＋fragment0..9またはB0I p7一体項＋fragment0..6、T=.1 / q1 / actual幅.1、P0除外。測定付きfull controlled Hadamard wrapper、symmetric_directional、Qiskit1.3.0 / rz,sx,x,cx / opt1 / seed17。scalarはFull c_H、p7 c_H＋tail identityをtotal Tに一回。科学INPUT、fragment順、符号付き分解、元native / wrapper / compilerは不変。

## 固定locator

| 種別 | commit | path |
|---|---|---|
| 実行SOURCE | `729fe5370f53cea218f26e32ee262fee10e748ab` | `scripts/run_pf_m0_prices.py`、元Full/p7 caller・高速化helper |
| Full raw | `a7d066caed56a0c7d8e64c617606f99d0fe4e634` | `docs/research/d_full_pf_shortlist_prices_20261114/m0_direct/run01/RESULT.json` |
| p7 raw | 同上 | `docs/research/d_truncated_p7_t3_prices_20261114/m0_direct/run01/RESULT.json` |
| Full bound receipt | `471f0c3fa06bc99fc969eb6a9e291fa15446c76e` | `docs/research/d_full_pf_shortlist_prices_20261114/m0_direct/run01/{PRICES,SIDECARS,GUARD_CHECK}.json` |
| p7 bound receipt | 同上 | `docs/research/d_truncated_p7_t3_prices_20261114/m0_direct/run01/{PRICES,SIDECARS,GUARD_CHECK}.json` |

各行の`trial`/`receipt`/`sidecar`は配列selectorを持つ。sidecarの`implementation_evidence`はraw公開commitのproofへ固定されている。C guard SOURCEは既存 `c628e9e8447c11c984064b922cbce6c52dc72d48:src/trottertracks/price_implementation_guard.py`、policyはimplementation-onlyの `PF_M0_DIRECT_IMPLEMENTATION_ONLY_V1`。lowering/nativeは `H46_LONG_TIME_NATIVE_COMPOSE`、RTEは `NOT_APPLICABLE_NO_RTE`。compiler hashだけで実装同一性を認定していない。

価格recordは `DIRECT_COMPILED_DETERMINISTIC` / `DETERMINISTIC`、`EXACT_DETERMINISTIC_CIRCUIT_PRICE`。`diagnostic.model_qualified=false` を維持する。決定論回路なのでlaw平均のMC誤差を導入しない。一方、別actual幅の凍結価格モデルや未観測roundの価格を認定する材料として自動転用しない。

REPORTの6行表は見やすさのためX/Yを並べたもの。Cは12軸JSONを使う。各軸で全6指標、異なるwrapper fingerprint、元作用gate、SOURCE・scalar binding、元trialとguard proofを参照できる。m9の予測・2%規則、モデル判定は今回使用していない。

## 資源・ロードマップへの差分

取得wall100.876 s（報告・公開は別）、Full99.056 s / p7 71.543 sを並列実行、6 cell worker、最大worker RSS575.16 MiB。開始時idle CPU64 / available RAM約488.52 GiBに応じた配置である。各route contextとworker exit codeは全て0、signalなし、scientific retry0、cache補完0。段階別時間とresourceの正本は[run01/TIMING.json](run01/TIMING.json)。旧m9とq/prefixが異なるため、このwallから高速化倍率を主張しない。

統合担当へ：**m0のdirect coverageが12軸増え、6候補すべてのm0 X/Yが埋まった。モデル検証の追加PASS数は0（未実施）。** Cの156軸表へこの12行をoverlayする。旧m3～m6の保存証拠とm9結果はそのまま保持する。m3の既存consumer binding未完了という別課題も今回自動完了にしない。m1/m2、m7/m8、m10～m12を新たに資格付与せず、Nとの積算やG/winnerへ進まない。ロードマップ本体の編集は統合担当に任せる。

新callerは新規namespace専用でありresume / retry入口ではない。今回のdirectory/cache/QPY/progress/STOPを保存する。別roundや同run再取得には別途明示承認が必要。**commit/push・remote照合後mandatory STOP。**
