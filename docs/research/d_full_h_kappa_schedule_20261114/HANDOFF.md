# D → B/C/A：κ型Full-Hの接続と未承認計画

[REPORT](REPORT.md)／[R448確認](R448_CONFIGURATION.md)／[接続](KAPPA_CONNECTION.md)／[有限計画](FINITE_SCIENCE_PLAN.md)／[候補metadata](KAPPA_CANDIDATES.json)。各担当のbranch・worktree・実行中processは変更していない。

## 共通

- 追加したのは `src/trottertracks/full_h_kappa_layout.py` だけ（metadata専用）。`kappa_descriptor(value, proposal_source=...)` を `D.full_h_rte_spec(kappa=...)` に渡し、`layouts(spec, B=api.B)` の結果を `C.configure(..., layouts=...)` に渡す。生成後は `check_configuration(config, B=api.B, rows=rows)` を通す。
- κ候補のconfiguration・canonical/physical IDは、すべて [KAPPA_CANDIDATES.json](KAPPA_CANDIDATES.json) の値を使う（`BUILD_PLAN.py` で再生成可能）。表示用の小数からlayoutを作り直さない。κはいずれも `scientifically_adopted=false`。
- R448（`cfa6aee3…`）は対照として保持する。κ候補とphysical IDは共有しないので、signal・logB・価格・Nを移さない。

## B担当へ

- κ候補を評価するときは、既存 `I.signal_input` packetをそのまま受ける。`law_binding.kappa`、`event_mapping.kind=D_FULL_H_EXPLICIT_PHYSICAL_LAYOUT_V1` に加え、configurationにはκのspecが含まれる。
- B `select` では、Full-Hの明示layout family（R448）とκ供給family（κ1/2/4）が別familyになる。κ1/2/4は同じfamily内のinner κ候補になる。
- RMSE policyはR448・PRと同じA S0 policy（minkowski／`finite_B_diagnostic`）のまま生成した。`paper_kappa_envelope`（logB=1/κ）に切り替える場合は別identityになるので、比較familyの全methodで揃える。
- κ2のlogB包絡(λT)²/sは0.4913〜0.5で全roundほぼ一定になり（finite logBはその下）、R448のlogB 40のようなm12での爆発は起きない。ただしdecoder risk・十分N・finite-K biasの資格は未評価で、ここから優劣を結論しない。

## C担当へ

- 数値2 moduleとcallerの様式は変更なしで使える（S-1）。callerにはscope差替えと追加gate（ID一致、0<logB≤(λT)²/s＋one-ULP感度）が必要。人工τ gridでは、binary64のbがexp(τ²)の丸め値を1 ULP超える例があったので、余裕を付けずに比較すると誤ってSTOPする。
- `price_implementation_guard` は変更なしでκ layoutを解決する（人工testでPASS）。新ZZ Full-H recordには既存sidecar契約の各項目を付ける。
- 長時間価格はgroup分解（a＋b·s＋c·(q−1)、単一groupはR≤36,223）を推奨候補として渡す。単一group cellをcalibration recordとして受ける形式、n、精度、holdoutの凍結手順はCのmodel scopeで決める。短い回路の価格をqやRに比例させる式を検証済みとしない。

## A担当へ

- Full PFの候補やH_eff解析には変更・依存がない。κ定義の転記（`510fbacd`）とB/Aの総数式の一致を利用した（人工testで一致）。

## ユーザーへ

承認が必要な事項は [REPORT](REPORT.md) §8 にまとめた。現時点はmandatory STOPで、S-1、S-2、P-1、P-2のどれも開始していない。
