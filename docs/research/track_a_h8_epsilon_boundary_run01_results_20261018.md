# H8：保存候補集合での追加6精度の費用境界検証（run01）

2026-10-18 JST。利用者の**「実行準備ができたらそのまま実行してよい」**を受け、[並列検証計画](track_a_targeted_parallel_validation_plan_20261018.md)のC_EPS_H8だけを実装・人工検証・SOURCE封印後に一回実行した。**正常終了・mandatory STOP**。他のA/B/D/E/Fの科学計算は起動していない。

SOURCE `3156b036fb4a0015e09cce2cb73785d7819dcb75`。[結果bundle](../../artifacts/resource_applicability/track_a_h8_epsilon_boundary_run01/2026-10-18/README.md)、[12比較のexact値](../../artifacts/resource_applicability/track_a_h8_epsilon_boundary_run01/2026-10-18/comparisons.json)、[CSV](../../artifacts/resource_applicability/track_a_h8_epsilon_boundary_run01/2026-10-18/comparison_summary.csv)。初期のUNAPPROVED計画・INVALID grantは履歴として保持し、新しい承認を独立したC専用grantに結合した。有効grant原本・消費済みmarkerはprivateに保持し、公開物は消費済みreceiptとhashだけ。

## 結果

PR削減率は`100*(1-G_PR/G_det)`。正はPRの点費用が低く、負は保存決定論的対照の点費用が低い。各εで保存候補から別々に選び、候補を増やしていない。

| ε | 通常bias：PR削減率 | 経験的非相殺bias：PR削減率 |
|---:|---:|---:|
| .04 | +17.077851% | +24.164284% |
| .025 | +23.426602% | +24.244531% |
| .015 | +22.745930% | +24.448547% |
| .000875 | +19.082889% | +34.544599% |
| .00075 | +12.769532% | +26.574804% |
| .000625 | −0.497174% | +11.283196% |

通常biasでは、保存済みε=.05の−8.0230%と追加ε=.04の+17.077851%の間、および追加ε=.00075の+12.769532%と.000625の−0.497174%の間で、端点の符号が変わった。整数shotと対照の切替を含むため、連続関数の一意な根や厳密な逆転閾値を求めた結果とはしない。非相殺biasでは今回の6点は全て正だが、保存ε=.0005では−29.0910%である。今回範囲の全精度でPRが優位という主張はしない。

通常biasのPR点最良は緩い3点で`B2_p7_q2_2nd_r4_K2`、厳しい3点で`B2_p7_q8_2nd_r4_K4`。決定論点最良はε=.04が実測`B0I_p11_q1_2nd`、.025/.015がモデル`B0_p15_q2_2nd`、.000875/.00075がモデル`B0_p15_q8_2nd`、.000625が実測`B0I_p15_q4_4th`。**.000625の約−0.50%は、保存n32による有限標本の点比較**であり、PR母平均の逆転を証明するものではない。

非相殺biasの決定論点最良は緩い3点がモデル`B0I_p15_q2_2nd`、厳しい3点が実測`B0I_p15_q4_4th`。PR点最良は通常biasと同じ。モデル単価の残る点を実測済みと表示しない。

## 条件・入力・検証

Linear H8/1.00Å/STO-3G、保存tol-only DF rank28、16 spin modes/sector4900、T=.8、P=0、full Hadamard-wrapper RZ、経験的allowance倍率1、ε/√2、α_axis=.025、u/headroom≤.01。決定論強化184とPR確認union6/各n32を固定し、通常biasと非相殺biasを分離した。未知s4・複数2次の費用を補完しない。PR母平均・厳密bias・winnerの統計保証は付けない。

- 既存22-role snapshotを既存manifestのbytes/SHAで検証して読取専用で使用した。原signed成分の368軸、PR成分・端点・normalization・n32 pairing/seedの検査を既存関数で継承した。
- [入力manifest](../../references/h8_epsilon_boundary_inputs_v1.json)の公開Git3 fileは、6軸実測反映後の全会計行と正常最終gate、保存非相殺会計行。単価は保存行から取り、モデルの新しい予測・fitは0。通常/非相殺の保存単価はRZと実測flagの一致を検査し、元のラベルを残す。
- 新εの計算前に、ε=.05/.01/.001/.0005×190候補×2区分＝**1,520行のshot・総費用を完全再現**した。適格なPR端点のpaired SEも再現した。これは整合性gateであり、旧科学stageの再実行や新samplingではない。
- 新規は6精度×190候補×2区分＝**2,280会計行・12比較**。凍結D0 shot式をそのまま呼び、bias区分とε以外の定義を変えていない。欠測・UNDETERMINEDはnullとして保持し、適格な費用の欠測はSTOPする。
- 新規の人工8件と既存の人工156件、計164件PASS。paired shot重み・exact Fraction・n32拒否・欠測価格の0補完拒否・未解決domain・未承認/異scope拒否を検査した。author自身の検証で、独立レビューではない。

## 実行・高速化・保全

Python3.12.3、stdlibのみ。保存標本と単価は一度だけ読み、全εで再利用した。2,280行の軽いscalar計算ではprocess poolを増やさず、単一processで処理した。内側の分子計算・compile・samplingは0。新たな汎用高速化基盤や共有環境の変更も0。

2026-10-18 03:53:10 JST正常終了。SOURCE/入力/runtime fresh gateと終了時gateはPASS。driver記録wall **0.832秒**、外側native wait込み **0.962秒**、native exit0、wait完了。peak RAMはこのrunでは測定していない。過去のworker/RAM上限は持ち込まず、fresh affinityと空きメモリを観測記録した。

実行前のruntime profile作成でbuiltin mathの扱いを修正したが、その時点はgrant発行前・科学起動0だった。修正後の固定SOURCEで科学dispatchは1回、科学retryは0。元の全STOP・SOURCE・grant・ledger・marker、V1/V2 NO_GO、H5/H7、他worktreeは変更していない。完了済みRARE_SENSITIVITY・SELECTION_SAVEDは再実行していない。

## 次の作業

- **Codex**：今回のmandatory STOP後、次の作業ターンでB（希少層native接続）・D（直接実測scope）の実装/人工接続テスト/control予算を優先して完成する。A/E/Fも準備できた単位から同じ方針で進める。
- **Codex**：利用者の今回の「準備できたら実行」の承認は保持する。既定範囲の各単位について新SOURCE・入力・新grant・fresh gateを結合して実行できるようにし、再確認のためだけに承認を求め直さない。各単位のmandatory STOPと自動retry禁止は維持する。
- **ユーザー**：GitHub公開を行う場合、認証可能な環境から最終commitをpushする。現在この環境のHTTPS pushは認証不足で失敗しており、remote SHA一致は未確認。
