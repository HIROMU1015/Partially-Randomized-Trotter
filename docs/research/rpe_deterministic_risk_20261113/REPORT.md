# MCを使わないRPE risk・whole-shot評価基盤

2026-11-13 JST。基準B `c26d09ad8f23a47bedba22580177507fa5dcf840`、リソース評価系列の既存branch `codex/rpe-decoder-statistics-20261113`を再利用した。利用者方針に従い、主経路は二項分布の決定論的評価とする。旧MC・risk interval資産はそのまま保全した。今回は技術実装・人工検証だけで、実H4への適用、保存signal/S3の数値再解析、実RMSE/必要N/C/G計算は行っていない。人工MCも実行していない。

## 成果と再利用

新規は[rpe_deterministic_risk.py](../../../src/trottertracks/rpe_deterministic_risk.py)だけである。既存K.prepareのevaluation_target・signal binding・canonical/whole-vector検査、S.count_mean、R3のwrap/最近branch処理、W.compareとBのexact cost/selectを再利用する。元caller/decoder/statistics/risk_intervals/会計器は変更していない。

| 機能 | 実装 |
|---|---|
| round別二項分布 | 共通分母と整数分子による厳密漸化式。tail切捨て・float確率への変換なし |
| count→位相 | 元exact整数差→binary64比→math.atan2とR3を使用 |
| zero/tie/branch | roundのzeroとphase安全集合を列挙。tie/branchは前状態依存なので、条件付きtransition APIと有限support検査で扱う |
| 全RPE risk | 安全集合内の遷移を実decoderで検査し、最終round寄与＋初回unsafe事象のloss上界をexact rationalで合成 |
| whole-shot選択 | 外部から指定された配分・model MSE閾値・資格付きN/価格を既存W.compareへ接続。資格packetは作らない |

小さな全count列挙との一致/上界関係、zero/tie、誤branch、alias、underflow領域の非零確率、13roundとX/Yの区別、巨大整数N、人工価格でのG最小化を確認した。**新13＋MCを呼ばない回帰32＝45技術テストPASS**。[TESTS](TESTS.txt)に対象を明記した。primary agent自身の技術検査であり、独立した科学レビューではない。

## decoder条件と保証の範囲

元SOURCEは`pr_rpe_rmse.decode_supplied_coordinates`、`rpe_four_round_phase_validation.reconstruct_rpe_phase`、`rpe_decoder_fast`。測定位相はatan2、最近branchは円周距離最小、scalarの完全tieは最小indexだが、wrapperは距離差≤1e−12を未解決とする。全round中のzeroはtieより優先し、priorでclip・alias修正しない。

Claudeの一律π/3を実装保証として無条件採用していない。実数の理想則では`2ρ_(m−1)+ρ_m<π`が十分条件になるが、tie許容幅とbinary64境界の確認が別に必要である。今回、全good前状態×全good現在位相の隣接組を元R3で検査し、targetに最も近いlattice oracleへ遷移して未解決にならないことを帰納的に確認する。成立しない/指定検査範囲外ならriskはUNKNOWN。詳細な定義・証明は[HANDOFF](HANDOFF.md)。

上界は**与えたraw/IID count lawと現binary64 decoderに条件付くモデル上界**である。主値境界を跨ぐenergy lossも実直線上で含める。π/T型driftは精密な上界として認定していないため使わず、decoderの全出力主値範囲と外部の未解決loss上界で保守的に囲む。未解決energyはNoneのまま、特定の補完推定器は作らない。

確率とMSE和の算術はexact rationalであり、MC誤差やconfidenceはない。一方、raw数値誤差、target/state/実samplerへの資格、実数の理想位相則とbinary64実装の差を認証した意味ではない。元Minkowskiや直接総risk契約を新たに採択せず、科学的十分N/global boundはnull、energy qualificationはUNKNOWNを保持する。

## 効率と残る制約

round列挙はO((N_X+1)(N_Y+1))。oracle処理はNumPy batch、遷移検査は重複位相/前状態を除き、前状態方向をvector化する。全roundのcount直積は列挙しない。ただし有限隣接検査はO(Σ |V_(m−1)| |Φ_m|)であり、Claude提案のround列挙だけより重くなり得る。大きなN族で常に秒単位とは主張しない。count/pairの明示有限範囲外はUNKNOWNで、MCへfallbackしない。これらは外部指定の検査範囲でありCPU/RAM/worker capではない。

厳密な分母・分子は大きくなる。将来の保存時もfloat化せず、Python 3.11の巨大整数JSON桁数制限と容量を扱う必要がある。今回、共有環境やserializerの設定は変更していない。より軽い実装上の一様margin証明やdrift改良が必要かは、許可された有限適用で判断する。

## SOURCE・次の段階・STOP

Claude資料は利用者指定の`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/Claude_PRリソース評価_RPE総コスト_統合科学レビュー_20261113.md`の§5/§7.3/A7–A8を確認した。SHA256=`59002677698fb3304565dc822d1085fb63cd47f18b996fc20e6b340f2d69e648`。提案を自動採択せず、原資料を変更・転載していない。2026-11-09版研究共通運用指示を適用した。

ユーザーがtarget/loss/model claim/全配分・phase安全集合・有限範囲を採択し、科学実行を明示許可した後にのみ、BがLD1/LD2を評価する。[次の有限計画](NEXT_SCIENCE_PLAN.md)に不足条件を記載した。C/Dは元bindingと資格付きcompiled価格を供給し、既存会計gateを維持する。他担当のworktree・科学実行には介入していない。

commit/push・remote SHA/主要成果物照合後、**mandatory STOP**。新しい科学実行・実N/C/G最適化へ進まない。
