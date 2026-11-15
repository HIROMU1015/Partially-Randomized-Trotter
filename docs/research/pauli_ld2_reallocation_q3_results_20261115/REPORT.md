# PR-Pauli LD2：B_PRICE による2配分のQ3結果（FPN-B・OWN-B）

2026-11-15 JST、B2（Claude Code、主agentのみ。独立レビューではない）。

ユーザーが固定計画 `15c08ff29` のうち FPN-B と OWN-B を採択した範囲で実行した。**新規Q3は2件（各1回）。retry 0件、UNKNOWN 0件。** 両workerが原API return・partials・RESULT・資源記録を自分で保存し、callerも終了code 0で完了した。

## 結果

対象：PR-Pauli LD2 の固定候補（`d7dd4f52…`、S2・幅0.1・K2・元κ・Pauli辞書 `dd986e52…`）。τ = εE² ≈ 2.5395612850638132×10⁻⁶ Ha²（十進 εE=0.00159360010199040 のexact二乗）。

| 案 | 実配分 N_X = N_Y（m0〜m12） | MSE下界 Ha² | MSE上界 Ha² | 上界/τ | 判定 |
|---|---|---:|---:|---:|---|
| **FPN-B** | [48,45,42,39,36,33,29,26,23,19,**7,1,1**] | 3.5109473453885294224×10⁻⁵ | 3.5109473453885294438×10⁻⁵ | 約13.8 | **モデル不適合**（下界>τ） |
| **OWN-B** | [48,45,42,39,36,33,29,26,23,19,16,13,**5**] | 1.0542467660657135051×10⁻⁶ | 1.0542467660657850990×10⁻⁶ | 約0.415 | **適合** |
| （参照）VP | [48,45,42,39,36,33,29,26,23,19,16,13,9] | 6.2075×10⁻⁷ | 6.2075×10⁻⁷ | 約0.244 | 適合（`50be68b8`、再評価していない） |

- 表示は近似。判定は原APIのexact有理数で行った。正本は `run01/<案>/RESULT.json`（`record.diagnostic.deterministic_risk`）と [SUMMARY.json](SUMMARY.json)。
- 両案とも、有効上界は total状態集約DPで、bias²はFPN-B 約6.45×10⁻⁸・OWN-B 約6.26×10⁻⁹、Varは約3.504×10⁻⁵・約1.048×10⁻⁶。どちらも分散が支配的である。
- **FPN-Bの不適合は、評価したこの1列についての結論である。** 同じ予算の別の列、P7N-B、中間予算、A_SHAPE（OWN-A）は評価していない。

## モデル費用 T(N) とPF 2基準との差

価格はB2のモデル参照価格 C_ref（`a7eceecb`、凍結モデルの診断的適用、E[C]ではない）。比較には予算上限ではなく、実際のモデル費用 T(N) を使う。PFの基準は、BのPF6費用予算Q3で確認された最小適合費用（`b4296b110`）で、PF側の60軸も条件付き閉包価格である。

| 案 | 予算 | T(N) | うち m0〜m9 | うち m10〜m12（外挿） | 未使用予算 |
|---|---:|---:|---:|---:|---:|
| FPN-B | 615,304,502 | **600,045,823.85** | 180,706,976.08 | 419,338,847.77（69.9%） | 15,258,678.15 |
| OWN-B | 2,533,877,262.39…（T_refの3/4、丸め前） | **2,442,845,584.51** | 180,706,976.08 | 2,262,138,608.42（92.6%） | 91,031,677.89 |

exact値：FPN-B の T(N) = 54210221285613812/90343469、OWN-B = 220695144335642870/90343469。

| 案 | 基準 | D = T_PF − T(N) | m10〜m12の価格が一様に何%変われば費用関係が反転するか |
|---|---|---:|---:|
| FPN-B | Full PF S2（615,304,502） | **+15,258,678.15**（LD2の方が低い） | **+3.64%**（m10〜m12の価格が3.64%高ければ同額） |
| FPN-B | B0I p7 S2（435,481,090） | **−164,564,733.85**（LD2の方が高い） | **−39.24%**（m10〜m12の価格が39.24%低ければ同額） |
| OWN-B | Full PF S2（615,304,502） | **−1,827,541,082.51** | **−80.79%** |
| OWN-B | B0I p7 S2（435,481,090） | **−2,007,364,494.51** | **−88.74%** |

- 反転の割合 x は、m10〜m12 の全6軸のモデル参照価格を一律に (1+x) 倍したときに T(N) が T_PF と等しくなる値、x = (T_PF − T(N)) / T_{m10–12}(N) である。**感度の目安**であり、価格の誤差区間ではない。m0〜m9 の価格とPF側の価格は動かしていない。
- **FPN-Bの費用はFull PFの最小より低いが、FPN-BはQ3でモデル不適合**なので、この費用差を「適合した配分での比較」には使えない。
- **OWN-Bは適合した配分だが、費用はPFの最小の約4.0倍（Full PF）〜5.6倍（p7）である。** PFより低い費用にするには、m10〜m12 の価格がモデル値より約81%（Full PF）〜89%（p7）低い必要がある。
- 価格の根拠の範囲：m0〜m9 の価格もモデル参照値である。S0+（`2d3aa21f5`）が直接確認した長い回路は m9 の2 path（q=512）であり、m0〜m9 の全軸を監査したわけではない。m10〜m12（q≥1024）は観測の外側の外挿である。

## 分かったこと

- LD2では、m12 を9→5 shot、配分全体をVPからOWN-Bへ変えても、採択閾値を満たした（MSE上界はVPの約1.7倍、τの約42%）。モデル費用は T_ref（3,378,503,016.52）から約27.7%下がった。
- m10〜m12 を 7・1・1 shot まで削った FPN-B は、上界がτの約14倍で不適合だった。PFでも、m12 を1 shotまで削った M5・B_PRICE が不適合だった（`b4296b110`）。PFとPauliは信号が違うので直接の比較ではないが、同じ傾向である。
- 今回の2配分の範囲では、LD2で採択閾値を満たした最小のモデル費用は OWN-B の 2,442,845,584.51 で、PFの最小費用（435,481,090／615,304,502）には届いていない。これは評価した有限の配分についての結果であり、LD2の配分全体での最小やPFとの優劣を確定したものではない。

## 入力と実行

| 項目 | 内容 |
|---|---|
| 実行前の固定 | 配分・予算・T(N)・未使用予算・INPUTS・callerを `e094b2d60c5c4468711dea244fa2f91fb587ac12` でcommit・pushしてから実行（[SCOPE_AND_PREFLIGHT](SCOPE_AND_PREFLIGHT.md)） |
| 生成規則 | 重み VP/C_ref、各軸 1≤N≤VP、Bの breakpoint 規則（`06f4210e` の`generate`を正の有理数価格に広げただけ）。X/Yは別に生成し、同じ列になった |
| 信号・decoder・target・loss・閾値 | `f54b41a2e`の固定LD2 caseと採択済み評価条件をそのまま再利用（Z1＋T1 total decoder、保存DF10 solver proxy −2.1663874486347634 Ha、全事象平方loss、ideal IID・EXACT_AFFINE） |
| Q3経路 | `scripts/run_pauli_q3_three_quarter.py`（`f54b41a2e`と同一bytes）の worker・assess。closure gate・lemma gate・total DP gateを含む固定Pauli route |
| 新しい部分 | workerによる結果保存（`scripts/run_pauli_ld2_reallocation_q3_fixed.py`）。人工test 7件＋再利用経路の既存3件がPASS（[TESTS.txt](TESTS.txt)） |
| 閉包・範囲 | 両callで有限binary64閉包 `VERIFIED_FINITE_BINARY64_SUPPORT`（検査組 904,074／911,461）。非終端frontier 8192以下、遷移組 50,035,468／52,604,760（上限141,559,036以内）。追跡外残余の確率 約7.1×10⁻²⁴／3.9×10⁻²¹、残余loss上界 約2.1×10⁻²²／7.2×10⁻²⁰ Ha²（捨てずに会計） |
| 資源 | 2 job並列、各8 DP worker。全体wall 161.65秒。FPN-B wall 130.6秒・process-tree CPU 220.6秒、OWN-B wall 161.6秒・CPU 258.4秒。定期観測のprocess-tree RSS 約4.0 GB／4.1 GB（厳密なpeakではない） |
| 重複照合 | 既存のLD2 Q3はVP（適合）とV0（閾値判定欄のない診断）で、2配分はどちらとも、また互いにも一致しない。後続の保存結果・実行中jobも無かった（SCOPE_AND_PREFLIGHT） |

**資格の範囲**：保存ideal raw・IID二項・solver proxyに条件付いたモデル判定であり、十分N・最適N・物理的global RMSE・正式E[C]・正式G・winnerではない。ground/state/target包含・実sampler移送・数値certificateは未取得。transition allowanceは元のengineering allowanceである。AのδEは加算していない。

## 成果物

- [SUMMARY.json](SUMMARY.json)／[SUMMARY.csv](SUMMARY.csv)：2案の配分・判定・MSE上下界・T(N)とその内訳・PF 2基準との差・反転の割合（[summarize.py](summarize.py)が保存結果だけから作成）
- [INPUTS.json](INPUTS.json)：実行前に固定した配分・予算・各軸のN×C_ref・元価格行への参照
- `run01/<案>/`：API_RETURN、RESULT、PARTIAL_COARSE／LEMMA／TOTAL_DP、STAGES、ACCEL_PROGRESS、RESOURCE_FINAL、PERSISTED、SUPERVISOR_RECEIPT
- [HANDOFF](HANDOFF.md)：統合担当向け

固定SOURCE・VP/V0の原結果・旧STOPは変更していない。**mandatory STOP**：P7N-B・中間予算・OWN-A・別の列・retryへは進まない。
