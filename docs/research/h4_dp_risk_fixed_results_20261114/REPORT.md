# 実H4 PR-LD1/LD2：状態集約DPによる固定2件のMSE上界

2026-11-14 JST。利用者が明示承認した各LD1件・計2評価を完了した。**新しい科学実行はこの2回だけ**。Monte Carlo、追加shot探索、価格/G、AδEの加算は行っていない。結果は独立した新directoryへ保存し、旧input/raw/結果/STOPを保全した。

## 主要結果

同じ新評価内で旧上界、DP上界、min採用上界を計算した。両構成ともgood閉包検査が成立し、部分DPは質量を捨てずに有効な上界を返した。以下は近似表示で、正本は[LD1 exact結果](run01/RESULT_LD1.json)／[LD2 exact結果](run01/RESULT_LD2.json)、[exact分解](DECOMPOSITION.json)。

| 構成 | 旧MSE上界 (Ha²) | DP上界 (Ha²) | 採用上界 (Ha²) | 上界の縮小率 |
|---|---:|---:|---:|---:|
| LD1 | 2.398951760583e-1 | 9.039527981195e-4 | 9.039527981195e-4 | 99.62318842% |
| LD2 | 2.396879588138e-1 | 9.032441317634e-4 | 9.032441317634e-4 | 99.62315832% |

縮小率は`1−U_selected/U_old`。これは同じ条件付きモデルの**上界が約99.62%縮んだ**という結果であり、真のMSE低減率やLD1/LD2の優劣ではない。平方根表示はLD1約0.03006580779 Ha、LD2約0.03005402023 Haで、実RMSEの点推定・無条件保証としない。

## loss寄与と改善を制限する要因

旧式はunsafe全体に最大lossを掛けていた。新DPは解決するunsafe履歴も元decoderで伝播し、終端の実lossを計算する。既知の未解決事象と追跡外残余には元loss契約を維持した。

`U_DP = η + u L_U + r L_bad`、今回L_bad=L_U（約1127.771812556 Ha²）。

| 構成 | 解決終端loss寄与 η (Ha²) | 既知未解決寄与 u L_U (Ha²) | 追跡外寄与 r L_bad (Ha²) |
|---|---:|---:|---:|
| LD1 | 4.723399373426e-7 | 9.034804581821e-4 | 8.161198810382e-24 |
| LD2 | 4.722871596408e-7 | 9.027718446037e-4 | 6.360391053255e-24 |

未解決loss寄与は両構成ともDP上界の約99.9477%。残余寄与は約10⁻²⁴ Ha²であり、**今回の上界では追跡外残余より、現行decoderの未解決事象と採択済みloss familyが支配的**である。保持範囲を増やすことだけではこの既知未解決寄与を消せない。この観測は現在のN・モデル・loss下に限り、他Nでの傾向や最適Nを予測しない。

ηは追跡した解決履歴に条件付ける前の寄与で、条件付きMSEそのものではない。rawモデルには時間発展近似等も含み得るため、純shot誤差とも同定しない。Aの有効Hamiltonian由来δE又はδE²を加算・控除していない。global zeroのlossをu/r寄与へ別途加算することもない。

## 確率・zero/tie/unsafeの区別

下表のuは吸収済み未解決質量、z=P(any zero)、rは正確な追跡外質量。両構成ともu>zなので、最終未解決確率は`[u,u+r]`、zero優先後の実全RPE tie確率は`[u−z,u−z+r]`である。

| 構成 | u（未解決下端） | z（exact zero確率） | u−z（実tie下端） | 区間幅 r |
|---|---:|---:|---:|---:|
| LD1 | 8.011199146170e-7 | 9.998365097859e-8 | 7.011362636384e-7 | 7.236569241686e-27 |
| LD2 | 8.004915839826e-7 | 9.969347644789e-8 | 7.007981075348e-7 | 5.639785444576e-27 |

rが小さいため丸めた上下端は同じ表示になるが、**r>0であり厳密な点確率として扱わない**。実全体tie/unresolvedのexact-point fieldはnullを維持する。区間のexact端点は結果原本にある。元round別reference-oracle tieは全26行でexact0のままだが、実経路上のtieは上表の範囲となり、両者は異なる。

旧診断のany unsafeはLD1約2.1271564e−4、LD2約2.1253190e−4。unsafeは指定phase安全集合外で、実際の誤branchや未解決そのものではない。旧top-levelの未解決範囲も保全し、新しい区間はstate_refinement配下に記録した。

## 状態数・組数・実行負荷

保持上限8192/round、組上限141,559,036/構成。実際は各**86,036,445組**で全13roundを伝播し、組数不足で状態を残余へ移したroundはなかった。保持数のpruneだけが部分追跡の残余を生んだ。m0..2は全解決状態を保持、m3..12は8192を保持した。

| m | count位相群 | 保持状態数（各LD） | 累積遷移組（各LD） | LD1残余質量 | LD2残余質量 |
|---:|---:|---:|---:|---:|---:|
| 0 | 2593 | 2592 | 2593 | 0 | 0 |
| 1 | 2247 | 4467 | 5826817 | 0 | 0 |
| 2 | 1951 | 7757 | 14541934 | 0 | 0 |
| 3 | 1707 | 8192 | 27783133 | 2.477922082317e-62 | 1.437929345339e-63 |
| 4 | 1883 | 8192 | 43208669 | 1.110821135462e-43 | 1.046259724684e-44 |
| 5 | 1612 | 8192 | 56414173 | 4.264516707093e-37 | 1.348112457850e-37 |
| 6 | 964 | 8192 | 64311261 | 4.299394753964e-37 | 1.358514051222e-37 |
| 7 | 777 | 8192 | 70676445 | 9.230434183251e-36 | 3.380027115802e-36 |
| 8 | 584 | 8192 | 75460573 | 1.160959755768e-33 | 4.753209211377e-34 |
| 9 | 550 | 8192 | 79966173 | 2.697221074341e-29 | 1.830390591539e-29 |
| 10 | 383 | 8192 | 83103709 | 8.647030111748e-29 | 6.264032622555e-29 |
| 11 | 261 | 8192 | 85241821 | 7.236566336211e-27 | 5.639783161405e-27 |
| 12 | 97 | 8192 | 86036445 | 7.236569241686e-27 | 5.639785444576e-27 |

各round末で`retained + absorbed_unresolved + residual = 1`をexactに確認した。prune前の瞬間状態数の実測は元APIにないので報告値を作らない。

| 構成 | 評価wall秒 | worker CPU秒 | peak RSS (KiB) | peak RSS (MiB) | 遷移組 |
|---|---:|---:|---:|---:|---:|
| LD1 | 8839.693 | 8840.746 | 734736 | 717.52 | 86036445 |
| LD2 | 8768.844 | 8769.949 | 712392 | 695.70 | 86036445 |

2worker並行の全体wallは8840.282秒、約2時間27分20秒。開始2026-11-14 01:26:19 JST、両完了03:53:39 JST。worker CPUはimport/metadata/保存準備等を含むため、旧caller内のwallとは測定区間が少し異なる。現在は両worker終了済みで、チャットを閉じた間も同じprocessが計算を継続した。自workerだけの[30秒CPU/RSS観測](run01/RESOURCE_MONITOR.jsonl)と各RESULTのpeak値を保全した。

## 固定SOURCE・科学条件・検算

- 数値kernel：`06db9aefec13bb5c5be6d7c0f81f29b3a10b929c`。実行中も終了後もsrc/testsを変更していない。
- caller/INPUTS：`ce0517de6e7b58cf4bf4c69f99f998d86099c406`。[入力](INPUTS.json)／[実行記録](run01/EXECUTION.json)／[完了記録](run01/COMPLETION.json)。旧evaluate_oneを再利用し、D.evaluateを各1回だけ使用。範囲は4139bed3cから継承。
- A raw bundle：`91349e703e0d09ef702ccd108745dfaf1acc29f7`。元configuration a9abf8de、数値SOURCE fefd548e、raw carrier16210932を照合。
- H4 linear 1Å/STO-3G/tol-only DF10/sector(2,2)、保存H/state/snapshot、S2/L_D1又は2/元ΔPF=h=.1/K2/M12・元κ/prior/branch/alias/shift0/P0を保持。
- targetは保存solver energy−2.1663874486347634 Haの条件付きproxy。元targetのproxy=false/ground未資格とmean UNQUALIFIEDを変更しない。
- 原入力43898ada1と同じideal finite-K IID、EXACT_AFFINE_OF_BINARY64_RAW、ρ1、X/Y各軸`64,60,56,52,47,43,38,34,30,25,21,17,12`、BOUNDED_INTERVAL loss familyを保持。
- [VERIFY_RESULTS.py](VERIFY_RESULTS.py)が保存SOURCE・条件・全mean、旧上界/旧round診断のbit/exact一致、全DP質量とloss恒等式、範囲・nullを確認し両方PASS。[CHECKS](CHECKS.txt)。数値riskの再評価・追加試行ではない。

## UNKNOWNと次の判断

数値モデルの旧閉包とDP成立にUNKNOWN理由はない。一方、実全体tie/unresolvedの厳密点値は追跡外r>0のためUNKNOWNで、上記の有限区間のみを出す。保存state、solver targetのground資格、actual sampler law、raw/targetの数値allowanceと実数decoderへの包含は依然未取得。exact確率計算はそれらを認証しない。

今回の上界から正式な十分N/global energy保証を生成せず、元Minkowskiの各予算へ自動接続しない。上界が目標より大きいという理由だけで真のrisk超過・物理的精度未達・候補の不可能性を認定しない。モデル内のloss family上界を何の主張に使うかは、ユーザーのrisk/decoder/loss契約の判断が必要である。

技術的には既存whole-vector adapterへ接続でき、追跡範囲も今回の2件では残余を十分小さく囲んだ。しかし現在の上界の主制約は未解決lossであり、同条件で保持数だけを増やす科学実行は優先しない案を示す。shot数・偶奇・未解決率を変える新vector、decoder/loss変更等は別の判断と有限実行許可が必要。今回それらを採択・実装・評価していない。

1配分・各LD約2.4時間という負荷から、多数whole-vectorの探索を軽量とは扱わない。将来は採択された有限候補に絞り、既存exact会計と結ぶ。正式N/適格価格が揃うまではG/winnerはnull。新しいN qualifier/optimizerを追加せず、A/C/D/監査のworktree/processへ介入しない。

**次：ユーザーがrisk/decoder/lossと次の有限shot評価の要否・候補集合を判断する。Codex Bは新しい明示指示までmandatory STOP。** [HANDOFF](HANDOFF.md)。
