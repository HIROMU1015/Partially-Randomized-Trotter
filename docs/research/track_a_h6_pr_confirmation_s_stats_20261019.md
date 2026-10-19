# H6 PR独立確認：限定標本数設計 S_STATS の結果

2026-10-19 JST。利用者の「その方針で進めてよい」を、固定2候補の保存値による限定数値設計1回への承認として実施した。
状態は**COMPLETE_MANDATORY_STOP**。保存scalar passは1回、native exit0・親によるreap・SOURCE/input/runtime最終SHAはPASS。
**推奨nはε=.05で48、ε=.0005で104。最終nの採択・実行seal・sampling/compileは別段階で、今回実行していない。**

[保存計算結果と監査](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_s_stats/2026-10-19/README.md)。
数値設計SOURCEは`4ebccfeda760473b63765816f34ef68df9cbfaa4`。
Phase 0 commit `6d290fc8fb656f4db148e2c9f91b757d80ed83c2`と、そのSOURCE・契約・STOPは保持した。

## 条件と固定入力

linear H6、隣接1.00 Å、STO-3G、tol-only1e-8 DF rank19、T=.8、P=0。
元の二次PR、p4/r4/K2、無条件finite-RTE law、symmetric-directional full Hadamard wrapperを継承する。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed17の保存compiled RZと保存shotによる通常signed/factor1の費用Gを対象とする。
今回、新しい回路・shot・signal・preparationは生成していない。

|ε|PR候補|直接実測PF対照|sign(PF G−PR mean G)の期待方向|
|---|---|---|---:|
|.05|B2_p4_q2_2nd_r4_K2|B0I_p7_q1_2nd|−1：PR不利|
|.0005|B2_p4_q8_2nd_r4_K2|B0I_p11_q4_4th|+1：PR有利|

入力はPhase 0の固定plan（SHA `cef97c6da9a5e38d174a21d93f15912c3fba32b3a999f978a834e02a67a74f2d`）だけ。
その中の保存n8 mean・paired SE・固定PF G・候補固有K2 lawを使用し、raw trajectory費用の分散を再集計していない。
同一trajectoryのcos/sinを組にした保存SEを使い、軸の独立性を仮定して合成していない。

## 正規近似の感度シナリオ

family α=.05、2比較の各片側α=.025、計画power=.8、SD倍率f=1/2/4。
`s8=SE8*sqrt(8)`、`Δ=abs(PF_G−mean8)`、
`n_normal=ceil(((z_.975+z_.8)*f*s8/Δ)^2)`を評価し、計画nの下限を2とした。
αは古典的回路費用の計画用で、量子shot側のα・誤差配分は変更していない。

|ε|保存n8 mean G（exact）|保存paired SE|固定PF G（exact）|f=1の計画n|f=2|f=4|
|---|---:|---:|---:|---:|---:|---:|
|.05|5037196305/8|8425392.116288794|356097424|2|2|2|
|.0005|65509543717397/2|254602554677.51428|36384720355183|2|2|5|

下限適用前の正規式ceilingは、それぞれ1/1/1、1/2/5。小さいnを最終確認数として推奨していない。
H8の`normal_scenario`をそのまま再利用し、H6 adapterではmarginの絶対値を渡すため両端を大小順に並べた。
分散の入力は両方向ともPR探索n8のpaired SEで、元のPR/PF役割と期待方向は別フィールドに保存した。

n48での正規モデル計画powerは、6シナリオとも表示上ほぼ1（q8/f4でも約.999999999994）。
これは選抜n8のSE・固定効果量・正規/有限分散を仮定した計画値で、実power・CLT精度・tail分散の保証ではない。
未知σのt補正、選択後補正、bootstrap、CI、有意差検定は行っていない。

## K2 order-2 eventの可視性

K2のsupportは{0,2}。候補ごとの保存p2とR=q*rを保持し、元lawのIID occurrenceという仮定の下で
`π2=1−(1−p2)^R`、`V(n)=1−(1−π2)^n`を評価した。
保存probabilityの再fit・再正規化は行っていない。

|ε|保存p2（RTE eventごと）|R|trajectory中にorder2がある確率|n48で1回以上観測|97.5%以上となる最小n|
|---|---:|---:|---:|---:|---:|
|.05|.01739703152030961|8|.13099039917705854|99.881647%|27|
|.0005|.0011216662832458251|32|.03527622900502826|82.162129%|103|

最小nでの可視性はn27で97.742314%、n103で97.525389%。
このnは**少なくとも1回のorder2を観測するための計画値**であり、費用平均の必要精度・優位性の標本数ではない。
次数2の複数発生やcomponentごとの希少な費用寄与まで十分に推定できるとは主張しない。
未観測費用は0にせず、条件付きpilotの費用を今回候補へ移送していない。

## 候補別の推奨案

実値を見る前に、SOURCE内のpolicyで次の算定規則を固定した。
`推奨n = 8本単位切上げ[max(参考48, SD4倍の計画n, order2可視性97.5%の最小n)]`。
97.5%はH8にもあった可視性シナリオを参考にした追加の計画基準で、最終実験の採択条件ではない。
8本単位の丸めは設計上の選択で、worker数の制限ではない。

|ε|SD4倍の計画n|order2可視性の最小n|推奨n|推奨nでの可視性|主に効いた基準|
|---|---:|---:|---:|---:|---|
|.05|2|27|**48**|99.881647%|H8との共通参考規模|
|.0005|5|103|**104**|97.612684%|order2可視性＋8本単位切上げ|

H6を両方n48へ機械的に揃えるより、q8を104へ増やす案を推奨する。
この規則は費用母平均の保証に十分なnを導いたものではなく、独立した点順位確認の実用的な設計案である。
最終n・新seed registry・科学SOURCE・科学call予算はまだ固定していない。

## 実行・テスト・保全

新H6算術adapter・one-shot runnerの人工12件と、既存H8算術の人工11件、計23件PASS。
正負marginの同等性、exact fraction、下限2、normal式とpowerの整合、visibilityの最小n、拒否・markerを検査した。
実数値はSOURCE固定後の独立childで1回だけ評価し、結果SHA
`e51f91d9db4fae2d48547a575331b75af0a07f788f6af04af5ae0b223c817a87`を保存した。
完了資料はその保存結果の転記で、2回目の数値設計は行っていない。

native exit0、親のwait4/reap、wall .238秒。SOURCE7 file・実際に使うruntime89 file・入力3記録を実行前後にSHA照合した。
数値設計専用approval/grant/one-shot/予約/結果を別private領域に保存し、sampling/compileのgrantは発行していない。
新sampling/native compile/signal/shot再計算/新seed/raw分散再集計はすべて0。
CPU/RAM/workerの恒常cap、OS/cgroup/共有venv/他jobの変更、AI subagent、pushはいずれも0。
今回の計算は短いscalar算術であり、大規模な並列実行は必要なかった。
検査は作者による技術・metadata確認で、別reviewerの独立レビューや外部再現ではない。

## 停止後の担当

**ユーザー（次）**：推奨n=48/104の採否と、最終実行契約準備への進行を判断する。
**Codex（その指示後）**：採択n、旧・後続cohortと未完了予約を含む全seed衝突照合、SOURCE/input/runtime、科学call予算、並列実行計画を固定する。
**ユーザー（その後、別承認）**：最終sealを確認して独立sampling/compileを承認する。
**Codex（現在）**：限定数値設計を完了・local commitしmandatory STOP。最終契約や科学実行へ自動移行しない。
