# RPE MSE上界の状態集約による改善

2026-11-14 JST。基準`6106e4d969bb08e95166cbd2b2cc55b9840c7535`、既存branch `codex/rpe-decoder-statistics-20261113`を再利用した。**今回は技術実装・人工検証のみ。保存H4への新しいrisk適用・再解析、shot探索、MC、実価格/Gは実行していない。**

## 原因と変更

旧式はall-good終端寄与に`P(any unsafe) × L_bad`を加える。unsafeは宣言したphase安全集合外であり、実際には元decoderで解決し、終端energy lossが小さい履歴も含む。全unsafeを同じ主値範囲lossで囲む点が粗い。一方、実zero/tieはenergy=Noneであり、採択済みloss familyを縮めたり0にしたりできない。旧H4結果の約0.24 Ha²とloss約1127.77 Ha²は[固定結果の引用](../pr_ld1_ld2_deterministic_risk_results_20261113/REPORT.md)であり、今回再計算していない。

[既存module](../../../src/trottertracks/rpe_deterministic_risk.py)に任意の`options.state_refinement`を追加した。元count表・整数二項確率・R3最近branch処理を使い、解決中のbinary64推定値を完全一致で集約する動的計画法（DP）である。unsafeを理由に即最大lossへ落とさず、解決した終端状態では実際のenergy lossを使う。

zero/tieに遭遇した履歴は未解決として吸収し、元loss familyを適用する。追跡範囲外へ落とした履歴は確率質量を捨てず、旧L_badで囲む。旧上界とDP上界の小さい方を採るため、成立した新上界は旧上界より大きくならない。DPの成立確認が失敗した場合はUNKNOWNで、未検証の改善値や旧値を正式結果へ自動fallbackしない。

既定（state_refinementなし）の経路は保持した。元decoder/statistics/caller/whole-shot/cost-selectは変更しておらず、新decoderやN資格producerも追加していない。旧SOURCE/input/raw/結果/STOPと実行callerを保全した。[定義・証明・API](HANDOFF.md)。

## 人工データと検証

**新12＋既存45＝57件PASS**。既存45は元決定論的risk13、controller17、whole-shot8、決定論的decoder7。MCを使うテストは除外し、MC/物理providerと保存実signalの受入をmockで禁止した。[TEST_RUN](TEST_RUN.txt)／[再現手順](TESTS.md)。

- 小countの全履歴を独立comb/Fraction確率と元scalar decoderで列挙し、完全DPのloss区間・zero/tie/未解決確率がexact一致。
- 不完全DPでも全履歴のriskを囲み、選択上界≤旧上界。状態数/遷移組の外部範囲、残余質量、zero優先、誤branchを含む全遷移・alias・主値境界・signed zero・極小非零確率を確認。
- 全13round/X/Yのwhole-vector、原入力不変、PMFの重複計算なし、SOURCE不一致拒否、N/global/G null、人工価格の既存会計接続を確認。
- count質量不整合・非有限遷移等でUNKNOWN、未採択loss/旧good閉包不成立を迂回しないことを確認。

[人工比較の正本](SYNTHETIC_COMPARISON.json)：M1/h10/target0、両round raw=(0,0)、各軸N=(1,1)、未解決loss=[0,20]。

| 手法 | MSE上界（人工単位、近似表示） | 備考 |
|---|---:|---|
| 旧式 | 15.000385531421918 | unsafeを一律最大lossで評価 |
| 完全DP | 0.032384639441074455 | 全履歴列挙とexact一致、実未解決0 |
| 部分DPと旧式のmin | 15.000385531421918 | 部分DP単独18.750867445699313、残余質量0.9375のため旧式を選択 |

これは人工的な改善例であり、実H4の改善率やrisk値を推定するものではない。新しいH4数値欄は作っていない。

## 残る限界と次の範囲

完全DPは全count履歴の直積を列挙しないが、round mの候補状態は最大でq_m×count位相数に増え得る。dyadic q自体の増加、exact整数のbit長と遷移計算も重くなり得る。外部から指定した有限追跡範囲で残余を囲めるが、改善を常に得られるとは限らない。範囲拡大に対する単調改善も仮定しない。

元Minkowski、proxy/state/実sampler/数値誤差の資格は別問題。rawモデルriskを純shot項と同定せず、AδEを加算・控除しない。十分N/global bound/正式Gはnullのまま。

[有限適用・shot計画](FINITE_PLAN.md)では、まず元LD1/LD2の同一vectorを各1件、新方式で診断する案を示す。新配分の採択・評価や正式必要Nは別段階である。**ユーザー**が追跡範囲と科学実行を明示許可した後だけ、**Codex B**が指定された有限範囲を適用する。今回はcommit/push・remote照合後に**mandatory STOP**。
