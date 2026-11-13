# S3：ユーザー採択とB実行への引渡し

2026-11-13 JST。[最終科学計画](SCIENTIFIC_PLAN.md)／[REPORT](REPORT.md)。**計画のみ。以下の推奨セットは未採択で、科学実行も未許可・未実行。** 元SOURCE・結果・KEEP・STOPは保全した。承認証明書やgrant発行は必要ない。

## ユーザーが採択する具体案

| 項目 | 推奨セット（変更希望があれば実行前に指定） |
|---|---|
| SOURCE | コードB ab12b69120323379b0aae572e7f62692a5819125、入力B bf515a976f1987ad6a19c61d14689b27b0894542、A bba150834/169d3316の元来歴 |
| 範囲 | canonical 587d20f7…0e911、固定13round、保存nominal1配分、1実験のみ。M/κ/N/方式探索なし |
| Target | 保存solver energyを条件付きproxyとして採択。厳密ground/RMSE保証なし、energy shift0 |
| Shot law | ideal finite-K/fresh IID trajectory+Bornの周辺binomial、軸/round/反復独立を仮定。保存binary64 meanのplug-in、actual law不明 |
| Caller/decoder | exact整数和→Fraction→binary64、既存R3 batch hook、元scalar tie/zero/prior/alias規則 |
| 主loss | UNDEFINED。条件付きRMSE点推定、未解決率CI、解決分MSE寄与。主population RMSE/CI=null |
| 任意の副評価 | 未解決squared loss∈[0,4096] Ha²の感度envelopeを併記するか。特定fallback energy/全試行RMSEの定義ではない |
| Support | [−32,32] Haの全到達出力とproxyを覆う保守域、U=4096という前提を採るか。採らなければ副評価/有界loss CIを使わない |
| 数値不確かさ | Aの経験的allowance/bias・unknown/nullを参照表示。新伝播・新誤差認証なし、FPモデル条件付き |
| 精度/confidence | 副評価端点のη=.01U=40.96 Ha²（粗いpilot）。CPと端点CIは各97.5%、joint≥95%。条件付きRMSE点推定のCIは付かない |
| 有限反復/RNG | n=43,819固定、PCG64 seed20261113。零観測CP解像度10^−4と既存Hoeffding式から設計。結果を見て追加しない |

副評価を採らない場合も同じn/CP97.5%で主評価のみを実行できる。その場合MSEの精度保証はない。逆に化学精度のMSE精度を必須にするなら、このpilot案を採択せず、ηとUを先に決めて別の有限nを設計する。科学条件の採択だけで実行許可と曖昧な場合は実行しない。実行は固定SOURCE・上記有限scopeへの明示許可後に限る。

## Bが再利用する呼出し

以下は**採択・実行許可後の例**であり、今回実行していない。`prepared`はbf515の保存STATISTICS_INPUTSを無変更で読む。`adopted`は将来の採択条件を表す通常のrun設定で、承認証明の仕組みではない。

```python
from trottertracks import rpe_decoder_statistics as S
from trottertracks.rpe_decoder_fast import decode_sampled_counts

proxy = prepared['target']['saved_truth_proxy']['reference_solver_energy_ha']
simulation = S.simulate_prepared(
    prepared, repetitions=adopted['repetitions'], seed=adopted['seed'],
    policy=adopted['iid_policy'],
    allow_conditional_unqualified=True,  # この条件付きscopeの明示許可後のみ
    batch_decoder=decode_sampled_counts,
)
common = dict(target_energy_ha=proxy, iid_repetitions=True,
              confidence=adopted['component_confidence'])
primary = S.summarize(simulation['trials'], **common,
    unresolved_loss=dict(kind='UNDEFINED', source=adopted['plan_source']))
if adopted['include_loss_envelope']:
    envelope = S.summarize(simulation['trials'], **common,
        unresolved_loss=dict(kind='BOUNDED_INTERVAL', lower=0.,
            upper=adopted['loss_upper'], source=adopted['plan_source']),
        loss_support=dict(upper=adopted['loss_upper'], source=adopted['plan_source']))
```

policyは`kind=S.IID`、`target_law='ideal_finite_RTE_law'`、SOURCEを明示する。主/副summaryは同じtrial ensembleの再集計で、新simulationを追加しない。副summaryの`population_RMSE_CI`という既存field名は、報告時には「採択したloss域の条件付きrisk envelope」と説明する。primaryがF=0時に返す全観測RMSE/端点も、population RMSEへ昇格しない。

prepared中の未採択target/未資格metadataは原記録として保持し、**新runの採択条件**でproxy/SOURCE/評価意味を明示する。元`truth_proxy_adopted=false`、物理qualification、十分N/global bound/Gを改変してgateを通さない。実行前のidentity/値域/support検査、完了後の全trial会計とSTOPは[計画第5〜8節](SCIENTIFIC_PLAN.md)に従う。

## 境界・出力・次の担当

出力先候補は本directoryの`run01/`（現在未作成）。新規directory専用とし、SOURCE/入力参照・採択条件、全trial、主summary、採択した場合だけ副summary、完了/部分STOPの状態を保存する。関係する通常のSOURCE/入力保全を行い、無関係なworktree HEADを凍結しない。反復完了は科学的PASSを意味しない。

新signal/normalization/物理mean・target再評価、実sampler trajectory、shot最適化、価格/G、4方式最適化、CI方式の変更は含めない。未解決率/精度が期待に届かなくても自動追加・再seedしない。十分N/global bound/Gはnullのまま。A/C/Dへ新しい実行指示を出さない。

**次：ユーザー**が条件と有限実行を明示許可。**Codex B**がその範囲だけ実行・報告しmandatory STOP。**今回は**計画のcommit/push・remote SHA/主要成果物照合後mandatory STOP。
