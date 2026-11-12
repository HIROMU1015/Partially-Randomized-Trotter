# A/B/C最終統合の接続API

[REPORT](REPORT.md)／[SOURCE対応](SOURCE_MAP.json)／[人工結果](SYNTHETIC_SUMMARY.json)／[TESTS](TESTS.txt)。A 510fbacd・C公開2a882fc9/実装09489eedをbyte一致で取り込み、B82bc774を最小拡張した。

## APIの順序

```python
from trottertracks import pr_rpe_rmse as A
from trottertracks import pr_rpe_rmse_controller as B
from trottertracks import pr_rpe_rmse_integration as I

# args: M, kappa_paper, design, tail_norm, policy。A最終APIと同じ。
# policyの科学的採用や実meanの評価はこのコードで自動実行しない。
schedule = A.schedule(**args)
config = B.generate(I.a_spec(args, method='PR'))

# A自身が作ったb_handoffの該当PF/PR packetをそのまま渡す。
n = I.n_from_a(config, a_packet, inputs=args)
# nはnominalを保持するがN_X/Yとrmse_bound_haはnull/未資格。

lookup = I.MeanPriceLookup(config, supplied_saved_C_records, selections=cohorts)
prices = lookup()  # Cの受動lookupのみ。新mean/fit/compileは起動しない。
qualification = B.assess(config, n, prices, cohorts=cohorts)
```

現在のA packetでは`qualification.eligible=false`、G=nullとなる。後段の技術確認には、別の明示SYNTHETIC_ONLY packetでmock Nとmock RMSE条件を与える。`nominal_N_X/Y`をN_X/Yへコピーする関数は提供しない。実科学の十分N・boundが公開される場合は新しいversionのA契約を確認してadapterを改訂し、現固定SOURCEのnull条件を便宜的に緩めない。

`B.cost`/`B.select`は人工入力または別途許可された実評価だけで呼ぶ。`finite_cohort_scenario`は条件付きscenarioに限り、全law価格ではない。共有κのconfiguration単位で選択し、不同κのroundを組み合わせない。

## interface対応表

| A/Cの最終field/規則 | Bへの接続 |
|---|---|
| A.KAPPA_DEFINITION / A.schedule | I.a_specで名前/SOURCEと全入力を固定し、B.generateの明示A tagからそのまま利用 |
| pf_grid固定δ / H6 early_overrides | Aが検査する。B ceilへ変換しない。非整数gridを拒否 |
| 物理candidate/配分/実δ | row.physical_candidate/structureへ保全。B candidateにはsource_candidate_id/physical_methodも保持 |
| M・共通κ | 全configurationと全row identityへ含める。PF κはnull |
| A raw policy/N packet | 元policyをerror_contract、元packetをA_original_packetへcopy。nominalとnull十分Nを区別 |
| C旧APIの物理数値表現 | A rowからlookup viewを作成。identityはBの全schedule identityを返す |
| C law/domain表示 | copy上だけ必要なR/K/qを整え、元law/model domainへ復元。値/SOURCE/cohort/資格は変更不可 |
| C passive MeanPriceProvider | B所有instanceだけにAPI façadeを設定。kernel・fit・compilerへのfallbackなし |
| C1 model false / missing | false/null/OUTSIDE_DOMAINのまま。old identityをnewへ付け替えない |

profileにはcompiler/control/full Hadamard wrapper/P0が必要。Aの最小TOY demoに不足するprofile項目は、人工integration testが明示したmock profileを供給する。実SOURCEへ暗黙の設定を補完しない。CのH4/H6 recipe domainは維持し、PF/初期H6/純RTEについての不足を隠さない。

## 検証

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B \
  tests/check_pr_rpe_rmse_integration.py
```

入力はすべて人工。Aのmock mean評価とCのclosed reward kernelを使った人工価格生成はこのtest内だけで、providerのlookupやpacket変換からは数値leafを呼ばない。適格なmock Nは元A nominalと別SOURCE・別値で与える。新12件＋A/C/B・HP/R/provider/会計/C1の関連回帰をTESTSへ記録する。科学的保証や実Gの検査ではない。

## 残る入力・承認範囲

1. **A/ユーザー**：finite-K経験的Nの成立、共通κ/event対応、H6負時間・早期例外、state/effective evolution、alias/decoder/全round RMSEと誤差契約を明確化。十分N・bound・適用条件をSOURCE付きで供給する。
2. **C/ユーザー**：同一の候補・分割・公式・実幅・qPF/R/配分・compiler・lawに対するphysical closure、平均価格/残差、校正/独立検証、価格精度/全law/共分散を定義。PF・早期H6・純RTEの価格経路も不足のまま保つ。
3. **ユーザー**：新mean/Nの評価、保存再解析、S0、fresh compile/fit等が必要なら、それぞれの固定候補/SOURCE/input/精度・有限budgetを別途明示許可する。旧P2/P3や40/16 wrapper計画を自動実行しない。
4. **B**：公開された科学的に適格な新versionのrecordを受け入れ、実G/κ/M/外側最適化は別途明示許可された有限menuでだけ行う。

報告状態はTECHNICAL_INTEGRATION_PASS／SCIENTIFIC_QUALIFICATION_PENDING／RESOURCE_OPTIMIZATION_NOT_RUN。今回の完了後はmandatory STOPし、新しい科学計算・詳細科学レビュー・実最適化へ自動進行しない。
