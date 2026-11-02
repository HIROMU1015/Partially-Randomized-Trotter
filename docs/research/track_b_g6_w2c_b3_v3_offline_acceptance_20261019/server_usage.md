# 将来サーバー導入手順（今回未接続）

担当は別途指示後のサーバーCodex。公開consumer content commitはpublication_receipt.jsonで固定する。consumerはPython3.10以降・標準libraryのみ。旧producerの実環境を上書きせず、別checkout/専用venvと新しいoutput先を使う。依存はsource_inventory.jsonの5新module＋旧7consumer module＋package init。科学moduleをimportしない。

## 人工smoke検査

固定consumer checkoutのrootで、未存在output pathを指定する。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m trottertracks.algorithm_codesign.g6_w2c_b3_v3_offline_cli \
  --artificial-manifest docs/research/track_b_g6_w2c_b3_v3_offline_acceptance_20261019/published_cases/light-PF-0/manifest.json \
  --fixed-reference docs/research/track_b_g6_w2c_b3_v3_offline_acceptance_20261019/fixed_reference \
  --new-output /tmp/b3-v3-independent-new-report
```

7 outputs: verification JSON/MD、cost JSON/MD、acceptance JSON/MD、正常のみcomparison CSV。入力/referenceと祖先・子孫関係のある出力先、既存outputへの上書きを拒否する。exit0は正常完了又は分類可能な意図障害であり、科学GOではない。cost mismatchや入力不整合等はexit2。必ずJSONのstatus/pendingを読む。

## Input manifest

schema `b3_v3_offline_input_v1`、artificial_only=true、固定producer公開/measured commit、case label、raw.tar.gz pathと物理SHA、host_report path、任意の完全contract pathを指定する。相対pathはmanifest directory内だけ。archive rootはartificial-output、pair/control/state/summaryの固定ファイル。原reports safe_json Fraction wrapperはadapterがdecodeする。正確な例はpublished_cases各manifest。

過去runtime_rootのpathへアクセスせず保存identityを照合する。SOURCE/runtime/schema/contract識別値をauthorization credentialにしない。公開contract欠損や実体不足を新しいfileで補った場合は元公開47件と別の証拠版・manifestにする。

## 将来の実データ

現版CLIは人工manifest専用で、正式adoption/real authorizationを受け付けない。今回の固定v3 schema用consumer導入と人工smokeまでは準備済み。将来の正式SOURCE/schema・science contractが採択されてから、別指示の下で実データmanifest/取得経路とidentity境界を確認する必要がある。P07/P08/P11等の証拠不足も残る。正式producerの再実装・schema推測・実claim作成は行わない。

今回サーバー接続・sampling・本番データ再解析・正式採択は実施していない。
