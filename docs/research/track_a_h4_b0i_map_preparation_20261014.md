# H4 B0+I：全面240回路の実行準備

2026-10-14 JST。**準備完成、本計算は未起動。**
[全bundle・source/profile/input/budget binding](../../artifacts/resource_applicability/track_a_h4_b0i_map_preparation/2026-10-14/README.md)を入口とする。
起点は公開確認済み結果commit `c377b074ab6cc442d6eb73ca4417bc56fa663d5c`。
新SOURCE `0f84119302de869c904bfd0135af3880a6c16bf2`、独立branch `track-a-h4-b0i-map-preparation-20261014`。
旧worktree・source・6件のnative証拠・原稿・比較結果は保持した。

## 完成した範囲

先行科学レビューで標準案とされた同一domainの全面対照を準備した。H4 linear/STO-3G/DF12、8 system＋ancilla、T=.8、
ordinary control、structured Gaussian、二次PF、6距離(.70/.80/.90/1.10/1.40/1.60 Å)、prefix3/4/5/6/9、q1/2/4/8の両軸。
全120構成／240 wrapperを固定し、prefix4/5の元PM1-B0 labelsも明示mappingした。
B0+Iを元B0に追加する評価であり、元B0/B1/B2/B3や保存PR結果を上書きしない。

旧48 sourceは全件不変、新7 Python source/testsと合わせclosure55を実SOURCEのGit blobへ固定した。
新19限定stdlib/mock検査PASS、scientific import／array読込／real・artificial compile／spawn／affinity0。
同じ科学レビュー・旧44人工tests・旧benchmarkを繰り返していない。限定差分reviewと公開binding verifierを実施し、CIや独立再現とは呼ばない。

## 既存6件の再利用と会計

6件の旧complete/native終了・cleanup、SOURCE/入力/compiler、geometry/prefix/q/axis/tail/phase/controlを新taskへ結合した。
実行時に元と同じbuilderで回路を作り、numeric fingerprintが旧6件とexact一致した場合だけmetricsを再利用する。
新SOURCE／namespaceのfingerprintは別に保存し、旧source／task／実費用の出自を明記する。
一致しなければSTOPし、残り予算を使って勝手に再compileしない。旧B0費用・partial/cache・失敗reservationは使わない。

これにより新規実compile対象は234件、累積73029＋234＝73263／74805、残1542件となる案。
carry charge13615239890 bytes／記録wall268330.081695186秒を保持し、失敗の返却・budget resetをしない。
旧pilotも新stageも全出力枠をchargeする保守見積り14151959250 bytesでも17GiB内。

## 一括で確認する新実行条件

| 項目 | 実行案 |
|---|---|
| SOURCE | `0f84119302de869c904bfd0135af3880a6c16bf2` |
| 範囲 | 240 wrapper：旧6件を実build/FP照合して再利用、234 new compile |
| CPU／並列 | worker0、driver1、observer2、worker1／内部thread1 |
| メモリ | driver/worker AS・RSS各8GiB、observer AS256MiB・RSS64MiB |
| admission／監視 | 空き120.25GiB、host full PSI<1%、nonroot0、OOM増加なし、5秒監視 |
| wall | stage最大4時間（旧pilot1時間から変更）、task最大10分 |
| 保存 | 全stage256MiB、observer最大160MiB（旧32MiBから変更）、run1024 inode（旧256から変更） |
| fresh容量 | 320MiB／1024 inode、native quotaの十分な残量または既知非有効 |
| compiler／科学 | 既存固定venv・Qiskit options・凍結入力・PF条件を維持 |
| 終端 | completeまたはfail-closed STOP後に終了、retry/resume/次stageなし |

4時間はSTOP上限であり予想所要時間ではない。q8/prefix9の性能・peak memoryは未測定である。
出力256MiBを維持するため、4h×1Hz×最大8KiBの観測logを118022144 bytesと見積り、160MiB observer枠内に納める。
240 results＋234 reservations＝474 filesのため、inodeを1024へ変更する。
private-temp/cache32MiBとmetadata/block rounding8MiBを見込み、実使用が上限を超えればSTOPする。

候補environment/compilerは既存profileに一致し、旧host compilerのoutput完全同一性は未確認のまま。
6 NPZ計82321596 bytesとfreeze27115 bytesはstreaming SHA一致、JSON metadataからH/DF/state/inputを再照合した。
準備時の空きメモリ約466GiB・disk約373GiB・CPU0/1/2 sample0%・pressure0・quota KNOWNは予約でも直前gateの代用でもない。
H8/他jobを停止・移動せず、承認後のfresh利用可能性で起動可否を判断する。

今回の「次の作業へ」の指示で具体的な実行案まで完成した。旧6件のgrantを拡張せず、上記新SOURCE・234件・
4時間／160MiB／1024 inodeの条件を結ぶ新grantの承認待ちとした。
`approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、`absolute_launch_command=null`、runtime seal false。
承認後は通常のbindingとfresh gateを確認して一度起動し、同じ科学レビューや旧人工testsを再実施しない。
原稿の主要claim・全面費用地図の解釈は実測後に判断し、成功を理由に追加stageへ自動進行しない。
