# 実行policy候補：技術的に確定したものと未採択事項

目的とmath-only scopeは今回のユーザー指示で指定済み。旧候補のままにして再選択を求めない。
次のR1は、既存driverに実装されたpolicyをまとめた**採択候補**で、正式採択済みとは書き換えない。

| 項目 | 実装済み／固定事実 | R1の提案・採択範囲 |
|---|---|---|
| seed | frozen G7 SHA256-counter、H=160のdigest上位bit、余剰破棄 | `B3_FULL_N_NEW_20261019_V1/arm/axis` |
| stream共有 | 8つのarm/axis文字列をA/B・3phase-passで再使用する設計 | 同じprefix、X/Yは異なる識別子。fresh-IIDや確率的独立とは主張しない |
| setup | 24 generator constructor、Full F3 prepare3（内包constructor）、canonical preflight9 | arm/layer/phase-passの専用child、X/Yで保持state共有 |
| identity | 初期・X前・Y前・終了の4 source/state照合/pair | 同一p/x/m/law/precision/source/preflightのみ共有 |
| cache | 新law/packet/angle memoなし、元guard/hash/requeryを保持 | OS cache未制御、他job cache不操作 |
| 二pass | 同じ保存Nをbaselineとinstrumentedで別々に完了するengine | observer比較のため両方の実費を払う。純overheadの認証／差引きをしない |
| 保存 | correctness全16/axisとprofile初期16/axisの完全書類＋全trial compact hash | 後続の完全event復元を諦める保持範囲を採択する |
| 出力 | exclusive JSONL、chain/file SHA、rows/bytes、無buffer、4096行flush＋close | result64GiB／log256MiBを維持。append/resumeなし |

## Seedと来歴

namespace/arm/axisの文字列規則だけを照合し、実bitstreamは生成していない。
同一arm/axisに6回のbit-provider初期化（correctness A/B・baseline A/B・instrumented A/B）、全48回。
これは24 generator constructorと別のsetup処理で、sampling callsの増加ではない。
旧`G7-development-bitstream-v1`と文字列が異なり、旧128-trial診断seedを延長しているとは言わない。
使用済みgrant/claim/markerの実IDを取得／再利用していない。将来の実ID・scope一意性は本番admissionで検査する。
固定SHAのtrace一致は決定論的再現性であり、fresh-IID保証やCPU母平均の統計certificateではない。
G7のα_axisや16-axis familywise規約をCPU confidenceへ転用しない。

## setup・online・outputの課金

Full F3 prepare内のconstructorを24に加えて27回とは数えない。canonical preflight9、Full prepare3、
共有budget file load1、budget()再評価0。X/Yでlaw/stateを保持し、source identityとheld stateを照合する。
各sample内の元canonical conditional law再構築とFull packetをsetup cacheへ移さない。
Full angle時のguard/event→packet再queryも残す。同じ実費をsetupとonlineへ二重課金しない。
parent source/module load、dispatch、共有final outputをchild totalへ混ぜない。
parent dispatch wallはchild待ち時間を含むため、両者を足さない。
粗いstageと細かいfunction partitionは各々排他的に外側totalへ照合し、両partition同士を加算しない。

各pairを新しいchildに分離するが、forkのparent inputやcoordinator prefixもchild baselineに含む。
baseline RSSを併記し、HWM差を純generator memoryと断定しない。OS file cacheのcold/warm制御は行わず、
固定順序の環境依存の記述的観測とする。実験順や並列競合を変えた一般優位は主張しない。
独立benchmarkを同時実行しない理由は、比較worker間の競合と最初の失敗後の追加callを避けるため。
CPU/RAM/worker/wallの人工上限ではない。空き資源をlaunch前に再確認し、他jobや設定を変更しない。

## 保存から独立に確認できること／できないこと

完全JSONLからchain、file SHA、行数/bytes、完全行prefixと整合性を再検査できる。
保持した初期event/angle/audit全文はexact hex有理値を復号してphase、word順序、πW、angle evidenceを監査できる。
後続のcompact event/evidence/angle hashは来歴・同一性の照合先になるが、元eventを逆算／復元する情報ではない。
全N traceの独立再生は、新しい科学実行を要するため未承認。hash一致を内容の独立数値再証明とは扱わない。
markerや結果ファイルが存在することだけでも、全N完了・accuracy・native費用を証明しない。
flushはfsyncではなく、OS kill／電源断での耐久性を保証しない。最後の自己receipt書込費はnull、0と扱わない。

64GiB/256MiBは変更しない。実outputが収まるか、disk/I/O圧迫がないかは未実測。
超過／保存失敗はprefix保全＋STOPで、保持を勝手に削ったり追加trialで補充しない。
耐久性や全event保持、別cache方式は別の合理的policyにはなり得るが、現R1へ黙って追加しない。
R1を選べばこれらの代替policyを選ぶ必要はなく、実費観測の解釈を上記scopeに限定する。
