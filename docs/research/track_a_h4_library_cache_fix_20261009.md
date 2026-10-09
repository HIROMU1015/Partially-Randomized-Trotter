# H4 Matplotlib保存先修正・再実行予算確認（2026-10-09）

Matplotlibの設定directory作成で止まった経路を修正し、限定人工検証で合格した。
**本計算は再起動していない。累積10GiBの予算条件が不合格のため、再実行planは未seal、approved/runtime_authorization=false。**

資料は[修正・binding bundle](../../artifacts/resource_applicability/track_a_h4_library_cache_fix/2026-10-09/README.md)にまとめた。
旧runの証拠・one-shot marker・worktree・失敗分課金を保持する。

## SOURCEと変更範囲

- branch：`track-a-h4-library-cache-fix-20261009`
- SOURCE：`b8b3ce6e8c98f1ec0419a7af79c5d7c5f3a3b9bb`
- 起点REVIEW/結果commit：`0bfeb3e6da83ad4bb2af8d6061eb944fb8e35218`
- 修正前SOURCE：`6bd1ba01cd71ec3e2071082963c9f07478dada9a`
- SOURCEはsource/tests/schemaの9fileのみ。closureは33→36。[旧/new blob・SHA](../../artifacts/resource_applicability/track_a_h4_library_cache_fix/2026-10-09/source_freeze_v1.json)。

`library_cache.py`を追加し、home配下の新しいprivate plotting dependency cacheをbyte/profile固定した。
driverとworkerのprocess環境に`MPLCONFIGDIR`を設定する。既存HOME/XDG設定、venv、共有設定は変更しない。
Matplotlibがimport時に行う既存directoryへの`mkdir(exist_ok=True)`のprobeだけを認める。
そのmkdirはEEXISTとなり、新directoryやcache fileの作成・書換え・rename/removeは引き続き拒否する。
cacheのinventory・bytes・SHA・Matplotlib versionの欠落や変更はSTOP。

`/home/AbeHiromu`は利用者のprivate homeであり、共有system directoryではない。
前回の二つのrun保存先への制限は、私が累積byte予算の管理に追加した制限だった。
home自体の利用許可と、runtimeで課金管理する保存先の制限を区別する。
前回traceの`shared/outside run write forbidden`という文言は実際の停止証拠として保存するが、
homeへの書込みを共有環境変更と分類する説明は訂正する。
正確な試行pathは前回traceに記録されておらず、今回も推測で補完しない。

科学条件・template・SCF/DF/state・signal式・回路builder・compiler optionsとcandidate environment/compilerは不変。
SOURCEの変更により将来のseed identityは再結合が必要で、旧random/partial結果・科学cacheを混合しない。
追加transpile、実入力array読込、入力再生成、本番worker、own affinity変更、GPU query/useは今回0。

## 限定検証

[47件のbinding/cleanup/cache回帰と3 import case](../../artifacts/resource_applicability/track_a_h4_library_cache_fix/2026-10-09/limited_tests_v1.json)はPASS。
cold fixtureを新規作成した後、driverとworkerそれぞれのruntime write guardで
`openfermion`、`matplotlib.pyplot`、`qiskit`のimport経路を検証した。
cacheは1file/29,816 bytes、両guarded caseでhash不変。各caseは約1.4–1.6秒、current process peak RSS約235–237MiB。
科学API呼出し・transpile・実NPZは使わない。

人工import caseは各1process、AS2GiB/RSS512MiB、wall60秒、file/cache4MiB以内。
外部processを起動せず、fontconfig/platform subprocessを人工caseだけOSErrorで拒否し、
Matplotlibの通常のfont directory scanとplatform fallbackを用いた。
本体ではその既存cacheを読む。これはdependency I/Oの回帰であり、全科学pipeline成功やcompiler output完全同一性の証明ではない。
cleanup回帰だけ最大4最小人工process、実12workersは起動しない。
全testの計画・上限・ownershipはhomeの`ARTIFICIAL_TEST_PLAN_v2.json`へ実行前に保存した。
初回のfixture時刻と`/dev/null` exemption不足による人工test失敗を保持し、修正後PASSを採用する。

## STOP・入力・carryの結合

旧hostの6凍結NPZ/freeze/native proofの固定受領bindingを保持し、streaming byte SHAとmetadataだけを照合した。
第一newhost runのjournal/launch-stop/observer/native14identityを追加結合する。
全14identityは退出を再確認。旧hostの未記録exit code・正確な終了時刻は未記録のまま。
前回newhostの実exit1、actual予約0・完了0を保持し、古い20 actualを返却しない。

| carry | 今回固定値 |
|---|---:|
| 消費/予約actual | 20 |
| 累積charged bytes | 4,428,938,712 |
| 累積保守的wall秒 | 5,472.345380863175 |
| 累積actual cap | 74,804 |
| 残actual | 74,784 |

全74,784 logical wrappersのstatic worst actualは74,784。carry込み74,804であり、cache節約は保証しない。
新しい候補run IDは`h4-newhost-signal-compile-20261009-run02`。
output/controlは今回evidence root配下の未使用pathに結合し、旧root/one-shotを再利用・削除しない。

## 再実行の障害と承認案

固定保存形式・最大72h observer trace・control log・journal・temp/final・新library cacheを含む
再実行の累積charge上限見積りは**13,165,893,832 bytes（12.261694GiB）**。
現在の10GiB capに2,428,475,592 bytes不足する。
これは実際に使用したディスクbytesではなく、返却しない過去予約を含む累積課金値である。
新outputのphysical容量案は5GiB/301,000 inodesで、空き容量が大きくても累積charge不合格は解消しない。

このため新planは未seal、auth/reviewはapproved=false/runtime_authorization=false。
既承認のprivate environment/compiler、worker12 CPUs `[2,4,5,6,8,9,10,11,12,13,14,15]`、
driver16/observer18、内部thread1、observer AS256MiB/RSS64MiB/admission120.25GiBとactual74804の承認は保持する。
それらの再承認は求めない。新SOURCEは現在10GiBを強制し、13GiB proposalだけでは起動できない。

残る承認案は、**累積output課金上限だけ10GiB→13GiBへ変更し、変更後に固定するSOURCEでmapを一度再実行する**こと。
carryの返却・resetはせず、72h、各roleのAS/RSS、headroom16GiB、monitor5秒、科学条件、compiler optionsは維持する。
完了またはfail-closed STOP後に終了し、自動retry・入力再生成・次stageは行わない。
明示承認後はcapの認可gate更新を新SOURCEへ固定し、binding/独立最終reviewと
fresh CPU/memory/PSI/OOM/fs/block/inode/quota、未使用root/one-shotを確認してから一度起動する。

10GiBを維持する選択も可能だが、その場合は今後の保存・observer形式の最大費用を減らす別実装と検証が必要。
過去の予約を取り消して収めることはしない。
現時点のabsolute_launch_commandはnull、実行0。

## 公開

独立担当`/root/independent_final_review`の最終判定は
[IMPLEMENTATION_SCOPE_PASS_LAUNCH_BUDGET_FAIL](../../artifacts/resource_applicability/track_a_h4_library_cache_fix/2026-10-09/independent_fix_review_v1.json)。
追加の必須実装修正なし。SOURCE36/旧-new blob、profile/cache/input/control/carryと三digestを独立照合した。
draft拒否、現SOURCEで13GiB拒否、10GiB累積budget拒否がone-shot/affinity/科学処理より前に成立することを純mockで確認。
担当は既存47回帰/3importを再実行せず、原本対応を照合した。実装PASSをproduction成功やfresh launch readinessとは扱わない。

source/testsのSOURCE commitと、sourceを変更しない軽量REVIEW_BUNDLE commitを分離する。
cache file、NPZ、raw runtime/checkpoint/log、credential・内部SSH情報をcommitしない。
最終REVIEWのactual40文字SHA・remote状態は最終報告とhome外部publication receiptで確認する。
