# 次scope：残ったcanonical producerだけを対象にする

**未実装・未実行。今回の科学実行許可はない。** [REPORT](REPORT.md)／[入力契約](CONSUMER_INTEGRATION.md)の未資格は、固定finite-bit lawからevent列を生成するproducerのSOURCEである。compiler/resetの再検証や全18round会計を繰り返してこの不足を埋めない。

## 1. 先に必要な判断・SOURCE確定

GPT／ユーザーが、既存integer_cdf／draw_cdf／_make_eventを結ぶ固定producerを実装・採用するか判断する。今回、適合する完成済みproducerは確認できなかった。law/sourceの申告欄を埋めるだけで採用しない。

次の担当Codexは、許可された場合に限りそのproducerを最小実装し、SOURCE上でorder/component drawの順序、bits、独立性の仮定、_make_eventへのindex対応、normalization、shot間の再利用禁止等を確認する。これは新law探索やoptimizerではないが、今回の指示が認めたinterfaceの範囲を越えるため、自動開始しない。新producerの固定SHAがまだないので、下記は**科学campaignの実行準備案**であり、今すぐ実行できる固定契約ではない。

## 2. 必要となった場合の最小実入力診断案

既存Q1/Q2/Q3-Nを再実行しない。SOURCEの静的証明と人工テストだけで接続資格が閉じれば、新しい科学診断を必須にしない。実H4入力との数値対応がなお不足する場合だけ、producerの単一event kernelに限定した1 finite campaignを固定する。

| 項目 | 準備する最小範囲 |
|---|---|
| 固定SOURCE | 元科学f2638a78、caller5fdb68e4、今回consumerの公開commit、採択された新producer commit。後者を実行前に固定し、推測・placeholderで開始しない |
| 固定入力 | common.pkl.xz／pickleの既存hash、保存STAGE4_RPE metadata、既存component順・cfg/dist生成関数、integer_cdf／draw_cdf／_make_event |
| 対象context | m0のr1/bits66と、m17に対応するr32/bits88の**event kernel**。後者をm17 whole trajectoryと称さない。q/r/bitsは保存rowへ結ぶ |
| 有限fixture | 各contextでorder0とorder2の各1件、計4 kernel fixture。rotation index0、order2のproducts index1,0を既存_make_eventへ渡す対応を使う。draw整数は対象CDF cellの左端とし、実行前SOURCE／fixture定義で固定する |
| 科学call | 固定cache読込1回、2contextの既存cfg/distとorder/component CDFの評価、4件のproducer kernel評価と固定indexを使う旧_make_event参照。RNG統計samplingや全q*r生成は行わない |
| 比較 | 同じindex列を既存_make_eventへ渡したeventと、producer kernelが返すeventのorder、ID／ordered products、sign、angle binary値、phase、normalizationの一致 |
| correctness／phase | 既存SOURCEのbinary値・event意味を保持。phase fit・事後補正はしない。実F/G、Reference core、simulation、B1位相評価は0件 |
| 価格 | 同じ固定compiler／reset構造であることのSOURCE照合のみ。native価格取得・RPE／総RZ・CT評価は0件 |

この4件は入力配線の有限correctness診断であり、canonical lawや独立性を統計的に証明するものではない。分布の資格にはSOURCEからのfloor-CDF law・draw独立性・旧finite-bit proposal diskとの対応が別途必要である。CDF cellが欠損・空、保存metadataと不一致等なら別indexや条件へ拡張せずSTOPする。

## 3. 承認・判定・STOP

上記の実入力診断を行う場合は、新producer SHAと全fixture、SOURCE、科学call範囲を確定した後、**その限定範囲の数値実行に明示承認が必要**。実装許可と科学実行許可を混同しない。承認証明書や新grantは不要。

- SOURCE証明と人工／有限correctnessが揃い、固定lawとの対応を示せた場合も、V6-Pの数値は保存結果として移送し、新総計を作らない。研究成果の正式採択はGPT／ユーザーが判断する。
- producerの意味・normalization・固定cache／bits／event順序が旧条件と違えば、未資格または負結果として保存し、旧V6-Pを流用しない。
- 入力不一致、CDF／kernel不一致、独立性・provenanceの不足は救済samplingや別lawで補完しない。部分結果とSTOPを残す。
- 新compiler contextが必要と判明した場合だけ、旧文書の局所P／X/Y検証scopeへ戻す。今回の接続は同じcontextなのでcompileを自動追加しない。

完了または不一致時にSOURCE・入力・結果・検算・STOPを保全して公開しmandatory STOP。新しいm17 whole build、全18round RPE runner、追加最適化、T-count／T-depthへ進まない。現時点の次担当は**GPT／ユーザー**であり、この案の実装・数値campaignは開始していない。
