# q4_sample00入力・再利用状況

本資料は実行済み科学結果ではなく、原本未取得で停止したpreflightの記録。追加の承認packetや管理frameworkは作らない。

## 取得できた固定binding

正本は人工基点 `a54d04059bfc61adef43aa2d40fb9c001b62378f` に含まれる次の公開資料。

1. `artifacts/track_d_c6a_paired_price/2026-11-06/run01/SOURCE.json`
2. 同`event_bindings.json`
3. 同`paths/q4_sample00/payload_archive.json`
4. 同`payload_publication.json`

`q=4, sample=0, sample_id=q4_sample00`、seed domain `track_d_c6a_q2_q4_paired_price_pilot_v1:calibration:q4`、保存seed4個、`record_sha256=919f2e784920581380cb4090c59fff34741d914e9d340d249bca554eba8cadb1`を確認した。seedは既存値の引用だけで、新しいseed算出/PRNG/sample呼出しは0。今回のpath選定は利用者指定で固定され、価格による選び直しはない。

## 必要な原本と取扱い

最小の次の入力は、上記hashと一致する既存event原本。通常の旧runner保存位置はC6a runの`local/events/q4_sample00.json`、archive内では`events.json`。同じ`q4_sample00`という名前だけでは適格でなく、183,097 bytes/期待SHA256と公開SOURCE・endpoint/input/law・4×32 event順序/符号/phase/normalizationを確認する必要がある。

対応する既存archiveは6,950,831 bytes、SHA256 `a631cd394073f7770d2723eb0cfa93fc6a8f703534b9cdce88b7c126381a007b`。その20 memberは公開hash索引から照合できる。保存QPY/PASSが利用できれば、Strong X/Y・中央Reference X/Yを重複compileしない。Archive取得はGitHubへのraw公開許可を意味せず、従来どおりprivateの独立作業領域へ保持する。

中央以外が選ばれ新規buildが必要な場合には、C6a SOURCEの元endpoint/number-CDF8入力packet・既存CAR参照と準備contextも照合する。今回、それらを再生成したり、別Hamiltonianから代用したりしていない。

## 原本回収後の既定順序（今回未実行）

1. 不変selectorと固定C7-4関数に保存event metadataだけを渡し、cut選択を一回実施する。`l_i=1+r_i+w_i`、候補1/2/3、score=max(prefix,suffix)+1、同点は中央優先を変更しない。decision/hashをSOURCEに結び付けて保存してから、個別価格を読む。
2. 既存のStrong/中央cutについて同じevent/順序・H/endpoint・scalar/identity/χ・signed rotation/normalization・canonical回路意味論・融合・P0/phase compiler・精度/全入力postcheck・保存QPY/PASSを確認する。ファイル配置だけの差は再compile理由にしない。
3. cut2なら中央と同じ構成で追加depth改善なし。適格な旧結果を再利用できれば新build/compile/postcheck0でよい。原本なしにcut2と仮定して0改善とはしない。
4. cut1/3なら既存`prepare_steps`のcut指定で一つの新bundleを作る。中央の再利用不適格時だけ追加bundleを認め最大2。全256入力・全reference出力/χ/actual A†/vacuum ray/X/Yを既存checker/abs1e−10・observable Frobenius/signal2e−10・rtol0・phase fitなしで確認する。
5. 新P0は通常Reference選択のX/Y2本、旧Strong/中央も再利用不能なら合計最大6本。cut2重複はcompileしない。P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=False/topology-free/宣言phase保持。q64修正を無断に適用せず、17-qubit全Operatorを作らない。source/input/phase不整合ならcompile枠内でもSTOP。
6. 全7資源の選択−中央を主評価、選択−Strongを副評価とする。一pathの負結果でもrule/path/qを変更せず公開後STOP。

今回は入力内容を確認できないため、この段階へ進んでいない。SOURCEの静的一致だけでIR・データの再利用資格が成立したとはしない。新しいadapterが不可欠だと分かる前に、重複gadget/builder/simulatorやテスト基盤を作ることも避けた。

## 観測とUNKNOWN

SOURCE互換と指定pathの公開bindingは確認済み。原本所在地、実event適格性、cut、reuse資格、正確性、全資源差はUNKNOWN。今回の停止は原本可用性の問題であり、候補Bの数理や実H4有効性への否定証拠ではない。必要最小限の次作業は原本の回収であり、別path探索・候補Cへの自動移行・全RPE再評価ではない。
