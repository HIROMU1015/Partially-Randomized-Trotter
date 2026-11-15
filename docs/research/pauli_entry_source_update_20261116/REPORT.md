# A共通PR-Pauli入口：新旧SOURCEの通常intake

B実装SOURCE `830cf9bd87ec5b1d27d760b1762d60cc828abeac`。新A SOURCE `4e5b8bef634e3f071fe63df9683d7245df984d98`と旧`93a5ef3b9fed337cc4d26ee684defd304fa0eac0`の両方を、通常`accept_saved`で受け入れる。**関連21技術test PASS。今回のPauli接続確認では実signal/compile/Q3各0件。** 同時進行の別承認された決定論PF探索とはSOURCE・run directory・予算を分離している。

## 照合した変更とBの差分

A報告：`08812cd30cf46c73346695cc20b6552c8359314a`の`docs/research/pr_pauli_m5_k2_composition_20261116/REPORT.md`。

- Aのnormalizer budgetは、要求round数からsigned piece数の合計へ変更。Bではcount-only関数を照合し、13roundのm5_bestなら上限143となることを人工metadataで確認した。正規化子自体は計算していない。
- Aのround_supportは、B identityの有理数T/実stepとphysical candidateのbinary64をexact比較し、不一致を`ACTUAL_STEP_OR_T_NOT_BINARY64_EXACT`としてUNSUPPORTEDにする。
- moduleとrunnerのSOURCEが変わったため、Bのbyte一致対象を固定2版へ拡張した。出力commitごとの登録は増やさない。両fileが同一の固定版に一致する必要があり、新旧を混ぜたcodeや未登録の変更は受け入れない。
- `assembly_implementation_source`を受入全体と各signal_packetの両方へ保存。これは出力packetのsource、各roundの元数値SOURCEと別である。旧SOURCEを新SOURCEへ付け替えない。
- `bind_packet`、target、原mean/normalization/scalar、辞書helper、law、Cの価格guard/plan本体、decoder/risk kernelは不変。

## 検証

新SOURCEのpacket関数をそのまま純metadataとして読み、保存LD1/LD2/LD3/LD5から再組立てたpacketを通常intakeへ通した。configuration、13meanのraw/normalization/uncertainty/provider_binding、辞書identity、target/SOURCEは従来経路と一致した。新SOURCEのκ変更・binary64 ΔPF変更・新辞書LD4の人工packetもC.signal_inputとprovider_binding_checkまでPASS。従来の旧SOURCE側5項目の回帰検証もPASSした。

新旧版の判別、変更code/helper/candidate/law/carrier/SOURCE不一致の拒否、元packet不変、UNQUALIFIEDとnull N/Gの保全を確認。[TESTS.txt](TESTS.txt)：7項目＋既存14項目＝21 PASS。人工packet/辞書/carrierはtemporary Git内のみで、実signal catalogには登録していない。Aのprovider本体や実参照lookupは呼ばず、対応判定関数の分岐だけ人工入力で検査した。

## B2へ伝える2つの状態

厳密ΔPF=1/5、保存H4のh=binary64 0.1を使う人工scheduleでは、**m2～m12はAの新planでUNSUPPORTED / ACTUAL_STEP_OR_T_NOT_BINARY64_EXACT**。providerは実行しない。m0/m1はこの理由では未対応にならない。binary64の0.2はこの厳密1/5と別入力である。

Cの価格接続は別の段階である。同じ有理数/float表現差を持つ人工complete packetは、signal-only投影の受入検査を満たしても、Cの既存exact-step価格bridgeでは拒否され、価格接続はUNKNOWNのまま。これはAのUNSUPPORTEDをUNKNOWNへ言い換えるものではない。新Aの実行では該当roundの信号が取得できず、完備packetが作れないことも別に保持する。旧記録を新providerの対応済み証拠にはしない。

[HANDOFF.md](HANDOFF.md)と[INTAKE_LOCATOR.json](INTAKE_LOCATOR.json)に固定API・2状態のSOURCEを記載した。元packet・raw・旧STOPを変更せず、Pauliについて新たな科学実行へ進まない。ロードマップ本体の更新は統合担当へ委ねる。
